"""The sweep harness: measure across the universe, report the fraction, never a boolean.

Companion to ``causality.py``, and built for the same reason — a correctness argument that
was being made by hand, made once, properly, and reused.

**Why this exists.** GB-27 found that the parity floor of 352 bars, recorded in the spec as
*verified*, held in **32 of 125** symbol-timestamp pairs. It had been measured on one symbol
at one timestamp, and that point was one of the 32. The audit that followed found the
pattern: every claim in this project that survived scrutiny came from a task that happened
to have a harness to sweep with, and every claim that did not came from a task that measured
by hand on whatever was in front of it. The fix is not more care. It is this file.

**Two properties, both learned from that failure.**

1. **A sweep reports a fraction, never a boolean.** ``32 of 125`` is the number that would
   have caught the floor; ``it passes`` is the number that did not. :class:`SweepResult`
   therefore **refuses to be used as a truth value** - ``if result:`` raises ``TypeError``
   and names the attributes to read instead. A caller who wants a boolean has to say which
   fraction satisfies them, which is the decision that was previously being made by accident.
2. **A sweep of one cell is refused.** Not discouraged - refused, because a sweep of one is
   the bug being fixed here, and a harness that permits it is a harness that will be used
   that way at 02:00 on a deadline.

Two shapes cover every claim this project makes:

- :func:`sweep` for *"X holds"* - the parity floor, a causality property, an invariant.
- :func:`contest` for *"A beats B"* - an initialisation study, a model comparison, any
  ranking. It reports per-cell winners and the aggregate, so "A is better" can never again
  rest on the one cell somebody happened to run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

# A sweep of one cell is the failure this harness exists to prevent. Two is the smallest
# number that can disagree with itself; the sweep that caught the parity floor used 125.
MIN_CELLS = 2


class SweepTooSmall(ValueError):
    """Raised when a sweep would run on fewer than :data:`MIN_CELLS` cells."""


@dataclass(frozen=True)
class Cell:
    """One measurement, and where it was taken."""

    symbol: str
    key: Any  # a timestamp, a fold number, whatever indexes the second axis
    held: bool
    detail: Any = None

    def __str__(self) -> str:
        mark = "hold" if self.held else "FAIL"
        return f"{self.symbol}@{self.key}: {mark}" + (
            f" ({self.detail})" if self.detail is not None else ""
        )


@dataclass(frozen=True)
class SweepResult:
    """Every cell, and the fraction that held.

    Deliberately **not** a boolean. ``bool(result)`` raises: a sweep whose answer collapses
    to yes-or-no has thrown away the number that matters, and this project has already paid
    for that once.
    """

    cells: tuple[Cell, ...]

    def __bool__(self) -> bool:
        raise TypeError(
            "a SweepResult is a fraction, not a verdict - the whole point of this harness. "
            f"You have {self.held} of {self.total} cells holding "
            f"({self.fraction:.1%}). Read .fraction, .all_held, .held or .failures, and "
            "say which fraction satisfies you."
        )

    @property
    def total(self) -> int:
        return len(self.cells)

    @property
    def held(self) -> int:
        return sum(1 for cell in self.cells if cell.held)

    @property
    def fraction(self) -> float:
        return self.held / self.total if self.total else 0.0

    @property
    def all_held(self) -> bool:
        """The strong claim, stated explicitly rather than implied by truthiness."""
        return self.held == self.total

    @property
    def failures(self) -> tuple[Cell, ...]:
        return tuple(cell for cell in self.cells if not cell.held)

    def failures_by_symbol(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cell in self.failures:
            counts[cell.symbol] = counts.get(cell.symbol, 0) + 1
        return dict(sorted(counts.items()))

    def summary(self) -> str:
        detail = self.failures_by_symbol()
        failing = ", ".join(f"{s}:{n}" for s, n in detail.items()) or "none"
        return (
            f"{self.held} of {self.total} cells hold ({self.fraction:.1%}); "
            f"failing: {failing}"
        )


def sweep(
    measure: Callable[[str, Any], bool | tuple[bool, Any]],
    symbols: Sequence[str],
    keys: Sequence[Any],
) -> SweepResult:
    """Run ``measure`` over every ``(symbol, key)`` and report how many cells held.

    Args:
        measure: ``measure(symbol, key)`` returning ``held`` or ``(held, detail)``. The
            detail is carried into the cell so a failure can be diagnosed without a rerun.
        symbols: The universe axis. One symbol is allowed only if there are enough keys -
            what is refused is a sweep too small to disagree with itself.
        keys: Timestamps, fold numbers, or anything else indexing the second axis.

    Returns:
        A :class:`SweepResult`. Not a boolean; see the class.

    Raises:
        SweepTooSmall: fewer than :data:`MIN_CELLS` cells would be measured.
    """
    _require_enough(symbols, keys)

    cells: list[Cell] = []
    for symbol in symbols:
        for key in keys:
            outcome = measure(symbol, key)
            held, detail = outcome if isinstance(outcome, tuple) else (outcome, None)
            cells.append(Cell(symbol=symbol, key=key, held=bool(held), detail=detail))
    return SweepResult(cells=tuple(cells))


@dataclass(frozen=True)
class Contest:
    """Which arm won, per cell and in aggregate. For every *"A beats B"* claim.

    ``wins`` counts cells, not magnitude: an arm that wins narrowly in nine cells and loses
    catastrophically in one is a different animal from one that wins on average, and a
    ranking that only reports the mean cannot tell them apart.
    """

    arms: tuple[str, ...]
    scores: tuple[tuple[str, Any, dict[str, float]], ...]  # (symbol, key, {arm: score})
    higher_is_better: bool

    @property
    def total(self) -> int:
        return len(self.scores)

    def winner_of(self, cell: dict[str, float]) -> str:
        pick = max if self.higher_is_better else min
        return pick(cell, key=lambda arm: cell[arm])

    def wins(self) -> dict[str, int]:
        counts = dict.fromkeys(self.arms, 0)
        for _, _, cell in self.scores:
            counts[self.winner_of(cell)] += 1
        return counts

    def means(self) -> dict[str, float]:
        return {
            arm: sum(cell[arm] for _, _, cell in self.scores) / self.total
            for arm in self.arms
        }

    def beats(self, arm: str, other: str) -> int:
        """Cells in which ``arm`` scores better than ``other``. The pairwise number."""
        better = (lambda a, b: a > b) if self.higher_is_better else (lambda a, b: a < b)
        return sum(1 for _, _, cell in self.scores if better(cell[arm], cell[other]))

    def summary(self) -> str:
        wins = ", ".join(f"{arm}:{n}/{self.total}" for arm, n in self.wins().items())
        means = ", ".join(f"{arm}={value:.4f}" for arm, value in self.means().items())
        return f"wins by cell: {wins}\nmeans: {means}"


def contest(
    measure: Callable[[str, Any, str], float],
    symbols: Sequence[str],
    keys: Sequence[Any],
    arms: Sequence[str],
    higher_is_better: bool = True,
) -> Contest:
    """Score every arm in every cell, so a ranking rests on cells rather than on one run.

    Raises:
        SweepTooSmall: fewer than :data:`MIN_CELLS` cells, or fewer than two arms - a
            contest of one arm is not a contest.
    """
    _require_enough(symbols, keys)
    if len(arms) < 2:
        raise SweepTooSmall(
            f"a contest needs at least two arms to compare, got {list(arms)}"
        )

    scores = tuple(
        (symbol, key, {arm: float(measure(symbol, key, arm)) for arm in arms})
        for symbol in symbols
        for key in keys
    )
    return Contest(arms=tuple(arms), scores=scores, higher_is_better=higher_is_better)


def _require_enough(symbols: Sequence[str], keys: Sequence[Any]) -> None:
    cells = len(symbols) * len(keys)
    if cells < MIN_CELLS:
        raise SweepTooSmall(
            f"a sweep of {cells} cell(s) is not a sweep - {len(symbols)} symbol(s) x "
            f"{len(keys)} key(s). This harness exists because a claim measured at one "
            "point entered the spec as verified and held in 32 of 125 cells when it was "
            f"finally swept. Supply at least {MIN_CELLS} cells."
        )


__all__ = [
    "MIN_CELLS",
    "Cell",
    "Contest",
    "SweepResult",
    "SweepTooSmall",
    "contest",
    "sweep",
]
