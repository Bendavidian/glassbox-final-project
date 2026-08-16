"""X: walk-forward fold generation with strict boundaries.

Train 24 months, validate 3, test 3, roll 3, per ``cfg.walkforward``. Cross-validation is
invalid for time series and is never used.

**The embargo is the whole point of this module.** Adjacent splits do not overlap and
every timestamp belongs to exactly one split, and that is *not enough*. A window ending on
the last training bar has target ``y = r[t+1] .. r[t+H]``, and with ``H = 4`` those four
returns are the first four bars of validation. The model is fitted on a window whose label
is the outcome of the period it is about to be validated on. Nothing about this is
visible: the ranges are disjoint, no timestamp is shared, and a naive no-overlap assertion
passes cleanly.

So each split's last ``H`` bars are dropped as window *ends*. See :func:`embargo_bars` for
the derivation. ``tests/backtest/test_walkforward.py`` asserts the property directly and
fails when the embargo is set to zero.

Implemented in GB-17.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandas_market_calendars as mcal

from glassbox.config.loader import Config


@dataclass(frozen=True, eq=False)
class Fold:
    """One walk-forward fold.

    ``train``, ``val`` and ``test`` are the timestamps a window may **end** on, already
    embargoed. Slice the feature frame by these and every window is safe by construction;
    there is no second, unembargoed view of a fold to reach for by mistake.

    A window ending at the first timestamp of ``train`` reads ``input_len`` bars of history
    behind it, which lie *before* the fold. That is correct and deliberate — it is exactly
    what the live loop does — and it does not leak, because those bars are in the past. The
    fold grid is anchored so that this history always exists.

    ``eq=False``: comparing two folds would compare DatetimeIndexes elementwise and raise
    on the ambiguous truth value. Compare the fields you mean.
    """

    number: int  # 1-based, in chronological order
    train: pd.DatetimeIndex
    val: pd.DatetimeIndex
    test: pd.DatetimeIndex
    embargo: int  # bars dropped from the end of each split


def embargo_bars(cfg: Config) -> int:
    """Bars to drop from the end of each split, derived rather than remembered.

    ``build_windows`` defines the target of a window ending at position ``p`` as
    ``y[h] = target[p + 1 + h]`` for ``h`` in ``0 .. H-1``, so the target occupies
    positions ``p+1 .. p+H``. If a split ends at position ``b``, keeping that target inside
    the split requires ``p + H <= b``, so the last usable end position is ``b - H`` and the
    dropped ends are ``b-H+1 .. b`` — exactly ``H`` of them.

    **It is H, not H-1.** The tempting off-by-one comes from thinking of the boundary bar
    as the one to remove. Work it through at ``H = 4`` with a split ending at position 9:
    ``p=9`` leaks 4 bars, ``p=8`` leaks 3, ``p=7`` leaks 2, ``p=6`` leaks 1, and ``p=5`` is
    the first clean end. ``5 == 9 - 4``. Four ends dropped.

    Nor is it "H plus something". No extra margin is needed for the *input* side: a
    validation window reads ``input_len`` bars of history that reach back into the training
    range, and that is not leakage — it is the model consuming past data exactly as it will
    live. Padding the start of a split would discard usable data to prevent a problem that
    does not exist.
    """
    return cfg.window.horizon


def make_folds(index: pd.DatetimeIndex, cfg: Config) -> list[Fold]:
    """Generate walk-forward folds over ``index``.

    Args:
        index: The trading days to split — normally a feature frame's index, so that
            warm-up rows are already gone. Must be sorted, unique and tz-aware.
        cfg: Resolved configuration; supplies ``walkforward`` and ``window``.

    Returns:
        Folds in chronological order, at most ``cfg.walkforward.max_folds``. When more
        folds fit than the cap allows, the **most recent** are kept — see
        ``DECISIONS.md``. A series too short for even one complete fold yields an empty
        list rather than raising: "no fold fits" is an answer, and the caller reports it.

    Raises:
        ValueError: The index is unsorted, has duplicates, or is not tz-aware.
    """
    _require_usable_index(index)
    embargo = embargo_bars(cfg)
    plan = cfg.walkforward

    # A window ending at the first timestamp of the first fold must have `input_len` bars
    # of history behind it, so the grid starts there rather than at index[0]. Anchoring at
    # index[0] would hand back a train range whose earliest timestamps cannot produce a
    # window at all, and the caller would silently get fewer windows than the fold claims.
    if len(index) <= cfg.window.input_len:
        return []
    anchor = index[cfg.window.input_len - 1]

    folds: list[Fold] = []
    step = 0
    while True:
        train_start = anchor + pd.DateOffset(months=plan.step_months * step)
        val_start = train_start + pd.DateOffset(months=plan.train_months)
        test_start = val_start + pd.DateOffset(months=plan.val_months)
        test_end = test_start + pd.DateOffset(months=plan.test_months)

        if test_end > index[-1]:
            # The last fold would report on a truncated test range. A fold covering two
            # months of test alongside folds covering three is a comparability problem the
            # study would inherit, so it is dropped rather than shortened.
            break

        splits = [
            _embargoed(index, start, end, embargo)
            for start, end in (
                (train_start, val_start),
                (val_start, test_start),
                (test_start, test_end),
            )
        ]
        if all(len(split) > 0 for split in splits):
            train, val, test = splits
            folds.append(
                Fold(
                    number=len(folds) + 1,
                    train=train,
                    val=val,
                    test=test,
                    embargo=embargo,
                )
            )
        step += 1

    kept = folds[-plan.max_folds :] if len(folds) > plan.max_folds else folds
    return [
        Fold(
            number=position,
            train=fold.train,
            val=fold.val,
            test=fold.test,
            embargo=fold.embargo,
        )
        for position, fold in enumerate(kept, start=1)
    ]


def target_range(
    index: pd.DatetimeIndex, end: pd.Timestamp, cfg: Config
) -> pd.DatetimeIndex:
    """The bars a window ending at ``end`` takes its target from: ``end+1 .. end+H``.

    Exposed because it is the definition the embargo is derived from, and a test that
    restates that definition itself would pass whenever both copies were wrong together.
    """
    position = int(index.get_loc(end))
    return index[position + 1 : position + 1 + cfg.window.horizon]


def trading_days(
    cfg: Config, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DatetimeIndex:
    """Sessions the exchange calendar declares open in ``[start, end]``, as UTC midnights.

    The bar index this module splits already carries only trading days, so this is used to
    *check* boundaries rather than to place them — see the calendar test. Placing folds by
    selecting from the real index means a boundary cannot land on a holiday in the first
    place.
    """
    sessions = mcal.get_calendar(cfg.data.calendar).valid_days(
        start_date=start.date(), end_date=end.date()
    )
    # mcal returns tz-aware UTC; normalise to midnight so these compare equal to the bar
    # index, which data.historical pins to UTC midnight.
    return pd.DatetimeIndex(sessions).normalize()


def _embargoed(
    index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp, embargo: int
) -> pd.DatetimeIndex:
    """The bars in ``[start, end)``, less the final ``embargo`` of them."""
    window = index[(index >= start) & (index < end)]
    if embargo <= 0:
        return window
    return window[:-embargo]


def _require_usable_index(index: pd.DatetimeIndex) -> None:
    if index.tz is None:
        raise ValueError("the index must be tz-aware; the project works in UTC")
    if not index.is_monotonic_increasing:
        raise ValueError("the index must be sorted ascending")
    if not index.is_unique:
        raise ValueError("the index must not contain duplicate timestamps")


__all__ = ["Fold", "embargo_bars", "make_folds", "target_range", "trading_days"]
