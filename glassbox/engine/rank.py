"""L4: cross-sectional selection - the top ``signal.top_k`` names by trend strength.

Runs between the signal layer and the risk layer: signals say *which symbols qualify*,
this says *which of them get the capital*, and risk says *how much*. It selects and orders;
it never creates a signal, never sizes one, and never drops one for any reason but rank.

**Determinism is the whole contract.** The live loop and a replay of the same day must
select the same names in the same order, or GB-32's replay proves nothing and two runs of
GB-49's study are not comparable. Two properties give that:

1. **Order by trend strength, descending.** The quantity the decision layer already
   computed, so ranking introduces no second opinion about what "strongest" means.
2. **Ties break alphabetically by symbol.** Python's sort is stable, so a tie would
   otherwise resolve by whatever order the caller happened to build its list in - which is
   dictionary order, which is insertion order, which is the order symbols came back from a
   broker call. That is a real source of run-to-run difference and it costs one sort key to
   remove.

Only ``enter_long`` competes for a slot. ``hold`` and ``exit`` are not candidates: an exit
is an obligation on capital already committed and must never be crowded out by a better
opportunity elsewhere.

Implemented in GB-28.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from glassbox.config.loader import Config
from glassbox.contracts.schemas import Signal
from glassbox.engine.signal import ENTER_LONG


def rank_signals(signals: Sequence[Signal], cfg: Config) -> list[Signal]:
    """The ``enter_long`` signals worth acting on, strongest first, capped at ``top_k``.

    Args:
        signals: Every verdict for one bar, in any order.
        cfg: Resolved configuration; supplies ``signal.top_k``.

    Returns:
        At most ``top_k`` signals, ordered by descending ``trend_strength`` with ties broken
        by symbol name. The list is a **subset in a new order** - no signal is modified.

    Raises:
        ValueError: a candidate carries a non-finite ``trend_strength``. Sorting NaN is
            silently order-dependent, and a comparison that quietly does nothing is exactly
            the class of bug this module exists to rule out.
    """
    candidates = [signal for signal in signals if signal.action == ENTER_LONG]
    for signal in candidates:
        if not math.isfinite(signal.trend_strength):
            raise ValueError(
                f"{signal.symbol} has a non-finite trend_strength "
                f"({signal.trend_strength!r}); ranking it would depend on sort order"
            )

    return sorted(candidates, key=_rank_key)[: cfg.signal.top_k]


def _rank_key(signal: Signal) -> tuple[float, str]:
    """Strongest first, then alphabetical.

    The negation rather than ``reverse=True``: reversing would also reverse the tie-break,
    so equal-strength symbols would come back in descending alphabetical order and the
    docstring above would be a lie.
    """
    return (-signal.trend_strength, signal.symbol)


__all__ = ["rank_signals"]
