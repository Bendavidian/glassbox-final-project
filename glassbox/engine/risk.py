"""L4: position sizing, gross exposure caps and stop-loss/take-profit attachment.

No configuration may produce an over-limit position; this is property-tested.

:func:`shares_for` lands early, in GB-18, because it is the **single** place a notional
becomes a share count. The backtester (`backtest/engine.py`) and the live executor
(`engine/executor.py`, GB-22) must both call it. Two copies of that arithmetic is the
GB-7 failure family again: if the backtest buys 99.98 shares and the executor floors to
99, the two systems describe different things, every test still passes, and the
divergence shows up only as an unexplained gap between backtest and paper results.

Implemented in GB-21; :func:`shares_for` in GB-18.
"""

from __future__ import annotations

import math

# Alpaca's minimum fractional order quantity. A broker property, not a tuning knob —
# the same reasoning that keeps the RSI period out of the config file. An order below it
# would be rejected by the broker, so the backtest must not pretend it filled.
MIN_SHARES = 0.001


def shares_for(notional: float, price: float) -> float:
    """Convert a cash notional into a share count at ``price``.

    **Fractional shares, not whole ones.** Alpaca supports fractional trading, the study
    sizes positions as a percentage of equity, and flooring to whole shares would
    discretise that percentage differently for a $500 stock than for a $50 one — turning
    a uniform risk rule into one that depends on price level. A backtest that flooded
    would also be systematically under-invested against the live account.

    When the notional buys less than :data:`MIN_SHARES`, the result is ``0.0`` and the
    caller does not trade: the broker would reject the order, so filling it in a backtest
    would be inventing a trade that could not happen.

    Raises:
        ValueError: ``price`` is not positive, or either argument is not finite.
    """
    if not math.isfinite(notional) or not math.isfinite(price):
        raise ValueError(
            f"notional and price must be finite, got {notional!r} and {price!r}"
        )
    if price <= 0.0:
        raise ValueError(f"price must be positive, got {price!r}")
    if notional <= 0.0:
        return 0.0

    shares = notional / price
    return shares if shares >= MIN_SHARES else 0.0


__all__ = ["MIN_SHARES", "shares_for"]
