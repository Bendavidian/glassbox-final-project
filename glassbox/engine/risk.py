"""L4: position sizing, gross exposure caps and stop-loss/take-profit attachment.

No configuration may produce an over-limit position; this is property-tested.

**Sizing may reject or shrink an order. It may never create one.** Every order that leaves
this module traces to an ``enter_long`` signal that reached it, so the risk layer can only
ever reduce what the decision layer asked for. A sizer that could invent an order would
make the decision layer's record incomplete, and GB-32's replay would not reproduce.

**Three caps, applied together, and the tightest wins.** Each is checked against the same
``equity`` so they cannot disagree about the denominator:

1. ``risk.max_position_pct`` of equity, per position.
2. ``risk.max_gross_exposure`` of equity, across every position at once - counting the
   exposure already open, which is why :func:`size_positions` accumulates as it goes rather
   than sizing each signal in isolation.
3. **Available cash**, ``equity - gross_exposure``. Not a risk rule but an arithmetic one:
   an account cannot spend what it does not hold. It is here rather than left to the
   backtester because the live executor has no such guard, and a sizer that overdraws in
   one environment and not the other is the GB-7 failure family again.

:func:`position_sizer` is the same arithmetic behind the ``PositionSizer`` protocol the
backtester declared in GB-18, so the engine and the live path size identically by
construction rather than by agreement. ``engine._require_sizeable`` therefore cannot fire
for it, and ``tests/engine/test_risk.py`` proves that by running the engine's own guard
over Hypothesis-generated inputs.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from glassbox.config.loader import Config
from glassbox.contracts.schemas import Signal
from glassbox.engine.signal import ENTER_LONG

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


@dataclass(frozen=True)
class Order:
    """One sized position, with its protective levels already attached.

    ``notional`` is ``shares * price`` after every cap and after the share conversion, so
    it is what the account actually commits rather than what was asked for. ``price`` is
    the **reference** price sizing used - the fill will differ by slippage, which is the
    backtester's business and the broker's, not the sizer's.
    """

    symbol: str
    shares: float
    price: float
    notional: float
    stop_loss: float
    take_profit: float


def size_positions(
    signals: Sequence[Signal],
    equity: float,
    prices: Mapping[str, float],
    cfg: Config,
    gross_exposure: float = 0.0,
) -> list[Order]:
    """Turn the decision layer's entries into orders the account can actually pay for.

    Args:
        signals: The verdicts, **in rank order** - sizing runs after ``rank.py`` and
            respects the order it was given, because the gross cap is spent first-come and
            the highest-ranked signal should get the room.
        equity: Total account value, cash plus positions at their marks.
        prices: ``{symbol: reference price}``. A symbol with no price gets **no order**:
            the same reading as the backtester's "a missing bar is a halt, not an exit".
        cfg: Resolved configuration; supplies every cap and both protective levels.
        gross_exposure: Marked value of positions already open. Default 0.0 for a flat
            account. It is an argument rather than a lookup because this module holds no
            state and must not learn where the book lives.

    Returns:
        Orders, in the order the signals arrived, for the entries that survived the caps.
        Only ``enter_long`` produces one; ``hold`` and ``exit`` produce none. A symbol
        appearing twice is sized once - a duplicate would pyramid a position no risk rule
        asked for.

    Raises:
        ValueError: ``equity``, ``gross_exposure`` or a price is not finite, or a price is
            not positive. A non-positive **equity** is not an error: a blown or empty
            account produces no orders, which is the correct behaviour and not an exception
            the live loop should have to catch at 16:30.
    """
    _require_finite(equity, "equity")
    _require_finite(gross_exposure, "gross_exposure")

    orders: list[Order] = []
    committed = gross_exposure
    seen: set[str] = set()

    for signal in signals:
        if signal.action != ENTER_LONG or signal.symbol in seen:
            continue
        price = prices.get(signal.symbol)
        if price is None:
            continue
        _require_price(price, signal.symbol)
        seen.add(signal.symbol)

        shares = shares_for(room_for(equity, committed, cfg), price)
        if shares <= 0.0:
            continue

        notional = shares * price
        orders.append(
            Order(
                symbol=signal.symbol,
                shares=shares,
                price=price,
                notional=notional,
                stop_loss=price * (1.0 - cfg.risk.stop_loss_pct),
                take_profit=price * (1.0 + cfg.risk.take_profit_pct),
            )
        )
        committed += notional

    return orders


def position_sizer(
    signal: Signal, equity: float, gross_exposure: float, cfg: Config
) -> float:
    """The ``PositionSizer`` the backtester and the live executor both use.

    One signal at a time, because that is the shape ``backtest.engine`` asks for, and the
    same arithmetic as :func:`size_positions` because both call :func:`room_for`. The
    engine's own book supplies ``gross_exposure``, so the gross cap is enforced across a
    backtest exactly as it is across a live account.
    """
    del signal  # sizing is a function of the account, not of which symbol it is
    _require_finite(equity, "equity")
    _require_finite(gross_exposure, "gross_exposure")
    return room_for(equity, gross_exposure, cfg)


def room_for(equity: float, gross_exposure: float, cfg: Config) -> float:
    """Notional this account may commit to one more position. Never negative.

    The single place the three caps meet. ``min`` of: the per-position cap, what is left
    under the gross cap, and the cash on hand. Public because GB-22's executor must be able
    to ask the question before it sends an order, and a second implementation of this
    arithmetic is how a backtest and a live account start describing different systems.
    """
    if not (equity > 0.0):
        return 0.0
    return max(
        0.0,
        min(
            equity * cfg.risk.max_position_pct,
            equity * cfg.risk.max_gross_exposure - gross_exposure,
            equity - gross_exposure,
        ),
    )


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _require_price(price: float, symbol: str) -> None:
    if not math.isfinite(price) or price <= 0.0:
        raise ValueError(f"{symbol}: price must be finite and positive, got {price!r}")


__all__ = [
    "MIN_SHARES",
    "Order",
    "position_sizer",
    "room_for",
    "shares_for",
    "size_positions",
]
