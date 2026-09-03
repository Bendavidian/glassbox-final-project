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
from decimal import ROUND_DOWN, Decimal

from glassbox.config.loader import Config
from glassbox.contracts.schemas import Signal
from glassbox.engine.signal import ENTER_LONG

# Alpaca's minimum order size, **verified against the live paper API in GB-22** and not
# taken from memory. It is a minimum NOTIONAL, not a minimum share count:
#
#   0.001 shares of AAPL (~$0.31)  -> {"code":40310000,"message":"cost basis must be >=
#                                      minimal amount of order 1"}
#   $0.50 notional                 -> {"code":42210000,"message":"notional amount must be
#                                      >= 1.00"}
#   $1.00 notional                 -> accepted
#   0.003 shares of a $305 stock   -> rejected ($0.92 of cost basis)
#   0.003278689 shares of the same -> accepted ($1.0004)
#
# GB-18 wrote this as `MIN_SHARES = 0.001` and flagged it for verification. The unit was
# wrong, and the difference is behavioural rather than cosmetic: a share-count floor
# rejects $0.40 of a $500 stock while permitting $0.40 of a $50 one, and permits $0.31 of
# AAPL, which the broker refuses. A notional floor is the rule the broker actually applies.
#
# A broker property, not a tuning knob — the same reasoning that keeps the RSI period out
# of the config file.
MIN_ORDER_NOTIONAL = 1.00

# Alpaca stores a quantity to nine decimal places and **silently truncates** beyond that:
# submitting 0.123456789012 shares comes back recorded as 0.123456789 (GB-22, measured).
# The conversion floors to the same precision so the backtest cannot believe it holds a
# quantity the broker would never have filled. Flooring rather than rounding, because
# rounding up can cost more cash than the sizer was shown to have.
QUANTITY_DECIMALS = 9


def shares_for(notional: float, price: float) -> float:
    """Convert a cash notional into a share count at ``price``.

    **The single place this conversion happens.** The backtester and the live executor both
    call it, so a fill in one cannot describe a different quantity from a fill in the other
    — see the module docstring and the parity test in ``tests/engine/test_executor.py``.

    **Fractional shares, not whole ones.** Alpaca supports fractional trading, the study
    sizes positions as a percentage of equity, and flooring to whole shares would
    discretise that percentage differently for a $500 stock than for a $50 one — turning
    a uniform risk rule into one that depends on price level. A backtest that flooded
    would also be systematically under-invested against the live account.

    **Below :data:`MIN_ORDER_NOTIONAL` the result is ``0.0``** and the caller does not
    trade: the broker rejects the order, so filling it in a backtest would be inventing a
    trade that could not happen. The test is on the notional, not on the share count,
    because that is the rule Alpaca applies — verified against the live API, not assumed.

    The share count is floored to :data:`QUANTITY_DECIMALS`, which is what the broker
    stores. A tighter number would make the backtest hold a quantity the live account
    could not.

    Raises:
        ValueError: ``price`` is not positive, or either argument is not finite.
    """
    if not math.isfinite(notional) or not math.isfinite(price):
        raise ValueError(
            f"notional and price must be finite, got {notional!r} and {price!r}"
        )
    if price <= 0.0:
        raise ValueError(f"price must be positive, got {price!r}")
    if notional < MIN_ORDER_NOTIONAL:
        return 0.0

    # Decimal, not `math.floor(x * 1e9) / 1e9`. The scaling trick is exact only while the
    # scaled value stays inside float64's contiguous integer range (2**53): Hypothesis
    # found a $0.01 price and a $46M account, where 4.6e9 shares scale to 4.6e18 and the
    # floor lands on a number that is not the floor. Decimal quantisation is exact at every
    # magnitude, and this runs once per order, so its cost is irrelevant.
    quantised = (Decimal(notional) / Decimal(price)).quantize(
        Decimal(1).scaleb(-QUANTITY_DECIMALS), rounding=ROUND_DOWN
    )
    shares = float(quantised)
    # The floor can drop the order a hair under the minimum on an expensive stock; a
    # quantity that buys less than the minimum is still no trade.
    return shares if shares * price >= MIN_ORDER_NOTIONAL else 0.0


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


#: Which of the three terms in :func:`room_for` was the binding minimum, and whether the
#: result was zero. **Observation vocabulary only** - nothing in the sizing path branches
#: on these, and adding one changes no number.
#:
#: ``REDUCED_BY_GROSS`` is the case that matters and the one a naive instrument misses.
#: When the gross headroom is merely *smaller* than the per-position cap, the cap has
#: bound - it shrank the entry - and the notional comes back **positive**. It looks like
#: an ordinary trade. Counting only zero notionals would report "the cap rarely binds"
#: and be wrong in the direction that flatters the risk layer.
UNCONSTRAINED = "UNCONSTRAINED"
REDUCED_BY_GROSS = "REDUCED_BY_GROSS"
BLOCKED_BY_GROSS = "BLOCKED_BY_GROSS"
BLOCKED_BY_OTHER = "BLOCKED_BY_OTHER"
TIE = "TIE"
NO_EQUITY = "NO_EQUITY"


@dataclass(frozen=True)
class RoomDetail:
    """The three caps, the notional, and which term bound - for **measurement only**.

    :func:`room_for` returns a bare float and therefore discards which of its three terms
    was the minimum. That is a real defect and it is **not fixed here**: this type is an
    observation channel beside the sizing path, not a change to it. ``room_for`` derives
    its return value from :attr:`notional` so the arithmetic exists in one place - a second
    copy of a ``min`` is how a measurement starts disagreeing with the thing it measures.

    **Ties are recorded, not resolved.** With ``max_position_pct`` 0.10 and
    ``max_gross_exposure`` 0.50, term A equals term B exactly when ``gross_exposure`` is
    ``0.40 * equity`` - four full-size positions - so a tie is the arithmetic boundary at
    which the gross cap begins to matter, not a floating-point coincidence. Every fifth
    full-size entry lands on it. A tie-break chosen inside the instrument would decide the
    headline count by itself, so :attr:`case` reports ``TIE`` and all three terms are kept
    for a policy applied afterwards and stated in the output.
    """

    equity: float
    gross_exposure: float
    per_position: float  # A: equity * max_position_pct
    gross_headroom: float  # B: equity * max_gross_exposure - gross_exposure
    cash: float  # C: equity - gross_exposure
    notional: float
    case: str

    @property
    def gross_bound(self) -> bool:
        """Did the **gross cap** bind, reduced or blocked? The headline predicate.

        ``TIE`` is excluded: whether a tie counts is the policy the caller states, and
        folding it in here would hide it.
        """
        return self.case in (REDUCED_BY_GROSS, BLOCKED_BY_GROSS)


def room_detail(equity: float, gross_exposure: float, cfg: Config) -> RoomDetail:
    """:func:`room_for`'s answer with its reasoning attached. Pure; no I/O, no state.

    The returned :attr:`RoomDetail.notional` is the value ``room_for`` returns, computed
    by the same expression rather than by a second one.
    """
    if not (equity > 0.0):
        return RoomDetail(
            equity=equity,
            gross_exposure=gross_exposure,
            per_position=math.nan,
            gross_headroom=math.nan,
            cash=math.nan,
            notional=0.0,
            case=NO_EQUITY,
        )
    per_position = equity * cfg.risk.max_position_pct
    gross_headroom = equity * cfg.risk.max_gross_exposure - gross_exposure
    cash = equity - gross_exposure
    notional = max(0.0, min(per_position, gross_headroom, cash))

    # `A == B` is the ONLY tie that can make the gross term ambiguous, and it is exact
    # rather than approximate. B == C would need `equity * max_gross_exposure == equity`,
    # i.e. a gross cap of 1.0 or a zero equity, and the zero is already returned above -
    # so with `max_gross_exposure` at 0.50 it is unreachable. A == C needs
    # `gross_exposure == 0.9 * equity`, at which B is negative and is the strict minimum
    # anyway. So the single tie worth recording is A == B, at `gross_exposure == 0.40 *
    # equity` under the configured 0.10 and 0.50, and when it holds the two are jointly
    # the minimum (C is larger, or A == B could not have been reached).
    if gross_headroom == per_position:
        case = TIE
    elif gross_headroom < per_position and gross_headroom < cash:
        case = REDUCED_BY_GROSS if notional > 0.0 else BLOCKED_BY_GROSS
    elif notional > 0.0:
        # The gross term was not binding. `A` normally is - a full-size entry is the
        # per-position cap - so this reads "unconstrained BY THE GROSS CAP", which is the
        # only question this vocabulary exists to answer.
        case = UNCONSTRAINED
    else:
        case = BLOCKED_BY_OTHER

    return RoomDetail(
        equity=equity,
        gross_exposure=gross_exposure,
        per_position=per_position,
        gross_headroom=gross_headroom,
        cash=cash,
        notional=notional,
        case=case,
    )


def room_for(equity: float, gross_exposure: float, cfg: Config) -> float:
    """Notional this account may commit to one more position. Never negative.

    The single place the three caps meet. ``min`` of: the per-position cap, what is left
    under the gross cap, and the cash on hand. Public because GB-22's executor must be able
    to ask the question before it sends an order, and a second implementation of this
    arithmetic is how a backtest and a live account start describing different systems.

    Derived from :func:`room_detail` so the ``min`` is written once. The signature and the
    returned float are unchanged, and :func:`room_detail` is not consulted by any caller on
    the sizing path - which is why adding it moves no number.
    """
    return room_detail(equity, gross_exposure, cfg).notional


def recording_sizer(records: list[RoomDetail]):
    """A :class:`~glassbox.backtest.engine.PositionSizer` that appends what it decided.

    **It observes; it does not decide differently.** The notional it returns comes from
    :func:`room_detail`, the same value :func:`position_sizer` would return for the same
    arguments, so a backtest run with this sizer produces the same trades, the same equity
    curve and the same metrics as one run without it. That claim is not left to
    inspection - ``experiments.exposure`` runs the reference condition both ways and
    compares, and ``tests/engine/test_room_detail.py`` asserts the equality directly.

    Args:
        records: The list to append to. Supplied by the caller so this holds no state of
            its own and two concurrent measurements cannot write into each other.
    """

    def sizer(
        signal: Signal, equity: float, gross_exposure: float, cfg: Config
    ) -> float:
        del signal  # sizing is a function of the account, not of which symbol it is
        _require_finite(equity, "equity")
        _require_finite(gross_exposure, "gross_exposure")
        detail = room_detail(equity, gross_exposure, cfg)
        records.append(detail)
        return detail.notional

    sizer.__name__ = "recording_position_sizer"
    return sizer


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _require_price(price: float, symbol: str) -> None:
    if not math.isfinite(price) or price <= 0.0:
        raise ValueError(f"{symbol}: price must be finite and positive, got {price!r}")


__all__ = [
    "BLOCKED_BY_GROSS",
    "BLOCKED_BY_OTHER",
    "MIN_ORDER_NOTIONAL",
    "NO_EQUITY",
    "QUANTITY_DECIMALS",
    "REDUCED_BY_GROSS",
    "TIE",
    "UNCONSTRAINED",
    "Order",
    "RoomDetail",
    "position_sizer",
    "recording_sizer",
    "room_detail",
    "room_for",
    "shares_for",
    "size_positions",
]
