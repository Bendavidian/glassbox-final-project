"""X: the event-driven backtester - fees, slippage and stop-loss/take-profit.

Bars are iterated one at a time, in order. There is no vectorised shortcut anywhere in
this module, and that is a correctness requirement rather than a style preference: the
reference project's look-ahead lives precisely in its vectorised mark-to-market, where a
bar's close reaches a fill priced at that bar's open. A loop that only ever reads the bar
it is standing on cannot do that.

Four pricing rules decide the numbers this project reports. Each is stated here, tested
individually, and recorded in DECISIONS.md with the measurement that chose it.

1. **A signal fills at the NEXT bar's open.** A signal computed from bar t's close cannot
   fill at that close. The decision is made after the session ends, the order is
   market-on-open, and it fills at t+1. This is also what GB-26's live loop can do, so
   backtest and live agree by construction.
2. **On a bar that breaches both the stop and the target, the stop fills.** Daily OHLC
   cannot say which came first, so it is an assumption either way, and the assumption
   should be the one that cannot be accused of flattering. Measured: it decides 0.08% of
   resolutions, so the conservative choice is close to free.
3. **Slippage is adverse on both sides.** Buys pay ``price * (1 + s)``, sells receive
   ``price * (1 - s)``. Round trip on a flat trade is exactly ``2 * (fee_bps +
   slippage_bps)`` of the reference notional - 6.0 bps at the configured rates.
4. **A gap through a stop fills at the open, not the stop.** If the bar opens beyond the
   level, that level was never available. Measured on the real universe: 15.6% of stop
   exits gap, and pretending otherwise would understate each by a mean of 117 bps - about
   19x the entire friction model. The trade log records ``stop_gap`` separately from
   ``stop`` so the report can say how much of the drawdown came from gaps.

Stop and target comparisons are written correctly and each case is tested; see
reference/REFERENCE_AUDIT.md for the inverted comparisons in the reference project.

Long-only, per spec 2.1.

Implemented in GB-18.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from glassbox.config.loader import Config
from glassbox.contracts.schemas import Signal

BPS = 10_000.0

ENTER_LONG = "enter_long"
EXIT = "exit"

# Exit reasons. The `_gap` variants exist so the report can separate "the stop rule cost
# this" from "the market gapped and the stop was never available".
STOP = "stop"
STOP_GAP = "stop_gap"
TARGET = "target"
TARGET_GAP = "target_gap"
SIGNAL = "signal"
END_OF_DATA = "end_of_data"


class PositionSizer(Protocol):
    """How much to deploy for a signal. Implemented by ``engine/risk.py`` in GB-21.

    Returns a **notional**, not a share count, so every price conversion stays inside the
    engine and a sizer cannot accidentally define its own fill price.
    """

    def __call__(
        self, signal: Signal, equity: float, gross_exposure: float, cfg: Config
    ) -> float:
        """Target notional for this signal. ``0.0`` means do not trade."""
        ...


@dataclass(frozen=True)
class Trade:
    """One completed round trip.

    ``entry_price`` and ``exit_price`` are **reference** prices - the levels the rules
    chose, before slippage. Slippage lives in ``costs`` rather than being folded into the
    prices, so the log answers "what did the rule pick?" and "what did the frictions take?"
    separately. ``net_pnl == gross_pnl - costs`` exactly, and is asserted for every trade.
    """

    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    size: float  # shares
    entry_price: float
    exit_price: float
    gross_pnl: float
    costs: float
    net_pnl: float
    exit_reason: str


@dataclass(frozen=True, eq=False)
class BacktestResult:
    """Equity curve and trade log.

    ``eq=False``: comparing two results would compare Series elementwise and raise on the
    ambiguous truth value. Compare the fields you mean.
    """

    equity: pd.Series  # one point per bar, marked at the close, never NaN
    trades: tuple[Trade, ...]

    def trade_log(self) -> pd.DataFrame:
        """The trade log as a frame, in exit order. Empty but correctly shaped if no trade."""
        columns = [
            "symbol",
            "entry_time",
            "exit_time",
            "size",
            "entry_price",
            "exit_price",
            "gross_pnl",
            "costs",
            "net_pnl",
            "exit_reason",
        ]
        rows = [
            {column: getattr(trade, column) for column in columns}
            for trade in self.trades
        ]
        return pd.DataFrame(rows, columns=columns)


@dataclass
class _OpenPosition:
    """Engine-internal state. Not part of any contract."""

    symbol: str
    entry_time: pd.Timestamp
    shares: float
    entry_reference: float  # the bar open the rules chose, before slippage
    entry_fee: float
    stop: float
    target: float


def run_backtest(
    bars: Mapping[str, pd.DataFrame],
    signals: Mapping[pd.Timestamp, Sequence[Signal]],
    sizer: PositionSizer,
    cfg: Config,
) -> BacktestResult:
    """Run the event loop over ``bars``, acting on ``signals``.

    Args:
        bars: ``{symbol: canonical bar frame}``. Each carries ``open, high, low, close``
            on a sorted, tz-aware DatetimeIndex.
        signals: ``{timestamp: signals decided at that bar's close}``. They are acted on at
            the **next** bar's open, which is the only thing a daily system can do.
        sizer: Position sizing, injected. GB-21 supplies the real one; the engine holds no
            risk logic of its own.
        cfg: Resolved configuration - starting cash, frictions, stop and target fractions.

    Returns:
        A :class:`BacktestResult` whose equity curve has one point per bar in the union of
        the input indexes.

    Raises:
        ValueError: ``bars`` is empty, a frame lacks a required column or a sorted index,
            or the sizer returns something the engine will not act on.
    """
    index = _master_index(bars)
    slippage = cfg.backtest.slippage_bps / BPS
    fee = cfg.backtest.fee_bps / BPS

    cash = float(cfg.backtest.initial_cash)
    positions: dict[str, _OpenPosition] = {}
    trades: list[Trade] = []
    curve: list[float] = []

    pending: Sequence[Signal] = ()
    for position_in_index, timestamp in enumerate(index):
        is_final_bar = position_in_index == len(index) - 1

        # 1. Orders decided at the previous close, filled at this open. Exits first: the
        #    cash they release is available to this bar's entries, which is what a broker
        #    does and what the live loop will experience.
        for signal in _actions(pending, EXIT):
            held = positions.pop(signal.symbol, None)
            if held is not None:
                price = _price(bars, signal.symbol, timestamp, "open")
                if price is not None:
                    cash += _close_out(held, price, SIGNAL, timestamp, trades, cfg)

        for signal in _actions(pending, ENTER_LONG):
            if signal.symbol in positions:
                # Already long. Adding here would silently pyramid, which no risk rule
                # asked for and which the reference project does by accident.
                continue
            price = _price(bars, signal.symbol, timestamp, "open")
            if price is None:
                continue
            equity = _equity(cash, positions, bars, timestamp)
            exposure = _gross_exposure(positions, bars, timestamp)
            notional = _require_sizeable(
                sizer(signal, equity, exposure, cfg), cash, sizer, signal
            )
            if notional <= 0.0:
                continue

            fill = price * (1.0 + slippage)
            shares = notional / fill
            entry_fee = shares * fill * fee
            if shares * fill + entry_fee > cash:
                # The fee pushes the order past the cash the sizer was shown. Shrink to
                # fit rather than overdraw: an account cannot go negative.
                shares = cash / (fill * (1.0 + fee))
                entry_fee = shares * fill * fee
            cash -= shares * fill + entry_fee
            positions[signal.symbol] = _OpenPosition(
                symbol=signal.symbol,
                entry_time=timestamp,
                shares=shares,
                entry_reference=price,
                entry_fee=entry_fee,
                stop=price * (1.0 - cfg.risk.stop_loss_pct),
                target=price * (1.0 + cfg.risk.take_profit_pct),
            )

        # 2. Stops and targets, against this bar's range. A position opened at this bar's
        #    open is exposed to the rest of it, so this runs after entries.
        for symbol in list(positions):
            held = positions[symbol]
            bar = _bar(bars, symbol, timestamp)
            if bar is None:
                continue
            exit_price, reason = _exit_level(held, bar)
            if reason is None:
                continue
            del positions[symbol]
            cash += _close_out(held, exit_price, reason, timestamp, trades, cfg)

        # 3. Nothing is left open past the data. Liquidating at the final close keeps the
        #    curve honest; the reason lets GB-19 exclude these if it wants to.
        if is_final_bar:
            for symbol in list(positions):
                held = positions.pop(symbol)
                price = _price(bars, symbol, timestamp, "close")
                if price is not None:
                    cash += _close_out(held, price, END_OF_DATA, timestamp, trades, cfg)

        curve.append(_equity(cash, positions, bars, timestamp))
        pending = signals.get(timestamp, ())

    return BacktestResult(
        equity=pd.Series(curve, index=index, name="equity", dtype="float64"),
        trades=tuple(trades),
    )


def round_trip_cost_bps(cfg: Config) -> float:
    """Total friction on a round trip, in basis points of the reference notional.

    Both legs pay slippage and both pay a fee, so the frictions simply add. Exposed
    because it is the arithmetic the engine implements, and a test asserts the engine
    agrees with it on an actual flat trade rather than only on paper.
    """
    return 2.0 * (cfg.backtest.fee_bps + cfg.backtest.slippage_bps)


def _exit_level(held: _OpenPosition, bar: pd.Series) -> tuple[float, str | None]:
    """The price and reason this position exits on this bar, or ``(nan, None)``.

    Long-only, so there are two rules and both are written in the direction the reference
    project inverted: the stop triggers when the **low** falls to it, the target when the
    **high** rises to it.

    The stop is tested first. When one bar breaches both, daily OHLC cannot say which came
    first, and the conservative reading is the one that survives review.
    """
    if bar["low"] <= held.stop:
        if bar["open"] <= held.stop:
            # It opened through the level. Filling at the stop would grant a price that
            # was never quoted.
            return float(bar["open"]), STOP_GAP
        return held.stop, STOP

    if bar["high"] >= held.target:
        if bar["open"] >= held.target:
            return float(bar["open"]), TARGET_GAP
        return held.target, TARGET

    return math.nan, None


def _close_out(
    held: _OpenPosition,
    reference: float,
    reason: str,
    timestamp: pd.Timestamp,
    trades: list[Trade],
    cfg: Config,
) -> float:
    """Record the trade and return the cash the exit releases."""
    slippage = cfg.backtest.slippage_bps / BPS
    fee = cfg.backtest.fee_bps / BPS

    fill = reference * (1.0 - slippage)
    exit_fee = held.shares * fill * fee
    proceeds = held.shares * fill - exit_fee

    gross = held.shares * (reference - held.entry_reference)
    costs = (
        held.shares * held.entry_reference * slippage
        + held.shares * reference * slippage
        + held.entry_fee
        + exit_fee
    )
    trades.append(
        Trade(
            symbol=held.symbol,
            entry_time=held.entry_time,
            exit_time=timestamp,
            size=held.shares,
            entry_price=held.entry_reference,
            exit_price=reference,
            gross_pnl=gross,
            costs=costs,
            net_pnl=gross - costs,
            exit_reason=reason,
        )
    )
    return proceeds


def _require_sizeable(
    notional: float, cash: float, sizer: PositionSizer, signal: Signal
) -> float:
    """Refuse a sizing the engine will not act on.

    GB-21's property test promises no configuration produces an over-limit position. This
    is the engine holding it to that promise rather than trusting it.
    """
    name = getattr(sizer, "__name__", type(sizer).__name__)
    if not math.isfinite(notional):
        raise ValueError(f"{name} returned {notional!r} for {signal.symbol}")
    if notional < 0.0:
        raise ValueError(
            f"{name} returned a negative notional {notional!r} for {signal.symbol}; "
            "this system is long-only"
        )
    if notional > cash:
        raise ValueError(
            f"{name} asked for {notional:.2f} of notional for {signal.symbol} with "
            f"{cash:.2f} of cash available; a backtest may not trade on margin"
        )
    return notional


def _actions(signals: Sequence[Signal], action: str) -> list[Signal]:
    return [signal for signal in signals if signal.action == action]


def _bar(
    bars: Mapping[str, pd.DataFrame], symbol: str, timestamp: pd.Timestamp
) -> pd.Series | None:
    """This symbol's bar, or ``None`` when it did not trade on this day."""
    frame = bars.get(symbol)
    if frame is None or timestamp not in frame.index:
        return None
    return frame.loc[timestamp]


def _price(
    bars: Mapping[str, pd.DataFrame],
    symbol: str,
    timestamp: pd.Timestamp,
    column: str,
) -> float | None:
    bar = _bar(bars, symbol, timestamp)
    return None if bar is None else float(bar[column])


def _equity(
    cash: float,
    positions: Mapping[str, _OpenPosition],
    bars: Mapping[str, pd.DataFrame],
    timestamp: pd.Timestamp,
) -> float:
    """Cash plus positions marked at this bar's close."""
    return cash + _gross_exposure(positions, bars, timestamp)


def _gross_exposure(
    positions: Mapping[str, _OpenPosition],
    bars: Mapping[str, pd.DataFrame],
    timestamp: pd.Timestamp,
) -> float:
    total = 0.0
    for held in positions.values():
        price = _price(bars, held.symbol, timestamp, "close")
        total += held.shares * (price if price is not None else held.entry_reference)
    return total


REQUIRED_COLUMNS = ("open", "high", "low", "close")


def _master_index(bars: Mapping[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Every timestamp any symbol traded on, sorted, deduplicated."""
    if not bars:
        raise ValueError("no bars to backtest")

    combined: pd.DatetimeIndex | None = None
    for symbol, frame in bars.items():
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{symbol} is missing the {missing} column(s)")
        if not frame.index.is_monotonic_increasing:
            raise ValueError(f"{symbol} bars must be sorted ascending")
        combined = frame.index if combined is None else combined.union(frame.index)

    assert combined is not None
    return pd.DatetimeIndex(combined).sort_values()


__all__ = [
    "END_OF_DATA",
    "ENTER_LONG",
    "EXIT",
    "SIGNAL",
    "STOP",
    "STOP_GAP",
    "TARGET",
    "TARGET_GAP",
    "BacktestResult",
    "PositionSizer",
    "Trade",
    "round_trip_cost_bps",
    "run_backtest",
]
