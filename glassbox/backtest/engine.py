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

5. **A missing bar is a halt, not an exit.** A symbol with no bar on a day cannot be
   traded on that day, so an exit order stays live and fills at its next traded open, and
   the position is marked at its own last printed close rather than at the price it was
   bought at. A symbol whose history ends before the universe's does is liquidated at its
   own last close. The alternative - dropping the position when no price is available -
   deletes the holding from the book while leaving its value out of cash, and it is not an
   edge case: it fires whenever one symbol has a shorter history than the rest.

Stop and target comparisons are written correctly and each case is tested; see
reference/REFERENCE_AUDIT.md for the inverted comparisons in the reference project.

``_assert_accounted`` holds the whole module to one identity - with the book flat, final
equity equals initial cash plus the sum of the trade log - and it runs on every backtest,
not only in tests.

**Every number this module produces is a TOTAL return, not a price return.**
``data/historical.py`` fetches with ``auto_adjust=True``, so dividends and splits are
folded into the price series: a dividend appears as a smaller downward step on the
ex-date rather than as separate cash. Nothing here adds dividend income, because it is
already in the prices. A reader comparing these figures against a price-only benchmark
would find a discrepancy of roughly the universe's dividend yield per year and have no
way to explain it, so GB-57 must state the convention in the results chapter.

Notional becomes a share count through ``engine.risk.shares_for`` - the one place that
conversion happens, shared with GB-22's live executor so the two cannot diverge.

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
from glassbox.engine.risk import shares_for

# The verdict vocabulary belongs to the layer that produces verdicts, and is imported
# rather than respelled here. GB-18 wrote its own copies; GB-20 removed them, because two
# spellings of "enter_long" in two modules is the GB-7 failure family - each side keeps its
# tests and the system quietly stops trading. Re-exported so `engine.ENTER_LONG` still
# resolves for callers that reach for it through the backtester.
from glassbox.engine.signal import ENTER_LONG, EXIT

BPS = 10_000.0

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

    ``strategy_exit`` is ``False`` only for ``end_of_data``: the position was liquidated
    because the data ran out, not because the strategy decided anything. **GB-19's rule,
    pinned here rather than left to be invented later:** such trades ARE included in the
    equity curve and total return, because the curve must be complete and the capital was
    genuinely returned - but they are EXCLUDED from hit rate, average trade and any other
    per-decision statistic, because no decision was made. Filter on this field, never on
    an ``exit_reason`` string.
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
    strategy_exit: bool  # False when the exit was administrative, not a decision


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
            "strategy_exit",
        ]
        rows = [
            {column: getattr(trade, column) for column in columns}
            for trade in self.trades
        ]
        return pd.DataFrame(rows, columns=columns)


@dataclass
class _OpenPosition:
    """Engine-internal state. Not part of any contract.

    ``last_mark`` is the most recent close this symbol actually printed, and
    ``last_mark_time`` is when. A symbol that stops printing bars - a halt, a delisting, or
    simply a shorter history than the rest of the universe - is still worth something, and
    it is worth its last traded close rather than the price it was bought at. Carrying the
    mark also removes a look-ahead: sizing an order filled at bar t's open must not consult
    bar t's close, so the sizer is shown exposure marked at the previous close.
    """

    symbol: str
    entry_time: pd.Timestamp
    shares: float
    entry_reference: float  # the bar open the rules chose, before slippage
    entry_fee: float
    stop: float
    target: float
    last_mark: float
    last_mark_time: pd.Timestamp


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
        unfilled: list[Signal] = []
        for signal in _actions(pending, EXIT):
            held = positions.get(signal.symbol)
            if held is None:
                continue
            price = _price(bars, signal.symbol, timestamp, "open")
            if price is None:
                # No bar for this symbol today. **A missing bar is a halt, not an exit.**
                # The order stays live and fills at the symbol's next traded open. Closing
                # the position here - or, worse, dropping it - would delete the holding
                # from the book while leaving its value out of cash.
                unfilled.append(signal)
                continue
            del positions[signal.symbol]
            cash += _close_out(held, price, SIGNAL, timestamp, trades, cfg)

        for signal in _actions(pending, ENTER_LONG):
            if signal.symbol in positions:
                # Already long. Adding here would silently pyramid, which no risk rule
                # asked for and which the reference project does by accident.
                continue
            price = _price(bars, signal.symbol, timestamp, "open")
            if price is None:
                # An unfilled ENTRY expires; an unfilled EXIT above is carried forward.
                # The asymmetry is deliberate. An entry is a bet on a forecast made from a
                # window ending at a specific bar, and by the time the symbol trades again
                # that forecast is stale - a live loop would recompute it, not resurrect
                # the order. An exit is an obligation on capital already committed, and
                # abandoning it would leave the position open with nothing to close it.
                continue
            equity = _equity(cash, positions)
            exposure = _gross_exposure(positions)
            notional = _require_sizeable(
                sizer(signal, equity, exposure, cfg), cash, sizer, signal
            )
            if notional <= 0.0:
                continue

            fill = price * (1.0 + slippage)
            shares = shares_for(notional, fill)
            if shares <= 0.0:
                # Below the broker's minimum fractional quantity. The order would be
                # rejected, so filling it here would invent a trade that cannot happen.
                continue
            entry_fee = shares * fill * fee
            if shares * fill + entry_fee > cash:
                # The fee pushes the order past the cash the sizer was shown. Shrink to
                # fit rather than overdraw: an account cannot go negative.
                shares = shares_for(cash / (1.0 + fee), fill)
                if shares <= 0.0:
                    continue
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
                last_mark=price,
                last_mark_time=timestamp,
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

        # 3. Mark every survivor at this bar's close, where there is one. A symbol that did
        #    not trade keeps the mark it had, which is its own last printed close.
        for held in positions.values():
            close = _price(bars, held.symbol, timestamp, "close")
            if close is not None:
                held.last_mark = close
                held.last_mark_time = timestamp

        # 4. Nothing is left open past the data. Liquidating keeps the curve honest; the
        #    reason lets GB-19 exclude these if it wants to. A symbol whose history ends
        #    before the master index does is liquidated at ITS OWN last close, stamped with
        #    ITS OWN last traded timestamp - not dropped, and not marked at a price it
        #    never printed.
        if is_final_bar:
            for symbol in list(positions):
                held = positions.pop(symbol)
                cash += _close_out(
                    held, held.last_mark, END_OF_DATA, held.last_mark_time, trades, cfg
                )

        curve.append(_equity(cash, positions))
        fresh = signals.get(timestamp, ())
        superseded = {signal.symbol for signal in _actions(fresh, EXIT)}
        pending = [
            *(signal for signal in unfilled if signal.symbol not in superseded),
            *fresh,
        ]

    _assert_accounted(curve[-1], positions, trades, cfg)
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
            strategy_exit=reason != END_OF_DATA,
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


def _equity(cash: float, positions: Mapping[str, _OpenPosition]) -> float:
    """Cash plus positions at their current marks."""
    return cash + _gross_exposure(positions)


def _gross_exposure(positions: Mapping[str, _OpenPosition]) -> float:
    return sum(held.shares * held.last_mark for held in positions.values())


# Equity is a sum of products of floats; the identity below is exact in real arithmetic
# but not in binary. This tolerance is far tighter than a cent on a 100k account and far
# looser than the accumulated rounding of a multi-year run.
ACCOUNTING_TOLERANCE = 1e-6


def _assert_accounted(
    final_equity: float,
    positions: Mapping[str, _OpenPosition],
    trades: Sequence[Trade],
    cfg: Config,
) -> None:
    """Standing invariant: with nothing open, equity is cash in plus profit realised.

    Every cash movement in this engine belongs to some trade, and ``net_pnl`` is defined as
    exactly that movement. So once the book is flat, the curve's last point must equal the
    starting cash plus the sum of the log - and any gap means money left the account
    without a trade recording it. That is not a rounding question; it is the signature of a
    position dropped from the book, which is the defect this check exists to make loud.

    Skipped when a position is still open, which the loop's final-bar liquidation should
    make impossible; the guard is written so that a future change which leaves one open
    fails visibly rather than tripping this assertion for the wrong reason.
    """
    if positions:
        return
    expected = float(cfg.backtest.initial_cash) + sum(trade.net_pnl for trade in trades)
    if abs(final_equity - expected) > ACCOUNTING_TOLERANCE:
        raise AssertionError(
            f"backtest accounting is broken: final equity {final_equity!r} but initial "
            f"cash plus {len(trades)} trades is {expected!r} "
            f"(gap {final_equity - expected:+.6f}); cash moved without a trade recording it"
        )


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
    if len(combined) == 0:
        raise ValueError("no bars to backtest")
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
