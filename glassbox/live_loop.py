"""X: the live decision loop — ``python -m glassbox.live_loop``.

Spec §3.5's seven steps, every ``cfg.live.poll_seconds`` while the market is open, with the
protection policy running **before** them because entries come after protection.

**Nothing here decides anything.** Every rule this loop applies was ruled and tested
elsewhere: the band is ``engine.signal``'s, the ordering is ``engine.rank``'s, the sizing is
``engine.risk``'s, the submission is ``engine.executor``'s, the truth is
``engine.reconcile``'s and the arithmetic of the explanation is ``contracts.Attribution``'s.
This module is the scheduler and the bookkeeper. Where it appears to make a decision, it is
implementing a recorded ruling, and the ruling is named.

**It may not import the validation harness**, and the import contract enforces that. So it
cannot train a model or calibrate a band; it **loads** both, and
``smoke_offline --prepare-live`` writes them.

Three orderings in this module are load-bearing and none is arbitrary:

1. **Trades are emitted before reconciliation, not after.** A filled stop makes the
   position vanish, and reconciliation correctly drops the holding — which is the only
   record of the entry basis. Emitting first means the trade is built while the system
   still knows why it held the thing.
2. **Protection runs before the forecast.** Rule 2 of the protection policy (2026-08-18):
   re-arm every open position at the start of every session, before anything else. A cycle
   that forecast first would submit an entry before verifying the protection it is supposed
   to follow.
3. **The first cycle may reduce risk and may not add any.** See :func:`run_cycle`.

Implemented in GB-26.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal as signal_module
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

from glassbox import records
from glassbox.config.loader import Config, config_hash, load_config
from glassbox.contracts.schemas import DecisionRecord, Forecast, Trade
from glassbox.data.live import data_source, load_live_bars
from glassbox.engine import risk
from glassbox.engine.executor import (
    BUY,
    CO_PILOT,
    SELL,
    AlpacaBroker,
    Broker,
    BrokerOrder,
    Submission,
    execute,
)
from glassbox.engine.rank import rank_signals
from glassbox.engine.reconcile import Book, Holding, reconcile
from glassbox.engine.signal import EXIT, Thresholds, decide
from glassbox.explain.channel import attribute
from glassbox.explain.narrate import EN, narrate
from glassbox.features.builder import (
    build_feature_frame,
    history_requirement,
    min_history_bars,
)
from glassbox.model.predict import Predictor, load_predictor, predict_window

LOGGER = logging.getLogger("glassbox.live_loop")

# `cfg.live.market_open_il` names its own zone. A config key for it could only ever
# disagree with the field it describes, which is why this is a constant and not a setting -
# flagged rather than buried, because rule 5 is config over constants.
ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")

# Rule 1 of the protection policy: arm in the **same cycle that observes the fill**, polling
# for it rather than waiting for the next tick. The open is the wrong minute to be idle.
FILL_POLL_ATTEMPTS = 10
FILL_POLL_SECONDS = 1.5

# Rule 3: a position without protection is a risk event. Arm immediately, and flatten at
# market if arming fails twice in succession - an unprotected position is worse than a
# closed one.
ARMING_STRIKES = 2

STOP_LEG = "stop"
TARGET_LEG = "target"
LEGS = (STOP_LEG, TARGET_LEG)
LEG_SUFFIX = {STOP_LEG: records.STOP_SUFFIX, TARGET_LEG: records.TARGET_SUFFIX}

FINISHED = frozenset(
    {
        "filled",
        "canceled",
        "cancelled",
        "expired",
        "rejected",
        "done_for_day",
        "replaced",
    }
)

BOOK_FILE = "book.json"
ENTRY_FILLS_FILE = "entry_fills.json"
THRESHOLDS_FILE = "thresholds.json"
CHECKPOINT_DIR = "checkpoint"


class LiveError(RuntimeError):
    """Something the session cannot start without. Reported without a traceback."""


# ── reports ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CycleReport:
    """What one cycle did. Everything a post-mortem needs, and nothing derived."""

    cycle_id: str
    at: pd.Timestamp
    entries_allowed: bool
    ranked_over: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()
    divergences: tuple[str, ...] = ()
    rearmed: tuple[str, ...] = ()
    flattened: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    submissions: tuple[Submission, ...] = ()
    trades: tuple[Trade, ...] = ()
    failed_step: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.failed_step is None


@dataclass(frozen=True)
class SessionReport:
    """The session, as a whole. Printed at shutdown."""

    session_id: str
    banner: str
    cycles: tuple[CycleReport, ...]
    open_orders: tuple[str, ...]
    stopped_by: str

    def summary(self) -> str:
        failed = [cycle for cycle in self.cycles if not cycle.ok]
        lines = [
            f"session {self.session_id} stopped by {self.stopped_by}",
            f"  cycles completed     : {len(self.cycles)}, {len(failed)} with a failed step",
            f"  decisions recorded   : {sum(len(c.decisions) for c in self.cycles)}",
            f"  orders submitted     : {sum(len(c.submissions) for c in self.cycles)}",
            f"  trades emitted       : {sum(len(c.trades) for c in self.cycles)}",
            f"  positions re-armed   : {sum(len(c.rearmed) for c in self.cycles)}",
            f"  positions flattened  : {sum(len(c.flattened) for c in self.cycles)}",
        ]
        for cycle in failed:
            lines.append(
                f"  FAILED {cycle.cycle_id} at {cycle.failed_step}: {cycle.error}"
            )
        lines.append(
            f"  open orders at exit  : {len(self.open_orders)}"
            + ("" if not self.open_orders else " — " + "; ".join(self.open_orders))
        )
        return "\n".join(lines)


@dataclass
class LiveState:
    """Everything the loop carries between cycles. Persisted where a restart needs it."""

    cfg: Config
    broker: Broker
    predictor: Predictor
    thresholds: Thresholds
    state_dir: Path
    language: str = EN
    book: Book = field(default_factory=Book)
    entry_fills: dict[str, list] = field(default_factory=dict)
    arming_failures: dict[str, int] = field(default_factory=dict)
    seen_orders: set[str] = field(default_factory=set)
    cycles: int = 0

    def save(self) -> None:
        self.book.save(self.state_dir / BOOK_FILE)
        (self.state_dir / ENTRY_FILLS_FILE).write_text(
            json.dumps(self.entry_fills, indent=2), encoding="utf-8"
        )


# ── the calendar, which decides when there is a market at all ────────────────


def market_session(
    cfg: Config, when: pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """The exchange's own open and close for ``when``'s date, in UTC, or ``None``.

    Read from the calendar rather than from the configured window, so a holiday closes the
    loop and a **half-day closes it early**. The configured Israel window says when the
    process is willing to run; the calendar says whether there is a market to run against,
    and a 13:00 ET early close would otherwise leave the loop polling a shut exchange until
    23:00 Israel time.
    """
    schedule = mcal.get_calendar(cfg.data.calendar).schedule(
        start_date=when.date(), end_date=when.date()
    )
    if schedule.empty:
        return None
    row = schedule.iloc[0]
    return pd.Timestamp(row["market_open"]).tz_convert("UTC"), pd.Timestamp(
        row["market_close"]
    ).tz_convert("UTC")


def israel_window(cfg: Config, when: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The configured Israel-time window on ``when``'s date, in UTC."""
    local = when.tz_convert(ISRAEL_TZ)
    bounds = []
    for text in (cfg.live.market_open_il, cfg.live.market_close_il):
        hour, minute = (int(part) for part in text.split(":"))
        bounds.append(
            pd.Timestamp(
                datetime(
                    local.year, local.month, local.day, hour, minute, tzinfo=ISRAEL_TZ
                )
            ).tz_convert("UTC")
        )
    return bounds[0], bounds[1]


def in_session(cfg: Config, when: pd.Timestamp) -> bool:
    """True when the exchange is open **and** the configured window is."""
    session = market_session(cfg, when)
    if session is None:
        return False
    opened, closed = session
    window_open, window_close = israel_window(cfg, when)
    return max(opened, window_open) <= when < min(closed, window_close)


def last_completed_session(cfg: Config, when: pd.Timestamp) -> pd.Timestamp:
    """The date of the most recent session whose close is at or before ``when``.

    During a session that is yesterday, because today's bar is still forming. This is the
    single definition of "completed" that :func:`drop_incomplete_bar` and :func:`is_stale`
    both read, so the two cannot disagree about which bar is the last real one.
    """
    schedule = mcal.get_calendar(cfg.data.calendar).schedule(
        start_date=(when - pd.Timedelta(days=14)).date(), end_date=when.date()
    )
    closes = pd.DatetimeIndex(schedule["market_close"]).tz_convert("UTC")
    completed = closes[closes <= when]
    if len(completed) == 0:
        raise LiveError(
            f"no completed exchange session in the fortnight before {when}; the calendar "
            f"{cfg.data.calendar!r} cannot place the last completed bar"
        )
    return completed[-1].normalize()


def drop_incomplete_bar(
    frame: pd.DataFrame, cfg: Config, when: pd.Timestamp, symbol: str = ""
) -> pd.DataFrame:
    """Remove any bar for a session that has not closed yet, logging each one.

    The model is trained on completed daily bars. A bar for the session now in progress has
    a close that is simply the last trade so far, and feeding it to the window produces a
    forecast from a number that will still change — which is not a bug the parity test can
    see, because the schema is identical.
    """
    cutoff = last_completed_session(cfg, when)
    incomplete = frame.index[frame.index > cutoff]
    for timestamp in incomplete:
        LOGGER.info(
            "dropping in-progress bar %s for %s: the %s session has not closed",
            timestamp.strftime("%Y-%m-%d"),
            symbol or "the universe",
            timestamp.strftime("%Y-%m-%d"),
        )
    return frame.loc[frame.index <= cutoff]


def is_stale(frame: pd.DataFrame, cfg: Config, when: pd.Timestamp) -> bool:
    """True when the last completed bar predates the last completed session."""
    if frame.empty:
        return True
    return frame.index[-1].normalize() < last_completed_session(cfg, when)


# ── the dry-run broker ───────────────────────────────────────────────────────


@dataclass
class DryRunBroker:
    """Every read passes through; every write is logged and refused.

    A wrapper rather than a branch in the loop, so ``--dry-run`` exercises ``execute``,
    ``protect`` and the fill poll on the real code path against real positions, real equity
    and the real order history. A dry run that skipped those calls would test the scheduler
    and nothing else.
    """

    inner: Broker
    submitted: list[str] = field(default_factory=list)
    _ids: int = 0

    def _refuse(self, description: str) -> BrokerOrder:
        self._ids += 1
        LOGGER.warning("DRY RUN: would have submitted %s", description)
        self.submitted.append(description)
        return BrokerOrder(
            id=f"dry-{self._ids}",
            symbol=description.split()[-1],
            side=BUY,
            quantity=0.0,
            status="dry_run",
            raw={"client_order_id": "dry-run"},
        )

    def submit_market_order(
        self, symbol, quantity, side, client_order_id
    ) -> BrokerOrder:
        return self._refuse(f"MARKET {side} {quantity:.9f} {symbol}")

    def submit_stop_order(
        self, symbol, quantity, stop_price, client_order_id
    ) -> BrokerOrder:
        return self._refuse(f"STOP sell {quantity:.9f} at {stop_price:.4f} {symbol}")

    def submit_limit_order(
        self, symbol, quantity, limit_price, client_order_id
    ) -> BrokerOrder:
        return self._refuse(f"LIMIT sell {quantity:.9f} at {limit_price:.4f} {symbol}")

    def cancel_order(self, order_id: str) -> None:
        LOGGER.warning("DRY RUN: would have cancelled %s", order_id)
        self.submitted.append(f"CANCEL {order_id}")

    def get_orders(self) -> list[BrokerOrder]:
        return self.inner.get_orders()

    def get_positions(self) -> dict[str, float]:
        return self.inner.get_positions()

    def get_account(self) -> dict[str, float]:
        return self.inner.get_account()


# ── loading what the harness produced ────────────────────────────────────────


def load_thresholds(path: str | Path) -> Thresholds:
    """Read the calibrated band, or stand aside.

    A missing file is **not** an error and **not** a default band. It is
    ``Thresholds.never()`` — the same object GB-20 returns for a fold whose validation found
    nothing worth trading, with the same consequence: no entry can fire, the session runs,
    records everything and trades nothing, and GB-32 narrates it as standing aside. Any
    other choice would mean inventing a threshold, and a made-up band is the one input that
    would make every decision in the session unexplainable.
    """
    source = Path(path)
    if not source.is_file():
        LOGGER.warning(
            "no calibrated band at %s; standing aside for the whole session. Produce one "
            "with: python -m glassbox.smoke_offline --prepare-live <dir>",
            source,
        )
        return Thresholds.never()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if raw.get("stood_aside"):
        LOGGER.warning(
            "the calibrated band at %s stands aside: validation found no candidate with a "
            "positive Sharpe. No entry can fire this session, which is the ruling and not "
            "a failure",
            source,
        )
        return Thresholds.never()
    return Thresholds(
        lower=float(raw["lower"]), upper=_optional_float(raw.get("upper"))
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def session_banner(state: LiveState, when: pd.Timestamp, dry_run: bool) -> str:
    """What this session is, on one screen, before it does anything.

    Names the tape, because GB-30 measured an undetected feed downgrade at up to 190x the
    price tolerance; names the config hash, because a decision record is replayable only
    against the configuration that made it; and names the band, because a session that
    stands aside should say so at the top rather than at the end.
    """
    cfg = state.cfg
    session = market_session(cfg, when)
    thresholds = state.thresholds
    band = (
        "stood aside (no calibrated band)"
        if not thresholds.fires
        else f"lower={thresholds.lower:.6f} upper="
        + ("none" if thresholds.upper is None else f"{thresholds.upper:.6f}")
    )
    hours = (
        "closed today"
        if session is None
        else f"{session[0]:%H:%M} - {session[1]:%H:%M} UTC"
    )
    return "\n".join(
        [
            "=" * 78,
            f"GlassBox live session  {when:%Y-%m-%d %H:%M %Z}"
            + ("  [DRY RUN]" if dry_run else ""),
            "=" * 78,
            f"  data source        : {data_source()}",
            f"  model              : {state.predictor.model.name} from {state.state_dir}",
            (
                f"  channels           : {list(cfg.channels.active_channels)} "
                f"(input_len={cfg.window.input_len}, horizon={cfg.window.horizon})"
            ),
            f"  history required   : {history_requirement(cfg)}",
            f"  config hash        : {config_hash(cfg)}",
            f"  thresholds         : {band}",
            f"  mode               : {cfg.live.mode}",
            f"  universe           : {list(cfg.universe)}  top_k={cfg.signal.top_k}",
            f"  poll               : every {cfg.live.poll_seconds}s",
            f"  exchange session   : {hours}",
            (
                f"  configured window  : {cfg.live.market_open_il}"
                f"-{cfg.live.market_close_il} Israel time"
            ),
            "=" * 78,
        ]
    )


# ── the cycle ────────────────────────────────────────────────────────────────


def run_cycle(state: LiveState, when: pd.Timestamp) -> CycleReport:
    """One pass of the loop.

    **The first cycle may reduce risk and may not add any.** No entry is submitted on it;
    exits, protection and reconciliation all run in full. Three reasons, in order of
    weight: rule 2 of the protection policy says re-arm before entries and the first cycle
    *is* that moment; reconciliation has just rebuilt the book from a single observation of
    the account; and after a mid-session restart an entry the previous process submitted can
    be briefly absent from ``get_orders``, so entering again would double a position the
    loop cannot un-double. An exit is not deferred for the same reason GB-28 will not let
    one be crowded out — it is an obligation on capital already committed.
    """
    state.cycles += 1
    cycle_id = f"{state.state_dir.name}-{state.cycles:04d}"
    entries_allowed = state.cycles > 1
    log = _cycle_logger(cycle_id)
    log("step 1/10 orders: reading the broker's order history")

    try:
        orders = state.broker.get_orders()

        # Trades BEFORE reconciliation. A filled stop makes the position vanish and
        # reconcile correctly drops the holding - which is the only record of the entry
        # basis. Emitting first builds the trade while the system still knows why it held.
        log("step 2/10 trades: emitting from sells filled since the last cycle")
        fresh = [
            order
            for order in orders
            if order.id not in state.seen_orders
            and order.side == SELL
            and order.status in records.FILLED
        ]
        trades = tuple(
            records.emit_trades(
                fresh, state.book, {k: tuple(v) for k, v in state.entry_fills.items()}
            )
        )
        for trade in trades:
            log(
                f"trade {trade.symbol} {trade.exit_reason} size={trade.size:.9f} "
                f"net={trade.net_pnl:+.2f} costs={trade.costs:.4f}"
            )
        state.seen_orders |= {order.id for order in orders}

        log("step 3/10 reconcile: the broker is the truth")
        result = reconcile(state.broker, state.book)
        state.book = result.book
        for symbol in list(state.entry_fills):
            if symbol not in state.book.managed:
                state.entry_fills.pop(symbol)
        divergences = tuple(str(divergence) for divergence in result.divergences)

        log(
            "step 4/10 protection: verify both legs, re-arm, flatten on a second failure"
        )
        rearmed, flattened = protect_book(state, orders, log)
        state.save()

        log("step 5/10 data: completed daily bars for the universe")
        bars = load_live_bars(
            list(state.cfg.universe),
            min_history_bars(state.cfg),
            requirement=history_requirement(state.cfg),
        )

        log("step 6/10 features: drop the in-progress bar, then the keystone builder")
        frames: dict[str, pd.DataFrame] = {}
        stale: list[str] = []
        for symbol in sorted(bars):
            completed = drop_incomplete_bar(bars[symbol], state.cfg, when, symbol)
            if is_stale(completed, state.cfg, when):
                last = "none" if completed.empty else f"{completed.index[-1]:%Y-%m-%d}"
                LOGGER.warning(
                    "[%s] %s is STALE: last completed bar %s, expected %s. Excluded from "
                    "ranking and entries; its protective legs are unaffected and were "
                    "verified in step 4 - a stale window cannot justify opening risk and "
                    "must not suspend the management of risk already taken",
                    cycle_id,
                    symbol,
                    last,
                    f"{last_completed_session(state.cfg, when):%Y-%m-%d}",
                )
                stale.append(symbol)
                continue
            frames[symbol] = build_feature_frame(completed, state.cfg)

        if not frames:
            log("every symbol is stale; no forecast this cycle. Protection stands.")
            return CycleReport(
                cycle_id=cycle_id,
                at=when,
                entries_allowed=entries_allowed,
                stale=tuple(stale),
                divergences=divergences,
                rearmed=tuple(rearmed),
                flattened=tuple(flattened),
                trades=trades,
            )

        log(f"step 7/10 forecast: {len(frames)} symbol(s), stale {stale or 'none'}")
        forecasts: dict[str, Forecast] = {}
        for symbol, frame in frames.items():
            forecasts[symbol] = predict_window(
                state.predictor, frame, state.cfg, symbol, frame.index[-1]
            )

        log("step 8/10 decide, rank, size")
        signals = {
            symbol: decide(forecast, state.thresholds, state.cfg)
            for symbol, forecast in forecasts.items()
        }
        ranked = rank_signals(list(signals.values()), state.cfg)
        prices = {symbol: float(bars[symbol]["close"].iloc[-1]) for symbol in frames}
        account = state.broker.get_account()
        gross = sum(
            state.book.committed(symbol) * prices.get(symbol, 0.0)
            for symbol in state.book.symbols()
        )
        orders_to_place = (
            risk.size_positions(
                [s for s in ranked if s.symbol not in state.book.symbols()],
                float(account.get("equity", 0.0)),
                prices,
                state.cfg,
                gross_exposure=gross,
            )
            if entries_allowed
            else []
        )
        if not entries_allowed:
            log(
                "first cycle: entries are held back until protection has been verified "
                "once. Exits, protection and reconciliation ran in full"
            )

        log("step 9/10 explain, narrate, execute, persist")
        decisions: list[str] = []
        submissions: list[Submission] = []
        sized = {order.symbol: order for order in orders_to_place}
        for symbol in sorted(signals):
            decision_id = f"{cycle_id}-{symbol}"
            verdict = signals[symbol]
            forecast = forecasts[symbol]
            found = attribute(
                state.predictor.model,
                _window_of(state, frames[symbol], symbol, forecast.as_of),
                state.cfg.channels.active_channels,
            )
            order = sized.get(symbol)
            story = narrate(
                forecast,
                found,
                verdict,
                state.thresholds,
                state.cfg,
                order=order,
                language=state.language,
            )
            LOGGER.info("[%s] %s", decision_id, story.text)

            submission = None
            if verdict.action == EXIT and symbol in state.book.managed:
                submission = exit_position(state, symbol, decision_id, log)
            elif order is not None:
                submission = execute(state.broker, order, decision_id, state.cfg)
                _absorb_entry(state, order, submission, when, log)
            if submission is not None:
                submissions.append(submission)

            records.save_decision(
                DecisionRecord(
                    as_of=forecast.as_of,
                    symbol=symbol,
                    forecast=forecast,
                    attribution=found,
                    signal=verdict,
                    order=None if order is None else _order_dict(order, decision_id),
                    narrative=story.text,
                    config_hash=config_hash(state.cfg),
                ),
                state.state_dir,
            )
            decisions.append(decision_id)

        log("step 10/10 persist: book and entry fills")
        state.save()
        return CycleReport(
            cycle_id=cycle_id,
            at=when,
            entries_allowed=entries_allowed,
            ranked_over=tuple(sorted(frames)),
            stale=tuple(stale),
            divergences=divergences,
            rearmed=tuple(rearmed),
            flattened=tuple(flattened),
            decisions=tuple(decisions),
            submissions=tuple(submissions),
            trades=trades,
        )

    except Exception as failure:
        LOGGER.exception("[%s] cycle failed", cycle_id)
        return CycleReport(
            cycle_id=cycle_id,
            at=when,
            entries_allowed=entries_allowed,
            failed_step=type(failure).__name__,
            error=str(failure),
        )


# ── protection: the five rules, implemented ──────────────────────────────────


def protect_book(
    state: LiveState, orders: Sequence[BrokerOrder], log
) -> tuple[list[str], list[str]]:
    """Verify both legs of every managed position, arm what is missing, flatten on a
    second consecutive failure, and cancel a sibling whose partner has filled.

    Rules 2, 3 and 4 of the protection policy (DECISIONS, 2026-08-18). Reconciliation
    detects a missing leg and deliberately does not act on it; this is where acting lives,
    because ``reconcile`` may only ever read.
    """
    live: dict[str, dict[str, BrokerOrder]] = {}
    filled_leg: dict[str, str] = {}
    for order in orders:
        if order.side != SELL:
            continue
        leg = _leg_of(order)
        if leg is None:
            continue
        if order.status in FINISHED:
            if order.status in records.FILLED:
                filled_leg.setdefault(order.symbol, leg)
            continue
        live.setdefault(order.symbol, {})[leg] = order

    rearmed: list[str] = []
    flattened: list[str] = []
    for symbol, holding in sorted(state.book.managed.items()):
        legs = live.get(symbol, {})

        if symbol in filled_leg:
            # Rule 4: one leg filled, so cancel the other in this cycle and verify the
            # cancellation rather than assuming the fill implies it.
            for leg, order in sorted(legs.items()):
                state.broker.cancel_order(order.id)
                log(
                    f"{symbol}: {filled_leg[symbol]} filled, cancelled sibling {leg} {order.id}"
                )
            still_live = [
                order
                for order in state.broker.get_orders()
                if order.symbol == symbol
                and order.side == SELL
                and order.status not in FINISHED
                and _leg_of(order) is not None
            ]
            if still_live:
                LOGGER.error(
                    "%s: %s protective order(s) still live after cancelling the sibling "
                    "of a filled leg; a naked sell can be left behind by this",
                    symbol,
                    len(still_live),
                )
            continue

        missing = [leg for leg in LEGS if leg not in legs]
        if not missing:
            continue

        LOGGER.error(
            "RISK EVENT: %s is a managed position missing its %s. Arming now (rule 3); "
            "%s consecutive failures flatten it at market",
            symbol,
            " and ".join(missing),
            ARMING_STRIKES,
        )
        try:
            for leg in missing:
                _arm_leg(state, symbol, holding, leg, log)
            state.arming_failures.pop(symbol, None)
            rearmed.append(symbol)
        except Exception as failure:  # noqa: BLE001 - the strike count is the policy
            strikes = state.arming_failures.get(symbol, 0) + 1
            state.arming_failures[symbol] = strikes
            LOGGER.error(
                "%s: arming failed (%s/%s): %s",
                symbol,
                strikes,
                ARMING_STRIKES,
                failure,
            )
            if strikes >= ARMING_STRIKES:
                _flatten(state, symbol, holding, log)
                flattened.append(symbol)
                state.arming_failures.pop(symbol, None)
    return rearmed, flattened


def _arm_leg(state: LiveState, symbol: str, holding: Holding, leg: str, log) -> None:
    client_order_id = f"{holding.decision_id}{LEG_SUFFIX[leg]}"
    if leg == STOP_LEG:
        order = state.broker.submit_stop_order(
            symbol=symbol,
            quantity=holding.quantity,
            stop_price=holding.stop_loss,
            client_order_id=client_order_id,
        )
        log(f"{symbol}: stop re-armed id={order.id} at {holding.stop_loss:.4f}")
    else:
        order = state.broker.submit_limit_order(
            symbol=symbol,
            quantity=holding.quantity,
            limit_price=holding.take_profit,
            client_order_id=client_order_id,
        )
        log(f"{symbol}: target re-armed id={order.id} at {holding.take_profit:.4f}")


def _flatten(state: LiveState, symbol: str, holding: Holding, log) -> None:
    """Close at market. An unprotected position is worse than a closed one (rule 3)."""
    LOGGER.error(
        "FLATTENING %s at market: arming failed %s times in succession and an unprotected "
        "position is worse than a closed one",
        symbol,
        ARMING_STRIKES,
    )
    order = state.broker.submit_market_order(
        symbol=symbol,
        quantity=holding.quantity,
        side=SELL,
        client_order_id=f"{holding.decision_id}-flatten",
    )
    log(f"{symbol}: flatten submitted id={order.id}")


def exit_position(
    state: LiveState, symbol: str, decision_id: str, log
) -> Submission | None:
    """Close a managed position on a signal exit, cancelling protection first.

    Co-Pilot applies here exactly as it does to an entry: the mode means a human approves
    what the system does, and a system that asked permission to buy but not to sell would
    be two policies wearing one name. The stop and the target stay live meanwhile, so a
    recommended-but-unapproved exit is a position that is still protected.
    """
    holding = state.book.managed[symbol]
    order = risk.Order(
        symbol=symbol,
        shares=holding.quantity,
        price=holding.entry_price,
        notional=holding.quantity * holding.entry_price,
        stop_loss=holding.stop_loss,
        take_profit=holding.take_profit,
    )
    if state.cfg.live.mode == CO_PILOT:
        log(
            f"co_pilot: recommending EXIT {holding.quantity:.9f} {symbol}; not submitted"
        )
        return Submission(
            decision_id=decision_id,
            order=order,
            status="pending_approval",
            reason="co_pilot mode: an exit needs approval too (GB-37)",
        )

    for leg_order in state.broker.get_orders():
        if (
            leg_order.symbol == symbol
            and leg_order.side == SELL
            and _leg_of(leg_order)
            and leg_order.status not in FINISHED
        ):
            state.broker.cancel_order(leg_order.id)
            log(f"{symbol}: cancelled {leg_order.id} before the signal exit")

    sold = state.broker.submit_market_order(
        symbol=symbol,
        quantity=holding.quantity,
        side=SELL,
        client_order_id=decision_id,
    )
    log(f"{symbol}: signal exit submitted id={sold.id} status={sold.status}")
    return Submission(
        decision_id=decision_id, order=order, status="submitted", entry=sold
    )


def _absorb_entry(
    state: LiveState, order: risk.Order, submission: Submission, when: pd.Timestamp, log
) -> None:
    """Poll for the entry fill and take the position into the book in the same cycle.

    Rule 1: arm protection in the cycle that observes the fill. ``executor.execute`` already
    arms when the entry comes back filled; a market order accepted a moment before the fill
    would otherwise wait a whole poll interval, and the open is the wrong minute to be idle.
    """
    entry = submission.entry
    if entry is None:
        return

    for attempt in range(FILL_POLL_ATTEMPTS):
        if entry.filled_quantity > 0.0:
            break
        time.sleep(FILL_POLL_SECONDS)
        entry = _refresh(state.broker, entry)
        log(
            f"{order.symbol}: waiting for the entry fill ({attempt + 1}/{FILL_POLL_ATTEMPTS}) status={entry.status}"
        )

    if entry.filled_quantity <= 0.0:
        log(
            f"{order.symbol}: entry {entry.id} still unfilled after "
            f"{FILL_POLL_ATTEMPTS} polls; protection will be armed by the cycle that sees "
            "the fill (step 4)"
        )
        return

    filled_price = float(entry.filled_price or order.price)
    if not submission.protection:
        for leg in LEGS:
            _arm_leg(
                state,
                order.symbol,
                Holding(
                    symbol=order.symbol,
                    quantity=entry.filled_quantity,
                    decision_id=submission.decision_id,
                    entry_price=filled_price,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                ),
                leg,
                log,
            )

    state.book.managed[order.symbol] = Holding(
        symbol=order.symbol,
        quantity=entry.filled_quantity,
        decision_id=submission.decision_id,
        entry_price=filled_price,
        stop_loss=order.stop_loss,
        take_profit=order.take_profit,
    )
    # The entry basis GB-29 needs, measured rather than assumed: the entry's cost is what
    # the fill differed from the price the sizer used, which is the live half of the
    # comparison GB-57 makes against the modelled 2 bps.
    state.entry_fills[order.symbol] = [
        _iso(when),
        filled_price,
        abs(filled_price - order.price) * entry.filled_quantity,
        entry.id,
    ]
    log(
        f"{order.symbol}: filled {entry.filled_quantity:.9f} at {filled_price:.4f} "
        f"(reference {order.price:.4f}); taken into the book and protected"
    )


# ── the session ──────────────────────────────────────────────────────────────


def run_session(
    cfg: Config,
    state_dir: str | Path,
    *,
    broker: Broker | None = None,
    dry_run: bool = False,
    language: str = EN,
    clock=None,
    sleep=time.sleep,
    max_cycles: int | None = None,
) -> SessionReport:
    """Poll the market for a session, or return immediately when there is no market.

    Args:
        cfg: Resolved configuration.
        state_dir: Where the checkpoint, the band, the book and the decision log live.
        broker: Injected for tests. The live path builds :class:`AlpacaBroker`, which
            refuses a non-paper endpoint before any call.
        dry_run: Wrap the broker so every write is logged and refused.
        language: Narration language.
        clock: ``() -> pd.Timestamp`` in UTC. Injected so a session can be tested without
            waiting for one.
        sleep: Injected for the same reason.
        max_cycles: Stop after this many cycles. ``None`` runs to the close.
    """
    root = Path(state_dir)
    clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
    predictor = load_predictor(root / CHECKPOINT_DIR, cfg)
    thresholds = load_thresholds(root / THRESHOLDS_FILE)

    live_broker = broker if broker is not None else AlpacaBroker()
    if dry_run:
        live_broker = DryRunBroker(live_broker)

    state = LiveState(
        cfg=cfg,
        broker=live_broker,
        predictor=predictor,
        thresholds=thresholds,
        state_dir=root,
        language=language,
        book=Book.load(root / BOOK_FILE),
        entry_fills=_load_entry_fills(root / ENTRY_FILLS_FILE),
    )

    started = clock()
    banner = session_banner(state, started, dry_run)
    for line in banner.splitlines():
        LOGGER.info(line)

    stopping = {"reason": ""}

    def request_stop(signum, frame) -> None:  # pragma: no cover - signal delivery
        del signum, frame
        stopping["reason"] = "SIGINT"
        LOGGER.warning("SIGINT received; finishing the current cycle and stopping")

    previous = _install_sigint(request_stop)
    cycles: list[CycleReport] = []
    try:
        while True:
            now = clock()
            if stopping["reason"]:
                break
            if max_cycles is not None and len(cycles) >= max_cycles:
                stopping["reason"] = f"max_cycles={max_cycles}"
                break
            if not in_session(cfg, now):
                if cycles:
                    stopping["reason"] = "the market closed"
                    break
                LOGGER.info(
                    "no session at %s (%s): the exchange is shut or the configured window "
                    "has not opened. Nothing is fetched",
                    now,
                    cfg.data.calendar,
                )
                stopping["reason"] = "outside the session"
                break
            cycles.append(run_cycle(state, now))
            if stopping["reason"]:
                break
            sleep(cfg.live.poll_seconds)
    except KeyboardInterrupt:  # pragma: no cover - depends on delivery timing
        stopping["reason"] = "KeyboardInterrupt"
    finally:
        _restore_sigint(previous)

    open_orders = _open_orders(state)
    state.save()
    return SessionReport(
        session_id=root.name,
        banner=banner,
        cycles=tuple(cycles),
        open_orders=open_orders,
        stopped_by=stopping["reason"] or "the market closed",
    )


def _open_orders(state: LiveState) -> tuple[str, ...]:
    """Every order still working at shutdown, named. "Nothing in flight" is a list."""
    try:
        return tuple(
            f"{order.symbol} {order.side} id={order.id} status={order.status} "
            f"qty={order.quantity:.9f}"
            for order in state.broker.get_orders()
            if order.status not in FINISHED
        )
    except Exception as failure:  # noqa: BLE001 - shutdown must still report
        LOGGER.error("could not list open orders at shutdown: %s", failure)
        return (f"UNKNOWN: {failure}",)


# ── helpers ──────────────────────────────────────────────────────────────────


def _cycle_logger(cycle_id: str):
    def log(message: str) -> None:
        LOGGER.info("[%s] %s", cycle_id, message)

    return log


def _leg_of(order: BrokerOrder) -> str | None:
    client_id = records._client_order_id(order)
    for leg, suffix in LEG_SUFFIX.items():
        if client_id.endswith(suffix):
            return leg
    return None


def _refresh(broker: Broker, entry: BrokerOrder) -> BrokerOrder:
    for order in broker.get_orders():
        if order.id == entry.id:
            return order
    return entry


def _window_of(state: LiveState, frame: pd.DataFrame, symbol: str, as_of: pd.Timestamp):
    """The same window the forecast was made from, for the attribution.

    Built by the keystone with the checkpoint's own statistics, so the explanation
    decomposes the forecast that was actually made rather than a re-derivation of it.
    """
    from glassbox.features.builder import build_windows

    batch = build_windows(
        frame, state.cfg, symbol, stats=state.predictor.stats_for(symbol), as_of=as_of
    )
    return batch.X[0]


def _order_dict(order: risk.Order, decision_id: str) -> dict:
    return {
        "decision_id": decision_id,
        "symbol": order.symbol,
        "shares": order.shares,
        "price": order.price,
        "notional": order.notional,
        "stop_loss": order.stop_loss,
        "take_profit": order.take_profit,
    }


def _iso(when: pd.Timestamp) -> str:
    return pd.Timestamp(when).tz_convert("UTC").isoformat()


def _load_entry_fills(path: Path) -> dict[str, list]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _install_sigint(handler):  # pragma: no cover - depends on the host
    try:
        return signal_module.signal(signal_module.SIGINT, handler)
    except ValueError:
        return None


def _restore_sigint(previous) -> None:  # pragma: no cover - depends on the host
    if previous is not None:
        try:
            signal_module.signal(signal_module.SIGINT, previous)
        except ValueError:
            pass


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        stream=sys.stdout,
    )
    try:
        report = run_session(
            load_config(),
            args.state_dir,
            dry_run=args.dry_run,
            language=args.language,
            max_cycles=args.max_cycles,
        )
    except (LiveError, FileNotFoundError, ValueError) as failure:
        print(f"live_loop: {failure}", file=sys.stderr)
        return 2

    print()
    print(report.summary())
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m glassbox.live_loop",
        description=(
            "Run the live decision cycle for one session. Reads a checkpoint and a "
            "calibrated band produced by 'python -m glassbox.smoke_offline --prepare-live'."
        ),
    )
    parser.add_argument(
        "--state-dir",
        default="checkpoints/live",
        help="where the checkpoint, band, book and decision log live",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run every step except order submission",
    )
    parser.add_argument("--language", choices=("en", "he"), default="en")
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="stop after this many cycles (default: run to the close)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI, not by tests
    raise SystemExit(main())


__all__ = [
    "ARMING_STRIKES",
    "CycleReport",
    "DryRunBroker",
    "LiveError",
    "LiveState",
    "SessionReport",
    "drop_incomplete_bar",
    "in_session",
    "is_stale",
    "last_completed_session",
    "load_thresholds",
    "main",
    "market_session",
    "run_cycle",
    "run_session",
    "session_banner",
]
