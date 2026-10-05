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
import os
import signal as signal_module
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

from glassbox import records
from glassbox.config.loader import (
    Config,
    config_hash,
    load_config,
    model_config_hash,
)
from glassbox.contracts.schemas import DecisionRecord, Forecast, Signal, Trade
from glassbox.data.live import data_source, load_live_bars
from glassbox.engine import risk
from glassbox.engine.executor import (
    BUY,
    CO_PILOT,
    PENDING_APPROVAL,
    SELL,
    AlpacaBroker,
    Broker,
    BrokerOrder,
    RetryingBroker,
    Submission,
    approve,
    client_order_id,
    decline,
    execute,
)
from glassbox.engine.rank import rank_signals
from glassbox.engine.reconcile import Book, Holding, reconcile
from glassbox.engine.signal import ENTER_LONG, EXIT, Thresholds, decide
from glassbox.explain.channel import attribute
from glassbox.explain.narrate import EN, narrate
from glassbox.explain.spectral import Spectral, explain_spectral
from glassbox.faults import Unavailable
from glassbox.features.builder import (
    TARGET_CHANNEL,
    build_feature_frame,
    history_requirement,
    min_history_bars,
)
from glassbox.live_lock import EXIT_REFUSED, LockRefused, acquire, release
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

# The ordinary end of a session, and the **only** reason a multi-session run continues to
# the next one. Named because `run_session` writes it and `run_sessions` reads it, and a
# string spelt in two places is the defect family this project has five instances of.
CLOSED = "the market closed"

# GB-40's rehearsal defaults. Operator inputs for a one-off run rather than strategy
# parameters, which is why they are CLI arguments with these as defaults rather than
# configuration: a rehearsal is something a person decides to do on a particular
# afternoon, and a configured value would be a value in every model's config hash.
#: Where the deployed loop keeps its checkpoint, band, book and decision log. Named rather
#: than repeated so the CLI default and the dry-run guard below cannot drift apart.
DEFAULT_STATE_DIR = "checkpoints/live"

#: Where the session log is written, one file per calendar day. An operator path like
#: DEFAULT_STATE_DIR rather than a configured value, for the reason recorded just above: a
#: settings key would become a value in every model's config hash, and where a log file
#: lands says nothing about what any model learned.
DEFAULT_LOG_DIR = "logs"

#: ``main`` returns this when a run ended holding positions it could not flatten. Distinct
#: from the general failure code because the operator action is different and urgent: a
#: position is open at the broker and nothing is managing it.
EXIT_CLOSE_OUT_FAILED = 4

#: One format for both handlers. A second copy of a format string is a second thing to
#: change, and stdout and the file quietly diverging is the kind of drift that is only
#: discovered when the terminal is gone and the file is all that is left.
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"

#: **UTC, and it says so.** ``DailyLogFile`` names its file for the UTC day, but
#: ``asctime`` defaults to local time, so on 26 Aug 2026 ``live-2026-08-25.log`` carried
#: lines stamped ``2026-08-26 00:00``: a reader looking for 01:00 on the 26th opens the
#: wrong file. Two places holding one fact - which day this record belongs to - in two
#: timezones. The loop's own messages were already UTC (``heartbeat ...Z``), so the
#: divergence was inside a single line. Pinned by
#: ``test_the_record_stamp_and_the_filename_agree_on_the_day``.
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

REHEARSAL_NOTIONAL = 25.0
REHEARSAL_CLOSE_OUT_MINUTES = 15

# The rehearsal band. `Thresholds` refuses a lower bound of zero, so this is the smallest
# positive value that carries meaning at float64 - unmistakably not a calibration, which
# is exactly what it is for.
PERMISSIVE_LOWER = 1e-9

#: The bar column the loop-side target reads. Named here rather than spelled at the
#: comparison, so the one place it is used cannot drift from the canonical frame.
HIGH = "high"

STOP_LEG = "stop"
TARGET_LEG = "target"

#: What the loop arms and verifies every cycle. **The stop, and only the stop** (ruled
#: 26 Aug 2026): a working sell order holds the entire position at Alpaca, so a standalone
#: stop and a standalone limit cannot both exist - the second is refused with
#: ``insufficient qty available``, measured against a 97.38-share position, so this is not
#: a fractional-only constraint. The stop is the one that bounds a loss, so the stop is the
#: one that is armed; the target is evaluated by the loop on completed bars.
LEGS = (STOP_LEG,)

#: Both suffixes, because a leg is still *recognised* after it stopped being *armed*: the
#: account can hold a limit leg from a session that ran before the ruling, and one this
#: module could not identify would be a protective order nothing cancels and a fill nothing
#: explains. Recognising more than we write is the safe direction.
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
#: Evidence that the loop reached a moment, beside the lock that says whose loop it is.
#: **Written on every cycle, idle ones included**, which is the half ``BOOK_FILE`` cannot
#: cover: the book is persisted only inside a session, so a loop idling correctly overnight
#: left no trace at all and the dashboard, ageing it from the book's mtime, called a
#: healthy loop NOT RESPONDING at the next open (found 23 Sep 2026).
HEARTBEAT_FILE = "heartbeat.json"
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
    settled: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()
    divergences: tuple[str, ...] = ()
    rearmed: tuple[str, ...] = ()
    adopted: tuple[str, ...] = ()
    flattened: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    submissions: tuple[Submission, ...] = ()
    trades: tuple[Trade, ...] = ()
    failed_step: str | None = None
    error: str | None = None
    unreachable: bool = False
    """The cycle failed because the outside world did not answer, not because of a defect.

    Worth separating in the report: an unreachable broker costs one poll and is expected
    on a domestic connection, while a ``ValueError`` in the same slot is a bug that has
    been swallowed by the same ``except``.
    """

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
    config_drifted: bool = False
    """Whether the settings file had changed since this process started.

    On the report rather than only in the log, because the log is where it is easy to
    miss and the report is what the gate reads.
    """
    closed_out: tuple[str, ...] = ()
    """Symbols the **stop close-out** flattened, which no cycle can report.

    `close_out_on_stop` runs in `run_session`'s `finally`, after the cycle list is closed,
    so its work is invisible to `sum(len(c.flattened) for c in cycles)`. On 25 Aug 2026 a
    rehearsal cancelled a stop and sold the position on shutdown and the summary said
    `positions flattened : 0` - the gate's own evidence contradicting the run it describes,
    on the very line that answers rehearsal condition 3.
    """
    close_out_failed: tuple[str, ...] = ()
    """Symbols whose close-out sell did **not** reach the broker. Named, never counted."""
    held_at_exit: tuple[str, ...] = ()
    """Symbols the book still holds when the process ends.

    **`open orders at exit : 0` is not the same statement and was read as though it were.**
    A failed close-out cancels the protective legs and then fails to sell, leaving zero
    working orders and a real position - the cleanest-looking line in the summary
    describing the worst state the loop can end in. The two facts are printed together
    because only the pair is an answer.
    """

    held_at_open: tuple[str, ...] = ()
    """Symbols the book already held when this session started.

    **The GATE 2 log has to say how the overnight residual was handled**, and it cannot
    say that from the cycle records alone: a position re-armed in cycle 1 looks identical
    to one opened in cycle 1. This is the session's own answer to *did we carry risk
    across a close, and for how long was it unprotected* (ruled 23 Aug 2026).
    """

    def summary(self) -> str:
        failed = [cycle for cycle in self.cycles if not cycle.ok]
        lines = [
            f"session {self.session_id} stopped by {self.stopped_by}",
            f"  cycles completed     : {len(self.cycles)}, {len(failed)} with a failed step",
            f"  decisions recorded   : {sum(len(c.decisions) for c in self.cycles)}",
            (
                "  bars already decided : "
                f"{sum(len(c.settled) for c in self.cycles)} symbol-cycles"
                " not re-decided"
            ),
            f"  orders submitted     : {sum(len(c.submissions) for c in self.cycles)}",
            f"  trades emitted       : {sum(len(c.trades) for c in self.cycles)}",
            f"  positions re-armed   : {sum(len(c.rearmed) for c in self.cycles)}",
            (
                "  configuration        : "
                + (
                    "DRIFTED - the settings file changed after this process started; the "
                    "running configuration is not the one on disk"
                    if self.config_drifted
                    else "matches the file on disk"
                )
            ),
            (
                "  overnight residual   : "
                + (
                    f"{len(self.held_at_open)} position(s) unprotected from the open"
                    f" until first arming - {', '.join(self.held_at_open)}"
                    if self.held_at_open
                    else "none; the session opened flat"
                )
            ),
            f"  positions adopted    : {sum(len(c.adopted) for c in self.cycles)}",
            (
                "  cycles skipped       : "
                f"{sum(1 for c in self.cycles if c.unreachable)}"
                " on an unreachable broker or feed"
            ),
            (
                "  positions flattened  : "
                f"{sum(len(c.flattened) for c in self.cycles)} in-cycle (rule 3)"
                + (
                    ""
                    if not self.closed_out
                    else f", {len(self.closed_out)} by the stop close-out"
                    f" - {', '.join(self.closed_out)}"
                )
            ),
        ]
        for cycle in failed:
            lines.append(
                f"  FAILED {cycle.cycle_id} at {cycle.failed_step}: {cycle.error}"
            )
        lines.append(
            f"  open orders at exit  : {len(self.open_orders)}"
            + ("" if not self.open_orders else " — " + "; ".join(self.open_orders))
        )
        # Printed beside the orders, never instead of them: zero working orders and a
        # position still held is the state a reader is most likely to mistake for a clean
        # exit, and it is exactly the state a failed close-out produces.
        lines.append(
            "  positions at exit    : "
            + (
                "none; the book is empty"
                if not self.held_at_exit
                else f"{len(self.held_at_exit)} STILL HELD - {', '.join(self.held_at_exit)}"
            )
        )
        for symbol in self.close_out_failed:
            lines.append(
                f"  CLOSE-OUT FAILED     : {symbol} is still held and its protective legs "
                "were cancelled before the sell was refused. Close it by hand"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class Rehearsal:
    """A deliberately permissive run that proves the execution path, and nothing else.

    **GATE 2 criterion 2, as amended on 20 Aug 2026.** The criterion asks whether a
    loop-produced decision becomes an order, fills, is reconciled, adopted and protected
    against a real broker. That is a question about the machine. It was conflated with a
    question about the model - *does the deployed band trade?* - which already has an
    answer, correctly no, and which no amount of engineering can change. A rehearsal
    separates them.

    Four conditions, and each closes a way this could become a lie:

    1. **The band is stated** in the banner and the provenance of every record it writes,
       which is never ``live``. A rehearsal that looked live in the log would be worse
       than no rehearsal.
    2. **Nothing it produces may reach a metric.** ``records.is_reportable`` says so once,
       and the suite asserts it rather than trusting each caller to remember.
    3. **It closes out before the session close.** The protective legs are DAY orders, so
       a position held overnight is unprotected overnight - the condition Ben set on this
       gate. A rehearsal that holds overnight fails the rehearsal.
    4. **It is small.** The point is that the path works, not that it was consequential.
    """

    reason: str
    notional: float
    close_out_minutes: int

    @property
    def provenance(self) -> str:
        return records.rehearsal_provenance(self.reason)


@dataclass(frozen=True)
class Decision:
    """The one decision a symbol has at a given completed bar.

    Cached because the ruling of 19 Aug 2026 says a decision is made once per completed
    bar. It is kept for the whole session rather than for one cycle because two later
    cycles need it: the exit pass, which re-sends an obligation the broker has not yet
    taken, and the guard that stops the loop deciding the same bar again.
    """

    bar: pd.Timestamp
    signal: Signal
    decision_id: str


@dataclass
class LiveState:
    """Everything the loop carries between cycles. Persisted where a restart needs it."""

    cfg: Config
    broker: Broker
    predictor: Predictor
    thresholds: Thresholds
    state_dir: Path
    language: str = EN
    provenance: str = records.LIVE
    fetch: Callable[[], dict[str, pd.DataFrame]] | None = None
    """Where the bars come from. ``None`` is the live fetch.

    Injected so **GB-38's replay drives this same cycle** rather than a copy of it. A
    replay that reimplemented the loop would prove the reimplementation works, which is
    the one thing nobody needs to know.
    """
    rehearsal: Rehearsal | None = None
    book: Book = field(default_factory=Book)
    decided: dict[str, Decision] = field(default_factory=dict)
    """The latest decision per symbol. Seeded from the log so a restart does not re-decide."""
    seeded: bool = False
    entry_fills: dict[str, list] = field(default_factory=dict)
    arming_failures: dict[str, int] = field(default_factory=dict)
    seen_orders: set[str] = field(default_factory=set)
    cycles: int = 0

    def save(self) -> None:
        self.book.save(self.state_dir / BOOK_FILE)
        (self.state_dir / ENTRY_FILLS_FILE).write_text(
            json.dumps(self.entry_fills, indent=2), encoding="utf-8"
        )


def decision_id_for(as_of: pd.Timestamp, symbol: str) -> str:
    """A decision's name, and the broker's idempotency key.

    Derived from **the bar, not the cycle**. Under the ruling of 19 Aug 2026 a decision
    happens once per completed daily bar, so the cycle that happened to carry it is an
    accident of polling and has no business in its name - the cycle is still on every log
    line, where tracing needs it.

    The change buys a guard the loop could not otherwise have. Measured against the paper
    account on 18 Aug 2026: Alpaca refuses a repeated ``client_order_id`` with
    ``40010001 client_order_id must be unique``. Because ``execute`` passes this string
    straight through as the entry's ``client_order_id``, a second entry for the same bar
    is refused **at the broker** - which is the only place that can win a race against a
    fill that has not yet appeared.
    """
    return f"{pd.Timestamp(as_of):%Y%m%d}-{symbol}"


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


def permissive_band() -> Thresholds:
    """The band a rehearsal runs on. **Not calibrated, and it says so everywhere.**

    :data:`PERMISSIVE_LOWER` is the smallest band ``Thresholds`` admits — it refuses zero,
    which is the contract having teeth — so any upward forecast at all qualifies. It could
    not be mistaken for the output of a grid search, and that is the point: a rehearsal
    band that looked like a plausible calibration would invite somebody to read its
    results as results, and the whole of the amended criterion is that this run tests the
    machine and says nothing about the model.
    """
    return Thresholds(lower=PERMISSIVE_LOWER)


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


def overnight_residual(book: Book, when: pd.Timestamp) -> list[str]:
    """What a session that opens holding a position must say about protection.

    **The ruling of 23 Aug is that the legs stay `DAY` and the residual is reported**, so
    the residual has to appear somewhere a reader will see it. A GATE 2 log that says how
    the overnight gap was handled is worth more than a footnote in a document, and a
    footnote is what this becomes if nothing prints it.

    Args:
        book: The book as loaded from disk, before this session's first reconciliation.
        when: Session start, UTC.

    Returns:
        Banner lines, empty when the book holds nothing - a session that starts flat has
        no residual and should not print a warning about one.

    The interval is stated rather than estimated: DAY orders are expired by the broker at
    the close, so an open position carried **no broker-side protection** from that close
    until the first cycle of this session arms it. Rule 2 of the GB-26 policy makes that
    cycle the first thing this session does, which bounds the gap for every session the
    loop runs and does nothing for one it misses.
    """
    held = sorted(book.managed)
    if not held:
        return []
    opened = ", ".join(
        f"{symbol} (decision {book.managed[symbol].decision_id})" for symbol in held
    )
    return [
        "  " + "!" * 74,
        f"  OVERNIGHT RESIDUAL : {len(held)} position(s) held into this session",
        f"  positions          : {opened}",
        (
            "  protection         : NONE between the previous close and this session's"
            " first arming"
        ),
        (
            "  why                : the protective legs are TimeInForce.DAY and the"
            " broker expires them at the close (ruled 23 Aug 2026; GTC is refused on a"
            " fractional quantity and flattening would break train/live parity)"
        ),
        (
            "  handling           : rule 2 re-arms every open position before any entry,"
            " and this session's first cycle is that moment"
        ),
        "  " + "!" * 74,
    ]


def config_drift(cfg: Config) -> list[str]:
    """Banner lines when the configuration on disk is no longer the one running.

    **A long-running process reads its configuration once, which makes the configuration
    mutable during the run.** On 23 Aug 2026 the settings file was edited at 17:00 while a
    grid started at 16:36 was still running; the results file it wrote carries a hash
    matching nothing on disk, and nothing told anyone. **A three-day live run makes that
    worse, not better**: every decision record it writes is the audit trail GATE 2 and
    §7 both read, and a record stamped with a configuration that no longer exists cannot
    be replayed against the settings that produced it.

    **It does not reload.** A live loop that silently changed its own behaviour mid-run
    would be worse than one that is stale and says so - a position could be opened under
    one risk policy and managed under another, with nothing in the log marking the seam.
    The remedy is a restart the operator chooses, not one the loop takes.

    Args:
        cfg: The configuration this process is running under, loaded at startup.

    Returns:
        Banner lines, empty when the file on disk still matches. A failure to read the
        file is itself reported rather than swallowed: a settings file that has become
        unreadable during a run is not evidence that nothing changed.
    """
    try:
        on_disk = load_config()
    except (FileNotFoundError, ValueError) as failure:
        return [
            "  " + "!" * 74,
            "  CONFIG DRIFT      : the settings file cannot be read while this process runs",
            f"  error             : {failure}",
            "  running with      : the configuration loaded at startup, unchanged",
            "  " + "!" * 74,
        ]

    running, current = config_hash(cfg), config_hash(on_disk)
    if running == current:
        return []

    models_moved = model_config_hash(cfg) != model_config_hash(on_disk)
    return [
        "  " + "!" * 74,
        "  CONFIG DRIFT      : the settings file has changed since this process started",
        f"  running           : {running}",
        f"  on disk           : {current}",
        (
            "  model settings    : CHANGED - the running model is not the one this "
            "configuration would build, and every record from here on is stamped with a "
            "configuration that does not describe it"
            if models_moved
            else "  model settings    : unchanged; the difference is outside the sections "
            "that shape a model"
        ),
        (
            "  action            : NOT reloaded. A loop that changed its own behaviour "
            "mid-run would be worse than one that is stale and says so. Restart the loop "
            "to adopt the new settings"
        ),
        "  " + "!" * 74,
    ]


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
            *(
                []
                if state.rehearsal is None
                else [
                    "  " + "!" * 74,
                    f"  REHEARSAL          : {state.rehearsal.reason}",
                    "  band               : DELIBERATELY PERMISSIVE - not the calibrated one",
                    f"  provenance         : {state.rehearsal.provenance} (never 'live')",
                    f"  size cap           : {state.rehearsal.notional:,.2f} notional",
                    (
                        f"  close-out          : {state.rehearsal.close_out_minutes}"
                        " minutes before the close; nothing is carried overnight"
                    ),
                    (
                        "  reportable         : NO. No metric, table or figure may"
                        " include a decision from this run"
                    ),
                    "  " + "!" * 74,
                ]
            ),
            *(
                []
                if not dry_run
                else [
                    "  " + "!" * 74,
                    "  DRY RUN            : broker writes are refused and logged",
                    (
                        "  provenance         : 'live' - a KNOWN DEFECT. This run does"
                        " NOT stamp its records"
                    ),
                    (
                        "  reportable         : records written here are"
                        " INDISTINGUISHABLE from real ones and"
                    ),
                    (
                        "                       `records.is_reportable` admits them."
                        " Do not analyse or report"
                    ),
                    (
                        "                       anything this run records. See"
                        " DECISIONS.md, 2026-08-24."
                    ),
                    "  " + "!" * 74,
                ]
            ),
            *config_drift(cfg),
            *overnight_residual(state.book, when),
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

    **Decide once per completed bar; manage every cycle.** (Ruling, 19 Aug 2026.) An entry
    decision is a function of daily data, so it can change once a day - at 60-second
    polling the loop would otherwise re-derive the identical verdict 390 times and record
    390 identical decisions per symbol per session. Reconciliation, protection, trade
    emission and exits are explicitly outside the rule and run on every cycle, because
    prices move intraday and the risk already taken moves with them.

    The rule is the generalisation of the first-cycle ruling below: **a cycle may always
    reduce risk; adding it is what is rationed.**

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
        # **A rehearsal emits no trades at all.** `Trade` carries no provenance - it is a
        # frozen contract (spec 4.2) and GB-19 reads it without translation - so the only
        # place a rehearsal can be kept out of the study's trade log is here, by never
        # putting one in. Condition 2 of the rehearsal, enforced at the single point that
        # can enforce it.
        trades = (
            ()
            if state.rehearsal is not None
            else tuple(
                records.emit_trades(
                    fresh,
                    state.book,
                    {k: _entry_fill(v) for k, v in state.entry_fills.items()},
                )
            )
        )
        for trade in trades:
            log(
                f"trade {trade.symbol} {trade.exit_reason} size={trade.size:.9f} "
                f"net={trade.net_pnl:+.2f} costs={trade.costs:.4f}"
            )
        # **To disk, not only to the log** (28 Aug 2026). GB-29 called this module's
        # counterpart "the live trade log" and there was no log: these objects were built,
        # written to the session log as text, and dropped. The two live trades this system
        # has made existed only as lines somebody would have to grep, and no panel or
        # metric could read them. A rehearsal reaches here with an empty tuple, so the
        # condition-2 gate above is still the single point that keeps rehearsal trades out.
        records.save_trades(trades, state.state_dir)
        state.seen_orders |= {order.id for order in orders}

        log("step 3/10 reconcile: the broker is the truth")
        result = reconcile(state.broker, state.book)
        state.book = result.book
        for symbol in list(state.entry_fills):
            if symbol not in state.book.managed:
                state.entry_fills.pop(symbol)
        divergences = tuple(str(divergence) for divergence in result.divergences)

        adopted = adopt_own_positions(state, orders, log)

        log(
            "step 4/10 protection: verify the stop, re-arm, flatten on a second failure"
        )
        rearmed, flattened = protect_book(state, orders, log)
        state.save()

        log("step 5/10 data: completed daily bars for the universe")
        bars = (
            state.fetch()
            if state.fetch is not None
            else load_live_bars(
                list(state.cfg.universe),
                min_history_bars(state.cfg),
                requirement=history_requirement(state.cfg),
                attempts=state.cfg.live.retry_attempts,
                backoff=state.cfg.live.retry_backoff_seconds,
            )
        )

        log("step 6/10 features: drop the in-progress bar, then the keystone builder")
        frames: dict[str, pd.DataFrame] = {}
        # The completed OHLCV bars beside the feature frames, because they answer a
        # different question: the builder's output is one column per channel and carries no
        # price at all, and the loop-side target is a comparison against a high.
        completed_bars: dict[str, pd.DataFrame] = {}
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
            completed_bars[symbol] = completed

        _seed_decided(state, when)

        if not frames:
            log("every symbol is stale; no forecast this cycle. Protection stands.")
            return CycleReport(
                cycle_id=cycle_id,
                at=when,
                entries_allowed=entries_allowed,
                stale=tuple(stale),
                divergences=divergences,
                rearmed=tuple(rearmed),
                adopted=tuple(adopted),
                flattened=tuple(flattened),
                # An exit decided at an earlier cycle is an obligation, not a forecast, so
                # it is still owed even on a cycle that can compute nothing new.
                submissions=tuple(_send_exits(state, orders, log)),
                trades=trades,
            )

        bars_at = {symbol: frame.index[-1] for symbol, frame in frames.items()}
        undecided = [
            symbol
            for symbol in sorted(frames)
            if state.decided.get(symbol) is None
            or state.decided[symbol].bar != bars_at[symbol]
        ]
        settled = tuple(symbol for symbol in sorted(frames) if symbol not in undecided)
        if settled:
            log(
                "already decided at this bar, not re-decided: "
                + ", ".join(f"{s} at {bars_at[s]:%Y-%m-%d}" for s in settled)
            )

        target_exits = _send_target_exits(state, completed_bars, orders, log)

        log(
            f"step 7/10 forecast: {len(undecided)} undecided, {len(settled)} settled, "
            f"stale {stale or 'none'}"
        )
        forecasts: dict[str, Forecast] = {}
        for symbol in undecided:
            forecasts[symbol] = predict_window(
                state.predictor, frames[symbol], state.cfg, symbol, bars_at[symbol]
            )

        closing_out = _rehearsal_close_out(state, when, log)

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
            if entries_allowed and not closing_out
            else []
        )
        if state.rehearsal is not None:
            orders_to_place = [
                _shrink_to_rehearsal_size(order, state.rehearsal)
                for order in orders_to_place
            ]
        if not entries_allowed:
            log(
                "first cycle: entries are held back until protection has been verified "
                "once. Exits, protection and reconciliation ran in full"
            )

        log("step 9/10 explain, narrate, execute, persist — bars not yet decided")
        decisions: list[str] = []
        submissions: list[Submission] = []
        sized = {order.symbol: order for order in orders_to_place}
        for symbol in sorted(signals):
            decision_id = decision_id_for(bars_at[symbol], symbol)
            verdict = signals[symbol]

            if not entries_allowed and verdict.action == ENTER_LONG:
                # Where the two rulings meet. The first cycle may not add risk (GB-26),
                # and a bar is decided only once (19 Aug 2026) - so recording this verdict
                # now would strand it: the next cycle would decline to re-decide and the
                # entry it justifies would never be submitted, turning "the first cycle
                # does not enter" into "the session never enters".
                #
                # A decision is recorded when it is complete. A verdict the loop is
                # forbidden to act on is not yet a decision it has made, so the bar is
                # left undecided and the next cycle records it once, with the order it
                # produced and a narrative written in the knowledge of that order. The
                # cost is one repeated forward pass on the first two cycles of a session,
                # not on all 390.
                log(
                    f"{symbol}: ENTER_LONG withheld and the bar left undecided; the "
                    "first cycle may not add risk, and the next cycle records it once"
                )
                continue

            forecast = forecasts[symbol]
            # **GB-53's spectral view was built, tested, rendered by the dashboard, and
            # never populated** (found 28 Aug 2026 while taking the screenshots the
            # architecture report is blocked on). `explain_spectral` had no caller
            # anywhere in `glassbox/`, so every decision record carried
            # `per_frequency=None` and the panel could not draw from a real record under
            # any model. Structural, not nominal: a model that cannot enumerate its own
            # frequency maps takes the channel view unchanged, so this is inert for
            # DLinear and persistence rather than branching on a name.
            explain_with = (
                explain_spectral
                if isinstance(state.predictor.model, Spectral)
                else attribute
            )
            found = explain_with(
                state.predictor.model,
                _window_of(state, frames[symbol], symbol, forecast.as_of),
                state.cfg.channels.active_channels,
                # The forecast this explains is in raw log returns (GB-16 restores it), so
                # the explanation must be too, or the dashboard renders contributions in a
                # unit the number above them is not in.
                scale=state.predictor.stats_for(symbol).scale_for(TARGET_CHANNEL),
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

            record = DecisionRecord(
                as_of=forecast.as_of,
                symbol=symbol,
                forecast=forecast,
                attribution=found,
                signal=verdict,
                order=None if order is None else _order_dict(order, decision_id),
                narrative=story.text,
                config_hash=config_hash(state.cfg),
                provenance=state.provenance,
            )

            # **The decision is on disk before the order is at the broker, always.**
            # GB-39's restart rule, and the ordering is the whole of it: if the process
            # dies between the two, a filled position exists whose decision_id names a
            # record that `adopt_own_positions` can find, so the next start books and
            # protects it. Written the other way round, a crash in the same instant would
            # leave a position at the broker with nothing on disk that explains it -
            # unadoptable by construction, quarantined for ever, and never protected.
            records.save_decision(record, state.state_dir)
            state.decided[symbol] = Decision(
                bar=bars_at[symbol], signal=verdict, decision_id=decision_id
            )
            decisions.append(decision_id)

            submission = None
            if order is not None:
                submission = execute(state.broker, order, decision_id, state.cfg)
                if submission.status == PENDING_APPROVAL:
                    # GB-37: the recommendation outlives the cycle, because the approver
                    # is a different process and a recommendation nobody can answer is
                    # not a recommendation.
                    records.save_pending(
                        state.state_dir,
                        {
                            "decision_id": decision_id,
                            "as_of": _iso(forecast.as_of),
                            "symbol": symbol,
                            "shares": order.shares,
                            "price": order.price,
                            "notional": order.notional,
                            "stop_loss": order.stop_loss,
                            "take_profit": order.take_profit,
                            "narrative": story.text,
                            "provenance": state.provenance,
                            "record": records.encode_decision(record),
                        },
                    )
                    log(f"{symbol}: queued for approval as {decision_id}")
                else:
                    _absorb_entry(state, order, submission, when, log)
            if submission is not None:
                submissions.append(submission)

        submissions.extend(target_exits)
        submissions.extend(_send_exits(state, orders, log))

        log("step 10/10 persist: book and entry fills")
        state.save()
        return CycleReport(
            cycle_id=cycle_id,
            at=when,
            entries_allowed=entries_allowed,
            ranked_over=tuple(sorted(frames)),
            settled=settled,
            stale=tuple(stale),
            divergences=divergences,
            rearmed=tuple(rearmed),
            adopted=tuple(adopted),
            flattened=tuple(flattened),
            decisions=tuple(decisions),
            submissions=tuple(submissions),
            trades=trades,
        )

    except Unavailable as failure:
        # GB-39: the outside world did not answer. Loud, skipped, and the session
        # continues - a dropped connection costs one poll, and crashing on it would cost
        # every position its protection for the rest of the day.
        LOGGER.error(
            "[%s] CYCLE SKIPPED - the broker or the data feed is unreachable: %s. "
            "Nothing was decided and nothing was submitted; the next poll will try "
            "again. Protective legs already at the broker are unaffected by this",
            cycle_id,
            failure,
        )
        return CycleReport(
            cycle_id=cycle_id,
            at=when,
            entries_allowed=entries_allowed,
            failed_step="Unavailable",
            error=str(failure),
            unreachable=True,
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


def _rehearsal_close_out(state: LiveState, when: pd.Timestamp, log) -> bool:
    """Flatten a rehearsal's positions before the close, and stop it opening more.

    **Condition 3 of the rehearsal, and it is Ben's GATE 2 condition applied where it
    bites.** The protective legs are ``TimeInForce.DAY`` and expire at the close, so a
    position carried overnight is unprotected overnight - which the backtest models as
    protected. A rehearsal exists to prove the execution path, and holding overnight
    proves something nobody asked about while taking a risk nobody sized. *A rehearsal
    that holds overnight fails the rehearsal.*

    Returns:
        Whether the loop is in the close-out window, in which case no entry is sized.
    """
    if state.rehearsal is None:
        return False
    _, closes = market_session(state.cfg, when)
    if when < closes - pd.Timedelta(minutes=state.rehearsal.close_out_minutes):
        return False

    if state.book.managed:
        LOGGER.warning(
            "REHEARSAL CLOSE-OUT: %s minutes to the close at %s. Flattening %s so nothing "
            "is carried overnight - the protective legs are DAY orders and expire at the "
            "close, so an overnight hold would be an unprotected hold",
            state.rehearsal.close_out_minutes,
            f"{closes:%H:%M}",
            ", ".join(sorted(state.book.managed)),
        )
    for symbol, holding in sorted(state.book.managed.items()):
        _flatten_for_close(state, symbol, holding, log)
    log("rehearsal close-out window: no entry will be sized")
    return True


def release_protective_legs(state: LiveState, symbol: str) -> tuple[str, ...]:
    """Cancel every live protective leg on ``symbol``, and name the ones cancelled.

    **A sell cannot be submitted while another sell holds the quantity**, and the broker
    holds the *whole* position for any working sell order. So every path that closes a
    position has to release the legs first, and a path that does not is not a slower way
    to close - it is a way that cannot close at all.

    Extracted on 25 Aug 2026, when the rehearsal showed the two paths had diverged.
    ``_flatten_for_close`` cancelled first and worked; ``_flatten`` - rule 3's remedy for a
    position that has lost its protection - sold directly and failed on every attempt, so
    the escape hatch was unreachable under exactly the condition it exists to escape. Two
    implementations of one operation differing in the step that makes it work is the
    two-places defect at its purest, so this is the one implementation and both call it.
    """
    cancelled = []
    for order in state.broker.get_orders():
        if (
            order.symbol == symbol
            and order.side == SELL
            and _leg_of(order)
            and order.status not in FINISHED
        ):
            state.broker.cancel_order(order.id)
            cancelled.append(order.id)
    return tuple(cancelled)


def _flatten_for_close(state: LiveState, symbol: str, holding: Holding, log) -> bool:
    """Cancel the stop and sell at market. Idempotent through the client_order_id.

    "Both legs" until the ruling of 26 Aug 2026, which left the stop as the only
    protective order at the broker; ``release_protective_legs`` still cancels a
    pre-ruling limit leg if the account holds one.

    Returns whether the sell reached the broker. The caller needs to know: a close-out
    that failed leaves a real position behind, and treating it as done would put the
    reassuring line in the log for the one case that needs the alarming one.

    **`release_protective_legs` is inside the `try`, and on 27 Aug 2026 it was not.** It
    reads `get_orders` and calls `cancel_order` - two network calls - so an unreachable
    broker raised `Unavailable` from the line *before* the guarded one. That escaped
    `close_out_on_stop`, escaped `run_session`'s `finally`, and left `main` with a
    traceback instead of a session report. Every other broker path in this loop treats
    unreachability as a skipped cycle; this one treated it as fatal, and it is the path
    that runs when nobody is watching.
    """
    try:
        release_protective_legs(state, symbol)
        sold = state.broker.submit_market_order(
            symbol=symbol,
            quantity=holding.quantity,
            side=SELL,
            client_order_id=f"{holding.decision_id}-closeout",
        )
        log(f"{symbol}: rehearsal close-out submitted id={sold.id}")
        return True
    except Exception as failure:  # noqa: BLE001 - retried on the next poll
        LOGGER.error(
            "RISK EVENT: the rehearsal close-out for %s failed (%s). It is retried on "
            "the next poll; if the session ends first the position is carried overnight "
            "UNPROTECTED and must be closed by hand",
            symbol,
            failure,
        )
        return False


def close_out_on_stop(
    state: LiveState, when: pd.Timestamp, log=lambda message: None
) -> tuple[str, ...]:
    """Flatten a rehearsal's positions on **any** stop. Returns the symbols that failed.

    **Condition 3 had a hole and the GATE 2 plan walked straight into it** (24 Aug 2026).
    :func:`_rehearsal_close_out` fires only once the clock reaches
    ``close_out_minutes`` before the exchange close. The gate plan stops the rehearsal at
    the *open* - prove the execution path, stop, restart under the deployed band - roughly
    five hours earlier, and a rehearsal stopped there was flattened by nothing at all. The
    position survived in the saved book, its protective legs were DAY orders due to expire
    at the close, and the deployed session would have restarted holding something a
    rehearsal opened. *A rehearsal that holds overnight fails the rehearsal* has to mean
    **any** stop, not only the one the clock reaches on its own.

    **A no-op unless a rehearsal is active, and that is not an optimisation.** Flattening
    on stop is right for a rehearsal and wrong for the deployed loop: the backtest holds
    overnight, so a live loop that flattened whenever it stopped would be running a
    different strategy from the one being evaluated - closing a protection gap by opening a
    **parity** gap, which is what the 23 Aug DAY/GTC ruling refused to do.

    Only symbols whose sell reached the broker leave the book. One that did not is left in
    place and named in the return value, because a position the book has forgotten is worse
    than one it still shows.
    """
    if state.rehearsal is None or not state.book.managed:
        return ()

    LOGGER.warning(
        "REHEARSAL CLOSE-OUT ON STOP: flattening %s before this run ends. The protective "
        "legs are DAY orders, so a rehearsal position left open would be unprotected the "
        "moment this process is gone",
        ", ".join(sorted(state.book.managed)),
    )
    failed: list[str] = []
    for symbol, holding in sorted(state.book.managed.items()):
        # Belt and braces around `_flatten_for_close`, which already catches. The point is
        # structural rather than defensive: this runs in a `finally`, it is the last thing
        # standing between a rehearsal and an unprotected overnight hold, and **anything**
        # it raises costs the session report as well as the flatten. One symbol that
        # cannot be closed must not stop the loop trying the next one.
        try:
            closed = _flatten_for_close(state, symbol, holding, log)
        except Exception as failure:  # noqa: BLE001 - reported, never fatal
            LOGGER.error(
                "RISK EVENT: the close-out for %s raised (%s). The position is still open "
                "at the broker and its DAY legs expire at the close",
                symbol,
                failure,
            )
            closed = False
        if closed:
            state.book.managed.pop(symbol, None)
        else:
            failed.append(symbol)
    if failed:
        LOGGER.error(
            "RISK EVENT: %s could not be flattened on stop and %s still open at the "
            "broker with DAY legs. Close by hand",
            ", ".join(failed),
            "is" if len(failed) == 1 else "are",
        )
    del when  # the stop time is not a condition; any stop flattens
    return tuple(failed)


def _shrink_to_rehearsal_size(order: risk.Order, rehearsal: Rehearsal) -> risk.Order:
    """Cap an order at the rehearsal notional. **Condition 4: it is small.**

    The sizer is left alone and its output is capped, rather than the rehearsal being
    given its own sizing path: the point of the rehearsal is that the ordinary path runs,
    and a second sizer would be one more thing that is not the thing being proven.
    """
    if order.notional <= rehearsal.notional:
        return order
    scale = rehearsal.notional / order.notional
    return replace(
        order,
        shares=order.shares * scale,
        notional=rehearsal.notional,
    )


def adopt_own_positions(
    state: LiveState, orders: Sequence[BrokerOrder], log
) -> list[str]:
    """Take back a position this system opened but did not finish booking. **GB-39.**

    **The hole this closes was a gate blocker.** ``engine/reconcile.py`` quarantines any
    position the book does not know about and never adopts one, and that ruling is right
    *at that layer*: the reconciler is read-only, it sees quantities rather than reasons,
    and a position it cannot explain is one it must not pretend to manage. But two ordinary
    events produce a position the loop genuinely did open and has not yet booked - the
    entry fill arriving after ``_absorb_entry``'s poll window expires, and the process
    dying between submission and fill. Quarantined, those are **never protected**, so the
    loop's own fills would sit at the broker with no stop and no target until a human
    noticed.

    **What makes adoption safe is evidence, not a guess.** The position is adopted only
    when all of this holds:

    1. the broker reports a **filled buy** for the symbol,
    2. whose ``client_order_id`` is a decision id this system mints - ``YYYYMMDD-SYMBOL``,
    3. naming a decision that is **on disk** in this deployment's own log, under this
       provenance,
    4. and that record carries the order it produced, with the stop and target that
       decision chose.

    Anything short of that stays quarantined. A position bought by hand in the Alpaca web
    console has no decision record and is not adoptable; nor is one from a different
    config, because the record is looked up in this store. So the ruling in ``reconcile``
    is not weakened - the system still refuses to manage what it cannot explain. This is
    the case where it **can** explain it, and the explanation is a record it wrote itself
    before it sent the order.

    The stop and target come from the record rather than from the broker, because the
    broker never knew them: the protective legs may not have been armed at all. The entry
    price comes from the **fill**, not the record, because the fill is what happened.
    """
    adopted: list[str] = []
    fills = {
        client_order_id(order): order
        for order in orders
        if order.side == BUY and order.status in records.FILLED
    }
    for symbol in sorted(state.book.unmanaged):
        for decision_id, entry in fills.items():
            if entry.symbol != symbol:
                continue
            stored = _decision_named(state, decision_id, symbol)
            if stored is None or not stored.order:
                continue
            quantity = state.book.unmanaged[symbol]
            filled_price = float(entry.filled_price or stored.order["price"])
            state.book.managed[symbol] = Holding(
                symbol=symbol,
                quantity=quantity,
                decision_id=decision_id,
                entry_price=filled_price,
                stop_loss=float(stored.order["stop_loss"]),
                take_profit=float(stored.order["take_profit"]),
            )
            state.book.unmanaged.pop(symbol)
            state.entry_fills.setdefault(
                symbol,
                [
                    _iso(_filled_at(entry)),
                    filled_price,
                    abs(filled_price - float(stored.order["price"])) * quantity,
                    entry.id,
                ],
            )
            adopted.append(symbol)
            LOGGER.warning(
                "ADOPTED %s: the broker holds %.9f that reconciliation quarantined, and "
                "%s is a decision in this log with the order it produced. Booked at the "
                "fill price %.4f with stop %.4f and target %.4f; protection is armed in "
                "this cycle. This is the loop's own fill catching up with its book, not a "
                "position of unknown origin",
                symbol,
                quantity,
                decision_id,
                filled_price,
                float(stored.order["stop_loss"]),
                float(stored.order["take_profit"]),
            )
            log(f"{symbol}: adopted from quarantine under {decision_id}")
            break
    return adopted


def _decision_named(
    state: LiveState, decision_id: str, symbol: str
) -> DecisionRecord | None:
    """The stored decision a ``client_order_id`` names, or ``None`` if it names none.

    The id is parsed rather than trusted: ``YYYYMMDD-SYMBOL`` is the only shape this system
    mints, so anything else is somebody else's order and the lookup is not even attempted.
    """
    day, _, tail = decision_id.partition("-")
    if tail != symbol or len(day) != 8 or not day.isdigit():
        return None
    try:
        as_of = pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:]}", tz="UTC")
    except ValueError:
        return None
    for stored in records.load_decisions(
        as_of, as_of, state.state_dir, state.provenance
    ):
        if (
            stored.symbol == symbol
            and decision_id_for(stored.as_of, symbol) == decision_id
        ):
            return stored
    return None


def _filled_at(order: BrokerOrder) -> pd.Timestamp:
    raw = order.raw
    when = (
        raw.get("filled_at")
        if isinstance(raw, dict)
        else getattr(raw, "filled_at", None)
    )
    return pd.Timestamp(when) if when is not None else pd.Timestamp.now(tz="UTC")


def _seed_decided(state: LiveState, when: pd.Timestamp) -> None:
    """Rebuild what has already been decided, once, from the log on disk.

    A process restarted mid-session starts with an empty cache. Without this it would
    re-decide bars it had already recorded, ``records.save_decision`` would refuse the
    duplicate - correctly - and the cycle would fail on that refusal every minute until
    the close. The month is the window because the month is the unit the store is already
    written in; only today matters, and a month is cheap to read.
    """
    if state.seeded:
        return
    state.seeded = True
    now = pd.Timestamp(when)
    month = (
        now.tz_convert("UTC") if now.tzinfo else now.tz_localize("UTC")
    ).normalize()
    try:
        stored = records.load_decisions(
            month.replace(day=1), now, state.state_dir, state.provenance
        )
    except records.DuplicateDecision as failure:
        raise LiveError(
            f"the decision log cannot be read, so the loop cannot tell what it has "
            f"already decided and will not guess: {failure}"
        ) from failure
    for record in stored:
        state.decided[record.symbol] = Decision(
            bar=record.as_of,
            signal=record.signal,
            decision_id=decision_id_for(record.as_of, record.symbol),
        )
    if stored:
        LOGGER.info(
            "seeded from the log: %s already decided, latest bar %s",
            ", ".join(sorted({record.symbol for record in stored})),
            f"{max(record.as_of for record in stored):%Y-%m-%d}",
        )


def _send_exits(
    state: LiveState, orders: Sequence[BrokerOrder], log
) -> list[Submission]:
    """Send the exits the bar decided, every cycle, until the broker has them.

    **Outside the decide-once rule**, by the ruling of 19 Aug 2026. An entry is a function
    of daily data and is decided once; an exit is an *obligation on capital already
    committed*, so a submission that failed is re-sent on the next poll rather than at the
    next bar. Nothing is re-decided here - the verdict is the one the bar produced - it is
    re-*sent*.

    **The broker is asked whether it already has the order**, not a flag in memory. The
    exit's ``client_order_id`` is a function of the bar, so one glance at the order
    history answers "did this exit go?" correctly across a restart, where a flag would
    have said no and sent a second market sell.
    """
    already = {records.client_order_id(order) for order in orders}
    submissions: list[Submission] = []
    for symbol in sorted(state.book.managed):
        decision = state.decided.get(symbol)
        if decision is None or decision.signal.action != EXIT:
            continue
        client_order_id = f"{decision.decision_id}-exit"
        if client_order_id in already:
            continue
        try:
            submission = exit_position(state, symbol, client_order_id, log)
        except Exception as failure:  # noqa: BLE001 - retried, never fatal
            LOGGER.error(
                "RISK EVENT: %s was decided EXIT at %s and the submission failed (%s). "
                "The position is still held and its protective legs are still live; the "
                "exit is re-sent on the next poll",
                symbol,
                f"{decision.bar:%Y-%m-%d}",
                failure,
            )
            continue
        if submission is not None:
            submissions.append(submission)
    return submissions


def _send_target_exits(
    state: LiveState,
    completed_bars: dict[str, pd.DataFrame],
    orders: Sequence[BrokerOrder],
    log,
) -> list[Submission]:
    """The target, evaluated by the loop, because it cannot be an order at the broker.

    **The second half of the ruling of 26 Aug 2026.** A working sell holds the whole
    position at Alpaca, so the one protective order is the stop; the take-profit is
    therefore not a resting limit but a level this function watches. When a **completed**
    daily bar's high has reached it, the position is closed at market - which in a session
    that polls from the open means the fill lands at the open, and that is exactly what
    `backtest.engine`'s ``target_in_loop`` arm models. The study's headline arm and the
    live path describe one system.

    **Completed bars only, and that is the whole of the causality argument here.** The
    in-progress bar is dropped upstream by ``drop_incomplete_bar``, so the high this reads
    is a fact about a session that has ended. Reading an intraday high would be the live
    path quietly acquiring information the backtest never had, and every number in the
    study would then describe something else.

    A stale symbol is skipped: its frame has no bar for the last completed session, so
    there is nothing to evaluate. It keeps its stop, per the stale-data ruling - the target
    is an opportunity and the stop is the risk control, and only one of them may depend on
    fresh data.
    """
    already = {records.client_order_id(order) for order in orders}
    submissions: list[Submission] = []
    for symbol in sorted(state.book.managed):
        bars = completed_bars.get(symbol)
        if bars is None or bars.empty:
            continue
        holding = state.book.managed[symbol]
        high = float(bars[HIGH].iloc[-1])
        if high < holding.take_profit:
            continue

        client_order_id = f"{holding.decision_id}{records.TARGET_IN_LOOP_SUFFIX}"
        if client_order_id in already:
            continue

        LOGGER.info(
            "%s: target %.4f reached on the completed bar %s (high %.4f). Closing at "
            "market - the target is the loop's, not the broker's",
            symbol,
            holding.take_profit,
            f"{bars.index[-1]:%Y-%m-%d}",
            high,
        )
        try:
            submission = exit_position(state, symbol, client_order_id, log)
        except Exception as failure:  # noqa: BLE001 - retried next cycle, never fatal
            LOGGER.error(
                "%s: the target exit failed to submit (%s). The position is still held "
                "and its stop is still live; this is re-sent on the next poll",
                symbol,
                failure,
            )
            continue
        if submission is not None:
            submissions.append(submission)
    return submissions


def answer_pending(
    cfg: Config,
    broker: Broker,
    state_dir: str | Path,
    decision_id: str,
    approved: bool,
) -> Submission:
    """Approve or decline a queued Co-Pilot recommendation. **GB-37.**

    **Both answers write a decision record.** A rejection that left no trace would make the
    log a record of what the system wanted rather than of what happened, and "the operator
    said no" is a decision — arguably the more interesting one, since it is the only place a
    human enters the loop.

    Approving submits through ``executor.approve``, which re-derives the quantity from the
    order's notional and price exactly as ``execute`` does, and arms protection on the fill.
    Declining touches no broker at all, which is GB-37's acceptance criterion.

    Raises:
        LiveError: no recommendation with that id is queued. Answering twice is the
            realistic way that happens - two dashboard tabs, one click each - and the
            second answer must not resubmit.
    """
    root = Path(state_dir)
    queued = [
        entry
        for entry in records.load_pending(root)
        if entry["decision_id"] == decision_id
    ]
    if not queued:
        raise LiveError(
            f"no recommendation queued as {decision_id!r}; it was answered already, or "
            "this is a stale view of the queue"
        )
    entry = queued[0]
    order = risk.Order(
        symbol=entry["symbol"],
        shares=float(entry["shares"]),
        price=float(entry["price"]),
        notional=float(entry["notional"]),
        stop_loss=float(entry["stop_loss"]),
        take_profit=float(entry["take_profit"]),
    )

    if approved:
        submission = approve(broker, order, decision_id, cfg)
        answer = f"Approved by the operator. {submission.status}: " + (
            f"entry {submission.entry.id}"
            if submission.entry is not None
            else str(submission.reason)
        )
    else:
        submission = decline(order, decision_id)
        answer = "Declined by the operator. No order was sent to the broker."

    original = records.decode_decision(entry["record"])
    # Amended, not appended: the loop already recorded this decision when it queued the
    # recommendation, and the operator's answer belongs to that decision rather than
    # being a second one at the same bar.
    records.amend_decision(
        replace(
            original,
            narrative=f"{original.narrative} {answer}",
            order=None if not approved else original.order,
        ),
        root,
    )
    records.resolve_pending(root, decision_id)
    LOGGER.info("[%s] %s", decision_id, answer)
    return submission


# ── protection: the five rules, implemented ──────────────────────────────────


def protect_book(
    state: LiveState, orders: Sequence[BrokerOrder], log
) -> tuple[list[str], list[str]]:
    """Verify the stop of every managed position, arm it if missing, flatten on a second
    consecutive failure, and cancel any protective order left after a stop has filled.

    Rules 2, 3 and 4 of the protection policy (DECISIONS, 2026-08-18), **as rewritten
    on 26 Aug 2026**: until then this verified two legs, a stop and a limit, and that
    policy was superseded because a working sell holds the whole position at Alpaca -
    ``LEGS`` is ``(STOP_LEG,)``. The sibling cancellation survives only for a limit leg
    armed before the ruling; under the current policy there is no sibling to cancel.
    Reconciliation
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
            # A protective order filled, so the position is gone or going. Nothing should
            # remain, but a limit leg armed before the ruling of 26 Aug 2026 can still be
            # live - and a sell order outliving the position it protected is a naked sell.
            # Cancelled here and the cancellation verified, rather than assumed from the
            # fill. Under the current policy `legs` is empty and this loop does nothing.
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
        attempt = _next_arming(orders, holding.decision_id)
        try:
            for leg in missing:
                _arm_leg(state, symbol, holding, leg, log, attempt)
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


def _next_arming(orders: Sequence[BrokerOrder], decision_id: str) -> int:
    """The next unused arming number for a position's protective legs.

    **Measured against the paper account, 18 Aug 2026: Alpaca refuses a
    ``client_order_id`` it has seen before, including after the original was cancelled or
    expired** (``40010001 client_order_id must be unique``). The legs are submitted
    ``TimeInForce.DAY``, so they expire at every close; re-arming them the next morning
    under yesterday's id would be refused, that refusal counts as an arming failure, and
    :data:`ARMING_STRIKES` of them flatten the position at market. A position held
    overnight would have been liquidated two cycles into the next session for a reason
    that has nothing to do with the strategy.

    The number is derived from **the broker's own order history** rather than from a
    counter in memory, so a restart cannot reset it into a collision.
    """
    used = 0
    for order in orders:
        client_id = records.client_order_id(order)
        if not client_id.startswith(f"{decision_id}#"):
            continue
        number = client_id[len(decision_id) + 1 :].split("-")[0]
        if number.isdigit():
            used = max(used, int(number))
    return used + 1


def _arm_leg(
    state: LiveState, symbol: str, holding: Holding, leg: str, log, attempt: int = 1
) -> None:
    if leg != STOP_LEG:
        # Not a defensive nicety: until 26 Aug 2026 this function armed a limit here, and
        # the refusal it earned counted as an arming failure - two of which flatten the
        # position. A caller that reintroduces the second leg should fail here, loudly,
        # rather than have `insufficient qty available` liquidate a healthy position.
        raise LiveError(
            f"{leg!r} is not armed at the broker: a working sell holds the whole "
            "position, so the stop is the only protective order (ruled 26 Aug 2026)"
        )
    client_order_id = f"{holding.decision_id}#{attempt}{LEG_SUFFIX[leg]}"
    order = state.broker.submit_stop_order(
        symbol=symbol,
        quantity=holding.quantity,
        stop_price=holding.stop_loss,
        client_order_id=client_order_id,
    )
    log(f"{symbol}: stop re-armed id={order.id} at {holding.stop_loss:.4f}")


def _flatten(state: LiveState, symbol: str, holding: Holding, log) -> None:
    """Close at market. An unprotected position is worse than a closed one (rule 3)."""
    LOGGER.error(
        "FLATTENING %s at market: arming failed %s times in succession and an unprotected "
        "position is worse than a closed one",
        symbol,
        ARMING_STRIKES,
    )
    # The legs first, or this sell cannot be placed at all: whichever leg is still live
    # holds the entire quantity, so a market sell for that quantity is refused with
    # `insufficient qty available`. Measured on 25 Aug 2026 - 13 consecutive cycles tried
    # to flatten NVDA and every one failed here, while the close-out path, which cancels
    # first, closed the same position on its first attempt.
    released = release_protective_legs(state, symbol)
    if released:
        log(f"{symbol}: cancelled {len(released)} protective leg(s) before flattening")
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
    be two policies wearing one name. The stop stays live meanwhile, so a
    recommended-but-unapproved exit is a position that is still protected. (Until 26 Aug
    2026 this read "the stop and the target"; since that ruling the target is not at the
    broker but evaluated by the loop on completed bars.)
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
        # Measured 18 Aug 2026: a liquid market order filled 8 ms after submission, so
        # 10 x 1.5s is generous for the ordinary case and useless for anything that
        # queues. What follows when it does expire is the point:
        #
        #   the order stays working, the position is NOT in the book, and the next cycle
        #   cannot resubmit - the bar is decided and the client_order_id is taken. When
        #   the fill lands, reconciliation quarantines the position and
        #   `adopt_own_positions` takes it back in the same cycle, before protection.
        #
        # So the exposure is bounded by one poll interval, not by a human noticing.
        LOGGER.warning(
            "%s: entry %s is still working after %s polls. The position is not yet in the "
            "book; when it fills, reconciliation will quarantine it and adoption will "
            "take it back on that cycle and arm its stop. No resubmission is possible - "
            "the bar is decided and %s is already used at the broker",
            order.symbol,
            entry.id,
            FILL_POLL_ATTEMPTS,
            submission.decision_id,
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
    rehearsal: Rehearsal | None = None,
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
        rehearsal: Run with a **deliberately permissive band** to prove the execution
            path, per GATE 2 criterion 2 as amended on 20 Aug 2026. The calibrated band on
            disk is ignored and :func:`permissive_band` is used instead; every record
            carries the rehearsal's provenance and none is reportable. See
            :class:`Rehearsal` for the four conditions this run must satisfy.
    """
    root = Path(state_dir)
    if dry_run and root.resolve() == Path(DEFAULT_STATE_DIR).resolve():
        raise LiveError(
            "a dry run may not use the deployed state directory "
            f"({DEFAULT_STATE_DIR}). Pass --state-dir pointing at a copy. "
            "Two reasons, and the first one bit on 24 Aug 2026. A dry run alongside a "
            "live loop means two processes writing the same book and decision log, so "
            "the verification step would corrupt the session it exists to verify. And "
            "`--dry-run` refuses broker writes but does NOT change provenance: a "
            "decision it records is written as `live` and is indistinguishable from a "
            "real one, which `records.is_reportable` would then admit into the study."
        )
    clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
    predictor = load_predictor(root / CHECKPOINT_DIR, cfg)
    thresholds = (
        permissive_band()
        if rehearsal is not None
        else load_thresholds(root / THRESHOLDS_FILE)
    )

    # GB-39. Outermost so the dry run's refusals are not retried, and so every call the
    # loop makes - reads and writes alike - goes through one policy rather than each call
    # site remembering to.
    live_broker = RetryingBroker(broker if broker is not None else AlpacaBroker(), cfg)
    if dry_run:
        live_broker = DryRunBroker(live_broker)

    state = LiveState(
        cfg=cfg,
        broker=live_broker,
        predictor=predictor,
        thresholds=thresholds,
        state_dir=root,
        language=language,
        provenance=(records.LIVE if rehearsal is None else rehearsal.provenance),
        rehearsal=rehearsal,
        book=Book.load(root / BOOK_FILE),
        entry_fills=_load_entry_fills(root / ENTRY_FILLS_FILE),
    )

    started = clock()
    # Read before the first cycle reconciles, which is the only moment the book still
    # describes what was carried **into** the session rather than what it holds now.
    held_at_open = tuple(sorted(state.book.managed))
    drifted = bool(config_drift(cfg))
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
                    stopping["reason"] = CLOSED
                    break
                LOGGER.info(
                    "no session at %s (%s): the exchange is shut or the configured window "
                    "has not opened. Nothing is fetched",
                    now,
                    cfg.data.calendar,
                )
                stopping["reason"] = "outside the session"
                break
            # Before the work, not after it: a cycle that fails, or one wedged inside an
            # unbounded socket read, still proves the loop was alive at this moment - and
            # its heartbeat then stops ageing, which is what NOT RESPONDING is for.
            write_heartbeat(state.state_dir, now, "in session")
            cycles.append(run_cycle(state, now))
            _log_run_of_failures(cycles)
            if stopping["reason"]:
                break
            sleep(cfg.live.poll_seconds)
    except KeyboardInterrupt:  # pragma: no cover - depends on delivery timing
        stopping["reason"] = "KeyboardInterrupt"
    finally:
        _restore_sigint(previous)
        # In the `finally`, so it runs on SIGINT, on Ctrl+C, on max_cycles and on an
        # unexpected exception alike - every way this run can end while holding something.
        # A no-op unless a rehearsal is active; the deployed loop must hold overnight or it
        # is not the strategy the backtest evaluated.
        before_close_out = set(state.book.managed)
        close_out_failed = close_out_on_stop(state, clock())
        closed_out = tuple(sorted(before_close_out - set(state.book.managed)))

    open_orders = _open_orders(state)
    state.save()
    return SessionReport(
        session_id=root.name,
        banner=banner,
        cycles=tuple(cycles),
        open_orders=open_orders,
        stopped_by=stopping["reason"] or CLOSED,
        held_at_open=held_at_open,
        config_drifted=drifted,
        closed_out=closed_out,
        close_out_failed=close_out_failed,
        held_at_exit=tuple(sorted(state.book.managed)),
    )


def idle_state(cfg: Config, when: pd.Timestamp) -> str:
    """Why the loop is not trading right now, in the words a log reader needs.

    "outside session" and "holiday" are different facts about the same silence, and a
    person reading a night of heartbeats needs to know which one they are looking at.
    """
    if market_session(cfg, when) is None:
        return "weekend" if when.dayofweek >= 5 else "holiday"
    return "outside session"


def write_heartbeat(state_dir: str | Path, when: pd.Timestamp, state: str) -> Path:
    """Leave evidence, in :data:`HEARTBEAT_FILE`, that the loop reached ``when``.

    **Written atomically.** A reader may arrive mid-write at any moment - the dashboard
    polls this file every few seconds - and a half-written JSON object is unparseable
    evidence, which the dashboard is obliged to read as NOT RESPONDING. A temporary file
    and ``os.replace`` mean a reader sees either the previous beat or this one, never
    half of either.

    The companion of the log line :func:`heartbeat` returns, and for the other audience: a
    person reads the log, a program stats the file.
    """
    root = Path(state_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"at": f"{when:%Y-%m-%dT%H:%M:%SZ}", "state": state}, indent=2)
    temporary = root / f"{HEARTBEAT_FILE}.{os.getpid()}.tmp"
    temporary.write_text(payload + "\n", encoding="utf-8")
    written = root / HEARTBEAT_FILE
    os.replace(temporary, written)
    return written


def heartbeat(
    cfg: Config, state_dir: str | Path, when: pd.Timestamp, started: pd.Timestamp
) -> str:
    """One line proving the process is alive while it has nothing to do.

    **A loop that died at 02:00 and a loop correctly idling produce identical output:
    nothing.** That is tolerable for a run that lasts one session and is not tolerable for
    one that spans three, because the overnight protection residual is bounded by rule 2
    only while the process is **alive** - and the banner that reports the residual has
    nothing to be true about if the process is not there to print it. With a heartbeat, a
    silent log means dead rather than quiet.

    Deliberately one line and deliberately not a cycle: no fetch, no broker call, no
    decision record. Idling for seventeen hours must not grow the decision store, and a
    cycle record every ``poll_seconds`` while the market is shut would do exactly that.
    """
    book = Book.load(Path(state_dir) / BOOK_FILE)
    held = ",".join(sorted(book.managed)) or "none"
    return (
        f"heartbeat {when:%Y-%m-%dT%H:%M:%SZ} state={idle_state(cfg, when)} "
        f"positions={held} uptime={_uptime(when - started)}"
    )


def _uptime(delta: pd.Timedelta) -> str:
    """``2d 03:14:00``. Days are separate because "51 hours" is not a readable number."""
    total = max(int(delta.total_seconds()), 0)
    days, rest = divmod(total, 86_400)
    hours, rest = divmod(rest, 3_600)
    minutes, seconds = divmod(rest, 60)
    return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"


def run_sessions(
    cfg: Config,
    state_dir: str | Path,
    *,
    sessions: int = 1,
    broker: Broker | None = None,
    dry_run: bool = False,
    language: str = EN,
    clock=None,
    sleep=time.sleep,
    max_cycles: int | None = None,
    rehearsal: Rehearsal | None = None,
) -> tuple[SessionReport, ...]:
    """Run ``sessions`` exchange sessions in one process, idling between them.

    Args:
        sessions: How many sessions to complete before returning. GATE 2 criterion 6 asks
            for two or three, and running them in one process is what removes the half of
            the overnight residual rule 2 cannot cover - a session the loop **misses**,
            because nobody started it.
        The rest are :func:`run_session`'s and are passed through unchanged.

    Returns:
        One :class:`SessionReport` per session, **in order and separate**. They are not
        merged: each carries its own ``held_at_open``, which is the answer to *did this
        session begin holding risk*, and merging them would destroy exactly that.

    **Only an ordinary close continues to the next session.** A run stopped by SIGINT, by
    ``max_cycles`` or by anything else returns what it has: a loop that stopped for a
    reason should not silently start again tomorrow.
    """
    clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
    started = clock()
    reports: list[SessionReport] = []
    last_beat: pd.Timestamp | None = None

    while len(reports) < sessions:
        now = clock()
        if in_session(cfg, now):
            report = run_session(
                cfg,
                state_dir,
                broker=broker,
                dry_run=dry_run,
                language=language,
                clock=clock,
                sleep=sleep,
                max_cycles=max_cycles,
                rehearsal=rehearsal,
            )
            if not report.cycles:
                # The session closed between our check and its. Not a session, and not a
                # reason to stop - keep idling and wait for the next one.
                sleep(cfg.live.poll_seconds)
                continue
            reports.append(report)
            LOGGER.info("session %d of %d complete", len(reports), sessions)
            if report.stopped_by != CLOSED:
                break
            continue

        # The file on every idle pass, the log line at the heartbeat interval. The log is
        # for a person reading a night of output, where one line every 15 minutes is
        # readable and one a minute is not; the file is for the dashboard, which asks how
        # long ago the loop was last alive and gets a worse answer the coarser this is.
        write_heartbeat(state_dir, now, idle_state(cfg, now))
        if (
            last_beat is None
            or (now - last_beat).total_seconds() >= cfg.live.heartbeat_seconds
        ):
            LOGGER.info(heartbeat(cfg, state_dir, now, started))
            last_beat = now
        sleep(cfg.live.poll_seconds)

    return tuple(reports)


def _log_run_of_failures(cycles: list[CycleReport]) -> None:
    """Say how long the outside world has been unreachable, loudly, once a cycle.

    Deliberately **not** a circuit breaker (out of GB-39's scope, and rightly): nothing
    here stops the loop or changes what it attempts. A run of failures is still worth
    counting out loud, because one skipped cycle and forty skipped cycles read identically
    in a log that only ever says "skipped", and the second is a session that has silently
    stopped trading.
    """
    run = 0
    for cycle in reversed(cycles):
        if cycle.ok:
            break
        run += 1
    if run >= 2:
        # **Says only what a failed cycle can know** (5 Oct 2026). This line used to
        # promise that every protective leg at the broker was still live. It said so for
        # two sessions while the broker held no working order at all: the stops are DAY
        # orders, they had expired at the previous close, and step 4, which re-arms them,
        # sits after the step that was failing. Nothing here reads the broker, and a
        # cycle does not record which step failed, so this can say neither that the
        # stops are live nor that step 4 was skipped - only that protection is unconfirmed.
        LOGGER.error(
            "%d cycles in a row have failed (%s). The loop is still polling, but nothing "
            "has been decided across those cycles and protection is NOT confirmed: a "
            "cycle that fails before step 4 verifies and re-arms no stop, and every stop "
            "from a previous session expired at its close. Check the broker for working "
            "stops",
            run,
            cycles[-1].error,
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
    client_id = records.client_order_id(order)
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


def _entry_fill(stored: Sequence) -> tuple[pd.Timestamp, float, float, str]:
    """A stored entry fill, typed as ``records.emit_trades`` reads it.

    **The one place the stored form becomes the typed one.** Both writers (`_absorb_entry`
    and `adopt_own_positions`) keep the time as an ISO string, because `entry_fills` is
    what `LiveState.save` serialises, so the value is a string in memory as well as on
    disk. Until 5 Oct 2026 nothing turned it back: the string reached ``Trade.entry_time``
    and the first trade the deployed loop ever emitted crashed step 2 in ``.isoformat()``,
    on every cycle of two sessions, with protection never reached.
    """
    entry_time, price, cost, order_id = stored
    return (
        pd.Timestamp(entry_time).tz_convert("UTC"),
        float(price),
        float(cost),
        order_id,
    )


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


class DailyLogFile(logging.FileHandler):
    """The session log on disk: one file per calendar day, appended to, never truncated.

    Three properties, and each closes a way the GATE 2 evidence could stop existing:

    - **Appending** (``mode="a"``), so a restart adds to the day's file rather than
      truncating it. The rehearsal plan is *rehearse, stop, restart under the deployed
      band*, and a truncating handler would delete the first half at the moment of the
      restart - where it would read as a session that never ran.
    - **One file per calendar day**, so a multi-session run lands in files a person can
      read one at a time.
    - **Re-targeted when the day turns**, not fixed at startup. ``--sessions 3`` idles
      through two midnights inside a single process, so a filename computed once would put
      all three sessions in the first day's file and make the other two dates lies. The
      check is per record, which is cheap beside the write it guards.
    """

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._day = self._today()
        super().__init__(self._path(self._day), mode="a", encoding="utf-8")

    @staticmethod
    def _today() -> str:
        """UTC, matching every other timestamp the loop computes (spec §4.1)."""
        return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")

    def _path(self, day: str) -> Path:
        return self._directory / f"live-{day}.log"

    def emit(self, record: logging.LogRecord) -> None:
        day = self._today()
        if day != self._day:
            self._day = day
            # Deliberately not self.close(): that marks the handler closed and
            # deregisters it, and this one has to keep serving the next day's records.
            if self.stream is not None:
                self.flush()
                self.stream.close()
            self.baseFilename = str(self._path(day).absolute())
            self.stream = self._open()
        super().emit(record)


# ── CLI ──────────────────────────────────────────────────────────────────────


def launch_mode(args: argparse.Namespace) -> str:
    """What kind of run this is, in the words the lock file and the refusal will use.

    The three are not interchangeable and the refusal has to say which, because the two
    that matter look identical in a process list: a deployed session and a rehearsal both
    read the same book and write the same ``pending.json``, and it is the rehearsal's
    permissive band that makes the pair dangerous rather than merely redundant.
    """
    if args.rehearsal is not None:
        # Through records rather than by concatenation: the mode a person reads in the
        # refusal is then literally the provenance those records would have carried.
        return records.rehearsal_provenance(args.rehearsal)
    return "dry-run" if args.dry_run else "deployed"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        log_file = DailyLogFile(args.log_dir)
    except OSError as failure:
        # Fatal on purpose. A rehearsal that runs and leaves no evidence is worse than one
        # that refuses to start, and refusing costs nothing here: nothing has traded yet.
        print(f"live_loop: cannot open the session log: {failure}", file=sys.stderr)
        return 2
    # Built once and given to both handlers rather than left to `basicConfig`, because the
    # UTC converter is an attribute of a formatter instance and there is no `basicConfig`
    # argument for it. Set on the instance, not on `logging.Formatter`, so importing this
    # module never changes how somebody else's logging prints.
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    formatter.converter = time.gmtime
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    log_file.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console, log_file],
        # `basicConfig` is a silent no-op when the root logger already has a handler, and
        # a silent no-op here means no session log at all - the exact failure this is
        # fixing, reintroduced by anything that touches logging before main does. As the
        # entry point, main owns the root logger rather than hoping to be first.
        force=True,
    )
    LOGGER.info("session log: %s", log_file.baseFilename)

    # Before the config, before the checkpoint, before anything that could take a second:
    # the whole point is that a launch which must not happen does not get far enough to
    # read a book it must not touch. Logged as well as printed, because "a second launch
    # was refused" is exactly the sort of thing a gate log should be able to show.
    try:
        holder = acquire(args.state_dir, mode=launch_mode(args))
    except LockRefused as refusal:
        LOGGER.error("%s", refusal)
        print(f"live_loop: {refusal}", file=sys.stderr)
        return EXIT_REFUSED
    LOGGER.info(
        "state directory %s locked by PID %d (%s)",
        args.state_dir,
        holder.pid,
        holder.mode,
    )

    try:
        reports = run_sessions(
            load_config(),
            args.state_dir,
            sessions=args.sessions,
            dry_run=args.dry_run,
            language=args.language,
            max_cycles=args.max_cycles,
            rehearsal=(
                None
                if args.rehearsal is None
                else Rehearsal(
                    reason=args.rehearsal,
                    notional=args.rehearsal_notional,
                    close_out_minutes=args.rehearsal_close_out,
                )
            ),
        )
    except (LiveError, FileNotFoundError, ValueError) as failure:
        print(f"live_loop: {failure}", file=sys.stderr)
        return 2
    finally:
        release(args.state_dir, holder)

    # Through the logger, not print, and line by line as the banner already is: the
    # summary IS the gate evidence - cycles completed, decisions recorded, open orders at
    # exit - and a summary that only ever reached stdout would be the one thing missing
    # from the file written to preserve it.
    for report in reports:
        for line in report.summary().splitlines():
            LOGGER.info(line)

    # **A failed close-out must produce a record, not a traceback - and not a green exit
    # either.** The report above already names the symbols; this makes the process status
    # say so too, so a supervisor, a cron wrapper or a person reading `echo $?` learns
    # that positions were left open without having to parse the log.
    unclosed = sorted({s for report in reports for s in report.close_out_failed})
    if unclosed:
        LOGGER.error(
            "RISK EVENT: this run ended without flattening %s. %s open at the broker with "
            "no protective order once the DAY legs expire at the close, and no process is "
            "managing %s. Close by hand",
            ", ".join(unclosed),
            "It is" if len(unclosed) == 1 else "They are",
            "it" if len(unclosed) == 1 else "them",
        )
        return EXIT_CLOSE_OUT_FAILED
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
        default=DEFAULT_STATE_DIR,
        help="where the checkpoint, band, book and decision log live",
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help=(
            "where the session log is written, as one live-YYYY-MM-DD.log per calendar "
            "day, appended to across restarts"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run every step except order submission",
    )
    parser.add_argument("--language", choices=("en", "he"), default="en")
    parser.add_argument(
        "--sessions",
        type=int,
        default=1,
        help=(
            "how many exchange sessions to run in this process, idling between them. "
            "GATE 2 criterion 6 asks for two or three, and one process across all of "
            "them removes the half of the overnight residual rule 2 cannot cover: a "
            "session nobody was there to start"
        ),
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="stop after this many cycles (default: run to the close)",
    )
    parser.add_argument(
        "--rehearsal",
        metavar="REASON",
        default=None,
        help=(
            "prove the execution path with a DELIBERATELY PERMISSIVE band (GATE 2 "
            "criterion 2). The reason is required and goes into the provenance of every "
            "record, which is never 'live'. Nothing this run produces is reportable"
        ),
    )
    parser.add_argument(
        "--rehearsal-notional",
        type=float,
        default=REHEARSAL_NOTIONAL,
        help=(
            "cap on a rehearsal order, in account currency. Small on purpose: the point "
            f"is that the path works (default: {REHEARSAL_NOTIONAL:,.0f})"
        ),
    )
    parser.add_argument(
        "--rehearsal-close-out",
        type=int,
        metavar="MINUTES",
        default=REHEARSAL_CLOSE_OUT_MINUTES,
        help=(
            "minutes before the exchange close at which a rehearsal flattens. The "
            "protective legs are DAY orders, so an overnight hold is an unprotected hold "
            f"(default: {REHEARSAL_CLOSE_OUT_MINUTES})"
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI, not by tests
    raise SystemExit(main())


__all__ = [
    "ARMING_STRIKES",
    "CLOSED",
    "DEFAULT_STATE_DIR",
    "HEARTBEAT_FILE",
    "PERMISSIVE_LOWER",
    "REHEARSAL_CLOSE_OUT_MINUTES",
    "REHEARSAL_NOTIONAL",
    "CycleReport",
    "DailyLogFile",
    "DryRunBroker",
    "LiveError",
    "LiveState",
    "Rehearsal",
    "SessionReport",
    "adopt_own_positions",
    "answer_pending",
    "close_out_on_stop",
    "config_drift",
    "drop_incomplete_bar",
    "heartbeat",
    "idle_state",
    "in_session",
    "is_stale",
    "last_completed_session",
    "load_thresholds",
    "main",
    "market_session",
    "overnight_residual",
    "permissive_band",
    "release_protective_legs",
    "run_cycle",
    "run_session",
    "run_sessions",
    "session_banner",
    "write_heartbeat",
]
