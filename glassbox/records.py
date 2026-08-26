"""Decision records on disk, and the live trade log GB-19 reads without translation.

Two jobs that belong together because both answer "what did the system actually do?".

**Records.** Every :class:`DecisionRecord` is one line of JSONL under
``decisions/YYYY-MM.jsonl``, carrying the ``config_hash`` so a record ties to the exact
settings that produced it. Monthly files rather than one growing file: a session appends,
a reader loads a range, and a month is small enough to open in an editor when something
looks wrong at 16:31.

**Trades.** A live exit happens *at the broker*. The stop fills, the position vanishes from
``get_positions``, and ``engine/reconcile.py`` drops it - correctly, because it is read-only
and a filled stop looks exactly like a missing position from there. **Nothing else records
the exit**, so without this module there is no live trade log at all and GB-19's metrics
cannot run over paper results. The report is supposed to show backtest and live over the
same period; that is structurally impossible until something builds the ``Trade``.

:func:`emit_trades` builds them from the broker's **filled order history**, matching fills
to the book's managed holdings. Four rules, each from GB-29's scope:

- ``exit_reason`` comes from **which order filled** - the stop leg, the limit leg, or a
  signal exit - in the backtester's own vocabulary, ``stop_gap`` and ``target_gap``
  included where the fill price shows the level was gapped through.
- ``costs`` come from the **actual fill prices the broker reports**, never from the
  configured bps. This is the one place the live system can measure what the backtest
  assumes, and the difference is a reportable result: GB-57 states realised slippage
  against the 2 bps the study modelled.
- **No trade is emitted for a quarantined position.** The system did not open it, has no
  entry basis for it, and inventing one would put a number with no provenance into the
  study's own trade log.
- A trade is emitted **on the closing fill**, because that is when the round trip exists.

**Why a top-level module.** It is imported by the live loop and by the dashboard, and it
must never reach the validation harness - so it sits beside ``live_loop.py`` and is named in
the same forbidden-import contract. That is also why ``Trade`` moved to
``contracts/schemas.py``: a shared type could not live in ``backtest/``.

Implemented in GB-29.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from glassbox.contracts.schemas import (
    Attribution,
    DecisionRecord,
    Forecast,
    Signal,
    Trade,
)
from glassbox.engine.executor import BrokerOrder, client_order_id
from glassbox.engine.reconcile import Book, Holding

LOGGER = logging.getLogger(__name__)

MONTH_FORMAT = "%Y-%m"

# Exit reasons, in the backtester's vocabulary. Not imported from it: this module may not
# reach `backtest`, and repeating four strings is the lesser evil against breaking the
# contract that keeps the harness out of the live path. The parity is asserted in
# `tests/test_records.py` against the backtester's own constants, so a rename there fails
# here rather than silently producing a log GB-19 reads differently.
STOP = "stop"
STOP_GAP = "stop_gap"
TARGET = "target"
TARGET_GAP = "target_gap"
#: The only target exit the live path can take (ruled 26 Aug 2026): noticed on a completed
#: daily bar and filled at the next open, because Alpaca refuses every multi-leg order
#: class on a fractional quantity and the one broker-side protective order is the stop.
#: Distinct from `TARGET` and `TARGET_GAP`, which describe a broker-side limit filling
#: intraday - a different fill at a different moment, and a reader counting "target exits"
#: must be able to tell which system produced them.
TARGET_IN_LOOP = "target_in_loop"
SIGNAL = "signal"

# The suffix `executor.protect` gives the stop, which is how a fill is attributed.
STOP_SUFFIX = "-stop"
#: **No longer armed.** Kept because the account can still hold a limit leg from a session
#: that ran before the ruling of 26 Aug 2026, and a fill this module cannot attribute is a
#: trade the report explains wrongly. Recognised, never written.
TARGET_SUFFIX = "-target"
#: The loop's own target exit: a market sell, so the suffix is what separates it from a
#: signal exit. It does not end with `TARGET_SUFFIX`, so the two never collide.
TARGET_IN_LOOP_SUFFIX = "-target-in-loop"

FILLED = frozenset({"filled", "partially_filled"})
SELL = "sell"

# Where a decision came from (GB-38). `LIVE` is the default on `DecisionRecord` and the
# default filter here, and the direction of that default is the safety property: forgetting
# to filter hides replayed decisions rather than passing them off as real ones.
LIVE = "live"
REPLAY_PREFIX = "replay:fold-"

# GB-40, ruled 20 Aug 2026. A rehearsal is the live loop, against the real broker, driven
# by a deliberately permissive band so the EXECUTION PATH can be proven when the deployed
# band stands aside. Its decisions are real in every sense except the one that matters for
# a result: the band that produced them was not the one validation selected. So they carry
# a provenance of their own and **never** `LIVE` - which means the default filter hides
# them exactly as it hides a replay, and no metric can pick one up by forgetting to look.
REHEARSAL_PREFIX = "rehearsal:"

ANY_PROVENANCE = "*"

PENDING_FILE = "pending.json"


class DuplicateDecision(ValueError):
    """A second decision record for a bar that has already been decided.

    Raised on the way in by :func:`save_decision` and on the way out by
    :func:`load_decisions`, because a duplicate that reaches the store by a route neither
    of them owns - a hand edit, a crashed process re-run, a second loop pointed at the
    same directory - must still surface rather than silently double every metric read
    from it.
    """


def replay_provenance(fold: int) -> str:
    """The provenance string a replayed decision carries. Names the fold, not a boolean."""
    return f"{REPLAY_PREFIX}{fold}"


def is_replay(provenance: str) -> bool:
    return provenance.startswith(REPLAY_PREFIX)


def rehearsal_provenance(reason: str) -> str:
    """The provenance a rehearsal's decisions carry. Names the reason, not a boolean.

    The same shape as :func:`replay_provenance` and for the same argument: a reader who
    finds one of these in a log should learn *why* it is not live from the string itself,
    rather than having to know that a flag somewhere meant something.
    """
    if not reason.strip():
        raise ValueError("a rehearsal must state its reason; it goes in every record")
    return f"{REHEARSAL_PREFIX}{reason.strip()}"


def is_rehearsal(provenance: str) -> bool:
    return provenance.startswith(REHEARSAL_PREFIX)


def is_reportable(provenance: str) -> bool:
    """Whether a decision may reach a metric, a table or a figure in the report.

    **Only live decisions are.** A replay is a recorded day played back; a rehearsal is a
    real order placed under a band nobody selected. Both are evidence that the machine
    works and neither is evidence about the strategy, so both are excluded here rather
    than at each of the places that would otherwise have to remember.
    """
    return provenance == LIVE


# ── decision records ─────────────────────────────────────────────────────────


def decision_key(record: DecisionRecord) -> tuple[str, str, str, str]:
    """What makes a decision unique: the bar, the symbol, the config, and the run.

    A decision is a function of one completed daily bar under one configuration, so
    ``(as_of, symbol, config_hash)`` names it - that is the ruling of 19 Aug 2026, and the
    duplication it forbids is the loop re-deciding the same bar on every 60-second poll.

    **``provenance`` is a fourth element, and it widens the key rather than narrowing it.**
    Live trading and a replay of fold 13 are two runs of the world; a record from each,
    landing on the same bar under the same config, is two facts rather than one fact
    written twice. Leaving provenance out would refuse the second as a duplicate, which is
    a false refusal in the one direction that loses data. Everything the ruling is aimed
    at - the 390 identical re-decisions a session would otherwise record - shares a
    provenance, so the widening costs nothing it was meant to catch.
    """
    return (_iso(record.as_of), record.symbol, record.config_hash, record.provenance)


def save_decision(record: DecisionRecord, root: str | Path) -> Path:
    """Append one record to ``root/decisions/YYYY-MM.jsonl``. Returns the file written.

    Raises:
        DuplicateDecision: this bar has already been decided for this symbol under this
            config. The loop keeps its own guard, so this should be unreachable - which is
            exactly how a doubled metric gets written, and why the store checks too.
    """
    path = month_file(root, pd.Timestamp(record.as_of))
    path.parent.mkdir(parents=True, exist_ok=True)
    key = decision_key(record)
    if key in _keys_in(path):
        raise DuplicateDecision(
            f"{path} already holds a decision for {key[1]} at {key[0]} under config "
            f"{key[2][:12]} ({key[3]}). A decision is a function of one completed daily "
            "bar, so a second record for that bar is the same decision written twice and "
            "would double every count, mean and rate taken over this log"
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_encode(record), separators=(",", ":")) + "\n")
    return path


def amend_decision(record: DecisionRecord, root: str | Path) -> Path:
    """Replace the stored record for this decision. The one writer here that is not an append.

    **An operator's answer is not a second decision.** GB-37 records a Co-Pilot
    recommendation when the loop makes it, and again with the answer attached when a human
    approves or rejects it - and under the ruling of 19 Aug 2026 those are one decision at
    one bar, so the second write updates the first rather than adding a row. Written as an
    append it would have doubled exactly the count the ruling exists to protect, and the
    approval rate computed over the log would have been the operator's answers divided by
    twice the recommendations.

    The rewrite goes through a temporary file and one atomic replace, because losing a
    month of decisions to a process killed halfway through a rewrite would be a far worse
    failure than the duplicate this prevents.

    Raises:
        KeyError: no record with this decision's key is stored. Amending something that
            was never recorded means the caller is working from a stale view, and writing
            it as a new record would hide that.
    """
    path = month_file(root, pd.Timestamp(record.as_of))
    key = decision_key(record)
    lines, replaced = [], False
    for stored in _read_month(path):
        if decision_key(stored) == key:
            lines.append(json.dumps(_encode(record), separators=(",", ":")))
            replaced = True
        else:
            lines.append(json.dumps(_encode(stored), separators=(",", ":")))
    if not replaced:
        raise KeyError(
            f"{path} holds no decision for {key[1]} at {key[0]} under config "
            f"{key[2][:12]} ({key[3]}), so there is nothing to amend"
        )
    temporary = path.with_suffix(".jsonl.amending")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _keys_in(path: Path) -> set[tuple[str, str, str, str]]:
    """Every decision key already in one month file."""
    return {decision_key(record) for record in _read_month(path)}


def load_decisions(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    root: str | Path,
    provenance: str = LIVE,
) -> list[DecisionRecord]:
    """Every record with ``start <= as_of <= end`` and this provenance, chronologically.

    **``provenance`` defaults to ``LIVE``, and that default is the safety property.**
    Replayed decisions share the store with real ones — two stores would mean two readers,
    and GB-29 already rejected duplication-for-safety when it moved ``Trade`` into the
    contract rather than keeping one type per side. The risk a single store carries is a
    filtering bug, and defaulting to live answers it in the only direction that is safe: a
    caller who forgets the filter loses sight of replayed decisions, and can never be shown
    one as though it were live. Pass :data:`ANY_PROVENANCE` to see everything, or a
    specific ``"replay:fold-13"`` to see one replay.

    **Uniqueness is asserted on load**, over every record in the month files the range
    touches rather than only over the ones the filters keep - a duplicate hiding two days
    outside the window is still a duplicate, and a reader who cannot see it will trust the
    numbers it corrupts. This raises where :func:`_read_month` merely logs, and the
    difference is deliberate: a malformed line is what a process killed mid-write leaves
    behind and skipping it costs one record, while a duplicate is a defect in whatever
    wrote it and costs the correctness of everything computed downstream.

    Reads only the month files the range touches, so a year of decisions does not have to
    be parsed to answer a question about one week.

    Bounds are **inclusive** and are read as UTC: a naive timestamp is assumed UTC rather
    than compared against an aware one, which raises. A bound given as a plain date - the
    obvious way to ask for "August" - extends to the **end** of that day, because
    ``load_decisions("2026-08-01", "2026-08-31")`` meaning "everything except the 31st"
    would be a trap rather than a convention.
    """
    first = _as_utc(start)
    last = _as_utc(end, end_of_day=True)
    seen: set[tuple[str, str, str, str]] = set()
    records: list[DecisionRecord] = []
    for month in _months_between(first, last):
        path = month_file(root, month)
        for record in _read_month(path):
            key = decision_key(record)
            if key in seen:
                raise DuplicateDecision(
                    f"{path} holds more than one decision for {key[1]} at {key[0]} under "
                    f"config {key[2][:12]} ({key[3]}). Every count, mean and rate taken "
                    "over this log is wrong by an unknown factor until the extra records "
                    "are removed"
                )
            seen.add(key)
            if first <= record.as_of <= last and (
                provenance == ANY_PROVENANCE or record.provenance == provenance
            ):
                records.append(record)
    return sorted(records, key=lambda record: (record.as_of, record.symbol))


def save_pending(root: str | Path, entry: dict[str, Any]) -> Path:
    """Queue one Co-Pilot recommendation for a human to approve or reject. **GB-37.**

    Persisted rather than held in the loop's memory, because the approver is a different
    process — the dashboard — and because a recommendation that vanished when the loop
    restarted would be a recommendation nobody ever answered.

    **No expiry.** A recommendation waits until it is answered. An expiry timer would mean
    the system silently deciding "no" on the operator's behalf and recording nothing, which
    is exactly the kind of unrecorded decision this project exists to remove.
    """
    path = Path(root) / PENDING_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    queue = load_pending(root)
    queue = [item for item in queue if item["decision_id"] != entry["decision_id"]]
    queue.append(entry)
    path.write_text(json.dumps(queue, indent=2), encoding="utf-8")
    return path


def load_pending(root: str | Path) -> list[dict[str, Any]]:
    """Every recommendation still awaiting an answer, oldest first."""
    path = Path(root) / PENDING_FILE
    if not path.is_file():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")))


def resolve_pending(root: str | Path, decision_id: str) -> list[dict[str, Any]]:
    """Take one recommendation off the queue once it has been answered."""
    path = Path(root) / PENDING_FILE
    queue = [item for item in load_pending(root) if item["decision_id"] != decision_id]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(queue, indent=2), encoding="utf-8")
    return queue


def month_file(root: str | Path, when: pd.Timestamp) -> Path:
    """The file a record with this timestamp belongs in."""
    return (
        Path(root) / "decisions" / f"{pd.Timestamp(when).strftime(MONTH_FORMAT)}.jsonl"
    )


def _months_between(first: pd.Timestamp, last: pd.Timestamp) -> Iterator[pd.Timestamp]:
    month = first.normalize().replace(day=1)
    while month <= last:
        yield month
        month = (month + pd.Timedelta(days=32)).replace(day=1)


def _read_month(path: Path) -> Iterator[DecisionRecord]:
    if not path.is_file():
        return
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            yield _decode(json.loads(line))
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            # One malformed line must not cost the rest of the month. It is logged loudly
            # and skipped: a session killed mid-write leaves exactly this, and refusing the
            # whole file would lose every decision that did land.
            LOGGER.error("records: %s line %d is unreadable: %s", path, number, error)


def encode_decision(record: DecisionRecord) -> dict[str, Any]:
    """The JSON form of a record. Public because GB-37's pending queue carries one whole.

    A queued recommendation has to survive a restart with everything a decision record
    needs — the forecast, the attribution, the narrative — because the answer is written by
    a different process and must produce a record indistinguishable in shape from the one
    the loop wrote.
    """
    return _encode(record)


def decode_decision(raw: dict[str, Any]) -> DecisionRecord:
    """The inverse of :func:`encode_decision`."""
    return _decode(raw)


def _encode(record: DecisionRecord) -> dict[str, Any]:
    attribution = record.attribution
    return {
        "as_of": _iso(record.as_of),
        "symbol": record.symbol,
        "forecast": {
            "path": [float(value) for value in record.forecast.path],
            "symbol": record.forecast.symbol,
            "as_of": _iso(record.forecast.as_of),
        },
        "attribution": {
            "per_channel": {
                key: float(value) for key, value in attribution.per_channel.items()
            },
            "per_lag": (
                None
                if attribution.per_lag is None
                else [[float(v) for v in row] for row in attribution.per_lag]
            ),
            "per_frequency": (
                None
                if attribution.per_frequency is None
                else {
                    str(key): float(value)
                    for key, value in attribution.per_frequency.items()
                }
            ),
            "gain_phase": (
                None
                if attribution.gain_phase is None
                else {
                    str(key): [float(pair[0]), float(pair[1])]
                    for key, pair in attribution.gain_phase.items()
                }
            ),
            "per_level": (
                None
                if attribution.per_level is None
                else {
                    str(key): float(value)
                    for key, value in attribution.per_level.items()
                }
            ),
            "forecast_total": float(attribution.forecast_total),
        },
        "signal": asdict(record.signal),
        "order": record.order,
        "narrative": record.narrative,
        "config_hash": record.config_hash,
        "provenance": record.provenance,
    }


def _decode(raw: dict[str, Any]) -> DecisionRecord:
    attribution = raw["attribution"]
    return DecisionRecord(
        as_of=pd.Timestamp(raw["as_of"]),
        symbol=raw["symbol"],
        forecast=Forecast(
            path=np.array(raw["forecast"]["path"], dtype="float32"),
            symbol=raw["forecast"]["symbol"],
            as_of=pd.Timestamp(raw["forecast"]["as_of"]),
        ),
        attribution=Attribution(
            per_channel=dict(attribution["per_channel"]),
            per_lag=(
                None
                if attribution["per_lag"] is None
                else np.array(attribution["per_lag"], dtype="float64")
            ),
            per_frequency=(
                None
                if attribution["per_frequency"] is None
                else {
                    float(key): value
                    for key, value in attribution["per_frequency"].items()
                }
            ),
            gain_phase=(
                None
                if attribution["gain_phase"] is None
                else {
                    float(key): (pair[0], pair[1])
                    for key, pair in attribution["gain_phase"].items()
                }
            ),
            # ``.get`` rather than ``[...]``, and the asymmetry is deliberate. Every
            # other key has existed since the log's first line; ``per_level`` was added on
            # 24 Aug 2026 **while a multi-day GATE 2 run was writing records**, so records
            # without it exist and are correct. Indexing would turn every one of them into
            # a ``KeyError`` at read time - the dashboard, replay and the gate log all
            # decode this file. A field added after a log has started is a field that must
            # read as absent, not as broken.
            per_level=(
                None
                if attribution.get("per_level") is None
                else {
                    str(key): float(value)
                    for key, value in attribution["per_level"].items()
                }
            ),
            forecast_total=attribution["forecast_total"],
        ),
        signal=Signal(**raw["signal"]),
        order=raw["order"],
        narrative=raw["narrative"],
        config_hash=raw["config_hash"],
        # Absent in records written before GB-38. They were live, so that is what they
        # decode as - the default is a fact about those files, not a fallback.
        provenance=raw.get("provenance", LIVE),
    )


def _as_utc(value: pd.Timestamp | str, end_of_day: bool = False) -> pd.Timestamp:
    """A bound, in UTC. A bare date used as an upper bound covers its whole day."""
    stamp = pd.Timestamp(value)
    stamp = (
        stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    )
    if end_of_day and stamp == stamp.normalize():
        stamp = stamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return stamp


def _iso(when: pd.Timestamp) -> str:
    """ISO-8601 in UTC. A naive timestamp is **assumed** UTC rather than guessed at."""
    stamp = pd.Timestamp(when)
    stamp = (
        stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    )
    return stamp.isoformat()


# ── the live trade log ───────────────────────────────────────────────────────


def emit_trades(
    orders: Sequence[BrokerOrder],
    book: Book,
    entry_fills: dict[str, tuple[pd.Timestamp, float, float, str]],
) -> list[Trade]:
    """Build ``Trade`` records from filled broker orders. One per closing fill.

    Args:
        orders: Orders from the broker, filled since the last cycle. Only a **sell** closes
            a position; a buy's fill is what ``entry_fills`` carries.
        book: The reconciled book. **Only managed holdings produce trades** - a quarantined
            position has no entry basis and gets none.
        entry_fills: ``{symbol: (entry_time, entry_price, entry_cost, entry_order_id)}``,
            recorded by the live loop when the position was opened. Kept by the caller
            rather than re-derived here: the broker's order history is paged, and an entry
            can age out of it while its position is still open.

    Returns:
        Trades in exit-time order. ``costs`` is what the broker actually charged, never the
        configured bps.
    """
    trades: list[Trade] = []
    for order in sorted(orders, key=_filled_at):
        if order.side != SELL or order.status not in FILLED:
            continue
        if order.symbol not in book.managed:
            LOGGER.warning(
                "records: a sell filled for %s, which is not a managed holding; no trade "
                "emitted - the system has no entry basis for a position it did not open",
                order.symbol,
            )
            continue
        if order.symbol not in entry_fills:
            LOGGER.warning(
                "records: no recorded entry fill for %s; no trade emitted", order.symbol
            )
            continue

        holding = book.managed[order.symbol]
        entry_time, entry_price, entry_cost, entry_id = entry_fills[order.symbol]
        exit_price = float(order.filled_price or 0.0)
        size = float(order.filled_quantity)
        gross = (exit_price - entry_price) * size
        costs = entry_cost + _exit_cost(order, holding, exit_price, size)

        trades.append(
            Trade(
                symbol=order.symbol,
                entry_time=entry_time,
                exit_time=_filled_at(order),
                size=size,
                entry_price=entry_price,
                exit_price=exit_price,
                gross_pnl=gross,
                costs=costs,
                net_pnl=gross - costs,
                exit_reason=exit_reason_for(
                    order, holding.stop_loss, holding.take_profit
                ),
                strategy_exit=True,
                entry_order_id=entry_id,
                exit_order_id=order.id,
            )
        )
    return trades


def exit_reason_for(order: BrokerOrder, stop_loss: float, take_profit: float) -> str:
    """Which leg filled, in the backtester's vocabulary, gaps included.

    The leg is identified by the ``client_order_id`` suffix ``executor.protect`` gave it,
    **not** by guessing from the price - two levels can sit close together and a price-only
    guess would mislabel a trade the report then explains wrongly.

    The **gap** variants are then decided by price, which is the only thing that can decide
    them: a stop that filled *below* its level was gapped through. The difference between
    ``stop`` and ``stop_gap`` is exactly what GB-57 needs to separate rule cost from gap
    cost, and it is the same distinction the backtester records.
    """
    client_id = client_order_id(order)
    fill = float(order.filled_price or 0.0)

    if client_id.endswith(TARGET_IN_LOOP_SUFFIX):
        # Tested before TARGET_SUFFIX only for the reader's sake - the two suffixes cannot
        # both match - and it takes no gap variant: it is a market order at the open, so
        # "filled away from the level" is the whole of what it is, not a special case.
        return TARGET_IN_LOOP
    if client_id.endswith(STOP_SUFFIX):
        return STOP_GAP if fill < stop_loss else STOP
    if client_id.endswith(TARGET_SUFFIX):
        return TARGET_GAP if fill > take_profit else TARGET
    return SIGNAL


def realised_slippage_bps(
    trades: Iterable[Trade], modelled_bps: float
) -> dict[str, float]:
    """What the frictions actually cost, against what the study assumed.

    The backtest charges ``modelled_bps`` per round trip by construction. This is the same
    quantity measured from fills that really happened, which is the one comparison the live
    system can make that the backtest cannot - GB-57 reports the difference.
    """
    measured = [
        10_000.0 * trade.costs / (trade.entry_price * trade.size)
        for trade in trades
        if trade.entry_price > 0 and trade.size > 0
    ]
    if not measured:
        return {"n": 0.0, "measured_bps": float("nan"), "modelled_bps": modelled_bps}
    mean = float(np.mean(measured))
    return {
        "n": float(len(measured)),
        "measured_bps": mean,
        "modelled_bps": modelled_bps,
        "excess_bps": mean - modelled_bps,
    }


def _exit_cost(
    order: BrokerOrder, holding: Holding, exit_price: float, size: float
) -> float:
    """Slippage against the level the rule chose, from the fill the broker reported.

    A stop whose level was 97.00 and which filled at 96.80 cost 0.20 a share, and that is a
    measurement rather than an assumption. A signal exit has no level to measure against, so
    it contributes nothing here - its cost is already inside the fill price.
    """
    client_id = client_order_id(order)
    if client_id.endswith(STOP_SUFFIX):
        return max(0.0, holding.stop_loss - exit_price) * size
    if client_id.endswith(TARGET_SUFFIX):
        return max(0.0, holding.take_profit - exit_price) * size
    return 0.0


def _filled_at(order: BrokerOrder) -> pd.Timestamp:
    raw = order.raw
    when = (
        raw.get("filled_at")
        if isinstance(raw, dict)
        else getattr(raw, "filled_at", None)
    )
    return (
        pd.Timestamp(when) if when is not None else pd.Timestamp.min.tz_localize("UTC")
    )


__all__ = [
    "ANY_PROVENANCE",
    "LIVE",
    "PENDING_FILE",
    "REHEARSAL_PREFIX",
    "REPLAY_PREFIX",
    "SIGNAL",
    "STOP",
    "STOP_GAP",
    "TARGET",
    "TARGET_GAP",
    "TARGET_IN_LOOP",
    "TARGET_IN_LOOP_SUFFIX",
    "DuplicateDecision",
    "amend_decision",
    "client_order_id",
    "decision_key",
    "decode_decision",
    "emit_trades",
    "encode_decision",
    "exit_reason_for",
    "is_rehearsal",
    "is_replay",
    "is_reportable",
    "load_decisions",
    "load_pending",
    "month_file",
    "realised_slippage_bps",
    "rehearsal_provenance",
    "replay_provenance",
    "resolve_pending",
    "save_decision",
    "save_pending",
]
