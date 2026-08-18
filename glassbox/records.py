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
from glassbox.engine.executor import BrokerOrder
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
SIGNAL = "signal"

# The suffixes `executor.protect` gives its two legs, which is how a fill is attributed.
STOP_SUFFIX = "-stop"
TARGET_SUFFIX = "-target"

FILLED = frozenset({"filled", "partially_filled"})
SELL = "sell"


# ── decision records ─────────────────────────────────────────────────────────


def save_decision(record: DecisionRecord, root: str | Path) -> Path:
    """Append one record to ``root/decisions/YYYY-MM.jsonl``. Returns the file written."""
    path = month_file(root, pd.Timestamp(record.as_of))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_encode(record), separators=(",", ":")) + "\n")
    return path


def load_decisions(
    start: pd.Timestamp | str, end: pd.Timestamp | str, root: str | Path
) -> list[DecisionRecord]:
    """Every record with ``start <= as_of <= end``, in chronological order.

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
    records = [
        record
        for month in _months_between(first, last)
        for record in _read_month(month_file(root, month))
        if first <= record.as_of <= last
    ]
    return sorted(records, key=lambda record: (record.as_of, record.symbol))


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
            "forecast_total": float(attribution.forecast_total),
        },
        "signal": asdict(record.signal),
        "order": record.order,
        "narrative": record.narrative,
        "config_hash": record.config_hash,
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
            forecast_total=attribution["forecast_total"],
        ),
        signal=Signal(**raw["signal"]),
        order=raw["order"],
        narrative=raw["narrative"],
        config_hash=raw["config_hash"],
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
    client_id = _client_order_id(order)
    fill = float(order.filled_price or 0.0)

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
    client_id = _client_order_id(order)
    if client_id.endswith(STOP_SUFFIX):
        return max(0.0, holding.stop_loss - exit_price) * size
    if client_id.endswith(TARGET_SUFFIX):
        return max(0.0, holding.take_profit - exit_price) * size
    return 0.0


def _client_order_id(order: BrokerOrder) -> str:
    raw = order.raw
    if isinstance(raw, dict):
        return str(raw.get("client_order_id", ""))
    return str(getattr(raw, "client_order_id", "") or "")


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
    "SIGNAL",
    "STOP",
    "STOP_GAP",
    "TARGET",
    "TARGET_GAP",
    "emit_trades",
    "exit_reason_for",
    "load_decisions",
    "month_file",
    "realised_slippage_bps",
    "save_decision",
]
