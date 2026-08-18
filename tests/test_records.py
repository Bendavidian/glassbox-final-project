"""GB-29 acceptance: a record survives the round trip, and a live exit becomes a Trade.

The trade half is the one that matters. Without it a live stop fills, the position vanishes
from ``get_positions``, the reconciler drops it — correctly, it is read-only — and **nothing
records the exit**. GB-19's metrics would then have no live trade log to run over, and the
report could not show backtest and live on the same period at all.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from fake_broker import FakeBroker

from glassbox import records
from glassbox.backtest import engine, metrics
from glassbox.contracts.schemas import (
    Attribution,
    DecisionRecord,
    Forecast,
    Signal,
    Trade,
)
from glassbox.engine.executor import SELL, BrokerOrder
from glassbox.engine.reconcile import Book, Holding
from glassbox.engine.signal import ENTER_LONG

SYMBOL = "AAPL"
DECISION = "gb29-0001"
ENTRY_TIME = pd.Timestamp("2026-08-10 13:30", tz="UTC")
EXIT_TIME = pd.Timestamp("2026-08-14 15:45", tz="UTC")
ENTRY_PRICE = 100.0
STOP_LEVEL = 97.0
TARGET_LEVEL = 106.0


def a_record(as_of: str = "2026-08-17 20:00", symbol: str = SYMBOL) -> DecisionRecord:
    stamp = pd.Timestamp(as_of, tz="UTC")
    return DecisionRecord(
        as_of=stamp,
        symbol=symbol,
        forecast=Forecast(
            path=np.array([0.001, -0.002, 0.003, 0.004], dtype="float32"),
            symbol=symbol,
            as_of=stamp,
        ),
        attribution=Attribution(
            per_channel={"close_logret": 0.004, "rsi14": 0.002},
            per_lag=None,
            per_frequency=None,
            gain_phase=None,
            forecast_total=0.006,
        ),
        signal=Signal(
            symbol=symbol,
            action=ENTER_LONG,
            trend_strength=0.006,
            up_points=3,
            passed_threshold=True,
        ),
        order={"notional": 10_000.0, "shares": 100.0},
        narrative="the 12-day cycle dominates",
        config_hash="abc123",
    )


def a_holding(quantity: float = 100.0) -> Holding:
    return Holding(
        symbol=SYMBOL,
        quantity=quantity,
        decision_id=DECISION,
        entry_price=ENTRY_PRICE,
        stop_loss=STOP_LEVEL,
        take_profit=TARGET_LEVEL,
    )


def a_fill(
    leg: str, price: float, quantity: float = 100.0, status: str = "filled"
) -> BrokerOrder:
    return BrokerOrder(
        id=f"order-{leg}",
        symbol=SYMBOL,
        side=SELL,
        quantity=quantity,
        status=status,
        filled_quantity=quantity if status == "filled" else 0.0,
        filled_price=price,
        raw={"client_order_id": f"{DECISION}{leg}", "filled_at": EXIT_TIME},
    )


ENTRY_FILLS = {SYMBOL: (ENTRY_TIME, ENTRY_PRICE, 2.0, "order-entry")}


# ── decision records ─────────────────────────────────────────────────────────


def test_a_record_round_trips_losslessly(tmp_path) -> None:
    original = a_record()

    records.save_decision(original, tmp_path)
    [restored] = records.load_decisions("2026-08-01", "2026-08-31", tmp_path)

    assert restored.as_of == original.as_of
    assert restored.symbol == original.symbol
    assert restored.config_hash == original.config_hash
    assert restored.narrative == original.narrative
    assert restored.order == original.order
    assert restored.signal == original.signal
    assert restored.attribution.per_channel == original.attribution.per_channel
    np.testing.assert_array_equal(restored.forecast.path, original.forecast.path)


def test_the_forecast_path_survives_as_float32(tmp_path) -> None:
    """A numpy array serialises as a list and must come back as an array of the same dtype,
    or the replayed decision is not the decision that was made."""
    records.save_decision(a_record(), tmp_path)
    [restored] = records.load_decisions("2026-08-01", "2026-08-31", tmp_path)

    assert isinstance(restored.forecast.path, np.ndarray)
    assert restored.forecast.path.dtype == np.float32


def test_timestamps_are_iso_8601_utc_on_disk(tmp_path) -> None:
    path = records.save_decision(a_record(), tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    assert raw["as_of"] == "2026-08-17T20:00:00+00:00"
    assert raw["forecast"]["as_of"].endswith("+00:00")


def test_a_naive_timestamp_is_assumed_utc_rather_than_guessed(tmp_path) -> None:
    naive = a_record()
    object.__setattr__(naive, "as_of", pd.Timestamp("2026-08-17 20:00"))

    path = records.save_decision(naive, tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    assert raw["as_of"] == "2026-08-17T20:00:00+00:00"


def test_records_land_in_a_monthly_file(tmp_path) -> None:
    records.save_decision(a_record("2026-08-17 20:00"), tmp_path)
    records.save_decision(a_record("2026-09-02 20:00"), tmp_path)

    assert (tmp_path / "decisions" / "2026-08.jsonl").is_file()
    assert (tmp_path / "decisions" / "2026-09.jsonl").is_file()


def test_the_config_hash_ties_a_record_to_its_settings(tmp_path) -> None:
    """Without it a replayed record could be re-run under different settings and the
    difference would look like nondeterminism rather than a config change."""
    records.save_decision(a_record(), tmp_path)
    [restored] = records.load_decisions("2026-08-01", "2026-08-31", tmp_path)

    assert restored.config_hash == "abc123"


def test_loading_a_range_spans_months_and_excludes_outside_it(tmp_path) -> None:
    for when in ("2026-07-30 20:00", "2026-08-17 20:00", "2026-09-02 20:00"):
        records.save_decision(a_record(when), tmp_path)

    loaded = records.load_decisions("2026-08-01", "2026-09-30", tmp_path)

    assert [str(r.as_of.date()) for r in loaded] == ["2026-08-17", "2026-09-02"]


def test_records_come_back_in_chronological_order(tmp_path) -> None:
    for when in ("2026-08-20 20:00", "2026-08-10 20:00", "2026-08-15 20:00"):
        records.save_decision(a_record(when), tmp_path)

    loaded = records.load_decisions("2026-08-01", "2026-08-31", tmp_path)

    assert [r.as_of for r in loaded] == sorted(r.as_of for r in loaded)


def test_a_missing_month_is_empty_rather_than_an_error(tmp_path) -> None:
    assert records.load_decisions("2020-01-01", "2020-12-31", tmp_path) == []


def test_one_corrupt_line_does_not_lose_the_month(tmp_path, caplog) -> None:
    """A session killed mid-write leaves exactly this, and refusing the whole file would
    lose every decision that did land."""
    records.save_decision(a_record("2026-08-10 20:00"), tmp_path)
    path = records.month_file(tmp_path, pd.Timestamp("2026-08-10", tz="UTC"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"as_of": "truncated\n')
    records.save_decision(a_record("2026-08-11 20:00"), tmp_path)

    loaded = records.load_decisions("2026-08-01", "2026-08-31", tmp_path)

    assert len(loaded) == 2
    assert any("unreadable" in record.message for record in caplog.records)


# ── the live trade log ───────────────────────────────────────────────────────


def test_a_filled_stop_becomes_a_trade() -> None:
    """The case GB-23 could not cover: the position vanishes and something must record it."""
    book = Book(managed={SYMBOL: a_holding()})

    [trade] = records.emit_trades([a_fill("-stop", 97.0)], book, ENTRY_FILLS)

    assert isinstance(trade, Trade)
    assert trade.symbol == SYMBOL
    assert trade.exit_reason == records.STOP
    assert trade.entry_time == ENTRY_TIME
    assert trade.exit_time == EXIT_TIME
    assert trade.exit_order_id == "order--stop"
    assert trade.entry_order_id == "order-entry"


def test_a_stop_that_gapped_through_is_labelled_stop_gap() -> None:
    """The distinction GB-57 needs to separate rule cost from gap cost, and the same one the
    backtester records."""
    book = Book(managed={SYMBOL: a_holding()})

    [trade] = records.emit_trades([a_fill("-stop", 95.0)], book, ENTRY_FILLS)

    assert trade.exit_reason == records.STOP_GAP


def test_a_target_and_its_gap_are_distinguished() -> None:
    book = Book(managed={SYMBOL: a_holding()})

    [clean] = records.emit_trades([a_fill("-target", 106.0)], book, ENTRY_FILLS)
    [gapped] = records.emit_trades([a_fill("-target", 108.0)], book, ENTRY_FILLS)

    assert clean.exit_reason == records.TARGET
    assert gapped.exit_reason == records.TARGET_GAP


def test_a_signal_exit_is_neither_leg() -> None:
    """No suffix, so no level to have been gapped through."""
    book = Book(managed={SYMBOL: a_holding()})

    [trade] = records.emit_trades([a_fill("", 101.0)], book, ENTRY_FILLS)

    assert trade.exit_reason == records.SIGNAL


def test_the_leg_is_identified_by_order_id_not_by_price() -> None:
    """Two levels can sit close together, and a price-only guess would mislabel a trade the
    report then explains wrongly."""
    tight = Holding(
        symbol=SYMBOL,
        quantity=100.0,
        decision_id=DECISION,
        entry_price=100.0,
        stop_loss=99.9,
        take_profit=100.1,
    )
    book = Book(managed={SYMBOL: tight})

    [trade] = records.emit_trades([a_fill("-target", 100.0)], book, ENTRY_FILLS)

    assert trade.exit_reason == records.TARGET  # priced between both levels


def test_costs_come_from_the_fill_not_from_the_configured_bps() -> None:
    """The one measurement the live system can make that the backtest cannot.

    A stop whose level was 97.00 and which filled at 96.80 cost 0.20 a share. Nothing in
    the config appears in that number.
    """
    book = Book(managed={SYMBOL: a_holding()})

    [trade] = records.emit_trades([a_fill("-stop", 96.80)], book, ENTRY_FILLS)

    assert trade.costs == pytest.approx(2.0 + 0.20 * 100.0)
    assert trade.net_pnl == pytest.approx(trade.gross_pnl - trade.costs)


def test_realised_slippage_is_reported_against_the_modelled_bps() -> None:
    """GB-57's number: what the frictions really cost, against the 2 bps the study assumed."""
    book = Book(managed={SYMBOL: a_holding()})
    trades = records.emit_trades([a_fill("-stop", 96.80)], book, ENTRY_FILLS)

    measured = records.realised_slippage_bps(trades, modelled_bps=6.0)

    assert measured["n"] == 1.0
    assert measured["measured_bps"] == pytest.approx(10_000 * 22.0 / 10_000.0)
    assert measured["excess_bps"] == pytest.approx(measured["measured_bps"] - 6.0)


def test_no_trade_is_emitted_for_a_quarantined_position(caplog) -> None:
    """The 0.01 AAPL probe, in the trade log's terms: the system did not open it, has no
    entry basis for it, and must not invent one."""
    quarantined = Book(unmanaged={SYMBOL: 0.01})

    trades = records.emit_trades([a_fill("-stop", 97.0)], quarantined, ENTRY_FILLS)

    assert trades == []
    assert any("not a managed holding" in r.message for r in caplog.records)


def test_no_trade_without_a_recorded_entry_fill(caplog) -> None:
    """A trade with an invented entry price would be a number with no provenance in the
    study's own trade log."""
    book = Book(managed={SYMBOL: a_holding()})

    assert records.emit_trades([a_fill("-stop", 97.0)], book, {}) == []
    assert any("no recorded entry fill" in r.message for r in caplog.records)


def test_a_buy_fill_does_not_close_anything() -> None:
    book = Book(managed={SYMBOL: a_holding()})
    entry = BrokerOrder(
        id="order-entry",
        symbol=SYMBOL,
        side="buy",
        quantity=100.0,
        status="filled",
        filled_quantity=100.0,
        filled_price=100.0,
        raw={"client_order_id": DECISION, "filled_at": ENTRY_TIME},
    )

    assert records.emit_trades([entry], book, ENTRY_FILLS) == []


def test_an_unfilled_sell_does_not_close_anything() -> None:
    book = Book(managed={SYMBOL: a_holding()})

    assert (
        records.emit_trades([a_fill("-stop", 97.0, status="new")], book, ENTRY_FILLS)
        == []
    )


def test_trades_come_back_in_exit_order() -> None:
    book = Book(managed={SYMBOL: a_holding()})
    later = a_fill("-target", 106.0)
    earlier = a_fill("-stop", 97.0)
    object.__setattr__(
        earlier, "raw", {"client_order_id": f"{DECISION}-stop", "filled_at": ENTRY_TIME}
    )

    trades = records.emit_trades([later, earlier], book, ENTRY_FILLS)

    assert [t.exit_time for t in trades] == sorted(t.exit_time for t in trades)


# ── one type, and GB-19 reads it unchanged ───────────────────────────────────


def test_a_live_trade_is_the_same_type_as_a_backtest_trade() -> None:
    """The decision GB-29 had to make, asserted rather than described.

    Two types with a shared shape would need a translation layer for GB-19, and the live
    path may not import the backtester — so the shared type moved to `contracts`.
    """
    assert engine.Trade is Trade
    assert records.emit_trades.__annotations__["return"] == "list[Trade]"


def test_gb19_metrics_run_over_a_live_trade_log_unchanged() -> None:
    """The requirement, end to end: no translation, no adapter, no second code path."""
    book = Book(managed={SYMBOL: a_holding()})
    live = records.emit_trades(
        [a_fill("-target", 106.0)], book, ENTRY_FILLS
    ) + records.emit_trades([a_fill("-stop", 96.5)], book, ENTRY_FILLS)

    index = pd.date_range("2026-08-10", periods=5, freq="B", tz="UTC")
    arm = metrics.ArmResult(
        name="paper",
        equity=pd.Series(
            [100_000.0, 100_400.0, 100_200.0, 100_600.0, 100_550.0], index=index
        ),
        trades=tuple(live),
    )

    assert len(arm.strategy_trades) == 2
    assert not math.isnan(metrics.hit_rate(arm))
    assert not math.isnan(metrics.average_trade(arm))
    assert metrics.total_return(arm) == pytest.approx(0.0055)


def test_a_backtest_trade_has_no_order_ids() -> None:
    """The live-only fields default to None, so the backtest is unaffected by their
    existence and GB-19 never reads them."""
    backtest_trade = Trade(
        symbol=SYMBOL,
        entry_time=ENTRY_TIME,
        exit_time=EXIT_TIME,
        size=1.0,
        entry_price=100.0,
        exit_price=101.0,
        gross_pnl=1.0,
        costs=0.06,
        net_pnl=0.94,
        exit_reason="signal",
        strategy_exit=True,
    )

    assert backtest_trade.entry_order_id is None
    assert backtest_trade.exit_order_id is None


def test_the_exit_vocabulary_matches_the_backtesters_exactly() -> None:
    """`records.py` may not import `backtest`, so the four strings are repeated there. This
    is what stops the two copies drifting: rename one and this fails."""
    assert records.STOP == engine.STOP
    assert records.STOP_GAP == engine.STOP_GAP
    assert records.TARGET == engine.TARGET
    assert records.TARGET_GAP == engine.TARGET_GAP
    assert records.SIGNAL == engine.SIGNAL


def test_the_fake_broker_still_satisfies_what_records_reads() -> None:
    """A trade log built from a fake broker's orders is the shape the real one produces."""
    broker = FakeBroker(prices={SYMBOL: 100.0})
    broker.positions[SYMBOL] = 100.0
    broker.submit_stop_order(SYMBOL, 100.0, 97.0, f"{DECISION}-stop")

    [order] = broker.get_orders()

    assert records.exit_reason_for(order, STOP_LEVEL, TARGET_LEVEL) in {
        records.STOP,
        records.STOP_GAP,
    }
