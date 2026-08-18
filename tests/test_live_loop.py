"""GB-26 acceptance: the live cycle, and every ruling it is supposed to implement.

The loop runs against ``FakeBroker`` and a synthetic universe, so the suite needs no
network, no credentials and no market. What is asserted here is not that the steps run —
that is the easy half — but that the rulings hold: completed bars only, the caller's
history floor, protection before entries, a first cycle that cannot add risk, and a stale
symbol that loses its entry but keeps its stop.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest
from fake_broker import FakeBroker

from glassbox import live_loop, records
from glassbox.config.loader import Config, load_config
from glassbox.contracts.schemas import ChannelStats
from glassbox.engine.executor import BUY, SELL, BrokerOrder
from glassbox.engine.reconcile import Book, Holding
from glassbox.engine.signal import Thresholds
from glassbox.features import builder
from glassbox.model.persistence import PersistenceForecaster
from glassbox.model.predict import Predictor

INPUT_LEN = 30
HORIZON = 4
BARS = 420
UNIVERSE = ("AAPL", "MSFT")
PRICE = 100.0

# A Wednesday inside the NYSE session: 2026-08-12 14:00 UTC is 10:00 in New York.
NOW = pd.Timestamp("2026-08-12 14:00", tz="UTC")


# ── a forecaster whose forecast is whatever the test needs ───────────────────


class ConstantForecaster(PersistenceForecaster):
    """Predicts a fixed cumulative return, and can decompose it exactly.

    A stub rather than a fitted model, because what is under test is the loop. It still
    goes through ``Attribution.from_terms``, so a decomposition that did not close would
    fail here exactly as it would in production.
    """

    def __init__(self, input_len: int, horizon: int, channels, total: float) -> None:
        super().__init__(input_len, horizon)
        self.name = "constant"
        self.channels = tuple(channels)
        self.total = total

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._require_window_shape(X)
        return np.full((X.shape[0], self.horizon), self.total / self.horizon, "float32")

    def linear_terms(self, x, channels):
        weight = np.zeros((self.horizon, self.input_len))
        weight[:, 0] = 1.0
        values = np.zeros(self.input_len)
        values[0] = float(np.float32(self.total / self.horizon))
        terms = {channel: () for channel in channels}
        terms[channels[0]] = ((weight, values),)
        return terms


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    base = load_config()
    return replace(
        base,
        universe=UNIVERSE,
        window=replace(base.window, input_len=INPUT_LEN, horizon=HORIZON),
        data=replace(base.data, cache_dir=str(tmp_path / "cache")),
    )


def sessions(cfg: Config, end: pd.Timestamp, count: int) -> pd.DatetimeIndex:
    """Real NYSE session dates ending at ``end``, so the calendar arithmetic is real."""
    days = mcal.get_calendar(cfg.data.calendar).valid_days(
        (end - pd.Timedelta(days=count * 3)).date(), end.date()
    )
    return pd.DatetimeIndex(days).normalize()[-count:]


def make_bars(cfg: Config, end: pd.Timestamp, count: int = BARS) -> pd.DataFrame:
    """Completed sessions only, ending at the last one that closed before ``end``.

    The volume varies deliberately: a constant one gives ``vol_z`` a zero standard
    deviation, every row becomes NaN and the warm-up trim returns an empty frame.
    """
    index = sessions(cfg, live_loop.last_completed_session(cfg, end), count)
    close = pd.Series(
        [PRICE + 5.0 * math.sin(i / 9.0) for i in range(len(index))],
        index=index,
        dtype="float64",
    )
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": pd.Series(
                [1_000_000.0 + 7_000.0 * math.cos(i / 5.0) for i in range(len(index))],
                index=index,
            ),
            "log_return": np.log(close / close.shift(1)),
            "source": pd.Series("alpaca", index=index, dtype="string"),
        },
        index=index,
    )


def a_predictor(cfg: Config, total: float = 0.0) -> Predictor:
    channels = cfg.channels.active_channels
    stats = ChannelStats(
        channels=channels,
        mean=tuple(0.0 for _ in channels),
        std=tuple(1.0 for _ in channels),
        fitted_start=pd.Timestamp("2020-01-01", tz="UTC"),
        fitted_end=pd.Timestamp("2025-01-01", tz="UTC"),
        n_rows=100,
    )
    return Predictor(
        model=ConstantForecaster(INPUT_LEN, HORIZON, channels, total),
        stats=dict.fromkeys(cfg.universe, stats),
        config_hash="test",
        channels=channels,
    )


def a_state(
    cfg: Config,
    tmp_path: Path,
    broker: FakeBroker,
    *,
    total: float = 0.0,
    thresholds: Thresholds | None = None,
) -> live_loop.LiveState:
    return live_loop.LiveState(
        cfg=cfg,
        broker=broker,
        predictor=a_predictor(cfg, total),
        thresholds=thresholds or Thresholds(lower=0.004),
        state_dir=tmp_path,
    )


@pytest.fixture
def broker() -> FakeBroker:
    return FakeBroker(prices=dict.fromkeys(UNIVERSE, PRICE))


@pytest.fixture
def stub_predictor(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_session` loads a checkpoint from disk; these tests inject one instead."""
    monkeypatch.setattr(
        live_loop, "load_predictor", lambda directory, config: a_predictor(cfg, 0.0)
    )


@pytest.fixture
def stub_bars(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace the network fetch, and record what the loop asked it for."""
    asked: dict = {}
    frames = {symbol: make_bars(cfg, NOW) for symbol in UNIVERSE}

    def fake_fetch(symbols, min_bars, *, requirement="", lookback_days=None):
        asked.update(
            symbols=list(symbols),
            min_bars=min_bars,
            requirement=requirement,
            lookback_days=lookback_days,
        )
        return {symbol: frames[symbol].copy() for symbol in symbols}

    monkeypatch.setattr(live_loop, "load_live_bars", fake_fetch)
    asked["frames"] = frames
    return asked


# ── completed bars only ──────────────────────────────────────────────────────


def test_drop_incomplete_bar_removes_the_session_in_progress(
    cfg: Config, caplog: pytest.LogCaptureFixture
) -> None:
    """Constructed in-progress bar: today's, while today's session is still open."""
    frame = make_bars(cfg, NOW)
    today = pd.Timestamp(NOW.date(), tz="UTC")
    assert today not in frame.index  # the fixture holds completed sessions only
    in_progress = frame.iloc[[-1]].copy()
    in_progress.index = pd.DatetimeIndex([today])
    with_partial = pd.concat([frame, in_progress])

    with caplog.at_level("INFO", logger="glassbox.live_loop"):
        kept = live_loop.drop_incomplete_bar(with_partial, cfg, NOW, "AAPL")

    assert today not in kept.index
    assert len(kept) == len(with_partial) - 1
    assert f"{today:%Y-%m-%d}" in caplog.text
    assert "has not closed" in caplog.text


def test_drop_incomplete_bar_is_a_no_op_on_completed_history(cfg: Config) -> None:
    frame = make_bars(cfg, NOW)

    assert live_loop.drop_incomplete_bar(frame, cfg, NOW, "AAPL").equals(frame)


def test_the_last_completed_session_is_yesterday_during_a_session(cfg: Config) -> None:
    """Today's bar is still forming while the market is open."""
    completed = live_loop.last_completed_session(cfg, NOW)

    assert completed < pd.Timestamp(NOW.date(), tz="UTC")


# ── the history floor is the caller's, not input_len ─────────────────────────


def test_the_fetch_asks_for_min_history_bars(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """The bug GB-26's row was written against: asking for `input_len` returns 163 bars."""
    live_loop.run_cycle(a_state(cfg, tmp_path, broker), NOW)

    assert stub_bars["min_bars"] == builder.min_history_bars(cfg)
    assert stub_bars["min_bars"] != cfg.window.input_len
    assert "rsi14 warm-up" in stub_bars["requirement"]


# ── stale data: dropped from entries, never from protection ──────────────────


def test_a_stale_symbol_is_excluded_and_named(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    stub_bars["frames"]["MSFT"] = stub_bars["frames"]["MSFT"].iloc[:-3]

    report = live_loop.run_cycle(a_state(cfg, tmp_path, broker), NOW)

    assert report.stale == ("MSFT",)
    assert report.ranked_over == ("AAPL",)


def test_a_stale_symbol_keeps_its_protection(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """The asymmetry that is the whole ruling: no new risk, but risk already taken is
    still managed."""
    stub_bars["frames"]["MSFT"] = stub_bars["frames"]["MSFT"].iloc[:-3]
    broker.positions["MSFT"] = 10.0
    state = a_state(cfg, tmp_path, broker)
    state.book = Book(
        managed={
            "MSFT": Holding(
                symbol="MSFT",
                quantity=10.0,
                decision_id="d1",
                entry_price=PRICE,
                stop_loss=97.0,
                take_profit=106.0,
            )
        }
    )

    report = live_loop.run_cycle(state, NOW)

    assert "MSFT" in report.stale
    assert "MSFT" in report.rearmed
    assert "stop" in broker.submitted_kinds
    assert "limit" in broker.submitted_kinds


def test_every_symbol_stale_is_a_no_op_that_still_reconciles(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    for symbol in UNIVERSE:
        stub_bars["frames"][symbol] = stub_bars["frames"][symbol].iloc[:-3]
    broker.positions["AAPL"] = 5.0

    report = live_loop.run_cycle(a_state(cfg, tmp_path, broker), NOW)

    assert report.ok
    assert set(report.stale) == set(UNIVERSE)
    assert report.ranked_over == ()
    assert report.decisions == ()
    assert report.divergences  # the quarantine still happened


# ── the first cycle may reduce risk and may not add any ──────────────────────


def test_the_first_cycle_submits_no_entry(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    state = a_state(
        cfg, tmp_path, broker, total=0.05, thresholds=Thresholds(lower=0.004)
    )

    first = live_loop.run_cycle(state, NOW)

    assert first.entries_allowed is False
    assert "market" not in broker.submitted_kinds
    assert first.decisions  # it still decided, explained and recorded


def test_the_second_cycle_does_submit_an_entry(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    state = a_state(auto, tmp_path, broker, total=0.05)

    live_loop.run_cycle(state, NOW)
    second = live_loop.run_cycle(state, NOW)

    assert second.entries_allowed is True
    assert "market" in broker.submitted_kinds


def test_the_first_cycle_still_exits(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """An exit is an obligation on capital already committed, so it is never deferred."""
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    broker.positions["AAPL"] = 4.0
    state = a_state(auto, tmp_path, broker, total=-0.05)
    state.book = Book(
        managed={
            "AAPL": Holding(
                symbol="AAPL",
                quantity=4.0,
                decision_id="d1",
                entry_price=PRICE,
                stop_loss=97.0,
                take_profit=106.0,
            )
        }
    )

    report = live_loop.run_cycle(state, NOW)

    assert report.entries_allowed is False
    sells = [order for order in broker.orders if order.side == SELL]
    assert any(records._client_order_id(order).endswith("-AAPL") for order in sells)


# ── protection: rules 2, 3 and 4 ─────────────────────────────────────────────


def test_a_missing_leg_is_rearmed(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    broker.positions["AAPL"] = 3.0
    state = a_state(cfg, tmp_path, broker)
    state.book = Book(
        managed={
            "AAPL": Holding(
                symbol="AAPL",
                quantity=3.0,
                decision_id="d1",
                entry_price=PRICE,
                stop_loss=97.0,
                take_profit=106.0,
            )
        }
    )

    report = live_loop.run_cycle(state, NOW)

    assert report.rearmed == ("AAPL",)
    armed = [records._client_order_id(o) for o in broker.orders if o.side == SELL]
    assert "d1-stop" in armed
    assert "d1-target" in armed


def test_two_consecutive_arming_failures_flatten_the_position(
    cfg: Config, tmp_path: Path
) -> None:
    """An unprotected position is worse than a closed one (rule 3)."""

    class RefusingBroker(FakeBroker):
        def submit_stop_order(self, symbol, quantity, stop_price, client_order_id):
            raise RuntimeError("simulated arming failure")

    broker = RefusingBroker(prices=dict.fromkeys(UNIVERSE, PRICE))
    broker.positions["AAPL"] = 3.0
    holding = Holding(
        symbol="AAPL",
        quantity=3.0,
        decision_id="d1",
        entry_price=PRICE,
        stop_loss=97.0,
        take_profit=106.0,
    )
    state = a_state(cfg, tmp_path, broker)
    state.book = Book(managed={"AAPL": holding})

    first, _ = live_loop.protect_book(state, broker.get_orders(), lambda m: None)
    assert first == []
    assert state.arming_failures["AAPL"] == 1

    _, flattened = live_loop.protect_book(state, broker.get_orders(), lambda m: None)

    assert flattened == ["AAPL"]
    assert any(
        order.side == SELL and records._client_order_id(order).endswith("-flatten")
        for order in broker.orders
    )


def test_a_filled_leg_cancels_its_sibling_and_the_cancellation_is_verified(
    cfg: Config, broker: FakeBroker, tmp_path: Path
) -> None:
    holding = Holding(
        symbol="AAPL",
        quantity=3.0,
        decision_id="d1",
        entry_price=PRICE,
        stop_loss=97.0,
        take_profit=106.0,
    )
    state = a_state(cfg, tmp_path, broker)
    state.book = Book(managed={"AAPL": holding})
    orders = [
        BrokerOrder(
            id="stop-1",
            symbol="AAPL",
            side=SELL,
            quantity=3.0,
            status="filled",
            filled_quantity=3.0,
            filled_price=97.0,
            raw={"client_order_id": "d1-stop"},
        ),
        BrokerOrder(
            id="target-1",
            symbol="AAPL",
            side=SELL,
            quantity=3.0,
            status="new",
            raw={"client_order_id": "d1-target"},
        ),
    ]

    live_loop.protect_book(state, orders, lambda m: None)

    assert "target-1" in broker.cancelled


def test_a_fill_is_protected_in_the_cycle_that_saw_it(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """Rule 1: the open is the wrong minute to be idle."""
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    state = a_state(auto, tmp_path, broker, total=0.05)

    live_loop.run_cycle(state, NOW)
    report = live_loop.run_cycle(state, NOW)

    assert report.submissions
    assert "stop" in broker.submitted_kinds
    assert "limit" in broker.submitted_kinds
    assert state.book.managed  # taken into the book in the same cycle
    assert set(state.entry_fills) == set(state.book.managed)


# ── the session ──────────────────────────────────────────────────────────────


def test_outside_the_window_nothing_is_fetched(
    cfg: Config,
    broker: FakeBroker,
    stub_bars: dict,
    stub_predictor: None,
    tmp_path: Path,
) -> None:
    before_open = pd.Timestamp("2026-08-12 08:00", tz="UTC")
    report = live_loop.run_session(
        cfg, tmp_path, broker=broker, clock=lambda: before_open, sleep=lambda s: None
    )

    assert report.cycles == ()
    assert report.stopped_by == "outside the session"
    assert "min_bars" not in stub_bars


def test_a_holiday_is_not_a_session(cfg: Config) -> None:
    christmas = pd.Timestamp("2026-12-25 15:00", tz="UTC")

    assert live_loop.market_session(cfg, christmas) is None
    assert not live_loop.in_session(cfg, christmas)


def test_a_session_runs_to_its_cycle_cap(
    cfg: Config,
    broker: FakeBroker,
    stub_bars: dict,
    stub_predictor: None,
    tmp_path: Path,
) -> None:
    report = live_loop.run_session(
        cfg,
        tmp_path,
        broker=broker,
        clock=lambda: NOW,
        sleep=lambda s: None,
        max_cycles=3,
    )

    assert len(report.cycles) == 3
    assert all(cycle.ok for cycle in report.cycles)
    assert report.stopped_by == "max_cycles=3"


def test_an_interrupt_finishes_cleanly_and_reports_open_orders(
    cfg: Config,
    broker: FakeBroker,
    stub_bars: dict,
    stub_predictor: None,
    tmp_path: Path,
) -> None:
    """ "Nothing in flight" has to be a printed list rather than a hope."""
    broker.orders.append(
        BrokerOrder(
            id="working-1",
            symbol="AAPL",
            side=BUY,
            quantity=1.0,
            status="new",
            raw={"client_order_id": "x"},
        )
    )

    def interrupt(seconds: float) -> None:
        raise KeyboardInterrupt

    report = live_loop.run_session(
        cfg, tmp_path, broker=broker, clock=lambda: NOW, sleep=interrupt
    )

    assert report.stopped_by == "KeyboardInterrupt"
    assert len(report.cycles) == 1
    assert any("working-1" in entry for entry in report.open_orders)


def test_a_failing_step_ends_the_cycle_and_not_the_session(
    cfg: Config,
    broker: FakeBroker,
    stub_predictor: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*args, **kwargs):
        raise RuntimeError("the data vendor fell over")

    monkeypatch.setattr(live_loop, "load_live_bars", explode)

    report = live_loop.run_session(
        cfg,
        tmp_path,
        broker=broker,
        clock=lambda: NOW,
        sleep=lambda s: None,
        max_cycles=2,
    )

    assert len(report.cycles) == 2
    assert all(not cycle.ok for cycle in report.cycles)
    assert "the data vendor fell over" in report.cycles[0].error
    assert "RuntimeError" in report.summary()


def test_dry_run_submits_nothing_but_still_decides(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    state = live_loop.LiveState(
        cfg=auto,
        broker=live_loop.DryRunBroker(broker),
        predictor=a_predictor(auto, 0.05),
        thresholds=Thresholds(lower=0.004),
        state_dir=tmp_path,
    )

    live_loop.run_cycle(state, NOW)
    report = live_loop.run_cycle(state, NOW)

    assert report.decisions
    assert broker.orders == []  # the real broker never saw anything
    assert state.broker.submitted  # the dry-run broker recorded what it refused
    assert (tmp_path / "decisions").exists()


# ── the banner ───────────────────────────────────────────────────────────────


def test_the_banner_names_the_feed_the_model_and_the_config(
    cfg: Config, broker: FakeBroker, tmp_path: Path
) -> None:
    banner = live_loop.session_banner(
        a_state(cfg, tmp_path, broker), NOW, dry_run=False
    )

    assert "feed=sip" in banner
    assert "constant" in banner  # the model
    assert "rsi14" in banner  # the channel set
    assert "445 bars is input_len" not in banner  # the smaller test window
    assert "config hash" in banner
    assert "lower=0.004000" in banner
    assert cfg.live.mode in banner


def test_the_banner_says_when_the_session_stands_aside(
    cfg: Config, broker: FakeBroker, tmp_path: Path
) -> None:
    state = a_state(cfg, tmp_path, broker, thresholds=Thresholds.never())

    assert "stood aside" in live_loop.session_banner(state, NOW, dry_run=False)


# ── the band the harness wrote ───────────────────────────────────────────────


def test_a_missing_band_stands_aside_rather_than_inventing_one(tmp_path: Path) -> None:
    """A made-up threshold is the one input that would make a session unexplainable."""
    assert not live_loop.load_thresholds(tmp_path / "nothing.json").fires


def test_a_band_that_stood_aside_is_read_back_as_never(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"stood_aside": True, "lower": None}), encoding="utf-8")

    assert not live_loop.load_thresholds(path).fires


def test_a_real_band_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps({"stood_aside": False, "lower": 0.0042, "upper": 0.05}),
        encoding="utf-8",
    )

    thresholds = live_loop.load_thresholds(path)

    assert thresholds.lower == pytest.approx(0.0042)
    assert thresholds.upper == pytest.approx(0.05)


# ── the leg suffixes are the ones the executor actually writes ───────────────


def test_the_leg_suffixes_match_the_executors(cfg: Config) -> None:
    """`protect` names the legs and `records` reads the names; this arms them. Three
    readers of one convention, so the convention is asserted rather than assumed."""
    assert live_loop.LEG_SUFFIX[live_loop.STOP_LEG] == records.STOP_SUFFIX
    assert live_loop.LEG_SUFFIX[live_loop.TARGET_LEG] == records.TARGET_SUFFIX
