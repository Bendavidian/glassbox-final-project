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
from fake_broker import FakeBroker, FakeBrokerError

from glassbox import faults, live_loop, records
from glassbox.config.loader import Config, load_config
from glassbox.contracts.schemas import ChannelStats
from glassbox.engine.executor import BUY, SELL, BrokerOrder, RetryingBroker
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
# A decision is stamped with the bar it was made from, which is the session before.
MONTH = pd.Timestamp("2026-08-01", tz="UTC")


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

    def fake_fetch(
        symbols,
        min_bars,
        *,
        requirement="",
        lookback_days=None,
        attempts=1,
        backoff=0.0,
    ):
        asked.update(
            symbols=list(symbols),
            min_bars=min_bars,
            requirement=requirement,
            lookback_days=lookback_days,
            attempts=attempts,
            backoff=backoff,
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
    # GB-39: the retry policy is the deployment's, not the data module's. `load_live_bars`
    # defaults to no retry, so a loop that forgot to pass it would silently lose the
    # policy rather than fail.
    assert stub_bars["attempts"] == cfg.live.retry_attempts
    assert stub_bars["backoff"] == cfg.live.retry_backoff_seconds


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
    # And the bar is left UNDECIDED rather than recorded-and-stranded. Recording it here
    # would collide with the ruling of 19 Aug 2026: the next cycle would decline to
    # re-decide, and the entry would never be submitted at all.
    assert first.decisions == ()


def test_the_second_cycle_does_submit_an_entry(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    state = a_state(auto, tmp_path, broker, total=0.05)

    live_loop.run_cycle(state, NOW)
    second = live_loop.run_cycle(state, NOW)

    assert second.entries_allowed is True
    assert "market" in broker.submitted_kinds
    # Recorded exactly once, by the cycle that could act on it - so the record carries the
    # order and a narrative written in the knowledge of it.
    assert second.decisions
    assert len(records.load_decisions(MONTH, NOW, tmp_path)) == len(second.decisions)


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
    assert any(records.client_order_id(order).endswith("-AAPL-exit") for order in sells)


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
    armed = [records.client_order_id(o) for o in broker.orders if o.side == SELL]
    assert "d1#1-stop" in armed
    assert "d1#1-target" in armed


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
        order.side == SELL and records.client_order_id(order).endswith("-flatten")
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
    def unreachable(*args, **kwargs):
        # What the real `load_live_bars` raises once its own attempts are spent.
        raise faults.Unavailable("the daily bar fetch failed after 3 attempts")

    monkeypatch.setattr(live_loop, "load_live_bars", unreachable)

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
    assert "after 3 attempts" in report.cycles[0].error
    # GB-39: the feed is unreachable, which is a skipped cycle rather than a defect, and
    # the session says so rather than reporting the raw exception type.
    assert all(cycle.unreachable for cycle in report.cycles)
    assert "cycles skipped       : 2" in report.summary()


def test_a_defect_is_not_dressed_up_as_a_connectivity_problem(
    cfg: Config,
    broker: FakeBroker,
    stub_predictor: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two share one `except`, and telling them apart is the point of separating them.

    An unreachable broker costs one poll and is expected on a domestic connection. A
    `ValueError` in the same slot is a bug, and a report that called it a connectivity
    problem would be the loop hiding its own defect behind the network.
    """

    def a_real_bug(*args, **kwargs):
        raise ValueError("the window came back the wrong shape")

    monkeypatch.setattr(live_loop, "load_live_bars", a_real_bug)

    report = live_loop.run_session(
        cfg,
        tmp_path,
        broker=broker,
        clock=lambda: NOW,
        sleep=lambda s: None,
        max_cycles=1,
    )

    assert report.cycles[0].failed_step == "ValueError"
    assert report.cycles[0].unreachable is False
    assert "cycles skipped       : 0" in report.summary()


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


# ── decide once per completed bar, manage every cycle (ruling, 19 Aug 2026) ──


def stored_decisions(root: Path) -> list:
    return records.load_decisions(MONTH, NOW, root)


def test_n_cycles_in_one_session_decide_each_bar_once_and_protect_every_time(
    cfg: Config,
    broker: FakeBroker,
    stub_bars: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ruling, as one assertion in each direction.

    A decision is a function of one completed daily bar, and that bar does not change
    during a session - so at 60-second polling the loop used to re-derive the identical
    verdict on every one of a session's 390 cycles and record 390 identical rows per
    symbol, 1,950 a day over the real five-symbol universe. Risk management is explicitly
    outside the rule and still runs on every cycle, because prices move intraday and the
    risk already taken moves with them.
    """
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    state = a_state(auto, tmp_path, broker, total=0.05)

    verified: list[int] = []
    real_protect = live_loop.protect_book
    monkeypatch.setattr(
        live_loop,
        "protect_book",
        lambda *args, **kwargs: (verified.append(1), real_protect(*args, **kwargs))[1],
    )

    cycles = 8
    for _ in range(cycles):
        live_loop.run_cycle(state, NOW)

    stored = stored_decisions(tmp_path)
    assert len(verified) == cycles
    assert sorted(record.symbol for record in stored) == sorted(UNIVERSE)
    assert len({(record.as_of, record.symbol) for record in stored}) == len(UNIVERSE)


def test_a_repeated_entry_is_never_submitted_when_the_fill_has_not_appeared(
    cfg: Config, stub_bars: dict, tmp_path: Path
) -> None:
    """The race the ruling closes, made to happen rather than argued about.

    ``fill=False`` is the case that matters: the entry is accepted and no position appears.
    Reconciliation therefore cannot see it, the book does not hold the symbol, and the book
    was the only thing standing between the loop and a second identical entry - which makes
    it a race rather than a guard. At 60-second polling a market order can be sent many
    times over before the first fill lands.
    """
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    broker = FakeBroker(prices=dict.fromkeys(UNIVERSE, PRICE), fill=False)
    state = a_state(auto, tmp_path, broker, total=0.05)

    for _ in range(6):
        live_loop.run_cycle(state, NOW)

    buys = [order for order in broker.orders if order.side == BUY]
    assert len(buys) == len(UNIVERSE)
    assert len({records.client_order_id(order) for order in buys}) == len(UNIVERSE)


def test_the_entry_client_order_id_is_the_bar_and_not_the_cycle(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """The second guard, and the only one that can win a race at the broker.

    Measured against the paper account on 18 Aug 2026: Alpaca refuses a repeated
    ``client_order_id`` with ``40010001 client_order_id must be unique``. Deriving the id
    from the bar rather than from the polling cycle is what turns that refusal into a
    guarantee that one bar can produce at most one entry.
    """
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    state = a_state(auto, tmp_path, broker, total=0.05)

    live_loop.run_cycle(state, NOW)
    live_loop.run_cycle(state, NOW)

    bar = live_loop.last_completed_session(cfg, NOW)
    buys = [records.client_order_id(o) for o in broker.orders if o.side == BUY]
    assert f"{bar:%Y%m%d}-AAPL" in buys


def test_a_restart_mid_session_reads_what_it_has_already_decided(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """A new process starts with an empty cache; the log on disk is what it reads instead.

    Without the seed it would re-decide bars it had already recorded, the store would
    refuse the duplicate - correctly - and the cycle would fail on that refusal every
    minute until the close.
    """
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    first = a_state(auto, tmp_path, broker, total=0.05)
    live_loop.run_cycle(first, NOW)
    live_loop.run_cycle(first, NOW)
    before = len(stored_decisions(tmp_path))

    restarted = a_state(auto, tmp_path, broker, total=0.05)
    report = live_loop.run_cycle(restarted, NOW)

    assert report.ok
    assert report.decisions == ()
    assert set(report.settled) == set(UNIVERSE)
    assert len(stored_decisions(tmp_path)) == before


def test_an_exit_the_broker_already_has_is_not_sent_again(
    cfg: Config, stub_bars: dict, tmp_path: Path
) -> None:
    """Exits run every cycle. Running is not the same as re-sending."""
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    broker = FakeBroker(prices=dict.fromkeys(UNIVERSE, PRICE), fill=False)
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

    for _ in range(4):
        live_loop.run_cycle(state, NOW)

    sent = [
        order
        for order in broker.orders
        if records.client_order_id(order).endswith("-exit")
    ]
    assert len(sent) == 1


def test_an_exit_the_broker_refused_is_re_sent_on_the_next_cycle(
    cfg: Config, stub_bars: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exit is an obligation on capital already committed, so a failure is retried.

    This is the half of "manage every cycle" that an entry does not get: the loop does not
    re-decide the bar, it re-sends the order that bar decided, until the broker has it.
    """
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    broker = FakeBroker(prices=dict.fromkeys(UNIVERSE, PRICE), fill=False)
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

    refusals: list[str] = []
    real_submit = broker.submit_market_order

    def refuse_once(symbol, quantity, side, client_order_id):
        if side == SELL and not refusals:
            refusals.append(client_order_id)
            raise RuntimeError("the venue rejected it")
        return real_submit(symbol, quantity, side, client_order_id)

    monkeypatch.setattr(broker, "submit_market_order", refuse_once)

    first = live_loop.run_cycle(state, NOW)
    second = live_loop.run_cycle(state, NOW)

    assert refusals
    assert first.ok  # a refused exit does not take the cycle down with it
    assert second.ok
    sent = [
        order
        for order in broker.orders
        if records.client_order_id(order).endswith("-exit")
    ]
    assert len(sent) == 1


def test_re_arming_never_reuses_a_client_order_id(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """Measured against the paper account, 18 Aug 2026.

    Alpaca refuses a ``client_order_id`` it has seen before **including after the original
    was cancelled or expired**. The protective legs are submitted ``TimeInForce.DAY``, so
    they expire at every close; re-arming them the next morning under yesterday's id would
    have been refused, that refusal counts as an arming failure, and ARMING_STRIKES of them
    flatten the position at market. A position held overnight would have been liquidated
    two cycles into the next session for a reason unconnected to the strategy.
    """
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

    live_loop.run_cycle(state, NOW)
    broker.orders = [
        (
            replace(order, status="expired")
            if records.client_order_id(order).startswith("d1#")
            else order
        )
        for order in broker.orders
    ]
    second = live_loop.run_cycle(state, NOW)

    assert second.rearmed == ("AAPL",)
    armed = [
        records.client_order_id(order)
        for order in broker.orders
        if order.side == SELL and records.client_order_id(order).startswith("d1#")
    ]
    assert "d1#1-stop" in armed
    assert "d1#2-stop" in armed
    assert len(armed) == len(set(armed))


# ── GB-39: faults, and the restart that must not orphan or double ───────────


def test_retry_gives_up_after_the_configured_attempts(cfg: Config) -> None:
    """Three attempts, and the exhaustion is a distinct type rather than the raw error."""
    calls, waits = [], []

    def always_fails():
        calls.append(1)
        raise ConnectionError("the socket went away")

    with pytest.raises(faults.Unavailable, match="after 3 attempts"):
        faults.retry(
            always_fails,
            description="a call",
            attempts=cfg.live.retry_attempts,
            backoff=cfg.live.retry_backoff_seconds,
            sleep=waits.append,
        )

    assert len(calls) == cfg.live.retry_attempts
    assert waits == [1.0, 2.0]  # exponential, and it gives up inside one poll


def test_retry_returns_the_first_success_without_waiting() -> None:
    waits = []
    calls = []

    def fails_once():
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("slow")
        return "the answer"

    assert (
        faults.retry(
            fails_once,
            description="a call",
            attempts=3,
            backoff=1.0,
            sleep=waits.append,
        )
        == "the answer"
    )
    assert waits == [1.0]


def test_the_original_error_is_chained_not_swallowed() -> None:
    """The thing that broke is more useful than the fact that it broke three times."""
    with pytest.raises(faults.Unavailable) as caught:
        faults.retry(
            lambda: (_ for _ in ()).throw(ConnectionError("DNS")),
            description="a call",
            attempts=2,
            backoff=0.0,
            sleep=lambda _: None,
        )

    assert isinstance(caught.value.__cause__, ConnectionError)


def test_a_read_that_recovers_is_not_reported_as_a_failure(cfg: Config) -> None:
    inner = FakeBroker(prices={"AAPL": PRICE})
    inner.positions["AAPL"] = 2.0
    failures = []
    real = inner.get_positions

    def flaky():
        if not failures:
            failures.append(1)
            raise ConnectionError("reset by peer")
        return real()

    inner.get_positions = flaky
    quick = replace(cfg, live=replace(cfg.live, retry_backoff_seconds=0.0))

    assert RetryingBroker(inner, quick).get_positions() == {"AAPL": 2.0}


def test_a_write_refused_as_a_duplicate_on_retry_is_read_as_a_receipt(
    cfg: Config,
) -> None:
    """The measurement of 18 Aug 2026 turned into a guarantee.

    A submission that times out without a response either did not reach the broker or did.
    Retrying tells the two apart: absent, the retry places it; present, the retry is
    refused because the ``client_order_id`` is taken - and that refusal is the receipt for
    the attempt that appeared to fail. Without this the loop would report a failure for an
    order it actually has, and the position would go unbooked and unprotected.
    """
    inner = FakeBroker(prices={"AAPL": PRICE})
    quick = replace(cfg, live=replace(cfg.live, retry_backoff_seconds=0.0))
    broker = RetryingBroker(inner, quick)
    attempts = []
    real = inner.submit_market_order

    def lands_then_pretends_to_fail(symbol, quantity, side, client_order_id):
        attempts.append(client_order_id)
        if len(attempts) == 1:
            real(symbol, quantity, side, client_order_id)  # it DID reach the broker
            raise TimeoutError("no response")
        raise FakeBrokerError("client_order_id must be unique")

    inner.submit_market_order = lands_then_pretends_to_fail

    order = broker.submit_market_order("AAPL", 1.0, BUY, "20260819-AAPL")

    assert len(attempts) == 2
    assert records.client_order_id(order) == "20260819-AAPL"
    assert len([o for o in inner.orders if o.side == BUY]) == 1  # not doubled


def test_a_duplicate_refusal_on_the_FIRST_attempt_is_still_an_error(
    cfg: Config,
) -> None:
    """Only a retry can be a receipt. A first attempt refused as a duplicate means
    something else already used the id, and swallowing that would hide it."""
    inner = FakeBroker(prices={"AAPL": PRICE})
    quick = replace(cfg, live=replace(cfg.live, retry_backoff_seconds=0.0))

    def refuse(symbol, quantity, side, client_order_id):
        raise FakeBrokerError("client_order_id must be unique")

    inner.submit_market_order = refuse

    with pytest.raises(faults.Unavailable):
        RetryingBroker(inner, quick).submit_market_order("AAPL", 1.0, BUY, "d1")


# ── adoption: the loop's own fill is never left quarantined ─────────────────


def a_filled_entry(broker: FakeBroker, symbol: str, decision_id: str, quantity: float):
    """Put a filled buy at the broker under one of our decision ids, as a crash would."""
    return broker.submit_market_order(symbol, quantity, BUY, decision_id)


def test_a_position_the_loop_opened_is_adopted_not_quarantined(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """Reconciliation quarantines what it cannot explain, which is right at that layer.
    Adoption is the case where the loop CAN explain it - with a record it wrote itself
    before it sent the order."""
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    state = a_state(auto, tmp_path, broker, total=0.05)

    live_loop.run_cycle(state, NOW)  # decides nothing; withholds the entries
    second = live_loop.run_cycle(state, NOW)  # decides, submits, books

    assert second.decisions
    assert "AAPL" in state.book.managed

    # Now lose the book, as a kill -9 between submission and the book write would.
    restarted = a_state(auto, tmp_path, broker, total=0.05)
    report = live_loop.run_cycle(restarted, NOW)

    assert "AAPL" in report.adopted
    assert "AAPL" in restarted.book.managed
    assert "AAPL" not in restarted.book.unmanaged
    holding = restarted.book.managed["AAPL"]
    assert holding.stop_loss > 0 and holding.take_profit > holding.stop_loss


def test_an_adopted_position_is_protected_in_the_cycle_that_adopts_it(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """The whole reason adoption runs before step 4. A position adopted and left unarmed
    until the next poll is the failure adoption exists to remove, one cycle later."""
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    state = a_state(auto, tmp_path, broker, total=0.05)
    live_loop.run_cycle(state, NOW)
    live_loop.run_cycle(state, NOW)

    # The legs die overnight, as DAY orders do, so the restarted process finds the
    # position bare - which is the case that has to end protected.
    for order in list(broker.orders):
        if order.side == SELL:
            broker.orders.remove(order)

    restarted = a_state(auto, tmp_path, broker, total=0.05)
    report = live_loop.run_cycle(restarted, NOW)

    assert "AAPL" in report.adopted
    assert "AAPL" in report.rearmed
    armed = [
        records.client_order_id(o)
        for o in broker.orders
        if o.side == SELL and o.symbol == "AAPL"
    ]
    assert any(name.endswith("-stop") for name in armed)
    assert any(name.endswith("-target") for name in armed)


def test_a_position_with_no_decision_behind_it_stays_quarantined(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """Bought by hand in the Alpaca console. The reconciler's ruling is untouched: the
    system still refuses to manage what it cannot explain."""
    broker.prices["NVDA"] = PRICE
    a_filled_entry(broker, "NVDA", "bought-by-hand", 3.0)
    state = a_state(cfg, tmp_path, broker)

    report = live_loop.run_cycle(state, NOW)

    assert report.adopted == ()
    assert "NVDA" in state.book.unmanaged
    assert "NVDA" not in state.book.managed


def test_a_decision_id_shaped_order_with_no_record_stays_quarantined(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """The id is evidence only because the record is. A well-formed id naming nothing in
    this log is somebody else's order, or ours under a different config."""
    broker.prices["NVDA"] = PRICE
    a_filled_entry(broker, "NVDA", "20260811-NVDA", 3.0)
    state = a_state(cfg, tmp_path, broker)

    report = live_loop.run_cycle(state, NOW)

    assert report.adopted == ()
    assert "NVDA" in state.book.unmanaged


def test_the_decision_reaches_disk_before_the_order_reaches_the_broker(
    cfg: Config, stub_bars: dict, tmp_path: Path
) -> None:
    """GB-39's restart rule, asserted as an ordering rather than as an outcome.

    Written the other way round, a crash between the two leaves a position at the broker
    with nothing on disk that explains it - unadoptable by construction, quarantined for
    ever, and never protected.
    """
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    broker = FakeBroker(prices=dict.fromkeys(UNIVERSE, PRICE))
    state = a_state(auto, tmp_path, broker, total=0.05)
    live_loop.run_cycle(state, NOW)

    on_disk_at_submission = {}
    real = broker.submit_market_order

    def note_the_log(symbol, quantity, side, client_order_id):
        on_disk_at_submission[client_order_id] = [
            r.symbol for r in stored_decisions(tmp_path)
        ]
        return real(symbol, quantity, side, client_order_id)

    broker.submit_market_order = note_the_log
    live_loop.run_cycle(state, NOW)

    assert on_disk_at_submission
    for decision_id, symbols in on_disk_at_submission.items():
        assert decision_id.split("-")[1] in symbols


def test_a_kill_between_submission_and_fill_ends_booked_and_protected(
    cfg: Config, stub_bars: dict, tmp_path: Path
) -> None:
    """The restart-safety test GB-39 exists for, run as the sequence it models.

    The process submits an entry, the fill has not appeared, and the process dies before
    anything is booked - which is exactly what the poll window expiring leaves behind too.
    A new process starts against the same state directory and the same broker. It must
    not resubmit, and the position must end up managed and protected rather than
    quarantined and naked.
    """
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    unfilled = FakeBroker(prices=dict.fromkeys(UNIVERSE, PRICE), fill=False)
    dying = a_state(auto, tmp_path, unfilled, total=0.05)

    live_loop.run_cycle(dying, NOW)
    live_loop.run_cycle(dying, NOW)  # submits; nothing fills; nothing is booked
    submitted = [o for o in unfilled.orders if o.side == BUY]
    assert submitted
    assert dying.book.managed == {}  # the process dies here

    # The fills land while nobody is watching, and a new process starts.
    filled = FakeBroker(prices=dict.fromkeys(UNIVERSE, PRICE))
    for order in submitted:
        filled.submit_market_order(
            order.symbol, order.quantity, BUY, records.client_order_id(order)
        )
    restarted = a_state(auto, tmp_path, filled, total=0.05)

    report = live_loop.run_cycle(restarted, NOW)

    assert set(report.adopted) == {o.symbol for o in submitted}
    for order in submitted:
        assert order.symbol in restarted.book.managed
        assert order.symbol not in restarted.book.unmanaged
    assert report.rearmed == report.adopted  # protected in the cycle that adopted them
    # And no duplicate: the bar is decided and the client_order_id is spent.
    assert len([o for o in filled.orders if o.side == BUY]) == len(submitted)
    assert len(stored_decisions(tmp_path)) == len(UNIVERSE)
