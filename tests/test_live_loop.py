"""GB-26 acceptance: the live cycle, and every ruling it is supposed to implement.

The loop runs against ``FakeBroker`` and a synthetic universe, so the suite needs no
network, no credentials and no market. What is asserted here is not that the steps run —
that is the easy half — but that the rulings hold: completed bars only, the caller's
history floor, protection before entries, a first cycle that cannot add risk, and a stale
symbol that loses its entry but keeps its stop.
"""

from __future__ import annotations

import json
import logging
import math
import time
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
    # One protective order, and it is the stop. A stale symbol keeps the thing that bounds
    # its loss; the target is an opportunity and cannot be evaluated without a fresh bar.
    assert "limit" not in broker.submitted_kinds


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
    assert not any(name.endswith("-target") for name in armed)


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
    assert "limit" not in broker.submitted_kinds
    assert state.book.managed  # taken into the book in the same cycle
    assert set(state.entry_fills) == set(state.book.managed)


# ── GB-26 / 26 Aug 2026: the target is the loop's, not the broker's ─────────


def a_managed_position(
    state: live_loop.LiveState,
    symbol: str = "AAPL",
    *,
    take_profit: float,
    quantity: float = 3.0,
) -> None:
    state.broker.positions[symbol] = quantity
    state.book = Book(
        managed={
            symbol: Holding(
                symbol=symbol,
                quantity=quantity,
                decision_id="d1",
                entry_price=PRICE,
                stop_loss=97.0,
                take_profit=take_profit,
            )
        }
    )


def test_a_target_reached_on_the_completed_bar_closes_the_position(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """The half of the ruling that is a capability rather than a removal.

    The take-profit stopped being an order at the broker because a working sell holds the
    whole position, so it has to be a level the loop watches. Without this the live system
    would have no target exit at all - and `backtest.engine`'s `target_in_loop` arm, which
    is now the study's default, would describe a system nobody built.
    """
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    state = a_state(auto, tmp_path, broker)
    high = float(stub_bars["frames"]["AAPL"]["high"].iloc[-1])
    a_managed_position(state, take_profit=high - 1.0)

    report = live_loop.run_cycle(state, NOW)

    sells = [
        o
        for o in broker.orders
        if o.symbol == "AAPL"
        and o.side == SELL
        and records.client_order_id(o).endswith(records.TARGET_IN_LOOP_SUFFIX)
    ]
    assert len(sells) == 1, "the loop-side target did not close the position"
    assert report.submissions


def test_a_target_not_yet_reached_leaves_the_position_alone(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    state = a_state(auto, tmp_path, broker)
    high = float(stub_bars["frames"]["AAPL"]["high"].iloc[-1])
    a_managed_position(state, take_profit=high + 10.0)

    live_loop.run_cycle(state, NOW)

    assert not any(
        records.client_order_id(o).endswith(records.TARGET_IN_LOOP_SUFFIX)
        for o in broker.orders
    )


def test_the_target_exit_is_sent_once_however_many_cycles_run(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """A market sell repeated every 60s would sell a position the loop no longer holds.

    Answered from the broker's order history rather than a flag, for the reason
    `_send_exits` is: the id is a function of the position, so a restart reads it correctly
    where a flag in memory would say "not sent" and send a second one.
    """
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    state = a_state(auto, tmp_path, broker)
    high = float(stub_bars["frames"]["AAPL"]["high"].iloc[-1])
    a_managed_position(state, take_profit=high - 1.0)

    live_loop.run_cycle(state, NOW)
    live_loop.run_cycle(state, NOW)
    live_loop.run_cycle(state, NOW)

    sent = [
        o
        for o in broker.orders
        if records.client_order_id(o).endswith(records.TARGET_IN_LOOP_SUFFIX)
    ]
    assert len(sent) == 1


def test_a_stale_symbol_does_not_take_a_target_exit(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """The stale-data ruling, applied to the new path: no fresh bar, no target.

    The asymmetry is deliberate and it is the same one as before - a stale window may not
    justify acting on an opportunity, and may not suspend the stop that bounds the loss.
    """
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    high = float(stub_bars["frames"]["MSFT"]["high"].iloc[-1])
    stub_bars["frames"]["MSFT"] = stub_bars["frames"]["MSFT"].iloc[:-3]
    state = a_state(auto, tmp_path, broker)
    a_managed_position(state, "MSFT", take_profit=high - 50.0)

    report = live_loop.run_cycle(state, NOW)

    assert "MSFT" in report.stale
    assert "MSFT" in report.rearmed, "a stale symbol still keeps its stop"
    assert not any(
        records.client_order_id(o).endswith(records.TARGET_IN_LOOP_SUFFIX)
        for o in broker.orders
    )


def test_the_target_exit_cancels_the_stop_before_selling(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """The defect that cost 13 consecutive cycles on 25 Aug 2026, in its new place.

    A working sell holds the whole position, so a market sell placed while the stop is live
    is refused with `insufficient qty available`. That is the same constraint that removed
    the limit leg, and the target exit is the newest thing that could trip over it.
    """
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    state = a_state(auto, tmp_path, broker)
    high = float(stub_bars["frames"]["AAPL"]["high"].iloc[-1])
    a_managed_position(state, take_profit=high - 1.0)
    live_loop._arm_leg(
        state, "AAPL", state.book.managed["AAPL"], live_loop.STOP_LEG, lambda m: None
    )

    live_loop.run_cycle(state, NOW)

    assert broker.positions.get("AAPL", 0.0) == pytest.approx(0.0)
    assert not [
        o
        for o in broker.orders
        if o.symbol == "AAPL" and o.side == SELL and o.status not in live_loop.FINISHED
    ]


def test_a_limit_leg_is_refused_by_the_armer_rather_than_by_the_broker(
    cfg: Config, broker: FakeBroker, tmp_path: Path
) -> None:
    """Reintroducing the second leg must fail here, not at Alpaca.

    At the broker it fails as `insufficient qty available`, which the loop counts as an
    arming failure - and `ARMING_STRIKES` of those flatten a perfectly healthy position.
    That is how a 97.38-share NVDA position was liquidated on 25 Aug 2026.
    """
    state = a_state(cfg, tmp_path, broker)
    a_managed_position(state, take_profit=106.0)

    with pytest.raises(live_loop.LiveError, match="only protective order"):
        live_loop._arm_leg(
            state,
            "AAPL",
            state.book.managed["AAPL"],
            live_loop.TARGET_LEG,
            lambda m: None,
        )


def test_the_high_column_is_the_one_the_canonical_frame_carries(cfg: Config) -> None:
    """`HIGH` is a second copy of a column name that `data.historical` owns. Pinned rather
    than trusted: a rename there would make the target silently unreachable, and nothing
    else in the loop reads a price column."""
    from glassbox.data.historical import OHLCV_COLUMNS

    assert live_loop.HIGH in OHLCV_COLUMNS


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


# ── the overnight residual, on the page rather than in a document ────────────


def a_holding(symbol: str = "AAPL") -> Holding:
    return Holding(
        symbol=symbol,
        quantity=1.5,
        decision_id=f"20260824-{symbol}",
        entry_price=100.0,
        stop_loss=98.0,
        take_profit=104.0,
    )


def test_a_session_that_opens_flat_prints_no_residual_warning() -> None:
    """A warning that fires when there is nothing to warn about is one nobody reads."""
    assert live_loop.overnight_residual(Book(), NOW) == []


def test_a_session_that_opens_holding_states_the_gap_and_how_it_is_handled() -> None:
    """**The ruling of 23 Aug is that the legs stay DAY and the residual is reported**, so
    a GATE 2 log has to say how the gap was handled rather than leave it as a footnote.
    """
    book = Book(managed={"AAPL": a_holding()})

    lines = " ".join(live_loop.overnight_residual(book, NOW))

    assert "OVERNIGHT RESIDUAL" in lines
    assert "20260824-AAPL" in lines
    assert "NONE between the previous close" in lines
    assert "TimeInForce.DAY" in lines
    assert "rule 2 re-arms" in lines


def test_the_residual_reaches_the_banner(
    cfg: Config, broker: FakeBroker, tmp_path: Path
) -> None:
    state = a_state(cfg, tmp_path, broker)
    state.book = Book(managed={"MSFT": a_holding("MSFT")})

    assert "OVERNIGHT RESIDUAL" in live_loop.session_banner(state, NOW, dry_run=False)


def test_the_session_summary_answers_whether_risk_was_carried_across_a_close() -> None:
    """The cycle records cannot answer it: a position re-armed in cycle 1 looks exactly
    like one opened in cycle 1."""
    carried = live_loop.SessionReport(
        session_id="s",
        banner="",
        cycles=(),
        open_orders=(),
        stopped_by="the market closed",
        held_at_open=("AAPL",),
    )
    flat = live_loop.SessionReport(
        session_id="s", banner="", cycles=(), open_orders=(), stopped_by="x"
    )

    assert "unprotected from the open" in carried.summary()
    assert "AAPL" in carried.summary()
    assert "the session opened flat" in flat.summary()


# ── the configuration can move under a long run ──────────────────────────────


def test_no_drift_when_the_running_config_is_the_one_on_disk() -> None:
    assert live_loop.config_drift(load_config()) == []


def test_a_live_only_drift_says_the_model_is_unchanged() -> None:
    """The loop should say what moved, because "config changed" and "your model is not
    the model" are different emergencies."""
    stale = replace(load_config(), live=replace(load_config().live, poll_seconds=99999))

    lines = " ".join(live_loop.config_drift(stale))

    assert "CONFIG DRIFT" in lines
    assert "model settings    : unchanged" in lines
    assert "NOT reloaded" in lines


def test_a_model_shaping_drift_says_the_running_model_is_wrong() -> None:
    stale = replace(load_config(), window=replace(load_config().window, input_len=999))

    lines = " ".join(live_loop.config_drift(stale))

    assert "CHANGED" in lines
    assert "not the one this configuration would build" in lines


def test_an_unreadable_settings_file_is_reported_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A settings file that has become unreadable during a run is not evidence that
    nothing changed."""

    def missing(*args, **kwargs):
        raise FileNotFoundError("settings file not found: nowhere.yaml")

    running = load_config()
    monkeypatch.setattr(live_loop, "load_config", missing)

    lines = " ".join(live_loop.config_drift(running))

    assert "cannot be read" in lines
    assert "loaded at startup, unchanged" in lines


def test_the_session_summary_says_whether_the_configuration_drifted() -> None:
    drifted = live_loop.SessionReport(
        session_id="s",
        banner="",
        cycles=(),
        open_orders=(),
        stopped_by="x",
        config_drifted=True,
    )
    steady = live_loop.SessionReport(
        session_id="s", banner="", cycles=(), open_orders=(), stopped_by="x"
    )

    assert "DRIFTED" in drifted.summary()
    assert "matches the file on disk" in steady.summary()


# ── a run that spans more than one session ───────────────────────────────────


class AdvancingClock:
    """A clock the loop moves by sleeping. Seventeen idle hours cost seventeen calls."""

    def __init__(self, start: pd.Timestamp) -> None:
        self.now = start

    def __call__(self) -> pd.Timestamp:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now = self.now + pd.Timedelta(seconds=seconds)


def coarse(cfg: Config, seconds: int = 3600) -> Config:
    """The same loop at one cycle an hour, so a whole session fits in a test."""
    return replace(
        cfg,
        live=replace(cfg.live, poll_seconds=seconds, heartbeat_seconds=seconds),
    )


def test_idle_state_tells_a_weekend_from_a_holiday_from_an_evening(
    cfg: Config,
) -> None:
    """Three different facts about the same silence, and a reader of a night of
    heartbeats needs to know which one they are looking at."""
    assert live_loop.idle_state(cfg, pd.Timestamp("2026-08-23 12:00", tz="UTC")) == (
        "weekend"
    )
    assert live_loop.idle_state(cfg, pd.Timestamp("2026-12-25 12:00", tz="UTC")) == (
        "holiday"
    )
    assert live_loop.idle_state(cfg, pd.Timestamp("2026-08-24 06:00", tz="UTC")) == (
        "outside session"
    )


def test_the_heartbeat_says_when_what_and_for_how_long(
    cfg: Config, tmp_path: Path
) -> None:
    """**A loop that died at 02:00 and a loop correctly idling produce identical output:
    nothing.** The heartbeat is what makes silence mean dead rather than quiet."""
    Book(managed={"AAPL": a_holding()}).save(tmp_path / "book.json")
    started = pd.Timestamp("2026-08-24 20:00", tz="UTC")

    line = live_loop.heartbeat(
        cfg, tmp_path, started + pd.Timedelta(hours=27, minutes=14), started
    )

    assert line.startswith("heartbeat 2026-08-25T23:14:00Z")
    assert "state=outside session" in line
    assert "positions=AAPL" in line
    assert "uptime=1d 03:14:00" in line


def test_the_heartbeat_says_none_rather_than_nothing_when_flat(
    cfg: Config, tmp_path: Path
) -> None:
    line = live_loop.heartbeat(cfg, tmp_path, NOW, NOW)

    assert "positions=none" in line
    assert "uptime=0d 00:00:00" in line


def test_two_sessions_in_one_process_are_reported_separately(
    cfg: Config,
    broker: FakeBroker,
    stub_bars: dict,
    stub_predictor: None,
    tmp_path: Path,
) -> None:
    """GATE 2 criterion 6 needs two or three, and one process across all of them is what
    removes the half of the overnight residual rule 2 cannot cover.

    They are **not merged**: each report carries its own ``held_at_open``, which is the
    answer to *did this session begin holding risk*, and one merged report cannot say it
    twice.
    """
    clock = AdvancingClock(pd.Timestamp("2026-08-24 14:00", tz="UTC"))

    reports = live_loop.run_sessions(
        coarse(cfg),
        tmp_path,
        sessions=2,
        broker=broker,
        clock=clock,
        sleep=clock.sleep,
    )

    assert len(reports) == 2
    assert all(report.cycles for report in reports)
    assert all(isinstance(report.held_at_open, tuple) for report in reports)
    # The second ran on the next exchange day, not twice on the same one.
    assert reports[0].cycles[0].at.date() < reports[1].cycles[0].at.date()


def test_the_session_that_opens_holding_says_so_and_the_next_one_does_not(
    cfg: Config,
    broker: FakeBroker,
    stub_bars: dict,
    stub_predictor: None,
    tmp_path: Path,
) -> None:
    """`held_at_open` is read **before** the first reconciliation, which is the only
    moment the book still describes what was carried *into* the session."""
    Book(managed={"AAPL": a_holding()}).save(tmp_path / "book.json")
    clock = AdvancingClock(pd.Timestamp("2026-08-24 14:00", tz="UTC"))

    reports = live_loop.run_sessions(
        coarse(cfg),
        tmp_path,
        sessions=2,
        broker=broker,
        clock=clock,
        sleep=clock.sleep,
    )

    assert reports[0].held_at_open == ("AAPL",)
    assert "unprotected from the open" in reports[0].summary()


def test_the_loop_proves_it_is_alive_while_it_waits_for_the_next_session(
    cfg: Config,
    broker: FakeBroker,
    stub_bars: dict,
    stub_predictor: None,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = AdvancingClock(pd.Timestamp("2026-08-24 21:00", tz="UTC"))

    with caplog.at_level(logging.INFO, logger="glassbox.live_loop"):
        reports = live_loop.run_sessions(
            coarse(cfg),
            tmp_path,
            sessions=1,
            broker=broker,
            clock=clock,
            sleep=clock.sleep,
        )

    beats = [line for line in caplog.messages if line.startswith("heartbeat ")]
    assert len(reports) == 1
    assert len(beats) >= 8  # the evening and the night before the next open
    assert any("state=outside session" in beat for beat in beats)


def test_the_idle_loop_leaves_evidence_a_program_can_read(
    cfg: Config,
    broker: FakeBroker,
    stub_bars: dict,
    stub_predictor: None,
    tmp_path: Path,
) -> None:
    """**The log line is for a person; the file is for the dashboard.** The book is
    persisted only inside a session, so before 23 Sep 2026 a loop idling correctly
    overnight left nothing on disk for seventeen hours - and the console, ageing it from
    book.json's mtime, called a healthy loop NOT RESPONDING at the next open.
    """
    clock = AdvancingClock(pd.Timestamp("2026-08-24 21:00", tz="UTC"))

    live_loop.run_sessions(
        coarse(cfg),
        tmp_path,
        sessions=1,
        broker=broker,
        clock=clock,
        sleep=clock.sleep,
    )

    beat = json.loads((tmp_path / live_loop.HEARTBEAT_FILE).read_text(encoding="utf-8"))
    # The last write wins, and the loop ran into the next session, so the final beat is
    # the one from inside it. What matters is that the idle passes wrote theirs too.
    assert pd.Timestamp(beat["at"]).tzinfo is not None
    assert beat["state"] in {"in session", "outside session", "weekend", "holiday"}


def test_an_idle_pass_writes_its_heartbeat_before_it_sleeps(
    cfg: Config, tmp_path: Path
) -> None:
    """The idle branch on its own, with no session to reach: one pass, one file.

    The log line is throttled to ``heartbeat_seconds`` because a night of output has to
    stay readable; the file is not, because the dashboard asks *how long ago* and every
    skipped write makes that answer worse.
    """
    stop = RuntimeError("one pass is enough")
    clock = AdvancingClock(pd.Timestamp("2026-08-23 12:00", tz="UTC"))  # a Sunday

    def sleep_once(seconds: float) -> None:
        raise stop

    with pytest.raises(RuntimeError):
        live_loop.run_sessions(
            coarse(cfg), tmp_path, sessions=1, clock=clock, sleep=sleep_once
        )

    beat = json.loads((tmp_path / live_loop.HEARTBEAT_FILE).read_text(encoding="utf-8"))
    assert beat["at"] == "2026-08-23T12:00:00Z"
    assert beat["state"] == "weekend"


def test_the_heartbeat_is_replaced_atomically_and_never_seen_half_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**A reader can arrive at any moment**, and the dashboard reads this file every few
    seconds. A half-written JSON object is unparseable evidence, which the dashboard is
    obliged to read as NOT RESPONDING - an alarm raised by the act of writing a heartbeat.

    The spy stands where ``os.replace`` does: at the instant of the swap it reads what a
    reader would see at the destination, which must be the *previous* beat, whole. A write
    that went straight to the destination would never reach the spy at all.
    """
    first = pd.Timestamp("2026-09-23 10:00", tz="UTC")
    second = first + pd.Timedelta(seconds=60)
    live_loop.write_heartbeat(tmp_path, first, "outside session")

    seen: list[dict] = []
    real_replace = live_loop.os.replace

    def spy(src, dst):
        seen.append(json.loads(Path(dst).read_text(encoding="utf-8")))
        return real_replace(src, dst)

    monkeypatch.setattr(live_loop.os, "replace", spy)
    live_loop.write_heartbeat(tmp_path, second, "outside session")

    assert seen == [{"at": "2026-09-23T10:00:00Z", "state": "outside session"}]
    final = json.loads(
        (tmp_path / live_loop.HEARTBEAT_FILE).read_text(encoding="utf-8")
    )
    assert final["at"] == "2026-09-23T10:01:00Z"
    assert not list(
        tmp_path.glob(f"{live_loop.HEARTBEAT_FILE}.*.tmp")
    ), "temp left behind"


def test_idling_produces_no_cycles_and_so_no_records(
    cfg: Config,
    broker: FakeBroker,
    stub_bars: dict,
    stub_predictor: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**A heartbeat every fifteen minutes is fine; a cycle record every sixty seconds
    while the market is shut is not.** Seventeen idle hours must cost the decision store
    nothing, so the count of cycles run must equal the count of cycles reported.
    """
    calls = {"n": 0}
    real = live_loop.run_cycle

    def counted(state, when):
        calls["n"] += 1
        return real(state, when)

    monkeypatch.setattr(live_loop, "run_cycle", counted)
    clock = AdvancingClock(pd.Timestamp("2026-08-24 21:00", tz="UTC"))

    reports = live_loop.run_sessions(
        coarse(cfg),
        tmp_path,
        sessions=1,
        broker=broker,
        clock=clock,
        sleep=clock.sleep,
    )

    assert calls["n"] == sum(len(report.cycles) for report in reports)
    assert calls["n"] > 0


def test_a_run_stopped_for_a_reason_does_not_start_again_tomorrow(
    cfg: Config,
    broker: FakeBroker,
    stub_bars: dict,
    stub_predictor: None,
    tmp_path: Path,
) -> None:
    """Only an ordinary close continues to the next session."""
    clock = AdvancingClock(pd.Timestamp("2026-08-24 14:00", tz="UTC"))

    reports = live_loop.run_sessions(
        coarse(cfg),
        tmp_path,
        sessions=3,
        broker=broker,
        clock=clock,
        sleep=clock.sleep,
        max_cycles=1,
    )

    assert len(reports) == 1
    assert reports[0].stopped_by == "max_cycles=1"


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
    assert not any(name.endswith("-target") for name in armed)


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


# ── GB-40: the rehearsal proves the machine and never the model ─────────────


def a_rehearsal(notional: float = 25.0, close_out: int = 15) -> live_loop.Rehearsal:
    return live_loop.Rehearsal(
        reason="gate2-execution-path",
        notional=notional,
        close_out_minutes=close_out,
    )


def rehearsing(cfg: Config, tmp_path: Path, broker: FakeBroker, total: float = 0.05):
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    state = a_state(auto, tmp_path, broker, total=total)
    state.rehearsal = a_rehearsal()
    state.provenance = state.rehearsal.provenance
    state.thresholds = live_loop.permissive_band()
    return state


def test_a_rehearsal_never_writes_a_live_record(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """Condition 1. A rehearsal that looked live in the log would be worse than none."""
    state = rehearsing(cfg, tmp_path, broker)

    live_loop.run_cycle(state, NOW)
    live_loop.run_cycle(state, NOW)

    assert records.load_decisions(MONTH, NOW, tmp_path) == []
    everything = records.load_decisions(MONTH, NOW, tmp_path, records.ANY_PROVENANCE)
    assert everything
    assert all(records.is_rehearsal(r.provenance) for r in everything)
    assert all("gate2-execution-path" in r.provenance for r in everything)


def test_nothing_a_rehearsal_produces_is_reportable(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """Condition 2, asserted on GB-19's two inputs: the decision log and the trade log."""
    state = rehearsing(cfg, tmp_path, broker)
    live_loop.run_cycle(state, NOW)
    second = live_loop.run_cycle(state, NOW)

    everything = records.load_decisions(MONTH, NOW, tmp_path, records.ANY_PROVENANCE)
    assert everything
    assert not any(records.is_reportable(r.provenance) for r in everything)
    # And no `Trade` is emitted at all - `Trade` carries no provenance, so the only way to
    # keep a rehearsal out of the study's trade log is never to put one in.
    assert second.trades == ()


def test_a_rehearsal_emits_no_trade_even_when_a_sell_fills(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """The case the previous test cannot reach on its own: a filled sell is exactly what
    `emit_trades` exists to turn into a Trade."""
    state = rehearsing(cfg, tmp_path, broker, total=-0.05)
    broker.positions["AAPL"] = 4.0
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
    state.entry_fills["AAPL"] = ["2026-08-11T13:30:00+00:00", PRICE, 0.0, "e1"]

    first = live_loop.run_cycle(state, NOW)
    second = live_loop.run_cycle(state, NOW)

    assert any(order.side == SELL for order in broker.orders)
    assert first.trades == () and second.trades == ()


def test_a_rehearsal_order_is_capped_at_its_notional(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """Condition 4. The sizer is left alone and its output is capped: the point of the
    rehearsal is that the ordinary path runs."""
    state = rehearsing(cfg, tmp_path, broker)

    live_loop.run_cycle(state, NOW)
    second = live_loop.run_cycle(state, NOW)

    placed = [s for s in second.submissions if s.entry is not None]
    assert placed
    for submission in placed:
        assert submission.order.notional == pytest.approx(25.0)
        # The sizer's own price, not the fixture's: the cap scales the order the ordinary
        # path produced rather than re-deriving one.
        assert submission.order.shares == pytest.approx(25.0 / submission.order.price)


def test_a_rehearsal_flattens_before_the_close_and_opens_nothing(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """Condition 3, which is Ben's DAY-legs condition applied where it bites.

    A rehearsal that holds overnight fails the rehearsal: the protective legs expire at
    the close, so the position would be unprotected in exactly the window the backtest
    models as protected.
    """
    state = rehearsing(cfg, tmp_path, broker)
    live_loop.run_cycle(state, NOW)
    live_loop.run_cycle(state, NOW)
    assert state.book.managed

    _, closes = live_loop.market_session(cfg, NOW)
    near_close = closes - pd.Timedelta(minutes=5)
    report = live_loop.run_cycle(state, near_close)

    closed_out = [
        order
        for order in broker.orders
        if records.client_order_id(order).endswith("-closeout")
    ]
    assert closed_out
    assert report.decisions == ()  # nothing new is opened in the close-out window


def test_a_rehearsal_stopped_cleanly_mid_session_leaves_nothing_behind(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """**Condition 3 has a hole the close-out window does not cover** (found 24 Aug 2026).

    ``_rehearsal_close_out`` fires only once the clock reaches ``close_out_minutes``
    before the exchange close. The GATE 2 plan stops the rehearsal **five hours earlier**
    - prove the execution path at the open, stop, restart under the deployed band - and a
    rehearsal stopped there was flattened by nothing at all. The position persisted in the
    saved book, the protective legs were DAY orders that expire at the close, and the
    deployed session restarted on top of a position a rehearsal had opened.

    *A rehearsal that holds overnight fails the rehearsal* has to mean any stop, not only
    the one the clock reaches on its own.
    """
    state = rehearsing(cfg, tmp_path, broker)
    live_loop.run_cycle(state, NOW)
    live_loop.run_cycle(state, NOW)
    assert state.book.managed, "the rehearsal opened nothing; harness problem"

    _, closes = live_loop.market_session(cfg, NOW)
    assert NOW < closes - pd.Timedelta(minutes=15)  # nowhere near the close-out window

    live_loop.close_out_on_stop(state, NOW, log=lambda message: None)

    closed_out = [
        order
        for order in broker.orders
        if records.client_order_id(order).endswith("-closeout")
    ]
    assert (
        closed_out
    ), "a clean mid-session stop left the rehearsal position at the broker"
    assert not state.book.managed


class _UnreachableOnOrders:
    """A broker that answers nothing, the way DNS failure does.

    Wraps a real `FakeBroker` and fails only `get_orders`, because that is the call that
    actually raised on 27 Aug 2026: `release_protective_legs` reads the order list before
    it cancels anything, so the close-out died one line before the guarded `try`.
    """

    def __init__(self, inner: FakeBroker) -> None:
        self._inner = inner

    def get_orders(self):
        raise faults.Unavailable(
            "get_orders failed after 3 attempts: getaddrinfo failed"
        )

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_a_close_out_that_cannot_reach_the_broker_reports_instead_of_raising(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """**The defect of 27 Aug 2026, and it cost an unprotected overnight hold.**

    `close_out_on_stop` runs in `run_session`'s `finally`. Every other broker path in this
    loop treats an unreachable broker as a skipped cycle; this one let `Unavailable` out,
    so it escaped `run_session`, escaped `main`, and the process died with a traceback
    instead of a session report. Two rehearsal positions were left open at the broker, and
    their DAY stops expired at the close with no process alive to re-arm them.

    The close-out must therefore be *reporting* code, not raising code: whatever it cannot
    close, it names.
    """
    state = rehearsing(cfg, tmp_path, broker)
    live_loop.run_cycle(state, NOW)
    live_loop.run_cycle(state, NOW)
    assert state.book.managed, "the rehearsal opened nothing; harness problem"
    held = sorted(state.book.managed)

    state.broker = _UnreachableOnOrders(broker)
    failed = live_loop.close_out_on_stop(state, NOW, log=lambda message: None)

    assert sorted(failed) == held, "the unreachable close-out must name every symbol"
    assert sorted(state.book.managed) == held, (
        "a position that was not sold must stay in the book - a position the book has "
        "forgotten is worse than one it still shows"
    )


def test_one_symbol_that_cannot_be_closed_does_not_stop_the_others(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """The close-out is the last thing between a rehearsal and an overnight hold, so it
    tries every symbol even after one has failed."""
    state = rehearsing(cfg, tmp_path, broker)
    live_loop.run_cycle(state, NOW)
    live_loop.run_cycle(state, NOW)
    held = sorted(state.book.managed)
    if len(held) < 2:
        pytest.skip("needs two open rehearsal positions to say anything")

    doomed = held[0]
    real_submit = broker.submit_market_order

    def refuse_one(*, symbol: str, **kwargs):
        if symbol == doomed:
            raise faults.Unavailable("submit_market_order failed after 3 attempts")
        return real_submit(symbol=symbol, **kwargs)

    broker.submit_market_order = refuse_one
    failed = live_loop.close_out_on_stop(state, NOW, log=lambda message: None)

    assert list(failed) == [doomed]
    assert doomed in state.book.managed
    for symbol in held[1:]:
        assert (
            symbol not in state.book.managed
        ), "a reachable symbol was left unflattened"


def test_the_process_exits_non_zero_naming_what_it_could_not_flatten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bare_root_logger
) -> None:
    """**A failed close-out must produce a record, not a traceback - and not a green exit.**

    A wrapper reading `$?` learned nothing on 27 Aug: the process died on an unhandled
    exception, which is indistinguishable from any other crash, and the one fact that
    mattered - two positions are open and nothing is managing them - reached no channel at
    all. The report names them and so does the exit status.
    """
    report = live_loop.SessionReport(
        session_id="s",
        banner="",
        cycles=(),
        open_orders=(),
        stopped_by="the market closed",
        close_out_failed=("GOOGL", "NVDA"),
    )
    monkeypatch.setattr(live_loop, "run_sessions", lambda *a, **k: (report,))

    code = live_loop.main(
        ["--log-dir", str(tmp_path), "--state-dir", str(tmp_path / "state")]
    )

    assert code == live_loop.EXIT_CLOSE_OUT_FAILED
    assert code != 0

    # Asserted against the file rather than `caplog`, and not by preference: `main` owns
    # the root logger and calls `basicConfig(force=True)`, which drops caplog's handler.
    # The file is the channel that matters anyway - it is what the gate reads.
    day = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    written = (tmp_path / f"live-{day}.log").read_text(encoding="utf-8")
    assert "GOOGL" in written and "NVDA" in written
    assert "RISK EVENT" in written
    assert "Close by hand" in written


def test_a_clean_close_out_still_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bare_root_logger
) -> None:
    """The other half: the new exit code must mean what it says, not fire on every run."""
    report = live_loop.SessionReport(
        session_id="s",
        banner="",
        cycles=(),
        open_orders=(),
        stopped_by="the market closed",
        closed_out=("GOOGL",),
    )
    monkeypatch.setattr(live_loop, "run_sessions", lambda *a, **k: (report,))

    assert (
        live_loop.main(
            ["--log-dir", str(tmp_path), "--state-dir", str(tmp_path / "state")]
        )
        == 0
    )


def test_a_rehearsal_writes_no_trade_log(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """Condition 2 of the rehearsal, at the one point that can enforce it.

    A rehearsal emits no Trade at all - ``Trade`` carries no provenance and cannot, since
    the contract is frozen - so the gate is that the tuple is empty before it ever reaches
    ``records.save_trades``, not a filter inside it. Now that trades are persisted, that
    gate is the only thing standing between a rehearsal and the study's trade log.
    """
    state = rehearsing(cfg, tmp_path, broker)
    live_loop.run_cycle(state, NOW)
    live_loop.run_cycle(state, NOW)

    assert not (tmp_path / records.TRADES_FILE).exists()


def test_a_deployed_session_stopped_cleanly_does_NOT_flatten(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """**The other half, and it is the half that protects the study.**

    Flattening on stop is right for a rehearsal and wrong for the deployed loop. The
    backtest holds overnight, so a live loop that flattened at every stop would be running
    a different strategy from the one being evaluated - the parity gap the 23 Aug DAY/GTC
    ruling refused to open. ``close_out_on_stop`` is a no-op unless a rehearsal is active.
    """
    state = a_state(
        replace(cfg, live=replace(cfg.live, mode="auto")), tmp_path, broker, total=0.05
    )
    state.thresholds = live_loop.permissive_band()
    live_loop.run_cycle(state, NOW)
    live_loop.run_cycle(state, NOW)
    held = sorted(state.book.managed)
    assert held, "nothing opened; harness problem"

    live_loop.close_out_on_stop(state, NOW, log=lambda message: None)

    assert sorted(state.book.managed) == held
    assert not [
        order
        for order in broker.orders
        if records.client_order_id(order).endswith("-closeout")
    ]


def test_run_session_closes_out_on_every_stop_path(
    cfg: Config,
    broker: FakeBroker,
    stub_bars: dict,
    stub_predictor: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**The wiring, asserted so it cannot pass vacuously.**

    The first version of this test ran a rehearsal session to its cycle cap and asserted
    the broker held nothing. It passed with the wiring **removed**, because that session
    never opened a position at all - cycle 1 withholds entries until protection has been
    verified once, and later cycles do not re-decide the same bar. An assertion about an
    empty broker is worthless when the broker was always going to be empty. So this asserts
    the *call*, on the stop path, with the state it was given.

    :func:`test_a_rehearsal_stopped_cleanly_mid_session_leaves_nothing_behind` proves the
    function does the right thing to a state that **is** holding something. Together they
    close the chain; neither alone does.
    """
    seen: list[tuple[str, ...]] = []
    real = live_loop.close_out_on_stop

    def spy(state, when, log=lambda message: None):
        seen.append(tuple(sorted(state.book.managed)))
        return real(state, when, log)

    monkeypatch.setattr(live_loop, "close_out_on_stop", spy)
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))

    report = live_loop.run_session(
        auto,
        tmp_path,
        broker=broker,
        clock=lambda: NOW,
        sleep=lambda s: None,
        max_cycles=3,
        rehearsal=a_rehearsal(),
    )

    assert report.stopped_by == "max_cycles=3"
    assert len(seen) == 1, "the stop path did not call close_out_on_stop exactly once"


def test_run_session_closes_out_even_when_the_session_raises(
    cfg: Config,
    broker: FakeBroker,
    stub_bars: dict,
    stub_predictor: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It sits in the ``finally``, so an unexpected exception cannot skip it.

    The ordinary stop paths are the ones a plan uses; this is the one nobody plans for, and
    it is the one where a stranded position would be least expected and least noticed.
    """
    called: list[bool] = []
    monkeypatch.setattr(
        live_loop,
        "close_out_on_stop",
        lambda state, when, log=lambda m: None: called.append(True) or (),
    )

    def explode(state, now):
        raise RuntimeError("cycle blew up")

    monkeypatch.setattr(live_loop, "run_cycle", explode)
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))

    with pytest.raises(RuntimeError, match="cycle blew up"):
        live_loop.run_session(
            auto,
            tmp_path,
            broker=broker,
            clock=lambda: NOW,
            sleep=lambda s: None,
            rehearsal=a_rehearsal(),
        )

    assert called, "an exception escaped without the rehearsal being closed out"


def test_the_banner_says_the_run_is_a_rehearsal_and_why_it_is_not_reportable(
    cfg: Config, broker: FakeBroker, tmp_path: Path
) -> None:
    state = rehearsing(cfg, tmp_path, broker)

    banner = live_loop.session_banner(state, NOW, dry_run=False)

    assert "REHEARSAL" in banner
    assert "gate2-execution-path" in banner
    assert "DELIBERATELY PERMISSIVE" in banner
    assert "never 'live'" in banner
    assert "reportable         : NO" in banner


def test_the_permissive_band_could_not_be_mistaken_for_a_calibration() -> None:
    band = live_loop.permissive_band()

    assert band.fires
    assert band.lower == live_loop.PERMISSIVE_LOWER
    assert (
        band.lower > 0.0
    )  # Thresholds refuses zero, which is the contract having teeth


def test_a_rehearsal_must_state_its_reason() -> None:
    """It goes into every record, so an empty one would produce a provenance that says
    nothing about why the decision is not live."""
    with pytest.raises(ValueError, match="must state its reason"):
        records.rehearsal_provenance("   ")


def test_replay_and_rehearsal_are_both_unreportable_and_live_is_not() -> None:
    assert records.is_reportable(records.LIVE)
    assert not records.is_reportable(records.replay_provenance(13))
    assert not records.is_reportable(records.rehearsal_provenance("gate2"))


def test_a_dry_run_may_not_use_the_deployed_state_directory(cfg: Config) -> None:
    """**A verification step must not be able to damage what it verifies** (24 Aug 2026).

    Two reasons, and the first one nearly bit. Validating the read timeout meant running a
    dry run while the GATE 2 session was live, and both default to `checkpoints/live` -
    two processes writing one book and one decision log, so the check would have corrupted
    the session it existed to protect. Copying the state directory was the fix, and a
    runbook line saying "remember to copy it" is a note; this is the mechanism.

    The second reason is worse and is not about concurrency at all. `--dry-run` refuses
    broker *writes* and does **not** change provenance, so a decision it records is written
    as `live` and is indistinguishable from a real one - `records.is_reportable` would admit
    it into the study. That is a defect in its own right and is recorded as one; refusing
    the deployed directory keeps it out of the log that matters until it is fixed.
    """
    with pytest.raises(live_loop.LiveError, match="may not use the deployed state"):
        live_loop.run_session(cfg, live_loop.DEFAULT_STATE_DIR, dry_run=True)


def test_a_dry_run_against_a_copy_is_allowed(
    cfg: Config,
    broker: FakeBroker,
    stub_bars: dict,
    stub_predictor: None,
    tmp_path: Path,
) -> None:
    """The guard names the deployed directory, not dry runs. Pointed at a copy it is the
    intended way to exercise the live path without touching the account or the log."""
    report = live_loop.run_session(
        cfg,
        tmp_path,
        broker=broker,
        dry_run=True,
        clock=lambda: NOW,
        sleep=lambda s: None,
        max_cycles=1,
    )

    assert report.stopped_by == "max_cycles=1"


def test_every_dry_run_says_its_records_carry_live_provenance(
    cfg: Config, broker: FakeBroker, tmp_path: Path
) -> None:
    """**Unconditional, because the containment it backs up is not one** (24 Aug 2026).

    `run_session` refuses a dry run against `checkpoints/live`, which covers the
    concurrency case. It does not cover a dry run against a *copied* directory - which
    still records `live` provenance, still produces records `records.is_reportable` admits,
    and is exactly what a tired person runs on a Wednesday. "Tomorrow's runs are not dry" is
    a fact about intention; this is the line that survives the person who forgot.

    Removed when dry runs get `dry:<reason>` the way rehearsals got `rehearsal:<reason>`
    and `is_reportable` rejects both. Until then the banner is the only thing standing
    between a dry run and the study's inputs.
    """
    state = a_state(cfg, tmp_path, broker)
    banner = live_loop.session_banner(state, NOW, dry_run=True)

    assert "DRY RUN" in banner
    assert "KNOWN DEFECT" in banner
    assert "INDISTINGUISHABLE" in banner
    assert "Do not analyse or report" in banner
    assert "DECISIONS.md" in banner
    # and it must not appear when the run is real
    assert "KNOWN DEFECT" not in live_loop.session_banner(state, NOW, dry_run=False)


# ── the session log has to outlive the terminal ──────────────────────────────


@pytest.fixture
def bare_root_logger():
    """A root logger with no handlers, because `basicConfig` is a no-op when it has any.

    pytest's own logging plugin attaches handlers to the root, so without this the call
    under test would silently do nothing and the assertions would be measuring pytest.
    """
    root = logging.getLogger()
    saved, saved_level = root.handlers[:], root.level
    root.handlers = []
    yield root
    for handler in root.handlers:
        handler.close()
    root.handlers, root.level = saved, saved_level


def test_the_session_log_is_written_to_a_file_named_for_the_day(tmp_path: Path) -> None:
    """**The terminal is not the record.** On 24 Aug 2026 a session's log existed only in
    a closed terminal, and the run could not be summarised at all."""
    handler = live_loop.DailyLogFile(tmp_path)
    day = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    try:
        handler.handle(
            logging.LogRecord("x", logging.INFO, "f", 1, "a cycle ran", None, None)
        )
    finally:
        handler.close()

    written = tmp_path / f"live-{day}.log"
    assert written.exists()
    assert "a cycle ran" in written.read_text(encoding="utf-8")


def test_a_restart_appends_rather_than_truncating(tmp_path: Path) -> None:
    """The rehearsal plan is *rehearse, stop, restart under the deployed band*. A
    truncating handler would delete the first half at the restart, where the deletion
    would read as a session that never ran."""
    for message in ("first process", "second process"):
        handler = live_loop.DailyLogFile(tmp_path)
        try:
            handler.handle(
                logging.LogRecord("x", logging.INFO, "f", 1, message, None, None)
            )
        finally:
            handler.close()

    day = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    text = (tmp_path / f"live-{day}.log").read_text(encoding="utf-8")
    assert "first process" in text
    assert "second process" in text


def test_the_file_is_retargeted_when_the_day_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--sessions 3` idles through two midnights inside one process. A filename computed
    once at startup would put all three sessions in the first day's file and make the
    other two dates lies - the two-places defect, in the log's own name."""
    handler = live_loop.DailyLogFile(tmp_path)
    try:
        monkeypatch.setattr(
            live_loop.DailyLogFile, "_today", staticmethod(lambda: "2026-08-24")
        )
        handler.handle(
            logging.LogRecord("x", logging.INFO, "f", 1, "before midnight", None, None)
        )
        monkeypatch.setattr(
            live_loop.DailyLogFile, "_today", staticmethod(lambda: "2026-08-25")
        )
        handler.handle(
            logging.LogRecord("x", logging.INFO, "f", 1, "after midnight", None, None)
        )
    finally:
        handler.close()

    first = (tmp_path / "live-2026-08-24.log").read_text(encoding="utf-8")
    second = (tmp_path / "live-2026-08-25.log").read_text(encoding="utf-8")
    assert "before midnight" in first and "after midnight" not in first
    assert "after midnight" in second and "before midnight" not in second


def test_the_record_stamp_and_the_filename_agree_on_the_day(tmp_path: Path) -> None:
    """The day this record belongs to is written twice - in the filename and in the line -
    and until 26 Aug 2026 the two were in different timezones.

    `live-2026-08-25.log` held lines stamped `2026-08-26 00:00`, because the handler rolls
    on the UTC day and `asctime` defaulted to local time. Harmless to the run and exactly
    the confusion a gate log must not have: a reader looking for 01:00 on the 26th opens
    the wrong file. Asserted rather than commented, because the two copies are three hours
    apart here and would be zero apart on a UTC machine - where a convention would look
    fine and stay wrong.
    """
    handler = live_loop.DailyLogFile(tmp_path)
    formatter = logging.Formatter(
        live_loop.LOG_FORMAT, datefmt=live_loop.LOG_DATE_FORMAT
    )
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    try:
        handler.handle(
            logging.LogRecord("x", logging.INFO, "f", 1, "a cycle ran", None, None)
        )
    finally:
        handler.close()

    written = next(tmp_path.glob("live-*.log"))
    day_in_the_name = written.stem.removeprefix("live-")
    day_in_the_line = written.read_text(encoding="utf-8").split("T")[0]
    assert day_in_the_line == day_in_the_name


def test_the_summary_reaches_the_file_and_not_only_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bare_root_logger
) -> None:
    """**The summary IS the gate evidence** - cycles completed, decisions recorded, open
    orders at exit. It was `print`ed, so a file handler alone would have preserved every
    line except the one the gate reads."""
    report = live_loop.SessionReport(
        session_id="s",
        banner="",
        cycles=(),
        open_orders=(),
        stopped_by="the market closed",
    )
    monkeypatch.setattr(live_loop, "run_sessions", lambda *a, **k: (report,))

    # A named state directory, not the default: the default is the deployed session's,
    # and `main` now takes a lock on it. A test that reaches for the real one would
    # either be refused by a running loop or take a lock on a live directory.
    exit_code = live_loop.main(
        ["--log-dir", str(tmp_path), "--state-dir", str(tmp_path / "state")]
    )
    assert exit_code == 0

    day = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    text = (tmp_path / f"live-{day}.log").read_text(encoding="utf-8")
    assert "stopped by the market closed" in text
    assert "open orders at exit" in text


def test_both_handlers_share_one_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bare_root_logger
) -> None:
    """A second copy of a format string is a second thing to change, and stdout and the
    file diverging is only discovered when the terminal is gone."""
    report = live_loop.SessionReport(
        session_id="s", banner="", cycles=(), open_orders=(), stopped_by="x"
    )
    monkeypatch.setattr(live_loop, "run_sessions", lambda *a, **k: (report,))

    live_loop.main(["--log-dir", str(tmp_path), "--state-dir", str(tmp_path / "state")])

    formats = {h.formatter._fmt for h in bare_root_logger.handlers}
    assert formats == {live_loop.LOG_FORMAT}
    assert {h.formatter.datefmt for h in bare_root_logger.handlers} == {
        live_loop.LOG_DATE_FORMAT
    }
    assert {h.formatter.converter for h in bare_root_logger.handlers} == {time.gmtime}


# ── GB-40 / 25 Aug 2026: rule 3's remedy must be reachable ──────────────────


def test_rule_3_flatten_releases_the_legs_and_ends_flat(
    cfg: Config, broker: FakeBroker, tmp_path: Path
) -> None:
    """**The escape hatch has to be reachable under the condition it escapes.**

    Found live on 25 Aug 2026, on the first order this system ever placed. NVDA's entry
    filled, the stop armed, and the target was refused - the broker holds the *whole*
    position for any working sell order, so the second leg had no quantity to claim. Rule
    3 then did what it promises and flattened at market, and **the flatten was refused for
    the same reason**: it is also a sell. Thirteen consecutive cycles tried and failed,
    each logged as an unreachable broker, and the loop decided nothing for twenty minutes
    while holding a real position.

    The close-out path had been right all along - it cancels the legs and then sells - so
    the fix is one implementation instead of two rather than new behaviour.

    The state is built directly rather than by running cycles, because with the fix in
    place a cycle no longer *leaves* a position in this condition: it flattens it. The
    condition under test is a position holding one working leg, which is what rule 3 meets.

    This test could not have existed before ``FakeBroker`` modelled ``held_for_orders``:
    the fake permitted a second sell the broker refuses, so the suite stayed green while
    the live path could not execute at all.
    """
    state = a_state(cfg, tmp_path, broker)
    symbol, quantity = "AAPL", 0.118742281
    broker.positions[symbol] = quantity
    holding = Holding(
        symbol=symbol,
        quantity=quantity,
        decision_id="20260824-AAPL",
        entry_price=PRICE,
        stop_loss=97.0,
        take_profit=106.0,
    )
    state.book = Book(managed={symbol: holding})
    # One live leg, holding the entire position - exactly the live shape.
    broker.submit_stop_order(
        symbol=symbol,
        quantity=quantity,
        stop_price=97.0,
        client_order_id=f"{holding.decision_id}{records.STOP_SUFFIX}",
    )
    assert broker.held_for_orders(symbol) == pytest.approx(quantity)
    with pytest.raises(FakeBrokerError, match="insufficient qty"):
        broker.submit_market_order(symbol, quantity, SELL, "would-fail")

    live_loop._flatten(state, symbol, holding, log=lambda message: None)

    assert broker.get_positions().get(symbol, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert not [
        order
        for order in broker.get_orders()
        if order.symbol == symbol
        and order.side == SELL
        and order.status not in live_loop.FINISHED
    ], "a protective leg is still working after the flatten"


def test_the_summary_reports_the_close_out_flatten_and_what_is_still_held() -> None:
    """**The gate reads this line, and on 25 Aug 2026 it was false.**

    A rehearsal cancelled a stop and sold the position on shutdown, and the summary said
    `positions flattened : 0` - because `close_out_on_stop` runs in `run_session`'s
    `finally`, after the cycle list is closed, so no `CycleReport` can carry it. The one
    line answering rehearsal condition 3 contradicted the run it described.

    `open orders at exit : 0` is the same defect one line down. It was **right by luck**:
    it is computed after the close-out cancelled the legs, so a close-out that cancelled
    and then failed to sell would print zero working orders beside a real position - the
    cleanest-looking summary the loop can produce for the worst state it can end in.
    """
    flattened = live_loop.SessionReport(
        session_id="s",
        banner="",
        cycles=(),
        open_orders=(),
        stopped_by="SIGINT",
        closed_out=("NVDA",),
        held_at_exit=(),
    )
    summary = flattened.summary()
    assert "1 by the stop close-out" in summary
    assert "NVDA" in summary
    assert "positions at exit    : none" in summary

    # And the state that must never read as clean: legs cancelled, sell refused.
    stranded = live_loop.SessionReport(
        session_id="s",
        banner="",
        cycles=(),
        open_orders=(),
        stopped_by="SIGINT",
        closed_out=(),
        close_out_failed=("NVDA",),
        held_at_exit=("NVDA",),
    )
    stranded_summary = stranded.summary()
    assert "open orders at exit  : 0" in stranded_summary
    assert "STILL HELD - NVDA" in stranded_summary
    assert "CLOSE-OUT FAILED" in stranded_summary
    assert "Close it by hand" in stranded_summary


# ── GB-29 / 5 Oct 2026: trade emission, end to end ───────────────────────────
#
# On 2 and 5 Oct 2026 the deployed loop emitted its first trade and crashed step 2 on every
# cycle of two sessions: the stored entry time is a string, and `Trade` received it as one.
# The trade itself was a phantom - a 2026-08-28 closeout paired with a 2026-10-01 entry.
# No test had reached either: the records tests hand `emit_trades` a `pd.Timestamp`, and the
# one `run_cycle` test that seeded the stored string was a rehearsal, which empties the
# tuple before the string is read. So these go through `run_cycle`, outside a rehearsal,
# with the entry fill written by the loop and, after a restart, read back from disk.

LATER = NOW + pd.Timedelta(hours=2)


def opened_by_the_loop(
    cfg: Config, broker: FakeBroker, tmp_path: Path
) -> live_loop.LiveState:
    """AAPL opened by `run_cycle` itself, so its entry fill is what `_absorb_entry` stores."""
    broker.now = NOW
    auto = replace(cfg, live=replace(cfg.live, mode="auto"))
    state = a_state(auto, tmp_path, broker, total=0.05)
    live_loop.run_cycle(state, NOW)
    live_loop.run_cycle(state, NOW)
    assert "AAPL" in state.book.managed, "nothing opened; harness problem"
    return state


def restarted(state: live_loop.LiveState) -> live_loop.LiveState:
    """A new process on the same state directory, built from disk as `run_session` builds it.

    Nothing carries over from memory - in particular not ``seen_orders``, so every order in
    the broker's history is fresh to the first cycle, as it is in production.
    """
    root = state.state_dir
    return live_loop.LiveState(
        cfg=state.cfg,
        broker=state.broker,
        predictor=state.predictor,
        thresholds=state.thresholds,
        state_dir=root,
        book=Book.load(root / live_loop.BOOK_FILE),
        entry_fills=live_loop._load_entry_fills(root / live_loop.ENTRY_FILLS_FILE),
    )


def working_stop(broker: FakeBroker, symbol: str) -> BrokerOrder:
    [stop] = [
        order
        for order in broker.orders
        if order.symbol == symbol
        and order.side == SELL
        and order.status not in live_loop.FINISHED
    ]
    return stop


def trade_log_lines(root: Path) -> list[str]:
    path = root / records.TRADES_FILE
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.mark.parametrize("restart", [True, False], ids=["restart", "same-process"])
def test_a_stop_exit_reaches_the_trade_log_with_a_typed_entry_time(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path, restart: bool
) -> None:
    """Guard A. The stored entry fill, string and all, becomes a typed `Trade` on disk.

    Both variants, because the value is a string in memory as well as in the file: the
    restart reads it from ``entry_fills.json``, the same process reads the list the writer
    left. **The same-process variant fills the stop before any cycle has read it as
    working**, which is the one in-session case T1 does not drop; a cycle in between is
    T1, pinned by the strict xfail below.
    """
    state = opened_by_the_loop(cfg, broker, tmp_path)
    stored = json.loads(
        (tmp_path / live_loop.ENTRY_FILLS_FILE).read_text(encoding="utf-8")
    )
    # The premise: production's format, on disk and in memory, not a constructed Trade.
    assert isinstance(stored["AAPL"][0], str)
    assert isinstance(state.entry_fills["AAPL"][0], str)

    stop = working_stop(broker, "AAPL")
    level = state.book.managed["AAPL"].stop_loss
    broker.now = LATER
    broker.trigger(stop.id, level)
    if restart:
        state = restarted(state)

    report = live_loop.run_cycle(state, LATER)

    assert report.ok, report.error
    [trade] = report.trades
    assert records.load_trades(tmp_path) == [trade]
    assert isinstance(trade.entry_time, pd.Timestamp)
    assert trade.entry_time == pd.Timestamp(stored["AAPL"][0]) == NOW
    assert trade.exit_time == LATER
    assert trade.exit_reason == records.STOP
    assert trade.exit_order_id == stop.id


def test_a_sell_from_an_earlier_round_trip_is_not_a_trade_after_a_restart(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """Guard B, the phantom of 2 and 5 Oct 2026, reproduced in its production order.

    An earlier round trip closed weeks ago; the loop opens the symbol again; the DAY stop
    expires overnight; a new process starts. Its first cycle sees the old closeout as
    fresh, and the symbol is managed with an entry fill - exactly the pairing that matched
    on symbol alone. It must emit nothing, crash nothing, and reach step 4, which on 5 Oct
    is the step that never ran.
    """
    broker.now = NOW - pd.Timedelta(days=40)
    broker.submit_market_order("AAPL", 0.5, BUY, "20260701-AAPL")
    closeout = broker.submit_market_order("AAPL", 0.5, SELL, "20260701-AAPL-closeout")
    state = opened_by_the_loop(cfg, broker, tmp_path)
    assert closeout.raw["filled_at"] < pd.Timestamp(state.entry_fills["AAPL"][0])
    expired = working_stop(broker, "AAPL")
    broker.orders = [
        replace(order, status="expired") if order.status == "new" else order
        for order in broker.orders
    ]

    report = live_loop.run_cycle(restarted(state), LATER)

    assert report.ok, report.error
    assert trade_log_lines(tmp_path) == []
    assert report.trades == ()
    # Step 4 ran: the stop that expired overnight was replaced by a new working one.
    assert "AAPL" in report.rearmed
    assert working_stop(broker, "AAPL").id != expired.id


@pytest.mark.xfail(
    strict=True,
    reason="T1 (PROGRESS, 30 Sep 2026): a stop read as working enters seen_orders, so "
    "its later fill is never fresh and never becomes a Trade. Not fixed on 5 Oct.",
)
def test_a_stop_seen_working_and_then_filled_becomes_one_trade(
    cfg: Config, broker: FakeBroker, stub_bars: dict, tmp_path: Path
) -> None:
    """T1's acceptance, as written in its row: one Trade on the fill cycle, none after."""
    state = opened_by_the_loop(cfg, broker, tmp_path)
    live_loop.run_cycle(state, NOW)  # the stop is read as working here
    stop = working_stop(broker, "AAPL")
    broker.now = LATER
    broker.trigger(stop.id, state.book.managed["AAPL"].stop_loss)

    on_the_fill = live_loop.run_cycle(state, LATER)
    after = live_loop.run_cycle(state, LATER)

    assert len(on_the_fill.trades) == 1
    assert after.trades == ()


def test_a_run_of_failures_makes_no_claim_about_the_stops(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """On 2 and 5 Oct 2026 this line said every protective leg was still live while the
    broker held no working order. The function reads nothing from the broker and does not
    know which step failed, so it may only say that protection is unconfirmed."""
    failed = live_loop.CycleReport(
        cycle_id="c",
        at=NOW,
        entries_allowed=True,
        failed_step="AttributeError",
        error="boom",
    )
    with caplog.at_level(logging.ERROR, logger="glassbox.live_loop"):
        live_loop._log_run_of_failures([failed, failed])

    [message] = [
        record.getMessage()
        for record in caplog.records
        if "cycles in a row" in record.getMessage()
    ]
    assert "still live" not in message
    assert "protection is NOT confirmed" in message
