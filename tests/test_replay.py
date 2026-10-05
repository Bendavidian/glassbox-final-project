"""GB-37 and GB-38 acceptance: Co-Pilot's two answers, and a replay that says it is one.

The two tasks are tested together because the ruling coupled them: the deployed band
stands aside, so the only recommendation Co-Pilot can be shown is one replay produces.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fake_broker import FakeBroker

from glassbox import live_loop, records
from glassbox.config.loader import Config, load_config
from glassbox.contracts.schemas import (
    Attribution,
    DecisionRecord,
    Forecast,
    Signal,
)
from glassbox.engine import risk
from glassbox.engine.executor import DECLINED, SELL, SUBMITTED, approve, decline
from glassbox.engine.signal import ENTER_LONG
from glassbox.replay import ReplayBroker, ReplayError

SYMBOL = "AAPL"
PRICE = 100.0
AS_OF = pd.Timestamp("2025-07-21", tz="UTC")


def an_order(notional: float = 1000.0) -> risk.Order:
    return risk.Order(
        symbol=SYMBOL,
        shares=notional / PRICE,
        price=PRICE,
        notional=notional,
        stop_loss=PRICE * 0.97,
        take_profit=PRICE * 1.06,
    )


def a_record(provenance: str = records.LIVE) -> DecisionRecord:
    return DecisionRecord(
        as_of=AS_OF,
        symbol=SYMBOL,
        forecast=Forecast(
            path=np.array([0.01, 0.01, 0.01, 0.01], dtype="float32"),
            symbol=SYMBOL,
            as_of=AS_OF,
        ),
        attribution=Attribution(
            per_channel={"close_logret": 0.03, "rsi14": 0.01},
            per_lag=None,
            per_frequency=None,
            gain_phase=None,
            forecast_total=0.04,
        ),
        signal=Signal(
            symbol=SYMBOL,
            action=ENTER_LONG,
            trend_strength=0.04,
            up_points=4,
            passed_threshold=True,
        ),
        order={"symbol": SYMBOL},
        narrative="AAPL on 2025-07-21: the model predicts a rise.",
        config_hash="x",
        provenance=provenance,
    )


def queue_one(root: Path, provenance: str = records.LIVE) -> str:
    """Queue a recommendation the way the loop does - **record first, then queue**.

    The loop writes the decision when it makes it and queues the recommendation beside it,
    so answering *amends* that record rather than adding a second one at the same bar
    (ruling, 19 Aug 2026). A fixture that queued without recording would let the answer
    path append, which is the behaviour the ruling forbids.
    """
    order = an_order()
    decision_id = "cycle-0001-AAPL"
    records.save_decision(a_record(provenance), root)
    records.save_pending(
        root,
        {
            "decision_id": decision_id,
            "as_of": AS_OF.isoformat(),
            "symbol": SYMBOL,
            "shares": order.shares,
            "price": order.price,
            "notional": order.notional,
            "stop_loss": order.stop_loss,
            "take_profit": order.take_profit,
            "narrative": "…",
            "provenance": provenance,
            "record": records.encode_decision(a_record(provenance)),
        },
    )
    return decision_id


@pytest.fixture
def cfg() -> Config:
    return load_config()


@pytest.fixture
def broker() -> FakeBroker:
    return FakeBroker(prices={SYMBOL: PRICE})


# ── GB-37: the two answers ───────────────────────────────────────────────────


def test_approving_submits_and_arms_protection(cfg: Config, broker: FakeBroker) -> None:
    """One protective order, and it is the stop (ruled 26 Aug 2026). The count changed;
    what is being tested - that an approval arms protection rather than leaving the
    position bare - did not."""
    submission = approve(broker, an_order(), "d1", cfg)

    assert submission.status == SUBMITTED
    assert submission.entry is not None
    assert len(submission.protection) == 1


def test_approval_ignores_the_mode_because_it_is_the_answer_to_it(
    cfg: Config, broker: FakeBroker
) -> None:
    """Re-checking `live.mode` would make an approval conditional on a setting the
    approver cannot see."""
    assert cfg.live.mode == "co_pilot"

    assert approve(broker, an_order(), "d1", cfg).status == SUBMITTED


def test_declining_leaves_no_order_at_the_broker(broker: FakeBroker) -> None:
    """GB-37's acceptance criterion, asserted against the broker rather than the return."""
    submission = decline(an_order(), "d1")

    assert submission.status == DECLINED
    assert submission.entry is None
    assert submission.protection == ()
    assert broker.orders == []


def test_an_approval_below_the_minimum_notional_is_refused_not_sent(
    cfg: Config, broker: FakeBroker
) -> None:
    submission = approve(broker, an_order(notional=0.5), "d1", cfg)

    assert submission.status != SUBMITTED
    assert broker.orders == []


def test_both_answers_write_a_decision_record(
    cfg: Config, broker: FakeBroker, tmp_path: Path
) -> None:
    """A rejection that left no trace would make the log a record of what the system
    wanted rather than of what happened."""
    for approved in (True, False):
        root = tmp_path / str(approved)
        root.mkdir()
        decision_id = queue_one(root)

        live_loop.answer_pending(cfg, broker, root, decision_id, approved)

        written = records.load_decisions(AS_OF, AS_OF, root)
        assert len(written) == 1  # amended, not appended
        assert ("Approved" if approved else "Declined") in written[0].narrative


def test_a_declined_recommendation_records_no_order(
    cfg: Config, broker: FakeBroker, tmp_path: Path
) -> None:
    decision_id = queue_one(tmp_path)

    live_loop.answer_pending(cfg, broker, tmp_path, decision_id, False)

    assert records.load_decisions(AS_OF, AS_OF, tmp_path)[0].order is None


def test_answering_takes_it_off_the_queue(
    cfg: Config, broker: FakeBroker, tmp_path: Path
) -> None:
    decision_id = queue_one(tmp_path)

    live_loop.answer_pending(cfg, broker, tmp_path, decision_id, True)

    assert records.load_pending(tmp_path) == []


def test_answering_twice_is_refused_rather_than_resubmitted(
    cfg: Config, broker: FakeBroker, tmp_path: Path
) -> None:
    """Two dashboard tabs, one click each, is the realistic way this happens."""
    decision_id = queue_one(tmp_path)
    live_loop.answer_pending(cfg, broker, tmp_path, decision_id, True)
    submitted = len(broker.orders)

    with pytest.raises(live_loop.LiveError, match="answered already"):
        live_loop.answer_pending(cfg, broker, tmp_path, decision_id, True)

    assert len(broker.orders) == submitted


def test_the_queue_survives_a_restart(tmp_path: Path) -> None:
    """The approver is a different process, so the recommendation cannot live in memory."""
    decision_id = queue_one(tmp_path)

    reread = records.load_pending(tmp_path)

    assert [entry["decision_id"] for entry in reread] == [decision_id]
    assert json.loads((tmp_path / records.PENDING_FILE).read_text(encoding="utf-8"))


# ── GB-38: provenance ────────────────────────────────────────────────────────


def test_a_replayed_record_names_its_fold(tmp_path: Path) -> None:
    records.save_decision(a_record(records.replay_provenance(13)), tmp_path)

    stored = records.load_decisions(AS_OF, AS_OF, tmp_path, records.ANY_PROVENANCE)

    assert stored[0].provenance == "replay:fold-13"
    assert records.is_replay(stored[0].provenance)


def test_the_default_filter_hides_replays_rather_than_showing_them_as_live(
    tmp_path: Path,
) -> None:
    """The direction of the default is the safety property."""
    records.save_decision(a_record(records.LIVE), tmp_path)
    records.save_decision(
        replace(a_record(records.replay_provenance(13)), symbol="MSFT"), tmp_path
    )

    default = records.load_decisions(AS_OF, AS_OF, tmp_path)
    everything = records.load_decisions(AS_OF, AS_OF, tmp_path, records.ANY_PROVENANCE)

    assert [r.symbol for r in default] == [SYMBOL]
    assert len(everything) == 2


def test_one_replay_can_be_asked_for_by_name(tmp_path: Path) -> None:
    records.save_decision(a_record(records.replay_provenance(13)), tmp_path)
    records.save_decision(
        replace(a_record(records.replay_provenance(8)), symbol="MSFT"), tmp_path
    )

    eleven = records.load_decisions(
        AS_OF, AS_OF, tmp_path, records.replay_provenance(13)
    )

    assert [r.symbol for r in eleven] == [SYMBOL]


def test_a_record_written_before_provenance_existed_decodes_as_live(
    tmp_path: Path,
) -> None:
    """They were live. The default is a fact about those files, not a fallback."""
    raw = records.encode_decision(a_record())
    raw.pop("provenance")
    path = records.month_file(tmp_path, AS_OF)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    assert records.load_decisions(AS_OF, AS_OF, tmp_path)[0].provenance == records.LIVE


# ── GB-38: the replay broker ─────────────────────────────────────────────────


def bar(high: float, low: float, close: float) -> dict:
    return {SYMBOL: {"open": close, "high": high, "low": low, "close": close}}


def test_a_market_order_fills_at_the_current_bar() -> None:
    broker = ReplayBroker()
    broker.advance(bar(101.0, 99.0, 100.0))

    order = broker.submit_market_order(SYMBOL, 2.0, "buy", "d1")

    assert order.status == "filled"
    assert order.filled_price == pytest.approx(100.0)
    assert broker.get_positions() == {SYMBOL: 2.0}


def test_a_stop_fills_when_the_bars_low_reaches_it() -> None:
    broker = ReplayBroker()
    broker.advance(bar(101.0, 99.0, 100.0))
    broker.submit_market_order(SYMBOL, 2.0, "buy", "d1")
    broker.submit_stop_order(SYMBOL, 2.0, 97.0, "d1-stop")

    resolved = broker.advance(bar(100.0, 96.0, 96.5))

    assert resolved and "stop filled" in resolved[0]
    assert broker.get_positions() == {}


def test_a_stop_and_a_target_on_the_same_bar_resolve_stop_first() -> None:
    """GB-18's ruling 1: daily OHLC cannot order them, so the conservative reading wins."""
    broker = ReplayBroker()
    broker.advance(bar(101.0, 99.0, 100.0))
    broker.submit_market_order(SYMBOL, 2.0, "buy", "d1")
    broker.submit_stop_order(SYMBOL, 2.0, 97.0, "d1-stop")
    broker.submit_limit_order(SYMBOL, 2.0, 106.0, "d1-target")

    resolved = broker.advance(bar(107.0, 96.0, 100.0))

    assert len(resolved) == 1
    assert "stop filled" in resolved[0]


@pytest.mark.xfail(
    strict=True,
    reason="ReplayBroker stamps no filled_at (found 5 Oct 2026, not closed then). A "
    "replayed stop exit reads as filled at Timestamp.min, before any entry, so "
    "records.emit_trades refuses it as a sell from an earlier round trip.",
)
def test_a_replayed_stop_fill_carries_its_fill_time() -> None:
    broker = ReplayBroker()
    broker.advance(bar(101.0, 99.0, 100.0))
    broker.submit_market_order(SYMBOL, 2.0, "buy", "d1")
    stop = broker.submit_stop_order(SYMBOL, 2.0, 97.0, "d1-stop")
    broker.advance(bar(100.0, 96.0, 96.5))

    [filled] = [order for order in broker.get_orders() if order.id == stop.id]
    assert filled.status == "filled"
    assert "filled_at" in filled.raw


def test_an_untouched_level_does_not_fill() -> None:
    broker = ReplayBroker()
    broker.advance(bar(101.0, 99.0, 100.0))
    broker.submit_market_order(SYMBOL, 2.0, "buy", "d1")
    broker.submit_stop_order(SYMBOL, 2.0, 97.0, "d1-stop")

    assert broker.advance(bar(102.0, 98.0, 101.0)) == []
    assert broker.get_positions() == {SYMBOL: 2.0}


def test_a_cancelled_leg_never_fills() -> None:
    broker = ReplayBroker()
    broker.advance(bar(101.0, 99.0, 100.0))
    broker.submit_market_order(SYMBOL, 2.0, "buy", "d1")
    stop = broker.submit_stop_order(SYMBOL, 2.0, 97.0, "d1-stop")
    broker.cancel_order(stop.id)

    assert broker.advance(bar(100.0, 90.0, 95.0)) == []


def test_the_replay_broker_reports_an_account() -> None:
    broker = ReplayBroker()
    broker.advance(bar(101.0, 99.0, 100.0))
    broker.submit_market_order(SYMBOL, 10.0, "buy", "d1")

    account = broker.get_account()

    assert account["equity"] == pytest.approx(100_000.0)
    assert account["cash"] == pytest.approx(99_000.0)


def test_selling_reduces_the_position_and_returns_cash() -> None:
    broker = ReplayBroker()
    broker.advance(bar(101.0, 99.0, 100.0))
    broker.submit_market_order(SYMBOL, 4.0, "buy", "d1")
    broker.submit_market_order(SYMBOL, 4.0, SELL, "d1-exit")

    assert broker.get_positions() == {}
    assert broker.get_account()["cash"] == pytest.approx(100_000.0)


def test_a_replay_without_a_manifest_says_how_to_make_one(
    cfg: Config, tmp_path: Path
) -> None:
    from glassbox.replay import replay_fold

    with pytest.raises(ReplayError, match="prepare-replay"):
        replay_fold(cfg, tmp_path)
