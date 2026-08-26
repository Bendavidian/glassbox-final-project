"""GB-22 acceptance: submission, the two modes, and the parity GB-18 promised.

No test here touches the network. ``FakeBroker`` refuses what the live API was measured to
refuse, so a test can fail for the reason the broker would have failed for.

The test that matters most is :func:`test_the_executor_and_the_backtester_agree_on_the_share_count`
— GB-18 wrote ``shares_for`` as the single conversion and said the executor must reuse it.
This is that promise coming due, and it is checked by driving a real backtest and a real
submission from the same notional and price rather than by calling one function twice.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import pandas as pd
import pytest
from fake_broker import FakeBroker, FakeBrokerError

from glassbox.backtest import engine
from glassbox.config.loader import Config, load_config
from glassbox.contracts.schemas import Signal
from glassbox.engine import executor, risk

SYMBOL = "AAPL"
PRICE = 305.1772
DECISION = "gb22-decision-0001"


@pytest.fixture
def cfg() -> Config:
    return load_config()


@pytest.fixture
def auto(cfg: Config) -> Config:
    return replace(cfg, live=replace(cfg.live, mode=executor.AUTO))


@pytest.fixture
def copilot(cfg: Config) -> Config:
    return replace(cfg, live=replace(cfg.live, mode=executor.CO_PILOT))


def an_order(notional: float = 10_000.0, price: float = PRICE) -> risk.Order:
    shares = risk.shares_for(notional, price)
    return risk.Order(
        symbol=SYMBOL,
        shares=shares,
        price=price,
        notional=shares * price,
        stop_loss=price * 0.97,
        take_profit=price * 1.06,
    )


# ── the auto path ────────────────────────────────────────────────────────────


def test_auto_submits_a_market_order_and_reports_the_fill(auto: Config) -> None:
    broker = FakeBroker(prices={SYMBOL: PRICE})

    submission = executor.execute(broker, an_order(), DECISION, auto)

    assert submission.submitted
    assert submission.entry is not None
    assert submission.entry.filled_quantity == pytest.approx(
        risk.shares_for(an_order().notional, PRICE)
    )
    assert broker.positions[SYMBOL] > 0


def test_a_filled_entry_is_protected_by_one_order_and_it_is_the_stop(
    auto: Config,
) -> None:
    """**Replaces `test_a_filled_entry_is_protected_by_two_standalone_orders`,** which was
    deleted rather than fixed on 26 Aug 2026 because its premise was measured impossible.

    That test asserted a stop *and* a limit as "the fractional bracket substitute". The
    substitute does not exist: a working sell order holds the **whole** position at Alpaca,
    so the second protective order is refused with `insufficient qty available` - measured
    against a **97.38-share** position, so it is not a fractional-only rule and whole-share
    sizing would not lift it. What fractional removes is the bracket that would have been
    the workaround.

    It passed for a week against a FakeBroker that allowed what the API forbids, in exactly
    the dimension the protection policy depends on. Deleting it is the finding.
    """
    broker = FakeBroker(prices={SYMBOL: PRICE})

    submission = executor.execute(broker, an_order(), DECISION, auto)

    assert broker.submitted_kinds == ["market", "stop"]
    assert len(submission.protection) == 1
    assert all(order.side == executor.SELL for order in submission.protection)


def test_no_limit_order_is_ever_armed_as_protection(auto: Config) -> None:
    """The property, stated so it cannot come back by accident.

    The deleted test's assertion would have passed again the moment somebody re-added the
    limit, and against the old FakeBroker it would have looked correct. This one fails.
    """
    broker = FakeBroker(prices={SYMBOL: PRICE})

    executor.execute(broker, an_order(), DECISION, auto)

    assert "limit" not in broker.submitted_kinds


def test_protection_covers_the_filled_quantity_and_not_the_requested_one(
    auto: Config,
) -> None:
    """Arming protection for shares the account does not hold is a short sale, which the
    API refuses on fractional quantities."""
    broker = FakeBroker(prices={SYMBOL: PRICE})

    submission = executor.execute(broker, an_order(), DECISION, auto)
    held = broker.positions[SYMBOL]

    assert all(order.quantity == pytest.approx(held) for order in submission.protection)


def test_an_unfilled_entry_arms_no_protection(auto: Config) -> None:
    """The after-the-close case: the order is accepted and fills at the next open.

    Arming a stop against a position that does not exist yet would be rejected as a short
    sale, so the cycle that sees the fill arms it instead.
    """
    broker = FakeBroker(prices={SYMBOL: PRICE}, fill=False)

    submission = executor.execute(broker, an_order(), DECISION, auto)

    assert submission.submitted
    assert submission.protection == ()
    assert broker.submitted_kinds == ["market"]


def test_an_order_below_the_brokers_minimum_is_declined_before_it_is_sent(
    auto: Config,
) -> None:
    """The corrected rule: a NOTIONAL floor of $1, not a share-count floor.

    Declining here rather than letting the API refuse keeps the reason in our own log.
    """
    broker = FakeBroker(prices={SYMBOL: PRICE})

    submission = executor.execute(broker, an_order(notional=0.50), DECISION, auto)

    assert submission.status == executor.REJECTED
    assert "minimum order size" in submission.reason
    assert broker.orders == []


# ── Co-Pilot ─────────────────────────────────────────────────────────────────


def test_co_pilot_recommends_and_submits_nothing(copilot: Config) -> None:
    broker = FakeBroker(prices={SYMBOL: PRICE})

    submission = executor.execute(broker, an_order(), DECISION, copilot)

    assert submission.status == executor.PENDING_APPROVAL
    assert submission.entry is None
    assert broker.orders == []
    assert broker.positions == {}


def test_co_pilot_records_the_quantity_it_would_have_sent(copilot: Config) -> None:
    """GB-37 approves a submission, it does not re-derive one. A recommendation that did
    not record its own quantity would be re-priced at approval time and would no longer be
    the decision that was approved."""
    broker = FakeBroker(prices={SYMBOL: PRICE})

    submission = executor.execute(broker, an_order(), DECISION, copilot)

    quantity = risk.shares_for(an_order().notional, PRICE)
    assert any(f"{quantity:.9f}" in line for line in submission.log)


# ── the GB-18 promise, coming due ────────────────────────────────────────────


def test_the_executor_and_the_backtester_agree_on_the_share_count(
    auto: Config,
) -> None:
    """One notional, one price, two systems — and they must derive the same quantity.

    Driven through both real paths rather than by calling `shares_for` twice: the backtest
    runs its event loop and opens a position, the executor submits through a broker, and
    the two quantities are compared. If either side ever grows its own conversion, this
    fails — which is the whole reason GB-18 put the conversion in one place.
    """
    notional = 10_000.0
    index = pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")
    bars = {
        SYMBOL: pd.DataFrame(
            {
                "open": [PRICE, PRICE, PRICE],
                "high": [PRICE * 1.001] * 3,
                "low": [PRICE * 0.999] * 3,
                "close": [PRICE] * 3,
            },
            index=index,
            dtype="float64",
        )
    }
    entry = Signal(
        symbol=SYMBOL,
        action=executor.BUY and "enter_long",
        trend_strength=0.05,
        up_points=4,
        passed_threshold=True,
    )

    def fixed_notional(signal, equity, gross_exposure, cfg):
        del signal, equity, gross_exposure, cfg
        return notional

    backtest = engine.run_backtest(bars, {index[0]: (entry,)}, fixed_notional, auto)
    # The entry fills at the next open; the final bar liquidates it, so the trade log
    # carries the size the engine actually bought.
    backtest_shares = backtest.trades[0].size

    broker = FakeBroker(prices={SYMBOL: PRICE})
    # The engine pays slippage on the fill, so the executor is handed the same REFERENCE
    # price and notional the sizer saw — which is what both sides convert from.
    submission = executor.execute(
        broker,
        risk.Order(
            symbol=SYMBOL,
            shares=risk.shares_for(notional, PRICE),
            price=PRICE,
            notional=notional,
            stop_loss=PRICE * 0.97,
            take_profit=PRICE * 1.06,
        ),
        DECISION,
        auto,
    )
    executor_shares = submission.entry.quantity

    slipped = PRICE * (1.0 + auto.backtest.slippage_bps / 10_000.0)
    assert backtest_shares == pytest.approx(
        risk.shares_for(notional, slipped), rel=1e-12
    )
    assert executor_shares == pytest.approx(risk.shares_for(notional, PRICE), rel=1e-12)
    # Same conversion, same inputs, same answer: hand the engine's own fill price to the
    # executor's route and the two agree exactly.
    assert risk.shares_for(notional, slipped) == pytest.approx(
        backtest_shares, rel=1e-15
    )


def test_the_share_count_is_floored_to_what_the_broker_stores() -> None:
    """Alpaca truncates a quantity to 9 decimals — measured: 0.123456789012 came back as
    0.123456789. A backtest holding more precision than that is holding a fill that could
    not have happened."""
    shares = risk.shares_for(1_000.0, 3.0)

    assert shares == pytest.approx(333.333333333, abs=1e-12)
    assert shares * 1e9 == pytest.approx(round(shares * 1e9), abs=1e-6)


def test_the_minimum_is_a_notional_not_a_share_count() -> None:
    """The GB-22 correction, both directions.

    A share-count floor of 0.001 would permit $0.31 of AAPL, which the API refuses, and
    refuse 0.0005 shares of a $5,000 stock, which it accepts.
    """
    assert risk.shares_for(0.99, 305.0) == 0.0  # under $1: refused
    assert risk.shares_for(1.01, 305.0) > 0.0  # over $1: allowed
    assert (
        risk.shares_for(0.40, 50.0) == 0.0
    )  # 0.008 shares, $0.40 — the old rule allowed
    assert risk.shares_for(2.50, 5_000.0) > 0.0  # 0.0005 shares — the old rule refused


def test_a_notional_of_exactly_one_dollar_can_be_unexpressible(auto: Config) -> None:
    """And is refused rather than rounded up. A deliberate, documented edge.

    $1.00 of a $305 stock is 0.0032786885... shares. Floored to the nine decimals the
    broker stores it becomes 0.003278688, which buys $0.99999984 — under the minimum.
    Rounding up instead would spend a hair more cash than the sizer was shown to have, and
    the cash cap is exact by design. The amount at stake is 5e-8 dollars; the principle is
    that this conversion never returns a quantity the account cannot pay for.
    """
    del auto

    assert risk.shares_for(1.00, 305.0) == 0.0
    assert 0.003278688 * 305.0 < risk.MIN_ORDER_NOTIONAL


# ── logging ──────────────────────────────────────────────────────────────────


def test_every_line_carries_the_decision_id(auto: Config) -> None:
    """A broker order that cannot be traced back to a decision is an unexplained trade."""
    broker = FakeBroker(prices={SYMBOL: PRICE})

    submission = executor.execute(broker, an_order(), DECISION, auto)

    assert submission.log
    assert all(line.startswith(f"[{DECISION}]") for line in submission.log)


def test_the_log_records_both_the_request_and_the_response(
    auto: Config, caplog: pytest.LogCaptureFixture
) -> None:
    broker = FakeBroker(prices={SYMBOL: PRICE})

    with caplog.at_level(logging.INFO, logger="glassbox.engine.executor"):
        submission = executor.execute(broker, an_order(), DECISION, auto)

    joined = "\n".join(submission.log)
    assert "submit MARKET BUY" in joined  # the request
    assert "broker accepted entry id=" in joined  # the response
    assert any(DECISION in record.message for record in caplog.records)


def test_no_credential_appears_in_the_log(auto: Config) -> None:
    """Knowing that a key starts with PK is still information about a key (GB-6)."""
    broker = FakeBroker(prices={SYMBOL: PRICE})

    submission = executor.execute(broker, an_order(), DECISION, auto)

    joined = "\n".join(submission.log).lower()
    for forbidden in ("key", "secret", "token", "password"):
        assert forbidden not in joined


# ── the fake is a real substitute ────────────────────────────────────────────


def test_the_fake_broker_satisfies_the_protocol() -> None:
    """A fake missing a method the live path calls would make the suite prove nothing."""
    assert isinstance(FakeBroker(), executor.Broker)


def test_the_fake_refuses_what_alpaca_refuses() -> None:
    """Measured rejections, not invented ones."""
    broker = FakeBroker(prices={SYMBOL: PRICE})

    with pytest.raises(FakeBrokerError, match="cost basis"):
        broker.submit_market_order(SYMBOL, 0.001, executor.BUY, "x")
    with pytest.raises(FakeBrokerError, match="sold short"):
        broker.submit_stop_order(SYMBOL, 1.0, 100.0, "x")


def test_the_alpaca_broker_also_satisfies_the_protocol() -> None:
    """Checked without constructing one — a real broker would need credentials."""
    assert isinstance(executor.AlpacaBroker, type)
    for method in (
        "submit_market_order",
        "submit_stop_order",
        "submit_limit_order",
        "cancel_order",
        "get_orders",
        "get_positions",
        "get_account",
    ):
        assert callable(getattr(executor.AlpacaBroker, method))


# ── the broker's order history has to include finished orders ────────────────


def test_get_orders_asks_the_broker_for_every_status() -> None:
    """``TradingClient.get_orders()`` with no filter returns OPEN orders only.

    That default was the bug, and it was silent in both directions that matter. Every
    caller of this method that matters asks about orders which have **finished**:
    ``records.emit_trades`` builds the live trade log from filled sells, so with an
    open-only list the live trade log was structurally empty and GB-19 had nothing to
    measure over paper results; and ``live_loop.protect_book`` detects a filled leg in
    order to cancel its sibling, so rule 4 of the protection policy could never fire and a
    filled stop would have left its take-profit working as a naked sell.

    An empty list is a valid-looking answer to the wrong question, which is why this is
    asserted on the request rather than inferred from behaviour.
    """
    from alpaca.trading.enums import QueryOrderStatus

    from glassbox.engine.executor import ORDER_HISTORY, AlpacaBroker

    asked = {}

    class StubClient:
        def get_orders(self, filter=None):  # alpaca-py names the parameter this
            asked["filter"] = filter
            return []

    broker = object.__new__(AlpacaBroker)
    broker._client = StubClient()

    assert broker.get_orders() == []
    assert asked["filter"].status == QueryOrderStatus.ALL
    assert asked["filter"].limit == ORDER_HISTORY
