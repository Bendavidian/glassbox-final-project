"""GB-23 acceptance: the broker is the truth, and the book follows it.

Every test drives ``FakeBroker`` — no network, no credentials, no market session. The three
divergences GB-23 names each get a case, and so does the property that matters most: a book
that has been deliberately desynchronised is corrected on the next cycle, whatever the
reason it went wrong.
"""

from __future__ import annotations

import json
import logging

import pytest
from fake_broker import FakeBroker

from glassbox.engine import reconcile
from glassbox.engine.executor import SELL, BrokerOrder

SYMBOL = "AAPL"
OTHER = "MSFT"
PRICE = 305.1772


def holding(
    symbol: str = SYMBOL, quantity: float = 0.5, decision: str = "d-1"
) -> reconcile.Holding:
    return reconcile.Holding(
        symbol=symbol,
        quantity=quantity,
        decision_id=decision,
        entry_price=PRICE,
        stop_loss=PRICE * 0.97,
        take_profit=PRICE * 1.06,
    )


def protection(symbol: str = SYMBOL, quantity: float = 0.5) -> list[BrokerOrder]:
    """The two live sell orders a protected position carries (GB-22: no bracket)."""
    return [
        BrokerOrder(
            id=f"{symbol}-stop",
            symbol=symbol,
            side=SELL,
            quantity=quantity,
            status="new",
        ),
        BrokerOrder(
            id=f"{symbol}-target",
            symbol=symbol,
            side=SELL,
            quantity=quantity,
            status="accepted",
        ),
    ]


def broker_with(positions: dict[str, float], orders: list[BrokerOrder]) -> FakeBroker:
    broker = FakeBroker(prices={SYMBOL: PRICE, OTHER: 400.0})
    broker.positions = dict(positions)
    broker.orders = list(orders)
    return broker


# ── case 1: a broker position the system has no record of ────────────────────


def test_an_unknown_position_is_quarantined_not_adopted() -> None:
    """The 0.01 AAPL probe, in test form.

    Adopting it would put a position under management with no entry price, no stop and no
    decision behind it — the system could neither protect nor explain it. Quarantine keeps
    it visible without pretending it is managed.
    """
    broker = broker_with({SYMBOL: 0.01}, [])

    result = reconcile.reconcile(broker, reconcile.Book())

    assert SYMBOL not in result.book.managed
    assert result.book.unmanaged == {SYMBOL: 0.01}
    assert [d.kind for d in result.divergences] == [reconcile.UNKNOWN_POSITION]


def test_an_unknown_position_is_not_silently_ignored_either() -> None:
    """It consumes buying power the sizer would otherwise believe is free."""
    broker = broker_with({SYMBOL: 0.01}, [])

    result = reconcile.reconcile(broker, reconcile.Book())

    assert result.book.committed(SYMBOL) == pytest.approx(0.01)
    assert not result.clean  # it is reported, every cycle it persists


def test_a_quarantined_position_is_reported_once_not_every_cycle() -> None:
    """A divergence that never resolves becomes noise, and noise is how a real one gets
    missed. Once the book records the quarantine, the next cycle is quiet about it."""
    broker = broker_with({SYMBOL: 0.01}, [])

    first = reconcile.reconcile(broker, reconcile.Book())
    second = reconcile.reconcile(broker, first.book)

    assert first.of_kind(reconcile.UNKNOWN_POSITION)
    assert not second.of_kind(reconcile.UNKNOWN_POSITION)
    assert second.book.unmanaged == {SYMBOL: 0.01}  # still quarantined, still counted


def test_a_quarantined_position_that_changes_size_is_reported_again() -> None:
    """Something is moving it that the system does not control. That is worth a line."""
    broker = broker_with({SYMBOL: 0.01}, [])
    first = reconcile.reconcile(broker, reconcile.Book())

    broker.positions[SYMBOL] = 0.02
    second = reconcile.reconcile(broker, first.book)

    assert second.of_kind(reconcile.UNKNOWN_POSITION)
    assert second.book.unmanaged == {SYMBOL: 0.02}


def test_an_unknown_position_is_never_given_protective_levels() -> None:
    """It has none to be given: there is no decision to take a stop or a target from."""
    broker = broker_with({SYMBOL: 0.01}, [])

    result = reconcile.reconcile(broker, reconcile.Book())

    assert result.book.managed == {}
    assert not result.of_kind(reconcile.MISSING_PROTECTION)


# ── case 2: a local position the broker does not have ────────────────────────


def test_a_position_the_broker_does_not_have_is_dropped() -> None:
    broker = broker_with({}, [])
    book = reconcile.Book(managed={SYMBOL: holding()})

    result = reconcile.reconcile(broker, book)

    assert result.book.managed == {}
    assert [d.kind for d in result.divergences] == [reconcile.MISSING_POSITION]


def test_a_filled_stop_looks_exactly_like_a_missing_position() -> None:
    """Which is why it is resolved rather than raised: this is the normal path for an exit
    the system did not initiate in this cycle."""
    broker = broker_with({}, [])
    book = reconcile.Book(managed={SYMBOL: holding()})

    result = reconcile.reconcile(broker, book)
    divergence = result.divergences[0]

    assert divergence.local == pytest.approx(0.5)
    assert divergence.broker == 0.0
    assert "filled stop" in divergence.detail


# ── case 3: a quantity mismatch from a partial fill ──────────────────────────


def test_a_partial_fill_takes_the_brokers_quantity() -> None:
    broker = broker_with({SYMBOL: 0.3}, protection(quantity=0.3))
    book = reconcile.Book(managed={SYMBOL: holding(quantity=0.5)})

    result = reconcile.reconcile(broker, book)

    assert result.book.managed[SYMBOL].quantity == pytest.approx(0.3)
    assert [d.kind for d in result.divergences] == [reconcile.QUANTITY_MISMATCH]


def test_a_partial_fill_keeps_the_local_provenance() -> None:
    """The broker is the truth about what is held, not about why it was bought."""
    broker = broker_with({SYMBOL: 0.3}, protection(quantity=0.3))
    book = reconcile.Book(managed={SYMBOL: holding(quantity=0.5, decision="d-42")})

    corrected = reconcile.reconcile(broker, book).book.managed[SYMBOL]

    assert corrected.decision_id == "d-42"
    assert corrected.entry_price == pytest.approx(PRICE)
    assert corrected.stop_loss == pytest.approx(PRICE * 0.97)


def test_a_float_representation_difference_is_not_a_divergence() -> None:
    """Nine decimals is what the broker stores (GB-22), so anything below that is noise."""
    broker = broker_with({SYMBOL: 0.5 + 1e-12}, protection())
    book = reconcile.Book(managed={SYMBOL: holding(quantity=0.5)})

    assert reconcile.reconcile(broker, book).clean


# ── the property that matters: a desynchronised book is corrected ────────────


def test_a_deliberately_desynchronised_book_is_corrected_next_cycle() -> None:
    """Three wrongs at once, none of which the book knows about.

    This is the failure GB-23 exists to prevent: the process died, or a fill arrived while
    it was down, and the belief no longer describes the account. One cycle repairs it.
    """
    broker = broker_with(
        {SYMBOL: 0.25, "NVDA": 0.02}, protection(SYMBOL, 0.25)
    )  # AAPL half-filled, NVDA unknown, MSFT gone
    book = reconcile.Book(
        managed={
            SYMBOL: holding(quantity=0.5),
            OTHER: holding(symbol=OTHER, quantity=1.0),
        }
    )

    result = reconcile.reconcile(broker, book)

    assert {d.kind for d in result.divergences} == {
        reconcile.QUANTITY_MISMATCH,
        reconcile.MISSING_POSITION,
        reconcile.UNKNOWN_POSITION,
    }
    assert result.book.managed[SYMBOL].quantity == pytest.approx(0.25)
    assert OTHER not in result.book.managed
    assert result.book.unmanaged == {"NVDA": 0.02}
    # And the corrected book is stable: reconciling it again changes nothing.
    assert reconcile.reconcile(broker, result.book).clean


def test_reconciliation_does_not_mutate_the_book_it_was_given() -> None:
    """A caller that logs the before and after must have a before to log."""
    broker = broker_with({SYMBOL: 0.25}, protection(SYMBOL, 0.25))
    book = reconcile.Book(managed={SYMBOL: holding(quantity=0.5)})

    reconcile.reconcile(broker, book)

    assert book.managed[SYMBOL].quantity == pytest.approx(0.5)


def test_reconciliation_never_submits_or_cancels_anything() -> None:
    """It reads. A reconciler that traded would be the state machine GB-23 was scoped away
    from, and the one place that only ever reads is worth keeping."""
    broker = broker_with({SYMBOL: 0.25, "NVDA": 0.02}, [])
    book = reconcile.Book(
        managed={SYMBOL: holding(quantity=0.5), OTHER: holding(OTHER)}
    )

    reconcile.reconcile(broker, book)

    assert broker.submitted_kinds == []
    assert broker.cancelled == []


# ── detection only: protection that is not live ──────────────────────────────


def test_a_managed_position_without_two_live_legs_is_reported() -> None:
    """GB-26 rule 3 acts on this; GB-23 only sees it."""
    broker = broker_with({SYMBOL: 0.5}, [protection()[0]])  # stop only
    book = reconcile.Book(managed={SYMBOL: holding()})

    result = reconcile.reconcile(broker, book)

    assert [d.kind for d in result.divergences] == [reconcile.MISSING_PROTECTION]
    assert result.book.managed[SYMBOL].quantity == pytest.approx(0.5)


def test_finished_orders_do_not_count_as_live_protection() -> None:
    """A filled stop protects nothing, and a cancelled one protects less."""
    stale = [
        BrokerOrder(id="a", symbol=SYMBOL, side=SELL, quantity=0.5, status="filled"),
        BrokerOrder(id="b", symbol=SYMBOL, side=SELL, quantity=0.5, status="canceled"),
    ]
    broker = broker_with({SYMBOL: 0.5}, stale)

    result = reconcile.reconcile(broker, reconcile.Book(managed={SYMBOL: holding()}))

    assert result.of_kind(reconcile.MISSING_PROTECTION)


def test_a_fully_protected_position_is_clean() -> None:
    broker = broker_with({SYMBOL: 0.5}, protection())

    result = reconcile.reconcile(broker, reconcile.Book(managed={SYMBOL: holding()}))

    assert result.clean


# ── the book survives a restart ──────────────────────────────────────────────


def test_the_book_round_trips_through_disk(tmp_path) -> None:
    book = reconcile.Book(managed={SYMBOL: holding()}, unmanaged={"NVDA": 0.02})
    path = tmp_path / "state" / "book.json"

    book.save(path)
    restored = reconcile.Book.load(path)

    assert restored.managed[SYMBOL] == book.managed[SYMBOL]
    assert restored.unmanaged == book.unmanaged


def test_a_missing_book_is_the_first_run_not_an_error(tmp_path) -> None:
    assert reconcile.Book.load(tmp_path / "nothing.json") == reconcile.Book()


def test_a_corrupt_book_raises_rather_than_starting_empty(tmp_path) -> None:
    """Starting from an empty book would look like starting from a belief while holding
    none — the system would think the account is flat and size as if it were."""
    path = tmp_path / "book.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        reconcile.Book.load(path)


# ── it says so out loud ──────────────────────────────────────────────────────


def test_every_divergence_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    broker = broker_with({SYMBOL: 0.25, "NVDA": 0.02}, protection(SYMBOL, 0.25))
    book = reconcile.Book(
        managed={SYMBOL: holding(quantity=0.5), OTHER: holding(OTHER)}
    )

    with caplog.at_level(logging.WARNING, logger="glassbox.engine.reconcile"):
        result = reconcile.reconcile(broker, book)

    assert len(caplog.records) == len(result.divergences)
    assert all(record.levelno >= logging.WARNING for record in caplog.records)


def test_a_divergence_reads_as_a_sentence() -> None:
    """It ends up in a run log a human reads at 16:31, not in a debugger."""
    broker = broker_with({SYMBOL: 0.01}, [])

    text = str(reconcile.reconcile(broker, reconcile.Book()).divergences[0])

    assert text.startswith("unknown_position: AAPL")
    assert "quarantined" in text
