"""GB-40 hardening: no HTTP read may block a live cycle indefinitely.

The session of 24 Aug 2026 lost 90 minutes inside two reads that had no timeout, producing
two log lines in the first half hour and three in the second. These tests pin the bound
that closes it, and the last two are the ones Ben asked for by name: a stalled call must
**raise** rather than block, and the retry path must treat that as an ordinary failure
rather than something new to handle.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import requests

from glassbox.config.loader import Config, load_config
from glassbox.data.http import (
    READ_TIMEOUT_SECONDS,
    UnboundedClientError,
    bound_reads,
)
from glassbox.engine.executor import RetryingBroker


class _Session:
    """Stands in for ``requests.Session``, recording what it was called with."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def request(self, method: str, url: str, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return "response"


class _Client:
    def __init__(self) -> None:
        self._session = _Session()


@pytest.fixture
def cfg() -> Config:
    return load_config()


def test_every_request_carries_the_timeout_not_only_the_ones_that_stalled() -> None:
    """**The bound is on the session, not on the call sites.**

    ``get_orders`` and the bar fetch are the two that were caught stalling; binding only
    those would leave every other request the SDK makes unbounded, which is a stall waiting
    for its turn.
    """
    client = bound_reads(_Client())

    client._session.request("GET", "/v2/orders")
    client._session.request("POST", "/v2/orders", json={"symbol": "AAPL"})
    client._session.request("DELETE", "/v2/positions/AAPL")

    assert [call["timeout"] for call in client._session.calls] == [
        READ_TIMEOUT_SECONDS
    ] * 3


def test_a_caller_that_asks_for_its_own_timeout_keeps_it() -> None:
    """``setdefault``, not assignment. The bound is a floor under callers that forget, not
    a ceiling over callers that decided."""
    client = bound_reads(_Client())

    client._session.request("GET", "/v2/account", timeout=2.0)

    assert client._session.calls[0]["timeout"] == 2.0


def test_binding_twice_does_not_nest_the_wrapper() -> None:
    """Construction paths overlap and a client can be handed here more than once. Nesting
    would still work and would make a stack trace one layer deeper for nothing."""
    client = bound_reads(_Client())
    once = client._session.request

    bound_reads(client)

    assert client._session.request is once


def test_a_client_without_a_session_is_refused_loudly() -> None:
    """**Silence here would restore the defect.**

    This reaches into the SDK's private ``_session``. If a release renames it, doing
    nothing quietly would leave every read unbounded again while every behavioural test
    kept passing - a mechanism that stopped being one, with nothing to say so.
    """

    class Changed:
        pass

    with pytest.raises(UnboundedClientError, match="cannot be bounded"):
        bound_reads(Changed())


def test_the_installed_alpaca_clients_still_expose_a_session() -> None:
    """The guard above is only useful if it is guarding something real **today**.

    Constructed with dummy credentials and never called, so this needs no network and no
    secret: the SDK builds its session at construction. If this fails, the bound is not
    being applied to the live path and the previous test's refusal is what production would
    hit at 16:30.
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    for client in (
        TradingClient(api_key="dummy", secret_key="dummy", paper=True),
        StockHistoricalDataClient(api_key="dummy", secret_key="dummy"),
    ):
        bound = bound_reads(client)
        assert callable(bound._session.request)


# ── the two Ben asked for by name ────────────────────────────────────────────


def test_a_stalled_read_raises_rather_than_blocking() -> None:
    """**The whole point.** A read that hangs must become an exception, on a clock, rather
    than a process that is alive and silent."""
    client = _Client()
    seen: dict = {}

    def hang(method: str, url: str, **kwargs):
        seen.update(kwargs)
        raise requests.exceptions.ReadTimeout("read timed out")

    client._session.request = hang
    bound_reads(client)

    with pytest.raises(requests.exceptions.ReadTimeout):
        client._session.request("GET", "/v2/orders")

    assert seen["timeout"] == READ_TIMEOUT_SECONDS


def test_the_retry_path_treats_a_timeout_as_an_ordinary_failure(cfg: Config) -> None:
    """A timeout is a failed attempt, not a new kind of event.

    GB-39's policy retries a read and gives up after ``retry_attempts``; a
    ``ReadTimeout`` has to travel that path like any other connection error, or the bound
    would convert an invisible stall into an unhandled exception - trading a silent failure
    for a loud crash rather than for a skipped cycle.
    """

    class Timing:
        def __init__(self) -> None:
            self.attempts = 0

        def get_positions(self) -> dict[str, float]:
            self.attempts += 1
            raise requests.exceptions.ReadTimeout("read timed out")

    inner = Timing()
    quick = replace(cfg, live=replace(cfg.live, retry_backoff_seconds=0.0))

    with pytest.raises(Exception) as caught:
        RetryingBroker(inner, quick).get_positions()

    assert inner.attempts == cfg.live.retry_attempts
    assert "read timed out" in str(caught.value)


def test_a_timeout_that_recovers_is_not_reported_as_a_failure(cfg: Config) -> None:
    """The other half: one slow read must not cost the cycle.

    The slowest healthy call measured on 24 Aug was 19.75 s against a 30 s bound, and it
    was the first cycle recovering from an outage. Recovery is exactly when a read is
    slowest, so a single timeout followed by a success has to come out as a success.
    """

    class Flaky:
        def __init__(self) -> None:
            self.attempts = 0

        def get_positions(self) -> dict[str, float]:
            self.attempts += 1
            if self.attempts == 1:
                raise requests.exceptions.ReadTimeout("read timed out")
            return {"AAPL": 2.0}

    inner = Flaky()
    quick = replace(cfg, live=replace(cfg.live, retry_backoff_seconds=0.0))

    assert RetryingBroker(inner, quick).get_positions() == {"AAPL": 2.0}
    assert inner.attempts == 2


def test_both_live_clients_are_constructed_through_the_bound() -> None:
    """**The wiring, not the function.** Every test above would pass with both call sites
    reverted.

    Asserted structurally rather than by constructing the clients, because doing that needs
    credentials and this must fail in CI on a machine with none. A client built outside
    ``bound_reads`` is a client with no timeout, which is the whole defect.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "glassbox"
    expected = {
        "engine/executor.py": "TradingClient",
        "data/live.py": "StockHistoricalDataClient",
    }

    for relative, constructor in expected.items():
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        wrapped = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "bound_reads"
            and any(
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == constructor
                for inner in ast.walk(node)
            )
        ]
        assert wrapped, f"{relative}: {constructor} is built outside bound_reads"
