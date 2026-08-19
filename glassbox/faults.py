"""L0: what to do when the outside world does not answer. **GB-39.**

Two things reach outside this process — the broker and the market-data API — and both fail
in the same three ways: a timeout, a dropped connection, and a 5xx from a service that is
briefly unwell. None of those is a reason to lose a session, and all of them are a reason
to wait a moment and ask again.

**Why a module of its own.** Its callers sit at opposite ends of the layer stack:
``data/live.py`` is L1 and ``engine/executor.py`` is L5, so neither can import a helper
from the other without inverting the stack. A retry loop written twice is a backoff policy
written twice, and the second copy is the one that drifts. This sits below both.

**Neither caller decides the policy.** ``attempts`` and ``backoff`` are arguments, and
``load_live_bars`` defaults them to *no retry* — a module with no configuration should do
what it was asked rather than what somebody once thought sensible. The live loop passes
``cfg.live.retry_attempts`` and ``cfg.live.retry_backoff_seconds``, which is where a
policy belongs.

**What it deliberately is not.** No circuit breaker, no health endpoint, no failure budget,
no half-open state. Those are the shape of a service that must stay up for other services;
this is a single process that polls a market for six and a half hours and may skip a poll.
A breaker here would add a second state machine whose disagreement with reality is a new
failure mode, to protect against a failure mode that costs one cycle.

**Retrying a write is safe here, and it is not safe in general.** Every order this system
submits carries a ``client_order_id`` derived from the decision, and Alpaca refuses a
repeat of one (`40010001`, measured 18 Aug 2026). So a submission that timed out without a
response is either absent — in which case the retry places it — or present, in which case
the retry is refused and the refusal *is the confirmation that the first attempt landed*.
:class:`engine.executor.RetryingBroker` reads it that way. Without the deterministic id
this module would have to refuse to retry writes at all.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

LOGGER = logging.getLogger(__name__)


class Unavailable(RuntimeError):
    """The outside world did not answer, and the attempts are spent.

    Distinct from every other error the loop can raise, because the response to it is
    different: a cycle that fails on this has learned nothing about the market and should
    be skipped and retried on the next poll, while a cycle that fails on a ``ValueError``
    has found a defect and should be read.
    """


def retry[T](
    call: Callable[[], T],
    *,
    description: str,
    attempts: int,
    backoff: float,
    sleep: Callable[[float], None] = time.sleep,
    log: logging.Logger = LOGGER,
) -> T:
    """Call ``call``, retrying with exponential backoff. Raise :class:`Unavailable` if it
    never succeeds.

    Args:
        call: The thing to attempt. Takes no arguments so the caller binds its own.
        description: What is being attempted, in words, for the log. Every retry line
            names it — "the daily bar fetch", "submit_market_order AAPL" — because a log
            that says only "retrying" is a log nobody can act on at 16:31.
        attempts: Total attempts including the first. ``1`` means no retry, which is the
            honest default for a caller that has not been given a policy.
        backoff: Seconds before the second attempt. Doubles each time after that, so
            ``attempts=3, backoff=1.0`` waits 1s then 2s and gives up 3s in — inside one
            60-second poll, which is the constraint that sets the ceiling.
        sleep: Injected so the suite does not spend the backoff.
        log: Injected for the same reason.

    Raises:
        Unavailable: every attempt failed. The original error is chained, because the
            thing that actually broke is more useful than the fact that it broke three
            times.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")

    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as failure:  # noqa: BLE001 - the caller decides what is fatal
            last = failure
            if attempt == attempts:
                break
            delay = backoff * (2 ** (attempt - 1))
            log.warning(
                "%s failed (attempt %d/%d): %s: %s. Retrying in %.1fs",
                description,
                attempt,
                attempts,
                type(failure).__name__,
                failure,
                delay,
            )
            sleep(delay)

    log.error(
        "%s failed on all %d attempts; the last error was %s: %s",
        description,
        attempts,
        type(last).__name__,
        last,
    )
    raise Unavailable(
        f"{description} failed after {attempts} attempts: {last}"
    ) from last


__all__ = ["Unavailable", "retry"]
