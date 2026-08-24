"""L1: the read timeout every Alpaca client shares.

**Why this module exists, measured rather than argued.** On 24 Aug 2026 the first GATE 2
live session lost 90 minutes inside two HTTP reads - 30m 06s in the data client's bar
fetch and 59m 49s in the broker client's ``get_orders``. Neither call had a timeout,
because the Alpaca SDK takes no ``timeout`` parameter and its ``RESTClient._one_request``
calls ``self._session.request(method, url, **opts)`` bare, so ``requests`` applies its
default of *block until the OS gives up*.

**What that costs is not latency, it is legibility.** A loop blocked on a socket read is
alive and not polling, and from outside it is indistinguishable from a loop that has died -
the exact failure the heartbeat was written to eliminate, and one the heartbeat does not
cover because heartbeats fire only *outside* a session. Stall A produced two log lines in
half an hour; stall B produced three in an hour.

**And the retry policy did not help, because it bounds the wrong thing.** ``retry_attempts:
3`` bounds the *number* of attempts and nothing bounded their *duration*, so it is a
guarantee about count that reads like a guarantee about time. One stall hit attempt 1 and
the other attempt 3; neither ran out of attempts, because neither attempt ever returned.

The policy is applied to the SDK's **session** rather than to the call sites that happened
to stall, so every request either client makes is bounded rather than the two that were
caught. A timeout on some calls is a stall waiting for the others.

Implemented in GB-40 hardening, 24 Aug 2026.
"""

from __future__ import annotations

from typing import Any

#: Seconds any single HTTP read may take before it is abandoned as failed.
#:
#: **Measured from the 24 Aug 2026 session, over 520 healthy single calls** - spans
#: containing no retry, so each is one request rather than a retry sequence:
#:
#: =========================  ====  ====  ====  =====
#: call                       p50   p95   p99   max
#: =========================  ====  ====  ====  =====
#: ``get_orders``             0.22  1.52  4.57  19.75
#: reconcile + protection     0.40  2.31  8.47  10.11
#: daily bar fetch            1.94  5.01  6.55  17.92
#: =========================  ====  ====  ====  =====
#:
#: **This was ruled at 30 s first, on a premise the measurement did not support, and the
#: wrong number is kept here with its reason because it is more useful than the right
#: number alone.** The 30 s ruling assumed a maximum around 5 s and therefore roughly six
#: times headroom. Then 520 calls were measured and the maximum was **19.75 s** - so 30 s
#: was 1.52x headroom, not 6x.
#:
#: **What moved the value was not the ratio but where the maximum came from.** That 19.75 s
#: call was the *first cycle recovering from an outage*, which is the worst possible moment
#: for a spurious timeout and the one where it is **self-reinforcing**: recovery is slow,
#: the timeout fires, a cycle that was about to succeed is skipped, and the retry meets the
#: same slow path. A bound tuned on healthy-period latency is tuned on the case that does
#: not matter.
#:
#: **The costs are asymmetric.** Too low skips a working cycle while the system is already
#: degraded. Too high detects a real stall in 138 s instead of 93 s - and since both exceed
#: one 60 s poll, the cycle is skipped either way. The whole difference is 42 seconds, and a
#: one-hour rehearsal window still gets 26 attempts at 45 s.
#:
#: Zero of the 520 measured calls came near 45 s; the margin is **2.28x over the slowest**.
#: Worst case is ``3 attempts x 45 s + 3 s of backoff = 138 s``, after which the cycle skips
#: and the next poll tries again. Late and audible beats punctual and silent.
#:
#: **A dry-run cannot validate this number.** It runs in a healthy period, and the case the
#: value turns on - recovery latency - is not reproducible on demand. A clean dry-run says
#: the bound does not fire on healthy calls; it says nothing about the case that set it.
#:
#: **Change the reasoning above, not this number on its own.** The value is downstream of a
#: measurement, and a number moved without moving the measurement is a number nobody can
#: check.
READ_TIMEOUT_SECONDS = 45.0


class UnboundedClientError(RuntimeError):
    """Raised when a client cannot be given a bounded read.

    **Loud on purpose.** The whole value of this module is that no request escapes the
    timeout, and it reaches into the SDK's private ``_session`` to get there. If a future
    SDK release renames or removes it, silently doing nothing would restore the exact
    defect this was written for - a stall nobody can see - while every test that checks
    behaviour rather than plumbing kept passing.
    """


def bound_reads[C](client: C, seconds: float = READ_TIMEOUT_SECONDS) -> C:
    """Give an Alpaca SDK client a bounded read on **every** request it makes.

    The SDK exposes no ``timeout`` parameter, so the bound is applied by wrapping the
    ``requests.Session`` the client already holds rather than by replacing it: a fresh
    session would drop the authentication headers and retry configuration the SDK put
    there. ``setdefault`` rather than assignment, so a caller that asks for its own timeout
    still gets it.

    Args:
        client: An Alpaca SDK client holding a ``requests.Session`` at ``_session``.
        seconds: The read bound. Defaults to :data:`READ_TIMEOUT_SECONDS`.

    Returns:
        The same client, for use inline at construction.

    Raises:
        UnboundedClientError: The client holds no usable session.
    """
    session: Any = getattr(client, "_session", None)
    if session is None or not callable(getattr(session, "request", None)):
        raise UnboundedClientError(
            f"{type(client).__name__} holds no `_session` with a callable `request`, so "
            "its reads cannot be bounded. The Alpaca SDK has changed shape; without a "
            "bound, one hung socket stalls a live cycle indefinitely and the log goes "
            "silent - see glassbox/data/http.py for the 90 minutes that cost on "
            "24 Aug 2026."
        )
    if getattr(session, "_glassbox_bounded", False):
        return client

    original = session.request

    def bounded(method: str, url: str, **kwargs: Any):
        kwargs.setdefault("timeout", seconds)
        return original(method, url, **kwargs)

    session.request = bounded
    session._glassbox_bounded = True
    return client


__all__ = ["READ_TIMEOUT_SECONDS", "UnboundedClientError", "bound_reads"]
