"""Shared helpers for the dashboard tests.

One definition of "a verdict", imported by name rather than copied into each module: a
second hand-rolled builder would be free to produce a `Liveness` the real constructor
never makes - `LIVE` with no loop present, say - and every test using it would then be
testing that invention instead of the page.
"""

from __future__ import annotations

import pytest

from glassbox.config.loader import Config, load_config
from glassbox.dashboard import app


@pytest.fixture
def cfg_stub() -> Config:
    """The real config. The strip prints the model, the channels and the config hash, and
    a stub would let those drift from what the page actually renders."""
    return load_config()


def a_verdict(
    *,
    lock: str = app.LOCK_ALIVE,
    band_fires: bool = True,
    in_session: bool = True,
    loop_age: float = 0.0,
    broker_age: float = 0.0,
    heartbeat: float = 900.0,
) -> app.Liveness:
    """A verdict **through the real constructor**, from the inputs that produce it.

    Built from causes rather than from the state word, so a test that wants NOT RESPONDING
    has to describe a loop that holds the lock and has stopped writing, which is the thing
    the page is claiming when it prints those words.
    """
    return app.liveness(
        lock=lock,
        band_fires=band_fires,
        in_session=in_session,
        loop_age=loop_age,
        broker_age=broker_age,
        heartbeat=heartbeat,
    )
