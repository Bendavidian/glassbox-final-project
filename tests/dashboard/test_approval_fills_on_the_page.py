"""GB-67 step 2a: the Approve and Reject fills, measured on the rendered page.

**The only browser test in the suite, and in its own file for that reason.** It starts a
Streamlit server and drives Chromium against it, which no other dashboard test does - they
read the stylesheet's text, and on 2026-09-29 both of them passed over fills that had never
rendered: `.gb-approve button` matched nothing, because each `st.markdown` is its own
element and the wrapper div closed before the button existed.

It is not skipped when Chromium is missing. A test an environment can switch off is a
mechanism only where it runs, so CI installs the browser (`.github/workflows/ci.yml`).
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from glassbox import records
from glassbox.dashboard import app

DECISION_ID = "20260929-PAGE"
#: A cold Streamlit start on a CI runner, not a sleep: the test waits for the button and
#: stops waiting the moment it appears.
STARTUP_SECONDS = 90

#: The real Co-Pilot panel under the real stylesheet, and nothing else of the page -
#: `main` needs a broker and ends in a refresh loop, and neither is what is measured here.
PAGE = """
import sys
from pathlib import Path

import streamlit as st

from glassbox.config.loader import load_config
from glassbox.dashboard import app

cfg = load_config()
verdict = app.liveness(
    lock=app.LOCK_NONE,
    band_fires=True,
    in_session=False,
    loop_age=float("inf"),
    broker_age=0.0,
    heartbeat=cfg.live.heartbeat_seconds,
)
st.markdown(app.stylesheet(), unsafe_allow_html=True)
app._copilot_panel(Path(sys.argv[1]), cfg, verdict, st)
"""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _rgb(hex_colour: str) -> str:
    red, green, blue = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgb({red}, {green}, {blue})"


def _wait_for_port(port: int, server: subprocess.Popen, log: Path) -> None:
    deadline = time.monotonic() + STARTUP_SECONDS
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise AssertionError(
                f"streamlit exited with {server.returncode}:\n{log.read_text()}"
            )
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise AssertionError(f"streamlit never listened on {port}:\n{log.read_text()}")


def test_approve_and_reject_are_filled_on_the_rendered_page(tmp_path: Path) -> None:
    """**The fill is the warning, so it has to be on the page and not only in the CSS.**

    One pending entry, the real panel, the real stylesheet: the Approve button's computed
    background must be GAIN and the Reject button's LOSS. The buttons are found by their
    own widget keys, not by the containers the fill is selected through, so removing the
    containers fails this on the colour rather than on a missing element.
    """
    records.save_pending(
        tmp_path,
        {
            "decision_id": DECISION_ID,
            "as_of": "2026-09-29T00:00:00+00:00",
            "symbol": "PAGE",
            "shares": 1.0,
            "price": 100.0,
            "notional": 100.0,
            "stop_loss": 97.0,
            "take_profit": 106.0,
            "narrative": "A recommendation rendered to measure its controls.",
            "provenance": records.LIVE,
        },
    )
    script = tmp_path / "page.py"
    script.write_text(PAGE, encoding="utf-8")
    log = tmp_path / "streamlit.log"
    port = _free_port()

    with log.open("w", encoding="utf-8") as sink:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(script),
                "--server.headless",
                "true",
                "--server.address",
                "127.0.0.1",
                "--server.port",
                str(port),
                "--browser.gatherUsageStats",
                "false",
                "--",
                str(tmp_path),
            ],
            stdout=sink,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_port(port, server, log)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                try:
                    page = browser.new_page()
                    page.goto(f"http://127.0.0.1:{port}")
                    fills = {}
                    for answer, prefix in (("APPROVE", "a"), ("REJECT", "r")):
                        button = page.locator(f".st-key-{prefix}-{DECISION_ID} button")
                        button.wait_for(timeout=STARTUP_SECONDS * 1000)
                        fills[answer] = button.evaluate(
                            "e => getComputedStyle(e).backgroundColor"
                        )
                finally:
                    browser.close()
        finally:
            server.terminate()
            server.wait(timeout=30)

    assert fills["APPROVE"] == _rgb(
        app.GAIN
    ), f"APPROVE is {fills['APPROVE']} on the page, not GAIN {_rgb(app.GAIN)}"
    assert fills["REJECT"] == _rgb(
        app.LOSS
    ), f"REJECT is {fills['REJECT']} on the page, not LOSS {_rgb(app.LOSS)}"
