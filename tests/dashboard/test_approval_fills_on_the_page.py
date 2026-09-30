"""GB-67 steps 2a and 2: the approval controls and dialog, measured on the rendered page.

**The only browser tests in the suite, and in their own file for that reason.** They start
a Streamlit server and drive Chromium against it, which no other dashboard test does -
those read the stylesheet's text, and on 2026-09-29 two of them passed over fills that had
never rendered: `.gb-approve button` matched nothing, because each `st.markdown` is its own
element and the wrapper div closed before the button existed.

**One server and one browser for the whole module**, shared by every test here, so the
cost is paid once per run however many page-level assertions are added.

They are not skipped when Chromium is missing. A test an environment can switch off is a
mechanism only where it runs, so CI installs the browser (`.github/workflows/ci.yml`).
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import Page, sync_playwright

from glassbox import records
from glassbox.dashboard import app, tokens

DECISION_ID = "20260929-PAGE"
#: A cold Streamlit start on a CI runner, not a sleep: every wait here ends the moment
#: what it waits for appears.
STARTUP_SECONDS = 90

#: The dialog and the Co-Pilot panel under the real stylesheet, called as `main` calls
#: them - `main` itself needs a broker and ends in a refresh loop, neither of which is
#: what is measured here.
PAGE = """
import sys
from pathlib import Path

import streamlit as st

from glassbox.config.loader import load_config
from glassbox.dashboard import app

cfg = load_config()
root = Path(sys.argv[1])
verdict = app.liveness(
    lock=app.LOCK_NONE,
    band_fires=True,
    in_session=False,
    loop_age=float("inf"),
    broker_age=0.0,
    heartbeat=cfg.live.heartbeat_seconds,
)
st.markdown(app.stylesheet(), unsafe_allow_html=True)
app.approval_modal(root, cfg, verdict, st)
app._copilot_panel(root, cfg, verdict, st)
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


@pytest.fixture(scope="module")
def served(tmp_path_factory) -> Iterator[tuple[Page, str]]:
    """One pending entry, one server, one browser: ``(page, url)`` for the module."""
    root = tmp_path_factory.mktemp("page")
    records.save_pending(
        root,
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
            "record": {"signal": {"action": "enter_long"}},
        },
    )
    script = root / "page.py"
    script.write_text(PAGE, encoding="utf-8")
    log = root / "streamlit.log"
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
                str(root),
            ],
            stdout=sink,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_port(port, server, log)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                try:
                    yield browser.new_page(), f"http://127.0.0.1:{port}"
                finally:
                    browser.close()
        finally:
            server.terminate()
            server.wait(timeout=30)


def _open(page: Page, url: str) -> None:
    page.goto(url)
    page.locator('[data-testid="stDialog"] [role="dialog"]').wait_for(
        timeout=STARTUP_SECONDS * 1000
    )


def _style(page: Page, selector: str, prop: str) -> str:
    element = page.locator(selector).first
    element.wait_for(state="attached", timeout=STARTUP_SECONDS * 1000)
    return element.evaluate(f"e => getComputedStyle(e).{prop}")


def test_approve_and_reject_are_filled_on_the_rendered_page(served) -> None:
    """**The fill is the warning, so it has to be on the page and not only in the CSS.**

    The panel's Approve must compute to GAIN and its Reject to LOSS. The buttons are found
    by their own widget keys, not by the containers the fill is selected through, so
    removing the containers fails this on the colour rather than on a missing element.
    """
    page, url = served
    _open(page, url)
    approve = _style(page, f".st-key-a-{DECISION_ID} button", "backgroundColor")
    reject = _style(page, f".st-key-r-{DECISION_ID} button", "backgroundColor")

    assert approve == _rgb(app.GAIN), f"APPROVE is {approve}, not GAIN {_rgb(app.GAIN)}"
    assert reject == _rgb(app.LOSS), f"REJECT is {reject}, not LOSS {_rgb(app.LOSS)}"


def test_the_modal_is_ground_square_and_filled_on_the_rendered_page(served) -> None:
    """**Guard 7: the dialog obeys the fill rules on the page, not only in the CSS.**

    Its overlay and its panel compute to GROUND with a zero radius, its title to a step of
    the type scale, and its own Approve and Reject to GAIN and LOSS.
    """
    page, url = served
    _open(page, url)
    overlay = '[data-testid="stDialog"]'
    panel = '[data-testid="stDialog"] > div'
    measured = {
        "overlay background": _style(page, overlay, "backgroundColor"),
        "panel background": _style(page, panel, "backgroundColor"),
        "panel radius": _style(page, panel, "borderRadius"),
        "title size": _style(page, f"{overlay} h2", "fontSize"),
        "APPROVE": _style(
            page, f".st-key-modal-a-{DECISION_ID} button", "backgroundColor"
        ),
        "REJECT": _style(
            page, f".st-key-modal-r-{DECISION_ID} button", "backgroundColor"
        ),
    }
    expected = {
        "overlay background": _rgb(tokens.GROUND),
        "panel background": _rgb(tokens.GROUND),
        "panel radius": "0px",
        "title size": f"{tokens.TYPE_PROSE}px",
        "APPROVE": _rgb(app.GAIN),
        "REJECT": _rgb(app.LOSS),
    }
    wrong = {k: v for k, v in measured.items() if v != expected[k]}
    assert not wrong, f"on the page {wrong}; expected {expected}"


def test_opening_the_modal_focuses_neither_answer(served) -> None:
    """**Guard 3 on the page: no answer is one keystroke away when the dialog opens.**

    Focus lands on the dialog itself, and the first Tab reaches Close - dismissal, which
    answers nothing - before either answer.
    """
    page, url = served
    _open(page, url)
    focused = page.evaluate("() => document.activeElement.getAttribute('role')")
    assert focused == "dialog", f"on open, focus is on {focused!r}, not the dialog"
    page.keyboard.press("Tab")
    label = page.evaluate("() => document.activeElement.getAttribute('aria-label')")
    assert label == "Close", f"the first Tab reaches {label!r}, not Close"
