"""GB-67 steps 2a to 5a: the approval controls and dialog, measured on the rendered page.

**The only browser tests in the suite, and in their own file for that reason.** They start
a Streamlit server and drive Chromium against it, which no other dashboard test does -
those read the stylesheet's text, and on 2026-09-29 two of them passed over fills that had
never rendered: `.gb-approve button` matched nothing, because each `st.markdown` is its own
element and the wrapper div closed before the button existed.

**The page ends with the real `_refresh`, as `main` does.** Until 2026-10-01 it did not,
so every run of the test page finished while no run of `main` ever did, and a dialog that
stayed on screen after its decision was answered survived seven of these guards. A page
that omits the mechanism driving the real one is a more forgiving double.

**One server and one browser for the whole module**, shared by every test here, so the
cost is paid once per run however many page-level assertions are added. A test may ask
for a shorter refresh with ``?poll=N``; without it the cycle is long enough that no
refresh lands inside a measurement.

They are not skipped when Chromium is missing. A test an environment can switch off is a
mechanism only where it runs, so CI installs the browser (`.github/workflows/ci.yml`).
"""

from __future__ import annotations

import ast
import json
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from playwright.sync_api import Page, sync_playwright

from glassbox import records
from glassbox.dashboard import app, tokens
from glassbox.engine.signal import ENTER_LONG
from tests.dashboard.test_app import a_record

DECISION_ID = "20260929-PAGE"
APPROVED_ID = "20260930-BTST"
#: A cold Streamlit start on a CI runner, not a sleep: every wait here ends the moment
#: what it waits for appears.
STARTUP_SECONDS = 90
#: Short enough that a refresh lands inside the test, long enough that a click and the
#: answer it causes fall inside one cycle.
SHORT_POLL = 4
REPO = Path(__file__).resolve().parents[2]

#: The display entry every test but the last one reads. Its ``record`` is the nested key
#: the gate reads, not a decision on disk: nothing here answers it.
SHOWN = {
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
}

#: `main`'s order - the dialog, the strip, the panel, the refresh - under the real
#: stylesheet. `main` itself needs a broker; here the project's ``FakeBroker`` stands in
#: for ``AlpacaBroker`` wherever an answer would construct one, and writes every market
#: order it is asked for to ``orders.jsonl`` so the test process can count them.
PAGE = """
import dataclasses
import json
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, sys.argv[2])

import streamlit as st

from glassbox.config.loader import load_config
from glassbox.dashboard import app
from glassbox.engine import executor
from tests.fake_broker import FakeBroker


class Recording(FakeBroker):
    def submit_market_order(self, symbol, quantity, side, client_order_id):
        with (root / "orders.jsonl").open("a", encoding="utf-8") as sink:
            sink.write(json.dumps({"symbol": symbol, "side": side,
                                   "client_order_id": client_order_id}) + "\\n")
        return super().submit_market_order(symbol, quantity, side, client_order_id)


executor.AlpacaBroker = lambda: Recording(prices={"PAGE": 100.0, "BTST": 100.0})

st.session_state[app.RUN_STARTED] = time.monotonic()
st.session_state["runs"] = st.session_state.get("runs", 0) + 1
cfg = load_config()
poll = int(st.query_params.get("poll", "600"))
cfg = dataclasses.replace(cfg, live=dataclasses.replace(cfg.live, poll_seconds=poll))
verdict = app.liveness(
    lock=app.LOCK_NONE,
    band_fires=True,
    in_session=False,
    loop_age=float("inf"),
    broker_age=0.0,
    heartbeat=cfg.live.heartbeat_seconds,
)
st.markdown(app.stylesheet(), unsafe_allow_html=True)
queue = app.approval_modal(root, cfg, verdict, st)
st.markdown(app.session_strip(cfg, verdict, root, "live", 0, pending=len(queue)),
            unsafe_allow_html=True)
app._copilot_panel(root, cfg, verdict, st)
st.markdown(f"page runs={st.session_state['runs']}")
app._refresh(cfg, st)
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
def served(tmp_path_factory) -> Iterator[tuple[Page, str, Path]]:
    """One pending entry, one server, one browser: ``(page, url, root)`` for the module."""
    root = tmp_path_factory.mktemp("page")
    records.save_pending(root, SHOWN)
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
                str(REPO),
            ],
            stdout=sink,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_port(port, server, log)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                try:
                    yield browser.new_page(), f"http://127.0.0.1:{port}", root
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
    The page ends with the real refresh, because a page whose runs finish is not `main`.
    """
    page, url, _ = served
    _open(page, url)
    approve = _style(page, f".st-key-a-{DECISION_ID} button", "backgroundColor")
    reject = _style(page, f".st-key-r-{DECISION_ID} button", "backgroundColor")

    assert approve == _rgb(app.GAIN), f"APPROVE is {approve}, not GAIN {_rgb(app.GAIN)}"
    assert reject == _rgb(app.LOSS), f"REJECT is {reject}, not LOSS {_rgb(app.LOSS)}"


def test_the_modal_is_ground_square_and_filled_on_the_rendered_page(served) -> None:
    """**Guard 7: the dialog obeys the fill rules on the page, not only in the CSS.**

    Its overlay and its panel compute to GROUND with a zero radius, its title to a step of
    the type scale, and its own Approve and Reject to GAIN and LOSS. The page ends with
    the real refresh, because a page whose runs finish is not `main`.
    """
    page, url, _ = served
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
    answers nothing - before either answer. The page ends with the real refresh, because
    a page whose runs finish is not `main`.
    """
    page, url, _ = served
    _open(page, url)
    focused = page.evaluate("() => document.activeElement.getAttribute('role')")
    assert focused == "dialog", f"on open, focus is on {focused!r}, not the dialog"
    page.keyboard.press("Tab")
    label = page.evaluate("() => document.activeElement.getAttribute('aria-label')")
    assert label == "Close", f"the first Tab reaches {label!r}, not Close"


def test_the_pending_count_is_above_the_fold_once_the_dialog_is_dismissed(
    served,
) -> None:
    """**GB-67 step 3, guard 3: a dismissed dialog still leaves the page saying so.**

    After Close, the strip's pending count is inside the viewport without scrolling, and
    it is what the browser finds at its own centre - nothing covers it. The page here puts
    the strip first, after the dialog, because `main` does; that premise is pinned too, so
    this cannot pass on an order `main` no longer has. The page ends with the real
    refresh, because a page whose runs finish is not `main`.
    """
    page, url, _ = served
    _open(page, url)
    page.locator('[data-testid="stDialog"] [aria-label="Close"]').click()
    page.locator('[data-testid="stDialog"]').wait_for(
        state="hidden", timeout=STARTUP_SECONDS * 1000
    )
    element = page.get_by_text("1 AWAITING AN ANSWER")
    element.wait_for(timeout=STARTUP_SECONDS * 1000)
    where = element.evaluate("""e => {
            const r = e.getBoundingClientRect();
            const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
            return {top: r.top, bottom: r.bottom, left: r.left, right: r.right,
                    width: innerWidth, height: innerHeight,
                    hit: e === hit || e.contains(hit)};
        }""")
    inside = (
        0 <= where["top"]
        and where["bottom"] <= where["height"]
        and 0 <= where["left"]
        and where["right"] <= where["width"]
    )
    assert inside, f"the pending count is not above the fold: {where}"
    assert where["hit"], f"something covers the pending count: {where}"

    main = next(
        node
        for node in ast.walk(ast.parse(Path(app.__file__).read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    (modal,) = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "approval_modal"
    ]
    after = sorted(
        (
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "markdown"
            and node.lineno > modal.lineno
        ),
        key=lambda node: node.lineno,
    )
    drawn = ast.unparse(after[0].args[0]) if after else None
    assert drawn is not None and drawn.startswith(
        "session_strip("
    ), f"the first thing main draws after the dialog is {drawn}, not the strip"


def _approvable(root: Path) -> dict:
    """A pending entry the real ``answer_pending`` can answer: its decision is on disk,
    because an answer amends the record rather than adding one."""
    order = {
        "decision_id": APPROVED_ID,
        "symbol": "BTST",
        "shares": 1.0,
        "price": 100.0,
        "notional": 100.0,
        "stop_loss": 97.0,
        "take_profit": 106.0,
    }
    record = a_record(day="2026-09-30", symbol="BTST", order=order)
    record = replace(
        record,
        signal=replace(record.signal, action=ENTER_LONG),
        narrative="A recommendation approved on the real refresh.",
    )
    records.save_decision(record, root)
    return {
        **{key: order[key] for key in order},
        "as_of": "2026-09-30T00:00:00+00:00",
        "narrative": record.narrative,
        "provenance": records.LIVE,
        "record": records.encode_decision(record),
    }


def _runs(page: Page) -> int:
    text = page.get_by_text("page runs=").first.inner_text()
    return int(text.rsplit("=", 1)[1])


def test_an_answered_decision_leaves_the_page_on_the_real_refresh(served) -> None:
    """**GB-67 step 5a: after an answer, the dialog is gone and stays gone.**

    On 2026-10-01 a GOOGL approval went through and the dialog stayed on screen with the
    same record, because no run of `main` ever finished and Streamlit clears an element a
    newer run stopped sending only when a run finishes. Here the real ``answer_pending``
    runs against the project's ``FakeBroker``: after APPROVE the dialog is gone within one
    cycle and still gone after the next refresh, the queue is empty, and the broker was
    asked for exactly one order. The page ends with the real refresh, because that is the
    mechanism whose absence hid this defect for the whole task.
    """
    page, url, root = served
    shown = records.load_pending(root)
    for entry in shown:
        records.resolve_pending(root, entry["decision_id"])
    records.save_pending(root, _approvable(root))
    try:
        page.goto(f"{url}/?poll={SHORT_POLL}")
        page.locator(f".st-key-modal-a-{APPROVED_ID} button").click(
            timeout=STARTUP_SECONDS * 1000
        )
        page.locator('[data-testid="stDialog"]').wait_for(
            state="detached", timeout=(SHORT_POLL + 2) * 1000
        )
        before = _runs(page)
        page.wait_for_function(
            "n => Number((/page runs=(\\d+)/.exec(document.body.innerText) || [0, 0])[1])"
            " > n",
            arg=before,
            timeout=(2 * SHORT_POLL + 5) * 1000,
        )

        assert not page.locator(
            '[data-testid="stDialog"]'
        ).count(), "the answered decision's dialog came back after a refresh"
        assert records.load_pending(root) == [], records.load_pending(root)
        orders = [
            json.loads(line)
            for line in (root / "orders.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        sent = [order for order in orders if order["client_order_id"] == APPROVED_ID]
        assert len(sent) == 1, f"the broker was asked for {len(sent)} orders: {orders}"
    finally:
        for entry in shown:
            records.save_pending(root, entry)


ANSWERED_ELSEWHERE = "20260930-ELSE"
STILL_PENDING = "20260930-STAY"


def test_a_decision_answered_elsewhere_is_not_offered_again(served) -> None:
    """**GB-67 step 5b: the dialog reads the queue, not the list it was opened with.**

    Two decisions are pending and the dialog is open. One is then answered elsewhere -
    resolved on disk behind the page - and its APPROVE is clicked. A click inside the
    dialog reruns only the dialog, so a body that kept the list it was opened with would
    offer that decision again. It must vanish, the other must stay answerable, nothing
    may raise, and no order may be asked for. The page ends with the real refresh,
    because a page whose runs finish is not `main`.
    """
    page, url, root = served
    shown = records.load_pending(root)
    for entry in shown:
        records.resolve_pending(root, entry["decision_id"])
    for decision_id, symbol in ((ANSWERED_ELSEWHERE, "ELSE"), (STILL_PENDING, "STAY")):
        records.save_pending(
            root, {**SHOWN, "decision_id": decision_id, "symbol": symbol}
        )
    try:
        _open(page, url)
        answered = page.locator(f".st-key-modal-a-{ANSWERED_ELSEWHERE} button")
        answered.wait_for(timeout=STARTUP_SECONDS * 1000)
        records.resolve_pending(root, ANSWERED_ELSEWHERE)
        answered.click()
        answered.wait_for(state="detached", timeout=10_000)

        assert page.locator(f".st-key-modal-a-{STILL_PENDING} button").count() == 1
        errors = page.locator('[data-testid="stException"]')
        assert not errors.count(), errors.first.inner_text()
        sent = root / "orders.jsonl"
        log = sent.read_text(encoding="utf-8") if sent.is_file() else ""
        assert ANSWERED_ELSEWHERE not in log, f"an order was asked for: {log}"
    finally:
        for decision_id in (ANSWERED_ELSEWHERE, STILL_PENDING):
            records.resolve_pending(root, decision_id)
        for entry in shown:
            records.save_pending(root, entry)
