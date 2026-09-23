"""The page has one answer about the loop, and every surface gives that answer.

**D2, 23 Sep 2026.** On 22 Sep the console said all of this at once, about a loop that had
not run since 3 Sep: ``SESSION OUTSIDE MARKET HOURS``, ``LAST CYCLE 444.3h ago``, ``LOOP
NOT RESPONDING``, ``LAST READ 26660 MIN AGO`` - and beside it a green ``LIVE`` pill over
``NEXT CYCLE 0s``, with three more green ``LIVE`` pills below. Four surfaces, four answers,
and the one that read RUNNING was reading the number of open positions.

What is pinned here: the bands and their precedence, both boundaries from both sides, that
the SESSION word and the CYCLE badge cannot disagree, and that nothing on the page wears
green or says LIVE unless the loop wrote within one heartbeat.
"""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

import pandas as pd
import pytest

from glassbox.dashboard import app
from glassbox.engine.reconcile import Book, Holding
from tests.dashboard.conftest import a_verdict

HEARTBEAT = 900.0

#: A held position that is **losing**, so the only green the page could show is a live
#: pill. A winning position renders its PnL in GAIN legitimately - that is the gain/loss
#: family, not the liveness one - and would make the "no green" assertion unfalsifiable.
LOSING = Holding(
    symbol="WMT",
    quantity=92.038821904,
    decision_id="d-wmt",
    entry_price=108.70,
    stop_loss=105.39,
    take_profit=112.00,
)
CLOSES = pd.Series([108.00, 107.95, 107.91])


def _rows(verdict: app.Liveness) -> list[app.PositionRow]:
    """One protected position, so the STOP ROOM cell reads a room figure and the words
    "NO LIVE STOP" never enter the page through D1's cell."""
    stops = app.live_stops([app.OpenOrder("WMT", "sell", "stop", 105.39)])
    return app.position_rows(
        Book(managed={"WMT": LOSING}), {"WMT": LOSING.quantity}, {"WMT": 107.91}, stops
    )


def a_page(cfg, verdict: app.Liveness) -> str:
    """Every surface that carries the verdict, rendered into one string.

    This is the page as far as a pure function can build it: `main` itself is Streamlit and
    untestable, so `test_every_live_pill_on_the_page_is_built_from_the_verdict` holds the
    other half - that `main` passes this verdict to each of these and invents no pill.
    """
    rows = _rows(verdict)
    return "".join(
        [
            app.session_strip(cfg, verdict, "checkpoints/live", "live", len(rows)),
            app.region(
                "Session equity", verdict.pill("5 sessions, 0 trades"), "<svg></svg>"
            ),
            app.region("Cycle", verdict.pill("loop cadence"), "<svg></svg>"),
            app.region(
                "Positions",
                verdict.pill(f"{len(rows)} held"),
                app.position_table(rows, verdict),
            ),
            app.region(
                "WMT close and forecast",
                app.forecast_source(pd.Timestamp("2026-09-02"), verdict.state),
                "<svg></svg>",
            ),
            app.sparkline_svg(CLOSES, rows[0], loop_present=verdict.loop_present),
        ]
    )


# ── the bands ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("expected", "inputs"),
    [
        (app.ASIDE, {"band_fires": False}),
        (app.NOT_RESPONDING, {"loop_age": 2 * HEARTBEAT + 1}),
        (app.NOT_RESPONDING, {"lock": app.LOCK_UNREADABLE}),
        (app.NOT_RUNNING, {"lock": app.LOCK_NONE}),
        (app.CLOSED, {"in_session": False}),
        (app.SLOW, {"loop_age": HEARTBEAT + 1}),
        (app.LIVE, {}),
    ],
    ids=[
        "band cannot fire",
        "lock held, no write",
        "lock unreadable",
        "no lock",
        "outside the session",
        "one heartbeat behind",
        "writing now",
    ],
)
def test_each_band_is_reached_by_the_condition_that_names_it(
    expected: str, inputs: dict
) -> None:
    assert a_verdict(heartbeat=HEARTBEAT, **inputs).state == expected


@pytest.mark.parametrize(
    ("expected", "inputs"),
    [
        # Row 1 over row 3: a band that cannot fire is the more surprising fact, and a
        # reader told anything else waits for a trade that cannot come.
        (app.ASIDE, {"band_fires": False, "lock": app.LOCK_NONE}),
        # Row 2 over row 4: silence from a loop that holds the lock is not explained by
        # the market being shut.
        (app.NOT_RESPONDING, {"loop_age": 3 * HEARTBEAT, "in_session": False}),
        # Row 3 over row 4: no process at all outranks the clock.
        (app.NOT_RUNNING, {"lock": app.LOCK_NONE, "in_session": False}),
        # Row 4 over row 5: outside the session nobody is waiting for a cycle, so a late
        # write is not SLOW.
        (app.CLOSED, {"in_session": False, "loop_age": HEARTBEAT + 1}),
    ],
    ids=[
        "aside over not running",
        "silent over shut",
        "no process over shut",
        "shut over slow",
    ],
)
def test_the_earlier_row_wins_when_two_conditions_hold(
    expected: str, inputs: dict
) -> None:
    """Precedence, each pair chosen so both rows' conditions are true at once."""
    assert a_verdict(heartbeat=HEARTBEAT, **inputs).state == expected


def test_a_lock_that_cannot_be_parsed_is_not_responding_rather_than_absent(
    tmp_path: Path,
) -> None:
    """Read from a real lock file, not from a stub: ``read_lock`` raises ``LockRefused``
    for a file it cannot parse, and a dashboard that swallowed that into "no loop" would
    report the quiet answer about a loop that may be trading."""
    from glassbox.live_lock import lock_path

    assert app.loop_lock(tmp_path) == app.LOCK_NONE

    lock_path(tmp_path).write_text("{ not json", encoding="utf-8")

    assert app.loop_lock(tmp_path) == app.LOCK_UNREADABLE
    assert a_verdict(lock=app.loop_lock(tmp_path)).state == app.NOT_RESPONDING


def test_a_lock_naming_a_dead_process_is_not_running(tmp_path: Path) -> None:
    """A PID nobody is running is what ``acquire`` reclaims, so the page says NOT RUNNING.
    This process's own PID is alive by construction, which gives the other half."""
    from glassbox.live_lock import lock_path

    held = {
        "pid": os.getpid(),
        "mode": "auto",
        "command": "live_loop",
        "acquired": "2026-09-23T10:00:00Z",
    }
    lock_path(tmp_path).write_text(json.dumps(held), encoding="utf-8")
    assert app.loop_lock(tmp_path) == app.LOCK_ALIVE

    # A PID that cannot be running: max_pid + 1 on any platform this runs on.
    lock_path(tmp_path).write_text(
        json.dumps({**held, "pid": 2**31 - 1}), encoding="utf-8"
    )

    assert app.loop_lock(tmp_path) == app.LOCK_NONE


# ── the boundaries, swept ────────────────────────────────────────────────────

#: Both boundaries from both sides, including the exact values. The bands are defined so
#: that a threshold must be *crossed* to escalate, which is only visible at the value.
SWEEP = [
    (0.0, app.LIVE),
    (HEARTBEAT - 1, app.LIVE),
    (HEARTBEAT, app.LIVE),
    (HEARTBEAT + 1, app.SLOW),
    (2 * HEARTBEAT - 1, app.SLOW),
    (2 * HEARTBEAT, app.SLOW),
    (2 * HEARTBEAT + 1, app.NOT_RESPONDING),
    (19 * 86400, app.NOT_RESPONDING),
]


@pytest.mark.parametrize(("loop_age", "expected"), SWEEP)
def test_the_boundaries_belong_to_the_calmer_band(
    loop_age: float, expected: str
) -> None:
    assert a_verdict(loop_age=loop_age, heartbeat=HEARTBEAT).state == expected


@pytest.mark.parametrize(("loop_age", "expected"), SWEEP)
def test_the_session_word_and_the_cycle_badge_can_never_disagree(
    cfg_stub, loop_age: float, expected: str
) -> None:
    """The defect, as a property: on 22 Sep the strip said NOT RESPONDING while the CYCLE
    pill said LIVE, because each computed its own answer from a different age."""
    verdict = a_verdict(loop_age=loop_age, heartbeat=HEARTBEAT)

    strip = app.session_strip(cfg_stub, verdict, "checkpoints/live", "live", 1)
    badge = app.region("Cycle", verdict.pill("loop cadence"), "<svg></svg>")

    assert _session_word(strip) == expected
    assert _pill_kinds(badge) == [expected]


def _session_word(html: str) -> str:
    return re.search(
        r'<span class="gb-stat-key">SESSION</span><span class="gb-stat-value"[^>]*>'
        r"([^<]*)</span>",
        html,
    ).group(1)


def _pill_kinds(html: str) -> list[str]:
    return re.findall(r'<span class="gb-pill"[^>]*>([^<]*)<span', html)


def test_moving_the_one_threshold_moves_every_consumer(cfg_stub) -> None:
    """**Negative control 1.** The same age, read under a shorter heartbeat, must move the
    strip and the badge together - which is only true while one number feeds both."""
    slow = a_verdict(loop_age=1000.0, heartbeat=HEARTBEAT)
    silent = a_verdict(loop_age=1000.0, heartbeat=400.0)

    assert slow.state == app.SLOW and silent.state == app.NOT_RESPONDING
    for verdict in (slow, silent):
        strip = app.session_strip(cfg_stub, verdict, "checkpoints/live", "live", 1)
        badge = app.region("Cycle", verdict.pill("loop cadence"), "<svg></svg>")
        assert _session_word(strip) == verdict.state
        assert _pill_kinds(badge) == [verdict.state]
        # The badge too, and this is what catches a consumer that kept its own copy of
        # the threshold: at 1000s it agrees with a 900s heartbeat and not with a 400s one.
        raised = 'class="gb-not-responding"' in strip
        assert raised == (verdict.state == app.NOT_RESPONDING)


# ── no green, and no LIVE, unless the loop is live ───────────────────────────


@pytest.mark.parametrize(
    "inputs",
    [
        {"band_fires": False},
        {"loop_age": 19 * 86400},
        {"lock": app.LOCK_NONE},
        {"lock": app.LOCK_UNREADABLE},
        {"in_session": False},
        {"loop_age": HEARTBEAT + 1},
    ],
    ids=["aside", "not responding", "not running", "unreadable", "shut", "slow"],
)
def test_a_page_that_is_not_live_wears_no_green_and_says_no_live(
    cfg_stub, inputs: dict
) -> None:
    """Every surface at once. The position is losing, so GAIN cannot enter the page
    through the PnL, and the position is protected, so "NO LIVE STOP" cannot enter it
    through D1's stop cell: any green or any LIVE here is a liveness claim."""
    verdict = a_verdict(heartbeat=HEARTBEAT, **inputs)
    assert verdict.state != app.LIVE

    page = a_page(cfg_stub, verdict)

    assert app.GAIN not in page, "green on a page whose loop is not live"
    assert "LIVE" not in _without_showing(page), "the word LIVE on a page not live"
    assert set(_pill_kinds(page)) == {verdict.state}
    assert _session_word(page) == verdict.state


def _without_showing(html: str) -> str:
    """The page minus the ``SHOWING`` stat, which is the one place the word LIVE survives.

    It names **which record stream the page is bound to** - the live directory rather than
    a replay - which is the GB-63 provenance question and not the loop's state. It is set
    in DIM beside the key SHOWING, never in green, and the assertion below keeps it that
    way. Cut here rather than left to match, because a page that is not live must carry no
    other LIVE anywhere.
    """
    without = re.sub(
        r'<span><span class="gb-stat-key">SHOWING</span>.*?</span></span>', "", html
    )
    assert without != html, "the SHOWING stat is gone; this exclusion now hides nothing"
    return without


def test_a_live_page_does_say_live_and_wears_the_green(cfg_stub) -> None:
    """The other direction, so the test above cannot pass by the page rendering nothing."""
    page = a_page(cfg_stub, a_verdict(heartbeat=HEARTBEAT))

    assert app.GAIN in page
    assert set(_pill_kinds(page)) == {app.LIVE}
    assert _session_word(page) == app.LIVE


def test_the_strip_states_the_book_count_without_calling_it_liveness(cfg_stub) -> None:
    """The position count keeps its own word. It was the SESSION word until 23 Sep 2026,
    which is how a page whose loop was nineteen days dead came to read RUNNING."""
    verdict = a_verdict(lock=app.LOCK_NONE, loop_age=19 * 86400)

    strip = app.session_strip(cfg_stub, verdict, "checkpoints/live", "live", 1)

    assert "BOOK" in strip and "1 HELD" in strip
    assert _session_word(strip) == app.NOT_RUNNING
    assert "18 days ago" not in strip and "19 days ago" in strip


# ── the wiring, not just the parts ───────────────────────────────────────────


def test_every_live_pill_on_the_page_is_built_from_the_verdict() -> None:
    """**A test that a function exists is not a test that it is called.** `main` is
    Streamlit and cannot be rendered here, so this reads its source: every `region` call
    must take its source from the verdict, from `forecast_source`, or from an explicitly
    named BACKTEST or REPLAY pill - and `Source(LIVE, ...)` may appear nowhere at all.
    """
    tree = ast.parse(Path(app.__file__).read_text(encoding="utf-8"))
    invented = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "region"
        ):
            continue
        source = node.args[1] if len(node.args) > 1 else None
        if isinstance(source, ast.Name):  # a local built earlier in the same function
            continue
        rendered = ast.unparse(source)
        if not rendered.startswith(("verdict.pill(", "forecast_source(", "Source(")):
            invented.append(rendered)
        if rendered.startswith("Source(") and "LIVE" in rendered:
            invented.append(rendered)

    assert not invented, f"a region builds its own live pill: {invented}"
    assert "Source(LIVE" not in Path(app.__file__).read_text(encoding="utf-8")


def test_the_locals_that_carry_a_pill_come_from_the_verdict() -> None:
    """The other half of the guard above: three regions take their source from a local
    (`live_source`, `cycle_source`, `pos_source`, `activity_source`), so those locals are
    pinned to `verdict.pill(...)` here rather than being waved through as Names."""
    source = Path(app.__file__).read_text(encoding="utf-8")

    for local in ("live_source", "cycle_source", "pos_source", "activity_source"):
        assignment = re.search(rf"^\s*{local} = (.+)$", source, re.MULTILINE)
        assert assignment, f"{local} is gone; this guard now names nothing"
        assert assignment.group(1).startswith(
            "verdict.pill("
        ), f"{local} is not built from the verdict: {assignment.group(1)}"


def test_the_stylesheet_rule_for_a_dead_loop_is_actually_emitted(cfg_stub) -> None:
    """`.gb-not-responding` was defined in LOSS red and rendered by nothing: both branches
    of the old `staleness_html` emitted `.gb-stale`. A rule no code path reaches is a
    stylesheet entry, not a mechanism."""
    silent = a_verdict(loop_age=19 * 86400, heartbeat=HEARTBEAT)
    strip = app.session_strip(cfg_stub, silent, "checkpoints/live", "live", 1)

    assert ".gb-not-responding" in app.stylesheet()
    assert 'class="gb-not-responding"' in strip
    assert "LOOP NOT RESPONDING" in strip and "19 days ago" in strip


def test_a_position_is_managed_only_while_a_loop_is_there_to_manage_it() -> None:
    """Carried from D1. The stop is a DAY order: it expires at the close and only a
    running loop re-arms it, so MANAGED is a claim about the loop as much as the broker.
    """
    rows = _rows(a_verdict())
    running = app.position_table(rows, a_verdict())
    stopped = app.position_table(rows, a_verdict(lock=app.LOCK_NONE))

    assert "MANAGED" in running and "STOP ONLY" not in running
    assert "STOP ONLY" in stopped and "MANAGED" not in stopped


def test_the_target_is_drawn_only_while_a_loop_is_there_to_execute_it() -> None:
    """Carried from D1. The target is not at the broker at all - it is a rule the loop
    applies each cycle - so with no loop a line at that level promises an exit nothing
    will execute."""
    row = _rows(a_verdict())[0]

    running = app.sparkline_svg(CLOSES, row, loop_present=True)
    stopped = app.sparkline_svg(CLOSES, row, loop_present=False)

    # Dashed, and in the rule colour: the entry line is the same colour but solid, so the
    # dash is what separates the target from it.
    target = f'stroke="{app.ORANGE_DIM}" stroke-width="1" stroke-dasharray="3 3"'
    assert target in running, "the target rule is drawn"
    assert target not in stopped
    assert "target" in running and "target" not in stopped
    assert f'stroke="{app.ORANGE}"' in stopped, "the live stop is still drawn"


def test_one_formatter_writes_every_age_on_the_page(cfg_stub) -> None:
    """`444.3h ago` and `26660 MIN AGO` were the same afternoon written two ways, in two
    places, neither of which said nineteen days."""
    verdict = a_verdict(loop_age=444.3 * 3600, broker_age=26660 * 60)

    strip = app.session_strip(cfg_stub, verdict, "checkpoints/live", "live", 1)

    assert app.ago(444.3 * 3600) == "18 days ago"
    assert "18 days ago" in strip
    # The two spellings the strip carried on 22 Sep 2026, by their exact text: a bare
    # "444" would match the LOSS red #EF4444 and pass whatever the strip said.
    assert "444.3h ago" not in strip and "26660 MIN AGO" not in strip
    assert app.ago(float("inf")) == app.EM_DASH, "an age nobody can measure is a dash"
