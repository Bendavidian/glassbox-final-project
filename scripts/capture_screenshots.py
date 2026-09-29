"""Capture the console frames the architecture report uses, from a running dashboard.

**Reconstructed on 28 Sep 2026 from a session transcript.** The six frames committed in
``report/screenshots/`` by ``5109431`` were produced on 1 Sep 2026 by ``shoot6.py``, a
headless Playwright script written in a Claude Code session's scratchpad and run twice:
at 14:51 UTC against ``checkpoints/live`` and at 14:53 UTC against
``checkpoints/fits-demo``. It was never committed and the scratchpad has since been
deleted, so six committed figures had no producing code a clone could run. The text below
is rebuilt from the tool call that wrote it (transcript ``32d4fbc3``, line 6003) and the
edit that added the warm-up (line 6008), with four changes made before it could be safe:

1. **``--out`` is required and has no default, and nothing is ever overwritten.** The
   original wrote straight into ``report/screenshots/``, including ``01-overview.png``,
   ``03-forecast-path.png`` and ``06-reflow-1280.png`` - the three frames CAPTIONS.md
   retains as the only evidence of defects that have since been fixed. No shot here is
   named after a retained frame, and a target that already exists is refused.
2. **``--decision`` names the record to expand.** The original clicked the first expander
   on the page, which is whichever decision happens to sort newest.
3. **Preconditions refuse rather than warn.** The original photographed whatever was
   there. Frame 02 is refused if the strip reads NOT RUNNING or NOT RESPONDING, or if the
   positions or activity region is empty: an empty positions panel is indistinguishable
   from a broker read failure (GB-55 §55.11), and a frame of one proves nothing.
4. **The young-session frame (07) is its own shot.** It must be taken within the first
   few readings of a session: the dashboard adds one equity reading per refresh, and the
   original deliberately held the page open so readings would build up - the opposite of
   the condition 07 exists to show. Take 07 first, before any other capture has attached
   a client, and do not pass a long ``--settle`` for it.

Usage - start the dashboard headless, then point this at it::

    streamlit run glassbox/dashboard/app.py --server.port 8540 --server.headless true \\
        -- --state-dir checkpoints/live --source live
    python scripts/capture_screenshots.py http://localhost:8540 --out <dir> \\
        --shot 07 02 03b 04 06b --decision 20260922-NVDA

    streamlit run glassbox/dashboard/app.py --server.port 8541 --server.headless true \\
        -- --state-dir checkpoints/fits-demo --source replay:fold-13
    python scripts/capture_screenshots.py http://localhost:8541 --out <dir> \\
        --shot 05 --decision <YYYYMMDD-SYMBOL>

Every precondition for the requested shots is checked before the first file is written, so
a refusal leaves nothing behind. Needs the ``dev`` extra and a browser:
``pip install -e ".[dev]"`` then ``python -m playwright install chromium``.

Exit codes: 0 every requested frame written, 2 refused (a precondition failed or a target
exists), 3 the environment cannot run it.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from glassbox import records
    from glassbox.dashboard.app import (
        EQUITY_THIN_READINGS,
        NOT_RESPONDING,
        NOT_RUNNING,
        provenance_label,
    )
except ImportError:  # pragma: no cover - the wrong interpreter
    print(
        "capture_screenshots: run this with the project's interpreter, e.g.\n"
        "  .venv/Scripts/python.exe scripts/capture_screenshots.py",
        file=sys.stderr,
    )
    raise SystemExit(3) from None

WIDE = {"width": 1920, "height": 1080}
NARROW = {"width": 1280, "height": 900}

#: Retained and dated by CAPTIONS.md. Nothing this script writes may carry these names.
RETAINED = frozenset({"01-overview.png", "03-forecast-path.png", "06-reflow-1280.png"})

#: The region labels and the readings caption this script anchors on, as the dashboard
#: renders them. ``tests/test_capture_screenshots.py`` pins each against the dashboard
#: source, so a renamed region fails there rather than in a frame of the wrong thing.
POSITIONS = "Positions"
ACTIVITY = "Recent activity"
SESSION_EQUITY = "Session equity"
FORECASTS = "Forecast paths"
PER_CHANNEL = "PER-CHANNEL CONTRIBUTION"
PER_FREQUENCY = "PER-FREQUENCY CONTRIBUTION"
READINGS = re.compile(r"(\d+) READINGS THIS SESSION")

#: The live frames read the broker's record stream; the spectral frame reads a replay.
BROKER = provenance_label(records.LIVE)
REPLAY = "REPLAY"

#: The curve and the opening-balance rule need two readings to be drawn at all.
YOUNG_AT_LEAST = 2

DECISION_ID = re.compile(r"^(\d{4})(\d{2})(\d{2})-([A-Z.]+)$")


@dataclass(frozen=True)
class Shot:
    """One frame: its file, its viewport, what to scroll to, and what it requires."""

    filename: str
    viewport: dict[str, int]
    anchor: str | None
    showing: str
    needs_decision: bool = False
    needs_loop_and_book: bool = False
    young: bool = False


#: In the order they are taken. 07 first, before the run has added readings of its own.
SHOTS: dict[str, Shot] = {
    "07": Shot("07-young-session.png", WIDE, SESSION_EQUITY, BROKER, young=True),
    "02": Shot(
        "02-activity-rows.png", WIDE, POSITIONS, BROKER, needs_loop_and_book=True
    ),
    "03b": Shot("03b-forecast-path-fixed.png", WIDE, FORECASTS, BROKER),
    "04": Shot("04-attribution.png", WIDE, PER_CHANNEL, BROKER, needs_decision=True),
    "05": Shot(
        "05-spectral-fits.png", WIDE, PER_FREQUENCY, REPLAY, needs_decision=True
    ),
    "06b": Shot("06b-reflow-1280-fixed.png", NARROW, None, BROKER),
}

# A refusal in the code rather than a sentence in the header: if a shot is ever renamed
# onto a retained frame, the script does not import.
if RETAINED & {shot.filename for shot in SHOTS.values()}:  # pragma: no cover
    raise SystemExit("capture_screenshots: a shot is named after a retained frame")


class Refused(Exception):
    """A precondition failed. Nothing has been written."""


@dataclass(frozen=True)
class PageState:
    """What the page showed when it was read, before any frame was taken."""

    session: str
    showing: str
    positions_empty: bool
    activity_empty: bool
    readings: int | None
    decision_matches: int


def expander_prefix(decision: str) -> str:
    """``20260922-NVDA`` as its expander title begins: ``2026-09-22  NVDA  ``.

    The dashboard titles each expander from ``as_of`` and ``symbol`` (``expander_title``),
    which is exactly what a decision id carries.
    """
    match = DECISION_ID.match(decision)
    if match is None:
        raise Refused(f"--decision {decision!r} is not of the form YYYYMMDD-SYMBOL")
    year, month, day, symbol = match.groups()
    return f"{year}-{month}-{day}  {symbol}  "


def targets(out: Path, keys: list[str]) -> dict[str, Path]:
    """Where each requested shot will be written. Refuses any target that exists."""
    paths = {key: out / SHOTS[key].filename for key in keys}
    existing = sorted(str(path) for path in paths.values() if path.exists())
    if existing:
        raise Refused(f"refusing to overwrite: {', '.join(existing)}")
    return paths


def refusals(keys: list[str], state: PageState, decision: str | None) -> list[str]:
    """Every reason the requested shots cannot be taken from this page. Empty means go."""
    reasons: list[str] = []
    for key in keys:
        shot = SHOTS[key]
        if not state.showing.startswith(shot.showing):
            reasons.append(
                f"{key}: the strip reads SHOWING {state.showing}, and this frame needs "
                f"{shot.showing}"
            )
        if shot.needs_loop_and_book:
            if state.session in (NOT_RUNNING, NOT_RESPONDING):
                reasons.append(f"{key}: the strip reads {state.session}")
            if state.positions_empty:
                reasons.append(
                    f"{key}: the positions panel is empty, which is indistinguishable "
                    "from a broker read failure"
                )
            if state.activity_empty:
                reasons.append(f"{key}: the activity region has nothing recorded")
        if shot.needs_decision:
            if decision is None:
                reasons.append(
                    f"{key}: needs --decision; the first expander is not a choice"
                )
            elif state.decision_matches != 1:
                reasons.append(
                    f"{key}: {state.decision_matches} expanders match --decision "
                    f"{decision}, and exactly one must"
                )
        if shot.young:
            if state.readings is None:
                reasons.append(f"{key}: no reading count is drawn on the page")
            elif not YOUNG_AT_LEAST <= state.readings < EQUITY_THIN_READINGS:
                reasons.append(
                    f"{key}: {state.readings} readings this session; the young-session "
                    f"frame needs {YOUNG_AT_LEAST} to {EQUITY_THIN_READINGS - 1}"
                )
    return reasons


def _label(page, label: str):
    """The section label that reads exactly ``label``, ignoring the CSS case transform."""
    exact = re.compile(f"^{re.escape(label)}$", re.IGNORECASE)
    return page.locator(".gb-label", has_text=exact)


def _region(page, label: str):
    """The region whose label is exactly ``label``."""
    return page.locator(".gb-region").filter(has=_label(page, label))


def _strip(page) -> dict[str, str]:
    """The status strip as ``{key: value}``, read from the page rather than recomputed."""
    return page.evaluate("""() => Object.fromEntries(
            [...document.querySelectorAll('.gb-strip > span')]
              .filter(s => s.querySelector('.gb-stat-key'))
              .map(s => [s.querySelector('.gb-stat-key').innerText.trim(),
                         s.querySelector('.gb-stat-value').innerText.trim()]))""")


def _expander(page, decision: str):
    """The expander for one decision, matched on the start of its title."""
    title = re.compile("^" + re.escape(expander_prefix(decision)))
    summary = page.locator("summary", has_text=title)
    return page.locator('[data-testid="stExpander"]').filter(has=summary)


def _empty(region) -> bool:
    """Absent, or rendering the body a region draws when it has nothing to show."""
    return not region.count() or bool(region.locator(".gb-empty").count())


def read_state(page, decision: str | None) -> PageState:
    strip = _strip(page)
    equity = _region(page, SESSION_EQUITY)
    # text_content, not inner_text: the count is drawn inside an SVG.
    text = equity.first.text_content() if equity.count() else None
    count = READINGS.search(text or "")
    return PageState(
        session=strip.get("SESSION", ""),
        showing=strip.get("SHOWING", ""),
        positions_empty=_empty(_region(page, POSITIONS)),
        activity_empty=_empty(_region(page, ACTIVITY)),
        readings=int(count.group(1)) if count else None,
        decision_matches=_expander(page, decision).count() if decision else 0,
    )


def _centre_mouse(page) -> None:
    size = page.viewport_size
    page.mouse.move(size["width"] // 2, size["height"] // 2)


def _scroll_to(element, page, target: int = 90, tries: int = 40) -> bool:
    """Wheel the page until ``element`` sits ``target`` px from the top. As on 1 Sep."""
    _centre_mouse(page)
    for _ in range(tries):
        if not element.count():
            return False
        box = element.bounding_box()
        if box is None:
            return False
        delta = box["y"] - target
        if abs(delta) < 25:
            return True
        page.mouse.wheel(0, max(-900, min(900, delta)))
        time.sleep(0.35)
    return True


def _to_top(page) -> None:
    _centre_mouse(page)
    page.mouse.wheel(0, -20000)
    time.sleep(2)


def capture(
    url: str, out: Path, keys: list[str], decision: str | None, settle: int
) -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            'capture_screenshots: needs the dev extra - pip install -e ".[dev]" - then '
            "python -m playwright install chromium",
            file=sys.stderr,
        )
        return 3

    paths = targets(out, keys)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport=WIDE)
        page.goto(url, timeout=120_000)
        page.wait_for_load_state("networkidle")
        # Every refresh of an attached page adds an equity reading. The 1 Sep run held the
        # page open so the curve would be sampled; 07 needs the opposite, so keep it short.
        time.sleep(settle)

        reasons = refusals(keys, read_state(page, decision), decision)
        if reasons:
            browser.close()
            raise Refused("; ".join(reasons))

        out.mkdir(parents=True, exist_ok=True)
        for key in [k for k in SHOTS if k in keys]:
            shot = SHOTS[key]
            page.set_viewport_size(shot.viewport)
            time.sleep(2)
            _to_top(page)
            anchor = None
            if shot.needs_decision:
                expander = _expander(page, decision)
                expander.locator("summary").click()
                time.sleep(4)
                # Scoped to the one expander: a collapsed one can carry the same text hidden.
                anchor = expander.get_by_text(shot.anchor).first
            elif shot.anchor:
                anchor = _label(page, shot.anchor).first
            if anchor is not None and not _scroll_to(anchor, page):
                browser.close()
                raise Refused(f"{key}: {shot.anchor!r} is not on the page")
            page.screenshot(path=str(paths[key]))
            print(f"wrote {paths[key]}")
        browser.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/capture_screenshots.py")
    parser.add_argument("url", help="a running dashboard, e.g. http://localhost:8540")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="directory to write into; required, and an existing file is never replaced",
    )
    parser.add_argument("--shot", nargs="+", required=True, choices=list(SHOTS))
    parser.add_argument("--decision", help="the record to expand, e.g. 20260922-NVDA")
    parser.add_argument(
        "--settle",
        type=int,
        default=12,
        help="seconds to let the page render before reading it (default 12, as on 1 Sep)",
    )
    args = parser.parse_args(argv)

    try:
        return capture(args.url, args.out, args.shot, args.decision, args.settle)
    except Refused as refusal:
        print(f"capture_screenshots: refused - {refusal}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
