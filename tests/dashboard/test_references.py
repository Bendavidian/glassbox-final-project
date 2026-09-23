"""Every performance panel states the reference its result is measured against.

**D3, 23 Sep 2026.** Spec §7.3: *no result is ever reported as an absolute number; every
result is reported as a delta against a stated reference.* The console reported two of
them as absolutes. The cumulative equity panel drew ``+45.51%`` in green against its own
starting capital while the same file held 938 buy-and-hold rows over identical dates
chaining to ``+125.19%``; the fold panel drew green bars against 0%, and fold 13 - the
demonstration fold - showed ``+2.03%`` as a green bar on a quarter when the market made
``+9.36%``.

Both files already carried the reference. Both panels filtered it out on the way to the
chart, in `arm_rows`, one line before it was drawn.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from glassbox.dashboard import app
from tests.dashboard.test_app import a_daily_frame, a_results_frame

ARM = "dlinear"

#: What a stated reference looks like on the face of a panel. Spec §7.3 names three:
#: persistence for error, always-long for direction, buy and hold for return and Sharpe.
REFERENCES = ("BUY AND HOLD", "ALWAYS-LONG", "PERSISTENCE")

#: The frame each panel is drawn from. Hand-written, and the test below asserts it covers
#: every ``*_region`` the module exports - so a new panel fails here until somebody says
#: which artefact it reads, rather than being skipped by an enumeration that never saw it.
FRAMES = {
    "shape_region": "results",
    "fold_region": "results",
    "cumulative_region": "daily",
    "calendar_region": "daily",
}

#: **Panels that still do not state a reference, each with the reason it is still here.**
#: The goal is an empty dict. Anything in it is reported as unfixed rather than passing
#: quietly, and `test_an_exemption_that_stopped_being_true_fails` removes the entry's
#: cover the moment the panel starts stating one.
UNREFERENCED = {
    "shape_region": (
        "Five of its six axes are absolutes: only DIRECTION carries a reference "
        "(direction_reference). SHARPE, RETURN and DRAWDOWN could take buy and hold from "
        "the same file; CANCELLATION and FLATNESS have no buy-and-hold analogue, because "
        "the reference makes no forecast. Not started on 23 Sep 2026: half of it would be "
        "a panel that states a reference on four axes and not on two, which is the "
        "defect it has now."
    ),
    "calendar_region": (
        "Every tile is a day's change against nothing. The reference exists in the same "
        "file - buy and hold has a row for each of those dates - so the fix is a second "
        "value per tile or a tile of the difference, which is a redesign of the "
        "encoding rather than a filter change. Not started on 23 Sep 2026."
    ),
}


def _frames() -> dict[str, pd.DataFrame]:
    return {
        "results": app.reference_rows(app.load_results()),
        "daily": app.load_daily_equity(),
    }


def rendered_panels() -> dict[str, str]:
    """Every performance panel the module exports, rendered from the real artefacts.

    The same shape as the colour guard that enumerates every chart builder: the list comes
    from ``dir(app)``, so a panel added without a reference cannot pass by being absent
    from a list somebody wrote by hand.
    """
    found = {name for name in dir(app) if name.endswith("_region")}
    unclassified = found - set(FRAMES)
    assert not unclassified, (
        f"a panel nobody said which artefact it reads: {sorted(unclassified)}. "
        "Add it to FRAMES; if it states no reference it belongs in UNREFERENCED too."
    )
    frames = _frames()
    return {
        name: getattr(app, name)(frames[FRAMES[name]], ARM) for name in sorted(found)
    }


def test_every_performance_panel_states_its_reference_or_is_named_as_unfixed() -> None:
    """The guard. A panel either names the thing its result is measured against, or it is
    in `UNREFERENCED` with a reason a reader can act on."""
    for name, html in rendered_panels().items():
        if name in UNREFERENCED:
            assert UNREFERENCED[name].strip(), f"{name} is exempt with no reason given"
            continue
        assert any(
            word in html for word in REFERENCES
        ), f"{name} reports a result against nothing; §7.3 forbids it"
        # Naming the reference is not stating it. A panel whose artefact has lost its
        # reference rows says "BUY AND HOLD NOT IN THIS FILE", which names it and reports
        # an absolute anyway - and would satisfy the line above on the strength of the
        # words in its own apology.
        assert (
            "NOT IN THIS FILE" not in html
        ), f"{name} could not find its reference and drew the result anyway"


def test_an_exemption_that_stopped_being_true_fails() -> None:
    """An exemption list nobody can leave is a list that outlives its reasons. A panel
    that starts stating a reference fails here until its entry is deleted."""
    panels = rendered_panels()

    for name in UNREFERENCED:
        assert not any(word in panels[name] for word in REFERENCES), (
            f"{name} now states a reference - delete its UNREFERENCED entry, which is "
            "claiming otherwise"
        )


# ── the two panels D3 fixed ──────────────────────────────────────────────────


def test_the_cumulative_panel_draws_buy_and_hold_beside_the_arm() -> None:
    """The arm's own figure is no longer presented as a result on its own: both curves,
    both levels, and the gap between them as a number."""
    svg = app.cumulative_equity_svg(app.load_daily_equity(), ARM)

    assert svg.count("<polyline") == 2, "one curve is the arm's own number, again"
    assert f'stroke="{app.REFERENCE_MARK}"' in svg
    assert "BUY AND HOLD +125.19%" in svg
    assert "DLINEAR +45.51%" in svg
    assert "-79.68 PTS" in svg, "the delta §7.3 asks for"


def test_the_cumulative_panel_says_when_its_reference_is_missing() -> None:
    """A file with no reference rows does not get a quiet single curve: the panel names
    the reference it could not find, because that absence is the reader's business."""
    svg = app.cumulative_equity_svg(a_daily_frame(folds=2), ARM)

    assert svg.count("<polyline") == 1
    assert "BUY AND HOLD NOT IN THIS FILE" in svg


def test_a_fold_that_lost_to_the_market_cannot_read_as_a_win() -> None:
    """**Fold 13, the demonstration fold.** +2.03% is a green bar against zero and a loss
    of seven points against the quarter it was measured in. The reference rule is drawn
    across the bar, so the comparison is one look rather than two numbers held in a head.
    """
    rows = pd.concat(
        [
            a_results_frame(folds=1, total_return=0.0203),
            a_results_frame(folds=1, model=app.BUY_AND_HOLD, total_return=0.0936),
        ]
    )

    svg = app.fold_bars_svg(rows, ARM)

    bar = re.search(rf'<rect x="[^"]*" y="([^"]*)"[^>]*fill="{app.ARM_MARK}"', svg)
    rule = re.search(
        rf'<line x1="[^"]*" y1="([^"]*)"[^>]*stroke="{app.REFERENCE_MARK}"', svg
    )
    assert bar and rule, "the bar and its reference are both drawn"
    # Smaller y is higher on the canvas, and the market's quarter is the higher mark.
    assert float(rule.group(1)) < float(bar.group(1))
    assert "0 OF 1 FOLDS BEAT BUY AND HOLD" in svg


def test_the_fold_panel_counts_the_folds_that_beat_the_market() -> None:
    """On the real study: four of sixteen, and the worst margin is the fold the demo is
    built around. Neither number existed on this panel before 23 Sep 2026."""
    svg = app.fold_bars_svg(app.reference_rows(app.load_results()), ARM)

    assert "4 OF 16 FOLDS BEAT BUY AND HOLD" in svg
    assert "WORST f13 -7.33 PTS" in svg
    assert "BEST f1 +5.03 PTS" in svg


def test_neither_fixed_panel_carries_a_status_colour() -> None:
    """R11, and the reason: green above a zero line answers *did it gain*, on two panels
    whose question is *did it beat the reference*. Those have different answers here - the
    arm gained 45% and lost 80 points - and the chart may not answer the easier one.

    `test_status_colour_never_enters_a_ramp_chart` holds this for every ramp surface; this
    states it for these two by name, because it is the reason they were rebuilt.
    """
    daily, results = app.load_daily_equity(), app.reference_rows(app.load_results())
    panels = {
        "cumulative": app.cumulative_equity_svg(daily, ARM),
        "folds": app.fold_bars_svg(results, ARM),
    }

    for name, svg in panels.items():
        found = [c for c in app.STATUS_COLOURS if c.lower() in svg.lower()]
        assert not found, f"{name} carries status colour {found}"
        assert any(c in svg for c in app.RAMP), f"{name} lost the ramp it encodes with"


@pytest.mark.parametrize("panel", ["cumulative_region", "fold_region"])
def test_the_region_hands_the_chart_the_reference_rows(panel: str) -> None:
    """**Where the reference was dropped.** Both regions filtered the frame to the arm and
    passed that to the chart, so the buy-and-hold rows were gone one line before the chart
    could draw them. The pill still takes the filtered rows - it states the arm's fold
    range - and the chart takes the whole frame."""
    frames = _frames()
    html = getattr(app, panel)(frames[FRAMES[panel]], ARM)

    assert "BUY AND HOLD" in html
    assert "BACKTEST" in html, "the pill still names the source"
