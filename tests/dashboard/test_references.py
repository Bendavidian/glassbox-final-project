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
        # Case-insensitive: a chart shouts its reference in a label and a caption says it
        # in a sentence, and both are the panel stating it.
        assert any(
            word in html.upper() for word in REFERENCES
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
        assert not any(word in panels[name].upper() for word in REFERENCES), (
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


# ── the radar, D4 ────────────────────────────────────────────────────────────

#: The caption today's axes produce, in full. Pinned rather than recomputed: the caption
#: is built from `RADAR_AXES`, so it can never contradict them - and this is the other
#: half, so an axis changing sides is *noticed* rather than silently redescribed.
RADAR_CAPTION = (
    "4 of 6 axes carry a reference: Direction against the always-long bar, Sharpe, "
    "Return and Drawdown against buy and hold. Cancellation and Flatness have none - "
    "the reference makes no forecast - and say so on the axis. No composite score."
)


def test_the_radar_caption_describes_todays_axes() -> None:
    """The caption it replaces read *Six measured axes, each against its own reference*
    on a chart where five of six had none: true when written, and tied to nothing."""
    assert app.radar_caption() == RADAR_CAPTION
    assert RADAR_CAPTION in app.shape_region(_frames()["results"], ARM)


def test_every_radar_axis_either_names_its_reference_or_says_it_has_none() -> None:
    """No axis prints a bare absolute. Four print ``value vs reference``; the two with no
    analogue print the value and `NO REFERENCE`, which is a statement about themselves.
    """
    axes = {
        point.label: point for point in app.radar_axis_values(_frames()["results"], ARM)
    }

    assert len(axes) == len(app.RADAR_AXES)
    for axis in app.RADAR_AXES:
        point = axes[axis.label]
        if axis.referenced:
            assert point.against.startswith("vs "), f"{axis.label} names no reference"
            assert point.reference_unit == app.REFERENCE_RING
        else:
            assert point.against == app.NO_REFERENCE, f"{axis.label} is unmarked"
            assert point.reference_unit is None


def test_the_radar_places_each_axis_against_its_reference_not_its_own_best_fold() -> (
    None
):
    """**The old scale was self-normalising against ``arm[column].max()``**, so the same
    mean drew a different radius depending on the arm's best fold - and two arms with the
    same shape meant different things. Same means, different maxima, same geometry."""
    steady = pd.concat(
        [
            a_results_frame(folds=2, total_return=0.02),
            a_results_frame(folds=2, model=app.BUY_AND_HOLD, total_return=0.05),
        ]
    )
    spiky = steady.copy()
    arm = (spiky["model"] == ARM).to_numpy()
    spiky.loc[arm, "total_return"] = [0.0, 0.04]  # same mean, double the maximum

    def unit_for(frame: pd.DataFrame) -> float:
        return next(
            point.unit
            for point in app.radar_axis_values(frame, ARM)
            if point.label == "RETURN"
        )

    assert unit_for(steady) == pytest.approx(unit_for(spiky))
    assert unit_for(steady) < app.REFERENCE_RING, "0.02 against 0.05 is behind"


def test_an_axis_the_arm_leads_on_sits_outside_the_reference_ring() -> None:
    """Both directions, on the real study: the arm is behind on return and ahead on
    drawdown, where lower is better - so the shape crosses the ring rather than sitting
    wholly inside it, which is what makes the ring worth drawing."""
    units = {
        point.label: point.unit
        for point in app.radar_axis_values(_frames()["results"], ARM)
    }

    assert units["RETURN"] < app.REFERENCE_RING
    assert units["DIRECTION"] < app.REFERENCE_RING
    assert units["DRAWDOWN"] > app.REFERENCE_RING, "0.0231 drawdown against 0.0540"


def test_an_arm_that_forecasts_nothing_does_not_draw_the_best_flatness() -> None:
    """**Found while choosing the scale, and it is the older defect.** Flatness is
    mean|forecast| / mean|actual|, where 1.0 is right-sized; the axis was scored as though
    lower were better, so persistence - which forecasts exactly nothing and measures 0.0 -
    drew the best flatness on the page while making no forecast at all.
    """

    def flatness_unit(value: float) -> float:
        frame = a_results_frame(folds=2, flatness=value)
        return next(
            point.unit
            for point in app.radar_axis_values(frame, ARM)
            if point.label == "FLATNESS"
        )

    assert flatness_unit(1.0) == pytest.approx(1.0), "right-sized is the best flatness"
    assert flatness_unit(0.0) == pytest.approx(0.0), "forecasting nothing is not best"
    assert flatness_unit(2.0) == pytest.approx(0.0), "twice right-sized is not best"


def test_the_radar_draws_one_reference_mark_per_referenced_axis() -> None:
    """The reference is a tick per axis, not a second polygon: a closed shape through four
    of six vertices would draw a line across the two axes that have no reference."""
    svg = app.radar_svg(_frames()["results"], ARM)

    ticks = svg.count(f'stroke="{app.REFERENCE_MARK}"')
    assert ticks == sum(1 for axis in app.RADAR_AXES if axis.referenced) == 4
    assert svg.count("<polygon") == 5, "four rings and one shape, no reference polygon"


def test_the_radar_is_drawn_in_the_ramp_and_never_in_status_colour() -> None:
    """The polygon was hardcoded `fill=GAIN_FILL stroke=GAIN`, so an arm behind its
    reference on every axis that has one still arrived green."""
    svg = app.radar_svg(_frames()["results"], ARM)

    assert f'stroke="{app.ARM_MARK}"' in svg
    assert app.GAIN not in svg and app.GAIN_FILL not in svg
    assert not [c for c in app.STATUS_COLOURS if c.lower() in svg.lower()]


@pytest.mark.parametrize("panel", ["cumulative_region", "fold_region", "shape_region"])
def test_the_region_hands_the_chart_the_reference_rows(panel: str) -> None:
    """**Where the reference was dropped.** Both regions filtered the frame to the arm and
    passed that to the chart, so the buy-and-hold rows were gone one line before the chart
    could draw them. The pill still takes the filtered rows - it states the arm's fold
    range - and the chart takes the whole frame."""
    frames = _frames()
    html = getattr(app, panel)(frames[FRAMES[panel]], ARM)

    assert "BUY AND HOLD" in html.upper()
    assert "BACKTEST" in html, "the pill still names the source"
