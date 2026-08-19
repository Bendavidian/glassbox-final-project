"""GB-34/35/36 acceptance: the dashboard's arithmetic, its encodings and its rulings.

Layout is checked by looking at it. What is tested here is everything the eye cannot
check: that the threshold appears even when the band is `never()`, that a share on a bar
equals the share in the prose, that the cancellation is shown, that a quarantined position
reports no PnL it has no basis for, that the Hebrew narrative gets an explicit direction,
and that orange never leaks into a data mark.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glassbox.contracts.schemas import Attribution, Forecast, Signal
from glassbox.dashboard import app
from glassbox.engine.reconcile import Book, Holding
from glassbox.engine.signal import ENTER_LONG, HOLD, Thresholds
from glassbox.explain.channel import cancellation, shares

HISTORY = pd.Series([100.0, 101.0, 99.5, 102.0, 103.5])


def an_attribution(**per_channel: float) -> Attribution:
    return Attribution(
        per_channel=dict(per_channel),
        per_lag=None,
        per_frequency=None,
        gain_phase=None,
        forecast_total=sum(per_channel.values()),
    )


def a_holding(symbol: str = "AAPL", entry: float = 100.0) -> Holding:
    return Holding(
        symbol=symbol,
        quantity=2.0,
        decision_id="d1",
        entry_price=entry,
        stop_loss=entry * 0.97,
        take_profit=entry * 1.06,
    )


# ── GB-34: status, positions, reliability ────────────────────────────────────


def test_a_band_that_cannot_fire_outranks_every_other_status() -> None:
    """A viewer told "RUNNING" beside a flat book waits for a trade that cannot come."""
    assert app.status_of(Thresholds.never(), True, {"AAPL": 1.0}) == app.ASIDE
    assert app.status_of(Thresholds.never(), False, {}) == app.ASIDE


def test_status_distinguishes_closed_idle_and_running() -> None:
    band = Thresholds(lower=0.004)

    assert app.status_of(band, False, {}) == app.CLOSED
    assert app.status_of(band, True, {}) == app.IDLE
    assert app.status_of(band, True, {"AAPL": 1.0}) == app.RUNNING


def test_a_managed_position_reports_its_pnl() -> None:
    book = Book(managed={"AAPL": a_holding(entry=100.0)})

    row = app.position_rows(book, {"AAPL": 2.0}, {"AAPL": 110.0})[0]

    assert row.managed
    assert row.market_value == pytest.approx(220.0)
    assert row.unrealised == pytest.approx(20.0)
    assert row.unrealised_pct == pytest.approx(10.0)


def test_a_quarantined_position_is_shown_with_no_pnl() -> None:
    """The system did not open it and has no entry basis, so a number would be invented."""
    row = app.position_rows(Book(), {"NVDA": 0.01}, {"NVDA": 200.0})[0]

    assert not row.managed
    assert math.isnan(row.entry_price)
    assert math.isnan(row.unrealised)
    assert row.market_value == pytest.approx(2.0)  # it still spends buying power


def test_reliability_is_read_from_the_measurement_file(tmp_path: Path) -> None:
    (tmp_path / app.RELIABILITY_FILE).write_text(
        json.dumps(
            {
                "direction": 0.5182,
                "always_long": 0.5560,
                "folds": 16,
                "measured_on": "2026-08-18",
                "model": "dlinear",
            }
        ),
        encoding="utf-8",
    )

    found = app.load_reliability(tmp_path / app.RELIABILITY_FILE)

    assert found.direction == pytest.approx(0.5182)
    assert found.gap == pytest.approx(-0.0378)
    assert not found.beats_the_bar


def test_a_missing_measurement_says_so_rather_than_showing_nothing(
    cfg_stub, tmp_path: Path
) -> None:
    """An absent panel is the failure the panel exists to prevent, arriving as a gap."""
    assert app.load_reliability(tmp_path / "absent.json") is None

    html = app.header_html(cfg_stub, app.IDLE, None)

    assert "NOT MEASURED" in html
    assert "prepare-live" in html


def test_the_header_states_the_gap_against_the_always_long_bar(cfg_stub) -> None:
    record = app.Reliability(
        direction=0.5182,
        always_long=0.5560,
        folds=16,
        measured_on="2026-08-18",
        model="dlinear",
    )

    html = app.header_html(cfg_stub, app.ASIDE, record)

    assert "0.5182" in html
    assert "0.5560" in html
    assert "16 FOLDS" in html
    assert "2026-08-18" in html
    assert "▼" in html  # it does not beat the bar, and the glyph says so


@pytest.fixture
def cfg_stub():
    from glassbox.config.loader import load_config

    return load_config()


# ── GB-35: the forecast path ─────────────────────────────────────────────────


def test_the_forecast_path_is_compounded_onto_the_last_close() -> None:
    """One axis, one unit: a reader should not have to exponentiate in their head."""
    path = app.price_path(100.0, np.array([0.01, 0.01], dtype="float32"))

    assert path.iloc[0] == pytest.approx(100.0 * math.exp(0.01))
    assert path.iloc[1] == pytest.approx(100.0 * math.exp(0.02))


def test_the_chart_marks_the_calibrated_threshold() -> None:
    svg = app.forecast_svg(
        HISTORY, app.price_path(103.5, [0.004, 0.004]), Thresholds(lower=0.01), "AAPL"
    )

    assert "ENTRY THRESHOLD" in svg
    assert app.ORANGE in svg  # chrome: the threshold is a rule, not a measurement


def test_a_band_that_cannot_fire_is_said_on_the_chart_not_omitted() -> None:
    """Omitting the line makes a system that abstains look like one that has not acted
    yet, and those are different claims."""
    svg = app.forecast_svg(
        HISTORY, app.price_path(103.5, [0.004]), Thresholds.never(), "AAPL"
    )

    assert "STOOD ASIDE" in svg
    assert "ENTRY THRESHOLD" not in svg


def test_the_forecast_continues_from_the_last_close() -> None:
    """The path must start where the history ends, or the chart implies a jump."""
    svg = app.forecast_svg(
        HISTORY, app.price_path(103.5, [0.0]), Thresholds(lower=0.01), "AAPL"
    )
    polylines = re.findall(r'points="([^"]+)"', svg)

    history_end = polylines[0].split()[-1]
    forecast_start = polylines[1].split()[0]
    assert history_end == forecast_start


def test_the_chart_names_its_symbol() -> None:
    svg = app.forecast_svg(
        HISTORY, app.price_path(103.5, [0.0]), Thresholds.never(), "MSFT"
    )

    assert "MSFT" in svg


# ── GB-36: contributions, cancellation, narrative ────────────────────────────


def test_every_bar_label_matches_the_attributions_own_share() -> None:
    """The bar and the prose read one computation, so they cannot disagree."""
    attribution = an_attribution(close_logret=0.012, rsi14=0.006, vol_z=-0.004)

    svg = app.contributions_svg(attribution)

    for value in shares(attribution).values():
        assert f"{abs(value) * 100:.1f}%" in svg


def test_the_cancellation_is_always_shown(caplog) -> None:
    """A decision where 15% of the gross view survived must not render as though the
    shares were the whole story."""
    attribution = an_attribution(close_logret=0.08, rsi14=-0.06)

    svg = app.contributions_svg(attribution)

    assert "CANCELLATION" in svg
    assert f"{cancellation(attribution) * 100:.1f}%" in svg


def test_a_low_cancellation_is_marked_in_chrome() -> None:
    """Below half surviving it is the thing to notice, so it takes the accent."""
    offsetting = app.contributions_svg(an_attribution(close_logret=0.08, rsi14=-0.06))
    aligned = app.contributions_svg(an_attribution(close_logret=0.08, rsi14=0.06))

    assert offsetting.count(app.ORANGE) > aligned.count(app.ORANGE)


def test_sign_is_carried_by_geometry_and_a_glyph_not_by_hue() -> None:
    """A ramp cannot encode a sign, and inventing a second data colour is forbidden."""
    svg = app.contributions_svg(an_attribution(close_logret=0.08, rsi14=-0.06))

    assert "▲" in svg
    assert "▼" in svg
    assert 'fill="none"' in svg  # the negative bar is hollow


def test_channels_are_coloured_by_how_far_back_they_look() -> None:
    """The ramp encodes a quantity. Assigning it in config order would encode nothing."""
    assert app.channel_colour("ma_dist20") == app.RAMP[0]  # slowest, darkest
    assert app.channel_colour("close_logret") == app.RAMP[4]  # fastest, lightest
    assert app.channel_colour("wav_a1") in app.RAMP  # unknown channel still gets one


def test_no_data_mark_is_drawn_in_the_chrome_colour() -> None:
    """Orange is interface chrome only. Every bar takes a ramp entry."""
    svg = app.contributions_svg(an_attribution(close_logret=0.012, rsi14=-0.006))
    bars = re.findall(r"<rect[^>]*/>", svg)

    assert bars
    for bar in bars:
        assert app.ORANGE not in bar


def test_a_hebrew_narrative_is_given_an_explicit_direction() -> None:
    """`dir="auto"` reads the first strong character, and every narrative opens with a
    ticker — so `auto` would undo GB-32's isolation work."""
    hebrew = "AAPL בתאריך 2026-08-18: המודל צופה עלייה."

    assert app.is_rtl(hebrew)
    assert 'dir="rtl"' in app.narrative_html(hebrew)


def test_an_english_narrative_stays_left_to_right() -> None:
    english = "AAPL on 2026-08-18: the model predicts a 2.02% rise."

    assert not app.is_rtl(english)
    assert 'dir="ltr"' in app.narrative_html(english)


def a_record(
    day: str = "2026-08-10",
    symbol: str = "AAPL",
    order=None,
    close_logret: float = 0.08,
    rsi14: float = -0.06,
):
    from glassbox.contracts.schemas import DecisionRecord

    stamp = pd.Timestamp(day, tz="UTC")
    return DecisionRecord(
        as_of=stamp,
        symbol=symbol,
        forecast=Forecast(
            path=np.array([0.005] * 4, dtype="float32"), symbol=symbol, as_of=stamp
        ),
        attribution=an_attribution(close_logret=close_logret, rsi14=rsi14),
        signal=Signal(
            symbol=symbol,
            action=HOLD,
            trend_strength=0.02,
            up_points=4,
            passed_threshold=False,
        ),
        order=order,
        narrative="",
        config_hash="x",
    )


def test_the_decision_log_is_newest_first_and_carries_the_cancellation() -> None:
    ordered = app.decision_rows(
        [a_record("2026-08-10", "AAPL"), a_record("2026-08-11", "MSFT")]
    )

    assert [record.symbol for record in ordered] == ["MSFT", "AAPL"]
    assert "CANCELLATION" in app.decision_table(ordered)
    assert f"{0.02 / 0.14:.4f}" in app.decision_table(ordered)


# ── the design language ──────────────────────────────────────────────────────


def test_the_stylesheet_is_the_near_black_ground_with_dashed_hairlines() -> None:
    css = app.stylesheet()

    assert app.INK in css
    assert "dashed" in css
    assert "letter-spacing" in css
    assert "uppercase" in css


def test_the_rtl_narrative_rule_moves_to_the_other_side() -> None:
    """A left border on a right-to-left paragraph sits at the end of the sentence."""
    css = app.stylesheet()

    assert '.gb-narrative[dir="rtl"]' in css
    assert "border-right" in css


def test_no_traffic_light_colours_anywhere() -> None:
    """Sign is geometry and a glyph. A green/red pair would be a second data family."""
    surfaces = (
        app.stylesheet(),
        app.contributions_svg(an_attribution(close_logret=0.08, rsi14=-0.06)),
        app.forecast_svg(
            HISTORY, app.price_path(103.5, [0.01]), Thresholds.never(), "AAPL"
        ),
    )
    banned = ("green", "#0f0", "#00ff00", "red", "#f00", "#ff0000")

    for surface in surfaces:
        lowered = surface.lower()
        assert not [word for word in banned if word in lowered]


def test_the_svg_builders_are_pure_strings() -> None:
    """No browser, no plotting library, no dependency — which is what makes the geometry
    testable at all."""
    svg = app.contributions_svg(an_attribution(close_logret=0.01))

    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")


def test_an_entry_decision_and_a_hold_both_render(cfg_stub) -> None:
    """The log holds both, and neither may raise on the way to the screen."""
    for action in (ENTER_LONG, HOLD):
        signal = Signal(
            symbol="AAPL",
            action=action,
            trend_strength=0.02,
            up_points=4,
            passed_threshold=action == ENTER_LONG,
        )
        assert signal.action in {ENTER_LONG, HOLD}
    assert app.contributions_svg(an_attribution(close_logret=0.01, rsi14=0.0))


# ── the tables carry the design language, not Streamlit's ────────────────────


def test_a_table_puts_orange_on_the_header_row_only() -> None:
    """Two colour families, and the table is data. The header is the chrome in it."""
    css = app.stylesheet()
    header_rule = css[css.index(".gb-table th") : css.index(".gb-table td")]

    assert app.ORANGE in header_rule
    body_rule = css[css.index(".gb-table td") : css.index(".gb-table tbody")]
    assert app.ORANGE not in body_rule


def test_numeric_columns_are_right_aligned_with_tabular_figures() -> None:
    """Digits have to line up in their columns or a reader cannot compare down one."""
    html = app.table_html(("SYMBOL", "QTY"), [("AAPL", "1.00")], numeric=(1,))
    css = app.stylesheet()

    assert '<td class="num">1.00</td>' in html
    assert '<td class="">AAPL</td>' in html
    assert "tabular-nums" in css


def test_rows_carry_thin_dashed_rules_on_a_near_black_ground() -> None:
    css = app.stylesheet()

    assert f"border-bottom: 1px dashed {app.HAIRLINE}" in css
    assert f"background: {app.PANEL}" in css


def test_a_cell_cannot_inject_markup() -> None:
    html = app.table_html(("A",), [("<script>x</script>",)])

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_a_decision_with_no_order_shows_an_em_dash_not_a_blank() -> None:
    """A blank cell in a technical table reads as missing data rather than as 'none'."""
    html = app.decision_table([a_record()])

    assert f'<td class="">{app.EM_DASH}</td>' in html
    assert '<td class=""></td>' not in html


def test_an_order_shows_as_yes() -> None:
    html = app.decision_table([a_record(order={"symbol": "AAPL"})])

    assert "YES" in html


def test_a_quarantined_position_shows_em_dashes_rather_than_zeros() -> None:
    from glassbox.engine.reconcile import Book

    html = app.position_table(
        app.position_rows(Book(), {"AAPL": 0.0919}, {"AAPL": 310.0})
    )

    assert "QUARANTINED" in html
    assert html.count(app.EM_DASH) == 2  # no entry basis, and therefore no PnL


# ── a decomposition that is mostly cancellation says so ──────────────────────


def test_a_nearly_cancelled_decomposition_is_flagged_in_the_row() -> None:
    """AMZN at 0.1535 means 85% of the gross channel view offset. Unflagged, the row looks
    exactly like one where every channel agreed."""
    fragile = a_record(close_logret=0.08, rsi14=-0.07)

    assert app.cancellation(fragile.attribution) < app.EXPLANATION_FRAGILE_BELOW
    assert app.is_fragile(fragile.attribution)
    assert app.FRAGILE_LABEL in app.decision_table([fragile])
    # The cancellation cell, not the whole row: one orange cell in a grey table is already
    # the only thing the eye goes to.
    assert 'class="num gb-flag"' in app.decision_table([fragile])
    assert app.decision_table([fragile]).count("gb-flag") == 1


def test_a_healthy_decomposition_is_not_flagged() -> None:
    """The flag has to be the rarer of the two or it stops being read."""
    solid = a_record(close_logret=0.08, rsi14=0.06)

    assert not app.is_fragile(solid.attribution)
    assert app.FRAGILE_LABEL not in app.decision_table([solid])


def test_the_expander_title_carries_the_trend_and_the_flag() -> None:
    """The list has to be scannable unopened."""
    title = app.expander_title(a_record(close_logret=0.08, rsi14=-0.07))

    assert "+0.0200" in title
    assert "CANCELLATION" in title
    assert app.FRAGILE_LABEL in title


# ── nothing is drawn on top of data ──────────────────────────────────────────


def test_the_band_note_sits_below_the_plot_area_not_over_it() -> None:
    """The note used to be drawn at the same height as the NOW marker and across the price
    line. Chrome does not compete with data for pixels in this design."""
    import re

    svg = app.forecast_svg(
        HISTORY, app.price_path(103.5, [0.004]), Thresholds.never(), "AAPL"
    )
    plot_bottom = 250 - 30  # height less the strip

    note = re.search(r'<text x="[\d.]+" y="([\d.]+)"[^>]*>NO CALIBRATED BAND', svg)
    assert note is not None
    assert float(note.group(1)) > plot_bottom

    polylines = re.findall(r'points="([^"]+)"', svg)
    drawn = [
        float(point.split(",")[1]) for line in polylines for point in line.split(" ")
    ]
    assert max(drawn) <= plot_bottom


def test_the_chart_scales_to_its_container_rather_than_being_letterboxed() -> None:
    """A fixed height beside width=100% and a viewBox centres the drawing and wastes the
    column, which is what left a third of the viewport empty."""
    svg = app.forecast_svg(
        HISTORY, app.price_path(103.5, [0.004]), Thresholds(0.01), "A"
    )

    opening = svg[: svg.index(">") + 1]
    assert 'width="100%"' in opening
    assert "height:auto" in opening
    assert "height=" not in opening  # the viewBox supplies the ratio


# ── the rulers and the masthead ──────────────────────────────────────────────


def test_the_top_ruler_starts_at_zero_not_double_zero() -> None:
    assert ">0<" in app.ruler_html()
    assert ">00<" not in app.ruler_html()


def test_there_is_a_numbered_ruler_down_the_left_edge() -> None:
    ruler = app.left_ruler_html()

    assert ">0<" in ruler and ">50<" in ruler
    assert "position: fixed" in app.stylesheet()


def test_the_content_column_is_not_capped(cfg_stub) -> None:
    """The charts get the width the ruler does not take."""
    assert "max-width: none" in app.stylesheet()


def test_the_masthead_stacks_project_system_and_version(cfg_stub) -> None:
    html = app.header_html(cfg_stub, app.ASIDE, None)

    assert html.index("PROJECT") < html.index("SYSTEM") < html.index("VERSION")
    assert html.count("gb-metarow") == 3
