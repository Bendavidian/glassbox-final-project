"""GB-34/35/36 acceptance: the dashboard's arithmetic, its encodings and its rulings.

Layout is checked by looking at it. What is tested here is everything the eye cannot
check: that the threshold appears even when the band is `never()`, that a share on a bar
equals the share in the prose, that the cancellation is shown, that a quarantined position
reports no PnL it has no basis for, that the Hebrew narrative gets an explicit direction,
and that orange never leaks into a data mark.
"""

from __future__ import annotations

import ast
import csv
import html
import inspect
import json
import math
import re
import tokenize
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glassbox.config.loader import load_config
from glassbox.contracts.schemas import (
    Attribution,
    DecisionRecord,
    Forecast,
    Signal,
)
from glassbox.dashboard import app, tokens
from glassbox.engine.reconcile import Book, Holding
from glassbox.engine.signal import ENTER_LONG, HOLD, Thresholds
from glassbox.explain.channel import cancellation, shares
from tests.dashboard.conftest import a_verdict

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


# `status_of` and its two tests were deleted on 23 Sep 2026, not moved. It chose RUNNING
# or IDLE from the *position count* and never read the loop, so the tests that pinned that
# mapping were pinning the D2 defect. `tests/dashboard/test_liveness.py` tests what
# replaced it, including the band precedence the first of them was written for.


def test_a_managed_position_reports_its_pnl() -> None:
    book = Book(managed={"AAPL": a_holding(entry=100.0)})

    row = app.position_rows(book, {"AAPL": 2.0}, {"AAPL": 110.0}, {})[0]

    assert row.managed
    assert row.market_value == pytest.approx(220.0)
    assert row.unrealised == pytest.approx(20.0)
    assert row.unrealised_pct == pytest.approx(10.0)


def test_a_quarantined_position_is_shown_with_no_pnl() -> None:
    """The system did not open it and has no entry basis, so a number would be invented."""
    row = app.position_rows(Book(), {"NVDA": 0.01}, {"NVDA": 200.0}, {})[0]

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


# `cfg_stub` moved to tests/dashboard/conftest.py on 23 Sep 2026, when a second dashboard
# test module needed it. One definition, not two that drift.


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


def test_the_rtl_narrative_rule_moves_to_the_other_side() -> None:
    """A left border on a right-to-left paragraph sits at the end of the sentence."""
    css = app.stylesheet()

    assert '.gb-narrative[dir="rtl"]' in css
    assert "border-right" in css


def data_surfaces() -> dict[str, str]:
    """Every surface that encodes data with colour, rendered.

    The spectral panel is included with a FITS-shaped attribution rather than an empty
    one, because an empty panel is a string that trivially contains no colour and would
    pass this rule by rendering nothing.
    """
    return {
        "contributions": app.contributions_svg(
            an_attribution(close_logret=0.08, rsi14=-0.06)
        ),
        "forecast": app.forecast_svg(
            HISTORY, app.price_path(103.5, [0.01]), Thresholds.never(), "AAPL"
        ),
        "spectral": app.spectral_panel(a_spectral_attribution()),
        "sparkline": app.sparkline_svg(
            pd.Series([100.0, 99.0, 98.0, 97.0]),
            app.PositionRow(
                "AAPL",
                1.0,
                100.0,
                97.0,
                True,
                stop_loss=94.0,
                take_profit=112.0,
                live_stop=94.0,
                stop_working=True,
                orders_read=True,
            ),
            loop_present=True,
        ),
    }


def ramp_surfaces() -> dict[str, str]:
    """Every chart where the ramp **is** the encoding, rendered.

    One list, read by both directions of the rule: nothing else may use the ramp, and
    these may not use status colour. Two hand-written lists would drift the day a chart
    joined one of them, which is what happened on 23 Sep 2026 when D3 moved the two
    backtest panels here - a reader of one test would have seen a rule the other did not
    enforce.
    """
    reference = app.reference_rows(app.load_results())
    daily = app.load_daily_equity()
    return {
        "contributions": app.contributions_svg(
            an_attribution(close_logret=0.08, rsi14=-0.06)
        ),
        "spectral": app.spectral_panel(a_spectral_attribution()),
        # Joined on 23 Sep 2026. Both draw an arm against buy and hold, and green above a
        # zero line would answer "did it gain" on a panel asking "did it beat the
        # reference" - fold 13 gained 2.03% and lost to the market by seven points.
        "cumulative_equity_svg": app.cumulative_equity_svg(daily, "dlinear"),
        "fold_bars_svg": app.fold_bars_svg(reference, "dlinear"),
        # And the radar, whose polygon was hardcoded to GAIN fill and stroke: an arm
        # behind its reference on all four axes that have one still arrived green.
        "radar_svg": app.radar_svg(reference, "dlinear"),
    }


def test_status_colour_never_enters_a_ramp_chart() -> None:
    """**Narrowed in GB-63c, and narrowed rather than loosened.**

    The rule was once "status colour never enters a data-encoding chart", and under the
    card language that meant every chart. Under Vermillion Slate the ramp survives in a
    few surfaces and green-and-red is the primary language everywhere else - so a forecast
    dashed in GAIN is now correct rather than a violation, and a test still forbidding it
    would have to be deleted to let the design through.

    What survives is the part that was always the point: **the two families must not meet
    inside one chart.** Where the ramp encodes which band, status colour would make a
    reader ask what green means on an axis already spending colour on frequency.
    `test_the_ramp_appears_in_no_chart_that_encodes_no_data` holds the other direction.
    """
    for name, surface in ramp_surfaces().items():
        assert surface, f"{name} rendered nothing, so this asserted nothing about it"
        found = [c for c in app.STATUS_COLOURS if c.lower() in surface.lower()]
        assert not found, f"{name} encodes with the ramp and took status colour {found}"


def test_the_data_chart_rule_can_fail(monkeypatch) -> None:
    """Proof the guard bites, on a **real** chart, because a guard nobody has seen fail
    is a note.

    The status colour is pushed into the data ramp and the contributions chart is rendered
    again through its own code path. If the check above cannot reject that, it cannot
    reject anything - `STATUS_COLOURS` could be emptied, or the surfaces list could stop
    covering the charts, and every assertion would still pass in green.
    """
    monkeypatch.setattr(app, "RAMP", (app.GAIN,) * len(app.RAMP))
    (
        app.channel_colour.cache_clear()
        if hasattr(app.channel_colour, "cache_clear")
        else None
    )

    violating = app.contributions_svg(
        an_attribution(close_logret=0.08, rsi14=-0.06)
    ).lower()
    found = [c for c in app.STATUS_COLOURS if c.lower() in violating]

    assert found, "a chart drawn entirely in the gain colour passed the data-chart rule"


# ── GB-63: the live elements ─────────────────────────────────────────────────


def a_position(price: float, stop: float = 94.0, target: float = 112.0):
    """A position whose stop is **working at the broker**, stated rather than inherited:
    since 22 Sep 2026 the bands measure to the live order, not to the book's level."""
    return app.PositionRow(
        "AAPL",
        1.0,
        100.0,
        price,
        True,
        stop_loss=stop,
        take_profit=target,
        live_stop=stop,
        stop_working=True,
        orders_read=True,
    )


@pytest.mark.parametrize(
    ("price", "expected"),
    [
        (100.0, app.ORDINARY),  # at entry, all the room unspent
        (98.0, app.ORDINARY),  # 67% left
        (97.0, app.APPROACHING),  # exactly half - the boundary belongs to the warning
        (95.5, app.CLOSE),  # exactly a quarter - likewise
        (93.0, app.CLOSE),  # through the stop
    ],
)
def test_the_stop_bands_are_fractions_of_the_room_the_position_was_given(
    price: float, expected: str
) -> None:
    """A 2% move means something different against a 3% stop than against a 10% one, and
    a percentage of price cannot tell them apart. Both boundaries belong to the more
    serious band: a threshold that reads 'ordinary' exactly at half is one that has to be
    crossed before it warns."""
    assert a_position(price).stop_proximity == expected


def test_only_the_innermost_band_says_what_happens_next() -> None:
    """Emphasis says *look*; the innermost band has to say *why*, while there is still
    time to act on it."""
    assert app.stop_note(a_position(98.0)) == ""
    assert app.stop_note(a_position(97.0)) == ""

    near = app.stop_note(a_position(95.0))
    assert "STOP FILL IS NEAR" in near and "MARKET" in near
    through = app.stop_note(a_position(93.0))
    assert "AT OR THROUGH ITS STOP" in through


def test_a_quarantined_position_claims_no_stop_room() -> None:
    from glassbox.engine.reconcile import Book

    row = app.position_rows(Book(), {"AAPL": 1.0}, {"AAPL": 310.0}, {})[0]

    assert math.isnan(row.stop_room)
    assert row.stop_proximity == app.ORDINARY, "an unknown stop is not an alarm"


def test_the_cycle_countdown_is_measured_from_the_loops_own_write(tmp_path) -> None:
    """Not from the panel's refresh timer, which would tick smoothly past a dead loop.

    The write is the heartbeat file since 23 Sep 2026, not book.json's mtime: the book is
    persisted only inside a session, so ageing from it called a loop idling correctly
    overnight NOT RESPONDING. `tests/dashboard/test_liveness.py` holds that case and the
    rest of the sources; this holds the shape - no evidence is `inf`, a fresh write is
    nearly nothing, and a stale book on its own buys the loop no credit at all.
    """
    from glassbox import live_loop

    now = pd.Timestamp.now(tz="UTC")
    assert math.isinf(app.cycle_age(tmp_path, now))

    (tmp_path / "book.json").write_text("{}", encoding="utf-8")
    assert math.isinf(app.cycle_age(tmp_path, now)), "the book is not a heartbeat"

    live_loop.write_heartbeat(tmp_path, now, "in session")
    age = app.cycle_age(tmp_path, pd.Timestamp.now(tz="UTC"))

    assert 0.0 <= age < 30.0


def test_a_silent_loop_outranks_a_stale_broker_read() -> None:
    """Two different silences: old numbers, versus a system that is not running.

    Both now come from the verdict rather than from an age this function thresholds for
    itself, and the loop's silence takes the LOSS class the stylesheet had defined and
    never emitted.
    """
    live = a_verdict()
    stale_read = a_verdict(broker_age=12.0)
    silent = a_verdict(loop_age=2000.0)  # lock held, two heartbeats without a write

    assert silent.state == app.NOT_RESPONDING
    assert app.staleness_html(live) == ""
    assert "BROKER READ" in app.staleness_html(stale_read)
    assert "gb-stale" in app.staleness_html(stale_read)
    assert "LOOP NOT RESPONDING" in app.staleness_html(silent)
    assert "gb-not-responding" in app.staleness_html(silent)


def test_the_equity_curve_survives_a_half_written_line(tmp_path) -> None:
    """It is appended on every refresh and read on the next, so a torn line is a real
    possibility and losing the session's curve to it would be the wrong trade."""
    now = pd.Timestamp.now(tz="UTC")
    app.append_equity(tmp_path, now - pd.Timedelta(minutes=2), 100_000.0)
    with (tmp_path / app.EQUITY_FILE).open("a", encoding="utf-8") as stream:
        stream.write('{"at": "not a tim\n')
    app.append_equity(tmp_path, now, 100_050.0)

    curve = app.load_equity(tmp_path, now)

    assert len(curve) == 2
    assert float(curve.iloc[-1]) == 100_050.0


def test_a_nan_equity_is_never_written(tmp_path) -> None:
    """A failed broker read must not enter the curve as a point."""
    app.append_equity(tmp_path, pd.Timestamp.now(tz="UTC"), float("nan"))

    assert not (tmp_path / app.EQUITY_FILE).exists()


def test_the_equity_curve_is_coloured_by_its_own_sign() -> None:
    now = pd.Timestamp.now(tz="UTC")
    index = pd.date_range(now - pd.Timedelta(minutes=4), periods=5, freq="min")

    rising = app.equity_svg(pd.Series([100.0, 101, 102, 103, 104], index=index))
    falling = app.equity_svg(pd.Series([104.0, 103, 102, 101, 100], index=index))

    assert app.GAIN in rising and app.LOSS not in rising
    assert app.LOSS in falling and app.GAIN not in falling


def test_a_decision_is_marked_new_for_exactly_one_refresh() -> None:
    """A badge that persisted would stop meaning *this arrived while you were looking*."""

    a = a_record(day="2025-07-07", symbol="AAPL")
    b = a_record(day="2025-07-08", symbol="MSFT")

    first, seen = app.newly_written([a], set())
    assert first == set(), "the first render marks nothing; everything is new"

    second, seen = app.newly_written([a, b], seen)
    assert second == {(b.as_of, b.symbol)}

    third, _ = app.newly_written([a, b], seen)
    assert third == set(), "the mark survived into a second refresh"


def test_every_status_colour_is_redundant_with_a_sign_and_a_glyph() -> None:
    """**Rule two.** Colour never carries the fact alone, so the panel reads in greyscale.

    Asserted on `status_html`, which is the only producer of status colour in the module -
    the redundancy is a property of that function rather than a convention each call site
    remembers, which is what makes this one assertion sufficient.
    """
    for value, glyph in ((1.25, "▲"), (-1.25, "▼"), (0.0, "—")):
        rendered = app.status_html(value, f"{value:+.2f}")
        assert glyph in rendered, f"{value} rendered without its glyph"
        assert f"{value:+.2f}" in rendered, f"{value} rendered without its sign"


def test_a_quarantined_gain_gets_no_arrow_it_cannot_justify() -> None:
    """NaN is not a direction. An arrow on an unknown PnL asserts a sign nobody measured."""
    rendered = app.status_html(float("nan"), app.EM_DASH)

    assert "▲" not in rendered and "▼" not in rendered
    assert app.MUTED in rendered


def test_only_module_built_html_reaches_a_raw_column() -> None:
    """`table_html(raw=...)` skips escaping, so its blast radius is pinned here.

    Named in `table_html`'s own docstring, and written because a docstring that cites a
    test which does not exist is the defect this project keeps finding in other forms.
    Escaping is the default for every cell that came from a file, a broker or a record;
    `raw` exists so the status colour can reach one column at all. The guard is on the
    module's source: exactly one call site may pass it.
    """
    source = Path(app.__file__).read_text(encoding="utf-8")

    # Two call sites, both named here. The count is the guard: a third would be a caller
    # opting out of escaping without anyone deciding it should.
    assert source.count("raw=(") == 2, "a new caller is opting out of escaping"
    # STOP ROOM joined UNREALISED on 22 Sep 2026: `stop_cell` builds the dim INTENDED span
    # from constants and formatted floats, and nothing in it came from a file or a broker.
    assert "raw=(5, 6)" in source, "position_table's UNREALISED and STOP ROOM columns"
    assert "raw=(4,)" in source, "activity_table's VALUE column"


def _detail_cells(html: str) -> list[str]:
    """The DETAIL column of every rendered activity row, tags stripped."""
    return [
        re.sub(r"<[^>]+>", "", row.split("<td")[4].split(">", 1)[1]).strip()
        for row in html.split("<tr")[1:]
        if row.count("<td") > 4
    ]


def test_a_decision_that_carried_an_order_cannot_render_as_one_that_did_not() -> None:
    """**Order and no order rendered the same cell, and one of them was an exit.**

    On 3 September 2026 seven EXIT decisions were written against an empty book. Each drew
    a red `gb-row-loss` row reading EXIT beside `LIVE` - identical to a row where a
    position was closed - because DETAIL restated provenance, the one thing every live row
    shares. `decision_table` has carried the order as YES/em dash since GB-34; the table
    the console actually renders was dropping it.
    """
    without = _detail_cells(app.activity_table([a_record()], []))
    carried = _detail_cells(
        app.activity_table([a_record(order={"symbol": "AAPL", "shares": 1.0})], [])
    )

    assert without and carried, "the rows under test did not render"
    assert without != carried, (
        f"a decision with order=None renders the same DETAIL as one with an order: "
        f"{without} == {carried}"
    )


def test_the_detail_cell_says_where_the_row_came_from_as_well() -> None:
    """Order presence is added to provenance, not swapped for it - a replayed row must
    still say so on its own face, which is `provenance_label`'s whole reason."""
    replayed = app.decision_detail(
        DecisionRecord(
            **{
                **a_record().__dict__,
                "provenance": "replay:fold-13",
            }
        )
    )

    assert "REPLAY:FOLD-13" in replayed
    assert "ORDER" in replayed


def test_a_raw_column_still_escapes_every_other_cell() -> None:
    """The opt-out is per column, not per table."""
    html = app.table_html(
        ("A", "B"), [("<script>alert(1)</script>", "<b>ok</b>")], raw=(1,)
    )

    assert "&lt;script&gt;" in html, "a non-raw cell was not escaped"
    assert "<b>ok</b>" in html, "the raw cell was escaped"


def test_the_status_pair_is_the_one_recorded_in_decisions() -> None:
    """The palette is a ruling, so the constants are pinned to it rather than adjustable.

    A colour changed here and not in DECISIONS would put the report and the product in
    two different palettes, which is the two-places family in a place nobody greps.
    """
    assert (app.GAIN, app.LOSS) == ("#22C55E", "#EF4444")
    assert app.STATUS_COLOURS == (app.GAIN, app.LOSS)
    assert app.GAIN_FILL == "rgba(34,197,94,0.15)"
    assert app.LOSS_FILL == "rgba(239,68,68,0.15)"


# ── GB-53: the spectral panel ────────────────────────────────────────────────


def a_spectral_attribution(dead_period: float = 120.0) -> Attribution:
    """A FITS-shaped attribution: a frequency view, a gain/phase map, one dead bin."""
    view = {
        math.inf: 0.0040,
        24.0: 0.0031,
        12.0: -0.0022,
        8.0: 0.0015,
        dead_period: 0.0,
    }
    pairs = {
        24.0: (0.83, 1.96),
        12.0: (0.41, -0.55),
        8.0: (0.22, 0.12),
        dead_period: (0.0, 0.0),
    }
    return Attribution(
        per_channel={"close_logret": sum(view.values())},
        per_lag=None,
        per_frequency=view,
        gain_phase=pairs,
        forecast_total=sum(view.values()),
    )


def test_the_panel_is_absent_rather_than_empty_for_a_model_without_frequencies() -> (
    None
):
    """**The FITS gate, expressed structurally.** `per_frequency` is None for DLinear and
    persistence, so the panel does not render - an empty frame would read as a fault
    rather than as a property of the deployed model."""
    dlinear = an_attribution(close_logret=0.01, rsi14=-0.004)

    assert dlinear.per_frequency is None
    assert app.spectral_panel(dlinear) == ""
    assert app.spectral_panel(a_spectral_attribution()) != ""


def test_frequencies_are_keyed_by_period_in_days_not_by_bin_index() -> None:
    """ "The 17-day cycle" means something to a reader and "bin 7" does not."""
    svg = app.spectral_svg(a_spectral_attribution())

    assert "24.0-DAY" in svg
    assert "BIN 7" not in svg
    assert "bin" not in svg.lower().replace("bin 0", "")


def test_the_rin_mean_is_not_labelled_as_a_frequency() -> None:
    """It is keyed at infinity because it has to be keyed at something. Rendering that as
    an "inf-day cycle" would invent a cycle nobody measured."""
    assert app.period_label(math.inf) == app.RIN_MEAN_LABEL
    assert "RIN MEAN" in app.spectral_svg(a_spectral_attribution())
    assert "INF" not in app.spectral_svg(a_spectral_attribution()).upper().replace(
        "INFORMATION", ""
    )


def test_the_dead_row_is_shown_as_dead_and_says_why() -> None:
    """**Not omitted.** A reader who never sees it cannot know the architecture allocates
    a row that multiplies zero on every forward pass."""
    svg = app.spectral_svg(a_spectral_attribution())

    assert app.DEAD_BIN_LABEL in svg
    assert "MULTIPLIES ZERO" in svg
    assert "RIN REMOVES THE WINDOW MEAN" in svg
    assert "stroke-dasharray" in svg  # geometry, not colour, marks it


def test_the_response_chart_carries_the_measurement_not_only_the_curve() -> None:
    """**The caption is the point of the chart.** A curve read as "what the model learned
    about the market" is the black-box failure this project opposes, committed by the
    explanation layer - the worst place for it."""
    svg = app.response_svg([8.0, 12.0, 24.0, 120.0], [0.22, 0.41, 0.83, 0.0])

    assert "WHITE NOISE" in svg
    assert "86%" in svg
    assert "+0.9485" in svg
    assert "GB-48" in svg
    assert "48 MODELS" in svg


def test_the_fragility_flag_is_not_silent_when_no_cycle_carries_the_forecast() -> None:
    """Loud in the channel panel, so it must not be quiet here. Measured on real data: the
    largest share across 24 contributors was 0.211."""
    spread = Attribution(
        per_channel={"close_logret": 0.004},
        per_lag=None,
        per_frequency={float(period): 0.001 for period in range(5, 25)},
        gain_phase={float(period): (0.1, 0.0) for period in range(5, 25)},
        forecast_total=0.020,
    )

    assert "NO SINGLE CYCLE CARRIES THIS FORECAST" in app.spectral_svg(spread)
    assert "NO SINGLE CYCLE" not in app.spectral_svg(
        Attribution(
            per_channel={"close_logret": 0.01},
            per_lag=None,
            per_frequency={12.0: 0.01, 8.0: 0.0001},
            gain_phase={12.0: (1.0, 0.0), 8.0: (0.1, 0.0)},
            forecast_total=0.0101,
        )
    )


def test_phase_is_reported_in_days_and_names_the_direction() -> None:
    """Radians of an unnamed cycle are not a thing anyone can picture."""
    svg = app.gain_phase_svg(a_spectral_attribution())

    assert "LEADS" in svg
    assert "1.96 D" in svg
    assert "LAGS" in svg
    assert "RAD" not in svg.upper()


def test_the_ramp_runs_dark_for_long_and_light_for_short_like_the_channels() -> None:
    """Same ordering as `channel_colour`, so a reader who has learned one reads the other
    for free: the ramp encodes how far back a thing looks."""
    periods = [120.0, 24.0, 12.0, 8.0]

    longest = app.period_colour(120.0, periods)
    shortest = app.period_colour(8.0, periods)

    assert app.RAMP.index(longest) < app.RAMP.index(shortest)
    assert app.period_colour(math.inf, periods) == app.MUTED  # not on the scale


def test_the_spectral_marks_use_the_data_ramp_and_no_other_colour() -> None:
    """The ramp is the data encoding. A second colour family inside these marks would make
    the panel unreadable in greyscale, which is the rule the whole design rests on."""
    svg = app.spectral_panel(
        a_spectral_attribution(), response=([8.0, 12.0, 24.0], [0.22, 0.41, 0.83])
    )
    allowed = {
        *app.RAMP,
        app.ORANGE,
        app.ORANGE_DIM,
        app.PAPER,
        app.MUTED,
        app.INK,
        app.PANEL,
        app.HAIRLINE,
    }

    used = set(re.findall(r"#[0-9A-Fa-f]{6}", svg))
    assert used <= allowed, used - allowed


def test_frequency_shares_use_the_same_denominator_as_channel_shares() -> None:
    """Routed through `explain.channel.shares` rather than recomputed, so share-of-gross
    has one definition in this codebase."""
    view = app.frequency_shares(a_spectral_attribution())

    assert sum(abs(value) for value in view.values()) == pytest.approx(1.0)
    assert all(-1.0 <= value <= 1.0 for value in view.values())


def test_a_response_with_nothing_to_plot_says_so() -> None:
    svg = app.response_svg([12.0], [0.4])

    assert "NO RETAINED CYCLE TO PLOT" in svg


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


def test_numeric_columns_are_right_aligned_with_tabular_figures() -> None:
    """Digits have to line up in their columns or a reader cannot compare down one."""
    html = app.table_html(("SYMBOL", "QTY"), [("AAPL", "1.00")], numeric=(1,))
    css = app.stylesheet()

    assert '<td class="num">1.00</td>' in html
    assert '<td class="">AAPL</td>' in html
    assert "tabular-nums" in css


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
        app.position_rows(Book(), {"AAPL": 0.0919}, {"AAPL": 310.0}, {}), a_verdict()
    )

    assert "QUARANTINED" in html

    # Three columns, and the count is stated as the rule rather than as a number: the
    # system did not open this position, so it knows no entry basis, no PnL derived from
    # one, and no stop it set. Each of those is an em dash. GB-63 added the third when it
    # added STOP ROOM, and a bare `== 2` would have read as a regression rather than as
    # one more thing the system correctly declines to claim.
    unknowable = ("ENTRY", "UNREALISED", "STOP ROOM")
    assert html.count(app.EM_DASH) == len(unknowable)


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
    line. Chrome does not compete with data for pixels in this design.

    **The boundary is read off the chart, not restated here.** This test carried
    ``plot_bottom = 250 - 30`` - a copy of `forecast_svg`'s height and strip - and the
    typography pass moved both to make room for 12px annotations. The test failed, and it
    failed for a reason that had nothing to do with what it asserts: the chart was still
    correct and the copy was stale. The separator the chart already draws between the plot
    and the annotation strip *is* the boundary, so it is what the assertion uses.
    """
    import re

    svg = app.forecast_svg(
        HISTORY, app.price_path(103.5, [0.004]), Thresholds.never(), "AAPL"
    )
    # The plot/strip separator is the only full-width dashed hairline this chart
    # draws, so it is findable without knowing where the strip begins.
    separator = re.search(
        '<line x1="0.0" y1="([0-9.]+)"[^>]*stroke-dasharray="2 4"', svg
    )
    assert separator is not None, "the chart drew no plot/strip separator"
    boundary = float(separator.group(1))

    note = re.search(r'<text x="[\d.]+" y="([\d.]+)"[^>]*>NO CALIBRATED BAND', svg)
    assert note is not None
    assert float(note.group(1)) > boundary

    polylines = re.findall(r'points="([^"]+)"', svg)
    drawn = [
        float(point.split(",")[1]) for line in polylines for point in line.split(" ")
    ]
    assert max(drawn) <= boundary


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


# ── the band's selection context (ruled 20 Aug 2026) ────────────────────────


def test_the_band_carries_how_it_was_selected(tmp_path: Path) -> None:
    """A band on 8 trades and a band on 80 are not the same claim.

    ``Thresholds`` carries ``lower`` and ``upper`` and nothing about provenance, which is
    right for a frozen contract and not enough for a reader — so the masthead states the
    validation Sharpe, the trade count it rests on, and that it is a **grid maximum**.
    Under pure noise the maximum of fifteen candidates on eight trades is positive almost
    surely, and a reader who is not told that reads a selected statistic as an estimate.
    """
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps(
            {
                "stood_aside": False,
                "lower": 0.026,
                "upper": None,
                "val_sharpe": 0.4826,
                "val_trades": 8,
                "fold": 16,
            }
        ),
        encoding="utf-8",
    )

    context = app.band_context(path)

    assert context is not None
    assert not context.stood_aside
    assert "+0.483" in context.summary
    assert "8 TRADES" in context.summary
    assert "GRID MAXIMUM" in context.summary
    assert "FOLD 16" in context.summary


def test_a_stood_aside_band_says_why_rather_than_quoting_a_sharpe(
    tmp_path: Path,
) -> None:
    """Standing aside is a decision, not a missing number."""
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps({"stood_aside": True, "val_sharpe": -3.38, "val_trades": 7}),
        encoding="utf-8",
    )

    summary = app.band_context(path).summary

    assert "STOOD ASIDE" in summary
    assert "-3.38" not in summary  # the rejected candidate is not the deployed band


def test_no_band_artefact_is_not_a_band_context(tmp_path: Path) -> None:
    assert app.band_context(tmp_path / "absent.json") is None


# ── GB-63b: the console's rules ──────────────────────────────────────────────


def a_source() -> app.Source:
    return app.Source(app.BACKTEST, "folds 1-16")


def test_a_card_cannot_be_built_without_saying_where_its_numbers_came_from() -> None:
    """The failure this type exists to prevent is a console that looks live while showing
    backtest numbers - which would discredit the project's central claim far more
    effectively than any missing feature."""
    with pytest.raises(TypeError):
        app.Source(app.BACKTEST)  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="which data"):
        app.Source(app.LIVE, "   ")

    with pytest.raises(ValueError, match="unknown source"):
        app.Source("GUESSED", "somewhere")


def test_every_region_renders_its_source_pill() -> None:
    """`card()` is gone as of GB-63c region 7 - there are no cards. The pill survives it
    unchanged, which is the point: the source declaration outlived the container it was
    first attached to, because it was never a property of the box."""
    html = app.region("Return by fold", a_source(), "<svg/>")

    assert "gb-pill" in html
    assert "BACKTEST" in html
    assert "folds 1-16" in html, "a backtest region must state its fold range"


def test_a_backtest_pill_states_a_fold_range_and_not_a_bare_label() -> None:
    """'BACKTEST' alone lets a reader assume a period nobody stated."""
    rows = pd.DataFrame({"fold": [1, 2, 3, 4]})
    assert app.fold_range(rows) == "folds 1-4"

    gappy = pd.DataFrame({"fold": [1, 2, 9]})
    assert app.fold_range(gappy) == "3 folds"

    assert app.fold_range(pd.DataFrame()) == "no folds"


def test_a_card_with_too_little_data_says_how_little(tmp_path) -> None:
    """Never a silent backfill from the other source, and never an empty axis."""
    source = app.Source(app.LIVE, "3 sessions, 2 trades")
    body = app.too_little(source, "1 equity reading this session - the curve needs two")

    assert "needs two" in body
    assert "gb-empty" in body


def test_every_status_coloured_cell_carries_a_sign_or_arrow() -> None:
    """**The one rule kept from the old palette**, because it costs nothing and protects
    a reader who cannot separate the two hues. The greyscale gate is gone - on a console
    where P&L is green-and-red by design it could only have been weakened to pass."""
    for value in (1.25, -1.25, 0.0):
        rendered = app.status_html(value, f"{value:+.2f}")
        glyph = {1.25: "▲", -1.25: "▼", 0.0: "—"}[value]

        assert glyph in rendered, f"{value} rendered without its glyph"
        assert f"{value:+.2f}" in rendered, f"{value} rendered without its sign"


def test_the_ramp_stays_inside_the_charts_that_encode_data() -> None:
    """It encodes *which channel*, *which band*, and since 23 Sep 2026 *the arm against
    its reference* - a quantity, not a direction. In a chart that encodes none of those it
    would be a third colour family competing for the same eye.

    **The surfaces come from `ramp_surfaces`, not from a list written here.** They used to
    be written here, and the list named radar and fold-bars as outsiders; when D3 made the
    fold chart draw its reference in the ramp, this test was asserting the opposite of
    what the design now says. The other direction - that no *other* chart takes the ramp -
    is held by `test_the_ramp_appears_in_no_chart_that_encodes_no_data`, which enumerates
    every builder rather than naming a few.
    """
    for name, surface in ramp_surfaces().items():
        assert any(
            c in surface for c in app.RAMP
        ), f"{name} lost the ramp it encodes with"
        assert not [
            c for c in app.STATUS_COLOURS if c in surface
        ], f"status colour entered {name}, which encodes with the ramp"


def test_the_ramp_boundary_can_fail(monkeypatch) -> None:
    """Proof the guard bites, on a real chart rather than a synthetic string."""
    monkeypatch.setattr(app, "RAMP", (app.GAIN,) * len(app.RAMP))

    violating = app.contributions_svg(an_attribution(close_logret=0.08, rsi14=-0.06))

    assert [
        c for c in app.STATUS_COLOURS if c in violating
    ], "the guard cannot detect it"


def test_the_radar_has_no_composite_score() -> None:
    """The reference dashboard shows 'Edge Score 76.8'. There is no such quantity here,
    and inventing one in a system whose thesis is exact attribution would be the opposite
    of the point."""
    ref = app.reference_rows(app.load_results())
    if ref.empty:
        pytest.skip("no results.csv to read")

    axes = app.radar_axis_values(ref, "dlinear")

    assert len(axes) == len(app.RADAR_AXES)
    for point in axes:
        assert point.label, "every axis is labelled with its own name"
        assert 0.0 <= point.unit <= 1.0
        assert point.printed, "every axis prints its measured value in its own units"
    assert not hasattr(app, "edge_score")


def test_each_radar_axis_is_measured_against_its_own_reference() -> None:
    """These quantities are not commensurable, and pretending otherwise is how a composite
    gets invented. **Each names where its own reference comes from**: direction from a
    column of the arm's rows, three from the buy-and-hold arm, and two from nowhere -
    which they say on the chart rather than printing a bare absolute (D4, 23 Sep 2026).
    """
    named = {axis.label: axis for axis in app.RADAR_AXES}

    assert named["DIRECTION"].reference_column == "direction_reference"
    assert named["SHARPE"].reference_arm == app.BUY_AND_HOLD
    assert named["RETURN"].reference_arm == app.BUY_AND_HOLD
    assert named["DRAWDOWN"].reference_arm == app.BUY_AND_HOLD
    assert not named["CANCELLATION"].referenced
    assert not named["FLATNESS"].referenced
    assert all(axis.column for axis in app.RADAR_AXES), "every axis names a column"


def test_a_backtest_card_never_borrows_the_live_source(tmp_path) -> None:
    """A card may not mix sources. The equity area is BACKTEST, so an absent artefact
    renders the empty state rather than the live session curve."""
    empty = app.cumulative_equity_svg(pd.DataFrame(), "dlinear")

    assert empty == "", "an absent artefact must not fall back to another source"


def test_the_session_strip_names_the_bound_directory(cfg_stub) -> None:
    """The GATE 2 defect, as a test: on 27 Aug the panel was bound to `checkpoints/live`
    while the rehearsal wrote to `checkpoints/rehearsal`, so its queue rendered empty and
    correct and criterion 2 had to be satisfied through the API."""
    html = app.session_strip(
        cfg_stub,
        a_verdict(loop_age=42.0),
        "checkpoints/rehearsal",
        "rehearsal:gate2",
        1,
    )

    assert "checkpoints/rehearsal" in html
    assert "REHEARSAL:GATE2" in html
    assert "BOUND" in html


def test_the_reliability_card_states_the_gap_against_the_bar() -> None:
    """Not optional and not small. A console that shows P&L while hiding how often the
    decider is right is the black box this project exists to oppose."""
    record = app.Reliability(
        direction=0.4959,
        always_long=0.5564,
        folds=16,
        measured_on="2026-08-23",
        model="dlinear",
    )

    body = app.reliability_body(record, None)

    assert "0.4959" in body and "0.5564" in body
    assert "16 FOLDS" in body.upper()
    assert app.LOSS in body, "a negative gap is red"
    assert "▼" in body, "and carries its arrow"


def test_an_unmeasured_reliability_says_so_rather_than_showing_nothing() -> None:
    body = app.reliability_body(None, None)

    assert "NOT MEASURED" in body


def test_the_reference_condition_yields_one_row_per_fold_per_arm() -> None:
    """**The bug the first screenshot of the rebuild showed.**

    channels and target_in_loop are part of the condition, and filtering neither left one
    arm appearing four times - twice for the feature sets, twice for the execution models.
    The fold chart drew 64 bars labelled f1 f1 f2 f3 f4 f4 and the equity curve chained
    356 daily points over 178 distinct dates. Visible only because the duplicated fold
    labels were printed under the bars.
    """
    ref = app.reference_rows(app.load_results())
    if ref.empty:
        pytest.skip("no results.csv to read")

    for model in ("dlinear", "fits", "wits"):
        arm = ref[ref.model == model]
        if arm.empty:
            continue
        assert (
            len(arm) == arm.fold.nunique()
        ), f"{model} appears more than once per fold"

    assert (
        len(ref[ref.model == "buy_and_hold"]) > 0
    ), "the market reference carries no channel set and must survive the filter"


def test_a_daily_curve_has_one_point_per_date() -> None:
    """Same root cause, and the one that silently doubled the calendar's day count."""
    daily = app.load_daily_equity()
    if daily.empty:
        pytest.skip("report/daily_equity.csv not generated")

    arm = app.arm_rows(daily, "dlinear")

    assert len(arm) == arm.date.nunique()


# ── GB-63c: the starfield and the flat language ─────────────────────────────


def test_the_starfield_is_identical_on_every_render() -> None:
    """**This is the whole design, not a nicety.**

    Streamlit re-executes the script and rebuilds the DOM on every rerun, so the field is
    re-injected once a minute. A field seeded from the clock would reshuffle each time,
    which is an animation of something that has not changed - the one thing this console
    refuses to do. Determinism is what makes re-injection equivalent to injecting once.
    """
    assert app.starfield() == app.starfield()
    assert app.starfield(seed=1337) != app.starfield(
        seed=7
    ), "the seed is not reaching the field; determinism would be an accident"


def test_every_star_carries_a_negative_delay() -> None:
    """The subtler half. A CSS animation restarts when its node is replaced, so without a
    seeded negative delay each rerun would reset every star to the start of its drift -
    deterministic in position and random in phase, which is a reshuffle by another name.
    """
    field = app.starfield()

    assert field.count("<i ") == app.STAR_COUNT
    assert field.count("animation-delay:-") == app.STAR_COUNT


def test_the_starfield_is_hidden_from_assistive_technology() -> None:
    """It carries no information. A screen reader announcing forty-eight empty elements
    would be worse than the decoration is worth."""
    field = app.starfield()

    assert 'aria-hidden="true"' in field
    assert "pointer-events: none" in app.stylesheet()


def test_reduced_motion_stops_the_drift_and_keeps_the_field() -> None:
    """The field stays; the drift stops. Removing the points as well would take something
    away to answer a question that was asked about motion."""
    css = app.stylesheet()

    assert "@media (prefers-reduced-motion: reduce)" in css
    # Sliced to the end of the block rather than by a character count: the block opens
    # with a comment explaining why the field stays, and a fixed-width slice landed inside
    # it. A test that reads a fixed number of characters of CSS is a test that breaks when
    # somebody explains themselves.
    block = css.split("@media (prefers-reduced-motion: reduce)")[1]
    block = block[: block.index("}", block.index(".gb-stars"))]
    assert "animation: none" in block


def test_nothing_is_filled_and_nothing_is_rounded() -> None:
    """**No cards.** Every fill must be the ground itself or a `transparent` that removes
    one of Streamlit's, and every radius must be an explicit 0 - except the star dots,
    which are 1px circles and the one place a radius means something."""
    # This asserts the stylesheet's TEXT, not the page. On 2026-09-29 it passed while the
    # Approve and Reject fills it names had never rendered. Page-level rendering of those
    # two fills is asserted by tests/dashboard/test_approval_fills_on_the_page.py.
    css = app.stylesheet()

    # Approve and Reject are the two deliberate exceptions and the assertion names them,
    # rather than loosening to "some fills are allowed". They are the only place a click
    # moves money, and the fill is the warning - so the test's job is to prove they are
    # still the ONLY two, which a set comparison does and a count would not.
    fills = {fill.strip() for fill in re.findall(r"background:\s*([^;!]+)", css)}
    assert fills == {
        app.GROUND,
        "transparent",
        app.GAIN,
        app.LOSS,
    }, f"a fill crept back, or an exception was lost: {sorted(fills)}"
    assert css.count(f"background: {app.GAIN} !important") == 1, "one Approve"
    assert css.count(f"background: {app.LOSS} !important") == 1, "one Reject"

    radii = re.findall(r"border-radius:\s*([^;]+);", css)
    assert set(radii) <= {"0", "50%"}, f"a rounded container crept back: {radii}"
    assert radii.count("50%") == 1, "only the star dots are round"


def test_the_type_is_one_monospace_family_with_tabular_figures() -> None:
    """Tabular figures are a readability requirement in a dense monospace table: a reader
    compares magnitudes by scanning a column, and proportional digits break that."""
    css = app.stylesheet()

    assert "IBM Plex Mono" in css
    assert "font-variant-numeric: tabular-nums" in css
    assert "ui-monospace" in css, "the fallback stack must survive a font failure"


def test_the_narrative_keeps_its_own_stack_and_its_rtl_rule() -> None:
    """Monospace is for the English chrome and the numerals. A Hebrew sentence set in IBM
    Plex Mono would fall back anyway, and this rule was dropped once already in a rewrite
    and restored by a test."""
    css = app.stylesheet()

    assert '.gb-narrative[dir="rtl"]' in css
    assert "border-right: 2px solid" in css


def test_a_region_carries_a_rule_and_no_box() -> None:
    """The rule and the space above it are the whole separation. A region that needs to
    feel distinct gets more space, not a border on four sides."""
    html = app.region(
        "Return by fold", app.Source(app.BACKTEST, "folds 1-16"), "<svg/>"
    )

    assert "gb-region-rule" in html
    assert "gb-pill" in html and "folds 1-16" in html

    # The pill is exempt and deliberately so: it is a tag, and the source declaration
    # is the thing that keeps live and backtest apart. What must not appear is a
    # container - a fill, a radius, or a four-sided border on the region itself.
    # Asserted on the region's own chrome rather than on every character inside it,
    # which is what the first version of this test did, and it failed on the pill.
    chrome = html.split('<div class="gb-region-head"')[0]
    assert "gb-region-rule" in chrome, "the 2px accent rule is the separation"
    assert "background" not in chrome
    assert "border" not in chrome, (
        "the rule belongs in the stylesheet, not inline - an inline border here "
        "would be a second place the chrome vocabulary is defined"
    )


def test_the_reliability_region_states_whether_the_band_fires() -> None:
    """The record says how good the forecast was; the band says how selective the system
    is about acting on it. A reader who sees only one has half the picture."""
    record = app.Reliability(
        direction=0.4959,
        always_long=0.5564,
        folds=16,
        measured_on="2026-08-23",
        model="dlinear",
    )
    band = app.BandContext(stood_aside=False, val_sharpe=2.661, val_trades=17, fold=13)

    body = app.reliability_body(record, band)

    assert "gb-figure" in body, "the gap is the large figure"
    assert app.LOSS in body and chr(9660) in body, "a negative gap is red and arrowed"
    assert "BELOW THE ALWAYS-LONG BAR" in body
    assert "BAND FIRES" in body
    assert "16 FOLDS" in body and "2026-08-23" in body


def test_a_stood_aside_band_says_so_rather_than_showing_a_threshold() -> None:
    record = app.Reliability(
        direction=0.50,
        always_long=0.55,
        folds=16,
        measured_on="2026-08-23",
        model="dlinear",
    )
    aside = app.BandContext(stood_aside=True, val_sharpe=None, val_trades=None, fold=9)

    body = app.reliability_body(record, aside)

    assert "BAND STANDS ASIDE" in body


def test_the_status_strip_is_one_line_of_stated_facts(cfg_stub) -> None:
    html = app.session_strip(
        cfg_stub,
        a_verdict(loop_age=42.0),
        "checkpoints/rehearsal",
        "rehearsal:gate2",
        1,
    )

    for key in (
        "SESSION",
        "LAST CYCLE",
        "BOOK",
        "MODEL",
        "CHANNELS",
        "UNIVERSE",
        "BOUND",
        "SHOWING",
        "CONFIG",
    ):
        assert key in html, f"the strip dropped {key}"
    assert "checkpoints/rehearsal" in html
    assert "gb-strip" in html


def _equity(values: list[float]) -> str:
    index = pd.date_range("2026-08-29 16:30", periods=len(values), freq="min", tz="UTC")
    return app.equity_svg(pd.Series(values, index=index))


def _equity_height(svg: str) -> int:
    return int(re.search(r'viewBox="0 0 \d+ (\d+)"', svg).group(1))


@pytest.mark.parametrize(
    ("readings", "band"),
    [(2, app.EQUITY_THIN_BAND), (9, app.EQUITY_THIN_BAND)]
    + [(10, app.EQUITY_BAND), (11, app.EQUITY_BAND)],
    ids=["2 readings", "9 readings", "10 readings", "11 readings"],
)
def test_the_equity_panel_shrinks_below_ten_readings(readings: int, band: int) -> None:
    """**All four pinned, and the shrunken side is below ten.** Nine readings or fewer get
    the thin band; ten or more the full one. The drawable range starts at exactly 2 - the
    panel renders its empty state below that - so 2 is a case the panel is in, not an
    edge it approaches.

    The chrome does not shrink with it: `EQUITY_HEADER` and `EQUITY_FOOTER` are what carry
    the title, the opening-balance label and the reading count, and taking their space is
    how a label gets clipped in silence.
    """
    svg = _equity([100.0 + index for index in range(readings)])

    assert app.equity_band(readings) == band
    assert _equity_height(svg) == app.EQUITY_HEADER + band + app.EQUITY_FOOTER
    assert f"{readings} READINGS THIS SESSION" in svg, "the count survives the shrink"


def test_a_two_reading_panel_is_small_but_still_a_chart() -> None:
    """Small is not degenerate. At two readings the panel still draws its curve, its
    opening-balance rule and that rule's label, and they still clear each other - which
    the geometry guards hold for every chart, this one by name because it is the case the
    shrink was built for.
    """
    svg = _equity([100_000.0, 100_013.35])

    assert _equity_height(svg) == 112
    assert svg.count("<polyline") >= 1, "the curve is still drawn"
    assert "100,000" in svg, "the opening balance is still labelled"
    assert "2 READINGS THIS SESSION" in svg
    labels = list(_labels(svg))
    for first in range(len(labels)):
        for second in range(first + 1, len(labels)):
            _, _, ax0, ax1, atop, abottom = labels[first]
            _, _, bx0, bx1, btop, bbottom = labels[second]
            assert not (
                ax0 < bx1 and bx0 < ax1 and atop < bbottom and btop < abottom
            ), "two labels overlap in the thin panel"


def test_the_equity_curve_is_green_above_the_opening_and_red_below() -> None:
    """Colouring the whole line by its closing sign was the wrong reading of the data: a
    session that spends most of itself under water and closes a cent up is not a green
    session, and one line in one colour cannot say that."""
    assert app.GAIN in _equity([100.0, 101.0, 102.0])
    assert app.LOSS not in _equity([100.0, 101.0, 102.0])

    assert app.LOSS in _equity([100.0, 99.0, 98.0])
    assert app.GAIN not in _equity([100.0, 99.0, 98.0])


def test_the_opening_point_does_not_decide_the_side() -> None:
    """**The edge the first implementation got wrong.**

    The first point *is* the opening, so `values[0] >= opening` is always true and every
    session began with a green stub - including one that fell from the first tick. The
    opening is not on a side; the first move is.
    """
    falling = _equity([100.0, 99.0, 98.0])

    assert app.GAIN not in falling, "a falling session opened with a green segment"


def test_a_curve_that_crosses_the_opening_changes_colour_at_the_crossing() -> None:
    crossing = _equity([100.0, 102.0, 98.0, 103.0])

    assert crossing.count("<polyline") == 3, "one run per side of the opening"
    assert app.GAIN in crossing and app.LOSS in crossing


def test_a_flat_session_still_renders_and_is_centred() -> None:
    """Flat is the common case: the loop stands aside on most bars."""
    flat = _equity([100.0, 100.0, 100.0])

    assert "READINGS THIS SESSION" in flat
    assert "<polyline" in flat


def _runs(svg: str) -> list[tuple[str, int]]:
    """``(stroke, point count)`` for each run of the session curve, in drawing order."""
    return [
        (stroke, len(points.split()))
        for points, stroke in re.findall(
            r'<polyline points="([^"]+)" fill="none" stroke="([^"]+)"', svg
        )
    ]


def test_a_run_at_the_opening_is_neither_a_gain_nor_a_loss() -> None:
    """**The live session that found this had made no move at all for 64 of 69 readings.**

    The side used to be seeded from the first value that *differed* from the opening, which
    reads forward through the whole series and applies the answer backwards. One rule, two
    symptoms, and the session passed through both in one afternoon: while the book was
    empty the fallback made `first_move` the opening itself, so `GAIN if opening >= opening`
    painted a motionless session **green at +0.00**; the moment WMT filled at 19:40:40Z and
    the account went $13 down, the same line reached forward to that drop and repainted all
    64 unchanged readings **loss red**. The calendar, the fold bars and the cumulative area
    have all had a third state since they were written; this was the one chart without one.

    **The series here is written out, and that is the mechanism rather than the fixture
    style.** The premise of this test - *a run of readings sits exactly on the opening* -
    was a property of the live account until 19:41:40Z on 3 Sep 2026 and has not been true
    since: the curve now holds 14 distinct values and falls correctly in red. A guard that
    had read `checkpoints/live` would have gone on passing while asserting nothing, and
    anyone verifying by eye today would watch a correct chart and conclude the defect was
    fixed. **This is the premise-in-a-config-default defect wearing production data**: the
    default that moved was not a config key but the state of the market, which no commit
    records and no review can pin. Every equity fixture in this file is a literal list for
    that reason, and none of them reads the loop's state directory.
    """
    svg = _equity([100.0] * 5 + [99.0, 98.0])

    runs = _runs(svg)

    assert [stroke for stroke, _ in runs] == [app.MUTED, app.LOSS]
    assert runs[0][1] == 5, "the unchanged readings are one flat run, not a red one"
    assert app.GAIN not in svg


def test_a_session_that_only_ever_loses_is_flat_then_red_and_not_red_throughout() -> (
    None
):
    """The live shape, end to end: the reading that moved is the only one coloured."""
    svg = _equity([100_000.18] * 64 + [99_994.20, 99_986.83])

    assert [stroke for stroke, _ in _runs(svg)] == [app.MUTED, app.LOSS]


def test_the_curve_floor_is_the_risk_policys_worst_single_position() -> None:
    """**One fact in two places, and this is what makes them equal.**

    `EQUITY_MIN_SPAN` is the largest loss the risk policy permits from a single position -
    `max_position_pct` of equity stopped out at `stop_loss_pct` - so a move that fills the
    chart is a move the size of the worst thing one position can do. Deriving it would mean
    a pure chart builder reading `settings.yaml`; pinning it is the alternative this
    project's rules allow, and without the pin the two numbers drift the first time the
    policy changes and nothing anywhere notices.
    """
    risk = load_config().risk

    assert app.EQUITY_MIN_SPAN == pytest.approx(
        risk.max_position_pct * risk.stop_loss_pct
    )


def test_a_near_flat_session_is_not_drawn_as_a_cliff() -> None:
    """**Nothing left the box; the chart drew a cliff out of nothing happening.**

    A curve pinned to its own min and max always fills the plot, whatever the move was
    worth. The live session moved $13.35 on $100,000 - 1.3 basis points - and spent every
    unit of plot height on it: 64 unchanged readings welded to the ceiling and the last
    three on the floor. That is a lie about magnitude, and it is the more dangerous of the
    two failures because it looks like data rather than like a defect.
    """
    values = [100_000.18] * 64 + [99_986.83]
    top, bottom = app.EQUITY_HEADER, 180 - app.EQUITY_FOOTER

    y, _ = app._vertical(values, top, bottom, app.EQUITY_MIN_SPAN)
    travel = abs(y(min(values)) - y(max(values)))

    assert travel < (bottom - top) * 0.1, "13 basis points filled a tenth of the plot"
    # And the floor is a floor, not a ceiling: a move worth the whole policy still fills it.
    whole = [100_000.0, 100_000.0 * (1 - app.EQUITY_MIN_SPAN)]
    wide, _ = app._vertical(whole, top, bottom, app.EQUITY_MIN_SPAN)

    assert abs(wide(whole[1]) - wide(whole[0])) == pytest.approx(bottom - top, abs=0.5)


def test_the_curve_is_clamped_to_its_plot_rather_than_trusted_to_land_in_it() -> None:
    """Mapping min to the floor and max to the ceiling contains the path only while the
    extremes come from the same values being drawn - which stops being true the moment the
    floor widens the range, and was never true of a value that is not a number."""
    top, bottom = 32.0, 148.0

    y, _ = app._vertical([100.0, 101.0], top, bottom, app.EQUITY_MIN_SPAN)

    assert y(1e9) == top and y(-1e9) == bottom
    assert y(float("inf")) == top and y(float("-inf")) == bottom
    assert top <= y(float("nan")) <= bottom


def test_the_countdown_is_a_depleting_accent_rule() -> None:
    """One pixel, vermillion, and it moves because time passed - not because a timer on
    this page is running. The seconds come from the loop's own last write."""
    full = app.countdown_svg(60, 60)
    spent = app.countdown_svg(0, 60)

    assert app.ACCENT in full and 'stroke-width="1"' in full
    assert "NEXT CYCLE" in full
    assert full != spent, "the rule must shorten as the seconds run down"


# ── GB-63c region 4: the four regions where the pill defect lived ───────────


def a_results_frame(
    folds: int = 16, model: str = "dlinear", **override
) -> pd.DataFrame:
    """A reference-condition results frame: one row per fold, one arm, one feature set."""
    row = {
        "model": model,
        "channels": "C0_base",
        "direction": 0.51,
        "direction_reference": 0.55,
        "sharpe": -0.31,
        "total_return": -0.012,
        "max_drawdown": 0.08,
        "cancellation": 0.61,
        "flatness": 0.42,
    }
    row.update(override)
    return pd.DataFrame([{**row, "fold": fold} for fold in range(1, folds + 1)])


def a_daily_frame(
    folds: int = 3,
    days_per_fold: int = 60,
    model: str = "dlinear",
    drift: float = 0.001,
) -> pd.DataFrame:
    """Dated daily equity for one arm: consecutive business days, folds end to end.

    `drift=0.0` gives an exactly flat curve, which is what an arm that stands aside in
    every fold actually produces - persistence does, in all sixteen.
    """
    dates = pd.bdate_range("2022-07-04", periods=folds * days_per_fold)
    rows, step = [], 0
    for fold in range(1, folds + 1):
        equity = 100000.0
        for index in range(days_per_fold):
            equity *= 1.0 + (drift if index % 2 else -drift)
            rows.append(
                {
                    "model": model,
                    "channels": "C0_base",
                    "fold": fold,
                    "date": dates[step],
                    "equity": equity,
                }
            )
            step += 1
    return pd.DataFrame(rows)


def test_the_radar_has_six_measured_axes_and_no_composite() -> None:
    """The reference dashboards show one number under the shape. This project has no such
    quantity, and inventing one in a system whose thesis is exact attribution would be the
    opposite of the point."""
    ref = app.reference_rows(app.load_results())
    if ref.empty:
        pytest.skip("no results.csv to read")

    axes = app.radar_axis_values(ref, "dlinear")

    assert len(axes) == 6
    for point in axes:
        assert point.label and point.printed, "every axis is named and prints its value"
        assert 0.0 <= point.unit <= 1.0
    assert not hasattr(app, "edge_score")


def test_flatness_is_an_axis_where_being_right_sized_is_best() -> None:
    """**Rewritten on 23 Sep 2026, because it pinned the wrong direction.**

    It asserted `higher is False` - lower flatness is better - and flatness is
    mean|forecast| / mean|actual|, where 1.0 is right-sized and 0.0 is a model that
    forecasts nothing at all. Under "lower is better" persistence, which forecasts
    exactly nothing, drew the best flatness on the page. The axis is two-sided: its best
    value is 1.0 and either side of it is worse.

    It earns its place for the reason 7.3 bans MAE from a table without it: across these
    arms MAE is close to a monotone function of flatness, so a shape showing error without
    flatness would be showing one axis twice and calling one of them accuracy.
    """
    named = {axis.label: axis for axis in app.RADAR_AXES}

    assert named["FLATNESS"].best == 1.0
    assert not named["FLATNESS"].referenced, "buy and hold makes no forecast to compare"


def _folds_in(path: str, model: str) -> list[int]:
    """The folds one arm holds in a CSV, read with the stdlib rather than through the
    module under test.

    An independent parse on purpose: comparing `fold_range(load_results())` against
    `fold_range(load_results())` would pass with both sides wrong in the same direction,
    which is the shape the defect had in the first place.
    """
    condition = {
        "anchor": "0",
        "lr": "0.001",
        "control": "real",
        "target_in_loop": "True",
        "cutoff_period_days": "5",
    }
    folds = set()
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["model"] != model or row.get("skipped") == "True":
                continue
            if row.get("channels") not in ("", "C0_base"):
                continue
            if any(row[key] != value for key, value in condition.items() if key in row):
                continue
            folds.add(int(row["fold"]))
    return sorted(folds)


def test_each_region_states_the_range_of_the_file_it_actually_read() -> None:
    """**The pill defect, verified against the files rather than against the source text.**

    `results.csv` and `report/daily_equity.csv` are two artefacts that can be any number of
    folds apart, and today they are: sixteen against three. A partial daily file had all
    four of these regions announcing "folds 1-16", the overclaim the pills exist to prevent
    committed by the pills. Each range is checked against the CSV that region read, parsed
    here by an independent path - not against the config, and not against a sibling region.
    """
    if not (Path(app.RESULTS_PATH).is_file() and Path(app.DAILY_EQUITY_PATH).is_file()):
        pytest.skip("both artefacts are needed to compare their ranges")

    reference, daily = app.reference_rows(app.load_results()), app.load_daily_equity()
    cases = (
        (app.shape_region, reference, app.RESULTS_PATH),
        (app.fold_region, reference, app.RESULTS_PATH),
        (app.cumulative_region, daily, app.DAILY_EQUITY_PATH),
        (app.calendar_region, daily, app.DAILY_EQUITY_PATH),
    )
    for build, frame, path in cases:
        folds = _folds_in(path, "dlinear")
        assert folds == list(
            range(1, len(folds) + 1)
        ), f"{path} is no longer contiguous; this fixture assumes it is"
        expected = f"{Path(path).name} folds 1-{folds[-1]}"

        rendered = build(frame, "dlinear")

        assert expected in rendered, f"{build.__name__} does not state its own range"


def test_a_region_cannot_borrow_a_siblings_range() -> None:
    """The same defect on frames chosen to disagree, so the assertion has something to
    catch on the day the two artefacts happen to hold the same folds."""
    reference, daily = a_results_frame(folds=16), a_daily_frame(folds=3)

    from_results = (
        app.shape_region(reference, "dlinear"),
        app.fold_region(reference, "dlinear"),
    )
    from_daily = (
        app.cumulative_region(daily, "dlinear"),
        app.calendar_region(daily, "dlinear"),
    )

    for rendered in from_results:
        assert "results.csv folds 1-16" in rendered
        assert "folds 1-3" not in rendered, "a results region took the daily range"
    for rendered in from_daily:
        assert "daily_equity.csv folds 1-3" in rendered
        assert "folds 1-16" not in rendered, "a daily region took the results range"


def test_a_region_names_the_file_and_not_only_the_kind() -> None:
    """Side by side, "folds 1-16" and "folds 1-3" read as a bug. BACKTEST alone lets a
    reader take two regions as disagreeing about the same data rather than agreeing about
    different data, which is the half of the defect a correct range does not fix."""
    rendered = app.calendar_region(a_daily_frame(folds=3), "dlinear")

    assert "daily_equity.csv" in rendered
    assert "BACKTEST" in rendered


def test_a_region_has_one_frame_in_scope() -> None:
    """**The mechanism, rather than the convention it replaced.**

    Naming the locals `from_results` and `from_daily` in `main` asked a reader to pick the
    right one. This asks nothing of anybody: the pill is built from the rows the chart
    draws, inside the function that draws them, and no second frame is present to pick
    wrongly from. The signature is what holds that, so the signature is what is pinned - a
    region that grows a second frame fails here rather than in a screenshot.
    """
    regions = (
        app.shape_region,
        app.cumulative_region,
        app.fold_region,
        app.calendar_region,
    )
    for build in regions:
        parameters = list(inspect.signature(build).parameters)

        assert len(parameters) == 2, f"{build.__name__} has a second frame in scope"
        assert parameters[0] in ("reference", "daily")
        assert parameters[1] == "model"


def test_one_selection_rule_serves_both_the_chart_and_its_pill() -> None:
    """A label disagreeing with its own picture would need `arm_rows` to disagree with
    itself. The filter is not optional: both artefacts carry each arm twice, once per
    channel set, and filtering on `model` alone counted every date twice."""
    daily = a_daily_frame(folds=2)
    doubled = pd.concat([daily, daily.assign(channels="C1_wide")], ignore_index=True)

    arm = app.arm_rows(doubled, "dlinear")

    assert len(arm) == len(daily)
    assert app.fold_range(arm) == "folds 1-2"


# Four calendar tests were deleted with the chart on 23 Sep 2026 (D5): the day count
# against the tiles drawn, the no-silent-truncation scale, the derived height, and the
# flat-day half of `test_a_flat_result_is_never_drawn_as_a_gain`. Each pinned a property
# of a heatmap that no longer exists - 922 days at 2px a tile - and the counts it carried
# are now the panel. `tests/dashboard/test_references.py` holds what replaced them,
# including that a day which did not move is still counted.


def test_a_flat_result_is_never_drawn_as_a_gain() -> None:
    """**Flat is the common case here, not the edge case.** This loop stands aside on most
    bars: persistence stands aside in all sixteen folds, and 143 of FITS's 175 daily
    changes are exactly zero. `>= 0` painted every one of them green, so the panel read as
    a mostly-green year for an arm that did nothing."""
    flat = a_daily_frame(folds=2, drift=0.0)

    area = app.cumulative_equity_svg(flat, "dlinear")

    assert app.GAIN not in area and app.GAIN_FILL not in area


def test_a_flat_cumulative_curve_is_centred_not_floored() -> None:
    """The same scaling rule as the session curve, and it had been fixed in only one of
    the two places it was written. An arm aside in every fold chains to an exactly flat
    curve, and `max(high - low, 1e-9)` drove every point of it to the floor."""
    top, bottom = 16.0, 270.0

    y, flat = app._vertical([1.0, 1.0, 1.0], top, bottom, app.EQUITY_MIN_SPAN)

    assert flat
    assert y(1.0) == (top + bottom) / 2

    moving, not_flat = app._vertical([1.0, 2.0], top, bottom, app.EQUITY_MIN_SPAN)

    assert not not_flat
    assert moving(2.0) == top and moving(1.0) == bottom


def test_a_fold_that_stood_aside_is_marked_rather_than_absent() -> None:
    """A zero bar has no height, so without a mark of its own it is indistinguishable from
    a fold that was never measured - and it is not a rare case."""
    rows = a_results_frame(folds=4, total_return=0.0)

    svg = app.fold_bars_svg(rows, "dlinear")

    assert "4 OF 4 FOLDS FLAT" in svg
    assert svg.count("<line") == 5, "the zero rule, plus a mark for each flat fold"
    assert app.GAIN not in svg, "a fold that stood aside was coloured as a gain"


# ── GB-63c region 5: the row language ───────────────────────────────────────


def a_decision(action: str, provenance: str = "live", symbol: str = "NVDA"):
    """A DecisionRecord thin enough for the activity table, with the fields it reads."""
    return DecisionRecord(
        as_of=pd.Timestamp("2026-09-01"),
        symbol=symbol,
        forecast=Forecast(
            path=np.array([0.01], dtype="float32"),
            symbol=symbol,
            as_of=pd.Timestamp("2026-09-01"),
        ),
        attribution=an_attribution(close_logret=0.01),
        signal=Signal(
            symbol=symbol,
            action=action,
            trend_strength=0.01,
            up_points=3,
            passed_threshold=True,
        ),
        order=None,
        narrative="",
        config_hash="c" * 64,
        provenance=provenance,
    )


def test_action_is_recoverable_from_the_row_text() -> None:
    """**The rule the whole row language rests on.**

    The 2px left border is an ACCENT on a fact, never the fact itself. A reader who cannot
    resolve the border - colour-blind, a greyscale print, a projector - must still be able
    to say what the system did, and the WHAT column is where they read it.
    """
    html = app.activity_table(
        [a_decision(app.ENTER_LONG), a_decision(app.EXIT), a_decision(app.HOLD)], []
    )

    for action in ("ENTER_LONG", "EXIT", "HOLD"):
        assert action in html, f"{action} is not recoverable without the border"


def test_the_border_states_what_the_system_did() -> None:
    """The second channel: two independent statements in one row."""
    html = app.activity_table([a_decision(app.ENTER_LONG), a_decision(app.EXIT)], [])

    assert "gb-row-gain" in html
    assert "gb-row-loss" in html


def test_a_hold_row_is_dimmed_and_ruled_rather_than_coloured() -> None:
    """Standing aside is not a gain or a loss, and colouring it either would be a claim
    the system did not make."""
    html = app.activity_table([a_decision(app.HOLD)], [])

    assert "gb-row-hold" in html
    assert "gb-row-gain" not in html and "gb-row-loss" not in html


def test_a_rehearsal_row_is_marked_as_not_live() -> None:
    """Vermillion is not an action. It marks a fact ABOUT the row - this did not come from
    the deployed band - which is exactly the distinction the source pills exist for, at
    row granularity."""
    html = app.activity_table(
        [a_decision(app.ENTER_LONG, provenance="rehearsal:gate2-execution-path")], []
    )

    assert "gb-row-accent" in html
    assert "gb-row-gain" not in html, "provenance outranks the action"


def test_a_freshly_written_row_takes_the_accent_for_one_refresh() -> None:
    """Freshness outranks provenance, which outranks the action: is this new, is this
    real, what was it. On the next refresh it settles to its durable mark."""
    record = a_decision(app.HOLD)
    key = {(record.as_of, record.symbol)}

    marked = app.activity_table([record], [], fresh=key)
    settled = app.activity_table([record], [], fresh=frozenset())

    assert "gb-row-accent" in marked
    assert "gb-row-accent" not in settled
    assert "gb-row-hold" in settled


def test_the_action_keys_come_from_the_engine_not_from_a_second_list() -> None:
    """A dashboard that spelled its own 'enter_long' would be a second copy of the signal
    vocabulary, and the two would diverge the first time the engine renamed one."""
    assert set(app.ROW_CLASSES) == {app.ENTER_LONG, app.EXIT, app.HOLD}


def test_approve_and_reject_are_the_only_filled_elements() -> None:
    """**The one place a click moves money, and the only place with a fill.**

    Every other surface on the page is type and rules on the ground. The weight of these
    two is the warning, and ground-coloured text on a solid field is what makes them read
    as controls rather than as a status somebody is being shown.
    """
    # This asserts the stylesheet's TEXT, not the page. On 2026-09-29 it passed while
    # neither fill had ever rendered: the selector matched nothing. Page-level rendering is
    # asserted by tests/dashboard/test_approval_fills_on_the_page.py.
    css = app.stylesheet()

    assert f"background: {app.GAIN} !important" in css
    assert f"background: {app.LOSS} !important" in css
    assert css.count(f"color: {app.GROUND} !important") >= 2


def test_both_answers_are_wired_and_neither_reuses_a_widget_key() -> None:
    """Streamlit registers a button by key, so two buttons sharing one would collide - and
    the first version of this panel left the old REJECT branch behind alongside the new
    one, which is exactly that collision. Both answers reach `answer_pending`, one True
    and one False, because a rejection that left no trace would make the log a record of
    what the system wanted rather than of what happened.
    """
    source = Path(app.__file__).read_text(encoding="utf-8")

    # Not redundant with guard 1b, and neither is to be deleted as a copy of the other.
    # This counts raw source and 1b walks the AST; on 2026-09-29 break 1 of guard 7 showed
    # the two can disagree, because a reroute through a local wrapper left these strings
    # intact. Since the key_prefix refactor (GB-67, 2026-09-29) this asserts that each
    # control and each answer is WRITTEN ONCE in the source - it no longer asserts that
    # runtime keys are unique, because one `{key_prefix}` string can produce two keys or
    # the same one twice. Runtime key uniqueness is guard 1b, assertion 4.
    assert source.count("key=f\"{key_prefix}a-{entry['decision_id']}\"") == 1
    assert source.count("key=f\"{key_prefix}r-{entry['decision_id']}\"") == 1
    assert source.count('entry["decision_id"], True') == 1
    assert source.count('entry["decision_id"], False') == 1


def _bindings(tree: ast.Module, module: str, name: str) -> set[str]:
    """Every local name ``module.name`` is reachable by: the import, any ``as`` alias, and
    any plain assignment of one of those names to another."""
    names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == module
        for alias in node.names
        if alias.name == name
    }
    grew = True
    while grew:
        grew = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Name)
                and node.value.id in names
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in names:
                        names.add(target.id)
                        grew = True
    return names


def _calls_to(tree: ast.Module, names: set[str], attribute: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id in names)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == attribute)
        )
    ]


def test_the_approval_path_is_pinned_at_every_call_site() -> None:
    """**GB-67, guard 7: the entry approval path, as it stood on 29 Sep 2026.**

    It is the only approval path that exists - the Co-Pilot exit path never queues - and
    the one the AMZN approval of 28 Sep 2026 took. Every call to `answer_pending` in the
    module, by any binding, must pass ``cfg``, an ``AlpacaBroker()`` built at the call
    site, ``root``, the entry subscripted by ``"decision_id"``, and a literal answer, and
    the literals across the module must be exactly ``{True, False}``.

    **Module-scoped, not scoped to ``_copilot_panel``,** so that moving the controls into a
    helper leaves it unchanged, and routing them through a wrapper or a rename fails it
    rather than removing the only call it could see.

    **The reject side builds an ``AlpacaBroker`` it never uses** - ``decline`` touches no
    broker, but the constructor still demands credentials. That is behaviour, pinned as
    it is, so a change to it is noticed rather than tidied in passing.

    It overlaps `test_both_answers_are_wired_and_neither_reuses_a_widget_key` on one
    property, that both answers are wired, and asserts it structurally rather than by
    counting text. **It does not count call sites**: that is a different property, and a
    second copy of the count would be free to disagree with the first.
    """
    tree = ast.parse(Path(app.__file__).read_text(encoding="utf-8"))
    brokers = _bindings(tree, "glassbox.engine.executor", "AlpacaBroker")
    calls = _calls_to(
        tree, _bindings(tree, "glassbox.live_loop", "answer_pending"), "answer_pending"
    )

    assert calls, "no call to answer_pending: the approval path was rerouted or renamed"

    answers = set()
    for call in calls:
        where = f"line {call.lineno}: {ast.unparse(call)}"
        assert (
            not call.keywords and len(call.args) == 5
        ), f"{where} - not five positionals"
        cfg, broker, root, decision, answer = call.args
        assert isinstance(cfg, ast.Name) and cfg.id == "cfg", f"{where} - not cfg"
        assert (
            isinstance(broker, ast.Call)
            and not broker.args
            and not broker.keywords
            and (
                (isinstance(broker.func, ast.Name) and broker.func.id in brokers)
                or (
                    isinstance(broker.func, ast.Attribute)
                    and broker.func.attr == "AlpacaBroker"
                )
            )
        ), f"{where} - the broker is not an AlpacaBroker() built at the call site"
        assert isinstance(root, ast.Name) and root.id == "root", f"{where} - not root"
        assert (
            isinstance(decision, ast.Subscript)
            and isinstance(decision.slice, ast.Constant)
            and decision.slice.value == "decision_id"
        ), f"{where} - not a subscript keyed by the constant 'decision_id'"
        assert isinstance(answer, ast.Constant) and isinstance(
            answer.value, bool
        ), f"{where} - the answer is not a literal True or False"
        answers.add(answer.value)

    assert answers == {True, False}, f"answers wired: {answers}; both must be"


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(found) == 1, f"{name} is defined {len(found)} times"
    return found[0]


def _keyed_by_a_decision(call: ast.Call) -> ast.JoinedStr | None:
    """The ``key`` of a widget keyed the way Approve and Reject are - by a decision id -
    or ``None``."""
    for keyword in call.keywords:
        if (
            keyword.arg == "key"
            and isinstance(keyword.value, ast.JoinedStr)
            and any(
                isinstance(part, ast.FormattedValue)
                and isinstance(part.value, ast.Subscript)
                and isinstance(part.value.slice, ast.Constant)
                and part.value.slice.value == "decision_id"
                for part in keyword.value.values
            )
        ):
            return keyword.value
    return None


def _approval_controls_in(tree: ast.Module, answers: list[ast.Call]) -> list[ast.Call]:
    """Every approval control in the module, by what it is rather than what it says: a
    widget keyed by a decision id, or one whose answer gates a call to the approval path.
    A caption is not consulted, so a button relabelled CONFIRM is still found."""
    wired = {id(call) for call in answers}
    found = {
        id(node): node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _keyed_by_a_decision(node) is not None
    }
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        assigned = {
            target.id: node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for branch in ast.walk(function):
            if not isinstance(branch, ast.If) or not any(
                id(node) in wired for part in branch.body for node in ast.walk(part)
            ):
                continue
            test = branch.test
            control = assigned.get(test.id) if isinstance(test, ast.Name) else test
            if isinstance(control, ast.Call):
                found[id(control)] = control
    return list(found.values())


def test_every_approval_control_is_rendered_by_the_one_helper() -> None:
    """**GB-67, guard 1b: one place wires an approval, and every surface goes through it.**

    Guard 7 pins the *shape* of a call to `answer_pending`; this pins *where* the calls
    are and *how many*. Exactly one approve and one reject in the module, both inside
    `_approval_controls`; every approval control inside it too, found structurally - keyed
    by a decision id, or wired to the approval path - because a guard that looked for the
    word APPROVE would pass a button captioned CONFIRM; and `_copilot_panel` reaching its
    controls by calling the helper.

    **The key prefix is what lets a second surface exist.** It is required, every control
    key starts with it, and no two call sites pass the same one, so two surfaces rendering
    the pair in one run cannot collide on a widget key.
    """
    tree = ast.parse(Path(app.__file__).read_text(encoding="utf-8"))
    helper = _function(tree, "_approval_controls")
    inside = {id(node) for node in ast.walk(helper)}
    answers = _calls_to(
        tree, _bindings(tree, "glassbox.live_loop", "answer_pending"), "answer_pending"
    )

    controls = _approval_controls_in(tree, answers)
    stray = [
        f"line {c.lineno}: {ast.unparse(c)}" for c in controls if id(c) not in inside
    ]
    assert not stray, f"an approval control outside _approval_controls: {stray}"
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_approval_controls"
        for node in ast.walk(_function(tree, "_copilot_panel"))
    ), "_copilot_panel no longer reaches its controls through _approval_controls"

    outside = [f"line {a.lineno}" for a in answers if id(a) not in inside]
    assert not outside, f"answer_pending called outside _approval_controls: {outside}"
    literals = [
        call.args[-1].value
        for call in answers
        if call.args and isinstance(call.args[-1], ast.Constant)
    ]
    assert (
        len(answers) == 2 and literals.count(True) == 1 and literals.count(False) == 1
    ), f"{len(answers)} call(s) to answer_pending, answers {literals}: one of each"

    parameters = helper.args
    optional = dict(
        zip([a.arg for a in parameters.kwonlyargs], parameters.kw_defaults, strict=True)
    )
    positional = [a.arg for a in parameters.posonlyargs + parameters.args]
    defaulted = positional[len(positional) - len(parameters.defaults) :]
    assert (
        "key_prefix" in optional or "key_prefix" in positional
    ), "_approval_controls takes no key_prefix"
    assert (
        optional.get("key_prefix") is None and "key_prefix" not in defaulted
    ), "key_prefix has a default, so a surface that forgets it collides silently"
    for control in controls:
        key = _keyed_by_a_decision(control)
        first = key.values[0] if key is not None else None
        assert (
            isinstance(first, ast.FormattedValue)
            and isinstance(first.value, ast.Name)
            and first.value.id == "key_prefix"
        ), f"line {control.lineno}: a control key that does not start with key_prefix"

    prefixes = []
    for site in ast.walk(tree):
        if not (
            isinstance(site, ast.Call)
            and isinstance(site.func, ast.Name)
            and site.func.id == "_approval_controls"
        ):
            continue
        given = [k.value for k in site.keywords if k.arg == "key_prefix"]
        if not given and "key_prefix" in positional:
            index = positional.index("key_prefix")
            given = site.args[index : index + 1]
        assert (
            given
            and isinstance(given[0], ast.Constant)
            and isinstance(given[0].value, str)
        ), f"line {site.lineno}: key_prefix is not a literal string"
        prefixes.append(given[0].value)
    assert len(prefixes) == len(
        set(prefixes)
    ), f"two surfaces share a key_prefix and would collide: {prefixes}"


def test_card_is_gone_and_nothing_calls_it() -> None:
    """**Deleted in region 7, when nothing called it.**

    It survived six regions on purpose: removing it at the start would have left six
    commits behind that could not render a page, which is the opposite of what committing
    per region is for. The guard is on the module, not on the attribute alone, so a
    re-introduced helper under the same name would fail here too.
    """
    source = Path(app.__file__).read_text(encoding="utf-8")

    assert not hasattr(app, "card")
    assert "def card(" not in source
    assert "card(" not in source.replace("region(", ""), "a call site survived"


def test_the_forecast_region_states_that_the_bar_advances_once_per_day() -> None:
    """The model consumes completed daily bars and this subscription refuses intraday
    quotes. Nothing on the page may be styled to imply a live feed, and the caption is
    where that is said rather than left to be inferred from the chart."""
    source = Path(app.__file__).read_text(encoding="utf-8")

    assert "Last completed bar" in source
    assert "advances once per trading day" in source


def test_the_ramp_appears_in_no_chart_that_encodes_no_data() -> None:
    """**The boundary guard, with the hole closed.**

    The earlier version checked the radar and the fold chart, and the forecast chart -
    which was drawing its price line and its forecast in three ramp colours - was not in
    the list. A guard that names its surfaces will always be one surface behind the code;
    this one enumerates every chart builder the module exports and fails on an
    unclassified one, so a new chart must be assigned a side rather than defaulting to
    unchecked.

    **The territory grew on 23 Sep 2026 and the mechanism did not.** D3 put the two
    backtest panels on the meaning side, because each now draws an arm against buy and
    hold and that comparison is data. Two names moved between the lists; nothing was
    exempted, and the same assertion runs over both sides.
    """
    import pandas as pd

    history = pd.Series([100.0, 101.0, 99.5, 102.0, 103.5])
    row = app.PositionRow(
        "AAPL", 1.0, 100.0, 97.0, True, stop_loss=94.0, take_profit=112.0
    )

    ramp_is_meaning = {
        "contributions_svg": ramp_surfaces()["contributions"],
        "spectral_panel": ramp_surfaces()["spectral"],
        "cumulative_equity_svg": ramp_surfaces()["cumulative_equity_svg"],
        "fold_bars_svg": ramp_surfaces()["fold_bars_svg"],
        "radar_svg": ramp_surfaces()["radar_svg"],
    }
    ramp_is_trespass = {
        "forecast_svg": app.forecast_svg(
            history, app.price_path(103.5, [0.01]), Thresholds(lower=0.004), "AAPL"
        ),
        "equity_svg": _equity([100.0, 101.0, 99.0]),
        "countdown_svg": app.countdown_svg(30, 60),
        "sparkline_svg": app.sparkline_svg(history, row, loop_present=True),
    }

    # Every chart builder the module exports is on one side or the other. A new one that
    # is on neither fails here, which is the point: unclassified must not mean unchecked.
    charted = {name for name in dir(app) if name.endswith("_svg")} | {"spectral_panel"}
    classified = (
        set(ramp_is_meaning)
        | set(ramp_is_trespass)
        # Spectral's three builders are one surface: `spectral_panel` above renders them
        # together, so checking each would assert the same thing three times.
        | {"spectral_svg", "gain_phase_svg", "response_svg"}
        # `_svg` is the shared wrapper every chart is built with, not a chart.
        | {"_svg"}
    )
    assert (
        charted <= classified
    ), f"unclassified chart builders: {sorted(charted - classified)}"

    for name, svg in ramp_is_meaning.items():
        assert any(c in svg for c in app.RAMP), f"{name} lost the ramp it needs"
        assert not [
            c for c in app.STATUS_COLOURS if c in svg
        ], f"{name} took status colour"

    for name, svg in ramp_is_trespass.items():
        assert not [c for c in app.RAMP if c in svg], f"the ramp escaped into {name}"


def test_the_forecast_is_dashed_in_the_direction_it_predicts() -> None:
    """A forecast is a direction, so it takes the status colour of the move it predicts.
    The history behind it is context and takes DIM."""
    import pandas as pd

    history = pd.Series([100.0, 101.0, 99.5, 102.0, 103.5])
    rising = app.forecast_svg(
        history, app.price_path(103.5, [0.01]), Thresholds(lower=0.004), "AAPL"
    )
    falling = app.forecast_svg(
        history, app.price_path(103.5, [-0.01]), Thresholds(lower=0.004), "AAPL"
    )

    assert app.GAIN in rising and app.LOSS not in rising
    assert app.LOSS in falling and app.GAIN not in falling
    assert app.DIM in rising, "the price history is context, not a value"


# ── GB-63d: the type scale ───────────────────────────────────────────────────
#
# **The whole point of a scale is that it can be checked in one place.** Before this pass
# the console set type at eight sizes - 7, 8, 9, 11, 11.5, 12.5, 13 and 26 - four of them
# written as bare integers at SVG call sites, where no reviewer and no test ever saw two
# of them together. That is the two-places family at its most diffuse: no two of the eight
# disagreed, because no two were ever compared, and the scale existed only as an average
# of forty independent decisions.
#
# **These guards measure the rendered output, not the files that produce it.** The sizes
# above can be written down here, and `size=9` can be written in a comment in `app.py`,
# without moving a single assertion - which is the property the three previous versions of
# this mistake did not have.

#: Advance width of one monospace character, as a fraction of the em. IBM Plex Mono is
#: 0.6; every fallback in the stack is at or below it, so a label that fits at 0.6 fits in
#: whatever the browser actually loaded.
MONO_ADVANCE = 0.60

#: How far the type extends above the baseline and below it, as fractions of the em.
#: Generous on both sides - the assertion should fail before a glyph is clipped.
ASCENT, DESCENT = 0.80, 0.25

_TEXT = re.compile(
    '<text x="([-0-9.]+)" y="([-0-9.]+)" fill="[^"]*" font-size="([0-9]+)"'
    ' font-family="[^"]*" letter-spacing="[-0-9.]+" text-anchor="([a-z]+)">(.*?)</text>'
)


def _labels(svg: str):
    """``(size, text, x0, x1, top, bottom)`` for every label, in viewBox units."""
    for match in _TEXT.finditer(svg):
        x, y = float(match.group(1)), float(match.group(2))
        size, anchor = int(match.group(3)), match.group(4)
        text = html.unescape(match.group(5))
        width = len(text) * size * (MONO_ADVANCE + app.TRACK_LABEL)
        if anchor == "start":
            x0 = x
        elif anchor == "end":
            x0 = x - width
        else:
            x0 = x - width / 2
        yield size, text, x0, x0 + width, y - size * ASCENT, y + size * DESCENT


def _every_chart() -> dict[str, str]:
    """One rendering of every chart the console draws, including the shapes that appear
    only when something is wrong - a truncated calendar, a band that cannot fire, a
    response with too few points to plot. Those carry the longest strings on the page and
    are exactly the ones nobody looks at before taking a screenshot."""
    reference = app.reference_rows(app.load_results())
    daily = app.load_daily_equity()
    row = app.PositionRow(
        "AAPL", 1.0, 100.0, 97.0, True, stop_loss=94.0, take_profit=112.0
    )
    spectral = a_spectral_attribution()
    return {
        "equity_svg": _equity([100.0, 101.0, 99.0]),
        "equity_svg flat": _equity([100.0, 100.0, 100.0]),
        # **The two shapes the enumeration never held, and the collision lived in the
        # gap.** The balance label rides on `y(opening)`, so it meets the header only when
        # the opening is the session high and the footer only when it is the session low -
        # that is, only in a session that purely loses or purely gains. `[100, 101, 99]`
        # is neither, and it was the only session this list rendered.
        "equity_svg losing": _equity([100_000.18] * 64 + [99_994.20, 99_986.83]),
        "equity_svg gaining": _equity([100.0, 101.0, 102.0]),
        "countdown_svg full": app.countdown_svg(60, 60),
        "countdown_svg spent": app.countdown_svg(0, 60),
        "sparkline_svg": app.sparkline_svg(HISTORY, row, loop_present=True),
        "radar_svg": app.radar_svg(reference, "dlinear"),
        "cumulative_equity_svg": app.cumulative_equity_svg(daily, "dlinear"),
        "fold_bars_svg": app.fold_bars_svg(reference, "dlinear"),
        "forecast_svg": app.forecast_svg(
            HISTORY, app.price_path(103.5, [0.01]), Thresholds(lower=0.004), "AAPL"
        ),
        "forecast_svg no band": app.forecast_svg(
            HISTORY, app.price_path(103.5, [0.004]), Thresholds.never(), "AAPL"
        ),
        "contributions_svg": app.contributions_svg(
            an_attribution(close_logret=0.08, rsi14=-0.06, ma_dist20=0.03)
        ),
        "spectral_svg": app.spectral_svg(spectral),
        "gain_phase_svg": app.gain_phase_svg(spectral),
        "response_svg": app.response_svg(
            [8.0, 12.0, 24.0, 120.0], [0.22, 0.41, 0.83, 0.0]
        ),
        "response_svg thin": app.response_svg([12.0], [0.4]),
    }


def test_every_size_in_the_stylesheet_is_a_step_of_the_scale() -> None:
    """Read off the **rendered** CSS, not the source. A guard on the source counts what a
    file contains, and this project has shipped three of those - a 200-character slice and
    two occurrence counts, all broken by somebody explaining the code underneath them.
    What the browser receives is the thing being claimed."""
    css = app.stylesheet()

    sizes = {float(v) for v in re.findall("font-size:[ ]*([0-9.]+)px", css)}
    leadings = {float(v) for v in re.findall("line-height:[ ]*([0-9.]+)", css)}
    tracking = {float(v) for v in re.findall("letter-spacing:[ ]*(-?[0-9.]+)em", css)}

    assert sizes <= set(tokens.TYPE_STEPS), f"off-scale sizes: {sorted(sizes)}"
    assert leadings <= {
        tokens.LEADING_UI,
        tokens.LEADING_PROSE,
        tokens.LEADING_FIGURE,
    }, f"off-scale leading: {sorted(leadings)}"
    assert tracking <= {
        tokens.TRACK_LABEL,
        tokens.TRACK_FIGURE,
    }, f"off-scale tracking: {sorted(tracking)}"


def test_the_page_states_its_own_leading_rather_than_inheriting_it() -> None:
    """Every size on this page was set and no leading was, so the line spacing of the whole
    console was whatever Streamlit's theme supplied - a typographic decision taken by a
    dependency, and invisible until the dependency changes it."""
    css = app.stylesheet()

    base = css.split(".block-container")[0]

    assert f"line-height: {tokens.LEADING_UI}" in base, "the base sets no leading"
    assert tokens.LEADING_UI >= 1.5, "prose leading is the floor, not an aspiration"
    assert tokens.LEADING_PROSE >= tokens.LEADING_UI


def test_no_chart_writes_a_size_at_its_call_site() -> None:
    """A bare `size=9` in a chart builder is a size nothing can see beside the other seven.

    Asserted as a property of the code rather than as a count of anything: a literal is
    forbidden outright, so a new chart cannot introduce one and no number here needs
    updating when a chart is added.

    **Comments and docstrings are stripped first, and that is the whole point.** Scanning
    raw source would mean the sentence *explaining* why `size=9` was removed reintroduces
    it - documentation becomes a hazard, and the cheapest way to go green is to delete the
    explanation. This project has shipped that defect three times in three costumes: a
    fixed character offset, an occurrence count, and a count inflated by the comment
    introducing the thing it counted. `tokenize` is the boundary that separates what the
    interpreter runs from what a person wrote about it.
    """
    code = []
    with tokenize.open(app.__file__) as handle:
        for token in tokenize.generate_tokens(handle.readline):
            if token.type not in (tokenize.COMMENT, tokenize.STRING):
                code.append(token.string)
    literals = re.findall("size=[0-9]", "".join(code).replace(" ", ""))

    assert not literals, f"{len(literals)} literal type sizes at call sites"


def test_no_chart_label_falls_below_the_type_floor() -> None:
    """**The floor has to hold inside a chart too, and that is where it did not.**

    SVG text is sized in user units, so a label's size on screen is its authored size times
    the ratio of drawn width to viewBox width - which means a chart that scales freely has
    no floor at all. `_svg` pins the drawn width at or above the viewBox width, so a user
    unit is at least a pixel; this asserts the other half, that no chart authors a label
    below the floor to begin with.
    """
    for name, svg in _every_chart().items():
        assert svg, f"{name} rendered nothing, so this test asserted nothing about it"
        for size, text, *_ in _labels(svg):
            assert size >= tokens.TYPE_FLOOR, (
                f"{name} sets {text[:40]!r} at {size} units, below the "
                f"{tokens.TYPE_FLOOR}px floor"
            )


def test_no_chart_label_is_drawn_outside_its_own_chart() -> None:
    """**Raising the type is what makes this necessary, so it arrives with it.**

    Every size on this page went up by three to five units, and a label that fitted at 8
    does not necessarily fit at 12: the radar's axis values, the calendar's weekday ruler
    and the spectral panel's dead-bin labels all had to be given room. A chart whose text
    runs off its own canvas is clipped in silence - it renders, it looks deliberate, and
    the missing half of a caption is visible only to somebody who knew what it said.
    """
    escaped = []
    for name, svg in _every_chart().items():
        box = re.search('viewBox="0 0 ([0-9]+) ([0-9]+)"', svg)
        assert box is not None, f"{name} drew no viewBox"
        edge_x, edge_y = int(box.group(1)), int(box.group(2))
        for _, text, x0, x1, top, bottom in _labels(svg):
            if x0 < 0 or x1 > edge_x or top < 0 or bottom > edge_y:
                escaped.append(
                    f"{name}: {text[:44]!r} spans x {x0:.0f}..{x1:.0f} of {edge_x}, "
                    f"y {top:.0f}..{bottom:.0f} of {edge_y}"
                )

    assert not escaped, "labels drawn outside their chart: " + " | ".join(escaped)


def test_no_chart_label_is_drawn_over_another_label() -> None:
    """**Containment was half the property, and the other half is what shipped.**

    `test_no_chart_label_is_drawn_outside_its_own_chart` measures every label against the
    viewBox, so a caption running off the canvas fails. Two labels at the same coordinates
    are both comfortably inside it and it says nothing: on 3 Sep 2026 the session curve
    rendered `EQ100,000ITY` - the title at x 12.0-65.3 and the opening balance at x
    21.8-84.0, both on y 17-31 - and every geometry guard in this file passed. The type
    pass that raised these labels from 9 units to 12 is what widened the balance from 41.6
    units to 62.2 and pushed it under the title, so the pass that fixed three overlaps by
    eye introduced a fourth and could not see it.

    The overlaps that pass found were found in screenshots. This is the mechanism.
    """
    collisions = []
    for name, svg in _every_chart().items():
        labels = list(_labels(svg))
        for first in range(len(labels)):
            for second in range(first + 1, len(labels)):
                _, text_a, ax0, ax1, atop, abottom = labels[first]
                _, text_b, bx0, bx1, btop, bbottom = labels[second]
                if ax0 < bx1 and bx0 < ax1 and atop < bbottom and btop < abottom:
                    collisions.append(f"{name}: {text_a[:28]!r} over {text_b[:28]!r}")

    assert not collisions, "labels drawn over each other: " + " | ".join(collisions)


# ── GB-63e: the marks, not only the labels ───────────────────────────────────
#
# **A guard built for half a class reads as a guard for the class.** Every geometry
# assertion this file had ran over `<text>`: `_labels` is a regex on text elements, and
# both containment tests above walk it. Every polyline, polygon, rect and circle the
# console draws - which is to say all of the data - was unchecked, and the only assertion
# on any path anywhere is one line inside the forecast test, comparing that chart's
# maximum y against its own annotation strip. One chart, one edge, one direction.


def _attribute(element, name: str) -> float:
    return float(element.get(name, 0.0))


def _marks(svg: str):
    """``(tag, x0, y0, x1, y1)`` for every drawn mark, in viewBox units.

    Polyline, polygon, rect, circle and line. `<text>` is deliberately absent: labels have
    their own guards above, which measure a glyph box this cannot. **The list of tags is
    not written down twice** - it was, as a `MARKS` constant beside a dispatch that names
    each tag again, which is a second copy of a fact with nothing to keep the two equal.
    Each branch computes a different box, so one tuple could never have driven them.

    Parsed as XML rather than scanned for: a mark's extent is a property of its attributes
    and not of the order somebody wrote them in, and this file has shipped three guards
    whose value was a property of the source text rather than of the code.
    """
    for element in ET.fromstring(svg).iter():
        tag = element.tag.split("}")[-1]
        if tag in ("polyline", "polygon"):
            for point in element.get("points", "").split():
                x, y = (float(value) for value in point.split(","))
                yield tag, x, y, x, y
        elif tag == "circle":
            cx, cy = _attribute(element, "cx"), _attribute(element, "cy")
            radius = _attribute(element, "r")
            yield tag, cx - radius, cy - radius, cx + radius, cy + radius
        elif tag == "rect":
            x, y = _attribute(element, "x"), _attribute(element, "y")
            width = _attribute(element, "width")
            height = _attribute(element, "height")
            yield tag, x, y, x + width, y + height
        elif tag == "line":
            x1, y1 = _attribute(element, "x1"), _attribute(element, "y1")
            x2, y2 = _attribute(element, "x2"), _attribute(element, "y2")
            yield tag, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _escapes(svg: str) -> list[str]:
    """Every drawn mark that leaves the chart's own viewBox.

    The test is written as ``not (inside)`` rather than as ``outside`` so that a coordinate
    which is not a number is reported: every comparison against `nan` is false, and a
    vertex with no position is not a vertex that is contained.
    """
    box = re.search('viewBox="0 0 ([0-9]+) ([0-9]+)"', svg)
    assert box is not None, "the chart drew no viewBox"
    edge_x, edge_y = int(box.group(1)), int(box.group(2))
    return [
        f"<{tag}> spans x {x0:.0f}..{x1:.0f} of {edge_x}, y {y0:.0f}..{y1:.0f} of {edge_y}"
        for tag, x0, y0, x1, y1 in _marks(svg)
        if not (x0 >= 0 and y0 >= 0 and x1 <= edge_x and y1 <= edge_y)
    ]


def test_no_chart_draws_a_mark_outside_its_own_chart() -> None:
    """The data half of the containment property, over the same enumeration as the labels.

    A path outside the viewBox is clipped in silence exactly as a label is, and it takes
    the reading with it: a line that leaves the bottom of its box is a number the reader
    cannot see and does not know is missing.
    """
    escaped = []
    for name, svg in _every_chart().items():
        assert svg, f"{name} rendered nothing, so this test asserted nothing about it"
        escaped += [f"{name}: {line}" for line in _escapes(svg)]

    assert not escaped, "marks drawn outside their chart: " + " | ".join(escaped)


def test_the_containment_guard_can_see_a_mark_in_every_chart() -> None:
    """**The guard above is green, and green means nothing until this passes.**

    A containment test over an enumeration is vacuous for any chart whose marks it cannot
    parse - a builder that emitted a `<path d="...">` would sail through it silently, and
    so would one whose output `ET` could not read. This puts a mark outside each chart in
    turn and asserts the checker reports it, so coverage is a property of the enumeration
    rather than of the fixture list somebody happened to write.
    """
    blind = []
    for name, svg in _every_chart().items():
        edge_y = int(re.search('viewBox="0 0 [0-9]+ ([0-9]+)"', svg).group(1))
        below = svg.replace(
            "</svg>",
            f'<polyline points="10.0,{edge_y + 40}.0 20.0,{edge_y + 40}.0"/></svg>',
        )
        if not _escapes(below):
            blind.append(name)

    assert not blind, "the guard cannot see a mark in: " + ", ".join(blind)


#: D7's sparkline case: a long whose price has gapped below its own stop, so the stop is
#: above every visible bar. Stop stated at the broker since D1 (22 Sep 2026): the
#: sparkline draws a stop rule only from a live stop. The book's level is not the live
#: one, so a sparkline reading the book cannot pass the premise test that follows.
D7_GAPPED_BELOW_STOP = app.PositionRow(
    "AAPL",
    1.0,
    100.0,
    97.0,
    True,
    stop_loss=94.0,
    take_profit=float("nan"),
    live_stop=1e6,
    stop_working=True,
    orders_read=True,
)


@pytest.mark.xfail(strict=True, reason="GB-63e found these; closing them is not GB-63e")
@pytest.mark.parametrize(
    "name",
    ["sparkline stop off scale", "response negative gain", "forecast non-finite close"],
)
def test_a_chart_holds_its_marks_on_a_series_that_runs_off_the_scale(name: str) -> None:
    """**Three escapes the widened guard found, recorded as failures rather than as prose.**

    Each is a scale computed from a subset of what is then drawn against it, which is the
    same defect three times:

    - `sparkline_svg` takes ``low = min(prices.min(), stop)`` but never takes the max with
      it, so a long whose price has gapped below its own stop - the stop is then above
      every visible bar - draws that rule at y = -12,998,634 of 78. Reachable today: it
      needs only a quarantined target, which the builder already branches on. **The
      figure moved on 23 Sep 2026** and the defect did not: D5a reserved a 14px band at
      the top of the sparkline for its symbol, so the canvas is 78 units rather than 64
      and the plot starts lower. It was -12,998,648 of 64.
    - `response_svg` normalises by ``max(gains)``, so a negative gain larger in magnitude
      than the largest positive one puts a vertex at y = 1340 of 288.
    - `forecast_svg` takes low and high over the *finite* values and then maps every value,
      so one non-finite close emits ``y=-inf``.

    `xfail(strict=True)` rather than a comment, because a gap that is written down is a
    note: this turns red the day one of them is fixed and the entry is removed with it.
    """
    gapped = HISTORY.copy()
    gapped.iloc[2] = float("inf")
    svg = {
        # Stop stated at the broker since D1 (22 Sep 2026): the sparkline draws a stop
        # rule only from a live stop.
        "sparkline stop off scale": lambda: app.sparkline_svg(
            HISTORY, D7_GAPPED_BELOW_STOP, loop_present=True
        ),
        "response negative gain": lambda: app.response_svg(
            [8.0, 12.0, 24.0, 120.0], [0.22, 0.41, 0.83, -5.0]
        ),
        "forecast non-finite close": lambda: app.forecast_svg(
            gapped, app.price_path(103.5, [0.01]), Thresholds(lower=0.004), "AAPL"
        ),
    }[name]()

    assert not _escapes(svg), f"{name}: " + " | ".join(_escapes(svg))


def test_the_d7_sparkline_fixture_draws_its_live_stop_above_every_close() -> None:
    """**The premise of the strict xfail above, held outside it.** An assertion inside an
    xfail is swallowed by it, so without this the sparkline case could go on failing after
    its rule had silently stopped drawing, and read as D7 when it was not.

    One stop rule, drawn from the live 1e6 rather than the book's 94.00, and therefore
    above every close - which is the scenario D7 is about. Stated as position relative to
    the closes rather than as a coordinate, so it stays true once D7 fixes the scale.
    """
    svg = app.sparkline_svg(HISTORY, D7_GAPPED_BELOW_STOP, loop_present=True)

    rules = re.findall(
        rf'<line x1="[^"]*" y1="([^"]*)" x2="[^"]*" y2="[^"]*" stroke="{app.ORANGE}"',
        svg,
    )
    points = re.search(
        rf'<polyline points="([^"]*)" fill="none" stroke="{app.PAPER}"', svg
    )
    closes = [float(point.split(",")[1]) for point in points.group(1).split()]

    assert len(rules) == 1, "the fixture draws exactly one stop rule"
    assert float(rules[0]) < min(closes), "the rule is above every close"


def test_a_chart_never_scales_below_the_viewbox_its_type_is_measured_in() -> None:
    """The mechanism behind the floor, asserted rather than described.

    Without it the floor is a statement about authored user units and says nothing about
    what a reader sees: at 1400 units in a 1000px column a 12-unit label is 8.6px. Deleting
    the `min-width` would leave every other test in this file green.
    """
    svg = app.countdown_svg(30, 60)

    opening = svg[: svg.index(">") + 1]

    assert "min-width:460px" in opening, "the chart can scale below its own viewBox"
    assert 'width="100%"' in opening, "and it must still grow to fill a wide column"


def test_the_charts_and_the_page_set_type_in_one_family() -> None:
    """`_text` wrote its own font stack and IBM Plex Mono was not in it. Every chart label
    set in the system monospace while the page around it set in Plex - on a console whose
    stylesheet asserts one family, in the charts that fill six report screenshots."""
    svg = app.countdown_svg(30, 60)

    assert f'font-family="{tokens.MONO}"' in svg
    assert tokens.MONO in app.stylesheet()


def test_chart_tracking_is_a_fraction_of_the_size_not_a_constant() -> None:
    """A fixed 1.2 user units is 17% of the em at size 7 and 11% at size 11, so the
    smallest labels - already the hardest to read - were tracked half again as loosely as
    the largest. The relationship is now the stylesheet's, in both directions."""
    svg = _equity([100.0, 101.0])

    pairs = {
        (int(size), float(track))
        for size, track in re.findall(
            'font-size="([0-9]+)" font-family="[^"]*" letter-spacing="([-0-9.]+)"', svg
        )
    }

    assert (
        len(pairs) > 1
    ), "this chart draws one size, so it cannot show the relationship"
    for size, track in pairs:
        assert track == pytest.approx(size * tokens.TRACK_LABEL, abs=0.01)


# ── 3 Sep 2026: a working look-ahead guard that read as a dead feed ──────────
#
# The panel said `LIVE 2026-09-02 bar` while the clock said the 3rd and the market was
# open. Nothing was wrong: the loop drops the in-progress bar because the session has not
# closed. But the panel stated the fact and withheld the reason, so a correct guard and a
# stale feed rendered identically - and the person who misread it wrote the loop. These
# tests hold the reason in place, and hold the pill to the one claim it can support.

_OPEN_CLOCK = pd.Timestamp("2026-09-03 19:13", tz="UTC")
_LAST_BAR = pd.Timestamp("2026-09-02")
_MANDATE = "Last completed bar 2026-09-02 · advances once per trading day"


def _region_notes() -> list[tuple[ast.expr | None, ast.expr | None]]:
    """Every ``region(...)`` call in the module as ``(source, note)`` AST nodes."""
    tree = ast.parse(Path(app.__file__).read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "region"
        ):
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        source = node.args[1] if len(node.args) > 1 else keywords.get("source")
        note = node.args[3] if len(node.args) > 3 else keywords.get("note")
        calls.append((source, note))
    return calls


def test_a_closed_market_captions_the_chart_exactly_as_it_did_before() -> None:
    """The old wording is complete when the market is shut: there is no absence to
    explain, and a panel that explains itself when nothing is missing teaches a reader to
    stop reading the line."""
    note = app.bar_note(_LAST_BAR, _OPEN_CLOCK, session_open=False)

    assert note == _MANDATE


def test_an_open_session_says_why_today_has_no_bar() -> None:
    """The defect, stated as the behaviour that fixes it."""
    note = app.bar_note(_LAST_BAR, _OPEN_CLOCK, session_open=True)

    assert note == (
        f"{_MANDATE} · today's session is open, so 2026-09-03 has no completed bar yet"
    )


def test_the_mandated_string_stays_first_and_whole_in_both_branches() -> None:
    """``GLASSBOX_PHASE2_EXPANSION.md`` requires the chart to carry
    ``LAST COMPLETED BAR <date> · ADVANCES ONCE PER TRADING DAY``. The reason is appended
    to that sentence and may never replace it, so this session cannot drift the panel out
    of the document that governs its scope."""
    for session_open in (True, False):
        note = app.bar_note(_LAST_BAR, _OPEN_CLOCK, session_open=session_open)
        assert note.startswith(_MANDATE)


def test_no_previous_bar_is_captioned_without_its_reason_while_open() -> None:
    """**The guard the session was opened for.**

    Not one date but every date the panel could hold during an open session: a bar from
    yesterday, from a long weekend, from a fortnight of held positions. If any of them can
    reach the caption without the explanation, the reader is back to telling a working
    look-ahead guard from a dead feed by eye.
    """
    for days in range(1, 15):
        as_of = _OPEN_CLOCK.tz_localize(None).normalize() - pd.Timedelta(days=days)
        note = app.bar_note(as_of, _OPEN_CLOCK, session_open=True)

        assert "today's session is open" in note, f"{as_of:%Y-%m-%d} explains nothing"
        assert "has no completed bar yet" in note
        assert f"{_OPEN_CLOCK:%Y-%m-%d}" in note, "the absent day is not named"


def test_a_bar_from_the_current_day_is_not_explained_away() -> None:
    """The clause is owed only when today is genuinely absent. A bar carrying today's date
    during an open session is a different situation - and one the look-ahead guard says
    should not arise - so the caption must not narrate an absence that is not there."""
    note = app.bar_note(
        _OPEN_CLOCK.tz_localize(None).normalize(), _OPEN_CLOCK, session_open=True
    )

    assert note == "Last completed bar 2026-09-03 · advances once per trading day"


def test_the_forecast_card_takes_its_caption_from_the_real_session_state() -> None:
    """**A test that exercises a function is not a test that the function is called.**

    ``explain_spectral`` was defined, exported, unit-tested and had no caller, and the
    panel it fed was empty for a week. The five tests above would pass identically if
    ``main`` still built its caption inline, so this one reads the call site: the forecast
    region's note comes from ``bar_note``, its pill from ``forecast_source``, and neither
    argument to ``bar_note`` is a literal - a hardcoded ``True`` or a fixed date makes the
    branch unreachable while every test above stays green.
    """
    wired = [
        (source, note)
        for source, note in _region_notes()
        if isinstance(note, ast.Call)
        and isinstance(note.func, ast.Name)
        and note.func.id == "bar_note"
    ]

    assert len(wired) == 1, "exactly one region captions itself by session state"
    source, note = wired[0]
    assert isinstance(source, ast.Call) and isinstance(source.func, ast.Name)
    assert source.func.id == "forecast_source", "the card kept an inline Source()"
    assert len(note.args) == 3, "bar_note is called with bar, clock and session state"
    assert not [arg for arg in note.args if isinstance(arg, ast.Constant)], (
        "a literal clock or session flag makes the open-session branch unreachable "
        "without failing a single assertion above"
    )


def test_the_caption_is_written_in_exactly_one_place() -> None:
    """The wiring guard above proves one call site reaches ``bar_note``; this proves no
    other call site writes the caption itself. Docstrings are excluded because the
    sentence explaining a rule must not be able to break it - the fixed-offset defect in
    its fifth costume."""
    tree = ast.parse(Path(app.__file__).read_text(encoding="utf-8"))
    authors = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        docstring = ast.get_docstring(node, clean=False)
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Constant)
                and isinstance(child.value, str)
                and child.value != docstring
                and "Last completed bar" in child.value
            ):
                authors.add(node.name)

    assert authors == {"bar_note"}, f"the caption is also built in {sorted(authors)}"


def test_the_pill_names_its_source_and_has_no_way_to_name_freshness() -> None:
    """**The pill states the SOURCE. A reader took it for FRESHNESS, which it never
    measured.**

    The type exists to stop backtest numbers wearing a live face; no test has ever asked
    it about recency. Freshness is already stated twice - ``staleness_html`` and ``LAST
    CYCLE`` in ``session_strip`` - and a third copy here is a second place for one fact to
    diverge from.

    The signature is the mechanism rather than the docstring, and D2 kept it that way. On
    23 Sep 2026 the pill's kind became the page's liveness verdict, because a green LIVE
    pill on a loop that stopped on 3 Sep is the same misreading this test was written for.
    What the function receives is the **word** - never the verdict object, which carries
    both ages, and never a clock - so it still cannot say how old the bar is.
    """
    source = app.forecast_source(_LAST_BAR, app.LIVE)

    assert source.kind == app.LIVE
    assert source.detail == "live feed · 2026-09-02 bar"
    assert list(inspect.signature(app.forecast_source).parameters) == ["as_of", "state"]
    assert app.forecast_source(_LAST_BAR, app.NOT_RUNNING).kind == app.NOT_RUNNING


def test_the_reason_reaches_the_page_as_written() -> None:
    """``region`` escapes its note, and ``escape`` here covers ``&<>`` but not the
    apostrophe in *today's*. That is load-bearing and undeclared, so it is asserted
    against the renderer rather than against a reading of it: swapping in ``html.escape``
    would print ``today&#x27;s`` on the face of the panel and fail here."""
    note = app.bar_note(_LAST_BAR, _OPEN_CLOCK, session_open=True)

    rendered = app.region(
        "AAPL close and forecast",
        app.forecast_source(_LAST_BAR, app.LIVE),
        "<svg></svg>",
        note,
    )

    assert note in rendered, "the caption was altered on its way to the page"
    assert "live feed · 2026-09-02 bar" in rendered
