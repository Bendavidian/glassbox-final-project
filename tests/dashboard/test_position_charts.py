"""Every position gets a slot, every slot names its symbol, and none of them is silent.

**D5a, 23 Sep 2026.** The POSITIONS panel held three rows for the first time - UNH, V and
WMT - and rendered **one** chart beneath them, carrying no symbol. UNH and V are
quarantined, so `sparkline_svg` returned an empty string for each and `st.markdown("")`
drew nothing; the chart that did render was WMT's, sitting directly under a three-row
table with nothing on it to say so. A reader attaches that chart to the first row.

Three ways a chart went missing, all silent, none logged:

- the price fetch failed, which removed **every** chart at once and looked exactly like
  three quarantined positions (`_recent_closes` returned ``{}`` for both);
- the symbol had no recent closes;
- the position is held at the broker but was never opened by this system.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from glassbox.dashboard import app
from glassbox.engine.reconcile import Book, Holding

CLOSES = pd.Series([100.00, 101.00, 99.50, 102.00, 103.50])

#: The panel as it stood on 23 Sep 2026: one position the system opened, two it did not.
WMT = Holding(
    symbol="WMT",
    quantity=92.038821904,
    decision_id="d-wmt",
    entry_price=108.70,
    stop_loss=105.39,
    take_profit=115.22,
)
HELD = {"UNH": 3.0, "V": 5.0, "WMT": WMT.quantity}
MARKS = {"UNH": 300.00, "V": 280.00, "WMT": 107.91}


def three_rows() -> list[app.PositionRow]:
    stops = app.live_stops([app.OpenOrder("WMT", "sell", "stop", 105.39)])
    return app.position_rows(Book(managed={"WMT": WMT}), HELD, MARKS, stops)


def slots(
    closes: dict[str, pd.Series] | None, *, loop_present: bool = True
) -> list[str]:
    """One slot per row, exactly as `main` builds them."""
    return [
        app.sparkline_slot(
            row,
            (closes or {}).get(row.symbol),
            prices_read=closes is not None,
            loop_present=loop_present,
        )
        for row in three_rows()
    ]


def _texts(svg: str) -> list[str]:
    """The visible text of a chart: `<text>` contents, never the aria-label."""
    return [value.strip() for value in re.findall(r"<text[^>]*>([^<]*)</text>", svg)]


# ── every chart names itself ─────────────────────────────────────────────────


def test_every_rendered_sparkline_carries_its_symbol_as_visible_text() -> None:
    """**Enumerated over positions, not written for one.** The aria-label named the
    symbol before this and named it invisibly, which is why one unlabelled chart under a
    three-row table could be read as the first row's."""
    closes = {symbol: CLOSES for symbol in HELD}
    charts = {
        row.symbol: app.sparkline_svg(CLOSES, row, loop_present=True)
        for row in three_rows()
        if app.sparkline_svg(CLOSES, row, loop_present=True)
    }

    assert charts, "no chart rendered, so this test asserted nothing"
    for symbol, svg in charts.items():
        assert symbol in _texts(svg), f"{symbol}'s chart does not name itself visibly"
    # And through the slot, which is what the page actually renders.
    for slot in slots(closes):
        if slot.startswith("<svg"):
            assert any(symbol in _texts(slot) for symbol in HELD)


def test_the_symbol_survives_stripping_every_aria_label() -> None:
    """The label that was already there is not the fix. Remove it and the symbol must
    still be on the face of the chart."""
    row = next(row for row in three_rows() if row.symbol == "WMT")

    svg = app.sparkline_svg(CLOSES, row, loop_present=True)
    stripped = re.sub(r'aria-label="[^"]*"', "", svg)

    assert "WMT" in stripped
    assert "WMT" in _texts(stripped)


# ── a slot per row, and a reason in each empty one ───────────────────────────


def test_three_rows_render_three_slots_and_none_of_them_is_empty() -> None:
    """**The assertion that makes silent loss impossible.** Three positions, three slots,
    each either a chart or a stated reason - which is what the panel did not do."""
    for closes, reading in (
        ({symbol: CLOSES for symbol in HELD}, "prices read"),
        ({}, "prices read, nothing returned"),
        (None, "fetch failed"),
    ):
        rendered = slots(closes)

        assert len(rendered) == len(HELD), f"{reading}: a row lost its slot"
        for row, slot in zip(three_rows(), rendered, strict=True):
            assert slot.strip(), f"{reading}: {row.symbol} rendered nothing at all"
            assert (
                slot.startswith("<svg") or row.symbol in slot
            ), f"{reading}: {row.symbol}'s slot names no symbol"


@pytest.mark.parametrize(
    ("closes", "symbol", "reason"),
    [
        (None, "WMT", app.PRICES_UNREADABLE),
        ({}, "WMT", app.NO_RECENT_CLOSES),
        ({"UNH": CLOSES}, "UNH", app.NOT_IN_BOOK),
    ],
    ids=["prices could not be read", "no recent closes", "not opened by this system"],
)
def test_each_reason_renders_in_place_naming_the_symbol(
    closes: dict[str, pd.Series] | None, symbol: str, reason: str
) -> None:
    """One test per reason. Each names the symbol, because a reason attached to no symbol
    is the same ambiguity as a chart attached to no symbol."""
    rendered = dict(
        zip([row.symbol for row in three_rows()], slots(closes), strict=True)
    )

    assert reason in rendered[symbol]
    assert symbol in rendered[symbol]


def test_the_three_reasons_are_three_different_sentences() -> None:
    """A future edit that collapses them into one generic message fails here. They are
    different problems: a broken feed is not a symbol without history, and neither is a
    position the system never opened."""
    reasons = (app.PRICES_UNREADABLE, app.NO_RECENT_CLOSES, app.NOT_IN_BOOK)

    assert len(set(reasons)) == 3
    # And the failed-fetch message may not claim the symbol has no data, which is the
    # misreading it exists to prevent.
    assert "SYMBOL" not in app.PRICES_UNREADABLE
    assert slots(None)[0] != slots({})[0], "a broken feed reads as an empty one"


# ── the value-level distinction the messages rest on ─────────────────────────


def test_a_failed_price_read_answers_none_and_an_empty_one_answers_a_mapping(
    monkeypatch, cfg_stub
) -> None:
    """**Asserted at the value, not through the message.** `_recent_closes` returned ``{}``
    for both a failed read and an empty one, so the page could not tell them apart at any
    distance from here - the three vanished charts were that single value, rendered.
    """
    from glassbox.data import live

    def explode(*_args, **_kwargs):
        raise RuntimeError("the feed is down")

    monkeypatch.setattr(live, "load_live_bars", explode)
    assert app._recent_closes(cfg_stub) is None, "a failed read must not answer {}"

    monkeypatch.setattr(live, "load_live_bars", lambda *_a, **_k: {})
    assert (
        app._recent_closes(cfg_stub) == {}
    ), "an empty read is a mapping, not a failure"
