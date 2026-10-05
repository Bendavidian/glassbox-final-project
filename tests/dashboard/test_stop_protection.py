"""The positions panel claims a stop only when one is working at the broker.

**D1, 22 Sep 2026.** The panel read ``WMT ... STOP ROOM 76% TO 105.39 ... MANAGED`` while
the broker held no open order at all: WMT's DAY stop 8820c4d7 expired at 20:01:31Z on
3 Sep 2026 and nothing re-armed it. The level came from ``book.json``, which records what
the loop *intended* when it opened the position, and an intention is not protection. The
sparkline under the row drew the same level as a dashed stop rule. Each branch is pinned
here - a live order, a trailing stop with and without a level, no live order, and a read
that failed - because the defect was one branch answering for another.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from glassbox.dashboard import app, tokens
from glassbox.engine.reconcile import Book, Holding
from tests.dashboard.conftest import a_verdict

#: WMT as the book held it on 22 Sep 2026. The stop cell does not read the target.
WMT = Holding(
    symbol="WMT",
    quantity=92.038821904,
    decision_id="d-wmt",
    entry_price=108.70,
    stop_loss=105.39,
    take_profit=math.nan,
)

#: A last price at which the book's intended stop reproduces the panel's "76% TO 105.39"
#: exactly, so the defect's own string is what each refusal asserts is absent.
PRICE = 107.91


def a_stop(order_type: str = "stop", level: float | None = 104.00) -> app.OpenOrder:
    return app.OpenOrder("WMT", "sell", order_type, level)


def _row(
    stops: dict[str, app.OpenOrder] | None,
    price: float = PRICE,
    holding: Holding = WMT,
) -> app.PositionRow:
    symbol = holding.symbol
    return app.position_rows(
        Book(managed={symbol: holding}),
        {symbol: holding.quantity},
        {symbol: price},
        stops,
    )[0]


def _rendered(stops: dict[str, app.OpenOrder] | None) -> str:
    # A running loop, so STATE can still say MANAGED: these tests are about the stop, and
    # `test_liveness.py` holds what the loop's own state does to that word.
    return app.position_table([_row(stops)], a_verdict())


def _cell(html: str, header: str) -> tuple[str, str]:
    """``(class, content)`` of the one rendered row's cell under ``header``.

    The column is found by its heading rather than by an index, so a column added before it
    moves the lookup with it instead of silently reading a neighbour.
    """
    heads = re.findall(r"<th[^>]*>(.*?)</th>", html)
    cells = re.findall(r'<td class="([^"]*)">(.*?)</td>', html)
    return cells[heads.index(header)]


def test_the_book_stop_would_have_reproduced_the_panel() -> None:
    """The fixture's premise, asserted: without it the absence checks below could pass
    because the fixture never produced the defect's string in the first place."""
    # The panel's formula on 22 Sep 2026, kept as history: the share of the entry-to-stop
    # distance. Since 5 Oct 2026 the cell prints the fall to the stop instead, which would
    # read "2.3% TO 105.39" here; the absence checks look for "105.39" and "% TO", which
    # neither formula can produce without the book's level reaching the cell.
    room = (PRICE - WMT.stop_loss) / (WMT.entry_price - WMT.stop_loss)

    assert f"{room * 100:.0f}% TO {WMT.stop_loss:,.2f}" == "76% TO 105.39"


# ── the stop cell ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("order_type", ["stop", "stop_limit", "trailing_stop"])
def test_the_stop_room_is_measured_to_the_order_working_at_the_broker(
    order_type: str,
) -> None:
    """(i) A live stop at a level **different** from the book's, so the source is visible
    in the output: (107.91 - 104.00) / 107.91 is 3.6%, and the book's 105.39 would have
    read 2.3%. A trailing stop that reports its level is measured the same way. (Until
    5 Oct 2026 the figure was the share of the entry-to-stop distance, and this test
    pinned 83% against the book's 76%.)
    """
    html = _rendered(app.live_stops([a_stop(order_type, 104.00)]))

    assert _cell(html, "STOP ROOM") == ("", "3.6% TO 104.00")
    assert "105.39" not in html
    assert _cell(html, "STATE")[1] == "MANAGED"


def test_no_working_stop_reads_unprotected_and_never_as_room() -> None:
    """(ii) The read succeeded and the broker holds nothing: WMT on 22 Sep 2026."""
    html = _rendered({})

    klass, cell = _cell(html, "STOP ROOM")
    assert cell.startswith("NO LIVE STOP · UNPROTECTED")
    # The book's level is shown as intent, in the dim colour, never as a room figure.
    assert '<span class="gb-meta">INTENDED 105.39 NOT AT BROKER</span>' in cell
    assert "% TO" not in html
    # In the accent, through the table's existing mark, and never in loss red: loss is a
    # direction of value, and an absent stop is a fact about protection.
    assert klass == "gb-flag"
    assert tokens.LOSS.lower() not in cell.lower()
    assert "MANAGED" not in html


def test_a_failed_order_read_says_so_and_never_falls_back_to_the_book() -> None:
    """(iii) "Could not ask" is not "no stop", and it is certainly not "protected"."""
    html = _rendered(None)

    klass, cell = _cell(html, "STOP ROOM")
    assert cell == "STOP UNKNOWN · BROKER READ FAILED"
    assert klass == "gb-flag"
    assert "% TO" not in html
    assert "105.39" not in html
    assert "MANAGED" not in html


def test_a_trailing_stop_with_no_reported_level_is_protection_without_a_room() -> None:
    """Protected, so not marked and not UNPROTECTED; no level, so no room figure. Printing
    UNPROTECTED here would be the defect in the other direction."""
    html = _rendered(app.live_stops([a_stop("trailing_stop", None)]))

    assert _cell(html, "STOP ROOM") == ("", "LIVE TRAILING STOP · LEVEL NOT REPORTED")
    assert "% TO" not in html
    assert _cell(html, "STATE")[1] == "MANAGED"


def test_the_stop_note_promises_no_order_the_broker_does_not_hold() -> None:
    """At 104.00 WMT is through its intended 105.39. The stop note says *the stop is a
    market order at the broker*; with no order there, that sentence is the same false
    claim as the room figure, in prose."""
    assert app.stop_note(_row({}, price=104.00)) == ""


# ── the room figure ──────────────────────────────────────────────────────────

#: AMZN as the book held it on 5 Oct 2026. The broker's stop is the book's 238.3775 at a
#: cent's precision, and the panel read "214% TO 238.38" with the last price at 253.70.
AMZN = Holding(
    symbol="AMZN",
    quantity=0.888706071,
    decision_id="20260925-AMZN",
    entry_price=245.54,
    stop_loss=238.3775,
    take_profit=260.495,
)


def _amzn(price: float) -> str:
    stops = app.live_stops([app.OpenOrder("AMZN", "sell", "stop", 238.38)])
    return app.position_table([_row(stops, price, AMZN)], a_verdict())


def test_the_stop_room_is_the_fall_to_the_stop_as_a_share_of_the_last_price() -> None:
    """AMZN on 5 Oct 2026: (253.70 - 238.38) / 253.70 is 6.0%, the fall that fills the
    stop. Beside a price level that is the only reading a figure invites, and the share
    of the entry-to-stop distance it replaced, 214%, invited it falsely."""
    assert _cell(_amzn(253.70), "STOP ROOM") == ("", "6.0% TO 238.38")


@pytest.mark.parametrize("price", [238.38, 237.00], ids=["at the stop", "through it"])
def test_a_price_at_or_below_its_stop_says_so_in_words(price: float) -> None:
    """No zero and no negative room: "0.0% TO" reads as a figure, and "-0.6% TO" as a
    room the reader has to decode. The mark is the band's, and does not change."""
    klass, cell = _cell(_amzn(price), "STOP ROOM")

    assert cell == "AT OR BELOW STOP 238.38"
    assert klass == "gb-flag"


def test_a_room_the_old_figure_put_above_100_percent_reads_below_it() -> None:
    """WMT at 115.00 against a live 104.00: 234% of its entry-to-stop distance, because a
    price above its entry has more than all the room it was given. As a fall to the stop
    it is 9.6%, and no long above a positive stop can need 100%."""
    row = _row(app.live_stops([a_stop("stop", 104.00)]), price=115.00)
    # The premise, or the test could pass on a row that never exceeded 100%.
    assert f"{row.stop_room * 100:.0f}%" == "234%"

    cell = _cell(app.position_table([row], a_verdict()), "STOP ROOM")[1]

    assert cell == "9.6% TO 104.00"
    assert float(cell.split("%")[0]) < 100


# ── which orders are protection ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "order",
    [
        app.OpenOrder("WMT", "sell", "limit", None),  # a target: it fills on the way up
        # Alpaca sends no stop price on a limit or a market order. These carry one anyway,
        # so the order type is the only thing that can refuse them: without them, counting
        # a limit sell as a stop was a break no test could see.
        app.OpenOrder("WMT", "sell", "limit", 104.00),
        app.OpenOrder("WMT", "sell", "market", 104.00),
        app.OpenOrder("WMT", "buy", "stop", 104.00),  # protects nothing on a long
        app.OpenOrder("AAPL", "sell", "stop", 104.00),  # somebody else's stop
    ],
    ids=[
        "limit sell",
        "limit sell with a stop price",
        "market sell with a stop price",
        "buy stop",
        "other symbol",
    ],
)
def test_only_a_sell_stop_on_the_symbol_protects_it(order: app.OpenOrder) -> None:
    html = _rendered(app.live_stops([order]))

    assert _cell(html, "STOP ROOM")[1].startswith("NO LIVE STOP · UNPROTECTED")


def test_the_highest_reported_level_is_the_one_measured() -> None:
    """On a long the highest stop fills first, and a level the broker reports outranks a
    trailing stop that reports none."""
    orders = [a_stop("trailing_stop", None), a_stop("stop", 103.00), a_stop("stop")]

    # 103.00 would read 4.6%. Until 5 Oct 2026 this pinned "83% TO 104.00".
    assert _cell(_rendered(app.live_stops(orders)), "STOP ROOM")[1] == "3.6% TO 104.00"


def test_the_order_read_asks_for_open_orders_and_keeps_the_stop_price() -> None:
    """The SDK boundary. The stub offers ``get_orders`` and nothing else, so a read that
    reached for any other client method - one that could change an order - fails."""
    from alpaca.trading.enums import OrderSide, OrderType, QueryOrderStatus

    asked = []

    class Client:
        def get_orders(self, request):
            asked.append(request)
            return [
                SimpleNamespace(
                    symbol="WMT",
                    side=OrderSide.SELL,
                    order_type=OrderType.STOP,
                    stop_price="104.0",
                ),
                SimpleNamespace(
                    symbol="WMT",
                    side=OrderSide.SELL,
                    order_type=OrderType.TRAILING_STOP,
                    stop_price=None,
                ),
            ]

    orders = app.open_orders(Client())

    assert [request.status for request in asked] == [QueryOrderStatus.OPEN]
    assert orders == [a_stop("stop", 104.0), a_stop("trailing_stop", None)]


# ── the sparkline ────────────────────────────────────────────────────────────

#: With a target, so the band is drawn and can be seen to go with the stop.
TARGETED = replace(WMT, take_profit=112.00)

#: The middle close sits exactly at the live stop, so the rule's height can be read off
#: the chart's own price line rather than recomputed here from a copy of its scale.
CLOSES = pd.Series([100.00, 104.00, 110.00])


def _sparkline(stops: dict[str, app.OpenOrder] | None) -> str:
    return app.sparkline_svg(CLOSES, _row(stops, holding=TARGETED), loop_present=True)


def _stop_rules(svg: str) -> list[str]:
    """The height of every vermillion line. The stop rule is the only one the sparkline
    draws; the target's is ``ORANGE_DIM`` and the band is a fill."""
    return re.findall(
        rf'<line x1="[^"]*" y1="([^"]*)" x2="[^"]*" y2="[^"]*" stroke="{app.ORANGE}"',
        svg,
    )


def _close_heights(svg: str) -> list[str]:
    line = re.search(
        rf'<polyline points="([^"]*)" fill="none" stroke="{app.PAPER}"', svg
    )
    return [point.split(",")[1] for point in line.group(1).split()]


def test_the_sparkline_rule_is_drawn_at_the_live_stop() -> None:
    """At the height of the 104.00 close, where the live stop is, and not at the book's
    105.39: one rule, and the band it bounds."""
    svg = _sparkline(app.live_stops([a_stop("stop", 104.00)]))

    assert _stop_rules(svg) == [_close_heights(svg)[1]]
    assert f'fill="{app.ORANGE}"' in svg
    assert "against its live stop" in svg


@pytest.mark.parametrize(
    "stops",
    [{}, None, {"WMT": a_stop("trailing_stop", None)}],
    ids=["no live stop", "read failed", "trailing stop with no level"],
)
def test_the_sparkline_draws_no_stop_it_cannot_place_at_the_broker(
    stops: dict[str, app.OpenOrder] | None,
) -> None:
    """No rule and no band in any style: a line at the book's 105.39 reads as a stop
    whether it is dashed, dim or solid, and so does the edge of a band."""
    svg = _sparkline(stops)

    assert _stop_rules(svg) == []
    assert f'fill="{app.ORANGE}"' not in svg
    # The label names what is drawn, so it may still name the target - the loop is running
    # in these cases - but it may not name a stop.
    assert "live stop" not in svg
    assert _close_heights(svg), "the closes are still drawn"


def _entry_rules(svg: str) -> list[str]:
    """The height of every solid hairline: the entry rule. The target's rule is the same
    colour but dashed, and WMT has no target here."""
    return re.findall(
        rf'<line x1="[^"]*" y1="([^"]*)" x2="[^"]*" y2="[^"]*" '
        rf'stroke="{app.HAIRLINE}" stroke-width="1"/>',
        svg,
    )


@pytest.mark.parametrize(
    "book_stop", [105.39, math.nan], ids=["book stop", "no book stop"]
)
def test_a_position_the_system_opened_keeps_its_closes_and_entry_without_a_stop(
    book_stop: float,
) -> None:
    """**The gate is "the system opened it", not "the book has a stop".** A position with
    no live stop keeps its price line and its entry and loses only the stop rule and the
    band. The case without a book level is the one the two gates disagree on: under the
    old gate it drew nothing at all."""
    closes = pd.Series([106.00, 108.70, 110.00])  # the middle close sits at the entry

    svg = app.sparkline_svg(
        closes, _row({}, holding=replace(WMT, stop_loss=book_stop)), loop_present=True
    )

    heights = _close_heights(svg)
    assert _entry_rules(svg) == [heights[1]]
    assert _stop_rules(svg) == []
    assert f'fill="{app.ORANGE}"' not in svg


def test_a_position_the_system_did_not_open_draws_no_sparkline() -> None:
    """The other half of the gate: a quarantined holding has no entry and no stop the
    system set, so there is nothing to draw its closes against."""
    row = app.position_rows(Book(), {"WMT": 1.0}, {"WMT": PRICE}, {})[0]

    assert app.sparkline_svg(CLOSES, row, loop_present=True) == ""


# ── the raw column's blast radius ────────────────────────────────────────────

#: Everything `stop_cell` may emit that is not a number: its one class, its closing tag
#: and its fixed phrases. Written out here rather than read from the module, so a change
#: to the wording is a change to this test too.
FIXED = (
    '<span class="gb-meta">',
    "</span>",
    "LIVE TRAILING STOP · LEVEL NOT REPORTED",
    "STOP UNKNOWN · BROKER READ FAILED",
    "NO LIVE STOP · UNPROTECTED",
    "NOT AT BROKER",
    "INTENDED",
    "AT OR BELOW STOP",
    "% TO",
    "—",
)

#: What the numbers and separators are made of. ``,`` is the thousands separator of
#: ``:,.2f``, and it opens neither a tag nor an entity. (Until 5 Oct 2026 this also allowed
#: ``-``, the sign of a room below its stop; that room is words now, so a minus is refused.)
NUMERIC = set("0123456789.%· ,")

#: A symbol that would render as markup if the cell ever carried it unescaped.
HOSTILE = "<b>WMT</b>"


def _every_stop_cell() -> list[str]:
    """The cell's content in every branch, under a symbol that is itself markup."""
    wmt = replace(WMT, symbol=HOSTILE)
    dear = replace(WMT, symbol=HOSTILE, entry_price=1250.00, stop_loss=1200.00)

    def stop(order_type: str, level: float | None) -> dict[str, app.OpenOrder]:
        return app.live_stops([app.OpenOrder(HOSTILE, "sell", order_type, level)])

    rows = [
        _row(stop("stop", 104.00), holding=wmt),  # room
        _row(stop("stop", 104.00), price=103.00, holding=wmt),  # through the stop
        _row(stop("stop", 1187.50), price=1230.00, holding=dear),  # thousands
        _row(stop("stop", 109.00), holding=wmt),  # above the entry and the price
        _row(stop("stop", 109.00), price=110.00, holding=wmt),  # above the entry only
        _row(stop("trailing_stop", 104.50), holding=wmt),
        _row(stop("trailing_stop", None), holding=wmt),
        _row({}, holding=wmt),
        _row({}, holding=dear),
        _row(None, holding=wmt),
        app.position_rows(Book(), {HOSTILE: 1.0}, {HOSTILE: 107.91}, {})[0],
    ]
    return [
        _cell(app.position_table([row], a_verdict()), "STOP ROOM")[1] for row in rows
    ]


def test_the_stop_cell_carries_nothing_but_its_own_words_and_numbers() -> None:
    """**The compensating test for ``raw=(5, 6)``.** The STOP ROOM column is not escaped,
    so its content is held to an allowlist: remove the fixed fragments and what remains
    must be digits and separators. A symbol, a book value or any other string reaching
    the cell unescaped leaves letters or markup behind and fails here."""
    cells = _every_stop_cell()

    leftovers = []
    for cell in cells:
        rest = cell
        for fragment in sorted(FIXED, key=len, reverse=True):
            rest = rest.replace(fragment, "")
        if not set(rest) <= NUMERIC:
            leftovers.append(f"{cell!r} leaves {sorted(set(rest) - NUMERIC)}")

    assert not leftovers, "; ".join(leftovers)
    # Not vacuous: every branch was rendered, so every fragment was actually exercised.
    for fragment in FIXED:
        assert any(
            fragment in cell for cell in cells
        ), f"no branch rendered {fragment!r}"
