"""GB-21 acceptance: no configuration produces an over-limit position.

The caps are stated as properties rather than examples, because the failure they guard
against is not "this case is wrong" but "some combination of equity, prices, ranking and
config is wrong". Hypothesis searches that space; a table of hand-picked cases cannot.

The strongest test here is not an inequality of my own. It is
``engine._require_sizeable`` - the backtester's own guard, written in GB-18 before this
module existed, precisely so the engine would hold GB-21 to its promise rather than trust
it. Running that guard over generated inputs turns "these two agree" into "they cannot
disagree".
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from glassbox.backtest import engine
from glassbox.config.loader import Config, load_config
from glassbox.contracts.schemas import Signal
from glassbox.engine import risk
from glassbox.engine.signal import ENTER_LONG, EXIT, HOLD

BASE = load_config()
SYMBOLS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL")


def within(value: float, limit: float) -> bool:
    """``value <= limit``, allowing for the float round trip through a share count.

    A notional becomes a share count and back (``shares * price``), which is exact in real
    arithmetic and not in binary. The slack is **relative** because Hypothesis generates
    accounts up to 1e9, where a fixed 1e-9 is smaller than one ulp; at that size the
    allowance is a ten-thousandth of a cent.
    """
    return value <= limit + max(1e-9, abs(limit) * 1e-12)


def signal(symbol: str, action: str = ENTER_LONG) -> Signal:
    return Signal(
        symbol=symbol,
        action=action,
        trend_strength=0.05,
        up_points=4,
        passed_threshold=True,
    )


equities = st.floats(
    min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False
)
prices = st.floats(min_value=0.01, max_value=1e5, allow_nan=False, allow_infinity=False)
fractions = st.floats(min_value=0.001, max_value=1.0, allow_nan=False)


@st.composite
def accounts(draw) -> tuple[Config, float, float, dict[str, float], list[Signal]]:
    """A whole sizing situation: config, account, prices and a ranked signal list."""
    position_pct = draw(fractions)
    gross_pct = draw(fractions)
    assume(position_pct <= gross_pct)  # the loader enforces this; see test_config.py

    cfg = replace(
        BASE,
        risk=replace(
            BASE.risk, max_position_pct=position_pct, max_gross_exposure=gross_pct
        ),
    )
    equity = draw(equities)
    # Exposure can legitimately exceed the cap after prices move, so it is drawn against
    # equity rather than against the cap: the sizer must cope with a book already too big.
    exposure = draw(st.floats(min_value=0.0, max_value=max(equity, 1.0)))
    chosen = draw(st.lists(st.sampled_from(SYMBOLS), min_size=0, max_size=8))
    price_map = {symbol: draw(prices) for symbol in SYMBOLS}
    return cfg, equity, exposure, price_map, [signal(s) for s in chosen]


# ── the caps hold, whatever the inputs ───────────────────────────────────────


@given(accounts())
@settings(max_examples=400, deadline=None)
def test_no_position_exceeds_the_per_position_cap(account) -> None:
    cfg, equity, exposure, price_map, signals = account

    orders = risk.size_positions(signals, equity, price_map, cfg, exposure)

    cap = equity * cfg.risk.max_position_pct
    for order in orders:
        assert within(order.notional, cap)


@given(accounts())
@settings(max_examples=400, deadline=None)
def test_total_exposure_never_exceeds_the_gross_cap(account) -> None:
    """Counting what was already open. Sizing each signal in isolation would pass a
    per-position check and blow the gross one on the third order."""
    cfg, equity, exposure, price_map, signals = account

    orders = risk.size_positions(signals, equity, price_map, cfg, exposure)
    committed = exposure + sum(order.notional for order in orders)

    # An account already over the cap gets no orders at all rather than a negative one.
    assert within(committed, max(exposure, equity * cfg.risk.max_gross_exposure))


@given(accounts())
@settings(max_examples=400, deadline=None)
def test_the_account_can_always_pay_for_what_was_sized(account) -> None:
    """Cash, not a risk rule: the arithmetic one. `equity - gross_exposure` is what is
    left, and the live executor has no guard of its own to catch an overdraw."""
    cfg, equity, exposure, price_map, signals = account

    orders = risk.size_positions(signals, equity, price_map, cfg, exposure)

    assert within(sum(order.notional for order in orders), max(0.0, equity - exposure))


@given(accounts())
@settings(max_examples=400, deadline=None)
def test_the_engines_own_guard_cannot_fire_for_this_sizer(account) -> None:
    """The property GB-18 wrote the guard for, checked with the guard itself.

    `_require_sizeable` raises on a non-finite, negative or unaffordable notional. Running
    it over generated accounts makes "the sizer satisfies the engine" a proof rather than a
    claim that has to be re-argued whenever either side changes.
    """
    cfg, equity, exposure, _prices, _signals = account
    assume(equity >= exposure)  # in the engine, equity - exposure IS cash, so never < 0

    notional = risk.position_sizer(signal(SYMBOLS[0]), equity, exposure, cfg)

    engine._require_sizeable(
        notional, equity - exposure, risk.position_sizer, signal("X")
    )


@given(accounts())
@settings(max_examples=200, deadline=None)
def test_the_two_entry_points_agree_on_the_first_order(account) -> None:
    """`size_positions` and `position_sizer` are one arithmetic behind two shapes.

    If they could differ, the backtest and the live loop would size differently and every
    comparison between them would be measuring the difference.
    """
    cfg, equity, exposure, price_map, _ = account
    symbol = SYMBOLS[0]

    batched = risk.size_positions([signal(symbol)], equity, price_map, cfg, exposure)
    single = risk.position_sizer(signal(symbol), equity, exposure, cfg)

    if batched:
        # The batched form went through the share conversion, so it is at most the raw
        # notional and short of it by less than the broker's quantity resolution — one unit
        # of the ninth decimal place, priced. Above ~1e7 shares a float cannot represent
        # nine decimals at all, so the allowance is the LARGER of the broker's resolution
        # and the float's; Hypothesis reaches that regime with a $0.01 price and a $21M
        # account, which no real universe contains but the property must still hold in.
        resolution = batched[0].price / 10.0**risk.QUANTITY_DECIMALS
        slack = max(resolution, abs(single) * 1e-12) + 1e-9
        assert within(batched[0].notional, single)
        assert single - batched[0].notional < slack
    else:
        assert risk.shares_for(single, price_map[symbol]) == 0.0


# ── the edges, named rather than left to the generator ───────────────────────


def test_zero_equity_produces_no_orders_rather_than_raising() -> None:
    """A blown or empty account is a state, not an exception the live loop must catch."""
    orders = risk.size_positions(
        [signal("AAPL")], 0.0, {"AAPL": 100.0}, BASE, gross_exposure=0.0
    )

    assert orders == []
    assert risk.position_sizer(signal("AAPL"), 0.0, 0.0, BASE) == 0.0


def test_a_negative_equity_produces_no_orders() -> None:
    assert risk.size_positions([signal("AAPL")], -5_000.0, {"AAPL": 100.0}, BASE) == []


def test_a_book_already_over_the_gross_cap_produces_no_orders() -> None:
    """Prices move; a book can be over the cap without anyone having broken a rule. The
    sizer's job is then to add nothing, not to reduce what is already held."""
    over = BASE.risk.max_gross_exposure * 100_000.0 + 1_000.0

    orders = risk.size_positions(
        [signal("AAPL")], 100_000.0, {"AAPL": 100.0}, BASE, gross_exposure=over
    )

    assert orders == []


def test_a_non_finite_equity_is_refused() -> None:
    with pytest.raises(ValueError, match="equity must be finite"):
        risk.size_positions([signal("AAPL")], math.nan, {"AAPL": 100.0}, BASE)


def test_a_non_positive_price_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="AAPL: price"):
        risk.size_positions([signal("AAPL")], 100_000.0, {"AAPL": 0.0}, BASE)


def test_a_symbol_with_no_price_gets_no_order() -> None:
    """The backtester's reading of a missing bar: a halt, not an invitation to guess."""
    orders = risk.size_positions(
        [signal("AAPL"), signal("MSFT")], 100_000.0, {"MSFT": 50.0}, BASE
    )

    assert [order.symbol for order in orders] == ["MSFT"]


# ── it may shrink or reject an order; it may never create one ────────────────


@given(accounts())
@settings(max_examples=200, deadline=None)
def test_every_order_traces_to_an_enter_long_signal(account) -> None:
    cfg, equity, exposure, price_map, signals = account

    orders = risk.size_positions(signals, equity, price_map, cfg, exposure)

    entries = {s.symbol for s in signals if s.action == ENTER_LONG}
    assert {order.symbol for order in orders} <= entries
    assert len(orders) == len({order.symbol for order in orders})


def test_hold_and_exit_signals_produce_nothing() -> None:
    """Sizing never opens a position the decision layer did not ask for."""
    quiet = [signal("AAPL", HOLD), signal("MSFT", EXIT)]

    assert (
        risk.size_positions(quiet, 100_000.0, {"AAPL": 10.0, "MSFT": 10.0}, BASE) == []
    )


def test_a_repeated_symbol_is_sized_once() -> None:
    """A duplicate would pyramid a position no risk rule asked for."""
    twice = [signal("AAPL"), signal("AAPL")]

    orders = risk.size_positions(twice, 100_000.0, {"AAPL": 100.0}, BASE)

    assert len(orders) == 1


def test_the_ranked_order_is_respected_when_the_gross_cap_binds() -> None:
    """First come, first sized: rank.py put the best signal first and it gets the room."""
    tight = replace(
        BASE,
        risk=replace(BASE.risk, max_position_pct=0.10, max_gross_exposure=0.25),
    )
    ranked = [signal(s) for s in ("NVDA", "AAPL", "MSFT", "AMZN")]
    price_map = dict.fromkeys(SYMBOLS, 100.0)

    orders = risk.size_positions(ranked, 100_000.0, price_map, tight)

    assert [order.symbol for order in orders] == ["NVDA", "AAPL", "MSFT"]
    assert orders[0].notional == pytest.approx(10_000.0)  # full allocation
    assert orders[1].notional == pytest.approx(10_000.0)
    assert orders[2].notional == pytest.approx(5_000.0)  # shrunk to the remaining room
    assert sum(o.notional for o in orders) == pytest.approx(25_000.0)


# ── the protective levels ────────────────────────────────────────────────────


@given(accounts())
@settings(max_examples=200, deadline=None)
def test_the_stop_is_below_entry_and_the_target_above(account) -> None:
    """Long-only, so there is a right side to each and the reference project got both
    wrong; see REFERENCE_AUDIT.md."""
    cfg, equity, exposure, price_map, signals = account

    for order in risk.size_positions(signals, equity, price_map, cfg, exposure):
        assert order.stop_loss < order.price
        assert order.take_profit > order.price


def test_the_levels_come_from_the_config_not_from_a_constant() -> None:
    orders = risk.size_positions([signal("AAPL")], 100_000.0, {"AAPL": 200.0}, BASE)
    order = orders[0]

    assert order.stop_loss == pytest.approx(200.0 * (1 - BASE.risk.stop_loss_pct))
    assert order.take_profit == pytest.approx(200.0 * (1 + BASE.risk.take_profit_pct))


def test_an_order_below_the_brokers_minimum_is_dropped_rather_than_rounded() -> None:
    """`shares_for` returns 0 below MIN_SHARES; the broker would reject it, so filling it
    in a backtest would invent a trade that cannot happen."""
    orders = risk.size_positions([signal("AAPL")], 0.01, {"AAPL": 1e5}, BASE)

    assert orders == []
