"""The gross-cap observation channel: it must see everything and change nothing.

``room_detail`` exists so ``experiments.exposure`` can answer "did the gross cap bind, and
in how many folds" without altering the sizing it measures. Two properties carry that, and
the first is the one that would invalidate every number if it failed:

1. **``room_for`` still returns exactly what it returned**, for every input, bit for bit.
   It is now derived from ``room_detail`` rather than computing its own ``min``, which
   removes the second copy - but a derivation that changes a value is not a refactor.
2. **``recording_sizer`` returns what ``position_sizer`` returns.** A sizer that observed
   by deciding differently would make the measurement a description of itself.

The case vocabulary is tested by construction rather than by example where possible: the
tie is at an exact, derivable point, and the test computes it from the config rather than
hardcoding 0.40.
"""

from __future__ import annotations

import math
from dataclasses import replace

from hypothesis import given, settings
from hypothesis import strategies as st

from glassbox.config.loader import load_config
from glassbox.contracts.schemas import Signal
from glassbox.engine import risk
from glassbox.engine.signal import ENTER_LONG

BASE = load_config()

MONEY = st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False)


def a_signal() -> Signal:
    return Signal(
        symbol="AAPL",
        action=ENTER_LONG,
        trend_strength=0.01,
        up_points=3,
        passed_threshold=True,
    )


# ── 1. the refactor moved no number ──────────────────────────────────────────


def _room_for_as_it_was(equity: float, gross_exposure: float, cfg) -> float:
    """`room_for`'s body as of da37401, before it was derived from `room_detail`.

    Kept here deliberately as the thing the new implementation is compared against. This
    is the one place in the project where a second copy of an expression is correct: its
    entire purpose is to disagree if the derivation drifts.
    """
    if not (equity > 0.0):
        return 0.0
    return max(
        0.0,
        min(
            equity * cfg.risk.max_position_pct,
            equity * cfg.risk.max_gross_exposure - gross_exposure,
            equity - gross_exposure,
        ),
    )


@given(equity=MONEY, gross=MONEY)
@settings(max_examples=500, deadline=None)
def test_room_for_returns_exactly_what_it_returned_before(
    equity: float, gross: float
) -> None:
    """Bit-identical, not merely close. `==` on floats is the assertion that matters."""
    assert risk.room_for(equity, gross, BASE) == _room_for_as_it_was(
        equity, gross, BASE
    )


@given(
    equity=MONEY,
    gross=MONEY,
    position_pct=st.floats(min_value=0.001, max_value=1.0),
    gross_pct=st.floats(min_value=0.001, max_value=1.0),
)
@settings(max_examples=300, deadline=None)
def test_the_identity_holds_under_other_configurations_too(
    equity: float, gross: float, position_pct: float, gross_pct: float
) -> None:
    """The configured 0.10/0.50 is not what makes the derivation correct."""
    cfg = replace(
        BASE,
        risk=replace(
            BASE.risk, max_position_pct=position_pct, max_gross_exposure=gross_pct
        ),
    )
    assert risk.room_for(equity, gross, cfg) == _room_for_as_it_was(equity, gross, cfg)


@given(equity=MONEY, gross=MONEY)
@settings(max_examples=300, deadline=None)
def test_the_detail_carries_the_value_room_for_returns(
    equity: float, gross: float
) -> None:
    assert risk.room_detail(equity, gross, BASE).notional == risk.room_for(
        equity, gross, BASE
    )


@given(equity=MONEY, gross=MONEY)
@settings(max_examples=300, deadline=None)
def test_the_recording_sizer_decides_what_the_real_sizer_decides(
    equity: float, gross: float
) -> None:
    """Observation only. If this fails, every count in `experiments.exposure` is void."""
    seen: list[risk.RoomDetail] = []
    watched = risk.recording_sizer(seen)
    assert watched(a_signal(), equity, gross, BASE) == risk.position_sizer(
        a_signal(), equity, gross, BASE
    )


def test_the_recording_sizer_records_one_row_per_call() -> None:
    seen: list[risk.RoomDetail] = []
    watched = risk.recording_sizer(seen)
    for _ in range(3):
        watched(a_signal(), 100_000.0, 0.0, BASE)
    assert len(seen) == 3
    assert {d.case for d in seen} == {risk.UNCONSTRAINED}


def test_two_recorders_do_not_share_state() -> None:
    """The list is the caller's, so a second measurement cannot pollute the first."""
    left: list[risk.RoomDetail] = []
    right: list[risk.RoomDetail] = []
    risk.recording_sizer(left)(a_signal(), 100_000.0, 0.0, BASE)
    assert len(left) == 1
    assert right == []


# ── 2. the case vocabulary ───────────────────────────────────────────────────


def test_a_flat_account_is_unconstrained_by_the_gross_cap() -> None:
    detail = risk.room_detail(100_000.0, 0.0, BASE)
    assert detail.case == risk.UNCONSTRAINED
    assert not detail.gross_bound
    assert detail.notional == 100_000.0 * BASE.risk.max_position_pct


def test_the_gross_cap_reduces_before_it_blocks_and_the_notional_stays_positive() -> (
    None
):
    """**The case a naive instrument misses.** A reduced entry looks like a normal trade."""
    equity = 100_000.0
    gross = equity * (BASE.risk.max_gross_exposure - 0.05)  # 5% of headroom left

    detail = risk.room_detail(equity, gross, BASE)

    assert detail.case == risk.REDUCED_BY_GROSS
    assert detail.gross_bound
    assert detail.notional > 0.0
    assert detail.notional < equity * BASE.risk.max_position_pct


def test_the_gross_cap_blocks_when_the_headroom_is_gone() -> None:
    equity = 100_000.0
    detail = risk.room_detail(equity, equity * BASE.risk.max_gross_exposure, BASE)
    assert detail.case == risk.BLOCKED_BY_GROSS
    assert detail.gross_bound
    assert detail.notional == 0.0


def test_the_tie_sits_where_the_arithmetic_says_and_is_not_resolved() -> None:
    """`A == B` at `gross == (max_gross - max_position) * equity`, derived not hardcoded.

    Under the configured 0.10 and 0.50 that is 0.40 - four full-size positions - so every
    fifth full-size entry lands on it exactly. A tie-break chosen inside `room_detail`
    would decide the headline count by itself; it reports TIE and keeps all three terms.
    """
    equity = 100_000.0
    boundary = BASE.risk.max_gross_exposure - BASE.risk.max_position_pct
    detail = risk.room_detail(equity, equity * boundary, BASE)

    assert detail.case == risk.TIE
    assert not detail.gross_bound  # the policy lives in the caller, not here
    assert detail.per_position == detail.gross_headroom


def test_a_non_positive_equity_returns_before_any_term_is_computed() -> None:
    detail = risk.room_detail(0.0, 0.0, BASE)
    assert detail.case == risk.NO_EQUITY
    assert detail.notional == 0.0
    assert math.isnan(detail.per_position)
    assert math.isnan(detail.gross_headroom)
    assert math.isnan(detail.cash)


def test_cash_can_block_without_the_gross_cap_being_responsible() -> None:
    """BLOCKED_BY_OTHER is not the risk layer binding and must never be counted as it."""
    cfg = replace(
        BASE, risk=replace(BASE.risk, max_position_pct=1.0, max_gross_exposure=1.0)
    )
    detail = risk.room_detail(100_000.0, 100_000.0, cfg)
    assert detail.case == risk.BLOCKED_BY_OTHER
    assert not detail.gross_bound
    assert detail.notional == 0.0


@given(equity=MONEY, gross=MONEY)
@settings(max_examples=500, deadline=None)
def test_every_call_lands_in_exactly_one_case(equity: float, gross: float) -> None:
    cases = {
        risk.UNCONSTRAINED,
        risk.REDUCED_BY_GROSS,
        risk.BLOCKED_BY_GROSS,
        risk.BLOCKED_BY_OTHER,
        risk.TIE,
        risk.NO_EQUITY,
    }
    assert risk.room_detail(equity, gross, BASE).case in cases


@given(equity=MONEY, gross=MONEY)
@settings(max_examples=500, deadline=None)
def test_gross_bound_implies_the_headroom_was_the_minimum(
    equity: float, gross: float
) -> None:
    """The headline predicate cannot fire on a call the gross term did not decide."""
    detail = risk.room_detail(equity, gross, BASE)
    if detail.gross_bound:
        assert detail.gross_headroom < detail.per_position
        assert detail.gross_headroom < detail.cash
