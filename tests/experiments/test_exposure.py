"""The gross-exposure measurement: what it counts, and what it must not quietly drop.

These are unit tests over synthetic records. The expensive property - that instrumenting
changes no number - is asserted directly in ``tests/engine/test_room_detail.py`` and
proved end to end by the module itself, which runs every fold twice and aborts if the
metrics tables differ.
"""

from __future__ import annotations

import pandas as pd
import pytest

from glassbox.config.loader import load_config
from glassbox.engine import risk
from glassbox.experiments import exposure

BASE = load_config()


def a_record(fold: int, case: str, gross_fraction: float = 0.2) -> dict:
    return {
        "fold": fold,
        "arm": "dlinear",
        "call": 0,
        "equity": 100_000.0,
        "gross_exposure": 100_000.0 * gross_fraction,
        "gross_fraction": gross_fraction,
        "term_a_per_position": 10_000.0,
        "term_b_gross_headroom": 5_000.0,
        "term_c_cash": 80_000.0,
        "notional": 5_000.0,
        "case": case,
    }


def frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(exposure._COLUMNS))


# ── the defect this module had, kept as a guard ──────────────────────────────


def test_a_fold_that_made_no_sizing_call_stays_in_the_denominator() -> None:
    """**The headline is a fraction and a vanishing fold shrinks the wrong half.**

    A fold whose band stands aside proposes no entry, so the sizer is never called and the
    fold contributes no rows. Deriving the fold list from the records dropped it entirely,
    turning "bound in 1 of 16" into "1 of 1" - a fifteen-fold overstatement, reported with
    a straight face. `measure` returns the folds that RAN for exactly this reason.
    """
    records = frame([a_record(2, risk.REDUCED_BY_GROSS)])

    text = exposure.report(records, BASE, folds=(1, 2))

    assert "1 of 2" in text
    assert "folds that made no sizing call     : 1" in text
    assert "stood aside" in text


def test_a_silent_fold_is_never_reported_as_not_binding() -> None:
    """ "No calls" and "called, did not bind" are different facts."""
    records = frame([a_record(2, risk.UNCONSTRAINED)])

    lines = exposure.report(records, BASE, folds=(1, 2)).splitlines()
    fold_one = [ln for ln in lines if ln.strip().startswith("1 ")]

    assert fold_one, "fold 1 must appear in the per-fold table"
    assert "no" not in fold_one[0].split()  # not the word 'no' - a dash


# ── what counts as binding ───────────────────────────────────────────────────


def test_the_headline_counts_reduced_as_well_as_blocked() -> None:
    """The reducing case is the one a naive instrument misses; it must lead."""
    records = frame(
        [a_record(1, risk.REDUCED_BY_GROSS), a_record(2, risk.BLOCKED_BY_GROSS)]
    )

    text = exposure.report(records, BASE, folds=(1, 2))

    assert "2 of 2" in text


def test_a_fold_where_only_the_cap_reduced_still_counts_as_bound() -> None:
    records = frame([a_record(1, risk.REDUCED_BY_GROSS)])
    assert "1 of 1" in exposure.report(records, BASE, folds=(1,))


def test_blocked_by_other_is_not_the_risk_layer_binding() -> None:
    """A or C blocking is cash or the per-position cap, not the gross cap."""
    records = frame([a_record(1, risk.BLOCKED_BY_OTHER)])

    text = exposure.report(records, BASE, folds=(1,))

    assert "0 of 1" in text
    assert "NOT the risk layer" in text


def test_a_tie_is_reported_separately_and_does_not_swing_the_headline() -> None:
    """The tie policy is stated, not silently applied - the count must be visible."""
    records = frame([a_record(1, risk.TIE)])

    text = exposure.report(records, BASE, folds=(1,))

    assert "0 of 1" in text  # TIE_COUNTS_AS_BINDING is False
    assert "TIE               (A == B exactly) : 1" in text
    assert "TIE POLICY" in text


def test_no_equity_is_counted_and_is_not_a_cap_event() -> None:
    records = frame([a_record(1, risk.NO_EQUITY)])

    text = exposure.report(records, BASE, folds=(1,))

    assert "0 of 1" in text
    assert "NO_EQUITY         (early return)   : 1" in text


# ── provenance travels with the number ───────────────────────────────────────


def test_the_method_is_in_the_output_not_only_in_a_docstring() -> None:
    """The whole reason this module exists rather than a scratchpad.

    DECISIONS.md's 16.00% / peak-51% figures have no producing code. A number that arrives
    without its method repeats that, so the method is asserted to be present.
    """
    text = exposure.report(frame([a_record(1, risk.UNCONSTRAINED)]), BASE, folds=(1,))

    for expected in (
        "METHOD",
        "max_position_pct",
        "max_gross_exposure",
        "TIE POLICY",
        "run TWICE",
        "lower bound",
        "NOT fixed here",  # room_for's discarded term is scoped separately, and says so
    ):
        assert expected in text, f"the method statement lost {expected!r}"


def test_the_report_states_the_concurrent_hold_ceiling_against_the_cap() -> None:
    """The mechanism is concurrent holds and duration, not entries per bar."""
    text = exposure.report(frame([a_record(1, risk.UNCONSTRAINED)]), BASE, folds=(1,))

    ceiling = len(BASE.universe) * BASE.risk.max_position_pct

    assert "ACCUMULATION" in text
    assert f"{ceiling:.2f}" in text
    assert "not entries" in text


def test_a_peak_above_the_cap_is_explained_rather_than_left_to_look_like_a_breach() -> (
    None
):
    """Gross is marked value; the cap bounds entries. A rise carries gross above it."""
    records = frame([a_record(1, risk.UNCONSTRAINED, gross_fraction=0.5057)])

    text = exposure.report(records, BASE, folds=(1,))

    assert "EXCEEDS" in text
    assert "enforced at ENTRY" in text


# ── the comparison that gates the whole run ──────────────────────────────────


def test_identical_is_true_for_equal_tables_and_false_for_any_difference() -> None:
    left = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    assert exposure.identical(left, left.copy())

    moved = left.copy()
    moved.loc[0, "a"] = 1.0 + 1e-15
    assert not exposure.identical(left, moved)


def test_identical_treats_a_stood_aside_nan_as_agreement() -> None:
    """Both runs write NaN for an undefined Sharpe; that is agreement, not difference."""
    left = pd.DataFrame({"sharpe": [float("nan"), 1.0]})
    assert exposure.identical(left, left.copy())


@pytest.mark.parametrize(
    "case",
    [
        risk.UNCONSTRAINED,
        risk.REDUCED_BY_GROSS,
        risk.BLOCKED_BY_GROSS,
        risk.BLOCKED_BY_OTHER,
        risk.TIE,
        risk.NO_EQUITY,
    ],
)
def test_every_case_the_vocabulary_defines_can_be_reported(case: str) -> None:
    """A case the report cannot render is a case that would vanish from the counts."""
    exposure.report(frame([a_record(1, case)]), BASE, folds=(1,))
