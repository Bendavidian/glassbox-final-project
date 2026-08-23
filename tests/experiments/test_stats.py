"""GB-51 acceptance: paired Wilcoxon, three references, and the family counted.

What is tested here is everything that can be wrong while every number still looks
plausible: a metric paired against the wrong reference, a pairing that crosses a fold or a
condition, a correction that is not monotone, and a p-value computed on so few folds that
it could never have been significant.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from glassbox.experiments import stats, study


def a_frame(**overrides) -> pd.DataFrame:
    """One condition, three arms plus the market reference, sixteen folds each."""
    rng = np.random.default_rng(11)
    rows = []
    for model, channels in (
        ("persistence", "C0_base"),
        ("dlinear", "C0_base"),
        ("fits", "C0_base"),
        (study.BUY_AND_HOLD, ""),
    ):
        for fold in range(16):
            row = dict.fromkeys(study.COLUMNS, math.nan)
            row.update(
                {
                    "anchor": 0,
                    "lr": 1e-3,
                    "control": study.REAL,
                    "cutoff_period_days": 5,
                    "model": model,
                    "channels": channels,
                    "fold": fold,
                    "skipped": False,
                    "direction_reference": 0.55,
                    "direction": 0.50 + rng.normal(0, 0.01),
                    "mae": 0.015 + rng.normal(0, 0.001),
                    "sharpe": rng.normal(0, 1.0),
                    "total_return": rng.normal(0, 0.02),
                }
            )
            if model == "persistence":
                row["direction"] = math.nan
                row["sharpe"] = math.nan
            if model == study.BUY_AND_HOLD:
                row["direction"] = row["direction_reference"]
                row["mae"] = math.nan
            row.update(overrides)
            rows.append(row)
    return pd.DataFrame(rows, columns=list(study.COLUMNS))


# ── the three-reference rule ─────────────────────────────────────────────────


def test_each_metric_carries_its_own_reference() -> None:
    """**The correction to spec 7.4**, expressed as data rather than as prose.

    7.4 asked for every arm against persistence. Persistence forecasts zero and takes no
    trades, so its direction and Sharpe cells are empty: a single reference would test two
    of the four metrics against a NaN.
    """
    assert stats.REFERENCES["mae"].label == stats.PERSISTENCE
    assert stats.REFERENCES["direction"].label == stats.ALWAYS_LONG
    assert stats.REFERENCES["sharpe"].label == study.BUY_AND_HOLD
    assert stats.REFERENCES["total_return"].label == study.BUY_AND_HOLD
    assert len({r.label for r in stats.REFERENCES.values()}) == 3


def test_direction_is_paired_against_the_fold_s_own_up_rate() -> None:
    """Always-long is a **per-fold** number, not 0.5 and not another arm's row."""
    frame = a_frame()

    table = stats.wilcoxon(frame, metrics=("direction",))
    row = table[table["model"] == "dlinear"].iloc[0]

    assert row["reference"] == stats.ALWAYS_LONG
    assert row["reference_mean"] == pytest.approx(0.55)
    assert row["n_folds"] == 16


def test_an_arm_is_never_tested_against_itself() -> None:
    """Persistence against persistence on MAE is a row of zeros with a p-value."""
    table = stats.wilcoxon(a_frame())

    persistence = table[table["model"] == "persistence"]
    assert "mae" not in set(persistence["metric"])
    assert set(persistence["metric"]) == {"total_return"}


def test_the_market_reference_takes_no_tests() -> None:
    """Its direction **is** the always-long bar by construction; the test is a tautology."""
    table = stats.wilcoxon(a_frame())

    assert study.BUY_AND_HOLD not in set(table["model"])


def test_mae_pairs_against_persistence_on_the_same_channels() -> None:
    frame = a_frame()

    table = stats.wilcoxon(frame, metrics=("mae",))
    row = table[table["model"] == "fits"].iloc[0]

    baseline = frame[(frame["model"] == "persistence")]["mae"]
    assert row["reference"] == stats.PERSISTENCE
    assert row["reference_mean"] == pytest.approx(baseline.mean())


# ── the pairing ──────────────────────────────────────────────────────────────


def test_pairing_is_by_fold_and_not_by_row_order() -> None:
    """A results file is not required to be sorted, and a positional pair is silent."""
    frame = a_frame()
    shuffled = frame.sample(frac=1.0, random_state=5).reset_index(drop=True)

    ordered = stats.wilcoxon(frame, metrics=("mae",))
    scrambled = stats.wilcoxon(shuffled, metrics=("mae",))

    key = ["model", "channels", "metric"]
    pd.testing.assert_frame_equal(
        ordered.sort_values(key).reset_index(drop=True)[[*key, "p", "mean_delta"]],
        scrambled.sort_values(key).reset_index(drop=True)[[*key, "p", "mean_delta"]],
    )


def test_a_pairing_never_crosses_a_condition() -> None:
    """Fold 3 at anchor 21 is not fold 3 at anchor 0; pairing them compares periods."""
    first = a_frame()
    second = a_frame()
    second["anchor"] = 21
    second["sharpe"] = second["sharpe"] + 10.0

    table = stats.wilcoxon(pd.concat([first, second]), metrics=("sharpe",))

    assert set(table["anchor"]) == {0, 21}
    for anchor in (0, 21):
        assert (table[table["anchor"] == anchor]["n_folds"] == 16).all()
    # The +10 shift moved the arm and its reference together, so the delta is unchanged.
    deltas = table.groupby("anchor")["mean_delta"].first()
    assert deltas[0] == pytest.approx(deltas[21])


def test_a_fold_the_arm_could_not_score_is_dropped_from_its_pair() -> None:
    """A NaN Sharpe is an arm that did not trade enough, not a Sharpe of zero."""
    frame = a_frame()
    mask = (frame["model"] == "fits") & (frame["fold"] < 4)
    frame.loc[mask, "sharpe"] = math.nan

    table = stats.wilcoxon(frame, metrics=("sharpe",))

    assert table[table["model"] == "fits"].iloc[0]["n_folds"] == 12
    assert table[table["model"] == "dlinear"].iloc[0]["n_folds"] == 16


def test_a_skipped_cell_is_not_a_test() -> None:
    frame = a_frame()
    frame.loc[frame["model"] == "fits", "skipped"] = True

    table = stats.wilcoxon(frame)

    assert "fits" not in set(table["model"])


# ── the numbers ──────────────────────────────────────────────────────────────


def test_the_statistic_is_scipy_s_and_not_a_reimplementation() -> None:
    """The test itself is a library call; what this module owns is the pairing."""
    frame = a_frame()

    row = stats.wilcoxon(frame, metrics=("mae",))
    row = row[row["model"] == "dlinear"].iloc[0]

    arm = frame[(frame["model"] == "dlinear")].set_index("fold")["mae"]
    base = frame[(frame["model"] == "persistence")].set_index("fold")["mae"]
    expected = scipy_stats.wilcoxon(arm.to_numpy(), base.loc[arm.index].to_numpy())
    assert row["p"] == pytest.approx(expected.pvalue)
    assert row["statistic"] == pytest.approx(expected.statistic)


def test_too_few_moved_folds_gives_no_p_value_rather_than_a_powerless_one() -> None:
    """Below :data:`stats.MIN_FOLDS` the exact test cannot reach 0.05 at all."""
    frame = a_frame()
    arm = (frame["model"] == "dlinear") & (frame["fold"] >= stats.MIN_FOLDS - 2)
    base = (frame["model"] == "persistence") & (frame["fold"] >= stats.MIN_FOLDS - 2)
    frame.loc[arm, "mae"] = 0.02
    frame.loc[base, "mae"] = 0.02

    row = stats.wilcoxon(frame, metrics=("mae",))
    row = row[row["model"] == "dlinear"].iloc[0]

    assert math.isnan(row["p"])
    assert (
        row["n_folds"] == 16
    )  # the folds are still reported; only the test is withheld


def test_sixteen_folds_cannot_produce_a_p_below_the_exact_floor() -> None:
    """The floor the module docstring states, pinned rather than asserted in prose.

    No claim in this study can be significant past 3.05e-5 however large its effect, and a
    reader comparing p-values across studies with different fold counts needs to know it.
    """
    perfect = scipy_stats.wilcoxon(np.arange(1.0, 17.0), np.zeros(16))

    assert perfect.pvalue == pytest.approx(2 / 2**16)
    assert perfect.pvalue == pytest.approx(3.0517578125e-05)


# ── multiple comparisons ─────────────────────────────────────────────────────


def test_holm_steps_down_and_is_monotone() -> None:
    adjusted = stats.holm(np.array([0.01, 0.04, 0.03]))

    # 3x0.01, then max(that, 2x0.03), then max(that, 1x0.04).
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert (adjusted >= np.array([0.01, 0.04, 0.03])).all()


def test_holm_caps_at_one() -> None:
    assert stats.holm(np.array([0.6, 0.7])) == pytest.approx([1.0, 1.0])


def test_a_test_that_could_not_be_computed_does_not_enlarge_the_family() -> None:
    """A NaN is not a comparison anybody made, and charging for it would be a penalty
    paid for nothing."""
    with_nan = stats.holm(np.array([0.01, math.nan, 0.03]))
    without = stats.holm(np.array([0.01, 0.03]))

    assert math.isnan(with_nan[1])
    assert with_nan[[0, 2]] == pytest.approx(without)


def test_every_row_states_the_size_of_the_family_it_was_corrected_in() -> None:
    """**The family is every test in the call**, and it is on the row rather than in a
    footnote so a p-value cannot be quoted without it."""
    table = stats.wilcoxon(a_frame())

    assert (table["n_tests"] == len(table)).all()
    assert len(table) > 1


def test_the_family_narrows_when_the_caller_narrows_the_table() -> None:
    """Which is how a report corrects over the tests it actually shows."""
    frame = a_frame()
    second = a_frame()
    second["control"] = study.NOISE

    everything = stats.wilcoxon(pd.concat([frame, second]))
    real_only = stats.wilcoxon(study.reportable(pd.concat([frame, second])))

    assert real_only["n_tests"].iloc[0] < everything["n_tests"].iloc[0]
    assert set(real_only["control"]) == {study.REAL}


# ── the gate ─────────────────────────────────────────────────────────────────


def test_the_single_gate_works_on_this_table_unchanged() -> None:
    """One gate for both tables. A second filter is a second thing that can disagree."""
    frame = a_frame()
    noise = a_frame()
    noise["control"] = study.NOISE

    table = stats.wilcoxon(pd.concat([frame, noise]))
    kept = study.reportable(table)

    assert set(table["control"]) == {study.REAL, study.NOISE}
    assert set(kept["control"]) == {study.REAL}


def test_the_null_conditions_are_tested_too() -> None:
    """The standing requirement of 20 Aug: the effect size under the control is reported
    beside the real one, so it must be computed."""
    noise = a_frame()
    noise["control"] = study.NOISE

    table = stats.wilcoxon(noise)

    assert len(table) > 0
    assert set(table["control"]) == {study.NOISE}


def test_the_columns_are_the_declared_ones() -> None:
    assert list(stats.wilcoxon(a_frame()).columns) == list(stats.COLUMNS)
