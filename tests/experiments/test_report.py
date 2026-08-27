"""GB-52 acceptance: the report regenerates from results.csv and cannot omit its guards.

The report is the artefact a reader in October will actually hold, so what is tested here
is the three things that would let it mislead one quietly: an MAE column with no flatness
beside it, a claim shown at one anchor as though it held at three, and a stale data
snapshot that appears nowhere on the page.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from glassbox.backtest import metrics
from glassbox.experiments import report, stats, study


def a_results_file(snapshot: str = "2026-08-13") -> pd.DataFrame:
    """A results table with the shape the real grid writes: every axis, every control."""
    rng = np.random.default_rng(19)
    rows: list[dict] = []
    for condition in study.conditions():
        for model, channels in study.arms(condition.models):
            skipped = (model, channels) in study.SKIPPED
            for fold in [-1] if skipped else range(1, 17):
                row = dict.fromkeys(study.COLUMNS, math.nan)
                row.update(
                    {
                        # From the condition, never re-listed here: on 26 Aug 2026 this
                        # fixture named four axes by hand and `target_in_loop` was not one
                        # of them, so every synthetic row left it blank, the spoke and the
                        # centre became the same condition, and the paired tests saw 32
                        # folds where there are 16.
                        **condition.columns,
                        "model": model,
                        "channels": channels,
                        "fold": fold,
                        "skipped": skipped,
                        "reason": "spec 6.4" if skipped else "",
                        "n_trades": 6,
                        "data_snapshot_last_bar": snapshot,
                        "seconds": 1.0,
                    }
                )
                if not skipped:
                    row.update(
                        {
                            "direction": 0.50 + rng.normal(0, 0.02),
                            "direction_reference": 0.5560,
                            "mae": 0.0155 + rng.normal(0, 0.001),
                            "flatness": 0.8 + rng.normal(0, 0.05),
                            "sharpe": rng.normal(0, 1.0),
                            "total_return": rng.normal(0, 0.02),
                        }
                    )
                    if model == "fits":
                        row["cof"] = 120 // condition.cutoff
                        row["dead_row_fraction"] = condition.cutoff / 120
                    if model == "persistence":
                        row["direction"] = math.nan
                        row["sharpe"] = math.nan
                        row["total_return"] = 0.0
                rows.append(row)
        # The market reference, once per condition.
        for fold in range(1, 17):
            row = dict.fromkeys(study.COLUMNS, math.nan)
            row.update(
                {
                    "anchor": condition.anchor,
                    "lr": condition.lr,
                    "control": condition.control,
                    "cutoff_period_days": condition.cutoff,
                    "model": study.BUY_AND_HOLD,
                    "channels": "",
                    "fold": fold,
                    "skipped": False,
                    "reason": "",
                    "direction": 0.5560,
                    "direction_reference": 0.5560,
                    "total_return": 0.078,
                    "sharpe": 1.5,
                    "n_trades": 1,
                    "data_snapshot_last_bar": snapshot,
                    "seconds": 0.07,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=list(study.COLUMNS))


@pytest.fixture
def frame() -> pd.DataFrame:
    return a_results_file()


# ── the snapshot is on the page ──────────────────────────────────────────────


def test_the_data_snapshot_is_the_first_line_of_the_header(frame) -> None:
    """A report built on stale data says so on its own face, not only in a run log."""
    lines = report.header(frame)

    assert "2026-08-13" in lines[0]
    assert "snapshot" in lines[0].lower()


def test_two_snapshots_in_one_file_are_refused(frame) -> None:
    """Averaging across vintages would be a result about two different markets."""
    frame.loc[frame["anchor"] == study.ANCHORS[1], "data_snapshot_last_bar"] = (
        "2026-07-01"
    )

    with pytest.raises(ValueError, match="two different markets"):
        report.header(frame)


def test_the_snapshot_reaches_the_rendered_page(frame, tmp_path) -> None:
    path = report.render(frame, tmp_path, dpi=60)

    assert "2026-08-13" in path.read_text(encoding="utf-8")


# ── MAE never alone ──────────────────────────────────────────────────────────


def test_flatness_sits_immediately_beside_mae_in_the_summary(frame) -> None:
    """Structural: the order comes from `metrics.columns_for`, so printing MAE alone means
    deleting the pairing rather than forgetting it."""
    columns = list(report.summary(frame).columns)

    assert columns[columns.index("mae") + 1] == "flatness"
    assert metrics.COMPANIONS["mae"] == "flatness"


def test_the_rank_correlations_are_printed_with_the_table(frame, tmp_path) -> None:
    """§7.3 as amended: a reader shown MAE without them will read it as accuracy."""
    text = report.render(frame, tmp_path, dpi=60).read_text(encoding="utf-8")

    assert "Spearman(MAE, flatness)" in text
    assert "Spearman(MAE, direction)" in text


def test_the_correlations_are_measured_not_quoted(frame) -> None:
    ranks = report.correlations(frame)

    assert set(ranks) == {"across_arms", "across_arm_folds"}
    for level in ranks.values():
        assert set(level) == {"flatness", "direction"}
        assert all(-1.0 <= value <= 1.0 for value in level.values())


def test_both_aggregation_levels_are_reported(frame, tmp_path) -> None:
    """**The defect this pins, found 27 Aug 2026 while drafting GB-57.**

    The report printed one coefficient, computed over arm-folds, under a fixed sentence
    describing the arm-level relationship. On the real grid they are +0.109 and +1.000 —
    a number and a sentence disagreeing about what they describe, in the artefact the
    results chapter is generated from. Both levels are now printed and labelled.
    """
    text = report.render(frame, tmp_path, dpi=60).read_text(encoding="utf-8")

    assert "across the" in text and "arms" in text
    assert "pooled over the" in text
    assert "arm-level" in text


def test_the_prose_never_claims_more_than_the_coefficient(frame, tmp_path) -> None:
    """The mechanism, not the fix: the wording is derived from the number.

    A fixed string cannot be wrong about a number it never reads. Deriving it means a
    weakening relationship weakens the sentence with it, and nobody has to notice.
    """
    text = report.render(frame, tmp_path, dpi=60).read_text(encoding="utf-8")
    rho = report.correlations(frame)["across_arms"]["flatness"]

    assert report.describe_correlation(rho) in text
    if abs(rho) < report._MONOTONE:
        assert "close to a monotone" not in text


@pytest.mark.parametrize(
    ("rho", "expected"),
    [
        (1.0, "close to a monotone increasing function"),
        (-0.95, "close to a monotone decreasing function"),
        (0.8, "strongly related but not monotone"),
        (0.5, "moderately related"),
        (0.1, "weakly related at best"),
        (float("nan"), "not computable here"),
    ],
)
def test_the_wording_follows_the_coefficient(rho: float, expected: str) -> None:
    assert report.describe_correlation(rho) == expected


def test_the_significance_floor_is_derived_from_the_folds_present(
    frame, tmp_path
) -> None:
    """`3.05e-5` was written down beside "16 folds", both as literals. The exact
    two-sided Wilcoxon floor is `2**(1-n)`, so a grid with a different fold count would
    have printed a bound that was simply false."""
    text = report.render(frame, tmp_path, dpi=60).read_text(encoding="utf-8")
    folds = int(stats.wilcoxon(study.reportable(frame))["n_folds"].max())

    assert f"on {folds} folds" in text
    assert f"{2.0 ** (1 - folds):.3g}" in text


# ── every metric against its own reference ───────────────────────────────────


def test_each_metric_is_reported_against_its_own_reference(frame) -> None:
    """The three-reference rule, sourced from `stats.REFERENCES` rather than restated."""
    table = report.summary(frame)

    for metric in report.SUMMARY_METRICS:
        expected = stats.REFERENCES[metric].label
        assert set(table[f"{metric}_reference"]) == {expected}


def test_the_summary_holds_the_other_axes_at_their_reference(frame) -> None:
    """Averaging over three learning rates would report the mean of an axis as a result,
    and the MAE column is conditional on the rate (§7 (1n))."""
    table = report.summary(frame)

    # Three anchors, one row per arm at each - not one row per arm per rate per cutoff.
    assert set(table["anchor"]) == set(study.ANCHORS)
    assert len(table) == len(study.ANCHORS) * (len(study.live_arms()) + 1)


def test_the_summary_carries_the_corrected_p_beside_the_raw_one(frame) -> None:
    table = report.summary(frame)

    for metric in report.SUMMARY_METRICS:
        assert f"{metric}_p" in table.columns
        assert f"{metric}_p_holm" in table.columns


# ── every claim at every anchor and against both nulls ───────────────────────


def test_every_claim_is_shown_at_all_three_anchors(frame) -> None:
    """A claim shown at one anchor is a property of that anchor (§7.4)."""
    for claim in report.CLAIMS:
        table = report.claim_table(frame, claim)
        assert list(table["anchor"]) == sorted(study.ANCHORS)


def test_every_claim_is_shown_against_both_null_conditions(frame) -> None:
    """The standing requirement of 20 Aug: the effect under the control beside the real
    one, because grid sensitivity and a null control catch different failures."""
    for claim in report.CLAIMS:
        table = report.claim_table(frame, claim)
        assert set(study.CONTROLS) <= set(table.columns)
        assert table[study.NOISE].notna().any()


def test_the_claims_reach_the_rendered_page(frame, tmp_path) -> None:
    text = report.render(frame, tmp_path, dpi=60).read_text(encoding="utf-8")

    for claim in report.CLAIMS:
        assert claim.name in text


# ── the COF sweep ────────────────────────────────────────────────────────────


def test_the_cof_curve_has_one_row_per_cutoff_with_its_geometry(frame) -> None:
    """A cutoff on its own is a number nobody can interpret."""
    curve = report.cof_curve(frame)

    assert list(curve["cutoff_period_days"]) == sorted(study.CUTOFFS)
    assert (curve["cof"] == 120 // curve["cutoff_period_days"]).all()
    assert curve["dead_row_fraction"].notna().all()


def test_the_cof_curve_carries_each_cutoffs_own_null_control(frame) -> None:
    """Testing only the centre would leave three of the four unfalsifiable."""
    curve = report.cof_curve(frame)

    assert curve["noise_direction"].notna().all()
    assert curve["real_minus_noise_points"].notna().all()


# ── regenerating from the file alone ─────────────────────────────────────────


def test_the_report_regenerates_from_the_csv_and_nothing_else(frame, tmp_path) -> None:
    """No cache, no checkpoint, no rerun: a figure must be rebuildable in October."""
    csv = tmp_path / "results.csv"
    frame.to_csv(csv, index=False)

    assert (
        report.main(["--results", str(csv), "--out", str(tmp_path), "--dpi", "60"]) == 0
    )
    assert (tmp_path / report.REPORT_FILE).is_file()


def test_the_figures_are_written(frame, tmp_path) -> None:
    boxes = report.boxplots(frame, tmp_path / "boxes.png", dpi=60)
    curve = report.cof_plot(frame, tmp_path / "cof.png", dpi=60)

    assert boxes.read_bytes().startswith(b"\x89PNG")
    assert curve.read_bytes().startswith(b"\x89PNG")


def test_a_file_from_an_older_study_is_refused_by_name(frame, tmp_path) -> None:
    """Silently partial is the failure mode: every missing column is a missing guard."""
    csv = tmp_path / "old.csv"
    frame.drop(columns=["flatness", "cof"]).to_csv(csv, index=False)

    with pytest.raises(ValueError, match="flatness"):
        report.load(csv)


def test_a_missing_number_renders_as_an_empty_cell_not_as_nan(frame) -> None:
    """`nan` in a results table reads as a value somebody computed."""
    rendered = report._md(pd.DataFrame([{"a": 1.0, "b": math.nan}]))

    assert "nan" not in rendered
    assert "|  |" in rendered
