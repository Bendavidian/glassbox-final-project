"""GB-49 acceptance: one command, every axis a column, and the null arms gated.

The grid itself is exercised end to end on one fold when the cache is present. What is
tested unconditionally is everything that can go wrong silently: a skipped cell recorded as
blank instead of as a decision, a null-control row averaged into a result, a column pairing
that drifts, and a "deterministic" run that is not.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glassbox.backtest import metrics
from glassbox.config.loader import (
    MODEL_SHAPING_SECTIONS,
    Config,
    config_hash,
    load_config,
    model_config_hash,
)
from glassbox.data.historical import LOG_RETURN
from glassbox.experiments import study
from glassbox.model import fits
from glassbox.model.fits import FITSForecaster


@pytest.fixture
def cfg() -> Config:
    return load_config()


def cache_ready(cfg: Config, repo_root: Path) -> bool:
    cache = repo_root / cfg.data.cache_dir
    return all((cache / f"{symbol}.parquet").is_file() for symbol in cfg.universe)


def synthetic_bars(n: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.013, n)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype("float64"),
            LOG_RETURN: np.concatenate([[np.nan], np.diff(np.log(close))]),
            "source": pd.Series("yfinance", index=index, dtype="string"),
        },
        index=index,
    )


# ── the design ───────────────────────────────────────────────────────────────


def test_the_star_departs_from_one_reference_on_every_axis() -> None:
    """A centre plus one departure per axis, and the COF spokes carry their control.

    The full cross buys interactions nobody asked about at seven times the wall clock; a
    star gives every axis a **common reference to depart from**, which is what a
    sensitivity analysis needs.
    """
    design = study.conditions()

    assert sum(condition.is_reference for condition in design) == 1
    assert {c.anchor for c in design} == set(study.ANCHORS)
    assert {c.lr for c in design} == set(study.LEARNING_RATES)
    assert {c.control for c in design} == set(study.CONTROLS)
    assert {c.cutoff for c in design} == set(study.CUTOFFS)

    centre = design[0]
    for condition in design[1:]:
        moved = {
            axis
            for axis, differs in (
                ("anchor", condition.anchor != centre.anchor),
                ("lr", condition.lr != centre.lr),
                ("control", condition.control != centre.control),
                ("cutoff", condition.cutoff != centre.cutoff),
            )
            if differs
        }
        # One axis, except a COF spoke, which departs on its cutoff **and** carries its
        # own null control - ruled 23 Aug, because a cutoff tested only on real data is a
        # cutoff nobody can falsify.
        assert moved == {"cutoff"} or moved == {"cutoff", "control"} or len(moved) == 1


def test_every_cutoff_is_measured_against_its_own_null_control() -> None:
    """Not only the centre. `If FITS at cutoff 2 scores the same on white noise as on real
    data, that is the finding for that cutoff`."""
    design = study.conditions()

    for cutoff in study.CUTOFFS:
        controls = {c.control for c in design if c.cutoff == cutoff}
        assert study.REAL in controls
        assert study.NOISE in controls


def test_the_centre_of_the_cof_sweep_is_the_configured_cutoff(cfg: Config) -> None:
    """**The mechanism, not the convention.** The sweep departs from the deployed value,
    so the deployed value has to be the one the sweep starts at - and a constant that
    merely happens to equal the config today is the two-places defect waiting to happen.
    """
    assert study.CUTOFFS[0] == cfg.fits.cutoff_period_days


def test_a_cof_spoke_runs_fits_alone() -> None:
    """The cutoff reaches persistence and DLinear through nothing at all, so their rows
    at cutoff 20 would duplicate their rows at cutoff 5."""
    centre, spoke = (
        study.Condition(0, 1e-3, study.REAL, study.CUTOFFS[0]),
        study.Condition(0, 1e-3, study.REAL, study.CUTOFFS[1]),
    )

    assert centre.models == study.MODELS
    assert spoke.models == ("fits",)
    assert len(study.live_arms(spoke.models)) == 1


def test_the_full_cross_is_available_and_is_the_product() -> None:
    """For the day somebody does want an interaction."""
    assert len(study.conditions(full=True)) == (
        len(study.ANCHORS)
        * len(study.LEARNING_RATES)
        * len(study.CONTROLS)
        * len(study.CUTOFFS)
    )


def test_the_plan_is_reportable_before_the_run(cfg: Config) -> None:
    """A run of this size is a decision, so the size is available without making it."""
    star = study.plan(cfg)
    full = study.plan(cfg, full=True)

    assert star.arms == 5  # six pairs at the centre, one deliberately skipped
    assert len(star.conditions) == 13  # 7 as of GB-49, plus six COF spokes
    assert star.cells == 7 * 5 + 6 * 1
    assert star.trainings == star.cells * cfg.walkforward.max_folds
    assert full.cells == 27 * 5 + 81 * 1
    assert star.seconds() < full.seconds()


def test_the_estimate_prices_each_arm_at_what_it_measured(cfg: Config) -> None:
    """A flat mean over the arms would misprice a FITS-only spoke by 3x.

    The measured costs span 60x - buy-and-hold 0.07s a fold against FITS 4.16s - so an
    estimate is only useful if it knows the **mix**, which is the whole difference between
    a centre condition and a COF spoke.
    """
    fits_only = study.Plan(
        conditions=(study.Condition(0, 1e-3, study.REAL, study.CUTOFFS[1]),), folds=1
    )

    expected = study.PER_FOLD_SECONDS["fits"] + study.PER_FOLD_SECONDS["buy_and_hold"]
    assert fits_only.seconds() == pytest.approx(expected)


def test_the_empty_cell_is_named_and_reasoned() -> None:
    """Spec §6.4's deliberate hole. A blank cell reads as a run that failed."""
    assert ("fits", "C2_hybrid") in study.SKIPPED
    assert "6.4" in study.SKIPPED[("fits", "C2_hybrid")]
    assert ("fits", "C2_hybrid") in study.arms()  # present, and skipped


# ── the null controls ────────────────────────────────────────────────────────


def test_a_shuffled_world_keeps_the_returns_and_loses_their_order() -> None:
    bars = synthetic_bars()

    shuffled = study.null_bars(bars, study.SHUFFLED, seed=1)

    real = bars[LOG_RETURN].dropna().to_numpy()
    fake = shuffled[LOG_RETURN].dropna().to_numpy()
    np.testing.assert_allclose(np.sort(real), np.sort(fake))
    assert not np.allclose(real, fake)


def test_a_noise_world_matches_the_variance_and_nothing_else(
    cfg: Config,
) -> None:
    bars = synthetic_bars()

    noisy = study.null_bars(bars, study.NOISE, seed=1)

    real = bars[LOG_RETURN].dropna().to_numpy()
    fake = noisy[LOG_RETURN].dropna().to_numpy()
    assert fake.std() == pytest.approx(real.std(), rel=0.1)
    assert not np.allclose(np.sort(real), np.sort(fake))


def test_the_prices_are_rebuilt_from_the_new_returns(cfg: Config) -> None:
    """**The backtester must trade the world the model was fitted in.**

    Perturbing the features alone would leave the equity curve describing the real market
    and the forecasts describing a synthetic one, and every trading metric would compare
    two different universes.
    """
    bars = synthetic_bars()

    noisy = study.null_bars(bars, study.NOISE, seed=1)

    rebuilt = np.log(noisy["close"] / noisy["close"].shift(1)).dropna().to_numpy()
    np.testing.assert_allclose(
        rebuilt, noisy[LOG_RETURN].dropna().to_numpy(), atol=1e-12
    )
    assert not np.allclose(noisy["close"], bars["close"])
    # The bar shape survives: an OHLC frame that stopped being one would break the engine.
    assert (noisy["high"] >= noisy["close"]).all()
    assert (noisy["low"] <= noisy["close"]).all()


def test_the_real_control_is_the_frame_itself(cfg: Config) -> None:
    bars = synthetic_bars()

    assert study.null_bars(bars, study.REAL, seed=1) is bars


def test_the_same_seed_rebuilds_the_same_world() -> None:
    """'Rerunning with the same seed reproduces identical numbers' starts here."""
    bars = synthetic_bars()

    first = study.null_bars(bars, study.SHUFFLED, seed=99)
    second = study.null_bars(bars, study.SHUFFLED, seed=99)
    other = study.null_bars(bars, study.SHUFFLED, seed=100)

    pd.testing.assert_frame_equal(first, second)
    assert not np.allclose(first[LOG_RETURN].dropna(), other[LOG_RETURN].dropna())


def test_an_unknown_control_is_refused(cfg: Config) -> None:
    with pytest.raises(ValueError, match="unknown control"):
        study.null_bars(synthetic_bars(), "wishful", seed=1)


# ── BLAS threads ─────────────────────────────────────────────────────────────


def test_blas_is_pinned_by_default_and_the_cli_agrees() -> None:
    """**Measured to cost nothing, so it is the default rather than a flag.**

    Pinning removes thread count as a source of cross-machine divergence. It removes only
    that one - instruction sets, BLAS builds and `libm` remain - so this must never be
    reported as making reproduction architecture-independent.
    """
    import torch

    assert study.BLAS_THREADS == 1

    before = torch.get_num_threads()
    try:
        assert study.pin_threads() == before
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)


def test_running_the_grid_does_not_repin_the_caller_s_process() -> None:
    """`run` is a library call; `main` is the command. A function that silently repinned
    the process would be a side effect nobody asked for."""
    import inspect

    assert "pin_threads" not in inspect.getsource(study.run)


# ── provenance: was this file written by the configuration on disk? ─────────


def test_a_results_file_from_this_configuration_is_clean(
    cfg: Config, repo_root: Path
) -> None:
    """**The committed `results.csv` must match the committed `settings.yaml`.**

    A grid reads its configuration once at start, which makes the configuration mutable
    during a 20-minute run. On 23 Aug 2026 `settings.yaml` was edited at 17:00 while a
    grid started at 16:36 was still running, and the file it wrote carried a hash matching
    nothing on disk. Nothing in the system would have noticed; this is what notices.
    """
    frame = pd.read_csv(repo_root / study.RESULTS_FILE)

    stamp = study.provenance(frame, cfg)

    assert stamp.models_match, stamp.warning()
    assert stamp.config_matches, stamp.warning()
    assert stamp.ok


def test_a_live_only_change_warns_and_does_not_condemn(cfg: Config) -> None:
    """**The same split the checkpoint gate uses**, and for the same reason: a guard that
    invalidates a 20-minute grid over a polling interval is a guard somebody weakens."""
    frame = a_table()
    frame["model_config_hash"] = model_config_hash(cfg)
    frame["config_hash"] = config_hash(cfg)
    moved = replace(cfg, live=replace(cfg.live, poll_seconds=cfg.live.poll_seconds + 1))

    stamp = study.provenance(frame, moved)

    assert stamp.models_match
    assert not stamp.config_matches
    assert not stamp.ok
    assert "still stands" in stamp.warning()
    assert "live" in stamp.candidates


def test_a_model_shaping_change_condemns_the_file(cfg: Config) -> None:
    frame = a_table()
    frame["model_config_hash"] = model_config_hash(cfg)
    frame["config_hash"] = config_hash(cfg)
    moved = replace(cfg, window=replace(cfg.window, input_len=cfg.window.input_len + 1))

    stamp = study.provenance(frame, moved)

    assert not stamp.models_match
    assert "may be quoted" in stamp.warning()


def test_the_candidate_sections_exclude_everything_that_shapes_a_model() -> None:
    """A hash cannot say what changed; it can say what **cannot** have."""
    frame = a_table()
    frame["model_config_hash"] = model_config_hash(load_config())
    frame["config_hash"] = "not the current one"

    stamp = study.provenance(frame, load_config())

    assert stamp.candidates
    assert not set(stamp.candidates) & set(MODEL_SHAPING_SECTIONS)


# ── the gate ─────────────────────────────────────────────────────────────────


def a_table() -> pd.DataFrame:
    rows = []
    for control in study.CONTROLS:
        for skipped in (False, True):
            row = dict.fromkeys(study.COLUMNS, math.nan)
            row.update(
                {
                    "control": control,
                    "skipped": skipped,
                    "model": "fits",
                    "channels": "C0_base",
                    "direction": 0.9,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=list(study.COLUMNS))


def test_only_real_unskipped_rows_may_reach_a_metric() -> None:
    """**The single gate**, in ``records.is_reportable``'s shape and for its reason.

    A null arm measures the machine, not the market. Averaging one into a result produces
    a number that looks ordinary and means nothing, and filtering at each caller is how one
    of them eventually forgets.
    """
    table = a_table()

    kept = study.reportable(table)

    assert len(kept) == 1
    assert set(kept["control"]) == {study.REAL}
    assert not kept["skipped"].any()


def test_the_null_rows_are_in_the_same_file_as_the_real_ones() -> None:
    """They are only meaningful **beside** their real counterparts.

    Two artefacts can drift apart in columns or in vintage, and splitting them turns the
    comparison the controls exist for into a join. One file, one gate — the choice this
    project already made for replay and rehearsal provenance.
    """
    table = a_table()

    assert set(table["control"]) == set(study.CONTROLS)
    assert len(study.reportable(table)) < len(table)


# ── the columns ──────────────────────────────────────────────────────────────


def test_flatness_sits_immediately_beside_mae() -> None:
    """§7.3's pairing, in the file GB-52 reads. Structural, not editorial."""
    columns = list(study.COLUMNS)

    assert columns[columns.index("mae") + 1] == "flatness"


def test_the_cutoff_is_followed_by_what_it_decides() -> None:
    """`cutoff_period_days` alone is a number nobody can interpret. The retained bin count
    and the dead-row share sit beside it for the reason `flatness` sits beside `mae`."""
    columns = list(study.COLUMNS)

    assert columns[columns.index("cutoff_period_days") + 1] == "cof"
    assert columns[columns.index("cof") + 1] == "dead_row_fraction"


def test_the_geometry_is_derived_from_the_model_and_not_recomputed(cfg: Config) -> None:
    """`COF = L // cutoff` is exactly the arithmetic that gets written down twice."""
    length, horizon = cfg.window.input_len, cfg.window.horizon

    for cutoff in study.CUTOFFS:
        model = FITSForecaster(
            input_len=length,
            horizon=horizon,
            channels=("close_logret",),
            cutoff_period_days=cutoff,
        )
        assert fits.cof_for(length, cutoff) == model.cof
        assert fits.out_bins_for(length, horizon, cutoff) == model.out_bins
        # The dead row is `out_bins` complex weights of `COF x out_bins`.
        assert fits.dead_row_fraction(length, horizon, cutoff) == pytest.approx(
            model.out_bins * 2 / model.n_parameters
        )


def test_every_axis_and_every_required_column_is_present() -> None:
    """The five axes added after §7.4 was written, plus the snapshot vintage."""
    required = {
        "anchor",
        "lr",
        "control",
        "cutoff_period_days",
        "cof",
        "dead_row_fraction",
        "flatness",
        "cancellation",
        "data_snapshot_last_bar",
        "model",
        "channels",
        "fold",
        "direction",
        "mae",
    }

    assert required <= set(study.COLUMNS)


def test_the_metrics_named_in_the_columns_exist() -> None:
    """A column nobody can fill would print an empty study."""
    for name in ("mae", "flatness", "rmse", "direction_accuracy", "sharpe"):
        assert hasattr(metrics, name)


# ── end to end, when the cache is there ──────────────────────────────────────


def test_one_fold_of_the_whole_grid_runs(cfg: Config, repo_root: Path) -> None:
    """Cache → features → folds → train → calibrate → backtest → row, for every arm.

    One fold, because the suite must stay quick; the 16-fold run is reported in
    PROGRESS.md. What this asserts is that every cell produces a row and that the skipped
    cell produces one too.
    """
    if not cache_ready(cfg, repo_root):
        pytest.skip("no cached history; this test needs data_cache/")

    table = study.run(cfg, n_folds=1)

    assert list(table.columns) == list(study.COLUMNS)
    # Seven centre conditions at six arms plus the market reference, and six COF spokes at
    # two arms plus it, one fold each.
    assert len(table) == 7 * (len(study.arms()) + 1) + 6 * (
        len(study.arms(("fits",))) + 1
    )

    market = table[table["model"] == study.BUY_AND_HOLD]
    assert len(market) == 13  # once per condition, not once per arm
    assert market["mae"].isna().all()  # it makes no forecast, so the cell is empty
    assert market["total_return"].notna().all()
    assert (
        market["cof"].isna().all()
    )  # the cutoff is FITS geometry, not a study setting

    skipped = table[table["skipped"].astype(bool)]
    assert len(skipped) == 13  # one per condition
    assert (skipped["model"] == "fits").all()
    assert (skipped["channels"] == "C2_hybrid").all()
    assert skipped["reason"].str.contains("6.4").all()

    real = study.reportable(table)
    # Five real centre conditions x six, plus three real COF spokes x two.
    assert len(real) == 5 * 6 + 3 * 2

    swept = real[(real["model"] == "fits") & (real["anchor"] == 0)]
    assert set(swept["cutoff_period_days"]) == set(study.CUTOFFS)
    assert (swept["cof"] == 120 // swept["cutoff_period_days"]).all()
    assert (swept["dead_row_fraction"] > 0).all()

    # A univariate arm's cancellation is 1.0 by arithmetic, so the cell is empty rather
    # than perfect — suppressed at the point of the number, not only in a caveat.
    assert table[table["model"] == "fits"]["cancellation"].isna().all()
    assert table[table["model"] == "dlinear"]["cancellation"].notna().all()
    assert real["data_snapshot_last_bar"].nunique() == 1
    assert real["direction"].notna().any()


def test_the_same_grid_twice_gives_the_same_numbers(
    cfg: Config, repo_root: Path
) -> None:
    """The determinism §7.4 asks for, asserted rather than assumed.

    **One condition and one fold, and the code now says so.** It ran the whole grid twice
    while claiming this, which cost twelve extra grids per suite run and proved nothing
    the one cell does not: the property is about the seed reaching every stage, and it
    holds for one cell or for none.
    """
    if not cache_ready(cfg, repo_root):
        pytest.skip("no cached history; this test needs data_cache/")

    narrow = replace(cfg, channels=replace(cfg.channels, active="C0_base"))
    one = (study.Condition(0, 1e-3, study.REAL, study.CUTOFFS[0]),)

    first = study.run(narrow, n_folds=1, design=one)
    second = study.run(narrow, n_folds=1, design=one)

    columns = [c for c in study.COLUMNS if c != "seconds"]  # wall time is not a result
    pd.testing.assert_frame_equal(first[columns], second[columns])
