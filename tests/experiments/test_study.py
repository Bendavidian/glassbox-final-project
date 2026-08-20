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
from glassbox.config.loader import Config, load_config
from glassbox.data.historical import LOG_RETURN
from glassbox.experiments import study


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
    """Seven conditions: a centre plus one departure per axis.

    The full cross buys interactions nobody asked about at four times the wall clock; a
    star gives every axis a **common reference to depart from**, which is what a
    sensitivity analysis needs.
    """
    design = study.conditions()

    assert len(design) == 7
    assert sum(condition.is_reference for condition in design) == 1
    assert {c.anchor for c in design} == set(study.ANCHORS)
    assert {c.lr for c in design} == set(study.LEARNING_RATES)
    assert {c.control for c in design} == set(study.CONTROLS)

    # Every non-reference condition differs from the centre in exactly one axis.
    centre = design[0]
    for condition in design[1:]:
        differences = sum(
            (
                condition.anchor != centre.anchor,
                condition.lr != centre.lr,
                condition.control != centre.control,
            )
        )
        assert differences == 1, condition


def test_the_full_cross_is_available_and_is_the_product() -> None:
    """For the day somebody does want an interaction."""
    assert len(study.conditions(full=True)) == (
        len(study.ANCHORS) * len(study.LEARNING_RATES) * len(study.CONTROLS)
    )


def test_the_plan_is_reportable_before_the_run(cfg: Config) -> None:
    """A run of this size is a decision, so the size is available without making it."""
    star = study.plan(cfg)
    full = study.plan(cfg, full=True)

    assert star.arms == 5  # six pairs, one deliberately skipped
    assert star.cells == 35
    assert star.trainings == 35 * cfg.walkforward.max_folds
    assert full.cells == 135
    assert star.seconds() < full.seconds()


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


def test_every_axis_and_every_required_column_is_present() -> None:
    """The four axes added after §7.4 was written, plus the snapshot vintage."""
    required = {
        "anchor",
        "lr",
        "control",
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
    # Seven conditions, six arms plus the market reference, one fold each.
    assert len(table) == 7 * (len(study.arms()) + 1)

    market = table[table["model"] == study.BUY_AND_HOLD]
    assert len(market) == 7  # once per condition, not once per arm
    assert market["mae"].isna().all()  # it makes no forecast, so the cell is empty
    assert market["total_return"].notna().all()

    skipped = table[table["skipped"].astype(bool)]
    assert len(skipped) == 7  # one per condition
    assert (skipped["model"] == "fits").all()
    assert (skipped["channels"] == "C2_hybrid").all()
    assert skipped["reason"].str.contains("6.4").all()

    real = study.reportable(table)
    # Five real conditions x (five live arms + the market reference).
    assert len(real) == 5 * 6

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

    One condition and one fold: the property is about the seed reaching every stage, and
    it either holds for one cell or it holds for none.
    """
    if not cache_ready(cfg, repo_root):
        pytest.skip("no cached history; this test needs data_cache/")

    narrow = replace(cfg, channels=replace(cfg.channels, active="C0_base"))

    first = study.run(narrow, n_folds=1)
    second = study.run(narrow, n_folds=1)

    columns = [c for c in study.COLUMNS if c != "seconds"]  # wall time is not a result
    pd.testing.assert_frame_equal(first[columns], second[columns])
