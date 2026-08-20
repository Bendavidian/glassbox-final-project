"""GB-44 acceptance: FITS selected by configuration, and one model serving five symbols.

Two claims, and they are different claims.

**The switch.** ``model.active: fits`` must select FITS with no other change — no import
added, no branch taken, no test edited. The registry of spec 4.3 is what makes that true,
and the thing that could quietly break it is not the registry but the *other* list: the
config layer validates ``model.active`` against ``VALID_MODELS`` and may not import the
model layer, so two lists must agree and nothing made them. A name in one and not the
other is a switch that either cannot be reached or passes validation and fails at train
time.

**The sharing.** "One model serves all five symbols" is asserted by *breaking* it rather
than by reading it: perturbing the single weight matrix must move every symbol's forecast.
A count of models proves nothing about whether they are shared, and a run that quietly
fitted five would still return one object.

The regime itself comes from ``fits.individual_weights``, which spec 5 declared, spec 6.4
documented and nothing consulted until GB-44.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glassbox.config.loader import VALID_MODELS, Config, load_config
from glassbox.features.builder import build_feature_frame, build_windows
from glassbox.model import ALL_FORECASTERS
from glassbox.model import train as trainer
from glassbox.model.fits import FITSForecaster
from glassbox.model.ltsf import DLinearForecaster
from tests.model.test_train import (
    HORIZON,
    INPUT_LEN,
    TRAIN_ROWS,
    VAL_ROWS,
    synthetic_bars,
)

# The real universe, so "one model serves all five symbols" is the sentence the test
# asserts rather than a two-symbol stand-in for it. The bars are synthetic — this suite
# runs offline — but the names and the count are the study's.
UNIVERSE = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL")


@pytest.fixture
def cfg() -> Config:
    """The project's configuration with FITS selected and the geometry made small.

    ``model.active`` is the only field that carries meaning for this file; the window and
    epoch counts are shrunk so the suite stays quick, and are the same reductions
    ``tests/model/test_train.py`` makes.
    """
    base = load_config()
    return replace(
        base,
        universe=UNIVERSE,
        window=replace(base.window, input_len=INPUT_LEN, horizon=HORIZON),
        model=replace(base.model, active="fits", epochs=8, patience=3),
    )


@pytest.fixture
def universe(cfg: Config) -> dict[str, pd.DataFrame]:
    """Five symbols on one calendar, five different price paths."""
    return {
        symbol: build_feature_frame(synthetic_bars(seed=seed), cfg)
        for seed, symbol in enumerate(UNIVERSE, start=3)
    }


@pytest.fixture
def splits(universe: dict[str, pd.DataFrame]) -> tuple[pd.DatetimeIndex, ...]:
    index = universe[UNIVERSE[0]].index
    train = index[INPUT_LEN - 1 : INPUT_LEN - 1 + TRAIN_ROWS][:-HORIZON]
    start = INPUT_LEN - 1 + TRAIN_ROWS
    val = index[start : start + VAL_ROWS][:-HORIZON]
    return train, val


@pytest.fixture
def run(
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
) -> trainer.TrainingRun:
    train_index, val_index = splits
    return trainer.train(universe, cfg, train_index, val_index)


# ── the switch ───────────────────────────────────────────────────────────────


def test_the_registry_and_the_config_layer_name_the_same_models() -> None:
    """The two lists that must agree, and which nothing made agree until now.

    ``config/loader.py`` cannot import ``model`` — the layer contract forbids it — so
    ``VALID_MODELS`` is a second copy of the registry's keys, written by hand. A name in
    ``VALID_MODELS`` alone passes configuration validation and raises a ``KeyError`` deep
    in ``train``; a name in the registry alone is a model no configuration can select.
    Either way "``model.active`` is the only change needed" would be false, and no other
    test in this suite would notice.
    """
    assert set(VALID_MODELS) == set(ALL_FORECASTERS)


def test_the_config_switch_is_the_only_difference_between_two_arms(
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
) -> None:
    """One call, two configurations differing in one string, two architectures.

    The assertion that matters is not that FITS appears — it is that **nothing else
    moved**: the same universe, the same splits, the same window counts and the same
    scalers, which is what "with no other change" means operationally.
    """
    train_index, val_index = splits
    other = replace(cfg, model=replace(cfg.model, active="dlinear"))

    fits = trainer.train(universe, cfg, train_index, val_index)
    dlinear = trainer.train(universe, other, train_index, val_index)

    assert isinstance(fits.model, FITSForecaster)
    assert isinstance(dlinear.model, DLinearForecaster)
    assert fits.model.name == "fits"

    assert fits.n_train_windows == dlinear.n_train_windows
    assert fits.n_val_windows == dlinear.n_val_windows
    assert fits.stats == dlinear.stats
    assert fits.model.fitted.symbols == dlinear.model.fitted.symbols


def test_the_switch_survives_the_checkpoint_round_trip(
    run: trainer.TrainingRun,
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
    tmp_path: Path,
) -> None:
    """Selecting FITS by configuration has to reach the *deployed* model, not just the run.

    ``load_checkpoint`` rebuilds the class from the manifest's name through the same
    registry, so this is the other half of the switch: a name that trains and cannot be
    loaded back is not integrated.
    """
    train_index, val_index = splits
    saved = trainer.train(
        universe, cfg, train_index, val_index, checkpoint_dir=tmp_path / "fits"
    )
    loaded = trainer.load_checkpoint(tmp_path / "fits", cfg)

    assert isinstance(loaded.model, FITSForecaster)
    assert loaded.manifest["model"]["name"] == "fits"

    windows = _windows_for(UNIVERSE[0], universe, cfg, saved, train_index)
    assert np.array_equal(loaded.model.predict(windows), saved.model.predict(windows))


# ── the sharing ──────────────────────────────────────────────────────────────


def test_shared_training_produces_one_model_serving_all_five_symbols(
    run: trainer.TrainingRun,
) -> None:
    """The acceptance criterion of GB-44, in the terms spec 8 states it.

    One fitted object, one weight matrix, five symbols in the provenance, and five
    scalers — because normalisation staying per symbol is what makes the pooling
    legitimate (2026-08-17). The window count is the fifth symbol's worth of evidence:
    a pooled run sees every symbol's windows, not one symbol's.
    """
    assert isinstance(run.model, FITSForecaster)
    assert run.model.fitted.symbols == tuple(sorted(UNIVERSE))
    assert set(run.stats) == set(UNIVERSE)
    assert run.model.weight.shape == (run.model.cof, run.model.out_bins)
    assert run.n_train_windows % len(UNIVERSE) == 0


def test_one_weight_matrix_drives_every_symbols_forecast(
    run: trainer.TrainingRun,
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
) -> None:
    """Sharing asserted by breaking it, which is the only way to assert it.

    A run that secretly fitted five models would still return one object and still list
    five symbols. What no such run could survive is this: perturb the **single** weight
    matrix and *every* symbol's forecast must move. If one symbol were served by weights
    of its own, its forecast would sit still.

    **Row 1, not row 0**, and the reason is a finding rather than a detail: row 0 maps the
    DC bin, which RIN has already set to zero, so perturbing it moves nothing for any
    symbol and this test would pass for the wrong reason. See
    ``test_fits.py::test_the_dc_row_is_allocated_and_cannot_learn``.
    """
    train_index, _ = splits
    windows = {
        symbol: _windows_for(symbol, universe, cfg, run, train_index)
        for symbol in UNIVERSE
    }
    before = {symbol: run.model.predict(rows) for symbol, rows in windows.items()}

    disturbed = run.model.weight.copy()
    disturbed[1, 0] += 1.0
    run.model._set_weight(disturbed)

    for symbol, rows in windows.items():
        after = run.model.predict(rows)
        assert not np.allclose(after, before[symbol]), (
            f"{symbol}'s forecast did not move when the shared weight did, so it is not "
            "being served by the shared weight"
        )


def test_the_checkpoint_records_which_regime_produced_the_weights(
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
    tmp_path: Path,
) -> None:
    """GB-25 should read the regime, not infer it from a symbol count.

    A one-symbol universe is ambiguous by counting — it is the same run under either
    setting — so the manifest states which setting was in force.
    """
    train_index, val_index = splits
    trainer.train(
        universe, cfg, train_index, val_index, checkpoint_dir=tmp_path / "shared"
    )
    manifest = json.loads(
        (tmp_path / "shared" / trainer.MANIFEST_FILE).read_text(encoding="utf-8")
    )

    assert manifest["training"]["weight_sharing"] == "universe"
    assert manifest["training"]["symbols"] == sorted(UNIVERSE)


# ── the other setting ────────────────────────────────────────────────────────


def test_individual_weights_refuses_a_pooled_universe(
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
) -> None:
    """The flag was decorative until GB-44; the refusal is what makes it mean something.

    The message has to carry the remedy, because the remedy *is* the regime: one fitted
    model cannot hold five weight sets while ``Forecaster.predict`` takes no symbol, so
    per-symbol weights are five calls and five checkpoints.
    """
    train_index, val_index = splits
    per_symbol = replace(cfg, fits=replace(cfg.fits, individual_weights=True))

    with pytest.raises(ValueError) as raised:
        trainer.train(universe, per_symbol, train_index, val_index)

    message = str(raised.value)
    assert "individual_weights" in message
    assert "one symbol at a time" in message
    for symbol in UNIVERSE:
        assert symbol in message


def test_individual_weights_trains_one_symbol_at_a_time(
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
    tmp_path: Path,
) -> None:
    """The regime is reachable, and it produces genuinely different weights per symbol."""
    train_index, val_index = splits
    per_symbol = replace(cfg, fits=replace(cfg.fits, individual_weights=True))

    weights = []
    for symbol in UNIVERSE:
        run = trainer.train(
            {symbol: universe[symbol]},
            per_symbol,
            train_index,
            val_index,
            checkpoint_dir=tmp_path / symbol,
        )
        assert run.model.fitted.symbols == (symbol,)
        manifest = json.loads(
            (tmp_path / symbol / trainer.MANIFEST_FILE).read_text(encoding="utf-8")
        )
        assert manifest["training"]["weight_sharing"] == "per_symbol"
        weights.append(run.model.weight.copy())

    for position, weight in enumerate(weights[1:], start=1):
        assert not np.allclose(weight, weights[0]), (
            f"{UNIVERSE[position]} learned the same weights as {UNIVERSE[0]}, so the "
            "per-symbol regime is not per symbol"
        )


def test_a_universe_of_one_is_accepted_under_either_setting(
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
) -> None:
    """A universe of one has nothing to share weights across, so the regimes coincide.

    Refusing it under ``false`` would refuse a run that is not wrong — and every unit test
    in ``test_train.py`` is one.
    """
    train_index, val_index = splits
    one = {UNIVERSE[0]: universe[UNIVERSE[0]]}
    per_symbol = replace(cfg, fits=replace(cfg.fits, individual_weights=True))

    shared_run = trainer.train(one, cfg, train_index, val_index)
    individual_run = trainer.train(one, per_symbol, train_index, val_index)

    assert np.allclose(shared_run.model.weight, individual_run.model.weight)


def _windows_for(
    symbol: str,
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    run: trainer.TrainingRun,
    index: pd.DatetimeIndex,
) -> np.ndarray:
    """One symbol's training windows, normalised by that symbol's own statistics."""
    batch = build_windows(universe[symbol], cfg, symbol, stats=run.stats[symbol])
    return trainer.select_windows(batch, index).X
