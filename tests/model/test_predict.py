"""GB-16 acceptance: the two inference paths agree, and neither runs under the wrong rules.

Two properties carry this module. The first is that batched and single-window inference
produce the same number for the same window — if they can drift, the backtest and the live
loop are describing different systems and every offline result is a claim about something
that will not run. The second is that a checkpoint refuses to score under a configuration
it was not trained under, or for a symbol it was never trained on: both failures are silent
otherwise, producing a forecast that looks exactly like a good one.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glassbox.config.loader import Config, config_hash, load_config, model_config_hash
from glassbox.contracts.schemas import Forecast, WindowBatch
from glassbox.features.builder import build_feature_frame, build_windows
from glassbox.model import predict as inference
from glassbox.model import train as trainer
from tests.model.test_train import synthetic_bars

SYMBOL = "TEST"
OTHER = "OTHER"
UNKNOWN = "NEVER"

INPUT_LEN = 30
HORIZON = 4
TRAIN_ROWS = 220
VAL_ROWS = 60


@pytest.fixture
def cfg() -> Config:
    base = load_config()
    return replace(
        base,
        window=replace(base.window, input_len=INPUT_LEN, horizon=HORIZON),
        model=replace(base.model, epochs=6, patience=3),
    )


@pytest.fixture
def universe(cfg: Config) -> dict[str, pd.DataFrame]:
    return {
        SYMBOL: build_feature_frame(synthetic_bars(seed=7), cfg),
        OTHER: build_feature_frame(synthetic_bars(seed=11), cfg),
    }


@pytest.fixture
def splits(universe: dict[str, pd.DataFrame]) -> tuple[pd.DatetimeIndex, ...]:
    index = universe[SYMBOL].index
    train = index[INPUT_LEN - 1 : INPUT_LEN - 1 + TRAIN_ROWS][:-HORIZON]
    val_start = INPUT_LEN - 1 + TRAIN_ROWS
    val = index[val_start : val_start + VAL_ROWS][:-HORIZON]
    test = index[val_start + VAL_ROWS :][:-HORIZON]
    return train, val, test


@pytest.fixture
def checkpoint(
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
    tmp_path: Path,
) -> Path:
    train_index, val_index, test_index = splits
    trainer.train(universe, cfg, train_index, val_index, test_index, tmp_path / "fold1")
    return tmp_path / "fold1"


@pytest.fixture
def predictor(checkpoint: Path, cfg: Config) -> inference.Predictor:
    return inference.load_predictor(checkpoint, cfg)


# ── the two paths agree ──────────────────────────────────────────────────────


def test_the_two_paths_agree_bit_for_bit(
    predictor: inference.Predictor,
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
) -> None:
    """The property GB-27 will assert across train and live, asserted here across the API.

    Both paths run ``build_windows`` — one with ``as_of`` set, one without — and GB-9 made
    that a filter over the same list of end positions rather than a second implementation.
    So this must be exact, not close. Every test timestamp is checked, not a sample.
    """
    _, _, test_index = splits
    frame = universe[SYMBOL]
    stats = predictor.stats_for(SYMBOL)
    batch = trainer.select_windows(
        build_windows(frame, cfg, SYMBOL, stats=stats), test_index
    )

    batched = inference.predict_batch(predictor, batch)

    for row, as_of in enumerate(batch.timestamps):
        single = inference.predict_window(predictor, frame, cfg, SYMBOL, as_of)
        np.testing.assert_array_equal(single.path, batched[row])


def test_the_single_window_path_returns_a_forecast(
    predictor: inference.Predictor,
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
) -> None:
    """The live loop's shape: a `Forecast`, not a bare array."""
    _, _, test_index = splits

    forecast = inference.predict_window(
        predictor, universe[SYMBOL], cfg, SYMBOL, test_index[0]
    )

    assert isinstance(forecast, Forecast)
    assert forecast.path.shape == (HORIZON,)
    assert forecast.path.dtype == np.float32
    assert forecast.symbol == SYMBOL
    assert forecast.as_of == test_index[0]


def test_the_batch_path_returns_one_row_per_window(
    predictor: inference.Predictor,
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
) -> None:
    _, _, test_index = splits
    batch = trainer.select_windows(
        build_windows(universe[SYMBOL], cfg, SYMBOL, stats=predictor.stats_for(SYMBOL)),
        test_index,
    )

    predicted = inference.predict_batch(predictor, batch)

    assert predicted.shape == (len(batch.timestamps), HORIZON)
    assert predicted.dtype == np.float32


def test_a_pooled_batch_scores_every_symbol(
    predictor: inference.Predictor,
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
) -> None:
    """One model, five symbols — the point of universe-wide training."""
    _, _, test_index = splits
    pooled = WindowBatch.concat(
        [
            trainer.select_windows(
                build_windows(
                    universe[symbol], cfg, symbol, stats=predictor.stats_for(symbol)
                ),
                test_index,
            )
            for symbol in sorted(universe)
        ]
    )

    predicted = inference.predict_batch(predictor, pooled)

    assert predicted.shape == (len(pooled.timestamps), HORIZON)
    assert np.isfinite(predicted).all()


# ── the per-symbol scaler ────────────────────────────────────────────────────


def test_each_symbol_is_normalised_by_its_own_statistics(
    predictor: inference.Predictor,
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
) -> None:
    """The caller names a symbol, never a scaler.

    Asserted by contradiction: scoring SYMBOL's window under OTHER's statistics gives a
    different forecast, so the choice of scaler is load-bearing and the API is right not
    to let a caller make it.
    """
    _, _, test_index = splits
    as_of = test_index[0]
    frame = universe[SYMBOL]

    correct = inference.predict_window(predictor, frame, cfg, SYMBOL, as_of)
    wrong_scaler = predictor.model.predict(
        build_windows(
            frame, cfg, SYMBOL, stats=predictor.stats_for(OTHER), as_of=as_of
        ).X
    )[0]

    assert predictor.stats_for(SYMBOL) != predictor.stats_for(OTHER)
    assert not np.allclose(correct.path, wrong_scaler)


def test_an_untrained_symbol_is_refused(
    predictor: inference.Predictor,
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
) -> None:
    """No fallback scaler, and no scaler fitted at prediction time.

    Falling back to another symbol's statistics, or to none, presents the shared weights a
    distribution they were never fitted on and returns a plausible number. Fitting one now
    is worse — the only data available includes the period being predicted.
    """
    _, _, test_index = splits

    with pytest.raises(
        ValueError, match="holds no normalisation statistics for 'NEVER'"
    ):
        inference.predict_window(
            predictor, universe[SYMBOL], cfg, UNKNOWN, test_index[0]
        )


def test_the_refusal_names_the_symbols_it_does_know(
    predictor: inference.Predictor,
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
) -> None:
    _, _, test_index = splits

    with pytest.raises(ValueError, match=r"trained on \['OTHER', 'TEST'\]"):
        inference.predict_window(
            predictor, universe[SYMBOL], cfg, UNKNOWN, test_index[0]
        )


def test_a_pooled_batch_carrying_an_untrained_symbol_is_refused(
    predictor: inference.Predictor,
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
) -> None:
    """The batch path checks too, so an unknown symbol cannot be smuggled in as a row."""
    _, _, test_index = splits
    batch = trainer.select_windows(
        build_windows(universe[SYMBOL], cfg, SYMBOL, stats=predictor.stats_for(SYMBOL)),
        test_index,
    )
    smuggled = replace(batch, symbols=(UNKNOWN,) + batch.symbols[1:])

    with pytest.raises(ValueError, match="holds no normalisation statistics"):
        inference.predict_batch(predictor, smuggled)


def test_the_predictor_lists_the_symbols_it_can_score(
    predictor: inference.Predictor,
) -> None:
    assert predictor.symbols == (OTHER, SYMBOL)


# ── the configuration guard ──────────────────────────────────────────────────


def test_a_checkpoint_from_another_config_will_not_load(
    checkpoint: Path, cfg: Config
) -> None:
    """Named in both directions, so the message says what to change.

    The gate is `model_config_hash` since 20 Aug 2026, so the config that drifts here has
    to be one that could have shaped the weights.
    """
    drifted = replace(
        cfg, window=replace(cfg.window, input_len=cfg.window.input_len + 8)
    )

    with pytest.raises(ValueError) as raised:
        inference.load_predictor(checkpoint, drifted)

    assert model_config_hash(cfg) in str(raised.value)
    assert model_config_hash(drifted) in str(raised.value)


def test_the_hash_can_be_rechecked_at_the_point_of_use(
    predictor: inference.Predictor, cfg: Config
) -> None:
    """GB-26's loop runs for a session and can outlive the config it started with."""
    inference.require_current_config(predictor, cfg)
    drifted = replace(cfg, signal=replace(cfg.signal, top_k=3))

    with pytest.raises(ValueError, match="refusing to forecast under rules"):
        inference.require_current_config(predictor, drifted)


def test_the_predictor_records_the_checkpoints_hash(
    predictor: inference.Predictor, cfg: Config, checkpoint: Path
) -> None:
    manifest = json.loads((checkpoint / "checkpoint.json").read_text())

    assert predictor.config_hash == manifest["config_hash"]
    assert predictor.config_hash == config_hash(cfg)


# ── the channel guard ────────────────────────────────────────────────────────


def test_a_different_channel_set_is_refused(
    predictor: inference.Predictor,
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
) -> None:
    _, _, test_index = splits
    batch = trainer.select_windows(
        build_windows(universe[SYMBOL], cfg, SYMBOL, stats=predictor.stats_for(SYMBOL)),
        test_index,
    )
    renamed = replace(batch, channels=("a", "b", "c", "d", "e"))

    with pytest.raises(ValueError, match="the weights are indexed by position"):
        inference.predict_batch(predictor, renamed)


def test_a_reordered_channel_set_is_refused(
    predictor: inference.Predictor,
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
) -> None:
    """Order matters as much as membership, and no shape check would catch a swap."""
    _, _, test_index = splits
    batch = trainer.select_windows(
        build_windows(universe[SYMBOL], cfg, SYMBOL, stats=predictor.stats_for(SYMBOL)),
        test_index,
    )
    swapped = replace(batch, channels=batch.channels[::-1])

    assert sorted(swapped.channels) == sorted(predictor.channels)
    with pytest.raises(ValueError, match="a different order applies each weight"):
        inference.predict_batch(predictor, swapped)


# ── the checkpoint is what decides ───────────────────────────────────────────


def test_inference_reproduces_the_training_runs_own_predictions(
    checkpoint: Path,
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
) -> None:
    """Round-tripping through disk must not change a single value.

    GB-49 trains once and scores many times from the saved artefact; if the two differed,
    every reported number would belong to a model that no longer exists.
    """
    train_index, val_index, test_index = splits
    fresh = trainer.train(universe, cfg, train_index, val_index, test_index)
    loaded = inference.load_predictor(checkpoint, cfg)
    batch = trainer.select_windows(
        build_windows(universe[SYMBOL], cfg, SYMBOL, stats=fresh.stats[SYMBOL]),
        test_index,
    )

    np.testing.assert_array_equal(
        fresh.model.predict(batch.X), inference.predict_batch(loaded, batch)
    )


def test_a_window_with_too_little_history_says_so(
    predictor: inference.Predictor, universe: dict[str, pd.DataFrame], cfg: Config
) -> None:
    """The live loop's real failure: asking for a forecast before enough bars exist."""
    frame = universe[SYMBOL]

    with pytest.raises(ValueError, match="bars behind it"):
        inference.predict_window(predictor, frame, cfg, SYMBOL, frame.index[2])
