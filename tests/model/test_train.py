"""GB-15 acceptance: one training run, and the leak it is arranged to make impossible.

The hard requirement is that normalisation statistics see the training split and nothing
else. It is the second most likely place for leakage in this system after the builder, and
like the builder's version it is invisible downstream: a scaler fitted across the whole
frame produces a model that looks slightly better on every metric and cannot be caught by
any of them.

So the statistics are checked three ways here — against an independent recomputation on the
training rows, against the *wrong* answer computed on the whole frame, and through the
checkpoint, which is the form GB-25's audit will read them in.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glassbox.config.loader import Config, config_hash, load_config
from glassbox.features.builder import build_feature_frame, build_windows
from glassbox.model import train as trainer
from glassbox.model.history import EpochLoss, read_history, write_history

SYMBOL = "TEST"
INPUT_LEN = 30
HORIZON = 4
EPOCHS = 12
PATIENCE = 3

BARS = 420
TRAIN_ROWS = 220
VAL_ROWS = 60


@pytest.fixture
def cfg() -> Config:
    base = load_config()
    return replace(
        base,
        window=replace(base.window, input_len=INPUT_LEN, horizon=HORIZON),
        model=replace(base.model, epochs=EPOCHS, patience=PATIENCE),
    )


def synthetic_bars(bars: int = BARS, seed: int = 7) -> pd.DataFrame:
    """A wandering price path with volume, in the canonical bar shape.

    Built rather than downloaded: this suite must run offline and identically on any
    machine, and GB-4's cache is a snapshot that a fresh clone does not have.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2018-01-02", periods=bars, freq="B", tz="UTC")
    steps = rng.standard_normal(bars) * 0.012
    close = 100.0 * np.exp(np.cumsum(steps))
    frame = pd.DataFrame(
        {
            "open": close * (1 + rng.standard_normal(bars) * 0.001),
            "high": close * (1 + np.abs(rng.standard_normal(bars)) * 0.004),
            "low": close * (1 - np.abs(rng.standard_normal(bars)) * 0.004),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, bars).astype("float64"),
            "log_return": np.concatenate([[np.nan], np.diff(np.log(close))]),
            "source": "yfinance",
        },
        index=index,
    )
    return frame


@pytest.fixture
def frame(cfg: Config) -> pd.DataFrame:
    return build_feature_frame(synthetic_bars(), cfg)


@pytest.fixture
def splits(frame: pd.DataFrame) -> tuple[pd.DatetimeIndex, ...]:
    """Train / validation / test window-end ranges, embargoed by ``HORIZON``.

    Built the way ``make_folds`` builds them — contiguous and separated by the embargo —
    without importing it: ``model`` sits below ``backtest`` in the layer contract, and a
    test that reached across it would be asserting an import the package may not make.
    """
    index = frame.index
    train = index[INPUT_LEN - 1 : INPUT_LEN - 1 + TRAIN_ROWS][:-HORIZON]
    val_start = INPUT_LEN - 1 + TRAIN_ROWS
    val = index[val_start : val_start + VAL_ROWS][:-HORIZON]
    test = index[val_start + VAL_ROWS :][:-HORIZON]
    return train, val, test


@pytest.fixture
def run(
    frame: pd.DataFrame, cfg: Config, splits: tuple[pd.DatetimeIndex, ...]
) -> trainer.TrainingRun:
    train_index, val_index, test_index = splits
    return trainer.train(
        frame, cfg, SYMBOL, train_index, val_index, held_out=test_index
    )


# ── the hard requirement: statistics from the training split alone ───────────


def test_statistics_match_a_recomputation_on_the_training_rows(
    frame: pd.DataFrame, cfg: Config, splits: tuple[pd.DatetimeIndex, ...]
) -> None:
    """Recomputed independently of ``fit_stats``, on exactly the rows training may see."""
    train_index, _, _ = splits
    rows = frame.loc[train_index[0] : train_index[-1]]

    stats = trainer.training_stats(frame, train_index, cfg)

    for position, channel in enumerate(cfg.channels.active_channels):
        column = rows[channel].astype("float64")
        assert stats.mean[position] == pytest.approx(float(column.mean()), rel=1e-12)
        assert stats.std[position] == pytest.approx(float(column.std()), rel=1e-12)


def test_statistics_are_not_the_ones_the_whole_frame_would_give(
    frame: pd.DataFrame, cfg: Config, splits: tuple[pd.DatetimeIndex, ...]
) -> None:
    """The leak, stated as the thing that must NOT be true.

    A test that only checked the training numbers would pass just as happily if the
    training range happened to cover the whole frame. This pins the difference.
    """
    train_index, _, _ = splits
    stats = trainer.training_stats(frame, train_index, cfg)
    whole_frame = [
        float(frame[channel].astype("float64").mean())
        for channel in cfg.channels.active_channels
    ]

    assert stats.n_rows < len(frame)
    assert any(
        abs(fitted - leaked) > 1e-9
        for fitted, leaked in zip(stats.mean, whole_frame, strict=True)
    )


def test_the_fitted_range_is_exactly_the_training_range(
    run: trainer.TrainingRun, splits: tuple[pd.DatetimeIndex, ...]
) -> None:
    """GB-25 audits by comparing these to the fold. An identity, not an approximation."""
    train_index, val_index, test_index = splits

    assert run.stats.fitted_start == train_index[0]
    assert run.stats.fitted_end == train_index[-1]
    assert run.stats.fitted_end < val_index[0]
    assert run.stats.fitted_end < test_index[0]


def test_validation_is_normalised_by_the_training_statistics(
    frame: pd.DataFrame, cfg: Config, splits: tuple[pd.DatetimeIndex, ...]
) -> None:
    """Windows are built once and split afterwards, so there is no second scaler to use.

    Asserted at the data, not at the code: the validation windows the trainer produces are
    bit-identical to ones normalised with the training statistics, and are *not* what
    validation-fitted statistics would produce.
    """
    train_index, val_index, _ = splits
    stats = trainer.training_stats(frame, train_index, cfg)
    with_training_stats = build_windows(frame, cfg, SYMBOL, stats=stats)
    val_stats = trainer.training_stats(frame, val_index, cfg)
    with_val_stats = build_windows(frame, cfg, SYMBOL, stats=val_stats)

    keep = with_training_stats.timestamps.isin(val_index)
    selected = trainer.select_windows(with_training_stats, val_index, "validation")

    np.testing.assert_array_equal(selected.X, with_training_stats.X[keep])
    assert not np.allclose(selected.X, with_val_stats.X[keep])


# ── determinism ──────────────────────────────────────────────────────────────


def test_two_runs_with_the_same_seed_produce_identical_weights(
    frame: pd.DataFrame, cfg: Config, splits: tuple[pd.DatetimeIndex, ...]
) -> None:
    """Bit-identical, not close. The named acceptance test for this task.

    Nothing here reseeds a global generator, so this also holds when the two runs are
    interleaved with other work — which is what GB-49's grid will do.
    """
    train_index, val_index, _ = splits

    first = trainer.train(frame, cfg, SYMBOL, train_index, val_index)
    second = trainer.train(frame, cfg, SYMBOL, train_index, val_index)

    for channel in cfg.channels.active_channels:
        np.testing.assert_array_equal(
            first.model.weights_for(channel)["trend"],
            second.model.weights_for(channel)["trend"],
        )
        np.testing.assert_array_equal(
            first.model.weights_for(channel)["remainder"],
            second.model.weights_for(channel)["remainder"],
        )
    assert first.history == second.history
    assert first.epochs_run == second.epochs_run


def test_a_different_seed_produces_different_weights(
    frame: pd.DataFrame, cfg: Config, splits: tuple[pd.DatetimeIndex, ...]
) -> None:
    """Proof the previous test is asserting something.

    With zero initialisation the seed controls only mini-batch order, so the difference is
    small — but it must exist, or the equality above would hold for a trainer that ignored
    the seed entirely.
    """
    train_index, val_index, _ = splits
    other = replace(cfg, meta=replace(cfg.meta, seed=cfg.meta.seed + 1))

    first = trainer.train(frame, cfg, SYMBOL, train_index, val_index)
    second = trainer.train(frame, other, SYMBOL, train_index, val_index)

    assert not np.array_equal(
        first.model.weights_for("close_logret")["trend"],
        second.model.weights_for("close_logret")["trend"],
    )


def test_weights_do_not_depend_on_the_held_out_range(
    frame: pd.DataFrame, cfg: Config, splits: tuple[pd.DatetimeIndex, ...]
) -> None:
    """``held_out`` is recorded, never consumed.

    The checkpoint carries the test range so GB-25's audit is self-contained, and this is
    what keeps that from being a hole: passing it, or not passing it, produces the same
    model to the last bit.
    """
    train_index, val_index, test_index = splits

    blind = trainer.train(frame, cfg, SYMBOL, train_index, val_index)
    told = trainer.train(
        frame, cfg, SYMBOL, train_index, val_index, held_out=test_index
    )

    np.testing.assert_array_equal(
        blind.model.weights_for("close_logret")["trend"],
        told.model.weights_for("close_logret")["trend"],
    )


# ── checkpoints ──────────────────────────────────────────────────────────────


def test_a_checkpoint_round_trip_reproduces_predictions_exactly(
    frame: pd.DataFrame,
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
    tmp_path: Path,
) -> None:
    """Not "close": identical. A reloaded model that predicts differently is a new model."""
    train_index, val_index, test_index = splits
    run = trainer.train(
        frame, cfg, SYMBOL, train_index, val_index, test_index, tmp_path / "fold1"
    )
    windows = build_windows(frame, cfg, SYMBOL, stats=run.stats)

    reloaded = trainer.load_checkpoint(tmp_path / "fold1", cfg)

    np.testing.assert_array_equal(
        run.model.predict(windows.X), reloaded.model.predict(windows.X)
    )


def test_the_checkpoint_carries_the_statistics_it_trained_with(
    frame: pd.DataFrame,
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
    tmp_path: Path,
) -> None:
    """The audit reads statistics from the file, never from a live pipeline."""
    train_index, val_index, test_index = splits
    run = trainer.train(
        frame, cfg, SYMBOL, train_index, val_index, test_index, tmp_path / "fold1"
    )

    reloaded = trainer.load_checkpoint(tmp_path / "fold1", cfg)

    assert reloaded.stats == run.stats
    assert reloaded.stats.fitted_start == train_index[0]
    assert reloaded.stats.fitted_end == train_index[-1]


def test_the_checkpoint_answers_the_leakage_question_on_its_own(
    frame: pd.DataFrame,
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
    tmp_path: Path,
) -> None:
    """GB-25's audit, performed here against the manifest alone.

    Nothing is recomputed and no fold generator is consulted: the scaler range, the window
    range the model was fitted on, and the held-out range are all in the file, and the
    verdict is two comparisons.
    """
    train_index, val_index, test_index = splits
    trainer.train(
        frame, cfg, SYMBOL, train_index, val_index, test_index, tmp_path / "fold1"
    )

    manifest = json.loads((tmp_path / "fold1" / "checkpoint.json").read_text())
    model_state = json.loads((tmp_path / "fold1" / "model.json").read_text())
    held_out_start = pd.Timestamp(manifest["training"]["held_out_start"])

    scaler_end = pd.Timestamp(manifest["stats"]["fitted_end"])
    weights_end = pd.Timestamp(model_state["fitted"]["fitted_end"])

    assert scaler_end < held_out_start
    assert weights_end < held_out_start
    assert pd.Timestamp(manifest["training"]["val_end"]) < held_out_start


def test_a_checkpoint_from_a_different_config_is_refused(
    frame: pd.DataFrame,
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
    tmp_path: Path,
) -> None:
    """The hash covers the whole configuration, so an unrelated key still refuses.

    ``risk.stop_loss_pct`` has nothing to do with training, and that is the point: a list
    of "fields that matter" is wrong the first time someone adds a field and forgets it.
    """
    train_index, val_index, _ = splits
    trainer.train(frame, cfg, SYMBOL, train_index, val_index, None, tmp_path / "fold1")
    drifted = replace(cfg, risk=replace(cfg.risk, stop_loss_pct=0.04))

    assert config_hash(drifted) != config_hash(cfg)
    with pytest.raises(ValueError, match="refusing to load a model shaped by other"):
        trainer.load_checkpoint(tmp_path / "fold1", drifted)


def test_a_checkpoint_from_this_config_loads(
    frame: pd.DataFrame,
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
    tmp_path: Path,
) -> None:
    """The other half of the previous test: the guard is not simply always refusing."""
    train_index, val_index, _ = splits
    trainer.train(frame, cfg, SYMBOL, train_index, val_index, None, tmp_path / "fold1")

    loaded = trainer.load_checkpoint(tmp_path / "fold1", cfg)

    assert loaded.manifest["config_hash"] == config_hash(cfg)
    assert loaded.model.name == cfg.model.active


def test_a_checkpoint_of_another_version_is_refused(
    frame: pd.DataFrame,
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
    tmp_path: Path,
) -> None:
    train_index, val_index, _ = splits
    trainer.train(frame, cfg, SYMBOL, train_index, val_index, None, tmp_path / "fold1")
    path = tmp_path / "fold1" / "checkpoint.json"
    manifest = json.loads(path.read_text())
    manifest["version"] = trainer.CHECKPOINT_VERSION + 1
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="checkpoint, and this build reads version"):
        trainer.load_checkpoint(tmp_path / "fold1", cfg)


def test_loading_a_directory_without_a_manifest_says_so(
    cfg: Config, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint.json"):
        trainer.load_checkpoint(tmp_path, cfg)


# ── early stopping ───────────────────────────────────────────────────────────


def test_early_stopping_fires_when_validation_loss_rises(cfg: Config) -> None:
    """A synthetic task whose validation split disagrees with its training split.

    Training targets are a clean linear function of the input; validation targets are that
    function negated, so every step toward the training rule makes validation worse. The
    learning rate is raised for this test only: at the configured ``0.001`` the weights
    move so slowly from zero that validation still improves for nine epochs, which would
    make the test about the step size rather than about the mechanism. Raised, validation
    turns within two epochs and the run ends ``patience`` epochs after its best.
    """
    frame = build_feature_frame(synthetic_bars(seed=3), cfg)
    index = frame.index
    train_index = index[INPUT_LEN - 1 : INPUT_LEN - 1 + 200][:-HORIZON]

    run = _adversarial_run(frame, _decisive(cfg), train_index, index)

    assert run.stopped_early is True
    assert run.epochs_run == run.best_epoch + cfg.model.patience
    assert run.epochs_run < cfg.model.epochs


def test_early_stopping_does_not_fire_when_validation_keeps_improving(
    frame: pd.DataFrame, cfg: Config, splits: tuple[pd.DatetimeIndex, ...]
) -> None:
    """Proof the previous test found a mechanism rather than a hair trigger.

    ``patience`` epochs is a short fuse, so a trainer that miscounted would stop on
    almost anything. Here the full ``epochs`` must run.
    """
    train_index, val_index, _ = splits
    short = replace(cfg, model=replace(cfg.model, epochs=3, patience=PATIENCE))

    run = trainer.train(frame, short, SYMBOL, train_index, val_index)

    assert run.stopped_early is False
    assert run.epochs_run == 3


def test_without_a_validation_split_there_is_nothing_to_stop_on(
    frame: pd.DataFrame, cfg: Config, splits: tuple[pd.DatetimeIndex, ...]
) -> None:
    train_index, _, _ = splits

    run = trainer.train(frame, cfg, SYMBOL, train_index, val_index=None)

    assert run.epochs_run == cfg.model.epochs
    assert run.stopped_early is False
    assert run.best_epoch is None
    assert all(record.val_loss is None for record in run.history)


def test_the_restored_weights_are_the_best_epochs_not_the_last(cfg: Config) -> None:
    """Early stopping that kept the final weights would be a stopwatch, not a regulariser.

    On the adversarial task validation turns early and then only worsens, so the weights
    that survive must be the best epoch's rather than the last epoch's.
    """
    frame = build_feature_frame(synthetic_bars(seed=3), cfg)
    index = frame.index
    train_index = index[INPUT_LEN - 1 : INPUT_LEN - 1 + 200][:-HORIZON]

    run = _adversarial_run(frame, _decisive(cfg), train_index, index)
    best = run.history[run.best_epoch - 1]

    assert best.val_loss < run.history[-1].val_loss
    assert best.val_loss == min(record.val_loss for record in run.history)


def _decisive(cfg: Config) -> Config:
    """The same config with a learning rate large enough to move weights in one epoch."""
    return replace(cfg, model=replace(cfg.model, lr=0.5))


def _adversarial_run(
    frame: pd.DataFrame,
    cfg: Config,
    train_index: pd.DatetimeIndex,
    index: pd.DatetimeIndex,
) -> trainer.TrainingRun:
    """Fit on a batch whose validation targets are the negation of the training rule."""
    val_index = index[INPUT_LEN - 1 + 200 : INPUT_LEN - 1 + 320][:-HORIZON]
    stats = trainer.training_stats(frame, train_index, cfg)
    everything = build_windows(frame, cfg, SYMBOL, stats=stats)

    train_batch = trainer.select_windows(everything, train_index, "training")
    val_batch = trainer.select_windows(everything, val_index, "validation")
    rule = train_batch.X[:, -1, 0:1] * 0.01
    train_batch = _with_targets(train_batch, np.repeat(rule, HORIZON, axis=1))
    val_rule = val_batch.X[:, -1, 0:1] * -0.01
    val_batch = _with_targets(val_batch, np.repeat(val_rule, HORIZON, axis=1))

    model = trainer.ALL_FORECASTERS[cfg.model.active](cfg, cfg.channels.active_channels)
    model.fit(train_batch, val_batch)
    return trainer.TrainingRun(
        model=model,
        stats=stats,
        history=model.history,
        epochs_run=len(model.history),
        best_epoch=model.best_epoch,
        stopped_early=model.stopped_early,
        n_train_windows=len(train_batch.timestamps),
        n_val_windows=len(val_batch.timestamps),
        checkpoint=None,
        seconds=0.0,
    )


def _with_targets(batch, y: np.ndarray):
    return replace(batch, y=y.astype("float32"))


# ── split hygiene ────────────────────────────────────────────────────────────


def test_overlapping_train_and_validation_splits_are_refused(
    frame: pd.DataFrame, cfg: Config, splits: tuple[pd.DatetimeIndex, ...]
) -> None:
    """A caller that validated on its own training data would look excellent everywhere."""
    train_index, _, _ = splits

    with pytest.raises(ValueError, match="train and val splits share"):
        trainer.train(frame, cfg, SYMBOL, train_index, train_index[-20:])


def test_a_held_out_range_overlapping_training_is_refused(
    frame: pd.DataFrame, cfg: Config, splits: tuple[pd.DatetimeIndex, ...]
) -> None:
    train_index, val_index, _ = splits

    with pytest.raises(ValueError, match="train and test splits share"):
        trainer.train(frame, cfg, SYMBOL, train_index, val_index, train_index[:5])


def test_an_empty_training_split_is_refused(frame: pd.DataFrame, cfg: Config) -> None:
    with pytest.raises(ValueError, match="the train split is empty"):
        trainer.train(frame, cfg, SYMBOL, frame.index[:0])


def test_a_split_with_no_windows_says_which_one(
    frame: pd.DataFrame, cfg: Config, splits: tuple[pd.DatetimeIndex, ...]
) -> None:
    """The last ``HORIZON`` timestamps of a frame end no window: their target is unknown."""
    train_index, _, _ = splits

    with pytest.raises(ValueError, match="no window ends on any of the .* validation"):
        trainer.train(frame, cfg, SYMBOL, train_index, frame.index[-HORIZON:])


# ── the loss curve ───────────────────────────────────────────────────────────


def test_the_loss_curve_is_written_with_one_row_per_epoch(
    frame: pd.DataFrame,
    cfg: Config,
    splits: tuple[pd.DatetimeIndex, ...],
    tmp_path: Path,
) -> None:
    train_index, val_index, _ = splits
    run = trainer.train(
        frame, cfg, SYMBOL, train_index, val_index, None, tmp_path / "fold1"
    )

    curve = read_history(tmp_path / "fold1" / "history.csv")

    assert len(curve) == run.epochs_run
    assert [record.epoch for record in curve] == list(range(1, run.epochs_run + 1))
    assert all(record.val_loss is not None for record in curve)


def test_the_loss_curve_round_trips_exactly(tmp_path: Path) -> None:
    """``repr``, not a format string: a rounded curve cannot prove two runs matched."""
    curve = (
        EpochLoss(epoch=1, train_loss=0.1234567890123456, val_loss=0.9876543210987654),
        EpochLoss(epoch=2, train_loss=1e-17, val_loss=None),
    )

    write_history(tmp_path / "history.csv", curve)

    assert read_history(tmp_path / "history.csv") == curve


def test_a_file_that_is_not_a_loss_curve_is_refused(tmp_path: Path) -> None:
    (tmp_path / "history.csv").write_text("epoch,loss\n1,0.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="is not a loss curve"):
        read_history(tmp_path / "history.csv")


def test_training_loss_falls_over_the_run(
    frame: pd.DataFrame, cfg: Config, splits: tuple[pd.DatetimeIndex, ...]
) -> None:
    """A convex objective descended by Adam: the training curve must go down.

    Not a performance claim — it is the check that the curve being written is the model's
    own loss and not, say, the same number repeated because the weights never moved.
    """
    train_index, val_index, _ = splits

    run = trainer.train(frame, cfg, SYMBOL, train_index, val_index)

    assert run.history[-1].train_loss < run.history[0].train_loss
