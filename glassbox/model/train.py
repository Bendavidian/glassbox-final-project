"""L3: one training run - statistics, fitting, checkpointing and the loss curve.

The training *loop* lives inside each forecaster's ``fit``, because spec 4.3 puts it
there: ``fit(batch, val)`` is the contract, and early stopping is named in it. This module
is what surrounds one call to that contract - deciding what the model is allowed to see,
recording what it saw, and writing both down. Keeping the two apart is what lets GB-41 add
FITS without touching this file.

**The hard requirement, and how it is made structural rather than remembered.**
Normalisation statistics are fitted on the training split alone. That is enforced here by
construction rather than by care:

* :func:`training_stats` is handed only the rows in ``[train_index[0], train_index[-1]]``,
  so it cannot see a validation or test row - the same argument that keeps ``fit_stats``
  outside ``build_windows`` (spec 3.3).
* Windows are then built **once**, with those statistics, and split by timestamp
  afterwards. There is no second ``build_windows`` call that could be handed a second
  ``ChannelStats``, so validation cannot be normalised by its own mean and variance even
  by accident. The leak has nowhere to enter.
* The statistics travel into the checkpoint whole, ``fitted_start`` and ``fitted_end``
  included, so GB-25 answers "were these fitted on training data only?" by reading the
  file rather than by reading this module.

**What a checkpoint is.** A directory, not a file, holding three artefacts that answer
three different questions:

===================  ==========================================================
``model.json``       what the model *is* - weights, and the ``FitProvenance`` of
                     GB-11 saying which windows produced them
``checkpoint.json``  what the run *was* - config hash, ``ChannelStats``, the
                     ranges shown to the model, and how training ended
``history.csv``      how the fitting *went* - per-epoch train and validation loss
===================  ==========================================================

A checkpoint is refused on load when its config hash differs from the configuration now in
force. This is deliberately strict: the hash covers every resolved configuration value, so
changing an unrelated key invalidates the checkpoint. The alternative - hashing only the
fields believed to matter - requires maintaining a list of which fields those are, and
that list is wrong the first time someone adds a field and forgets it. A false refusal
costs a retrain; a false acceptance puts a model trained under other rules into a results
table.

**One model across the universe**, ruled on 2026-08-17. Per symbol a fold gives 501
windows against DLinear's 4,800 parameters - 0.42x, underdetermined - where pooling five
symbols gives 2.09x. The decisive argument is comparability rather than fit: FITS at 1,200
parameters is *already* overdetermined per symbol at 1.67x, so training DLinear per symbol
while FITS trains pooled would handicap one arm and the study would report that handicap as
architecture.

**Normalisation stays per symbol**, and that is what makes pooling legitimate: it removes
symbol-specific scale so the shared weights learn the structure common across symbols. A
pooled scaler would leave NVDA's inputs systematically larger than MSFT's, and shared
weights have no parameter with which to express a symbol-specific response. So a checkpoint
holds ``{symbol: ChannelStats}``, and GB-16 applies each window's own symbol's statistics.

Implemented in GB-15.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from glassbox.config.loader import Config, config_hash
from glassbox.contracts.schemas import ChannelStats, WindowBatch
from glassbox.features.builder import build_windows, fit_stats
from glassbox.model import ALL_FORECASTERS
from glassbox.model.history import EpochLoss, read_history, write_history

CHECKPOINT_VERSION = 1

MANIFEST_FILE = "checkpoint.json"
MODEL_FILE = "model.json"
HISTORY_FILE = "history.csv"


@dataclass(frozen=True, eq=False)
class TrainingRun:
    """The result of one call to :func:`train`.

    ``eq=False``: it holds a model and a numpy-backed batch summary, so a default equality
    would compare arrays elementwise and raise on the ambiguous truth value.

    ``seconds`` is wall time and is deliberately **not** written to any file. Everything
    this module puts on disk is reproducible byte for byte from the config and the data,
    and a timestamp would end that.
    """

    model: Any  # a fitted Forecaster
    stats: dict[str, ChannelStats]  # one scaler per symbol, never pooled
    history: tuple[EpochLoss, ...]
    epochs_run: int
    best_epoch: int | None
    stopped_early: bool
    n_train_windows: int
    n_val_windows: int
    checkpoint: Path | None
    seconds: float


@dataclass(frozen=True, eq=False)
class LoadedCheckpoint:
    """What :func:`load_checkpoint` returns: the model, its statistics, and the manifest.

    The statistics come back beside the model rather than inside it because that is where
    they are used - ``build_windows`` takes them, the forecaster does not. A model that
    carried its own scaler would give the live loop two places to look and one of them
    would eventually be stale.
    """

    model: Any
    stats: dict[str, ChannelStats]
    manifest: dict
    history: tuple[EpochLoss, ...]


def training_stats(
    frame: pd.DataFrame, train_index: pd.DatetimeIndex, cfg: Config
) -> ChannelStats:
    """Fit normalisation statistics on the training rows and nothing else.

    Args:
        frame: The feature frame, from ``build_feature_frame``.
        train_index: The timestamps a training window may end on, as ``make_folds``
            returns them - already embargoed.
        cfg: Resolved configuration; supplies the active channel set.

    Returns:
        :class:`ChannelStats` whose ``fitted_start`` and ``fitted_end`` are exactly
        ``train_index[0]`` and ``train_index[-1]``.

    The rows the embargo removed from the end of the training split are excluded here too,
    although including them would be causal - they are still before the validation period.
    They are left out so that ``fitted_end == train_index[-1]`` holds exactly, which turns
    GB-25's audit from a comparison of two nearly-equal dates into an identity.

    Raises:
        ValueError: ``train_index`` is empty, or a channel is missing or constant.
    """
    if len(train_index) == 0:
        raise ValueError("cannot fit statistics without a training split")
    rows = frame.loc[train_index[0] : train_index[-1]]
    return fit_stats(rows, cfg)


def train(
    frames: Mapping[str, pd.DataFrame],
    cfg: Config,
    train_index: pd.DatetimeIndex,
    val_index: pd.DatetimeIndex | None = None,
    held_out: pd.DatetimeIndex | None = None,
    checkpoint_dir: str | Path | None = None,
) -> TrainingRun:
    """Fit the active forecaster across the universe on one fold, and optionally save it.

    Args:
        frames: ``{symbol: feature frame}``. Each is the **whole** frame, not a slice: a
            window ending on the first training timestamp reads ``input_len`` bars of
            history behind it, which lie before the fold and are legitimately in the past.
            A single-symbol mapping is a valid special case.
        cfg: Resolved configuration. ``model.active`` selects the forecaster.
        train_index: Timestamps a training window may end on.
        val_index: Timestamps a validation window may end on. ``None`` trains for the full
            ``cfg.model.epochs`` with no early stopping, because there is nothing to stop
            on.
        held_out: The fold's test timestamps. **Never used to build data.** It is asserted
            disjoint from the other two and recorded in the checkpoint, so that GB-25 can
            answer "did training overlap the period this model is scored on?" from the
            checkpoint alone. ``test_weights_do_not_depend_on_the_held_out_range`` proves
            it changes nothing.
        checkpoint_dir: Where to write the three artefacts. ``None`` trains without
            writing, which is what the tests and GB-49's sweeps do.

    Returns:
        A :class:`TrainingRun` whose ``stats`` holds one :class:`ChannelStats` per symbol.

    Raises:
        ValueError: ``frames`` is empty, the splits overlap, a split is empty, or a symbol
            contributes no window to a split.
    """
    if not frames:
        raise ValueError("cannot train on an empty universe")
    _require_disjoint(("train", train_index), ("val", val_index), ("test", held_out))

    # Sorted, not insertion-ordered: the pooled batch's row order decides the mini-batch
    # partition, so a caller passing the same symbols in a different order would otherwise
    # get different weights from the same data. Determinism must not depend on how a
    # dictionary was built.
    symbols = sorted(frames)

    stats = {}
    for symbol in symbols:
        try:
            stats[symbol] = training_stats(frames[symbol], train_index, cfg)
        except ValueError as error:
            # Which symbol failed is the first thing the caller needs, and a pooled run
            # otherwise reports "close_logret is constant" with no way to tell whose.
            raise ValueError(f"{symbol}: {error}") from error

    # Built once per symbol, with that symbol's training statistics, then split by
    # timestamp. There is no second call that could be handed a second ChannelStats, and
    # `concat` refuses to pool batches that do not belong together.
    everything = {
        symbol: build_windows(frames[symbol], cfg, symbol, stats=stats[symbol])
        for symbol in symbols
    }
    train_batch = WindowBatch.concat(
        [
            select_windows(everything[symbol], train_index, f"{symbol} training")
            for symbol in symbols
        ]
    )
    val_batch = (
        None
        if val_index is None
        else WindowBatch.concat(
            [
                select_windows(everything[symbol], val_index, f"{symbol} validation")
                for symbol in symbols
            ]
        )
    )

    model = ALL_FORECASTERS[cfg.model.active](cfg, cfg.channels.active_channels)
    started = time.perf_counter()
    model.fit(train_batch, val_batch)
    seconds = time.perf_counter() - started

    history: tuple[EpochLoss, ...] = tuple(getattr(model, "history", ()))
    run = TrainingRun(
        model=model,
        stats=stats,
        history=history,
        epochs_run=len(history),
        best_epoch=getattr(model, "best_epoch", None),
        stopped_early=bool(getattr(model, "stopped_early", False)),
        n_train_windows=len(train_batch.timestamps),
        n_val_windows=0 if val_batch is None else len(val_batch.timestamps),
        checkpoint=None,
        seconds=seconds,
    )
    if checkpoint_dir is None:
        return run

    written = save_checkpoint(checkpoint_dir, run, cfg, val_index, held_out)
    return replace(run, checkpoint=written)


def save_checkpoint(
    directory: str | Path,
    run: TrainingRun,
    cfg: Config,
    val_index: pd.DatetimeIndex | None = None,
    held_out: pd.DatetimeIndex | None = None,
) -> Path:
    """Write the three artefacts and return the directory.

    Every field written here answers a question GB-25 would otherwise have to answer by
    re-running the pipeline; :func:`load_checkpoint` reads them back.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    run.model.save(str(target / MODEL_FILE))
    write_history(target / HISTORY_FILE, run.history)

    manifest = {
        "version": CHECKPOINT_VERSION,
        # Ties the weights to every value that shaped them. Checked on load.
        "config_hash": config_hash(cfg),
        "seed": cfg.meta.seed,
        "model": {
            "name": run.model.name,
            "file": MODEL_FILE,
            "channels": list(cfg.channels.active_channels),
            "input_len": run.model.input_len,
            "horizon": run.model.horizon,
        },
        # One scaler per symbol, whole. `fitted_start`/`fitted_end` are the leakage
        # question, and GB-16 needs the mean and std to normalise a live window the same
        # way training did.
        "stats": {
            symbol: {
                "channels": list(stats.channels),
                "mean": list(stats.mean),
                "std": list(stats.std),
                "fitted_start": stats.fitted_start.isoformat(),
                "fitted_end": stats.fitted_end.isoformat(),
                "n_rows": stats.n_rows,
            }
            for symbol, stats in sorted(run.stats.items())
        },
        "training": {
            "symbols": sorted(run.stats),
            "n_train_windows": run.n_train_windows,
            "n_val_windows": run.n_val_windows,
            # Validation is *seen* data: early stopping selects on it. Recorded as
            # exposure, not as a spectator range.
            "val_start": _edge(val_index, 0),
            "val_end": _edge(val_index, -1),
            # Never shown to the model. Recorded so the audit is self-contained.
            "held_out_start": _edge(held_out, 0),
            "held_out_end": _edge(held_out, -1),
            "epochs_configured": cfg.model.epochs,
            "epochs_run": run.epochs_run,
            "best_epoch": run.best_epoch,
            "stopped_early": run.stopped_early,
            "patience": cfg.model.patience,
            "lr": cfg.model.lr,
            "batch_size": cfg.model.batch_size,
        },
        "history_file": HISTORY_FILE,
    }
    (target / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return target


def load_checkpoint(directory: str | Path, cfg: Config) -> LoadedCheckpoint:
    """Rebuild a checkpoint, refusing one trained under a different configuration.

    Raises:
        FileNotFoundError: The directory holds no manifest.
        ValueError: The manifest is a different version, names an unknown forecaster, or
            carries a config hash that is not the current one.
    """
    target = Path(directory)
    manifest_path = target / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{target} holds no {MANIFEST_FILE}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"{target} is a version {manifest.get('version')!r} checkpoint, and this "
            f"build reads version {CHECKPOINT_VERSION}"
        )

    current = config_hash(cfg)
    stored = manifest.get("config_hash")
    if stored != current:
        raise ValueError(
            f"{target} was trained under config {stored} and the configuration now in "
            f"force is {current}; refusing to load a model shaped by other settings"
        )

    name = manifest["model"]["name"]
    if name not in ALL_FORECASTERS:
        raise ValueError(
            f"{target} names forecaster {name!r}, which this build does not have; "
            f"known: {sorted(ALL_FORECASTERS)}"
        )

    # The class comes from the registry rather than a second name->class table that could
    # drift from it. Constructing and discarding one instance is cheaper than maintaining
    # two lists that must agree, and the hash check above guarantees `cfg` is the config
    # this checkpoint was trained under, so the construction is correctly shaped.
    channels = tuple(manifest["model"]["channels"])
    blueprint = ALL_FORECASTERS[name](cfg, channels)
    model = type(blueprint).load(str(target / manifest["model"]["file"]))

    stats = {
        symbol: ChannelStats(
            channels=tuple(stored["channels"]),
            mean=tuple(stored["mean"]),
            std=tuple(stored["std"]),
            fitted_start=pd.Timestamp(stored["fitted_start"]),
            fitted_end=pd.Timestamp(stored["fitted_end"]),
            n_rows=stored["n_rows"],
        )
        for symbol, stored in manifest["stats"].items()
    }

    history_path = target / manifest.get("history_file", HISTORY_FILE)
    history = read_history(history_path) if history_path.is_file() else ()

    return LoadedCheckpoint(
        model=model, stats=stats, manifest=manifest, history=history
    )


def select_windows(
    batch: WindowBatch, keep: pd.DatetimeIndex, what: str = "requested"
) -> WindowBatch:
    """The windows of ``batch`` whose timestamps are in ``keep``.

    Public because GB-19 and GB-49 need the same operation: build once, then cut a split
    out by timestamp. A second implementation of it elsewhere is how the two ends of a
    fold start disagreeing about which windows belong to which split.

    Slicing the *frame* instead would be wrong: a window ending on the first timestamp of a
    split needs ``input_len`` bars of history behind it, and those lie outside the split.
    Filtering windows keeps that history and still lets no window's *end* escape the split.
    """
    mask = batch.timestamps.isin(keep)
    if not mask.any():
        raise ValueError(
            f"no window ends on any of the {len(keep)} {what} timestamps; the split is "
            "shorter than the horizon, or its rows are not in this frame"
        )
    rows = np.flatnonzero(mask)
    return WindowBatch(
        X=batch.X[rows],
        y=batch.y[rows],
        channels=batch.channels,
        timestamps=batch.timestamps[rows],
        symbols=tuple(batch.symbols[row] for row in rows),
        source=batch.source,
    )


def _require_disjoint(*splits: tuple[str, pd.DatetimeIndex | None]) -> None:
    """Refuse overlapping splits, naming the pair and a timestamp they share.

    A caller that hands the same range twice would train on its own validation set and
    every downstream number would look excellent. The check is cheap; the failure is not.
    """
    named = [(name, index) for name, index in splits if index is not None]
    for position, (left_name, left) in enumerate(named):
        if len(left) == 0:
            raise ValueError(f"the {left_name} split is empty")
        for right_name, right in named[position + 1 :]:
            shared = left.intersection(right)
            if len(shared) > 0:
                raise ValueError(
                    f"the {left_name} and {right_name} splits share {len(shared)} "
                    f"timestamp(s), the first being {shared[0]}; they must be disjoint"
                )


def _edge(index: pd.DatetimeIndex | None, position: int) -> str | None:
    return None if index is None or len(index) == 0 else index[position].isoformat()


__all__ = [
    "CHECKPOINT_VERSION",
    "HISTORY_FILE",
    "MANIFEST_FILE",
    "MODEL_FILE",
    "LoadedCheckpoint",
    "TrainingRun",
    "load_checkpoint",
    "save_checkpoint",
    "select_windows",
    "train",
    "training_stats",
]
