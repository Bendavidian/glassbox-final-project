"""L3: inference - batched over a :class:`WindowBatch`, and one window at a time for the
live loop.

Two entry points, one path. :func:`predict_batch` scores a batch that already exists;
:func:`predict_window` assembles the single window ending at a timestamp and scores that.
The second calls ``features.builder.build_windows`` with ``as_of`` set, which is the same
function and the same arithmetic the first's batch came from - GB-9's one-slicing-
implementation property. So the two agree by construction, and a test asserts it
bit-for-bit rather than trusting the argument.

**A checkpoint decides everything.** The weights, the channel set, the window geometry and
the per-symbol normalisation all come out of the checkpoint directory. No value that shaped
the model is read from the live configuration; it is passed in only so its hash can be
**compared** against the checkpoint's, and a difference is refused with both hashes named.
A model scored under rules other than the ones that trained it produces numbers that look
ordinary and mean nothing.

**Per-symbol statistics, supplied by the checkpoint and not by the caller.** Training is
universe-wide with a per-symbol scaler (ruling of 2026-08-17), so a window must be
normalised by *its own symbol's* mean and standard deviation. The caller therefore names a
symbol, not a scaler: passing statistics in would let a caller normalise AAPL's window with
NVDA's numbers, and nothing downstream could tell.

**A symbol the checkpoint has no statistics for is refused**, naming it and listing the
ones that exist. It is tempting to fall back - to the nearest symbol's scaler, or to none -
and both are wrong. The shared weights were fitted on inputs whose per-symbol scale had
already been removed, so feeding them an unscaled or wrongly-scaled window presents a
distribution the model never saw, and the result is a plausible-looking forecast rather than
an error. Fitting a scaler for the new symbol at prediction time is worse: the only data
available to fit it on includes the period being predicted. Refusing keeps "what the model
saw" the same question at inference time as it is in GB-25's audit.

Implemented in GB-16.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from glassbox.config.loader import Config, config_hash
from glassbox.contracts.schemas import ChannelStats, Forecast, WindowBatch
from glassbox.features.builder import build_windows
from glassbox.model.train import load_checkpoint


@dataclass(frozen=True, eq=False)
class Predictor:
    """A checkpoint made ready to score windows.

    ``eq=False``: it holds a forecaster and a mapping of numpy-backed statistics, so a
    default equality would compare arrays elementwise and raise.
    """

    model: Any  # a Forecaster rebuilt from the checkpoint
    stats: Mapping[str, ChannelStats]
    config_hash: str
    channels: tuple[str, ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        """The symbols this checkpoint can score, sorted."""
        return tuple(sorted(self.stats))

    def stats_for(self, symbol: str) -> ChannelStats:
        """This symbol's normalisation statistics, or refuse naming what is available."""
        if symbol not in self.stats:
            raise ValueError(
                f"the checkpoint holds no normalisation statistics for {symbol!r}; it "
                f"was trained on {list(self.symbols)}. Scoring an untrained symbol would "
                "feed the shared weights a distribution they never saw, and the result "
                "would look like a forecast rather than an error"
            )
        return self.stats[symbol]


def load_predictor(directory: str | Path, cfg: Config) -> Predictor:
    """Load a checkpoint for inference, refusing one from a different configuration.

    Args:
        directory: A checkpoint directory written by ``model.train.save_checkpoint``.
        cfg: The configuration now in force. Used **only** for the hash comparison and to
            rebuild the forecaster's shape; no value shaping the model is read from it.

    Returns:
        A :class:`Predictor`.

    Raises:
        FileNotFoundError: The directory holds no manifest.
        ValueError: The checkpoint's config hash is not the current one - the message names
            both - or the manifest is unreadable.
    """
    loaded = load_checkpoint(directory, cfg)
    return Predictor(
        model=loaded.model,
        stats=loaded.stats,
        config_hash=loaded.manifest["config_hash"],
        channels=tuple(loaded.manifest["model"]["channels"]),
    )


def predict_batch(predictor: Predictor, batch: WindowBatch) -> np.ndarray:
    """Score every window in ``batch``.

    The batch is assumed already normalised - it came from ``build_windows`` with the
    statistics this checkpoint holds. What is checked here is that every symbol in it is
    one the checkpoint knows, so a pooled batch cannot smuggle in a symbol the model was
    never trained on and have its rows scored anyway.

    Args:
        predictor: From :func:`load_predictor`.
        batch: Windows to score.

    Returns:
        ``(B, H)`` float32 log-return paths, in the batch's own row order.

    Raises:
        ValueError: The batch's channels differ from the checkpoint's, or it carries a
            symbol the checkpoint has no statistics for.
    """
    _require_channels(predictor, batch.channels)
    for symbol in batch.unique_symbols:
        predictor.stats_for(symbol)
    return predictor.model.predict(batch.X)


def predict_window(
    predictor: Predictor,
    frame: pd.DataFrame,
    cfg: Config,
    symbol: str,
    as_of: pd.Timestamp,
) -> Forecast:
    """Score the single window ending at ``as_of``. The live loop's entry point.

    The window is assembled by ``build_windows(as_of=...)`` - the keystone, not a second
    slicing path - and normalised with ``symbol``'s own statistics from the checkpoint.
    That is what makes this identical to the corresponding row of :func:`predict_batch`,
    and ``test_the_two_paths_agree_bit_for_bit`` holds it to that.

    Args:
        predictor: From :func:`load_predictor`.
        frame: The symbol's feature frame, with at least ``builder.min_history_bars(cfg)``
            rows behind ``as_of``.
        cfg: Resolved configuration, for the window geometry and channel set.
        symbol: Which symbol this frame is. Selects the statistics; it is not a label.
        as_of: The bar the window ends on.

    Returns:
        A :class:`Forecast` for ``symbol`` as of ``as_of``.

    Raises:
        ValueError: The symbol is unknown to the checkpoint, the channels differ, or
            ``as_of`` is not in the frame or has too little history behind it.
    """
    _require_channels(predictor, cfg.channels.active_channels)
    stats = predictor.stats_for(symbol)
    window = build_windows(frame, cfg, symbol, stats=stats, as_of=as_of)
    path = predictor.model.predict(window.X)[0]
    return Forecast(
        path=path.astype("float32"), symbol=symbol, as_of=pd.Timestamp(as_of)
    )


def _require_channels(predictor: Predictor, channels: tuple[str, ...]) -> None:
    """Refuse a channel set the checkpoint was not trained on.

    Order matters as much as membership: the weights are indexed by position, so a
    reordered channel set applies each weight matrix to a different series and produces a
    forecast that is wrong in a way no shape check would catch.
    """
    if tuple(channels) != predictor.channels:
        raise ValueError(
            f"the checkpoint was trained on channels {list(predictor.channels)} but was "
            f"given {list(channels)}; the weights are indexed by position, so a different "
            "set or a different order applies each weight to a different series"
        )


def require_current_config(predictor: Predictor, cfg: Config) -> None:
    """Re-check the hash at the point of use.

    :func:`load_predictor` already refuses a stale checkpoint, but a long-lived process -
    GB-26's live loop runs for a session - can outlive the configuration it started with.
    Exposed so the loop can assert freshness each cycle rather than only at startup.

    Raises:
        ValueError: The hashes differ. Both are named.
    """
    current = config_hash(cfg)
    if predictor.config_hash != current:
        raise ValueError(
            f"this predictor was trained under config {predictor.config_hash} and the "
            f"configuration now in force is {current}; refusing to forecast under rules "
            "the model was not trained on"
        )


__all__ = [
    "Predictor",
    "load_predictor",
    "predict_batch",
    "predict_window",
    "require_current_config",
]
