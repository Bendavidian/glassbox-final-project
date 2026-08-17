"""L3: `DLinearForecaster` - trend/remainder decomposition with one linear map per
component per channel.

The weights are the explanation, and are kept accessible by channel name for GB-30.

Follows LTSF-Linear (Zeng et al., AAAI 2023). Three deliberate departures from the
reference implementation, each recorded in DECISIONS.md:

**Per-channel weights, not shared.** The reference maps every channel through one
``nn.Linear`` and emits a forecast per channel. This project forecasts a single series
from a channel set, and GB-30 must decompose that forecast **by channel** exactly. So each
channel gets its own weight matrix per component, and the forward pass is a plain sum of
per-channel terms::

    forecast[h] = sum_c  ( Wt[c] @ trend[:, c] + Wr[c] @ remainder[:, c] )[h]

Attribution is then a regrouping of terms already computed, not a reconstruction.

**No intercept.** ``nn.Linear`` carries a bias; this does not. The target is a log return
with a mean near zero, so an intercept buys almost nothing - and it would break the one
property the project exists to hold, since ``sum(per_channel.values())`` would fall short
of the forecast by the bias and there is no honest channel to charge it to. Without one,
the exactness assertion in ``Attribution`` holds by construction rather than by
bookkeeping.

**Zero initialisation, not the reference's.** ``nn.Linear``'s default bound assumes a
fan-in of ``L`` because the reference maps one channel to one output. This architecture
sums ``C x 2`` such maps, so that bound is wrong by construction and measurably harmful.
See :meth:`DLinearForecaster._initial_weights` for the numbers.

**Torch fits, numpy predicts.** The decomposition has no parameters, so it is applied once
before training rather than inside the forward pass. Fitting uses torch; the fitted weights
are then held as numpy and inference is pure numpy. Inference therefore has no global torch
state to be perturbed by, which is what makes determinism (contract property 2, and the two
properties that depend on it) true by construction.

Implemented in GB-13.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from glassbox.config.loader import Config
from glassbox.contracts.schemas import Attribution, FitProvenance, WindowBatch
from glassbox.model.history import EpochLoss

CHECKPOINT_VERSION = 1

# The decomposition window, from the LTSF-Linear paper. An architecture definition rather
# than a tuning knob - a DLinear with a configurable kernel is a different model, and the
# study compares models, not kernels. Same reasoning as the RSI period in GB-8.
DECOMP_KERNEL = 25

TREND = "trend"
REMAINDER = "remainder"
COMPONENTS = (TREND, REMAINDER)


def decompose(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split ``(B, L, C)`` into a centred moving-average trend and its remainder.

    The moving average is **centred**: each end is padded by ``(kernel - 1) // 2`` repeats
    of the first and last bar, so the output is the same length as the input and
    ``trend[l]`` averages lags ``l-12 .. l+12``.

    **This is not look-ahead, and the reason is worth stating precisely.** The averaging
    runs inside one input window, whose every bar is at or before the window's own
    timestamp `t`; the forecast is of bars after `t`. Nothing here reads a bar the window
    does not already contain. The causality rule constrains what a window may contain
    relative to its target - it does not require lag `l` to be computed from lags `<= l`,
    and no forecaster in this project could work if it did. Each window is padded from its
    own first and last bar, so two overlapping windows produce different trend values at
    the same timestamp: the trend is a property of the window, not of the series. Contract
    property 6 is what holds this honest - it perturbs later windows and requires earlier
    predictions to be bit-identical, which a decomposition that reached across the batch
    would fail.

    One boundary artefact worth knowing when reading a trend plot: edge replication pulls
    ``trend[0]`` and ``trend[L-1]`` toward the first and last observation, because half
    their window is a repeat of it. That is the published design, not a defect, but it does
    mean the most recent trend value is the least informative one.

    The split is **per channel** and could not be otherwise: a moving average of a
    multi-channel window is a moving average of each channel. That is what keeps the
    weights separable and GB-30's attribution exact.
    """
    if X.ndim != 3:
        raise ValueError(
            f"input must be a 3-dimensional (B, L, C) array, got {X.shape}"
        )

    pad = (DECOMP_KERNEL - 1) // 2
    front = np.repeat(X[:, :1, :], pad, axis=1)
    back = np.repeat(X[:, -1:, :], pad, axis=1)
    padded = np.concatenate([front, X, back], axis=1)

    windows = np.lib.stride_tricks.sliding_window_view(padded, DECOMP_KERNEL, axis=1)
    trend = windows.mean(axis=-1)
    return trend, X - trend


class DLinearForecaster:
    """Decomposition-Linear: one map from ``input_len`` to ``horizon`` per component,
    per channel.

    Attributes:
        trend_weights: ``{channel: (H, L) array}``. GB-30 reads these directly.
        remainder_weights: ``{channel: (H, L) array}``.
    """

    def __init__(
        self,
        input_len: int,
        horizon: int,
        channels: tuple[str, ...],
        cfg: Config | None = None,
    ) -> None:
        """``cfg`` supplies the training hyperparameters and is required only by ``fit``.

        A model rebuilt by :meth:`load` has none, because a checkpoint records weights
        rather than how to produce them. Calling ``fit`` on one says so rather than
        inventing defaults.
        """
        self.name = "dlinear"
        self.input_len = input_len
        self.horizon = horizon
        self.channels = tuple(channels)
        self.fitted: FitProvenance | None = None
        self._cfg = cfg

        # How the last fit went. Not model state and not in the checkpoint - a model
        # rebuilt by `load` reports an empty history, which is the truth: it did not
        # train. GB-15 writes these to a sidecar file for the report.
        self.history: tuple[EpochLoss, ...] = ()
        self.best_epoch: int | None = None
        self.stopped_early = False

        shape = (len(self.channels), self.horizon, self.input_len)
        self._trend = np.zeros(shape, dtype="float64")
        self._remainder = np.zeros(shape, dtype="float64")

    # ── the weights, keyed by channel name ───────────────────────────────────

    @property
    def trend_weights(self) -> dict[str, np.ndarray]:
        """``{channel: (horizon, input_len)}`` for the trend component."""
        return dict(zip(self.channels, self._trend, strict=True))

    @property
    def remainder_weights(self) -> dict[str, np.ndarray]:
        """``{channel: (horizon, input_len)}`` for the remainder component."""
        return dict(zip(self.channels, self._remainder, strict=True))

    def weights_for(self, channel: str) -> dict[str, np.ndarray]:
        """Both components for one channel, as ``{"trend": ..., "remainder": ...}``.

        The accessor GB-30 uses. It raises on an unknown channel rather than returning an
        empty mapping, so an attribution can never quietly omit a channel.
        """
        if channel not in self.channels:
            raise ValueError(
                f"{channel!r} is not one of this model's channels {list(self.channels)}"
            )
        position = self.channels.index(channel)
        return {TREND: self._trend[position], REMAINDER: self._remainder[position]}

    @property
    def n_parameters(self) -> int:
        """Learnable weights: ``len(COMPONENTS) * C * H * L``. No bias."""
        return self._trend.size + self._remainder.size

    # ── the Forecaster protocol ──────────────────────────────────────────────

    def fit(self, batch: WindowBatch, val: WindowBatch | None = None) -> None:
        """Fit both weight sets by gradient descent on mean squared error.

        Hyperparameters come from ``cfg.model``. When ``val`` is given, training stops
        early after ``patience`` epochs without improvement and the best weights are
        restored; without it, training runs the full ``epochs``.

        ``cfg.model.batch_size`` of ``None`` means one batch of everything. Randomness -
        batch order, and initialisation if it ever needs any - is drawn from a local
        generator seeded with ``cfg.meta.seed``, never from torch's global state. Two runs
        of the same config therefore produce identical weights, which is what GB-15
        asserts; under full batch they are identical across *different* seeds too.

        Normalisation statistics are not stored here: ``features.builder`` applies them
        before the batch is assembled, and duplicating them in the model would create two
        places for them to disagree. GB-15's checkpoint stores them alongside the weights.

        Sets ``history``, ``best_epoch`` and ``stopped_early`` as a side effect. They
        describe the run, not the model, and are not written into the checkpoint.
        """
        import torch

        if self._cfg is None:
            raise ValueError(
                "this model was rebuilt from a checkpoint and holds no training "
                "configuration; construct it with a Config to fit it"
            )
        self._require_batch(batch)
        plan = self._cfg.model
        generator = torch.Generator().manual_seed(self._cfg.meta.seed)

        inputs, targets = self._tensors(batch, torch)
        weights = self._initial_weights(torch, generator)
        optimiser = torch.optim.Adam(weights, lr=plan.lr)

        validation = None if val is None else self._tensors(val, torch)
        best_loss, best_state, waited = math.inf, None, 0
        curve: list[EpochLoss] = []
        self.best_epoch, self.stopped_early = None, False

        windows = targets.shape[0]
        size = windows if plan.batch_size is None else plan.batch_size
        # A permutation of a *single* batch changes nothing but the order floats are summed
        # in, and that is not nothing: measured across three seeds it moved full-batch
        # weights by a relative 5e-15, which is enough to make "same seed, same weights"
        # the only determinism available. Skipping it when there is one batch makes
        # full-batch training bit-identical across seeds as well, which is what `null`
        # should mean. With mini-batches the permutation decides the partition and is the
        # whole point, so it runs.
        one_batch = size >= windows

        for epoch in range(1, plan.epochs + 1):
            order = (
                torch.arange(windows)
                if one_batch
                else torch.randperm(windows, generator=generator)
            )
            for start in range(0, windows, size):
                rows = order[start : start + size]
                optimiser.zero_grad()
                loss = torch.nn.functional.mse_loss(
                    self._forward(torch, weights, [side[rows] for side in inputs]),
                    targets[rows],
                )
                loss.backward()
                optimiser.step()

            # Both losses at the epoch's end, over the whole split. Averaging the
            # mini-batch losses instead would measure weights that no longer exist and
            # would not be comparable to the validation number beside it.
            with torch.no_grad():
                score = (
                    None
                    if validation is None
                    else self._loss(torch, weights, validation)
                )
                curve.append(
                    EpochLoss(
                        epoch=epoch,
                        train_loss=self._loss(torch, weights, (inputs, targets)),
                        val_loss=score,
                    )
                )

            if score is None:
                continue
            if score < best_loss:
                best_loss, waited = score, 0
                self.best_epoch = epoch
                best_state = [tensor.detach().clone() for tensor in weights]
            else:
                waited += 1
                if waited >= plan.patience:
                    self.stopped_early = True
                    break

        final = best_state if best_state is not None else weights
        self._trend = final[0].detach().numpy().astype("float64")
        self._remainder = final[1].detach().numpy().astype("float64")
        self.history = tuple(curve)
        self.fitted = FitProvenance.from_batch(batch, self.input_len, self.horizon)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(B, L, C) float32 -> (B, H) float32."""
        self._require_window_shape(X)
        trend, remainder = decompose(np.asarray(X, dtype="float64"))
        forecast = np.einsum("chl,blc->bh", self._trend, trend) + np.einsum(
            "chl,blc->bh", self._remainder, remainder
        )
        return forecast.astype("float32")

    def explain(self, x: np.ndarray, channels: tuple[str, ...]) -> Attribution:
        """(L, C) -> the exact per-channel decomposition of this window's forecast.

        Not an approximation and not a re-derivation: the terms summed here are the same
        terms ``predict`` sums, regrouped by channel. That is the whole reason the weights
        are per channel, and it is why this project can ban SHAP rather than argue with it.
        """
        if tuple(channels) != tuple(self.channels):
            raise ValueError(
                f"explain was given channels {list(channels)}, but this model holds "
                f"{list(self.channels)}"
            )
        self._require_window_shape(x[None, ...])

        trend, remainder = decompose(np.asarray(x, dtype="float64")[None, ...])
        per_channel = {
            channel: float(
                np.sum(
                    self._trend[position] @ trend[0, :, position]
                    + self._remainder[position] @ remainder[0, :, position]
                )
            )
            for position, channel in enumerate(self.channels)
        }
        return Attribution(
            per_channel=per_channel,
            per_lag=None,
            per_frequency=None,
            gain_phase=None,
            forecast_total=float(sum(per_channel.values())),
        )

    def save(self, path: str) -> None:
        """Write the checkpoint as JSON. ``repr`` round-trips float64 exactly, so a
        reloaded model predicts bit-identically."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self._state()), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> DLinearForecaster:
        """Rebuild from a checkpoint. Reads no configuration file: the shape a model was
        trained at is a property of the checkpoint, not of whatever config is on disk now.
        """
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"{path} is a version {state.get('version')!r} checkpoint, and this "
                f"build reads version {CHECKPOINT_VERSION}"
            )

        model = cls(
            input_len=state["input_len"],
            horizon=state["horizon"],
            channels=tuple(state["channels"]),
        )
        model._trend = np.asarray(state["trend_weights"], dtype="float64")
        model._remainder = np.asarray(state["remainder_weights"], dtype="float64")

        fitted = state["fitted"]
        if fitted is not None:
            model.fitted = FitProvenance(
                channels=tuple(fitted["channels"]),
                symbols=tuple(fitted["symbols"]),
                source=fitted["source"],
                fitted_start=pd.Timestamp(fitted["fitted_start"]),
                fitted_end=pd.Timestamp(fitted["fitted_end"]),
                n_windows=fitted["n_windows"],
                input_len=fitted["input_len"],
                horizon=fitted["horizon"],
            )
        return model

    # ── internals ────────────────────────────────────────────────────────────

    def _forward(self, torch, weights, inputs):
        trend_w, remainder_w = weights
        trend, remainder = inputs
        return torch.einsum("chl,blc->bh", trend_w, trend) + torch.einsum(
            "chl,blc->bh", remainder_w, remainder
        )

    def _loss(self, torch, weights, split) -> float:
        """Mean squared error of ``weights`` over a whole ``(inputs, targets)`` split."""
        split_inputs, split_targets = split
        return float(
            torch.nn.functional.mse_loss(
                self._forward(torch, weights, split_inputs), split_targets
            )
        )

    def _initial_weights(self, torch, generator):
        """Zero. The model starts as the persistence baseline and learns away from it.

        Porting ``nn.Linear``'s default ``U(-1/sqrt(L), 1/sqrt(L))`` is wrong here, and
        measurably so. That bound assumes a fan-in of ``L`` because the reference maps one
        channel to one output; this architecture **sums** ``C x 2`` such maps, so the real
        fan-in is ``C * 2 * L`` - 1200 against 120 - and a random start emits forecasts
        around 1.8 when the target's standard deviation is 0.015, roughly 120x too large.
        Correcting the fan-in to ``1/sqrt(C * 2 * L)`` only reduces that to 0.58.

        Zero has none of that problem and costs nothing: the objective is convex, so the
        initialisation cannot change the optimum, only the path to it. Starting from zero
        means every weight the model moves is something it learned, and early stopping
        degrades toward the baseline rather than toward noise. Measured out of sample on
        walk-forward fold 1:

        =========================  =========  ===========  ====================
        initialisation             MAE        direction    largest contribution
        =========================  =========  ===========  ====================
        paper, ``1/sqrt(L)``       4.00x      0.344        1.39
        fan-in, ``1/sqrt(2CL)``    1.96x      0.328        0.43
        **zeros**                  **1.94x**  **0.557**    **0.058**
        =========================  =========  ===========  ====================

        The last column is why this matters beyond accuracy: with the paper's
        initialisation the attribution is still exact but unreadable, a forecast of 0.01
        explained by contributions of +1.39 and -1.20 that cancel. Zero-initialised weights
        leave contributions on the same scale as the forecast, which is what GB-30 has to
        render and a supervisor has to believe.

        ``generator`` is unused here but kept in the signature: it seeds batch order in
        ``fit``, and a future initialisation that needs randomness should draw from it
        rather than from torch's global state.
        """
        del generator
        shape = (len(self.channels), self.horizon, self.input_len)
        return [
            torch.zeros(shape, dtype=torch.float64).requires_grad_(True)
            for _ in COMPONENTS
        ]

    def _tensors(self, batch: WindowBatch, torch):
        trend, remainder = decompose(batch.X.astype("float64"))
        return (
            [torch.from_numpy(trend), torch.from_numpy(remainder)],
            torch.from_numpy(batch.y.astype("float64")),
        )

    def _state(self) -> dict:
        return {
            "version": CHECKPOINT_VERSION,
            "name": self.name,
            "input_len": self.input_len,
            "horizon": self.horizon,
            "channels": list(self.channels),
            "trend_weights": self._trend.tolist(),
            "remainder_weights": self._remainder.tolist(),
            "fitted": (
                None
                if self.fitted is None
                else {
                    "channels": list(self.fitted.channels),
                    "symbols": list(self.fitted.symbols),
                    "source": self.fitted.source,
                    "fitted_start": self.fitted.fitted_start.isoformat(),
                    "fitted_end": self.fitted.fitted_end.isoformat(),
                    "n_windows": self.fitted.n_windows,
                    "input_len": self.fitted.input_len,
                    "horizon": self.fitted.horizon,
                }
            ),
        }

    def _require_batch(self, batch: WindowBatch) -> None:
        if tuple(batch.channels) != tuple(self.channels):
            raise ValueError(
                f"the batch carries channels {list(batch.channels)}, but this model "
                f"holds {list(self.channels)}"
            )

    def _require_window_shape(self, X: np.ndarray) -> None:
        if X.ndim != 3:
            raise ValueError(
                f"input must be a 3-dimensional (B, L, C) array, got {X.shape}"
            )
        if X.shape[1] != self.input_len:
            raise ValueError(f"input must have {self.input_len} lags, got {X.shape[1]}")
        if X.shape[2] != len(self.channels):
            raise ValueError(
                f"input must have {len(self.channels)} channels, got {X.shape[2]}"
            )


__all__ = ["COMPONENTS", "DECOMP_KERNEL", "DLinearForecaster", "decompose"]
