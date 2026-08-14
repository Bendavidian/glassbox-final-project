"""L0: the frozen data schemas of spec 4.2 - WindowBatch, Forecast, Attribution,
Signal, DecisionRecord, and the FeatureFrame conventions.

These interfaces do not change without a recorded decision in DECISIONS.md. They are
what make three models and two feature sets a config switch rather than three
codebases.

Every schema is a frozen dataclass and validates its own shape on construction, so a
malformed window or forecast fails at the boundary that produced it rather than deep
inside a model.

Implemented in GB-3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# ── L2 output ────────────────────────────────────────────────
# FeatureFrame: pd.DataFrame, DatetimeIndex (UTC), one column
# per canonical channel name. No NaNs after warm-up trim.

FLOAT32 = np.dtype(np.float32)


def _fail(field_path: str, requirement: str, value: Any) -> None:
    """Raise ``ValueError`` naming the field, its requirement and what was found.

    Deliberately duplicated from ``config/loader.py`` rather than imported. Importing
    it would be legal under the layer contract - config sits below contracts - but it
    would drag the YAML and dotenv import graph into every module that touches a
    schema, to share three lines. The shared thing here is the message convention, not
    the code, and both copies are asserted against it in tests.
    """
    raise ValueError(f"{field_path} must be {requirement}, got {value!r}")


@dataclass(frozen=True)
class WindowBatch:
    """Output of builder.build_windows(). The universal model input."""

    X: np.ndarray  # (B, L, C) float32 — B windows, L lags, C channels
    y: np.ndarray  # (B, H)    float32 — target log-return path
    channels: tuple[str, ...]  # length C, ordered, matches X's last axis
    timestamps: pd.DatetimeIndex  # length B, the 't' of each window
    symbol: str
    source: str  # the data source these windows were built from, e.g. "yfinance"

    def __post_init__(self) -> None:
        if self.X.ndim != 3:
            _fail("WindowBatch.X", "a 3-dimensional (B, L, C) array", self.X.shape)
        if self.y.ndim != 2:
            _fail("WindowBatch.y", "a 2-dimensional (B, H) array", self.y.shape)
        if self.X.dtype != FLOAT32:
            _fail("WindowBatch.X", "float32", self.X.dtype)
        if self.y.dtype != FLOAT32:
            _fail("WindowBatch.y", "float32", self.y.dtype)
        if self.X.shape[0] != self.y.shape[0]:
            _fail(
                "WindowBatch.y",
                f"one target row per window ({self.X.shape[0]})",
                self.y.shape[0],
            )
        if self.X.shape[0] != len(self.timestamps):
            _fail(
                "WindowBatch.timestamps",
                f"one timestamp per window ({self.X.shape[0]})",
                len(self.timestamps),
            )
        if self.X.shape[2] != len(self.channels):
            _fail(
                "WindowBatch.channels",
                f"one name per channel in X ({self.X.shape[2]})",
                len(self.channels),
            )


@dataclass(frozen=True)
class ChannelStats:
    """Normalisation statistics for one symbol's channels.

    Fitted on a training split only, by ``features.builder.fit_stats``, and passed into
    ``build_windows``. Never fitted inside window assembly: doing so would fold the test
    period's mean and variance into the training input, which is the single most likely
    place for leakage to enter this system.

    ``mean`` and ``std`` are positional against ``channels``, so ordering is explicit and
    two folds' statistics compare with a plain equality check. Every field is a float,
    a string or a timestamp, so a checkpoint (GB-15) serialises it without a custom
    encoder.

    ``fitted_start`` and ``fitted_end`` record which rows produced these numbers. They
    are what make "were these fitted on training data only?" an answerable question:
    GB-25's leakage audit intersects that range against the fold's test range.
    """

    channels: tuple[str, ...]  # ordered, matches WindowBatch's channel axis
    mean: tuple[float, ...]  # aligned to `channels`
    std: tuple[float, ...]  # aligned to `channels`
    fitted_start: pd.Timestamp  # first row the statistics were fitted on
    fitted_end: pd.Timestamp  # last row — with fitted_start, identifies the split
    n_rows: int

    def __post_init__(self) -> None:
        if len(self.mean) != len(self.channels):
            _fail(
                "ChannelStats.mean",
                f"one value per channel ({len(self.channels)})",
                len(self.mean),
            )
        if len(self.std) != len(self.channels):
            _fail(
                "ChannelStats.std",
                f"one value per channel ({len(self.channels)})",
                len(self.std),
            )
        if self.n_rows <= 0:
            _fail("ChannelStats.n_rows", "a positive row count", self.n_rows)
        if self.fitted_start > self.fitted_end:
            _fail(
                "ChannelStats.fitted_start",
                f"no later than fitted_end ({self.fitted_end})",
                self.fitted_start,
            )
        for channel, deviation in zip(self.channels, self.std, strict=True):
            if not deviation > 0:
                _fail(f"ChannelStats.std[{channel}]", "greater than zero", deviation)


@dataclass(frozen=True)
class FitProvenance:
    """What a forecaster was fitted on. Recorded by every implementation, uniformly.

    A no-op ``fit`` is honest, but a checkpoint that records nothing about its training
    data cannot be audited: "was this model fitted on data that overlaps its test fold?"
    has to be answerable from the checkpoint alone, for every forecaster, or GB-25's
    leakage audit becomes a manual reading of training scripts. This is the same argument
    that put ``fitted_start`` / ``fitted_end`` on :class:`ChannelStats`, applied one layer
    up.

    ``symbols`` is a tuple although a ``WindowBatch`` carries one symbol, because spec 6.4
    trains one model across the universe (``individual_weights: false``). GB-44 will fit
    from a batch assembled over five symbols, and a tuple absorbs that without a contract
    change.

    Normalisation statistics are deliberately **not** duplicated here. Their fitted range
    already lives on :class:`ChannelStats`, which the builder produces and the checkpoint
    stores; recording the same range in two places invites the two to disagree, and the
    audit would then have to decide which one to believe.
    """

    channels: tuple[str, ...]  # the channel set and its order
    symbols: tuple[str, ...]  # every symbol whose windows were used
    source: str  # provenance of the underlying bars, e.g. "yfinance"
    fitted_start: pd.Timestamp  # first window timestamp in the training batch
    fitted_end: pd.Timestamp  # last — with fitted_start, the training range
    n_windows: int
    input_len: int  # L, as the model was configured
    horizon: int  # H

    @classmethod
    def from_batch(
        cls, batch: WindowBatch, input_len: int, horizon: int
    ) -> FitProvenance:
        """Derive provenance from the batch a forecaster was handed.

        A classmethod rather than a helper in each model, so Persistence, DLinear and
        FITS cannot record subtly different things and leave the audit comparing
        apples to pears.
        """
        if len(batch.timestamps) == 0:
            _fail("FitProvenance", "a batch with at least one window", 0)
        return cls(
            channels=batch.channels,
            symbols=(batch.symbol,),
            source=batch.source,
            fitted_start=batch.timestamps[0],
            fitted_end=batch.timestamps[-1],
            n_windows=len(batch.timestamps),
            input_len=input_len,
            horizon=horizon,
        )

    def __post_init__(self) -> None:
        if self.n_windows <= 0:
            _fail("FitProvenance.n_windows", "a positive window count", self.n_windows)
        if self.fitted_start > self.fitted_end:
            _fail(
                "FitProvenance.fitted_start",
                f"no later than fitted_end ({self.fitted_end})",
                self.fitted_start,
            )
        if not self.symbols:
            _fail("FitProvenance.symbols", "at least one symbol", self.symbols)


@dataclass(frozen=True)
class Forecast:
    """A predicted log-return path for one symbol, as of one timestamp."""

    path: np.ndarray  # (H,) float32 — predicted log returns
    symbol: str
    as_of: pd.Timestamp

    def __post_init__(self) -> None:
        if self.path.ndim != 1:
            _fail("Forecast.path", "a 1-dimensional (H,) array", self.path.shape)


@dataclass(frozen=True, kw_only=True)
class Attribution:
    """Exact decomposition. MUST satisfy:
    sum(per_channel.values()) ≈ forecast.path.sum()  (atol=1e-5)

    per_lag is optional. The per-lag heatmap (GB-31) is cut from scope;
    implementations return None. It is retained in the schema so the feature
    can be added later without a contract change."""

    per_channel: dict[str, float]  # channel → contribution to Σ path
    per_lag: np.ndarray | None = None  # (L, C) contribution heatmap — optional
    per_frequency: dict[float, float] | None  # FITS only: period(days) → contrib
    gain_phase: dict[float, tuple[float, float]] | None  # FITS only
    forecast_total: float

    def __post_init__(self) -> None:
        if self.per_lag is not None and self.per_lag.ndim != 2:
            _fail(
                "Attribution.per_lag",
                "a 2-dimensional (L, C) array",
                self.per_lag.shape,
            )


@dataclass(frozen=True)
class Signal:
    """The decision layer's verdict for one symbol on one bar."""

    symbol: str
    action: str  # "enter_long" | "hold" | "exit"
    trend_strength: float
    up_points: int
    passed_threshold: bool


@dataclass(frozen=True)
class DecisionRecord:
    """The complete, replayable record of one decision."""

    as_of: pd.Timestamp
    symbol: str
    forecast: Forecast
    attribution: Attribution
    signal: Signal
    order: dict | None  # None in co-pilot-pending or hold
    narrative: str
    config_hash: str  # ties the record to the exact config that made it


__all__ = [
    "Attribution",
    "ChannelStats",
    "DecisionRecord",
    "FitProvenance",
    "Forecast",
    "Signal",
    "WindowBatch",
]
