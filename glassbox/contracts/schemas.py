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
    "DecisionRecord",
    "Forecast",
    "Signal",
    "WindowBatch",
]
