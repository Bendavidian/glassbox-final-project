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

from collections.abc import Sequence
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
    """Output of builder.build_windows(). The universal model input.

    ``symbols`` is **one entry per window**, not one per batch. A single-symbol batch
    carries ``("AAPL",) * B``; a pooled batch carries the symbol each window came from.

    The plural is a contract change made on 2026-08-17, when the study ruled that every
    forecaster trains across the universe. A pooled batch genuinely has one symbol per
    window, and the alternatives were worse: taking a *sequence* of batches into the
    protocol pushes pooling into every consumer, and keeping a single string forces the
    pooled case to be encoded somewhere outside the contract — the implicitness that
    promoting :class:`ChannelStats` out of ``build_windows`` removed.
    ``FitProvenance.symbols`` already anticipated the plural case; this makes
    ``WindowBatch`` express it too.
    """

    X: np.ndarray  # (B, L, C) float32 — B windows, L lags, C channels
    y: np.ndarray  # (B, H)    float32 — target log-return path
    channels: tuple[str, ...]  # length C, ordered, matches X's last axis
    timestamps: pd.DatetimeIndex  # length B, the 't' of each window
    symbols: tuple[str, ...]  # length B, the symbol each window came from
    source: str  # the data source these windows were built from, e.g. "yfinance"

    @property
    def unique_symbols(self) -> tuple[str, ...]:
        """The distinct symbols in this batch, sorted.

        A derived accessor rather than a stored field, so it cannot fall out of step with
        ``symbols``. Three callers wanted ``tuple(sorted(set(...)))`` and three copies of
        it is how two of them end up sorting differently.
        """
        return tuple(sorted(set(self.symbols)))

    @classmethod
    def concat(cls, batches: Sequence[WindowBatch]) -> WindowBatch:
        """Pool several batches into one, refusing any that do not belong together.

        Universe-wide training (2026-08-17) needs one batch spanning every symbol, and
        this is the only place that assembly happens. It validates rather than trusts:

        * **Channel tuples must match exactly, order included.** Pooling batches with
          different channel sets is a bug that produces a silently wrong model — the
          shared weights would be applied to a different meaning at the same index, and
          every downstream number would look ordinary.
        * **Window geometry must match.** Different ``input_len`` or ``horizon`` cannot be
          stacked at all, and the failure should name the mismatch rather than surface as
          a numpy broadcasting error.
        * **Sources must match.** GB-8 measured a 4-9% volume level difference between
          vendors; pooling across them is the same defect as splicing within one series.

        The result's ``timestamps`` are **not** monotonic: they are the concatenation of
        several symbols' ranges, so the same date appears once per symbol. That is correct
        for a pooled batch and is why ``FitProvenance.from_batch`` takes the minimum and
        maximum rather than the first and last.

        Raises:
            ValueError: ``batches`` is empty, or the batches disagree on channels,
                geometry or source.
        """
        batches = list(batches)
        if not batches:
            raise ValueError("cannot concatenate an empty sequence of batches")

        first = batches[0]
        for other in batches[1:]:
            if other.channels != first.channels:
                _fail(
                    "WindowBatch.concat",
                    f"every batch to carry the channels {list(first.channels)}",
                    list(other.channels),
                )
            if other.X.shape[1:] != first.X.shape[1:]:
                _fail(
                    "WindowBatch.concat",
                    f"every batch to have windows of shape {first.X.shape[1:]}",
                    other.X.shape[1:],
                )
            if other.y.shape[1] != first.y.shape[1]:
                _fail(
                    "WindowBatch.concat",
                    f"every batch to have a horizon of {first.y.shape[1]}",
                    other.y.shape[1],
                )
            if other.source != first.source:
                _fail(
                    "WindowBatch.concat",
                    f"every batch to come from {first.source!r}; a window must be "
                    "assembled from one source",
                    other.source,
                )

        return cls(
            X=np.concatenate([batch.X for batch in batches], axis=0),
            y=np.concatenate([batch.y for batch in batches], axis=0),
            channels=first.channels,
            timestamps=pd.DatetimeIndex(
                np.concatenate([batch.timestamps.to_numpy() for batch in batches])
            ),
            symbols=tuple(symbol for batch in batches for symbol in batch.symbols),
            source=first.source,
        )

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
        if len(self.symbols) != self.X.shape[0]:
            _fail(
                "WindowBatch.symbols",
                f"one symbol per window ({self.X.shape[0]})",
                len(self.symbols),
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

    ``symbols`` is a tuple because every forecaster trains across the universe (ruling of
    2026-08-17, and spec 6.4's ``individual_weights: false`` for FITS). It was a tuple
    before ``WindowBatch`` carried more than one symbol, which is what let that change
    happen without touching this class.

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

        **Minimum and maximum, not first and last.** A pooled batch's timestamps are the
        concatenation of several symbols' ranges and are therefore not monotonic, so
        ``timestamps[-1]`` is the last symbol's end rather than the batch's. GB-25 compares
        this range against a fold's test range, and a range that understated its own extent
        would let a leak pass.
        """
        if len(batch.timestamps) == 0:
            _fail("FitProvenance", "a batch with at least one window", 0)
        return cls(
            channels=batch.channels,
            symbols=batch.unique_symbols,
            source=batch.source,
            fitted_start=batch.timestamps.min(),
            fitted_end=batch.timestamps.max(),
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
class Trade:
    """One completed round trip. **The same type in a backtest and in a live session.**

    ``entry_price`` and ``exit_price`` are **reference** prices - the levels the rules
    chose, before slippage. Slippage lives in ``costs`` rather than being folded into the
    prices, so the log answers "what did the rule pick?" and "what did the frictions take?"
    separately. ``net_pnl == gross_pnl - costs`` exactly, and is asserted for every trade.

    ``strategy_exit`` is ``False`` only for an administrative exit - ``end_of_data`` in a
    backtest, a liquidation in a live session: the position was closed because something
    ended, not because the strategy decided anything. **GB-19's rule, pinned here rather
    than left to be invented later:** such trades ARE included in the equity curve and
    total return, because the curve must be complete and the capital was genuinely
    returned - but they are EXCLUDED from hit rate, average trade and any other
    per-decision statistic. Filter on this field, never on an ``exit_reason`` string.

    **Why this is a contract rather than a backtest-local type (GB-29).** It began in
    ``backtest/engine.py``. GB-29 needs the live path to emit trades that GB-19's metrics
    read **without a translation layer**, and the live path may not import the validation
    harness - so a shared type cannot live there. Two types with a shared shape was the
    alternative and it is worse: every metric would need to accept both, the duck typing
    would be untested until one side grew a field, and "structurally identical" is a
    property a reviewer has to check by eye rather than one the type system holds.

    ``entry_order_id`` and ``exit_order_id`` are the broker's, and are ``None`` for every
    backtest trade - a simulated fill has no order to point at. They are **optional
    additions rather than a second type**: GB-19 never reads them, GB-32's replay needs
    them to tie a trade back to what the broker actually did, and a live trade that could
    not name its own orders would be unreconcilable.
    """

    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    size: float  # shares
    entry_price: float
    exit_price: float
    gross_pnl: float
    costs: float
    net_pnl: float
    exit_reason: str
    strategy_exit: bool  # False when the exit was administrative, not a decision
    entry_order_id: str | None = None  # live only; a backtest fill has no broker order
    exit_order_id: str | None = None


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
    "Trade",
    "WindowBatch",
]
