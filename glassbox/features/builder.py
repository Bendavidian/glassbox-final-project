"""L2 THE KEYSTONE: the only place in the codebase where a model input window is
assembled (spec 3.3).

Offline training and the live loop call this same function with the same config, which
is what structurally guarantees the model sees the same thing in both worlds. Do not
create a second assembly path. Produces a WindowBatch.

Three properties this module exists to hold:

**One slicing implementation.** ``as_of=None`` yields every valid window; ``as_of=t``
yields exactly one. They are not two code paths — ``as_of`` filters the same list of
end positions the batch path uses, and the same arithmetic runs on both. GB-27 asserts
byte-identical output for a shared timestamp, and it should have nothing to catch.

**Statistics come from outside.** ``fit_stats`` is a separate function the caller applies
to a training split. Fitting inside ``build_windows`` would fold the test period's mean
and variance into the training input — the single most likely place for leakage to enter
this system, and invisible in every downstream number if it happened.

**One window, one source.** A frame spliced from two vendors puts a step change inside
every normalising window that straddles the join: GB-8 measured Alpaca reporting 4-9%
more volume than yfinance, which would fabricate a ``vol_z`` spike out of nothing.
``build_feature_frame`` refuses a frame whose provenance is missing or mixed.

Implemented in GB-9. Train/live parity is asserted in GB-27.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from glassbox.config.loader import Config
from glassbox.contracts.schemas import ChannelStats, WindowBatch
from glassbox.data.historical import LOG_RETURN, SOURCE_COLUMN
from glassbox.features import indicators, wavelets

# Channel name -> the function that produces it from a canonical bar frame.
#
# `close_logret` is the log-return series, not the raw price. Spec 4.1 fixes the model
# input series as log returns and spec 7.3 bans price levels as a modelling target. The
# name says so because GB-30 renders it: "62% from the close_logret channel" is
# unambiguous where "the close channel" would not be. The raw `close` column stays in the
# bars frame, where the backtester takes it for PnL.
# **Every builder takes ``(bars, cfg)``**, including the four that ignore the second
# argument. GB-47's wavelet channels need the configuration - family, level count, window -
# and the alternative was a second table for the configured ones, which is the defect class
# CLAUDE.md 3 now names: one fact in two places with nothing keeping them equal.
CHANNEL_BUILDERS = {
    "close_logret": lambda bars, cfg: bars[LOG_RETURN],
    "rsi14": lambda bars, cfg: indicators.rsi14(bars),
    "vol_z": lambda bars, cfg: indicators.vol_z(bars),
    "mom10": lambda bars, cfg: indicators.mom10(bars),
    "ma_dist20": lambda bars, cfg: indicators.ma_dist20(bars),
    "wav_a1": lambda bars, cfg: wavelets.approximation(bars, cfg, level=1),
    "wav_a2": lambda bars, cfg: wavelets.approximation(bars, cfg, level=2),
    "wav_a3": lambda bars, cfg: wavelets.approximation(bars, cfg, level=3),
}

# The channel the forecast targets (spec 6.4: the spectral pipeline is univariate).
TARGET_CHANNEL = "close_logret"

# How much history each channel needs behind a window before its value at a given
# timestamp is **byte-identical** to the number training computed there.
#
# For a windowed (FIR) indicator that is its window length, and it is exact: the value
# depends on that many bars and on nothing before them. For a **recursive** (IIR) one it is
# a decay argument, and GB-27 proved the first version of that argument wrong.
#
# **What the bound must be on.** The seed's contribution to the value at bar t is
# `weight(k) x seed_difference`, where `weight(k) = (13/14)^k` for Wilder's RSI and
# `seed_difference` is the gap between the true early average gain/loss and whatever a
# truncated history produced. The original derivation bounded **the weight alone** below
# 1e-7 — `(13/14)^218 < 1e-7`, giving 14 + 218 = 232 and a floor of 352 — which is only
# correct if the seed difference is at most 1. It is not bounded by 1: it is a difference
# of average gains, in price units. Near RSI 50 a float32 ulp is 5.95e-06 and GB-27
# measured a residual of 3.815e-06, the same order of magnitude. **352 was marginal by
# construction**, which is why the sweep found byte-identity in 32 of 125 symbol-timestamp
# pairs rather than in none or in all.
#
# **The target is now 1e-10**, three orders of magnitude tighter than the original 1e-7,
# which covers a seed difference of up to ~1000x the value's own ulp:
#
#     (13/14)^k < 1e-10  ->  k = ceil(ln(1e-10) / ln(13/14)) = 311
#     warm-up = 14 (the seed window) + 311 = 325
#     min_history_bars = input_len 120 + 325 = 445
#
# **If this is ever changed, change the TARGET and re-derive.** A number tuned until a
# sweep passes is how 352 got here; 1e-10 is a stated margin that can be argued with.
#
# `indicators.RSI_WARMUP` **is** this number, imported rather than repeated. GB-8 held a
# second constant at 77, from a 1e-2 target with the same known-bad derivation, to answer
# "when does the value stop remembering its seed?" — which is the same question as "when is
# it byte-identical to what training computed?". GB-27 unified them: one derivation to
# argue with rather than two constants to keep in step.
#
# **A value may be an int or a function of the configuration** (GB-47). The indicators'
# warm-ups are module constants; the wavelets' is ``wavelet.rolling_window``, which is a
# configured value, and hardcoding 64 here would put a magic number beside the setting it
# is supposed to follow. `_warmup_bars` resolves either form.
#
# **And the wavelet warm-up is exact where the RSI's is a bound.** A DWT of a trailing
# window depends on that window and on nothing before it, so ``rolling_window`` bars is not
# a tolerance argument - two callers holding the same 64 bars compute the same number, to
# the last bit. There is no residue to bound and no target to re-derive.
PARITY_WARMUP = {
    "close_logret": 1,
    "rsi14": indicators.RSI_WARMUP,
    "vol_z": indicators.VOL_Z_WINDOW,
    "mom10": indicators.MOMENTUM_LOOKBACK,
    "ma_dist20": indicators.MA_DIST_WINDOW,
    "wav_a1": wavelets.rolling_window,
    "wav_a2": wavelets.rolling_window,
    "wav_a3": wavelets.rolling_window,
}


def min_history_bars(cfg: Config) -> int:
    """The fewest bars a caller must supply for the active channel set.

    ``input_len`` alone is not enough: a window of L bars whose channels need warm-up
    behind them yields nothing, or worse, yields values that differ from the ones
    training computed at the same timestamps. The live loop (GB-26) requests this.

    Raises:
        ValueError: An active channel has not declared a warm-up.
    """
    channels = cfg.channels.active_channels
    _reject_unknown(channels, PARITY_WARMUP, "declared no parity warm-up")
    return cfg.window.input_len + max(
        _warmup_bars(channel, cfg) for channel in channels
    )


def deepest_warmup_channel(cfg: Config) -> str:
    """The active channel whose warm-up sets :func:`min_history_bars`.

    Ties break on the channel name, so the answer does not depend on config ordering.
    """
    channels = cfg.channels.active_channels
    _reject_unknown(channels, PARITY_WARMUP, "declared no parity warm-up")
    return max(channels, key=lambda channel: (_warmup_bars(channel, cfg), channel))


def history_requirement(cfg: Config) -> str:
    """One line saying where :func:`min_history_bars` comes from, for an error message.

    ``data.live`` refuses a short history and has to explain why, but it sits **below**
    ``features`` in the layer stack and cannot compute the floor or name the channel that
    sets it. So the layer that knows writes the sentence and the caller carries it down.
    That keeps the data layer ignorant of channel warm-ups, which is correct, without
    making its refusal message useless, which would not be.
    """
    channel = deepest_warmup_channel(cfg)
    return (
        f"{min_history_bars(cfg)} bars is input_len {cfg.window.input_len} plus "
        f"{_warmup_bars(channel, cfg)} bars of {channel} warm-up, the deepest of the "
        "active channels"
    )


def _warmup_bars(channel: str, cfg: Config) -> int:
    """One channel's warm-up, whether it is a constant or a configured value."""
    declared = PARITY_WARMUP[channel]
    return declared(cfg) if callable(declared) else declared


def build_feature_frame(bars: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Assemble every channel of the active config into one frame.

    Args:
        bars: Canonical bar frame from ``data.historical`` or ``data.live``, carrying a
            single-valued ``source`` column.
        cfg: Resolved configuration; supplies the active channel set.

    Returns:
        A frame with one column per channel, in config order, indexed by the bars' UTC
        DatetimeIndex with the warm-up rows trimmed away. No NaNs remain. Provenance
        travels in ``frame.attrs["source"]``.

    Raises:
        ValueError: Provenance is missing or mixed, a channel is not implemented, or a
            NaN survives the warm-up trim.
    """
    source = _single_source(bars)
    channels = cfg.channels.active_channels
    _reject_unknown(channels, CHANNEL_BUILDERS, "is not implemented")

    frame = pd.DataFrame(
        {channel: CHANNEL_BUILDERS[channel](bars, cfg) for channel in channels},
        index=bars.index,
    )[list(channels)]

    frame = _trim_warmup(frame)
    if frame.isna().to_numpy().any():
        raise ValueError("NaNs remain after the warm-up trim; the bars have gaps")

    frame.attrs["source"] = source
    return frame


def fit_stats(frame: pd.DataFrame, cfg: Config) -> ChannelStats:
    """Fit per-channel normalisation statistics over exactly the rows given.

    The caller passes a training split. This function has no idea what a fold is, which
    is the point: it cannot accidentally see a test period it was not handed.

    Raises:
        ValueError: The frame is empty, a channel is missing, or a channel is constant
            and so has no scale to normalise by.
    """
    channels = cfg.channels.active_channels
    if frame.empty:
        raise ValueError("cannot fit statistics on an empty frame")

    missing = [channel for channel in channels if channel not in frame.columns]
    if missing:
        raise ValueError(f"the frame is missing channels: {missing}")

    means: list[float] = []
    deviations: list[float] = []
    for channel in channels:
        column = frame[channel].astype("float64")
        deviation = float(column.std())
        if not deviation > 0:
            raise ValueError(
                f"{channel} is constant over the fitted range, so it has no scale; "
                "normalising it would divide by zero"
            )
        means.append(float(column.mean()))
        deviations.append(deviation)

    return ChannelStats(
        channels=channels,
        mean=tuple(means),
        std=tuple(deviations),
        fitted_start=frame.index[0],
        fitted_end=frame.index[-1],
        n_rows=len(frame),
    )


def build_windows(
    frame: pd.DataFrame,
    cfg: Config,
    symbol: str,
    stats: ChannelStats | None = None,
    as_of: pd.Timestamp | None = None,
) -> WindowBatch:
    """Assemble model input windows from a feature frame.

    Args:
        frame: Output of :func:`build_feature_frame`.
        cfg: Resolved configuration; supplies ``input_len``, ``horizon`` and channels.
        symbol: Carried into the batch.
        stats: Normalisation statistics from :func:`fit_stats`, fitted on a training
            split by the caller. ``None`` leaves the channels unnormalised.
        as_of: When given, return exactly the window ending at this timestamp. When
            ``None``, return every window whose target is fully known.

    Returns:
        A :class:`WindowBatch`. ``y`` is NaN where the H future bars do not exist yet,
        which is the live case: the target is genuinely not known, and saying so beats
        inventing a value or raising.

    Raises:
        ValueError: The frame lacks provenance or a channel, ``as_of`` is not in the
            index or has too little history behind it, or ``stats`` describes different
            channels than the config.
    """
    channels = cfg.channels.active_channels
    input_len = cfg.window.input_len
    horizon = cfg.window.horizon

    source = frame.attrs.get("source")
    if not source:
        raise ValueError(
            "the feature frame carries no provenance; build it with build_feature_frame"
        )

    missing = [channel for channel in channels if channel not in frame.columns]
    if missing:
        raise ValueError(f"the frame is missing channels: {missing}")

    if stats is not None and stats.channels != channels:
        raise ValueError(
            f"stats describe channels {list(stats.channels)}, "
            f"but the active config is {list(channels)}"
        )

    values = frame.loc[:, list(channels)].to_numpy(dtype="float32", copy=True)
    targets = frame[_target_channel(channels)].to_numpy(dtype="float32", copy=True)
    n_rows = len(frame)

    # One list of end positions, filtered rather than branched. A window ending at
    # position p spans [p - input_len + 1, p], so it uses only rows <= p: the causality
    # rule holds by construction of the slice, not by a check afterwards.
    positions = _end_positions(n_rows, input_len, horizon, frame.index, as_of)

    windows = np.lib.stride_tricks.sliding_window_view(values, input_len, axis=0)
    # sliding_window_view gives (n - L + 1, C, L); the window ending at p starts at
    # p - L + 1, and the contract wants (B, L, C).
    selected = windows[[position - input_len + 1 for position in positions]]
    X = np.ascontiguousarray(selected.transpose(0, 2, 1), dtype="float32")

    y = np.full((len(positions), horizon), np.nan, dtype="float32")
    for row, position in enumerate(positions):
        end = position + 1 + horizon
        if end <= n_rows:
            y[row] = targets[position + 1 : end]

    if stats is not None:
        mean = np.asarray(stats.mean, dtype="float32")
        deviation = np.asarray(stats.std, dtype="float32")
        X = ((X - mean) / deviation).astype("float32")
        # **The target is scaled too, by the target channel's own deviation** (ruled
        # 20 Aug 2026). Divided, never centred: see `restore_targets` for why the
        # difference decides whether attribution can stay exact.
        y = (y / np.float32(stats.scale_for(_target_channel(channels)))).astype(
            "float32"
        )

    return WindowBatch(
        X=X,
        y=y,
        channels=channels,
        timestamps=frame.index[positions],
        # One frame is one symbol, so every window in it carries the same name. The batch
        # becomes genuinely plural only when `WindowBatch.concat` pools several.
        symbols=(symbol,) * len(positions),
        source=str(source),
    )


def _end_positions(
    n_rows: int,
    input_len: int,
    horizon: int,
    index: pd.DatetimeIndex,
    as_of: pd.Timestamp | None,
) -> list[int]:
    """Every position a window may end at, or just the one ``as_of`` names."""
    if n_rows < input_len:
        raise ValueError(
            f"the frame has {n_rows} rows, fewer than the {input_len} an input window "
            "needs; see min_history_bars(cfg)"
        )

    if as_of is None:
        # Training: the target must be fully known, so stop `horizon` bars from the end.
        return list(range(input_len - 1, n_rows - horizon))

    timestamp = pd.Timestamp(as_of)
    if timestamp not in index:
        raise ValueError(f"as_of {timestamp} is not in the frame's index")

    position = int(index.get_loc(timestamp))
    if position < input_len - 1:
        raise ValueError(
            f"as_of {timestamp} has {position + 1} bars behind it, fewer than the "
            f"{input_len} an input window needs"
        )
    return [position]


def target_scales(batch: WindowBatch, stats: Mapping[str, ChannelStats]) -> np.ndarray:
    """``(B,)`` — the number each row's target was divided by, row by row.

    A pooled batch holds several symbols and each was scaled by its own statistics, so the
    inverse is per row and not per batch. Getting that wrong is the bug this function
    exists to have one place to get right.

    Raises:
        ValueError: A symbol in the batch has no statistics.
    """
    channel = _target_channel(batch.channels)
    missing = sorted(set(batch.symbols) - set(stats))
    if missing:
        raise ValueError(
            f"no statistics for {missing}; the batch carries "
            f"{list(batch.unique_symbols)} and the mapping holds {sorted(stats)}"
        )
    return np.array(
        [stats[symbol].scale_for(channel) for symbol in batch.symbols], dtype="float64"
    )


def restore_targets(
    values: np.ndarray, batch: WindowBatch, stats: Mapping[str, ChannelStats]
) -> np.ndarray:
    """Undo :func:`build_windows`' target scaling. ``(B, H)`` in, ``(B, H)`` out.

    **The inverse lives beside the transform**, in the keystone module, because a
    transform whose inverse is written somewhere else is a transform that will one day be
    applied twice or not at all. Every point where a forecast or a realised path leaves
    the model layer calls this: ``model.predict``'s two entry points, and the offline
    runner's calibration and backtest paths.

    **Multiplicative, because the transform is.** ``build_windows`` divides the target and
    does not centre it, and the reason is spec 4.4 rather than convenience: centring makes
    ``predict`` affine, so the forecast carries a constant belonging to no channel and
    ``Attribution.from_terms`` refuses the decomposition — its error message names exactly
    this case. Scaling alone keeps every stage linear, so attribution stays exact and the
    inverse is one multiplication. Measured over 16 folds, the two forms are
    indistinguishable anyway: MAE 0.016061 against 0.016068 for DLinear and 0.015841
    against 0.015802 for FITS.
    """
    scales = target_scales(batch, stats)
    if values.shape[0] != len(scales):
        raise ValueError(
            f"cannot restore {values.shape[0]} rows against a batch of {len(scales)}"
        )
    return np.asarray(values, dtype="float64") * scales[:, None]


def _target_channel(channels: tuple[str, ...]) -> str:
    """The channel the forecast targets: the close log-return series."""
    if TARGET_CHANNEL not in channels:
        raise ValueError(
            f"the active channel set has no `{TARGET_CHANNEL}` channel to target"
        )
    return TARGET_CHANNEL


def _single_source(bars: pd.DataFrame) -> str:
    """Return the frame's one provenance value, or refuse.

    A frame spliced from two vendors is the failure this guards: GB-8 measured a 4-9%
    volume level difference between sources, which inside a 20-bar z-score window
    manufactures a spike that never happened.
    """
    if SOURCE_COLUMN not in bars.columns:
        raise ValueError(
            f"the bars carry no {SOURCE_COLUMN} column, so their provenance is unknown; "
            "load them through data.historical or data.live"
        )

    sources = sorted({str(value) for value in bars[SOURCE_COLUMN].dropna().unique()})
    if not sources:
        raise ValueError(f"the {SOURCE_COLUMN} column is empty")
    if len(sources) > 1:
        raise ValueError(
            f"the bars mix {len(sources)} sources ({sources}); a window must be "
            "assembled from one source, or its normalising windows straddle a step "
            "change in the data"
        )
    return sources[0]


def _trim_warmup(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the leading rows where any channel is still warming up."""
    valid = frame.notna().all(axis=1)
    if not valid.any():
        return frame.iloc[0:0]
    return frame.loc[valid.idxmax() :]


def _reject_unknown(channels: tuple[str, ...], known: dict, complaint: str) -> None:
    unknown = [channel for channel in channels if channel not in known]
    if unknown:
        raise ValueError(f"channel(s) {unknown}: {complaint}")
