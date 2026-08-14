"""L2: the pure trailing-window indicators rsi14, vol_z, mom10 and ma_dist20.

Every value at time t uses only bars <= t. Centred windows are forbidden here and are
tested for in GB-10. Each function is pure: DataFrame in, Series out, input untouched.

Warm-up rows are NaN and stay NaN. Nothing is backfilled and no ``min_periods`` is used
to manufacture an early value out of too little data — a number that looks like an
indicator but was computed from four bars is worse than a gap, because only the gap is
visible downstream.

**On vol_z across data sources.** Volume is the one field yfinance and Alpaca disagree
about (GB-7 measured 34-111 bps on same-day bars). A z-score over a trailing window
cancels any factor that is constant *within that window*, so what matters is not the
level of the disagreement but its stability. Measured over all 2668 overlapping bars,
2016-2026: the Alpaca/yfinance volume ratio averages 1.04-1.09 with a standard deviation
of 0.045-0.076, and it drifts across eras — roughly 1.08-1.12 in 2016-2019, 1.05-1.13 in
2020-2022, then 1.006-1.014 from 2023. That drift is multi-year, so inside any 20-bar
window the ratio is effectively constant and divides out. The residue is day-to-day noise
in the ratio, which moves the z-score by a median of 0.016-0.039 and a p95 of 0.15-0.29
in units where the z-score's own standard deviation is 1.06. vol_z is therefore safe in
the live channel set.

One condition follows from this and is not optional: **a single window must come from a
single source.** Splicing cached yfinance history onto live Alpaca bars inside one 20-bar
window would put a ~5% step in the volume level right inside the normalising window and
manufacture a z-score spike out of nothing. GB-9 assembles windows from one loader at a
time, and must keep doing so.

**On the RSI warm-up.** Wilder's average is recursive: it is seeded with a simple mean
of the first ``RSI_PERIOD`` changes and then updated as ``(prev * 13 + new) / 14``. That
seed never fully decays; its weight only falls geometrically, by ``(1 - 1/period)`` per
bar. So the first published value is not on the same footing as the thousandth. Rows are
therefore held NaN until the seed's weight drops below ``RSI_SEED_TOLERANCE``, which at
1% works out to 77 rows. That costs 77 of 2668 bars, under 3% of the history, and buys
the guarantee that no emitted RSI is still remembering its own warm-up.

Implemented in GB-8.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

CLOSE = "close"
VOLUME = "volume"

# These are indicator definitions, not tunable configuration. The periods are baked into
# the canonical channel names frozen in spec 4.1 — rsi14, vol_z, mom10, ma_dist20 — so
# changing one produces a different channel rather than a different setting, and every
# model, attribution and stored decision record referring to it would silently mean
# something else. Same rule-5 exception as LARGE_MOVE_LOG_RETURN in GB-5, recorded in
# DECISIONS.md.
RSI_PERIOD = 14
VOL_Z_WINDOW = 20
MOMENTUM_LOOKBACK = 10
MA_DIST_WINDOW = 20

# The seed's weight after k further bars is (1 - 1/period) ** k. Hold rows NaN until that
# falls below this tolerance: 1% gives 63 bars past the seed, so 77 rows in total.
RSI_SEED_TOLERANCE = 0.01
RSI_WARMUP = RSI_PERIOD + math.ceil(
    math.log(RSI_SEED_TOLERANCE) / math.log(1 - 1 / RSI_PERIOD)
)


def rsi14(df: pd.DataFrame) -> pd.Series:
    """14-period Wilder RSI on close.

    Returns:
        A Series named ``rsi14``, in [0, 100], NaN for the first ``RSI_WARMUP`` rows.
    """
    close = _column(df, CLOSE)

    change = close.diff()
    gain = change.clip(lower=0.0)
    loss = -change.clip(upper=0.0)

    avg_gain = _wilder_average(gain, RSI_PERIOD)
    avg_loss = _wilder_average(loss, RSI_PERIOD)

    # Where there were no losses at all the ratio is undefined and RSI is 100 by
    # definition; where there were no gains either, the price never moved and 50 is the
    # neutral reading.
    rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.where((avg_loss != 0.0) | (avg_gain != 0.0), 50.0)

    rsi.iloc[:RSI_WARMUP] = np.nan
    return rsi.rename("rsi14")


def vol_z(df: pd.DataFrame) -> pd.Series:
    """Rolling 20-day z-score of volume, over a trailing window ending at t.

    The sample standard deviation (``ddof=1``) is used, matching pandas' default.

    Returns:
        A Series named ``vol_z``, NaN for the first ``VOL_Z_WINDOW - 1`` rows.
    """
    volume = _column(df, VOLUME).astype("float64")

    window = volume.rolling(VOL_Z_WINDOW)
    z = (volume - window.mean()) / window.std()
    return z.rename("vol_z")


def mom10(df: pd.DataFrame) -> pd.Series:
    """Simple 10-bar return: ``close / close.shift(10) - 1``.

    Returns:
        A Series named ``mom10``, NaN for the first ``MOMENTUM_LOOKBACK`` rows.
    """
    close = _column(df, CLOSE)
    return (close / close.shift(MOMENTUM_LOOKBACK) - 1.0).rename("mom10")


def ma_dist20(df: pd.DataFrame) -> pd.Series:
    """Fractional distance of close from its trailing 20-bar mean.

    Returns:
        A Series named ``ma_dist20``, NaN for the first ``MA_DIST_WINDOW - 1`` rows.
    """
    close = _column(df, CLOSE)
    moving_average = close.rolling(MA_DIST_WINDOW).mean()
    return ((close - moving_average) / moving_average).rename("ma_dist20")


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a column as float64, copied so the caller's frame is never touched."""
    if name not in df.columns:
        raise ValueError(f"the frame is missing the {name} column")
    return df[name].astype("float64").copy()


def _wilder_average(values: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothed average: a simple mean seed, then ``(prev * (n-1) + x) / n``.

    Written as an explicit recursion rather than an ``ewm`` call because Wilder's seeding
    is not what ``ewm`` does by default, and the difference is invisible in the output.
    """
    averaged = pd.Series(np.nan, index=values.index, dtype="float64")
    raw = values.to_numpy(dtype="float64")

    # values[0] is NaN because it came from diff(); the seed is the mean of the first
    # `period` real observations, which sit at positions 1..period.
    if len(raw) <= period:
        return averaged

    running = float(np.mean(raw[1 : period + 1]))
    averaged.iloc[period] = running
    for position in range(period + 1, len(raw)):
        running = (running * (period - 1) + raw[position]) / period
        averaged.iloc[position] = running

    return averaged
