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
held NaN until the seed's influence falls below ``RSI_SEED_TOLERANCE``.

**That tolerance was 1e-2, giving 77 rows, and GB-27 deleted it.** Two things were wrong
with it. It bounded the seed's **weight** rather than ``weight x seed_difference``, which
is the quantity that has to fall below resolution - the same error that made the parity
floor of 352 bars marginal. And it answered a question the parity warm-up in
``builder.PARITY_WARMUP`` was already answering, with a different number: "meaningful"
means "independent of the seed", and so does "byte-identical to what training computed".
**Two names for one quantity, and one of them had a known-bad derivation.** They are now a
single number - 1e-10, 325 rows - so there is one derivation to argue with instead of two
to keep in step.

The cost is small and it is **not** zero, measured rather than asserted. Trimming 325 rows
instead of 77 takes the feature frame from 2591 rows to 2343 and moves its start from
2016-04-25 to 2017-04-19; everything discarded is 2016-2017, and the earliest kept fold
begins training in April 2020, so no kept fold reads a discarded bar. But ``make_folds``
anchors its calendar grid on the feature frame's start, so the grid itself shifted by about
a week - fold 1's training now begins 2020-04-06 rather than 2020-04-13 - and individual
validation and test window counts move by one or two. **Every number measured before this
change therefore shifts slightly**, because the folds are not the same folds; the headline
figures were re-measured rather than carried over.

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

# The seed's weight after k further bars is (1 - 1/period) ** k. Rows are held NaN until
# the seed's INFLUENCE - weight times the seed difference, not weight alone - falls below
# this tolerance.
#
# 1e-10, not the 1e-2 this held until GB-27. The old target bounded the weight only, which
# is sufficient only if the seed difference is at most 1; it is a difference of average
# gains, in price units. Near RSI 50 a float32 ulp is 3.8147e-06
# (`np.spacing(np.float32(50))`, 2**-18), and the residual GB-27 measured was 3.815e-06:
# **one ulp**, so the observed failure was a single-ulp rounding flip. (Until 5 Oct 2026
# this comment gave the ulp as 5.95e-06, which is 100 x 2**-24, not the spacing at 50.)
# 1e-10 leaves three orders of magnitude of margin over the value's own ulp.
#
# This is deliberately the SAME number as `builder.PARITY_WARMUP["rsi14"]`, which imports
# it: "the value no longer remembers its seed" and "the value is byte-identical to what
# training computed" are one question, and answering it twice with two constants is how
# they drift apart.
RSI_SEED_TOLERANCE = 1e-10
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
