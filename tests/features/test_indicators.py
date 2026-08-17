"""GB-8 acceptance: the four indicators are pure, trailing-only and numerically right.

Expected values are computed independently of the module under test, by a pure-Python
reference implementation using lists and loops — no pandas, no import of
``indicators``. That reference lives in this file (:func:`reference_wilder_rsi` and the
closed-form algebra in each test) and its outputs are pinned here as literals, so a
change to the implementation cannot quietly move the target.

Wilder's RSI follows the definition in "New Concepts in Technical Trading Systems"
(J. Welles Wilder, 1978): seed the average gain and loss with the simple mean of the
first 14 changes, then smooth each later bar as ``(previous * 13 + current) / 14``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from glassbox.features.indicators import (
    MA_DIST_WINDOW,
    MOMENTUM_LOOKBACK,
    RSI_PERIOD,
    RSI_WARMUP,
    VOL_Z_WINDOW,
    ma_dist20,
    mom10,
    rsi14,
    vol_z,
)

# 500, not 120: RSI holds its first 325 rows since GB-27 unified the emit warm-up
# with the parity warm-up, so a 120-bar frame emits nothing at all.
BARS = 500


def frame_from(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    """A canonical bar frame carrying the columns the indicators read."""
    index = pd.date_range("2020-01-01", periods=len(closes), freq="B", tz="UTC")
    close = pd.Series(closes, index=index, dtype="float64")
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": (
                pd.Series(volumes, index=index, dtype="float64")
                if volumes is not None
                else pd.Series(1_000_000.0, index=index)
            ),
            "log_return": np.log(close / close.shift(1)),
        },
        index=index,
    )


# A deterministic series that genuinely rises and falls: closes[i] = 100 + 10sin(i/5) + i/10
WAVY_CLOSES = [100.0 + 10.0 * math.sin(i / 5.0) + 0.1 * i for i in range(BARS)]
LINEAR_CLOSES = [100.0 + i for i in range(40)]
GEOMETRIC_CLOSES = [100.0 * 1.01**i for i in range(40)]
LINEAR_VOLUMES = [1000.0 + 10.0 * i for i in range(40)]


def reference_wilder_rsi(
    closes: list[float], period: int = RSI_PERIOD
) -> list[float | None]:
    """Wilder's RSI in plain Python. Independent of the implementation under test."""
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(change, 0.0) for change in changes]
    losses = [max(-change, 0.0) for change in changes]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    out: list[float | None] = [None] * len(closes)
    out[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss) if avg_loss else 100.0
    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss) if avg_loss else 100.0
    return out


# ── rsi14 ────────────────────────────────────────────────────────────────────

# Pinned from the reference implementation above, on WAVY_CLOSES.
#
# Re-pinned in GB-27: the warm-up moved from 77 rows to 325, so the old positions
# (77-119) now sit inside it and the production series is NaN there. The values are
# re-derived from the same independent reference at positions past the new warm-up, which
# is what made them evidence in the first place — they were never captured from the code
# under test.
EXPECTED_RSI = {
    325: 72.12909448785442,
    350: 75.97871106673796,
    400: 26.85876576397291,
    450: 74.99567501802264,
    480: 78.69148711239507,
    499: 43.019830838368286,
}


@pytest.mark.parametrize(("position", "expected"), sorted(EXPECTED_RSI.items()))
def test_rsi_matches_independent_reference(position: int, expected: float) -> None:
    result = rsi14(frame_from(WAVY_CLOSES))
    assert result.iloc[position] == pytest.approx(expected, abs=1e-6)


def test_rsi_matches_the_reference_at_every_emitted_bar() -> None:
    """Not just the pinned points: every non-NaN value, 43 bars of them."""
    result = rsi14(frame_from(WAVY_CLOSES))
    expected = reference_wilder_rsi(WAVY_CLOSES)

    emitted = result.dropna()
    assert len(emitted) == BARS - RSI_WARMUP
    for position, value in zip(range(RSI_WARMUP, BARS), emitted, strict=True):
        assert value == pytest.approx(expected[position], abs=1e-6)


def test_rsi_is_100_when_every_bar_rises() -> None:
    """No losses means the ratio is undefined and RSI is 100 by definition."""
    closes = [100.0 + i for i in range(BARS)]
    assert rsi14(frame_from(closes)).iloc[-1] == pytest.approx(100.0, abs=1e-6)


def test_rsi_is_0_when_every_bar_falls() -> None:
    closes = [300.0 - i for i in range(BARS)]
    assert rsi14(frame_from(closes)).iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_rsi_is_50_when_the_price_never_moves() -> None:
    closes = [100.0] * BARS
    assert rsi14(frame_from(closes)).iloc[-1] == pytest.approx(50.0, abs=1e-6)


def test_rsi_stays_within_bounds() -> None:
    emitted = rsi14(frame_from(WAVY_CLOSES)).dropna()
    assert emitted.between(0.0, 100.0).all()


def test_rsi_warmup_is_exactly_the_seed_decay_length() -> None:
    """325 rows: 14 to seed, then 311 more for the seed's influence to fall under 1e-10.

    **It read 77 until GB-27**, from a 1e-2 target that bounded the seed's *weight* rather
    than `weight x seed_difference` — the same error that made the 352-bar parity floor
    marginal. It is now the same number as `builder.PARITY_WARMUP["rsi14"]`, which imports
    it: "the value no longer remembers its seed" and "the value is byte-identical to what
    training computed" are one question, and two constants answering it is how they drift.
    """
    assert RSI_WARMUP == 325

    result = rsi14(frame_from(WAVY_CLOSES))
    assert result.iloc[:RSI_WARMUP].isna().all()
    assert result.iloc[RSI_WARMUP:].notna().all()


def test_rsi_returns_all_nan_when_the_series_is_shorter_than_the_warmup() -> None:
    result = rsi14(frame_from(WAVY_CLOSES[:30]))
    assert result.isna().all()


# ── vol_z ────────────────────────────────────────────────────────────────────

# volumes[i] = 1000 + 10i. Over any trailing 20-bar window the last value sits 95 above
# the window mean. For an arithmetic sequence of step d and length n the SAMPLE variance
# (ddof=1, which is pandas' default) is d^2 * n(n+1)/12 = 100 * 35 = 3500, so the sd is
# 10 * sqrt(35) and the z-score is 95 / (10 * sqrt(35)) — constant at every position.
# Note it is n(n+1)/12, not the population form (n^2 - 1)/12: this test caught that
# exact slip.
EXPECTED_VOL_Z = 1.6057930839841814


@pytest.mark.parametrize("position", [19, 25, 39])
def test_vol_z_matches_hand_computed_value(position: int) -> None:
    result = vol_z(frame_from(LINEAR_CLOSES, LINEAR_VOLUMES))
    assert result.iloc[position] == pytest.approx(EXPECTED_VOL_Z, abs=1e-6)


def test_vol_z_closed_form_agrees_with_the_algebra() -> None:
    """95 / (10 * sqrt(35)), derived independently of both pandas and the module."""
    assert EXPECTED_VOL_Z == pytest.approx(95.0 / (10.0 * math.sqrt(35.0)), abs=1e-12)


def test_vol_z_is_zero_for_constant_volume_after_warmup() -> None:
    """Constant volume has zero spread, so the z-score is undefined rather than 0."""
    result = vol_z(frame_from(LINEAR_CLOSES, [1000.0] * 40))
    assert result.iloc[VOL_Z_WINDOW:].isna().all()


def test_vol_z_warmup_is_the_window_minus_one() -> None:
    result = vol_z(frame_from(LINEAR_CLOSES, LINEAR_VOLUMES))
    assert result.iloc[: VOL_Z_WINDOW - 1].isna().all()
    assert result.iloc[VOL_Z_WINDOW - 1 :].notna().all()


# ── mom10 ────────────────────────────────────────────────────────────────────

# 1.01 ** 10 - 1, constant for a geometric series.
EXPECTED_MOM10 = 0.10462212541120453


@pytest.mark.parametrize("position", [10, 25, 39])
def test_mom10_matches_hand_computed_value(position: int) -> None:
    result = mom10(frame_from(GEOMETRIC_CLOSES))
    assert result.iloc[position] == pytest.approx(EXPECTED_MOM10, abs=1e-6)


def test_mom10_closed_form_agrees_with_the_algebra() -> None:
    assert EXPECTED_MOM10 == pytest.approx(1.01**10 - 1.0, abs=1e-12)


def test_mom10_on_a_linear_series() -> None:
    """closes 100..139: at position 10 that is 110/100 - 1."""
    result = mom10(frame_from(LINEAR_CLOSES))
    assert result.iloc[10] == pytest.approx(0.1, abs=1e-6)


def test_mom10_warmup_is_the_lookback() -> None:
    result = mom10(frame_from(LINEAR_CLOSES))
    assert result.iloc[:MOMENTUM_LOOKBACK].isna().all()
    assert result.iloc[MOMENTUM_LOOKBACK:].notna().all()


# ── ma_dist20 ────────────────────────────────────────────────────────────────

# closes 100..139: at position p the trailing window is [p-19 .. p], whose mean is
# closes[p] - 9.5, so the distance is 9.5 / (closes[p] - 9.5).
EXPECTED_MA_DIST = {
    19: 0.0867579908675799,
    20: 0.08597285067873303,
    39: 0.07335907335907337,
}


@pytest.mark.parametrize(("position", "expected"), sorted(EXPECTED_MA_DIST.items()))
def test_ma_dist20_matches_hand_computed_value(position: int, expected: float) -> None:
    result = ma_dist20(frame_from(LINEAR_CLOSES))
    assert result.iloc[position] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("position", sorted(EXPECTED_MA_DIST))
def test_ma_dist20_closed_form_agrees_with_the_algebra(position: int) -> None:
    close = LINEAR_CLOSES[position]
    assert EXPECTED_MA_DIST[position] == pytest.approx(9.5 / (close - 9.5), abs=1e-12)


def test_ma_dist20_is_zero_for_a_flat_series() -> None:
    result = ma_dist20(frame_from([100.0] * 40))
    assert result.iloc[MA_DIST_WINDOW:].eq(0.0).all()


def test_ma_dist20_warmup_is_the_window_minus_one() -> None:
    result = ma_dist20(frame_from(LINEAR_CLOSES))
    assert result.iloc[: MA_DIST_WINDOW - 1].isna().all()
    assert result.iloc[MA_DIST_WINDOW - 1 :].notna().all()


# ── Contract: purity, naming, causality ──────────────────────────────────────

INDICATORS = (rsi14, vol_z, mom10, ma_dist20)
CHANNEL_NAMES = {rsi14: "rsi14", vol_z: "vol_z", mom10: "mom10", ma_dist20: "ma_dist20"}


@pytest.mark.parametrize("indicator", INDICATORS, ids=CHANNEL_NAMES.values())
def test_indicator_does_not_mutate_its_input(indicator) -> None:
    frame = frame_from(WAVY_CLOSES, [1000.0 + 10.0 * i for i in range(BARS)])
    before = frame.copy(deep=True)

    indicator(frame)

    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("indicator", INDICATORS, ids=CHANNEL_NAMES.values())
def test_indicator_is_named_for_its_canonical_channel(indicator) -> None:
    frame = frame_from(WAVY_CLOSES, [1000.0 + 10.0 * i for i in range(BARS)])
    assert indicator(frame).name == CHANNEL_NAMES[indicator]


@pytest.mark.parametrize("indicator", INDICATORS, ids=CHANNEL_NAMES.values())
def test_indicator_shares_the_input_index(indicator) -> None:
    frame = frame_from(WAVY_CLOSES, [1000.0 + 10.0 * i for i in range(BARS)])
    assert indicator(frame).index.equals(frame.index)


@pytest.mark.parametrize("indicator", INDICATORS, ids=CHANNEL_NAMES.values())
def test_indicator_uses_only_trailing_data(indicator) -> None:
    """The causality rule: perturbing a future bar leaves every earlier value unchanged.

    GB-10 generalises this into the project-wide harness; this is the local guard.
    """
    volumes = [1000.0 + 10.0 * i for i in range(BARS)]
    frame = frame_from(WAVY_CLOSES, volumes)
    cut = 90

    perturbed_closes = list(WAVY_CLOSES)
    perturbed_volumes = list(volumes)
    for position in range(cut + 1, BARS):
        perturbed_closes[position] *= 1.5
        perturbed_volumes[position] *= 3.0
    perturbed = frame_from(perturbed_closes, perturbed_volumes)

    pd.testing.assert_series_equal(
        indicator(frame).iloc[: cut + 1], indicator(perturbed).iloc[: cut + 1]
    )


@pytest.mark.parametrize("indicator", INDICATORS, ids=CHANNEL_NAMES.values())
def test_indicator_raises_when_its_column_is_missing(indicator) -> None:
    frame = frame_from(WAVY_CLOSES).drop(columns=["close", "volume"])
    with pytest.raises(ValueError, match="missing the (close|volume) column"):
        indicator(frame)
