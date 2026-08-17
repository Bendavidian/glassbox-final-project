"""GB-10 acceptance: nothing in the feature path reaches past its own timestamp.

Every indicator, the keystone's feature frame and the keystone's windows are perturbed in
the future and asserted unchanged in the past, at three split points and in two modes.

The last two tests are the ones that matter most. A causality suite that cannot fail is
worse than no suite, because it produces a green tick where an audit should have been, so
this file also carries deliberately leaky functions and asserts the harness rejects them —
one leak per mode, which is how the two modes earn their place.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from causality import SCALE, SHUFFLE, assert_causal, assert_fit_isolated, perturb

from glassbox.config.loader import Config, load_config
from glassbox.contracts.schemas import ChannelStats
from glassbox.features import builder, indicators

# 1600, not 500: RSI's warm-up takes 325 rows since GB-27, and the harness splits the BAR
# frame at 25/50/75%. The earliest split must have WINDOW output behind it, and the first
# window ends at bar 354 (325 of warm-up plus the 30-bar window), so 0.25 x BARS must clear
# 354. 1600 gives 400.
BARS = 1600
INPUT_LEN = 30
HORIZON = 4

INDICATORS = (
    indicators.rsi14,
    indicators.vol_z,
    indicators.mom10,
    indicators.ma_dist20,
)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """A smaller window than production, so 500 bars reach past the earliest split."""
    base = load_config()
    return replace(
        base,
        data=replace(base.data, cache_dir=str(tmp_path / "cache")),
        window=replace(base.window, input_len=INPUT_LEN, horizon=HORIZON),
    )


@pytest.fixture
def bars() -> pd.DataFrame:
    """A canonical bar frame that rises, falls and changes sign often.

    A monotone series would let an order-sensitive leak hide, because permuting the future
    of a straight line barely disturbs it.
    """
    index = pd.date_range("2020-01-01", periods=BARS, freq="B", tz="UTC")
    close = pd.Series(
        [
            100.0 + 10.0 * math.sin(i / 7.0) + 4.0 * math.cos(i / 2.3) + 0.05 * i
            for i in range(BARS)
        ],
        index=index,
        dtype="float64",
    )
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": pd.Series(
                [
                    1_000_000.0 + 5_000.0 * math.cos(i / 5.0) + 100.0 * i
                    for i in range(BARS)
                ],
                index=index,
            ),
            "log_return": np.log(close / close.shift(1)),
            "source": pd.Series("yfinance", index=index, dtype="string"),
        },
        index=index,
    )


# ── the real functions ───────────────────────────────────────────────────────


@pytest.mark.parametrize("indicator", INDICATORS, ids=lambda fn: fn.__name__)
def test_every_indicator_is_causal(indicator, bars: pd.DataFrame) -> None:
    assert_causal(indicator, bars)


def test_build_feature_frame_is_causal(bars: pd.DataFrame, cfg: Config) -> None:
    def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
        return builder.build_feature_frame(frame, cfg)

    assert_causal(feature_frame, bars)


def test_build_windows_is_causal(bars: pd.DataFrame, cfg: Config) -> None:
    """A window ending at or before the split contains nothing from after it."""
    assert_causal(_windows_as_frame(cfg), bars)


def test_fit_stats_uses_only_the_training_range(
    bars: pd.DataFrame, cfg: Config
) -> None:
    """The other leakage surface: statistics must not see past their own boundary."""
    frame = builder.build_feature_frame(bars, cfg)
    assert_fit_isolated(_fit_on_training(cfg), frame)


def test_fitted_range_matches_the_rows_actually_used(
    bars: pd.DataFrame, cfg: Config
) -> None:
    """GB-25 audits leakage by intersecting this range with the fold's test range.

    That audit is only as good as the range's honesty, so assert it here: the recorded
    range must be the training slice itself, not the frame the fitter was handed.
    """
    frame = builder.build_feature_frame(bars, cfg)
    split = frame.index[len(frame) // 2]

    stats = _fit_on_training(cfg)(frame, split)
    training = frame.loc[:split]

    assert stats.fitted_start == training.index[0]
    assert stats.fitted_end == split
    assert stats.n_rows == len(training)


# ── the harness has teeth ────────────────────────────────────────────────────


def test_the_harness_rejects_a_centred_window(bars: pd.DataFrame) -> None:
    """The canonical accident: a centred rolling mean averages the future into today."""
    with pytest.raises(AssertionError, match="LOOKS AHEAD"):
        assert_causal(leaky_centred_mean, bars)


def test_the_harness_rejects_tomorrows_close(bars: pd.DataFrame) -> None:
    """The blatant one: the value at t is simply the bar after t."""
    with pytest.raises(AssertionError, match="LOOKS AHEAD"):
        assert_causal(leaky_tomorrows_close, bars)


def test_the_harness_rejects_a_fitter_that_forgets_to_slice(
    bars: pd.DataFrame, cfg: Config
) -> None:
    """The leak GB-25 exists to find: statistics fitted on the whole series."""
    frame = builder.build_feature_frame(bars, cfg)
    with pytest.raises(AssertionError, match="is not isolated"):
        assert_fit_isolated(_fit_on_everything(cfg), frame)


# ── the two modes are complementary, and neither alone is enough ─────────────


def test_scale_alone_cannot_see_a_leaked_direction(bars: pd.DataFrame) -> None:
    """``sign(1.5 * x) == sign(x)``, exactly.

    Multiplying the future by a positive scalar leaves the *sign* of every future value
    untouched, so a function leaking tomorrow's direction — which spec §1.4 names as one of
    the two headline metrics — passes a purely multiplicative test. This is the whole
    argument for the second mode, and it is asserted rather than claimed.
    """
    assert_causal(leaky_tomorrows_direction, bars, modes=(SCALE,))

    with pytest.raises(AssertionError, match="LOOKS AHEAD"):
        assert_causal(leaky_tomorrows_direction, bars, modes=(SHUFFLE,))


def test_shuffle_carries_no_evidence_about_a_leaked_mean(
    bars: pd.DataFrame, cfg: Config
) -> None:
    """A permutation preserves the multiset, so it cannot move a full-sample mean.

    The blindness runs the other way too, and here it has to be measured rather than
    asserted as a pass: the comparison is exact and floating-point addition is not
    associative, so reordering the summands does shift the mean in its last ulp. That
    shift is ~1e-16 relative — arithmetic noise, not evidence of a leak — where the scale
    mode moves the same statistic by tens of percent. Neither mode subsumes the other,
    which is why the default is both.
    """
    frame = builder.build_feature_frame(bars, cfg)
    split = frame.index[len(frame) // 2]
    baseline = builder.fit_stats(frame, cfg)

    shuffled = builder.fit_stats(perturb(frame, split, SHUFFLE, 1.5), cfg)
    scaled = builder.fit_stats(perturb(frame, split, SCALE, 1.5), cfg)

    assert _largest_relative_shift(shuffled, baseline) < 1e-12
    assert _largest_relative_shift(scaled, baseline) > 0.1


# ── the harness refuses to pass vacuously ────────────────────────────────────


def test_a_perturbation_that_changes_nothing_is_refused(bars: pd.DataFrame) -> None:
    """A scale factor of 1.0 leaves the input identical; every function would 'pass'."""
    with pytest.raises(ValueError, match="proves nothing"):
        assert_causal(indicators.mom10, bars, perturbation=1.0, modes=(SCALE,))


def test_an_empty_comparison_is_refused(bars: pd.DataFrame, cfg: Config) -> None:
    """Splitting before the first window leaves nothing to compare."""
    frame = builder.build_feature_frame(bars, cfg)
    with pytest.raises(ValueError, match="would pass vacuously"):
        assert_causal(_windows_as_frame(cfg), bars, split_at=frame.index[INPUT_LEN - 2])


def test_an_unknown_mode_is_refused(bars: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown perturbation mode"):
        assert_causal(indicators.mom10, bars, modes=("jitter",))


# ── helpers ──────────────────────────────────────────────────────────────────


def _windows_as_frame(cfg: Config):
    """Adapt ``build_windows`` to the harness: one row per window, indexed by its end.

    ``y`` is deliberately excluded. The target is the future by definition, so feeding it
    to a causality assertion would fail a correct implementation.
    """

    def windows(frame: pd.DataFrame) -> pd.DataFrame:
        features = builder.build_feature_frame(frame, cfg)
        batch = builder.build_windows(features, cfg, "TEST")
        flat = batch.X.reshape(len(batch.timestamps), -1)
        return pd.DataFrame(flat, index=batch.timestamps)

    windows.__name__ = "build_windows"
    return windows


def _fit_on_training(cfg: Config):
    """The correct fitter: given the whole frame, it uses only the training slice."""

    def fit(frame: pd.DataFrame, split_at: pd.Timestamp):
        return builder.fit_stats(frame.loc[:split_at], cfg)

    fit.__name__ = "fit_stats on the training slice"
    return fit


def _fit_on_everything(cfg: Config):
    """The leak: fits on every row it can see, then labels itself with the split."""

    def fit(frame: pd.DataFrame, split_at: pd.Timestamp):
        del split_at
        return builder.fit_stats(frame, cfg)

    fit.__name__ = "fit_stats on the whole frame"
    return fit


def _largest_relative_shift(observed: ChannelStats, baseline: ChannelStats) -> float:
    """The biggest relative move across every mean and every standard deviation."""
    before = np.array(baseline.mean + baseline.std, dtype="float64")
    after = np.array(observed.mean + observed.std, dtype="float64")
    return float(np.max(np.abs(after - before) / np.abs(before)))


def leaky_centred_mean(frame: pd.DataFrame) -> pd.Series:
    """A 20-bar mean centred on t, so half its window is in the future."""
    return frame["close"].rolling(20, center=True).mean()


def leaky_tomorrows_close(frame: pd.DataFrame) -> pd.Series:
    """Tomorrow's close, stamped today."""
    return frame["close"].shift(-1)


def leaky_tomorrows_direction(frame: pd.DataFrame) -> pd.Series:
    """The sign of tomorrow's return, stamped today — invisible to a positive multiply."""
    return np.sign(frame["log_return"].shift(-1))
