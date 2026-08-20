"""GB-47/GB-48 acceptance: causal rolling DWT bands, and proof the guard has teeth.

Three claims, and the third is the one that matters.

**Additivity.** ``a3 + d3 + d2 + d1`` reconstructs the return series. A decomposition whose
parts cannot be summed back to the whole is one nobody can check, and the same argument
GB-30 makes about attribution applies here: the invariant is only evidence if the two
things compared are independent, so the reconstruction is compared against the *input*
rather than against a sum of the module's own outputs.

**Causality.** GB-10's harness, in **both** perturbation modes at **three** splits — reused,
not rewritten, because a second harness is a second thing that can be wrong.

**Teeth.** The standard recipe — transform the whole series once — is applied here on
purpose, and the harness must reject it. Without that, "the wavelets are causal" is a green
tick where an audit should have been. It is the same move as ``BatchNormForecaster`` for
spec §4.4's property 6.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from causality import SCALE, SHUFFLE, assert_causal

from glassbox.config.loader import Config, load_config
from glassbox.features import builder, wavelets

# 1600 bars, for `test_no_lookahead.py`'s reason: the harness splits the BAR frame at 25%,
# and `build_feature_frame` trims RSI's 325 warm-up rows, so 0.25 x BARS must clear 325 or
# the earliest split has no output behind it and the harness refuses the vacuous assertion
# rather than passing it. 1600 gives 400.
BARS = 1600
LEVELS = (1, 2, 3)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    base = load_config()
    return replace(
        base,
        data=replace(base.data, cache_dir=str(tmp_path / "cache")),
        channels=replace(base.channels, active="C2_hybrid"),
    )


@pytest.fixture
def bars() -> pd.DataFrame:
    """A path that rises, falls and changes sign often, so an order-sensitive leak shows."""
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
            # Varying, not constant: `vol_z` divides by a rolling standard deviation,
            # and a flat volume makes every one of its values NaN — which surfaces as
            # "NaNs remain after the warm-up trim" and looks like a wavelet defect.
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


# ── additivity: the parts sum back to the whole ─────────────────────────────


def test_the_bands_reconstruct_the_window(cfg: Config) -> None:
    """``a3 + d3 + d2 + d1 == window``, compared against the INPUT and not against a sum
    of the module's own outputs — the independence GB-30's tautology finding demands."""
    rng = np.random.default_rng(cfg.meta.seed)
    window = rng.normal(0, 0.015, size=cfg.wavelet.rolling_window)

    bands = wavelets.band_values(window, cfg)
    rebuilt = bands["a3"] + bands["d3"] + bands["d2"] + bands["d1"]

    assert np.abs(rebuilt - window).max() < 1e-12


def test_each_approximation_is_the_next_one_plus_its_detail(cfg: Config) -> None:
    """``A_j = A_{j+1} + D_{j+1}`` — the property that makes the ladder a ladder."""
    rng = np.random.default_rng(cfg.meta.seed + 1)
    window = rng.normal(0, 0.015, size=cfg.wavelet.rolling_window)

    bands = wavelets.band_values(window, cfg)

    for level in (1, 2):
        assert (
            np.abs(
                bands[f"a{level}"] - (bands[f"a{level + 1}"] + bands[f"d{level + 1}"])
            ).max()
            < 1e-12
        )


def test_additivity_holds_on_a_real_shaped_series(
    bars: pd.DataFrame, cfg: Config
) -> None:
    """The same check on the actual log-return series rather than on noise."""
    width = cfg.wavelet.rolling_window
    returns = bars["log_return"].to_numpy(copy=True)[1 : width + 1]

    bands = wavelets.band_values(returns, cfg)
    rebuilt = bands["a3"] + bands["d3"] + bands["d2"] + bands["d1"]

    assert np.abs(rebuilt - returns).max() < 1e-12


# ── causality, through GB-10's harness ──────────────────────────────────────


@pytest.mark.parametrize("level", LEVELS)
def test_the_rolling_approximation_is_causal(
    level: int, bars: pd.DataFrame, cfg: Config
) -> None:
    """Both modes, three splits. The harness's defaults are the point, not a shortcut."""

    def channel(frame: pd.DataFrame) -> pd.Series:
        return wavelets.approximation(frame, cfg, level=level)

    channel.__name__ = f"wav_a{level}"
    assert_causal(channel, bars)


def test_the_feature_frame_with_wavelets_is_causal(
    bars: pd.DataFrame, cfg: Config
) -> None:
    """The keystone assembling C2_hybrid, wavelets included."""

    def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
        return builder.build_feature_frame(frame, cfg)

    assert_causal(feature_frame, bars)


# ── teeth: the standard recipe, and the harness rejecting it ────────────────


@pytest.mark.parametrize("mode", (SCALE, SHUFFLE))
def test_the_whole_series_transform_leaks_and_the_harness_says_so(
    mode: str, bars: pd.DataFrame, cfg: Config
) -> None:
    """**The test that makes the other tests mean something.**

    A DWT filter is two-sided, so a whole-series transform puts information from after
    ``t`` into the value at ``t``. It is the recipe most of the literature uses, it looks
    like a smoothing, and every downstream metric improves. Both perturbation modes catch
    it, which is worth asserting separately: a leak that only one mode sees would be a
    reason to doubt the other.
    """

    def leaking(frame: pd.DataFrame) -> pd.Series:
        return wavelets.whole_series_approximation(frame, cfg, level=3)

    leaking.__name__ = "whole_series_wav_a3"

    with pytest.raises(AssertionError, match="LOOKS AHEAD"):
        assert_causal(leaking, bars, modes=(mode,))


def test_the_two_versions_differ_on_the_same_data(
    bars: pd.DataFrame, cfg: Config
) -> None:
    """Otherwise the previous test would be rejecting a function nobody would write.

    They must be genuinely different computations, not the same one wearing two names.
    """
    causal = wavelets.approximation(bars, cfg, level=3)
    leaking = wavelets.whole_series_approximation(bars, cfg, level=3)

    both = causal.notna() & leaking.notna()
    assert both.sum() > 100
    assert not np.allclose(causal[both], leaking[both])


# ── the rolling contract ────────────────────────────────────────────────────


@pytest.mark.parametrize("level", LEVELS)
def test_values_begin_at_the_rolling_window_and_never_before(
    level: int, bars: pd.DataFrame, cfg: Config
) -> None:
    """§8's acceptance for GB-47: emits values from bar 64 onward.

    Row 0 of the log-return series is NaN by construction, so the first fully populated
    window ends at row ``rolling_window`` — 64 — and not at ``rolling_window - 1``.
    """
    series = wavelets.approximation(bars, cfg, level=level)
    width = cfg.wavelet.rolling_window

    assert series.iloc[:width].isna().all()
    assert series.iloc[width:].notna().all()
    assert int(series.notna().to_numpy().argmax()) == width


@pytest.mark.parametrize("level", LEVELS)
def test_the_value_at_t_is_the_last_sample_of_its_own_window(
    level: int, bars: pd.DataFrame, cfg: Config
) -> None:
    """Recomputed independently, from the window alone, at several bars.

    This is what "causal" means concretely: hand the function the same 64 bars in any
    other context and it must produce the same number.
    """
    series = wavelets.approximation(bars, cfg, level=level)
    returns = bars["log_return"].to_numpy(copy=True)
    width = cfg.wavelet.rolling_window

    for end in (width, width + 1, 300, len(returns) - 1):
        window = returns[end - width + 1 : end + 1]
        expected = wavelets.band_values(window, cfg)[f"a{level}"][-1]
        assert series.iloc[end] == pytest.approx(expected, abs=1e-15)


def test_the_bars_frame_is_never_mutated(bars: pd.DataFrame, cfg: Config) -> None:
    """Purity, spec §3: DataFrame in, Series out, nothing touched."""
    before = bars.copy(deep=True)

    wavelets.approximation(bars, cfg, level=2)

    pd.testing.assert_frame_equal(bars, before)


def test_a_level_outside_the_configured_count_is_refused(
    bars: pd.DataFrame, cfg: Config
) -> None:
    for level in (0, cfg.wavelet.levels + 1):
        with pytest.raises(ValueError, match="outside the configured"):
            wavelets.approximation(bars, cfg, level=level)


def test_a_window_with_a_nan_is_refused_rather_than_filled(cfg: Config) -> None:
    window = np.full(cfg.wavelet.rolling_window, 0.01)
    window[3] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        wavelets.band_values(window, cfg)


# ── what landing the wavelets did and did not change ────────────────────────


def test_c2_hybrid_now_builds(bars: pd.DataFrame, cfg: Config) -> None:
    """It raised until GB-47. The channel set is the study's hybrid arm."""
    frame = builder.build_feature_frame(bars, cfg)

    assert list(frame.columns) == list(cfg.channels.active_channels)
    assert set(wavelets.APPROXIMATION_CHANNELS) <= set(frame.columns)
    assert not frame.isna().to_numpy().any()


def test_the_wavelets_do_not_move_the_history_floor(cfg: Config) -> None:
    """Measured, and it is the opposite of what was expected.

    The wavelet warm-up is 64 bars and RSI's is 325, so ``min_history_bars`` stays at
    **445** for both channel sets. The floor is a maximum, not a sum, and Wilder's
    recursion still dominates.

    The two warm-ups are also different *kinds* of number, which is worth keeping visible:
    RSI's 325 is a tolerance argument about how fast a seed decays, re-derived in GB-27
    after the first derivation proved wrong. The wavelets' 64 is exact — a DWT of a
    trailing window depends on that window and on nothing before it.
    """
    base = replace(cfg, channels=replace(cfg.channels, active="C0_base"))

    assert builder.min_history_bars(cfg) == builder.min_history_bars(base) == 445
    assert builder.deepest_warmup_channel(cfg) == "rsi14"
    assert "rsi14" in builder.history_requirement(cfg)


def test_the_warmup_table_follows_the_configuration(cfg: Config) -> None:
    """The wavelet entry is a function of ``cfg``, not a literal 64 beside the setting.

    Widening the rolling window has to move the floor, or the two would disagree the first
    time anybody tuned it — and the disagreement would surface as a live window quietly
    built from short history.
    """
    wider = replace(cfg, wavelet=replace(cfg.wavelet, rolling_window=400))

    assert builder.min_history_bars(wider) == cfg.window.input_len + 400
    # `wav_a3` rather than `wav_a1`: the three share a warm-up and the tie breaks on the
    # channel name, so the answer does not depend on config ordering.
    assert builder.deepest_warmup_channel(wider) == "wav_a3"
