"""GB-45 acceptance: the per-frequency decomposition, exact and keyed in days.

The channel view says *which input series* drove a forecast. This one says *which cycle*,
and it is the view §6.5 builds the demonstration's strongest visual out of. The properties
that matter are the same ones §4.4 states for channels — the parts sum to the whole, and
the check is capable of failing — plus two that are specific to a spectral model: the RIN
mean is part of the forecast and belongs to no frequency, and bin 0 contributes nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from glassbox.config.loader import Config, load_config
from glassbox.contracts.schemas import Attribution
from glassbox.explain import spectral
from glassbox.model import ALL_FORECASTERS
from glassbox.model.fits import FITSForecaster, amplitude_scale

WINDOWS = 200


@pytest.fixture
def cfg() -> Config:
    return load_config()


@pytest.fixture
def channels(cfg: Config) -> tuple[str, ...]:
    return cfg.channels.active_channels


@pytest.fixture
def model(cfg: Config, channels: tuple[str, ...]) -> FITSForecaster:
    """A model with a non-trivial weight, set directly rather than fitted.

    Fitting would work and would be slower and less controlled; every property here is a
    property of the decomposition, not of training, and a weight drawn from a seeded
    generator exercises the arithmetic harder than an early-stopped one does.
    """
    built = ALL_FORECASTERS["fits"](cfg, channels)
    rng = np.random.default_rng(cfg.meta.seed)
    shape = (built.cof, built.out_bins)
    built._set_weight(
        rng.normal(0, 0.2, size=shape) + 1j * rng.normal(0, 0.2, size=shape)
    )
    return built


@pytest.fixture
def windows(cfg: Config, channels: tuple[str, ...]) -> np.ndarray:
    rng = np.random.default_rng(cfg.meta.seed + 5)
    shape = (WINDOWS, cfg.window.input_len, len(channels))
    return rng.normal(0, 0.012, size=shape).astype("float32")


def sinusoid(cfg: Config, channels: tuple[str, ...], period: float) -> np.ndarray:
    """One window carrying a single cycle in the close column and noise elsewhere."""
    t = np.arange(cfg.window.input_len, dtype="float64")
    window = np.zeros((cfg.window.input_len, len(channels)), dtype="float32")
    window[:, channels.index("close_logret")] = 0.01 * np.sin(2 * math.pi * t / period)
    return window


def identity_weight(model: FITSForecaster) -> np.ndarray:
    """The frequency-preserving map: bin ``k`` in, bin ``round(η·k)`` out, gain 1."""
    eta = amplitude_scale(model.input_len, model.horizon)
    weight = np.zeros((model.cof, model.out_bins), dtype="complex128")
    for index in range(model.cof):
        weight[index, min(round(index * eta), model.out_bins - 1)] = 1.0
    return weight


# ── the decomposition is the forward pass, split ────────────────────────────


def test_the_frequency_maps_and_the_mean_sum_to_the_forecast_map(
    model: FITSForecaster,
) -> None:
    """The whole decomposition, before any window touches it.

    Every part is measured by running the real forward pass with one bin unmasked, so this
    asserts that masking partitions the pipeline rather than approximating it. A residual
    here would be a defect in the model, not in the explanation.
    """
    parts = model.mean_matrix() + sum(model.frequency_matrices().values())

    assert np.abs(parts - model.forecast_matrix()).max() < 1e-12


def test_the_contributions_sum_to_the_forecast_over_many_windows(
    model: FITSForecaster, windows: np.ndarray
) -> None:
    """§4.4's exactness, on the frequency view, and capable of failing.

    The total comes from ``predict`` rather than from the decomposition's own sum, which is
    the correction GB-30 made after property 4 spent a week comparing a number to itself.
    """
    worst = 0.0
    for window in windows:
        view = spectral.per_frequency(model, window)
        predicted = float(model.predict(window[None, ...])[0].sum())
        worst = max(worst, abs(sum(view.values()) - predicted))

    assert worst < 1e-7, f"worst residual {worst:.3e} over {len(windows)} windows"


def test_the_rin_mean_is_part_of_the_forecast_and_belongs_to_no_frequency(
    model: FITSForecaster, windows: np.ndarray
) -> None:
    """Drop the mean term and the decomposition must fail to close.

    This is the check that the mean is genuinely carried rather than incidentally small:
    ``Attribution.from_terms`` names this case in its own error message, and the test makes
    the error happen instead of trusting the message.
    """
    window = windows[0]
    closes = window[:, model.channels.index("close_logret")].astype("float64")
    predicted = float(model.predict(window[None, ...])[0].sum())

    without_the_mean = {
        f"period_{index}": ((matrix, closes),)
        for index, matrix in model.frequency_matrices().items()
    }

    with pytest.raises(ValueError, match="does not close"):
        Attribution.from_terms(without_the_mean, predicted)


def test_bin_zero_contributes_identically_zero(model: FITSForecaster) -> None:
    """The dead row of GB-44, stated in the decomposition rather than left to be wondered at.

    RIN removes the window mean and the rFFT's bin 0 *is* that mean, so row 0 of the weight
    multiplies nothing on any window. It is reported as a measured zero, not dropped.
    """
    assert np.abs(model.frequency_matrices()[0]).max() < 1e-15

    # And the infinite-period entry is therefore exactly the mean path.
    window = np.zeros((model.input_len, len(model.channels)), dtype="float32")
    window[:, model.channels.index("close_logret")] = 0.004
    view = spectral.per_frequency(model, window)

    # rel=1e-6, not tighter: the window is float32, so 0.004 is 0.0040000001 before
    # the decomposition sees it, and a float64 tolerance would be measuring the cast.
    assert view[spectral.DC_PERIOD] == pytest.approx(0.004 * model.horizon, rel=1e-6)


# ── periods, gains and phases ───────────────────────────────────────────────


def test_periods_are_in_days_not_bin_indices(cfg: Config) -> None:
    """`L / k`, with the zero-frequency component at an infinite period."""
    assert spectral.period_days(0, 120) == math.inf
    assert spectral.period_days(1, 120) == 120.0
    assert spectral.period_days(10, 120) == 12.0
    assert spectral.period_days(24, 120) == 5.0

    with pytest.raises(ValueError, match="must not be negative"):
        spectral.period_days(-1, 120)


def test_a_single_cycle_is_attributed_to_its_own_period(
    model: FITSForecaster, cfg: Config, channels: tuple[str, ...]
) -> None:
    """The claim the demonstration makes out loud: *this forecast is the 12-day cycle*.

    With the frequency-preserving weight, a window carrying one cycle has exactly one
    non-zero input bin, so every learned contribution must land on that period and the rest
    must be zero. It is the strongest available check that the keys mean what they say.
    """
    model._set_weight(identity_weight(model))
    window = sinusoid(cfg, channels, period=12.0)

    view = spectral.per_frequency(model, window)
    learned = {period: value for period, value in view.items() if math.isfinite(period)}

    assert max(learned, key=lambda period: abs(learned[period])) == pytest.approx(12.0)
    others = sum(
        abs(value) for period, value in learned.items() if abs(period - 12.0) > 1e-9
    )
    assert others < 1e-9, f"{others:.3e} of contribution landed off the 12-day cycle"


def test_gain_is_the_magnitude_of_the_same_frequency_weight(
    model: FITSForecaster,
) -> None:
    """A cycle's gain is ``|W[k, round(η·k)]|`` — the element carrying its own frequency."""
    eta = amplitude_scale(model.input_len, model.horizon)
    view = spectral.gain_phase(model)

    for index in (1, 5, 10, model.cof - 1):
        column = min(round(index * eta), model.out_bins - 1)
        gain, _ = view[spectral.period_days(index, model.input_len)]
        assert gain == pytest.approx(abs(model.weight[index, column]))


def test_the_phase_is_reported_in_days_not_radians(model: FITSForecaster) -> None:
    """``φ / 2π × period``: a quarter turn of a 12-day cycle is three days.

    Radians of an unnamed cycle are not a thing a supervisor can picture, and §6.5's
    narrative sentence quotes days.
    """
    eta = amplitude_scale(model.input_len, model.horizon)
    weight = np.zeros((model.cof, model.out_bins), dtype="complex128")
    bin_of_twelve_days = model.input_len // 12
    weight[bin_of_twelve_days, round(bin_of_twelve_days * eta)] = 2.0 * np.exp(
        1j * math.pi / 2
    )
    model._set_weight(weight)

    gain, shift = spectral.gain_phase(model)[12.0]

    assert gain == pytest.approx(2.0)
    assert shift == pytest.approx(3.0)


def test_the_dc_entry_reports_no_shift(model: FITSForecaster) -> None:
    """An infinite period cannot be shifted by a finite number of days."""
    _, shift = spectral.gain_phase(model)[spectral.DC_PERIOD]

    assert shift == spectral.NO_SHIFT


# ── the response curve GB-46 draws ──────────────────────────────────────────


def test_the_response_curve_is_sorted_by_period_and_excludes_dc(
    model: FITSForecaster, cfg: Config
) -> None:
    periods, gains = spectral.frequency_response(model)

    assert len(periods) == model.cof - 1  # every retained bin but the dead one
    assert np.all(np.isfinite(periods))
    assert list(periods) == sorted(periods)
    assert periods[-1] == pytest.approx(float(cfg.window.input_len))
    assert len(gains) == len(periods)
    assert np.all(gains >= 0.0)


def test_the_fastest_cycle_on_the_curve_is_the_configured_cutoff(
    model: FITSForecaster, cfg: Config
) -> None:
    """The low-pass is visible in the axis: nothing faster than the cutoff has a gain."""
    periods, _ = spectral.frequency_response(model)

    assert periods[0] >= cfg.fits.cutoff_period_days


def test_the_dominant_period_is_a_share_of_the_gross(
    model: FITSForecaster, windows: np.ndarray
) -> None:
    """Share of gross, never of net — percent of a near-zero forecast is unbounded."""
    period, share = spectral.dominant_period(model, windows[0])

    assert 0.0 <= share <= 1.0
    view = spectral.per_frequency(model, windows[0])
    assert abs(view[period]) == max(abs(value) for value in view.values())


# ── the two views, published together ───────────────────────────────────────


def test_both_views_close_against_the_same_total(
    model: FITSForecaster, windows: np.ndarray, channels: tuple[str, ...]
) -> None:
    """A reader must be able to move between the channel view and the frequency view."""
    found = spectral.explain_spectral(model, windows[0], channels)

    assert sum(found.per_channel.values()) == pytest.approx(
        found.forecast_total, abs=1e-9
    )
    assert sum(found.per_frequency.values()) == pytest.approx(
        found.forecast_total, abs=1e-9
    )
    assert found.gain_phase is not None


def test_the_scaled_explanation_is_the_unscaled_one_times_the_scale(
    model: FITSForecaster, windows: np.ndarray, channels: tuple[str, ...]
) -> None:
    """Both views carry the target scaling of 20 Aug 2026, or they disagree with each other."""
    scale = 0.0145
    plain = spectral.explain_spectral(model, windows[0], channels)
    raw = spectral.explain_spectral(model, windows[0], channels, scale=scale)

    assert raw.forecast_total == pytest.approx(plain.forecast_total * scale, rel=1e-9)
    for period, value in plain.per_frequency.items():
        assert raw.per_frequency[period] == pytest.approx(value * scale, rel=1e-6)

    # The gain and phase describe the WEIGHTS and are not in the forecast's unit, so they
    # do not move. Stating it here stops a later reader "fixing" it.
    assert raw.gain_phase == plain.gain_phase


def test_a_window_of_the_wrong_length_is_refused(
    model: FITSForecaster, channels: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="bars and the window has"):
        spectral.per_frequency(model, np.zeros((10, len(channels)), dtype="float32"))
