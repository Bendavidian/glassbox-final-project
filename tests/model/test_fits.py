"""GB-41 acceptance: the spectral core, and the properties the contract test cannot see.

``tests/model/test_forecaster_contract.py`` already runs spec §4.4's seven properties over
every registered forecaster, and FITS passes it with **no edit to that file**. What is left
here is what is specific to this model: that it is univariate and still attributes to every
channel, that the attribution matrix *is* the forward pass rather than a second derivation
of it, that RIN normalises per window, and that the amplitude scaling changes MAE while
leaving direction untouched.
"""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from glassbox.config.loader import Config, load_config
from glassbox.contracts.schemas import WindowBatch
from glassbox.model import ALL_FORECASTERS
from glassbox.model.fits import (
    CLOSE_CHANNEL,
    FITSForecaster,
    aligned_bins,
    amplitude_scale,
    cof_for,
    dead_row_fraction,
    extend_spectrum,
    mean_misalignment,
    out_bins_for,
)

SYMBOL = "AAPL"


@pytest.fixture
def cfg() -> Config:
    return load_config()


@pytest.fixture
def channels(cfg: Config) -> tuple[str, ...]:
    return cfg.channels.active_channels


@pytest.fixture
def model(cfg: Config, channels: tuple[str, ...]) -> FITSForecaster:
    return ALL_FORECASTERS["fits"](cfg, channels)


@pytest.fixture
def windows(cfg: Config, channels: tuple[str, ...]) -> np.ndarray:
    """Structured windows rather than noise: a cycle plus a drift, so a spectral model has
    something to find and the backcast has something to reconstruct."""
    rng = np.random.default_rng(1337)
    t = np.arange(cfg.window.input_len, dtype="float64")
    base = 0.01 * np.sin(2 * math.pi * t / 12.0) + 0.002
    stack = np.stack(
        [base + rng.normal(0, 0.002, size=cfg.window.input_len) for _ in range(24)]
    )
    full = np.repeat(stack[:, :, None], len(channels), axis=2)
    full[:, :, 1:] += rng.normal(
        0, 0.5, size=(24, cfg.window.input_len, len(channels) - 1)
    )
    return full.astype("float32")


def a_batch(
    windows: np.ndarray, channels: tuple[str, ...], horizon: int
) -> WindowBatch:
    import pandas as pd

    rng = np.random.default_rng(7)
    return WindowBatch(
        X=windows,
        y=rng.normal(0, 0.01, size=(windows.shape[0], horizon)).astype("float32"),
        timestamps=pd.date_range("2024-01-01", periods=windows.shape[0], tz="UTC"),
        symbols=tuple(SYMBOL for _ in range(windows.shape[0])),
        channels=channels,
        source="test",
    )


# ── univariate by design, and still attributing to everything ───────────────


def test_it_refuses_a_channel_set_without_the_series_it_forecasts(
    cfg: Config,
) -> None:
    with pytest.raises(ValueError, match="univariate by design"):
        FITSForecaster(
            input_len=cfg.window.input_len,
            horizon=cfg.window.horizon,
            channels=("rsi14", "vol_z"),
            cutoff_period_days=cfg.fits.cutoff_period_days,
        )


def test_attribution_names_every_active_channel_not_only_the_one_it_reads(
    model: FITSForecaster, windows: np.ndarray, channels: tuple[str, ...]
) -> None:
    """GB-33's ruling. An attribution that named only what the model reads would render as
    four missing panels beside DLinear's five, and would change what "exact" means between
    two arms of the same study."""
    attribution = model.explain(windows[0], channels)

    assert set(attribution.per_channel) == set(channels)
    for channel in channels:
        if channel != CLOSE_CHANNEL:
            assert attribution.per_channel[channel] == 0.0


def test_an_unread_channel_contributes_no_terms_rather_than_a_zero_weight(
    model: FITSForecaster, windows: np.ndarray, channels: tuple[str, ...]
) -> None:
    """The zero falls out of `Attribution.from_terms`'s arithmetic. A model that wrote its
    own zeros would be a model the exactness check never actually exercised — the mistake
    `PersistenceForecaster` used to make."""
    terms = model.linear_terms(windows[0], channels)

    assert terms[CLOSE_CHANNEL] != ()
    for channel in channels:
        if channel != CLOSE_CHANNEL:
            assert terms[channel] == ()


def test_changing_a_channel_it_does_not_read_changes_nothing(
    model: FITSForecaster, windows: np.ndarray
) -> None:
    """Univariate is a claim about behaviour, not only about the docstring."""
    disturbed = windows.copy()
    disturbed[:, :, 1:] *= 7.0

    assert np.array_equal(model.predict(windows), model.predict(disturbed))


# ── the attribution matrix is the forward pass ──────────────────────────────


def test_the_forecast_matrix_reproduces_predict(
    model: FITSForecaster, windows: np.ndarray, cfg: Config
) -> None:
    """The pipeline is linear in the window — mean subtraction and its inverse included —
    so the model *is* a matrix, and the attribution is that matrix rather than a second
    derivation that could drift from it."""
    matrix = model.forecast_matrix()
    closes = windows[:, :, 0].astype("float64")

    assert matrix.shape == (cfg.window.horizon, cfg.window.input_len)
    assert np.allclose(closes @ matrix.T, model.predict(windows), atol=1e-8)


def test_a_fitted_model_is_still_exactly_its_matrix(
    model: FITSForecaster, windows: np.ndarray, channels: tuple[str, ...], cfg: Config
) -> None:
    """Linearity is a property of the architecture, not of the zero initialisation."""
    model.fit(a_batch(windows, channels, cfg.window.horizon))

    closes = windows[:, :, 0].astype("float64")
    assert np.allclose(
        closes @ model.forecast_matrix().T, model.predict(windows), atol=1e-8
    )


def test_the_cached_matrix_cannot_outlive_the_weight(
    model: FITSForecaster, windows: np.ndarray, channels: tuple[str, ...], cfg: Config
) -> None:
    """The matrix is cached because it costs L forward passes. A cache that survived a fit
    would attribute a forecast the model no longer makes."""
    before = model.forecast_matrix().copy()
    model.fit(a_batch(windows, channels, cfg.window.horizon))

    assert not np.array_equal(before, model.forecast_matrix())


# ── RIN is per instance ─────────────────────────────────────────────────────


def test_rin_normalises_per_window_not_per_batch(
    model: FITSForecaster, windows: np.ndarray
) -> None:
    """Spec §4.4 property 6 is the permanent guard and `BatchNormForecaster` proves it can
    fail. This says the same thing directly: a window's forecast must not depend on what
    else is in the batch, and a batch mean is what makes it depend, because a batch is
    ordered in time."""
    alone = model.predict(windows[:1])
    in_company = model.predict(windows)[:1]

    assert np.array_equal(alone, in_company)

    shifted = windows.copy()
    shifted[1:] += 5.0  # move every OTHER window a long way
    assert np.array_equal(alone, model.predict(shifted)[:1])


def test_a_constant_offset_moves_the_forecast_by_that_offset(
    model: FITSForecaster, windows: np.ndarray
) -> None:
    """The inverse RIN adds the window's own mean back, so a level shift passes straight
    through. This is what makes a zero-initialised FITS a drift baseline rather than a
    zero baseline."""
    lifted = windows.copy()
    lifted[:, :, 0] += 0.03

    moved = model.predict(lifted) - model.predict(windows)
    assert np.allclose(moved, 0.03, atol=1e-6)


def test_a_zero_initialised_model_predicts_the_window_mean(
    model: FITSForecaster, windows: np.ndarray
) -> None:
    """Worth pinning because it differs from DLinear, where zero weights *are* the
    persistence baseline. Here the inverse RIN means an untrained FITS predicts the
    window's own average drift."""
    expected = windows[:, :, 0].astype("float64").mean(axis=1)

    assert np.allclose(model.predict(windows), expected[:, None], atol=1e-7)


# ── the shape the cutoff decides, and the checkpoint ────────────────────────


def test_the_measured_parameter_count(model: FITSForecaster, cfg: Config) -> None:
    """Replaces the predicted 1,200 of spec §6.1 with a measurement.

    Counted in **reals**, because that is what a parameter count means everywhere else in
    the study: reporting 600 complex weights against DLinear's 4,800 reals would flatter
    FITS by a factor of two on the one axis §6.1 warns the report not to borrow a framing
    from.
    """
    assert model.cof == cfg.window.input_len // cfg.fits.cutoff_period_days
    assert model.out_bins == math.ceil(
        amplitude_scale(cfg.window.input_len, cfg.window.horizon) * model.cof
    )
    assert model.n_parameters == model.cof * model.out_bins * 2


def test_the_dc_row_is_allocated_and_cannot_learn(
    model: FITSForecaster,
    windows: np.ndarray,
    channels: tuple[str, ...],
    cfg: Config,
) -> None:
    """1,200 is what the tensor holds. 1,150 is what can move. Both belong in the report.

    RIN subtracts each window's own mean, and the rFFT's bin 0 **is** that mean — so after
    RIN it is zero on every window, always. Row 0 of the complex weight matrix is therefore
    multiplied by zero on every forward pass: it takes no gradient, never leaves its
    initial value, and cannot change a forecast. That is ``out_bins`` complex weights —
    **50 reals at the configured geometry, 4.17% of the count spec §6.1 reports**.

    **Kept, not removed.** §6.2's low-pass keeps "the first ``COF`` bins" and bin 0 is one
    of them, and the source paper's architecture carries the same dead row for the same
    reason. Removing it would be a change to a specified pipeline bought for a cosmetic
    4%. What is not acceptable is quoting 1,200 as though all of it were capacity, so the
    number is pinned here and both figures go to GB-57.
    """
    horizon = cfg.window.horizon
    centred = windows[:, :, 0].astype("float64")
    centred = centred - centred.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centred, n=cfg.window.input_len, axis=1)

    # The input the row is multiplied by: zero to floating-point precision, beside a
    # neighbour that is emphatically not, so the assertion is a contrast and not a
    # tolerance chosen to pass.
    assert np.abs(spectrum[:, 0]).max() < 1e-12 < np.abs(spectrum[:, 1]).max()

    model.fit(a_batch(windows, channels, horizon))

    assert np.abs(model.weight[0]).max() < 1e-9
    assert np.abs(model.weight[1:]).max() > 1e-3

    # And the forecast is indifferent to it, which is the property that makes the count
    # wrong rather than merely unusual.
    before = model.predict(windows)
    weight = model.weight.copy()
    model._set_weight(_bumped(weight, row=0))
    assert np.array_equal(model.predict(windows), before)

    model._set_weight(_bumped(weight, row=1))
    assert not np.allclose(model.predict(windows), before)


def _bumped(weight: np.ndarray, row: int) -> np.ndarray:
    """``weight`` with every entry of one row moved by ``1 + 1j``."""
    disturbed = weight.copy()
    disturbed[row, :] += 1.0 + 1.0j
    return disturbed


def test_a_wider_cutoff_keeps_fewer_frequencies(
    cfg: Config, channels: tuple[str, ...]
) -> None:
    """The one hyperparameter, behaving as its name says."""
    slower = ALL_FORECASTERS["fits"](
        replace(cfg, fits=replace(cfg.fits, cutoff_period_days=20)), channels
    )
    faster = ALL_FORECASTERS["fits"](
        replace(cfg, fits=replace(cfg.fits, cutoff_period_days=5)), channels
    )

    assert slower.cof < faster.cof
    assert slower.n_parameters < faster.n_parameters


def test_a_checkpoint_round_trips_the_complex_weight(
    model: FITSForecaster,
    windows: np.ndarray,
    channels: tuple[str, ...],
    cfg: Config,
    tmp_path: Path,
) -> None:
    model.fit(a_batch(windows, channels, cfg.window.horizon))
    path = tmp_path / "fits.json"
    model.save(str(path))

    reloaded = FITSForecaster.load(str(path))

    assert np.array_equal(reloaded.weight, model.weight)
    assert np.array_equal(reloaded.predict(windows), model.predict(windows))
    assert reloaded.fitted == model.fitted


def test_a_reloaded_model_refuses_to_fit_rather_than_inventing_defaults(
    model: FITSForecaster,
    windows: np.ndarray,
    channels: tuple[str, ...],
    cfg: Config,
    tmp_path: Path,
) -> None:
    path = tmp_path / "fits.json"
    model.save(str(path))

    with pytest.raises(ValueError, match="holds no training configuration"):
        FITSForecaster.load(str(path)).fit(
            a_batch(windows, channels, cfg.window.horizon)
        )


# ── the backcast, which B+F supervision exists to teach ─────────────────────


def test_the_backcast_has_the_shape_of_the_input_window(
    model: FITSForecaster, windows: np.ndarray, cfg: Config
) -> None:
    assert model.backcast(windows).shape == (windows.shape[0], cfg.window.input_len)


def test_training_improves_the_reconstruction(
    model: FITSForecaster, windows: np.ndarray, channels: tuple[str, ...], cfg: Config
) -> None:
    """Spec §5 fixes supervision at B+F, and this is what that buys: without the backcast
    term the layer has no reason to preserve the signal it interpolates, and the frequency
    response the report plots would mean nothing."""
    closes = windows[:, :, 0]
    before = float(np.abs(model.backcast(windows) - closes).mean())

    model.fit(a_batch(windows, channels, cfg.window.horizon))

    assert float(np.abs(model.backcast(windows) - closes).mean()) < before


# ── Ben's correction to §6.3, converted into a measurement ──────────────────


def test_the_amplitude_scale_changes_mae_and_cannot_change_direction(
    model: FITSForecaster, windows: np.ndarray
) -> None:
    """Spec §6.3 claimed the missing scale degraded direction accuracy. **It cannot.**

    Omitting it multiplies every sample by ``L/(L+H)``, a uniform *positive* factor. A
    positive scaling cannot change a sign, so direction accuracy is identical to the last
    decimal; and the per-fold calibrated band is fitted on forecasts carrying the same
    factor, so the signals barely move either. What it does change is MAE and MSE, which
    **improve**, because a flatter forecast sits closer to zero — and that corrupts the one
    axis on which FITS and DLinear are directly compared.

    This is the check that would have caught the wrong line when it was written, which is
    why it is a test rather than a paragraph.
    """
    scaled = model.predict(windows).astype("float64")
    unscaled = scaled / amplitude_scale(model.input_len, model.horizon)

    truth = np.roll(windows[:, :, 0].astype("float64")[:, : model.horizon], 1, axis=0)

    direction_scaled = (np.sign(scaled) == np.sign(truth)).mean()
    direction_unscaled = (np.sign(unscaled) == np.sign(truth)).mean()
    assert direction_scaled == direction_unscaled
    assert np.array_equal(np.sign(scaled), np.sign(unscaled))

    mae_scaled = float(np.abs(scaled - truth).mean())
    mae_unscaled = float(np.abs(unscaled - truth).mean())
    assert mae_scaled != mae_unscaled


def test_extend_spectrum_accepts_a_single_window(cfg: Config) -> None:
    """GB-42 passes ``(B, L)``; the loop-free callers find ``(L,)`` more natural."""
    t = np.arange(cfg.window.input_len, dtype="float64")
    flat = np.sin(2 * math.pi * t / 12.0)

    assert extend_spectrum(flat, 12).shape == (cfg.window.input_len + 12,)
    assert extend_spectrum(flat[None, :], 12).shape == (1, cfg.window.input_len + 12)


# -- the geometry the cutoff decides (GB-50) ---------------------------------


def test_the_output_bin_count_is_exact_where_the_two_grids_line_up() -> None:
    """**The defect the COF sweep exposed**, pinned at the value that exposed it.

    ``ceil(eta * COF)`` in floating point is correct on paper and wrong exactly where
    ``eta * COF`` is a whole number: at ``COF = 60`` the product evaluates to
    ``62.00000000000001`` and the ceiling came out 63 - one bin more than the architecture
    describes, 120 allocated reals nothing accounts for, and no stage downstream that
    would refuse it. It fires only on the **perfectly aligned** case, which is the one the
    formula exists to handle cleanly.
    """
    assert out_bins_for(120, 4, 2) == 62
    assert (
        math.ceil(amplitude_scale(120, 4) * 60) == 63
    )  # the float answer, for the record


@pytest.mark.parametrize("cutoff", [1, 2, 3, 4, 5, 8, 10, 12, 20, 24, 30, 40, 60])
def test_the_output_bin_count_is_the_ceiling_of_the_exact_ratio(cutoff: int) -> None:
    """The definition, in exact arithmetic, independently of how the module computes it."""
    length, horizon = 120, 4
    exact = Fraction(length + horizon, length) * cof_for(length, cutoff)

    assert out_bins_for(length, horizon, cutoff) == math.ceil(exact)


def test_the_dead_row_share_rises_as_the_cutoff_keeps_fewer_bins() -> None:
    """`1 / COF`, so the **smaller** model wastes proportionally **more**.

    RIN subtracts each window's mean and bin 0 of the rFFT is that mean, so row 0 takes no
    gradient at any cutoff. The intuition that a tighter filter is a leaner model is wrong
    in the only sense that matters here.
    """
    shares = [dead_row_fraction(120, 4, cutoff) for cutoff in (2, 5, 10, 20)]

    assert shares == pytest.approx([1 / 60, 1 / 24, 1 / 12, 1 / 6])
    assert shares == sorted(shares)


def test_the_dead_row_share_is_what_the_model_allocates_and_cannot_use() -> None:
    """Derived, not asserted: the fraction is read off a real weight matrix."""
    for cutoff in (2, 5, 10, 20):
        model = FITSForecaster(
            input_len=120,
            horizon=4,
            channels=(CLOSE_CHANNEL,),
            cutoff_period_days=cutoff,
        )
        assert dead_row_fraction(120, 4, cutoff) == pytest.approx(
            model.out_bins * 2 / model.n_parameters
        )


def test_the_interpolation_cost_rises_with_cof_and_then_saturates() -> None:
    """**Saturation, not a middle** — and the first reading of this said "peak".

    ``eta = 1 + H/L = 1 + 1/30``, so ``frac(eta*k) = frac(k/30)`` and the fractional part
    has a **period of 30 bins**. Mean misalignment therefore rises while the retained bins
    cover less than one cycle and settles at **0.25**, the mean distance of a uniform
    fractional part to the nearest integer, once they cover one or more.

    The apparent peak at the deployed cutoff of 5 is a **partial-cycle sampling
    artefact**: 24 of 30 bins covers ``k/30`` up to 0.77 and over-weights the far half.
    Asserting the peak would be asserting the artefact.
    """
    period = 120 // 4  # H/L = 1/30, so the fractional part repeats every 30 bins
    rising = [mean_misalignment(120, 4, c) for c in (60, 20, 10)]
    saturated = [mean_misalignment(120, 4, c) for c in (5, 4, 2)]

    assert cof_for(120, 10) < period <= cof_for(120, 4)
    assert rising == sorted(rising)  # below one cycle, the cost climbs with COF
    assert all(abs(value - 0.25) <= 0.04 for value in saturated)
    # Exactly a quarter where the retained bins are a whole number of cycles.
    assert mean_misalignment(120, 4, 4) == pytest.approx(0.25)
    assert mean_misalignment(120, 4, 2) == pytest.approx(0.25)


def test_more_retained_frequencies_cost_more_to_reconstruct() -> None:
    """The trade-off `COF` carries, stated as a property rather than as prose.

    Keeping more bins buys more information **and** worse reconstruction of each, because
    a higher ``k`` puts ``eta*k`` further from an integer. The two pull opposite ways, and
    the asymmetry is what makes the sweep a test rather than a formality: the cost
    saturates while the information does not.
    """
    keeping_more = mean_misalignment(120, 4, 10)  # COF 12
    keeping_fewer = mean_misalignment(120, 4, 20)  # COF 6

    assert cof_for(120, 10) > cof_for(120, 20)
    assert keeping_more > keeping_fewer


def test_bin_zero_is_aligned_at_every_cutoff_and_almost_nothing_else_is() -> None:
    """``eta = 31/30``, so ``eta*k`` is an integer only where ``30 | k``."""
    assert [aligned_bins(120, 4, c) for c in (2, 5, 10, 20)] == [2, 1, 1, 1]
