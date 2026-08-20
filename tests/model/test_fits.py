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
from pathlib import Path

import numpy as np
import pytest

from glassbox.config.loader import Config, load_config
from glassbox.contracts.schemas import WindowBatch
from glassbox.model import ALL_FORECASTERS
from glassbox.model.fits import (
    CLOSE_CHANNEL,
    FITSForecaster,
    amplitude_scale,
    extend_spectrum,
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
