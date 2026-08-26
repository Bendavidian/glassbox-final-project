"""GB-66 acceptance: WITS, the wavelet transform in the position FITS gives the rFFT.

What is asserted here is not that a wavelet model runs — that is the easy half — but the
properties the comparison against FITS depends on: that the transform is exactly a matrix
(which is what makes one forward pass serve inference, training and attribution), that the
band decomposition closes algebraically rather than approximately, that there is no
amplitude scale to restore, and that the two gaps left open fail rather than wait to be
remembered.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import pywt

from glassbox.config.loader import VALID_BOUNDARIES, load_config
from glassbox.model import ALL_FORECASTERS
from glassbox.model.wits import (
    MEAN_KEY,
    WITSForecaster,
    analysis_matrix,
    band_lengths,
    band_names,
    n_parameters_for,
    synthesis_matrix,
)

FAMILY = "db4"
LEVELS = 3
RETAINED = 2
CHANNELS = ("close_logret", "rsi14", "vol_z")
L, H = 120, 4


def a_model(boundary: str = "symmetric", seed: int = 0) -> WITSForecaster:
    """A WITS with random band maps. Untrained weights are zero, which hides sign errors."""
    model = WITSForecaster(
        input_len=L,
        horizon=H,
        channels=CHANNELS,
        family=FAMILY,
        levels=LEVELS,
        boundary=boundary,
        retained_bands=RETAINED,
    )
    rng = np.random.default_rng(seed)
    model._set_weights(
        [rng.standard_normal(weight.shape) * 0.1 for weight in model.weights]
    )
    return model


def a_window(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((L, len(CHANNELS))).astype("float32")


# ── the transform is exactly a matrix ────────────────────────────────────────


@pytest.mark.parametrize("boundary", VALID_BOUNDARIES)
def test_the_analysis_matrix_reproduces_pywt_exactly(boundary: str) -> None:
    """**The whole model rests on this.** A DWT with a fixed wavelet, level count and
    boundary is linear, so it is a matrix — and once it is a matrix, torch can train
    through it, which it cannot do through ``pywt``. If this drifts, the training path and
    the inference path stop being the same function and nothing else would say so."""
    matrix = analysis_matrix(L, FAMILY, LEVELS, boundary, RETAINED)
    x = np.random.default_rng(0).standard_normal(L)

    expected = np.concatenate(
        pywt.wavedec(x, FAMILY, mode=boundary, level=LEVELS)[:RETAINED]
    )
    assert np.abs(matrix @ x - expected).max() < 1e-12


@pytest.mark.parametrize("boundary", VALID_BOUNDARIES)
def test_the_synthesis_matrix_reproduces_pywt_exactly(boundary: str) -> None:
    """And the inverse, with the discarded bands zeroed at their L+H lengths."""
    analysis = analysis_matrix(L, FAMILY, LEVELS, boundary, RETAINED)
    synthesis = synthesis_matrix(L, H, FAMILY, LEVELS, boundary, RETAINED)
    coefficients = analysis @ np.random.default_rng(0).standard_normal(L)

    shapes = band_lengths(L + H, FAMILY, LEVELS, boundary)
    parts, offset = [], 0
    for size in shapes[:RETAINED]:
        parts.append(coefficients[offset : offset + size])
        offset += size
    parts.extend(np.zeros(size) for size in shapes[RETAINED:])
    expected = np.asarray(pywt.waverec(parts, FAMILY, mode=boundary))[: L + H]

    assert np.abs(synthesis @ coefficients - expected).max() < 1e-12


def test_the_retained_bands_are_the_same_length_on_both_sides() -> None:
    """**The structural claim the whole task turns on.** FITS must interpolate because bin
    ``k`` of a length-L transform is a different frequency from bin ``k`` of a length-L+H
    one. The retained wavelet bands are 21 coefficients at both lengths, so the maps are
    square and nothing is remapped onto a shifted grid — which is why the null control is a
    real test rather than a formality."""
    assert band_lengths(L, FAMILY, LEVELS, "symmetric")[:RETAINED] == (21, 21)
    assert band_lengths(L + H, FAMILY, LEVELS, "symmetric")[:RETAINED] == (21, 21)


# ── the numbers the report quotes ────────────────────────────────────────────


def test_the_measured_parameter_count_is_882() -> None:
    """Replaces the expansion's ~930 estimate, and it is 882 rather than 930 because the
    estimate assumed the output band grew with the window and it does not."""
    assert n_parameters_for(L, H, FAMILY, LEVELS, "symmetric", RETAINED) == 882
    assert a_model().n_parameters == 882


def test_no_amplitude_scale_is_needed() -> None:
    """**The measurement, not the assumption** (contrast spec 6.3). ``irfft`` normalises by
    output length and FITS must multiply back by ``(L+H)/L`` — a *uniform* 0.9677.
    ``waverec`` carries no such factor: with the maps set to the identity a slow sinusoid
    reconstructs at ratio ~1.0, and critically the ratio is **not** a constant across
    periods, which is what a missing normalisation would look like."""
    model = a_model()
    model._set_weights([np.eye(*weight.shape) for weight in model.weights])

    ratios = []
    for period in (60, 40, 30):
        t = np.arange(L)
        x = np.zeros((L, len(CHANNELS)), dtype="float32")
        x[:, 0] = 3.0 * np.sin(2 * np.pi * t / period)
        reconstruction = model.backcast(x[None, ...])[0]
        ratios.append(float(np.abs(reconstruction).max() / 3.0))

    assert all(0.9 < ratio < 1.1 for ratio in ratios)
    # A uniform L/(L+H) shrinkage would put every ratio on the same wrong constant.
    assert max(ratios) - min(ratios) > 1e-6


# ── attribution closes on both axes ──────────────────────────────────────────


def test_the_bands_and_the_mean_sum_to_the_forecast_matrix() -> None:
    """Algebraic rather than approximate, so the tolerance is float64 round-off and not
    spec 4.4's 1e-5. The bands are separable and reconstruction is additive; if this needed
    1e-5 the decomposition would be a coincidence rather than an identity."""
    model = a_model()
    total = sum(model.level_matrices().values()) + model.mean_matrix()
    assert np.abs(total - model.forecast_matrix()).max() < 1e-12


def test_per_level_sums_to_the_forecast() -> None:
    """GB-66's acceptance criterion for the explanation: the band totals **are** the
    forecast, including the RIN mean, which belongs to no band and still has to be there.

    **At spec 4.4's tolerance and not at float64 round-off, and the reason is worth stating
    because it is not the decomposition.** ``predict`` returns float32, as the protocol
    requires, so ``forecast_total`` carries a float32 rounding of about 1e-8 at these
    magnitudes while the band totals are computed in float64. The exact identity is the one
    between the *matrices*, asserted at 1e-12 by
    :func:`test_the_bands_and_the_mean_sum_to_the_forecast_matrix`; this asserts the same
    thing through the contract's own cast, so the residual is checked to be consistent with
    that cast rather than merely small.
    """
    model, window = a_model(), a_window()
    attribution = model.explain(window, CHANNELS)

    assert set(attribution.per_level) == {"a3", "d3", MEAN_KEY}
    total = sum(attribution.per_level.values())
    assert total == pytest.approx(attribution.forecast_total, abs=1e-5)
    assert abs(total - attribution.forecast_total) < 1e-6 * abs(total)


def test_per_lag_sums_to_the_forecast_and_names_only_the_channel_read() -> None:
    """The *when* axis. It must close for the same reason ``per_level`` does, and it must
    be zero in the channels WITS does not read — the same statement ``linear_terms`` makes
    by returning no terms at all for them."""
    model, window = a_model(), a_window()
    attribution = model.explain(window, CHANNELS)

    assert attribution.per_lag.shape == (L, len(CHANNELS))
    # Spec 4.4's tolerance, for the float32 reason `test_per_level_sums_to_the_forecast`
    # records: the residual is the protocol's cast, not the decomposition.
    total = float(attribution.per_lag.sum())
    assert total == pytest.approx(attribution.forecast_total, abs=1e-5)
    assert abs(total - attribution.forecast_total) < 1e-6 * abs(total)
    assert not attribution.per_lag[:, 1:].any()


def test_the_forecast_matrix_is_the_forecast() -> None:
    """``predict(x) == M @ x_close``. Measured against the real forward pass, because a
    matrix that merely looked right would make every attribution above vacuous."""
    model, window = a_model(), a_window()
    direct = model.predict(window[None, ...])[0]
    through = model.forecast_matrix() @ window[:, 0].astype("float64")
    assert np.abs(direct - through).max() < 1e-6


# ── RIN interacts with the basis the way the docstring claims ────────────────


def test_rin_changes_the_approximation_band_and_leaves_the_details_alone() -> None:
    """**GB-66's fourth question, as a measurement rather than an argument.** db4 has 4
    vanishing moments, so its detail filters annihilate constants: removing the window mean
    must move every ``cA`` coefficient and no ``cD`` coefficient. That is what makes
    keeping RIN in FITS's position safe — its whole effect is confined to one band."""
    x = np.random.default_rng(0).standard_normal(L)
    full = pywt.wavedec(x, FAMILY, mode="symmetric", level=LEVELS)
    centred = pywt.wavedec(x - x.mean(), FAMILY, mode="symmetric", level=LEVELS)

    assert np.abs(full[0] - centred[0]).min() > 1e-9  # every cA coefficient moved
    for detail_full, detail_centred in zip(full[1:], centred[1:], strict=True):
        assert np.abs(detail_full - detail_centred).max() < 1e-12


# ── the contract, the registry and the config layer ──────────────────────────


def test_wits_is_registered_and_constructible_from_config() -> None:
    """`model.active: wits` is the only change needed, which is what the registry is for."""
    cfg = load_config()
    model = ALL_FORECASTERS["wits"](cfg, cfg.channels.active_channels)
    assert model.name == "wits"
    assert model.input_len == cfg.window.input_len


def test_a_checkpoint_round_trips(tmp_path: Path) -> None:
    """Including the geometry. A checkpoint that could not say which boundary produced its
    weights would be loadable under another and silently wrong."""
    model, window = a_model(), a_window()
    path = tmp_path / "wits.json"
    model.save(str(path))
    restored = WITSForecaster.load(str(path))

    assert restored.boundary == model.boundary
    assert restored.family == model.family
    assert restored.retained_bands == model.retained_bands
    assert (
        np.abs(
            restored.predict(window[None, ...]) - model.predict(window[None, ...])
        ).max()
        < 1e-12
    )
    assert json.loads(path.read_text(encoding="utf-8"))["name"] == "wits"


def test_band_names_say_which_scale() -> None:
    assert band_names(3, 2) == ("a3", "d3")
    assert band_names(3, 4) == ("a3", "d3", "d2", "d1")


# ── the gaps left open, failing rather than waiting to be remembered ─────────


def test_the_shift_invariant_axis_refuses_rather_than_pretending() -> None:
    """**GB-66's third design question is not answered yet, and this is the refusal.**

    SWT is undecimated, so one map per band costs 29,760 reals against the DWT's 882 — it
    does not change the capacity argument so much as remove it. It is still a required
    axis. What blocks it is concrete: ``pywt.iswt`` needs a length divisible by
    ``2**levels`` and ``L+H = 124`` is not divisible by 8, so it needs a padding rule
    nobody has decided.

    A ``NotImplementedError`` naming the constraint is a mechanism; a docstring saying
    "later" is a note. This test is what makes the refusal stay true.
    """
    with pytest.raises(NotImplementedError, match="divisible by 2..levels"):
        WITSForecaster(
            input_len=L,
            horizon=H,
            channels=CHANNELS,
            family=FAMILY,
            levels=LEVELS,
            boundary="symmetric",
            retained_bands=RETAINED,
            shift_invariant=True,
        )


def test_the_boundary_axis_and_the_model_hash_land_together() -> None:
    """**An assertion on the pairing, not an xfail, and the difference matters.**

    Two things are missing and they must arrive in the same commit: GB-66's boundary axis
    as a column in ``results.csv``, and ``wits`` in ``MODEL_SHAPING_SECTIONS`` so two cells
    differing only by boundary are distinguishable by ``model_config_hash`` alone - the
    property ``study._cell_config`` routes the COF axis through ``cfg.fits`` to get.

    A ``strict`` xfail over both would only fire once *both* landed, so adding the axis
    without the hash would leave it xfailing quietly and the study would record a boundary
    sweep whose cells all claim the same model provenance. This asserts the equality
    instead: it passes today because neither is present, and fails the moment either one
    arrives alone.

    Why neither is present yet is recorded at the hold in ``loader.MODEL_SHAPING_SECTIONS``:
    adding the section moved the deployed ``model_config_hash`` on 25 Aug 2026 and
    ``load_predictor`` would have refused the deployed checkpoint hours before the GATE 2
    rehearsal.
    """
    from glassbox.config.loader import MODEL_SHAPING_SECTIONS
    from glassbox.experiments import study

    axis = any("boundary" in column for column in study.COLUMNS)
    hashed = "wits" in MODEL_SHAPING_SECTIONS
    assert axis == hashed, (
        "GB-66's boundary axis and `wits` in MODEL_SHAPING_SECTIONS must land together: "
        f"axis in results.csv={axis}, wits in the model hash={hashed}. An axis without "
        "the hash records a boundary sweep whose cells all claim the same provenance; the "
        "hash without the axis invalidates every checkpoint for a sweep nobody runs"
    )
