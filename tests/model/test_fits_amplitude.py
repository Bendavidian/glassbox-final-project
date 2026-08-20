"""GB-42 acceptance: the irFFT amplitude trap, guarded before a line of FITS was written.

Spec §6.3 calls this the trap to read before writing a line, and §6.2 puts the fix in the
pipeline as ``✱ SCALE — multiply by (L + H) / L``. This file was written **before** GB-41
so the implementation was written against a test rather than the other way round.

**The grid is chosen, and the choice is the point.** Bin ``k`` of a length-``L`` transform
is the frequency ``k/L``; the same physical frequency sits at bin ``k·η`` of a
length-``L+H`` transform. When ``η·k`` is not an integer the frequency cannot land on an
output bin, and **no parameter-free extension can reconstruct the input** — that gap is
exactly what :class:`FITSForecaster`'s learned complex layer exists to interpolate across.
At the configured ``H=4`` a 12-day cycle gives ``η·k = 10.333``; measured backcast error
**4.97**, which is the amplitude itself. At ``H=12`` it gives ``η·k = 11`` exactly, and the
error is **2.2e-14**.

So this file runs at ``H=12``, and it is not a workaround: an amplitude test at ``H=4``
would measure interpolation error and attribute it to amplitude, which is the same mistake
as asserting a recovered amplitude of 3.0 where only the *ratio* is exact.

**Three tests.** The first is the acceptance criterion. The second measures the trap on the
FFT libraries themselves, so this file demonstrates the bug it names rather than merely
asserting that something is fine. The third says why it costs a day.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from glassbox.model.fits import amplitude_scale, extend_spectrum

# A sinusoid both grids can represent exactly. `PERIOD` divides `INPUT_LEN`, so the signal
# lands on bin `INPUT_LEN // PERIOD` with no leakage into its neighbours; and `HORIZON` is
# chosen so that `eta * k` is an integer, so the frequency lands on an output bin too.
# Leakage or a fractional bin would put an error floor above the 1e-4 the spec asks for,
# and the test would then be measuring the grid rather than the amplitude.
INPUT_LEN = 120
HORIZON = 12
PERIOD = 12
AMPLITUDE = 3.0

TOLERANCE = 1e-4

# The horizon the project is actually configured at, where the frequency does NOT land on
# an output bin. Used to show that the choice above was necessary.
CONFIGURED_HORIZON = 4


def a_sinusoid(length: int = INPUT_LEN) -> np.ndarray:
    """``AMPLITUDE * sin(2*pi*t/PERIOD)``, shaped ``(1, length)`` as the pipeline wants."""
    t = np.arange(length, dtype="float64")
    return (AMPLITUDE * np.sin(2.0 * math.pi * t / PERIOD))[None, :]


def test_fits_amplitude_reconstruction():
    """Guards the irFFT amplitude trap. torch.fft.irfft normalises by output
    length, so extending from L to L+H shrinks every amplitude by L/(L+H)
    unless corrected by multiplying by (L+H)/L.

    Symptom if missed: forecasts plausible in shape but systematically flat,
    MAE and MSE IMPROVED because a flatter forecast sits closer to zero, and
    cross-model comparison corrupted — FITS would beat DLinear on MAE by being
    flatter rather than by being better. It looks like a modelling result and
    it is a bug.
    """
    window = a_sinusoid()
    extended = extend_spectrum(window, horizon=HORIZON)

    assert extended.shape == (1, INPUT_LEN + HORIZON)

    backcast = extended[:, :INPUT_LEN]
    assert np.allclose(backcast, window, atol=TOLERANCE), (
        f"the backcast is out by {np.abs(backcast - window).max():.3e}; a uniform "
        f"shortfall of {INPUT_LEN / (INPUT_LEN + HORIZON):.6f} is the amplitude trap of "
        "spec 6.3 and means the (L+H)/L scaling is missing"
    )

    # The amplitude has to survive into the forecast too, or the model is flat exactly
    # where it is being asked to predict.
    assert np.abs(extended).max() == pytest.approx(AMPLITUDE, abs=TOLERANCE)


def test_the_configured_horizon_cannot_be_reconstructed_without_the_learned_layer():
    """Why the grid above is chosen, measured rather than asserted.

    At ``H=4`` the 12-day cycle maps to bin 10.333, which does not exist. The parameter-free
    extension is then wrong by the order of the signal itself — and that is not a defect in
    :func:`extend_spectrum`, it is the reason `FITSForecaster` has a complex layer at all.
    A test that ran here would blame the amplitude for an interpolation error.
    """
    window = a_sinusoid()
    bin_index = INPUT_LEN / PERIOD
    eta = amplitude_scale(INPUT_LEN, CONFIGURED_HORIZON)

    assert not float(bin_index * eta).is_integer()

    extended = extend_spectrum(window, horizon=CONFIGURED_HORIZON)
    error = np.abs(extended[:, :INPUT_LEN] - window).max()

    assert error > AMPLITUDE / 2  # the order of the signal, not a rounding difference


def test_omitting_the_scale_shrinks_by_exactly_the_ratio():
    """The companion, and the reason the acceptance test is worth having.

    Measured on the FFT libraries directly, so it passes whatever GB-41 does. Inverting a
    spectrum to a length-``L+H`` signal divides every amplitude by exactly ``L/(L+H)``,
    because ``irfft`` normalises by the *output* length while the spectrum was built from
    the input length. **numpy and torch share the convention**, so this is not a quirk one
    could avoid by switching library — the correction has to live in our code.
    """
    window = a_sinusoid()
    total = INPUT_LEN + HORIZON
    eta = amplitude_scale(INPUT_LEN, HORIZON)

    spectrum = np.fft.rfft(window, n=INPUT_LEN, axis=1)
    padded = np.zeros((1, total // 2 + 1), dtype="complex128")
    for k in range(spectrum.shape[1]):
        target = round(k * eta)
        if target < padded.shape[1]:
            padded[:, target] = spectrum[:, k]

    unscaled = np.fft.irfft(padded, n=total, axis=1)
    scaled = unscaled * eta

    # **The ratio is the assertion, not the amplitude.** Re-sampling onto a longer grid
    # moves where the samples fall relative to the peak, so the sampled maximum need not
    # equal AMPLITUDE even when the scaling is perfect. The shortfall the trap causes is
    # uniform and multiplicative, which is what a ratio isolates and an amplitude
    # comparison would confound.
    ratio = float(np.abs(unscaled).max() / np.abs(scaled).max())
    assert ratio == pytest.approx(INPUT_LEN / total, abs=1e-12)

    # Every sample, not just the largest: the shortfall is uniform.
    assert np.allclose(unscaled * eta, scaled, atol=1e-12)

    # And torch normalises the same way, so the fix belongs in our pipeline.
    from_torch = torch.fft.irfft(torch.from_numpy(padded), n=total, dim=1).numpy()
    assert np.allclose(from_torch, unscaled, atol=1e-12)


def test_the_trap_is_large_enough_to_be_mistaken_for_a_result():
    """Why it costs a day rather than an hour: it does not look like a bug.

    A few per cent of flattening leaves the shape intact, so the forecast still looks
    plausible; and because a flatter forecast sits closer to zero, **MSE improves**. A
    reader watching MSE would conclude the model got better. What it actually breaks is
    the comparison: FITS would beat DLinear on MAE by being flatter.
    """
    total = INPUT_LEN + CONFIGURED_HORIZON
    shortfall = 1.0 - INPUT_LEN / total

    assert 0.01 < shortfall < 0.05  # visible in a metric, invisible in a plot

    truth = a_sinusoid(total)
    flattened = truth * (INPUT_LEN / total)

    assert float((flattened**2).mean()) < float((truth**2).mean())
