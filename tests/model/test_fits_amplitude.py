"""GB-42 acceptance: the irFFT amplitude trap, guarded before a line of FITS is written.

Spec §6.3 calls this the trap to read before writing a line, and §6.2 puts the fix in the
pipeline as ``✱ SCALE — multiply by (L + H) / L``. This file exists **before** GB-41 so
that the implementation is written against a test rather than the other way round.

**Two tests, and the second is what gives the first its teeth.**

:func:`test_fits_amplitude_reconstruction` is the acceptance criterion and needs FITS. It
is marked ``xfail(strict=True)`` until GB-41 lands: strict means that when the
implementation arrives and the test passes, the suite **fails** on the unexpected pass
until the marker is removed. The expectation therefore cannot be forgotten, and CI does
not sit red for a month in the meantime — a red CI that everybody knows is red is a CI
nobody reads, which is the same failure mode as a guard that fires spuriously.

:func:`test_omitting_the_scale_shrinks_by_exactly_the_ratio` needs nothing but ``torch``
and passes today. It measures the trap on the library itself, so this file demonstrates
the bug it names rather than merely asserting that something is fine.
"""

from __future__ import annotations

import math

import pytest
import torch

# A sinusoid the grid can represent exactly. `PERIOD` divides `INPUT_LEN`, so the signal
# lands on bin `INPUT_LEN // PERIOD` with no leakage into its neighbours - which matters,
# because leakage would put an error floor above the 1e-4 the spec asks for and the test
# would then be measuring the window rather than the amplitude.
INPUT_LEN = 120
HORIZON = 4
PERIOD = 12
AMPLITUDE = 3.0

TOLERANCE = 1e-4


def a_sinusoid(length: int = INPUT_LEN) -> torch.Tensor:
    """``AMPLITUDE * sin(2*pi*t/PERIOD)``, shaped ``(1, length)`` as the pipeline wants."""
    t = torch.arange(length, dtype=torch.float64)
    return (AMPLITUDE * torch.sin(2.0 * math.pi * t / PERIOD)).unsqueeze(0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "GB-41 has not landed. `glassbox.model.fits.extend_spectrum` is the interface "
        "this test fixes: (B, L) in, (B, L+H) out, amplitude-correct, whose first L "
        "samples reconstruct the input. Strict, so the pass that arrives with GB-41 "
        "fails the suite until this marker is removed."
    ),
)
def test_fits_amplitude_reconstruction():
    """Guards the irFFT amplitude trap. torch.fft.irfft normalises by output
    length, so extending from L to L+H shrinks every amplitude by L/(L+H)
    unless corrected by multiplying by (L+H)/L.

    Symptom if missed: forecasts plausible in shape but systematically flat,
    direction accuracy quietly degraded, and MSE possibly IMPROVED because a
    flatter forecast sits closer to zero. It looks like a modelling problem
    and it is a bug.
    """
    from glassbox.model.fits import extend_spectrum

    window = a_sinusoid()
    extended = extend_spectrum(window, horizon=HORIZON)

    assert extended.shape == (1, INPUT_LEN + HORIZON)

    backcast = extended[:, :INPUT_LEN]
    assert torch.allclose(backcast, window, atol=TOLERANCE), (
        f"the backcast is out by {float((backcast - window).abs().max()):.3e}; a "
        f"uniform shortfall of {INPUT_LEN / (INPUT_LEN + HORIZON):.6f} is the amplitude "
        "trap of spec 6.3 and means the (L+H)/L scaling is missing"
    )

    # The amplitude has to survive into the forecast too, or the model is flat exactly
    # where it is being asked to predict.
    assert float(extended.abs().max()) == pytest.approx(AMPLITUDE, abs=TOLERANCE)


def test_omitting_the_scale_shrinks_by_exactly_the_ratio():
    """The companion, and the reason the test above is worth having.

    Measured on ``torch.fft`` directly, so it passes today and will keep passing whatever
    GB-41 does. Zero-padding a length-L spectrum to a length-(L+H) inverse transform
    reproduces the signal shape and divides every amplitude by exactly ``L / (L+H)``,
    because ``irfft`` normalises by the *output* length while the spectrum was built from
    the input length. At L=120 and H=4 that is a **3.2 % flattening** - small enough to
    read as a modelling result and large enough to move direction accuracy.
    """
    window = a_sinusoid()
    total = INPUT_LEN + HORIZON

    spectrum = torch.fft.rfft(window, n=INPUT_LEN)
    padded = torch.zeros(1, total // 2 + 1, dtype=spectrum.dtype)
    # Bin k of a length-L transform is frequency k/L; the same frequency sits at bin
    # k*(L+H)/L of a length-(L+H) transform. PERIOD divides both lengths here, so the
    # mapping is exact and the comparison is about amplitude alone.
    for k in range(spectrum.shape[1]):
        target = k * total // INPUT_LEN
        if target < padded.shape[1]:
            padded[0, target] = spectrum[0, k]

    unscaled = torch.fft.irfft(padded, n=total)
    scaled = unscaled * (total / INPUT_LEN)

    # **The ratio is the assertion, not the amplitude.** Re-sampling onto a longer grid
    # moves where the samples fall relative to the peak, so the sampled maximum of the
    # extended signal need not equal AMPLITUDE even when the scaling is perfect. The
    # shortfall the trap causes is uniform and multiplicative, which is exactly what a
    # ratio isolates and an amplitude comparison would confound.
    ratio = float(unscaled.abs().max() / scaled.abs().max())
    assert ratio == pytest.approx(INPUT_LEN / total, abs=1e-12)

    # Every sample, not just the largest: the shortfall is uniform.
    assert torch.allclose(unscaled * (total / INPUT_LEN), scaled, atol=1e-12)
    assert float(unscaled.abs().max()) < float(scaled.abs().max())


def test_the_trap_is_large_enough_to_be_mistaken_for_a_result():
    """Why it costs a day rather than an hour: it does not look like a bug.

    A 3.2 % flattening leaves the shape intact, so the forecast still looks plausible; and
    because a flatter forecast sits closer to zero, **MSE improves**. A reader watching
    MSE would conclude the model got better.
    """
    total = INPUT_LEN + HORIZON
    shortfall = 1.0 - INPUT_LEN / total

    assert 0.01 < shortfall < 0.05  # visible in a metric, invisible in a plot

    truth = a_sinusoid(total)
    flattened = truth * (INPUT_LEN / total)

    mse_flat = float(((flattened - 0.0) ** 2).mean())
    mse_true = float(((truth - 0.0) ** 2).mean())
    assert mse_flat < mse_true  # closer to zero, and that is the whole trap
