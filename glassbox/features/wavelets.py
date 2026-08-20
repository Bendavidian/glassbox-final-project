"""L2: causal rolling DWT approximation channels wav_a1..wav_a3.

The decomposition runs over a trailing window only, and emits values from the first
fully populated window onward. Centring is forbidden.

**Why this is not how wavelets are usually applied to a price series, and why it must
not be.** The standard recipe transforms the *whole* series once and takes the resulting
bands as features. That leaks, and not marginally: a DWT filter is **two-sided**, so the
coefficient covering time ``t`` is a weighted sum of samples on both sides of ``t``, and
the reconstruction at ``t`` therefore carries information from ``t+1``, ``t+2`` and
further — at level 3 with ``db4``, from tens of bars ahead. A model fed those bands is
being shown the future in a form that looks exactly like a smoothing, and every metric
downstream improves. ``tests/features/test_wavelets.py`` demonstrates the leak by
building the whole-series version and watching GB-10's harness reject it, so the claim
here is a measurement rather than a warning.

**So: one window per bar.** For each ``t`` the trailing ``rolling_window`` returns are
decomposed on their own, and the **last** sample of each reconstructed band is the value
at ``t``. The window ending at ``t`` contains no bar after ``t``, so causality holds by
construction of the slice rather than by a check afterwards — the same argument that makes
``builder.build_windows`` causal.

**Exact, not asymptotic.** Unlike Wilder's RSI, whose seed decays geometrically and whose
warm-up is therefore a tolerance argument (see ``indicators.RSI_WARMUP``), a DWT of a
trailing window depends on that window and on nothing before it. Two callers holding the
same ``rolling_window`` bars compute the same number, exactly. The warm-up is the window
length and there is no residue to bound.

**The bands are additive by construction**, which is what makes GB-48's check meaningful:
``A_j = A_{j+1} + D_{j+1}``, so ``signal = a3 + d3 + d2 + d1`` to floating-point precision.
The details are produced here even though no channel uses them, because a decomposition
whose parts cannot be summed back to the whole is one nobody can check.

Implemented in GB-47.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pywt

from glassbox.config.loader import Config
from glassbox.data.historical import LOG_RETURN

# The channels this module publishes, and the level each one reconstructs.
APPROXIMATION_CHANNELS = ("wav_a1", "wav_a2", "wav_a3")


def rolling_window(cfg: Config) -> int:
    """Bars per decomposition, and therefore the warm-up.

    Named rather than read inline because ``builder.PARITY_WARMUP`` needs the same number
    and a second literal is how two answers to one question start to differ.
    """
    return cfg.wavelet.rolling_window


def band_values(window: np.ndarray, cfg: Config) -> dict[str, np.ndarray]:
    """Every reconstructed band of one window, as full-length arrays.

    Args:
        window: ``(W,)`` real samples — a trailing slice of the log-return series.
        cfg: Resolved configuration; supplies the family, level count and padding mode.

    Returns:
        ``{"a1".."aN", "d1".."dN"}``, each the length of ``window``. ``a_j`` is the level-
        ``j`` approximation and ``d_j`` the level-``j`` detail, so that
        ``a_j == a_{j+1} + d_{j+1}`` and ``window == aN + dN + ... + d1``.

    Raises:
        ValueError: The window is not one-dimensional, is shorter than the configured
            window, or holds a NaN.

    **Reconstructed rather than returned as coefficients**, because a coefficient vector is
    shorter than the window and lives on its own time grid — there is no "the value at
    ``t``" to read off it. Zeroing every other band and inverting puts each band back on
    the input's grid, where the last sample is the value at the window's last bar.
    """
    # A copy, not a view: `pywt` writes through the buffer it is handed, so a read-only
    # array — which is what a pandas column hands back — raises rather than computes. The
    # copy is also what makes this function pure in the sense spec §3 requires.
    values = np.array(window, dtype="float64")
    if values.ndim != 1:
        raise ValueError(f"band_values takes one window, (W,), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("the window holds a NaN or an infinity")

    levels = cfg.wavelet.levels
    coefficients = pywt.wavedec(
        values, wavelet=cfg.wavelet.family, mode=cfg.wavelet.mode, level=levels
    )

    bands: dict[str, np.ndarray] = {}
    for level in range(1, levels + 1):
        bands[f"d{level}"] = _rebuild(
            coefficients, cfg, keep={_detail_index(level, levels)}, length=len(values)
        )
        # A_j keeps the coarsest approximation plus every detail *slower* than level j.
        kept = {0} | {
            _detail_index(deeper, levels) for deeper in range(level + 1, levels + 1)
        }
        bands[f"a{level}"] = _rebuild(coefficients, cfg, keep=kept, length=len(values))
    return bands


def approximation(bars: pd.DataFrame, cfg: Config, level: int) -> pd.Series:
    """The causal rolling level-``level`` approximation of the log-return series.

    Args:
        bars: Canonical bar frame carrying ``log_return``.
        cfg: Resolved configuration.
        level: 1, 2 or 3 — which approximation to publish.

    Returns:
        A Series named ``wav_a{level}``, NaN until the first fully populated window and
        float64 thereafter.

    Raises:
        ValueError: ``level`` is outside the configured level count, or ``log_return`` is
            missing.

    Every value is the **last sample** of a decomposition of the ``rolling_window`` returns
    ending at that bar. Nothing after the bar enters it, and the row is NaN rather than
    computed from a short window — a number that looks like a wavelet band but came from
    twelve bars is worse than a gap, because only the gap is visible downstream.
    """
    if not 1 <= level <= cfg.wavelet.levels:
        raise ValueError(
            f"level {level} is outside the configured 1..{cfg.wavelet.levels}"
        )
    if LOG_RETURN not in bars.columns:
        raise ValueError(f"the bars frame has no {LOG_RETURN!r} column")

    # `copy=True`: pandas hands back a read-only view and `pywt` writes through its
    # buffer, so a view would raise rather than compute — and copying is also what keeps
    # this function pure, since the caller's frame can never be touched.
    series = bars[LOG_RETURN].astype("float64").to_numpy(copy=True)
    width = rolling_window(cfg)
    out = np.full(len(series), np.nan, dtype="float64")

    # One `wavedec` and one `waverec` per bar rather than the six `band_values` would
    # rebuild: the details are needed by GB-48's additivity check and by nothing in the
    # pipeline, and this loop runs once per bar per symbol.
    kept = {0} | {
        _detail_index(deeper, cfg.wavelet.levels)
        for deeper in range(level + 1, cfg.wavelet.levels + 1)
    }

    for end in range(width - 1, len(series)):
        window = series[end - width + 1 : end + 1]
        if not np.isfinite(window).all():
            continue
        coefficients = pywt.wavedec(
            window,
            wavelet=cfg.wavelet.family,
            mode=cfg.wavelet.mode,
            level=cfg.wavelet.levels,
        )
        out[end] = _rebuild(coefficients, cfg, keep=kept, length=width)[-1]

    return pd.Series(out, index=bars.index, name=f"wav_a{level}")


def whole_series_approximation(
    bars: pd.DataFrame, cfg: Config, level: int
) -> pd.Series:
    """**The leaking version, kept deliberately.** Transforms the whole series at once.

    This is the standard recipe, and GB-48 hands it to GB-10's causality harness to watch
    the harness reject it. Keeping it in the module rather than in the test file is the
    same choice as ``model.BatchNormForecaster``: the thing a guard is supposed to catch
    should exist, or "the guard works" is an assertion about code nobody wrote.

    **Never call this from the pipeline.** ``build_feature_frame`` does not know it exists.
    """
    series = bars[LOG_RETURN].astype("float64").to_numpy()
    usable = np.isfinite(series)
    out = np.full(len(series), np.nan, dtype="float64")
    if usable.sum() >= cfg.wavelet.levels * 2:
        out[usable] = band_values(series[usable], cfg)[f"a{level}"]
    return pd.Series(out, index=bars.index, name=f"wav_a{level}")


def _rebuild(
    coefficients: list[np.ndarray], cfg: Config, keep: set[int], length: int
) -> np.ndarray:
    """Invert ``coefficients`` with every position outside ``keep`` zeroed."""
    parts = [
        part if index in keep else np.zeros_like(part)
        for index, part in enumerate(coefficients)
    ]
    rebuilt = pywt.waverec(parts, wavelet=cfg.wavelet.family, mode=cfg.wavelet.mode)
    # `waverec` returns one extra sample for an odd-length input; the reconstruction is
    # aligned at the start, so the head is the window.
    return np.asarray(rebuilt[:length], dtype="float64")


def _detail_index(level: int, levels: int) -> int:
    """Where ``cD_level`` sits in ``wavedec``'s output ``[cA_n, cD_n, ..., cD_1]``."""
    return levels - level + 1


__all__ = [
    "APPROXIMATION_CHANNELS",
    "approximation",
    "band_values",
    "rolling_window",
    "whole_series_approximation",
]
