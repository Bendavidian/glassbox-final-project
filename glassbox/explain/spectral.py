"""L6: FITS per-frequency attribution - contribution, gain and phase shift per retained
cycle, plus the learned frequency response (spec 6.5).

``forecast = irFFT(W · rFFT(x))`` is linear, so each retained input frequency contributes
an exactly computable amount to each output point. This module turns that into the three
things §6.5 asks for: **which cycle drove this forecast**, **how the model transforms that
cycle**, and the frequency-response curve GB-46 plots.

**Everything here is keyed by period in DAYS, never by bin index.** Bin 7 means nothing to
a reader and "the 17-day cycle" means something to everybody; the conversion is
``period = L / k``, and it is done once, here, rather than in each caller.

**Two things about the zero-frequency entry, and they are different things.**

``per_frequency[inf]`` is the **RIN mean's** contribution. RIN subtracts each window's mean
before the transform and adds it back after, so the mean reaches the forecast *without
passing through the complex layer at all*. It is part of the forecast and belongs to no
learned frequency, and a decomposition that omitted it could not close - spec 4.4's
tolerance would catch it, and ``Attribution.from_terms``'s own error message names this
case. It is often the largest single entry, because a smoothed extrapolation of a drifting
series is mostly its drift.

``gain_phase[inf]`` is the **learned bin-0 row**, and it is dead. RIN has already removed
the mean, so the rFFT's bin 0 - which *is* that mean - is zero on every window and row 0 of
the weight multiplies nothing (GB-44: 50 of the 1,200 parameters, 4.17%, allocated and
unable to learn). Its measured contribution is identically zero and it is **reported rather
than dropped**, for the reason FITS attributes 0.0 to channels it does not read: an absent
entry and a zero entry say different things, and a reader who finds no bin 0 wonders
whether it was forgotten.

**Exactness is not re-implemented here.** The per-frequency contributions go through
:meth:`Attribution.from_terms`, the one function in this codebase allowed to sum a
decomposition, so the frequency view is refused on a residual exactly as the channel view
is. The matrices it sums are measured from the model's own forward pass
(:meth:`FITSForecaster.frequency_matrices`), not re-derived from the pipeline description
above - a second copy of the pipeline written for the explanation is a second copy that can
drift from the one that forecasts.

Implemented in GB-45. The frequency-response visualisation is GB-46.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Protocol, runtime_checkable

import numpy as np

from glassbox.contracts.schemas import Attribution
from glassbox.explain.channel import attribute

# The period of the zero-frequency component. Infinite rather than a sentinel: a cycle of
# zero frequency genuinely has no finite period, and `math.inf` sorts and prints correctly
# where -1 or 0 would need explaining at every use.
DC_PERIOD = math.inf

# A phase shift is only meaningful against a finite period, so the DC entry reports 0.0.
# The row is dead in any case (see the module docstring).
NO_SHIFT = 0.0


@runtime_checkable
class Spectral(Protocol):
    """A forecaster that can enumerate its own per-frequency maps.

    Structural rather than nominal, matching :class:`~glassbox.explain.channel.
    ChannelLinear`: this module names no model class, so a second spectral architecture
    would be explained by it without an edit here.
    """

    input_len: int
    horizon: int
    cof: int
    out_bins: int
    weight: np.ndarray
    channels: tuple[str, ...]
    cutoff_period_days: int

    def frequency_matrices(self) -> dict[int, np.ndarray]: ...

    def mean_matrix(self) -> np.ndarray: ...

    def forecast_matrix(self) -> np.ndarray: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


def period_days(bin_index: int, input_len: int) -> float:
    """Bin ``k`` of a length-``L`` transform, as a period in days.

    ``L / k``, and :data:`DC_PERIOD` for ``k = 0``. This is the *input* grid's reading; the
    same physical frequency sits at bin ``k·η`` of the output grid, which is what the
    learned layer exists to interpolate across (§6.5, GB-42) and is not a different period.
    """
    if bin_index < 0:
        raise ValueError(f"bin index must not be negative, got {bin_index}")
    return DC_PERIOD if bin_index == 0 else input_len / bin_index


def per_frequency(model: Spectral, x: np.ndarray) -> dict[float, float]:
    """``{period in days: contribution}`` for one window, summing to the forecast.

    Args:
        model: A fitted FITS-like forecaster.
        x: One window, ``(L, C)`` or ``(L,)``. Only the close column is read; FITS is
            univariate by design (§6.4).

    Returns:
        One entry per retained bin, keyed by period, **plus** :data:`DC_PERIOD` carrying
        the RIN mean. The values sum to ``predict(x).sum()`` within
        ``EXACTNESS_TOLERANCE`` - enforced, not asserted afterwards.

    Raises:
        ValueError: The decomposition does not close.
    """
    closes = _close_column(model, x)
    total = float(np.asarray(model.forecast_matrix() @ closes).sum())
    return _frequency_view(model, closes, total, scale=1.0)


def gain_phase(model: Spectral) -> dict[float, tuple[float, float]]:
    """``{period in days: (gain, phase shift in days)}`` - how the model treats each cycle.

    The complex weight is a full ``(COF, out_bins)`` matrix, so a cycle's row has several
    entries. The one reported is the **same-frequency** element, ``W[k, round(η·k)]``: the
    output bin that carries the input bin's own frequency onto the longer grid. Its
    magnitude is the amplitude gain and its argument the phase shift, which is what complex
    multiplication means and what §6.5's narrative sentence quotes.

    **Phase is converted to days**, ``φ / 2π × period``, because radians of an unnamed
    cycle are not a thing anyone can picture. A positive shift means the model **advances**
    that cycle: its output leads the input by that many days.

    The off-diagonal weight in each row is real and is not reported here - it is the
    leakage between neighbouring output bins, and it is already inside
    :func:`per_frequency`'s contributions, which are exact.
    """
    eta = (model.input_len + model.horizon) / model.input_len
    view: dict[float, tuple[float, float]] = {}
    for index in range(model.cof):
        column = min(round(index * eta), model.out_bins - 1)
        element = model.weight[index, column]
        period = period_days(index, model.input_len)
        shift = (
            NO_SHIFT
            if math.isinf(period)
            else float(np.angle(element)) / (2.0 * math.pi) * period
        )
        view[period] = (float(np.abs(element)), shift)
    return view


def frequency_response(model: Spectral) -> tuple[np.ndarray, np.ndarray]:
    """``(periods, gains)`` sorted by period, for GB-46's plot.

    The single strongest visual in the demonstration is ``|W|`` against period: it is what
    the bot learned to listen to, and there is no equivalent for a non-linear model. The
    DC entry is **excluded** - an infinite period cannot be placed on an axis, and the row
    is dead in any case - so the curve runs from the cutoff period up to ``L`` days.
    """
    finite = {
        period: gain
        for period, (gain, _) in gain_phase(model).items()
        if math.isfinite(period)
    }
    periods = np.array(sorted(finite), dtype="float64")
    return periods, np.array([finite[period] for period in periods], dtype="float64")


def dominant_period(model: Spectral, x: np.ndarray) -> tuple[float, float]:
    """``(period, share)`` - the cycle carrying the largest share of the gross view.

    **Share of the gross, not of the net**, for ``explain.channel.shares``' reason: percent
    of a near-zero forecast is unbounded and flips sign as the forecast crosses zero, which
    on daily log returns is most of the time. The DC entry competes on the same footing,
    and it frequently wins - which is a result about the model, not a defect in the metric.
    """
    contributions = per_frequency(model, x)
    gross = sum(abs(value) for value in contributions.values())
    period = max(contributions, key=lambda key: abs(contributions[key]))
    return period, (0.0 if gross == 0.0 else abs(contributions[period]) / gross)


def explain_spectral(
    model: Spectral,
    x: np.ndarray,
    channels: tuple[str, ...],
    scale: float = 1.0,
) -> Attribution:
    """The full explanation of one window: per channel **and** per frequency.

    Args:
        model: A fitted FITS-like forecaster.
        x: One window, ``(L, C)``.
        channels: The channel names, ordered to match ``x``'s last axis.
        scale: The symbol's target deviation, to publish in raw log returns rather than in
            the scaled unit the model was fitted in. Both views carry it, so they remain
            comparable with each other and with the forecast.

    Returns:
        An :class:`Attribution` whose ``per_channel`` is
        :func:`~glassbox.explain.channel.attribute`'s and whose ``per_frequency`` and
        ``gain_phase`` are this module's. **Both decompositions close against the same
        total**, which is the property that lets a reader move between the two views
        without wondering whether they describe the same forecast.

    Raises:
        ValueError: Either decomposition does not close, or the channels are not the
            model's.
    """
    channel_view = attribute(model, x, channels, scale=scale)
    closes = _close_column(model, x)
    frequency_view = _frequency_view(
        model, closes, channel_view.forecast_total / scale, scale=scale
    )
    return replace(
        channel_view,
        per_frequency=frequency_view,
        gain_phase=gain_phase(model),
    )


def _frequency_view(
    model: Spectral, closes: np.ndarray, total: float, scale: float
) -> dict[float, float]:
    """The per-frequency contributions, summed by ``Attribution.from_terms`` and no other.

    ``from_terms`` keys on strings and this view keys on periods, so the labels are mapped
    out and back. That is worth the small indirection: the alternative is a second place in
    the codebase that adds contributions and compares them to a total, and the whole point
    of GB-30's ruling is that there is exactly one.
    """
    matrices = {DC_PERIOD: model.mean_matrix()}
    for index, matrix in model.frequency_matrices().items():
        period = period_days(index, model.input_len)
        matrices[period] = matrices.get(period, 0.0) + matrix

    labels = {f"period_{period}": period for period in matrices}
    terms = {
        label: ((matrices[period] * scale, closes),) for label, period in labels.items()
    }
    closed = Attribution.from_terms(terms, total * scale)
    return {labels[label]: value for label, value in closed.per_channel.items()}


def _close_column(model: Spectral, x: np.ndarray) -> np.ndarray:
    """The window's close-log-return column, as ``(L,)`` float64."""
    window = np.asarray(x, dtype="float64")
    if window.ndim == 1:
        column = window
    elif window.ndim == 2:
        column = window[:, model.channels.index("close_logret")]
    else:
        raise ValueError(f"expected one window, (L, C) or (L,), got {window.shape}")
    if column.shape[0] != model.input_len:
        raise ValueError(
            f"this model reads {model.input_len} bars and the window has "
            f"{column.shape[0]}"
        )
    return column


__all__ = [
    "DC_PERIOD",
    "NO_SHIFT",
    "Spectral",
    "dominant_period",
    "explain_spectral",
    "frequency_response",
    "gain_phase",
    "per_frequency",
    "period_days",
]
