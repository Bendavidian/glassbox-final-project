"""L6: exact per-channel attribution.

Attribution is algebra, not approximation. For a linear forecaster the contribution of
channel ``c`` is that channel's own linear map applied to its own input window, and the
contributions sum to the forecast **exactly** — within
:data:`~glassbox.contracts.schemas.EXACTNESS_TOLERANCE`, which is the float32 cast on
``predict``'s output and nothing else. SHAP, LIME, captum and every other
perturbation-based library are banned, and ``tests/explain/test_channel.py`` enforces the
ban rather than leaving it to a docstring.

**Where the algebra lives, and why not here.** ``explain`` sits *above* ``model`` in the
layer stack (spec 3.1), so a model cannot delegate upward without an import cycle and a
weakened architecture guard. The summation therefore lives one layer below both, as
:meth:`glassbox.contracts.schemas.Attribution.from_terms`, and this module and every
forecaster's ``explain`` call the same function. One decomposition, no exception in the
import contract, and the exactness property is enforced where the schema declaring it
lives. Each layer keeps a job: the model says *which* terms exist, which is architecture;
the contract sums them and refuses a residual, which is the invariant; this module is the
face GB-32 and GB-53 call, and it re-checks the model's answer against ``predict`` itself.

**Percentage shares are normalised by the gross, not the net** (ruled in GB-30). See
:func:`shares` for the argument; the short version is that percent-of-net is unbounded and
discontinuous at a zero forecast, which on daily log returns is where the dashboard renders
most often, and a reader who meets "400%" concludes the explanation is broken.

``per_lag`` stays ``None``: the heatmap is GB-31, which is cut, and
``Attribution.per_lag`` is optional for exactly that reason.

Implemented in GB-30.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from glassbox.contracts.schemas import EXACTNESS_TOLERANCE, Attribution


@runtime_checkable
class ChannelLinear(Protocol):
    """A forecaster whose forecast is a sum of per-channel linear terms.

    Structural rather than nominal, so :func:`attribute` works against any model that can
    enumerate its own terms without this module importing a single one of them.
    """

    channels: tuple[str, ...]

    def linear_terms(
        self, x: np.ndarray, channels: tuple[str, ...]
    ) -> dict[str, tuple[tuple[np.ndarray, np.ndarray], ...]]:
        """``{channel: ((W, v), ...)}`` for the single window ``x``."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def explain(self, x: np.ndarray, channels: tuple[str, ...]) -> Attribution: ...


def attribute(
    forecaster: ChannelLinear, x: np.ndarray, channels: tuple[str, ...]
) -> Attribution:
    """The exact per-channel decomposition of one window's forecast.

    Args:
        forecaster: Any model exposing :class:`ChannelLinear`. Every forecaster in
            ``glassbox.model`` does.
        x: One window, ``(L, C)``.
        channels: The channel names, ordered to match ``x``'s last axis.

    Returns:
        An :class:`Attribution` whose ``per_channel`` sums to ``forecast_total`` and whose
        ``forecast_total`` is what ``forecaster.predict`` returns for this window.
        ``per_lag``, ``per_frequency`` and ``gain_phase`` are ``None`` — the first because
        GB-31 is cut, the last two because they are FITS's and belong to
        ``explain.spectral``.

    Raises:
        ValueError: The channels are not the forecaster's, the window has the wrong shape,
            or the decomposition does not close.

    **Why this re-checks a model that already checked itself.** ``forecaster.explain``
    builds its attribution through ``Attribution.from_terms``, which refuses a residual —
    so the sum is already known to close *against the total the model reported*. What is
    not established by that is whether the model reported the right total. This is the
    layer that publishes an explanation to a dashboard and a supervisor, so it asks
    ``predict`` directly and compares. It is one extra forward pass through a linear map,
    for five attributions a day in the live loop, and it turns spec 4.4's property 3 from
    a test into a runtime guarantee.
    """
    attribution = forecaster.explain(x, channels)
    predicted = float(np.asarray(forecaster.predict(x[None, ...])[0]).sum())
    residual = predicted - attribution.forecast_total
    if abs(residual) > EXACTNESS_TOLERANCE:
        raise ValueError(
            f"{type(forecaster).__name__}.explain reports a forecast total of "
            f"{attribution.forecast_total!r}, but predict returns "
            f"{predicted!r} for the same window — a residual of {residual:.3e}. The "
            "attribution is internally consistent and describes a different forecast "
            "from the one the model made, which is the one failure an exactness check "
            "inside the model cannot see."
        )
    return attribution


def shares(attribution: Attribution) -> dict[str, float]:
    """Each channel's share of the **gross** contribution, as a signed fraction.

    ``share[c] = contribution[c] / Σ|contribution|``, so every share lies in ``[-1, +1]``,
    the magnitudes sum to exactly 1, and the sign survives — "this one pushed up, that one
    pushed down" is still readable.

    **Why not percent-of-forecast.** With contributions of ``+0.08`` and ``-0.06`` behind a
    forecast of ``+0.02``, dividing by the net gives **400%** and **-300%**. That
    denominator is unbounded, and worse, it is discontinuous: it blows up and flips sign as
    the forecast crosses zero, which on daily log returns is where this project renders
    most of its explanations. A reader who meets 400% concludes the explanation is broken,
    and on that denominator they are right to. On the gross the same window reads **+57%**
    and **-43%**, which is true, bounded and legible.

    The cancellation those two numbers no longer show is not discarded — it moves to
    :func:`cancellation`, which reports it directly instead of smuggling it into a
    percentage.

    **A forecast of exactly zero is defined, and this never returns NaN.** The gross
    denominator separates the two cases that percent-of-net conflates:

    * **Every contribution is exactly zero** — persistence, on every window. The gross is
      zero too, and every share is ``0.0``. That preserves GB-11's ruling that the baseline
      reports each active channel with a zero rather than an empty mapping: a renderer then
      shows *no channel drove this, because nothing was predicted*, which is the truth,
      rather than a blank panel indistinguishable from a bug.
    * **Contributions that cancel to a zero forecast** — a real model can do this. The
      gross is non-zero, so the shares are well-defined and non-zero, and it is
      :func:`cancellation` that reads 0.0.

    So the ratio is undefined only in the single case where "nothing" is the true answer,
    and there the answer is zero rather than an error. That is the strongest argument for
    this denominator and the reason the zero case needs no special-casing anywhere else.

    ``per_channel`` remains the exact quantity that sums to the forecast. These are a
    rendering aid and never replace it; a panel shows both.
    """
    gross = sum(abs(value) for value in attribution.per_channel.values())
    if gross == 0.0:
        return dict.fromkeys(attribution.per_channel, 0.0)
    return {
        channel: value / gross for channel, value in attribution.per_channel.items()
    }


def cancellation(attribution: Attribution) -> float:
    """How much of the gross channel view survived into the forecast, in ``[0, 1]``.

    ``|Σ contribution| / Σ|contribution|``. One means every channel pointed the same way;
    zero means they cancelled exactly. The ``+0.08 / -0.06`` window reads **0.143**, which
    licenses GB-32 to write *the channels largely cancelled — 14% of the gross view
    survived into the forecast*, a sentence percent-of-net cannot express at all.

    Returns 0.0 when nothing was contributed, matching :func:`shares`: a forecast built
    from nothing has nothing that survived, and NaN in a dashboard is a bug report rather
    than a reading.
    """
    values = attribution.per_channel.values()
    gross = sum(abs(value) for value in values)
    return 0.0 if gross == 0.0 else abs(sum(values)) / gross


__all__ = ["ChannelLinear", "attribute", "cancellation", "shares"]
