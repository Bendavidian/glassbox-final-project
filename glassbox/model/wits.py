"""L3: `WITSForecaster` - RIN, causal DWT, band retention, one real linear map per
retained band, inverse DWT (GB-66).

**Why this model exists, and it is not to trade.** GB-46 measured that ~86% of FITS's
learned frequency response is grid geometry rather than market structure: the white-noise
curve correlates with the real one at **+0.9485**, and the mechanism is
``frac(eta*k) = frac(k*H/L)`` - a frequency that does not land on an output bin must be
interpolated across, on every window, by the learned layer. That is a claim about FITS, or
a claim about extending a *global* basis, and one architecture cannot tell the two apart.

WITS puts a DWT where FITS puts the rFFT and asks whether the pathology reproduces. The
prediction was recorded before a line was written: **if the geometry critique is correct,
WITS will not show ~86% data-independence under the null control. If it does, the critique
is wrong, and that is worth as much.**

**Why the interpolation has no counterpart here, measured.** At ``L=120, H=4, db4, J=3``
``wavedec`` returns bands of ``[21, 21, 35, 63]`` and at ``L+H=124`` it returns
``[21, 21, 36, 65]``. The **retained** bands are length 21 on both sides, so the per-band
maps are *square* and the length extension comes entirely from reconstructing with the
**discarded** detail bands zero-padded at their longer lengths. Nothing is remapped onto a
shifted grid. That is the structural difference the null control exists to test.

**No amplitude trap** (contrast spec 6.3). ``irfft`` normalises by output length, so FITS
must multiply back by ``(L+H)/L`` - a uniform 0.9677 at the configured geometry.
``waverec`` carries no such factor: measured with the maps set to the identity, a 60-day
sinusoid reconstructs at a ratio of **1.0008**, and the ratio does not sit at a constant
across periods (0.94 to 1.07) but wanders with how much of the signal the discarded d1/d2
bands carried. There is no uniform scale to restore, and ``test_wits.py`` records that as a
measurement rather than this docstring asserting it.

**The whole pipeline is linear in the input window**, mean-subtraction and its inverse
included, so ``predict(x) == M @ x_close`` for an exact ``(H, L)`` matrix - the same
property that lets FITS attribute exactly without a perturbation library.

**And the DWT is *itself* a matrix**, which is what makes the torch training path possible
at all: torch has no wavelet transform, so a gradient path through ``pywt`` does not exist.
:func:`analysis_matrix` and :func:`synthesis_matrix` measure the operator by pushing basis
vectors through ``pywt``, and every consumer - numpy inference, torch training, attribution
- multiplies by those same two arrays. There is no second implementation to drift from.
Verified exact to **5e-16**.

**Univariate by design** (spec 6.4), like FITS: WITS reads ``close_logret`` and attributes
to every active channel, the channels it does not read contributing no terms at all.

Implemented in GB-66.
"""

from __future__ import annotations

import json
import math
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pywt

from glassbox.config.loader import VALID_BOUNDARIES, Config
from glassbox.contracts.schemas import Attribution, FitProvenance, WindowBatch
from glassbox.model.history import EpochLoss

CHECKPOINT_VERSION = 1

# The one channel WITS reads. Spec 6.4: the transform pipeline is univariate by design.
CLOSE_CHANNEL = "close_logret"

#: The key :attr:`Attribution.per_level` carries the RIN mean under. It is not a band, and
#: it has to be there: RIN reaches the forecast without passing through any learned map, so
#: a decomposition over bands alone would not close to the forecast.
MEAN_KEY = "rin_mean"


def band_lengths(
    length: int, family: str, levels: int, boundary: str
) -> tuple[int, ...]:
    """``[cA_J, cD_J, cD_J-1, ..., cD_1]`` lengths for a window of ``length``.

    Measured from ``pywt`` rather than derived from the filter length and the level count,
    for the reason :func:`analysis_matrix` gives: the arithmetic is easy to get subtly
    wrong at the boundary, and there is no reason to hold a second copy of it.
    """
    zeros = np.zeros(length, dtype="float64")
    return tuple(
        len(c) for c in pywt.wavedec(zeros, family, mode=boundary, level=levels)
    )


def band_names(levels: int, retained_bands: int) -> tuple[str, ...]:
    """``("a3", "d3")`` at ``J=3`` keeping two - the keys ``per_level`` is written under.

    Named for the band rather than numbered, because ``Attribution.per_level`` is read by a
    person: ``"a3"`` says which scale where ``0`` says which slot.
    """
    names = [f"a{levels}"] + [f"d{levels - k}" for k in range(levels)]
    return tuple(names[:retained_bands])


@cache
def analysis_matrix(
    input_len: int, family: str, levels: int, boundary: str, retained_bands: int
) -> np.ndarray:
    """The exact ``(R, L)`` matrix with ``A @ x == concat(wavedec(x)[:retained_bands])``.

    **Measured from ``pywt``, not re-derived from it.** A DWT with a fixed wavelet, level
    count and boundary mode is a linear operator - the extension, the filtering and the
    decimation are each linear - so the map is a matrix whose columns are ``f(e_i)``.
    Pushing the ``L`` basis windows through the library gives those columns from the code
    that actually transforms, which is the same discipline
    :meth:`WITSForecaster.forecast_matrix` applies one level up.

    **This is also what makes ``fit`` possible at all**, and it is the one place WITS
    departs from FITS's structure rather than mirroring it. FITS calls ``torch.fft.rfft``
    in its loss; torch has no wavelet transform, so there is no gradient path through
    ``pywt``. Once the transform is a matrix the whole forward pass is matmuls -
    differentiable, and shared array-for-array with inference.

    Returned read-only because it is cached and shared.
    """
    columns = []
    for index in range(input_len):
        impulse = np.zeros(input_len, dtype="float64")
        impulse[index] = 1.0
        coeffs = pywt.wavedec(impulse, family, mode=boundary, level=levels)
        columns.append(np.concatenate(coeffs[:retained_bands]))
    matrix = np.stack(columns, axis=1)
    matrix.flags.writeable = False
    return matrix


@cache
def synthesis_matrix(
    input_len: int,
    horizon: int,
    family: str,
    levels: int,
    boundary: str,
    retained_bands: int,
) -> np.ndarray:
    """The exact ``(L+H, R_out)`` matrix reconstructing from the retained bands alone.

    The discarded detail bands are zeroed **at their ``L+H`` lengths**, and that is what
    lengthens the series: spec 6.2's zero-pad, in the wavelet basis. Nothing is stretched or
    remapped onto a shifted grid, because the retained bands are the same length on both
    sides - which is the structural claim the whole task turns on.

    Returned read-only because it is cached and shared.
    """
    total = input_len + horizon
    shapes = band_lengths(total, family, levels, boundary)
    width = sum(shapes[:retained_bands])

    columns = []
    for index in range(width):
        unit = np.zeros(width, dtype="float64")
        unit[index] = 1.0
        parts, offset = [], 0
        for size in shapes[:retained_bands]:
            parts.append(unit[offset : offset + size])
            offset += size
        parts.extend(
            np.zeros(size, dtype="float64") for size in shapes[retained_bands:]
        )
        columns.append(np.asarray(pywt.waverec(parts, family, mode=boundary))[:total])
    matrix = np.stack(columns, axis=1)
    matrix.flags.writeable = False
    return matrix


def n_parameters_for(
    input_len: int,
    horizon: int,
    family: str,
    levels: int,
    boundary: str,
    retained_bands: int,
) -> int:
    """Learnable reals: one map per retained band, block-diagonal.

    **882** at the configured geometry, against FITS's 1,200 allocated and DLinear's 4,800.
    Real weights throughout, so the reals-versus-complex counting question FITS's
    ``n_parameters`` has to answer does not arise here.

    Derived here and read by the property, so a sweep reporting the count per geometry and
    the model holding it cannot disagree.
    """
    inside = band_lengths(input_len, family, levels, boundary)[:retained_bands]
    outside = band_lengths(input_len + horizon, family, levels, boundary)[
        :retained_bands
    ]
    return sum(a * b for a, b in zip(inside, outside, strict=True))


class WITSForecaster:
    """Wavelet Interpolation Time Series forecasting: one real map per retained band.

    Attributes:
        weights: one ``(in_b, out_b)`` real matrix per retained band, block-diagonal by
            construction. **The block-diagonality is not an efficiency; it is what buys
            ``per_level``.** A single joint map over the concatenated bands (42x42, 1,764
            reals) would let a forecast sample depend on both bands through weights with no
            canonical split, and band totals would stop being a decomposition and become an
            attribution heuristic. Exactness is this project's thesis, and this is the one
            place it would have been quietly abandoned.
    """

    def __init__(
        self,
        input_len: int,
        horizon: int,
        channels: tuple[str, ...],
        family: str,
        levels: int,
        boundary: str,
        retained_bands: int,
        shift_invariant: bool = False,
        cfg: Config | None = None,
    ) -> None:
        """Shape comes from arguments rather than from ``cfg``, for :meth:`load`'s sake.

        A checkpoint that could not say which wavelet, how many levels and which boundary
        produced its weights would be unloadable - the same reason FITS passes
        ``cutoff_period_days`` rather than reading it back off a config file that has since
        moved.
        """
        if CLOSE_CHANNEL not in channels:
            raise ValueError(
                f"WITS forecasts {CLOSE_CHANNEL!r} and this channel set does not contain "
                f"it: {list(channels)}. The transform pipeline is univariate by design "
                "(spec 6.4); it reads one series and attributes to all of them"
            )
        if boundary not in VALID_BOUNDARIES:
            raise ValueError(
                f"boundary must be one of {list(VALID_BOUNDARIES)}, got {boundary!r}"
            )
        if levels <= 0:
            raise ValueError(f"levels must be positive, got {levels}")
        if not 1 <= retained_bands <= levels + 1:
            raise ValueError(
                f"retained_bands must be between 1 and {levels + 1} for {levels} levels, "
                f"got {retained_bands}"
            )
        if shift_invariant:
            # A refusal rather than a stub, and rather than a docstring saying "later".
            # SWT is undecimated, so every band stays length L and one map per band costs
            # 120*124*2 = 29,760 reals - 25x FITS, which does not change the capacity
            # argument so much as remove it. It is a real axis and GB-66 asks for it
            # measured, but `pywt.iswt` requires a length divisible by 2**levels and
            # L+H = 124 is not divisible by 8, so it needs a padding rule that has not been
            # decided. `test_wits.py` carries a strict xfail against this message so the
            # gap fails rather than waits to be remembered.
            raise NotImplementedError(
                "the shift-invariant (SWT) axis is not built: pywt.iswt requires a length "
                f"divisible by 2**levels = {2**levels}, and input_len + horizon = "
                f"{input_len + horizon} is not. Closing it needs a padding rule, and the "
                "arm costs 29,760 reals against the DWT's 882, so it is not capacity "
                "matched to FITS and has to be reported as its own arm"
            )

        self.name = "wits"
        self.input_len = input_len
        self.horizon = horizon
        self.channels = tuple(channels)
        self.family = family
        self.levels = levels
        self.boundary = boundary
        self.retained_bands = retained_bands
        self.shift_invariant = shift_invariant
        self.fitted: FitProvenance | None = None
        self._cfg = cfg

        # How the last fit went. Not model state and not in the checkpoint - a model
        # rebuilt by `load` reports an empty history, which is the truth: it did not train.
        self.history: tuple[EpochLoss, ...] = ()
        self.best_epoch: int | None = None
        self.stopped_early = False

        self.weights = [
            np.zeros((inside, outside), dtype="float64")
            for inside, outside in zip(self.in_lengths, self.out_lengths, strict=True)
        ]
        self._matrix: np.ndarray | None = None
        self._mean_matrix: np.ndarray | None = None
        self._band_matrices: dict[str, np.ndarray] | None = None

    # ── the shape the wavelet decides ────────────────────────────────────────

    @property
    def in_lengths(self) -> tuple[int, ...]:
        """Retained band lengths over the input window."""
        return band_lengths(self.input_len, self.family, self.levels, self.boundary)[
            : self.retained_bands
        ]

    @property
    def out_lengths(self) -> tuple[int, ...]:
        """Retained band lengths over the extended window.

        Equal to :attr:`in_lengths` at the configured geometry, which is the whole point:
        the maps are square and nothing is remapped onto a shifted grid.
        """
        return band_lengths(
            self.input_len + self.horizon, self.family, self.levels, self.boundary
        )[: self.retained_bands]

    @property
    def bands(self) -> tuple[str, ...]:
        """``("a3", "d3")`` - the keys :attr:`Attribution.per_level` is written under."""
        return band_names(self.levels, self.retained_bands)

    @property
    def n_parameters(self) -> int:
        """Learnable reals, block-diagonal: **882** at the configured geometry.

        **Unlike FITS there is no dead row.** RIN removes the window mean, and in a Fourier
        basis the mean *is* bin 0, so FITS allocates a row that multiplies zero on every
        forward pass - 50 reals, 4.17%, allocated and unusable. A wavelet basis spreads the
        mean across every ``cA`` coefficient, because db4's scaling filter has non-zero DC
        gain, so no row is identically silenced and the allocated count is the effective
        one.
        """
        return sum(int(weight.size) for weight in self.weights)

    # ── the Forecaster protocol ──────────────────────────────────────────────

    def fit(self, batch: WindowBatch, val: WindowBatch | None = None) -> None:
        """Fit the per-band maps by gradient descent, supervised on backcast and forecast.

        ``B+F`` as spec 5 fixes it for FITS, and for the same reason: the backcast term is
        what teaches a map to place an input band on the right output band. Hyperparameters
        come from ``cfg.model``, exactly as DLinear's and FITS's do.

        The forward pass here is the torch mirror of :meth:`_extend`, and it is a mirror
        that cannot drift: both multiply by the *same* :func:`analysis_matrix` and
        :func:`synthesis_matrix` arrays.
        """
        import torch

        if self._cfg is None:
            raise ValueError(
                "this model was rebuilt from a checkpoint and holds no training "
                "configuration; construct it with a Config to fit it"
            )
        self._require_batch(batch)
        plan = self._cfg.model
        generator = torch.Generator().manual_seed(self._cfg.meta.seed)

        inputs, targets = self._tensors(batch, torch)
        blocks = [
            torch.zeros(weight.shape, dtype=torch.float64).requires_grad_(True)
            for weight in self.weights
        ]
        optimiser = torch.optim.Adam(blocks, lr=plan.lr)

        validation = None if val is None else self._tensors(val, torch)
        best_loss, best_state, waited = math.inf, None, 0
        curve: list[EpochLoss] = []
        self.best_epoch, self.stopped_early = None, False

        windows = targets.shape[0]
        size = windows if plan.batch_size is None else plan.batch_size
        # See DLinear and FITS: permuting a single batch changes only the order floats are
        # summed in, which loses bit-identical weights across seeds for nothing.
        one_batch = size >= windows

        for epoch in range(1, plan.epochs + 1):
            order = (
                torch.arange(windows)
                if one_batch
                else torch.randperm(windows, generator=generator)
            )
            for start in range(0, windows, size):
                rows = order[start : start + size]
                optimiser.zero_grad()
                self._loss_tensor(torch, blocks, inputs[rows], targets[rows]).backward()
                optimiser.step()

            with torch.no_grad():
                score = (
                    None
                    if validation is None
                    else float(self._loss_tensor(torch, blocks, *validation))
                )
                curve.append(
                    EpochLoss(
                        epoch=epoch,
                        train_loss=float(
                            self._loss_tensor(torch, blocks, inputs, targets)
                        ),
                        val_loss=score,
                    )
                )

            if score is None:
                continue
            if score < best_loss:
                best_loss, waited = score, 0
                self.best_epoch = epoch
                best_state = [block.detach().clone() for block in blocks]
            else:
                waited += 1
                if waited >= plan.patience:
                    self.stopped_early = True
                    break

        final = best_state if best_state is not None else blocks
        self._set_weights([block.detach().numpy() for block in final])
        self.history = tuple(curve)
        self.fitted = FitProvenance.from_batch(batch, self.input_len, self.horizon)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(B, L, C) float32 -> (B, H) float32."""
        self._require_window_shape(X)
        closes = np.asarray(X, dtype="float64")[:, :, self._close_position]
        return self._extend(closes)[:, self.input_len :].astype("float32")

    def backcast(self, X: np.ndarray) -> np.ndarray:
        """(B, L, C) -> (B, L), the reconstruction the B+F loss supervises.

        Public because the report shows it: a backcast that tracks the input is the visible
        evidence that the retained bands carried the signal rather than the noise.
        """
        self._require_window_shape(X)
        closes = np.asarray(X, dtype="float64")[:, :, self._close_position]
        return self._extend(closes)[:, : self.input_len].astype("float32")

    def forecast_matrix(self) -> np.ndarray:
        """The exact ``(H, L)`` matrix with ``predict(x) == M @ x_close``.

        Measured by pushing the ``L`` basis windows through :meth:`_extend`, so the
        attribution cannot drift from the forecast the way a hand-written derivation would.
        """
        if self._matrix is None:
            basis = np.eye(self.input_len, dtype="float64")
            self._matrix = self._extend(basis)[:, self.input_len :].T.copy()
        return self._matrix

    def mean_matrix(self) -> np.ndarray:
        """The ``(H, L)`` map the **RIN mean** carries, with every band silenced.

        RIN subtracts the window mean and adds it back, so the mean reaches the forecast
        without passing through any learned map. It belongs to no band, so a decomposition
        that omitted it could not close - spec 4.4's tolerance would catch it.

        Measured by running the forward pass with every band masked, so it is whatever the
        implementation does rather than what this docstring says it does.
        """
        if self._mean_matrix is None:
            basis = np.eye(self.input_len, dtype="float64")
            self._mean_matrix = self._extend(basis, keep=())[
                :, self.input_len :
            ].T.copy()
        return self._mean_matrix

    def level_matrices(self) -> dict[str, np.ndarray]:
        """``{band name: (H, L)}`` - what each retained band contributes.

        Together with :meth:`mean_matrix` these sum **exactly** to
        :meth:`forecast_matrix`, and exactly means at float64 round-off rather than at spec
        4.4's 1e-5: the bands are separable and reconstruction is additive, so the identity
        is algebraic. Measured the way the total is - run the real forward pass with one
        band unmasked and subtract the mean path every masked pass still carries.
        """
        if self._band_matrices is None:
            basis = np.eye(self.input_len, dtype="float64")
            background = self.mean_matrix()
            self._band_matrices = {
                band: (
                    self._extend(basis, keep=(band,))[:, self.input_len :].T
                    - background
                ).copy()
                for band in self.bands
            }
        return self._band_matrices

    def linear_terms(
        self, x: np.ndarray, channels: tuple[str, ...]
    ) -> dict[str, tuple[tuple[np.ndarray, np.ndarray], ...]]:
        """(L, C) -> ``{channel: ((W, v), ...)}``, the terms ``predict`` sums.

        One pair for ``close_logret`` and **no terms at all** for every other active
        channel, exactly as FITS's and persistence's do - a model that does not read a
        channel has no term in it, and the ``0.0`` falls out of
        :meth:`Attribution.from_terms` rather than being written down here.
        """
        if tuple(channels) != tuple(self.channels):
            raise ValueError(
                f"explain was given channels {list(channels)}, but this model holds "
                f"{list(self.channels)}"
            )
        self._require_window_shape(x[None, ...])

        close = np.asarray(x, dtype="float64")[:, self._close_position]
        matrix = self.forecast_matrix()
        return {
            channel: ((matrix, close),) if channel == CLOSE_CHANNEL else ()
            for channel in self.channels
        }

    def explain(self, x: np.ndarray, channels: tuple[str, ...]) -> Attribution:
        """(L, C) -> the exact decomposition, on **both** axes.

        ``per_channel`` closes to the forecast at spec 4.4's tolerance, as every arm's
        does, and comes from the same :meth:`Attribution.from_terms` path FITS uses.

        ``per_level`` carries the band totals and ``per_lag`` the time localisation -
        *which* band and *when* within the window. That pair is what a Fourier basis
        structurally cannot give, and it is the reason this model's explanation is better
        than FITS's rather than merely different: "34% from the 8-16 day band, and it
        happened in the last three days" is the *when* spec 1.4 promised.

        ``per_level`` includes :data:`MEAN_KEY` alongside the bands. The RIN mean reaches
        the forecast without passing through any learned map, so a decomposition over bands
        alone would not sum to the forecast, and GB-66 requires that it does.
        """
        total = float(self.predict(x[None, ...])[0].sum())
        attribution = Attribution.from_terms(self.linear_terms(x, channels), total)

        close = np.asarray(x, dtype="float64")[:, self._close_position]
        per_level = {
            band: float((matrix @ close).sum())
            for band, matrix in self.level_matrices().items()
        }
        per_level[MEAN_KEY] = float((self.mean_matrix() @ close).sum())

        # (L, C): each input lag's contribution to the forecast total, in the close column
        # and zero in the channels this model does not read - the same statement
        # `linear_terms` makes, on the other axis.
        per_lag = np.zeros((self.input_len, len(self.channels)), dtype="float64")
        per_lag[:, self._close_position] = self.forecast_matrix().sum(axis=0) * close

        return Attribution(
            per_channel=attribution.per_channel,
            per_lag=per_lag,
            per_frequency=None,
            gain_phase=None,
            per_level=per_level,
            forecast_total=attribution.forecast_total,
        )

    def save(self, path: str) -> None:
        """Write the checkpoint as JSON. ``repr`` round-trips float64 exactly."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self._state()), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> WITSForecaster:
        """Rebuild from a checkpoint. Reads no configuration file: the shape a model was
        trained at is a property of the checkpoint, not of whatever config is on disk now.
        """
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"{path} is a version {state.get('version')!r} checkpoint, and this "
                f"build reads version {CHECKPOINT_VERSION}"
            )

        model = cls(
            input_len=state["input_len"],
            horizon=state["horizon"],
            channels=tuple(state["channels"]),
            family=state["family"],
            levels=state["levels"],
            boundary=state["boundary"],
            retained_bands=state["retained_bands"],
            shift_invariant=state["shift_invariant"],
        )
        model._set_weights(
            [np.asarray(block, dtype="float64") for block in state["weights"]]
        )

        fitted = state["fitted"]
        if fitted is not None:
            model.fitted = FitProvenance(
                channels=tuple(fitted["channels"]),
                symbols=tuple(fitted["symbols"]),
                source=fitted["source"],
                fitted_start=pd.Timestamp(fitted["fitted_start"]),
                fitted_end=pd.Timestamp(fitted["fitted_end"]),
                n_windows=fitted["n_windows"],
                input_len=fitted["input_len"],
                horizon=fitted["horizon"],
            )
        return model

    # ── internals ────────────────────────────────────────────────────────────

    @property
    def _close_position(self) -> int:
        return self.channels.index(CLOSE_CHANNEL)

    @property
    def _analysis(self) -> np.ndarray:
        return analysis_matrix(
            self.input_len, self.family, self.levels, self.boundary, self.retained_bands
        )

    @property
    def _synthesis(self) -> np.ndarray:
        return synthesis_matrix(
            self.input_len,
            self.horizon,
            self.family,
            self.levels,
            self.boundary,
            self.retained_bands,
        )

    def _set_weights(self, weights: list[np.ndarray]) -> None:
        """The one place the weights change, so the cached matrices cannot outlive them."""
        expected = list(zip(self.in_lengths, self.out_lengths, strict=True))
        if len(weights) != len(expected):
            raise ValueError(
                f"expected {len(expected)} band maps for {self.retained_bands} retained "
                f"bands, got {len(weights)}"
            )
        for band, weight, shape in zip(self.bands, weights, expected, strict=True):
            if weight.shape != shape:
                raise ValueError(
                    f"the map for band {band!r} must be {shape}, got {weight.shape}"
                )
        self.weights = [np.asarray(weight, dtype="float64") for weight in weights]
        self._matrix = None
        self._mean_matrix = None
        self._band_matrices = None

    def _extend(
        self, closes: np.ndarray, keep: tuple[str, ...] | None = None
    ) -> np.ndarray:
        """``(B, L)`` -> ``(B, L+H)``: the pipeline, in numpy.

        **RIN is per instance.** ``closes.mean(axis=1)`` is each window's own mean; a batch
        mean would pull later windows into earlier predictions, because a batch is ordered
        in time. Spec 4.4's property 6 exists for this exact bug.

        **RIN's effect is confined to the approximation band, and that is a fact about the
        basis rather than a choice.** db4 has 4 vanishing moments, so its detail filters
        annihilate constants: subtracting the window mean changes every ``cA`` coefficient
        and no ``cD`` coefficient at all. It stays where FITS puts it - before the
        transform - because the expansion requires the same normalisation for the
        comparison to isolate the transform, and moving it would confound the two.

        ``keep`` names the bands left live, ``None`` meaning all of them and ``()`` meaning
        none - which is how :meth:`mean_matrix` and :meth:`level_matrices` are *measured
        through this function* rather than re-derived beside it. A second copy of the
        pipeline written for the explanation is a second copy that can drift from the one
        that forecasts.
        """
        live = self.bands if keep is None else tuple(keep)
        mean = closes.mean(axis=1, keepdims=True)
        coefficients = (closes - mean) @ self._analysis.T

        mapped = np.zeros((closes.shape[0], sum(self.out_lengths)), dtype="float64")
        read = write = 0
        for band, weight, inside, outside in zip(
            self.bands, self.weights, self.in_lengths, self.out_lengths, strict=True
        ):
            if band in live:
                mapped[:, write : write + outside] = (
                    coefficients[:, read : read + inside] @ weight
                )
            read += inside
            write += outside

        return mapped @ self._synthesis.T + mean

    def _loss_tensor(self, torch, blocks, closes, targets):
        """Backcast plus forecast, spec 5's ``B+F``. The torch mirror of :meth:`_extend`."""
        # `np.array` rather than `ascontiguousarray`: the cached matrices are deliberately
        # read-only, and torch refuses to guarantee anything about a tensor sharing
        # non-writable memory. A copy of 42x120 and 124x42 floats is not worth the warning.
        analysis = torch.from_numpy(np.array(self._analysis))
        synthesis = torch.from_numpy(np.array(self._synthesis))

        mean = closes.mean(dim=1, keepdim=True)
        coefficients = (closes - mean) @ analysis.T

        pieces, read = [], 0
        for block, inside in zip(blocks, self.in_lengths, strict=True):
            pieces.append(coefficients[:, read : read + inside] @ block)
            read += inside
        extended = torch.cat(pieces, dim=1) @ synthesis.T + mean

        loss = torch.nn.functional.mse_loss(extended[:, self.input_len :], targets)
        return loss + torch.nn.functional.mse_loss(
            extended[:, : self.input_len], closes
        )

    def _tensors(self, batch: WindowBatch, torch):
        closes = batch.X.astype("float64")[:, :, self._close_position]
        return torch.from_numpy(closes), torch.from_numpy(batch.y.astype("float64"))

    def _state(self) -> dict:
        return {
            "version": CHECKPOINT_VERSION,
            "name": self.name,
            "input_len": self.input_len,
            "horizon": self.horizon,
            "channels": list(self.channels),
            "family": self.family,
            "levels": self.levels,
            "boundary": self.boundary,
            "retained_bands": self.retained_bands,
            "shift_invariant": self.shift_invariant,
            "weights": [weight.tolist() for weight in self.weights],
            "fitted": (
                None
                if self.fitted is None
                else {
                    "channels": list(self.fitted.channels),
                    "symbols": list(self.fitted.symbols),
                    "source": self.fitted.source,
                    "fitted_start": self.fitted.fitted_start.isoformat(),
                    "fitted_end": self.fitted.fitted_end.isoformat(),
                    "n_windows": self.fitted.n_windows,
                    "input_len": self.fitted.input_len,
                    "horizon": self.fitted.horizon,
                }
            ),
        }

    def _require_batch(self, batch: WindowBatch) -> None:
        if tuple(batch.channels) != tuple(self.channels):
            raise ValueError(
                f"the batch carries channels {list(batch.channels)}, but this model "
                f"holds {list(self.channels)}"
            )

    def _require_window_shape(self, X: np.ndarray) -> None:
        if X.ndim != 3:
            raise ValueError(
                f"input must be a 3-dimensional (B, L, C) array, got {X.shape}"
            )
        if X.shape[1] != self.input_len:
            raise ValueError(f"input must have {self.input_len} lags, got {X.shape[1]}")
        if X.shape[2] != len(self.channels):
            raise ValueError(
                f"input must have {len(self.channels)} channels, got {X.shape[2]}"
            )


__all__ = [
    "CLOSE_CHANNEL",
    "MEAN_KEY",
    "VALID_BOUNDARIES",
    "WITSForecaster",
    "analysis_matrix",
    "band_lengths",
    "band_names",
    "n_parameters_for",
    "synthesis_matrix",
]
