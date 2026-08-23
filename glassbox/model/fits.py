"""L3: `FITSForecaster` - RIN, rFFT, low-pass filter, complex linear layer, irFFT
(spec 6.2).

One complex weight matrix learns an amplitude gain and a phase shift per retained
frequency, which is exactly what complex multiplication means. That single matrix is the
entire model.

**Univariate by design (spec 6.4).** FITS consumes ``close_logret`` and nothing else, so
the ``FITS x C2_hybrid`` cell of the study grid is deliberately empty: feeding pre-filtered
wavelet bands to a model whose first act is to filter frequencies is redundant.

**But attribution names every active channel.** A model that attributed only what it reads
would render as four missing panels beside DLinear's five, and would quietly change what
"exact" means between arms of the same study (GB-33). The channels FITS does not consume
contribute **no terms**, exactly as :class:`PersistenceForecaster`'s do, and the ``0.0``
falls out of :meth:`Attribution.from_terms`'s arithmetic rather than being written here.

**The whole pipeline is linear in the input window**, mean-subtraction and its inverse
included, so ``predict(x) == M @ x_close`` for an exact ``(H, L)`` matrix. :meth:`forecast_matrix`
builds ``M`` by pushing the ``L`` basis vectors through the **actual forward pass**, so the
matrix is a measurement of the implementation rather than a second derivation of it that
could drift from it. Attribution is then that matrix applied to the window, which is why
this model can hold spec §4.4's exactness properties without a perturbation library.

**Torch fits, numpy predicts** - DLinear's precedent, and for its reason: inference holds no
global torch state to be perturbed by, so contract property 2 is true by construction and
properties 5 and 6 mean something.

**The amplitude trap is not optional** (spec 6.3). ``irfft`` normalises by the *output*
length, so extending from ``L`` to ``L+H`` shrinks every sample by ``L/(L+H)`` unless it is
multiplied back by ``(L+H)/L``. See :func:`extend_spectrum` and
``tests/model/test_fits_amplitude.py``.

Implemented in GB-41. The amplitude scaling and its sinusoid test are GB-42.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from glassbox.config.loader import Config
from glassbox.contracts.schemas import Attribution, FitProvenance, WindowBatch
from glassbox.model.history import EpochLoss

CHECKPOINT_VERSION = 1

# The one channel FITS reads. Spec 6.4: the spectral pipeline is univariate by design.
CLOSE_CHANNEL = "close_logret"


def amplitude_scale(input_len: int, horizon: int) -> float:
    """``(L + H) / L`` — spec 6.3's correction, in one place so it cannot be half-applied."""
    return (input_len + horizon) / input_len


# ── the geometry the cutoff decides, as functions ────────────────────────────
#
# The properties below delegate to these. GB-50 sweeps the cutoff and has to report the
# retained bin count and the dead-row share **per cutoff**, and computing either from the
# cutoff a second time is the two-places defect this project has now seen five instances
# of. One derivation, two readers.


def cof_for(input_len: int, cutoff_period_days: int) -> int:
    """``L // cutoff_period_days`` — how many low-frequency bins survive the filter."""
    return input_len // cutoff_period_days


def out_bins_for(input_len: int, horizon: int, cutoff_period_days: int) -> int:
    """``ceil(η · COF)`` — the bins the complex layer emits.

    **Integer arithmetic, and the reason is a defect GB-50 exposed.** Computing this as
    ``math.ceil(amplitude_scale(L, H) * COF)`` is correct on paper and wrong in floating
    point exactly where ``η · COF`` is a whole number: at ``L=120, H=4, COF=60`` the
    product evaluates to ``62.00000000000001`` and the ceiling is **63**, one bin more
    than the formula asks for and 120 allocated reals that the architecture does not
    describe. The failure is silent - nothing downstream refuses an extra output bin - and
    it fires **only on the perfectly aligned case**, which is the one the formula exists
    to handle cleanly.

    The deployed cutoff of 5 gives 24.8 and is unaffected either way, so this changes no
    existing checkpoint and no published number.
    """
    cof = cof_for(input_len, cutoff_period_days)
    return -(-(input_len + horizon) * cof // input_len)


def dead_row_fraction(input_len: int, horizon: int, cutoff_period_days: int) -> float:
    """The share of allocated reals that row 0 holds and can never move.

    RIN subtracts each window's mean and bin 0 of the rFFT **is** that mean, so row 0 of
    the weight matrix multiplies zero on every forward pass. It is ``out_bins`` complex
    weights out of ``COF · out_bins``, which reduces to ``1 / COF`` — written as the ratio
    rather than as the reciprocal because the numerator is the thing being counted.

    **The share grows as the cutoff rises**, which is the opposite of the intuition that a
    smaller model wastes less: at ``cutoff_period_days = 2`` the dead row is 1.7% of the
    layer and at 20 it is 16.7%, because the same one row is a larger part of fewer.
    """
    cof = cof_for(input_len, cutoff_period_days)
    out = out_bins_for(input_len, horizon, cutoff_period_days)
    return (out * 2) / (cof * out * 2)


def aligned_bins(input_len: int, horizon: int, cutoff_period_days: int) -> int:
    """Retained bins whose frequency lands exactly on an output bin.

    ``η · k`` is an integer only for those, and every other retained frequency must be
    interpolated on every window — which is what the learned layer is *for*. See the note
    on :func:`extend_spectrum`.
    """
    eta = amplitude_scale(input_len, horizon)
    return sum(
        1
        for k in range(cof_for(input_len, cutoff_period_days))
        if abs(eta * k - round(eta * k)) < 1e-9
    )


def mean_misalignment(input_len: int, horizon: int, cutoff_period_days: int) -> float:
    """Mean ``|η·k − round(η·k)|`` over the retained bins: 0 aligned, 0.5 worst.

    Gain tracks this at Spearman −0.9427 on a trained model (GB-48), so it is the geometric
    quantity the frequency response is mostly reporting.
    """
    eta = amplitude_scale(input_len, horizon)
    cof = cof_for(input_len, cutoff_period_days)
    return float(np.mean([abs(eta * k - round(eta * k)) for k in range(cof)]))


def extend_spectrum(x: np.ndarray, horizon: int) -> np.ndarray:
    """Extend ``(B, L)`` to ``(B, L+H)`` through the frequency domain, amplitude-correct.

    **This is not a stage of the forward pass.** It is the amplitude machinery with the
    learned layer removed, which is what allows spec 6.3's test to exist *before* any
    training runs — GB-42 fixed this interface for exactly that.

    **It maps bins rather than padding them, and the difference is not cosmetic.** Bin
    ``k`` of a length-``L`` transform is the frequency ``k/L``; the same bin of a
    length-``L+H`` transform is ``k/(L+H)``. Leaving a coefficient where it was therefore
    *stretches* the signal in time. Measured on a 3.0-amplitude 12-day sinusoid over
    ``L=120``: padding in place leaves a backcast error of **4.97**, which is the amplitude
    itself. Mapping bin ``k`` to ``round(k·η)`` leaves **2.2e-14** — when ``η·k`` is an
    integer.

    **When it is not an integer, no parameter-free extension can reconstruct**, and that is
    the whole reason :class:`FITSForecaster` has a learned complex layer: at the configured
    ``H=4``, ``η·k`` for a 12-day cycle is 10.333, and the layer is what interpolates
    between bins. So this function is exact on a grid where the input's frequencies land on
    output bins, and approximate otherwise — which is the right tool for isolating the
    amplitude scaling and the wrong one for forecasting.

    Args:
        x: ``(B, L)`` real. Also accepts ``(L,)``.
        horizon: ``H``, the number of samples to extend by.

    Returns:
        ``(B, L+H)`` real, scaled by ``(L+H)/L``.
    """
    window = np.asarray(x, dtype="float64")
    flat = window.ndim == 1
    if flat:
        window = window[None, :]
    if window.ndim != 2:
        raise ValueError(f"extend_spectrum takes (B, L) or (L,), got {window.shape}")

    input_len = window.shape[1]
    total = input_len + horizon
    eta = amplitude_scale(input_len, horizon)

    spectrum = np.fft.rfft(window, n=input_len, axis=1)
    extended = np.zeros((window.shape[0], total // 2 + 1), dtype="complex128")
    for bin_in in range(spectrum.shape[1]):
        bin_out = round(bin_in * eta)
        if bin_out < extended.shape[1]:
            extended[:, bin_out] = spectrum[:, bin_in]

    out = np.fft.irfft(extended, n=total, axis=1) * eta
    return out[0] if flat else out


class FITSForecaster:
    """Frequency Interpolation Time Series forecasting: one complex linear layer.

    Attributes:
        weight: ``(COF, out_bins)`` complex. The whole model. ``abs(weight)`` is the
            amplitude gain per retained frequency and ``angle(weight)`` the phase shift —
            spec 6.5's frequency-response plot reads these directly.
    """

    def __init__(
        self,
        input_len: int,
        horizon: int,
        channels: tuple[str, ...],
        cutoff_period_days: int,
        cfg: Config | None = None,
    ) -> None:
        """``cfg`` supplies the training hyperparameters and is required only by ``fit``.

        ``cutoff_period_days`` is passed rather than read from ``cfg`` because a model
        rebuilt by :meth:`load` has no configuration and the cutoff decides its *shape* —
        a checkpoint that could not say how many bins it kept would be unloadable.
        """
        if CLOSE_CHANNEL not in channels:
            raise ValueError(
                f"FITS forecasts {CLOSE_CHANNEL!r} and this channel set does not contain "
                f"it: {list(channels)}. The spectral pipeline is univariate by design "
                "(spec 6.4); it reads one series and attributes to all of them"
            )
        if cutoff_period_days <= 0:
            raise ValueError(
                f"cutoff_period_days must be positive, got {cutoff_period_days}"
            )

        self.name = "fits"
        self.input_len = input_len
        self.horizon = horizon
        self.channels = tuple(channels)
        self.cutoff_period_days = cutoff_period_days
        self.fitted: FitProvenance | None = None
        self._cfg = cfg

        # How the last fit went. Not model state and not in the checkpoint - a model
        # rebuilt by `load` reports an empty history, which is the truth: it did not train.
        self.history: tuple[EpochLoss, ...] = ()
        self.best_epoch: int | None = None
        self.stopped_early = False

        self.weight = np.zeros((self.cof, self.out_bins), dtype="complex128")
        self._matrix: np.ndarray | None = None
        self._mean_matrix: np.ndarray | None = None
        self._bin_matrices: dict[int, np.ndarray] | None = None

    # ── the shape the cutoff decides ─────────────────────────────────────────

    @property
    def cof(self) -> int:
        """Cut-off frequency: how many low-frequency bins survive the filter.

        ``L // cutoff_period_days``. Everything faster than the cutoff period is discarded
        before the model sees it, which is the one hyperparameter FITS has.
        """
        return cof_for(self.input_len, self.cutoff_period_days)

    @property
    def out_bins(self) -> int:
        """``ceil(eta * COF)`` — the bins the complex layer emits."""
        return out_bins_for(self.input_len, self.horizon, self.cutoff_period_days)

    @property
    def n_parameters(self) -> int:
        """Learnable reals: ``COF * out_bins * 2``, a real and an imaginary part each.

        Counted in reals rather than complex numbers because that is what a parameter count
        means everywhere else in the study, and comparing 600 complex weights against
        DLinear's 4,800 reals would flatter FITS by a factor of two.

        **This is what the tensor holds, and it is 50 more than the model can use.** RIN
        subtracts each window's mean and the rFFT's bin 0 *is* that mean, so row 0 of the
        weight matrix multiplies zero on every forward pass: it takes no gradient, stays at
        its initial value, and cannot move a forecast. ``out_bins`` complex weights - 50
        reals at the configured geometry, **4.17%** - are allocated and dead, so the
        effective count is **1,150**. The row is kept because §6.2's low-pass keeps "the
        first ``COF`` bins" and the source paper's architecture carries the same dead row;
        what would be wrong is quoting the allocated number as capacity.
        ``test_fits.py::test_the_dc_row_is_allocated_and_cannot_learn`` pins both figures.
        """
        return self.weight.size * 2

    # ── the Forecaster protocol ──────────────────────────────────────────────

    def fit(self, batch: WindowBatch, val: WindowBatch | None = None) -> None:
        """Fit the complex layer by gradient descent, supervised on **backcast and forecast**.

        Spec 5 fixes ``fits.supervision`` at ``B+F`` and the toggle is out of scope: the
        loss is the forecast's error plus the reconstruction's. The backcast term is what
        teaches the layer to place an input frequency on the right output bin — without it
        the layer has no reason to preserve the signal it is interpolating, and the
        frequency response the report plots would mean nothing.

        Hyperparameters come from ``cfg.model``, exactly as DLinear's do. The weight is
        held as two real tensors and combined with ``torch.complex`` in the forward pass:
        gradients flow through, and the parameter count is then plainly the two of them.
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
        real = torch.zeros(self.weight.shape, dtype=torch.float64).requires_grad_(True)
        imag = torch.zeros(self.weight.shape, dtype=torch.float64).requires_grad_(True)
        optimiser = torch.optim.Adam([real, imag], lr=plan.lr)

        validation = None if val is None else self._tensors(val, torch)
        best_loss, best_state, waited = math.inf, None, 0
        curve: list[EpochLoss] = []
        self.best_epoch, self.stopped_early = None, False

        windows = targets.shape[0]
        size = windows if plan.batch_size is None else plan.batch_size
        # See DLinear: permuting a single batch changes only the order floats are summed
        # in, which is enough to lose bit-identical weights across seeds for nothing.
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
                self._loss_tensor(
                    torch, real, imag, inputs[rows], targets[rows]
                ).backward()
                optimiser.step()

            with torch.no_grad():
                score = (
                    None
                    if validation is None
                    else float(self._loss_tensor(torch, real, imag, *validation))
                )
                curve.append(
                    EpochLoss(
                        epoch=epoch,
                        train_loss=float(
                            self._loss_tensor(torch, real, imag, inputs, targets)
                        ),
                        val_loss=score,
                    )
                )

            if score is None:
                continue
            if score < best_loss:
                best_loss, waited = score, 0
                self.best_epoch = epoch
                best_state = (real.detach().clone(), imag.detach().clone())
            else:
                waited += 1
                if waited >= plan.patience:
                    self.stopped_early = True
                    break

        final = best_state if best_state is not None else (real, imag)
        self._set_weight(
            final[0].detach().numpy() + 1j * final[1].detach().numpy(),
        )
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
        evidence that the frequency response means something, and one that does not says
        the low-pass filter threw away the signal rather than the noise.
        """
        self._require_window_shape(X)
        closes = np.asarray(X, dtype="float64")[:, :, self._close_position]
        return self._extend(closes)[:, : self.input_len].astype("float32")

    def forecast_matrix(self) -> np.ndarray:
        """The exact ``(H, L)`` matrix with ``predict(x) == M @ x_close``.

        **Measured from the forward pass, not re-derived from it.** Every stage is linear —
        subtracting the window mean and adding it back included — and ``f(0) == 0``, so the
        map is a matrix whose columns are ``f(e_i)``. Pushing the ``L`` basis windows
        through :meth:`_extend` gives those columns from the same code inference uses, so
        the attribution cannot drift from the forecast the way a hand-written derivation
        would. It costs ``L`` forward passes of a 600-weight model, once.
        """
        if self._matrix is None:
            basis = np.eye(self.input_len, dtype="float64")
            self._matrix = self._extend(basis)[:, self.input_len :].T.copy()
        return self._matrix

    def mean_matrix(self) -> np.ndarray:
        """The ``(H, L)`` map the **RIN mean** carries, with every learned bin silenced.

        RIN subtracts the window mean and adds it back, so the mean reaches the forecast
        without passing through the complex layer at all. It is part of the forecast and
        belongs to no frequency the layer learned, so a decomposition that omitted it could
        not close - spec 4.4's tolerance would catch it, and
        ``Attribution.from_terms``'s error message names this exact case.

        Measured by running the forward pass with a zero mask, so it is whatever the
        implementation does rather than what the docstring above says it does.
        """
        if self._mean_matrix is None:
            basis = np.eye(self.input_len, dtype="float64")
            silent = np.zeros(self.cof, dtype="float64")
            self._mean_matrix = self._extend(basis, keep=silent)[
                :, self.input_len :
            ].T.copy()
        return self._mean_matrix

    def frequency_matrices(self) -> dict[int, np.ndarray]:
        """``{input bin k: (H, L)}`` - what each retained frequency contributes.

        Together with :meth:`mean_matrix` these sum **exactly** to
        :meth:`forecast_matrix`, which is the property GB-45's decomposition rests on and
        the one its acceptance test asserts. Each is measured the same way the total is:
        push the ``L`` basis windows through the real forward pass with one bin unmasked,
        and subtract the mean path that every masked pass still carries.

        **Bin 0 is identically zero and is still reported.** RIN has already removed the
        window mean, so the rFFT's bin 0 - which *is* that mean - is zero on every window,
        and the weights in row 0 multiply nothing (GB-44). Returning it as a measured zero
        rather than dropping it is the same choice as attributing 0.0 to a channel FITS
        does not read: an absent entry and a zero entry say different things.
        """
        if self._bin_matrices is None:
            basis = np.eye(self.input_len, dtype="float64")
            background = self.mean_matrix()
            matrices = {}
            for index in range(self.cof):
                mask = np.zeros(self.cof, dtype="float64")
                mask[index] = 1.0
                alone = self._extend(basis, keep=mask)[:, self.input_len :].T
                matrices[index] = (alone - background).copy()
            self._bin_matrices = matrices
        return self._bin_matrices

    def linear_terms(
        self, x: np.ndarray, channels: tuple[str, ...]
    ) -> dict[str, tuple[tuple[np.ndarray, np.ndarray], ...]]:
        """(L, C) -> ``{channel: ((W, v), ...)}``, the terms ``predict`` sums.

        One pair for ``close_logret`` and **no terms at all** for every other active
        channel — not a zero weight and not an omission. A model that does not read a
        channel has no term in it, and an empty tuple is the exact statement of that; the
        ``0.0`` then falls out of :meth:`Attribution.from_terms` rather than being written
        down here. This is :class:`PersistenceForecaster`'s pattern, and it is what keeps
        every arm of the study on one code path (GB-30, GB-33).
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
        """(L, C) -> the exact per-channel decomposition of this window's forecast.

        The total comes from :meth:`predict` rather than from the contributions' own sum,
        so ``from_terms`` compares two independently computed quantities and spec 4.4's
        properties 3 and 4 are both real checks on this object.
        """
        return Attribution.from_terms(
            self.linear_terms(x, channels),
            float(self.predict(x[None, ...])[0].sum()),
        )

    def save(self, path: str) -> None:
        """Write the checkpoint as JSON. ``repr`` round-trips float64 exactly."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self._state()), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> FITSForecaster:
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
            cutoff_period_days=state["cutoff_period_days"],
        )
        model._set_weight(
            np.asarray(state["weight_real"], dtype="float64")
            + 1j * np.asarray(state["weight_imag"], dtype="float64")
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

    def _set_weight(self, weight: np.ndarray) -> None:
        """The one place the weight changes, so the cached matrix cannot outlive it."""
        if weight.shape != (self.cof, self.out_bins):
            raise ValueError(
                f"the weight must be ({self.cof}, {self.out_bins}) for input_len "
                f"{self.input_len}, horizon {self.horizon} and cutoff "
                f"{self.cutoff_period_days}; got {weight.shape}"
            )
        self.weight = np.asarray(weight, dtype="complex128")
        self._matrix = None
        self._mean_matrix = None
        self._bin_matrices = None

    def _extend(self, closes: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
        """``(B, L)`` -> ``(B, L+H)``: spec 6.2's pipeline, in numpy.

        **RIN is per instance.** ``closes.mean(axis=1)`` is each window's own mean; a batch
        mean would pull later windows into earlier predictions, because a batch is ordered
        in time. Spec 4.4's property 6 exists for this exact bug and ``BatchNormForecaster``
        is the model written to prove the property can fail.

        ``keep`` is a ``(COF,)`` mask over the retained input bins, ``None`` meaning all of
        them - which is every caller but :meth:`frequency_matrices`. It exists so the
        per-frequency decomposition of GB-45 can be **measured through this function**
        rather than re-derived beside it, the same discipline as :meth:`forecast_matrix`:
        a second copy of the pipeline written for the explanation is a second copy that can
        drift from the one that forecasts.
        """
        total = self.input_len + self.horizon
        mean = closes.mean(axis=1, keepdims=True)
        spectrum = np.fft.rfft(closes - mean, n=self.input_len, axis=1)
        retained = spectrum[:, : self.cof]
        if keep is not None:
            retained = retained * np.asarray(keep, dtype="float64")
        mapped = retained @ self.weight

        padded = np.zeros((closes.shape[0], total // 2 + 1), dtype="complex128")
        padded[:, : self.out_bins] = mapped
        extended = np.fft.irfft(padded, n=total, axis=1)

        # Spec 6.3. Without this every amplitude is short by L/(L+H).
        return extended * amplitude_scale(self.input_len, self.horizon) + mean

    def _loss_tensor(self, torch, real, imag, closes, targets):
        """Backcast plus forecast, spec 5's ``B+F``. The torch mirror of :meth:`_extend`."""
        total = self.input_len + self.horizon
        mean = closes.mean(dim=1, keepdim=True)
        spectrum = torch.fft.rfft(closes - mean, n=self.input_len, dim=1)
        mapped = spectrum[:, : self.cof] @ torch.complex(real, imag)

        padded = torch.zeros((closes.shape[0], total // 2 + 1), dtype=torch.complex128)
        padded = padded.index_copy(
            1, torch.arange(self.out_bins), mapped[:, : self.out_bins]
        )
        extended = torch.fft.irfft(padded, n=total, dim=1)
        extended = extended * amplitude_scale(self.input_len, self.horizon) + mean

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
            "cutoff_period_days": self.cutoff_period_days,
            "weight_real": self.weight.real.tolist(),
            "weight_imag": self.weight.imag.tolist(),
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
    "FITSForecaster",
    "aligned_bins",
    "amplitude_scale",
    "cof_for",
    "dead_row_fraction",
    "extend_spectrum",
    "mean_misalignment",
    "out_bins_for",
]
