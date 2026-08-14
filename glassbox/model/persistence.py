"""L3: `PersistenceForecaster`, the zero-return baseline.

Every result in the project is reported as a delta against this model.

Implemented in GB-11.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from glassbox.contracts.schemas import Attribution, FitProvenance, WindowBatch

CHECKPOINT_VERSION = 1


class PersistenceForecaster:
    """Tomorrow equals today. Predicts zero log return.

    Every result in this project is reported as a delta against this model.
    Without it there is no way to tell whether a model learned anything or
    merely echoed the last price.

    A zero log-return path is the exact statement of "the price does not move", since
    ``ln(C_t / C_t) == 0``. Predicting zero is therefore not a placeholder — it is the
    random-walk null hypothesis this project is measured against, and on daily equity
    data it is a genuinely hard baseline to beat.
    """

    def __init__(self, input_len: int, horizon: int) -> None:
        self.name = "persistence"
        self.input_len = input_len
        self.horizon = horizon
        self.fitted: FitProvenance | None = None

    def fit(self, batch: WindowBatch, val: WindowBatch | None = None) -> None:
        """Learn nothing, but record what was offered.

        There are no parameters to estimate. The provenance is still recorded, because
        GB-25 audits every checkpoint the same way and an exemption for the baseline
        would be an exemption in exactly the arm every other arm is compared against.
        """
        del val  # no early stopping without parameters to stop
        self.fitted = FitProvenance.from_batch(batch, self.input_len, self.horizon)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(B, L, C) float32 -> (B, H) float32 of zeros."""
        self._require_window_shape(X)
        return np.zeros((X.shape[0], self.horizon), dtype=np.float32)

    def explain(self, x: np.ndarray, channels: tuple[str, ...]) -> Attribution:
        """(L, C) -> an Attribution in which every channel contributes exactly zero.

        Not an empty attribution: every active channel appears, with 0.0. A downstream
        renderer that iterates channels then shows the baseline's honest answer — "no
        channel drove this, because nothing was predicted" — rather than an empty panel
        indistinguishable from a bug.
        """
        self._require_window_shape(x[None, ...])
        return Attribution(
            per_channel=dict.fromkeys(channels, 0.0),
            per_lag=None,
            per_frequency=None,
            gain_phase=None,
            forecast_total=0.0,
        )

    def save(self, path: str) -> None:
        """Write the checkpoint as JSON: there is no weight tensor to serialise."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self._state(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> PersistenceForecaster:
        """Rebuild from a checkpoint, provenance included."""
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"{path} is a version {state.get('version')!r} checkpoint, and this "
                f"build reads version {CHECKPOINT_VERSION}"
            )

        model = cls(input_len=state["input_len"], horizon=state["horizon"])
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

    def _state(self) -> dict:
        return {
            "version": CHECKPOINT_VERSION,
            "name": self.name,
            "input_len": self.input_len,
            "horizon": self.horizon,
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

    def _require_window_shape(self, X: np.ndarray) -> None:
        """Refuse an input the model was not configured for.

        Returning zeros of the right shape for any input would make this class pass a
        contract test that a real forecaster would fail, which is the opposite of what a
        baseline is for.
        """
        if X.ndim != 3:
            raise ValueError(
                f"input must be a 3-dimensional (B, L, C) array, got {X.shape}"
            )
        if X.shape[1] != self.input_len:
            raise ValueError(f"input must have {self.input_len} lags, got {X.shape[1]}")


__all__ = ["PersistenceForecaster"]
