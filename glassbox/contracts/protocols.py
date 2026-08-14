"""L0: the ``Forecaster`` protocol of spec 4.3 - fit, predict, explain, save, load.

Every model implements exactly this, so nothing downstream knows which model it holds.
This is the contract that makes the comparison possible: swapping Persistence for
DLinear for FITS is a config change, not a code change.

A model is not integrated until it passes tests/test_forecaster_contract.py unchanged.

Implemented in GB-3.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from glassbox.contracts.schemas import Attribution, WindowBatch


@runtime_checkable
class Forecaster(Protocol):
    """The single interface every forecaster implements.

    ``runtime_checkable`` supports ``isinstance`` checks in the parameterised contract
    test. It verifies that the members exist, not their signatures, so it is a guard
    against an incomplete implementation, never a substitute for the contract test.
    """

    name: str
    input_len: int  # L
    horizon: int  # H

    def fit(self, batch: WindowBatch, val: WindowBatch | None = None) -> None:
        """Train. MUST use only `batch` (+ `val` for early stopping).
        MUST store any normalisation statistics internally."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(B, L, C) float32 → (B, H) float32 log-return paths."""
        ...

    def explain(self, x: np.ndarray, channels: tuple[str, ...]) -> Attribution:
        """(L, C) single window → exact Attribution.
        MUST satisfy the exactness assertion in Attribution's docstring."""
        ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> Forecaster: ...


__all__ = ["Forecaster"]
