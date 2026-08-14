"""GB-3 acceptance: the frozen schemas construct, reject malformed input and expose a
runtime-checkable Forecaster protocol."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pandas as pd
import pytest

from glassbox.contracts.protocols import Forecaster
from glassbox.contracts.schemas import (
    Attribution,
    DecisionRecord,
    Forecast,
    Signal,
    WindowBatch,
)

BATCH, LAGS, CHANNELS, HORIZON = 4, 12, 3, 2


def make_batch(**overrides: Any) -> WindowBatch:
    """A valid WindowBatch, with named fields overridable to build invalid variants."""
    fields: dict[str, Any] = {
        "X": np.zeros((BATCH, LAGS, CHANNELS), dtype=np.float32),
        "y": np.zeros((BATCH, HORIZON), dtype=np.float32),
        "channels": ("close", "rsi14", "vol_z"),
        "timestamps": pd.date_range("2026-01-01", periods=BATCH, tz="UTC"),
        "symbol": "AAPL",
    }
    fields.update(overrides)
    return WindowBatch(**fields)


def make_attribution(**overrides: Any) -> Attribution:
    fields: dict[str, Any] = {
        "per_channel": {"close": 0.4, "rsi14": -0.1},
        "per_lag": None,
        "per_frequency": None,
        "gain_phase": None,
        "forecast_total": 0.3,
    }
    fields.update(overrides)
    return Attribution(**fields)


# ── Construction ─────────────────────────────────────────────────────────────


def test_valid_window_batch_constructs() -> None:
    batch = make_batch()
    assert batch.X.shape == (BATCH, LAGS, CHANNELS)
    assert batch.channels == ("close", "rsi14", "vol_z")
    assert batch.symbol == "AAPL"


def test_valid_forecast_constructs() -> None:
    forecast = Forecast(
        path=np.zeros(HORIZON, dtype=np.float32),
        symbol="MSFT",
        as_of=pd.Timestamp("2026-01-05", tz="UTC"),
    )
    assert forecast.path.shape == (HORIZON,)


def test_valid_attribution_constructs_without_per_lag() -> None:
    """GB-31 is cut, so implementations return None for the heatmap."""
    assert make_attribution().per_lag is None


def test_attribution_accepts_a_per_lag_matrix() -> None:
    """The field is retained so GB-31 can be added later without a contract change."""
    per_lag = np.zeros((LAGS, CHANNELS), dtype=np.float32)
    assert make_attribution(per_lag=per_lag).per_lag is per_lag


def test_attribution_is_keyword_only() -> None:
    """kw_only=True is load-bearing: per_lag carries a default (DECISIONS 2026-08-14)."""
    with pytest.raises(TypeError):
        Attribution({"close": 0.4}, None, None, None, 0.4)  # type: ignore[misc]


def test_decision_record_holds_the_full_decision() -> None:
    record = DecisionRecord(
        as_of=pd.Timestamp("2026-01-05", tz="UTC"),
        symbol="NVDA",
        forecast=Forecast(
            path=np.zeros(HORIZON, dtype=np.float32),
            symbol="NVDA",
            as_of=pd.Timestamp("2026-01-05", tz="UTC"),
        ),
        attribution=make_attribution(),
        signal=Signal(
            symbol="NVDA",
            action="enter_long",
            trend_strength=0.02,
            up_points=3,
            passed_threshold=True,
        ),
        order=None,
        narrative="Driven by the close channel.",
        config_hash="0" * 64,
    )
    assert record.order is None
    assert record.config_hash == "0" * 64


# ── Frozen ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("instance", "attribute"),
    [
        (make_batch(), "symbol"),
        (
            Forecast(
                path=np.zeros(HORIZON, dtype=np.float32),
                symbol="AAPL",
                as_of=pd.Timestamp("2026-01-05", tz="UTC"),
            ),
            "symbol",
        ),
        (make_attribution(), "forecast_total"),
        (
            Signal(
                symbol="AAPL",
                action="hold",
                trend_strength=0.0,
                up_points=0,
                passed_threshold=False,
            ),
            "action",
        ),
    ],
    ids=["window_batch", "forecast", "attribution", "signal"],
)
def test_schemas_are_frozen(instance: Any, attribute: str) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(instance, attribute, "mutated")


# ── Shape validation ─────────────────────────────────────────────────────────


def test_x_must_be_three_dimensional() -> None:
    with pytest.raises(ValueError, match=r"WindowBatch\.X must be a 3-dimensional"):
        make_batch(X=np.zeros((BATCH, LAGS), dtype=np.float32))


def test_y_must_be_two_dimensional() -> None:
    with pytest.raises(ValueError, match=r"WindowBatch\.y must be a 2-dimensional"):
        make_batch(y=np.zeros(BATCH, dtype=np.float32))


def test_x_must_be_float32() -> None:
    with pytest.raises(ValueError, match=r"WindowBatch\.X must be float32"):
        make_batch(X=np.zeros((BATCH, LAGS, CHANNELS), dtype=np.float64))


def test_y_must_be_float32() -> None:
    with pytest.raises(ValueError, match=r"WindowBatch\.y must be float32"):
        make_batch(y=np.zeros((BATCH, HORIZON), dtype=np.float64))


def test_targets_must_align_with_windows() -> None:
    with pytest.raises(ValueError, match=r"WindowBatch\.y must be one target row"):
        make_batch(y=np.zeros((BATCH + 1, HORIZON), dtype=np.float32))


def test_timestamps_must_align_with_windows() -> None:
    with pytest.raises(
        ValueError, match=r"WindowBatch\.timestamps must be one timestamp"
    ):
        make_batch(timestamps=pd.date_range("2026-01-01", periods=BATCH + 1, tz="UTC"))


def test_channel_names_must_align_with_x() -> None:
    with pytest.raises(ValueError, match=r"WindowBatch\.channels must be one name"):
        make_batch(channels=("close", "rsi14"))


def test_forecast_path_must_be_one_dimensional() -> None:
    with pytest.raises(ValueError, match=r"Forecast\.path must be a 1-dimensional"):
        Forecast(
            path=np.zeros((1, HORIZON), dtype=np.float32),
            symbol="AAPL",
            as_of=pd.Timestamp("2026-01-05", tz="UTC"),
        )


def test_per_lag_must_be_two_dimensional_when_present() -> None:
    with pytest.raises(
        ValueError, match=r"Attribution\.per_lag must be a 2-dimensional"
    ):
        make_attribution(per_lag=np.zeros(LAGS, dtype=np.float32))


def test_validation_message_follows_the_config_convention() -> None:
    """Same "{field} must be {requirement}, got {value!r}" shape as config/loader.py."""
    with pytest.raises(ValueError) as excinfo:
        make_batch(X=np.zeros((BATCH, LAGS, CHANNELS), dtype=np.float64))
    assert str(excinfo.value) == "WindowBatch.X must be float32, got dtype('float64')"


# ── The Forecaster protocol ──────────────────────────────────────────────────


class _CompleteForecaster:
    """A structurally complete implementation. Behaviour is irrelevant here."""

    def __init__(self) -> None:
        self.name = "complete"
        self.input_len = LAGS
        self.horizon = HORIZON

    def fit(self, batch: WindowBatch, val: WindowBatch | None = None) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((X.shape[0], self.horizon), dtype=np.float32)

    def explain(self, x: np.ndarray, channels: tuple[str, ...]) -> Attribution:
        return make_attribution()

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> _CompleteForecaster:
        return cls()


class _IncompleteForecaster:
    """Missing explain, save and load."""

    def __init__(self) -> None:
        self.name = "incomplete"
        self.input_len = LAGS
        self.horizon = HORIZON

    def fit(self, batch: WindowBatch, val: WindowBatch | None = None) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((X.shape[0], self.horizon), dtype=np.float32)


def test_protocol_is_runtime_checkable() -> None:
    assert isinstance(_CompleteForecaster(), Forecaster)


def test_incomplete_implementation_is_rejected() -> None:
    assert not isinstance(_IncompleteForecaster(), Forecaster)
