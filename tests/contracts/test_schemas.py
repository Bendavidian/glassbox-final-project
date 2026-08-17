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
    FitProvenance,
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
        "channels": ("close_logret", "rsi14", "vol_z"),
        "timestamps": pd.date_range("2026-01-01", periods=BATCH, tz="UTC"),
        "symbols": ("AAPL",) * BATCH,
        "source": "yfinance",
    }
    fields.update(overrides)
    return WindowBatch(**fields)


def make_attribution(**overrides: Any) -> Attribution:
    fields: dict[str, Any] = {
        "per_channel": {"close_logret": 0.4, "rsi14": -0.1},
        "per_lag": None,
        "per_frequency": None,
        "gain_phase": None,
        "forecast_total": 0.3,
    }
    fields.update(overrides)
    return Attribution(**fields)


def make_provenance(**overrides: Any) -> FitProvenance:
    fields: dict[str, Any] = {
        "channels": ("close_logret", "rsi14", "vol_z"),
        "symbols": ("AAPL",),
        "source": "yfinance",
        "fitted_start": pd.Timestamp("2020-01-01", tz="UTC"),
        "fitted_end": pd.Timestamp("2021-01-01", tz="UTC"),
        "n_windows": BATCH,
        "input_len": LAGS,
        "horizon": HORIZON,
    }
    fields.update(overrides)
    return FitProvenance(**fields)


# ── Construction ─────────────────────────────────────────────────────────────


def test_valid_window_batch_constructs() -> None:
    batch = make_batch()
    assert batch.X.shape == (BATCH, LAGS, CHANNELS)
    assert batch.channels == ("close_logret", "rsi14", "vol_z")
    assert batch.symbols == ("AAPL",) * BATCH


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
        Attribution({"close_logret": 0.4}, None, None, None, 0.4)  # type: ignore[misc]


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
        narrative="Driven by the close_logret channel.",
        config_hash="0" * 64,
    )
    assert record.order is None
    assert record.config_hash == "0" * 64


# ── Frozen ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("instance", "attribute"),
    [
        (make_batch(), "symbols"),
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
        make_batch(channels=("close_logret", "rsi14"))


def test_symbols_must_align_with_windows() -> None:
    """One entry per window, not one per batch — the 2026-08-17 contract change."""
    with pytest.raises(ValueError, match=r"WindowBatch\.symbols must be one symbol"):
        make_batch(symbols=("AAPL",))


# ── Pooling across the universe ──────────────────────────────────────────────
#
# Every forecaster trains across the universe, so a batch spanning five symbols is the
# ordinary case rather than the exotic one. `concat` is the only place that assembly
# happens, and every refusal below is a bug it exists to make loud.


def test_a_batch_carries_one_symbol_per_window() -> None:
    batch = make_batch(symbols=("AAPL", "AAPL", "MSFT", "MSFT"))

    assert len(batch.symbols) == batch.X.shape[0]
    assert batch.unique_symbols == ("AAPL", "MSFT")


def test_unique_symbols_is_sorted_and_deduplicated() -> None:
    """Derived, never stored, so it cannot fall out of step with ``symbols``."""
    batch = make_batch(symbols=("NVDA", "AAPL", "NVDA", "MSFT"))

    assert batch.unique_symbols == ("AAPL", "MSFT", "NVDA")


def test_concat_pools_windows_and_keeps_each_windows_symbol() -> None:
    first = make_batch(symbols=("AAPL",) * BATCH)
    second = make_batch(
        symbols=("MSFT",) * BATCH,
        timestamps=pd.date_range("2026-02-01", periods=BATCH, tz="UTC"),
    )

    pooled = WindowBatch.concat([first, second])

    assert pooled.X.shape == (2 * BATCH, LAGS, CHANNELS)
    assert pooled.symbols == ("AAPL",) * BATCH + ("MSFT",) * BATCH
    assert pooled.unique_symbols == ("AAPL", "MSFT")
    assert len(pooled.timestamps) == 2 * BATCH


def test_a_pooled_batch_has_non_monotonic_timestamps() -> None:
    """Stated rather than left to be discovered.

    Each symbol contributes the same date range, so a pooled batch revisits every date
    once per symbol. Anything reading ``timestamps[0]`` or ``timestamps[-1]`` as the
    batch's extent is wrong — which is why ``FitProvenance`` takes min and max.
    """
    first = make_batch(symbols=("AAPL",) * BATCH)
    second = make_batch(symbols=("MSFT",) * BATCH)

    pooled = WindowBatch.concat([first, second])

    assert not pooled.timestamps.is_monotonic_increasing
    assert pooled.timestamps.min() == first.timestamps[0]
    assert pooled.timestamps.max() == first.timestamps[-1]


def test_concat_refuses_batches_with_different_channels() -> None:
    """The silent one: shared weights applied to a different meaning at the same index."""
    first = make_batch()
    second = make_batch(channels=("close_logret", "vol_z", "rsi14"))

    with pytest.raises(ValueError, match="every batch to carry the channels"):
        WindowBatch.concat([first, second])


def test_concat_refuses_batches_with_different_window_shapes() -> None:
    first = make_batch()
    second = make_batch(X=np.zeros((BATCH, LAGS + 1, CHANNELS), dtype=np.float32))

    with pytest.raises(ValueError, match="windows of shape"):
        WindowBatch.concat([first, second])


def test_concat_refuses_batches_with_different_horizons() -> None:
    first = make_batch()
    second = make_batch(y=np.zeros((BATCH, HORIZON + 1), dtype=np.float32))

    with pytest.raises(ValueError, match="a horizon of"):
        WindowBatch.concat([first, second])


def test_concat_refuses_batches_from_different_sources() -> None:
    """GB-8 measured a 4-9% volume difference between vendors; pooling across them is the
    same defect as splicing within one series."""
    first = make_batch()
    second = make_batch(source="alpaca")

    with pytest.raises(ValueError, match="a window must be assembled from one source"):
        WindowBatch.concat([first, second])


def test_concat_refuses_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="empty sequence of batches"):
        WindowBatch.concat([])


def test_concat_of_one_batch_is_that_batch() -> None:
    only = make_batch()

    pooled = WindowBatch.concat([only])

    np.testing.assert_array_equal(pooled.X, only.X)
    assert pooled.symbols == only.symbols
    assert pooled.source == only.source


def test_provenance_of_a_pooled_batch_spans_every_symbol() -> None:
    """The audit's question is "what did this model see?", and the answer is all of it."""
    first = make_batch(symbols=("AAPL",) * BATCH)
    second = make_batch(
        symbols=("MSFT",) * BATCH,
        timestamps=pd.date_range("2025-06-01", periods=BATCH, tz="UTC"),
    )
    pooled = WindowBatch.concat([first, second])

    provenance = FitProvenance.from_batch(pooled, LAGS, HORIZON)

    assert provenance.symbols == ("AAPL", "MSFT")
    assert provenance.n_windows == 2 * BATCH
    # min/max, not first/last: the last window is MSFT's, which ends earlier here.
    assert provenance.fitted_start == second.timestamps[0]
    assert provenance.fitted_end == first.timestamps[-1]
    assert provenance.fitted_end != pooled.timestamps[-1]


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


# ── FitProvenance ────────────────────────────────────────────────────────────


def test_fit_provenance_is_derived_from_the_batch() -> None:
    """One derivation for every model, so GB-25 never compares apples to pears."""
    batch = make_batch()
    provenance = FitProvenance.from_batch(batch, LAGS, HORIZON)

    assert provenance.channels == batch.channels
    assert provenance.symbols == batch.unique_symbols
    assert provenance.source == batch.source
    assert provenance.fitted_start == batch.timestamps[0]
    assert provenance.fitted_end == batch.timestamps[-1]
    assert provenance.n_windows == len(batch.timestamps)
    assert (provenance.input_len, provenance.horizon) == (LAGS, HORIZON)


def test_fit_provenance_rejects_a_range_that_runs_backwards() -> None:
    with pytest.raises(ValueError, match="FitProvenance.fitted_start"):
        make_provenance(
            fitted_start=pd.Timestamp("2021-01-01", tz="UTC"),
            fitted_end=pd.Timestamp("2020-01-01", tz="UTC"),
        )


def test_fit_provenance_rejects_an_empty_window_count() -> None:
    with pytest.raises(ValueError, match="FitProvenance.n_windows"):
        make_provenance(n_windows=0)


def test_fit_provenance_rejects_a_batch_with_no_windows() -> None:
    """An empty training batch is a bug upstream; recording it as provenance hides it."""
    empty = make_batch(
        X=np.zeros((0, LAGS, CHANNELS), dtype=np.float32),
        y=np.zeros((0, HORIZON), dtype=np.float32),
        timestamps=pd.DatetimeIndex([], tz="UTC"),
        symbols=(),
    )
    with pytest.raises(ValueError, match="at least one window"):
        FitProvenance.from_batch(empty, LAGS, HORIZON)


# ── The Forecaster protocol ──────────────────────────────────────────────────


class _CompleteForecaster:
    """A structurally complete implementation. Behaviour is irrelevant here."""

    def __init__(self) -> None:
        self.name = "complete"
        self.input_len = LAGS
        self.horizon = HORIZON
        self.fitted: FitProvenance | None = None

    def fit(self, batch: WindowBatch, val: WindowBatch | None = None) -> None:
        self.fitted = FitProvenance.from_batch(batch, self.input_len, self.horizon)

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


class _ProvenanceFreeForecaster(_CompleteForecaster):
    """Complete but for ``fitted`` — a model that cannot say what it was trained on."""

    def __init__(self) -> None:
        super().__init__()
        del self.fitted


def test_protocol_is_runtime_checkable() -> None:
    assert isinstance(_CompleteForecaster(), Forecaster)


def test_a_forecaster_without_provenance_is_rejected() -> None:
    """GB-11 added ``fitted`` to the protocol, so the structural check must require it."""
    assert not isinstance(_ProvenanceFreeForecaster(), Forecaster)


def test_incomplete_implementation_is_rejected() -> None:
    assert not isinstance(_IncompleteForecaster(), Forecaster)
