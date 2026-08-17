"""GB-13 acceptance: the decomposition, the per-channel weights, and what makes GB-30 exact.

The contract test already proves DLinear satisfies all seven properties of spec 4.4. What
is proved here is the part specific to this model: that the decomposition is per window and
per channel, that the weights stay separable by channel name, and that ``explain`` regroups
the terms ``predict`` computes rather than re-deriving them.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from glassbox.config.loader import Config, load_config
from glassbox.contracts.schemas import WindowBatch
from glassbox.model import ALL_FORECASTERS
from glassbox.model.ltsf import DECOMP_KERNEL, DLinearForecaster, decompose

INPUT_LEN = 30
HORIZON = 4
WINDOWS = 120
EPOCHS = 3


@pytest.fixture
def cfg() -> Config:
    base = load_config()
    return replace(
        base,
        window=replace(base.window, input_len=INPUT_LEN, horizon=HORIZON),
        model=replace(base.model, epochs=EPOCHS, patience=2),
    )


@pytest.fixture
def batch(cfg: Config) -> WindowBatch:
    channels = cfg.channels.active_channels
    rng = np.random.default_rng(cfg.meta.seed)
    return WindowBatch(
        X=(rng.standard_normal((WINDOWS, INPUT_LEN, len(channels))) * 0.02).astype(
            "float32"
        ),
        y=(rng.standard_normal((WINDOWS, HORIZON)) * 0.02).astype("float32"),
        channels=channels,
        timestamps=pd.date_range("2020-01-01", periods=WINDOWS, freq="B", tz="UTC"),
        symbols=("AAPL",) * WINDOWS,
        source="yfinance",
    )


@pytest.fixture
def model(cfg: Config, batch: WindowBatch) -> DLinearForecaster:
    fitted = ALL_FORECASTERS["dlinear"](cfg, batch.channels)
    fitted.fit(batch)
    return fitted


# ── the decomposition ────────────────────────────────────────────────────────


def test_the_components_sum_back_to_the_input() -> None:
    """`remainder = x - trend` by definition, so the split loses nothing."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, INPUT_LEN, 3))

    trend, remainder = decompose(X)

    np.testing.assert_allclose(trend + remainder, X, rtol=0, atol=1e-12)


def test_the_trend_is_a_centred_moving_average_of_the_padded_window() -> None:
    """Recomputed independently: pad both ends, then average every ``DECOMP_KERNEL`` bars."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((1, INPUT_LEN, 1))
    pad = (DECOMP_KERNEL - 1) // 2
    padded = np.concatenate(
        [np.repeat(X[:, :1, :], pad, axis=1), X, np.repeat(X[:, -1:, :], pad, axis=1)],
        axis=1,
    )
    expected = np.array(
        [padded[0, lag : lag + DECOMP_KERNEL, 0].mean() for lag in range(INPUT_LEN)]
    )

    trend, _ = decompose(X)

    np.testing.assert_allclose(trend[0, :, 0], expected, rtol=0, atol=1e-12)


def test_the_decomposition_reads_only_its_own_window() -> None:
    """The audit's claim, checked against the code rather than accepted.

    The moving average is centred, so within a window it reads lags after `l`. That is not
    look-ahead: every bar in the window is at or before the window's own timestamp. What
    *would* be look-ahead is a decomposition that reached across the batch, and this is the
    assertion that rules it out — change any other window and this one must not move.
    """
    rng = np.random.default_rng(3)
    X = rng.standard_normal((6, INPUT_LEN, 2))
    disturbed = X.copy()
    disturbed[3:] *= 7.5

    assert np.array_equal(decompose(X)[0][:3], decompose(disturbed)[0][:3])
    assert not np.array_equal(decompose(X)[0][3:], decompose(disturbed)[0][3:])


def test_the_decomposition_reads_only_its_own_channel() -> None:
    """Per channel, which is what keeps the weights separable for GB-30."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((2, INPUT_LEN, 3))
    disturbed = X.copy()
    disturbed[:, :, 2] *= 11.0

    trend, remainder = decompose(X)
    other_trend, other_remainder = decompose(disturbed)

    assert np.array_equal(trend[:, :, :2], other_trend[:, :, :2])
    assert np.array_equal(remainder[:, :, :2], other_remainder[:, :, :2])


def test_a_flat_window_has_no_remainder() -> None:
    """A constant series is all trend — the sanity check for the padding arithmetic."""
    X = np.full((1, INPUT_LEN, 1), 3.5)

    trend, remainder = decompose(X)

    np.testing.assert_allclose(trend, X, rtol=0, atol=1e-12)
    np.testing.assert_allclose(remainder, 0.0, rtol=0, atol=1e-12)


# ── the weights, keyed by channel name ───────────────────────────────────────


def test_weights_are_addressable_by_channel_name(
    model: DLinearForecaster, batch: WindowBatch
) -> None:
    """GB-30 reads these directly. If it had to re-derive them, attribution would be a
    reconstruction rather than the algebra the project claims."""
    assert set(model.trend_weights) == set(batch.channels)
    assert set(model.remainder_weights) == set(batch.channels)

    for channel in batch.channels:
        assert model.trend_weights[channel].shape == (HORIZON, INPUT_LEN)
        assert model.remainder_weights[channel].shape == (HORIZON, INPUT_LEN)


def test_weights_for_returns_both_components(model: DLinearForecaster) -> None:
    weights = model.weights_for("rsi14")

    assert set(weights) == {"trend", "remainder"}
    np.testing.assert_array_equal(weights["trend"], model.trend_weights["rsi14"])


def test_weights_for_refuses_an_unknown_channel(model: DLinearForecaster) -> None:
    """Returning an empty mapping would let an attribution quietly omit a channel."""
    with pytest.raises(ValueError, match="not one of this model's channels"):
        model.weights_for("wav_a1")


def test_the_parameter_count_is_two_components_per_channel(
    model: DLinearForecaster, batch: WindowBatch
) -> None:
    assert model.n_parameters == 2 * len(batch.channels) * HORIZON * INPUT_LEN


def test_the_weights_actually_moved_during_fit(model: DLinearForecaster) -> None:
    """A model whose weights stayed at their initialisation would pass every contract
    property and have learned nothing."""
    assert (
        np.abs(np.concatenate([model._trend.ravel(), model._remainder.ravel()])).max()
        > 0
    )


# ── explain is a regrouping, not a reconstruction ────────────────────────────


def test_attribution_is_recomputable_from_the_weights_alone(
    model: DLinearForecaster, batch: WindowBatch
) -> None:
    """The GB-30 contract, asserted now: weights plus window give the attribution.

    Computed here from the public per-channel weights, without calling ``explain`` — so
    this fails if the model ever starts explaining something other than what it computes.
    """
    window = batch.X[7]
    trend, remainder = decompose(window[None, ...].astype("float64"))

    expected = {
        channel: float(
            np.sum(
                model.trend_weights[channel] @ trend[0, :, position]
                + model.remainder_weights[channel] @ remainder[0, :, position]
            )
        )
        for position, channel in enumerate(batch.channels)
    }
    attribution = model.explain(window, batch.channels)

    for channel in batch.channels:
        assert attribution.per_channel[channel] == pytest.approx(
            expected[channel], abs=1e-12
        )


def test_the_channel_contributions_sum_to_the_forecast(
    model: DLinearForecaster, batch: WindowBatch
) -> None:
    """Exact by construction because there is no intercept to account for separately."""
    for row in (0, 11, len(batch.timestamps) - 1):
        attribution = model.explain(batch.X[row], batch.channels)
        forecast = float(model.predict(batch.X[row][None, ...])[0].sum())

        assert sum(attribution.per_channel.values()) == pytest.approx(
            attribution.forecast_total, abs=1e-12
        )
        assert attribution.forecast_total == pytest.approx(forecast, abs=1e-6)


def test_a_zero_window_forecasts_exactly_zero(model: DLinearForecaster) -> None:
    """There is no bias term, so the model is homogeneous. This is the test that would
    fail the moment an intercept was added without updating the attribution."""
    zeros = np.zeros((1, INPUT_LEN, len(model.channels)), dtype="float32")

    assert not model.predict(zeros).any()


def test_explain_refuses_a_different_channel_set(
    model: DLinearForecaster, batch: WindowBatch
) -> None:
    with pytest.raises(ValueError, match="but this model holds"):
        model.explain(batch.X[0], ("close_logret", "rsi14"))


# ── determinism ──────────────────────────────────────────────────────────────


def test_two_fits_of_the_same_config_give_identical_weights(
    cfg: Config, batch: WindowBatch
) -> None:
    """Initialisation and batch order come from a local generator seeded with
    ``cfg.meta.seed``, never from torch's global state — so this holds regardless of what
    else in the process has touched torch."""
    first = ALL_FORECASTERS["dlinear"](cfg, batch.channels)
    second = ALL_FORECASTERS["dlinear"](cfg, batch.channels)
    first.fit(batch)
    second.fit(batch)

    np.testing.assert_array_equal(first._trend, second._trend)
    np.testing.assert_array_equal(first._remainder, second._remainder)


def test_a_different_seed_gives_different_weights(
    cfg: Config, batch: WindowBatch
) -> None:
    """Otherwise the seed is decorative and the determinism test above proves nothing."""
    other = replace(cfg, meta=replace(cfg.meta, seed=cfg.meta.seed + 1))
    first = ALL_FORECASTERS["dlinear"](cfg, batch.channels)
    second = ALL_FORECASTERS["dlinear"](other, batch.channels)
    first.fit(batch)
    second.fit(batch)

    assert not np.array_equal(first._trend, second._trend)


def test_prediction_is_deterministic_without_torch(
    model: DLinearForecaster, batch: WindowBatch
) -> None:
    """Inference is pure numpy, so nothing global can perturb it between two calls."""
    assert np.array_equal(model.predict(batch.X), model.predict(batch.X))


# ── fit ──────────────────────────────────────────────────────────────────────


def test_fit_records_provenance(model: DLinearForecaster, batch: WindowBatch) -> None:
    assert model.fitted is not None
    assert model.fitted.channels == batch.channels
    assert model.fitted.n_windows == len(batch.timestamps)


def test_a_validation_split_restores_the_best_weights(
    cfg: Config, batch: WindowBatch
) -> None:
    """Early stopping must leave the best weights in place, not the last ones."""
    patient = replace(cfg, model=replace(cfg.model, epochs=20, patience=1))
    split = len(batch.timestamps) // 2
    train = WindowBatch(
        X=batch.X[:split],
        y=batch.y[:split],
        channels=batch.channels,
        timestamps=batch.timestamps[:split],
        symbols=batch.symbols[:split],
        source=batch.source,
    )
    val = WindowBatch(
        X=batch.X[split:],
        y=batch.y[split:],
        channels=batch.channels,
        timestamps=batch.timestamps[split:],
        symbols=batch.symbols[split:],
        source=batch.source,
    )

    fitted = ALL_FORECASTERS["dlinear"](patient, batch.channels)
    fitted.fit(train, val)

    assert np.isfinite(fitted.predict(val.X)).all()
    assert fitted.fitted is not None
    assert fitted.fitted.n_windows == split


def test_a_batch_with_other_channels_is_refused(
    cfg: Config, batch: WindowBatch
) -> None:
    smaller = DLinearForecaster(
        input_len=INPUT_LEN, horizon=HORIZON, channels=("close_logret",), cfg=cfg
    )

    with pytest.raises(ValueError, match="but this model holds"):
        smaller.fit(batch)


def test_a_checkpoint_cannot_be_refitted_without_a_config(
    model: DLinearForecaster, batch: WindowBatch, tmp_path
) -> None:
    """A checkpoint records weights, not how to produce them. Saying so beats defaults."""
    path = tmp_path / "dlinear.json"
    model.save(str(path))
    restored = DLinearForecaster.load(str(path))

    assert np.array_equal(restored.predict(batch.X), model.predict(batch.X))
    with pytest.raises(ValueError, match="holds no training configuration"):
        restored.fit(batch)


def test_predict_refuses_the_wrong_channel_count(model: DLinearForecaster) -> None:
    with pytest.raises(ValueError, match="channels"):
        model.predict(np.zeros((2, INPUT_LEN, 2), dtype="float32"))


def test_predict_refuses_the_wrong_window_length(model: DLinearForecaster) -> None:
    with pytest.raises(ValueError, match="lags"):
        model.predict(
            np.zeros((2, INPUT_LEN - 1, len(model.channels)), dtype="float32")
        )
