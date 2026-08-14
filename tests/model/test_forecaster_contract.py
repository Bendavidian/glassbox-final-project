"""GB-11 acceptance: the contract test of spec 4.4, parameterised over every forecaster.

**Adding a model touches one line.** Append a factory to :data:`FORECASTERS`; every
property below then runs against it. GB-13 registers DLinear and GB-41 registers FITS,
and neither task edits anything else in this file. That is why each property is a plain
function of ``(model, batch, tmp_path)`` rather than a test body: the same function is
called by the real parameterised test and by the teeth test below, so a property cannot
be strict for a broken model and lax for a real one.

**The properties have teeth before a second model exists.** Persistence satisfies every
one of them trivially — it returns zeros — so a suite that only ever ran against it would
be untested itself, and would still pass if a property's assertion were deleted. Each
property therefore also runs against a forecaster deliberately broken in exactly that one
respect, and is asserted to reject it. This is the standard GB-10 set for the causality
harness, applied to the contract.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from causality import assert_causal

from glassbox.config.loader import Config, load_config
from glassbox.contracts.protocols import Forecaster
from glassbox.contracts.schemas import Attribution, FitProvenance, WindowBatch
from glassbox.model.persistence import PersistenceForecaster

INPUT_LEN = 30
HORIZON = 4
WINDOWS = 200

# Spec 4.4 properties 3 and 4 are algebraic identities, so one window proves almost
# nothing: an implementation that is exact on the data manifold and wrong off it would
# pass. A thousand random windows is cheap and covers the space the identity claims.
EXACTNESS_SAMPLES = 1000

EXACTNESS_TOLERANCE = 1e-5


# ── the registry: the only thing GB-13 and GB-41 touch ───────────────────────


def _persistence(batch: WindowBatch, cfg: Config) -> Forecaster:
    model = PersistenceForecaster(
        input_len=cfg.window.input_len, horizon=cfg.window.horizon
    )
    model.fit(batch)
    return model


FORECASTERS = [
    pytest.param(_persistence, id="persistence"),
    # GB-13: pytest.param(_dlinear, id="dlinear")
    # GB-41: pytest.param(_fits, id="fits")
]


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg() -> Config:
    """Production config with a smaller window, so the suite stays fast."""
    base = load_config()
    return replace(
        base, window=replace(base.window, input_len=INPUT_LEN, horizon=HORIZON)
    )


@pytest.fixture
def batch(cfg: Config) -> WindowBatch:
    """A deterministic batch with the production channel set and window shape."""
    channels = cfg.channels.active_channels
    rng = np.random.default_rng(cfg.meta.seed)
    timestamps = pd.date_range("2020-01-01", periods=WINDOWS, freq="B", tz="UTC")
    return WindowBatch(
        X=rng.standard_normal((WINDOWS, INPUT_LEN, len(channels))).astype("float32"),
        y=rng.standard_normal((WINDOWS, HORIZON)).astype("float32"),
        channels=channels,
        timestamps=timestamps,
        symbol="AAPL",
        source="yfinance",
    )


@pytest.fixture
def windows(cfg: Config) -> np.ndarray:
    """``EXACTNESS_SAMPLES`` random single windows, seeded from the config."""
    rng = np.random.default_rng(cfg.meta.seed + 1)
    shape = (EXACTNESS_SAMPLES, INPUT_LEN, len(cfg.channels.active_channels))
    return rng.standard_normal(shape).astype("float32")


# ── the properties of spec 4.4 ───────────────────────────────────────────────


def check_shape_and_dtype(
    model: Forecaster, batch: WindowBatch, tmp_path: Path
) -> None:
    """1. predict returns exactly (B, H) float32."""
    del tmp_path
    prediction = model.predict(batch.X)

    assert prediction.shape == (batch.X.shape[0], model.horizon)
    assert prediction.dtype == np.float32


def check_determinism(model: Forecaster, batch: WindowBatch, tmp_path: Path) -> None:
    """2. Two calls on the same input are bit-identical.

    The seed is fixed by the fixture, so any difference is unseeded state inside the
    model — a dropout layer left in eval, or a random tie-break.
    """
    del tmp_path

    assert np.array_equal(model.predict(batch.X), model.predict(batch.X))


def check_explain_total_matches_predict(
    model: Forecaster, batch: WindowBatch, tmp_path: Path, windows: np.ndarray
) -> None:
    """3. explain(x).forecast_total == predict(x[None])[0].sum(), over many windows."""
    del tmp_path
    for x in windows:
        attribution = model.explain(x, batch.channels)
        total = float(model.predict(x[None, ...])[0].sum())
        assert attribution.forecast_total == pytest.approx(
            total, abs=EXACTNESS_TOLERANCE
        )


def check_channels_sum_to_total(
    model: Forecaster, batch: WindowBatch, tmp_path: Path, windows: np.ndarray
) -> None:
    """4. The attribution is exact algebra: the parts sum to the whole."""
    del tmp_path
    for x in windows:
        attribution = model.explain(x, batch.channels)
        assert sum(attribution.per_channel.values()) == pytest.approx(
            attribution.forecast_total, abs=EXACTNESS_TOLERANCE
        )
        assert tuple(attribution.per_channel) == batch.channels


def check_save_load_roundtrip(
    model: Forecaster, batch: WindowBatch, tmp_path: Path
) -> None:
    """5. save -> load -> predict reproduces byte-identical output.

    Provenance is asserted to survive as well. A checkpoint that predicts identically but
    forgets what it was fitted on is exactly the checkpoint GB-25 cannot audit.
    """
    path = tmp_path / "checkpoint.json"
    model.save(str(path))
    restored = type(model).load(str(path))

    assert np.array_equal(restored.predict(batch.X), model.predict(batch.X))
    assert restored.fitted == model.fitted


def check_no_cross_window_leakage(
    model: Forecaster, batch: WindowBatch, tmp_path: Path
) -> None:
    """6. A window's prediction depends on that window alone.

    See the module note in ``tests/causality.py``: for a model, "the future" is later
    *windows*, not later lags. Every lag inside a window is at or before that window's own
    timestamp — GB-9 assembles it that way and GB-10 proves it — so the leak a model can
    still introduce is the batch: any statistic computed across ``X`` mixes windows, and
    because a batch is time-ordered, the mixture flows backwards from later windows into
    earlier predictions. Instance normalisation implemented as *batch* normalisation is
    that bug, and it is a live risk for FITS's RIN stage in GB-41.

    Reusing ``assert_causal`` rather than writing a model-specific check is what makes the
    two perturbation modes apply here too: ``scale`` catches a leaked batch mean, and
    ``shuffle`` catches a model carrying state across rows in order.
    """
    del tmp_path
    lags, channels = batch.X.shape[1], batch.X.shape[2]
    flat = pd.DataFrame(
        batch.X.reshape(len(batch.timestamps), -1), index=batch.timestamps
    )

    def predict_windows(frame: pd.DataFrame) -> pd.DataFrame:
        X = frame.to_numpy(dtype="float32").reshape(-1, lags, channels)
        return pd.DataFrame(model.predict(X), index=frame.index)

    predict_windows.__name__ = f"{model.name}.predict"
    assert_causal(predict_windows, flat)


def check_fit_records_provenance(
    model: Forecaster, batch: WindowBatch, tmp_path: Path
) -> None:
    """7. fit records what it was fitted on, truthfully.

    Not in spec 4.4's original six. Added in GB-11 with the protocol member it enforces:
    GB-25 audits every forecaster the same way, and a protocol member with no test is a
    docstring.
    """
    del tmp_path
    fitted = model.fitted

    assert isinstance(fitted, FitProvenance)
    assert fitted.channels == batch.channels
    assert fitted.symbols == (batch.symbol,)
    assert fitted.source == batch.source
    assert fitted.fitted_start == batch.timestamps[0]
    assert fitted.fitted_end == batch.timestamps[-1]
    assert fitted.n_windows == len(batch.timestamps)
    assert (fitted.input_len, fitted.horizon) == (model.input_len, model.horizon)


PROPERTIES = (
    check_shape_and_dtype,
    check_determinism,
    check_explain_total_matches_predict,
    check_channels_sum_to_total,
    check_save_load_roundtrip,
    check_no_cross_window_leakage,
    check_fit_records_provenance,
)

# The two exactness properties need the random-window sample; the rest do not. Keeping
# one uniform call signature is what lets the real test and the teeth test share them.
NEEDS_WINDOWS = frozenset(
    {check_explain_total_matches_predict, check_channels_sum_to_total}
)

# Property -> the property it presupposes.
#
# Properties 5 and 6 both work by recomputing and comparing exactly, so neither can hold
# for a model whose output changes between two calls, and — the part that matters —
# neither can tell non-determinism apart from the fault it is looking for. Reported as
# three plain failures, they would send a reader to debug a leak that is not there.
# So the dependency is structural: the dependants skip, naming property 2 as the cause,
# and exactly one red remains for the reader to follow.
#
# This mapping is the single definition of the relation. The confinement test derives its
# exemptions from it rather than restating them.
DEPENDS_ON = {
    check_save_load_roundtrip: check_determinism,
    check_no_cross_window_leakage: check_determinism,
}


def dependants_of(check) -> frozenset:
    """Every property that presupposes ``check``."""
    return frozenset(
        dependant
        for dependant, prerequisite in DEPENDS_ON.items()
        if prerequisite is check
    )


def unmet_dependency(check, model: Forecaster, batch: WindowBatch) -> str | None:
    """The reason ``check`` cannot be evaluated for ``model``, or ``None``.

    Returned as a string rather than raised, so the behaviour is directly assertable
    without catching pytest's control-flow exception.
    """
    prerequisite = DEPENDS_ON.get(check)
    if prerequisite is None:
        return None
    try:
        prerequisite(model, batch, Path())
    except AssertionError:
        return (
            f"{check.__name__} presupposes {prerequisite.__name__}, which fails for "
            f"{type(model).__name__}. Its result here would be meaningless: this check "
            "recomputes and compares exactly, so it cannot tell non-determinism apart "
            f"from the fault it looks for. Fix {prerequisite.__name__} first."
        )
    return None


def run_property(check, model, batch, tmp_path, windows) -> None:
    reason = unmet_dependency(check, model, batch)
    if reason is not None:
        pytest.skip(reason)
    if check in NEEDS_WINDOWS:
        check(model, batch, tmp_path, windows)
    else:
        check(model, batch, tmp_path)


# ── every registered forecaster satisfies every property ─────────────────────


@pytest.mark.parametrize("factory", FORECASTERS)
@pytest.mark.parametrize("check", PROPERTIES, ids=lambda fn: fn.__name__)
def test_forecaster_contract(
    factory,
    check,
    batch: WindowBatch,
    windows: np.ndarray,
    cfg: Config,
    tmp_path: Path,
) -> None:
    run_property(check, factory(batch, cfg), batch, tmp_path, windows)


@pytest.mark.parametrize("factory", FORECASTERS)
def test_forecaster_satisfies_the_protocol(
    factory, batch: WindowBatch, cfg: Config
) -> None:
    """A structural check, cheap and independent of the properties above."""
    assert isinstance(factory(batch, cfg), Forecaster)


# ── the properties have teeth ────────────────────────────────────────────────
#
# One deliberately broken forecaster per property, each breaking that property and no
# other. Every one of them subclasses Persistence, so the break is the only difference.


class WrongDtypeForecaster(PersistenceForecaster):
    """Returns float64. The shape is right, which is what makes it worth catching."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return super().predict(X).astype("float64")


class JitteryForecaster(PersistenceForecaster):
    """Draws from an unseeded generator, as a dropout layer left in train mode would.

    The jitter is smaller than the exactness tolerance, so the two algebraic properties
    still hold and the break stays as narrow as it can be. It cannot be narrowed further:
    see :data:`BROKEN` on why properties 5 and 6 presuppose property 2.
    """

    JITTER = EXACTNESS_TOLERANCE / 1000

    def predict(self, X: np.ndarray) -> np.ndarray:
        noise = np.random.default_rng().standard_normal((X.shape[0], self.horizon))
        return (super().predict(X) + self.JITTER * noise).astype("float32")


class MisreportingForecaster(PersistenceForecaster):
    """Predicts zero but claims a non-zero total — an explanation of a different model.

    The channels are adjusted to sum to the claimed total, so the attribution is
    internally consistent and only its agreement with ``predict`` is broken. An
    attribution can be perfectly self-consistent and describe nothing that was forecast.
    """

    CLAIMED_TOTAL = 0.01

    def explain(self, x: np.ndarray, channels: tuple[str, ...]) -> Attribution:
        per_channel = dict.fromkeys(channels, 0.0)
        per_channel[channels[0]] = self.CLAIMED_TOTAL
        return replace(
            super().explain(x, channels),
            per_channel=per_channel,
            forecast_total=self.CLAIMED_TOTAL,
        )


class NonAdditiveForecaster(PersistenceForecaster):
    """Channel contributions that do not sum to the total: attribution by approximation.

    This is the shape of every SHAP-style result the project bans, reproduced here so the
    test that forbids it is demonstrably able to see it.
    """

    def explain(self, x: np.ndarray, channels: tuple[str, ...]) -> Attribution:
        attribution = super().explain(x, channels)
        return replace(
            attribution, per_channel=dict.fromkeys(attribution.per_channel, 0.01)
        )


class AmnesiacForecaster(PersistenceForecaster):
    """Predicts identically after a round trip but forgets what it was fitted on."""

    @classmethod
    def load(cls, path: str) -> AmnesiacForecaster:
        model = super().load(path)
        model.fitted = None
        return model


class BatchNormForecaster(PersistenceForecaster):
    """Normalises against the whole batch instead of each instance.

    The realistic version of this bug is FITS's RIN stage subtracting a batch mean rather
    than a per-window mean. Because a batch is ordered in time, that pulls later windows
    into earlier predictions.
    """

    def predict(self, X: np.ndarray) -> np.ndarray:
        last = X[:, -1, 0]
        centred = (last - last.mean()).astype("float32")
        return np.repeat(centred[:, None], self.horizon, axis=1)


class ForgetfulFitForecaster(PersistenceForecaster):
    """A no-op fit that records nothing — the checkpoint GB-25 cannot audit."""

    def fit(self, batch: WindowBatch, val: WindowBatch | None = None) -> None:
        del batch, val


BROKEN = [
    pytest.param(WrongDtypeForecaster, check_shape_and_dtype, id="dtype"),
    pytest.param(JitteryForecaster, check_determinism, id="determinism"),
    pytest.param(
        MisreportingForecaster, check_explain_total_matches_predict, id="explain-total"
    ),
    pytest.param(NonAdditiveForecaster, check_channels_sum_to_total, id="additivity"),
    pytest.param(AmnesiacForecaster, check_save_load_roundtrip, id="save-load"),
    pytest.param(BatchNormForecaster, check_no_cross_window_leakage, id="batch-leak"),
    pytest.param(ForgetfulFitForecaster, check_fit_records_provenance, id="provenance"),
]


@pytest.mark.parametrize("broken,check", BROKEN)
def test_each_property_rejects_a_forecaster_that_breaks_it(
    broken,
    check,
    batch: WindowBatch,
    windows: np.ndarray,
    cfg: Config,
    tmp_path: Path,
) -> None:
    """The property under test must fail on the model built to break it."""
    model = broken(input_len=cfg.window.input_len, horizon=cfg.window.horizon)
    model.fit(batch)

    with pytest.raises(AssertionError):
        run_property(check, model, batch, tmp_path, windows)


@pytest.mark.parametrize("broken,check", BROKEN)
def test_a_broken_forecaster_still_passes_the_unrelated_properties(
    broken,
    check,
    batch: WindowBatch,
    windows: np.ndarray,
    cfg: Config,
    tmp_path: Path,
) -> None:
    """Each break is confined to its own property and that property's dependants.

    Without this, a single sloppy break could satisfy every ``pytest.raises`` above while
    leaving several properties unproven — the suite would look thorough and be hollow.
    It caught two such breaks on first run.

    Dependants are exempt and come from :data:`DEPENDS_ON`, not from a second list, so
    the relation cannot be stated twice and drift.
    """
    model = broken(input_len=cfg.window.input_len, horizon=cfg.window.horizon)
    model.fit(batch)
    exempt = dependants_of(check)

    for other in PROPERTIES:
        if other is check or other in exempt:
            continue
        run_property(other, model, batch, tmp_path, windows)


@pytest.mark.parametrize("dependant", sorted(DEPENDS_ON, key=lambda fn: fn.__name__))
def test_a_dependent_property_defers_instead_of_failing(
    dependant,
    batch: WindowBatch,
    windows: np.ndarray,
    cfg: Config,
    tmp_path: Path,
) -> None:
    """A reader facing three reds cannot tell which is real, and debugs the wrong one.

    So when property 2 fails, its dependants must not report as ordinary failures. They
    skip, naming property 2, leaving exactly one red to follow.
    """
    model = JitteryForecaster(
        input_len=cfg.window.input_len, horizon=cfg.window.horizon
    )
    model.fit(batch)

    reason = unmet_dependency(dependant, model, batch)
    assert reason is not None
    assert "check_determinism" in reason
    assert "meaningless" in reason

    with pytest.raises(pytest.skip.Exception, match="check_determinism"):
        run_property(dependant, model, batch, tmp_path, windows)


def test_a_sound_forecaster_has_no_unmet_dependencies(
    batch: WindowBatch, cfg: Config
) -> None:
    """The guard must not swallow a real failure by deferring for a healthy model."""
    model = _persistence(batch, cfg)

    for check in PROPERTIES:
        assert unmet_dependency(check, model, batch) is None


# ── the baseline says what it means ──────────────────────────────────────────


def test_persistence_predicts_exactly_zero(batch: WindowBatch, cfg: Config) -> None:
    """`ln(C_t / C_t) == 0`: predicting zero is the statement "the price does not move"."""
    model = _persistence(batch, cfg)

    assert not model.predict(batch.X).any()


def test_persistence_names_every_channel_in_its_attribution(
    batch: WindowBatch, cfg: Config
) -> None:
    """An empty attribution is indistinguishable from a bug; zeros are an answer."""
    attribution = _persistence(batch, cfg).explain(batch.X[0], batch.channels)

    assert tuple(attribution.per_channel) == batch.channels
    assert set(attribution.per_channel.values()) == {0.0}
    assert attribution.per_lag is None
    assert attribution.per_frequency is None
    assert attribution.gain_phase is None


def test_predict_refuses_a_window_of_the_wrong_length(
    batch: WindowBatch, cfg: Config
) -> None:
    """Zeros of any requested shape would pass a contract a real model would fail."""
    model = _persistence(batch, cfg)

    with pytest.raises(ValueError, match=f"{INPUT_LEN} lags"):
        model.predict(batch.X[:, :-1, :])


def test_an_unfitted_forecaster_records_no_provenance(cfg: Config) -> None:
    model = PersistenceForecaster(input_len=INPUT_LEN, horizon=HORIZON)

    assert model.fitted is None


def test_a_checkpoint_from_another_version_is_refused(
    batch: WindowBatch, cfg: Config, tmp_path: Path
) -> None:
    path = tmp_path / "checkpoint.json"
    _persistence(batch, cfg).save(str(path))
    path.write_text(
        path.read_text(encoding="utf-8").replace('"version": 1', '"version": 99'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="version"):
        PersistenceForecaster.load(str(path))
