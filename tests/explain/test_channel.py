"""GB-30 acceptance: per-channel attribution is exact algebra.

The acceptance criterion in spec 9 is one line — ``Σ contributions == forecast`` within
1e-5 — and :func:`test_the_decomposition_is_exact_over_a_thousand_windows` is it. It
reports the maximum observed deviation rather than only asserting the bound, because
"under the tolerance" and "under the tolerance by four orders of magnitude" are different
claims and only the second one says the arithmetic is exact.
"""

from __future__ import annotations

import ast
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from glassbox.config.loader import Config, load_config
from glassbox.contracts.schemas import (
    EXACTNESS_TOLERANCE,
    Attribution,
    ChannelStats,
    WindowBatch,
)
from glassbox.explain.channel import ChannelLinear, attribute, cancellation, shares
from glassbox.features.builder import build_feature_frame, build_windows
from glassbox.model import ALL_FORECASTERS

INPUT_LEN = 30
HORIZON = 4
WINDOWS = 120

# 400: RSI's warm-up takes 325 rows (GB-27), leaving 75 for windows of 30.
BARS = 400

# The acceptance criterion names a thousand. An algebraic identity that holds on the data
# manifold and fails off it would pass on one window, which is the failure this count
# exists to make impossible.
EXACTNESS_SAMPLES = 1000

# Perturbation-based attribution libraries. The ban is written in spec 4.2, in three module
# docstrings and in CLAUDE.md, and until now was enforced by none of them.
BANNED = ("shap", "lime", "captum", "alibi", "eli5", "interpret", "innvestigate")


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg() -> Config:
    """Production config at a smaller window, so the suite stays quick."""
    base = load_config()
    return replace(
        base,
        window=replace(base.window, input_len=INPUT_LEN, horizon=HORIZON),
        model=replace(base.model, epochs=3, patience=2),
    )


@pytest.fixture
def batch(cfg: Config) -> WindowBatch:
    """A deterministic batch with the production channel set."""
    channels = cfg.channels.active_channels
    rng = np.random.default_rng(cfg.meta.seed)
    return WindowBatch(
        X=rng.standard_normal((WINDOWS, INPUT_LEN, len(channels))).astype("float32"),
        y=rng.standard_normal((WINDOWS, HORIZON)).astype("float32"),
        channels=channels,
        timestamps=pd.date_range("2020-01-01", periods=WINDOWS, freq="B", tz="UTC"),
        symbols=("AAPL",) * WINDOWS,
        source="yfinance",
    )


@pytest.fixture
def model(batch: WindowBatch, cfg: Config) -> ChannelLinear:
    """A fitted DLinear — the model whose weights are the explanation."""
    built = ALL_FORECASTERS["dlinear"](cfg, batch.channels)
    built.fit(batch)
    return built


@pytest.fixture
def sample_windows(cfg: Config) -> np.ndarray:
    """``EXACTNESS_SAMPLES`` random single windows, seeded from the config."""
    rng = np.random.default_rng(cfg.meta.seed + 2)
    shape = (EXACTNESS_SAMPLES, INPUT_LEN, len(cfg.channels.active_channels))
    return rng.standard_normal(shape).astype("float32")


def an_attribution(**per_channel: float) -> Attribution:
    """A hand-built attribution, for the share arithmetic. Bypasses ``from_terms``."""
    return Attribution(
        per_channel=dict(per_channel),
        per_lag=None,
        per_frequency=None,
        gain_phase=None,
        forecast_total=sum(per_channel.values()),
    )


def make_bars(n: int = BARS) -> pd.DataFrame:
    """A canonical bar frame: deterministic, rising and falling, single provenance."""
    index = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    close = pd.Series(
        [100.0 + 10.0 * math.sin(i / 7.0) + 0.05 * i for i in range(n)],
        index=index,
        dtype="float64",
    )
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": pd.Series(
                [1_000_000.0 + 5_000.0 * math.cos(i / 5.0) for i in range(n)],
                index=index,
            ),
            "log_return": np.log(close / close.shift(1)),
            "source": pd.Series("yfinance", index=index, dtype="string"),
        },
        index=index,
    )


# ── the acceptance criterion ─────────────────────────────────────────────────


def test_the_decomposition_is_exact_over_a_thousand_windows(
    model: ChannelLinear, sample_windows: np.ndarray, cfg: Config, capsys
) -> None:
    """Σ per_channel == forecast within 1e-5, over 1000 random windows.

    Reports the worst deviation seen. That number is the finding, not the pass.
    """
    channels = cfg.channels.active_channels
    worst = 0.0
    for x in sample_windows:
        found = attribute(model, x, channels)
        forecast = float(model.predict(x[None, ...])[0].sum())
        worst = max(worst, abs(sum(found.per_channel.values()) - forecast))

    margin = "exact" if worst == 0.0 else f"{EXACTNESS_TOLERANCE / worst:,.0f}x margin"
    with capsys.disabled():
        print(
            f"\nGB-30: max |sum(per_channel) - forecast| over {EXACTNESS_SAMPLES} "
            f"windows = {worst:.3e}, tolerance {EXACTNESS_TOLERANCE:.0e} ({margin})"
        )
    assert worst <= EXACTNESS_TOLERANCE


def test_every_registered_forecaster_is_exact(
    batch: WindowBatch, sample_windows: np.ndarray, cfg: Config
) -> None:
    """The criterion is not DLinear's alone. Persistence must satisfy it too."""
    for name, factory in sorted(ALL_FORECASTERS.items()):
        built = factory(cfg, batch.channels)
        built.fit(batch)
        for x in sample_windows[:50]:
            found = attribute(built, x, batch.channels)
            assert sum(found.per_channel.values()) == pytest.approx(
                float(built.predict(x[None, ...])[0].sum()), abs=EXACTNESS_TOLERANCE
            ), name


# ── the normalisation guard, through a real model ───────────────────────────
#
# `Attribution.from_terms` is unit-tested in tests/contracts/test_schemas.py, where it
# lives. What belongs here is the same guard reached through a fitted forecaster and a
# real normalised window, because that is the path a defect would actually arrive on.


def test_an_unaccounted_intercept_is_caught(model: ChannelLinear, cfg: Config) -> None:
    """A model that grows a bias must fail here, not report a wrong explanation.

    This is the concrete form of "any reversible normalisation must be accounted for": a
    constant the forecast contains and no channel owns cannot be attributed, so the sum
    stops closing and construction refuses.
    """
    channels = cfg.channels.active_channels
    x = np.zeros((INPUT_LEN, len(channels)), dtype="float32")

    with pytest.raises(ValueError, match="does not close"):
        Attribution.from_terms(model.linear_terms(x, channels), forecast_total=0.02)


def test_normalisation_does_not_break_exactness(
    model: ChannelLinear, cfg: Config
) -> None:
    """The same window normalised and raw: both decompositions close.

    ``build_windows`` z-scores the input before the model sees it, so the mean-subtraction
    lives inside each channel's own term and there is no constant to add back. This pins
    that rather than assuming it — a scaler moved outside the per-channel terms would break
    exactness, and this is where it would surface.
    """
    channels = cfg.channels.active_channels
    frame = build_feature_frame(make_bars(), cfg)
    stats = ChannelStats(
        channels=channels,
        mean=tuple(float(frame[channel].mean()) for channel in channels),
        std=tuple(float(frame[channel].std()) for channel in channels),
        fitted_start=frame.index[0],
        fitted_end=frame.index[-1],
        n_rows=len(frame),
    )

    raw = build_windows(frame, cfg, "AAPL")
    scaled = build_windows(frame, cfg, "AAPL", stats=stats)

    assert not np.array_equal(raw.X[0], scaled.X[0])  # the scaler really did something
    for window in (raw.X[0], scaled.X[0]):
        found = attribute(model, window, channels)
        assert sum(found.per_channel.values()) == pytest.approx(
            float(model.predict(window[None, ...])[0].sum()), abs=EXACTNESS_TOLERANCE
        )


# ── attribute() re-checks what the model reported ────────────────────────────


def test_attribute_agrees_with_the_models_own_explain(
    model: ChannelLinear, sample_windows: np.ndarray, cfg: Config
) -> None:
    """One decomposition means the two paths return the same numbers, not close ones."""
    channels = cfg.channels.active_channels
    for x in sample_windows[:20]:
        assert (
            attribute(model, x, channels).per_channel
            == model.explain(x, channels).per_channel
        )


def test_attribute_refuses_a_model_that_explains_a_different_forecast(
    model: ChannelLinear, cfg: Config
) -> None:
    """Internally consistent and describing nothing the model forecast.

    ``from_terms`` cannot see this: the attribution closes against the total the model
    reported. Only a layer that asks ``predict`` itself can catch it, which is why
    ``attribute`` does.
    """
    channels = cfg.channels.active_channels
    x = np.zeros((INPUT_LEN, len(channels)), dtype="float32")

    class Misreporting:
        channels = model.channels

        def linear_terms(self, window, names):
            return model.linear_terms(window, names)

        def predict(self, X):
            return model.predict(X) + 1.0

        def explain(self, window, names):
            return model.explain(window, names)

    with pytest.raises(ValueError, match="describes a different forecast"):
        attribute(Misreporting(), x, channels)


def test_attribute_refuses_a_channel_set_the_model_does_not_hold(
    model: ChannelLinear, cfg: Config
) -> None:
    """Naming both sets, because the weights are indexed by position."""
    x = np.zeros((INPUT_LEN, len(cfg.channels.active_channels)), dtype="float32")

    with pytest.raises(ValueError, match="this model holds"):
        attribute(model, x, ("not_a_channel",))


def test_persistence_reports_every_channel_with_zero(
    batch: WindowBatch, cfg: Config
) -> None:
    """GB-11's ruling survives the move to from_terms."""
    baseline = ALL_FORECASTERS["persistence"](cfg, batch.channels)
    baseline.fit(batch)
    x = np.ones((INPUT_LEN, len(batch.channels)), dtype="float32")

    found = attribute(baseline, x, batch.channels)

    assert tuple(found.per_channel) == batch.channels
    assert set(found.per_channel.values()) == {0.0}
    assert found.forecast_total == 0.0


# ── shares: the opposite-sign ruling ─────────────────────────────────────────


def test_opposite_signs_give_bounded_shares() -> None:
    """The worked example from the ruling: +0.08 and −0.06 behind a forecast of +0.02.

    Percent-of-forecast would print 400% and −300%. Percent-of-gross prints +57% and −43%.
    """
    found = shares(an_attribution(a=0.08, b=-0.06))

    assert found["a"] == pytest.approx(0.08 / 0.14)  # +57.1%
    assert found["b"] == pytest.approx(-0.06 / 0.14)  # −42.9%
    assert sum(abs(value) for value in found.values()) == pytest.approx(1.0)


def test_the_cancellation_carries_what_the_shares_no_longer_say() -> None:
    """0.02 of a gross 0.14 survived: 14.3%."""
    assert cancellation(an_attribution(a=0.08, b=-0.06)) == pytest.approx(0.02 / 0.14)


def test_channels_pointing_the_same_way_cancel_nothing() -> None:
    assert cancellation(an_attribution(a=0.03, b=0.05)) == pytest.approx(1.0)


def test_shares_are_signed_so_direction_survives() -> None:
    found = shares(an_attribution(a=0.08, b=-0.06))

    assert found["a"] > 0
    assert found["b"] < 0


# ── shares: the zero-forecast ruling ─────────────────────────────────────────


def test_a_forecast_of_exactly_zero_from_no_contribution_gives_zero_shares() -> None:
    """Persistence's case. Not NaN, not an exception, not an empty mapping."""
    found = shares(an_attribution(a=0.0, b=0.0))

    assert found == {"a": 0.0, "b": 0.0}
    assert not any(math.isnan(value) for value in found.values())


def test_a_forecast_of_exactly_zero_by_cancellation_still_has_shares() -> None:
    """The case percent-of-forecast cannot express at all: the denominator would be zero.

    The channels genuinely did something and it genuinely cancelled, and both halves of
    that sentence are readable here.
    """
    found = shares(an_attribution(a=0.05, b=-0.05))

    assert found["a"] == pytest.approx(0.5)
    assert found["b"] == pytest.approx(-0.5)
    assert cancellation(an_attribution(a=0.05, b=-0.05)) == 0.0


def test_cancellation_of_nothing_is_zero_not_nan() -> None:
    assert cancellation(an_attribution(a=0.0, b=0.0)) == 0.0


@settings(max_examples=200, deadline=None)
@given(
    contributions=st.lists(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=8,
    )
)
def test_shares_are_always_bounded_and_sum_to_one(contributions: list[float]) -> None:
    """Two properties no example can establish: the bound, and the magnitude sum."""
    found = shares(
        an_attribution(**{f"c{i}": value for i, value in enumerate(contributions)})
    )

    assert all(-1.0 <= value <= 1.0 for value in found.values())
    total = sum(abs(value) for value in found.values())
    assert total == pytest.approx(1.0) or total == 0.0


@settings(max_examples=200, deadline=None)
@given(
    contributions=st.lists(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=8,
    )
)
def test_cancellation_is_always_a_fraction(contributions: list[float]) -> None:
    found = cancellation(
        an_attribution(**{f"c{i}": value for i, value in enumerate(contributions)})
    )

    assert 0.0 <= found <= 1.0 + 1e-9


# ── the ban, enforced ────────────────────────────────────────────────────────


def test_the_explain_layer_imports_no_perturbation_library(package_root: Path) -> None:
    """Spec 4.2's ban, asserted rather than asked for.

    A perturbation library in this package would not fail a test — it would produce
    plausible numbers that do not sum to the forecast, which is the one defect this
    project's central claim rules out. So the ban is checked structurally, in the same
    spirit as ``signal.py``'s no-numeric-literal rule.
    """
    offenders = []
    for path in sorted((package_root / "explain").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            offenders += [
                f"{path.name}: {name}" for name in names if name.split(".")[0] in BANNED
            ]

    assert not offenders
