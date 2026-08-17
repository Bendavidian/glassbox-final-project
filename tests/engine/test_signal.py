"""GB-20 acceptance: the three verdicts, and a module with no numbers in it.

The calibration half of GB-20 is tested in ``tests/backtest/test_calibrate.py``, mirroring
the split in the package: the decision logic is L4 and the grid search that chooses its
band runs the backtester, one layer up. The tests GB-20 named live in whichever file
matches the module under test - thresholds differing per fold and the fit-isolation proof
are there, the flat-forecast verdict and the no-constants scan are here.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glassbox.config.loader import Config, load_config
from glassbox.contracts.schemas import Forecast
from glassbox.engine import signal

HORIZON = 5
AS_OF = pd.Timestamp("2024-06-03", tz="UTC")

# A band with round numbers so a test reads at a glance. Nothing in the module knows them.
BAND = signal.Thresholds(lower=0.02, upper=0.10)


@pytest.fixture
def cfg() -> Config:
    return load_config()


def forecast(steps: list[float], symbol: str = "TEST", as_of=AS_OF) -> Forecast:
    return Forecast(path=np.array(steps, dtype="float32"), symbol=symbol, as_of=as_of)


def level(total: float, steps: int = HORIZON) -> list[float]:
    """A path summing to ``total`` with every step the same sign as it."""
    return [total / steps] * steps


# ── the three verdicts ───────────────────────────────────────────────────────


def test_a_flat_forecast_produces_hold(cfg: Config) -> None:
    """No move predicted, no action. The verdict GB-20 named."""
    verdict = signal.decide(forecast([0.0] * HORIZON), BAND, cfg)

    assert verdict.action == signal.HOLD
    assert verdict.trend_strength == 0.0
    assert verdict.up_points == 0
    assert verdict.passed_threshold is False


def test_a_forecast_inside_the_band_with_enough_up_points_enters(cfg: Config) -> None:
    verdict = signal.decide(forecast(level(0.05)), BAND, cfg)

    assert verdict.action == signal.ENTER_LONG
    assert verdict.passed_threshold is True
    assert verdict.trend_strength == pytest.approx(0.05, rel=1e-6)
    assert verdict.up_points == HORIZON


def test_a_forecast_below_the_band_holds(cfg: Config) -> None:
    """Positive but not enough to pay for the round trip the band was calibrated on."""
    verdict = signal.decide(forecast(level(0.01)), BAND, cfg)

    assert verdict.action == signal.HOLD
    assert verdict.passed_threshold is False


def test_a_forecast_above_the_upper_bound_holds(cfg: Config) -> None:
    """An implausible forecast is a reason to do nothing, not a reason to do more.

    This is the case that separates an upper bound from no upper bound, and it is the one
    a reader assumes is missing: a 40% five-day move is far outside anything the model
    produced in validation, so it is more likely a broken input than an opportunity.
    """
    verdict = signal.decide(forecast(level(0.40)), BAND, cfg)

    assert verdict.action == signal.HOLD
    assert verdict.passed_threshold is False
    assert verdict.trend_strength > BAND.upper


def test_the_band_can_pass_while_the_path_disagrees_with_itself(cfg: Config) -> None:
    """Strong enough, but carried by one step. ``passed_threshold`` records the difference.

    The whole point of the field: a decision record must be able to say "the trend was
    there and the path was not" rather than reporting an undifferentiated hold.
    """
    assert cfg.signal.min_up_points > 1
    one_jump = forecast([-0.01, -0.01, -0.01, -0.01, 0.09])

    verdict = signal.decide(one_jump, BAND, cfg)

    assert verdict.trend_strength == pytest.approx(0.05, rel=1e-5)
    assert verdict.up_points == 1
    assert verdict.passed_threshold is True  # the band admitted it
    assert verdict.action == signal.HOLD  # the confirmation did not


def test_a_forecast_that_turns_down_by_the_mirror_of_the_band_exits(
    cfg: Config,
) -> None:
    """Exit at ``-lower``: the conviction it took to get in, pointing the other way."""
    verdict = signal.decide(forecast(level(-0.05)), BAND, cfg)

    assert verdict.action == signal.EXIT
    assert verdict.passed_threshold is False


def test_mild_weakness_holds_rather_than_churning_out(cfg: Config) -> None:
    """Between ``-lower`` and ``lower`` nothing happens, which is why the mirror is not 0.

    A bare-zero exit would close a position on any forecast weakness at all - noise the
    entry rule would not have acted on - and pay the round trip for it.
    """
    verdict = signal.decide(forecast(level(-0.005)), BAND, cfg)

    assert verdict.action == signal.HOLD


def test_no_up_points_confirmation_is_required_to_exit(cfg: Config) -> None:
    """Getting out is easier than getting in: a position already carries risk."""
    verdict = signal.decide(forecast(level(-0.05)), BAND, cfg)

    assert verdict.up_points == 0
    assert verdict.action == signal.EXIT


# ── the band is the only source of thresholds ────────────────────────────────


def test_the_module_source_contains_no_numeric_threshold(package_root: Path) -> None:
    """No magic number, and no *number*, in the decision layer.

    The rule GB-20 asked for, checked on the syntax tree rather than by reading. Every
    magnitude must arrive as a calibrated argument or from the config; the only literals
    tolerated are ``0`` and ``1``, which appear as sign and dimension comparisons rather
    than as levels. A float literal anywhere in this module would be a threshold somebody
    chose by looking at results.
    """
    tree = ast.parse(
        (package_root / "engine" / "signal.py").read_text(encoding="utf-8")
    )
    numbers = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float)
        if not isinstance(node.value, bool)
    ]

    assert not [value for value in numbers if isinstance(value, float)]
    assert set(numbers) <= {
        0,
        1,
    }, f"numeric literals in signal.py: {sorted(set(numbers))}"


def test_the_config_offers_no_thresholds_to_fall_back_on(cfg: Config) -> None:
    """`min_trend` and `max_trend` are null, so nothing can quietly read them instead."""
    assert cfg.signal.min_trend is None
    assert cfg.signal.max_trend is None


def test_the_same_forecast_decides_differently_under_different_bands(
    cfg: Config,
) -> None:
    """The verdict is a function of the calibrated band, not of the forecast alone.

    If it were not, per-fold calibration would be decoration.
    """
    path = forecast(level(0.03))

    permissive = signal.decide(path, signal.Thresholds(lower=0.01), cfg)
    selective = signal.decide(path, signal.Thresholds(lower=0.05), cfg)

    assert permissive.action == signal.ENTER_LONG
    assert selective.action == signal.HOLD


# ── Thresholds refuses the bands that would be silently wrong ────────────────


def test_a_non_positive_lower_bound_is_refused() -> None:
    """Zero would enter on a forecast of no move and make entry and exit overlap."""
    with pytest.raises(ValueError, match="positive"):
        signal.Thresholds(lower=0.0)
    with pytest.raises(ValueError, match="positive"):
        signal.Thresholds(lower=-0.01)


def test_an_inverted_band_is_refused() -> None:
    with pytest.raises(ValueError, match="below lower"):
        signal.Thresholds(lower=0.05, upper=0.01)


def test_a_standing_aside_band_admits_nothing_and_exits_nothing(cfg: Config) -> None:
    """The fold that validation rejected: every verdict is hold, however strong the path."""
    aside = signal.Thresholds.never()

    assert aside.fires is False
    assert signal.decide(forecast(level(10.0)), aside, cfg).action == signal.HOLD
    assert signal.decide(forecast(level(-10.0)), aside, cfg).action == signal.HOLD


# ── the path itself ──────────────────────────────────────────────────────────


def test_trend_strength_is_the_cumulative_return_not_the_mean() -> None:
    """The quantity the direction metric scores, so the signal and the report agree."""
    steps = [0.01, -0.02, 0.03, 0.00, 0.01]

    assert signal.trend_strength(np.array(steps)) == pytest.approx(0.03, abs=1e-12)
    assert signal.up_points(np.array(steps)) == 3


def test_a_non_finite_forecast_is_refused_rather_than_held(cfg: Config) -> None:
    """A NaN that became `hold` would be indistinguishable from a considered decision."""
    with pytest.raises(ValueError, match="NaN"):
        signal.decide(forecast([0.01, math.nan, 0.01, 0.01, 0.01]), BAND, cfg)


# ── the shape the backtester and the live loop both consume ──────────────────


def test_decide_all_groups_by_timestamp_and_orders_by_symbol(cfg: Config) -> None:
    """One function builds the mapping for the backtest and for a live bar."""
    later = AS_OF + pd.Timedelta(days=1)
    decided = signal.decide_all(
        [
            forecast(level(0.05), symbol="MSFT"),
            forecast(level(0.05), symbol="AAPL"),
            forecast(level(-0.05), symbol="AAPL", as_of=later),
        ],
        BAND,
        cfg,
    )

    assert list(decided) == [AS_OF, later]
    assert [s.symbol for s in decided[AS_OF]] == ["AAPL", "MSFT"]
    assert decided[later][0].action == signal.EXIT


def test_the_backtester_reads_the_same_action_vocabulary() -> None:
    """One spelling of "enter_long", in the layer that produces it.

    Two copies is the GB-7 failure family: each side keeps its tests and the system quietly
    stops trading.
    """
    from glassbox.backtest import engine

    assert engine.ENTER_LONG is signal.ENTER_LONG
    assert engine.EXIT is signal.EXIT
