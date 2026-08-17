"""GB-19 acceptance: the numbers the report is made of, checked by hand.

Sharpe, max drawdown and direction accuracy are verified against literals computed away
from the code — a four-point equity curve worked out in decimal, and a nine-window
direction series counted by hand. If the expected values had been captured from a run,
agreement would be a tautology.

The three rulings each get their own section, because each one changes what a column in
the report means and two of them are silent when wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from glassbox.backtest import metrics
from glassbox.backtest.engine import END_OF_DATA, SIGNAL, STOP, Trade

# ── the hand-computed equity curve ───────────────────────────────────────────
#
#  bar  equity   return          running peak   drawdown
#   0   100.00   —               100.00         0.00
#   1   110.00   +0.10           110.00         0.00
#   2    99.00   -0.10           110.00         0.10   ← the deepest
#   3   108.90   +0.10           110.00         0.01
#
# returns          = [ 0.1, -0.1, 0.1 ]
# mean             = 0.1 / 3                      = 0.0333333333333333...
# deviations       = [ 0.0666..., -0.1333..., 0.0666... ]
# sum of squares   = 0.0044444... + 0.0177777... + 0.0044444... = 0.0266666...
# sample variance  = 0.0266666... / 2             = 0.0133333333333333...
# sample stdev     = 0.1154700538379251...
# Sharpe per bar   = 0.0333333... / 0.1154700...  = 0.2886751345948129
# total return     = 108.90 / 100.00 - 1          = 0.089
# max drawdown     = (110.00 - 99.00) / 110.00    = 0.10

CURVE = [100.00, 110.00, 99.00, 108.90]
SHARPE_PER_BAR = 0.2886751345948129
TOTAL_RETURN = 0.089
MAX_DRAWDOWN = 0.10

# Four bars spanning three calendar days, so the annualiser is exactly
#   (4 - 1) / (3 / 365.25) = 3 / 0.008213552...  = 365.25
PERIODS_PER_YEAR = 365.25


def curve(values: list[float] = CURVE, days: int | None = None) -> pd.Series:
    days = len(values) if days is None else days
    index = pd.date_range("2024-01-01", periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=index[:days], dtype="float64")


def make_trade(net_pnl: float, *, strategy: bool = True, reason: str = SIGNAL) -> Trade:
    """Only ``net_pnl`` and ``strategy_exit`` matter here; the rest is GB-18's business."""
    stamp = pd.Timestamp("2024-01-01", tz="UTC")
    return Trade(
        symbol="TEST",
        entry_time=stamp,
        exit_time=stamp,
        size=1.0,
        entry_price=100.0,
        exit_price=100.0 + net_pnl,
        gross_pnl=net_pnl,
        costs=0.0,
        net_pnl=net_pnl,
        exit_reason=reason,
        strategy_exit=strategy,
    )


def arm(
    name: str = "dlinear",
    values: list[float] = CURVE,
    trades: tuple[Trade, ...] = (),
    predicted: np.ndarray | None = None,
    actual: np.ndarray | None = None,
) -> metrics.ArmResult:
    return metrics.ArmResult(
        name=name,
        equity=curve(values),
        trades=trades,
        predicted=predicted,
        actual=actual,
    )


ENOUGH_TRADES = tuple(make_trade(pnl) for pnl in (10.0, -5.0, 8.0, -2.0, 4.0))


# ── the equity curve, by hand ────────────────────────────────────────────────


def test_periods_per_year_is_measured_from_the_folds_own_span() -> None:
    """Not a constant 252. Four bars over three days annualises at 365.25."""
    assert metrics.periods_per_year(curve().index) == pytest.approx(
        PERIODS_PER_YEAR, rel=1e-12
    )


def test_two_folds_of_different_length_annualise_differently() -> None:
    """The point of measuring it: a shared constant would scale these identically."""
    dense = pd.date_range("2024-01-01", periods=61, freq="B", tz="UTC")
    sparse = pd.date_range("2024-01-01", periods=61, freq="W", tz="UTC")

    assert metrics.periods_per_year(dense) > 200
    assert metrics.periods_per_year(sparse) < 60


def test_total_return_matches_the_hand_computed_value() -> None:
    assert metrics.total_return(arm()) == pytest.approx(TOTAL_RETURN, abs=1e-12)


def test_max_drawdown_matches_the_hand_computed_value() -> None:
    """110.00 down to 99.00 is 10% of the peak, and the later 1% dip does not beat it."""
    assert metrics.max_drawdown(arm()) == pytest.approx(MAX_DRAWDOWN, abs=1e-12)


def test_max_drawdown_is_zero_for_a_curve_that_only_rises() -> None:
    assert metrics.max_drawdown(arm(values=[100.0, 101.0, 102.0, 103.0])) == 0.0


def test_sharpe_matches_the_hand_computed_value() -> None:
    """Both halves checked against literals: the per-bar ratio and the annualiser."""
    expected = SHARPE_PER_BAR * math.sqrt(PERIODS_PER_YEAR)

    assert metrics.sharpe(arm(trades=ENOUGH_TRADES)) == pytest.approx(
        expected, rel=1e-12
    )
    # sqrt(365.25) = 19.1115149; 0.2886751345948129 * 19.1115149
    #              = 5.4848276 + 0.0321915 = 5.5170191
    assert expected == pytest.approx(5.5170191, rel=1e-7)


def test_sharpe_uses_the_sample_standard_deviation() -> None:
    """ddof=1. With three returns the population version differs by sqrt(3/2) — 22%."""
    population = SHARPE_PER_BAR * math.sqrt(3 / 2)

    assert metrics.sharpe(arm(trades=ENOUGH_TRADES)) != pytest.approx(
        population * math.sqrt(PERIODS_PER_YEAR), rel=1e-6
    )


def test_daily_returns_are_the_simple_period_over_period_ratios() -> None:
    np.testing.assert_allclose(
        metrics.daily_returns(curve()), [0.1, -0.1, 0.1], rtol=0, atol=1e-15
    )


def test_the_drawdown_curve_is_non_negative_everywhere() -> None:
    values = metrics.drawdown_curve(curve())

    np.testing.assert_allclose(values, [0.0, 0.0, 0.10, 0.01], rtol=0, atol=1e-12)
    assert (values >= 0).all()


# ── ruling 3: Sharpe comes from the curve, not from the trades ───────────────


def test_sharpe_ignores_the_trade_count_once_the_minimum_is_met() -> None:
    """The property that decides the ruling: same curve, same Sharpe.

    Two arms with identical equity curves must score identically however many trades
    produced them. A per-trade Sharpe would give the sparser arm a different number, and
    the study would then be comparing trade frequency rather than performance.
    """
    few = arm(trades=ENOUGH_TRADES)
    many = arm(trades=ENOUGH_TRADES + tuple(make_trade(1.0) for _ in range(20)))

    assert metrics.sharpe(few) == pytest.approx(metrics.sharpe(many), rel=1e-12)


def test_sharpe_reflects_days_spent_flat() -> None:
    """The other half of the ruling: sitting out is a cost the curve records.

    Both curves end at the same place, but one earned it steadily and one in a single
    jump followed by nothing. Per-trade returns cannot tell them apart; daily returns can.
    """
    steady = arm(values=[100.0, 102.0, 104.04, 106.1208], trades=ENOUGH_TRADES)
    lumpy = arm(values=[100.0, 106.1208, 106.1208, 106.1208], trades=ENOUGH_TRADES)

    assert metrics.total_return(steady) == pytest.approx(
        metrics.total_return(lumpy), rel=1e-9
    )
    assert metrics.sharpe(steady) != pytest.approx(metrics.sharpe(lumpy), rel=1e-6)


# ── ruling 2: too few trades to have a Sharpe ────────────────────────────────


def test_sharpe_is_undefined_below_the_trade_minimum() -> None:
    """Two trades cannot produce a risk-adjusted return; a number here would be fiction."""
    thin = arm(trades=tuple(make_trade(10.0) for _ in range(2)))

    assert metrics.MIN_TRADES_FOR_SHARPE == 5
    assert math.isnan(metrics.sharpe(thin))


def test_sharpe_is_defined_exactly_at_the_minimum() -> None:
    """The boundary, pinned: 5 is enough, 4 is not."""
    at = arm(trades=tuple(make_trade(1.0) for _ in range(5)))
    below = arm(trades=tuple(make_trade(1.0) for _ in range(4)))

    assert math.isfinite(metrics.sharpe(at))
    assert math.isnan(metrics.sharpe(below))


def test_administrative_exits_do_not_count_toward_the_trade_minimum() -> None:
    """They were not decisions, so they cannot license a Sharpe."""
    padded = arm(
        trades=tuple(
            make_trade(1.0, strategy=False, reason=END_OF_DATA) for _ in range(9)
        )
        + (make_trade(1.0),)
    )

    assert len(padded.trades) == 10
    assert len(padded.strategy_trades) == 1
    assert math.isnan(metrics.sharpe(padded))


def test_sharpe_is_undefined_for_a_curve_that_never_moved() -> None:
    """Zero variance is not infinite Sharpe. Reporting infinity would be worse than NaN."""
    flat = arm(values=[100.0] * 4, trades=ENOUGH_TRADES)

    assert math.isnan(metrics.sharpe(flat))


# ── ruling 1: direction accuracy, and what persistence's IS ──────────────────
#
# Nine windows, counted by hand on the H-day cumulative sign:
#
#   window   forecast sum   realised sum   call?   correct?
#     0          +0.02          +0.03       yes      yes
#     1          -0.01          -0.02       yes      yes
#     2          +0.01          -0.04       yes      no
#     3          -0.03          +0.01       yes      no
#     4          +0.02          +0.01       yes      yes
#     5          +0.01          +0.02       yes      yes
#     6          -0.02          -0.01       yes      yes
#     7          -0.01          +0.03       yes      no
#     8          +0.03          +0.02       yes      yes
#
# 6 correct of 9 called = 0.6666666666666666
# realised: 6 up, 3 down -> up rate 0.6666..., base rate max(0.666..., 0.333...) = 0.666...

FORECAST_SUMS = [0.02, -0.01, 0.01, -0.03, 0.02, 0.01, -0.02, -0.01, 0.03]
REALISED_SUMS = [0.03, -0.02, -0.04, 0.01, 0.01, 0.02, -0.01, 0.03, 0.02]
DIRECTION_ACCURACY = 6 / 9
BASE_RATE = 6 / 9


def paths(sums: list[float], horizon: int = 4) -> np.ndarray:
    """Spread each total evenly across the horizon, so the sums are what was intended."""
    return np.array([[value / horizon] * horizon for value in sums], dtype="float32")


def forecasting_arm(
    name: str = "dlinear", forecast: list[float] | None = None
) -> metrics.ArmResult:
    return arm(
        name=name,
        trades=ENOUGH_TRADES,
        predicted=paths(FORECAST_SUMS if forecast is None else forecast),
        actual=paths(REALISED_SUMS),
    )


def test_direction_accuracy_matches_the_hand_count() -> None:
    assert metrics.direction_accuracy(forecasting_arm()) == pytest.approx(
        DIRECTION_ACCURACY, abs=1e-12
    )


def test_direction_accuracy_reads_the_cumulative_sign_not_each_step() -> None:
    """It is the H-day trend the signal layer consumes, not the individual days.

    This forecast has the right total and the wrong shape: every window sums to the
    realised sign while three of its four steps point the other way. A per-step metric
    would score it near zero; the metric the study reports scores it 1.0.
    """
    horizon = 4
    predicted = np.array(
        [[-value, -value, -value, 4.0 * value] for value in REALISED_SUMS],
        dtype="float32",
    )
    zigzag = metrics.ArmResult(
        name="zigzag",
        equity=curve(),
        predicted=predicted,
        actual=paths(REALISED_SUMS, horizon),
    )

    assert metrics.direction_accuracy(zigzag) == pytest.approx(1.0, abs=1e-12)


def test_a_zero_forecast_makes_no_call_and_is_excluded() -> None:
    """Persistence forecasts zero, and zero agrees with nothing.

    Counting these as wrong would score a model that declines to predict at 0.0 rather
    than at chance, and every other arm's delta against it would be meaningless.
    """
    half_silent = list(FORECAST_SUMS)
    half_silent[2] = 0.0  # was wrong
    half_silent[3] = 0.0  # was wrong

    accuracy = metrics.direction_accuracy(forecasting_arm(forecast=half_silent))

    assert accuracy == pytest.approx(6 / 7, abs=1e-12)  # 6 of the 7 remaining calls
    assert accuracy > DIRECTION_ACCURACY


def test_persistence_has_no_direction_accuracy_at_all() -> None:
    """The consequence that shapes the report: NaN, not 0.0 and not 0.5.

    Persistence never expresses a direction, so it cannot be the reference for this
    column. 0.5 would be a number the baseline never produced.
    """
    persistence = forecasting_arm(name="persistence", forecast=[0.0] * 9)

    assert math.isnan(metrics.direction_accuracy(persistence))
    assert not math.isclose(0.5, 0.0)  # the tempting convention, deliberately not used


def test_a_delta_against_persistence_direction_is_undefined_too() -> None:
    """NaN propagates rather than being quietly replaced, so nothing reports a false lead."""
    persistence = forecasting_arm(name="persistence", forecast=[0.0] * 9)

    assert math.isnan(metrics.direction_accuracy(forecasting_arm(), persistence))


def test_the_base_rate_is_the_measured_majority_direction() -> None:
    """Equity drifts up, so 0.5 is the wrong bar. This one is measured, not assumed."""
    assert metrics.directional_base_rate(forecasting_arm()) == pytest.approx(
        BASE_RATE, abs=1e-12
    )


def test_the_base_rate_ignores_the_forecast_entirely() -> None:
    """It is a property of the test period, so every arm on one fold shares it."""
    first = metrics.directional_base_rate(forecasting_arm())
    second = metrics.directional_base_rate(forecasting_arm(forecast=[0.0] * 9))

    assert first == pytest.approx(second, abs=1e-12)


# ── forecast error ───────────────────────────────────────────────────────────


def test_mae_and_rmse_match_a_recomputation() -> None:
    predicted = paths(FORECAST_SUMS)
    actual = paths(REALISED_SUMS)
    scored = metrics.ArmResult(
        name="a", equity=curve(), predicted=predicted, actual=actual
    )
    difference = predicted.astype("float64") - actual.astype("float64")

    assert metrics.mae(scored) == pytest.approx(np.mean(np.abs(difference)), rel=1e-12)
    assert metrics.rmse(scored) == pytest.approx(
        np.sqrt(np.mean(difference**2)), rel=1e-12
    )


def test_a_flatter_forecast_wins_mae_which_is_why_it_is_not_the_headline() -> None:
    """§7.3's ban, demonstrated rather than asserted.

    Halving every forecast toward zero improves MAE here while leaving direction accuracy
    untouched — the exact misreading the reporting rule exists to prevent.
    """
    honest = forecasting_arm()
    flattened = forecasting_arm(forecast=[value / 2 for value in FORECAST_SUMS])

    assert metrics.mae(flattened) < metrics.mae(honest)
    assert metrics.direction_accuracy(flattened) == pytest.approx(
        metrics.direction_accuracy(honest), abs=1e-12
    )


def test_nan_targets_are_refused_rather_than_averaged() -> None:
    """``build_windows`` leaves y as NaN for a live window; a metric over it is meaningless."""
    actual = paths(REALISED_SUMS)
    actual[0, 0] = np.nan
    scored = metrics.ArmResult(
        name="a", equity=curve(), predicted=paths(FORECAST_SUMS), actual=actual
    )

    with pytest.raises(ValueError, match="complete windows only"):
        metrics.mae(scored)


def test_an_arm_without_forecasts_reports_nan_rather_than_raising() -> None:
    """A trading-only comparison is legitimate; it just has no forecast columns."""
    trading_only = arm(trades=ENOUGH_TRADES)

    assert math.isnan(metrics.mae(trading_only))
    assert math.isnan(metrics.direction_accuracy(trading_only))
    assert math.isfinite(metrics.total_return(trading_only))


# ── administrative exits ─────────────────────────────────────────────────────


def test_hit_rate_excludes_administrative_exits() -> None:
    """GB-18 pinned the rule; this is where it is enforced.

    Three decided trades, two of them winners, plus a losing end-of-data liquidation that
    must not drag the rate to 0.5.
    """
    trades = (
        make_trade(10.0),
        make_trade(5.0),
        make_trade(-3.0),
        make_trade(-100.0, strategy=False, reason=END_OF_DATA),
    )

    assert metrics.hit_rate(arm(trades=trades)) == pytest.approx(2 / 3, abs=1e-12)


def test_average_trade_excludes_administrative_exits() -> None:
    trades = (
        make_trade(10.0),
        make_trade(-4.0),
        make_trade(-100.0, strategy=False, reason=END_OF_DATA),
    )

    assert metrics.average_trade(arm(trades=trades)) == pytest.approx(3.0, abs=1e-12)


def test_total_return_includes_administrative_exits() -> None:
    """The other half of the rule: the capital really was returned, so the curve keeps it.

    Read from the equity curve, which is complete by construction — so this holds without
    the metric having to know about the flag at all.
    """
    liquidated = arm(trades=(make_trade(-100.0, strategy=False, reason=END_OF_DATA),))

    assert metrics.total_return(liquidated) == pytest.approx(TOTAL_RETURN, abs=1e-12)


def test_filtering_is_on_the_field_not_the_reason_string() -> None:
    """A trade flagged administrative is excluded whatever its reason reads.

    The rule GB-18 pinned is "filter on the boolean". A metric matching on `exit_reason`
    would pass every test above and fail here.
    """
    mislabelled = make_trade(-100.0, strategy=False, reason=STOP)

    assert metrics.hit_rate(arm(trades=(make_trade(1.0), mislabelled))) == 1.0


def test_a_fold_that_decided_nothing_has_no_hit_rate() -> None:
    assert math.isnan(metrics.hit_rate(arm(trades=())))
    assert math.isnan(metrics.average_trade(arm(trades=())))


# ── the report table ─────────────────────────────────────────────────────────


def test_summarise_gives_one_row_per_arm_with_deltas() -> None:
    persistence = forecasting_arm(name="persistence", forecast=[0.0] * 9)
    model = forecasting_arm(name="dlinear")

    table = metrics.summarise([persistence, model], persistence)

    assert list(table["arm"]) == ["persistence", "dlinear"]
    assert len(table) == 2
    for name, _, _ in metrics.METRICS:
        assert name in table.columns
        assert f"{name}_delta" in table.columns


def test_the_baselines_own_deltas_are_zero() -> None:
    """An honest self-comparison beats a blank row."""
    persistence = arm(name="persistence", trades=ENOUGH_TRADES)

    table = metrics.summarise([persistence], persistence)

    assert table["sharpe_delta"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert table["total_return_delta"].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_the_direction_column_is_referenced_to_chance_not_to_persistence() -> None:
    """The ruling, visible in the table: the reference is measured and it is recorded.

    Every other column is a persistence delta as §7.3 requires. This one cannot be,
    because persistence has no direction, and the table says which number was used
    instead rather than leaving a reader to guess.
    """
    persistence = forecasting_arm(name="persistence", forecast=[0.0] * 9)
    model = forecasting_arm(name="dlinear")

    table = metrics.summarise([model], persistence)
    row = table.iloc[0]

    assert row["direction_reference"] == pytest.approx(BASE_RATE, abs=1e-12)
    assert row["direction_accuracy_delta"] == pytest.approx(
        DIRECTION_ACCURACY - BASE_RATE, abs=1e-12
    )
    assert not math.isnan(row["direction_accuracy_delta"])


def test_the_table_carries_the_trade_counts_that_explain_a_blank_sharpe() -> None:
    """A NaN cell beside `n_strategy_trades = 2` explains itself; alone it does not."""
    thin = arm(name="thin", trades=(make_trade(1.0), make_trade(2.0)))

    table = metrics.summarise([thin], thin)
    row = table.iloc[0]

    assert row["n_trades"] == 2
    assert row["n_strategy_trades"] == 2
    assert math.isnan(row["sharpe"])


def test_every_metric_declares_which_direction_is_better() -> None:
    """So the report can render a delta's sign without a second table of conventions."""
    better_higher = {name: higher for name, _, higher in metrics.METRICS}

    assert better_higher["direction_accuracy"] is True
    assert better_higher["sharpe"] is True
    assert better_higher["mae"] is False
    assert better_higher["max_drawdown"] is False
