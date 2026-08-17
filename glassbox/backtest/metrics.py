"""X: forecast and trading metrics - MAE, RMSE, direction accuracy of the H-day trend,
total return, Sharpe, max drawdown, hit rate, all net of costs.

Written from scratch; reference/utils/metrics.py is unusable (see REFERENCE_AUDIT.md).
No result is reported as an absolute number: every metric carries a persistence delta,
and MSE on prices is banned as a headline metric.

Three rulings decide what the numbers mean. Each is stated here and tested.

**1. Sharpe is computed from the equity curve's daily returns, not from per-trade
returns.** They are different numbers and the report must say which. Daily returns are
what an investor experiences: they include the cost of sitting flat, which per-trade
Sharpe ignores entirely, so a strategy in the market ten days a year cannot look like one
in it every day. Per-trade Sharpe also has as many observations as trades - a dozen or so
per fold - so its estimate is mostly noise, and it is trivially inflated by taking fewer,
larger positions. Annualisation is well defined for a daily series and ill defined for a
series of trades with different holding periods. And two arms with the same equity curve
must get the same Sharpe whatever their trade counts, which only the curve version does.

**2. Below ``MIN_TRADES_FOR_SHARPE`` strategy trades, Sharpe is NaN rather than a number.**
See the constant for the reasoning.

**3. Direction accuracy is undefined for a forecast of exactly zero**, and the persistence
baseline forecasts zero every time. See :func:`direction_accuracy` - this is the ruling
with the largest consequence for the report, because it means **persistence cannot be the
baseline for the direction column** even though it is the baseline for everything else.
The reference used instead is :func:`always_long_accuracy`: the one constant strategy a
long-only system could actually have run, scoring the fold's own realised up rate. Not
0.5, which understates it, and not the per-fold majority class, which overstates it by
choosing the class with test-period knowledge.

Administrative exits - ``Trade.strategy_exit is False``, i.e. ``end_of_data`` - are
**included** in the equity curve and total return, because the curve must be complete and
the capital was genuinely returned, and **excluded** from hit rate and average trade,
because no decision was made. Filtering is on the boolean field, never on the
``exit_reason`` string, per the rule pinned in GB-18's ``Trade`` docstring.

Annualisation uses each fold's **own** bar count and calendar span rather than a constant
252. Folds are calendar months, so their bar counts differ, and a shared constant would
scale two folds' Sharpe by the same factor when they cover different amounts of time.

Implemented in GB-19.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from glassbox.backtest.engine import Trade

DAYS_PER_YEAR = 365.25

# Sharpe is an estimate, and below a handful of round trips it is an estimate of almost
# nothing. A 3-month test fold is ~61 bars; a strategy taking fewer than five round trips
# across it has held a position on a minority of days, so its equity curve is mostly flat
# and its return standard deviation is dominated by the few days it was exposed. The
# resulting ratio is large or small for reasons that have nothing to do with the strategy,
# and this project already treats `Sharpe > 2.0` as a leak alarm - a metric that can
# manufacture that number from two trades would make the alarm useless.
#
# Five is a judgement, not a derivation, and it is stated as one. What makes it safe is
# that the alternative is not "a slightly noisy number" but "a number with no sampling
# argument behind it at all", and that `summarise` reports the trade count beside the
# Sharpe so an empty cell explains itself.
MIN_TRADES_FOR_SHARPE = 5

# The trade count is a proxy, and GB-21 found the case where the proxy fails: **buy-and-hold
# has no strategy trades at all** - it enters once and its exit is the data running out -
# yet it is exposed on every bar of the fold, so its daily returns are entirely about what
# it held. The rule above was written for the opposite situation, a curve that is mostly
# flat because the account was mostly in cash, and the constant's own reasoning says so.
#
# So exposure is the second qualifying path, and it is the direct statement of the reason:
# an arm holding a position on at least half the fold's bars has an equity curve that
# describes its holdings, whatever its trade count. An arm satisfying neither has neither
# argument behind its Sharpe and still gets NaN.
MIN_EXPOSED_FRACTION = 0.5


@dataclass(frozen=True, eq=False)
class ArmResult:
    """One arm of the study on one fold: what it forecast and what it earned.

    ``predicted`` and ``actual`` are ``(B, H)`` log-return paths, aligned row for row.
    They are optional because a trading-only comparison is legitimate; the forecast
    metrics then return NaN rather than raising.

    ``eq=False``: it holds arrays and a Series, so a default equality would compare
    elementwise and raise on the ambiguous truth value.
    """

    name: str
    equity: pd.Series
    trades: tuple[Trade, ...] = ()
    predicted: np.ndarray | None = None
    actual: np.ndarray | None = None

    @property
    def strategy_trades(self) -> tuple[Trade, ...]:
        """Trades the strategy decided on.

        Excludes ``end_of_data`` liquidations, which the data running out caused rather
        than the strategy. Filters on the boolean, never on the ``exit_reason`` string.
        """
        return tuple(trade for trade in self.trades if trade.strategy_exit)


# ── forecast metrics ─────────────────────────────────────────────────────────


def mae(arm: ArmResult, baseline: ArmResult | None = None) -> float:
    """Mean absolute error over every horizon step. Lower is better.

    A **diagnostic** metric, not a headline one: §7.3 bans MSE on prices for the reason
    that also applies to MAE on returns - a forecast flattened toward zero wins it without
    being more useful. Report it beside direction accuracy, never instead of it.
    """
    return _delta(_mae_of(arm), baseline, _mae_of)


def rmse(arm: ArmResult, baseline: ArmResult | None = None) -> float:
    """Root mean squared error over every horizon step. Lower is better. Diagnostic."""
    return _delta(_rmse_of(arm), baseline, _rmse_of)


def direction_accuracy(arm: ArmResult, baseline: ArmResult | None = None) -> float:
    """Fraction of windows whose **H-day cumulative** forecast has the realised sign.

    The primary forecast metric: it is the quantity the signal layer consumes, and the
    only one of the three that a flatter forecast cannot win.

    **A forecast of exactly zero makes no directional call, and is excluded from the
    denominator.** ``sign(0)`` is 0, which agrees with nothing, so counting those windows
    as wrong would score a model that declines to predict at 0.0 rather than at chance -
    and counting them as right by convention would invent an opinion it never expressed.
    When no window carries a call the result is **NaN**, meaning undefined rather than
    zero.

    **This makes persistence's direction accuracy NaN, and that is the correct answer.**
    Persistence forecasts zero every time, so it never expresses a direction. It is the
    right baseline for MAE, RMSE, return and Sharpe - all of which it genuinely competes
    on - and it is not a directional model, so it cannot be the reference for this column.
    Substituting 0.5 would be inventing a number the baseline never produced, and the
    delta against it would be a comparison with a model that does not exist.

    The honest reference is :func:`always_long_accuracy` - the one constant strategy this
    project could actually have run - measured on the same realised returns rather than
    assumed. :func:`summarise` reports the direction column against it and records the
    number used; every other column is a persistence delta as §7.3 requires.
    """
    return _delta(_direction_of(arm), baseline, _direction_of)


def always_long_accuracy(arm: ArmResult) -> float:
    """Accuracy of calling **up** on every window: the fold's realised up rate.

    The reference the direction column is read against, and the only constant strategy
    that is **achievable**. It calls up every time, so it is right exactly as often as the
    H-day cumulative return is positive, and its score is therefore the test split's own up
    rate - reported per fold, so a reader can watch the bar move rather than assume 0.5.

    **The class choice comes from the training split; only the rate is measured on test.**
    Equities drift up, the up rate exceeds 0.5 in every training split this project has,
    and spec 2.1 is long-only - so "always up" is fixed before the test period is seen, and
    "always short" is not in the action space to begin with.

    That distinction is the whole point, and the earlier version of this function got it
    wrong: it returned ``max(up_rate, 1 - up_rate)`` per fold, which flips to "always
    short" on folds whose test period fell. Averaged over the 16 folds that read 0.5865
    against an up rate of 0.5625 - a bar 2.4 points above what any strategy could have
    reached, because no strategy knows in advance whether the coming quarter is up or down.
    It erred **against** the model, and it was still a bar set with test-period knowledge,
    which does not pass in this project in either direction.

    A window whose realised return is exactly zero counts against it: an up call was made
    and the market did not go up. NaN when there is nothing to measure.
    """
    if arm.actual is None or len(arm.actual) == 0:
        return math.nan
    realised = _require_finite(arm.actual, "actual").sum(axis=1)
    return float((realised > 0).mean())


# ── trading metrics ──────────────────────────────────────────────────────────


def total_return(arm: ArmResult, baseline: ArmResult | None = None) -> float:
    """Growth of the equity curve over the fold, as a fraction. Higher is better.

    **A total return, not a price return**: ``data/historical.py`` fetches with
    ``auto_adjust=True``, so dividends are already folded into the prices (GB-18).

    Administrative exits are **included** - the capital really was returned - because this
    is read from the curve, which is complete by construction.
    """
    return _delta(_total_return(arm), baseline, _total_return)


def sharpe(arm: ArmResult, baseline: ArmResult | None = None) -> float:
    """Annualised Sharpe of the equity curve's daily returns, at a zero risk-free rate.

    From the **curve**, not from per-trade returns - see the module docstring for why the
    two differ and why this one is reported. Annualised with the fold's own bar rate, from
    :func:`periods_per_year`.

    Returns NaN when the arm qualifies on **neither** path - fewer than
    :data:`MIN_TRADES_FOR_SHARPE` strategy trades *and* less than
    :data:`MIN_EXPOSED_FRACTION` of the fold's bars exposed - when there are fewer than two
    returns to take a standard deviation of, or when that standard deviation is zero. A
    curve that never moved has no risk-adjusted return, and reporting infinity for it would
    be worse than reporting nothing.
    """
    return _delta(_sharpe(arm), baseline, _sharpe)


def max_drawdown(arm: ArmResult, baseline: ArmResult | None = None) -> float:
    """Deepest peak-to-trough fall, as a positive fraction of the peak. Lower is better.

    Positive by convention, so 0.12 means the account fell 12% below its high-water mark.
    A curve that only rises returns 0.0.
    """
    return _delta(_max_drawdown(arm), baseline, _max_drawdown)


def hit_rate(arm: ArmResult, baseline: ArmResult | None = None) -> float:
    """Fraction of **strategy** trades that made money after costs.

    Administrative exits are excluded: no decision was made, so counting them would
    measure where the data happened to end. NaN when the fold decided no trades.
    """
    return _delta(_hit_rate(arm), baseline, _hit_rate)


def average_trade(arm: ArmResult, baseline: ArmResult | None = None) -> float:
    """Mean net PnL of a **strategy** trade, in account currency. Administrative excluded."""
    return _delta(_average_trade(arm), baseline, _average_trade)


# ── the report table ─────────────────────────────────────────────────────────

# Ordered so the table reads primary-first: what the model got right, then what it earned.
# `higher` records which way is better, so the report can render a delta's sign correctly
# without a second table of conventions to keep in step with this one.
METRICS: tuple[tuple[str, Callable[..., float], bool], ...] = (
    ("direction_accuracy", direction_accuracy, True),
    ("mae", mae, False),
    ("rmse", rmse, False),
    ("total_return", total_return, True),
    ("sharpe", sharpe, True),
    ("max_drawdown", max_drawdown, False),
    ("hit_rate", hit_rate, True),
    ("average_trade", average_trade, True),
)

# The direction column's reference is always-long, measured, not persistence.
# See `direction_accuracy` and `always_long_accuracy`.
CHANCE_REFERENCED = frozenset({"direction_accuracy"})


def summarise(results: Sequence[ArmResult], baseline: ArmResult) -> pd.DataFrame:
    """One row per arm, every value beside its delta against the baseline.

    Args:
        results: The arms to report, in the order they should appear. The baseline may be
            among them; its own deltas are then zero, which is the honest self-comparison
            rather than a blank row.
        baseline: Persistence. §7.3 requires every result to be a delta against it.

    Returns:
        A frame with ``arm``, the window and trade counts, and for each metric a ``<name>``
        column and a ``<name>_delta`` column. ``direction_accuracy_delta`` is measured
        against :func:`always_long_accuracy` rather than against the baseline arm, and
        ``direction_reference`` records the number used - persistence forecasts zero and so
        has no direction to compare against. That reference **is** the fold's realised up
        rate, by definition, so the column doubles as the per-fold up rate and the bar is
        visible moving from fold to fold rather than fixed at an assumed 0.5.

    The trade counts are columns rather than a footnote because they are what makes a NaN
    Sharpe readable: a blank cell beside ``n_strategy_trades = 2`` explains itself.
    """
    rows = []
    for arm in results:
        row: dict[str, object] = {
            "arm": arm.name,
            "n_windows": 0 if arm.actual is None else len(arm.actual),
            "n_trades": len(arm.trades),
            "n_strategy_trades": len(arm.strategy_trades),
        }
        chance = always_long_accuracy(arm)
        row["direction_reference"] = chance
        for name, metric, _ in METRICS:
            value = metric(arm)
            row[name] = value
            row[f"{name}_delta"] = (
                value - chance if name in CHANCE_REFERENCED else metric(arm, baseline)
            )
        rows.append(row)

    columns = [
        "arm",
        "n_windows",
        "n_trades",
        "n_strategy_trades",
        "direction_reference",
        *(part for name, _, _ in METRICS for part in (name, f"{name}_delta")),
    ]
    return pd.DataFrame(rows, columns=columns)


# ── primitives, exposed because the report and the tests both want them ──────


def daily_returns(equity: pd.Series) -> np.ndarray:
    """Simple period-over-period returns of the curve. One shorter than the curve."""
    values = np.asarray(equity, dtype="float64")
    if len(values) < 2:
        return np.empty(0, dtype="float64")
    if (values[:-1] == 0).any():
        raise ValueError("the equity curve touches zero; a return is undefined there")
    return values[1:] / values[:-1] - 1.0


def periods_per_year(index: pd.DatetimeIndex) -> float:
    """Bars per year, from this fold's own calendar span. NaN if it cannot be measured.

    Not a constant 252. Folds are calendar months and their bar counts differ - holidays,
    a short month, a symbol that stopped trading - so a shared constant would annualise two
    folds by the same factor when they cover different amounts of time. Measuring it also
    makes the number self-describing: a fold whose bars imply 180 per year has a gap in it,
    and that is worth seeing rather than smoothing over.
    """
    if len(index) < 2:
        return math.nan
    span = (index[-1] - index[0]).total_seconds() / 86_400.0
    if span <= 0:
        return math.nan
    return (len(index) - 1) / (span / DAYS_PER_YEAR)


def exposed_fraction(arm: ArmResult) -> float:
    """Fraction of the fold's bars on which the arm held **any** position.

    Counts **every** trade, administrative exits included: a bar on which the account held
    a position was exposed to the market regardless of how that position eventually ended.
    Overlapping trades count a bar once, so five symbols held on the same day is one
    exposed bar and not five.

    The second qualifying path for :func:`sharpe` - see :data:`MIN_EXPOSED_FRACTION`.
    """
    bars = pd.DatetimeIndex(arm.equity.index)
    if len(bars) == 0 or not arm.trades:
        return 0.0
    held = np.zeros(len(bars), dtype=bool)
    for trade in arm.trades:
        held |= (bars >= trade.entry_time) & (bars <= trade.exit_time)
    return float(held.mean())


def drawdown_curve(equity: pd.Series) -> np.ndarray:
    """Fractional fall below the running high-water mark, at every bar. Non-negative."""
    values = np.asarray(equity, dtype="float64")
    if len(values) == 0:
        return np.empty(0, dtype="float64")
    peak = np.maximum.accumulate(values)
    if (peak <= 0).any():
        raise ValueError(
            "the equity curve is non-positive; a drawdown fraction is undefined"
        )
    return (peak - values) / peak


# ── internals ────────────────────────────────────────────────────────────────


def _delta(
    value: float, baseline: ArmResult | None, of: Callable[[ArmResult], float]
) -> float:
    """The metric, or its difference from the baseline's when one is given."""
    if baseline is None:
        return value
    return value - of(baseline)


def _forecast(
    arm: ArmResult, statistic: Callable[[np.ndarray, np.ndarray], float]
) -> float:
    if arm.predicted is None or arm.actual is None:
        return math.nan
    predicted = _require_finite(arm.predicted, "predicted")
    actual = _require_finite(arm.actual, "actual")
    if predicted.shape != actual.shape:
        raise ValueError(
            f"predicted {predicted.shape} and actual {actual.shape} must be the same shape"
        )
    if predicted.size == 0:
        return math.nan
    return statistic(predicted, actual)


def _mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - actual)))


def _rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def _direction(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Sign agreement on the H-day cumulative return, over windows that made a call."""
    forecast_sign = np.sign(predicted.sum(axis=1))
    realised_sign = np.sign(actual.sum(axis=1))
    called = forecast_sign != 0
    if not called.any():
        return math.nan
    return float((forecast_sign[called] == realised_sign[called]).mean())


# One per forecast statistic, so a metric and its baseline are computed by the same code
# path. `_delta` needs a callable that turns an arm into a number, and these are it.
def _mae_of(arm: ArmResult) -> float:
    return _forecast(arm, _mae)


def _rmse_of(arm: ArmResult) -> float:
    return _forecast(arm, _rmse)


def _direction_of(arm: ArmResult) -> float:
    return _forecast(arm, _direction)


def _total_return(arm: ArmResult) -> float:
    values = np.asarray(arm.equity, dtype="float64")
    if len(values) < 2 or values[0] == 0:
        return math.nan
    return float(values[-1] / values[0] - 1.0)


def _sharpe(arm: ArmResult) -> float:
    if (
        len(arm.strategy_trades) < MIN_TRADES_FOR_SHARPE
        and exposed_fraction(arm) < MIN_EXPOSED_FRACTION
    ):
        return math.nan
    returns = daily_returns(arm.equity)
    if len(returns) < 2:
        return math.nan
    deviation = float(np.std(returns, ddof=1))
    if deviation == 0.0:
        return math.nan
    annualiser = periods_per_year(pd.DatetimeIndex(arm.equity.index))
    if not math.isfinite(annualiser):
        return math.nan
    return float(np.mean(returns)) / deviation * math.sqrt(annualiser)


def _max_drawdown(arm: ArmResult) -> float:
    curve = drawdown_curve(arm.equity)
    return math.nan if len(curve) == 0 else float(curve.max())


def _hit_rate(arm: ArmResult) -> float:
    decided = arm.strategy_trades
    if not decided:
        return math.nan
    return float(np.mean([trade.net_pnl > 0 for trade in decided]))


def _average_trade(arm: ArmResult) -> float:
    decided = arm.strategy_trades
    if not decided:
        return math.nan
    return float(np.mean([trade.net_pnl for trade in decided]))


def _require_finite(values: np.ndarray, name: str) -> np.ndarray:
    """Refuse NaN rather than propagating it.

    ``build_windows`` leaves ``y`` as NaN for a live window whose target is not known yet,
    which is correct there and meaningless here: a metric averaged over unknown targets is
    a number with no interpretation. Scoring is done on complete windows.
    """
    array = np.asarray(values, dtype="float64")
    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} carries NaN or infinity; metrics are computed on complete windows "
            "only, and build_windows leaves y as NaN where the target is not yet known"
        )
    return array


__all__ = [
    "DAYS_PER_YEAR",
    "METRICS",
    "MIN_EXPOSED_FRACTION",
    "MIN_TRADES_FOR_SHARPE",
    "ArmResult",
    "always_long_accuracy",
    "average_trade",
    "daily_returns",
    "direction_accuracy",
    "drawdown_curve",
    "exposed_fraction",
    "hit_rate",
    "mae",
    "max_drawdown",
    "periods_per_year",
    "rmse",
    "sharpe",
    "summarise",
    "total_return",
]
