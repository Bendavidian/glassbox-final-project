"""X: per-fold threshold calibration - a coarse grid, scored by the real backtester.

The trend band that ``engine/signal.py`` applies is chosen here, on the **validation**
split of each fold, and never by a human reading test results. ``signal.min_trend`` and
``signal.max_trend`` are ``null`` in the config file precisely because they are learned.

**Why this is not inside ``engine/signal.py``, where GB-20 asked for it.** The ruling that
calibration must run the actual backtester rather than a proxy for Sharpe decides where the
code can live. ``backtest`` sits above ``engine`` in the layer contract, so a
``calibrate_thresholds`` in the engine layer would have to import the harness - breaking
both the layers contract and the standing rule that the live path never imports the
validation harness (``live_loop`` imports ``signal``, so it would inherit the dependency).
The alternative, injecting the backtester as a callable, hides the same dependency behind
an argument and buys nothing. So the decision logic stays at L4 and the calibration that
scores it sits at the harness layer, which is where every other "run the whole thing and
measure" module already lives. Spec 3.4 carries the module; nothing else about GB-20 moved.

**Why Sharpe itself, not a proxy.** A proxy - hit rate on the raw forecast, mean return per
signal, the count of windows that pass - tunes for a quantity the study does not report,
and each of them ignores something the backtester models: the next-open fill, the gap
through a stop, the 6.0 bps round trip, the capital a position ties up while it sits. A
band chosen on a proxy would be optimal for a system this project is not building.

**Ruling: a fold whose best candidate has no positive validation Sharpe stands aside.**
See :func:`calibrate_thresholds`.

Implemented in GB-20.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from glassbox.backtest import metrics
from glassbox.backtest.engine import PositionSizer, run_backtest
from glassbox.config.loader import Config
from glassbox.contracts.schemas import Forecast
from glassbox.engine.signal import Thresholds, decide_all, strengths_of

# The grid, and the only numbers in this module. They are **quantiles of the validation
# split's own forecast distribution**, not return magnitudes: a band of 0.004 means nothing
# without knowing what that fold's model produced, and a grid of absolute levels would need
# rewriting for every horizon, universe and channel set the study compares. Quantiles make
# the grid scale-free, so the same five levels are meaningful for DLinear, FITS and any
# later arm.
#
# Coarse on purpose. Five entry levels and three ceilings is 15 backtests per fold; a fine
# grid would fit the validation split's noise and hand the test split a band that was
# chosen by luck.
LOWER_QUANTILES = (0.50, 0.60, 0.70, 0.80, 0.90)

# `None` means "no ceiling", and it is on the grid rather than assumed: whether an upper
# bound helps is a question the fold answers, not one this module decides.
UPPER_QUANTILES: tuple[float | None, ...] = (0.95, 0.99, None)


@dataclass(frozen=True)
class Candidate:
    """One band on the grid and what it earned on the validation split."""

    lower: float
    upper: float | None
    sharpe: float
    trades: int
    total_return: float

    @property
    def score(self) -> float:
        """Sharpe, with NaN read as the worst possible score rather than skipped.

        A band that produced fewer than ``metrics.MIN_TRADES_FOR_SHARPE`` round trips gets
        NaN from :func:`metrics.sharpe`, and that interlock is doing real work here: a band
        so selective that validation saw three trades has no evidence behind it, and must
        not win the grid on a number computed from three observations.
        """
        return -math.inf if math.isnan(self.sharpe) else self.sharpe


@dataclass(frozen=True)
class Calibration:
    """What calibration chose for one fold, and the evidence it chose from."""

    thresholds: Thresholds
    val_sharpe: float
    val_trades: int
    stood_aside: bool
    candidates: tuple[Candidate, ...]

    @property
    def trades_in_test(self) -> bool:
        """False when the fold stands aside, so a report can count those folds."""
        return self.thresholds.fires


def calibrate_thresholds(
    val_forecasts: Sequence[Forecast],
    val_bars: dict[str, pd.DataFrame],
    sizer: PositionSizer,
    cfg: Config,
) -> Calibration:
    """Choose this fold's trend band by running the backtester on the validation split.

    Args:
        val_forecasts: Every forecast whose window **ends inside the validation split**,
            for every symbol. Built from the same checkpoint the test split will use.
        val_bars: ``{symbol: bar frame}`` restricted to the validation range. **The caller
            slices, and must not include a single test bar** - calibration is fitting, and
            ``tests/backtest/test_calibrate.py`` holds it to GB-10's ``assert_fit_isolated``.
        sizer: The position sizer. The **same** one the test run will use: a band chosen
            under one capital rule is not the band another rule would have chosen.
        cfg: Resolved configuration.

    Returns:
        A :class:`Calibration` carrying the chosen band, its validation Sharpe and trade
        count, and every candidate tried - so the report can show the surface rather than
        assert the winner.

    Raises:
        ValueError: no forecasts were supplied, so there is nothing to calibrate on.

    **A fold whose best candidate has no positive validation Sharpe stands aside.** It
    returns :meth:`Thresholds.never`, trades nothing on the test split, and reports a flat
    curve with ``stood_aside=True``. The alternative - taking the least-bad band - would
    put money behind a rule that validation had just said loses it, and would report the
    maximum of fifteen losing candidates selected on the same data that scored them, which
    is an overfitting procedure with a positive number at the end of it. Standing aside
    costs the study nothing it is entitled to: the forecast metrics are unaffected, the
    fold's return is a true zero rather than an estimated loss, and "validation rejected
    every band" is a result worth reporting. GB-57 must state how many folds did it.
    """
    if len(val_forecasts) == 0:
        raise ValueError(
            "calibrate_thresholds needs at least one validation forecast; an empty split "
            "cannot choose a band, and defaulting to one would be the hardcoding this "
            "function exists to prevent"
        )

    candidates = tuple(
        _score(lower, upper, val_forecasts, val_bars, sizer, cfg)
        for lower, upper in _grid(strengths_of(val_forecasts))
    )
    if not candidates:
        # Every forecast on the split pointed down, so no band could ever fire.
        return Calibration(
            thresholds=Thresholds.never(),
            val_sharpe=math.nan,
            val_trades=0,
            stood_aside=True,
            candidates=(),
        )

    # Ties break toward the **more selective** band: same evidence, less exposure to
    # frictions and fewer chances for the test split to differ from validation.
    best = max(candidates, key=lambda candidate: (candidate.score, candidate.lower))
    stood_aside = not (best.score > 0.0)

    return Calibration(
        thresholds=(
            Thresholds.never()
            if stood_aside
            else Thresholds(lower=best.lower, upper=best.upper)
        ),
        val_sharpe=best.sharpe,
        val_trades=best.trades,
        stood_aside=stood_aside,
        candidates=candidates,
    )


def _grid(strengths: np.ndarray) -> tuple[tuple[float, float | None], ...]:
    """The (lower, upper) pairs to try, from the validation forecast distribution.

    Quantiles are taken over the **positive** strengths only. A quantile of everything
    would put the median below zero on a fold whose forecasts mostly pointed down, and a
    non-positive entry threshold is refused by :class:`Thresholds` - correctly, since it
    would mean entering on a forecast of no move.

    Pairs where the ceiling would sit at or below the floor are dropped rather than
    clamped, and duplicates - quantiles collide on a coarse distribution - are dropped too,
    so a repeated band cannot win the grid twice.
    """
    positive = strengths[strengths > 0.0]
    if positive.size == 0:
        return ()

    pairs: list[tuple[float, float | None]] = []
    for lower_quantile in LOWER_QUANTILES:
        lower = float(np.quantile(positive, lower_quantile))
        if not (lower > 0.0):
            continue
        for upper_quantile in UPPER_QUANTILES:
            upper = (
                None
                if upper_quantile is None
                else float(np.quantile(positive, upper_quantile))
            )
            if upper is not None and upper <= lower:
                continue
            if (lower, upper) not in pairs:
                pairs.append((lower, upper))
    return tuple(pairs)


def _score(
    lower: float,
    upper: float | None,
    val_forecasts: Sequence[Forecast],
    val_bars: dict[str, pd.DataFrame],
    sizer: PositionSizer,
    cfg: Config,
) -> Candidate:
    """Run the real backtester on the validation split with this band."""
    thresholds = Thresholds(lower=lower, upper=upper)
    result = run_backtest(
        val_bars, decide_all(val_forecasts, thresholds, cfg), sizer, cfg
    )
    arm = metrics.ArmResult(
        name=f"lower={lower:.6g},upper={upper}",
        equity=result.equity,
        trades=result.trades,
    )
    return Candidate(
        lower=lower,
        upper=upper,
        sharpe=metrics.sharpe(arm),
        trades=len(arm.strategy_trades),
        total_return=metrics.total_return(arm),
    )


__all__ = [
    "LOWER_QUANTILES",
    "UPPER_QUANTILES",
    "Calibration",
    "Candidate",
    "calibrate_thresholds",
]
