"""GB-20 acceptance: the threshold grid, and what a fold does when nothing on it works.

Calibration is fitting, so it gets the fitting harness: GB-10's ``assert_fit_isolated``
holds it to the boundary it was handed, exactly as the scaler is held. A band chosen with
one glance at the test split would flatter every number the study reports and nothing else
in the pipeline would notice.

The forecaster used here is a trailing momentum average - cheap, and **causal**, which the
isolation test depends on: a forecaster that read the future would fail the harness for
a reason that has nothing to do with the code under test.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from causality import SPLIT_FRACTIONS, assert_fit_isolated

from glassbox.backtest import calibrate, metrics
from glassbox.backtest.engine import run_backtest
from glassbox.backtest.walkforward import make_folds
from glassbox.config.loader import Config, load_config
from glassbox.contracts.schemas import Forecast, Signal
from glassbox.engine.signal import Thresholds, decide_all

SYMBOL = "SYNTH"
LOOKBACK = 5
SEED = 1337  # frozen: a seed that varies turns a real failure into a flaky test


@pytest.fixture
def cfg() -> Config:
    return load_config()


def fixed_fraction(
    signal: Signal, equity: float, gross_exposure: float, cfg: Config
) -> float:
    """A stand-in sizer. Real sizing is GB-21; what matters here is that it is the same
    one every candidate is scored under."""
    del signal, gross_exposure
    return equity * cfg.risk.max_position_pct


def frame_from(returns: np.ndarray) -> pd.DataFrame:
    close = 100.0 * np.exp(np.cumsum(returns))
    index = pd.date_range("2020-01-01", periods=len(close), freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
        },
        index=index,
    )


def regime_frame(n: int = 800, block: int = 40, drift: float = 0.004) -> pd.DataFrame:
    """Alternating trend regimes: a trailing momentum rule is genuinely profitable here.

    Needed so the isolation test has teeth - if calibration stood aside on this data, both
    the perturbed and unperturbed runs would return the same empty band and the assertion
    would pass without touching the question.
    """
    rng = np.random.default_rng(SEED)
    base = np.where((np.arange(n) // block) % 2 == 0, drift, -drift)
    return frame_from(base + rng.normal(0, 0.002, n))


def whipsaw_frame(n: int = 400) -> pd.DataFrame:
    """Every up-move is followed by a larger down-move, so momentum is exactly wrong."""
    rng = np.random.default_rng(SEED)
    base = np.where(np.arange(n) % 20 < 10, 0.004, -0.006)
    return frame_from(base + rng.normal(0, 0.002, n))


def momentum_forecasts(
    frames: dict[str, pd.DataFrame], horizon: int, upto: pd.Timestamp | None = None
) -> list[Forecast]:
    """Trailing mean log return, repeated across the horizon. Reads no bar after ``t``."""
    forecasts: list[Forecast] = []
    for symbol, frame in frames.items():
        momentum = np.log(frame["close"]).diff().rolling(LOOKBACK).mean()
        for stamp, value in momentum.items():
            if upto is not None and stamp > upto:
                continue
            if np.isfinite(value):
                forecasts.append(
                    Forecast(
                        path=np.full(horizon, value, dtype="float32"),
                        symbol=symbol,
                        as_of=stamp,
                    )
                )
    return forecasts


def calibrate_on(
    frame: pd.DataFrame, split: pd.Timestamp, cfg: Config
) -> calibrate.Calibration:
    """Calibrate on everything up to ``split``, exactly as a walk-forward caller would."""
    frames = {SYMBOL: frame.loc[:split]}
    return calibrate.calibrate_thresholds(
        momentum_forecasts(frames, cfg.window.horizon),
        frames,
        fixed_fraction,
        cfg,
    )


# ── the thresholds are learned, per fold ─────────────────────────────────────


def test_calibrated_thresholds_differ_across_folds_on_real_data(
    repo_root: Path, cfg: Config
) -> None:
    """The point of calibrating per fold. One band for the whole study would be a constant
    chosen by a human, which is the leak GB-25 audits for."""
    cache = repo_root / "data_cache"
    symbols = ["AAPL", "MSFT"]
    if not all((cache / f"{symbol}.parquet").is_file() for symbol in symbols):
        pytest.skip("no cached history; run scripts to populate data_cache/")

    frames = {
        symbol: pd.read_parquet(cache / f"{symbol}.parquet")[
            ["open", "high", "low", "close"]
        ].astype("float64")
        for symbol in symbols
    }
    common = frames[symbols[0]].index
    for frame in frames.values():
        common = common.intersection(frame.index)
    folds = make_folds(common, cfg)[:5]
    assert len(folds) >= 3

    calibrations = []
    for fold in folds:
        val_bars = {
            symbol: frame.loc[fold.val[0] : fold.val[-1]]
            for symbol, frame in frames.items()
        }
        forecasts = [
            forecast
            for forecast in momentum_forecasts(frames, cfg.window.horizon)
            if fold.val[0] <= forecast.as_of <= fold.val[-1]
        ]
        calibrations.append(
            calibrate.calibrate_thresholds(forecasts, val_bars, fixed_fraction, cfg)
        )

    chosen = {
        found.thresholds.lower for found in calibrations if found.thresholds.fires
    }
    assert len(chosen) >= 2, f"every fold chose the same band: {chosen}"

    # And the grid itself moved, not only the winner - the candidates are quantiles of each
    # fold's own forecast distribution.
    first_rungs = {found.candidates[0].lower for found in calibrations}
    assert len(first_rungs) == len(calibrations)


def test_the_grid_is_scale_free(cfg: Config) -> None:
    """Ten times the forecast is the same decision, because the grid is quantiles.

    A grid of absolute levels would need rewriting for every horizon, universe and channel
    set the study compares, and a band tuned for DLinear's magnitudes would silently
    disable FITS.
    """
    frames = {SYMBOL: regime_frame()}
    forecasts = momentum_forecasts(frames, cfg.window.horizon)
    inflated = [
        Forecast(
            path=forecast.path * 10.0, symbol=forecast.symbol, as_of=forecast.as_of
        )
        for forecast in forecasts
    ]

    plain = calibrate.calibrate_thresholds(forecasts, frames, fixed_fraction, cfg)
    scaled = calibrate.calibrate_thresholds(inflated, frames, fixed_fraction, cfg)

    # rel=1e-6, not 1e-12: forecast paths are float32, so scaling one by ten and scaling
    # the quantile of the unscaled ones disagree in the last few bits. A grid of absolute
    # levels would miss by orders of magnitude, which is what this is testing for.
    assert plain.thresholds.fires
    assert scaled.thresholds.lower == pytest.approx(
        plain.thresholds.lower * 10.0, rel=1e-6
    )
    assert scaled.val_trades == plain.val_trades
    assert scaled.val_sharpe == pytest.approx(plain.val_sharpe, rel=1e-9)


# ── calibration is fitting, and fitting is held to its boundary ──────────────


def test_calibration_reads_nothing_after_its_boundary(cfg: Config) -> None:
    """GB-10's harness, applied to the band. Test data present or absent, same answer."""
    frame = regime_frame()

    for fraction in SPLIT_FRACTIONS:
        split = frame.index[int(len(frame) * fraction)]
        assert calibrate_on(frame, split, cfg).thresholds.fires, (
            "the band must actually fire at every split, or the isolation assertion "
            "compares two empty results and proves nothing"
        )

    assert_fit_isolated(
        lambda perturbed, split: calibrate_on(perturbed, split, cfg).thresholds, frame
    )


# ── the ruling: no positive validation Sharpe means no trading ───────────────


def test_a_fold_with_no_profitable_band_stands_aside(cfg: Config) -> None:
    """Validation said every band loses money, so the fold trades nothing.

    The alternative - the least-bad band - reports the maximum of fifteen losing
    candidates, selected on the same data that scored them.
    """
    frames = {SYMBOL: whipsaw_frame()}

    found = calibrate.calibrate_thresholds(
        momentum_forecasts(frames, cfg.window.horizon), frames, fixed_fraction, cfg
    )

    assert found.stood_aside is True
    assert found.trades_in_test is False
    assert found.thresholds == Thresholds.never()
    assert found.candidates, "the grid was tried; it simply lost"
    assert max(candidate.score for candidate in found.candidates) <= 0.0


def test_a_fold_that_stands_aside_takes_no_trade_at_all(cfg: Config) -> None:
    """The consequence, run through the backtester: a flat curve and an honest zero."""
    frames = {SYMBOL: whipsaw_frame()}
    forecasts = momentum_forecasts(frames, cfg.window.horizon)
    found = calibrate.calibrate_thresholds(forecasts, frames, fixed_fraction, cfg)

    result = run_backtest(
        frames, decide_all(forecasts, found.thresholds, cfg), fixed_fraction, cfg
    )

    assert result.trades == ()
    assert result.equity.iloc[-1] == pytest.approx(cfg.backtest.initial_cash, abs=1e-9)
    assert (
        metrics.total_return(metrics.ArmResult(name="aside", equity=result.equity))
        == 0.0
    )


def test_a_split_whose_forecasts_all_point_down_stands_aside(cfg: Config) -> None:
    """No positive forecast means no band could fire, and the grid is empty rather than
    clamped to something that would."""
    frames = {SYMBOL: regime_frame()}
    downward = [
        Forecast(
            path=-np.abs(forecast.path), symbol=forecast.symbol, as_of=forecast.as_of
        )
        for forecast in momentum_forecasts(frames, cfg.window.horizon)
    ]

    found = calibrate.calibrate_thresholds(downward, frames, fixed_fraction, cfg)

    assert found.stood_aside is True
    assert found.candidates == ()
    assert math.isnan(found.val_sharpe)


def test_a_thin_band_cannot_win_on_a_sharpe_from_three_trades(cfg: Config) -> None:
    """The interlock with `MIN_TRADES_FOR_SHARPE`: no evidence, no score.

    A band selective enough to trade twice would otherwise be free to post a spectacular
    ratio and take the fold.
    """
    thin = calibrate.Candidate(
        lower=0.05, upper=None, sharpe=math.nan, trades=3, total_return=0.4
    )
    ordinary = calibrate.Candidate(
        lower=0.01, upper=None, sharpe=0.3, trades=11, total_return=0.02
    )

    assert thin.score == -math.inf
    assert max((thin, ordinary), key=lambda c: (c.score, c.lower)) is ordinary


# ── it is the backtester's own number, not a proxy ───────────────────────────


def test_the_reported_sharpe_is_the_backtesters_own(cfg: Config) -> None:
    """Re-run the chosen band and the number must come back identical.

    A proxy for Sharpe - hit rate, mean return per signal, the count of passing windows -
    would tune for a quantity the study does not report and would ignore the next-open
    fill, the gap through a stop and the 6 bps round trip.
    """
    frames = {SYMBOL: regime_frame()}
    forecasts = momentum_forecasts(frames, cfg.window.horizon)

    found = calibrate.calibrate_thresholds(forecasts, frames, fixed_fraction, cfg)
    result = run_backtest(
        frames, decide_all(forecasts, found.thresholds, cfg), fixed_fraction, cfg
    )
    rerun = metrics.ArmResult(name="rerun", equity=result.equity, trades=result.trades)

    assert found.thresholds.fires
    assert metrics.sharpe(rerun) == pytest.approx(found.val_sharpe, rel=1e-12)
    assert len(rerun.strategy_trades) == found.val_trades


def test_every_candidate_on_the_grid_is_reported(cfg: Config) -> None:
    """The surface, not just the winner: a report can show what was tried."""
    frames = {SYMBOL: regime_frame()}

    found = calibrate.calibrate_thresholds(
        momentum_forecasts(frames, cfg.window.horizon), frames, fixed_fraction, cfg
    )

    assert len(found.candidates) == len(set(found.candidates))
    assert all(candidate.lower > 0.0 for candidate in found.candidates)
    assert all(
        candidate.upper is None or candidate.upper > candidate.lower
        for candidate in found.candidates
    )


def test_calibrating_on_nothing_is_refused(cfg: Config) -> None:
    """An empty split cannot choose a band, and a default would be the hardcoded number
    this whole task exists to remove."""
    with pytest.raises(ValueError, match="at least one validation forecast"):
        calibrate.calibrate_thresholds(
            [], {SYMBOL: regime_frame()}, fixed_fraction, cfg
        )


# ── the grid is the only place numbers live ──────────────────────────────────


def test_the_only_numbers_in_the_module_are_the_grid(package_root: Path) -> None:
    """GB-20's rule, checked on the syntax tree.

    Every numeric literal in ``calibrate.py`` must belong to a quantile grid at module
    level. A stray float elsewhere would be a threshold, a fee or a cap chosen here instead
    of in the config or on the grid.
    """
    source = (package_root / "backtest" / "calibrate.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    grid_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id in {"LOWER_QUANTILES", "UPPER_QUANTILES"}
            for target in node.targets
        ):
            grid_lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in {"LOWER_QUANTILES", "UPPER_QUANTILES"}
        ):
            grid_lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))

    stray = [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
        and node.lineno not in grid_lines
        and node.value not in (0, 1)
    ]

    assert (
        grid_lines
    ), "the grid definition was not found; this test would pass vacuously"
    assert not stray, f"numbers outside the grid: {stray}"
