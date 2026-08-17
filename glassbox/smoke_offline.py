"""X: the offline end-to-end smoke command - ``python -m glassbox.smoke_offline``.

Runs the full offline path in one command: data, features, model, backtest, metrics,
with the persistence baseline on the same folds. This command is Gate 1.

**It never touches the network.** The cache is checked before anything else and a missing
file is a loud failure naming what is absent, because the alternative - `load_history`
quietly fetching - would turn "the offline path works" into "the offline path works when
yfinance is up", which is not the claim Gate 1 makes.

**Both arms run the same code.** The baseline is not simulated with a flat line: it is the
persistence forecaster driven through the same train → calibrate → decide → backtest path
as the model under test. That is the point of the command. Persistence forecasts zero, so
no band can fire, so it stands aside on every fold and its curve is flat - but it is flat
because the pipeline produced nothing to trade, not because this module drew a flat line.

Two reporting rulings, both visible in the output rather than only here:

1. **A fold that stands aside is a labelled row, and it counts in the aggregate.** Its
   return is a true zero, not a missing value. Excluding it would average only the folds
   where the strategy chose to act, which is selection on the strategy's own decision.
   Sharpe is the exception and excludes itself: a flat curve has fewer than
   ``MIN_TRADES_FOR_SHARPE`` trades, so it is NaN and drops out of the Sharpe mean by
   construction. Both counts are printed - folds aggregated, and folds in the Sharpe mean -
   so a reader can see which number rests on how many folds.
2. **The direction reference is printed beside direction accuracy**, in the column
   immediately to its right in the per-fold table and again in the aggregate block, with a
   legend naming it. A direction number without its bar is unreadable: 0.51 is above a coin
   flip and below always-long, and the difference is the whole finding.

Sizing is a fixed fraction of equity from ``risk.max_position_pct``. **GB-21 replaces it**
- this module holds the interface's simplest satisfying implementation so the smoke command
can exist before the risk layer, and the engine enforces the cap either way.

Implemented in GB-24.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from glassbox.backtest import metrics
from glassbox.backtest.calibrate import Calibration, calibrate_thresholds
from glassbox.backtest.engine import run_backtest
from glassbox.backtest.walkforward import Fold, make_folds
from glassbox.config.loader import Config, load_config
from glassbox.contracts.schemas import Forecast, Signal, WindowBatch
from glassbox.data.historical import load_history
from glassbox.engine.signal import decide_all
from glassbox.features.builder import build_feature_frame, build_windows
from glassbox.model import train as trainer

BASELINE = "persistence"
PRICE_COLUMNS = ["open", "high", "low", "close"]

# Printed for every arm, in this order. `direction_reference` sits immediately after
# `direction` so the two cannot be read apart - see ruling 2 in the module docstring.
FOLD_COLUMNS = (
    "fold",
    "arm",
    "aside",
    "trades",
    "direction",
    "dir_ref",
    "dir_delta",
    "mae",
    "mae_delta",
    "total_return",
    "sharpe",
    "max_dd",
    "hit_rate",
)


class SmokeError(RuntimeError):
    """Something the run cannot proceed without. Reported without a traceback."""


@dataclass(frozen=True, eq=False)
class ArmRun:
    """One arm on one fold: what it forecast, what it traded, what it was allowed to."""

    fold: int
    name: str
    result: metrics.ArmResult
    calibration: Calibration
    seconds: float


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code rather than raising."""
    args = _parse_args(argv)
    try:
        table, summary = run(
            load_config(), model=args.model, n_folds=args.folds, log=print
        )
    except SmokeError as failure:
        print(f"smoke_offline: {failure}", file=sys.stderr)
        return 2

    print()
    print(table.to_string(index=False))
    print()
    print(summary)
    return 0


def run(
    cfg: Config,
    model: str = "dlinear",
    n_folds: int = 1,
    log=lambda message: None,
) -> tuple[pd.DataFrame, str]:
    """The whole offline path, from the cache to a metrics table.

    Args:
        cfg: Resolved configuration.
        model: The arm to run. The persistence baseline is always run beside it, on the
            same folds, through the same code.
        n_folds: How many folds, taken from the **start** of the fold list in chronological
            order, so ``--folds 1`` names the same fold on every run.
        log: Progress sink. ``print`` from the CLI, silent from tests.

    Returns:
        ``(per-fold table, aggregate block)``.

    Raises:
        SmokeError: the cache is incomplete, the arm is unknown, or the data yields no
            fold. Each names what is wrong and what to do about it.
    """
    if model not in {BASELINE, "dlinear"}:
        raise SmokeError(f"unknown model {model!r}; choose persistence or dlinear")
    if n_folds < 1:
        raise SmokeError(f"--folds must be at least 1, got {n_folds}")

    started = time.perf_counter()
    bars = load_cached_bars(cfg)
    log(f"cache: {len(bars)} symbols, {min(len(f) for f in bars.values())}+ bars each")

    frames = {symbol: build_feature_frame(frame, cfg) for symbol, frame in bars.items()}
    folds = make_folds(_common_index(frames), cfg)[:n_folds]
    if not folds:
        raise SmokeError(
            "no complete walk-forward fold fits the cached history; the cache is too "
            "short for the configured train/val/test months"
        )
    log(f"folds: {len(folds)} of {n_folds} requested, first is fold {folds[0].number}")

    runs: list[ArmRun] = []
    for fold in folds:
        for arm in dict.fromkeys((model, BASELINE)):  # dedupe, keep order
            done = _run_arm(arm, fold, frames, bars, cfg)
            runs.append(done)
            log(
                f"fold {fold.number} {arm}: {done.seconds:.1f}s, "
                f"{len(done.result.strategy_trades)} trades"
                f"{', stood aside' if done.calibration.stood_aside else ''}"
            )

    table = fold_table(runs)
    summary = aggregate(runs, seconds=time.perf_counter() - started)
    return table, summary


def load_cached_bars(cfg: Config) -> dict[str, pd.DataFrame]:
    """Read every symbol from the parquet cache. **Never fetches.**

    ``data.historical.load_history`` downloads a symbol whose cache file is absent, which
    is right for that module and wrong here: this command's claim is that the offline path
    runs offline, and a silent fetch would leave that claim untested on exactly the machine
    where it matters. So the files are checked **first** and the loader is called only once
    every one of them is known to exist - which keeps a single loading path, and with it the
    provenance and schema checks `load_history` performs on a cached frame.

    Raises:
        SmokeError: naming every missing file and how to create it.
    """
    cache = Path(cfg.data.cache_dir)
    missing = [
        symbol for symbol in cfg.universe if not (cache / f"{symbol}.parquet").is_file()
    ]
    if missing:
        raise SmokeError(
            f"no cached history for {', '.join(missing)} in {cache}/. This command is "
            "offline by design and will not download. Populate the cache first with: "
            'python -c "from glassbox.config.loader import load_config; '
            "from glassbox.data.historical import load_history; "
            'cfg = load_config(); load_history(list(cfg.universe), cfg)"'
        )
    return load_history(sorted(cfg.universe), cfg)


def fold_table(runs: Sequence[ArmRun]) -> pd.DataFrame:
    """One row per arm per fold, in fold order, with the direction reference beside it."""
    baselines = {run.fold: run for run in runs if run.name == BASELINE}
    rows = []
    for run in runs:
        baseline = baselines[run.fold].result
        rows.append(
            {
                "fold": run.fold,
                "arm": run.name,
                "aside": "yes" if run.calibration.stood_aside else "",
                "trades": len(run.result.strategy_trades),
                "direction": metrics.direction_accuracy(run.result),
                "dir_ref": metrics.always_long_accuracy(run.result),
                "dir_delta": metrics.direction_accuracy(run.result)
                - metrics.always_long_accuracy(run.result),
                "mae": metrics.mae(run.result),
                "mae_delta": metrics.mae(run.result, baseline),
                "total_return": metrics.total_return(run.result),
                "sharpe": metrics.sharpe(run.result),
                "max_dd": metrics.max_drawdown(run.result),
                "hit_rate": metrics.hit_rate(run.result),
            }
        )
    return pd.DataFrame(rows, columns=list(FOLD_COLUMNS)).round(4)


def aggregate(runs: Sequence[ArmRun], seconds: float) -> str:
    """The block under the table: what each arm did across every fold it ran.

    Every mean states how many folds it rests on. The return mean includes folds that stood
    aside, because a fold that traded nothing returned exactly zero and that is a result;
    the Sharpe mean cannot include them, because a flat curve has no Sharpe at all. Saying
    which is which is the difference between an aggregate and an average of whatever
    happened to be a number.
    """
    lines = []
    for name in dict.fromkeys(run.name for run in runs):
        arm_runs = [run for run in runs if run.name == name]
        returns = np.array([metrics.total_return(r.result) for r in arm_runs])
        sharpes = np.array([metrics.sharpe(r.result) for r in arm_runs])
        direction = np.array([metrics.direction_accuracy(r.result) for r in arm_runs])
        reference = np.array([metrics.always_long_accuracy(r.result) for r in arm_runs])
        aside = sum(1 for r in arm_runs if r.calibration.stood_aside)
        defined = int(np.isfinite(sharpes).sum())

        lines.append(f"{name} over {len(arm_runs)} fold(s)")
        lines.append(
            f"  stood aside          : {aside}/{len(arm_runs)} "
            f"(counted in the return mean, absent from the Sharpe mean)"
        )
        lines.append(
            f"  total return         : mean {_fmt(_mean(returns)):>9} over "
            f"{len(arm_runs)} folds, positive in {int((returns > 0).sum())}"
        )
        lines.append(
            f"  sharpe               : mean {_fmt(_mean(sharpes)):>9} over "
            f"{defined} fold(s) where it is defined"
        )
        lines.append(
            f"  direction accuracy   : mean {_fmt(_mean(direction)):>9}  "
            f"vs always-long bar {_fmt(_mean(reference))}  "
            f"(beats it in {int((direction > reference).sum())}/{len(arm_runs)})"
        )
        lines.append("")

    lines.append(
        "dir_ref is the always-long bar: the fold's own realised up rate, which is what a "
        "long-only system"
    )
    lines.append(
        "could have scored by calling up every time. It is NOT 0.5 and it is not "
        "persistence, which forecasts"
    )
    lines.append("zero and therefore has no direction at all (spec 7.2).")
    lines.append(f"wall time: {seconds:.1f}s")
    return "\n".join(lines)


def size_position(
    signal: Signal, equity: float, gross_exposure: float, cfg: Config
) -> float:
    """A fixed fraction of equity per position. **Placeholder for GB-21.**

    Satisfies ``backtest.engine.PositionSizer`` and reads its fraction from
    ``risk.max_position_pct``, so there is no constant here to go stale. It does not
    implement the gross-exposure cap or the ranking that GB-21 owns; the engine's own
    validation still refuses anything the account cannot pay for.
    """
    del signal, gross_exposure
    return equity * cfg.risk.max_position_pct


def _run_arm(
    name: str,
    fold: Fold,
    frames: dict[str, pd.DataFrame],
    bars: dict[str, pd.DataFrame],
    cfg: Config,
) -> ArmRun:
    """Train, calibrate, forecast and backtest one arm on one fold."""
    started = time.perf_counter()
    arm_cfg = replace(cfg, model=replace(cfg.model, active=name))

    run = trainer.train(frames, arm_cfg, fold.train, fold.val, fold.test)

    val_batch = _windows(frames, run, arm_cfg, fold.val)
    calibration = calibrate_thresholds(
        _forecasts(val_batch, run.model.predict(val_batch.X)),
        _bars_between(bars, fold.val[0], fold.val[-1]),
        size_position,
        arm_cfg,
    )

    test_batch = _windows(frames, run, arm_cfg, fold.test)
    predicted = run.model.predict(test_batch.X)
    result = run_backtest(
        _bars_between(bars, fold.test[0], fold.test[-1]),
        decide_all(_forecasts(test_batch, predicted), calibration.thresholds, arm_cfg),
        size_position,
        arm_cfg,
    )

    return ArmRun(
        fold=fold.number,
        name=name,
        result=metrics.ArmResult(
            name=name,
            equity=result.equity,
            trades=result.trades,
            predicted=predicted,
            actual=test_batch.y,
        ),
        calibration=calibration,
        seconds=time.perf_counter() - started,
    )


def _windows(
    frames: dict[str, pd.DataFrame],
    run: trainer.TrainingRun,
    cfg: Config,
    index: pd.DatetimeIndex,
) -> WindowBatch:
    """Every symbol's windows ending in ``index``, pooled, each scaled by its own stats.

    The statistics come from the training run, so a validation or test window is normalised
    with numbers fitted on training rows only - the property GB-15 made structural and
    GB-25 audits.
    """
    return WindowBatch.concat(
        [
            trainer.select_windows(
                build_windows(frames[symbol], cfg, symbol, stats=run.stats[symbol]),
                index,
            )
            for symbol in sorted(frames)
        ]
    )


def _forecasts(batch: WindowBatch, predicted: np.ndarray) -> list[Forecast]:
    return [
        Forecast(path=predicted[row], symbol=batch.symbols[row], as_of=timestamp)
        for row, timestamp in enumerate(batch.timestamps)
    ]


def _bars_between(
    bars: dict[str, pd.DataFrame], first: pd.Timestamp, last: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """Price frames sliced to one split. The backtester sees no bar outside it."""
    return {
        symbol: frame.loc[first:last][PRICE_COLUMNS].astype("float64")
        for symbol, frame in bars.items()
    }


def _common_index(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Timestamps every symbol has, so a fold means the same thing for all of them."""
    common: pd.Index | None = None
    for frame in frames.values():
        common = frame.index if common is None else common.intersection(frame.index)
    if common is None or common.empty:
        raise SmokeError("the cached symbols share no common trading day")
    return pd.DatetimeIndex(common)


def _mean(values: np.ndarray) -> float:
    """Mean of the defined entries; NaN when there are none.

    ``np.nanmean`` of an all-NaN array is NaN *with a warning*, and persistence produces
    exactly that - it has no Sharpe and no direction on any fold. The absence is the
    baseline behaving correctly, not a numerical accident, so it is handled rather than
    warned about.
    """
    defined = values[np.isfinite(values)]
    return math.nan if defined.size == 0 else float(defined.mean())


def _fmt(value: float) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:+.4f}"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m glassbox.smoke_offline",
        description=(
            "Run the complete offline path - cached data, features, folds, training, "
            "threshold calibration, backtest - and print a metrics table beside the "
            "persistence baseline. Never touches the network."
        ),
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=1,
        help="how many folds to run, from the start of the fold list (default: 1)",
    )
    parser.add_argument(
        "--model",
        choices=(BASELINE, "dlinear"),
        default="dlinear",
        help="which arm to run; persistence is always run beside it (default: dlinear)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI, not by tests
    raise SystemExit(main())
