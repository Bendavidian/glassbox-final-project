"""X: the offline end-to-end smoke command - ``python -m glassbox.smoke_offline``.

Runs the full offline path in one command: data, features, model, backtest, metrics,
with the persistence baseline on the same folds. This command is Gate 1.

**It never touches the network.** The cache is checked before anything else and a missing
file is a loud failure naming what is absent, because the alternative - `load_history`
quietly fetching - would turn "the offline path works" into "the offline path works when
yfinance is up", which is not the claim Gate 1 makes.

**Every arm runs the same code.** No curve here is drawn by the reporting layer. The
persistence baseline is the persistence forecaster driven through the same train →
calibrate → decide → backtest path as the model under test; it forecasts zero, so no band
can fire, so it stands aside on every fold and its curve is flat - flat because the
pipeline produced nothing to trade. **Buy-and-hold goes through the same backtester too**,
so it pays entry slippage, both fees and the next-open fill rule, and a reader cannot
argue the comparison was arranged in the strategy's favour.

**Three references, because they answer three different questions** (spec §7.3):

- **MAE and RMSE against persistence.** Forecast skill against the random walk, which is
  what persistence is a hard baseline for.
- **Direction accuracy against always-long.** Persistence forecasts zero and expresses no
  direction at all, so it cannot be the reference for this column.
- **Return, Sharpe and drawdown against buy-and-hold**, with persistence shown beside it as
  the do-nothing floor. Persistence never trades, so comparing a trading strategy to it
  measures only that the strategy traded. Against cash, +0.44% a fold looks like a result;
  against an equal-weight hold of the same five symbols over 2022-2026 it is a different
  statement, and the honest table lets a reader see which.

Each column header names its own reference - ``mae_vs_pers``, ``dir_vs_long``,
``ret_vs_bh``, ``sharpe_vs_bh`` - so three baselines cannot look like three chances to find
a flattering one.

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

Sizing is ``engine.risk.position_sizer`` (GB-21) - the per-position cap, the gross-exposure
cap and the cash check, the same arithmetic the live executor will use. Buy-and-hold is the
one exception and is sized equal-weight, because it is a reference portfolio rather than a
strategy run under this project's risk rules; the consequence is stated where it is sized.

Implemented in GB-24; the buy-and-hold arm and the real sizer landed with GB-21.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from glassbox.backtest import metrics
from glassbox.backtest.calibrate import Calibration, calibrate_thresholds
from glassbox.backtest.engine import run_backtest
from glassbox.backtest.walkforward import Fold, make_folds
from glassbox.config.loader import Config, config_hash, load_config
from glassbox.contracts.schemas import ChannelStats, Forecast, Signal, WindowBatch
from glassbox.data.historical import load_history
from glassbox.engine import risk
from glassbox.engine.signal import ENTER_LONG, Thresholds, decide_all
from glassbox.explain.channel import (
    attribute,
    cancellation,
    cancellation_is_meaningful,
)
from glassbox.features.builder import (
    build_feature_frame,
    build_windows,
    restore_targets,
)
from glassbox.model import ALL_FORECASTERS
from glassbox.model import train as trainer

BASELINE = "persistence"
BUY_AND_HOLD = "buy_and_hold"
PRICE_COLUMNS = ["open", "high", "low", "close"]

# Printed for every arm, in this order. `dir_ref` sits immediately after `direction` so the
# two cannot be read apart - see ruling 2 in the module docstring - and every delta column
# names the reference it is measured against, because there are three of them.
FOLD_COLUMNS = (
    "fold",
    "arm",
    "aside",
    "trades",
    "direction",
    "dir_ref",
    "dir_vs_long",
    "mae",
    "flatness",
    "mae_vs_pers",
    "total_return",
    "ret_vs_bh",
    "sharpe",
    "sharpe_vs_bh",
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
    cancellation: float = math.nan
    """Mean surviving fraction of the gross channel view over the test windows (GB-30).

    Carried here rather than recomputed by GB-49, because it falls out of the explanation
    layer for free and a second computation is a second thing that can disagree. NaN for
    an arm that makes no forecast.
    """
    forecasts: bool = True
    """False for buy-and-hold, which makes a directional call and no magnitude forecast.

    Its MAE cell is then empty rather than filled with the error of a forecast it never
    made - the same treatment persistence's direction column gets, and for the same reason:
    a metric an arm does not produce is reported as absent, not as a number.
    """


def _for_model(cfg: Config, model: str | None) -> Config:
    """The config with ``model.active`` set, or unchanged when no arm was named.

    ``None`` means *whatever the config says*, which is what ``--model``'s default
    documents. Anything else is the caller overriding it, and the override has to reach
    the checkpoint or the flag is a note.
    """
    if model is None or model == cfg.model.active:
        return cfg
    return replace(cfg, model=replace(cfg.model, active=model))


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code rather than raising."""
    args = _parse_args(argv)
    try:
        # `--model` reaches every path, not only the smoke run. It did not until
        # 28 Aug 2026: both prepare flags called `load_config()` directly, so
        # `--model fits --prepare-replay 13 DIR` parsed the flag, printed no warning and
        # wrote a **DLinear** checkpoint. Found while preparing GB-63's spectral
        # screenshot, which came back with no frequency decomposition because the arm
        # that produces one was never trained. Same family as GB-24's argparse defect: a
        # flag that exists, is documented in `--help`, and is silently dropped on the
        # path that uses it.
        cfg = _for_model(load_config(), args.model)
        if args.prepare_live:
            prepare_live(cfg, args.prepare_live, log=print)
            return 0
        if args.prepare_replay:
            fold, directory = args.prepare_replay
            prepare_replay(cfg, int(fold), directory, log=print)
            return 0
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
    model: str | None = None,
    n_folds: int = 1,
    log=lambda message: None,
) -> tuple[pd.DataFrame, str]:
    """The whole offline path, from the cache to a metrics table.

    Args:
        cfg: Resolved configuration.
        model: The arm to run, or ``None`` to run the one ``model.active`` names. The
            persistence baseline is always run beside it, on the same folds, through the
            same code.

            **The default is the configuration, not a literal** (GB-44). A hardcoded
            ``"dlinear"`` here meant that setting ``model.active: fits`` changed the live
            checkpoint - ``prepare_live`` reads the config - and changed *nothing* about
            the study numbers this function produces, silently. The switch spec 4.3
            promises has to reach the runner that produces the results, or it is not the
            switch.
        n_folds: How many folds, taken from the **start** of the fold list in chronological
            order, so ``--folds 1`` names the same fold on every run.
        log: Progress sink. ``print`` from the CLI, silent from tests.

    Returns:
        ``(per-fold table, aggregate block)``.

    Raises:
        SmokeError: the cache is incomplete, the arm is unknown, or the data yields no
            fold. Each names what is wrong and what to do about it.
    """
    model = cfg.model.active if model is None else model
    if model not in ALL_FORECASTERS:
        raise SmokeError(
            f"unknown model {model!r}; the registry holds {sorted(ALL_FORECASTERS)}"
        )
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
            done = run_arm(arm, fold, frames, bars, cfg)
            runs.append(done)
            log(
                f"fold {fold.number} {arm}: {done.seconds:.1f}s, "
                f"{len(done.result.strategy_trades)} trades"
                f"{', stood aside' if done.calibration.stood_aside else ''}"
            )
        held = run_buy_and_hold(fold, frames, bars, cfg)
        runs.append(held)
        log(f"fold {fold.number} {BUY_AND_HOLD}: {held.seconds:.1f}s")

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


def prepare_live(cfg: Config, directory: str | Path, log=lambda m: None) -> Path:
    """Write the checkpoint and the calibrated band the live loop loads. **GB-26.**

    `live_loop.py` may not import this layer - the forbidden-import contract keeps the
    validation harness out of the live path - so it cannot train a model or calibrate a
    band, only load them. This is the other side of that: the harness's own entry point,
    which is already exempt, produces both artefacts and puts them where the loop looks.

    **The most recent complete fold, not a fresh split.** Its training range ends where its
    validation range begins, and that validation range is the newest data the walk-forward
    protocol allows a threshold to be chosen on. Fitting on everything up to yesterday and
    calibrating on the same rows would be the leakage the whole project is built to prevent,
    arriving through the back door of "but it is live now".

    A fold that stands aside writes a band that says so rather than a number: the live loop
    then runs, records and explains, and trades nothing - the same ruling GB-20 made for a
    backtest fold, applied to a session.

    Returns:
        The directory written.
    """
    target = Path(directory)
    bars = load_cached_bars(cfg)
    frames = {symbol: build_feature_frame(frame, cfg) for symbol, frame in bars.items()}
    folds = make_folds(_common_index(frames), cfg)
    if not folds:
        raise SmokeError(
            "no complete walk-forward fold fits the cached history, so there is nothing to "
            "train the live model on"
        )

    fold = folds[-1]
    log(
        f"preparing from fold {fold.number} of {len(folds)}: train "
        f"{fold.train[0]:%Y-%m-%d}..{fold.train[-1]:%Y-%m-%d}, val "
        f"{fold.val[0]:%Y-%m-%d}..{fold.val[-1]:%Y-%m-%d}"
    )

    run = trainer.train(frames, cfg, fold.train, fold.val, fold.test)
    val_batch = _windows(frames, run, cfg, fold.val)
    calibration = calibrate_thresholds(
        _forecasts(val_batch, run.model.predict(val_batch.X), run.stats),
        _bars_between(bars, fold.val[0], fold.val[-1]),
        risk.position_sizer,
        cfg,
    )

    trainer.save_checkpoint(target / "checkpoint", run, cfg, fold.val, fold.test)
    band = {
        "stood_aside": calibration.stood_aside,
        "lower": None if calibration.stood_aside else calibration.thresholds.lower,
        "upper": None if calibration.stood_aside else calibration.thresholds.upper,
        "val_sharpe": (
            None if math.isnan(calibration.val_sharpe) else calibration.val_sharpe
        ),
        "val_trades": calibration.val_trades,
        "fold": fold.number,
        "val_start": fold.val[0].isoformat(),
        "val_end": fold.val[-1].isoformat(),
        "config_hash": config_hash(cfg),
    }
    (target / "thresholds.json").write_text(
        json.dumps(band, indent=2), encoding="utf-8"
    )
    log(
        "band: "
        + (
            "stands aside - validation found no candidate with a positive Sharpe"
            if calibration.stood_aside
            else f"lower={calibration.thresholds.lower:.6f} "
            f"upper={calibration.thresholds.upper} "
            f"(val sharpe {calibration.val_sharpe:.3f}, {calibration.val_trades} trades)"
        )
    )
    write_reliability(cfg, target, log=log)
    log(f"written to {target}")
    return target


def prepare_replay(
    cfg: Config, fold_number: int, directory: str | Path, log=lambda m: None
) -> Path:
    """Write what GB-38's replay loads: a fold's checkpoint, its band and its test range.

    **The band comes from the fold's own validation split**, exactly as the fold's
    backtest used it. Calibrating on the test range replay is about to walk would be the
    leakage the project exists to prevent, dressed up as a demo.

    ``replay.py`` may not import this layer, so this is where the training and the
    calibration happen — the same division GB-26 made, for the same reason.
    """
    target = Path(directory)
    bars = load_cached_bars(cfg)
    frames = {symbol: build_feature_frame(frame, cfg) for symbol, frame in bars.items()}
    folds = {fold.number: fold for fold in make_folds(_common_index(frames), cfg)}
    if fold_number not in folds:
        raise SmokeError(
            f"fold {fold_number} is not in the grid; it holds {sorted(folds)}"
        )

    fold = folds[fold_number]
    run = trainer.train(frames, cfg, fold.train, fold.val, fold.test)
    val_batch = _windows(frames, run, cfg, fold.val)
    calibration = calibrate_thresholds(
        _forecasts(val_batch, run.model.predict(val_batch.X), run.stats),
        _bars_between(bars, fold.val[0], fold.val[-1]),
        risk.position_sizer,
        cfg,
    )
    if calibration.stood_aside:
        raise SmokeError(
            f"fold {fold_number} stands aside, so replaying it would demonstrate nothing "
            "firing. Choose a fold whose band fired"
        )

    trainer.save_checkpoint(target / "checkpoint", run, cfg, fold.val, fold.test)
    (target / "thresholds.json").write_text(
        json.dumps(
            {
                "stood_aside": False,
                "lower": calibration.thresholds.lower,
                "upper": calibration.thresholds.upper,
                "val_sharpe": calibration.val_sharpe,
                "val_trades": calibration.val_trades,
                "fold": fold.number,
                "config_hash": config_hash(cfg),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (target / "replay.json").write_text(
        json.dumps(
            {
                "fold": fold.number,
                "test_start": fold.test[0].isoformat(),
                "test_end": fold.test[-1].isoformat(),
                "test_index": [stamp.isoformat() for stamp in fold.test],
                "val_start": fold.val[0].isoformat(),
                "val_end": fold.val[-1].isoformat(),
                "config_hash": config_hash(cfg),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(
        f"fold {fold.number}: test {fold.test[0]:%Y-%m-%d}..{fold.test[-1]:%Y-%m-%d}, "
        f"band lower={calibration.thresholds.lower:.6f} "
        f"upper={calibration.thresholds.upper}, {len(fold.test)} bars"
    )
    log(f"written to {target}")
    return target


def write_reliability(cfg: Config, directory: str | Path, log=lambda m: None) -> Path:
    """Measure the model's directional track record and write it beside the checkpoint.

    **GB-34's reliability panel reads this file.** The numbers are measured here rather
    than written into the dashboard, for the reason every number in this project is
    measured somewhere it can go stale loudly: a hardcoded track record is right on the day
    it is typed and wrong from then on, and the panel exists precisely so a viewer is not
    misled about how well the decider decides.

    Measured over every fold the grid holds, at the anchor the config produces. GB-49's
    three-anchor sweep is the fuller answer and this is the one the deployed model can
    state about itself; the file records the fold count so the panel can say what it rests
    on.
    """
    target = Path(directory)
    table, _ = run(cfg, model=cfg.model.active, n_folds=cfg.walkforward.max_folds)
    arm = table[table["arm"] == cfg.model.active]
    measured = {
        "model": cfg.model.active,
        "direction": float(arm["direction"].mean()),
        "always_long": float(arm["dir_ref"].mean()),
        "beats_bar": int((arm["direction"] > arm["dir_ref"]).sum()),
        "folds": len(arm),
        "measured_on": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
        "config_hash": config_hash(cfg),
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "reliability.json").write_text(
        json.dumps(measured, indent=2), encoding="utf-8"
    )
    log(
        f"reliability: direction {measured['direction']:.4f} against an always-long bar "
        f"of {measured['always_long']:.4f}, beating it in {measured['beats_bar']} of "
        f"{measured['folds']} folds"
    )
    return target / "reliability.json"


def fold_table(runs: Sequence[ArmRun]) -> pd.DataFrame:
    """One row per arm per fold, in fold order, every delta naming its own reference.

    Three references, per §7.3: forecast error against persistence, direction against
    always-long, trading against buy-and-hold. Each lives in a column whose header says so,
    so a reader meets the reference at the same moment as the number.

    ``flatness`` sits immediately after ``mae`` and is not optional (ruled 20 Aug 2026):
    across arms MAE is close to a monotone function of it, so a reader given MAE alone
    cannot tell an arm that got better from one that got flatter.
    """
    persistence = {run.fold: run.result for run in runs if run.name == BASELINE}
    held = {run.fold: run.result for run in runs if run.name == BUY_AND_HOLD}
    rows = []
    for run in runs:
        baseline = persistence.get(run.fold)
        market = held.get(run.fold)
        direction = metrics.direction_accuracy(run.result)
        rows.append(
            {
                "fold": run.fold,
                "arm": run.name,
                "aside": "yes" if run.calibration.stood_aside else "",
                "trades": len(run.result.strategy_trades),
                "direction": direction,
                "dir_ref": metrics.always_long_accuracy(run.result),
                "dir_vs_long": direction - metrics.always_long_accuracy(run.result),
                "mae": metrics.mae(run.result) if run.forecasts else math.nan,
                # Immediately after `mae`, for `dir_ref`'s reason and a measured one:
                # MAE tracks flatness at Spearman +0.81 and direction at +0.01, so the
                # two columns cannot be allowed to be read apart (ruled 20 Aug 2026).
                "flatness": (
                    metrics.flatness(run.result) if run.forecasts else math.nan
                ),
                "mae_vs_pers": (
                    metrics.mae(run.result, baseline)
                    if run.forecasts and baseline is not None
                    else math.nan
                ),
                "total_return": metrics.total_return(run.result),
                "ret_vs_bh": _against(metrics.total_return, run.result, market),
                "sharpe": metrics.sharpe(run.result),
                "sharpe_vs_bh": _against(metrics.sharpe, run.result, market),
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


def equal_weight(n_symbols: int):
    """A sizer that puts ``1/n`` of equity into each position. **Buy-and-hold only.**

    It deliberately ignores ``max_position_pct`` and ``max_gross_exposure``, because
    buy-and-hold is a **reference portfolio, not a strategy run under this project's risk
    rules** - a fully invested hold is what the phrase means and what a reader will check
    against. The consequence must be stated wherever the comparison is: the strategy is
    capped at ``max_gross_exposure`` of equity while this arm is fully invested, so part of
    any gap between them is exposure rather than skill.

    It still cannot overdraw: ``equity - gross_exposure`` is exactly the cash the engine
    holds, and the last of the five entries would otherwise exceed it by the fees the first
    four paid.
    """

    def sizer(
        signal: Signal, equity: float, gross_exposure: float, cfg: Config
    ) -> float:
        del signal, cfg
        return max(0.0, min(equity / n_symbols, equity - gross_exposure))

    sizer.__name__ = f"equal_weight_{n_symbols}"
    return sizer


def run_arm(
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
        _forecasts(val_batch, run.model.predict(val_batch.X), run.stats),
        _bars_between(bars, fold.val[0], fold.val[-1]),
        risk.position_sizer,
        arm_cfg,
    )

    test_batch = _windows(frames, run, arm_cfg, fold.test)
    # Raw log returns from here down: the band, the backtester and every metric are in the
    # unit the market is in, not the unit the model was fitted in. The metrics read the
    # forecasts the backtester traded rather than restoring a second time - one forward
    # pass, one restoration, so the two cannot disagree.
    forecasts = _forecasts(test_batch, run.model.predict(test_batch.X), run.stats)
    predicted = np.stack([forecast.path for forecast in forecasts])
    actual = restore_targets(test_batch.y, test_batch, run.stats)
    result = run_backtest(
        _bars_between(bars, fold.test[0], fold.test[-1]),
        decide_all(forecasts, calibration.thresholds, arm_cfg),
        risk.position_sizer,
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
            actual=actual,
        ),
        calibration=calibration,
        seconds=time.perf_counter() - started,
        cancellation=_mean_cancellation(run.model, test_batch, arm_cfg),
    )


def _mean_cancellation(model, batch: WindowBatch, cfg: Config) -> float:
    """Mean ``|sum(c)| / sum(|c|)`` over the batch's windows.

    Spec 7 (1j): high cancellation is the signature of a linear model whose weights are
    fitting noise that mostly offsets, and it is the mechanism behind the direction result
    rather than a separate diagnostic. It costs one attribution per window of a linear
    map, and it falls out of a layer the system runs anyway.
    """
    channels = cfg.channels.active_channels
    found = [attribute(model, window, channels) for window in batch.X]
    if not found or not any(map(cancellation_is_meaningful, found)):
        # A univariate model reads exactly 1.0 whatever it did - one channel cannot
        # disagree with itself - so the cell is empty rather than perfect.
        return math.nan
    return float(np.mean([cancellation(one) for one in found]))


def run_buy_and_hold(
    fold: Fold,
    frames: dict[str, pd.DataFrame],
    bars: dict[str, pd.DataFrame],
    cfg: Config,
) -> ArmRun:
    """Equal-weight the universe at the first tradeable open, hold to the last close.

    Run, not drawn: one ``enter_long`` per symbol on the fold's first test bar, filled at
    the **next** bar's open like every other order, and liquidated at the final close as an
    administrative exit. Entry slippage, both fees and the gap rules all apply, so the arm
    is measured under the same frictions as the strategy it is a reference for.

    **The stop and the target are removed**, expressed as infinite fractions rather than by
    special-casing the engine: a 3% stop on a passive holding would make this arm "the
    project's risk rules applied to a passive entry", which is a different thing from
    buy-and-hold and would understate the market it is meant to represent.

    Its direction accuracy is the always-long bar **by construction** - it calls up on every
    window and nothing else - so the two are asserted equal. If they ever disagree, one of
    :func:`metrics.direction_accuracy` and :func:`metrics.always_long_accuracy` is wrong,
    and that is worth more than either number.
    """
    started = time.perf_counter()
    passive = replace(
        cfg, risk=replace(cfg.risk, stop_loss_pct=math.inf, take_profit_pct=math.inf)
    )
    symbols = sorted(bars)

    entries = tuple(
        # No trend view, no threshold consulted: this arm holds the market, it does not
        # forecast it. Recording that honestly beats inventing a conviction it never had.
        Signal(
            symbol=symbol,
            action=ENTER_LONG,
            trend_strength=0.0,
            up_points=0,
            passed_threshold=False,
        )
        for symbol in symbols
    )
    result = run_backtest(
        _bars_between(bars, fold.test[0], fold.test[-1]),
        {fold.test[0]: entries},
        equal_weight(len(symbols)),
        passive,
    )

    batch = WindowBatch.concat(
        [
            trainer.select_windows(
                build_windows(frames[symbol], cfg, symbol), fold.test
            )
            for symbol in symbols
        ]
    )
    always_up = np.ones_like(batch.y)
    arm = metrics.ArmResult(
        name=BUY_AND_HOLD,
        equity=result.equity,
        trades=result.trades,
        predicted=always_up,
        actual=batch.y,
    )
    _assert_is_the_bar(arm)

    return ArmRun(
        fold=fold.number,
        name=BUY_AND_HOLD,
        result=arm,
        calibration=Calibration(
            thresholds=Thresholds.never(),
            val_sharpe=math.nan,
            val_trades=0,
            stood_aside=False,  # it holds the market; it simply consults no band
            candidates=(),
        ),
        seconds=time.perf_counter() - started,
        forecasts=False,
    )


def _assert_is_the_bar(arm: metrics.ArmResult) -> None:
    """Buy-and-hold's realised direction accuracy must equal the always-long reference.

    The same quantity computed twice by two functions written for different purposes. They
    agree or something is wrong, and a silent disagreement would put a wrong bar under
    every direction number in the study.
    """
    scored = metrics.direction_accuracy(arm)
    reference = metrics.always_long_accuracy(arm)
    if not math.isclose(scored, reference, rel_tol=1e-12, abs_tol=1e-12):
        raise SmokeError(
            "buy-and-hold scored a direction accuracy of "
            f"{scored!r} while the always-long reference is {reference!r}. They are the "
            "same quantity by construction, so one of direction_accuracy and "
            "always_long_accuracy is computed wrongly."
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


def _forecasts(
    batch: WindowBatch,
    predicted: np.ndarray,
    stats: Mapping[str, ChannelStats],
) -> list[Forecast]:
    """Model output as `Forecast` objects, **in raw log returns**.

    `build_windows` scales the target by each symbol's own deviation, so a model fitted on
    those windows forecasts in that unit. Every `Forecast` this module produces is
    restored first, which is what puts the calibrated thresholds of GB-20 on the same
    scale as the prices the backtester trades - a band fitted to scaled forecasts would be
    numerically fine and would mean nothing.
    """
    raw = restore_targets(predicted, batch, stats).astype("float32")
    return [
        Forecast(path=raw[row], symbol=batch.symbols[row], as_of=timestamp)
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


def _against(
    metric, arm: metrics.ArmResult, reference: metrics.ArmResult | None
) -> float:
    """A trading metric's delta against buy-and-hold, or NaN when there is none."""
    return math.nan if reference is None else metric(arm, reference)


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
        choices=tuple(sorted(ALL_FORECASTERS)),
        default=None,
        help=(
            "which arm to run; persistence is always run beside it "
            "(default: whatever model.active names)"
        ),
    )
    parser.add_argument(
        "--prepare-live",
        metavar="DIR",
        default=None,
        help=(
            "train on the most recent fold, calibrate its band, and write both into DIR "
            "for GB-26's live loop to load. Runs nothing else."
        ),
    )
    parser.add_argument(
        "--prepare-replay",
        nargs=2,
        metavar=("FOLD", "DIR"),
        default=None,
        help=(
            "train on FOLD's training split, calibrate on its validation split, and write "
            "the checkpoint, band and test range into DIR for GB-38's replay. Runs "
            "nothing else."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI, not by tests
    raise SystemExit(main())
