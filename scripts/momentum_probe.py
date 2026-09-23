"""Why a cross-sectional momentum arm was declined: the shuffled control cannot kill it.

**This script exists because its numbers are quoted in `IDEAS_PARKED.md`.** Two figures in
this project's history were computed once, written into prose, and left with no producing
code a reviewer could run - the 16.00% gross exposure, and GB-46's ~86% geometry figure,
which appears in neither `results.csv` nor `report.md`. A parked entry that declines an arm
on measured grounds is a claim like any other, and a claim whose producing code does not
exist is the same defect wearing a decision.

**The arm that was considered.** Rank the universe by trailing 12-month return, hold the
top three, rebalance monthly. It forecasts nothing; it is a selection rule, in the family
`buy_and_hold` already occupies. Nothing here trains, and nothing here is an arm of the
study: this is the measurement that decided not to build one.

**The finding, and it is about the control rather than about momentum.** `study.null_bars`
under `SHUFFLED` permutes a symbol's log-return series and rebuilds prices from the
cumulative sum. **A permutation preserves a sum**, so every symbol's whole-sample return
survives the shuffle exactly, and the cross-sectional ordering of who won over the sample
is preserved perfectly. Under that permutation any trailing window is a sample without
replacement from the whole series, so its expected value is the window length times the
symbol's full-sample mean - which is a partial readout of the sample's outcome, including
the bars after `t`. A signal that is a monotone function of accumulated per-symbol return
is therefore **structurally immune** to this control, and the study's standing requirement
that every headline claim face both robustness tests has a case where one cannot fire.

`NOISE` does bite: it draws each symbol's returns from `N(0, sd)`, so every symbol has zero
expected drift and the ranking becomes noise.

**The discriminator, which is the reusable part.** Whether a control can falsify an arm of
this shape is answerable before running the arm, by one number per control:

    Spearman(real full-sample drift, control full-sample drift) across the universe

It is 1.0 for `shuffled` and near zero for `noise`. Any future arm whose signal accumulates
per-symbol return should be checked against this before its shuffled row is read as
evidence.

**Rebalance on the FIRST bar of a new month, never the last bar of the old one.** Only the
first is knowable at `t`: a bar is the last of its month only once the next bar arrives.
The distinction is not academic here, because **the causality harness cannot tell the two
apart** - `causality.perturb` disturbs values and never touches the index, so a month-end
rule would pass `assert_causal` in both modes at all three splits. The harness would
certify either rule; only one of them is causal.

**Why the parameters are a dataclass here rather than `settings.yaml`.** Rule 5 puts
configuration in the config file, and this is the boundary of it: a `momentum:` section
would move `config_hash` for every configuration in the project, for an arm that is parked
and will not be built. `top_n` is deliberately **not** `cfg.signal.top_k` either - that is
the live loop's cap on band-passing candidates, a different quantity with a different
meaning, and binding a parked probe to the deployed config would couple them for no reason.
Every field is overridable on the command line, and the override reaches the measurement
rather than being parsed and dropped, which this project has paid for twice in
`smoke_offline`'s `--model`.

Usage::

    python scripts/momentum_probe.py --out report/momentum_probe.csv
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from scipy import stats as scipy_stats

    # Inside the guard, as in `operator_null_control`: a clone without `pip install -e .`
    # must be told that, rather than getting a bare ModuleNotFoundError from the project's
    # own package. `ImportError` because a half-installed package raises the parent.
    from glassbox.config.loader import Config, load_config
    from glassbox.experiments.study import CONTROLS, REAL, SHUFFLED, null_bars
    from glassbox.smoke_offline import _common_index, load_cached_bars
except ImportError as missing:  # pragma: no cover - the wrong-interpreter guard
    raise SystemExit(
        f"""cannot import {missing.name!r}: this script needs the project environment.
    .venv/Scripts/activate       (Windows)
    source .venv/bin/activate    (macOS, Linux)
then, if the environment is new:  pip install -e ".[dev]" """
    ) from missing

#: The smallest number of non-zero paired differences the signed-rank test is allowed.
#: Below this the exact test cannot reach 0.05 at all, so a p from it would be a number
#: with no power behind it. The same bound and the same reason as `stats.MIN_FOLDS`.
MIN_PAIRS = 6

COLUMNS = (
    "control",
    "quantity",
    "subject",
    "n",
    "value",
    "p",
)


@dataclass(frozen=True)
class Design:
    """The rule being probed. One object, so no caller can hold a stale half of it."""

    lookback: int = 252
    """Trading days in the trailing ranking window. ~12 months, the classical look-back.
    It sits below `builder.min_history_bars`' 445, so a live arm of this shape would need
    no history the loop does not already request."""

    forward: int = 21
    """Trading days scored forward from each rebalance. ~1 month, the holding period."""

    top_n: int = 3
    """Names held. Not `cfg.signal.top_k` - see the module docstring."""


def _world(cfg: Config, bars: dict[str, pd.DataFrame], control: str) -> dict:
    """The universe under one control, through the study's own path.

    `null_bars` with `cfg.meta.seed + index` over the sorted symbols, which is what
    `study._condition_rows` does line for line. The controls have to be the study's rather
    than a reimplementation, or a finding about the control would be a finding about this
    file.
    """
    return {
        symbol: null_bars(frame, control, cfg.meta.seed + index)
        for index, (symbol, frame) in enumerate(sorted(bars.items()))
    }


def _closes(world: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One close column per symbol, on the index every symbol shares."""
    index = _common_index(world)
    return pd.DataFrame(
        {symbol: frame["close"].loc[index] for symbol, frame in sorted(world.items())},
        index=index,
    )


def month_starts(index: pd.DatetimeIndex) -> np.ndarray:
    """Positions of the first bar of each calendar month.

    Computed from the previous bar rather than the next one, which is the whole point: the
    test is "this bar's month differs from the last bar's month", and that is answerable
    standing on the bar. "This bar is the last of its month" is not.
    """
    stamps = pd.DatetimeIndex(index)
    key = stamps.year.to_numpy() * 100 + stamps.month.to_numpy()
    return np.flatnonzero(np.r_[True, np.diff(key) != 0])


def _paired_p(left: np.ndarray, right: np.ndarray) -> float:
    """Two-sided paired signed-rank p, or NaN when there is not enough to test."""
    moved = (left - right)[left != right]
    if moved.size < MIN_PAIRS:
        return math.nan
    return float(scipy_stats.wilcoxon(left, right, zero_method="wilcox").pvalue)


def _rho(left: pd.Series, right: pd.Series) -> float:
    """Spearman between two aligned series, or NaN when either is degenerate."""
    if len(left) < 3 or left.nunique() < 2 or right.nunique() < 2:
        return math.nan
    return float(scipy_stats.spearmanr(left, right).statistic)


def _mean(values: np.ndarray) -> float:
    """Mean of the defined entries; NaN when there are none."""
    defined = np.asarray(values, dtype="float64")
    defined = defined[np.isfinite(defined)]
    return math.nan if defined.size == 0 else float(defined.mean())


def _row(
    control: str,
    quantity: str,
    subject: str,
    n: int,
    value: float,
    p: float = math.nan,
) -> dict:
    return {
        "control": control,
        "quantity": quantity,
        "subject": subject,
        "n": int(n),
        "value": value,
        "p": p,
    }


def measure_control(
    cfg: Config, bars: dict[str, pd.DataFrame], control: str, design: Design
) -> tuple[list[dict], pd.Series]:
    """Every quantity for one control, plus that control's per-symbol full-sample drift.

    The drift is returned beside the rows because the discriminator compares it **across**
    controls, and a function measuring one control cannot hold both sides of that.
    """
    close = _closes(_world(cfg, bars, control))
    log_close = np.log(close)

    # Trailing: bars t-lookback..t, so it is known standing on t. Forward: bars
    # t..t+forward, the realised outcome, used only to score and never to select.
    trailing = log_close - log_close.shift(design.lookback)
    forward = log_close.shift(-design.forward) - log_close
    later = trailing.shift(-design.lookback)

    dates = [
        stamp
        for stamp in close.index[month_starts(close.index)]
        if trailing.loc[stamp].notna().all() and forward.loc[stamp].notna().all()
    ]

    alignment: list[float] = []
    chosen: list[float] = []
    reference: list[float] = []
    held: list[str] = []
    for stamp in dates:
        rank, realised = trailing.loc[stamp], forward.loc[stamp]
        alignment.append(_rho(rank, realised))
        picks = rank.nlargest(design.top_n).index
        held.extend(picks)
        chosen.append(float(realised[picks].mean()))
        reference.append(float(realised.mean()))

    # Does the ranking itself persist? A separate date set, because it needs a trailing
    # window a further `lookback` bars on, and reporting it under `dates`' n would
    # overstate what it was measured over.
    persistence = [
        _rho(trailing.loc[stamp], later.loc[stamp])
        for stamp in dates
        if later.loc[stamp].notna().all()
    ]

    top, flat = np.array(chosen), np.array(reference)
    counts = pd.Series(held, dtype="object").value_counts()
    drift = log_close.iloc[-1] - log_close.iloc[0]
    finite = [value for value in alignment if math.isfinite(value)]

    rows = [
        _row(control, "rebalances", "", len(dates), float(len(dates))),
        _row(
            control,
            "spearman_trailing_forward",
            "",
            len(finite),
            _mean(np.array(finite)),
            (
                float(scipy_stats.ttest_1samp(finite, 0.0).pvalue)
                if len(finite) > 1
                else math.nan
            ),
        ),
        _row(control, "top_n_forward_return", "", top.size, _mean(top)),
        _row(control, "equal_weight_forward_return", "", flat.size, _mean(flat)),
        _row(
            control,
            "top_n_minus_equal_weight",
            "",
            top.size,
            _mean(top - flat) if top.size else math.nan,
            _paired_p(top, flat) if top.size else math.nan,
        ),
        _row(
            control,
            "rank_persistence_one_lookback_on",
            "",
            len(persistence),
            _mean(np.array(persistence)),
        ),
    ]
    rows += [
        _row(control, "months_held", symbol, len(dates), float(counts.get(symbol, 0)))
        for symbol in close.columns
    ]
    # The raw material of the discriminator, so the correlation below can be audited from
    # this file rather than only from the printed number.
    rows += [
        _row(control, "full_sample_log_drift", symbol, 1, float(drift[symbol]))
        for symbol in close.columns
    ]
    return rows, drift


def measure(cfg: Config, design: Design, log=print) -> pd.DataFrame:
    """One row per (control, quantity, subject), for every control the study runs."""
    bars = load_cached_bars(cfg)
    log(f"{len(bars)} symbols from the cache, no network, nothing trained")

    rows: list[dict] = []
    drifts: dict[str, pd.Series] = {}
    for control in CONTROLS:
        measured, drift = measure_control(cfg, bars, control, design)
        rows.extend(measured)
        drifts[control] = drift
        log(f"  {control:9} done")

    # **The discriminator.** Whether a control can falsify an arm whose signal accumulates
    # per-symbol return is exactly whether that control moved the cross-sectional ordering
    # of accumulated return. `real` against itself is the identity and is written anyway,
    # so the column reads as a scale rather than as an absence.
    rows += [
        _row(
            control,
            "drift_correlation_vs_real",
            "",
            len(drifts[REAL]),
            _rho(drifts[REAL], drifts[control]),
        )
        for control in CONTROLS
    ]
    return pd.DataFrame(rows, columns=list(COLUMNS))


def _value(table: pd.DataFrame, control: str, quantity: str) -> float:
    found = table[(table["control"] == control) & (table["quantity"] == quantity)]
    return math.nan if found.empty else float(found["value"].iloc[0])


def _p(table: pd.DataFrame, control: str, quantity: str) -> float:
    found = table[(table["control"] == control) & (table["quantity"] == quantity)]
    return math.nan if found.empty else float(found["p"].iloc[0])


def method(cfg: Config, design: Design) -> str:
    """The method, in the output, because a number without one is prose with decimals."""
    pad = " " * 20
    return "\n".join(
        [
            "METHOD",
            (
                f"  universe        : {len(cfg.universe)} symbols, read from "
                f"{cfg.data.cache_dir}/ only. No network. Nothing is trained."
            ),
            (
                f"  signal          : trailing log return over {design.lookback} "
                "bars, known standing on t."
            ),
            f"  selection       : the top {design.top_n} by that rank, equally weighted.",
            (
                "  rebalance       : the FIRST bar of each new calendar month. Never "
                "the last bar of the old one -"
            ),
            (
                f"{pad}a bar is the last of its month only once the next bar arrives, "
                "and `causality.perturb`"
            ),
            (
                f"{pad}never touches the index, so `assert_causal` would certify "
                "either rule."
            ),
            (
                "  scored on       : the realised log return over the next "
                f"{design.forward} bars. Used to score, never to select."
            ),
            (
                "  controls        : `study.null_bars`, seeded `meta.seed + index` "
                "over the sorted symbols -"
            ),
            (
                f"{pad}the study's own path, so a finding about a control is not a "
                "finding about this file."
            ),
            (
                "  paired test     : two-sided Wilcoxon signed-rank, "
                f"`zero_method='wilcox'`, refused below {MIN_PAIRS} moved pairs."
            ),
            (
                "  NOT an arm      : this runs no backtest, pays no fee or slippage, "
                "and appears in no results.csv."
            ),
            (
                f"{pad}It is the measurement that declined an arm, not a measurement "
                "of one."
            ),
        ]
    )


def verdict(table: pd.DataFrame) -> str:
    """Whether each control can falsify an arm of this shape, and how to know in advance."""
    real_edge = _value(table, REAL, "top_n_minus_equal_weight")
    shuffled_edge = _value(table, SHUFFLED, "top_n_minus_equal_weight")
    direction = "MORE" if shuffled_edge > real_edge else "less"
    lines = [
        "VERDICT",
        (
            f"  On real data the rule earns {real_edge:+.5f} per rebalance over equal "
            f"weight (paired p = "
            f"{_p(table, REAL, 'top_n_minus_equal_weight'):.4f}). On SHUFFLED returns"
        ),
        (
            f"  it earns {shuffled_edge:+.5f} (p = "
            f"{_p(table, SHUFFLED, 'top_n_minus_equal_weight'):.4f}) - {direction} "
            "than on the market."
        ),
        "",
        (
            "  THE DISCRIMINATOR - Spearman of full-sample log drift against the real "
            "world, across the universe."
        ),
        (
            "  A control can falsify a signal that accumulates per-symbol return only "
            "if it moves this ordering:"
        ),
    ]
    for control in CONTROLS:
        correlation = _value(table, control, "drift_correlation_vs_real")
        if control == REAL:
            note = "the identity, printed as the scale"
        elif correlation > 0.9:
            note = (
                "INERT: the ordering survives, so this control cannot falsify the arm"
            )
        else:
            note = (
                "BITES: the ordering is destroyed, so this control can falsify the arm"
            )
        lines.append(f"    {control:9} {correlation:+.4f}   {note}")

    lines += [
        "",
        (
            "  A permutation preserves a sum, so a shuffled symbol keeps its "
            "whole-sample return exactly, and any"
        ),
        (
            "  trailing window under it is a partial readout of the full-sample "
            "outcome - including bars after t."
        ),
        (
            "  Surviving the shuffle is therefore neither a bug nor a finding here. "
            "It is a property of the control."
        ),
    ]

    held = table[(table["control"] == REAL) & (table["quantity"] == "months_held")]
    rebalances = _value(table, REAL, "rebalances")
    top = held.sort_values("value", ascending=False).head(3)
    names = ", ".join(
        f"{row.subject} {row.value / rebalances * 100:.1f}%" for row in top.itertuples()
    )
    lines += [
        "",
        f"  CONCENTRATION on real data, over {int(rebalances)} rebalances: {names}.",
        (
            "  Spec 2.4 criterion 3 admits only securities tradable through to the "
            "cache's last bar, so this rule"
        ),
        (
            "  concentrates a survivor-selected universe onto the few that rose most. "
            "Nothing in this repository can"
        ),
        (
            "  bound that: there is no delisted series and no point-in-time "
            "membership anywhere in the tree."
        ),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    default = Design()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="report/momentum_probe.csv")
    parser.add_argument("--lookback", type=int, default=default.lookback)
    parser.add_argument("--forward", type=int, default=default.forward)
    parser.add_argument("--top-n", type=int, default=default.top_n)
    args = parser.parse_args(argv)

    design = Design(lookback=args.lookback, forward=args.forward, top_n=args.top_n)
    cfg = load_config()

    print(method(cfg, design))
    print()
    table = measure(cfg, design)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    print()
    portfolio = table[table["subject"] == ""].pivot(
        index="quantity", columns="control", values="value"
    )
    print(portfolio.reindex(columns=list(CONTROLS)).to_string())
    print()
    print(verdict(table))
    print(f"\nwrote {out}: {len(table)} rows")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI
    raise SystemExit(main())
