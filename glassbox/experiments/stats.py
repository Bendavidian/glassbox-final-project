"""X: paired Wilcoxon signed-rank of every arm against **its own** reference.

Spec 7.4 asked for "paired Wilcoxon signed-rank of each arm vs persistence on per-fold
direction accuracy and Sharpe". **That sentence is wrong and is corrected here** (ruled
23 Aug 2026): a single reference for every metric reintroduces the error 7.3 already fixed
once. Persistence forecasts zero, so it has **no direction to compare against** and takes
**no trades**, and its ``direction`` and ``sharpe`` cells in ``results.csv`` are literally
empty. Testing direction against it would compare a number to a NaN; testing Sharpe
against it would compare a strategy to an arm that never traded.

**The three-reference rule, and each metric is tested against the reference that means
something for it:**

============  ==========================  ==========================================
metric        reference                   why
============  ==========================  ==========================================
``mae``       persistence, same channels  the forecast-error baseline (7.3)
``direction`` always-long, same fold      the realised up rate, per fold, not 0.5
``sharpe``    buy-and-hold, same fold     a trading metric needs a trading reference
``total_return`` buy-and-hold, same fold  the same, and 7.3's total-return convention
============  ==========================  ==========================================

**Pairing is by fold**, within one condition. A fold is the unit because the two arms saw
the same windows in the same regime, which is what makes the difference a paired
observation rather than two samples from different periods.

**Multiple comparisons are counted and corrected, and the family is stated.** With 16
folds and this many arms the grid runs tens of tests, and at ``alpha = 0.05`` a handful of
significant results is what pure chance produces. Every row therefore carries ``n_tests``
- the size of the family, which is **every test computed in the same call** - and
``p_holm``, the Holm-Bonferroni adjusted value. Holm rather than Bonferroni because it is
uniformly more powerful and costs nothing; Holm rather than Benjamini-Hochberg because
FWER is the guarantee a reader of a results table assumes, and because BH's guarantee
needs a dependence assumption nobody here has checked. **The raw ``p`` stays in the table
beside it**: an adjusted value alone hides how much of the adjustment the family size did.

**A floor worth knowing before reading any of it.** The exact two-sided signed-rank test
on ``n`` folds cannot return a p below ``2**(1-n)`` - 3.05e-5 at the 16 this grid runs,
and the printed bound is computed rather than quoted - so no claim in this study can
be significant past that however large its effect. Holm over a family of 50 leaves a
smallest achievable adjusted value of about 1.5e-3, which is still comfortably
significant - the correction is survivable, and a claim that does not survive it was not
close.

Implemented in GB-51.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from glassbox.experiments import study
from glassbox.experiments.study import BUY_AND_HOLD, PAIR_KEYS

ALWAYS_LONG = "always_long"
PERSISTENCE = "persistence"


@dataclass(frozen=True)
class Reference:
    """Where a metric's paired comparison comes from.

    Attributes:
        arm: The model whose rows carry the reference, or ``None`` when the reference is
            a column of the arm's **own** row - which is how always-long is stored, since
            the realised up rate is a property of the fold and not of another arm.
        column: The column holding the reference value.
        label: What the reference is called in the output.
        same_channels: Pair only against the reference's row for the same channel set.
            True for persistence, whose windows must be the arm's own; False for
            buy-and-hold, which has no channel set because it uses no features.
    """

    arm: str | None
    column: str
    label: str
    same_channels: bool = False


REFERENCES: dict[str, Reference] = {
    "mae": Reference(PERSISTENCE, "mae", PERSISTENCE, same_channels=True),
    "direction": Reference(None, "direction_reference", ALWAYS_LONG),
    "sharpe": Reference(BUY_AND_HOLD, "sharpe", BUY_AND_HOLD),
    "total_return": Reference(BUY_AND_HOLD, "total_return", BUY_AND_HOLD),
}

COLUMNS = (
    *PAIR_KEYS,
    "model",
    "channels",
    "metric",
    "reference",
    "n_folds",
    "mean",
    "sd",
    "reference_mean",
    "mean_delta",
    "statistic",
    "p",
    "p_holm",
    "n_tests",
    "skipped",
)

# The smallest two-sided p the exact test can return on n folds is 2 / 2**n. Below this
# many folds the test cannot reach 0.05 at all, so a p from it would be a number with no
# power behind it rather than an absence of effect.
MIN_FOLDS = 6


def wilcoxon(frame: pd.DataFrame, metrics: Sequence[str] | None = None) -> pd.DataFrame:
    """One row per (condition, arm, metric): mean, sd and the paired p-value.

    Args:
        frame: A per-fold results table with :data:`study.COLUMNS`.
        metrics: Which metrics to test. Defaults to every key of :data:`REFERENCES`.

    Returns:
        A frame with :data:`COLUMNS`, sorted by condition then arm then metric. It carries
        ``control`` and ``skipped`` so that ``study.reportable`` gates it **unchanged** -
        one gate for both tables, rather than a second filter that can disagree with the
        first.

    Null-control conditions are tested and returned like any other. A p-value under white
    noise is not a result about the market, and it is exactly what the standing
    requirement of 20 Aug asks to see beside the real one; the gate keeps it out of any
    table that claims to be about the market.
    """
    wanted = tuple(REFERENCES) if metrics is None else tuple(metrics)
    live = frame[~frame["skipped"].astype(bool)]

    rows: list[dict] = []
    for key, condition in live.groupby(list(PAIR_KEYS), dropna=False, sort=False):
        for (model, channels), arm in condition.groupby(
            ["model", "channels"], dropna=False, sort=False
        ):
            if model == BUY_AND_HOLD:
                # Its direction **is** the always-long reference by construction and it
                # makes no forecast, so every test it could take is a tautology or a NaN.
                continue
            for metric in wanted:
                row = _test(
                    dict(zip(PAIR_KEYS, key, strict=True)),
                    model,
                    channels,
                    arm,
                    condition,
                    metric,
                )
                if row is not None:
                    rows.append(row)

    table = pd.DataFrame(rows, columns=list(COLUMNS))
    table["n_tests"] = len(table)
    table["p_holm"] = holm(table["p"].to_numpy(dtype="float64"))
    return table


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the order given.

    Step-down: the smallest raw value is multiplied by ``n``, the next by ``n - 1``, and
    so on, then made non-decreasing and capped at 1. Controls the family-wise error rate
    without assuming anything about dependence between the tests.

    NaNs pass through as NaN and **do not count towards the family**: a test that could
    not be computed is not a comparison anybody made.
    """
    values = np.asarray(p, dtype="float64")
    adjusted = np.full(values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return adjusted

    order = finite[np.argsort(values[finite], kind="stable")]
    remaining = order.size
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (remaining - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def _test(
    condition: dict,
    model: str,
    channels: str,
    arm: pd.DataFrame,
    peers: pd.DataFrame,
    metric: str,
) -> dict | None:
    """One paired test, or ``None`` when there is nothing to compare.

    Returns ``None`` rather than a row of NaNs when the arm **is** its own reference or
    the metric is absent for it: a row saying "persistence against persistence, p = NaN"
    is noise in a table whose whole job is to be read.
    """
    reference = REFERENCES[metric]
    if reference.arm == model:
        return None

    paired = _pairs(arm, peers, channels, metric, reference)
    if paired is None:
        return None
    values, against = paired

    difference = values - against
    moved = difference[difference != 0.0]
    if moved.size < MIN_FOLDS:
        statistic, p = math.nan, math.nan
    else:
        result = scipy_stats.wilcoxon(values, against, zero_method="wilcox")
        statistic, p = float(result.statistic), float(result.pvalue)

    return {
        **condition,
        "model": model,
        "channels": channels,
        "metric": metric,
        "reference": reference.label,
        "n_folds": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else math.nan,
        "reference_mean": float(against.mean()),
        "mean_delta": float(difference.mean()),
        "statistic": statistic,
        "p": p,
        "p_holm": math.nan,  # filled once the family is known; see `wilcoxon`
        "n_tests": 0,
        "skipped": False,
    }


def _pairs(
    arm: pd.DataFrame,
    peers: pd.DataFrame,
    channels: str,
    metric: str,
    reference: Reference,
) -> tuple[np.ndarray, np.ndarray] | None:
    """The arm's per-fold values beside its reference's, aligned on ``fold``."""
    if metric not in arm.columns:
        return None

    values = arm.set_index("fold")[metric]
    if reference.arm is None:
        against = arm.set_index("fold")[reference.column]
    else:
        source = peers[peers["model"] == reference.arm]
        if reference.same_channels:
            source = source[source["channels"] == channels]
        if source.empty:
            return None
        against = source.set_index("fold")[reference.column]

    joined = pd.concat([values, against], axis=1, join="inner").dropna()
    if joined.empty:
        return None
    return (
        joined.iloc[:, 0].to_numpy(dtype="float64"),
        joined.iloc[:, 1].to_numpy(dtype="float64"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Print the test table for one results file. Returns an exit code rather than raising.

    A command of its own rather than a stage of ``study.main`` because this module imports
    ``study`` and the reverse import would be a cycle - and because the tests are a pure
    function of the file, so re-deriving them costs nothing and re-running a 20-minute
    grid to see a p-value would be absurd.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=study.RESULTS_FILE)
    parser.add_argument("--out", default=None, help="also write the table as CSV")
    parser.add_argument(
        "--all-controls",
        action="store_true",
        help="test the null-control conditions too, widening the family accordingly",
    )
    args = parser.parse_args(argv)

    path = Path(args.results)
    if not path.is_file():
        print(f"stats: no results file at {path}", file=sys.stderr)
        return 2

    frame = pd.read_csv(path)
    table = wilcoxon(frame if args.all_controls else study.reportable(frame))
    if table.empty:
        print("stats: no comparable arms in the file", file=sys.stderr)
        return 2

    shown = table.drop(columns=["skipped"])
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(shown.to_string(index=False))
    print()
    # Derived, not written down. The exact two-sided Wilcoxon on n pairs cannot go below
    # 2**(1-n); stating "16 folds" and "3.05e-5" as literals made the bound a claim about
    # a grid rather than about this one, and a re-run with a different fold count would
    # have printed a false bound with nothing to catch it (27 Aug 2026).
    pairs = int(table["n_folds"].max()) if len(table) else 0
    floor = 2.0 ** (1 - pairs) if pairs else float("nan")
    print(
        f"{len(table)} tests in the family. `p` is uncorrected; `p_holm` is "
        f"Holm-Bonferroni over all {len(table)} of them, controlling the family-wise "
        f"error rate. The exact test on {pairs} folds cannot return a p below {floor:.3g}."
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALWAYS_LONG",
    "COLUMNS",
    "MIN_FOLDS",
    "PERSISTENCE",
    "REFERENCES",
    "Reference",
    "holm",
    "main",
    "wilcoxon",
]
