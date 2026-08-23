"""X: results.csv to summary tables, per-fold boxplots and the COF curve.

**The report regenerates from the CSV alone.** Nothing here trains, backtests or reads a
cache: given `results.csv` and nothing else it produces the whole chapter, so a figure can
be rebuilt in October without a 22-minute run and without the machine that made it. The
significance columns come from :mod:`glassbox.experiments.stats`, which is a pure function
of the same file, rather than from a second artefact that could drift out of step with it.

**Three things are structural here rather than remembered**, because each is a rule this
project has already watched fail:

``data_snapshot_last_bar`` **in the header**
    A report built on stale data says so on its own face. It is the first line, not a
    footnote, and :func:`header` refuses a file whose rows disagree about it.
``mae`` **never without** ``flatness``
    The column order comes from :func:`metrics.columns_for` and
    :data:`metrics.COMPANIONS`, the same pairing the fold table uses, so printing MAE
    alone means deleting the pairing rather than forgetting it. The rank correlations are
    printed **with** every MAE table, not once in a methods note: across arms MAE is close
    to a monotone function of flatness and carries almost no information about accuracy.
``every claim at every anchor and against both nulls``
    :data:`CLAIMS` is a list of effects, and :func:`claim_table` renders each one across
    the three fold-grid anchors and the three control conditions. A claim shown at one
    anchor is not reportable (7.4), and a claim not shown against its null control is not
    either (the standing requirement of 20 Aug).

Every metric is reported as a delta against **its own** reference under the three-reference
rule - MAE against persistence, direction against always-long, trading metrics against
buy-and-hold - which is :data:`stats.REFERENCES` and not a second copy of it.

Implemented in GB-52.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy import stats as scipy_stats

from glassbox.backtest import metrics
from glassbox.dashboard.app import HAIRLINE, INK, MUTED, ORANGE, PANEL, PAPER, RAMP
from glassbox.experiments import stats, study

DEFAULT_DPI = 200
REPORT_FILE = "report.md"

#: Metrics the summary table reports, each beside its own reference.
SUMMARY_METRICS = ("direction", "mae", "sharpe", "total_return")


@dataclass(frozen=True)
class Claim:
    """One headline claim, expressed as an effect a table can compute.

    Attributes:
        name: How the claim reads in the report.
        effect: Given the rows of one (anchor, control) cell, the effect size in the
            claim's own units, or NaN when the cell cannot express it.
        units: What the number is, for the column header.
    """

    name: str
    effect: Callable[[pd.DataFrame], float]
    units: str


def _edge_over_bar(cell: pd.DataFrame, model: str, channels: str) -> float:
    """An arm's direction accuracy less the fold's realised always-long rate, in points."""
    arm = cell[(cell["model"] == model) & (cell["channels"] == channels)]
    if arm.empty:
        return math.nan
    return float((arm["direction"] - arm["direction_reference"]).mean() * 100)


def _gap(cell: pd.DataFrame, left: tuple[str, str], right: tuple[str, str]) -> float:
    """One arm's direction accuracy less another's, in points."""
    first = cell[(cell["model"] == left[0]) & (cell["channels"] == left[1])]
    second = cell[(cell["model"] == right[0]) & (cell["channels"] == right[1])]
    if first.empty or second.empty:
        return math.nan
    return float((first["direction"].mean() - second["direction"].mean()) * 100)


CLAIMS: tuple[Claim, ...] = (
    Claim(
        "No arm beats the always-long bar (DLinear C0)",
        lambda cell: _edge_over_bar(cell, "dlinear", "C0_base"),
        "points vs always-long",
    ),
    Claim(
        "No arm beats the always-long bar (FITS C0)",
        lambda cell: _edge_over_bar(cell, "fits", "C0_base"),
        "points vs always-long",
    ),
    Claim(
        "FITS beats DLinear on direction",
        lambda cell: _gap(cell, ("fits", "C0_base"), ("dlinear", "C0_base")),
        "points",
    ),
    Claim(
        "The wavelets help DLinear",
        lambda cell: _gap(cell, ("dlinear", "C2_hybrid"), ("dlinear", "C0_base")),
        "points",
    ),
)


# ── reading the file ─────────────────────────────────────────────────────────


def load(path: str | Path = study.RESULTS_FILE) -> pd.DataFrame:
    """Read a results file and refuse one this report cannot describe.

    Raises:
        FileNotFoundError: No file at ``path``.
        ValueError: The file is missing a column :data:`study.COLUMNS` declares.
    """
    frame = pd.read_csv(path)
    missing = [column for column in study.COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing {missing}; it was written by an older study than this "
            "report describes, and a table built from it would be silently partial"
        )
    return frame


def header(frame: pd.DataFrame) -> list[str]:
    """The lines every table in this report sits under.

    **The data snapshot is the first of them.** A report built on a stale cache says so on
    its own face rather than only in a run log, which is the whole of GB-52's requirement.

    Raises:
        ValueError: The rows disagree about the snapshot. Two vintages in one file cannot
            be summarised under one header, and averaging across them would be a result
            about two different markets.
    """
    real = study.reportable(frame)
    snapshots = sorted(set(real["data_snapshot_last_bar"].dropna()))
    if len(snapshots) != 1:
        raise ValueError(
            f"results.csv carries {len(snapshots)} data snapshots {snapshots}; a summary "
            "over more than one would average two different markets"
        )
    folds = int(real["fold"].max()) if not real.empty else 0
    return [
        f"**Data snapshot: last bar {snapshots[0]}**",
        (
            f"{len(frame)} rows, {len(real)} reportable "
            f"({len(frame) - len(real)} null-control or skipped) · "
            f"{folds} folds · anchors {sorted(set(real['anchor']))} · "
            f"learning rates {sorted(set(real['lr']))} · "
            f"cutoffs {sorted(set(real['cutoff_period_days'].dropna().astype(int)))}"
        ),
    ]


# ── the summary table ────────────────────────────────────────────────────────


def summary(frame: pd.DataFrame, anchor: int | None = None) -> pd.DataFrame:
    """One row per arm and anchor: every metric beside **its own** reference and the delta.

    Args:
        frame: A results table.
        anchor: Restrict to one fold-grid anchor. ``None`` keeps all three, one row each,
            which is what §7.4 requires of a headline claim.

    **Held at the deployed learning rate and cutoff.** The `lr` and `COF` axes have their
    own tables; averaging across them here would report a mean over three learning rates
    as though it were a result, and the MAE column in particular is conditional on the
    rate (§7 (1n)).

    The references come from :data:`stats.REFERENCES`, so the three-reference rule has one
    definition in this codebase and this table cannot drift from the significance tests
    that use it. ``mae`` is followed by ``flatness`` through
    :func:`metrics.columns_for` rather than by hand.
    """
    real = study.reportable(frame)
    real = real[
        (real["lr"] == study.LEARNING_RATES[0])
        & (real["cutoff_period_days"] == study.CUTOFFS[0])
    ]
    if anchor is not None:
        real = real[real["anchor"] == anchor]

    tests = stats.wilcoxon(real).set_index(["anchor", "model", "channels", "metric"])

    rows: list[dict] = []
    for (arm_anchor, model, channels), arm in real.groupby(
        ["anchor", "model", "channels"], sort=True
    ):
        row: dict[str, object] = {
            "anchor": arm_anchor,
            "arm": model if not channels else f"{model} · {channels}",
            "n_folds": int(arm["fold"].nunique()),
            "n_trades": float(arm["n_trades"].mean()),
        }
        for name in SUMMARY_METRICS:
            reference = stats.REFERENCES[name]
            row[name] = float(arm[name].mean()) if name in arm else math.nan
            if name in metrics.COMPANIONS:
                companion = metrics.COMPANIONS[name]
                row[companion] = float(arm[companion].mean())
            row[f"{name}_reference"] = reference.label
            key = (arm_anchor, model, channels, name)
            row[f"{name}_delta"] = _lookup(tests, key, "mean_delta")
            row[f"{name}_p"] = _lookup(tests, key, "p")
            row[f"{name}_p_holm"] = _lookup(tests, key, "p_holm")
        rows.append(row)

    columns = [
        "anchor",
        "arm",
        "n_folds",
        "n_trades",
        *(
            part
            for name in SUMMARY_METRICS
            for part in (
                *metrics.columns_for(name)[:-1],
                f"{name}_reference",
                f"{name}_delta",
                f"{name}_p",
                f"{name}_p_holm",
            )
        ),
    ]
    return pd.DataFrame(rows, columns=columns)


def _lookup(tests: pd.DataFrame, key: tuple, column: str) -> float:
    """One test's value, or NaN where the arm is its own reference and took no test."""
    try:
        return float(tests.loc[key, column])
    except (KeyError, TypeError, ValueError):
        return math.nan


def correlations(frame: pd.DataFrame) -> dict[str, float]:
    """Spearman of MAE against flatness and against direction, over every arm-fold.

    **Printed with every MAE table**, per §7.3 as amended: across arms MAE is close to a
    monotone function of how flat the forecast is and carries almost no information about
    accuracy, and a reader who is not shown that will read the MAE column as accuracy.
    """
    real = study.reportable(frame).dropna(subset=["mae"])
    out: dict[str, float] = {}
    for against in ("flatness", "direction"):
        pair = real.dropna(subset=[against])
        out[against] = (
            float(scipy_stats.spearmanr(pair["mae"], pair[against]).statistic)
            if len(pair) > 2
            else math.nan
        )
    return out


# ── the claims, at every anchor and against both nulls ───────────────────────


def claim_table(frame: pd.DataFrame, claim: Claim) -> pd.DataFrame:
    """One claim's effect size at every anchor and under every control.

    A claim shown at one anchor is a property of that anchor (7.4) and a claim not shown
    against its null control has not been tested (20 Aug), so the two axes are the rows
    and the columns of a single table rather than two separate exercises.
    """
    live = frame[
        (~frame["skipped"].astype(bool))
        & (frame["lr"] == study.LEARNING_RATES[0])
        & (frame["cutoff_period_days"] == study.CUTOFFS[0])
    ]
    rows = []
    for anchor in sorted(set(live["anchor"])):
        row: dict[str, object] = {"anchor": anchor}
        for control in study.CONTROLS:
            cell = live[(live["anchor"] == anchor) & (live["control"] == control)]
            row[control] = claim.effect(cell) if not cell.empty else math.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=["anchor", *study.CONTROLS])


# ── the COF sweep ────────────────────────────────────────────────────────────


def cof_curve(frame: pd.DataFrame) -> pd.DataFrame:
    """The sweep as a curve: one row per cutoff, real beside its own null control.

    Carries the geometry the cutoff decides - the retained bin count and the dead-row
    share - because a cutoff on its own is a number nobody can interpret.
    """
    swept = frame[
        (frame["model"] == "fits")
        & (frame["anchor"] == study.ANCHORS[0])
        & (frame["lr"] == study.LEARNING_RATES[0])
        & (~frame["skipped"].astype(bool))
    ]
    rows = []
    for cutoff in sorted(set(swept["cutoff_period_days"].dropna().astype(int))):
        cell = swept[swept["cutoff_period_days"] == cutoff]
        real = cell[cell["control"] == study.REAL]
        noise = cell[cell["control"] == study.NOISE]
        if real.empty:
            continue
        rows.append(
            {
                "cutoff_period_days": cutoff,
                "cof": int(real["cof"].iloc[0]),
                "dead_row_fraction": float(real["dead_row_fraction"].iloc[0]),
                "n_folds": int(real["fold"].nunique()),
                "direction": float(real["direction"].mean()),
                "direction_reference": float(real["direction_reference"].mean()),
                "edge_points": float(
                    (real["direction"] - real["direction_reference"]).mean() * 100
                ),
                "mae": float(real["mae"].mean()),
                "flatness": float(real["flatness"].mean()),
                "sharpe": float(real["sharpe"].mean()),
                "total_return": float(real["total_return"].mean()),
                "noise_direction": (
                    float(noise["direction"].mean()) if not noise.empty else math.nan
                ),
                "real_minus_noise_points": (
                    float((real["direction"].mean() - noise["direction"].mean()) * 100)
                    if not noise.empty
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


# ── the figures ──────────────────────────────────────────────────────────────


def boxplots(frame: pd.DataFrame, out: Path, dpi: int = DEFAULT_DPI) -> Path:
    """Per-fold direction accuracy, one box per arm, with the always-long bar drawn on.

    The bar is the reference the direction column is read against, so it belongs on the
    figure rather than in its caption: a box sitting below a line is the finding.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = study.reportable(frame)
    real = real[real["anchor"] == study.ANCHORS[0]]
    real = real[real["cutoff_period_days"] == study.CUTOFFS[0]]
    arms = [
        (model, channels)
        for model, channels in sorted(
            {(m, c) for m, c in zip(real["model"], real["channels"], strict=True)}
        )
        if model != study.BUY_AND_HOLD
    ]
    series = [
        real[(real["model"] == m) & (real["channels"] == c)]["direction"].dropna()
        for m, c in arms
    ]
    labels = [f"{m}\n{c}" if c else m for m, c in arms]
    keep = [index for index, values in enumerate(series) if len(values)]

    figure, axes = plt.subplots(figsize=(1.6 * max(len(keep), 1) + 2, 4.2), dpi=dpi)
    figure.patch.set_facecolor(INK)
    axes.set_facecolor(PANEL)
    box = axes.boxplot(
        [series[index] for index in keep],
        tick_labels=[labels[index] for index in keep],
        patch_artist=True,
        medianprops={"color": PAPER, "linewidth": 1.4},
        whiskerprops={"color": MUTED},
        capprops={"color": MUTED},
        flierprops={"markeredgecolor": MUTED, "markersize": 3},
    )
    for patch, colour in zip(box["boxes"], _ramp(len(keep)), strict=True):
        patch.set_facecolor(colour)
        patch.set_edgecolor(HAIRLINE)

    bar = float(real["direction_reference"].mean())
    axes.axhline(bar, color=ORANGE, linewidth=1.2, linestyle="--")
    axes.annotate(
        f"always-long {bar:.4f}",
        xy=(0.99, bar),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
        color=ORANGE,
        fontsize=8,
    )
    axes.set_ylabel("direction accuracy", color=PAPER, fontsize=9)
    axes.tick_params(colors=MUTED, labelsize=8)
    for spine in axes.spines.values():
        spine.set_color(HAIRLINE)
    figure.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=INK)
    plt.close(figure)
    return out


def cof_plot(frame: pd.DataFrame, out: Path, dpi: int = DEFAULT_DPI) -> Path:
    """The COF sweep as a curve, real against its null control at every cutoff."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = cof_curve(frame)
    figure, axes = plt.subplots(figsize=(6.4, 4.0), dpi=dpi)
    figure.patch.set_facecolor(INK)
    axes.set_facecolor(PANEL)

    axes.plot(
        curve["cof"], curve["direction"], marker="o", color=RAMP[-2], label="real"
    )
    axes.plot(
        curve["cof"],
        curve["noise_direction"],
        marker="s",
        linestyle="--",
        color=MUTED,
        label="white noise",
    )
    axes.axhline(
        float(curve["direction_reference"].mean()),
        color=ORANGE,
        linewidth=1.2,
        linestyle=":",
        label="always-long",
    )
    axes.set_xscale("log", base=2)
    axes.set_xticks(curve["cof"])
    axes.set_xticklabels(
        [
            f"{cof}\ncutoff {cut}"
            for cof, cut in zip(curve["cof"], curve["cutoff_period_days"], strict=True)
        ]
    )
    axes.set_xlabel("retained bins (COF)", color=PAPER, fontsize=9)
    axes.set_ylabel("direction accuracy", color=PAPER, fontsize=9)
    axes.tick_params(colors=MUTED, labelsize=8)
    for spine in axes.spines.values():
        spine.set_color(HAIRLINE)
    legend = axes.legend(facecolor=PANEL, edgecolor=HAIRLINE, fontsize=8)
    for text in legend.get_texts():
        text.set_color(PAPER)
    figure.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=INK)
    plt.close(figure)
    return out


def _ramp(count: int) -> list[str]:
    """One colour per box, from the project's single data ramp."""
    if count <= 1:
        return [RAMP[-2]]
    steps = len(RAMP) - 2
    return [RAMP[round(index * steps / (count - 1))] for index in range(count)]


# ── the whole chapter ────────────────────────────────────────────────────────


def render(frame: pd.DataFrame, out_dir: Path, dpi: int = DEFAULT_DPI) -> Path:
    """Write the report and its figures. Returns the path of the markdown file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ranks = correlations(frame)
    lines = ["# GlassBox Trader — study results", "", *header(frame), ""]

    lines += [
        "## Summary",
        "",
        (
            "Every metric is a delta against **its own** reference "
            f"({', '.join(f'`{k}` → {v.label}' for k, v in stats.REFERENCES.items())})."
        ),
        "",
        _md(summary(frame)),
        "",
        (
            f"**Spearman(MAE, flatness) = {ranks['flatness']:+.3f}** against "
            f"**Spearman(MAE, direction) = {ranks['direction']:+.3f}** — MAE across these "
            "arms is close to a monotone function of how flat the forecast is and carries "
            "almost no information about accuracy (§7.3). `flatness` is the column beside "
            "it."
        ),
        "",
        "## Every claim, at every anchor and against both nulls",
        "",
    ]
    for claim in CLAIMS:
        lines += [
            f"### {claim.name}",
            "",
            f"*{claim.units}*",
            "",
            _md(claim_table(frame, claim)),
            "",
        ]

    lines += ["## The COF sweep", "", _md(cof_curve(frame)), ""]

    tests = stats.wilcoxon(study.reportable(frame))
    lines += [
        "## Paired Wilcoxon signed-rank",
        "",
        (
            f"**{len(tests)} tests in the family**, and the family is every test in this "
            "table. `p` is uncorrected and `p_holm` is Holm-Bonferroni over all of them, "
            "controlling the family-wise error rate. The exact two-sided test on 16 folds "
            "cannot return a p below **3.05e-5**, so no claim here can be significant "
            "past that however large its effect."
        ),
        "",
        _md(tests.drop(columns=["skipped"])),
        "",
    ]

    boxes = boxplots(frame, out_dir / "direction_by_arm.png", dpi)
    curve = cof_plot(frame, out_dir / "cof_sweep.png", dpi)
    lines += [
        "## Figures",
        "",
        f"![Per-fold direction accuracy by arm]({boxes.name})",
        "",
        f"![The COF sweep]({curve.name})",
        "",
    ]

    path = out_dir / REPORT_FILE
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _md(frame: pd.DataFrame) -> str:
    """A markdown table, rounded so six decimals of noise do not read as precision.

    Hand-rolled rather than ``DataFrame.to_markdown``, which needs `tabulate` - a new
    dependency for table punctuation, against CLAUDE.md §4's preference for what is
    already declared.
    """
    rounded = frame.round(4)
    columns = [str(name) for name in rounded.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for values in rounded.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    return "\n".join(lines)


def _cell(value: object) -> str:
    """One cell. An empty cell for a missing number, never the string ``nan``."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code rather than raising."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=study.RESULTS_FILE)
    parser.add_argument("--out", default="report")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    args = parser.parse_args(argv)

    try:
        frame = load(args.results)
        path = render(frame, Path(args.out), args.dpi)
    except (FileNotFoundError, ValueError) as failure:
        print(f"report: {failure}", file=sys.stderr)
        return 2
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CLAIMS",
    "REPORT_FILE",
    "SUMMARY_METRICS",
    "Claim",
    "boxplots",
    "claim_table",
    "cof_curve",
    "cof_plot",
    "correlations",
    "header",
    "load",
    "main",
    "render",
    "summary",
]
