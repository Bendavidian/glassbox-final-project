"""X: does the gross-exposure cap ever bind, and in how many folds?

**Why this module exists rather than a notebook.** DECISIONS.md carries "average gross
exposure 16.00% of equity (peak 51%)", produced by a reconstruction that exists nowhere in
this repository. A reported number whose producing code cannot be rerun is the defect this
project is named for, so the fold count below is produced here, by code someone else can
run, and the method is printed with the numbers rather than described in a message.

**What is measured.** Every call the test-period backtest makes to the position sizer, for
the arm ``model.active`` names, over the configured fold grid. At each call the three terms
of :func:`glassbox.engine.risk.room_for` are recorded - the per-position cap ``A``, the
gross headroom ``B``, and the cash ``C`` - together with the returned notional and which
term was the minimum. Nothing here decides anything: :func:`~glassbox.engine.risk.
recording_sizer` returns exactly what ``position_sizer`` would return, and this module
proves it by running every fold **twice**, once instrumented and once not, and comparing
the metrics tables bit for bit. If they differ the run aborts.

**Why counting blocked entries alone would be wrong.** A cap that *reduces* an entry
returns a **positive** notional and looks like an ordinary trade; only a cap that *blocks*
one returns zero. An instrument that watched for zero notionals would report "the cap
rarely binds" and be wrong in the direction that flatters the risk layer. The reduced case
is counted first here, and the two are never merged.

**Ties are recorded, not resolved.** ``A == B`` exactly when ``gross_exposure`` is
``0.40 * equity`` under the configured 0.10 and 0.50 - four full-size positions, the
arithmetic boundary at which the gross cap starts to matter. Every fifth full-size entry
lands on it. The tie policy is applied after recording, stated in the output, and its
count reported separately, so a reader can see how much of the headline rests on it.

**What the sizer cannot see.** ``gross_exposure`` is the exposure *before* each entry, so
the peak reported is the peak **observed at a sizing call** and is a lower bound on the
true peak: exposure also moves as open positions are marked between entries, and no sizer
is consulted then.

Usage::

    python -m glassbox.experiments.exposure              # the configured fold grid
    python -m glassbox.experiments.exposure --folds 4    # a shorter run
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from glassbox import smoke_offline
from glassbox.config.loader import Config, load_config
from glassbox.engine import risk

#: Where the per-call records land. One row per sizing call, so the counts below can be
#: recomputed by anyone without rerunning the folds.
RECORDS_FILE = "report/gross_exposure.csv"

#: The tie policy, applied AFTER recording and stated in the output. `A == B` means the
#: gross headroom is exactly equal to the per-position cap: the gross term is jointly the
#: minimum, so the cap is at its binding boundary but has not yet reduced anything. Counted
#: as NOT binding, which is the conservative choice for a headline that claims it binds.
TIE_COUNTS_AS_BINDING = False


def measure(
    cfg: Config, n_folds: int, log=lambda message: None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[int, ...]]:
    """Run the folds twice; return ``(records, instrumented, control, fold numbers)``.

    The two tables are the per-fold metrics ``smoke_offline.fold_table`` builds. They are
    returned rather than compared here so the caller can show the comparison it made.

    **The fold numbers are returned separately and this is not tidiness.** A fold whose
    band stood aside makes *no sizing call at all*, so it contributes no rows and would
    vanish from a fold list derived from the records - taking the denominator of the
    headline with it and turning "the cap bound in 1 of 16 folds" into "1 of 1". A fold
    that never sized is an observation, not an absence.
    """
    arm = cfg.model.active
    bars, frames, folds = smoke_offline.prepare_folds(cfg, n_folds, log)

    rows: list[dict] = []
    watched: list[smoke_offline.ArmRun] = []
    control: list[smoke_offline.ArmRun] = []

    for fold in folds:
        seen: list[risk.RoomDetail] = []
        watched.append(
            smoke_offline.run_arm(
                arm,
                fold,
                frames,
                bars,
                cfg,
                backtest_sizer=risk.recording_sizer(seen),
            )
        )
        control.append(smoke_offline.run_arm(arm, fold, frames, bars, cfg))
        for order, detail in enumerate(seen):
            rows.append(
                {
                    "fold": fold.number,
                    "arm": arm,
                    "call": order,
                    "equity": detail.equity,
                    "gross_exposure": detail.gross_exposure,
                    "gross_fraction": (
                        detail.gross_exposure / detail.equity
                        if detail.equity
                        else float("nan")
                    ),
                    "term_a_per_position": detail.per_position,
                    "term_b_gross_headroom": detail.gross_headroom,
                    "term_c_cash": detail.cash,
                    "notional": detail.notional,
                    "case": detail.case,
                }
            )
        log(f"fold {fold.number}: {len(seen)} sizing calls")

    return (
        pd.DataFrame(rows, columns=list(_COLUMNS)),
        smoke_offline.fold_table(watched),
        smoke_offline.fold_table(control),
        tuple(int(fold.number) for fold in folds),
    )


_COLUMNS = (
    "fold",
    "arm",
    "call",
    "equity",
    "gross_exposure",
    "gross_fraction",
    "term_a_per_position",
    "term_b_gross_headroom",
    "term_c_cash",
    "notional",
    "case",
)


def identical(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Are the two metrics tables the same to the bit?

    ``DataFrame.equals`` compares dtype and value and treats NaN as equal to NaN, which is
    what is wanted here - a fold that stands aside writes NaN into the Sharpe column in
    both runs and that is agreement, not disagreement.
    """
    return left.equals(right)


def report(records: pd.DataFrame, cfg: Config, folds: Sequence[int]) -> str:
    """The counts, with the method attached. Pure: builds a string, writes nothing.

    ``folds`` is every fold that RAN, not every fold that produced a record - see
    :func:`measure`. A fold that stood aside is reported with zero calls rather than
    dropped, because dropping it would shrink the headline's denominator.
    """
    cap = cfg.risk.max_gross_exposure
    per_position = cfg.risk.max_position_pct
    ceiling = len(cfg.universe) * per_position

    reduced = records[records.case == risk.REDUCED_BY_GROSS]
    blocked = records[records.case == risk.BLOCKED_BY_GROSS]
    other = records[records.case == risk.BLOCKED_BY_OTHER]
    ties = records[records.case == risk.TIE]
    no_equity = records[records.case == risk.NO_EQUITY]

    binding_cases = [risk.REDUCED_BY_GROSS, risk.BLOCKED_BY_GROSS]
    if TIE_COUNTS_AS_BINDING:
        binding_cases.append(risk.TIE)
    bound = records[records.case.isin(binding_cases)]

    folds = sorted(int(n) for n in folds)
    bound_folds = sorted(int(n) for n in bound.fold.unique())
    silent = [n for n in folds if n not in set(records.fold.astype(int))]

    out: list[str] = []
    add = out.append

    add("GROSS-EXPOSURE CAP: DID IT BIND, AND IN HOW MANY FOLDS")
    add("=" * 78)
    add("")
    add("METHOD")
    add("-" * 78)
    add(
        f"  Every call the test-period backtest made to the position sizer, for arm"
        f" {cfg.model.active!r}\n"
        f"  over {len(folds)} folds. At each call the three terms of `risk.room_for` were"
        f" recorded:\n"
        f"    A = equity * max_position_pct      ({per_position})\n"
        f"    B = equity * max_gross_exposure - gross_exposure   (cap {cap})\n"
        f"    C = equity - gross_exposure        (cash)\n"
        f"  with the returned notional and which term was the minimum. `room_for` returns"
        f"\n  `max(0, min(A, B, C))` and discards which term bound; that defect is NOT"
        f" fixed here.\n"
        f"  `risk.recording_sizer` observes only - it returns what `position_sizer`"
        f" returns, and\n  every fold was run TWICE, instrumented and not, with the"
        f" metrics tables compared."
    )
    add("")
    add(
        f"  TIE POLICY: A == B exactly (gross_exposure == "
        f"{cap - per_position:.2f} * equity, i.e."
        f" {int((cap - per_position) / per_position)} full-size\n"
        f"  positions) is counted as "
        f"{'BINDING' if TIE_COUNTS_AS_BINDING else 'NOT BINDING'}. Ties are recorded, not"
        f" resolved: the raw\n  terms are in {RECORDS_FILE} and the count is reported"
        f" separately below."
    )
    add("")
    add(
        "  PEAK is the peak gross fraction OBSERVED AT A SIZING CALL, a lower bound on"
        " the\n  true peak: exposure also moves as open positions are marked, and no"
        " sizer is\n  consulted then."
    )
    add("")
    add("PER FOLD")
    add("-" * 78)
    add(
        f"  {'fold':>4} {'calls':>6} {'bound':>6} {'reduced':>8} {'blocked':>8}"
        f" {'tie':>5} {'peak gross':>11}"
    )
    for fold in folds:
        f = records[records.fold == fold]
        if f.empty:
            # The band stood aside: no entry was proposed, so the sizer was never asked.
            # Printed rather than skipped - "no calls" and "no binding" are different
            # facts and merging them would overstate what the cap was tested against.
            add(
                f"  {fold:>4} {0:>6} {'-':>6} {'-':>8} {'-':>8} {'-':>5} {'stood aside':>11}"
            )
            continue
        add(
            f"  {fold:>4} {len(f):>6}"
            f" {'yes' if len(f[f.case.isin(binding_cases)]) else 'no':>6}"
            f" {len(f[f.case == risk.REDUCED_BY_GROSS]):>8}"
            f" {len(f[f.case == risk.BLOCKED_BY_GROSS]):>8}"
            f" {len(f[f.case == risk.TIE]):>5}"
            f" {f.gross_fraction.max():>10.4f}"
        )
    add("")
    add("ACROSS FOLDS")
    add("-" * 78)
    add(f"  sizing calls recorded              : {len(records)}")
    add(
        f"  folds that made no sizing call     : {len(silent)}"
        f"{'   ' + str(silent) if silent else ''}"
    )
    add("    (band stood aside; counted in the denominator, never as 'did not bind')")
    add("")
    add(
        f"  HEADLINE - folds where the gross   : {len(bound_folds)} of {len(folds)}"
        f"{'   ' + str(bound_folds) if bound_folds else ''}"
    )
    add("    cap was the binding term at least")
    add("    once (reduced or blocked)")
    add("")
    add(f"  REDUCED_BY_GROSS  (B min, > 0)     : {len(reduced)}")
    add(f"  BLOCKED_BY_GROSS  (B min, == 0)    : {len(blocked)}")
    add(f"  BLOCKED_BY_OTHER  (A or C min, ==0): {len(other)}   <- NOT the risk layer")
    add(f"  TIE               (A == B exactly) : {len(ties)}")
    add(f"  NO_EQUITY         (early return)   : {len(no_equity)}")
    add(
        f"  UNCONSTRAINED                      : "
        f"{len(records[records.case == risk.UNCONSTRAINED])}"
    )
    add("")
    peak = records.gross_fraction.max()
    add(f"  peak gross fraction observed       : {peak:.4f}")
    add(f"  mean gross fraction at a call      : {records.gross_fraction.mean():.4f}")
    if peak > cap:
        add("")
        add(
            f"  NOTE: the peak EXCEEDS the {cap} cap, and that is correct rather than a\n"
            f"  violation. The cap is enforced at ENTRY - it bounds what may be added -\n"
            f"  while `gross_exposure` is the marked value of positions already open. A\n"
            f"  book entered at the cap that then rises in value carries gross above it,\n"
            f"  and no rule sells to get back under. Worth stating because a reader who\n"
            f"  sees {peak:.4f} against a {cap} cap will otherwise read it as a breach."
        )
    add("")
    add("THE ARITHMETIC THIS TESTS")
    add("-" * 78)
    add(
        f"  top_k is {cfg.signal.top_k} and max_position_pct is {per_position}, so new"
        f" entries per bar cap at\n"
        f"  {cfg.signal.top_k * per_position:.2f} and the gross cap can only bind through"
        f" ACCUMULATION across overlapping\n"
        f"  holds. What universe size changes is the CEILING on concurrent holds:"
        f" {len(cfg.universe)} symbols\n"
        f"  allow up to {ceiling:.2f} against a cap of {cap}"
        + (
            ", so the cap can be exceeded and is a real\n  constraint."
            if ceiling > cap
            else ", so the cap can be TOUCHED BUT NEVER EXCEEDED\n"
            "  and is very nearly decorative."
        )
    )
    add(
        "  The drivers are therefore concurrent-hold count and hold duration, not entries"
        "\n  per bar."
    )
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code rather than raising."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--out", default=RECORDS_FILE)
    args = parser.parse_args(argv)

    cfg = load_config()
    n_folds = cfg.walkforward.max_folds if args.folds is None else args.folds

    try:
        records, watched, control, folds = measure(cfg, n_folds, log=print)
    except smoke_offline.SmokeError as failure:
        print(f"exposure: {failure}", file=sys.stderr)
        return 2

    print()
    print("INSTRUMENTATION CHANGED NOTHING - the proof, not the assertion")
    print("-" * 78)
    same = identical(watched, control)
    print(f"  metrics tables identical (instrumented vs control): {same}")
    if not same:
        print(
            "  ABORTING. The instrumentation altered a result, so every count below "
            "would be\n  a measurement of the instrument. Nothing was written.",
            file=sys.stderr,
        )
        return 1
    print(f"  compared {len(watched)} fold rows x {len(watched.columns)} columns")
    print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.out, index=False)
    print(report(records, cfg, folds))
    print()
    print(f"wrote {args.out}: {len(records)} sizing calls")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "RECORDS_FILE",
    "TIE_COUNTS_AS_BINDING",
    "identical",
    "measure",
    "report",
]
