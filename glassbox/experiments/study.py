"""X: the study grid runner - models x feature configs over walk-forward folds with fixed
seeds, producing results.csv.

The FITS x C2_hybrid cell is deliberately empty (spec 6.4), and it is written into the CSV
as a **skipped row carrying its reason** rather than left absent: a blank cell in a results
table reads as a run that failed, and this one is a design decision.

**Four axes were added after spec 7.4 was written**, each because a measurement forced it,
and each is a column here:

``anchor``
    Three fold-grid offsets. A one-week shift once took a correlation from -0.47 to
    -0.0025 (GB-49's grid-sensitivity ruling, 18 Aug).
``lr``
    The learning rate is an arm of the study, not a constant of it: 71% of DLinear's
    scaled weights sat below Adam's step at ``1e-3``, and the MAE ordering **reverses**
    as the rate falls (20 Aug).
``flatness``
    Immediately beside ``mae``, per 7.3. Across arms MAE is close to a monotone function
    of it - Spearman +0.81 - and carries almost no information about accuracy.
``control``
    ``real``, ``shuffled`` or ``noise``. **The null control, and it is not optional.**

**Why both grid sensitivity and a null control, and why neither alone.** They catch
different failures, and this project has one demonstration of each on its own data. The
``r = -0.47`` timing correlation **died to a grid shift** and would have passed a null
control. The FITS phase advance **died to a null control** and passed grid sensitivity at
**48 of 48 cells**, more stably than the correlation that vanished. A claim that survives
one test and not the other is not a result.

**The null arms are in this file and must never be averaged with the real ones.**
:func:`reportable` is the single place that says which rows may reach a metric - the same
shape as ``records.is_reportable``, which solved this exact problem for replay and
rehearsal provenance. One artefact and one gate beats two artefacts that can drift apart
in columns or in vintage, and the null rows are only meaningful *beside* their real
counterparts anyway: splitting them across files turns the comparison into a join.

**Determinism.** Every arm is seeded from ``meta.seed``, the null controls included, so
re-running the same grid reproduces identical numbers. ``data_snapshot_last_bar`` records
the cache vintage the whole grid ran against (GB-4: the cache is a snapshot and never
refreshes itself), so two runs a month apart are distinguishable in the file rather than
by memory.

Implemented in GB-49, with the COF sweep in GB-50 and paired Wilcoxon tests in GB-51.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from glassbox.backtest import metrics
from glassbox.backtest.walkforward import make_folds
from glassbox.config.loader import (
    MODEL_SHAPING_SECTIONS,
    Config,
    config_hash,
    load_config,
    model_config_hash,
)
from glassbox.data.historical import LOG_RETURN
from glassbox.features.builder import build_feature_frame
from glassbox.model import ALL_FORECASTERS, fits
from glassbox.smoke_offline import (
    SmokeError,
    _common_index,
    load_cached_bars,
    run_arm,
    run_buy_and_hold,
)

# ── the axes ─────────────────────────────────────────────────────────────────

# **Derived, because this was a third copy of the registry** (24 Aug 2026). It read
# ``("persistence", "dlinear", "fits")`` by hand, which agreed with the registry on the
# day it was written and would have gone on agreeing until a model was registered - at
# which point the study would have quietly run three arms and omitted the fourth. That
# is the GB-44 defect exactly: the runner that produces every study number unable to
# select the model the study is about. ``dict`` preserves insertion order, so the arm
# order is the registration order rather than an accident.
MODELS = tuple(ALL_FORECASTERS)
CHANNEL_SETS = ("C0_base", "C2_hybrid")

# Spec 6.4: feeding pre-filtered wavelet bands to a model whose first act is to filter
# frequencies is redundant. Recorded in the output, not left blank.
SKIPPED = {
    ("fits", "C2_hybrid"): (
        "deliberately empty (spec 6.4): FITS is univariate and filters frequencies "
        "itself, so pre-filtered wavelet bands are redundant"
    ),
    # GB-66. The same ruling, and the case is stronger rather than merely analogous:
    # `C2_hybrid` adds wav_a1..a3, which are causal rolling DWT bands, to a model whose
    # first act is a causal DWT. That is not redundancy of the FITS kind - a filter fed
    # pre-filtered input - but the *same transform applied twice*, and the second pass
    # would decompose bands that are already single-scale. Filling the cell to square the
    # table is exactly what spec 6.4 says not to do.
    ("wits", "C2_hybrid"): (
        "deliberately empty (spec 6.4): WITS is univariate and its first act is a DWT, so "
        "feeding it the wav_a1..a3 DWT bands applies the same transform twice"
    ),
}

# Trading days. One third of `step_months` at the configured 3, which is the offset the
# 18 Aug ruling fixed.
ANCHORS = (0, 21, 42)

LEARNING_RATES = (1e-3, 1e-4, 1e-5)

# Spec 7.4's axis 3, and the centre is the configured value - asserted against
# ``fits.cutoff_period_days`` by a test rather than kept equal by hand.
#
# **FITS only, and the restriction is not an economy.** The cutoff is the one FITS
# hyperparameter; it reaches persistence and DLinear through nothing at all, so running
# them at cutoff 20 would produce rows identical to their rows at cutoff 5 and invite a
# reader to average a duplicate. See `Condition.models`.
CUTOFFS = (5, 2, 10, 20)

# Mean seconds per arm-fold, measured over the 20 Aug grid (679 rows). They span 60x, so
# a flat mean over the arms mis-estimates any design whose mix of arms differs from that
# one - which every COF spoke does, being FITS-only.
#
# **This one cannot be derived - the values are measurements - so a test pins it
# instead.** A model registered without a timing here would not fail; it would silently
# under-report the wall time of every plan containing it, which is the estimate the
# ten-minute rule depends on.
PER_FOLD_SECONDS = {
    "persistence": 0.39,
    "dlinear": 1.46,
    "fits": 4.16,
    # GB-66. **Anchored rather than measured on the grid, and the difference is stated
    # because it matters for the ten-minute rule.** WITS has no grid run yet, so this is
    # FITS's grid figure scaled by a ratio measured locally on a 2,505-window fold with a
    # warmup run discarded: fits 2.40s (sd 0.11) against wits 2.75s (sd 0.07) over four
    # repeats, a ratio of 1.146. Replace it with the grid's own number when GB-66's arm
    # runs - the acceptance criteria ask for the wall time reported either way.
    #
    # **Slower than FITS on fewer parameters**, which is worth not being surprised by: 882
    # reals against 1,200, but 64 epochs before early stopping against 51.
    "wits": 4.77,
    "buy_and_hold": 0.07,
}

REAL = "real"
SHUFFLED = "shuffled"
NOISE = "noise"
CONTROLS = (REAL, SHUFFLED, NOISE)

# The market reference, written as its own rows rather than as a column on every arm.
# It makes no forecast, so its `mae` and `direction` cells are empty - the treatment
# persistence's direction column gets, and for the same reason.
BUY_AND_HOLD = "buy_and_hold"

RESULTS_FILE = "results.csv"

#: The dated daily equity curve, written beside the results table. **A grid run used to
#: produce exactly one artefact and it was per-fold**, so nothing downstream could ask what
#: happened on a given day - `metrics.ArmResult.equity` is a dated daily series and it was
#: discarded when the run ended. GB-63b's calendar and cumulative-equity cards need it, and
#: `dashboard` sits below `experiments` in the layers contract so it cannot compute one.
#:
#: **Reference condition only.** Every arm-fold of the full grid would be roughly a
#: hundred times `results.csv` for a curve nobody plots; the reference condition is the one
#: the report's headline numbers come from, and a card that plotted a COF spoke would be
#: answering a question nobody asked.
DAILY_EQUITY_FILE = "report/daily_equity.csv"

DAILY_EQUITY_COLUMNS = (
    "anchor",
    "lr",
    "control",
    "target_in_loop",
    "cutoff_period_days",
    "model",
    "channels",
    "fold",
    "date",
    "equity",
)

# The columns that identify one condition, and therefore the columns a paired test may
# not cross. GB-51 pairs by fold *within* a condition: fold 3 at anchor 21 and fold 3 at
# anchor 0 are different windows, and pairing them would compare two periods rather than
# two arms. Derived from `Condition.columns` below rather than listed again - see its
# docstring for what a second list of the axes costs.

# `mae` and `flatness` are adjacent and in this order, per 7.3. The pairing is expressed
# here, in the one place that builds the row, rather than remembered at each table.
COLUMNS = (
    "anchor",
    "lr",
    "control",
    # Execution fidelity: whose target it is. Beside the other axes because it identifies
    # a condition, and in PAIR_KEYS for the same reason - pairing a loop-side fold against
    # a broker-side one would compare two execution models rather than two arms.
    "target_in_loop",
    # The COF axis and the two quantities it decides, adjacent for the same reason `mae`
    # and `flatness` are: `cutoff_period_days` is a number nobody can interpret without
    # them. `cof` is the retained bin count and `dead_row_fraction` the share of the layer
    # that bin 0 holds and can never move - both NaN for an arm that is not FITS, since
    # they are FITS geometry and not study settings.
    "cutoff_period_days",
    "cof",
    "dead_row_fraction",
    "model",
    "channels",
    "fold",
    "skipped",
    "reason",
    "stood_aside",
    "n_windows",
    "n_trades",
    "direction",
    "direction_reference",
    "mae",
    "flatness",
    "rmse",
    "cancellation",
    "total_return",
    "sharpe",
    "max_drawdown",
    "seconds",
    "config_hash",
    "model_config_hash",
    "data_snapshot_last_bar",
)


@dataclass(frozen=True)
class Condition:
    """One point in the sensitivity design: a grid anchor, a rate and a control."""

    anchor: int
    lr: float
    control: str
    cutoff: int = CUTOFFS[0]
    target_in_loop: bool = True
    """Whether the take-profit is the loop's or the broker's. **Default `True`, and that
    is a ruling rather than a preference** (25 Aug 2026).

    Alpaca refuses every multi-leg order class on a fractional quantity - `bracket` and
    `oco` both return `{"code":42210000,"message":"fractional orders must be simple
    orders"}`, measured - so a fractional position carries one broker-side protective
    order and it is the stop. The target is therefore evaluated by the loop against
    completed daily bars and fills at the next open.

    `False` is the classical backtest and it is a **spoke**, not the centre. It is not a
    fantasy - whole-share sizing permits brackets - but it is a different design point
    with its own cost, so it is reported as the comparison rather than as the headline.
    A centre of `False` would mean every published number described a system that cannot
    be built at this venue while the caveat lived where nobody reads it.
    """

    @property
    def columns(self) -> dict[str, object]:
        """The columns that identify this condition in a results row. **One list.**

        Every writer of a results row derives from this - `_row`, `_skipped_row`, and any
        fixture that builds a table with the shape the grid writes. On 26 Aug 2026 the
        report fixture re-listed the axes by hand, so `target_in_loop` was absent from
        every synthetic row; the spoke and the centre then differed in no column at all,
        collapsed into one group, and the paired tests read 32 folds where there are 16.
        Each list was internally consistent, which is why nothing caught it until the
        pairing crashed - the two-places family, with the second place in a test.

        `cof` and `dead_row_fraction` are not here: they are FITS geometry derived from
        the config that actually ran, and `_geometry` owns them.
        """
        return {
            "anchor": self.anchor,
            "lr": self.lr,
            "control": self.control,
            "target_in_loop": self.target_in_loop,
            "cutoff_period_days": self.cutoff,
        }

    @property
    def is_reference(self) -> bool:
        return (
            self.anchor == 0
            and self.lr == LEARNING_RATES[0]
            and self.control == REAL
            and self.cutoff == CUTOFFS[0]
            and self.target_in_loop
        )

    @property
    def models(self) -> tuple[str, ...]:
        """Which models this condition runs.

        Every model at the centre cutoff; **FITS alone on a COF spoke**, because the
        cutoff is FITS's one hyperparameter and does not reach the other two. A
        persistence row at cutoff 20 would be identical to its row at cutoff 5 in every
        column that is a result, and a duplicate in a results file is something a reader
        eventually averages.
        """
        return MODELS if self.cutoff == CUTOFFS[0] else ("fits",)


#: The columns a paired test may not cross - which is exactly the set that identifies a
#: condition, so it is read off :attr:`Condition.columns` rather than written again. An
#: axis added to `Condition` therefore joins the pairing by construction; the alternative
#: is a second list that stays correct until somebody adds the axis to only one of them.
PAIR_KEYS: tuple[str, ...] = tuple(
    Condition(anchor=ANCHORS[0], lr=LEARNING_RATES[0], control=REAL).columns
)


@dataclass(frozen=True)
class Plan:
    """What a run will do, before it does it."""

    conditions: tuple[Condition, ...]
    folds: int

    @property
    def arms(self) -> int:
        """Live arms at the centre. A COF spoke runs one; see :attr:`Condition.models`."""
        return len(live_arms(MODELS))

    @property
    def cells(self) -> int:
        return sum(len(live_arms(condition.models)) for condition in self.conditions)

    @property
    def trainings(self) -> int:
        """Arm-folds. The market reference is counted by :meth:`seconds`, not here."""
        return self.cells * self.folds

    def seconds(self) -> float:
        """A wall-time estimate from **measured** per-arm costs, not from a flat mean.

        See :data:`PER_FOLD_SECONDS`. The market reference is included because it runs
        once per condition and is not free. An estimate rather than a promise: it is here
        so a run of this size is a decision rather than a surprise.
        """
        arms_cost = sum(
            PER_FOLD_SECONDS[model]
            for condition in self.conditions
            for model, _ in live_arms(condition.models)
        )
        market = len(self.conditions) * PER_FOLD_SECONDS[BUY_AND_HOLD]
        return (arms_cost + market) * self.folds


def conditions(full: bool = False) -> tuple[Condition, ...]:
    """The sensitivity design.

    **A star, not a cross product, and the choice is stated because it is a real one.**
    The full cross of four axes - three anchors, three rates, three controls and four
    cutoffs - is **108 conditions, 243 live cells, 3,888 arm-folds**, and it has never
    been run, so it has no measured wall time. The star is **14 conditions, 54 live cells,
    864 arm-folds**, measured at **100 min 56 s** on the twenty-symbol universe (2 Sep
    2026, ``da37401``; PROGRESS's GB-61 row). What the extra 94 conditions buy is
    *interactions* - does the learning-rate effect differ at anchor 42 under shuffled
    returns - and nobody asked that question. What every axis needs is a **common
    reference to depart from**, which is what a star gives: one departure per axis, each
    comparable against the same centre.

    **CORRECTED 23 Sep 2026.** This read *"27 conditions and about 2.2 hours; the star is
    7 and about 35 minutes"*, and spec 7.4 carried the same pair. Both predated the COF
    axis (GB-50) and the ``target_in_loop`` spoke, so both counted three axes where the
    code builds four - and **each was internally consistent, so the two agreed with each
    other and not with the function three lines below them.** Cells rather than
    ``conditions x arms`` throughout, because a COF spoke runs FITS alone; multiplying
    gives the wrong number and gives it in the flattering direction.

    ``full=True`` runs the cross product, for the day somebody does want an interaction.
    """
    if full:
        return tuple(
            Condition(anchor, lr, control, cutoff)
            for anchor in ANCHORS
            for lr in LEARNING_RATES
            for control in CONTROLS
            for cutoff in CUTOFFS
        )
    reference = Condition(ANCHORS[0], LEARNING_RATES[0], REAL, CUTOFFS[0])
    return (
        reference,
        *(replace(reference, anchor=anchor) for anchor in ANCHORS[1:]),
        *(replace(reference, lr=lr) for lr in LEARNING_RATES[1:]),
        *(replace(reference, control=control) for control in CONTROLS[1:]),
        # **Each COF spoke carries its own null control** (ruled 23 Aug). A cutoff that
        # scores the same on white noise as on real data is the finding *for that cutoff*,
        # and testing only the centre would leave three of the four unfalsifiable.
        *(
            replace(reference, cutoff=cutoff, control=control)
            for cutoff in CUTOFFS[1:]
            for control in (REAL, NOISE)
        ),
        # **The execution-fidelity spoke.** `target_in_loop=False` is the broker-side
        # target every published number in this study assumed before 25 Aug 2026. It is
        # kept as a departure from the centre so the report can state the difference,
        # which is the chapter's turning quantity - not because it describes a system
        # this venue supports for fractional sizing.
        replace(reference, target_in_loop=False),
    )


def arms(models: Sequence[str] = MODELS) -> tuple[tuple[str, str], ...]:
    """Every ``(model, channel set)`` pair for ``models``, skipped cells included."""
    return tuple((model, channels) for model in models for channels in CHANNEL_SETS)


def live_arms(models: Sequence[str] = MODELS) -> tuple[tuple[str, str], ...]:
    """The pairs that actually train: :func:`arms` less the deliberately empty cells."""
    return tuple(pair for pair in arms(models) if pair not in SKIPPED)


def plan(cfg: Config, full: bool = False, n_folds: int | None = None) -> Plan:
    """What :func:`run` will do, without doing it."""
    return Plan(
        conditions=conditions(full),
        folds=cfg.walkforward.max_folds if n_folds is None else n_folds,
    )


# ── the null controls ────────────────────────────────────────────────────────


def null_bars(bars: pd.DataFrame, control: str, seed: int) -> pd.DataFrame:
    """A synthetic market with the same return distribution and no structure.

    Args:
        bars: A canonical bar frame.
        control: :data:`REAL` returns it unchanged; :data:`SHUFFLED` permutes the
            log-return series; :data:`NOISE` replaces it with Gaussian draws of the
            **same standard deviation**.
        seed: Drawn from ``meta.seed`` by the caller, so a re-run reproduces the world.

    Returns:
        A frame of the same shape and index, with prices rebuilt from the new returns so
        that **the backtester trades the same world the model was fitted in**. Perturbing
        the features alone would leave the equity curve describing the real market and the
        forecasts describing a synthetic one, and every trading metric would be a
        comparison between two different universes.

    Raises:
        ValueError: ``control`` is not one of :data:`CONTROLS`.

    Matched variance rather than matched everything: the point is to remove the temporal
    structure while leaving the scale, so that an effect which survives is an effect that
    did not need the structure.
    """
    if control not in CONTROLS:
        raise ValueError(f"unknown control {control!r}; choose from {list(CONTROLS)}")
    if control == REAL:
        return bars

    rng = np.random.default_rng(seed)
    returns = bars[LOG_RETURN].to_numpy(dtype="float64", copy=True)
    usable = np.isfinite(returns)
    live = returns[usable]
    if control == SHUFFLED:
        replacement = rng.permutation(live)
    else:
        replacement = rng.normal(0.0, float(live.std()), size=live.size)

    rebuilt = returns.copy()
    rebuilt[usable] = replacement
    close = float(bars["close"].iloc[0]) * np.exp(
        np.cumsum(np.nan_to_num(rebuilt, nan=0.0))
    )

    scale = close / bars["close"].to_numpy(dtype="float64")
    synthetic = bars.copy()
    for column in ("open", "high", "low", "close"):
        synthetic[column] = bars[column].to_numpy(dtype="float64") * scale
    synthetic[LOG_RETURN] = rebuilt
    return synthetic


# ── the run ──────────────────────────────────────────────────────────────────


BLAS_THREADS = 1
"""Threads the grid pins BLAS to. **The default, measured rather than assumed.**

Thread count changes how a matmul splits its reduction, so it is one of the ways two
machines disagree in the last bits. Pinning removes it, and on 23 Aug 2026 it was measured
to cost nothing:

============  ===================================  =========
condition     runs (counterbalanced U P P U)       mean
============  ===================================  =========
unpinned      40.7 s, 36.8 s                       38.7 s
pinned        40.9 s, 37.7 s                       39.3 s
============  ===================================  =========

The difference between conditions is **0.5 s** against a spread **within** each condition
of 3.2-3.9 s, so it is not distinguishable from run-to-run noise. The plausible reason is
a fact about this study worth stating beside the parameter counts: **the models are too
small to parallelise.** DLinear holds 4,800 reals and FITS 1,200, and a ``(120, 4)`` by
``(120,)`` matmul does not saturate fourteen cores - thread dispatch costs about what the
parallelism returns.

**Pinning was also verified not to change the numbers**, which is what made defaulting it
safe: a pinned run reproduces an unpinned one bit-for-bit on this machine, so the
committed ``results.csv`` is still regenerated by the default command.

**What this does and does not buy.** It removes **one** source of cross-machine
divergence. It does not remove different instruction sets dispatching different BLAS
kernels, different BLAS builds, or a different platform ``libm`` for transcendentals -
single-threaded on two different CPUs can still differ. **Cross-architecture reproduction
remains untested**, and defaulting this must not be reported as making it
architecture-independent.
"""


def pin_threads(threads: int = BLAS_THREADS) -> int:
    """Pin BLAS to ``threads`` and return the count that was in force before.

    Called by :func:`main` rather than by :func:`run`, so a library caller's global torch
    state is never changed underneath it - a function that silently repinned the process
    would be a side effect nobody asked for. See :data:`BLAS_THREADS`.
    """
    import torch

    before = torch.get_num_threads()
    torch.set_num_threads(threads)
    return before


def data_snapshot(bars: Mapping[str, pd.DataFrame]) -> str:
    """The one date every cached symbol ends on, or a refusal naming the disagreement.

    **This was a ``max`` and the ``max`` was a lie** (GB-61, 24 Aug 2026). Expanding the
    universe from 5 to 20 fetched fifteen symbols on a day the committed five did not have,
    so the cache held two snapshot dates - 2026-08-13 for the incumbents and 2026-08-21 for
    the new names. ``_common_index`` intersects, so every fold would have been computed
    correctly on the shorter window; ``max`` would have written **2026-08-21** into
    ``data_snapshot_last_bar`` on all 877 rows, and that column is the provenance GB-57
    quotes. The folds would have been right and the label wrong, which is worse than both
    being wrong, because nothing downstream disagrees with itself.

    A ``max`` reports the newest and **hides** the disagreement, which makes it an
    instrument that cannot report the one fault it is positioned to see. Truncating the
    cache fixed 24 August; a refusal is what fixes the next symbol somebody adds.

    Raises:
        SmokeError: The cached symbols do not share a last bar, naming each group.
    """
    by_date: dict[str, list[str]] = {}
    for symbol, frame in bars.items():
        by_date.setdefault(frame.index[-1].date().isoformat(), []).append(symbol)
    if len(by_date) > 1:
        groups = "; ".join(
            f"{date}: {', '.join(sorted(symbols))}"
            for date, symbols in sorted(by_date.items())
        )
        raise SmokeError(
            "the cached symbols end on different dates, so this run has no single data "
            f"snapshot to record: {groups}. Every result row carries "
            "`data_snapshot_last_bar`, and one value cannot describe two cache states. "
            "Refetch the laggards or truncate the leaders to the common last bar."
        )
    return next(iter(by_date))


def run(
    cfg: Config,
    full: bool = False,
    n_folds: int | None = None,
    log=lambda message: None,
    design: Sequence[Condition] | None = None,
    daily: list[dict] | None = None,
) -> pd.DataFrame:
    """Every arm at every condition, one row per fold.

    Args:
        cfg: Resolved configuration. Supplies the universe, the fold plan and the seed.
        full: Run the cross product instead of the star. See :func:`conditions`.
        n_folds: Cap the folds, for a smoke run. ``None`` uses ``walkforward.max_folds``.
        log: Progress sink.
        design: Run these conditions instead of the star or the cross. For a test that
            needs one cell rather than the grid - determinism is a property of the seed
            reaching every stage, and it holds for one condition or for none, so
            asserting it over thirteen costs twelve grids and proves nothing more.

    Returns:
        A frame with :data:`COLUMNS`, including one skipped row per skipped cell per
        condition, carrying its reason.

    Raises:
        SmokeError: The cache is incomplete or yields no fold.
    """
    bars = load_cached_bars(cfg)
    snapshot = data_snapshot(bars)
    chosen = tuple(conditions(full) if design is None else design)
    sized = Plan(
        conditions=chosen,
        folds=cfg.walkforward.max_folds if n_folds is None else n_folds,
    )
    log(
        f"{sized.cells} cells over {sized.folds} folds = {sized.trainings} arm-folds, "
        f"about {sized.seconds() / 60:.0f} minutes"
    )

    rows: list[dict] = []
    for condition in chosen:
        rows.extend(
            _condition_rows(
                cfg, condition, bars, snapshot, n_folds=n_folds, log=log, daily=daily
            )
        )
    return pd.DataFrame(rows, columns=list(COLUMNS))


@dataclass(frozen=True)
class Provenance:
    """Whether a results file was produced by the configuration now on disk.

    **The same split the checkpoint gate uses, applied to a second artefact** (23 Aug
    2026). A change to a live-only key cannot move a number, so invalidating a 20-minute
    grid over one would be the spurious-retrain problem GB-25 already solved once - and a
    guard that fires spuriously is a guard somebody eventually weakens.

    Attributes:
        models_match: ``model_config_hash`` for the current configuration appears in the
            file. **False means the file describes different models** and nothing in it
            may be quoted.
        config_matches: ``config_hash`` for the current configuration appears in the file.
            False with ``models_match`` true means only non-model settings moved.
        candidates: The sections the difference must lie in, when there is one. Derived
            rather than measured: a hash cannot say what changed, but it can say what
            cannot have - anything in :data:`MODEL_SHAPING_SECTIONS` would have moved
            ``model_config_hash`` too.
    """

    models_match: bool
    config_matches: bool
    candidates: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.models_match and self.config_matches

    def warning(self) -> str:
        """One line for a report header, or empty when the file matches."""
        if self.ok:
            return ""
        if not self.models_match:
            return (
                "STALE: no row in this file was produced by the committed configuration's "
                "model settings. The models it describes are not the models this "
                "configuration builds, and no number in it may be quoted"
            )
        return (
            "The committed configuration has moved since this file was written, in a "
            "section that cannot change a model: the difference is in one of "
            f"{list(self.candidates)}, because anything under "
            f"{list(MODEL_SHAPING_SECTIONS)} would have moved `model_config_hash` too. "
            "Every number here still stands; the provenance stamp does not"
        )


def provenance(frame: pd.DataFrame, cfg: Config) -> Provenance:
    """Compare a results file's recorded hashes against the configuration on disk.

    **The reference cell reproduces the base configuration exactly**, which is what makes
    this checkable: at the centre condition the arm whose model, channels, rate and cutoff
    are the configured ones runs under `cfg` unchanged, so its hash *is*
    ``config_hash(cfg)``. If that hash is absent, this file was written by a different
    configuration - or the configuration was edited while the grid ran, which is how the
    23 Aug instance happened and which nothing else would have caught.
    """
    models = model_config_hash(cfg) in set(frame["model_config_hash"].dropna())
    full = config_hash(cfg) in set(frame["config_hash"].dropna())
    others = tuple(
        name
        for name in Config.__dataclass_fields__
        if name not in MODEL_SHAPING_SECTIONS
    )
    return Provenance(
        models_match=models,
        config_matches=full,
        candidates=() if full else others,
    )


def reportable(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows that may reach a metric: real data, and nothing skipped.

    **The single gate**, in ``records.is_reportable``'s shape and for its reason. A null
    arm is a measurement of the machine and not of the market, and averaging one into a
    result would produce a number that looks ordinary and means nothing. Filtering at each
    caller is how one of them eventually forgets.
    """
    return frame[(frame["control"] == REAL) & (~frame["skipped"].astype(bool))]


def _condition_rows(
    cfg: Config,
    condition: Condition,
    bars: dict[str, pd.DataFrame],
    snapshot: str,
    n_folds: int | None,
    log,
    daily: list[dict] | None = None,
) -> list[dict]:
    """Every arm of one condition.

    ``daily`` is an explicit sink rather than a second return value or a module-level
    collector: the caller decides whether the dated equity curves are wanted, and a reader
    of this signature can see that they leave through it. ``None`` collects nothing.
    """
    world = {
        symbol: null_bars(frame, condition.control, cfg.meta.seed + index)
        for index, (symbol, frame) in enumerate(sorted(bars.items()))
    }

    rows: list[dict] = []

    # **The market reference is run once per condition, not once per arm.** It does not
    # depend on the model, and both channel sets trim the same warm-up so both yield the
    # same fold grid - running it per arm would be five identical backtests and five
    # identical rows. It is a row rather than a discarded object because 7.3 reports
    # trading metrics against buy-and-hold and GB-52 regenerates from this file alone.
    reference = _cell_config(cfg, condition)
    market = {s: build_feature_frame(f, reference) for s, f in world.items()}
    for fold in _folds_or_raise(market, reference, condition.anchor, n_folds):
        held = run_buy_and_hold(fold, market, world, reference)
        rows.append(
            _row(condition, BUY_AND_HOLD, "", fold.number, held, snapshot, reference)
        )
        _collect_daily(daily, condition, BUY_AND_HOLD, "", fold.number, held)

    for model, channels in arms(condition.models):
        cell = replace(
            _cell_config(cfg, condition),
            model=replace(cfg.model, active=model, lr=condition.lr),
            channels=replace(cfg.channels, active=channels),
        )
        reason = SKIPPED.get((model, channels))
        if reason is not None:
            rows.append(
                _skipped_row(condition, model, channels, reason, snapshot, cell)
            )
            continue

        frames = {s: build_feature_frame(f, cell) for s, f in world.items()}
        folds = _folds_or_raise(frames, cell, condition.anchor, n_folds)

        started = time.perf_counter()
        for fold in folds:
            arm = run_arm(model, fold, frames, world, cell)
            rows.append(
                _row(condition, model, channels, fold.number, arm, snapshot, cell)
            )
            _collect_daily(daily, condition, model, channels, fold.number, arm)
        log(
            f"  anchor {condition.anchor:>2} lr {condition.lr:g} "
            f"cof@{condition.cutoff:<2} {condition.control:8} {model:11} {channels:9} "
            f"{len(folds)} folds in {time.perf_counter() - started:5.1f}s"
        )
    return rows


def _cell_config(cfg: Config, condition: Condition) -> Config:
    """The configuration one condition runs under.

    The cutoff goes through ``cfg.fits`` rather than being handed to the forecaster
    directly, so it reaches ``model_config_hash`` and two cells that differ only in the
    cutoff are distinguishable in the file by their hash alone.
    """
    return replace(
        cfg,
        model=replace(cfg.model, lr=condition.lr),
        fits=replace(cfg.fits, cutoff_period_days=condition.cutoff),
        backtest=replace(cfg.backtest, target_in_loop=condition.target_in_loop),
    )


def _geometry(model: str, cfg: Config) -> dict:
    """What the cutoff decides, for a FITS row; empty cells for anything else.

    Derived from ``model.fits`` rather than recomputed here - the fifth instance of the
    two-places family arrived from a copy that was locally correct, and ``COF = L // c``
    is exactly the kind of arithmetic that gets written down twice.

    ``cutoff_period_days`` is **not** here: it identifies the condition and
    :attr:`Condition.columns` writes it, so that a reader of the row and a paired test
    agree on which cell they are looking at without two sources for the same number.
    """
    if model != "fits":
        return {"cof": math.nan, "dead_row_fraction": math.nan}
    length, horizon = cfg.window.input_len, cfg.window.horizon
    cutoff = cfg.fits.cutoff_period_days
    return {
        "cof": fits.cof_for(length, cutoff),
        "dead_row_fraction": fits.dead_row_fraction(length, horizon, cutoff),
    }


def _folds_or_raise(frames, cfg: Config, anchor: int, n_folds: int | None):
    """The fold grid at one anchor, refusing an empty one rather than looping zero times."""
    folds = make_folds(_common_index(frames)[anchor:], cfg)
    if n_folds is not None:
        folds = folds[:n_folds]
    if not folds:
        raise SmokeError("no complete walk-forward fold fits the cached history")
    return folds


def _collect_daily(
    sink: list[dict] | None,
    condition: Condition,
    model: str,
    channels: str,
    fold: int,
    arm,
) -> None:
    """Append one row per bar of this arm-fold's equity curve, or do nothing.

    **Reference condition and real data only.** A null-control curve is a curve of a model
    trained on noise, which is meaningful in the results table beside its real twin and
    meaningless plotted on a calendar as though it were a month somebody lived through.
    """
    if sink is None or not condition.is_reference or condition.control != REAL:
        return
    equity = getattr(arm.result, "equity", None)
    if equity is None or len(equity) == 0:
        return
    columns = condition.columns
    for stamp, value in equity.items():
        sink.append(
            {
                **columns,
                "model": model,
                "channels": channels,
                "fold": fold,
                "date": pd.Timestamp(stamp).strftime("%Y-%m-%d"),
                "equity": float(value),
            }
        )


def _row(
    condition: Condition,
    model: str,
    channels: str,
    fold: int,
    arm,
    snapshot: str,
    cfg: Config,
) -> dict:
    result = arm.result
    # **An arm that makes no magnitude forecast reports no error**, per `ArmRun.forecasts`
    # and `fold_table`'s treatment of the same flag. Buy-and-hold calls up on every window,
    # so its DIRECTION is real and is the always-long bar by construction; its `predicted`
    # array is that call and not a log-return path, so filling `mae`, `rmse` or `flatness`
    # from it would report the error of a forecast it never made. The first run of this
    # grid did exactly that - MAE 1.0014 and flatness 53.0 on a series whose scale is
    # 0.015 - which is the failure the flag exists to prevent.
    forecast_error = arm.forecasts
    return {
        **condition.columns,
        **_geometry(model, cfg),
        "model": model,
        "channels": channels,
        "fold": fold,
        "skipped": False,
        "reason": "",
        "stood_aside": arm.calibration.stood_aside,
        "n_windows": 0 if result.actual is None else len(result.actual),
        "n_trades": len(result.strategy_trades),
        "direction": metrics.direction_accuracy(result),
        "direction_reference": metrics.always_long_accuracy(result),
        "mae": metrics.mae(result) if forecast_error else math.nan,
        "flatness": metrics.flatness(result) if forecast_error else math.nan,
        "rmse": metrics.rmse(result) if forecast_error else math.nan,
        "cancellation": arm.cancellation,
        "total_return": metrics.total_return(result),
        "sharpe": metrics.sharpe(result),
        "max_drawdown": metrics.max_drawdown(result),
        "seconds": arm.seconds,
        "config_hash": config_hash(cfg),
        "model_config_hash": model_config_hash(cfg),
        "data_snapshot_last_bar": snapshot,
    }


def _skipped_row(
    condition: Condition,
    model: str,
    channels: str,
    reason: str,
    snapshot: str,
    cfg: Config,
) -> dict:
    row = dict.fromkeys(COLUMNS, math.nan)
    row.update(
        {
            **condition.columns,
            **_geometry(model, cfg),
            "model": model,
            "channels": channels,
            "fold": -1,
            "skipped": True,
            "reason": reason,
            "config_hash": config_hash(cfg),
            "model_config_hash": model_config_hash(cfg),
            "data_snapshot_last_bar": snapshot,
        }
    )
    return row


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code rather than raising."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=RESULTS_FILE)
    parser.add_argument(
        "--daily-out",
        default=DAILY_EQUITY_FILE,
        help=(
            "where the dated daily equity curve is written, for the reference condition "
            f"on real data only (default: {DAILY_EQUITY_FILE})"
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="the cross product of every axis instead of the star; see `conditions`",
    )
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=BLAS_THREADS,
        help=(
            "threads to pin BLAS to. The default of 1 removes thread count as a source "
            "of cross-machine divergence and was measured to cost nothing - these models "
            "are too small to parallelise. Pass 0 to leave the process untouched. See "
            "`BLAS_THREADS`"
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print the cell count and the wall-time estimate, and run nothing",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.blas_threads:
        was = pin_threads(args.blas_threads)
        print(f"BLAS threads {was} -> {args.blas_threads}")
    design = plan(cfg, args.full, args.folds)
    # Not "conditions x arms": a COF spoke runs FITS alone, so the product would be wrong
    # and wrong in the flattering direction.
    print(
        f"{len(design.conditions)} conditions, {design.cells} cells "
        f"({design.arms} arms at the centre, FITS alone on a COF spoke), "
        f"{design.folds} folds each = {design.trainings} arm-folds, "
        f"about {design.seconds() / 60:.0f} minutes"
    )
    if args.plan_only:
        return 0

    daily: list[dict] = []
    try:
        table = run(cfg, full=args.full, n_folds=args.folds, log=print, daily=daily)
    except SmokeError as failure:
        print(f"study: {failure}", file=sys.stderr)
        return 2

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    real = reportable(table)
    print(
        f"wrote {args.out}: {len(table)} rows, {len(real)} reportable "
        f"({len(table) - len(real)} null-control or skipped)"
    )

    if daily:
        curves = pd.DataFrame(daily, columns=list(DAILY_EQUITY_COLUMNS))
        Path(args.daily_out).parent.mkdir(parents=True, exist_ok=True)
        curves.to_csv(args.daily_out, index=False)
        print(
            f"wrote {args.daily_out}: {len(curves)} daily rows over "
            f"{curves['date'].nunique()} distinct dates, reference condition only"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ANCHORS",
    "BLAS_THREADS",
    "BUY_AND_HOLD",
    "CHANNEL_SETS",
    "COLUMNS",
    "CONTROLS",
    "CUTOFFS",
    "DAILY_EQUITY_COLUMNS",
    "DAILY_EQUITY_FILE",
    "LEARNING_RATES",
    "MODELS",
    "NOISE",
    "PAIR_KEYS",
    "PER_FOLD_SECONDS",
    "REAL",
    "RESULTS_FILE",
    "SHUFFLED",
    "SKIPPED",
    "Condition",
    "Plan",
    "Provenance",
    "arms",
    "conditions",
    "live_arms",
    "main",
    "null_bars",
    "pin_threads",
    "plan",
    "provenance",
    "reportable",
    "run",
]
