# IDEAS PARKED

Ideas that are **not** in scope for this project. Recorded so they are visibly a
choice rather than an omission, and so they do not leak into the codebase mid-sprint.

**Rule: nothing here is implemented before Gate 3. No exceptions.**

---

## Analytic resampling initialisation — learn the residual, not the geometry

**Motivation, measured on real data 20 Aug 2026** (GB-45/GB-46, and the reason this is a
research idea rather than a hunch): **86% of FITS's learned frequency response is
reproduced by a model trained on white noise.** The gain curves correlate at **+0.9485**,
and gain tracks grid misalignment `|η·k − round(η·k)|` at **Spearman −0.9427,
p = 1.8e-11**. The complex layer is spending most of its 1,150 effective parameters
learning a **deterministic resampling operator** — how to move bin `k` of a length-`L`
grid onto a length-`L+H` one — leaving roughly **14%** for anything about the data.

**The idea.** That operator can be written in closed form. Initialise the complex weight
with it and train only the **residual**, so the capacity goes to the data instead of to
the geometry. Zero initialisation currently makes the model spend its first epochs
rediscovering an operator that was never in question.

**What would make it a result rather than a refactor.** The comparison is not "does MAE
improve" — §7.3 bans that as a headline and the measurement above is exactly why. It is
whether the *residual* response, once the geometry is subtracted, still correlates with a
white-noise control. If it does not, the model has learned something about the market and
the current architecture was hiding it under an operator. If it does, FITS at this
configuration has nothing to say and that is worth reporting too.

**Why parked.** It changes the architecture, invalidates every checkpoint, and needs its
own null control and grid sweep to mean anything — which is a task, not an afternoon. It
also interacts with the horizon: the cost it removes is a function of `η = 1 + H/L` and
vanishes entirely at `H = L`, so the benefit is configuration-specific and the study would
have to say at which configurations it exists.

---

## Regime Guard — reconstruction head as out-of-distribution detector

FITS uses the same architecture for anomaly detection: instead of forecasting forward,
reconstruct the current window and treat reconstruction error as an anomaly score.
Applied to trading, a spike means the market's spectral structure has diverged from
the training distribution — the model is outside what it knows. The bot would reduce
exposure and surface: "Unusual market state, reconstruction error at the 97th
percentile. New entries paused."

Estimated cost: one week. Excluded solely for schedule. Written up in the report
as declared future work.

> **The one conditional exception to the rule above** (decided 14 Aug 2026, recorded in
> `DECISIONS.md`). This idea returns to consideration **at GATE 2, if and only if that
> gate is green** — never earlier, because it extends FITS and a red GATE 2 cancels FITS
> outright. If GATE 2 is red or partial, the question is closed for this project. Until
> GATE 2 the answer is no, and "we are ahead of schedule" is not an argument that
> reopens it.

## Adaptive per-symbol cutoff frequency

Choose the FITS cutoff per symbol from measured spectral content rather than using
one global value. Second-order optimisation.

## SWT / MODWT shift-invariant wavelet comparison

The undecimated transform avoids the shift-variance of the DWT. A third arm in the
feature axis. No schedule room.

## Cross-sectional ranking objective

Train directly on relative ranking across the universe rather than on absolute
return, since ranking is an easier target than level prediction. Interesting;
changes the loss and the whole evaluation frame.

### Cross-sectional momentum as a selection arm — declined 23 Sep 2026

**A different idea from the paragraph above, and the difference is the reason it got as
far as a measurement.** That one changes the **training objective**: the model still
forecasts, and what moves is the loss it is fitted against. This one **trains nothing**.
It ranks the twenty symbols by trailing 12-month return, holds the top three and
rebalances monthly — a selection rule in the family `buy_and_hold` already occupies, which
is why it looked cheap: no `Forecaster`, no touch on spec §4, no checkpoint refused.

**Deliberately not a `GB-NN` task.** Phase 2 closed on 15 September under its own hard
boundary, so this is a parked idea and not a numbered one. `tests/test_phase2_ledger.py`
pins the Phase 2 task set to GB-61…GB-66 and a new heading in the expansion would go red
against it.

**The producing code exists and is committed**, which is the point of this entry being
quotable at all: `scripts/momentum_probe.py`, writing `report/momentum_probe.csv`. It reads
`data_cache/` only, trains nothing, and runs the three controls through `study.null_bars`
on the study's own seeding so that a finding about a control is not a finding about the
probe. Every figure below is one of its rows. It rebalances on the **first** bar of a new
month rather than the last bar of the old one, because only the first is knowable standing
on `t` — and `causality.perturb` never touches the index, so `assert_causal` would certify
either rule.

**What was measured, on the committed cache.** Every row below is a cell of
`report/momentum_probe.csv`, and `tests/test_momentum_probe_figures.py` fails if either
side is edited without the other. **Do not reformat this table**: its shape is the guard's
contract. A `quantity` is a CSV `quantity`, optionally `:SYMBOL` for a per-symbol row and
optionally `@p` to address that row's `p` column instead of its `value`.

| control | quantity | value |
|---|---|---|
| real | rebalances | 115 |
| real | top_n_minus_equal_weight | +0.005965 |
| real | top_n_minus_equal_weight@p | 0.2491 |
| shuffled | top_n_minus_equal_weight | +0.007690 |
| shuffled | top_n_minus_equal_weight@p | 0.0856 |
| noise | top_n_minus_equal_weight | -0.007067 |
| noise | top_n_minus_equal_weight@p | 0.1954 |
| real | spearman_trailing_forward | +0.01703 |
| shuffled | spearman_trailing_forward | +0.01730 |
| noise | spearman_trailing_forward | -0.05466 |
| real | rank_persistence_one_lookback_on | +0.05357 |
| shuffled | rank_persistence_one_lookback_on | -0.01320 |
| noise | rank_persistence_one_lookback_on | -0.16504 |
| real | drift_correlation_vs_real | +1.0000 |
| shuffled | drift_correlation_vs_real | +1.0000 |
| noise | drift_correlation_vs_real | +0.0812 |
| real | months_held:NVDA | 78 |
| real | months_held:LLY | 40 |
| real | months_held:TSLA | 36 |

The guard covers this table and nothing else in this file. The cost figures further down
come from `results.csv` and `PROGRESS.md`, not from the probe, and are not pinned by it.

**The arm scores HIGHER on shuffled returns than on the market, and that is a fact about
the control rather than about the arm.** `null_bars` permutes a symbol's log-return series
and rebuilds prices from the cumulative sum; a permutation preserves a sum, so every
symbol's whole-sample return survives exactly and the cross-sectional ordering of who won
is preserved perfectly — `drift_correlation_vs_real` above is the identity under
`shuffled` and near zero under `noise`. The shuffle *does* destroy the temporal
persistence of the ranking, which changes sign, and the edge survives anyway, because the
edge never came from persistence. **The shuffled
control is structurally inert against this arm**, so one of the two robustness tests every
headline claim is required to face could not have fired on it. See the PROGRESS row of
23 Sep for the general form and the pre-registerable discriminator.

**Survivorship, stated plainly and unbounded.** Over those 115 rebalances the rule holds
**NVDA in 78 of them**, LLY in 40 and TSLA in 36 — 67.8%, 34.8% and 31.3%, each derived
from two rows of the table above rather than stated a third time; `V` is never held. Spec §2.4
criterion 3 admits only securities continuously tradable to the cache's last bar, and §2.4
already records that the step from *admissible* to *these twenty* was judgment — `AVGO`,
`ORCL`, `KO` and `PEP` clear all five criteria and appear nowhere. So this arm's headline
number would be dominated by one name whose presence in the universe was a judgment call,
inside a set selected for having survived. Reporting it as a delta against `buy_and_hold`
removes the *level* of that bias and not the *concentration*: the reference spreads the
survivors' drift over twenty names and the arm concentrates it on the three that rose most.
**Nothing in this repository can bound it** — there is no delisted series and no
point-in-time index membership anywhere in the tree, and `data/historical.py` fetches
`cfg.universe` and nothing else.

**Cost if taken up.** Small in compute and not small anywhere else. In the `buy_and_hold`
shape it is one row per condition per fold: **224 rows, 14 conditions × 16 folds**, of
which only 96 are distinct backtests — a selection arm depends on `anchor`, `control` and
`target_in_loop` and on neither `lr` nor `cutoff_period_days`, so 128 of them would be
byte-identical duplicates. Its own compute is **one to two minutes** at `buy_and_hold`'s
measured 0.257 s a row; the real price is that `results.csv` is written whole with no
append path, so landing it costs the **full grid re-run, measured at 100 min 56 s** on
2 Sep. The Holm family grows from 117 reportable tests to 135, which moves every corrected
p-value in the report without costing a survivor. And the backtest path has no
cross-sectional selection layer at all — `rank_signals` is live-only — so this would be the
first arm whose signal set at a bar depends on other symbols, which makes it structurally
unlike every arm it would be compared against.

**It was declined on none of those costs.** It was declined because the two things worth
having from it — the inert control and the discriminator that predicts it — are findings
about the method, and the probe already produced them without an arm.

## Sentiment / news channel

Fragile external dependency, no room in 8 weeks.

## Intraday resolution

Data cost, microstructure effects, roughly 5x the complexity.

---

## New ideas

_Add below. Date, one paragraph, and an honest cost estimate._

## 2026-08-26 — Whole-share sizing as a declared design point

Fractional sizing is what forces the loop-side target: Alpaca refuses every multi-leg
order class on a fractional quantity, so no bracket. Whole-share sizing would permit a
bracket — but **not two independent protective orders**, because a working sell holds the
whole position regardless of size (measured against 97.38 shares). So it buys the bracket,
not the constraint's removal.

**The cost, measured rather than estimated** (20-symbol universe, `max_position_pct 0.10`
of 100k = $10,000 target notional, flooring to whole shares):

| symbol | at the last bar | mean over the study history | worst single bar |
|---|---|---|---|
| AAPL | 2.32% | 0.61% | 3.30% |
| AMZN | 1.90% | 0.64% | 2.67% |
| GOOGL | 3.02% | 0.56% | 3.87% |
| MSFT | 0.62% | 1.17% | 4.98% |
| NVDA | 0.87% | 0.22% | 2.12% |

**0.6%–3.0% at current prices, mean 1.75%; up to 4.98% on a single bar.** The history
means are lower because NVDA's split-adjusted history includes sub-dollar prices, where
flooring costs almost nothing — so **the trade-off is time-varying, not constant**:
whole-share sizing was nearly free for most of the study period and is expensive now.

**Deliberately not an arm on the grid.** It would confound sizing with execution model,
and the grid already carries the axis that matters. Cost if taken up: a sizing flag, a
`risk.shares_for` branch, and a full grid re-run.

## GB-62 — The C3 feature arm, declined 4 Sep 2026

**Parked 22 Sep 2026, and the parking records a decline, not a deferral.** GB-62 would have
added `C3_extended` — ATR, Bollinger position, a volume-flow measure and a longer-horizon
momentum — as a third feature arm against DLinear. Its reinstatement clause in
`GLASSBOX_PHASE2_EXPANSION.md` was live and its condition met on 3 Sep, and it was declined
on 4 Sep. **The reasoning is recorded once, in `docs/GB57_OUTLINE.md` §57.9 item 8
(`d69c167`), and is deliberately not copied here** — two copies of a reason diverge like two
copies of anything else. In one line: a cost decision under the 15 September boundary, not
a judgement that C3 is worthless, taken because the WITS operator comparison had already
answered the question C3 was a proxy for, and answered it harder. The limitation stays
bounded there: this study tested two feature configurations, both chosen by its author, and
cannot exclude that a third would have behaved differently.

**Cost if taken up:** see §57.9 item 8. In short, about 15 minutes of grid compute in the
best case, and a full grid re-run in the worst — any indicator whose warm-up exceeds 325
bars moves `min_history_bars` and refuses every checkpoint.

## GB-64 — Sentiment as a Co-Pilot annotation

**Parked 22 Sep 2026: out of scope for submission.** GB-64 would show recent headlines in
the Co-Pilot panel after a decision, labelled as context that did not enter it. It touches
the project's sentiment ruling directly — *"sentiment never enters the model and never
enters a backtest"*, recorded in full in `GLASSBOX_PHASE2_EXPANSION.md` §5 and carried into
the report as `docs/GB57_OUTLINE.md` §57.9 item 7 — because its whole design is to keep
sentiment beside the decision and out of it, and the mechanism meant to guarantee that is
not built. **Its first acceptance criterion is unmet:** `DECISIONS.md:267` records that the
existing import contracts do not stop `glassbox.backtest` or `glassbox.experiments` from
importing a provider under `glassbox.data`, so the promised enforcement is *"discipline in a
contract's clothes"* until a new `forbidden` contract names the sentiment module. Distinct
from *Sentiment / news channel* above, which is sentiment as a model input and is excluded
by the ruling itself rather than by time.

**Cost if taken up:** 3 SP as estimated — the `forbidden` import contract, a live-path-only
provider, a panel label stating the annotation is not a model input, and a test that
removing the provider changes no number in `results.csv`.

## GB-65 — Demonstration, folded into GB-54 and GB-55

**Parked 22 Sep 2026: its rehearsals are delivered through GB-54's demo script, and its
report figures through GB-55, rather than as a separate task.** GB-65 specified four
things. They are listed here with where each now
stands, because folding one task into another is how scope disappears without anybody
deciding to drop it:

- **Two full rehearsals, fold 13 then fold 1** — inside GB-54's own definition in spec §9,
  *"Both folds, in this order: 13 then 1."* Covered.
- **The four architecture-report screenshots** — **assigned to GB-55**, ruled 22 Sep 2026.
  The six frames now in `report/screenshots/` predate the twenty-symbol flip, so GB-55
  inherits a retake, not a set of finished figures.
- **The caption rule for the spectral-panel capture**, that FITS is not the deployed arm and
  beats neither reference — **assigned to GB-55 with the screenshots**, ruled 22 Sep 2026.
- **The vision video's shot 6 recording** — **cut**, ruled 22 Sep 2026. Dropped, not
  deferred: no task carries it, and this line is the record that its absence was decided.

GB-54 is not started as of 22 Sep 2026. GB-55's row in spec §9 names the screenshots and
the caption rule as of the same day, so the assignment is recorded where GB-55's owner
will read it and not only here.

**Cost if taken up separately:** none. The rehearsals are GB-54's, the screenshots and the
caption rule are GB-55's, and shot 6 is cut.

## The dashboard honesty pass — what landed 22–23 Sep 2026, and what did not

**Parked 24 Sep 2026.** A pass over `glassbox/dashboard/app.py` against one rule: *the
console must never state something a reasonable reader will take as true when it is not.*
Eight items were fixed. The remainder is recorded here because a defect found, numbered and
left is scope that disappears without anybody deciding to drop it — the same failure mode
this file's GB-65 entry exists for.

**What landed.** Each was proven by breaking the guard it added, confirming failure, and
restoring the file byte-identical:

- **D1 — the stop room measured against the book, not the broker** (`9b760d5`, 22 Sep 2026).
  The panel read `stop_loss` from `book.json`, so a stop that had been cancelled, filled or
  never placed still rendered as protection. It now reads the broker's working stop and says
  `NO LIVE STOP · UNPROTECTED` or `STOP UNKNOWN · BROKER READ FAILED` when it cannot.
- **D2 — one liveness verdict** (`780823a`, 23 Sep 2026). Each surface decided separately
  whether the loop was alive, so the page could claim live in one region and stale in
  another. Six state words, one function, read by every surface that makes the claim.
- **D2b — liveness resting on evidence** (`cb92425`, 23 Sep 2026). The loop now writes a
  heartbeat a program can read, so the page's claim rests on the loop's own last write
  rather than on the page's timer, and `LIVE` means one thing everywhere it appears.
- **D3 — the two headline backtest panels against buy and hold** (`f87deec`, 23 Sep 2026).
  Cumulative equity and the fold bars reported absolutes; spec §7.3 asks for a reference.
- **D4 — the radar's references** (`9f95b9c`, 23 Sep 2026). Five axes of six printed a bare
  number under a caption reading *"each against its own reference"*. Also removed green from
  a data encoding, and corrected flatness, which was scored one-sided on a two-sided metric.
- **D5a — a chart slot for every position** (`b294e04`, 23 Sep 2026). Sparklines were drawn
  for some positions and not others, none of them labelled, so a chart could be read against
  the wrong row. Every position gets a slot, every slot says whose it is, and a slot with no
  chart gives one of three distinguishable reasons.
- **D5 — the daily strip deleted** (`fb633f8`, 23 Sep 2026). 922 trading days in a 700px row
  is 2px a tile, which renders as a smear. The counts and their window are the panel, and it
  declares on its face that day counts are not a result.
- **D6 — the session equity panel sized to its session** (`b866cc4`, 23 Sep 2026). Two
  readings across a full plot band read as an empty panel rather than a short one, which is
  the state the live session is in every morning until the loop has polled ten times.

The report screenshots were re-captioned on 24 Sep 2026 (`4bbd224`) rather than retaken:
three frames document defects this pass fixed, and are preserved and dated because fixing a
defect destroys the ability to photograph it. That is GB-55's retake, narrowed.

**What was left unfixed.**

- **D7 — three charts compute a scale from a subset of what they draw against it.** Held as
  `xfail(strict=True)` at `tests/dashboard/test_app.py:2484`, with its premise pinned
  outside the xfail so the case cannot pass for the wrong reason. `sparkline_svg` takes
  `min(prices, stop_loss)` but never the max, so a position gapped below its own stop draws
  that rule at y = −12,998,634 of 78; `response_svg` normalises by `max(gains)`, so a
  negative gain past it lands off the canvas; `forecast_svg` takes its range over the finite
  values and then maps every value, so one non-finite close emits `y=-inf`. Not done because
  it is one defect in three places and belongs in a pass of its own; a strict xfail turns red
  the day any of the three is fixed, so this entry cannot rot quietly.
- **D8 — every broker failure renders as an empty book.** The outer `except Exception` in
  `_broker_view` at `glassbox/dashboard/app.py:3940` returns `{}, {}, {}, None`, so absent
  credentials, a network failure, a rate limit and an account genuinely holding nothing are
  indistinguishable on the page. **This is the defect the bound state directory was added to
  fix, in a second place:** an empty queue and no recommendations read alike unless the page
  says which it is. D1 narrowed the inner read — `stops = None` now means *could not ask*
  rather than *no stop* — and the outer handler was deliberately left alone as out of scope.
  Not done because it needs the same `None` vs `{}` distinction `_recent_closes` already
  draws for prices, applied to four return values and every caller of them.
- **D9, D10 and D11 — numbered in the session's prompts and never enumerated to it.** They
  were referred to only as the range "D7 to D11". This entry cannot state what they are, and
  guessing would put three invented items into the one file whose purpose is to record real
  decisions. **Recorded as a gap on purpose:** whoever holds those three numbers should write
  them in here, and until they do, "D7 to D11" names five items of which this project has a
  written record of two.

**Cost if taken up separately:** D7 is bounded and mechanical — one scale rule applied in
three builders, with the failing tests already written. D8 is the larger one, because the
honest answer changes four return values and every reader of them, and because it cannot be
verified without deliberately breaking the credential path. Neither blocks Gate 3. D9 to D11
cannot be costed, which is the cost of not writing them down.
