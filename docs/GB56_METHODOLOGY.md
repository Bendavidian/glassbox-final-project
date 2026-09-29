# GB-56 — Methodology and Leak Freedom

This chapter establishes that the results in GB-57 measure what they claim to measure. A null
result places an unusual burden on its methodology: a study that finds nothing is
indistinguishable from a study whose pipeline was disconnected, so every claim in GB-57 rests
on the mechanisms described here rather than on the numbers themselves.

The chapter is organised around one question asked repeatedly: **what would have to fail for
this claim to be wrong, and does anything check it?**

## 56.1 — What leak freedom means here

Look-ahead bias is the failure mode this chapter is built against. It has three distinct
forms and they require different defences, which is why they are separated here rather than
treated as one concern.

**Temporal leakage.** A feature at time `t` uses a bar from after `t`. A rolling mean computed
with a centred window, a normalisation fitted over the whole series, an indicator whose
library implementation looks ahead by one bar. §56.3 is the defence.

**Evaluation leakage.** The test period influences the model through a path other than the
features: a hyperparameter chosen by looking at test performance, a threshold calibrated on
the data it is evaluated on, a model selected because it scored well. §56.2 and §56.5 are the
defences.

**Implementation leakage.** The training path and the live path assemble their inputs
differently, so a model validated offline is not the model that runs. Nothing about this is
visible in the metrics — both paths produce plausible numbers. §56.4 is the defence.

A fourth category is named here because this project's headline finding depends on it and no
harness can catch it: a model whose training data contains the evaluation period. §56.9
records the decision that follows.

**The standard applied throughout is the one stated in GB-57 §57.10: a description of what
the code should do is not a mechanism.** Each defence below is a test that fails when the
property is violated, and each was verified by violating it deliberately.

## 56.2 — Walk-forward evaluation and the embargo

### Walk-forward, not a single split

The study uses **sixteen walk-forward folds**. Each trains on a 24-month window and tests on
the window that immediately follows it; the windows then advance. No fold ever tests on a
period any of its own training data postdates.

A single train-test split would give one number per arm and no way to distinguish a model
from a lucky period. Sixteen folds give a paired sample, which is what makes §56.7's tests
possible: each comparison is fold-by-fold against the same reference over the same window,
so period effects cancel rather than accumulate.

### The embargo

Between the end of a training window and the start of the test window sits an **H-bar
embargo**, where H is the forecast horizon.

The reason is specific. A feature window at the last training bar reaches forward through the
label: the target for that bar is the return over the next H bars, which overlaps the first H
bars of the test period. Without the embargo, the model would have been trained on a label
computed from bars it is about to be evaluated on. The embargo removes exactly that overlap
and nothing more.

### What this does not protect against

Walk-forward removes temporal ordering violations between folds. It does not remove them
*within* a feature window, and it does not prevent a feature implementation from looking
ahead by one bar. That is §56.3.

It also does not prevent the fold boundaries themselves from being a lucky choice, which is
why the study runs the same evaluation at three fold-grid anchors — 0, 21 and 42 trading days
— and reports a result only if it survives all three. §56.6 treats that as one of the two
robustness tests, and records that it once killed a reported correlation of `r = −0.47`.

## 56.3 — The causality harness

Walk-forward prevents a fold from testing on its own past. It says nothing about whether a
feature at time `t` was computed using a bar from after `t`. That is a property of each
feature's implementation, and it cannot be established by inspection: a rolling window that
looks ahead by one bar produces a series that is indistinguishable from a correct one without
testing for it.

`tests/causality.py` is the mechanism. It takes a function that maps a frame of bars to a
derived series, splits the frame at a point, perturbs everything **after** the split, and
asserts that everything **before** the split is unchanged. If a value before the split moves,
the function saw a bar it should not have.

### Two perturbation modes, because they catch different leaks

**Scale.** Multiply the post-split values by a constant. This catches any statistic fitted
over the whole series — a normalisation using the full-sample mean or standard deviation, a
min-max scaling, a z-score computed once over everything. Those are the leaks that survive a
reordering, because they depend on the magnitude of future values rather than their order.

**Shuffle.** Permute the post-split values, preserving their distribution. This catches
directional leaks — anything that depends on the *sequence* of future bars rather than their
aggregate. A centred rolling window, a peak detector, a next-bar return used as an input.

Neither mode subsumes the other. A whole-sample mean survives the shuffle exactly, because a
permutation preserves a sum; a centred window survives the scale test if the constant happens
to cancel. Both are asserted at **three different split points** per feature, so a leak that
happens not to bite at one boundary is still caught.

### What it is applied to

Every feature in the active channel set: `close_logret`, `rsi14`, `vol_z`, `mom10`,
`ma_dist20`, and the causal DWT bands. Each is asserted causal in both modes at all three
splits, and each assertion was verified by introducing a deliberate look-ahead and confirming
the harness fails.

The wavelet channels needed the harness most. A standard DWT is not causal — it uses the
whole signal — so the implementation here is a trailing variant, and the property that makes
it usable is exactly the one the harness tests rather than one the author can assert.

### What the harness structurally cannot catch

Stated here rather than left implicit, because a limitation of the instrument belongs beside
the instrument.

**It never changes the index.** `perturb` disturbs values, not dates. So any rule that depends
on *which date it is* rather than on values passes in both modes regardless of whether it is
knowable at `t`. This was discovered while scoping a monthly rebalancing rule: "rebalance on
the last bar of the month" cannot be known at `t` — you learn a bar was the month's last only
when the next one arrives — and it passes the harness at all three splits, identically to
"rebalance on the first bar of a new month", which is knowable. **The harness certifies
either one.** Any calendar-dependent rule needs its own explicit test.

**It is frame-shaped, not symbol-shaped.** Every caller passes one symbol's bars, so nothing
cross-sectional is tested. The harness would work on a wide frame — one column per symbol —
and `perturb` disturbs every numeric column after the split, which is the right perturbation
for a function from a wide frame to per-date ranks. No such caller exists, because no
cross-sectional arm entered the study.

**It tests the feature path, not the label.** Label construction is governed by the embargo in
§56.2 and by the horizon, and the harness has no view of either.

## 56.4 — One place where a model input window is assembled

This is the defence against implementation leakage, and it is the one whose absence leaves no
trace in any metric.

Two paths need a model input window. The training path builds one from cached historical bars;
the live loop builds one from bars fetched at runtime. If the two differ — a column ordered
differently, a warm-up truncated differently, a normalisation applied at a different point —
then the model validated offline is not the model that runs, and **both paths produce
plausible numbers**. Nothing fails. There is no error to detect.

### The rule

`features/builder.py` is the only place in the project where a model input window is
assembled. Training calls it; the live loop calls it; the replay calls it. No second
implementation exists, and the rule is a standing ruling rather than a convention.

That removes the failure by construction rather than by testing for it. But "there is only
one implementation" is itself a claim, and its mechanism is not structural: neither
import-linter contract constrains where a window is built, so nothing prevents a second
builder from being written. What would catch one is the parity sweep below, which tests the
outcome rather than the structure — a second path that assembled a different window would
fail its byte-identical assertion.

### The parity sweep

Structure is not behaviour. A single shared function can still produce different results on
the two paths if they hand it different inputs — a different number of warm-up bars, a
different slice boundary, a different dtype.

So the sweep compares **byte-identical output**. For every symbol in the universe, at every
one of 25 timestamps, it builds the window through the training path and through the live
path and asserts the two are identical, not merely close. At twenty symbols that is **500
comparisons**, and the test asserts `identical == total` rather than a tolerance.

The count is derived — `SWEEP_TIMESTAMPS * len(cfg.universe)` — and not written as a literal.
A sibling test that pinned the same product as `25` survived months of green and broke on the
universe flip while this one did not; GB-57 §57.10 records it as an instance.

### The floor

The sweep is run at a **445-bar floor**, and the number is derived rather than chosen:
`input_len` 120 plus the deepest per-channel warm-up, which is `rsi14`'s 325 bars. It is a
**max** over the active channels, not a sum — a sum would be 496 — because the warm-ups
overlap in time rather than stacking.

325 is itself derived: Wilder's RSI recursion decays by a factor of 13/14 per bar, and 325
bars is where the influence of the first bar falls below 1e-10. Below that floor the two paths
can disagree in the last decimal places purely because one has seen more history, and the
sweep found exactly that: at an earlier 352-bar floor, only 32 of 125 pairs were identical.

The floor is verified rather than assumed on every configuration change. When the universe
moved from five symbols to twenty the floor was recomputed and stayed 445, because universe
size appears nowhere in the derivation.

### What it buys

The live loop that ran on 23 September assembled its windows through the same function, at the
same floor, as the grid that produced every number in GB-57. That is not an inference from
code review; it is the property the sweep asserts, at 500 comparisons, on every suite run.

## 56.5 — Thresholds calibrated on validation only

A forecast is not a decision. The model emits a predicted return over the horizon; a threshold
turns that number into enter, hold or exit. Where the threshold comes from determines whether
the trading results in GB-57 mean anything.

### The split, and what each part may touch

Each fold divides into three windows: **train**, **validation**, **test**. Training fits the
weights. Validation decides when to stop training and where to place the entry threshold. Test
is read once, at the end, and influences nothing.

The threshold is calibrated **per fold, on validation only**. A single global threshold chosen
across all folds would let each fold's test period inform every other fold's decisions, which
is evaluation leakage through a path the embargo does not cover.

### Why this is the failure that would be easiest to hide

A threshold calibrated on test would produce trading numbers that look excellent and are
meaningless, and nothing in the metrics would say so. The MAE would be unchanged, because
forecast error does not depend on the threshold. Direction would be unchanged, because it is
scored on the forecast sign. **Only the trading metrics would move**, and they would move in
the flattering direction, which §57.10 records as the direction nobody investigates.

So the defence is structural rather than statistical: the test slice is not passed to the
calibration function at all.

### The bias this leaves, and its direction

The validation split is used twice — for early stopping and for threshold calibration — and
that is a real flaw, stated rather than defended.

It does not leak into test. What it means is that thresholds are calibrated on forecasts that
are already slightly overfit to validation, so on test they are **miscalibrated rather than
inflated**. The double use degrades test performance; it does not flatter it. Every test
figure in GB-57 is therefore a lower bound with respect to this particular flaw.

The magnitude is unmeasured. Bounding it would take one re-run with thresholds calibrated on a
held-out slice, which GB-57 §57.11 records as open.

A related consequence: the validation Sharpe is a **selected maximum** and is not reportable.
It appears in the live console as the provenance of the deployed band — `VAL SHARPE +0.785`
— and nowhere in the results as a performance claim.

### What the universe flip did to the threshold

The calibrated entry threshold moved from **0.026076 to 0.006715** on the same fold, same
window, same code — roughly four times more permissive. Validation trades rose from 8 to 53
and validation Sharpe from 0.483 to 0.785.

Only the universe changed. A threshold calibrated over a cross-section four times wider lands
in a different place, and the deployed system went from standing aside on almost every bar to
entering on several. This is reported because it is a measured consequence of a
configuration change that touched no threshold logic, and because a reader comparing live
behaviour before and after would otherwise attribute it to a code change.

## 56.6 — Two robustness tests, and why neither subsumes the other

Every headline claim in this study is required to survive two independent checks. The
requirement is a standing ruling rather than a convention, and the case for it is empirical:
each check has killed a finding the other passed.

### Grid sensitivity

The same evaluation is run at three **fold-grid anchors** — 0, 21 and 42 trading days. Shifting
the boundaries changes which windows train and which test, without changing the data, the
model or the metric.

It asks: is this finding a property of the market, or of where I happened to cut?

It killed a reported correlation of **r = −0.47**, which was convincing at one anchor and
absent at the others.

### Null controls

The same evaluation is run on data whose structure has been deliberately destroyed, in two
ways.

**White noise.** Each symbol's returns are replaced by N(0, σ) with matched variance. Every
symbol has zero expected drift and no temporal structure at all.

**Shuffled returns.** Each symbol's log-return series is permuted and prices are rebuilt by
cumulative sum. The distribution is preserved and the order is destroyed.

They ask: would this finding appear even if there were nothing to find?

A null control killed a measured phase advance of **+1.9582 days with a standard deviation of
0.0203** across 48 independently trained models — a coefficient of variation of 1.04%. It had
passed grid sensitivity at 48 of 48 cells. Then it survived being trained on white noise,
which revealed it as a property of a deterministic operator rather than of the market.
**Agreement that tight across four years of equity returns is evidence about the machine, not
about the data**, and that is now a standing rule in its own right: suspicious stability calls
for a control, not a larger sweep.

### Why neither subsumes the other

Grid sensitivity varies *where you look*. Null controls vary *what you look at*. A finding that
is an artefact of the fold boundaries survives every null control, because the data is real
and the artefact is in the split. A finding that is a property of the architecture survives
every anchor, because the architecture does not move when the boundaries do.

Each of the two examples above is a case the other missed. That is the argument, and it is
made by measurement rather than by assertion.

### A class of claim the shuffled control cannot falsify

This is a limitation of the method, discovered in September and not present in the study
design. It is stated here because §56 is where a reader looks to find out what the checks are
worth.

**A permutation preserves a sum.** `study.null_bars` permutes a symbol's log-return series and
rebuilds prices by cumulative sum, so the symbol's whole-sample return survives exactly: across
the twenty symbols the shuffled final close differs from the real one by at most 1.43e-14
relative, which is `cumsum` reordering rather than a difference in the sum.

**Spearman(real full-sample drift, shuffled full-sample drift) across the universe is
+1.0000**, against **+0.0812** for white noise.

The consequence is the finding. Under that permutation any trailing window is a sample
*without replacement* from the whole series, so its expectation is the window length times the
symbol's full-sample mean — **a partial readout of the sample's outcome, including the bars
after `t`**. Any signal that is a monotone function of accumulated per-symbol return is
therefore immune to this control by construction.

Demonstrated rather than argued, on a cross-sectional momentum rule scoped and declined the
same day: top three by trailing 252 bars, rebalanced monthly, earns +0.00596 per rebalance over
equal weight on real data (paired p 0.2491) and **+0.00769 on shuffled (p 0.0856), more than
on the market** — while the rank persistence it is supposed to trade collapses from +0.0536 to
−0.0132. The edge survives the shuffle because it never came from persistence.

**Nothing would have said so.** A shuffled row would have come back, survived, and been read as
evidence.

**The pre-registerable discriminator**, which is the reusable part: Spearman of full-sample log
drift between real and the control, across the universe, run *before* the arm. Above 0.9, that
control cannot falsify the arm and its shuffled row means nothing. `shuffled` reads 1.0000 and
`noise` reads 0.0812. `scripts/momentum_probe.py` reproduces every figure deterministically
into `report/momentum_probe.csv`.

This does not weaken any arm reported in GB-57: those arms forecast, and a forecast is not a
monotone function of accumulated return. It narrows the rule, and the narrowing is stated here
rather than left for a reader to discover in a future study.

### One structural limit on how the two can be combined

`study.conditions` builds a **star, not a cross**: one reference condition at the centre with
spokes each moving a single variable. The control spokes sit at the centre only.

| control | anchor 0 | anchor 21 | anchor 42 |
|---|---|---|---|
| noise | 208 | 0 | 0 |
| real | 544 | 112 | 112 |
| shuffled | 112 | 0 | 0 |

So the two checks cannot currently be **crossed**. "Flat at all three anchors under both null
controls" is unavailable in this design rather than unmeasured, and GB-57 §57.9 states the
narrower claim that is supported. Closing it is a cross rather than a star: 108 conditions
against 14.

## 56.7 — Statistics: pairing, correction, and what is reportable

### Paired, because the folds are the unit

Every comparison in GB-57 is **paired by fold**: arm against reference, same fold, same window,
same data. Comparing two means across sixteen folds would treat period effects as noise; the
paired form cancels them, because a fold that was hard for the model was hard for its reference
too.

The test is the **two-sided Wilcoxon signed-rank test**, exact rather than normal-approximated.
It is non-parametric, which matters here: fold returns are not normally distributed, and
sixteen observations is too few to appeal to asymptotics.

### Correction over the whole family, in one pass

The study runs many tests — every arm, anchor, metric and condition on real data — and running
many tests guarantees some will cross any fixed threshold by chance.

The family is **117 reportable tests**, and Holm-Bonferroni correction is applied across all
117 **in a single pass**, not per metric or per arm. The decomposition is 39 total_return,
27 sharpe, 27 direction and 24 mae, over 9 reportable conditions. With all controls included
the count is 162.

`buy_and_hold` contributes zero tests. It is excluded by name, because its directional accuracy
is a tautology — an always-long rule is right exactly when the market rises — and it makes no
forecast to score. It appears throughout as a reference and never as a subject.

Correcting in one pass has a consequence that matters when anything changes: **every corrected
p-value moves together.** The universe flip did not add results to an existing table; it
rebuilt the family, so "of 117 tests, 17 survive" became "of 117, 29" and could not be carried
across from the earlier draft. GB-57 §57.8 treats why the count moved.

### The ceiling, stated beside the numbers

The exact two-sided Wilcoxon test on **sixteen paired folds cannot return a p-value below
3.05 × 10⁻⁵**, however large the effect. It is a property of the number of pairs, not of the
data.

With 117 tests in the Holm family, the smallest attainable corrected value is bounded well away
from zero. **This design can fail to reject, and can detect only fairly large effects. It is
not capable of establishing a small edge**, and no claim in GB-57 should be read as if it were.

Stating a ceiling beside a p-value is unusual and it is deliberate: a reader who does not know
the bound will read a corrected 0.0036 as weaker evidence than it is, and a reader who assumes
the design could have found a small effect will read the null result as stronger than it is.

### What is reportable, and where that is enforced

Not every row in `results.csv` may reach a metric. A run that was skipped, a condition that
exists only as a control, an arm that produced no Sharpe because it stood aside for the whole
fold — each is excluded, and the exclusion lives in **one place** rather than being reproduced
at each call site.

Two consequences worth naming:

**Sharpe rests on a selected subset.** Where an arm stands aside for an entire fold it produces
no Sharpe, so Sharpe comparisons rest on between 13 and 16 folds with a mean of 14.63, against 16
for direction, MAE and total return. The surviving folds are those in which the model chose to
trade, which is not a random subset. GB-57 reports Sharpe as descriptive rather than
inferential for this reason.

**A test can return a row with no p-value.** When fewer than six folds show a non-zero
difference, the test returns NaN rather than a number. The printed family size counts rows and
the correction ignores NaN, and the two agree today because all 162 p-values are finite. They
would diverge for an arm that ties its reference on most folds, and that divergence is recorded
rather than assumed absent.

### One defect in the reported statistics, corrected

The `sd` column printed beside each arm is that arm's **own per-fold spread**, not the paired
difference — and only the paired difference feeds the signed-rank test. Over the 39
total_return cells present at both universe sizes, the reported spread **rose** ×1.42 and grew
in 25 of 39 cells, while the paired-difference sd **fell** ×0.38 and fell in **39 of 39**.

The two quantities move in opposite directions and only the lower one is unanimous. A reader
auditing GB-57 §57.8's power argument against the printed column would read the mechanism
backwards, which is why the distinction is stated in both chapters.

## 56.8 — Determinism and provenance

A study whose headline is negative has to be re-runnable by someone who does not trust it.
Determinism and provenance are what make that possible.

### Bit-identical on the same hardware

The full grid produces **bit-identical** results across runs on one machine. Not statistically
equivalent, not equal to a tolerance — the same bytes.

This is a stronger property than it first appears, because it is asserted rather than hoped
for. A test re-runs a condition and compares the output to the committed one, and it was that
test which cost roughly 136 seconds of the suite's runtime after the universe flip, measured
rather than estimated. GB-57 §57.10 records why that figure matters: an earlier report of it
as "about 50 minutes" was wrong by a factor of 22, because a CPU counter summed across 20
threads was read as elapsed time.

Everything the grid depends on is pinned: seeds, thread counts, and the BLAS thread count
explicitly reduced to one, which is why the grid is single-threaded by design and why it takes
100 minutes 56 seconds rather than a quarter of that.

### What is not claimed

**Cross-architecture reproducibility is untested.** Floating-point reduction order in the
training path makes bit-exact equality across architectures unlikely, and no attempt was made
to establish it. The claim is the one that was measured: same hardware, bit-identical.

Stating the narrower claim matters here. A reader who reproduces this study on different
hardware and gets different last-decimal values should know that was expected.

### Provenance: which snapshot a number came from

Every row in `results.csv` carries `config_hash`, `model_config_hash` and
`data_snapshot_last_bar`. A result names the configuration and the data vintage that produced
it, and a file whose rows disagree about the snapshot is refused rather than averaged.

The two hashes are separate on purpose. `model_config_hash` covers the settings that **shape
a model** — channels, input length, horizon, architecture — and `config_hash` covers
everything, including settings that describe what the system does *with* a model. A universe
change moves the second and not the first, which is correct: the checkpoint is still valid, the
study it belongs to is not.

**That distinction has a cost, and it was paid.** When the universe moved from five symbols to
twenty, the deployed checkpoint passed the startup gate because the model-shaping sections were
unchanged — true, and irrelevant. Twelve seconds later the prediction path refused to score a
symbol the checkpoint had never been trained on, with a message stating why: *"scoring an
untrained symbol would feed the shared weights a distribution they never saw, and the result
would look like a forecast rather than an error."* GB-57 §57.10 files it as an instance of a
guard catching a failure one layer above it had passed.

### The clean-clone audit

Reproducibility was verified by cloning the repository fresh and running the full suite from
nothing. The audit found **three defects**, of which the most instructive was that
`import glassbox` succeeded from the repository root without the editable install — so every
prior test run had exercised the source tree rather than the installed package, and no test
had ever run against the thing a user would get.

The data cache is committed for this reason. A clone with no data can run nothing, and a
result that cannot be reproduced from a clone is a claim rather than a measurement.

The audit is scheduled to run again against the submitted tag.

### One honesty rule, and why it is worth a paragraph

**No report in this project says a suite passed unless `pytest` ran with no path argument and
no marker, and the exit code was read directly rather than through a pipe.**

The rule exists because a piped invocation once reported exit 0 on a suite with a failure, the
status being the pipe's. It was then found that the rule named an invocation which **could not
run in this repository at all** — bare `pytest` exited 2 with zero tests collected, while every
reported figure had come from `python -m pytest`. The numbers were true of the tests; the
provenance was wrong; and no report had said so. The fix was to make the named command work
rather than to rename the rule, because the rule is the honesty mechanism and the invocation is
the detail.

## 56.9 — What was excluded, and why

Three things this study could have included and did not. Each exclusion is a methodological
decision rather than a shortfall, and each is stated with the reasoning that produced it.

### LLM sentiment is never a model input and never enters a backtest

This is the only exclusion that is a standing ruling rather than a scope decision, and it is
the one most worth a reader's attention.

Sentiment extracted from news by a language model is an obvious feature for a trading system,
and it is excluded absolutely. Three reasons, in ascending order of weight:

**An LLM's training data contains the evaluation period.** A model asked in 2026 to score the
sentiment of a 2024 article answers with knowledge of what happened next. It cannot be made
not to. This is look-ahead bias, and it enters through a path the causality harness in §56.3
**structurally cannot catch**: the harness perturbs the input series and checks that past
values do not move. It has no view of what a language model knows.

**It breaks bit-identical reproducibility.** §56.8's guarantee depends on every input being
pinned. An external model behind an API is not pinned, is not versioned in any way the study
controls, and may be retired.

**The failure would be invisible and flattering.** A sentiment feature carrying look-ahead
would improve every trading metric and would not fail any test in this project. It is the
defect class GB-57 §57.10 is about, in the one place where nothing could catch it.

The rule is therefore absolute rather than conditional, and the chapter that explains the
exclusion is worth more to this study than the feature would have been.

### Cross-sectional selection, scoped and declined with its measurement

A momentum rule — rank the twenty symbols by trailing return, hold the top few, rebalance
monthly — was specified as a possible arm and is not in the study. It was not declined on
intuition: a read-only probe measured it first, and two of the three reasons are findings in
their own right.

**It does not work here.** Spearman between trailing 252-day return and next-month return is
+0.0170 at p 0.60. Top three against equal weight earns +0.00596 per rebalance, paired
p 0.2491.

**The shuffled control cannot falsify it**, for the structural reason in §56.6, which means
one of the project's two required robustness tests would have been inert on this arm and the
write-up would have had to say so rather than report "survived the shuffle".

**It is the most survivorship-exposed thing the study could run.** Over 115 monthly
rebalances the rule holds NVDA in 67.8% of months — two months in three, the arm is a bet on
one name, whose presence in the universe was a judgment call, inside a set selected for having
survived. The delta convention does not rescue it: reporting against buy-and-hold puts the
survivorship inside the reference, which removes the *level* of the bias and not the
*concentration*.

`scripts/momentum_probe.py` reproduces every figure deterministically into
`report/momentum_probe.csv`, and the arm is recorded in `IDEAS_PARKED.md` with its costs. A
decline recorded on measured grounds is a claim with the same obligation as any other.

### A third feature configuration, declined on cost

`C3_extended` — ATR, Bollinger position, a volume-flow measure, longer-horizon momentum — was
specified and is not in the study. GB-57 §57.9 states the costs: the incremental compute is
small, but every new indicator needs a causality proof and a hand-computed fixture, OBV has no
causal trailing definition, and an ATR at period 20 would raise the parity floor from 445 to
589 and invalidate every existing checkpoint.

The stronger reason is that it asks a question about the author's own choices — *is the null
an artefact of these two feature sets?* — which can only be answered by making a third set of
them. GB-66 replaced the model's transform instead, and GB-57 §57.5 reports what that found.

**What a reader is owed:** this study tested two feature configurations, both chosen by its
author, and cannot exclude that a third would have behaved differently.

### The shape these three share

Each exclusion removes something that would have made the results look better or the study
look larger. The sentiment ruling removes a feature that would have improved every trading
metric through a leak nothing could detect. The momentum arm removes the only arm whose
shuffled control would have come back clean for the wrong reason. The third feature set
removes a question answerable only by more of the same choices.

**A study that finds nothing has to be more careful about what it excludes than one that finds
something**, because every exclusion is a place a reader can suspect the finding was
manufactured. That is why each is stated with its measurement rather than with its rationale
alone.