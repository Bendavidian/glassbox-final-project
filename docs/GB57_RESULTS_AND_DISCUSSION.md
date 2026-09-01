# GlassBox Trader — Results and Discussion

*Draft, 27 August 2026. Every number in this chapter carries its source. Sources are
`results.csv` (1,110 rows, snapshot last bar 2026-08-13) and `report/report.md`, generated
from it by `python -m glassbox.experiments.report`. Where a claim cannot be supported from
those two files, it is marked **[UNSOURCED]** and listed again at the end; nothing marked
that way should reach the submitted chapter without the measurement behind it.*

---

## 1. The result: nothing was found, and the instrument was working

The study's headline is negative. No forecasting arm in this project beats the trivial
always-long baseline on directional accuracy, at any fold-grid anchor, under either null
control. That sentence is easy to write and easy to disbelieve, because a null result and a
broken experiment produce the same table. The distinguishing evidence is the control, and
it is what this section leads with.

Directional accuracy under the buildable execution model, at fold-grid anchor 0, averaged
over sixteen folds and every condition sharing that anchor
(`results.csv`, rows with `skipped=false`, `channels=C0_base`, `target_in_loop=true`):

| arm | real data | white noise | shuffled returns | noise − real | shuffled − real |
|---|---|---|---|---|---|
| buy-and-hold | 0.5560 | **0.5064** | 0.5418 | **−0.0495** | −0.0142 |
| DLinear | 0.5006 | 0.5006 | 0.5021 | +0.0000 | +0.0014 |
| FITS | 0.4958 | 0.5041 | 0.5039 | **+0.0083** | +0.0081 |
| WITS | 0.5031 | 0.5050 | 0.5074 | **+0.0018** | +0.0043 |

The always-long reference rate is 0.5395 at anchor 0, 0.5523 at anchor 21 and 0.5654 at
anchor 42 (`report/report.md`, Summary table, `direction_reference`).

Read the model rows first. FITS scores **eight basis points better on white noise than on
the market**. WITS scores two basis points better on noise than on the market. DLinear is
identical to four decimal places — 0.5006 on real data and 0.5006 on noise. Destroying the
signal entirely, by replacing four years of equity returns with Gaussian noise of matched
variance, costs these models nothing. That is not a small effect that failed to reach
significance; it is the absence of the effect at the resolution the experiment can see.

The pattern does not depend on where the folds are cut. Across the three fold-grid
anchors, on real data (`report/report.md`, Summary table):

| arm | anchor 0 | anchor 21 | anchor 42 |
|---|---|---|---|
| DLinear | 0.5006 | 0.4973 | 0.5006 |
| FITS | 0.4958 | 0.5056 | 0.5095 |
| WITS | 0.5031 | 0.5067 | 0.5066 |

Every cell sits within seven basis points of a coin flip, and every one is below the
always-long rate for its anchor. Shifting the fold boundaries by 21 and 42 trading days —
which changes which windows train and which test, and is the check that killed an earlier
`r = −0.47` correlation finding — moves nothing.

Now read the first row of the null table, because it carries the argument. Buy-and-hold is
the only arm that **degrades** under the noise control, and it degrades by 4.95 points,
two orders of magnitude more than any model arm moves. This is exactly what its mechanism
predicts and what no other arm's mechanism predicts. Buy-and-hold has no forecast; its
entire content is the upward drift of the underlying series, and the white-noise control
is constructed precisely to remove that drift. When the drift goes, buy-and-hold's edge
goes with it. When the drift goes, the models are unaffected, because they never had any.

A control that moves for exactly one arm, and for the one arm whose mechanism says it
must, is what separates *"we found nothing"* from *"we measured that there was nothing to
find, and the instrument can detect something when something is there."* Without the
buy-and-hold row this table would be consistent with a pipeline that had silently
disconnected the data. With it, the pipeline is demonstrably connected, and the models are
demonstrably not using what flows through it.

The statistical picture agrees and is worth stating in the form that most constrains it.
The report runs a family of **117 paired Wilcoxon signed-rank tests** across every arm,
anchor, metric and condition on real data, Holm-Bonferroni corrected over the whole family
(`report/report.md`, Paired Wilcoxon signed-rank). Seventeen results survive correction at
α = 0.05. **Every single one of them is on `mae`, and every single one has the model
performing worse than the persistence baseline** — larger error, not smaller. Not one
result on `direction`, `sharpe` or `total_return` survives. The smallest corrected p-value
achieved by any directional claim is 0.2605; by any Sharpe claim, 0.1914; by any return
claim, 0.3930 (computed from `results.csv` via `experiments.stats.wilcoxon` restricted to
`control='real'`, the same call `report.render` makes).

There is a ceiling here that must be stated alongside those numbers rather than left for a
reader to discover. The exact two-sided Wilcoxon test on sixteen paired folds cannot return
a p-value below **3.05 × 10⁻⁵** however large the effect
(`report/report.md`, Paired Wilcoxon signed-rank, preamble). With 117 tests in the Holm
family, the smallest attainable corrected value is bounded well away from zero. This
experiment is therefore capable of *failing to reject* and capable of detecting only quite
large effects; it is not capable of establishing a small edge, and no claim in this chapter
should be read as if it were. What it can do — and did — is show that the effects present
are not larger than the noise floor of a sixteen-fold design, while a control effect that
*is* larger shows up plainly at 4.95 points.

The one place the design does reject, it rejects against us. Seventeen corrected results
say the learned models have *higher* forecast error than a model that predicts zero return
every day. Section 4 returns to why that particular metric is the one with the statistical
power, and why that is a fact about the metric rather than about the models.

---

## 2. The execution finding: a backtest of a system that cannot be built

The second result is not about forecasting at all. It emerged from a live rehearsal in
August 2026 and it changes what every trading number in this study describes.

The backtester, as originally written and as every published figure in this project
assumed, exits a position in one of two ways: a protective stop below the entry, or a
take-profit limit above it, both resting at the broker, either able to fill intraday the
moment price crosses it. That is the standard construction and it is what the trading
literature assumes. It cannot be implemented at this venue.

Alpaca refuses every multi-leg order class on a fractional quantity. Submitting a bracket
order and, separately, an OCO order on a fractional position both return
`{"code":42210000,"message":"fractional orders must be simple orders"}` — measured directly
against the paper API rather than inferred from one case to the other. **[UNSOURCED — the
API responses are not in `results.csv` or `report.md`.]** Separately and more fundamentally,
a working sell order at this broker holds the *entire* position, so a standalone stop and a
standalone limit cannot coexist either: whichever is submitted second is refused for
insufficient quantity. That second constraint was measured against a 97.38-share position,
which establishes that it is **not** a fractional-only rule — whole-share sizing would not
lift it. What fractional sizing removes is the bracket order that would otherwise have been
the workaround. **[UNSOURCED.]**

A position at this venue can therefore carry exactly one broker-side protective order. The
stop is the one that must exist, because it is the one that bounds a loss. The take-profit
must be evaluated by the trading loop instead — and the loop reads completed daily bars
only, so it cannot observe an intraday crossing at all. It learns that the target was
reached when the bar closes, and acts at the next open.

This is a different system, and the study now measures both. The `target_in_loop` axis in
`results.csv` carries `false` for the classical broker-side target and `true` for the
buildable loop-side one, with `true` as the default so that the headline numbers describe
the system that exists. At the reference condition, paired fold by fold over sixteen folds
(`results.csv`, `anchor=0`, `lr=0.001`, `cutoff_period_days=5`, `control='real'`,
`channels='C0_base'`):

| arm | trades, broker-side | trades, loop-side | change | mean fold return, broker-side | loop-side |
|---|---|---|---|---|---|
| DLinear | 166 | 140 | −16% | −0.003119 | −0.003948 |
| FITS | 242 | 208 | −14% | +0.004188 | +0.007019 |
| WITS | 218 | 155 | −29% | +0.001801 | +0.004564 |

The trade count falls in every arm, and the mechanism is capital rather than signal. A
delayed exit holds capital through at least one additional session, so entries that the
broker-side system would have funded are never funded at all. Twenty-nine per cent of
WITS's trades simply do not happen.

The return difference does not resolve. Paired per fold, the mean differences are −0.000829
(sd 0.006392) for DLinear, +0.002832 (sd 0.008933) for FITS and +0.002763 (sd 0.023358) for
WITS, giving ratios of mean to standard error of **−0.52, +1.27 and +0.47** across sixteen
folds. All three sit far inside the noise, and the sign is not even consistent: DLinear is
worse under the buildable model while FITS and WITS are better. **The honest conclusion is
that the buildable system is measurably *different* and not measurably *worse*.**

The history of this finding belongs in the chapter, because the first description of it was
wrong in a way the next measurement caught. It was initially characterised as "32.7%
worse", from a single run over the whole universe. That figure is a real point estimate and
it stands as one, but the paired fold test says the difference is not separable from
fold-to-fold variation. Reporting the point estimate as the result would have been a
stronger claim than the data supports. This is the fifth occasion in this project on which
a first description was narrowed by a subsequent measurement, and the second on which the
narrowing corrected a description that had already been adopted. **[UNSOURCED — the count
of five is a project-history claim, not a figure in either file.]**

One structural point makes the interaction between this section and Section 1 clean rather
than lucky. `direction`, `mae` and `flatness` are **byte-identical across the two execution
arms** for every model, verified over the reference condition in `results.csv`. They must
be: they are properties of the forecast, and the axis changes only how a fill is priced.
The execution model therefore could not have rescued the null result and did not damage it.
The finding of Section 1 is the same finding under either system, now established for the
one that can actually be built.

---

## 3. The geometry critique: what a frequency-domain model learns when it learns nothing

FITS is a frequency-domain forecaster: it takes the real FFT of the input window, discards
bins above a cutoff, applies a learned complex linear map to the survivors, zero-pads and
inverts to a longer window. The learned map is the model. If the model has learned
something about equity markets, that map should differ when the market is replaced by
noise.

It does not, and the reason is geometric rather than empirical. Extending a window of
length `L` to length `L + H` moves every frequency's position in the output grid by a
factor `η = (L + H) / L`. A frequency that lands exactly on an output bin is passed
through; one that does not must be interpolated across the gap, on every window, by the
learned layer. The interpolation is a function of `frac(η·k)` — of the grid geometry — and
not of the data. A large share of the layer's capacity is therefore spent on an operation
that would be identical if the input were noise.

Measured, the learned frequency response on real data correlates with the response learned
on white noise at **+0.9485**, which is approximately 86% of the variance shared with a
model that has seen no market at all. **[UNSOURCED — the frequency-response curves are
produced by `test_frequency_response.py` and the GB-46 figure, neither of which is
`results.csv` or `report.md`.]**

What `results.csv` does support is the sweep over the cutoff, and it is the corroborating
evidence. The cutoff is FITS's one hyperparameter; varying it varies the number of retained
bins (`cof`) and the fraction of the layer occupied by the immovable zero-frequency row
(`dead_row_fraction`). Sweeping it changes the trade-off between resolution and capacity,
and the theory predicts no optimum — there is nothing to optimise for, because the capacity
is being spent on geometry either way (`report/report.md`, The COF sweep):

| cutoff (days) | retained bins | dead-row fraction | direction | edge vs always-long (points) | real − noise (points) |
|---|---|---|---|---|---|
| 2 | 60 | 0.0167 | 0.4971 | −5.93 | −0.71 |
| 5 | 24 | 0.0417 | 0.5055 | −5.09 | +0.31 |
| 10 | 12 | 0.0833 | 0.4969 | −5.95 | −0.79 |
| 20 | 6 | 0.1667 | 0.4956 | −6.08 | −0.93 |

The sweep spans a tenfold range in retained bins and a tenfold range in dead-row fraction.
Directional accuracy varies across it by one percentage point, unsystematically, and every
cutoff underperforms the always-long bar by five to six points. In three of the four
cutoffs FITS is *worse* on real data than on its own white-noise twin; in the fourth it is
better by 0.31 points. There is no cutoff at which the model works, and no trend pointing
toward one outside the swept range. The hyperparameter that theory says should not have an
optimum does not have one.

This matters beyond FITS, and it is why WITS exists. The critique above is either a claim
about FITS specifically or a claim about the general method of extending a *global* basis,
and one architecture cannot distinguish the two. WITS substitutes a discrete wavelet
transform — a local basis — for the rFFT in the same position, with the same instance
normalisation and the same supervision, so the comparison isolates the transform. The
structural prediction was recorded before implementation: because the retained wavelet
coefficient counts are identical at `L` and `L + H` (21 and 21 at the configured geometry),
the per-band maps are square and nothing is remapped onto a shifted grid, so the
interpolation FITS must learn has no counterpart. **[UNSOURCED — the coefficient-length
measurement is in `model/wits.py` and `test_wits.py`, not in either results file.]**

The corresponding null-control measurement for WITS — the correlation between its learned
operator on real data and on noise, which is the number that would confirm or refute the
critique — **has not been run.** This is the single largest gap in the chapter and Section
6 returns to it.

---

## 4. Methodology as result: what the checks caught, and what they did not

A study whose headline is negative rests entirely on whether its instruments were working.
This project produced several findings about its own instruments that are more transferable
than its trading results, and they belong in the chapter as results rather than as an
appendix.

**Two robustness tests, and neither subsumes the other.** Grid sensitivity asks whether a
finding survives moving the fold boundaries; a null control asks whether it survives
destroying the signal. They catch different things, which was established by both catching
something the other missed. Grid sensitivity killed a reported correlation of `r = −0.47`
that did not survive re-anchoring. A null control killed a measured "phase advance" of
+1.9582 days with a standard deviation of 0.0203 across 48 independently trained models —
a coefficient of variation of 1.04% — which had passed grid sensitivity at 48 of 48 cells
and then survived being trained on white noise, revealing it as a property of a
deterministic operator rather than of the market. Agreement that tight across four years of
equity returns is evidence about the machine, not about the data. **[UNSOURCED — both
figures are project history, not in either results file.]**

**A check can run, pass, and measure the wrong thing.** This is distinct from a check that
never ran, and it is the more expensive failure because nothing about it looks like a
failure. Four instances in this project: a contract property that compared a number to
itself and could not fail; `pytest -q | tail -4` reporting exit 0 on a suite with one
failure, because the exit status was the pipe's; `import glassbox` succeeding from the
repository root without the editable install, so the installation under test was never the
one anybody had; and a read-timeout dry run returning clean on twelve cycles in a healthy
period, when the value being validated had been derived from outage latency the dry run
could not reproduce. **[UNSOURCED.]**

A fifth instance is present in this chapter's own source material, and it was found while
drafting. `report/report.md` prints, immediately under the summary table:

> **Spearman(MAE, flatness) = +0.109** against **Spearman(MAE, direction) = −0.247** — MAE
> across these arms is close to a monotone function of how flat the forecast is and carries
> almost no information about accuracy.

The sentence is a fixed string in `report.py`; it is printed whatever the numbers turn out
to be. And at +0.109 the number does not support it. But the sentence is not wrong — the
statistic is being computed at the wrong level of aggregation. Recomputed over the six
**arms** rather than over the 624 arm-fold observations, Spearman(MAE, flatness) is
**+1.000**: the six arms rank identically by mean error and by mean flatness
(persistence 0.0153/0.000, WITS 0.0156/0.201, DLinear C0 0.0158/0.228, FITS 0.0159/0.254,
DLinear C2 0.0162/0.306, from `results.csv` grouped by arm). The relationship §7.3 asserts
is not merely present, it is perfect — and the report's own instrument was pointed one
level too low to see it, while a hardcoded sentence asserted the conclusion regardless.
Both halves of that are the finding.

This has a direct bearing on Section 1. The seventeen results that survive Holm correction
are all on MAE, and MAE is now shown to rank these arms exactly as flatness does. The
metric with all the statistical power in this study is a metric that measures how close to
zero a forecast is. Persistence forecasts exactly zero and has both the lowest error and
zero flatness. The models forecast slightly more than zero and are punished for it. That is
why every surviving result says a model is worse than persistence, and it is why this
project bans MAE as a headline and requires flatness in the adjacent column.

**A feature can be built, exported, unit-tested and never invoked.** On 28 August 2026,
while producing this report's own screenshots, GB-53's spectral panel rendered nothing.
`explain_spectral` — the function that decomposes a FITS forecast by frequency — was
defined, exported and covered by its own test file, and had **no caller anywhere in the
package**. Every decision record ever written by the live loop or the replay carried
`per_frequency=None`. The panel was complete, the decomposition was complete, and the two
had never been connected; the task had been closed as done. The tests passed because they
exercised the function directly, which is not the same claim as *the system produces this*.
The corrective question is narrow and general: **when a feature is finished, ask what would
fail if the wiring were deleted and only the function remained.** Here the answer was
nothing, which is the definition of an untested integration. Wired now, and the evidence is
67 of 67 replayed records carrying the decomposition with both views closing on the same
total to 2.551e-09.

**The same flag defect, twice, in the same file.** `smoke_offline --model fits
--prepare-replay 13 DIR` parsed the flag, warned about nothing, and wrote a **DLinear**
checkpoint, because the prepare paths called the config loader directly while only the
smoke run received the override. GB-24 records the first occurrence, where the CLI could
not select the model the study is about. What makes the second worth reporting is the
timing: it was caught by a screenshot that came back empty, **before** any grid ran on it,
rather than after a run whose results were quietly about a different architecture. Had it
gone unnoticed, the study's FITS arm would have been DLinear — which is the failure GB-49
came within one task of shipping in 2025.

**The honesty mechanism committed the defect it was built to prevent.** The console
carries a source pill on every card — LIVE, BACKTEST or REPLAY, with a required detail —
for one reason: this system has made two live trades and stands aside on most bars, while
the backtest has 166 trades over sixteen folds, and a panel that looked live while showing
backtest numbers would discredit the project's central claim more effectively than any
missing feature. On 28 August two of those cards read one artefact and were labelled from
another: they announced **"folds 1-16" while holding three**. The pill was derived once for
the page from the frame most cards used, and the two cards that read a different file
inherited a claim about somebody else's rows.

This is the two-places family with the second place being *a label about the data*, and it
is worth stating generally because the irony is the instructive part: **building a
mechanism against a class of error does not place the mechanism outside that class.** A
wrong pill is worse than no pill, because it converts *I should check this* into *I have
checked this*. The corrective is unglamorous and the same as every other instance — derive
the label from the thing it labels, and pin the call sites with a test.

**A practice adopted for one reason keeps covering a failure nobody had connected to it,
and that is now a pattern rather than luck.** Three instances. The data snapshot was
committed so that a result could name the vintage it ran against; it is what made GB-59's
clean-clone audit possible at all, because a clone with no data can run nothing. The
one-decision-per-completed-bar rule was ruled to stop 1,950 identical rows a day being
written; it is what stopped a stray relaunched rehearsal opening a second position against
a live account, because the bar it would have decided had already been decided. And
committing this console per region was asked for so a redesign could be bisected; what it
actually paid for was recovery, when a scripted edit deleted sixteen tests from three
already-committed regions and the previous commit made that a two-minute restore rather
than a reconstruction from memory.

The transferable form is not *these three practices are good*. It is that a constraint
which forces work to be **decomposed and recorded at each step** buys options that cannot
be named in advance — and that this is an argument for the constraint which does not depend
on the reason it was adopted, and survives that reason turning out to be the less important
one.

**A test double more permissive than the system it stands for is a second implementation of
your assumptions.** At one point 1,181 tests passed while the live execution path could not
place a protected order at all. The fake broker permitted a standalone stop and a standalone
limit to coexist on one position; the real broker does not, for the reason given in Section
2. The fake was permissive in exactly the dimension the protection policy depended on, so
every test of that policy was a test of the fake's belief about Alpaca rather than of
Alpaca. Making the double faithful turned 26 tests red across three files — a measurement
of what had never been tested, not a regression. One of those tests was deleted rather than
repaired, because its premise was impossible: it asserted that a filled entry is protected
by two standalone orders. A test deleted because the thing it asserts cannot exist is a
finding. **[UNSOURCED.]**

**Reproducibility.** [UNSOURCED — see Section 6.]

The organising principle behind all of these, and the one this project would offer another
student, is that **only something that runs is a mechanism**. A fact recorded in two places
diverges unless a test compares them; a test gated by a cache or a marker is a note until it
executes in CI; a description of intended behaviour in a comment or a plan constrains
nothing; and a capability reachable only by a flag somebody must remember is, in practice,
a capability that does not run. That last form is the subtlest, because there is no failing
state to detect: a multi-day live run was launched without the flag that made it multi-day,
completed one session, stopped, and looked exactly like a run that had finished its work.
**[UNSOURCED.]**

---

## 5. Limitations

These are stated rather than defended, and each is a reason to read a specific claim in
this chapter more weakly than it might otherwise read.

**The evaluation window is recent and short.** The study uses sixteen walk-forward folds
over a data snapshot ending 2026-08-13 (`report/report.md`, header). It therefore contains
neither the 2018 volatility episode nor the March 2020 crash. Every drawdown, Sharpe and
maximum-drawdown figure in this chapter describes a period without a severe market
dislocation, and the risk machinery is correspondingly untested against one. The exact
calendar span of the folds is **[UNSOURCED]** — `results.csv` records fold indices, not
fold dates.

**The validation split is used twice**, for early stopping and for threshold calibration.
This does not leak into the test period, but it means the validation Sharpe is a selected
maximum and is not reportable. The direction of the resulting bias is worth stating,
because it makes this a caveat rather than a reason to discount the numbers: thresholds are
calibrated on forecasts already slightly overfit to validation, so on test they are
*miscalibrated* rather than *inflated*. The double use degrades test performance; it does
not flatter it. Every test figure here is therefore a lower bound with respect to this
particular flaw. **[UNSOURCED — the argument is sound but the magnitude has not been
measured.]**

**Reproducibility is same-hardware only.** Determinism was verified by re-running the grid
on one machine. Cross-architecture reproducibility has not been tested, and floating-point
reduction order in the training path makes it unlikely to hold bit-exactly.
**[UNSOURCED — see Section 6.]**

**The risk layer is correct, enforced and inert.** Position sizing, exposure caps and
per-symbol limits are implemented and tested, but at a five-symbol universe
(`glassbox/config/settings.yaml`: AAPL, MSFT, NVDA, AMZN, GOOGL) with a 10% cap per
position the portfolio constraints are never the binding one. The layer has therefore been
demonstrated to be *correct* and has never been demonstrated to be *effective*, which are
different claims. A twenty-symbol universe is cached and prepared but the deployed
configuration and the grid still run five.

**The null controls exist at one anchor only.** This is a limitation of the present
`results.csv` and it qualifies a claim made in Section 1. White-noise and shuffled controls
were run at anchor 0; anchors 21 and 42 carry real data only (`results.csv`, grouped by
anchor and control: anchor 0 has 208 noise and 112 shuffled rows, anchors 21 and 42 have
112 real rows each and nothing else). So "flat at all three anchors" is established for
real data, and "holds against both nulls" is established at anchor 0. The conjunction —
that the null result is grid-insensitive *and* control-verified at every anchor — is not
yet supported.

**Sharpe is computed on a selected subset of folds.** Where an arm stands aside for a whole
fold it produces no Sharpe, so Sharpe comparisons rest on between 6 and 14 folds with a
mean of 10.2, against 16 for direction, MAE and total return (computed from `results.csv`
via `experiments.stats.wilcoxon`). The folds that survive are those in which the model
chose to trade, which is not a random subset. Sharpe figures in this chapter should be read
as descriptive, not inferential.

---

## 6. What this chapter cannot yet support

The following claims appear above marked **[UNSOURCED]**. Each is believed true and each
needs a measurement, a re-run or an extracted artefact before the chapter is submissible.

**Blocking — the chapter's argument depends on these:**

1. **The WITS null control.** The correlation between WITS's learned operator on real data
   and on white noise, against FITS's +0.9485. This is GB-66's stated acceptance criterion
   and the entire reason WITS was built. Without it Section 3 poses its question and does
   not answer it, and the geometry critique remains a claim about FITS alone.
2. **The FITS +0.9485 figure itself.** It is produced by the GB-46 frequency-response work
   and is not in either results file. It needs to be re-derived from the current snapshot
   and committed as a figure with its own provenance, or the headline of Section 3 rests on
   a number a reader cannot check.
3. **Null controls at anchors 21 and 42.** Currently anchor 0 only. Until they run, the
   sentence "flat at all three anchors under both nulls" cannot be written. This is a grid
   re-run, not new code.

**Required for the methodology section to be a result rather than an anecdote:**

4. **The Alpaca API responses** — the two `42210000` refusals and the
   `insufficient qty available` refusal on 97.38 shares. These are the empirical basis of
   the entire execution finding and currently exist only as transcript. They need to be
   captured as a dated artefact under version control.
5. **The reproducibility audit.** Referenced in Section 5 and entirely absent: no figure,
   no procedure, no record of what was re-run or how equality was checked. Either the audit
   is run and recorded, or the claim is withdrawn.
6. **The `r = −0.47` and the +1.9582-day phase advance.** Both are quoted as evidence that
   the two robustness tests catch different things, and neither is in the results files.
7. **The FakeBroker counts** — 1,181 passing, 26 turning red. Recoverable from git history
   and CI logs, but not from `results.csv`.
8. **The count of "five narrowings".** Either enumerate them with dates or drop the count
   and keep the two that are documented.

**Needed to state a limitation precisely:**

9. **Fold calendar dates.** `results.csv` carries fold indices only, so "no 2018 and no
   March 2020" is currently an inference from the snapshot date rather than a fact read off
   the file. The walk-forward split should emit fold start and end dates.
10. **The magnitude of the validation-reuse bias.** The direction is argued; the size is
    unmeasured. A single re-run with thresholds calibrated on a held-out slice would bound
    it.

**One finding that arrived while drafting and should be fixed before the report is
regenerated:**

11. **`report.py` computes the MAE/flatness Spearman over arm-folds and prints a sentence
    that describes the arm-level relationship.** The arm-level value is +1.000 and the
    printed value is +0.109. The number should be computed at the level the claim is made,
    and the sentence should be derived from the number rather than asserted beside it —
    otherwise it is exactly the failure mode Section 4 is about, sitting inside the report
    that Section 4 appears in.
