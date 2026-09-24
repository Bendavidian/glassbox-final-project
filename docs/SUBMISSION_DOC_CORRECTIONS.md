# Submission document corrections

Claims in the submission chapters that later measurement contradicted, recorded here
rather than by silently rewriting the chapter. Each entry names the document, quotes the
claim verbatim, gives the corrected figures, and says what produced the correction.

**Why this file exists rather than an edit.** A chapter that is quietly rewritten leaves
no evidence that the earlier number was ever believed, and the fact that a measurement
overturned it is itself a result. The chapters are corrected from this list at submission
time; until then the list is the record.

---

## 1. GB-57 §1 — "Not one result on `direction`, `sharpe` or `total_return` survives"

**Document:** `docs/GB57_RESULTS_AND_DISCUSSION.md`, §1, "The result: nothing was found,
and the instrument was working".

**Claim as written:**

> The report runs a family of **117 paired Wilcoxon signed-rank tests** across every arm,
> anchor, metric and condition on real data, Holm-Bonferroni corrected over the whole
> family. Seventeen results survive correction at α = 0.05. **Every single one of them is
> on `mae`, and every single one has the model performing worse than the persistence
> baseline** — larger error, not smaller. Not one result on `direction`, `sharpe` or
> `total_return` survives. The smallest corrected p-value achieved by any directional
> claim is 0.2605; by any Sharpe claim, 0.1914; by any return claim, 0.3930.

**Status: false as of the 2 Sep 2026 grid** (20-symbol universe, `results.csv` written
15:35:27Z, 1110 rows). The family size and the correction are unchanged; the survivors are
not.

| | five symbols | twenty symbols |
|---|---|---|
| family size | 117 | 117 |
| survive Holm at α = 0.05 | 17 | **29** |
| metrics among survivors | `mae` only | **`direction` (3), `mae` (22), `total_return` (4)** |
| all survivors model-worse | yes, 17/17 | **yes, 29/29** |
| smallest corrected p — `direction` | 0.2605 | **0.0036** |
| smallest corrected p — `sharpe` | 0.1914 | 0.1531 |
| smallest corrected p — `total_return` | 0.3930 | **0.0058** |

**Corrected sentence.** Twenty-nine results survive correction. Twenty-two are on `mae`,
three on `direction` and four on `total_return`; **all twenty-nine have the model worse
than its own reference**. `sharpe` remains the one metric on which nothing survives. The
three directional results are all at anchor 42, one for each of DLinear, FITS and WITS,
each 5.2 to 6.3 points below the always-long bar.

**The interpretation must change with the number, and in the opposite direction to the
obvious reading.** More survivors does not mean the models got worse. Joining the two
Wilcoxon families on the full condition key, for the same 29 cells:

| metric | mean abs effect (5) | mean abs effect (20) | ratio | grew in |
|---|---|---|---|---|
| `direction` | 0.0599 | 0.0580 | **0.97** | 2 of 3 |
| `total_return` | 0.0827 | 0.0473 | **0.57** | 0 of 4 |
| `mae` | 0.000620 | 0.000475 | **0.77** | 4 of 22 |

**Effect sizes did not grow; in 23 of 29 cells they are the same or smaller.** One further
cell rose by 2.6e-5, which is a tie at any reportable precision and is counted as grown here.
The number
of paired observations is 16 at both universe sizes, so the extra power did not come from
more folds. It came from the collapse of the paired-difference standard deviation — the
quantity the signed-rank test actually consumes — as each fold's metric moved from an
average over five symbols to an average over twenty:

| arm (anchor 42, real, lr 1e-3) | metric | sd of paired difference (5) | (20) | ratio |
|---|---|---|---|---|
| persistence | `total_return` | 0.11663 | 0.04633 | 0.397 |
| FITS | `total_return` | 0.09623 | 0.03068 | 0.319 |
| WITS | `total_return` | 0.10144 | 0.03246 | 0.320 |
| DLinear | `direction` | 0.06571 | 0.02576 | 0.392 |
| FITS | `direction` | 0.06514 | 0.04209 | 0.646 |
| WITS | `direction` | 0.05708 | 0.04217 | 0.739 |

**So the correct claim is that Phase 1 was underpowered, and the deficit was always
there.** On `total_return` the models are measurably *less* bad at twenty symbols — the
shortfall against buy-and-hold shrank to 57% of its five-symbol size — and the result
became significant anyway, because the estimate is roughly three times more precise. The
Phase 1 null on `direction` and `total_return` was in part a statement about the
resolution of a five-symbol design, not about the models.

**This does not weaken the chapter's conclusion; it strengthens it.** Every one of the 29
surviving results has the model losing to its reference. Where Phase 1 could only fail to
reject, the twenty-symbol study rejects — against the models.

**Unchanged and still worth stating beside it:** the exact two-sided Wilcoxon on 16 paired
folds still cannot return a p below **3.05 × 10⁻⁵**, because n is still 16. The design's
ceiling did not move; only its noise floor did.

**Produced by:** `python -m glassbox.experiments.report` run twice — once on the
20-symbol `results.csv`, once on the 5-symbol table recovered from HEAD
(`git show HEAD:results.csv`, blob `93d1241`) — and the two Wilcoxon tables joined on
`(anchor, lr, control, target_in_loop, cutoff_period_days, model, channels, metric,
reference)`. The five-symbol side reproduces every published Phase 1 figure it was checked
against to four decimal places, which is the control that makes the twenty-symbol column
readable.

---

## 2. GB-57 §1 — the always-long reference rate at anchor 0

**Document:** `docs/GB57_RESULTS_AND_DISCUSSION.md`, §1.

**Claim as written:**

> The always-long reference rate is 0.5395 at anchor 0, 0.5523 at anchor 21 and 0.5654 at
> anchor 42.

**Status: wrong at anchor 0, and wrong on the five-symbol data it was written from** —
so this is a pre-existing error, not a consequence of the universe change. Anchors 21 and
42 reproduce exactly. Anchor 0 recomputes to **0.5563**, and the null-control table three
lines above it in the same section gives buy-and-hold's real-data direction at anchor 0 as
**0.5560** — which the project asserts *is* the always-long bar, by a self-check that runs
on every fold. The section therefore contradicts itself.

At twenty symbols the three rates are **0.5529, 0.5479, 0.5565**, all slightly lower, as a
broader and less concentrated basket carries less upward drift.

**Found because this pass recomputed the number instead of copying it.**

---

## 3. Any claim that the arms are flat "at all three anchors under both null controls"

**Status: structurally unavailable, not merely unmeasured.** The distinction matters,
because "we did not measure it" invites someone to go and measure it, and "the design
cannot express it" does not.

Row counts by control and anchor, from the 2 Sep 2026 twenty-symbol `results.csv`
(`skipped=False`), verbatim:

```
control x anchor row counts
anchor      0    21   42
noise     208     0    0
real      544   112  112
shuffled  112     0    0
```

**There are no `noise` or `shuffled` rows at anchors 21 or 42.** The sensitivity design is
a **star, not a cross product** (`study.conditions`): each axis departs once from a common
centre, and the control axis therefore has its spokes at anchor 0 only. A null-control
comparison exists at the centre and nowhere else.

**The consequence for the chapter.** Grid sensitivity and the null control are this
project's two independent robustness checks — §"Suspicious stability" records that the
`r = −0.47` correlation died to a grid shift and would have passed a null control, while
the FITS phase advance died to a null control after passing grid sensitivity at 48 of 48
cells. **The two cannot currently be crossed.** No statement of the form "the arms score
better on noise at every anchor" can be supported, and any document making one is wrong by
design rather than by omission. Closing it means adding `noise` and `shuffled` conditions
at anchors 21 and 42 — a cross rather than a star on those two axes — which is a scope
decision, not a bug fix.

---

## 4. Buy-and-hold's degradation under the noise control is significant after correction

**Addition rather than correction**, and it strengthens §1's central argument.

GB-57 §1 argues that buy-and-hold is the only arm that degrades under the white-noise
control, and that a control moving for exactly one arm — the one whose mechanism says it
must — separates *"we found nothing"* from *"we measured that there was nothing to find."*
That argument is currently made on the size of the gap alone. It now has a p-value.

Paired by fold within the reference condition (anchor 0, lr 1e-3, cutoff 5,
`target_in_loop=True`, `C0_base`; buy-and-hold matched on the rest, since it carries no
`channels` value), across the `control` axis, twenty-symbol `results.csv`:

| arm | n | mean(noise − real) | sd | W | p | p_holm |
|---|---|---|---|---|---|---|
| **buy_and_hold** | 16 | **−0.0517** | 0.0593 | 14.0 | **0.0034** | **0.0134** |
| dlinear | 16 | +0.0184 | 0.0301 | 22.0 | 0.0174 | 0.0521 |
| fits | 16 | +0.0087 | 0.0346 | 49.0 | 0.3484 | 0.6968 |
| wits | 16 | +0.0035 | 0.0386 | 65.0 | 0.8999 | 0.8999 |

**Buy-and-hold's degradation survives correction at p_holm = 0.0134. No model arm's gap
does.** The instrument detects a control effect where the mechanism predicts one and
detects nothing on the arms that never had a forecast, and both halves of that are now
measured rather than asserted.

**THE AGGREGATION CAVEAT, AND IT MUST TRAVEL WITH THE NUMBER.** This test **crosses the
`control` column**, and `control` is one of the columns that identifies a condition
(`Condition.columns`). The report's family of 117 pairs by fold **within** a condition and
therefore contains no test of this shape. **These four tests are not among the 117**, and
the Holm correction above is over **these four arms only** — stated here rather than
implied, because quoting `p_holm = 0.0134` beside the report's corrected values would
invite a reader to assume one family where there are two.

**DLinear's gap is reported as unresolved, not as a finding.** At +0.0184 it is the
largest of the three model gaps and the only one with a consistent fold pattern (positive
in 13 of 16, against WITS's 7 of 16, which is a coin flip), and it clears the uncorrected
threshold at p = 0.0174 — but **it fails Holm at 0.0521**. The chapter's claim stays
*"destroying the signal costs these models nothing"*; the stronger reading, *"the models
do better on destroyed signal"*, is not supported at sixteen folds and must not be written.

---

## 5. GB-66's stated acceptance criterion cannot be met, and the criterion is the error

**Document:** `GLASSBOX_PHASE2_EXPANSION.md`, §GB-66 "Done when".

**Claim as written:**

> **The null control is run and the correlation between the learned response on real data
> and on white noise is reported against FITS's +0.9485.** This is the acceptance
> criterion that matters; performance is not.

**Status: structurally unmeasurable.** Not unmeasured — *unmeasurable*, and no amount of
running closes it.

`response` is the magnitude of the learned gain **per retained rFFT bin**: **23 values**
at `input_len` 120 under the 5-day cutoff. WITS is configured `retained_bands: 2`
(cA3 + cD3), so the counterpart curve holds **two** values, and a Pearson correlation over
two points is ±1 by construction and carries no information. Taking all four DWT bands
gives n = 4 against 23, and a per-band aggregate over an octave is a different object from
a per-bin gain at one frequency. **A correlation over 2 or 4 values must never be printed
beside +0.9485 as though it were the same measurement.**

`scripts/operator_null_control.py` already refuses it, twice, and said so before anyone
asked: `measure()` gates the row behind `if model == "fits"` — *"it exists only where
there are bins to have it"* — and `_correlate` returns NaN for fewer than three values.

**This is a result, not a gap.** The expansion set a criterion that could not be met, and
finding that out required building WITS. It says the geometry critique is **Fourier-
specific by construction**: the thing being criticised — a global basis remapped onto a
shifted grid, forcing the learned layer to interpolate — has no counterpart in a basis
whose per-band maps are square. **GB-66 is complete.**

**The verdict rests on the quantity that does transfer**, and that comparison is
legitimate where the response comparison is not: the flattened `(H, L)` forecast matrix,
**n = 480 for both architectures**, computed by the same function on the same folds with
the same seed and budget, differing only in what the close channel holds.

| | five symbols (27 Aug) | twenty symbols (3 Sep) |
|---|---|---|
| FITS operator | −0.0127 (t vs 0 = −0.44) | **+0.1946** (t = +7.24) |
| WITS operator | +0.0331 (t = +0.75) | **+0.0838** (t = +2.56) |
| paired WITS − FITS | +0.0458 | **−0.1108** |
| Wilcoxon p | 0.5282 | **0.0042** |
| WITS lower in | 9 of 16 folds | **13 of 16 folds** |
| verdict | **did not hold** | **HELD** |

**The five-symbol study would have recorded "prediction did not hold" from a comparison
with no power** — neither correlation was distinguishable from zero. This is GB-61's
Wilcoxon finding in a second, independent place.

---

## 6. FITS's +0.9485 and the 86% are five-symbol figures, and the 86% is not reproducible

**Document:** `docs/GLASSBOX_PROJECT_SPEC.md` §2.4, `docs/GB57_RESULTS_AND_DISCUSSION.md`
§3, `IDEAS_PARKED.md`, `glassbox/model/wits.py`, `glassbox/dashboard/app.py`
(`SPECTRAL_NOISE_CORRELATION = 0.9485`).

**Three numbers that must be kept apart**, because they are currently used
interchangeably:

| figure | what it is |
|---|---|
| **+0.9485** | GB-46's published headline, from its own frequency-response work |
| **+0.8967** | `operator_null_control.py` over 16 folds at **five** symbols, 27 Aug |
| **+0.9780 ± 0.0105** | the same script at **twenty** symbols, 3 Sep |

**The critique got stronger, not weaker.** r² rises **0.804 → 0.957**, and the
real-versus-noise curves moved *closer*: mean absolute difference **0.0390 → 0.0256**.
Whatever else changed at twenty symbols, the data-independence the geometry critique rests
on is larger.

**The 86% cannot be recomputed from the artefact that is supposed to hold its evidence.**
Spec §2.4 derives it as `1 − 0.0319 / 0.2335` — mean absolute difference over **curve
range** — and `report/null_control.csv` records the difference but **not the range**. The
number therefore has no reproducible path from the file, which is the same defect as the
16.00%/peak-51% reconstruction, in a claim the chapter leans on harder. Restated in the
quantity that *is* reproducible: **r² = 95.7% at twenty against 80.4% at five.** The
substance holds and the published 86% understates it.

**A note that must travel with this number, or the project's own heuristic will reject
it.** Across 16 independently trained folds the response correlation has **CV 1.08%**
(0.0105 / 0.9780), against 3.09% at five. CLAUDE.md records that a CV near 1% across
independently trained models is evidence the measurement is *about the machine rather than
about the market* — the test that killed the FITS phase advance at CV 1.04%. **Here that
is the finding, not the alarm.** The claim under test is precisely that this curve is
deterministic operator geometry rather than market structure, so a 1% CV corroborates it.
Applied mechanically, the heuristic would reject the one result it confirms.

---

## 7. The narrowed claim — "geometry in the envelope, not in the structure" — no longer holds as stated

**Document:** `scripts/operator_null_control.py`, `verdict()`, which is where the
narrowing is currently written; it has not yet reached a chapter, and
`GB57_RESULTS_AND_DISCUSSION.md` §3 still carries the unnarrowed +0.9485 line marked
`[UNSOURCED]`.

**Claim as written:**

> FITS's operator correlation is far below its response correlation. The two measure
> different things — a signed (H, L) matrix against a curve of gain magnitudes — so the
> 86% figure is a claim about the gain curve specifically and **must not be quoted as a
> claim about the whole learned operator**.

**The basis for that narrowing was a five-symbol measurement in which FITS's signed
operator was indistinguishable from zero:** −0.0127, sd 0.1141 over 16 folds,
**t vs 0 = −0.44**. On that evidence the envelope was grid-determined and the structure
within it was not, and the narrowing was correct.

**At twenty symbols it is +0.1946, sd 0.1076, t vs 0 = +7.24.** It is no longer
indistinguishable from zero — it is seven standard errors from it.

**Restatement.** More of FITS is grid-determined than the narrowed claim allowed. The
data-independence is present **in the gain envelope and in part of the signed operator
too** — r² = 0.038 of the operator, small in absolute terms but no longer nothing, and
established rather than merely unrefuted. The narrowing must be rewritten as a statement
about *degree*: the envelope is overwhelmingly geometry (r² 0.957), the signed operator
is partly geometry (r² 0.038 and firmly non-zero), and the two must not be quoted with
the same number — which was the correct half of the original claim and survives.

**WITS, stated honestly, because the temptation is to state it otherwise.** WITS's signed
operator is **+0.0838, t = +2.56** — also non-zero. **WITS does not escape the pathology;
it escapes it less** (paired difference −0.1108, Wilcoxon p = 0.0042, lower in 13 of 16
folds). That is still GB-66's finding and it is *stronger* for being stated this way: a
local basis reduces grid-determination rather than removing it, which is what a real
mechanism looks like. A claim that WITS escaped it entirely would be contradicted by its
own t-statistic.

---

## 8. Phase 1's negative verdicts at five symbols were power failures, twice over

**For GB-57 §57.8.** Two independent negative results, produced by different instruments
on different quantities, both reversed at twenty symbols — and in both cases the reversal
came from precision rather than from a larger effect.

| | GB-61: the Wilcoxon family | GB-66: the operator comparison |
|---|---|---|
| verdict at five | 17 of 117 survive Holm, **none** on `direction`, `sharpe` or `total_return` | prediction **did not hold**: p = 0.5282, 9 of 16 folds |
| verdict at twenty | **29** survive, spanning `direction`, `mae`, `total_return` | prediction **HELD**: p = 0.0042, 13 of 16 folds |
| effect size | **same or smaller in 23 of 29 cells**; `total_return` shrank to 57% | FITS −0.0127 → +0.1946, WITS +0.0331 → +0.0838 |
| n | **16 folds, unchanged** | **16 folds, unchanged** |
| what actually moved | paired-difference sd fell by **2.5–3×** | neither correlation differed from zero at five (t = −0.44, +0.75); both do at twenty (t = +7.24, +2.56) |

**Two instances is a claim about the design, not a coincidence.** A five-symbol,
sixteen-fold walk-forward could not resolve effects that were present the whole time.
Each fold's statistic was an average over five symbols; at twenty the same statistic is
four times better sampled, the per-fold noise collapses, and the same underlying effects
clear the threshold. Nothing about the models changed.

**The consequence for how Phase 1's negatives should be read.** A negative result from
that design is evidence of *absence of a large effect*, not evidence of absence. Where
Phase 1 reported "no result survives", the correct reading is "no result survives **at
this resolution**" — and in both cases examined here, results did survive once the
resolution improved. **This does not weaken the study's headline**, which is that the
models do not beat their references: every one of the 29 surviving Holm results has the
model *losing*, and the twenty-symbol operator comparison confirms the geometry critique
rather than overturning it. It weakens only the *negatives about the negatives* — the
places where Phase 1 said an effect could not be detected and treated that as evidence
the effect was not there.

---

## 9. GB-57 §57.10 — the `sd` instance was right about the mechanism and wrong in its figures

**Document:** the GB-57 §57.10 skeleton (v2, 3 Sep 2026), second principle, *"a green
result answering a different question is the most comfortable kind of wrong"*.

**Claim as written:**

> **The report's `sd` column is the arm's spread, not the paired difference**, and for
> total_return it moved the *wrong way* (0.0129 → 0.0162). Stopping there would have
> reported the power mechanism backwards with correct numbers attached.

**Status: the mechanism is confirmed; the parenthesis was wrong and is replaced.**

The mechanism holds exactly as stated. `experiments/stats.py:232` computes
`"sd": float(values.std(ddof=1))`, where `values` is the **arm's own per-fold metric**.
The paired difference is formed on the line above and reaches only `mean_delta` and the
signed-rank test itself. So the column a reader sees beside a p-value is not the quantity
that produced it.

**The two figures were not sd values.** `0.0162` is DLinear C2's **mean MAE** from the v1
chapter's flatness ranking (`GB57_RESULTS_AND_DISCUSSION.md`, the
`persistence 0.0153 / WITS 0.0156 / DLinear C0 0.0158 / FITS 0.0159 / DLinear C2 0.0162`
line); it appears nowhere in `report/report.md` as an sd. `0.0129` **is** a real figure —
DLinear, `C0_base`, `total_return`, five symbols — so the pair was one correct number and
one substituted from a different metric, which is the harder kind to notice.

**Recomputed, over the 39 `total_return` cells present at both universe sizes**, by
running the committed `stats.wilcoxon` on `results.csv` at HEAD and on the five-symbol
table at `da37401^`:

| quantity | at 5 | at 20 | ratio | grew at twenty |
|---|---|---|---|---|
| **reported `sd`** (the arm's own spread) | 0.01605 | 0.02286 | **×1.424** | **25 of 39** |
| **true paired-difference sd** (what the test consumes) | 0.13567 | 0.05154 | **×0.380** | **0 of 39** |

Reference condition (anchor 0, lr 1e-3, real, `target_in_loop=True`, cutoff 5, `C0_base`):

| arm | reported sd, 5 → 20 | paired-difference sd, 5 → 20 |
|---|---|---|
| persistence | 0.00000 → 0.00000 | 0.15467 → 0.05989 |
| dlinear | **0.01295 → 0.02528** | 0.14703 → 0.03978 |
| fits | 0.02994 → 0.03325 | 0.14131 → 0.05162 |
| wits | 0.02190 → 0.03408 | 0.14583 → 0.04733 |

**The claim is stronger than the draft made it, not weaker.** The two quantities move in
**opposite directions**, and not marginally: the reported column rises in 25 of 39 cells
while the quantity the test actually consumes falls in **39 of 39**. A reader auditing
§57.8's power argument against the `sd` column printed beside it would have found the
evidence pointing the wrong way and concluded the argument was backwards.

**What produced the correction.** The evidence audit of 3 Sep 2026, which required every
57.10 instance to carry a commit, `file:line` or test node ID before it could appear in
the chapter. This instance carried a mechanism that verified and figures that did not.
**It is itself an instance of the principle it describes**: the sentence was true, the
numbers beside it were about a different quantity, and nothing about the pair looked
wrong — the chapter's own drafting committing the error the chapter is about.

---

## 10. GB-57 §57.2 — the `C0_base` filter drops the arm the null-result argument rests on

**Document:** the GB-57 §57.10 skeleton (v2, 3 Sep 2026), second principle, and the filter
note under §57.2's five-beside-twenty direction table.

**Claim as written:**

> **A `channels == C0_base` filter silently drops buy-and-hold**, the one arm whose
> degradation carries the entire null-result argument.

**Status: confirmed, mechanically, and it needs no anecdote.** Counted directly on
`results.csv` at HEAD:

| | rows |
|---|---|
| `results.csv`, excluding the header | 1110 |
| `model == buy_and_hold` | **224** |
| of those, carrying a non-empty `channels` value | **0** |
| kept by `channels == C0_base` | 608 |
| — of which `buy_and_hold` | **0 of 224** |
| kept by `channels == C0_base or model == buy_and_hold` | 832 |

Every `buy_and_hold` row carries `channels` as NaN, because buy-and-hold has no feature
configuration to name — it is a reference arm, not a model arm. So the equality is not
merely unlikely to match, it **cannot** match: the filter that looks like "hold the
feature configuration fixed" is the filter that removes the reference the comparison is
against, and it removes 100% of it while leaving 608 rows behind that look like a complete
table.

**Why this is the second principle and not carelessness.** The filter is correct for every
model arm, the resulting table is well-formed, every remaining number is right, and the
missing row is the **−0.0517 degradation that separates "we found nothing" from "we
measured that there was nothing to find"** (entry 4). A green table answering a different
question, with no visible defect — and the arm it drops is the one §57.3 calls the whole
argument.

**What produced the correction.** Found while assembling §57.2's table during the 3 Sep
drafting pass; the committed table at entry 4 above already carries the correct filter and
states it in its caption. Recorded here because the chapter documents this failure class
in other people's code and in its own tooling, and declining to record the two instances
its own drafting produced would make the section worth less than it claims to be.
