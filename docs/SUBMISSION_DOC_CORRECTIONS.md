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
| `direction` | 0.0599 | 0.0580 | **0.97** | 1 of 3 |
| `total_return` | 0.0827 | 0.0473 | **0.57** | 0 of 4 |
| `mae` | 0.000636 | 0.000491 | **0.77** | 3 of 22 |

**Effect sizes did not grow; in 26 of 29 cells they are the same or smaller.** The number
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
