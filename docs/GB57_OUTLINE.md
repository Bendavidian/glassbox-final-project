# GB-57 — Results, Discussion, Future Work

**Outline v3, 3 September 2026, gated on the evidence audit.** All Phase 2 runs are
complete. Every `‹PENDING›` from v1 is now filled with a measured value; `‹GAP›`
marks evidence the project still does not hold.

Commits carrying this material: `da37401` (grid at twenty), `c22d30c` (risk instrumentation),
`e7bdc2c` (GB-66 operator pass), `02d0eb4` (corrections). Suite 1390 passed, exit 0.

**Reporting rule, applied to every table:** each result is a delta against *its own*
reference — MAE against persistence, direction against always-long, trading metrics
against buy-and-hold. No composite score. No number without its reference on the same line.

**Audience:** a CS academic who does not work in finance.

---

## 57.1 — What was measured, and against what

Half a page. GB-56 owns methods; this exists so a reader knows what a row *is*.

- The star: arms, anchors, controls. One diagram. **Note that it is a star and not a
  cross** — this matters in 57.9.
- The three references, and why three rather than one.
- Walk-forward, 16 folds, H-bar embargo.
- **The universe changed mid-study.** Phase 1 measured five symbols; everything below is
  twenty. Say it here, once, plainly.
- Grid wall time at twenty: **1 h 40 m 56 s** (6,056 s), single-threaded by design
  (`BLAS threads 14 → 1`) for bit-identical reproducibility. Predictions of ~88 and ~135
  minutes were both guesses; this is the measured figure.

---

## 57.2 — The null result

**The headline, and it goes first.**

> These models extract nothing from this market that they do not also extract from white
> noise.

A statement about the models, not the market. Say so in the first paragraph, or a reader
supplies "the market is efficient" and the study does not claim that.

### Direction accuracy, five symbols beside twenty

Anchor 0, `target_in_loop=True`, `C0_base` (plus buy-and-hold, which carries no channels
value — see the filter note below).

| arm | real 5 | real 20 | noise 5 | noise 20 | shuf 5 | shuf 20 | n−r 5 | n−r 20 |
|---|---|---|---|---|---|---|---|---|
| buy_and_hold | 0.5560 | 0.5528 | 0.5064 | 0.5011 | 0.5418 | 0.5365 | **−0.0495** | **−0.0517** |
| dlinear | 0.5006 | 0.4981 | 0.5006 | 0.5156 | 0.5021 | 0.5053 | +0.0000 | +0.0175 |
| fits | 0.4958 | 0.4948 | 0.5041 | 0.5015 | 0.5039 | 0.5026 | +0.0083 | +0.0067 |
| wits | 0.5031 | 0.4976 | 0.5050 | 0.5001 | 0.5074 | 0.5025 | +0.0018 | +0.0025 |

**Filter note, worth one line in the caption:** buy-and-hold carries no `channels` value,
so a bare `channels == C0_base` filter silently drops the one arm whose degradation
carries the whole argument. The correct filter is `channels == C0_base or model ==
buy_and_hold`. Counted on `results.csv`: **224 of 1110 rows are `buy_and_hold` and
not one carries a non-empty `channels` value**, so the bare filter keeps 608 rows and
none of the 224, where the correct filter keeps 832.

### The +0.0175 that is not a finding

DLinear scores better on white noise than on real data at twenty. The tempting stronger
claim — *not "identical on destroyed signal" but "better on it"* — **does not survive
correction** and must not be written.

| arm | mean(noise − real) | sd | p | p_holm | positive folds |
|---|---|---|---|---|---|
| dlinear | +0.0184 | 0.0301 | 0.0174 | **0.0521** | 13/16 |
| fits | +0.0087 | 0.0346 | 0.3484 | 0.6968 | 10/16 |
| wits | +0.0035 | 0.0386 | 0.8999 | 0.8999 | 7/16 |
| buy_and_hold | −0.0517 | 0.0593 | 0.0034 | **0.0134** | — |

DLinear clears the uncorrected threshold and fails Holm over four arms at 0.0521 — just
above 0.05. It is different in *degree*, not in kind: one of three same-signed gaps being
the largest is what "largest of three" means. **The claim stays "destroying the signal
costs these models nothing", and +0.0175 is reported as an unresolved gap.**

Provenance caveat, in the text: this test crosses the `control` column, whereas the
report's 117 pair by fold within a condition. Holm here is over these four arms only.

### Why the structure exists and is reachable

A constant always-long rule beats chance by 5.5 points over the same period. The structure
exists and is reachable — just not by explicit or implicit frequency decomposition at this
horizon. This paragraph is what stops the chapter reading as a failure report.

---

## 57.3 — The instrument works

**The most important section, and the shortest. Lead the presentation with it.**

`buy_and_hold` degrades on noise by **−0.0517, p = 0.0034, p_holm = 0.0134** — significant
after correction. Nothing else degrades.

At five symbols this was an unqualified −0.0495. **It now has a p-value attached**, which
is a stronger statement than §1 currently makes: the instrument detects a control effect
exactly where the mechanism predicts one, and detects nothing on the arms that never had a
forecast. That asymmetry is the argument.

Without this, "we found nothing" and "we built it wrong" are indistinguishable.

---

## 57.4 — MAE measures flatness

`Spearman(MAE, flatness) = +1.000` across arms.

Of 117 Holm-corrected tests, **29 survive at twenty** (17 at five) — see 57.8 for why that
number moved. 22 of the 29 are MAE, and every one says a model is worse than persistence.

> The metric carrying most statistically robust results in this study measures how close
> to zero a forecast is.

A finding about forecast-evaluation practice, not an apology.

---

## 57.5 — The geometry critique of FITS

### The gain-magnitude response

| figure | source | r² |
|---|---|---|
| +0.9485 | GB-46's published headline | 0.900 |
| +0.8967 | `operator_null_control.py`, 16 folds, five symbols, 27 Aug | 0.804 |
| **+0.9780 ± 0.0105** | same script, twenty symbols, 3 Sep | **0.957** |

**It moved, and toward the critique.** Real-versus-noise curves moved closer (mean absolute
difference 0.0390 → 0.0256). The data-independence the critique rests on is *larger* at
twenty.

`frac(η·k) = frac(k·H/L)` — one curve in two parameterisations.

**The 86% is not reproducible and must be replaced.** §2.4 derives it as
`1 − 0.0319/0.2335` — mean absolute difference over curve range — and `null_control.csv`
records the numerator but not the denominator. Same defect class as the 16.00%/peak-51%
reconstruction, in a claim the chapter leans on harder. **Restate as r² = 95.7% at twenty
against 80.4% at five.** The substance holds; 86% understated it.

### The signed operator — the narrowed claim weakens

The published narrowing is *"the envelope is geometry; the structure within it is not"*,
resting on the signed operator being indistinguishable from zero. **At twenty it is not.**

| | five | twenty |
|---|---|---|
| FITS | −0.0127 (t = −0.44) | **+0.1946 (t = +7.24)** |
| WITS | +0.0331 (t = +0.75) | **+0.0838 (t = +2.56)** |
| paired WITS − FITS | +0.0458 | −0.1108 |
| Wilcoxon p | 0.5282 | **0.0042** |
| WITS lower in | 9/16 | 13/16 |
| verdict | did not hold | **HELD** |

Both are the flattened `(H, L)` forecast matrix, n = 480, same function, folds, seed and
budget — differing only in what the close channel holds. **This comparison is legitimate
where the response comparison is not.**

**Restate as a matter of degree:** envelope overwhelmingly geometry (r² 0.957), signed
operator partly geometry (r² 0.038, firmly non-zero). More of FITS is grid-determined than
the narrowed claim allowed. The two halves must still never be quoted with the same number.

**WITS, stated honestly: it does not escape the pathology; it escapes it less.** A local
basis *reducing* grid-determination rather than removing it is what a real mechanism looks
like — and a claim that WITS escaped entirely would be refuted by its own t-statistic.

### GB-66 is complete, and its acceptance criterion is unmeasurable

The expansion plan set the criterion: *"the correlation between the learned response on
real data and on white noise is reported against FITS's +0.9485. This is the acceptance
criterion that matters; performance is not."*

**That quantity cannot exist for WITS**, and `operator_null_control.py` refuses it in code
with the reason recorded:

- FITS retains 23 rFFT bins (`input_len` 120 → 61 bins, 5-day cutoff retains 23).
- WITS's `retained_bands: 2`. Pearson over two points is ±1 by construction and carries
  zero information; `_correlate` already returns NaN for `size < 3`.
- Even all four DWT bands gives n = 4, and the values are different objects — a per-band
  aggregate over an octave versus a per-bin gain at one frequency.

The script emits an **absent row rather than a NaN row**, deliberately, because a NaN
invites a reader to wonder whether it failed.

**So GB-66 is COMPLETE with the finding that its own acceptance criterion is structurally
unmeasurable.** The critique is Fourier-specific by construction, and this was learned only
by building the substitute. Two documents and the supervising author all read the absent
row as "never measured" before reading the code.

---

## 57.6 — The execution finding

Alpaca refuses every multi-leg order class on a fractional quantity:
`{"code":42210000,"message":"fractional orders must be simple orders"}`. A working sell
holds the whole position, measured against 97.38 shares — so it is not fractional-specific.

> A position can carry exactly one broker-side protective order, and it is the stop.

The target moved into the loop. **The published backtest describes a system that cannot be
built at this venue.** Trade count falls 16% / 14% / 29%; the return difference is not
resolvable at sixteen folds.

> The buildable system is measurably different, not measurably worse.

`‹GAP›` The three refusals live only in a session transcript. Commit them as a recorded
fixture with the request that produced each, or state that the evidence is a transcript.

---

## 57.7 — The risk layer stops being inert

At five symbols the gross cap bound **once in sixteen folds**. GB-57 had to describe a
layer that was correct, enforced and doing nothing.

### At twenty it bound in 13 of 16

| case | count |
|---|---|
| REDUCED_BY_GROSS (B minimum, notional > 0) | 387 |
| BLOCKED_BY_GROSS (B minimum, notional = 0) | 147 |
| BLOCKED_BY_OTHER (A or C minimum) | **0** — cash never blocked an entry |
| TIE (A = B exactly) | **0** |
| NO_EQUITY | 0 |
| UNCONSTRAINED | 429 |
| total sizing calls | 963 |

The three non-binding folds (1, 3, 4) are the three thinnest — 0, 17 and 15 sizing calls
against 27–130 elsewhere. No anomaly needs explaining.

Peak gross reached 0.5212 against a 0.50 cap. **Correct, not a breach:** the cap bounds
what may be *added* at entry; `gross_exposure` is the marked value of what is already
open, and nothing sells to get back under. State this inline or a reader sees a violation.

### The measurement design was load-bearing

**The reducing case is 2.6× the blocking case.** An instrument watching for zero notionals
— the obvious design, and what this chapter's brief originally specified — would have
counted 147 of 534 cap events, 28%, and concluded the cap rarely binds. **Wrong, and wrong
in the direction that flatters the risk layer.**

Ties were recorded rather than resolved at record time, so the tie policy could be stated
afterward and its weight measured. It turned out to be 0, which is a measured answer
rather than an assumption.

### The layer binds without leaving evidence, in both modes

`room_for` returns `max(0, min(A, B, C))` and discards which term was the minimum. When
gross headroom merely *reduces* an entry, the notional comes back positive and the trade
looks ordinary — **the invisible case is the one where the cap is doing its work.** When it
blocks, the entry exits silently through `notional <= 0.0 → continue`.

This is why the question was unanswerable for a month. It is an instance of both principles
in 57.10 at once.

### The mechanism, and a tension neither of GB-61's arguments predicted

`top_k = 2 × max_position_pct 0.10` caps new entries at 0.20 per bar, so the cap binds only
through accumulation across overlapping holds. The concurrent-hold ceiling is
**5 × 0.10 = 0.50 exactly** at five symbols — touchable, never exceedable, so the cap was
very nearly decorative — against **2.00** at twenty, four times the cap. Drivers are
concurrent-hold count and hold duration.

**534 of 963 sizing calls — 55% — were cap events.** The cap is now the binding constraint
on more than half of all sizing decisions. That sits in tension with GB-61's other stated
purpose, that ranking becomes a real selection at `top_k = 2` out of 20: **if the cap
dictates size in more than half of calls, it may be selecting more than `rank.py` does.**
Both of GB-61's goals were met and they now pull against each other. A measured consequence
of universe size, not a defect.

`report/gross_exposure.csv` holds all 963 sizing calls, so every count above is
recomputable — the defect that sank the 16.00% figure does not repeat here.

---

## 57.8 — What universe size changed

### The Phase 1 conclusion survived; the Phase 1 statistics did not

Holm survivors **17 → 29**. The family widens from `{mae}` to `{direction, mae,
total_return}`. The smallest corrected p on direction goes **0.2605 → 0.0036**.
**All 29 survivors have the model losing.**

GB-57 §1's sentence *"Not one result on direction, sharpe or total_return survives"* is now
false.

### It is power, not effect — measured, not argued

Effect sizes are the same or **smaller** in 26 of 29 cells.

| family | mean \|Δ\| at 5 | at 20 | ratio |
|---|---|---|---|
| direction (3 survivors) | 0.0599 | 0.0580 | 0.97 |
| total_return (4) | 0.0827 | 0.0473 | **0.57** |
| MAE (22) | 0.000636 | 0.000491 | 0.77 |

`n` is 16 at both universe sizes. **The extra power did not come from more folds.** It came
from the collapse of the paired-difference standard deviation — the quantity the signed-rank
test actually consumes — because each fold's metric moved from an average over five symbols
to an average over twenty:

| arm, metric | sd of paired diff (5) | (20) | ratio |
|---|---|---|---|
| persistence, total_return | 0.11663 | 0.04633 | 0.397 |
| fits, total_return | 0.09623 | 0.03068 | 0.319 |
| wits, total_return | 0.10144 | 0.03246 | 0.320 |
| dlinear, direction | 0.06571 | 0.02576 | 0.392 |
| fits, direction | 0.06514 | 0.04209 | 0.646 |
| wits, direction | 0.05708 | 0.04217 | 0.739 |

**The strongest form:** on total_return the models are *less bad* at twenty — the shortfall
against buy-and-hold shrank to 57% of its five-symbol size — **and it became significant
anyway**, because the estimate is roughly three times more precise.

Ceiling unmoved: the exact test on 16 folds still cannot go below 3.05 × 10⁻⁵. Only the
noise floor dropped.

### The same pattern appeared twice, in independent instruments

The operator comparison (57.5) shows it again: at five symbols neither FITS's nor WITS's
signed-operator correlation was distinguishable from zero, and the WITS-vs-FITS verdict
"did not hold". At twenty, Wilcoxon p = 0.0042 and it holds. **Two independent negative
verdicts at five symbols were both power failures, not absence of effect. Two instances is
a claim about Phase 1's design, not a coincidence.**

**Phase 1's null on direction and total_return was partly a small-sample artefact.** The
deficit was always there; a five-symbol design could not resolve it. Where Phase 1 could
only fail to reject, this study rejects — **against the models.** It weakens only the
negatives about the negatives.

### Also here

Determinacy 2.09× → 8.35×. Whether `rank.py`'s cross-sectional selection starts carrying
weight — subject to the cap tension in 57.7.

---

## 57.9 — Limitations

Every one is stronger stated by the author than discovered by the reader.

1. **Survivorship bias, encoded in the selection rule itself.** Criterion 3 admits only
   securities continuously tradable through to the cache's last bar — knowable only in
   2026. Criterion 2 does the same for listing date. The `buy_and_hold` arm and the
   always-long bar are biased upward. Model arms are deltas against those references, so
   the bias sits inside the reference rather than being quietly removed. **An arm beating a
   survivor-inflated buy-and-hold is a stronger claim than the same arm beating an unbiased
   one.**

2. **The universe is not rule-derived.** The five criteria are an admissibility filter, not
   a selection procedure. They admit several hundred names and return twenty. Criterion 4
   states no threshold, ranking, source or date. AVGO, ORCL, KO and PEP satisfy all five
   and appear nowhere. The step from admissible to these twenty was judgment.

3. **Sector and factor concentration.** Seven of twenty are the mega-cap cluster and move
   on one factor. Four of eleven GICS sectors have zero representation. On the pre-2023
   scheme, V and MA are Information Technology, concentrating it further.

4. **The null controls exist at anchor 0 only, and this is structural rather than an
   omission.** Row counts across the control axis:

   | control | anchor 0 | 21 | 42 |
   |---|---|---|---|
   | noise | 208 | 0 | 0 |
   | real | 544 | 112 | 112 |
   | shuffled | 112 | 0 | 0 |

   `study.conditions` builds a **star, not a cross**, and puts control spokes at the centre
   only. So *"flat at all three anchors under both null controls"* is not merely unmeasured
   — it is unavailable in this design. The project's two independent robustness checks — the
   ones that killed `r = −0.47` and the phase advance respectively — **cannot currently be
   crossed.** Name what closing it would cost.

5. **Reproducibility is bit-identical on the same hardware.** Cross-architecture is untested
   and named as such. `‹GAP›` The audit has no recorded figure or procedure.

6. `‹GAP›` **Fold calendar dates are inferred from the snapshot date**; `results.csv` holds
   indices.

7. **Sentiment was excluded, deliberately.** An LLM's training data includes the test
   period, so it is look-ahead the causality harness structurally cannot catch, and it
   breaks bit-identical reproducibility. One page, in this chapter, not in future work.

8. **The third feature arm, declined 4 September 2026.** `C3_extended` — a third feature
   configuration adding ATR, Bollinger position, a volume-flow measure and a
   longer-horizon momentum — was specified as GB-62 and is **not in this study. The clause
   permitting it was live and its condition was met.** `GLASSBOX_PHASE2_EXPANSION.md`
   deferred C3 rather than rejecting it: *"If GB-66 lands before 8 September, C3 returns as
   a spoke."* GB-66 landed on 3 September. The arm was therefore available and was
   declined, on 4 September, for the reasons below.

   **This is a cost decision under a deadline, not a judgement that C3 is worthless.** The
   measured incremental compute is small — 128 rows at ~7.0 s/row, about 15 minutes,
   derived from `results.csv`'s own `seconds` column, where DLinear runs 4.80 s/row at five
   channels and 6.47 at eight. The costs that decide it are elsewhere. Every new indicator
   must be pure, trailing-only, pass the GB-10 causality harness in both perturbation modes
   at three splits, and carry a hand-computed fixture test with literal expected values.
   **OBV has no causal trailing-window definition** — it is a cumulative sum from the series
   start — so it is a design question before it is an indicator. And `min_history_bars` is a
   **max over per-channel warm-ups**: ATR at period 14 shares `rsi14`'s Wilder recursion and
   lands on the same 325-bar warm-up, but **ATR at period 20 needs ~449**, which would raise
   the parity floor from 445 to 569, change the window geometry, change the config hash, and
   cause every existing checkpoint to be refused — **re-running the entire grid, not the C3
   rows**. The governing boundary is not the 3 October code freeze but the expansion's own
   **15 September**, after which *"Phase 2 closes in whatever state it is in"* — eleven days,
   against five unwritten chapters.

   **The stronger reason is that GB-66 already answered the question C3 was a proxy for, and
   answered it harder.** C3 asks *"is the null result just bad features?"* — a question about
   our choices, answerable only by making a third set of our own choices. GB-66 replaced the
   model's transform rather than its inputs, and its acceptance criterion turned out to be
   **structurally unmeasurable**: WITS has no per-bin frequency response to correlate against
   FITS's, because a basis whose per-band maps are square has no counterpart to a global
   basis remapped onto a shifted grid. That is a stronger statement than a third feature arm
   could produce, and it is a statement about the method rather than about a feature list.
   The null already survives four model arms, two feature arms, three fold-grid anchors, four
   cutoffs, two null controls, and a universe change from five symbols to twenty.

   **What a reader is owed, and it is the honest form of the limitation:** the expansion
   itself predicted the outcome — *"the likely outcome is that they do not, and that is the
   point"* — and a prediction is not a measurement. **This study tested two feature
   configurations, both chosen by its author, and cannot exclude that a third would have
   behaved differently.** The claim made here is bounded accordingly.

---

## 57.10 — The methodological contribution

**The section a software-faculty reader will grade highest.** Audited 3 September against
the repository: 31 listed instances, 0 NOT FOUND, 2 corrected, 12 further instances found
that were not on the list.

**Structure, and this matters more than the content.** Forty anchored bullets is a list, not
an argument. Write **three worked exemplars** at ~250 words each — principle, what happened,
what it cost, what closed it — and put every remaining instance in a single anchored table
in an appendix. A reader who reads three properly understands the claim; a reader who skims
forty understands nothing and suspects padding.

**Anchoring rule:** every instance carries a commit SHA, `file:line`, or test node ID. No
instance appears without one.

---

### The three principles

**P1 — Only something that runs is a mechanism.** A fact stored twice needs something making
the copies equal. A test skippable by a flag, cache or marker is not a mechanism. A
description of what the code should do — comment, docstring, plan or default — is not a
mechanism.

**P2 — A green result answering a different question is the most comfortable kind of wrong.**
Nothing about it looks like a failure.

**P3 — A mechanism installed for one reason covering a failure nobody connected to it.**

---

### Exemplar 1 (P2) — the feed downgrade that every schema test would have passed

A silent Alpaca feed downgrade from SIP to IEX produces the same columns, the same dtypes,
the same index and the same provenance string. **Every schema test in the suite passes.**
Prices are out by up to **193 basis points against a 1 bp tolerance — roughly 190×** — and
until 18 August there was no backstop at all.

`DECISIONS.md:2615-2620` · `tests/data/test_live_schema.py:238-244`

The largest measured cost of any P2 instance in the project, and the clearest statement of
the principle: the tests were correct, comprehensive, and answering a question about
*shape* when the failure was about *value*.

### Exemplar 2 (P3) — the dead DC row, found by a test written to prove something else

A test written to demonstrate weight sharing perturbed row 0 and **nothing moved for
anybody**. That is not what a sharing test looks for. Following it found **50 of FITS's
1,200 parameters allocated and unable to learn**.

`DECISIONS.md:1656-1658`, `:1660ff` · `tests/model/test_fits.py::test_the_dc_row_is_allocated_and_cannot_learn`

Textbook P3, and it lands directly on the architecture this chapter is about — 4% of the
model's parameter budget, inert, discovered by accident.

### Exemplar 3 (P1 → P3) — one guard, both sides of the ledger

`test_tree_has_no_extra_modules` is a **P1** mechanism: it pins spec §3.4's module list
against the package tree so a module cannot exist without a spec entry. It is also what went
red when `dashboard/tokens.py` was added without one, and **stayed red for four commits**
that each reported a passing dashboard subset (`3e00950`: *"THE SUITE HAS BEEN RED SINCE
REGION 1"*, red from `1c9cf49`).

So a P1 guard caught a P3 failure — a mechanism built for spec consistency catching a
reporting failure nobody connected to it — and it did so in one anchor.

**Note for accuracy:** this is a *separate episode* from the eight-region incident recorded
in `CLAUDE.md:74`, where the suite was green. Do not merge them.

---

### The full instance table

Appendix material. Every row anchored; principle in the first column.

| P | instance | anchor |
|---|---|---|
| 1 | SPEC_MODULES and the tree | `test_scaffold.py:14`; `::test_spec_module_exists`, `::test_tree_has_no_extra_modules` |
| 1 | VALID_MODELS / ALL_FORECASTERS / `smoke_offline`'s literal — **and a fourth copy**: `run()` defaulted to the literal `"dlinear"` rather than `model.active`, so `model.active: fits` changed the live checkpoint and nothing about the study | `loader.py:42`, `model/__init__.py:66`, `smoke_offline.py:920`; `DECISIONS.md:1628-1634`, `:1236` |
| 1 | Exit reasons in two modules | `engine.py:84-97` vs `records.py:72-82`; `test_records.py:453-457` |
| 1 | pyproject and the lockfile | `test_scaffold.py:155-191` |
| 1 | The deselected test | `DECISIONS.md:925-928` |
| 1 | The `--sessions` default | `live_loop.py:2272`, `:2641`; `README.md:183,195,209` |
| 1 | The pill guard counting its own comment — the third instance arrived **one turn after** the line was written, in a comment explaining the thing the guard guards | `7c98dbb` body. *Anchor is the record and the fix; no broken version was ever committed* |
| 1 | The fixed-width stylesheet slice | `test_app.py:1189-1194`. Same caveat as above |
| 1 | Two further costumes of the same family: a test carrying `250 - 30`, a copy of the chart's own height; and the type-scale guard now stripping comments via `tokenize`, because scanning raw source means the sentence explaining a removal reintroduces it | `PROGRESS.md:316` |
| 1 | Three copies of the universe; spec §75 said "20" while §599 said five; closed by a spec-parsing test, 8/8 mutations caught | `d68071f`; `test_config.py::test_every_universe_written_in_the_spec_matches_the_configuration` |
| 1 | The honesty rule named an invocation that could not run — bare `pytest` exited 2 with zero collected, while every "N passed" came from `python -m pytest` | `DECISIONS.md:5258-5300`; `fbe6d5d`; `ci.yml` |
| 1 | `test_train_live_parity` derived its count; `test_sweep` pinned `== 25`. Same property, only the pinned one broke | `test_sweep.py:189-194` |
| 1 | `room_for` discards which term bound — still true; `room_detail` observes beside the sizing path rather than fixing it | `risk.py:334`, stated at `:249` |
| 1 | 16.00% / peak-51% gross exposure: producing code never committed | `DECISIONS.md:3497`; `PROGRESS.md:320`; `spec:312` |
| 1 | The 86%: denominator (curve range) never recorded | `DECISIONS.md:1159-1160`; `null_control.csv` header |
| 1 | A **half-true** claim, harder to catch than a false one: a handoff said WITS's null control had run. The *operator* rows had (16 folds, +0.0331); the *response* row never existed and cannot | `PROGRESS.md:317`, `:319` (wrong row kept, not deleted) |
| 1 | `test_the_same_grid_twice_gives_the_same_numbers`: docstring said one condition and one fold; the body ran the whole grid, 26 condition-grids per suite run | `CLAUDE.md §3` |
| 1 | A comment reading "GB-41 adds it, not GB-24" when GB-41 did not | `DECISIONS.md:1638-1641` |
| 1 | A hand-written contrast table wrong in one of seven: `#2E3742` given as 1.27 when it is 1.57 — replaced by a computation | `test_tokens.py:41-47` |
| 2 | Property 4 comparing a number to itself | `spec:1233(a)`; `DECISIONS.md:1144`, `:2714` |
| 2 | `exit 0` through `| tail -4` | `70e3c2a`; `CLAUDE.md:76` |
| 2 | CI green on 1,026 tests with 26 skipped | `spec:1233` |
| 2 | `import glassbox` succeeding without the editable install | `DECISIONS.md:642-650`; `PROGRESS.md:301` |
| 2 | The dry run clearing the healthy case, not the case the bound was set by | `DECISIONS.md §2026-08-24:36-42,113-114`; `284edcb` |
| 2 | **FakeBroker permitting what Alpaca forbids** — 1,181 passing tests over a live path that could not execute | `PROGRESS.md:311`; `spec:1197`; `fake_broker.py:161`; `test_executor.py:77` |
| 2 | `explain_spectral` with no caller, so GB-53's panel never rendered | `spectral.py:185`; `live_loop.py:1114-1121` |
| 2 | The contrast test passed over 1.27:1 text, because the element was classified as the rule it was named for | `RULE = #202832`; `test_tokens.py::test_every_text_colour_clears_aa_against_the_ground`; `PROGRESS.md:316` |
| 2 | **A second, independent blind spot in the same test: AA knows nothing about size.** DIM cleared AA at 5.19:1 and was set at 9px — *"the palette test above passed while the thing it was protecting was unreadable in print"* | `test_tokens.py:105-115` |
| 2 | **The calendar painted a flat day green.** `GAIN if value >= 0`, in from `2639649`, out at `3e00950`. Of 175 daily changes, 152 flat / 10 up / 13 down — so the project's central finding, that the system declines to trade, rendered as a mostly-winning year | `test_app.py::test_a_flat_result_is_never_drawn_as_a_gain`; `57a7f41` body: *"a green tile turns 'nothing happened' into 'something good happened'"* |
| 2 | The session summary misreporting itself: `positions flattened : 0` on a rehearsal that cancelled a stop and sold the position; `open orders at exit : 0` — the cleanest-looking line, and exactly what a failed close-out prints. Five instances in the honesty layer | `PROGRESS.md:315`; `b421dc5` |
| 2 | `.gb-not-responding` is defined and never emitted: `staleness_html` sends both branches to `.gb-stale`, so "LOOP NOT RESPONDING" renders in ordinary DIM, against its own docstring. **An open instance — not fixed, still true** | `app.py:2343`, `:947`; `PROGRESS.md:316` |
| 2 | The report's `sd` column is the arm's own per-fold spread, not the paired difference, and only the latter feeds the signed-rank test — **and the two move in opposite directions.** Over the 39 `total_return` cells present at both universe sizes the reported `sd` **rises 0.01605 → 0.02286 (×1.424), up in 25 of 39**, while the paired-difference sd the test actually consumes **falls 0.13567 → 0.05154 (×0.380), down in 39 of 39 — unanimous.** A reader auditing §57.8's power argument against the column printed beside the p-value would have read the mechanism backwards | `experiments/stats.py:232`; recomputed at `7997ed7`, `SUBMISSION_DOC_CORRECTIONS.md` §9 |
| 2 | An instrument watching zero notionals would count 147 of 534 cap events — 28%, wrong in the direction that flatters the risk layer | `exposure.py:18-23`; `risk.py:232-236`; `gross_exposure.csv` |
| 2 | A bare `channels == C0_base` filter drops every buy-and-hold row, the one arm whose degradation carries the argument. **224 of `results.csv`'s 1110 rows are `buy_and_hold` and 0 carry a non-empty `channels` value** — a reference arm has no feature configuration to name, so the equality *cannot* match rather than merely failing to: the bare filter keeps 608 rows and none of the 224, where the correct one keeps 832 | `results.csv`, counted at `7997ed7`, `SUBMISSION_DOC_CORRECTIONS.md` §10; correct filter at `SUBMISSION_DOC_CORRECTIONS.md:165-166` |
| 3 | Tracking `data_cache` for CI made GB-59 possible — the clean clone ran 0 skipped | `7c98dbb`; `.gitignore:19-23`; `PROGRESS.md:301` |
| 3 | The once-per-bar rule stopped a stray rehearsal opening a second position — **and keep its qualifier**: it refused *by accident*, and a new bar completes every session, so that protection expires. Which is why the lock was built | `7c98dbb`; `DECISIONS.md:5104-5113` |
| 3 | **The same rule, the other side of the ledger:** the first `close_out_on_stop` test passed with the wiring removed. It asserted an empty broker on a session that was never going to open a position — vacuous *because of* the once-per-bar rule | `DECISIONS.md:227-233` |
| 3 | Per-region commits, asked for bisectability, paid off when a scripted edit deleted sixteen tests from three committed regions | `7c98dbb`; rationale stated in advance at `0f09587` |
| 3 | `test_every_script_guards_its_imports` caught a missing matplotlib hours before it bit — *"A test written for one reason caught a different failure."* **With its P1 sting:** the rule was added to CLAUDE.md at 12:00 and the fourth instance shipped at 12:06 | `DECISIONS.md:1244-1247` |
| 3 | `predict.py:80` refused to score ABBV against a five-symbol checkpoint, after the drift gate above it passed the checkpoint as valid — true and irrelevant | `predict.py:80`; `PROGRESS.md:318` |
| 3 | `operator_null_control.py` emits an absent row rather than a NaN row, so the gap is legible as a refusal. Three readers still took it for "never measured" | `operator_null_control.py:164`, `:110-122`; `PROGRESS.md:323` |

### Two corrections the audit forced, and they belong in the chapter

Both are P2 instances committed *by the chapter's own drafting*, and **both are now
resolved** — recorded as corrections 9 and 10 in `SUBMISSION_DOC_CORRECTIONS.md` at
`7997ed7`, rather than quietly fixed:

- The `0.0129 → 0.0162` figures were asserted as a total_return sd comparison and are MAE
  means — though `0.0129` *is* a real sd (DLinear, `C0_base`, five symbols), so the pair
  was one correct number and one from a different metric, which is the harder kind to
  notice. **Recomputed**, and the claim came back stronger: the two quantities move in
  opposite directions, and the one the test consumes falls in 39 of 39 cells.
- *"My first pass dropped it"* about the buy-and-hold filter was a session recollection
  with no committed artefact behind it. **Cut**, and replaced by the counts — 224 rows,
  0 with a `channels` value — which are the whole instance and need no anecdote.

Say so in the text. A chapter about green results answering different questions that
declines to report its own is worth less than one that does.

---
## 57.11 — Future work

Each item is a question the study leaves open, not a feature list.

- The C3 feature arm (GB-62, deferred): does a null with three feature configurations
  differ from one with two?
- The geometry critique on a third architecture — FITS-specific or method-specific?
- **Crossing the grid-anchor axis with the null-control axis**, which the star design
  currently forbids (57.9 item 4).
- Horizons other than daily. The claim is scoped to this horizon and says so.
- A universe with delisted names, to measure what criterion 3 costs.
- Recording which term bound in `room_for`, so the risk layer leaves evidence.
- **Whether the gross cap now selects more than `rank.py` does** (57.7).

---

## Writing order

Not the chapter order:

1. **57.10** — needs no new numbers, highest value. Write it first. **Gate it on the
   evidence audit** — every instance carries a SHA or node ID or it does not appear.
2. **57.9** — same reason.
3. **57.5, 57.7, 57.8** — the material measured on 2–3 September, freshest and most
   structured. Write while the reasoning is still available.
4. **57.2, 57.3, 57.4** — organising, not writing from scratch.
5. **57.6** — needs the transcript gap closed or named.
6. **57.1, 57.11** — last. The introduction is easiest once the chapter exists.

**15 September is a boundary.** Nothing in this chapter now depends on a run that has not
happened.
