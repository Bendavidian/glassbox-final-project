# DECISIONS

Architectural decision records. One entry per decision that changes a contract,
adds a dependency, or cuts scope. Newest first.

Format: date · decision · reasoning · consequence.

---

## 2026-08-24 — Phase 2 opens: one location for the dates, one contract amendment, and a third copy of the registry

Five rulings, all of them the same shape: **a second copy of a fact, replaced by either a
derivation or a test.** `GLASSBOX_PHASE2_EXPANSION.md` arrived as a second authority
alongside the spec, and every one of these is a seam it opened.

**1. The dates leave §8 and live only in the expansion's schedule table.** §8 held a sprint
calendar; the expansion opened with a schedule of its own, and the two had **already
disagreed** — §8 said GATE 2 fell on 25 September while the expansion ran its sessions on
24–30 August. The expansion carries a precedence sentence covering exactly this case, and a
precedence sentence is a **note**: it resolves the conflict for a reader holding both
documents and does nothing for a reader who opens one. §8 keeps what a date cannot replace
— the gate criteria, the checklists and the cancellation rule each gate carries — and the
dates are gone from it.

**2. `CLAUDE.md` §0 names both documents, and "single source of truth" is withdrawn for
scope.** It was the first line every session read and it had become false in one specific
way. The spec is now stated as the authority for contracts, architecture, methodology and
the gate criteria; the expansion for scope after GATE 2. Where they conflict, the spec wins
on contracts and the expansion wins on scope.

**3. GB-64's enforcement claim was false as written, and is now an acceptance criterion.**
The expansion said sentiment fetching is *"enforced by the import contract, not by
discipline."* It is not. The layers contract places `glassbox.data` **below** `backtest` and
`experiments`, so a provider under `data/` is freely importable by the two modules GB-64
forbids, and the existing `forbidden` contract only points the other way — it keeps
`live_loop`, `replay` and `records` out of the harness. GB-64 must **add** a `forbidden`
contract naming the sentiment module. Until it exists the claim is discipline in a
contract's clothes.

**4. `Attribution.per_level` is added to the §4 contract for GB-66.** The cheap route was to
reuse `per_frequency` keyed by band; refused, because a field named for a frequency holding
a level is the two-places defect committed **inside a single schema**, where no test can see
it — each copy is internally consistent and only a reader who knows which model wrote the
record can tell what a key means. A wavelet decomposition has two axes and both already have
homes: `per_level` for the band totals, and `per_lag` for the time localisation, retained
unused since GB-31 with the words *"so the feature can be added later without a contract
change."* GB-66 is that later.

**Consequence, and it is why the decoder is asymmetric.** The field was added **while a
multi-day GATE 2 run was writing the decision log**. Records without the key exist and are
correct, so `records._decode` reads it with `.get` where it indexes every other key: the
dashboard, replay and the gate log all decode that file, and indexing would turn every line
written before 15:44 today into a `KeyError`. A field added after a log has started must
read as **absent**, not as broken.

**5. There was a third copy of the model registry, and the test that was supposed to cover
this could not see it.** `test_the_registry_and_the_config_layer_name_the_same_models`
asserts `set(VALID_MODELS) == set(ALL_FORECASTERS)` — two lists **enumerated by name**, so it
detects those two diverging and is blind to a third appearing. That is the same defect one
level up. Written as a scan, it found one immediately:
`experiments/study.py` held `MODELS = ("persistence", "dlinear", "fits")`, which **agreed**
with the registry and would have gone on agreeing until a model was registered — at which
point the study would have run three arms and silently omitted the fourth. **That is the
GB-44 defect, in the same file family**: the runner that produces every study number, unable
to select the model the study is about. GB-66 registers `wits` in six days.

`MODELS` is now `tuple(ALL_FORECASTERS)`, so the copy is gone rather than pinned.
`PER_FOLD_SECONDS` **cannot** be derived — its values are measurements — so it is pinned by a
test instead; a model registered without a timing would not fail, it would silently
under-report the wall time of every plan containing it, which is the estimate the
ten-minute rule depends on. The scan catches a complete copy **and a stale one**, and was
proven to fail against both before it was committed.

---

## 2026-08-23 — GB-50: the COF sweep rides on the star, and each cutoff carries its own control

**Decision 1 — three more spokes, not a fourth crossed axis.**

| | conditions | cells | arm-folds | estimate |
|---|---|---|---|---|
| star as of GB-49 | 7 | 35 | 560 | 14.8 min |
| **+ COF spokes (this task)** | **13** | **41** | **656** | **21.6 min** |
| full cross, all four axes | 108 | 216 | 3,456 | 148 min |

**It rides, and the reason is not economy.** The cutoff is FITS's one hyperparameter and
reaches persistence and DLinear through **nothing at all**: their rows at cutoff 20 would
be identical to their rows at cutoff 5 in every column that is a result. Running them
would not be a waste so much as a **hazard** — a duplicate in a results file is something
a reader eventually averages. `Condition.models` therefore returns FITS alone on a COF
spoke, and the plan counts cells per condition rather than multiplying.

**Decision 2 — every cutoff gets its own null control, not only the centre.** Six new
conditions rather than three. A cutoff tested only on real data is a cutoff nobody can
falsify, and the standing requirement of 20 Aug applies to each of them separately: if
FITS at cutoff 2 scores the same on white noise as on real data, **that is the finding for
that cutoff**.

**Decision 3 — the wall-time estimate is now per-arm and measured.** The old flat 3.9 s
per arm-fold predicted 36 minutes for a grid that took 15.1. The measured per-fold costs
span 60x — buy-and-hold 0.07 s, persistence 0.39, DLinear 1.46, FITS 4.16 — so a flat mean
misprices any design whose **mix** differs from the one it was averaged over, which every
FITS-only spoke does. The same estimator run against the GB-49 design gives 14.8 minutes
against the 15.1 measured.

**Consequence — two configuration-specific results become curves, and both invert the
natural reading.**

| cutoff | COF | out bins | reals | dead row | aligned bins | mean misalignment |
|---|---|---|---|---|---|---|
| 2 | 60 | 62 | **7,440** | 1.67% | 2/60 | 0.2500 |
| **5 (deployed)** | 24 | 25 | 1,200 | 4.17% | 1/24 | **0.2833** |
| 10 | 12 | 13 | 312 | 8.33% | 1/12 | 0.1833 |
| 20 | 6 | 7 | 84 | **16.67%** | 1/6 | 0.0833 |

1. **The dead DC row's share rises with the cutoff.** It is `1 / COF` exactly, so the
   *smaller* model wastes proportionally *more*: 1.67% at cutoff 2 against **16.67%** at
   cutoff 20. The intuition that a tighter filter is a leaner model is wrong in the only
   sense that matters here.
2. **The interpolation cost saturates rather than peaking, and the deployed cutoff is on
   the plateau.** See the correction below: mean `|η·k − round(η·k)|` rises with `COF` and
   settles at **0.25**, and cutoffs 2 and 5 are both in that regime.
3. **§6.1's parameter framing is cutoff-specific.** At cutoff 2 FITS holds **7,440** reals
   against DLinear's 4,800, so *"FITS is the smaller model"* — true at the deployed cutoff
   by 4x — **inverts inside this project's own sweep**. Any sentence comparing the two
   sizes must name the cutoff it holds at.

**Correcting the first reading of this table, which was mine.** I wrote that the cost is
"worst in the middle" and that the deployed cutoff landed on the peak of an axis nobody was
looking at. **The pattern is saturation, not a middle**, and the corrected version is weaker
and truer. `η = 1 + H/L = 1 + 1/30`, so `frac(η·k) = frac(k/30)` and the fractional part has
a **period of 30 bins**. Mean misalignment therefore rises with `COF` while the retained
bins cover less than one full cycle, and **saturates at 0.25** — the mean distance of a
uniform fractional part to the nearest integer — once they cover one or more:

| cutoff | COF | cycles of 30 | mean misalignment |
|---|---|---|---|
| 60 | 2 | 0.07 | 0.0167 |
| 20 | 6 | 0.20 | 0.0833 |
| 10 | 12 | 0.40 | 0.1833 |
| **5 (deployed)** | 24 | 0.80 | **0.2833** |
| 4 | 30 | **1.00** | **0.2500** |
| 2 | 60 | **2.00** | **0.2500** |

At exact multiples of the period the value is **exactly 0.2500**. The apparent peak at
`COF` 24 is a **partial-cycle sampling artefact** — 24 of 30 bins covers `k/30 ∈ [0, 0.77]`,
which over-weights the far half — and the gap between 0.2833 and 0.2500 is sampling rather
than structure. So: **every low cutoff is saturated and the deployed one is among them.**
Landing on a plateau is a weaker claim than landing on a peak, and it is the one the numbers
support; the peak version invites a reader to check the arithmetic and find it marginal.

**The `H` table of 20 Aug is the same curve in a different parameter**, and saying so
keeps two entries from reading as two mechanisms. The alignment period in `k` is `L/H`, so
across `H` at fixed `COF = 24` the retained bins span 0.2 cycles at `H = 1` (0.0958, still
rising), 0.8 at `H = 4` (0.2833), 2.4 at `H = 12` (0.2333) and 12 at `H = 60` (0.2500) —
**rise, then the same 0.25 plateau** — and the 0.0000 at `H = 120` is not the far end of a
curve but the one exact escape from it, `η = 2`. What survives unchanged from that entry is
its actual claim: the cost is **not monotone in `H/L`**, `H = 1` is genuinely better than
`H = 4`, and `H = 120` is free. What does not survive is reading either table as a peak.

**And the curve carries a trade-off the source paper does not discuss.** Retaining more
frequencies buys **more information** and **worse reconstruction of each one**, because more
retained bins means higher `k` and higher `k` means `η·k` further from an integer. `COF` is
therefore not only *how much to keep*; it is also *how well what you kept can be
reconstructed*, and the two pull opposite ways. **The trade-off is not symmetric, which is
the part that makes it testable:** the reconstruction cost **saturates at 0.25** while the
information retained keeps rising, so the geometry alone does **not** predict a middle
optimum.

**And there is a third force, which is the one most likely to produce an actual optimum,
because it does not saturate either.** The complex layer grows as `COF · ceil(η·COF)`,
so statistical determinacy falls fast. Measured against **2,505 training windows × `H = 4`
= 10,020 equations** per fold — fold 1, five symbols, 501 train timestamps each, counted
the way `train.py` builds them rather than by slicing the frame first:

| cutoff | COF | reals | equations per parameter |
|---|---|---|---|
| 2 | 60 | **7,440** | **1.35×** |
| **5 (deployed)** | 24 | 1,200 | 8.35× |
| 10 | 12 | 312 | 32.12× |
| 20 | 6 | 84 | 119.29× |

So the three forces are: **information retained**, which rises with `COF`; **reconstruction
quality**, which falls and **saturates at 0.25**; and **statistical determinacy**, which
falls and **does not saturate**. At cutoff 2 the layer is very nearly underdetermined.

**The prediction, recorded before the sweep ran.** If there is an optimum, **it is not at
cutoff 2**, and the reason will be **statistical rather than geometric**. **If cutoff 2 does
win, that is evidence against the determinacy story and is worth as much as the opposite.**
All of this is written here before the results, so that whatever the sweep returns cannot be
read as a post-hoc explanation of it.

**And the sweep found a defect in the geometry it was measuring.** `out_bins` was
`ceil(η · COF)` in floating point, which is correct on paper and wrong **exactly where
`η · COF` is a whole number**: at `COF = 60` the product evaluates to `62.00000000000001`
and the ceiling came out **63**, one bin more than the architecture describes and 120
allocated reals that nothing accounts for. It is silent — no stage downstream refuses an
extra output bin — and it fires only on the **perfectly aligned** case, which is the one
the formula exists to handle cleanly. Now integer arithmetic. **The deployed cutoff of 5
gives 24.8 and is unaffected**, so no checkpoint and no published number moves; the defect
was reachable only by sweeping the axis, which is an argument for sweeping axes.

All three are **derived from `model/fits.py`** and asserted equal to the forecaster's own
properties, rather than recomputed in the study: `COF = L // cutoff` is exactly the
arithmetic that gets written down twice, and this project has five instances of that
family already.

---

## 2026-08-23 — GB-51: three references, and the family is counted

**Decision 1 — spec §7.4's "each arm vs persistence" is wrong and is corrected in place.**
Persistence forecasts zero and takes no trades, so its `direction` and `sharpe` cells in
`results.csv` are **empty**. A single reference would have tested two of the four metrics
against a NaN and reported whatever survived. The three-reference rule of §7.3 applies
here identically: **MAE against persistence** on the same channel set, **direction against
always-long** at that fold's realised up rate, **Sharpe and total return against
buy-and-hold** at the same fold.

**Decision 2 — pairing is by fold, within a condition, and may not cross one.** Fold 3 at
anchor 21 is not fold 3 at anchor 0. Pairing them would compare two periods rather than
two arms, and it would do so invisibly, because both are called fold 3.

**Decision 3 — multiple comparisons are corrected, and the correction is Holm.** The
instruction was that either applying one or declining one is defensible, and that not
mentioning it is not. Both are reported: raw `p` **and** `p_holm` side by side, with
`n_tests` on every row. Holm-Bonferroni rather than Bonferroni because it is uniformly
more powerful at no cost, and rather than Benjamini-Hochberg because FWER is the guarantee
a reader of a results table assumes and BH's needs a dependence assumption nobody here has
checked. **The family is every test computed in the same call**, which means a report
narrows the family by narrowing the table — and that is a property to state rather than to
leave implicit, because a p-value corrected over 98 tests and one corrected over 30 are
different numbers from the same data.

**And the floor, which matters as much as the values.** The exact two-sided signed-rank
test on 16 folds **cannot return a p below `2 / 2^16 = 3.05e-5`**. No claim in this study
can be significant past that however large its effect. Holm over a family of ~50 still
leaves a smallest achievable adjusted value near 1.5e-3, so the correction is survivable —
a claim that fails it was not close.

**Consequence.** `experiments/stats.py` is a **pure function from `results.csv` to a test
table**: it runs nothing, trains nothing and needs no cache, so GB-52 regenerates it from
the file rather than reading a second artefact that could drift out of step. It carries
`control` and `skipped`, so `study.reportable` gates it **unchanged** — one gate for both
tables. A test is **withheld rather than reported powerless**: below six moved folds the
exact test cannot reach 0.05 at all, and a p from it would be a number with no power
behind it rather than an absence of effect.

---

## 2026-08-24 — The live pass: what refreshes, what is marked stale, and what a closing stop says

**Decision 1 — two caches, two TTLs, because they are two problems.** Local reads
(decision store, book, thresholds, heartbeat) at **5 s**: a few KB of filesystem, free,
and enough to give the next-cycle countdown a credible tick. The **broker** read at
**30 s**: Alpaca's paper tier allows ~200 requests/minute, a positions-plus-orders refresh
is 2 calls, so 5 s would be **1,440 calls/hour** and 30 s is **240** — about 2% of the
ceiling. **And the dashboard is not the only client:** the live loop polls the same
account every 60 s from another process, so the ceiling is shared and the interaction is
real rather than theoretical. A single shared TTL would either hammer the API or freeze
the countdown.

**Nothing is lost by 30 s.** The underlying instrument advances **once per trading day**;
a position figure 30 s old is current by every standard the data supports.

**Decision 2 — a stale broker read is shown as stale, never hidden and never silently
reused.** On a failed call the panel keeps the last value, **dims it, and stamps it**
`BROKER UNREACHABLE — LAST READ 47s AGO`. Not blanked: a position you cannot see is worse
than one you can see is old. Not unmarked: **an unmarked stale number beside a stop
distance is a reading someone could act on**, which is the same defect as the exit-0 pipe
of 23 Aug — a value that looks current and is not. Above **two heartbeat intervals** it
stops being a staleness note and becomes the **loop-not-responding** state in the header,
because at that point the question is no longer "is this price fresh" but "is anything
running".

**Decision 3 — distance-to-stop changes what the row SAYS as it closes, not how it
moves** (ruled 24 Aug, for the live pass rather than for GB-53):

| distance | treatment |
|---|---|
| beyond **half** the stop distance | ordinary weight, ordinary row |
| inside **half** | the figure takes emphasis and the row is marked |
| inside **a quarter** | the row states plainly that a stop fill is near, **and what will happen when it fills**: the leg fills, the sibling is cancelled in the same cycle, and a Trade is emitted with reason `stop` or `stop_gap` |

**No animation and no colour change** — weight, a glyph, and a sentence, so the panel still
reads in greyscale. **The reasoning is the reliability line's:** the panel should tell a
reader what is **about to happen**, not only what is true. A stop firing is the one event
of the GATE 2 sessions where a human watching would want thirty seconds of warning, and it
is the one the data can actually give.

**Decision 4 — the frame is an instrument panel, not a trading terminal, and that decides
the price chart.** This system trades **completed daily bars** (GB-26), and GB-7 measured
that this subscription **refuses `get_stock_latest_quote` on SIP** — there is no intraday
price to show even if we wanted one. So the chart carries
`LAST COMPLETED BAR <date> · ADVANCES ONCE PER TRADING DAY` **in the same register as the
reliability line**, not as small print. **Movement is allowed only where the underlying
number moved:** a countdown ticks because time passes, a P&L figure changes because the
price changed. Nothing pulses, sweeps or animates to look alive. Styling motion onto a
daily bar would be the **interface** committing the misreading the spectral panel's caption
exists to prevent.

---

## 2026-08-23 — The spectral panel's capture is deferred, and its caption is mandatory

**Decision 1 — the four report screenshots are captured by hand, and `figures/screenshots/`
is tracked.** Driving a Streamlit dashboard headlessly needs a browser stack, and adding
one the night before the first GATE 2 session is a risk with no upside; producing mockups
instead is the dishonesty this project exists to oppose. The earlier captures live in
`checkpoints/screenshots/`, which is **gitignored** — they are not in the repository and a
clone has never had them. The new directory is tracked, with a README stating the exact
dashboard state each capture must be taken under, so the specification does not live in a
conversation.

**Decision 2 — the spectral panel's capture waits until after the sessions of 24-26 Aug.**
The panel renders only when `model.active` is `fits`; the deployed arm is `dlinear`.
Switching it the evening before the critical session is a **forgetting risk on the wrong
evening**, and it would move `config_hash` and trip the drift guard built the same day.
With the loop stopped after the sessions there is no running process to drift against and
the switch is free. **The first three screenshots are captured under the deployed
configuration, which is what they should show.**

**Decision 3 — the caption on that screenshot is mandatory and is recorded here so it does
not depend on anyone remembering it in three weeks.** Required text:

> FITS is **not** the deployed arm. This panel is shown under `model.active: fits`, set for
> this capture only. The study found FITS beats **neither** reference: direction **0.5055**
> against an always-long bar of **0.5564**, and **+0.31 points** against its own
> white-noise twin.

**Reasoning.** A screenshot showing a capability the deployed system does not run, without
saying so, is exactly the misreading the panel itself exists to prevent — and it would be
committed by **the report** rather than by the dashboard, which is worse. The same argument
that puts the 86%-is-geometry measurement inside the panel puts this caption under the
figure.

---

## 2026-08-23 — GB-59 pulled forward: a clean clone reproduces, and three defects it found

**Decision.** The reproducibility audit, scheduled for the last week, was run **eight
weeks early**. Every reason to run it in October was a reason to run it now: the data
snapshot is tracked, `results.csv` carries every axis, `report.py` regenerates from that
file alone. It either passes and the last week has one less thing in it, or it finds
something and finding it now costs a day rather than the submission. **It found three
things, and none of them would have been cheaper in October.**

**The result, with its boundary named.** A clone taken **from the remote** — which also
tests that everything needed was pushed, a failure a local clone cannot see — into a fresh
venv, interpreter and directory:

| | outcome |
|---|---|
| `pytest` | **1129 passed, 0 skipped**, 8:30 — identical to the development machine |
| the grid | 877 rows, **every result column bit-identical**; 19 min against 31 on a machine also running tests |
| `prepare_live` | band identical at `lower = 0.026076278765685856`, `model.json` **byte-identical** (sha256 `afde41c2…`) |
| `report.py` | `report.md` and both figures, from `results.csv` alone |

**The deployed artefact reproducing is the more surprising half.** That path runs through
training, per-fold validation calibration **and** a threshold grid search, and comes back
bit-identical. **The boundary: one machine.** Process, venv, interpreter and directory
varied; CPU, core count, instruction set and OS did not, and BLAS reduction order can
differ with any of them. **Cross-architecture reproduction is untested and is reported as
untested.**

**BLAS is pinned to one thread by default** (`--blas-threads`, default 1), on a
counterbalanced measurement rather than the first one:

| | runs (U P P U) | mean |
|---|---|---|
| unpinned | 40.7 s, 36.8 s | 38.7 s |
| pinned | 40.9 s, 37.7 s | 39.3 s |

**0.5 s between conditions against 3.2-3.9 s within them** — not distinguishable from
noise. The first attempt, 70.8 s against 62.3 s, was an artefact of order and machine load
and is **withdrawn**. The likely reason it is free belongs beside the parameter counts:
**these models are too small to parallelise** — DLinear holds 4,800 reals and FITS 1,200,
and a `(120, 4)` by `(120,)` matmul does not saturate fourteen cores. **Defaulting it was
verified not to change the numbers** — a pinned run reproduces an unpinned one
bit-for-bit — so the committed `results.csv` is still regenerated by the default command.

**And the inference it does not support.** Pinning removes **one** source of cross-machine
divergence. It does not remove different instruction sets dispatching different kernels,
different BLAS builds, or a different platform `libm`; single-threaded on two different
CPUs can still differ in the last bits. **Defaulting it does not make the claim
architecture-independent**, and reporting it that way would be the overreach the boundary
above exists to prevent.

**Defect 1 — the configuration is mutable during a long run, and nothing said so.**
`settings.yaml` was edited at 17:00 while a grid started at 16:36 was still running, so
the committed `results.csv` carried a `config_hash` matching nothing on disk. Only the
provenance column caught it. **Closed twice:** `study.provenance` compares both hashes and
`report.header` prints the warning where a reader of the tables cannot miss it; and
`live_loop.config_drift` runs at **every session start**, because a three-day live run
makes this worse — every decision record is the audit trail GATE 2 and §7 read. **It does
not reload:** a loop that silently changed its own behaviour mid-run could open a position
under one risk policy and manage it under another with nothing marking the seam.

**The split is the checkpoint gate's, applied to a second artefact.** `model_config_hash`
mismatch condemns the file; `config_hash`-only mismatch warns and names the sections the
difference must be in. A guard that invalidated a 20-minute grid over a polling interval
is a guard somebody eventually weakens — the spurious-retrain problem, solved once
already on 19 Aug.

**Defect 2 — the README's install procedure was a note, and its author skipped it.** Step
3 is three commands; the third, the editable install, was omitted while running this very
audit. **A non-editable install still imports correctly from the repository root**,
because Python puts the working directory on the path, so every test would have passed
while testing an install nobody would ever have. `scripts/setup.ps1` now runs all three,
stops at the first failure, and **verifies**: `import glassbox` resolves, the package is
editable and resolving to the repository rather than `site-packages`, and every
heavyweight dependency imports. It refuses Python below 3.12 naming the cause, which pip
does not.

**Defect 3 — a deep clone lost a tracked file and reported success.** From a 187-character
root, measured in the order the failures occur: `git clone` dropped one tracked file,
reported **every** file as deleted because git's own `stat` calls fail, and **exited 0**;
with `core.longpaths=true` the clone completed but **`python -m venv` failed at
`ensurepip`**, leaving `python.exe` with no `pip.exe`; `pip install` was never reached, so
the `lxml` error the README described is not the first thing anyone hits. **Two different
settings fix different steps** — `git config core.longpaths` fixes git only; the Windows
registry `LongPathsEnabled` is what Python needs — and the README said only "enable long
paths".

**And this one was fixable rather than documentable.** Tracked path lengths: n = 172,
median 28, p95 56, p99 64, **max 83** — a single outlier, the reference project's own
training log, which `REFERENCE_AUDIT.md` does not cite. Removing it:

| | longest tracked path | maximum clone root |
|---|---|---|
| before | 83 | **176** |
| after | **66** | **193** |

The audit's deep root was **187**, so the repository now clones cleanly at a depth where
it silently lost a file. Nothing cited by the audit was removed — `backtesting.py`,
`evaluation.py`, `models/DLinear.py`, `models_backtest.py`, `strategies.py`,
`data_provider/data_loader.py`, `utils/metrics.py`, `utils/tools.py` all stay — and
nothing is destroyed: the file remains retrievable at `a547c0a` and every earlier commit.
**A README row is a note; removing the path is a mechanism.** The row is rewritten anyway,
because a shorter path raises the ceiling rather than removing it.

**Consequence.** `results.csv` is replaced by the regenerated artefact — **not a hand
correction**: a complete machine-generated file from a clean clone whose `config_hash`
matches the committed configuration, with every result cell identical to what was already
reported. `checkpoints/live` was regenerated in the main repo only after a backup and a
field-by-field comparison: `model.json` and `history.csv` byte-identical, every threshold
field identical, `model_config_hash`, `seed`, `stats` and `training` identical, and
`config_hash` the only thing that moved.

---

## 2026-08-23 — The COF sweep, and a correction to what significance can be asked to do

**The sweep, measured.** 877 rows, 576 reportable, 41 cells, 656 arm-folds, 29.3 minutes.

| cutoff | COF | direction | vs always-long | **vs its own noise twin** | MAE | flatness |
|---|---|---|---|---|---|---|
| 2 | 60 | 0.4971 | −5.93 | **−0.71** | 0.0167 | 0.4187 |
| **5** | 24 | 0.5055 | −5.09 | **+0.31** | 0.0158 | 0.2733 |
| 10 | 12 | 0.4969 | −5.95 | **−0.79** | 0.0156 | 0.2048 |
| 20 | 6 | 0.4956 | −6.08 | **−0.93** | 0.0155 | 0.1560 |

**No optimum at any cutoff.** Three of four sit *below* their own white-noise twin and the
fourth is +0.31 points over it, which is inside noise; all four sit five to six points
below the always-long bar. A sweep finding nothing at every cutoff is a stronger statement
than not having swept, and it is the specific question §1.4 set out to ask.

**Correction 1 — the determinacy prediction has weak evidence against it, not an absence
of evidence.** It was first recorded as *untested* because it was conditional on there
being an optimum. That was the wrong reading. The term predicts **visibly higher
fold-to-fold variance at 1.35×**, and the per-fold sd of direction is
**0.0592 / 0.0627 / 0.0630 / 0.0546** across determinacy ratios spanning **1.35× to 119×** —
flat. **An absence of the predicted variance is information, not silence.** The reading
everything else in this study supports: **if the model learns nothing, there is nothing to
overfit, so determinacy does not bite.**

**Correction 2 — two claims were being conflated, and only one of them is this study's.**

| | claim | what it is | support |
|---|---|---|---|
| **(a)** | **No arm beats the always-long bar** | a claim of **absence** | negative in **9 of 9** anchor-arm cells, **−4.3 to −6.5 points**, holds under **both** null controls, no positive cell anywhere |
| (b) | Every arm is *significantly worse* | a **positive** claim of an effect | survives Holm at **3 of 9** in the pre-specified nine-test family |

**§1.4 asked (a). The report makes (a) and does not make (b) anywhere.** Reporting the
Holm result as a weakening of the headline was an error: it weakens (b), which we should
not have been claiming.

**And the sentence about the instrument travels with the claim.** Holm-Bonferroni guards
against **false positives**. Applied to a claim of **absence** it makes it *easier* to
conclude that nothing was found — **not a protection but a bias toward this study's own
conclusion**. It is reported because declining to report it would be worse, and the claim
rests on **nine negative cells and a five-point gap**, not on a p-value. A reader who knows
statistics will look for that sentence, and its absence would cost more than the correction
does.

**Correction 3 — the eleven Holm survivors are not a finding about the models. They are the
flatness artefact in significance clothing.** All eleven of the 79 real-only tests that
survive correction are **MAE against persistence, with the trained model worse**.
Persistence forecasts **zero**, which is **maximally flat**. Across the COF sweep
**Spearman(MAE, flatness) = +1.00** — MAE falls 0.0167 → 0.0155 exactly as flatness falls
0.4187 → 0.1560, and the best-MAE cutoff forecasts at **15.6% of the truth's magnitude**.

**So the only results in this study that survive a family-wise correction are measuring
flatness, and the winner is the model that forecasts nothing.** That is the **third and
cleanest** demonstration of §7.3's ban — after the 16-fold scaler comparison and the
Spearman +0.811/+0.668 measurement — and the first delivered by the significance machinery
itself. It is stronger as a finding **about the metric** than it would be as one about the
models, and it is reported that way.

**A note on the wall clock, because an estimator nobody trusts is one nobody reads.** The
run took 29.3 minutes against a 22-minute estimate, and the cause was **concurrent test
runs on the same machine**, not the sweep: cutoff 2's sixteen folds took **97.4 s** against
the centre cutoff's **97.5 s**, so the 6× larger model cost nothing measurable. The
estimator is sound and the machine was not idle. The run log now names the cutoff, which it
did not — the six COF-spoke lines were indistinguishable from one another.

---

## 2026-08-23 — A multi-day run needs a heartbeat, and the loop could not span a day at all

**The gap, and it was found by planning the run rather than by running it.** The decision
to leave the loop running between sessions rests on the process being **alive**, and
**a loop that died at 02:00 and a loop correctly idling produce identical output: nothing.**
Rule 2 bounds the overnight residual only while the process lives, and the residual banner
of this morning has nothing to be true about if nothing is there to print it.

**And a second thing the plan assumed and the code did not do.** `run_session` **returns**
when the market closes — it has no idle state at all, and returns immediately when called
outside a session. "Leave it running between sessions" was not a configuration choice; it
was a feature that did not exist. The longest run to date is a handful of cycles, so
nothing had ever asked for it.

**Decision — three additions, each small and each closing one of the three ways a
multi-day run fails quietly.**

1. **`heartbeat`.** One line every `live.heartbeat_seconds` (900) while idle: timestamp,
   state, positions held, uptime. `idle_state` separates **weekend**, **holiday** and
   **outside session**, because they are three different facts about the same silence and
   a reader of a night of heartbeats needs to know which one they are looking at. **A
   silent log now means dead rather than quiet.**
2. **`run_sessions`.** Runs N exchange sessions in one process, idling between them, and
   returns **one `SessionReport` per session, separate and in order**. They are not merged:
   each carries its own `held_at_open`, which is the answer to *did this session begin
   holding risk*, and a merged report cannot say it twice. **Only an ordinary close
   continues to the next session** — a run stopped by SIGINT or by `max_cycles` returns
   what it has, because a loop that stopped for a reason should not silently start again
   tomorrow. `--sessions N` on the CLI.
3. **The idle path runs no cycle.** No fetch, no broker call, no decision record. Seventeen
   idle hours cost the decision store nothing, which is asserted rather than assumed: a
   test counts `run_cycle` invocations across a run that idles overnight and requires the
   count to equal the number of cycles **reported**. A heartbeat every fifteen minutes is
   ~68 lines a day; a cycle record every sixty seconds while the market is shut would be
   ~1,000.

**Consequence.** `live.heartbeat_seconds` is a new config key. It is in the `live` section,
which is **not** in `MODEL_SHAPING_SECTIONS`, so `model_config_hash` does not move and no
checkpoint is refused — the split made on 19 Aug for exactly this reason paying off a
second time. `CLOSED = "the market closed"` becomes a module constant because `run_session`
writes it and `run_sessions` reads it, and a string spelt in two places is the defect family
this project has five instances of. 8 new tests, including two sessions in one process
driven by a clock the loop advances by sleeping, so seventeen idle hours cost seventeen
iterations.

---

## 2026-08-23 — RULING: the protective legs stay DAY, and the residual is stated rather than closed

**Decision.** `TimeInForce.DAY` stays. **No code change.** The GATE 2 condition recorded on
20 Aug offered two ways to close the overnight gap and **both are refused**, so the
condition is amended rather than satisfied: the gap is accepted, and it is **reported as a
measured limitation**.

**Why GTC is not available.** It is not a preference. Measured against the paper API and
recorded in this file: `qty=0.5` with `time_in_force=GTC` returns
`{"code":42210000,"message":"fractional orders must be DAY orders"}`. GTC is therefore only
reachable by **rounding to whole shares**, which reintroduces exactly the price-level
discretisation GB-18 chose fractional sizing to avoid — and it would not be a local change
to the executor. `engine.risk.shares_for` is shared by the backtester and the executor, so
whole-share sizing changes **position sizing across the entire study**, and every arm in
`results.csv` would be describing a different sizing rule from the one the report explains.

**Why flattening at the close is worse.** The backtest **holds overnight**. A live loop that
flattens at the close is running a different strategy from the one being evaluated, and the
divergence is not a rounding residual — it is a different exit rule on every position. That
breaks train/live parity, which is the property this project is built on and the one
`test_train_live_parity` exists to defend. Closing a protection gap by introducing a parity
gap is not a fix.

**What already covers most of it.** The five-rule policy of 18 Aug, unchanged: arm in the
same cycle that observes the fill; **re-arm every open position before any entry at the
start of every session**; verify both legs live every cycle and treat a naked position as a
risk event; cancel the surviving leg when one fills; and state the residual as the arming
interval rather than as "unprotected overnight".

**THE RESIDUAL, STATED AS MEASURED AND NOT AS UNLIKELY.** For the interval between a
session open and a successful re-arm, and for **any session the loop does not run at all**,
an open position carries **no broker-side protection**. Not "brief", not "unlikely": DAY
orders are expired by the broker at the close, so the position is naked from the open until
the loop arms it, and if the process is not running that morning it stays naked for the
whole session. The defence is **rule 2 and it is a real defence but a partial one** —
re-arming runs first on any start, so a loop that starts at all closes the gap before it
does anything else. It does nothing for the session in which the loop never starts.

**Consequence.** The GATE 2 condition is amended in spec §8 with its original text kept.
GB-57 carries the residual in the words above. `executor.py` is untouched: this is a ruling
that the current behaviour is the right one, and a ruling that changes no code still gets an
entry, because the next reader will otherwise re-open it.

**And the residual is printed rather than documented, because a ruling nobody sees at the
moment it applies is a footnote.** `live_loop.overnight_residual` emits a banner block at
session start whenever the book already holds something: the symbols and their decision
ids, that protection was **NONE between the previous close and this session's first
arming**, why (`TimeInForce.DAY`, and both escapes refused), and how it is handled (rule 2
re-arms before any entry, and the first cycle is that moment). A session that opens flat
prints nothing, because a warning that fires when there is nothing to warn about is one
nobody reads. `SessionReport.held_at_open` carries the same fact into the session summary,
read **before the first reconciliation** — the only moment the book still describes what
was carried *into* the session rather than what it holds now. The cycle records cannot
answer this on their own: a position re-armed in cycle 1 looks identical to one opened in
cycle 1.

**Operating decision for the live sessions of 24-26 Aug: leave the loop running between
sessions.** The residual has two halves and rule 2 closes only one of them. Re-arming
before entries bounds the gap for every session the loop **runs**; it does nothing for a
session the loop **misses**, and a process started by hand at 16:30 is a process that can
fail to be started. A loop that runs continuously across the close is inside the market
guard all night — it does nothing until the calendar says there is a market — and then
arms in its first cycle of the new session. The fallback, if it dies overnight, is to start
it **before** 16:30; that is a worse plan and it is the one being avoided.

---

## 2026-08-20 — Which claims survive which test: the grid, classified

**Decision.** Every claim the grid can speak to is classified against **both** required
tests, and the classification is the reportable object rather than any single number.
35 cells, 560 arm-folds, **15.1 minutes of arm time** against a 36-minute estimate;
679 rows, 480 reportable.

| claim | grid sensitivity (3 anchors) | null control (white noise) | verdict |
|---|---|---|---|
| **No arm beats the always-long bar** | **−4.25 to −6.48 pts** at every anchor for every arm; Wilcoxon p ≤ 0.021 in 7 of 9 cells | the arms sit at their noise level, the bar does not | **survives both** |
| FITS beats DLinear on direction | sign-stable: +0.96 / +0.83 / +0.88 pts | **+0.18 pts under white noise** | **survives neither** — p = 0.34 / 0.61 / 0.49, never significant |
| The wavelets help DLinear | **fails**: +0.30 → +1.24 → +1.36 pts, p = 0.46 → 0.103 → **0.039** — significant at exactly one anchor of three | **+0.57 pts under white noise**, a third of the anchor-0 effect | **survives neither** |
| Any arm extracts signal at all | — | **−0.47 / −0.74 / +0.31 pts** against its own noise twin, p = 0.94 / 0.86 / 0.94 | **survives neither** |

**Only one claim survives both, and it is the negative one.** Every positive claim the grid
suggests dies to at least one test, and two of them die to both. The wavelet claim is the
`r = −0.47` pattern exactly: **significant at one anchor of three**, and a third of its
anchor-0 effect present with no data at all.

**A validation of the control design, and it is worth reporting for its own sake.** The
always-long bar under the two nulls:

| always-long bar | direction | edge over chance |
|---|---|---|
| real data | 0.5560 | **+5.60 pts** |
| shuffled returns | 0.5418 | +4.18 pts |
| white noise | 0.5064 | **+0.64 pts** |

**Shuffling preserves the return distribution including its positive mean, so drift
survives it; zero-mean noise has none, so the bar collapses to chance.** The two controls
therefore separate **order** from **drift**, and the one that kills the bar is the one that
should. That is the controls behaving as designed, measured rather than asserted — and it
is what licenses reading "the arms are at their noise level" as a fact about the arms.

**Determinism confirmed**, not assumed: `test_the_same_grid_twice_gives_the_same_numbers`
runs the grid twice and compares every column but `seconds`, which is wall time and not a
result.

**The cancellation column now reports what it can.** `dlinear` 224 of 224 cells filled,
mean **0.6174**; `fits`, `persistence` and `buy_and_hold` **0 of 112 / 224 / 112** — empty
rather than 1.0000 or 0.0000, because a model with fewer than two contributing channels has
nothing that could cancel. Suppressed at the point of the number, per the ruling.

---

## 2026-08-20 — The first grid artefact was wrong, and the test that catches it was already written

**What happened.** `results.csv` reported **MAE 1.0014 and flatness 53.05** for the
buy-and-hold rows, on a series whose scale is 0.015. Buy-and-hold makes a directional call
and **no magnitude forecast**: its `predicted` array is that call, not a log-return path.
`ArmRun.forecasts` exists precisely to say so, and `smoke_offline.fold_table` honours it —
*"a metric an arm does not produce is reported as absent, not as a number"*. My `_row`
ignored the flag.

**The order of operations was the error, not the omission.** The test asserting it —
`market["mae"].isna().all()` — was **already written** when the 40-minute grid was
launched; it is cache-gated and slow, so it had been deselected from the quick run and had
never executed. **A test written and not run is a note**, which is the rule in `CLAUDE.md`
§3 arriving from a direction it had not been stated in: the rule says a *comment* is not a
mechanism, and this says a *deselected test* is not either.

**Consequence.** `mae`, `rmse` and `flatness` are gated on `arm.forecasts` in the one place
the row is built; `direction` and `direction_reference` are not, because the always-long
call is real and is the bar by construction. The grid was re-run rather than patched — a
results file corrected by hand is a results file nobody can reproduce, and the re-run also
re-exercises the determinism the study claims. **Rule of thumb worth keeping: run the test
that validates an artefact before spending forty minutes producing it.**

---

## 2026-08-20 — The first full grid: no arm distinguishes the market from noise

**Decision.** Reported as it came out. 679 rows, 480 reportable, 199 null-control or
skipped; 7 conditions × 16 folds × 5 live arms plus the market reference.

**The reference condition** (anchor 0, `lr` 1e-3, real data), 16 folds:

| arm | direction | bar | MAE | flatness | cancellation | return | Sharpe | aside |
|---|---|---|---|---|---|---|---|---|
| dlinear C0_base | 0.4959 | 0.5564 | 0.0161 | 0.316 | 0.578 | −0.31% | −0.33 | 5/16 |
| dlinear C2_hybrid | 0.4989 | 0.5564 | 0.0167 | 0.422 | 0.545 | +0.05% | −0.18 | 4/16 |
| fits C0_base | **0.5055** | 0.5564 | 0.0158 | 0.273 | 1.000 | +0.42% | +0.38 | 2/16 |
| persistence | — | 0.5564 | **0.0153** | 0.000 | 0.000 | 0.00% | — | 16/16 |

**No arm beats the always-long bar. Persistence still wins MAE.** And the wavelets do not
rescue DLinear: C2_hybrid moves direction from 0.4959 to 0.4989, both below the bar.

**The null control is the finding, and it is unambiguous.** Direction accuracy:

| arm | real | shuffled | white noise |
|---|---|---|---|
| dlinear C0_base | 0.4959 | 0.5021 | 0.5006 |
| dlinear C2_hybrid | 0.4989 | 0.5071 | 0.5063 |
| fits C0_base | 0.5055 | 0.5039 | 0.5025 |

**Every arm scores the same on real data as on a market with no temporal structure at
all**, and DLinear scores *worse* on the real one. Mean total return says it again: FITS
returns **+0.42% on the real market and +1.32% on shuffled returns**.

**The phrasing is fixed, and the first version of it was too short.** *"There is no
measurable difference between this market and noise, to these models"* is saved only by
its last four words, and a reader remembers the first clause. On the same footing:

| | direction accuracy |
|---|---|
| always-long, real data | **0.5564** |
| DLinear, real | 0.4959 |
| DLinear, white noise | 0.5006 |
| FITS, real | 0.5055 |
| FITS, white noise | 0.5025 |

**The models sit at their noise level. The market does not.** Always-long extracts **5.6
points above chance** from the same windows, scored by the same function, so the structure
exists and is reachable. The claim is therefore:

> **These models extract nothing from this market that they do not also extract from white
> noise.** That is a statement about the models, not about the market: over the same
> period, a constant always-long rule beats chance by 5.6 points, so the structure exists
> and is reachable — just not by explicit or implicit frequency decomposition at this
> horizon, on this universe, with these channels.

That answers §1.4 with a **control** rather than only with a baseline, and it is far harder
to attack than "the market is noise" — which is a claim this data refutes.

**And it survives grid sensitivity**, which is the right direction for a null: direction
runs 0.4959 / 0.4973 / 0.5006 across the three anchors for DLinear and 0.5055 / 0.5056 /
0.5095 for FITS, below the bar at every one.

**The learning-rate axis reproduces §7.3's argument in one table.** DLinear C0_base:

| `lr` | flatness | MAE |
|---|---|---|
| 1e-3 | 0.316 | 0.0161 |
| 1e-4 | 0.091 | 0.0153 |
| 1e-5 | **0.019** | **0.0153** |

At `1e-5` the model forecasts at **1.9% of the truth's magnitude** and its MAE **equals
persistence's exactly**. The best-MAE model is the one that forecasts nothing, measured on
this project's own grid rather than argued from the definition.

**One measurement artefact, flagged rather than reported as a result.** FITS's
`cancellation` is **exactly 1.0000** in every cell, and that is arithmetic rather than
evidence: FITS is univariate, so one channel contributes and `|Σc| / Σ|c|` is 1 by
construction. **Cancellation is only meaningful for a multi-channel model**, and GB-57 must
say so where it presents the column — otherwise FITS reads as the arm whose explanation
never cancels, which would be a claim about the model rather than about the channel count.

---

## 2026-08-20 — GB-49: two decisions the grid forced, reported rather than assumed

**Decision 1 — the design is a star, not a cross product.**

| | conditions | cells | arm-folds | estimate |
|---|---|---|---|---|
| full cross (3 anchors × 3 rates × 3 controls) | 27 | 135 | 2,160 | **~140 min** |
| star: one centre, one departure per axis | 7 | 35 | 560 | **~36 min** |

The instruction was to report the count and, if it exceeded an hour, say which axes to
cross fully and why. It does — 2.3 hours — so: **cross nothing fully; depart from one
centre on each axis.** The 20 extra conditions buy *interactions* (does the learning-rate
effect differ at anchor 42 under shuffled returns), which nobody asked and which no claim
in this project rests on. What each axis actually needs is a **common reference to depart
from**, and that is exactly what a star gives. `--full` runs the cross product for the day
somebody wants an interaction, and `--plan-only` prints both numbers before anything runs,
so a 36-minute command is a decision rather than a surprise.

**Decision 2 — the null arms go in the same `results.csv`, gated, not in a second file.**

The requirement is that they are **never averaged with arms that are about the market**.
The strongest available mechanism is not physical separation — a second file can be read
alongside the first by anyone, and it can drift out of step in columns or in vintage.
It is a **single named gate**, which is the shape this project already used for exactly
this problem: `records.is_reportable` says which provenances may reach a metric, once,
rather than at each caller. `study.reportable` is the same function for the same reason,
and a test asserts a null row cannot pass it.

There is a second argument, and it is the one that settles it: **the null rows are only
meaningful beside their real counterparts.** The whole point is the comparison — real
+1.9640 against noise +1.9309 — and splitting them across files turns that comparison into
a join, which is a step somebody eventually gets wrong in the other direction.

**Consequence.** Every axis added since §7.4 was written is a column: `anchor`, `lr`,
`control`, `flatness` (immediately beside `mae`), `cancellation`,
`data_snapshot_last_bar`. The deliberately empty FITS × C2_hybrid cell is written as a
**skipped row carrying its reason** — a blank cell reads as a run that failed, and this one
is a design decision. `run_arm` and `run_buy_and_hold` became public in `smoke_offline`
rather than being re-implemented here, so there is one path from a fold to an `ArmRun`;
`cancellation` moved onto `ArmRun` for the same reason, since it falls out of the
explanation layer for free.

---

## 2026-08-20 — STANDING REQUIREMENT: every headline claim gets both tests

**Decision.** Grid sensitivity **and** a null control, for every headline claim, reported
beside the real effect size.

**Reasoning, and it is two demonstrations rather than an argument.** The tests catch
different failures and neither subsumes the other:

| | grid sensitivity | null control |
|---|---|---|
| `r = −0.47` timing correlation | **died** (→ −0.0025 under a one-week shift) | would have passed |
| FITS phase advance | passed, **48 of 48 cells**, sd 0.0203 | **died** (+1.9640 real vs +1.9309 on white noise) |

Either test alone would have let one of these two through, and both were written up as
findings before the test that killed them was run. The controls are **within-window
shuffled returns** and **white noise with matched variance**, and the world is rebuilt end
to end — prices regenerated from the new returns — because perturbing the features alone
would leave the backtester trading the real market while the model forecast a synthetic
one, and every trading metric would compare two different universes.

**And a heuristic for when to reach for the control first**, now in `CLAUDE.md` §3: on
financial data a **coefficient of variation near 1%** across independently trained models
is evidence the measurement is about the machine. The phase advance was **1.04%**. Four
years of equity returns do not produce agreement that tight; a deterministic operator does.
**Suspicious stability calls for a control, not a larger sweep** — a larger sweep would
have made the claim look stronger.

---

## 2026-08-20 — A result about FITS, not only about our measurement

**Decision.** The 86% figure is reported in GB-57 as a **result about the architecture as
published**, not as a caveat on the figure.

**The claim.** If **86% of the learned frequency response is reproduced by a model trained
on white noise** — gain curves correlating at **+0.9485**, gain against
`|η·k − round(η·k)|` at **Spearman −0.9427, p = 1.8e-11** — then the complex layer is
spending most of its 1,150 effective parameters on a **deterministic resampling operator
that could be written in closed form**, leaving roughly **14%** for anything about the
data. Any implementation faithful to the paper has this; it is not an artefact of ours.

**It is configuration-specific, and the cost is a function of `η = 1 + H/L`.** Measured
over the `COF = 24` retained bins at `L = 120`:

| H | η | bins aligned | mean misalignment |
|---|---|---|---|
| 1 | 121/120 | 1/24 | 0.0958 |
| **4 (configured)** | **31/30** | **1/24** | **0.2833** |
| 12 | 11/10 | 3/24 | 0.2333 |
| 60 | 3/2 | 12/24 | 0.2500 |
| 120 | 2 | **24/24** | **0.0000** |

**Correcting the natural framing, which was mine and the supervisor's both:** the cost is
**not monotone in `H/L`**, and "short horizons are the worst case" is only half right.
`H = 120` (`η = 2`) is free, and `H = 1` is *better* than `H = 4` — because at `η − 1 =
1/120` even the highest retained bin is barely displaced. **The worst case is the middle,
and this project's configuration sits in it.**

**Consequence.** The implication is **parked, not chased**: initialise the complex layer
with the analytic resampling operator and learn only the residual, so the capacity goes to
the data — `IDEAS_PARKED.md`, with this measurement as its motivation and with the note
that the interesting test is not whether MAE improves but whether the *residual* response
still correlates with a white-noise control.

**And the sentence that makes it land stays:** the *trough* is stable across all 48 models
(median 8.00 days, range 8.00–9.23) and the *peak* is not (median 40, range 17.1–120).
**The geometry is the stable half and the data is the unstable half**, which is the
opposite of the natural guess.

---

## 2026-08-20 — The phase finding is withdrawn, and the sweep would have confirmed it

**Decision.** The claim that FITS advances 15–30 day cycles by about two days — recorded
this morning as GB-41's momentum result "arriving from the other side" — is **withdrawn**.
It is a property of the B+F objective's geometry, not of equity returns.

**The sweep asked for was run and it did not catch it.** 16 folds × 3 fold-grid anchors, 48
independently trained models: **+1.9582 days, sd 0.0203, positive in 48 of 48 cells**, and
indistinguishable between anchors (+1.9569 / +1.9538 / +1.9640). That is *more* stable than
the r = −0.47 timing correlation that vanished under a one-week shift. **Stability across
folds would have confirmed the claim**, which is the whole problem: a check that a correct
explanation and an incorrect one both pass is not evidence. It is §4.4's exactness tautology
in a new costume, and it is the third time this project has met that shape.

**What separates them is a control, not more folds.** Fit the same model on data with no
temporal structure:

| | phase, 15–30 day band | mean gain |
|---|---|---|
| real returns | **+1.9640 days** | 0.8191 |
| within-window shuffle | **+1.9553 days** | 0.8286 |
| white noise | **+1.9309 days** | 0.8435 |
| untrained (zero weights) | +0.0000 | 0.0 |

**It survives the complete destruction of temporal structure.** And so does most of the
rest: the whole gain curve on white noise correlates with the real-data curve at **+0.9485**,
trough at **8.00 days in both**, mean absolute difference **0.0319** against a curve range of
**0.2335** — about **86% of the response's variation is data-independent**.

**The replacement finding is better than the one it displaces.** Gain tracks **grid
misalignment**: over the 23 retained bins, averaged across all 48 models, **Pearson
r = −0.9049 and Spearman −0.9427 (p = 1.8e-11)** against `|η·k − round(η·k)|`. The
best-aligned bin `k=1` (120 days, misalignment 0.033) has gain **0.8102**; the worst,
`k=15` — **8.00 days, misalignment exactly 0.500** — has **0.5758**. The trough is the
**half-integer bin**, the frequency that lands exactly between two output bins and cannot be
reconstructed. The phase even changes sign at that boundary (bins 1–14 about +1.9 days,
15–23 about −1.0), which no market story predicts and grid geometry does.

So **GB-46's figure keeps its place and changes its meaning**: it shows the cost of
interpolation — §6.1's `η = 31/30` made visible in trained weights — rather than a discovery
about equity cycles. What remains data-dependent is the ~14% that differs from the noise
model, and the fact that the *peak* period is unstable (median 40 days, range 17.1–120)
where the *trough* is not (median 8.00, range 8.00–9.23). **The stable half is the geometry
and the unstable half is the data**, which is the opposite of how a reader would guess.

**Consequence.** The withdrawn sentence stays in the spec beside its correction, and the
general lesson goes to GB-57: **when a measurement is suspiciously stable, the next test is
a control and not a larger sweep.** Twelve hours separated writing the claim from
withdrawing it, and the only reason it was caught is that the instruction was "do not trust
it yet" rather than "confirm it".

---

## 2026-08-20 — GB-47/GB-48: the wavelets, and the floor that did not move

**Decision.** `features/wavelets.py` decomposes **one trailing window per bar** and emits
the last sample of each reconstructed band. `PARITY_WARMUP` entries may now be a function of
the configuration.

**Why not the standard recipe.** Transforming the whole series once and taking the bands as
features is what most of the literature does, and it leaks: a DWT filter is **two-sided**, so
the reconstruction at `t` is a weighted sum of samples on both sides of `t` — at level 3 with
`db4`, tens of bars ahead. It looks exactly like a smoothing, and every downstream metric
improves. `wavelets.whole_series_approximation` implements it **on purpose**, for
`BatchNormForecaster`'s reason, and GB-48 asserts that GB-10's harness rejects it in **both**
perturbation modes — separately, because a leak only one mode saw would be a reason to doubt
the other.

**The warm-up is a different kind of number from RSI's**, and the difference is worth
keeping visible. RSI's 325 is a tolerance argument about how fast a recursive seed decays,
re-derived once already after the first derivation proved wrong. The wavelets' 64 is
**exact**: a DWT of a trailing window depends on that window and on nothing before it, so two
callers holding the same 64 bars compute the same number to the last bit. There is no residue
to bound.

**Measured, and against expectation: `min_history_bars` does not move.** It is a **maximum**
rather than a sum, and RSI's 325 still dominates the wavelets' 64, so the floor stands at
**445** for `C0_base` and `C2_hybrid` alike. The C2_hybrid parity sweep is byte-identical at
that floor over **5 symbols × 25 timestamps**, inheriting GB-27's sweep rather than being
certified at a point — which is the lesson that file exists to carry.

**One thing did change shape.** `CHANNEL_BUILDERS` now takes `(bars, cfg)` for **every**
channel, including the four that ignore the second argument. The alternative was a second
table for the configured builders, which is the defect class `CLAUDE.md` §3 now names in its
general form: one fact in two places with nothing keeping them equal. Same reasoning for the
warm-up table admitting a callable — hardcoding 64 would put a literal beside the setting it
is supposed to follow, and the two would disagree the first time anybody tuned it.

---

## 2026-08-20 — CI red on the fourth instance of one defect class in two days

**Decision.** ``requirements.lock`` gains matplotlib and its five transitive pins, and
``tests/test_scaffold.py`` gains a test asserting that **every dependency declared in
``pyproject.toml`` is pinned in the lock**.

**What happened.** Run #17 failed with eight errors in ``tests/test_frequency_response.py``.
The local suite was green on 1,001 tests. The cause: **CI installs from the lock**, not from
the manifest — ``pip install -r requirements.lock`` then ``pip install -e . --no-deps`` — so
adding ``matplotlib`` to ``pyproject.toml`` installed it on the developer's machine and
nowhere else.

**This is the third-list defect again, and it is the fourth instance in two days.**
``VALID_MODELS`` against ``ALL_FORECASTERS``; ``smoke_offline``'s argparse ``choices``
against both; ``smoke_offline``'s ``run()`` default against ``model.active``; and now
``pyproject.toml`` against ``requirements.lock``. Every one is the same shape: **a fact
written in two places with nothing making them equal**, each half internally consistent, and
the disagreement invisible until something downstream reads the wrong copy.

**What worked, and it is worth naming.** The failure arrived as *"run this with the
project's interpreter"* on stderr rather than as a bare ``ModuleNotFoundError`` — because
``test_every_script_guards_its_imports`` refused the script until it carried an import
guard, hours before matplotlib was missing anywhere. **A test written for one reason caught
a different failure**, which is the argument for property testing GB-57 already carries from
GB-21, arriving again.

**Consequence.** The guard asserts the **declared** set, not the transitive closure: the
lock holds the closure and the manifest deliberately does not, so equality in both
directions would fail on every indirect pin. And it is a test rather than a note, per
``CLAUDE.md`` §3 — written into that file **the same day**, which is the part that stings:
the rule was added at 12:00 and the fourth instance shipped at 12:06.

---

## 2026-08-20 — RULING 2: MAE stays, and the pairing is structural

**Decision.** MAE remains a reported column. **It never appears without ``flatness`` in the
adjacent column**, and every table carrying it carries the rank correlations. §7.3 is
amended to cite the measurement rather than the argument.

**Reasoning.** Dropping the standard metric of the field, in a project comparing itself to
two papers that report it, reads as hiding a result rather than as methodological care. The
stronger move is to report it with its pathology attached, and the pathology is now this
project's own measurement: **Spearman(MAE, flatness) = +0.811 (DLinear) and +0.668 (FITS)
against Spearman(MAE, direction) = +0.009 and −0.128**, over 80 fold × arm cells each, and
**the 16-fold MAE winner is persistence, which forecasts exactly nothing and has a flatness
of 0.0.** A reader given MAE alone cannot tell an arm that got better from one that got
flatter.

**Structural, not editorial.** ``metrics.COMPANIONS`` builds the summary's columns and
``smoke_offline.FOLD_COLUMNS`` is asserted against the same rule, so printing MAE without
flatness means deleting the pairing rather than forgetting it — the ``dir_ref``-beside-
``direction`` precedent, applied to the metric that needed it more. GB-49 carries
``flatness`` in ``results.csv`` beside ``mae``.

**Consequence.** §7.3 used to ban MAE as a headline on a general argument about flattened
forecasts. It now bans it on a number measured here, which is a different kind of claim and
a stronger one — and the same move as GB-30's exactness correction: *"we assert X" and "we
assert X in a way capable of failing" are different claims.*

---

## 2026-08-20 — GB-45 and GB-46: the frequency view, and the figure that is the model

**Decision.** ``explain/spectral.py`` produces the per-frequency decomposition, gain and
phase; ``scripts/frequency_response.py`` draws it.

**Three design points, each following an existing ruling rather than inventing one.**

- **The matrices are measured, not derived.** ``FITSForecaster._extend`` gained an optional
  bin mask, so each frequency's ``(H, L)`` map is produced by running the **real forward
  pass** with one bin unmasked — ``forecast_matrix``'s discipline, for the same reason: a
  second copy of the pipeline written for the explanation is a second copy that can drift
  from the one that forecasts. The parts sum back to the whole at **4.4e-16**.
- **The sum goes through ``Attribution.from_terms``**, the one function in the codebase
  allowed to add a decomposition and refuse a residual (GB-30). The frequency view keys on
  periods and ``from_terms`` keys on strings, so labels are mapped out and back — a small
  indirection bought to avoid a second place that adds contributions and compares them to a
  total.
- **Days, not bin indices**, everywhere, including the plot's axis. "The 17-day cycle" means
  something to a supervisor; "bin 7" does not. Phase converts as ``φ/2π × period``, positive
  meaning the model advances that cycle.

**Two facts share the infinite-period key, and the module says so rather than letting a
reader guess.** ``per_frequency[inf]`` is the **RIN mean's** contribution — it bypasses the
complex layer entirely and is the largest single contributor in **37.2%** of test windows.
``gain_phase[inf]`` is the **learned bin-0 row**, which is dead (GB-44). Reporting bin 0 as a
measured zero rather than dropping it is the same choice as attributing 0.0 to a channel FITS
does not read: an absent entry and a zero entry say different things.

**Measured on fold 16, and the response is not flat.** Gain runs **0.596 to 0.830**, with a
trough at **8.6 days** and a peak at **24 days**: the model suppresses the fastest cycles that
survived the low-pass and favours the three-to-six-week band. The phase is the more striking
half — a consistent **+1.9 to +2.1 days** across that band, meaning the model **advances**
those cycles by about two days. **That is the momentum finding of GB-41 arriving from the
other side**, in a quantity the explanation layer produces anyway and with no extra
experiment.

The dominant contributor per window: the RIN mean **37.2%**, the 15-day cycle **22.4%**,
120-day 17.2%, 17.1-day 13.1%. **No contributor's mean share of the gross exceeds 0.211** —
24 contributors, none dominant, which is GB-30's cancellation story in the frequency domain.

**Consequence.** GB-46 lives in ``scripts/`` because it reads a checkpoint and writes a file,
which is §3.4's definition of an operational entry point — and because putting matplotlib in
``explain`` would let a rendering dependency into the live loop's import graph. It defines no
colour of its own, so the dashboard's green/red ban covers it, enforced by a test for a hex
literal in the source rather than by a repeated list of banned words.

---

## 2026-08-20 — RULING 1: the forecast target is scaled, and it is NOT for the MAE

**Decision.** ``build_windows`` divides the forecast target by the target channel's own
standard deviation. ``features.builder.restore_targets`` is the inverse, applied wherever a
forecast leaves the model layer. ``CHECKPOINT_VERSION`` goes to **2**.

**The reason is B+F, and only B+F. Say so, because a later reader will assume otherwise.**
``loss = mse(forecast, y) + mse(backcast, x)`` is an unweighted sum, so it is the objective
§6.2 describes only when the two terms are in one unit. They were not: the backcast was
supervised at unit variance and the forecast at ``0.015²``, making the objective **98.5%
backcast** — measured, 67× — so **FITS has not been training the objective the spec says it
trains**. That is a correctness defect, and it was invisible in the way that matters: the
loss fell, every test passed, nothing complained.

**The MAE improvement is a side effect and is not the motive.** It will look like the motive
— the number moves a lot — and §7.3 bans MAE as a headline precisely because it tracks
flatness at Spearman +0.81. The evidence that the fix is *correct* rather than merely better
is elsewhere: FITS lands at **0.015841 against the raw arm's 0.015822**, a difference of
0.12%, which is what scale-equivariance predicts for a linear model behind RIN. **A fix that
lands where the algebra says it should is a fix; one that merely improves a metric is a
tuning choice.**

**Scaled, never centred.** Centring makes ``predict`` affine, so the forecast carries a
constant belonging to no channel and ``Attribution.from_terms`` refuses the decomposition —
its error message names this exact case. Measured, the two forms are indistinguishable
(DLinear 0.016061 against 0.016068; FITS 0.015841 against 0.015802), so §4.4's exactness
decides it at no cost.

**y's statistics are the target channel's own, and are a derived accessor rather than a
stored field.** The ruling asked for ``ChannelStats`` to carry them beside ``X``'s, and it
does — but as ``scale_for(channel)`` rather than as a new number. The forecast target *is*
``close_logret`` one horizon ahead, so a separately fitted target scale would be a second
copy of one statistic and a **slightly different** one, which would reintroduce exactly the
mismatch the scaling exists to remove: the two B+F terms are in identical units only when
the divisor is identical. It inherits the training-split range and the GB-10 isolation
assertion for free, because it is the same object.

**Where the inversion had to go, and why it is not inside ``predict``.** The ruling asked
for ``predict`` to invert before returning. It cannot: ``Forecaster.predict`` (§4.3, frozen)
takes ``(B, L, C)`` and **no symbol**, and the scale is per symbol, so one fitted model has
nothing to route on — the same obstacle ``individual_weights`` met in GB-44. The inverse
therefore sits at the **model layer's boundary**, in one named function beside the transform
it undoes, applied at every point a number leaves: both entry points of ``model/predict.py``,
and the calibration and backtest paths of ``smoke_offline``. The requirement behind the
ruling holds — nothing above ``model`` sees a standardised value — and it is asserted
rather than asserted-about: the band is calibrated on restored forecasts, the metrics
compare restored predictions against restored truths, and a pooled batch is restored **row
by row** by each row's own symbol's number.

**The attribution carries the same factor**, or the dashboard would render contributions in
one unit beside a forecast in another, ~65× apart, with every exactness check still passing
because each side is internally consistent. ``attribute(..., scale=...)`` multiplies the
**weights**, so the sum still closes inside ``from_terms`` and no second summation exists.

**Consequence: every reported number changed, and not all of them improved.** See the next
entry. ``CHECKPOINT_VERSION`` is the mechanism that stops an old checkpoint being loaded and
forecasting ~65× wrong — nothing in the config hash covers a change to the data pipeline, so
the version guard is what is left.

---

## 2026-08-20 — What the target scaling did to every headline, measured and not softened

**Decision.** The pre-scaling numbers are retired and the new ones stand. They are worse on
every metric the study reports, and that is recorded here rather than absorbed quietly.

**Measured, 16 folds, before and after** (DLinear, the deployed arm):

| | before | after |
|---|---|---|
| folds standing aside | 3/16 | **5/16** |
| direction accuracy | 0.5182 | **0.4959** |
| always-long bar | 0.5560 | 0.5564 |
| total return, mean per fold | +0.40% | **−0.31%** (positive in 5 of 16) |
| Sharpe, where defined | +0.62 | **−0.33** (9 folds) |
| buy-and-hold | +7.73%, Sharpe 1.39 | unchanged |

**The honest reading, and it is not "the fix made things worse".** DLinear has **no backcast
term**, so none of the B+F argument applies to it; what the ruling changed for DLinear is its
**conditioning** — the median learned weight moves from **0.54× Adam's step to 2.50×**. And
the conditioning measurement already said what that would do: **the 0.5182 belonged to the
arm in which 71% of the weights sat below the optimiser's step size.** It was the score of a
model too under-trained to move its weights, not a directional edge. Training it properly
removes it.

**So the null headline is strengthened, not weakened.** A result that survives only while the
optimiser cannot resolve the weights was never a result. The supervisor's own line on this
was written before the number moved: *an optimiser mismatch could hide a real result — it
could not manufacture a null one.* This is the other half of that sentence, and it happens to
cut the other way: the mismatch was manufacturing a small **positive**, and correcting it left
the null.

**One consequence needs a ruling and is not taken here.** The most recent fold **no longer
stands aside**: fold 16 calibrates to `lower=0.026076` on a validation Sharpe of **+0.483 over
8 trades**, where before the fix its best candidate scored **−3.38 over 7**. So the deployed
system **now trades**, and §7's required finding — *"the deployed system declines to trade,
and that is a result", which must not be softened* — **is no longer true of the current
deployment**. It remains exactly true of the system as it stood on 18–20 Aug, and the report
can say so with dates. Whether the sentence stays as a finding about a past deployment or is
replaced is the supervisor's call, not this file's.

---

## 2026-08-20 — matplotlib, for one figure, and the alternative was worse

**Decision.** ``matplotlib>=3.11.1`` joins the dependencies. Required by CLAUDE.md §4: a new
dependency needs a line here.

**Reasoning.** GB-46's frequency response is §6.5's "single strongest visual in the demo" and
GB-57 needs it as a raster at 200+ dpi. The dashboard hand-writes SVG and will keep doing so —
Streamlit renders it inline and the geometry is testable as a string — but hand-writing a log
axis, tick formatting, a shaded cutoff region and a legend for a report figure is roughly 200
lines of fiddly code that would look worse and be no more testable. The project already
carries torch, streamlit and pyarrow; matplotlib is the smallest of them and the most
standard.

**Consequence.** It is imported **inside** ``render`` and only in ``scripts/``, so no package
module gains a rendering dependency and the live loop's import graph is untouched. The figure
**defines no colour of its own** — the palette comes from ``dashboard/app.py`` — so the
green/red ban asserted there covers it, and a test for a hex literal anywhere in the source
enforces that it stays that way.

---

## 2026-08-20 — The first head-to-head, and it is negative

**Decision.** The result is recorded as it came out, in the form GB-57 will report it: as a
delta against persistence on MAE and against the **always-long bar** on direction, per §7.3
and GB-19's ruling that persistence cannot be the reference for a direction column.

**Measured.** 16 folds, the shared model of GB-44 per arm per fold, scored through
``backtest/metrics.py`` rather than a re-implementation of it — which matters, because
``direction_accuracy`` excludes windows whose forecast is exactly zero from the denominator
and a report-side redefinition is how two ends of a study start disagreeing about a number.

| arm | MAE | persistence delta | better in | direction | above the bar in |
|---|---|---|---|---|---|
| persistence | **0.015286** | — | — | NaN (forecasts zero) | — |
| dlinear | 0.031413 | +0.016127 | **0/16** | **0.5182** (sd 0.0559) | **4/16** |
| fits | 0.096841 | +0.081555 | **0/16** | **0.5098** (sd 0.0735) | **4/16** |
| always-long | — | — | — | **0.5560** (sd 0.0927) | — |

**Neither model beats persistence on MAE in a single fold, and neither beats the achievable
constant strategy on direction.** They beat *each other* in 8 folds each, which is the
cleanest way to say that at this configuration the architectures are indistinguishable.

**Two things a reader must be told beside that table, both measured today.** First, **the
MAE column as configured is not a model comparison**: FITS's 0.0968 is the B+F unit mismatch
and DLinear's 0.0314 is the optimiser-resolution artefact, and at their best conditioning the
two land at 0.015629 and 0.015802 — **still behind persistence's 0.015286**, so the
conclusion survives its own correction. Second, DLinear's **0.5182 against a bar of 0.5560**
reproduces GB-27's independently re-measured figure to four decimals, which is the check that
this pipeline and GB-19's agree.

**Wall time, shared model, one fold**: FITS **mean 8.29s, median 6.78s, max 17.87s** (fold 15,
2,480 training windows); DLinear mean 2.58s. FITS is ~3× slower with 4× fewer parameters,
because it runs 60 epochs to DLinear's 23 before early stopping and pays for two FFTs a step.

**Consequence.** Reported as an outcome, not a shortfall, and in the same register as
GB-26's stood-aside band: **a study that reports a null result it measured is worth more than
one that reports an effect it hoped for.** GB-57 carries it with the conditioning caveat
attached to the MAE column and explicitly *not* attached to the direction column.

---

## 2026-08-20 — The scaler finding was two mechanisms under one label, and neither is the scaler

**Decision.** Nothing in the configuration changes. Yesterday's conclusion — *"the per-symbol
scaler does not earn its place"* — is **withdrawn as stated**: it was one label over two
unrelated mechanisms, and the third instance of the failure mode named above, arriving the
same day it was named and from my own report.

**The hypothesis under test** (the supervisor's, and it is arithmetic rather than opinion).
With the scaler ``X`` is unit variance, so a forecast near 0.015 out of a summed linear map
wants a single weight around **6.1e-4** — while **Adam's step at ``lr=1e-3`` is 1.0e-3, larger
than the weight it is looking for**. Without the scaler the wanted weight is roughly 41× the
step. If that is what drove the MAE result, then every MAE number in this project measures a
learning-rate choice as much as it measures a model.

**Measured: 16 folds × 2 models × 5 arms, 160 fits, MAE and direction always in raw
log-return units** (the ``y``-standardised arm's forecasts are inverted before scoring).

**1. DLinear — the hypothesis is confirmed, and the predicted number is nearly exact.**

| arm | median abs(w) | ×step | max abs(w) | below step | MAE | direction | flatness |
|---|---|---|---|---|---|---|---|
| scaled X, lr 1e-3 | 5.435e-04 | **0.54×** | 6.34e-03 | **71.1%** | 0.031413 | 0.5182 | 1.637 |
| raw X, lr 1e-3 | 7.067e-04 | 0.71× | 1.44e-02 | 60.8% | 0.020439 | 0.4688 | 0.733 |
| scaled X, lr 1e-4 | 1.571e-04 | 1.57× | 2.37e-03 | 36.3% | 0.018940 | 0.4888 | 0.616 |
| scaled X, lr 1e-5 | 3.332e-05 | 3.33× | 2.74e-04 | 17.6% | **0.015629** | 0.4935 | 0.191 |
| scaled X **and y**, lr 1e-3 | 2.496e-03 | **2.50×** | 2.51e-02 | 23.5% | 0.016068 | 0.4947 | 0.297 |

The predicted 6.1e-4 against a measured **5.4e-4**, and **71% of the learned weights sit below
the optimiser's step**. The MAE gap closes and reverses exactly as predicted: the scaled arm
loses to raw on MAE in **0 of 16** folds at ``lr=1e-3``, wins **14 of 16** at ``1e-4`` and
**16 of 16** at ``1e-5``. Standardising ``y`` moves the median weight to **2.50× the step** and
wins 16 of 16 at the unchanged learning rate, which is the predicted fix landing where it was
predicted to land.

**2. FITS — the hypothesis is refuted, and the real mechanism is a different one.** FITS's
weights are nowhere near the optimiser's floor: median **1.88e-2, nearly 19× the step**, with
only 7% below it. Lowering the learning rate closes nothing — the scaled arm loses on MAE in
**0 of 16 folds at every learning rate tried**. What is actually wrong is in the objective:

| FITS arm | forecast MSE | backcast MSE | backcast ÷ forecast |
|---|---|---|---|
| raw X, lr 1e-3 | 5.039e-04 | 3.428e-04 | **0.68×** |
| scaled X, lr 1e-3 | 1.017e-02 | 6.649e-01 | **67.0×** |
| scaled X, lr 1e-5 | 7.331e-03 | 9.701e-01 | **137.2×** |
| scaled X **and y**, lr 1e-3 | 9.656e-01 | 6.672e-01 | **0.69×** |

**B+F supervision requires ``X`` and ``y`` to be in the same unit.** With ``X`` standardised
and ``y`` left in raw log returns, the backcast term is supervised at unit variance and the
forecast term at 0.015², so the sum is **98.5% backcast** and the objective silently stops
being B+F. The model reconstructs a unit-variance window and emits a forecast **4.8× too
large** — measured flatness 4.83 against a right-sized 1.0 — and its MAE is 6× DLinear's for
that reason and no other. Standardising ``y`` puts the two terms back in one unit (0.69×) and
the MAE returns to **0.015802**, the best FITS number of the five arms and identical to the
raw arm to four decimals, which is what scale-equivariance predicts for a linear model behind
RIN.

**3. And a third thing, which says the whole MAE column is measuring something else.**
``flatness`` = mean abs(forecast) ÷ mean abs(y). Over all 80 fold × arm cells per model,
**Spearman(MAE, flatness) = +0.811 for DLinear and +0.668 for FITS**, while
**Spearman(MAE, direction) = +0.009 and −0.128**. The best-MAE DLinear arm forecasts at
**19%** of the truth's magnitude. So lowering the learning rate "fixes" MAE by flattening the
forecast — which is **precisely what §7.3 bans MAE as a headline for**, arriving here as a
correction to the fix rather than to the bug.

**4. The scaler does not remove the conditioning problem. It moves it.** Per channel, one
fold, DLinear:

| | mean abs(x) with scaler | median abs(w) | mean abs(x) without | median abs(w) |
|---|---|---|---|---|
| close_logret | 0.700 | 5.73e-04 (0.57×) | **0.0145** | 1.61e-03 (**1.61×**) |
| rsi14 | 0.820 | 4.01e-04 (0.40×) | **55.0** | 4.44e-05 (**0.04×**) |
| vol_z | 0.735 | 7.66e-04 (0.77×) | 0.770 | 6.17e-04 (0.62×) |
| mom10 | 0.780 | 2.79e-04 (0.28×) | 0.049 | 1.08e-03 (1.08×) |
| ma_dist20 | 0.781 | 2.87e-04 (0.29×) | 0.038 | 1.10e-03 (1.10×) |

With the scaler **every channel sits below the step**, 0.28× to 0.77×. Without it the inputs
span a factor of **3,800** — `rsi14` runs 0..100 while `close_logret` is 0.015 — and the
weights split: the target channel climbs to 1.61× the step and becomes resolvable, while
`rsi14`'s falls to **0.04×, twenty-two times below the step**. "Without the scaler ``X`` is
around 0.015" is true of one channel of five.

**5. What does not move, in any arm.** Direction accuracy: FITS **0.5056 to 0.5098 across all
five arms, a spread of 0.0041**; DLinear 0.4688 to 0.5182. An optimiser mismatch could hide a
real result; it could not manufacture a null one, and there is nothing in the direction
numbers to hide. **The headline is unchanged and is not conditional on any of this.**

**Consequence.** Configuration untouched, per the instruction. Awaiting a ruling on the ``y``
standardisation, which is the only arm that fixes both models at the learning rate in force.
Two things follow whatever is ruled: **GB-49 sweeps learning rate alongside its other axes**,
because a fixed ``lr`` is a hidden arm of the study; and **GB-57 states that every MAE
comparison is conditional on the learning rate and on the flatness it buys**, with the
Spearman figures above as the evidence. The direction column carries no such condition.

---

## 2026-08-20 — GB-44: `fits.individual_weights` stops being decorative, and what that cost

**Decision.** ``model/train.py`` consults ``fits.individual_weights`` (spec §5, §6.4). Under
``false`` — the default and the 2026-08-17 ruling — one model is fitted across the universe,
as it already was. Under ``true`` a **pooled universe is refused**, naming the remedy, and the
checkpoint manifest records the regime it was trained under as ``training.weight_sharing``.

**Reasoning.** The key was declared in the configuration contract, documented in §6.4 and
read by the loader from GB-2 onward, and **nothing consulted it**: it could be set to either
value and the system behaved identically. A key that reads as honoured and is not is worse
than no key at all, because a reader of the config file — the supervisor, at the defence —
is entitled to believe the file describes the system.

Two consequences follow, and neither is a preference:

- **``true`` is a caller-side regime, and that is forced by a frozen contract.**
  ``Forecaster.predict`` (§4.3) takes ``(B, L, C)`` and **no symbol**, so a single fitted
  model has nothing to route on and cannot hold five weight sets. Giving it one means
  changing the protocol every layer above depends on, which rule 1 forbids and which is not
  worth a study variant. What ``train`` already supports is the other half: ``frames`` may
  name one symbol, which GB-15's docstring has called "a valid special case" from the
  beginning. So ``individual_weights: true`` means *one call per symbol, one checkpoint per
  symbol*, and the mistake worth refusing is being handed the universe under that setting —
  which would pool five symbols into one weight set while the configuration says the
  opposite. A universe of **one** is accepted under either setting, because a universe of
  one has nothing to share weights across and the two regimes are then the same run.
- **It governs every arm, not only FITS**, despite living in the ``fits`` section. The
  section is where weight sharing is discussed; the force behind the key is the 2026-08-17
  comparability ruling, which is about arms being trained alike. A regime applied to FITS
  alone would train it per symbol while DLinear pooled — exactly the handicap that ruling
  exists to prevent, and the study would report the handicap as architecture.

**The defect the switch test was written for, and it was in a third place.**
``smoke_offline`` — the one command that runs the whole offline path and produces every
number the study reports — held its own list: ``--model``'s argparse ``choices`` literal
still read ``(persistence, dlinear)`` after GB-41 registered FITS, and ``run``'s signature
defaulted to the **string** ``"dlinear"`` rather than to ``cfg.model.active``. So setting
``model.active: fits`` changed the live checkpoint (``prepare_live`` reads the config) and
changed **nothing** about the study numbers, silently. A switch that reaches one half of a
system is worse than one that reaches neither, because the two halves then disagree without
saying so. Both literals are gone: the choices come from ``ALL_FORECASTERS`` and the default
is ``None``, meaning *whatever the configuration names*. The test that proves it is not a
parser test — it is one fold of cache → features → folds → train → calibrate → backtest with
FITS selected by configuration alone and no ``model=`` argument anywhere.

The codebase had anticipated this and still missed it: ``test_an_unknown_model_is_refused_by_the_parser``
carried the comment *"GB-41 adds it, not GB-24"*, and GB-41 did not. **A note saying a later
task will do something is not a mechanism**, which is the same lesson as §4.4's tautology in a
different costume.

**A second defect this closed, found while asserting the switch.** ``config/loader.py``
validates ``model.active`` against ``VALID_MODELS`` and **may not import the model layer** —
the layer contract forbids it — so ``ALL_FORECASTERS``'s keys are copied there by hand, and
nothing checked that the two agree. A name in ``VALID_MODELS`` alone passes configuration
validation and raises a ``KeyError`` inside ``train``; a name in the registry alone is a
model no configuration can select. Either way "``model.active`` is the only change needed"
would be false and no other test would notice. ``test_fits_integration.py`` now asserts the
two sets are equal.

**Consequence.** The sharing claim is asserted by **breaking** it rather than by reading it:
a run that quietly fitted five models would still return one object and still list five
symbols, so the test perturbs the single weight matrix and requires *every* symbol's forecast
to move. That test is what found the dead DC row below — its first attempt perturbed row 0,
and nothing moved for anybody.

---

## 2026-08-20 — Measured: 50 of FITS's 1,200 parameters cannot learn, and the row stays

**Decision.** Spec §6.1 now reports **1,200 allocated / 1,150 effective**, and the dead row
is **kept** rather than removed.

**Reasoning, measured on the configured geometry after a full fold's training.** RIN
subtracts each window's own mean, and the rFFT's bin 0 *is* that mean — so after RIN it is
zero on every window, and row 0 of the complex weight matrix multiplies zero on every forward
pass. It takes no gradient, never leaves its initial value, and cannot change a forecast:

| | measured |
|---|---|
| ``max abs(w)`` on row 0, after 100 epochs | **2.1e-11** |
| ``max abs(w)`` on rows 1.. | **0.89** (median 3.1e-02) |
| rFFT bin 0 after RIN | **3.6e-15** |
| rFFT bin 1 after RIN | **13.1** |
| forecast change when every row-0 weight moves by ``1+1j`` | **exactly 0.0** |
| the same perturbation on row 1 | **6.2** |

That is ``out_bins = 25`` complex weights — **50 reals, 4.17% of the count** — allocated and
dead. The FITS-to-DLinear ratio is 4.17× rather than 4×, which changes nothing about §6.1's
warning and everything about whether the number means what it says.

**Why it stays.** §6.2's low-pass keeps "the first ``COF`` bins" and bin 0 is one of them;
the source paper's architecture carries the same dead row for the same reason. Removing it is
a change to a specified pipeline, a retrain of every checkpoint and a changed parameter count,
bought for a cosmetic 4% — and rule 4 does not permit an architecture change nobody asked for.
**What is not acceptable is quoting 1,200 as capacity**, and that is the part fixed here.

**Consequence.** ``test_fits.py::test_the_dc_row_is_allocated_and_cannot_learn`` pins both
figures, because a number in a docstring drifts and an asserted one does not. GB-57 reports
allocated and effective side by side, and the ``n_parameters`` docstring carries the reason
at the point a reader would otherwise take the number at face value.

---

## 2026-08-20 — A named failure mode: MEASURING TWO EFFECTS AND BLAMING ONE

**Decision.** This project now has a name for a defect it has produced twice in two days,
from both ends of the review, and the name is written down so the third instance is
recognised rather than re-derived.

**The failure mode.** A measurement moves. Two mechanisms could have moved it. The write-up
names one. Nothing in the result says the other was there, the sentence reads as a finding
rather than a guess, and it survives review because the *number* is right — it is the
*attribution* that is wrong. It is the attribution bug of §4.4 wearing prose instead of
code: **a quantity is only evidence for a cause if some other cause could have been ruled
out, and an unruled-out cause leaves the claim unfalsified rather than confirmed.**

**Instance 1 — the supervisor's, spec §6.3.** The symptom paragraph said that omitting the
``(L+H)/L`` scale leaves "direction accuracy quietly degraded". Direction accuracy *is*
near chance, so the sentence matched the world. But two things were in the room — a
flattened forecast and a model with no directional edge — and the paragraph gave one cause
for both. A uniform positive factor cannot change a sign, so the scale explains the MAE
half and **nothing at all** of the direction half. Corrected 20 Aug, with the wrong line
kept beside its correction.

**Instance 2 — mine, GB-42.** The amplitude test fed a 12-day sinusoid at the configured
``H=4`` and asserted the backcast reconstructs. Two effects sit in that error: the
amplitude shrinkage the test exists to catch, and the bin-interpolation error of a
frequency that lands at ``η·k = 10.333`` and has no output bin to land on. The test
attributed the sum to amplitude. Measured, the interpolation term is **the whole of it** —
4.97 against an amplitude of 3.0 — so the test could not have passed a correct
implementation. The same slip in miniature, caught the same day: the first draft asserted a
recovered amplitude of exactly 3.0, where re-sampling onto a longer grid moves where the
samples fall relative to the peak and only the *ratio* is exact.

**Consequence.** Neither instance was found by a test, and neither could have been: both
were assertions about *why*. What found them was arithmetic done on purpose against a
sentence already written. The practical rule this leaves is small and mechanical — **when
a write-up names a cause, ask what else was in the room, and prefer the claim that isolates
one variable to the claim that explains everything** — and it is now the first question
this project asks of its own findings. GB-57 carries the failure mode by name, because it
generalises well past this codebase.

---

## 2026-08-20 — Spec §6.3 corrected: the amplitude trap is a comparison bug, not a performance one

**Decision.** §6.3's symptom paragraph is rewritten. It claimed that omitting the
``(L+H)/L`` scaling leaves "direction accuracy quietly degraded". **It cannot.**

**Reasoning.** Omitting the scale multiplies every sample by exactly ``L/(L+H)`` — a
uniform, *positive* factor, 0.9677 at ``L=120, H=4``. A positive scaling cannot change a
sign, so the cumulative forecast scales, its sign is identical, and direction accuracy is
unchanged to the last decimal. The per-fold trend thresholds are calibrated on forecasts
carrying the same factor, so the band adapts and the signals barely move either.

What it does damage is narrower and worse:

- **MAE and MSE improve**, because a flatter forecast sits closer to zero.
- **Cross-model comparison is corrupted.** FITS would beat DLinear on MAE by being flatter
  rather than by being better, on the one axis where the two are compared directly.

So the bug is real and GB-42 earns its place, for a different reason than the spec gave.
Its damage is **muted by §7.3 banning MAE as a headline**, adopted for an unrelated reason.

**Consequence.** The wrong line is kept in the spec with its correction beside it, the same
standard as the "unprotected overnight" correction: a document that reads better than its
history is a document nobody can check. And the correction is now a **measurement** rather
than an argument — ``test_the_amplitude_scale_changes_mae_and_cannot_change_direction``
asserts identical direction and differing MAE on real windows, which is the check that
would have caught the line when it was written.

---

## 2026-08-20 — GB-42's grid was wrong, and no correct FITS could have passed it

**Decision.** ``tests/model/test_fits_amplitude.py`` moves from the configured ``H=4`` to
``H=12``, and the choice of grid is stated in the test as part of what it tests.

**Reasoning, measured.** Bin ``k`` of a length-``L`` transform is the frequency ``k/L``;
the same physical frequency sits at bin ``k·η`` of a length-``L+H`` transform. Two things
follow, and the test as committed knew neither:

| | 12-day cycle, L=120 | naive zero-pad | frequency-preserving |
|---|---|---|---|
| **H=4** (configured) | ``η·k = 10.333`` | err **4.97** | err **4.97** |
| H=12 | ``η·k = 11.000`` | err 5.97 | err **2.2e-14** |

**Naive zero-padding never reconstructs** — leaving a coefficient at its old index changes
its frequency, which stretches the signal in time. And **frequency-preserving mapping
reconstructs only when ``η·k`` is an integer**, which at ``H=4`` it is not. So the
committed test asserted something no correct implementation could satisfy, and being
``xfail(strict=True)`` it would have turned the suite red the moment GB-41 landed and
worked.

**This is the same error as asserting a recovered amplitude of 3.0 where only the ratio is
exact**: measuring two effects and attributing both to one. An amplitude test at ``H=4``
measures interpolation error and blames the amplitude.

**Consequence.** The gap at ``H=4`` is not a defect to fix — **it is the reason FITS has a
learned complex layer at all.** The layer is what interpolates between bins that do not
line up. A second test now measures the ``H=4`` failure explicitly, so the grid choice is
justified in the file rather than in a commit message.

---

## 2026-08-20 — GB-41: the four measurements

**1. Parameter count: 1,200 reals, measured.** ``COF = 120 // 5 = 24`` input bins mapped to
``ceil(1.0333 × 24) = 25`` output bins, ``24 × 25 = 600`` complex weights. The prediction in
§6.1 was exactly right and is now a measurement. **Counted in reals deliberately** — a
complex weight is two learnable numbers, and reporting 600 against DLinear's 4,800 would
flatter FITS by a factor of two on the very axis §6.1 warns the report not to borrow a
framing from.

**2. RIN is instance normalisation, and the contract test catches the alternative.** Broken
deliberately once — ``closes.mean()`` instead of ``closes.mean(axis=1, keepdims=True)``,
one keyword:

```
FITS (per-window RIN)        max change in window 0 = 0.0            HOLDS
BatchRINFITS (batch mean)    max change in window 0 = 4.849950e+00   LEAKS
FITS           contract property 6: PASSED
BatchRINFITS   contract property 6: FAILED -> fits.predict LOOKS AHEAD: perturbing rows
               after 2024-01-09 (scale) changed its value at 2024-01-01
```

Spec §4.4's note — *"property 6 is what catches instance normalisation implemented as batch
normalisation... a live risk for FITS's RIN stage in GB-41"* — was written before FITS
existed and is now confirmed against it.

**3. The mean-reversion hypothesis is rejected for FITS. It is a momentum model.**
Agreement between ``sign(forecast)`` and ``sign(trailing H-day return)``, over 16 folds and
4,690 test windows:

| model | agreement | sd | above 0.50 |
|---|---|---|---|
| dlinear | **0.4991** | 0.0630 | 8/16 |
| fits | **0.5725** | 0.0733 | **13/16** |

Above 0.50 means the forecast *continues* the recent move. **FITS leans with the trailing
return in 13 of 16 folds; DLinear is indistinguishable from a coin flip.** A low-pass filter
that keeps only cycles of five days and longer is a smoother, and a smoothed extrapolation
continues a trend — so the result is what the architecture implies, but it was hypothesised
the other way and is now measured.

**Recorded, because the hypothesis and its correction are both part of the result.** The
hypothesis on file — a low-pass filter, so a mean-reverting forecast — was the supervisor's,
and it was **backwards**. What makes it worth recording rather than quietly dropping is that
**the architecture implied the answer before any data was touched**: a low-pass filter is a
smoother, a smoothed extrapolation continues a trend, and continuation is momentum. The
measurement was not needed to *find* the direction — it was needed to establish that the
implementation does what the architecture says, and to give the size. GB-57 reports the
hypothesis, its direction, and the fact that the architecture answered it, because a paper
that reports only confirmed hypotheses is a paper that has hidden its reasoning.

*Corrected mid-measurement:* the first run took the trailing return from the **scaled**
window. ``build_windows`` applies ``(x - mean)/std``, and the mean shift can flip the sign
of a small return, so that measured the scaler as much as the model. On raw returns FITS
moved 0.5906 → **0.5725** and DLinear 0.5006 → **0.4991**. The conclusion held; the numbers
did not.

**4. The per-symbol scaler does not earn its place under FITS.** Over the same 16 folds,
each model fitted twice — once with the fitted per-symbol statistics and once with identity
statistics. ``y`` is **not** scaled by ``build_windows`` (only ``X`` is), so the MAE
comparison is in one unit and is like-for-like:

| model | MAE with scaler | without | scaler better | direction with | without | scaler better |
|---|---|---|---|---|---|---|
| dlinear | 0.031413 | **0.020060** | 0/16 | **0.5182** | 0.4657 | **12/16** |
| fits | 0.096841 | **0.015846** | 0/16 | 0.5098 | **0.5133** | 9/16 |

**The two axes disagree, and the disagreement is the answer.** On MAE the scaler is worse
in 16 of 16 folds for both models — because ``X`` is standardised to unit variance while
``y`` stays in raw log returns near 0.015, so the model must learn a gain of about 1/65 and
a zero start with a fixed epoch budget does not get there. On direction the scaler **earns
its place under DLinear** (0.5182 against 0.4657, better in 12 of 16) and **does not under
FITS** (0.5098 against 0.5133, better in only 9 of 16 and worse on the mean).

**Nothing is changed on the strength of this.** It is one training budget and one
initialisation, MAE is banned as a headline by §7.3, and the scaler also serves the four
channels FITS does not read but DLinear does. It is recorded as a measured asymmetry between
the arms for GB-49 to sweep and GB-57 to report, not as a configuration change.

**Consequence.** FITS is registered in ``ALL_FORECASTERS`` and passes the §4.4 contract test
**with no edit to that test**, which was the integration claim. Attribution names every
active channel with 0.0 for the four it does not read (GB-33's ruling), and is exact because
the whole pipeline is linear in the window — the ``(H, L)`` matrix is built by pushing the
``L`` basis vectors through the **actual forward pass**, so it is a measurement of the
implementation rather than a second derivation that could drift from it.

---

## 2026-08-20 — GATE 2 criterion 2 amended, because the system behaved correctly

**Decision.** Spec §8's GATE 2 criterion 2 is replaced. It read *"at least one order
placed, filled, and reconciled"*. It now reads:

> **2. EXECUTION PATH PROVEN LIVE.** In a live session, a loop-produced decision becomes an
> order, fills, is reconciled, adopted and protected. This may be demonstrated with a
> deliberately permissive **rehearsal band** when the deployed band stands aside — the
> criterion tests the machine, not the model.
>
> **2b. DEPLOYED BAND BEHAVES CORRECTLY.** Whether it trades or stands aside, the behaviour
> matches what validation selected, and the gate log states which occurred.

**Reasoning, and it is the reason this entry exists rather than a one-line edit.** The
original wording conflated two questions — *does the execution path work end to end against
a real broker?* and *does the deployed model trade?* — and only the first is a gate. The
second is a **result**, it already has an answer, and the answer is no. The most recent
complete fold's best calibrated candidate scored a validation Sharpe of −3.38 over 7
trades, so `Thresholds.never()` is in force and no entry can fire.

**The amendment was prompted by the system behaving correctly, not by it failing.** That is
worth stating plainly because the opposite reading is the natural one. A criterion that
cannot be met is usually a sign of unfinished work; here it was a sign of a finished system
declining to trade, and of a criterion that had encoded an assumption nobody had noticed
making — that a deployed model would want to trade. Left as written, 25 Sep would have
cancelled FITS by a rule aimed at an unfinished product. **The criterion was fixed, not the
system.**

**Consequence — four conditions, each closing a way a rehearsal could become a lie.**

1. **It says what it is.** The band appears in the run banner, and the provenance of every
   record is `rehearsal:<reason>` — never `live`. `records.load_decisions` defaults to
   live, so the same default that hides a replay hides this.
2. **Nothing it produces is reportable.** `records.is_reportable` is the one place that
   says so, and the suite asserts it over GB-19's two inputs. The trade log needed more
   than a filter: `Trade` carries no provenance and cannot, because it is a frozen contract
   (§4.2) that GB-19 reads without translation — so **a rehearsal emits no trade at all**,
   enforced at the single point that can enforce it.
3. **It closes out before the close**, 15 minutes by default. The protective legs are DAY
   orders, so an overnight hold is an unprotected hold. *A rehearsal that holds overnight
   fails the rehearsal.* This is the DAY-legs condition applied where it bites.
4. **It is small**, capped at a notional the operator names, 25 by default. The sizer is
   left alone and its output is capped, because the point is that the ordinary path runs.

The band itself is `Thresholds(lower=1e-9)` — the smallest the contract admits, since
`Thresholds` refuses zero. It could not be mistaken for the output of a grid search, which
is exactly what it is for.

**Criterion 6 — two or three separate live sessions — stands unchanged and unmet.** It is a
floor on observation, not on the model, and no amendment reaches it.

---

## 2026-08-20 — The config hash is split rather than narrowed

**Decision.** `config_hash` keeps its exact current meaning: the full resolved
configuration, written to every `DecisionRecord`, answering *"under what settings was this
decision made?"*. A second function is added. `model_config_hash` covers only the sections
that can shape a trained weight — `data`, `window`, `wavelet`, `fits`, `channels`, `model`
— plus `meta.seed`. **Checkpoints gate on the narrow one.** `load_checkpoint` refuses on
`model_config_hash` and *logs* when `config_hash` differs while `model_config_hash`
matches, so a live-only change is visible without being fatal. Both go in the manifest.

**Reasoning.** Adding `live.retry_attempts` and `live.retry_backoff_seconds` for GB-39 —
two numbers a polling loop reads and no weight can see — changed `config_hash` and made
`load_checkpoint` refuse every existing model. The guard was correct and the cost was pure
waste. Narrowing `config_hash` itself was rejected for the right reason: it would change
what `DecisionRecord.config_hash` promises, and that field is a frozen contract whose whole
value is that a record ties to *the exact* settings that produced it. So the answer is a
second field, not a narrower first one.

**The danger is not the retrain.** It is Sprint 4, where FITS and the COF sweep touch the
configuration repeatedly, and **a guard that fires spuriously every time is a guard
somebody eventually weakens or works around**. Fixed before that pressure exists rather
than under it.

**Consequence.** `universe` is deliberately *not* in the model hash, even though it decides
which symbols were trained on. `Predictor.stats_for` already refuses an untrained symbol by
name — *"the checkpoint holds no normalisation statistics for X; it was trained on [...]"* —
so a universe change surfaces as a precise refusal at the point of use rather than as a
blanket retrain. `meta.version` is excluded too: it versions the configuration *schema*,
which no weight sees. Tests assert both directions — a live-only key leaves the model hash
untouched, and `window.input_len` moves it — plus one that walks every model-shaping
section by name, so a section added to the list without effect is caught.

---

## 2026-08-20 — CI is read, not assumed

**Decision.** `scripts/ci_status.py` reports the CI conclusion for a commit, using the
credential git already holds for github.com. Exit codes make it usable in a shell
condition: 0 succeeded, 1 failed, 2 no run yet, 3 no credential.

**Reasoning.** Three consecutive reports were written on an **assumed** green. `gh` is not
authenticated on this machine and the repository is private, so the unauthenticated API
returns 404 — and the gap between "I could not check" and "it is fine" got crossed silently
three times. The credential that pushes the commit can read the run; there was never a
reason to guess.

**Consequence.** Verified for the commit that prompted this: run #13, `e4b55b2`,
`completed / success`, and every run back to #8 is green. The token is read through
`git credential fill` and never printed.

---

## 2026-08-20 — GB-40, GATE 2 walked: 3 of 6 green, and FITS is not cancelled today

**Decision.** GATE 2 is **not passed**. **FITS proceeds into Sprint 4 as planned**: the §8
cancellation rule is armed, not fired, and its trigger is the gate's own date of **25 Sep**,
not a rehearsal of the gate run five weeks early.

**The walk, with evidence rather than assertion.**

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | Live loop runs a full session unattended | **FAIL** | Cycles have run — a dry-run cycle and three real ones on 18 Aug — but no open-to-close unattended session is recorded. Outside the window `run_session` returns `stopped_by='outside the session'` after 0 cycles, which is the guard working, not a session. |
| 2 | At least one order placed, filled and reconciled | **FAIL** | 0 of the live log's decisions produced an order. The loop has *reconciled* two real Alpaca fills — GB-22's entry and the operator flatten — but has never *placed* one. |
| 3 | Every decision carries an exact attribution, visible in the dashboard | **PASS** | 75 records across live and replay; worst `abs(sum(per_channel) - forecast_total)` = **1.248e-08** against a 1e-5 tolerance, an 800x margin. Every record has channels, every one renders, and the dashboard shows the table, the bars and the titles. |
| 4 | Replay reproduces a recorded day offline | **PASS** | 14 bars of fold 13, 14 cycles, 0 failed, 70 decisions — run with `load_live_bars` and `AlpacaBroker` **replaced by functions that raise**, so "offline" is measured rather than assumed. |
| 5 | Co-Pilot approve and reject | **PASS, replayed** | The loop queued 2 recommendations; one approved → submitted with both protective legs armed, one declined → broker order count unchanged; 70 records after, not 72, because the answer amends. Never exercised against Alpaca. |
| 6 | Two or three separate live sessions, not one (`SOLO_BUILD_PLAN` §4.1) | **FAIL** | One completed bar has been decided live: 2026-08-17. |

**Reasoning — the three failures reduce to two causes, and neither is a defect.**

**(a) The deployed band stands aside.** Criteria 2 and 5 are both downstream of it: a band
that cannot fire cannot produce an order, and therefore cannot produce a recommendation to
approve. This is the result GB-20 ruled on and Ben ruled is a headline rather than a
problem — the most recent fold's best calibrated candidate scored a validation Sharpe of
−3.38 over 7 trades, so the system deployed today does not trade.

**(b) Live session time.** `SOLO_BUILD_PLAN` §4.1 is explicit that a session is an evening
of the operator's time, and that the gate needs two or three of them. One bar has been
decided. That is a scheduling fact, not an engineering one.

**Why the §8 rule does not fire today.** The rule exists to stop a research arm starting
when the system is not yet a product **at the gate date**. Applying it on 20 Aug to a gate
committed for 25 Sep would cancel FITS on a rehearsal, five weeks before the question it
answers is due. The rehearsal is worth running early — it is how the blocker below was
found — but its result is a status, not a verdict.

**Consequence, and this is the part that needs a ruling before 25 Sep.**

**Criterion 2 is structurally unreachable under the deployed configuration, not merely
unmet.** The honest deployment choice is the most recent complete fold; that fold stands
aside; a band that stands aside can never place an order. So no amount of session time
fixes criteria 2 or 5 — they will read FAIL on 25 Sep exactly as they do today, and FITS
would then be cancelled by a rule aimed at an unfinished product, on a system that is
finished and has correctly decided not to trade.

Three ways out, and the choice is Ben's:

1. **Deploy a fold whose band fires.** Honest only if the selection rule is stated in
   advance and is not "the one that trades" — which is the fold-13 discipline applied to
   deployment.
2. **Let a labelled rehearsal satisfy it**: run the loop against Alpaca with a firing
   band, clearly recorded as a rehearsal of the execution path rather than as the deployed
   system trading. This proves the mechanism, which is what the criterion is for.
3. **Accept the replay path as satisfying it**, on the grounds that the same `run_cycle`
   places, fills and reconciles there — and state in the report that the live half was
   never exercised.

Option 2 is the one that answers the criterion as written without bending the deployment,
and it interacts with the DAY-legs condition below: a rehearsal that opens a position must
close it before the session ends.

---

## 2026-08-20 — GATE 2 CONDITION: no overnight hold while the protective legs are DAY

**Decision.** Recorded as a **condition of GATE 2**, not as an open question.

> The loop must not hold a position overnight while protective legs are DAY. Either the
> policy moves to GTC before any overnight hold occurs, or the loop flattens at the close.
> Decide it before GATE 2, not during it. If the band stands aside through the gate, note
> in the gate log that the condition was **never exercised** — an untested guarantee is not
> the same as a satisfied one.

**Reasoning.** Alpaca expires DAY orders at the close, so a position held overnight is
unprotected overnight — which the backtest models as protected, making the residual a
number the study would otherwise state wrongly. The morning half of this is closed: the
re-arm no longer collides with a consumed `client_order_id`, so the position is no longer
liquidated at market two cycles into the next session. The overnight gap itself is not, and
closing it changes the protection policy rather than fixing a bug in it.

**Consequence.** As of 20 Aug 2026 the condition is **not exercised**: no position has been
held overnight by the loop, and the deployed band stands aside, so none can be. That is
recorded here so the gate log cannot later read as though the guarantee had been tested.

---

## 2026-08-20 — A stated principle: a decision is recorded when it is COMPLETE, not when it is computed

**Decision.** The general form of the rule GB-39 needed, stated once so it applies beyond
the case that produced it:

> **A decision is written to the log at the moment it is complete — when the system has
> both made it and done what it implies. A verdict the system has computed but is
> forbidden, unable or not yet permitted to act on is not a decision it has made, and
> recording it as one creates a decision nobody will ever revisit.**

**Reasoning.** Two rulings collided and the collision was silent. "The first cycle may
reduce risk and may not add any" (GB-26) and "a bar is decided once" (19 Aug) are both
right, and together they stranded every entry: the first cycle computed an `ENTER_LONG`,
recorded it, and the second cycle then declined to re-decide a bar that was already in the
log — so the entry was never submitted. **"The first cycle does not enter" would have
become "the session never enters", every session, and every test still passed**, because
each ruling was tested against its own case and neither test asked what the other did.

The resolution generalises. Recording *computation* makes the log a record of what the
system thought; recording *completion* makes it a record of what the system did, which is
what this project exists to produce. The same principle decides GB-37's amendment — an
operator's answer completes a decision rather than starting a new one — and it decides the
ordering GB-39 depends on, where the record reaches disk before the order reaches the
broker precisely because the decision is complete at the moment it is *sent*, not at the
moment it is filled.

**Consequence.** Where a verdict cannot be acted on, the bar is left undecided and the next
cycle that can act records it once, with the order it produced and a narrative written in
the knowledge of that order. The cost is a repeated forward pass on the first two cycles of
a session. The alternative was a class of defect that passes every test.

---

## 2026-08-20 — Measured: an operational config key invalidates every trained checkpoint

**Decision.** Recorded rather than fixed. `config_hash` covers the whole of
`settings.yaml`, so adding `live.retry_attempts` and `live.retry_backoff_seconds` for GB-39
changed the hash from `39692e78…` to `421d54ae…`, and `load_checkpoint` refused every
existing checkpoint with *"refusing to load a model shaped by other settings"*. Both the
live checkpoint and the replay fold had to be regenerated.

**Reasoning.** The guard is correct and must not be weakened: it cannot know that a polling
retry count has no bearing on a trained weight, and a hash that tried to know would be a
hash somebody has to keep right. But the cost is real and asymmetric — a key that cannot
possibly affect a model forces a retrain of every model, and on a day with a market session
in it that is an hour that buys nothing.

**Consequence.** Not changed now: narrowing the hash to the model-shaping sections would
alter what `DecisionRecord.config_hash` means, and that field is a frozen contract (§4.2)
whose whole value is that a record ties to *the exact settings that produced it*. Stated
here so that a config change is planned as a retrain, and so GB-59's clean-clone audit
expects it. If it becomes a real cost, the answer is a second field, not a narrower first
one.

---

## 2026-08-19 — RULING: decide once per completed bar, manage every cycle

**Decision.** An entry decision is computed and recorded **once per `(as_of, symbol)`**.
Later cycles in the same session neither re-decide nor re-record it. Reconciliation,
protection, trade emission and exits are **explicitly outside the rule** and run on every
cycle. `records.save_decision` refuses a second record for the same key and
`records.load_decisions` asserts uniqueness on load, so a duplicate arriving by any route
surfaces rather than silently doubling a metric.

**Reasoning.** A decision's `as_of` is the last **completed** daily bar, which does not
change during a session. At 60-second polling a 6.5-hour session is 390 cycles, so the loop
re-derived the identical verdict from identical inputs 390 times and wrote 390 identical
rows per symbol — **1,950 decision records a day** over the five-symbol universe. The log
was the visible half. The question it exposed was whether, with a live band, the loop would
attempt an **entry** every 60 seconds.

This is the generalisation of GB-26's first-cycle ruling: **a cycle may always reduce risk;
adding it is what is rationed.** An entry is a function of daily data and can change once a
day. An exit is an obligation on capital already committed, and prices move intraday, so it
is re-*sent* until the broker has it — never re-*decided*.

**What would have prevented a repeated entry submission today: the Book, and only as a
race.** `run_cycle` sized entries over `[s for s in ranked if s.symbol not in
state.book.symbols()]`, and the book is rebuilt each cycle from the broker's **positions**.
A position appears only on the fill. `_absorb_entry` polls 10 times at 1.5s and, if the fill
has not landed, returns without booking anything — so the next cycle sees no position, no
holding, and sizes the same entry again. The executor added nothing: `decision_id` was
`f"{cycle_id}-{symbol}"`, a **different** string every cycle, so Alpaca's own duplicate
rejection could never fire. This is exactly the race, not a guard.
`tests/test_live_loop.py::test_a_repeated_entry_is_never_submitted_when_the_fill_has_not_appeared`
runs the loop against a broker that accepts and never fills, and pins one buy per symbol.

**Consequence.**

- `decision_id` is now `f"{as_of:%Y%m%d}-{symbol}"` — **derived from the bar, not the
  cycle**. Because `execute` passes it through as the entry's `client_order_id`, a second
  entry for one bar is refused **at the broker**, which is the only place that can win a
  race against a fill that has not appeared. Measured, see the entry below.
- `LiveState.decided` caches the day's verdict per symbol and is **seeded from the log** at
  the first cycle, so a restart mid-session does not re-decide bars it already recorded.
- Exits move out of the decision loop into `_send_exits`, which runs every cycle and asks
  **the broker** — not a flag in memory — whether the exit order is already there.
- `records.amend_decision` is added, and GB-37's approval uses it. An operator's answer is
  not a second decision at the same bar; appended, it would have doubled exactly the count
  this ruling protects, and the approval rate computed over the log would have been the
  answers divided by twice the recommendations.
- **Where the two rulings meet.** The first cycle may not add risk, and a bar is decided
  once. Recording an `ENTER_LONG` verdict on the first cycle would strand it — the next
  cycle would decline to re-decide and the entry would never be submitted, turning "the
  first cycle does not enter" into "the session never enters". So a verdict the loop is
  forbidden to act on **leaves the bar undecided**: a decision is recorded when it is
  complete. The cost is one repeated forward pass on the first two cycles of a session,
  not on all 390.
- The existing live store held **20 lines for 5 unique decisions**. Repaired; see PROGRESS.

---

## 2026-08-19 — Measured: Alpaca consumes a `client_order_id` permanently, and three defects fall out of it

**Decision.** Two facts were measured against the paper account rather than assumed, and
each closes a live defect. The third defect is **reported and not fixed**, because it
changes the protection policy and that is Ben's ruling to make.

**Measured, 18 Aug 2026.** Submitting a limit order with `client_order_id` X, then
submitting X again:

- while the original is **working** → refused, `40010001 client_order_id must be unique`
- after the original is **cancelled** → refused, same code

So an id is consumed permanently. That is what makes the deterministic `decision_id` above
a real guard, and it is also what broke re-arming.

**Defect 1 — `AlpacaBroker.get_orders()` returned open orders only.** `TradingClient
.get_orders()` with no filter defaults to `status=open`. Every caller that matters asks
about orders that have **finished**: `records.emit_trades` builds the live trade log from
filled sells, so **the live trade log was structurally empty** and GB-19 had nothing to
measure over paper results; and `protect_book` detects a filled leg in order to cancel its
sibling, so rule 4 could never fire and **a filled stop would leave its take-profit working
as a naked sell**. Both silent — an empty list is a valid-looking answer to the wrong
question. Now requests `QueryOrderStatus.ALL`; every caller already filtered on status
itself. Asserted on the request, not inferred from behaviour.

The flatten on 18 Aug showed this masked: no trade was emitted, which was correct because
the position was quarantined — and *also* unavoidable, because no filled order was visible.
The right outcome arrived for one reason while a second reason would have produced it
anyway. That is the shape of a defect that survives a test suite.

**Defect 2 — re-arming reused a consumed id.** `protect_book` re-armed
`f"{holding.decision_id}-stop"`, the id the leg already had. Refused → arming failure →
`ARMING_STRIKES` of them → **`_flatten` at market**. Now `f"{decision_id}#{n}-stop"`, with
`n` derived from **the broker's own order history** so a restart cannot reset it into a
collision. The `-stop`/`-target` suffix is preserved, so `records.exit_reason_for` still
reads the leg from the id.

**Defect 3 — the protective legs are `TimeInForce.DAY`. NOT FIXED; Ben's call.** They
expire at every close, so a position held overnight is **unprotected overnight**, which the
backtest models as protected. Defect 2 then turned the next morning's re-arm into a forced
liquidation two cycles into the session. Defect 2 is fixed, so the liquidation is gone; the
overnight gap is not, and closing it means `DAY → GTC`, which is a change to the protection
policy recorded on 18 Aug rather than a bug in it. **No position has been held overnight by
the loop** — the only position the account has held was GB-22's, quarantined and never
protected — so this is latent, not realised. It is also moot while the deployed band stands
aside. Flagged for a ruling rather than adapted silently (CLAUDE.md §5).

**Consequence.** `ORDER_HISTORY = 500` is a new module constant in `executor.py`, flagged
rather than buried: rule 5 is config over constants, and the justification is that a key
here could only ever ask for *less* than everything the broker will tell us about, and
"reconcile over fewer orders than are available" is not a policy anyone wants to set.

---

## 2026-08-18 — GB-22 closed: the order filled at the open, and the position is quarantined

**The acceptance criterion is met, and the same run surfaces an operational fact worth
stating.**

| | |
|---|---|
| order | `385e982a-2a20-467b-85e6-a3d6f6bf89c4` |
| status | **FILLED**, 0.081919619 of 0.081919619 |
| fill price | **307.49** |
| filled at | **2026-08-18 13:30:02.113 UTC — two seconds after the open** |

That is exactly the fill GB-18's ruling 2 models: a decision made after the close cannot
fill at that close, it fills at the next session's open. The backtester's convention and
the broker's behaviour agree, measured rather than assumed.

**Protection did not arm, and every step behaved as designed.** `executor.protect` returned
empty at submission because the entry had not filled — its own log line says so: *protection
is not armed yet and must be armed by the cycle that sees the fill.* GB-26's rule 1 exists
to arm it in that cycle. But **GB-22's order was submitted by a one-off script rather than
by the loop**, so no `Holding` was ever written, and reconciliation therefore sees a
position with no decision behind it.

**Reconciliation quarantined it**, correctly: `UNKNOWN_POSITION`, 0.091919619 AAPL — the
0.01 probe and the 0.0819 entry merged into one broker position. And a quarantined position
is **never protected**, by design: *the system cannot protect or explain what it did not
open.*

**So the fact to state plainly: the only orders this system protects are the ones it
decided.** An order placed outside the loop is counted against buying power, shown in the
dashboard as `QUARANTINED`, and left alone. That is the correct behaviour of GB-23's
read-only reconciler, and it is also a live gap: the paper account currently holds
**0.0919 AAPL, unprotected**, at an average entry of 307.238389. Ben's standing instruction
to flatten both AAPL positions is outstanding and is **not** executed here — he is reviewing
the dashboard the position appears in, and flattening would empty the screen mid-review.

---

## 2026-08-18 — GB-38: which fold, and whether replay shares the decision store

### 1. Fold 13, chosen on exit-reason coverage — and the coincidence stated

**Decision.** Replay drives **fold 13**, test range **2025-07-07 to 2025-09-29**, band
`lower = 0.056133`, `upper = 0.122473`, calibrated on that fold's own validation split.

**The criterion was fixed before the numbers were looked at**: the fold should contain a
stop, a target and a signal exit. Measured across all 16 folds, **fold 13 is the only one
that contains all five exit kinds** — `stop` 2, `stop_gap` 1, `target` 2, `target_gap` 2,
`signal` 6, over 13 trades. The next best are folds 8, 5, 1 and 7 with four kinds each.

**The uncomfortable part, stated rather than left to be noticed.** Fold 13 also returns
**+2.71%**, the second best of the 13 folds whose band fired, with a Sharpe of 2.902. That
is a coincidence of the selection rule and not the rule itself, and the honest way to hold
it is:

- The **full return distribution** of eligible folds is on the record: −3.96%, −3.59%,
  −0.61%, −0.60%, −0.31%, +0.24%, +0.34%, +1.22%, +1.63%, +2.45%, +2.67%, **+2.71%**,
  +4.23%. Fold 13 is 12th of 13 ascending.
- **The fold is a parameter, not a constant.** `smoke_offline --prepare-replay FOLD DIR`
  takes it, so nothing is baked in and any fold can be shown.
- **Fold 1 is the recommended counterweight** and is equally replayable: four exit kinds,
  10 stops and 4 gapped stops, **−3.59%** and a Sharpe of −3.391. A demonstration that
  shows only fold 13 is showing a good quarter.
- Sharpe 2.902 does **not** trip §7.3's leak alarm: the alarm is on the aggregate, a
  per-fold Sharpe on ~60 bars has a standard error near 2.05 annualised, and the
  distribution is symmetric — folds 4 and 7 also exceed +2.0 while folds 1 and 2 reach
  −3.39 and −4.57.

### 2. One store, and the default filter is the safety property

**Decision.** Replayed decisions share `decisions/` with live ones.
`DecisionRecord.provenance` carries `"live"` or `"replay:fold-13"` — **a string naming the
fold, not a boolean** — and `records.load_decisions` **defaults to live**.

**Reasoning.** Two stores are the safer-sounding option and they duplicate the reader:
GB-19's metrics, the dashboard and the report would each need two paths, which is the
defect GB-29 already rejected when it moved `Trade` into the contract rather than keeping
one type per side. The real risk of a single store is a filtering bug, and the answer is to
make the failure direction safe rather than to hope: **a caller who forgets the filter sees
live decisions only.** A replayed decision can go *missing* from a view; it can never be
shown as real. The inverse default would have exactly the opposite property.

Naming the fold rather than setting a flag is the second half. If a replayed record ever
does reach a live view, it arrives labelled `REPLAY:FOLD-13` on its own row rather than as
an unlabelled row that a reader has to know to distrust — the dashboard labels **both**
origins, so neither can be read by default.

**Consequence.** `DecisionRecord` gains `provenance: str = "live"` (a §4.2 addition,
authorised by the GB-38 ruling). Records written before it decode as live, which is what
they were.

### 3. `ReplayBroker` is a demo instrument, not a backtester

It fills a market order at the bar and checks the protective legs against the bar's range,
**stop before target** as GB-18 ruled — but it models no gaps, no slippage, no fees and no
accounting identity, and `replay.py` may not import the harness that does. **A PnL produced
by replay is not a result and must never be reported as one.** What replay demonstrates is
that a decision can be made, explained, answered and recorded; what the strategy earns is
GB-18's question and is answered there. This is stated at the top of the module so nobody
quotes a replay figure into the report.

**Measured end to end**: 14 bars of fold 13 produced 70 decision records and **2
recommendations**; one was approved — submitted, 47.59 AAPL, **both protective legs armed**
— and one was declined, leaving the broker's order count unchanged. 72 records visible as
`replay:fold-13`, **0 visible as live**.

---

## 2026-08-18 — A stated principle: risk already taken is managed whatever the data says

**Principle, not a task ruling.** It was decided in GB-26 for stale market data and it
generalises past that case, so it is recorded here in the form it should be applied in:

> **A degraded input may never justify opening risk, and may never suspend the management
> of risk already taken.**

**Reasoning.** The two halves are different questions and the natural phrasing — "skip the
symbol" — collapses them. Opening a position is an act of judgement that requires the input
to be trustworthy. Managing an open position is an obligation that exists whether or not it
is. The system holds the second even when it declines the first, and the instruments that
carry it — the stop and the target sitting at the broker — are deliberately the ones that do
not depend on the model at all.

**Where it already applies, and where it is going to.** Stale bars (GB-26): the symbol is
dropped from ranking and entries and its protective legs are still verified and re-armed. A
missing or stood-aside band (GB-20, GB-26): no entry can fire, and every open position is
still reconciled and protected. It should be applied the same way to any future case where
an input is doubted — a failed feature build, a checkpoint hash mismatch, a broker read that
returns something implausible. **The test of a correct implementation is the asymmetry:** if
a degradation stops entries *and* stops protection, it has been implemented as a pause
rather than as a refusal, and a pause is the wrong shape.

---

## 2026-08-18 — GB-38 absorbs two GATE 2 items, because the deployed band stands aside

**Decision.** Ruled by Ben. GB-38's replay is extended to **drive the live loop's own
cycle** — the same code path, not a reimplementation — over historical bars from **a fold
whose band did fire**, using **the thresholds that fold actually calibrated**. It exercises
forecasting, ranking, risk, attribution, narration, the decision record, and **Co-Pilot
approve and reject**, entirely offline. Not implemented yet; recorded now.

**Reasoning.** The most recent complete fold stands aside (validation Sharpe −3.38), so the
live system cannot produce a recommendation. That leaves two GATE 2 items with nothing to
demonstrate: Co-Pilot approve/reject has nothing to approve, and replay has no day that
traded. Settling the path now rather than at the gate is the point — the alternative is
discovering it on the day.

**It must be visibly a replay.** Every replayed decision record carries a **replay flag and
its source fold**, so a replayed decision can never be mistaken for a live one in the log,
the dashboard or the report. Without that the two GATE 2 items would be demonstrated on
records indistinguishable from real ones, which would make the demonstration itself the kind
of thing this project exists to rule out.

**Today's live session is recorded too, band and all.** A recorded session in which the
system correctly declines to trade is worth having on its own terms, and the demonstration
shows both: the real system abstaining, and a replayed fold where it acts.

---

## 2026-08-18 — GB-33 is satisfied by work already done, and hands GB-41 one constraint

**Decision.** GB-33 is **closed without a new test**. §4.4's exactness properties are
parameterised over `model.ALL_FORECASTERS` and run over **1,000 random windows per
forecaster**; GB-30 added a second, independent pass in `tests/explain/test_channel.py`
that iterates the same registry through `explain.channel.attribute`. Both properties have
teeth — `MisreportingForecaster` breaks property 3 and `NonAdditiveForecaster` breaks
property 4, and each is asserted to be rejected.

**The registry is the integration point.** Adding `"fits": _fits` to `ALL_FORECASTERS` is
the whole of GB-41's wiring; no test file is edited, which is deliberate — a test a model
can edit is a test that model has judged itself with.

**One constraint this hands GB-41, stated so it is met rather than discovered.** The
contract asserts `tuple(attribution.per_channel) == batch.channels`, and §6.4 makes FITS
univariate. So **FITS must return every active channel, with `0.0` for the ones it does not
consume**, exactly as Persistence does. An attribution naming only the channels a model
reads would render as a dashboard with panels missing, and would quietly change what
"exact" means between arms.

---

## 2026-08-18 — GB-34/35/36: chrome and data are different colour families

**Decision.** Ben's ruling, implemented. **Orange is interface chrome only** — labels,
rules, borders, registration marks, active states, and the calibrated threshold, which is a
rule line rather than a measurement. The **blue spectral ramp is the data encoding**, dark
for slow and light for fast, as in the Hebrew proposal, the architecture report's three
figures and the vision script.

**Two consequences that had to be decided here.**

**1. Sign is carried by geometry and a glyph, never by hue.** The one data family available
is a *ramp*, and a ramp cannot encode a sign without inventing a second data colour, which
the ruling forbids. So a positive contribution is a **filled** bar right of the zero rule
and a negative one is a **hollow** bar left of it, and every signed figure is prefixed ▲/▼.
A test asserts no traffic-light pair appears anywhere. The side effect is worth having: the
chart is readable in greyscale, which is the test of whether colour was doing the work.

**2. The ramp is assigned by channel speed, not by config order.** A ramp encodes a
quantity; the quantity here is how far back a channel looks, so `ma_dist20` takes the
darkest entry and `close_logret` the lightest. Assigning it in config order would look the
same and encode nothing.

**The charts are hand-built SVG rather than a plotting library.** The design language is
dashed hairlines, numbered rulers and registration marks, which a chart library fights
rather than helps — and an SVG builder is a pure function returning a string, so the
threshold line, the ramp assignment and the bar geometry are unit-testable without a
browser. It also adds no dependency, which `DECISIONS.md` would otherwise have to justify.

**Three smaller rulings, each with its reason.**

- **`STOOD ASIDE` outranks every other status.** A session whose band cannot fire is not
  idle between decisions; it has decided in advance that it will not act. A status line
  reading `RUNNING` beside a flat book would leave a viewer waiting for a trade that cannot
  come.
- **A `never()` band is said on the chart, not omitted.** Dropping the threshold line makes
  a system that abstains look like one that has not acted yet, and those are different
  claims.
- **A Hebrew narrative gets an explicit `dir`, not `dir="auto"`.** `auto` resolves from the
  first strong character and every narrative GB-32 writes opens with a ticker, so `auto`
  would call a Hebrew sentence left-to-right and undo the isolation work that keeps a
  percentage on the correct side of its label.

**The reliability panel reads a measured file.** `smoke_offline --prepare-live` now writes
`reliability.json`, and the panel renders `NOT MEASURED` when it is absent rather than
hiding itself. A hardcoded track record is right on the day it is typed and wrong from then
on, which is the exact failure the panel exists to prevent. Measured today: direction
**0.5182** against an always-long bar of **0.5560**, beating it in **4 of 16 folds** — the
same figures as the anchor-0 column of the three-anchor table, arrived at independently.

---

## 2026-08-18 — GB-26: two rulings taken in the live loop, and what the first run found

### 1. Stale data — the symbol loses its entry and keeps its stop

**Decision.** A symbol whose last completed bar predates the last completed exchange
session is excluded from forecasting, ranking and entries for that cycle, **named in the
log and in the cycle report**, and **has its protective legs verified and re-armed exactly
as any other position's**.

**Reasoning.** *A stale window may never justify opening risk, and may never suspend the
management of risk already taken.* Those are two different questions and the previous
framing — "skip the symbol" — collapsed them. Steps 1 to 4 of the cycle run on the broker's
truth, which does not go stale, so protection is unaffected. What genuinely cannot be
computed is a **model-driven** exit, and the honest statement is that the stop and the
target are the risk control that does not depend on the model.

The exclusion is loud rather than silent because `top_k = 2` is taken across the universe:
dropping a symbol changes which two are chosen, and a reader has to be able to see that it
happened before reading the selection. The report carries what was ranked over and what was
dropped.

**No partial-universe abort.** A rule like "abort below three symbols" is a number nobody
measured, and the visible drop list already lets a reader discount the selection. If *every*
symbol is stale the cycle is a logged no-op after reconciliation — there is nothing to
forecast, and protection has already run.

### 2. The first cycle may reduce risk and may not add any

**Decision.** No entry is submitted on the first cycle after startup. Exits, reconciliation
and protection all run in full.

**Reasoning**, in order of weight:

1. **The protection policy's own rule 2** says re-arm every open position before entries,
   and the first cycle *is* that moment. Entering in the same cycle would submit an entry
   before the re-arm it is supposed to follow had been verified.
2. **Reconciliation has just rebuilt the book from a single observation** of the account. A
   quarantine decided microseconds earlier is a belief with no second witness, and sizing
   new risk against it trusts a one-sample view.
3. **A mid-session restart must not double an entry.** Broker order state is eventually
   consistent, and an entry the previous process submitted can be briefly absent from
   `get_orders`. One cycle of observation lets it appear. Re-entering a position we already
   hold is a failure the loop cannot undo; waiting one poll interval is not.

An exit is **not** deferred, for the reason GB-28 refuses to let one be crowded out: it is
an obligation on capital already committed, and delay is the cost.

### 3. Supporting choices, each with its reason

- **The session window is the exchange calendar's, not the configured clock's.** The Israel
  window says when the process is willing to run; `pandas_market_calendars` says whether
  there is a market. A holiday closes the loop and **a half-day closes it early** — the
  configured 23:00 Israel close would otherwise poll a shut exchange for three hours after a
  13:00 ET early close.
- **`ISRAEL_TZ` is a module constant, and this is a flagged rule-5 bend.** Rule 5 is config
  over constants. The justification is that `cfg.live.market_open_il`'s own field name fixes
  the zone, so a config key could only ever disagree with the field it describes.
- **Trades are emitted before reconciliation.** A filled stop makes the position vanish and
  `reconcile` correctly drops the holding — which is the only record of the entry basis.
  Emitting first builds the trade while the system still knows why it held the thing. This
  ordering is load-bearing and is stated at the top of the module.
- **`--dry-run` is a wrapping broker, not a skipped call.** `DryRunBroker` passes every read
  through to the real broker and refuses every write, so the dry run exercises `execute`,
  `protect` and the fill poll against real positions and real equity. A dry run that
  branched around them would test the scheduler and nothing else.
- **`smoke_offline --prepare-live DIR`** writes the checkpoint and the band, because
  `live_loop` may not import the harness. It trains on the **most recent complete fold** and
  calibrates on that fold's validation range — fitting up to yesterday and calibrating on
  the same rows would be the project's own leakage rule broken through the back door of
  "but it is live now".
- **A missing or stood-aside band is `Thresholds.never()`**, never a default. The session
  runs, records and explains, and trades nothing. Any other choice means inventing a
  threshold, and a made-up band is the one input that would make every decision in the
  session unexplainable.

### What the first real run found, before the open

`--prepare-live` trained on **fold 16 of 16** (train 2024-01-08 to 2025-12-29, validation
2026-01-06 to 2026-03-27) and **the fold stands aside**: the best candidate on the grid
scored a validation Sharpe of **−3.38** over 7 trades. So the session on 2026-08-18 will
forecast, explain and record, and **will not trade**. That is GB-20's ruling applied to a
live session rather than a backtest fold, and it is the correct outcome rather than a
failure to configure.

A forced dry-run cycle against the live SIP feed completed all ten steps: 610 bars per
symbol against a floor of 445, no stale symbol, the 0.01 AAPL probe correctly quarantined,
five decisions recorded and narrated, nothing submitted.

**One thing the narration surfaced that is worth GB-57's attention.** Four of the five
symbols tripped GB-32's offsetting sentence — the surviving fraction of the gross channel
view was **15.4%, 20.6%, 23.1% and 49.2%**, against the 50% line. The model's forecasts on
this data are largely **small differences of larger opposing channel contributions**, which
is exactly the property Ben's `cancellation` column in GB-49 was added to track, measured
here for the first time on live data.

---

## 2026-08-18 — The dashboard carries the model's track record, standing, in the panel

**Decision.** Ruled by Ben off the back of GB-32's output. GB-34's dashboard must show the
model's **measured reliability** persistently in the panel — direction accuracy, the
always-long bar it is measured against, the fold count behind it, and the date it was
measured — rather than caveating each narrated decision. Recorded in spec §9's GB-34 row
and its "Done when".

**Reasoning.** GB-32 renders *the model predicts a 2.02% rise over the next 4 trading
days*, which reads as a confident claim. This project's own measurement is that the same
model's directional calls are **indistinguishable from chance**, falling **3.8 to 5.0
points short of the always-long bar at every one of three fold-grid anchors**, beating that
bar in 4 or 5 folds of 16.

The narration is nevertheless right not to hedge every sentence. A caveat on every line is
noise, and a reader learns to skip noise — which would make the hedge worse than useless,
because it would look like disclosure while functioning as decoration. The fix is
structural rather than textual: put the track record in the **frame**, where it is standing,
unavoidable and identical whichever decision is on screen.

**The general principle, which belongs in GB-57 as well as in the code:** *a system that
explains a decision while hiding the decider's track record is doing the thing this project
exists to oppose.* An explanation makes a decision legible; it does not make it right, and
a legible wrong decision presented without its base rate is more persuasive than an opaque
one — which is a worse outcome than the black box the project set out to replace.

**Consequence.** GB-34's row and acceptance criterion carry it. The numbers it must display
are the three-anchor table recorded on this date, and they are a range across anchors rather
than a single figure, per the same day's grid-sensitivity ruling.

---

## 2026-08-18 — GB-26's two blockers closed, and GB-32's bidirectional rule

### 1. The feed is asserted, not commented (blocker, ruled by Ben)

**Decision.** `data/live.py` names `LIVE_FEED = DataFeed.SIP` and
`LIVE_ADJUSTMENT = Adjustment.ALL` as constants, `_fetch_bars` uses them, `data_source()`
renders them, and `load_live_bars` logs that line on **every fetch** — not only at session
start, so every set of bars in the run log carries the tape it came from. Three tests: the
constants are what they should be, the **request object handed to the SDK actually carries
them**, and the line reaches the log.

**Reasoning.** The module docstring already said a silent downgrade to IEX "would change
the numbers without changing the schema", and GB-30's sweep priced it: **up to 193 bps on
the low, 173 on the open, 90 on the close**, against a 1 bp tolerance and a 2 bps modelled
slippage — roughly 190x the tolerance. Every schema test in the suite would still pass.
A claim with a measured cost that large and no backstop is the definition of something to
fix before the live loop, and asserting the constant is not enough on its own: a constant
can exist and go unused, so the assertion that matters is on the **request**.

### 2. The lookback and the guard are the caller's floor, not `input_len` (blocker)

**Decision.** `load_live_bars`'s signature changes to
`load_live_bars(symbols, min_bars, *, requirement="", lookback_days=None)`. `cfg` is gone —
it was only ever read for `window.input_len`, which is the wrong number. The default
lookback is `lookback_days_for(min_bars)`, and the guard raises below `min_bars`.

**Reasoning.** Measured on 2026-08-18: the default resolved to `input_len × 2` = 240
calendar days, which returned **163 bars** against a `min_history_bars` of **445**, and the
guard raised only below `input_len` = 120 — so 163 passed. The failure did not corrupt
anything (`rsi` is NaN through its whole 325-bar warm-up, so `build_feature_frame` returned
an empty frame) but it surfaced downstream during market hours with nothing naming the
cause.

**Why `min_bars` is required rather than defaulted.** `data` sits **below** `features` in
§3.1, so `live.py` cannot call `min_history_bars` and cannot be trusted to restate it —
restating it as `input_len` is exactly how this was wrong. Making the caller name it turns
a layer constraint into an explicit argument. For the same reason `features.builder` gains
`history_requirement(cfg)` and `deepest_warmup_channel(cfg)`: the refusal message must name
the channel that sets the floor, and only the feature layer knows it, so that layer writes
the sentence and the caller carries it down.

**The message now names all four things** Ben asked for — bars received, calendar days
requested, the floor, and the channel behind it — plus why the failure would otherwise have
been quiet. A test asserts each part, and another counts **real NYSE sessions** in the
default window sliding across a decade and asserts the **worst** case still clears the
floor, so the calendar-to-sessions margin is measured rather than assumed.

### 3. Bidirectional text in GB-32 — decided here and reported

**Decision.** Three rules, all asserted:

1. **Direction is metadata.** `narrate` returns a `Narrative` carrying `direction`, and the
   renderer sets `dir="rtl"`. Direction is never inferred from the first strong character —
   which would fail immediately, because every sentence here opens with a ticker.
2. **Every LTR run is wrapped in U+2066 / U+2069 isolates.** Not LRE/PDF embeddings and not
   bare LRM marks. An embedding's contents can still influence the resolved level of the
   surrounding text and an LRM only nudges the one boundary it sits at, so both require the
   author to reason about every adjacency and both fail silently when a new adjacency
   appears. An isolate cannot influence its surroundings at all, which is the property
   being bought.
3. **An atom is the whole unit, sign and percent included.** `-1.8%` is one atom, so the
   minus can never separate from the digits; `rsi14 +57.1%` is one atom, so a channel
   cannot be parted from its share. The list of atoms then reads right-to-left while each
   atom reads left-to-right, which is correct Hebrew typography rather than a compromise.

English gets none of this — isolates are invisible but real and would end up in the
decision log — and a test asserts their absence there.

**Reasoning.** The failure this prevents is a percentage rendering on the wrong side of its
label, which bit this project once already in SVG. The tests do not check the rendering,
which would need a shaping engine; they check the two structural properties that make the
rendering correct: **no Latin or numeric run appears outside an isolate**, and **every
isolate is balanced and never nested**. Both are decidable from the string.

**Consequence.** Spec §9's GB-26, GB-32 and GB-57 rows carry these. The two blocker items
in `PROGRESS.md`'s open questions are closed.

---

## 2026-08-18 — GB-30: the decomposition lives on the schema, and shares are of the gross

**Three decisions, one ruled by Ben and two taken here and reported.**

### 1. Where the one decomposition lives — ruled by Ben

GB-30's brief was "move DLinear's inline explain logic into `explain/channel.py` and have
the model delegate to it, so there is one decomposition rather than one per model". Taken
literally that breaks spec §3.1: `explain` sits **above** `model`, so the delegation is an
upward import and a module cycle, and it would cost an `ignore_imports` exception in the
layers contract.

**Ruling: the summation becomes `Attribution.from_terms`, a classmethod on the schema
itself (L0).** Both the model's `explain` and `explain.channel.attribute` call it. No
exception, no cycle, and `lint-imports` still reports 2 kept / 0 broken. Precedent for
putting real work on a frozen schema is `WindowBatch.concat` and `FitProvenance.from_batch`;
no field changed, so §4.2 is untouched.

The version that costs nothing is also the stronger one. The exactness property was declared
in `Attribution`'s docstring and enforced only by a test. It is now enforced **at
construction**, so every attribution in the system is built by the one function that refuses
a residual — and that refusal names the two causes worth checking first, an intercept and a
reversible normalisation added back outside the per-channel terms.

Each layer keeps a real job: the model says *which* terms exist, which is architecture; the
contract sums them and holds the invariant; `channel.py` is the face GB-32 and GB-53 call
and **re-checks the reported total against `predict` itself**. That last check is not
redundant. `from_terms` can only prove the parts sum to the total *the model reported*; a
model that reports a total it did not forecast is internally consistent and describes
nothing, and only a layer that asks `predict` directly can see it.

**One change that follows, and it removes a tautology.** `forecast_total` used to be
`sum(per_channel.values())`, which made spec §4.4's property 4 true by construction —
the assertion could not fail. It now comes from `predict`, so properties 3 and 4 are two
independent checks. The residue this introduces is the float32 cast on `predict`'s output
and nothing else: **5.005e-08 worst over 1,000 random windows, a 200× margin** under the
1e-5 tolerance. `tests/model/test_ltsf.py` had asserted the old identity at `abs=1e-12` and
now bounds the residual by `H × ulp(max|path|)`, which is a measurement rather than a
loosened tolerance.

### 2. Percentage shares when contributions have opposite signs — decided here

**Shares are normalised by the gross:** `share[c] = c / Σ|c|`.

Ben's example — `+0.08`, `−0.06`, forecast `+0.02` — reads **+57%** and **−43%** rather than
400% and −300%. Magnitudes sum to exactly 100%, every share lies in [−100%, +100%], and the
sign survives.

Percent-of-net is rejected on two grounds, and the second is the decisive one. It is
unbounded, so a reader meets 400% and concludes the explanation is broken — on that
denominator they are right to. And it is **discontinuous**: it blows up and flips sign as
the forecast crosses zero, which on daily log returns is exactly where this project renders
most of its explanations.

The cancellation the shares no longer show is not discarded — it is reported directly, as
`cancellation = |Σc| / Σ|c|` = **0.143** for that window. GB-32 can then write *the channels
largely cancelled: 14% of the gross view survived into the forecast*, which is the honest
sentence and the one percent-of-net cannot express at all. `per_channel` remains the exact
quantity that sums to the forecast; shares are a rendering aid and the panel shows both.

### 3. Attribution of a forecast of exactly zero — decided here

**Defined, total, and never NaN.** The gross denominator separates two cases that
percent-of-net conflates:

- **Every contribution exactly zero** — persistence, on every window. The gross is zero too
  and every share is `0.0`. That preserves GB-11's ruling that the baseline reports each
  active channel with a zero rather than an empty mapping: a renderer then shows *no channel
  drove this, because nothing was predicted*, rather than a blank panel indistinguishable
  from a bug.
- **Contributions that cancel to a zero forecast** — a real model can do this. The gross is
  non-zero, so the shares are well-defined and non-zero, and it is `cancellation` that reads
  0.0.

So the ratio is undefined only in the single case where "nothing" is the true answer, and
there the answer is zero rather than an error. **That is the strongest argument for the
gross denominator**, and it is why the zero case needs no special-casing anywhere else in
the system.

**Consequence.** `PersistenceForecaster.explain` no longer writes `dict.fromkeys(channels,
0.0)` by hand: it reports **no terms at all** for each channel, and the zeros fall out of
the same arithmetic every other model goes through. A baseline that constructs its own
answer is a baseline the exactness check never actually exercised.

**One thing this task added that was not asked for, and the reason.** The ban on SHAP, LIME
and captum is written in spec §4.2, in three module docstrings and in `CLAUDE.md`, and was
enforced by nothing. A syntax-tree test now asserts that no module under `explain/` imports
any of them. It is the same treatment `signal.py`'s no-numeric-literal rule gets, and it
matters more here: a perturbation library would not fail a test, it would produce plausible
numbers that do not sum to the forecast — the one defect this project's central claim exists
to rule out.

---

## 2026-08-18 — Every headline claim is measured at three fold-grid anchors

**Decision.** Ruled by Ben, as a standing requirement rather than a one-off check. GB-49's
study runs the whole grid at **three fold-grid anchors**, offset from one another by roughly
one third of `step_months` — 21 trading sessions at `step_months: 3`. `results.csv` carries
the anchor as a column. GB-52's report shows, for every headline claim, whether it holds at
all three. GB-57 carries a required methods subsection on the protocol and what it found.

**Reasoning.** The rule is written *from* a failure, not in anticipation of one. GB-27b's
warm-up unification moved the fold boundaries by about a week and the timing-versus-market
correlation went from `r = -0.47` to `r = -0.0025`. Nothing about the single-anchor
measurement said it might. If a one-week shift can erase a finding, other findings may be
equally fragile and a single-grid study has no way to tell which — the sensitivity is not a
robustness garnish, it is the only instrument that separates a finding from a property of
the grid. Most work in this area reports one grid, so reporting the sensitivity is a
methodological point the report can make rather than a defensive footnote.

**Cost.** 16 folds run in about 31 seconds, so three anchors is about 93. The two-claim
re-check below took 3 minutes 40, because it runs buy-and-hold on every fold as well.

**Consequence.** Spec §9's GB-49, GB-52 and GB-57 rows carry it. **A claim shown at one
anchor is not reportable.**

---

## 2026-08-18 — The two claims already relied on, re-checked at three anchors: one holds, one splits

**Decision.** Recorded, not acted on. No change to the model, the band, the sizer or the
config. Anchors are offsets applied to the index handed to `make_folds`, so anchor `+21` is
the same data with the fold boundaries moved one month later.

**Claim B — direction accuracy against the always-long bar. Holds at all three.**

| anchor | first test | DLinear | always-long | gap | t | beats the bar |
|---|---|---|---|---|---|---|
| +0 | 2022-07-06 | 0.5182 | 0.5560 | −0.0377 | −1.53 | 4/16 |
| +21 sessions | 2022-08-08 | 0.5027 | 0.5523 | −0.0496 | −2.55 | 5/16 |
| +42 sessions | 2022-06-06 | 0.5236 | 0.5654 | −0.0418 | −2.39 | 4/16 |

Same sign, a gap of 3.8 to 5.0 points, the bar beaten in 4 or 5 folds of 16, at every
anchor. This is the most grid-stable claim in the project and it is a **negative** one: the
model does not beat always calling up. Its stability is exactly why it is reportable.

**Claim A — β tracks measured exposure. The identity holds; the significance does not.**

| anchor | measured exposure | β | β t | β 95% CI | α per fold | α t | R² | stood aside |
|---|---|---|---|---|---|---|---|---|
| +0 | 0.0892 | 0.0919 | 3.29 | [+0.032, +0.152] | −0.31% | −0.65 | 0.436 | 3/16 |
| +21 sessions | 0.0934 | 0.0507 | 1.51 | [−0.021, +0.122] | −0.21% | −0.44 | 0.141 | 3/16 |
| +42 sessions | 0.1390 | 0.0848 | 2.28 | [+0.005, +0.165] | +0.11% | +0.20 | 0.271 | 1/16 |

Three readings, and they must not be collapsed into one:

1. **The measured exposure lies inside β's 95% interval at every anchor.** The identity
   "β is the exposure" survives.
2. **α is never distinguishable from zero** — |t| ≤ 0.65 at every anchor, and it changes
   sign between them. *The exposure explains the returns and there is no alpha* survives.
3. **β's own significance does not survive.** At +21 its interval includes zero (t = 1.51).
   The previous entry's "with β significant (t = 3.29)" was true of one anchor and is
   **amended here**: it holds at two anchors of three.

The point estimates are not stable either — exposure 0.089 to 0.139, β 0.051 to 0.092, R²
0.14 to 0.44 — so GB-57 reports each as a range across anchors, never as a single number.

**One reconstruction check worth recording.** Average gross exposure is rebuilt bar by bar
from the trade log and each symbol's own closes, holding a position from `entry_time` to
`exit_time` inclusive. At anchor +0 that reproduces the **0.0892** already in the record to
four decimals, which is what makes the other two anchors comparable to it. The exclusive
variant — dropping the exit bar's close — gives 0.0720, so the definition is not a detail:
it moves the number by 19%.

**Consequence.** Spec §9's GB-57 row carries both tables and replaces its pre-GB-27b
direction figures (bar 0.5625, DLinear 0.5071, 5 of 16), which were measured on a fold grid
that no longer exists.

---

## 2026-08-18 — GB-7's deferral is withdrawn: SIP works, and the re-sweep confirms the claim

**Decision.** The GB-7 audit item recorded as "deferred, needs live SIP data this account no
longer serves" is **withdrawn as wrongly premised**, and the claim it deferred is re-swept
and confirmed. I recorded that this account "returns 403 on recent SIP". That is not what
the account does, and the imprecision produced a deferral that was never necessary.

**What the account actually refuses.** Recent-*data* endpoints, not the SIP feed:

| request | result |
|---|---|
| daily bars, SIP, 45 days | **served** — 31 sessions to 2026-08-17, the most recent completed bar |
| daily bars, SIP, 400 days | **served** — 273 sessions per symbol |
| minute bars, SIP, last 2h | served (empty — the market is shut) |
| `get_stock_latest_bar`, SIP | **refused** — `subscription does not permit querying recent SIP data` |
| `get_stock_latest_quote`, SIP | **refused** — same message |

`data/live.py` requests **daily bars over completed sessions**, which is the served case.

**Answering the three questions Ben posed before GB-26.**

1. **Which feed does the live client resolve to for the most recent completed daily bar?**
   SIP. `_fetch_bars` passes `feed=DataFeed.SIP` as a hardcoded enum with no fallback path,
   and the request succeeds today.
2. **Is it the same feed the training data was compared against in GB-7?** Yes. So the
   parity question is not "does an unvalidated feed reach the live loop" — it does not.
3. **Does it agree with the training source?** Re-swept over **273 sessions × 5 symbols =
   1,365 bar-comparisons**, against GB-7's single 163-bar window. SIP against the cached
   yfinance training source, worst cell of the whole sweep: **open 0.51, high 0.46, low
   0.56, close 0.30 bps.** GB-7's 1 bp tolerance **holds on the sweep.** Volume's worst is
   **213.6 bps**, so GB-7's stated 34–111 bps range widens to **34–214**.

**What IEX would cost, since that was the feared path.** Same 273 sessions, IEX against the
same training source: **open up to 173 bps (GOOGL), low up to 193 bps (MSFT), close up to 90
bps (AMZN)**, volume ~9,900 bps. Against a 1 bp tolerance and a 2 bps modelled slippage, a
silent IEX downgrade would be worth up to **190× the price tolerance**. It cannot happen
silently — the feed is a constant and losing entitlement raises — but **nothing in the suite
asserts that constant**, which is the one gap this leaves open.

**A second gap, found while establishing the first, and it is the shape Ben predicted.**
`load_live_bars` derives its default lookback as `input_len × 2 = 240` calendar days. That
returns **163 trading bars** against a `min_history_bars` of **445** — 282 short — and the
function's own guard raises only below `input_len` (120), so **163 passes it**. About 645
calendar days are needed. The failure is *not* a silent divergence: `indicators.rsi` emits
NaN for its whole 325-bar warm-up, so `build_feature_frame` returns an **empty frame**
rather than contaminated values — the NaN warm-up is what protects parity here. But nothing
on that path names the cause, and a caller trusting the default gets an empty frame with no
explanation.

**Consequence.** `data/live.py`'s docstring carries the re-swept table and the entitlement
boundary. Spec §9's GB-26 row carries the lookback measurement. Two items go to Ben for a
ruling rather than being acted on here: **(a)** a test asserting the feed constant, since "a
silent downgrade would change the numbers without changing the schema" is a claim with no
backstop; **(b)** restating `load_live_bars`'s default and guard in terms of
`min_history_bars`, which is GB-26 scope.

---

## 2026-08-17 — GB-20: the direction reference is always-long, not the per-fold majority class

**Decision.** `metrics.directional_base_rate` — `max(up_rate, 1 − up_rate)`, added hours
earlier by GB-19 — is replaced by `metrics.always_long_accuracy`, which returns the fold's
realised **up rate**. The class is fixed from the training split; only the rate is measured
on test. `direction_reference` stays in the summary table and now carries a number that is
by definition the fold's up rate, so the bar is visible moving from fold to fold.

**Reasoning.** The two numbers GB-19 reported did not reconcile: a mean up rate of 0.5625
against a "chance level" of 0.5865. The gap is the whole error — `max(up, 1 − up)` computed
per fold and then averaged flips to "always short" on the folds whose test period fell, and
selects that class from the **test** data being scored. No strategy knows in advance whether
the coming quarter is up or down, and §2.1 is long-only, so "always short" is not in the
action space at all. The bar was therefore both unachievable and set with test-period
knowledge. It erred *against* the model — it made every arm look worse — and that does not
make it admissible: a bar built from information the system could not have had is
inadmissible in either direction. Always-long is the one constant strategy this project
could have run, and the up rate exceeds 0.5 in **16 of 16 training splits** (0.506–0.588),
so the class choice needs no test data.

**Consequence.** The direction bar falls from 0.5865 to 0.5625. DLinear's 0.4751 now beats
it in **4 folds of 16** rather than 1 — still below the bar overall, by 8.7 points on
average. Spec §7.2 and the GB-19, GB-20 and GB-57 rows in §9 carry the correction with both
rejected references named. A test on a synthetic fold whose test period fell pins the
difference: always-long scores 3/9 there and the majority class would have reported 6/9.

---

## 2026-08-17 — GB-20 diagnostic: the direction shortfall is a down-bias, not an inverted signal

**Decision.** Recorded as a finding, not acted on. No sign-flip hunt, no inversion of the
forecast, no change to the target alignment or the trend decomposition.

**Reasoning.** DLinear scoring below the always-long bar invites the reading that a
consistently wrong classifier carries information. Measured over all 16 folds, under **both**
batch settings — the diagnostic was first run on the full-batch model and re-run on the
mini-batch model the re-check restored, because a finding about a configuration nobody uses
is not a finding:

| | **batch 64 (configured)** | full batch (measured, then reverted) |
|---|---|---|
| calls up | **0.5385** (0.330–0.800) | 0.4569 (0.144–0.590) |
| market went up | 0.5625 | 0.5625 |
| information-free at that call rate | **0.5016** | 0.4893 |
| direction accuracy | **0.5071** | 0.4751 |
| residual vs information-free | **+0.54 pts**, above it in 10/16 | −1.42 pts, below it in 10/16 |
| inverted forecast | **0.4925** — worse than as-is | 0.5245 — better than as-is, still under the bar |
| beats always-long | 5/16 | 4/16 |

1. **The shortfall is a down-bias, not error.** The model calls up less often than the
   market rose under both settings, and an information-free model calling up at the same
   rate — `q·p + (1−q)·(1−p)` — scores within half a point of what the model actually
   scores. Under the configured setting the model scores **above** its own information-free
   counterfactual, so there is nothing anti-informative left to explain.
2. **No sign is flipped.** Inverting the configured model makes it **worse** (0.4925 against
   0.5071) and reaches the bar in fewer folds. A sign error in the target alignment or the
   trend subtraction would show up as inversion beating the bar consistently. It does not,
   under either setting.
3. **It is uniform, not concentrated.** Per symbol the configured model runs 0.4952 (AAPL)
   to 0.5261 (AMZN), a 3.1-point spread straddling 0.50; the per-fold shortfall against the
   bar is −5.6 points with sd 8.8. Nothing points at one symbol or one period.

**Consequence.** The claim "0.4751 is below the worst achievable by pure noise" does not
survive measurement even for the model it was made about: the noise floor is not 0.4875 but
0.4893 at that model's own call rate, and 0.4375 at the extreme of `q`. For the configured
model the question does not arise — it sits marginally above its own floor. GB-57 must
report the **call rate beside the direction accuracy**, because without it a reader cannot
tell a biased model from an anti-informative one, and must quote the configured model's
figures with any full-batch number labelled as the reverted variant.

What is left is a model that is **information-free on direction and cannot represent
drift** — it scores its own no-information level while always-long scores 5.6 points more.
That is the open question for GB-41, and the entry above gives it a hypothesis.

---

## 2026-08-17 — GB-20: thresholds are calibrated by the backtester, one layer above the engine

**Decision.** `engine/signal.py` holds the decision logic and **no numbers at all**;
`backtest/calibrate.py` holds `calibrate_thresholds`, which scores a coarse grid by running
the real backtester on the fold's validation split. Spec §3.4 gains the module.

**Reasoning.** GB-20 asked for `calibrate_thresholds` inside `signal.py`, and the ruling
that it must run the actual backtester rather than a proxy makes that impossible: `backtest`
sits above `engine` in the layer contract, and `live_loop` imports `signal`, so the function
would drag the validation harness into the live path and break both contracts at once.
Injecting the backtester as a callable would hide the same dependency behind an argument.
Splitting it puts each half where its dependencies already are — and the live loop then
carries a band that was chosen offline, which is what it will actually do.

The grid is expressed as **quantiles of the validation split's own forecast distribution**
rather than as absolute return levels. A band of 0.004 means nothing without knowing what
that fold's model produced; a grid of absolute levels would need rewriting for every
horizon, universe and channel set, and a band tuned to DLinear's magnitudes would silently
disable FITS. A test scales every forecast by ten and gets the same trades.

**Consequence.** Five entry quantiles × three ceilings = 15 backtests per fold, ~0.1s each.
`Thresholds` is a frozen contract-shaped value passed into `decide`, so the same band feeds
the backtest and the live loop. The action vocabulary (`enter_long` / `hold` / `exit`) moved
into `signal.py` and `backtest/engine.py` now imports it rather than keeping its own copy.

---

## 2026-08-17 — GB-20: a fold with no profitable band stands aside rather than trading the least-bad one

**Decision.** When no candidate on the grid earns a **positive** validation Sharpe, the fold
returns `Thresholds.never()`: no trades, a flat equity curve, `stood_aside=True`, and the
best candidate's Sharpe kept on the record. A NaN Sharpe — fewer than five validation trades
— counts as not positive.

**Reasoning.** The alternative is to trade a rule that validation had just said loses money,
and to report the **maximum of fifteen losing candidates selected on the same data that
scored them**, which is an overfitting procedure with a positive-looking number at the end.
Standing aside costs the study nothing it is entitled to: the forecast metrics are
unaffected, the fold's return is a true zero rather than an estimated loss, and "validation
rejected every band" is itself a result. The interlock with `MIN_TRADES_FOR_SHARPE` does
real work here — a band selective enough to trade twice cannot win a fold on a ratio
computed from two observations.

**Consequence.** Measured on the real universe with DLinear: **3 folds of 16 stood aside
under full batch, 1 of 16 under mini-batch**. GB-57 must report the count, because a study
that silently trades nothing on a fifth of its folds and one that trades everywhere are
different studies. Spec §9's GB-20 row carries the ruling.

---

## 2026-08-18 — GB-29: one `Trade` type, and it moves to `contracts/schemas.py`

**Decision.** A live trade and a backtest trade are **the same type**, not two types with a
shared shape. `Trade` moves from `backtest/engine.py` to `contracts/schemas.py` (a §4.2
addition) and gains two optional fields, `entry_order_id` and `exit_order_id`, which are
`None` for every backtest trade.

**Reasoning, and the layer contract settles it before the design argument does.** GB-19's
metrics must read a live trade log **without a translation layer**. That rules out two
types: every metric would have to accept both, the duck typing would be untested until one
side grew a field, and "structurally identical" would be a property a reviewer checks by eye
rather than one the type system holds.

Given one type, where does it live? Not in `backtest/`: the live path may not import the
validation harness, and `records.py` — which holds the live trade log — is exactly the module
that would be tempted to. So the shared type belongs in `contracts`, the layer both sides
already depend on. **The import contract decided the design**, which is the point of having
one; `glassbox.records` is now named in the forbidden-import contract so the temptation is
mechanically unavailable.

The broker order IDs are the one thing live carries that a backtest cannot: a simulated fill
has no order to point at. They are **optional fields rather than a second type** — GB-19
never reads them, GB-32's replay needs them to tie a trade to what the broker actually did,
and a live trade that could not name its own orders would be unreconcilable.

**Consequence.** `engine.Trade is contracts.Trade` is asserted, and so is the end-to-end
requirement: GB-19's `hit_rate`, `average_trade` and `total_return` run over a log built by
`records.emit_trades` with no adapter. One duplication is accepted and guarded: the four exit
reasons are repeated in `records.py` because it may not import `backtest`, and a test asserts
they equal the backtester's own constants, so a rename there fails here rather than silently
producing a log GB-19 reads differently.

**The measurement this unlocks.** `costs` come from the fill prices the broker reports, so
`realised_slippage_bps` compares what the frictions actually cost against the 6.0 bps the
study charges by construction. That is the one comparison the live system can make and the
backtest cannot, and GB-57 reports it.

---

## 2026-08-18 — GB-28: ranking breaks ties alphabetically, and that is the whole design

**Decision.** `rank_signals` orders by descending `trend_strength` and breaks ties by symbol
name, then takes `signal.top_k`. Only `enter_long` competes.

**Reasoning.** Python's sort is stable, so a tie without a second key resolves by the
caller's insertion order — which in the live loop is dictionary order, which is the order
symbols came back from a broker call. That is a real source of run-to-run difference: two
identical days would select different names and GB-32's replay would not reproduce. One sort
key removes it. The negation of the strength rather than `reverse=True`, because reversing
would also reverse the tie-break and the alphabetical order would silently run Z to A.

Exits and holds are not candidates: **an exit is an obligation on capital already
committed** and must never be crowded out by a better opportunity elsewhere.

A non-finite `trend_strength` is refused rather than sorted — NaN compares false against
everything, so ordering it is silently order-dependent, which is the exact class of bug the
tie-break exists to remove.

---

## 2026-08-18 — GB-25's decomposition re-swept: the correlation does not reproduce at all

**Decision.** The `r = -0.47` timing-versus-market correlation is **withdrawn**, and with it
the "independent corroboration" framing GB-57 was told to carry. Ben withdrew his own praise
of it on the triage rule — n = 16, 95% CI [-0.783, +0.034], includes zero — and the
re-measurement is worse than that: **on the fold grid as it now stands, r = -0.0025**, CI
[-0.498, +0.494]. It did not merely fail significance. It vanished.

**Reasoning.** GB-27b's warm-up unification shifted the fold grid by about a week. Every
number in the decomposition was re-measured on the new grid, with intervals:

| | old grid | **new grid** | 95% CI | t |
|---|---|---|---|---|
| average gross exposure | 0.1600 | **0.0892** | — | — |
| earned by exposure | +92.6 bps | **+98.1 bps** | [−11.2, +207.4] | +1.91 |
| given back to timing | −37.0 bps | **−51.8 bps** | [−113.5, +9.9] | −1.79 |
| paid in costs | −10.7 bps | **−6.1 bps** | [−9.1, −3.1] | −4.29 |
| = actual | +44.8 bps | **+40.2 bps** | [−74.5, +154.9] | +0.75 |
| β on market return | 0.141 | **0.0919** | [+0.032, +0.152] | +3.29 |
| R² | 0.65 | **0.436** | — | — |
| α per fold | −0.59% | **−0.31%** | [−1.32%, +0.70%] | −0.65 |
| **r(timing, market)** | **−0.47** | **−0.0025** | **[−0.498, +0.494]** | −0.01 |

**What survives, and it is the part that matters.** β ≈ average exposure on **both** grids —
0.141 against 0.160, then 0.092 against 0.089 — with β significant (t = 3.29) and α not
(t = −0.65). **Amended 2026-08-18:** re-checked at three fold-grid anchors, β ≈ exposure
and “α is not distinguishable from zero” hold at all three, but β's own significance holds
at only two of three (t = 1.51 at the +21 anchor). See the entry of that date. *The exposure explains the returns and there is no alpha* is robust to the grid
moving. So is the shape of the split: earned by being in the market, most of it given back
to timing and friction, with the timing term never distinguishable from zero.

**What does not survive is anything that rested on one 16-point correlation.** A one-week
shift in fold boundaries took r from −0.47 to −0.0025. Every level moved too — exposure
nearly halved, because the strategy now stands aside on 3 folds of 16 rather than 1.

**Consequence.** GB-57 reports the split with intervals and **must not** claim the
forecast-side down-bias is corroborated by the equity curve: the equity-side half of that
pair is gone. What remains is GB-20's forecast measurement alone — the model calls up 53.85%
of the time against a 56.25% up rate — which is a measurement on its own terms and needs no
second witness to be worth reporting.

**The audit item this creates.** Ben's own claim entered the record as "the strongest
finding in the project" and is now withdrawn twice over. It is in the single-point audit
list as an item introduced by review rather than by implementation, because the report's
claims are not only the implementer's to check.

---

## 2026-08-18 — GB-13's initialisation study, re-swept: the ranking survives, the reason changes

**Decision.** Zero initialisation stands. The **single-fold table in `ltsf._initial_weights`
and in GB-57 is replaced by the swept one**, because the swept result does not say what the
single fold said.

**Reasoning.** The original table was **walk-forward fold 1, one symbol**, under the
full-batch variant later reverted — the same shape as the parity floor, a comparison that
looked decisive at one point. Re-run through `tests/sweep.py` at **3 arms x 16 folds x 5
symbols = 80 cells**, under the configured model (`batch_size: 64`):

| axis | paper `1/sqrt(L)` | fan-in `1/sqrt(2CL)` | **zeros** |
|---|---|---|---|
| MAE vs persistence (lower better) | 2.6902 — **0/80** wins | **2.1947 — 46/80** | 2.1902 — 34/80 |
| direction accuracy (higher better) | 0.4750 — 22/80 | 0.4703 — 17/80 | **0.5182 — 41/80** |
| largest contribution (lower better) | 1.1110 — 0/80 | 0.4501 — 0/80 | **0.0653 — 80/80** |

**What survives.** That the paper's `1/sqrt(L)` is wrong for this architecture: it wins
**zero cells on MAE and zero on legibility**, and zeros beats it in 75 of 80 cells on MAE.
The fan-in argument — that summing `C x 2` maps makes the true fan-in 1200, not 120 — is
confirmed by the fan-in arm closing almost the whole MAE gap.

**What does not survive is the MAE half of the ranking.** Zeros and fan-in are a tie:
means 2.1902 against 2.1947, a difference of 0.2%, and **fan-in wins more cells, 46 to 34**.
The original table's 1.94x against 1.96x was always that tie; one fold made it look like an
order. Anyone quoting "zeros has the best MAE" is quoting a coin flip.

**What is stronger than the original claimed.** Legibility is **unanimous, 80 of 80**, and
it is the axis this project exists for: the largest single-channel contribution is 0.065
against 1.111, seventeen times smaller. The single-fold legibility numbers (1.39 / 0.43 /
0.058) reproduce almost exactly as swept medians (1.02 / 0.41 / 0.063) — that measurement
was sound; the accuracy ones were not.

**The levels do not reproduce at all.** Swept MAE is 2.69 / 2.19 / 2.19 against the fold's
4.00 / 1.96 / 1.94, and swept direction 0.475 / 0.470 / 0.518 against 0.344 / 0.328 / 0.557.

**Consequence.** GB-57 reports the swept table, and states the decision as it actually
stands: **zeros is chosen on direction and on legibility, not on MAE**, where it ties with a
correctly-scaled random initialisation. That is a more interesting finding than the original
— a published default is wrong for a summed architecture, correcting its fan-in recovers the
accuracy, and only initialising at zero also buys an attribution a human can read.

---

## 2026-08-18 — `tests/sweep.py`: the harness that was missing, and what it refuses

**Decision.** A sweep harness beside `causality.py`, with two properties that are refusals
rather than conventions:

1. **A `SweepResult` cannot be used as a boolean.** `bool(result)` raises `TypeError` and
   names the attributes to read instead. A caller who wants a verdict must say which
   fraction satisfies them.
2. **A sweep of one cell is refused**, with an error that explains why by citing the floor.

`sweep()` covers *"X holds"*; `contest()` covers *"A beats B"*, reporting per-cell wins as
well as means.

**Reasoning.** The audit after the parity-floor correction found the mechanism rather than
a list of suspects: **every claim in this project that survived scrutiny came from a task
that happened to have a harness to sweep with, and every claim that did not came from a
task that measured by hand on whatever was in front of it.** That is a statement about
tooling, not about care, so the fix is a tool. `32 of 125` is the number that would have
caught the floor; `it passes` is the number that did not — hence the boolean refusal, which
makes the failure mode unavailable rather than merely discouraged.

`contest` counts **wins by cell** as well as means because an arm that wins narrowly nine
times and loses catastrophically once is a different animal from one that wins on average,
and a mean cannot tell them apart.

**Consequence.** Held to the same standard as `causality.py`: the decisive test replays the
352-bar measurement on real data, and synthetic partial sweeps prove the harness reports
fractions strictly between none and all. Every future claim of either shape is one call.

---

## 2026-08-18 — GB-27: `RSI_WARMUP` is deleted and unified with the parity warm-up

**Decision.** `indicators.RSI_SEED_TOLERANCE` moves from **1e-2 to 1e-10**, so `RSI_WARMUP`
becomes **325**, and `builder.PARITY_WARMUP["rsi14"]` **imports it** rather than repeating a
number.

**Reasoning.** The 77-row warm-up had the same defect just corrected in the parity floor —
it bounded the seed's **weight** below 1e-2 rather than `weight × seed_difference` — and it
answered the same question in a different unit. "The value no longer remembers its seed"
and "the value is byte-identical to what training computed" are one question. Two constants
answering it is how they drift; repairing both derivations separately would have kept two
things to keep in step.

**Cost, measured rather than assumed — and it is not quite zero.**

| | before | after |
|---|---|---|
| feature rows per symbol | 2591 | **2343** (−248, exactly 325 − 77) |
| first feature row | 2016-04-25 | 2017-04-19 |
| folds | 16 | 16 |
| fold 1 training starts | 2020-04-13 | **2020-04-06** |
| windows per symbol | 2468 | 2220 |

Ben's check was that no kept fold reads a discarded bar, and that holds: everything dropped
is 2016–2017 and the earliest kept fold begins training in April 2020. But `make_folds`
anchors its calendar grid on `index[input_len - 1]` of the **feature frame**, so a frame
that starts later moves the anchor and the whole month grid shifts by about a week. Fold
counts are unchanged and no fold's *train* size moved materially, but individual val/test
window counts move by one or two (fold 3's test went 57 → 58).

**So every previously measured number shifts slightly**, because the folds are not the same
folds. The headline figures were re-measured after the change rather than carried over; see
the GB-27 PROGRESS row.

**A consequence worth more than the tidiness.** With the trim at 325 rows, a tail shorter
than `min_history_bars` no longer assembles a window at all — `build_feature_frame` leaves
fewer rows than `input_len` and the builder **refuses**. The silent below-floor divergence
that GB-27 was written to catch is now **unreachable through the public path**: a caller
asking for too little history gets an error, not a subtly wrong window.

That costs one thing, stated plainly: the 414-versus-445 measurement (120 of 125) can no
longer be reproduced through the public path, because 414 now refuses. It stands as a
recorded measurement justifying the 1e-10 target rather than as a live assertion, and the
below-floor tests assert the refusal instead. A loud failure that cannot be observed
degrading is a better trade than a silent one that can.

---

## 2026-08-18 — GB-27: the parity floor was marginal by construction; 352 becomes 445

**Decision.** `PARITY_WARMUP["rsi14"]` becomes **325**, so `min_history_bars` returns
**445** for `C0_base`. Derived, not swept:

```
(13/14)^k < 1e-10  ->  k = ceil(ln(1e-10) / ln(13/14)) = 311
warm-up = 14 (the seed window) + 311 = 325
floor   = input_len 120 + 325 = 445
```

**Reasoning.** GB-9's derivation bounded **the seed's weight** below 1e-7 —
`(13/14)^218 < 1e-7`, giving 232 and a floor of 352. Ben's correction: what must fall
below float32 resolution is `weight × seed_difference`, and the seed difference is **not
bounded by 1**. It is the gap between the true early average gain/loss and whatever a
truncated history produces — a difference of average gains, in price units. Near RSI 50 a
float32 ulp is 5.95e-06 and the measured residual was **3.815e-06**, the same order of
magnitude. **352 was marginal by construction**, which is precisely why it held in some
pairs and not others rather than in none or in all.

**Measured, five symbols × twenty-five timestamps, 125 comparisons:**

| tail | byte-identical | failing |
|---|---|---|
| **352** (1e-7, the old floor) | **32 / 125** | every symbol |
| 400 | 116 / 125 | AMZN, MSFT |
| **414** (1e-9) | **120 / 125** | AMZN |
| 420 | 125 / 125 | — |
| **445** (1e-10, the new floor) | **125 / 125** | — |

The 1e-9 case earns its own standing test: 414 is **not** universally byte-identical, so
the extra margin in 445 is measured rather than assumed. Byte-identity is empirically
reached at 420 in this sweep; the floor is 445 because **a number that a sweep happens to
pass is exactly how 352 arrived**. The target is the thing to argue with, and it is stated
in the constant's comment so the next person changes 1e-10 rather than 445.

**Why a tolerance was rejected.** The alternative was to define parity as bit-identical for
the FIR channels and ≤1 ulp for the recursive one. "Within one ulp" is the kind of *close
enough* that lets a genuine one-bar offset hide, and a tolerance is a place bugs live.
Byte-identity is a property nobody can argue with.

**Consequence.** The live loop requests 445 bars rather than 352 — one extra page. GB-9's
row, §7.2, `ARCHITECTURE.md`, `README.md` and the GB-9 DECISIONS entry are corrected **in
place, as corrections**, stating what was actually verified (one symbol, one timestamp) and
what the sweep found. The parity test now **sweeps by construction**: a single-point parity
test is what produced the error, and a test that could produce it again is not fixed.

---

## 2026-08-18 — GB-23: an unexplained position is quarantined, not adopted and not ignored

**Decision.** `engine/reconcile.py` resolves every divergence **in favour of the broker**
and logs it. The three cases:

1. **A broker position with no local record is quarantined.** It is recorded in
   `Book.unmanaged` and counted by `Book.committed`, but never given protective levels and
   never traded.
2. **A local position the broker does not have is dropped.**
3. **A quantity mismatch takes the broker's number** and keeps the local provenance.

**Reasoning.** Ben's prior on case 1 was that it must not be adopted, and the reasoning
holds up when the alternatives are written out. Adopting it means managing a position with
**no entry price, no stop, no target and no decision behind it** — the system could neither
protect it (protection is derived from an entry price it does not have) nor explain it
(GB-32's replay would have nothing to replay). A system whose whole claim is that every
position traces to a forecast cannot quietly acquire one that does not. Ignoring it is
equally wrong for the reason he gave: it **consumes buying power the sizer believes is
free**, so ignoring it lets the sizer over-commit the account. Quarantine is the only
option honest about both — visible, counted, untouched. Refusing to run was considered and
rejected: a paper account can acquire a stray position for reasons that have nothing to do
with this system, and a live loop that refuses to start on account of one is a loop that
does not run.

Case 2 is not primarily an error path: **a filled stop looks exactly like it**, which is why
it resolves silently-but-logged rather than raising.

**Consequence, and the limit of the scope.** Reconciliation corrects **existence and
quantity**. It cannot correct **provenance**, because the broker has none to offer — a book
that lies about *why* it holds something will keep that lie through any number of cycles.
Demonstrated live during the acceptance run: a hand-desynchronised book claiming a managed
AAPL holding kept its (fabricated) decision ID while its quantity was corrected to the
broker's. The defence against that is that only the executor writes provenance, and it
writes it from the decision that caused the order.

`MISSING_PROTECTION` is **detection only**. GB-26 rule 3 acts on it. A reconciler that
started submitting and cancelling would be the order-lifecycle state machine
`SOLO_BUILD_PLAN.md` §2 cut, and the one module in the live path that only ever reads is
worth keeping that way.

---

## 2026-08-18 — GB-26 protection policy: five rules, ruled before the loop is wired

**Decision.** Ben's ruling, recorded now and **implemented in GB-26**, not here. It closes
the gap the GB-22 parity finding opened.

1. **Arm protection in the same cycle that observes the fill.** After submitting an entry,
   **poll** for the fill rather than waiting for the next 60-second tick. The open is the
   most volatile minute of the session and it is the wrong minute to be idle.
2. **Re-arm every open position's stop and limit at the start of every session, before
   anything else in the cycle.** Entries come after protection.
3. **Every cycle verifies that every open position has both legs live at the broker.** A
   position without protection is a **risk event, not a warning**: log loudly, arm
   immediately, and **if arming fails twice in succession, flatten the position at market**.
   An unprotected position is worse than a closed one.
4. **When one leg fills, cancel the other in the same cycle** — there is no OCO linkage to
   do it. **Verify** the cancellation rather than assuming the fill implies it.
5. **GB-57 states the residual honestly:** the backtest's stop is continuously present, the
   live stop is **re-established each session**, and the exposure is the interval between
   the open and arming plus any arming failure. It must **not** be overstated as "live is
   unprotected overnight" — that is not what happens.

**Reasoning.** The stop is a fixed price, so the overnight case that looked like a hole is
not one: a gap through the level leaves the re-armed stop marketable at the open and it
fires there, which is what GB-18's ``stop_gap`` rule already models. The real exposure is
the arming interval and, far more dangerously, a **silent arming failure** — bounded and
visible versus unbounded and invisible. Rules 1 and 2 shrink the first; rule 3 converts the
second from silent into loud, with a flatten as the terminal answer because an unprotected
position is worse than a closed one.

**Consequence.** GB-26's "Done when" carries all five. GB-23 supplies the input rule 3
needs: reconciliation reports which holdings have no live protective orders, as a detection
without an action.

---

## 2026-08-17 — GB-22: `MIN_SHARES` was the right idea in the wrong unit

**Decision.** `engine.risk.MIN_SHARES = 0.001` becomes **`MIN_ORDER_NOTIONAL = 1.00`**, and
`shares_for` tests the **notional** rather than the share count. A second constant,
`QUANTITY_DECIMALS = 9`, floors the share count to the precision the broker stores.

**Reasoning.** GB-18 wrote the constant as "Alpaca's minimum fractional order quantity" and
flagged it for verification at GB-22. Verified against the **live paper API**, verbatim:

| submitted | response |
|---|---|
| `qty=0.001` AAPL (~$0.31) | `{"code":40310000,"message":"cost basis must be >= minimal amount of order 1"}` |
| `notional=0.50` | `{"code":42210000,"message":"notional amount must be >= 1.00"}` |
| `notional=1.00` | accepted |
| `qty=0.003` of a $305 stock ($0.92) | rejected |
| `qty=0.003278689` of the same ($1.0004) | accepted |
| `qty=0.123456789012` | accepted, **recorded as `0.123456789`** |

Ben's reading was right and the difference is behavioural rather than cosmetic. A
share-count floor **permits $0.31 of AAPL**, which the broker refuses, and **refuses 0.0005
shares of a $5,000 stock** ($2.50), which it accepts. The rule was in the wrong unit, so it
was wrong at both ends — and it errs by *permitting* orders that will be rejected live,
which is the direction that would have surfaced as an unexplained live/backtest divergence
rather than as a test failure.

The precision finding was not asked for and matters as much: Alpaca **silently truncates**
a quantity to nine decimals. A backtest carrying more precision than that believes in a fill
that could not have happened, so the floor lives inside the one shared conversion and both
systems inherit it.

**Consequence.** `shares_for` uses `Decimal` quantisation rather than
`floor(x * 1e9) / 1e9`: Hypothesis found a $0.01 price and a $46M account where the scaled
value leaves float64's exact integer range and the "floor" lands on the wrong number. The
property test that caught it is GB-21's, which is the argument for having written it.

---

## 2026-08-17 — GB-22 parity gap: Alpaca refuses a bracket on a fractional quantity

**Decision.** Recorded as a **live/backtest parity gap**, and worked around with two
standalone day orders rather than by changing the backtester or abandoning fractional
sizing. The gap itself is reported, not closed.

**Reasoning.** Ben asked whether the live path can express what the backtest assumes. It
cannot, and the API says so directly:

| submitted | response |
|---|---|
| BRACKET on `qty=0.5` | `{"code":42210000,"message":"fractional orders must be simple orders"}` |
| OCO on `qty=0.5` | same |
| OTO on `qty=0.5` | same |
| **BRACKET on `qty=1`** | **accepted, with both legs** |
| `qty=0.5` with `time_in_force=GTC` | `{"code":42210000,"message":"fractional orders must be DAY orders"}` |
| **STOP sell `qty=0.01` against a held fractional position** | **accepted** |
| **LIMIT sell `qty=0.01` against the same** | **accepted** |

So the refusal is about the order **class**, not about protective orders as such. A
fractional position can be protected — by a stop and a limit submitted **separately**, which
is what `executor.protect` does. Three consequences follow, and all three are limitations
the report must carry rather than implementation details:

1. **No OCO linkage.** If the stop fills, the target is still live. The caller must cancel
   it, and there is a window in which both could fill.
2. **Day orders only.** Protection **expires at every close** and is re-established each
   session.
3. **The residual divergence is narrower than it first appears**, and the first version of
   this entry overstated it as "unprotected overnight". Ben's correction, which is right:
   the stop is a **fixed price** set at entry, so an overnight gap below it leaves the
   re-armed stop **immediately marketable at the open** and it fires at roughly the open —
   which is precisely what GB-18's ``stop_gap`` rule models. An intraday touch is covered
   because the stop is armed through the session. The backtest is not modelling protection
   the live system lacks; it is modelling protection the live system **re-establishes each
   morning**. What actually diverges is **(a)** the seconds between the open and the arming
   and **(b)** a cycle in which arming fails and nothing notices — and (b) is far the more
   dangerous, because (a) is bounded and visible while (b) is silent.

The alternative — rounding to whole shares so a bracket becomes legal — was rejected: GB-18
chose fractional sizing precisely because flooring discretises a percentage-of-equity rule
by price level, and it would replace a stated limitation with a silent distortion of every
position size in the study.

**Consequence.** `executor.protect` arms protection only on the **filled** quantity, because
arming against a position that does not exist is a short sale and
`{"message":"fractional orders cannot be sold short"}` refuses it. An order submitted after
the close therefore fills at the next open **unprotected until the next cycle**, which is
GB-26's problem and is written into GB-22's row as an open question rather than solved here.

---

## 2026-08-17 — GB-25 finding: the return decomposes into exposure, timing and friction

**Decision.** Recorded, not acted on, at Ben's instruction. No change to the model, the
band, the sizer or the config follows from it. It is the decomposition GB-57 needs so that
"+0.45% per fold" cannot be read as evidence the system found something.

**Reasoning.** A strategy with **no timing skill at all** earns roughly its average exposure
times the market's return. Measuring that counterfactual splits the result into three parts
that a single number hides. Per fold, over 16 folds, reconstructing exposure bar by bar from
the trade log and each symbol's own closes:

| | per fold |
|---|---|
| average gross exposure (time-weighted) | **16.00%** of equity (peak 51%) |
| buy-and-hold return | +7.37% |
| **earned by being in the market** (0.16 × 7.37%) | **+92.6 bps** |
| **given back to timing** | **−37.0 bps** |
| **paid in costs** | **−10.7 bps** |
| **= actual** | **+44.8 bps** |

So the strategy does lag its own beta: the exposure it held would have earned nearly twice
what it made. Two ways of putting the same thing, both measured:

- The exposure that would explain +44.8 bps with **zero** timing skill is **6.1%**, against a
  measured 16.0%.
- Regressing fold return on market return gives **β = 0.141** (se 0.028, t = 5.1, R² 0.65) —
  statistically indistinguishable from the 0.160 average exposure, so the exposure explains
  the returns — with an intercept of **−0.59% per fold** (se 0.44, t = −1.4). The alpha is
  negative and **not** distinguishable from zero.

**Neither is the timing residual itself.** Mean −37.0 bps, sd 198 bps, sem 49.5,
**t = −0.75**, positive in 7 folds of 16. There is no measurable timing skill in either
direction, which is the honest finding.

One structure worth reporting beside it: the timing residual is **negatively correlated with
the market return**, r = −0.47 (t = −1.98, n = 16). The strategy gives back most in the
strongest up quarters — fold 4, market +28.5%, residual −3.45% — and gains most in the worst
down quarter — fold 11, market −21.8%, residual +4.80%. That is the signature of a model
that under-calls up, and it is the same down-bias the GB-20 diagnostic measured from the
forecasts alone. Two independent routes to the same property.

**Consequence.** GB-57 reports the three-way split rather than the headline. The sentence it
licenses is precise: *the strategy earned 93 bps a fold from being in the market, gave back
37 to timing and 11 to costs, and the timing term is not distinguishable from zero.*

**Caveat, stated because it bounds the claim:** the counterfactual assumes the exposure was
to the equal-weight basket, while the strategy holds whichever symbols it picked. Fold 12
contributes a 0% exposure and a 0.0 return, correctly, since it stood aside.

---

## 2026-08-17 — Three references, because persistence is a degenerate baseline for trading

**Decision.** Spec §7.3's reporting rule is amended, ruled by Ben. Forecast metrics (MAE,
RMSE) are deltas against **persistence**; direction accuracy is against the **always-long
bar**; return, Sharpe and max drawdown are against **buy-and-hold**, with persistence shown
beside them as the do-nothing floor. A **buy-and-hold arm** is added and runs through the
same backtester as every other arm.

**Reasoning.** Persistence stands aside on 16 folds of 16 — it forecasts zero, so no band
can fire — which makes its return exactly 0.0000 and its Sharpe undefined. The return and
Sharpe columns were therefore **deltas against cash**. The strategy's +0.45% per fold reads
as a result against that, and the folds tile 2022-07 to 2026-07, a period in which an
equal-weight hold of these five symbols returned far more. A reader who knows the period
would have seen it immediately and correctly discounted the whole table. It is the same
failure the direction column had two days earlier — a reference that cannot lose — arriving
through a different column.

Measured, once the arm existed: **buy-and-hold returns +7.37% per fold against DLinear's
+0.45%**, Sharpe **1.48 against 0.69**, positive in 10 folds of 16 against 9. The strategy
does not beat the market it trades in, and the table now says so.

**Consequence.** Three references in one table, so each column's header names its own:
`mae_vs_pers`, `dir_vs_long`, `ret_vs_bh`, `sharpe_vs_bh`. GB-57 must state what each
answers, or three baselines will read as three attempts to find a flattering one.

Two details of the arm, both stated where they are implemented:

- **It is run, not drawn.** One `enter_long` per symbol on the first test bar, filled at the
  next open like every other order, liquidated at the final close. Entry slippage, both
  fees and the gap rules apply, so nobody can argue the comparison was arranged in the
  strategy's favour.
- **Its stop and target are removed**, expressed as infinite fractions rather than by
  special-casing the engine. A 3% stop on a passive holding would make the arm "this
  project's risk rules applied to a passive entry", which is a different thing and would
  understate the market it represents.
- **It is fully invested and the strategy is capped at `max_gross_exposure`.** Part of the
  gap is exposure rather than skill, and GB-57 must say so.

**The assertion that makes it self-checking:** buy-and-hold calls up on every window, so its
direction accuracy **is** the always-long bar. The two are computed by different functions
for different purposes and are asserted equal on every fold. They agree; had they not, one
of them would have been putting a wrong bar under every direction number in the study.

---

## 2026-08-17 — GB-21: Sharpe gains a second qualifying path, because buy-and-hold has no trades

**Decision.** `metrics.sharpe` returns NaN when an arm has fewer than
`MIN_TRADES_FOR_SHARPE` strategy trades **and** was exposed on less than
`MIN_EXPOSED_FRACTION` (0.5) of the fold's bars. Either condition alone now qualifies it.
The five-trade rule is unchanged; a second path is added beside it.

**Reasoning.** Buy-and-hold has **no strategy trades at all** — it enters once and its exit
is the data running out, which is administrative by GB-18's definition — so under the
original rule it could never have a Sharpe, and the reference it was just made for would
have been NaN in every cell. The trade count was always a proxy: GB-19's own comment says
the concern is "a curve that is mostly flat because the account was mostly in cash". That
is a statement about **exposure**, and buy-and-hold is the maximally exposed arm there is.
So the reason is now checked directly. An arm that qualifies on neither path — persistence,
or a strategy that traded twice — still gets NaN, and the existing tests for that case pass
unchanged, because their trades open and close on the same bar.

**Consequence.** `exposed_fraction` is public: GB-57 wants it beside the Sharpe anyway, as
the number that explains why a fold's ratio is what it is. This amends an approved ruling
and is flagged as such rather than folded in quietly.

---

## 2026-08-17 — Open question for GB-41: the down-bias hypothesis is mean reversion

**Decision.** Recorded, not tested. GB-41 starts from this hypothesis rather than from
scratch. Nothing in the model or the features changes now.

**Reasoning.** The GB-20 diagnostic established that DLinear's direction shortfall is a
**down-bias** — it calls up on 45.7% of windows against a 56.25% up rate — and left open
*why* a zero-initialised, no-intercept model forecasts down more often than up on data that
rose. Ben's hypothesis, which fits every measurement taken so far: **short-horizon mean
reversion**. If the model learns "recently up → predict down", and the market mostly rose,
it will call down more often than up. Weak mean reversion in daily equity returns is a real
and documented effect, and **the absence of an intercept is what makes it decisive**: the
model cannot capture the drift and then detect reversion *around* it, so it captures only
the reversion. The drift is the part it is structurally unable to represent, and the drift
is exactly what the always-long bar consists of.

**The test, cheap and available now.** Correlate the sign of the model's forecast against
the sign of the trailing `H`-day return in the input window, per fold and pooled. A strong
negative correlation confirms it. Deliberately **not run** as part of GB-20 — it belongs
with GB-41's diagnostics, where an intercept variant or a de-drifted target can be measured
against it rather than merely discussed.

**Consequence.** If confirmed, the finding is a structural one worth the report: the
no-intercept constraint that makes attribution exact (`Σ per_channel == forecast`, §4.2)
also removes the model's ability to represent drift. That is a genuine explainability /
accuracy trade-off, measured rather than asserted, and it belongs in GB-57 beside the
initialisation finding.

---

## 2026-08-17 — §7.3's `Sharpe > 2.0` alarm applies to the aggregate; the worked example

**Decision.** §7.3 gains a paragraph, ruled by Ben: **the alarm is about the aggregate
figure across folds, not any single fold.** A per-fold Sharpe on ~60 bars has a standard
error near 2.0 annualised, so single-fold excursions past 2.0 are expected under a true
Sharpe of zero. Evidence of leakage is an *aggregate* above 2.0, or a per-fold distribution
**asymmetric** on the upside without matching negative excursions. Both the aggregate and
the per-fold spread must be recorded whenever the rule is invoked. This entry is the worked
example the rule refers to: the alarm fired, was audited, and correctly did not stick.

**Reasoning.** The batch-size re-check produced per-fold test Sharpes above 2.0 in **8 of
32 fold-runs** — up to 4.56 — and §7.3 requires a stop and an audit of `builder.py` and the
fold boundaries. The audit:

- **The arithmetic that settles it.** The standard error of a Sharpe estimated on n ≈ 60
  daily observations is ≈ **0.129**, which annualises to **2.05**. A true Sharpe of zero
  therefore throws ±2 routinely. The largest observed value, 4.56, is **2.23 σ**, and the
  expected maximum |z| over 32 fold-runs is around **2.5**. The extreme is smaller than what
  32 draws from an empty surface produce.

- The causality harness passes on the keystone and the indicators, in both perturbation
  modes; the fold embargo of exactly `H` window-ends is tested against the failing case;
  and calibration is now held to `assert_fit_isolated`, so the band cannot see the test
  split either.
- The distribution is **symmetric**: the same runs produce −3.27, −3.79 and −2.25. A leak
  produces one-sided inflation, not a spread of large numbers in both directions.
- Each figure annualises a **3-month** fold of roughly 60 bars carrying 8–28 trades. The
  sampling error on an annualised ratio from that many observations is of exactly this
  order, which is the same argument that put `MIN_TRADES_FOR_SHARPE` in `metrics.py`.
- The aggregate, which is what the rule is aimed at, is **+0.65** across 16 folds.

**Consequence.** No leak found; the per-fold values are read as variance. GB-57 must report
the fold-level spread rather than a best fold, and any *aggregate* above 2.0 re-triggers the
rule with none of this reasoning available as an excuse.

---

## 2026-08-17 — GB-20: `model.batch_size` reverts to 64; the MAE-based adoption is overturned

**Decision.** `model.batch_size: 64`. Full batch (`null`) was adopted earlier the same day
and is reverted. The `null` option and its validation stay in the loader — the setting is
reverted, not the feature.

**Reasoning.** The adoption rested on a 16/16 improvement in **MAE**, which §7.3 bans as a
headline metric, so it was recorded as provisional and GB-20 was made to carry a re-check on
the metrics the study reports. The re-check ran end to end — train, calibrate on validation,
backtest on test — over all 16 folds under both settings:

| | full batch | mini-batch 64 |
|---|---|---|
| mean total return per fold | +0.07% | **+0.44%** |
| folds with a positive return | 6/16 | **9/16** |
| mean Sharpe (defined folds) | −0.09 (n=10) | **+0.65 (n=14)** |
| head-to-head Sharpe | better in 3/16 | **better in 7/16** |
| mean direction accuracy | 0.4751 | **0.5071** |
| direction, head to head | better in 3/16 | **better in 13/16** |
| mean MAE | **0.0206** | 0.0332 |
| folds standing aside | 3/16 | 1/16 |

Full batch wins MAE in 16 folds of 16 and loses every other comparison. That is exactly the
pattern §7.3 predicts: converging more completely to the MSE optimum produces a flatter
forecast, which wins an error metric and forecasts nothing. The one metric that pointed the
other way was the one the rule says not to steer by.

**Consequence.** The seed is load-bearing again — it decides the mini-batch partition — so
the tests that GB-2 rewrote when full batch was adopted are rewritten back: the configured
setting once more implies "a different seed gives different weights", and the full-batch
property (any seed, bit-identical weights) is kept as a test of that code path with the
config forced. Direction accuracy at 0.5071 is still **below the always-long bar of
0.5625**, so this is a choice between two arms that do not beat always calling up, and GB-57
must say so rather than presenting the winner as a result.

**Caveat kept in front.** 16 folds, per-fold Sharpes ranging from −3.3 to +4.6: the
comparison is directionally consistent across four metrics but is not a significance claim.

---

## 2026-08-17 — GB-19: persistence has no direction accuracy, and 0.5 is not the chance level

**Decision.** Three rulings in `backtest/metrics.py`.

**1. Sharpe is computed from the equity curve's daily returns, not from per-trade returns.**
They are different numbers and the report must say which. Daily returns are what an
investor experiences: they include the cost of sitting flat, which per-trade Sharpe ignores
entirely, so a strategy in the market ten days a year cannot be made to look like one in it
every day. Per-trade Sharpe also has as many observations as trades — a dozen or so per
fold — and is trivially inflated by taking fewer, larger positions. Annualisation is well
defined for a daily series and ill defined for trades with different holding periods. The
property that settles it: **two arms with the same equity curve must get the same Sharpe**
whatever their trade counts, or the study is comparing trade frequency. Tested directly.

**2. Below five strategy trades, Sharpe is NaN.** A 3-month test fold is ~61 bars; a
strategy taking fewer than five round trips has held a position on a minority of days, so
its return standard deviation is dominated by the few days it was exposed and the ratio is
large or small for reasons unrelated to the strategy. This project already treats
`Sharpe > 2.0` as a leak alarm, and a metric that can manufacture that from two trades makes
the alarm useless. **Five is a judgement, not a derivation, and is stated as one** — what
makes it safe is that the alternative is not "a slightly noisy number" but one with no
sampling argument at all, and that `summarise` prints the trade count beside the Sharpe so
a blank cell explains itself. Administrative exits do not count toward the minimum.

**3. Direction accuracy is undefined for a forecast of exactly zero — so persistence has
none, and it cannot be the baseline for that column.**

`sign(0)` is 0, which agrees with nothing. Counting a zero forecast as wrong would score a
model that declines to predict at **0.0** rather than at chance; counting it as right by
convention would invent an opinion it never expressed. So zero-forecast windows are excluded
from the denominator, and when none remain the result is NaN.

Persistence forecasts zero every time. **Measured on all 16 folds: its direction accuracy is
NaN in 16/16.** That is the correct answer, not a gap to be filled. Persistence is the right
baseline for MAE, RMSE, return and Sharpe, all of which it genuinely competes on. It is not
a directional model, and substituting 0.5 for it would put a number in the table that the
baseline never produced.

**What the reference should be instead, and why this is not a technicality.** The honest bar
is the majority-class rate — the accuracy of always calling the direction that happened more
often — measured from the same realised returns rather than assumed. Measured across the 16
folds:

| | value |
|---|---|
| H-day cumulative return positive | mean 0.5625 (0.4203 – 0.6842) |
| **Measured chance level** `max(up, 1−up)` | **mean 0.5865** (0.5049 – 0.6866) |
| DLinear direction accuracy | mean 0.4751 (0.3825 – 0.5379) |
| DLinear beats the **measured** chance level | **1 / 16 folds** |
| DLinear beats a naive 0.5 | 6 / 16 folds |

**Reporting against 0.5 would have flattered the model by 8.65 points and turned "beats
chance in 1 fold of 16" into "beats chance in 6".** That is precisely the misreading §7.3
exists to prevent, arriving through the baseline rather than through the metric. The
`summarise` table therefore carries a `direction_reference` column recording the number
actually used, so a reader is never left to assume it was 0.5.

**Consequence.** §7.2 and §7.3 record that the direction column is referenced to the
measured base rate while every other column is a persistence delta. `ArmResult` is the unit
every metric takes, so a metric and its baseline are computed by the same code path.
Annualisation uses each fold's own bar count and calendar span, never a constant 252 —
folds are calendar months and their bar counts differ. 36 tests.

**Read alongside GB-15's finding.** Direction accuracy of 0.4751 was already known to be at
chance; it is now known to be *below* the measurable chance level in 15 folds of 16. The
model is not yet good, and the report must say so in those terms.

---

## 2026-08-17 — FINDING: a pooled batch is not monotonic, so `FitProvenance` understated its own range

**The finding.** `FitProvenance.from_batch` derived the training range as
`timestamps[0]` and `timestamps[-1]`. That is correct for a single-symbol batch and wrong
for a pooled one, because pooling concatenates several symbols' ranges and the result
revisits every date once per symbol. The last row is the *last symbol's* last window, not
the batch's latest.

**Why it matters, and why it would not have been noticed.** GB-25's leakage audit asks
whether the range a model was fitted on intersects the range it is scored on, and it reads
that range from the checkpoint. A range that understates its own extent shrinks the
intersection — so a model that *had* seen part of its test period could report a training
range that ends before it, and the audit would pass. The audit would have been reading a
number that was quietly not what it claimed to be, on exactly the question the audit exists
to answer.

Nothing about it is visible from the outside. The dates are real dates, in the right order,
inside the right fold. Only the *extent* is wrong, and only when symbols end at different
timestamps — which is the ordinary case the moment the universe is pooled.

**The fix.** `timestamps.min()` and `.max()`. Recorded here as a finding rather than a fix
note because the general lesson outlives it: **a derivation that was correct under an old
contract does not announce itself when the contract changes.** `from_batch` was not edited
by the contract change, passed every existing test, and was wrong from the moment
`WindowBatch` learned to hold more than one symbol. Two tests now pin it — one asserting a
pooled batch's timestamps are non-monotonic, one asserting `fitted_end != timestamps[-1]`
for a pooled batch whose symbols end at different dates.

---

## 2026-08-17 — Full batch adopted as `model.batch_size: null`, on a metric this project bans

**Decision.** `model.batch_size` becomes `null` — one batch of everything. Ben's ruling on
the spelling: `null` reads as an absence of mini-batching rather than a lie about the
field's name, cannot silently revert when a fold grows, and is the idiom `signal.min_trend`
already uses for a deliberately-absent value. The loader accepts `null` or a positive
integer and nothing else — not zero, not a float, not `true`, not a string — so a typo
cannot become "full batch" by accident.

**What decided it.** 16 pooled folds, full batch better on test MAE in **16/16**, median
1.288× against 2.058× persistence. Not an artefact of `patience` truncating mini-batch:
against mini with `patience` disabled it is also 16/16.

**What that evidence is not.** The decision was made on **MAE, which spec §7.3 bans as a
headline metric — and bans for exactly the reason that applies here.** A flatter forecast
wins MAE. Full batch converges more completely to the MSE optimum, so a flatter forecast is
the *expected* consequence of the change rather than an independent confirmation of it. And
direction accuracy moved the other way in the same experiment, 0.5071 → 0.4751, which is
plausibly the same phenomenon seen from the other side. Both direction figures are near
chance and 16 folds will not separate them, so this is not evidence *against* full batch
either. It is evidence that the decision rests on a **diagnostic** metric rather than a
headline one.

**Therefore this decision is provisional and carries a scheduled re-check.** Once GB-19 and
GB-20 exist — metrics, and thresholds calibrated so a backtest produces a Sharpe — both
batch settings are re-measured on **Sharpe and direction accuracy**, and full batch is
reverted if the headline metrics disagree with MAE. The re-check is written into GB-20's
"Done when" so it cannot be forgotten. That is the only honest way to hold a decision made
on a banned metric.

**One thing the ruling bought that the measurement had said was unavailable.** With one
batch the permutation changes only the order floats are summed in — measured at a relative
5e-15 across seeds, which was the reason full batch was rejected the first time. It is now
skipped when there is a single batch, so full-batch training is **bit-identical across
different seeds**, not merely across repeats of the same one. Two tests hold it: a
2,000-window fold trained identically under three seeds, and a mini-batch run that still
differs under two, so the first is not passing vacuously.

**Unchanged and worth keeping in front.** Full batch at 1.288× is better than mini and
still worse than persistence. The improvement is real and the model is not yet good.

---

## 2026-08-17 — GB-16: the checkpoint decides everything, and full batch now wins 16/16

**Decision.** Four rulings in `model/predict.py`, plus one measurement that reopens a
choice made three commits ago.

1. **Per-symbol statistics come from the checkpoint, never from the caller.** The caller
   names a *symbol*; it cannot hand in a scaler. Passing statistics in would let a caller
   normalise AAPL's window with NVDA's numbers, and nothing downstream could tell.
2. **A symbol the checkpoint has no statistics for is refused**, naming it and listing what
   exists. Both fallbacks are wrong. Using another symbol's scaler, or none, presents the
   shared weights a distribution they were never fitted on, and the result is a
   plausible-looking forecast rather than an error. Fitting a scaler at prediction time is
   worse: the only data available to fit it on includes the period being predicted.
   Refusing keeps "what did the model see?" the same question at inference time that it is
   in GB-25's audit.
3. **A different channel set is refused, and so is a reordered one.** The weights are
   indexed by position, so a swap applies each weight matrix to a different series and no
   shape check would catch it.
4. **The hash can be re-checked at the point of use**, not only at load. GB-26's loop runs
   for a session and can outlive the configuration it started with.

**Measured.** The two inference paths are **bit-identical** across every test window — 305
pooled windows batched in 24.6 ms, 0.88 ms per single window. That is not luck: both call
`build_windows`, one with `as_of` set, and GB-9 made that a filter over the same list of
end positions rather than a second implementation. The test asserts every window, not a
sample.

**Reopened: full batch versus mini batch, re-measured under pooling as Ben asked.**

| 16 pooled folds | test MAE vs persistence | direction | best epoch |
|---|---|---|---|
| mini-batch (64) | mean 2.217× · median 2.058× | 0.5071 | 2.9 |
| mini-batch, `patience` disabled | mean 2.158× · median 2.054× | 0.4924 | 29.1 |
| **full batch (2505)** | **mean 1.345× · median 1.288×** | 0.4751 | 44.4 |

**Full batch is better in 16/16 folds**, and against unpatienced mini in 16/16 too — so the
gap is not `patience` truncating mini. Under per-symbol training the same comparison was
57/80 and worth 8%; pooled it is unanimous and worth **37% of the median ratio**.

The mechanism is the one the earlier decision predicted would change. Per symbol the
problem was underdetermined at 0.42× equations per parameter, and mini-batch gradient noise
was a second regulariser acting on a real null space. Pooled it is **overdetermined at
2.09×** — there is no null space left to regularise, so the noise is only noise, and the
exact gradient wins. Mini's best epoch swinging from 2.9 to 29.1 depending on whether
`patience` is on, with no change in the outcome, is that noise visible in the curve.

**Not changed here, because it needs a ruling.** Switching means expressing "full batch"
through `model.batch_size`, and that field names a mini-batch *size*: it would have to hold
a number larger than any fold's window count, which is a lie about the field and reverts
silently the day a fold outgrows it. The options are a §5 value change with that caveat
accepted, or a new §5 field. Both are Ben's.

**Still worse than the baseline.** Full batch at 1.288× median MAE and 0.4751 direction is
better than mini and still not better than persistence. The improvement is real and the
model is not yet good; §7.3 requires both halves to be said together.

---

## 2026-08-17 — CONTRACT CHANGE: `WindowBatch.symbol` becomes `symbols`; universe-wide training; per-symbol scaler

**Decision.** Three rulings by Ben, one of which changes a frozen contract.

1. **Every forecaster trains across the universe**, not per symbol.
2. **`WindowBatch.symbol: str` becomes `symbols: tuple[str, ...]` of length B**, one entry
   per window. A single-symbol batch carries `("AAPL",) * B`. `WindowBatch.concat(batches)`
   is added as the only place pooling happens.
3. **Normalisation statistics are fitted per symbol, never pooled.**

**Reasoning — (1).** GB-15 measured the sample counts: per symbol a fold gives 501 windows,
so `501 × 4 = 2004` equations against DLinear's 4,800 parameters — **0.42×,
underdetermined**; across five symbols it is **2.09×, overdetermined**. Ben's addition, and
it is the decisive half: **FITS at 1,200 parameters is already overdetermined per symbol at
1.67×.** Training DLinear per symbol while FITS is trained universe-wide would handicap
DLinear in a way FITS is not handicapped, and GB-49's grid would measure that handicap and
report it as architecture. This is therefore a **correctness requirement for the
comparison**, not a performance preference — which is a stronger claim than the one GB-15
made, and the right one.

**Reasoning — (2).** A pooled batch genuinely has one symbol per window, so the plural is
the honest shape. The alternatives are worse. Taking a *sequence* of batches into the
`Forecaster` protocol pushes pooling into every consumer, so three models would each
implement it and two would eventually disagree. Keeping a single string forces the pooled
case to be encoded outside the contract — a naming convention, a parallel array, a
comment — which is exactly the implicitness removed when `ChannelStats` was promoted out of
`build_windows`. `FitProvenance.symbols` was already a tuple, so it absorbed the change
without an edit.

`concat` validates rather than trusts, because each thing it refuses is silent if allowed:

| Refused | Why it would not be noticed |
|---|---|
| Different channel tuples | Shared weights applied to a different meaning at the same index; every number stays plausible |
| Different window geometry | Surfaces as a numpy broadcast error far from the cause |
| Different `source` | GB-8 measured a 4-9% vendor volume difference; pooling across vendors is the same defect as splicing within one series |

One consequence worth stating: a pooled batch's `timestamps` are **not monotonic** — each
symbol contributes the same date range. `FitProvenance.from_batch` therefore takes
`timestamps.min()` and `.max()` rather than the first and last. Under the old derivation a
pooled batch would have reported the *last symbol's* end as the batch's end, understating
its own extent — and GB-25 compares that range against the test range, so a range that
understates itself lets a leak pass.

**Reasoning — (3), Ben's.** Per-symbol normalisation is what *makes* pooling legitimate: it
removes symbol-specific scale so the shared weights learn the structure common across
symbols. A pooled scaler leaves NVDA's inputs systematically larger than MSFT's, and shared
weights cannot express a symbol-specific response — the model would be asked to fit a
difference it has no parameters for. `fit_stats` already takes one frame, so this needs no
new machinery; the checkpoint stores `{symbol: ChannelStats}` and inference applies each
window's own.

**Consequence.** Spec §4.2 updated. `build_windows` emits `(symbol,) * B`.
`select_windows` slices `symbols` alongside `X`. Thirteen tests added for the new surface,
covering each refusal. Committed on its own, before GB-16, because burying a contract
change inside a feature is how the next reader misses it.

**Open for GB-41.** FITS applies RIN, a per-window instance normalisation, on top of the
per-symbol scaler. Two normalisation stages in sequence is not automatically wrong, but
whether the scaler still earns its place under FITS is a question to measure rather than
assume. Added to GB-41's report items.

---

## 2026-08-17 — GB-15: the checkpoint is a directory, and early stopping is two mechanisms

**Decision.** Five rulings, three of them measured.

1. **The training loop stays inside `fit`; `train.py` is what surrounds one call to it.**
   Spec §4.3 puts the loop there — `fit(batch, val)` is the contract and early stopping is
   named in it. `train.py` decides what the model may see, records what it saw, and writes
   both down. GB-41 adds FITS without touching it.
2. **A checkpoint is a directory of three artefacts**, not a file: `model.json` (weights +
   `FitProvenance`), `checkpoint.json` (config hash, `ChannelStats`, the ranges shown to
   the model, how training ended), `history.csv` (per-epoch losses). They answer three
   different questions and one of them is not a checkpoint at all — the loss curve is a
   report artefact, and keeping it out leaves every byte of the checkpoint reproducible.
3. **Windows are built once with the training statistics and split by timestamp
   afterwards.** There is no second `build_windows` call that could be handed a second
   `ChannelStats`, so validation cannot be normalised by its own mean and variance even by
   accident. This is the structural version of the hard requirement, not the careful one.
4. **The config-hash check on load refuses on any difference**, including an unrelated key.
   Hashing only the fields believed to matter needs a list of which fields those are, and
   that list is wrong the first time someone adds a field and forgets it. A false refusal
   costs a retrain; a false acceptance puts a model trained under other rules into a
   results table. **Known cost, stated now:** GB-20 calibrates `signal.min_trend` and
   `max_trend`; if those are written back into `settings.yaml`, every checkpoint trained
   before that write is refused. That is correct behaviour and a scheduling fact.
5. **Mini-batch at the configured `batch_size: 64`**, not full batch — see below.

**Reasoning — what early stopping actually does, measured on all 80 arms** (5 symbols ×
16 folds, real cached data). Two mechanisms are habitually spoken of as one, and they are
separable:

| Mechanism | Effect on test MAE vs persistence |
|---|---|
| Final-epoch weights, no selection | 2.757× (median 2.616×) |
| **Best-epoch selection on validation** | **2.004× (median 1.938×)** — better in **80/80** arms, 27.3% mean improvement |
| Adding `patience: 10` on top | 1.999× — **identical model in 79/80 arms** |

So on a convex objective early stopping is not guarding against divergence, and the part
that matters is **best-epoch selection**, not `patience`. `patience` is a compute budget:
it saved a mean of 76.5 of the 100 configured epochs and changed the chosen epoch once in
80 runs.

What selection regularises against is the **null space**. With `input_len 120` there are
501 training windows per fold, so `501 × 4 = 2004` equations against 4,800 parameters —
**0.42× — an underdetermined system.** The training optimum is not a point but an affine
subspace of dimension ≥ 2,796 on which training loss is identical and validation loss is
not. Descending from zero traverses high-curvature directions first, so an early iterate is
a low-norm solution and a late one has fitted noise; stopping the path early is
approximately ridge regularisation. Approximately, not exactly: Adam preconditions the
gradient, so the classical equivalence for plain gradient descent is a guide here, not a
proof. Measured: training loss falls 0.00712 → 0.00141 over 100 epochs while validation
loss falls to 0.00145 at epoch 17 and then **rises to 0.00337**.

**Reasoning — full batch versus mini batch, measured on the same 80 arms.**

| | test MAE vs persistence | direction | epochs |
|---|---|---|---|
| mini-batch (64) | mean 1.999× · median 1.938× | 0.4971 | 23.5 |
| full batch (501) | mean 1.950× · median 1.785× | 0.5113 | 36.1 |
| full better in | **57/80** | 40/80 | — |

Full batch has a real MAE advantage. It is chosen against anyway, for three reasons.

*The decisive argument for it is measurably false.* Full batch was supposed to remove the
seed from the picture entirely, making determinism structural rather than a property to
test. It does not: `randperm` still runs, and summing 501 terms in a different order gives
a different last bit. Measured across three seeds the weights differ by a **relative 4.8e-15**
— deterministic to fifteen figures, but not bit-identical, so the seed test is still
needed and nothing is bought.

*The advantage is in a metric neither arm is winning.* Both configurations sit near 2×
persistence MAE and at 0.50 direction. Choosing between them on an 8% improvement in a
ratio where both are twice as bad as the baseline is optimising a number that is not the
point.

*Expressing "full batch" as a size is fragile.* `batch_size` names a mini-batch size; to
mean "all of them" it would have to hold a number larger than any fold's window count, and
it would silently revert to mini-batch the day a fold grew past it.

**Revisit at the universe ruling.** Per symbol the problem is underdetermined at 0.42×;
across five symbols it is **overdetermined at 2.09×**, which changes the regularisation
argument that this decision rests on. Re-measure then rather than inheriting this.

**Consequence.** `glassbox/model/history.py` is a new module — added to spec §3.4 and to
`tests/test_scaffold.py`, both of which refused it until it was declared. `EpochLoss` lives
there because `ltsf.py` produces it and `train.py` writes it, and `train.py` already imports
`ltsf` through the package `__init__`; putting the type in either would make the pair
circular, and putting it in `contracts/schemas.py` would change a frozen contract to hold a
reporting artefact. `DLinearForecaster` gains `history`, `best_epoch` and `stopped_early`,
which describe the run rather than the model and are deliberately absent from `model.json`.
`select_windows` is public because GB-19 and GB-49 need the same operation.

**Open, and not decided here: per-symbol or universe-wide training.** See the GB-15 row in
spec §9. It requires a contract decision and is Ben's ruling.

---

## 2026-08-17 — GB-18 fix: a missing bar is a halt, not an exit; and the accounting invariant

**Decision.** Five changes to `backtest/engine.py`, all consequences of one defect found in
review.

1. A position is removed from the book **only after** a price has been obtained for it.
2. An exit order that cannot fill — the symbol printed no bar that day — is **carried
   forward** to that symbol's next traded open.
3. An entry order that cannot fill **expires**. The asymmetry with (2) is deliberate.
4. An open position is marked at its **own last printed close**, carried on the position as
   `last_mark`, not at the price it was bought at.
5. `_assert_accounted` runs at the end of **every** backtest, not only in tests: with the
   book flat, final equity must equal initial cash plus the sum of the trade log.

**Reasoning.** The engine popped the position from `positions` before calling `_price`.
When the symbol had no bar that day, the position was gone from the book, no cash was
credited, and no `Trade` was recorded — the holding was simply deleted. Measured on the
committed engine, on a 100,000 account holding one 10,000 position:

| Case | Final equity | Cash + Σ net_pnl | Gap | Trades logged |
|---|---|---|---|---|
| Exit signal on a day the symbol has no bar | 89,999.00 | 100,000.00 | **−10,001.00** | 0 |
| Symbol whose last bar precedes the final timestamp | 89,999.00 | 100,000.00 | **−10,001.00** | 0 |

**10.0% of the account, silently, with an empty trade log.** The gap is exactly the
position's notional plus its entry fee — the whole holding, not a rounding error. The
second case is not an edge case: it fires whenever one symbol's history ends before the
universe's does, which is an ordinary consequence of listing dates, and it would have hit
every multi-symbol backtest in the study.

On (2) versus (3): an exit is an obligation on capital already committed, so abandoning it
leaves a position open with nothing to close it. An entry is a bet on a forecast computed
from a window ending at a specific bar; by the time the symbol trades again that forecast
is stale, and GB-26's live loop would recompute it rather than resurrect the order. Carrying
entries forward would make the backtest describe a system the live loop cannot be.

On (4): marking a halted position at its entry price is the same defect wearing different
clothes — it reports a holding at a price that is not merely stale but *never was current*
after the entry bar. In the measured second case it moved the curve by 89.98 on a day
nothing happened. Carrying `last_mark` also removed a look-ahead nobody had noticed: sizing
an order that fills at bar `t`'s open previously consulted bar `t`'s **close**, which is the
reference project's error in a subtler form. The sizer now sees exposure marked at the
previous close.

On (5): every cash movement in the engine belongs to some trade, and `net_pnl` is defined as
exactly that movement, so the identity is exact in real arithmetic. That makes it a genuine
invariant rather than a heuristic, and cheap enough to run always. A test-only assertion
would not have caught this — the defect needed a bar to be missing, and no test had one.

**Consequence.** `_OpenPosition` gains `last_mark` / `last_mark_time`. `_equity` and
`_gross_exposure` no longer take `bars` or a timestamp. Seven tests added, six of which fail
against the previous engine. Spec §9's GB-18 row records the fifth rule and the invariant.
The invariant will fail loudly if a future change leaves cash unaccounted, which is the
point: this class of bug is invisible in every metric GB-19 computes.

---

## 2026-08-16 — GB-13: DLinear departs from the reference in four places, one of them measured

**Decision.** `DLinearForecaster` uses **per-channel** weights, **no intercept**, **zero
initialisation**, and fits with torch while predicting with numpy. The forecaster registry
moves out of the contract test into `glassbox/model/__init__.py` as `ALL_FORECASTERS`.

**Reasoning.**

*Per-channel weights, because GB-30 decides it.* The reference maps every channel through
one shared `nn.Linear` and emits a forecast per channel; this project forecasts one series
from a channel set. Giving each channel its own matrix per component makes the forward pass
a plain sum of per-channel terms, so `explain` is a **regrouping of the terms `predict`
already computed** rather than a reconstruction. A joint split — decomposing some
projection of the channel set — would mix channels into the components and make the
weights inseparable, which would end the exactness claim. So the question in the task
brief answers itself: attribution must decompose by channel, therefore the split is per
channel. It also could not be otherwise mechanically: a moving average of a multi-channel
window is a moving average of each channel.

*No intercept.* `nn.Linear` carries a bias. With one, `sum(per_channel.values())` falls
short of the forecast by exactly the bias and there is no honest channel to charge it to —
`Attribution.per_channel` is keyed by channel name, and an `"intercept"` key would be a
channel that does not exist. Per-channel biases would restore exactness but are collinear
and meaningless. Since the target is a log return with a mean near zero, an intercept buys
almost nothing, so it goes. Exactness then holds by construction, and a test asserts a zero
window forecasts exactly zero — which would fail the moment someone added one.

*Zero initialisation, and this one was found by measuring rather than reasoning.* Porting
`nn.Linear`'s `U(-1/sqrt(L), 1/sqrt(L))` is wrong for this architecture. That bound assumes
a fan-in of `L` because the reference maps one channel to one output; this model **sums**
`C x 2` such maps, so the true fan-in is `C * 2 * L` — 1200 against 120. A random start
therefore emits forecasts around 1.8 against a target standard deviation of 0.015, about
120x too large, and 100 epochs are spent travelling back. Measured out of sample on
walk-forward fold 1:

| Initialisation | MAE vs persistence | Direction accuracy | Largest contribution |
|---|---|---|---|
| Paper, `1/sqrt(L)` | 4.00× | 0.344 | 1.39 |
| Fan-in corrected, `1/sqrt(2CL)` | 1.96× | 0.328 | 0.43 |
| **Zeros** | **1.94×** | **0.557** | **0.058** |

The last column matters as much as the first. With the paper's initialisation the
attribution is still exact and completely unreadable: a forecast of 0.01 explained by
contributions of +1.39 and −1.20 that cancel. Zero-initialised weights leave contributions
on the same scale as the forecast, which is what GB-30 must render and a supervisor must
believe. The objective is convex, so initialisation cannot change the optimum — only the
path — and starting at zero means the model **starts as the persistence baseline** and every
weight it moves is something it learned.

*Torch fits, numpy predicts.* The decomposition has no parameters, so it is applied once
before training rather than inside a forward pass. Inference is then pure numpy with no
global torch state to be perturbed by, which makes contract property 2 — and properties 5
and 6, which depend on it — true by construction rather than by discipline.

*The registry moved into the package.* `ALL_FORECASTERS` lives in `glassbox/model/__init__.py`
and the contract test iterates it. A test a model can edit is a test that model has judged
itself with; now GB-41 adds a line to the registry and changes nothing in the file that
judges it. It holds factories rather than classes because constructors legitimately differ,
and forcing a construction signature into the frozen protocol would be a contract change
bought for tidiness.

**On the audit's centred-moving-average claim: it is correct, and I checked the code rather
than taking it.** The padding repeats each window's own first and last bar, so `trend[l]`
averages lags `l-12 .. l+12` **within the window** — every one of which is at or before the
window's own timestamp, while the forecast is of bars after it. The causality rule
constrains what a window may contain relative to its target; it does not require lag `l` to
be computed from lags `<= l`, and no forecaster here could work if it did. What *would* be
a violation is a decomposition reaching across the batch, and contract property 6 catches
exactly that: it perturbs later windows and requires earlier predictions bit-identical.
`test_the_decomposition_reads_only_its_own_window` asserts it directly as well. Two things
worth knowing that the audit does not say: the trend is a property of the **window**, not of
the series, so two overlapping windows disagree about the trend at the same timestamp; and
edge replication pulls `trend[L-1]` toward the last observation, making the most recent
trend value the least informative one.

**Parameter count, and a correction the report will need.** DLinear holds
`2 components × 5 channels × (4 × 120) = 4,800` weights and no bias. Spec §6.1 describes
FITS as "~10k parameters", which is the paper's figure for its own configuration, not ours:
at `L = 120` and `cutoff_period_days = 5`, `COF = 24` and the output is
`ceil(124/120 × 24) = 25` bins, so the complex layer holds `24 × 25 = 600` complex weights,
**1,200 real parameters**. DLinear is therefore **4× larger than FITS**, not smaller — the
opposite of what "~10k" suggests. GB-41 must report the actual count and GB-57 must not
repeat the paper's figure.

**Consequence.** On fold 1, DLinear's MAE is 1.94× persistence with direction accuracy
0.557. Worse on MAE is expected and is not tuned away: 4,800 parameters against 501
training windows is heavily over-parameterised, and §7.3 bans MSE-style metrics as headline
numbers precisely because a flatter forecast wins them. Direction accuracy — the quantity
the signal layer consumes — is above chance. One fold on one symbol proves nothing either
way; GB-51's paired Wilcoxon across folds is what decides, and tuning against this fold now
would be fitting to the test set.

---

## 2026-08-16 — GB-18 addenda: one share conversion, one exit flag, one return convention

**Decision.** The notional-to-shares conversion lives in **one** function,
`engine.risk.shares_for`, shared by the backtester and by GB-22's executor. `Trade` gains
`strategy_exit: bool`. The engine docstring and spec §9 GB-57 record that every figure this
project reports is a **total** return.

**Reasoning.**

*One conversion, or the two systems describe different things.* The engine sizes
fractionally — 99.980003999200 shares in the hand-checked scenario. That is a legitimate
choice only if the live executor does exactly the same. If the backtest buys 99.98 and the
executor floors to 99, the divergence is invisible: both modules pass their own tests, and
the gap surfaces months later as an unexplained difference between backtest and paper
results. This is the GB-7 failure family — a perfect schema match with the numbers wrong —
caught this time *before* the second module exists. `shares_for` lives in `engine/risk.py`
(L4), which the harness may import and the executor already will, so neither needs its own
arithmetic. Spec §9's GB-22 row now requires it and asks for a test asserting both callers
derive the same count from the same inputs.

*Fractional, and a floor at the broker's minimum.* Flooring to whole shares would
discretise a percentage-of-equity rule differently for a $500 stock than for a $50 one,
turning a uniform risk rule into one that depends on price level, and would leave the
backtest systematically under-invested against the live account. Where a notional buys less
than the broker's minimum the answer is **no trade**: the broker would reject the order, so
filling it in a backtest invents a trade that cannot happen. It is a module constant rather
than config for the GB-8 reason — it is a property of the venue, not a knob to tune.
**Superseded on 2026-08-17:** GB-22 verified the constant against the live API and found the
unit wrong — it is a minimum NOTIONAL of $1.00, not `MIN_SHARES = 0.001`. See that entry.

*`strategy_exit`, so GB-19 filters on a field.* Five `end_of_data` liquidations appeared in
the real-universe sweep. The rule is pinned now rather than invented later: they **are**
included in the equity curve and total return, because the curve must be complete and the
capital genuinely was returned — and they **are excluded** from hit rate, average trade and
every other per-decision statistic, because no decision was made. A boolean rather than a
string match, so a later rename of an exit reason cannot silently change which trades a
metric counts.

*Every reported number is a total return.* `data/historical.py` fetches with
`auto_adjust=True`, so dividends are folded into the price series: a dividend appears as a
smaller downward step on the ex-date, not as separate cash. Nothing in the engine adds
dividend income because it is already in the prices. Unstated, a reader comparing against a
price-only benchmark would find a discrepancy of roughly the universe's dividend yield per
year and no way to explain it. Now in the engine docstring and required in GB-57's results
chapter.

**Consequence.** A GB-2 guard (`test_no_module_reads_settings_or_environ_directly`) fired on
the new module because its *docstring* mentioned the config file by name — a false positive,
since the guard is a substring scan over the whole source. Reworded rather than loosened;
the guard stays strict. Worth knowing that it will fire again on any module that explains
in prose why something is not configurable.

---

## 2026-08-16 — GB-18: four backtest pricing rules, chosen by measurement

**Decision.** A signal fills at the **next bar's open**. On a bar breaching both levels the
**stop** fills. Slippage is **adverse on both sides**. A **gap through a level fills at the
open**, and is logged distinctly. `settings.yaml` gains `backtest.initial_cash`.

**Reasoning.** All four were measured on the real universe before being chosen — 13,309
entry opportunities across the five symbols at the configured 3% stop and 6% target.

*Next bar's open, not the signal bar's close.* A signal computed from bar `t`'s close
cannot fill at that close; that is the look-ahead GB-10 exists to catch, and the reference
project commits a subtler version of it — pricing a same-bar open fill using that bar's
close. The decisive argument is not realism but **parity**: GB-26's live loop polls after
the session, drops the incomplete bar, and can only act on the next open. Modelling the
backtest the same way makes the two agree by construction, which is the `builder.py`
argument applied to execution. Filling at `t`'s close would make the backtest describe a
system that cannot be built.

*The stop wins an ambiguous bar.* Daily OHLC cannot say which came first, so it is an
assumption either way and the assumption should be the one that cannot be accused of
flattering — this project treats `Sharpe > 2.0` as a leak alarm, and a reviewer who finds
"target-first" has a reason to discount every number. The measurement supplies the second
argument: **10 of 13,309 resolutions, 0.08%**. The conservative choice is essentially free,
which makes it the easiest trade in the module.

*Slippage adverse on both sides.* Buys pay `price × (1 + s)`, sells receive
`price × (1 − s)`. Round trip on a flat trade is exactly `2 × (fee_bps + slippage_bps)` of
the reference notional — **6.0 bps**, asserted against the engine's own output rather than
left as arithmetic in a docstring. The model captures neither size-dependent impact nor
liquidity thinning in stress; that is a stated limitation for the report, not a silent one.

*A gap fills at the open — and this is the ruling that actually moves the numbers.* If the
bar opens beyond the level, that level was never available, and filling there is fiction —
exactly what the reference project does. Measured: **15.6%** of stop exits gap through, and
filling at the stop instead would understate each by a mean of **116.7 bps** of entry
notional (median 65.7, p90 296.4, max 1,407.5). Against a 6.0 bps total friction budget,
this single rule is worth roughly **19× the entire fee-and-slippage model** on the trades it
touches. Re-measured through the finished engine: 14.16% of stop exits gapped, mean 132.4
bps — and 168 *target* gaps as well, which the pre-implementation sweep had not counted.

The log therefore distinguishes `stop` from `stop_gap` and `target` from `target_gap`.
GB-19 and GB-57 can then report how much of the drawdown came from gaps rather than from the
stop rule — a distinction that cannot be recovered later if the engine does not record it
now.

*`backtest.initial_cash: 100000.0` added to spec §5.* The engine needs a starting equity
and there was nowhere to get one. A module constant violates rule 5; a required argument
moves the magic number into GB-24 rather than removing it. The value matches the Alpaca
paper account exactly, so backtest and paper-trading numbers are directly comparable rather
than merely similar.

**Consequence.** Sizing is injected through a `PositionSizer` protocol declared in the
engine, so no risk logic is stubbed there; the engine validates the returned notional and
refuses anything negative, non-finite or larger than available cash, which holds GB-21 to
its property test rather than trusting it. `Trade` and `BacktestResult` stay in `engine.py`
rather than `contracts/schemas.py`, on the same reasoning as `Fold` in GB-17. Spec §9's
GB-18 row now names all four rules, so a future reader cannot mistake them for
implementation detail.

---

## 2026-08-16 — GB-17: the embargo is exactly H, and four smaller rulings that follow

**Decision.** Walk-forward folds drop the last `H` window-ends from **every** split. Folds
are defined by **calendar months**, not bar counts. A fold whose test range is not fully
covered by the data is **dropped, not shortened**. When more folds fit than
`max_folds` allows, the **most recent** are kept.

**Reasoning.**

*The embargo is `H`, derived from `build_windows` rather than recalled from the
literature.* `build_windows` labels a window ending at position `p` with
`y[h] = target[p + 1 + h]` for `h` in `0 .. H-1`, so the label occupies positions
`p+1 .. p+H`. For that label to stay inside a split ending at `b`, we need `p + H <= b`,
so the last usable end is `b - H` and the dropped ends are `b-H+1 .. b` — **exactly `H`**.

It is not `H-1`. That off-by-one comes from thinking of the boundary bar as the one to
remove; the label starts at `p+1`, so `p = b-H` is already clean. Worked at `H=4` with a
split ending at position 9: `p=9` leaks 4 label bars, `p=8` leaks 3, `p=7` leaks 2, `p=6`
leaks 1, `p=5` is clean, and `5 == 9-4`. Confirmed empirically — with the embargo removed,
**exactly 4** training windows per fold carry a label reaching into validation.

It is also not "`H` plus something". No margin is needed on the *input* side: a validation
window reads `input_len` bars of history that reach back into the training range, and that
is not leakage — it is the model consuming past data exactly as it will live. Padding the
start of a split would discard usable data to prevent a problem that does not exist.

*Folds are calendar months.* `settings.yaml` states the plan in months, so bar counts would
mean either a config change or a module quietly reinterpreting its own config — rule 5
either way. Months also keep fold boundaries legible in the report: "test period Jul–Sep
2022" is a sentence, "test bars 1508–1570" is not. The cost is measured and small: across
the 16 folds, train ranges vary 496–501 bars (spread 5), val 57–61, test 57–61 (spread 4
each, about 7% of a quarter).

What that means for comparing metrics across folds: per-fold sample sizes differ slightly,
so a fold's direction accuracy has a marginally different standard error, and any
annualised statistic must use **that fold's actual bar count** rather than a constant
`252/4`. It does not bias the comparison, because the study design is **paired** — GB-51
runs a Wilcoxon signed-rank of each arm against persistence on per-fold values, and both
members of a pair come from the same fold and therefore the same bar count. Fold-size
variation cancels inside each pair.

*The test split is embargoed too, for a different reason than train and val.* Train and
val are embargoed to prevent **leakage**. Nothing is fitted on test, so no leakage argument
applies there. It is embargoed for **independence between folds**: without it, fold *i*'s
last test windows would be labelled with returns drawn from fold *i+1*'s test range, and
consecutive folds' scores would share outcome bars. GB-51 treats each fold as one
observation in a paired signed-rank test, so overlapping outcomes would correlate exactly
the numbers that test assumes independent. One uniform rule across all three splits is also
simply less to get wrong.

*A truncated final fold is dropped.* A fold reporting two months of test beside folds
reporting three is a comparability problem the study would inherit, and the cost of
dropping it is one fold out of thirty. `make_folds` emits a fold only when its full
calendar span is covered by the data.

*The cap keeps the most recent folds.* Thirty complete folds fit the 2016–2026 data and
`max_folds` is 16, so half are discarded and **which half is a real choice, not an
implementation detail**. Keeping the most recent means the test periods run 2022-07 →
2026-07: the market the system will actually meet on the paper account, including the 2022
drawdown and everything after it. Keeping the earliest would have tested on 2018–2022 and
left the last four years — the most recent regime, and the one the live demo runs in —
entirely unused. The price is that bars before 2020-04-13 are not used by any kept fold;
that is 2.5 years of the 10.6 available, and it is the oldest and least representative end.

*The cost of that choice has a second half, and it belongs in the report.* Keeping the
recent folds narrows **regime coverage**: the kept test periods contain no 2018 volatility
episode and no March 2020. If the study finds every arm performing similarly, one honest
reading is not "frequency structure does not help" but "the test period did not contain a
regime in which the arms differ". Spec §9's GB-57 now requires the results chapter to state
that, so it is a stated limitation rather than something a reader has to infer.

**Consequence.** Spec §9's GB-17 row said only "no timestamp appears in two folds' train
and test", which the embargo makes insufficient as a definition of done; it now states the
embargo requirement and the failing-without-it test. Spec §7.1 gains two clarifications:
the embargo itself, and the fact that **folds roll rather than block**, so a timestamp in
fold *i*'s test range does legitimately reappear in fold *i+2*'s training range — measured
at 60 shared bars. That is walk-forward working as intended, not leakage, because within
any one fold training is strictly before test. The literal cross-fold reading of the old
wording is unachievable for a rolling design and would have described a blocked one.

---

## 2026-08-14 — The Regime Guard returns to consideration at GATE 2, and only if it is green

**Decision.** The Regime Guard — the FITS reconstruction head used as an out-of-distribution
detector — stays in `IDEAS_PARKED.md` and stays unimplemented. It is reconsidered **at
GATE 2, if and only if that gate is green**. A red or partial GATE 2 closes the question
for this project; it is written up as declared future work per spec §11 and nothing else.

**Reasoning.** Sprint 1 closed on day 6 of a 14-day budget, and surplus is exactly when a
cut idea argues its way back in. Naming the condition now, while there is no pressure,
means the decision is made on a rule rather than on how the week happens to feel.

GATE 2 is the right condition and no earlier gate is. Spec §8 already says a red GATE 2
cancels FITS outright — and the Regime Guard is a FITS *extension*, so considering it
before FITS itself is proven live would be reasoning about the roof while the walls are
unbuilt. GATE 2 green means the live loop, execution and explanation all work, which is
the only state in which a one-week addition is a real option rather than a wish.

The estimate stands at one week (spec §11), and one week is precisely the buffer the
revised gate targets opened. That is the trap: the buffer exists so Sprint 4's report has
room and so GATE 2 can absorb two or three live market sessions without pressure
(`SOLO_BUILD_PLAN.md` §4). Spending it on scope converts schedule risk back into exactly
the shape it had before Sprint 1 ran ahead — with the difference that the report, which is
what the project is graded on, would be the thing paying for it.

**Consequence.** `IDEAS_PARKED.md`'s standing rule — nothing implemented before GATE 3 —
gains one named, conditional exception, and it is the only one. If GATE 2 is green, the
decision is taken then, recorded here, and requires a spec §9 task entry before a line is
written. Until then the answer is no, and "we have time now" is not an argument that
reopens it.

---

## 2026-08-14 — GB-11: `FitProvenance` enters the protocol, and the contract test gets teeth

**Decision.** Spec §4.2 gains `FitProvenance` and §4.3 gains `Forecaster.fitted`. Every
forecaster records what it was fitted on, including a no-op `fit`. Property 6 of §4.4 is
asserted with GB-10's `assert_causal` rather than a model-specific check, and §4.4 gains a
seventh property covering the protocol addition. `tests/model/test_forecaster_contract.py`
carries a deliberately broken forecaster per property.

**Reasoning.**

*Provenance belongs in the protocol, not in each model.* GB-25 audits leakage by asking
one question of a checkpoint — "does what this was fitted on overlap what it was tested
on?" — and that question has to be answerable the same way for Persistence, DLinear and
FITS or the audit becomes a manual reading of three training scripts. A protocol member is
what "uniformly, for every forecaster" means in this codebase. `FitProvenance.from_batch`
is a classmethod so the three models cannot record subtly different things and leave the
audit comparing apples to pears. `symbols` is a tuple against GB-44, which fits one model
across the universe; `fitted` is `None` before `fit`, so "never trained" is distinguishable
from "trained on nothing".

*Normalisation statistics are deliberately not duplicated into it.* Their fitted range
already lives on `ChannelStats`, which the builder produces and the checkpoint stores.
Recording the same range twice invites the two to disagree, and the audit would then have
to decide which to believe. **If GB-15 finds the checkpoint cannot link a model to the
statistics it was normalised with, adding `stats: ChannelStats | None` to `FitProvenance`
is the additive move — that is the trigger, and it has not fired yet.**

*Property 6 reuses `assert_causal`, because for a model "the future" means later windows.*
Every lag inside a window is at or before that window's own timestamp — GB-9 assembles it
that way and GB-10 proved it — so there is no within-window future for a model to reach
into. What a model can still do is compute a statistic across the batch, and because a
batch is ordered in time that pulls later windows into earlier predictions. Instance
normalisation implemented as *batch* normalisation is exactly that bug and is a live risk
for FITS's RIN stage in GB-41. Reuse also inherits both perturbation modes: `scale` sees a
leaked batch mean, `shuffle` sees a model carrying state across rows in order. A
model-specific check would have had to reinvent both, and the second one is the mode this
project nearly did without.

*Every property is proved able to fail, before a second model exists.* Persistence
satisfies all seven trivially by returning zeros, so a suite that only ever ran against it
would still pass with an assertion deleted — the same hollowness GB-10 refused. Seven
deliberately broken forecasters, each subclassing Persistence so the break is the only
difference, and a second test asserting each break is **confined** to its own property.
That confinement test caught two of my own breaks on first run: a jitter that also broke
the exactness identities, and a misreported total that also broke additivity. Both were
narrowed.

*A measured dependency between the properties.* Properties 5 and 6 both work by
recomputing and comparing exactly, so neither holds for a non-deterministic model and
neither can tell non-determinism apart from the fault it is hunting. Property 2 must pass
before 5 and 6 mean anything. Recorded in §4.4 rather than worked around, because FITS may
arrive with a stochastic layer and this is the order to debug it in.

**Consequence.** Adding a model is one line appended to `FORECASTERS`. Verified by
registering a second forecaster and re-collecting: 27 tests became 35, all passing, with no
other edit. GB-13 and GB-41 inherit all seven properties for free. `_CompleteForecaster` in
`tests/contracts/test_schemas.py` needed `fitted` added — the GB-3 structural test caught
the protocol change by itself, which is the behaviour that test exists for.

---

## 2026-08-14 — A credential reached a local commit; the rule fired, and is now a test

**What happened.** While committing GB-10, `git add -A` staged `.env.example`, which had
been edited outside the task to hold a real Alpaca **paper** API key and secret in place
of its placeholders. The blanket stage swept a file the task never touched into commit
`2dee07b`. It was caught in the post-commit review, before any push.

**Response, in the order the standing GB-6 rule prescribes.** Work stopped and the finding
was reported immediately rather than quietly fixed. The keys were rotated and revoked at
Alpaca. Only then was history rewritten: `.env.example` restored from `787e1c3` and the
commit amended to `a2915f4`. Verified afterwards — `2dee07b` is unreachable from any
branch, nothing was ever pushed, `git log -p` contains no key-shaped string, and `.env` is
untracked and ignored.

**Why the rule was worth having even though exposure was local.** The keys were paper
credentials on an account holding virtual money, never pushed, in a private repository —
the realistic loss was zero. Rotating anyway is the point: a security rule that is
followed only when the loss looks large is a rule that will be reasoned away exactly when
it matters. The cost here was five minutes, and it bought a documented, rehearsed
response. GB-59's clean-clone audit and the report's methodology chapter both want a
repository whose posture is demonstrated rather than asserted.

**The enforcement.** `tests/test_no_secrets.py` asserts that **no tracked file** contains a
credential-shaped string, by shape rather than by known value — a test that matches
today's secrets is useless against tomorrow's and would have to contain them to work. Two
patterns: a vendor key prefix (`PK`/`AK` followed by a long uppercase run) and any
unbroken 32-character-plus alphanumeric run mixing case and digits. A separate rule holds
`.env.example` to placeholders that are obviously placeholders — every honest one has
separators, so an opaque run of 16 or more characters is a paste from any vendor, not only
the two prefixes. Run against the offending blob it flags both values; against the
restored file, nothing. Six innocent strings — a git SHA, a lockfile hash, the paper
endpoint, the placeholders themselves — are asserted **not** to match, because a scanner
that fires on commit hashes gets switched off and then guards nothing. The failure message
reports file, line and match length, never the matched text: a security test that prints
what it found has moved the secret into a CI log.

**The process change, which matters more than the test.** `git add -A` is now banned in
CLAUDE.md §3, with staging by named path and a full read of `git diff --cached` added to
the §2 definition of done. The test is a backstop for the case where a human misses
something; the practice is not committing work you have not read. Blanket staging is how
the next one would get in too.

---

## 2026-08-14 — GB-10: two perturbation modes, two harnesses, and it lives in `tests/`

**Decision.** `tests/causality.py` exposes `assert_causal` and `assert_fit_isolated`. Both
run two perturbation modes by default — `scale` (multiply future rows by 1.5) and
`shuffle` (permute future rows with a frozen seed). The harness stays in `tests/` rather
than in the package, reached from anywhere in the suite via `pythonpath = ["tests"]`.

**Reasoning.**

*A scalar multiply has a blind spot, and it is exactly this project's blind spot.*
`sign(1.5 · x) == sign(x)`, exactly, so a function leaking **tomorrow's direction** passes
a purely multiplicative causality test with every value bit-identical. Direction accuracy
is one of the two headline metrics in §1.4, which makes that the single most expensive
leak the project could ship, and the one perturbation the spec named cannot see it.
Measured on real AAPL bars: `np.sign(log_return.shift(-1))` **passes** under `scale` at
all three splits and is **rejected** under `shuffle`. The second mode is not belt and
braces; it covers the case that matters most.

*The blindness runs both ways, so neither mode replaces the other.* A permutation
preserves the multiset, so it cannot move a full-sample mean or variance — the leak that
`fit_stats` on the whole frame commits. Measured: `scale` moves those statistics by
4.4e+00 relative, `shuffle` by 1.218e-15. That residue is float non-associativity, not
information: the exact comparison still rejects the leaky fitter under `shuffle`, but for
an arithmetic reason rather than an evidential one, and it is recorded here so nobody
later mistakes it for detection. Both modes run by default because each one is blind
where the other sees.

*Fitting needs its own harness.* `assert_causal` compares a value per timestamp, and a set
of normalisation statistics has no such value — it is one object for a whole range, so
there is nothing to slice at `t`. `assert_fit_isolated` therefore hands the fitter the
**whole** frame plus the boundary, exactly as a walk-forward caller holds it, and asserts
the result is unchanged when rows outside the range are perturbed. Slicing before the call
would test the harness's own slicing rather than the fitter's; handing over the full frame
catches the realistic mistake, which is a fitter that forgets to slice.

*The harness refuses to pass vacuously.* Three ways a causality test can be green while
testing nothing: a perturbation that changes no input, a comparison prefix with no rows in
it, and a prefix that is entirely NaN warm-up. All three raise `ValueError` naming the
problem, and each has its own test. A test that cannot fail is worse than no test, because
it leaves a green tick where an audit should have been.

*It lives in `tests/`, and the trigger to move it is named.* It is test infrastructure, and
`glassbox/` is governed by two contracts it would sit awkwardly inside: spec §3.4 fixes
the module list, so adding it there needs a spec amendment, and the import-linter layer
contract governs everything in the package — putting a validation harness in the layer
stack is precisely what §3.1 keeps out of the live path. In `tests/` the live path
*cannot* import it, structurally. Both known future consumers (GB-48, GB-25) are tests, so
they reach it through the same `pythonpath` entry. **If a non-test consumer ever needs it
— a script, or the dashboard — that is the moment to promote it into the package with a
spec §3.4 entry, not before.**

**Consequence.** GB-48 reuses `assert_causal` rather than writing a second harness; spec
§9 now says so on both rows. GB-25's leakage audit has `assert_fit_isolated` plus the
`ChannelStats` fitted range, which GB-10 also asserts is truthful.

---

## 2026-08-14 — The canonical channel is `close_logret`, not `close`

**Decision.** The channel carrying the close log-return series is renamed `close` →
`close_logret` everywhere it appears as a channel name: spec §4.1's canonical list and
§6.4, `settings.yaml`'s `C0_base` and `C2_hybrid`, `builder.CHANNEL_BUILDERS`,
`builder.PARITY_WARMUP`, `builder.TARGET_CHANNEL`, and the tests. The `close` column of
the OHLCV bar frame is untouched — that one really is the price, and the backtester takes
it for PnL.

**Reasoning.** The behaviour was already right; the name was not. A channel called `close`
that contains a return is a trap in the deliverable itself, not a style preference. GB-30
renders per-channel attribution as prose, and "62% of the forecast came from the close
channel" leaves a reader — a supervisor, an examiner — unable to tell whether that means
the price level or its return. Ambiguity in the explanation layer is a product defect for
a project whose entire thesis is that the box is made of glass. The two readings are not
equally harmless either: the price reading suggests a model that trades on levels, which
is precisely what §7.3 bans, so the ambiguous name invites the reader to suspect the one
error the design most carefully avoids.

Renaming costs six find-and-replaces at GB-9. After the model layer, the attribution
layer, saved checkpoints and a written report have consumed the name, it costs all of
those plus a migration for anything already serialised.

**Consequence.** `close` is now unambiguous throughout: as a channel name it does not
exist, and as a column name it is always the price. GB-47 adds `wav_a1..a3` alongside
`close_logret`, and GB-30's narratives name the channel exactly as the config does.

---

## 2026-08-14 — GB-9: the keystone, and two contract additions

**Decision.** Spec §4.2 gains `ChannelStats` and `WindowBatch.source`. Provenance travels
as a `source` column on the bar frame. Features are computed once on the full series and
sliced per fold. `min_history_bars(cfg)` is derived from per-channel declared warm-ups.
The channel named `close` carries the close **log-return** series. `build_windows`
returns a NaN target when the future is not yet known.

**Reasoning.**

*The two contract additions cost one line each today.* `ChannelStats` crosses a layer
boundary and is persisted in a checkpoint, which is exactly `WindowBatch`'s profile, so
it belongs beside it rather than in `features/`. Its `fitted_start` / `fitted_end` make
"were these statistics fitted on training data only?" answerable rather than hoped —
GB-25's leakage audit intersects that range against the fold's test range. This is the
GB-0b lesson applied early: `Attribution` was cheap to fix before three models consumed
it, and expensive after.

*Provenance as a column, not `.attrs`.* Measured: `.attrs` survive slicing but are erased
by a concat of differing values and are not stored by parquet, so the cache would lose
them and unrelated pandas operations would raise false alarms. A test that fails for
unrelated reasons teaches you to ignore it. A column survives everything and shows a
splice directly as two distinct values.

*Compute once, slice per fold.* Both are causally valid since every indicator is
trailing. Recomputing per fold would pay the warm-up at every boundary — 232 bars against
a 24-month training window is 46% of it, and a 3-month test slice could not produce a
single RSI value. It would also make the same timestamp carry different feature values in
different folds, breaking cross-fold comparability and train/live parity. What must *not*
be computed once is the normalisation statistics, which is why `fit_stats` is separate.

*The 352-bar parity warm-up, verified empirically.* A live window built from a tail of N
bars was compared against the same window built from the full 2668-bar history, on real
AAPL data:

| Tail bars | Byte-identical | Max abs difference |
|---|---|---|
| 197 | no | 4.261e-03 |
| 250 | no | 7.713e-05 |
| 300 | no | 5.960e-07 |
| **352** | **yes** | **0.000e+00** |

Every difference sits in the `rsi14` channel, as predicted. 197 — `input_len` plus the
emit warm-up — leaves a permanent train/live gap of about 0.06 RSI points that GB-27
would never see, because the parity test compares builder outputs on one frame rather
than on histories of different length. 352 is where the seed's residue falls below
float32 resolution and the gap becomes exactly zero.

> **SUPERSEDED 2026-08-18 by GB-27 — and the way it was wrong is the instructive part.**
> The table above is real, and it is **one symbol at one timestamp**. Swept over five
> symbols and twenty-five timestamps, 352 bars produced byte-identical windows in **32 of
> 125 pairs**; AAPL at that timestamp was one of the 32. The derivation bounded the seed's
> **weight** below 1e-7, which is only sufficient if the seed **difference** is at most 1,
> and it is not — it is a difference of average gains in price units. Near RSI 50 a float32
> ulp is 5.95e-06 and the measured residual was 3.815e-06, the same order of magnitude:
> **352 was marginal by construction.** The floor is now **445**, from a 1e-10 target
> (`(13/14)^311`), and the standing assertion sweeps rather than samples. See the GB-27
> entry.

*`close` is a log return.* Spec §4.1 fixes the model input series as log returns and §7.3
bans price levels as a headline quantity, so the channel named `close` is the modelled
quantity *of* close. Raw prices stay in the bars frame, where the backtester takes them
for PnL. — *Superseded the same day: the behaviour stands, the name does not. See
"The canonical channel is `close_logret`" above.*

*A NaN target for the live window.* At the most recent bar the next H returns do not
exist. NaN says so; a zero would be a fabricated observation and an exception would force
a second code path, which is the one thing this module exists to prevent.

**Consequence.** GB-26 must request `min_history_bars(cfg)` bars, now in its "Done when".
GB-47 must declare `wav_a1..a3: 64` in `PARITY_WARMUP`, after which the number updates
itself. `C2_hybrid` raises until then rather than silently degrading to C0.

---

## 2026-08-14 — GB-8: RSI emits nothing until its seed has decayed; vol_z is source-safe

**Decision.** `rsi14` holds its first 77 rows NaN, not the conventional 14. `vol_z` stays
in the channel set for both training and live. Indicator periods are module constants.

**Reasoning.**

*RSI warm-up.* Wilder's average is recursive: it is seeded with a simple mean of the
first 14 changes and then updated as `(prev * 13 + new) / 14`. The seed never leaves —
its weight decays geometrically at `(1 - 1/14)` per bar, so the conventional first
published value at bar 14 is *entirely* seed. Holding rows until the seed's weight falls
below 1% gives 63 further bars, so 77 in total: `RSI_WARMUP = RSI_PERIOD +
ceil(log(0.01) / log(1 - 1/14))`, computed rather than chosen. The cost is 77 of 2668
bars, under 3% of the history. The alternative — emitting values that are technically
defined but still remembering their own warm-up — would put a subtly different quantity
into the first weeks of every fold, and no test would see it.

*vol_z across sources.* Measured over all 2668 overlapping bars rather than assumed. The
Alpaca/yfinance volume ratio averages 1.04-1.09, sd 0.045-0.076, and drifts by era:
~1.08-1.12 (2016-2019), ~1.05-1.13 (2020-2022), ~1.006-1.014 (2023-2026). The drift is
real but multi-year, so within any 20-bar window the ratio is constant and cancels in the
z-score. Residual divergence between a z computed from each source: median 0.016-0.039,
p95 0.15-0.29, against a z whose own sd is 1.06 — a few percent of a standard deviation.
The tails reach 1.3-4.6, but every one of those lands on 2018-05-02 or 2018-05-03 for all
five symbols at once, which is a vendor-side event on two days, not per-symbol noise.

**Consequence.** vol_z is safe in the live channel set, on one condition that is now
stated in the module docstring: **a single window must be assembled from a single
source.** Splicing cached yfinance history onto live Alpaca bars inside one 20-bar window
would put a ~5% level step inside the normalising window and fabricate a z-score spike.
GB-9 must keep assembling each window from one loader. The two 2018 dates are worth a
line in the data-quality appendix of the report; they are visible in `vol_z` and in
nothing else.

---

## 2026-08-14 — GB-7: one normaliser defines the schema; SIP + Adjustment.ALL; no live cache

**Decision.** `historical.normalise_bars` is the single definition of the bar schema, and
`data/live.py` calls it. Live requests pin `feed=SIP` and `adjustment=ALL`. Live does not
cache. The shared normaliser also pins three things the two vendors disagree about:
volume dtype (`int64`), index resolution (`ms`), and index freq (`None`).

**Reasoning.** Measured, not assumed, on this account:

* **Adjustment.** Alpaca's default is `RAW`. Across AAPL's 2020-08-31 4:1 split, RAW
  closes 499.75 where yfinance's adjusted series closes 121.08; `SPLIT` alone closes
  124.94 because it leaves dividends in. Only `ALL` matches. Taking the default would
  have produced a *perfect* schema match with prices off by a factor of four — the exact
  failure GB-27 would not catch, since shapes and dtypes would agree.
* **Feed.** SIP returns 2669 daily bars back to 2016-01-04; IEX returns 1521 back to
  2020-07-27. Both exceed the 120-bar input window, so either would serve the live loop,
  but SIP is the consolidated tape and is what yfinance reports. Pinning it means a lost
  subscription fails loudly instead of silently switching to a single venue's prices.
* **No cache.** GB-4 caches because a study needs a fixed snapshot. Live has the
  opposite requirement: a cached bar served into a trading decision is a stale price.
  The same reasoning produces opposite designs because the two modules answer different
  questions.
* **Schema by construction.** Two implementations that agree today drift tomorrow. One
  function that both call cannot. The dtype and index pins came out of measurement:
  Alpaca reports volume as `float64` and stamps bars at midnight New York in microsecond
  resolution; yfinance reports `int64` at the bare date in milliseconds.

**Consequence.** Prices from the two sources agree to under 1 bp; volume differs by
34-111 bps and always will, so no feature may assume cross-source volume equality.
During market hours Alpaca returns an in-progress bar for the current day — see the
ruling below.

**Ruling, same day: the live loop uses completed bars only.** The in-progress bar for
the current session is dropped. The model is trained exclusively on completed daily bars,
so feeding it a half-formed close puts the live input distribution outside the training
distribution — and GB-27 would still pass, because the window *shapes* match. This is the
class of failure that parity tests cannot see, which is why it is decided now rather than
in Sprint 3: it fixes the `as_of` semantics GB-9 must implement.

Implemented in GB-26, not before, and required to be visible rather than incidental: a
function named for what it does (`drop_incomplete_bar` or similar), a test that
constructs an in-progress bar and asserts its exclusion, and a log line naming the
dropped timestamp. Spec §9 carries this in GB-26's "Done when" so it sits in the
acceptance criterion, not only here.

---

## 2026-08-14 — GB-6: the paper endpoint is a constant, and the guard lives in the config layer

**Decision.** `PAPER_ENDPOINT` is a module constant in `config/loader.py`.
`alpaca_credentials()` now defaults `base_url` to it when `ALPACA_BASE_URL` is unset
(it previously returned `None`), and a new `require_paper_endpoint()` raises unless the
resolved endpoint is exactly that. `scripts/smoke_alpaca.py` calls the guard before it
constructs a client. Operational scripts live in `scripts/`, outside the package.

**Reasoning.** Three small choices, one theme — put the safety check where it can be
tested and cannot be forgotten:

* The endpoint is not a tunable, so it is not a `settings.yaml` key. Spec §5 froze that
  schema, and a value whose whole purpose is to be immovable does not belong in a file
  people edit. This is the second deliberate exception to CLAUDE.md rule 5, after
  `LARGE_MOVE_LOG_RETURN`; both are flagged rather than buried.
* Defaulting an unset endpoint to `None` pushed the decision to every caller. Defaulting
  it to paper makes the safe value the default, so forgetting to configure anything is
  safe rather than undefined.
* The guard sits in the config layer, not in the script, because a script cannot be
  imported by the test suite. Nine tests now cover the accept and refuse paths, including
  a lookalike host (`paper-api.alpaca.markets.evil.example`) and plain HTTP.

`scripts/` is documented in spec §3.4 as sitting outside the package: scripts import
`glassbox`, nothing imports them, they are not packaged, and they are outside the layer
contract. They are deliberately *not* in `SPEC_MODULES`, which guards the importable
tree; the scaffold test only walks `glassbox/`, so no exemption was needed — only a line
in the spec so the next session does not put library code there.

**Consequence.** Any future broker-touching code calls `require_paper_endpoint()` first.
`tests/config/test_config.py` also now scans `scripts/` for direct environment access, so
the single-source-of-truth rule extends beyond the package.

---

## 2026-08-14 — Considered and rejected: caching site-packages in CI

**Decision.** CI keeps `actions/setup-python`'s pip cache and installs from
`requirements.lock` on every run. No site-packages or virtualenv caching is added.

**Reasoning.** Measured, not assumed. Run 176fe01 (cold cache) took **2m45s**; run
067d10f (warm cache, lockfile unchanged) took **2m12s**. The 33-second difference
confirms what the mechanism predicts: `setup-python`'s pip cache stores the HTTP download
cache, so torch is not re-downloaded but every wheel is still reinstalled, and
installation dominates. Caching site-packages keyed on the lockfile hash would skip
installation, but it breaks in subtle ways — absolute paths in `.pth` files, editable
install metadata — and CI that fails incomprehensibly costs far more than CI that takes
thirty seconds longer. At roughly 40 remaining pushes to submission, the whole
optimisation is worth about 20 minutes.

**Consequence.** Expect ~2m10s per push. If a future change makes CI materially slower —
a heavier dependency, a long test suite — revisit with fresh measurements rather than
from memory of this entry.

---

## 2026-08-14 — GB-4: the parquet cache is a snapshot, and downloads are per-symbol

**Decision.** `load_history` downloads one symbol per yfinance call rather than
requesting the universe in bulk. A cached symbol is returned exactly as stored, however
old; there is no incremental top-up. Refreshing is explicit, via `force_refresh=True`.

**Reasoning.** Per-symbol keeps one parsing path — yfinance's column layout changes shape
with the number of tickers requested — and maps one-to-one onto the per-symbol cache
files, so a partial cache fetches only what is missing and one bad ticker cannot fail the
universe. Five daily requests cost about eleven seconds, once.

Staleness is the more consequential half. Spec §7.4 caches the study's data to parquet so
the grid runs against a fixed snapshot, and GB-59 requires a clean clone to reproduce
reported results. A loader that silently extended its data each day would make two runs
of the same study disagree with no code change, and the disagreement would be invisible
in the output. Explicit refresh makes the snapshot boundary a decision someone took,
rather than a side effect of what day it happens to be.

**Consequence.** Re-running the study tomorrow uses today's data unless the cache is
refreshed deliberately. Every cache hit logs its last bar, so staleness is visible in the
run log. The live loop does not use this module — it reads `data/live.py` (GB-7).

**Enforcement, added the same day.** A log line is not enough: it relies on someone
remembering to refresh in the last week of the project, which is exactly the week nobody
remembers anything. So the snapshot date is carried into the deliverable instead. Spec §9
now requires `results.csv` to carry a `data_snapshot_last_bar` column (GB-49) and the
generated report to print it in the summary table header (GB-52). A report built on
August data then says so on the page the supervisor reads, which turns a discipline
problem into a visible fact. Implemented in GB-49 and GB-52, not before.

---

## 2026-08-14 — Python floor raised to 3.12; CI installs from `requirements.lock`

**Decision.** `requires-python` becomes `>=3.12`, CI pins `python-version: "3.12"`, and
ruff and black target `py312`. CI installs `pip install -r requirements.lock` followed by
`pip install -e . --no-deps` instead of resolving from `pyproject.toml`. `CLAUDE.md` §4
and spec §12 now say Python 3.12+.

**Reasoning.** CI went red the moment dependency lower bounds were added. `numpy>=2.5.2`
and `scipy>=1.18.0` — the versions this project is developed against — both declare
`requires_python >= 3.12`, so pip cannot satisfy them on the 3.11 floor the workflow
pinned, and the job died in the install step before reaching torch. The 3.11 floor was
never real: it was a claim in `pyproject.toml` that the dependency set could not honour,
and CI was the only thing testing it. Raising the floor makes the metadata true.
Installing from the lockfile closes the second half of the gap — CI and a developer
machine now install the identical set, so GB-59's clean-clone audit is exercised on every
push rather than being discovered in the report week.

**Consequence.** Python 3.11 is no longer supported; the project requires 3.12+, which
is what the development machine already runs. A dependency change now means regenerating
`requirements.lock`, or CI installs the old set and the failure appears as a confusing
test error rather than a resolution error. The lockfile is frozen on 3.12, so the CI
Python version and the lockfile must be changed together.

---

## 2026-08-14 — GB-3: the validation harness sits above the live path in the layer contract

**Decision.** The import-linter layers contract, highest first, is: `experiments`,
`backtest`, `dashboard`, `explain`, `engine`, `model`, `features`, `data`, `contracts`,
`config`. A second `forbidden` contract stops `live_loop` and `replay` — top-level
modules the layers contract does not reach — from importing `backtest` or `experiments`.
`smoke_offline` is deliberately exempt: it is the harness's own entry point.

**Reasoning.** Spec §3.1 draws the harness as cross-cutting rather than as a layer, so
it has no natural rung on the ladder. What the spec actually requires is asymmetric: the
harness must be free to drive models, features and data — a walk-forward run does
exactly that — while the live path must never reach into it. Placing the harness at the
top gets both from one contract, because a layers contract already forbids lower layers
from importing higher ones. The alternative, putting it at the bottom, would have
inverted the rule and let the live loop import the backtester.

**Consequence.** A backtest module may import anything below it. Nothing in the live
path can import a backtest or experiments module, and the tests prove it: a deliberate
`live_loop -> backtest.engine` import breaks the forbidden contract, and a deliberate
`data -> features` import breaks the layers contract. `reference/` is outside both
contracts by construction, since `root_package = "glassbox"`.

---

## 2026-08-14 — Dependencies pinned to lower bounds; `requirements.lock` committed

**Decision.** Every runtime and dev dependency in `pyproject.toml` carries a `>=` lower
bound set to the version resolved in the development environment (pandas 3.0.5,
numpy 2.5.2, torch 2.13.0, and so on). `requirements.lock` — the full 90-package
`pip freeze` of that environment — is committed alongside it. A clean clone that must
reproduce reported results installs the lockfile; `pip install -e ".[dev]"` remains the
day-to-day path.

**Reasoning.** The environment resolved to pandas 3.0.5, a major release with breaking
changes, and to numpy 2.5.2 and torch 2.13.0. Unversioned dependencies make GB-59's
acceptance criterion — a clean clone reproduces the reported results — unachievable,
and the failure would surface in October during the report week rather than now. Lower
bounds document what the code was written against; the lockfile is what actually
reproduces. Both are needed: bounds alone drift, a lock alone hides the intent.

**Consequence.** GB-59 verifies the clean clone against `requirements.lock`, and GB-12's
README must point a newcomer at it. The lockfile is regenerated whenever a dependency is
added or deliberately upgraded, and that regeneration is itself an entry in this file.
`pip freeze --exclude-editable` is used so the lockfile carries no absolute path to this
machine's checkout.

---

## 2026-08-14 — GB-1: the declared dependency set

**Decision.** `pyproject.toml` declares twelve runtime dependencies — pandas, numpy,
pyyaml, torch, yfinance, alpaca-py, PyWavelets, scipy, streamlit,
pandas-market-calendars, python-dotenv, pyarrow — and five dev extras: pytest,
hypothesis, import-linter, ruff, black. The formatters live in the extra so that
`pip install -e ".[dev]"` is the single command that produces a working local
environment, and CI installs exactly what a developer installs. Versions are unpinned
for now; GB-59 (reproducibility audit) is where a lock is added if a clean clone does
not reproduce.

**Reasoning.** This is the baseline set the spec's architecture already implies — one
library per named responsibility, nothing speculative. Recording it here satisfies the
`CLAUDE.md` §4 rule that a dependency needs a line in this file, so later additions are
visible as additions rather than lost in a diff.

**Consequence.** Anything beyond this list requires a new entry here before it is
installed. `reference/` is excluded from packaging, linting and pytest collection in the
same file, so its dependencies are never ours.

---

## 2026-08-14 — Sprint-table weekday labels corrected against the 2026 calendar

**Decision.** `SOLO_BUILD_PLAN.md` §7 dated the code freeze "Fri 3 Oct". 3 October 2026
is a **Saturday**; the row now reads "Sat 3 Oct". All 52 dated cells across the four
sprint tables were checked against the real calendar — this was the only error. The
freeze date itself (3 October) is unchanged.

**Reasoning.** The code freeze is the one deadline in the plan that converts unfinished
work into declared future work. A day-name that disagrees with the date invites the
reader to resolve the conflict in whichever direction suits them, which in practice
means Saturday's work quietly leaking past the freeze.

**Consequence.** Sprint 4 day 8 (Sat 3 Oct — GB-49/50/51) and the freeze now agree:
the study runner, COF sweep and significance tests land *on* the freeze day, not after
it. No schedule change; the plan already had it that way.

---

## 2026-08-14 — `Attribution.per_lag` becomes optional; `Attribution` becomes keyword-only

**Decision.** In spec §4.2, `per_lag: np.ndarray` becomes `per_lag: np.ndarray | None = None`,
documented in the docstring as optional because GB-31 is cut. The dataclass is
additionally declared `kw_only=True`.

**Reasoning.** GB-31 (per-lag heatmap) is cut from scope, but the frozen schema declared
`per_lag` as required — a contract that obliges every implementation to populate a field
nobody consumes. The alternatives were filling it with zeros, which fakes data an
exactness test could later assert against, or deleting the field, which would make
adding the heatmap a contract change. Making it optional keeps the shape of the future
feature at zero present cost. `kw_only=True` is a mechanical consequence: a dataclass
field with a default may not precede fields without one, so in-place optionality
requires either reordering the frozen field order or constructing by keyword. Keyword
construction is the smaller change and the better style for a five-field record.

**Consequence.** Implementations return `per_lag=None`. `Attribution` is always
constructed with named arguments — positional construction now raises `TypeError`.
**Do not "simplify" `kw_only=True` away.** It is load-bearing: without it, a defaulted
`per_lag` sitting before the three fields that have no default is a `TypeError` at
import time. Reordering the fields to avoid that would change a frozen contract for no
benefit, and `forecast_total` cannot take a default because it is the exactness target. The
exactness contract (§4.4, GB-33) is untouched: it tests `per_channel` and
`forecast_total` only. GB-31 can be un-cut later without a contract change.

---

## 2026-08-14 — `SOLO_BUILD_PLAN.md` governs execution; the Jira 70/30 split is a formal record

**Decision.** One developer writes every line. `SOLO_BUILD_PLAN.md` is the real build
order. The `Own` column in spec §9, the load-balance table, and the two-person split in
`CLAUDE.md` §6 exist only for the formal academic record and are ignored when planning
work. `CLAUDE.md` §6 has been replaced with a single line saying exactly that.

**Reasoning.** Three documents disagreed about who builds what, and `CLAUDE.md` is read
at the start of every session. An agent that believes a task belongs to someone else may
defer, stub, or wait on it — the one failure mode a solo build cannot absorb.

**Consequence.** Task ownership carries no scheduling meaning. Frozen contracts (§4)
remain in force, now justified by the cost of changing an interface mid-build rather
than by parallel work. `glassbox_jira_tasks.csv` and the submitted documents keep the
70/30 split unchanged.

---

## 2026-08-12 — FITS adopted as the spectral core; DLinear retained as baseline

**Decision.** The model layer holds three forecasters: `Persistence` (reference),
`DLinear` (established baseline, already understood), and `FITS` (new spectral core).
Wavelet channels remain, repositioned from "the core contribution" to
"the explicit arm of the comparative study".

**Reasoning.** FITS and wavelets are not competitors — they sit in different layers.
FITS is a model; wavelets are features. Keeping DLinear is what makes FITS measurable;
without a baseline there is no claim to make. This framing turns an assumption
("wavelets help") into a measurable question ("explicit or implicit frequency
decomposition — which helps more, if either?").

**Consequence.** The study grid becomes 3 models x 2 feature configs, with the
`FITS x C2_hybrid` cell deliberately empty (see spec §6.4).

---

## 2026-08-12 — Execution window compressed from 14 weeks to 8

**Decision.** The plan is rebuilt around 15 Aug – 10 Oct 2026: four two-week sprints,
three hard gates.

**Reasoning.** The original 14-week plan is not executable in 8 weeks.

**Consequence.** The ablation grid shrinks; `NLinear` becomes optional; the
Regime Guard moves to declared future work; Sprint 4 carries the entire report
and nothing technical may spill into it.

---

## Template

## YYYY-MM-DD — <short title>

**Decision.**

**Reasoning.**

**Consequence.**
