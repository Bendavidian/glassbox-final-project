# GlassBox Trader — Phase 2 Expansion

**Opened:** 23 August 2026 · **Closes:** 14 September 2026 · **Submission:** 10 October 2026

> Add this file to the repository root and to the Claude Project knowledge. It is the
> authority for everything built after GATE 2, and it sits alongside
> `GLASSBOX_PROJECT_SPEC.md` rather than replacing it. Where the two conflict, the spec
> wins on contracts and this file wins on scope.

---

## 1. Why this phase exists

The core system was complete on 23 August, roughly three weeks ahead of the plan. The
project could be submitted as it stands: a working autonomous system, a leak-free
methodology, a comparative study with null controls, and bit-identical reproducibility
from a clean clone.

Phase 2 is not about adding features to a finished thing. It exists because three
specific weaknesses were **measured** during Phase 1 and can be closed with the time
that remains:

| Weakness, as measured | Task |
|---|---|
| Determinacy at 2.09×, and the gross-exposure cap numerically redundant at 5 symbols — it bound once in 16 folds | GB-61 |
| No answer to "was the null result just bad feature selection?" — the study has two feature arms and both were ours | GB-62 *(deferred; see GB-66)* |
| 86% of FITS's learned response is grid geometry, measured — but on one architecture, so it is a claim about FITS or about the method and we cannot tell which | **GB-66** |
| The dashboard is static where the data is live, and shows nothing while a position is open | GB-63 |

GB-64 and GB-65 are product and demonstration work, not corrections.

**The test every Phase 2 item had to pass:** does it close something Phase 1 measured, or
is it a feature we would like to have? Two ideas failed that test and are recorded in §6
with their reasons.

---

## 2. The tasks

### GB-61 · Universe expansion, 5 → 20 symbols · 5 SP

Three problems close at once, and each was measured rather than assumed.

| Symbols | Windows/fold | Equations | Determinacy | Gross if all held |
|---|---|---|---|---|
| 5 (current) | 2,480 – 2,505 (mean 2,492) | 9,920 – 10,020 | 2.07× – 2.09× | **0.50 — exactly the cap** |
| **20** | **9,920 – 10,020 (mean 9,968)** | **39,680 – 40,080** | **8.27× – 8.35×** | **2.00 against a 0.50 cap** |

> **Measured 24 Aug 2026, and the first draft of this table was one fold of sixteen.** It
> read 2,505 → 10,020 and 2.09× → 8.35×, which is **fold 1** — the largest of the sixteen,
> not the typical one. Counts fall monotonically across the grid as the calendar shortens
> the training span, so the honest figures are the ranges above. Taken **off the batch
> `train.py` builds**, per the standing rule that window counts are read from the batch and
> never reconstructed by slicing the frame first.
>
> **The ×4 was exactly right and only the base was wrong.** Every fold's 20-symbol count is
> its 5-symbol count times 4.000 — all twenty symbols share one index, so the scaling is
> exact rather than approximate. The determinacy improvement stands as claimed; the level
> it starts and ends at is 1% lower than written.

**Determinacy.** 4,800 parameters against 10,020 equations is thin. Four times the
samples for the same parameter budget is the cheapest capacity improvement available,
and it costs no model change.

**The risk layer stops being inert.** GB-21 found that `5 × max_position_pct 0.10 = 0.50
= max_gross_exposure`, so the gross cap is numerically redundant and bound exactly once
across 16 folds. GB-57 currently has to report a risk layer that is correct, enforced and
doing nothing. At 20 symbols it binds well below the point where every position is
held — but **how often it actually binds is the measurement, not the claim.** The
acceptance criterion below asks for the fold count, and that number is what GB-57
reports.

**Ranking becomes a selection.** `top_k = 2` out of 5 is barely a choice. Out of 20 it is
one, and the cross-sectional ranking in `rank.py` starts carrying weight.

**Split across GATE 2, and the split is the whole of it.** This task is scheduled in
the same window as the gate sessions, which would otherwise contradict this file's own
claim to govern only what is built *after* GATE 2. It does not, because the task has
two halves with different blast radii. **The preparatory half runs alongside the gate**
- caching, the quality report, the parity sweep and the `min_history_bars`
verification - because none of it touches the deployed path: the live loop reads
Alpaca through `data/live.py` and never reads `data_cache/`. **The `universe:` flip and
the grid re-run wait until the sessions are done**, because editing the deployed
configuration under a running multi-day loop is the mutation GB-59 recorded as its
first defect, and `live_loop.config_drift` warns at session start without reloading.

**Universe:** extend to 20 large-cap US equities with full history from 2016. Liquidity
and history are the only criteria — no sector or performance screening, because a
universe chosen on outcomes is a universe chosen with hindsight. State the selection rule
in the spec before choosing the names.

**Done when:**
- 20 symbols cached, quality-checked, and passing the same GB-5 report
- `min_history_bars` unchanged (it is a max, not a sum) — verified, not assumed
- The parity sweep passes across all 20 symbols at the current floor
- The gross-exposure cap is measured to bind in at least one fold, with the count reported
- Grid wall time reported; ~88 minutes expected against 22 today
- Every Phase 1 headline claim re-measured at 20 symbols, at all three anchors and both
  null controls, with any claim that changes sign or significance called out explicitly

> **This may change the results.** More symbols means more training data and a real
> cross-sectional selection. If a claim moves, that is the finding — report it as a
> measured consequence of universe size, not as a correction to Phase 1.

---

### GB-62 · The C3 feature arm · 3 SP

The study has two feature configurations and both were chosen by us. Nothing in the
report currently answers a reviewer asking *"is your null result just bad features?"*

Add `C3_extended`: `C0_base` plus a set of standard technical indicators the literature
uses and this project does not — at minimum ATR, Bollinger position, OBV or a volume-flow
measure, and a longer-horizon momentum. Declare each one's `PARITY_WARMUP` and let
`min_history_bars` update itself.

Then run it as another spoke on the star, against DLinear, with its own null controls.

**Done when:**
- Every indicator is pure, trailing-only, and passes the GB-10 causality harness in both
  perturbation modes at three splits
- Each has a hand-computed fixture test, expected values written as literals
- `C3_extended` appears in `results.csv` with its null-control twins
- GB-57 carries the answer plainly: whether more features change the direction result

> The likely outcome is that they do not, and that is the point. A null result with three
> feature configurations tested is a different claim from a null result with two, and the
> difference is what a reviewer will ask about.

---

### GB-66 · WITS — the wavelet transform as the model core · 8 SP

**This replaces GB-62 in the schedule, not in addition to it.** The C3 arm asked "were
the features bad?"; this asks a sharper question and answers a measurement Phase 1 made.

Today the wavelets are *features* — `wav_a1..a3` fed as input channels. WITS moves the
transform *inside the model*, in the position FITS gives to the rFFT.

**Why this is a test rather than another arm.** GB-46 measured that 86% of FITS's learned
frequency response is grid geometry rather than market structure: the white-noise curve
correlates with the real one at +0.9485, the 8-day trough is the half-integer bin, and the
mechanism is `frac(η·k) = frac(k·H/L)`. The complex layer spends most of its capacity on a
deterministic resampling operator.

**A DWT does not have that problem.** Wavelet coefficients are localised in time, so
extending the series does not require remapping a global basis. The interpolation cost is
specific to Fourier.

**The falsifiable prediction, recorded before implementation:**

> If the geometry critique is correct, WITS will not show ~86% data-independence under the
> null control. If it does, the critique of FITS is wrong — and that is worth as much as
> confirming it.

**The pipeline**, mirroring §6.2 so the two are comparable:

```
x : (B, L)  log returns
  → RIN, as FITS. Same normalisation, so the comparison isolates the transform.
  → causal DWT, db4, J levels, over the input window
  → keep the retained levels; discard the fastest, as the LPF does
  → ONE REAL LINEAR MAP PER LEVEL, coefficients → extended coefficients
  → inverse DWT → (B, L+H)
  → inverse RIN
  → split: [:L] backcast, [L:] forecast, B+F supervision as FITS
```

**Parameter count, computed:** at L=120, db4, J=3, keeping a3+d3 gives ~30 coefficients
mapping to ~31, so ~930 real parameters against FITS's 1,200 and DLinear's 4,800. Real
weights throughout, so the reals-versus-complex counting question does not arise. Report
the measured count.

**Three design questions to decide and report, not assume:**

1. **How the extension works.** FITS zero-pads and inverse-transforms. A DWT has no
   equivalent — the coefficients must be mapped to a longer series. One linear map per
   level is the obvious choice and keeps the levels separable for attribution, but say
   why, and say what the alternative would have been.

2. **Boundary handling.** Symmetric extension mirrors the past forward. That is not
   leakage — no future information exists in it — but it is a strong assumption at exactly
   the edge the forecast depends on. Measure it: compare symmetric, zero and periodic
   modes, and report the effect on the forecast rather than choosing on convention.

3. **Shift variance.** The standard DWT depends on where the window starts. SWT/MODWT is
   shift-invariant but redundant, which changes the parameter count and the capacity
   argument. Treat it as an axis, not a default, and measure both.

**Attribution stays exact, and it needs a field that does not exist yet.** Every stage is
linear, so `forecast_matrix` works unchanged and `Attribution.from_terms` closes as it
does for FITS. But wavelet attribution has **two axes** and `per_frequency` is not one of
them: storing a *level* in a field named for a *frequency* would be the two-places defect
committed inside a single schema. Both axes already have homes. **`per_level` is added to
the §4 contract as optional**, for the band totals, and **`per_lag` - retained in the
schema precisely so a view could be added later without a contract change - carries the
time localisation**, which is the whole reason WITS's explanation beats FITS's. One
`DECISIONS.md` entry, before GB-66 starts.

**And the explanation improves in kind, not only in wording:** wavelet coefficients are localised in time and Fourier
coefficients are not. The panel can say

> "34% from the 8–16 day band, and it happened in the last three days"

where FITS can only say which cycle. That is the *when* §1.4 promised and Fourier
structurally cannot give.

**Done when:**
- Passes the full `Forecaster` contract test unchanged, registered in `ALL_FORECASTERS`
- Causal at the window level, verified by GB-10's harness in both perturbation modes
- Attribution per level in `per_level`, summing to the forecast through `from_terms`,
  with the time localisation in `per_lag` - a figure FITS has no equivalent for
- **The null control is run and the correlation between the learned response on real data
  and on white noise is reported against FITS's +0.9485.** This is the acceptance
  criterion that matters; performance is not.
- Boundary mode and shift-variance measured as axes, both in `results.csv`
- Measured parameter count replacing the ~930 estimate in the spec

> Expect the trading result to be null, like everything else. The finding is whether a
> second architecture reproduces the geometry pathology or escapes it, and that is a
> statement about FITS as published rather than about this project's models.

---

### GB-63 · The live dashboard pass · 5 SP

The dashboard renders correctly and does not feel live. Everything below is either
genuinely live data or a threshold that changes what a row says.

**The colour system extends from two roles to three:**

| Role | Colour | Where |
|---|---|---|
| Chrome | orange | labels, rules, borders, active states |
| Data encoding | blue ramp | attribution, spectral, frequency response — the ramp means a quantity |
| **Status** | green / red | P&L, price change, protection state, live/stale |

**The rule that keeps it coherent:** status colour never appears inside a data-encoding
chart, and never carries information alone. Every green or red is redundant with a sign
and a glyph, so the greyscale test survives in its true form — *no information is carried
by colour alone* — rather than as a blanket ban on the pair.

**Palette constraint:** the chrome accent is already orange-vermillion, so an ordinary red
collides with it. The loss colour must lean magenta-crimson, clearly not orange; the gain
colour must lean jade-teal, clearly not the blue ramp's endpoints. Both muted. Propose
hexes with their perceptual distance from the existing roles and get sign-off before
committing.

**Live elements — each because the underlying number moves:**
- Session equity curve from cached snapshots. Equity changes intraday as prices move, and
  nothing currently shows it.
- Per-position sparkline with the stop and target as a band
- Cycle countdown as a depleting rule
- P&L in status colour with an arrow; distance to stop with threshold behaviour: beyond
  half, ordinary; inside half, emphasised and marked; inside a quarter, the row states
  that a stop fill is near and what will happen when it fills
- A newly written decision marks itself new for one refresh

**Caching:** local reads at 5s, broker reads at 30s, **separate caches**. A shared TTL
either hammers the API or freezes the countdown. The dashboard competes with the live loop
for the same account from another process.

**Staleness:** a failed broker read keeps the last value, dims it, and stamps the age. It
does not blank — a position you cannot see is worse than one you can see is old. Above two
heartbeat intervals it escalates from a staleness note to a loop-not-responding state in
the header, because the question changes.

**What must not happen.** The price chart advances once per trading day: the model consumes
completed daily bars (GB-26), and GB-7 measured that this subscription refuses
`get_stock_latest_quote` on SIP, so there is no intraday price even if we wanted one.
Nothing pulses, sweeps or animates to look alive. A number moves because it moved. The
chart carries `LAST COMPLETED BAR <date> · ADVANCES ONCE PER TRADING DAY` in the same
register as the reliability line, not as small print.

The right frame is an **instrument panel**, not a trading terminal. Instrument panels are
live and technical, and every reading measures something real.

---

### GB-64 · Sentiment as a Co-Pilot annotation · 3 SP

**Ruling, and it is not negotiable: sentiment never enters the model and never enters a
backtest.** See §5 for the full reasoning.

What is built instead: after the bot has decided and explained, the Co-Pilot panel shows
recent headlines and context for the symbol, **explicitly labelled as information that did
not enter the decision.**

**Done when:**
- **A new `forbidden` import contract names the sentiment module and lists
  `glassbox.backtest` and `glassbox.experiments` as forbidden importers.** This is an
  acceptance criterion rather than a note because **the contract as it stands does not
  prevent this**: the layers contract places `glassbox.data` *below* `backtest` and
  `experiments`, so a provider living under `data/` is freely importable by exactly the
  two modules this task forbids, and the existing `forbidden` contract only points the
  other way - it keeps `live_loop`, `replay` and `records` out of the harness. Until
  that contract exists the claim is discipline in a contract's clothes.
- Sentiment data is fetched only in the live path, never in `study.py`, `train.py` or
  anything under `backtest/` - enforced by the contract above.
- No `DecisionRecord` field carries it, so it cannot reach `results.csv`
- The panel states, in the interface and not only in a docstring, that the annotation is
  context for the operator and not a model input
- A test asserts that the study's inputs contain no sentiment field, and that removing the
  sentiment provider entirely changes no number in `results.csv`

---

### GB-65 · Demonstration · 3 SP

The four architecture-report screenshots, the vision video's shot 6 recording, and two
full rehearsals. Fold 13 then fold 1 — the good quarter first so the bad one is what the
room remembers.

The spectral-panel capture requires `model.active: fits`, which is not the deployed arm.
Its caption must state that FITS is not deployed and that the study found it beats neither
reference. A screenshot showing a capability the deployed system does not run, without
saying so, is the misreading the panel itself exists to prevent — committed by the report.

---

## 3. Schedule

| Window | Work |
|---|---|
| 24–30 Aug | GATE 2 live sessions · GB-53 · **GB-61 preparatory half only** (cache, quality, parity, `min_history_bars`) · **GB-66 begins** |
| after the sessions | **GB-61 completes**: the `universe:` flip and the grid re-run |
| 31 Aug – 7 Sep | **GB-66 completes** · GB-63 |
| 8–14 Sep | GB-64 · GB-65 · re-run the full grid at 20 symbols with three model arms |
| **15 Sep** | **Report writing begins. Phase 2 closes in whatever state it is in.** |
| 15 Sep – 5 Oct | GB-55, GB-56, GB-57, GB-58 |
| 6–9 Oct | GB-59 re-run · rehearsals |
| 10 Oct | GB-60 submission |

**15 September is a hard boundary, not a target.** Five of the remaining tasks are
writing, GB-57 is the largest and the one that carries the grade, and scope always
expands while time does not. Anything unfinished on 15 September goes to
`IDEAS_PARKED.md` with what was done and what was left.

---

## 4. What does not change

Every Phase 1 ruling stands. In particular:

- **The frozen contracts.** New work implements them; it does not amend them without a
  `DECISIONS.md` entry.
- **Only something that runs is a mechanism.** A fact stored twice needs something making
  the copies equal; a test skippable by a flag, cache or marker is not a mechanism; a
  description of what the code should do, in a comment or a docstring or a plan, is not a
  mechanism. Five instances in Phase 1, three of them caught by a test written for
  something else.
- **The three-reference reporting rule.** MAE against persistence, direction against the
  always-long bar, trading metrics against buy-and-hold.
- **Both robustness tests on every headline claim.** Grid sensitivity across three
  anchors, and a null control against shuffled and white-noise arms. They catch different
  things: `r = −0.47` died to the first and the phase finding died to the second, and
  neither test alone was sufficient.
- **Suspicious stability calls for a control, not a larger sweep.** A coefficient of
  variation near 1% across independently trained models on financial data is evidence the
  measurement is about the machine.
- **Bit-identical reproducibility**, scoped to the same hardware with cross-architecture
  named untested.

---

## 5. The sentiment ruling, in full

This is recorded at length because it is the most consequential exclusion in the project
and because the reasoning is the deliverable, not the exclusion.

**An LLM's training data includes the period being backtested.** Asking a model about
AAPL in March 2023 returns an answer informed by what happened afterwards. That is
look-ahead in its purest form.

**And no test in this project can catch it.** The causality harness perturbs future bars
and asserts past feature values are unchanged. An LLM does not read our bars — it reads
its own memory. The harness would pass while the leak flowed freely, which makes this
worse than an ordinary leak: the mechanism designed to catch leakage would actively
certify it.

**It also breaks reproducibility.** API responses vary between calls, so `results.csv`
would stop reproducing bit-for-bit — the claim GB-59 established on 23 August.

**And it is thematically backwards.** A project whose thesis is exact, algebraic
attribution does not add a component from which nothing can be attributed.

**What goes in the report instead:** a section explaining why sentiment was excluded, with
this reasoning. Most projects in this space add an LLM sentiment feature, obtain an
excellent backtest, and do not understand why. Explaining the trap is worth more than
falling into it, and it is a contribution rather than an omission.

---

## 6. Considered and rejected

| Idea | Why not |
|---|---|
| GB-62, the C3 feature arm | Deferred, not rejected. GB-66 answers a sharper question with the same time. If GB-66 lands before 8 September, C3 returns as a spoke. |
| Sentiment or news as a model feature | §5. Structural look-ahead the harness cannot catch. |
| Social-media hype signals | Same look-ahead problem, plus no historical archive that is causally clean. |
| Letting a user change the deployed universe | The deployed configuration must be one thing. A labelled exploration mode is acceptable; changing `model.active` or the universe from the UI is not. |
| Intraday bars | Data cost, microstructure effects, and it invalidates every Phase 1 measurement. |
| Regime Guard | Still parked. It returns only if everything above lands before 15 September, which is unlikely and is fine. |

---

## 7. The standing goal

**A final project that would survive being read by someone who does this professionally.**

Not the most features. The system already does more than the brief asked. What makes it
defensible is the part most projects skip: a null control that says the models are
noise-equivalent, a measured architectural critique of a published paper, bit-identical
reproducibility, an audit that found four wrong claims in our own specification, and a
deployed system that declines to trade when validation finds nothing.

**Phase 2 must strengthen that, not dilute it.** Every item above closes something Phase 1
measured. The moment an addition stops being traceable to a measurement, it belongs in
`IDEAS_PARKED.md`.
