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
