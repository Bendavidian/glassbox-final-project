# GB-57 — Results, Discussion and Future Work

This is the submitted chapter. `docs/GB57_OUTLINE.md` is the plan it is written against.

Every result is a delta against its own reference — MAE against persistence, direction
against always-long, Sharpe and total return against buy-and-hold — and there is no
composite score anywhere in this chapter.

Every figure traces to `results.csv`, `report/report.md`, `report/gross_exposure.csv`,
`report/null_control.csv` or `report/momentum_probe.csv`, and this chapter carries no
[UNSOURCED] markers.

---

## 57.1 — What was measured, and against what

This section exists so a reader knows what a row in the results table is before seeing one.
The methodology is GB-56's; this is the minimum needed to read this chapter.

### The design

The study evaluates six arms — persistence, buy-and-hold, DLinear, FITS, WITS, and DLinear on
a second feature configuration — over **sixteen walk-forward folds**, each training on a
24-month window and testing on the window that follows, with an H-bar embargo between them so
no test bar can share a feature window with a training bar.

Conditions form a **star, not a cross**: one reference condition at the centre, with spokes
each moving a single variable. Two fold-grid anchors at 21 and 42 trading days, two learning
rates, four cutoff periods, one execution-model spoke, and two null controls. Fourteen
conditions, 54 live cells, 864 arm-folds. A full cross would be 108 conditions and 3,888
arm-folds; it was never run.

That the design is a star rather than a cross has a consequence that §57.9 records: the
control spokes sit at the centre only, so the null controls exist at anchor 0 and cannot be
crossed with the grid-anchor axis.

### The three references

**No result in this chapter is reported as an absolute number.** Every one is a delta against
a stated reference, and the three are not interchangeable:

| metric | reference | why |
|---|---|---|
| MAE, RMSE | persistence | the trivial forecast of zero change |
| direction | always-long | the trivial rule that is right whenever the market rises |
| Sharpe, total return | buy-and-hold | the trivial portfolio |

There is no composite score anywhere in this chapter. A single number combining six axes
would hide exactly the disagreements between them that §57.4 depends on.

### The universe changed mid-study

Phase 1 measured five symbols. Everything reported here is twenty, from a grid re-run in full
on 2 September at a measured 100 minutes 56 seconds. §57.8 treats what moved and why, and the
five-symbol figures are shown beside the twenty-symbol ones wherever the comparison is the
finding rather than a footnote.

### Where every figure comes from

`results.csv` (1,110 rows), `report/report.md` generated from it,
`report/daily_equity.csv`, `report/gross_exposure.csv` (963 sizing calls),
`report/null_control.csv` and `report/momentum_probe.csv`. Each is committed, and each figure
in this chapter is re-derivable from one of them.

### One filter that matters

`buy_and_hold` carries no `channels` value — a reference arm has no feature configuration to
name — so a bare `channels == C0_base` filter cannot match any of its 224 rows. The correct
filter is `channels == C0_base or model == buy_and_hold`. This is stated here because the
arm it drops is the one §57.3's entire argument rests on.

## 57.2 — The null result

The study's headline is negative, and it is the finding rather than the absence of one.

> **These models extract nothing from this market that they do not also extract from white
> noise.**

That is a statement about the models, not about the market. It is stated first because a
reader who hears "found nothing" will otherwise supply "the market is efficient", which this
study does not claim and cannot support.

### Directional accuracy, five symbols beside twenty

Anchor 0, `target_in_loop=true`, `channels=C0_base` or `model=buy_and_hold`, averaged over
sixteen folds and every condition sharing that anchor (`results.csv`):

| arm | real 5 | real 20 | noise 5 | noise 20 | shuffled 5 | shuffled 20 | noise − real, 20 |
|---|---|---|---|---|---|---|---|
| buy_and_hold | 0.5560 | 0.5528 | 0.5064 | 0.5011 | 0.5418 | 0.5365 | **−0.0517** |
| dlinear | 0.5006 | 0.4981 | 0.5006 | 0.5156 | 0.5021 | 0.5053 | +0.0175 |
| fits | 0.4958 | 0.4948 | 0.5041 | 0.5015 | 0.5039 | 0.5026 | +0.0067 |
| wits | 0.5031 | 0.4976 | 0.5050 | 0.5001 | 0.5074 | 0.5025 | +0.0025 |

**A note on the filter, because it is load-bearing.** `buy_and_hold` carries no channels
value: all 224 of its rows in `results.csv` have the field empty, because a reference arm has
no feature configuration to name. A bare `channels == C0_base` filter therefore keeps 608 rows
and **none** of the 224 — it cannot match them rather than merely failing to. The one arm
whose behaviour carries the entire argument of §57.3 is the one a careless filter silently
removes.

Read the model rows first. Every model scores at least as well on destroyed signal as on real
data. Replacing four years of equity returns with Gaussian noise of matched variance costs
these models nothing. That is not a small effect that failed to reach significance; it is the
absence of the effect at the resolution the experiment can see.

### The +0.0175 that is not a finding

DLinear scores better on white noise than on real data at twenty symbols. The tempting
stronger claim — not *"identical on destroyed signal"* but *"better on it"* — **does not
survive correction and is not made.**

Paired by fold within the reference condition, across the control axis:

| arm | mean (noise − real) | sd | p | p Holm | positive folds |
|---|---|---|---|---|---|
| dlinear | +0.0184 | 0.0301 | 0.0174 | **0.0521** | 13 of 16 |
| fits | +0.0087 | 0.0346 | 0.3484 | 0.6968 | 10 of 16 |
| wits | +0.0035 | 0.0386 | 0.8999 | 0.8999 | 7 of 16 |
| buy_and_hold | −0.0517 | 0.0593 | 0.0034 | **0.0134** | — |

DLinear clears the uncorrected threshold and fails Holm over the four arms at 0.0521, just
above 0.05. It is different in *degree*, not in kind: one of three same-signed gaps being the
largest is what "largest of three" means, and WITS's 7 of 16 is a coin flip. **The claim
therefore remains that destroying the signal costs these models nothing, and the +0.0175 is
reported as an unresolved gap.**

One provenance caveat: this test crosses the `control` column, whereas the study's family of
117 pairs by fold *within* a condition. The Holm correction here is over these four arms only
and these are not four of the 117.

### The result does not depend on where the folds are cut

On real data, across the three fold-grid anchors:

| arm | anchor 0 | anchor 21 | anchor 42 |
|---|---|---|---|
| dlinear | 0.4981 | 0.5021 | 0.5044 |
| fits | 0.4948 | 0.4909 | 0.4934 |
| wits | 0.4976 | 0.4958 | 0.4976 |
| **always-long bar** | **0.5530** | **0.5479** | **0.5565** |

Every cell sits within one percentage point of a coin flip, the largest deviation being FITS
at anchor 21, and every one is below the always-long rate for its anchor. Shifting the fold
boundaries by 21 and 42 trading days
changes which windows train and which test, and it is the check that killed an earlier
`r = −0.47` correlation finding. It moves nothing here.

§57.9 records what this claim does *not* cover: the null controls exist at anchor 0 only, and
that is structural rather than unmeasured.

### The statistical picture, and its ceiling

The study runs a family of **117 paired Wilcoxon signed-rank tests** across every arm, anchor,
metric and condition on real data, Holm corrected over the whole family in one pass. **29
survive at α = 0.05, and every one has the model performing worse than its reference.**
§57.8 explains why that count moved from 17, and why the movement is precision rather than
effect.

The ceiling must be stated beside those numbers rather than left for a reader to find. The
exact two-sided Wilcoxon test on sixteen paired folds cannot return a p-value below
**3.05 × 10⁻⁵** however large the effect, and with 117 tests in the Holm family the smallest
attainable corrected value is bounded well away from zero. This design can fail to reject, and
can detect only fairly large effects. **It is not capable of establishing a small edge, and no
claim in this chapter should be read as if it were.**

### The structure exists and is reachable

A constant always-long rule beats chance by 5.3 points over the same period, and buy-and-hold
compounds to +125.19% over the 938 trading days in `report/daily_equity.csv` against DLinear's
+45.51% on the same file over the same dates.

So the structure is there and it is reachable. **It is not reachable by explicit or implicit
frequency decomposition at this horizon**, which is a narrower claim than market efficiency
and is the one this study supports.

## 57.3 — The instrument works

This is the shortest section in the chapter and the one the rest depends on.

A null result and a disconnected pipeline produce the same table. Nothing in §57.2
distinguishes them, and no amount of additional model arms would: a broken feed would leave
every arm flat, which is exactly what §57.2 reports. The distinguishing evidence has to come
from an arm whose mechanism predicts that it *must* move when the signal is destroyed.

`buy_and_hold` is that arm. It makes no forecast; its entire content is the upward drift of
the underlying series. The white-noise control is constructed precisely to remove that drift,
drawing N(0, σ) per symbol with matched variance, so every symbol has zero expected drift.
When the drift goes, buy-and-hold's edge must go with it.

It does.

| arm | mean (noise − real) | p | p Holm | 
|---|---|---|---|
| **buy_and_hold** | **−0.0517** | **0.0034** | **0.0134** |
| dlinear | +0.0184 | 0.0174 | 0.0521 |
| fits | +0.0087 | 0.3484 | 0.6968 |
| wits | +0.0035 | 0.8999 | 0.8999 |

Buy-and-hold degrades by 5.17 points under the noise control, and it is the only arm that
degrades at all. **It is significant after correction**, which the five-symbol study could
state only as a point estimate.

The asymmetry is the argument, and it is worth stating in the form that most constrains it.
The control moves for exactly one arm; that arm is the one whose mechanism says it must; and
it does not move for the three arms that never had a forecast to lose. Without this row,
§57.2's table would be equally consistent with a pipeline that had silently disconnected the
data. With it, the pipeline is demonstrably connected and the models are demonstrably not
using what flows through it.

A detector that stays silent is worth something only once it has been shown to fire.

### What this section does not establish

It establishes that the instrument detects a control effect where one is present at this
magnitude. It does not establish sensitivity at smaller magnitudes: 5.17 points is a large
effect, and §57.2's ceiling of 3.05 × 10⁻⁵ bounds what this design can resolve. The honest
statement is that the instrument is demonstrably not dead, not that it is demonstrably
sensitive to whatever a model might plausibly have found.

§57.9 records the narrowing that matters: the controls run at anchor 0 only, and that is a
property of the study's star design rather than a measurement left undone. It also records a
class of claim the shuffled control cannot falsify at all, which does not touch the arms here
because they forecast.

## 57.4 — MAE measures flatness

Of the 117 Holm-corrected tests, 29 survive, and **22 of the 29 are on MAE**. Every one says
a model has higher forecast error than persistence. The metric carrying most of the
statistically robust results in this study deserves examining before those results are read.

### The correlation, and the level it has to be computed at

Across the six arms, ranked by mean error and by mean flatness:

**Spearman(MAE, flatness) = +1.000.** The six arms rank identically on the two.

| arm | mean MAE | mean flatness |
|---|---|---|
| persistence C0 | 0.0128 | 0.000 |
| persistence C2 | 0.0128 | 0.000 |
| WITS | 0.0130 | 0.157 |
| FITS | 0.0131 | 0.188 |
| DLinear C0 | 0.0134 | 0.249 |
| DLinear C2 | 0.0136 | 0.295 |

Persistence forecasts exactly zero. It has both the lowest error and no flatness to measure.
The models forecast slightly more than zero and are penalised in proportion. That is why every
surviving MAE result says a model is worse than persistence: **the metric with all the
statistical power in this study is a metric that measures how close to zero a forecast is.**

The same relationship computed over the 624 individual arm-fold observations is **+0.072**,
which supports nothing. The claim is about how arms rank, so the statistic has to be computed
over arms. Pooling the folds inside each arm dilutes exactly the quantity being claimed.

### The report asserted the conclusion regardless of the number

This is recorded here because the defect sat inside the report the chapter cites.

`report.py` computed the correlation over arm-folds and printed a **hardcoded sentence**
describing the arm-level relationship. The sentence was a fixed string: it would have printed
whatever the number turned out to be, and at +0.072 the number did not support it. The
relationship it asserted was not merely present, it was perfect — and the report's own
instrument was pointed one level too low to see it while prose asserted the conclusion beside
it.

Both halves are fixed. `report.py:290-328` computes and returns both levels, and
`describe_correlation` at `:269-287` derives the wording from the value across four bands.
The fix is verified on the emission path rather than on the function in isolation:
`test_report.py:177-179` computes the real coefficient and asserts the derived wording appears
in the generated text. §57.10 files it as an instance.

### What follows for the rest of the chapter

MAE is banned as a headline in this project and appears only with flatness in the adjacent
column, structurally rather than by convention. The 22 surviving MAE results are reported as
what they are: evidence that the models produce larger forecasts than persistence does, which
is a fact about magnitude and not about accuracy.

Directional accuracy, which is the metric the headline rests on, carries three of the 29
survivors. §57.8 explains why that number is three rather than zero.

## 57.5 — The geometry critique of FITS

FITS is a frequency-domain forecaster. It takes the real FFT of the input window, discards
bins above a cutoff, applies a learned complex linear map to the survivors, zero-pads and
inverts to a longer window. The learned map is the model. If the model has learned something
about equity markets, that map should differ when the market is replaced by noise.

### The geometry that has to be learned whatever the data is

Extending a window of length `L` to length `L + H` moves every frequency's position in the
output grid by a factor `η = (L + H) / L`. A frequency that lands exactly on an output bin is
passed through; one that does not must be interpolated across the gap, on every window, by
the learned layer. The interpolation is a function of `frac(η·k)` — of the grid geometry —
and not of the data. `frac(η·k) = frac(k·H/L)`: one curve in two parameterisations.

A share of the layer's capacity is therefore spent on an operation that would be identical if
the input were noise. The question is how large that share is, and it is measurable: train
the same architecture on white noise and correlate the two learned responses.

### The gain-magnitude response: geometry, and more of it than first reported

| figure | source | r² |
|---|---|---|
| +0.9485 | GB-46's published figure, five symbols | 0.900 |
| +0.8967 | `operator_null_control.py`, 16 folds, five symbols | 0.804 |
| **+0.9780 ± 0.0105** | same script, twenty symbols | **0.957** |

The correlation rose with the universe, and the real and noise curves moved closer together:
mean absolute difference 0.0390 → 0.0256. **The data-independence the critique rests on is
larger at twenty symbols than it was at five.**

An earlier statement of this result gave "86% of the learned gain response is geometry". That
figure is not reproducible and is withdrawn: it was derived as `1 − 0.0319/0.2335`, mean
absolute difference over curve range, and `report/null_control.csv` records the numerator and
not the denominator. **The reproducible statement is r² = 95.7% at twenty symbols against
80.4% at five**, and it is the stronger of the two.

### The signed operator: the narrowed claim weakens

The claim as previously narrowed was *"the envelope is geometry; the structure within it is
not"*, and it rested on the full signed operator being indistinguishable from zero. At twenty
symbols it is not.

| | five symbols | twenty symbols |
|---|---|---|
| FITS | −0.0127 (t = −0.44) | **+0.1946 (t = +7.24)** |
| WITS | +0.0331 (t = +0.75) | **+0.0838 (t = +2.56)** |
| paired WITS − FITS | +0.0458 | −0.1108 |
| Wilcoxon p | 0.5282 | **0.0042** |
| WITS lower in | 9 of 16 folds | 13 of 16 |
| verdict | did not hold | **held** |

Both quantities are the flattened `(H, L)` forecast matrix, n = 480, computed by the same
function over the same folds with the same seed and the same budget, differing only in what
the close channel holds. This comparison is legitimate where the response comparison below is
not.

**Restated as a matter of degree:** the envelope is overwhelmingly geometry (r² 0.957) and the
signed operator is partly geometry (r² 0.038, firmly non-zero). More of FITS is
grid-determined than the narrowed claim allowed. The two halves must still never be quoted
with the same number — one is an order of magnitude larger than the other — but neither is
zero.

**WITS, stated honestly: it does not escape the pathology; it escapes it less.** A local basis
reducing grid-determination rather than removing it is what a real mechanism looks like, and a
claim that WITS escaped entirely would be refuted by its own t-statistic.

Both five-symbol verdicts were negative and both were **power failures rather than absences**;
§57.8 treats that as its own finding, because it happened twice in independent instruments.

### GB-66 is complete, and its acceptance criterion is unmeasurable

WITS exists because the critique above is either a claim about FITS specifically or a claim
about extending any *global* basis, and one architecture cannot distinguish the two. WITS
substitutes a discrete wavelet transform — a local basis — for the rFFT in the same position,
with the same instance normalisation and the same supervision, so the comparison isolates the
transform.

Its stated acceptance criterion was the response correlation, reported against FITS's. **That
quantity cannot exist for WITS**, and the null-control script refuses it in code rather than
emitting a NaN:

- FITS retains 24 rFFT bins: `input_len` 120 gives 61 bins and the 5-day cutoff keeps 24. The
  response curve reports 23 of them, dropping the infinite-period DC entry — which is the dead
  row Case 2 found.
- WITS retains two bands. A Pearson correlation over two points is ±1 by construction and
  carries no information; the script's own `_correlate` returns NaN below three values.
- Even taking all four DWT bands gives n = 4 against the 23 points of FITS's response curve,
  and the values are different objects: a per-band aggregate over an octave against a per-bin
  gain at one frequency.

The script emits an **absent row rather than a NaN row**, deliberately, because a NaN invites a
reader to wonder whether the measurement failed.

So GB-66 is complete, and its finding is that **its own acceptance criterion is structurally
unmeasurable**. The critique is Fourier-specific by construction, and that was learned only by
building the substitute. What does transfer is the signed operator, and it says WITS is
grid-determined less than FITS is, at p = 0.0042.

Three readers — two project documents and the author — read the absent row as "never measured"
before reading the code. That is recorded in §57.10.

## 57.6 — The execution finding

This result is not about forecasting. It emerged from live rehearsals against a real broker
and it changes what every trading number in this study describes.

### The construction the literature assumes

A backtester exits a position one of two ways: a protective stop below the entry or a
take-profit limit above it, both resting at the broker, either able to fill intraday the
moment price crosses it. That is the standard construction and it is what this project's
backtester originally implemented.

**It cannot be built at this venue**, and two separate refusals establish it.

### Refusal one: no multi-leg order on a fractional quantity

Submitting a bracket order, and separately an OCO order, on a fractional position both return
`{"code":42210000,"message":"fractional orders must be simple orders"}`, measured directly
against the paper API rather than inferred from one case to the other.

Worse, a working sell order at this broker holds the **entire** position, so a standalone stop
and a standalone limit cannot coexist either: whichever is submitted second is refused for
insufficient quantity. That second constraint was measured against a 97.38-share position,
which establishes it is **not** a fractional-only rule. Whole-share sizing would not lift it.
What fractional sizing removes is the bracket order that would have been the workaround.

So a position at this venue carries exactly one broker-side protective order. The stop is the
one that must exist, because it is the one that bounds a loss. The take-profit moved into the
trading loop — which reads completed daily bars only, cannot observe an intraday crossing, and
learns the target was reached when the bar closes.

### Refusal two: no duration beyond the session

`executor.py:25` records the second refusal:
`{"code":42210000,"message":"fractional orders must be DAY orders"}`.

A fractional position cannot request GTC at all. The protective stop is therefore
`time_in_force=DAY` by API rule rather than by choice, and the broker expires it at every
close.

**Composed, the two refusals give a hard limit: fractional implies no OCO and no GTC,
therefore no protection that survives 16:00 ET.** A ruling this project had carried as
undecided — whether to move stops from DAY to GTC — is not undecided. It is unavailable, and
taking it would require whole-share sizing, which changes position sizing and the fractionable
criterion in the universe rule. That makes it a scope decision, not an order parameter.

### Measured on a live position, not inferred

Entry `9b6f0d86` filled 2026-09-03 at 19:40:40Z: 92.038821904 WMT at 108.70. Protective stop
`8820c4d7`, SELL STOP at 105.39, submitted 19:42:11Z with `time_in_force=day` **read back from
the broker rather than from the code's intent**, and `status=EXPIRED`,
`expired_at=2026-09-03T20:01:31.801Z`.

The position was then held across the close with no protection at all, deliberately, on a
paper account, to obtain evidence the re-arm path had never produced.

### The re-arm path, first execution

Every session log from 25 August to 3 September reads `overnight residual: none; the session
opened flat`. That path had never run.

On 23 September it did. The session opened with a banner naming the residual, reconciliation
reported `missing_protection` with zero live sells, and step 4 armed a new stop at the same
105.3905 from the book. Later the same session, two entries the broker had filled before the
book knew of them were quarantined as `unknown_position` and then adopted under the decisions
that had produced them, each booked with stop and target and armed in the same cycle. Three
positions were re-armed in one session.

The session closed with `open orders at exit : 3 ... status=new`, all three `tif=day`. §57.10
records what that line is: true when written and false as the session's terminal artefact,
eighty-two seconds later.

### What it does to the numbers

The study measures both systems. The `target_in_loop` axis carries `false` for the classical
broker-side target and `true` for the buildable loop-side one, with `true` as the default so
the headline numbers describe the system that exists.

Paired fold by fold at the reference condition:

| arm | trades, broker-side | trades, loop-side | change |
|---|---|---|---|
| DLinear | 590 | 554 | −6% |
| FITS | 607 | 566 | −7% |
| WITS | 507 | 480 | −5% |

The trade count falls in every arm, and the mechanism is capital rather than signal. A delayed
exit holds capital through at least one additional session, so entries the broker-side system
would have funded are never funded at all.

The return difference does not resolve. Paired per fold, the ratios of mean to standard error
are +1.63, +1.96 and −0.51 across sixteen folds. All three sit inside the noise and the sign
is not consistent: DLinear and FITS are better under the buildable model while WITS is worse,
and at five symbols the pattern ran the other way. **The buildable system is measurably
*different* and not measurably *worse*.**

An earlier description of this finding was "32.7% worse", from a single run over the whole
universe. That is a real point estimate and it stands as one, but the paired fold test says
the difference is not separable from fold-to-fold variation, and reporting the point estimate
as the result would have been a stronger claim than the data supports.

### Why this does not rescue or damage §57.2

`direction`, `mae` and `flatness` are **byte-identical across the two execution arms** for
every model at the reference condition. They must be: they are properties of the forecast, and
the axis changes only how a fill is priced. The execution model therefore could not have
altered the null result in either direction. §57.2's finding is the same finding under either
system, now established for the one that can actually be built.

### What this means beyond this project

The construction assumed by the trading literature — a resting stop and a resting target,
both live, both able to fill intraday — is not available for a fractional position at this
venue, and the constraint is discoverable only by connecting to a broker. A simulator would
never produce it. **Every backtest that assumes it describes a system that cannot be built
here**, and this study is able to say so because it submitted the orders and read the
refusals.

## 57.7 — The risk layer stops being inert

At five symbols the gross exposure cap bound in **one of sixteen folds**. An earlier draft of
this chapter had to describe a layer that was correct, enforced and doing nothing.

### At twenty symbols it binds in 13 of 16

Measured by `glassbox/experiments/exposure.py`, which records every sizing call and persists
them to `report/gross_exposure.csv`:

| case | count |
|---|---|
| REDUCED_BY_GROSS — the cap shrank the entry, notional > 0 | 387 |
| BLOCKED_BY_GROSS — the cap refused it, notional = 0 | 147 |
| BLOCKED_BY_OTHER — cash or the per-position cap | **0** |
| TIE — per-position cap and gross headroom exactly equal | **0** |
| NO_EQUITY | 0 |
| UNCONSTRAINED | 429 |
| total sizing calls | 963 |

The three folds in which the cap never bound are folds 1, 3 and 4, and they are the three
thinnest — 0, 17 and 15 sizing calls against 27 to 130 elsewhere. No anomaly needs explaining.

Cash never blocked an entry. Every zero notional in 963 calls was the gross cap.

### The measurement design decided the answer

**The reducing case is 2.6 times the blocking case.** An instrument watching for zero
notionals — the obvious design, and the one this measurement was originally specified with —
would have counted 147 of 534 cap events, 28%, and concluded the cap rarely binds. Wrong, and
wrong in the direction that flatters the risk layer.

Ties were recorded rather than resolved at record time, so the tie policy could be stated
afterwards and its weight measured rather than assumed. The count is 0, so none of the
headline rests on it — but that is a measured answer, and collapsing the ambiguity inside the
instrument would have baked a choice into the measurement and hidden it.

### The layer binds without leaving evidence, in both of its modes

`room_for` returns `max(0, min(A, B, C))` over the per-position cap, the gross headroom and
available cash, and discards which term was the minimum.

When gross headroom merely **reduces** an entry, the notional comes back positive and the
trade looks ordinary. That is the invisible case, and it is the one where the cap is doing its
work. When it **blocks**, the entry exits through `notional <= 0.0 → continue`, which is
ambiguous between the cap, exhausted cash and a non-positive equity early return.

This is why the question was unanswerable for a month, and it is an instance of both
principles in §57.10 at once. The instrumentation added for this measurement observes beside
the sizing path rather than fixing it: it was proved not to alter a single metric by running
the reference condition with instrumentation on and off and comparing all sixteen fold rows
across sixteen columns, plus 800 property-based examples asserting `room_for` is bit-identical
to its pre-change body.

Peak gross reached 0.5212 against a 0.50 cap. **Correct, not a breach:** the cap bounds what
may be *added* at entry, while gross exposure is the marked value of what is already open, and
nothing sells to get back under.

### The mechanism, and a tension neither argument for the expansion predicted

With `top_k = 2` and `max_position_pct = 0.10`, new entries cap at 0.20 per bar, so the gross
cap can bind only through accumulation across overlapping holds. What universe size changes is
the ceiling on concurrent holds: five symbols permit at most 5 × 0.10 = 0.50, **exactly** the
cap, so it can be touched and never exceeded — which is why it was very nearly decorative.
Twenty permit up to 2.00, four times the cap.

**534 of 963 sizing calls — 55% — were cap events.** The cap is now the binding constraint on
more than half of all sizing decisions.

That sits in tension with the expansion's other stated purpose, that cross-sectional ranking
becomes a real selection at `top_k = 2` out of twenty rather than out of five. If the cap
dictates size in more than half of calls, **it may be selecting more than the ranking does.**
Both goals were met and they now pull against each other. This is a measured consequence of
universe size, not a defect, and it is not resolved here.

`report/gross_exposure.csv` holds all 963 sizing calls, so every count above is recomputable.

### What the live sessions added

The backtest measures the cap; the live path exercised the machinery around it. On
23 September a single session ran 322 cycles with no failed step, held three positions
simultaneously for the first time, and re-armed protective stops three times. Two of those
were positions the broker had filled before the book knew of them: reconciliation quarantined
them as `unknown_position`, and the adoption path then recognised the loop's own decisions
behind them and booked them with stop and target. Neither the quarantine nor the adoption path
had ever executed before.

Three positions at roughly 10% each is 30% gross against the 50% cap. The cap did not bind
live, which is consistent with the backtest: it binds through accumulation, and one session
does not accumulate.

## 57.8 — What universe size changed

The universe moved from five symbols to twenty part-way through the study, and the grid was
re-run in full. The Phase 1 conclusion survived. The Phase 1 statistics did not.

### Holm survivors: 17 → 29

The family widens from `{mae}` alone to `{direction, mae, total_return}`. The smallest
corrected p-value on direction goes **0.2605 → 0.0036**. **All 29 survivors still have the
model losing.**

An earlier statement in this project — *"not one result on direction, sharpe or total_return
survives"* — is now false and is withdrawn.

Where Phase 1 could only fail to reject, this study rejects: against the models.

### It is power, not effect, and that was measured rather than argued

Effect sizes are the same or **smaller** in 23 of the 29 cells. One further cell rose by
2.6e-5, which is a tie at any reportable precision and is counted as grown here.

| family | mean \|Δ\| at five | at twenty | ratio |
|---|---|---|---|
| direction (3 survivors) | 0.0599 | 0.0580 | 0.97 |
| total_return (4) | 0.0827 | 0.0473 | **0.57** |
| MAE (22) | 0.000620 | 0.000475 | 0.77 |

`n` is 16 at both universe sizes. The extra power did not come from more folds. It came from
the collapse of the **paired-difference standard deviation**, which is the quantity the
signed-rank test consumes, because each fold's metric moved from an average over five symbols
to an average over twenty:

| arm, metric | sd of paired difference at five | at twenty | ratio |
|---|---|---|---|
| persistence, total_return | 0.11663 | 0.04633 | 0.397 |
| fits, total_return | 0.09623 | 0.03068 | 0.319 |
| wits, total_return | 0.10144 | 0.03246 | 0.320 |
| dlinear, direction | 0.06571 | 0.02576 | 0.392 |
| fits, direction | 0.06514 | 0.04209 | 0.646 |
| wits, direction | 0.05708 | 0.04217 | 0.739 |

**The strongest form of the finding:** on total_return the models are *less bad* at twenty —
the shortfall against buy-and-hold shrank to 57% of its five-symbol size — **and it became
significant anyway**, because the estimate is roughly three times more precise.

The ceiling is unmoved. The exact two-sided Wilcoxon test on sixteen paired folds cannot
return a p-value below 3.05 × 10⁻⁵ however large the effect, and with a Holm family of this
size the smallest attainable corrected value is bounded well away from zero. Only the noise
floor dropped.

**A caution about the reported `sd` column, because it points the other way.** The `sd`
printed beside each arm is that arm's own per-fold spread, not the paired difference, and only
the paired difference feeds the test. Over the 39 total_return cells present at both sizes,
the reported spread **rose** ×1.42 and grew in 25 of 39 cells, while the paired-difference sd
**fell** ×0.38 and fell in 39 of 39. The two quantities move in opposite directions and only
the lower one is unanimous. A reader auditing this section against the printed column would
read the mechanism backwards.

### The same pattern, in an independent instrument

§57.5's operator comparison shows it again. At five symbols neither FITS's nor WITS's signed
operator correlation was distinguishable from zero — −0.0127 at t = −0.44, +0.0331 at
t = +0.75 — and the WITS-versus-FITS verdict did not hold, at Wilcoxon p = 0.5282. At twenty
the same comparison gives p = 0.0042 and holds, with WITS lower in 13 of 16 folds against 9.

Two independent negative verdicts at five symbols, in different instruments measuring
different quantities, were **both power failures rather than absences of effect**. Two
instances is a claim about Phase 1's design, not a coincidence.

**Phase 1's null on direction and total_return was partly a small-sample artefact.** The
deficit was always there; a five-symbol design could not resolve it. This weakens only the
negatives about the negatives. The headline is untouched and better supported: every one of
the 29 surviving results says a model is worse than its reference.

### What else moved

The risk layer's cap went from binding in one fold to binding in thirteen; §57.7 treats it.
The calibrated entry threshold moved from 0.026076 to 0.006715 on the same fold and window —
roughly four times more permissive, with validation trades rising from 8 to 53 and validation
Sharpe from 0.483 to 0.785. Only the universe changed. A threshold calibrated on validation
lands in a different place when the cross-section it is calibrated over is four times wider,
and the deployed system went from standing aside on almost every bar to entering on several.

Determinacy rose from 2.09× to 8.35×.

### What it cost to know this

One full grid re-run, measured at 100 minutes 56 seconds, single-threaded by design for
bit-identical reproducibility. Every corrected p-value in the study was recomputed, not
extended: the family is rebuilt in one pass, so statements of the form "of 117 tests, 17
survive" become "of 117, 29" and cannot be carried across from the earlier draft.

## 57.9 — Limitations

These are stated rather than defended. Each is a reason to read a specific claim in this
chapter more weakly than it might otherwise read, and each is named here rather than left
for a reader to find.

### The universe is selected for survival, and the selection rule says so

Spec §2.4 admits a candidate only if all five criteria hold, and criterion 3 requires
**continuous tradability through to the cache's last bar** — no suspension, no delisting.
Criterion 2 requires complete history from 2016. Both are knowable only in 2026.

So survivorship bias is not an oversight in this study; it is written into the admissibility
rule. The consequence has a direction and it belongs beside every number here: the
`buy_and_hold` arm is biased upward, because a basket of twenty companies selected for
having survived and stayed large cannot contain the ones that did not, and the always-long
direction bar is inflated the same way.

Model arms are reported as deltas against those references, so the bias sits **inside** the
reference each arm is measured against rather than being quietly removed. An arm beating a
survivor-inflated buy-and-hold is a stronger claim than the same arm beating an unbiased
one. Nothing in this repository can bound the bias: bounding it needs delisted securities or
point-in-time index membership, and the cache holds twenty survivors and no membership
history.

### The universe is not derived from its own rule

The five criteria are an **admissibility filter, not a selection procedure**, and the
difference is where the hindsight sits. Every one of the twenty passes all five; the twenty
do not follow from them. The criteria admit several hundred US large caps with complete
history from 2016, and returning exactly twenty needs a step the rule does not contain.

Criterion 4 is where that shows. *"Large capitalisation at the selection date"* states no
threshold, no ranking, no source and no date, so it cannot be executed against a candidate
list and cannot produce a set. It can only justify one already drawn. AVGO, ORCL, KO and PEP
satisfy all five criteria and appear nowhere in §2.4, neither admitted nor excluded, and they
are four of many. The step from admissible to these twenty was judgment, and it is a
limitation of this universe rather than a property of the rule.

No threshold is invented here to close the gap, deliberately: a rule written to fit names
already chosen is hindsight with better grammar, and it would read as a derivation while
being a rationalisation.

### The universe is concentrated on one factor

Seven of the twenty are the US mega-cap cluster and move together. Four of the eleven GICS
sectors — Industrials, Materials, Real Estate and Utilities — have no representation at all.
On the pre-2023 classification, V and MA are Information Technology rather than Financials,
which concentrates the book further. Any result that depends on cross-sectional dispersion
is measured on a universe with less of it than twenty names implies.

### One of the two robustness tests cannot fire against a whole class of claim

This is a limitation of the *method*, discovered in September and not present in the study
design.

The project's rule is that every headline claim gets both robustness tests: three fold-grid
anchors, and null controls on shuffled and white-noise arms. The two catch different things,
which is established by each having caught something the other missed.

But a permutation preserves a sum. `study.null_bars` permutes a symbol's log-return series
and rebuilds prices by cumulative sum, so each symbol's whole-sample return survives exactly:
across the twenty symbols the shuffled final close differs from the real one by at most
1.43e-14 relative, and **Spearman(real full-sample drift, shuffled full-sample drift) across
the universe is +1.0000**, against +0.0812 for white noise.

Under that permutation, any trailing window is a sample without replacement from the whole
series, so its expectation is the window length times the symbol's full-sample mean — a
partial readout of the sample's outcome, **including the bars after t**. Any signal that is
a monotone function of accumulated per-symbol return is therefore immune to the shuffled
control by construction.

Demonstrated rather than argued, on a cross-sectional momentum rule scoped and declined the
same day: top three by trailing 252 bars, rebalanced monthly, earns +0.00596 per rebalance
over equal weight on real data (paired p 0.2491) and **+0.00769 on shuffled (p 0.0856) —
more than on the market** — while the rank persistence it is supposed to trade collapses
from +0.0536 to −0.0132. The edge survives the shuffle because it never came from
persistence.

Nothing would have said so. A shuffled row would have come back, survived, and been read as
evidence. The pre-registerable discriminator is one line: Spearman of full-sample log drift
between real and the control, across the universe. Above 0.9, that control cannot falsify
the arm and its shuffled row means nothing. `report/momentum_probe.csv` carries the figures
and `scripts/momentum_probe.py` reproduces them deterministically.

This does not weaken any claim in §57.2 to §57.8: the arms reported there forecast, and a
forecast is not a monotone function of accumulated return. It narrows the rule.

### The null controls exist at one anchor only, and this is structural

Row counts across the control axis in `results.csv`:

| control | anchor 0 | anchor 21 | anchor 42 |
|---|---|---|---|
| noise | 208 | 0 | 0 |
| real | 544 | 112 | 112 |
| shuffled | 112 | 0 | 0 |

`study.conditions` builds a **star, not a cross**, and puts the control spokes at the centre
only. So *"flat at all three anchors under both null controls"* is not merely unmeasured in
this snapshot — it is unavailable in this design. The project's two independent robustness
checks, the ones that killed an `r = −0.47` correlation and a phase advance respectively,
**cannot currently be crossed**. Closing it means a cross rather than a star: 108 conditions
against 14, which was never run and is recorded as unmeasured rather than estimated.

What is supported is narrower and is what this chapter claims: the null result is flat
across three anchors on real data, and it holds against both controls at anchor 0.

### The evaluation window is recent and short

Sixteen walk-forward folds over a snapshot ending 2026-08-13. The window contains neither
the 2018 volatility episode nor the March 2020 crash, so every drawdown, Sharpe and
maximum-drawdown figure describes a period without a severe dislocation, and the risk
machinery is correspondingly untested against one. The exact calendar span of each fold is
not recoverable from `results.csv`, which records fold indices rather than dates; the absence
of 2018 and 2020 is an inference from the snapshot date rather than a fact read off the file.

### The validation split is used twice

For early stopping and for threshold calibration. This does not leak into the test period,
but it means the validation Sharpe is a selected maximum and is not reportable.

The direction of the resulting bias is worth stating, because it makes this a caveat rather
than a reason to discount the numbers: thresholds are calibrated on forecasts already
slightly overfit to validation, so on test they are **miscalibrated rather than inflated**.
The double use degrades test performance; it does not flatter it. Every test figure here is
a lower bound with respect to this particular flaw. The magnitude is unmeasured.

### Sharpe rests on a selected subset of folds

Where an arm stands aside for a whole fold it produces no Sharpe, so Sharpe comparisons rest
on between 13 and 16 folds with a mean of 14.63, against 16 for direction, MAE and total
return. The folds that survive are those in which the model chose to trade, which is not a
random subset. Sharpe figures in this chapter are descriptive, not inferential.

### Reproducibility is same-hardware only

Determinism was verified by re-running the grid on one machine, bit-identically.
Cross-architecture reproducibility has not been tested, and floating-point reduction order in
the training path makes bit-exact equality unlikely to hold across architectures. The claim
made is the one that was measured.

### A third feature arm was available and was declined

`C3_extended` — a third feature configuration adding ATR, Bollinger position, a volume-flow
measure and longer-horizon momentum — was specified and is not in this study. The clause
permitting it was live and its condition was met.

The incremental compute is small: 128 rows at roughly 7.0 s per row, derived from
`results.csv`'s own `seconds` column. The costs that decided it are elsewhere. Every new
indicator must be pure, trailing only, pass the causality harness in both perturbation modes
at three splits, and carry a hand-computed fixture test with literal expected values. OBV has
no causal trailing-window definition, being a cumulative sum from the series start, so it is
a design question before it is an indicator. And `min_history_bars` is a **max** over
per-channel warm-ups: ATR at period 14 shares `rsi14`'s Wilder recursion and lands on the same
325-bar warm-up, but ATR at period 20 needs a 469-bar warm-up — 449 decay steps plus the
20-bar seed — which would raise the parity floor from
445 to 589, change the window geometry, change the config hash, and cause every existing
checkpoint to be refused — re-running the entire grid rather than the C3 rows.

The stronger reason is that GB-66 already answered the question C3 was a proxy for. C3 asks
whether the null result is an artefact of feature choice, a question about our choices
answerable only by making a third set of them. GB-66 replaced the model's transform rather
than its inputs, and its result is reported in §57.5.

**What a reader is owed:** this study tested two feature configurations, both chosen by its
author, and cannot exclude that a third would have behaved differently. The claim is bounded
accordingly.

## 57.10 — The methodological contribution
This project produced a working system and a null result. It also produced something the
plan did not anticipate: a catalogue of ways a test suite can be green and wrong. Each
instance below carries a commit, a file and line, or a test node ID; instances that could
not be evidenced were cut rather than softened.

The three principles were not adopted from the literature. Each was written down after the
second or third time the same shape of failure appeared, and each has cost measurable work.

**P1. Only something that runs is a mechanism.** A fact stored in two places needs
something that makes the copies equal. A test that a flag, a cache or a marker can skip is
not a mechanism. A description of what the code should do, whether in a comment, a
docstring, a plan or a default value, is not a mechanism.

**P2. A green result answering a different question is the most comfortable kind of
wrong.** Nothing about it looks like a failure.

**P3. A mechanism installed for one reason covers a failure nobody connected to it.**

### Case 1: the feed downgrade that every schema test would have passed

The live data layer requests Alpaca's SIP feed. If that request is silently downgraded to
IEX, the response has the same columns, the same dtypes, the same index and the same
provenance string. It is a valid bar series from a legitimate exchange. Every schema test
in the suite passes.

The prices are different. Measured against SIP over the same session, IEX quotes diverged
by up to 193 basis points, against a tolerance of one basis point set for reconciling the
two sources — a factor of roughly 190. A backtest built on such a series would be
internally consistent, reproducible, and describing a market that did not exist.

Until 18 August there was no test that could see it. The suite verified that the data had
the right shape, which it did. Nothing verified that it had the right values, and nothing
needed to fail for the error to enter every downstream number.

The fix is not a better schema test. It is a different question, asked separately:
`tests/data/test_live_schema.py:238-244` compares the two feeds directly and asserts the
divergence stays inside the tolerance the reconciliation assumes. The original tests were
correct and comprehensive. They were answering a question about shape while the failure
lived in value.

### Case 2: the dead DC row, found by a test written to prove something else

FITS shares one linear map across the whole universe, so a test was written to demonstrate
that sharing: perturb one row of the learned matrix and every symbol's forecast should
move. The test perturbed row 0 and **nothing moved for anybody**.

That is not what a sharing test looks for. A sharing test expects the perturbation to
propagate; it has no opinion about a row that propagates to nothing. Following the anomaly
found that **50 of FITS's 1,200 parameters were allocated and unable to learn** — the
zero-frequency row of the complex map, which the architecture reserves and the gradient
never reaches.

`DECISIONS.md:1656-1658`, `:1660ff` ·
`tests/model/test_fits.py::test_the_dc_row_is_allocated_and_cannot_learn`

Four per cent of the model's parameter budget, inert, discovered by accident and now pinned
by a test that asserts exactly the property that was found rather than the one that was
sought. The instance is P3 in its cleanest form: a mechanism installed to demonstrate
weight sharing caught a capacity defect nobody had connected to it, and it lands on the
architecture this chapter's geometry critique is about.

### Case 3: one guard, both sides of the ledger

`test_tree_has_no_extra_modules` is a **P1** mechanism. It pins the module list in spec
§3.4 against the package tree, so a module cannot exist in the code without an entry in the
specification. Two copies of one fact, with something that makes them equal.

It is also what went red when `dashboard/tokens.py` was added without a spec entry, and it
**stayed red for four consecutive commits**, each of which reported a passing subset of the
dashboard tests rather than the suite. The commit that eventually recorded this states it
directly: *"THE SUITE HAS BEEN RED SINCE REGION 1"* (`3e00950`; red from `1c9cf49`).

So a guard built for specification consistency caught a reporting failure nobody had
connected to it, and the two sides arrive in one anchor. The P1 mechanism did its job; the
P3 benefit was free.

**Note for accuracy, because two episodes are easily merged:** this is not the eight-region
incident recorded in `CLAUDE.md:74`, where the suite was green throughout. They are
separate, and conflating them would overstate both.

### The full instance table

Appendix material. Every row anchored; the principle is in the first column.

| P | instance | anchor |
|---|---|---|
| 1 | `SPEC_MODULES` and the package tree | `test_scaffold.py:14`; `::test_spec_module_exists` |
| 1 | `VALID_MODELS`, `ALL_FORECASTERS` and `smoke_offline`'s literal — **and a fourth copy**: `run()` defaulted to the literal `"dlinear"` rather than `model.active`, so `model.active: fits` changed the live checkpoint and nothing about the study | `loader.py:42`, `model/__init__.py:66`, `smoke_offline.py:920`; `DECISIONS.md:1628-1634` |
| 1 | Exit reasons in two modules | `engine.py:84-97` vs `records.py:72-82`; `test_records.py:453-457` |
| 1 | pyproject and the lockfile | `test_scaffold.py:155-191` |
| 1 | The `--sessions` default: a multi-day live run launched without the flag that made it multi-day completed one session, stopped, and looked exactly like a run that had finished | `live_loop.py:2641-2644` |
| 1 | The pill guard counting its own comment — the third instance arrived **one turn after** the line was written, inside a comment explaining the thing the guard guards | `7c98dbb` |
| 1 | The fixed-width stylesheet slice | `test_app.py:1189-1194` |
| 1 | A guard against unsourced claims in this chapter, proposed and refused: it would have gone red on the header sentence promising no such markers, because the sentence has to name the marker. Third costume, arriving inside the turn that wrote the rule | 23 Sep 2026 |
| 1 | Three written copies of the universe in two files; spec line 75 said twenty while line 599 said five. Closed by a spec-parsing test, 8 of 8 mutations caught | `d68071f`; `test_config.py::test_every_universe_written_in_the_spec_matches_the_configuration` |
| 1 | The honesty rule named an invocation that could not run: bare `pytest` exited 2 with zero collected, while every "N passed" came from `python -m pytest` | `DECISIONS.md:5258-5300`; `fbe6d5d` |
| 1 | `test_train_live_parity` derived its count; `test_sweep` pinned `== 25`. Same property, two writings, both green for months, and only the pinned one broke on the flip | `test_sweep.py:189-194` |
| 1 | `room_for` discards which term bound, so the risk layer binds without evidence in both of its modes | `risk.py:334`, stated at `:249` |
| 1 | `predict.require_current_config` has no caller. It is the only function that refuses on the wide config hash; its docstring says the loop asserts freshness each cycle, and the loop does not | `predict.py:195`; `test_predict.py:296` |
| 1 | `reconcile.py:287` requires two protective legs. The policy was retired 26 August and `live_loop.py:168` carries the one-leg truth. The retirement swept for **language** — four prose sites — and could not see a threshold | `reconcile.py:287` vs `live_loop.py:168`; `PROGRESS.md:389-398` |
| 1 | Two reported numbers whose producing code does not exist: the 16.00% gross exposure, and the 86% geometry figure whose denominator was never recorded | `DECISIONS.md:3497`, `:1159-1160` |
| 1 | A **half-true** claim, harder to catch than a false one: a handoff said WITS's null control had run. The *operator* rows had; the *response* row never existed and cannot | `PROGRESS.md:317`, `:319` |
| 1 | `test_the_same_grid_twice_gives_the_same_numbers`: the docstring said one condition and one fold; the body ran the whole grid | `CLAUDE.md §3` |
| 1 | A hand-written contrast table wrong in one of seven, replaced by a computation | `test_tokens.py:41-47` |
| 1 | `git reset` is a step of the commit-tree method, not a repair: plumbing that avoided writing the shared index left it wrong instead, showing three new files as staged deletions | `cfac51c` |
| 2 | A contract property that compared a number to itself and could not fail | `spec:1233(a)`; `DECISIONS.md:1144` |
| 2 | `pytest -q \| tail -4` reporting exit 0 on a suite with one failure, because the status was the pipe's | `70e3c2a`; `CLAUDE.md:76` |
| 2 | CI green on 1,026 tests with 26 skipped | `spec:1233` |
| 2 | `import glassbox` succeeding from the repository root without the editable install, so the installation under test was never the one anybody had | `DECISIONS.md:642-650` |
| 2 | A read-timeout dry run returning clean on twelve cycles in a healthy period, when the bound had been derived from outage latency the dry run could not reproduce | `284edcb` |
| 2 | **FakeBroker permitting what Alpaca forbids** — 1,181 passing tests over a live path that could not place a protected order. Making the double faithful turned 26 tests red across three files, which is a measurement of what had never been tested. One test was deleted rather than repaired, because its premise was impossible | `PROGRESS.md:311`; `fake_broker.py:161`; `test_executor.py:77` |
| 2 | `explain_spectral` defined, exported, unit-tested and never called; every decision record carried `per_frequency=None` and GB-53's panel had never rendered. Wired now: 67 of 67 replayed records carry the decomposition, both views closing to 2.551e-09 | `spectral.py:185`; `live_loop.py:1114-1121` |
| 2 | Four commits reporting a passing dashboard subset while the full suite was red | `3e00950` |
| 2 | The contrast test passed over the least legible text on the page, 1.27:1, because the element was classified as the rule it was named for | `test_tokens.py::test_every_text_colour_clears_aa_against_the_ground` |
| 2 | **A second, independent blind spot in the same test: AA knows nothing about size.** `--dim` cleared AA at 5.19:1 and was set at 9px — *"the palette test above passed while the thing it was protecting was unreadable in print"* | `test_tokens.py:105-115` |
| 2 | **The calendar painted a flat day green.** `GAIN if value >= 0`. Of FITS's 175 daily changes, 143 were flat, so the project's central finding — that the system declines to trade — rendered as a mostly winning year. **Recorded rather than re-derivable:** the figure is the one the test docstring carries, taken against the partial `daily_equity.csv` of the time; the committed file now holds all sixteen folds and gives a different count on a different slice | `test_app.py::test_a_flat_result_is_never_drawn_as_a_gain` |
| 2 | The session summary misreporting itself: `positions flattened : 0` on a rehearsal that cancelled a stop and sold the position, because close-out runs after the cycle list closes; and `open orders at exit : 0`, the cleanest looking line, which is exactly what a failed close-out prints | `PROGRESS.md:315`; `b421dc5` |
| 2 | The same summary line, measured live: `open orders at exit : 3 ... status=new` printed at 20:00:51Z on 23 September; all three stops were `tif=day` and expired at the close. True when written, false as the session's terminal artefact — and GATE 2's protection evidence rests on these summaries | `logs/live-2026-09-23.log` |
| 2 | `report.py` computed Spearman(MAE, flatness) over 624 arm-folds and printed a hardcoded sentence describing the arm-level relationship: **+0.072 printed, +1.000 true**. Both halves fixed — the statistic is now computed at both levels and the wording is derived from the value, verified on the emission path rather than on the function | `report.py:269-328`, `:557-572`; `test_report.py:177-179` |
| 2 | The report's `sd` column is the arm's own spread, not the paired difference, and only the latter feeds the signed-rank test. Reported sd rose 0.01605 → 0.02286 (×1.42, up in 25 of 39 cells) while the paired-difference sd fell 0.13567 → 0.05154 (×0.38, down in 39 of 39). The two move in opposite directions and only the lower one is unanimous | `stats.py:232`; `7997ed7` |
| 2 | An instrument watching for zero notionals would have counted 147 of 534 cap events — 28%, wrong in the direction that flatters the risk layer | `exposure.py:18-23`; `gross_exposure.csv` |
| 2 | A bare `channels == C0_base` filter drops every buy-and-hold row: 224 of 1,110 rows, none carrying a channels value. The filter keeps 608 where the correct one keeps 832 | `7997ed7` |
| 2 | `buy_and_hold`'s 224 rows in `results.csv` describe only 96 distinct backtests, so 128 of them duplicate another row's results while differing in `lr`, `cutoff_period_days` and `config_hash` — not byte-identical rows, which is why nothing that compares whole rows can see them. Averaging the file naively gives mean Sharpe 1.8931 against 1.9344 de-duplicated — **the duplication moves the number away from §7.3's 2.0 alarm**, toward the side nobody investigates | `study.py:302-308`; `f4d187e` |
| 2 | `Divergence.broker` holds a quantity in every divergence except `_unprotected`, where it holds an order count. `local=92.038821904 broker=1.000000000` reads as a quantity mismatch and means 92 shares held with one live sell | `reconcile.py:280` |
| 3 | Tracking `data_cache` for CI made GB-59's clean-clone audit possible: a clone with no data can run nothing | `7c98dbb`; `PROGRESS.md:301` |
| 3 | The once-per-completed-bar rule, ruled to stop 1,950 identical rows a day (five symbols; 7,800 at twenty), stopped a stray relaunched rehearsal opening a second position against a live account. **Keep its qualifier**: it refused by accident, and a new bar completes every session, so that protection expires | `DECISIONS.md:5104-5113` |
| 3 | **The same rule, the other side:** the first `close_out_on_stop` test passed with the wiring removed, because it asserted an empty broker on a session that was never going to open a position | `DECISIONS.md:227-233` |
| 3 | Per-region commits, asked for so a redesign could be bisected, paid off when a scripted edit deleted sixteen tests from three committed regions | `7c98dbb`; rationale stated in advance at `0f09587` |
| 3 | `test_every_script_guards_its_imports` caught a missing matplotlib hours before it bit — *"A test written for one reason caught a different failure."* **With its P1 sting:** the rule was added at 12:00 and the fourth instance shipped at 12:06 | `DECISIONS.md:1244-1247` |
| 3 | `predict.py:80` refused to score ABBV against a five-symbol checkpoint after the drift gate above it passed the checkpoint as valid — true and irrelevant. Its message states why: *"the result would look like a forecast rather than an error"* | `predict.py:80`; `PROGRESS.md:318` |
| 3 | `operator_null_control.py` emits an absent row rather than a NaN row, so the gap reads as a refusal. Three readers still took it for "never measured" | `operator_null_control.py:164`; `PROGRESS.md:323` |
| 3 | The `unknown_position` quarantine and the adoption path, neither of which had ever run, executed together and unattended on 23 September: two broker fills arrived before the book knew of them, were quarantined, then adopted under the decisions that produced them | `logs/live-2026-09-23.log`; `positions adopted : 2` |

### A selection effect in the defects themselves

Five of the findings above were examined for the direction in which they erred, and the
pattern is not a property of any one of them.

| defect | direction | how it was found |
|---|---|---|
| the shuffled control is inert | calm | side effect of scoping an arm |
| `require_current_config` unwired | calm | side effect of costing a config section |
| duplicate rows pull Sharpe down, away from the 2.0 alarm | calm | side effect of counting rows |
| `broker=` as a count reads as bookkeeping | calm | side effect of reading `reconcile` for something else |
| `reconcile.py:287` warns every cycle | alarm | **reported**, by the person reading the log it fires in |

Four of the five point in the reassuring direction, and all four were stumbled upon while
looking for something else. The one pointing at alarm is the only one that was reported.
Its eighteen days undetected are not a counter-example; they are the cost of nobody reading
that log until the night it was read.

That contrast is the evidence. An error that points at trouble acquires a reader; an error
that points at calm does not. **A defect that makes a number look worse is found by the
person it annoys. A defect that makes a number look better has no such reader.**

The class also predicts where to look: at quantities sitting just below a threshold that
would have triggered an investigation. `buy_and_hold`'s 1.8931 against the 2.0 alarm in
§7.3 is exactly that shape, and it is exactly where the third instance was hiding. The
free version of the practice is to record, when closing a defect, which direction it erred:
erring toward calm is evidence of siblings, erring toward alarm is evidence of a reader.

### Two corrections this chapter's own drafting produced

Both are P2 instances committed while writing the material above, and they are reported
here rather than quietly fixed.

The first is the `sd` column. An earlier draft asserted it moved "the wrong way" on
total_return and gave 0.0129 → 0.0162 as the evidence. The mechanism was real; the figures
were not those figures. One was a genuine arm spread and the other was a mean MAE from a
different table. Recomputed against `stats.py` over the 39 total_return cells present at
both universe sizes, the true comparison is stronger than the claim it replaced: the
reported spread rose ×1.42 and grew in 25 of 39 cells, while the paired difference the test
actually consumes fell ×0.38 and fell in **39 of 39**. Two quantities moving in opposite
directions, one of them unanimous, in adjacent columns of the same table.

The second is smaller and the reason it matters is the same. A sentence claiming that the
first pass at a filter had dropped the buy-and-hold row was a recollection with no committed
artefact behind it. The mechanical claim stands on its own and does not need the anecdote:
224 of `results.csv`'s 1,110 rows are `buy_and_hold` and **none** carries a channels value,
so a bare `channels == C0_base` filter cannot match them. It does not merely fail to; it
cannot.

A chapter about green results answering different questions that declined to report its own
would be worth less than one that does.

### What the principles cost, and what they are worth

None of this was free. The universe flip alone re-ran a 101-minute grid and re-quoted every
corrected p-value in the study. Making one test double faithful turned 26 tests red across
three files. The spec-parsing guard exists because a contradiction inside one document
survived unseen for weeks, and the only reason anybody could state how long is that the
commits are dated.

What they bought is the one thing a null result cannot do without. A study that reports
finding nothing is indistinguishable, from the outside, from a study whose pipeline was
disconnected. §57.3 is the answer to that, and §57.3 is only believable because the
instruments in this section were held to the standard described here. The control that
degrades is worth exactly as much as the reader's confidence that it would have been caught
had it not.

That is the claim this section makes, and it is narrower than it might appear. Not that
this project's code is unusually correct. That its **reporting** was held to a standard its
own findings repeatedly proved it needed, and that the standard is transferable in a way the
trading results are not.
## 57.11 — Future work

Each item is a question this study leaves open, not a feature list.

**Cross the grid-anchor axis with the null-control axis.** The star design puts the control
spokes at the centre, so "flat at all three anchors under both controls" is unavailable
rather than unmeasured (§57.9). Closing it is a cross rather than a star: 108 conditions
against 14.

**Make the chapter's figures traceable by mechanism rather than by discipline.** Every number
here is re-derivable from a committed file, and nothing enforces that. A test that extracts
the figures from the chapter and confirms each against its named source would fail when a
number goes stale, and its cheapest fix would be to re-read the number. This was scoped and
not built; the guard originally proposed instead — one asserting the absence of unsourced
markers — was refused for the reason in §57.10, that its cheapest path to green is deleting
an admission.

**Record which term bound in `room_for`.** The risk layer binds without leaving evidence in
both of its modes (§57.7). The instrumentation added for this study observes beside the
sizing path; making the sizing path itself state which constraint applied would remove the
need for it.

**Whether the gross cap now selects more than the ranking does.** 55% of sizing calls are cap
events (§57.7). The expansion's two stated purposes pull against each other and this study
does not resolve which dominates.

**The geometry critique on a third architecture.** WITS established that the critique is
Fourier-specific by construction (§57.5). Whether it is specific to global bases generally
needs a third transform that is neither global nor a wavelet.

**A third feature configuration.** Declined here with its costs stated (§57.9). The open
question it would answer is narrow: whether the null result is an artefact of two feature
sets chosen by one author.

**Horizons other than daily.** Every claim in this chapter is scoped to a four-day horizon on
daily closing prices, and says so. Nothing here bears on intraday structure.

**A universe containing delisted securities.** This would bound the survivorship exposure
that §57.9 states and cannot measure. It needs point-in-time index membership, which is a
data acquisition problem rather than a modelling one.

**Cross-architecture reproducibility.** Determinism is bit-identical on one machine and
untested across architectures (§57.9).

**Unattended operation.** The live loop is launched by hand for one session at a time. A
scheduler is a few lines; what is not a few lines is the decision to remove the human approval
step, and the constraint in §57.6 that no fractional position can carry protection across a
close. A system shown not to beat its reference does not warrant either change.