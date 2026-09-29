# GB-58 — Presentation Deck

Fifteen slides, roughly twelve minutes, plus a live demo. Speaker notes are the substance;
the slides carry the minimum a room can read from a distance.

**The structure is not chronological.** It leads with the instrument, then shows the instrument
working, then reports what it found. A chronological telling would arrive at the null result
before the reader has any reason to believe it.

| # | slide | purpose |
|---|---|---|
| 1 | The claim | one sentence, before anything else |
| 2 | What was built | the system, in one diagram |
| 3 | The question | what is actually being asked |
| 4 | The instrument | walk-forward, references, controls |
| 5 | **The instrument works** | the control that degrades |
| 6 | The null result | the headline table |
| 7 | Why you can believe it | the asymmetry, stated |
| 8 | MAE measures flatness | the metric finding |
| 9 | The geometry critique | FITS is mostly grid |
| 10 | The execution finding | two broker refusals |
| 11 | **Live demo** | a decision, explained |
| 12 | What universe size changed | power, not effect |
| 13 | The methodological contribution | three principles, one case each |
| 14 | Limitations | stated, not discovered |
| 15 | What this is worth | the close |

---

## Slide 1 — The claim

**On the slide:**

> Most trading systems report profits that turn out to be artefacts.
> I built one that can prove when it is being fooled.

**Speaker notes.** Open here and nowhere else. The room will otherwise spend the first five
minutes waiting to hear whether the bot made money, and the answer will land as a failure
rather than as a finding. Say the sentence, pause, and move on. Do not defend it yet.

---

## Slide 2 — What was built

**On the slide:** the seven-layer diagram, one line each.
L1 Data → two sources, one shape
L2 Features → one place a model input is assembled
L3 Models → four forecasters, one frozen protocol
L4 Engine → event-driven backtester, validated to the cent
L5 Execution → real broker, risk caps, human approval
L6 Explain → Σ per-channel = forecast, exactly
L7 Console → reads files, computes nothing


**Speaker notes.** Thirty seconds. The point is scale and completeness, not detail: this is a
working system that trades a real broker, not a notebook. If asked for numbers: 1,565 tests
collected, twenty symbols, sixteen walk-forward folds, 322 cycles in a single live session with
zero failed steps. **Say "collected", not "passing"** — the last recorded full run predates the
guards added since, so a pass count is not established.

---

## Slide 3 — The question

**On the slide:**

> Do DLinear, FITS and WITS extract structure from daily closing prices
> that is not also present in white noise?

**Speaker notes.** Name the three as published architectures, not as things you invented. This
is the slide that makes the project a study rather than a build. The question has a falsifiable
answer and the rest of the talk is the answer plus the evidence that it can be trusted.

---

## Slide 4 — The instrument

**On the slide:** three bullets and one table.

- Sixteen walk-forward folds, H-bar embargo
- Every result a delta against its own reference
- Two robustness tests on every headline claim

| metric | reference |
|---|---|
| MAE | persistence |
| direction | always-long |
| Sharpe, return | buy-and-hold |

**Speaker notes.** One minute, and this is the slide that earns the rest. Three references
because they answer different questions; no composite score because a single number would hide
the disagreement between them, which turns out to be slide 8. Two robustness tests because each
has killed a finding the other passed — name both if there is time, and if there is not, say
"each has killed a finding the other passed" and move on.

---

## Slide 5 — The instrument works

**This is the most important slide in the deck.**

**On the slide:** one table, four rows.

| arm | noise − real | p (Holm) |
|---|---|---|
| **buy_and_hold** | **−0.0517** | **0.0134** |
| DLinear | +0.0184 | 0.0521 |
| FITS | +0.0087 | 0.6968 |
| WITS | +0.0035 | 0.8999 |

One line beneath it:

> Buy-and-hold is the only arm that degrades when the signal is destroyed,
> and it is the only arm whose mechanism says it must.

**Speaker notes.** Ninety seconds, and slow down here.

A null result and a disconnected pipeline produce the same table. Nothing on slide 6 can tell
them apart. So before showing what was found, show that the instrument can detect something
when something is there.

Buy-and-hold makes no forecast — its entire content is the market's upward drift. The
white-noise control removes exactly that drift. So it must degrade, and it does, by 5.17 points,
significant after correction. The models do not move, because they never had a forecast to lose.

The sentence to land: **a detector that stays silent is worth something only once it has been
shown to fire.**

---

## Slide 6 — The null result

**On the slide:** the headline table, five columns.

| arm | real | white noise | shuffled |
|---|---|---|---|
| always-long bar | 0.5528 | 0.5011 | 0.5365 |
| DLinear | 0.4981 | 0.5156 | 0.5053 |
| FITS | 0.4948 | 0.5015 | 0.5026 |
| WITS | 0.4976 | 0.5001 | 0.5025 |

> These models extract nothing from this market
> that they do not also extract from white noise.

**Speaker notes.** State the claim, then immediately state what it is not. **It is a claim about
the models, not about the market.** If it is left unqualified the room supplies "so the market is
efficient", which the study does not claim and cannot support.

The evidence that structure exists and is reachable: an always-long rule beats chance by 5.3
points over the same period, and buy-and-hold compounds to +125% against the model's +45%. **The
structure is there. It is not reachable by frequency or wavelet decomposition at this horizon.**

If asked about DLinear scoring higher on noise than on real data: it fails Holm at 0.0521, just
above the threshold, so it is reported as an unresolved gap and not as a finding. Do not claim
it.

---

## Slide 7 — Why you can believe it

**On the slide:** three lines, no table.

- The control moves for exactly one arm
- That arm is the one whose mechanism says it must
- It does not move for the three that never had a forecast

**Speaker notes.** Thirty seconds. This slide exists because slides 5 and 6 are usually
remembered separately, and the argument is the relationship between them.

Without the buy-and-hold row, slide 6 is equally consistent with a pipeline that silently
disconnected. With it, the pipeline is demonstrably connected and the models are demonstrably
not using what flows through it.

**What this does not establish:** sensitivity at small magnitudes. 5.17 points is a large
effect. Say so before someone asks — the honest claim is that the instrument is demonstrably
not dead, not that it is demonstrably sensitive to anything a model might have found.

---

## Slide 8 — MAE measures flatness

**On the slide:** the arm table and one coefficient.

| arm | mean MAE | mean flatness |
|---|---|---|
| persistence | 0.0128 | 0.000 |
| WITS | 0.0130 | 0.157 |
| FITS | 0.0131 | 0.188 |
| DLinear | 0.0134 | 0.249 |

> **Spearman(MAE, flatness) = +1.000**
> The metric carrying most of this study's robust results
> measures how close to zero a forecast is.

**Speaker notes.** One minute, and this is the slide a methods-minded examiner will remember.

Of 29 results surviving Holm correction, 22 are MAE and all say a model is worse than
persistence. Persistence forecasts exactly zero: lowest error, no flatness. The models forecast
slightly more than zero and are penalised in proportion.

So the finding is not "the models are inaccurate". It is **"MAE ranks these arms by how little
they say"**, which is a claim about evaluation practice in the field rather than about this
project.

If asked why +1.000 and not the +0.072 that appears in an earlier draft: the claim is about how
six arms rank, so the statistic is computed over six arms. Pooling 624 arm-folds measures
something else. **That level error was in this project's own report generator, and fixing it is
one of the instances on slide 13.**

---

## Slide 9 — The geometry critique

**On the slide:**

| what is measured | five symbols | twenty symbols |
|---|---|---|
| gain-magnitude response, real vs noise | r² 0.804 | **r² 0.957** |
| signed operator, FITS | −0.013 (t = −0.44) | **+0.195 (t = +7.24)** |
| signed operator, WITS | +0.033 (t = +0.75) | +0.084 (t = +2.56) |

> 95.7% of what FITS learns is reproduced by training on white noise.

**Speaker notes.** Ninety seconds. This is the highest-status finding in the deck for a
CS audience, because it is a measured criticism of a published architecture.

The mechanism in one sentence: extending a window from length L to L+H moves every frequency's
position in the output grid, so the layer must learn an interpolation that depends on the grid
geometry and not on the data. Train the same architecture on noise and the learned response is
nearly identical.

**Say the correction out loud.** An earlier version of this claim said 86%, and that figure is
withdrawn because its denominator was never recorded. The reproducible figure is r² = 95.7%,
and it is larger. Volunteering that is stronger than being asked.

**WITS, stated honestly:** it does not escape the pathology, it escapes it less — +0.084 is not
zero either, and the WITS-versus-FITS comparison holds at p = 0.0042.

**And the finding GB-66 produced:** the acceptance criterion defined for WITS is structurally
unmeasurable. FITS's response curve has 23 points; WITS has two bands; a correlation over two points is
±1 by construction. The critique is Fourier-specific by construction, **and that was learned
only by building the substitute.**

---

## Slide 10 — The execution finding

**On the slide:** two API responses, verbatim.
{"code":42210000,"message":"fractional orders must be simple orders"}
{"code":42210000,"message":"fractional orders must be DAY orders"}

> A fractional position at this venue cannot carry protection past the close.
> The backtest in the literature describes a system that cannot be built here.

**Speaker notes.** One minute. This is the finding a room of engineers responds to fastest.

The standard construction is a resting stop and a resting target, both live at the broker,
either able to fill intraday. Two refusals remove it: no multi-leg order on a fractional
quantity, and no GTC on a fractional quantity. A working sell also holds the entire position,
measured against 97.38 shares, so this is not a fractional-only rule.

**A simulator would never produce this.** It is discoverable only by connecting to a real
broker and reading the refusal.

Measured on a live position: stop `8820c4d7`, submitted 19:42:11Z with `time_in_force=day`
**read back from the broker rather than from the code's intent**, expired at 20:01:31Z.

The study measures both systems. Trade count falls 6%, 7% and 5%; the return difference does
not resolve and the sign is not even consistent. **The buildable system is measurably different
and not measurably worse.**

---

## Slide 11 — Live demo

**On the slide:** nothing. Switch to the console.

**Speaker notes.** Three minutes, and rehearse it twice.

**What to show, in order:**

1. **The status strip.** Universe 20, config hash, and the liveness band. Say that the date is
   yesterday's because today's bar has not closed, and that the panel says so.

2. **A single decision, expanded.** Pick one with heavy cancellation — the live session has
   several. Read the per-channel contributions aloud and then the surviving fraction: *"only
   19.5% of the gross view survived into the forecast."* This is the null result visible at one
   decision.

3. **A hold, and why it is a hold.** *"The calibrated entry band starts at +0.67% and the
   forecast reached +0.44%, so the threshold was not met."* A system that declines most bars
   needs its refusals legible.

4. **The positions table.** Three positions, each with entry beside last, unrealised with sign
   and percentage, stop room as distance to the stop. Every column carries its reference.

5. **The reliability figure.** Red, and a delta against the always-long bar. The finding, on
   the page, in the system's own words.

**If the loop is not running**, say so and show the recorded session instead. Do not improvise
a live launch in front of the room.

**Contingency:** have `report/screenshots/` open in a second window. If the console fails,
switch without comment and keep talking.

---

## Slide 12 — What universe size changed

**On the slide:**

| | five symbols | twenty symbols |
|---|---|---|
| Holm survivors | 17 | **29** |
| smallest corrected p, direction | 0.2605 | **0.0036** |
| mean effect size, total_return | 0.0827 | **0.0473** |

> The effect sizes got **smaller**. The measurement got three times more precise.

**Speaker notes.** One minute, and the trap here is being misheard.

The obvious reading is "with more data the models got worse". That is wrong and the table says
so: effect sizes are the same or smaller in 23 of 29 cells. **The number of folds did not
change either — 16 at both sizes.**

What changed is the paired-difference standard deviation, which is what the signed-rank test
consumes. It fell by roughly a factor of three, because each fold's metric moved from an
average over five symbols to an average over twenty.

**The strongest form:** on total return the models are *less bad* at twenty, and it became
significant anyway.

So this is a finding about the study's design rather than about the models: **Phase 1's null on
direction and total return was partly a small-sample artefact.** The deficit was always there
and a five-symbol design could not resolve it. It weakens only the negatives about the
negatives — all 29 survivors still have the model losing.

**And the same pattern appeared twice**, in the operator comparison on slide 9, which is what
makes it a claim about the design rather than a coincidence.

---

## Slide 13 — The methodological contribution

**On the slide:** three principles, one case each.

> **P1. Only something that runs is a mechanism.**
> A fact stored twice needs something making the copies equal.
> *The honesty rule named an invocation that could not run in this repository.*

> **P2. A green result answering a different question is the most comfortable kind of wrong.**
> *A silent feed downgrade to IEX passes every schema test. Prices out by 193 bps against a
> 1 bp tolerance.*

> **P3. A mechanism installed for one reason covers a failure nobody connected to it.**
> *A test written to prove weight sharing found 50 of FITS's 1,200 parameters allocated and
> unable to learn.*

**Speaker notes.** Ninety seconds, and this is the section a software-faculty examiner will
grade highest. Say the principle, then the case, then stop. Do not enumerate the other forty.

If asked how many instances there are: around forty-five, each carrying a commit, a file and
line, or a test node ID, in the results chapter's appendix. Instances that could not be
evidenced were cut rather than softened.

**One extra, if the room is engaged.** Five defects were examined for the direction in which
they erred. Four pointed at calm and all four were stumbled upon while looking for something
else; the one pointing at alarm was the only one reported. **A defect that makes a number look
worse is found by the person it annoys. A defect that makes a number look better has no such
reader.**

---

## Slide 14 — Limitations

**On the slide:** five, in one list.

- Survivorship bias is written into the universe selection rule
- The universe is admissible by the rule, not derived from it
- The shuffled control cannot falsify a whole class of claim
- Null controls exist at one fold-grid anchor, structurally
- Sixteen folds bound the smallest reachable p-value at 3.05 × 10⁻⁵

**Speaker notes.** One minute, delivered flat. These are stated rather than defended, and the
purpose of the slide is that the room hears them from you rather than finding them.

Two are worth a sentence each:

**Survivorship is in criterion 3 itself** — the rule admits only securities continuously
tradable through to the cache's last bar, which is knowable only in 2026. The bias sits inside
the reference each arm is measured against, so an arm beating a survivor-inflated buy-and-hold
would be a stronger claim, not a weaker one. Nothing in the repository can bound it.

**The shuffled control is inert against any signal that is a monotone function of accumulated
return**, because a permutation preserves a sum. Demonstrated on a momentum rule that scored
*better* on shuffled data than on real. The pre-registerable check is one line, and it did not
exist until September.

---

## Slide 15 — What this is worth

**On the slide:** four lines.

> A measured answer to a published question: three architectures, no signal.
> A measured criticism of one of them: 95.7% grid geometry.
> A venue constraint the literature assumes away.
> A discipline for knowing when a test is lying to you.

**Speaker notes.** Sixty seconds, and end on the last line.

The system does not beat buy-and-hold. It was built to find out whether it could, and the
answer is no, and **the value is that the answer can be trusted** — which is the one thing a
null result cannot do without.

If there is time for one closing sentence:

> **Most trading systems report profits that turn out to be artefacts. This one reports nothing,
> and can prove the instrument was working.**

Stop there. Do not add a future-work slide out loud; it is in the report.


