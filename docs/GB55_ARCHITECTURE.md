# GB-55 — Architecture and System Design

This chapter describes what was built. GB-56 establishes that it measures what it claims to;
GB-57 reports what it found. This chapter is the object both of those refer to.

The system is a seven-layer pipeline from market data to an explained trading decision, plus
an evaluation harness that runs the same code offline. The organising constraint throughout
is that **the offline and live paths share one implementation wherever a difference between
them would be invisible in the output.**

Figures in this chapter are captures of the running console, taken against a session that
actually ran. Where a figure documents a defect that has since been fixed, it is dated and
retained rather than refreshed, for the reason given in GB-57 §57.10.

## 55.1 — The seven layers

The system is seven layers. Each has one responsibility, one direction of dependency, and one
reason to exist that the layer above it could not satisfy.

| layer | responsibility | the failure it exists to prevent |
|---|---|---|
| **L1 Data** | fetch and cache bars from two sources | the feature layer behaving differently depending on where a bar came from |
| **L2 Features** | assemble a model input window | the training path and the live path building different windows |
| **L3 Models** | produce a forecast from a window | a model's interface leaking into the code that uses it |
| **L4 Engine** | turn forecasts into positions and equity | an evaluation that prices fills optimistically |
| **L5 Execution** | place orders, enforce risk, gate on approval | a decision reaching the broker without a bound on the loss |
| **L6 Explain** | decompose a decision exactly | an explanation that is a plausible story rather than the arithmetic |
| **L7 Console** | show state to an operator | a page that states something a reader will take as true and is not |

### The dependency rule, and where it is enforced

Dependencies run **downward only**. L2 may import L1; L1 may not import L2. Nothing in the
package imports the console, and the console imports from the live loop lazily, inside
functions, so an import cycle cannot form.

This is not a convention. The **import linter** runs in CI with two contracts and fails the
build on a violation, which is why the layering statement in this chapter is a description of
what the code does rather than of what it was meant to do.

### The property the whole design turns on

**One implementation wherever a difference would be invisible.**

Two paths need bars: the grid and the live loop. Two paths need a feature window. Two paths
need a forecast, a signal, a position size. In each case a second implementation would produce
plausible numbers on both sides and nothing would fail — the defect class GB-57 §57.10 is
about, in the places where it would be most expensive.

So:

- `normalise_bars` is the single shape both data sources produce, and the feature layer cannot
  tell yfinance from Alpaca.
- `features/builder.py` is the only place a model input window is assembled, and the parity
  sweep in GB-56 §56.4 asserts byte-identical output across 500 comparisons.
- The `Forecaster` protocol is frozen, and all four models satisfy it, so the engine has no
  branch on which model is running.
- The backtester and the live executor share the signal and sizing path; only the fill differs,
  because one is simulated and one is a broker.

### What runs where

The **grid** (`experiments/study.py`) runs L1 to L6 offline over sixteen folds and fourteen
conditions, writes `results.csv`, and never touches a broker.

The **live loop** (`live_loop.py`) runs L1 to L6 against real market data and a real broker,
once per completed bar, and writes decision records.

The **replay** runs the live loop's own code over historical decisions, which is how the
spectral panel's decomposition was verified on 67 of 67 records.

The **console** (L7) reads what the other three wrote and computes nothing that is not already
in a file.

That last point is deliberate and is a load-bearing constraint rather than a stylistic one: a
console that computes its own figures is a second implementation of every metric it shows.

## 55.2 — L1 Data: two sources behind one shape

Two sources provide bars. **yfinance** supplies the historical cache used by the grid;
**Alpaca** supplies live bars during a session and is the broker the system trades through.

They return different column names, different index types, different treatments of splits and
dividends, and different behaviour on a missing session. If any of that reached the feature
layer, a model trained on cached history would receive subtly different input when it ran
live, and nothing downstream would report it.

### `normalise_bars` is the boundary

Both adapters produce the same shape through one function: the same columns in the same order,
the same dtypes, a timezone-aware index on the exchange calendar, and adjusted prices in both
cases. **Nothing above L1 can tell the two sources apart**, and that is the property, not a
convenience.

The adjustment choice is worth naming. Raw prices contain split discontinuities that a return
series reads as a one-day move of several hundred per cent — a known hazard this project hit
before the adjustment was made explicit. Both paths request `adjustment=all`.

### The cache, and why it is committed

Twenty symbols from 2016, cached as parquet, **committed to the repository**.

Committing data into a source repository is unusual and the reason is GB-56 §56.8's audit: a
clean clone with no data can run nothing, so a result that cannot be reproduced from a clone
is a claim rather than a measurement. The cache is what makes the clean-clone audit possible
at all.

It also makes the grid deterministic in a second sense: the vintage is fixed. Every row in
`results.csv` carries `data_snapshot_last_bar`, and a file whose rows disagree about it is
refused rather than averaged.

### Quality, checked rather than assumed

A data quality module reports gaps, NaNs, split artefacts and calendar mismatches per symbol,
and it is run when the cache changes rather than once at the start. Across the twenty-symbol
cache it reports zero missing bars against the NYSE calendar.

The live path checks a different property: it requests **890 calendar days to obtain at least
445 bars each**, because the parity floor is expressed in trading bars and the request is in
calendar days. The margin is deliberate and sized for holidays.

### The in-progress bar, dropped explicitly

A bar for today exists from the moment the session opens and is not final until it closes.
Using it would be look-ahead of the plainest kind.

The live loop drops it, per symbol, with the reason logged: *"dropping in-progress bar
2026-09-23 for AAPL: the 2026-09-23 session has not closed."* The decision is therefore taken
on yesterday's completed bar, which is why the console shows a date one day behind during an
open session and why the console now says so in words.

## 55.3 — L2 Features: the keystone builder

This layer turns bars into the array a model sees. It is called the keystone because it is the
one place where a difference between the offline and the live path would be both consequential
and invisible, and it is the layer GB-56 §56.4 is entirely about.

### The channels

Five channels in the deployed configuration, `C0_base`:

| channel | what it is | warm-up |
|---|---|---|
| `close_logret` | log return of the close | 1 |
| `rsi14` | Wilder's relative strength index, period 14 | 325 |
| `vol_z` | z-score of realised volatility | 20 |
| `mom10` | ten-bar momentum | 10 |
| `ma_dist20` | distance from a twenty-bar moving average | 20 |

A second configuration, `C2_hybrid`, adds three causal wavelet bands — `wav_a1` to `wav_a3` —
and exists so the study has two feature sets rather than one. GB-57 §57.9 states what having
two rather than three costs.

Every channel is **pure and trailing**: a function of bars up to `t` and nothing else, with no
state carried between calls. That is asserted rather than intended, by the causality harness
in GB-56 §56.3, in both perturbation modes at three split points, and each assertion was
verified by introducing a deliberate look-ahead.

### The window, and the floor that makes it comparable

A model input is `input_len = 120` bars across the active channels. Producing a valid window
at bar `t` requires history before `t` for the channels to warm up, and the requirement is a
**maximum over channels, not a sum**:
min_history_bars = input_len + max(warm-ups) = 120 + 325 = 445


A sum would be 496 and would be wrong: the warm-ups run concurrently in time, not in sequence.

**325 is itself derived.** Wilder's RSI is recursive — each value depends on the previous one —
so the influence of the first bar decays by 13/14 per step rather than ending. 325 bars is
where that influence falls below 1e-10.

Below the floor the two paths disagree in the last decimal places purely because one has seen
more history. This is not a hypothesis: at an earlier 352-bar floor the parity sweep found only
**32 of 125 pairs identical**.

The floor is recomputed rather than assumed whenever the configuration changes. When the
universe moved from five symbols to twenty it stayed 445, because universe size appears nowhere
in the derivation.

### Why the wavelet channels needed the harness most

A standard discrete wavelet transform is **not causal**: it decomposes the whole signal, so a
coefficient near the start is influenced by values near the end. Using one as a feature would
be look-ahead of a kind that is invisible in the output — the series looks like a perfectly
ordinary smoothed price.

The implementation here is a trailing variant, and the property that makes it usable is exactly
the one the harness tests. **That is the difference between a feature the author believes is
causal and a feature that is asserted causal by something that fails when it is not.**

### One function, three callers

`features/builder.py` is called by the training path, the live loop and the replay. No second
implementation exists, the import linter forbids one, and the parity sweep asserts
byte-identical output across 500 comparisons at the 445-bar floor on every suite run.

GB-56 §56.4 treats why byte-identical rather than close, and records that the sweep's own count
is derived from `SWEEP_TIMESTAMPS * len(cfg.universe)` rather than written as a literal — a
sibling test that pinned the same product as `25` broke on the universe flip while this one did
not.

## 55.4 — L3 Models: four forecasters behind one protocol

Four arms produce forecasts. They are deliberately small, and the smallness is a design
decision rather than a limitation of effort.

| arm | parameters | what it is |
|---|---|---|
| persistence | 0 | predicts zero change; the reference for MAE |
| DLinear | 4,800 | decomposition into trend and remainder, one linear map each |
| FITS | 1,200 | real FFT, low-pass cutoff, learned complex linear map, inverse |
| WITS | 882 | discrete wavelet transform in the same position as FITS's rFFT |

`buy_and_hold` is a fifth arm in the study but not a forecaster: it holds everything and
predicts nothing. It is the reference for Sharpe and total return, and GB-57 §57.3 turns on
its behaviour under a null control.

### One frozen protocol

All four satisfy the same `Forecaster` protocol, and the protocol is **frozen**: it is part of
the specification's contract section and does not change to accommodate a model.

The consequence is that nothing above L3 branches on which model is running. The engine, the
signal layer, the executor and the console handle a forecast, not a DLinear forecast. When
WITS was added, no code outside L3 changed to accept it.

A frozen interface has a cost, and it was paid once: GB-66's acceptance criterion could not be
satisfied because WITS has no per-bin frequency response to compare against FITS's. **The
protocol did not bend to make the comparison possible**, and GB-57 §57.5 reports the finding
that produced.

### Universe-wide shared weights, per-symbol scalers

One model is trained across the whole universe rather than one model per symbol. With twenty
symbols and 4,800 parameters, per-symbol training would mean twenty models of 4,800 parameters
on a fifth of the data each.

Normalisation is **per symbol**, held in the checkpoint as a table of statistics. This is the
division that matters: the weights are shared because the pattern is assumed universal; the
scale is not, because a 400-dollar stock and a 100-dollar stock are not on the same scale.

That division is also what caught a deployment error. When the universe moved to twenty
symbols, the startup gate compared configuration hashes, found the **model-shaping** sections
unchanged, and passed the five-symbol checkpoint as valid — true, and irrelevant. Twelve
seconds later `predict.stats_for` refused to score a symbol the checkpoint had no statistics
for, with a message stating why: *"scoring an untrained symbol would feed the shared weights a
distribution they never saw, and the result would look like a forecast rather than an error."*
GB-57 §57.10 files it.

### Why small models

Three reasons, and none of them is that a larger model was out of reach.

**The question is about the transform, not about capacity.** FITS and WITS differ in one place
— rFFT against DWT — and everything else is held identical so the comparison isolates it. A
larger model would add a confound.

**Determinacy.** At twenty symbols the ratio of observations to parameters is 8.35 to 1. A
model with more parameters than the data supports produces a result about the fitting procedure
rather than about the market, which is precisely the failure this study exists to detect.

**The null result is stronger from a small model.** A large model that finds nothing invites
the reading that it was undertrained. A 1,200-parameter linear map that finds nothing on the
same data as a 4,800-parameter one, and a 882-parameter one alongside them, says something
about the signal rather than about the search.

### What is not here

No recurrent or attention architecture, and the exclusion is scope rather than principle. The
question this study asks — does frequency or wavelet decomposition extract structure from daily
closes — is answered by the four arms above. A transformer would answer a different question
and would carry a determinacy problem the study is designed to avoid.

## 55.5 — L4 Engine: the event-driven backtester

The engine turns a stream of forecasts into positions, fills and an equity curve. It is the
layer where an evaluation becomes optimistic if anything is allowed to happen out of order.

### Event-driven, not vectorised

A vectorised backtest computes signals for the whole series and then multiplies by returns. It
is fast, and it makes several errors easy to commit invisibly: a position sized on a price it
could not have known, an exit priced at a level reached earlier in the same bar, an entry and
its own stop filling on the same bar with the ordering chosen implicitly.

This engine processes one bar at a time, in order, with a fixed sequence within the bar: open
positions are marked, exits are evaluated before entries, sizing sees only the equity that
existed before this bar's entries, and nothing is priced at a level that was not available
when the decision was made.

It is **validated to the cent** against hand-computed fixtures — a small set of trades whose
expected cash, quantity and equity were worked out by hand and written as literals, so a
change to the engine that alters an outcome fails rather than producing a different plausible
number.

### Walk-forward is the engine's caller, not its content

The engine does not know about folds. `experiments/study.py` calls it once per fold with that
fold's test slice, and the fold structure, the embargo and the anchors live in the caller.

This is why the same engine runs the grid, the replay and the live path's simulation of a fill:
there is nothing fold-shaped inside it to disagree with the live loop.

### Signal and sizing, shared with the live path

Three steps turn a forecast into an order, and all three are the same code offline and live:

**Signal.** A forecast crosses the calibrated entry threshold, or it does not. The threshold is
per fold and calibrated on validation only — GB-56 §56.5. A second gate applies after it: a
forecast admitted by the band still requires a majority of its horizon steps to point the same
way, which is why the live log shows decisions of the form *"the band admitted the forecast
(+0.72%), but only 2 of its 4 steps point up, against the 3 required."*

**Ranking.** Where more candidates clear the band than may be entered, `rank.py` orders them.
In the deployed configuration `top_k = 2` caps new entries per bar. This is live-only: the
backtest enters every symbol that clears its band, bounded by the exposure cap rather than by
a count, and GB-57 §57.7 records the consequence.

**Sizing.** `risk.py` returns a notional as the minimum of three terms: the per-position cap at
10% of equity, the gross headroom against a 50% cap, and available cash.

### The risk layer's one structural weakness, named here

`room_for` returns `max(0, min(A, B, C))` and **discards which term was the minimum**.

When the gross headroom merely reduces an entry, the notional comes back positive and the trade
looks ordinary. When it blocks, the entry exits through a path that is ambiguous between the
cap, exhausted cash and a non-positive equity early return.

So the risk layer enforces correctly and leaves no evidence of having done so, in either mode.
Measuring how often the cap binds required a separate instrumented pass that observes beside
the sizing path — GB-57 §57.7 reports 13 of 16 folds and 963 sizing calls, and records that an
instrument watching only for zero notionals would have counted 28% of the events and concluded
the opposite.

### What the engine cannot model, and how that was discovered

The engine originally exited a position through a resting stop **or** a resting target, both
live at the broker, either able to fill intraday. That is the construction the literature
assumes.

Alpaca refuses it. The engine therefore carries a `target_in_loop` axis: `false` for the
classical broker-side target and `true` for the buildable loop-side one, with `true` as the
default so the headline numbers describe the system that exists. GB-57 §57.6 reports both.

This is the one place where the architecture was changed by a measurement rather than by a
design decision, and it is recorded that way.

## 55.6 — L5 Execution: broker, risk and the Co-Pilot gate

L5 is where a decision becomes an order at a real venue. It is the only layer whose failures
cost money rather than accuracy, and it is designed around the assumption that **the broker is
the truth**.

### The broker is the truth

Local state can be wrong. The process can be killed mid-order, an order can fill after the
loop stopped reading, a position can exist that the system did not open.

So reconciliation runs every cycle and treats Alpaca's answer as authoritative. It reports
three divergences:

| divergence | meaning | handling |
|---|---|---|
| `unknown_position` | the broker holds something with no decision behind it | quarantined: counted against buying power, never traded |
| `missing_protection` | a managed position has fewer live protective orders than required | armed immediately; two consecutive arming failures flatten it at market |
| a sell filled for an unmanaged symbol | no entry basis exists | warned, no trade emitted |

The quarantine rule is strict on purpose: **the system cannot protect or explain what it did
not open**, so it refuses to manage it rather than adopting it silently.

There is an adoption path for the case where the quarantine is the system's own fill catching
up with its book. On 23 September both ran unattended for the first time: two entries filled
before the book knew of them, were quarantined as unknown, then adopted under the decisions
that had produced them, each booked with stop and target and armed in the same cycle.

### Protection, and what the venue allows

Every managed position carries a protective stop at the broker. The intended construction was
a stop **and** a target, both resting, either able to fill intraday.

Alpaca refuses it, twice over: no multi-leg order class on a fractional quantity, and a working
sell holds the entire position so two standalone legs cannot coexist. A second refusal forbids
GTC on a fractional order, so the stop is `time_in_force=DAY` by API rule and expires at every
close.

The consequences are structural rather than incidental:

- **The target moved into the loop.** It is evaluated against completed daily bars, so it
  cannot observe an intraday crossing and acts at the next open.
- **A position carries no protection between the close and the next session's first arming.**
  The loop re-arms before any entry, and the session banner states the residual and the reason
  rather than leaving it to be discovered.
- **The DAY-to-GTC question is not open.** It is unavailable, and taking it would require
  whole-share sizing, which changes position sizing and the fractionable criterion in the
  universe rule.

GB-57 §57.6 reports this as a result. It is stated here as a property of the architecture,
because the architecture is shaped by it.

### The Co-Pilot gate

The deployed mode is `co_pilot`. Exits, reconciliation and protection run autonomously; **a new
entry is queued for approval and does not reach the broker until an operator approves it in the
console.**

The asymmetry is deliberate. Exiting and protecting reduce risk and are time-critical; entering
adds risk and is not. A system that can close a position without asking but cannot open one is
the conservative shape of the same machine.

A fully autonomous mode exists in configuration and is not deployed. GB-57 §57.11 states the
reason: a system shown not to beat its reference does not warrant removing the human from the
loop.

### One rule that turned out to matter for a reason it was not adopted for

A decision is taken **once per completed bar per symbol**, and repeated cycles within the same
bar do not re-decide. The rule was adopted to stop the loop writing roughly 1,950 identical
records a day — 390 cycles across the five-symbol universe of the time, which is 7,800 a day
at twenty.

What it actually prevented was a stray relaunched rehearsal opening a second position against
a live account, because the bar it would have decided had already been decided. GB-57 §57.10
files it, with the qualifier that matters: it refused **by accident**, and a new bar completes
every session, so that protection expires. The lock exists because of it.

## 55.7 — L6 Explain: exact attribution and narration

This is the layer the system is named for. A Glass Box is not a system that produces an
explanation; it is one whose explanation **is the arithmetic of the decision** rather than a
plausible account of it.

### Exact, and what exact means

For every forecast, the layer decomposes the output into per-channel contributions such that
Σ per_channel == forecast

to within 1e-5, asserted rather than hoped for, and measured at **5e-08 over 1,000 windows**.

This is possible because the models are linear. A linear map's output is the sum of its
inputs' contributions by construction, so the decomposition is not an approximation of the
model's behaviour — it **is** the model's behaviour, rearranged.

That is the distinction from attribution methods applied to opaque models. SHAP and LIME
estimate what a model might be doing; this reports what it did. The cost is that it only works
for this class of model, and that cost is a design choice: the study's question is about what
frequency and wavelet decomposition extract, and both are linear.

### Per-frequency and per-level

Two further decompositions exist, one per architecture:

**FITS** decomposes by retained frequency bin. Which bin contributed what to the forecast, at
24 retained bins under the deployed cutoff, of which the frequency-response curve reports 23:
the infinite-period DC entry is dropped, because it cannot be placed on a period axis and is
the dead row in any case. This is the quantity GB-57 §57.5's geometry critique measures.

**WITS** decomposes by wavelet level. Which band contributed what, over two retained bands.

They are not comparable, and that non-comparability is GB-66's finding rather than a gap: a
correlation over two values is ±1 by construction, so the acceptance criterion defined against
FITS's per-bin curve cannot exist for WITS.

### Cancellation, and why it is printed

The decomposition routinely shows large opposing contributions that nearly cancel. From a live
session:

> NVDA: `rsi14 −59.4%, ma_dist20 +14.9%, mom10 +14.6%, close_logret +10.7%, vol_z −0.3%`.
> *These largely offset: only 19.5% of the gross view survived into the forecast.*

A forecast of −0.71% that is the residue of contributions summing to five times that in
absolute terms is a different object from a forecast of −0.71% on which the channels agree.
The narration states which it is, every time, with the surviving fraction.

This is not decoration. It is GB-57 §57.4's finding visible at the level of a single decision:
the models produce forecasts close to zero because their channels disagree, and the metric that
rewards them for it is MAE.

### Narration

Each decision is rendered as prose, in English and Hebrew, from the decomposition and the
thresholds — not from a template with the numbers substituted, but derived, so a sentence
cannot assert something the numbers do not support.

That distinction was learned. GB-57 §57.4 records a sentence in the report that was a fixed
string and printed a conclusion whatever the coefficient turned out to be. The narration layer
was built the other way and the report was fixed to match.

The narration also states the **decision boundary it did not cross**, which is what makes a
hold legible: *"the calibrated entry band for this fold starts at +0.67% and the forecast
reached +0.44%, so the threshold was not met."* A system that declines to trade on most bars
needs its refusals to be as explained as its entries, or the log reads as inactivity.

### The one thing the layer will not claim

`explain.py`'s own line, printed on the console beneath the reliability figure:

> **An explanation makes a decision legible; it does not make it right.**

The attribution is exact about what the model computed. It says nothing about whether the model
should be believed, and GB-57 is the chapter that answers that separately.

## 55.8 — L7 Console: the operator surface

The console shows state to an operator. Its design constraint is narrower than "be clear": a
page that states something a reader will take as true and is not, is a defect of the same class
as a wrong number in the report.

That constraint was not satisfied on the first attempt, and GB-57 §57.10 records five
instances of this layer misreporting the thing it exists to report. What follows describes the
design that resulted.

### It computes nothing

The console reads `results.csv`, `report/daily_equity.csv`, the decision records and the live
state files. It does not recompute a metric.

A console that computes its own figures is a second implementation of every metric it shows,
and the two would diverge silently. This is the same rule as L2's single builder, applied to a
surface rather than to a path.

### Every panel says where its data came from

Each panel carries a source pill: **LIVE**, **BACKTEST** or **REPLAY**, with a required detail
naming the artefact and its range — as the page read on 1 September 2026,
`BACKTEST results.csv folds 1-16` beside `BACKTEST daily_equity.csv folds 1-3`. The committed
`daily_equity.csv` has since been completed to all sixteen folds, so both pills read
`folds 1-16` today; the dated pair is kept because it is what the disagreement looked like.

The detail is not decoration. Two pills that disagree side by side are the design working:
they are two artefacts with two ranges, and the pill says so rather than leaving a reader to
assume one number covers both.

The pill was also the layer's first recorded defect. It was derived once for the page from the
frame most cards used, so two cards reading a different file inherited a claim about someone
else's rows and announced *folds 1-16* while holding three. Each region is now a function
taking exactly one frame, with the pill built inside the function that draws the chart, and a
test pins the signatures so no sibling frame is in scope to pick wrongly from.

### Three colour roles, enforced

| role | colour | rule |
|---|---|---|
| chrome | orange | labels, rules, frames. **Never a value** |
| data | blue ramp | attribution and spectral panels only |
| status | green and red | gain and loss only. **Never inside a data-encoding chart** |

A guard enumerates every chart builder from `dir(app)` rather than from a hand-written list, so
a new chart must be assigned a side rather than defaulting to unchecked.

The rule's hardest case was flatness. The system stands aside on most bars, so a calendar
colouring zero as a gain rendered the project's central finding — *the system declines to
trade* — as a mostly winning year. Zero is now a third state with a muted colour, because zero
is not a direction.

### Liveness has one source of truth

The status strip reports whether the loop is alive, in bands: **STOOD ASIDE**, **NOT
RESPONDING**, **NOT RUNNING**, **OUTSIDE MARKET HOURS**, **SLOW**, **LIVE**. Every surface that
shows liveness reads the same verdict object, so a stale loop reads stale everywhere on the
page simultaneously.

Before that, the status word was derived from **position count** and labelled as liveness: it
read IDLE with the loop writing three seconds ago and IDLE with the loop dead for twenty-eight
hours. The two thresholds were also written in two places. One verdict, applied once, replaced
both.

### The decision is legible, including the refusal

The console's job on a system that mostly declines to trade is to make the declining legible.
Each decision expands to its per-channel contributions, the surviving fraction after
cancellation, the threshold it did or did not cross, and the narration.

The reliability figure leads the page as a **delta against the always-long bar**, in red,
because that is the finding. The absolute return of the arm appears nowhere without its
reference beside it.

### Figures

`report/screenshots/` holds captures taken against sessions that actually ran, with
`CAPTIONS.md` stating what each shows and under what conditions.

Three frames are **retained and dated rather than refreshed**: they document defects that have
since been fixed, and fixing a defect destroys the ability to photograph it. GB-57 §57.10
states the rule.

## 55.9 — The live loop: ten steps, once per bar

The live loop is where all seven layers run together against a real venue. It polls every 60
seconds during the configured session and executes the same ten steps each cycle, in a fixed
order. The order is the design.

| # | step | why it is here and not later |
|---|---|---|
| 1 | read the broker's order history | nothing is decided before the venue's state is known |
| 2 | emit trades from sells filled since the last cycle | a closed round trip is recorded before anything new is opened |
| 3 | reconcile: the broker is the truth | divergences are known before they are acted on |
| 4 | protection: verify, re-arm, flatten on a second failure | **risk is bounded before any new risk is added** |
| 5 | fetch completed daily bars | the in-progress bar is dropped here, with the reason logged |
| 6 | build features through the keystone builder | the same function the grid calls |
| 7 | forecast the undecided symbols | already-decided bars are skipped, not re-decided |
| 8 | decide, rank, size | the band, then `top_k`, then `room_for` |
| 9 | explain, narrate, execute, persist | the decision record is written **before** the order is built |
| 10 | persist book and entry fills | local state catches up last |

### Four properties of that ordering

**Protection precedes entry.** Step 4 runs before step 8. A position that lost its stop
overnight is armed before any new position is opened, so the system cannot add risk while
existing risk is unbounded.

**A decision is recorded before it is acted on.** The decision record is written at step 9 and
the closing order is built afterwards. This is deliberate — a record that exists only if the
order succeeded would lose exactly the cases worth studying — and it has a consequence GB-57
§57.10 records: an EXIT record carries no order, whether or not the symbol was held, so the
record alone cannot distinguish a real close from a no-op.

**One decision per completed bar.** Repeated cycles within the same bar do not re-decide; the
log states `0 undecided, 20 settled`. The rule was adopted to stop 1,950 identical records a
day at five symbols — 7,800 at twenty — and turned out to prevent a stray rehearsal opening a
second position.

**The first cycle may not add risk.** Entries are withheld on cycle one and recorded on cycle
two, so protection has been verified at least once before anything is opened.

### Failure handling

A failed cycle is caught and polling continues. It is **not a circuit breaker**, deliberately:
a transient broker error should not stop a loop that still needs to protect open positions.
Consecutive failures are counted and logged with the reason, and the summary reports how many
cycles failed.

A protective arming that fails twice consecutively **flattens the position at market**. That is
the one place the system acts without asking in a way that reduces a position to zero, and it
is the correct asymmetry: unbounded risk is worse than an unwanted exit.

### What one session looks like

From 23 September 2026, launched once and left to run:
cycles completed : 322, 0 with a failed step
decisions recorded : 20
bars already decided : 6418 symbol-cycles not re-decided
orders submitted : 2
positions re-armed : 3
positions adopted : 2
cycles skipped : 0 on an unreachable broker or feed
overnight residual : 1 position(s) unprotected from the open until first arming
positions at exit : 3 STILL HELD

Six and a half hours, twenty decisions, 6,418 symbol-cycles correctly declined as
already-decided, and zero failed steps.

### The loop is launched by hand

`--sessions N` runs N sessions and exits; the process idles between sessions and re-arms at
each open. Nothing restarts it after a crash, a reboot or an ordinary exit, and there is no
scheduler or supervisor.

That is a deployment decision rather than a technical limit, and GB-57 §57.11 states the
reasoning: a system shown not to beat its reference does not warrant unattended operation, and
the constraint in §55.6 means no fractional position can carry protection across a close in any
case.

## 55.10 — What the architecture refuses to do

An architecture is judged on what it prevents as much as on what it permits. This section lists
the refusals, each with the mechanism that enforces it, because a refusal stated in a document
and not enforced in code is not a property of the system.

### It will not decide on an incomplete bar

A bar for today exists from the open and is not final until the close. The live loop drops it
per symbol, with the reason logged, and decides on yesterday's completed bar.

*Enforced by:* the drop is unconditional in the feature path, and the console states the
resulting date gap in words during an open session.

### It will not score a symbol the checkpoint has never seen

`predict.stats_for` raises rather than returning a number for a symbol with no normalisation
statistics in the checkpoint, and the message says why: *"scoring an untrained symbol would
feed the shared weights a distribution they never saw, and the result would look like a
forecast rather than an error."*

*Enforced by:* the raise. It fired for real when the universe moved to twenty and the deployed
checkpoint still held five, after the startup gate above it had passed the checkpoint as valid.

### It will not manage a position it did not open

A position at the broker with no decision behind it is quarantined: counted against buying
power, never traded, never protected. **The system cannot protect or explain what it did not
open**, so it declines rather than adopting silently.

*Enforced by:* reconciliation's `unknown_position` path, with a separate adoption route for the
case where the position is the system's own fill catching up with its book.

### It will not add risk while existing risk is unbounded

Step 4 precedes step 8. A position missing its stop is armed before any new position is opened,
and two consecutive arming failures flatten it at market.

*Enforced by:* the step order and the arming failure counter.

### It will not open a position without a human

Deployed mode is Co-Pilot. Exits, protection and reconciliation are autonomous; entries queue
for approval.

*Enforced by:* the executor, which creates a pending submission rather than an order. A fully
autonomous mode exists in configuration and is not deployed.

### It will not report a number without its reference

No result in GB-57 is an absolute number, and no console panel shows arm performance without
the reference beside it.

*Enforced by:* the metric companion table, which pairs MAE with flatness structurally rather
than by convention, and by the panel audit that drove the console rebuild.

### It will not take an LLM's word for anything

Sentiment from a language model is never a model input and never enters a backtest. The reason
is that an LLM's training data contains the evaluation period, so it is look-ahead the causality
harness structurally cannot catch, and it breaks bit-identical reproducibility.

*Enforced by:* absence, and by the ruling being standing rather than reconsidered per task.
GB-56 §56.9 states it in full.

### It will not call a suite green on an invocation nobody was told to run

No report says a suite passed unless `pytest` ran with no path and no marker and the exit code
was read directly rather than through a pipe.

*Enforced by:* the rule, and by the discovery that the rule had named a command which could not
run in this repository — fixed by making the named command work rather than by renaming the
rule.

### It will not let a fact live in two places without something comparing them

The spec's module tree against the package tree; the universe in the spec against
`settings.yaml`; pyproject against the lockfile; exit reasons across two modules.

*Enforced by:* a test per pair, each verified by breaking the copy and confirming the test
fails. GB-57 §57.10 records the instances where this was learned, including the ones where it
had not yet been applied.

## 55.11 — Known open behaviours

Four behaviours are known, recorded with file and line, and not fixed. They are listed here
rather than in a defect tracker because a reader of this chapter should know what the system
does, not only what it was designed to do.

**A broker read failure renders an empty positions card under a live pill.** `_broker_view`
catches `Exception` and returns empty collections, so a credentials expiry, a network fault or
a rate limit is indistinguishable on the page from a genuinely flat book. The positions table
iterates the broker's positions, so `book.json` alone cannot produce a row. The failure mode is
reachable and the page does not say so.

**Three chart data paths can leave their plot box.** A near-zero value range makes the
autoscale expand and the polyline is drawn outside the viewBox. Three builders are affected and
each is pinned by a strict expected-failure marker, so a fix that lands without removing the
marker fails the suite.

**The reconciliation detector requires two protective legs and the policy is one.** The
threshold survived the retirement of the two-leg policy because the sweep that retired it
searched for **language** — four prose sites — and a numeric threshold is not prose. The warning
therefore fires every cycle for a condition unreachable at this venue, and nothing acts on it:
the enforcer recounts from its own source and never reads the detector. A warning that can
never be off carries no information and trains a reader past the line that would matter.

**Decision records from two universes share one activity table.** Records carry `config_hash`,
which separates five-symbol from twenty-symbol decisions cleanly, but the console does not
compare a record's hash against the current configuration, so both appear in one list with
nothing marking the boundary.

### Why these are here rather than closed

Each is recorded with its anchor and each has a stated cost. Two touch the live decision path
while a loop can be running; one requires a browser capture to settle; one is a display
distinction rather than a correctness defect.

The argument for listing them is the chapter's own standard. GB-57 §57.10 is about mechanisms
that report on a system and are not outside the class of failures they report on. **A chapter
describing an architecture, which omitted the behaviours that architecture currently has, would
be the same shape.**

