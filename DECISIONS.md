# DECISIONS

Architectural decision records. One entry per decision that changes a contract,
adds a dependency, or cuts scope. Newest first.

Format: date · decision · reasoning · consequence.

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
(t = −0.65). *The exposure explains the returns and there is no alpha* is robust to the grid
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
