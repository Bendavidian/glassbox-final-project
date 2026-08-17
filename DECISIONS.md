# DECISIONS

Architectural decision records. One entry per decision that changes a contract,
adds a dependency, or cuts scope. Newest first.

Format: date · decision · reasoning · consequence.

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
than `MIN_SHARES` (0.001, Alpaca's minimum fractional quantity) the answer is **no trade**:
the broker would reject the order, so filling it in a backtest invents a trade that cannot
happen. `MIN_SHARES` is a module constant rather than config for the GB-8 reason — it is a
property of the venue, not a knob to tune.

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
