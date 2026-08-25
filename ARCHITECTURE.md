# Architecture

How GlassBox Trader is put together, and why.

This document explains **structure and reasoning**. It does not restate the specification.
Where a number, schema field or task ID is authoritative, this file points at
[`docs/GLASSBOX_PROJECT_SPEC.md`](docs/GLASSBOX_PROJECT_SPEC.md) rather than copying it —
two copies of a contract drift, and the drift is always discovered by the copy that was
wrong. If this file and the spec disagree, **the spec is right and this file is a bug**.

---

## 1. The layer stack

Eight layers plus one cross-cutting harness. **Data flows strictly upward. No layer may
import from a layer above it.** Spec §3.1 is authoritative; this is the shape:

```
┌──────────────────────────────────────────────────────────────┐
│  L7  OBSERVABILITY      dashboard · decision log · replay    │
├──────────────────────────────────────────────────────────────┤
│  L6  EXPLAINABILITY     channel attribution · spectral       │
├──────────────────────────────────────────────────────────────┤
│  L5  EXECUTION          Alpaca orders · auto / co-pilot      │
├──────────────────────────────────────────────────────────────┤
│  L4  DECISION           signal · ranking · risk              │
├──────────────────────────────────────────────────────────────┤
│  L3  MODEL              Forecaster protocol                  │
│                         Persistence · DLinear · FITS         │
├──────────────────────────────────────────────────────────────┤
│  L2  FEATURE            indicators · wavelets · builder      │
├──────────────────────────────────────────────────────────────┤
│  L1  DATA               historical (yfinance) · live (Alpaca)│
├──────────────────────────────────────────────────────────────┤
│  L0  CONFIG & CONTRACTS settings.yaml · typed schemas        │
└──────────────────────────────────────────────────────────────┘

        ╔══════════════════════════════════════════════════╗
        ║  X  VALIDATION HARNESS  (cross-cutting, L1–L4)   ║
        ║     walk-forward · backtester · metrics · study  ║
        ╚══════════════════════════════════════════════════╝
```

### Responsibilities

| Layer | Owns | Must not |
|---|---|---|
| **L0 Config** | `settings.yaml`, typed dataclasses, validation on load | Contain logic |
| **L1 Data** | Fetching, caching, quality checks, UTC normalisation | Compute features |
| **L2 Feature** | Indicators, wavelets, **window assembly** | Know which model consumes it |
| **L3 Model** | Fit, predict, expose weights | Know about orders, money, or brokers |
| **L4 Decision** | Forecast → signal → rank → size → risk-check | Talk to a broker |
| **L5 Execution** | Broker API, order lifecycle, position reconciliation | Make decisions |
| **L6 Explain** | Turn weights and inputs into exact attributions and prose | Alter a decision |
| **L7 Observability** | Render, log, replay | Compute anything a decision depends on |

### This is enforced, not documented

`pyproject.toml` declares two import-linter contracts, checked in CI by `lint-imports`:

1. **Layers** — the stack above, in order. An upward import fails the build.
2. **The live path never imports the validation harness** — `backtest/` and
   `experiments/` may import downward freely, but nothing in the live path may import
   them. A backtest utility reachable from `live_loop.py` is how a fold boundary ends up
   inside a live decision.

Both contracts were verified with deliberate violations at `GB-3` and reverted. A contract
nobody has watched fail is a contract nobody knows works.

### Where the harness lives

`backtest/` and `experiments/` are *inside* the package but *above* the live path in the
contract. They are development-time tools that consume L1–L4 and are consumed by nothing.
`tests/causality.py` is different: it is test infrastructure, so it lives in `tests/` and
the live path cannot reach it at all. The trigger to promote it into the package is
written down in `DECISIONS.md` — a non-test consumer — and has not fired.

---

## 2. The frozen contracts

Spec §4 defines them; `glassbox/contracts/` implements them. **They do not change without
an entry in [`DECISIONS.md`](DECISIONS.md).** Every one is a frozen dataclass that
validates its own shape on construction, so a malformed object fails at the boundary that
produced it rather than three layers downstream.

| Schema | Crosses | Carries |
|---|---|---|
| `WindowBatch` | L2 → L3 | `X (B, L, C)`, `y (B, H)`, channels, timestamps, symbol, **source** |
| `ChannelStats` | L2 → L3, and into checkpoints | Per-channel mean and std, **and the date range they were fitted on** |
| `FitProvenance` | L3 → checkpoints | What a model was fitted on: channels, symbols, source, range, window count |
| `Forecast` | L3 → L4 | The predicted log-return path for one symbol |
| `Attribution` | L3 → L6 | Per-channel contributions, optionally per-frequency and gain/phase |
| `Signal` | L4 → L5 | Action, trend strength, whether the threshold was passed |
| `DecisionRecord` | L4 → L7 | The whole decision, replayable, tied to a config hash |

`Forecaster` (spec §4.3) is the protocol every model implements — `fit`, `predict`,
`explain`, `save`, `load`, plus `fitted`. **Nothing downstream knows which model it is
holding.** That is what makes three models and two feature sets a config switch rather
than three codebases, and it is why the comparative study is a table rather than an
argument.

### Three fields that exist to make a question answerable

Each was added because "is this leaking?" was otherwise a matter of trust:

- **`WindowBatch.source`** and the `source` column on every bar frame. A frame spliced
  from two vendors puts a step change inside every normalising window that straddles the
  join. Measured at `GB-8`: Alpaca reports 4–9% more volume than yfinance for the same
  bars, which would fabricate a `vol_z` spike out of nothing. `build_feature_frame`
  refuses a frame whose provenance is missing or mixed.
- **`ChannelStats.fitted_start` / `fitted_end`.** Normalisation statistics are the single
  most likely leakage path in this system. Recording the rows they came from turns "were
  these fitted on training data only?" into a question the `GB-25` audit can answer by
  intersecting two ranges.
- **`FitProvenance`.** The same argument one layer up, for the model itself. Every
  forecaster records it, including a `fit` that trains nothing — an exemption for the
  baseline would be an exemption in exactly the arm every other arm is compared against.

---

## 3. The keystone

**`glassbox/features/builder.py` is the only place in the codebase where a model input
window is assembled.** Offline training and the live loop call the same function with the
same config.

### Why one path and not two

The alternative — a training pipeline and a live pipeline that agree by convention — fails
in the worst possible way. The two drift by a warm-up bar, an off-by-one in a rolling
window, or a normalisation applied in one and not the other; the model still produces
plausible numbers; backtest results stay good; live results quietly are not. Nothing in
the system reports an error, because from every other module's point of view nothing is
wrong. A single path makes that class of bug unrepresentable rather than merely tested
for. `GB-27` asserts byte-identical windows from both paths, and it should have nothing to
catch.

The same reasoning applies inside the module. `build_windows(..., as_of=t)` returns exactly
one window and `as_of=None` returns every valid one, but they are **not two code paths**:
`as_of` filters the same list of end positions the batch path uses, and the same arithmetic
runs on both. Verified on real data — the `as_of` output is byte-identical
(`np.array_equal`) to the corresponding batch row.

### Counting the windows of a split

`train.py` calls `build_windows` **once over the whole frame** and splits the result by
timestamp afterwards. That is a **leakage-prevention decision, not a convenience**: there
is no second `build_windows` call that could be handed a second `ChannelStats`, so
validation cannot be normalised by its own mean and variance even by accident.

It has a consequence that is invisible from outside the module and produces a wrong number
rather than an error. **Slicing the frame to a split first and windowing it afterwards
drops the last `input_len + horizon - 1` rows**, because those windows cannot find their
target inside the slab. Measured on fold 1 of the configured grid, 23 Aug 2026:

| how the count was taken | windows | equations at `H = 4` |
|---|---|---|
| **from the batch `train.py` builds** — the authority | **2,505** | **10,020** |
| by slicing the frame to `fold.train` first | 1,890 | 7,560 |

A **25% undercount**, and the wrong figure looks entirely plausible. It reached a
determinacy calculation in GB-50 before a second measurement caught it. **Take window
counts from the batch, never by reconstructing them.** For a training split the number is
`len(fold.train) * len(universe)`, and `FitProvenance` records it on the checkpoint for
exactly this reason.

### What the module refuses to do

- **It does not fit normalisation statistics.** `fit_stats` is a separate function the
  caller applies to a training split, and the result is passed *in*. Fitting inside window
  assembly would fold the test period's mean and variance into the training input, which
  is the likeliest way leakage would enter this system and would be invisible in every
  number downstream.
- **It does not accept mixed provenance.** See `WindowBatch.source` above.
- **It does not silently degrade.** A channel set naming an unimplemented channel raises,
  naming the channel and the task that will add it. A study arm labelled `C2` that had
  quietly assembled `C0` would be reported as a wavelet result.

### `min_history_bars`, and the 445-bar floor

A live caller must supply `min_history_bars(cfg)` bars — **445** for the default channel
set, not the window length of 120. (**It read 352 until GB-27**; the correction and why it
matters are below the table.) The gap is warm-up: a recursive indicator's value at a
timestamp depends on how much history preceded it, so a window built from a short tail
differs from the window training computed at the same timestamp. The difference is small,
permanent, and invisible.

Measured against the full 2668-bar AAPL history:

| Tail bars | Byte-identical | Max abs difference |
|---|---|---|
| 197 | no | 4.261e-03 |
| 250 | no | 7.713e-05 |
| 300 | no | 5.960e-07 |
| **352** | **yes, for this symbol and timestamp** | **0.000e+00** |

197 — the window length plus the point where RSI stops *visibly* distorting — leaves a
permanent train/live gap of roughly 0.06 RSI points.

**That table is one symbol at one timestamp, and GB-27 showed it certified a floor that
does not hold.** Swept over five symbols and twenty-five timestamps, 352 bars gave
byte-identical windows in **32 of 125 pairs**. The derivation bounded the seed's *weight*
below 1e-7, which is only enough if the seed *difference* is at most 1 — it is a difference
of average gains in price units, and near RSI 50 the measured residual (3.815e-06) is the
same order as a float32 ulp (5.95e-06). The target is now **1e-10**, giving
`(13/14)^311 = 311` bars of decay, a warm-up of 325 and a floor of **445**, which sweeps
125 of 125. The intermediate 1e-9 target (414 bars) reaches 120 of 125 — measured, so the
extra margin is justified rather than assumed.

The number is **derived, not written down**: each channel declares its own warm-up, and
`min_history_bars` is `input_len + max(declared)`. `GB-47`'s wavelets declare 64 and the
number updates itself.

---

## 4. Causality

> Every value computed at time `t` uses only bars `≤ t`.

This is the project's most important correctness property, and the field it works in is
one where violating it makes results better rather than worse — which is why it is
enforced by a harness rather than by care.

`tests/causality.py` provides `assert_causal(fn, frame)`: compute `fn(frame)`, perturb
every row after a split point, recompute, and require every value at or before the split
to be bit-for-bit unchanged. Three split points, and **two perturbation modes**, because
either alone has a blind spot:

- **`scale`** multiplies future rows by a constant. Blind to anything scale-invariant —
  `sign(1.5·x) == sign(x)` exactly, so a function leaking *tomorrow's direction* passes it
  with every value identical. Direction accuracy is one of the two headline metrics.
- **`shuffle`** permutes future rows. Blind to anything permutation-invariant — a leaked
  full-sample mean or variance survives it untouched.

Both are demonstrated, not argued: the suite contains one leak per mode, each passing
under the other mode. The harness also refuses to pass vacuously — an inert perturbation,
an empty comparison prefix, or an all-NaN prefix each raise rather than reporting green.

`assert_fit_isolated` covers the other leakage surface. A set of statistics is one object
for a whole range, so there is no per-timestamp value to compare; the fitter is handed the
**whole** frame plus the boundary, exactly as a walk-forward caller holds it, and a fitter
that forgets to slice is caught rather than assumed away.

### What causality means for a model

A model consumes an already-built window, and every lag inside it is at or before that
window's own timestamp by construction. So for a model, "the future" is later **windows**,
not later lags — and the leak that remains is a statistic computed across a time-ordered
batch. Instance normalisation implemented as *batch* normalisation is that bug. A
single-window check cannot see it, because a batch of one centres to zero. The forecaster
contract test asserts it with the same `assert_causal`.

---

## 5. Explainability

Every model in the system is **linear**, and that is a design constraint rather than a
simplification. A linear model's forecast decomposes exactly:

```
sum(attribution.per_channel.values()) == forecast   (within 1e-5, asserted)
```

For FITS the same algebra runs in the frequency domain, giving per-frequency contributions
and a gain/phase pair per retained cycle — spec §6.5.

**SHAP, LIME, captum and every other perturbation-based attribution method are banned from
this codebase.** They approximate a decomposition that here is available in closed form,
and an approximation cannot be asserted. A glass box whose explanation is itself a model
is not a glass box.

---

## 6. Validation

Spec §7 is authoritative. The three rules:

1. **Zero leakage** — enforced by the harness in §4, not by review.
2. **Walk-forward only.** Train 24 months → validate 3 → test 3 → roll 3. Cross-validation
   is invalid for time series and is never used.
3. **Realistic costs.** Fees and slippage in every backtest.

And the reporting rule, which is not negotiable: **no result is reported as an absolute
number.** Every result is a delta against the persistence baseline, on direction accuracy
and Sharpe. MSE on prices is banned as a headline metric — on daily equity data a random
walk wins it. A backtest Sharpe above 2.0 is treated as a **leak alarm**, not a success.

This is why `PersistenceForecaster` predicts exactly zero and is not a placeholder:
`ln(C_t / C_t) == 0`, so a zero log-return path is the exact statement "the price does not
move". It is the null hypothesis every other arm is measured against.

---

## 7. Configuration

One file — `glassbox/config/settings.yaml` — parsed by `glassbox/config/loader.py` into frozen
dataclasses that validate every field and raise naming the exact path that was wrong.
**No module contains a magic number.** Two consequences worth stating:

- Swapping model or channel set is one line of YAML. The comparative study's grid is a
  config sweep, not a set of branches.
- `config_hash()` produces a stable SHA-256 of the resolved config, and every
  `DecisionRecord` carries it. A decision can be tied to the exact configuration that
  produced it, which is what makes replay meaningful.

Credentials never appear in config. They load from `.env` through `alpaca_credentials()`,
and `require_paper_endpoint()` refuses any endpoint other than the paper one — before
opening a connection, not after.
