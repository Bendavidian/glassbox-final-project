# Sprint 2 Prompts — Offline Vertical Slice

**29 August – 11 September 2026 · 9 tasks · ends at GATE 1**

**Goal:** one command runs data → features → DLinear → backtest → metrics, with the
persistence baseline on the same folds.

Session-start and task-close prompts are in `SPRINT_1.md`. Use them here too.

---

## Days 1–3 · GB-13 · DLinear forecaster · 5 SP

> Plan mode. This is the first real model and it sets the pattern FITS will follow.

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-13.
Use Plan mode: describe your approach and wait for approval.

Implement model/ltsf.py with DLinearForecaster, following the LTSF-Linear design
(Zeng et al., AAAI 2023):

- Decompose each input channel into a moving-average trend component and a remainder
  (series minus trend), using a trailing moving average with the kernel size as a
  module constant
- One linear map from input_len to horizon per component, per channel
- The forecast is the sum of all component outputs
- Retain every weight as a named attribute keyed by channel, so explain/channel.py in
  GB-30 can read them directly without re-deriving anything

Implement the full Forecaster protocol from spec §4.3: fit, predict, explain, save,
load. For now explain may compute the per-channel decomposition directly here; GB-30
will move the shared logic into explain/channel.py.

Requirements:
- Registered in ALL_FORECASTERS so tests/test_forecaster_contract.py picks it up with
  no change to that file
- fit uses only the batch and the optional validation batch
- Normalisation statistics stored internally on fit and reused on predict
- Deterministic given a fixed seed

Acceptance: passes the full contract test unchanged; trains on one fold and produces
finite forecasts; weights accessible by channel name.

Report the parameter count.
```

---

## Days 4–5 · GB-15 · Training loop · 5 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-15.

Implement model/train.py:
- Batching over a WindowBatch, Adam optimiser, MSE loss
- Early stopping on the validation batch with cfg.model.patience
- Checkpoint saving: weights plus the normalisation statistics used, plus the config
  hash from GB-2, so a checkpoint is always traceable to the settings that produced it
- Full seeding of Python, NumPy and PyTorch from cfg.meta.seed
- Loss curves logged per epoch to a file for later use in the report

Hard requirement: normalisation statistics are computed on the TRAINING split only
and saved with the checkpoint. Never on validation, never on test. This is the second
most likely place for leakage to enter the system after the builder.

Write tests/test_train.py:
- Two runs with the same seed produce bit-identical weights
- A checkpoint round-trip reproduces predictions exactly
- Statistics saved in the checkpoint match those computed on the training split alone
- Early stopping triggers on a synthetic task where validation loss rises

Acceptance: all pass. Report how many epochs a single fold takes and the wall time.
```

---

## Day 6 · GB-16 · Inference · 2 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-16.

Implement model/predict.py supporting both batched inference over a WindowBatch and
single-window inference for the live loop. Loads a checkpoint together with its saved
normalisation statistics and its config hash, and refuses to run if the current config
hash differs — a mismatch means the model was trained under different settings.

Write tests asserting the batch and single-window paths produce identical results for
the same window, and that a config-hash mismatch raises.

Acceptance: paths agree; output shape and dtype match the contract.
```

---

## Day 7 · Review day

```
Review the model layer built so far. Tell me:
1. Anywhere normalisation statistics could touch validation or test data
2. Anywhere a config value is hardcoded
3. Any place where DLinear's implementation would make it hard to add FITS behind
   the same protocol in Sprint 4
4. Any test that would still pass if the implementation were wrong

Do not fix anything yet. Give me the list.
```

---

## Day 8 · GB-17 · Walk-forward fold generator · 3 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-17.

Implement backtest/walkforward.py generating folds per cfg.walkforward:
train_months, then val_months, then test_months, rolling forward by step_months,
capped at max_folds.

Each fold exposes explicit train, val and test DatetimeIndex ranges. Use the NYSE
trading calendar so month boundaries land on real trading days.

Write tests/test_walkforward.py:
- No timestamp appears in both the train and test range of any fold
- Within a fold: train.max() < val.min() < val.max() < test.min()
- Folds are strictly ordered in time
- Fold count respects max_folds
- A short series produces fewer folds rather than raising or silently overlapping

Acceptance: all pass. Report how many folds the real data produces from 2016 to today.
```

---

## Days 9–11 · GB-18 · Event-driven backtester · 8 SP

> The largest single task in the project. Plan mode. Budget three full days.

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-18.
Use Plan mode and wait for my approval before writing code.

Implement backtest/engine.py, an event-driven backtester:
- Iterate bars in strict chronological order. No vectorised shortcuts that could
  permit look-ahead — this is a correctness requirement, not a style preference.
- Accept signals per bar, size positions via engine/risk.py (which does not exist
  yet — define the interface you need and I will implement it in GB-21; for now use
  a simple injected sizing function)
- Apply cfg.backtest.fee_bps and slippage_bps on both entry and exit
- Honour stop-loss and take-profit. Document the intrabar ordering rule you chose:
  when both levels are breached in the same bar, which triggers first, and why.
- Emit an equity curve and a trade log (entry time, exit time, symbol, size, entry
  price, exit price, gross PnL, costs, net PnL, exit reason)

Write tests/test_backtest.py:
- A hand-constructed three-trade scenario reproduces manually computed PnL to the
  cent, including costs. Write the expected values as literals computed by hand.
- A trade that hits stop-loss exits at the stop level, not the close
- Costs are applied twice per round trip
- Feeding future-shifted prices does not change past equity values

Acceptance: the hand-checked scenario matches exactly. Report the intrabar rule you
chose and the total cost of one round trip in basis points.
```

---

## Day 12 · GB-19 · Metrics · 3 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-19.

Implement backtest/metrics.py:

Forecast metrics: MAE, RMSE, and direction_accuracy — the sign agreement between the
predicted H-day cumulative return and the realised one. Direction accuracy is the
quantity the signal layer actually consumes, so it is the primary forecast metric.

Trading metrics: total return, annualised Sharpe, maximum drawdown, hit rate — all
computed from the equity curve net of costs.

Every function takes an optional baseline argument and returns the delta against it.
Add a summarise(results, baseline) helper producing the table shape used in the report:
one row per arm, every value expressed as a delta against persistence.

Write tests/test_metrics.py verifying Sharpe and max drawdown against a synthetic
equity curve with known values, and direction accuracy against a hand-counted series.

Acceptance: verified against synthetic data. Per CLAUDE.md, MSE on prices must not be
exposed as a headline metric — if you include it at all, name it clearly as diagnostic
only.
```

---

## Day 13 · GB-20 · Signal engine with calibrated thresholds · 5 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-20.

Implement engine/signal.py:
- Compute trend strength across the forecast path (cumulative predicted return) and
  up_points (how many of the H steps are positive)
- Emit a Signal per spec §4.2: "enter_long", "hold" or "exit"
- The lower and upper trend thresholds are CALIBRATED PER FOLD on the validation
  split by maximising validation Sharpe. They are never hardcoded and never touch
  test data. cfg.signal.min_trend and max_trend are null in the config precisely
  because they are learned.

Implement calibrate_thresholds(val_forecasts, val_returns, cfg) -> tuple[float, float]
using a coarse grid search over candidate thresholds.

Write tests/test_signal.py:
- Calibrated thresholds differ across folds on real data
- No numeric threshold constant appears in the module source (assert by reading the
  file and checking for float literals outside the grid definition)
- Calibration given only validation data produces the same result whether or not
  test data is present in the frame
- A flat forecast produces "hold"

Acceptance: all pass. Report the thresholds calibrated on the first three folds.
```

---

## Day 14 · GB-24 + GB-25 · Smoke command and GATE 1 · 5 SP

**GB-24 first:**

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-24.

Wire a single command: python -m glassbox.smoke_offline

It must run the complete offline path with no manual steps and no network access:
load cached data → build features → generate folds → train DLinear on the first fold
→ calibrate thresholds on validation → forecast the test window → run the backtester
→ print a metrics table showing DLinear and the persistence baseline side by side,
with the delta.

Add --folds N to run more than one fold.

Acceptance: one command produces a metrics table in under five minutes on a laptop.
Report the table it produces on the first fold.
```

**Then GB-25 — the gate:**

```
Read CLAUDE.md, docs/GLASSBOX_PROJECT_SPEC.md §8 and SOLO_BUILD_PLAN.md §5.

Walk the GATE 1 checklist and give me an honest pass/fail on each item with evidence:
- python -m glassbox.smoke_offline runs data → features → DLinear → backtest → metrics
- Persistence baseline produces numbers on the same folds
- test_no_lookahead.py and test_forecaster_contract.py green
- At least one walk-forward fold completes end to end

Then, separately, audit for leakage:
- Confirm normalisation statistics are fitted on training data only
- Confirm signal thresholds are calibrated on validation data only
- Confirm no fold has overlapping train and test timestamps
- Report the backtest Sharpe. If it is above 2.0, treat it as a leakage alarm and
  find the cause before we proceed.

Record the outcome in PROGRESS.md under the gate log. If any item is red, list
exactly what must be fixed or cut before Sprint 3 begins. Do not soften the assessment.
```

---

## GATE 1 — 11 September

**If green:** Sprint 3 begins. The system now has a brain and a way to measure it.

**If red:** stop all new work. The likely culprits are GB-18 (backtester correctness)
and GB-20 (threshold calibration leaking). Fix or cut, record the decision in
`DECISIONS.md`, and only then move on.

> A backtest Sharpe above 2.0 on daily equities is not a success. It is a bug you
> have not found yet. Every hour spent finding it now saves the entire project from
> reporting a fiction.
