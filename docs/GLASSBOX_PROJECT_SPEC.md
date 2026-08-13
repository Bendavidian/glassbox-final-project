# GlassBox Trader — Master Project Specification

**Version:** 3.0 · **Status:** Approved scope, in execution
**Window:** 15 Aug 2026 → 10 Oct 2026 (8 weeks, 4 sprints)
**Team:** Ben (brain / research / methodology, ~100% of tasks)
**Institution:** B.Sc. Computer Science, Bar-Ilan University — Final Year Project

---

> ## READ THIS FIRST — Instructions for Claude Code
>
> This file is the **single source of truth**. Before writing any code:
>
> 1. **Locate the current phase.** Run `git log --oneline -15` and read `PROGRESS.md`. Cross-reference against §9 (Task Register). Every task ID `GB-NN` maps to exactly one deliverable.
> 2. **Never violate a contract.** §4 defines frozen interfaces. If a task seems to require changing a contract, **stop and ask** — do not silently adapt.
> 3. **Never bypass a gate.** §8 defines three hard gates. Work belonging to a later phase does not start until the gate before it is green.
> 4. **Tests are the definition of done.** Every task in §9 lists its acceptance test. A task without a passing test is not complete.
> 5. **No feature invention.** If an idea is not in this document, it goes into `IDEAS_PARKED.md` — it does not go into the codebase.
> 6. **Config over constants.** No magic numbers in modules. Everything comes from `config/settings.yaml`.

---

## 1. Product Definition

### 1.1 What it is

GlassBox Trader is an autonomous algorithmic trading system that trades US equities on an **Alpaca paper-trading account** using real market data and virtual money, and — unlike conventional black-box bots — **explains every decision it makes, in real time, with mathematically exact attribution.**

The bot monitors a fixed universe of five liquid US stocks. For each, it runs a forecasting model that predicts the log-return trajectory for the next `H` trading days. It ranks the universe, applies risk limits, executes orders, and renders for every decision: the forecast path, the exact contribution of every input channel or frequency band, and the resulting action.

### 1.2 Definition of success

**This is a systems-engineering project, not a research paper and not a profit-seeking venture.**

Success is: the system runs end-to-end (connects → forecasts → decides → executes → explains → monitors), the methodology is leak-free and verifiable, and the comparative study is honestly reported.

Success is explicitly **not** beating the market. A result showing that no configuration beats a persistence baseline net of costs is a **valid, reportable, and expected** outcome.

### 1.3 The three pillars

| # | Pillar | What it means concretely |
|---|---|---|
| **P1** | **Glass-box brain** | Every model in the system is linear. Attribution is exact algebra (`Σ contributions == forecast`, asserted in tests), not a post-hoc approximation like SHAP. |
| **P2** | **Frequency-aware perception** | Market structure lives at multiple time-scales simultaneously. The project compares two ways of exposing that structure: **explicit** (wavelet channels fed as features) vs **implicit** (FITS operating in the frequency domain internally). |
| **P3** | **Honest validation** | Zero look-ahead, walk-forward only, realistic costs, and every result reported against a persistence baseline. |

### 1.4 The research question

> **Does frequency-domain structure in daily equity returns improve trading decisions — and is it better exposed explicitly as engineered features (causal DWT channels) or implicitly inside the model (FITS spectral interpolation)?**

Measured on **direction accuracy** and **Sharpe net of fees and slippage**, against a persistence baseline, across walk-forward folds.

---

## 2. Scope Boundaries

### 2.1 In scope

- Fixed universe: `AAPL, MSFT, NVDA, AMZN, GOOGL`
- Daily bars only
- Long-only positions
- Three forecasters: `Persistence` (baseline), `DLinear` (established baseline), `FITS` (new core)
- Two feature configurations: `C0` (price + technical indicators), `C2` (C0 + causal wavelet approximations)
- Live loop against Alpaca paper, Co-Pilot mode, Replay mode
- Streamlit dashboard with live explanations
- Walk-forward comparative study with statistical testing

### 2.2 Explicitly out of scope — declared, not forgotten

| Item | Reason |
|---|---|
| Sentiment / FinBERT / news channels | No room in 8 weeks; adds a fragile external dependency |
| Intraday or minute bars | Data cost, microstructure effects, 5× complexity |
| Short selling, options, leverage | Risk-model complexity without pedagogical return |
| Regime Guard (FITS reconstruction head as OOD detector) | **Documented as future work.** Strong idea, ~1 week, no slack for it. See §11. |
| Adaptive per-stock cutoff frequency | Second-order optimisation |
| Reproducing the FITS paper benchmarks (ETT/Traffic/Weather) | Does not advance a trading product |
| Real money | Never |

### 2.3 The compression note

The original plan assumed 14 weeks. The actual window is 8. Consequences, applied throughout this document:

- The ablation grid shrinks from `4 configs × 2 models × 20+ folds` to `3 models × 2 configs + a COF sweep`, over a capped fold count.
- `NLinear` is optional (`GB-14`); `DLinear` alone satisfies the baseline requirement.
- Regime Guard moves from stretch goal to declared future work.
- Sprint 4 carries the entire report; nothing technical may spill into it.

---

## 3. Layered Architecture

### 3.1 The layer stack

The system is eight layers plus one cross-cutting harness. **Data flows strictly upward. No layer may import from a layer above it.** This is enforced by an import-linter test (`GB-3`).

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

### 3.2 Layer responsibilities

| Layer | Owns | Must not |
|---|---|---|
| **L0 Config** | `settings.yaml`, typed dataclasses, validation on load | Contain logic |
| **L1 Data** | Fetching, caching, quality checks, UTC normalisation | Compute features |
| **L2 Feature** | Indicators, wavelets, **window assembly** | Know which model consumes it |
| **L3 Model** | Fit, predict, expose weights | Know about orders, money, or brokers |
| **L4 Decision** | Forecast → signal → rank → size → risk-check | Talk to a broker |
| **L5 Execution** | Broker API, order lifecycle, position reconciliation | Make decisions |
| **L6 Explain** | Turn weights + inputs into exact attributions and prose | Alter a decision |
| **L7 Observability** | Render, log, replay | Compute anything |
| **X Harness** | Walk-forward splits, backtest, metrics, experiment grid | Touch the live broker |

### 3.3 The keystone: `features/builder.py`

**This is the single most important design decision in the system.**

`builder.py` is the **only** place in the codebase where a model input window is assembled. Both offline training and the live loop call the identical function with the identical config. This structurally guarantees the model sees the same thing in both worlds — the most common source of silent failure in live trading systems.

A dedicated parity test (`GB-27`) asserts that the training path and the live path produce byte-identical windows for the same timestamp.

### 3.4 Repository layout

```
glassbox/
├── config/
│   ├── settings.yaml          # single source of truth (§5)
│   └── loader.py              # typed load + validation, fails loud
├── contracts/
│   ├── schemas.py             # FeatureFrame, WindowBatch, Forecast,
│   │                          # Attribution, Signal, DecisionRecord
│   └── protocols.py           # Forecaster protocol (§4.3)
├── data/
│   ├── historical.py          # yfinance → parquet cache
│   ├── live.py                # Alpaca Data API → same schema
│   └── quality.py             # gaps, NaNs, splits, calendar checks
├── features/
│   ├── indicators.py          # rsi14, vol_z, mom10, ma_dist20
│   ├── wavelets.py            # ★ causal rolling DWT → wav_a1..a3
│   └── builder.py             # ★ THE KEYSTONE — window assembly
├── model/
│   ├── persistence.py         # baseline: forecast = 0 return
│   ├── ltsf.py                # DLinear (+ optional NLinear)
│   ├── fits.py                # ★ FITS core
│   ├── train.py               # training loop, seeds, checkpoints
│   └── predict.py             # inference
├── explain/
│   ├── channel.py             # exact per-channel / per-lag attribution
│   ├── spectral.py            # ★ per-frequency attribution, gain & phase
│   └── narrate.py             # attribution → Hebrew/English prose
├── engine/
│   ├── signal.py              # forecast → signal, calibrated thresholds
│   ├── rank.py                # cross-sectional top-picks selection
│   ├── risk.py                # sizing, exposure caps, SL/TP
│   └── executor.py            # Alpaca orders, auto / co-pilot
├── backtest/
│   ├── engine.py              # event-driven, fees, slippage, SL/TP
│   ├── walkforward.py         # fold generation
│   └── metrics.py             # forecast + trading metrics
├── experiments/
│   ├── study.py               # ★ the comparative grid runner
│   └── report.py              # results.csv → tables + plots
├── live_loop.py               # the autonomous cycle
├── replay.py                  # recorded-day playback
├── smoke_offline.py           # offline end-to-end smoke command, GB-24
└── dashboard/
    └── app.py                 # Streamlit
```

### 3.5 The live decision cycle

```
  every 60s during US market hours (16:30–23:00 Israel time)
       │
       ▼
  ① L1  pull latest daily bars for the universe (Alpaca)
       ▼
  ② L2  builder.py assembles the active-config window per stock
       ▼
  ③ L3  forecaster.predict(X) → H-day log-return path
       ▼
  ④ L4  signal.py: trend strength → {enter, hold, exit}
        rank.py:   select top-K across the universe
        risk.py:   size positions, enforce caps, attach SL/TP
       ▼
  ⑤ L6  explain: exact attribution + natural-language reason
       ▼
  ⑥ L5  executor: submit to Alpaca  ─or─  Co-Pilot: await approval
       ▼
  ⑦ L7  persist DecisionRecord; dashboard refreshes
```

---

## 4. Frozen Contracts

**These interfaces do not change without an explicit decision recorded in `DECISIONS.md`.** They are what makes three models and two feature sets a config switch rather than three codebases.

### 4.1 Data conventions

| Rule | Value |
|---|---|
| Bar frequency | Daily |
| Index | `pandas.DatetimeIndex`, tz-aware, **UTC** internally; display Asia/Jerusalem |
| Model input series | **Log returns**: `r_t = ln(C_t / C_{t-1})` |
| PnL math | Raw close prices |
| Column names | Stable identifiers, lowercase snake_case |
| Missing data | Never forward-filled across a gap > 1 bar without an explicit flag |
| **No look-ahead** | Every value at time `t` uses only bars `≤ t`. Hard rule. Tested. |

**Canonical channel names:**

```
close      rsi14      vol_z      mom10      ma_dist20
wav_a1     wav_a2     wav_a3     wav_d1     wav_d2     wav_d3
```

### 4.2 Core schemas (`contracts/schemas.py`)

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd

# ── L2 output ────────────────────────────────────────────────
# FeatureFrame: pd.DataFrame, DatetimeIndex (UTC), one column
# per canonical channel name. No NaNs after warm-up trim.

@dataclass(frozen=True)
class WindowBatch:
    """Output of builder.build_windows(). The universal model input."""
    X: np.ndarray          # (B, L, C) float32 — B windows, L lags, C channels
    y: np.ndarray          # (B, H)    float32 — target log-return path
    channels: tuple[str, ...]   # length C, ordered, matches X's last axis
    timestamps: pd.DatetimeIndex  # length B, the 't' of each window
    symbol: str

@dataclass(frozen=True)
class Forecast:
    path: np.ndarray       # (H,) float32 — predicted log returns
    symbol: str
    as_of: pd.Timestamp

@dataclass(frozen=True, kw_only=True)
class Attribution:
    """Exact decomposition. MUST satisfy:
       sum(per_channel.values()) ≈ forecast.path.sum()  (atol=1e-5)

       per_lag is optional. The per-lag heatmap (GB-31) is cut from scope;
       implementations return None. It is retained in the schema so the feature
       can be added later without a contract change."""
    per_channel: dict[str, float]        # channel → contribution to Σ path
    per_lag: np.ndarray | None = None    # (L, C) contribution heatmap — optional
    per_frequency: dict[float, float] | None  # FITS only: period(days) → contrib
    gain_phase: dict[float, tuple[float, float]] | None  # FITS only
    forecast_total: float

@dataclass(frozen=True)
class Signal:
    symbol: str
    action: str            # "enter_long" | "hold" | "exit"
    trend_strength: float
    up_points: int
    passed_threshold: bool

@dataclass(frozen=True)
class DecisionRecord:
    """The complete, replayable record of one decision."""
    as_of: pd.Timestamp
    symbol: str
    forecast: Forecast
    attribution: Attribution
    signal: Signal
    order: dict | None     # None in co-pilot-pending or hold
    narrative: str
    config_hash: str       # ties the record to the exact config that made it
```

**Note on `Attribution(kw_only=True)`.** A dataclass field carrying a default may not
precede fields without one, so making `per_lag` optional in place requires either
reordering the fields or constructing by keyword. `kw_only=True` (Python 3.10+) keeps
the declared field order intact and costs only that `Attribution` is always built with
named arguments — which is what a five-field record should do anyway. Recorded in
`DECISIONS.md`, 2026-08-14.

### 4.3 The `Forecaster` protocol (`contracts/protocols.py`)

**This is the contract that makes the whole comparison possible.** Every model implements exactly this. Nothing downstream knows which model it is holding.

```python
from typing import Protocol
import numpy as np

class Forecaster(Protocol):
    name: str
    input_len: int      # L
    horizon: int        # H

    def fit(self, batch: WindowBatch, val: WindowBatch | None = None) -> None:
        """Train. MUST use only `batch` (+ `val` for early stopping).
        MUST store any normalisation statistics internally."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(B, L, C) float32 → (B, H) float32 log-return paths."""
        ...

    def explain(self, x: np.ndarray, channels: tuple[str, ...]) -> Attribution:
        """(L, C) single window → exact Attribution.
        MUST satisfy the exactness assertion in Attribution's docstring."""
        ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "Forecaster": ...
```

**Implementations:**

| Class | Module | Notes |
|---|---|---|
| `PersistenceForecaster` | `model/persistence.py` | `predict` returns zeros. `explain` returns empty attribution. ~15 lines. **Every result is reported against this.** |
| `DLinearForecaster` | `model/ltsf.py` | Trend/remainder decomposition + one linear map per component per channel. Weights are the explanation. |
| `FITSForecaster` | `model/fits.py` | rFFT → LPF → complex linear → irFFT. §6. |

### 4.4 The contract test

`tests/test_forecaster_contract.py` is parameterised over **all** implementations and asserts:

1. Output shape is exactly `(B, H)` and `float32`
2. `predict` is deterministic given a fixed seed
3. `explain(x).forecast_total ≈ predict(x[None])[0].sum()` within `1e-5`
4. `Σ attribution.per_channel.values() ≈ forecast_total` within `1e-5`
5. `save` → `load` → `predict` reproduces identical output
6. Perturbing `x[t+1:]` does not change `predict(x[:t+1])`

**A new model is not integrated until it passes this test unchanged.**

---

## 5. Configuration Contract (`config/settings.yaml`)

```yaml
meta:
  version: 3
  seed: 1337

universe: [AAPL, MSFT, NVDA, AMZN, GOOGL]

data:
  start: "2016-01-01"
  cache_dir: "data_cache"
  calendar: "NYSE"

window:
  input_len: 120        # L — bars the model sees (FITS prefers long look-backs)
  horizon: 4            # H — days forecast ahead

wavelet:
  family: db4
  levels: 3
  rolling_window: 64    # trailing bars per causal decomposition
  mode: symmetric

fits:
  cutoff_period_days: 5 # keep frequencies with period >= this; the ONE hyperparam
  supervision: "B+F"    # "F" = forecast only, "B+F" = backcast + forecast
  individual_weights: false   # false = one shared model across the universe

channels:
  C0_base:    [close, rsi14, vol_z, mom10, ma_dist20]
  C2_hybrid:  [close, rsi14, vol_z, mom10, ma_dist20, wav_a1, wav_a2, wav_a3]
  active: C0_base       # updated after the study picks a winner

model:
  active: dlinear       # persistence | dlinear | fits
  epochs: 100
  patience: 10
  lr: 0.001
  batch_size: 64

signal:
  min_trend: null       # calibrated on validation — never hardcoded
  max_trend: null
  min_up_points: 3
  top_k: 2

risk:
  max_position_pct: 0.10
  max_gross_exposure: 0.50
  stop_loss_pct: 0.03
  take_profit_pct: 0.06

backtest:
  fee_bps: 1.0
  slippage_bps: 2.0

walkforward:
  train_months: 24
  val_months: 3
  test_months: 3
  step_months: 3
  max_folds: 16         # capped for the 8-week budget

live:
  poll_seconds: 60
  mode: co_pilot        # auto | co_pilot
  market_open_il: "16:30"
  market_close_il: "23:00"
```

---

## 6. The FITS Core — Implementation Specification

### 6.1 What FITS is

FITS (Xu, Zeng & Xu, ICLR 2024, arXiv:2307.03756) reframes forecasting as **interpolation in the complex frequency domain**. A single complex-valued linear layer with ~10k parameters learns, for each retained frequency, an **amplitude gain** and a **phase shift** — which is exactly what complex multiplication means.

### 6.2 The pipeline

```
x : (B, L)  log returns, one channel
  │
  ├─ RIN         subtract instance mean (and optionally divide by std); store it
  │
  ├─ rFFT        → (B, L//2 + 1) complex
  │
  ├─ LPF         keep the first `COF` bins; discard the rest
  │              COF = L // cutoff_period_days
  │
  ├─ ComplexLinear   (B, COF) → (B, ceil(η·COF))     η = (L + H) / L
  │                  ONE complex weight matrix. This is the entire model.
  │
  ├─ zero-pad    → (B, (L+H)//2 + 1)
  │
  ├─ irFFT       → (B, L+H)
  │
  ├─ ✱ SCALE     multiply by (L + H) / L        ← see §6.3
  │
  ├─ inverse RIN add the mean back
  │
  └─ split       [:L] = backcast (supervised if B+F)
                 [L:] = forecast  → (B, H)
```

### 6.3 ✱ THE AMPLITUDE TRAP — read before writing a line

`torch.fft.irfft` normalises by the output length. Extending from `L` to `L+H` therefore shrinks every amplitude by a factor of `L / (L+H)`.

**Symptom if missed:** forecasts look plausible in shape but are systematically flat. Direction accuracy degrades quietly. MSE may even look *better*, because a flatter forecast is closer to zero. This is a silent bug that can cost a full day.

**Fix:** multiply the irFFT output by `(L + H) / L`.

**Test (`GB-42`):** feed a pure sinusoid of known amplitude, assert the reconstructed backcast segment matches the input within `1e-4`. This test must exist before any training runs.

### 6.4 Multivariate handling

FITS in the paper shares weights across channels. Here, the forecast target is the `close` log-return channel only. Auxiliary channels (indicators, wavelets) are **not** fed to FITS in the primary configuration — the spectral pipeline is univariate by design.

This creates a **deliberately empty cell** in the study grid: `FITS × C2_hybrid`. Feeding pre-filtered wavelet bands into a model whose first act is to filter frequencies is redundant. **State this reasoning explicitly in the report.** Do not fill a cell to make a table look complete.

Weight sharing across the universe (`individual_weights: false`) is the default: one model trained on all five symbols gives 5× the samples for the same 10k parameters.

### 6.5 Spectral explainability (`explain/spectral.py`)

Because `forecast = irFFT(W · rFFT(x))` is linear, each retained input frequency contributes an exactly computable amount to each output point.

Deliverables:

| Output | Meaning |
|---|---|
| `per_frequency: {period_days → contribution}` | Which cycle drove this forecast |
| `gain_phase: {period_days → (gain, phase_shift_days)}` | How the model transforms that cycle |
| Frequency-response plot | `|W|` vs period — **what the bot learned to listen to** |
| Frequency × horizon-day heatmap | Which cycle drives which forecast day |

Target narrative output:

> "This forecast is dominated by the ~12-day cycle (34% of the signal), which the model amplifies 1.4× and shifts 2.1 days forward. Everything faster than 5 days was removed by the low-pass filter."

The frequency-response plot is the single strongest visual in the demo. There is no equivalent for a non-linear model.

---

## 7. Validation Methodology

### 7.1 Three absolute rules

1. **Zero information leakage.** Every feature value and every threshold uses only data available at that moment. Enforced by `tests/test_no_lookahead.py`, which perturbs future bars and asserts past values are unchanged.
2. **Walk-forward only.** Train 24 months → validate 3 (threshold calibration + early stopping) → test 3 → roll 3. Cross-validation is invalid for time series and is never used.
3. **Realistic costs.** `fee_bps` and `slippage_bps` applied in every backtest. Alpaca paper simulates realistic fills.

### 7.2 Metrics

| Family | Metrics |
|---|---|
| Forecast quality | MAE, RMSE, **direction accuracy of the H-day trend** ← the quantity the signal layer actually consumes |
| Trading quality | Total return, **Sharpe**, max drawdown, hit rate — all net of costs |
| Reference | **Persistence baseline, on every single table** |

### 7.3 The reporting rule — non-negotiable

> **No result is ever reported as an absolute number. Every result is reported as a delta against persistence, on direction accuracy and Sharpe.**

MSE on prices is **banned as a headline metric.** On daily equity data a random walk wins it, and reporting it invites the exact misreading the project exists to avoid.

If the study produces `Sharpe > 2.0` in backtest, treat it as a **leak alarm**, not a success. Stop and audit `builder.py` and the fold boundaries.

### 7.4 The comparative study grid (`experiments/study.py`)

**Axis 1 — Model:** `Persistence` · `DLinear` · `FITS`
**Axis 2 — Feature config:** `C0_base` · `C2_hybrid`
**Axis 3 — COF sweep (FITS only):** cutoff period ∈ {2, 5, 10, 20} days

| | C0_base | C2_hybrid |
|---|---|---|
| Persistence | reference | reference |
| DLinear | ✓ | ✓ ← *does explicit wavelet decomposition help?* |
| FITS | ✓ ← *does implicit spectral filtering help?* | ✗ *deliberately empty, §6.4* |

**Protocol:** walk-forward, up to 16 folds, 5 symbols, fixed seeds, data snapshot cached to parquet.
**Statistics:** mean ± std across folds; paired Wilcoxon signed-rank of each arm vs persistence on per-fold direction accuracy and Sharpe.
**Output:** `results.csv` + a generated report with the summary table and per-fold boxplots.
**Promotion:** the winning arm is written back to `channels.active` and `model.active`. One line in the config.

---

## 8. Phases and Gates

Four sprints, three gates. **A gate is a hard stop, not a checkpoint.**

| Sprint | Dates | Theme | Ends at |
|---|---|---|---|
| **S1** | 15 Aug – 28 Aug | Foundations: config, contracts, data, features | — |
| **S2** | 29 Aug – 11 Sep | Offline vertical slice: model → backtest → metrics | **GATE 1** |
| **S3** | 12 Sep – 25 Sep | Live end-to-end: loop, execution, explain, dashboard | **GATE 2** |
| **S4** | 26 Sep – 10 Oct | FITS, wavelets, the study, the report | **GATE 3** |

### GATE 1 — 11 Sep · "The offline slice is real"

- [ ] `python -m glassbox.smoke_offline` runs data → features → DLinear → backtest → metrics
- [ ] Persistence baseline produces numbers on the same folds
- [ ] `test_no_lookahead.py` and `test_forecaster_contract.py` green
- [ ] At least one walk-forward fold completes end to end

**If red:** stop all new work. Sprint 3 does not begin. Fix or cut.

### GATE 2 — 25 Sep · "The system is a product"

- [ ] Live loop runs a full session against Alpaca paper without manual intervention
- [ ] At least one order placed, filled, and reconciled
- [ ] Every decision carries an exact attribution visible in the dashboard
- [ ] Replay mode reproduces a recorded day offline
- [ ] Co-Pilot mode approval flow works

**If red:** **FITS is cancelled, not postponed.** Sprint 4 becomes hardening + report. A working v1 is submitted. This is an acceptable outcome and is stated as such in the proposal.

### GATE 3 — 10 Oct · "Submitted"

- [ ] Study runs from one command and produces `results.csv`
- [ ] Every table reports deltas vs persistence
- [ ] Technical report complete
- [ ] Live demo (or Replay) rehearsed end to end
- [ ] Repository tagged, reproducible from a clean clone

**Sprint 4 rule:** anything not working by **3 Oct** does not enter the report as a result. It is written up as declared future work. No exceptions — the last week is for writing.

---

## 9. Task Register

**Legend:** Owner `B` = Ben · `N` = Noy. SP = story points (1 ≈ half a day).
Full importable version: `glassbox_jira_tasks.csv` (Jira CSV import, 60 issues + 12 epics).

**Distribution:** Ben 42 tasks (70%) · Noy 18 tasks (30%).

**The `Own` column is a formal academic record only.** Execution follows
`SOLO_BUILD_PLAN.md`: one developer writes every line. Ignore ownership when planning work.

### Cut from scope

Three tasks below are **cut**, per `SOLO_BUILD_PLAN.md` §2. Their rows are struck through
rather than deleted, so a reader sees the decision instead of a gap.

| ID | Reason for the cut |
|---|---|
| GB-14 | `NLinear` — `DLinear` alone satisfies the baseline requirement (2 SP saved) |
| GB-31 | Per-lag attribution heatmap — per-channel attribution already proves exactness; the heatmap is presentation polish (3 SP saved). Consequence: `Attribution.per_lag` is optional, §4.2. |
| GB-43 | Backcast/forecast supervision toggle — `B+F` is hardcoded per §5; the ablation is interesting, not essential (3 SP saved) |

### Epics

| Key | Epic | Layer |
|---|---|---|
| GB-E1 | Project Foundation & Governance | L0 |
| GB-E2 | Data Layer | L1 |
| GB-E3 | Feature Layer | L2 |
| GB-E4 | Model Layer & Contract | L3 |
| GB-E5 | Validation Harness | X |
| GB-E6 | Decision Engine | L4 |
| GB-E7 | Execution & Live Loop | L5 |
| GB-E8 | Explainability Layer | L6 |
| GB-E9 | Observability & Dashboard | L7 |
| GB-E10 | FITS Spectral Core | L3 |
| GB-E11 | Wavelet Perception Channels | L2 |
| GB-E12 | Comparative Study & Report | X |

### Sprint 1 — Foundations · 15–28 Aug

| ID | Task | Epic | Own | SP | Done when |
|---|---|---|---|---|---|
| GB-1 | Repo scaffold, package layout, ruff + black + pytest, GitHub Actions CI | E1 | B | 2 | CI green on empty test suite |
| GB-2 | `settings.yaml` + typed loader with fail-loud validation | E1 | B | 2 | Invalid config raises with a named field |
| GB-3 | `contracts/schemas.py` + `protocols.py` + import-linter layer test | E1 | B | 3 | Upward import raises in CI |
| GB-4 | Historical loader (yfinance) with parquet cache | E2 | B | 3 | 2016→today for 5 symbols cached, second call hits cache |
| GB-5 | Data quality module: gaps, NaNs, splits, NYSE calendar alignment | E2 | N | 3 | Report per symbol; known 2020 gaps flagged |
| GB-6 | Alpaca paper account, `.env` secrets handling, connectivity smoke test | E2 | N | 2 | Account equity printed from CLI; no secret in git |
| GB-7 | `data/live.py` — Alpaca Data API, identical schema to historical | E2 | N | 3 | Same-day bar matches yfinance within tolerance |
| GB-8 | `indicators.py` — rsi14, vol_z, mom10, ma_dist20 | E3 | B | 3 | Values match a hand-computed fixture |
| GB-9 | `builder.py` — window assembly from channel config → `WindowBatch` | E3 | B | 5 | Shapes correct for C0 and C2; warm-up trimmed |
| GB-10 | `test_no_lookahead.py` — perturb-future causality harness | E3 | B | 3 | Perturbing `t+1` leaves all values at `t` unchanged |
| GB-11 | `PersistenceForecaster` + the `Forecaster` contract test | E4 | B | 3 | Contract test passes for persistence |
| GB-12 | `README.md`, `ARCHITECTURE.md`, `PROGRESS.md`, `DECISIONS.md` | E1 | N | 2 | A newcomer can run the project from README alone |

### Sprint 2 — Offline Vertical Slice · 29 Aug – 11 Sep → **GATE 1**

| ID | Task | Epic | Own | SP | Done when |
|---|---|---|---|---|---|
| GB-13 | `DLinearForecaster` — trend/remainder decomposition + linear maps | E4 | B | 5 | Contract test passes; trains on one fold |
| ~~GB-14~~ | ~~`NLinearForecaster`~~ · **CUT** — `DLinear` alone satisfies the baseline requirement | E4 | B | ~~2~~ | — |
| GB-15 | `train.py` — loop, seeds, early stopping, checkpoints, scaler stats | E4 | B | 5 | Two runs with same seed give identical weights |
| GB-16 | `predict.py` — batch + single-window inference | E4 | B | 2 | Matches `train.py` outputs on held-out data |
| GB-17 | `walkforward.py` — fold generator with strict boundaries | E5 | B | 3 | No timestamp appears in two folds' train and test |
| GB-18 | `backtest/engine.py` — event-driven, fees, slippage, SL/TP | E5 | B | 8 | Hand-checked 3-trade scenario matches by hand |
| GB-19 | `metrics.py` — MAE, RMSE, direction accuracy, return, Sharpe, MDD, hit rate | E5 | B | 3 | Verified against a synthetic equity curve |
| GB-20 | `signal.py` — trend strength → signal, thresholds calibrated on validation | E6 | B | 5 | Thresholds differ per fold; none hardcoded |
| GB-21 | `risk.py` — sizing, gross exposure cap, SL/TP attachment | E6 | N | 3 | Property test: no config produces an over-limit position |
| GB-22 | `executor.py` — Alpaca order submission, manual smoke order | E7 | N | 5 | One paper order placed and confirmed |
| GB-23 | Order lifecycle + position reconciliation against Alpaca | E7 | N | 5 | Local state matches broker after a forced restart |
| GB-24 | `smoke_offline` — one command: data → features → model → backtest | E5 | B | 3 | Single command produces a metrics table |
| GB-25 | **GATE 1 review** — offline slice + persistence comparison written up | E1 | B | 2 | Checklist §8 fully green, recorded in PROGRESS.md |

### Sprint 3 — Live End-to-End · 12–25 Sep → **GATE 2**

| ID | Task | Epic | Own | SP | Done when |
|---|---|---|---|---|---|
| GB-26 | `live_loop.py` — scheduler, market-hours guard, 60s polling | E7 | N | 5 | Runs a full session unattended |
| GB-27 | **Train/live parity test** — identical windows from both paths | E3 | B | 3 | Byte-identical `X` for the same timestamp |
| GB-28 | `rank.py` — cross-sectional top-K selection | E6 | B | 2 | Deterministic ordering; ties broken stably |
| GB-29 | `DecisionRecord` persistence (JSONL) + config hashing | E6 | B | 3 | A record can be replayed into an identical decision |
| GB-30 | `explain/channel.py` — exact per-channel attribution | E8 | B | 5 | `Σ contributions == forecast` within 1e-5 |
| ~~GB-31~~ | ~~Per-lag attribution heatmap data~~ · **CUT** — presentation polish; per-channel attribution already proves exactness. `Attribution.per_lag` is optional (§4.2) | E8 | B | ~~3~~ | — |
| GB-32 | `narrate.py` — attribution → readable prose | E8 | B | 3 | Three sample decisions render correctly |
| GB-33 | Attribution exactness test across all models | E8 | B | 2 | Parameterised test green for every forecaster |
| GB-34 | Dashboard skeleton — layout, positions, PnL | E9 | N | 5 | Live positions visible and refreshing |
| GB-35 | Dashboard — forecast path chart per symbol | E9 | N | 3 | Path renders with history overlay |
| GB-36 | Dashboard — decision log + channel contribution bars | E9 | N | 5 | Every logged decision is inspectable |
| GB-37 | Co-Pilot mode — recommendation + approve/reject flow | E7 | N | 5 | Rejection leaves no order at the broker |
| GB-38 | Replay mode — recorded-day playback, fully offline | E9 | N | 5 | A recorded day replays with no network |
| GB-39 | Fault handling — retries, backoff, connectivity loss, restart safety | E7 | N | 3 | Kill -9 mid-cycle leaves no orphan state |
| GB-40 | **GATE 2 review** — live E2E with explanations, demo dry run | E1 | B | 2 | Checklist §8 fully green |

### Sprint 4 — FITS, Study, Report · 26 Sep – 10 Oct → **GATE 3**

| ID | Task | Epic | Own | SP | Done when |
|---|---|---|---|---|---|
| GB-41 | `fits.py` core — RIN, rFFT, LPF, complex linear, irFFT | E10 | B | 8 | Contract test passes |
| GB-42 | **Amplitude scale fix `(L+H)/L` + sinusoid reconstruction test** | E10 | B | 2 | Known sinusoid reconstructed within 1e-4 |
| ~~GB-43~~ | ~~Backcast + forecast supervision (`B+F`) toggle~~ · **CUT** — `B+F` hardcoded per §5; the ablation is not essential | E10 | B | ~~3~~ | — |
| GB-44 | FITS integration + shared-weights-across-universe training | E10 | B | 3 | One model serves all 5 symbols |
| GB-45 | `spectral.py` — per-frequency attribution, gain and phase extraction | E8 | B | 5 | Contributions sum to forecast within 1e-5 |
| GB-46 | Learned frequency-response visualisation (`\|W\|` vs period) | E8 | B | 3 | Plot generated from a trained model |
| GB-47 | `wavelets.py` — causal rolling DWT → `wav_a1..a3` | E11 | B | 5 | Emits values from bar 64 onward |
| GB-48 | Wavelet causality + additivity tests | E11 | B | 3 | `a3+d3+d2+d1 ≈ returns`; future perturbation inert |
| GB-49 | `study.py` — grid runner, seeds, `results.csv` | E12 | B | 5 | One command runs the full grid |
| GB-50 | COF sweep for FITS (2/5/10/20-day cutoff) | E12 | B | 3 | Four arms in results, one row each |
| GB-51 | Paired Wilcoxon vs persistence on direction accuracy and Sharpe | E12 | B | 3 | p-values in the results table |
| GB-52 | `report.py` — tables + per-fold boxplots from `results.csv` | E12 | B | 3 | Report regenerates from CSV alone |
| GB-53 | Dashboard — spectral explanation panel | E9 | N | 5 | Frequency bars + gain/phase visible live |
| GB-54 | Demo script + two full rehearsals | E12 | N | 3 | Runs end to end twice without intervention |
| GB-55 | Technical report — architecture and system design chapters | E12 | N | 5 | Draft reviewed by Ben |
| GB-56 | Technical report — methodology and leak-freedom chapters | E12 | B | 5 | Draft reviewed by Noy |
| GB-57 | Technical report — results, discussion, future work | E12 | B | 5 | Every table carries a persistence delta |
| GB-58 | Presentation deck | E12 | N | 3 | Rehearsed within time limit |
| GB-59 | Reproducibility audit: clean clone → full run | E1 | B | 3 | Fresh machine reproduces results |
| GB-60 | **GATE 3** — final tag, submission | E1 | B | 2 | Repo tagged `v1.0-submission` |

### Load balance

| Owner | Tasks | Story points | Share |
|---|---|---|---|
| Ben | 42 | 146 | 70% |
| Noy | 18 | 70 | 30% |

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **8 weeks is not enough** | High | High | Gates cut scope automatically. Gate 2 red ⇒ FITS cancelled, v1 submitted. |
| Forecasts too flat to trade | High | Medium | Signals use trend *direction*, not magnitude; thresholds calibrated on validation |
| Nothing beats persistence | **Expected** | Low | Declared in §1.2 as a valid outcome; the system and methodology are the deliverables |
| Look-ahead leak inflates results | Medium | **Critical** | Automated causality tests; `Sharpe > 2` treated as a leak alarm, not a win |
| FITS amplitude bug | High if unaware | Medium | §6.3 documents it; `GB-42` tests it before any training |
| Alpaca connectivity / market hours | Medium | Medium | 60s polling not websockets; Replay mode is fully offline |
| Two-person coordination gaps | Medium | Medium | Frozen contracts (§4) mean Noy never waits on Ben's model work |
| Scope creep | **High** | High | `IDEAS_PARKED.md`. No idea enters the codebase mid-sprint. |

---

## 11. Declared Future Work

Recorded here so that it is visibly a **choice**, not an omission.

**Regime Guard — FITS reconstruction head as an out-of-distribution detector.**
FITS uses the same architecture for anomaly detection: instead of forecasting forward, reconstruct the current window and treat reconstruction error as an anomaly score. Applied to trading: a spike in reconstruction error means the market's spectral structure has diverged from the training distribution — the model is outside what it knows. The bot would then reduce exposure and surface: *"Unusual market state — reconstruction error at the 97th percentile. New entries paused."* Estimated cost: one week. Excluded solely for schedule.

**Others:** adaptive per-symbol cutoff frequency; SWT/MODWT shift-invariant comparison; cross-sectional ranking objective trained directly; sentiment channel; intraday resolution.

---

## 12. Coding Conventions

- Python 3.11+, `ruff` + `black` defaults, type hints everywhere, `pytest`
- Every feature function is **pure**: DataFrame in → DataFrame out, no hidden state, no I/O
- No config values hardcoded in modules — always via `config/loader.py`
- Any change to feature or model code requires the full causality and contract suites to pass
- Small commits, imperative messages, one module per PR where possible
- Every task commit message begins with its Jira key: `GB-42: fix irFFT amplitude scaling`
- **When torn between clever and simple: simple.** This codebase is demonstrated live to a supervisor.

---

*GlassBox Trader v3 — the name is literal: a trading bot you can see into, now perceiving the market across the frequency spectrum.*
