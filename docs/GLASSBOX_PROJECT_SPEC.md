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

**ANSWERED, 20 Aug 2026, with a control rather than only with a baseline** (GB-49, 16 folds, three fold-grid anchors):

> **These models extract nothing from this market that they do not also extract from white noise.** That is a statement about the models, not about the market: over the same period, a constant always-long rule beats chance by **5.6 points**, so the structure exists and is reachable — just not by explicit or implicit frequency decomposition at this horizon, on this universe, with these channels.

| on the same footing | direction accuracy |
|---|---|
| always-long, real data | **0.5564** |
| DLinear, real | 0.4959 |
| DLinear, white noise | 0.5006 |
| FITS, real | 0.5055 |
| FITS, white noise | 0.5025 |

**The models sit at their noise level. The market does not.**

The phrasing is fixed and the shorter version — *"there is no measurable difference between this market and noise"* — **must not survive anywhere in the report**. It drops the half that carries the meaning: a reader remembers the first clause, and the first clause alone says the market is noise, which the always-long bar disproves in the same table.

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

A dedicated parity test (`GB-27`) asserts that the training path and the live path produce byte-identical windows for the same timestamp — **swept across every symbol and at least twenty timestamps, never sampled at one point**. Sampling is what certified the earlier 352-bar floor that held in only 32 of 125 pairs.

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
│   ├── train.py               # one training run: stats, fit, checkpoint, curve
│   ├── history.py             # the per-epoch loss record (GB-15)
│   └── predict.py             # inference
├── explain/
│   ├── channel.py             # exact per-channel / per-lag attribution
│   ├── spectral.py            # ★ per-frequency attribution, gain & phase
│   └── narrate.py             # attribution → Hebrew/English prose
├── engine/
│   ├── signal.py              # forecast → signal, calibrated thresholds
│   ├── rank.py                # cross-sectional top-picks selection
│   ├── risk.py                # sizing, exposure caps, SL/TP
│   ├── executor.py            # Alpaca orders, auto / co-pilot
│   └── reconcile.py           # broker is truth; local state follows (GB-23)
├── backtest/
│   ├── engine.py              # event-driven, fees, slippage, SL/TP
│   ├── walkforward.py         # fold generation
│   ├── metrics.py             # forecast + trading metrics
│   └── calibrate.py           # per-fold threshold grid, scored by the
│                              # backtester itself (GB-20)
├── experiments/
│   ├── study.py               # ★ the comparative grid runner
│   ├── stats.py               # paired Wilcoxon, three references, Holm (GB-51)
│   └── report.py              # results.csv → tables + plots
├── faults.py                  # retry + backoff; the one policy for an unreachable
│                              # broker or feed (GB-39). Below `data` and `engine`
│                              # because both call it and neither can hold it
├── records.py                # DecisionRecord JSONL + the live trade log (GB-29)
├── live_loop.py               # the autonomous cycle
├── replay.py                  # recorded-day playback
├── smoke_offline.py           # offline end-to-end smoke command, GB-24
└── dashboard/
    └── app.py                 # Streamlit
```

**Outside the package.** `scripts/` holds operational entry points — `smoke_alpaca.py`
(GB-6) is the first. They import `glassbox`; nothing imports them. They are not packaged,
are not part of the layer contract, and never hold library code: anything a module would
want to reuse belongs in the package instead. They are deliberately absent from
`SPEC_MODULES` in `tests/test_scaffold.py`, which guards the importable tree above.

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
close_logret   rsi14      vol_z      mom10      ma_dist20
wav_a1         wav_a2     wav_a3     wav_d1     wav_d2     wav_d3
```

`close_logret` is the log-return series `r_t = ln(C_t / C_{t-1})`, not the price level.
The name is explicit because the explanation layer renders it: "62% from the
`close_logret` channel" is unambiguous where "62% from the `close` channel" would leave
a reader unable to tell price from return. The raw `close` column stays in the bar
frame, where the backtester takes it for PnL — that one really is the price.

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
    symbols: tuple[str, ...]      # length B, the symbol EACH WINDOW came from
    source: str            # provenance: the data source these windows were built from

    @property
    def unique_symbols(self) -> tuple[str, ...]: ...   # sorted, deduplicated

    @classmethod
    def concat(cls, batches) -> "WindowBatch": ...     # pool; validates before stacking

@dataclass(frozen=True)
class ChannelStats:
    """Normalisation statistics, fitted on a training split only (GB-9).
       Passed into build_windows; NEVER fitted inside it."""
    channels: tuple[str, ...]     # ordered, matches WindowBatch's channel axis
    mean: tuple[float, ...]       # aligned to `channels`
    std: tuple[float, ...]        # aligned to `channels`
    fitted_start: pd.Timestamp    # first row the statistics were fitted on
    fitted_end: pd.Timestamp      # last row — with fitted_start, identifies the split
    n_rows: int

@dataclass(frozen=True)
class FitProvenance:
    """What a forecaster was fitted on (GB-11). Every implementation records this at
       fit time, even a no-op fit, so GB-25 can audit any checkpoint the same way.
       Built by FitProvenance.from_batch(batch, input_len, horizon)."""
    channels: tuple[str, ...]     # the channel set and its order
    symbols: tuple[str, ...]      # tuple: GB-44 fits one model across the universe
    source: str                   # provenance of the underlying bars
    fitted_start: pd.Timestamp    # first window timestamp in the training batch
    fitted_end: pd.Timestamp      # last — with fitted_start, the training range
    n_windows: int
    input_len: int
    horizon: int

@dataclass(frozen=True)
class Trade:
    """One completed round trip. THE SAME TYPE in a backtest and in a live session
    (GB-29). GB-19's metrics must read a live trade log with no translation layer,
    and the live path may not import the harness - so the shared type lives here
    rather than in backtest/engine.py. The order IDs are the broker's and are None
    for every backtest trade: a simulated fill has no order to point at."""
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    size: float                  # shares
    entry_price: float           # reference price, before slippage
    exit_price: float
    gross_pnl: float
    costs: float                 # net_pnl == gross_pnl - costs, exactly
    net_pnl: float
    exit_reason: str             # stop | stop_gap | target | target_gap | signal | ...
    strategy_exit: bool          # False when the exit was administrative
    entry_order_id: str | None = None   # live only
    exit_order_id: str | None = None


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

**Note on `WindowBatch.symbols` (contract change, 2026-08-17).** It was `symbol: str`.
Every forecaster now trains **across the universe** — see §7.4 and the GB-15 row in §9 —
and a pooled batch genuinely has one symbol per window. The rejected alternatives: taking
a *sequence* of batches into the `Forecaster` protocol pushes pooling into every consumer,
and keeping a single string forces the pooled case to be encoded somewhere outside the
contract, which is the implicitness that promoting `ChannelStats` out of `build_windows`
removed. `FitProvenance.symbols` was already plural, so it absorbed the change unaltered.

`WindowBatch.concat(batches)` is the **only** place pooling happens. It refuses batches
that disagree on channel tuple (including order), window geometry, or `source` — pooling
across vendors is the same defect as splicing within one series (GB-8). A pooled batch's
`timestamps` are **not monotonic**: each symbol contributes the same date range. That is
why `FitProvenance.from_batch` takes `timestamps.min()` and `.max()` rather than the first
and last, and a test asserts the two differ.

**Normalisation is per symbol, never pooled (ruling, 2026-08-17).** Per-symbol scaling is
what *makes* pooling legitimate: it removes symbol-specific scale so the shared weights
learn the structure common across symbols. A pooled scaler would leave NVDA's inputs
systematically larger than MSFT's, and shared weights cannot express a symbol-specific
response — the model would be asked to fit a difference it has no parameters for. A
checkpoint therefore stores `{symbol: ChannelStats}`, and inference applies each window's
own symbol's statistics.

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
    fitted: FitProvenance | None   # None until fit() has been called

    def fit(self, batch: WindowBatch, val: WindowBatch | None = None) -> None:
        """Train. MUST use only `batch` (+ `val` for early stopping).
        MUST store any normalisation statistics internally.
        MUST set `self.fitted` to `FitProvenance.from_batch(batch, ...)`, even when
        training itself is a no-op — a checkpoint that cannot say what it was
        fitted on cannot be audited for leakage (GB-25)."""
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
6. Perturbing `x[t+1:]` does not change `predict(x[:t+1])` — for a model, "the future" is
   later **windows**, not later lags: every lag in a window is at or before that window's
   own timestamp by construction (GB-9), so the leak a model can still introduce is a
   statistic computed across the batch. Asserted with GB-10's `assert_causal`.
7. `fit` records `FitProvenance` truthfully, even when training is a no-op (GB-11)

**A new model is not integrated until it passes this test unchanged.** Adding one is a
single line appended to `FORECASTERS` in the test; nothing else in that file changes.

**Two dependencies between the properties, measured in GB-11.** Properties 5 and 6 both
work by recomputing and comparing exactly, so neither holds for a non-deterministic model
and neither can tell non-determinism apart from the fault it is looking for. Property 2
must pass before 5 and 6 mean anything.

**A note before FITS (GB-41).** Property 6 is what catches instance normalisation
implemented as *batch* normalisation. Because a batch is ordered in time, a batch
statistic pulls later windows into earlier predictions — and a single-window check cannot
see it, since a batch of one centres to zero.

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
  C0_base:    [close_logret, rsi14, vol_z, mom10, ma_dist20]
  C2_hybrid:  [close_logret, rsi14, vol_z, mom10, ma_dist20, wav_a1, wav_a2, wav_a3]
  active: C0_base       # updated after the study picks a winner

model:
  active: dlinear       # persistence | dlinear | fits
  epochs: 100
  patience: 10
  lr: 0.001
  batch_size: 64        # a positive int, or null for one batch of everything.
                        # Full batch was adopted on MAE and reverted by GB-20's re-check
                        # on Sharpe and direction, both 2026-08-17

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
  initial_cash: 100000.0  # matches the Alpaca paper account, so the two are comparable
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

FITS (Xu, Zeng & Xu, ICLR 2024, arXiv:2307.03756) reframes forecasting as **interpolation in the complex frequency domain**. A single complex-valued linear layer learns, for each retained frequency, an **amplitude gain** and a **phase shift** — which is exactly what complex multiplication means.

> **Parameter counts — read the numbers here, not the papers'.** The "~10k" quoted for FITS
> and the "~140k" quoted for DLinear are the source papers' figures for the source papers'
> configurations, and neither describes this project. At `input_len 120` / `horizon 4` this
> project's **DLinear holds 4,800 parameters, measured** (GB-13); its **FITS holds 1,200,
> measured** (GB-41) — `COF = 120 // 5 = 24` input bins mapped to `ceil(1.0333 × 24) = 25`
> output bins, `24 × 25 = 600` complex weights, **counted as 1,200 reals**. The prediction
> was exactly right; it is now a measurement.
>
> **Counted in reals, deliberately.** A complex weight is two learnable numbers, and
> reporting 600 against DLinear's 4,800 would flatter FITS by a factor of two on the very
> axis this box warns the report not to borrow a framing from.
>
> **1,200 is allocated; 1,150 is effective, and both must be reported** (measured 20 Aug
> 2026). RIN subtracts each window's own mean, and the rFFT's bin 0 *is* that mean — so
> after RIN it is zero on every window, and **row 0 of the complex weight matrix multiplies
> zero on every forward pass**. It takes no gradient, never leaves its initial value, and
> cannot change a forecast. Measured after a full fold's training: `max |w|` on row 0 is
> **2.1e-11** against **0.89** on the rest, the DC bin after RIN is **3.6e-15** against bin
> 1's **13.1**, and perturbing every weight in row 0 by `1+1j` moves the forecast by
> **exactly 0.0** while the same perturbation on row 1 moves it by **6.2**. That is
> `out_bins = 25` complex weights, **50 reals, 4.17% of the count**, allocated and dead.
>
> **The row stays.** §6.2's low-pass keeps "the first `COF` bins" and bin 0 is one of them,
> and the source paper's architecture carries the same dead row for the same reason —
> removing it is a change to a specified pipeline bought for a cosmetic 4%, and is not
> taken here. What is not acceptable is quoting the allocated number as capacity.
> `tests/model/test_fits.py::test_the_dc_row_is_allocated_and_cannot_learn` pins both.
>
> The ordering survives — FITS is still the smaller model — but **the framing does not.**
> The papers set up a heavyweight baseline against a tiny challenger, a 14× gap. Here the
> gap is 4×, and the baseline itself has **fewer parameters than the paper's FITS**: the
> DLinear in this project is smaller than the model the FITS paper holds up as remarkably
> small. "Tiny model beats large model" is not a claim this project's numbers support, and
> the report must not borrow it.
>
> The comparison is also **not like-for-like**: DLinear here consumes five channels and
> FITS is univariate by design (§6.4). Part of the 4× is breadth of input, not depth of
> model. GB-57 must report both measured counts and state both of these facts.

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

### ✱ B+F REQUIRES `X` AND `y` IN THE SAME UNIT — do not "simplify" this away

`loss = mse(forecast, y) + mse(backcast, x)` is a **sum of two terms with no weights**, so
it is only the objective §6.2 describes when the two are measured in the same unit. They
are not, unless the target is scaled: `build_windows` standardises the input channels, so
the backcast is supervised at unit variance, while raw daily log returns put the forecast
term at `0.015²`.

**Measured 20 Aug 2026 over 16 folds:** with `X` standardised and `y` raw, the backcast
term is **67× the forecast term** — the objective is **98.5% backcast**. The model spends
itself reconstructing its own input and emits a forecast **4.8× too large**, and its MAE is
6× DLinear's for that reason and no other. Lowering the learning rate changes **nothing**
(0 of 16 folds at `1e-3`, `1e-4` and `1e-5` alike): it is not a conditioning problem, it is
a units problem.

**The fix is in `build_windows`, not here**: the target is divided by the target channel's
own deviation — the same number the input column was divided by — which restores the ratio
to **0.69×** and the MAE to the raw arm's value to four decimals, as scale-equivariance
predicts for a linear model behind RIN. `features.builder.restore_targets` is the inverse,
applied wherever a forecast leaves the model layer.

**Divided, never centred.** Centring makes `predict` affine, so the forecast carries a
constant belonging to no channel and `Attribution.from_terms` refuses the decomposition —
its error message names this exact case. The two forms are indistinguishable on the fold
grid anyway (MAE 0.016061 against 0.016068 for DLinear; 0.015841 against 0.015802 for
FITS), so exactness decides it.

**This is a correctness defect when it is absent, not a tuning preference.** The loss falls,
every test passes and nothing complains — which is why the requirement is written into the
pipeline rather than left as a property of whoever last touched the builder.

### 6.3 ✱ THE AMPLITUDE TRAP — read before writing a line

`torch.fft.irfft` normalises by the output length. Extending from `L` to `L+H` therefore shrinks every amplitude by a factor of `L / (L+H)`.

**Symptom if missed — corrected 20 Aug 2026, and the correction matters more than the original.** Omitting the scale multiplies every sample by exactly `L/(L+H)`: a **uniform, positive** factor, 0.9677 at `L=120, H=4`.

- **MAE and MSE improve**, because a flatter forecast sits closer to zero. A reader watching either would conclude the model got better.
- **Cross-model comparison is corrupted.** FITS would beat DLinear on MAE by being flatter rather than by being better — on the one axis where the two are directly compared. **This is the real damage, and it is a comparison bug rather than a performance bug.**
- Its damage is **muted by §7.3 banning MAE as a headline**, which this project adopted for an unrelated reason.

> **An earlier version of this paragraph claimed "direction accuracy degrades quietly". That is wrong.** A uniform positive scaling cannot change a sign, so the cumulative forecast scales, its sign is identical, and direction accuracy is unchanged to the last decimal. The per-fold trend thresholds are calibrated on forecasts carrying the same factor, so the band adapts and the signals barely move either. The line is kept here rather than quietly replaced, because a document that reads better than its history is a document nobody can check.
>
> `tests/model/test_fits.py::test_the_amplitude_scale_changes_mae_and_cannot_change_direction` turns the correction into a measurement: it asserts identical direction and differing MAE on real windows. It is the check that would have caught the wrong line when it was written.

**Fix:** multiply the irFFT output by `(L + H) / L`.

**Test (`GB-42`):** feed a pure sinusoid of known amplitude, assert the reconstructed backcast segment matches the input within `1e-4`. This test must exist before any training runs.

**The test's grid is chosen, and the choice is part of the test.** Bin `k` of a length-`L` transform is the frequency `k/L`; the same physical frequency sits at bin `k·η` of a length-`L+H` transform. When `η·k` is not an integer the frequency cannot land on an output bin, and **no parameter-free extension can reconstruct the input** — that gap is exactly what the learned complex layer interpolates across. Measured on a 3.0-amplitude 12-day sinusoid over `L=120`: at the configured `H=4`, `η·k = 10.333` and the backcast error is **4.97**, the amplitude itself; at `H=12`, `η·k = 11` and the error is **2.2e-14**. GB-42 therefore runs at `H=12`. An amplitude test at `H=4` would measure interpolation error and attribute it to amplitude.

### 6.4 Multivariate handling

FITS in the paper shares weights across channels. Here, the forecast target is the `close_logret` channel only. Auxiliary channels (indicators, wavelets) are **not** fed to FITS in the primary configuration — the spectral pipeline is univariate by design.

This creates a **deliberately empty cell** in the study grid: `FITS × C2_hybrid`. Feeding pre-filtered wavelet bands into a model whose first act is to filter frequencies is redundant. **State this reasoning explicitly in the report.** Do not fill a cell to make a table look complete.

Weight sharing across the universe (`individual_weights: false`) is the default: one model trained on all five symbols gives 5× the samples for the same parameter count (**1,200 allocated / 1,150 effective, measured** here, not the paper's 10k — see §6.1).

**The key controls training from GB-44, and it controls every arm.** It was declared here and in §5 and read by the loader from GB-2 onward, and until GB-44 nothing consulted it — it could be set to either value and the system behaved identically, which reads as honoured and is not. `model/train.py` now consults it and the checkpoint manifest records which regime produced the weights, under `training.weight_sharing`.

Two consequences follow, and both come from decisions already taken rather than from GB-44:

- **`true` is a caller-side regime, because of a frozen contract.** `Forecaster.predict` (§4.3) takes `(B, L, C)` and **no symbol**, so one fitted model has nothing to route on and cannot hold five weight sets; giving it one would change the protocol every layer above depends on. What `train` already supports is the other half — `frames` may name a single symbol — so `individual_weights: true` means *call `train` once per symbol, and keep one checkpoint per symbol*. Handing it the whole universe under that setting is refused, naming the remedy.
- **It governs every arm, not only FITS, despite living in the `fits` section.** The section is where weight sharing is discussed, but the force behind the key is the 2026-08-17 comparability ruling, and that ruling is about arms being trained alike. A regime applied to FITS alone would train it per symbol while DLinear pooled — precisely the handicap the ruling exists to prevent, and the study would report that handicap as architecture.

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

**Delivered in GB-45/GB-46, and three things the design of it settled.**

- **The RIN mean is part of the forecast and belongs to no frequency.** RIN subtracts the window mean before the transform and adds it back after, so the mean reaches the forecast *without passing through the complex layer at all*. It is carried at `per_frequency[inf]`, and a decomposition that omitted it could not close — the exactness tolerance would catch it, and `Attribution.from_terms`'s error message names this case. **Measured on fold 16, it is the largest single contributor in 37.2% of test windows**, which is what a smoothed extrapolation of a drifting series should look like.
- **Bin 0 contributes identically zero, and is reported rather than dropped.** Its learned row is the dead one of §6.1: RIN has already removed the mean, so the rFFT's bin 0 *is* that mean and the weights in row 0 multiply nothing. An absent entry and a measured zero say different things, and a reader who finds no bin 0 wonders whether it was forgotten. So `per_frequency[inf]` is the mean's contribution and `gain_phase[inf]` is the dead row's gain — **two different facts under one key**, stated in the module rather than left to be inferred.
- **Exactness is not re-implemented.** The per-frequency contributions go through `Attribution.from_terms`, the one function allowed to sum a decomposition, and the `(H, L)` matrix behind each frequency is **measured from the model's own forward pass** with one bin unmasked — not re-derived from the pipeline description above. Their sum reproduces `forecast_matrix()` to **4.4e-16**.

**✱ WHAT THE FREQUENCY RESPONSE IS, AND WHAT IT IS NOT — measured 20 Aug 2026, and the first reading of it was wrong.**

The first measurement, on fold 16 alone, read: *gain 0.596 to 0.830, trough at 8.6 days, peak at 24 days, and a consistent **+1.9 to +2.1 day phase advance** across the 15–30 day band — the model advances those cycles, which is GB-41's momentum finding arriving from the other side.* **That sentence is withdrawn.** It is kept here, struck through in substance, because a document that reads better than its history is one nobody can check.

**The sweep did not catch it and could not have.** Across 16 folds × 3 fold-grid anchors — 48 independently trained models — the phase advance in that band is **+1.9582 days, sd 0.0203, positive in 48 of 48 cells**, and indistinguishable between anchors (+1.9569 / +1.9538 / +1.9640). That is *more* stable than the r = −0.47 correlation that vanished under a one-week shift, so stability across folds would have **confirmed** the claim. **Stability was the wrong test**, and this is the exactness tautology of §4.4 in a new costume: a check that a correct explanation and an incorrect one both pass is not evidence.

**The control that separates them is a model trained on data with no temporal structure.** Fitted on the same fold with the close channel replaced by white noise, and with the within-window ordering destroyed:

| | phase in the 15–30 day band | mean gain in that band |
|---|---|---|
| real returns | **+1.9640 days** | 0.8191 |
| within-window shuffle | **+1.9553 days** | 0.8286 |
| white noise | **+1.9309 days** | 0.8435 |
| untrained (zero weights) | +0.0000 | 0.0 |

**The phase advance survives the complete destruction of temporal structure, so it is not a fact about the market.** And it is not only the phase: the whole gain curve on white noise correlates with the curve on real returns at **+0.9485**, with the trough at **8.00 days in both** and a mean absolute difference of **0.0319** against a curve range of **0.2335** — so roughly **86% of the response's own variation is data-independent**.

**What the response actually shows, and it is the better finding.** Gain tracks **grid misalignment**. Bin `k` must be placed at bin `η·k` of the longer output grid, and how well it can be placed depends on how close `η·k` is to an integer. Over the 23 retained bins, averaged across all 48 models:

> **Pearson r = −0.9049, Spearman = −0.9427 (p = 1.8e-11)** between a bin's mean gain and `|η·k − round(η·k)|`. Best-aligned bin `k=1` (120 days, misalignment 0.033) has gain **0.8102**; worst-aligned `k=15` (**8.00 days**, misalignment **exactly 0.500**) has gain **0.5758**.

So the trough at 8 days is **the half-integer bin — the frequency that lands exactly between two output bins and therefore cannot be reconstructed**. It is §6.1's `η = 31/30` finding made visible in the trained weights, which is why the figure keeps its place in the demonstration and changes its meaning: **GB-46 shows the cost of interpolation, not a discovery about equity cycles.** The phase's sign even flips at that boundary — bins 1–14 average about +1.9 days and bins 15–23 about −1.0 — which no market story predicts and grid geometry does.

**What survives as data-dependent:** the ~14% of the curve that differs from the noise model, and the fact that the *peak* period is **not** stable (median 40 days, range 17.1–120 across the 48 models) where the *trough* is (median 8.00, range 8.00–9.23). The stable half is the geometry; the unstable half is the data. GB-57 must state it that way round.

The dominant contributor per window: **the RIN mean 37.2%**, then the **15-day cycle 22.4%**, 120-day 17.2%, 17.1-day 13.1%. **No contributor's mean share of the gross view exceeds 0.211**, which is the cancellation story of GB-30 again — 24 contributors, none dominant.

---

## 7. Validation Methodology

### 7.1 Three absolute rules

1. **Zero information leakage.** Every feature value and every threshold uses only data available at that moment. Enforced by `tests/test_no_lookahead.py`, which perturbs future bars and asserts past values are unchanged.
2. **Walk-forward only.** Train 24 months → validate 3 (threshold calibration + early stopping) → test 3 → roll 3. Cross-validation is invalid for time series and is never used.
   - **The splits are embargoed by `H` bars** (GB-17). A window ending on the last training bar carries the label `r[t+1] .. r[t+H]`, which lies inside validation, so disjoint ranges are not sufficient — the last `H` window-ends of each split are dropped. Derivation and the failing-without-it test are in `backtest/walkforward.py`.
   - Because folds **roll** rather than block, a timestamp in fold *i*'s test range does reappear in fold *i+2*'s **training** range. That is walk-forward working as intended — retraining on data that has since become available — and is not leakage: within any one fold, training is strictly before test.
3. **Realistic costs.** `fee_bps` and `slippage_bps` applied in every backtest. Alpaca paper simulates realistic fills.

### 7.2 Metrics

| Family | Metrics |
|---|---|
| Forecast quality | MAE, RMSE, **direction accuracy of the H-day trend** ← the quantity the signal layer actually consumes |
| Trading quality | Total return, **Sharpe**, max drawdown, hit rate — all net of costs |
| Reference | **Persistence baseline, on every single table** — except the direction column, below |

**Sharpe is computed from the equity curve's daily returns, not from per-trade returns**
(GB-19). They are different numbers. Daily returns include the cost of sitting flat, which
per-trade Sharpe ignores; and two arms with the same equity curve must score the same
whatever their trade counts, or the study compares trade frequency. Below **five strategy
trades** Sharpe is NaN rather than a number, and the summary table prints the trade count
beside it so a blank cell explains itself. Annualisation uses each fold's **own** bar count
and calendar span, never a constant 252.

**Direction accuracy is referenced to always-long, not to persistence** (GB-19, corrected
by GB-20). Persistence forecasts zero, `sign(0)` agrees with nothing, and a zero forecast
expresses no direction — so **persistence's direction accuracy is undefined, measured NaN in
16 folds of 16.** It remains the baseline for MAE, RMSE, return and Sharpe, which it
genuinely competes on.

The direction reference is the **always-long** strategy: it calls up on every window, so it
scores that fold's realised **up rate**. The class is fixed from the training split — the up
rate exceeds 0.5 in all 16 training splits, and §2.1 is long-only, so "always short" is not
in the action space — and only the rate is measured on test. Across the 16 folds the bar is
**0.5625**, not 0.5; DLinear scores **0.5071** under the configured `batch_size: 64` and
beats it in **5 folds of 16**. (The full-batch variant measured 0.4751 and 4 of 16; it was
reverted the same day and any quotation of it must say so.)

Two references were rejected, and both errors matter:
- **0.5** understates the bar by 6.25 points and flatters every arm.
- **The per-fold majority class**, `max(up_rate, 1 − up_rate)` — the first version of this
  rule — overstates it at **0.5865**, because it flips to "always short" on folds whose
  test period fell. That is a bar chosen with test-period knowledge and it is not
  achievable by any strategy, since no system knows in advance whether the coming quarter
  is up or down. It erred *against* the model, and it was still inadmissible.

The summary table carries a `direction_reference` column recording the number used; it is
by definition that fold's up rate, so the bar is visible moving fold to fold.

### 7.3 The reporting rule — non-negotiable

> **No result is ever reported as an absolute number. Every result is reported as a delta against a stated reference — and the reference depends on what the metric measures.**

**Three references** (ruled 2026-08-17, GB-21):

- **Forecast metrics (MAE, RMSE) are reported as deltas against persistence.**
- **Direction accuracy is reported against the always-long bar** (§7.2).
- **Trading metrics (return, Sharpe, max drawdown) are reported against BUY-AND-HOLD**,
  with persistence shown alongside as the do-nothing floor.

Persistence is the right reference for forecast skill and a degenerate one for trading,
because it never trades; comparing a trading strategy to cash measures only that it traded.
The report must state why three references are used and what each one answers — a reader
who sees three baselines and no explanation will assume they were chosen to flatter.

MSE on prices is **banned as a headline metric.** On daily equity data a random walk wins it, and reporting it invites the exact misreading the project exists to avoid.

**MAE stays as a reported column, and never appears alone** (ruled 20 Aug 2026). Dropping the standard metric of the field, in a project comparing itself to two papers that report it, reads as hiding a result rather than as methodological care. The stronger move is to report it with its pathology attached, and the pathology is now **measured on this project's own data rather than argued from the definition**:

> **MAE never appears in any table without `flatness` in the adjacent column**, where `flatness` = mean `|forecast|` ÷ mean `|actual|` and 1.0 is right-sized. Every table carrying MAE carries this line: **Spearman(MAE, flatness) = +0.81 (DLinear) and +0.67 (FITS); Spearman(MAE, direction) = +0.01 and −0.13; and the 16-fold MAE winner is persistence, which forecasts exactly nothing and has a flatness of 0.0.**

So a reader given MAE alone cannot tell an arm that got **better** from one that got **flatter**. That is what turns a misleading metric into a demonstrated finding, and it is the difference between banning a metric on an argument — which is what this section used to do — and banning it on a measurement. The pairing is **structural, not editorial**: `metrics.COMPANIONS` builds the summary's columns and `smoke_offline.FOLD_COLUMNS` is asserted against it, so printing MAE without flatness means deleting the pairing rather than forgetting it. GB-49 carries `flatness` in `results.csv` beside `mae` for the same reason.

If the study produces `Sharpe > 2.0` in backtest, treat it as a **leak alarm**, not a success. Stop and audit `builder.py` and the fold boundaries.

**The alarm applies to the AGGREGATE figure across folds, not to any single fold** (ruled
2026-08-17, after GB-20's re-check tripped it). A per-fold Sharpe on roughly 60 bars carries
a standard error near **2.0 annualised** — 0.129 on the daily ratio, times √252 — so
single-fold excursions past 2.0 are expected under a **true Sharpe of zero** and are not
evidence of leakage. What *is* evidence: an aggregate above 2.0, or a per-fold distribution
that is **asymmetric** — inflated on the upside without matching negative excursions. Record
both the aggregate and the per-fold spread whenever the rule is invoked. The worked example
of the rule being applied and correctly not firing is in `DECISIONS.md`, 2026-08-17.

### 7.4 The comparative study grid (`experiments/study.py`)

**Axis 1 — Model:** `Persistence` · `DLinear` · `FITS`
**Axis 2 — Feature config:** `C0_base` · `C2_hybrid`
**Axis 3 — COF sweep (FITS only):** cutoff period ∈ {2, 5, 10, 20} days. **It rides on the star as three more spokes rather than becoming a cross** (GB-50): the cutoff is FITS's one hyperparameter and reaches persistence and DLinear through nothing at all, so a persistence row at cutoff 20 would duplicate its row at cutoff 5 and invite a reader to average it. **Each spoke carries its own null control**, so a cutoff is falsifiable on its own terms rather than inheriting the centre's. `results.csv` carries `cutoff_period_days` with `cof` and `dead_row_fraction` immediately beside it, for the reason `flatness` sits beside `mae`: the cutoff alone is a number nobody can interpret.

| | C0_base | C2_hybrid |
|---|---|---|
| Persistence | reference | reference |
| DLinear | ✓ | ✓ ← *does explicit wavelet decomposition help?* |
| FITS | ✓ ← *does implicit spectral filtering help?* | ✗ *deliberately empty, §6.4* |

**Protocol:** walk-forward, up to 16 folds, 5 symbols, fixed seeds, data snapshot cached to parquet.
**Training regime — the same for every arm (ruled 2026-08-17).** One model is fitted across
the whole universe, with a **per-symbol** scaler. This is a correctness requirement for the
grid, not a tuning choice: FITS at 1,200 parameters is already overdetermined per symbol at
1.67×, while DLinear at 4,800 is underdetermined at 0.42×, so training the two under
different regimes would handicap one arm and the grid would report that handicap as
architecture. Any arm added later inherits the regime. See §4.2 and the GB-15 row in §9.
**Statistics:** mean ± std across folds; paired Wilcoxon signed-rank of each arm against **its own reference**, paired by fold. **CORRECTED 23 Aug 2026.** This line read *"of each arm vs persistence on per-fold direction accuracy and Sharpe"*, and that is the single-reference error §7.3 already fixed once: persistence forecasts zero and takes no trades, so its `direction` and `sharpe` cells in `results.csv` are **empty**, and two of the four metrics would have been tested against a NaN. The three-reference rule applies here as everywhere else — **MAE against persistence** on the same channel set, **direction against always-long** at that fold's realised up rate, **Sharpe and total return against buy-and-hold** at the same fold. **Multiple comparisons are counted, corrected and stated**: every row carries `n_tests` (the family is every test computed in the same call, so a report narrows the family by narrowing the table) and `p_holm` beside the raw `p`. Holm-Bonferroni rather than Bonferroni because it is uniformly more powerful, and rather than Benjamini-Hochberg because FWER is the guarantee a reader of a results table assumes and BH's needs a dependence assumption nobody has checked. **And the floor matters as much as the values:** the exact two-sided signed-rank test on 16 folds cannot return a p below `2 / 2^16 = 3.05e-5`, so no claim in this study can be significant past that however large its effect.
**Axes added after this section was written, each because a measurement forced it, and each a column of `results.csv`:** `anchor` (three fold-grid offsets), `lr` (three rates — the learning rate is an arm, not a constant), `flatness` (immediately beside `mae`, §7.3), `cancellation`, `data_snapshot_last_bar`, and `control` (`real` / `shuffled` / `noise`).

**The design is a star, not a cross product, and the choice is stated because it is a real one.** The full cross of 3 anchors × 3 rates × 3 controls is **27 conditions × 5 live arms × 16 folds = 2,160 arm-folds, about 140 minutes**. The star — one centre plus one departure per axis — is **7 conditions = 560 arm-folds, about 36 minutes**. The 20 extra conditions buy *interactions* (does the learning-rate effect differ at anchor 42 under shuffled returns) that nobody asked about; what every axis needs is a **common reference to depart from**, which is what a star gives. `--full` runs the cross product for the day somebody wants an interaction.

**The null arms live in the same `results.csv` and are gated, not separated.** They must never be averaged with arms that are about the market, and the mechanism is `study.reportable` — one file, one gate, the shape `records.is_reportable` already uses for replay and rehearsal provenance. Two artefacts can drift apart in columns or in vintage, and the null rows are only meaningful *beside* their real counterparts: splitting them turns the comparison the controls exist for into a join.

**The deliberately empty cell is written into the file with its reason**, not left blank. A blank cell in a results table reads as a run that failed; this one is a design decision (§6.4).

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
- [ ] **2. EXECUTION PATH PROVEN LIVE.** In a live session, a loop-produced decision becomes an order, fills, is reconciled, adopted and protected. This may be demonstrated with a deliberately permissive **rehearsal band** when the deployed band stands aside — the criterion tests the machine, not the model. **Since 20 Aug 2026 the deployed band fires** (validation Sharpe +0.483 over 8 trades, a grid maximum), so the criterion can be met by the deployed system in an ordinary session, **which is better** and is the route to take. The rehearsal path **stays in this spec**: it was the right amendment, it is implemented and tested, and it is needed again the moment a future fold stands aside.
- [ ] **2b. DEPLOYED BAND BEHAVES CORRECTLY.** Whether it trades or stands aside, the behaviour matches what validation selected, and the gate log states which occurred.
- [ ] Every decision carries an exact attribution visible in the dashboard
- [ ] Replay mode reproduces a recorded day offline
- [ ] Co-Pilot mode approval flow works

> **Amended 20 Aug 2026, and the amendment is recorded because of what prompted it.** The
> original criterion 2 read "at least one order placed, filled, and reconciled", which
> conflated two questions: *does the execution path work end to end against a real
> broker?* and *does the deployed model trade?* Only the first is a gate. The second is a
> **result**, and it already has an answer — no, correctly, because the most recent
> complete fold stands aside. Left as written, 25 Sep would have cancelled FITS by a rule
> aimed at an unfinished product, on a finished system that had decided not to trade.
>
> **A rehearsal carries four conditions**, each closing a way it could become a lie: its
> band is stated in the run banner and in the provenance of every record it writes, which
> is `rehearsal:<reason>` and **never** `live`; **no metric, table or figure in the report
> may include a rehearsal decision**, asserted by a test over GB-19's inputs rather than
> left to each caller; the position is **closed before the session close**, satisfying the
> DAY-legs condition below — a rehearsal that holds overnight fails the rehearsal; and it
> is **sized small**, because the point is that the path works, not that it was
> consequential.
>
> **Condition on this gate — AMENDED 23 Aug 2026 by ruling; the original text is kept
> below because a condition that quietly changes shape is worse than one that is refused
> in the open.** As written on 20 Aug it read: *"the loop must not hold a position
> overnight while the protective legs are `TimeInForce.DAY`. Either the policy moves to GTC
> before any overnight hold occurs, or the loop flattens at the close."*
>
> **Both escapes are refused, and the gap is accepted and reported instead.** GTC is not
> available: Alpaca returns `fractional orders must be DAY orders` (code `42210000`,
> measured), so GTC is reachable only by rounding to whole shares, and `shares_for` is
> shared with the backtester — whole-share sizing would change position sizing across the
> **entire study**, not just the executor. Flattening at the close is worse still: the
> backtest holds overnight, so a live loop that flattens is running a different strategy
> from the one being evaluated, and a protection gap would be closed by opening a
> **parity** gap.
>
> **What the gate now requires instead:** the gate log states the ruling, and states the
> residual in these terms — *for the interval between a session open and a successful
> re-arm, and for any session the loop does not run at all, an open position carries no
> broker-side protection.* Rule 2 of the GB-26 policy (re-arm before any entry, first thing
> on any start) closes the first case for every session the loop runs, and does nothing for
> a session it misses. If the band stands aside through the gate the log still records that
> the condition was **never exercised** — an untested guarantee is not a satisfied one.
>
> **And it is a mechanism, not a paragraph.** `live_loop.overnight_residual` prints the
> block at session start whenever the book already holds something, and
> `SessionReport.held_at_open` — read **before** the first reconciliation, the only moment
> the book still describes what was carried *into* the session — puts the same fact in the
> session summary. The cycle records cannot answer it: a position re-armed in cycle 1 looks
> identical to one opened in cycle 1. **Operating decision for the GATE 2 sessions: the
> loop runs continuously between them**, so the half of the residual rule 2 cannot cover —
> a session the loop misses entirely — does not arise.

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
| GB-9 | `builder.py` — window assembly from channel config → `WindowBatch`. Declares `min_history_bars`, the parity floor a live caller must supply. **Corrected in GB-27 from 352 to 445, and the correction is the instructive part:** the original warm-up bounded the RSI seed's *weight* below 1e-7, which is sufficient only if the seed *difference* is at most 1 — it is a difference of average gains in price units, and near RSI 50 the residual (3.815e-06) is the same order as a float32 ulp (5.95e-06), so 352 was marginal by construction. It was **verified on one symbol at one timestamp** and holds in **32 of 125 symbol-timestamp pairs**. The target is now 1e-10, `(13/14)^311`, warm-up 325, floor 445 | E3 | B | 5 | Shapes correct for C0 and C2; warm-up trimmed; `min_history_bars` is derived from a stated decay target rather than tuned until a sweep passes |
| GB-10 | `test_no_lookahead.py` — perturb-future causality harness in reusable `tests/causality.py` | E3 | B | 3 | Perturbing `t+1` leaves all values at `t` unchanged, in **both** perturbation modes; a deliberately leaky function is rejected |
| GB-11 | `PersistenceForecaster` + the `Forecaster` contract test. Adds `FitProvenance` (§4.2) and `Forecaster.fitted` (§4.3) | E4 | B | 3 | Contract test passes for persistence; a future model plugs in by appending one line to `FORECASTERS`; every property is demonstrably able to fail |
| GB-12 | `README.md`, `ARCHITECTURE.md`, `PROGRESS.md`, `DECISIONS.md`. The README must state that a clean install reproducing reported results uses `pip install -r requirements.lock`, not the unpinned upper bounds in `pyproject.toml` | E1 | N | 2 | A newcomer can run the project from README alone |

### Sprint 2 — Offline Vertical Slice · 29 Aug – 11 Sep → **GATE 1**

| ID | Task | Epic | Own | SP | Done when |
|---|---|---|---|---|---|
| GB-13 | `DLinearForecaster` — trend/remainder decomposition + linear maps, **one map per component per channel** so GB-30's attribution is a regrouping rather than a reconstruction. No intercept, so `Σ per_channel == forecast` holds by construction. Registered in `glassbox.model.ALL_FORECASTERS` | E4 | B | 5 | Contract test passes **unchanged**; trains on one fold; weights addressable by channel name |
| ~~GB-14~~ | ~~`NLinearForecaster`~~ · **CUT** — `DLinear` alone satisfies the baseline requirement | E4 | B | ~~2~~ | — |
| GB-15 | `train.py` — one training run: statistics, fit, checkpoint, loss curve. The loop itself stays inside `fit` per §4.3. **Normalisation statistics are fitted on the training split alone**, and made so structurally: windows are built **once** with those statistics and split by timestamp afterwards, so there is no second `build_windows` call to hand a second `ChannelStats`. A checkpoint is a **directory** — `model.json` (weights + `FitProvenance`), `checkpoint.json` (config hash, `ChannelStats`, ranges seen, how training ended), `history.csv` (per-epoch losses). Loading **refuses any config-hash difference**, unrelated keys included. **Training is universe-wide with a per-symbol scaler** (ruled 2026-08-17): per symbol a fold gives 501 windows, so `501 × 4 = 2004` equations against DLinear's 4,800 parameters — 0.42×, underdetermined — where pooling five symbols gives 2.09×. The decisive argument is comparability, not fit: **FITS at 1,200 parameters is already overdetermined per symbol at 1.67×**, so training DLinear per symbol while FITS trains pooled would handicap one arm and GB-49's grid would report that handicap as architecture. `train` takes `{symbol: frame}` and iterates it **sorted**, because the pooled batch's row order decides the mini-batch partition and determinism must not depend on how a dictionary was built. The checkpoint holds `{symbol: ChannelStats}` | E4 | B | 5 | Two runs with same seed give identical weights, **bit-identical**; a checkpoint round-trip reproduces predictions exactly; the stored statistics equal a recomputation on the training rows and differ from a whole-frame fit; early stopping fires on a synthetic task whose validation loss rises; a checkpoint from a different config hash is refused; symbol order does not change the weights |
| GB-16 | `predict.py` — batch + single-window inference. **The checkpoint decides everything**: weights, channel set, window geometry and the per-symbol scaler all come from it, and the live configuration is passed in only so its hash can be compared. **Per-symbol statistics come from the checkpoint, never from the caller** — the caller names a symbol, so it cannot normalise AAPL's window with NVDA's numbers. **A symbol the checkpoint has no statistics for is refused**, naming it and listing what exists: falling back to another scaler or to none presents the shared weights a distribution they never saw and returns a plausible forecast rather than an error, and fitting a scaler at prediction time is worse because the only data available includes the period being predicted. A **reordered** channel set is refused as firmly as a different one — the weights are indexed by position. `require_current_config` re-checks the hash at the point of use, because GB-26's loop can outlive the config it started with | E4 | B | 2 | The batch and single-window paths are **bit-identical** for the same window, asserted on every test timestamp; a config-hash mismatch raises naming both hashes; an untrained symbol raises; inference reproduces the training run's own predictions exactly |
| GB-17 | `walkforward.py` — fold generator with strict boundaries and a **target embargo** | E5 | B | 3 | Within a fold, no timestamp appears in two splits **and no window's target crosses a split boundary** — the last `H` window-ends of each split are dropped. Disjoint ranges alone do not prevent the leak. Asserted directly, and the assertion fails when the embargo is set to zero |
| GB-18 | `backtest/engine.py` — event-driven, fees, slippage, SL/TP. **Four pricing rules, each tested:** a signal fills at the **next bar's open** (never the signal bar's close); on a bar breaching both levels the **stop** fills; slippage is **adverse on both sides**, so a flat round trip costs `2 × (fee_bps + slippage_bps)`; a **gap through a level fills at the open**, logged as `stop_gap` / `target_gap` so the report can separate rule cost from gap cost. A fifth rule was added after review: **a missing bar is a halt, not an exit** — an unfilled exit is carried to the symbol's next traded open, an unfilled *entry* expires, a halted position is marked at its own last printed close, and a symbol whose history ends early is liquidated at that close. Sizing is **injected** (`PositionSizer`), never stubbed in the engine. A **standing accounting invariant** runs at the end of every backtest, not only in tests: with the book flat, final equity equals initial cash plus the sum of the trade log | E5 | B | 8 | Hand-checked 3-trade scenario matches by hand, to the cent, against literals derived independently of the engine; the halt and short-history cases both reconcile to the invariant |
| GB-19 | `metrics.py` — MAE, RMSE, direction accuracy, return, Sharpe, MDD, hit rate. **Administrative exits** (`Trade.strategy_exit is False`, i.e. `end_of_data`) are **included** in the equity curve and total return — the curve must be complete and the capital was genuinely returned — and **excluded** from hit rate, average trade and every other per-decision statistic, because no decision was made. Filter on the boolean field, never on the `exit_reason` string. **Three rulings, all in §7.2:** Sharpe from the equity curve's daily returns rather than per-trade returns; NaN below five strategy trades; and **direction accuracy referenced to always-long rather than to persistence**, because persistence forecasts zero and so has no direction — measured NaN in 16 folds of 16, with the achievable bar at the fold's own up rate, 0.5625 across the 16, rather than 0.5. The first version of this rule used the per-fold majority class (0.5865) and was **corrected in GB-20**: taking the majority class flips to "always short" on folds whose test period fell, which is a bar set with test-period knowledge and unreachable by a long-only system, however conservatively it errs | E5 | B | 3 | Sharpe, max drawdown and direction accuracy match literals computed by hand away from the code; an administrative exit changes total return but not hit rate; a trade flagged administrative is excluded whatever its `exit_reason` reads; persistence's direction accuracy is NaN and the delta against it stays NaN rather than being quietly replaced |
| GB-20 | `signal.py` — trend strength → signal, thresholds calibrated on validation, plus `backtest/calibrate.py`, which holds the grid search. **The calibration lives one layer up from the decision logic**: the ruling is that candidates are scored by the real backtester, and the engine layer may not import the harness — nor may the live path, which imports `signal`. `signal.py` contains **no numeric literal at all** (a syntax-tree test enforces it); the grid in `calibrate.py` is expressed as **quantiles of the fold's own validation forecast distribution**, so it is scale-free across arms. **Ruling: a fold whose best candidate has no positive validation Sharpe stands aside** — `Thresholds.never()`, no trades, a flat curve and `stood_aside=True` — rather than trading the least-bad of fifteen candidates validation had just rejected. GB-57 reports how many folds stood aside. **The scheduled `model.batch_size` re-check ran here and reverted the setting to 64.** Full batch had been adopted on a 16/16 MAE improvement, a metric §7.3 bans as a headline; measured end to end on the metrics the study does report, mini-batch wins on all three — mean total return +0.44% vs +0.07% per fold (positive in 9/16 vs 6/16), mean Sharpe +0.65 (n=14) vs −0.09 (n=10) with full better in only 3 folds of 16 head to head, and direction 0.5071 vs 0.4751, better in 13 of 16. Full batch won MAE in 16 of 16 and lost everything else, which is precisely the flattening §7.3 predicts | E6 | B | 5 | Thresholds differ per fold; none hardcoded; calibration passes `assert_fit_isolated`; a fold with no profitable band trades nothing; the batch-size re-check is reported on Sharpe and direction, and the setting in force at the end of it is the one the headline metrics chose |
| GB-21 | `risk.py` — sizing, gross exposure cap, SL/TP attachment. `size_positions` for the live path and `position_sizer` for the backtester are **one arithmetic behind two shapes** (`room_for`), so a backtest and a live account cannot size differently. **Three caps, tightest wins:** `max_position_pct`, `max_gross_exposure` counting exposure already open, and **available cash** — the last is arithmetic rather than risk, and lives here because the live executor has no guard of its own. Sizing **may reject or shrink an order and may never create one**. The property tests run the engine's own `_require_sizeable` over Hypothesis-generated accounts, so "the sizer satisfies the engine" is a proof rather than an agreement | E6 | N | 3 | Property test: no config produces an over-limit position, and the engine's guard cannot fire for this sizer |
| GB-22 | `executor.py` — Alpaca order submission, manual smoke order. **Must convert notional to shares through `engine.risk.shares_for`**, the same function `backtest/engine.py` uses (GB-18) — never its own arithmetic and never the broker's default rounding. Fractional shares, and a notional below `risk.MIN_ORDER_NOTIONAL` places no order — **verified against the live paper API in GB-22, where GB-18's `MIN_SHARES = 0.001` proved to be the right idea in the wrong unit**: Alpaca enforces a **$1.00 minimum notional**, not a share-count floor, and silently truncates a quantity to **nine decimals**, both of which now live inside the one shared conversion. **Parity finding, established against the API rather than assumed:** Alpaca **refuses a bracket, OCO or OTO order on a fractional quantity** (`fractional orders must be simple orders`) while accepting the same bracket on one whole share, so the backtester's "entry carries a stop and a target" **cannot be expressed as one live order**. A held fractional position *can* be protected by a **standalone stop and a standalone limit**, which is what the executor does — at the cost of three limitations the report must carry: no OCO linkage (if one fills the other must be cancelled), **day orders only** (`fractional orders must be DAY orders`), so protection expires at every close and is **re-established each session**, and a residual divergence that is **narrower than it first appears** — the stop is a fixed price, so an overnight gap leaves the re-armed stop marketable at the open and it fires there, which is exactly what GB-18's `stop_gap` rule models. What genuinely diverges is the **seconds between the open and the arming** and, far more dangerously, **a cycle in which arming fails and nothing notices**. GB-26 carries the five-rule policy that closes both.  If the backtest buys 99.98 shares and the executor floors to 99, the two systems describe different things and every test still passes | E7 | N | 5 | One paper order placed and confirmed; a test asserts the executor and the backtester derive the same share count from the same notional and price, driven through both real paths rather than by calling the conversion twice; the FakeBroker suite needs no network and refuses what the live API was measured to refuse |
| GB-23 | Position reconciliation against Alpaca. **Scoped down from a full order-lifecycle state machine** (`SOLO_BUILD_PLAN.md` §2): reconcile-from-truth is what prevents the failure that matters. Positions and open orders are fetched every cycle and on start-up, compared against local state, and **every divergence is resolved in favour of the broker** and logged loudly. **Three cases, each ruled:** a broker position the system has no record of is **quarantined, never adopted** — the system has no entry price, stop or provenance for it, so it can neither protect nor explain it, but it is recorded and counted against buying power because it is not free; a local position the broker does not have is **dropped**, which is also what a filled stop looks like; a quantity mismatch **takes the broker's number**, which is how a partial fill is absorbed. Also detects holdings with **no live protective orders**, which is the input GB-26's rule 3 acts on — detection here, action there | E7 | N | 5 | Local state matches broker after a forced restart, verified by killing a cycle mid-flight against the real paper account; FakeBroker tests cover all three divergence cases and a deliberately desynchronised book |
| GB-24 | `smoke_offline` — one command: data → features → folds → train → calibrate → backtest → table. `--folds N` (default 1, from the start of the fold list) and `--model {persistence,dlinear}`. **Offline by construction**: the cache is checked first and a missing file exits 2 naming the symbols, because `load_history` would otherwise download and the claim would silently weaken to "works when yfinance is up". **Every arm is run, not drawn** — persistence and buy-and-hold both go through the same train/calibrate/backtest path. **Two reporting rulings:** a fold that stands aside is a labelled row and **counts in the return mean** (its return is a true zero; averaging only the folds where the strategy chose to act is selection on the strategy's own decision) while the Sharpe mean excludes it by construction, both counts printed; and `dir_ref` sits in the column immediately right of `direction`, with every delta column naming its own reference | E5 | B | 3 | Single command produces a metrics table in under five minutes with no network — measured at 39.3s for 16 folds, 8.4s for one |
| GB-25 | **GATE 1 review** — offline slice + persistence comparison written up | E1 | B | 2 | Checklist §8 fully green, recorded in PROGRESS.md |

### Sprint 3 — Live End-to-End · 12–25 Sep → **GATE 2**

| ID | Task | Epic | Own | SP | Done when |
|---|---|---|---|---|---|
| GB-26 | `live_loop.py` — scheduler, market-hours guard, 60s polling. **Must drop the in-progress bar for the current session** via a function named for what it does (`drop_incomplete_bar` or similar), logging each drop with its timestamp — the model is trained on completed daily bars only. **Must request at least `builder.min_history_bars(cfg)` bars** (**445** for `C0_base` since GB-27; it read 352 until then), never `input_len`: recursive channels need warm-up beyond the window, or the live values differ from the training values at the same timestamp. **Measured 2026-08-18, and this is not hypothetical:** `data/live.py` derives its default lookback as `input_len × 2 = 240` calendar days, which returned **163 trading bars** — 282 short of the 445 required — and `load_live_bars`'s own guard raises only below `input_len` (120), so 163 **passes**. About **645 calendar days** are needed for 445 sessions. The failure is not a silent divergence — `indicators.rsi` emits NaN for its whole 325-bar warm-up, so `build_feature_frame` returns an **empty frame** rather than contaminated values — but nothing on that path names the real cause, and a caller trusting the default gets an empty frame with no explanation. Both the default and the guard must be restated in terms of `min_history_bars` **Carries the five-rule protection policy (ruled 2026-08-18, DECISIONS):** (1) arm protection in the **same cycle that observes the fill**, polling for it rather than waiting for the next tick — the open is the wrong minute to be idle; (2) **re-arm every open position's stop and limit at the start of every session, before anything else** — entries come after protection; (3) every cycle **verifies both legs are live at the broker**, and a position without protection is a **risk event, not a warning** — log loudly, arm immediately, and **flatten at market if arming fails twice in succession**, because an unprotected position is worse than a closed one; (4) when one leg fills, **cancel the other in the same cycle and verify the cancellation** rather than assuming the fill implies it; (5) GB-57 states the residual as the **arming interval plus any arming failure**, never as "unprotected overnight". **Two rulings taken in GB-26 and recorded here.** **(a) Stale data — the symbol loses its entry and keeps its stop.** A symbol whose last completed bar predates the last completed exchange session is excluded from forecasting, ranking and entries, and is **named in the cycle log and in the cycle report** — silence is the thing to avoid, not the exclusion, because `top_k` is taken across the universe and dropping a symbol genuinely changes which are chosen. But its **protective legs are verified and re-armed exactly as any other position's**: *a stale window may never justify opening risk and may never suspend the management of risk already taken.* Its model-driven exit cannot be computed, and the honest statement is that the stop and the target are the risk control that does not depend on the model. If every symbol is stale the cycle is a logged no-op after reconciliation; there is no partial-universe abort threshold, because that would be a number nobody measured. **(b) The first cycle may reduce risk and may not add any.** No entry is submitted on it; exits, reconciliation and protection all run in full. Rule 2 above says re-arm before entries and the first cycle *is* that moment; reconciliation has just rebuilt the book from a single observation of the account; and after a mid-session restart an entry the previous process submitted can be briefly absent from the order history, so re-entering would double a position the loop cannot un-double. An exit is not deferred, for the reason GB-28 will not let one be crowded out. **The session window is the exchange calendar's, not the configured clock's** — the Israel window says when the process is willing to run, the calendar says whether there is a market, so a holiday closes the loop and a half-day closes it early. **`smoke_offline --prepare-live DIR` writes the checkpoint and the band the loop loads**, because `live_loop` may not import the harness; a **missing or stood-aside band is `Thresholds.never()`**, so the session runs, records and explains and trades nothing, rather than inventing a threshold that would make every decision in it unexplainable. **RULING 19 Aug 2026 — decide once per completed bar, manage every cycle.** A decision's `as_of` is the last *completed* daily bar and does not change during a session, so at 60s polling the loop re-derived the identical verdict 390 times and recorded 1,950 identical rows a day. The entry decision is now computed and recorded once per `(as_of, symbol)`; reconciliation, protection, trade emission and exits are explicitly outside the rule and run every cycle. `decision_id` becomes `{as_of:%Y%m%d}-{symbol}` — **derived from the bar, not the cycle** — so the entry's `client_order_id` makes a second entry for one bar refusable at the broker (measured: Alpaca returns `40010001 client_order_id must be unique`, and consumes an id permanently, including after cancellation). **The two rulings compose**: a verdict the first cycle may not act on leaves the bar *undecided*, because recording it would strand it — the next cycle would decline to re-decide and the entry would never be sent. Acceptance adds: N cycles in one session produce exactly one entry decision per symbol while protection is verified N times; a repeated entry is never submitted when the fill has not appeared; a restart mid-session reads what it already decided. | E7 | N | 5 | Runs a full session unattended; an in-progress bar is excluded by an explicitly tested function, and each drop appears in the run log; every open position has both protective legs verified live in every cycle, and a simulated arming failure twice in a row flattens the position; the live fetch requests at least `min_history_bars(cfg)`; a stale symbol is excluded from ranking, named in the report and **still re-armed**; the first cycle submits no entry and does submit an exit; `--dry-run` produces decisions and records while submitting nothing; an interrupt finishes the cycle and prints every order still working |
| GB-27 | **Train/live parity test** — identical windows from both paths. **Sweeps, never samples:** every symbol in the universe at ≥20 timestamps, byte-identical `X` by `np.array_equal`, no tolerance. A tolerance is where a one-bar offset hides. It also asserts the teeth: a one-bar offset breaks parity, a bar *after* the window cannot change it (GB-10 causality), too little history **raises** rather than differing quietly, a spliced two-source frame is refused, and the per-window `symbols` tuple survives pooling. **This task corrected the parity floor** — see GB-9 — after the swept version failed at 352 where the sampled version had passed | E3 | B | 3 | Byte-identical `X` across the whole universe at ≥20 timestamps; the previous floor demonstrably fails the same sweep; the 1e-9 floor (414) is shown not to be universally identical, so the declared margin is measured rather than assumed |
| GB-28 | `rank.py` — cross-sectional top-K selection. Orders by descending `trend_strength`, **breaks ties alphabetically by symbol**, then takes `signal.top_k`. The tie-break is the design: Python's sort is stable, so without a second key a tie resolves by the caller's insertion order, which in the live loop is the order symbols came back from a broker call — two identical days would select different names and GB-32's replay would not reproduce. Only `enter_long` competes; **an exit is an obligation on capital already committed** and is never crowded out. A non-finite strength is refused rather than sorted | E6 | B | 2 | Deterministic ordering across repeated runs and across input orderings; ties broken alphabetically, and the tie-break proven independent of the descending sort |
| GB-29 | `DecisionRecord` persistence (JSONL) + config hashing. **Also emits live `Trade` records structurally identical to the backtester's**, so GB-19's metrics run unchanged over paper results. Without this the live trade log does not exist: a stop or limit fill makes the position vanish from `get_positions`, GB-23 drops it (correctly — it is read-only), and **nothing records the exit**, so the report cannot show backtest and live over the same period. Construct them from the broker's **filled order history** — query orders filled since the last cycle, match to the `Book`'s managed positions by symbol and order ID, and emit a `Trade` on the closing fill. **Requirements:** `exit_reason` derived from *which* order filled — the stop leg, the limit leg, or a signal exit — using the backtester's own vocabulary, **including `stop_gap` and `target_gap` where the fill price shows the level was gapped through**; **costs computed from the actual fill prices the broker reports, not from the configured bps** — this is the one place the live system can measure what the backtest assumes, and GB-57 states the realised slippage against the 2 bps modelled; **no `Trade` is ever emitted for a quarantined position**, since the system did not open it and has no entry basis for it **Uniqueness is enforced at both ends (ruling, 19 Aug 2026):** `save_decision` refuses a second record for the same `(as_of, symbol, config_hash, provenance)` and `load_decisions` asserts uniqueness on load, over every record in the month files the range touches rather than only those the filters keep. A torn line is still tolerated and logged — that is what a process killed mid-write leaves behind — but a duplicate raises, because it is a defect in the writer and corrupts every count, mean and rate taken over the log. `provenance` is in the key and **widens** it: a live decision and a replayed one at the same bar are two runs of the world, not one fact written twice. **A third provenance (GB-40, 20 Aug 2026):** `rehearsal:<reason>`, for a live run on a deliberately permissive band. `records.is_reportable` is the single place that says which provenances may reach a metric — only `live` — so replay and rehearsal are excluded once rather than at each caller. | E6 | B | 3 | A record can be replayed into an identical decision; a FakeBroker test in which a position disappears with a filled stop order in the history emits a `Trade` with the correct reason and prices; GB-19's metrics run over a live trade log with no changes |
| GB-30 | `explain/channel.py` — exact per-channel attribution. **The summation lives one layer below, as `Attribution.from_terms`** (ruled 2026-08-18): `explain` sits *above* `model` in §3.1's stack, so a model delegating upward would need a cycle and an exception in the import contract. Putting the algebra on the schema that declares the invariant costs neither, and makes the project's central claim structural — **every `Attribution` in the system is built by the one function that refuses a residual**, naming an intercept and a re-added normalisation as the two causes worth checking. Each layer keeps a job: the model says *which* terms exist (architecture), the contract sums them (invariant), `channel.py` is the face GB-32 and GB-53 call and **re-checks the model's total against `predict` itself** — the one failure an exactness check inside a model cannot see is a model that explains a forecast it did not make. **`forecast_total` now comes from `predict`, not from the contributions' own sum**, so §4.4's properties 3 and 4 are two real checks rather than one check and one identity; the residue is then the float32 cast, measured at **5.005e-08 worst over 1,000 random windows**, a 200× margin. **Percentage shares are normalised by the gross, not the net**: `share[c] = c / (sum of the absolute contributions)`, so opposite signs read +57% / −43% rather than 400% / −300%, every share is bounded in [−100%, +100%] and the sign survives; the cancellation that hides is reported separately, as the net contribution over the gross one. Percent-of-net is rejected because it is unbounded and **discontinuous at a zero forecast**, which on daily log returns is where the dashboard renders most often. **A forecast of exactly zero is defined**: all contributions zero (persistence) gives every share 0.0, never NaN and never an empty panel, preserving GB-11's ruling; contributions that *cancel* to zero keep well-defined shares and a cancellation of 0.0 — the gross denominator is undefined only in the one case where "nothing" is the true answer | E8 | B | 5 | `Σ contributions == forecast` within 1e-5 **over 1,000 random windows, with the maximum observed deviation reported**; a decomposition that does not close is refused at construction; shares are bounded and defined at a zero forecast; no perturbation library is importable from `explain/`, asserted by a syntax-tree test |
| ~~GB-31~~ | ~~Per-lag attribution heatmap data~~ · **CUT** — presentation polish; per-channel attribution already proves exactness. `Attribution.per_lag` is optional (§4.2) | E8 | B | ~~3~~ | — |
| GB-32 | `narrate.py` — attribution → readable prose, English and Hebrew. Three parts in a fixed order: what was forecast, which channels produced it ranked by share, and what was done about it with size and stop. **Every number is read off the objects, never recomputed** — the magnitude is `Attribution.forecast_total`, the shares are `channel.shares`, the levels are the `Thresholds` the decision actually used — so the prose and the dashboard cannot disagree by a rounding step and leave the explanation under suspicion. **The cancellation is spoken, not hidden**: below `OFFSETTING_BELOW = 0.5` (the point at which the opposing side reaches a third of the leading one) the narrative says the channels largely offset and names the surviving fraction, rather than presenting a small difference of large numbers as the whole story. **A hold names the calibrated band it did not reach**, and distinguishes the three causes — below the entry level, above the upper bound (implausible, not weak), or admitted by the band and unconfirmed by the path — because one sentence for all three would be false in two of them. **Persistence renders as a baseline that made no directional call**, never as a model predicting flat: `sign(0)` is not a forecast of no movement, and a model whose channels *cancel* to zero gets a different sentence because that zero is a view. **No claim about profitability, in either language, ever** — asserted against a word list rather than left to the author. **Bidirectional text (ruled 2026-08-18):** direction travels as metadata on the returned `Narrative` so the renderer sets `dir`, never inferred from the first strong character; every left-to-right run is wrapped in **U+2066/U+2069 isolates** rather than LRE/PDF embeddings or bare LRM marks, because only an isolate cannot influence the level of the text around it; and an atom is the **whole** unit including its sign and its percent sign, so a percentage can never be reordered away from its label | E8 | B | 3 | Enter, hold and exit render in both languages; every percentage in the prose equals the one in the `Attribution` it came from; a low-cancellation case produces the offsetting sentence; no rendering in either language contains a profitability claim; in Hebrew no Latin or numeric run appears outside an isolate and every isolate is balanced |
| GB-33 | Attribution exactness test across all models. **Satisfied by the work already done rather than by a new test** (confirmed 2026-08-18): §4.4's properties 3 and 4 are parameterised over `model.ALL_FORECASTERS` and run over **1,000 random windows per forecaster**, and GB-30 added a second, independent pass in `tests/explain/test_channel.py` that iterates the same registry through `explain.channel.attribute`. **The registry is the integration point**: adding `"fits": _fits` to `ALL_FORECASTERS` is the whole of GB-41's wiring, and no test file is edited — a test a model can edit is a test that model has judged itself with. Both properties have teeth: `MisreportingForecaster` breaks 3 and `NonAdditiveForecaster` breaks 4, and each is asserted to be rejected. **One constraint this hands GB-41, stated here so it is met rather than discovered:** the contract asserts `tuple(per_channel) == batch.channels`, and §6.4 makes FITS univariate — so FITS must return **every active channel**, with `0.0` for the ones it does not consume, exactly as Persistence does. An attribution that named only the channels a model reads would be an attribution the dashboard renders with panels missing | E8 | B | 2 | Parameterised test green for every forecaster |
| GB-34 | Dashboard skeleton — layout, positions, PnL. **Must carry the model's measured reliability persistently, in the panel rather than per decision** (ruled 2026-08-18, after GB-32): **direction accuracy, the always-long bar it is measured against, the number of folds it rests on, and the date of the measurement.** GB-32's narration reads as a confident claim — *the model predicts a 2.02% rise over the next 4 trading days* — and it is right not to caveat every sentence, because a hedge on every line is noise a reader learns to skip. The place for the track record is therefore the frame, not the sentence: standing, unavoidable, and the same whichever decision is on screen. This project's own measurement is that the model's directional calls are **indistinguishable from chance and fall 3.8 to 5.0 points short of always-long at every one of three grid anchors**, and **a system that explains a decision while hiding the decider's track record is doing the thing this project exists to oppose** **Rendered as HTML, not `st.dataframe`** — near-black ground, monospace uppercase headers at wide tracking, thin dashed row rules, orange on the header row only, numerics right-aligned with tabular figures. The numbered ruler runs down the left edge as well as across the top (top ruler reads `0`, not `00`), the content column is uncapped so charts use the full width, and the masthead stacks PROJECT / SYSTEM / VERSION. | E9 | N | 5 | Live positions visible and refreshing; the reliability panel is visible on every screen and states direction accuracy, the always-long bar, the fold count and the measurement date. **The numbers are read from `reliability.json`, measured by `smoke_offline --prepare-live`** — a hardcoded track record is right on the day it is typed and wrong from then on, and a missing file renders as `NOT MEASURED` rather than as an absent panel. **Colour ruling (2026-08-18, not negotiable):** orange is **interface chrome only** — labels, rules, borders, registration marks, active states, and the calibrated threshold, which is a rule line rather than a measurement. The **blue spectral ramp is the data encoding**, dark for slow and light for fast, as in the Hebrew proposal, the architecture report's three figures and the vision script. **A consequence that had to be decided:** a ramp cannot encode a sign without a second data colour, so **sign is carried by geometry and a glyph, never by hue** — positive filled and right of the zero rule, negative hollow and left, every signed figure prefixed ▲/▼. The charts are hand-built SVG, which is what makes the threshold line, the ramp assignment and the bar geometry unit-testable without a browser, and adds no dependency |
| GB-35 | Dashboard — forecast path chart per symbol **Nothing is drawn on top of data.** Every annotation — the NOW marker and the state of the calibrated band — sits in a strip below the plot area behind a hairline; only the threshold rule and the registration line remain inside the plot, and both are reference geometry rather than labels. The chart carries no `height` attribute beside `width=100%`, because a fixed height beside a viewBox letterboxes the drawing and centres it, which is what left a third of the viewport empty. | E9 | N | 3 | Path renders with history overlay |
| GB-36 | Dashboard — decision log + channel contribution bars **A decomposition that is mostly cancellation says so.** Below `EXPLANATION_FRAGILE_BELOW = 0.20` the row is marked FRAGILE in its cancellation cell and in the expander title. The level is algebra, not taste: cancellation is the reciprocal of how much the decomposition amplifies an error in any single channel, so 0.20 is fivefold amplification. Expander titles carry the trend strength and the cancellation so the list is scannable unopened, and a decision that produced no order shows an em dash — a blank cell in a technical table reads as missing data. | E9 | N | 5 | Every logged decision is inspectable |
| GB-37 | Co-Pilot mode — recommendation + approve/reject flow. When `live.mode` is `co_pilot` the loop **queues** the explained recommendation instead of executing it, and the dashboard surfaces it with Approve and Reject. **The queue is persisted**, because the approver is a different process and a recommendation that vanished on restart is one nobody ever answered. **No expiry timer** (ruled 2026-08-18): a timer would be the system deciding "no" on the operator's behalf and recording nothing, which is the unrecorded decision this project exists to remove. **Both answers write a decision record** — a rejection that left no trace would make the log a record of what the system wanted rather than of what happened, and "the operator said no" is the only place a human enters the loop. `executor.approve` does **not** re-check `live.mode`: the mode decides whether an order needs an answer, and this is the answer. Answering twice is refused rather than resubmitted, because two dashboard tabs with one click each is the realistic way that happens **An answer amends the decision, it does not add one** (ruling, 19 Aug 2026). The loop records the decision when it queues the recommendation, so `records.amend_decision` replaces that record in place — through a temp file and one atomic rename. Appended, the approval rate over the log would have been the answers divided by twice the recommendations. | E7 | N | 5 | Rejection leaves no order at the broker, asserted against the broker and not against the return value; both answers write a record; a second answer is refused |
| GB-38 | Replay mode — recorded-day playback, fully offline. **Scope extended 2026-08-18, and it now carries two GATE 2 items that cannot otherwise be demonstrated.** The deployed model's most recent fold **stands aside**, so no live session can produce a recommendation — which leaves Co-Pilot approve/reject (GB-37) with nothing to approve and replay with no day that traded. **Replay drives the live loop's own cycle**, the same code path and not a reimplementation, over historical bars from **a fold whose band did fire**, with **the thresholds that fold actually calibrated**. That exercises forecasting, ranking, risk, attribution, narration, the decision record and Co-Pilot approve **and** reject, entirely offline. **It must be visibly a replay:** every decision record carries a **replay flag and its source fold**, so a replayed decision can never be mistaken for a live one in the log, the dashboard or the report. **Today's live session is recorded too**, band and all — a recorded session in which the system correctly declines to trade is worth having, and the demonstration shows both: the real system abstaining, and a replayed fold where it acts | E9 | N | 5 | A recorded day replays with no network; it runs the live loop's cycle rather than a copy of it; every replayed record is flagged with its source fold; Co-Pilot approve and reject are both exercised |
| GB-39 | Fault handling — retries, backoff, connectivity loss, restart safety **Scope, tightly (ruled 20 Aug 2026):** retry with exponential backoff on every broker and data call, three attempts; on connectivity loss log loudly, skip the cycle and continue rather than crash; restart safety such that an interrupted cycle leaves no orphaned or duplicated order. **No circuit breaker and no health endpoint.** `glassbox/faults.py` is added to §3.4 for this: its two callers sit at opposite ends of the layer stack, so neither can hold the helper without inverting it. **Retrying a write is safe only because the `client_order_id` is bar-derived** — a retry refused as a duplicate is the receipt for the attempt that timed out, and the order is recovered rather than reported as failed. **A gate blocker was found and closed here:** `reconcile` quarantines an unknown position and never adopts, so the loop's own fill — arriving after `_absorb_entry`'s poll window expires, or after a crash between submission and fill — was quarantined and therefore **never protected**. `live_loop.adopt_own_positions` runs between reconciliation and protection and takes the position back **only on evidence**: a filled buy whose `client_order_id` is a decision id this system mints, naming a decision on disk in this log under this provenance, carrying the order it produced. Anything short of that stays quarantined, so GB-23's ruling is untouched. **The ordering that makes adoption possible is itself the restart rule: the decision reaches disk before the order reaches the broker.** Acceptance adds: a process killed between submission and fill restarts with the position **booked and protected, not quarantined**; a position with no decision behind it stays quarantined; a well-formed decision id naming no record stays quarantined. | E7 | N | 3 | Kill -9 mid-cycle leaves no orphan state |
| GB-40 | **GATE 2 review** — live E2E with explanations, demo dry run **Walked 20 Aug 2026, five weeks before the commitment date: 3 of 6 green, not passed, and FITS is NOT cancelled** — the §8 rule fires on the gate's own date, not on a rehearsal. PASS: exact attribution visible in the dashboard (75 records, worst residual 1.248e-08 against 1e-5); replay reproduces a recorded day with `load_live_bars` and `AlpacaBroker` replaced by hard failures; Co-Pilot approve and reject, replayed. FAIL: no full unattended session; no order placed by the loop; one live session rather than the two or three `SOLO_BUILD_PLAN` §4.1 requires. **The finding: criterion 2 is structurally unreachable rather than merely unmet** — the honest deployment is the most recent complete fold, that fold stands aside, and a band that stands aside can never place an order, so it will read FAIL on 25 Sep exactly as it does today. Three ways out are recorded in DECISIONS.md; the choice is Ben's and is needed before the gate date. **Condition carried, and recorded as never exercised:** the loop must not hold a position overnight while the protective legs are DAY. | E1 | B | 2 | Checklist §8 fully green |

### Sprint 4 — FITS, Study, Report · 26 Sep – 10 Oct → **GATE 3**

| ID | Task | Epic | Own | SP | Done when |
|---|---|---|---|---|---|
| GB-41 | `fits.py` core — RIN, rFFT, LPF, complex linear, irFFT. **Report the actual parameter count**: at `input_len 120` and `cutoff_period_days 5`, `COF = 24` and the output is `ceil(124/120 × 24) = 25` bins, so the complex layer holds `24 × 25 = 600` complex = **1,200 real** parameters — **measured 20 Aug 2026, and exactly as predicted**. It is not the ~10k of §6.1, which is the paper's figure for its own configuration. DLinear holds **4,800 measured** (GB-13), so FITS is the smaller model by 4×, not the 14× the papers imply — see the §6.1 note. **Also report whether the per-symbol scaler still earns its place under FITS.** FITS applies RIN, a per-window instance normalisation, on top of the per-symbol `ChannelStats` the checkpoint holds (§4.2). Two normalisation stages in sequence is not automatically wrong — they remove different things, one a symbol's level and the other a window's — but whether the first still contributes once the second runs is a question to **measure**, not to assume. Run FITS with and without the per-symbol scaler and report the difference **Measured 20 Aug 2026: 1,200 real parameters**, exactly as predicted — 24 input bins to 25 output bins, 600 complex weights, counted as reals because a complex weight is two learnable numbers and reporting 600 against DLinear's 4,800 would flatter FITS twofold. **Attribution names every active channel** with 0.0 for the four FITS does not read (GB-33's ruling): the pipeline is linear in the window, so the model is an `(H, L)` matrix, and that matrix is built by pushing the `L` basis vectors through the **actual forward pass** rather than re-derived, so the explanation cannot drift from the forecast. Torch fits and numpy predicts, as DLinear does, which makes contract property 2 true by construction. **RIN is per-instance and property 6 catches the alternative**: a deliberately broken batch-mean variant moves the first window's prediction by 4.85 and fails with *LOOKS AHEAD*, confirming §4.4's note that was written before FITS existed. **Two measurements recorded against this task.** The mean-reversion hypothesis is **rejected**: FITS agrees with the trailing H-day return **0.5725** of the time, above 0.50 in **13 of 16 folds**, against DLinear's **0.4991** in 8 — a low-pass filter is a smoother and a smoothed extrapolation continues a trend, so FITS is a momentum model. And the **per-symbol scaler does not earn its place under FITS**: MAE is worse with it in 16/16 folds for both models, but direction improves with it for DLinear (0.5182 vs 0.4657, better in 12/16) and not for FITS (0.5098 vs 0.5133, better in 9/16). Recorded as a measured asymmetry between the arms for GB-49 to sweep, not acted on: one training budget, one initialisation, and MAE is banned as a headline by §7.3. | E10 | B | 8 | Contract test passes; the reported parameter count is **measured**, not quoted from the paper, and §6.1's predicted 1,200 is updated to the measured figure; the scaler-under-RIN measurement is reported either way |
| GB-42 | **Amplitude scale fix `(L+H)/L` + sinusoid reconstruction test** **Written before GB-41, per §6.3.** `tests/model/test_fits_amplitude.py` holds three tests. The acceptance test feeds a sinusoid of known amplitude and period and asserts the reconstructed backcast within 1e-4; it was marked `xfail(strict=True)` until GB-41 landed, so the pass that arrived with the implementation **would have failed the suite** until the marker was removed — the expectation could not be forgotten and CI did not sit red in the meantime, which would have been the same failure as a guard that fires spuriously. **The marker was removed when GB-41 landed and the test passes.** It fixes GB-41's interface: `fits.extend_spectrum(x, horizon)`, `(B, L)` in and `(B, L+H)` out, amplitude-correct, whose first `L` samples reconstruct the input. **The companion measures the trap on the FFT libraries themselves** — numpy and torch share the convention, so the correction has to live in our code: omitting the scale divides every sample by exactly `L/(L+H)`, so the file demonstrates the bug it names rather than merely asserting that something is fine. A third test states why it costs a day — the shortfall is 3.2% at L=120, H=4: invisible in a plot, and it **improves MSE**, because a flatter forecast sits closer to zero. **Corrected 20 Aug 2026, and no correct implementation could have passed the original.** The test ran at the configured `H=4`, where a 12-day cycle maps to bin `η·k = 10.333` — a bin that does not exist — so the parameter-free extension is wrong by the order of the signal itself (4.97 against an amplitude of 3.0) whatever the amplitude scaling does. It now runs at `H=12`, where `η·k = 11` exactly and the error is **2.2e-14**, with the grid choice stated in the test and a companion measuring the `H=4` failure so the choice is justified in the file. **The gap at `H=4` is not a defect to fix — it is the reason FITS has a learned complex layer.** | E10 | B | 2 | Known sinusoid reconstructed within 1e-4 |
| ~~GB-43~~ | ~~Backcast + forecast supervision (`B+F`) toggle~~ · **CUT** — `B+F` hardcoded per §5; the ablation is not essential | E10 | B | ~~3~~ | — |
| GB-44 | FITS integration + shared-weights-across-universe training. **The switch and the sharing are two different claims and are tested as two.** The switch: `model.active: fits` selects FITS with no other change, which rests on two lists agreeing that nothing made agree — the config layer validates against `VALID_MODELS` and **may not import the model layer**, so `ALL_FORECASTERS`'s keys are copied there by hand; a name in one and not the other is a switch that either cannot be reached or passes validation and raises inside `train`. A test now asserts the two sets are equal. **And there was a third list, which was the stale one:** `smoke_offline`'s `--model` choices still read `(persistence, dlinear)` after GB-41, and its `run` defaulted to the string `"dlinear"` rather than to `model.active` — so the one command that produces every study number could not select the model the study is about, while `prepare_live` (which reads the config) could. Both literals now come from the registry and the configuration respectively. The sharing: **asserted by breaking it** — perturbing the single weight matrix must move *every* symbol's forecast, since a run that quietly fitted five models would still return one object and still list five symbols. That test found the DC row of §6.1 by failing: the first perturbation it tried was row 0, which moves nothing for anybody. `fits.individual_weights` stops being decorative here (§6.4) | E10 | B | 3 | One model serves all 5 symbols, asserted by perturbing the shared weight; `VALID_MODELS` and `ALL_FORECASTERS` name the same models; a pooled universe under `individual_weights: true` is refused, naming the remedy; the manifest records the regime |
| GB-45 | `spectral.py` — per-frequency attribution, gain and phase extraction. **Keyed by period in DAYS, never by bin index**: "the 17-day cycle" means something to a reader and "bin 7" does not. Phase is converted from radians to days as `φ/2π × period`, positive meaning the model **advances** that cycle. The gain reported for a cycle is the **same-frequency** element `W[k, round(η·k)]` — the output bin carrying the input bin's own frequency onto the longer grid — and the off-diagonal leakage is not reported there but *is* inside the contributions, which are exact. The per-frequency `(H, L)` matrices are **measured from the model's own forward pass** with one bin unmasked, the same discipline as `forecast_matrix`, and sum back to it at **4.4e-16**. See §6.5 for the RIN mean, bin 0, and what the model was measured to listen to | E8 | B | 5 | Contributions sum to forecast within 1e-5, through `Attribution.from_terms` and no second summation; removing the RIN mean term makes the decomposition fail to close; a single-cycle window is attributed to its own period to 1e-9 |
| GB-46 | Learned frequency-response visualisation (`\|W\|` vs period). **In `scripts/`, not in the package** — it reads a checkpoint and writes a file, which is spec §3.4's definition of an operational entry point, and putting matplotlib in the `explain` layer would let a rendering dependency reach the live loop's import graph. The axis is in **days**, the cutoff is annotated rather than implied (a curve that simply stopped would read as a model that lost interest rather than one that was told to), and **no colour is defined locally** — the palette is imported from the dashboard, so the green/red ban asserted there covers this figure too, enforced by a test for a hex literal anywhere in the source. Adds **matplotlib** as a dependency; see DECISIONS. **What the figure shows is not what its title first claimed** — see §6.5: ~86% of the curve's variation is reproduced by a model trained on white noise, and gain tracks grid misalignment at r = −0.90. It stays the demonstration's strongest visual and it demonstrates the **interpolation cost**, which is a better story than the one it displaced | E8 | B | 3 | Plot generated from a trained model, at 200+ dpi, with the curve asserted to be `\|W[k, round(η·k)]\|` read off the model's own weight |
| GB-47 | `wavelets.py` — causal rolling DWT → `wav_a1..a3` from the **log-return** series. **One decomposition per bar, over the trailing `rolling_window` returns only, emitting the last sample of each reconstructed band.** The standard recipe transforms the whole series once, and it leaks: a DWT filter is **two-sided**, so the reconstruction at `t` carries information from bars after `t` — at level 3 with `db4`, from tens of bars ahead — and it looks exactly like a smoothing while every downstream metric improves. **The warm-up is exact where RSI's is a bound**: a DWT of a trailing window depends on that window and on nothing before it, so `rolling_window` is not a tolerance argument and there is no residue to derive. **Measured, and against expectation: the floor does not move.** `min_history_bars` is a maximum rather than a sum, RSI's 325 still dominates the wavelets' 64, and it stands at **445** for both channel sets. `PARITY_WARMUP` entries may now be a function of the configuration, so widening `wavelet.rolling_window` moves the floor instead of leaving a literal 64 beside the setting it should follow | E11 | B | 5 | Emits values from bar 64 onward; the C2_hybrid parity sweep is byte-identical at 445 over 5 symbols × 25 timestamps |
| GB-48 | Wavelet causality + additivity tests, via GB-10's `assert_causal`. **The harness is reused, not rewritten** — a second harness is a second thing that can be wrong. Both perturbation modes at three splits, on each of `wav_a1..a3` and on the whole `C2_hybrid` feature frame. **Additivity is compared against the input**, not against a sum of the module's own outputs, which is GB-30's independence requirement applied here. **And the teeth are the point:** `wavelets.whole_series_approximation` exists on purpose — the standard recipe, kept in the module for `BatchNormForecaster`'s reason — and both modes must reject it, asserted separately, because a leak only one mode saw would be a reason to doubt the other | E11 | B | 3 | `a3+d3+d2+d1 ≈ returns` to 1e-12; future perturbation inert; the whole-series version raises `LOOKS AHEAD` under `scale` and under `shuffle`; the two versions are asserted to differ, so the rejection is not of a function nobody would write |
| GB-49 | `study.py` — grid runner, seeds, `results.csv`. **STANDING REQUIREMENT — every headline claim gets BOTH tests** (ruled 20 Aug 2026). **Grid sensitivity:** does it survive different fold boundaries? **Null control:** does it survive destroying the signal? Run the arm on **within-window-shuffled returns** and on **white noise with matched variance**, and report the effect size on each beside the real one. **Neither test subsumes the other, and this project has one demonstration of each on its own data:** the `r = −0.47` timing correlation **died to a grid shift** (to −0.0025) and would have passed a null control; the FITS phase advance **died to a null control** (+1.9640 real against +1.9309 on white noise) and passed grid sensitivity at **48 of 48 cells**, more stably than the correlation that vanished. A claim surviving one and not the other is not a result. `control` is a column of `results.csv` — `real` / `shuffled` / `noise` — and `study.reportable` is the single gate that keeps a null arm out of a metric, in `records.is_reportable`'s shape and for its reason. **The learning rate is an axis of the grid, not a constant of it** (ruled 20 Aug 2026 on the measurement in DECISIONS). A fixed `model.lr: 0.001` is a hidden arm: measured over 16 folds, DLinear's per-symbol-scaled arm learns a **median weight of 5.4e-4 against an Adam step of 1.0e-3**, with **71% of its weights below the step**, and the MAE ordering against the unscaled arm **reverses** as the rate falls — 0 of 16 folds at `1e-3`, 14 of 16 at `1e-4`, 16 of 16 at `1e-5`. A study that reports MAE at one learning rate reports the learning rate. Sweep `1e-3`, `1e-4`, `1e-5` and carry `lr` as a column. **`results.csv` carries `flatness` immediately beside `mae`** (§7.3), so the pairing is a property of the file rather than of whoever writes the table from it. **And carry `flatness` — mean |forecast| ÷ mean |actual| — beside every MAE**, because over 80 fold × arm cells Spearman(MAE, flatness) is **+0.811** for DLinear and **+0.668** for FITS while Spearman(MAE, direction) is **+0.009** and **−0.128**: without that column a reader cannot tell an arm that got better from an arm that got flatter, which is the whole of §7.3's objection expressed as data rather than as advice. **`results.csv` must carry a `data_snapshot_last_bar` column** recording the last bar date of the cached data each arm ran against (see GB-4: the cache is a snapshot and never refreshes itself). **Grid sensitivity is part of the study, not an afterthought** (ruled 2026-08-18). The whole grid runs at **three fold-grid anchors**, offset from one another by roughly one third of `step_months` — 21 trading sessions at `step_months: 3` — and `results.csv` carries the **anchor** as a column. The reason is measured rather than hypothetical: GB-27b shifted the grid by one week and the timing-versus-market correlation went from r = −0.47 to r = −0.0025, so a claim resting on one anchor cannot be told apart from a property of that anchor. **A finding that survives three anchors is a finding; one that does not is a property of the grid, and the report says so.** Cost is about 93 seconds: 16 folds run in 31. **`results.csv` also carries `cancellation`** (GB-30's net-over-gross contribution ratio, averaged over the arm's windows), ruled 2026-08-18: an arm whose channels largely cancel is producing a **small difference of large numbers**, which is both numerically fragile and a statement about what it learned — GB-13 already measured initialisation changing the largest single contribution seventeenfold with the forecast barely moving, and that is exactly this quantity moving. Tracking it across arms and anchors costs one column | E12 | B | 5 | One command runs the full grid; every row carries `data_snapshot_last_bar`, its **fold-grid anchor**, its **learning rate**, its **null-control condition** and its mean **`cancellation`**, with `flatness` immediately beside `mae`; the grid runs at all three anchors in one command; the deliberately empty FITS × C2_hybrid cell appears as a **skipped row carrying its reason** rather than as a blank; `reportable` is the only route from the file to a metric and drops every null-control and skipped row; **the same seed reproduces identical numbers**, asserted rather than assumed; and `--plan-only` reports the cell count and wall-time estimate before anything runs |
| GB-50 | COF sweep for FITS (2/5/10/20-day cutoff). **It rides on the star as three more spokes rather than becoming a fourth crossed axis** (decided 23 Aug 2026): the cutoff is FITS's one hyperparameter and reaches persistence and DLinear through nothing at all, so their rows at cutoff 20 would duplicate their rows at cutoff 5 — a duplicate in a results file is something a reader eventually averages. **Each spoke carries its own null control**, so a cutoff is falsifiable on its own terms: *if FITS at cutoff 2 scores the same on white noise as on real data, that is the finding for that cutoff*. **`results.csv` carries `cutoff_period_days` with `cof` and `dead_row_fraction` immediately beside it**, for the reason `flatness` sits beside `mae` — the cutoff alone is a number nobody can interpret — and all three are **derived from `model/fits.py`** rather than recomputed, because `COF = L // cutoff` is exactly the arithmetic that gets written down twice. **The sweep turns two configuration-specific measurements into curves, and both invert the natural reading.** (a) **The dead DC row's share rises with the cutoff**: it is `1 / COF`, so **1.67% at cutoff 2 and 16.67% at cutoff 20** — the smaller model wastes proportionally *more*, because the same one dead row is a larger part of fewer. (b) **The interpolation cost saturates rather than peaking, and the deployed cutoff is on the plateau**: `η = 1 + H/L = 1 + 1/30`, so `frac(η·k) = frac(k/30)` has a **period of 30 bins**, and mean `|η·k − round(η·k)|` rises with `COF` while the retained bins cover less than one cycle and **saturates at 0.25** — the mean distance of a uniform fractional part to the nearest integer — once they cover one or more. Measured: **0.2500 / 0.2833 / 0.1833 / 0.0833** at cutoffs 2 / 5 / 10 / 20, and **exactly 0.2500** at `COF` 30 and 60. The apparent peak at cutoff 5 is a **partial-cycle sampling artefact** — 24 of 30 bins over-weights the far half — so the claim to write is *every low cutoff is saturated and the deployed one is among them*, **not** *the deployed cutoff landed on the worst cell*. The weaker claim is the true one, and the stronger one invites a reader to check the arithmetic and find it marginal. (c) **The parameter count is a function of the cutoff and §6.1's framing is cutoff-specific**: **7,440 / 1,200 / 312 / 84** reals, so at cutoff 2 FITS is **larger than DLinear's 4,800** and the paper's tiny-challenger framing inverts inside this project's own sweep | E12 | B | 3 | Four cutoffs in `results.csv`, each with its `cof` and `dead_row_fraction`; each carries a null-control arm; the geometry columns are asserted equal to the forecaster's own properties rather than recomputed; the centre of the sweep is asserted equal to `fits.cutoff_period_days` |
| GB-51 | Paired Wilcoxon signed-rank of every arm against **its own reference**, paired by fold. **The title of this row used to read "vs persistence" and that was wrong** (corrected 23 Aug 2026): persistence forecasts zero and takes no trades, so its `direction` and `sharpe` cells are **empty**, and a single reference would have tested two of the four metrics against a NaN. **The three-reference rule** applies here as in §7.3 — **MAE against persistence** on the same channel set, **direction against always-long** at that fold's realised up rate, **Sharpe and total return against buy-and-hold** at the same fold. **Pairing may not cross a condition:** fold 3 at anchor 21 is not fold 3 at anchor 0, and pairing them compares two periods rather than two arms. **Multiple comparisons are counted, corrected and stated rather than left to the reader.** Every row carries `n_tests` — **the family is every test computed in the same call**, so a report narrows the family by narrowing the table — and `p_holm` **beside** the raw `p`, never instead of it, because an adjusted value alone hides how much of the adjustment the family size did. Holm-Bonferroni rather than Bonferroni because it is uniformly more powerful at no cost, and rather than Benjamini-Hochberg because FWER is the guarantee a reader of a results table assumes and BH's needs a dependence assumption nobody here has checked. **The floor matters as much as the values:** the exact two-sided signed-rank test on 16 folds cannot return a p below **`2 / 2^16 = 3.05e-5`**, so no claim in this study can be significant past that however large its effect — and Holm over a family of ~50 still leaves a smallest achievable adjusted value near 1.5e-3, so the correction is survivable and a claim that fails it was not close. **A test is withheld rather than reported powerless:** below `MIN_FOLDS = 6` moved folds the exact test cannot reach 0.05 at all, and a p from it would be a number with no power behind it rather than an absence of effect. **Derived, never stored:** `stats.wilcoxon` is a pure function from `results.csv` to the test table, so GB-52 regenerates it rather than reading a second artefact that could drift | E12 | B | 3 | Mean, sd and p are columns of a table the report regenerates from `results.csv` alone; each metric is asserted to carry its own reference; a pairing that crossed a fold or a condition fails a test; `n_tests` is on every row and `p_holm` beside every `p`; `study.reportable` gates the test table unchanged |
| GB-52 | `report.py` — tables, per-fold boxplots and the COF curve, **from `results.csv` alone**. Nothing in it trains, backtests or reads a cache, so a figure is rebuildable in October without a 22-minute run and without the machine that made it; the significance columns come from `stats.wilcoxon`, itself a pure function of the same file, rather than from a second artefact that could drift. **The summary header prints `data_snapshot_last_bar` as its first line**, so a report built on stale data says so on its own face — and a file carrying **two** snapshots is **refused**, because a summary across vintages would average two different markets. **Every headline claim is shown at all three fold-grid anchors and under all three control conditions** in one table: a claim shown at one anchor is a property of that anchor (§7.4), and a claim not shown against its null control has not been tested (standing requirement, 20 Aug). `CLAIMS` is a list of **effects** rather than prose, so adding a claim means adding a computation that a test can check. **Three pairings are structural rather than remembered:** column order comes from `metrics.columns_for`, so `mae` cannot be printed without `flatness` beside it without deleting the pairing; the rank correlations are printed **with** every MAE table rather than once in a methods note; and every metric's reference comes from `stats.REFERENCES`, so the three-reference rule has one definition in the codebase. **The summary holds `lr` and `COF` at their reference values** and gives each its own table, because averaging MAE over three learning rates would report the mean of an axis as a result (§7 (1n)). Markdown is hand-rolled rather than through `DataFrame.to_markdown`, which would add `tabulate` as a dependency for table punctuation; a missing number renders as an **empty cell**, never as `nan`, which reads as a value somebody computed | E12 | B | 3 | Report regenerates from CSV alone; the snapshot date is the first line of the header and two snapshots are refused; every headline claim appears at all three anchors and against both nulls; `flatness` is asserted adjacent to `mae` and the rank correlations are asserted present in the rendered page; the figures are written as PNG |
| GB-53 | Dashboard — spectral explanation panel | E9 | N | 5 | Frequency bars + gain/phase visible live |
| GB-54 | Demo script + two full rehearsals **Both folds, in this order: 13 then 1.** Fold 13 is the demonstration fold (the only one of sixteen containing all five exit kinds) and is also the second-best by return at +2.71%, Sharpe 2.902. Fold 1 is its counterweight: four exit kinds, 10 stops and 4 gapped stops, **−3.59%**, Sharpe −3.391. The good quarter is shown first so the bad one is what the room remembers. A demo showing only fold 13 is showing a good quarter. | E12 | N | 3 | Runs end to end twice without intervention |
| GB-55 | Technical report — architecture and system design chapters | E12 | N | 5 | Draft reviewed by Ben |
| GB-56 | Technical report — methodology and leak-freedom chapters | E12 | B | 5 | Draft reviewed by Noy |
| GB-57 | Technical report — results, discussion, future work. **THE RESULTS CHAPTER OPENS WITH THIS AND NOT WITH A METRIC** (ruled 20 Aug 2026): *we found and fixed a correctness defect, and the headline metrics degraded.* The B+F objective was measured at **98.5% backcast** — FITS was not training the objective §6.2 describes — and correcting it moved DLinear's conditioning from a median weight of 0.54× Adam's step to 2.50×. Put the before and after side by side:

| | before the 20 Aug correction | after |
|---|---|---|
| direction accuracy | 0.5182 | **0.4959** (always-long bar 0.5564) |
| total return, mean per fold | +0.40% | **−0.31%** |
| Sharpe, where defined | +0.62 | **−0.33** |
| folds standing aside | 3/16 | **5/16** |

And state what it means rather than leaving it as an embarrassment: **the 0.5182 belonged to the arm in which 71% of the learned weights sat below the optimiser's step size** — the score of a model too under-trained to move its weights. A result that survives only while the optimiser cannot resolve the weights was never a result, so **the null is strengthened by this, not weakened.** "We found and fixed a correctness defect and the headline metrics degraded" is a sentence almost nobody writes, and it is the one that makes the rest of the report credible.  The data-quality appendix must note the **2026-08 finding that yfinance and Alpaca volumes disagree materially on 2018-05-02 and 2018-05-03 for all five symbols at once**, and must state the **feed assertion with its measured cost** (GB-30, 18 Aug): the live client requests SIP, the consolidated tape, and over 273 sessions × 5 symbols SIP agrees with the yfinance training source to **0.56 bps at worst** on any price field — while **IEX would be out by up to 193 bps**, roughly **190× the tolerance and 90× the entire modelled friction budget of 2 bps slippage**. A feed downgrade is invisible to every schema check, which is why the feed is a hardcoded constant, asserted by a test and logged on every fetch — a two-day vendor-side event, visible in `vol_z` and in no other channel. **Every reported return is a TOTAL return**, because `data/historical.py` fetches with `auto_adjust=True` and dividends are folded into the price series (GB-18); state this, or a reader comparing against a price-only benchmark finds an unexplained discrepancy of roughly the universe's dividend yield per year. **The results chapter must also discuss the regime coverage the fold cap buys and costs** (GB-17): `max_folds: 16` keeps the most recent 16 of 30 candidate folds, so the test periods run 2022-07 → 2026-07 — the market the paper account actually meets — but contain **no 2018 volatility episode and no March 2020**. If every arm performs similarly, one honest reading is that the test period did not contain a regime in which the arms differ. State this rather than leaving a reader to infer it. **Two further items are required, both methodological findings rather than results.** (1) **Weight initialisation in a summed architecture** (GB-13, **re-swept in GB-27**): DLinear sums one linear map per component per channel, so its true fan-in is `C × 2 × L = 1200`, not the `L = 120` the reference implementation's default `U(-1/√L, 1/√L)` assumes. **The report must use the swept table — 3 arms × 16 folds × 5 symbols, 80 cells — and not the single-fold one it replaces**, because the two disagree:

| axis | paper `1/√L` | fan-in `1/√(2CL)` | **zeros** |
|---|---|---|---|
| MAE vs persistence (lower better) | 2.6902 — **0/80** wins | **2.1947 — 46/80** | 2.1902 — 34/80 |
| direction accuracy (higher better) | 0.4750 — 22/80 | 0.4703 — 17/80 | **0.5182 — 41/80** |
| largest contribution (lower better) | 1.1110 — 0/80 | 0.4501 — 0/80 | **0.0653 — 80/80** |

**What must be claimed:** the published default is wrong for a summed architecture — it wins zero cells on MAE and zero on legibility — and correcting the fan-in recovers almost all of the accuracy. **Zeros is chosen on direction and on legibility, not on MAE**, where it ties with the fan-in correction (2.1902 vs 2.1947, and fan-in wins more cells, 46 to 34). Legibility is the unanimous axis, 80 of 80, with the largest single-channel contribution seventeen times smaller — and it is the axis this project exists for. **The original single-fold figures (4.00×, 1.94×, 0.344, 0.557) must not be quoted**: they were one fold of one symbol under a reverted batch setting, the levels do not reproduce, and quoting "zeros has the best MAE" quotes a coin flip. The legibility numbers *did* reproduce (1.39 / 0.43 / 0.058 against swept medians 1.02 / 0.41 / 0.063), and that contrast is itself worth reporting: the mechanical claim held at one point, the statistical ones did not. **Direction accuracy must be reported against the always-long bar** (GB-19, corrected in GB-20): that bar is the fold's own realised up rate. **Re-measured on the current fold grid at three anchors on 2026-08-18**; the figures below replace the pre-GB-27b ones this row carried — bar 0.5625, DLinear 0.5071, 5 of 16 — which were measured on a fold grid that no longer exists and must not be quoted:

| anchor | DLinear | always-long bar | gap | t | beats the bar |
|---|---|---|---|---|---|
| +0 | 0.5182 | 0.5560 | −0.0377 | −1.53 | 4/16 |
| +21 sessions | 0.5027 | 0.5523 | −0.0496 | −2.55 | 5/16 |
| +42 sessions | 0.5236 | 0.5654 | −0.0418 | −2.39 | 4/16 |

**This is the most grid-stable claim in the project**: same sign at every anchor, a gap of 3.8 to 5.0 points, and the bar beaten in 4 or 5 folds of 16 whichever anchor is used. The two rejected references gave 6 of 16 under a naive 0.5 and 1 of 16 under the per-fold majority class the first version of the rule used — both measured on the pre-GB-27b grid, and both quoted only to name the error. Both rejected references are errors of the same kind arriving through the baseline rather than the metric (§7.3): 0.5 flatters, the majority class sets an unachievable bar using test-period knowledge. The report must state which reference is used and that it is the fold's own up rate. **The diagnostic behind the shortfall must be reported too** (GB-20): DLinear calls up on **53.85%** of windows against a 56.25% up rate, and an information-free model calling up at that same rate would score **0.5016** — within half a point of what the model actually scores, so the 5.6-point shortfall against always-long is the **drift the model cannot represent**, not an inverted signal. Inverting the forecast makes it *worse* (0.4925), which rules out a sign error in the target alignment or the trend subtraction. Per symbol the spread is 0.4952–0.5261, straddling 0.50, so nothing is concentrated. The report must not claim the model is anti-informative; the honest statement is that it is **information-free on direction and structurally unable to represent drift** — see the GB-41 hypothesis in `DECISIONS.md`. Any figure from the reverted full-batch variant (0.4751, calls up 45.7%) must be labelled as such. **Report those three figures as what they are — one fold of one symbol, run under identical conditions, so the *comparison between initialisations* is sound but the *levels* are not.** GB-15 later measured all 80 arms (5 symbols × 16 folds) and found direction accuracy of **0.4971**, i.e. at chance; the 0.557 was a single fold and does not generalise. Quoting it as the model's direction accuracy would be the exact error §7.3 exists to prevent. **Standing rule for the whole chapter: every reported figure is the full-grid number, and any single-fold or single-symbol figure is labelled as such in the same sentence.** A number that does not say how many arms produced it is not reportable. (1a) **Early stopping is two mechanisms, and only one of them matters** (GB-15). Best-epoch selection on validation improved test MAE in **80/80** arms, 2.757× → 2.004× against persistence, a 27.3% mean improvement; adding `patience: 10` on top produced an **identical model in 79/80** arms and merely saved a mean 76.5 of 100 epochs. On a convex objective early stopping is therefore not a guard against divergence but a regulariser against the null space of an underdetermined system — 2,004 equations against 4,800 parameters, 0.42× — and the report should say which half is doing the work rather than crediting "early stopping" as one thing. (1b) **A worked example of why MSE-family metrics are banned as headline numbers** (GB-2, GB-15, GB-20) — the clearest demonstration the project has produced, and it must appear with **both tables shown**. `model.batch_size: null` was adopted on a **16/16 improvement in MAE**; the re-check scheduled at the same moment measured both settings end to end on the metrics the study reports and reversed the decision — mini-batch wins total return (+0.44% vs +0.07% per fold, positive in 9/16 vs 6/16), Sharpe (+0.65 vs −0.09, better in 7 folds of 16 head to head) and **direction accuracy in 13 folds of 16** (0.5071 vs 0.4751), while full batch wins **MAE in 16 folds of 16 and nothing else**. Converging more completely to the MSE optimum produces a flatter forecast, which wins an error metric and forecasts nothing. The report must present this as the mechanism §7.3 exists to prevent, caught in the project's own history rather than quoted from the literature — and must add that both arms remain **below the always-long bar of 0.5625**, so the reversal chose the better of two arms that do not beat always calling up. (1c) **Why three references, and what each one answers** (GB-21) — required, because a reader who meets three baselines with no explanation will assume they were chosen to flatter. Persistence answers "is the forecast better than a random walk?" and is a hard baseline for MAE and RMSE. Always-long answers "is the direction call better than the only constant call a long-only system could have made?" — persistence forecasts zero and expresses no direction, so it cannot answer it. Buy-and-hold answers "is the trading better than owning the same five symbols?" — persistence never trades, so measuring against it measures only that the strategy traded, and over 2022-07 → 2026-07 that comparison would have been badly misleading. **The measured answer must be stated plainly: buy-and-hold returns +7.37% per fold against DLinear's +0.45%, Sharpe 1.48 against 0.69, positive in 10 folds of 16 against 9. The strategy does not beat the market it trades in.** Two caveats belong in the same paragraph: buy-and-hold is **fully invested** while the strategy is capped at `max_gross_exposure = 0.50`, so part of the gap is exposure rather than skill; and buy-and-hold carries **no stop and no target**, which is what the phrase means and which the report should say. (1d) **The return decomposes into exposure, timing and friction, and the report must show the split rather than the headline** (GB-25). Average gross exposure is **16.0%** of equity, time-weighted (peak 51%); the market returned +7.37% a fold; so **+92.6 bps a fold was earned by being in the market at all**, **−37.0 bps was given back to timing** and **−10.7 bps paid in costs**, summing to the +44.8 bps reported. Regressing fold return on market return gives β = **0.141** (t = 5.1, R² 0.65) against a measured exposure of 0.160 — the exposure explains the returns — with an alpha of **−0.59% a fold (t = −1.4)**. **The timing term is not distinguishable from zero** (mean −37 bps, sd 198, t = −0.75, positive in 7 folds of 16), and the report must say so rather than claiming the system times badly. **A correlation that was in this row and is now withdrawn, which the report should say plainly.** An earlier version claimed the timing residual is negatively correlated with the market return (r = −0.47) and called it independent corroboration of GB-20's forecast-side down-bias. On 16 observations that r carries a 95% CI of [−0.783, +0.034] — it includes zero — and **re-measured after GB-27b shifted the fold grid by one week it is r = −0.0025, CI [−0.498, +0.494]**. It did not weaken; it vanished. The corroboration claim goes with it: what remains is GB-20's forecast-side measurement alone, which stands on its own terms. **The decomposition itself must be reported with intervals**, because every level moved with the grid while the structure did not: earned by exposure **+98.1 bps** [−11.2, +207.4], given back to timing **−51.8** [−113.5, +9.9], costs **−6.1** [−9.1, −3.1], actual **+40.2** [−74.5, +154.9]. **β = 0.0919** (t = 3.29) against a measured average exposure of **0.0892**, R² 0.436, **α = −0.31% per fold and not distinguishable from zero** (t = −0.65). **Re-measured at three fold-grid anchors on 2026-08-18, and the claim splits in two:**

| anchor | first test | measured exposure | β | β t | β 95% CI | α per fold | α t | R² |
|---|---|---|---|---|---|---|---|---|
| +0 | 2022-07-06 | 0.0892 | 0.0919 | 3.29 | [+0.032, +0.152] | −0.31% | −0.65 | 0.436 |
| +21 sessions | 2022-08-08 | 0.0934 | 0.0507 | 1.51 | [−0.021, +0.122] | −0.21% | −0.44 | 0.141 |
| +42 sessions | 2022-06-06 | 0.1390 | 0.0848 | 2.28 | [+0.005, +0.165] | +0.11% | +0.20 | 0.271 |

**What survives all three anchors:** the measured exposure lies inside β's 95% interval at every one, and **α is never distinguishable from zero** (t no larger than 0.65 in magnitude, and it changes sign between anchors). *The exposure explains the returns and there is no alpha* is therefore reportable. **What does not survive:** β's own significance. At the +21 anchor its interval includes zero (t = 1.51), so the sentence "β is significantly positive" holds at two anchors of three and must not be written as though it held at all of them. The point estimates are not stable either — exposure 0.089 to 0.139, β 0.051 to 0.092, R² 0.14 to 0.44 — so **every level in this paragraph must be reported as a range across anchors, never as a single number**. The pre-GB-27b grid gave β 0.141 against exposure 0.160, consistent with the same reading. (1e-i) **The double use of validation, and the direction of its bias** — thresholds are calibrated on forecasts already slightly overfit to validation, so on test they are **miscalibrated rather than inflated**. The double use **degrades** test performance rather than flattering it, which makes every test figure a **lower bound** with respect to this flaw. State the direction, not just the existence: a reader who is told only that validation was used twice will discount the numbers, and would be wrong to. (1f) **The argument for property testing, in one sentence** (GB-21, GB-22): a Hypothesis property test written to check position caps caught a **float64 range bug in a different function a day later** — `floor(x * 1e9) / 1e9` is not the floor once the scaled value leaves float64's contiguous integer range, which it does at a $0.01 price and a $46M account. No example-based test in this project would have generated that pair. Report it as what it is: the test found a defect its author was not looking for. (1e) **The risk layer is correct, enforced and currently inert** (GB-21) — state this plainly rather than letting a reader assume it constrains anything. `max_position_pct` 0.10 × 5 symbols = 0.50 = `max_gross_exposure`, so at this universe size the gross cap is **numerically redundant** with the per-position cap; over 16 folds it bound **exactly once**, removing two trades. Bound the statement rather than dismissing the layer: it would begin to bind under a **larger universe**, a **higher `max_position_pct`**, or a **`top_k` above 5**. (1k) **A design principle the report should state, because it generalises past this dashboard** (GB-34/35/36, 2026-08-18). **Colour that carries information must be separable from colour that carries interface**, and the way to tell whether it is, is to ask what survives in greyscale. This project has one data colour family — a *ramp*, dark for slow and light for fast — and a ramp cannot encode a **sign** without inventing a second data colour, which would put chrome and data back in the same visual channel. So **sign is carried by geometry and a glyph**: a positive contribution is a filled bar right of the zero rule, a negative one is hollow and left, and every signed figure is prefixed ▲ or ▼. **The consequence is the test:** the chart reads correctly with the colour removed, which is precisely the check of whether colour was doing work that nothing else was doing. **Its enforcement is named and mechanical** — `tests/dashboard/test_app.py::test_no_traffic_light_colours_anywhere` asserts that no green/red pair appears in the stylesheet or in any generated chart, so the principle cannot decay into a comment. The same discipline governs the ramp's assignment: it runs by **channel look-back**, slowest darkest, because a ramp encodes a quantity and one assigned in config order would look identical and mean nothing. (1i) **The rule operating, twice, with different outcomes** (GB-26, re-dated 2026-08-20). **Required, and it must not be softened or presented as a limitation.** The paragraph to write is this one: *"The deployment of 18-20 August stood aside; validation found no candidate with a positive Sharpe. After the y-scaling correction of 20 August the same fold calibrates a firing band on a validation Sharpe of +0.483 over 8 trades. Both are the rule operating. The first is the stronger demonstration and the second is the current state."* **The rule is not tightened**, and the report must say why: changing a pre-registered decision rule because we dislike what it selected is the exact failure this project exists to avoid, and it would be indefensible in a defence. The rule also demonstrably discriminates — **5 of 16 folds stand aside** — and a rule admitting noise freely would never stand aside at all. What changes is the reporting: **wherever the deployed band appears it carries its selection context** — the validation Sharpe, the trade count it rests on, and that it is a **grid maximum over fifteen candidates**. Under pure noise the maximum of fifteen candidates on eight trades is positive almost surely, so a band on 8 trades and a band on 80 are not the same claim and must not read alike. The dashboard masthead carries it (`dashboard.app.BandContext`), the gate log carries it, and every mention here carries it. The background: `smoke_offline --prepare-live` trained on the most recent complete fold — fold 16 of 16, train 2024-01-08 to 2025-12-29, validation 2026-01-06 to 2026-03-27 — and the fold **stands aside**: the best candidate on the threshold grid scored a validation Sharpe of **−3.38** over 7 trades. **3 of 16 folds stand aside, the most recent among them, and the system deployed on 2026-08-18 does not trade.** The report must state this as an outcome of the design rather than as a shortfall of it: GB-20's ruling — a fold whose best candidate has no positive validation Sharpe trades nothing — was made for a backtest fold and is here operating in production. **A bot that abstains when validation finds nothing is a bot whose decisions mean something**, and refusing to invent a band because a demonstration would look better is the discipline of this project in one behaviour. (1j) **Cancellation: what the attribution says about why the direction result is what it is** (GB-30, GB-32, measured live 2026-08-18). The surviving fraction of the gross channel view — the net contribution over the gross one — was measured on the first live cycle at **15.4% (AMZN), 20.6% (NVDA), 23.1% (MSFT) and 49.2% (GOOGL)**, four of five symbols below half, with only AAPL above it. **High cancellation is the signature of a linear model whose weights are fitting noise that mostly offsets**, and unlike most such diagnoses it is directly measurable and falls out of the explanation layer for free — no extra experiment, no perturbation, no second model. The report must present it **beside the direction result**, as its mechanism: the model is information-free on direction, and the attribution shows why, in a quantity the attribution produces anyway. GB-49's `cancellation` column carries it across arms and anchors. **One caveat that must travel with the column** (measured 20 Aug 2026): FITS's cancellation is **exactly 1.0000** in every cell, and that is arithmetic rather than evidence — it is univariate, so one channel contributes and `|Σc| / Σ|c|` is 1 by construction. **Cancellation is only meaningful for a multi-channel model.** Reported without that, FITS reads as the arm whose explanation never cancels, which would be a claim about the model rather than about the channel count. (1h) **A test that cannot fail is not a test — the exactness tautology** (GB-30, found 2026-08-18). **Required as a methodological finding, not a footnote.** Spec §4.4 states two exactness properties: (3) `explain(x).forecast_total == predict(x).sum()`, and (4) `Σ per_channel == forecast_total`. Both were asserted over 1,000 random windows from GB-11 onward, and the suite was green throughout. But every forecaster built its attribution with `forecast_total = sum(per_channel.values())`, so **property 4 compared a number against itself** — `x == x`, for every model, for a week. One of the two checks the project's central claim rests on **was not capable of failing**, and nothing in a passing suite could say so. Taking the total from `predict` instead makes the two quantities independent, and the residual becomes a real measurement rather than a guaranteed zero: **5.005e-08 worst over 1,000 windows, a 200× margin** under the 1e-5 tolerance — a number the project did not previously have. The report must state the general point plainly, because it is the argument for this project's whole approach to testing: **"we assert exactness" and "we assert exactness in a way capable of failing" are different claims**, and only the second is evidence. The same defect in the same class was found the same day and is worth naming beside it: `PersistenceForecaster.explain` hand-wrote `dict.fromkeys(channels, 0.0)`, so **the baseline every result is reported against constructed its own answer** and never went through the arithmetic the check was checking. It now reports *no terms* per channel and the zeros fall out of the shared code path. The lesson generalises past attribution: a property is only as strong as the independence of the two things it compares, and a self-reported invariant is not one. (1g) **Fold-grid sensitivity — a required methods subsection, and a methodological contribution in its own right** (ruled 2026-08-18). Most work in this area reports a single walk-forward grid. This project reports every headline claim at **three fold-grid anchors** offset by roughly one third of `step_months` (GB-49), because it has a worked example of why that matters: a one-week shift in fold boundaries took the timing-versus-market correlation from **r = −0.47 to r = −0.0025**. It did not weaken — it vanished, and nothing about the single-anchor measurement said it might. The subsection must state the protocol, state the cost (about 93 seconds for three anchors), and give the worked example. It must also report **what the sensitivity found**, which was not uniform: direction accuracy against the always-long bar held at all three anchors in sign, magnitude and fold count; β's *point estimate* tracked measured exposure at all three, but β's *significance* did not, failing at one anchor of three; and α was indistinguishable from zero at all three. **The honest general claim is that grid sensitivity separates claims by kind, not that it validates them.** (1t) **Report claims by what they survived, not by what they measured** (GB-49, 20 Aug 2026). **Required as a table, because the classification is the finding.** Every claim the grid can speak to, against both required tests: *no arm beats the always-long bar* — **−4.25 to −6.48 points at every anchor for every arm, Wilcoxon p ≤ 0.021 in 7 of 9 cells** — **survives both**; *FITS beats DLinear* (+0.96 / +0.83 / +0.88 points, sign-stable but **p = 0.34 / 0.61 / 0.49** and **+0.18 points under white noise**) — **survives neither**; *the wavelets help DLinear* (**+0.30 → +1.24 → +1.36** points across anchors, significant at **exactly one anchor of three**, and **+0.57 points present under white noise**) — **survives neither**, and it is the `r = −0.47` pattern repeating; *any arm extracts signal at all* (**−0.47 / −0.74 / +0.31** points against its own noise twin, p = 0.94 / 0.86 / 0.94) — **survives neither**. **Only the negative claim survives both tests.** (1u) **The controls are validated by what they do to the bar, and this must be reported before the claims that rest on them.** The always-long edge over chance is **+5.60 points on real data, +4.18 on shuffled returns, +0.64 on white noise**. Shuffling preserves the return distribution including its positive mean, so drift survives it; zero-mean noise has none, so the bar collapses to chance. **The two controls separate order from drift**, and the one that kills the bar is the one that should — which is what licenses reading "the arms sit at their noise level" as a fact about the arms rather than a property of the controls. (1s) **THE CENTRAL RESULT, and its phrasing is fixed** (GB-49, 20 Aug 2026). **Required verbatim, and the shorter version must not survive anywhere in the report.** **These models extract nothing from this market that they do not also extract from white noise.** That is a statement about the models, not about the market: over the same period, a constant always-long rule beats chance by **5.6 points**, so the structure exists and is reachable — just not by explicit or implicit frequency decomposition at this horizon, on this universe, with these channels. Put the five numbers on the same footing where the claim is made — always-long real **0.5564**; DLinear real **0.4959** against white noise **0.5006**; FITS real **0.5055** against noise **0.5025** — because the comparison only lands when the bar and the arms and the controls are read together. **The models sit at their noise level; the market does not.** Two things make this stronger than a baseline comparison and both must be said. First, it is a **control**, so it rules out the reading that the metric is insensitive: a rule that *does* beat chance by 5.6 points is measured on the same windows by the same function. Second, always-long's class choice is **fixed from the training splits** and only its rate is measured on test (§7.2), so the bar is not a number borrowed from the future. And state the negative precisely rather than generally: the failure is of **explicit and implicit frequency decomposition at `H=4`, on five US large caps, with `C0_base`/`C2_hybrid`** — not of frequency methods, not of these architectures, and not of the market. (1q) **"When a measurement is suspiciously stable, the next test is a control, not a larger sweep"** (20 Aug 2026). **Required verbatim**, with what happened around it. The FITS phase advance was written up as a finding at midday and **withdrawn twelve hours later**. In between, the sweep that was asked for was run — 16 folds × 3 fold-grid anchors, 48 independently trained models — and it **confirmed** the claim: **+1.9582 days, sd 0.0203, positive in 48 of 48 cells**, more stable than the `r = −0.47` correlation that had vanished under a one-week grid shift. What killed it was a control: fitted on **white noise** the same band gives **+1.9309 days**. The report must state the process fact as well as the result, because it is the part that generalises: **it was caught only because the instruction was "do not trust it yet" rather than "confirm it"** — a supervisor's scepticism aimed at a specific number, before any evidence existed that it was wrong. A larger sweep would have made the claim look stronger. The heuristic that now sits in `CLAUDE.md` §3 came out of it: on financial data a **coefficient of variation near 1%** across independently trained models — this one was **1.04%** — is evidence about the machine, not the market. (1r) **FITS spends most of its capacity on a resampling operator, and that is a result about the architecture as published** (measured 20 Aug 2026). **Required as a result, not as a caveat on the figure.** If **86% of the learned frequency response is reproduced by a model trained on white noise** — gain curves correlating at **+0.9485**, gain tracking `|η·k − round(η·k)|` at **Spearman −0.9427, p = 1.8e-11** — then the complex layer is spending most of its 1,150 effective parameters on a **deterministic resampling operator that could be written in closed form**, leaving roughly **14%** for anything about the data. Two things make the critique precise rather than rhetorical. **(a) It is configuration-specific, and the cost is a function of `η = 1 + H/L`.** Measured over the `COF = 24` retained bins at `L = 120`: at the configured **`H = 4`, `η = 31/30`, exactly 1 of 24 bins aligns and the mean misalignment is 0.2833 — the worst point in the table**; at `H = 60`, `η = 3/2`, **12 of 24** align; at `H = 120`, `η = 2`, **all 24** align and the cost is **exactly zero**. **Correcting the natural framing:** the cost is *not* monotone in `H/L`. Very short horizons are fine — `H = 1` gives mean misalignment 0.0958 — because `η − 1` is then too small to displace even the highest retained bin by much. The worst case is the middle, and **the project's configuration sits in it**. (b) The implication is **parked, not chased** — see `IDEAS_PARKED.md`: initialise the complex layer with the analytic resampling operator and learn only the residual, so the capacity goes to the data instead of to the geometry. **And keep the inversion, because it is the sentence that makes the finding land:** the *trough* is stable across all 48 models (median 8.00 days, range 8.00–9.23) and the *peak* is not (median 40, range 17.1–120) — **the geometry is the stable half and the data is the unstable half, which is the opposite of the natural guess.** (1o) **FITS allocates 4.17% of its parameters to a row that cannot learn, and so does the paper's** (GB-44, measured 20 Aug 2026). **Required, and it must be framed as an observation about the architecture as published, not about this implementation.** RIN subtracts each window's own mean; the rFFT's bin 0 *is* that mean; so row 0 of the complex weight matrix multiplies zero on every forward pass, takes no gradient, never leaves its initial value and cannot change a forecast. Any implementation faithful to the paper has it. **Measured three independent ways**, which is what makes it a finding rather than an inference: the *weight* (`max |w|` on row 0 = **2.1e-11** after a full fold's training, against **0.89** on the rest), the *input* (bin 0 after RIN = **3.6e-15**, against bin 1's **13.1**), and a *perturbation* (moving every row-0 weight by `1+1j` changes the forecast by **exactly 0.0**; the same on row 1 moves it by **6.2**). The count is therefore **1,200 allocated / 1,150 effective**, and both numbers appear wherever the parameter count appears — §6.1, §6.5 and the comparison against DLinear's 4,800, which is 4.17× rather than 4×. The row is **kept**: §6.2's low-pass keeps the first `COF` bins and bin 0 is one of them, so removing it is an architecture change bought for a cosmetic 4%. (1p) **A defect class the project names and the report should state, because it generalises well past this codebase** (GB-44, 20 Aug 2026). `smoke_offline` — the one command that produces every number the study reports — held a **third** copy of the model registry: `--model`'s argparse `choices` literal, plus a `run()` signature defaulting to the *string* `"dlinear"` rather than to `model.active`. So the frozen-registry design of §4.3 worked exactly as intended, `VALID_MODELS` agreed with `ALL_FORECASTERS`, and setting `model.active: fits` still changed the deployed checkpoint while changing **nothing** about the results. **Had GB-49 run before this was found, the FITS arm of the study would have been DLinear and the report would have compared a model to itself** — and every consistency check in the project would have passed, because the two arms really were consistent. The generalisation: *a switch that reaches one half of a system is worse than one that reaches neither, because the two halves then agree with themselves and disagree with each other silently.* And the reason it survived: the test that should have caught it carried the comment **"GB-41 adds it, not GB-24"**, and GB-41 did not. **A note saying a later task will do something is not a mechanism** — now a rule in `CLAUDE.md` §3, beside the staging rule, for the same reason that one is there. **And report this verbatim, because the six minutes prove the point better than the rule states it:** the rule went into `CLAUDE.md` at 12:00 on 20 Aug 2026 and the fourth instance of the same defect shipped at 12:06 — `matplotlib` declared in `pyproject.toml` and absent from `requirements.lock`, which is what CI installs from, so the local suite was green on 1,001 tests and run #17 was red. **A rule is not a mechanism.** The general form now sits above it in `CLAUDE.md`: *any fact stored in two places needs a mechanism making them equal, or they will diverge* — four instances, and **three of the four were caught by a test written for something else**. (1n) **Every MAE comparison in this study is conditional on the learning rate, and the direction column is not** (measured 20 Aug 2026, 16 folds × 2 models × 5 arms). **Required, and it must be stated where the MAE tables are, not in a limitations section.** Three things were measured and all three change what an MAE number means. (a) **Optimiser resolution.** With the per-symbol scaler at `lr: 0.001`, DLinear's median learned weight is **5.4e-4 against an Adam step of 1.0e-3** and **71% of its weights sit below the step**; the scaled-versus-unscaled MAE ordering **reverses** as the rate falls (0 of 16 folds at `1e-3`, 16 of 16 at `1e-5`). (b) **Supervision units.** FITS's weights are 19× the step and the rate changes nothing for it; its inflated MAE is a **B+F unit mismatch** — with `X` standardised and `y` in raw log returns the backcast term is **67× the forecast term**, the objective is 98.5% backcast, and the forecast comes out **4.8× too large**. Standardising `y` returns the ratio to 0.69× and the MAE to the raw arm's value. (c) **Flatness.** Spearman(MAE, flatness) = **+0.811** and **+0.668**; Spearman(MAE, direction) = **+0.009** and **−0.128**. The best-MAE arm forecasts at **19% of the truth's magnitude**. So MAE across these arms is close to a monotone function of how flat the forecast is and carries almost no information about accuracy — §7.3's ban, arriving as a measurement. **What survives all of it:** direction accuracy moves by **0.0041 across all five FITS arms** (0.5056–0.5098). An optimiser mismatch can hide a real result; it cannot manufacture a null one, so the near-chance direction headline is **not** conditional on any of this and must not be reported as though it were. (1m) **Why FITS has a learned complex layer at all — an architectural explanation the source paper does not give** (GB-41/GB-42, measured 20 Aug 2026). The paper describes the layer as learning an amplitude gain and a phase shift per retained frequency, which is what complex multiplication *means* but not what the layer is *for*. The reason falls out of the grids: **bin `k` of a length-`L` transform is the frequency `k/L`, and the same physical frequency sits at bin `k·η` of a length-`L+H` transform**, `η = (L+H)/L`. When `η·k` is not an integer that frequency has no output bin to land on, and **no parameter-free extension can reconstruct even a pure sinusoid the input represents exactly**. Measured on a 3.0-amplitude 12-day cycle over `L = 120`: at the configured `H = 4`, `η·k = 10.333` and the frequency-preserving extension is out by **4.97 — the amplitude itself**; at `H = 12`, `η·k = 11` exactly and the error is **2.2e-14**. So the learned layer's first job is **interpolation between two grids that do not line up**, and gain and phase are simply how a linear map expresses that. This also bounds how much work the layer is doing: `η = 31/30` at this configuration, so `η·k` is an integer only where `30 | k`, and across the `COF = 24` retained bins **that is bin 0 alone** — every non-trivial frequency the model keeps is off-grid, on every window. **The sentence to write is: the layer is not interpolating occasionally, it is interpolating always.** Report it as an architectural finding rather than a footnote: it explains the component better than the source paper's own framing does — "a complex linear layer learns an amplitude gain and a phase shift per frequency" says what the layer *is* and never what it is *for* — and it is why GB-42's amplitude test states its grid as part of what it tests. (1aa) **A principle applied to the person who wrote it is a principle that has been tested** (23 Aug 2026). **Required as one line, because it is the only evidence that the rules in `CLAUDE.md` are load-bearing rather than decorative.** Three times in one day the *only something that runs is a mechanism* principle caught its own author: a **docstring** that said `test_the_same_grid_twice_gives_the_same_numbers` ran one condition while its body ran the whole grid; a **window count** written into a docstring that was 25% wrong and had to become a test; and a **README step skipped while auditing that very README** — the editable install, omitted from the GB-59 clean-clone run, which would have audited the author's install procedure instead of the documented one. Each was found by applying the rule rather than by remembering it, and the third became `scripts/setup.ps1`: **a documented procedure a human must execute correctly is a note; a script that runs and verifies is a mechanism.** (1ab) **THE REPRODUCIBILITY RESULT, WITH ITS BOUNDARY NAMED** (GB-59, 23 Aug 2026). **Required with the limit stated in the same breath as the claim — a claim that states its own boundary is worth more than one that quietly overreaches.** A clone taken **from the remote**, a fresh venv, a fresh interpreter and a fresh directory regenerate `results.csv` with **every one of 877 rows bit-identical in every result column** — direction, MAE, flatness, RMSE, Sharpe, total return, max drawdown, cancellation, trade and window counts, and `model_config_hash`. Only `seconds` (wall time, not a result) and `config_hash` differed, and the second was a defect of ours, not of the pipeline. **The deployed artefact reproduces too, and that is the more surprising half**: `prepare_live` in the clean clone returned a band identical to the deployed one at `lower = 0.026076278765685856`, `val_sharpe = 0.48258203344932654`, same fold and same validation window, with `model.json` **byte-identical** (sha256 `afde41c2…`). That path runs through training, per-fold validation calibration **and** a threshold grid search — three stages that look stochastic and are not. **THE BOUNDARY:** both runs were on **one machine**. Process, venv, interpreter install and directory varied; **CPU, core count, instruction set and OS did not**, and BLAS reduction order can differ with any of them. **Cross-architecture reproduction is untested**, and it is exactly the case a supervisor reproducing this will be in. **Report it as untested rather than assumed.** **BLAS is now pinned to one thread by default** (`--blas-threads`, default 1), on a counterbalanced measurement rather than the first one: unpinned **40.7 s / 36.8 s** against pinned **40.9 s / 37.7 s**, a between-condition difference of **0.5 s** against a within-condition spread of **3.2–3.9 s** — not distinguishable from run-to-run noise. The first attempt (70.8 s against 62.3 s) was an artefact of order and machine load and is withdrawn. **And a fact about the study that belongs beside the parameter counts:** the likely reason pinning is free is that **these models are too small to parallelise** — DLinear holds 4,800 reals and FITS 1,200, and a `(120, 4)` by `(120,)` matmul does not saturate fourteen cores. **Defaulting it was verified not to change the numbers**: a pinned run reproduces an unpinned one bit-for-bit, so the committed `results.csv` is still regenerated by the default command. **What it must NOT be reported as.** Pinning removes **one** source of cross-machine divergence, not all of them: different instruction sets dispatch different BLAS kernels, different BLAS builds order operations differently, and transcendentals come from the platform's `libm`. Single-threaded on two different CPUs can still differ in the last bits. **Defaulting this does not make the claim architecture-independent, and writing that it does would be the overreach this claim exists to avoid.** (1ac) **A reasoning pattern worth naming, because it recurs: an instrument that only reports negatives can still deliver a positive answer** (23 Aug 2026). **A hash cannot say what changed, but it can say what cannot have.** `config_hash` covers everything and `model_config_hash` covers only what shapes a weight, so a file whose full hash has moved while its model hash has not **proves the numbers stand**: anything model-shaping would necessarily have moved both. The difference is confined to sections that cannot reach a model, without anyone inspecting the old configuration or even having it. The same shape appears in the null controls — destroying the signal cannot create an effect, so an effect that survives destruction is proven not to depend on the signal — and in the dead DC row, where a weight that multiplies zero cannot move a forecast however large it grows. **Report it as a pattern rather than three coincidences**: when a direct measurement is unavailable, ask what the instrument makes impossible and read the answer off that. (1z) **A mechanism installed for one reason covered a failure nobody had connected to it, and that is the strongest argument for installing them** (GB-59, 23 Aug 2026). **Required beside the *only something that runs is a mechanism* principle, because it is the half of the argument that is hard to make in advance.** `data_cache/*.parquet` was un-ignored on **20 Aug** for one narrow reason: CI reported **26 skipped** tests, and a mechanism that can be skipped by a flag is a mechanism only when the flag is off. Nobody was thinking about reproducibility. Three days later the **GB-59 clean-clone audit** — scheduled for the last week of the project and pulled forward — depends on it completely: a clone without the snapshot cannot run the cache-gated suite, cannot run the grid, and cannot regenerate `results.csv` at all, so the audit would have reported nothing and the finding would have arrived in October. **The cost of the mechanism was 800 KB and the benefit was an audit becoming possible**, and no one could have written the second reason down at the time. Report it as the concrete case for the principle rather than as a coincidence. **The same reading applies to the README's long-path row**: a troubleshooting remedy verified by *moving* an existing clone and never executed from a fresh one is a **note**, not a mechanism, and GB-59 is the first thing that has ever run it. (1x) **THE HEADLINE CLAIM, AND THE SENTENCE ABOUT THE INSTRUMENT THAT MUST TRAVEL WITH IT** (GB-50/GB-51, 23 Aug 2026). **Required verbatim.** *No arm beats the always-long bar. The effect is negative in **9 of 9 anchor-arm cells**, between **−4.3 and −6.5 points**, and holds under both null controls. It is significant raw at **8 of 9** and survives Holm at **3 of 9** within the pre-specified nine-test family. The claim the study makes is the **absence of a positive effect**, which the sign stability and the effect size support directly; the stronger claim that every arm is **significantly worse** is reported as not surviving correction, and is not relied on.* **And state plainly why the correction is the wrong instrument here, because a reader who knows statistics will look for that sentence and its absence would cost more than the correction does:** Holm-Bonferroni guards against **false positives**. Applied to a claim of **absence** it makes it *easier* to conclude that nothing was found — which is not a protection but a **bias toward this study's own conclusion**. It is reported because declining to report it would be worse, and it is not the evidence the claim rests on. §1.4 asked whether an edge exists; the answer is that none does, and that answer is carried by nine negative cells and a five-point gap, not by a p-value. **Two claims must never be conflated here:** *(a) no arm beats always-long* — a claim of absence, which is what this study reports — and *(b) every arm is significantly worse* — a positive claim of an effect, which does not survive correction and **must not appear anywhere in the report**. (1y) **THE ONLY STATISTICALLY ROBUST RESULTS IN THE STUDY MEASURE FLATNESS, AND THE WINNER IS THE MODEL THAT FORECASTS NOTHING** (GB-51, 23 Aug 2026). **Required, and it is a finding about the metric rather than about the models — which is what makes it strong.** Of **79 tests** in the real-only family, **11 survive Holm at 0.05, and all eleven are MAE against persistence with the trained model worse**. Persistence forecasts **zero**, which is **maximally flat**; across the COF sweep **Spearman(MAE, flatness) = +1.00** over the four cutoffs, MAE falling 0.0167 → 0.0155 as flatness falls 0.4187 → 0.1560, and the best-MAE cutoff forecasts at **15.6% of the truth's magnitude**. So the only results in this study that survive a family-wise correction are **the flatness artefact in significance clothing**. This is the **third and cleanest** demonstration of why §7.3 bans MAE as a headline — after the 16-fold scaler comparison and the Spearman +0.811/+0.668 measurement — and the first in which the ban is vindicated by the significance machinery itself. (1w) **THREE FORCES ACT ON `COF` AND ONLY ONE OF THEM SATURATES — the framing for the sweep, recorded before it ran** (GB-50, written 23 Aug 2026, so that whatever the sweep returns cannot be read as a post-hoc explanation of it). **Required as one claim, not as three findings that agree.** **(i) Grid misalignment is one curve in two parameterisations, and this supersedes reading either table as a peak.** `frac(η·k) = frac(k·H/L)`, so the fractional part has a period of **`L/H` bins** and the only quantity that matters is **how many alignment cycles the retained bins span**. Across `COF` at `H = 4` (period 30): **0.07 cycles → 0.0167, 0.20 → 0.0833, 0.40 → 0.1833, 0.80 → 0.2833, 1.00 → 0.2500, 2.00 → 0.2500**. Across `H` at `COF = 24` (period `120/H`): **0.20 cycles → 0.0958 (`H=1`), 0.80 → 0.2833 (`H=4`), 2.40 → 0.2333 (`H=12`), 12.0 → 0.2500 (`H=60`)**. Both **rise while the retained bins cover less than one cycle and then plateau at 0.25**, the mean distance of a uniform fractional part to the nearest integer, exactly 0.2500 at whole cycles. `H = 120` giving **0.0000** is **not the far end of a curve** — it is the single exact escape from it, `η = 2`. **The apparent peak at `COF` 24 is a partial-cycle sampling artefact**: 24 of 30 bins covers `k/30 ∈ [0, 0.77]` and over-weights the far half. So write *every low cutoff is saturated and the deployed one is among them*, **never** *the deployed cutoff landed on the worst cell* — the weaker claim is the true one, and the stronger one invites a reader to check the arithmetic and find it marginal. **(ii) The three forces.** Retaining more frequencies buys **more information**; it buys **worse reconstruction of each**, because higher `k` puts `η·k` further from an integer; and it buys **worse statistical determinacy**, because the complex layer grows as `COF · ceil(η·COF)`. Measured against **2,505 training windows × `H = 4` = 10,020 equations** per fold (fold 1, five symbols, 501 train timestamps each): equations per parameter runs **cutoff 2: 7,440 reals at 1.35×; cutoff 5: 1,200 at 8.35×; cutoff 10: 312 at 32.12×; cutoff 20: 84 at 119.29×**. **`COF` is therefore not only *how much to keep* — it is *how well what you kept can be reconstructed* and *how well what you kept can be estimated*, and all three pull against each other.** **(iii) The asymmetry is what makes the sweep a test rather than a formality.** Information retained **rises** and does not saturate; reconstruction cost **falls and saturates at 0.25**; determinacy **falls and does not saturate**. Because the geometric cost is capped and the statistical one is not, **the geometry alone predicts no middle optimum** and the force most likely to produce one is statistical. **(iv) The prediction, recorded before the result.** If there is an optimum, **it is not at cutoff 2**, and the reason will be **statistical rather than geometric** — at 1.35× the layer is very nearly underdetermined. **If cutoff 2 does win, that is evidence against the determinacy story and is worth as much as the opposite.** The sweep must report which of the three, if any, moved anything: a sweep finding nothing at every cutoff is a stronger statement than not having swept, and it is the specific question §1.4 set out to ask. **(v) MEASURED 23 Aug 2026 — no optimum, and weak evidence AGAINST the determinacy story rather than an absence of evidence.** Direction against each cutoff's own white-noise twin is **−0.71 / +0.31 / −0.79 / −0.93 points** at cutoffs 2 / 5 / 10 / 20, and against the always-long bar **−5.93 / −5.09 / −5.95 / −6.08**: every cutoff at or below its noise level, and all four five to six points below the bar. **The determinacy term predicts visibly higher fold-to-fold variance at 1.35× and there is none**: per-fold sd of direction is **0.0592 / 0.0627 / 0.0630 / 0.0546** across determinacy ratios spanning **1.35× to 119×**, which is flat. **An absence of the predicted variance is information, not silence.** The reading everything else in this study supports: **if the model learns nothing, there is nothing to overfit, so determinacy does not bite.** Record it that way rather than as untested. (1v) **The overnight protection residual, stated as measured rather than as unlikely** (ruled 23 Aug 2026). **Required, and the phrasing matters as much as the fact.** The protective legs are `TimeInForce.DAY` and stay that way: Alpaca refuses GTC on a fractional quantity (`42210000`, measured), so GTC costs whole-share rounding, and `shares_for` is shared with the backtester — the change would land on **every arm in the study**, not on the executor. Flattening at the close is refused for a stronger reason: the backtest holds overnight, so flattening live would trade a different strategy from the one being reported, closing a protection gap by opening a **parity** gap. **State the residual in full: for the interval between a session open and a successful re-arm, and for any session the loop does not run at all, an open position carries no broker-side protection.** Not "brief" and not "unlikely" — the broker expires the legs at the close, so the position is naked from the open until the loop arms it, and through any morning the process is not running. The defence is rule 2 of the GB-26 policy — re-arm before any entry, first thing on any start — which closes the first case for every session the loop runs and **does nothing for a session it misses**. Report it beside the fractional-order finding it descends from, not in a limitations appendix, and note that no position has yet been held overnight by the loop, so this is **latent rather than realised**. (2) **Parameter counts**, both measured, with the §6.1 note's two caveats stated in full: the papers' framing of a heavyweight baseline against a tiny challenger does not hold here — this project's DLinear is smaller than the paper's FITS — and the comparison is not like-for-like, since DLinear consumes five channels and FITS is univariate by design | E12 | B | 5 | Every table carries a persistence delta; the appendix records the 2018 volume event; the results chapter states the fold cap's regime-coverage trade-off, the total-return convention, the initialisation finding with its measurements, and both measured parameter counts with both caveats |
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

- Python 3.12+, `ruff` + `black` defaults, type hints everywhere, `pytest`
- Every feature function is **pure**: DataFrame in → DataFrame out, no hidden state, no I/O
- No config values hardcoded in modules — always via `config/loader.py`
- Any change to feature or model code requires the full causality and contract suites to pass
- Small commits, imperative messages, one module per PR where possible
- Every task commit message begins with its Jira key: `GB-42: fix irFFT amplitude scaling`
- **When torn between clever and simple: simple.** This codebase is demonstrated live to a supervisor.

---

*GlassBox Trader v3 — the name is literal: a trading bot you can see into, now perceiving the market across the frequency spectrum.*
