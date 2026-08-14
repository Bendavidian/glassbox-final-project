# Sprint 1 Prompts — Foundations

**15–28 August 2026 · 12 tasks · you build all of them**

One task per Claude Code session. Plan mode for anything above 2 SP.
Read every diff before accepting.

---

### Session start — paste this first, every session

```
Read CLAUDE.md, then run git log --oneline -15 and read PROGRESS.md.
Tell me which task we are on, what is left in this sprint, and whether we are
on schedule against SOLO_BUILD_PLAN.md. Do not write any code yet.
```

### Task close — paste this at the end, every task

```
Run pytest, ruff check . and black --check . and report the results.
Then update PROGRESS.md with this task's row: date and one line on what was built.
Then give me the git commit command with the task ID as the message prefix.
Do not commit yourself — I will review the diff first.
```

---

## Day 1 · GB-1 · Repository scaffold, tooling and CI · 2 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-1.

Create the glassbox package with exactly the directory layout in spec §3.4.
Every package directory gets an __init__.py. Every module named in §3.4 gets created
as an empty file with a module-level docstring stating its single responsibility per
the layer table in §3.2 — do not implement anything yet.

Also create:
- pyproject.toml with ruff and black at defaults, Python 3.11 target, and these
  dependencies: pandas, numpy, pyyaml, torch, yfinance, alpaca-py, PyWavelets,
  scipy, streamlit, pandas-market-calendars, python-dotenv, pyarrow.
  Dev extras: pytest, hypothesis, import-linter.
- tests/ mirroring the package structure, with a conftest.py
- .gitignore covering data_cache/, .env, checkpoints/, __pycache__, .pytest_cache
- .github/workflows/ci.yml running ruff check, black --check and pytest on push

Acceptance: CI passes on an empty suite. The tree matches §3.4 exactly. Nothing
secret or cached is tracked by git.

Structure only in this task. No logic.
```

---

## Day 2 · GB-2 · Configuration contract and typed loader · 2 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-2.

Write config/settings.yaml exactly as given in spec §5 — every key, same names,
same values, same nesting. Do not add keys. Do not rename keys.

Then implement config/loader.py:
- Parse into frozen dataclasses mirroring the structure (MetaConfig, DataConfig,
  WindowConfig, WaveletConfig, FitsConfig, ChannelConfig, ModelConfig, SignalConfig,
  RiskConfig, BacktestConfig, WalkforwardConfig, LiveConfig, and a top-level Config)
- Validate on load and fail loud. At minimum:
    * universe non-empty, unique, uppercase tickers
    * window.input_len > window.horizon
    * wavelet.rolling_window >= 2 ** wavelet.levels
    * fits.cutoff_period_days >= 2 and < window.input_len
    * fits.supervision in {"F", "B+F"}
    * every *_pct field in (0, 1]
    * risk.max_position_pct <= risk.max_gross_exposure
    * model.active in {"persistence", "dlinear", "fits"}
    * channels.active names an existing key in channels
    * all walkforward month values positive integers
- Every failure raises ValueError naming the exact field path, e.g.
  "risk.max_position_pct must be in (0, 1], got 1.4"
- Expose load_config(path) -> Config and config_hash(cfg) -> str, a stable SHA-256
  of the resolved configuration
- Alpaca credentials load from .env via python-dotenv and are exposed here only

Write tests/test_config.py: a valid load returns the right types; each invalid
variant raises ValueError with the field name in the message; the hash is stable
and changes when any value changes.

Acceptance: all pass. No module anywhere reads settings.yaml or os.environ directly.
```

---

## Day 3 · GB-3 · Frozen contracts and layer discipline · 3 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-3.

Implement contracts/schemas.py with WindowBatch, Forecast, Attribution, Signal and
DecisionRecord — copy the definitions from spec §4.2 exactly, including docstrings.
All frozen dataclasses. Add __post_init__ shape validation:
- WindowBatch: X.ndim == 3, y.ndim == 2, X.shape[0] == y.shape[0] == len(timestamps),
  X.shape[2] == len(channels), both float32
- Forecast: path.ndim == 1
- Attribution: per_lag.ndim == 2

Implement contracts/protocols.py with the Forecaster Protocol exactly as in §4.3,
using typing.Protocol with @runtime_checkable.

Then enforce the layer rule from §3.1: data flows strictly upward. Configure
import-linter in pyproject.toml with the layer order config < contracts < data <
features < model < engine < explain < dashboard, and add tests/test_layers.py that
runs the contract and fails on violation.

Verify it has teeth: temporarily add an upward import, confirm the test fails, then
remove it. Report what you observed.

Acceptance: schemas importable and frozen; shape validation rejects malformed input;
the layer test fails on a deliberate violation and passes once removed.
```

---

## Day 4 · GB-4 · Historical data loader with parquet cache · 3 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-4.

Implement data/historical.py:
- load_history(symbols, cfg, force_refresh=False) -> dict[str, pd.DataFrame]
- Fetch daily OHLCV via yfinance from cfg.data.start to today, auto_adjust=True
- Index normalised to tz-aware UTC DatetimeIndex, sorted, no duplicates
- Columns lowercase snake_case: open, high, low, close, volume
- Add log_return = ln(close / close.shift(1)); the first row stays NaN, never filled
- Cache each symbol to {cfg.data.cache_dir}/{symbol}.parquet. A second call with the
  cache present must not touch the network.
- Log which symbols came from cache and which were fetched

Write tests/test_historical.py using a small committed CSV fixture so the suite never
needs the network:
- Cached load returns data identical to the fetched load
- Index is tz-aware UTC, monotonic, no duplicates
- log_return matches a hand-computed value
- Loading succeeds from cache when yfinance is monkeypatched to raise

Acceptance: all five symbols cached from 2016 to today; tests green offline.
```

---

## Day 5 · GB-5 · Data quality report · 2 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-5.

Implement data/quality.py:
- check_quality(frame, symbol, cfg) -> QualityReport (frozen dataclass) covering:
    * missing bars against the NYSE calendar (pandas_market_calendars)
    * NaN count per column
    * suspected splits or discontinuities: any single-bar absolute log return
      above 0.25, listed with date and magnitude
    * first date, last date, total bar count
    * duplicate index entries
- report_all(frames, cfg) prints a readable table and writes
  data_cache/quality_report.json

Write tests against a synthetic frame with deliberately injected gaps, NaNs and a
2:1 split, asserting each is detected.

Acceptance: report generated for all five symbols. March 2020 volatility and any
real split dates are flagged rather than silently passing. Commit the JSON report as
a data-provenance artifact.

Keep this lean — it is a 2-point task. Do not build a validation framework.
```

---

## Day 6 · GB-6 · Alpaca account and secrets · 2 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-6.

I have created an Alpaca paper trading account. Implement credential handling:
- ALPACA_API_KEY and ALPACA_SECRET_KEY load from a local .env via python-dotenv
- They are read ONLY through config/loader.py — no module touches os.environ directly
- Commit .env.example with placeholders; .env stays git-ignored
- Fail loud with a clear message if either variable is missing

Write scripts/smoke_alpaca.py connecting and printing account equity, buying power,
and whether the account is a paper account. It must refuse to run if it detects a
live-trading endpoint — this is a hard safety check, not a warning.

Acceptance: the smoke script prints live account equity. Then run
`git log -p | grep -i "alpaca_secret"` and report the result — confirm no credential
ever entered version control history.
```

---

## Day 7 · Review day

No new tasks. Do this instead:

```
Review everything committed so far. For each module, tell me:
1. Anything that violates a rule in CLAUDE.md §4
2. Anything hardcoded that should come from config
3. Any function that is not pure but should be
4. Any test that would still pass if the implementation were wrong

Do not fix anything yet. Give me the list first.
```

Then fix what it finds, and read the full diff of Sprint 1 to date yourself.

---

## Day 8 · GB-7 · Live market data client · 3 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-7.

Implement data/live.py fetching recent daily bars from the Alpaca Data API.

The hard requirement: it returns the IDENTICAL schema to data/historical.py — same
column names, same dtypes, same tz-aware UTC index, including log_return. The feature
layer must not be able to tell the two sources apart. This is what makes the train/live
parity test in GB-27 possible, and that test is the structural guarantee of the whole
system.

Write tests/test_live_schema.py asserting historical and live loaders return identical
dtypes and column sets, using a recorded API response fixture so the test needs no
network.

Separately write scripts/compare_sources.py comparing a same-day bar from both sources
and reporting the difference. Document the observed tolerance in the module docstring.

Acceptance: schema equality test passes; same-day bar matches yfinance within the
documented tolerance.
```

---

## Day 9 · GB-8 · Technical indicators · 3 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-8.

Implement features/indicators.py with four pure functions. Each takes a DataFrame
with the canonical OHLCV + log_return columns and returns a pd.Series named with the
canonical channel name from spec §4.1:

- rsi14(df)     -> 14-period Wilder RSI on close
- vol_z(df)     -> rolling 20-day z-score of volume
- mom10(df)     -> close / close.shift(10) - 1
- ma_dist20(df) -> (close - close.rolling(20).mean()) / close.rolling(20).mean()

Hard requirements:
- Pure: no I/O, no global state, no mutation of the input
- Every window TRAILING. Never center=True. This is the causality rule from
  CLAUDE.md §4 and it is the most important property in this module.
- Warm-up rows are NaN. Do not backfill. Do not use min_periods to fake early values.
- Window lengths are module-level constants with a comment noting they are indicator
  definitions rather than tunable configuration

Write tests/test_indicators.py:
- Each indicator matches a hand-computed fixture of at least 30 bars to 1e-6.
  Compute the expected values independently and write them as literals — do NOT
  generate the expected value with the same code under test.
- Each function leaves its input unmodified
- Warm-up length is exactly as expected for each indicator

Acceptance: all tests pass; all four functions pure and trailing-only.
```

---

## Days 10–12 · GB-9 · The window builder · 5 SP

> **Use Plan mode. Read the plan carefully before approving.** This is the keystone
> module — every module downstream inherits its correctness, and its contract is
> expensive to change once the model layer depends on it.

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-9.
Read spec §3.3 before starting. Use Plan mode: describe your approach and wait for
my approval before writing code.

This is the ONLY place in the system where a model input window is assembled. Both
offline training and the live loop will call it. Design accordingly.

Implement features/builder.py:

    def build_feature_frame(bars: pd.DataFrame, cfg: Config) -> pd.DataFrame
        Assemble all canonical channels for the active channel config.
        Returns a FeatureFrame: UTC DatetimeIndex, one column per channel,
        no NaNs after warm-up trimming.

    def fit_stats(frame: pd.DataFrame, cfg: Config) -> ChannelStats
        Compute normalisation statistics. Called by the caller, on the training
        split only.

    def build_windows(
        frame: pd.DataFrame,
        cfg: Config,
        symbol: str,
        stats: ChannelStats | None = None,
        as_of: pd.Timestamp | None = None,
    ) -> WindowBatch

Requirements:
- Channel order is exactly cfg.channels[cfg.channels.active], deterministic
- Target y is the future path of the `close_logret` channel, H steps ahead
- Normalisation statistics are PASSED IN, never fitted inside build_windows.
  Fitting inside would leak test data into training — this is the single most
  likely place for leakage to enter the system.
- The window ending at timestamp t uses only rows <= t. No exceptions.
- as_of=None yields all valid windows (training path). as_of=<timestamp> yields
  exactly one window ending at that timestamp (live path). Both share the same
  slicing code — do not write two implementations, not even temporarily.
- float32 throughout

Write tests/test_builder.py:
- Shapes correct for both C0_base and C2_hybrid
- Channel order matches config order exactly
- as_of returns byte-identical output to the batch path for that timestamp
  (np.array_equal)
- The last row of X for a window ending at t equals the frame row at t
- y for a window ending at t equals log returns at t+1 .. t+H
- Passing stats fitted on a different range changes X, proving stats are external

Acceptance: all of the above. If anything about the live path is unclear, ask before
implementing.
```

---

## Day 13 · GB-10 · Causality harness · 3 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-10.

Build a REUSABLE harness in tests/causality.py. It is applied again to the wavelet
module in GB-48, so design it as a utility rather than a one-off test.

    def assert_causal(fn, frame, split_at=None, perturbation=1.5):
        """Assert fn uses no information after each timestamp.

        Compute fn(frame). Create a copy with all rows after split_at multiplied
        by `perturbation`, recompute, and assert every value at or before split_at
        is unchanged. Repeat at several split points.
        """

Then in tests/test_no_lookahead.py apply it to:
- all four indicators from GB-8
- build_feature_frame from GB-9
- build_windows from GB-9 (assert X is unchanged for windows ending at or before
  the split point)

Test at 25%, 50% and 75% through the series.

CRITICAL — prove the harness has teeth. Write a deliberately leaky function, e.g.
close.rolling(20, center=True).mean(), and assert with pytest.raises that
assert_causal rejects it. A causality test that cannot fail is worthless.

Acceptance: every real function passes; the leaky function fails. Report the split
points used and the result for each function.
```

---

## Day 14 · GB-11 + GB-12 · Baseline, contract test, docs · 5 SP

**GB-11 first:**

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-11.

PART 1 — model/persistence.py

    class PersistenceForecaster:
        """Tomorrow equals today. Predicts zero log return.

        Every result in this project is reported as a delta against this model.
        Without it there is no way to tell whether a model learned anything or
        merely echoed the last price.
        """

Implements the full Forecaster protocol from spec §4.3. predict returns
np.zeros((B, H), dtype=np.float32). explain returns an Attribution with every channel
contributing 0.0, a zero per_lag matrix, per_frequency=None, gain_phase=None,
forecast_total=0.0. fit is a no-op. save/load are trivial. Around 30 lines.

PART 2 — tests/test_forecaster_contract.py

Parameterise over a registry ALL_FORECASTERS so adding DLinear (GB-13) and FITS
(GB-41) later requires only registering the class, with NO changes to this file.
Design for that now.

Assert all six properties from spec §4.4:
1. predict output shape exactly (B, H), dtype float32
2. predict deterministic given a fixed seed — call twice, assert identical
3. explain(x).forecast_total == predict(x[None])[0].sum() within 1e-5
4. sum(explain(x).per_channel.values()) == forecast_total within 1e-5
5. save -> load -> predict reproduces byte-identical output
6. perturbing x[t+1:] does not change predict on the window ending at t

Acceptance: contract test passes for PersistenceForecaster, structured so future
models plug in by registration alone.
```

**Then GB-12:**

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-12.

Write:
- README.md — what the project is in three sentences, prerequisites, install steps,
  credential setup, how to run the offline smoke path, how to run the tests, and the
  repository map. Written so someone with no prior context goes from clone to running.
- ARCHITECTURE.md — the layer stack, the responsibilities table, the frozen contracts,
  and why builder.py is the keystone. Cross-reference the spec rather than duplicating it.

CLAUDE.md, PROGRESS.md, DECISIONS.md, IDEAS_PARKED.md and SOLO_BUILD_PLAN.md already
exist. Do not overwrite them.

Acceptance: following README on a clean machine works end to end. Every step you get
stuck on is a bug in the README, not in your memory.
```

---

## Sprint 1 finish line — 28 August

- [ ] `pytest` green in full
- [ ] CI green on main
- [ ] Causality harness passes for all indicators and the builder, and **rejects the
      deliberately leaky function**
- [ ] Contract test passes for `PersistenceForecaster`
- [ ] All five symbols cached and quality-checked
- [ ] Alpaca connectivity confirmed, no credential in git history
- [ ] `PROGRESS.md` shows twelve completed rows

**If GB-9 or GB-10 is shaky, stop and fix before Sprint 2.** A leak introduced here
surfaces in week six as an impressive backtest that turns out to be worthless — and
by then you will not have time to find it.
