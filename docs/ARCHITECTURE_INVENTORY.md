# Architectural Inventory — external components and the system as built

> **What this is.** A read-only inventory of the repository, taken 24 September 2026, to
> feed two deliverables: the architecture deck the department requires, and the three-minute
> video. It exists because of the submission rule that drives it — *"every use of an external
> component must be explicitly disclosed, or it counts as plagiarism"* — so nothing here is
> invented and nothing is omitted.
>
> **Method.** Every claim is cited to a file, a file and line, or a `DECISIONS.md` /
> `PROGRESS.md` entry. Nothing was edited, staged or committed to produce it, and **`pytest`
> was not run** — which is why §12 reports counted test *definitions* and says so, rather than
> a collected-item count it could not have obtained honestly.
>
> Paths are repo-root-relative throughout, in the convention `DECISIONS.md`, `PROGRESS.md`
> and `docs/GB55_ARCHITECTURE.md` already use.

---

# PART A — EXTERNAL COMPONENTS

## A.1 Runtime dependencies

13 declared in `pyproject.toml:17-31`. `requirements.lock` pins **96** distributions — the
18 declared plus **78 transitive**. CI installs the lock, then the package with `--no-deps`
(`.github/workflows/ci.yml`).

| Package | Pinned | What **this** project uses it for | Cite |
|---|---|---|---|
| `pandas` | 3.0.5 | Every bar frame, feature frame and results table. The `FeatureFrame` convention (UTC `DatetimeIndex`, one column per channel) is a pandas contract. | `contracts/schemas.py`, `features/builder.py` |
| `numpy` | 2.5.2 | All inference is pure numpy — weights are held as arrays after fitting, so inference carries no global torch state. This is what makes determinism true by construction. | `glassbox/model/ltsf.py:33-36` |
| `pyyaml` | 6.0.3 | Parses `settings.yaml`. Imported in exactly one place; a test forbids any other module reading settings or `os.environ`. | `glassbox/config/loader.py:32`, guard at `tests/config/test_config.py:549` |
| `torch` | 2.13.0 | **Training only**, imported *inside* `fit` in each forecaster and in the study's thread pinning. Never at module scope. | `glassbox/model/ltsf.py:200`, `glassbox/model/fits.py:291`, `glassbox/model/wits.py:337`, `glassbox/experiments/study.py:525` |
| `yfinance` | 1.6.0 | The historical price source, `auto_adjust=True`, one symbol per call, cached to parquet. | `glassbox/data/historical.py:41,132` |
| `alpaca-py` | 0.44.0 | Two clients: `StockHistoricalDataClient` for live daily bars, `TradingClient` for paper orders/positions/account. | `glassbox/data/live.py:77-79`, `glassbox/engine/executor.py` |
| `PyWavelets` | 1.9.0 | (a) the causal rolling DWT feature channels `wav_a1..a3`; (b) WITS's analysis/synthesis matrices, *measured* from `pywt` by pushing basis vectors through it rather than re-derived; (c) the config validator asks `pywt.dwt_max_level` rather than reimplementing the level ceiling. | `glassbox/features/wavelets.py:90`, `glassbox/model/wits.py:124,164`, `glassbox/config/loader.py:500` |
| `scipy` | 1.18.0 | Exactly two call sites: paired Wilcoxon signed-rank, and the Spearman that pairs MAE with flatness. | `glassbox/experiments/stats.py:221`, `glassbox/experiments/report.py:320` |
| `streamlit` | 1.61.1 | Hosts the console. Imported inside `main()`. Charts are **hand-built SVG strings**, not a chart library — so geometry is testable without a browser. | `glassbox/dashboard/app.py:3602` |
| `pandas-market-calendars` | 5.4.0 | The exchange calendar: walk-forward fold boundaries, the data-quality gap report, and the live loop's session window (a holiday closes the loop; a half-day closes it early). | `glassbox/backtest/walkforward.py:26`, `glassbox/data/quality.py:24` |
| `python-dotenv` | 1.2.2 | Loads `.env` for Alpaca credentials, in the config layer only. | `glassbox/config/loader.py:33` |
| `pyarrow` | 24.0.0 | Parquet engine for the committed data snapshot (`data_cache/*.parquet`, 20 symbols). Used via `to_parquet`/`read_parquet`, never imported by name. | `glassbox/data/historical.py:108,215` |
| `matplotlib` | 3.11.1 | Three report rasters only: the frequency-response figure, the per-arm direction boxplots, the COF curve. Imported *inside* the render functions. | `scripts/frequency_response.py:38`, `glassbox/experiments/report.py:412,475` |

**Provenance of the set.** `DECISIONS.md:4959` records the original twelve; `DECISIONS.md:1436`
records matplotlib as a later addition with its argument. CLAUDE.md §4 requires a
`DECISIONS.md` line per dependency, and every one of the 13 has one.

> One stale claim to fix before it reaches a slide: the matplotlib entry's *Consequence* says
> it is imported "only in `scripts/`". It is now also imported inside two functions in
> `glassbox/experiments/report.py`. The substance of the claim (no module gains a rendering
> dependency at import time) still holds; the wording no longer does.

## A.2 Dev / test dependencies

5 declared, `pyproject.toml:35-41`.

| Package | Pinned | Use here | Cite |
|---|---|---|---|
| `pytest` | 9.1.1 | The suite. `pythonpath = [".", "tests"]` so **bare `pytest`** works — the only invocation CLAUDE.md licenses the word "green" for. | `pyproject.toml:60-68` |
| `hypothesis` | 6.165.5 | Property tests in exactly 3 files: the risk layer's "no config produces an over-limit position", `room_detail`, and per-channel attribution exactness. | `tests/engine/test_risk.py:20`, `tests/engine/test_room_detail.py:23`, `tests/explain/test_channel.py:20` |
| `import-linter` | 2.13 | Machine-enforces the layer rule. Two contracts, run both by the tool and from inside a test. | `pyproject.toml:80-127`, `tests/contracts/test_layers.py:33` |
| `ruff` | 0.16.3 | Lint, `target-version = py312`, `reference/` excluded. | `pyproject.toml:45-47` |
| `black` | 26.5.1 | Format check, `reference/` excluded. | `pyproject.toml:49-56` |

`.hypothesis/` exists in the tree (example database). Transitive-only names worth knowing if
anyone asks why they are in the lock: `grimp` (import-linter's graph),
`exchange_calendars` / `pyluach` / `korean_lunar_calendar` (market-calendars),
`altair` / `pydeck` / `narwhals` / `watchdog` (streamlit),
`curl_cffi` / `peewee` / `multitasking` / `beautifulsoup4` (yfinance),
`sympy` / `networkx` / `filelock` / `fsspec` (torch).

**The guard that keeps the two lists honest:** `test_every_declared_dependency_is_pinned_in_the_lock`
(`tests/test_scaffold.py:152`). It exists because on 20 Aug 2026 matplotlib was added to
`pyproject.toml` alone — green locally, red in CI run #17. It asserts the declared set ⊆ the
lock, deliberately **not** equality, because the lock holds the transitive closure.

## A.3 Vendored, copied, adapted

### `reference/algotrading_project-main/` — 2.7 MB, read-only, never imported

An existing LTSF-Linear trading project brought in for study. Its terms are written down in
`reference/REFERENCE_AUDIT.md`, which pre-dates the code and names what may be taken, what is
broken, and what must not be touched.

Three independent mechanisms keep it outside the build, none of them a convention:

- `include = ["glassbox*"]`, `exclude = ["reference*", ...]` — excluded from packaging
  (`pyproject.toml:43`)
- excluded from ruff and black
- `norecursedirs` for pytest; and the import-linter contract is **rooted at `glassbox`**, so as
  `tests/contracts/test_layers.py:52` puts it, *"`reference/` is outside the contract by
  construction, not by an exclusion list."*

**The audit's approved-reuse table, verbatim in intent:**

| Source | Target | What was taken |
|---|---|---|
| `models/DLinear.py` | GB-13 | The `series_decomp`/`moving_avg` blocks and the two linear heads |
| `evaluation.py` | GB-19 | Six return/risk formulas, judged correct |
| `backtesting.py`, `models_backtest.py` | GB-18 | Event-loop **structure** and the `Position`/`ActionType`/`PositionType` enums — structure only |
| `utils/tools.py` | GB-15 | `EarlyStopping`, `adjust_learning_rate` |
| `data_provider/data_loader.py` | GB-9 | **Reference only, do not port** |

**What was refused, and why — this is the strongest disclosure material in the repository:**

- `utils/metrics.py` is unusable: `SHARP()` returns a hardcoded `12`; `SHARP2`/`SHARP5` are
  each defined twice with the second shadowing the first; one variant contains
  `pred = true + 0.01`, overwriting the prediction with ground truth. `backtest/metrics.py`
  says so in its own header (`glassbox/backtest/metrics.py:4`) and was written from scratch.
- `strategies.py` inverts **all four** stop/target comparisons. The audit lists each correct
  form; `glassbox/backtest/engine.py:38` cites the audit and the four cases are each covered
  by an explicit test.
- The single train/val/test split was rejected for walk-forward.
- One thing explicitly marked *not* a bug: `moving_avg` pads both ends, making the trend a
  centred moving average. That is correct **inside the input window** `[t-L+1, t]`, all of
  which precedes the forecast. Centring remains forbidden in `features/indicators.py` and
  `features/wavelets.py`, which compute along the full series.

### DLinear — a **reimplementation from the paper**, with four departures, not a port

`glassbox/model/ltsf.py:6`: *"Follows LTSF-Linear (Zeng et al., AAAI 2023). Three deliberate
departures from the reference implementation, each recorded in DECISIONS.md"* — the header
names three, `DECISIONS.md:4127` is titled "GB-13: DLinear departs from the reference in four
places, one of them measured."

1. **Per-channel weights, not shared.** Reference maps every channel through one `nn.Linear`.
   Here each channel gets its own matrix per component, so attribution is a regrouping of terms
   already computed rather than a reconstruction.
2. **No intercept.** `nn.Linear` carries a bias; this does not — a bias would break
   `sum(per_channel.values()) == forecast` with no honest channel to charge it to.
3. **Zero initialisation, not the reference's.** `nn.Linear`'s default bound assumes fan-in
   `L`; this architecture sums `C×2` such maps, so the bound is wrong by construction. The
   measured table is in the docstring at `glassbox/model/ltsf.py:414`.
4. **Torch fits, numpy predicts.**

The one thing carried over unchanged is `DECOMP_KERNEL = 25`, flagged as *"from the LTSF-Linear
paper. An architecture definition rather than a tuning knob"* (`glassbox/model/ltsf.py:55`).

### FITS — an **original implementation from the paper's description**, no reference code

Cited in the spec as *FITS (Xu, Zeng & Xu, ICLR 2024, arXiv:2307.03756)*
(`docs/GLASSBOX_PROJECT_SPEC.md:709`). There is **no FITS source in `reference/`** — the
reference project carries only `DLinear.py`, `Linear.py`, `NLinear.py`. `model/fits.py`
(655 lines) is written from the spec's §6.2 pipeline description. The header claims no lineage
and credits no source implementation. Two places where the implementation reasons about the
published architecture rather than copying it: the dead row is retained because *"the source
paper's architecture carries the same dead row"* (`glassbox/model/fits.py:270`), and spec
§6.3's amplitude trap is a correction the project derived and then found the spec had stated
wrongly (`DECISIONS.md:1737`).

### WITS — **original**, with a recorded pre-registered prediction

`glassbox/model/wits.py` (753 lines), GB-66. No paper, no reference. It exists to test whether
FITS's ~86% data-independence is a property of *extending a global basis* or of FITS
specifically. The prediction was written into `GLASSBOX_PHASE2_EXPANSION.md` before a line was
written, and `scripts/operator_null_control.py:1-10` quotes it back.

### Everything else

No file in `glassbox/` carries a "copied from", "adapted from" or "courtesy of" header.
Persistence, the builder, the causality harness, the sweep harness, the executor, the
reconciler, the live loop, the console and the explain layer are original.

## A.4 External services

| Service | Provides | Called from | Constraints encoded in code |
|---|---|---|---|
| **yfinance** (Yahoo, unauthenticated) | Historical daily OHLCV, `auto_adjust=True` → **total returns, not price returns** | `glassbox/data/historical.py:132` | Per-symbol, never bulk. Cache is authoritative; a stale file is returned unchanged unless `force_refresh=True`, so the study runs against a fixed snapshot. |
| **Alpaca Data API** | Live daily bars in a schema identical to yfinance's | `glassbox/data/live.py` | `Adjustment.ALL` (RAW closes AAPL at 499.75 where adjusted closes 121.08); feed hardcoded to **SIP** with no fallback (IEX measured wrong by up to **193 bps**, against a 1 bp tolerance). **No cache** — a cached bar in a trading decision is a stale price. |
| **Alpaca Trading API (paper only)** | Order submission, positions, account, order history | `glassbox/engine/executor.py`, `glassbox/engine/reconcile.py` | `require_paper_endpoint` refuses any non-paper base URL **before any call is made** (`glassbox/config/loader.py:802`). Both clients share a 45 s read timeout applied to the SDK's *session* (`glassbox/data/http.py`). |
| **GitHub REST API** | CI run status for a commit | `scripts/ci_status.py` | Dev tooling only, not in `glassbox/`. Uses the git credential helper; exit codes 0/1/2/3 so it works in a shell condition. |

Credentials: `.env` (gitignored), template at `.env.example`. `tests/test_no_secrets.py` scans
every tracked file for credential-shaped strings and asserts `.env` is untracked — written
after a real Alpaca paper key reached a local commit on 14 Aug 2026.

**The account's SIP entitlement is narrower than "SIP works"**, and that boundary is
documented: historical daily bars are served; `get_stock_latest_bar` /
`get_stock_latest_quote` are refused with *"subscription does not permit querying recent SIP
data"*. Anything reaching for a live quote rather than a completed bar meets that wall
(`glassbox/data/live.py:36-40`).

## A.5 Code derived from documentation, tutorials or forum answers

**No comment anywhere in `glassbox/`, `tests/` or `scripts/` records code taken from a
tutorial, blog post or forum answer.** The whole family of markers was searched. What exists
instead is a distinct and stronger pattern: constants and behaviours **measured against the
live API and dated**, with the vendor's verbatim error string quoted.

- `{"code":42210000,"message":"fractional orders must be simple orders"}` —
  `glassbox/engine/executor.py:15`, with the full submitted/response table at
  `DECISIONS.md:3432`
- `cost basis must be >= minimal amount of order 1` → `MIN_ORDER_NOTIONAL`, *"verified against
  the live paper API in GB-22 and not from the documentation"* (`glassbox/engine/risk.py:49`)
- `insufficient qty available` — a working sell order holds the whole position, measured
  against a **97.38-share** position so it is not fractional-specific (`DECISIONS.md:5075`)
- The Alpaca SDK's `RESTClient._one_request` calling `self._session.request(...)` bare, with no
  `timeout` parameter — read from the SDK source, and `UnboundedClientError` raised if a future
  SDK renames the private `_session` (`glassbox/data/http.py:5-8`)

One documentation-derived choice is worth naming in the deck because it is the opposite of
copying: `pywt.dwt_max_level` is **asked** rather than reimplemented, with the reason stated —
*"a second copy of a fact pywt already holds, and this project has six instances of what that
costs"* (`glassbox/config/loader.py:495-500`).

## A.6 AI-assisted development — the accurate account

This is recorded in four independent places. It is unusually complete; it should be disclosed
exactly as it stands.

**1. Commit co-authorship. 130 of 132 commits carry a `Co-Authored-By: Claude` trailer.**

| Trailer | Commits |
|---|---|
| `Claude Opus 5 <noreply@anthropic.com>` | 88 |
| `Claude Opus 5 (1M context) <noreply@anthropic.com>` | 42 |
| *(none)* | 2 |

All 132 commits are authored by `bendavidian <bendben13@gmail.com>`. The two without a trailer
are the two document-only commits that bootstrap the project: `918a5f2` "GB-0: project charter,
spec, build plan and reference code" (14 Aug) and `c738131` "GB-0: Phase 2 expansion plan, GB-61
to GB-66" (24 Aug). History runs 14 Aug 2026 → 24 Aug 2026 (commit dates).

**2. `CLAUDE.md`** — checked into the repository root, ~250 lines. It is the operating
contract: six rules, a definition of done, non-negotiable engineering rules (causality, purity,
the keystone, exactness, staging discipline), and nine numbered "instances" of one failure
class, each with the date it cost time and what it cost. It is read at the start of every
session.

**3. `prompts/`** — `SPRINT_1.md` (447 lines), `SPRINT_2.md` (281), `SPRINT_3.md` (410),
`SPRINT_4.md` (434). The actual session prompts, one task per session, with the session-start
and task-close boilerplate. This is the working record of *how* the assistance was directed.

**4. `START_HERE.md`** — states the division explicitly: *"For the university — you submit
these, Claude Code never reads them"* (line 11) versus *"For the repository — Claude Code reads
these on every session"* (line 20), and describes `CLAUDE.md` as *"what stops Claude from
inventing."*

**5. The method itself is in the record**, and it is the most honest thing here.
`DECISIONS.md` (5,425 lines, 121 entries) repeatedly records where the assistant was **wrong
and was corrected by the supervisor or by measurement** — the parity floor derivation corrected
by Ben (`DECISIONS.md:3269`), the scaler hypothesis supplied by the supervisor and then
confirmed for DLinear and *refuted* for FITS (`DECISIONS.md:1498`), the overnight-residual
description narrowed by Ben's correction (`DECISIONS.md:3432`), and an entry that names the
pattern: *"That is the fifth time a first description in this family has been narrowed by a
measurement."*

One line worth putting on the disclosure slide as-is, because it is verifiable from the tree:
**`glassbox/explain/` imports no perturbation-attribution library, and a test asserts it**
(`tests/explain/test_channel.py:397`). Attribution here is algebra, not an LLM's or a library's
opinion. And `docs/GB55_ARCHITECTURE.md:717` already carries the section heading *"It will not
take an LLM's word for anything."*

---

# PART B — THE SYSTEM AS BUILT

## B.7 Module tree — 51 files, 22,450 lines

### L0 — config & contracts (1,433)

| Module | Lines | Responsibility |
|---|---|---|
| `glassbox/config/loader.py` | 820 | The only module that reads `settings.yaml` or `os.environ`. Typed config, fail-loud validation naming the offending field, the split config hash, the paper-endpoint guard. |
| `glassbox/config/settings.yaml` | 92 | 13 sections. Every runtime value in the system originates here. |
| `glassbox/contracts/schemas.py` | 553 | The frozen schemas: `WindowBatch`, `ChannelStats`, `FitProvenance`, `Trade`, `Forecast`, `Attribution`, `Signal`, `DecisionRecord`. |
| `glassbox/contracts/protocols.py` | 58 | The `Forecaster` protocol — five members, `runtime_checkable`. |
| `glassbox/faults.py` | 118 | What to do when the outside world does not answer: timeout, refusal, wrong answer. |

### L1 — data (748)

| Module | Lines | Responsibility |
|---|---|---|
| `glassbox/data/historical.py` | 227 | yfinance → parquet cache, tz-aware UTC, `log_return`. The cache is authoritative. |
| `glassbox/data/live.py` | 251 | Alpaca daily bars in the identical schema, via the same `normalise_bars`. |
| `glassbox/data/quality.py` | 140 | Per-symbol gaps, NaNs, large moves, duplicates. Reports facts; judges nothing. |
| `glassbox/data/http.py` | 130 | The 45 s read timeout every Alpaca client shares. |

### L2 — features (873)

| Module | Lines | Responsibility |
|---|---|---|
| `glassbox/features/builder.py` | 475 | **THE KEYSTONE.** The only place a model input window is assembled. |
| `glassbox/features/wavelets.py` | 210 | Causal rolling DWT channels `wav_a1..a3`. |
| `glassbox/features/indicators.py` | 188 | `rsi14`, `vol_z`, `mom10`, `ma_dist20`. Trailing windows only. |

### L3 — models (2,270)

| Module | Lines | Responsibility |
|---|---|---|
| `glassbox/model/wits.py` | 753 | WITS — causal DWT, band retention, per-band linear map, inverse DWT. |
| `glassbox/model/fits.py` | 655 | FITS — RIN, rFFT, low-pass, complex linear layer, irFFT. |
| `glassbox/model/train.py` | 555 | One training run: stats on the training split only, fitting, checkpointing, loss curve. |
| `glassbox/model/ltsf.py` | 493 | DLinear — trend/remainder decomposition, per-channel linear maps. |
| `glassbox/model/predict.py` | 220 | Batched and single-window inference. |
| `glassbox/model/persistence.py` | 154 | The zero-return baseline every result is a delta against. |
| `glassbox/model/__init__.py` | 80 | `ALL_FORECASTERS` — the one registry, holding factories not classes. |
| `glassbox/model/history.py` | 80 | `EpochLoss`. Its own module because both ends need it and neither may import the other. |

### L4/L5 — engine (1,730)

| Module | Lines | Responsibility |
|---|---|---|
| `glassbox/engine/executor.py` | 700 | Order submission to Alpaca paper. Auto and Co-Pilot. Never makes a decision. |
| `glassbox/engine/risk.py` | 405 | Sizing, gross-exposure cap, stop attachment. Property-tested. |
| `glassbox/engine/reconcile.py` | 317 | The broker is the truth; local state follows it, loudly. |
| `glassbox/engine/signal.py` | 235 | Forecast path → `enter_long` / `hold` / `exit`. No numeric threshold in the source. |
| `glassbox/engine/rank.py` | 73 | Top-`k` by trend strength, alphabetical tie-break. |

### L6 — explain (1,002)

| Module | Lines | Responsibility |
|---|---|---|
| `glassbox/explain/narrate.py` | 498 | Attribution → prose, English and Hebrew. |
| `glassbox/explain/spectral.py` | 274 | Per-frequency contribution, gain and phase shift; the learned frequency response. |
| `glassbox/explain/channel.py` | 230 | Exact per-channel attribution. |

### L7 — console (4,308)

| Module | Lines | Responsibility |
|---|---|---|
| `glassbox/dashboard/app.py` | 4107 | The Streamlit trading-desk console. Computes nothing; every region carries a source pill. |
| `glassbox/dashboard/tokens.py` | 201 | Colour tokens — its own module so the contrast test can walk every one without knowing their names. |

### X — harness, above the live path (3,215)

| Module | Lines | Responsibility |
|---|---|---|
| `glassbox/experiments/study.py` | 1031 | The grid runner → `results.csv`. |
| `glassbox/experiments/report.py` | 687 | `results.csv` → `report.md` + two figures. Regenerates from the CSV alone. |
| `glassbox/experiments/exposure.py` | 358 | Does the gross-exposure cap ever bind, and in how many folds? |
| `glassbox/experiments/stats.py` | 339 | Paired Wilcoxon of every arm against **its own** reference. |
| `glassbox/backtest/metrics.py` | 569 | Every forecast and trading metric, each with its reference. |
| `glassbox/backtest/engine.py` | 555 | Event-driven backtester. No vectorised shortcut anywhere. |
| `glassbox/backtest/calibrate.py` | 242 | Per-fold threshold calibration on the **validation** split. |
| `glassbox/backtest/walkforward.py` | 204 | Train 24 / val 3 / test 3, roll 3. Cross-validation never used. |

### Top level (4,292)

| Module | Lines | Responsibility |
|---|---|---|
| `glassbox/live_loop.py` | 2768 | The live decision loop. Ten steps per bar, protection before entries. |
| `glassbox/smoke_offline.py` | 951 | The offline end-to-end command, and the three prepare paths. |
| `glassbox/records.py` | 810 | Decision records on disk and the live trade log, in `backtest`'s own `Trade` type. |
| `glassbox/replay.py` | 435 | Recorded-day playback, fully offline. |
| `glassbox/live_lock.py` | 279 | One loop per state directory, enforced rather than remembered. |

## B.8 Layer boundaries, as the import linter enforces them

Two contracts, both in `pyproject.toml:76-127`, `root_package = "glassbox"`.

**Contract 1 — "Layers: data flows strictly upward"**, highest first:

```
glassbox.experiments
glassbox.backtest
glassbox.dashboard
glassbox.explain
glassbox.engine
glassbox.model
glassbox.features
glassbox.data
glassbox.contracts
glassbox.config
```

The comment states the non-obvious placement: *"backtest and experiments sit above the whole
live path, not inside it. Spec 3.1 calls them a cross-cutting harness rather than a layer, and
the constraint that matters is that the harness is never imported by the live path. Placing
them on top makes that structural: nothing below can reach up to them, while they remain free
to drive models, features and data, which a walk-forward run must do."*

**Contract 2 — "The live path never imports the validation harness"** (type: `forbidden`):

```
source_modules    = glassbox.live_loop, glassbox.replay, glassbox.records
forbidden_modules = glassbox.backtest, glassbox.experiments
```

The comment on `records.py` is the interesting one: *"`records.py` holds the live trade log, so
it is exactly the module that would be tempted to import `backtest.engine.Trade` — which is why
that type moved to `contracts/schemas.py` and why this contract now names it."*

`smoke_offline.py` is **deliberately absent** from contract 2 — it is the harness's own entry
point and must import `backtest`.

Enforcement is doubled: the tool runs in CI, **and** `tests/contracts/test_layers.py` runs it
in-process (`test_no_layer_violations`), asserts both contract names are still configured
(`test_contract_is_configured`), and asserts the root package (`test_root_package_is_glassbox`)
— so deleting a contract from the config turns the suite red rather than silently reducing
coverage.

## B.9 Frozen interfaces

**`Forecaster` protocol** — `glassbox/contracts/protocols.py`, spec §4.3.

```python
name: str;  input_len: int;  horizon: int;  fitted: FitProvenance | None
fit(batch: WindowBatch, val: WindowBatch | None = None) -> None
predict(X: np.ndarray) -> np.ndarray          # (B,L,C) f32 → (B,H) f32
explain(x, channels) -> Attribution           # (L,C) single window → exact
save(path) -> None;  classmethod load(path) -> Forecaster
```

Implemented by **four** forecasters, registered in `glassbox/model/__init__.py` as
`ALL_FORECASTERS`: `persistence`, `dlinear`, `fits`, `wits`. The registry holds **factories,
not classes**, because constructors legitimately differ. *"A model is not integrated until it
passes tests/test_forecaster_contract.py unchanged"* — and the docstring records that GB-41
added FITS with no edit to the contract test.

**Frozen schemas** — `glassbox/contracts/schemas.py`, all `@dataclass(frozen=True)`:

| Type | Line | Note |
|---|---|---|
| `WindowBatch` | 57 | Validates ndim, dtype `float32`, and alignment of targets/timestamps/channels/symbols. `concat` refuses mismatched batches. |
| `ChannelStats` | 195 | Per-symbol normalisation statistics. |
| `FitProvenance` | 276 | What a model was fitted on. `from_batch(...)`, and `fit` **must** set it even when training is a no-op. |
| `Trade` | 349 | **One type for backtest and live** — moved here in GB-29 so `records.py` need not import `backtest`. |
| `Forecast` | 396 | |
| `Attribution` | 409 | `kw_only=True`. Holds the exactness assertion. `per_lag` optional. |
| `Signal` | 507 | |
| `DecisionRecord` | 518 | The full decision. |

`EXACTNESS_TOLERANCE = 1e-5` is defined once at `glassbox/contracts/schemas.py:41` and
**imported** by the contract test rather than restated
(`tests/model/test_forecaster_contract.py:48`).

**Two more interfaces that behave as contracts:** the `Broker` protocol in
`glassbox/engine/executor.py` (seven calls: `submit_market_order`, `submit_stop_order`,
`submit_limit_order`, `cancel_order`, `get_orders`, `get_positions`, `get_account`) —
implemented by `AlpacaBroker`, `ReplayBroker`, the dry-run wrapping broker, and
`tests/fake_broker.py`; and the `FeatureFrame` convention (UTC `DatetimeIndex`, one column per
canonical channel, no NaNs after warm-up trim), which has no class but is spec'd at §4.2.

## B.10 Entry points

**Package commands** (`python -m glassbox.<module>`):

| Command | Does | Writes |
|---|---|---|
| `smoke_offline` | Data → features → folds → training → calibration → backtest → metrics table beside persistence. Never touches the network. `--folds N`, `--model {dlinear,fits,persistence,wits}` | stdout table |
| `smoke_offline --prepare-live DIR` | Trains on the most recent fold, calibrates its band. Runs nothing else. | `DIR/checkpoint/`, `thresholds.json`, `reliability.json` |
| `smoke_offline --prepare-replay FOLD DIR` | Trains on FOLD's training split, calibrates on its validation split. | `DIR/checkpoint/`, `thresholds.json`, `replay.json` |
| `live_loop` | The live decision cycle. `--state-dir`, `--log-dir`, `--dry-run`, `--language {en,he}`, `--sessions N`, `--max-cycles`, `--rehearsal REASON`, `--rehearsal-notional`, `--rehearsal-close-out` | book, decisions, trades, heartbeat, session log |
| `replay` | Drives the live loop's cycle over one fold's test range, offline. `--state-dir`, `--language`, `--max-bars`, `--model` | decision records under the replay state dir |
| `experiments.study` | The grid. `--out`, `--folds`, `--full`, `--plan-only`, `--daily-out` | `results.csv`, `report/daily_equity.csv` |
| `experiments.report` | `--results`, `--out`, `--dpi` | `report/report.md`, `direction_by_arm.png`, `cof_sweep.png` |
| `experiments.stats` | Paired Wilcoxon. `--results`, `--out` | stdout table, optional CSV |
| `experiments.exposure` | `--folds`, `--out` | `report/gross_exposure.csv` |

**Streamlit:** `streamlit run glassbox/dashboard/app.py -- --state-dir ... --source ...` —
arguments, **not** environment variables, because only the config layer may read the
environment (`glassbox/dashboard/app.py:4044-4048`).

**Scripts:** `scripts/smoke_alpaca.py` (connectivity, refuses non-paper),
`scripts/compare_sources.py` (yfinance vs Alpaca field by field),
`scripts/frequency_response.py` (the GB-46 figure), `scripts/momentum_probe.py`,
`scripts/operator_null_control.py` (GB-66's acceptance measurement), `scripts/ci_status.py`,
`scripts/setup.ps1`.

**Exit codes that carry meaning:** `live_lock.EXIT_REFUSED = 3` (another loop holds this state
dir), `live_loop.EXIT_CLOSE_OUT_FAILED = 4`, `ci_status` 0/1/2/3, `smoke_offline` 2 on a missing
cache without a traceback.

> **For the deck:** `README.md` still carries three *"Not yet implemented"* notes — for
> `smoke_offline` (line 174), the dashboard (line 259), and implicitly the data snippets. All
> three are stale: `smoke_offline.py` is 951 lines and `dashboard/app.py` is 4,107. Worth
> correcting before anyone reads the README alongside the deck.

## B.11 Artefacts

| Artefact | Written by | Read by |
|---|---|---|
| `data_cache/{SYMBOL}.parquet` (20 files) | `data/historical.py:108` | `data/historical.py:215`; committed **deliberately** so cache-gated tests run in CI — guarded by `test_the_data_snapshot_is_tracked_so_gated_tests_run_in_ci` (`tests/test_scaffold.py:201`) |
| `results.csv` (1,110 rows, 28 cols) | `experiments/study.py:977` | `experiments/report.py:130`, `experiments/stats.py:297`, `dashboard/app.py:385` |
| `report/daily_equity.csv` | `experiments/study.py:987` | `dashboard/app.py:398` |
| `report/report.md` | `experiments/report.py:621` | humans / GB-57 |
| `report/direction_by_arm.png`, `report/cof_sweep.png` | `experiments/report.py:468,522` | the report |
| `report/gross_exposure.csv` | `experiments/exposure.py:341` | the report |
| `report/momentum_probe.csv` | `scripts/momentum_probe.py:481` | `IDEAS_PARKED.md` quotes its numbers |
| `report/null_control.csv` | `scripts/operator_null_control.py:237` | GB-66's acceptance |
| `figures/frequency_response.png` | `scripts/frequency_response.py:154` | the report; §6.5's "single strongest visual" |
| **Checkpoint directory** — `checkpoint/model.json`, `checkpoint.json`, `history.csv` | `model/train.py:364`, each forecaster's `save`, `model/history.py:59` | `model/train.py:383`, each forecaster's `load`, `dashboard/app.py`, `scripts/frequency_response.py` |
| `thresholds.json` | `smoke_offline.py:372,427` | `live_loop.py`, `replay.py`, `dashboard/app.py:3908` |
| `reliability.json` | `smoke_offline.py:493` | `dashboard/app.py:212` |
| `replay.json` (test range manifest) | `smoke_offline.py:442` | `replay.py:360` |
| `book.json` | `engine/reconcile.py:108` | `live_loop.py`, `dashboard/app.py:3628` |
| `decisions/YYYY-MM.jsonl` | `records.py:212,341` | `records.py:462`, `dashboard/app.py`, `replay.py` |
| `trades.jsonl` | `records.py:243` | `records.py:259`, `backtest/metrics.py` — **in `backtest`'s own `Trade` type, no translation** |
| `pending.json` (Co-Pilot queue) | `records.py:424,441` | `records.py:433`, `dashboard/app.py` |
| `entry_fills.json` | `live_loop.py:438` | `live_loop.py`, `records.py` |
| `equity_snapshots.jsonl` | `dashboard/app.py:882` | `dashboard/app.py:897` |
| `heartbeat.json` | `live_loop.py:2262` | `dashboard/app.py:3033` |
| `live.lock` | `live_lock.py:198` | `live_lock.py:127`, `dashboard/app.py` liveness |
| `quality_report.json` | `data/quality.py:123` | humans |
| `logs/live-YYYY-MM-DD.log` | `live_loop` logging, one per calendar day, appended across restarts | humans |
| `report/screenshots/*.png` + `CAPTIONS.md` | captured by hand | the architecture report |

Live state directories on disk: `checkpoints/live`, `live-next`, `replay`, `fits-demo`,
`rehearsal`, `screenshots`, plus two dated backups.

## B.12 Test counts, and the rule-level guards

**Counting method, stated because it matters:** `pytest` was **not** run, so no collected-item
count can be reported. What follows is a count of `def test_` **definitions** by directory.
Parametrised tests expand at collection, so the collected total is meaningfully higher — there
are 78 `@pytest.mark.*` decorators including 77 `parametrize` uses. `PROGRESS.md` records the
suite at 605 / 809 / 835 at various milestones, and `CLAUDE.md` cites 1,181 on 25 Aug 2026.

| Directory | Files | `def test_` | parametrize |
|---|---|---|---|
| `tests/` (root) | 15 | 261 | 5 |
| `tests/dashboard/` | 7 | 216 | 18 |
| `tests/model/` | 8 | 145 | 9 |
| `tests/backtest/` | 4 | 126 | 3 |
| `tests/features/` | 5 | 109 | 16 |
| `tests/engine/` | 6 | 104 | 0 |
| `tests/experiments/` | 4 | 88 | 2 |
| `tests/explain/` | 3 | 65 | 17 |
| `tests/data/` | 4 | 57 | 0 |
| `tests/contracts/` | 2 | 47 | 2 |
| `tests/config/` | 1 | 27 | 5 |
| **Total** | **59** | **1,245** | **77** |

26,050 lines of test against 22,450 lines of package — a ratio of **1.16:1**.

**Shared test infrastructure** (not test files): `tests/causality.py` (271), `tests/sweep.py`
(239), `tests/fake_broker.py` (212), `conftest.py` ×2.

### The guards that enforce a project rule rather than a function's behaviour

This is the list for the slide. These do not test a function; they test that the project's own
rules remain true.

**Architecture & layering**

- `test_no_layer_violations`, `test_contract_is_configured`, `test_root_package_is_glassbox` —
  `tests/contracts/test_layers.py`
- `test_spec_module_exists`, `test_tree_has_no_extra_modules` — the package tree must match
  spec §3.4 exactly, in both directions (`tests/test_scaffold.py`)
- `test_every_script_guards_its_imports`

**Config-over-constants (CLAUDE.md rule 5)**

- `test_no_module_reads_settings_or_environ_directly` and
  `test_no_script_reads_the_environment_directly` — AST-based, plus
  `test_the_guard_catches_a_real_read_and_not_a_mention_of_one` which tests the guard itself
- `test_the_module_source_contains_no_numeric_threshold` (`tests/engine/test_signal.py:138`)
- `test_the_only_numbers_in_the_module_are_the_grid` (`tests/backtest/test_calibrate.py:344`)
- `test_every_config_section_is_classified_as_shaping_or_not`,
  `test_every_model_shaping_section_moves_the_model_hash`

**Causality (the field's failure mode)**

- `test_every_indicator_is_causal`, `test_build_feature_frame_is_causal`,
  `test_build_windows_is_causal`, `test_fit_stats_uses_only_the_training_range` — and four tests
  that the **harness itself** can detect a leak: `test_the_harness_rejects_a_centred_window`,
  `..._tomorrows_close`, `..._a_fitter_that_forgets_to_slice`, plus one leak per perturbation
  mode
- `test_no_timestamp_appears_in_two_splits_of_one_fold`,
  `test_no_training_target_reaches_the_validation_range`,
  `test_no_test_target_reaches_the_next_folds_test_range`

**Exactness**

- `test_the_explain_layer_imports_no_perturbation_library` — the SHAP/LIME/captum ban, enforced
- `test_every_registered_forecaster_is_exact` — iterates `ALL_FORECASTERS`, so a new model
  cannot skip it
- `test_forecaster_contract` + 5 meta-tests that the contract test can actually fail a broken
  forecaster

**Two-places (the family CLAUDE.md names nine times)**

- `test_every_declared_dependency_is_pinned_in_the_lock` — `pyproject.toml` vs
  `requirements.lock`
- `test_every_universe_written_in_the_spec_matches_the_configuration`,
  `test_default_config_values_match_the_spec`
- `test_every_registered_model_is_selectable_from_the_command_line` — `ALL_FORECASTERS` vs
  `smoke_offline`'s argparse
- `test_a_condition_is_identified_by_the_columns_it_writes` — fails whenever two study
  conditions would write identical columns
- `test_every_pair_key_is_a_column_the_grid_writes`
- `test_the_caption_is_written_in_exactly_one_place`

**Reporting honesty (spec §7.3)**

- `test_mae_never_appears_in_a_summary_without_flatness_beside_it`
- `test_every_metric_declares_which_direction_is_better`,
  `test_every_companion_named_is_a_metric_the_summary_computes`
- `test_every_delta_column_names_its_own_reference`,
  `test_the_legend_names_the_reference_and_rules_out_the_two_wrong_ones`
- `test_every_claim_is_shown_at_all_three_anchors`,
  `test_every_claim_is_shown_against_both_null_conditions`
- `test_every_empty_cell_is_named_and_reasoned`,
  `test_every_cutoff_is_measured_against_its_own_null_control`
- `test_the_data_snapshot_is_the_first_line_of_the_header`,
  `test_two_snapshots_in_one_file_are_refused`
- `test_the_method_is_in_the_output_not_only_in_a_docstring`

**Console honesty (the source-pill family)**

- `test_every_region_renders_its_source_pill`,
  `test_a_backtest_pill_states_a_fold_range_and_not_a_bare_label`
- `test_every_performance_panel_states_its_reference_or_is_named_as_unfixed`
- `test_every_live_pill_on_the_page_is_built_from_the_verdict`
- `test_no_data_mark_is_drawn_in_the_chrome_colour`,
  `test_every_status_colour_is_redundant_with_a_sign_and_a_glyph` (the traffic-light ban),
  `test_every_colour_the_module_defines_is_classified`,
  `test_the_luminance_formula_is_the_one_wcag_specifies`
- `test_only_module_built_html_reaches_a_raw_column`

**Secrets & process**

- `test_no_tracked_file_contains_a_secret_shaped_string`, `test_env_is_not_tracked`, plus three
  tests that the scanner detects real keys and does not cry wolf
- `test_the_data_snapshot_is_tracked_so_gated_tests_run_in_ci` — CLAUDE.md instance 2, made
  mechanical
- The whole of `tests/test_phase2_ledger.py`: every GB-61..66 task must be recorded complete
  **or** parked, never both, never neither, with a **resolvable citation** (a SHA, a path, or a
  test node id)
- `test_the_same_grid_twice_gives_the_same_numbers` — determinism

**The harnesses' own refusals** (`tests/sweep.py`): `bool(SweepResult)` raises `TypeError`; a
sweep of one cell is refused. These are not tests — they are refusals in the tool, so the
failure mode is *unavailable* rather than discouraged.

---

# PART C — THE THREE MINUTES

Four, from the record. Ordered by how well they play on camera, and every number below is
quoted from `DECISIONS.md` or `PROGRESS.md`, not from memory.

## 1. A finding that survived every test, and was wrong

**The problem.** FITS appeared to advance 15–30 day market cycles by about two days — a real,
publishable-sounding claim about equity momentum.

**Attempt 1 — more folds.** The sweep asked for was run: **16 folds × 3 fold-grid anchors, 48
independently trained models. +1.9582 days, sd 0.0203, positive in 48 of 48 cells**, and
indistinguishable between anchors (+1.9569 / +1.9538 / +1.9640). That is a coefficient of
variation of **1.04%**. It passed. *"Stability across folds would have confirmed the claim,
which is the whole problem: a check that a correct explanation and an incorrect one both pass is
not evidence."*

**Attempt 2 — a control instead of a bigger sweep.** Fit the same model on data with no
temporal structure:

| | phase, 15–30 day band |
|---|---|
| real returns | +1.9640 days |
| within-window shuffle | +1.9553 days |
| **white noise** | **+1.9309 days** |
| untrained (zero weights) | +0.0000 |

**It survived the complete destruction of temporal structure.** The whole gain curve on white
noise correlates with the real-data curve at **+0.9485** — about **86% of the learned frequency
response is data-independent.**

**The replacement finding is better than the one it displaced.** Gain tracks **grid
misalignment**: Pearson **−0.9049**, Spearman **−0.9427, p = 1.8e-11** against
`|η·k − round(η·k)|`. The worst bin, `k=15`, sits at **exactly 0.500** misalignment — the
frequency that lands precisely between two output bins and cannot be reconstructed. The phase
even changes sign at that boundary, *"which no market story predicts and grid geometry does."*

**Evidence:** `DECISIONS.md:1133` and `DECISIONS.md:1091`. **Twelve hours** separated writing
the claim from withdrawing it. The rule it produced is now in CLAUDE.md: *"Suspicious stability
calls for a control, not a larger sweep."* Two tests, and neither subsumes the other — grid
sensitivity killed a correlation of `r = −0.47`; the null control killed the phase advance,
which had passed grid sensitivity at **48 of 48**.

**Why it is the best thirty seconds of the video:** it is not "I found a bug." It is a method
claim with a number attached, and it generalises past this project.

## 2. A verified claim that held in 32 of 125 cases

**The problem.** The live loop and offline training must build byte-identical windows, or
nothing downstream is comparable. The spec recorded a parity floor of **352 bars** as
*verified*.

**Attempt 1.** GB-9 derived 352 by bounding the RSI seed's *weight* below 1e-7. It was tested —
at one symbol, at one timestamp. It passed.

**The correction (Ben's).** What must fall below float32 resolution is
`weight × seed_difference`, and the seed difference is **not bounded by 1**. Near RSI 50 a
float32 ulp is 5.95e-06 and the measured residual was **3.815e-06** — the same order of
magnitude. *"352 was marginal by construction, which is precisely why it held in some pairs and
not others rather than in none or in all."*

**Attempt 2 — sweep it.** Five symbols × twenty-five timestamps, 125 comparisons:

| tail | byte-identical |
|---|---|
| **352** (the old floor) | **32 / 125** |
| 400 | 116 / 125 |
| 414 | 120 / 125 |
| 420 | 125 / 125 |
| **445** (adopted) | **125 / 125** |

The floor is 445, not 420, because *"a number that a sweep happens to pass is exactly how 352
arrived."* A tolerance was considered and rejected: *"'Within one ulp' is the kind of close
enough that lets a genuine one-bar offset hide."*

**The finding underneath the finding.** The audit that followed did not produce a list of
suspects; it produced a mechanism: *"every claim in this project that survived scrutiny came
from a task that happened to have a harness to sweep with, and every claim that did not came
from a task that measured by hand on whatever was in front of it. That is a statement about
tooling, not about care, so the fix is a tool."*

That tool is `tests/sweep.py`, and its two properties are refusals: a `SweepResult` **cannot be
used as a boolean** (`bool(result)` raises `TypeError` and names the attributes to read
instead), and **a sweep of one cell is refused**. `32 of 125` is the number that would have
caught the floor; `it passes` is the number that did not.

**Evidence:** `DECISIONS.md:3269`, `DECISIONS.md:3190`, and the standing audit at
`PROGRESS.md:810` which re-classifies every single-point claim in the project into confirmed /
re-swept / deferred-with-cost-stated.

## 3. 1,181 tests passed while the live path could not place a single order

**The problem.** The protection policy — five rules the whole of GATE 2 depends on — was fully
tested. Every test passed.

**Attempt 1 (GB-22, 17 Aug).** Discovered by submitting orders to the real paper API: Alpaca
refuses a bracket on a fractional quantity (`fractional orders must be simple orders`; the same
bracket on one whole share is accepted). Workaround adopted: two standalone day orders, a stop
and a limit. `MIN_SHARES = 0.001` was found to be *"the right idea in the wrong unit"* — Alpaca
enforces a **$1.00 minimum notional** — and the API **silently truncates quantity to nine
decimals**. The GB-21 property test caught a numerical flaw while this landed:
`floor(x * 1e9) / 1e9` is not the floor once the scaled value leaves float64's exact integer
range. Now `Decimal`.

**Attempt 2 (25–26 Aug).** The two standalone orders never worked either. **A working sell
order holds the whole position**, so the second is refused with `insufficient qty available` —
measured against a **97.38-share** position, so it is not fractional-specific.
`tests/fake_broker.py` had been permissive in **exactly the dimension the protection policy
depends on**, so every test of that policy was a test of the fake's opinion of Alpaca.

**Making the double faithful turned 26 tests red across three files** — recorded in
`PROGRESS.md` as *"a measurement, not a regression"*, and the number is the size of what was
never being tested. One test, `test_a_filled_entry_is_protected_by_two_standalone_orders`, was
**deleted rather than fixed, because its premise was impossible.**

**The resolution is the study's turning sentence** — and this is the thing to put on screen:

> *The published backtest describes a system that cannot be built at this venue. Under the
> buildable model the trade count falls in every arm — 16%, 14%, 29% — because a delayed exit
> holds capital and entries that would have followed never happen. The buildable system is
> measurably **different**, not measurably worse.*

Execution fidelity became a **study axis**, not a caveat: `target_in_loop` is a column in
`results.csv` and `true` is the default. And it did not rescue the null — `direction` is a
forecast metric and is **byte-identical across both arms**.

**Evidence:** `DECISIONS.md:3432`, `DECISIONS.md:5075`, `DECISIONS.md:5126`, `PROGRESS.md:349`,
`PROGRESS.md:437`. The rule it produced: **a test double may refuse more than the real thing,
never less.**

## 4. The result is a null, and the control is what makes it a result

Worth 30 seconds because it is the answer to *"so did it work?"* and the honest answer is the
interesting one.

Direction accuracy, anchor 0, mean over folds and conditions:

| model | real | noise | noise − real |
|---|---|---|---|
| **buy_and_hold** | 0.5560 | **0.5064** | **−0.0495** |
| dlinear | 0.5006 | 0.5006 | +0.0000 |
| fits | 0.4958 | 0.5041 | **+0.0083** |
| wits | 0.5031 | 0.5050 | +0.0018 |

FITS scores **better on white noise than on the market**. DLinear is identical to four decimals.
No model reaches the always-long bar (0.5395) at any of the three anchors.

*"`buy_and_hold` is the only arm that degrades on noise, and that is the control working. It is
the one arm whose entire content is drift, so it is the one arm that must lose something when
the drift is destroyed — and it loses 4.95 points. A control that moves the right way for
exactly one arm, and for the arm whose mechanism predicts it, is what turns 'we found nothing'
into **'we measured that there was nothing to find, and the instrument can detect something when
it is there.'**"*

**Evidence:** `PROGRESS.md:459`. Provenance for every number: snapshot `2026-08-13`, 1,110 rows,
768 reportable, 28 conditions, 16 folds. **22 rows skipped, every one with its reason in the
file.**

---

## Honourable mentions, if there is slack

- **The 90-minute stall** (`DECISIONS.md:10`) — no HTTP read had a timeout because the Alpaca
  SDK takes no `timeout` parameter. *"The retry policy did not help, because it bounds the wrong
  thing: `retry_attempts: 3` bounds the number of attempts and nothing bounded their
  duration."* The chosen value moved 30 s → 45 s not on the ratio but on the **provenance of the
  maximum** — the slowest healthy call was the first cycle *recovering from an outage*, the one
  moment where a spurious timeout is self-reinforcing. And *"a dry run cannot settle it"*,
  because the case the value turns on is not reproducible on demand.
- **Two mechanisms under one label** (`DECISIONS.md:1498`) — the supervisor's arithmetic: with
  the scaler on, DLinear's wanted weight is ~6.1e-4 while **Adam's step at `lr=1e-3` is 1.0e-3,
  larger than the weight it is looking for**. Measured median: **5.435e-04**, with **71.1% of
  learned weights below the optimiser's step**. Confirmed for DLinear (0/16 → 14/16 → 16/16
  folds as `lr` drops), and **refuted for FITS**, whose real defect was elsewhere: with `X`
  standardised and `y` in raw log returns, the B+F objective became **98.5% backcast** and
  silently stopped being B+F.
- **The 40-minute grid killed by a test that was already written** (`DECISIONS.md:914`) —
  cache-gated, deselected, never executed. *"Run the test that validates an artefact before
  spending forty minutes producing it."*
