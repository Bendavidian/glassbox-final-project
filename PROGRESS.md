# PROGRESS

The running log of what is built. **Claude Code reads this file to locate the current phase.**
Append one line per completed task. Newest at the bottom of each sprint.

---

## Current state

**Sprint:** 1 — Foundations · **complete, 14 Aug 2026** (12/12, day 6 of 14)
**Sprint 2:** in progress — GB-17, GB-18 complete 16 Aug
**Next task:** GB-19, metrics (completes the harness chain; then the model chain, GB-13)
**Last gate passed:** none — GATE 1 target revised to ~22 Aug, commitment 11 Sep
**Blockers:** none

---

## Pre-sprint

| Task | Date | What was built |
|---|---|---|
| GB-0b | 14 Aug 2026 | Resolved three doc contradictions: solo plan governs execution (CLAUDE.md §6), `Attribution.per_lag` optional + `kw_only` (spec §4.2), GB-14/31/43 marked cut in spec §9, code freeze corrected to Sat 3 Oct. Three DECISIONS entries. |

## Sprint 1 — Foundations · 15–28 Aug 2026

| Task | Date | Owner | What was built |
|---|---|---|---|
| GB-1 | 14 Aug 2026 | Ben | **Complete.** Package tree per spec §3.4 as docstring-only stubs (incl. `smoke_offline.py`), `pyproject.toml` (deps + dev extras + ruff/black/pytest config), `tests/` mirror with conftest and a scaffold test that fails on any module not in §3.4, `.gitignore`, Windows CI on Python 3.11. `reference/` excluded from packaging, lint and collection. ruff/black/pytest green: 43 tests. |
| GB-2 | 14 Aug 2026 | Ben | **Complete.** `settings.yaml` verbatim from spec §5; `config/loader.py` parses it into 13 frozen dataclasses, validates every field with `ValueError` naming the exact path, exposes `load_config`, `config_hash` (stable SHA-256) and lazy `alpaca_credentials` from `.env`. 39 config tests; suite at 82. |
| GB-3 | 14 Aug 2026 | Ben | **Complete.** `contracts/schemas.py` — five frozen schemas from spec §4.2 with `__post_init__` shape validation; `contracts/protocols.py` — runtime-checkable `Forecaster`; import-linter layers + forbidden contracts in `pyproject.toml` with the harness above the live path; `tests/contracts/` proves both contracts have teeth. Suite at 108. |
| GB-4 | 14 Aug 2026 | Ben | **Complete.** `data/historical.py` — `load_history` fetches daily bars per symbol via yfinance, normalises to UTC/lowercase/sorted/deduplicated, adds unfilled `log_return`, and caches to `{cache_dir}/{symbol}.parquet`; a cache hit never touches the network. All five symbols cached 2016-01-04 → 2026-08-13, 2668 bars each. 16 offline tests on a committed CSV fixture; suite at 124. |
| GB-5 | 14 Aug 2026 | Ben | **Complete.** `data/quality.py` — `check_quality` reports missing NYSE trading days, per-column NaN counts, single-bar log returns above 0.25, duplicates and span; `report_all` prints a table and writes `data_cache/quality_report.json` (committed as provenance). Real universe: 0 missing bars, 0 duplicates, 1 NaN each (the unfilled first return), 1 large move total — NVDA 2016-11-11, +29.8%. 12 tests; suite at 136. |
| GB-6 | 14 Aug 2026 | Ben | **Complete.** Credentials load only via `alpaca_credentials()`; `require_paper_endpoint()` refuses any non-paper endpoint (verified, exit 2); `.env.example` committed, `.env` ignored; git history scan clean. `scripts/smoke_alpaca.py` prints the paper account: ACTIVE, equity 100000, buying power 400000. Scripts guard their imports and name the venv command. |
| GB-7 | 14 Aug 2026 | Ben | **Complete.** `data/live.py` fetches Alpaca daily bars through `historical.normalise_bars`, the single schema definition, so live and historical are indistinguishable by construction. Pins `feed=SIP` and `adjustment=ALL` (default RAW is off by 4× across a split); no caching. Measured vs yfinance over 163 bars: prices agree to <0.3 bps, volume differs 34–111 bps. 10 schema tests on a recorded response fixture; suite at 155. |
| GB-8 | 14 Aug 2026 | Ben | **Complete.** `features/indicators.py` — `rsi14` (Wilder, explicit recursion), `vol_z`, `mom10`, `ma_dist20`; all pure, trailing-only, warm-up left NaN. RSI holds 77 rows, not 14, until its seed's weight decays below 1%. vol_z verified source-safe over 2668 bars. 53 tests against an independent pure-Python reference; suite at 208. |
| GB-9 | 14 Aug 2026 | Ben | **Complete.** `features/builder.py` — the keystone. `build_feature_frame` (provenance-gated, warm-up trimmed), `fit_stats` (external, records its fitted range), `build_windows` (one slicing path; `as_of` filters the same positions the batch path uses). Contract additions: `ChannelStats` and `WindowBatch.source` in spec §4.2. `min_history_bars` = 352 for C0, derived from per-channel warm-ups and verified byte-identical against full history. 30 tests; suite at 240. |
| GB-10 | 14 Aug 2026 | Ben | **Complete.** `tests/causality.py` — reusable `assert_causal` (per-timestamp) and `assert_fit_isolated` (fitting), each running two perturbation modes at 25/50/75% splits. All four indicators, `build_feature_frame` and `build_windows` pass on real AAPL bars; a centred rolling mean, tomorrow's close and a whole-frame fitter are all rejected. `scale` alone is blind to a leaked *direction* and `shuffle` alone to a leaked *mean*, both measured. 16 tests; suite at 256. |
| GB-11 | 14 Aug 2026 | Ben | **Complete.** `model/persistence.py` — `PersistenceForecaster`, the zero-return baseline every result is reported against; JSON checkpoint, versioned. `tests/model/test_forecaster_contract.py` — spec §4.4's six properties plus a seventh, parameterised over a registry so GB-13/GB-41 plug in by appending one line (verified: 27 tests → 35). Seven deliberately broken forecasters prove every property can fail, and a confinement test proves each break is narrow. Contract additions: `FitProvenance` (§4.2) and `Forecaster.fitted` (§4.3). 43 tests; suite at 299. |
| GB-12 | 14 Aug 2026 | Ben | **Complete.** `README.md` — clone to running with no prior context: prerequisites, venv activation spelled out per shell, install from `requirements.lock`, credential setup, the runnable snippets, the test suites that carry the correctness argument, the repository map, and a troubleshooting table. `ARCHITECTURE.md` — layer stack, responsibilities, frozen contracts, the keystone, causality, explainability, validation; cross-references the spec rather than duplicating it. `readme` re-enabled in `pyproject.toml`. |

### Sprint 1 review

**Closed 14 Aug 2026 — day 6 of a 14-day budget.** 12 of 12 tasks complete. `pytest` 299
green, `ruff`/`black` clean, CI green, both import-linter contracts kept.

#### Contract and spec changes, one line each

Every one is recorded in full in `DECISIONS.md`.

| Ruling | Change |
|---|---|
| GB-0b | `Attribution.per_lag` optional and the dataclass `kw_only`, so GB-31's cut costs no contract change later |
| GB-0b | GB-14/31/43 struck through in spec §9 rather than deleted, so a reader sees the decision instead of a gap |
| GB-0b | `SOLO_BUILD_PLAN.md` governs execution; the Jira 70/30 split is a formal record only (CLAUDE.md §6) |
| GB-0b | Code freeze corrected to Sat 3 Oct after checking every weekday label against the real 2026 calendar |
| GB-1 | `smoke_offline.py` added to spec §3.4; `scripts/` declared outside the package |
| GB-2 | Python floor raised to 3.12 — the pinned numpy and scipy were never installable on 3.11 |
| GB-7 | Alpaca pinned to `feed=SIP`, `adjustment=ALL`; `live.py` does not cache |
| GB-8 | Indicator periods are definitions, not config — a `vol_z` with a configurable window is a different indicator |
| GB-9 | `WindowBatch.source` and `ChannelStats` added to spec §4.2; provenance travels as a column, not `.attrs` |
| GB-9 | Features computed once on the full series and sliced per fold; statistics fitted per fold |
| GB-9 | `min_history_bars(cfg)` derived from per-channel warm-ups; spec §9 GB-26 must request it |
| GB-9 | The canonical channel renamed `close` → `close_logret` — a channel named for a price that holds a return is a defect in the explanation layer |
| GB-10 | `tests/causality.py` is shared test infrastructure in `tests/`, with the promotion trigger named |
| GB-11 | `FitProvenance` added to spec §4.2 and `Forecaster.fitted` to §4.3; §4.4 gains a seventh property |
| GB-11 | Spec §4.4 records that properties 5 and 6 presuppose property 2, and the test defers rather than triple-failing |
| Security | `git add -A` banned in CLAUDE.md §3 after a real key reached a local commit; `tests/test_no_secrets.py` enforces it |
| Scope | The Regime Guard returns to consideration at GATE 2, and only if that gate is green |

#### The four findings that were the sprint's real output

The code was mostly specified in advance. These were not, and each changes what a later
task must do.

**1. Alpaca's default `RAW` adjustment silently disagrees with yfinance by a split
factor.** Across AAPL's 2020 4:1 split, `RAW` closes 499.75 where the adjusted series
closes 121.08. Every dtype and schema assertion passes; only the prices are wrong, by 4×.
Discovered by comparing the two sources field by field rather than by trusting a matching
schema. Now pinned to `adjustment=ALL`, with the measured tolerance table in the module
docstring. The related trap: Alpaca stamps bars at 04:00 UTC and yfinance at 00:00, so a
naive join of the same trading day returns zero rows while every assertion passes.

**2. The 352-bar parity floor.** A live window built from a 197-bar tail — the window
length plus the point where RSI stops visibly distorting — differs from the training
window at the same timestamp by 4.261e-03, permanently and invisibly. 352 bars is where
Wilder's seed decays below float32 resolution and the difference becomes exactly zero
(197 → 4.261e-03, 250 → 7.713e-05, 300 → 5.960e-07, 352 → 0.000e+00). GB-27's parity test
could not have found this: it compares builder outputs on one frame, not on histories of
different length. The number is now derived from per-channel declared warm-ups, so GB-47's
wavelets update it without an edit.

**3. A multiplicative perturbation is blind to a leaked direction.** `sign(1.5·x) ==
sign(x)`, exactly — so a function leaking tomorrow's direction passes the perturb-the-
future causality test specified for GB-10 with every value bit-identical. Direction
accuracy is one of the two headline metrics in spec §1.4, which makes it the most
expensive leak the project could ship. Demonstrated with a leaky function that passes
under `scale` and fails under `shuffle`. The blindness runs both ways: a permutation
cannot move a full-sample mean (measured: `scale` moves it 4.4e+00 relative, `shuffle`
1.2e-15), so neither mode subsumes the other and both run by default.

**4. For a model, "the future" is later windows, not later lags.** Every lag in a window is
at or before that window's own timestamp by construction, so the causality property spec
§4.4 states literally is not the one that bites. What remains is a statistic computed
across a time-ordered batch — instance normalisation implemented as *batch* normalisation,
which pulls later windows into earlier predictions. It is a live risk for FITS's RIN stage
in GB-41, and no single-window check can find it, because a batch of one centres to zero.
Now property 6, asserted with GB-10's harness.

#### Still open

- **`smoke_offline`, `live_loop` and the dashboard are stubs.** README says so at each
  command. `python -m glassbox.smoke_offline` exits 0 silently today; GB-24 fills it.
- **GB-26 carries two standing requirements** it must satisfy: a named, tested
  `drop_incomplete_bar`, and a live fetch of at least `min_history_bars(cfg)` bars.
- **GB-47 must declare `wav_a1..a3: 64`** in `builder.PARITY_WARMUP`. `C2_hybrid` raises
  until it does, rather than silently assembling `C0`.
- **GB-57 must record the 2018-05-02/03 vendor volume event** in the data-quality
  appendix: yfinance and Alpaca disagree materially on those two days for all five symbols
  at once, visible in `vol_z` and in no other channel.
- **GB-15 may need `stats` on `FitProvenance`.** If a checkpoint cannot link a model to
  the statistics it was normalised with, that is the additive move. The trigger has not
  fired.
- **No blocking questions.**

## Sprint 2 — Offline Vertical Slice · GATE 1 target ~22 Aug (commitment 11 Sep)

Order follows `SOLO_BUILD_PLAN.md` §7: the harness chain (GB-17 → GB-18 → GB-19) runs
before the model chain, because the backtester is the largest task in the project and
depends on nothing in the model layer.

| Task | Date | Owner | What was built |
|---|---|---|---|
| GB-17 | 16 Aug 2026 | Ben | **Complete.** `backtest/walkforward.py` — calendar-month folds on real NYSE sessions, capped at `max_folds` keeping the **most recent**, truncated final folds dropped rather than shortened. Carries the **target embargo**: each split drops its last `H` window-ends, derived from `build_windows`' label definition (`p+1 .. p+H`), so no window's label crosses a split boundary. Real AAPL data yields 16 folds of 30 candidates. 23 tests, including two leak assertions that fail when the embargo is set to zero. Suite at 325. |
| GB-18 | 16 Aug 2026 | Ben | **Complete.** `backtest/engine.py` — event-driven, one bar at a time, no vectorised shortcut anywhere. Four pricing rules, each measured before being chosen and each tested: fills at the **next** open; the **stop** wins an ambiguous bar (decides 0.08% of resolutions); slippage **adverse both sides**, flat round trip exactly 6.0 bps; a **gap fills at the open** (15.6% of stop exits, worth ~19× the friction model), logged as `stop_gap`. Sizing injected via `PositionSizer`; no risk logic in the engine. Contract addition: `backtest.initial_cash`. Hand-checked three-trade scenario matches literals derived independently in 40-digit decimal. 30 tests; suite at 357. |
| GB-19 | | Ben | |
| GB-13 | | Ben | |
| GB-15 | | Ben | |
| GB-16 | | Ben | |
| GB-20 | | Ben | |
| GB-21 | | Noy | |
| GB-24 | | Ben | |
| GB-25 | | Ben | **GATE 1 review** |

## Sprint 3 — Live End-to-End · 12–25 Sep 2026 → GATE 2

_not started_

## Sprint 4 — FITS, Study, Report · 26 Sep – 10 Oct 2026 → GATE 3

_not started_

---

## Gate log

| Gate | Date | Result | Notes |
|---|---|---|---|
| GATE 1 | ~22 Aug (commitment 11 Sep) | pending | Sprint 1 closed on day 6 of 14 |
| GATE 2 | ~1–3 Sep (commitment 25 Sep) | pending | Needs 2–3 real market sessions; see `SOLO_BUILD_PLAN.md` §4.1 |
| GATE 3 | 10 Oct 2026 | pending | **Unchanged.** The submission date does not move. |

---

## Open questions

_Claude Code: write blocking questions here rather than guessing._

_The 14 Aug CI failure is resolved — see DECISIONS.md, "Python floor raised to 3.12";
green on 176fe01._

_None open._
