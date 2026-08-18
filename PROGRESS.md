# PROGRESS

The running log of what is built. **Claude Code reads this file to locate the current phase.**
Append one line per completed task. Newest at the bottom of each sprint.

---

## Current state

**Sprint:** 1 — Foundations · **complete, 14 Aug 2026** (12/12, day 6 of 14)
**Sprint 2:** **complete** — GB-17, GB-18, GB-13 on 16 Aug; GB-18 fix, GB-15, the
`WindowBatch.symbols` contract change, GB-16, GB-19, GB-20, GB-24, GB-21 and GB-25 on
17 Aug
**Next task:** GB-39 fault handling and restart safety, then GATE 2. GB-37 and GB-38 are
complete and were demonstrated together: a replayed fold produced two recommendations, one
approved (submitted, both protective legs armed) and one declined (broker untouched).
**Outstanding operational item:** the paper account holds **0.0919 AAPL, unprotected and
quarantined** — GB-22's fill, submitted outside the loop, so the system correctly refuses
to manage it. Ben's standing instruction to flatten it has not been executed. GB-26 is complete and ran a full dry-run cycle
against the live SIP feed on 18 Aug: ten steps, 610 bars a symbol against a floor of 445,
the 0.01 AAPL probe quarantined, five decisions recorded and narrated, nothing submitted.
**The session on 18 Aug will not trade, and that is the correct outcome:** the most recent
complete fold **stands aside** — its best calibrated candidate scored a validation Sharpe of
**−3.38** over 7 trades — so `Thresholds.never()` is in force and no entry can fire. **One
live constraint GB-26 designs around:** the account refuses **recent** SIP data, so anything
reaching for a live quote rather than a completed daily bar meets a wall.
**Last gate passed:** **GATE 1, 17 Aug 2026**, 25 days before its commitment date
**Blockers:** none. Standing note for GB-57, re-measured on the fold grid as it stands after
GB-27's warm-up unification: the strategy beats **neither** reference it should be read
against. Direction **0.5182** against an always-long bar of **0.5560**; return **+0.40% per
fold against buy-and-hold's +7.73%**, Sharpe **0.62** against **1.39**. The whole chain runs
end to end in one command, so these are measurements rather than expectations.

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
| GB-9 | 14 Aug 2026 | Ben | **Complete.** `features/builder.py` — the keystone. `build_feature_frame` (provenance-gated, warm-up trimmed), `fit_stats` (external, records its fitted range), `build_windows` (one slicing path; `as_of` filters the same positions the batch path uses). Contract additions: `ChannelStats` and `WindowBatch.source` in spec §4.2. `min_history_bars` = 352 for C0, derived from per-channel warm-ups and verified byte-identical against full history. 30 tests; suite at 240. **Corrected 18 Aug by GB-27:** that verification was one symbol at one timestamp and holds in 32 of 125 symbol-timestamp pairs; the floor is now **445**, derived at a 1e-10 decay target. |
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
| GB-18 | 16 Aug 2026 | Ben | **Complete.** `backtest/engine.py` — event-driven, one bar at a time, no vectorised shortcut anywhere. Four pricing rules, each measured before being chosen and each tested: fills at the **next** open; the **stop** wins an ambiguous bar (decides 0.08% of resolutions); slippage **adverse both sides**, flat round trip exactly 6.0 bps; a **gap fills at the open** (15.6% of stop exits, worth ~19× the friction model), logged as `stop_gap`. Sizing injected via `PositionSizer`; no risk logic in the engine. Contract addition: `backtest.initial_cash`. Hand-checked three-trade scenario matches literals derived independently in 40-digit decimal. Notional→shares goes through `engine.risk.shares_for`, the one conversion GB-22 must reuse; `Trade.strategy_exit` flags administrative exits for GB-19; total-return convention recorded. 37 tests; suite at 364. |
| GB-18 fix | 17 Aug 2026 | Ben | **Complete.** Review found the engine popping a position from its book *before* checking whether the symbol had a price that day. Measured on the committed engine: **−10,001.00 on a 100,000 account, 10.0%, with an empty trade log** — in both an exit-on-a-halted-day case and a symbol-history-ends-early case, the second of which fires whenever one symbol's listing is shorter than the universe's. Fifth pricing rule added: **a missing bar is a halt, not an exit** — exits carry forward to the symbol's next traded open, entries expire, halted positions mark at their own last printed close, and a symbol ending early liquidates at that close. Marking at `last_mark` also removed a look-ahead: sizing an order filling at bar `t`'s open previously consulted bar `t`'s **close**. New standing invariant `_assert_accounted` runs on **every** backtest: flat book ⇒ final equity == initial cash + Σ `net_pnl`. 7 tests, 6 of which fail against the previous engine. Suite at 402. |
| GB-19 | 17 Aug 2026 | Ben | **Complete.** `backtest/metrics.py` — forecast and trading metrics, every one taking an optional baseline and returning the delta, plus `summarise` for the report table. Sharpe, max drawdown and direction accuracy checked against literals computed by hand away from the code. **Three rulings.** Sharpe comes from the **equity curve's daily returns**, not per-trade returns — the settling property is that two arms with the same curve must score the same whatever their trade counts. Below **five strategy trades** it is NaN, and the table prints the trade count beside it so a blank cell explains itself. And **direction accuracy is referenced to the measured chance level, not to persistence**: persistence forecasts zero, `sign(0)` agrees with nothing, so its direction accuracy is undefined — **NaN in 16/16 folds**. Measured across those folds the real chance level is **0.5865**, not 0.5; DLinear scores **0.4751** and beats it in **1/16 folds**, against 6/16 if 0.5 were used. Reporting against 0.5 would have flattered every arm by 8.65 points. Annualisation uses each fold's own bar count, never 252. 36 tests; suite at 510. **Corrected the same day by GB-20** — see that row: the majority-class bar was itself inadmissible, and the reference is now always-long at 0.5625. The 0.4751 quoted here is the full-batch model, reverted the same day; the configured model scores **0.5071** and beats the bar in **5/16**. |
| GB-13 | 16 Aug 2026 | Ben | **Complete.** `model/ltsf.py` — `DLinearForecaster`, centred moving-average trend plus remainder, one linear map per component **per channel** so GB-30's attribution is a regrouping rather than a reconstruction. No intercept, so `Σ per_channel == forecast` holds by construction. **Zero initialisation**, chosen by measurement: the reference's `1/sqrt(L)` assumes a fan-in of 120 where this summed architecture has 1200, and costs 4.00× persistence MAE and 0.344 direction against 1.94× and 0.557 for zeros. Torch fits, numpy predicts, so inference determinism is structural. Registry moved to `glassbox.model.ALL_FORECASTERS`; the contract test now needs no edit for GB-41. 4,800 parameters. 23 tests + 8 contract; suite at 395. |
| GB-15 | 17 Aug 2026 | Ben | **Complete.** `model/train.py` + `model/history.py` — one training run around one `fit` call, per §4.3. The hard requirement is structural, not careful: windows are built **once** with the training statistics and split by timestamp afterwards, so there is no second `build_windows` call to hand a second `ChannelStats`. A checkpoint is a **directory** — `model.json`, `checkpoint.json`, `history.csv` — and loading refuses any config-hash difference. `held_out` is recorded so GB-25's audit is self-contained, and a test proves the weights are bit-identical with and without it. **Measured on all 80 arms (5 symbols × 16 folds):** best-epoch selection improved test MAE in **80/80** (2.757× → 2.004× persistence, 27.3%), while `patience` produced an **identical model in 79/80** — early stopping is a regulariser and a compute budget, and only the first matters. Full batch beat mini on MAE in 57/80 but its decisive argument was measurably false (weights differ across seeds by rel 4.8e-15, so it is not seed-free), so mini-batch stands. **Direction accuracy across 80 arms is 0.4971 — at chance**, which corrects GB-13's single-fold 0.557. 27 tests; suite at 430. Open for ruling: per-symbol vs universe-wide training. |
| GB-3 rev | 17 Aug 2026 | Ben | **Contract change.** `WindowBatch.symbol: str` → `symbols: tuple[str, ...]`, one entry per window, plus `WindowBatch.concat` as the only place pooling happens — it refuses batches disagreeing on channel tuple (order included), window geometry or `source`. `FitProvenance.from_batch` now takes `timestamps.min()/.max()`: a pooled batch's timestamps are not monotonic, and the old derivation would have reported the last symbol's end as the batch's, understating the range GB-25 audits. Committed on its own before GB-16. 13 tests. |
| GB-15 rev | 17 Aug 2026 | Ben | **Universe-wide training, per-symbol scaler**, per Ben's ruling. `train` takes `{symbol: frame}` and iterates it **sorted** — the pooled batch's row order decides the mini-batch partition, so determinism must not depend on dictionary insertion order. Checkpoint holds `{symbol: ChannelStats}`. A symbol whose scaler cannot be fitted, or which contributes no window, names itself. 9 tests. |
| GB-16 | 17 Aug 2026 | Ben | **Complete.** `model/predict.py` — batched and single-window inference, **bit-identical** on every test window (both call `build_windows`; GB-9 made `as_of` a filter over the same end positions, not a second path). The checkpoint decides everything; the live config is passed in only for the hash comparison, re-checkable at the point of use via `require_current_config` because GB-26's loop can outlive its config. Per-symbol statistics come **from the checkpoint, never the caller** — the caller names a symbol, so it cannot score AAPL with NVDA's scaler. An untrained symbol is **refused**, naming what exists: both fallbacks return a plausible forecast instead of an error, and fitting a scaler now would use data from the period being predicted. A reordered channel set is refused as firmly as a different one. Measured: 305 pooled windows in 24.6 ms; 0.88 ms per single window. 16 tests; suite at 466. |
| GB-20 | 17 Aug 2026 | Ben | **Complete.** `engine/signal.py` — trend strength (cumulative predicted return), `up_points`, and the three verdicts; **not one numeric literal in the module**, enforced by a syntax-tree test. Exit is the **mirror** of the entry threshold, `-lower`, so a bare zero never enters the source and a position is not churned out on weakness the entry rule would have ignored. `backtest/calibrate.py` (new, spec §3.4) holds the grid search: it lives one layer up because the ruling is that candidates are scored by the **real backtester**, and the engine layer — and therefore the live path — may not import the harness. The grid is **quantiles of each fold's own validation forecasts**, so it is scale-free across arms; a test scales every forecast by 10 and gets the same trades. Calibration passes `assert_fit_isolated`. **Ruling: a fold whose best band has no positive validation Sharpe stands aside** — no trades, flat curve, recorded — rather than trading the least-bad of 15 candidates validation rejected; measured 3/16 folds under full batch, 1/16 under mini. **Direction reference corrected**: the majority-class bar shipped hours earlier was set with test-period knowledge, so it is replaced by **always-long**, 0.5625, which DLinear beats in 4/16 folds. **Diagnostic**: the shortfall is a **down-bias**, not an inverted signal — the model calls up 45.7% of the time against a 56.25% up rate, an information-free model at that call rate scores 0.4893, inverting scores 0.5245 and still misses the bar, and the residual −1.4 points is not significant (t ≈ −1.5). **`batch_size` re-check ran and reverted the setting to 64**: mini-batch wins return, Sharpe and direction; full batch wins MAE 16/16 and nothing else. 29 tests; suite at 543. |
| GB-21 | 17 Aug 2026 | Ben | **Complete.** `engine/risk.py` — `size_positions` for the live path and `position_sizer` for the backtester, **one arithmetic behind two shapes** (`room_for`), so a backtest and a live account cannot size differently. **Three caps, tightest wins:** per-position, gross exposure counting what is already open, and **available cash** — the last is arithmetic rather than risk and lives here because the live executor has no guard of its own. Sizing may reject or shrink an order and **may never create one**; `hold` and `exit` produce nothing and a repeated symbol is sized once. Hypothesis property tests over randomised equity, prices, ranked signal sets **and randomised caps**, including the strongest form available: the engine's own `_require_sizeable`, written in GB-18 before this module existed, is run over generated accounts — so "the sizer satisfies the engine" is a proof rather than an agreement. 18 tests. **Measured against GB-24's placeholder over 16 folds:** mean return +0.0044 → +0.0045, Sharpe +0.655 → +0.690, trades 272 → 270. It barely moved, and the reason is worth recording: **5 symbols × `max_position_pct` 0.10 = 0.50 = `max_gross_exposure`**, so the gross cap is numerically redundant with the per-position cap at this universe size and binds only when open positions appreciate — it did so on one fold of 16. The risk layer is correct and is not yet constraining anything the placeholder was not. |
| GB-24 | 17 Aug 2026 | Ben | **Complete.** `python -m glassbox.smoke_offline` runs the whole offline path in one command — cache → features → folds → training statistics on train only → DLinear → threshold calibration on validation → forecast → backtest → metrics table with persistence beside it. `--folds N` (default 1, taken from the start of the fold list so the default names the same fold every run) and `--model {persistence,dlinear}`. **Offline by construction:** the parquet files are checked first and the run aborts with exit 2 naming every missing symbol and the command that creates them, because `load_history` would otherwise download and the claim "the offline path works" would quietly become "…when yfinance is up". **The baseline is not drawn, it is run:** persistence goes through the same train → calibrate → decide → backtest path, forecasts zero, and therefore stands aside on 16/16 folds — a flat curve produced by the pipeline rather than by the reporting layer. **Ruling: a fold that stands aside is a labelled row and counts in the return mean** (its return is a true zero; averaging only the folds where the strategy chose to act is selection on the strategy's own decision), while the Sharpe mean excludes it by construction — both counts printed. **Ruling: `dir_ref` sits in the column immediately right of `direction`**, with a legend naming it as the always-long bar and ruling out both 0.5 and persistence. Measured: **fold 1 in 10.1s, all 16 folds in 31.1s** on a laptop with no network, against a five-minute budget. Sizing was a config-driven placeholder, **replaced by GB-21 the same day**. 16 tests; suite at 559. **Extended with GB-21:** a **buy-and-hold arm**, run through the same backtester — equal weight, entered at the first tradeable open, held to the last close, no stop and no target (expressed as infinite fractions rather than by special-casing the engine). It exists because persistence stands aside 16/16, which made the return and Sharpe columns **deltas against cash**; measured, buy-and-hold returns **+7.37% per fold against DLinear's +0.45%**, Sharpe **1.48 against 0.69**. Its direction accuracy **is** the always-long bar by construction and the run asserts the two agree on every fold — a self-check on the bar under every direction number in the study. 23 tests. |
| GB-25 | 17 Aug 2026 | Ben | **Complete — GATE 1 PASSED.** Four checklist items green (one qualified: persistence produces forecast numbers on every fold, and its trading numbers are degenerate by construction, which is why buy-and-hold exists). Leakage audit run over all 16 folds and 5 symbols, and treated as the work rather than a formality: scaler `fitted_start/end` equals the training range **exactly in 80/80 symbol-folds**; calibration re-run **with test bars present in the frames** leaves the chosen band unchanged, so the caller's slice is not the only thing keeping test data out; zero split overlaps and an embargo of **exactly H = 4 bars at all 32 boundaries**; aggregate Sharpe **+0.6895** against the amended §7.3 alarm, with **5 folds above +2 and 3 below −2** — the symmetry the rule asks for. The audit's own first run reported a failure that turned out to be a bug in the audit, not the code; it is recorded in the gate log rather than quietly re-run. Full detail in the gate log below. |

## Sprint 3 — Live End-to-End · 12–25 Sep 2026 → GATE 2

| Task | Date | Owner | What was built |
|---|---|---|---|
| GB-22 | 17 Aug 2026 | Ben | **Complete.** `engine/executor.py` — the Alpaca SDK behind a five-call `Broker` protocol (`submit_market_order`, `submit_stop_order`, `submit_limit_order`, `cancel_order`, `get_orders`, `get_positions`, `get_account`), so `tests/fake_broker.py` is a **complete** substitute and the suite never needs credentials, a network or a market session. `execute` converts through `engine.risk.shares_for` — GB-18's single conversion — logs every request and response against the **decision ID**, and honours `live.mode`: `auto` submits, `co_pilot` returns a pending recommendation and sends nothing (GB-37 approves it). **Two verifications against the live paper API, neither taken from memory, both of which changed the code.** (1) `MIN_SHARES = 0.001` was the right idea in the wrong unit: Alpaca enforces a **$1.00 minimum notional** (`cost basis must be >= minimal amount of order 1`), so the constant is now `MIN_ORDER_NOTIONAL`; it also **silently truncates a quantity to nine decimals**, so the conversion floors to the same precision. (2) **Alpaca refuses a bracket on a fractional quantity** (`fractional orders must be simple orders`, and the same bracket on one whole share is accepted), so the backtester's attached stop and target **cannot be expressed as one live order**; a held fractional position can be protected by a **standalone stop plus a standalone limit**, which is what `protect` does, at the cost of no OCO linkage, day-only expiry needing a re-arm each session, and a live stop that is not always present where the backtest's is. The GB-21 property test caught a real numerical flaw while this landed — `floor(x * 1e9) / 1e9` is not the floor once the scaled value leaves float64's exact integer range — now `Decimal`. 18 tests; suite at 605. |
| GB-23 | 18 Aug 2026 | Ben | **Complete.** `engine/reconcile.py` (new, spec §3.4) — **reconcile-from-truth, not an order-lifecycle state machine**, per `SOLO_BUILD_PLAN.md` §2's cut. Positions and open orders are fetched every cycle, compared against a persisted `Book`, and **every divergence resolves in favour of the broker** and is logged at WARNING. **Three rulings.** An unexplained broker position is **quarantined** — recorded and counted against buying power, never given protective levels, never traded: adopting it would manage a position with no entry price, stop or decision behind it, and ignoring it would let the sizer over-commit the account. A local position the broker does not have is **dropped**, which is also exactly what a filled stop looks like. A quantity mismatch **takes the broker's number** and keeps the local provenance, which is how a partial fill is absorbed. A fourth check, `MISSING_PROTECTION`, is **detection only** — the input GB-26 rule 3 acts on; a reconciler that submitted or cancelled would be the state machine this task was scoped away from. **Acceptance, against the real paper account:** a cycle killed mid-flight with `os._exit(9)` left the book on disk untouched; a hand-desynchronised book claiming 1.0 MSFT and 0.5 AAPL was corrected in one cycle to the broker's actual `{AAPL: 0.01}` with all three divergence kinds reported, and the corrected book reconciled clean on the next pass. **Limit of the scope, found during that run:** reconciliation corrects existence and quantity but **cannot correct provenance** — the broker has none to offer, so a book that lies about *why* it holds something keeps the lie. 21 tests; suite at 626. |
| GB-27 | 18 Aug 2026 | Ben | **Complete — and it did its job by failing.** `tests/features/test_train_live_parity.py` builds the model input window twice — the training path over full history, and the live path from a bar **tail** re-normalised through `normalise_bars` with Alpaca provenance — and asserts `np.array_equal`, no tolerance. **The first version passed on AAPL at one timestamp. Swept over five symbols and twenty-five timestamps it failed 93 of 125**, and the cause was the declared floor, not the test: GB-9's warm-up bounded the RSI seed's *weight* below 1e-7, which is sufficient only if the seed *difference* is at most 1, and it is a difference of average gains in price units. Near RSI 50 the residual (3.815e-06) is the same order as a float32 ulp (5.95e-06) — **352 was marginal by construction**. **Floor corrected to 445**, derived from a 1e-10 target: `(13/14)^311`, warm-up 325. Measured: 352 → 32/125, 414 (the 1e-9 target) → 120/125, **445 → 125/125**. The 1e-9 case is a standing test, so the margin is measured rather than assumed. The test now **sweeps by construction** — a single-point parity test is what produced the error. It also pins the teeth: a one-bar offset breaks parity, a bar *after* the window cannot change it (GB-10 causality — an earlier assertion had this backwards), too little history **raises** rather than differing quietly, a spliced two-source frame is refused, and the per-window `symbols` tuple survives pooling into a five-symbol batch. Corrections landed in place, as corrections, in spec §7.2 and the GB-9/GB-27 rows, `ARCHITECTURE.md`, `README.md` and the GB-9 DECISIONS entry. 16 tests; suite at 643. |
| GB-27b | 18 Aug 2026 | Ben | **Complete.** Three rulings from the parity-floor post-mortem. **(1) `tests/sweep.py`**, beside `causality.py`: `sweep()` for "X holds", `contest()` for "A beats B", with two properties that are refusals rather than conventions — a `SweepResult` **cannot be used as a boolean** (`bool(result)` raises and names what to read instead), and a **sweep of one cell is refused** with an error citing the floor. Held to `causality.py`'s standard: the decisive test replays the 352-bar measurement on real data. 19 tests. **(2) `RSI_WARMUP` deleted and unified** with the parity warm-up at 325 (`RSI_SEED_TOLERANCE` 1e-2 → 1e-10); `PARITY_WARMUP["rsi14"]` now *imports* it, so there is one derivation rather than two constants to keep in step. **Cost measured, and not quite zero:** feature rows 2591 → 2343, first row 2016-04-25 → 2017-04-19, folds still 16 and no kept fold reads a discarded bar — but `make_folds` anchors its month grid on the feature frame's start, so the grid shifted about a week (fold 1 training 2020-04-13 → 2020-04-06) and individual val/test counts move by one or two. **Every previously measured number therefore shifts slightly**; the headline figures were re-measured rather than carried over: direction **0.5182** vs an always-long bar of **0.5560** (4/16), return **+0.40%** per fold vs buy-and-hold's **+7.73%**, Sharpe **+0.62** vs **1.39**, 3/16 folds standing aside. The conclusion is unchanged: the strategy beats neither reference. **A consequence worth more than the tidiness:** a tail below `min_history_bars` no longer assembles a window at all — the builder **refuses** — so the silent below-floor divergence is unreachable through the public path. The cost is that the 414-vs-445 measurement can no longer be reproduced live; it stands as a recorded result and the below-floor tests assert the refusal. **(3) GB-13's initialisation study re-swept** through the new harness: 3 arms × 16 folds × 5 symbols = 80 cells, under the configured `batch_size: 64`. **The ranking survives and its reason changes.** Paper `1/√L` wins **0/80** on MAE and **0/80** on legibility — that half is confirmed and strengthened. But zeros and the fan-in correction are a **tie on MAE** (2.190 vs 2.195, and fan-in wins more cells, 46 to 34): the original 1.94× vs 1.96× was always that tie, and one fold made it look like an order. Zeros wins **direction** (0.518 vs 0.470/0.475, 41/80) and **legibility unanimously, 80/80** (largest contribution 0.065 vs 0.450 vs 1.111). The single-fold *levels* do not reproduce; the single-fold *legibility* numbers do, almost exactly. GB-57 now carries the swept table and must not quote the old figures. |
| GB-28 | 18 Aug 2026 | Ben | **Complete.** `engine/rank.py` — order by descending `trend_strength`, **break ties alphabetically**, take `signal.top_k`. The tie-break is the whole design: Python's sort is stable, so without a second key a tie resolves by the caller's insertion order, which in the live loop is the order symbols came back from a broker call — two identical days would select different names and GB-32's replay would not reproduce. Negating the strength rather than `reverse=True`, because reversing would also reverse the tie-break and run the alphabet backwards. Only `enter_long` competes: **an exit is an obligation on capital already committed** and must never be crowded out. A non-finite strength is refused rather than sorted, since NaN compares false against everything and ordering it is silently order-dependent. 14 tests. |
| GB-29 | 18 Aug 2026 | Ben | **Complete.** `glassbox/records.py` (new, spec §3.4) — `DecisionRecord` persisted as JSONL under `decisions/YYYY-MM.jsonl` with the `config_hash`, `save_decision` / `load_decisions(start, end)`, numpy arrays as lists, timestamps as ISO-8601 UTC, lossless round trip including float32 dtype. A corrupt line is logged and skipped rather than costing the month — a session killed mid-write leaves exactly that. Bounds are inclusive and a bare end date covers its whole day, because "August 1st to 31st" excluding the 31st would be a trap. **Plus the live trade log**: `emit_trades` builds `Trade` records from the broker's filled-order history, `exit_reason` from **which leg filled** (by `client_order_id` suffix, not by price — two levels can sit close together) with `stop_gap`/`target_gap` decided by the fill price, **costs from the actual fills rather than the configured bps**, and `realised_slippage_bps` for GB-57's comparison against the modelled 6.0 bps. **Nothing is emitted for a quarantined position** — no entry basis, no invented number. **Ruling: a live `Trade` and a backtest `Trade` are one type**, moved to `contracts/schemas.py` (§4.2 addition) with two optional broker-order-ID fields; the layer contract forced it, since the live path may not import the harness and GB-19 must read both with no translation layer. `glassbox.records` is named in the forbidden-import contract so the temptation is mechanically unavailable. 27 tests; suite at 705. |
| GB-30 | 18 Aug 2026 | Ben | **Complete.** `explain/channel.py` — `attribute`, `shares`, `cancellation`. **The one decomposition lives on the schema**, as `Attribution.from_terms` (Ben's ruling): `explain` sits above `model` in §3.1, so "move the logic up and delegate" would have needed a cycle and an `ignore_imports` exception; putting the summation on `Attribution` costs neither and makes exactness **structural** — every attribution in the system is built by the function that refuses a residual, and the refusal names an intercept and a re-added normalisation as the causes to check. `forecast_total` now comes from `predict` rather than from the contributions' own sum, which turns §4.4's property 4 from an identity into a real check; the residue is the float32 cast, **5.005e-08 worst over 1,000 random windows, a 200× margin**. `attribute` re-checks the model's total against `predict` itself, because `from_terms` can only prove the parts sum to the total the model *reported*. **Shares are of the gross, not the net**: +57%/−43% rather than 400%/−300%, bounded, signed, with the cancellation reported separately (0.143 on that window). **A zero forecast is defined** — all-zero contributions give all-zero shares, cancelling contributions keep their shares and read a cancellation of 0.0. Persistence now reports **no terms** per channel instead of hand-written zeros. A syntax-tree test asserts no perturbation library is importable from `explain/`, which nothing enforced before. 26 + 9 tests; suite at 731. |
| GB-26 pre-work | 18 Aug 2026 | Ben | **Both blockers closed before the live loop.** **(1) The feed is asserted.** `LIVE_FEED`/`LIVE_ADJUSTMENT` are named constants, `data_source()` renders them, and `load_live_bars` logs that line on **every fetch** rather than only at session start. Three tests, and the one that matters asserts the **request object handed to the SDK**, since a constant can exist and go unused. The cost this backstops was measured, not imagined: an IEX downgrade is worth up to 193 bps against a 1 bp tolerance, and every schema test would still pass. **(2) The lookback and the guard are the caller's floor.** `load_live_bars(symbols, min_bars, *, requirement, lookback_days)` — `cfg` is gone, because the only thing it supplied was `input_len`, which is the wrong number. The refusal names bars received, calendar days requested, the floor and the channel behind it, via new `builder.history_requirement` / `deepest_warmup_channel`: `data` sits below `features` and cannot compute the floor, so the layer that knows writes the sentence and the caller carries it down. A test counts **real NYSE sessions** in the default window sliding across a decade and asserts the **worst** case still clears 445. 8 tests. |
| GB-32 | 18 Aug 2026 | Ben | **Complete.** `explain/narrate.py` — attribution and signal to prose, English and Hebrew. Three parts in a fixed order: what was forecast, which channels produced it ranked by share, and what was done with size and stop. **Every number is read off the objects, never recomputed**, so the prose and the dashboard cannot disagree by a rounding step. **The cancellation is spoken**: below `OFFSETTING_BELOW = 0.5` — the point where the opposing side reaches a third of the leading one — the narrative says the channels largely offset and names the surviving fraction. **A hold names its band** and separates the three causes; saying "the threshold was not met" when the band admitted the forecast and the path did not confirm it would be false. **Persistence renders as no directional call**, and a model whose channels *cancel* to zero gets a different sentence, because that zero is a view. **No profitability claim in either language**, asserted against a word list. **Bidi:** direction is metadata on the returned `Narrative`, never sniffed from the first strong character (every sentence opens with a ticker); every LTR run is wrapped in U+2066/U+2069 **isolates** rather than LRE/PDF or LRM, because only an isolate cannot influence the level around it; an atom is the whole unit including its sign and percent, so a percentage cannot be reordered away from its label. Tested structurally — no Latin or numeric run outside an isolate, every isolate balanced and never nested. 42 tests; suite at 782. |
| GB-26 | 18 Aug 2026 | Noy | **Complete.** `live_loop.py` — spec §3.5's seven steps every `poll_seconds`, with three more before them because **entries come after protection**: emit trades, reconcile, then verify and re-arm both legs. **Trades are emitted before reconciliation** — a filled stop makes the position vanish and `reconcile` correctly drops the holding, which is the only record of the entry basis. **Two rulings, both reported:** a **stale symbol loses its entry and keeps its stop** (*a stale window may never justify opening risk and may never suspend the management of risk already taken*), named in the log and the report because `top_k` is taken across the universe; and **the first cycle may reduce risk and may not add any** — no entry, exits unaffected, because rule 2 says re-arm before entries and the first cycle is that moment, because the book was just rebuilt from one observation, and because a mid-session restart must not double an entry the previous process already submitted. **The session window is the exchange calendar's**, so a holiday closes the loop and a half-day closes it early. `--dry-run` is a **wrapping broker**, not a skipped call, so it exercises `execute`, `protect` and the fill poll against real positions. A missing or stood-aside band is `Thresholds.never()` — never a default, because a made-up threshold makes every decision in the session unexplainable. `smoke_offline --prepare-live DIR` writes the checkpoint and band, since the loop may not import the harness. **`ISRAEL_TZ` is a flagged rule-5 bend**, justified by `market_open_il`'s own field name. 26 tests; suite at 809. |
| GB-33 | 18 Aug 2026 | Ben | **Complete — closed without a new test.** §4.4's exactness properties are already parameterised over `model.ALL_FORECASTERS` and run over **1,000 random windows per forecaster**, and GB-30 added a second independent pass through `explain.channel.attribute` over the same registry. Both have teeth (`MisreportingForecaster`, `NonAdditiveForecaster`). **The registry is the integration point** — GB-41 adds one line and edits no test. **One constraint handed to GB-41:** the contract asserts `tuple(per_channel) == batch.channels` and §6.4 makes FITS univariate, so FITS must name **every** active channel with `0.0` for the ones it does not consume. |
| GB-34/35/36 | 18 Aug 2026 | Noy | **Complete.** `dashboard/app.py` — positions with live PnL and quarantine state, equity, bot status, the active model/channels/config hash; forecast paths with the calibrated threshold; the decision log with narrative and per-channel contribution bars. **Chrome and data are different colour families** (Ben's ruling): orange for labels, rules, borders and the threshold; the blue spectral ramp for data, dark for slow and light for fast, **assigned by channel speed rather than config order** because a ramp encodes a quantity. **Sign is geometry and a glyph, never hue** — a ramp cannot encode a sign without a second data colour — so positive is filled and right, negative hollow and left, every figure prefixed ▲/▼, and a test asserts no traffic-light pair anywhere. **Charts are hand-built SVG**: the design language is dashed hairlines and numbered rulers, which a chart library fights, and a pure string builder makes the threshold, the ramp and the bar geometry testable without a browser or a new dependency. **`STOOD ASIDE` outranks every other status**, and a `never()` band is **said on the chart rather than omitted** — omitting it makes abstaining look like waiting. **The cancellation is always shown**, in the accent below half. A Hebrew narrative gets an explicit `dir`, not `dir="auto"`. The reliability panel reads `reliability.json`, written by `--prepare-live`: direction **0.5182** against an always-long bar of **0.5560**, beaten in **4 of 16** folds — measured, so it cannot go stale, and `NOT MEASURED` when absent. 26 tests; suite at 835. |
| GB-22 closed | 18 Aug 2026 | Ben | **Acceptance met.** Order `385e982a…` **FILLED 0.081919619 AAPL at 307.49**, at **13:30:02.113 UTC — two seconds after the open**, which is exactly the next-open fill GB-18's ruling 2 models. **Protection did not arm, and every step behaved as designed:** `protect` returned empty because the entry had not filled at submission, and GB-26's rule 1 arms in the cycle that sees the fill — but the order was submitted by a one-off script, so no `Holding` exists and reconciliation **quarantined** the merged 0.091919619 position. A quarantined position is never protected, by design. **The fact worth stating: the only orders this system protects are the ones it decided.** |
| GB-37/38 | 18 Aug 2026 | Noy | **Complete.** Co-Pilot queues the explained recommendation instead of executing it; the dashboard offers Approve and Reject; **both answers write a decision record**, because a rejection that left no trace would make the log a record of what the system wanted rather than what happened. **No expiry timer** — a timer is the system deciding "no" and recording nothing. `approve` does not re-check `live.mode`, since the mode asks the question and this is the answer; answering twice is refused rather than resubmitted. **Replay drives the live loop's own cycle** — `LiveState.fetch` is injectable, so every other step is the identical code — over **fold 13**, the only fold of 16 containing all five exit kinds. Its +2.71% return is a coincidence of that criterion and is recorded as such, with the full return distribution and **fold 1 (−3.59%) named as the counterweight**; the fold is a CLI parameter, not a constant. **One decision store**, with `DecisionRecord.provenance` naming the fold rather than setting a flag, and `load_decisions` **defaulting to live** — so a forgotten filter hides replayed decisions rather than passing them off as real. `ReplayBroker` is a demo instrument, not a backtester, and its PnL is never a result. Measured: 14 bars → 70 records, 2 recommendations, one approved with both legs armed and one declined leaving the broker untouched. 21 tests; suite at 856. |

## Sprint 4 — FITS, Study, Report · 26 Sep – 10 Oct 2026 → GATE 3

_not started_

---

## Gate log

| Gate | Date | Result | Notes |
|---|---|---|---|
| GATE 1 | 17 Aug 2026 (commitment 11 Sep) | **PASS** | Four checklist items green, four leakage checks green, one qualified. Detail below. |
| GATE 2 | ~1–3 Sep (commitment 25 Sep) | pending | Needs 2–3 real market sessions; see `SOLO_BUILD_PLAN.md` §4.1 |
| GATE 3 | 10 Oct 2026 | pending | **Unchanged.** The submission date does not move. |

### GATE 1 — 17 Aug 2026 · PASS · GB-25

**Checklist (spec §8, `SOLO_BUILD_PLAN.md` §7).**

| Item | Result | Evidence |
|---|---|---|
| `python -m glassbox.smoke_offline` runs data → features → DLinear → backtest → metrics | **GREEN** | One command, no network, no manual step. 8.4s for one fold, 39.3s for 16, against a five-minute budget. |
| Persistence produces numbers on the same folds | **GREEN, qualified** | It produces **forecast** numbers on every fold — MAE 0.0191 on fold 1, and it is the reference for MAE and RMSE. Its **trading** numbers are degenerate by construction: it forecasts zero, no band can fire, so it stands aside 16/16 with return 0.0000 and no Sharpe. That is correct behaviour and it is *why* the buy-and-hold arm was added. Ticking this item without the qualification would misdescribe what the baseline does. |
| `test_no_lookahead.py` and `test_forecaster_contract.py` green | **GREEN** | 54 passed. Paths are `tests/features/test_no_lookahead.py` and `tests/model/test_forecaster_contract.py`; the checklist names them without directories. |
| At least one walk-forward fold completes end to end | **GREEN** | 16 of 16 complete: train → calibrate on validation → forecast → backtest → metrics, for three arms each. |

**Leakage audit** — the real work of GB-25, run over all 16 folds and all 5 symbols.

| Check | Result | Evidence |
|---|---|---|
| Normalisation statistics fitted on training rows only | **GREEN** | `ChannelStats.fitted_start/end` equals the fold's training range **exactly**, in **80/80** symbol-folds — an identity, not an inequality, because GB-15 excludes the embargoed tail so the comparison is exact. Every `fitted_end` precedes its fold's test start by ≥ 97 days. |
| Signal thresholds calibrated on validation only | **GREEN** | Calibration re-run with the **test bars present in the price frames**: the chosen band is unchanged on every fold tried, so the caller's slice is not the only thing keeping test data out. Every forecast handed to calibration ends inside its own validation range (285 per fold, none outside). `assert_fit_isolated` covers the same property in the suite under both perturbation modes. |
| No fold has overlapping train/test timestamps; the embargo holds | **GREEN** | 16 folds, zero overlaps between any pair of splits. The gap between adjacent splits is **exactly 4 bars = H** in every one of the 32 boundaries checked. |
| Aggregate Sharpe against the amended §7.3 rule | **GREEN** | Aggregate **+0.6895** over the 14 folds where it is defined, far below the 2.0 alarm. Per-fold spread −3.263 → +4.553, sd 2.323, **5 folds above +2.0 and 3 below −2.0** — the symmetry the amended rule asks for, and what a true Sharpe of zero produces at this sample size. |

**Nothing is red.** Four limitations are recorded rather than hidden, none of them gating:

1. **Validation is used twice** — for early stopping and for threshold calibration. This does not leak into test, but it means the *validation* Sharpe (up to 5.0) is a selected maximum and is not reportable. Already true of every walk-forward study that tunes anything; GB-57 must say so. **And it must say which way the bias runs, because that is what makes it a caveat rather than a reason to discount the numbers:** thresholds are calibrated on forecasts already slightly overfit to validation, so on test they are **miscalibrated rather than inflated**. The double use **degrades** test performance; it does not flatter it. Any test figure reported here is therefore a **lower bound** with respect to this particular flaw.
2. **The strategy does not beat either reference.** Direction 0.5071 against an always-long bar of 0.5625; return +0.45% per fold against buy-and-hold's +7.37%. GATE 1 asks whether the slice is real, not whether it is profitable — but the numbers are the numbers.
3. **The risk layer is correct, enforced and currently inert** (see GB-21).
4. **The smoke command retrains per run** rather than loading GB-15's checkpoints. Not a gate item; it costs 39s over 16 folds, so there is no pressure to change it.

---

## Single-point claims audit (GB-27, 18 Aug 2026)

**Why this list exists.** The 352-bar parity floor entered the spec marked *verified* on
the strength of one symbol at one timestamp, and a sweep found it held in 32 of 125
symbol-timestamp pairs. Every other numerical claim marked verified or measured was checked
for the same failure mode. **Nothing below has been re-verified** — this is the list, as
asked, so that the ones most likely to be wrong in the same way are known before GB-57
cites them.

### Triage rule (adopted 18 Aug 2026)

**Mechanical or identity claims held at one point because they are algebra; statistical
claims did not.** GB-27b is the evidence: the single-fold *legibility* numbers reproduced
almost exactly under an 80-cell sweep (1.39 / 0.43 / 0.058 → 1.02 / 0.41 / 0.063) while the
single-fold *accuracy* numbers did not (4.00× / 1.96× / 1.94× → 2.69 / 2.19 / 2.19). So:
confirm mechanical claims are **asserted somewhere in the suite** and move on; **re-sweep**
anything statistical that GB-57 will quote.

### Mechanical — confirmed asserted, not re-swept

| Claim | Asserted in |
|---|---|
| Round trip on a flat trade is exactly 6.0 bps | `tests/backtest/test_engine.py` (`round_trip_cost_bps`) |
| Flat book ⇒ final equity = initial cash + Σ net_pnl | `engine._assert_accounted`, every backtest, plus `test_engine.py` |
| `Σ per_channel == forecast` within 1e-5 | `tests/model/test_forecaster_contract.py`, `test_ltsf.py` |
| Buy-and-hold's direction accuracy **is** the always-long bar | runtime `_assert_is_the_bar` + `tests/test_smoke_offline.py` |
| Live and historical share one schema definition | `normalise_bars`, asserted in `test_historical.py` and the parity sweep |
| Alpaca's fractional refusals ($1 notional, simple-orders-only, 9 decimals) | `tests/fake_broker.py` mirrors them; `tests/engine/test_executor.py` asserts them |
| Causality: perturbing the future leaves the prefix bit-identical | `tests/features/test_no_lookahead.py`, both modes, three splits |
| RAW vs adjusted differs by a split factor | mechanical (a corporate action), pinned in `data/live.py`'s docstring |

### Statistical — re-swept

| Claim | Result |
|---|---|
| GB-13 initialisation ranking | **Re-swept, 80 cells.** Ranking survives, its reason changes: zeros wins direction and legibility (80/80), ties fan-in on MAE. See the GB-27b row. |
| GB-25 decomposition (β, R², timing residual) | **Re-swept with intervals.** β 0.0919 (t = 3.29) ≈ exposure 0.0892, α −0.31% (t = −0.65, CI includes zero), timing −51.8 bps (t = −1.79). Structure robust across both fold grids; every level moved. |
| **r = −0.47, timing vs market** — *introduced by review, not by implementation* | **Withdrawn.** Ben withdrew it on the triage rule (n = 16, CI [−0.783, +0.034] includes zero); re-measured on the shifted grid it is **r = −0.0025**, CI [−0.498, +0.494]. It vanished rather than weakened. The "independent corroboration" framing goes with it. |

### Statistical — deferred, with the cost stated

`GB-7`'s price/volume agreement (<1 bp prices, 34–111 bps volume) rests on **one 163-bar
window** across five symbols. Re-sweeping needs live Alpaca data across a longer and more
stressed period; this account's SIP entitlement returns 403 on recent data, so the sweep
would run on IEX and measure a different question. **Not run — it would cost more than an
hour and would not answer the claim as stated.**

`RSI_WARMUP = 77` left this list a different way: GB-27b **deleted it**, unifying it with
the parity warm-up. A claim with a known-bad derivation was removed rather than re-verified.

### Established on a single symbol, fold, timestamp or machine

| Claim | Sample | Risk | Consequence if wrong |
|---|---|---|---|
| `min_history_bars` = 352, "verified byte-identical" (GB-9) | AAPL, 1 timestamp | **realised** | Corrected 18 Aug to 445. This is the exemplar. |
| **Initialisation study** — MAE 4.00× vs 1.94×, direction 0.344 vs 0.557, largest contribution 1.39 vs 0.058 (GB-13) | **walk-forward fold 1, one symbol** | **high** | GB-57 reports the *comparison between initialisations* as a methodological finding. The levels are already flagged as single-fold (GB-15's 80-arm rerun gave 0.4971 against the 0.557), but the **ranking** rests on one fold of one symbol and has never been swept. |
| `RSI_WARMUP` = 77, "computed rather than chosen" (GB-8) | derivation, no sweep | **medium** | **Same derivation flaw as 352**: it bounds the seed's *weight* below 1%, not weight × seed difference. Its consequence is contained — it decides only when RSI stops emitting NaN, and parity is now governed by the separate 325-bar warm-up — but the reasoning is the one just shown to be insufficient. |
| Only `Adjustment.ALL` matches yfinance; RAW is off by a split factor (GB-7) | **AAPL, one 4:1 split, 2020-08-31** | low | Structural (a split factor), not statistical — but "only ALL matches" is verified on one corporate action of one symbol. A dividend-heavy symbol was never checked. |
| SIP returns 2669 bars to 2016, IEX 1521 to 2020 (GB-7) | one account, one query | low | An account/subscription property, not a market one. |
| Price agreement <1 bp, volume 34–111 bps (GB-7) | 5 symbols, **one 163-bar window** | low–medium | Multi-symbol but a single recent window; a stressed period was never sampled. |
| Alpaca's fractional rules — $1 notional floor, `fractional orders must be simple orders`, 9-decimal truncation (GB-22) | **AAPL only** | low–medium | Venue rules *should* be symbol-independent, but `fractionable` is a per-asset flag, so all four rejections were observed on one asset. |
| Predict throughput — 305 windows in 24.6 ms, 0.88 ms per window (GB-16) | one machine, one run | negligible | Performance only; nothing depends on it. |
| Smoke wall time — 8.4 s for one fold, 39.3 s for 16 (GB-24) | one machine | negligible | Performance only. |
| Causality: `scale` passes a leaked direction, `shuffle` catches it (GB-10) | one synthetic series; the real-data check was AAPL | low | The synthetic series is adversarial by construction (sign changes often), arguably stronger than one real symbol — but it is still one series. |

### Closed since the audit was written

**GB-7's price/volume agreement — re-swept 18 Aug and confirmed.** It was recorded as
deferred on the premise that this account "returns 403 on recent SIP". That premise was
wrong: what the account refuses is the *recent-data* endpoints (`get_stock_latest_bar`,
`get_stock_latest_quote`), while historical daily bars on SIP are served, including the most
recent completed session. Re-swept over **273 sessions × 5 symbols = 1,365 bar-comparisons**
against GB-7's single 163-bar window: worst price deviation **0.56 bps**, so the 1 bp
tolerance holds; volume's worst is **213.6 bps**, so GB-7's stated 34–111 bps range is
widened to **34–214**. IEX, had the live loop ever used it, would be out by up to **193 bps**
on price — 190× the tolerance. See DECISIONS, 18 Aug.

**The r = −0.47 correlation — withdrawn**, and every headline claim now runs at three
fold-grid anchors (DECISIONS, 18 Aug). Direction against the always-long bar holds at all
three; β tracks measured exposure at all three but is significant at only two.

### Established on a sweep, and sound as stated

`13,309` entry resolutions across five symbols for GB-18's four pricing rules, with the
gap rule re-measured through the finished engine (15.6% → 14.16%); GB-15's early-stopping
result on **80/80 arms**; the batch-size decision on 80 arms then 16 folds, and GB-20's
re-check on 16 folds; GB-19/GB-20's direction bar and down-bias diagnostic on 16 folds ×
5 symbols; GB-21's sizer before/after on 16 folds; GB-25's exposure decomposition on 16
folds; GB-5's data-quality report on 5 symbols × 2668 bars; GB-27's own floor on 125
symbol-timestamp pairs.

**The pattern worth naming:** every claim in the sound list came from a task that had a
harness to sweep with. Every claim in the risky list came from a task that did not, and the
measurement was taken by hand on whatever was in front of it. The fix is not more
discipline — it is that a claim entering the spec should name its sample size in the same
sentence as its number.

---

## Open questions

_Claude Code: write blocking questions here rather than guessing._

- **GB-22 → GB-26: when is protection armed, and what covers the gap?** Alpaca refuses a
  bracket on a fractional quantity, so a stop and a target are two standalone **day** orders
  that can only be armed once the entry has **filled**. An order decided after the close
  fills at the next open, so the position is unprotected from the open until the cycle that
  sees the fill arms it, and every position is unprotected overnight because day orders
  expire at the close. Three candidate policies, none of them free: (a) arm at the next
  poll, accepting a gap of one cycle; (b) poll until filled before the loop proceeds,
  blocking the cycle; (c) round to whole shares so a real bracket becomes legal, which
  reintroduces the price-level discretisation GB-18 chose fractional sizing to avoid. This
  needs a ruling before GB-26 wires the loop.

_Both GB-30 feed items are **closed** — Ben ruled them blockers for GB-26 and they were
fixed on 18 Aug: the feed is asserted on the request object and logged on every fetch, and
`load_live_bars` now takes the caller's floor. See DECISIONS, 18 Aug._

_The 14 Aug CI failure is resolved — see DECISIONS.md, "Python floor raised to 3.12";
green on 176fe01._

_None open._
