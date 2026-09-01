# PROGRESS

The running log of what is built. **Claude Code reads this file to locate the current phase.**
Append one line per completed task. Newest at the bottom of each sprint.

---

## Current state

**Sprint:** 1 — Foundations · **complete, 14 Aug 2026** (12/12, day 6 of 14)
**Sprint 2:** **complete** — GB-17, GB-18, GB-13 on 16 Aug; GB-18 fix, GB-15, the
`WindowBatch.symbols` contract change, GB-16, GB-19, GB-20, GB-24, GB-21 and GB-25 on
17 Aug
**Sprint 3:** **complete** — GB-26, GB-32, GB-33, GB-34/35/36, GB-37, GB-38, GB-39 and
GB-40. **GATE 2 walked on 20 Aug, five weeks early: 3 of 6 green, not passed, FITS not
cancelled.** See the gate log.
**Sprint 4 is under way:** GB-42, GB-41 and GB-44 done, in that order, as §6.3 requires.
FITS is registered, passes the contract test with no edit to it, and is selected by
`model.active` alone — one shared model across the five symbols, asserted by perturbing the
shared weight.
**The first head-to-head is measured and it is negative.** Over 16 folds neither model beats
persistence on MAE (16/16 against both) and neither beats the always-long bar on direction —
DLinear **0.5182**, FITS **0.5098**, bar **0.5560**, each above it in 4 folds of 16.
**Both conditioning rulings are implemented.** The target is scaled (ruling 1) and MAE now
travels with `flatness` (ruling 2). **One new ruling is open**, and it is a consequence of
the first: the deployed fold no longer stands aside, so §7's required "the deployed system
declines to trade" finding is true of the 18–20 Aug deployment and not of the current one.
**GB-45, GB-46, GB-47 and GB-48 are done.** §6.5's deliverables exist — the per-frequency
view, gain and phase in days, and the figure at `figures/frequency_response.png` — and
`C2_hybrid` builds, so the study's hybrid arm is reachable. **The figure shows the
interpolation cost, not a discovery about equity cycles**: ~86% of the response is
reproduced by a model trained on white noise, and gain tracks grid misalignment at r = −0.90.
**GATE 2 criterion 2 no longer needs a rehearsal**: the deployed band fires, so an ordinary
live session meets it, which is better. The rehearsal path stays for the next fold that
stands aside.
**Next task:** GB-50 (the COF sweep) and GB-51 (paired Wilcoxon), then GB-52's report,
which regenerates from `results.csv` alone.
**Criterion 6 needs two or three separate sessions and only Wednesday to Friday are
available.** Still open for Ben: DAY versus GTC on the protective legs, outside a
rehearsal.
**The account is flat** — the quarantined 0.0919 AAPL was sold at 310.394 on 18 Aug on
instruction, and the following cycle's reconciliation logged the drop. GB-26 is complete and ran a full dry-run cycle
against the live SIP feed on 18 Aug: ten steps, 610 bars a symbol against a floor of 445,
the 0.01 AAPL probe quarantined, five decisions recorded and narrated, nothing submitted.
**The session on 18–20 Aug would not trade, and that was the correct outcome:** the most
recent complete fold **stood aside** — its best calibrated candidate scored a validation
Sharpe of **−3.38** over 7 trades — so `Thresholds.never()` was in force and no entry could
fire. **This changed on 20 Aug and needs Ben's ruling.** After ruling 1's target scaling the
same fold calibrates to `lower=0.026076` on a validation Sharpe of **+0.483 over 8 trades**,
so **the deployed system now trades**. Spec §7's required finding — *"the deployed system
declines to trade, and that is a result", which must not be softened* — remains exactly true
of the deployment as it stood on 18–20 Aug and is **no longer true of the current one**.
Whether the report keeps it as a dated finding about a past deployment or replaces it is
Ben's call. **One
live constraint GB-26 designs around:** the account refuses **recent** SIP data, so anything
reaching for a live quote rather than a completed daily bar meets a wall.
**Last gate passed:** **GATE 1, 17 Aug 2026**, 25 days before its commitment date
**THE CENTRAL RESULT, and the phrasing is fixed** (GB-49, 20 Aug): *These models extract
nothing from this market that they do not also extract from white noise. That is a
statement about the models, not about the market: over the same period a constant
always-long rule beats chance by 5.6 points, so the structure exists and is reachable —
just not by explicit or implicit frequency decomposition at this horizon, on this universe,
with these channels.* On the same footing: always-long real **0.5564**; DLinear real
**0.4959** vs white noise **0.5006**; FITS real **0.5055** vs noise **0.5025**. **The models
sit at their noise level; the market does not.** The shorter version — "no measurable
difference between this market and noise" — must not survive anywhere.

**CONFIRMED AND SHARPENED 23 Aug** (GB-50/GB-51, 656 arm-folds over four cutoffs): *no arm
beats the always-long bar* is negative in **9 of 9** anchor-arm cells, **−4.3 to −6.5
points**, under **both** null controls, with no positive cell anywhere; the COF sweep finds
nothing at **every** cutoff, three of four sitting **below their own white-noise twin**.
**That claim is one of absence** and rests on the sign stability and the effect size. The
stronger claim — *every arm is significantly worse* — survives Holm at 3 of 9 and **is not
made**. Holm guards against false positives, so applied to a claim of absence it makes it
*easier* to conclude nothing was found: **a bias toward our own conclusion rather than a
protection**, reported for honesty and not relied on.

**Blockers:** none. Standing note for GB-57, **re-measured 20 Aug after the target scaling
(ruling 1) and these numbers replace the ones before it**: the strategy beats **neither**
reference it should be read against. Direction **0.4959** against an always-long bar of
**0.5564**; return **−0.31% per fold** against buy-and-hold's **+7.73%**, Sharpe **−0.33**
(9 folds) against **+1.39**; **5 of 16 folds stand aside**. The whole chain runs end to end
in one command, so these are measurements rather than expectations.

**What moved and why, because the direction of the change matters.** The previous figures
were 0.5182 / +0.40% / Sharpe 0.62 / 3 folds aside. DLinear has **no backcast term**, so
none of ruling 1's B+F argument applies to it — what changed for DLinear is its
**conditioning**, the median learned weight going from **0.54× Adam's step to 2.50×**. The
0.5182 was the score of the arm in which **71% of the weights sat below the optimiser's step
size**: a model too under-trained to move its weights, not a directional edge. **The null
headline is strengthened by this, not weakened** — a result that survives only while the
optimiser cannot resolve the weights was never a result.

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
| Ruling: one decision per bar | 19 Aug 2026 | Ben | **Complete.** A decision's `as_of` is the last *completed* daily bar, so at 60s polling the loop re-derived the identical verdict 390 times a session and wrote **1,950 identical records a day**. The entry decision is now made and recorded once per `(as_of, symbol)`; reconciliation, protection, trade emission and exits stay outside the rule and run every cycle. `save_decision` refuses a duplicate and `load_decisions` asserts uniqueness on load. `decision_id` becomes `{as_of:%Y%m%d}-{symbol}`, so a second entry for one bar is refused **at the broker**. 8 new live-loop tests, 9 new record tests. |
| Race confirmed | 19 Aug 2026 | Ben | **Answered: the Book, and only as a race.** Sizing excluded symbols already in the book, but the book is rebuilt from broker **positions**, which appear only on the fill; `_absorb_entry` gives up after 10 polls and books nothing, so the next cycle re-entered. The executor added nothing — `decision_id` carried the cycle id, so Alpaca's own duplicate rejection could not fire. Pinned by a test that runs the loop against a broker which accepts and never fills. |
| Three live defects | 19 Aug 2026 | Ben | **Two fixed, one referred.** (1) `AlpacaBroker.get_orders()` returned **open orders only** (Alpaca's default), so `emit_trades` could never see a filled sell — **the live trade log was structurally empty** — and `protect_book` rule 4 could never fire, leaving a filled stop's take-profit working as a naked sell. Fixed to `QueryOrderStatus.ALL`. (2) Re-arming reused a `client_order_id` Alpaca had already consumed (**measured**: refused even after cancellation), which counts as an arming failure and flattens at market on the second strike; leg ids now carry an attempt number derived from the broker's own order history. (3) The legs are `TimeInForce.DAY` and expire nightly — **referred to Ben**, see Open questions. |
| Account flattened | 19 Aug 2026 | Ben | **Done, on instruction.** Sold **0.091919619 AAPL at 310.394**, filled 2026-08-18 15:38:59.713802 UTC, 8 ms after submission, `client_order_id=operator-flatten-quarantined-aapl`. The account is flat: equity 100,000.28, cash 100,000.28. The following cycle logged `reconcile: missing_position: AAPL local=0.091919619 broker=0.000000000 — the broker holds nothing; dropping the local holding. A filled stop or target looks exactly like this`, and emitted **no trade**, which is correct twice over: the position was quarantined so `emit_trades` refuses it by design, and defect (1) meant no filled order was visible anyway. |
| Dashboard notes | 19 Aug 2026 | Noy | **All seven done.** Both tables rendered as HTML in the design language (near-black ground, monospace uppercase headers at wide tracking, thin dashed row rules, orange on the header row only, numerics right-aligned with tabular figures) — and the expanders restyled with them, since they carried the same default chrome; the band note moved to a strip below the plot area; a numbered left ruler added and the content column uncapped (the charts were letterboxed by a fixed `height` beside `width=100%`); em dash for no order; `EXPLANATION_FRAGILE_BELOW = 0.20` flags a nearly-cancelled decomposition in its cell and in the expander title; expander titles carry the trend strength; top ruler reads `0`; masthead stacks PROJECT / SYSTEM / VERSION. 16 new dashboard tests. |
| Store repaired | 19 Aug 2026 | Ben | **20 lines → 5 unique decisions** in `checkpoints/live`, and 72 → 70 in `checkpoints/replay`. The live duplicates were four copies each of one bar's five decisions, differing only in whether GB-38's `provenance` key was present — every one decodes identically. The two replay duplicates were GB-37's approve/decline records, which now **amend** the decision rather than appending a second at the same bar. |
| GB-39 | 20 Aug 2026 | Noy | **Complete.** `faults.py` — `retry` with exponential backoff (3 attempts, 1s then 2s, giving up 3s in, well inside one 60s poll) and `Unavailable`, the type that says the outside world did not answer rather than that a defect was found. `executor.RetryingBroker` wraps every broker call; **a write is safe to retry only because the `client_order_id` is bar-derived** — a retry refused as a duplicate is read as the receipt for the attempt that timed out, and the order is recovered rather than reported as a failure. An unreachable cycle is logged loudly, skipped, and the session continues; a run of two or more says so. No circuit breaker and no health endpoint, per scope. 11 new tests. |
| GB-39 blocker closed | 20 Aug 2026 | Noy | **`adopt_own_positions`.** Confirmed first, as asked: `reconcile` quarantines an unknown position and **never adopts**, and never looks at `client_order_id` — so the loop's own fill, arriving after `_absorb_entry`'s poll window or after a crash, was quarantined and therefore **never protected**. That was the gate blocker. Adoption now runs between reconciliation and protection, and takes a position back only on evidence: a filled buy whose `client_order_id` is a decision id this system mints, naming a decision **on disk in this log under this provenance**, carrying the order it produced. Anything short of that stays quarantined — the reconciler's ruling is untouched. **The ordering that makes it possible: the decision reaches disk before the order reaches the broker.** |
| **GATE 2 criterion 2: SATISFIED** | 27 Aug 2026 | Ben | **The execution path is proven live, end to end, under the policy ruled on 26 Aug.** Two Co-Pilot recommendations from the 2026-08-26 bar approved during the 27 Aug rehearsal session against `checkpoints/rehearsal`: `20260826-GOOGL` filled 0.073529411 at **340.528** (entry `ce604858`), `20260826-NVDA` filled 0.111706931 at **226.986** (entry `57fc23a8`). Both were **quarantined by `reconcile` and then adopted** under their own decision ids — GOOGL in cycle rehearsal-0102, NVDA in rehearsal-0103 — which is `adopt_own_positions` doing exactly what GB-39's blocker closure built it for. Each then carried **one** protective order and it was the stop: GOOGL `2178555b` at 329.8000, NVDA `f8e86870` at 217.0859, client ids `...#1-stop`. **No limit order was attempted for either**, so the `insufficient qty available` failure of 25 Aug cannot recur by construction rather than by luck. The loop-side target evaluated on the last completed bar (2026-08-26) and found neither target reached — 360.40 for GOOGL, 237.2279 for NVDA — so no exit was sent, which is the correct null outcome and not an absence of the mechanism. **Satisfied through the API path, not the panel.** `answer_pending` was called directly because the dashboard was not running, and its `--state-dir` defaults to `checkpoints/live` while `--source` defaults to `live`; a bare `streamlit run` therefore shows an empty queue and an empty decision history for a rehearsal. The approval logic is identical — `app.py:1381` calls the same `answer_pending` — but **GB-37's panel path is now the untested half and must be exercised in replay against a non-default state directory**, with both `--state-dir` and `--source` passed, before the gate is walked. **Two observations recorded rather than fixed.** (1) The stops carry `#1`, meaning they were armed by rule 3's re-arm path and not by `executor.protect`: both entries were `pending_new` when `answer_pending` returned, so `protect` correctly returned nothing and the cycle that saw the fill armed them — rule 1 working, and the first time it has been observed live. (2) A recommendation approved a day after its bar carries stop and target levels derived from the **older** price. NVDA was sized at 223.7999 and filled at 226.986, so its intended 3.0% stop distance is a realised **4.4%**. Latent on 25 Aug, realised here, and it belongs in GB-57 beside the execution-model finding. |
| The state-directory lock reclaimed a stale lock in production | 1 Sep 2026 | Ben | **Unprompted, and the first time that path has run outside a test.** A session started at 16:11 found the lock left behind by PID 9836, the deployed loop that died over the weekend without releasing it, probed the PID, found it gone, and logged `reclaiming a stale lock on checkpoints\live: PID 9836 (mode 'deployed') since 2026-08-26T10:42:20Z is no longer running` before taking it. That is the branch built on 26 Aug for exactly this case - a loop killed by the host never gets to clean up after itself - working on a real stale lock rather than on a synthetic one, and saying so rather than reclaiming silently. |
| **GB-40 — GATE 2 WALKED IN FULL: 7 of 7 GREEN, PASSED** | 28 Aug 2026 | Ben | **Criterion 1** full unattended session — 24 Aug, open to close, 211 of 266 cycles completed through three broker outages (30, 6 and 19 cycles skipped) with no intervention; corroborated 27 Aug by 261 consecutive cycles from the open with no gap over five minutes. **Criterion 2** execution path proven live — see the row above. **2b** deployed band behaves correctly — on all four live-decided bars (2026-08-17, -21, -24, -26) every symbol returned `hold` with `passed_threshold=False`: the band was evaluated and declined, not stood aside for want of calibration, and the gate log states which occurred as §8 requires. **Attribution** — 100 records carrying `per_channel`, worst residual **1.248e-08** against the 1e-5 contract, zero violations. **Replay** — 70 records at `replay:fold-13` across 14 trading days, with `load_live_bars` and `AlpacaBroker` replaced by hard failures so the offline claim is enforced rather than asserted. **Co-Pilot** — approve exercised **live** on two fractional orders; decline exercised in replay, with one `Declined by the operator` record and two tests. **Criterion 6** two or three separate live sessions — **three**: 24, 25 and 27 Aug. **The reading is stated because it decides the verdict:** the ~18 Aug run is excluded, because it was three cycles and criterion 1 already reads FAIL for it, and an incomplete run must not count twice — once as a failure and once as a tally. **§8's cancellation rule was never invoked and FITS is not cancelled.** It fires only on a red gate on its own date; the gate is not red. It was also never in play on the two occasions it looked close: the 20 Aug walk read 3 of 6 five weeks early and the rule is dated to the gate, and criterion 2's earlier *structurally unreachable* finding was resolved by amending the criterion to test the machine rather than the model — a band that declines is a result, not a gate failure. **THE CAVEAT, AND IT IS A CONDITION ON THE EVIDENCE RATHER THAN ON THE CRITERION.** The 27 Aug rehearsal that produced criterion 2's evidence **violated the fourth §8 rehearsal condition: it held overnight.** The machine suspended at 18:23Z, 1h37m before the close, so the scheduled 22:45-local flatten never fired; the market closed and Alpaca cancelled the DAY stops; the machine woke at 21:38Z and the on-stop close-out raised `Unavailable` out of `main` because `release_protective_legs` sat outside its `try`. Two positions were carried **unprotected for about eighteen hours** and were closed by hand at the 28 Aug open (GOOGL 340.528 → 340.700, NVDA 226.986 → 227.010, realised **+0.0153** in total). The execution evidence is unaffected — fill, adopt, stop-alone and loop-side target all stand and are logged — but the run that produced it broke one of the four conditions attached to it, and that belongs beside the verdict rather than inside it. The defect is fixed and committed in `f392b6e`; the machine-suspend cause is outside the code and has now broken three consecutive live runs. |
| GB-40 | 20 Aug 2026 | Ben | **GATE 2 walked: 3 of 6 green. Not passed, and FITS is not cancelled** — the §8 rule fires on the gate's own date of 25 Sep, not on a rehearsal run five weeks early. PASS: exact attribution visible in the dashboard (75 records, worst residual **1.248e-08** against 1e-5); replay reproduces a recorded day **with `load_live_bars` and `AlpacaBroker` replaced by hard failures**; Co-Pilot approve and reject, replayed. FAIL: no full unattended session; no order placed by the loop; one live session, not two or three. **The finding that matters: criterion 2 is structurally unreachable, not merely unmet** — a band that stands aside can never place an order, so it will read FAIL on 25 Sep exactly as it does today. Three ways out are in DECISIONS; the choice is Ben's. |
| GATE 2 criterion amended | 20 Aug 2026 | Ben | **Spec §8 criterion 2 replaced, and the amendment was prompted by the system behaving correctly.** The old wording conflated *does the execution path work against a real broker?* with *does the deployed model trade?* — only the first is a gate; the second is a result and already has an answer. Criterion 2 is now EXECUTION PATH PROVEN LIVE, demonstrable with a **rehearsal band** when the deployed band stands aside; 2b is DEPLOYED BAND BEHAVES CORRECTLY. Four conditions implemented: provenance `rehearsal:<reason>`, never `live`; **no rehearsal decision may reach a metric** — `records.is_reportable`, plus a rehearsal emits **no `Trade` at all**, because `Trade` carries no provenance and cannot (frozen contract §4.2); flatten 15 minutes before the close, satisfying the DAY-legs condition; capped at a small notional. Band is `Thresholds(lower=1e-9)` — the smallest the contract admits, since it refuses zero. 10 new tests. |
| Config hash split | 20 Aug 2026 | Ben | **`model_config_hash` added; `config_hash` keeps its exact meaning.** The narrow one covers `data`, `window`, `wavelet`, `fits`, `channels`, `model` and `meta.seed`; checkpoints gate on it, and a difference in the full hash is **logged, not raised**. The danger was never the one retrain — it is Sprint 4, where FITS and the COF sweep touch config repeatedly and a guard that fires spuriously is one somebody weakens. `universe` is deliberately out: `stats_for` already refuses an untrained symbol by name. A guard test fails the suite if a new top-level section appears that nobody has classified — which answers the objection the old whole-config gate had recorded. 6 new tests. |
| GB-42 | 20 Aug 2026 | Ben | **Written before GB-41, per §6.3 — the test comes first.** `tests/model/test_fits_amplitude.py`: the acceptance test feeds a sinusoid of known amplitude and period and asserts the backcast within 1e-4, marked `xfail(strict=True)` so the pass that arrives with GB-41 **fails the suite** until the marker is removed. It fixes GB-41's interface: `fits.extend_spectrum(x, horizon)`. **The companion passes today**, measuring the trap on `torch.fft` itself — omitting the scale divides every sample by exactly `L/(L+H)` — so the file demonstrates the bug it names. A third test states why it costs a day: 3.2% at L=120/H=4, invisible in a plot, and it **improves MSE**. FITS itself is not implemented. |
| CI is read, not assumed | 20 Aug 2026 | Ben | `scripts/ci_status.py`, using the credential git already holds. Three consecutive reports had been written on an assumed green because `gh` is unauthenticated here and the repo is private. Verified: run **#13, `e4b55b2`, success**, and every run back to #8. |

## Sprint 4 — FITS, Study, Report · 26 Sep – 10 Oct 2026 → GATE 3

| Task | Date | Owner | What was built |
|---|---|---|---|
| GB-42 | 20 Aug 2026 | Ben | **Written before GB-41, per §6.3.** Sinusoid reconstruction within 1e-4, plus a companion measuring the trap on the FFT libraries themselves and a third saying why it costs a day. |
| GB-42 corrected | 20 Aug 2026 | Ben | **The grid was wrong and no correct FITS could have passed it.** Bin `k` is frequency `k/L` on the input grid and `k/(L+H)` on the output grid, so naive zero-padding **never** reconstructs (err 4.97, the amplitude itself) and frequency-preserving mapping reconstructs only when `η·k` is an integer — which at the configured `H=4` it is not (`η·k = 10.333`). Moved to `H=12` (`η·k = 11`, err **2.2e-14**), with the grid choice stated in the test and a second test measuring the `H=4` failure. Same error as asserting amplitude where only the ratio is exact. **The gap at `H=4` is not a defect — it is why FITS has a learned layer.** |
| Spec §6.3 corrected | 20 Aug 2026 | Ben | The symptom paragraph claimed direction accuracy degrades. **A uniform positive scaling cannot change a sign.** It is a **comparison bug**: MAE and MSE *improve*, and FITS would beat DLinear on MAE by being flatter rather than better. Wrong line kept beside its correction, and converted into a test. |
| GB-41 | 20 Aug 2026 | Ben | **`model/fits.py`.** RIN → rFFT → LPF → one complex linear layer → zero-pad → irFFT → `(L+H)/L` → inverse RIN → split, per §6.2. Univariate (§6.4) and **attributing to every active channel** with 0.0 for the four it does not read (GB-33). The pipeline is linear in the window, so the model **is** an `(H, L)` matrix — built by pushing the `L` basis vectors through the *actual forward pass*, so attribution is a measurement of the implementation rather than a second derivation. Torch fits, numpy predicts. B+F supervision hardcoded, zero init. Registered in `ALL_FORECASTERS`; **passes the §4.4 contract test with no edit to that test**. 18 new tests. |
| GB-41 measurements | 20 Aug 2026 | Ben | **1,200 parameters measured** (24×25 complex, counted as reals — **1,150 of them effective; see the dead-DC-row entry below** — 600 against DLinear's 4,800 would flatter FITS twofold). **RIN is per-instance and property 6 catches the alternative**: a batch-mean variant moves window 0 by 4.85 and fails with *"LOOKS AHEAD"*. **Mean reversion rejected — FITS is a momentum model**: agreement with the trailing move **0.5725**, above 0.50 in **13/16** folds, against DLinear's **0.4991** in 8/16. **The per-symbol scaler does not earn its place under FITS** — MAE worse in 16/16 for both models, but direction better with the scaler for DLinear (0.5182 vs 0.4657, 12/16) and not for FITS (0.5098 vs 0.5133, 9/16). Nothing changed on the strength of it. **The scaler sentence is superseded the same day — see the conditioning row below. It was one label over two mechanisms and neither of them is the scaler.** |

| Conditioning | 20 Aug 2026 | Ben | **The scaler finding was two mechanisms under one label, and the MAE column measures flatness.** 16 folds × 2 models × 5 arms, 160 fits. **DLinear: the optimiser-resolution hypothesis is confirmed** — median learned weight **5.4e-4 against an Adam step of 1.0e-3** (predicted 6.1e-4), **71% of weights below the step**, and the scaled-vs-raw MAE ordering **reverses** with the rate: 0/16 folds at `1e-3`, 14/16 at `1e-4`, 16/16 at `1e-5`. **FITS: refuted** — its weights are **19× the step** and no learning rate closes anything (0/16 at all three). Its real defect is a **B+F unit mismatch**: with `X` standardised and `y` raw the backcast term is **67× the forecast term**, the objective is 98.5% backcast, and the forecast is **4.8× too large**. Standardising `y` restores 0.69× and the MAE. **And the fix is not a fix:** Spearman(MAE, flatness) **+0.811 / +0.668** against Spearman(MAE, direction) **+0.009 / −0.128** — the best-MAE arm forecasts at **19% of the truth's magnitude**, which is §7.3's ban arriving as data. **Direction moves 0.0041 across all five FITS arms**, so the null headline is not conditional on any of it. Configuration untouched; GB-49 gains a learning-rate axis and a `flatness` column. |
| GB-44 | 20 Aug 2026 | Ben | **The switch and the sharing, tested as two claims.** `model.active: fits` selects FITS with no other change — and that rests on `VALID_MODELS` (config layer, which **may not import** the model layer) agreeing with `ALL_FORECASTERS`, which nothing checked; a test now asserts the two sets are equal. **A third list was the stale one** — `smoke_offline`'s `--model` choices still read `(persistence, dlinear)` and its `run` defaulted to the string `"dlinear"`, so `model.active: fits` changed the live checkpoint and **nothing** about the study numbers. Both now come from the registry and the config; the proof is one fold end to end with FITS selected by configuration alone. Sharing is asserted by **breaking** it: perturbing the single weight matrix must move *every* symbol's forecast. `fits.individual_weights` stops being decorative — declared in §5, documented in §6.4, read by the loader since GB-2 and **consulted by nothing** until now. Under `true` a pooled universe is **refused** naming the remedy, because `Forecaster.predict` takes no symbol and one model cannot hold five weight sets; per-symbol weights are one call and one checkpoint per symbol. The manifest records the regime. 9 new tests. |
| The dead DC row | 20 Aug 2026 | Ben | **Found by the sharing test failing.** Its first perturbation was row 0 of the weight matrix and **nothing moved for any symbol**: RIN subtracts the window mean, the rFFT's bin 0 *is* that mean, so row 0 multiplies zero on every forward pass. Measured after a full fold: `max abs(w)` row 0 = **2.1e-11** against **0.89** elsewhere, bin 0 after RIN = **3.6e-15** against bin 1's **13.1**, and perturbing every row-0 weight by `1+1j` moves the forecast by **exactly 0.0**. So **1,200 allocated, 1,150 effective — 4.17% dead**. The row **stays** (§6.2 keeps the first `COF` bins; the paper's architecture has the same row); the parameter count is corrected everywhere it appears. |
| First head-to-head | 20 Aug 2026 | Ben | **FITS against DLinear, persistence and the always-long bar, 16 folds, scored through the project's own metric functions.** MAE: persistence **0.015286**, DLinear 0.031413, FITS 0.096841 — **both models lose to persistence in 16/16 folds**, and at their best conditioning they still only reach 0.015629 and 0.015802. Direction: DLinear **0.5182**, FITS **0.5098**, against an always-long bar of **0.5560** — **each beats the bar in 4 of 16 folds**, and they beat each other in 8 folds each. **The MAE column as configured is not a model comparison** — FITS's number is the B+F unit mismatch and DLinear's is the optimiser-resolution artefact. DLinear's 0.5182/0.5560 reproduces GB-27's re-measured figure to four decimals. Shared-model wall time: FITS **mean 8.29s, median 6.78s, max 17.87s** on fold 15 over 2,480 windows; DLinear mean 2.58s. |

| Ruling 1 | 20 Aug 2026 | Ben | **The forecast target is scaled, on the B+F argument alone.** `loss = mse(forecast, y) + mse(backcast, x)` is an unweighted sum, so it is §6.2's objective only when the terms share a unit — and they did not: **backcast 67× the forecast term, the objective 98.5% backcast**, so FITS was not training what the spec says it trains. `build_windows` now divides the target by the target channel's **own** deviation (the same divisor the input column got, so the two terms are in *identical* units); `restore_targets` is the inverse, applied at every point a number leaves the model layer. **Divided, never centred** — centring makes `predict` affine and `Attribution.from_terms` refuses a decomposition carrying a constant; measured, the two forms are indistinguishable. **Not for the MAE**, and DECISIONS says so explicitly: the evidence the fix is *correct* is that FITS lands at **0.015841 against the raw arm's 0.015822**, which is what scale-equivariance predicts. Two deviations reported: y's statistics are a **derived accessor** on `ChannelStats` rather than a stored duplicate, and the inverse sits at the **model layer's boundary** because `Forecaster.predict` takes no symbol. `CHECKPOINT_VERSION` → 2; both deployed checkpoints regenerated. 9 new tests. |
| Ruling 2 | 20 Aug 2026 | Ben | **MAE stays, and never appears alone.** `metrics.flatness` = mean\|forecast\| ÷ mean\|actual\|, 1.0 right-sized. `COMPANIONS` builds the summary's columns and `FOLD_COLUMNS` is asserted against the same rule, so the pairing is **structural**: printing MAE without flatness means deleting the pairing rather than forgetting it. §7.3 amended to cite the measurement rather than the argument — Spearman(MAE, flatness) **+0.81 / +0.67** against Spearman(MAE, direction) **+0.01 / −0.13**, and the 16-fold MAE winner is persistence, which forecasts nothing and has flatness **0.0**. GB-49 carries the column in `results.csv`. |
| GB-45 | 20 Aug 2026 | Ben | **`explain/spectral.py`.** Per-frequency contribution, gain and phase, **keyed by period in days** and phase converted as `φ/2π × period`. The `(H, L)` map behind each frequency is **measured from the model's own forward pass** with one bin unmasked — `forecast_matrix`'s discipline — and the parts sum back to the whole at **4.4e-16**; the contributions go through `Attribution.from_terms`, so there is still exactly one place that adds a decomposition. Two facts share the infinite-period key and the module says so: `per_frequency[inf]` is the **RIN mean**, which bypasses the layer entirely, and `gain_phase[inf]` is the **dead bin-0 row**. Removing the mean term makes the decomposition fail to close, and a test makes that happen rather than trusting the docstring. 15 new tests. |
| GB-46 | 20 Aug 2026 | Ben | **`scripts/frequency_response.py`** — `\|W\|` against period in days, cutoff annotated, 220 dpi, `figures/frequency_response.png`. In `scripts/` because it reads a checkpoint and writes a file (§3.4), and because matplotlib in `explain` would put a rendering dependency in the live loop's import graph. **Defines no colour of its own** — the palette is the dashboard's, so the green/red ban asserted there covers it, enforced by a test for a hex literal in the source. Adds matplotlib (DECISIONS). 8 new tests. |
| GB-49 | 20 Aug 2026 | Ben | **`experiments/study.py`** — one command, every axis a column. Persistence/DLinear/FITS × C0_base/C2_hybrid with **FITS × C2_hybrid written in as a skipped row carrying its reason**, not left blank. Columns added since §7.4: `anchor`, `lr`, `control`, `flatness` (immediately beside `mae`), `cancellation`, `data_snapshot_last_bar`. **Two decisions reported rather than assumed:** the design is a **star, not a cross product** — full cross 27 conditions / 135 cells / 2,160 arm-folds / ~140 min against the star's 7 / 35 / 560 / ~36 min, and the extra 20 buy interactions nobody asked about; and the null arms live in **the same `results.csv` behind one gate**, `study.reportable`, in `records.is_reportable`'s shape — two files can drift in columns or vintage, and the null rows are only meaningful *beside* their real counterparts. `run_arm` and `run_buy_and_hold` became public in `smoke_offline` so there is one path from a fold to an `ArmRun`. 17 new tests. |
| Standing requirement | 20 Aug 2026 | Ben | **Every headline claim gets grid sensitivity AND a null control.** Neither subsumes the other, and the project has one demonstration of each: `r = −0.47` **died to a grid shift** and would have passed a control; the FITS phase advance **died to a control** and passed grid sensitivity at **48/48 cells**. The controls rebuild the world end to end — prices regenerated from shuffled or matched-variance returns — because perturbing features alone would have the backtester trading the real market while the model forecast a synthetic one. **Heuristic in CLAUDE.md:** on financial data a **CV near 1%** across independently trained models is evidence about the machine; the phase advance was **1.04%**. |
| A result about FITS | 20 Aug 2026 | Ben | **86% of the learned frequency response is grid geometry, so the complex layer spends most of its capacity on a deterministic resampling operator** — a defensible critique of the architecture **as published**, measured on real data, and reported as a result rather than a caveat. **Configuration-specific**, and the framing needed correcting: the cost is a function of `η = 1 + H/L` and is **not monotone** — `H=120` (`η=2`) is free, `H=1` is *better* than `H=4`, and **the worst point measured is the configured one** (1 of 24 bins aligned, mean misalignment 0.2833). Parked in `IDEAS_PARKED.md`: initialise with the analytic operator and learn the residual. |
| Phase finding withdrawn | 20 Aug 2026 | Ben | **The sweep confirmed it and the control killed it.** 48 models (16 folds × 3 anchors): **+1.9582 days, sd 0.0203, positive in 48/48** — *more* stable than the r = −0.47 correlation that vanished under a grid shift, so stability would have blessed it. The control: fitted on **white noise** the same band gives **+1.9309 days**, and the whole gain curve correlates with the real-data curve at **+0.9485** with the trough at **8.00 days in both** — about **86% of the response is data-independent**. The replacement is better: gain tracks **grid misalignment** at **r = −0.9049, Spearman −0.9427 (p = 1.8e-11)**, and the 8-day trough is the **half-integer bin**, `η·k = 15.5`, the frequency that lands exactly between two output bins. GB-46's figure keeps its place and shows the **interpolation cost**, not equity cycles. **When a measurement is suspiciously stable, the next test is a control, not a larger sweep.** |
| GB-47 | 20 Aug 2026 | Ben | **`features/wavelets.py`** — `wav_a1..a3` from the log-return series, **one decomposition per bar over the trailing 64 returns**, emitting the last sample of each reconstructed band. The standard whole-series recipe leaks because DWT filters are two-sided, and it is implemented deliberately so GB-48 can watch the harness reject it. `CHANNEL_BUILDERS` now takes `(bars, cfg)` for every channel and `PARITY_WARMUP` admits a config-dependent entry, both to avoid a second table. `C2_hybrid` builds. |
| GB-48 | 20 Aug 2026 | Ben | **Causality and additivity, through GB-10's harness rather than a second one.** Both modes at three splits on each of `wav_a1..a3` and on the whole C2_hybrid feature frame; `a3+d3+d2+d1` reconstructs the input to **1e-12**, compared against the *input* and not a sum of the module's own outputs; and the whole-series version raises `LOOKS AHEAD` under **both** `scale` and `shuffle`, asserted separately, with the two versions asserted to differ so the rejection is not of a function nobody would write. 22 new tests. |
| The floor did not move | 20 Aug 2026 | Ben | **Measured, against expectation.** `min_history_bars` is a maximum, not a sum: RSI's 325-bar warm-up still dominates the wavelets' 64, so the floor stands at **445** for both channel sets. The two numbers are different kinds — RSI's is a tolerance argument about a decaying seed, the wavelets' is **exact**, since a DWT of a trailing window depends on that window and nothing before it. **The C2_hybrid parity sweep is byte-identical at 445 over 5 symbols × 25 timestamps**, inheriting GB-27's sweep rather than being certified at a point. |
| The band's selection context | 20 Aug 2026 | Ben | **The rule stays; the reporting changes.** The deployed band fires on a validation Sharpe of **+0.483 over 8 trades** — a **maximum over fifteen candidates**, and under noise the max of fifteen on eight trades is positive almost surely. Tightening a pre-registered rule because we dislike what it selected is the failure this project exists to avoid, and the rule demonstrably discriminates (**5 of 16 folds stand aside**). So `dashboard.app.BandContext` puts the fold, the validation Sharpe, the trade count and *grid maximum of 15 candidates* in the masthead, and §7's stood-aside finding is **re-dated rather than deleted**. GATE 2 criterion 2 is now reachable by the deployed system without a rehearsal; the rehearsal path stays in the spec. |
| CI red, and the fourth instance | 20 Aug 2026 | Ben | **Run #17 failed on a green local suite**: CI installs from `requirements.lock`, not from `pyproject.toml`, so declaring matplotlib in the manifest installed it locally and nowhere else. **The same defect class as `smoke_offline`'s third registry copy** — a fact in two places with no mechanism keeping them equal — and the **fourth instance in two days**. Lock updated; `test_every_declared_dependency_is_pinned_in_the_lock` is the mechanism, and it is proved capable of failing. **What worked:** the failure read *"run this with the project's interpreter"* rather than `ModuleNotFoundError`, because `test_every_script_guards_its_imports` had forced an import guard onto the script hours earlier — a test written for one reason catching a different failure. |
| The response, measured | 20 Aug 2026 | Ben | **WITHDRAWN THE SAME DAY — see the row below.** As first written: **The learned response is not flat, and the phase is the finding.** Fold 16, 2,480 training windows: gain runs **0.596 to 0.830**, trough at **8.6 days**, peak at **24 days** — the model suppresses the fastest cycles that survived the low-pass and favours the three-to-six-week band. Across that band the phase shift is a consistent **+1.9 to +2.1 days**: the model **advances** those cycles by about two days, which is **GB-41's momentum finding arriving from the other side**, with no extra experiment. Dominant contributor over 290 test windows: the **RIN mean 37.2%**, the **15-day cycle 22.4%**, 120-day 17.2%, 17.1-day 13.1% — and **no contributor's mean share of the gross exceeds 0.211**, which is GB-30's cancellation story in the frequency domain. |
| RULING: DAY stays | 23 Aug 2026 | Ben | **The protective legs stay `TimeInForce.DAY`, and no code changed.** The GATE 2 condition of 20 Aug offered two escapes and **both are refused**: GTC is unavailable on a fractional quantity (`42210000 fractional orders must be DAY orders`, measured), so it costs whole-share rounding — and `shares_for` is shared with the backtester, so that would change **position sizing across the entire study**, not the executor; and flattening at the close would trade a **different strategy** from the one being reported, closing a protection gap by opening a **parity** gap. **The residual is stated as measured, not as unlikely:** for the interval between a session open and a successful re-arm, **and for any session the loop does not run at all**, an open position carries **no broker-side protection**. Rule 2 of the GB-26 policy closes the first case for every session the loop runs and does **nothing** for a session it misses. Latent, not realised: no position has been held overnight by the loop. GATE 2's condition amended in §8 with its original text kept beside it; GB-57 carries the residual verbatim. |
| GB-50 | 23 Aug 2026 | Ben | **The COF sweep, riding on the star as three more spokes.** `cutoff_period_days` ∈ 2/5/10/20, **FITS only** — the cutoff reaches persistence and DLinear through nothing at all, so their rows at cutoff 20 would duplicate their rows at cutoff 5, and a duplicate is what a reader eventually averages. **Each spoke carries its own null control** (6 new conditions, not 3): a cutoff tested only on real data is a cutoff nobody can falsify. 13 conditions, **41 cells, 656 arm-folds**; measured **877 rows, 576 reportable, 29.3 minutes** of arm time. `results.csv` gains `cutoff_period_days`, `cof` and `dead_row_fraction`, adjacent for the reason `flatness` sits beside `mae`, all three **derived from `model/fits.py`**. **The dead DC row is `1/COF`, so the smaller model wastes proportionally more** — 1.67% at cutoff 2 against **16.67%** at cutoff 20 — and the parameter count runs **7,440 / 1,200 / 312 / 84**, so at cutoff 2 FITS is **larger than DLinear's 4,800**: *"FITS is the smaller model"* is a statement about a configuration, not an architecture, and this project's own sweep inverts it. **THE SWEEP FINDS NOTHING AT EVERY CUTOFF, and that is the answer to §1.4 rather than an absence of one.** Direction against each cutoff's own white-noise twin: **−0.71 / +0.31 / −0.79 / −0.93 points** at cutoffs 2 / 5 / 10 / 20 — every cutoff at or below its noise level. Against the always-long bar: **−5.93 / −5.09 / −5.95 / −6.08 points**, all four. **No optimum — and weak evidence AGAINST the determinacy story rather than an absence of it.** The term predicts visibly higher fold-to-fold variance at 1.35× and there is none: per-fold sd of direction is **0.0592 / 0.0627 / 0.0630 / 0.0546** across ratios spanning **1.35× to 119×**, which is flat. **An absence of the predicted variance is information, not silence**, and the reading everything else supports is that **if the model learns nothing there is nothing to overfit, so determinacy does not bite**. **And the sweep reproduces §7.3 on a new axis**: as the cutoff rises MAE falls **0.0167 → 0.0155** and flatness falls **0.4187 → 0.1560**, Spearman(MAE, flatness) = **+1.00** across the four cutoffs, and the best-MAE cutoff forecasts at **15.6% of the truth's magnitude**. |
| Geometry reading corrected | 23 Aug 2026 | Ben | **"Worst in the middle" was wrong; the pattern is saturation.** `η = 1 + H/L = 1 + 1/30`, so `frac(η·k) = frac(k/30)` has a **period of 30 bins**: mean misalignment rises while the retained bins cover less than one cycle and **settles at 0.25**, the mean distance of a uniform fractional part to the nearest integer, and is **exactly 0.2500** at COF 30 and 60. The apparent peak at the deployed cutoff (0.2833 at COF 24) is a **partial-cycle sampling artefact** — 24 of 30 bins over-weights the far half. So the claim is *every low cutoff is saturated and the deployed one is among them*, not *the deployed cutoff landed on the worst cell*. **Landing on a plateau is the weaker and truer claim**, and the peak version invites a reader to check the arithmetic and find it marginal. The test that asserted the peak was asserting the artefact. |
| Three forces on COF, pre-registered | 23 Aug 2026 | Ben | **One claim in GB-57, written before the sweep ran.** **(i) The misalignment tables for `COF` and for `H` are one curve.** `frac(η·k) = frac(k·H/L)`, period `L/H` bins, and the only quantity is **how many alignment cycles the retained bins span** — measured across `H` at `COF = 24`: **0.20 cycles → 0.0958, 0.80 → 0.2833, 2.40 → 0.2333, 12.0 → 0.2500**, the same rise-then-plateau. `H = 120` at 0.0000 is not the far end of a curve but the **single exact escape**, `η = 2`. Two entries dated 20 and 23 Aug are one mechanism. **(ii) Three forces, and only one saturates.** Information retained **rises**; reconstruction quality **falls and saturates at 0.25**; **statistical determinacy falls and does not** — against **2,505 training windows × H = 4 = 10,020 equations** per fold the layer runs **7,440 reals at 1.35×, 1,200 at 8.35×, 312 at 32.12×, 84 at 119.29×**. **(iii) The prediction, recorded before the result:** if there is an optimum it is **not at cutoff 2**, and the reason will be **statistical rather than geometric**; if cutoff 2 wins, that is evidence **against** the determinacy story. **Measured: no optimum, and the prediction has weak evidence against it rather than none.** The determinacy term predicts higher fold-to-fold variance at 1.35× and the spreads are flat across a 1.35×–119× range, which is information rather than silence: if the model learns nothing there is nothing to overfit. |
| The sweep found a defect in its own axis | 23 Aug 2026 | Ben | **`out_bins` was `ceil(η · COF)` in floating point**, which is right on paper and wrong **exactly where `η · COF` is a whole number**: at `COF = 60` the product evaluates to `62.00000000000001` and the ceiling came out **63** — one output bin more than the architecture describes and **120 allocated reals nothing accounts for**. Silent: no stage downstream refuses an extra bin. It fires **only on the perfectly aligned case**, which is the one the formula exists to handle cleanly. Now integer arithmetic, with the exact-ratio definition pinned across thirteen cutoffs. **The deployed cutoff of 5 gives 24.8 and is unaffected**, so no checkpoint and no published number moves — the defect was reachable only by sweeping the axis, which is an argument for sweeping axes. |
| GB-51 | 23 Aug 2026 | Ben | **`experiments/stats.py` — paired Wilcoxon against *its own* reference, and §7.4's "vs persistence" corrected in place.** Persistence forecasts zero and takes no trades, so its `direction` and `sharpe` cells are **empty**: one reference would have tested two of four metrics against a NaN. **The three-reference rule** — MAE against persistence on the same channels, direction against **always-long at that fold's realised up rate**, Sharpe and total return against buy-and-hold. **Pairing is by fold and may not cross a condition.** **Multiple comparisons are counted, corrected and stated**: `n_tests` on every row — the family is every test in the call, so a report narrows the family by narrowing the table — and `p_holm` **beside** the raw `p`. **The floor matters as much as the values:** the exact two-sided test on 16 folds **cannot** return a p below `2/2^16 = 3.05e-5`. A test below six moved folds is **withheld rather than reported powerless**. **Measured on the grid: 79 tests in the real-only family, 11 survive Holm, and all eleven are MAE against persistence with the trained model worse — which is the flatness artefact in significance clothing rather than a finding about the models.** Persistence forecasts **zero**, which is maximally flat, and Spearman(MAE, flatness) = **+1.00** across the COF sweep. **The only results in this study that survive a family-wise correction are measuring flatness, and the winner is the model that forecasts nothing** — the third and cleanest vindication of §7.3's ban, and the first delivered by the significance machinery itself. 20 new tests. |
| The headline claim, and the instrument it does not rest on | 23 Aug 2026 | Ben | **Two claims were being conflated and only one of them is ours.** *(a) No arm beats always-long* is a claim of **absence**: negative in **9 of 9** anchor-arm cells, **−4.3 to −6.5 points**, holding under **both** null controls, with no positive cell anywhere. *(b) Every arm is significantly worse* is a **positive claim of an effect**, and it survives Holm at only **3 of 9** within the pre-specified nine-test family. **§1.4 asked (a); the report makes (a) and does not make (b) anywhere.** **And the sentence about the instrument travels with the claim:** Holm-Bonferroni guards against **false positives**, so applying it to a claim of absence makes it *easier* to conclude nothing was found — **not a protection but a bias toward this study's own conclusion**. It is reported because declining to report it would be worse, and the claim rests on nine negative cells and a five-point gap rather than on a p-value. A reader who knows statistics will look for that sentence and its absence would cost more than the correction does. |
| The determinism test ran the grid it said it would not | 23 Aug 2026 | Ben | Its docstring said *"one condition and one fold"* and its body ran the **whole grid twice** — 26 condition-grids per suite run once GB-50 widened the grid. `study.run` now takes an explicit `design` and the test passes one condition; the property is about the seed reaching every stage and holds for one cell or for none. |
| GB-52 | 23 Aug 2026 | Ben | **`experiments/report.py` — the chapter, from `results.csv` alone.** Nothing in it trains, backtests or reads a cache, so a figure is rebuildable in October without the run or the machine that made it; the significance columns come from `stats.wilcoxon`, a pure function of the same file. **`data_snapshot_last_bar` is the first line of the header**, and a file carrying **two** snapshots is **refused** — a summary across vintages would average two different markets. **Every headline claim is shown at all three anchors and under all three controls in one table**; `CLAIMS` is a list of **effects** rather than prose, so adding a claim means adding a computation a test can check. **Three pairings are structural:** column order from `metrics.columns_for` (made public for this), the rank correlations printed **with** every MAE table, and every reference from `stats.REFERENCES`. The summary **holds `lr` and `COF` at their reference values** rather than averaging over them. Markdown is hand-rolled rather than adding `tabulate`, and a missing number renders as an **empty cell** rather than `nan`. 18 new tests. |
| The overnight residual is printed, not documented | 23 Aug 2026 | Ben | **A ruling nobody sees at the moment it applies is a footnote.** `live_loop.overnight_residual` emits a banner block at session start whenever the book already holds something — the symbols and decision ids, that protection was **NONE between the previous close and this session's first arming**, why (`TimeInForce.DAY`, both escapes refused), and how it is handled (rule 2 re-arms before any entry). A session that opens flat prints nothing. `SessionReport.held_at_open` carries the fact into the session summary, **read before the first reconciliation** — the only moment the book still describes what was carried *into* the session. The cycle records cannot answer it: a position re-armed in cycle 1 looks identical to one opened in cycle 1. |
| The multi-day run needed two things it did not have | 23 Aug 2026 | Ben | **The plan to leave the loop running between sessions assumed a capability the code did not have**, found by planning the run rather than by running it: `run_session` **returns** at the close and returns immediately when called outside a session, so it could not span a day at all. And a **loop that died at 02:00 and a loop correctly idling produce identical output: nothing** — rule 2 bounds the overnight residual only while the process is alive. Three additions. **`heartbeat`**: one line every `live.heartbeat_seconds` (900) while idle — timestamp, state, positions, uptime — with `idle_state` separating **weekend**, **holiday** and **outside session**, three different facts about the same silence. **`run_sessions`**: N sessions in one process, **one report each, separate and in order**, because each carries its own `held_at_open` and a merged report cannot say it twice; **only an ordinary close continues to the next session**. **The idle path runs no cycle** — asserted by counting `run_cycle` invocations across a run that idles overnight and requiring the count to equal the cycles *reported*: ~68 heartbeat lines a day against the ~1,000 cycle records a 60-second idle poll would write. `live.heartbeat_seconds` sits in the `live` section, **not** in `MODEL_SHAPING_SECTIONS`, so no checkpoint is refused — the 19 Aug hash split paying off a second time. 8 new tests. |
| CLAUDE.md: three rules were one rule | 23 Aug 2026 | Ben | **Consolidated into a single principle — *only something that runs is a mechanism* — with three named instances and the incident behind each**: a fact stored twice (five instances, three caught by a test written for something else); a test skippable by a cache, a marker or an environment (the deselected test that would have caught the `results.csv` defect); and a **description** of what the code should do, in a comment, a docstring or a plan (the `"GB-41 adds it"` comment GB-41 did not honour, and the determinism docstring above). Plus the practice: **before any job longer than ten minutes, run the tests that validate its output.** Three rules that rhymed invited being read as three; one principle with three instances cannot be. |
| GB-59, eight weeks early | 23 Aug 2026 | Ben | **A clean clone from the remote reproduces the study and the deployment.** Fresh venv, interpreter and directory: **1129 passed / 0 skipped**; the grid regenerated **877 rows with every result column bit-identical**; `prepare_live` returned a band identical at `lower = 0.026076278765685856` with `model.json` **byte-identical**; `report.py` rebuilt the chapter from `results.csv` alone. **The boundary is named: one machine.** Cross-architecture is **untested**, and `--deterministic` pins BLAS as the documented path at a **measured cost of none** (70.8 s unpinned vs 62.3 s pinned, pinned run cache-warm, so *at most zero*). **Three defects found.** (1) The configuration is mutable during a long run — `settings.yaml` edited at 17:00 under a grid started at 16:36 — closed by `study.provenance` and by `live_loop.config_drift` at every session start, which **does not reload**. (2) The README's install procedure was a note and its author skipped step 3; a non-editable install still imports from the repo root, so everything would have passed while testing an install nobody has — `scripts/setup.ps1` now runs and **verifies** it. (3) A 187-char clone root **dropped a tracked file and exited 0**; pruning one uncited reference log took the longest tracked path from **83 to 66** and the clone-root ceiling from **176 to 193**, so the depth that failed now works. 9 new tests. |
| Window counts come from the batch | 23 Aug 2026 | Ben | **A leakage-prevention decision produces a wrong number rather than an error for anyone measuring from outside it.** `train.py` calls `build_windows` **once over the whole frame** and splits by timestamp, so no second call can receive a second `ChannelStats`. Slicing the frame to a split *first* and windowing after drops the last `input_len + horizon - 1` rows: **2,505 windows the correct way against 1,890 the other**, a **25% undercount** that reached a determinacy calculation before a second measurement caught it. In the builder's docstring, in `ARCHITECTURE.md`, and — because a docstring is not a mechanism — pinned by a test asserting both counts exactly. |
| GB-53 | 24 Aug 2026 | Ben | **The spectral explanation panel — and the FITS gate is structural rather than a model name.** `Attribution.per_frequency` is `None` for DLinear and persistence, so `spectral_panel` returns `""` and the panel is **absent rather than blank** — an empty frame reads as a fault, an absent one as a property of the deployed model. Frequencies keyed by **period in days**, never bin index; the RIN mean is labelled as the mean and not as an `inf`-day cycle; **bin 0 is shown as dead and says why** rather than being omitted; and the response chart carries GB-48's measurement (86% reproduced by a white-noise model, r = +0.9485 across 48 models) as the caption, because a curve read as *what the model learned about the market* is this project's own failure mode committed by the explanation layer. Sign is glyph and geometry, never colour. 11 tests. |
| Phase 2 opens | 24 Aug 2026 | Ben | **Five seams closed, all one shape: a second copy of a fact, replaced by a derivation or a test.** The **dates leave §8** and live only in the expansion's schedule — the two had *already* disagreed (§8 said GATE 2 fell 25 Sep; the expansion ran sessions 24–30 Aug), and the expansion's precedence sentence is a note that resolves it only for a reader holding both documents. §8 keeps the gate criteria, checklists and cancellation rules. **`CLAUDE.md` §0 names both documents** and withdraws *single source of truth* for scope. **GB-64's "enforced by the import contract" was false**: the layers contract puts `glassbox.data` *below* `backtest` and `experiments`, so a provider under `data/` is importable by exactly the two modules GB-64 forbids — adding that `forbidden` contract is now an acceptance criterion. **`Attribution.per_level` amended into §4** for GB-66 rather than reusing `per_frequency`, which would be the two-places defect inside one schema; `per_lag`, retained unused since GB-31 for precisely this, carries the time axis. |
| The registry had a third copy | 24 Aug 2026 | Ben | **The test covering this enumerated two lists by name and could not see a third appear** — the same defect one level up. Rewritten as an AST scan, it found one immediately: `experiments/study.py` held `MODELS = ("persistence", "dlinear", "fits")`, which **agreed** with the registry and would have gone on agreeing until a model was registered, at which point the study would have run three arms and silently omitted the fourth. **That is the GB-44 defect in the same file family**: the runner producing every study number, unable to select the model the study is about — and GB-66 registers `wits` in six days. `MODELS` is now `tuple(ALL_FORECASTERS)`, so the copy is gone rather than pinned; `PER_FOLD_SECONDS` cannot be derived (its values are measurements) so a test pins it, because a model registered without a timing under-reports the wall time the ten-minute rule depends on. The scan catches a complete copy **and a stale one**, proven against both before commit. |
| GB-61, the safe half | 24 Aug 2026 | Ben | **20 symbols cached and quality-checked; the deployed `universe:` deliberately untouched.** Split across GATE 2: the preparatory half runs alongside the sessions because the live loop reads Alpaca through `data/live.py` and never reads `data_cache/`; the `universe:` flip and the grid re-run wait until the loop stops, since editing the deployed config under a running multi-day loop is GB-59's first defect. **Selection rule written into spec §2.4 before any name was chosen**, with all 20 admitted, every excluded candidate recorded against the criterion that excluded it, and the limit the rule does *not* remove stated plainly: it removes performance hindsight, not **existence** hindsight — requiring history from 2016 selects survivors, and the always-long bar is inflated by the same selection. **Measured, not assumed:** `min_history_bars` is **445 at 5 and at 20** (a max over channels, not a sum over symbols — `input_len` 120 + 325 of `rsi14`); GB-5 reports **0 missing bars, 0 duplicates, 1 structural NaN** across all 20; the parity sweep is **500 of 500 byte-identical** at the 445 floor over 25 timestamps each. |
| A second snapshot date, and the `max` that would have hidden it | 24 Aug 2026 | Ben | **Fetching 15 new symbols split the cache's provenance and nothing would have caught it.** The incumbents end 2026-08-13; the new names arrived at 2026-08-21. `_common_index` intersects, so every fold would have been computed **correctly** on the shorter window — while `study.run`'s `snapshot = max(...)` stamped **2026-08-21** onto `data_snapshot_last_bar` in all 877 rows, the provenance GB-57 quotes. Folds right, label wrong, and nothing downstream disagreeing with itself. A `max` reports the newest and *hides* the disagreement, which makes it an instrument blind to the one fault it is positioned to see. The 15 were truncated to the committed snapshot, preserving GB-59's bit-identical claim, and `data_snapshot` now **refuses** a split cache naming both groups. Truncating fixed today; the refusal fixes the next symbol somebody adds. |
| The determinacy table was one fold of sixteen | 24 Aug 2026 | Ben | **Measured off the batch rather than multiplied, and the base was fold 1.** The expansion recorded 2,505 → 10,020 windows and 2.09× → 8.35× determinacy; swept across all sixteen folds the ranges are **2,480–2,505 (mean 2,492)** and **9,920–10,020 (mean 9,968)**, giving **2.07×–2.09×** and **8.27×–8.35×**. Counts fall monotonically as the calendar shortens the training span, so 2,505 is the largest cell, not the typical one. **The ×4 was exactly right and only the base was wrong** — every fold's 20-symbol count is its 5-symbol count times 4.000, because all twenty share one index. Second time this project has taken a window count from the wrong place, and the first time a test was already watching for it. |
| The session log existed only in a terminal | 25 Aug 2026 | Ben | **A session that ran could not be summarised at all.** `main` configured logging with `stream=sys.stdout` and no file handler, so the 24 Aug log lived in a closed terminal: cycles attempted, skip reasons, WARNING and ERROR lines and uptime were all gone, and the only trace on disk was `decisions/2026-08.jsonl` — 10 records, which is **not** 10 cycles and cannot answer any of it. `DailyLogFile` writes `logs/live-YYYY-MM-DD.log`, one file per calendar day, `mode="a"` so *rehearse, stop, restart* appends rather than truncating the first half. **Re-targeted when the day turns rather than named once at startup**, because `--sessions 3` idles through two midnights in one process and a startup-computed filename would put all three sessions in the first day's file and make the other two dates lies. **Two defects found while wiring it.** The summary was `print`ed, not logged, so a file handler alone would have preserved every line *except* the one the gate reads — it now goes through `LOGGER` line by line as the banner already did. And `basicConfig` is a **silent no-op when the root logger already has a handler**, which is this same failure reintroduced by anything touching logging before `main`; `force=True` makes the entry point own the root logger. `--log-dir`/`DEFAULT_LOG_DIR` is an operator path like `DEFAULT_STATE_DIR` rather than a settings key, which would enter every model's config hash. 5 tests. |
| Two loops, one state directory | 26 Aug 2026 | Ben | **The hazard occurred, and what stopped it was an accident that expires.** A stale terminal relaunched a rehearsal against the deployed session's state directory. Nothing was harmed **only** because the bar it would have decided was already decided, so the *one decision per completed bar* rule refused it — and a new bar completes every session. `glassbox/live_lock.py`: `main` takes an exclusive lock before loading a config, so a launch that must not happen never reaches the book; a second launch names the holding PID and mode and exits **3**, distinct from `2`; a lock whose PID is gone is reclaimed and the reclaim is logged, so a hard kill never needs a file deleted by hand. **It could not require restarting the running loop** — that run *is* criterion 6 — so `adopt` writes a lock for a live PID, and when the run ends the file is reclaimed by the stale path: the transitional case degrades into the designed case. Liveness uses `OpenProcess`/`GetExitCodeProcess`, **never `os.kill(pid, 0)`**, which on Windows calls `TerminateProcess` and would have killed the very session this protects — there is a test that probes a live child and asserts it survives. Every unresolvable case refuses: an unreadable lock is not assumed stale. **Proven, not described:** the real hazard command was run against the live directory and refused by name, exit 3, with the refusal in the session log. 17 tests. It also found two existing tests calling `main()` against the **default** state directory — the deployed one — which nothing had said. |
| One protective order, and it is the stop | 26 Aug 2026 | Ben | **The five-rule policy said "verify both legs" and both legs were never possible.** Alpaca refuses every multi-leg class on a fractional quantity (`bracket` *and* `oco`, measured directly rather than inferred from GB-22's bracket result), **and** a working sell order holds the whole position, so a standalone stop and a standalone limit cannot coexist — the second is refused with `insufficient qty available`. Measured against a **97.38-share** position, so it is not fractional-specific: whole-share sizing would not lift it, and fractional removes the *workaround*, not the constraint. Rules 2–4 rewritten: the broker holds the stop, and the loop evaluates the target against **completed daily bars**, exiting at market — which fills at the next open and is exactly `backtest.engine`'s `target_in_loop` arm, so the study's default and the live path describe one system. `_arm_leg` now **raises** on a target leg rather than letting the broker's refusal count as an arming failure; two of those flatten a healthy position, which is what happened to NVDA on 25 Aug. `TARGET_SUFFIX` is still recognised so a leg armed under the old policy is cancelled and attributed rather than orphaned. **26 tests red across three files when `FakeBroker` was made faithful** — a measurement, not a regression. One was **deleted rather than fixed**: `test_a_filled_entry_is_protected_by_two_standalone_orders`, whose premise was impossible, replaced by a one-order test plus a test that no limit is ever armed. 8 new tests for the loop-side target. |
| Execution fidelity measured, not asserted | 26 Aug 2026 | Ben | **39 of 166 trades were target exits worth +25,434.01, and the live loop cannot reach one of them** — it never sees intraday. `target_in_loop` is now a `Condition` axis with `true` the **default** and `false` a spoke. Paired per fold at the reference condition: DLinear −0.003119 → −0.003948, FITS +0.004188 → +0.007019, WITS +0.001801 → +0.004564; trades 166→140, 242→208, 218→155. **The difference is inside fold noise** — mean/(sd/√16) = −0.52, +1.27, +0.47 — which narrows the earlier "32.7% worse" from a single whole-universe run to a point estimate 16 folds cannot resolve. Direction, MAE and flatness are **byte-identical** across arms, the check that the axis touches execution and nothing else. **The null strengthens under the buildable model:** FITS and WITS both score higher on white noise (0.5041, 0.5050) and on shuffled returns (0.5039, 0.5074) than on real data (0.4958, 0.5031), and no model reaches the always-long reference at any anchor. `false` is a spoke, so it runs at anchor 0 only — the divergence has no grid-sensitivity measurement and GB-57 must say so. |
| The axis broke the report, and the fixture was the second copy | 26 Aug 2026 | Ben | **Nine report tests failed on a fixture that hand-listed the condition axes.** `target_in_loop` was not among them, so every synthetic row left it blank, the spoke and the centre differed in **no column at all**, and the paired Wilcoxon read 32 folds where there are 16. Each list was internally consistent, which is why nothing caught it until the pairing crashed. Fixed by **derivation, not by adding a line**: `Condition.columns` is the one list, read by `_row`, `_skipped_row`, the fixture, **and `PAIR_KEYS`**, which is now `tuple(...columns)` so an axis added to `Condition` joins the pairing by construction. Guarded by `test_a_condition_is_identified_by_the_columns_it_writes`, which fails whenever two conditions would write identical columns — the exact shape of this defect. **Same day, same root:** two backtest tests silently changed meaning when the default flipped, because their premise lived in the config rather than in the test; both are now pinned to `target_in_loop=False` explicitly. Both are new instances in CLAUDE.md. |
| The log rolled on UTC and stamped local | 26 Aug 2026 | Ben | `DailyLogFile` names its file for the **UTC** day while `%(asctime)s` defaulted to **local**, so `live-2026-08-25.log` carried lines stamped `2026-08-26 00:00` — a reader looking for 01:00 on the 26th opens the wrong file, which is exactly the confusion a gate log must not have. Both handlers now share one formatter with `converter = time.gmtime` and `datefmt` matching the heartbeat's own `...Z`. Pinned by a test asserting the filename day equals the line day, **asserted rather than commented** because the two copies are three hours apart here and zero apart on a UTC machine, where a convention would look fine and stay wrong. **The running session keeps the old formatter and cannot be fixed without a restart** — but sessions run 13:30–20:00 UTC, the same calendar day in both zones, so **no session line can be misfiled**; only overnight heartbeats, each of which carries an unambiguous `...Z` in its own message body. |
| The honesty layer's third instance, and the rate claim refused | 1 Sep 2026 | Ben | **GB-57 §4 now carries five instances of the honesty layer misreporting itself, not two.** Step 4's label in `live_loop.py` still names the two-leg policy **retired 26 Aug** (Ben's note said 28 Aug; `DECISIONS.md` dates the ruling 26th and the chapter uses that). The session summary had already misreported twice on its own account and both were documented only as docstrings: `positions flattened : 0` on a rehearsal that cancelled a stop and sold the position, because the close-out runs after the cycle list closes; and `open orders at exit : 0`, the cleanest-looking line in the summary, which is exactly what a *failed* close-out prints. Four of the five are fixed by deriving the display from something that already knew better (`fold_range`, `status_colour`, the close-out tuple, `positions at exit`); the step label is recorded **unfixed** because its repair is gated behind GB-61's grid. **The rate comparison was refused rather than written.** "Failed as often as the code it describes" needs a per-component defect rate the project never measured, and a denominator that would have to include the mechanisms that have held — `data_snapshot_last_bar`, `metrics.COMPANIONS`, `is_reportable`. The chapter claims the count and the direction of the error, and says plainly that asserting an unquantified rate *in that paragraph* would be the next instance of the pattern it describes. The duplicated telling of the source-pill instance was consolidated at the same time. |

### Found in the 1 Sep session log — deferred behind GB-61's grid (recorded 1 Sep 2026)

Four findings from reading cycles 135–142 of the 1 Sep live session. **The loop is healthy**
— all ten steps, zero errors, the once-per-bar rule holding, the in-progress bar dropped on
all five symbols. These are behaviour and evidence defects, not failures. **Ben's sequencing
is explicit: after the push and after GB-61's `universe:` flip and grid re-run.** The push is
done (`57a7f41` is on `origin/main`); the grid is **not** — PROGRESS records only GB-61's safe
half, and `results.csv` is still the 26 Aug five-symbol grid. None is urgent and none may
delay the report.

1. **Step 4's label describes a policy that no longer exists.** `live_loop.py:959` reads
   `step 4/10 protection: verify both legs, re-arm, flatten on a second failure`. The two-leg
   policy was retired **26 Aug 2026** (DECISIONS): one protective order reaches the broker and
   it is the stop, the target is evaluated in the loop. This line is the gate evidence log, so
   a reader in October concludes the retired policy is in force. **The sweep is wider than the
   one label** — `live_loop.py` also carries two-leg language at lines **1314** (`Cancel both
   legs and sell at market`), **1780** (`Verify both legs of every managed position`) and
   **2029** (`take it back on that cycle and arm both legs`). Acceptance: a test that the
   protection step's label names **one** leg, plus the full sweep of step labels, log messages
   and docstrings in that file, reported.
2. **Step 9 reads as the opposite of the state.** Step 7 reports `0 undecided, 5 settled` and
   step 9 then says `bars not yet decided`. Reword so the message states what is true: nothing
   to execute because every bar is settled.
3. **The loop re-fetches 3,055 bars every 60 s for a decision that cannot change.** 611 bars
   × 5 symbols, roughly 390 cycles a session, all identical because the last completed bar does
   not move until tomorrow. Steps 1–4 must run every cycle — that is risk management. Steps 5–6
   need only run when a new completed bar might exist. **Report before deciding**: measured API
   calls per session, wall time of steps 5–6 against the cycle total, and whether the fetch can
   be gated on a cheap check that the latest bar has advanced. **The argument against must be
   made, not assumed**: a late correction from the provider is the case for keeping the waste,
   and Ben would rather keep it than introduce a cache that can serve a stale bar.
4. **The cadence drifts.** Intervals measure 64–65 s against `poll_seconds: 60`, because the
   loop sleeps a fixed interval *after* the work rather than targeting a fixed cadence — roughly
   25 fewer cycles a session than the configuration implies. Not a fault, but *cycles per
   session* is a number the gate log quotes and is currently unpredictable. Decide: target the
   cadence, or document the drift. Say which.

**Open recommendation, and it is a tension between two of this project's own rules.** CLAUDE.md
instance 3 says a gap you are not closing now gets a **failing guard** — an `xfail(strict=True)`
or an assertion — because a written note is not a mechanism, and the block above is exactly a
written note. It is a note *by instruction*, since the work is deferred. The cheap way to make
findings 1 and 2 mechanisms without doing the deferred work is a strict-xfail apiece, which
fails the moment somebody "fixes" them silently and costs nothing now. **Not done: it is Ben's
call, because it puts two red-by-design tests in the suite while GATE 2 evidence is being read.**

---

## The result, for GB-57 (measured 26 Aug 2026, snapshot 2026-08-13)

### The turning sentence

> The published backtest describes a system that cannot be built at this venue:
> fractional orders must be simple orders, so a fractional position cannot carry a
> broker-side target. Under the buildable model the trade count falls in every arm —
> **16%, 14%, 29%** — because a delayed exit holds capital and entries that would have
> followed never happen. The return difference is **not resolvable at sixteen folds**.
> The buildable system is measurably **different**, not measurably worse.

**And the structural point, which is what makes the null robust rather than lucky.**
`direction` is a forecast metric and therefore **invariant to the execution axis** —
verified byte-identical across both arms, along with `mae` and `flatness`. The buildable
model could not have rescued the null and did not damage it. The result is the same one,
now established for the system that can actually be built.

**The correction is part of the record.** The first description of this divergence was
"32.7% worse", from a single whole-universe run. Paired per fold it is
`mean/(sd/√16)` = **−0.52, +1.27, +0.47** — all far under 2. The point estimate stands as
a point estimate; the difference is inside fold noise. That is the **fifth** time a first
description in this family has been narrowed by a measurement, and the second time one
already adopted was narrowed rather than merely corrected.

### The headline: the null, and the control that proves the instrument works

Direction accuracy under `target_in_loop=true`, anchor 0, mean over folds and conditions:

| model | real | noise | shuffled | noise − real | shuf − real |
|---|---|---|---|---|---|
| **buy_and_hold** | 0.5560 | **0.5064** | 0.5418 | **−0.0495** | −0.0142 |
| dlinear | 0.5006 | 0.5006 | 0.5021 | +0.0000 | +0.0014 |
| fits | 0.4958 | **0.5041** | 0.5039 | **+0.0083** | +0.0081 |
| wits | 0.5031 | **0.5050** | 0.5074 | **+0.0018** | +0.0043 |

Always-long reference: **0.5395** (anchor 0), 0.5523 (21), 0.5654 (42).

**FITS scores 0.0083 better on white noise than on the market. WITS 0.0018. DLinear is
identical to four decimals — 0.5006 on real data and 0.5006 on noise.** No model reaches
the always-long bar at any anchor, and the pattern is flat across all three:

| model | anchor 0 | anchor 21 | anchor 42 |
|---|---|---|---|
| dlinear | 0.5006 | 0.4973 | 0.5006 |
| fits | 0.4958 | 0.5056 | 0.5095 |
| wits | 0.5031 | 0.5067 | 0.5066 |

**`buy_and_hold` is the only arm that degrades on noise, and that is the control working.**
It is the one arm whose entire content is drift, so it is the one arm that must lose
something when the drift is destroyed — and it loses 4.95 points. A control that moves the
right way for exactly one arm, and for the arm whose mechanism predicts it, is what turns
*"we found nothing"* into ***"we measured that there was nothing to find, and the
instrument can detect something when it is there."***

### The two arms, reference condition, paired per fold (n = 16)

| model | arm | trades | mean fold return | Sharpe | max DD | vs buy&hold |
|---|---|---|---|---|---|---|
| dlinear | false | 166 | −0.003119 | −0.3296 | 0.01321 | −0.08046 |
| dlinear | **true** | 140 | −0.003948 | −0.5900 | 0.01302 | −0.08128 |
| fits | false | 242 | +0.004188 | +0.3774 | 0.01823 | −0.07315 |
| fits | **true** | 208 | +0.007019 | +0.3023 | 0.01622 | −0.07032 |
| wits | false | 218 | +0.001801 | +0.8180 | 0.01642 | −0.07554 |
| wits | **true** | 155 | +0.004564 | +0.7203 | 0.01187 | −0.07277 |
| buy&hold | both | 0 | +0.077337 | +1.3889 | 0.10390 | — |

Paired difference (`true − false`): dlinear −0.000829 (sd 0.006392), fits +0.002832
(sd 0.008933), wits +0.002763 (sd 0.023358).

**Limit that must be stated:** `false` is a spoke, so it runs at **anchor 0 only**. The
divergence itself has no grid-sensitivity measurement. Reporting it as if three anchors
supported it would be the error §7.4 exists to prevent.

### The empty cells are deliberate and say so

**22 rows skipped, every one with its reason in the file:** FITS × `C2_hybrid` ×14, and
WITS × `C2_hybrid` ×8. FITS is univariate and filters frequencies itself, so pre-filtered
wavelet bands are redundant; WITS is univariate and its **first act is a DWT**, so feeding
it the `wav_a1..a3` DWT bands applies the same transform twice. A study that records *why*
a cell is empty is a study a reviewer can check — and the alternative, filling the cells to
square the table, is exactly what spec §6.4 forbids.

**Provenance for every number above:** snapshot `2026-08-13`, 1,110 rows, 768 reportable,
28 conditions, 16 folds, `C0_base` unless stated.

---

## GATE 2 — the process was restarted before session 3 (26 Aug 2026, 13:42 local / 10:42Z)

**Recorded here because the gate log has to say what produced its evidence, and tonight's
session was produced by different code from the two before it.**

**What was restarted.** PID 40040 → 9256, launched 25 Aug 17:47 local under
`--sessions 3`, stopped at 13:42 local on 26 Aug after 19h 52m uptime. Relaunched
immediately as PID 39900 → 9836, same command. **The state was verified clean before the
stop and nothing was lost:** no broker positions, no working orders, `book.json` empty,
`entry_fills.json` empty, `pending.json` empty. A forced stop was required — Windows
cannot deliver a console interrupt to another process's console — and cost only the
summary lines of a session that had not started.

**Sessions 1 and 2 ran under the pre-ruling code. Session 3 does not.** The stopped
process was launched before the ruling of 26 Aug and would have armed **two** protective
legs on any filled entry: a stop and a limit. That is the configuration measured
impossible — a working sell holds the whole position, so the limit is refused with
`insufficient qty available` — and the refusal counts as an arming failure, two of which
flatten a healthy position. Tonight it would have produced a gate log full of errors that
were **already diagnosed and already fixed**, which is noise in the evidence rather than
evidence.

**Criterion 6 is not compromised by the restart.** It asks for *two or three separate live
sessions*, not for one process. Monday 24 Aug was a session, Tuesday 25 Aug was a session,
and tonight is the third. The `--sessions 3` flag remains the mechanism that stops a
session being **missed** because nobody was there to start it, and the restarted process
carries it.

**What the restart bought, none of which the old process could have.** The one protective
order is now the stop and the target is the loop's, so the policy the overnight re-arm
would exercise is one that can actually run — and Wednesday into Thursday is the first
time that re-arm is exercised anywhere but `FakeBroker`. The state directory is locked, so
a stray rehearsal is refused rather than stopped by luck. And the session log is UTC in
both the filename and the line stamps.

**Three things the restart itself demonstrated, on the real system rather than in a test.**
The stale-lock path ran for the first time: the adopted lock naming a dead PID 40040 was
reclaimed and said so —

```
2026-08-26T10:42:20Z WARNING glassbox.live_lock reclaiming a stale lock on
    checkpoints\live: PID 40040 (mode 'deployed') ... is no longer running
2026-08-26T10:42:20Z INFO    glassbox.live_loop state directory checkpoints/live
    locked by PID 9836 (deployed)
```

The timezone fix is visible **inside one file**, which is the clearest form the evidence
could take — `live-2026-08-26.log` carries `2026-08-26 13:40:01,140` from the old process
and `2026-08-26T10:42:20Z` from the new one, three hours apart and the same instant. And
the lock is held by the **worker** (9836) rather than the launcher stub, which is the
process whose death should release it.

**What the gate log must still say about the accidental protection.** On 25 Aug a stray
rehearsal was refused by the *one decision per completed bar* rule, not by any mechanism
built to refuse it. That protection expired today, when a new bar completed. It is now a
lock, and the lock was proven by running the real hazard command against the live state
directory and being refused by name with exit 3.

---

## Scope boundary — the live dashboard pass (ruled 24 Aug 2026, before it starts)

**Recorded before tomorrow, so it is a decision rather than a judgement made while
enjoying the work.**

**What is product and what is not.** **GB-53** — the spectral explanation panel — is in
the spec and blocks four screenshots for the architecture report. It is product and it
lands. The **TradingView-style live pass** — header state, live positions, arriving
decisions, the instrument-panel framing — is **not in the spec**. It was approved and
specified without going through `IDEAS_PARKED.md` first, which is what that file exists
for (CLAUDE.md rule 4).

**The boundary:**

| | |
|---|---|
| **GB-53** | lands as its own commit, before the live pass |
| **the live pass** | gets **tomorrow's session window only** — from when the loop starts until **23:00** — and stops at that boundary **in whatever state it is in** |
| **whatever has not landed by 23:00** | goes into `IDEAS_PARKED.md` with what was done and what was left. **Not carried forward** |
| **Wednesday** | starts **GB-57** |

**Timeboxed, not scope-boxed, and the difference is the whole point:** scope always
expands and time does not. A scope box invites "just one more row"; a clock does not
negotiate.

**Why that window specifically.** The live path is **frozen while the loop runs**, so
nothing else can safely be touched; and the results chapter cannot be written on data not
yet collected. It is the one window in which dashboard work costs nothing else.

**Why it stops at 23:00.** Five of the eight remaining tasks are **writing**; **GB-57 is
about 15% done and is most of what is left**; and the dashboard is **already good enough**
for the screenshots and the demo. A better dashboard does not move the submission and a
missing results chapter does.

## GATE 2 session log

**Written 23 Aug 2026, before the first session runs.** A criterion written after seeing
the outcome is a criterion fitted to it — the same reason GB-57's claims went in before
the grid. **All three sessions are assessed against this list, unchanged**, and it is
walked at 23:00 as a checklist rather than judged while tired. A row is **PASS**, **FAIL**
or **N/A with the reason**; "roughly right" is not one of the options.

Criterion 6 needs two or three separate sessions; the band fires, so an ordinary session
also satisfies criterion 2 by the **deployed system** rather than by a rehearsal.

### The checklist

**A PASS does not advance every criterion, and the two ledgers are separate.** A session
in which the band produced no signal on any symbol passes **every row** — nothing was done
that must not be, nothing omitted that must be — and it advances **criterion 6** while
leaving **criterion 2** exactly where it was. Criterion 2 needs an order placed by the
deployed system; criterion 6 needs sessions run. Record them separately at 23:00, or a
clean Monday reads as more progress than it was.

| # | What | How it is checked | Why it is on the list |
|---|---|---|---|
| 1 | **Cycles completed against cycles expected** | `len(report.cycles)` against the session length in minutes ÷ `poll_seconds` — 16:30–23:00 at 60 s is **390**, less any cycle the market-hours guard correctly refused | A short count is either a crash or a guard firing, and the two look identical in a summary |
| 2 | **One entry decision per symbol per completed bar** | `decisions recorded` = 5 on the first cycle that decides, and `bars already decided` accumulating thereafter; **not** 5 × cycles | The 19 Aug ruling. Before it the loop recorded 1,950 identical rows a day |
| 3 | **Orders submitted, filled, reconciled, adopted** | `orders submitted`, then the fill visible in the book, then `reconcile` agreeing with the broker on the next cycle, and `positions adopted` = 0 unless the account held something the loop did not open | A submission nobody reconciled is a position the system believes in and the broker does not — and an adoption nobody expected is a position with no decision behind it |
| 4 | **Protection armed in the same cycle as the fill** | `positions re-armed` ≥ 1 in the cycle that observed the fill, and **both legs** live at the broker | Rule 1. The open is the wrong minute to be idle |
| 5 | **Both legs verified every cycle** | every cycle reports the position as protected; no cycle logs a naked position | Rule 3. A naked position is a risk event, not a warning |
| 6 | **No arming failure, or exactly the documented response to one** | zero arming failures; if any, `ARMING_STRIKES` behaviour observed and the flatten either fired at two in a row or did not need to | The 20 Aug liquidation came from an arming failure nobody had watched |
| 7 | **No GAP in the heartbeat exceeding two intervals** | not "lines exist at the right spacing" but **the largest gap between consecutive heartbeat timestamps**, which must be under 2 x `heartbeat_seconds`. Compute it; do not eyeball it | **A loop that died at 02:00 and restarted at 08:00 emits correctly spaced lines on both sides of a six-hour hole**, and "present at spacing" passes it. The hole is the exact thing the heartbeat was added to detect, so the check has to be on the gap |
| 8 | **The idle period wrote no cycle records** | the decision store's file count unchanged between the close and the next open | ~68 heartbeat lines a day against the ~1,000 cycle records a 60 s idle poll would write |
| 9 | **One `SessionReport` per session, in order** | `len(reports)` = sessions completed; each has its own `session_id`, cycles and `stopped_by` | A merged report cannot say `held_at_open` twice |
| 10 | **`held_at_open` correct on every session** | session 1 opens flat unless the account already holds something; **session 2 lists exactly what session 1 left open** | Read *before* the first reconciliation — the only moment the book describes what was carried *into* the session |
| 11 | **The overnight banner appears if and only if a position was carried** | `OVERNIGHT RESIDUAL` block present at session 2's open iff session 1 ended holding; absent otherwise | A warning that fires when there is nothing to warn about is one nobody reads |
| 12 | **Rule 2 re-arms the carried position before any entry** | at session 2's open, `positions re-armed` precedes any `orders submitted` **in the same cycle ordering** | This is the moment the protection policy stops being a design and becomes a measurement |
| 13 | **Every decision is explainable** | each decision record carries an exact attribution, residual < 1e-5, and `provenance = live` — never `rehearsal:*` | §4.4 exactness, and the gate's own criterion |
| 14 | **The band's selection context travels** | the banner names the deployed band **with** its validation Sharpe, trade count and grid rank | Ruled 20 Aug: wherever the band appears, its selection context appears with it |
| 15 | **The run stopped for a stated reason** | `stopped_by` is `the market closed` for sessions 1 and 2, and whatever ended session 3 | Only an ordinary close continues to the next session; anything else is a finding |
| 16 | **The config-drift line is present or absent, and which is recorded** | every session banner either carries no `CONFIG DRIFT` block, or carries one naming both hashes; `configuration :` in each session summary reads `matches the file on disk` | Added 23 Aug after the grid was found to have been written under a configuration edited mid-run. **Do not edit `settings.yaml` during the three-day run**; if it happens, this is what says so |

### A failed session versus a session with a finding

**These are different and the difference is decided by this table, not at 23:00.** A
session is assessed as one of three things:

| verdict | what it means | what happens next |
|---|---|---|
| **PASS** | every row PASS or N/A-with-reason | counts towards criterion 6 |
| **PASS, with a finding** | every row PASS or N/A, **and** something was observed that the system handled correctly but that nobody had seen before — a stale symbol excluded, a retried fetch, an adoption, a band declining on a day it was expected to fire | **counts towards criterion 6**, and the finding is written up. A system behaving correctly in a case nobody had exercised is the point of running three sessions rather than one |
| **FAIL** | any row FAIL | does **not** count towards criterion 6, the cause is fixed, and the session is re-run on a later day |

**The trap this table exists to close:** at 23:00 a tired reader wants to call an
unexpected-but-correct behaviour a failure, or a genuine failure a finding. The rule is
mechanical — **a row is FAIL only if the system did something it must not do or failed to
do something it must.** Anything else is a finding.

### What is NOT a failure

- **A sell that filled for a symbol the book does not manage**, logged as *"a sell filled
  for X, which is not a managed holding; no trade emitted — the system has no entry basis
  for a position it did not open"*. **This is pre-existing broker activity on the paper
  account, not something the loop did**, and it appeared in cycle 1 of the first GATE 2
  session for AAPL. Left unqualified it reads as the system doing something odd in its very
  first live cycle, which is the opposite of what happened: the loop found a fill it had no
  entry basis for and **declined to fabricate a trade record**, which is the behaviour GB-29
  specified. The gate log must say which it was rather than leave a reader to guess.
- **A skipped cycle when the broker or the data feed is unreachable**, provided the log says
  *nothing decided, nothing submitted, protective legs unaffected*. See the 24 Aug DECISIONS
  entry: three outages, 55 of 266 cycles skipped, cause local DNS rather than Alpaca. A
  skipped cycle is the loop refusing to act on data it does not have. **A cycle that stalls
  silently for an hour is a different thing and IS a failure** — see row 7 on heartbeat
  spacing, and note that the heartbeat does not cover a stall *inside* a session.
- **The band standing aside on a given day.** A session that correctly declines to trade is
  a passing session for criteria 1, 2, 5, 7–11, 13–15, and **N/A with the reason** for 3, 4,
  6 and 12. It is not a failure of the loop, and recording it as one would be fitting the
  criterion to the desired outcome.
- **A stale symbol excluded from ranking**, provided it is named in the cycle log and its
  protective legs are still verified. That is ruling (a) of GB-26 working.
- **An unreachable broker or feed for one cycle**, provided the retry policy logs it and the
  next cycle proceeds. A domestic connection drops.

### Session results

_Filled in at 23:00 on each of 24, 25 and 26 Aug 2026, against the list above._

| # | 24 Aug | 25 Aug | 26 Aug |
|---|---|---|---|
| 1 | **FAIL** — 211 completed of ~390 expected; 266 attempted, 55 skipped. 90 min lost inside two unbounded HTTP reads | | |
| 2 | PASS — 5 decisions at the 2026-08-21 bar, once each, none re-decided across 266 cycles | | |
| 3 | N/A — the band stood aside on all five; no order submitted | | |
| 4 | N/A — no fill to arm against | | |
| 5 | N/A — flat the whole session | | |
| 6 | N/A — no position to arm | | |
| 7 | PASS **with a finding** — idle heartbeats at exactly 15 min (15:44, 15:59, 16:14, 16:29, 23:00). The 391-min gap is the session itself, by design — **and is exactly why row 7 could not see the 90 min of stalls** | | |
| 8 | PASS — the idle period wrote no cycle records; the decision store gained 5 rows, all from cycle 1 | | |
| 9 | PASS — one report, `session 1 of 3 complete` | | |
| 10 | PASS — opened flat; `book.json` empty at open and at close | | |
| 11 | PASS — no position carried, and no `OVERNIGHT RESIDUAL` block appeared | | |
| 12 | N/A — session 2 has not run | | |
| 13 | PASS — residuals 4.1e-11 to 8.7e-10, all « 1e-5; provenance `live` on all five | | |
| 14 | **FAIL** — the banner prints `thresholds : lower=0.026076 upper=none` and **omits the validation Sharpe (+0.483), the trade count (8) and the grid rank**. `thresholds.json` holds all three. The 20 Aug ruling that the band's selection context travels wherever the band appears **was recorded and never implemented** | | |
| 15 | PASS **with a finding** — the session ended at the close as required, but `stopped_by` is **not printed**, so the reason is inferred from `session 1 of 3 complete` rather than read | | |
| 16 | PASS — no `CONFIG DRIFT` block, and the banner carries `config hash` | | |

**Verdict, 24 Aug: FAIL on two rows, and both are findings worth more than the session.**

**Row 1** is the 90 minutes lost inside two HTTP reads with no timeout — closed the same
night by `glassbox/data/http.py`, measured and ruled at 45 s. **Row 14 is the one to
notice**: a ruling of 20 Aug, written down, never built, and invisible until a checklist
walked the banner line by line. It is the first principle exactly — *a description of what
the code should do is not a mechanism* — and it survived four days and a CI suite because
nothing executed it. It is **not** blocking: the band's context is in `thresholds.json` and
in the gate log, so no decision was made without it. It goes on the list for after GATE 2.

**Neither failure touches criterion 2**, which is where the session leaves the gate: the
deployed band stood aside on all five symbols, decisively rather than marginally — the best
signal reached **14% of the lower bound** and also missed `min_up_points`. Criterion 6
advances by one session; **criterion 2 is exactly where it was**, which is why the rehearsal
is scheduled for 25 Aug at the open.


## Gate log

| Gate | Date | Result | Notes |
|---|---|---|---|
| GATE 1 | 17 Aug 2026 (commitment 11 Sep) | **PASS** | Four checklist items green, four leakage checks green, one qualified. Detail below. |
| GATE 2 | walked 20 Aug 2026 (commitment 25 Sep) | **3 of 6 — not passed; criterion 2 since amended** | Rehearsed five weeks early, so the §8 cancellation is armed rather than fired. Criterion 2 was found to be structurally unreachable and was **amended on 20 Aug** to test the execution path rather than the deployed model. Detail below. |
| GATE 3 | 10 Oct 2026 | pending | **Unchanged.** The submission date does not move. |

### GATE 2 — walked 20 Aug 2026 · 3 of 6 · GB-40

Run five weeks before its commitment date, so this is a **status, not a verdict**. Every
line is answered by running something; nothing here is asserted.

| Item | Result | Evidence |
|---|---|---|
| Live loop runs a full session unattended | **FAIL** | A dry-run cycle and three real cycles on 18 Aug. No open-to-close session. Outside the window `run_session` returns `stopped_by='outside the session'` after 0 cycles — the guard working, not a session. |
| 2. Execution path proven live | **NOT YET RUN** | **Criterion amended 20 Aug** (DECISIONS): it tests the machine, not the model, and may be demonstrated with a rehearsal band. The rehearsal path is built and tested; it has not been run against Alpaca, because the market was shut when the amendment landed. Under the old wording this read FAIL and was unreachable: 0 live decisions produced an order, and none could. The loop has **reconciled** two real Alpaca fills — GB-22's entry (quarantined, correctly) and the operator flatten (dropped, `missing_position`) — but has never **placed** one. |
| 2b. Deployed band behaves correctly | **PASS** | The most recent complete fold stands aside — its best calibrated candidate scored a validation Sharpe of −3.38 over 7 trades — and the loop stands aside with it: `Thresholds.never()` in force, 5 decisions recorded and narrated, 0 orders. The behaviour matches what validation selected. **The gate log states which occurred: it stood aside.** |
| Every decision carries an exact attribution, visible in the dashboard | **PASS** | 75 records live + replay; worst `abs(sum(per_channel) − forecast_total)` = **1.248e-08** against a 1e-5 tolerance. Every record has channels, every one renders, all visible in the table, the bars and the expander titles. |
| Replay reproduces a recorded day offline | **PASS** | Fold 13, 14 bars → 14 cycles, 0 failed, 70 decisions, with `load_live_bars` and `AlpacaBroker` **replaced by functions that raise**. Offline is measured, not assumed. |
| Co-Pilot approve and reject | **PASS, replayed** | 2 queued by the loop; one approved → submitted, both legs armed; one declined → broker order count unchanged. 70 records after, not 72, because an answer amends. Never exercised against Alpaca. |
| Two or three separate live sessions (`SOLO_BUILD_PLAN` §4.1) | **FAIL as walked 20 Aug; in progress since 24 Aug** | One completed bar decided live at the walk: 2026-08-17. Since: Monday 24 Aug, Tuesday 25 Aug, and Wednesday 26 Aug in progress — **two or three, which is what the criterion asks**. Sessions 1 and 2 ran under the pre-ruling two-leg policy; the process was restarted before session 3 and the reason is logged above. |

**The three failures reduce to two causes, and neither is a defect:** the deployed band
stands aside, so no order and no recommendation can exist; and the plan's own floor of two
or three evenings at the desk has not been met.

**Criterion 2 was structurally unreachable, and the criterion was fixed rather than the
system.** A band that stands aside can never place an order, so the old wording would have
read FAIL on 25 Sep exactly as on 20 Aug, and FITS would have been cancelled by a rule aimed
at an unfinished product on a system that had correctly decided not to trade. Ben amended
§8 the same day: criterion 2 now tests the **execution path**, demonstrable with a rehearsal
band, and criterion 2b asks separately whether the **deployed band** behaved as validation
selected. See DECISIONS.md, 20 Aug.

**Criterion 6 stands unchanged and unmet.** It is a floor on observation, not on the model,
and no amendment reaches it. Wednesday to Friday are the sessions available.

**Condition, recorded and NOT exercised.** *The loop must not hold a position overnight
while protective legs are DAY.* No position has been held overnight by the loop, and the
deployed band stands aside, so the condition has never been tested. An untested guarantee is
not a satisfied one. **A rehearsal is now bound by it in code**: `_rehearsal_close_out`
flattens 15 minutes before the exchange close and opens nothing after, so the first run that
could exercise the condition cannot violate it.

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

  **Promoted 20 Aug to a GATE 2 condition** (DECISIONS.md): *the loop must not hold a
  position overnight while protective legs are DAY — either the policy moves to GTC before
  any overnight hold occurs, or the loop flattens at the close.* As of 20 Aug the condition
  is recorded as **never exercised**, because no position has been held overnight and the
  deployed band stands aside.

  **The overnight half is measured, not inferred.**
  Alpaca refuses a `client_order_id` it has seen before **including after the original was
  cancelled or expired** (`40010001`, measured against the paper account). So the DAY legs
  expire at the close, the next morning's re-arm was submitted under yesterday's id, that
  refusal counted as an arming failure, and `ARMING_STRIKES` of them **flattened the
  position at market** — a liquidation two cycles into the session for a reason unconnected
  to the strategy. The id collision is fixed (leg ids now carry an attempt number derived
  from the broker's order history), so the liquidation is gone. **The overnight gap is
  not.** Closing it means `DAY → GTC`, which changes the protection policy recorded on
  18 Aug rather than fixing a bug in it, so it is Ben's ruling and has not been taken.
  Latent rather than realised: no position has been held overnight by the loop, and the
  deployed band stands aside.

_Both GB-30 feed items are **closed** — Ben ruled them blockers for GB-26 and they were
fixed on 18 Aug: the feed is asserted on the request object and logged on every fetch, and
`load_live_bars` now takes the caller's floor. See DECISIONS, 18 Aug._

_The 14 Aug CI failure is resolved — see DECISIONS.md, "Python floor raised to 3.12";
green on 176fe01._

_None open._
