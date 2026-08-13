# GlassBox Trader — Solo Build Plan

**Builder:** Ben, alone · **Window:** 15 Aug – 10 Oct 2026 · **56 days**
**Companion documents:** `docs/GLASSBOX_PROJECT_SPEC.md` (the contract) · `prompts/SPRINT_*.md` (the execution)

> Jira still shows the 70/30 split with Noy for the formal record. **This document is
> the real build order**, written on the assumption that one pair of hands writes
> every line. Read it before starting, and re-read the current sprint section on the
> first day of each sprint.

---

## 1. The capacity problem, stated honestly

The two-person plan is 216 story points, where one point is roughly half a day of
focused work. That is 108 working days. You have about 40.

Claude Code does not multiply your capacity by three. What it does reliably is
compress *implementation* time on tasks that are specified precisely enough to have
one correct answer — which is exactly why every prompt in this pack names the
function signature, the acceptance test and the failure mode. It does not compress
the time you spend reading diffs, debugging a live broker connection at 22:00, or
writing the report.

Assume an effective capacity of **110–120 points**. Everything below is scoped to fit.

## 2. What was cut, and why

| Cut | Original | Reason |
|---|---|---|
| `NLinear` forecaster | GB-14, 2 SP | DLinear alone satisfies the baseline requirement |
| Per-lag attribution heatmap | GB-31, 3 SP | Per-channel attribution already proves exactness; the heatmap is presentation polish |
| Backcast/forecast supervision toggle | GB-43, 3 SP | Hardcode `B+F`. The ablation is interesting and not essential. |
| Full order-lifecycle state machine | GB-23, 5 → 2 SP | Reduce to: reconcile positions against the broker on every cycle, resolve in favour of the broker |
| Co-Pilot expiry and three-path logging | GB-37, 5 → 2 SP | Reduce to approve / reject. No timer. |
| Fault-handling depth | GB-39, 3 → 2 SP | Retry with backoff and restart safety only |
| Dashboard scope | GB-34/35/36, 13 → 8 SP | Positions, forecast chart, decision log with attribution bars. Nothing else. |
| Spectral dashboard panel | GB-53, 5 → 3 SP | Frequency bars and the response curve. No live gain/phase table. |
| Report chapters | GB-55/56/57, 15 → 11 SP | You write all of them, so there is no review round-trip |

**Total after cuts: ~118 SP.** Tight, but real.

## 3. Three rules that protect the schedule

**Never work ahead of a gate.** If Gate 1 is red on 11 September, Sprint 3 does not
start. Fix or cut. The gate is what converts "I am behind" into a decision rather
than a slow drift into a failed submission.

**One task, one session, one commit.** Do not batch. A session that has run for two
hours has drifted from `CLAUDE.md` and will start inventing.

**Friday is not a build day.** Reserve the last working day of each week for reading
your own diffs, fixing what the tests missed, and updating `PROGRESS.md`. The plan
below already assumes this.

---

## 4. Sprint 1 — Foundations · 15–28 August

**Goal:** the data and feature layers exist, are causally correct, and are proven so
by tests. Nothing forecasts yet.

| Day | Date | Task | SP | What lands |
|---|---|---|---|---|
| 1 | Sat 15 Aug | GB-1 | 2 | Repo scaffold, tooling, CI green |
| 2 | Sun 16 Aug | GB-2 | 2 | `settings.yaml` + typed loader, fails loud |
| 3 | Mon 17 Aug | GB-3 | 3 | Frozen schemas, `Forecaster` protocol, layer test |
| 4 | Tue 18 Aug | GB-4 | 3 | Historical loader + parquet cache |
| 5 | Wed 19 Aug | GB-5 | 2 | Data quality report |
| 6 | Thu 20 Aug | GB-6 | 2 | Alpaca paper account, secrets handling |
| 7 | Fri 21 Aug | — | — | **Review day.** Read every diff so far. |
| 8 | Sat 22 Aug | GB-7 | 3 | Live data client, schema identical to historical |
| 9 | Sun 23 Aug | GB-8 | 3 | Four indicators, trailing windows only |
| 10–12 | Mon–Wed 24–26 Aug | **GB-9** | 5 | **The builder.** The most important module. |
| 13 | Thu 27 Aug | GB-10 | 3 | Causality harness, proven to have teeth |
| 14 | Fri 28 Aug | GB-11, GB-12 | 5 | Persistence + contract test, README |

**Sprint 1 is done when:** `pytest` is green, CI is green, the causality harness
rejects a deliberately leaky function, and the contract test passes for
`PersistenceForecaster`.

> **GB-9 gets three days on purpose.** Every module downstream inherits its
> correctness, and its contract is expensive to change once the model layer depends
> on it. If it slips, take a fourth day from GB-12 rather than rushing it.

---

## 5. Sprint 2 — Offline vertical slice · 29 August – 11 September → **GATE 1**

**Goal:** one command runs data → features → DLinear → backtest → metrics, with the
persistence baseline on the same folds.

| Day | Date | Task | SP | What lands |
|---|---|---|---|---|
| 1–3 | Sat–Mon 29–31 Aug | GB-13 | 5 | `DLinearForecaster`, passes the contract test |
| 4–5 | Tue–Wed 1–2 Sep | GB-15 | 5 | Training loop, seeded, reproducible |
| 6 | Thu 3 Sep | GB-16 | 2 | Inference, batch and single-window |
| 7 | Fri 4 Sep | — | — | **Review day.** |
| 8 | Sat 5 Sep | GB-17 | 3 | Walk-forward fold generator |
| 9–11 | Sun–Tue 6–8 Sep | **GB-18** | 8 | **The backtester.** Fees, slippage, SL/TP. |
| 12 | Wed 9 Sep | GB-19 | 3 | Metrics, every one reporting a delta |
| 13 | Thu 10 Sep | GB-20 | 5 | Signal engine, thresholds calibrated on validation |
| 14 | Fri 11 Sep | GB-24, GB-25 | 5 | `smoke_offline`, **GATE 1 review** |

### GATE 1 — 11 September

- [ ] `python -m glassbox.smoke_offline` runs the full offline path in one command
- [ ] Persistence baseline produces numbers on the same folds
- [ ] `test_no_lookahead.py` and `test_forecaster_contract.py` green
- [ ] At least one walk-forward fold completes end to end

**If red:** stop. Sprint 3 does not begin. The most likely culprits are GB-18 and
GB-20 — the backtester is the largest single task in the project, and threshold
calibration is where leakage hides.

---

## 6. Sprint 3 — Live end to end · 12–25 September → **GATE 2**

**Goal:** the system is a product. It trades unattended and explains itself.

| Day | Date | Task | SP | What lands |
|---|---|---|---|---|
| 1 | Sat 12 Sep | GB-21 | 3 | Risk layer, property-tested caps |
| 2–3 | Sun–Mon 13–14 Sep | GB-22, GB-23 | 7 | Executor + broker reconciliation |
| 4–5 | Tue–Wed 15–16 Sep | GB-26 | 5 | Live loop, unattended session |
| 6 | Thu 17 Sep | GB-27, GB-28 | 5 | **Parity test**, ranking |
| 7 | Fri 18 Sep | — | — | **Review day.** |
| 8 | Sat 19 Sep | GB-29 | 3 | Decision records + config hashing |
| 9–10 | Sun–Mon 20–21 Sep | GB-30, GB-33 | 7 | **Exact attribution** + exactness test |
| 11 | Tue 22 Sep | GB-32 | 3 | Natural-language narration |
| 12–13 | Wed–Thu 23–24 Sep | GB-34/35/36 | 8 | Dashboard: positions, forecast, decision log |
| 14 | Fri 25 Sep | GB-37/38/39, GB-40 | 6 | Co-Pilot, Replay, faults, **GATE 2** |

### GATE 2 — 25 September · the decisive checkpoint

- [ ] Live loop runs a full session against Alpaca paper unattended
- [ ] At least one order placed, filled and reconciled
- [ ] Every decision carries an exact attribution visible in the dashboard
- [ ] Replay mode reproduces a recorded day with no network
- [ ] Co-Pilot approve/reject works

**If red: FITS is cancelled, not postponed.** Sprint 4 becomes hardening and the
report. You submit a working v1 with a DLinear-versus-persistence comparison. That
is a complete, defensible project — the proposal says so explicitly, so nobody is
surprised.

> **GB-27, the parity test, is the highest-leverage hour of Sprint 3.** If training
> and live assemble windows differently, everything downstream is subtly wrong and
> nothing will tell you. Do not skip it because the loop "seems to work".

---

## 7. Sprint 4 — FITS, study, report · 26 September – 10 October → **GATE 3**

**Goal:** the research arm lands, the study runs, the report is written.

| Day | Date | Task | SP | What lands |
|---|---|---|---|---|
| 1–3 | Sat–Mon 26–28 Sep | **GB-41, GB-42** | 10 | **FITS core + the amplitude test** |
| 4 | Tue 29 Sep | GB-44 | 3 | FITS integrated, shared weights across the universe |
| 5–6 | Wed–Thu 30 Sep–1 Oct | GB-45, GB-46 | 8 | Spectral attribution + frequency-response plot |
| 7 | Fri 2 Oct | GB-47, GB-48 | 8 | Causal wavelets + causality/additivity tests |
| 8 | Sat 3 Oct | GB-49, GB-50, GB-51 | 11 | Study runner, COF sweep, significance tests |
| — | **Fri 3 Oct** | — | — | **CODE FREEZE.** Anything not working is future work. |
| 9 | Sun 4 Oct | GB-52, GB-53 | 6 | Report generator, spectral dashboard panel |
| 10–12 | Mon–Wed 5–7 Oct | GB-55, GB-56, GB-57 | 11 | The technical report |
| 13 | Thu 8 Oct | GB-58, GB-54 | 5 | Deck + two demo rehearsals |
| 14 | Fri 9 Oct | GB-59 | 3 | Reproducibility audit from a clean clone |
| 15 | Sat 10 Oct | GB-60 | 2 | **GATE 3** — tag `v1.0-submission`, submit |

### The Sprint 4 rule

Anything not working by **3 October** does not enter the report as a result. It is
written up as declared future work. The last week is for writing, and the report is
what you are actually graded on.

> **GB-42 before any training run.** The irFFT amplitude trap costs a full day if you
> discover it after training, because the symptom — systematically flat forecasts with
> a slightly *better* MSE — looks like a modelling problem rather than a bug. Write the
> sinusoid test first.

---

## 8. Where the schedule will actually hurt

| Risk | When | What you do |
|---|---|---|
| GB-18 backtester overruns | Sprint 2 | Cut SL/TP intrabar handling to end-of-bar. Document it. |
| Alpaca behaves unexpectedly | Sprint 3 | Replay mode is the demo. Do not let a broker quirk block the gate. |
| Attribution does not sum exactly | Sprint 3 | Almost always a normalisation term left out of the decomposition. Check the RIN inverse. |
| FITS forecasts look flat | Sprint 4 | GB-42. Every time. |
| Report runs out of days | Sprint 4 | Cut GB-53 (spectral panel) first — it is the only Sprint 4 item that is purely presentational |

## 9. The daily loop

```
morning   open a fresh Claude Code session
          paste the session-start prompt
          confirm which GB-NN you are on
build     paste the task prompt · Plan mode for anything over 2 SP
          read the plan · push back · let it build
verify    pytest · ruff check . · black --check .
          READ THE DIFF. Every line.
close     paste the task-close prompt
          update PROGRESS.md · commit with the GB-NN prefix
```

Two habits decide whether this works:

**Read every diff.** You are the only reviewer. A line you cannot explain to your
supervisor does not go in.

**Update `PROGRESS.md` every single time.** It is the project's memory between
sessions. A session that cannot locate the current phase will guess, and a guessing
agent in a codebase with frozen contracts is how contracts get quietly broken.
