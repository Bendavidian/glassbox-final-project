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

**Never work ahead of a gate.** If Gate 1 is red, Sprint 3 does not start. Fix or cut.
The gate is what converts "I am behind" into a decision rather than a slow drift into a
failed submission.

**One task, one session, one commit.** Do not batch. A session that has run for two
hours has drifted from `CLAUDE.md` and will start inventing.

**Running ahead is buffer, not a new deadline.** Sprint 1 closed on day 6 of a 14-day
budget. That does not move the submission date; it moves the *risk*. Spend the surplus
on the two things that cannot be compressed later — live market sessions and review
bandwidth (§4) — not on starting Sprint 2 tasks in a hurry.

---

## 4. Two constraints that do not compress

Everything else in this plan is elastic. These two are not, and both bite late, which is
why they are stated before the sequences rather than after.

### 4.1 Live market sessions

`GB-22` (first paper order) and `GB-26` (unattended live loop) cannot be tested against
anything but a real session. The US market is open **16:30–23:00 Israel time, weekdays
only**, and `GATE 2` cannot be called on a single successful run: it needs **at least two
or three separate sessions**, because the failures that matter — a partial fill, a
reconnect, an in-progress bar arriving mid-cycle — do not all appear on the first evening.

Consequences for scheduling:

- **Land `GB-22` and `GB-26` early in a week, never on a Friday.** A Friday landing means
  the first real test is the following Monday and three sessions become a nine-day tail.
- A session is an evening of *your* time, not the agent's. Two hours from 16:30, at the
  desk, watching.
- Sprint 3 needs at least one full working week with market sessions in it. Compressing
  Sprint 3 below that does not buy time, it just moves `GATE 2` past the sessions that
  would have validated it.

### 4.2 Review bandwidth

The quality mechanism in this project is not the test suite. It is that **every diff is
read, line by line, by one person** — which is also the mechanism that caught the
credential in `.env.example` after the suite had gone green.

That mechanism has a hard ceiling: **three tasks per working session, maximum.** Past
three, diffs get skimmed, and a skimmed diff is worse than an unreviewed one because it
carries the confidence of a review without the substance. When a day's tasks are large
(`GB-18`, `GB-41`), the cap is one.

This is why "Claude Code makes me faster" has a limit that is not about the agent.

---

## 5. Gate targets

Sprint 1 finished ahead of plan, so every gate has a revised target. **The original date
is the commitment; the revised date is the intent.** A gate called early is a gate whose
slack goes to the next sprint. A gate missed against the revised date is not late — it is
late only against the original.

| Gate | Revised target | Original commitment | Meaning |
|---|---|---|---|
| **GATE 1** | ~22 Aug | 11 Sep | The offline slice is real |
| **GATE 2** | ~1–3 Sep | 25 Sep | The system is a product — **the decisive one** |
| **GATE 3** | 10 Oct | 10 Oct | Submitted. **Unchanged, and not negotiable.** |

`GATE 3` does not move, because the submission date does not move. The three weeks the
revised targets open up are not spare capacity to fill with scope — §2's cuts stay cut.
They are there so that Sprint 4's report has room and so that `GATE 2` can absorb the
market sessions §4.1 describes without pressure.

---

## 6. Sprint 1 — Foundations · **complete**

**Goal:** the data and feature layers exist, are causally correct, and are proven so by
tests. Nothing forecasts yet.

```
GB-1 scaffold ─→ GB-2 config ─→ GB-3 contracts ─┬─→ GB-4 historical ─→ GB-5 quality
                                                │        │
                                                │        └─→ GB-6 credentials ─→ GB-7 live
                                                │
                                                └─→ GB-8 indicators ─→ GB-9 BUILDER ─┬─→ GB-10 causality
                                                                                     └─→ GB-11 persistence
                                                                                            │
                                                                                     GB-12 docs ←┘
```

All twelve landed on 14 Aug. See `PROGRESS.md` for the per-task record and the Sprint 1
review, which lists the four findings that were the sprint's real output.

---

## 7. Sprint 2 — Offline vertical slice → **GATE 1** (~22 Aug)

**Goal:** one command runs data → features → DLinear → backtest → metrics, with the
persistence baseline on the same folds.

**Two independent chains.** They only meet at `GB-20`, which is worth knowing on the day
one of them blocks — if DLinear is fighting you, the harness chain is not waiting on it.

```
model chain      GB-13 DLinear ─→ GB-15 train ─→ GB-16 predict ─┐
                   (needs GB-9, GB-11)                          ├─→ GB-20 signal ─→ GB-24 smoke ─→ GB-25 GATE 1
harness chain    GB-17 walkforward ─→ GB-18 BACKTEST ─→ GB-19 metrics ─┘
                   (needs GB-4 only)
```

**Start with `GB-17` → `GB-18`, not with `GB-13`.** The backtester is the largest single
task in the project at 8 SP and depends on nothing in the model layer. Front-loading it
puts the biggest schedule risk where there is still room to absorb it, and `GB-13` is the
better-specified task of the two, so it suffers less from being second.

### GATE 1 — target ~22 Aug (commitment 11 Sep)

- [ ] `python -m glassbox.smoke_offline` runs the full offline path in one command
- [ ] Persistence baseline produces numbers on the same folds
- [ ] `test_no_lookahead.py` and `test_forecaster_contract.py` green
- [ ] At least one walk-forward fold completes end to end

**If red:** stop. Sprint 3 does not begin. The most likely culprits are `GB-18` and
`GB-20` — the backtester is the largest single task, and threshold calibration is where
leakage hides.

---

## 8. Sprint 3 — Live end to end → **GATE 2** (~1–3 Sep)

**Goal:** the system is a product. It trades unattended and explains itself.

```
GB-21 risk ─→ GB-22 executor ★ ─→ GB-23 reconcile ─→ GB-26 live loop ★ ─→ GB-27 PARITY
                                                            │
                                                            ├─→ GB-28 rank ─→ GB-29 records
                                                            │
                                                            └─→ GB-30 attribution ─→ GB-33 exactness ─→ GB-32 narrate
                                                                                                            │
                                    GB-34/35/36 dashboard ←──────────────────────────────────────────────────┘
                                            │
                                            └─→ GB-37 co-pilot · GB-38 replay · GB-39 faults ─→ GB-40 GATE 2

★ needs a live market session — see §4.1. Land these early in a week, never on a Friday.
```

### GATE 2 — target ~1–3 Sep (commitment 25 Sep) · the decisive checkpoint

- [ ] Live loop runs a full session against Alpaca paper unattended
- [ ] At least one order placed, filled and reconciled
- [ ] Every decision carries an exact attribution visible in the dashboard
- [ ] Replay mode reproduces a recorded day with no network
- [ ] Co-Pilot approve/reject works
- [ ] **Two or three separate live sessions, not one** (§4.1)

**If red: FITS is cancelled, not postponed.** Sprint 4 becomes hardening and the report.
You submit a working v1 with a DLinear-versus-persistence comparison. That is a complete,
defensible project — the proposal says so explicitly, so nobody is surprised.

> **`GB-27`, the parity test, is the highest-leverage hour of Sprint 3.** If training and
> live assemble windows differently, everything downstream is subtly wrong and nothing
> will tell you. Do not skip it because the loop "seems to work". GB-9 already found the
> mechanism: a live fetch shorter than `min_history_bars(cfg)` produces values that differ
> from training at the same timestamp, silently.

---

## 9. Sprint 4 — FITS, study, report → **GATE 3** (10 Oct, unchanged)

**Goal:** the research arm lands, the study runs, the report is written.

```
GB-42 sinusoid test ─→ GB-41 FITS core ─→ GB-44 shared weights ─→ GB-45 spectral ─→ GB-46 response plot
   (write it FIRST)                                                                          │
GB-47 wavelets ─→ GB-48 causality (reuse assert_causal)                                       │
   │                                                                                          │
   └────────────────┬─────────────────────────────────────────────────────────────────────────┘
                    ↓
       GB-49 study ─→ GB-50 COF sweep ─→ GB-51 Wilcoxon ─→ GB-52 report generator · GB-53 spectral panel
                    ↓
            ═══ CODE FREEZE — 3 Oct ═══
                    ↓
       GB-55 · GB-56 · GB-57 report ─→ GB-54 rehearsals · GB-58 deck ─→ GB-59 clean-clone audit ─→ GB-60 GATE 3
```

### The Sprint 4 rule

Anything not working by **3 October** does not enter the report as a result. It is written
up as declared future work. The last week is for writing, and the report is what you are
actually graded on. This date does **not** move with the revised gate targets — an early
`GATE 2` buys report time, it does not buy build time.

> **`GB-42` before any training run** — note it comes *before* `GB-41` in the sequence
> above, and that is deliberate. The irFFT amplitude trap costs a full day if you discover
> it after training, because the symptom — systematically flat forecasts with a slightly
> *better* MSE — looks like a modelling problem rather than a bug.

---

## 10. Where the schedule will actually hurt

| Risk | When | What you do |
|---|---|---|
| GB-18 backtester overruns | Sprint 2 | Cut SL/TP intrabar handling to end-of-bar. Document it. |
| Alpaca behaves unexpectedly | Sprint 3 | Replay mode is the demo. Do not let a broker quirk block the gate. |
| Attribution does not sum exactly | Sprint 3 | Almost always a normalisation term left out of the decomposition. Check the RIN inverse. |
| FITS forecasts look flat | Sprint 4 | GB-42. Every time. |
| Report runs out of days | Sprint 4 | Cut GB-53 (spectral panel) first — it is the only Sprint 4 item that is purely presentational |

## 11. The daily loop

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
