# Sprint 3 Prompts — Live End to End

**12–25 September 2026 · 14 tasks · ends at GATE 2**

**Goal:** the system becomes a product. It trades unattended and explains itself.

This sprint decides whether FITS happens. Protect it.

---

## Day 1 · GB-21 · Risk layer · 3 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-21.

Implement engine/risk.py:
- size_positions(signals, equity, prices, cfg) -> list[Order]
- Cap each position at cfg.risk.max_position_pct of equity
- Cap total gross exposure at cfg.risk.max_gross_exposure
- Attach stop_loss and take_profit levels to every order from cfg.risk
- Risk runs AFTER ranking and BEFORE execution. It may reject or shrink an order.
  It may never create one.

Replace the injected sizing function GB-18 used with this real implementation.

Write tests/test_risk.py using Hypothesis:
- Property test over randomised equity, prices and signal sets: no combination
  produces a position above max_position_pct or total exposure above
  max_gross_exposure
- Zero equity produces no orders rather than raising
- Stop and take-profit levels are on the correct side of the entry price

Acceptance: the property test finds no violating input across at least 500 examples.
```

---

## Days 2–3 · GB-22 + GB-23 · Execution and reconciliation · 7 SP

**GB-22:**

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-22.

Implement engine/executor.py submitting orders to Alpaca paper.

Wrap the Alpaca SDK behind a thin Broker interface (submit, cancel, get_positions,
get_orders, get_account) so a FakeBroker can be substituted in tests. This matters:
you cannot run the test suite against a live API on every commit.

- Market orders with attached stop-loss and take-profit (bracket orders)
- Log every request and response with the decision ID
- Two modes: "auto" submits immediately, "co_pilot" returns a pending recommendation
  without submitting. Implement only the auto path here; GB-37 adds the approval flow.

Write tests using FakeBroker covering submission, rejection handling, and that a
co_pilot-mode call submits nothing.

Acceptance: one real paper order placed, filled and confirmed via the API — report
the order ID and fill price. FakeBroker tests pass with no network.
```

**GB-23 (reduced scope — 2 SP, not 5):**

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-23.

Implement position reconciliation, scoped tightly:
- On start-up and on every cycle, fetch positions and open orders from the broker
- Compare against local state
- Resolve any divergence IN FAVOUR OF THE BROKER, and log the divergence loudly
- Handle partial fills by trusting the broker's reported quantity

Do NOT build a full order-lifecycle state machine. Reconcile-from-truth is enough for
this project and it is what actually prevents the failure mode we care about.

Write a test with FakeBroker where local state is deliberately desynchronised, and
assert it is corrected on the next cycle.

Acceptance: killing the process mid-cycle and restarting produces local state matching
the broker. Test this for real against the paper account and report what happened.
```

---

## Days 4–5 · GB-26 · The live loop · 5 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-26.
Use Plan mode.

Implement live_loop.py executing the seven-step cycle from spec §3.5 every
cfg.live.poll_seconds during the configured Israel-time market window:

1. Pull latest daily bars for the universe (data/live.py)
2. builder.build_windows(..., as_of=now) per symbol — the SAME function training uses
3. forecaster.predict
4. signal → rank → risk
5. explain (stub for now; GB-30 fills it in)
6. execute, or hold if co-pilot
7. persist the decision record (stub; GB-29 fills it in)

Requirements:
- Skip cleanly outside market hours and on NYSE holidays
- Structured logging at every step with a cycle ID
- A --dry-run flag that runs everything except order submission
- Graceful shutdown on SIGINT that leaves no in-flight order unaccounted for

Acceptance: runs a complete session unattended with --dry-run. Report the number of
cycles completed and any step that failed. Then run it live for one session.
```

---

## Day 6 · GB-27 + GB-28 · Parity test and ranking · 5 SP

> **GB-27 is the highest-leverage hour of this sprint.** If training and live assemble
> windows differently, everything downstream is subtly wrong and nothing will tell you.

**GB-27:**

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-27.

Write tests/test_train_live_parity.py.

Take a historical timestamp. Build the model input window twice:
- through the training path: build_windows(historical_frame, cfg, as_of=None), then
  select the window ending at that timestamp
- through the live path: build_windows(replayed_live_frame, cfg, as_of=timestamp)

where replayed_live_frame is the same data passed through data/live.py's schema
normalisation, simulating what the live loop would receive.

Assert np.array_equal on the X arrays — byte-identical, not approximately equal.
Assert the dtype is float32 on both. Assert the channel tuples match exactly.

Then prove the test has teeth: introduce a deliberate one-bar offset in the live path,
confirm the test fails, and remove it. Report what you observed.

Acceptance: byte-identical arrays; the test fails on a deliberate offset.

If this test does not pass, stop and tell me. Do not adjust the test to make it pass.
```

**GB-28:**

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-28.

Implement engine/rank.py: rank the universe by forecast trend strength and select the
top cfg.signal.top_k names. Ordering must be deterministic, with ties broken stably by
symbol name so repeated runs never differ.

Write tests asserting identical inputs always produce identical ordering, and that
ties resolve alphabetically.

Acceptance: deterministic ordering; tie-breaking covered by a test.
```

---

## Day 7 · Review day

```
The live loop now runs. Review it for the failure modes that matter:
1. Any path where the live loop could see data the training path could not
2. Any place an order could be submitted twice
3. Any state that would be lost or corrupted on an unclean shutdown
4. Any silent exception swallow

Do not fix anything yet. Give me the list, ordered by how badly it would hurt.
```

---

## Day 8 · GB-29 · Decision records · 3 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-29.

Persist every DecisionRecord (spec §4.2) as one line of JSONL under
decisions/YYYY-MM.jsonl. Include config_hash from GB-2 so any record ties to the exact
settings that produced it.

Implement:
- save_decision(record) -> None
- load_decisions(start, end) -> list[DecisionRecord]

Numpy arrays serialise as lists; timestamps as ISO-8601 UTC. Round-trip must be lossless.

This is the data source for both Replay mode (GB-38) and the dashboard decision log
(GB-36), so get the schema right now.

Write tests: a record survives save → load with every field intact including the
forecast path and the attribution; changing any config value changes the hash.

Acceptance: lossless round-trip. Report the size of one record in bytes.
```

---

## Days 9–10 · GB-30 + GB-33 · Exact attribution · 7 SP

> The intellectual core of the project. Plan mode.

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-30.
Use Plan mode.

Implement explain/channel.py.

For a linear forecaster, the contribution of channel c to the forecast is that
channel's own linear map applied to its own input window. These contributions sum
EXACTLY to the forecast — it is an algebraic identity, not an estimate.

    def attribute(forecaster, x, channels) -> Attribution

Compute per_channel contributions and their percentage shares. Fill per_lag with a
zero matrix of the right shape for now (the heatmap is cut from scope).

Critical details:
- Reversible normalisation must be accounted for. If the model subtracts a mean and
  adds it back, that constant is part of the forecast and must appear in the
  decomposition or the sum will not close. This is the most common reason the
  exactness assertion fails — check it first if you see a mismatch.
- Never import shap, lime, captum or any perturbation-based library. If you find
  yourself reaching for one, the decomposition is wrong.

Move DLinear's inline explain logic here and have the model delegate to it.

Acceptance: sum of per_channel contributions equals the forecast within 1e-5 for one
thousand random windows. Report the maximum observed deviation across those runs.
```

**Then GB-33:**

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-33.

Extend tests/test_forecaster_contract.py so the exactness assertion runs for every
registered forecaster over 1000 random windows each, not just one.

Make sure the structure means FITS (GB-41) is covered automatically when registered,
with no change to this file.

Acceptance: parameterised test green for Persistence and DLinear; the registry
mechanism verified by adding a dummy forecaster, confirming it is picked up, and
removing it.
```

---

## Day 11 · GB-32 · Natural-language narration · 3 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-32.

Implement explain/narrate.py turning an Attribution and a Signal into readable prose,
covering the three parts from spec §4:
1. The forecast path — "predicting a 1.8% rise over the next 4 days"
2. The contributing channels, ranked by share — "62% from the RSI channel, which is
   leaving oversold territory"
3. The action, size and stop — "entered long on AAPL at 7% of equity, stop at -3%"

Support English and Hebrew output, selected by argument.

Requirements:
- Percentages in the prose must be computed from the Attribution object, never
  recomputed independently — they must agree exactly with what the dashboard shows
- A "hold" decision produces a sentence explaining why the threshold was not met
- No claim about profitability, ever

Write tests rendering three representative decisions (enter, hold, exit) in both
languages and asserting the percentages match the Attribution.

Acceptance: all three render correctly in both languages.
```

---

## Days 12–13 · GB-34 + GB-35 + GB-36 · Dashboard · 8 SP

> Three tasks, two days. Streamlit is fast. Resist adding anything not listed.

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement tasks GB-34, GB-35
and GB-36 together in dashboard/app.py.

GB-34 — layout and state:
- Current positions with live PnL, account equity, bot status (running / idle /
  outside market hours), the active model and channel configuration
- Auto-refresh on an interval

GB-35 — forecast path per symbol:
- Recent price history with the predicted H-day return path extending beyond it
- The calibrated entry threshold marked

GB-36 — decision log with attribution:
- Chronological list of decisions read from the JSONL store (GB-29)
- Expandable per decision showing: the narrative from GB-32, and a horizontal bar
  chart of per-channel contributions with percentage labels

Use the project colour encoding: darker tones for slow-frequency / trend components,
lighter for fast. Keep the palette consistent with the specification document so the
report figures and the live dashboard read as one product.

Scope discipline: implement exactly these three things. No settings page, no manual
order entry, no equity curve chart, no symbol search. Those are not in scope and this
sprint has no slack.

Acceptance: live positions visible and refreshing; the forecast chart renders for all
five symbols with history aligned to correct dates; every logged decision is
inspectable and its contribution bars sum visibly to 100%.
```

---

## Day 14 · GB-37 + GB-38 + GB-39 + GB-40 · Co-Pilot, Replay, faults, GATE 2 · 6 SP

**GB-37 (reduced — 2 SP):**

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-37.

Implement Co-Pilot mode, scoped tightly: when cfg.live.mode is "co_pilot", the loop
surfaces the explained recommendation in the dashboard with Approve and Reject buttons
instead of executing. Approve submits; Reject does not. Both write a decision record.

No expiry timer. No three-path logging beyond approve/reject.

Write a test with FakeBroker asserting that Reject leaves zero submitted orders and
still produces a decision record with order=None.

Acceptance: both paths work end to end against the paper account.
```

**GB-38 (3 SP) — this is your demo safety net:**

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-38.

Implement replay.py: replay a recorded trading day from stored bars and decision
records at configurable speed, FULLY OFFLINE. The dashboard must behave exactly as it
does live, including explanations.

This is the demonstration safety net. The live demo must not depend on US market hours
or on connectivity.

Add scripts/record_session.py to capture a live session's bars and decisions into a
replayable bundle.

Write a test that runs a replay with all network calls monkeypatched to raise, and
asserts the expected number of cycles completed with attributions present.

Acceptance: a recorded day replays end to end with the network disabled. Record a real
session this week so you have material to replay.
```

**GB-39 (2 SP):**

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-39.

Add fault handling, scoped tightly:
- Retry with exponential backoff on all broker and data calls (3 attempts)
- On connectivity loss, log loudly, skip the cycle, and continue rather than crashing
- Restart safety: an interrupted cycle leaves no orphaned or duplicated order

Do not build a circuit breaker or a health-check endpoint. Out of scope.

Test: simulated API failure recovers without crashing; a kill during order submission
leaves no duplicate order after restart, verified against the paper account.

Acceptance: both verified. Report what happened when you actually killed it mid-order.
```

**GB-40 — the gate:**

```
Read CLAUDE.md, docs/GLASSBOX_PROJECT_SPEC.md §8 and SOLO_BUILD_PLAN.md §6.

Walk the GATE 2 checklist and give me an honest pass/fail on each with evidence:
- Live loop runs a full session against Alpaca paper without manual intervention
- At least one order placed, filled and reconciled
- Every decision carries an exact attribution visible in the dashboard
- Replay mode reproduces a recorded day offline
- Co-Pilot approve/reject works

Then run a full demo rehearsal in Replay mode and time it.

Record the outcome in PROGRESS.md. Then state clearly, in one line, whether FITS
proceeds in Sprint 4 or is cancelled per the spec §8 rule. Record that decision in
DECISIONS.md either way.

Do not soften the assessment. A cancelled FITS with a working system is a good
outcome; a half-built FITS on top of a broken system is not.
```

---

## GATE 2 — 25 September

**If green:** Sprint 4 begins. You have a working product; everything from here is
upside.

**If red: FITS is cancelled, not postponed.** Sprint 4 becomes hardening plus the
report, and you submit a working v1 with a DLinear-versus-persistence comparison.
The proposal declares this as an acceptable outcome in advance, so nobody is surprised
and nothing is a failure. Record the decision and move on without regret.
