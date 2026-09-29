# Session review — live run of 2026-09-24

**Method: read-only.** The closed session was not restarted, no code, config or state file
was modified, and nothing was committed. Evidence is `logs/live-2026-09-24.log` (16,596
lines), `checkpoints/live/book.json`, and three read-only Alpaca queries
(`get_all_positions`, `get_orders`) run against the paper endpoint after the close. Broker
facts stated as "now" are as of those queries.

## Verdict on the two lines

| # | Line | Verdict |
|---|------|---------|
| 1 | `positions at exit : 5 STILL HELD` vs `open orders at exit : 4` | **Not a protection failure.** JNJ carried a live stop for the whole session. The asymmetry is a sampling artifact of Alpaca's staggered expiry sweep. **But** the `missing_protection` detector is saturated and could not have told you otherwise. |
| 2 | `orders submitted : 287` | **The counter does not count order submissions.** All 287 are `pending_approval` recommendations that never left the process. The broker created 7 orders today, and the counter counts none of them. |

The most important fact found, which neither line states: **all five positions are
unprotected right now.**

---

## 1. JNJ

### Is JNJ a managed position, and does it carry a stop price

Yes to both. `checkpoints/live/book.json` (mtime 2026-09-24 23:00):

```json
"JNJ": {
  "symbol": "JNJ",
  "quantity": 35.929963733,
  "decision_id": "20260923-JNJ",
  "entry_price": 273.16,
  "stop_loss": 266.12435,
  "take_profit": 290.8163
}
```

It is in `managed`, not `unmanaged`, and it carries `stop_loss` 266.12435. The broker agrees
on the position: `JNJ qty=35.929963733 avg_entry=273.1600`.

JNJ was not held at the open. It was bought during the session at 14:34:29Z, appeared to
reconciliation as an `unknown_position`, and was adopted at cycle 17 because `20260923-JNJ`
was a decision in the log with the order it produced. This is the `positions adopted : 2`
line (JNJ and PG).

### Did reconcile ever report missing_protection for JNJ

Yes — **268 times**, on every cycle from adoption at cycle 17 to the last cycle at
19:59:32Z. It reported it for every other managed symbol too, on every cycle after arming:

| Symbol | `missing_protection` warnings | live-sell count reported |
|--------|------------------------------:|--------------------------|
| JNJ | 268 | `broker=1.000000000` on all 268 |
| PG | 279 | `broker=1.000000000` on all 279 |
| UNH | 285 | `broker=0.000000000` once (cycle 1), then `1.000000000` x 284 |
| V | 285 | `broker=0.000000000` once (cycle 1), then `1.000000000` x 284 |
| WMT | 285 | `broker=0.000000000` once (cycle 1), then `1.000000000` x 284 |

**1,402 warnings, and 1,399 of them are false.** The reason is in
`glassbox/engine/reconcile.py:275`:

```python
if live.get(symbol, 0) < 2
```

`_unprotected` counts live sell orders and raises `MISSING_PROTECTION` for any managed
position with fewer than **two**. The ruling of 26 Aug 2026 — recorded at length in
`executor.protect`'s own docstring — is that two protective orders **cannot exist at this
venue at any size**: a working sell holds the whole position, so the second standalone sell
is refused with `insufficient qty available`, and every multi-leg order class is refused on
a fractional quantity. The system therefore arms exactly one leg by design, and
`live_loop.py:168` says so:

```python
LEGS = (STOP_LEG,)
```

So the detector's threshold is 2 and the system's reality is 1. `broker=1` — the fully
protected state — is reported as a divergence, and `broker=0` — a genuinely naked position —
is reported identically. **The signal is saturated: it is on permanently, and it cannot
distinguish a protected position from an unprotected one.** A JNJ that really had been naked
all day would have produced exactly the log you already have.

The two thresholds are the same fact in two places, and the suite pins the stale one.
`tests/engine/test_reconcile.py:249` is named
`test_a_managed_position_without_two_live_legs_is_reported` and asserts that a stop-only
position — the only protected state Alpaca permits — **is** a divergence. Its sibling
`test_a_fully_protected_position_is_clean` builds its fixture from a `protection()` helper
(line 38) that returns two live sell orders and is docstringed "the two live sell orders a
protected position carries". That state is one the broker refuses. Both tests pass, both are
internally consistent, and nothing compares `reconcile`'s threshold to `LEGS`:
`grep -rn "LEGS" tests/` returns nothing.

This is CLAUDE.md Instance 1 with the second copy in `tests/`, and it is invisible by
construction because `reconcile` is detection-only — nothing acts on the divergence, so
nothing ever contradicts it.

### Did step 4 ever attempt to arm it, succeed, or fail

Once, at cycle 17, and it **succeeded**:

```
14:34:48Z ERROR  RISK EVENT: JNJ is a managed position missing its stop. Arming now (rule 3)
14:34:48Z INFO   [live-0017] JNJ: stop re-armed id=52bf234a-b29e-4896-bca3-20b4a0432c69 at 266.1243
```

`protect_book` uses `LEGS = (STOP_LEG,)`, so from cycle 18 onward it saw the stop present and
correctly did nothing. It never attempted JNJ again, never failed, and therefore never
recorded a strike — which is why `positions flattened : 0 in-cycle (rule 3)` is correct
rather than suspicious. Rule 3 flattens after two *consecutive arming failures*; there were
zero arming failures all day. All five arms in the session succeeded first time:

| Cycle | Time | Symbol | Stop order id | Stop |
|------:|------|--------|---------------|-----:|
| 1 | 14:15:27Z | UNH | `9e23810e` | 360.1513 |
| 1 | 14:15:27Z | V | `2d428e9d` | 353.0072 |
| 1 | 14:15:27Z | WMT | `60fd5c62` | 105.3905 |
| 6 | 14:21:33Z | PG | `903baf2c` | 143.9868 |
| 17 | 14:34:48Z | JNJ | `52bf234a` | 266.1243 |

That is the `positions re-armed : 5` line, and it is the only line in the summary that
describes real orders.

### Is there a live protective order for JNJ at Alpaca right now

**No — and there is none for any of the five.**

```
POSITIONS NOW   : JNJ, PG, UNH, V, WMT  (5)
OPEN ORDERS NOW : 0
```

All seven of today's orders are in a terminal state. The five stops all expired at the close,
as DAY orders must:

| Symbol | Stop submitted | Status | Expired at |
|--------|----------------|--------|------------|
| JNJ | 14:34:48.896Z | expired | **20:00:16.190Z** |
| PG | 14:21:33.427Z | expired | 20:01:23.297Z |
| V | 14:15:27.704Z | expired | 20:01:34.688Z |
| WMT | 14:15:27.842Z | expired | 20:02:10.093Z |
| UNH | 14:15:27.559Z | expired | 20:02:24.890Z |

None filled (`filled_qty=0` on all five), which is also why `trades emitted : 0` is correct.

### Was JNJ held unprotected without reconcile noticing?

**No.** Stated plainly, because the question deserves a plain answer: JNJ carried a live stop
at 266.12 continuously from 14:34:48Z until the broker expired it at 20:00:16Z. It was not
naked during the session.

The summary asymmetry has a mundane cause. `_open_orders` (`live_loop.py:2402`) samples the
broker once, in the shutdown path, and it ran at **20:00:43Z** — twenty-seven seconds after
JNJ's stop expired and forty seconds before PG's. Alpaca's expiry sweep is staggered across
2m 08s, and the report was taken inside it. At that instant JNJ's stop was `expired` and the
other four were still `new`, which is exactly what the line printed. Sample at 20:02:30Z and
it would have read `open orders at exit : 0`.

So the line is a true statement about 20:00:43Z and a misleading statement about the session.
It is misleading in the dangerous direction: it reads as "four protected, one not", when the
truth one hundred and seven seconds later was "none protected". The `SessionReport` docstring
for `held_at_exit` already warns that `open orders at exit : 0` beside a real position is the
worst state the loop can end in and that only the pair is an answer; here the pair is
*correct at the sampling instant* and still conveys the wrong picture, because the quantity
is racing while it is read.

Two genuine but small unprotected windows did occur today, and neither is JNJ's stop
disappearing — both are the gap between an approved entry filling and the next cycle arming
it:

| Symbol | Entry filled | Stop armed | Naked for |
|--------|--------------|------------|----------:|
| PG | 14:20:26.771Z | 14:21:33.427Z | 66.7 s |
| JNJ | 14:34:31.917Z | 14:34:48.896Z | 17.0 s |

This is `executor.protect` behaving as documented: it is called by `_submit` immediately
after the market buy is accepted, sees `filled_quantity == 0`, declines to arm a short sale,
and leaves protection to "the cycle that sees the fill". The gap is inherent to that design,
not a malfunction — but it is unmeasured and unreported anywhere in the summary.

Finally, the overnight state now is the documented one, not a new defect: the banner at
14:15:26Z announced that protective legs are `TimeInForce.DAY`, that the broker expires them
at the close, and that rule 2 re-arms every open position before any entry at the next
session's first cycle. Five positions will be unprotected from now until that first cycle —
against three this morning.

---

## 2. What `orders submitted : 287` counts

### The counter

`live_loop.py:295`:

```python
f"  orders submitted     : {sum(len(c.submissions) for c in self.cycles)}",
```

It counts `Submission` objects, and a `Submission` is the outcome of an *execution attempt*,
not a sent order. `executor.Submission` (line 156) carries a `status` field and even exposes

```python
@property
def submitted(self) -> bool:
    return self.status == SUBMITTED
```

— which the summary does not consult. `pending_approval`, `rejected` and `declined` outcomes
are all counted as submitted orders.

### Which call sites increment it

Four, all writing the per-cycle `submissions` list:

| Site | Source | Contributed today |
|------|--------|------------------:|
| `live_loop.py:1205` | `execute(...)` — the entry for a decided bar | **2** |
| `live_loop.py:1207` | `_send_target_exits(...)` — take-profit reached on a completed bar | 0 |
| `live_loop.py:1208` | `_send_exits(...)` — signal exits owed | **285** |
| `live_loop.py:1025` | `_send_exits(...)` on the all-symbols-stale early return | 0 (never stale) |

### The arithmetic: 285 + 2 = 287

**285 — one phantom UNH exit per cycle.** UNH's 2026-09-23 bar decided `exit` (confirmed in
`checkpoints/live/decisions/2026-09.jsonl`), recorded at cycle 1. In `co_pilot` mode
`exit_position` (`live_loop.py:1967`) logs the recommendation and returns a `Submission` with
`status="pending_approval"` **without touching the broker**:

```python
if state.cfg.live.mode == CO_PILOT:
    log(f"co_pilot: recommending EXIT {holding.quantity:.9f} {symbol}; not submitted")
    return Submission(..., status="pending_approval", ...)
```

`_send_exits` deduplicates by asking the broker whether `{decision_id}-exit` is already in
its order history. Nothing is ever sent, so that id can never appear, so the guard never
fires and the exit is re-offered on every poll for the rest of the day. The log carries
exactly 285 `co_pilot: recommending EXIT ... UNH; not submitted` lines, one per cycle, from
`[live-0001]` at 14:15:38Z to `[live-0285]` at 19:59:43Z — all for UNH, none for any other
symbol.

The comment at `_send_exits` explains re-sending as an obligation on committed capital that a
failed submission must retry. That reasoning is sound for `auto` mode. Under `co_pilot` the
same code path turns a single unapproved recommendation into 285 counted "submissions", and
the counter makes a queue of one look like sustained activity.

**2 — the entry recommendations.** JNJ and PG, at cycle 2:

```
14:16:52Z [20260923-JNJ] co_pilot: recommending BUY 35.929963733 JNJ, ... ; not submitted
14:16:52Z [20260923-PG]  co_pilot: recommending BUY 67.474811371 PG,  ... ; not submitted
14:16:52Z [live-0002] JNJ: queued for approval as 20260923-JNJ
14:16:53Z [live-0002] PG: queued for approval as 20260923-PG
```

Also `pending_approval`, also counted. There are exactly two `recommending BUY` lines and two
`queued for approval` lines in the file.

**So: zero of the 287 reached the broker.** The counter counts something that is not an order
submission, and the name for it is *recommendations produced* — 287 `Submission` objects, all
`pending_approval`, 285 of them re-offers of one unapproved exit.

### What the broker actually created today

Seven orders, from a `status=ALL, after=2026-09-24T00:00Z` query:

```
by kind   : {'stop': 5, 'market': 2}
by side   : {'sell': 5, 'buy': 2}
by status : {'expired': 5, 'filled': 2}
by symbol : {'JNJ': 2, 'PG': 2, 'UNH': 1, 'V': 1, 'WMT': 1}
```

| Time | Symbol | Side | Kind | Status | Qty | client_order_id |
|------|--------|------|------|--------|----:|-----------------|
| 14:15:27Z | UNH | sell | stop | expired | 26.984607719 | `20260922-UNH#2-stop` |
| 14:15:27Z | V | sell | stop | expired | 27.530713745 | `20260922-V#2-stop` |
| 14:15:27Z | WMT | sell | stop | expired | 92.038821904 | `20260902-WMT#4-stop` |
| 14:20:21Z | PG | **buy** | market | **filled** | 67.474811371 | `20260923-PG` |
| 14:21:33Z | PG | sell | stop | expired | 67.474811371 | `20260923-PG#1-stop` |
| 14:34:29Z | JNJ | **buy** | market | **filled** | 35.929963733 | `20260923-JNJ` |
| 14:34:48Z | JNJ | sell | stop | expired | 35.929963733 | `20260923-JNJ#1-stop` |

**Neither group is in the 287.**

The five stops were created by `protect_book` -> `_arm_leg`, which returns its work in
`rearmed`, not `submissions`. They are reported as `positions re-armed : 5` and are absent
from the order-submission count entirely — even though they are the only orders the loop
itself submitted all day.

### What submitted the two market buys

**The dashboard's approval path, in a different process.** Both buys carry the bare
`decision_id` as their `client_order_id` — `20260923-JNJ`, `20260923-PG` — which is the
signature of `executor._submit` (line 463: `client_order_id=decision_id`). `_submit` has two
callers, `execute` in `auto` mode and `approve`. The session ran in `co_pilot`, so it was
`approve`, reached from `glassbox/dashboard/app.py:4037`:

```python
if approved:
    answer_pending(cfg, AlpacaBroker(), root, entry["decision_id"], True)
```

An operator pressed APPROVE on the two queued recommendations. Corroboration:

- `checkpoints/live/pending.json` is now `[]` — the queue was answered and cleared.
- `approve` logs `operator APPROVED ...` through `glassbox.engine.executor`, and that string
  appears **nowhere** in `logs/` — the Streamlit process does not write to the session log
  file. `logs/live-2026-09-24.log` is the only log file written today.
- The loop's own evidence is consistent with an outside actor: it saw the positions appear
  with no book entry, raised `unknown_position`, and adopted them (`positions adopted : 2`)
  rather than recognising orders it had placed.
- The stops on those two positions carry `#1-stop`, the loop's arming-attempt suffix, not
  `-stop`, which is what `executor.protect` would have written. Confirms `protect` returned
  empty because the entry had not filled yet, and the loop armed both a cycle later.

So the honest account of today's order flow is: **7 orders — 2 entries submitted by a human
through the dashboard, and 5 stops submitted by the loop's protection rule. The loop
submitted no entries and no exits, which is what `co_pilot` means.** The line
`orders submitted : 287` describes none of it.

---

## What is true right now

- **5 positions held: JNJ, PG, UNH, V, WMT.**
- **0 protective orders live.** All five stops expired between 20:00:16Z and 20:02:24Z.
- The UNH exit decided on the 2026-09-23 bar is still unapproved and still owed. It will be
  re-offered on every cycle of the next session until someone answers it.
- `pending.json` is empty, so nothing is queued for approval.
- Next session's cycle 1 will re-arm all five under rule 2, and the overnight-residual banner
  will report five unprotected positions rather than three.

## Not fixed

Per instruction, nothing was changed. The two defects worth a task each:

1. **`reconcile._unprotected` requires two live legs; the venue permits one.** The detector is
   permanently on for every managed position and cannot detect a naked position. Fixing it
   means deriving the threshold from `live_loop.LEGS` rather than restating it, retiring the
   `protection()` test helper that fabricates an impossible two-sell state, and adding the
   guard that fails when the two disagree. Today the protection question had to be answered
   from the broker because the detector built to answer it is saturated.
2. **`orders submitted` counts `Submission` objects of any status.** It counted 287 when 0
   were submitted, and omitted the 5 that were. `Submission.submitted` already exists and is
   unused; the re-offer loop in `_send_exits` under `co_pilot` is the multiplier.

A third, smaller: `_open_orders` samples once during a staggered expiry sweep, so
`open orders at exit` is a race. It cannot be made true by resampling — the honest form is to
state the sampling instant, or to report which managed positions have a live stop rather than
how many orders are working.

A fourth, noted only because it touches rule 5: `ORDER_HISTORY = 500` is a hardcoded constant
in `glassbox/engine/executor.py:86`. It did not bite today — 7 orders — but it silently bounds
every protection check in the system, and `get_orders` is "newest first".
