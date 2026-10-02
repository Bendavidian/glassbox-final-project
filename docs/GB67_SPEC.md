> **Note, 2026-10-02.** Committed after the task closed; everything below the rule is the
> brief exactly as it was given, unedited and unrenumbered. Sections 3 and 9 assert that the
> mechanism was correct throughout and that only visibility was missing. That was false.
> Building against this brief found four further defects: the Approve and Reject fills had
> never rendered on the page; the sizing price was labelled as a price, though it was
> neither a fill nor a close; no run of the page ever finished, so an answered decision
> stayed on screen and invited a second answer; and the co-pilot exit path never enqueues
> at all (`glassbox/live_loop.py:1185` is the only `save_pending` call), which is still
> open. What was built, and against which commits, is the **GB-67 CLOSED** row in
> `PROGRESS.md`.

---

# Task spec — GB-67: pending approvals must be impossible to miss

**Handoff, 28 September 2026.** Paste this whole document into a new conversation. Your job
there is to turn it into Claude Code prompts, one at a time, and review what comes back.

---

## 1. The problem, measured

The deployed mode is `co_pilot`: exits, protection and reconciliation run autonomously, and
**every new entry and every exit waits for a human to approve it in the console.**

On 28 September the loop ran 271 cycles and recommended the same exit **271 times**:

```
[live-0271] co_pilot: recommending EXIT 26.984607719 UNH; not submitted
```

The operator saw none of them. Across four live sessions since 22 September, **not one exit
has ever been approved**, and `trades emitted` has read `0` in every session summary since the
project began, because a trade is only counted when a round trip closes.

The recommendations are not lost. They are written to `pending.json` and rendered in a panel.
**The panel is below the fold on a long page, and it is silent.** Nothing about the page
changes when the queue goes from empty to non-empty.

**That is the whole defect: a decision that requires a human is presented as passively as one
that does not.**

---

## 2. Scope — read this before anything else

This task changes **how a pending approval is presented, and nothing else.**

### In scope

- Where and how a pending approval appears on screen
- Whether the operator can miss it
- The wording and layout of the approval card
- Guards that assert the above

### Out of scope, and each of these is a hard stop

| Not in scope | Why it matters |
|---|---|
| **Decision logic** — the band, the step-majority gate, ranking, sizing | A presentation task that changes what the system decides is no longer a presentation task |
| **The executor, or anything that submits to the broker** | Approving must reach the broker by exactly the path it reaches it today |
| **`records.py`'s pending queue format**, or what `answer_pending` does | The queue is already correct; only its visibility is wrong |
| **The live loop**, in any file | It runs unattended and is the only thing that can lose money |
| **Auto-approval, timeouts, or defaults** | Co-Pilot exists because entering adds risk. A modal that can be dismissed into an approval is worse than a panel that is missed |
| **Sound, desktop notifications, browser push** | Streamlit cannot do these reliably, and an examiner's machine may block them |
| **Anything under `glassbox/` other than `dashboard/`** | If a change is needed outside the dashboard, stop and report it rather than making it |

**If the work cannot be done inside `glassbox/dashboard/` and `tests/dashboard/`, stop and say
so.** That boundary is the safety property of this task.

---

## 3. What the system already does, so nothing is rebuilt

Verify each of these against the code before relying on it; they come from session reports
rather than from a direct read.

- `records.py` writes the queue to `checkpoints/live/pending.json` and reads it back.
- `answer_pending` is the GB-37 approval path. It runs **outside** `run_cycle`, produces no
  `CycleReport`, and is therefore invisible to the session summary — a known, recorded gap.
- The dashboard has a Co-Pilot panel that renders each pending item with its narration and an
  Approve and a Reject control. It **returns early when the queue is empty**, so an empty
  queue renders nothing at all.
- The page refreshes on the dashboard's own cadence. A new pending item appears on the next
  refresh, not instantly.
- Every panel carries a source pill naming its artefact. A new surface must carry one too.

**The approval mechanism works.** An operator who scrolls to the panel and clicks Approve gets
an order at the broker. This task does not touch that path.

---

## 4. What to build

### 4.1 The modal

When the pending queue is non-empty, the console opens a **modal dialog over the page**, on
load and whenever the queue transitions from empty to non-empty.

Streamlit 1.61 supports `@st.dialog`. Confirm the API in the installed version before writing
against it rather than assuming the decorator's name or signature.

**Each pending item in the modal shows:**

| Element | Content |
|---|---|
| Action | ENTER LONG or EXIT, unmissable, in the status colour for that direction |
| Instrument | symbol, quantity, price, notional |
| Protection | stop and target for an entry; for an exit, what is being closed and its unrealised result |
| Why | the narration already generated for that decision, verbatim — do not re-word it |
| Provenance | the decision id and its bar date |
| Controls | **Approve** and **Reject**, equally weighted |

**Rules on the controls, all binding:**

- **Neither is a default.** No pre-selected button, no Enter-key shortcut, no focus that makes
  one a single keystroke away.
- **Dismissing the modal is not an answer.** Closing it leaves the item pending, and the modal
  reopens on the next load. It may not be permanently dismissed.
- **A rejection is recorded**, exactly as it is today. The console already states this: *"a
  rejection is recorded too: the log is what happened, not what was wanted."*
- **Nothing is approved in bulk.** One decision, one answer.

### 4.2 The persistent indicator

A modal can be closed. Once it is, the page must still show that the queue is non-empty:

- A count in the status strip at the top of the page, visible without scrolling, reading the
  same verdict source the other pills read.
- It must be **absent, not zero**, when the queue is empty — the page should not carry a
  permanently-on element, which is a defect this project has recorded.

### 4.3 What the modal must not do

- It must not animate, pulse or sweep. The GB-63 brief forbids motion that makes a page look
  alive, and the reason is recorded: a dead loop once rendered a green LIVE badge.
- It must not appear when the queue is empty.
- It must not compute anything. The dashboard reads files and computes nothing; the modal
  reads the same queue the panel reads.

---

## 5. Design constraints

The console has an enforced visual language. A new surface obeys it or the guards fail.

| Rule | Detail |
|---|---|
| Three colour roles | orange is chrome and never a value; the blue ramp encodes data and appears only in the attribution and spectral panels; green and red are gain and loss only, and never inside a data-encoding chart |
| Type scale | four steps, 12 / 14 / 16 / 30 px. Nothing below 12 |
| Tokens | every colour and size comes from `dashboard/tokens.py`. A literal at a call site fails an existing guard |
| Contrast | every text colour clears its threshold against the ground, and the existing test asserts it |
| No motion | see 4.3 |

**ENTER LONG and EXIT may use the status colours** — they are directions, which is exactly what
green and red encode. The modal's frame and labels are chrome.

---

## 6. Guards required

Each must be **verified by breaking it**: make the change the guard forbids, confirm the test
fails, restore, confirm it passes. Report caught/tried for every one.

1. **The modal opens when the queue is non-empty** and does not when it is empty.
2. **Closing the modal does not answer the item.** After a dismissal the queue is unchanged.
3. **Neither control is a default.** No pre-selected button and no single-keystroke approval.
4. **The persistent indicator is absent on an empty queue** and shows the true count otherwise.
5. **Every element in the modal comes from the pending record**, not from a recomputation. A
   narration in the modal is the narration in the record.
6. **The modal obeys the colour guard** — it is enumerated by the existing chart or surface
   guard rather than exempted from it.
7. **The approval path is unchanged.** A test that asserts the modal's Approve calls the same
   function the panel's Approve calls today.

Guard 7 is the one that protects the rest of the system, and it is the one to write first.

---

## 7. Working rules for every prompt

These are the project's standing rules. Include the relevant ones verbatim in each prompt.

1. **One session writes `glassbox/dashboard/app.py` at a time.** Two sessions editing it
   clobber each other and git cannot see the seam. The index is shared: a file staged by one
   session is swept into another's bare `git commit`.
2. **Report before editing.** For each item, the session reports what exists and where, then
   changes it.
3. **Verify by breaking.** Every guard is proved by breaking the thing it guards.
4. **Bare `pytest`, no path, no `-k`, no `-m`, no pipe.** Report the summary line, the exit
   code and the collected count. `python -m pytest` is the weaker invocation and does not
   license the word green.
5. **Commit by named path.** Never `git add -A`. For a shared file such as `PROGRESS.md`,
   stage only your own lines with a filtered blob built against the current HEAD.
6. **A live loop may be running.** It writes to `logs/` and `checkpoints/` every 60 seconds.
   Do not stop it, do not commit its output, and confirm both directories are gitignored
   before staging.
7. **No relative time words in durable records.** Write the date.
8. **Push only when told**, then report the pushed range and CI's conclusion.

---

## 8. Suggested sequence

One session throughout, since every item touches the same file.

| Step | What | Gate before moving on |
|---|---|---|
| 0 | **Report only.** Where the Co-Pilot panel lives, what it renders, what `answer_pending` does, and whether `st.dialog` is available in the installed Streamlit. No edits | Ben reads the report |
| 1 | **Guard 7 first** — a test asserting the approval path is what it is today. It must pass before anything changes, and still pass after | caught/tried |
| 2 | The modal, with its content and controls | full suite green |
| 3 | The persistent indicator | full suite green |
| 4 | Guards 1 to 6 | caught/tried for each |
| 5 | Full suite, commit by named path, push | CI green |

**Step 0 may produce a reason not to proceed** — for instance if `st.dialog` cannot coexist
with the page's refresh cadence. If it does, that is a result, not a failure. Report it and
stop rather than working around it.

---

## 9. How this is recorded

This is a live-system usability defect found by operating the system, and it belongs in the
record as one:

> The console presented a decision requiring a human exactly as passively as one that did not.
> Across four sessions the same exit was recommended several hundred times and never seen. The
> mechanism was correct throughout — the queue was written, the panel rendered it, and an
> approval would have reached the broker. **What was missing was any reason for the operator to
> look.**

That belongs in `PROGRESS.md`, and it is worth a line in the report's usability discussion:
an interface that is correct and unnoticed is not a working interface.

---

## 10. What success looks like

An examiner who opens the console during a live session, having never seen it before, **cannot
proceed without noticing that the system is asking them something**, and can tell from the
screen alone what is being asked, why, and what each answer does.

Nothing else about the system changes.
