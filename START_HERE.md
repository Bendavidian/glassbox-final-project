# START HERE — What every file is, and what to do with it

Read this once, do §2, then never think about file organisation again.

---

## 1. The file map

Nine files. Three audiences. Here is exactly where each one goes and when you open it.

### For the university — you submit these, Claude Code never reads them

| File | What it is | What you do with it |
|---|---|---|
| `GlassBox_Trader_Specification_v3.html` | The Hebrew proposal, 15 sections, five diagrams | Open in Chrome → Print → Save as PDF → send to your supervisor |
| `GlassBox_Trader_Architecture_Report.docx` | The architecture report in the course's required structure | Fill in Noy's surname, the supervisor and the project code. Replace the four screenshot placeholders after Sprint 3. Submit. |
| `glassbox_jira_tasks.csv` | 12 epics + 60 tasks with dates and owners | Jira → Import issues from CSV. **Keep the 70/30 split with Noy** — this is the formal record. |
| `GlassBox_Vision_Video_Script.md` | The 52-second video script, shot by shot | Send to the friend making the video. He needs nothing else. |

### For the repository — Claude Code reads these on every session

| File | Where it goes | Why it matters |
|---|---|---|
| `CLAUDE.md` | repo root | **Read automatically at the start of every session.** The six rules, the definition of done, the causality and exactness requirements. This is what stops Claude from inventing. |
| `GLASSBOX_PROJECT_SPEC.md` | `docs/` | The frozen contracts, the schemas, the FITS pipeline, the gates. Every prompt tells Claude to read it. |
| `PROGRESS.md` | repo root | **The project's memory between sessions.** Update it after every task or the next session will not know where you are. |
| `DECISIONS.md` | repo root | Every architectural decision, dated. Add an entry whenever you change a contract, add a dependency, or cut scope. |
| `IDEAS_PARKED.md` | repo root | Where new ideas go instead of into the codebase. |
| `REFERENCE_AUDIT.md` | `reference/` | What to reuse from your existing code and what is broken in it. See §3. |

### For you — the build order

| File | When you open it |
|---|---|
| `SOLO_BUILD_PLAN.md` | Day one of each sprint. The day-by-day schedule, the cut list, the gates. |
| `prompts/SPRINT_1.md` | 15–28 Aug — one prompt per day, copy-paste |
| `prompts/SPRINT_2.md` | 29 Aug – 11 Sep |
| `prompts/SPRINT_3.md` | 12–25 Sep |
| `prompts/SPRINT_4.md` | 26 Sep – 10 Oct |

---

## 2. Setting up the repository

```bash
mkdir glassbox-trader && cd glassbox-trader
git init
mkdir -p docs prompts reference

# repo root — Claude Code reads these
mv ~/Downloads/CLAUDE.md .
mv ~/Downloads/PROGRESS.md .
mv ~/Downloads/DECISIONS.md .
mv ~/Downloads/IDEAS_PARKED.md .
mv ~/Downloads/SOLO_BUILD_PLAN.md .

# the contract
mv ~/Downloads/GLASSBOX_PROJECT_SPEC.md docs/

# the prompts
mv ~/Downloads/SPRINT_*.md prompts/

# your existing code, as read-only reference
unzip ~/Downloads/algotrading_project-main.zip -d reference/
mv ~/Downloads/REFERENCE_AUDIT.md reference/

git add .
git commit -m "GB-0: project charter, spec, build plan and reference code"
code .
```

Then install the Claude Code extension: `Cmd+Shift+X` (Mac) or `Ctrl+Shift+X`
(Windows/Linux), search "Claude Code", Install. First launch opens a browser to sign in.

**Verify the setup before writing any code.** Open a Claude Code panel and paste:

```
Read CLAUDE.md, SOLO_BUILD_PLAN.md and docs/GLASSBOX_PROJECT_SPEC.md.
Then read reference/REFERENCE_AUDIT.md and confirm you understand what may and may
not be reused from reference/algotrading_project-main/.

Summarise in ten lines: what this project is, what the six rules are, which task we
start with, and what is broken in the reference code. Do not write any code.
```

If that summary is right, your setup is right.

---

## 3. Your existing code — what it is worth

`reference/algotrading_project-main/` is an LTSF-Linear trading project. It is a
genuine head start on three modules and a genuine trap on two. I read all of it.
Here is the honest assessment.

### Reuse — this saves you real days

| File | Feeds | Notes |
|---|---|---|
| `models/DLinear.py` | **GB-13** | The decomposition and the two linear heads are correct and match the paper. Wrap it in the `Forecaster` protocol rather than rewriting it. Saves most of a day. |
| `models/NLinear.py`, `Linear.py` | — | Cut from scope, but keep them; free ablation material if you find slack. |
| `evaluation.py` | **GB-19** | `calc_annualized_sharpe`, `calc_max_drawdown`, `calc_sortino`, `calc_total_return` are all correct. Port them directly. |
| `backtesting.py` + `models_backtest.py` | **GB-18** | The event-loop shape — enter, check SL/TP, close, track balance — is a sound skeleton. The `Position` / `ActionType` / `PositionType` enums are clean. Reuse the structure. |
| `utils/tools.py` | **GB-15** | `EarlyStopping` and `adjust_learning_rate` are standard and fine. |
| `data_provider/data_loader.py` | GB-9 (reference only) | Its windowing logic is a useful reference. **Do not port it** — GB-9 replaces it, because the builder must serve both training and live. |

### Fix before reuse — a real bug

**`strategies.py`, all four stop-loss and take-profit comparisons are inverted.**

```python
# current — fires when price is ABOVE the stop, i.e. almost always
if position.type == PositionType.LONG and row['Low'] >= long_stop_loss_price:

# correct — a long stop fires when the low DROPS TO the stop
if position.type == PositionType.LONG and row['Low'] <= long_stop_loss_price:
```

The same inversion appears in the long take-profit (`High <=` should be `>=`), the
short stop-loss (`High <=` should be `>=`) and the short take-profit (`Low >=` should
be `<=`). All four.

This is worth understanding rather than just patching. It means every backtest that
repo has ever produced had stops and targets firing at the wrong moments — and it
would never crash, never warn, and never look obviously wrong. **This is precisely
why GB-18's acceptance criterion is a hand-computed three-trade scenario checked to
the cent.** A backtester that runs is not a backtester that is correct.

### Do not reuse

**`utils/metrics.py`.** It is unsalvageable:
- `SHARP()` returns a hardcoded `12`
- `SHARP2` and `SHARP5` are each defined twice; the second definition silently
  shadows the first
- One version contains `pred = true + 0.01` — it overwrites the model's prediction
  with the ground truth, then measures the error

Write `backtest/metrics.py` from scratch in GB-19, and port only from `evaluation.py`.

**The single train/validation/test split.** The reference splits once. Your project
requires walk-forward — GB-17. This is a methodology difference, not a code difference.

### One thing that looks like a bug and is not

`models/DLinear.py`'s `moving_avg` pads both ends of the series, which makes the
trend a *centred* moving average. That would normally be a look-ahead violation.

It is not one here, and the reason is worth being able to say out loud in your
defence: the decomposition operates **inside the input window**, and every bar in that
window is already in the past relative to the forecast. Centring within `[t-L+1, t]`
uses no information after `t`.

Where centring *would* be fatal is in `features/indicators.py` and
`features/wavelets.py`, which compute values along the whole series. That is exactly
why GB-8 forbids `center=True` and GB-10 tests for it.

### One thing that is done right

`data_loader.py` fits the scaler on the training slice only
(`self.scaler.fit(train_data.values)`) and then transforms everything. That is the
correct pattern, and it is the same rule GB-9 and GB-15 enforce. Whoever wrote that
knew what they were doing.

---

## 4. The daily loop

```
morning   fresh Claude Code session
          paste the session-start prompt from prompts/SPRINT_1.md
          confirm which GB-NN you are on

build     paste the task prompt for today
          Plan mode for anything over 2 SP — read the plan, push back, then build

verify    pytest · ruff check . · black --check .
          READ THE DIFF. Every line.

close     paste the task-close prompt
          update PROGRESS.md · commit with the GB-NN prefix
```

**One task per session.** Do not batch. A session running two hours has drifted from
`CLAUDE.md` and will start inventing.

**Read every diff.** You are the only reviewer on this project. A line you cannot
explain to your supervisor does not go in — and the inverted stop-loss above is what
happens when nobody reads.

**Update `PROGRESS.md` every time.** It is the memory. A session that cannot locate
the current phase will guess, and a guessing agent in a codebase with frozen contracts
is how contracts get quietly broken.

---

## 5. What happens on 15 August

Open `prompts/SPRINT_1.md`, scroll to **Day 1 · GB-1**, paste it.

That is the whole thing.
