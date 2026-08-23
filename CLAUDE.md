# CLAUDE.md — Operating Instructions for Claude Code

> You are working on **GlassBox Trader**, a final-year B.Sc. project at Bar-Ilan University.
> Read this file completely before your first action in any session.

## 0. Orientation — do this first, every session

1. Read `docs/GLASSBOX_PROJECT_SPEC.md`. It is the **single source of truth**.
2. Run `git log --oneline -15` and read `PROGRESS.md` to find the current task.
3. Confirm which `GB-NN` task you are working on. State it before writing code.
4. If the task is ambiguous or seems to conflict with the spec, **stop and ask**.

## 1. The six rules

| # | Rule |
|---|---|
| 1 | **Never violate a contract.** §4 of the spec defines frozen interfaces. If a task appears to require changing one, stop and ask. Do not silently adapt. |
| 2 | **Never bypass a gate.** §8 defines three hard gates. Work belonging to a later phase does not begin until the preceding gate is green. |
| 3 | **Tests are the definition of done.** Every task lists its acceptance test. A task without a passing test is not complete. |
| 4 | **No feature invention.** If an idea is not in the spec, it goes into `IDEAS_PARKED.md`, not the codebase. |
| 5 | **Config over constants.** No magic numbers in modules. Everything comes from `config/settings.yaml` via `config/loader.py`. |
| 6 | **Simple over clever.** This codebase is demonstrated live to a supervisor. |

## 2. Definition of done for every task

A task is complete only when **all** of these hold:

- [ ] The acceptance test named in the spec exists and passes
- [ ] `pytest` passes in full — not just the new test
- [ ] `ruff check .` and `black --check .` pass
- [ ] The code contains no hardcoded config values
- [ ] `PROGRESS.md` is updated with the task ID, the date, and one line on what was built
- [ ] Staged by **named path**, and `git diff --cached` read in full before committing
- [ ] Committed with the task ID as the message prefix: `GB-42: fix irFFT amplitude scaling`

## 3. Non-negotiable engineering rules

**Causality.** Every value computed at time `t` uses only bars `<= t`. This is the most important correctness property in the project. Never use a centred rolling window. Never fit a scaler on data that includes the test period. Never compute a threshold using future information.

**Purity.** Every feature function is pure: DataFrame in, DataFrame out. No hidden state, no I/O, no global mutation.

**The keystone.** `features/builder.py` is the only place in the codebase where a model input window is assembled. Both offline training and the live loop call it with the same config. Do not create a second path.

**Exactness.** Attribution is algebra, not approximation. `sum(per_channel.values()) == forecast` within `1e-5`. Never import SHAP, LIME, captum, or any perturbation-based attribution library.

**Staging.** Never `git add -A`, `git add .`, or `git commit -a`. Stage the paths you changed, by name, and read `git diff --cached` in full before every commit. On 14 Aug 2026 a blanket `git add -A` swept a real Alpaca key into a commit from a file nobody had touched in the task. Blanket staging commits work you did not write and have not read. `tests/test_no_secrets.py` is the backstop, not the practice.

**Reporting.** Every result is a delta against the persistence baseline. MSE on prices is banned as a headline metric, and **MAE never appears in a table without `flatness` in the adjacent column** (§7.3 — across arms MAE tracks flatness at Spearman +0.81 and direction at +0.01). A backtest Sharpe above 2.0 is a leakage alarm, not a success — stop and audit.

**ONLY SOMETHING THAT RUNS IS A MECHANISM.** This is one rule with several instances, not several rules that happen to rhyme, and every instance below has cost this project real time. A second copy of a fact, a comment, a docstring, a `PROGRESS.md` sentence, a test that is never executed — each is a **record of an intention**, and none of them is enforcement. When you find yourself writing down something that must stay true, ask what would **fail** if it stopped being true, and if the answer is nothing, you have written a note. **Instance 1 — a fact stored in two places will diverge unless a test makes them equal.** Not a convention, not a reviewer: a **test**. Five instances in this project, and **three of the first four were caught by a test written for something else** — `SPEC_MODULES` against the package tree; `VALID_MODELS` against `ALL_FORECASTERS` against `smoke_offline`'s argparse literal; exit reasons defined in two modules; `pyproject.toml` against `requirements.lock`. The shape is always the same: each copy is internally consistent, so every check passes, and the disagreement is invisible until something downstream reads the wrong one. Derive the second copy from the first, or write the test that pins them together, in the same commit. **Instance 2 — a mechanism that can be skipped by a flag is a mechanism only when the flag is off.** Any test gated by a cache, a marker or an environment must **run in CI** even if it runs nowhere else. On 20 Aug 2026 the test that would have caught a defect in `results.csv` was written *before* the 40-minute run that produced the file and had never executed, because it was cache-gated and deselected. A test written and not run is a note. The data snapshot is committed for exactly this reason. **Instance 3 — a description of what the code should do is not a mechanism, whether it sits in a comment, a docstring or a plan.** On 20 Aug 2026 `test_an_unknown_model_is_refused_by_the_parser` carried the comment *"GB-41 adds it, not GB-24"*; GB-41 did not, and `smoke_offline` could not select the model the entire study is about while `prepare_live` could. Had GB-49 run first, the FITS arm would have been DLinear and the report would have compared a model to itself. On 23 Aug 2026 `test_the_same_grid_twice_gives_the_same_numbers` said *"one condition and one fold"* in its docstring and ran the **whole grid twice** in its body — 26 condition-grids per suite run once GB-50 widened the grid, invisible until the cost grew. If you find a gap you are not closing now, leave a **failing** guard — an `xfail(strict=True)`, an assertion on the property, a refusal in the code — or close it. **Never pipe a command whose exit code matters — capture the status first, filter after.** On 23 Aug 2026 `pytest -q | tail -4` reported a suite as passing that had one failure; the pipe returned `tail`'s status and the harness said exit 0. **This is the worst instance of the family, because the mechanism that would have caught it — CI running `pytest` directly — existed and was bypassed by a local shortcut.** A convenience wrapper less trustworthy than the slow path is a trap. Read the output, never the status, when the two can disagree. **And the practice that follows from all of it: before starting any job longer than ten minutes, run the tests that validate its output.** Two grid runs were discarded on 20 Aug 2026 — about an hour — and both were killed by tests that were **already written** when the run started and had simply not been executed. A long job is the worst place to discover that its output is wrong, because the cost of finding out is the job.

**Suspicious stability calls for a control, not a larger sweep.** On financial data, a coefficient of variation near **1%** across independently trained models is evidence the measurement is about the machine rather than about the market. The FITS phase advance came in at **+1.9582 days with sd 0.0203 — a CV of 1.04% — across 48 models at three fold-grid anchors**, and it survived being trained on white noise. Four years of equity returns do not produce agreement that tight; a deterministic operator does. **Two tests, and neither subsumes the other:** grid sensitivity asks *does it survive different fold boundaries* and killed `r = −0.47`; a null control asks *does it survive destroying the signal* and killed the phase advance, which had passed grid sensitivity at 48 of 48 cells.

## 4. Conventions

- Python 3.12+, type hints everywhere, `ruff` + `black` defaults
- Tests in `tests/`, mirroring the package structure
- One module per commit where possible; small commits
- Docstrings state the contract, not the implementation
- Prefer standard library and the already-declared dependencies. Adding a dependency requires a line in `DECISIONS.md`.

## 5. When you are stuck

Say so. Do not guess at a contract, invent a config key, or stub a function and move on. Write the question in `PROGRESS.md` under `## Open questions` and stop.

## 6. Team

The Jira 70/30 split with Noy — and the `Own` column in spec §9 — is a **formal academic record only**; `SOLO_BUILD_PLAN.md` is the real build order and one developer writes every line, so ignore task ownership entirely and never break a frozen contract to unblock yourself.
