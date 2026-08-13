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
- [ ] Committed with the task ID as the message prefix: `GB-42: fix irFFT amplitude scaling`

## 3. Non-negotiable engineering rules

**Causality.** Every value computed at time `t` uses only bars `<= t`. This is the most important correctness property in the project. Never use a centred rolling window. Never fit a scaler on data that includes the test period. Never compute a threshold using future information.

**Purity.** Every feature function is pure: DataFrame in, DataFrame out. No hidden state, no I/O, no global mutation.

**The keystone.** `features/builder.py` is the only place in the codebase where a model input window is assembled. Both offline training and the live loop call it with the same config. Do not create a second path.

**Exactness.** Attribution is algebra, not approximation. `sum(per_channel.values()) == forecast` within `1e-5`. Never import SHAP, LIME, captum, or any perturbation-based attribution library.

**Reporting.** Every result is a delta against the persistence baseline. MSE on prices is banned as a headline metric. A backtest Sharpe above 2.0 is a leakage alarm, not a success — stop and audit.

## 4. Conventions

- Python 3.11+, type hints everywhere, `ruff` + `black` defaults
- Tests in `tests/`, mirroring the package structure
- One module per commit where possible; small commits
- Docstrings state the contract, not the implementation
- Prefer standard library and the already-declared dependencies. Adding a dependency requires a line in `DECISIONS.md`.

## 5. When you are stuck

Say so. Do not guess at a contract, invent a config key, or stub a function and move on. Write the question in `PROGRESS.md` under `## Open questions` and stop.

## 6. Team

The Jira 70/30 split with Noy — and the `Own` column in spec §9 — is a **formal academic record only**; `SOLO_BUILD_PLAN.md` is the real build order and one developer writes every line, so ignore task ownership entirely and never break a frozen contract to unblock yourself.
