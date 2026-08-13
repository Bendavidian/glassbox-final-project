# PROGRESS

The running log of what is built. **Claude Code reads this file to locate the current phase.**
Append one line per completed task. Newest at the bottom of each sprint.

---

## Current state

**Sprint:** 1 — Foundations (15–28 Aug 2026)
**Next task:** GB-3
**Last gate passed:** none
**Blockers:** none

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
| GB-3 | | Ben | |
| GB-4 | | Ben | |
| GB-5 | | Noy | |
| GB-6 | | Noy | |
| GB-7 | | Noy | |
| GB-8 | | Ben | |
| GB-9 | | Ben | |
| GB-10 | | Ben | |
| GB-11 | | Ben | |
| GB-12 | | Noy | |

## Sprint 2 — Offline Vertical Slice · 29 Aug – 11 Sep 2026 → GATE 1

_not started_

## Sprint 3 — Live End-to-End · 12–25 Sep 2026 → GATE 2

_not started_

## Sprint 4 — FITS, Study, Report · 26 Sep – 10 Oct 2026 → GATE 3

_not started_

---

## Gate log

| Gate | Date | Result | Notes |
|---|---|---|---|
| GATE 1 | 11 Sep 2026 | pending | |
| GATE 2 | 25 Sep 2026 | pending | |
| GATE 3 | 10 Oct 2026 | pending | |

---

## Open questions

_Claude Code: write blocking questions here rather than guessing._
