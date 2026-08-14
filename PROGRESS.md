# PROGRESS

The running log of what is built. **Claude Code reads this file to locate the current phase.**
Append one line per completed task. Newest at the bottom of each sprint.

---

## Current state

**Sprint:** 1 — Foundations (15–28 Aug 2026)
**Next task:** GB-6
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
| GB-3 | 14 Aug 2026 | Ben | **Complete.** `contracts/schemas.py` — five frozen schemas from spec §4.2 with `__post_init__` shape validation; `contracts/protocols.py` — runtime-checkable `Forecaster`; import-linter layers + forbidden contracts in `pyproject.toml` with the harness above the live path; `tests/contracts/` proves both contracts have teeth. Suite at 108. |
| GB-4 | 14 Aug 2026 | Ben | **Complete.** `data/historical.py` — `load_history` fetches daily bars per symbol via yfinance, normalises to UTC/lowercase/sorted/deduplicated, adds unfilled `log_return`, and caches to `{cache_dir}/{symbol}.parquet`; a cache hit never touches the network. All five symbols cached 2016-01-04 → 2026-08-13, 2668 bars each. 16 offline tests on a committed CSV fixture; suite at 124. |
| GB-5 | 14 Aug 2026 | Ben | **Complete.** `data/quality.py` — `check_quality` reports missing NYSE trading days, per-column NaN counts, single-bar log returns above 0.25, duplicates and span; `report_all` prints a table and writes `data_cache/quality_report.json` (committed as provenance). Real universe: 0 missing bars, 0 duplicates, 1 NaN each (the unfilled first return), 1 large move total — NVDA 2016-11-11, +29.8%. 12 tests; suite at 136. |
| GB-6 | 14 Aug 2026 | Ben | **PENDING — awaiting real keys.** Code complete: credentials load only via `alpaca_credentials()`, `require_paper_endpoint()` refuses any non-paper endpoint (verified, exit 2), `.env.example` committed, `scripts/smoke_alpaca.py` written, git history scan clean. `.env` currently holds the placeholder values, so the account-equity print is unverified. 9 new tests; suite at 145. |
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

_The 14 Aug CI failure is resolved — see DECISIONS.md, "Python floor raised to 3.12";
green on 176fe01._

**14 Aug 2026 — GB-6 blocked on real Alpaca keys.** `.env` exists but contains the
placeholder values copied from `.env.example`, so `scripts/smoke_alpaca.py` reaches
Alpaca and receives `unauthorized`. Everything else in GB-6 is verified. Paste the paper
account's key and secret into `.env` and re-run `python scripts/smoke_alpaca.py`; it
should print account status, equity and buying power.
