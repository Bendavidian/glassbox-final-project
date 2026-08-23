# GlassBox Trader

GlassBox Trader is an autonomous algorithmic trading system that trades five US equities
on an **Alpaca paper-trading account** — real market data, virtual money, never a live
account. Unlike a conventional trading bot it explains every decision it makes, in real
time, with attribution that is exact algebra rather than a post-hoc approximation: the
per-channel contributions sum to the forecast, and that identity is asserted in the test
suite. It is a final-year B.Sc. project at Bar-Ilan University, and its deliverable is the
system and its methodology, not a profit — a result showing that nothing beats a
persistence baseline is a valid and expected outcome.

- **Specification** — [`docs/GLASSBOX_PROJECT_SPEC.md`](docs/GLASSBOX_PROJECT_SPEC.md) is the single source of truth
- **Architecture** — [`ARCHITECTURE.md`](ARCHITECTURE.md)
- **Progress and decisions** — [`PROGRESS.md`](PROGRESS.md), [`DECISIONS.md`](DECISIONS.md)

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.12 or newer** | Not 3.11. The pinned `numpy` and `scipy` both require 3.12+, so the dependency set has never been installable on 3.11. Check with `python --version`. |
| **git** | Also used by the test suite, which scans tracked files for credential-shaped strings. |
| **An Alpaca paper-trading account** | Free, at [app.alpaca.markets](https://app.alpaca.markets). Only needed for the live path; the offline path and the full test suite run without one. |

No GPU is required. Every model in the system is linear by design.

---

## Install

### 1. Clone and create a virtual environment

```powershell
git clone https://github.com/Bendavidian/glassbox-final-project.git
cd glassbox-final-project
python -m venv .venv
```

### 2. Activate it — do not skip this

**Every command in this README assumes an activated virtual environment.** Running
`python` without activating it uses the system interpreter, which has none of this
project's dependencies, and you get `ModuleNotFoundError` — for `pytest`, `pandas`,
`yfinance`, whichever the command needed first. That error means "you forgot to activate",
not "the install is broken".

Note that `import glassbox` on its own may still succeed from the repository root even
without activation, because Python puts the working directory on the path and finds the
`glassbox/` folder directly. That is a coincidence of where you are standing, not a
working install — the first dependency the code reaches for will fail.

```powershell
# Windows — PowerShell
.\.venv\Scripts\Activate.ps1

# Windows — Git Bash
source .venv/Scripts/activate

# macOS / Linux
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. Confirm the right interpreter is in front:

```powershell
python -c "import sys; print(sys.executable)"
```

The path it prints must be inside `.venv`. If it is not, activation did not take.

> If PowerShell refuses the activation script with an execution-policy error, run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and try
> again. It applies to that terminal only.

### 3. Install from the lockfile

```powershell
.\scripts\setup.ps1
```

**One command, because a procedure a reader must execute correctly is a note and a script
is a mechanism.** It runs the three commands below in order, stops at the first failure,
and then **verifies** rather than assumes: that `import glassbox` resolves, that the
package is installed **editable** and resolving to this repository, and that every
heavyweight dependency imports. It also refuses Python below 3.12 with a message naming
the cause, which pip does not.

It exists because on 23 Aug 2026 the third command was skipped while running this
project's own reproducibility audit — a non-editable install still imports correctly from
the repository root, because Python puts the working directory on the path, and breaks
only when something runs from elsewhere. That is the silent half the verification closes.

<details>
<summary>What the script runs, for a reader on a different shell</summary>

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install -e ".[dev]" --no-deps
```

Then check `python -c "import glassbox, pathlib; print(pathlib.Path(glassbox.__file__).resolve().parent.parent)"`
prints this repository's path and not a `site-packages` directory.

</details>

**Install from `requirements.lock`, not from `pyproject.toml`.** The lockfile pins the
exact versions every reported result was produced with. `pyproject.toml` carries lower
bounds, which are correct for development and will not reproduce a number in the report.
CI installs the lockfile too, so a green CI run is evidence about the same dependency set
you have.

The `--no-deps` on the last line is deliberate: it installs this package without letting
pip re-resolve and quietly upgrade anything the lockfile just pinned.

### 4. Verify

```powershell
pytest
```

Everything should pass. This needs no credentials and no network.

---

## Credentials

Only needed for the live Alpaca path. Skip this section if you are running the offline
path or the tests.

1. Create a **paper** account at [app.alpaca.markets](https://app.alpaca.markets) and
   generate an API key under the Paper Trading tab.
2. Copy the template and fill in your own values:

   ```powershell
   copy .env.example .env      # PowerShell
   cp .env.example .env        # Git Bash / macOS / Linux
   ```

3. Edit `.env` and replace the two placeholder values.

**Rules, enforced rather than advised:**

- `.env` is git-ignored. `.env.example` is committed and must contain **placeholders
  only** — `tests/test_no_secrets.py` fails the suite if any tracked file contains a
  credential-shaped string. That test exists because a real key once reached a local
  commit; see `DECISIONS.md`.
- This project never trades real money. `require_paper_endpoint()` refuses any endpoint
  other than `https://paper-api.alpaca.markets`, and refuses before opening a connection.
- Nothing in the codebase prints a key, not even partially masked.

Check the connection:

```powershell
python scripts/smoke_alpaca.py
```

It prints the account status, equity and buying power. It exits non-zero without printing
anything if the resolved endpoint is not the paper endpoint.

---

## Running

### The offline path

```powershell
python -m glassbox.smoke_offline
```

Data → features → model → backtest → metrics, in one command, with the persistence
baseline on the same folds. This is GATE 1's acceptance criterion.

> **Not yet implemented.** `smoke_offline` is `GB-24`, in Sprint 2. The module exists as a
> docstring-only stub so this command's shape is fixed and the scaffold test can hold it —
> today it prints nothing and exits 0. That is the stub, not a successful run. Until GB-24
> lands, the data and feature snippets below are the runnable parts.

### The live loop

```powershell
python -m glassbox.live_loop
```

> **Not yet implemented** — `GB-26`, Sprint 3. Runs in Co-Pilot mode by default
> (`live.mode` in `glassbox/config/settings.yaml`), where every order is proposed and
> waits for your approval.

### The dashboard

```powershell
streamlit run glassbox/dashboard/app.py
```

> **Not yet implemented** — `GB-34`, Sprint 3.

### Fetching data

The historical loader caches to `data_cache/` as parquet. The first call per symbol hits
yfinance; every call after that reads the cache and never touches the network.

```python
from glassbox.config.loader import load_config
from glassbox.data import historical, quality

cfg = load_config()
bars = historical.load_history(cfg.universe, cfg)     # {symbol: DataFrame}
quality.report_all(bars, cfg)                         # gaps, NaNs, large moves
```

**The cache is a snapshot and does not refresh itself.** Delete
`data_cache/<SYMBOL>.parquet` to re-fetch. The study (`GB-49`) records the snapshot's last
bar date in every results row so a stale run says so on its own face.

### Building a model input window

```python
from glassbox.features import builder

frame = builder.build_feature_frame(bars["AAPL"], cfg)   # one column per channel
stats = builder.fit_stats(frame.iloc[:split], cfg)       # fit on TRAINING rows only
batch = builder.build_windows(frame, cfg, "AAPL", stats=stats)
```

`builder.min_history_bars(cfg)` is the fewest bars a caller must supply — 445 for the
default channel set, not 120. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for why.

---

## Tests

```powershell
pytest                       # everything
pytest -q                    # quiet
pytest tests/features        # one area
ruff check .                 # lint
black --check .              # formatting
lint-imports                 # the layer contract: no module imports from a layer above it
```

The suite is offline and deterministic. It needs no credentials, no network and no cached
data — every test builds its own fixtures.

Four suites carry most of the project's correctness argument, and are worth knowing by
name:

| Suite | What it proves |
|---|---|
| `tests/features/test_no_lookahead.py` | No feature value at `t` moves when bars after `t` change. Perturbs the future in two modes at three split points, and rejects a deliberately leaky function. |
| `tests/model/test_forecaster_contract.py` | Every forecaster satisfies the same seven properties, including exact attribution. Runs against seven deliberately broken models to prove each property can fail. |
| `tests/contracts/test_layers.py` | The layer stack holds and the validation harness never enters the live path. |
| `tests/test_no_secrets.py` | No tracked file contains a credential-shaped string. |

---

## Repository map

```
glassbox/                  the package — see ARCHITECTURE.md for the layer stack
  config/                  settings.yaml and the typed, fail-loud loader
  contracts/               frozen schemas and the Forecaster protocol
  data/                    historical (yfinance) · live (Alpaca) · quality checks
  features/                indicators · wavelets · builder  ← the keystone
  model/                   persistence · DLinear · FITS · train · predict
  engine/                  signal · rank · risk · executor
  explain/                 channel attribution · spectral · narration
  backtest/                engine · walk-forward · metrics
  experiments/             the comparative study and its report
  dashboard/               the Streamlit app
  live_loop.py             the live trading loop
  replay.py                offline replay of a recorded day
  smoke_offline.py         the one-command offline path

tests/                     mirrors the package, plus shared harnesses
  causality.py             assert_causal / assert_fit_isolated — reusable, not a test
docs/
  GLASSBOX_PROJECT_SPEC.md the single source of truth
scripts/
  smoke_alpaca.py          paper-account connectivity check
  compare_sources.py       yfinance vs Alpaca, field by field
reference/                 read-only third-party material — excluded from lint, tests and packaging
data_cache/                parquet bar cache and the data-quality report
```

Governance files at the root: `CLAUDE.md` (how the codebase is worked on),
`PROGRESS.md` (what is built), `DECISIONS.md` (why, one entry per contract change),
`IDEAS_PARKED.md` (what is deliberately not built), `SOLO_BUILD_PLAN.md` (the build
order).

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: No module named 'pytest'` / `'pandas'` / `'yfinance'` | The virtual environment is not activated, or `pip install` ran against a different interpreter. Check with `python -c "import sys; print(sys.executable)"` — the path must be inside `.venv`. See step 2. |
| `ModuleNotFoundError: No module named 'glassbox'` | Same cause, seen from outside the repository root. From inside it, `glassbox` imports whether or not you activated — see the note in step 2. |
| **Anything failing on a deeply nested clone** — a `git clone` that reports success but leaves files missing, `git status` claiming every file is deleted, `python -m venv` failing at `ensurepip`, or `OSError [Errno 2]` during `pip install` naming a path under `lxml` | **Windows `MAX_PATH` is 260 and the longest path this repository tracks is 66 characters**, so the clone root must stay under **193**. Measured 23 Aug 2026 from a 187-character root, in the order the failures actually occur: **(1)** `git clone` dropped one tracked file, reported *every* file as deleted because git's own `stat` calls fail, and **exited 0** — success reported for data loss; **(2)** with `-c core.longpaths=true` the clone was complete, but **`python -m venv` then failed at `ensurepip`**, leaving `python.exe` with no `pip.exe`; **(3)** `pip install` was never reached, so the `lxml` error this row used to describe is not the first thing you hit. **Two different settings, fixing different steps:** `git config core.longpaths true` fixes **git only**; the Windows registry `LongPathsEnabled` (or a shallower root) is what **Python** needs. Running the git one alone makes the clone succeed and the venv still fail. **Simplest fix: clone nearer the drive root** — verified clean end to end from `C:\gb-audit`. |
| `unauthorized` from the Alpaca smoke script | `.env` still holds placeholders, or the keys are from a live account rather than a paper one. |
| The smoke script exits 2 without output | `ALPACA_BASE_URL` is not the paper endpoint. This is a refusal, not a bug. |
| `pytest` collects 0 tests | You are not in the repository root. |
| A test fails naming a credential-shaped string | A secret has reached a tracked file. Rotate the key first, then remove it — `DECISIONS.md` records the procedure. |

---

## Licence and status

Academic coursework, Bar-Ilan University, Aug–Oct 2026. Not investment advice, not a
product, and never connected to a live brokerage account.
