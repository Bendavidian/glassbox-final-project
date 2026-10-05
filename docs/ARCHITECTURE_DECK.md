# GlassBox Trader — Architecture and External Components

Submitted deck. Fourteen slides.

Every external component this project uses is disclosed here, with its version and what it
is used for. Every line traces to `docs/ARCHITECTURE_INVENTORY.md`, which cites the file and
line in the repository.

| # | slide |
|---|---|
| 1 | The system, in one view |
| 2 | Seven layers |
| 3 | Layer boundaries, enforced |
| 4 | Frozen interfaces |
| 5 | **Runtime dependencies** |
| 6 | **Development and test dependencies** |
| 7 | **The reference project: what was taken** |
| 8 | **The reference project: what was refused** |
| 9 | **The three models: port, reimplementation, original** |
| 10 | **External services** |
| 11 | **AI-assisted development** |
| 12 | Artefacts and data flow |
| 13 | What is enforced rather than intended |
| 14 | Disclosure statement |

---

## Slide 1 — The system, in one view

**On the slide:**

> An explainable algorithmic paper-trading system.
> 51 modules, 22,450 lines of package code, 26,281 lines of test.
> Trades a real broker. Every decision decomposed exactly.

| | |
|---|---|
| Language | Python 3.12 |
| Runtime dependencies | 13 declared, 99 pinned in the lock |
| Dev and test dependencies | 6 |
| Tests | 1,598 collected across 60 files |
| Test to code ratio | 1.16 : 1 |

**Notes.** Open with scale so the rest is read as a system rather than a script. The
dependency counts are on this slide deliberately — the disclosure starts here, not at slide 5.

---

## Slide 2 — Seven layers

**On the slide:**

| layer | responsibility | lines |
|---|---|---|
| L0 config, contracts | the only reader of settings; the frozen schemas | 1,433 |
| L1 data | two sources, one shape | 748 |
| L2 features | the only place a model input window is assembled | 873 |
| L3 models | four forecasters, one protocol | 2,270 |
| L4/L5 engine | sizing, risk, orders, reconciliation | 1,730 |
| L6 explain | exact attribution and narration | 1,002 |
| L7 console | operator surface; computes nothing | 4,308 |
| X harness | the grid, the report, the statistics | 3,215 |
| top level | live loop, replay, records, lock | 4,292 |

**Notes.** The harness sits *above* the live path rather than inside it: the constraint that
matters is that the live path never imports it. Console line count is large because every
chart is hand-built SVG rather than a chart library, which is what makes chart geometry
testable without a browser.

---

## Slide 3 — Layer boundaries, enforced

**On the slide:**

> Two import-linter contracts. `root_package = "glassbox"`.

**Contract 1 — data flows strictly upward**
experiments → backtest → dashboard → explain → engine
→ model → features → data → contracts → config

**Contract 2 — the live path never imports the validation harness**
source: live_loop, replay, records

> Enforcement is doubled: the tool runs in CI, and `tests/contracts/test_layers.py` runs it
> in-process and asserts both contracts are still configured.

**Notes.** The second contract exists because of `records.py` specifically: it holds the live
trade log, so it is the module most tempted to import `backtest.engine.Trade`. That type moved
to `contracts/schemas.py` instead. Deleting a contract from the config turns the suite red
rather than silently reducing coverage.

---

## Slide 4 — Frozen interfaces

**On the slide:**

> **`Forecaster` protocol** — `contracts/protocols.py`, five members, `runtime_checkable`
name, input_len, horizon, fitted
fit(batch, val=None) predict(X) -> (B,H)
explain(x, channels) save(path) / load(path)

> Implemented by four forecasters. **FITS was added with no edit to the contract test.**

> **Eight frozen schemas** — `contracts/schemas.py`, all `@dataclass(frozen=True)`:
> WindowBatch · ChannelStats · FitProvenance · Trade · Forecast · Attribution · Signal ·
> DecisionRecord

> **`Broker` protocol** — seven calls, four implementations: Alpaca, replay, dry-run,
> and the test double.

**Notes.** `EXACTNESS_TOLERANCE = 1e-5` is defined once and imported by the contract test
rather than restated. A model is not integrated until it passes
`tests/test_forecaster_contract.py` unchanged.

The cost of freezing was paid once: GB-66's acceptance criterion could not be satisfied
because WITS has no per-bin frequency response. **The protocol did not bend to make the
comparison possible**, and the result of that is in the report.

---

## Slide 5 — Runtime dependencies

**On the slide:** 13 declared in `pyproject.toml`. 99 distributions pinned in
`requirements.lock` — the declared set plus 80 transitive. CI installs the lock, then the
package with `--no-deps`.

| package | version | what this project uses it for |
|---|---|---|
| pandas | 3.0.5 | every bar frame, feature frame and results table |
| numpy | 2.5.2 | **all inference.** Weights are held as arrays after fitting, so inference carries no torch state — this is what makes determinism true by construction |
| torch | 2.13.0 | **training only.** Imported inside `fit`, never at module scope |
| yfinance | 1.6.0 | the historical price source, `auto_adjust=True`, cached to parquet |
| alpaca-py | 0.44.0 | live daily bars, and paper order submission |
| PyWavelets | 1.9.0 | the causal DWT feature channels; WITS's analysis matrices, **measured from pywt rather than re-derived**; the level ceiling, **asked rather than reimplemented** |
| scipy | 1.18.0 | two call sites: paired Wilcoxon, and the Spearman pairing MAE with flatness |
| streamlit | 1.61.1 | hosts the console. Imported inside `main()` |
| pandas-market-calendars | 5.4.0 | the exchange calendar: fold boundaries, gap reports, the live session window |
| pyyaml | 6.0.3 | parses `settings.yaml`, in exactly one module |
| python-dotenv | 1.2.2 | loads credentials, in the config layer only |
| pyarrow | 24.0.0 | parquet engine for the committed data snapshot |
| matplotlib | 3.11.1 | three report figures. Imported inside the render functions |

> **Every one of the 13 carries a dated entry in `DECISIONS.md` recording why it was added.**
> A test asserts the declared set is a subset of the lock.

**Notes.** Three lines are worth saying aloud.

**numpy for inference, torch for training** is a design decision and not an accident: it is
what makes bit-identical determinism achievable, because inference carries no global framework
state.

**PyWavelets is asked, not copied.** `pywt.dwt_max_level` is called rather than
reimplemented, with the reason recorded in the source: a second copy of a fact pywt already
holds. The WITS analysis matrices are *measured* from pywt by pushing basis vectors through it,
rather than derived by hand from the wavelet definition.

**The subset test exists because of a real failure:** on 20 August matplotlib was added to
`pyproject.toml` alone. Green locally, red in CI.

---

## Slide 6 — Development and test dependencies

**On the slide:** 6 declared.

| package | version | what it does here |
|---|---|---|
| pytest | 9.1.1 | the suite. `pythonpath = [".", "tests"]` so bare `pytest` works |
| hypothesis | 6.165.5 | property tests in exactly three files: the risk layer's "no configuration produces an over-limit position", `room_detail`, and attribution exactness |
| import-linter | 2.13 | machine-enforces the layer rule. Two contracts |
| ruff | 0.16.3 | lint, `target-version = py312`, `reference/` excluded |
| black | 26.5.1 | format check, `reference/` excluded |
| playwright | 1.61.0 | drives the console headless for the screenshot captures; pinned exactly at 1.61.0 |

> Shared test infrastructure written here, not imported:
> `tests/causality.py` (271 lines) · `tests/sweep.py` (239) · `tests/fake_broker.py` (212)

**Notes.** The three infrastructure files are the point of this slide. They are not
dependencies — they are the project's own harnesses, and two of the study's findings came from
them rather than from a library.

`tests/sweep.py` has two properties that are **refusals rather than features**: a
`SweepResult` cannot be used as a boolean (`bool()` raises `TypeError` and names the attributes
to read instead), and a sweep of a single cell is refused. It exists because a parity floor
recorded as "verified" had been tested at one symbol and one timestamp, and held in 32 of 125
cases when swept.

`tests/fake_broker.py` is the test double. Making it faithful to Alpaca's actual refusals
turned 26 tests red across three files — **a measurement of what had never been tested**, not a
regression. The rule it produced: **a test double may refuse more than the real thing, never
less.**

---

## Slide 7 — The reference project: what was taken

**On the slide:**

> `reference/algotrading_project-main/` — 2.7 MB, an existing LTSF-Linear trading project,
> brought in for study.
>
> **Read-only. Never imported. Not in the package.**

**Three mechanisms keep it outside the build, none of them a convention:**

| | |
|---|---|
| packaging | `include = ["glassbox*"]`, `exclude = ["reference*"]` |
| tooling | excluded from ruff and black; `norecursedirs` for pytest |
| imports | the import-linter contract is rooted at `glassbox`, so `reference/` is outside the contract **by construction, not by an exclusion list** |

**What was taken, per `reference/REFERENCE_AUDIT.md`:**

| source | target | what was taken |
|---|---|---|
| `models/DLinear.py` | GB-13 | the `series_decomp` / `moving_avg` blocks and the two linear heads |
| `evaluation.py` | GB-19 | six return and risk formulas, judged correct |
| `backtesting.py`, `models_backtest.py` | GB-18 | event-loop **structure** and the `Position` / `ActionType` / `PositionType` enums — structure only |
| `utils/tools.py` | GB-15 | `EarlyStopping`, `adjust_learning_rate` |
| `data_provider/data_loader.py` | GB-9 | **reference only, not ported** |

**Notes.** Say the ordering out loud: **`REFERENCE_AUDIT.md` was written before the code
was.** It is not a retrospective inventory of what happened to get used; it is a decision,
taken first, about what may be taken and what may not.

The three exclusion mechanisms matter because any one of them alone is a convention. Together
the directory cannot enter the package, cannot be linted as if it were ours, and cannot be
imported without the contract failing.

---

## Slide 8 — The reference project: what was refused

**On the slide:**

> The audit records refusals as well as permissions. These are the refusals, with the reason
> measured rather than asserted.

**`utils/metrics.py` — unusable, and `backtest/metrics.py` was written from scratch**

| defect | what it is |
|---|---|
| `SHARP()` | returns a hardcoded `12` |
| `SHARP2`, `SHARP5` | each defined twice, the second shadowing the first |
| one variant | contains `pred = true + 0.01` — **the prediction is overwritten with ground truth** |

> `backtest/metrics.py:4` says so in its own header.

**`strategies.py` — all four stop and target comparisons inverted**

> The audit lists each correct form. `backtest/engine.py:38` cites the audit, and the four
> cases each carry an explicit test.

**The single train/test split — rejected for walk-forward.**

**And one thing explicitly marked *not* a bug:** `moving_avg` pads both ends, making the trend
a centred moving average. That is **correct inside the input window**, all of which precedes
the forecast. Centring remains forbidden in `features/indicators.py` and `features/wavelets.py`,
which compute along the full series.

**Notes.** This is the slide that answers the plagiarism question properly, and it does it by
inversion.

Anyone can list what they used. **A reasoned refusal to use broken code — with the specific
defect named, the correct form derived independently, and a test pinning each case — is
evidence that the reference was read rather than copied.** `pred = true + 0.01` would have
produced a backtest that looked extraordinary and meant nothing.

The `moving_avg` note is the other half of the same evidence: a rule was not applied
mechanically. Centring is forbidden as a feature and permitted inside the input window, and
the distinction is stated with its reason.

---

## Slide 9 — The three models: what each one is

**On the slide:**

| model | parameters | lineage |
|---|---|---|
| **DLinear** | 4,800 | LTSF-Linear (Zeng et al., AAAI 2023). **Two lineages, stated separately:** the decomposition is adapted from the reference implementation; the two linear heads are reimplemented and depart from the paper as well |
| **FITS** | 1,200 | FITS (Xu, Zeng & Xu, ICLR 2024, arXiv:2307.03756). **Original implementation from the paper's description. No reference source exists in this repository.** |
| **WITS** | 882 | **Original. No paper, no reference.** Built to test whether FITS's data-independence is a property of extending a global basis or of FITS specifically |
| persistence | 0 | the zero-return baseline |

### DLinear has two lineages and they are not the same

**The decomposition — adapted from the reference implementation.** A centred moving average
with edge replication, kernel 25: the same algorithm and the same constant, rewritten from
torch `AvgPool1d` to numpy, as a function rather than two `nn.Module`s.

Edge replication is an implementation choice rather than a mathematical one — a moving average
could zero-pad, reflect, or return a shorter series, and the paper does not say which. **So
this is code-first provenance and is recorded as such.** One identifier survives the rewrite:
`front`. Nothing else does: no `nn.Module`, no `AvgPool1d`, no class names, and the tuple
order is reversed.

**The two linear heads — reimplemented, and they depart from the paper as well as from the
reference.** Four departures, each recorded in `DECISIONS.md` on 16 August 2026:

1. **Per-channel weights, not shared.** The reference and the paper both map every channel
   through one matrix. Here each channel gets its own per component, so attribution is a
   regrouping of terms already computed rather than a reconstruction.
2. **No intercept.** A bias would break `sum(per_channel) == forecast` with no honest channel
   to charge it to.
3. **Zero initialisation.** `nn.Linear`'s default bound assumes fan-in `L`; this architecture
   sums `C×2` such maps, so the bound is wrong by construction. The measured comparison is in
   the docstring.
4. **A single summed output series**, not a forecast per channel. Torch fits, numpy predicts.

> **Kernel 25** is carried over unchanged, flagged in the source as *"an architecture
> definition rather than a tuning knob"*. It is in the reference at `DLinear.py:48`.
>
> **The decomposition block is upstream of the reference too.** The `moving_avg` /
> `series_decomp` pair originates in Autoformer and was reused by LTSF-Linear, so the recorded
> chain is Autoformer → LTSF-Linear → this reference project → here.

**WITS's prediction was pre-registered.** It was written into the expansion plan before a line
of the model was written, and `scripts/operator_null_control.py` quotes it back when it
measures the result.

**Notes.** Say the two lineages separately and do not compress them into one word. The
comparison was done file against file: no line of the reference appears in
`glassbox/model/ltsf.py`, but the padding expression `(kernel - 1) // 2` is token-identical and
the five steps are in the same order. **Whoever wrote that function had the reference open.**
That is the fact, and stating it is stronger than a label that smooths it.

The heads are the opposite case and the record shows it: `DECISIONS.md` argues against each
reference choice individually, which is evidence of reading and departing rather than of
copying.

FITS is the strongest line on the slide: 655 lines written from a paper's description with no
source implementation available in the repository, and two places where it reasons about the
published architecture rather than copying it — the dead row retained because the source paper
carries the same one, and an amplitude correction the project derived before finding the spec
had stated it wrongly.

## Slide 10 — External services

**On the slide:**

| service | provides | constraints encoded in code |
|---|---|---|
| **yfinance** (unauthenticated) | historical daily OHLCV, `auto_adjust=True` → total returns | per-symbol, never bulk. The cache is authoritative: a stale file is returned unchanged unless forced, so the study runs against a fixed snapshot |
| **Alpaca Data API** | live daily bars, in a schema identical to yfinance's | `Adjustment.ALL`. Feed hardcoded to SIP **with no fallback** — IEX measured wrong by up to 193 bps against a 1 bp tolerance. **No cache:** a cached bar in a trading decision is a stale price |
| **Alpaca Trading API** | orders, positions, account, order history | **paper only.** `require_paper_endpoint` refuses any non-paper base URL **before any call is made**. Both clients share a 45 s read timeout |
| **GitHub REST API** | CI run status | dev tooling only, not in `glassbox/` |

**Credentials:** `.env`, gitignored, template committed. `tests/test_no_secrets.py` scans every
tracked file for credential-shaped strings and asserts `.env` is untracked — **written after a
real paper key reached a local commit on 14 August.**

> **The SIP entitlement's boundary is documented rather than assumed:** historical daily bars
> are served; `get_stock_latest_bar` is refused with *"subscription does not permit querying
> recent SIP data"*. Anything reaching for a live quote rather than a completed bar meets that
> wall.

**Notes.** Two lines are worth saying.

**The 45-second timeout exists because of a 90-minute stall.** No HTTP read had a timeout,
because the Alpaca SDK takes no `timeout` parameter. The retry policy did not help — it bounds
the number of attempts and nothing bounded their duration. The value was chosen on the
provenance of the slowest healthy call, not on a ratio, and a code guard raises if a future
SDK renames the private session it patches.

**The secrets test exists because a key reached a commit.** It is the mechanism, and the
incident is dated.

---

## Slide 11 — AI-assisted development

**On the slide, stated plainly:**

> This project was built with Claude Code. That is disclosed here in full, and it is recorded
> in four independent places in the repository.

| record | what it holds |
|---|---|
| **Commit trailers** | **161 of 163 commits** carry `Co-Authored-By: Claude`. The two without are the document-only commits that bootstrap the project |
| **`CLAUDE.md`** | ~250 lines, checked in. The operating contract: six rules, a definition of done, and **nine numbered instances of one failure class, each dated with what it cost** |
| **`prompts/`** | four sprint files, 1,572 lines. The actual session prompts, one task per session |
| **`START_HERE.md`** | states the division: *"For the university — you submit these, Claude Code never reads them"* against *"For the repository — Claude Code reads these on every session"* |

**And the method itself is in the record.** `DECISIONS.md` — 5,425 lines, 121 entries —
repeatedly records where the assistant was **wrong and was corrected**, by the supervisor or by
measurement:

- the parity floor derivation, corrected by the author
- the scaler hypothesis, supplied by the author, then confirmed for DLinear and **refuted for
  FITS**
- an entry naming the pattern: *"That is the fifth time a first description in this family has
  been narrowed by a measurement."*

> **`glassbox/explain/` imports no perturbation-attribution library, and a test asserts it.**
> Attribution here is algebra, not a model's opinion. The architecture chapter's §55.10 carries
> the heading *"It will not take an LLM's word for anything."*

**Notes.** Deliver this slide without apology and without over-explaining. The disclosure is
complete, it is mechanical rather than narrative, and the most useful thing on it is the
`DECISIONS.md` line: the record contains the corrections, not just the output.

If asked directly whether the code is yours: the design decisions, the rulings, the refusals
and the corrections are in the repository under your name, dated, with the reasoning. That is
the answer, and it is verifiable rather than asserted.

---

## Slide 12 — Artefacts and data flow

**On the slide:**
data_cache/.parquet ──► features ──► models ──► engine ──► executor ──► Alpaca
(20 symbols, │
committed) └──► decisions/.jsonl
trades.jsonl
book.json
experiments/study ──► results.csv ──► report.md, stats, exposure ──► the report
└─► daily_equity.csv ──┐
└──► the console (reads, computes nothing)


| artefact | written by | read by |
|---|---|---|
| `data_cache/*.parquet` (20) | `data/historical.py` | the grid, the live loop. **Committed**, so a clean clone can run |
| `results.csv` (1,110 rows) | `experiments/study.py` | the report, the statistics, the console |
| `report/daily_equity.csv` | the grid | the console |
| `report/gross_exposure.csv` | `experiments/exposure.py` | the report |
| `report/null_control.csv` | `scripts/operator_null_control.py` | GB-66's acceptance |
| `report/momentum_probe.csv` | `scripts/momentum_probe.py` | the parked-arm entry quotes it |
| checkpoint, `thresholds.json`, `reliability.json` | `smoke_offline --prepare-live` | the live loop, the console |
| `decisions/YYYY-MM.jsonl`, `trades.jsonl`, `book.json` | the live loop | the console, the replay |
| `live.lock`, `heartbeat.json` | `live_lock.py`, the live loop | the console's liveness |

**Notes.** The data cache is committed deliberately, and a test asserts it stays tracked. A
clone with no data can run nothing, and a result that cannot be reproduced from a clone is a
claim rather than a measurement.

Note the direction at the bottom right: **the console reads files and computes nothing.** A
console that computed its own figures would be a second implementation of every metric it
shows.

---

## Slide 13 — What is enforced rather than intended

**On the slide:** the project's own rules, each as a test that fails when the rule is broken.

| rule | enforced by |
|---|---|
| layers and the live-path boundary | `test_no_layer_violations`, plus two tests that the contracts are still configured |
| the package tree matches the spec, **both directions** | `test_spec_module_exists`, `test_tree_has_no_extra_modules` |
| only the config layer reads settings or the environment | AST-based, **plus a test that the guard catches a real read and not a mention of one** |
| no numeric threshold in the signal module's source | `test_the_module_source_contains_no_numeric_threshold` |
| every feature is causal | four causality tests, **plus four tests that the harness itself can detect a leak** |
| no timestamp appears in two splits of one fold | three leak-boundary tests |
| **the explain layer imports no perturbation library** | `test_the_explain_layer_imports_no_perturbation_library` |
| every registered forecaster is exact | iterates `ALL_FORECASTERS`, so a new model cannot skip it |
| declared dependencies are in the lock | the test written after matplotlib broke CI |
| MAE never appears without flatness beside it | `test_mae_never_appears_in_a_summary_without_flatness_beside_it` |
| every panel states its source and its reference | the console's pill and reference tests |
| no secret in any tracked file | plus three tests that the scanner detects real keys and does not cry wolf |
| the grid is deterministic | `test_the_same_grid_twice_gives_the_same_numbers` |

> **1,598 tests collected. 60 files. 26,281 lines of test against 22,450 of package.**

**Notes.** The pattern worth naming: several of these test the guard rather than the code. A
causality harness that cannot detect a leak is worse than none, so four tests deliberately
introduce leaks and assert the harness fails.

And two properties in `tests/sweep.py` are **refusals rather than tests**: a result cannot be
read as a boolean, and a single-cell sweep is refused. The failure mode is unavailable rather
than discouraged.

---

## Slide 14 — Disclosure statement

**On the slide:**

> **Everything external, in one place.**
>
> **13 runtime and 6 development dependencies**, each pinned, each with a dated entry
> recording why it was added.
>
> **One reference project**, `reference/algotrading_project-main/`, read-only and excluded
> from the build by three independent mechanisms. What was taken and what was refused is in
> `reference/REFERENCE_AUDIT.md`, **written before the code**.
>
> **Two published architectures.** DLinear reimplemented with four documented departures;
> FITS written from the paper's description with no source implementation available.
> **WITS is original.**
>
> **Four external services.** Alpaca paper only, enforced before any call. yfinance
> unauthenticated. No service is called from inside the package without its constraint
> encoded beside it.
>
> **Built with Claude Code**, recorded in 161 of 163 commit trailers, an operating contract
> in the repository root, and 1,572 lines of session prompts. `DECISIONS.md` records where
> the assistant was wrong and was corrected.
>
> **Nothing else external is used.** No file in `glassbox/` carries a "copied from" or
> "adapted from" header. Persistence, the feature builder, the causality harness, the sweep
> harness, the executor, the reconciler, the live loop, the console and the explain layer are
> original.

> Full detail with file and line citations: `docs/ARCHITECTURE_INVENTORY.md`.

**Notes.** Read this slide slowly. It is the one the plagiarism clause is answered by, and it
should be the last thing on screen.

The final paragraph is the strongest sentence in the deck and it is also the most falsifiable:
it can be checked against the tree in minutes. That is why it is stated as a claim about the
repository rather than about intent.
