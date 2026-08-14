# Sprint 4 Prompts — FITS, Study, Report

**26 September – 10 October 2026 · ends at GATE 3, submission**

**Goal:** the research arm lands, the study runs, the report is written.

**Code freeze: 3 October.** Anything not working by then is written up as declared
future work. The last week is for writing, and the report is what you are graded on.

---

## Days 1–3 · GB-41 + GB-42 · The FITS core · 10 SP

> Plan mode. **Write GB-42's test before you train anything.**

**GB-42 first — yes, before GB-41 is finished:**

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md §6.3, then write the test for GB-42
BEFORE the FITS implementation exists.

Write tests/test_fits_amplitude.py:

    def test_fits_amplitude_reconstruction():
        """Guards the irFFT amplitude trap. torch.fft.irfft normalises by output
        length, so extending a series from L to L+H shrinks every amplitude by
        L/(L+H) unless corrected by multiplying the output by (L+H)/L.

        Symptom if missed: forecasts look plausible in shape but are systematically
        flat, direction accuracy degrades quietly, and MSE may even improve because
        a flatter forecast sits closer to zero. It looks like a modelling problem
        and it is a bug.
        """
        period, amplitude = 12, 0.8
        t = np.arange(L)
        sinusoid = amplitude * np.sin(2 * np.pi * t / period)

        model = FITSForecaster(input_len=L, horizon=H, cutoff_period_days=5)
        backcast = model.reconstruct(sinusoid[None])[0]

        np.testing.assert_allclose(backcast, sinusoid, atol=1e-4)

Also write a companion test asserting that REMOVING the (L+H)/L scaling makes the
reconstruction fail with a ratio matching L/(L+H) — so we know the test detects
exactly the bug it is meant to detect.

The test will fail until GB-41 lands. That is intended. Do not implement FITS in
this task.
```

**Then GB-41:**

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md §6, then implement task GB-41.
Use Plan mode. Read §6.2 and §6.3 carefully before proposing an approach.

Implement model/fits.py with FITSForecaster, following the pipeline in spec §6.2
exactly:

    x : (B, L) log returns, close_logret channel only
      → RIN: subtract instance mean, store it for inversion
      → rFFT → (B, L//2 + 1) complex
      → LPF: keep the first COF bins, where COF = input_len // cutoff_period_days
      → ComplexLinear: (B, COF) → (B, ceil(eta * COF)), eta = (L + H) / L
        ONE complex weight matrix. This is the entire model.
      → zero-pad to (B, (L+H)//2 + 1)
      → irFFT → (B, L+H)
      → multiply by (L + H) / L        ← the amplitude correction, spec §6.3
      → inverse RIN: add the mean back
      → split: [:L] backcast, [L:] forecast

Requirements:
- Univariate by design: FITS consumes the `close_logret` channel only. Auxiliary
  channels are not fed to it. See spec §6.4 — this is deliberate, not an oversight.
- Supervision is B+F (backcast + forecast), hardcoded. The toggle is cut from scope.
- Expose reconstruct(x) returning the backcast segment, for the GB-42 test
- Implement the full Forecaster protocol and register in ALL_FORECASTERS
- Store the complex weight matrix as a named attribute so explain/spectral.py can
  read it directly in GB-45

Acceptance: tests/test_fits_amplitude.py passes. The full contract test passes
unchanged. Report the parameter count — it should be in the 5k–15k range.

If the amplitude test fails, do not adjust the test. Find the scaling error.
```

---

## Day 4 · GB-44 · Integration and shared weights · 3 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-44.

Two things:

1. Register FITSForecaster in the model factory so it is selectable by setting
   model.active: fits in settings.yaml, with no other code change anywhere.

2. Implement universe-wide weight sharing controlled by cfg.fits.individual_weights.
   When false (the default), one model trains on windows from all five symbols
   pooled together — five times the samples for the same parameter budget, which
   matters a great deal for a 10k-parameter model on ~2500 bars per symbol.

Write a test asserting that switching model.active to "fits" changes the active
forecaster with no other modification, and that shared training produces one model
that serves all five symbols.

Acceptance: config-only switching works; contract and attribution tests green for
FITS. Report training wall time for the shared model on one fold.
```

---

## Days 5–6 · GB-45 + GB-46 · Spectral explainability · 8 SP

> This produces the strongest visual in the demonstration.

**GB-45:**

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md §6.5, then implement task GB-45.

Implement explain/spectral.py.

Because forecast = irFFT(W · rFFT(x)) is linear, each retained input frequency
contributes an exactly computable amount to each output point.

Produce:
- per_frequency: dict mapping period in DAYS (not bin index) to that cycle's
  contribution to the summed forecast
- gain_phase: dict mapping period in days to (gain, phase_shift_days), extracted as
  the magnitude and argument of the complex weights. Convert phase from radians to
  days using the period, since "shifted 2.1 days forward" is what a reader understands.

Fill the Attribution's per_frequency and gain_phase fields, which are None for the
non-spectral models.

Same exactness requirement as GB-30: per-frequency contributions sum to the forecast
within 1e-5. Remember the RIN mean is part of the forecast and must appear in the
decomposition, or the sum will not close.

Acceptance: contributions sum to the forecast within 1e-5 across 1000 random windows.
Gain and phase are finite for every retained bin. Report the dominant period found on
a real AAPL window and its share of the forecast.
```

**GB-46:**

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-46.

Generate the learned frequency-response plot: weight magnitude |W| against period in
DAYS, from a trained checkpoint. This is a complete visual summary of what the model
learned to attend to, and it has no equivalent for a non-linear model — it is the
headline figure of the demonstration and the report.

Requirements:
- X axis labelled in days, not bin index. Log scale if it reads better.
- The low-pass cutoff clearly visible as the edge of the retained band, annotated
- Use the project palette: darker for slow frequencies, lighter for fast
- Save to figures/frequency_response_{symbol}.png at report resolution (200+ dpi)

Add scripts/plot_frequency_response.py taking a checkpoint path.

Acceptance: plot generated from a trained checkpoint. Show it to me and tell me what
it says about what the model learned.
```

---

## Day 7 · GB-47 + GB-48 · Causal wavelets · 8 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement tasks GB-47 and
GB-48 together.

GB-47 — features/wavelets.py

Produce wav_a1, wav_a2 and wav_a3 from the log-return series using PyWavelets with
cfg.wavelet (db4, 3 levels, rolling_window 64, symmetric mode).

THE CRITICAL DETAIL: for every bar t, decompose ONLY the trailing cfg.wavelet.
rolling_window bars and emit the last value of each reconstructed approximation
component. Standard practice transforms the whole series at once, which leaks future
information because DWT filters are two-sided. This implementation must not. This is
the methodological contribution of the project and it is stated as such in the report.

- Emits values from bar rolling_window onward; earlier bars are NaN
- Pure function, no hidden state
- Must run over five symbols and ~2500 bars in seconds — if it is slow, vectorise the
  windowing, but never at the cost of causality

GB-48 — tests/test_wavelets.py

1. Apply the assert_causal harness from GB-10. Perturbing bar t+1 must leave every
   channel value at t unchanged. Test at 25%, 50% and 75%.
2. Additivity: a3 + d3 + d2 + d1 reconstructs the original return series within float
   tolerance at every emitted bar.
3. Prove the causality test has teeth: replace the rolling decomposition with a
   whole-series transform, confirm the test fails, and remove it. Report what you saw.

Acceptance: both tests green across all five symbols; the whole-series variant fails
the causality test. Report the runtime for the full universe.
```

---

## Day 8 · GB-49 + GB-50 + GB-51 · The study · 11 SP

> Long day. All three tasks are mechanical once the pieces exist.

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md §7.4, then implement tasks GB-49,
GB-50 and GB-51.

GB-49 — experiments/study.py

Run the grid from spec §7.4:
  Axis 1 — model: Persistence, DLinear, FITS
  Axis 2 — features: C0_base, C2_hybrid
  Walk-forward, up to cfg.walkforward.max_folds folds, five symbols, fixed seeds,
  data snapshot cached to parquet so the study is reproducible.

The FITS × C2_hybrid cell is DELIBERATELY SKIPPED. Record the reason in the output
rather than leaving it blank: feeding pre-filtered wavelet bands into a model whose
first operation is frequency filtering is redundant. Do not fill a cell to make a
table look complete.

Emit a tidy results.csv: one row per (model, config, fold, symbol) with all forecast
and trading metrics.

GB-50 — extend the study with the FITS cutoff sweep over cutoff_period_days in
{2, 5, 10, 20}. This answers the project's central empirical question: at what minimum
frequency does exploitable signal still exist in daily equity returns, measured net of
costs.

GB-51 — implement paired Wilcoxon signed-rank tests of every arm against persistence,
on per-fold direction accuracy and per-fold Sharpe. Report mean and standard deviation
across folds alongside each p-value. Pair correctly by fold.

Acceptance: one command runs the full grid and writes results.csv. Rerunning with the
same seed reproduces identical numbers. Report the total wall time and the headline
table. If any arm shows Sharpe above 2.0, stop and audit before reporting it.
```

---

## Day 9 · GB-52 + GB-53 · Report generator and spectral panel · 6 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-52.

Implement experiments/report.py turning results.csv into:
- The summary table: one row per arm, every metric as a DELTA against persistence,
  with the p-value from GB-51
- Per-fold boxplots of direction accuracy and Sharpe by arm
- The cutoff sweep as a line plot: cutoff period in days against direction accuracy
  delta and Sharpe delta

Hard rules from CLAUDE.md: every table shows a persistence delta; MSE on prices never
appears as a headline metric.

The report must regenerate entirely from results.csv with no rerun of the study.

Acceptance: report regenerates from the CSV alone. Show me the summary table.
```

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then implement task GB-53.

Add a spectral explanation panel to the dashboard, shown only when FITS is the active
model and hidden cleanly otherwise:
- Contribution bars by period in days, using the project palette (dark = slow, light = fast)
- The learned frequency-response curve from GB-46

Scope: those two things. No live gain/phase table — cut from scope.

Acceptance: panel renders live under FITS, hides under DLinear.

Note: if you are behind schedule, this is the first task to cut. It is the only
remaining item that is purely presentational.
```

---

## CODE FREEZE — 3 October

Nothing technical after this point. Anything not working is written up as declared
future work in the report. This rule exists because the report is what you are graded
on, and a half-finished feature is worth less than a well-written explanation of why
it was deferred.

---

## Days 10–12 · GB-55 + GB-56 + GB-57 · The report · 11 SP

You write all three chapters. Use Claude Code as an editor and critic, not a ghostwriter
— you have to defend every sentence in the presentation.

**Architecture chapters (GB-55):**

```
I am writing the technical report. Read docs/GLASSBOX_PROJECT_SPEC.md, ARCHITECTURE.md
and the actual code, then help me draft the architecture chapters:
- The layer stack and why data flows strictly upward
- The frozen contracts and how they let three models coexist as a config switch
- builder.py as the train/live parity guarantee, and the parity test that proves it
- The live decision cycle
- Execution and risk design

Critical: every diagram and claim must match the code AS IMPLEMENTED, not as
originally planned. Where the implementation diverged from the spec, say so explicitly
and give the reason — that is a strength in a report, not a weakness.

Draft it in Markdown. I will edit. Flag anything you are unsure is accurate.
```

**Methodology chapters (GB-56):**

```
Help me draft the methodology chapters of the report:
- Walk-forward design and why cross-validation is invalid for time series
- The three absolute rules from spec §7.1 and how each is enforced BY TEST
- The causal wavelet computation and why the standard whole-series approach leaks
- Cost modelling
- The persistence-baseline reporting rule, and the reasoning behind excluding
  price MSE as a headline metric

For every methodological claim, point to the specific test file and test function
that enforces it. A claim without an enforcing test does not belong in this chapter.

Draft in Markdown. I will edit.
```

**Results chapters (GB-57):**

```
Help me draft the results, discussion and future work chapters, using results.csv
and the figures generated in GB-52.

Requirements:
- Every table carries a persistence delta column
- Report what the numbers do and do not support. If no arm beats persistence net of
  costs, say so plainly — the proposal declares that as a legitimate and expected
  outcome, and hedging it would be worse than stating it.
- Explain the deliberately empty FITS × C2_hybrid cell with its reasoning
- Include the declared future work from spec §11, especially the Regime Guard, framed
  as a scope decision rather than an omission
- No claim about market-beating performance anywhere

Draft in Markdown. Be direct about weak results. I will edit.
```

---

## Day 13 · GB-58 + GB-54 · Deck and rehearsals · 5 SP

```
Help me build the presentation deck:
1. The problem — opacity and multi-scale noise
2. The research question, two arms
3. The architecture — one slide, the layer stack
4. The demonstration
5. Results, honestly
6. Conclusions and future work

Lead the technical section with the frequency-response figure — it is the strongest
visual and it needs no explanation of the mathematics to land.

Every claim must trace to a figure in the report. Keep it to the time limit.
```

Then rehearse the demo twice, end to end, in Replay mode. Time both. If either needs
intervention, fix it — a demo that needs rescuing in front of a supervisor undoes a
lot of good work.

---

## Day 14 · GB-59 · Reproducibility audit · 3 SP

```
Read CLAUDE.md and docs/GLASSBOX_PROJECT_SPEC.md, then perform task GB-59.

Guide me through a full reproducibility audit:
1. Clone the repository to a fresh directory
2. Install from the lockfile in a clean virtual environment
3. Run the full pipeline: data fetch → features → study → report
4. Compare the generated results.csv against the committed one

Document every manual step required. Then either eliminate it or record it in the
README. A step that exists only in my head is a reproducibility failure.

Report: does a clean clone reproduce the reported results exactly? If not, what differs
and why?
```

---

## Day 15 · GB-60 · GATE 3 and submission · 2 SP

```
Read CLAUDE.md, docs/GLASSBOX_PROJECT_SPEC.md §8 and SOLO_BUILD_PLAN.md §7.

Walk the GATE 3 checklist and give me an honest pass/fail on each:
- The study runs from one command and produces results.csv
- Every table reports deltas against persistence
- The technical report is complete
- The demo has been rehearsed end to end twice
- The repository reproduces from a clean clone

Then give me the commands to tag the repository v1.0-submission and produce the final
archive.

Record the gate outcome in PROGRESS.md.
```

---

## Submission — 10 October 2026

What you hand in:

- A working autonomous trading system, running on a paper account, that explains
  every decision with exact attribution
- A comparative study answering a real question, with honest results and significance
  testing
- A technical report where every methodological claim points to the test that enforces it
- A demonstration that does not depend on market hours or connectivity

**One last thing.** If the study shows that nothing beats persistence net of costs,
that is the result. Report it plainly. A student who reports a null result with clean
methodology has demonstrated more than one who reports a Sharpe of 2.5 and cannot
explain where it came from — and the second one gets asked exactly that question in
the defence.
