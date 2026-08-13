# DECISIONS

Architectural decision records. One entry per decision that changes a contract,
adds a dependency, or cuts scope. Newest first.

Format: date · decision · reasoning · consequence.

---

## 2026-08-12 — FITS adopted as the spectral core; DLinear retained as baseline

**Decision.** The model layer holds three forecasters: `Persistence` (reference),
`DLinear` (established baseline, already understood), and `FITS` (new spectral core).
Wavelet channels remain, repositioned from "the core contribution" to
"the explicit arm of the comparative study".

**Reasoning.** FITS and wavelets are not competitors — they sit in different layers.
FITS is a model; wavelets are features. Keeping DLinear is what makes FITS measurable;
without a baseline there is no claim to make. This framing turns an assumption
("wavelets help") into a measurable question ("explicit or implicit frequency
decomposition — which helps more, if either?").

**Consequence.** The study grid becomes 3 models x 2 feature configs, with the
`FITS x C2_hybrid` cell deliberately empty (see spec §6.4).

---

## 2026-08-12 — Execution window compressed from 14 weeks to 8

**Decision.** The plan is rebuilt around 15 Aug – 10 Oct 2026: four two-week sprints,
three hard gates.

**Reasoning.** The original 14-week plan is not executable in 8 weeks.

**Consequence.** The ablation grid shrinks; `NLinear` becomes optional; the
Regime Guard moves to declared future work; Sprint 4 carries the entire report
and nothing technical may spill into it.

---

## Template

## YYYY-MM-DD — <short title>

**Decision.**

**Reasoning.**

**Consequence.**
