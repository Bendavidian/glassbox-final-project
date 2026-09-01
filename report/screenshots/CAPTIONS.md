# Console screenshots — captions for the architecture report

Real captures at 1920×1080 through headless Chromium against a running Streamlit instance.
Not mock-ups: each frame is what a reader sees.

The console was restyled on 1 September 2026 (GB-63c) into **Vermillion Slate**. No cards,
no boxes, no fills: everything sits on `#0E1116`, separated by three rule weights and
whitespace. One monospace family throughout. Vermillion `#E8542A` is chrome only — section
rules and active states — and never expresses a value; green and red do that.

Every text colour clears WCAG AA against the ground, computed rather than quoted:
text 16.99:1, gain 8.30, dim 5.19, accent 5.16, loss 5.03. The starfield sits deliberately
below the threshold of notice at 1.57:1 and 1.94:1 — if you can find a star in these frames
while reading a number, it is too strong.

> **PROVISIONAL in one respect only.** The cumulative-equity and calendar regions read a
> **3-fold** `report/daily_equity.csv`; their pills say `folds 1-3` and are correct. GB-61's
> grid will produce the full sixteen and those two regions will change. Everything else —
> the live regions, the radar, the fold chart, the reliability figure — is final.

Captures 01–04 and 06 are against **`checkpoints/live`**, during a real session on
1 September that decided bar `2026-08-31`. Capture 05 needs a FITS arm, which the deployed
configuration does not run, so it comes from a replay:

    streamlit run glassbox/dashboard/app.py -- --state-dir checkpoints/live --source live
    streamlit run glassbox/dashboard/app.py -- --state-dir checkpoints/fits-demo \
        --source replay:fold-13

---

### 01-overview.png — the console

The status strip is one line: session state, last cycle, model, channels, universe, the
**bound state directory**, the provenance being shown, and the config hash. The bound
directory is there because of a real failure — on 27 August the panel read
`checkpoints/live` while a rehearsal wrote to `checkpoints/rehearsal`, its Co-Pilot queue
rendered empty and correct, and GATE 2's criterion 2 had to be satisfied through the API
instead. An empty queue and no recommendations are indistinguishable unless the panel says
which directory it read.

**The two pills in the middle row disagree, and that is the design working.** The radar and
the fold chart read `results.csv` and say `folds 1-16`; the cumulative equity and the
calendar read `daily_equity.csv` and say `folds 1-3`. A previous version labelled all four
with sixteen while two of them held three — a source label describing a different file from
the one it labels, in the mechanism whose entire purpose is to keep live and backtest apart.
Each region now derives its range from the frame it actually renders.

**The radar has six measured axes and no number under it.** Direction against the
always-long bar, Sharpe, return, drawdown, cancellation, flatness — each against its own
reference, each labelled with both. The reference dashboards show a composite "edge score";
there is no such quantity in this project, and inventing one in a system whose thesis is
exact attribution would be the opposite of the point.

**Reliability is the largest figure on the page and it is red.** `▼ -0.0605`: direction
0.4959 against the always-long bar's 0.5564, over sixteen folds. A console that showed P&L
while hiding how often the decider is right would be the black box this project exists to
oppose. Beneath it the band states that it **fires** — fold 16, validation Sharpe +0.483
over 8 trades, a grid maximum of fifteen candidates — because the record says how good the
forecast was and the band says how selective the system is about acting on it.

`4 OF 16 FOLDS FLAT - STOOD ASIDE` under the fold chart, and `10 UP / 13 DOWN / 152 FLAT`
under the calendar. Standing aside is most of what this system does, and the figures say so
rather than leaving a reader to infer it from empty space.

### 02-activity-rows.png — the row language

**Two independent channels in one row.** A 2px left border states what the system did; the
numerals state which way it went. The border is an *accent* on a fact the row already
carries in words — the WHAT column reads `HOLD` — so a reader who cannot resolve the colour
loses nothing, which is what `test_action_is_recoverable_from_the_row_text` holds.

Every row here is `HOLD` at the dim rule, because the deployed band evaluated bar
`2026-08-31` and declined on all five symbols. That is the honest state of this system and
not a gap in the figure: `enter_long` takes green, `exit` takes red, and a rehearsal or
replay row takes vermillion, since provenance is a fact *about* a row rather than something
the system decided.

The trend values carry `▲` and `▼` beside the colour. Green and red are never the only
carrier of a sign.

### 03-forecast-path.png — the forecast against the entry threshold

Price history in `--dim`, the calibrated entry threshold as a dashed vermillion rule, and
the forecast continuing past the last completed bar dashed in **green or red by the
direction it predicts** — GOOGL and NVDA up, AAPL, AMZN and MSFT down.

**These charts drew in the blue ramp until this capture was taken.** The ramp encodes which
channel and which frequency band; a price line and a forecast are neither. It was found by
looking at a screenshot, because the boundary test checked the radar and the fold chart and
had never looked at the forecast chart — a guard whose substrate was not the thing being
claimed. The test now enumerates every chart builder the module exports and fails on an
unclassified one.

Each region states `Last completed bar 2026-08-31 · advances once per trading day`. The
model consumes completed daily bars and this subscription refuses intraday quotes, so
nothing is styled to imply a live feed.

### 04-attribution.png — exact attribution, per channel

One decision expanded: the narrative, then signed contribution bars summing to the forecast,
with the cancellation line stating how much of the gross view survived. Sign is position
relative to the centre line, never hue.

Attribution is algebra, not approximation: the worst
`|sum(per_channel) − forecast_total|` across these records is **2.551e-09** against the
1e-5 contract.

### 05-spectral-fits.png — the spectral panel, under FITS

**FITS is not the deployed arm.** The strip in 01 reads `MODEL DLINEAR`, which is correct;
this frame comes from a replay of fold 13 under FITS, run specifically so the frequency
decomposition could be shown at all. Nothing here describes what the live system is doing.

All 24 frequency contributors with the RIN mean as its own row, and the panel refusing to be
read as a dominant-cycle story: `NO SINGLE CYCLE CARRIES THIS FORECAST — STRONGEST IS 12.1%
OF 24 CONTRIBUTORS`. Below it, gain and phase in days rather than radians.

**This is the blue ramp's entire remaining territory**, along with the attribution bars in
04. No green or red appears in this frame; tests hold both directions of that boundary.

The panel had never rendered from a real decision record before 28 August 2026 —
`explain_spectral` was defined, exported and unit-tested, and had no caller anywhere in
`glassbox/`, so every record carried `per_frequency=None`.

### 06-reflow-1280.png — the layout at a narrow viewport

The same page at 1280×900. Sections stack, the status strip wraps, and nothing overflows.
There is no fixed width anywhere: with no cards to hold a shape, the layout is columns and
rules, and it reflows because nothing was pinned in the first place.
