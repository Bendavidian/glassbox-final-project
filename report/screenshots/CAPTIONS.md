# Console screenshots — captions for the architecture report

Real captures from a running Streamlit instance in a browser. Not mock-ups: each frame is
what a reader sees.

Two invocations produce every frame here:

    streamlit run glassbox/dashboard/app.py -- --state-dir checkpoints/live --source live
    streamlit run glassbox/dashboard/app.py -- --state-dir checkpoints/fits-demo \
        --source replay:fold-13

Two viewports: 1920×1080, and 1280×900 for the two reflow frames.

## Why three frames are older than the rest

**Evidence of a defect is a preserved artefact, not a reproducible one.** A figure that
documents a defect is kept and dated rather than refreshed, because fixing the defect
destroys the ability to retake it. Frames 01, 03 and 06 were captured on 1 September 2026
and are kept at that date; each documents at least one defect that has since been fixed,
and each names the commit that fixed it.

Two of the conditions these frames rest on are not merely inconvenient to reproduce — they
are **gone**. The flat-equity defect needed a run of readings sitting exactly on the
opening balance, which was a property of the live account until 19:41:40Z on
3 September 2026 and has not been true since. The source-pill divergence needed
`results.csv` and `report/daily_equity.csv` to hold different fold ranges; both hold folds
1–16 as of 24 September 2026, so the two pills cannot disagree again. Re-running the
console on 24 September 2026 produces neither.

## Capture conditions

One row per frame. No frame inherits another's conditions: a label is a copy of a fact
about the thing it labels, and it diverges from that thing exactly like any other second
copy, which is how the previous version of this file came to describe folds it did not
hold.

| Frame | Captured | Viewport | Universe | `results.csv` | `daily_equity.csv` | Config | State dir |
|---|---|---|---|---|---|---|---|
| `01-overview.png` | 1 Sep 2026 | 1920×1080 | 5 | folds 1–16 | folds 1–3 | `484B917496` | `checkpoints/live` |
| `02-activity-rows.png` | *awaiting capture* | 1920×1080 | 20 | folds 1–16 | folds 1–16 | `1325D8B280` | `checkpoints/live` |
| `03-forecast-path.png` | 1 Sep 2026 | 1920×1080 | 5 | folds 1–16 | folds 1–3 | `484B917496` | `checkpoints/live` |
| `03b-forecast-path-fixed.png` | *awaiting capture* | 1920×1080 | 20 | folds 1–16 | folds 1–16 | `1325D8B280` | `checkpoints/live` |
| `04-attribution.png` | *awaiting capture* | 1920×1080 | 20 | folds 1–16 | folds 1–16 | `1325D8B280` | `checkpoints/live` |
| `05-spectral-fits.png` | *awaiting capture* | 1920×1080 | 5 (replay) | — | — | `F74F34D3B0` | `checkpoints/fits-demo` |
| `06-reflow-1280.png` | 1 Sep 2026 | 1280×900 | 5 | folds 1–16 | folds 1–3 | `484B917496` | `checkpoints/live` |
| `06b-reflow-1280-fixed.png` | *awaiting capture* | 1280×900 | 20 | folds 1–16 | folds 1–16 | `1325D8B280` | `checkpoints/live` |
| `07-young-session.png` | *awaiting capture* | 1920×1080 | 20 | folds 1–16 | folds 1–16 | `1325D8B280` | `checkpoints/live` |

**`awaiting capture` means the file on disk is still the 1 September 2026 frame, or does
not exist.** Those rows state the conditions a capture must be taken under, not conditions
a capture was taken under. Each such caption below is marked the same way. No row here
claims a capture that has not happened.

---

# Preserved frames

## 01-overview.png — the console, 1 September 2026

Captured during a real session that decided bar `2026-08-31`, against `checkpoints/live`.
The status strip is one line: session state, last cycle, model, channels, universe, the
**bound state directory**, the provenance being shown, and the config hash. The bound
directory is there because of a real failure — on 27 August 2026 the panel read
`checkpoints/live` while a rehearsal wrote to `checkpoints/rehearsal`, its Co-Pilot queue
rendered empty and correct, and GATE 2's criterion 2 had to be satisfied through the API
instead. An empty queue and no recommendations are indistinguishable unless the panel says
which directory it read.

**The two pills in the middle row disagree, and that is the design working.** The radar
and the fold chart read `results.csv` and say `folds 1-16`; the cumulative equity and the
daily results read `daily_equity.csv` and say `folds 1-3`. A previous version labelled all
four with sixteen while two of them held three — a source label describing a different file
from the one it labels, in the mechanism whose entire purpose is to keep live and backtest
apart. Each region now derives its range from the frame it actually renders. **This frame
is the only evidence that the two ever legitimately disagreed**, because both files hold
folds 1–16 as of 24 September 2026.

**Reliability is the largest figure on the page and it is red.** `▼ -0.0605`: direction
0.4959 against the always-long bar's 0.5564, over sixteen folds. A console that showed P&L
while hiding how often the decider is right would be the black box this project exists to
oppose.

`4 OF 16 FOLDS FLAT - STOOD ASIDE` sits under the fold chart. Standing aside is most of
what this system does, and the figure says so rather than leaving a reader to infer it from
empty space.

This frame documents six defects that have since been fixed.

**1. A motionless session painted green.** `SESSION EQUITY` reads `— +0.00` beside a line
drawn in gain green. `_segments` seeded the side from the first reading that *differed*
from the opening — reading forward through the series and applying the answer backwards —
and with an empty book there was no such reading, so the fallback made the opening its own
first move and `GAIN if opening >= opening` painted a flat session as a gain. Green and red
are the page's gain/loss carriers, so the chart stated a gain the readout beside it denied.
Fixed in **22d7136** (3 September 2026): the side now comes from `status_colour`, the same
three-state function the fold bars and the cumulative area have always used, and flat is a
state rather than the absence of one.

**2. Eight readings stretched across the full plot band.** `8 READINGS THIS SESSION` under
a 116-unit plot inside a 180-unit card, which reads as an empty panel rather than a short
one — and the live session sits in that state every morning until the loop has polled ten
times. Fixed in **b866cc4** (D6, 23 September 2026): under ten readings the plot band is 48
units and the card 112, the chrome is untouched so no label is clipped, and the reading
count is drawn at every size because it is what tells a reader the panel is small on
account of the session being young rather than the result being small.

**3. A radar whose caption claimed references it did not have.** Five of the six axes —
Sharpe, return, drawdown, cancellation, flatness — print a bare number, while `DIRECTION`
alone reads `0.4959 vs 0.5564`. Beneath them the caption reads *"Six measured axes, each
against its own reference. No composite score."* The page stated something a reasonable
reader would take as true and which was true of one axis in six. Fixed in **9f95b9c** (D4,
23 September 2026): every axis carries its own reference and its own unit, and an axis with
no reference says `NO REFERENCE` rather than going quiet.

**4. The radar polygon drawn in green.** Green and red are gain/loss status on this page
and never appear inside a data encoding; the ramp encodes data. The polygon was a data
encoding wearing a status colour, which makes a shape read as a verdict. Fixed in
**9f95b9c** (D4) together with the reference work.

**5. Cumulative equity reported with no reference.** One curve, `▼ -1.20%` over 178 trading
days, with nothing to read it against — and spec §7.3 asks that no result be reported as an
absolute. Fixed in **f87deec** (D3, 23 September 2026): the panel draws the arm and buy and
hold as two curves and prints the delta between them.

**6. The daily results calendar.** The strip at the foot of the frame draws one tile per
trading day — 178 of them here, over the three folds `daily_equity.csv` held on
1 September 2026. Everything a reader took from it came from the counts printed beneath,
and the sixteen-fold file that replaced it would have put 922 tiles in a 700px row: 2px a
tile with no gap, which renders as a smear. Removed in **fb633f8** (D5,
23 September 2026): the counts and the window they cover are the panel, and the panel
declares in its caption that day counts are not a result.

**No frame in this set documents the positions panel**, because no position was held on
1 September 2026 — the frame reads `NO POSITIONS HELD`. The three defects closed in
**9b760d5** (D1, 22 September 2026), **780823a** (D2, 23 September 2026) and **b294e04**
(D5a, 23 September 2026) — a stop room measured against a stale book instead of the
broker's working stop, a liveness claim made by each surface separately, and position
sparklines that did not say whose they were — have no before-figure. Frame 02 is their
after-figure.

## 03-forecast-path.png — the forecast against the entry threshold, 1 September 2026

Price history in `--dim`, the calibrated entry threshold as a dashed vermillion rule, and
the forecast continuing past the last completed bar dashed in **green or red by the
direction it predicts** — GOOGL and NVDA up, AAPL, AMZN and MSFT down. Where a threshold
falls outside the plotted range the chart says so in words — MSFT reads
`ENTRY THRESHOLD 526.71 — OFF SCALE` — rather than drawing a rule it cannot place.

**These charts drew in the blue ramp until this capture was taken.** The ramp encodes which
channel and which frequency band; a price line and a forecast are neither. It was found by
looking at a screenshot, because the boundary test checked the radar and the fold chart and
had never looked at the forecast chart — a guard whose substrate was not the thing being
claimed. The test now enumerates every chart builder the module exports and fails on an
unclassified one.

**The defect this frame documents: a green pill reading `LIVE 2026-08-31 bar`, five times,
during an open session.** The bar date is correct and the sentence beneath each chart —
`Last completed bar 2026-08-31 · advances once per trading day` — is correct, because the
model consumes completed daily bars and the loop drops the bar in progress. The pill is the
problem. `LIVE` in green, beside the previous trading day's bar while the market was open,
is styling that implies a live feed; a working look-ahead guard and a dead feed rendered
identically, and the person who misread it wrote the loop. Fixed in two steps: **22d7136**
(3 September 2026) made the pill name its axis and let `bar` qualify the date, so a pill
that cannot see the clock cannot make a claim about it; **cb92425** (D2b,
23 September 2026) gave `LIVE` one meaning across every surface that uses it and made the
loop leave a heartbeat a program can read, so a claim of liveness rests on evidence rather
than on the page's own timer.

`03b-forecast-path-fixed.png` is the after-frame, same charts and same viewport.

## 06-reflow-1280.png — the layout at a narrow viewport, 1 September 2026

The same page at 1280×900. Sections stack, the status strip wraps, and nothing overflows.
There is no fixed width anywhere: with no cards to hold a shape, the layout is columns and
rules, and it reflows because nothing was pinned in the first place.

**Retained rather than retaken, because it is the only witness at this width to the six
defects catalogued under 01.** The flat-green equity line, the eight readings in a full
band, the unreferenced radar under a caption claiming otherwise, the green polygon, the
unreferenced cumulative curve and the calendar strip are all present here at 1280×900. The
design chapter claims the page reads at both widths; resting that claim, and the six
defects, on a single file at a single width would make the archive depend on one artefact.
Neither this frame nor 01 can be retaken.

`06b-reflow-1280-fixed.png` is the after-frame, same viewport. **The pair exists because
preserving this one costs the report its only current narrow-viewport evidence**: the claim
that the page reads at both widths cannot rest on a 1 September 2026 frame carrying six
defects, so the claim needs a frame of its own rather than inheriting one from 01.

---

# Frames awaiting capture

Each caption below describes a frame that **has not been captured**. The file on disk is
either the superseded 1 September 2026 frame or absent. Conditions are requirements.

## 02-activity-rows.png — the row language and the positions panel

*Awaiting capture: 1920×1080, `--state-dir checkpoints/live --source live`, universe 20,
config `1325D8B280`, with the loop running and the book non-empty.*

**Two independent channels in one row.** A 2px left border states what the system did; the
numerals state which way it went. The border is an *accent* on a fact the row already
carries in words — the WHAT column names the action — so a reader who cannot resolve the
colour loses nothing, which is what `test_action_is_recoverable_from_the_row_text` holds.
`enter_long` takes green, `exit` red, and a rehearsal or replay row vermillion, since
provenance is a fact *about* a row rather than something the system decided. The trend
values carry `▲` and `▼` beside the colour: green and red are never the only carrier of a
sign.

**This frame must show the positions panel, and it is the only frame that will.** The
superseded 1 September 2026 capture read `NO POSITIONS HELD`, so nothing in the archive
shows a stop room, a position state word, or a position sparkline. The capture must carry
each held position's stop cell, its state word, and its sparkline with its own symbol drawn
into the chart — the three things closed by **9b760d5** (D1), **780823a** (D2) and
**b294e04** (D5a).

## 03b-forecast-path-fixed.png — the same charts, after

*Awaiting capture: 1920×1080, `--state-dir checkpoints/live --source live`, universe 20,
config `1325D8B280`, same charts and same viewport as 03.*

The after half of the pair. Read against `03-forecast-path.png`: same chart, same viewport,
the pill no longer claiming what it cannot see.

## 04-attribution.png — exact attribution, per channel

*Awaiting capture: 1920×1080, `--state-dir checkpoints/live --source live`, universe 20,
config `1325D8B280`.*

One decision expanded: the narrative, then signed contribution bars summing to the
forecast, with the cancellation line stating how much of the gross view survived. Sign is
position relative to the centre line, never hue.

Attribution is algebra, not approximation, and the contract is
`|sum(per_channel) − forecast_total| < 1e-5`. The worst residual measured across the 67
replay records checked on 28 August 2026 was **2.551e-09**. That figure belongs to those
records and not to this frame; the residual for whatever decision this capture expands is
the one the panel itself prints.

## 05-spectral-fits.png — the spectral panel, under FITS

*Awaiting capture: 1920×1080, `--state-dir checkpoints/fits-demo --source replay:fold-13`.*

**FITS is not the deployed arm.** The status strip reads `MODEL DLINEAR`, which is correct;
this frame comes from a replay of fold 13 under FITS, run specifically so the frequency
decomposition could be shown at all. Nothing here describes what the live system is doing,
and its decision dates fall in 2025 rather than 2026 for that reason.

The panel shows every frequency contributor with the RIN mean as its own row, and refuses
to be read as a dominant-cycle story: `NO SINGLE CYCLE CARRIES THIS FORECAST — STRONGEST IS
12.1% OF 24 CONTRIBUTORS`. Below it, gain and phase in days rather than radians.

**This is the blue ramp's entire remaining territory**, along with the attribution bars in
04. No green or red appears in this frame; tests hold both directions of that boundary.

The panel had never rendered from a real decision record before 28 August 2026 —
`explain_spectral` was defined, exported and unit-tested, and had no caller anywhere in
`glassbox/`, so every record carried `per_frequency=None`.

**`checkpoints/fits-demo` was built on 28 August 2026 under config `F74F34D3B0` and the
five-symbol universe.** The model-shaping hash includes the model, so this replay is
correctly refused under a DLinear config; whether it loads under the 24 September 2026
configuration must be confirmed on screen before the frame is trusted.

## 06b-reflow-1280-fixed.png — the layout at a narrow viewport, after

*Awaiting capture: 1280×900, `--state-dir checkpoints/live --source live`, universe 20,
config `1325D8B280`, same viewport as 06.*

The after half of the 1280×900 pair, and the report's only current evidence that the page
reads at the narrow width. Sections stack, the status strip wraps, nothing overflows — and
the six defects catalogued under 01 are absent, which is the point of capturing it at this
width rather than inferring it from the 1920 frames.

Twenty symbols rather than five means more forecast cards stacking at this width, so this
capture is also where the second half of the deferred grid item shows itself or does not:
an odd symbol count leaves an unfilled, unlabelled cell in the two-column forecast grid.
The universe is even, so the cell should not appear; if it does, the item is worse than
recorded and `IDEAS_PARKED.md` needs amending rather than the page needing a patch here.

## 07-young-session.png — the session equity panel early in a session

*Awaiting capture: 1920×1080, `--state-dir checkpoints/live --source live`, within the
first nine equity readings of a session.*

The after-frame for defect 2 under 01. The panel is a 112-unit card rather than 180, the
curve and the opening-balance rule are both drawn, and the reading count stays legible —
small because the session is young, and saying so.

The dashboard writes its own equity readings to an append-only side file and keeps a
seven-hour window, so this frame can only be taken when that window holds fewer than ten
readings: early in a session that follows a gap of more than seven hours, or against a
fresh state directory. Outside that window the panel is at full height and the frame cannot
be taken.

---

# Colour and contrast

The console was restyled on 1 September 2026 (GB-63c) into **Vermillion Slate**. No cards,
no boxes, no fills: everything sits on `#0E1116`, separated by rule weights and whitespace.
Vermillion `#E8542A` is chrome only — section rules and active states — and never expresses
a value; green and red do that, and never appear inside a data encoding. The blue ramp
encodes data and never carries status. The type scale is 12 / 14 / 16 / 30 px with a floor
of 12.

**Two families, not one.** `IBM Plex Mono` sets the English chrome and every numeral, and
the Hebrew narrative keeps a sans stack, because a Hebrew sentence set in IBM Plex Mono
falls back anyway.

Every text colour clears WCAG AA against the ground. Computed from `glassbox/dashboard/tokens.py`
on 24 September 2026, not quoted from anywhere:

| Token | Colour | Contrast |
|---|---|---|
| `TEXT` | `#F0F3F7` | 16.99:1 |
| `GAIN` | `#22C55E` | 8.30:1 |
| `DIM` | `#9AA2AC` | 7.33:1 |
| `ACCENT` | `#E8542A` | 5.16:1 |
| `LOSS` | `#EF4444` | 5.03:1 |

The starfield sits deliberately below the threshold of notice at 1.57:1 and 1.94:1 — if you
can find a star in these frames while reading a number, it is too strong.

**This table describes the frames awaiting capture, not the three preserved ones.** `DIM`
was `#7D8794` at 5.19:1 when the 1 September 2026 frames were taken, and **18f6382**
(2 September 2026) raised it to `#9AA2AC` at 7.33:1 — one day after the captures. The dim
text in 01, 03 and 06 is therefore the darker value, and a reader measuring those frames
against the table above will find 5.19 rather than 7.33 and be right.

The previous version of this file gave 5.19 for the current page. That number was correct
when it was written and the token moved underneath it: a ratio copied out of a module is a
second copy of a fact, and it diverges from the fact exactly like every other second copy
in this project. Recomputing it here fixes this instance and not the class — nothing fails
if `DIM` moves again.
