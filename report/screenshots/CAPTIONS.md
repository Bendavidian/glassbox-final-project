# Dashboard screenshots — captions for the architecture report

Real captures at 1920×1080 through headless Chromium against a running Streamlit instance.
Not mock-ups, and not full-page strips: each frame is what a reader sees.

The console was rebuilt on 28 August 2026 (GB-63b). The previous blueprint language —
registration marks, numbered rulers, dashed panels, a PROJECT/SYSTEM/VERSION masthead — is
gone. Cards on a near-black ground, green for gain and red for loss.

> **PROVISIONAL — these five figures must be re-captured after GB-61's grid run.**
> They were taken against a **3-fold** `report/daily_equity.csv`, generated deliberately
> so the cumulative-equity and calendar cards would render at all: the artefact is new in
> GB-63b and no full grid has produced one yet. Their pills say `folds 1-3` and are
> correct, but the real artefact will carry sixteen and the two cards will look different.
> The other three cards, and every live panel, are unaffected. Do not leave report figures
> standing that show an artefact which no longer exists.

All captures are against `checkpoints/fits-demo`, a replay of **fold 13** written by
`smoke_offline --model fits --prepare-replay 13` and driven by `replay --model fits`:

    streamlit run glassbox/dashboard/app.py -- \
        --state-dir checkpoints/fits-demo --source replay:fold-13

---

### 01-overview.png — the console

Row 1 is what is happening now; rows 2 and 3 are what was measured. **Every card carries a
source pill and no card mixes sources**, because this system has made two live trades and
stands aside on most bars while the backtest has 166 trades over sixteen folds. A console
that filled a calendar with backtest results while looking live would discredit the one
claim the project is actually making.

The session strip states the **bound state directory** — `checkpoints\fits-demo` — beside
the provenance it is filtering for. That defect cost GATE 2 its panel path: on 27 August
the dashboard was bound to `checkpoints/live` while the rehearsal wrote to
`checkpoints/rehearsal`, its Co-Pilot queue rendered empty and correct, and criterion 2 had
to be satisfied through the API instead.

`LOOP NOT RESPONDING · LAST READ 729 MIN AGO` is the escalated staleness state: past two
heartbeat intervals the wording changes from *this number is old* to *the system is not
running*, because those are different problems. `NEXT CYCLE` is measured from the loop's
own last write to `book.json`, not from the page's refresh timer — a timer would tick
smoothly past a dead loop, which is an animation rather than a measurement.

**Session equity shows a flat line and says `5 READINGS THIS SESSION`.** That is correct
and it is the common case: the loop stands aside on most bars.

**The radar has no composite score.** Five axes, each a measured metric against **its own**
reference and labelled with both. The reference dashboard shows an "Edge Score"; there is
no such quantity here, and inventing one in a system whose thesis is exact attribution
would be the opposite of the point. DLinear's direction axis reads 0.4959 against the
always-long bar's 0.5564 — the shape is small because the measurements are.

**Reliability is large, red, and unavoidable**: `▼ -0.0605`. A console that showed P&L
while hiding how often the decider is right would be the black box this project exists to
oppose, and this one's answer is that it does not beat the always-long bar.

Captured under the committed 5-symbol universe, which the strip states. GB-61 will flip
that to 20 and every backtest number on the page will move.

Note the two pills in row 2 differ: the radar and the fold chart read `results.csv`
(**folds 1-16**) while the cumulative equity and calendar read `report/daily_equity.csv`
(**folds 1-3**, a partial artefact). A card states the range of its own data, never the
page's.

### 02-forecast-path.png — the forecast against the entry threshold

Per-symbol close history, the calibrated entry threshold as a dashed rule, `NOW` marking
the last completed bar, and the forecast continuing past it as a dashed segment. The
caption above the row states it plainly: **the price line advances once per trading day** —
a decision is taken on the last completed bar and does not change within a session. Nothing
here is styled to imply a live feed, because there is not one.

### 03-decision-log-attribution.png — exact attribution, per channel

One decision expanded: the narrative in plain language, then `PER-CHANNEL CONTRIBUTION` as
signed bars summing to the forecast, with the cancellation line stating how much of the
gross view survived. Under FITS the model is univariate, so `close_logret` carries 100.0%
and the other four channels sit at 0.0% — the architecture visible in the panel.

Attribution is algebra, not approximation: across the 67 records behind these captures the
worst `|sum(per_channel) − forecast_total|` is **2.551e-09** against the 1e-5 contract.

### 04-spectral-fits.png — the spectral panel, under FITS

**FITS is not the deployed arm.** The strip in 01 reads `MODEL DLINEAR` and that is
correct: the deployed configuration runs DLinear, and this panel exists because fold 13 was
replayed under FITS specifically so the frequency decomposition could be shown at all.
Nothing in this frame describes what the live system is currently doing.

All 24 frequency contributors with the RIN mean as its own row, and the panel refusing to
be read as a dominant-cycle story: `NO SINGLE CYCLE CARRIES THIS FORECAST — STRONGEST IS
12.1% OF 24 CONTRIBUTORS`. Below it, gain and phase in days rather than radians.

**This is the blue ramp's remaining territory.** It encodes *which band* — a quantity, not
a direction — and appears only inside the attribution and spectral panels. No green or red
appears in this frame; tests hold both directions of that boundary.

**The panel had never rendered from a real decision record before 28 August 2026.**
`explain_spectral` was defined, exported and unit-tested, and had no caller anywhere in
`glassbox/`, so every decision record carried `per_frequency=None`.

### 05-reflow-1280.png — the grid at a narrow viewport

The same page at 1280×900. The session strip wraps to two lines, the three cards of row 2
stay side by side and equal-height, and nothing overflows. Equal height comes from
Streamlit's own column flex — `[data-testid="stColumn"] { display:flex }` with
`.gb-card { height:100% }` — which is why a card must be a single `st.markdown` call: a
second call inserts another wrapper and the height chain breaks at it.
