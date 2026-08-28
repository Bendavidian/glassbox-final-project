# Dashboard screenshots — captions for the architecture report

Real captures at 1920×1080 through headless Chromium against a running Streamlit
instance. Not mock-ups, and not full-page strips: each frame is what a reader sees.

All four were taken against `checkpoints/fits-demo`, a replay of **fold 13** written by
`smoke_offline --model fits --prepare-replay 13` and driven by
`replay --model fits`. The dashboard was launched as:

    streamlit run glassbox/dashboard/app.py -- \
        --state-dir checkpoints/fits-demo --source replay:fold-13

---

### 01-overview.png — the instrument panel

The masthead, the cycle countdown, the session equity curve and the positions table.

`BOUND CHECKPOINTS\FITS-DEMO · SHOWING REPLAY:FOLD-13` is the GB-63 fix: on the night of
GATE 2's execution rehearsal the panel was bound to `checkpoints/live` while the rehearsal
wrote to `checkpoints/rehearsal`, so its Co-Pilot queue rendered empty and correct, and
criterion 2 had to be satisfied through `answer_pending` instead. An empty queue and no
recommendations are indistinguishable unless the panel says which directory it is reading.

The equity curve is drawn in the status colour of its own sign against the session's first
reading — here jade, `▲ +0.02` — with the opening level as a dashed rule so the sign is
readable as geometry and not only as colour. `NEXT CYCLE` is measured from the loop's own
last write to `book.json`, not from the panel's refresh timer; it reads `0S` here because
a replay directory has no live loop persisting to it, which is the honest rendering rather
than a full bar promising a cycle that is not coming.

Both positions read `QUARANTINED` with three em dashes. They are the two rehearsal
positions still open at the broker, seen from a state directory that did not open them:
the system knows no entry basis, no PnL derived from one, and no stop it set. Each of the
three is a thing it correctly declines to claim.

### 02-forecast-path.png — the forecast against the entry threshold

Per-symbol close history in the data ramp, the calibrated entry threshold as a dashed
chrome rule, `NOW` marking the last completed bar, and the forecast path continuing past
it as a dashed segment. The price line advances **once per trading day**: a decision is
taken on the last completed bar and does not change within a session.

### 03-decision-log-attribution.png — exact attribution, per channel

The decision log with one record expanded: the narrative in plain language, then
`PER-CHANNEL CONTRIBUTION` as signed bars summing to the forecast, with the cancellation
line stating how much of the gross view survived. Under FITS the model is univariate, so
`close_logret` carries 100.0% and the other four channels are flat at 0.0% — which is the
architecture being visible in the panel rather than an artefact of the capture.

Attribution is algebra, not approximation: across the 67 records behind these screenshots
the worst `|sum(per_channel) − forecast_total|` is **2.551e-09** against the 1e-5 contract.

### 04-spectral-fits.png — the spectral panel, under FITS

**FITS is not the deployed arm.** The masthead in 01 reads `SYSTEM DLINEAR`, and it is
correct: the deployed configuration runs DLinear, and this panel is produced by replaying
fold 13 under FITS specifically so that the frequency decomposition can be shown at all.
Nothing in this frame describes what the live system is currently doing.

The panel shows all 24 frequency contributors with the RIN mean as its own row, and states
`NO SINGLE CYCLE CARRIES THIS FORECAST — STRONGEST IS 12.1% OF 24 CONTRIBUTORS` — the
decomposition refusing to be read as a dominant-cycle story. `BIN 0 MULTIPLIES ZERO — RIN
REMOVES THE WINDOW MEAN BEFORE THE TRANSFORM` explains the one row that can never move.
Below it, gain and phase for the strongest cycles, in days rather than radians.

Everything here is drawn in the blue data ramp with chrome labels. No status colour
appears in this frame, which is the palette rule: inside a chart the ramp already owns
meaning, and a third colour family would make a reader ask what green encodes on an axis
that is spending colour on frequency.

**This panel had never rendered from a real decision record before 28 August 2026.**
`explain_spectral` existed, was exported and was unit-tested, and had no caller anywhere in
`glassbox/` — so every decision record carried `per_frequency=None` and the panel drew
nothing under any model. It is wired into the decision path now, structurally: a model that
cannot enumerate its own frequency maps takes the channel view unchanged, so the change is
inert for DLinear and persistence.
