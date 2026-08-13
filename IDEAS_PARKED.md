# IDEAS PARKED

Ideas that are **not** in scope for this project. Recorded so they are visibly a
choice rather than an omission, and so they do not leak into the codebase mid-sprint.

**Rule: nothing here is implemented before Gate 3. No exceptions.**

---

## Regime Guard — reconstruction head as out-of-distribution detector

FITS uses the same architecture for anomaly detection: instead of forecasting forward,
reconstruct the current window and treat reconstruction error as an anomaly score.
Applied to trading, a spike means the market's spectral structure has diverged from
the training distribution — the model is outside what it knows. The bot would reduce
exposure and surface: "Unusual market state, reconstruction error at the 97th
percentile. New entries paused."

Estimated cost: one week. Excluded solely for schedule. Written up in the report
as declared future work.

## Adaptive per-symbol cutoff frequency

Choose the FITS cutoff per symbol from measured spectral content rather than using
one global value. Second-order optimisation.

## SWT / MODWT shift-invariant wavelet comparison

The undecimated transform avoids the shift-variance of the DWT. A third arm in the
feature axis. No schedule room.

## Cross-sectional ranking objective

Train directly on relative ranking across the universe rather than on absolute
return, since ranking is an easier target than level prediction. Interesting;
changes the loss and the whole evaluation frame.

## Sentiment / news channel

Fragile external dependency, no room in 8 weeks.

## Intraday resolution

Data cost, microstructure effects, roughly 5x the complexity.

---

## New ideas

_Add below. Date, one paragraph, and an honest cost estimate._
