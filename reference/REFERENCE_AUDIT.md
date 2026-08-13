# REFERENCE_AUDIT.md — Instructions for Claude Code

`reference/algotrading_project-main/` is an existing LTSF-Linear trading project
brought in as a **read-only reference**. It is not part of the build.

## Hard rules

1. **Never import from `reference/`.** Nothing in `glassbox/` may depend on it. The
   import-linter contract (GB-3) must treat `reference/` as excluded from the package.
2. **Never copy a file wholesale.** Port the specific logic named below, adapted to
   the frozen contracts in `docs/GLASSBOX_PROJECT_SPEC.md` §4.
3. **Anything ported must pass our tests**, not the reference project's assumptions.

## Approved for reuse

| Source | Target task | What to take |
|---|---|---|
| `models/DLinear.py` | GB-13 | The `series_decomp` / `moving_avg` blocks and the two linear heads. Wrap in the `Forecaster` protocol. Keep weights accessible by channel name for GB-30. |
| `evaluation.py` | GB-19 | `calc_total_return`, `calc_annualized_return`, `calc_annualized_sharpe`, `calc_max_drawdown`, `calc_sortino`, `calc_downside_deviation`. These are correct. |
| `backtesting.py`, `models_backtest.py` | GB-18 | The event-loop structure and the `Position` / `ActionType` / `PositionType` enums. Structure only — see the bug below. |
| `utils/tools.py` | GB-15 | `EarlyStopping`, `adjust_learning_rate`. |
| `data_provider/data_loader.py` | GB-9 | **Reference only, do not port.** Its scaler-on-train-only pattern is correct and worth matching; its windowing is replaced by `builder.py`. |

## Known bug — fix on port

`strategies.py` inverts all four stop-loss and take-profit comparisons:

- long stop-loss uses `row['Low'] >= stop` — must be `<=`
- long take-profit uses `row['High'] <= target` — must be `>=`
- short stop-loss uses `row['High'] <= stop` — must be `>=`
- short take-profit uses `row['Low'] >= target` — must be `<=`

When implementing GB-18, write the correct comparisons and cover each of the four
cases with an explicit test. Do not copy this logic.

## Do not reuse

- **`utils/metrics.py`** — `SHARP()` returns a hardcoded `12`; `SHARP2` and `SHARP5`
  are each defined twice with the second shadowing the first; one variant contains
  `pred = true + 0.01`, overwriting the prediction with the ground truth. Write
  `backtest/metrics.py` from scratch in GB-19.
- **The single train/val/test split.** This project requires walk-forward (GB-17).

## Not a bug

`moving_avg` in `models/DLinear.py` pads both ends, making the trend a centred moving
average. This is correct here: the decomposition runs inside the input window
`[t-L+1, t]`, all of which precedes the forecast, so no information after `t` is used.

Centring is forbidden in `features/indicators.py` and `features/wavelets.py`, which
compute along the full series. GB-8 and GB-10 enforce that.
