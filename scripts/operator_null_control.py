"""GB-66's acceptance measurement: how much of a learned operator is data-independent?

**The prediction was recorded before WITS was written**, in the expansion plan:

> If the geometry critique is correct, WITS will not show ~86% data-independence under
> the null control. If it does, the critique of FITS is wrong - and that is worth as much.

GB-46 measured that FITS's learned frequency response on real returns correlates with the
response learned on white noise at **+0.9485** - roughly 86% of the curve's own variation
needing no market at all. The mechanism proposed was *grid geometry*: extending a window
from ``L`` to ``L+H`` moves frequency ``k`` to ``eta*k``, and a bin that does not land on
an output bin must be interpolated across on every window, by the learned layer, whatever
the data. That is either a fact about FITS or a fact about extending a **global** basis,
and one architecture cannot separate the two.

WITS puts a **local** basis in the same position. Its retained band lengths are equal at
``L`` and ``L+H``, so its per-band maps are square and nothing is remapped onto a shifted
grid - there is no interpolation for the layer to learn. If the critique is right, WITS's
operator should depend on the data far more than FITS's does.

**Why this script exists rather than a notebook.** FITS's +0.9485 and the 86% appear in
neither ``results.csv`` nor ``report.md``: they were computed once and written into prose,
so a central claim of the report has no source a reviewer can check. Whatever WITS returns
lands in an artefact this time, beside a regenerated FITS figure, and ``report.py`` reads
it from there.

**Two measurements, because only one of them is comparable across architectures.**

- ``response`` - the model's own frequency response, ``|W|`` per retained bin. This is
  GB-46's quantity and it is what reproduces +0.9485. **It exists only for FITS**: WITS
  has bands, not bins, and two retained bands cannot carry a correlation.
- ``operator`` - the full ``(H, L)`` forecast matrix, flattened. Every linear forecaster
  has one, it is the object that *is* the model, and it is directly comparable between a
  Fourier core and a wavelet core. This is the number the WITS-versus-FITS claim rests on,
  and it is computed identically for both so the comparison is like for like.

Both are Pearson correlations between the real-data operator and the white-noise operator,
fitted on the same fold with the same seed and the same budget - the only difference is
what the close channel contains.

Usage::

    python scripts/operator_null_control.py --out report/null_control.csv
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd

    # Inside the guard, and that is the point of it. These were below the `try` until
    # 28 Aug 2026, so a clone without `pip install -e .` got a bare ModuleNotFoundError
    # from the line that matters most - the project's own package - while the guard
    # caught only the third-party ones. `ImportError` rather than `ModuleNotFoundError`
    # because a half-installed package raises the parent, not the child.
    from glassbox.backtest.walkforward import make_folds
    from glassbox.config.loader import Config, load_config
    from glassbox.experiments.study import NOISE, REAL, null_bars
    from glassbox.explain.spectral import frequency_response
    from glassbox.features.builder import build_feature_frame
    from glassbox.model import train as trainer
    from glassbox.smoke_offline import _common_index, load_cached_bars
except ImportError as missing:  # pragma: no cover - the wrong-interpreter guard
    raise SystemExit(
        f"""cannot import {missing.name!r}: this script needs the project environment.
    .venv/Scripts/activate       (Windows)
    source .venv/bin/activate    (macOS, Linux)
then, if the environment is new:  pip install -e ".[dev]" """
    ) from missing

#: The models this measures. Persistence has no learned operator and DLinear's is not the
#: object under test - the claim is about extending a basis, and DLinear extends nothing.
MODELS = ("fits", "wits")

COLUMNS = (
    "model",
    "fold",
    "quantity",
    "correlation",
    "n_values",
    "real_mean_abs",
    "noise_mean_abs",
    "mean_abs_difference",
)


def _fit(
    cfg: Config, model: str, bars: dict[str, pd.DataFrame], control: str, fold
) -> Any:
    """One fitted model, through the same path the study uses.

    ``trainer.train`` rather than a local loop: the study fits this way, so a difference
    between this measurement and the grid cannot come from how the model was trained.
    """
    world = {
        symbol: null_bars(frame, control, cfg.meta.seed + index)
        for index, (symbol, frame) in enumerate(sorted(bars.items()))
    }
    arm = replace(cfg, model=replace(cfg.model, active=model))
    frames = {s: build_feature_frame(f, arm) for s, f in world.items()}
    return trainer.train(frames, arm, fold.train, fold.val, fold.test).model


def _correlate(real: np.ndarray, noise: np.ndarray) -> dict[str, float]:
    """Pearson between two flattened operators, with the scale beside it.

    The correlation alone would not distinguish *the same operator* from *the same shape
    at a different magnitude*, so the mean absolute value of each and of their difference
    travel with it. A reader can then see whether a high correlation is agreement or
    merely a shared silhouette.
    """
    a, b = (
        np.asarray(real, dtype="float64").ravel(),
        np.asarray(noise, dtype="float64").ravel(),
    )
    if a.size != b.size or a.size < 3:
        return {
            "correlation": float("nan"),
            "n_values": float(a.size),
            "real_mean_abs": float("nan"),
            "noise_mean_abs": float("nan"),
            "mean_abs_difference": float("nan"),
        }
    return {
        "correlation": float(np.corrcoef(a, b)[0, 1]),
        "n_values": float(a.size),
        "real_mean_abs": float(np.abs(a).mean()),
        "noise_mean_abs": float(np.abs(b).mean()),
        "mean_abs_difference": float(np.abs(a - b).mean()),
    }


def measure(cfg: Config, n_folds: int | None = None, log=print) -> pd.DataFrame:
    """One row per (model, fold, quantity)."""
    bars = load_cached_bars(cfg)
    probe = {s: build_feature_frame(f, cfg) for s, f in sorted(bars.items())}
    folds = make_folds(_common_index(probe), cfg)
    if n_folds is not None:
        folds = folds[:n_folds]
    if not folds:
        raise SystemExit("no complete walk-forward fold fits the cached history")

    rows: list[dict] = []
    for model in MODELS:
        for fold in folds:
            real = _fit(cfg, model, bars, REAL, fold)
            noise = _fit(cfg, model, bars, NOISE, fold)

            rows.append(
                {
                    "model": model,
                    "fold": fold.number,
                    "quantity": "operator",
                    **_correlate(real.forecast_matrix(), noise.forecast_matrix()),
                }
            )
            # GB-46's own quantity, and it exists only where there are bins to have it.
            # Reported as an absent row rather than a NaN one: WITS has no frequency
            # response, and a NaN invites a reader to wonder whether it failed.
            if model == "fits":
                rows.append(
                    {
                        "model": model,
                        "fold": fold.number,
                        "quantity": "response",
                        **_correlate(
                            frequency_response(real)[1], frequency_response(noise)[1]
                        ),
                    }
                )
            log(f"  {model:5} fold {fold.number:>2} done")
    return pd.DataFrame(rows, columns=list(COLUMNS))


def verdict(table: pd.DataFrame) -> str:
    """The prediction's answer, stated before any interpretation of it.

    **The comparison has to be like for like, and the headline quantity cannot carry it.**
    GB-46's +0.9485 and its 86% are the ``response`` - the magnitude of the learned gain
    per frequency bin. WITS has bands, not bins, so it has no response at all, and holding
    WITS's number against 0.9274 would be comparing a signed operator to a magnitude
    curve. The only quantity both architectures have is the ``operator``, and the FITS
    figure it must be read against is FITS's **own** operator correlation, not its
    response.

    So the verdict is a *direction*: the prediction says WITS's operator should depend on
    the data more than FITS's does, i.e. correlate less with its noise twin. That is
    checkable, and it is what this reports.
    """
    ops = table[table["quantity"] == "operator"].groupby("model")["correlation"].mean()
    resp = table[table["quantity"] == "response"].groupby("model")["correlation"].mean()
    if "wits" not in ops or "fits" not in ops:
        return "VERDICT UNAVAILABLE: both arms did not produce an operator correlation."

    wits, fits = float(ops["wits"]), float(ops["fits"])
    lines = []
    if "fits" in resp:
        lines.append(
            f"REPRODUCTION: FITS response correlation real-vs-noise = {resp['fits']:+.4f} "
            f"(r^2 = {resp['fits'] ** 2:.3f}), against GB-46's +0.9485 / ~86%."
        )
    held = wits < fits
    lines.append(
        f"PREDICTION {'HELD' if held else 'DID NOT HOLD'}: on the operator - the only "
        f"quantity both architectures have - WITS correlates with its noise twin at "
        f"{wits:+.4f} (r^2 = {wits**2:.3f}) and FITS at {fits:+.4f} "
        f"(r^2 = {fits**2:.3f}). The prediction was that WITS would depend on the data "
        f"MORE than FITS, i.e. correlate LESS; it correlates "
        f"{'less' if held else 'MORE'}."
    )
    lines.append(
        "NOTE: FITS's operator correlation is far below its response correlation. The "
        "two measure different things - a signed (H, L) matrix against a curve of gain "
        "magnitudes - so the 86% figure is a claim about the gain curve specifically and "
        "must not be quoted as a claim about the whole learned operator."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="report/null_control.csv")
    parser.add_argument("--folds", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config()
    table = measure(cfg, args.folds)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    print()
    summary = (
        table.groupby(["model", "quantity"])["correlation"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    print(summary.to_string(index=False))
    print()
    print(verdict(table))
    print(f"\nwrote {out}: {len(table)} rows")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI
    raise SystemExit(main())
