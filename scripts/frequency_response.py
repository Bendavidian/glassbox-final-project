"""GB-46: the learned frequency response — what the bot learned to listen to.

``|W|`` against period in **days**, read off a trained checkpoint. Spec §6.5 calls it the
single strongest visual in the demonstration, and the reason is that there is no
equivalent for a non-linear model: this is not an approximation of what the model attends
to, it *is* the model's one weight matrix, plotted.

Usage::

    python scripts/frequency_response.py                        # checkpoints/live
    python scripts/frequency_response.py --checkpoint DIR
    python scripts/frequency_response.py --out figures/fits.png --dpi 300

Exit codes: 0 the figure was written, 2 the checkpoint is missing or holds a model with no
frequency response — persistence and DLinear have none, and saying so beats drawing an
empty axis.

**The axis is in days, not bin index**, which is the whole point: "the 17-day cycle" means
something to a reader and "bin 7" does not. **The cutoff is annotated rather than implied**
— everything faster than ``fits.cutoff_period_days`` was removed by the low-pass and has no
weight at all, and a curve that simply stopped would look like a model that lost interest
rather than one that was told to.

**The palette is imported, not restated.** `dashboard/app.py` holds it, and a second copy
here would drift the first time either changed. The figure obeys the same rule the
dashboard does (§7 (1k)): the information is carried by position and shape, so it reads in
greyscale, and no green/red pair appears anywhere in it.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:
    import matplotlib  # noqa: F401

    from glassbox.config.loader import DEFAULT_SETTINGS_PATH  # noqa: F401
except ImportError:  # pragma: no cover - the wrong interpreter
    print(
        "frequency_response: run this with the project's interpreter, e.g."
        "\n  .venv/Scripts/python.exe scripts/frequency_response.py",
        file=sys.stderr,
    )
    raise SystemExit(3) from None

DEFAULT_CHECKPOINT = "checkpoints/live"
DEFAULT_OUT = "figures/frequency_response.png"

# 220, not 200: §GB-46 asks for 200+, and a figure that lands exactly on a floor is one
# nobody can round down. Large enough to survive being dropped into a report at half width.
DEFAULT_DPI = 220


def render(model, cfg, out: Path, dpi: int = DEFAULT_DPI) -> Path:
    """Draw the response of ``model`` to ``out`` and return the path written.

    Args:
        model: A fitted forecaster exposing ``frequency_response`` through
            ``explain.spectral`` — in practice FITS.
        cfg: Resolved configuration, for the cutoff period and the window length.
        out: Where to write. Parent directories are created.
        dpi: Raster resolution. GB-46 requires at least 200.

    Raises:
        ValueError: ``dpi`` is below 200, or the model exposes no frequency response.
    """
    import matplotlib

    matplotlib.use("Agg")  # no display, and none wanted: this writes a file
    import matplotlib.pyplot as plt

    from glassbox.dashboard.app import HAIRLINE, INK, MUTED, ORANGE, PANEL, PAPER, RAMP
    from glassbox.explain.spectral import frequency_response

    if dpi < 200:
        raise ValueError(f"GB-46 asks for at least 200 dpi, got {dpi}")

    periods, gains = frequency_response(model)
    cutoff = cfg.fits.cutoff_period_days

    figure, axes = plt.subplots(figsize=(9.0, 5.0))
    figure.patch.set_facecolor(INK)
    axes.set_facecolor(PANEL)

    # Everything left of the cutoff was removed by the low-pass. Shading it says the
    # absence is a decision rather than a gap in the data.
    axes.axvspan(1.0, cutoff, color=INK, alpha=0.85, zorder=0)
    axes.axvline(cutoff, color=ORANGE, linewidth=1.0, zorder=3)
    axes.text(
        cutoff * 1.04,
        max(gains) * 0.97 if len(gains) else 1.0,
        f"LOW-PASS CUTOFF  {cutoff} DAYS",
        color=ORANGE,
        fontsize=8,
        va="top",
        ha="left",
        family="monospace",
    )

    # The ramp encodes a quantity — dark for slow, light for fast — which is the axis
    # itself, so the colour restates the position and carries no information of its own.
    # That is the point: remove it and the chart still reads.
    axes.plot(periods, gains, color=RAMP[3], linewidth=1.2, zorder=4)
    axes.scatter(
        periods,
        gains,
        c=_ramp_by_speed(RAMP, len(periods)),
        s=26,
        zorder=5,
        edgecolor=INK,
        linewidth=0.5,
    )

    axes.set_xscale("log")
    axes.set_xlim(cutoff * 0.8, cfg.window.input_len * 1.15)
    axes.set_xlabel(
        "CYCLE PERIOD  (TRADING DAYS)",
        color=MUTED,
        fontsize=8,
        family="monospace",
        labelpad=10,
    )
    axes.set_ylabel(
        "GAIN  |W|", color=MUTED, fontsize=8, family="monospace", labelpad=10
    )
    axes.set_xticks([cutoff, 10, 20, 40, 60, cfg.window.input_len])
    axes.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}")
    )
    axes.tick_params(colors=MUTED, labelsize=8, which="both")
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_family("monospace")
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        axes.spines[side].set_color(HAIRLINE)
    axes.grid(True, color=HAIRLINE, linewidth=0.4, alpha=0.6, zorder=1)

    dead = "1,200 ALLOCATED / 1,150 EFFECTIVE" if model.name == "fits" else ""
    axes.set_title(
        f"WHAT THE MODEL LEARNED TO LISTEN TO   ·   {model.name.upper()}   ·   {dead}",
        color=PAPER,
        fontsize=9,
        family="monospace",
        loc="left",
        pad=16,
    )

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=dpi, facecolor=INK, bbox_inches="tight")
    plt.close(figure)
    return out


def _ramp_by_speed(ramp: tuple[str, ...], count: int) -> list[str]:
    """One colour per point, **lightest for the fastest cycle**.

    The periods run slow to fast right to left, and the project's one data colour family is
    a ramp meaning dark-for-slow (§7 (1k)). So the assignment restates the x position and
    encodes nothing the axis does not already say — which is exactly the test of whether
    colour is doing work: remove it and this chart still reads.
    """
    if count <= 1:
        return [ramp[-1]] * count
    steps = len(ramp) - 1
    return [ramp[steps - round(index * steps / (count - 1))] for index in range(count)]


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code rather than raising."""
    from glassbox.config.loader import load_config
    from glassbox.model.predict import load_predictor

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    args = parser.parse_args(argv)

    cfg = load_config()
    try:
        predictor = load_predictor(args.checkpoint, cfg)
    except (FileNotFoundError, ValueError) as failure:
        print(f"frequency_response: {failure}", file=sys.stderr)
        return 2

    model = predictor.model
    if not hasattr(model, "frequency_matrices"):
        print(
            f"frequency_response: {model.name!r} has no frequency response to draw; "
            "only a spectral model does. Train one with model.active: fits",
            file=sys.stderr,
        )
        return 2

    written = render(model, cfg, Path(args.out), dpi=args.dpi)
    from glassbox.explain.spectral import frequency_response

    periods, gains = frequency_response(model)
    loudest = periods[int(gains.argmax())] if len(gains) else math.nan
    print(f"wrote {written} at {args.dpi} dpi")
    print(
        f"loudest retained cycle: {loudest:.1f} days, gain {gains.max():.4f}; "
        f"quietest {periods[int(gains.argmin())]:.1f} days, gain {gains.min():.4f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
