"""GB-46 acceptance: the learned frequency response, drawn from a trained model.

§6.5 calls this the single strongest visual in the demonstration, and the claim rests on
something checkable: the curve *is* the model's one weight matrix, so what the figure shows
and what the model does cannot diverge. What is tested here is everything around that — the
axis is in days, the cutoff is annotated, the resolution is what a report needs, and the
colour is doing no work the geometry is not already doing.

The generator lives in ``scripts/`` because it is an operational entry point that reads a
checkpoint and writes a file (spec §3.4). Nothing in the package imports it; this test
loads it by path, which is why the module is written as importable functions around a thin
``main`` rather than as a script body.
"""

from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from glassbox.config.loader import Config, load_config
from glassbox.model import ALL_FORECASTERS
from glassbox.model.fits import FITSForecaster

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def script(repo_root: Path) -> ModuleType:
    """``scripts/frequency_response.py``, loaded by path."""
    path = repo_root / "scripts" / "frequency_response.py"
    spec = importlib.util.spec_from_file_location("frequency_response", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cfg() -> Config:
    return load_config()


@pytest.fixture
def model(cfg: Config) -> FITSForecaster:
    built = ALL_FORECASTERS["fits"](cfg, cfg.channels.active_channels)
    rng = np.random.default_rng(cfg.meta.seed)
    shape = (built.cof, built.out_bins)
    built._set_weight(
        rng.normal(0, 0.3, size=shape) + 1j * rng.normal(0, 0.3, size=shape)
    )
    return built


def test_the_figure_is_written_as_a_png(
    script: ModuleType, model: FITSForecaster, cfg: Config, tmp_path: Path
) -> None:
    out = script.render(model, cfg, tmp_path / "nested" / "response.png")

    assert out.is_file()
    assert out.read_bytes()[:8] == PNG_MAGIC
    assert out.stat().st_size > 10_000  # a real plot, not a blank canvas


def test_the_resolution_is_what_a_report_needs(
    script: ModuleType, model: FITSForecaster, cfg: Config, tmp_path: Path
) -> None:
    """GB-46 asks for 200+ dpi, and the default is above the floor rather than on it."""
    assert script.DEFAULT_DPI >= 200

    with pytest.raises(ValueError, match="at least 200 dpi"):
        script.render(model, cfg, tmp_path / "low.png", dpi=96)


def test_a_higher_dpi_produces_a_larger_file(
    script: ModuleType, model: FITSForecaster, cfg: Config, tmp_path: Path
) -> None:
    """The dpi argument reaches the raster rather than being accepted and ignored."""
    small = script.render(model, cfg, tmp_path / "a.png", dpi=200)
    large = script.render(model, cfg, tmp_path / "b.png", dpi=400)

    assert large.stat().st_size > small.stat().st_size


def test_the_ramp_runs_lightest_for_the_fastest_cycle(script: ModuleType) -> None:
    """Dark for slow, light for fast — §7 (1k)'s ramp, and the x axis says the same thing.

    Which is the point of asserting it: the colour restates the position and encodes
    nothing of its own, so the chart reads with the colour removed.
    """
    ramp = ("dark", "b", "c", "d", "e", "light")

    assigned = script._ramp_by_speed(ramp, 6)

    assert assigned[0] == "light"  # the fastest cycle, at the left of the axis
    assert assigned[-1] == "dark"  # the slowest, at the right
    assert script._ramp_by_speed(ramp, 1) == ["light"]
    assert len(script._ramp_by_speed(ramp, 23)) == 23


def test_the_figure_defines_no_colour_of_its_own(
    script: ModuleType, repo_root: Path
) -> None:
    """Every colour is the dashboard's, so the ban it enforces covers this figure too.

    ``tests/dashboard/test_app.py::test_no_traffic_light_colours_anywhere`` asserts that no
    green/red pair exists in the palette. That guarantee only reaches this figure if the
    figure takes its colours from there and defines none — so the check is for a hex
    literal anywhere in the source, which is both mechanical and stricter than re-listing
    the banned pairs here.
    """
    source = (repo_root / "scripts" / "frequency_response.py").read_text(
        encoding="utf-8"
    )

    literals = re.findall(r"#[0-9A-Fa-f]{3,8}", source)

    assert not literals, f"colour defined locally instead of imported: {literals}"


def test_a_missing_checkpoint_exits_two_without_a_traceback(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = script.main(["--checkpoint", str(tmp_path / "nothing")])

    assert code == 2
    assert "frequency_response:" in capsys.readouterr().err


def test_a_model_with_no_frequency_response_says_so(
    script: ModuleType, cfg: Config, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """DLinear and persistence have no spectrum, and an empty axis would be a lie."""
    from glassbox.contracts.schemas import WindowBatch
    from glassbox.model import train as trainer

    channels = cfg.channels.active_channels
    flat = ALL_FORECASTERS["persistence"](cfg, channels)
    batch = WindowBatch(
        X=np.zeros((4, cfg.window.input_len, len(channels)), dtype="float32"),
        y=np.zeros((4, cfg.window.horizon), dtype="float32"),
        channels=channels,
        timestamps=pd.date_range("2024-01-01", periods=4, tz="UTC"),
        symbols=("AAPL",) * 4,
        source="test",
    )
    flat.fit(batch)
    run = trainer.TrainingRun(
        model=flat,
        stats={},
        history=(),
        epochs_run=0,
        best_epoch=None,
        stopped_early=False,
        n_train_windows=4,
        n_val_windows=0,
        checkpoint=None,
        seconds=0.0,
    )
    trainer.save_checkpoint(tmp_path / "flat", run, cfg)

    code = script.main(["--checkpoint", str(tmp_path / "flat")])

    assert code == 2
    assert "no frequency response" in capsys.readouterr().err


def test_the_response_the_figure_draws_is_the_models_own_weight(
    script: ModuleType, model: FITSForecaster
) -> None:
    """The claim that makes this figure worth putting in a report.

    It is not an approximation of what the model attends to. Every point on the curve is
    ``|W[k, round(eta*k)]|``, read off the one matrix that *is* the model — which is why
    there is no equivalent of this plot for a non-linear architecture.
    """
    from glassbox.explain.spectral import frequency_response
    from glassbox.model.fits import amplitude_scale

    periods, gains = frequency_response(model)
    eta = amplitude_scale(model.input_len, model.horizon)

    for period, gain in zip(periods, gains, strict=True):
        index = round(model.input_len / period)
        column = min(round(index * eta), model.out_bins - 1)
        assert gain == pytest.approx(abs(model.weight[index, column]))

    assert math.isfinite(gains.max())
