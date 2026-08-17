"""GB-24 acceptance: one command, no network, a table with the baseline beside it.

The expensive half - running the real universe over real folds - is exercised by the
command itself and reported in PROGRESS.md; what is tested here is everything that can go
wrong silently: a fetch where there should be none, a fold that stood aside vanishing from
the aggregate, a direction number printed without its reference.

The end-to-end test runs one fold on the real cache and is skipped when the cache is
absent, so a clean clone stays green without a network.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from glassbox import smoke_offline
from glassbox.backtest import metrics
from glassbox.config.loader import Config, load_config
from glassbox.engine.signal import Thresholds


@pytest.fixture
def cfg() -> Config:
    return load_config()


def cache_ready(cfg: Config, repo_root: Path) -> bool:
    cache = repo_root / cfg.data.cache_dir
    return all((cache / f"{symbol}.parquet").is_file() for symbol in cfg.universe)


def fake_run(
    fold: int, name: str, equity: list[float], stood_aside: bool
) -> smoke_offline.ArmRun:
    """An ArmRun without the pipeline behind it, for the reporting tests."""
    index = pd.date_range("2024-01-01", periods=len(equity), freq="B", tz="UTC")
    return smoke_offline.ArmRun(
        fold=fold,
        name=name,
        result=metrics.ArmResult(
            name=name, equity=pd.Series(equity, index=index), trades=()
        ),
        calibration=smoke_offline.Calibration(
            thresholds=Thresholds.never() if stood_aside else Thresholds(lower=0.01),
            val_sharpe=math.nan,
            val_trades=0,
            stood_aside=stood_aside,
            candidates=(),
        ),
        seconds=0.0,
    )


# ── it is offline, and it says so loudly ─────────────────────────────────────


def test_a_missing_cache_fails_loudly_and_names_the_symbols(
    cfg: Config, tmp_path: Path
) -> None:
    """The whole claim of this command is that it runs offline.

    `load_history` downloads a symbol whose cache file is absent, which would turn "the
    offline path works" into "the offline path works when yfinance is up" — and the failure
    would appear as a slow success on the one machine where it matters.
    """
    empty = replace(cfg, data=replace(cfg.data, cache_dir=str(tmp_path)))

    with pytest.raises(smoke_offline.SmokeError, match=cfg.universe[0]):
        smoke_offline.load_cached_bars(empty)


def test_the_missing_cache_message_says_how_to_fix_it(
    cfg: Config, tmp_path: Path
) -> None:
    """An error a user cannot act on is a traceback with better formatting."""
    empty = replace(cfg, data=replace(cfg.data, cache_dir=str(tmp_path)))

    with pytest.raises(smoke_offline.SmokeError, match="load_history"):
        smoke_offline.load_cached_bars(empty)


def test_a_missing_cache_exits_two_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, cfg: Config, tmp_path: Path
) -> None:
    """The CLI reports it as a failure, not as a crash."""
    monkeypatch.setattr(
        smoke_offline,
        "load_config",
        lambda: replace(cfg, data=replace(cfg.data, cache_dir=str(tmp_path))),
    )

    assert smoke_offline.main(["--folds", "1"]) == 2


# ── the arguments ────────────────────────────────────────────────────────────


def test_the_defaults_are_one_fold_of_dlinear() -> None:
    args = smoke_offline._parse_args([])

    assert args.folds == 1
    assert args.model == "dlinear"


def test_an_unknown_model_is_refused_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        smoke_offline._parse_args(["--model", "fits"])  # GB-41 adds it, not GB-24


def test_zero_folds_is_refused(cfg: Config) -> None:
    with pytest.raises(smoke_offline.SmokeError, match="at least 1"):
        smoke_offline.run(cfg, n_folds=0)


# ── ruling 1: a fold that stood aside is labelled, and it counts ─────────────


def test_a_fold_that_stood_aside_is_labelled_rather_than_blank() -> None:
    """A blank row is indistinguishable from a run that failed."""
    runs = [
        fake_run(1, "dlinear", [100.0, 101.0], stood_aside=False),
        fake_run(1, "persistence", [100.0, 100.0], stood_aside=True),
        fake_run(2, "dlinear", [100.0, 100.0], stood_aside=True),
        fake_run(2, "persistence", [100.0, 100.0], stood_aside=True),
    ]

    table = smoke_offline.fold_table(runs)

    aside = table[table["aside"] == "yes"]
    assert len(aside) == 3
    assert (aside["trades"] == 0).all()
    assert (aside["total_return"] == 0.0).all()


def test_a_fold_that_stood_aside_counts_in_the_return_mean() -> None:
    """Its return is a true zero, not a missing value.

    Averaging only the folds where the strategy chose to act is selection on the strategy's
    own decision, and it inflates the mean by exactly the folds it declined.
    """
    runs = [
        fake_run(1, "dlinear", [100.0, 110.0], stood_aside=False),  # +10%
        fake_run(2, "dlinear", [100.0, 100.0], stood_aside=True),  # 0%
    ]

    summary = smoke_offline.aggregate(runs, seconds=0.0)

    assert "+0.0500 over 2 folds" in summary  # not +0.1000 over 1
    assert "stood aside          : 1/2" in summary


def test_the_summary_says_how_many_folds_each_mean_rests_on() -> None:
    """The Sharpe mean and the return mean cover different folds, and must say so."""
    runs = [
        fake_run(1, "dlinear", [100.0, 110.0], stood_aside=False),
        fake_run(2, "dlinear", [100.0, 100.0], stood_aside=True),
    ]

    summary = smoke_offline.aggregate(runs, seconds=0.0)

    assert "over 2 folds, positive in 1" in summary
    assert "fold(s) where it is defined" in summary


def test_an_arm_with_no_defined_sharpe_reports_n_a_rather_than_a_number() -> None:
    """Persistence never trades, so it has no Sharpe on any fold. NaN, not zero."""
    runs = [fake_run(1, "persistence", [100.0, 100.0], stood_aside=True)]

    summary = smoke_offline.aggregate(runs, seconds=0.0)

    assert "sharpe               : mean       n/a over 0 fold(s)" in summary


# ── ruling 2: the direction reference is never printed alone ─────────────────


def test_the_direction_reference_sits_immediately_beside_the_accuracy() -> None:
    """0.51 is above a coin flip and below always-long, and the difference is the finding.

    Column adjacency is the mechanism: a reader scanning the table cannot pick up one
    number without the other in view.
    """
    columns = list(smoke_offline.FOLD_COLUMNS)

    assert columns[columns.index("direction") + 1] == "dir_ref"
    assert columns[columns.index("dir_ref") + 1] == "dir_delta"


def test_the_table_carries_the_reference_on_every_row() -> None:
    """Including the baseline's, whose own direction accuracy is NaN."""
    runs = [
        fake_run(1, "dlinear", [100.0, 101.0], stood_aside=False),
        fake_run(1, "persistence", [100.0, 100.0], stood_aside=True),
    ]

    table = smoke_offline.fold_table(runs)

    assert "dir_ref" in table.columns
    assert len(table) == 2


def test_the_legend_names_the_reference_and_rules_out_the_two_wrong_ones() -> None:
    """A reader who skips the spec must still not read it as 0.5 or as persistence."""
    summary = smoke_offline.aggregate(
        [fake_run(1, "dlinear", [100.0, 101.0], stood_aside=False)], seconds=0.0
    )

    assert "always-long bar" in summary
    assert "NOT 0.5" in summary
    assert "persistence" in summary


# ── the whole path, on the real cache ────────────────────────────────────────


def test_one_command_runs_the_whole_offline_path(cfg: Config, repo_root: Path) -> None:
    """The acceptance criterion: cache → features → folds → train → calibrate → backtest.

    One fold, because the suite must stay quick; the 16-fold run is reported in
    PROGRESS.md. Both arms appear, and the baseline is the persistence forecaster driven
    through the same code rather than a flat line drawn by the reporting layer.
    """
    if not cache_ready(cfg, repo_root):
        pytest.skip("no cached history; this test needs data_cache/")

    table, summary = smoke_offline.run(cfg, model="dlinear", n_folds=1)

    assert set(table["arm"]) == {"dlinear", "persistence"}
    assert list(table.columns) == list(smoke_offline.FOLD_COLUMNS)
    assert table["fold"].nunique() == 1
    assert "dlinear over 1 fold(s)" in summary
    assert "persistence over 1 fold(s)" in summary


def test_persistence_stands_aside_because_the_pipeline_gave_it_nothing_to_trade(
    cfg: Config, repo_root: Path
) -> None:
    """The baseline forecasts zero, so no band can fire — and this is the honest route to
    a flat curve. If the reporting layer drew it instead, the command would not be proving
    that the pipeline runs end to end for both arms."""
    if not cache_ready(cfg, repo_root):
        pytest.skip("no cached history; this test needs data_cache/")

    table, _ = smoke_offline.run(cfg, model="persistence", n_folds=1)

    assert set(table["arm"]) == {"persistence"}  # deduped: the arm IS the baseline
    row = table.iloc[0]
    assert row["aside"] == "yes"
    assert row["trades"] == 0
    assert math.isnan(row["direction"])  # a zero forecast expresses no direction
    assert not math.isnan(row["dir_ref"])  # the bar is a property of the fold


def test_the_sizer_reads_its_fraction_from_the_config(cfg: Config) -> None:
    """No magic number in the placeholder GB-21 will replace."""
    signal = None  # unused by the sizer, which is the point

    notional = smoke_offline.size_position(signal, 100_000.0, 0.0, cfg)

    assert notional == pytest.approx(100_000.0 * cfg.risk.max_position_pct)
