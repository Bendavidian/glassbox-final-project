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

import numpy as np
import pandas as pd
import pytest

from glassbox import smoke_offline
from glassbox.backtest import metrics
from glassbox.config.loader import Config, load_config
from glassbox.contracts.schemas import ChannelStats, WindowBatch
from glassbox.engine import risk
from glassbox.engine.signal import Thresholds
from glassbox.model import ALL_FORECASTERS


@pytest.fixture
def cfg() -> Config:
    return load_config()


def cache_ready(cfg: Config, repo_root: Path) -> bool:
    cache = repo_root / cfg.data.cache_dir
    return all((cache / f"{symbol}.parquet").is_file() for symbol in cfg.universe)


def fake_run(
    fold: int, name: str, equity: list[float], stood_aside: bool, forecasts: bool = True
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
        forecasts=forecasts,
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


def test_the_default_arm_is_whatever_the_configuration_names() -> None:
    """The CLI defaults to **no** arm, so ``model.active`` decides (GB-44).

    It used to default to the literal ``"dlinear"``, which meant setting
    ``model.active: fits`` changed the live checkpoint — ``prepare_live`` reads the config
    — and changed **nothing** about the study numbers this command produces. A switch that
    reaches one half of the system is worse than one reaching neither, because the two
    halves then disagree without saying so.
    """
    args = smoke_offline._parse_args([])

    assert args.folds == 1
    assert args.model is None


def test_every_registered_model_is_selectable_from_the_command_line() -> None:
    """The third copy of the registry, and the one that was stale (GB-44).

    ``VALID_MODELS`` in the config layer and ``ALL_FORECASTERS`` in the model layer must
    agree, and ``tests/model/test_fits_integration.py`` asserts that. This command held a
    **third** list, an argparse ``choices`` literal, and it still read
    ``(persistence, dlinear)`` after GB-41 registered FITS — so the one runner that
    produces every study number could not select the model the study is about. The choices
    now come from the registry, so there is no list left to go stale.
    """
    for name in ALL_FORECASTERS:
        assert smoke_offline._parse_args(["--model", name]).model == name


def test_an_unknown_model_is_refused_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        smoke_offline._parse_args(["--model", "no-such-model"])


def test_an_unknown_model_is_refused_by_run_naming_the_registry(cfg: Config) -> None:
    """``run`` is a second door in — the sweeps and the tests call it directly."""
    with pytest.raises(smoke_offline.SmokeError, match="the registry holds"):
        smoke_offline.run(cfg, model="no-such-model", n_folds=1)


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
    assert columns[columns.index("dir_ref") + 1] == "dir_vs_long"


def test_every_delta_column_names_its_own_reference() -> None:
    """Three references in one table (§7.3), so no column may leave its own unstated.

    A bare `delta` column would be read against whichever baseline the reader had in mind,
    and two of the three would be wrong.
    """
    deltas = [c for c in smoke_offline.FOLD_COLUMNS if "_vs_" in c]

    assert deltas == ["dir_vs_long", "mae_vs_pers", "ret_vs_bh", "sharpe_vs_bh"]


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

    assert set(table["arm"]) == {"dlinear", "persistence", smoke_offline.BUY_AND_HOLD}
    assert list(table.columns) == list(smoke_offline.FOLD_COLUMNS)
    assert table["fold"].nunique() == 1
    assert "dlinear over 1 fold(s)" in summary
    assert "persistence over 1 fold(s)" in summary


def test_the_configured_model_reaches_the_runner_end_to_end(
    cfg: Config, repo_root: Path
) -> None:
    """GB-44's claim at system level: ``model.active: fits`` and **no other change**.

    Not a parser test. This is cache → features → folds → train → calibrate → backtest →
    metrics with FITS in it, selected by configuration alone and with no ``model=``
    argument anywhere, which is the only form of the claim that is worth making.
    """
    if not cache_ready(cfg, repo_root):
        pytest.skip("no cached history; this test needs data_cache/")

    spectral = replace(cfg, model=replace(cfg.model, active="fits"))

    table, summary = smoke_offline.run(spectral, n_folds=1)

    assert set(table["arm"]) == {"fits", "persistence", smoke_offline.BUY_AND_HOLD}
    assert "fits over 1 fold(s)" in summary


def test_persistence_stands_aside_because_the_pipeline_gave_it_nothing_to_trade(
    cfg: Config, repo_root: Path
) -> None:
    """The baseline forecasts zero, so no band can fire — and this is the honest route to
    a flat curve. If the reporting layer drew it instead, the command would not be proving
    that the pipeline runs end to end for both arms."""
    if not cache_ready(cfg, repo_root):
        pytest.skip("no cached history; this test needs data_cache/")

    table, _ = smoke_offline.run(cfg, model="persistence", n_folds=1)

    # Deduped: the arm IS the baseline. Buy-and-hold still runs, as the trading reference.
    assert set(table["arm"]) == {"persistence", smoke_offline.BUY_AND_HOLD}
    table = table[table["arm"] == "persistence"]
    row = table.iloc[0]
    assert row["aside"] == "yes"
    assert row["trades"] == 0
    assert math.isnan(row["direction"])  # a zero forecast expresses no direction
    assert not math.isnan(row["dir_ref"])  # the bar is a property of the fold


def test_the_command_sizes_with_the_risk_layer_not_a_stand_in(cfg: Config) -> None:
    """GB-24 shipped a placeholder; GB-21 replaced it. The caps now bind here too."""
    del cfg
    assert smoke_offline.risk.position_sizer is risk.position_sizer
    assert not hasattr(smoke_offline, "size_position")


# ── the buy-and-hold arm ─────────────────────────────────────────────────────


def test_buy_and_hold_direction_accuracy_is_the_always_long_bar(
    cfg: Config, repo_root: Path
) -> None:
    """The self-check: the same quantity, computed by two functions written apart.

    Buy-and-hold calls up on every window, so its direction accuracy IS the bar. If they
    ever disagree, one of `direction_accuracy` and `always_long_accuracy` is wrong and a
    bad bar sits under every direction number in the study. The run asserts it too; this
    proves the assertion is reached rather than skipped.
    """
    if not cache_ready(cfg, repo_root):
        pytest.skip("no cached history; this test needs data_cache/")

    table, _ = smoke_offline.run(cfg, model="dlinear", n_folds=1)
    row = table[table["arm"] == smoke_offline.BUY_AND_HOLD].iloc[0]

    assert row["direction"] == pytest.approx(row["dir_ref"], abs=1e-12)
    assert row["dir_vs_long"] == pytest.approx(0.0, abs=1e-12)


def test_the_bar_assertion_fires_when_the_two_disagree() -> None:
    """Proof the check above can fail, rather than passing because nothing is compared."""
    index = pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")
    mismatched = metrics.ArmResult(
        name="broken",
        equity=pd.Series([100.0, 101.0, 102.0], index=index),
        predicted=np.array([[0.01], [0.01]], dtype="float32"),
        actual=np.array(
            [[-0.01], [-0.01]], dtype="float32"
        ),  # never agrees: 0.0 vs 0.0
    )
    # Both are 0.0 here, so nudge one window up: direction 0.5, always-long 0.5 -> equal.
    # Make the forecast disagree with itself instead, which only direction_accuracy sees.
    forecast_declines = metrics.ArmResult(
        name="broken",
        equity=mismatched.equity,
        predicted=np.array([[-0.01], [0.01]], dtype="float32"),
        actual=np.array([[0.01], [0.01]], dtype="float32"),
    )

    with pytest.raises(smoke_offline.SmokeError, match="same quantity by construction"):
        smoke_offline._assert_is_the_bar(forecast_declines)


def test_buy_and_hold_reports_no_forecast_error(cfg: Config, repo_root: Path) -> None:
    """It makes a directional call and no magnitude forecast, so its MAE cell is empty.

    The same treatment persistence's direction column gets: a metric an arm does not
    produce is reported as absent, not as a number it never made.
    """
    if not cache_ready(cfg, repo_root):
        pytest.skip("no cached history; this test needs data_cache/")

    table, _ = smoke_offline.run(cfg, model="dlinear", n_folds=1)
    row = table[table["arm"] == smoke_offline.BUY_AND_HOLD].iloc[0]

    assert math.isnan(row["mae"])
    assert math.isnan(row["mae_vs_pers"])
    assert not math.isnan(row["total_return"])  # it does trade, and that IS reported


def test_buy_and_hold_holds_rather_than_being_stopped_out(
    cfg: Config, repo_root: Path
) -> None:
    """A 3% stop on a passive holding would make this arm something else entirely.

    Every exit must be administrative — the data running out — not a stop or a target.
    """
    if not cache_ready(cfg, repo_root):
        pytest.skip("no cached history; this test needs data_cache/")

    table, _ = smoke_offline.run(cfg, model="dlinear", n_folds=1)
    row = table[table["arm"] == smoke_offline.BUY_AND_HOLD].iloc[0]

    assert row["trades"] == 0  # strategy trades; the liquidations are administrative


def test_buy_and_hold_appears_in_the_table_and_the_aggregate(
    cfg: Config, repo_root: Path
) -> None:
    if not cache_ready(cfg, repo_root):
        pytest.skip("no cached history; this test needs data_cache/")

    table, summary = smoke_offline.run(cfg, model="dlinear", n_folds=1)

    assert smoke_offline.BUY_AND_HOLD in set(table["arm"])
    assert f"{smoke_offline.BUY_AND_HOLD} over 1 fold(s)" in summary


def test_the_equal_weight_sizer_cannot_overdraw(cfg: Config) -> None:
    """Fully invested, but never past the cash: the fifth entry would otherwise exceed it
    by the fees the first four paid."""
    sizer = smoke_offline.equal_weight(5)

    first = sizer(None, 100_000.0, 0.0, cfg)
    last = sizer(None, 100_000.0, 95_000.0, cfg)

    assert first == pytest.approx(20_000.0)
    assert last == pytest.approx(5_000.0)  # capped by cash, not by 1/n


def test_the_fold_table_never_prints_mae_without_flatness_beside_it() -> None:
    """§7.3's pairing, asserted on the table a reader actually sees (ruled 20 Aug 2026).

    ``metrics.summarise`` builds its columns from ``COMPANIONS``; this table writes its
    own, so it is the one that could drift out of step with the rule.
    """
    columns = list(smoke_offline.FOLD_COLUMNS)

    assert columns[columns.index("mae") + 1] == "flatness"
    assert columns[columns.index("flatness") + 1] == "mae_vs_pers"

    table = smoke_offline.fold_table(
        [fake_run(1, "dlinear", [100.0, 101.0], stood_aside=False)]
    )
    assert list(table.columns) == columns


def test_the_band_is_calibrated_on_forecasts_in_raw_units(cfg: Config) -> None:
    """Ruling of 20 Aug 2026, asserted rather than argued.

    ``build_windows`` scales the target, so a model fitted on those windows forecasts in
    that unit. GB-20's thresholds are quantiles of the validation forecast distribution
    and the backtester trades real prices, so a band fitted to scaled forecasts would be
    numerically fine and would mean nothing. Every ``Forecast`` this module builds is
    restored first, and the restoration is per row.
    """
    channels = cfg.channels.active_channels
    stats = {
        symbol: ChannelStats(
            channels=channels,
            mean=tuple(0.0 for _ in channels),
            std=tuple(deviation for _ in channels),
            fitted_start=pd.Timestamp("2024-01-01", tz="UTC"),
            fitted_end=pd.Timestamp("2024-02-01", tz="UTC"),
            n_rows=20,
        )
        for symbol, deviation in (("AAPL", 0.02), ("NVDA", 0.05))
    }
    batch = WindowBatch(
        X=np.zeros((2, cfg.window.input_len, len(channels)), dtype="float32"),
        y=np.ones((2, cfg.window.horizon), dtype="float32"),
        channels=channels,
        timestamps=pd.DatetimeIndex(["2024-03-01", "2024-03-01"], tz="UTC", name=None),
        symbols=("AAPL", "NVDA"),
        source="test",
    )
    scaled = np.ones((2, cfg.window.horizon), dtype="float32")

    forecasts = smoke_offline._forecasts(batch, scaled, stats)

    assert [f.symbol for f in forecasts] == ["AAPL", "NVDA"]
    np.testing.assert_allclose(forecasts[0].path, 0.02, rtol=1e-6)
    np.testing.assert_allclose(forecasts[1].path, 0.05, rtol=1e-6)
