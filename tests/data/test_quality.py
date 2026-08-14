"""GB-5 acceptance: the quality report detects injected gaps, NaNs, splits and duplicates."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest

from glassbox.config.loader import Config, load_config
from glassbox.data import quality

START, END = "2026-02-02", "2026-02-27"  # four clean NYSE weeks, no holidays


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    base = load_config()
    return replace(base, data=replace(base.data, cache_dir=str(tmp_path / "cache")))


def trading_days(cfg: Config) -> pd.DatetimeIndex:
    schedule = mcal.get_calendar(cfg.data.calendar).valid_days(START, END)
    return pd.DatetimeIndex(schedule).tz_convert("UTC").normalize()


def clean_frame(cfg: Config) -> pd.DataFrame:
    """A frame with one bar per trading day and a gentle upward drift."""
    index = trading_days(cfg)
    close = pd.Series(100.0 * (1.001 ** np.arange(len(index))), index=index)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
        },
        index=index,
    )
    frame["log_return"] = np.log(frame["close"] / frame["close"].shift(1))
    return frame


# ── A clean frame reports nothing ────────────────────────────────────────────


def test_clean_frame_has_no_findings(cfg: Config) -> None:
    report = quality.check_quality(clean_frame(cfg), "AAPL", cfg)

    assert report.missing_bars == ()
    assert report.large_moves == ()
    assert report.duplicate_dates == ()
    assert sum(report.nan_counts.values()) == 1  # the unfilled first log_return


def test_reports_span_and_count(cfg: Config) -> None:
    frame = clean_frame(cfg)
    report = quality.check_quality(frame, "AAPL", cfg)

    assert report.symbol == "AAPL"
    assert report.bar_count == len(frame)
    assert report.first_date == "2026-02-02"
    assert report.last_date == "2026-02-27"


# ── Injected defects ─────────────────────────────────────────────────────────


def test_detects_a_missing_trading_day(cfg: Config) -> None:
    frame = clean_frame(cfg)
    dropped = frame.index[5]
    report = quality.check_quality(frame.drop(index=dropped), "AAPL", cfg)

    assert report.missing_bars == (dropped.strftime("%Y-%m-%d"),)


def test_detects_a_multi_day_gap(cfg: Config) -> None:
    frame = clean_frame(cfg)
    dropped = frame.index[5:8]
    report = quality.check_quality(frame.drop(index=dropped), "AAPL", cfg)

    assert len(report.missing_bars) == 3


def test_counts_nans_per_column(cfg: Config) -> None:
    frame = clean_frame(cfg)
    frame.iloc[3, frame.columns.get_loc("volume")] = np.nan
    frame.iloc[4, frame.columns.get_loc("high")] = np.nan
    frame.iloc[5, frame.columns.get_loc("high")] = np.nan

    report = quality.check_quality(frame, "AAPL", cfg)

    assert report.nan_counts["volume"] == 1
    assert report.nan_counts["high"] == 2
    assert report.nan_counts["close"] == 0


def test_detects_a_two_for_one_split(cfg: Config) -> None:
    """An unadjusted 2:1 split halves the close: a log return of about -0.693."""
    frame = clean_frame(cfg)
    split_at = 10
    frame.iloc[split_at:, frame.columns.get_loc("close")] /= 2
    frame["log_return"] = np.log(frame["close"] / frame["close"].shift(1))

    report = quality.check_quality(frame, "AAPL", cfg)

    assert len(report.large_moves) == 1
    date, magnitude = report.large_moves[0]
    assert date == frame.index[split_at].strftime("%Y-%m-%d")
    assert magnitude == pytest.approx(-0.693, abs=0.01)


def test_ignores_moves_below_the_threshold(cfg: Config) -> None:
    frame = clean_frame(cfg)
    frame.iloc[7, frame.columns.get_loc("close")] *= 1.2  # ~0.18, under 0.25
    frame["log_return"] = np.log(frame["close"] / frame["close"].shift(1))

    report = quality.check_quality(frame, "AAPL", cfg)

    assert all(abs(value) > 0.25 for _, value in report.large_moves)


def test_detects_duplicate_dates(cfg: Config) -> None:
    frame = clean_frame(cfg)
    duplicated = pd.concat([frame, frame.iloc[[2]]]).sort_index()

    report = quality.check_quality(duplicated, "AAPL", cfg)

    assert report.duplicate_dates == (frame.index[2].strftime("%Y-%m-%d"),)


def test_computes_returns_when_the_column_is_absent(cfg: Config) -> None:
    frame = clean_frame(cfg).drop(columns=["log_return"])
    frame.iloc[6, frame.columns.get_loc("close")] /= 2

    report = quality.check_quality(frame, "AAPL", cfg)

    assert len(report.large_moves) == 2  # down into the halved bar, back up after it


# ── Failure modes ────────────────────────────────────────────────────────────


def test_empty_frame_raises(cfg: Config) -> None:
    with pytest.raises(ValueError, match="AAPL has no bars"):
        quality.check_quality(pd.DataFrame(), "AAPL", cfg)


def test_missing_close_raises(cfg: Config) -> None:
    frame = clean_frame(cfg).drop(columns=["close"])
    with pytest.raises(ValueError, match="AAPL is missing the close column"):
        quality.check_quality(frame, "AAPL", cfg)


# ── report_all ───────────────────────────────────────────────────────────────


def test_report_all_writes_json_and_prints_a_table(
    cfg: Config, capsys: pytest.CaptureFixture[str]
) -> None:
    frames = {"AAPL": clean_frame(cfg), "MSFT": clean_frame(cfg)}

    reports = quality.report_all(frames, cfg)

    assert set(reports) == {"AAPL", "MSFT"}

    printed = capsys.readouterr().out
    assert "symbol" in printed
    assert "AAPL" in printed and "MSFT" in printed

    path = Path(cfg.data.cache_dir) / quality.REPORT_FILENAME
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"AAPL", "MSFT"}
    assert payload["AAPL"]["bar_count"] == len(frames["AAPL"])
