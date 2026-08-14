"""L1: data quality report per symbol - gaps, NaNs, large moves, duplicates.

This module reports facts about the data. It does not judge them, grade them or decide
what to do about them: a missing bar might be a holiday the calendar disagrees about, and
a large move might be March 2020. Interpretation belongs to whoever reads the report.

On ``large_moves``: prices are loaded with ``auto_adjust=True``, so genuine splits and
dividends are already adjusted out of the series. A single-bar absolute log return above
0.25 in this data is therefore far more likely to be real volatility - a crash, an
earnings gap, NVDA in 2024 - than an adjustment artefact. The field is named for what it
measures, not for what it might mean.

Implemented in GB-5.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from glassbox.config.loader import Config

CLOSE = "close"
LOG_RETURN = "log_return"

# A single-bar absolute log return above this is reported. 0.25 is roughly a 28% move:
# well beyond ordinary daily volatility for a mega-cap, well below a 2:1 split (0.69).
# Deliberately a module constant rather than a config key - it is a reporting threshold
# for a human reading a table, not a value any model or decision consumes.
LARGE_MOVE_LOG_RETURN = 0.25

REPORT_FILENAME = "quality_report.json"


@dataclass(frozen=True)
class QualityReport:
    """What the data looks like for one symbol. Facts only."""

    symbol: str
    first_date: str
    last_date: str
    bar_count: int
    missing_bars: tuple[str, ...]
    nan_counts: dict[str, int]
    large_moves: tuple[tuple[str, float], ...]
    duplicate_dates: tuple[str, ...]


def check_quality(frame: pd.DataFrame, symbol: str, cfg: Config) -> QualityReport:
    """Report gaps, NaNs, large moves and duplicates for one symbol's bars.

    Args:
        frame: Daily bars indexed by a tz-aware UTC DatetimeIndex.
        symbol: The ticker, carried into the report.
        cfg: Resolved configuration; supplies ``data.calendar``.

    Returns:
        A :class:`QualityReport`. Empty tuples mean nothing was found.

    Raises:
        ValueError: The frame is empty, or lacks a close column.
    """
    if frame.empty:
        raise ValueError(f"{symbol} has no bars to check")
    if CLOSE not in frame.columns:
        raise ValueError(f"{symbol} is missing the {CLOSE} column")

    index = pd.DatetimeIndex(frame.index)
    returns = (
        frame[LOG_RETURN]
        if LOG_RETURN in frame.columns
        else np.log(frame[CLOSE] / frame[CLOSE].shift(1))
    )
    large = returns[returns.abs() > LARGE_MOVE_LOG_RETURN].dropna()

    return QualityReport(
        symbol=symbol,
        first_date=_label(index.min()),
        last_date=_label(index.max()),
        bar_count=len(frame),
        missing_bars=_missing_bars(index, cfg.data.calendar),
        nan_counts={str(column): int(frame[column].isna().sum()) for column in frame},
        large_moves=tuple(
            (_label(date), float(value)) for date, value in large.items()
        ),
        duplicate_dates=tuple(
            _label(date) for date in index[index.duplicated()].unique()
        ),
    )


def report_all(
    frames: dict[str, pd.DataFrame], cfg: Config
) -> dict[str, QualityReport]:
    """Check every symbol, print a table and write the report to the cache directory.

    The JSON file is a provenance artifact: it records what the data looked like when the
    study ran, so a result can be read alongside the state of its inputs.
    """
    reports = {
        symbol: check_quality(frame, symbol, cfg) for symbol, frame in frames.items()
    }

    header = f"{'symbol':<8}{'bars':>7}{'first':>13}{'last':>13}{'missing':>9}{'NaNs':>7}{'dups':>6}{'large':>7}"
    print(header)
    print("-" * len(header))
    for report in reports.values():
        print(
            f"{report.symbol:<8}{report.bar_count:>7}{report.first_date:>13}"
            f"{report.last_date:>13}{len(report.missing_bars):>9}"
            f"{sum(report.nan_counts.values()):>7}{len(report.duplicate_dates):>6}"
            f"{len(report.large_moves):>7}"
        )

    path = Path(cfg.data.cache_dir) / REPORT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {symbol: asdict(report) for symbol, report in reports.items()}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwritten to {path}")

    return reports


def _missing_bars(index: pd.DatetimeIndex, calendar: str) -> tuple[str, ...]:
    """Trading days the exchange calendar expects between first and last bar, but absent."""
    schedule = mcal.get_calendar(calendar).valid_days(
        start_date=index.min().date(), end_date=index.max().date()
    )
    expected = pd.DatetimeIndex(schedule).tz_convert("UTC").normalize()
    present = index.tz_convert("UTC").normalize()
    return tuple(_label(date) for date in expected.difference(present))


def _label(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%d")
