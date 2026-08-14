"""GB-4 acceptance: the historical loader normalises, caches and never fabricates.

Every test runs offline. yfinance is replaced by a stub reading a committed CSV, so a
network call in this module is a test failure, not a slow test.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from glassbox.config.loader import Config, load_config
from glassbox.data import historical

FIXTURE = Path(__file__).with_name("fixtures") / "sample_bars.csv"

# ln(110/100) and ln(121/110), computed by hand from the fixture's closes.
LOG_RETURN_DAY_2 = 0.09531017980432493
LOG_RETURN_DAY_3 = 0.09531017980432493
LOG_RETURN_DAY_4 = 0.09531017980432474


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """The real configuration, with the cache redirected into a temporary directory."""
    base = load_config()
    return replace(base, data=replace(base.data, cache_dir=str(tmp_path / "cache")))


def read_fixture() -> pd.DataFrame:
    """The fixture as yfinance would hand it over: Date index, title-case columns."""
    frame = pd.read_csv(FIXTURE, parse_dates=["Date"])
    return frame.set_index("Date")


def multiindex_frame(symbol: str) -> pd.DataFrame:
    """The fixture with yfinance 1.6's (field, ticker) column layout."""
    frame = read_fixture()
    frame.columns = pd.MultiIndex.from_product([list(frame.columns), [symbol]])
    return frame


@pytest.fixture
def stub_download(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Replace yf.download with the fixture. Returns a per-symbol call counter."""
    calls: dict[str, int] = {}

    def fake_download(symbol: str, **kwargs: Any) -> pd.DataFrame:
        calls[symbol] = calls.get(symbol, 0) + 1
        return multiindex_frame(symbol)

    monkeypatch.setattr(historical.yf, "download", fake_download)
    return calls


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any download attempt fails loudly."""

    def explode(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise AssertionError("yfinance was called; the cache should have served this")

    monkeypatch.setattr(historical.yf, "download", explode)


# ── Shape and normalisation ──────────────────────────────────────────────────


def test_returns_one_frame_per_symbol(
    cfg: Config, stub_download: dict[str, int]
) -> None:
    frames = historical.load_history(["AAPL", "MSFT"], cfg)
    assert set(frames) == {"AAPL", "MSFT"}


def test_columns_are_lowercase_ohlcv_plus_log_return(
    cfg: Config, stub_download: dict[str, int]
) -> None:
    frame = historical.load_history(["AAPL"], cfg)["AAPL"]
    assert list(frame.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "log_return",
    ]


def test_index_is_tz_aware_utc_monotonic_and_unique(
    cfg: Config, stub_download: dict[str, int]
) -> None:
    """The fixture is deliberately out of order and carries a duplicated date."""
    frame = historical.load_history(["AAPL"], cfg)["AAPL"]

    assert isinstance(frame.index, pd.DatetimeIndex)
    assert str(frame.index.tz) == "UTC"
    assert frame.index.is_monotonic_increasing
    assert frame.index.is_unique
    assert len(frame) == 4  # five raw rows, one a duplicate date


def test_duplicate_date_keeps_the_later_row(
    cfg: Config, stub_download: dict[str, int]
) -> None:
    frame = historical.load_history(["AAPL"], cfg)["AAPL"]
    assert frame.loc["2026-01-06", "close"] == 121.0


# ── log_return ───────────────────────────────────────────────────────────────


def test_log_return_matches_hand_computed_values(
    cfg: Config, stub_download: dict[str, int]
) -> None:
    """Closes are 100.0, 110.0, 121.0, 133.1; each step is ln(1.1)."""
    returns = historical.load_history(["AAPL"], cfg)["AAPL"]["log_return"]

    assert returns.iloc[1] == pytest.approx(LOG_RETURN_DAY_2, abs=1e-15)
    assert returns.iloc[2] == pytest.approx(LOG_RETURN_DAY_3, abs=1e-15)
    assert returns.iloc[3] == pytest.approx(LOG_RETURN_DAY_4, abs=1e-15)


def test_first_log_return_is_nan_and_never_filled(
    cfg: Config, stub_download: dict[str, int]
) -> None:
    """A filled first return is a fabricated observation."""
    returns = historical.load_history(["AAPL"], cfg)["AAPL"]["log_return"]

    assert np.isnan(returns.iloc[0])
    assert returns.iloc[1:].notna().all()


# ── Caching ──────────────────────────────────────────────────────────────────


def test_cache_file_written_per_symbol(
    cfg: Config, stub_download: dict[str, int]
) -> None:
    historical.load_history(["AAPL", "MSFT"], cfg)
    cache_dir = Path(cfg.data.cache_dir)

    assert (cache_dir / "AAPL.parquet").is_file()
    assert (cache_dir / "MSFT.parquet").is_file()


def test_second_call_serves_from_cache_without_the_network(
    cfg: Config, stub_download: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The acceptance criterion: a cache hit must not touch the network."""
    historical.load_history(["AAPL"], cfg)
    assert stub_download["AAPL"] == 1

    def explode(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise AssertionError("yfinance was called on a cache hit")

    monkeypatch.setattr(historical.yf, "download", explode)
    second = historical.load_history(["AAPL"], cfg)

    assert len(second["AAPL"]) == 4


def test_cached_frame_is_identical_to_the_fetched_frame(
    cfg: Config, stub_download: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    fetched = historical.load_history(["AAPL"], cfg)["AAPL"]

    def explode(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise AssertionError("yfinance was called on a cache hit")

    monkeypatch.setattr(historical.yf, "download", explode)
    cached = historical.load_history(["AAPL"], cfg)["AAPL"]

    assert_frame_equal(fetched, cached)


def test_force_refresh_refetches_and_overwrites(
    cfg: Config, stub_download: dict[str, int]
) -> None:
    historical.load_history(["AAPL"], cfg)
    historical.load_history(["AAPL"], cfg, force_refresh=True)

    assert stub_download["AAPL"] == 2


def test_partial_cache_fetches_only_what_is_missing(
    cfg: Config, stub_download: dict[str, int]
) -> None:
    historical.load_history(["AAPL"], cfg)
    historical.load_history(["AAPL", "MSFT"], cfg)

    assert stub_download == {"AAPL": 1, "MSFT": 1}


def test_stale_cache_is_returned_unchanged(
    cfg: Config, stub_download: dict[str, int], no_network: None
) -> None:
    """A cache ending before today is served as-is; refreshing is explicit (GB-59)."""
    cache_dir = Path(cfg.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    old = historical._normalise(multiindex_frame("AAPL"), "AAPL")
    old.to_parquet(cache_dir / "AAPL.parquet")

    frame = historical.load_history(["AAPL"], cfg)["AAPL"]

    assert frame.index[-1] == pd.Timestamp("2026-01-07", tz="UTC")


# ── Logging ──────────────────────────────────────────────────────────────────


def test_logs_name_cached_and_fetched_symbols(
    cfg: Config, stub_download: dict[str, int], caplog: pytest.LogCaptureFixture
) -> None:
    historical.load_history(["AAPL"], cfg)
    with caplog.at_level(logging.INFO, logger=historical.LOGGER.name):
        historical.load_history(["AAPL", "MSFT"], cfg)

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "AAPL: cache hit" in messages
    assert "MSFT: fetched" in messages
    assert "1 from cache (AAPL), 1 fetched (MSFT)" in messages


# ── Failure modes ────────────────────────────────────────────────────────────


def test_flat_column_frames_are_supported(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not every yfinance path returns MultiIndex columns."""
    monkeypatch.setattr(
        historical.yf, "download", lambda symbol, **kwargs: read_fixture()
    )
    frame = historical.load_history(["AAPL"], cfg)["AAPL"]

    assert list(frame.columns)[:5] == list(historical.OHLCV_COLUMNS)


def test_empty_download_raises_naming_the_symbol(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        historical.yf, "download", lambda symbol, **kwargs: pd.DataFrame()
    )
    with pytest.raises(ValueError, match="AAPL returned no bars"):
        historical.load_history(["AAPL"], cfg)


def test_missing_column_raises_naming_the_symbol(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    def without_volume(symbol: str, **kwargs: Any) -> pd.DataFrame:
        return read_fixture().drop(columns=["Volume"])

    monkeypatch.setattr(historical.yf, "download", without_volume)
    with pytest.raises(ValueError, match=r"AAPL is missing required columns"):
        historical.load_history(["AAPL"], cfg)
