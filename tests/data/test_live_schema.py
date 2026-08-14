"""GB-7 acceptance: the live and historical loaders are schema-indistinguishable.

This is the test that makes GB-27's train/live parity possible. It runs offline against
a recorded Alpaca response, so it needs no network and no credentials.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from glassbox.config.loader import Config, load_config
from glassbox.data import historical, live

ALPACA_FIXTURE = Path(__file__).with_name("fixtures") / "alpaca_bars_response.csv"
YFINANCE_FIXTURE = Path(__file__).with_name("fixtures") / "sample_bars.csv"


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """Real config, temporary cache, and an input window the fixture can satisfy."""
    base = load_config()
    return replace(
        base,
        data=replace(base.data, cache_dir=str(tmp_path / "cache")),
        window=replace(base.window, input_len=120),
    )


def alpaca_response() -> pd.DataFrame:
    """The recorded response, rebuilt with the SDK's (symbol, timestamp) index."""
    frame = pd.read_csv(ALPACA_FIXTURE, parse_dates=["timestamp"])
    return frame.set_index(["symbol", "timestamp"])


@pytest.fixture
def stub_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live, "_fetch_bars", lambda symbols, start: alpaca_response())


@pytest.fixture
def stub_historical(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download(symbol: str, **kwargs: Any) -> pd.DataFrame:
        frame = pd.read_csv(YFINANCE_FIXTURE, parse_dates=["Date"]).set_index("Date")
        frame.columns = pd.MultiIndex.from_product([list(frame.columns), [symbol]])
        return frame

    monkeypatch.setattr(historical.yf, "download", fake_download)


# ── The schema equality that GB-27 rests on ──────────────────────────────────


def test_column_sets_are_identical(
    cfg: Config, stub_live: None, stub_historical: None
) -> None:
    offline = historical.load_history(["AAPL"], cfg)["AAPL"]
    online = live.load_live_bars(["AAPL"], cfg)["AAPL"]

    assert list(offline.columns) == list(online.columns)


def test_dtypes_are_identical(
    cfg: Config, stub_live: None, stub_historical: None
) -> None:
    """Alpaca reports volume as float and yfinance as int; the schema pins it."""
    offline = historical.load_history(["AAPL"], cfg)["AAPL"]
    online = live.load_live_bars(["AAPL"], cfg)["AAPL"]

    pd.testing.assert_series_equal(offline.dtypes, online.dtypes)


def test_index_conventions_are_identical(
    cfg: Config, stub_live: None, stub_historical: None
) -> None:
    """Alpaca stamps midnight New York, yfinance the bare date. Both land on midnight UTC."""
    offline = historical.load_history(["AAPL"], cfg)["AAPL"]
    online = live.load_live_bars(["AAPL"], cfg)["AAPL"]

    assert offline.index.dtype == online.index.dtype
    assert str(offline.index.tz) == str(online.index.tz) == "UTC"
    assert (online.index == online.index.normalize()).all()
    assert online.index.is_monotonic_increasing
    assert online.index.is_unique


def test_frames_from_both_sources_concatenate_cleanly(
    cfg: Config, stub_live: None, stub_historical: None
) -> None:
    """The strongest form of the claim: the feature layer cannot tell them apart."""
    offline = historical.load_history(["AAPL"], cfg)["AAPL"]
    online = live.load_live_bars(["AAPL"], cfg)["AAPL"]

    combined = pd.concat([offline, online])

    assert list(combined.columns) == list(offline.columns)
    pd.testing.assert_series_equal(combined.dtypes, offline.dtypes)


# ── The live loader's own contract ───────────────────────────────────────────


def test_returns_one_frame_per_symbol(cfg: Config, stub_live: None) -> None:
    frames = live.load_live_bars(["AAPL", "MSFT"], cfg)

    assert set(frames) == {"AAPL", "MSFT"}


def test_extra_alpaca_columns_are_dropped(cfg: Config, stub_live: None) -> None:
    """trade_count and vwap exist in the response and must not reach the feature layer."""
    frame = live.load_live_bars(["AAPL"], cfg)["AAPL"]

    assert "trade_count" not in frame.columns
    assert "vwap" not in frame.columns
    assert list(frame.columns) == [*historical.OHLCV_COLUMNS, "log_return"]


def test_log_return_is_present_and_unfilled(cfg: Config, stub_live: None) -> None:
    frame = live.load_live_bars(["AAPL"], cfg)["AAPL"]

    assert pd.isna(frame["log_return"].iloc[0])
    assert frame["log_return"].iloc[1:].notna().all()


def test_too_few_bars_raises(cfg: Config, stub_live: None) -> None:
    """A short response must fail loudly: the builder cannot assemble a window from it."""
    demanding = replace(cfg, window=replace(cfg.window, input_len=500))

    with pytest.raises(ValueError, match=r"AAPL returned \d+ bars, fewer than the 500"):
        live.load_live_bars(["AAPL"], demanding)


def test_unknown_symbol_raises(cfg: Config, stub_live: None) -> None:
    with pytest.raises(ValueError, match="TSLA returned no live bars"):
        live.load_live_bars(["TSLA"], cfg)
