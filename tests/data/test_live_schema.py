"""GB-7 acceptance: the live and historical loaders are schema-indistinguishable.

This is the test that makes GB-27's train/live parity possible. It runs offline against
a recorded Alpaca response, so it needs no network and no credentials.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pandas_market_calendars as mcal
import pytest
from alpaca.data.enums import Adjustment, DataFeed

from glassbox.config.loader import Config, load_config
from glassbox.data import historical, live
from glassbox.features import builder

ALPACA_FIXTURE = Path(__file__).with_name("fixtures") / "alpaca_bars_response.csv"
YFINANCE_FIXTURE = Path(__file__).with_name("fixtures") / "sample_bars.csv"

# The recorded response holds 178 bars per symbol, so this is what the schema tests can
# ask for. It is passed explicitly because `load_live_bars` no longer defaults it — see
# `test_the_guard_is_the_callers_floor_not_input_len` for why that mattered.
MIN_BARS = 120


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
    online = live.load_live_bars(["AAPL"], MIN_BARS)["AAPL"]

    assert list(offline.columns) == list(online.columns)


def test_dtypes_are_identical(
    cfg: Config, stub_live: None, stub_historical: None
) -> None:
    """Alpaca reports volume as float and yfinance as int; the schema pins it."""
    offline = historical.load_history(["AAPL"], cfg)["AAPL"]
    online = live.load_live_bars(["AAPL"], MIN_BARS)["AAPL"]

    pd.testing.assert_series_equal(offline.dtypes, online.dtypes)


def test_index_conventions_are_identical(
    cfg: Config, stub_live: None, stub_historical: None
) -> None:
    """Alpaca stamps midnight New York, yfinance the bare date. Both land on midnight UTC."""
    offline = historical.load_history(["AAPL"], cfg)["AAPL"]
    online = live.load_live_bars(["AAPL"], MIN_BARS)["AAPL"]

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
    online = live.load_live_bars(["AAPL"], MIN_BARS)["AAPL"]

    combined = pd.concat([offline, online])

    assert list(combined.columns) == list(offline.columns)
    pd.testing.assert_series_equal(combined.dtypes, offline.dtypes)


# ── The live loader's own contract ───────────────────────────────────────────


def test_returns_one_frame_per_symbol(cfg: Config, stub_live: None) -> None:
    frames = live.load_live_bars(["AAPL", "MSFT"], MIN_BARS)

    assert set(frames) == {"AAPL", "MSFT"}


def test_extra_alpaca_columns_are_dropped(cfg: Config, stub_live: None) -> None:
    """trade_count and vwap exist in the response and must not reach the feature layer."""
    frame = live.load_live_bars(["AAPL"], MIN_BARS)["AAPL"]

    assert "trade_count" not in frame.columns
    assert "vwap" not in frame.columns
    assert list(frame.columns) == [*historical.OHLCV_COLUMNS, "log_return", "source"]


def test_live_bars_are_stamped_alpaca(cfg: Config, stub_live: None) -> None:
    """Both loaders stamp provenance, so the schemas stay identical and GB-9 can tell
    a spliced frame from a clean one."""
    frame = live.load_live_bars(["AAPL"], MIN_BARS)["AAPL"]

    assert frame["source"].unique().tolist() == [live.SOURCE_ALPACA]


def test_log_return_is_present_and_unfilled(cfg: Config, stub_live: None) -> None:
    frame = live.load_live_bars(["AAPL"], MIN_BARS)["AAPL"]

    assert pd.isna(frame["log_return"].iloc[0])
    assert frame["log_return"].iloc[1:].notna().all()


def test_too_few_bars_raises(stub_live: None) -> None:
    """A short response must fail loudly: the builder cannot assemble a window from it."""
    with pytest.raises(ValueError, match=r"AAPL returned \d+ daily bars"):
        live.load_live_bars(["AAPL"], 500)


def test_unknown_symbol_raises(cfg: Config, stub_live: None) -> None:
    with pytest.raises(ValueError, match="TSLA returned no live bars"):
        live.load_live_bars(["TSLA"], MIN_BARS)


# ── The floor is the caller's, and the message says whose it is ──────────────
#
# Measured 2026-08-18: the previous version derived its lookback from `input_len × 2` and
# guarded at `input_len`, so a request for 240 calendar days returned 163 bars against a
# `min_history_bars` of 445 — and passed. The failure then surfaced as an empty feature
# frame, because `rsi` is NaN through its whole 325-bar warm-up, with nothing naming the
# cause. These are the tests that stop it coming back.


def test_the_guard_is_the_callers_floor_not_input_len(
    cfg: Config, stub_live: None
) -> None:
    """178 bars satisfies `input_len` of 120 and must still be refused against 445."""
    floor = builder.min_history_bars(cfg)

    with pytest.raises(ValueError) as excinfo:
        live.load_live_bars(["AAPL"], floor)

    assert str(floor) in str(excinfo.value)


def test_the_refusal_names_requested_received_floor_and_cause(
    cfg: Config, stub_live: None
) -> None:
    """A message that says only "too few" leaves the reader to rediscover the warm-up."""
    with pytest.raises(ValueError) as excinfo:
        live.load_live_bars(["AAPL"], 445, requirement=builder.history_requirement(cfg))

    message = str(excinfo.value)
    assert "178 daily bars" in message  # received
    assert f"{live.lookback_days_for(445)} calendar days" in message  # requested
    assert "at least 445" in message  # the floor
    assert "rsi14 warm-up" in message  # the channel that sets it
    assert "EMPTY frame" in message  # why it would not have failed loudly


def test_history_requirement_names_the_deepest_channel(cfg: Config) -> None:
    """`features` writes the sentence because `data` sits below it and cannot."""
    assert builder.deepest_warmup_channel(cfg) == "rsi14"
    assert builder.history_requirement(cfg) == (
        "445 bars is input_len 120 plus 325 bars of rsi14 warm-up, the deepest of the "
        "active channels"
    )


def test_the_default_lookback_is_derived_from_the_floor_not_the_window(
    cfg: Config,
) -> None:
    """The arithmetic that was wrong, pinned to the right input."""
    assert live.lookback_days_for(builder.min_history_bars(cfg)) == 890
    assert (
        live.lookback_days_for(cfg.window.input_len) == 240
    )  # what it used to ask for


def test_the_default_lookback_yields_the_floor_on_the_real_calendar(
    cfg: Config,
) -> None:
    """Real NYSE sessions, worst window in a decade — not an average.

    The conversion from calendar days to trading sessions is a property of the exchange
    calendar, so this counts real sessions rather than trusting the 1.45 ratio. It slides
    the default window across ten years and takes the **minimum**, so the assertion covers
    the sparsest holiday stretch in the period rather than a typical one.
    """
    floor = builder.min_history_bars(cfg)
    window = live.lookback_days_for(floor)
    sessions = pd.DatetimeIndex(
        mcal.get_calendar(cfg.data.calendar).valid_days("2015-01-01", "2026-08-18")
    ).normalize()

    span = pd.Timedelta(days=window)
    covered = [
        end + 1 - int(sessions.searchsorted(session - span, side="left"))
        for end, session in enumerate(sessions)
        if session - span >= sessions[0]
    ]

    assert covered, "the calendar range is shorter than one lookback window"
    assert min(covered) >= floor, f"worst window held {min(covered)} sessions"


# ── The feed, asserted rather than commented ─────────────────────────────────
#
# A feed downgrade changes the prices and nothing else: same columns, same dtypes, same
# index, same provenance string. Every test above would still pass. Measured 2026-08-18,
# IEX differs from the training source by up to 193 bps against a tolerance of 1 bp and a
# modelled slippage of 2 bps, so this is a claim with a large measured cost and, until
# now, no backstop at all.


def test_the_feed_and_adjustment_are_pinned() -> None:
    """SIP is the consolidated tape yfinance reports; ALL matches auto_adjust=True."""
    assert live.LIVE_FEED is DataFeed.SIP
    assert live.LIVE_ADJUSTMENT is Adjustment.ALL


def test_the_request_actually_carries_the_pinned_feed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The constants existing is not the claim; the request using them is.

    Asserted against the object handed to the SDK, so a future edit that adds a `feed`
    argument somewhere else, or drops it, fails here rather than in a paper session.
    """
    seen: dict[str, Any] = {}

    class FakeClient:
        # `_session` because `bound_reads` refuses a client it cannot bound, and that
        # strictness is the point: a double that got waved through would mean the guard
        # could silently no-op on the real client too. A stand-in for an SDK client that
        # holds a session has to hold one.
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self._session = type("Session", (), {"request": lambda *a, **k: None})()

        def get_stock_bars(self, request: Any) -> Any:
            seen["feed"] = request.feed
            seen["adjustment"] = request.adjustment
            seen["timeframe"] = request.timeframe
            return type("Response", (), {"df": alpaca_response()})()

    monkeypatch.setattr(live, "StockHistoricalDataClient", FakeClient)
    monkeypatch.setattr(
        live,
        "alpaca_credentials",
        lambda: type("Credentials", (), {"api_key": "k", "secret_key": "s"})(),
    )

    live.load_live_bars(["AAPL"], MIN_BARS)

    assert seen["feed"] is DataFeed.SIP
    assert seen["adjustment"] is Adjustment.ALL


def test_the_feed_is_logged_on_every_fetch(
    stub_live: None, caplog: pytest.LogCaptureFixture
) -> None:
    """Not only at session start: every fetch records which tape produced its bars."""
    with caplog.at_level("INFO", logger="glassbox.data.live"):
        live.load_live_bars(["AAPL"], MIN_BARS)

    assert "feed=sip" in caplog.text
    assert "adjustment=all" in caplog.text


def test_data_source_is_the_line_gb26_logs_at_session_start() -> None:
    assert live.data_source() == "feed=sip, adjustment=all"
