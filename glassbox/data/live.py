"""L1: live daily bars from the Alpaca Data API, in a schema identical to
``historical.py``.

Schema identity is what allows the live loop and offline training to share
features/builder.py. It is not achieved by two implementations agreeing: both sources
pass through :func:`historical.normalise_bars`, which is the single definition of the
schema. GB-27's parity test rests on that.

Three things this module pins deliberately, each measured rather than assumed:

* **Adjustment.** Alpaca defaults to ``RAW`` — unadjusted prices. ``historical.py`` uses
  yfinance ``auto_adjust=True``. Measured on AAPL across its 2020-08-31 4:1 split, RAW
  closes 499.75 where adjusted closes 121.08, and ``SPLIT`` alone closes 124.94 because
  it leaves dividends in. Only ``Adjustment.ALL`` matches yfinance. This module always
  passes it. A default here would have produced a perfect schema match and prices off by
  a factor of four.
* **Feed.** ``SIP``, the consolidated tape, which is what yfinance reports. Measured on
  this account: SIP returns 2669 daily bars back to 2016-01-04, IEX only 1521 back to
  2020-07-27. IEX is a single venue with a small share of volume, so its daily OHLC is
  close to but not the consolidated print. If SIP access is ever lost the request fails
  loudly, which is the intended behaviour: a silent downgrade to IEX would change the
  numbers without changing the schema.

  **How far "close to but not" actually is, measured 2026-08-18 over 273 sessions x 5
  symbols against the same cached training source**: IEX is wrong by up to **173 bps on
  the open** (GOOGL), **193 bps on the low** (MSFT) and **90 bps on the close** (AMZN),
  against a 1 bp tolerance and a 2 bps modelled slippage. Volume is out by ~9,900 bps.
  The downgrade this module refuses to make is therefore worth up to **190x the price
  tolerance** — which is why the feed is a hardcoded enum with no fallback path rather
  than a configurable default.

  **The account's SIP entitlement is narrower than "SIP works", and the boundary matters.**
  Historical daily bars on SIP are served, including the most recent completed session,
  which is the only thing this module requests. The *recent-data* endpoints are refused:
  ``get_stock_latest_bar`` and ``get_stock_latest_quote`` both return
  ``subscription does not permit querying recent SIP data``. Anything in GB-26 that
  reaches for a live quote rather than a completed bar meets that wall.
* **No cache.** ``historical.py`` caches because a study needs a fixed snapshot (GB-4).
  Live has the opposite requirement: a cached bar served into a trading decision is a
  stale price, and the whole point of this module is freshness. Every call is a request.

**Measured tolerance against yfinance**, from ``scripts/compare_sources.py`` over 163
overlapping bars across the five-symbol universe on 14 Aug 2026, and **re-swept on
2026-08-18 over 273 overlapping sessions per symbol** — 1,365 bar-comparisons rather
than one window, which is what the GB-27b audit asks of any claim first measured at a
single point:

===========  ===============  =================  ==========================
Field        GB-7, 163 bars   Re-swept, 273/sym  Worst symbol
===========  ===============  =================  ==========================
open         0.2 bps          **0.51 bps**       NVDA
high         0.0 bps          **0.46 bps**       GOOGL
low          0.2 bps          **0.56 bps**       NVDA
close        0.0 bps          **0.30 bps**       NVDA
volume       111.4 bps        **213.6 bps**      NVDA
===========  ===============  =================  ==========================

So **prices agree to well under one basis point** — the residue is penny rounding on
the open, not a methodology difference — and the 1 bp assertion **survives the sweep**,
with 0.56 bps the worst cell of 1,365. **Volume does not agree**, and the sweep widens the
range GB-7 reported from 34-111 bps to **34-214 bps**; it never will agree, because the
vendors include different venues and trade conditions. Any feature built on volume must
therefore not assume cross-source equality; nothing in the C0 or C2 channel sets does
today. A price tolerance of 1 bp is a safe assertion; anything tighter will flake on the
open.

Implemented in GB-7.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from glassbox.config.loader import alpaca_credentials
from glassbox.data.historical import normalise_bars
from glassbox.faults import retry

LOGGER = logging.getLogger(__name__)

SOURCE_ALPACA = "alpaca"

# The two request parameters that decide *which numbers arrive*, named as constants so a
# test can assert them and every fetch can log them. See the module docstring for what a
# silent change to either is worth: 4x on adjustment, up to 190x the price tolerance on
# feed. Neither is configurable, and neither has a fallback path.
LIVE_FEED = DataFeed.SIP
LIVE_ADJUSTMENT = Adjustment.ALL

# A trading day is about 1.45 calendar days once weekends and holidays are counted.
# Two gives roughly a 38% margin, which covers the longest holiday stretches.
CALENDAR_DAYS_PER_TRADING_DAY = 2


def data_source() -> str:
    """The one line naming which tape these numbers come from. Logged on every fetch.

    A feed downgrade is invisible to every schema check — the columns, the dtypes, the
    index and the provenance string are all identical, and only the prices differ, by up
    to 190x the tolerance GB-7 measured. A claim with a cost that large should not rest on
    a reader remembering which constant is in the source, so it goes in the run log beside
    the bars it produced. GB-26 also calls this at session start, so the banner of every
    session names its own data source.
    """
    return f"feed={LIVE_FEED.value}, adjustment={LIVE_ADJUSTMENT.value}"


def lookback_days_for(bars: int) -> int:
    """Calendar days to request in order to receive at least ``bars`` sessions.

    Derived from the caller's bar requirement rather than from ``input_len``, because the
    two are not the same number and using the wrong one is the defect this function
    exists to remove: recursive channels need warm-up *behind* the window, so
    ``features.builder.min_history_bars`` is 445 for ``C0_base`` where ``input_len`` is
    120. Asking for ``input_len`` worth of calendar days returned 163 bars, measured on
    2026-08-18.
    """
    if bars <= 0:
        raise ValueError(f"bars must be positive, got {bars!r}")
    return bars * CALENDAR_DAYS_PER_TRADING_DAY


def load_live_bars(
    symbols: Sequence[str],
    min_bars: int,
    *,
    requirement: str = "",
    lookback_days: int | None = None,
    attempts: int = 1,
    backoff: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """Fetch recent daily bars for ``symbols``, in the historical loader's schema.

    **The caller states how much history it needs, and this refuses to return less.**
    ``min_bars`` is not defaulted and cannot be: the number is
    ``features.builder.min_history_bars(cfg)``, and this module sits *below* ``features``
    in the layer stack, so it cannot compute it. Making the caller name it is the honest
    version of that constraint — and it is what stops the floor being quietly restated as
    ``input_len``, which is exactly how the previous version was wrong.

    Args:
        symbols: Tickers to fetch.
        min_bars: The fewest bars every symbol must return. Pass
            ``features.builder.min_history_bars(cfg)``; passing ``window.input_len`` is
            the bug this parameter exists to prevent.
        requirement: One line saying where ``min_bars`` comes from, for the refusal
            message. ``features.builder.history_requirement(cfg)`` produces it.
        lookback_days: Calendar days of history to request. Defaults to
            :func:`lookback_days_for` of ``min_bars``.

    Returns:
        ``{symbol: DataFrame}`` with columns ``open, high, low, close, volume,
        log_return``, indexed by a sorted, unique, tz-aware UTC DatetimeIndex at
        midnight — the same schema ``historical.load_history`` returns.

        attempts: How many times to ask the vendor before giving up, including the
            first. **Defaults to 1 - no retry - because the retry policy is a deployment
            decision and this module has no configuration.** The live loop passes
            ``cfg.live.retry_attempts``; a caller that has not been given a policy gets
            the behaviour it asked for rather than a hidden one.
        backoff: Seconds before the second attempt, doubling thereafter.

    Raises:
        ValueError: A symbol returned no bars, or fewer than ``min_bars`` of them. **Not
            retried**: a short history is a true answer about the world, and asking again
            would return the same true answer three times.
        faults.Unavailable: the vendor did not answer at all, ``attempts`` times over.
    """
    if min_bars <= 0:
        raise ValueError(f"min_bars must be positive, got {min_bars!r}")
    if lookback_days is None:
        lookback_days = lookback_days_for(min_bars)

    start = datetime.now(UTC) - timedelta(days=lookback_days)
    LOGGER.info(
        "fetching %s: %s calendar days for at least %s bars each, %s",
        ", ".join(symbols),
        lookback_days,
        min_bars,
        data_source(),
    )
    # Only the network call is retried. Everything below it is validation of an answer
    # that did arrive, and a `ValueError` there is a fact rather than a fault.
    raw = retry(
        lambda: _fetch_bars(list(symbols), start),
        description=f"the daily bar fetch for {', '.join(symbols)}",
        attempts=attempts,
        backoff=backoff,
        log=LOGGER,
    )

    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        if symbol not in raw.index.get_level_values(0):
            raise ValueError(f"{symbol} returned no live bars since {start:%Y-%m-%d}")

        frame = normalise_bars(raw.xs(symbol), symbol, SOURCE_ALPACA)
        if len(frame) < min_bars:
            raise ValueError(
                f"{symbol} returned {len(frame)} daily bars over the {lookback_days} "
                f"calendar days requested, and at least {min_bars} are needed"
                + (f" — {requirement}" if requirement else "")
                + ". Proceeding on a short history would not fail loudly: the recursive "
                "channels emit NaN through their whole warm-up, so build_feature_frame "
                "returns an EMPTY frame rather than wrong values, and the failure "
                "surfaces later without naming this as the cause. Request more calendar "
                "days, or check the symbol has that much history at all."
            )

        frames[symbol] = frame
        LOGGER.info(
            "%s: %s live bars, %s to %s",
            symbol,
            len(frame),
            frame.index[0].strftime("%Y-%m-%d"),
            frame.index[-1].strftime("%Y-%m-%d"),
        )

    return frames


def _fetch_bars(symbols: list[str], start: datetime) -> pd.DataFrame:
    """Request daily bars and return the SDK's (symbol, timestamp) indexed frame.

    The seam the tests replace with a recorded response, so the suite needs no network.
    """
    credentials = alpaca_credentials()
    client = StockHistoricalDataClient(
        api_key=credentials.api_key,
        secret_key=credentials.secret_key,
    )
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        feed=LIVE_FEED,
        adjustment=LIVE_ADJUSTMENT,
    )
    return client.get_stock_bars(request).df
