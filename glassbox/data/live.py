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
* **No cache.** ``historical.py`` caches because a study needs a fixed snapshot (GB-4).
  Live has the opposite requirement: a cached bar served into a trading decision is a
  stale price, and the whole point of this module is freshness. Every call is a request.

**Measured tolerance against yfinance**, from ``scripts/compare_sources.py`` over 163
overlapping bars across the five-symbol universe on 14 Aug 2026:

===========  ==================  ==========================================
Field        Same-bar worst      Close, across every overlapping bar
===========  ==================  ==========================================
open         0.2 bps             median 0.00-0.08 bps per symbol
high         0.0 bps             p95 0.00-0.25 bps
low          0.2 bps             max 0.30 bps (NVDA, 2026-03-27)
close        0.0 bps             --
volume       111.4 bps           --
===========  ==================  ==========================================

So **prices agree to well under one basis point** — the residue is penny rounding on
the open, not a methodology difference. **Volume does not agree**, by 34-111 bps, and it
never will: the vendors include different venues and trade conditions. Any feature built
on volume must therefore not assume cross-source equality; nothing in the C0 or C2
channel sets does today. A price tolerance of 1 bp is a safe assertion; anything tighter
will flake on the open.

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

from glassbox.config.loader import Config, alpaca_credentials
from glassbox.data.historical import normalise_bars

LOGGER = logging.getLogger(__name__)

# A trading day is about 1.45 calendar days once weekends and holidays are counted.
# Two gives roughly a 38% margin, which covers the longest holiday stretches.
CALENDAR_DAYS_PER_TRADING_DAY = 2


def load_live_bars(
    symbols: Sequence[str],
    cfg: Config,
    lookback_days: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch recent daily bars for ``symbols``, in the historical loader's schema.

    Args:
        symbols: Tickers to fetch.
        cfg: Resolved configuration; supplies ``window.input_len``.
        lookback_days: Calendar days of history to request. Defaults to enough to cover
            ``window.input_len`` trading days with margin.

    Returns:
        ``{symbol: DataFrame}`` with columns ``open, high, low, close, volume,
        log_return``, indexed by a sorted, unique, tz-aware UTC DatetimeIndex at
        midnight — the same schema ``historical.load_history`` returns.

    Raises:
        ValueError: A symbol returned no bars, or fewer than ``window.input_len`` of
            them, which would leave the builder unable to assemble a window.
    """
    if lookback_days is None:
        lookback_days = cfg.window.input_len * CALENDAR_DAYS_PER_TRADING_DAY

    start = datetime.now(UTC) - timedelta(days=lookback_days)
    raw = _fetch_bars(list(symbols), start)

    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        if symbol not in raw.index.get_level_values(0):
            raise ValueError(f"{symbol} returned no live bars since {start:%Y-%m-%d}")

        frame = normalise_bars(raw.xs(symbol), symbol)
        if len(frame) < cfg.window.input_len:
            raise ValueError(
                f"{symbol} returned {len(frame)} bars, fewer than the "
                f"{cfg.window.input_len} the input window needs"
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
        feed=DataFeed.SIP,
        adjustment=Adjustment.ALL,
    )
    return client.get_stock_bars(request).df
