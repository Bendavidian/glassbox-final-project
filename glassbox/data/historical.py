"""L1: historical daily bars from yfinance, cached to parquet.

Returns a tz-aware UTC DatetimeIndex. A second call for the same range hits the cache.

Contract:

* :func:`load_history` returns ``{symbol: DataFrame}`` with columns
  ``open, high, low, close, volume, log_return``, indexed by a sorted, unique,
  tz-aware UTC ``DatetimeIndex``.
* ``log_return`` is ``ln(close_t / close_{t-1})``. The first row is NaN and is never
  filled: a filled first return is a fabricated observation.
* The cache is authoritative. A cached symbol is returned as it stands, however old,
  and is only refetched when ``force_refresh=True``. See the module note below.

Downloads are per-symbol, not bulk. yfinance returns a MultiIndex column frame whose
shape depends on how many tickers were requested, so one symbol per call keeps a single
parsing path, maps one-to-one onto the per-symbol parquet cache, allows a partial cache
hit to fetch only what is missing, and confines a delisting or a bad ticker to one
symbol instead of failing the universe. Five daily requests cost nothing.

**On staleness.** A cached file that ends before today is returned unchanged rather than
topped up. Spec 7.4 caches the study's data to parquet precisely so the grid is run
against a fixed snapshot, and GB-59 requires a clean clone to reproduce reported results.
A loader that silently extended its data would make two runs of the same study on
different days disagree with no code change, and the disagreement would be invisible. So
refreshing is an explicit act — ``force_refresh=True`` — and every cache hit logs its
last bar, which makes staleness visible rather than silent. The live loop does not read
this module; it reads ``data/live.py``.

Implemented in GB-4.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from glassbox.config.loader import Config

LOGGER = logging.getLogger(__name__)

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
CLOSE = "close"
VOLUME = "volume"
LOG_RETURN = "log_return"

# Provenance travels with the bars as a column rather than in `frame.attrs`, because
# attrs are silently dropped by many pandas operations and are not stored by parquet,
# so they would produce false alarms. A column survives slicing, concat and the cache,
# and a spliced frame shows up directly as two distinct values. features/builder.py
# refuses to assemble a window from a frame whose provenance is missing or mixed.
SOURCE_COLUMN = "source"
SOURCE_YFINANCE = "yfinance"

# The canonical index resolution. yfinance via parquet lands on milliseconds, the Alpaca
# SDK on microseconds; the same trading day in two resolutions is two different dtypes.
INDEX_UNIT = "ms"


def load_history(
    symbols: Sequence[str],
    cfg: Config,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load daily bars for ``symbols``, from cache where available.

    Args:
        symbols: Tickers to load, in any order.
        cfg: Resolved configuration; supplies ``data.start`` and ``data.cache_dir``.
        force_refresh: Refetch every symbol and overwrite its cache file.

    Returns:
        ``{symbol: DataFrame}``, one entry per requested symbol.

    Raises:
        ValueError: A download returned no bars, or a frame is missing a required
            column. The message names the symbol.
    """
    cache_dir = Path(cfg.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    from_cache: list[str] = []
    fetched: list[str] = []

    for symbol in symbols:
        cache_path = cache_dir / f"{symbol}.parquet"

        if cache_path.is_file() and not force_refresh:
            frame = _read_cache(cache_path, symbol)
            frames[symbol] = frame
            from_cache.append(symbol)
            LOGGER.info(
                "%s: cache hit, %s bars, %s to %s",
                symbol,
                len(frame),
                _date_label(frame.index[0]),
                _date_label(frame.index[-1]),
            )
            continue

        frame = _fetch(symbol, cfg.data.start)
        frame.to_parquet(cache_path)
        frames[symbol] = frame
        fetched.append(symbol)
        LOGGER.info(
            "%s: fetched %s bars, %s to %s, cached to %s",
            symbol,
            len(frame),
            _date_label(frame.index[0]),
            _date_label(frame.index[-1]),
            cache_path,
        )

    LOGGER.info(
        "load_history: %d from cache (%s), %d fetched (%s)",
        len(from_cache),
        ", ".join(from_cache) or "none",
        len(fetched),
        ", ".join(fetched) or "none",
    )
    return frames


def _fetch(symbol: str, start: str) -> pd.DataFrame:
    """Download one symbol's daily bars and normalise them."""
    raw = yf.download(
        symbol,
        start=start,
        end=_end_exclusive(),
        auto_adjust=True,
        progress=False,
    )
    return normalise_bars(raw, symbol, SOURCE_YFINANCE)


def _end_exclusive() -> str:
    """Today plus one day: yfinance treats ``end`` as exclusive, so this includes today."""
    tomorrow = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
    return tomorrow.strftime("%Y-%m-%d")


def normalise_bars(raw: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
    """Turn a raw bar frame into the one schema the feature layer is allowed to see.

    This is the single definition of that schema. ``data/live.py`` calls it too, so the
    two sources cannot drift apart: identical columns, identical dtypes, identical index
    convention, by construction rather than by two implementations agreeing. GB-27's
    parity test depends on that.

    Accepts any frame carrying OHLCV columns in any case, with or without MultiIndex
    columns and with or without extra columns, and returns
    ``open, high, low, close, volume, log_return``.
    """
    if raw is None or raw.empty:
        raise ValueError(f"{symbol} returned no bars")

    frame = raw.copy()

    # yfinance 1.6 returns MultiIndex columns even for a single ticker; the price field
    # is the first level.
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]

    missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{symbol} is missing required columns: {missing}")

    frame = frame.loc[:, list(OHLCV_COLUMNS)]
    frame.index = _normalise_index(frame.index, symbol)

    # Keep the last observation of a repeated date: a later row is a correction.
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.sort_index()

    # Alpaca returns volume as float, yfinance as int. Pin it, or two sources with the
    # same numbers still fail a dtype comparison.
    frame[VOLUME] = frame[VOLUME].astype("int64")

    frame[LOG_RETURN] = np.log(frame[CLOSE] / frame[CLOSE].shift(1))
    frame[SOURCE_COLUMN] = pd.Series(source, index=frame.index, dtype="string")
    return frame


def _normalise_index(index: pd.Index, symbol: str) -> pd.DatetimeIndex:
    """Coerce an index to a tz-aware UTC DatetimeIndex, at midnight, in one resolution.

    Alpaca stamps a daily bar at midnight New York (04:00 or 05:00 UTC); yfinance stamps
    the bare date. Both mean the same trading day, so both are floored to midnight UTC.
    """
    converted = pd.DatetimeIndex(pd.to_datetime(index))
    if converted.tz is None:
        converted = converted.tz_localize("UTC")
    else:
        converted = converted.tz_convert("UTC")
    converted = converted.normalize().as_unit(INDEX_UNIT)
    # freq is part of an index's identity to assert_frame_equal, and a parquet roundtrip
    # can restore an inferred one. Pin it to None so every path agrees.
    converted.freq = None
    converted.name = "date"
    if converted.hasnans:
        raise ValueError(f"{symbol} has an unparseable date in its index")
    return converted


def _read_cache(path: Path, symbol: str) -> pd.DataFrame:
    """Read a cached frame and re-assert the index and provenance contracts."""
    frame = pd.read_parquet(path)
    frame.index = _normalise_index(frame.index, symbol)
    # Caches written before provenance existed carry no source column; this file is a
    # yfinance snapshot by construction, so stamping it is a statement of fact.
    if SOURCE_COLUMN not in frame.columns:
        frame[SOURCE_COLUMN] = pd.Series(
            SOURCE_YFINANCE, index=frame.index, dtype="string"
        )
    return frame


def _date_label(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%d")
