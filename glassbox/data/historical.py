"""L1: historical daily bars from yfinance, cached to parquet.

Returns a tz-aware UTC DatetimeIndex. A second call for the same range hits the cache.

Implemented in GB-4.
"""
