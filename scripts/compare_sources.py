"""Compare a same-day bar from yfinance and Alpaca, field by field (GB-7).

Run from the repository root::

    python scripts/compare_sources.py

Reports the absolute and relative difference per field for the most recent bar both
sources agree exists, and the worst disagreement across the overlapping window. Use it
to measure the tolerance rather than to invent one.
"""

from __future__ import annotations

try:
    import pandas as pd

    from glassbox.config.loader import load_config
    from glassbox.data import historical, live
except ImportError as error:  # pragma: no cover - depends on the caller's interpreter
    raise SystemExit(
        f"cannot import {error.name!r}: this script needs the project environment.\n"
        "    .venv\\Scripts\\activate      (Windows)\n"
        "    source .venv/bin/activate    (macOS, Linux)\n"
        'then, if the environment is new:  pip install -e ".[dev]"'
    ) from error

FIELDS = ("open", "high", "low", "close", "volume")


def main() -> int:
    cfg = load_config()

    offline = historical.load_history(cfg.universe, cfg)
    online = live.load_live_bars(cfg.universe, cfg)

    print(
        f"\n{'symbol':<8}{'field':<9}{'yfinance':>16}{'alpaca':>16}{'abs diff':>14}{'bps':>10}"
    )
    print("-" * 73)

    worst_bps: dict[str, float] = {}
    for symbol in cfg.universe:
        left, right = offline[symbol], online[symbol]
        shared = left.index.intersection(right.index)
        if shared.empty:
            print(f"{symbol:<8}no overlapping dates")
            continue

        latest = shared[-1]
        for field in FIELDS:
            a, b = float(left.loc[latest, field]), float(right.loc[latest, field])
            bps = abs(a - b) / abs(a) * 10_000 if a else float("nan")
            worst_bps[field] = max(worst_bps.get(field, 0.0), bps)
            print(
                f"{symbol:<8}{field:<9}{a:>16,.4f}{b:>16,.4f}{abs(a - b):>14,.4f}{bps:>10,.1f}"
            )
        print(f"{'':<8}(bar {latest:%Y-%m-%d}, {len(shared)} overlapping bars)")

    print("\n--- worst same-bar disagreement across the universe, in basis points ---")
    for field, bps in worst_bps.items():
        print(f"    {field:<8}{bps:>12,.1f} bps")

    print("\n--- close-price agreement across every overlapping bar ---")
    for symbol in cfg.universe:
        left, right = offline[symbol], online[symbol]
        shared = left.index.intersection(right.index)
        if shared.empty:
            continue
        diff = (left.loc[shared, "close"] - right.loc[shared, "close"]).abs()
        rel = (diff / left.loc[shared, "close"].abs() * 10_000).dropna()
        print(
            f"    {symbol:<7}{len(shared):>5} bars   "
            f"median {rel.median():>7.2f} bps   p95 {rel.quantile(0.95):>8.2f} bps   "
            f"max {rel.max():>9.2f} bps   on {pd.Timestamp(rel.idxmax()):%Y-%m-%d}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
