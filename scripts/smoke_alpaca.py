"""Connectivity smoke test for the Alpaca paper account (GB-6).

Run from the repository root::

    python scripts/smoke_alpaca.py

Prints account equity, buying power, status and whether the endpoint is the paper one.
Refuses to run — non-zero exit, no connection attempted — if the resolved endpoint is
anything other than the paper endpoint.

Secrets are never printed, not even masked. Knowing that a key "starts with PK" is
still information about a key.
"""

from __future__ import annotations

import sys

from alpaca.trading.client import TradingClient

from glassbox.config.loader import (
    PAPER_ENDPOINT,
    alpaca_credentials,
    require_paper_endpoint,
)

EXIT_REFUSED = 2
EXIT_FAILED = 1


def main() -> int:
    try:
        credentials = alpaca_credentials()
    except ValueError as error:
        print(f"credentials: {error}", file=sys.stderr)
        return EXIT_FAILED

    # Fail closed before any network call: an endpoint that is not the paper endpoint
    # is a refusal, not a warning.
    try:
        require_paper_endpoint(credentials.base_url)
    except ValueError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return EXIT_REFUSED

    print(f"endpoint: {credentials.base_url}")

    try:
        client = TradingClient(
            api_key=credentials.api_key,
            secret_key=credentials.secret_key,
            paper=True,
        )
        account = client.get_account()
    except Exception as error:  # noqa: BLE001 - one message for any failure
        print(f"connection failed: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_FAILED

    print(f"account status:  {account.status}")
    print(f"paper account:   {credentials.base_url.rstrip('/') == PAPER_ENDPOINT}")
    print(f"equity:          {account.equity}")
    print(f"buying power:    {account.buying_power}")
    print(f"cash:            {account.cash}")
    print(f"trading blocked: {account.trading_blocked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
