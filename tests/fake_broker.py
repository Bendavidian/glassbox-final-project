"""A complete in-memory Broker. The suite never needs the network, credentials or a market.

It lives in ``tests/`` rather than in the package for the reason ``causality.py`` does: the
live path must not be able to import a fake broker. ``pythonpath = ["tests"]`` makes it
importable from any test, and GB-26's live-loop tests and GB-37's approval tests will use
the same one — a second fake would be a second definition of what a broker does.

**It refuses what Alpaca refuses.** The rejections are the ones GB-22 measured against the
paper API, verbatim, because a fake that accepts everything proves only that the code runs:

- a notional below $1.00 → ``cost basis must be >= minimal amount of order 1``
- a fractional quantity with anything but a simple order → ``fractional orders must be
  simple orders``
- selling more than is held → ``fractional orders cannot be sold short``
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from glassbox.engine.executor import BUY, SELL, BrokerOrder

MIN_NOTIONAL = 1.00


class FakeBrokerError(RuntimeError):
    """What the fake raises where Alpaca returns an APIError."""


@dataclass
class FakeBroker:
    """An account, its positions and its orders, held in memory.

    Args:
        prices: ``{symbol: price}``, used to value an order and to decide a fill.
        fill: When True a market order fills immediately at the price. When False it is
            accepted and left unfilled — the after-the-close case, where the fill happens
            at the next open and this cycle must not pretend otherwise.
        equity: Starting account value.
    """

    prices: dict[str, float] = field(default_factory=dict)
    fill: bool = True
    equity: float = 100_000.0
    cash: float = 100_000.0

    orders: list[BrokerOrder] = field(default_factory=list)
    positions: dict[str, float] = field(default_factory=dict)
    cancelled: list[str] = field(default_factory=list)
    submitted_kinds: list[str] = field(default_factory=list)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    # ── the Broker protocol ──────────────────────────────────────────────────

    def submit_market_order(
        self, symbol: str, quantity: float, side: str, client_order_id: str
    ) -> BrokerOrder:
        self.submitted_kinds.append("market")
        price = self._price(symbol)
        self._require_min_notional(quantity * price)
        if side == SELL:
            self._require_held(symbol, quantity)

        filled = quantity if self.fill else 0.0
        if filled:
            step = filled if side == BUY else -filled
            self.positions[symbol] = self.positions.get(symbol, 0.0) + step
            self.cash -= step * price
        return self._record(
            symbol=symbol,
            side=side,
            quantity=quantity,
            status="filled" if filled else "accepted",
            filled_quantity=filled,
            filled_price=price if filled else None,
            client_order_id=client_order_id,
        )

    def submit_stop_order(
        self, symbol: str, quantity: float, stop_price: float, client_order_id: str
    ) -> BrokerOrder:
        self.submitted_kinds.append("stop")
        self._require_held(symbol, quantity)
        return self._record(
            symbol=symbol,
            side=SELL,
            quantity=quantity,
            status="new",
            client_order_id=client_order_id,
        )

    def submit_limit_order(
        self, symbol: str, quantity: float, limit_price: float, client_order_id: str
    ) -> BrokerOrder:
        self.submitted_kinds.append("limit")
        self._require_held(symbol, quantity)
        return self._record(
            symbol=symbol,
            side=SELL,
            quantity=quantity,
            status="accepted",
            client_order_id=client_order_id,
        )

    def cancel_order(self, order_id: str) -> None:
        self.cancelled.append(order_id)

    def get_orders(self) -> list[BrokerOrder]:
        return list(self.orders)

    def get_positions(self) -> dict[str, float]:
        return dict(self.positions)

    def get_account(self) -> dict[str, float]:
        return {"equity": self.equity, "cash": self.cash}

    # ── the refusals, as measured against the real API ───────────────────────

    def _require_min_notional(self, notional: float) -> None:
        if notional < MIN_NOTIONAL:
            raise FakeBrokerError(
                f"cost basis must be >= minimal amount of order {MIN_NOTIONAL:.0f}"
            )

    def _require_held(self, symbol: str, quantity: float) -> None:
        if quantity > self.positions.get(symbol, 0.0) + 1e-12:
            raise FakeBrokerError("fractional orders cannot be sold short")

    def _price(self, symbol: str) -> float:
        if symbol not in self.prices:
            raise FakeBrokerError(f"no price for {symbol}")
        return self.prices[symbol]

    def _record(
        self,
        symbol: str,
        side: str,
        quantity: float,
        status: str,
        client_order_id: str,
        filled_quantity: float = 0.0,
        filled_price: float | None = None,
    ) -> BrokerOrder:
        order = BrokerOrder(
            id=f"fake-{next(self._ids)}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            status=status,
            filled_quantity=filled_quantity,
            filled_price=filled_price,
            raw={"client_order_id": client_order_id},
        )
        self.orders.append(order)
        return order
