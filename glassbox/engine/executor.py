"""L5: order submission to Alpaca paper. Auto and Co-Pilot.

The decision layer above it never talks to a broker; this module never makes a decision.
It receives a sized :class:`engine.risk.Order`, converts the notional through the one
conversion the backtester also uses, and submits.

**The broker is behind a protocol.** :class:`Broker` declares the five calls this project
needs - submit, cancel, positions, orders, account - and :class:`AlpacaBroker` is the only
implementation that touches the network. ``tests/fake_broker.py`` supplies the other, so
the suite never needs credentials, a network or a market session.

**Two parity findings, established against the live paper API in GB-22 rather than from
memory, both of which shape this module.**

1. **Alpaca refuses a bracket on a fractional quantity.** Verbatim, from the paper API:
   ``{"code":42210000,"message":"fractional orders must be simple orders"}`` for BRACKET,
   OCO and OTO alike, while the same bracket on **one whole share** is accepted with its
   legs. So the backtester's "entry carries a stop and a target" cannot be expressed as one
   order live, and this module attaches protection with **two standalone day orders**
   instead - which the API does accept against a held fractional position (measured).
   Three consequences follow, and they are limitations rather than details:

   - **No OCO linkage.** If the stop fills, the target is still live; the caller must
     cancel it. :func:`protect` returns both IDs so it can.
   - **Day only.** ``{"code":42210000,"message":"fractional orders must be DAY orders"}``,
     so protection **expires at every close and is re-established each session**.
   - **The residual divergence is smaller than "unprotected overnight", which is what an
     earlier version of this docstring claimed and is wrong.** The stop is a *fixed price*
     set at entry. A gap through it overnight leaves the re-armed stop immediately
     marketable at the open, so it fires at roughly the open - which is exactly GB-18's
     ``stop_gap`` rule. An intraday touch is covered because the stop is armed during the
     session. What genuinely diverges is (a) the seconds between the open and the arming,
     and (b) a cycle in which arming **fails and nothing notices** - the second being far
     the more dangerous. GB-26 carries the five-point policy that closes both; GB-57 states
     the residual in those terms and not as "the live system has no stop overnight".

2. **The minimum order size is a notional, not a share count.** See
   ``engine.risk.MIN_ORDER_NOTIONAL``; GB-18's ``MIN_SHARES = 0.001`` was the right idea in
   the wrong unit, and this module inherits the corrected rule by calling ``shares_for``.

**Modes.** ``live.mode: auto`` submits immediately. ``co_pilot`` returns a pending
recommendation and submits nothing - GB-37 adds the approval flow that turns one into the
other. The pending record carries everything a later approval needs, so approving is a
submission and not a re-derivation.

Every request and every response is logged against the **decision ID**, so a broker order
can be traced back to the forecast and the attribution that produced it. Credentials are
never logged.

Implemented in GB-22; Co-Pilot approval in GB-37.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from glassbox.config.loader import Config, alpaca_credentials, require_paper_endpoint
from glassbox.engine.risk import Order, shares_for

LOGGER = logging.getLogger(__name__)

AUTO = "auto"
CO_PILOT = "co_pilot"

BUY = "buy"
SELL = "sell"
DAY = "day"

SUBMITTED = "submitted"
PENDING_APPROVAL = "pending_approval"
REJECTED = "rejected"


@dataclass(frozen=True)
class BrokerOrder:
    """What the broker said about one order. The fields this project reads, and no more.

    ``raw`` keeps the broker's own object so a diagnosis is never blocked by this dataclass
    being too narrow, but nothing in the package may read it: anything the system depends on
    gets a named field first.
    """

    id: str
    symbol: str
    side: str
    quantity: float
    status: str
    filled_quantity: float = 0.0
    filled_price: float | None = None
    raw: Any = None


@runtime_checkable
class Broker(Protocol):
    """The five calls this project makes of a broker.

    Deliberately small. Every method a live session needs is here, so ``FakeBroker`` is a
    complete substitute rather than a partial one, and a test cannot pass because the fake
    quietly lacked the method the real path uses.
    """

    def submit_market_order(
        self, symbol: str, quantity: float, side: str, client_order_id: str
    ) -> BrokerOrder:
        """Submit a market order, ``time_in_force=day``, for a possibly fractional qty."""
        ...

    def submit_stop_order(
        self, symbol: str, quantity: float, stop_price: float, client_order_id: str
    ) -> BrokerOrder:
        """Submit a standalone stop **sell**. The fractional bracket substitute."""
        ...

    def submit_limit_order(
        self, symbol: str, quantity: float, limit_price: float, client_order_id: str
    ) -> BrokerOrder:
        """Submit a standalone limit **sell**. The take-profit half of the substitute."""
        ...

    def cancel_order(self, order_id: str) -> None: ...

    def get_orders(self) -> list[BrokerOrder]: ...

    def get_positions(self) -> dict[str, float]:
        """``{symbol: quantity}`` for every open position."""
        ...

    def get_account(self) -> dict[str, float]:
        """At least ``equity`` and ``cash``."""
        ...


@dataclass(frozen=True)
class Submission:
    """The outcome of one execution attempt, tied to the decision that caused it.

    ``entry`` is ``None`` in Co-Pilot mode and when the order was refused before reaching
    the broker; ``status`` says which. ``protection`` holds the stop and target orders that
    stand in for a bracket - empty when the entry did not fill or the mode is Co-Pilot.
    """

    decision_id: str
    order: Order
    status: str
    entry: BrokerOrder | None = None
    protection: tuple[BrokerOrder, ...] = ()
    reason: str | None = None
    log: tuple[str, ...] = field(default_factory=tuple)

    @property
    def submitted(self) -> bool:
        return self.status == SUBMITTED


def execute(broker: Broker, order: Order, decision_id: str, cfg: Config) -> Submission:
    """Act on one sized order, or decline to, according to ``live.mode``.

    Args:
        broker: Any :class:`Broker`. The live path passes :class:`AlpacaBroker`; tests pass
            ``FakeBroker`` and never touch the network.
        order: From ``engine.risk.size_positions``. Its ``notional`` and ``price`` are what
            reach the share conversion - **not** its ``shares`` field, so that the executor
            and the backtester derive the quantity by the same route from the same inputs.
        decision_id: The ``DecisionRecord`` this order belongs to. Every log line carries
            it, so a broker order can be traced back to the forecast that caused it.
        cfg: Resolved configuration; ``live.mode`` selects auto or Co-Pilot.

    Returns:
        A :class:`Submission`. In Co-Pilot mode its status is ``pending_approval`` and
        nothing was sent.
    """
    lines: list[str] = []

    def record(message: str) -> str:
        line = f"[{decision_id}] {message}"
        LOGGER.info(line)
        lines.append(line)
        return line

    quantity = shares_for(order.notional, order.price)
    record(
        f"sizing {order.symbol}: notional={order.notional:.2f} price={order.price:.4f} "
        f"-> quantity={quantity:.9f}"
    )

    if quantity <= 0.0:
        # Below the broker's notional minimum. Declining here rather than letting the API
        # refuse it keeps the reason in our own log, in our own words.
        return Submission(
            decision_id=decision_id,
            order=order,
            status=REJECTED,
            reason=(
                f"{order.symbol}: notional {order.notional:.2f} is below the broker's "
                f"minimum order size; no order was sent"
            ),
            log=tuple(lines),
        )

    if cfg.live.mode == CO_PILOT:
        record(
            f"co_pilot: recommending BUY {quantity:.9f} {order.symbol}, "
            f"stop={order.stop_loss:.4f} target={order.take_profit:.4f}; not submitted"
        )
        return Submission(
            decision_id=decision_id,
            order=order,
            status=PENDING_APPROVAL,
            reason="co_pilot mode: awaiting approval (GB-37)",
            log=tuple(lines),
        )

    record(
        f"submit MARKET BUY {quantity:.9f} {order.symbol} tif={DAY} "
        f"client_order_id={decision_id}"
    )
    entry = broker.submit_market_order(
        symbol=order.symbol,
        quantity=quantity,
        side=BUY,
        client_order_id=decision_id,
    )
    record(
        f"broker accepted entry id={entry.id} status={entry.status} "
        f"qty={entry.quantity:.9f} filled={entry.filled_quantity:.9f} "
        f"at={entry.filled_price}"
    )

    protection = protect(broker, order, entry, decision_id, record)
    return Submission(
        decision_id=decision_id,
        order=order,
        status=SUBMITTED,
        entry=entry,
        protection=protection,
        log=tuple(lines),
    )


def protect(
    broker: Broker,
    order: Order,
    entry: BrokerOrder,
    decision_id: str,
    record,
) -> tuple[BrokerOrder, ...]:
    """Attach a stop and a target to a **filled** entry, as two standalone day orders.

    The fractional bracket substitute. It runs only on the quantity actually filled: arming
    protection for shares the account does not hold would be a short sale, which Alpaca
    refuses on fractional quantities anyway
    (``{"message":"fractional orders cannot be sold short"}``, measured).

    Returns an empty tuple when nothing filled - which is the normal case for an order
    submitted after the close, where the fill happens at the next open and protection is
    armed by the next cycle rather than by this one.
    """
    if entry.filled_quantity <= 0.0:
        record(
            f"entry {entry.id} has not filled ({entry.status}); protection is not armed "
            "yet and must be armed by the cycle that sees the fill"
        )
        return ()

    protection: list[BrokerOrder] = []
    stop = broker.submit_stop_order(
        symbol=order.symbol,
        quantity=entry.filled_quantity,
        stop_price=order.stop_loss,
        client_order_id=f"{decision_id}-stop",
    )
    record(
        f"stop armed id={stop.id} at {order.stop_loss:.4f} (day order, expires at close)"
    )
    protection.append(stop)

    target = broker.submit_limit_order(
        symbol=order.symbol,
        quantity=entry.filled_quantity,
        limit_price=order.take_profit,
        client_order_id=f"{decision_id}-target",
    )
    record(
        f"target armed id={target.id} at {order.take_profit:.4f} (day order, expires at "
        "close). NOT linked to the stop: if one fills the other must be cancelled"
    )
    protection.append(target)
    return tuple(protection)


class AlpacaBroker:
    """The only thing here that touches the network.

    Refuses to construct against a non-paper endpoint, before any call is made. GB-6 put
    that check in the config layer; this repeats it at the point of use, because a broker
    object is exactly the thing that would otherwise carry a live endpoint into a session
    that believed it was paper.
    """

    def __init__(self, client: Any | None = None) -> None:
        if client is not None:
            self._client = client
            return
        credentials = alpaca_credentials()
        require_paper_endpoint(credentials.base_url)
        from alpaca.trading.client import TradingClient

        self._client = TradingClient(
            api_key=credentials.api_key,
            secret_key=credentials.secret_key,
            paper=True,
        )

    def submit_market_order(
        self, symbol: str, quantity: float, side: str, client_order_id: str
    ) -> BrokerOrder:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        return self._wrap(
            self._client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=OrderSide.BUY if side == BUY else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=client_order_id,
                )
            )
        )

    def submit_stop_order(
        self, symbol: str, quantity: float, stop_price: float, client_order_id: str
    ) -> BrokerOrder:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import StopOrderRequest

        return self._wrap(
            self._client.submit_order(
                StopOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    stop_price=round(stop_price, 2),
                    client_order_id=client_order_id,
                )
            )
        )

    def submit_limit_order(
        self, symbol: str, quantity: float, limit_price: float, client_order_id: str
    ) -> BrokerOrder:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest

        return self._wrap(
            self._client.submit_order(
                LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(limit_price, 2),
                    client_order_id=client_order_id,
                )
            )
        )

    def cancel_order(self, order_id: str) -> None:
        self._client.cancel_order_by_id(order_id)

    def get_orders(self) -> list[BrokerOrder]:
        return [self._wrap(order) for order in self._client.get_orders()]

    def get_positions(self) -> dict[str, float]:
        return {
            position.symbol: float(position.qty)
            for position in self._client.get_all_positions()
        }

    def get_account(self) -> dict[str, float]:
        account = self._client.get_account()
        return {"equity": float(account.equity), "cash": float(account.cash)}

    def refresh(self, order_id: str) -> BrokerOrder:
        """Re-read one order. The live loop polls this to see a fill."""
        return self._wrap(self._client.get_order_by_id(order_id))

    @staticmethod
    def _wrap(order: Any) -> BrokerOrder:
        return BrokerOrder(
            id=str(order.id),
            symbol=order.symbol,
            side=str(getattr(order.side, "value", order.side)),
            quantity=float(order.qty or 0.0),
            status=str(getattr(order.status, "value", order.status)),
            filled_quantity=float(order.filled_qty or 0.0),
            filled_price=(
                float(order.filled_avg_price)
                if order.filled_avg_price is not None
                else None
            ),
            raw=order,
        )


__all__ = [
    "AUTO",
    "BUY",
    "CO_PILOT",
    "DAY",
    "PENDING_APPROVAL",
    "REJECTED",
    "SELL",
    "SUBMITTED",
    "AlpacaBroker",
    "Broker",
    "BrokerOrder",
    "Submission",
    "execute",
    "protect",
]
