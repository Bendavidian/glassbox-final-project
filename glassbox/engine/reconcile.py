"""L5: the broker is the truth. Local state follows it, loudly.

**Reconcile-from-truth, not an order-lifecycle state machine.** ``SOLO_BUILD_PLAN.md`` §2
cut GB-23 to exactly this, and the cut is the right one: a state machine models what
*should* happen to an order, while this module asks the only question that prevents the
failure that matters — *is what the system believes still true?* A machine that tracked
every transition and still disagreed with the broker would be an elaborate way of being
wrong.

**Every divergence resolves in favour of the broker**, and every divergence is logged. The
broker holds the money and the shares; local state is a belief about them. When the two
disagree the belief is wrong by definition, whatever the reason.

**Three divergences, each with its own ruling.**

1. :data:`UNKNOWN_POSITION` — the broker holds something the system has no record of. It is
   **quarantined, never adopted**. The system has no entry price, no stop, no target and no
   decision behind it, so it can neither protect it nor explain it, and a system that cannot
   explain a position must not pretend to manage one. It is equally not ignored: it consumes
   buying power the sizer would otherwise believe is free, so it is recorded in
   :attr:`Book.unmanaged` and counted by :meth:`Book.committed`. Ignoring it would let the
   sizer over-commit the account; adopting it would put a position under management that no
   decision record can account for. Quarantine is the only option that is honest about both.
2. :data:`MISSING_POSITION` — the system believes it holds something the broker does not.
   The holding is **dropped**. This is not only an error case: it is exactly what a filled
   stop or target looks like from here, which is why it is resolved rather than raised.
3. :data:`QUANTITY_MISMATCH` — both agree the position exists and disagree about its size.
   **The broker's number wins**, and the local provenance is kept. That is how a partial
   fill is absorbed without a lifecycle machine to model it.

A fourth check is **detection only**: :data:`MISSING_PROTECTION` reports a managed holding
with no live stop or no live limit at the broker. GB-26's rule 3 decides what to do about it
— arm immediately, and flatten if arming fails twice. Nothing here acts on it, because a
reconciler that started cancelling and submitting would be the state machine this task
deliberately does not build.

The book is persisted as JSON so a restart can compare against what the previous process
believed. Its path is supplied by the caller rather than read from config: GB-26 owns where
the live loop keeps its state, and inventing a config key here would pre-empt that.

Implemented in GB-23.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from glassbox.engine.executor import SELL, Broker, BrokerOrder

LOGGER = logging.getLogger(__name__)

UNKNOWN_POSITION = "unknown_position"
MISSING_POSITION = "missing_position"
QUANTITY_MISMATCH = "quantity_mismatch"
MISSING_PROTECTION = "missing_protection"

# Quantities are compared to the precision the broker stores them at (GB-22 measured nine
# decimals), so a float representation difference is never reported as a divergence.
QUANTITY_TOLERANCE = 5e-10


@dataclass(frozen=True)
class Holding:
    """A position the system manages, and the decision that created it.

    Every field except ``quantity`` comes from the decision that opened the position and is
    never revised by reconciliation - the broker is the truth about *what is held*, not
    about *why*.
    """

    symbol: str
    quantity: float
    decision_id: str
    entry_price: float
    stop_loss: float
    take_profit: float


@dataclass
class Book:
    """What the system believes it holds. Persisted, so a restart can compare.

    ``managed`` holds positions with a decision behind them. ``unmanaged`` holds quantities
    the broker reports that no decision explains - quarantined, counted, never traded.
    """

    managed: dict[str, Holding] = field(default_factory=dict)
    unmanaged: dict[str, float] = field(default_factory=dict)

    def committed(self, symbol: str) -> float:
        """Every share of ``symbol`` the account holds, managed or not.

        The sizer must see this rather than :attr:`managed` alone: a quarantined position
        is not free capital just because the system did not choose it.
        """
        managed = self.managed[symbol].quantity if symbol in self.managed else 0.0
        return managed + self.unmanaged.get(symbol, 0.0)

    def symbols(self) -> set[str]:
        return set(self.managed) | set(self.unmanaged)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "managed": {
                        symbol: asdict(holding)
                        for symbol, holding in sorted(self.managed.items())
                    },
                    "unmanaged": dict(sorted(self.unmanaged.items())),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> Book:
        """Read a book, or return an empty one if there is nothing there.

        A missing file is the first run, not an error. A **corrupt** file is an error and
        raises: continuing from a book that cannot be parsed would mean continuing from no
        belief at all while appearing to continue from one.
        """
        source = Path(path)
        if not source.is_file():
            return cls()
        raw = json.loads(source.read_text(encoding="utf-8"))
        return cls(
            managed={
                symbol: Holding(**fields)
                for symbol, fields in raw.get("managed", {}).items()
            },
            unmanaged=dict(raw.get("unmanaged", {})),
        )


@dataclass(frozen=True)
class Divergence:
    """One disagreement between belief and truth, and how it was resolved."""

    kind: str
    symbol: str
    local: float
    broker: float
    detail: str

    def __str__(self) -> str:
        return (
            f"{self.kind}: {self.symbol} local={self.local:.9f} "
            f"broker={self.broker:.9f} - {self.detail}"
        )


@dataclass(frozen=True)
class Reconciliation:
    """The corrected book and everything that had to be corrected to get it."""

    book: Book
    divergences: tuple[Divergence, ...]

    @property
    def clean(self) -> bool:
        return not self.divergences

    def of_kind(self, kind: str) -> tuple[Divergence, ...]:
        return tuple(d for d in self.divergences if d.kind == kind)


def reconcile(broker: Broker, book: Book) -> Reconciliation:
    """Fetch the truth, correct the belief, and say what changed.

    Args:
        broker: Any :class:`executor.Broker`. Only its read calls are used - this module
            never submits or cancels anything, which is what keeps it a reconciler.
        book: What the system believed at the end of the last cycle. Not mutated; the
            corrected copy is returned.

    Returns:
        A :class:`Reconciliation` whose ``book`` matches the broker's positions exactly and
        whose ``divergences`` record every correction, in symbol order.
    """
    positions = broker.get_positions()
    orders = broker.get_orders()

    managed: dict[str, Holding] = {}
    unmanaged: dict[str, float] = {}
    divergences: list[Divergence] = []

    for symbol in sorted(book.symbols() | set(positions)):
        held = float(positions.get(symbol, 0.0))
        believed = book.committed(symbol)
        holding = book.managed.get(symbol)

        if held <= 0.0:
            if believed > 0.0:
                divergences.append(
                    Divergence(
                        kind=MISSING_POSITION,
                        symbol=symbol,
                        local=believed,
                        broker=0.0,
                        detail=(
                            "the broker holds nothing; dropping the local holding. A "
                            "filled stop or target looks exactly like this"
                        ),
                    )
                )
            continue

        if holding is None:
            # Quarantine. Not adopted - there is no decision behind it - and not ignored,
            # because it is spending buying power either way.
            unmanaged[symbol] = held
            if abs(held - book.unmanaged.get(symbol, 0.0)) > QUANTITY_TOLERANCE:
                divergences.append(
                    Divergence(
                        kind=UNKNOWN_POSITION,
                        symbol=symbol,
                        local=book.unmanaged.get(symbol, 0.0),
                        broker=held,
                        detail=(
                            "the broker holds a position with no decision behind it; "
                            "quarantined, counted against buying power, never traded. "
                            "The system cannot protect or explain what it did not open"
                        ),
                    )
                )
            continue

        if abs(held - holding.quantity) > QUANTITY_TOLERANCE:
            divergences.append(
                Divergence(
                    kind=QUANTITY_MISMATCH,
                    symbol=symbol,
                    local=holding.quantity,
                    broker=held,
                    detail=(
                        "taking the broker's quantity; a partial fill is absorbed here "
                        "rather than modelled by a lifecycle machine"
                    ),
                )
            )
        managed[symbol] = replace(holding, quantity=held)

    divergences.extend(_unprotected(managed, orders))

    for divergence in divergences:
        LOGGER.warning("reconcile: %s", divergence)

    return Reconciliation(
        book=Book(managed=managed, unmanaged=unmanaged),
        divergences=tuple(divergences),
    )


def _unprotected(
    managed: dict[str, Holding], orders: list[BrokerOrder]
) -> list[Divergence]:
    """Managed holdings with no live protective order. **Detection only.**

    GB-26's rule 3 decides what happens next - arm immediately, flatten if arming fails
    twice. Acting here would make this module the state machine GB-23 was scoped away from,
    and would put order submission in the one place that is supposed to only ever read.
    """
    live: dict[str, int] = {}
    for order in orders:
        if order.side == SELL and order.status not in _FINISHED:
            live[order.symbol] = live.get(order.symbol, 0) + 1

    return [
        Divergence(
            kind=MISSING_PROTECTION,
            symbol=symbol,
            local=holding.quantity,
            broker=float(live.get(symbol, 0)),
            detail=(
                "a managed position has fewer than two live protective orders; GB-26 "
                "rule 3 arms immediately and flattens if arming fails twice"
            ),
        )
        for symbol, holding in sorted(managed.items())
        if live.get(symbol, 0) < 2
    ]


# Statuses that mean an order is no longer working. Alpaca's spellings.
_FINISHED = frozenset(
    {
        "filled",
        "canceled",
        "cancelled",
        "expired",
        "rejected",
        "done_for_day",
        "replaced",
    }
)


__all__ = [
    "MISSING_POSITION",
    "MISSING_PROTECTION",
    "QUANTITY_MISMATCH",
    "QUANTITY_TOLERANCE",
    "UNKNOWN_POSITION",
    "Book",
    "Divergence",
    "Holding",
    "Reconciliation",
    "reconcile",
]
