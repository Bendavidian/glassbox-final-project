"""L7: recorded-day playback, fully offline.

Replay is the demo path when the broker or the network is unavailable — and since
2026-08-18 it carries two GATE 2 items that nothing else can, because the deployed band
stands aside and a live session can therefore produce no recommendation to approve.

**It drives the live loop's own cycle, not a copy of it.** ``live_loop.run_cycle`` takes
its bars from an injected source, so replay supplies cached history instead of the SIP
feed and every other step — the builder, the forecast, the band, the ranker, the sizer,
the attribution, the narration, the decision record — is the identical code. A replay that
reimplemented the loop would prove the reimplementation works, which is the one thing
nobody needs to know.

**Every record it writes says so.** ``DecisionRecord.provenance`` reads
``replay:fold-13``, naming the fold rather than setting a boolean, so a replayed decision
is legible as one in the log, the dashboard and the report. ``records.load_decisions``
defaults to live, so a reader who forgets to filter loses sight of replayed decisions and
can never be shown one as real.

**``ReplayBroker`` is a demo instrument and not a backtester, and the difference matters.**
It fills a market order at the bar's open and checks the protective legs against the bar's
range, stop before target — the same rulings ``backtest.engine`` implements — but it does
**not** model gaps, slippage, fees or the accounting identity, and this module may not
import the harness that does. **So a PnL produced by replay is not a result and must never
be reported as one.** What replay demonstrates is that a decision can be made, explained,
approved or declined and recorded; what the strategy earns is GB-18's question and is
answered there.

Implemented in GB-38.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd

from glassbox import live_loop, records
from glassbox.config.loader import VALID_MODELS, Config, load_config
from glassbox.data.historical import load_history
from glassbox.engine.executor import SELL, BrokerOrder
from glassbox.engine.reconcile import Book
from glassbox.model.predict import load_predictor

LOGGER = logging.getLogger("glassbox.replay")

FILLED = "filled"
NEW = "new"
CANCELLED = "canceled"

MANIFEST_FILE = "replay.json"


class ReplayError(RuntimeError):
    """Something replay cannot start without. Reported without a traceback."""


# ── the offline broker ───────────────────────────────────────────────────────


@dataclass
class ReplayBroker:
    """An account driven by historical bars. **A demo instrument, not a backtester.**

    See the module docstring: it fills at the open and checks stops before targets, and it
    models neither gaps nor frictions. Its equity is a number that lets sizing run, not a
    result.
    """

    equity: float = 100_000.0
    cash: float = 100_000.0
    orders: list[BrokerOrder] = field(default_factory=list)
    positions: dict[str, float] = field(default_factory=dict)
    cancelled: list[str] = field(default_factory=list)
    prices: dict[str, float] = field(default_factory=dict)
    fills: list[str] = field(default_factory=list)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    # ── the Broker protocol ──────────────────────────────────────────────────

    def submit_market_order(
        self, symbol: str, quantity: float, side: str, client_order_id: str
    ) -> BrokerOrder:
        price = self._price(symbol)
        if side == SELL:
            self.positions[symbol] = max(
                0.0, self.positions.get(symbol, 0.0) - quantity
            )
            self.cash += quantity * price
        else:
            self.positions[symbol] = self.positions.get(symbol, 0.0) + quantity
            self.cash -= quantity * price
        order = self._record(
            symbol, side, quantity, FILLED, client_order_id, quantity, price
        )
        self.fills.append(f"{side} {quantity:.9f} {symbol} at {price:.4f}")
        return order

    def submit_stop_order(
        self, symbol: str, quantity: float, stop_price: float, client_order_id: str
    ) -> BrokerOrder:
        order = self._record(symbol, SELL, quantity, NEW, client_order_id)
        object.__setattr__(
            order, "raw", {**order.raw, "level": stop_price, "leg": "stop"}
        )
        return order

    def submit_limit_order(
        self, symbol: str, quantity: float, limit_price: float, client_order_id: str
    ) -> BrokerOrder:
        order = self._record(symbol, SELL, quantity, NEW, client_order_id)
        object.__setattr__(
            order, "raw", {**order.raw, "level": limit_price, "leg": "target"}
        )
        return order

    def cancel_order(self, order_id: str) -> None:
        self.cancelled.append(order_id)
        self._replace(order_id, status=CANCELLED)

    def get_orders(self) -> list[BrokerOrder]:
        return list(self.orders)

    def get_positions(self) -> dict[str, float]:
        return {s: q for s, q in self.positions.items() if q > 0.0}

    def get_account(self) -> dict[str, float]:
        held = sum(q * self.prices.get(s, 0.0) for s, q in self.positions.items())
        return {"equity": self.cash + held, "cash": self.cash}

    # ── the bar clock ────────────────────────────────────────────────────────

    def advance(self, bar: dict[str, dict[str, float]]) -> list[str]:
        """Mark to a new bar and resolve any protective leg its range touched.

        **Stop before target**, which is GB-18's ruling 1 and the conservative reading:
        daily OHLC cannot say which came first, so the assumption that cannot be accused of
        flattering is the one taken.
        """
        self.prices = {symbol: float(row["close"]) for symbol, row in bar.items()}
        resolved: list[str] = []
        for order in list(self.orders):
            if order.status != NEW or order.symbol not in bar:
                continue
            level = order.raw.get("level")
            leg = order.raw.get("leg")
            if level is None:
                continue
            row = bar[order.symbol]
            touched = (
                float(row["low"]) <= level
                if leg == "stop"
                else float(row["high"]) >= level
            )
            if not touched:
                continue
            quantity = min(order.quantity, self.positions.get(order.symbol, 0.0))
            if quantity <= 0.0:
                continue
            self.positions[order.symbol] = (
                self.positions.get(order.symbol, 0.0) - quantity
            )
            self.cash += quantity * level
            self._replace(
                order.id,
                status=FILLED,
                filled_quantity=quantity,
                filled_price=level,
            )
            resolved.append(f"{leg} filled {order.symbol} at {level:.4f}")
        return resolved

    # ── internals ────────────────────────────────────────────────────────────

    def _price(self, symbol: str) -> float:
        if symbol not in self.prices:
            raise ReplayError(f"no replay price for {symbol}")
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
            id=f"replay-{next(self._ids)}",
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

    def _replace(self, order_id: str, **changes: Any) -> None:
        for index, order in enumerate(self.orders):
            if order.id == order_id:
                fields = {
                    "id": order.id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "status": order.status,
                    "filled_quantity": order.filled_quantity,
                    "filled_price": order.filled_price,
                    "raw": order.raw,
                }
                fields.update(changes)
                self.orders[index] = BrokerOrder(**fields)
                return


# ── the driver ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReplayReport:
    """What a replay did. Its PnL is deliberately absent — see the module docstring."""

    fold: int
    provenance: str
    bars: tuple[pd.Timestamp, ...]
    cycles: tuple[live_loop.CycleReport, ...]
    fills: tuple[str, ...]
    pending: int

    def summary(self) -> str:
        failed = [cycle for cycle in self.cycles if not cycle.ok]
        return "\n".join(
            [
                f"replay of fold {self.fold} ({self.provenance})",
                f"  bars replayed        : {len(self.bars)}"
                + (
                    ""
                    if not self.bars
                    else f"  {self.bars[0]:%Y-%m-%d} to {self.bars[-1]:%Y-%m-%d}"
                ),
                f"  cycles completed     : {len(self.cycles)}, {len(failed)} failed",
                f"  decisions recorded   : {sum(len(c.decisions) for c in self.cycles)}",
                f"  broker fills         : {len(self.fills)}",
                f"  awaiting approval    : {self.pending}",
            ]
        )


def replay_fold(
    cfg: Config,
    state_dir: str | Path,
    *,
    language: str = live_loop.EN,
    max_bars: int | None = None,
    log=lambda message: None,
) -> ReplayReport:
    """Drive the live loop over one fold's test range, entirely offline.

    ``state_dir`` is a directory written by ``smoke_offline --prepare-replay``: a
    checkpoint trained on the fold's training split, the band that fold's **validation**
    split calibrated, and a manifest naming the fold and its test range. Replay loads them
    and never trains — this module may not import the harness that does, which is the same
    constraint that shaped GB-26 and the same answer.
    """
    root = Path(state_dir)
    manifest = _manifest(root)
    fold = int(manifest["fold"])
    provenance = records.replay_provenance(fold)

    frames_source = load_history(sorted(cfg.universe), cfg)
    bars = {
        symbol: frame.loc[: manifest["test_end"]]
        for symbol, frame in frames_source.items()
    }
    session_dates = pd.DatetimeIndex(
        [pd.Timestamp(value) for value in manifest["test_index"]]
    )
    if max_bars is not None:
        session_dates = session_dates[:max_bars]

    broker = ReplayBroker()
    state = live_loop.LiveState(
        cfg=cfg,
        broker=broker,
        predictor=load_predictor(root / live_loop.CHECKPOINT_DIR, cfg),
        thresholds=live_loop.load_thresholds(root / live_loop.THRESHOLDS_FILE),
        state_dir=root,
        language=language,
        provenance=provenance,
        book=Book(),
    )

    cycles: list[live_loop.CycleReport] = []
    for when in session_dates:
        bar = {
            symbol: frame.loc[when].to_dict()
            for symbol, frame in bars.items()
            if when in frame.index
        }
        if not bar:
            continue
        for message in broker.advance(bar):
            log(f"{when:%Y-%m-%d} {message}")

        # The window must end on this bar, so the loop sees the history a live session
        # would have seen on that date and nothing after it. Slicing here rather than
        # inside the loop keeps `run_cycle` unaware that it is being replayed.
        state.fetch = _sliced(bars, when)
        cycles.append(live_loop.run_cycle(state, _during_session(when)))

    return ReplayReport(
        fold=fold,
        provenance=provenance,
        bars=tuple(session_dates),
        cycles=tuple(cycles),
        fills=tuple(broker.fills),
        pending=len(records.load_pending(root)),
    )


def _sliced(bars: dict[str, pd.DataFrame], when: pd.Timestamp):
    """A bar source frozen at ``when``. The loop drops the in-progress bar itself."""

    def fetch() -> dict[str, pd.DataFrame]:
        return {symbol: frame.loc[:when] for symbol, frame in bars.items()}

    return fetch


def _during_session(when: pd.Timestamp) -> pd.Timestamp:
    """A timestamp inside that date's session, so the loop's market checks behave.

    14:00 UTC is 10:00 in New York on every date the NYSE is open, standard time or
    daylight — the loop asks the calendar for the real boundaries and only needs a moment
    that falls between them.
    """
    return pd.Timestamp(when).normalize() + pd.Timedelta(hours=14)


def _manifest(root: Path) -> dict[str, Any]:
    import json

    path = root / MANIFEST_FILE
    if not path.is_file():
        raise ReplayError(
            f"no replay manifest at {path}. Produce one with: python -m "
            "glassbox.smoke_offline --prepare-replay <fold> <dir>"
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m glassbox.replay",
        description=(
            "Drive the live loop's cycle over one fold's test range, offline. Reads a "
            "directory written by 'smoke_offline --prepare-replay'."
        ),
    )
    parser.add_argument("--state-dir", default="checkpoints/replay")
    parser.add_argument("--language", choices=("en", "he"), default="en")
    parser.add_argument("--max-bars", type=int, default=None)
    parser.add_argument(
        "--model",
        choices=VALID_MODELS,
        default=None,
        help=(
            "which arm the checkpoint under --state-dir was trained as. Needed whenever "
            "that is not the arm the deployed configuration names: the model-shaping "
            "hash includes `model`, so a FITS checkpoint is correctly refused under a "
            "DLinear config. Replay is the demonstration path, and demonstrating an arm "
            "the system does NOT deploy is one of the things it is for (default: the "
            "configured arm)"
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        stream=sys.stdout,
    )
    try:
        cfg = load_config()
        if args.model is not None and args.model != cfg.model.active:
            # Stated out loud rather than applied quietly. A replay of an arm the system
            # does not deploy is a legitimate demonstration and a misleading screenshot
            # in equal measure, so the run says which arm it is before it produces one.
            print(
                f"replay: reading this checkpoint as {args.model.upper()}, which is NOT "
                f"the deployed arm ({cfg.model.active.upper()})"
            )
            cfg = replace(cfg, model=replace(cfg.model, active=args.model))
        report = replay_fold(
            cfg,
            args.state_dir,
            language=args.language,
            max_bars=args.max_bars,
            log=print,
        )
    except (ReplayError, FileNotFoundError, ValueError) as failure:
        print(f"replay: {failure}", file=sys.stderr)
        return 2

    print()
    print(report.summary())
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI
    raise SystemExit(main())


__all__ = [
    "MANIFEST_FILE",
    "ReplayBroker",
    "ReplayError",
    "ReplayReport",
    "main",
    "replay_fold",
]
