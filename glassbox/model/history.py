"""L3: the per-epoch record of a training run.

A separate module for one dataclass, because both ends need it and neither may import the
other: ``ltsf.py`` produces the record inside ``fit``, ``train.py`` writes it to disk, and
``train.py`` already imports ``ltsf`` through ``glassbox.model``. Putting the type in
either one would make the pair circular; putting it in ``contracts/schemas.py`` would
change a frozen contract to hold a reporting artefact.

The loss curve is **not** part of a checkpoint. A checkpoint says what a model is and what
it was fitted on; the curve says how the fitting went, which is a question for the report
(GB-57) and for judging whether ``patience`` is set sensibly. Keeping it in a sidecar file
also keeps every byte of the checkpoint reproducible.

Implemented in GB-15.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

HISTORY_HEADER = "epoch,train_loss,val_loss"


@dataclass(frozen=True)
class EpochLoss:
    """One epoch's losses.

    Both are measured over the **whole** split at the end of the epoch, not averaged over
    the mini-batches that produced them. An average of mini-batch losses is taken at
    weights that no longer exist by the epoch's end, so it is not comparable to the
    validation number beside it - and a curve whose two lines are measured differently is
    worse than no curve.

    ``val_loss`` is ``None`` when no validation split was supplied, which is also the case
    in which no early stopping can happen.
    """

    epoch: int  # 1-based
    train_loss: float
    val_loss: float | None


def write_history(path: str | Path, history: Sequence[EpochLoss]) -> Path:
    """Write the loss curve as CSV and return the path written.

    ``repr`` rather than a format string: a rounded loss curve cannot be used to check
    that two runs trained identically, which is one of the things this file is for.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [HISTORY_HEADER]
    lines.extend(
        f"{record.epoch},{record.train_loss!r},"
        f"{'' if record.val_loss is None else repr(record.val_loss)}"
        for record in history
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def read_history(path: str | Path) -> tuple[EpochLoss, ...]:
    """Read a loss curve back. Used by the report and by the round-trip test."""
    rows = Path(path).read_text(encoding="utf-8").splitlines()
    if not rows or rows[0] != HISTORY_HEADER:
        raise ValueError(
            f"{path} is not a loss curve; expected header {HISTORY_HEADER!r}"
        )
    return tuple(
        EpochLoss(
            epoch=int(epoch),
            train_loss=float(train_loss),
            val_loss=None if val_loss == "" else float(val_loss),
        )
        for epoch, train_loss, val_loss in (row.split(",") for row in rows[1:] if row)
    )


__all__ = ["HISTORY_HEADER", "EpochLoss", "read_history", "write_history"]
