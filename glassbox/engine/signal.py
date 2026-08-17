"""L4: forecast path to a signal - enter_long, hold or exit.

Thresholds are calibrated on the validation slice of each fold and are never hardcoded.
This is where leakage hides; see spec 7.1.

**This module holds no numbers.** Not one threshold magnitude appears in the source: the
band arrives as a :class:`Thresholds` argument, the confirmation count comes from
``cfg.signal.min_up_points``, and ``tests/engine/test_signal.py`` walks the syntax tree to
prove it. ``signal.min_trend`` and ``signal.max_trend`` are ``null`` in the config file for
the same reason - a number there would be a threshold chosen by a human looking at
results, which is the quiet form of the leak GB-25 audits for.

Calibration itself lives in ``backtest/calibrate.py``, one layer up, and NOT here. The
ruling that it must run the real backtester rather than a proxy puts it above the engine
layer, and importing ``backtest`` from here would break both the layer contract and the
rule that the live path never imports the validation harness. The split is deliberate:
this module decides, the harness above it chooses the band, and the live loop carries a
band that was chosen offline.

**The three verdicts.**

- ``enter_long`` - the cumulative forecast sits inside the calibrated band AND at least
  ``min_up_points`` of its steps point up.
- ``exit`` - the forecast has turned down by as much conviction as it took to get in, i.e.
  ``trend_strength <= -lower``. The **mirror** of the entry threshold rather than a bare
  zero: zero would be a number in this module, and "any forecast weakness at all" would
  churn a position out on noise the entry rule would not have acted on.
- ``hold`` - everything else, including a forecast **stronger** than ``upper``. An upper
  bound exists because a forecast far outside the range the model produced in validation
  is more likely a broken input than an opportunity, and the honest response to an
  implausible number is to do nothing.

The confirmation is asymmetric on purpose: ``min_up_points`` gates entries and not exits.
Getting out must be easier than getting in, because a position already carries risk.

Implemented in GB-20.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from glassbox.config.loader import Config
from glassbox.contracts.schemas import Forecast, Signal

# The verdict vocabulary. It lives here, in the layer that produces the verdicts, and
# `backtest/engine.py` imports it rather than repeating the strings: two spellings of
# "enter_long" in two modules is the GB-7 failure family - both sides keep their tests and
# the system stops trading.
ENTER_LONG = "enter_long"
HOLD = "hold"
EXIT = "exit"

ACTIONS = (ENTER_LONG, HOLD, EXIT)


@dataclass(frozen=True)
class Thresholds:
    """The trend band for one fold, calibrated on that fold's validation split.

    ``lower`` is the cumulative forecast a window must reach to be worth entering, and
    ``-lower`` is the level at which a held position is closed. ``upper`` is the point
    beyond which a forecast is treated as implausible rather than attractive; ``None``
    means no upper bound was selected.

    ``lower`` must be **positive**: a non-positive entry threshold would enter on a
    forecast of no move at all, and would make the entry and exit conditions overlap, so
    one window could be both. :meth:`never` is the one exception, and it is infinite rather
    than zero.
    """

    lower: float
    upper: float | None = None

    def __post_init__(self) -> None:
        # `0`, not `0.0`: the sign boundary, and the test that walks this module's syntax
        # tree refuses a float literal outright rather than judging which ones are levels.
        if math.isnan(self.lower) or self.lower <= 0:
            raise ValueError(
                f"Thresholds.lower must be positive (or infinite, for a fold that stands "
                f"aside), got {self.lower!r}"
            )
        if self.upper is not None:
            if not math.isfinite(self.upper):
                raise ValueError(
                    f"Thresholds.upper must be finite or None, got {self.upper!r}"
                )
            if self.upper < self.lower:
                raise ValueError(
                    f"Thresholds.upper ({self.upper}) is below lower ({self.lower}), so "
                    "the band is empty and no forecast could ever pass it"
                )

    @classmethod
    def never(cls) -> Thresholds:
        """A band nothing can pass: the fold stands aside and trades nothing.

        Returned by :func:`backtest.calibrate.calibrate_thresholds` when no candidate on
        the grid earned a positive validation Sharpe. An infinite lower bound is refused by
        no forecast and, mirrored, an exit at ``-inf`` never fires either - so the fold
        holds nothing and its flat curve is an honest report rather than a rule validation
        rejected.
        """
        return cls(lower=math.inf, upper=None)

    @property
    def fires(self) -> bool:
        """True if any forecast could pass this band. False for :meth:`never`."""
        return math.isfinite(self.lower)


def trend_strength(path: np.ndarray) -> float:
    """Cumulative predicted return across the forecast path.

    The sum of the predicted log returns, which is the log return over the whole horizon -
    exactly the quantity the direction metric scores, so the signal layer and the report
    are reading the same number.

    Raises:
        ValueError: the path is empty or carries a non-finite value. A NaN forecast that
            silently became ``hold`` would look like a considered decision.
    """
    values = _require_path(path)
    return float(values.sum())


def up_points(path: np.ndarray) -> int:
    """How many steps of the path point up.

    The confirmation term: a total can be carried by one large step, and a path that rises
    on most of its days is a different object from one that falls for four days and jumps
    on the fifth. Compared against ``cfg.signal.min_up_points``.
    """
    values = _require_path(path)
    return int((values > 0).sum())


def decide(forecast: Forecast, thresholds: Thresholds, cfg: Config) -> Signal:
    """Turn one forecast into one verdict.

    Args:
        forecast: One symbol's predicted path, as of one bar's close.
        thresholds: The band calibrated on this fold's validation split. Never read from
            the config - ``signal.min_trend`` and ``signal.max_trend`` are ``null``.
        cfg: Resolved configuration; supplies ``signal.min_up_points`` only.

    Returns:
        A :class:`Signal`. ``passed_threshold`` records whether the **band** admitted the
        forecast, independently of the up-points confirmation, so a decision record can
        distinguish "too weak to act on" from "strong enough but the path disagreed with
        itself".
    """
    strength = trend_strength(forecast.path)
    points = up_points(forecast.path)

    in_band = strength >= thresholds.lower and (
        thresholds.upper is None or strength <= thresholds.upper
    )
    confirmed = points >= cfg.signal.min_up_points

    if in_band and confirmed:
        action = ENTER_LONG
    elif strength <= -thresholds.lower:
        action = EXIT
    else:
        action = HOLD

    return Signal(
        symbol=forecast.symbol,
        action=action,
        trend_strength=strength,
        up_points=points,
        passed_threshold=in_band,
    )


def decide_all(
    forecasts: Iterable[Forecast], thresholds: Thresholds, cfg: Config
) -> Mapping[pd.Timestamp, tuple[Signal, ...]]:
    """Every forecast decided, grouped by the bar it was made on.

    The shape ``backtest.engine.run_backtest`` consumes, and the shape GB-26's live loop
    produces for a single timestamp. One function builds it for both, so a backtest and a
    live session cannot disagree about what the decision layer said.

    Signals within a timestamp are ordered by symbol, so a run is reproducible regardless
    of the order forecasts arrived in.
    """
    grouped: dict[pd.Timestamp, list[Signal]] = {}
    for forecast in forecasts:
        grouped.setdefault(pd.Timestamp(forecast.as_of), []).append(
            decide(forecast, thresholds, cfg)
        )
    return {
        timestamp: tuple(sorted(grouped[timestamp], key=lambda signal: signal.symbol))
        for timestamp in sorted(grouped)
    }


def strengths_of(forecasts: Sequence[Forecast]) -> np.ndarray:
    """The cumulative forecast of every path, in order. The grid is built from these."""
    return np.array([trend_strength(forecast.path) for forecast in forecasts])


def _require_path(path: np.ndarray) -> np.ndarray:
    values = np.asarray(path, dtype="float64")
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            f"a forecast path must be a non-empty (H,) array, got shape {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(
            "a forecast path carries NaN or infinity; refusing to decide rather than "
            "returning a hold that would look like a considered decision"
        )
    return values


__all__ = [
    "ACTIONS",
    "ENTER_LONG",
    "EXIT",
    "HOLD",
    "Thresholds",
    "decide",
    "decide_all",
    "strengths_of",
    "trend_strength",
    "up_points",
]
