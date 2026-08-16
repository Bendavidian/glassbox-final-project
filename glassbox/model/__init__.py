"""L3 model: forecasters behind the Forecaster protocol. Knows nothing of orders, money
or brokers.

``ALL_FORECASTERS`` is the one list of them. The contract test iterates it, so a new model
is integrated by adding a line here and nothing else — in particular, not by editing the
test that judges it. GB-41 adds ``fits``.

The registry holds **factories** rather than classes because constructors legitimately
differ: Persistence needs a window shape, DLinear also needs a channel set and training
hyperparameters. A factory absorbs that without forcing a construction signature into the
frozen protocol, which would be a contract change bought for tidiness.
"""

from collections.abc import Callable

from glassbox.config.loader import Config
from glassbox.contracts.protocols import Forecaster
from glassbox.model.ltsf import DLinearForecaster
from glassbox.model.persistence import PersistenceForecaster

ForecasterFactory = Callable[[Config, tuple[str, ...]], Forecaster]


def _persistence(cfg: Config, channels: tuple[str, ...]) -> Forecaster:
    del channels  # the baseline predicts zero whatever it is shown
    return PersistenceForecaster(
        input_len=cfg.window.input_len, horizon=cfg.window.horizon
    )


def _dlinear(cfg: Config, channels: tuple[str, ...]) -> Forecaster:
    return DLinearForecaster(
        input_len=cfg.window.input_len,
        horizon=cfg.window.horizon,
        channels=channels,
        cfg=cfg,
    )


ALL_FORECASTERS: dict[str, ForecasterFactory] = {
    "persistence": _persistence,
    "dlinear": _dlinear,
    # GB-41: "fits": _fits
}

__all__ = [
    "ALL_FORECASTERS",
    "DLinearForecaster",
    "ForecasterFactory",
    "PersistenceForecaster",
]
