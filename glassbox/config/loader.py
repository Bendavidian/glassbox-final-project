"""L0: load and validate ``settings.yaml`` into typed objects, failing loud with the
offending field named.

Every runtime value in the system originates here. No module reads ``settings.yaml`` or
``os.environ`` directly; they call :func:`load_config` or :func:`alpaca_credentials`.

Contract:

* :func:`load_config` returns a fully validated, frozen :class:`Config` or raises
  ``ValueError`` naming the exact field path, e.g.
  ``"risk.max_position_pct must be in (0, 1], got 1.4"``.
* :func:`config_hash` is a stable SHA-256 over the resolved configuration. Equal
  configurations hash equal across processes; changing any value changes the hash.
* :func:`alpaca_credentials` reads broker secrets from the environment (``.env`` via
  python-dotenv). It raises only when called, never at import, because most of the
  system never touches the broker.

Implemented in GB-2.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

DEFAULT_SETTINGS_PATH = Path(__file__).with_name("settings.yaml")

VALID_MODELS = ("persistence", "dlinear", "fits")
VALID_SUPERVISION = ("F", "B+F")
VALID_LIVE_MODES = ("auto", "co_pilot")

# Keys of `channels` that name a channel set rather than the active selection.
_ACTIVE_KEY = "active"


# ── Typed sections ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetaConfig:
    """Run identity: config schema version and the global random seed."""

    version: int
    seed: int


@dataclass(frozen=True)
class DataConfig:
    """Where history starts, where it is cached, and which exchange calendar applies."""

    start: str
    cache_dir: str
    calendar: str


@dataclass(frozen=True)
class WindowConfig:
    """The model input window: ``input_len`` bars in, ``horizon`` days out."""

    input_len: int
    horizon: int


@dataclass(frozen=True)
class WaveletConfig:
    """Causal rolling DWT settings for the wav_a1..wav_a3 channels."""

    family: str
    levels: int
    rolling_window: int
    mode: str


@dataclass(frozen=True)
class FitsConfig:
    """FITS spectral core: the cutoff period is the one hyperparameter."""

    cutoff_period_days: int
    supervision: str
    individual_weights: bool


@dataclass(frozen=True)
class ChannelConfig:
    """The named channel sets and which one is active.

    ``sets`` is a sorted tuple of ``(name, channels)`` pairs rather than a dict so the
    dataclass stays immutable and hashes deterministically.
    """

    sets: tuple[tuple[str, tuple[str, ...]], ...]
    active: str

    @property
    def active_channels(self) -> tuple[str, ...]:
        """The channel list named by ``active``, in declaration order."""
        return dict(self.sets)[self.active]

    @property
    def names(self) -> tuple[str, ...]:
        """Every declared channel-set name."""
        return tuple(name for name, _ in self.sets)


@dataclass(frozen=True)
class ModelConfig:
    """Which forecaster is active and the shared training hyperparameters."""

    active: str
    epochs: int
    patience: int
    lr: float
    batch_size: int


@dataclass(frozen=True)
class SignalConfig:
    """Signal thresholds. ``min_trend``/``max_trend`` are None until calibrated on the
    validation slice of a fold — they are never hardcoded."""

    min_trend: float | None
    max_trend: float | None
    min_up_points: int
    top_k: int


@dataclass(frozen=True)
class RiskConfig:
    """Position sizing and exit caps, all as fractions in (0, 1]."""

    max_position_pct: float
    max_gross_exposure: float
    stop_loss_pct: float
    take_profit_pct: float


@dataclass(frozen=True)
class BacktestConfig:
    """Trading frictions applied in every backtest."""

    fee_bps: float
    slippage_bps: float


@dataclass(frozen=True)
class WalkforwardConfig:
    """Fold geometry in months, plus the fold cap for the 8-week budget."""

    train_months: int
    val_months: int
    test_months: int
    step_months: int
    max_folds: int


@dataclass(frozen=True)
class LiveConfig:
    """Live loop cadence, execution mode and the Israel-time market window."""

    poll_seconds: int
    mode: str
    market_open_il: str
    market_close_il: str


@dataclass(frozen=True)
class Config:
    """The resolved, validated configuration. The only source of runtime values."""

    meta: MetaConfig
    universe: tuple[str, ...]
    data: DataConfig
    window: WindowConfig
    wavelet: WaveletConfig
    fits: FitsConfig
    channels: ChannelConfig
    model: ModelConfig
    signal: SignalConfig
    risk: RiskConfig
    backtest: BacktestConfig
    walkforward: WalkforwardConfig
    live: LiveConfig


@dataclass(frozen=True)
class AlpacaCredentials:
    """Broker secrets. Never logged, never written to a decision record."""

    api_key: str
    secret_key: str
    base_url: str | None


# ── Field access helpers ─────────────────────────────────────────────────────


def _fail(path: str, requirement: str, value: Any) -> None:
    """Raise ``ValueError`` naming the field path, its requirement and what was found."""
    raise ValueError(f"{path} must be {requirement}, got {value!r}")


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return a top-level mapping section, or raise naming it."""
    if name not in raw:
        raise ValueError(f"{name} is missing from the configuration")
    value = raw[name]
    if not isinstance(value, Mapping):
        _fail(name, "a mapping", value)
    return value


def _get(section: Mapping[str, Any], path: str, key: str) -> Any:
    """Return a required key from a section, or raise naming its full path."""
    if key not in section:
        raise ValueError(f"{path}.{key} is missing from the configuration")
    return section[key]


def _as_int(section: Mapping[str, Any], path: str, key: str) -> int:
    value = _get(section, path, key)
    # bool is an int subclass; an accidental `true` must not pass as a count.
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{path}.{key}", "an integer", value)
    return int(value)


def _as_positive_int(section: Mapping[str, Any], path: str, key: str) -> int:
    value = _as_int(section, path, key)
    if value <= 0:
        _fail(f"{path}.{key}", "a positive integer", value)
    return value


def _as_float(section: Mapping[str, Any], path: str, key: str) -> float:
    value = _get(section, path, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{path}.{key}", "a number", value)
    return float(value)


def _as_positive_float(section: Mapping[str, Any], path: str, key: str) -> float:
    value = _as_float(section, path, key)
    if value <= 0:
        _fail(f"{path}.{key}", "a positive number", value)
    return value


def _as_non_negative_float(section: Mapping[str, Any], path: str, key: str) -> float:
    value = _as_float(section, path, key)
    if value < 0:
        _fail(f"{path}.{key}", "zero or greater", value)
    return value


def _as_optional_float(section: Mapping[str, Any], path: str, key: str) -> float | None:
    value = _get(section, path, key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{path}.{key}", "a number or null", value)
    return float(value)


def _as_str(section: Mapping[str, Any], path: str, key: str) -> str:
    value = _get(section, path, key)
    if not isinstance(value, str) or not value.strip():
        _fail(f"{path}.{key}", "a non-empty string", value)
    return value


def _as_bool(section: Mapping[str, Any], path: str, key: str) -> bool:
    value = _get(section, path, key)
    if not isinstance(value, bool):
        _fail(f"{path}.{key}", "a boolean", value)
    return value


def _as_choice(
    section: Mapping[str, Any], path: str, key: str, choices: tuple[str, ...]
) -> str:
    value = _as_str(section, path, key)
    if value not in choices:
        _fail(f"{path}.{key}", f"one of {list(choices)}", value)
    return value


def _as_fraction(section: Mapping[str, Any], path: str, key: str) -> float:
    """A fraction in (0, 1] — the contract for every ``*_pct`` field."""
    value = _as_float(section, path, key)
    if not 0 < value <= 1:
        _fail(f"{path}.{key}", "in (0, 1]", value)
    return value


# ── Section builders ─────────────────────────────────────────────────────────


def _build_universe(raw: Mapping[str, Any]) -> tuple[str, ...]:
    if "universe" not in raw:
        raise ValueError("universe is missing from the configuration")
    value = raw["universe"]
    if not isinstance(value, list) or not value:
        _fail("universe", "a non-empty list of tickers", value)
    for index, symbol in enumerate(value):
        if not isinstance(symbol, str) or not symbol.strip():
            _fail(f"universe[{index}]", "a non-empty string", symbol)
        if symbol != symbol.upper():
            _fail(f"universe[{index}]", "an uppercase ticker", symbol)
    if len(set(value)) != len(value):
        _fail("universe", "unique tickers", value)
    return tuple(value)


def _build_meta(raw: Mapping[str, Any]) -> MetaConfig:
    section = _section(raw, "meta")
    return MetaConfig(
        version=_as_positive_int(section, "meta", "version"),
        seed=_as_int(section, "meta", "seed"),
    )


def _build_data(raw: Mapping[str, Any]) -> DataConfig:
    section = _section(raw, "data")
    return DataConfig(
        start=_as_str(section, "data", "start"),
        cache_dir=_as_str(section, "data", "cache_dir"),
        calendar=_as_str(section, "data", "calendar"),
    )


def _build_window(raw: Mapping[str, Any]) -> WindowConfig:
    section = _section(raw, "window")
    input_len = _as_positive_int(section, "window", "input_len")
    horizon = _as_positive_int(section, "window", "horizon")
    if input_len <= horizon:
        _fail("window.input_len", f"greater than window.horizon ({horizon})", input_len)
    return WindowConfig(input_len=input_len, horizon=horizon)


def _build_wavelet(raw: Mapping[str, Any]) -> WaveletConfig:
    section = _section(raw, "wavelet")
    levels = _as_positive_int(section, "wavelet", "levels")
    rolling_window = _as_positive_int(section, "wavelet", "rolling_window")
    minimum = 2**levels
    if rolling_window < minimum:
        _fail(
            "wavelet.rolling_window",
            f"at least 2 ** wavelet.levels ({minimum})",
            rolling_window,
        )
    return WaveletConfig(
        family=_as_str(section, "wavelet", "family"),
        levels=levels,
        rolling_window=rolling_window,
        mode=_as_str(section, "wavelet", "mode"),
    )


def _build_fits(raw: Mapping[str, Any], window: WindowConfig) -> FitsConfig:
    section = _section(raw, "fits")
    cutoff = _as_positive_int(section, "fits", "cutoff_period_days")
    if cutoff < 2:
        _fail("fits.cutoff_period_days", "at least 2", cutoff)
    if cutoff >= window.input_len:
        _fail(
            "fits.cutoff_period_days",
            f"less than window.input_len ({window.input_len})",
            cutoff,
        )
    return FitsConfig(
        cutoff_period_days=cutoff,
        supervision=_as_choice(section, "fits", "supervision", VALID_SUPERVISION),
        individual_weights=_as_bool(section, "fits", "individual_weights"),
    )


def _build_channels(raw: Mapping[str, Any]) -> ChannelConfig:
    section = _section(raw, "channels")
    if _ACTIVE_KEY not in section:
        raise ValueError("channels.active is missing from the configuration")

    sets: list[tuple[str, tuple[str, ...]]] = []
    for name, value in section.items():
        if name == _ACTIVE_KEY:
            continue
        if not isinstance(value, list) or not value:
            _fail(f"channels.{name}", "a non-empty list of channel names", value)
        for index, channel in enumerate(value):
            if not isinstance(channel, str) or not channel.strip():
                _fail(f"channels.{name}[{index}]", "a non-empty string", channel)
        if len(set(value)) != len(value):
            _fail(f"channels.{name}", "unique channel names", value)
        sets.append((name, tuple(value)))

    if not sets:
        _fail("channels", "at least one channel set", list(section))

    active = _as_str(section, "channels", _ACTIVE_KEY)
    declared = [name for name, _ in sets]
    if active not in declared:
        _fail("channels.active", f"one of {sorted(declared)}", active)

    return ChannelConfig(sets=tuple(sorted(sets)), active=active)


def _build_model(raw: Mapping[str, Any]) -> ModelConfig:
    section = _section(raw, "model")
    return ModelConfig(
        active=_as_choice(section, "model", "active", VALID_MODELS),
        epochs=_as_positive_int(section, "model", "epochs"),
        patience=_as_positive_int(section, "model", "patience"),
        lr=_as_positive_float(section, "model", "lr"),
        batch_size=_as_positive_int(section, "model", "batch_size"),
    )


def _build_signal(raw: Mapping[str, Any]) -> SignalConfig:
    section = _section(raw, "signal")
    min_trend = _as_optional_float(section, "signal", "min_trend")
    max_trend = _as_optional_float(section, "signal", "max_trend")
    if min_trend is not None and max_trend is not None and min_trend > max_trend:
        _fail(
            "signal.min_trend",
            f"no greater than signal.max_trend ({max_trend})",
            min_trend,
        )
    return SignalConfig(
        min_trend=min_trend,
        max_trend=max_trend,
        min_up_points=_as_positive_int(section, "signal", "min_up_points"),
        top_k=_as_positive_int(section, "signal", "top_k"),
    )


def _build_risk(raw: Mapping[str, Any]) -> RiskConfig:
    section = _section(raw, "risk")
    risk = RiskConfig(
        max_position_pct=_as_fraction(section, "risk", "max_position_pct"),
        max_gross_exposure=_as_fraction(section, "risk", "max_gross_exposure"),
        stop_loss_pct=_as_fraction(section, "risk", "stop_loss_pct"),
        take_profit_pct=_as_fraction(section, "risk", "take_profit_pct"),
    )
    if risk.max_position_pct > risk.max_gross_exposure:
        _fail(
            "risk.max_position_pct",
            f"no greater than risk.max_gross_exposure ({risk.max_gross_exposure})",
            risk.max_position_pct,
        )
    return risk


def _build_backtest(raw: Mapping[str, Any]) -> BacktestConfig:
    section = _section(raw, "backtest")
    return BacktestConfig(
        fee_bps=_as_non_negative_float(section, "backtest", "fee_bps"),
        slippage_bps=_as_non_negative_float(section, "backtest", "slippage_bps"),
    )


def _build_walkforward(raw: Mapping[str, Any]) -> WalkforwardConfig:
    section = _section(raw, "walkforward")
    return WalkforwardConfig(
        train_months=_as_positive_int(section, "walkforward", "train_months"),
        val_months=_as_positive_int(section, "walkforward", "val_months"),
        test_months=_as_positive_int(section, "walkforward", "test_months"),
        step_months=_as_positive_int(section, "walkforward", "step_months"),
        max_folds=_as_positive_int(section, "walkforward", "max_folds"),
    )


def _build_live(raw: Mapping[str, Any]) -> LiveConfig:
    section = _section(raw, "live")
    return LiveConfig(
        poll_seconds=_as_positive_int(section, "live", "poll_seconds"),
        mode=_as_choice(section, "live", "mode", VALID_LIVE_MODES),
        market_open_il=_as_str(section, "live", "market_open_il"),
        market_close_il=_as_str(section, "live", "market_close_il"),
    )


# ── Public API ───────────────────────────────────────────────────────────────


def load_config(path: str | Path | None = None) -> Config:
    """Load, validate and freeze the configuration at ``path``.

    Args:
        path: Settings file to read. Defaults to the packaged ``settings.yaml``.

    Returns:
        A fully validated :class:`Config`.

    Raises:
        FileNotFoundError: The settings file does not exist.
        ValueError: The file is not a mapping, or any field is missing or invalid.
            The message names the field path.
    """
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    if not settings_path.is_file():
        raise FileNotFoundError(f"settings file not found: {settings_path}")

    with settings_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, Mapping):
        # Deliberately ValueError, not TypeError: every configuration failure raises the
        # same type so callers catch one thing (noqa: the linter prefers TypeError here).
        raise ValueError(  # noqa: TRY004
            f"{settings_path} must contain a mapping, got {type(raw).__name__}"
        )

    window = _build_window(raw)
    return Config(
        meta=_build_meta(raw),
        universe=_build_universe(raw),
        data=_build_data(raw),
        window=window,
        wavelet=_build_wavelet(raw),
        fits=_build_fits(raw, window),
        channels=_build_channels(raw),
        model=_build_model(raw),
        signal=_build_signal(raw),
        risk=_build_risk(raw),
        backtest=_build_backtest(raw),
        walkforward=_build_walkforward(raw),
        live=_build_live(raw),
    )


def config_hash(cfg: Config) -> str:
    """Return a stable SHA-256 hex digest of the resolved configuration.

    The digest is computed over a canonical JSON rendering with sorted keys, so it is
    identical across processes and machines and changes if any value changes. It ties a
    ``DecisionRecord`` to the exact configuration that produced it (GB-29).
    """
    payload = json.dumps(
        asdict(cfg), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def alpaca_credentials(dotenv_path: str | Path | None = None) -> AlpacaCredentials:
    """Return the Alpaca credentials, reading ``.env`` if present.

    Credentials are resolved on call, never at import, so the offline majority of the
    system runs without any broker secret being configured.

    Args:
        dotenv_path: Explicit ``.env`` to read. Defaults to the nearest one found by
            python-dotenv, searching upward from the working directory.

    Returns:
        The populated :class:`AlpacaCredentials`.

    Raises:
        ValueError: A required variable is unset or empty, named in the message.
    """
    load_dotenv(dotenv_path=dotenv_path, override=False)

    api_key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    base_url = os.environ.get("ALPACA_BASE_URL", "").strip()

    for name, value in (("ALPACA_API_KEY", api_key), ("ALPACA_SECRET_KEY", secret_key)):
        if not value:
            raise ValueError(f"{name} is not set; add it to .env or the environment")

    return AlpacaCredentials(
        api_key=api_key,
        secret_key=secret_key,
        base_url=base_url or None,
    )
