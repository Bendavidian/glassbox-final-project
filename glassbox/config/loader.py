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

import pywt
import yaml
from dotenv import load_dotenv

DEFAULT_SETTINGS_PATH = Path(__file__).with_name("settings.yaml")

# The only broker endpoint this project may ever reach. A module constant in the config
# layer rather than a settings.yaml key: spec 5 froze the config schema, and this is not
# a tunable — it is the boundary that keeps the system away from real money (spec 2.2).
PAPER_ENDPOINT = "https://paper-api.alpaca.markets"

VALID_MODELS = ("persistence", "dlinear", "fits", "wits")
VALID_SUPERVISION = ("F", "B+F")
#: Boundary extensions WITS accepts (GB-66's second design question, measured as an axis).
#: Defined here rather than in `model/wits.py` because the layer contract lets `model`
#: import `config` and not the other way round, so this is the only side that both the
#: validator and the model can read - one copy rather than the `VALID_MODELS` situation,
#: which needs a test to hold two copies equal.
#:
#: `periodization` is deliberately absent: it changes the retained coefficient count (15
#: against 21) and so the parameter count (480 against 882), which would confound boundary
#: handling with capacity on an axis meant to isolate the boundary.
VALID_BOUNDARIES = ("symmetric", "zero", "periodic")
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
class WitsConfig:
    """WITS wavelet core (GB-66): which basis, how deep, how much is kept, which edge.

    `boundary` and `shift_invariant` are **axes the study measures**, not settings chosen
    on convention - GB-66's second and third design questions. They live here rather than
    as CLI flags because they shape the weights, which is what puts `wits` in
    `MODEL_SHAPING_SECTIONS`: a checkpoint trained under one boundary must not be loaded
    under another.
    """

    family: str
    levels: int
    retained_bands: int
    boundary: str
    shift_invariant: bool
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
    """Which forecaster is active and the shared training hyperparameters.

    ``batch_size`` is ``None`` for full-batch training - one batch of everything, every
    epoch. ``None`` rather than a very large integer: it reads as an absence of
    mini-batching rather than a lie about the field's name, and it cannot silently revert
    to mini-batching the day a fold grows past whatever number was written. Adopted
    2026-08-17 on a 16/16 measurement; see DECISIONS.md, including what that measurement
    does and does not establish.
    """

    active: str
    epochs: int
    patience: int
    lr: float
    batch_size: int | None


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
    """Starting equity and the trading frictions applied in every backtest."""

    initial_cash: float
    fee_bps: float
    slippage_bps: float
    target_in_loop: bool
    """Whether the take-profit is evaluated by the loop instead of resting at the broker.

    **The axis GB-49 added on 25 Aug 2026, and it exists because the venue forced it.**
    Alpaca refuses every multi-leg order class on a fractional quantity - `bracket` and
    `oco` both return `{"code":42210000,"message":"fractional orders must be simple
    orders"}`, measured - and a working sell holds the whole position, so a fractional
    position can carry exactly one broker-side protective order. The stop is that order.

    `false` is the classical backtest and what every published number in this study
    assumes: the target rests at the broker and fills intraday at the target price the
    moment the high crosses it.

    `true` is the system that can actually be built: the loop evaluates the target against
    **completed daily bars only** - `drop_incomplete_bar` is a ruling and GB-7 measured
    that `latest_quote` is refused on this SIP subscription, so intraday is unreachable -
    and the exit fills at the next available open.

    It sits in `backtest` rather than `risk` because it changes how a simulated fill is
    priced and nothing about the position's size or levels; and `backtest` is not in
    `MODEL_SHAPING_SECTIONS`, so switching it refuses no checkpoint.
    """


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
    retry_attempts: int
    retry_backoff_seconds: float
    heartbeat_seconds: int
    """Seconds between idle heartbeat lines on a run that spans more than one session.

    A loop that has died and a loop correctly idling produce identical output - nothing -
    so without a heartbeat there is no way to tell them apart, and the overnight
    protection residual is only bounded while the process is **alive**.
    """


@dataclass(frozen=True)
class Config:
    """The resolved, validated configuration. The only source of runtime values."""

    meta: MetaConfig
    universe: tuple[str, ...]
    data: DataConfig
    window: WindowConfig
    wavelet: WaveletConfig
    fits: FitsConfig
    wits: WitsConfig
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


def _as_optional_positive_int(
    section: Mapping[str, Any], path: str, key: str
) -> int | None:
    """A positive integer, or ``None`` meaning "not applicable".

    ``null`` is the same idiom ``signal.min_trend`` uses: a value that is deliberately
    absent rather than defaulted. Anything else - zero, a float, a string, ``true`` - is
    refused, so a typo cannot become "full batch" by accident.
    """
    value = _get(section, path, key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{path}.{key}", "a positive integer or null", value)
    if value <= 0:
        _fail(f"{path}.{key}", "a positive integer or null", value)
    return int(value)


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


def _build_wits(raw: Mapping[str, Any], window: WindowConfig) -> WitsConfig:
    section = _section(raw, "wits")
    family = _as_str(section, "wits", "family")
    if family not in pywt.wavelist(kind="discrete"):
        _fail("wits.family", "a discrete wavelet pywt knows", family)
    levels = _as_positive_int(section, "wits", "levels")
    retained = _as_positive_int(section, "wits", "retained_bands")
    if retained > levels + 1:
        _fail(
            "wits.retained_bands",
            f"at most levels + 1 ({levels + 1}), one approximation band and one detail "
            "band per level",
            retained,
        )
    # **pywt's own criterion, asked rather than reimplemented.** The obvious check -
    # `2**levels <= input_len` - is not merely weaker, it gives wrong advice: at L=30 it
    # admits J=3, which pywt warns is boundary-dominated because db4's 8-tap filter leaves
    # no coefficient untouched by the edge. Deriving the rule here would be a second copy
    # of a fact pywt already holds, and this project has six instances of what that costs.
    ceiling = pywt.dwt_max_level(window.input_len, pywt.Wavelet(family).dec_len)
    if levels > ceiling:
        _fail(
            "wits.levels",
            f"at most {ceiling} for a {family!r} filter over window.input_len "
            f"{window.input_len}; deeper and every coefficient is a boundary effect",
            levels,
        )
    return WitsConfig(
        family=family,
        levels=levels,
        retained_bands=retained,
        boundary=_as_choice(section, "wits", "boundary", VALID_BOUNDARIES),
        shift_invariant=_as_bool(section, "wits", "shift_invariant"),
        supervision=_as_choice(section, "wits", "supervision", VALID_SUPERVISION),
        individual_weights=_as_bool(section, "wits", "individual_weights"),
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
        batch_size=_as_optional_positive_int(section, "model", "batch_size"),
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
        initial_cash=_as_positive_float(section, "backtest", "initial_cash"),
        fee_bps=_as_non_negative_float(section, "backtest", "fee_bps"),
        slippage_bps=_as_non_negative_float(section, "backtest", "slippage_bps"),
        target_in_loop=_as_bool(section, "backtest", "target_in_loop"),
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
        retry_attempts=_as_positive_int(section, "live", "retry_attempts"),
        retry_backoff_seconds=_as_positive_float(
            section, "live", "retry_backoff_seconds"
        ),
        heartbeat_seconds=_as_positive_int(section, "live", "heartbeat_seconds"),
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
        wits=_build_wits(raw, window),
        channels=_build_channels(raw),
        model=_build_model(raw),
        signal=_build_signal(raw),
        risk=_build_risk(raw),
        backtest=_build_backtest(raw),
        walkforward=_build_walkforward(raw),
        live=_build_live(raw),
    )


# The sections a trained weight can depend on. Everything outside this list describes
# what the system DOES with a model, not what shaped one: how often it polls, how many
# times it retries, how it sizes a position, where the threshold sits.
#
# `universe` is deliberately absent even though it decides which symbols were trained on.
# A checkpoint carries per-symbol normalisation statistics and `Predictor.stats_for`
# already refuses an untrained symbol by name - "the checkpoint holds no normalisation
# statistics for X; it was trained on [...]" - so a universe change surfaces as an
# explicit refusal at the point of use rather than as a silently wrong forecast. Putting
# it in the hash would trade that precise message for a blanket retrain.
MODEL_SHAPING_SECTIONS = (
    "data",
    "window",
    "wavelet",
    "fits",
    # **`wits` is deliberately NOT here yet, and this is a dated hold rather than a
    # judgement that it does not belong.** It does: the COF axis routes through `cfg.fits`
    # precisely so two cells differing only in the cutoff are distinguishable by
    # `model_config_hash` alone (`study._cell_config`), and GB-66's boundary axis needs the
    # same property.
    #
    # Adding a section here changes `model_config_hash` for **every** configuration,
    # including one running `model.active: dlinear` that never reads `wits:`. Measured on
    # 25 Aug 2026: it moved the deployed hash from 234ab499... to 6f2a6978..., and
    # `train.load_predictor` refuses a checkpoint whose model hash moved - so the deployed
    # checkpoint would have been rejected and the GATE 2 rehearsal would not have started.
    #
    # Held until the grid re-run, on GB-61's precedent: the `universe:` flip and its grid
    # re-run wait until the loop stops, because editing the deployed config under a live
    # multi-day loop is GB-59's first defect. Nothing is lost meanwhile - a WITS checkpoint
    # carries its own family, levels, boundary and retained_bands and `WITSForecaster.load`
    # rebuilds from those, never from settings.yaml, so no checkpoint can be loaded at a
    # geometry it was not trained at. What is missing is only hash-level distinguishability
    # between two study cells that differ by boundary, and no such cell exists yet.
    #
    # `test_wits.py::test_the_boundary_axis_reaches_results_csv` is a strict xfail over
    # exactly that pair, so the day the axis lands, this fails until it is added.
    "channels",
    "model",
)


def config_hash(cfg: Config) -> str:
    """Return a stable SHA-256 hex digest of the resolved configuration.

    The digest is computed over a canonical JSON rendering with sorted keys, so it is
    identical across processes and machines and changes if any value changes. It ties a
    ``DecisionRecord`` to the exact configuration that produced it (GB-29).

    **This is the whole configuration and its meaning does not narrow.** It answers
    "under what settings was this decision made?", and the answer has to include the
    settings that decided *not* to trade as much as the ones that shaped the forecast.
    :func:`model_config_hash` is the narrower question and is a different function
    precisely so that this one can stay what a ``DecisionRecord`` promises.
    """
    payload = json.dumps(
        asdict(cfg), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_config_hash(cfg: Config) -> str:
    """A digest of only the settings that can shape a trained weight. **Checkpoints gate
    on this one.**

    ``config_hash`` covers everything, which is right for a decision record and wrong for
    a checkpoint. Measured on 19 Aug 2026: adding ``live.retry_attempts`` and
    ``live.retry_backoff_seconds`` - two numbers a polling loop reads and no weight can
    see - changed ``config_hash`` and made ``load_checkpoint`` refuse **every** existing
    model. The guard was correct and the cost was pure waste.

    **The danger is not that retrain.** It is Sprint 4, where FITS and the COF sweep touch
    the configuration repeatedly, and a guard that fires spuriously every time is a guard
    somebody eventually weakens or works around. This is fixed before that pressure
    exists rather than under it.

    Covers :data:`MODEL_SHAPING_SECTIONS` plus ``meta.seed`` - the seed decides the
    initialisation and the shuffling, so two models under identical settings and different
    seeds are different models. ``meta.version`` is excluded: it versions the
    configuration *schema*, not anything a weight sees.
    """
    payload = {
        section: asdict(getattr(cfg, section)) for section in MODEL_SHAPING_SECTIONS
    }
    payload["seed"] = cfg.meta.seed
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


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
        base_url=base_url or PAPER_ENDPOINT,
    )


def require_paper_endpoint(base_url: str | None) -> None:
    """Raise unless ``base_url`` is the Alpaca paper endpoint.

    The project trades paper money only (spec 2.2: real money, never). Anything that is
    about to talk to a broker calls this first, so pointing at the live endpoint fails
    closed rather than trading real money by accident.

    Raises:
        ValueError: ``base_url`` is unset or is not the paper endpoint.
    """
    if not base_url:
        raise ValueError(
            "ALPACA_BASE_URL is not set; refusing to connect without a verified endpoint"
        )
    if base_url.rstrip("/") != PAPER_ENDPOINT:
        raise ValueError(
            f"refusing to connect: endpoint is {base_url!r}, "
            f"and this project only ever trades paper money at {PAPER_ENDPOINT}"
        )
