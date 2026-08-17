"""GB-2 acceptance: the config layer loads, validates, fails loud and hashes stably."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import yaml

from glassbox.config.loader import (
    DEFAULT_SETTINGS_PATH,
    PAPER_ENDPOINT,
    AlpacaCredentials,
    BacktestConfig,
    ChannelConfig,
    Config,
    DataConfig,
    FitsConfig,
    LiveConfig,
    MetaConfig,
    ModelConfig,
    RiskConfig,
    SignalConfig,
    WalkforwardConfig,
    WaveletConfig,
    WindowConfig,
    alpaca_credentials,
    config_hash,
    load_config,
    require_paper_endpoint,
)

ALPACA_VARS = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL")


@pytest.fixture
def raw_settings() -> dict[str, Any]:
    """The packaged settings.yaml as a plain dict, safe for a test to mutate."""
    with DEFAULT_SETTINGS_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_settings(tmp_path: Path, raw: dict[str, Any]) -> Path:
    """Write a settings mapping to a temporary file and return its path."""
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def load_mutated(
    tmp_path: Path, raw: dict[str, Any], mutate: Callable[[dict[str, Any]], None]
) -> Config:
    """Apply ``mutate`` to a copy of ``raw`` and load the result."""
    mutated = copy.deepcopy(raw)
    mutate(mutated)
    return load_config(write_settings(tmp_path, mutated))


# ── A valid load ─────────────────────────────────────────────────────────────


def test_default_config_loads() -> None:
    """The packaged settings.yaml is valid and yields the declared section types."""
    cfg = load_config()

    assert isinstance(cfg, Config)
    assert isinstance(cfg.meta, MetaConfig)
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.window, WindowConfig)
    assert isinstance(cfg.wavelet, WaveletConfig)
    assert isinstance(cfg.fits, FitsConfig)
    assert isinstance(cfg.channels, ChannelConfig)
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.signal, SignalConfig)
    assert isinstance(cfg.risk, RiskConfig)
    assert isinstance(cfg.backtest, BacktestConfig)
    assert isinstance(cfg.walkforward, WalkforwardConfig)
    assert isinstance(cfg.live, LiveConfig)


def test_default_config_values_match_the_spec() -> None:
    """Spot-check the values the rest of the system depends on (spec 5)."""
    cfg = load_config()

    assert cfg.meta.seed == 1337
    assert cfg.universe == ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL")
    assert cfg.window.input_len == 120
    assert cfg.window.horizon == 4
    assert cfg.fits.cutoff_period_days == 5
    assert cfg.fits.supervision == "B+F"
    assert cfg.fits.individual_weights is False
    assert cfg.channels.active == "C0_base"
    assert cfg.channels.active_channels == (
        "close_logret",
        "rsi14",
        "vol_z",
        "mom10",
        "ma_dist20",
    )
    assert set(cfg.channels.names) == {"C0_base", "C2_hybrid"}
    assert cfg.model.active == "dlinear"
    assert cfg.model.batch_size == 64  # full batch adopted, then reverted — GB-20
    assert cfg.signal.min_trend is None
    assert cfg.signal.max_trend is None
    assert cfg.live.mode == "co_pilot"
    assert cfg.walkforward.max_folds == 16
    assert cfg.backtest.initial_cash == 100_000.0
    assert (cfg.backtest.fee_bps, cfg.backtest.slippage_bps) == (1.0, 2.0)


def test_config_is_frozen() -> None:
    """The resolved config cannot be mutated by a downstream module."""
    cfg = load_config()
    with pytest.raises(FrozenInstanceError):
        cfg.window.input_len = 60  # type: ignore[misc]


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(Path("no_such_settings.yaml"))


# ── Fail-loud validation ─────────────────────────────────────────────────────

INVALID_CASES: tuple[tuple[str, Callable[[dict[str, Any]], None], str], ...] = (
    (
        "universe_empty",
        lambda raw: raw.__setitem__("universe", []),
        "universe",
    ),
    (
        "universe_duplicate",
        lambda raw: raw.__setitem__("universe", ["AAPL", "AAPL"]),
        "universe",
    ),
    (
        "universe_lowercase",
        lambda raw: raw.__setitem__("universe", ["aapl", "MSFT"]),
        "universe[0]",
    ),
    (
        "input_len_not_greater_than_horizon",
        lambda raw: raw["window"].__setitem__("input_len", 4),
        "window.input_len",
    ),
    (
        "window_key_missing",
        lambda raw: raw["window"].pop("horizon"),
        "window.horizon",
    ),
    (
        "rolling_window_too_small",
        lambda raw: raw["wavelet"].__setitem__("rolling_window", 4),
        "wavelet.rolling_window",
    ),
    # `null` means full batch, so the guard has to tell "deliberately absent" apart from
    # every other falsy thing a typo could produce.
    (
        "batch_size_zero",
        lambda raw: raw["model"].__setitem__("batch_size", 0),
        "model.batch_size",
    ),
    (
        "batch_size_negative",
        lambda raw: raw["model"].__setitem__("batch_size", -1),
        "model.batch_size",
    ),
    (
        "batch_size_float",
        lambda raw: raw["model"].__setitem__("batch_size", 64.0),
        "model.batch_size",
    ),
    (
        "batch_size_bool",
        lambda raw: raw["model"].__setitem__("batch_size", True),
        "model.batch_size",
    ),
    (
        "batch_size_string",
        lambda raw: raw["model"].__setitem__("batch_size", "all"),
        "model.batch_size",
    ),
    (
        "batch_size_missing",
        lambda raw: raw["model"].pop("batch_size"),
        "model.batch_size",
    ),
    (
        "cutoff_below_two",
        lambda raw: raw["fits"].__setitem__("cutoff_period_days", 1),
        "fits.cutoff_period_days",
    ),
    (
        "cutoff_not_below_input_len",
        lambda raw: raw["fits"].__setitem__("cutoff_period_days", 120),
        "fits.cutoff_period_days",
    ),
    (
        "supervision_unknown",
        lambda raw: raw["fits"].__setitem__("supervision", "B"),
        "fits.supervision",
    ),
    (
        "initial_cash_zero",
        lambda raw: raw["backtest"].__setitem__("initial_cash", 0.0),
        "backtest.initial_cash",
    ),
    (
        "initial_cash_missing",
        lambda raw: raw["backtest"].pop("initial_cash"),
        "backtest.initial_cash",
    ),
    (
        "pct_above_one",
        lambda raw: raw["risk"].__setitem__("max_position_pct", 1.4),
        "risk.max_position_pct",
    ),
    (
        "pct_zero",
        lambda raw: raw["risk"].__setitem__("stop_loss_pct", 0),
        "risk.stop_loss_pct",
    ),
    (
        "pct_negative",
        lambda raw: raw["risk"].__setitem__("take_profit_pct", -0.01),
        "risk.take_profit_pct",
    ),
    (
        "position_exceeds_gross_exposure",
        lambda raw: raw["risk"].__setitem__("max_position_pct", 0.9),
        "risk.max_position_pct",
    ),
    (
        "model_active_unknown",
        lambda raw: raw["model"].__setitem__("active", "transformer"),
        "model.active",
    ),
    (
        "channels_active_unknown",
        lambda raw: raw["channels"].__setitem__("active", "C9_nonexistent"),
        "channels.active",
    ),
    (
        "walkforward_zero_months",
        lambda raw: raw["walkforward"].__setitem__("train_months", 0),
        "walkforward.train_months",
    ),
    (
        "walkforward_negative_months",
        lambda raw: raw["walkforward"].__setitem__("val_months", -3),
        "walkforward.val_months",
    ),
    (
        "walkforward_fractional_months",
        lambda raw: raw["walkforward"].__setitem__("test_months", 1.5),
        "walkforward.test_months",
    ),
    (
        "section_missing",
        lambda raw: raw.pop("risk"),
        "risk",
    ),
    (
        "live_mode_unknown",
        lambda raw: raw["live"].__setitem__("mode", "yolo"),
        "live.mode",
    ),
)


@pytest.mark.parametrize(
    ("mutate", "field"),
    [pytest.param(m, f, id=name) for name, m, f in INVALID_CASES],
)
def test_invalid_config_raises_naming_the_field(
    tmp_path: Path,
    raw_settings: dict[str, Any],
    mutate: Callable[[dict[str, Any]], None],
    field: str,
) -> None:
    """Every invalid variant raises ValueError naming the offending field path."""
    with pytest.raises(ValueError, match=re.escape(field)):
        load_mutated(tmp_path, raw_settings, mutate)


def test_error_message_states_requirement_and_value(
    tmp_path: Path, raw_settings: dict[str, Any]
) -> None:
    """The message format is the one named in the task: field, requirement, value."""
    with pytest.raises(ValueError) as excinfo:
        load_mutated(
            tmp_path,
            raw_settings,
            lambda raw: raw["risk"].__setitem__("max_position_pct", 1.4),
        )
    assert str(excinfo.value) == "risk.max_position_pct must be in (0, 1], got 1.4"


def test_not_a_mapping_raises(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


# ── Hashing ──────────────────────────────────────────────────────────────────


def test_hash_is_stable_across_loads() -> None:
    assert config_hash(load_config()) == config_hash(load_config())


def test_hash_is_a_sha256_hex_digest() -> None:
    digest = config_hash(load_config())
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_hash_ignores_key_order(tmp_path: Path, raw_settings: dict[str, Any]) -> None:
    """Reordering the file without changing a value leaves the hash unchanged."""
    reordered = {key: raw_settings[key] for key in sorted(raw_settings)}
    assert config_hash(load_config(write_settings(tmp_path, reordered))) == config_hash(
        load_config()
    )


@pytest.mark.parametrize(
    ("mutate", "label"),
    [
        (lambda raw: raw["meta"].__setitem__("seed", 7), "seed"),
        (lambda raw: raw["window"].__setitem__("horizon", 5), "horizon"),
        (lambda raw: raw["channels"].__setitem__("active", "C2_hybrid"), "channels"),
        (lambda raw: raw["risk"].__setitem__("stop_loss_pct", 0.04), "stop_loss"),
        (lambda raw: raw["universe"].append("TSLA"), "universe"),
        (lambda raw: raw["live"].__setitem__("mode", "auto"), "live_mode"),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_hash_changes_when_any_value_changes(
    tmp_path: Path,
    raw_settings: dict[str, Any],
    mutate: Callable[[dict[str, Any]], None],
    label: str,
) -> None:
    baseline = config_hash(load_config())
    assert config_hash(load_mutated(tmp_path, raw_settings, mutate)) != baseline


# ── Broker credentials ───────────────────────────────────────────────────────


def test_credentials_missing_raises_naming_the_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing credentials fail only when asked for, naming the variable."""
    for name in ALPACA_VARS:
        monkeypatch.delenv(name, raising=False)
    empty_env = tmp_path / ".env"
    empty_env.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="ALPACA_API_KEY"):
        alpaca_credentials(dotenv_path=empty_env)


def test_credentials_returned_when_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "key-123")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret-456")
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    empty_env = tmp_path / ".env"
    empty_env.write_text("", encoding="utf-8")

    creds = alpaca_credentials(dotenv_path=empty_env)

    assert isinstance(creds, AlpacaCredentials)
    assert creds.api_key == "key-123"
    assert creds.secret_key == "secret-456"
    # An unset endpoint defaults to paper: the safe value is the default, never None.
    assert creds.base_url == PAPER_ENDPOINT


def test_importing_the_loader_without_credentials_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Most of the system never touches the broker; import must stay silent."""
    for name in ALPACA_VARS:
        monkeypatch.delenv(name, raising=False)
    import importlib

    import glassbox.config.loader as loader_module

    importlib.reload(loader_module)


# ── The paper-endpoint guard ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "endpoint",
    [PAPER_ENDPOINT, f"{PAPER_ENDPOINT}/"],
    ids=["exact", "trailing_slash"],
)
def test_paper_endpoint_is_accepted(endpoint: str) -> None:
    require_paper_endpoint(endpoint)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.alpaca.markets",
        "https://api.alpaca.markets/",
        "http://paper-api.alpaca.markets",
        "https://paper-api.alpaca.markets.evil.example",
    ],
    ids=["live", "live_slash", "plain_http", "lookalike_host"],
)
def test_non_paper_endpoints_are_refused(endpoint: str) -> None:
    """Refusal, not a warning: the project never trades real money (spec 2.2)."""
    with pytest.raises(ValueError, match="refusing to connect"):
        require_paper_endpoint(endpoint)


@pytest.mark.parametrize("endpoint", [None, ""], ids=["none", "empty"])
def test_unset_endpoint_is_refused(endpoint: str | None) -> None:
    with pytest.raises(ValueError, match="ALPACA_BASE_URL is not set"):
        require_paper_endpoint(endpoint)


# ── The single-source-of-truth rule ──────────────────────────────────────────

FORBIDDEN_OUTSIDE_CONFIG = ("os.environ", "os.getenv", "getenv(", "settings.yaml")


def test_no_module_reads_settings_or_environ_directly(package_root: Path) -> None:
    """Only the config layer may read settings.yaml or the environment (CLAUDE.md 5)."""
    offenders: list[str] = []
    for path in package_root.rglob("*.py"):
        if path.parent.name == "config" or "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_OUTSIDE_CONFIG:
            if token in source:
                offenders.append(f"{path.relative_to(package_root)}: {token}")
    assert not offenders


def test_no_script_reads_the_environment_directly(repo_root: Path) -> None:
    """scripts/ sits outside the package, so extend the same rule to it explicitly."""
    offenders: list[str] = []
    for path in (repo_root / "scripts").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_OUTSIDE_CONFIG:
            if token in source:
                offenders.append(f"{path.relative_to(repo_root)}: {token}")
    assert not offenders
