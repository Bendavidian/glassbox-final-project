"""GB-2 acceptance: the config layer loads, validates, fails loud and hashes stably."""

from __future__ import annotations

import ast
import copy
import io
import re
import tokenize
from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from glassbox.config.loader import (
    DEFAULT_SETTINGS_PATH,
    MODEL_SHAPING_SECTIONS,
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
    model_config_hash,
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
    assert cfg.universe == (
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "TSLA",
        "LLY",
        "JPM",
        "JNJ",
        "V",
        "XOM",
        "UNH",
        "MA",
        "MRK",
        "PG",
        "HD",
        "COST",
        "WMT",
        "ABBV",
    )
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


def a_ticker_absent_from(universe: list[str]) -> str:
    """An uppercase ticker guaranteed not already in ``universe``.

    **Derived rather than named, and that is the whole of the fix.** The universe mutation
    below used to append the literal ``"TSLA"``, which made the test's validity rest on a
    fact about `settings.yaml` that the test did not control: GB-61 put TSLA in the
    universe, the append became a duplicate, `_build_universe` refuses duplicates by
    design, and the case **errored** instead of asserting anything about hashing. Any other
    hardcoded ticker carries the same defect one universe change later.
    """
    candidate = "A"
    while candidate in universe:
        candidate += "A"
    return candidate


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
        (
            lambda raw: raw["universe"].append(a_ticker_absent_from(raw["universe"])),
            "universe",
        ),
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


def code_only(source: str) -> str:
    """``source`` with comments and docstrings blanked, **every other string kept**.

    **The guard below scanned raw text, and a guard whose value is a property of the file
    rather than of the code punishes whoever explains the code.** A comment naming the
    settings file failed it exactly as an ``open()`` of that file would, so the cheapest
    way to go green was always to delete the sentence saying where a value came from - a
    mechanism working against the thing it protects, on a project whose fifth rule is that
    config is the single source of truth and whose modules are therefore expected to say
    so. It cost a real sentence on 3 Sep 2026: the note on
    :data:`~glassbox.dashboard.app.EQUITY_MIN_SPAN` explaining that the constant is pinned
    to the risk policy had to be written around the filename before the suite would pass.

    **Fourth costume of this family**, after the 200-character slice, the `from_daily`
    occurrence count, and the count inflated by the comment introducing the thing it
    counted. It is the worst of the four for a reason the others do not share: those three
    broke *when* somebody explained the code, while this one made the explanation itself
    the offence, so the incentive it created was to write less down.

    **String literals other than docstrings are deliberately kept, and that is the whole
    design.** An ``open()`` of the settings file *is* a string literal; stripping literals
    would blind the guard to the only spelling of the defect it exists to catch. Comments
    and docstrings are prose by definition and can never be a read; every other literal
    can be.

    Blanked rather than deleted, so line and column offsets stay valid while later spans
    are applied to the same text.
    """
    lines = source.splitlines(keepends=True)
    spans: list[tuple[int, int, int, int]] = []

    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, holders) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            doc = first.value
            spans.append(
                (doc.lineno, doc.col_offset, doc.end_lineno, doc.end_col_offset)
            )

    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            spans.append((*token.start, *token.end))

    for first_line, first_col, last_line, last_col in spans:
        for number in range(first_line, last_line + 1):
            line = lines[number - 1]
            start = first_col if number == first_line else 0
            end = last_col if number == last_line else len(line.rstrip("\n"))
            lines[number - 1] = line[:start] + " " * (end - start) + line[end:]
    return "".join(lines)


def forbidden_in(source: str) -> list[str]:
    """Every forbidden token this source *runs*, ignoring every one it merely mentions."""
    code = code_only(source)
    return [token for token in FORBIDDEN_OUTSIDE_CONFIG if token in code]


def test_no_module_reads_settings_or_environ_directly(package_root: Path) -> None:
    """Only the config layer may read the settings file or the environment (CLAUDE.md 5).

    Measured on the code, not on the file: see :func:`code_only` for why, and for the
    sentence this guard deleted before it was fixed.
    """
    offenders: list[str] = []
    for path in package_root.rglob("*.py"):
        if path.parent.name == "config" or "__pycache__" in path.parts:
            continue
        offenders += [
            f"{path.relative_to(package_root)}: {token}"
            for token in forbidden_in(path.read_text(encoding="utf-8"))
        ]
    assert not offenders


def test_no_script_reads_the_environment_directly(repo_root: Path) -> None:
    """scripts/ sits outside the package, so extend the same rule to it explicitly."""
    offenders: list[str] = []
    for path in (repo_root / "scripts").rglob("*.py"):
        offenders += [
            f"{path.relative_to(repo_root)}: {token}"
            for token in forbidden_in(path.read_text(encoding="utf-8"))
        ]
    assert not offenders


def test_the_guard_catches_a_real_read_and_not_a_mention_of_one() -> None:
    """**Both directions, because a guard loosened in one direction is deleted in the
    other.**

    Loosening this rule could never be verified by the suite going green: the suite was
    green the moment the explanation was deleted, which is the outcome being fixed. So the
    two halves are asserted against each other. Every spelling of an actual read must
    still be caught - including the two that live inside string literals, which is exactly
    what :func:`code_only` must not strip - and prose naming the same thing must not be.
    """
    reads = (
        'CONFIG = open("settings.yaml")',
        'PATH = Path(__file__).parent / "settings.yaml"',
        'KEY = os.environ["ALPACA_KEY"]',
        "KEY = os.getenv('ALPACA_KEY')",
        "KEY = getenv('ALPACA_KEY')",
    )
    for source in reads:
        assert forbidden_in(source), f"the guard stopped seeing a real read: {source}"

    mentions = (
        "# the floor is pinned to settings.yaml by a test\nX = 1\n",
        '"""Pinned to settings.yaml; os.environ belongs to the config layer."""\nX = 1\n',
        (
            'def f():\n    """Never reads settings.yaml, and os.getenv is forbidden."""\n'
            "    return 1\n"
        ),
        (
            "class C:\n"
            '    """Holds no settings.yaml path and calls no getenv(."""\n'
            "    value = 1\n"
        ),
    )
    for source in mentions:
        found = forbidden_in(source)
        assert not found, f"prose still counts as a read: {found} in {source!r}"


# ── the model hash is narrower than the config hash (ruled 20 Aug 2026) ──────


def test_a_live_only_key_leaves_the_model_hash_untouched() -> None:
    """Measured on 19 Aug 2026 and fixed here.

    Adding `live.retry_attempts` and `live.retry_backoff_seconds` — two numbers a polling
    loop reads and no weight can see — changed `config_hash` and made `load_checkpoint`
    refuse every existing model. The guard was right and the cost was pure waste. The
    danger is not that retrain; it is Sprint 4, where FITS and the COF sweep touch the
    configuration repeatedly and a guard that fires spuriously is one somebody weakens.
    """
    base = load_config()
    live_only = replace(
        base,
        live=replace(base.live, retry_attempts=9, poll_seconds=15, mode="auto"),
    )

    assert config_hash(live_only) != config_hash(base)
    assert model_config_hash(live_only) == model_config_hash(base)


def test_changing_the_input_length_changes_the_model_hash() -> None:
    """The other direction, which is what stops the split being a hole."""
    base = load_config()
    wider = replace(base, window=replace(base.window, input_len=60))

    assert model_config_hash(wider) != model_config_hash(base)
    assert config_hash(wider) != config_hash(base)


def test_every_model_shaping_section_moves_the_model_hash() -> None:
    """Named one by one, so a section added to the list without effect is caught."""
    base = load_config()
    moved = {
        "data": replace(base, data=replace(base.data, start="2017-01-01")),
        "window": replace(base, window=replace(base.window, horizon=8)),
        "wavelet": replace(base, wavelet=replace(base.wavelet, levels=2)),
        "fits": replace(base, fits=replace(base.fits, cutoff_period_days=10)),
        "channels": replace(base, channels=replace(base.channels, active="C2_hybrid")),
        "model": replace(base, model=replace(base.model, epochs=7)),
    }
    for section, changed in moved.items():
        assert model_config_hash(changed) != model_config_hash(base), section


def test_the_seed_is_part_of_the_model_hash_and_the_schema_version_is_not() -> None:
    """Two models under identical settings and different seeds are different models.
    `meta.version` versions the configuration schema, which no weight sees."""
    base = load_config()
    reseeded = replace(base, meta=replace(base.meta, seed=99))
    reversioned = replace(base, meta=replace(base.meta, version=base.meta.version + 1))

    assert model_config_hash(reseeded) != model_config_hash(base)
    assert model_config_hash(reversioned) == model_config_hash(base)


def test_the_full_hash_still_covers_everything() -> None:
    """`config_hash` does not narrow. It is what a DecisionRecord promises: the settings a
    decision was made under, including the ones that decided not to trade."""
    base = load_config()
    for changed in (
        replace(base, live=replace(base.live, poll_seconds=15)),
        replace(base, risk=replace(base.risk, max_position_pct=0.05)),
        replace(base, signal=replace(base.signal, top_k=3)),
    ):
        assert config_hash(changed) != config_hash(base)


def test_every_config_section_is_classified_as_shaping_or_not() -> None:
    """A new top-level section must be classified before the suite goes green again.

    This is the answer to the objection the old whole-config gate recorded: *a list of
    "fields that matter" is wrong the first time someone adds a field and forgets it.*
    `MODEL_SHAPING_SECTIONS` lists **sections**, so a new field inside `window` or `model`
    is covered the moment it exists. The remaining hole is a whole new section, and this
    closes it — adding one fails here until somebody has said which side it falls on.
    """
    # Sections that describe what the system DOES with a model rather than what shaped
    # one. Named individually rather than derived, because the point is that a human
    # decided each.
    not_shaping = {
        "meta",  # only `meta.seed` shapes a weight, and it is added explicitly
        "universe",  # `Predictor.stats_for` refuses an untrained symbol by name
        "signal",
        "risk",
        "backtest",
        "walkforward",
        "live",
        # **`wits` shapes weights and is still listed here, which is the one entry in this
        # set that is a dated hold rather than a verdict.** GB-66's boundary and level keys
        # decide a WITS weight, so on the merits the section belongs in
        # MODEL_SHAPING_SECTIONS - and the comment there says so and says when it moves.
        # Adding it changes `model_config_hash` for every configuration, which on
        # 25 Aug 2026 would have had `load_predictor` refuse the deployed checkpoint hours
        # before the GATE 2 rehearsal. Held until the grid re-run, on GB-61's precedent.
        # A WITS checkpoint carries its own geometry and is rebuilt from it, so nothing can
        # load at a geometry it was not trained at meanwhile.
        "wits",
    }
    sections = {field.name for field in fields(load_config())}

    assert sections == set(MODEL_SHAPING_SECTIONS) | not_shaping, (
        "a configuration section is neither in MODEL_SHAPING_SECTIONS nor listed here as "
        "not shaping a trained weight. Classify it: if a weight can depend on it, it "
        "belongs in the model hash and every checkpoint must be retrained"
    )


# ── the spec's copies of the universe ────────────────────────────────────────

#: The spec, read as text. **No missing-file guard, deliberately**: if this file is gone
#: the copies cannot be checked, and a test that skips when it cannot check is the defect
#: this one exists to close.
SPEC = Path(__file__).resolve().parents[2] / "docs" / "GLASSBOX_PROJECT_SPEC.md"

#: `universe: [A, B, C]` - the mirrored §5 settings block.
SETTINGS_BLOCK = re.compile(r"^universe: \[([^\]]+)\]$", re.MULTILINE)

#: One `| 3 | `NVDA` |` cell of §2.4's four-column table. The exclusion table below it
#: writes the ticker first and the criterion second, so this shape does not match it.
NUMBERED_CELL = re.compile(r"\|\s*(\d+)\s*\|\s*`([A-Z][A-Z.]*)`\s*\|")

#: §2's count claim. It said 20 while the settings block still said 5, unseen, until
#: GB-61 stage 2 - which is the divergence this whole test exists because of.
COUNT_CLAIM = re.compile(r"Fixed universe: \*\*(\d+) symbols\*\*")

#: What the parser must find. **Asserted explicitly**, because a parser that silently
#: matches nothing gives a test that passes by having checked nothing.
SPEC_COPIES = ("settings block 1", "rule-applied table")


def universes_in(text: str) -> dict[str, tuple[str, ...]]:
    """Every written list of the universe in the spec, keyed by where it is written.

    The rule-applied table is bounded to its own section rather than scanned for
    globally: `str.index` raises if either marker moves, which is the correct outcome -
    a parser that quietly finds nothing is worse than one that fails loudly.
    """
    copies = {
        f"settings block {index}": tuple(
            symbol.strip() for symbol in match.group(1).split(",")
        )
        for index, match in enumerate(SETTINGS_BLOCK.finditer(text), start=1)
    }
    start = text.index("**The rule applied")
    end = text.index("**Excluded candidates", start)
    numbered = {int(n): ticker for n, ticker in NUMBERED_CELL.findall(text[start:end])}
    copies["rule-applied table"] = tuple(numbered[index] for index in sorted(numbered))
    return copies


def test_every_universe_written_in_the_spec_matches_the_configuration() -> None:
    """**Three copies of the universe live in two files and nothing held them equal.**

    On 2 Sep 2026 §2 line 75 read "20 symbols" while the §5 settings block still listed
    five, and §2.4's table listed twenty. The spec contradicted itself and had done since
    before anyone noticed, because no check has ever parsed this document. `settings.yaml`
    is the only copy the suite could see, through a tuple pinned by hand in
    `test_default_config_values_match_the_spec`.

    This reads the spec and compares every copy against `load_config()`. It is deliberately
    unskippable: a missing spec file raises rather than skipping, and the copy count is
    asserted so a parser that matched nothing cannot pass for having checked nothing.
    """
    cfg = load_config()
    text = SPEC.read_text(encoding="utf-8")

    copies = universes_in(text)

    assert tuple(sorted(copies)) == tuple(
        sorted(SPEC_COPIES)
    ), f"expected {SPEC_COPIES}, parsed {tuple(sorted(copies))}"
    for where, symbols in copies.items():
        assert symbols == cfg.universe, (
            f"{where} disagrees with settings.yaml: "
            f"spec has {symbols}, config has {cfg.universe}"
        )

    claimed = COUNT_CLAIM.search(text)
    assert claimed, "§2's universe count claim is no longer parseable"
    assert int(claimed.group(1)) == len(cfg.universe), (
        f"§2 claims {claimed.group(1)} symbols, the configuration holds "
        f"{len(cfg.universe)}"
    )
