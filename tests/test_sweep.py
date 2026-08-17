"""The sweep harness held to its own standard: prove it can fail.

``causality.py`` earned its place by rejecting a centred rolling mean and tomorrow's close.
This one earns its place the same way — the decisive test replays the measurement that
caused it to exist, the 352-bar parity floor, and requires the harness to report a fraction
strictly between none and all. A harness that answered "passes" there is the harness that
let the floor into the spec.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sweep import MIN_CELLS, Contest, SweepResult, SweepTooSmall, contest, sweep

from glassbox.config.loader import load_config
from glassbox.data.historical import load_history, normalise_bars
from glassbox.data.live import SOURCE_ALPACA
from glassbox.features.builder import build_feature_frame, build_windows

SYMBOLS = ("AAPL", "MSFT", "NVDA")
KEYS = (1, 2, 3, 4)


# ── it reports a fraction, and refuses to be a verdict ───────────────────────


def test_a_result_reports_the_fraction() -> None:
    result = sweep(lambda symbol, key: key != 3, SYMBOLS, KEYS)

    assert result.total == 12
    assert result.held == 9
    assert result.fraction == pytest.approx(0.75)
    assert not result.all_held


def test_a_result_refuses_to_be_a_boolean() -> None:
    """The property that would have caught the parity floor.

    ``if result:`` on a 9-of-12 sweep is true, and reading it that way is exactly how a
    claim that holds in a quarter of cells gets recorded as verified.
    """
    result = sweep(lambda symbol, key: key != 3, SYMBOLS, KEYS)

    with pytest.raises(TypeError, match="fraction, not a verdict"):
        bool(result)

    with pytest.raises(TypeError, match="9 of 12"):
        # The point is that evaluating the condition raises, so the body is dead.
        if result:
            pass


def test_the_refusal_names_what_to_read_instead() -> None:
    """An error that stops a caller without telling them what to do is a worse error."""
    result = sweep(lambda symbol, key: True, SYMBOLS, KEYS)

    with pytest.raises(
        TypeError, match=r"\.fraction, \.all_held, \.held or \.failures"
    ):
        bool(result)


def test_all_held_is_available_but_must_be_asked_for() -> None:
    """The strong claim is still expressible — it just has to be said out loud."""
    assert sweep(lambda symbol, key: True, SYMBOLS, KEYS).all_held


# ── it refuses a sweep of one ────────────────────────────────────────────────


def test_a_single_cell_is_refused() -> None:
    with pytest.raises(SweepTooSmall, match="not a sweep"):
        sweep(lambda symbol, key: True, ("AAPL",), (1,))


def test_the_refusal_explains_why_rather_than_just_refusing() -> None:
    with pytest.raises(SweepTooSmall, match="32 of 125"):
        sweep(lambda symbol, key: True, ("AAPL",), (1,))


def test_an_empty_axis_is_refused() -> None:
    with pytest.raises(SweepTooSmall):
        sweep(lambda symbol, key: True, (), KEYS)
    with pytest.raises(SweepTooSmall):
        sweep(lambda symbol, key: True, SYMBOLS, ())


def test_one_symbol_is_allowed_when_there_are_enough_keys() -> None:
    """What is refused is a sweep too small to disagree with itself, not a single symbol.

    A one-symbol sweep over many timestamps is still a sweep; it is simply weaker evidence
    about the universe, which the per-symbol failure counts make visible.
    """
    result = sweep(lambda symbol, key: key < 3, ("AAPL",), KEYS)

    assert result.total == MIN_CELLS * 2
    assert result.fraction == pytest.approx(0.5)


# ── failures are diagnosable without a rerun ─────────────────────────────────


def test_failures_carry_their_detail() -> None:
    result = sweep(
        lambda symbol, key: (key != 2, f"measured {key * 10}"), SYMBOLS, KEYS
    )

    assert len(result.failures) == 3
    assert all("measured 20" in str(cell.detail) for cell in result.failures)


def test_failures_are_counted_per_symbol() -> None:
    """Which symbol fails is the first question asked of a failing sweep."""
    result = sweep(lambda symbol, key: symbol != "NVDA", SYMBOLS, KEYS)

    assert result.failures_by_symbol() == {"NVDA": 4}
    assert "NVDA:4" in result.summary()


def test_a_clean_sweep_says_so() -> None:
    assert "failing: none" in sweep(lambda s, k: True, SYMBOLS, KEYS).summary()


# ── the decisive test: replay the failure that caused this harness ───────────


@pytest.mark.parametrize("tail,expectation", [(352, "none"), (445, "all")])
def test_the_harness_reproduces_the_parity_floor_finding(tail, expectation) -> None:
    """Replay the measurement that caused this harness to exist.

    At the declared floor every cell holds; at the old 352-bar floor none do. **When this
    was written the 352 case was 32 of 125** — a partial result, which is precisely what a
    hand measurement reports as "verified" if it happens to land on one of the 32. It reads
    0 of 125 now only because the second half of GB-27 unified the emit warm-up with the
    parity warm-up, so a 352-bar tail no longer assembles a window at all and the builder
    refuses instead of differing quietly.

    The synthetic partial sweeps above carry that property now; this one carries the real
    data. Both matter: a harness that could only ever report all-or-nothing would be a
    boolean wearing a fraction's clothes.
    """
    cfg = load_config()
    cache = Path(__file__).resolve().parents[1] / cfg.data.cache_dir
    if not all((cache / f"{s}.parquet").is_file() for s in cfg.universe):
        pytest.skip("no cached history; this test needs data_cache/")

    bars = load_history(list(cfg.universe), cfg)
    common = None
    for frame in bars.values():
        common = frame.index if common is None else common.intersection(frame.index)
    stamps = list(common[: -cfg.window.horizon - 1][-5:])

    trained = {}
    for symbol, frame in bars.items():
        batch = build_windows(build_feature_frame(frame, cfg), cfg, symbol)
        trained[symbol] = {s: batch.X[i] for i, s in enumerate(batch.timestamps)}

    def identical(symbol: str, stamp: pd.Timestamp) -> tuple[bool, str]:
        position = bars[symbol].index.get_loc(stamp)
        raw = bars[symbol].iloc[max(0, position + 1 - tail) : position + 1]
        try:
            frame = build_feature_frame(
                normalise_bars(
                    raw[["open", "high", "low", "close", "volume"]],
                    symbol,
                    SOURCE_ALPACA,
                ),
                cfg,
            )
            live = build_windows(frame, cfg, symbol, as_of=stamp).X[0]
        except ValueError as refusal:
            # Below the floor the builder refuses outright since the warm-ups were
            # unified. A refusal is a cell that does not hold, recorded as such.
            return False, f"refused: {str(refusal)[:60]}"
        reference = trained[symbol][stamp]
        return bool(np.array_equal(live, reference)), (
            f"max|diff|={float(np.abs(live - reference).max()):.3e}"
        )

    result = sweep(identical, list(cfg.universe), stamps)

    assert result.total == 25
    if expectation == "all":
        assert result.all_held, result.summary()
    else:
        assert result.held == 0, result.summary()
        assert all("refused" in str(cell.detail) for cell in result.failures)


# ── contest: for every "A beats B" claim ─────────────────────────────────────


def test_a_contest_counts_wins_by_cell_not_by_mean() -> None:
    """An arm that wins narrowly nine times and loses catastrophically once is a different
    animal from one that wins on average, and a mean cannot tell them apart."""
    scores = {"steady": [1.0] * 4, "spiky": [0.9, 0.9, 0.9, 99.0]}

    outcome = contest(
        lambda symbol, key, arm: scores[arm][key - 1],
        ("AAPL",),
        KEYS,
        ("steady", "spiky"),
        higher_is_better=True,
    )

    assert outcome.wins() == {"steady": 3, "spiky": 1}
    assert outcome.means()["spiky"] > outcome.means()["steady"]


def test_a_contest_reports_pairwise_cells() -> None:
    outcome = contest(
        lambda symbol, key, arm: {"a": 1.0, "b": 2.0}[arm] + key,
        SYMBOLS,
        KEYS,
        ("a", "b"),
        higher_is_better=False,
    )

    assert outcome.beats("a", "b") == 12
    assert outcome.beats("b", "a") == 0
    assert outcome.wins() == {"a": 12, "b": 0}


def test_a_contest_of_one_arm_is_refused() -> None:
    with pytest.raises(SweepTooSmall, match="at least two arms"):
        contest(lambda s, k, a: 1.0, SYMBOLS, KEYS, ("only",))


def test_a_contest_of_one_cell_is_refused() -> None:
    with pytest.raises(SweepTooSmall, match="not a sweep"):
        contest(lambda s, k, a: 1.0, ("AAPL",), (1,), ("a", "b"))


def test_lower_is_better_is_honoured() -> None:
    """MAE and drawdown are minimised; a harness that assumed otherwise would rank the
    study backwards and read perfectly plausibly."""
    outcome = contest(
        lambda symbol, key, arm: {"good": 0.1, "bad": 0.9}[arm],
        SYMBOLS,
        KEYS,
        ("good", "bad"),
        higher_is_better=False,
    )

    assert outcome.wins() == {"good": 12, "bad": 0}


def test_the_types_are_what_the_module_exports() -> None:
    result = sweep(lambda s, k: True, SYMBOLS, KEYS)
    outcome = contest(lambda s, k, a: 1.0, SYMBOLS, KEYS, ("a", "b"))

    assert isinstance(result, SweepResult)
    assert isinstance(outcome, Contest)
