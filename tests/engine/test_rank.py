"""GB-28 acceptance: the same inputs select the same names, in the same order, always.

Ranking is three lines of sorting, and the tests are about the one property that is not
obvious: the tie-break. A stable sort with no second key resolves ties by the caller's
insertion order, which in the live loop is the order symbols came back from a broker call —
a real source of run-to-run difference that no test would otherwise see.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from glassbox.config.loader import Config, load_config
from glassbox.contracts.schemas import Signal
from glassbox.engine import rank
from glassbox.engine.signal import ENTER_LONG, EXIT, HOLD


@pytest.fixture
def cfg() -> Config:
    return load_config()


def signal(symbol: str, strength: float, action: str = ENTER_LONG) -> Signal:
    return Signal(
        symbol=symbol,
        action=action,
        trend_strength=strength,
        up_points=4,
        passed_threshold=True,
    )


# ── ordering ─────────────────────────────────────────────────────────────────


def test_the_strongest_come_first(cfg: Config) -> None:
    ranked = rank.rank_signals(
        [signal("AAPL", 0.01), signal("MSFT", 0.05), signal("NVDA", 0.03)], cfg
    )

    assert [s.symbol for s in ranked] == ["MSFT", "NVDA"]


def test_only_the_top_k_survive(cfg: Config) -> None:
    ranked = rank.rank_signals(
        [signal(s, 0.1 - i / 100) for i, s in enumerate("ABCDE")], cfg
    )

    assert cfg.signal.top_k == 2
    assert len(ranked) == 2


def test_identical_inputs_always_produce_identical_ordering(cfg: Config) -> None:
    """The acceptance criterion. Run it repeatedly and compare the whole ordering."""
    signals = [signal("AAPL", 0.02), signal("MSFT", 0.02), signal("NVDA", 0.09)]

    orderings = {
        tuple(s.symbol for s in rank.rank_signals(signals, cfg)) for _ in range(50)
    }

    assert len(orderings) == 1


def test_the_input_order_does_not_change_the_output(cfg: Config) -> None:
    """The live loop builds its list from a dict, and dict order is insertion order, which
    is broker-response order. If that reached the ranking, two identical days would select
    differently and nothing downstream would notice."""
    strengths = {"AAPL": 0.02, "MSFT": 0.02, "NVDA": 0.02, "AMZN": 0.05}
    forward = [signal(s, v) for s, v in strengths.items()]
    backward = list(reversed(forward))

    assert [s.symbol for s in rank.rank_signals(forward, cfg)] == [
        s.symbol for s in rank.rank_signals(backward, cfg)
    ]


# ── the tie-break ────────────────────────────────────────────────────────────


def test_ties_resolve_alphabetically(cfg: Config) -> None:
    ranked = rank.rank_signals(
        [signal("NVDA", 0.04), signal("AAPL", 0.04), signal("MSFT", 0.04)], cfg
    )

    assert [s.symbol for s in ranked] == ["AAPL", "MSFT"]


def test_the_tie_break_is_ascending_even_though_strength_is_descending(
    cfg: Config,
) -> None:
    """`reverse=True` would have reversed the tie-break too, and the alphabetical order
    would silently run Z to A. Negating the strength keeps the two keys independent."""
    wide = replace(cfg, signal=replace(cfg.signal, top_k=4))
    ranked = rank.rank_signals(
        [signal("ZZZZ", 0.04), signal("AAAA", 0.04), signal("MMMM", 0.09)], wide
    )

    assert [s.symbol for s in ranked] == ["MMMM", "AAAA", "ZZZZ"]


def test_a_partial_tie_orders_within_the_tied_group_only(cfg: Config) -> None:
    wide = replace(cfg, signal=replace(cfg.signal, top_k=5))
    ranked = rank.rank_signals(
        [
            signal("ZZZZ", 0.09),
            signal("BBBB", 0.04),
            signal("AAAA", 0.04),
            signal("CCCC", 0.11),
        ],
        wide,
    )

    assert [s.symbol for s in ranked] == ["CCCC", "ZZZZ", "AAAA", "BBBB"]


# ── what competes, and what does not ─────────────────────────────────────────


def test_only_enter_long_competes_for_a_slot(cfg: Config) -> None:
    """An exit is an obligation on capital already committed. It must never be crowded out
    by a better opportunity elsewhere, so it is not a candidate at all."""
    ranked = rank.rank_signals(
        [
            signal("AAPL", 0.01),
            signal("MSFT", 0.90, action=EXIT),
            signal("NVDA", 0.80, action=HOLD),
        ],
        cfg,
    )

    assert [s.symbol for s in ranked] == ["AAPL"]


def test_an_empty_slate_ranks_to_nothing(cfg: Config) -> None:
    assert rank.rank_signals([], cfg) == []
    assert rank.rank_signals([signal("AAPL", 0.5, action=HOLD)], cfg) == []


def test_ranking_returns_the_signals_unmodified(cfg: Config) -> None:
    """It selects and orders. Anything else belongs to the risk layer."""
    original = signal("AAPL", 0.05)

    ranked = rank.rank_signals([original, signal("MSFT", 0.01)], cfg)

    assert ranked[0] is original


def test_a_negative_strength_can_still_be_ranked_if_it_asked_to_enter(
    cfg: Config,
) -> None:
    """Ranking does not second-guess the decision layer. If a band admitted it, it competes
    — ordering is this module's only opinion."""
    ranked = rank.rank_signals([signal("AAPL", -0.02), signal("MSFT", -0.05)], cfg)

    assert [s.symbol for s in ranked] == ["AAPL", "MSFT"]


# ── refusals ─────────────────────────────────────────────────────────────────


def test_a_non_finite_strength_is_refused(cfg: Config) -> None:
    """NaN compares false against everything, so sorting it is silently order-dependent —
    the exact class of bug the tie-break exists to remove."""
    with pytest.raises(ValueError, match="non-finite"):
        rank.rank_signals([signal("AAPL", math.nan), signal("MSFT", 0.01)], cfg)


def test_a_non_candidate_with_a_nan_strength_is_ignored(cfg: Config) -> None:
    """Only candidates are checked: a `hold` never reaches the sort, so its strength cannot
    affect an ordering and refusing the whole slate over it would be theatre."""
    ranked = rank.rank_signals(
        [signal("AAPL", 0.01), signal("MSFT", math.nan, action=HOLD)], cfg
    )

    assert [s.symbol for s in ranked] == ["AAPL"]


def test_top_k_comes_from_the_config(cfg: Config) -> None:
    narrow = replace(cfg, signal=replace(cfg.signal, top_k=1))
    signals = [signal("AAPL", 0.05), signal("MSFT", 0.09)]

    assert len(rank.rank_signals(signals, narrow)) == 1
    assert len(rank.rank_signals(signals, cfg)) == 2
