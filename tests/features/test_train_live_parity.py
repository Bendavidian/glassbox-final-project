"""GB-27: the window the live loop builds is the window training built. Byte for byte.

This is the test the whole live path rests on. If it fails, every backtest number
describes a system that cannot be run, and no amount of downstream correctness recovers
that — the model would be scored on inputs it never sees in production.

**The comparison is exact.** ``np.array_equal``, not ``allclose``: a window that differs
in the last bit is a window computed by a different route, and "close enough" is how a
one-bar offset survives review. A tolerance is a place bugs live.

**It sweeps rather than samples, and that is the whole lesson of this task.** The first
version of this file checked one symbol at one timestamp, found byte-identity at the
then-declared floor of 352 bars, and passed. A sweep over five symbols and twenty-five
timestamps found byte-identity in **32 of 125 pairs** at that floor. The single point was
one of the lucky ones. A parity test that samples can certify a floor that does not hold,
which is exactly what happened, so this one sweeps by construction and any future channel
inherits the sweep.

**What "the live path" means here.** The same bars, taken as a **tail** rather than a full
history, passed through ``historical.normalise_bars`` with Alpaca's provenance — which is
what ``data/live.py`` does with an API response — then through the keystone with
``as_of=T``. That reproduces every difference the live loop actually has: a shorter frame,
a different source label, a single window instead of a batch, and a ``log_return`` column
recomputed from the tail rather than inherited from ten years of history.

Three things a naive parity test would miss, each with its own section: the parity floor
from GB-9 (corrected in GB-27), the single-source rule from GB-8, and the per-window
``symbols`` tuple the universe-wide training change introduced.

**A second correction followed the first.** GB-8 held a separate emit warm-up of 77 rows,
from a 1e-2 target with the same known-bad derivation, answering the same question in a
different unit. Unifying them at 325 rows means a tail below the declared floor no longer
assembles a window at all — the builder refuses. The silent-difference failure this file
was written to catch is now unreachable through the public path, which is why the
below-floor tests assert a **refusal** rather than a divergence.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glassbox.config.loader import Config, load_config
from glassbox.contracts.schemas import WindowBatch
from glassbox.data.historical import SOURCE_YFINANCE, load_history, normalise_bars
from glassbox.data.live import SOURCE_ALPACA
from glassbox.features.builder import (
    build_feature_frame,
    build_windows,
    fit_stats,
    min_history_bars,
)

SYMBOL = "AAPL"
OTHER = "MSFT"

# The floor derived at a 1e-9 target rather than 1e-10. Measured over the same sweep:
# 120 of 125 pairs, so it is NOT universally byte-identical and the extra margin in the
# declared floor is doing real work rather than padding a number.
FLOOR_AT_1E9 = 414

# Enough timestamps that a lucky one cannot carry the suite. Twenty-five, over five
# symbols, is 125 comparisons per assertion.
SWEEP_TIMESTAMPS = 25


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config()


@pytest.fixture(scope="module")
def universe(cfg: Config) -> dict[str, pd.DataFrame]:
    """Every symbol, through the **training** loader, or skip. No network, ever.

    ``load_history`` rather than ``read_parquet``: the cache file carries no ``source``
    column and the loader stamps it, so reading the parquet directly would compare the live
    path against something training never sees. The files are checked first so the loader
    cannot fall through to a download — the guard ``smoke_offline`` uses, for that reason.
    """
    cache = Path(__file__).resolve().parents[2] / cfg.data.cache_dir
    for symbol in cfg.universe:
        if not (cache / f"{symbol}.parquet").is_file():
            pytest.skip(f"no cached history for {symbol}; this test needs data_cache/")
    return load_history(list(cfg.universe), cfg)


@pytest.fixture(scope="module")
def trained_windows(
    universe: dict[str, pd.DataFrame], cfg: Config
) -> dict[str, dict[pd.Timestamp, np.ndarray]]:
    """Every training window, by symbol and timestamp. Built once; the sweep reads it."""
    built: dict[str, dict[pd.Timestamp, np.ndarray]] = {}
    for symbol, bars in universe.items():
        batch = build_windows(build_feature_frame(bars, cfg), cfg, symbol)
        built[symbol] = {
            stamp: batch.X[row] for row, stamp in enumerate(batch.timestamps)
        }
    return built


@pytest.fixture(scope="module")
def sweep_stamps(universe: dict[str, pd.DataFrame], cfg: Config) -> list[pd.Timestamp]:
    """Recent timestamps every symbol shares, excluding the last H where no target exists."""
    common = None
    for bars in universe.values():
        common = bars.index if common is None else common.intersection(bars.index)
    return list(common[: -cfg.window.horizon - 1][-SWEEP_TIMESTAMPS:])


def as_live(
    bars: pd.DataFrame, at: pd.Timestamp, tail: int, symbol: str
) -> pd.DataFrame:
    """The ``tail`` bars ending **at** ``at``, re-normalised as Alpaca would return them.

    Ending at ``at`` because that is what the live loop holds: the session's last completed
    bar is the one it is deciding on. Through ``normalise_bars`` — the single definition of
    the schema — rather than by relabelling, so the live path's own recomputations happen,
    including a ``log_return`` column rebuilt from the tail whose first row is NaN where the
    training frame inherits a real value from the bar before.
    """
    position = bars.index.get_loc(at)
    raw = bars.iloc[max(0, position + 1 - tail) : position + 1]
    return normalise_bars(
        raw[["open", "high", "low", "close", "volume"]], symbol, SOURCE_ALPACA
    )


def live_window(
    bars: pd.DataFrame, cfg: Config, at: pd.Timestamp, tail: int, symbol: str = SYMBOL
) -> WindowBatch:
    """The live path: a tail of bars, Alpaca provenance, one window ``as_of`` that bar."""
    frame = build_feature_frame(as_live(bars, at, tail, symbol), cfg)
    return build_windows(frame, cfg, symbol, as_of=at)


def sweep(
    universe: dict[str, pd.DataFrame],
    trained: dict[str, dict[pd.Timestamp, np.ndarray]],
    stamps: list[pd.Timestamp],
    cfg: Config,
    tail: int,
) -> tuple[int, int, dict[str, int]]:
    """``(identical, total, {symbol: failures})`` over every symbol and timestamp."""
    failures: dict[str, int] = {}
    total = 0
    for symbol, bars in universe.items():
        for stamp in stamps:
            if stamp not in trained[symbol]:
                continue
            total += 1
            live = live_window(bars, cfg, stamp, tail, symbol).X[0]
            if not np.array_equal(live, trained[symbol][stamp]):
                failures[symbol] = failures.get(symbol, 0) + 1
    return total - sum(failures.values()), total, failures


# ── the parity assertion, swept ──────────────────────────────────────────────


def test_every_symbol_at_every_timestamp_is_byte_identical_at_the_floor(
    universe, trained_windows, sweep_stamps, cfg: Config
) -> None:
    """The standing assertion. 125 comparisons, exact equality, no exceptions.

    A single-point version of this test certified a floor that held in 32 of 125 pairs.
    This is that mistake made structurally impossible.
    """
    identical, total, failures = sweep(
        universe, trained_windows, sweep_stamps, cfg, min_history_bars(cfg)
    )

    assert total == SWEEP_TIMESTAMPS * len(cfg.universe)
    assert identical == total, f"{total - identical} of {total} differ: {failures}"


def test_the_floor_is_the_derived_number(cfg: Config) -> None:
    """445 = 120 + 14 + 311, where (13/14)^311 < 1e-10.

    Pinned so that a change has to come with a change to the derivation in
    ``builder.PARITY_WARMUP``, rather than being tuned until a sweep passes — which is how
    the previous floor of 352 arrived.
    """
    import math

    decay = math.ceil(math.log(1e-10) / math.log(13 / 14))

    assert decay == 311
    assert min_history_bars(cfg) == cfg.window.input_len + 14 + decay == 445


@pytest.mark.parametrize("tail", [352, FLOOR_AT_1E9])
def test_below_the_declared_floor_the_builder_refuses_rather_than_differing(
    universe, sweep_stamps, cfg: Config, tail: int
) -> None:
    """The stronger property GB-27's second half bought, and it replaced a weaker one.

    Until the emit warm-up was unified with the parity warm-up, a 352- or 414-bar tail
    **assembled a window and returned slightly wrong numbers** — silently, in 93 and 5 of
    125 cells respectively. Now the warm-up trim takes 325 rows, so a 352-bar tail leaves
    27 rows against an input length of 120 and the builder **refuses**.

    That is a real trade: the 414-vs-445 divergence can no longer be observed through the
    public path, so the measurement that justified the 1e-10 margin (120 of 125 at 414)
    stands as a recorded result in DECISIONS rather than as a live assertion. In exchange,
    the failure mode it measured is now impossible to reach silently — which is the better
    end of the trade, because a refusal cannot be mistaken for a valid window.
    """
    assert tail < min_history_bars(cfg)

    for symbol in sorted(universe):
        with pytest.raises(ValueError, match="fewer than|bars behind it"):
            live_window(universe[symbol], cfg, sweep_stamps[-1], tail, symbol)


def test_the_refusal_names_the_shortfall(universe, sweep_stamps, cfg: Config) -> None:
    """A loop that asks for too little history must be told what it asked for."""
    with pytest.raises(ValueError, match="120"):
        live_window(universe[SYMBOL], cfg, sweep_stamps[-1], 352, SYMBOL)


def test_both_paths_produce_float32(
    universe, trained_windows, sweep_stamps, cfg: Config
) -> None:
    """A dtype difference would survive `allclose` and change every downstream number."""
    at = sweep_stamps[-1]
    live = live_window(universe[SYMBOL], cfg, at, min_history_bars(cfg))

    assert live.X.dtype == np.float32
    assert trained_windows[SYMBOL][at].dtype == np.float32


def test_the_channel_tuples_match_exactly(universe, cfg: Config, sweep_stamps) -> None:
    """Order included: the model reads channels positionally, so a reordering is a
    different input wearing the same name."""
    live = live_window(universe[SYMBOL], cfg, sweep_stamps[-1], min_history_bars(cfg))

    assert live.channels == cfg.channels.active_channels


def test_parity_survives_normalisation(universe, cfg: Config, sweep_stamps) -> None:
    """The live loop scales with the checkpoint's statistics, so parity must hold after
    scaling too — the same statistics applied to the same window."""
    at = sweep_stamps[-1]
    frame = build_feature_frame(universe[SYMBOL], cfg)
    stats = fit_stats(frame.iloc[: len(frame) // 2], cfg)

    trained = build_windows(frame, cfg, SYMBOL, stats=stats)
    row = int(np.flatnonzero(trained.timestamps == at)[0])
    live = build_windows(
        build_feature_frame(
            as_live(universe[SYMBOL], at, min_history_bars(cfg), SYMBOL), cfg
        ),
        cfg,
        SYMBOL,
        stats=stats,
        as_of=at,
    )

    assert np.array_equal(live.X[0], trained.X[row])


# ── teeth ────────────────────────────────────────────────────────────────────


def test_a_one_bar_offset_breaks_parity(
    universe, trained_windows, sweep_stamps, cfg: Config
) -> None:
    """Proof the equality above is asserting something.

    The offset is the failure this test exists to catch — a live loop that includes the
    in-progress bar, or drops one too many, is off by exactly this much.
    """
    at = sweep_stamps[-1]
    earlier = sweep_stamps[-2]

    offset = live_window(universe[SYMBOL], cfg, earlier, min_history_bars(cfg))

    assert not np.array_equal(offset.X[0], trained_windows[SYMBOL][at])


def test_a_bar_after_the_window_cannot_change_it(
    universe, trained_windows, sweep_stamps, cfg: Config
) -> None:
    """The correction to an error in the first version of this file.

    It asserted that dropping the tail's last bar breaks parity. It does not, and must not:
    every channel is trailing, so a bar **after** the window's end cannot reach into it.
    That is GB-10's causality property, and asserting the opposite was asserting that the
    keystone is broken. Here it is, the right way round.
    """
    at = sweep_stamps[-3]
    floor = min_history_bars(cfg)

    exactly = live_window(universe[SYMBOL], cfg, at, floor).X[0]
    with_more_after = build_windows(
        build_feature_frame(
            as_live(universe[SYMBOL], sweep_stamps[-1], floor + 2, SYMBOL), cfg
        ),
        cfg,
        SYMBOL,
        as_of=at,
    ).X[0]

    assert np.array_equal(exactly, trained_windows[SYMBOL][at])
    assert np.array_equal(with_more_after, trained_windows[SYMBOL][at])


def test_too_little_history_raises_rather_than_differing_quietly(
    universe, cfg: Config, sweep_stamps
) -> None:
    """The second correction. Below the input length the builder **refuses**.

    The first version expected a silent difference at 197 bars and failed because the
    builder raises instead — which is the better behaviour, and asserting it is worth more
    than asserting a difference. A loud refusal cannot be mistaken for a valid window.
    """
    at = sweep_stamps[-1]

    with pytest.raises(ValueError, match="bars behind it|fewer than"):
        live_window(universe[SYMBOL], cfg, at, tail=cfg.window.input_len - 3)


# ── the single-source rule (GB-8), in the parity context ─────────────────────


def test_a_spliced_frame_is_refused(universe, cfg: Config, sweep_stamps) -> None:
    """Splicing is exactly what a live loop is tempted to do: keep the cached history and
    append today's Alpaca bar.

    Tested in GB-9 already; tested again here because this is the file a reader consults
    when asking "can I just append?". The answer is no, and the reason is measured: the two
    vendors' volume disagrees by 34-111 bps (GB-7), so a spliced frame computes `vol_z`
    across a discontinuity that is a vendor artefact rather than a market event.
    """
    history = universe[SYMBOL].iloc[:-5]
    recent = as_live(universe[SYMBOL], sweep_stamps[-1], 5, SYMBOL)

    with pytest.raises(ValueError, match="source"):
        build_feature_frame(pd.concat([history, recent]), cfg)


def test_each_path_alone_is_single_source(universe, cfg: Config, sweep_stamps) -> None:
    """And the provenance travels, so a downstream consumer can tell them apart."""
    trained = build_feature_frame(universe[SYMBOL], cfg)
    live = build_feature_frame(
        as_live(universe[SYMBOL], sweep_stamps[-1], min_history_bars(cfg), SYMBOL), cfg
    )

    assert trained.attrs["source"] == SOURCE_YFINANCE
    assert live.attrs["source"] == SOURCE_ALPACA


def test_parity_holds_across_the_source_difference(
    universe, trained_windows, sweep_stamps, cfg: Config
) -> None:
    """The provenance differs and the numbers do not. That is the point of routing both
    sources through one `normalise_bars`: identity by construction, not by agreement."""
    at = sweep_stamps[-1]
    live = live_window(universe[SYMBOL], cfg, at, min_history_bars(cfg))

    assert live.source == SOURCE_ALPACA
    assert np.array_equal(live.X[0], trained_windows[SYMBOL][at])


# ── per-window symbols, from the universe-wide training change ───────────────


def test_the_live_window_carries_its_own_symbol(
    universe, cfg: Config, sweep_stamps
) -> None:
    """One entry per window, length B, since the `WindowBatch.symbols` change."""
    live = live_window(universe[SYMBOL], cfg, sweep_stamps[-1], min_history_bars(cfg))

    assert live.symbols == (SYMBOL,)
    assert len(live.symbols) == live.X.shape[0] == 1


def test_pooling_live_windows_preserves_each_symbol(
    universe, trained_windows, sweep_stamps, cfg: Config
) -> None:
    """The live loop scores the universe in one batch and the checkpoint applies a
    per-symbol scaler. A pooled batch that lost track of which row was which would scale
    AAPL with MSFT's statistics and nothing downstream would notice."""
    at = sweep_stamps[-1]
    floor = min_history_bars(cfg)

    pooled = WindowBatch.concat(
        [
            live_window(universe[symbol], cfg, at, floor, symbol)
            for symbol in (SYMBOL, OTHER)
        ]
    )

    assert pooled.symbols == (SYMBOL, OTHER)
    assert pooled.X.shape[0] == 2
    assert np.array_equal(pooled.X[0], trained_windows[SYMBOL][at])
    assert np.array_equal(pooled.X[1], trained_windows[OTHER][at])


def test_the_whole_universe_pools_into_one_batch_that_matches_training(
    universe, trained_windows, sweep_stamps, cfg: Config
) -> None:
    """Parity for the shape the live loop actually scores: every symbol, one timestamp."""
    at = sweep_stamps[-1]
    floor = min_history_bars(cfg)
    symbols = sorted(universe)

    pooled = WindowBatch.concat(
        [live_window(universe[symbol], cfg, at, floor, symbol) for symbol in symbols]
    )

    assert pooled.symbols == tuple(symbols)
    for row, symbol in enumerate(symbols):
        assert np.array_equal(pooled.X[row], trained_windows[symbol][at])


# ── the wavelets, swept at the same floor (GB-47) ────────────────────────────


@pytest.fixture(scope="module")
def hybrid_cfg(cfg: Config) -> Config:
    """``C2_hybrid``: the five base channels plus ``wav_a1..a3``."""
    return replace(cfg, channels=replace(cfg.channels, active="C2_hybrid"))


@pytest.fixture(scope="module")
def hybrid_windows(
    universe: dict[str, pd.DataFrame], hybrid_cfg: Config
) -> dict[str, dict[pd.Timestamp, np.ndarray]]:
    built: dict[str, dict[pd.Timestamp, np.ndarray]] = {}
    for symbol, bars in universe.items():
        batch = build_windows(build_feature_frame(bars, hybrid_cfg), hybrid_cfg, symbol)
        built[symbol] = {
            stamp: batch.X[row] for row, stamp in enumerate(batch.timestamps)
        }
    return built


def test_the_wavelet_channels_are_byte_identical_at_the_floor(
    universe, hybrid_windows, sweep_stamps, hybrid_cfg: Config
) -> None:
    """GB-47's channels inherit the sweep rather than being certified at a point.

    The lesson of this file is that a parity test which samples can certify a floor that
    does not hold, so a new channel set is swept the same way the old one is — five
    symbols, twenty-five timestamps, exact equality.

    **The floor does not move**, which was not the expectation. The wavelet warm-up is 64
    bars and RSI's is 325, and ``min_history_bars`` is a maximum rather than a sum, so 445
    stands for both channel sets. And the two numbers are different *kinds*: RSI's is a
    tolerance argument about a decaying seed, re-derived once already; the wavelets' is
    exact, because a DWT of a trailing window depends on that window and nothing before it.
    """
    assert min_history_bars(hybrid_cfg) == 445

    identical, total, failures = sweep(
        universe,
        hybrid_windows,
        sweep_stamps,
        hybrid_cfg,
        min_history_bars(hybrid_cfg),
    )

    assert total == SWEEP_TIMESTAMPS * len(hybrid_cfg.universe)
    assert identical == total, f"{total - identical} of {total} differ: {failures}"


def test_the_hybrid_window_carries_all_eight_channels(
    universe, sweep_stamps, hybrid_cfg: Config
) -> None:
    """Otherwise the sweep above could pass on a window that quietly dropped a channel."""
    symbol = min(universe)
    window = live_window(
        universe[symbol],
        hybrid_cfg,
        sweep_stamps[-1],
        min_history_bars(hybrid_cfg),
        symbol,
    )

    assert window.channels == hybrid_cfg.channels.active_channels
    assert len(window.channels) == 8
    assert window.X.shape == (1, hybrid_cfg.window.input_len, 8)
