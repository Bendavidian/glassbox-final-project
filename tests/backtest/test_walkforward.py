"""GB-17 acceptance: folds are ordered, bounded, on real sessions, and embargoed.

The embargo tests are the ones that matter. Everything else here would pass for a fold
generator that leaks the first `H` bars of every validation range into training — the
ranges are disjoint, no timestamp is shared, and the leak is invisible to every ordering
assertion. So those tests come with a teeth test that sets the embargo to zero and
requires them to fail.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import pandas as pd
import pytest

from glassbox.backtest import walkforward as wf
from glassbox.config.loader import Config, load_config

# Long enough for the 30-month fold span to fit several times over.
SERIES_START = pd.Timestamp("2016-01-01", tz="UTC")
SERIES_END = pd.Timestamp("2026-08-13", tz="UTC")


@pytest.fixture
def cfg() -> Config:
    return load_config()


@pytest.fixture
def index(cfg: Config) -> pd.DatetimeIndex:
    """Real NYSE sessions, so a boundary landing on a holiday would be visible."""
    return wf.trading_days(cfg, SERIES_START, SERIES_END)


@pytest.fixture
def folds(index: pd.DatetimeIndex, cfg: Config) -> list[wf.Fold]:
    return wf.make_folds(index, cfg)


# ── ordering and bounds ──────────────────────────────────────────────────────


def test_splits_are_ordered_within_a_fold(folds: list[wf.Fold]) -> None:
    assert folds
    for fold in folds:
        assert fold.train.max() < fold.val.min()
        assert fold.val.min() < fold.val.max()
        assert fold.val.max() < fold.test.min()
        assert fold.test.min() <= fold.test.max()


def test_no_timestamp_appears_in_two_splits_of_one_fold(folds: list[wf.Fold]) -> None:
    for fold in folds:
        combined = fold.train.append(fold.val).append(fold.test)
        assert combined.is_unique


def test_folds_are_strictly_ordered_in_time(folds: list[wf.Fold]) -> None:
    """Each fold's test period is entirely after the one before it."""
    for earlier, later in pairwise(folds):
        assert earlier.train.min() < later.train.min()
        assert earlier.val.min() < later.val.min()
        assert earlier.test.max() < later.test.max()
        assert earlier.test.min() < later.test.min()


def test_folds_are_numbered_in_order(folds: list[wf.Fold]) -> None:
    assert [fold.number for fold in folds] == list(range(1, len(folds) + 1))


def test_fold_count_respects_max_folds(index: pd.DatetimeIndex, cfg: Config) -> None:
    assert len(wf.make_folds(index, cfg)) <= cfg.walkforward.max_folds

    capped = replace(cfg, walkforward=replace(cfg.walkforward, max_folds=3))
    assert len(wf.make_folds(index, capped)) == 3


def test_the_cap_keeps_the_most_recent_folds(
    index: pd.DatetimeIndex, cfg: Config
) -> None:
    """Recorded in DECISIONS.md: the oldest regime is the one to drop, not the newest."""
    uncapped = replace(cfg, walkforward=replace(cfg.walkforward, max_folds=10_000))
    capped = replace(cfg, walkforward=replace(cfg.walkforward, max_folds=3))

    everything = wf.make_folds(index, uncapped)
    kept = wf.make_folds(index, capped)

    assert len(everything) > len(kept)
    assert [f.test[-1] for f in kept] == [f.test[-1] for f in everything[-3:]]


def test_a_short_series_yields_fewer_folds_rather_than_raising(
    index: pd.DatetimeIndex, cfg: Config
) -> None:
    """Fewer folds, never a partial one, never an exception."""
    counts = [len(wf.make_folds(index[:size], cfg)) for size in (400, 900, 1400, 2000)]

    assert counts == sorted(counts)
    assert counts[0] < counts[-1]


def test_a_series_too_short_for_one_fold_yields_no_folds(
    index: pd.DatetimeIndex, cfg: Config
) -> None:
    """ "No fold fits" is an answer. The caller reports it; it is not an error here."""
    assert wf.make_folds(index[:200], cfg) == []
    assert wf.make_folds(index[: cfg.window.input_len], cfg) == []


def test_a_truncated_final_fold_is_dropped_not_shortened(
    index: pd.DatetimeIndex, cfg: Config
) -> None:
    """A fold reporting two months of test beside folds reporting three is not comparable.

    Cutting the series just before a fold's test range completes must remove that fold
    entirely rather than hand back a short one.
    """
    uncapped = replace(cfg, walkforward=replace(cfg.walkforward, max_folds=10_000))
    full = wf.make_folds(index, uncapped)
    last = full[-1]

    trimmed = wf.make_folds(index[index < last.test[-1]], uncapped)

    assert len(trimmed) == len(full) - 1
    assert all(fold.test[-1] < last.test[-1] for fold in trimmed)


# ── the calendar ─────────────────────────────────────────────────────────────


def test_fold_boundaries_land_on_trading_days(
    folds: list[wf.Fold], index: pd.DatetimeIndex, cfg: Config
) -> None:
    sessions = set(wf.trading_days(cfg, index[0], index[-1]))

    for fold in folds:
        for split in (fold.train, fold.val, fold.test):
            assert split[0] in sessions
            assert split[-1] in sessions


def test_boundaries_snap_to_sessions_instead_of_calendar_arithmetic(
    folds: list[wf.Fold], index: pd.DatetimeIndex, cfg: Config
) -> None:
    """The naive generator uses the ``+N months`` date itself as the boundary.

    That date is a weekend or a holiday often enough to matter, so at least one fold must
    show a boundary that is *not* a clean month-anniversary of the grid anchor. If none
    does, the module is placing dates rather than selecting bars.
    """
    anchor = index[cfg.window.input_len - 1]
    anniversaries = {
        anchor + pd.DateOffset(months=months) for months in range(0, 200, 1)
    }

    starts = [split[0] for fold in folds for split in (fold.train, fold.val, fold.test)]

    assert any(start not in anniversaries for start in starts)


# ── the embargo: the point of this module ────────────────────────────────────


def test_the_embargo_is_exactly_the_horizon(cfg: Config) -> None:
    """Derived in ``embargo_bars``: a target spans p+1 .. p+H, so H ends are unusable."""
    assert wf.embargo_bars(cfg) == cfg.window.horizon


def test_target_range_is_the_h_bars_after_the_window(
    index: pd.DatetimeIndex, cfg: Config
) -> None:
    """The definition the embargo is derived from, asserted rather than assumed."""
    position = 500
    targets = wf.target_range(index, index[position], cfg)

    assert list(targets) == list(
        index[position + 1 : position + 1 + cfg.window.horizon]
    )
    assert len(targets) == cfg.window.horizon


def test_no_training_target_reaches_the_validation_range(
    folds: list[wf.Fold], index: pd.DatetimeIndex, cfg: Config
) -> None:
    """**The leak this module exists to prevent.**

    A window ending on the last training bar is labelled with the first `H` validation
    returns. Disjoint ranges do not prevent it, because the label reaches forward out of
    its own range.
    """
    assert folds
    for fold in folds:
        boundary = fold.val[0]
        for end in fold.train:
            targets = wf.target_range(index, end, cfg)
            assert targets.max() < boundary, (
                f"fold {fold.number}: the training window ending {end:%Y-%m-%d} is "
                f"labelled with returns through {targets.max():%Y-%m-%d}, which is at or "
                f"past the validation range starting {boundary:%Y-%m-%d}"
            )


def test_no_validation_target_reaches_the_test_range(
    folds: list[wf.Fold], index: pd.DatetimeIndex, cfg: Config
) -> None:
    """The same leak at the second boundary, where threshold calibration would absorb it."""
    assert folds
    for fold in folds:
        boundary = fold.test[0]
        for end in fold.val:
            targets = wf.target_range(index, end, cfg)
            assert targets.max() < boundary, (
                f"fold {fold.number}: the validation window ending {end:%Y-%m-%d} is "
                f"labelled with returns through {targets.max():%Y-%m-%d}, which is at or "
                f"past the test range starting {boundary:%Y-%m-%d}"
            )


def test_every_test_window_has_a_complete_target(
    folds: list[wf.Fold], index: pd.DatetimeIndex, cfg: Config
) -> None:
    """Every fold's test score is computable: no window is left with a partial label."""
    for fold in folds:
        assert len(wf.target_range(index, fold.test[-1], cfg)) == cfg.window.horizon


def test_no_test_target_reaches_the_next_folds_test_range(
    folds: list[wf.Fold], index: pd.DatetimeIndex, cfg: Config
) -> None:
    """Not leakage — nothing is trained on test — but independence between folds.

    If fold i's last test windows were labelled with returns from fold i+1's test range,
    consecutive folds' scores would share outcome bars. GB-51 runs a paired Wilcoxon
    across folds and treats each fold as one observation, so overlapping outcomes would
    correlate the very numbers the test assumes independent. This is why the test split is
    embargoed too, even though no model is fitted on it.
    """
    for fold, following in pairwise(folds):
        reach = wf.target_range(index, fold.test[-1], cfg).max()
        assert reach < following.test[0]


# ── the embargo tests have teeth ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "leak_test",
    [
        test_no_training_target_reaches_the_validation_range,
        test_no_validation_target_reaches_the_test_range,
    ],
    ids=["train-into-val", "val-into-test"],
)
def test_the_leak_tests_fail_without_an_embargo(
    leak_test, index: pd.DatetimeIndex, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Set the embargo to zero and the leak must reappear.

    Without this, both assertions above could be vacuously true — a fold generator that
    returned empty splits would satisfy them, and so would one whose target definition was
    wrong in the same direction as the test's.
    """
    monkeypatch.setattr(wf, "embargo_bars", lambda _cfg: 0)
    unembargoed = wf.make_folds(index, cfg)

    assert unembargoed
    assert all(fold.embargo == 0 for fold in unembargoed)

    with pytest.raises(AssertionError, match="which is at or past"):
        leak_test(unembargoed, index, cfg)


def test_an_embargo_of_zero_is_the_only_difference(
    index: pd.DatetimeIndex, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The teeth test must change the embargo and nothing else.

    If removing it also changed the fold count or the boundaries, the failure above would
    not be evidence about the embargo.
    """
    embargoed = wf.make_folds(index, cfg)

    monkeypatch.setattr(wf, "embargo_bars", lambda _cfg: 0)
    unembargoed = wf.make_folds(index, cfg)

    assert len(unembargoed) == len(embargoed)
    for without, with_ in zip(unembargoed, embargoed, strict=True):
        assert without.train[0] == with_.train[0]
        assert len(without.train) == len(with_.train) + cfg.window.horizon
        assert len(without.val) == len(with_.val) + cfg.window.horizon
        assert len(without.test) == len(with_.test) + cfg.window.horizon


# ── input validation ─────────────────────────────────────────────────────────


def test_a_naive_index_is_refused(index: pd.DatetimeIndex, cfg: Config) -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        wf.make_folds(index.tz_localize(None), cfg)


def test_an_unsorted_index_is_refused(index: pd.DatetimeIndex, cfg: Config) -> None:
    with pytest.raises(ValueError, match="sorted"):
        wf.make_folds(index[::-1], cfg)


def test_a_duplicated_index_is_refused(index: pd.DatetimeIndex, cfg: Config) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        wf.make_folds(index.append(index[-1:]), cfg)
