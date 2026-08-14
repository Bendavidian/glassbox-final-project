"""The causality harness: perturb the future, prove the past did not move.

This is the single mechanism standing between this project and the failure mode that
inflates results across the whole field. It is deliberately small and deliberately
reusable — GB-10 applies it to the indicators and the keystone, GB-48 reapplies it to the
wavelets, and GB-25's leakage audit reaches for the fitting half.

**Why two perturbation modes.** Multiplying the future by a positive scalar is blind to
anything scale-invariant: ``sign(1.5 * x) == sign(x)`` exactly, so a function leaking
*tomorrow's direction* — which is one of this project's two headline metrics — passes a
multiplicative test untouched. Permuting the future rows is blind in the opposite
direction: a permutation preserves the multiset, so a leak through the full sample's mean
or variance survives it unchanged. Neither mode subsumes the other, and
``tests/features/test_no_lookahead.py`` proves that with one leak per mode. Both run by
default.

One caveat, measured rather than assumed: the comparison is exact, and floating-point
addition is not associative, so permuting rows moves a sum in its last ulp — about 1e-16
relative. That is enough to make ``assert_fit_isolated`` reject a whole-frame fitter under
``shuffle`` too, but for a numerical reason rather than an informational one. It never
causes a false rejection, because a correct function is handed byte-identical inputs.

**Two harnesses, because there are two leakage surfaces.** ``assert_causal`` covers
per-timestamp functions: the value at ``t`` must not move when bars after ``t`` change.
``assert_fit_isolated`` covers fitting, which has no per-timestamp value to compare — a
fitter is handed the whole frame and a boundary, so a fitter that forgets to slice is
caught rather than assumed away.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

# Fractions of the way through the index at which the future is cut off. Three splits,
# so a function that happens to be causal near one boundary is not mistaken for a causal
# function.
SPLIT_FRACTIONS = (0.25, 0.50, 0.75)

SCALE = "scale"
SHUFFLE = "shuffle"
DEFAULT_MODES = (SCALE, SHUFFLE)

# Frozen so that a failure reproduces exactly. This is test infrastructure, not a module
# parameter: a seed that varies between runs turns a real leak into a flaky test, which
# is worse than no test at all.
SHUFFLE_SEED = 1337

Frame = pd.DataFrame


def assert_causal(
    fn: Callable[[Frame], pd.Series | pd.DataFrame],
    frame: Frame,
    split_at: pd.Timestamp | None = None,
    perturbation: float = 1.5,
    modes: Sequence[str] = DEFAULT_MODES,
) -> None:
    """Assert ``fn`` uses no information after each timestamp.

    Computes ``fn(frame)``, then rebuilds it from a frame whose rows after ``split_at``
    have been perturbed, and asserts that every output value at or before ``split_at`` is
    bit-for-bit unchanged. Repeats at several split points and in every mode.

    Args:
        fn: A pure function from a bar frame to a Series or DataFrame indexed by a subset
            of the frame's index. Values that legitimately look forward — a target ``y``,
            for instance — must not be part of what it returns.
        frame: A canonical bar frame, indexed by a sorted DatetimeIndex.
        split_at: The boundary. ``None`` uses :data:`SPLIT_FRACTIONS` of the index.
        perturbation: The multiplier the ``scale`` mode applies to future rows.
        modes: Which perturbations to apply. See the module docstring for why the default
            is both.

    Raises:
        AssertionError: A value at or before a split moved when the future changed.
        ValueError: The harness could not make a meaningful comparison — an unknown mode,
            a split with no future or no output behind it, or a perturbation that left the
            input unchanged. A test that cannot fail must say so rather than pass.
    """
    splits = (
        _split_points(frame.index) if split_at is None else [pd.Timestamp(split_at)]
    )
    baseline = _as_frame(fn(_prepare(frame)))

    for split in splits:
        for mode in modes:
            perturbed = perturb(frame, split, mode, perturbation)
            observed = _as_frame(fn(perturbed))
            _assert_prefix_identical(baseline, observed, split, _name(fn), mode)


def assert_fit_isolated(
    fit: Callable[[Frame, pd.Timestamp], object],
    frame: Frame,
    split_at: pd.Timestamp | None = None,
    perturbation: float = 1.5,
    modes: Sequence[str] = DEFAULT_MODES,
) -> None:
    """Assert ``fit`` uses only the rows at or before the boundary it is given.

    Fitting is the other leakage surface, and ``assert_causal`` does not reach it: a set
    of normalisation statistics is one object for a whole range, not a value per
    timestamp. So ``fit`` is handed the **whole** frame and the boundary, exactly as a
    walk-forward caller holds it, and must slice internally. Perturbing the rows after the
    boundary must leave the fitted object equal to itself.

    Args:
        fit: ``fit(frame, split_at) -> object``. The object must support ``==``.
        frame: A canonical bar frame or feature frame.
        split_at: The training boundary. ``None`` uses :data:`SPLIT_FRACTIONS`.
        perturbation: The multiplier the ``scale`` mode applies.
        modes: Which perturbations to apply.

    Raises:
        AssertionError: The fitted object changed when data outside its range changed.
        ValueError: The harness could not make a meaningful comparison.
    """
    splits = (
        _split_points(frame.index) if split_at is None else [pd.Timestamp(split_at)]
    )

    for split in splits:
        baseline = fit(_prepare(frame), split)
        for mode in modes:
            observed = fit(perturb(frame, split, mode, perturbation), split)
            if observed != baseline:
                raise AssertionError(
                    f"{_name(fit)} is not isolated: perturbing rows after "
                    f"{split:%Y-%m-%d} ({mode}) changed what it fitted.\n"
                    f"  fitted on the unperturbed frame: {baseline}\n"
                    f"  fitted on the perturbed frame:   {observed}"
                )


def _prepare(frame: Frame) -> Frame:
    """A copy whose numeric columns are float.

    Applied to the baseline as well as the perturbed frame, so the two differ only by the
    perturbation. Without it an integer ``volume`` column would change dtype under
    ``scale`` alone, and a dtype difference would masquerade as a causality failure.
    """
    prepared = frame.copy()
    for column in prepared.select_dtypes(include="number").columns:
        prepared[column] = prepared[column].astype("float64")
    return prepared


def perturb(frame: Frame, split: pd.Timestamp, mode: str, perturbation: float) -> Frame:
    """Return a copy of ``frame`` whose rows after ``split`` have been disturbed.

    Public because a caller sometimes needs to measure *how much* a perturbation moved a
    result rather than only whether it moved at all — see the mode-complementarity tests
    in ``tests/features/test_no_lookahead.py``.
    """
    prepared = _prepare(frame)
    future = np.flatnonzero(prepared.index > split)
    if future.size == 0:
        raise ValueError(
            f"split {split:%Y-%m-%d} leaves no rows after it, so there is nothing to "
            "perturb and the assertion would pass vacuously"
        )

    if mode == SCALE:
        for column in prepared.select_dtypes(include="number").columns:
            values = prepared[column].to_numpy(copy=True)
            values[future] *= perturbation
            prepared[column] = values
    elif mode == SHUFFLE:
        order = np.random.default_rng(SHUFFLE_SEED).permutation(future)
        # One permutation for every column, so each row stays internally coherent — a
        # shuffled bar is still a real bar, just on the wrong day. Column by column, so
        # dtypes survive; a whole-frame reindex would collapse a mixed frame to object.
        for column in prepared.columns:
            values = prepared[column].to_numpy(copy=True)
            values[future] = values[order]
            prepared[column] = values
    else:
        raise ValueError(f"unknown perturbation mode {mode!r}; known: {DEFAULT_MODES}")

    _assert_perturbation_bites(frame, prepared, future, split, mode)
    return prepared


def _assert_perturbation_bites(
    original: Frame,
    perturbed: Frame,
    future: np.ndarray,
    split: pd.Timestamp,
    mode: str,
) -> None:
    """Refuse a perturbation that changed nothing.

    A shuffle of a one-row tail, or a scale factor of 1.0, would leave the input identical
    and every causality assertion would pass without testing anything.
    """
    before = _prepare(original).iloc[future]
    after = perturbed.iloc[future]
    if not _differs(before, after):
        raise ValueError(
            f"the {mode} perturbation left the {future.size} row(s) after "
            f"{split:%Y-%m-%d} unchanged, so it proves nothing"
        )


def _assert_prefix_identical(
    baseline: Frame, observed: Frame, split: pd.Timestamp, name: str, mode: str
) -> None:
    """Compare everything at or before ``split``, exactly."""
    before = baseline.loc[baseline.index <= split]
    after = observed.loc[observed.index <= split]

    if before.empty:
        raise ValueError(
            f"{name} produced no values at or before {split:%Y-%m-%d}, so the assertion "
            "would pass vacuously; use a longer frame or a later split"
        )
    if not before.index.equals(after.index):
        raise AssertionError(
            f"{name} changed the SHAPE of its output before {split:%Y-%m-%d} under "
            f"{mode}: {len(before)} rows became {len(after)}"
        )
    if not before.notna().to_numpy().any():
        raise ValueError(
            f"{name} is entirely NaN at or before {split:%Y-%m-%d}, so comparing the "
            "prefix proves nothing; use a later split or a longer warm-up"
        )

    if _differs(before, after):
        row, column = _first_difference(before, after)
        raise AssertionError(
            f"{name} LOOKS AHEAD: perturbing rows after {split:%Y-%m-%d} ({mode}) "
            f"changed its value at {before.index[row]:%Y-%m-%d}.\n"
            f"  column {column!r}: {before.iat[row, column]!r} became "
            f"{after.iat[row, column]!r}"
        )


def _differs(before: Frame, after: Frame) -> bool:
    """True if any cell moved, counting NaN as equal to NaN."""
    return bool(_difference_mask(before, after).any())


def _difference_mask(before: Frame, after: Frame) -> np.ndarray:
    left = before.to_numpy()
    right = after.to_numpy()
    both_nan = pd.isna(left) & pd.isna(right)
    return ~(np.equal(left, right) | both_nan)


def _first_difference(before: Frame, after: Frame) -> tuple[int, int]:
    row, column = np.argwhere(_difference_mask(before, after))[0]
    return int(row), int(column)


def _split_points(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """The timestamps at :data:`SPLIT_FRACTIONS` of the way through ``index``."""
    return [index[int(len(index) * fraction)] for fraction in SPLIT_FRACTIONS]


def _as_frame(output: pd.Series | pd.DataFrame) -> Frame:
    """Normalise a Series result to a one-column frame so one comparison covers both."""
    if isinstance(output, pd.Series):
        return output.to_frame()
    return output


def _name(fn: Callable) -> str:
    return getattr(fn, "__name__", repr(fn))
