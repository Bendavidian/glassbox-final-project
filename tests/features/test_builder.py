"""GB-9 acceptance: the keystone assembles one window the same way twice.

The properties under test are the ones every downstream module inherits: channel order
is the config's, statistics come from outside, the live path and the training path are
one path, and a window ending at t contains nothing from after t.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glassbox.config.loader import Config, load_config
from glassbox.contracts.schemas import ChannelStats, WindowBatch
from glassbox.features import builder

BARS = 500
INPUT_LEN = 30
HORIZON = 4


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """A smaller window than production, so a fixture of a few hundred bars is enough."""
    base = load_config()
    return replace(
        base,
        data=replace(base.data, cache_dir=str(tmp_path / "cache")),
        window=replace(base.window, input_len=INPUT_LEN, horizon=HORIZON),
    )


def make_bars(
    n: int = BARS, source: str = "yfinance", scale: float = 1.0
) -> pd.DataFrame:
    """A canonical bar frame: deterministic, rising and falling, single provenance."""
    index = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    close = pd.Series(
        [scale * (100.0 + 10.0 * math.sin(i / 7.0) + 0.05 * i) for i in range(n)],
        index=index,
        dtype="float64",
    )
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": pd.Series(
                [
                    1_000_000.0 + 5_000.0 * math.cos(i / 5.0) + 100.0 * i
                    for i in range(n)
                ],
                index=index,
            ),
            "log_return": np.log(close / close.shift(1)),
            "source": pd.Series(source, index=index, dtype="string"),
        },
        index=index,
    )


# ── build_feature_frame ──────────────────────────────────────────────────────


def test_channel_order_is_exactly_the_config_order(cfg: Config) -> None:
    frame = builder.build_feature_frame(make_bars(), cfg)

    assert tuple(frame.columns) == cfg.channels.active_channels


def test_no_nans_survive_the_warmup_trim(cfg: Config) -> None:
    frame = builder.build_feature_frame(make_bars(), cfg)

    assert not frame.isna().to_numpy().any()
    assert len(frame) < BARS  # the warm-up really was trimmed


def test_close_logret_channel_is_the_log_return_series(cfg: Config) -> None:
    """Spec 4.1 fixes the model input series as log returns, and the name says so."""
    bars = make_bars()
    frame = builder.build_feature_frame(bars, cfg)

    pd.testing.assert_series_equal(
        frame["close_logret"], bars.loc[frame.index, "log_return"], check_names=False
    )


def test_feature_frame_carries_provenance(cfg: Config) -> None:
    frame = builder.build_feature_frame(make_bars(source="alpaca"), cfg)

    assert frame.attrs["source"] == "alpaca"


def test_c2_hybrid_raises_until_the_wavelets_land(cfg: Config) -> None:
    """Dropping the wav_* channels silently would let the study run a C2 arm that is
    really C0 and report it as a wavelet result."""
    c2 = replace(cfg, channels=replace(cfg.channels, active="C2_hybrid"))

    with pytest.raises(ValueError, match=r"wav_a1.*GB-47"):
        builder.build_feature_frame(make_bars(), c2)


# ── Provenance enforcement ───────────────────────────────────────────────────


def test_spliced_provenance_raises(cfg: Config) -> None:
    """The failure this guards: a 4-9% volume step inside vol_z's normalising window."""
    yfinance_half = make_bars(n=600, source="yfinance").iloc[:300]
    alpaca_half = make_bars(n=600, source="alpaca").iloc[300:]
    spliced = pd.concat([yfinance_half, alpaca_half])

    with pytest.raises(ValueError, match=r"mix 2 sources"):
        builder.build_feature_frame(spliced, cfg)


def test_missing_provenance_column_raises(cfg: Config) -> None:
    bars = make_bars().drop(columns=["source"])

    with pytest.raises(ValueError, match="carry no source column"):
        builder.build_feature_frame(bars, cfg)


def test_windows_refuse_a_frame_without_provenance(cfg: Config) -> None:
    frame = builder.build_feature_frame(make_bars(), cfg)
    stripped = frame.copy()
    stripped.attrs.clear()

    with pytest.raises(ValueError, match="carries no provenance"):
        builder.build_windows(stripped, cfg, "AAPL")


def test_batch_records_the_source_it_was_built_from(cfg: Config) -> None:
    frame = builder.build_feature_frame(make_bars(source="alpaca"), cfg)
    batch = builder.build_windows(frame, cfg, "AAPL")

    assert batch.source == "alpaca"


# ── min_history_bars ─────────────────────────────────────────────────────────


def test_min_history_bars_for_c0() -> None:
    """120 + 232: the input window plus RSI's parity warm-up, the largest in C0."""
    assert builder.min_history_bars(load_config()) == 352


def test_min_history_bars_tracks_the_active_channel_set(cfg: Config) -> None:
    """Derived from the config, so GB-47's wavelets update it without an edit here."""
    smaller = replace(
        cfg,
        channels=replace(
            cfg.channels,
            sets=(("only_close", ("close_logret",)),),
            active="only_close",
        ),
    )
    assert builder.min_history_bars(smaller) == cfg.window.input_len + 1


def test_min_history_bars_raises_for_an_undeclared_channel(cfg: Config) -> None:
    exotic = replace(
        cfg,
        channels=replace(
            cfg.channels,
            sets=(("exotic", ("close_logret", "wav_a1")),),
            active="exotic",
        ),
    )
    with pytest.raises(ValueError, match="wav_a1"):
        builder.min_history_bars(exotic)


# ── build_windows: shapes and contents ───────────────────────────────────────


def test_batch_shapes_and_dtypes(cfg: Config) -> None:
    frame = builder.build_feature_frame(make_bars(), cfg)
    batch = builder.build_windows(frame, cfg, "AAPL")

    channels = len(cfg.channels.active_channels)
    expected_windows = len(frame) - INPUT_LEN + 1 - HORIZON

    assert isinstance(batch, WindowBatch)
    assert batch.X.shape == (expected_windows, INPUT_LEN, channels)
    assert batch.y.shape == (expected_windows, HORIZON)
    assert batch.X.dtype == np.float32
    assert batch.y.dtype == np.float32
    assert batch.channels == cfg.channels.active_channels
    assert batch.symbol == "AAPL"
    assert len(batch.timestamps) == expected_windows


def test_last_row_of_x_is_the_frame_row_at_t(cfg: Config) -> None:
    frame = builder.build_feature_frame(make_bars(), cfg)
    batch = builder.build_windows(frame, cfg, "AAPL")

    for row in (0, 17, len(batch.timestamps) - 1):
        timestamp = batch.timestamps[row]
        expected = frame.loc[timestamp, list(cfg.channels.active_channels)]
        np.testing.assert_allclose(
            batch.X[row, -1, :], expected.to_numpy(dtype="float32"), rtol=0, atol=0
        )


def test_first_row_of_x_is_input_len_minus_one_bars_earlier(cfg: Config) -> None:
    frame = builder.build_feature_frame(make_bars(), cfg)
    batch = builder.build_windows(frame, cfg, "AAPL")

    position = frame.index.get_loc(batch.timestamps[5])
    expected = frame.iloc[position - INPUT_LEN + 1][list(cfg.channels.active_channels)]
    np.testing.assert_allclose(
        batch.X[5, 0, :], expected.to_numpy(dtype="float32"), rtol=0, atol=0
    )


def test_y_is_the_next_h_close_log_returns(cfg: Config) -> None:
    frame = builder.build_feature_frame(make_bars(), cfg)
    batch = builder.build_windows(frame, cfg, "AAPL")

    for row in (0, 42, len(batch.timestamps) - 1):
        position = frame.index.get_loc(batch.timestamps[row])
        expected = frame["close_logret"].iloc[position + 1 : position + 1 + HORIZON]
        np.testing.assert_allclose(
            batch.y[row], expected.to_numpy(dtype="float32"), rtol=0, atol=0
        )


def test_no_window_reaches_past_its_own_timestamp(cfg: Config) -> None:
    """Causality, stated directly: perturb every bar after t and the window ending at t
    is unchanged."""
    bars = make_bars()
    frame = builder.build_feature_frame(bars, cfg)
    batch = builder.build_windows(frame, cfg, "AAPL")

    cut = 300
    perturbed_bars = bars.copy()
    tail = perturbed_bars.index[cut + 1 :]
    perturbed_bars.loc[tail, ["open", "high", "low", "close"]] *= 1.5
    perturbed_bars.loc[tail, "volume"] *= 4.0
    perturbed_bars["log_return"] = np.log(
        perturbed_bars["close"] / perturbed_bars["close"].shift(1)
    )
    perturbed = builder.build_windows(
        builder.build_feature_frame(perturbed_bars, cfg), cfg, "AAPL"
    )

    keep = batch.timestamps <= bars.index[cut]
    assert keep.any()
    np.testing.assert_array_equal(batch.X[keep], perturbed.X[keep])


# ── build_windows: the live path is the training path ────────────────────────


def test_as_of_is_byte_identical_to_the_batch_row(cfg: Config) -> None:
    """GB-27 in miniature. np.array_equal, not approximately equal."""
    frame = builder.build_feature_frame(make_bars(), cfg)
    batch = builder.build_windows(frame, cfg, "AAPL")

    for row in (0, 1, 123, len(batch.timestamps) - 1):
        timestamp = batch.timestamps[row]
        single = builder.build_windows(frame, cfg, "AAPL", as_of=timestamp)

        assert single.X.shape == (1, INPUT_LEN, len(cfg.channels.active_channels))
        assert np.array_equal(single.X[0], batch.X[row])
        assert np.array_equal(single.y[0], batch.y[row])
        assert single.timestamps[0] == timestamp


def test_as_of_is_byte_identical_with_stats_applied(cfg: Config) -> None:
    frame = builder.build_feature_frame(make_bars(), cfg)
    stats = builder.fit_stats(frame.iloc[:200], cfg)
    batch = builder.build_windows(frame, cfg, "AAPL", stats=stats)

    row = 150
    single = builder.build_windows(
        frame, cfg, "AAPL", stats=stats, as_of=batch.timestamps[row]
    )

    assert np.array_equal(single.X[0], batch.X[row])


def test_as_of_at_the_last_bar_yields_a_nan_target(cfg: Config) -> None:
    """The live case: the next H bars do not exist yet, and saying so beats inventing."""
    frame = builder.build_feature_frame(make_bars(), cfg)
    single = builder.build_windows(frame, cfg, "AAPL", as_of=frame.index[-1])

    assert single.X.shape[0] == 1
    assert np.isnan(single.y).all()


def test_as_of_outside_the_index_raises(cfg: Config) -> None:
    frame = builder.build_feature_frame(make_bars(), cfg)

    with pytest.raises(ValueError, match="is not in the frame's index"):
        builder.build_windows(
            frame, cfg, "AAPL", as_of=pd.Timestamp("1999-01-04", tz="UTC")
        )


def test_as_of_without_enough_history_raises(cfg: Config) -> None:
    frame = builder.build_feature_frame(make_bars(), cfg)

    with pytest.raises(ValueError, match="bars behind it"):
        builder.build_windows(frame, cfg, "AAPL", as_of=frame.index[3])


def test_a_frame_shorter_than_the_window_raises(cfg: Config) -> None:
    frame = builder.build_feature_frame(make_bars(), cfg).iloc[: INPUT_LEN - 1]

    with pytest.raises(ValueError, match="fewer than the 30"):
        builder.build_windows(frame, cfg, "AAPL")


# ── Statistics are external ──────────────────────────────────────────────────


def test_stats_are_not_fitted_inside_build_windows(cfg: Config) -> None:
    """Stats fitted on a different range must change X. If they did not, build_windows
    would be normalising with something it computed itself — the leakage path."""
    frame = builder.build_feature_frame(make_bars(), cfg)

    early = builder.fit_stats(frame.iloc[:150], cfg)
    late = builder.fit_stats(frame.iloc[250:], cfg)

    unnormalised = builder.build_windows(frame, cfg, "AAPL")
    with_early = builder.build_windows(frame, cfg, "AAPL", stats=early)
    with_late = builder.build_windows(frame, cfg, "AAPL", stats=late)

    assert not np.array_equal(with_early.X, unnormalised.X)
    assert not np.array_equal(with_early.X, with_late.X)


def test_fit_stats_records_the_range_it_was_fitted_on(cfg: Config) -> None:
    """GB-25's leakage audit intersects this range against the fold's test range."""
    frame = builder.build_feature_frame(make_bars(), cfg)
    training = frame.iloc[:200]

    stats = builder.fit_stats(training, cfg)

    assert isinstance(stats, ChannelStats)
    assert stats.channels == cfg.channels.active_channels
    assert stats.fitted_start == training.index[0]
    assert stats.fitted_end == training.index[-1]
    assert stats.n_rows == 200


def test_fit_stats_values_match_the_frame(cfg: Config) -> None:
    frame = builder.build_feature_frame(make_bars(), cfg)
    training = frame.iloc[:200]

    stats = builder.fit_stats(training, cfg)

    for position, channel in enumerate(stats.channels):
        assert stats.mean[position] == pytest.approx(
            training[channel].mean(), abs=1e-12
        )
        assert stats.std[position] == pytest.approx(training[channel].std(), abs=1e-12)


def test_normalisation_uses_the_supplied_statistics(cfg: Config) -> None:
    frame = builder.build_feature_frame(make_bars(), cfg)
    stats = builder.fit_stats(frame.iloc[:200], cfg)

    raw = builder.build_windows(frame, cfg, "AAPL")
    scaled = builder.build_windows(frame, cfg, "AAPL", stats=stats)

    mean = np.asarray(stats.mean, dtype="float32")
    deviation = np.asarray(stats.std, dtype="float32")
    np.testing.assert_allclose(scaled.X, (raw.X - mean) / deviation, rtol=1e-6)


def test_stats_for_other_channels_are_refused(cfg: Config) -> None:
    frame = builder.build_feature_frame(make_bars(), cfg)
    wrong = ChannelStats(
        channels=("close_logret", "rsi14"),
        mean=(0.0, 50.0),
        std=(1.0, 10.0),
        fitted_start=frame.index[0],
        fitted_end=frame.index[10],
        n_rows=11,
    )

    with pytest.raises(ValueError, match="stats describe channels"):
        builder.build_windows(frame, cfg, "AAPL", stats=wrong)


def test_fit_stats_refuses_a_constant_channel(cfg: Config) -> None:
    frame = builder.build_feature_frame(make_bars(), cfg)
    flat = frame.copy()
    flat["rsi14"] = 50.0

    with pytest.raises(ValueError, match="rsi14 is constant"):
        builder.fit_stats(flat, cfg)


def test_fit_stats_refuses_an_empty_frame(cfg: Config) -> None:
    frame = builder.build_feature_frame(make_bars(), cfg)

    with pytest.raises(ValueError, match="empty frame"):
        builder.fit_stats(frame.iloc[0:0], cfg)
