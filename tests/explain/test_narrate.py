"""GB-32 acceptance: the narrative is readable, bilingual, and says only what is true.

Three decisions — enter, hold, exit — rendered in both languages, with the percentages
asserted against the ``Attribution`` they came from rather than against literals a future
edit could quietly drift away from.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from glassbox.config.loader import Config, load_config
from glassbox.contracts.schemas import Attribution, Forecast, Signal
from glassbox.engine.risk import Order
from glassbox.engine.signal import ENTER_LONG, EXIT, HOLD, Thresholds
from glassbox.explain.channel import cancellation, shares
from glassbox.explain.narrate import (
    EN,
    HE,
    LANGUAGES,
    LRI,
    LTR,
    OFFSETTING_BELOW,
    PDI,
    RTL,
    narrate,
)

AS_OF = pd.Timestamp("2026-08-18", tz="UTC")
HORIZON = 4

# Words that would turn an explanation into a recommendation. Neither language may use
# any of them, in any decision, ever.
FORBIDDEN = (
    "profit",
    "profitable",
    "guaranteed",
    "risk-free",
    "riskless",
    "sure thing",
    "opportunity",
    "should buy",
    "רווח",
    "מובטח",
    "ללא סיכון",
    "כדאי",
    "הזדמנות",
)


@pytest.fixture
def cfg() -> Config:
    return load_config()


def a_forecast(total: float, symbol: str = "AAPL", up: int = HORIZON) -> Forecast:
    """A path summing to ``total`` with exactly ``up`` positive steps."""
    steps = np.full(HORIZON, total / HORIZON, dtype="float32")
    for index in range(up, HORIZON):
        steps[index] = -abs(steps[index])
    steps[0] = np.float32(total - float(steps[1:].sum()))
    return Forecast(path=steps, symbol=symbol, as_of=AS_OF)


def an_attribution(**per_channel: float) -> Attribution:
    return Attribution(
        per_channel=dict(per_channel),
        per_lag=None,
        per_frequency=None,
        gain_phase=None,
        forecast_total=sum(per_channel.values()),
    )


def a_signal(action: str, strength: float, **overrides) -> Signal:
    fields = {
        "symbol": "AAPL",
        "action": action,
        "trend_strength": strength,
        "up_points": HORIZON,
        "passed_threshold": action == ENTER_LONG,
    }
    fields.update(overrides)
    return Signal(**fields)


AN_ORDER = Order(
    symbol="AAPL",
    shares=0.0819,
    price=305.59,
    notional=25.03,
    stop_loss=296.42,
    take_profit=323.93,
)


def bare(text: str) -> str:
    """The text with every directional isolate removed, for plain matching."""
    return text.replace(LRI, "").replace(PDI, "")


def atoms(text: str) -> list[str]:
    """The contents of every isolate, in order."""
    return re.findall(f"{LRI}(.*?){PDI}", text)


# ── the three decisions, in both languages ───────────────────────────────────


@pytest.mark.parametrize("language", LANGUAGES)
def test_enter_names_the_size_the_price_and_the_stop(
    cfg: Config, language: str
) -> None:
    attribution = an_attribution(close_logret=0.012, rsi14=0.006)
    narrative = narrate(
        a_forecast(0.018),
        attribution,
        a_signal(ENTER_LONG, 0.018),
        Thresholds(lower=0.004, upper=0.05),
        cfg,
        order=AN_ORDER,
        language=language,
    )
    text = bare(narrative.text)

    assert "AAPL" in text
    assert "2026-08-18" in text
    assert "0.0819" in text  # size
    assert "305.59" in text  # reference price
    assert "296.42" in text  # stop
    assert "323.93" in text  # target


@pytest.mark.parametrize("language", LANGUAGES)
def test_hold_explains_the_band_it_did_not_reach(cfg: Config, language: str) -> None:
    """A hold with no threshold named is not an explanation."""
    narrative = narrate(
        a_forecast(0.0018),
        an_attribution(close_logret=0.001, rsi14=0.0008),
        a_signal(HOLD, 0.0018, passed_threshold=False),
        Thresholds(lower=0.0042),
        cfg,
        language=language,
    )
    text = bare(narrative.text)

    assert "+0.42%" in text  # the calibrated entry level
    assert "+0.18%" in text  # what the forecast reached


@pytest.mark.parametrize("language", LANGUAGES)
def test_hold_above_the_upper_bound_says_implausible_not_weak(
    cfg: Config, language: str
) -> None:
    """The two holds have opposite causes and must not share a sentence."""
    narrative = narrate(
        a_forecast(0.06),
        an_attribution(close_logret=0.05, rsi14=0.01),
        a_signal(HOLD, 0.06, passed_threshold=False),
        Thresholds(lower=0.004, upper=0.05),
        cfg,
        language=language,
    )
    text = bare(narrative.text)

    assert "+5.13%" in text  # the upper bound
    expected = "implausible" if language == EN else "בלתי סבירה"
    assert expected in text


@pytest.mark.parametrize("language", LANGUAGES)
def test_hold_on_unconfirmed_steps_does_not_blame_the_threshold(
    cfg: Config, language: str
) -> None:
    """The band admitted it; the path disagreed with itself. Saying "the threshold was
    not met" here would be false, and `Signal.passed_threshold` exists to tell them
    apart."""
    narrative = narrate(
        a_forecast(0.018, up=2),
        an_attribution(close_logret=0.012, rsi14=0.006),
        a_signal(HOLD, 0.018, up_points=2, passed_threshold=True),
        Thresholds(lower=0.004),
        cfg,
        language=language,
    )
    text = bare(narrative.text)

    assert str(cfg.signal.min_up_points) in text
    assert "2" in text
    expected = "threshold was not met" if language == EN else "הסף לא נחצה"
    assert expected not in text


@pytest.mark.parametrize("language", LANGUAGES)
def test_exit_names_the_mirrored_level(cfg: Config, language: str) -> None:
    narrative = narrate(
        a_forecast(-0.009, up=0),
        an_attribution(close_logret=-0.006, rsi14=-0.003),
        a_signal(EXIT, -0.009, up_points=0, passed_threshold=False),
        Thresholds(lower=0.0042),
        cfg,
        language=language,
    )
    text = bare(narrative.text)

    assert "-0.42%" in text  # the exit level, the mirror of the entry threshold


@pytest.mark.parametrize("language", LANGUAGES)
def test_a_fold_that_stood_aside_says_so(cfg: Config, language: str) -> None:
    narrative = narrate(
        a_forecast(0.018),
        an_attribution(close_logret=0.012, rsi14=0.006),
        a_signal(HOLD, 0.018, passed_threshold=False),
        Thresholds.never(),
        cfg,
        language=language,
    )
    text = bare(narrative.text)

    assert "inf" not in text.lower()
    expected = "stood aside" if language == EN else "מחוץ לשוק"
    assert expected in text


# ── the percentages come from the Attribution, never recomputed ──────────────


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_share_matches_the_attribution_exactly(
    cfg: Config, language: str
) -> None:
    """What the prose prints and what the dashboard prints are one computation."""
    attribution = an_attribution(close_logret=0.012, rsi14=0.006, vol_z=-0.004)
    narrative = narrate(
        a_forecast(attribution.forecast_total),
        attribution,
        a_signal(ENTER_LONG, attribution.forecast_total),
        Thresholds(lower=0.004),
        cfg,
        order=AN_ORDER,
        language=language,
    )
    text = bare(narrative.text)

    for channel, fraction in shares(attribution).items():
        assert f"{channel} {fraction * 100:+.1f}%" in text


@pytest.mark.parametrize("language", LANGUAGES)
def test_the_headline_is_the_attributions_own_total(cfg: Config, language: str) -> None:
    attribution = an_attribution(close_logret=0.012, rsi14=0.006)
    narrative = narrate(
        a_forecast(attribution.forecast_total),
        attribution,
        a_signal(ENTER_LONG, attribution.forecast_total),
        Thresholds(lower=0.004),
        cfg,
        order=AN_ORDER,
        language=language,
    )

    expected = math.expm1(attribution.forecast_total) * 100
    assert f"{expected:.2f}%" in bare(narrative.text)


def test_channels_are_ranked_by_share_largest_first(cfg: Config) -> None:
    attribution = an_attribution(close_logret=0.004, rsi14=0.012, vol_z=-0.008)
    narrative = narrate(
        a_forecast(attribution.forecast_total),
        attribution,
        a_signal(ENTER_LONG, attribution.forecast_total),
        Thresholds(lower=0.004),
        cfg,
        order=AN_ORDER,
    )
    text = narrative.text

    assert text.index("rsi14") < text.index("vol_z") < text.index("close_logret")


# ── cancellation is spoken, not buried in the shares ─────────────────────────


@pytest.mark.parametrize("language", LANGUAGES)
def test_largely_offsetting_channels_produce_the_offsetting_sentence(
    cfg: Config, language: str
) -> None:
    """+0.08 and -0.06 behind +0.02: the shares are +57% and -43%, and saying only that
    would present a small difference of large numbers as the whole story."""
    attribution = an_attribution(close_logret=0.08, rsi14=-0.06)
    assert cancellation(attribution) < OFFSETTING_BELOW

    narrative = narrate(
        a_forecast(0.02),
        attribution,
        a_signal(ENTER_LONG, 0.02),
        Thresholds(lower=0.004),
        cfg,
        order=AN_ORDER,
        language=language,
    )
    text = bare(narrative.text)

    expected = "largely offset" if language == EN else "מקזזים"
    assert expected in text
    assert f"{cancellation(attribution) * 100:.1f}%" in text


@pytest.mark.parametrize("language", LANGUAGES)
def test_aligned_channels_do_not_get_the_offsetting_sentence(
    cfg: Config, language: str
) -> None:
    attribution = an_attribution(close_logret=0.012, rsi14=0.006)
    assert cancellation(attribution) == pytest.approx(1.0)

    narrative = narrate(
        a_forecast(0.018),
        attribution,
        a_signal(ENTER_LONG, 0.018),
        Thresholds(lower=0.004),
        cfg,
        order=AN_ORDER,
        language=language,
    )
    text = bare(narrative.text)

    unexpected = "largely offset" if language == EN else "מקזזים"
    assert unexpected not in text


# ── the baseline made no call; it did not predict flatness ───────────────────


@pytest.mark.parametrize("language", LANGUAGES)
def test_persistence_renders_as_no_directional_call(cfg: Config, language: str) -> None:
    """`sign(0)` is not a forecast of no movement, and the prose must not say it is."""
    attribution = an_attribution(close_logret=0.0, rsi14=0.0, vol_z=0.0)
    narrative = narrate(
        a_forecast(0.0, up=0),
        attribution,
        a_signal(HOLD, 0.0, up_points=0, passed_threshold=False),
        Thresholds(lower=0.004),
        cfg,
        language=language,
    )
    text = bare(narrative.text)

    expected = "no directional call" if language == EN else "לא הביע כיוון"
    assert expected in text
    for channel in attribution.per_channel:
        assert f"{channel} 0.0%" in text  # every channel still named, GB-11's ruling


@pytest.mark.parametrize("language", LANGUAGES)
def test_channels_that_cancel_exactly_are_not_the_baselines_sentence(
    cfg: Config, language: str
) -> None:
    """A zero forecast from opposing channels is a view. A zero from nothing is not."""
    narrative = narrate(
        a_forecast(0.0, up=2),
        an_attribution(close_logret=0.05, rsi14=-0.05),
        a_signal(HOLD, 0.0, up_points=2, passed_threshold=False),
        Thresholds(lower=0.004),
        cfg,
        language=language,
    )
    text = bare(narrative.text)

    unexpected = "no directional call" if language == EN else "לא הביע כיוון"
    assert unexpected not in text
    expected = "cancelled exactly" if language == EN else "קיזזו זה את זה"
    assert expected in text


# ── no claim about profitability, ever ───────────────────────────────────────


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("action", (ENTER_LONG, HOLD, EXIT))
def test_no_decision_in_any_language_claims_profitability(
    cfg: Config, language: str, action: str
) -> None:
    narrative = narrate(
        a_forecast(0.018),
        an_attribution(close_logret=0.012, rsi14=0.006),
        a_signal(action, 0.018),
        Thresholds(lower=0.004),
        cfg,
        order=AN_ORDER if action == ENTER_LONG else None,
        language=language,
    )
    text = bare(narrative.text).lower()

    assert not [word for word in FORBIDDEN if word in text]


# ── bidirectional text ───────────────────────────────────────────────────────


def test_direction_is_metadata_not_a_leading_character(cfg: Config) -> None:
    """A renderer sets `dir=` from this. Sniffing the first strong character fails when
    a sentence legitimately opens with a ticker, which every one of ours does."""
    arguments = (
        a_forecast(0.018),
        an_attribution(close_logret=0.012, rsi14=0.006),
        a_signal(ENTER_LONG, 0.018),
        Thresholds(lower=0.004),
        cfg,
    )

    assert narrate(*arguments, order=AN_ORDER, language=EN).direction == LTR
    assert narrate(*arguments, order=AN_ORDER, language=HE).direction == RTL


@pytest.mark.parametrize("action", (ENTER_LONG, HOLD, EXIT))
def test_hebrew_isolates_every_left_to_right_atom(cfg: Config, action: str) -> None:
    """Nothing Latin or numeric may sit loose in the Hebrew text.

    This is the assertion that stops a percentage rendering on the wrong side of its
    label: an isolated run cannot reorder against its surroundings, and a run that was
    never isolated can.
    """
    narrative = narrate(
        a_forecast(0.018),
        an_attribution(close_logret=0.012, rsi14=0.006),
        a_signal(action, 0.018),
        Thresholds(lower=0.004, upper=0.05),
        cfg,
        order=AN_ORDER if action == ENTER_LONG else None,
        language=HE,
    )
    outside = re.sub(f"{LRI}.*?{PDI}", "", narrative.text)

    assert not re.search(r"[A-Za-z0-9]", outside), outside


def test_hebrew_isolates_are_balanced_and_never_nested(cfg: Config) -> None:
    """An unbalanced isolate leaks into the rest of the paragraph; a nested one is a
    sign the atoms were composed rather than emitted whole."""
    narrative = narrate(
        a_forecast(0.018),
        an_attribution(close_logret=0.012, rsi14=0.006),
        a_signal(ENTER_LONG, 0.018),
        Thresholds(lower=0.004),
        cfg,
        order=AN_ORDER,
        language=HE,
    )

    depth = 0
    for character in narrative.text:
        depth += (character == LRI) - (character == PDI)
        assert 0 <= depth <= 1
    assert depth == 0


def test_a_signed_percentage_is_one_atom(cfg: Config) -> None:
    """The sign and the percent sign travel with their digits or they can be reordered
    away from them, which is the SVG failure this rule exists to prevent."""
    narrative = narrate(
        a_forecast(-0.009, up=0),
        an_attribution(close_logret=-0.006, rsi14=-0.003),
        a_signal(EXIT, -0.009, up_points=0, passed_threshold=False),
        Thresholds(lower=0.0042),
        cfg,
        language=HE,
    )

    signed = [atom for atom in atoms(narrative.text) if "%" in atom]
    assert signed
    for atom in signed:
        assert re.fullmatch(r"[a-z_0-9]* ?[-+]?\d+\.\d+%", atom), atom


def test_a_channel_never_parts_from_its_share(cfg: Config) -> None:
    attribution = an_attribution(close_logret=0.012, rsi14=0.006)
    narrative = narrate(
        a_forecast(0.018),
        attribution,
        a_signal(ENTER_LONG, 0.018),
        Thresholds(lower=0.004),
        cfg,
        order=AN_ORDER,
        language=HE,
    )

    for channel, fraction in shares(attribution).items():
        assert f"{channel} {fraction * 100:+.1f}%" in atoms(narrative.text)


def test_english_carries_no_isolates(cfg: Config) -> None:
    """They would be invisible but real, and would end up in the decision log."""
    narrative = narrate(
        a_forecast(0.018),
        an_attribution(close_logret=0.012, rsi14=0.006),
        a_signal(ENTER_LONG, 0.018),
        Thresholds(lower=0.004),
        cfg,
        order=AN_ORDER,
        language=EN,
    )

    assert LRI not in narrative.text
    assert PDI not in narrative.text


# ── refusals ─────────────────────────────────────────────────────────────────


def test_an_unsupported_language_is_refused(cfg: Config) -> None:
    with pytest.raises(ValueError, match="language must be one of"):
        narrate(
            a_forecast(0.018),
            an_attribution(close_logret=0.018),
            a_signal(ENTER_LONG, 0.018),
            Thresholds(lower=0.004),
            cfg,
            language="fr",
        )


def test_a_mismatched_symbol_is_refused(cfg: Config) -> None:
    """Prose that reads correctly and explains the wrong instrument is the worst
    possible output of this module."""
    with pytest.raises(ValueError, match="refusing to narrate"):
        narrate(
            a_forecast(0.018, symbol="MSFT"),
            an_attribution(close_logret=0.018),
            a_signal(ENTER_LONG, 0.018),
            Thresholds(lower=0.004),
            cfg,
        )


def test_an_unknown_action_is_refused(cfg: Config) -> None:
    """The vocabulary lives in engine.signal; this module must not invent a fourth word."""
    invented = replace(a_signal(ENTER_LONG, 0.018), action="sell_short")

    with pytest.raises(ValueError, match="cannot narrate the action"):
        narrate(
            a_forecast(0.018),
            an_attribution(close_logret=0.018),
            invented,
            Thresholds(lower=0.004),
            cfg,
        )
