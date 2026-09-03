"""L6: attribution to readable prose for the decision log and the dashboard.

Three parts, always in this order: what was forecast, which channels produced it, and
what was done about it. English and Hebrew, chosen by argument.

**Every number comes from the objects, never from a recomputation.** The forecast
magnitude is ``Attribution.forecast_total``, the shares are ``channel.shares``, the
cancellation is ``channel.cancellation``, and the band levels are the ``Thresholds`` the
decision actually used. A sentence that recomputed a percentage would be a second
implementation of the arithmetic, and the first time it disagreed with the dashboard by a
rounding step the explanation would be the thing under suspicion. So the renderer formats;
it does not calculate.

**It never claims profitability.** Not "a good entry", not "expected profit", not "a
strong opportunity". The prose says what was forecast, what drove it and what was done.
``tests/explain/test_narrate.py`` asserts this against a word list in both languages,
because it is the one property of this module a reader cannot check by looking.

**Cancellation is spoken, not hidden.** When little of the gross channel view survives
into the forecast, the shares alone would present a small difference of large opposing
numbers as though it were the whole story. Below :data:`OFFSETTING_BELOW` the narrative
says so in a sentence of its own (GB-30's ruling, implemented here).

**Bidirectional text — the decision, because Hebrew is RTL and the data is not.**
A percentage rendering on the wrong side of its label has bitten this project before, in
SVG, and Streamlit will have its own version of the problem. Three rules, all testable:

1. **Direction is metadata, not a character.** :func:`narrate` returns a
   :class:`Narrative` carrying ``direction``, so the renderer sets ``dir="rtl"`` on the
   container. Inferring direction from the first strong character is what fails when a
   sentence legitimately begins with a ticker.
2. **Every left-to-right atom is isolated** with U+2066 LEFT-TO-RIGHT ISOLATE and U+2069
   POP DIRECTIONAL ISOLATE. Isolates, not the older LRE/PDF embeddings and not bare LRM
   marks: an isolate's contents cannot influence the resolved level of the text around it
   *at all*, which is precisely the failure being prevented. An embedding can still bleed,
   and an LRM only nudges the boundary it is placed at, so both need the author to reason
   about every adjacency. This does not.
3. **An atom is the whole unit, including its sign and its unit.** ``-1.8%`` is one atom,
   so the minus and the percent can never separate from the digits; ``rsi14 +57.1%`` is
   one atom, so a channel cannot be parted from its share. The list *of* atoms then reads
   right-to-left while each atom reads left-to-right, which is correct Hebrew typography
   rather than a compromise.

English needs none of this and gets none of it, so an English narrative is plain text.

Implemented in GB-32.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from glassbox.config.loader import Config
from glassbox.contracts.schemas import Attribution, Forecast, Signal
from glassbox.engine.risk import Order
from glassbox.engine.signal import ENTER_LONG, EXIT, HOLD, Thresholds
from glassbox.explain.channel import cancellation, shares

EN = "en"
HE = "he"
LANGUAGES = (EN, HE)

LTR = "ltr"
RTL = "rtl"
DIRECTIONS = {EN: LTR, HE: RTL}

# Unicode 6.3 directional isolates. See rule 2 in the module docstring. Spelled by code
# point rather than pasted in as themselves: they are invisible characters, and an
# invisible character in a source file is exactly what a reviewer cannot review.
LRI = chr(0x2066)  # LEFT-TO-RIGHT ISOLATE
PDI = chr(0x2069)  # POP DIRECTIONAL ISOLATE

# Below this fraction of the gross channel view surviving into the forecast, the
# narrative says the channels largely offset.
#
# The level is chosen from the algebra rather than by taste. With two opposing channels
# of size `a` and `b`, `a > b`, the ratio is `(a - b) / (a + b)`, so 0.5 is exactly the
# point where the opposing side reaches a third of the leading one. Above that the
# headline is mostly one channel's view; below it the headline is mostly a difference,
# and a reader who is not told that will over-read a small number.
OFFSETTING_BELOW = 0.5


@dataclass(frozen=True)
class Narrative:
    """Rendered prose plus the direction a renderer must set on its container."""

    text: str
    language: str
    direction: str

    def __str__(self) -> str:
        return self.text


def narrate(
    forecast: Forecast,
    attribution: Attribution,
    signal: Signal,
    thresholds: Thresholds,
    cfg: Config,
    *,
    order: Order | None = None,
    language: str = EN,
) -> Narrative:
    """Render one decision as prose.

    Args:
        forecast: The predicted path, for the symbol, the date and the horizon length.
        attribution: GB-30's exact decomposition. Supplies every percentage.
        signal: The decision layer's verdict, including the strength the band was
            compared against.
        thresholds: The band this fold calibrated. Named in a hold or an exit, because a
            decision explained without its threshold is not explained.
        cfg: Resolved configuration; supplies ``signal.min_up_points`` only.
        order: The sized order, when one was produced. ``None`` in a hold, an exit, or a
            co-pilot decision awaiting approval.
        language: :data:`EN` or :data:`HE`.

    Returns:
        A :class:`Narrative`.

    Raises:
        ValueError: The language is not supported, or the attribution and the signal
            describe different symbols — which would produce prose that reads correctly
            and explains the wrong instrument.
    """
    if language not in LANGUAGES:
        raise ValueError(
            f"language must be one of {list(LANGUAGES)}, got {language!r}; a narrative "
            "in a language nobody asked for is worse than none"
        )
    if forecast.symbol != signal.symbol:
        raise ValueError(
            f"the forecast is for {forecast.symbol!r} and the signal for "
            f"{signal.symbol!r}; refusing to narrate two different instruments as one"
        )

    parts = (
        _forecast_sentence(forecast, attribution, language),
        _channel_sentences(attribution, language),
        _action_sentences(forecast, signal, thresholds, cfg, order, language),
    )
    return Narrative(
        text=" ".join(part for part in parts if part),
        language=language,
        direction=DIRECTIONS[language],
    )


# ── part 1: what was forecast ────────────────────────────────────────────────


def _forecast_sentence(
    forecast: Forecast, attribution: Attribution, language: str
) -> str:
    symbol = _atom(forecast.symbol, language)
    date = _atom(f"{forecast.as_of:%Y-%m-%d}", language)
    horizon = len(forecast.path)

    if _expresses_no_view(attribution):
        # Persistence, and anything else with no parameters. `sign(0)` is not a forecast
        # of no movement: the model was never asked a directional question and did not
        # answer one, and prose that renders it as "predicts no change" would put a view
        # in the baseline's mouth that the whole study is measured against.
        if language == HE:
            return (
                f"{symbol} בתאריך {date}: המודל לא הביע כיוון — כל הערוצים תרמו אפס "
                "בדיוק, וזו היעדר עמדה ולא תחזית שהמחיר לא יזוז."
            )
        return (
            f"{symbol} on {date}: the model made no directional call — every channel "
            "contributed exactly zero, which is the absence of a view rather than a "
            "forecast that the price will not move."
        )

    total = attribution.forecast_total
    if total == 0.0:
        # A real model whose channels cancelled exactly. Here "no net move" *is* the
        # forecast, which is a different statement from the one above.
        if language == HE:
            return (
                f"{symbol} בתאריך {date}: המודל צופה תנועה נטו אפסית על פני "
                f"{_sessions(horizon, language)} — הערוצים קיזזו זה את זה במדויק."
            )
        return (
            f"{symbol} on {date}: the model predicts no net move over the next "
            f"{_sessions(horizon, language)} — the channels cancelled exactly."
        )

    magnitude = _atom(f"{abs(_percent(total)):.2f}%", language)
    if language == HE:
        word = "עלייה" if total > 0 else "ירידה"
        return (
            f"{symbol} בתאריך {date}: המודל צופה {word} של {magnitude} על פני "
            f"{_sessions(horizon, language)}."
        )
    word = "rise" if total > 0 else "fall"
    return (
        f"{symbol} on {date}: the model predicts a {magnitude} {word} over the next "
        f"{_sessions(horizon, language)}."
    )


# ── part 2: which channels produced it ───────────────────────────────────────


def _channel_sentences(attribution: Attribution, language: str) -> str:
    ranked = sorted(
        shares(attribution).items(), key=lambda item: (-abs(item[1]), item[0])
    )
    listed = ", ".join(
        _atom(f"{channel} {_share(value)}", language) for channel, value in ranked
    )
    survived = cancellation(attribution)

    if language == HE:
        sentence = f"הערוצים, לפי חלקם בתרומה הגולמית: {listed}."
    else:
        sentence = f"Channels, by share of the gross view: {listed}."

    if _expresses_no_view(attribution) or survived >= OFFSETTING_BELOW:
        return sentence

    fraction = _atom(f"{survived * 100:.1f}%", language)
    if language == HE:
        return (
            f"{sentence} הערוצים מקזזים זה את זה במידה רבה: רק {fraction} מהתרומה "
            "הגולמית שרדו לתוך התחזית, ולכן המספר שלמעלה הוא הפרש קטן בין תרומות "
            "גדולות ומנוגדות."
        )
    return (
        f"{sentence} These largely offset: only {fraction} of the gross view survived "
        "into the forecast, so the figure above is a small difference between larger "
        "opposing contributions."
    )


# ── part 3: what was done about it ───────────────────────────────────────────


def _action_sentences(
    forecast: Forecast,
    signal: Signal,
    thresholds: Thresholds,
    cfg: Config,
    order: Order | None,
    language: str,
) -> str:
    if not thresholds.fires:
        if language == HE:
            return (
                "פעולה: אין. החלון הזה נותר מחוץ לשוק — הכיול לא מצא רצועת סף עם שארפ "
                "חיובי, ולכן לא הייתה כניסה אפשרית."
            )
        return (
            "Action: none. This fold stood aside — calibration found no threshold band "
            "with a positive validation Sharpe, so no entry was available."
        )

    if signal.action == ENTER_LONG:
        return _entry_sentence(signal, order, language)
    if signal.action == EXIT:
        return _exit_sentence(signal, thresholds, language)
    if signal.action == HOLD:
        return _hold_sentence(forecast, signal, thresholds, cfg, language)
    raise ValueError(
        f"{signal.symbol}: cannot narrate the action {signal.action!r}; the vocabulary "
        "is defined in engine.signal and this renderer must not invent a fourth word"
    )


def _entry_sentence(signal: Signal, order: Order | None, language: str) -> str:
    symbol = _atom(signal.symbol, language)
    if order is None:
        if language == HE:
            return (
                f"פעולה: כניסה ללונג ב-{symbol}. הפוזיציה לא תומחרה כאן, ולכן אין סטופ "
                "או יעד לדווח."
            )
        return (
            f"Action: enter long {symbol}. The position was not sized here, so there is "
            "no stop or target to report."
        )

    quantity = _atom(f"{order.shares:.4f}", language)
    price = _atom(f"{order.price:,.2f}", language)
    notional = _atom(f"{order.notional:,.2f}", language)
    stop = _atom(f"{order.stop_loss:,.2f}", language)
    target = _atom(f"{order.take_profit:,.2f}", language)

    if language == HE:
        return (
            f"פעולה: כניסה ללונג, {quantity} מניות של {symbol} במחיר {price}, חשיפה של "
            f"{notional}. סטופ ב-{stop}, יעד ב-{target}."
        )
    return (
        f"Action: enter long {quantity} shares of {symbol} at {price}, a notional of "
        f"{notional}. Stop at {stop}, target at {target}."
    )


def _exit_sentence(signal: Signal, thresholds: Thresholds, language: str) -> str:
    strength = _atom(f"{_percent(signal.trend_strength):+.2f}%", language)
    level = _atom(f"{_percent(-thresholds.lower):+.2f}%", language)

    if language == HE:
        return (
            f"פעולה: יציאה. התחזית ירדה ל-{strength}, בגובה רמת היציאה ({level}) או "
            "מתחתיה, שהיא השיקוף של סף הכניסה."
        )
    return (
        f"Action: exit. The forecast fell to {strength}, at or below the exit level of "
        f"{level}, which is the mirror of the entry threshold."
    )


def _hold_sentence(
    forecast: Forecast,
    signal: Signal,
    thresholds: Thresholds,
    cfg: Config,
    language: str,
) -> str:
    strength = _atom(f"{_percent(signal.trend_strength):+.2f}%", language)

    if signal.passed_threshold:
        # The band admitted it and the path did not confirm it. Saying only "the
        # threshold was not met" here would be false.
        points = _atom(str(signal.up_points), language)
        steps = _atom(str(len(forecast.path)), language)
        required = _atom(str(cfg.signal.min_up_points), language)
        if language == HE:
            return (
                f"פעולה: המתנה. הרצועה קיבלה את התחזית ({strength}), אך רק {points} "
                f"מתוך {steps} הצעדים מצביעים מעלה, מול {required} הנדרשים."
            )
        return (
            f"Action: hold. The band admitted the forecast ({strength}), but only "
            f"{points} of its {steps} steps point up, against the {required} required."
        )

    upper = thresholds.upper
    if upper is not None and signal.trend_strength > upper:
        apart = _apart(_percent(signal.trend_strength), _percent(upper))
        if apart is None:
            over = _atom(f"{BAND_FLOOR:.{BAND_PLACES[-1]}f}", language)
            bound = _atom(f"{_percent(upper):+.2f}%", language)
            if language == HE:
                return (
                    f"פעולה: המתנה. התחזית עברה את הגבול העליון של הרצועה ({bound}) "
                    f"בפחות מ-{over} נקודות אחוז, ומעליו תחזית נחשבת בלתי סבירה ולא "
                    "אטרקטיבית."
                )
            return (
                f"Action: hold. The forecast passed the band's upper bound of {bound} "
                f"by less than {over} of a percentage point, and beyond that bound a "
                "forecast is treated as implausible rather than attractive."
            )
        reached, bound = _atom(apart[0], language), _atom(apart[1], language)
        if language == HE:
            return (
                f"פעולה: המתנה. התחזית הגיעה ל-{reached}, מעל הגבול העליון של הרצועה "
                f"({bound}), שמעליו תחזית נחשבת בלתי סבירה ולא אטרקטיבית."
            )
        return (
            f"Action: hold. The forecast reached {reached}, above the band's upper "
            f"bound of {bound}, beyond which a forecast is treated as implausible "
            "rather than attractive."
        )

    apart = _apart(_percent(thresholds.lower), _percent(signal.trend_strength))
    if apart is None:
        entry = _atom(f"{_percent(thresholds.lower):+.2f}%", language)
        short = _atom(f"{BAND_FLOOR:.{BAND_PLACES[-1]}f}", language)
        if language == HE:
            return (
                f"פעולה: המתנה. סף הכניסה המכויל לחלון זה מתחיל ב-{entry} והתחזית "
                f"נפלה ממנו בפחות מ-{short} נקודות אחוז, כך שהסף לא נחצה."
            )
        return (
            f"Action: hold. The calibrated entry band for this fold starts at {entry} "
            f"and the forecast fell short of it by less than {short} of a percentage "
            "point, so the threshold was not met."
        )
    entry, reached = _atom(apart[0], language), _atom(apart[1], language)
    if language == HE:
        return (
            f"פעולה: המתנה. סף הכניסה המכויל לחלון זה מתחיל ב-{entry} והתחזית הגיעה "
            f"ל-{reached}, כך שהסף לא נחצה."
        )
    return (
        f"Action: hold. The calibrated entry band for this fold starts at {entry} and "
        f"the forecast reached {reached}, so the threshold was not met."
    )


# ── formatting ───────────────────────────────────────────────────────────────


def _atom(text: str, language: str) -> str:
    """Wrap a left-to-right run so it cannot reorder against the text around it.

    A no-op in English, where the paragraph direction already matches.
    """
    return f"{LRI}{text}{PDI}" if DIRECTIONS[language] == RTL else text


def _percent(log_return: float) -> float:
    """A log return as a simple percentage. The one conversion, used everywhere."""
    return math.expm1(log_return) * 100.0


#: How far a band sentence widens its two numbers before it stops trying to show them
#: apart, and the bound it states instead. Two decimals is the house format; below
#: ``BAND_FLOOR`` percentage points the difference is smaller than anything a reader could
#: act on, so the sentence names that bound rather than printing a distinction nobody can
#: see. The floor is the last entry read back as a width, so the two cannot drift.
BAND_PLACES = (2, 3, 4, 5, 6)
BAND_FLOOR = 10.0 ** -BAND_PLACES[-1]


def _apart(value: float, bound: float) -> tuple[str, str] | None:
    """``value`` and ``bound`` as percentages, widened until they read as different.

    **A sentence asserting a strict inequality may not print its two sides as the same
    number.** On 3 September 2026 COST forecast 0.6724% against a band starting at
    0.6738%; both round to ``+0.67%`` and the narration read *"starts at +0.67% and the
    forecast reached +0.67%, so the threshold was not met"* - a decision that was right,
    rendered as a contradiction. `signal.decide` compares the unrounded values and was
    never wrong; only what the reader was shown was. The band's ``lower`` is 0.0067, so a
    forecast landing within a rounding step of it is the case a calibrated threshold
    produces *most* often, not a corner.

    This is :func:`_share`'s rule one function further on - a value must not appear to be
    something it is not because the format rounded the difference away. `_share` fixes an
    unsigned zero; this widens a pair until it separates.

    Returns ``None`` when six decimals still cannot part them, which the caller states as
    a bound instead of papering over. That case is reachable rather than defensive:
    :func:`_percent` is ``expm1``, so two distinct log returns can land on one percentage
    double, and no precision would ever separate those.
    """
    for places in BAND_PLACES:
        shown, against = f"{value:+.{places}f}%", f"{bound:+.{places}f}%"
        if shown != against:
            return shown, against
    return None


def _share(fraction: float) -> str:
    """A share as a signed percentage, with an unsigned zero.

    ``+0.0%`` reads as a rounded-away positive contribution. A channel that contributed
    exactly nothing should not appear to have leaned either way.
    """
    return "0.0%" if fraction == 0.0 else f"{fraction * 100:+.1f}%"


def _sessions(horizon: int, language: str) -> str:
    if language == HE:
        return (
            "יום מסחר אחד"
            if horizon == 1
            else f"{_atom(str(horizon), language)} ימי מסחר"
        )
    return (
        "trading day"
        if horizon == 1
        else f"{_atom(str(horizon), language)} trading days"
    )


def _expresses_no_view(attribution: Attribution) -> bool:
    """True when no channel contributed anything at all.

    Read off the shares rather than the forecast: a share is zero only when its own
    contribution is zero, so all-zero shares means a zero gross, which is the case
    :func:`~glassbox.explain.channel.shares` documents as "nothing was predicted". A model
    whose channels *cancel* has a non-zero gross and is a different sentence.
    """
    return not any(shares(attribution).values())


__all__ = [
    "DIRECTIONS",
    "EN",
    "HE",
    "LANGUAGES",
    "LRI",
    "LTR",
    "OFFSETTING_BELOW",
    "PDI",
    "RTL",
    "Narrative",
    "narrate",
]
