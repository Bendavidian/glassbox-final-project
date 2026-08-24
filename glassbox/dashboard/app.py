"""L7: the Streamlit dashboard — positions and PnL, forecast paths, the decision log
with channel contribution bars, and the spectral panel.

**Two colour families, and they mean different things.** Orange is **interface chrome
only**: labels, rules, borders, registration marks, active states, and the calibrated
threshold, which is a rule line rather than a measurement. The **blue ramp is the data
encoding**, dark for slow and light for fast — the same ramp as the Hebrew proposal, the
architecture report's three figures and the vision script, so a reader who has seen any of
those reads this without relearning. Keeping chrome and data in separate families is not
decoration: it is what tells the eye which marks carry information.

**Sign is never carried by hue alone.** The one data family available is a *ramp*, and a
ramp cannot encode a sign without inventing a second data colour, which the ruling above
forbids. So a positive contribution is a filled bar to the right of the zero rule and a
negative one is a hollow bar to the left, and every signed number carries an explicit
``▲``/``▼``. The chart is readable in greyscale, which is the test of whether colour was
doing the work.

**The charts are hand-built SVG.** Not a plotting library: the design language here is
dashed hairlines, numbered rulers and registration marks, which a chart library fights
rather than helps, and an SVG builder is a **pure function returning a string**, so the
threshold line, the ramp and the bar geometry are unit-testable without a browser. It also
adds no dependency.

**A Hebrew narrative is rendered with an explicit ``dir``.** Not ``dir="auto"``: that reads
the first strong character, and every narrative GB-32 writes opens with a ticker, so
``auto`` would resolve a Hebrew sentence to left-to-right and put its percentages on the
wrong side of their labels. The direction is decided by looking for Hebrew letters.

Implemented in GB-34, GB-35 and GB-36. The spectral panel is GB-53.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from glassbox import records
from glassbox.config.loader import Config, config_hash, load_config
from glassbox.contracts.schemas import Attribution, DecisionRecord
from glassbox.engine.reconcile import Book
from glassbox.engine.signal import Thresholds
from glassbox.explain.channel import cancellation, shares

# ── the palette ──────────────────────────────────────────────────────────────

INK = "#0A0A0B"  # near-black ground
PANEL = "#101012"
HAIRLINE = "#2A2A2E"
PAPER = "#F2F0EC"  # display type
MUTED = "#7A7A80"

# Chrome. Labels, rules, borders, registration marks, active states, threshold lines.
ORANGE = "#E8542A"
ORANGE_DIM = "#8A3219"

# Data. Dark for slow, light for fast — the spectral ramp, applied to channels ordered by
# how much history each one looks back over.
RAMP = ("#0A2239", "#123F63", "#1B6CA8", "#2E97D4", "#6FC3EC", "#B7E3F7")

# Channels from slowest to fastest, which is the order the ramp is assigned in. The ramp
# is a *ramp*: it encodes a quantity, and the quantity here is how far back a channel
# looks. A palette assigned in config order would encode nothing.
#
# **The ordering is by NOMINAL period.** `rsi14` sits at 14 days on that reading, and its
# Wilder smoothing has a longer effective memory than its nominal period - GB-27 measured
# 325 bars for the seed to decay below float32 resolution. Most of its weight does sit in
# the recent ~30 bars, so the position is defensible, but anyone re-deriving this from
# effective memory rather than nominal period will reach a different answer and should know
# which of the two this list is.
CHANNEL_SPEED = (
    "ma_dist20",
    "rsi14",
    "vol_z",
    "mom10",
    "close_logret",
)

RUNNING = "RUNNING"
IDLE = "IDLE"
CLOSED = "OUTSIDE MARKET HOURS"
ASIDE = "STOOD ASIDE"

# Below this fraction of the gross channel view surviving into the forecast, a decision
# is flagged in the log and in its expander title. Not "weakly supported" - FRAGILE, which
# is a different claim: the explanation is a residue of contributions much larger than
# itself, and reading "which channel drove this" off it is reading rounding.
#
# The level comes from the algebra rather than from taste, like `OFFSETTING_BELOW` in
# `explain/narrate.py`, and it is a stricter bar because it answers a stricter question.
# Cancellation is `|sum(c)| / sum(|c|)`. Perturb one channel's contribution by `d` and the
# forecast moves by `d`; as a fraction OF THE FORECAST that is `d / |sum(c)|`, which is
# `1 / cancellation` times what the same perturbation would do if nothing cancelled. So
# cancellation is exactly the reciprocal of how much the decomposition amplifies an error
# in any single channel. At 0.20 the amplification is fivefold: a contribution wrong by
# one part in a hundred moves the forecast by five.
#
# `OFFSETTING_BELOW` at 0.5 says the channels largely offset, which is a fact about the
# forecast. This says the explanation of it should not be trusted, which is a fact about
# the decomposition, and it needs to be the rarer of the two flags or it stops being read.
EXPLANATION_FRAGILE_BELOW = 0.20
FRAGILE_LABEL = "FRAGILE"
# Which column of the decision log carries the cancellation, and therefore the flag.
CANCELLATION_COLUMN = 5
FRAGILE_MARK = "  FRAGILE"

# A blank cell in a technical table reads as missing data rather than as "none".
EM_DASH = "\u2014"

HISTORY_BARS = 60
RELIABILITY_FILE = "reliability.json"

# Hebrew block, for the direction decision. See the module docstring on why `dir="auto"`
# is the wrong tool here.
HEBREW = range(0x0590, 0x0600)

# What the fixed left ruler occupies, and therefore what the content is padded past.
GUTTER = "3.4rem"


# ── state ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Reliability:
    """The decider's track record, shown in the panel whatever else is on screen.

    An explanation makes a decision legible; it does not make it right. A legible wrong
    decision shown without its base rate is more persuasive than an opaque one, which is a
    worse outcome than the black box this project set out to replace. So this is chrome,
    not a tab.
    """

    direction: float
    always_long: float
    folds: int
    measured_on: str
    model: str

    @property
    def gap(self) -> float:
        return self.direction - self.always_long

    @property
    def beats_the_bar(self) -> bool:
        return self.gap > 0


def load_reliability(path: str | Path) -> Reliability | None:
    """Read the measured track record, or ``None``.

    ``None`` renders as ``NOT MEASURED`` rather than as a blank or a plausible default.
    A dashboard that quietly omitted the panel when the file was missing would be the
    failure the panel exists to prevent, arriving as an absence instead of a lie.
    """
    source = Path(path)
    if not source.is_file():
        return None
    raw = json.loads(source.read_text(encoding="utf-8"))
    return Reliability(
        direction=float(raw["direction"]),
        always_long=float(raw["always_long"]),
        folds=int(raw["folds"]),
        measured_on=str(raw["measured_on"]),
        model=str(raw.get("model", "unknown")),
    )


def status_of(
    thresholds: Thresholds, in_market_hours: bool, positions: dict[str, float]
) -> str:
    """What the bot is doing, in the order a reader needs to know it.

    ``STOOD ASIDE`` outranks ``RUNNING`` because it is the more surprising fact: a session
    whose calibrated band cannot fire is not idle between decisions, it has decided in
    advance that it will not act, and a status line reading ``RUNNING`` beside a flat book
    would leave a viewer waiting for a trade that cannot come.
    """
    if not thresholds.fires:
        return ASIDE
    if not in_market_hours:
        return CLOSED
    return RUNNING if positions else IDLE


@dataclass(frozen=True)
class PositionRow:
    """One held position, marked to the last price the dashboard could see."""

    symbol: str
    quantity: float
    entry_price: float
    price: float
    managed: bool

    @property
    def market_value(self) -> float:
        return self.quantity * self.price

    @property
    def unrealised(self) -> float:
        return (self.price - self.entry_price) * self.quantity

    @property
    def unrealised_pct(self) -> float:
        if self.entry_price <= 0:
            return math.nan
        return (self.price / self.entry_price - 1.0) * 100.0


def position_rows(
    book: Book, quantities: dict[str, float], prices: dict[str, float]
) -> list[PositionRow]:
    """Every position the broker reports, managed or quarantined.

    A quarantined position appears with no entry basis — the system did not open it and
    cannot say what it paid — so its PnL is NaN rather than a number computed from a price
    it never traded at. It is still shown, because it is still spending buying power.
    """
    rows = []
    for symbol in sorted(quantities):
        holding = book.managed.get(symbol)
        price = float(prices.get(symbol, math.nan))
        rows.append(
            PositionRow(
                symbol=symbol,
                quantity=float(quantities[symbol]),
                entry_price=holding.entry_price if holding else math.nan,
                price=price,
                managed=holding is not None,
            )
        )
    return rows


# ── SVG: the shared furniture ────────────────────────────────────────────────


def _svg(width: int, height: int, body: str, label: str) -> str:
    """One chart, sized by its container.

    **No ``height`` attribute.** With both ``width="100%"`` and a fixed ``height`` beside a
    ``viewBox``, the default ``preserveAspectRatio`` scales the drawing to fit *both* and
    centres it - which is why the charts sat letterboxed in the middle of the page with a
    third of the viewport empty. Given only a width, the ``viewBox`` supplies the ratio and
    the chart uses the full content column.
    """
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'style="display:block;height:auto" '
        f'role="img" aria-label="{label}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{width}" height="{height}" fill="{PANEL}"/>{body}</svg>'
    )


def _text(
    x: float, y: float, value: str, fill: str, size: int = 9, anchor: str = "start"
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
        f'font-family="ui-monospace,Menlo,Consolas,monospace" letter-spacing="1.2" '
        f'text-anchor="{anchor}">{value}</text>'
    )


def _rule(
    x1: float, y1: float, x2: float, y2: float, stroke: str, dash: str = ""
) -> str:
    dashes = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="1"{dashes}/>'
    )


def channel_colour(channel: str) -> str:
    """The ramp entry for a channel, dark for slow and light for fast."""
    if channel not in CHANNEL_SPEED:
        return RAMP[len(RAMP) // 2]
    position = CHANNEL_SPEED.index(channel)
    return RAMP[min(position, len(RAMP) - 1)]


# ── GB-35: the forecast path ─────────────────────────────────────────────────


def forecast_svg(
    history: pd.Series,
    path: pd.Series,
    thresholds: Thresholds,
    symbol: str,
    width: int = 1400,
    height: int = 250,
) -> str:
    """Recent closes with the predicted H-day path continuing past the last bar.

    ``history`` is prices; ``path`` is the forecast **projected into prices** by the
    caller, so the chart shows one quantity on one axis rather than asking a reader to hold
    a log-return scale in their head beside a price one.

    **Nothing is drawn on top of data.** Every annotation - the NOW marker, the state of
    the calibrated band - sits in a strip below the plot area, separated from it by a
    hairline. The only chrome inside the plot is the threshold rule itself and the
    registration line at the last completed bar, both of which are reference geometry
    rather than labels competing with the price line for the same pixels.

    The calibrated entry threshold is an orange dashed rule - chrome, because it is a
    decision boundary rather than a measurement. **When the band is ``never()`` the line is
    not omitted**: the strip says so in words where the rule would have been. Omitting it
    would make a session that cannot trade look like one that simply had not yet, which is
    the difference between abstaining and waiting.
    """
    left, right, top = 58, 14, 26
    strip = 30
    plot_h = height - top - strip
    plot_w = width - left - right

    values = [float(v) for v in list(history) + list(path)]
    finite = [v for v in values if math.isfinite(v)]
    low, high = (min(finite), max(finite)) if finite else (0.0, 1.0)
    if high <= low:
        high = low + 1.0
    span = high - low

    def x_at(index: int) -> float:
        return left + plot_w * (index / max(1, len(values) - 1))

    def y_at(value: float) -> float:
        return top + plot_h * (1.0 - (value - low) / span)

    body = [
        _rule(left, top, left, top + plot_h, HAIRLINE),
        _rule(left, top + plot_h, left + plot_w, top + plot_h, HAIRLINE),
        _text(6, top + 8, f"{high:,.2f}", MUTED),
        _text(6, top + plot_h, f"{low:,.2f}", MUTED),
        _text(left, 14, f"{symbol}  CLOSE / FORECAST", ORANGE),
    ]

    history_points = " ".join(
        f"{x_at(i):.1f},{y_at(float(v)):.1f}" for i, v in enumerate(history)
    )
    body.append(
        f'<polyline points="{history_points}" fill="none" stroke="{RAMP[2]}" '
        'stroke-width="1.5"/>'
    )

    joined = [float(history.iloc[-1]), *[float(v) for v in path]]
    offset = len(history) - 1
    path_points = " ".join(
        f"{x_at(offset + i):.1f},{y_at(v):.1f}" for i, v in enumerate(joined)
    )
    body.append(
        f'<polyline points="{path_points}" fill="none" stroke="{RAMP[4]}" '
        'stroke-width="2" stroke-dasharray="4 3"/>'
    )
    body.append(
        f'<circle cx="{x_at(len(values) - 1):.1f}" cy="{y_at(joined[-1]):.1f}" r="3" '
        f'fill="{RAMP[5]}"/>'
    )
    body.append(_rule(x_at(offset), top, x_at(offset), top + plot_h, ORANGE_DIM, "2 3"))

    # The strip. A hairline, then the annotations, none of them over the plot.
    baseline = top + plot_h
    body.append(_rule(0, baseline + 10, width, baseline + 10, HAIRLINE, "2 4"))
    note_y = baseline + 24
    # NOW sits under the rule it names rather than at the edge of the strip: a marker whose
    # label is a screen away from it is a label for something else.
    body.append(_text(x_at(offset), note_y, "NOW", ORANGE_DIM, anchor="middle"))

    if thresholds.fires:
        entry = float(history.iloc[-1]) * math.exp(thresholds.lower)
        if low <= entry <= high:
            body.append(
                _rule(left, y_at(entry), left + plot_w, y_at(entry), ORANGE, "5 4")
            )
            note = f"ENTRY THRESHOLD {entry:,.2f}"
        else:
            note = f"ENTRY THRESHOLD {entry:,.2f} — OFF SCALE"
    else:
        note = "NO CALIBRATED BAND — STOOD ASIDE"
    body.append(_text(left, note_y, note, ORANGE))

    return _svg(width, height, "".join(body), f"{symbol} forecast path")


def price_path(last_close: float, forecast_path) -> pd.Series:
    """The forecast log-return path compounded onto the last close.

    One axis, one unit. The alternative — a second axis in log returns — asks a reader to
    do the compounding in their head to see where the model thinks the price goes.
    """
    prices, level = [], float(last_close)
    for step in forecast_path:
        level *= math.exp(float(step))
        prices.append(level)
    return pd.Series(prices, dtype="float64")


# ── GB-36: the contribution bars ─────────────────────────────────────────────


def contributions_svg(
    attribution: Attribution, width: int = 1400, row_height: int = 24
) -> str:
    """Per-channel contributions as a diverging bar chart around a zero rule.

    Shares come from :func:`explain.channel.shares` — the same function the narrative
    reads, so the number in the prose and the number on the bar cannot disagree. Positive
    is filled and to the right, negative is hollow and to the left; the ramp encodes which
    channel rather than which sign, per the module docstring.
    """
    ranked = sorted(shares(attribution).items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    height = 34 + row_height * len(ranked) + 26
    left, right = 132, 76
    plot_w = width - left - right
    middle = left + plot_w / 2
    widest = max((abs(value) for _, value in ranked), default=0.0) or 1.0

    body = [
        _text(12, 16, "PER-CHANNEL CONTRIBUTION", ORANGE),
        _text(width - 12, 16, "SHARE OF GROSS VIEW", MUTED, anchor="end"),
        _rule(middle, 26, middle, height - 22, ORANGE_DIM),
    ]

    for index, (channel, value) in enumerate(ranked):
        y = 34 + index * row_height
        length = abs(value) / widest * (plot_w / 2 - 6)
        colour = channel_colour(channel)
        x = middle if value >= 0 else middle - length
        fill = colour if value >= 0 else "none"
        body.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(length, 0.6):.1f}" '
            f'height="{row_height - 12}" fill="{fill}" stroke="{colour}" '
            'stroke-width="1"/>'
        )
        body.append(_text(12, y + row_height - 14, channel.upper(), PAPER))
        glyph = "▲" if value > 0 else ("▼" if value < 0 else "·")
        body.append(
            _text(
                width - 12,
                y + row_height - 14,
                f"{glyph} {abs(value) * 100:.1f}%",
                PAPER,
                anchor="end",
            )
        )

    survived = cancellation(attribution) * 100.0
    note = f"CANCELLATION — {survived:.1f}% OF THE GROSS VIEW SURVIVED"
    if survived < EXPLANATION_FRAGILE_BELOW * 100.0:
        note += " — FRAGILE, THIS DECOMPOSITION IS A RESIDUE"
    body.append(_text(12, height - 8, note, ORANGE if survived < 50.0 else MUTED))
    return _svg(width, height, "".join(body), "per-channel contributions")


# ── GB-53: the spectral panel (§6.5) ─────────────────────────────────────────
#
# **It renders only when the attribution carries a frequency view**, which is the FITS
# gate expressed structurally rather than by naming a model: `Attribution.per_frequency`
# is None for DLinear and persistence, so the panel is *absent* rather than empty. An
# empty panel invites a reader to wonder what broke; an absent one says the deployed model
# does not decompose that way.

#: What the white-noise control reproduced of the learned frequency response (GB-48, 48
#: models at three fold-grid anchors). The panel states it because a beautiful curve read
#: as "what the model learned about the market" is the black-box failure this project
#: exists to oppose, committed by the explanation layer itself.
SPECTRAL_GEOMETRY_SHARE = 0.86
SPECTRAL_NOISE_CORRELATION = 0.9485

#: Below this, no single cycle carries the forecast. Measured on real data: the largest
#: share across 24 contributors was 0.211, so the flag fires and must not be silent here
#: when the equivalent is loud in the channel panel.
SPECTRAL_DOMINANT_BELOW = 0.25

RIN_MEAN_LABEL = "RIN MEAN"
DEAD_BIN_LABEL = "BIN 0 · DEAD"


def period_label(period: float) -> str:
    """A period as a label. The RIN mean is **not** a frequency and is not labelled as one.

    ``per_frequency`` keys the window mean at ``DC_PERIOD`` (infinity) because it has to
    key it at something. Rendering that as "inf-day cycle" would invent a cycle nobody
    measured; it is the mean RIN removed and added back, and it says so.
    """
    return RIN_MEAN_LABEL if math.isinf(period) else f"{period:.1f}-DAY"


def period_colour(period: float, periods: Sequence[float]) -> str:
    """The ramp entry for a period, **dark for long and light for short**.

    The same ordering :func:`channel_colour` uses, so a reader who has learned the channel
    bars reads these for free: the ramp encodes how far back a thing looks. The RIN mean
    takes no ramp entry - it is not a frequency, and giving it one would place it on a
    scale it does not sit on.
    """
    if math.isinf(period):
        return MUTED
    finite = sorted((value for value in periods if math.isfinite(value)), reverse=True)
    if period not in finite:
        return RAMP[len(RAMP) // 2]
    step = max(len(finite) - 1, 1)
    index = round(finite.index(period) * (len(RAMP) - 1) / step)
    return RAMP[min(index, len(RAMP) - 1)]


def frequency_shares(attribution: Attribution) -> dict[float, float]:
    """Each period's share of the **gross** frequency view.

    Routed through :func:`explain.channel.shares` rather than recomputed, so the
    denominator - share of gross, never percent-of-net - has one definition in this
    codebase. See that function for why the gross denominator is the only bounded choice
    on daily log returns.
    """
    view = attribution.per_frequency or {}
    keyed = replace(attribution, per_channel={repr(k): v for k, v in view.items()})
    by_key = shares(keyed)
    return {period: by_key[repr(period)] for period in view}


def spectral_svg(
    attribution: Attribution, width: int = 1400, row_height: int = 22
) -> str:
    """Per-frequency contributions, keyed by period in **days** rather than by bin index.

    "The 17-day cycle" means something to a reader and "bin 7" does not (§6.5). Sign is
    geometry and glyph - filled and right for positive, hollow and left for negative -
    so the panel reads in greyscale; the ramp encodes period, never sign.
    """
    view = attribution.per_frequency or {}
    ranked = sorted(
        frequency_shares(attribution).items(), key=lambda kv: (-abs(kv[1]), kv[0])
    )
    height = 34 + row_height * len(ranked) + 40
    left, right = 150, 92
    plot_w = width - left - right
    middle = left + plot_w / 2
    widest = max((abs(value) for _, value in ranked), default=0.0) or 1.0
    periods = list(view)

    body = [
        _text(12, 16, "PER-FREQUENCY CONTRIBUTION", ORANGE),
        _text(width - 12, 16, "SHARE OF GROSS VIEW", MUTED, anchor="end"),
        _rule(middle, 26, middle, height - 36, ORANGE_DIM),
    ]

    for index, (period, share) in enumerate(ranked):
        y = 34 + index * row_height
        length = abs(share) / widest * (plot_w / 2 - 6)
        colour = period_colour(period, periods)
        dead = view[period] == 0.0
        x = middle if share >= 0 else middle - length
        fill = "none" if share < 0 or dead else colour
        body.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(length, 0.6):.1f}" '
            f'height="{row_height - 10}" fill="{fill}" stroke="{colour}" '
            f'stroke-width="1"{" stroke-dasharray=\"2 2\"" if dead else ""}/>'
        )
        label = period_label(period)
        if dead and math.isfinite(period):
            label = f"{label} · {DEAD_BIN_LABEL}"
        body.append(_text(12, y + row_height - 12, label, MUTED if dead else PAPER))
        glyph = "▲" if share > 0 else ("▼" if share < 0 else "·")
        body.append(
            _text(
                width - 12,
                y + row_height - 12,
                f"{glyph} {abs(share) * 100:.1f}%",
                MUTED if dead else PAPER,
                anchor="end",
            )
        )

    strongest = max((abs(value) for _, value in ranked), default=0.0)
    if strongest < SPECTRAL_DOMINANT_BELOW:
        note = (
            f"NO SINGLE CYCLE CARRIES THIS FORECAST — STRONGEST IS "
            f"{strongest * 100:.1f}% OF {len(ranked)} CONTRIBUTORS"
        )
        body.append(_text(12, height - 20, note, ORANGE))
    body.append(
        _text(
            12,
            height - 6,
            "BIN 0 MULTIPLIES ZERO — RIN REMOVES THE WINDOW MEAN BEFORE THE TRANSFORM",
            MUTED,
        )
    )
    return _svg(width, height, "".join(body), "per-frequency contributions")


def gain_phase_svg(
    attribution: Attribution, top: int = 6, width: int = 1400, row_height: int = 22
) -> str:
    """Gain and phase shift for the strongest contributors, **phase in days**.

    Radians of an unnamed cycle are not a thing anyone can picture (§6.5), so a shift is
    reported as the days by which the model advances or delays that cycle. Positive means
    it **leads**.
    """
    pairs = attribution.gain_phase or {}
    ranked = [
        period
        for period, _ in sorted(
            frequency_shares(attribution).items(), key=lambda kv: (-abs(kv[1]), kv[0])
        )
        if period in pairs and math.isfinite(period)
    ][:top]
    height = 34 + row_height * max(len(ranked), 1) + 10

    body = [
        _text(12, 16, "GAIN AND PHASE, STRONGEST CYCLES", ORANGE),
        _text(width - 12, 16, "GAIN ×   PHASE IN DAYS", MUTED, anchor="end"),
    ]
    widest = max((abs(pairs[p][0]) for p in ranked), default=0.0) or 1.0
    for index, period in enumerate(ranked):
        gain, shift = pairs[period]
        y = 34 + index * row_height
        length = abs(gain) / widest * (width - 150 - 200)
        body.append(
            f'<rect x="150" y="{y:.1f}" width="{max(length, 0.6):.1f}" '
            f'height="{row_height - 10}" fill="{period_colour(period, list(pairs))}" '
            'stroke="none"/>'
        )
        body.append(_text(12, y + row_height - 12, period_label(period), PAPER))
        lead = "▲ LEADS" if shift > 0 else ("▼ LAGS" if shift < 0 else "· NO SHIFT")
        body.append(
            _text(
                width - 12,
                y + row_height - 12,
                f"{gain:.3f} ×   {lead} {abs(shift):.2f} D",
                PAPER,
                anchor="end",
            )
        )
    if not ranked:
        body.append(_text(12, 44, "NO FINITE CYCLE CONTRIBUTED", MUTED))
    return _svg(width, height, "".join(body), "gain and phase")


def response_svg(
    periods: Sequence[float],
    gains: Sequence[float],
    width: int = 1400,
    height: int = 260,
) -> str:
    """The learned frequency response, **with what it was measured to be**.

    **The caption is the point of this chart, not decoration.** A curve rendered without
    it invites "this is what the model learned about the market", and GB-48 measured that
    it is mostly not: a model trained on white noise reproduces it at r = +0.9485, about
    86% of the response. Rendering the curve and letting a reader infer market structure
    would be the black-box behaviour this project opposes, committed by the explanation
    layer - the worst place for it to happen.
    """
    left, right, top, floor = 60, 24, 34, height - 46
    plot_w = width - left - right
    if len(periods) < 2:
        return _svg(
            width,
            96,
            _text(12, 16, "LEARNED FREQUENCY RESPONSE", ORANGE)
            + _text(12, 44, "NO RETAINED CYCLE TO PLOT", MUTED),
            "frequency response",
        )

    logs = [math.log(period) for period in periods]
    lo, hi = min(logs), max(logs)
    span = (hi - lo) or 1.0
    tallest = max(gains) or 1.0
    points = [
        (
            left + (value - lo) / span * plot_w,
            floor - (gain / tallest) * (floor - top),
        )
        for value, gain in zip(logs, gains, strict=True)
    ]

    body = [
        _text(12, 16, "LEARNED FREQUENCY RESPONSE", ORANGE),
        _text(width - 12, 16, "GAIN × BY PERIOD", MUTED, anchor="end"),
        _rule(left, floor, width - right, floor, HAIRLINE),
    ]
    body.append(
        '<polyline fill="none" stroke="{}" stroke-width="1.4" points="{}"/>'.format(
            RAMP[-2], " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        )
    )
    for (x, y), period, gain in zip(points, periods, gains, strict=True):
        body.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" '
            f'fill="{period_colour(period, list(periods))}"/>'
        )
        del gain
    for period in (periods[0], periods[len(periods) // 2], periods[-1]):
        x = left + (math.log(period) - lo) / span * plot_w
        body.append(_text(x, floor + 14, f"{period:.0f}D", MUTED, anchor="middle"))

    body.append(
        _text(
            12,
            height - 20,
            f"DOMINATED BY INTERPOLATION COST, NOT BY MARKET STRUCTURE — "
            f"{SPECTRAL_GEOMETRY_SHARE * 100:.0f}% OF THIS CURVE IS REPRODUCED BY A MODEL "
            f"TRAINED ON WHITE NOISE",
            ORANGE,
        )
    )
    body.append(
        _text(
            12,
            height - 6,
            f"MEASURED GB-48: GAIN CURVES CORRELATE r = +{SPECTRAL_NOISE_CORRELATION:.4f} "
            "ACROSS 48 MODELS AT THREE FOLD-GRID ANCHORS",
            MUTED,
        )
    )
    return _svg(width, height, "".join(body), "learned frequency response")


def spectral_panel(attribution: Attribution, response=None, width: int = 1400) -> str:
    """The whole panel, or **empty when the model does not decompose by frequency**.

    Absent rather than blank: `Attribution.per_frequency` is None for DLinear and
    persistence, and a panel that rendered an empty frame for them would read as a fault
    rather than as a property of the deployed model.
    """
    if not attribution.per_frequency:
        return ""
    parts = [spectral_svg(attribution, width), gain_phase_svg(attribution, width=width)]
    if response is not None:
        periods, gains = response
        parts.append(response_svg(periods, gains, width))
    return "".join(parts)


def is_rtl(text: str) -> bool:
    """True when the string contains Hebrew letters.

    Deliberately not ``dir="auto"``. That resolves direction from the first strong
    character, and every narrative GB-32 writes opens with a ticker — so ``auto`` would
    call a Hebrew sentence left-to-right and undo the isolation work that keeps a
    percentage on the correct side of its label.
    """
    return any(ord(character) in HEBREW for character in text)


def narrative_html(text: str) -> str:
    direction = "rtl" if is_rtl(text) else "ltr"
    return f'<p class="gb-narrative" dir="{direction}">{text}</p>'


def pending_summary(entry: dict) -> str:
    """The one line above a recommendation's Approve and Reject controls. **GB-37.**"""
    return (
        f"{entry['symbol']}  {float(entry['shares']):.4f} @ "
        f"{float(entry['price']):,.2f}  =  {float(entry['notional']):,.2f}  ·  "
        f"STOP {float(entry['stop_loss']):,.2f}  ·  TARGET "
        f"{float(entry['take_profit']):,.2f}"
    )


def provenance_label(provenance: str) -> str:
    """How a decision's origin is shown. **A replayed decision says so on its own row.**

    Never a blank for live and a badge for replay — a viewer would then have to know that
    the absence means something. Both are labelled, so neither can be read by default.
    """
    return "LIVE" if provenance == records.LIVE else provenance.upper()


def is_fragile(attribution: Attribution) -> bool:
    """Whether this decomposition is too cancelled to explain the forecast it belongs to."""
    return cancellation(attribution) < EXPLANATION_FRAGILE_BELOW


def escape(value: object) -> str:
    """Text into a cell. The narratives are ours, but a symbol is not worth trusting."""
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def table_html(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    numeric: Sequence[int] = (),
    flagged: Sequence[tuple[int, int]] = (),
) -> str:
    """A table in the design language, rendered as HTML.

    Streamlit's own dataframe arrives with a white ground, its own typeface and its own
    sort chrome, which is three design languages on one page. This is the fourth thing the
    module builds as a pure string returning function - like the SVG builders, it is
    unit-testable without a browser.

    Args:
        headers: Column titles. Rendered uppercase, tracked wide, and **the only orange in
            the table** - the rows are data and data is not chrome.
        rows: Already-formatted cells. Formatting is the caller's, because the caller knows
            whether a number is a price, a share or a log return.
        numeric: Indices of columns to right-align with tabular figures, so digits line up
            in their columns and a reader can compare magnitudes down a column by eye.
        flagged: ``(row, column)`` pairs to mark. A marked cell carries the one exception
            to the orange rule: the mark is an annotation *about* the value rather than a
            value itself, which is the same category as a header. It is one cell rather
            than the whole row because a single coloured cell in a grey table is already
            the only thing on the page the eye goes to, and colouring the row would spend
            the loudest signal available on emphasis rather than on meaning.
    """
    right = set(numeric)
    marked = set(flagged)
    head = "".join(
        f'<th class="{"num" if index in right else ""}">{escape(title)}</th>'
        for index, title in enumerate(headers)
    )
    body = []
    for number, row in enumerate(rows):
        cells = []
        for index, cell in enumerate(row):
            style = "num " if index in right else ""
            if (number, index) in marked:
                style += "gb-flag"
            cells.append(f'<td class="{style.strip()}">{escape(cell)}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f'<table class="gb-table"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
    )


def position_table(rows: Sequence[PositionRow]) -> str:
    """The positions panel. A quarantined holding shows em dashes, never zeros."""
    return table_html(
        ("SYMBOL", "QTY", "ENTRY", "LAST", "VALUE", "UNREALISED", "STATE"),
        [
            (
                row.symbol,
                f"{row.quantity:.9f}",
                f"{row.entry_price:,.2f}" if row.managed else EM_DASH,
                f"{row.price:,.2f}",
                f"{row.market_value:,.2f}",
                (
                    EM_DASH
                    if not row.managed or math.isnan(row.unrealised)
                    else f"{'▲' if row.unrealised >= 0 else '▼'} "
                    f"{row.unrealised:,.2f} ({row.unrealised_pct:+.2f}%)"
                ),
                "MANAGED" if row.managed else "QUARANTINED",
            )
            for row in rows
        ],
        numeric=(1, 2, 3, 4, 5),
    )


def decision_rows(decisions: list[DecisionRecord]) -> list[DecisionRecord]:
    """The log, newest first. One row per decision, and after the ruling of 19 Aug 2026
    exactly one decision per completed bar per symbol."""
    return sorted(decisions, key=lambda r: (r.as_of, r.symbol), reverse=True)


def decision_table(decisions: list[DecisionRecord]) -> str:
    """The decision log as a table.

    **A blank cell reads as missing data**, so a decision that produced no order says so
    with an em dash rather than with nothing. And a decision whose channels almost entirely
    cancelled is marked ``FRAGILE`` in the cancellation cell: at 0.15 more than four fifths
    of the gross channel view has offset, the explanation beside it is a residue, and
    without the mark the row looks exactly like one where the channels agreed.
    """
    ordered = decision_rows(decisions)
    rows, flagged = [], []
    for number, record in enumerate(ordered):
        survived = cancellation(record.attribution)
        fragile = is_fragile(record.attribution)
        if fragile:
            flagged.append((number, CANCELLATION_COLUMN))
        rows.append(
            (
                f"{record.as_of:%Y-%m-%d}",
                record.symbol,
                record.signal.action.upper(),
                f"{record.signal.trend_strength:+.4f}",
                f"{record.attribution.forecast_total:+.4f}",
                f"{survived:.4f}{FRAGILE_MARK if fragile else ''}",
                "YES" if record.order else EM_DASH,
                provenance_label(record.provenance),
            )
        )
    return table_html(
        (
            "AS OF",
            "SYMBOL",
            "ACTION",
            "TREND",
            "FORECAST",
            "CANCELLATION",
            "ORDER",
            "SOURCE",
        ),
        rows,
        numeric=(3, 4, 5),
        flagged=flagged,
    )


def expander_title(record: DecisionRecord) -> str:
    """One line that says enough to decide whether to open it.

    The trend strength is what the list is scanned for, so it belongs in the closed title;
    the cancellation is what says whether the explanation inside is worth reading, so it
    belongs there too, and its flag with it.
    """
    survived = cancellation(record.attribution)
    title = (
        f"{record.as_of:%Y-%m-%d}  {record.symbol}  "
        f"{record.signal.action.upper()}  ·  TREND "
        f"{record.signal.trend_strength:+.4f}  ·  CANCELLATION {survived:.2f}"
    )
    return f"{title}  ·  {FRAGILE_LABEL}" if is_fragile(record.attribution) else title


# ── the design language, as CSS ──────────────────────────────────────────────


def stylesheet() -> str:
    """Near-black ground, dashed hairlines, monospace tracking, registration marks."""
    return f"""
<style>
  .stApp {{ background: {INK}; }}
  html, body, [class*="css"] {{ color: {PAPER}; }}
  .gb-label {{
      font-family: ui-monospace, Menlo, Consolas, monospace;
      text-transform: uppercase; letter-spacing: .18em; font-size: .68rem;
      color: {ORANGE}; margin: 1.4rem 0 .5rem;
  }}
  .gb-meta {{
      font-family: ui-monospace, Menlo, Consolas, monospace;
      text-transform: uppercase; letter-spacing: .14em; font-size: .62rem;
      color: {MUTED};
  }}
  .gb-display {{
      font-family: "Arial Narrow", "Helvetica Neue", Inter, sans-serif;
      font-weight: 800; font-stretch: condensed; text-transform: uppercase;
      letter-spacing: -.01em; line-height: .92; color: {PAPER};
      font-size: 2.6rem; margin: 0;
  }}
  .gb-display em {{ color: {ORANGE}; font-style: normal; }}
  .gb-panel {{
      border: 1px dashed {HAIRLINE}; background: {PANEL};
      padding: .9rem 1rem; position: relative; margin-bottom: .9rem;
  }}
  .gb-panel::before, .gb-panel::after {{
      content: "+"; position: absolute; color: {ORANGE_DIM};
      font-family: ui-monospace, monospace; font-size: .7rem; line-height: 1;
  }}
  .gb-panel::before {{ top: -.4rem; left: -.35rem; }}
  .gb-panel::after {{ top: -.4rem; right: -.35rem; }}
  .gb-ruler {{
      display: flex; justify-content: space-between; border-bottom: 1px solid {HAIRLINE};
      padding-bottom: .2rem; margin-bottom: .8rem;
  }}
  .gb-status {{
      display: inline-block; border: 1px solid {ORANGE}; color: {ORANGE};
      padding: .1rem .5rem; font-family: ui-monospace, monospace; font-size: .68rem;
      letter-spacing: .18em;
  }}
  .gb-narrative {{
      font-size: .92rem; line-height: 1.55; color: {PAPER};
      border-left: 2px solid {ORANGE_DIM}; padding: .2rem 0 .2rem .8rem;
  }}
  .gb-narrative[dir="rtl"] {{
      border-left: none; border-right: 2px solid {ORANGE_DIM};
      padding: .2rem .8rem .2rem 0;
  }}
  .gb-figure {{ color: {PAPER}; font-size: 1.6rem; font-weight: 700; }}
  .gb-metarow {{ display: flex; gap: .8rem; align-items: baseline; margin-bottom: .18rem; }}
  .gb-metakey {{
      font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .6rem;
      letter-spacing: .2em; color: {ORANGE_DIM}; min-width: 5.2rem;
  }}

  /* Let the content use the column. The left ruler is fixed to the viewport edge, so
     the container is padded past it rather than pushed by a spacer element. */
  .stApp .block-container {{
      max-width: none; padding-left: {GUTTER}; padding-right: 1.4rem;
      padding-top: 3.4rem;
  }}
  .gb-lruler {{
      position: fixed; left: 0; top: 0; bottom: 0; width: 2.6rem; z-index: 5;
      border-right: 1px solid {HAIRLINE}; background: {INK};
      display: flex; flex-direction: column; justify-content: space-between;
      align-items: center; padding: 4.2rem 0 1.6rem;
      font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .58rem;
      letter-spacing: .1em; color: {MUTED};
  }}
  .gb-lruler span::before {{ content: "— "; color: {ORANGE_DIM}; }}

  .gb-table {{
      width: 100%; border-collapse: collapse; background: {PANEL};
      font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .72rem;
      margin-bottom: .9rem; border: 1px dashed {HAIRLINE};
  }}
  .gb-table th {{
      color: {ORANGE}; text-transform: uppercase; letter-spacing: .18em;
      font-size: .6rem; font-weight: 400; text-align: left;
      padding: .55rem .7rem; border-bottom: 1px solid {ORANGE_DIM};
  }}
  .gb-table td {{
      color: {PAPER}; padding: .45rem .7rem; border-bottom: 1px dashed {HAIRLINE};
      white-space: nowrap;
  }}
  .gb-table tbody tr:last-child td {{ border-bottom: none; }}
  .gb-table th.num, .gb-table td.num {{
      text-align: right; font-variant-numeric: tabular-nums;
      font-feature-settings: "tnum" 1;
  }}
  /* The one place a cell carries chrome: the mark is an annotation ABOUT the value, not
     a value itself, which is the same category as a header. */
  .gb-table td.gb-flag {{ color: {ORANGE}; }}

  /* The expanders arrive with the same default chrome the tables did - a white ground and
     a sans face - and there is no HTML equivalent to build instead, because the widget is
     what holds the disclosure state. So the widget stays and its skin is replaced. */
  [data-testid="stExpander"] details {{
      background: {PANEL}; border: 1px dashed {HAIRLINE}; border-radius: 0;
      margin-bottom: .5rem;
  }}
  [data-testid="stExpander"] summary {{
      background: {PANEL}; color: {PAPER};
      font-family: ui-monospace, Menlo, Consolas, monospace;
      text-transform: uppercase; letter-spacing: .14em; font-size: .66rem;
  }}
  [data-testid="stExpander"] summary:hover {{ color: {ORANGE}; }}
  [data-testid="stExpander"] summary p {{
      font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .66rem;
      letter-spacing: .14em;
  }}
  [data-testid="stExpander"] summary svg {{ fill: {ORANGE_DIM}; }}
  [data-testid="stExpander"] details > div {{ background: {PANEL}; border: none; }}
</style>
"""


def ruler_html(marks: int = 4) -> str:
    """The numbered grid rule along the top edge. Plain integers, as the references have."""
    ticks = "".join(
        f'<span class="gb-meta">{value:d}</span>'
        for value in range(0, 25 * marks + 1, 25)
    )
    return f'<div class="gb-ruler">{ticks}</div>'


def left_ruler_html(marks: int = 6) -> str:
    """The numbered rule down the left edge.

    Fixed to the viewport rather than laid out in a column, because a column would only be
    as tall as its own contents and the ruler in the references runs the height of the
    page. The content is padded past it by :data:`GUTTER`, so the ruler costs a gutter
    rather than the third of the viewport the old layout left empty.
    """
    ticks = "".join(f"<span>{value * 10:d}</span>" for value in range(marks))
    return f'<div class="gb-lruler">{ticks}</div>'


@dataclass(frozen=True)
class BandContext:
    """How the deployed band was selected, not just what it is (ruled 20 Aug 2026).

    ``Thresholds`` carries ``lower`` and ``upper`` and nothing about where they came from,
    which is correct for a frozen contract and not enough for a reader. **A band resting on
    8 trades and a band resting on 80 are not the same claim and must not read alike**, and
    the number quoted is a **maximum over a grid of fifteen candidates**, which is a
    selected statistic rather than an estimate. Under pure noise the max of fifteen
    candidates on eight trades is positive almost surely.

    The rule that produced it is not weakened by saying so — it discriminates, standing
    aside on 5 of 16 folds — and a rule that admitted noise freely would never stand aside
    at all.
    """

    stood_aside: bool
    val_sharpe: float | None
    val_trades: int | None
    fold: int | None

    @property
    def summary(self) -> str:
        """One line, for the masthead."""
        if self.stood_aside or self.val_sharpe is None:
            return "STOOD ASIDE — VALIDATION FOUND NO CANDIDATE WITH A POSITIVE SHARPE"
        return (
            f"FOLD {self.fold} &nbsp;·&nbsp; VAL SHARPE {self.val_sharpe:+.3f} "
            f"&nbsp;·&nbsp; OVER {self.val_trades} TRADES &nbsp;·&nbsp; "
            "GRID MAXIMUM OF 15 CANDIDATES"
        )


def band_context(path: Path) -> BandContext | None:
    """Read the band's selection context from the artefact GB-20 wrote.

    ``None`` when there is no artefact — the loop then stands aside for a different reason
    (no band at all), which ``status_of`` already says.
    """
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return BandContext(
        stood_aside=bool(raw.get("stood_aside")),
        val_sharpe=raw.get("val_sharpe"),
        val_trades=raw.get("val_trades"),
        fold=raw.get("fold"),
    )


def header_html(
    cfg: Config,
    status: str,
    reliability: Reliability | None,
    band: BandContext | None = None,
) -> str:
    """Project block, status chip and the reliability panel, in one strip.

    **Stacked** PROJECT / SYSTEM / VERSION, as the references have it: three labelled rows
    read as a masthead, one run-on row reads as a breadcrumb.

    The reliability numbers sit here rather than behind a tab, because the requirement is
    that they are unavoidable rather than available.
    """
    if reliability is None:
        record = (
            '<span class="gb-label">RELIABILITY</span> '
            '<span class="gb-meta">NOT MEASURED — run smoke_offline --prepare-live</span>'
        )
    else:
        arrow = "▲" if reliability.beats_the_bar else "▼"
        record = (
            f'<span class="gb-label">DIRECTION</span> '
            f'<span class="gb-meta">{reliability.direction:.4f} vs ALWAYS-LONG '
            f"{reliability.always_long:.4f} &nbsp; {arrow} {reliability.gap:+.4f} "
            f"&nbsp; OVER {reliability.folds} FOLDS &nbsp; MEASURED "
            f"{reliability.measured_on}</span>"
        )

    def line(label: str, value: str) -> str:
        return (
            f'<div class="gb-metarow"><span class="gb-metakey">{label}</span>'
            f'<span class="gb-meta">{value}</span></div>'
        )

    return (
        '<div class="gb-panel">'
        + line("PROJECT", f'<span style="color:{ORANGE}">GLASSBOX TRADER</span>')
        + line(
            "SYSTEM",
            f"{cfg.model.active.upper()} &nbsp;/&nbsp; {cfg.channels.active.upper()} "
            f"&nbsp;/&nbsp; {len(cfg.universe)} SYMBOLS &nbsp;/&nbsp; TOP_K "
            f"{cfg.signal.top_k}",
        )
        + line(
            "VERSION",
            f"V{cfg.meta.version} &nbsp;·&nbsp; CONFIG {config_hash(cfg)[:12].upper()}",
        )
        + f'<div style="margin-top:.7rem"><span class="gb-status">{status}</span></div>'
        + f'<div style="margin-top:.6rem">{record}</div>'
        + (
            ""
            if band is None
            else '<div style="margin-top:.4rem">'
            '<span class="gb-label">BAND</span> '
            f'<span class="gb-meta">{band.summary}</span></div>'
        )
        + "</div>"
    )


# ── the app ──────────────────────────────────────────────────────────────────


def main(
    state_dir: str | Path = "checkpoints/live", source: str = records.LIVE
) -> None:  # pragma: no cover
    """Render the dashboard. Exercised by ``streamlit run``, not by the suite.

    Every computation it performs lives in the pure functions above, which are tested. What
    is left here is layout, and layout is checked by looking at it.
    """
    import streamlit as st

    root = Path(state_dir)
    cfg = load_config()
    st.set_page_config(page_title="GlassBox Trader", layout="wide")
    st.markdown(stylesheet(), unsafe_allow_html=True)
    st.markdown(left_ruler_html(), unsafe_allow_html=True)
    st.markdown(ruler_html(), unsafe_allow_html=True)

    thresholds = _thresholds(root)
    reliability = load_reliability(root / RELIABILITY_FILE)
    book = Book.load(root / "book.json")
    quantities, prices, account = _broker_view(cfg)
    hours = _in_market_hours(cfg)

    st.markdown(
        header_html(
            cfg,
            status_of(thresholds, hours, quantities),
            reliability,
            band_context(root / "thresholds.json"),
        ),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="gb-label">POSITIONS</div>', unsafe_allow_html=True)
    rows = position_rows(book, quantities, prices)
    if rows:
        st.markdown(position_table(rows), unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="gb-meta">NO POSITIONS HELD</div>', unsafe_allow_html=True
        )
    st.markdown(
        f'<div class="gb-meta">EQUITY {account.get("equity", float("nan")):,.2f}'
        f' &nbsp;·&nbsp; CASH {account.get("cash", float("nan")):,.2f}</div>',
        unsafe_allow_html=True,
    )

    _copilot_panel(root, cfg, st)

    decisions = _recent_decisions(root, source)
    if not decisions:
        st.markdown(
            '<div class="gb-meta">NO DECISIONS RECORDED YET</div>',
            unsafe_allow_html=True,
        )
        return

    latest = {record.symbol: record for record in decisions}
    st.markdown('<div class="gb-label">FORECAST PATHS</div>', unsafe_allow_html=True)
    closes = _recent_closes(cfg)
    for symbol in sorted(latest):
        history = closes.get(symbol)
        if history is None or history.empty:
            continue
        st.markdown(
            forecast_svg(
                history,
                price_path(float(history.iloc[-1]), latest[symbol].forecast.path),
                thresholds,
                symbol,
            ),
            unsafe_allow_html=True,
        )

    st.markdown('<div class="gb-label">DECISION LOG</div>', unsafe_allow_html=True)
    st.markdown(decision_table(decisions), unsafe_allow_html=True)
    for record in decision_rows(decisions):
        with st.expander(expander_title(record)):
            st.markdown(narrative_html(record.narrative), unsafe_allow_html=True)
            st.markdown(contributions_svg(record.attribution), unsafe_allow_html=True)
            # GB-53. Empty string for a model that does not decompose by frequency, so
            # the panel is absent under DLinear and persistence rather than blank.
            st.markdown(spectral_panel(record.attribution), unsafe_allow_html=True)

    _refresh(cfg, st)


def _refresh(cfg: Config, st) -> None:  # pragma: no cover - a loop by design
    """Re-run the page on the live loop's own cadence.

    ``cfg.live.poll_seconds`` rather than a refresh constant of its own: a panel showing
    state that changes once a cycle should refresh once a cycle, and two numbers for one
    cadence is how the two end up disagreeing.
    """
    st.markdown(
        f'<div class="gb-meta">AUTO-REFRESH EVERY {cfg.live.poll_seconds}S</div>',
        unsafe_allow_html=True,
    )
    time.sleep(cfg.live.poll_seconds)
    st.rerun()


def _thresholds(root: Path) -> Thresholds:  # pragma: no cover - I/O
    from glassbox.live_loop import load_thresholds

    return load_thresholds(root / "thresholds.json")


def _in_market_hours(cfg: Config) -> bool:  # pragma: no cover - clock
    from glassbox.live_loop import in_session

    return in_session(cfg, pd.Timestamp.now(tz="UTC"))


def _broker_view(cfg: Config):  # pragma: no cover - network
    """Positions, marks and the account, degrading to empty rather than crashing."""
    del cfg
    try:
        from glassbox.engine.executor import AlpacaBroker

        broker = AlpacaBroker()
        quantities = broker.get_positions()
        prices = {
            position.symbol: float(position.current_price)
            for position in broker._client.get_all_positions()
            if position.current_price is not None
        }
        return quantities, prices, broker.get_account()
    except Exception:  # noqa: BLE001 - a dashboard must render without credentials
        return {}, {}, {}


def _recent_closes(cfg: Config) -> dict[str, pd.Series]:  # pragma: no cover - network
    """Display history only.

    ``HISTORY_BARS`` is far below ``min_history_bars`` on purpose and is **not** a
    weakening of that floor: the model window is assembled by the live loop from a full
    history, and this is a picture of recent closes beside it. Nothing here reaches a
    forecast.
    """
    try:
        from glassbox.data.live import load_live_bars

        bars = load_live_bars(list(cfg.universe), HISTORY_BARS)
        return {
            symbol: frame["close"].tail(HISTORY_BARS) for symbol, frame in bars.items()
        }
    except Exception:  # noqa: BLE001 - the rest of the page still renders
        return {}


def _recent_decisions(
    root: Path, source: str
) -> list[DecisionRecord]:  # pragma: no cover - I/O
    """The log, filtered by provenance.

    ``records.load_decisions`` defaults to live, so the dashboard shows real decisions
    unless it is asked for a replay by name. The failure direction is the safe one: a
    forgotten filter hides replayed decisions rather than presenting them as live.
    """
    today = pd.Timestamp.now(tz="UTC")
    window = pd.Timedelta(days=30 if source == records.LIVE else 3650)
    return records.load_decisions(today - window, today, root, source)


def _copilot_panel(root: Path, cfg: Config, st) -> None:  # pragma: no cover - widgets
    """Recommendations awaiting an answer, with Approve and Reject. **GB-37.**

    Rendered above the log because it is the only thing on the page that is waiting for a
    person. **No expiry**: a recommendation waits until it is answered, because a timer
    would be the system deciding "no" on the operator's behalf and recording nothing.
    """
    from glassbox.engine.executor import AlpacaBroker
    from glassbox.live_loop import answer_pending

    queue = records.load_pending(root)
    if not queue:
        return

    st.markdown(
        '<div class="gb-label">AWAITING APPROVAL — CO-PILOT</div>',
        unsafe_allow_html=True,
    )
    for entry in queue:
        st.markdown(
            f'<div class="gb-panel"><div class="gb-meta">{entry["decision_id"]}'
            f" &nbsp;·&nbsp; {provenance_label(entry.get('provenance', records.LIVE))}"
            f'</div><div class="gb-figure">{pending_summary(entry)}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(narrative_html(entry.get("narrative", "")), unsafe_allow_html=True)
        approve_col, reject_col = st.columns(2)
        if approve_col.button("APPROVE", key=f"a-{entry['decision_id']}"):
            answer_pending(cfg, AlpacaBroker(), root, entry["decision_id"], True)
            st.rerun()
        if reject_col.button("REJECT", key=f"r-{entry['decision_id']}"):
            answer_pending(cfg, AlpacaBroker(), root, entry["decision_id"], False)
            st.rerun()


if __name__ == "__main__":  # pragma: no cover
    # Arguments, not environment variables: only the config layer may read the environment
    # (CLAUDE.md rule 5), and `tests/config/test_config.py` enforces it. Streamlit passes
    # anything after `--` straight through:
    #     streamlit run glassbox/dashboard/app.py -- --state-dir checkpoints/replay     #         --source replay:fold-13
    _parser = argparse.ArgumentParser(prog="glassbox dashboard")
    _parser.add_argument("--state-dir", default="checkpoints/live")
    _parser.add_argument("--source", default=records.LIVE)
    _args, _ = _parser.parse_known_args(sys.argv[1:])
    main(_args.state_dir, _args.source)


__all__ = [
    "ASIDE",
    "CLOSED",
    "IDLE",
    "ORANGE",
    "RAMP",
    "RUNNING",
    "BandContext",
    "PositionRow",
    "Reliability",
    "band_context",
    "channel_colour",
    "contributions_svg",
    "decision_rows",
    "decision_table",
    "escape",
    "expander_title",
    "forecast_svg",
    "frequency_shares",
    "gain_phase_svg",
    "header_html",
    "is_fragile",
    "is_rtl",
    "left_ruler_html",
    "load_reliability",
    "main",
    "narrative_html",
    "pending_summary",
    "period_colour",
    "period_label",
    "position_rows",
    "position_table",
    "price_path",
    "provenance_label",
    "response_svg",
    "spectral_panel",
    "spectral_svg",
    "status_of",
    "stylesheet",
    "table_html",
]
