"""L7: the Streamlit dashboard — a live trading-desk console.

**Rebuilt 28 Aug 2026 (GB-63b).** The previous language was a blueprint: registration
marks, numbered grid rulers, dashed hairline panels, a PROJECT/SYSTEM/VERSION masthead,
monospace uppercase throughout. That served a static document. This is a console somebody
watches while a loop is running, and it is built as one — cards on a near-black ground,
dense numeric type, green and red as the primary language.

**Green is gain and red is loss, everywhere, without exception.** Consistency matters more
than restraint on a console: a reader who has to remember which panel uses which convention
is a reader who will misread one. The rule that survives from the old language, because it
costs nothing and protects somebody: **every status-coloured figure also carries a sign or
an arrow**, so the number is readable if the colour is not. :func:`status_html` is the only
producer of status colour and it guarantees that structurally.

**The blue ramp survives, scoped to two charts.** It encodes *which channel* and *which
frequency band* — a quantity, not a direction — and it appears only inside the attribution
and spectral panels. Outside them the ramp would be a third colour family competing with
gain and loss for the same eye. Tests hold both halves of that boundary.

**Every card declares where its numbers came from.** This system has made two live trades
and stands aside on most bars, while the backtest has 166 trades over sixteen folds. A
console that filled a calendar with backtest results while looking live would discredit the
one claim this project is actually making, so a card carries a LIVE, BACKTEST or REPLAY
pill, a card may not mix sources, and a card without enough live data says how little it
has rather than quietly borrowing from the other side.

**The charts are hand-built SVG.** Not a plotting library: an SVG builder is a **pure
function returning a string**, so the threshold line, the ramp, the radar geometry and the
calendar tiles are unit-testable without a browser. It also adds no dependency, and it is
what lets one card be one ``st.markdown`` call — which matters, because Streamlit inserts
its own wrappers between consecutive calls and a card assembled from several of them
cannot hold a border.

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
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from glassbox import records
from glassbox.config.loader import Config, config_hash, load_config
from glassbox.contracts.schemas import Attribution, DecisionRecord
from glassbox.dashboard.tokens import (
    ACCENT,
    DIM,
    GAIN,
    GROUND,
    HEBREW_SANS,
    LEADING_FIGURE,
    LEADING_PROSE,
    LEADING_UI,
    LOSS,
    MONO,
    RULE,
    RULE_FAINT,
    STAR_A,
    STAR_B,
    TEXT,
    TRACK_FIGURE,
    TRACK_LABEL,
    TYPE_BODY,
    TYPE_FIGURE,
    TYPE_LABEL,
    TYPE_PROSE,
)
from glassbox.engine.reconcile import Book
from glassbox.engine.signal import ENTER_LONG, EXIT, HOLD, Thresholds
from glassbox.explain.channel import cancellation, shares

# ── the palette ──────────────────────────────────────────────────────────────

# The values live in `tokens.py`, which is where the contrast test walks them. These are
# aliases, kept under their old names because ~40 SVG builders below thread `ORANGE` and
# `MUTED` through as arguments - renaming them would be forty edits for no change in
# meaning, and the meaning is what these names still carry.
#
# **`PANEL` is now the ground.** There are no panels: every region sits directly on
# `#0E1116`, separated by rules and space. Pointing the name at the ground removes every
# fill in one edit rather than in forty, and the alias stays so the SVG builders that fill
# their own background with it keep drawing on the same colour the page does.
INK = GROUND
PANEL = GROUND
HAIRLINE = RULE
PAPER = TEXT
MUTED = DIM

# Chrome, and only chrome: section rules, active states, the symbol column, selection.
# **A number is never vermillion and a rule is never green.** The two families answer
# different questions - what kind of thing is this, and which way did it go - and a mark
# that could belong to either is a mark the reader has to decode.
ORANGE = ACCENT
ORANGE_DIM = RULE

# Data. Dark for slow, light for fast. **Deliberately NOT in `tokens.py`**: those are
# chrome and status, which carry text a reader must resolve and are therefore held to a
# contrast floor. A ramp entry is a mark inside a chart that encodes *which* channel or
# *which* band - a quantity, not a value and not a label - and holding it to the text
# floor would flatten the ramp into six colours of the same lightness, destroying the one
# thing it encodes.
#
# Its territory is exactly two charts, the attribution bars and the spectral panel.
# `test_the_ramp_stays_inside_attribution_and_spectral` holds both directions: the ramp
# appears nowhere else, and status colour appears nowhere inside those two.
RAMP = ("#0A2239", "#123F63", "#1B6CA8", "#2E97D4", "#6FC3EC", "#B7E3F7")

#: Area fills under the equity curves, at 15% - light enough that a rule reads through,
#: solid enough to carry the sign at a glance. **Three of them, because there are three
#: cases**: this loop stands aside on most bars, so an exactly flat result is the common
#: one and colouring it green says the system gained when it did nothing.
GAIN_FILL = "rgba(34,197,94,0.15)"
LOSS_FILL = "rgba(239,68,68,0.15)"
FLAT_FILL = "rgba(125,135,148,0.15)"

#: Every surface that encodes data with colour. The ramp's territory is exactly two charts;
#: `test_the_ramp_stays_inside_attribution_and_spectral` holds both directions of that.
STATUS_COLOURS = (GAIN, LOSS)

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


# ── cards, and the source every one of them must declare ────────────────────

LIVE, BACKTEST, REPLAY = "LIVE", "BACKTEST", "REPLAY"

#: Pill colours. BACKTEST is deliberately the dimmest of the three: it is the source with
#: the most rows and the least authority about what the system is doing right now, and a
#: reader skimming for what is live should not have their eye caught by it first.
_PILL = {LIVE: GAIN, BACKTEST: MUTED, REPLAY: ACCENT}


@dataclass(frozen=True)
class Source:
    """Where a card's numbers came from. **Required, never defaulted.**

    This system has made two live trades and stands aside on most bars, while the backtest
    has 166 trades across sixteen folds. The failure this type exists to prevent is a
    console that looks live while showing backtest numbers - which would discredit the
    project's central claim far more effectively than any missing feature.

    ``detail`` is required for the same reason the kind is: "BACKTEST" alone invites a
    reader to assume a period, so a backtest card states its fold range and a replay card
    states its fold. A live card states how much live data it actually has, which for this
    system is usually a small number and saying so is the point.
    """

    kind: str
    detail: str

    def __post_init__(self) -> None:
        if self.kind not in _PILL:
            raise ValueError(
                f"unknown source {self.kind!r}; expected one of {list(_PILL)}"
            )
        if not self.detail.strip():
            raise ValueError(
                f"a {self.kind} card must say which data it is showing. 'BACKTEST' alone "
                "lets a reader assume a period nobody stated"
            )

    @property
    def pill(self) -> str:
        return (
            f'<span class="gb-pill" style="color:{_PILL[self.kind]};'
            f'border-color:{_PILL[self.kind]}33">{escape(self.kind)}'
            f'<span class="gb-pill-detail">{escape(self.detail)}</span></span>'
        )


def region(label: str, source: Source, body: str, note: str = "") -> str:
    """One section: a 2px accent rule, the label with its source pill, then the body.

    **Replaces :func:`card`, and carries no box.** The rule and the space above it are the
    whole separation - a region that needs to feel distinct gets more space, not a border
    on four sides. Vermillion appears here and only here as a rule; it never expresses a
    value, which is why the pill's colour comes from the source and the numerals' from
    their sign.

    One string and one ``st.markdown``, for the reason :func:`card` had: Streamlit wraps
    every markdown call in a container of its own, and a region assembled from several
    calls gets those wrappers threaded between its parts.
    """
    return (
        '<div class="gb-region"><div class="gb-region-rule"></div>'
        f'<div class="gb-region-head"><span class="gb-label">{escape(label)}</span>'
        f"{source.pill}</div>"
        f"<div>{body}</div>"
        + (f'<div class="gb-note">{escape(note)}</div>' if note else "")
        + "</div>"
    )


def too_little(source: Source, message: str) -> str:
    """The body a card renders when it has nothing worth plotting.

    A card in this state says what it has - "3 sessions, 2 trades" - rather than drawing an
    empty axis, and never borrows from the other source to fill itself. Its own function
    because the temptation to backfill is strongest exactly here.
    """
    return f'<div class="gb-empty">{escape(message)}</div>'


# ── reading the study's artefacts ────────────────────────────────────────────
#
# `glassbox.dashboard` sits BELOW `backtest` and `experiments` in the layers contract
# (pyproject.toml), so it may not import either and cannot run a backtest. It reads their
# output files, which is not an import and does not cross the layer.

RESULTS_PATH = "results.csv"
DAILY_EQUITY_PATH = "report/daily_equity.csv"


def load_results(path: str | Path = RESULTS_PATH) -> pd.DataFrame:
    """The study's per-fold results, or an empty frame.

    Empty rather than raising: a clean clone has no grid output, and a dashboard that
    refused to start without one would be unusable on exactly the machine where somebody
    is trying to see whether the live loop works.
    """
    source = Path(path)
    if not source.is_file():
        return pd.DataFrame()
    return pd.read_csv(source)


def load_daily_equity(path: str | Path = DAILY_EQUITY_PATH) -> pd.DataFrame:
    """The dated daily equity curves, or an empty frame.

    Written by `experiments/study.py` for the reference condition on real data only. Absent
    until a grid has been run since 28 Aug 2026, and the cards that need it say so rather
    than drawing nothing.
    """
    source = Path(path)
    if not source.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(source)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def reference_rows(results: pd.DataFrame, channels: str = "C0_base") -> pd.DataFrame:
    """The reference condition on real data: the rows the report's headline comes from.

    Filtered here rather than at each card, so every backtest card on the page shows the
    same condition and a reader comparing two of them is comparing like with like.

    **`channels` and `target_in_loop` are part of the condition and filtering them is not
    optional.** Without them one arm appears four times - twice for the two feature sets
    and twice for the two execution models - and every card silently multiplies: the fold
    chart drew 64 bars labelled `f1 f1 f2 f3 f4 f4`, and the equity curve chained 356 daily
    points over 178 distinct dates, each date counted twice. Found in the first screenshot
    of the rebuild, by the duplicated fold labels being visible.
    """
    if results.empty:
        return results
    live = results[~results["skipped"].astype(bool)]
    wanted = [
        ("anchor", 0),
        ("control", "real"),
        ("lr", 0.001),
        ("cutoff_period_days", 5),
        ("target_in_loop", True),
    ]
    for column, value in wanted:
        if column in live.columns:
            live = live[live[column] == value]
    if "channels" in live.columns:
        # Buy-and-hold carries no channel set - it has no features - so it is kept
        # alongside rather than filtered out with the other arms' second copy.
        live = live[live["channels"].isna() | (live["channels"] == channels)]
    return live


def backtest_source(path: str, frame: pd.DataFrame) -> Source:
    """The pill for a region drawn from ``frame``, naming the file ``frame`` was read from.

    **Both halves come from the region's own data, which is region 4's whole subject.**
    Two of these regions read `results.csv` and two read `report/daily_equity.csv`, and the
    two artefacts can be any number of folds apart - a partial daily file had both of its
    cards announcing "folds 1-16" while holding three, the overclaim the pills exist to
    prevent, committed by the pills.

    The detail names the file for the half of that defect the fix alone does not cover:
    side by side, "folds 1-16" and "folds 1-3" read as a bug unless the pill says they are
    two different files. The range is computed from ``frame``, never from a sibling region
    and never from the config.
    """
    return Source(BACKTEST, f"{Path(path).name} {fold_range(frame)}")


def fold_range(frame: pd.DataFrame) -> str:
    """``folds 1-16`` for a source pill, or a count when the folds are not contiguous."""
    if frame.empty or "fold" not in frame.columns:
        return "no folds"
    folds = sorted({int(f) for f in frame["fold"] if int(f) > 0})
    if not folds:
        return "no folds"
    if folds == list(range(folds[0], folds[-1] + 1)):
        return f"folds {folds[0]}-{folds[-1]}"
    return f"{len(folds)} folds"


# ── status: colour that never travels alone ──────────────────────────────────

#: Fractions of the original stop distance still unspent. Beyond `NEAR`, a position is
#: ordinary; inside it, the row is emphasised and marked; inside `IMMINENT`, the row says
#: in words that a stop fill is close and what happens when it fills.
NEAR = 0.5
IMMINENT = 0.25

ORDINARY, APPROACHING, CLOSE = "ordinary", "approaching", "imminent"


def status_glyph(value: float) -> str:
    """The glyph that carries the sign when colour cannot.

    Printed beside every status-coloured number, which is what makes the colour redundant
    rather than load-bearing. `math.nan` gets the dash: a quarantined position has no
    basis to compute a gain from, and an arrow would assert a direction nobody measured.
    """
    if math.isnan(value):
        return "—"
    return "▲" if value > 0 else "▼" if value < 0 else "—"


def status_colour(value: float) -> str:
    """GAIN, LOSS, or MUTED for flat and unknown."""
    if math.isnan(value) or value == 0:
        return MUTED
    return GAIN if value > 0 else LOSS


def status_fill(value: float) -> str:
    """The area fill matching :func:`status_colour`, flat included.

    Paired with the stroke colour rather than chosen beside it, so an area and its outline
    cannot end up making different claims about the same number.
    """
    if math.isnan(value) or value == 0:
        return FLAT_FILL
    return GAIN_FILL if value > 0 else LOSS_FILL


def status_html(value: float, text: str) -> str:
    """A status-coloured figure that **cannot** be rendered without its glyph and sign.

    One function rather than a colour constant used at each call site, and that is the
    whole design: the redundancy rule is a property of this function, so a caller cannot
    forget it, and `test_every_status_colour_is_redundant` has one place to check.
    """
    return (
        f'<span style="color:{status_colour(value)}">{status_glyph(value)}&nbsp;'
        f"{text}</span>"
    )


@dataclass(frozen=True)
class PositionRow:
    """One held position, marked to the last price the dashboard could see."""

    symbol: str
    quantity: float
    entry_price: float
    price: float
    managed: bool
    #: The protective levels the book recorded when the position was opened. NaN for a
    #: quarantined position, which the system did not open and cannot describe.
    stop_loss: float = math.nan
    take_profit: float = math.nan

    @property
    def market_value(self) -> float:
        return self.quantity * self.price

    @property
    def stop_room(self) -> float:
        """Fraction of the original entry-to-stop distance still unspent.

        1.0 at the entry price, 0.0 at the stop, negative below it. Expressed against the
        *original* distance rather than as a percentage of price, because that is the
        quantity a reader is actually asking about: how much of the room this position was
        given has it used. A 2% move means something different on a 3% stop than on a 10%
        one, and a percentage of price cannot tell them apart.
        """
        span = self.entry_price - self.stop_loss
        if math.isnan(span) or span <= 0 or math.isnan(self.price):
            return math.nan
        return (self.price - self.stop_loss) / span

    @property
    def stop_proximity(self) -> str:
        """`ORDINARY`, `APPROACHING` or `CLOSE`. Unknown room reads as ordinary.

        Unknown reads ordinary and not as an alarm: a quarantined position has no stop the
        system set, so escalating it would be raising an alarm about a number that does
        not exist.
        """
        room = self.stop_room
        if math.isnan(room):
            return ORDINARY
        if room <= IMMINENT:
            return CLOSE
        if room <= NEAR:
            return APPROACHING
        return ORDINARY

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
                stop_loss=holding.stop_loss if holding else math.nan,
                take_profit=holding.take_profit if holding else math.nan,
            )
        )
    return rows


# ── SVG: the shared furniture ────────────────────────────────────────────────

#: The viewBox width of the charts that take the whole content column. **Chosen against
#: the type floor, not by eye.** :func:`_svg` never draws a chart narrower than its
#: viewBox, so this is also the narrowest content column at which the page does not
#: scroll: 1200 clears a 1512px laptop with Streamlit's own gutters, where the previous
#: 1400 did not and would have put every one of these charts behind a scrollbar.
WIDE_CHART = 1200

#: The drop from a radar axis label to the value beneath it. One line of
#: :data:`~glassbox.dashboard.tokens.TYPE_LABEL` at the console's leading, rounded up: the
#: previous 11 was less than the type is now tall, so the two lines overlapped.
RADAR_AXIS_LEADING = 15


def _svg(width: int, height: int, body: str, label: str) -> str:
    """One chart, sized by its container but never smaller than its own viewBox.

    **No ``height`` attribute.** With both ``width="100%"`` and a fixed ``height`` beside a
    ``viewBox``, the default ``preserveAspectRatio`` scales the drawing to fit *both* and
    centres it - which is why the charts sat letterboxed in the middle of the page with a
    third of the viewport empty. Given only a width, the ``viewBox`` supplies the ratio and
    the chart uses the full content column.

    **``min-width`` is what makes the type floor true inside a chart.** SVG text is sized in
    user units, so a label's rendered size is its authored size times the ratio of drawn
    width to viewBox width - which means a scaling chart has no size floor at all, and the
    9-unit labels this console shipped with rendered at 7px in a narrow column. Pinning the
    drawn width at or above the viewBox width makes that ratio at least 1, so
    :data:`~glassbox.dashboard.tokens.TYPE_LABEL` user units is at least ``TYPE_LABEL`` px
    everywhere. Past that the container scrolls, which is the trade taken deliberately: a
    reader can scroll a chart and cannot enlarge a caption.
    """
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'style="display:block;height:auto;min-width:{width}px" '
        f'role="img" aria-label="{label}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{width}" height="{height}" fill="{PANEL}"/>{body}</svg>'
    )


def _text(
    x: float,
    y: float,
    value: str,
    fill: str,
    size: int = TYPE_LABEL,
    anchor: str = "start",
) -> str:
    """One chart label, on the type scale and in the page's own family.

    **The family was the second place the font stack was defined.** This wrote
    ``ui-monospace,Menlo,Consolas,monospace`` - a stack IBM Plex Mono is not in - so every
    label in every chart set in the system monospace while the surrounding page set in
    Plex, on a page whose stylesheet asserts one family. Nothing failed: the two are both
    monospace and the disagreement reads as a rendering quirk rather than as a defect.

    **Tracking is proportional, and was not.** A constant 1.2 user units is 17% of the em
    at 7 units and 11% at 11, so the smallest labels - the ones already hardest to read -
    received half again the tracking of the largest. It is now
    :data:`~glassbox.dashboard.tokens.TRACK_LABEL` of the size, which is the same value the
    stylesheet gives uppercase chrome.
    """
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
        f'font-family="{MONO}" letter-spacing="{size * TRACK_LABEL:.2f}" '
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


# ── live elements: things whose number actually moves ────────────────────────
#
# Each of these exists because the quantity underneath it changes between refreshes and
# nothing on the page showed it. **None of them animates.** A pulse, a spinner or a fade
# would make the panel look alive whether or not anything had happened, which is the one
# thing an instrument must not do: a number here moves because it moved.

#: How many seconds of equity history the session curve keeps. A session is 6.5 hours;
#: this holds a full one and discards yesterday, because the curve is *this session's*.
EQUITY_WINDOW_SECONDS = 7 * 3600

EQUITY_FILE = "equity_snapshots.jsonl"

#: The session curve's plot area is the card minus a header band and a footer band.
#: **The title used to be drawn inside the plot**, at ``top + 4``, while the
#: opening-balance label is positioned at ``y(opening)`` - which lands on the plot's
#: ceiling whenever the opening is the session high, that is, whenever a session only ever
#: loses. On 3 Sep 2026 the two overprinted and the chart read ``EQ100,000ITY``: the title
#: spanned x 12.0-65.3 and the balance x 21.8-84.0, both on y 17-31. The fix is a band and
#: not a shorter string - two labels cannot share a line if neither may enter the other's
#: band - and both ends are reserved because the mirror case is real: a session that only
#: gains puts the same label on the plot floor, which cleared the reading count by 0.4
#: units. The type is 12 units on a 0.74-em advance, so a band has to hold about 13 units
#: of glyph plus the 3-unit drop the balance label takes below its rule.
EQUITY_HEADER, EQUITY_FOOTER = 32, 32

#: Baselines inside those bands: the title row, and the reading count as an inset from the
#: bottom edge. The count hangs from the card rather than from the plot floor, because the
#: balance label hangs *into* the top of the footer band and the two would meet again.
EQUITY_TITLE_Y, EQUITY_COUNT_INSET = 16, 10

#: The smallest vertical span an equity curve is drawn at, as a fraction of the level it
#: sits at. **Taken from the risk policy rather than by eye.** The worst outcome one
#: position is permitted is ``risk.max_position_pct`` of equity stopped out at
#: ``risk.stop_loss_pct`` - 10% x 3% = 30 basis points - so a move that fills this chart
#: from top to bottom is a move the size of the largest single loss the system can take,
#: and anything smaller is drawn smaller. On 3 Sep 2026 a live session that moved $13.35
#: on $100,000 filled the plot with 1.3 basis points; against this floor it moves 5 units
#: of 116, which is what 1.3 basis points looks like.
#:
#: **This is one fact in two places** - here and in `risk:` in
#: `glassbox/config/settings.yaml` - so
#: `test_the_curve_floor_is_the_risk_policys_worst_single_position` is what makes them
#: equal. Deriving it would mean reading config inside a pure chart function, and these
#: builders take a frame and return a string.
#:
#: **This sentence could not be written until the guard that forbade it was fixed.**
#: `test_no_module_reads_settings_or_environ_directly` scanned the raw source of every
#: module outside `glassbox/config/` for the filename, so naming the file *in prose* failed
#: exactly as an `open()` of it would, and the cheapest way to a green suite was to delete
#: the explanation - on a rule whose whole point is that config is the single source of
#: truth and that a module should say where its numbers come from. It now measures the
#: code: comments and docstrings are blanked, every other string literal kept.
EQUITY_MIN_SPAN = 0.10 * 0.03


def append_equity(root: str | Path, when: pd.Timestamp, equity: float) -> None:
    """Append one equity reading, if it is a number.

    **The dashboard writes exactly one file and this is it**, deliberately separate from
    the loop's book and decision log: the panel is a reader of the loop's state, and a
    second writer into `book.json` is the hazard the state-directory lock exists to
    prevent. An append-only side file cannot corrupt anything the loop reads.
    """
    if math.isnan(equity):
        return
    path = Path(root) / EQUITY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps({"at": when.isoformat(), "equity": float(equity)}) + "\n"
        )


def load_equity(root: str | Path, now: pd.Timestamp) -> pd.Series:
    """The session's equity readings, oldest first, trimmed to the window.

    A malformed line is skipped rather than fatal: this file is written on every refresh
    and read on the next one, so a half-written line is a real possibility and losing the
    whole curve to it would be the wrong trade.
    """
    path = Path(root) / EQUITY_FILE
    if not path.is_file():
        return pd.Series(dtype="float64")
    stamps, values = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            stamps.append(pd.Timestamp(row["at"]))
            values.append(float(row["equity"]))
        except (ValueError, KeyError, TypeError):
            continue
    if not stamps:
        return pd.Series(dtype="float64")
    series = pd.Series(values, index=pd.DatetimeIndex(stamps)).sort_index()
    return series[series.index >= now - pd.Timedelta(seconds=EQUITY_WINDOW_SECONDS)]


def _segments(
    points: list[tuple[float, float]], values: list[float], opening: float
) -> list[tuple[str, list[tuple[float, float]]]]:
    """Split a polyline at the opening level, one run per side - and flat is a side.

    **Colouring the whole line by its final sign was the wrong reading of the data.** A
    session that spends most of itself under water and closes a cent up is not a green
    session, and one line in one colour cannot say that. Splitting at the crossing lets
    the chart state where the equity *was*, not only where it ended.

    **Three sides, not two, and by :func:`status_colour` rather than by a rule of its
    own.** The first attempt seeded the side from ``values[0] >= opening``, which is always
    true, so every session opened with a green stub. The correction seeded it from the
    first value that *differed* from the opening - which reads ahead through the whole
    series and applies the answer backwards. On 3 Sep 2026 a live session sat at its
    opening balance for 64 of 69 readings and then lost $13, and all 64 unchanged readings
    were painted loss red by a move that had not happened yet. A reading that has not
    moved is not a loss. The calendar has said so since it gained its third tile,
    :func:`fold_bars_svg` says so with its aside rule, and :func:`cumulative_equity_svg`
    says so for the arm that stands aside in all sixteen folds; this was the one place on
    the page where flat had no state, and it now calls the function they call.

    A crossing between the two signed sides is interpolated, so the colour changes exactly
    at the opening level and the runs meet on the rule instead of overlapping it. A change
    into or out of flat needs no interpolation: one of the two samples *is* the opening.
    """
    runs: list[tuple[str, list[tuple[float, float]]]] = []
    side = status_colour(values[0] - opening)
    current: list[tuple[float, float]] = [points[0]]
    for index in range(1, len(points)):
        nxt = status_colour(values[index] - opening)
        if nxt != side:
            if MUTED in (side, nxt):
                # One of the two samples sits exactly on the opening, so the join is that
                # sample: there is nothing between them to interpolate, and the run that
                # is leaving or entering flat already ends on the rule.
                join = points[index - 1] if side == MUTED else points[index]
            else:
                previous, now = values[index - 1], values[index]
                fraction = (opening - previous) / (now - previous)
                (x0, y0), (x1, y1) = points[index - 1], points[index]
                join = (x0 + (x1 - x0) * fraction, y0 + (y1 - y0) * fraction)
            if join != current[-1]:
                current.append(join)
            runs.append((side, current))
            side, current = nxt, [join]
        if points[index] != current[-1]:
            current.append(points[index])
    runs.append((side, current))
    return [(colour, run) for colour, run in runs if len(run) > 1]


def _vertical(
    values: Sequence[float], top: float, bottom: float, min_span_fraction: float
) -> tuple[Callable[[float], float], bool]:
    """A y-mapper for a series, floored so a negligible move is drawn as negligible.

    **One definition, because the flat case is the common one and it had been written
    twice.** `equity_svg` got the fix in region 3 and `cumulative_equity_svg` did not: an
    arm that stands aside in every fold chains to an exactly flat curve - persistence does,
    in all sixteen - and `max(high - low, 1e-9)` drove every point of it to the floor of the
    region under a column of empty space. Two copies of a scaling rule, one corrected, and
    nothing to make them equal.

    **Exactly flat was the case that got fixed; near-flat is the one that hurts.** A curve
    pinned to its own min and max fills the plot whatever the move was worth. On 3 Sep 2026
    a live session that moved $13.35 on $100,000 - 1.3 basis points - spent every unit of
    plot height on it, welding 64 unchanged readings to the ceiling and dropping the last
    three to the floor. Nothing left the box: the chart drew a cliff out of nothing
    happening, which is a lie about magnitude and the worse of the two failures because it
    looks like data. ``min_span_fraction`` is the smallest span the plot may represent, as
    a fraction of the level the series sits at; a narrower series is centred inside it.
    :data:`EQUITY_MIN_SPAN` is the value and says where it comes from.

    **The mapper clamps, and the clamp is not redundant with the arithmetic.** Mapping min
    to `bottom` and max to `top` contains the path only while the extremes come from the
    same values being drawn - which stops being true the moment a floor widens the range,
    and was never true of a value that is not finite. Containment is now a property of the
    mapper rather than an argument about its caller.

    Returns the mapper and whether the series is flat, because a caller that fills under
    its curve has to know which of the two it drew.
    """
    low, high = min(values), max(values)
    flat = (high - low) <= abs(high) * 1e-9
    centre = (top + bottom) / 2
    floor = abs((high + low) / 2) * min_span_fraction
    if (high - low) < floor:
        middle = (high + low) / 2
        low, high = middle - floor / 2, middle + floor / 2

    def y(value: float) -> float:
        if flat or high <= low or math.isnan(value):
            return centre
        scaled = bottom - (value - low) / (high - low) * (bottom - top)
        return min(max(scaled, top), bottom)

    return y, flat


def equity_svg(series: pd.Series, width: int = 720, height: int = 180) -> str:
    """The session equity curve: gain above the opening balance, loss below, flat on it.

    **A flat session is the common case here, not an edge case.** The loop stands aside on
    most bars, so equity does not move, and scaling by ``max(high - low, 1e-9)`` once drove
    every point to the floor of the region under a column of empty space. A span negligible
    against the level is drawn as a centred flat line, which is what it is - see
    :data:`EQUITY_MIN_SPAN` for how small "negligible" is and why that number.

    The plot is the card minus :data:`EQUITY_HEADER` and :data:`EQUITY_FOOTER`, because the
    opening-balance label rides on ``y(opening)`` and the title and the reading count do
    not move: reserving the two bands is what keeps them off each other.
    """
    if series.empty:
        return ""
    left, right = 90, width - 20
    top, bottom = EQUITY_HEADER, height - EQUITY_FOOTER
    values = [float(value) for value in series.to_numpy(dtype="float64")]
    opening, latest = values[0], values[-1]
    change = latest - opening
    y, _ = _vertical(values, top, bottom, EQUITY_MIN_SPAN)

    step = (right - left) / max(len(values) - 1, 1)
    points = [(left + index * step, y(value)) for index, value in enumerate(values)]

    body = [
        _text(12, EQUITY_TITLE_Y, "EQUITY", DIM),
        _rule(left, y(opening), right, y(opening), RULE, dash="2 3"),
        _text(left - 6, y(opening) + 3, f"{opening:,.0f}", DIM, anchor="end"),
    ]
    for colour, run in _segments(points, values, opening):
        drawn = " ".join(f"{x:.1f},{point_y:.1f}" for x, point_y in run)
        body.append(
            f'<polyline points="{drawn}" fill="none" stroke="{colour}" '
            'stroke-width="1.5"/>'
        )
    body.append(
        _text(
            right,
            EQUITY_TITLE_Y,
            f"{status_glyph(change)} {change:+,.2f}",
            status_colour(change),
            size=TYPE_BODY,
            anchor="end",
        )
    )
    body.append(
        _text(
            12,
            height - EQUITY_COUNT_INSET,
            f"{len(values)} READINGS THIS SESSION",
            DIM,
        )
    )
    return _svg(width, height, "".join(body), "session equity curve")


def sparkline_svg(
    prices: pd.Series, row: PositionRow, width: int = 320, height: int = 64
) -> str:
    """One position's recent price against the band its stop and target define.

    The band is the point. A price is a number without them and a *position* with them:
    the same 218.50 is comfortable inside a wide band and nearly closed inside a narrow
    one, and the stop distance is the thing this panel keeps having to explain in words.
    Drawn in chrome, not status colour - the band is furniture, and the rule about status
    colour staying out of data-encoding marks applies here as it does to the ramp charts.
    """
    if prices.empty or math.isnan(row.stop_loss):
        return ""
    left, right, top, bottom = 4, width - 4, 6, height - 6
    values = prices.to_numpy(dtype="float64")
    low = min(float(values.min()), row.stop_loss)
    high = max(
        float(values.max()),
        row.take_profit if not math.isnan(row.take_profit) else float(values.max()),
    )
    span = max(high - low, 1e-9)

    def y(value: float) -> float:
        return bottom - (value - low) / span * (bottom - top)

    step = (right - left) / max(len(values) - 1, 1)
    points = " ".join(
        f"{left + index * step:.1f},{y(value):.1f}"
        for index, value in enumerate(values)
    )
    band = ""
    if not math.isnan(row.take_profit):
        band = (
            f'<rect x="{left}" y="{y(row.take_profit):.1f}" width="{right - left}" '
            f'height="{abs(y(row.stop_loss) - y(row.take_profit)):.1f}" '
            f'fill="{ORANGE}" fill-opacity="0.06"/>'
        )
    body = [
        band,
        _rule(left, y(row.stop_loss), right, y(row.stop_loss), ORANGE, dash="3 3"),
        (
            ""
            if math.isnan(row.take_profit)
            else _rule(
                left,
                y(row.take_profit),
                right,
                y(row.take_profit),
                ORANGE_DIM,
                dash="3 3",
            )
        ),
        (
            ""
            if math.isnan(row.entry_price)
            else _rule(left, y(row.entry_price), right, y(row.entry_price), HAIRLINE)
        ),
        f'<polyline points="{points}" fill="none" stroke="{PAPER}" stroke-width="1.2"/>',
    ]
    return _svg(
        width, height, "".join(body), f"{row.symbol} against its stop and target"
    )


def countdown_svg(
    remaining: float, total: float, width: int = 460, height: int = 44
) -> str:
    """Seconds to the next cycle, as a 1px accent rule that depletes.

    **It moves because time passed, which is the distinction the whole console rests on.**
    The remaining seconds come from the loop's own last write, not from this page's refresh
    timer - a timer would deplete smoothly past a dead loop, which would be an animation of
    something that had stopped happening.
    """
    total = max(total, 1e-9)
    left = max(0.0, min(remaining, total))
    track_from, track_to = 12, width - 48
    filled = (track_to - track_from) * (left / total)
    return _svg(
        width,
        height,
        "".join(
            [
                _text(12, 14, "NEXT CYCLE", DIM),
                _rule(track_from, 28, track_to, 28, RULE),
                _rule(track_from, 28, track_from + filled, 28, ACCENT),
                _text(
                    width - 12, 32, f"{left:.0f}s", TEXT, size=TYPE_BODY, anchor="end"
                ),
            ]
        ),
        "seconds until the next cycle",
    )


# ── staleness: a value that could not be refreshed says so ───────────────────


def staleness_html(age_seconds: float, heartbeat_seconds: float) -> str:
    """The age stamp beside a value the last read could not refresh.

    Empty while the read is current. Beyond two heartbeat intervals the wording escalates
    from *this number is old* to *the loop is not answering*, because those are different
    problems for a reader: the first is a stale panel and the second is a dead system, and
    a dashboard that renders them identically has hidden the one that matters.
    """
    if age_seconds <= 0:
        return ""
    if age_seconds > 2 * heartbeat_seconds:
        return (
            f'<span class="gb-stale">LOOP NOT RESPONDING &nbsp;·&nbsp; LAST READ '
            f"{age_seconds / 60:.0f} MIN AGO</span>"
        )
    return f'<span class="gb-stale">STALE &nbsp;·&nbsp; {age_seconds:.0f}S OLD</span>'


# ── BACKTEST regions ─────────────────────────────────────────────────────────


def arm_rows(
    frame: pd.DataFrame, model: str, channels: str = "C0_base"
) -> pd.DataFrame:
    """The rows of ``frame`` belonging to one arm: one model, one feature set.

    **One selection rule, read by every chart in this section and by every pill beside
    one.** The chart draws these rows and the pill states their range, so a label that
    disagreed with its own picture would need this function to disagree with itself. Before
    GB-63c each chart filtered inline and the pills were computed from the whole frame,
    which is the shape the fold-range defect took.

    Both artefacts carry every arm of the reference condition, so a filter on ``model``
    alone leaves two copies of each curve - one per channel set - and every date is then
    counted twice. Buy-and-hold carries no channel set, having no features, so it survives
    on the null rather than being filtered out with the second copies.
    """
    if "model" not in frame.columns:
        return frame
    arm = frame[frame["model"] == model]
    if "channels" in arm.columns:
        arm = arm[arm["channels"].isna() | (arm["channels"] == channels)]
    return arm


#: The radar's axes. Each is a measured metric with **its own** reference, because these
#: quantities are not commensurable and pretending otherwise is how a composite score gets
#: invented. `higher` says which direction is good, so drawdown can sit beside Sharpe
#: without either being silently negated.
RADAR_AXES = (
    ("DIRECTION", "direction", "direction_reference", True),
    ("SHARPE", "sharpe", None, True),
    ("RETURN", "total_return", None, True),
    ("DRAWDOWN", "max_drawdown", None, False),
    ("CANCELLATION", "cancellation", None, False),
    # Sixth axis, and lower is better: flatness measures how close to zero the forecast
    # sits, so a high value is a model declining to forecast. It earns a place here for
    # the reason 7.3 bans MAE from a table without it - across these arms MAE is close to
    # a monotone function of flatness, so a shape that showed error without flatness would
    # be showing the same axis twice and calling one of them accuracy.
    ("FLATNESS", "flatness", None, False),
)


def radar_axis_values(rows: pd.DataFrame, model: str) -> list[tuple[str, float, str]]:
    """``(label, unit_value, printed)`` per axis for one arm.

    ``unit_value`` is the radius, normalised **against that axis's own reference** and
    clamped to [0, 1]; ``printed`` is the measured number in its own units, which is what
    the card actually labels. There is deliberately no aggregate of these five: the
    reference dashboard shows an "Edge Score" and there is no such quantity in this
    project. Inventing one in a system whose thesis is exact attribution would be the
    opposite of the point.
    """
    arm = arm_rows(rows, model)
    out: list[tuple[str, float, str]] = []
    if arm.empty:
        return out
    for label, column, reference, higher in RADAR_AXES:
        if column not in arm.columns:
            continue
        value = float(arm[column].mean())
        if math.isnan(value):
            out.append((label, 0.0, EM_DASH))
            continue
        if reference and reference in arm.columns:
            base = float(arm[reference].mean())
            unit = 0.5 + (value - base) * 5.0 if not math.isnan(base) else 0.5
            printed = f"{value:.4f} vs {base:.4f}"
        else:
            span = max(abs(float(arm[column].max())), 0.02)
            unit = 0.5 + value / (2 * span)
            printed = f"{value:+.4f}" if abs(value) < 1 else f"{value:+.2f}"
        if not higher:
            unit = 1.0 - unit
        out.append((label, max(0.0, min(1.0, unit)), printed))
    return out


def radar_svg(
    rows: pd.DataFrame, model: str, width: int = 420, height: int = 300
) -> str:
    """A five-axis radar of measured metrics. **No number in the middle.**"""
    axes = radar_axis_values(rows, model)
    if len(axes) < 3:
        return ""
    cx, cy, radius = width / 2, height / 2 - 6, min(width, height) / 2 - 58
    body = []
    for ring in (0.25, 0.5, 0.75, 1.0):
        points = " ".join(
            f"{cx + radius * ring * math.sin(2 * math.pi * i / len(axes)):.1f},"
            f"{cy - radius * ring * math.cos(2 * math.pi * i / len(axes)):.1f}"
            for i in range(len(axes))
        )
        body.append(
            f'<polygon points="{points}" fill="none" stroke="{HAIRLINE}" stroke-width="1"/>'
        )
    shape = []
    for index, (label, unit, printed) in enumerate(axes):
        angle = 2 * math.pi * index / len(axes)
        px = cx + radius * unit * math.sin(angle)
        py = cy - radius * unit * math.cos(angle)
        shape.append(f"{px:.1f},{py:.1f}")
        lx = cx + (radius + 26) * math.sin(angle)
        ly = cy - (radius + 26) * math.cos(angle)
        anchor_at = (
            "middle"
            if abs(math.sin(angle)) < 0.3
            else ("start" if math.sin(angle) > 0 else "end")
        )
        body.append(_text(lx, ly, label, MUTED, anchor=anchor_at))
        body.append(
            _text(lx, ly + RADAR_AXIS_LEADING, printed, PAPER, anchor=anchor_at)
        )
    body.append(
        f'<polygon points="{" ".join(shape)}" fill="{GAIN_FILL}" '
        f'stroke="{GAIN}" stroke-width="1.5"/>'
    )
    return _svg(width, height, "".join(body), f"{model} performance shape")


def cumulative_equity_svg(
    daily: pd.DataFrame, model: str, width: int = 460, height: int = 300
) -> str:
    """Cumulative equity across the folds, as a filled area coloured by its own sign.

    Three signs, not two: an arm that stands aside in every fold chains to an exactly flat
    curve, which is a result rather than a gain.
    """
    if daily.empty or "model" not in daily.columns:
        return ""
    arm = arm_rows(daily, model).sort_values(["fold", "date"])
    if arm.empty:
        return ""
    # Each fold restarts at its own opening capital, so the folds are chained on their
    # returns rather than concatenated on their levels - otherwise every fold boundary
    # would show a jump the strategy never took.
    curve, level = [], 1.0
    for _, fold_rows in arm.groupby("fold", sort=True):
        values = fold_rows["equity"].to_numpy(dtype="float64")
        if len(values) < 2 or values[0] == 0:
            continue
        for value in values:
            curve.append((level * value / values[0], fold_rows["date"].iloc[0]))
        level = curve[-1][0]
    if len(curve) < 2:
        return ""
    series = [point for point, _ in curve]
    left, right, top, bottom = 54, width - 12, 16, height - 30
    # A flat chained curve sits at exactly 1.00, so the baseline rule and the curve are the
    # same line and both land in the middle - which is where a curve that never moved
    # belongs. See :func:`_vertical` for why this is not `max(high - low, 1e-9)`.
    y, _ = _vertical(series, top, bottom, EQUITY_MIN_SPAN)

    step = (right - left) / max(len(series) - 1, 1)
    points = [f"{left + i * step:.1f},{y(v):.1f}" for i, v in enumerate(series)]
    change = series[-1] - 1.0
    # `>= 0` would paint an arm that never traded green. Persistence stands aside in all
    # sixteen folds and its curve is exactly flat, which is a result and not a gain.
    colour, fill = status_colour(change), status_fill(change)
    area = f"{left},{y(1.0):.1f} " + " ".join(points) + f" {right},{y(1.0):.1f}"
    body = [
        f'<polygon points="{area}" fill="{fill}" stroke="none"/>',
        _rule(left, y(1.0), right, y(1.0), HAIRLINE, dash="2 3"),
        _text(left - 6, y(1.0) + 3, "1.00", MUTED, anchor="end"),
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{colour}" stroke-width="1.6"/>',
        _text(left, bottom + 18, f"{len(series)} TRADING DAYS", MUTED),
        _text(
            right,
            top + 4,
            f"{status_glyph(change)} {change * 100:+.2f}%",
            colour,
            size=TYPE_BODY,
            anchor="end",
        ),
    ]
    return _svg(width, height, "".join(body), f"{model} cumulative equity")


def fold_bars_svg(
    rows: pd.DataFrame, model: str, width: int = 460, height: int = 300
) -> str:
    """One bar per fold: green above zero, red below, a rule for a fold that stood aside.

    The third mark is not decoration. A fold that stood aside returns exactly zero, and a
    zero bar has no height - persistence stands aside in all sixteen, so without a mark of
    its own that arm renders as an empty axis.
    """
    arm = arm_rows(rows, model).sort_values("fold")
    if arm.empty:
        return ""
    values = [(int(r.fold), float(r.total_return)) for r in arm.itertuples()]
    values = [(f, v) for f, v in values if not math.isnan(v)]
    if not values:
        return ""
    left, right, top, bottom = 54, width - 12, 20, height - 34
    peak = max(abs(v) for _, v in values) or 1e-9
    zero = top + (bottom - top) / 2
    slot = (right - left) / len(values)
    body = [
        _rule(left, zero, right, zero, HAIRLINE),
        _text(left - 6, zero + 3, "0%", MUTED, anchor="end"),
    ]
    flat = sum(1 for _, value in values if value == 0)
    for index, (fold, value) in enumerate(values):
        magnitude = abs(value) / peak * (bottom - top) / 2
        x = left + index * slot + slot * 0.18
        w = slot * 0.64
        if value == 0:
            # **A fold that stood aside is not a fold that gained nothing.** Its bar has no
            # height, so without a mark of its own it is indistinguishable from a fold that
            # was never measured - and it is not a rare case: persistence stands aside in
            # all sixteen, and FITS in three.
            body.append(_rule(x, zero, x + w, zero, MUTED))
            continue
        y0 = zero - magnitude if value > 0 else zero
        body.append(
            f'<rect x="{x:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{magnitude:.1f}" '
            f'fill="{status_colour(value)}"/>'
        )
        if index % 3 == 0:
            body.append(
                _text(x + w / 2, bottom + 14, f"f{fold}", MUTED, anchor="middle")
            )
    best, worst = max(values, key=lambda v: v[1]), min(values, key=lambda v: v[1])
    # Coloured by what they are rather than by which end they sit at: the best of sixteen
    # flat folds is +0.00%, and printing that in green is the chart asserting a win.
    body.append(
        _text(
            left,
            top - 6,
            f"BEST f{best[0]} {best[1] * 100:+.2f}%",
            status_colour(best[1]),
        )
    )
    body.append(
        _text(
            right,
            top - 6,
            f"WORST f{worst[0]} {worst[1] * 100:+.2f}%",
            status_colour(worst[1]),
            anchor="end",
        )
    )
    if flat:
        body.append(
            _text(
                (left + right) / 2,
                bottom + 28,
                f"{flat} OF {len(values)} FOLDS FLAT - STOOD ASIDE",
                MUTED,
                anchor="middle",
            )
        )
    return _svg(width, height, "".join(body), f"{model} return by fold")


#: Tile and gap in px, largest first. The calendar keeps **one tile per trading day** and
#: shrinks the tile until the whole span fits, rather than dropping the days that do not.
CALENDAR_SCALES = ((13, 3), (9, 2), (6, 2), (4, 1), (3, 1), (2, 1), (2, 0))

#: What the weekday ruler occupies on the left, and the margin kept on the right.
CALENDAR_LEFT, CALENDAR_RIGHT = 76, 12

#: Where the tiles start, and the room the footer needs beneath them. **Two lines, not
#: one.** At 9px the day count and the up/down/flat tally sat on one baseline at opposite
#: ends of a 700-unit chart and cleared each other by 200 units; at the type floor the
#: longer count - "16 OF 190 TRADING DAYS - 174 EARLIER WEEKS NOT SHOWN" - runs past the
#: middle and they collide. Stacking them is the fix that does not shorten a sentence to
#: fit a chart.
CALENDAR_TOP, CALENDAR_FOOT = 26, 46

#: Below this row pitch the three weekday labels collide, and the axis they name is too
#: dense to read a day off anyway, so the band is labelled once instead of three times.
#: It tracks the type: three labels one row apart need a pitch above the height of the
#: type they are set in, which is why raising the floor raised this with it.
CALENDAR_RULER_PITCH = 16


def calendar_scale(weeks: int, width: int) -> tuple[int, int, int]:
    """``(tile, gap, weeks shown)`` for a span of ``weeks``. **Never a silent truncation.**

    The daily artefact holds three folds today and sixteen after the next grid - about 190
    weeks, fifteen times what a fixed 13px tile fits in this width. The first version drew
    every tile at 13px and dropped the ones past the right edge with a bare ``continue``
    while the footer went on counting the days it had not drawn. That is the pill defect
    one level down - a caption describing more data than the picture holds - and unlike the
    pill it would have arrived silently, on the first grid wide enough to trigger it.

    So the tile shrinks first, and only past the smallest tile that still reads as a mark
    does the chart drop anything: the oldest weeks, the recent end being the one a reader
    came for, and :func:`calendar_svg` then says how many weeks went with them.
    """
    span = max(width - CALENDAR_LEFT - CALENDAR_RIGHT, 1)
    for tile, gap in CALENDAR_SCALES:
        if weeks * (tile + gap) <= span:
            return tile, gap, weeks
    tile, gap = CALENDAR_SCALES[-1]
    return tile, gap, max(span // (tile + gap), 1)


def calendar_svg(daily: pd.DataFrame, model: str, width: int = 700) -> str:
    """A heatmap of daily equity change, one tile per trading day, by calendar week.

    **The height is derived rather than fixed**, because five weekday rows at the fitted
    tile size is the whole of what this chart is tall. A fixed 280 left the three-fold span
    ending at y=103 with 150px of nothing beneath it, which is the flat-equity lesson
    again: the case the chart is usually in was the one that looked broken.
    """
    if daily.empty or "model" not in daily.columns:
        return ""
    arm = arm_rows(daily, model).sort_values(["fold", "date"])
    if arm.empty:
        return ""
    changes: dict[pd.Timestamp, float] = {}
    for _, fold_rows in arm.groupby("fold", sort=True):
        values = fold_rows["equity"].to_numpy(dtype="float64")
        dates = list(fold_rows["date"])
        for i in range(1, len(values)):
            if values[i - 1]:
                changes[dates[i]] = values[i] / values[i - 1] - 1.0
    if not changes:
        return ""
    days = sorted(changes)
    peak = max(abs(v) for v in changes.values()) or 1e-9
    first = days[0] - pd.Timedelta(days=int(days[0].dayofweek))
    weeks = int((days[-1] - first).days // 7) + 1
    tile, gap, shown = calendar_scale(weeks, width)
    dropped, pitch = weeks - shown, tile + gap
    height = CALENDAR_TOP + 5 * pitch + CALENDAR_FOOT
    drawn = [day for day in days if int((day - first).days // 7) >= dropped]
    body = []
    for day in drawn:
        value = changes[day]
        week = int((day - first).days // 7) - dropped
        opacity = 0.18 + 0.82 * min(abs(value) / peak, 1.0)
        body.append(
            f'<rect x="{CALENDAR_LEFT + week * pitch}" '
            f'y="{CALENDAR_TOP + int(day.dayofweek) * pitch}" '
            f'width="{tile}" height="{tile}" rx="2" '
            f'fill="{status_colour(value)}" fill-opacity="{opacity:.2f}"/>'
        )
    if pitch >= CALENDAR_RULER_PITCH:
        for index, label in enumerate(("MON", "", "WED", "", "FRI")):
            if label:
                body.append(
                    _text(
                        CALENDAR_LEFT - 6,
                        CALENDAR_TOP + index * pitch + tile - 3,
                        label,
                        MUTED,
                        anchor="end",
                    )
                )
    else:
        body.append(
            _text(
                CALENDAR_LEFT - 6,
                CALENDAR_TOP + 2 * pitch + tile / 2 + 3,
                "MON-FRI",
                MUTED,
                anchor="end",
            )
        )
    up = sum(1 for day in drawn if changes[day] > 0)
    down = sum(1 for day in drawn if changes[day] < 0)
    # **The caption counts the tiles, not the rows.** They are the same number until the
    # span outgrows the width, and the case where they are not is the whole reason the
    # sentence is built rather than written.
    counted = (
        f"{len(drawn)} TRADING DAYS"
        if not dropped
        else f"{len(drawn)} OF {len(days)} TRADING DAYS - "
        f"{dropped} EARLIER WEEKS NOT SHOWN"
    )
    body.append(_text(CALENDAR_LEFT, height - 30, counted, MUTED))
    body.append(
        _text(
            CALENDAR_LEFT,
            height - 10,
            f"{up} UP / {down} DOWN / {len(drawn) - up - down} FLAT",
            MUTED,
        )
    )
    return _svg(width, height, "".join(body), f"{model} daily results calendar")


# ── the four measured regions, one frame each ────────────────────────────────
#
# **Each takes exactly one frame, and that is the mechanism.** A region cannot label itself
# with a sibling's range because no sibling's frame is in scope: the pill is built from the
# rows the chart draws, inside the function that draws them. Naming the locals
# `from_results` and `from_daily` in `main` was the previous attempt, and a convention a
# reader has to honour is a note - `test_a_region_has_one_frame_in_scope` pins the
# signatures so a second frame cannot be threaded back in without the test saying so.

#: What a region says when its artefact holds nothing to draw. Both name the file, built
#: from the same constant the loader defaults to, so neither can outlive a rename.
NO_REFERENCE_ROWS = f"no reference-condition rows in {RESULTS_PATH}"
NO_DAILY = f"{DAILY_EQUITY_PATH} not generated - run the study grid"


def shape_region(reference: pd.DataFrame, model: str) -> str:
    """The radar. Reads `results.csv`."""
    arm = arm_rows(reference, model)
    source = backtest_source(RESULTS_PATH, arm)
    return region(
        f"{model.upper()} performance shape",
        source,
        radar_svg(arm, model) or too_little(source, NO_REFERENCE_ROWS),
        "Six measured axes, each against its own reference. No composite score.",
    )


def cumulative_region(daily: pd.DataFrame, model: str) -> str:
    """The equity area. Reads `report/daily_equity.csv`."""
    arm = arm_rows(daily, model)
    source = backtest_source(DAILY_EQUITY_PATH, arm)
    return region(
        "Cumulative equity",
        source,
        cumulative_equity_svg(arm, model) or too_little(source, NO_DAILY),
        "Folds chained on returns, not concatenated on levels.",
    )


def fold_region(reference: pd.DataFrame, model: str) -> str:
    """The per-fold bars. Reads `results.csv`."""
    arm = arm_rows(reference, model)
    source = backtest_source(RESULTS_PATH, arm)
    return region(
        "Return by fold",
        source,
        fold_bars_svg(arm, model) or too_little(source, NO_REFERENCE_ROWS),
    )


def calendar_region(daily: pd.DataFrame, model: str) -> str:
    """The daily heatmap. Reads `report/daily_equity.csv`."""
    arm = arm_rows(daily, model)
    source = backtest_source(DAILY_EQUITY_PATH, arm)
    return region(
        "Daily results",
        source,
        calendar_svg(arm, model) or too_little(source, NO_DAILY),
    )


# ── GB-35: the forecast path ─────────────────────────────────────────────────


def forecast_svg(
    history: pd.Series,
    path: pd.Series,
    thresholds: Thresholds,
    symbol: str,
    width: int = WIDE_CHART,
    height: int = 260,
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
    left, right, top = 84, 14, 26
    strip = 34
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
        f'<polyline points="{history_points}" fill="none" stroke="{DIM}" '
        'stroke-width="1.5"/>'
    )

    joined = [float(history.iloc[-1]), *[float(v) for v in path]]
    offset = len(history) - 1
    path_points = " ".join(
        f"{x_at(offset + i):.1f},{y_at(v):.1f}" for i, v in enumerate(joined)
    )
    # **The ramp used to draw this chart, and it had no business here.** It encodes which
    # channel and which band; a price line and a forecast are neither. The forecast is a
    # direction, so it takes the status colour of the move it predicts, and the history is
    # context, so it takes DIM. Found by looking at a screenshot - the boundary test
    # checked the radar and the fold chart and never looked at this one, which is a guard
    # whose substrate was not the thing being claimed.
    direction = joined[-1] - joined[0]
    body.append(
        f'<polyline points="{path_points}" fill="none" '
        f'stroke="{status_colour(direction)}" stroke-width="2" stroke-dasharray="4 3"/>'
    )
    body.append(
        f'<circle cx="{x_at(len(values) - 1):.1f}" cy="{y_at(joined[-1]):.1f}" r="3" '
        f'fill="{status_colour(direction)}"/>'
    )
    body.append(_rule(x_at(offset), top, x_at(offset), top + plot_h, ORANGE_DIM, "2 3"))

    # The strip. A hairline, then the annotations, none of them over the plot.
    baseline = top + plot_h
    body.append(_rule(0, baseline + 10, width, baseline + 10, HAIRLINE, "2 4"))
    note_y = baseline + 24
    # NOW sits under the rule it names rather than at the edge of the strip: a marker whose
    # label is a screen away from it is a label for something else.
    #
    # **It is ORANGE and not ORANGE_DIM, and that was not a preference.** `ORANGE_DIM` is
    # `RULE`, which the palette classifies as structure rather than text and excludes from
    # the contrast assertion for that reason - so this label was the one piece of text on
    # the page at 1.27:1, effectively invisible, and invisible in a way no test could see
    # because the colour was being checked as the rule it is named for. The dashed rule
    # above still takes `ORANGE_DIM`; a label is not a rule, and matching the two was the
    # mistake. The sibling annotation on this same baseline has always been `ORANGE`.
    body.append(_text(x_at(offset), note_y, "NOW", ORANGE, anchor="middle"))

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


def forecast_source(as_of: pd.Timestamp) -> Source:
    """The forecast card's pill. **It states the SOURCE, and it cannot state freshness.**

    `LIVE` is one of three provenances and always has been - the type exists to stop
    backtest numbers wearing a live face, and no test has ever asked it about recency. But
    a green pill beside a bare date is read as *as of*, and on 3 Sep 2026 that misreading
    cost a reader who knows the system: the loop had correctly dropped the in-progress bar,
    the panel said `LIVE 2026-09-02` while the clock said the 3rd, and a working
    look-ahead guard was indistinguishable from a dead feed. So the detail now names the
    axis - *live feed* - and *bar* qualifies the date as a bar rather than a timestamp.

    **The signature is the mechanism.** This takes the bar and nothing else: no clock, no
    session state. A pill that cannot see the time cannot make a claim about it, which is
    a stronger guarantee than a convention that it should not. Freshness is stated twice
    already, by :func:`staleness_html` and by `LAST CYCLE` in :func:`session_strip`; a
    third copy here would be a second place for one fact to live and diverge from.
    """
    return Source(LIVE, f"live feed · {as_of:%Y-%m-%d} bar")


def bar_note(as_of: pd.Timestamp, now: pd.Timestamp, session_open: bool) -> str:
    """The forecast card's caption: which bar is drawn, and why today is not it.

    The mandated string comes first and unchanged - `GLASSBOX_PHASE2_EXPANSION.md` requires
    the chart to carry `Last completed bar <date> · advances once per trading day`, so the
    reason is appended to it and never replaces it.

    **What the appended clause buys.** The panel stated the fact and withheld the reason: a
    reader saw yesterday's date during an open market and had no way to tell a working
    look-ahead guard from staleness. The data was right and the page was unreadable, which
    is the harder defect of the two because nothing is wrong to find. Outside market hours
    the old wording is complete and nothing is added - the explanation is only owed while
    the absence needs explaining.

    Plain text, never markup: :func:`region` escapes what it renders, and a value object
    that returns markup only works while every caller agrees not to treat it as data.
    """
    note = f"Last completed bar {as_of:%Y-%m-%d} · advances once per trading day"
    if session_open and as_of.date() < now.date():
        note += (
            f" · today's session is open, so {now:%Y-%m-%d} has no completed bar yet"
        )
    return note


# ── GB-36: the contribution bars ─────────────────────────────────────────────


def contributions_svg(
    attribution: Attribution, width: int = WIDE_CHART, row_height: int = 24
) -> str:
    """Per-channel contributions as a diverging bar chart around a zero rule.

    Shares come from :func:`explain.channel.shares` — the same function the narrative
    reads, so the number in the prose and the number on the bar cannot disagree. Positive
    is filled and to the right, negative is hollow and to the left; the ramp encodes which
    channel rather than which sign, per the module docstring.
    """
    ranked = sorted(shares(attribution).items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    height = 34 + row_height * len(ranked) + 30
    left, right = 150, 100
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
    attribution: Attribution, width: int = WIDE_CHART, row_height: int = 22
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
    height = 34 + row_height * len(ranked) + 50
    left, right = 230, 100
    plot_w = width - left - right
    middle = left + plot_w / 2
    widest = max((abs(value) for _, value in ranked), default=0.0) or 1.0
    periods = list(view)

    body = [
        _text(12, 16, "PER-FREQUENCY CONTRIBUTION", ORANGE),
        _text(width - 12, 16, "SHARE OF GROSS VIEW", MUTED, anchor="end"),
        _rule(middle, 26, middle, height - 46, ORANGE_DIM),
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
        body.append(_text(12, height - 30, note, ORANGE))
    body.append(
        _text(
            12,
            height - 10,
            "BIN 0 MULTIPLIES ZERO — RIN REMOVES THE WINDOW MEAN BEFORE THE TRANSFORM",
            MUTED,
        )
    )
    return _svg(width, height, "".join(body), "per-frequency contributions")


def gain_phase_svg(
    attribution: Attribution,
    top: int = 6,
    width: int = WIDE_CHART,
    row_height: int = 22,
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
        length = abs(gain) / widest * (width - 230 - 300)
        body.append(
            f'<rect x="230" y="{y:.1f}" width="{max(length, 0.6):.1f}" '
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
    width: int = WIDE_CHART,
    height: int = 288,
) -> str:
    """The learned frequency response, **with what it was measured to be**.

    **The caption is the point of this chart, not decoration.** A curve rendered without
    it invites "this is what the model learned about the market", and GB-48 measured that
    it is mostly not: a model trained on white noise reproduces it at r = +0.9485, about
    86% of the response. Rendering the curve and letting a reader infer market structure
    would be the black-box behaviour this project opposes, committed by the explanation
    layer - the worst place for it to happen.
    """
    left, right, top, floor = 60, 24, 34, height - 68
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
        body.append(_text(x, floor + 18, f"{period:.0f}D", MUTED, anchor="middle"))

    body.append(
        _text(
            12,
            height - 28,
            f"DOMINATED BY INTERPOLATION COST, NOT BY MARKET STRUCTURE — "
            f"{SPECTRAL_GEOMETRY_SHARE * 100:.0f}% OF THIS CURVE IS REPRODUCED BY A MODEL "
            f"TRAINED ON WHITE NOISE",
            ORANGE,
        )
    )
    body.append(
        _text(
            12,
            height - 10,
            f"MEASURED GB-48: GAIN CURVES CORRELATE r = +{SPECTRAL_NOISE_CORRELATION:.4f} "
            "ACROSS 48 MODELS AT THREE FOLD-GRID ANCHORS",
            MUTED,
        )
    )
    return _svg(width, height, "".join(body), "learned frequency response")


def spectral_panel(
    attribution: Attribution, response=None, width: int = WIDE_CHART
) -> str:
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


def decision_detail(record: DecisionRecord) -> str:
    """The DETAIL cell for one decision row: what it carried, then where it came from.

    **Provenance alone was the least informative thing available, and it made an order and
    no order render the same cell.** On 3 September 2026 seven EXIT decisions were written
    against an empty book; each rendered as a red ``gb-row-loss`` reading ``EXIT`` beside
    ``LIVE``, indistinguishable from a row where a position was actually closed. The record
    holds ``order``, :func:`decision_table` has shown it as ``YES``/em dash since GB-34, and
    the table the console actually renders was throwing it away.

    **It says whether the decision carried an order. It does not say whether a position was
    closed, and it must not be read that way.** ``order`` is populated from ``sized``, which
    ``rank_signals`` fills with ``enter_long`` candidates only, so an EXIT record carries
    ``None`` *whether or not* the symbol was held - the closing order is built later, by
    ``live_loop._send_exits``, and never enters the record. Distinguishing a real close from
    one that closed nothing needs the book as it stood at decision time, which
    :class:`DecisionRecord` does not carry and cannot be given here: it is frozen in §4 of
    the spec. So this cell states the fact the record actually holds and stops short of the
    one it does not, which is why it reads ``ORDER -`` rather than anything like "closed
    nothing".
    """
    carried = "YES" if record.order else EM_DASH
    return f"ORDER {carried} · {provenance_label(record.provenance)}"


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
    raw: Sequence[int] = (),
    row_classes: Sequence[str] = (),
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
        row_classes: One CSS class per row, carrying the 2px left border that states what
            the system did. **An accent on a fact the row already states in words** - the
            WHAT column says ENTER_LONG, EXIT or HOLD - so a reader who cannot resolve the
            border colour loses nothing. Empty for tables that have no action to state.
        raw: Indices of columns whose cells are **already HTML built by this module** -
            in practice the one column that carries :func:`status_html`. Escaping stays
            the default for every other cell and for every value that came from a file,
            a broker or a record; this exists so that the status colour can reach a cell
            at all, and `test_only_module_built_html_reaches_a_raw_column` pins the fact
            that nothing else uses it.
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
            shown = str(cell) if index in set(raw) else escape(cell)
            cells.append(f'<td class="{style.strip()}">{shown}</td>')
        klass = row_classes[number] if number < len(row_classes) else ""
        opening = f'<tr class="gb-row {klass}">' if klass else "<tr>"
        body.append(f"{opening}{''.join(cells)}</tr>")
    return (
        f'<table class="gb-table"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
    )


def stop_note(row: PositionRow) -> str:
    """What a reader needs to know about a position near its stop, in words.

    Only at :data:`CLOSE`. Emphasis alone says *look here* and leaves the reader to infer
    what happens next; the point of the innermost band is that the consequence is spelled
    out while there is still time to act on it. Below the stop the wording changes again,
    because "approaching" is the wrong tense for a level already crossed.
    """
    room = row.stop_room
    if row.stop_proximity != CLOSE or math.isnan(room):
        return ""
    if room <= 0:
        return (
            f"{row.symbol} IS AT OR THROUGH ITS STOP {row.stop_loss:,.2f}. THE STOP IS A "
            "MARKET ORDER AT THE BROKER; WHEN IT FILLS THE POSITION IS CLOSED AND THE "
            "LOOP RECONCILES IT ON THE NEXT CYCLE"
        )
    return (
        f"{row.symbol} HAS SPENT {(1 - room) * 100:.0f}% OF ITS STOP DISTANCE "
        f"({row.price:,.2f} AGAINST {row.stop_loss:,.2f}). A STOP FILL IS NEAR: IT SELLS "
        "THE WHOLE POSITION AT MARKET AND THE LOOP BOOKS THE EXIT ON THE NEXT CYCLE"
    )


def position_table(rows: Sequence[PositionRow]) -> str:
    """The positions panel. A quarantined holding shows em dashes, never zeros.

    Two columns carry status. `UNREALISED` is coloured through :func:`status_html`, so it
    arrives with its glyph and its sign attached and reads in greyscale. `STOP ROOM` is
    *not* coloured: it is a distance rather than a direction, and colouring it would be
    the second data family the palette rule refuses. It is marked instead.
    """
    marks = [
        (number, 6)
        for number, row in enumerate(rows)
        if row.stop_proximity in (APPROACHING, CLOSE)
    ]
    return table_html(
        ("SYMBOL", "QTY", "ENTRY", "LAST", "VALUE", "UNREALISED", "STOP ROOM", "STATE"),
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
                    else status_html(
                        row.unrealised,
                        f"{row.unrealised:,.2f} ({row.unrealised_pct:+.2f}%)",
                    )
                ),
                (
                    EM_DASH
                    if math.isnan(row.stop_room)
                    else f"{row.stop_room * 100:.0f}% TO {row.stop_loss:,.2f}"
                ),
                "MANAGED" if row.managed else "QUARANTINED",
            )
            for row in rows
        ],
        numeric=(1, 2, 3, 4, 5),
        flagged=marks,
        raw=(5,),
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


def cycle_age(root: str | Path, now: pd.Timestamp) -> float:
    """Seconds since the live loop last persisted its book, or ``inf``.

    **Measured from the loop's own write, not from the dashboard's clock.** Step 10 of
    every cycle persists the book, so this file's mtime is the last time a cycle actually
    completed. A countdown driven by the panel's own refresh timer would tick smoothly
    while the loop lay dead, which is the exact failure the heartbeat exists to make
    visible - it would be an animation, not a measurement.
    """
    path = Path(root) / "book.json"
    if not path.is_file():
        return math.inf
    written = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
    return max((now - written).total_seconds(), 0.0)


def newly_written(
    decisions: Sequence[DecisionRecord], seen: set[str]
) -> tuple[set[str], set[str]]:
    """``(ids_to_mark, ids_now_seen)`` for one refresh.

    Marked for exactly one pass and then never again: a badge that persisted would stop
    meaning *this arrived while you were looking* and start meaning *this is recent*,
    which the timestamp already says. The caller keeps `seen` across refreshes; on the
    very first render nothing is marked, because everything is new and marking all of it
    would say nothing.
    """
    # Keyed on `(as_of, symbol)`, which is what identifies a decision in the contract -
    # `DecisionRecord` carries no `decision_id`, and the loop derives one from exactly
    # this pair. An earlier version of this function read `record.decision_id` and was
    # tested against a stub that had one; the real type does not, and the page crashed on
    # first render. A double with a field the real thing lacks is the same defect class as
    # a broker double that permits what the real broker refuses.
    current = {(record.as_of, record.symbol) for record in decisions}
    return (set() if not seen else current - seen), current


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


#: The seed. Fixed, and the whole design rests on it - see :func:`starfield`.
STAR_SEED = 1337
STAR_COUNT = 48


def starfield(seed: int = STAR_SEED, count: int = STAR_COUNT) -> str:
    """A fixed field of faint points behind the content.

    **It cannot be injected once, and the seed is why that does not matter.** Streamlit
    re-executes the script on every rerun and rebuilds the DOM; there is no primitive that
    survives that. What makes re-injection *equivalent* to injecting once is determinism:
    the same points at the same coordinates with the same animation parameters every
    render, so the browser is handed an identical subtree. A field that reshuffled would
    be an animation, and this project does not animate things that have not changed.

    **The negative delay is the part that is easy to get wrong.** A CSS animation restarts
    when its node is replaced, so without it every rerun would reset each star to the start
    of its drift - the field would be deterministic in *position* and non-deterministic in
    *phase*, which is a reshuffle by another name. Each star's delay is drawn from the same
    seed, so the phase at injection is a function of the seed rather than of when the page
    loaded. Between reruns the field drifts; at a rerun it returns to the same phase.
    Claiming smooth continuous drift would be claiming something Streamlit cannot deliver.

    At 1.57:1 and 1.94:1 against the ground none of this is visible, which is the intent:
    if a reader notices a star while reading a number, it is too strong.
    """
    points = random.Random(seed)
    dots = []
    for index in range(count):
        left = points.uniform(0, 100)
        top = points.uniform(0, 100)
        colour = STAR_B if points.random() < 0.4 else STAR_A
        drift = points.uniform(10, 16)
        duration = points.uniform(25, 40)
        delay = -points.uniform(0, duration)
        dots.append(
            f'<i style="left:{left:.2f}%;top:{top:.2f}%;background:{colour};'
            f"--drift:{drift:.1f}px;animation-duration:{duration:.1f}s;"
            f'animation-delay:{delay:.1f}s"></i>'
        )
    return f'<div class="gb-stars" aria-hidden="true">{"".join(dots)}</div>'


def stylesheet() -> str:
    """The whole visual language: three rule weights, one family, and space.

    **No fills anywhere.** No panel backgrounds, no borders on four sides, no rounded
    containers, no shadows. A region that needs to feel distinct gets more space, not a
    box. The entire chrome vocabulary is a 2px accent rule above a section label, a 1px
    rule between sub-regions, a 1px faint rule between table rows, and whitespace.

    **Tabular figures are a requirement, not a preference.** In a dense monospace table a
    reader compares magnitudes by scanning a column, and proportional digits break that at
    the one place the design is asking them to do it.

    **Every size, leading and tracking here is a token, and that is load-bearing rather
    than tidy.** This stylesheet set type at five sizes and eight tracking values, and the
    charts set four more at their own call sites - eight distinct sizes in all, no two of
    which were ever compared, because nothing in the codebase held them in one place. Five
    of the eight were under 12px on a near-black ground. They are now the four steps of
    :data:`~glassbox.dashboard.tokens.TYPE_STEPS`, one tracking value for uppercase chrome
    and one for the single large figure, and
    ``test_every_size_in_the_stylesheet_is_a_step_of_the_scale`` reads them back out of the
    rendered CSS - out of what the browser receives, not out of this file, because a guard
    that counts what a source file contains is broken by the next person who explains a
    rule in a comment.

    **The base states a line-height.** It did not, so the leading of every table row, label
    and caption on the page was whatever Streamlit's theme supplied - a typographic
    decision taken by a dependency, correct only by luck, and silently revisable by a
    version bump.
    """
    return f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');

  .stApp {{ background: {GROUND}; }}
  html, body, [class*="css"], .stMarkdown, p, div, span, td, th {{
      font-family: {MONO};
      color: {TEXT};
      font-size: {TYPE_BODY}px;
      line-height: {LEADING_UI};
      font-variant-numeric: tabular-nums;
      -webkit-font-smoothing: antialiased;
  }}
  .block-container {{ padding-top: 1.4rem; padding-bottom: 4rem; max-width: 1800px; }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  /* A chart is never scaled below its own viewBox, because that is the only way a label
     inside one has a size floor at all - see `_svg`. Past that width the container
     scrolls; the type does not shrink. */
  [data-testid="stMarkdownContainer"] {{ overflow-x: auto; }}

  /* ── the starfield ───────────────────────────────────────────────────── */
  .gb-stars {{
      position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden;
  }}
  .gb-stars i {{
      position: absolute; width: 1px; height: 1px; border-radius: 50%;
      animation-name: gb-drift; animation-timing-function: linear;
      animation-iteration-count: infinite; animation-direction: alternate;
  }}
  @keyframes gb-drift {{
      from {{ transform: translate(0, 0); }}
      to   {{ transform: translate(var(--drift), calc(var(--drift) * -0.6)); }}
  }}
  @media (prefers-reduced-motion: reduce) {{
      /* The field stays; the drift stops. Removing the points as well would take away
         something a reader may be relying on for depth, to solve a problem they asked
         about motion. */
      .gb-stars i {{ animation: none; }}
  }}
  [data-testid="stAppViewContainer"] {{ position: relative; z-index: 1; }}

  /* ── the rule vocabulary: three weights and nothing else ─────────────── */
  .gb-region {{ margin: 2.1rem 0 0; }}
  .gb-region-rule {{ border-top: 2px solid {ACCENT}; }}
  .gb-region-head {{
      display: flex; align-items: baseline; justify-content: space-between;
      gap: .8rem; padding: .5rem 0 .7rem;
  }}
  .gb-label {{
      font-size: {TYPE_LABEL}px; letter-spacing: {TRACK_LABEL}em;
      text-transform: uppercase; color: {DIM};
  }}
  .gb-sub-rule {{ border-top: 1px solid {RULE}; margin: .9rem 0; }}
  /* A region note is a sentence, not a label: no tracking, and prose leading. It was
     set at 9px with .1em of it, which is a caption styled like a chip. */
  .gb-note {{
      font-size: {TYPE_LABEL}px; line-height: {LEADING_UI}; color: {DIM};
      padding-top: .6rem;
  }}

  /* ── type scale ──────────────────────────────────────────────────────── */
  .gb-body, .gb-meta {{ font-size: {TYPE_BODY}px; color: {TEXT}; }}
  .gb-meta {{ color: {DIM}; }}
  .gb-emph {{ font-size: {TYPE_PROSE}px; }}
  .gb-figure {{
      font-size: {TYPE_FIGURE}px; font-weight: 500;
      letter-spacing: {TRACK_FIGURE}em; line-height: {LEADING_FIGURE};
  }}

  /* ── the source pill ─────────────────────────────────────────────────── */
  .gb-pill {{
      border: 1px solid; padding: .1rem .45rem; font-size: {TYPE_LABEL}px;
      letter-spacing: {TRACK_LABEL}em; white-space: nowrap; border-radius: 0;
  }}
  /* Inherits the pill's tracking rather than restating it: the detail is the same run of
     text at the same size, and a second value here could only ever drift from the first. */
  .gb-pill-detail {{ color: {DIM}; margin-left: .4rem; }}

  /* ── tables and the row language ─────────────────────────────────────── */
  .gb-table {{ width: 100%; border-collapse: collapse; font-size: {TYPE_BODY}px; }}
  .gb-table th {{
      text-align: left; font-size: {TYPE_LABEL}px; letter-spacing: {TRACK_LABEL}em;
      text-transform: uppercase;
      color: {DIM}; font-weight: 400; padding: .3rem .6rem .5rem;
      border-bottom: 1px solid {RULE};
  }}
  .gb-table td {{
      padding: .4rem .6rem; border-bottom: 1px solid {RULE_FAINT}; color: {TEXT};
  }}
  .gb-table tr:last-child td {{ border-bottom: none; }}
  .gb-table td.num, .gb-table th.num {{ text-align: right; }}
  /* Vermillion marks the symbol column and selection - never a value. */
  .gb-table td.gb-flag {{ color: {ACCENT}; }}

  /* Every row states what the system did, as a 2px left border. It is an ACCENT on a
     fact the WHAT column already carries in words: a reader who cannot see the border
     loses nothing, which is what `test_action_is_recoverable_from_the_row_text` holds.
     border-radius stays 0 - a single-sided border with rounded corners renders as a
     defect rather than as a choice. */
  .gb-table tr.gb-row td:first-child {{ padding-left: 8px; border-radius: 0; }}
  .gb-row-gain td:first-child {{ border-left: 2px solid {GAIN}; }}
  .gb-row-loss td:first-child {{ border-left: 2px solid {LOSS}; }}
  .gb-row-hold td:first-child {{ border-left: 2px solid {RULE}; }}
  .gb-row-hold td {{ color: {DIM}; }}
  .gb-row-accent td:first-child {{ border-left: 2px solid {ACCENT}; }}

  /* ── staleness ───────────────────────────────────────────────────────── */
  .gb-stale {{
      color: {DIM}; font-size: {TYPE_LABEL}px; letter-spacing: {TRACK_LABEL}em;
  }}
  .gb-not-responding {{
      color: {LOSS}; font-size: {TYPE_LABEL}px; letter-spacing: {TRACK_LABEL}em;
  }}
  /* A sentence about what a region does not have. Prose leading, no tracking. */
  .gb-empty {{
      color: {DIM}; font-size: {TYPE_BODY}px; line-height: {LEADING_UI};
      padding: 1.6rem 0;
  }}

  /* ── the status strip ────────────────────────────────────────────────── */
  .gb-strip {{ display: flex; flex-wrap: wrap; gap: 20px; align-items: baseline; }}
  .gb-stat-key {{
      font-size: {TYPE_LABEL}px; letter-spacing: {TRACK_LABEL}em;
      text-transform: uppercase; color: {DIM}; margin-right: .4rem;
  }}
  .gb-stat-value {{ font-size: {TYPE_BODY}px; color: {TEXT}; }}

  /* ── the narrative keeps its own stack and its RTL rule ──────────────── */
  .gb-narrative {{
      font-family: {HEBREW_SANS}; font-size: {TYPE_PROSE}px;
      line-height: {LEADING_PROSE}; color: {TEXT};
      border-left: 2px solid {ACCENT}; padding: .2rem 0 .2rem .8rem;
  }}
  .gb-narrative[dir="rtl"] {{
      border-left: none; border-right: 2px solid {ACCENT};
      padding: .2rem .8rem .2rem 0;
  }}

  /* ── expanders ───────────────────────────────────────────────────────── */
  [data-testid="stExpander"] details {{
      background: transparent; border: none; border-top: 1px solid {RULE_FAINT};
      border-radius: 0;
  }}
  [data-testid="stExpander"] summary {{ color: {TEXT}; font-size: {TYPE_BODY}px; }}
  [data-testid="stExpander"] summary:hover {{ color: {ACCENT}; }}
  [data-testid="stExpander"] summary svg {{ fill: {DIM}; }}
  [data-testid="stExpander"] details > div {{ background: transparent; border: none; }}

  /* ── buttons ─────────────────────────────────────────────────────────── */
  .stButton > button {{
      background: transparent; color: {TEXT}; border: 1px solid {RULE};
      border-radius: 0; font-family: {MONO}; font-size: {TYPE_BODY}px;
      letter-spacing: {TRACK_LABEL}em;
      width: 100%;
  }}
  .stButton > button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}

  /* **The only two filled elements on the entire page.** Everything else sits on the
     ground with rules and space; these two are solid because this is the one place a
     click moves money, and the weight is the warning. Ground-coloured text on a gain or
     loss field, so the fill is unmistakably the control rather than a status. */
  .gb-approve button {{
      background: {GAIN} !important; color: {GROUND} !important;
      border-color: {GAIN} !important; font-weight: 500;
  }}
  .gb-reject button {{
      background: {LOSS} !important; color: {GROUND} !important;
      border-color: {LOSS} !important; font-weight: 500;
  }}
  .gb-approve button:hover, .gb-reject button:hover {{
      color: {GROUND} !important; filter: brightness(1.12);
  }}
</style>
"""


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
        """One line of **plain text**, for whatever is showing the band.

        It carried literal &nbsp; until 28 Aug 2026, because the masthead that first
        used it inserted the string unescaped. Any caller that escaped it - as a card
        properly does - printed the entities on the face of the panel. A value object
        returning markup is presentation smuggled into data, and it only works while every
        caller agrees not to treat it as data.
        """
        if self.stood_aside or self.val_sharpe is None:
            return "STOOD ASIDE — VALIDATION FOUND NO CANDIDATE WITH A POSITIVE SHARPE"
        return (
            f"FOLD {self.fold}  ·  VAL SHARPE {self.val_sharpe:+.3f}"
            f"  ·  OVER {self.val_trades} TRADES  ·  GRID MAXIMUM OF 15 CANDIDATES"
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


# ── LIVE cards ───────────────────────────────────────────────────────────────


def _stat(key: str, value: str, dim: bool = False) -> str:
    colour = f' style="color:{DIM}"' if dim else ""
    return (
        f'<span><span class="gb-stat-key">{escape(key)}</span>'
        f'<span class="gb-stat-value"{colour}>{value}</span></span>'
    )


def session_strip(
    cfg: Config,
    state: str,
    state_dir: str | Path,
    source: str,
    cycle_seconds: float,
    stale: str = "",
) -> str:
    """Row 1: what is happening now, one line, 20px gaps.

    **The bound directory is here and not behind a control.** On 27 Aug the panel was
    pointed at `checkpoints/live` while the rehearsal wrote to `checkpoints/rehearsal`; its
    Co-Pilot queue rendered empty and correct, and criterion 2 had to be satisfied through
    the API. An empty queue and no recommendations are indistinguishable unless the panel
    says which directory it read.
    """
    colour = GAIN if state in (RUNNING, IDLE) else DIM if state == CLOSED else ACCENT
    age = (
        EM_DASH
        if math.isinf(cycle_seconds)
        else (
            f"{cycle_seconds:.0f}s ago"
            if cycle_seconds < 3600
            else f"{cycle_seconds / 3600:.1f}h ago"
        )
    )
    return (
        '<div class="gb-strip">' + f'<span><span class="gb-stat-key">SESSION</span>'
        f'<span class="gb-stat-value" style="color:{colour}">{escape(state)}</span></span>'
        + _stat("LAST CYCLE", escape(age))
        + _stat("MODEL", escape(cfg.model.active.upper()), dim=True)
        + _stat("CHANNELS", escape(cfg.channels.active.upper()), dim=True)
        + _stat("UNIVERSE", f"{len(cfg.universe)}", dim=True)
        + _stat("BOUND", escape(str(state_dir)), dim=True)
        + _stat("SHOWING", escape(source.upper()), dim=True)
        + _stat("CONFIG", escape(config_hash(cfg)[:10].upper()), dim=True)
        + (f"<span>{stale}</span>" if stale else "")
        + "</div>"
    )


def reliability_body(reliability: Reliability | None, band: BandContext | None) -> str:
    """The decider's track record, large.

    **Not optional and not small.** A console that shows P&L while hiding how often the
    decider is right is the black box this project exists to oppose - and this one's answer
    is that it does not beat the always-long bar, which is the number a reader is least
    likely to go looking for and most needs to see.

    The band sits beneath it because the two answer the same question from opposite ends:
    the record says how good the forecast was, the band says how selective the system is
    about acting on it, and a reader who sees only one of them has half the picture.
    """
    if reliability is None:
        return too_little(
            Source(BACKTEST, "not measured"),
            "NOT MEASURED - run smoke_offline --prepare-live",
        )
    gap = reliability.gap
    colour = GAIN if gap > 0 else LOSS
    beats = "BEATS" if gap > 0 else "BELOW"
    out = [
        (
            f'<div class="gb-figure" style="color:{colour}">{status_glyph(gap)} '
            f"{gap:+.4f}</div>"
        ),
        (
            f'<div class="gb-meta" style="padding-top:.5rem">DIRECTION '
            f"{reliability.direction:.4f} &nbsp; {escape(beats)} THE ALWAYS-LONG BAR "
            f"{reliability.always_long:.4f}</div>"
        ),
        (
            f'<div class="gb-meta">OVER {reliability.folds} FOLDS &nbsp; MEASURED '
            f"{escape(reliability.measured_on)} &nbsp; "
            f"{escape(reliability.model.upper())}</div>"
        ),
    ]
    if band is not None:
        fires = "STANDS ASIDE" if band.stood_aside else "FIRES"
        tone = DIM if band.stood_aside else GAIN
        out.append('<div class="gb-sub-rule"></div>')
        out.append(
            f'<div class="gb-meta"><span style="color:{tone}">BAND {escape(fires)}'
            f"</span> &nbsp; {escape(band.summary)}</div>"
        )
    return "".join(out)


#: What the system did, as a row class. Vermillion is not an action - it marks a row
#: whose provenance is not live, and a row written since the last refresh, both of which
#: are facts *about* the row rather than things the system decided.
ROW_CLASSES = {
    ENTER_LONG: "gb-row-gain",
    EXIT: "gb-row-loss",
    HOLD: "gb-row-hold",
}


def row_class(action: str, provenance: str = records.LIVE, fresh: bool = False) -> str:
    """The left border for one row, in priority order.

    Freshness outranks provenance, which outranks the action. That order is the reading
    order of the questions: *is this new*, then *is this real*, then *what was it*. A row
    that is both new and a rehearsal shows new for one refresh and then settles to the
    provenance mark, which is the more durable fact about it.
    """
    if fresh or provenance != records.LIVE:
        return "gb-row-accent"
    return ROW_CLASSES.get(action.lower(), "gb-row-hold")


def activity_table(
    decisions: Sequence[DecisionRecord],
    trades: Sequence,
    limit: int = 12,
    fresh: frozenset | set | None = None,
) -> str:
    """Recent activity: decisions, and any closed round trips beside them.

    **Two independent channels in one row.** The 2px left border states what the system
    did; the numerals state which way it went. Neither is the only carrier of its fact -
    the action is spelled in the WHAT column and the sign is carried by a glyph - so the
    row survives being read without colour, which is what
    `test_action_is_recoverable_from_the_row_text` holds.

    Both sources are LIVE and both come from the same state directory, so this is one
    source and not a mix. Trades are shown by exit time because that is when the row
    became true.
    """
    fresh = fresh or frozenset()
    rows: list[tuple] = []
    classes: list[str] = []
    for trade in sorted(trades, key=lambda t: t.exit_time, reverse=True)[:limit]:
        rows.append(
            (
                f"{trade.exit_time:%Y-%m-%d %H:%M}",
                trade.symbol,
                "TRADE",
                escape(trade.exit_reason),
                status_html(trade.net_pnl, f"{trade.net_pnl:+,.2f}"),
            )
        )
        classes.append("gb-row-gain" if trade.net_pnl >= 0 else "gb-row-loss")
    for record in list(decisions)[-limit:][::-1]:
        strength = record.signal.trend_strength
        action = record.signal.action
        rows.append(
            (
                f"{record.as_of:%Y-%m-%d}",
                record.symbol,
                escape(action.upper()),
                escape(decision_detail(record)),
                status_html(strength, f"{strength:+.4f}"),
            )
        )
        classes.append(
            row_class(
                action,
                record.provenance,
                fresh=(record.as_of, record.symbol) in fresh,
            )
        )
    if not rows:
        return ""
    return table_html(
        ("WHEN", "SYMBOL", "WHAT", "DETAIL", "VALUE"),
        rows[: limit * 2],
        numeric=(4,),
        raw=(4,),
        row_classes=classes[: limit * 2],
    )


# ── the app ──────────────────────────────────────────────────────────────────


def main(
    state_dir: str | Path = "checkpoints/live", source: str = records.LIVE
) -> None:  # pragma: no cover
    """Render the console. Exercised by ``streamlit run``, not by the suite.

    Every computation lives in the pure functions above, which are tested. What is left
    here is layout, and layout is checked by looking at it.

    **The rows are ordered by source, not by importance.** Row 1 is what is happening now;
    rows 2 and 3 are what was measured over sixteen folds. A reader should never have to
    check a pill to know which half of the page they are in - the pills are there to settle
    the question, not to be the only answer to it.
    """
    import streamlit as st

    root = Path(state_dir)
    cfg = load_config()
    now = pd.Timestamp.now(tz="UTC")
    st.set_page_config(page_title="GlassBox Trader", layout="wide")
    # One call, so the field and the rules it depends on arrive as a single node. Two
    # calls would let Streamlit insert a wrapper between them, and the field is
    # position:fixed behind everything - a wrapper with its own stacking context would
    # put it in front of the content it is meant to sit behind.
    st.markdown(stylesheet() + starfield(), unsafe_allow_html=True)

    read_broker = st.cache_data(ttl=BROKER_TTL_SECONDS, show_spinner=False)(
        _broker_view
    )
    read_closes = st.cache_data(ttl=BROKER_TTL_SECONDS, show_spinner=False)(
        _recent_closes
    )
    read_decisions = st.cache_data(ttl=LOCAL_TTL_SECONDS, show_spinner=False)(
        _recent_decisions
    )
    read_results = st.cache_data(ttl=60, show_spinner=False)(load_results)
    read_daily = st.cache_data(ttl=60, show_spinner=False)(load_daily_equity)

    thresholds = _thresholds(root)
    reliability = load_reliability(root / RELIABILITY_FILE)
    book = Book.load(root / "book.json")
    hours = _in_market_hours(cfg)

    try:
        quantities, prices, account = read_broker(cfg)
        st.session_state["last_broker"] = (quantities, prices, account, now)
        broker_age = 0.0
    except Exception:  # noqa: BLE001 - a refused read dims the panel, never empties it
        cached = st.session_state.get("last_broker")
        if cached is None:
            st.markdown(
                '<div class="gb-empty">NO BROKER READ HAS SUCCEEDED YET</div>',
                unsafe_allow_html=True,
            )
            return
        quantities, prices, account, at = cached
        broker_age = (now - at).total_seconds()

    age = cycle_age(root, now)
    loop_silent = age > 2 * cfg.live.heartbeat_seconds
    stale = staleness_html(
        age if loop_silent else broker_age, cfg.live.heartbeat_seconds
    )

    # ── row 1: now ───────────────────────────────────────────────────────────
    st.markdown(
        session_strip(
            cfg,
            status_of(thresholds, hours, quantities),
            root,
            source,
            age,
            stale,
        ),
        unsafe_allow_html=True,
    )

    equity = float(account.get("equity", math.nan))
    append_equity(root, now, equity)
    curve = load_equity(root, now)
    sessions = _live_session_count(root, source)
    trades = records.load_trades(root)

    left, right = st.columns([2, 1])
    with left:
        live_source = Source(LIVE, f"{sessions} sessions, {len(trades)} trades")
        body = (
            equity_svg(curve)
            if len(curve) > 1
            else too_little(
                live_source,
                f"{len(curve)} equity reading this session - the curve needs two",
            )
        )
        st.markdown(
            region(
                "Session equity",
                live_source,
                body,
                f"Cash {account.get('cash', math.nan):,.2f}",
            ),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            region(
                "Cycle",
                Source(LIVE, "loop cadence"),
                countdown_svg(
                    0.0 if math.isinf(age) else max(cfg.live.poll_seconds - age, 0.0),
                    cfg.live.poll_seconds,
                ),
                "Measured from the loop's own last write, not this page's timer.",
            ),
            unsafe_allow_html=True,
        )

    # ── rows 2-3: measured ───────────────────────────────────────────────────
    #
    # **This is where the pill defect lived, and the fix is that it is no longer here.**
    # Two of these regions read `results.csv` and two read `report/daily_equity.csv`; a
    # partial daily artefact had both of its cards labelled with the results file's range,
    # announcing "folds 1-16" while holding three. The first repair named the two locals
    # `from_results` and `from_daily` so a call site could not pick the wrong one without
    # it reading wrong - which is a convention, and a convention is a note. The pill is now
    # built inside the region, from the rows that region draws, so there is no second frame
    # here to pick wrongly from.
    results = read_results()
    reference = reference_rows(results)
    daily = read_daily()
    arm = cfg.model.active

    radar_col, equity_col, bars_col = st.columns(3)
    with radar_col:
        st.markdown(shape_region(reference, arm), unsafe_allow_html=True)
    with equity_col:
        st.markdown(cumulative_region(daily, arm), unsafe_allow_html=True)
    with bars_col:
        st.markdown(fold_region(reference, arm), unsafe_allow_html=True)

    rel_col, cal_col = st.columns(2)
    with rel_col:
        st.markdown(
            region(
                "Reliability",
                Source(
                    BACKTEST,
                    f"{reliability.folds} folds" if reliability else "not measured",
                ),
                reliability_body(reliability, band_context(root / "thresholds.json")),
                "An explanation makes a decision legible; it does not make it right.",
            ),
            unsafe_allow_html=True,
        )
    with cal_col:
        st.markdown(calendar_region(daily, arm), unsafe_allow_html=True)

    # ── row 4: positions and activity ────────────────────────────────────────
    rows = position_rows(book, quantities, prices)
    closes = read_closes(cfg) if rows else {}
    pos_source = Source(LIVE, f"{len(rows)} held")
    st.markdown(
        region(
            "Positions",
            pos_source,
            (
                position_table(rows)
                if rows
                else too_little(pos_source, "NO POSITIONS HELD")
            ),
        ),
        unsafe_allow_html=True,
    )
    for row in rows:
        note = stop_note(row)
        if note:
            st.markdown(f'<div class="gb-stale">{note}</div>', unsafe_allow_html=True)
        history = closes.get(row.symbol)
        if history is not None and not history.empty:
            st.markdown(sparkline_svg(history.tail(60), row), unsafe_allow_html=True)

    decisions = read_decisions(root, source)
    # Computed here, before the activity region draws, because a row written since the
    # last refresh takes the accent border for exactly one pass. Reading it after the
    # table would have been reading it too late - the mark would always be one refresh
    # behind the thing it marks.
    fresh, seen = newly_written(
        decisions, st.session_state.get("seen_decisions", set())
    )
    st.session_state["seen_decisions"] = seen
    activity_source = Source(LIVE, f"{len(decisions)} decisions, {len(trades)} trades")
    st.markdown(
        region(
            "Recent activity",
            activity_source,
            activity_table(decisions, trades, fresh=fresh)
            or too_little(activity_source, "NOTHING RECORDED IN THIS DIRECTORY YET"),
        ),
        unsafe_allow_html=True,
    )

    _copilot_panel(root, cfg, st)

    if not decisions:
        _refresh(cfg, st)
        return

    # ── row 6: forecast paths ────────────────────────────────────────────────
    latest = {record.symbol: record for record in decisions}
    st.markdown(
        '<div class="gb-label">Forecast paths</div>'
        '<div class="gb-meta">The price line advances once per trading day: a decision is '
        "taken on the last completed bar and does not change within a session.</div>",
        unsafe_allow_html=True,
    )
    if not closes:
        closes = read_closes(cfg)
    symbols = [s for s in sorted(latest) if s in closes and not closes[s].empty]
    for start in range(0, len(symbols), 2):
        for column, symbol in zip(
            st.columns(2), symbols[start : start + 2], strict=False
        ):
            history = closes[symbol]
            with column:
                st.markdown(
                    region(
                        f"{symbol} close and forecast",
                        forecast_source(latest[symbol].as_of),
                        forecast_svg(
                            history,
                            price_path(
                                float(history.iloc[-1]), latest[symbol].forecast.path
                            ),
                            thresholds,
                            symbol,
                        ),
                        bar_note(latest[symbol].as_of, now, hours),
                    ),
                    unsafe_allow_html=True,
                )

    # ── row 7: attribution and spectral ──────────────────────────────────────
    st.markdown(
        '<div class="gb-label">Decisions and attribution</div>', unsafe_allow_html=True
    )
    for record in decision_rows(decisions):
        title = expander_title(record)
        if (record.as_of, record.symbol) in fresh:
            # Plain text, not a styled span: a Streamlit expander label takes no HTML.
            title = f"{title}   · NEW"
        with st.expander(title):
            st.markdown(narrative_html(record.narrative), unsafe_allow_html=True)
            st.markdown(contributions_svg(record.attribution), unsafe_allow_html=True)
            # GB-53. Empty string for a model that does not decompose by frequency, so the
            # panel is absent under DLinear and persistence rather than blank.
            st.markdown(spectral_panel(record.attribution), unsafe_allow_html=True)

    _refresh(cfg, st)


def _live_session_count(root: Path, source: str) -> int:  # pragma: no cover - I/O
    """How many distinct bars this directory has decided. A proxy for sessions, and an
    honest one: one completed bar is decided per session by the 19 Aug ruling."""
    try:
        records_seen = _recent_decisions(root, source)
    except Exception:  # noqa: BLE001 - a count is never worth failing the page over
        return 0
    return len({record.as_of for record in records_seen})


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


#: Local files are cheap and change every cycle; the broker is rate-limited, shared with
#: the live loop, and charged against the same account. One TTL for both would either
#: hammer the API at the local cadence or freeze the countdown at the broker's, so they
#: are separate by construction rather than by convention.
LOCAL_TTL_SECONDS = 5
BROKER_TTL_SECONDS = 30


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

    Rendered above the log because it is the only thing on the page waiting for a person.
    **No expiry**: a recommendation waits until it is answered, because a timer would be
    the system deciding "no" on the operator's behalf and recording nothing.

    **The two buttons are the only filled elements on the page**, and that is the design
    rather than emphasis. Every other surface is type and rules on the ground; these are
    solid because this is the one place a click moves money, and the weight is the
    warning. Both answers are recorded - a rejection that left no trace would make the log
    a record of what the system wanted rather than of what happened, and "the operator
    said no" is the only place a human enters the loop.
    """
    from glassbox.engine.executor import AlpacaBroker
    from glassbox.live_loop import answer_pending

    queue = records.load_pending(root)
    if not queue:
        return

    body = []
    for entry in queue:
        body.append(
            f'<div class="gb-meta">{escape(entry["decision_id"])} &nbsp;·&nbsp; '
            f"{escape(provenance_label(entry.get('provenance', records.LIVE)))}</div>"
            f'<div class="gb-emph" style="padding:.4rem 0 .2rem">'
            f"{escape(pending_summary(entry))}</div>"
        )
    st.markdown(
        region(
            "Awaiting approval — Co-Pilot",
            Source(LIVE, f"{len(queue)} pending"),
            "".join(body),
            "A rejection is recorded too: the log is what happened, not what was wanted.",
        ),
        unsafe_allow_html=True,
    )

    for entry in queue:
        st.markdown(narrative_html(entry.get("narrative", "")), unsafe_allow_html=True)
        approve_col, reject_col = st.columns(2)
        with approve_col:
            st.markdown('<div class="gb-approve">', unsafe_allow_html=True)
            approved = st.button("APPROVE", key=f"a-{entry['decision_id']}")
            st.markdown("</div>", unsafe_allow_html=True)
        with reject_col:
            st.markdown('<div class="gb-reject">', unsafe_allow_html=True)
            rejected = st.button("REJECT", key=f"r-{entry['decision_id']}")
            st.markdown("</div>", unsafe_allow_html=True)
        if approved:
            answer_pending(cfg, AlpacaBroker(), root, entry["decision_id"], True)
            st.rerun()
        if rejected:
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
    "bar_note",
    "channel_colour",
    "contributions_svg",
    "decision_detail",
    "decision_rows",
    "decision_table",
    "escape",
    "expander_title",
    "forecast_source",
    "forecast_svg",
    "frequency_shares",
    "gain_phase_svg",
    "is_fragile",
    "is_rtl",
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
