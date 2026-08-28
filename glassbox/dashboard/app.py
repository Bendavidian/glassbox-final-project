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

INK = "#0B0E11"  # page ground
PANEL = "#12161B"  # card fill
HAIRLINE = "#1E252D"  # card border, 1px, subtle
PAPER = "#E6EAF0"  # primary text
MUTED = "#8A94A6"  # labels, axes, secondary text

# Neutral emphasis and selection ONLY. Not a third status colour: nothing on this console
# encodes a value in blue outside the ramp, and an accent that started carrying meaning
# would be the second data family the ramp already refuses to become.
ACCENT = "#3B82F6"

# Kept under their old names because the SVG builders below take colours as arguments and
# `ORANGE` is threaded through several of them as "the chrome colour". The hue changed;
# the role did not.
ORANGE = ACCENT
ORANGE_DIM = "#1E3A5F"

# Data. Dark for slow, light for fast — the spectral ramp, applied to channels ordered by
# how much history each one looks back over.
RAMP = ("#0A2239", "#123F63", "#1B6CA8", "#2E97D4", "#6FC3EC", "#B7E3F7")

# Status. Gain and loss, and nothing else. Ruled 27 Aug 2026; see DECISIONS for the
# CIE76 distances against every other role, measured from the constants above rather
# than sampled from a screenshot.
#
# **Two rules travel with these two colours, and both are enforced by tests rather than
# remembered.** (1) Status colour never appears inside a data-encoding chart: the ramp
# owns meaning there, and a third family would make a reader ask what green means on an
# axis that is already spending colour on frequency. (2) Status colour never carries
# information alone — every gain and loss is redundant with a sign and a glyph, so the
# panel reads in greyscale. A colour that is the only carrier of a fact is a fact a
# colour-blind reader does not have.
#
# The loss colour leans magenta deliberately. ORANGE is an orange-vermillion, so an
# ordinary red separates on the number and not in the eye — and the eye is what matters
# on a P&L figure. `#B03A5B` holds ΔE 47.65 from it where a conventional red would not.
GAIN = "#22C55E"
LOSS = "#EF4444"

#: Area fills under the equity curves. 15%, between the 12-18% the design calls for: light
#: enough that a gridline reads through it, solid enough to carry the sign at a glance.
GAIN_FILL = "rgba(34,197,94,0.15)"
LOSS_FILL = "rgba(239,68,68,0.15)"

#: Every surface that encodes data with colour. `test_status_colour_never_enters_a_data
#: _chart` renders each one and refuses to find GAIN or LOSS in it.
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


def card(title: str, source: Source, body: str, note: str = "") -> str:
    """One card: title, source pill, body, and an optional note under it.

    **One string, one ``st.markdown``.** Streamlit wraps every markdown call in its own
    container, so a card assembled from several calls cannot hold a border or an equal
    height - the wrappers land between the pieces. Building the whole card as a string
    also keeps it a pure function, testable without a browser, which is how every other
    surface in this module already works.
    """
    return (
        '<div class="gb-card">'
        f'<div class="gb-card-head"><span class="gb-card-title">{escape(title)}</span>'
        f"{source.pill}</div>"
        f'<div class="gb-card-body">{body}</div>'
        + (f'<div class="gb-card-note">{escape(note)}</div>' if note else "")
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


def equity_svg(series: pd.Series, width: int = 720, height: int = 180) -> str:
    """The session equity curve. Flat until something happens, and that is correct.

    Coloured by its own sign against the session's first reading, because equity is a P&L
    quantity and the palette's status role is exactly for that. The opening level is drawn
    as a rule so the sign is readable as geometry and not only as colour.
    """
    if series.empty:
        return ""
    left, right, top, bottom = 90, width - 20, 24, height - 26
    opening = float(series.iloc[0])
    latest = float(series.iloc[-1])
    change = latest - opening
    low, high = float(series.min()), float(series.max())

    # **A flat session is the common case here, not an edge case.** The loop stands aside
    # on most bars, so equity does not move, and dividing by max(high - low, 1e-9) drove
    # every point to  - the line and its label sat on the floor of the card under a
    # column of empty space. A span that is negligible against the level is drawn as a
    # centred flat line, which is what it is.
    flat = (high - low) <= abs(high) * 1e-9

    def y(value: float) -> float:
        if flat:
            return (top + bottom) / 2
        return bottom - (value - low) / (high - low) * (bottom - top)

    step = (right - left) / max(len(series) - 1, 1)
    points = " ".join(
        f"{left + index * step:.1f},{y(value):.1f}"
        for index, value in enumerate(series.to_numpy(dtype="float64"))
    )
    body = [
        _text(12, top + 4, "EQUITY", ORANGE),
        _rule(left, y(opening), right, y(opening), HAIRLINE, dash="2 3"),
        _text(left - 6, y(opening) + 3, f"{opening:,.0f}", MUTED, anchor="end"),
        (
            f'<polyline points="{points}" fill="none" '
            f'stroke="{status_colour(change)}" stroke-width="1.5"/>'
        ),
        _text(
            right,
            top + 4,
            f"{status_glyph(change)} {change:+,.2f}",
            status_colour(change),
            anchor="end",
        ),
        _text(12, bottom + 16, f"{len(series)} READINGS THIS SESSION", MUTED, size=8),
    ]
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
    remaining: float, total: float, width: int = 460, height: int = 40
) -> str:
    """The cycle countdown, as a rule that depletes. No motion, no pulse.

    It redraws shorter each refresh because time has actually passed, which is the whole
    distinction this panel is built on: the mark moves because the quantity moved.
    """
    total = max(total, 1e-9)
    left = max(0.0, min(remaining, total))
    track_from, track_to = 12, width - 54
    filled = (track_to - track_from) * (left / total)
    return _svg(
        width,
        height,
        "".join(
            [
                _text(12, 14, "NEXT CYCLE", MUTED, size=8),
                _rule(track_from, 28, track_to, 28, HAIRLINE),
                _rule(track_from, 28, track_from + filled, 28, ACCENT),
                _text(width - 12, 31, f"{left:.0f}s", PAPER, size=10, anchor="end"),
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


# ── BACKTEST cards ───────────────────────────────────────────────────────────

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
    arm = rows[rows["model"] == model]
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
        body.append(_text(lx, ly, label, MUTED, size=8, anchor=anchor_at))
        body.append(_text(lx, ly + 11, printed, PAPER, size=8, anchor=anchor_at))
    body.append(
        f'<polygon points="{" ".join(shape)}" fill="{GAIN_FILL}" '
        f'stroke="{GAIN}" stroke-width="1.5"/>'
    )
    return _svg(width, height, "".join(body), f"{model} performance shape")


def _one_arm(
    daily: pd.DataFrame, model: str, channels: str = "C0_base"
) -> pd.DataFrame:
    """One model's curves, one feature set. See :func:`reference_rows` for why both.

    The daily artefact carries every arm of the reference condition, so a filter on `model`
    alone leaves two copies of the same curve - one per channel set - and every date is
    then counted twice.
    """
    arm = daily[daily["model"] == model]
    if "channels" in arm.columns:
        arm = arm[arm["channels"].isna() | (arm["channels"] == channels)]
    return arm


def cumulative_equity_svg(
    daily: pd.DataFrame, model: str, width: int = 460, height: int = 300
) -> str:
    """Cumulative equity across the folds, as a filled area coloured by its own sign."""
    if daily.empty or "model" not in daily.columns:
        return ""
    arm = _one_arm(daily, model).sort_values(["fold", "date"])
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
    low, high = min(series), max(series)
    span = max(high - low, 1e-9)

    def y(value: float) -> float:
        return bottom - (value - low) / span * (bottom - top)

    step = (right - left) / max(len(series) - 1, 1)
    points = [f"{left + i * step:.1f},{y(v):.1f}" for i, v in enumerate(series)]
    change = series[-1] - 1.0
    colour, fill = (GAIN, GAIN_FILL) if change >= 0 else (LOSS, LOSS_FILL)
    area = f"{left},{y(1.0):.1f} " + " ".join(points) + f" {right},{y(1.0):.1f}"
    body = [
        f'<polygon points="{area}" fill="{fill}" stroke="none"/>',
        _rule(left, y(1.0), right, y(1.0), HAIRLINE, dash="2 3"),
        _text(left - 6, y(1.0) + 3, "1.00", MUTED, size=8, anchor="end"),
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{colour}" stroke-width="1.6"/>',
        _text(left, bottom + 18, f"{len(series)} TRADING DAYS", MUTED, size=8),
        _text(
            right,
            top + 4,
            f"{status_glyph(change)} {change * 100:+.2f}%",
            colour,
            size=11,
            anchor="end",
        ),
    ]
    return _svg(width, height, "".join(body), f"{model} cumulative equity")


def fold_bars_svg(
    rows: pd.DataFrame, model: str, width: int = 460, height: int = 300
) -> str:
    """One bar per walk-forward fold, green above zero and red below."""
    arm = rows[rows["model"] == model].sort_values("fold")
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
        _text(left - 6, zero + 3, "0%", MUTED, size=8, anchor="end"),
    ]
    for index, (fold, value) in enumerate(values):
        magnitude = abs(value) / peak * (bottom - top) / 2
        x = left + index * slot + slot * 0.18
        w = slot * 0.64
        y0 = zero - magnitude if value >= 0 else zero
        body.append(
            f'<rect x="{x:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{magnitude:.1f}" '
            f'fill="{GAIN if value >= 0 else LOSS}"/>'
        )
        if index % 3 == 0:
            body.append(
                _text(
                    x + w / 2, bottom + 14, f"f{fold}", MUTED, size=7, anchor="middle"
                )
            )
    best, worst = max(values, key=lambda v: v[1]), min(values, key=lambda v: v[1])
    body.append(
        _text(left, top - 6, f"BEST f{best[0]} {best[1] * 100:+.2f}%", GAIN, size=8)
    )
    body.append(
        _text(
            right,
            top - 6,
            f"WORST f{worst[0]} {worst[1] * 100:+.2f}%",
            LOSS,
            size=8,
            anchor="end",
        )
    )
    return _svg(width, height, "".join(body), f"{model} return by fold")


def calendar_svg(
    daily: pd.DataFrame, model: str, width: int = 700, height: int = 280
) -> str:
    """A heatmap of daily equity change, one tile per trading day, by calendar week."""
    if daily.empty or "model" not in daily.columns:
        return ""
    arm = _one_arm(daily, model).sort_values(["fold", "date"])
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
    tile, gap = 13, 3
    body = []
    for day in days:
        week = int((day - first).days // 7)
        x = 40 + week * (tile + gap)
        y = 26 + int(day.dayofweek) * (tile + gap)
        if x > width - tile:
            continue
        value = changes[day]
        opacity = 0.18 + 0.82 * min(abs(value) / peak, 1.0)
        body.append(
            f'<rect x="{x}" y="{y}" width="{tile}" height="{tile}" rx="2" '
            f'fill="{GAIN if value >= 0 else LOSS}" fill-opacity="{opacity:.2f}"/>'
        )
    for index, label in enumerate(("MON", "", "WED", "", "FRI")):
        if label:
            body.append(
                _text(
                    34,
                    26 + index * (tile + gap) + 10,
                    label,
                    MUTED,
                    size=7,
                    anchor="end",
                )
            )
    up = sum(1 for v in changes.values() if v > 0)
    body.append(_text(40, height - 10, f"{len(changes)} TRADING DAYS", MUTED, size=8))
    body.append(
        _text(
            width - 12,
            height - 10,
            f"{up} UP / {len(changes) - up} DOWN",
            MUTED,
            size=8,
            anchor="end",
        )
    )
    return _svg(width, height, "".join(body), f"{model} daily results calendar")


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
    raw: Sequence[int] = (),
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
        body.append(f"<tr>{''.join(cells)}</tr>")
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


def stylesheet() -> str:
    """The console's whole visual language, as one stylesheet.

    **Card equal-height is the only part that fights Streamlit.** `st.columns` renders each
    column as a flex child of a row, so a card that is `height:100%` inside a column that
    is `display:flex` fills the row's tallest card without any measurement. Targeting
    Streamlit's own `data-testid` attributes is load-bearing rather than cosmetic here, and
    it is the reason a card must be one `st.markdown` call: a second call inserts another
    wrapper between the column and the card, and the height chain breaks at it.

    Reflow needs no media queries. Streamlit stacks columns below its own breakpoint, so
    the grid collapses to one column on a narrow viewport on its own; only the type scale
    is adjusted.
    """
    return f"""
<style>
  .stApp {{ background: {INK}; }}
  html, body, [class*="css"] {{
      color: {PAPER};
      font-family: Inter, "SF Pro Text", -apple-system, "Segoe UI", sans-serif;
      font-feature-settings: "tnum" 1, "cv05" 1;
  }}
  .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1800px; }}
  #MainMenu, footer, header {{ visibility: hidden; }}

  /* ── cards ───────────────────────────────────────────────────────────── */
  [data-testid="stColumn"] {{ display: flex; }}
  [data-testid="stColumn"] > div {{ width: 100%; display: flex; }}
  .gb-card {{
      background: {PANEL}; border: 1px solid {HAIRLINE}; border-radius: 10px;
      padding: .85rem 1rem .9rem; width: 100%; height: 100%;
      display: flex; flex-direction: column; margin-bottom: .85rem;
  }}
  .gb-card-head {{
      display: flex; align-items: center; justify-content: space-between;
      gap: .6rem; margin-bottom: .7rem;
  }}
  .gb-card-title {{
      font-size: .74rem; letter-spacing: .1em; text-transform: uppercase;
      color: {MUTED}; font-weight: 600;
  }}
  .gb-card-body {{ flex: 1; display: flex; flex-direction: column; justify-content: center; }}
  .gb-card-note {{
      margin-top: .6rem; padding-top: .55rem; border-top: 1px solid {HAIRLINE};
      font-size: .68rem; color: {MUTED};
  }}

  /* The source pill. Every card carries one; a card cannot be built without it. */
  .gb-pill {{
      border: 1px solid; border-radius: 999px; padding: .1rem .5rem;
      font-size: .58rem; letter-spacing: .1em; font-weight: 700; white-space: nowrap;
  }}
  .gb-pill-detail {{
      color: {MUTED}; font-weight: 400; letter-spacing: .04em; margin-left: .4rem;
  }}

  /* A card with too little data says how little. It never borrows from the other
     source to fill itself, so this state has to be legible rather than apologetic. */
  .gb-empty {{
      color: {MUTED}; font-size: .78rem; text-align: center; padding: 2.2rem .5rem;
      border: 1px dashed {HAIRLINE}; border-radius: 8px;
  }}

  /* ── the session strip ────────────────────────────────────────────────── */
  .gb-strip {{
      display: flex; flex-wrap: wrap; gap: 1.6rem; align-items: baseline;
      background: {PANEL}; border: 1px solid {HAIRLINE}; border-radius: 10px;
      padding: .8rem 1.1rem; margin-bottom: .85rem;
  }}
  .gb-stat {{ display: flex; flex-direction: column; gap: .18rem; }}
  .gb-stat-key {{
      font-size: .58rem; letter-spacing: .12em; text-transform: uppercase; color: {MUTED};
  }}
  .gb-stat-value {{ font-size: .95rem; font-weight: 600; color: {PAPER}; }}
  .gb-stat-value.dim {{ color: {MUTED}; font-weight: 400; font-size: .8rem; }}

  .gb-state {{
      border-radius: 999px; padding: .15rem .7rem; font-size: .7rem; font-weight: 700;
      letter-spacing: .08em; border: 1px solid;
  }}

  /* ── numbers ──────────────────────────────────────────────────────────── */
  .gb-big {{ font-size: 2.1rem; font-weight: 700; line-height: 1.05; }}
  .gb-sub {{ font-size: .74rem; color: {MUTED}; margin-top: .3rem; }}

  /* ── tables ───────────────────────────────────────────────────────────── */
  .gb-table {{ width: 100%; border-collapse: collapse; font-size: .78rem; }}
  .gb-table th {{
      text-align: left; font-size: .58rem; letter-spacing: .11em; font-weight: 600;
      text-transform: uppercase; color: {MUTED}; padding: .35rem .6rem .45rem;
      border-bottom: 1px solid {HAIRLINE};
  }}
  .gb-table td {{
      padding: .45rem .6rem; border-bottom: 1px solid {HAIRLINE}; color: {PAPER};
  }}
  .gb-table tr:last-child td {{ border-bottom: none; }}
  .gb-table td.num, .gb-table th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .gb-table td.gb-flag {{ color: {ACCENT}; }}

  /* ── stale and status chrome ──────────────────────────────────────────── */
  .gb-stale {{
      color: {MUTED}; border: 1px solid {HAIRLINE}; border-radius: 6px;
      padding: .12rem .5rem; font-size: .62rem; letter-spacing: .06em;
  }}
  /* The Hebrew narrative flips its rule to the other side. Restored 28 Aug 2026 after
     the rebuild dropped it: the direction is decided by looking for Hebrew letters, and a
     rule that stayed on the left would sit at the end of the sentence rather than its
     start. Caught by test_the_rtl_narrative_rule_moves_to_the_other_side, which is the
     only thing that noticed. */
  .gb-narrative {{
      font-size: .92rem; line-height: 1.55; color: {PAPER};
      border-left: 2px solid {ACCENT}; padding: .2rem 0 .2rem .8rem;
  }}
  .gb-narrative[dir="rtl"] {{
      border-left: none; border-right: 2px solid {ACCENT};
      padding: .2rem .8rem .2rem 0;
  }}
  .gb-figure {{ color: {PAPER}; font-size: 1.4rem; font-weight: 700; }}
  .gb-label {{
      font-size: .72rem; letter-spacing: .1em; text-transform: uppercase;
      color: {MUTED}; margin: 1.1rem 0 .5rem; font-weight: 600;
  }}
  .gb-meta {{ font-size: .72rem; color: {MUTED}; }}

  /* ── expanders: the one widget that stays, reskinned ──────────────────── */
  [data-testid="stExpander"] details {{
      background: {PANEL}; border: 1px solid {HAIRLINE}; border-radius: 10px;
      margin-bottom: .5rem;
  }}
  [data-testid="stExpander"] summary {{ color: {PAPER}; font-size: .76rem; }}
  [data-testid="stExpander"] summary:hover {{ color: {ACCENT}; }}
  [data-testid="stExpander"] summary svg {{ fill: {MUTED}; }}
  [data-testid="stExpander"] details > div {{ background: {PANEL}; border: none; }}

  /* ── buttons ──────────────────────────────────────────────────────────── */
  .stButton > button {{
      background: {PANEL}; color: {PAPER}; border: 1px solid {HAIRLINE};
      border-radius: 8px; font-size: .76rem; font-weight: 600;
  }}
  .stButton > button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}

  @media (max-width: 900px) {{
      .gb-big {{ font-size: 1.6rem; }}
      .gb-strip {{ gap: 1rem; }}
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
    klass = "gb-stat-value dim" if dim else "gb-stat-value"
    return (
        f'<div class="gb-stat"><span class="gb-stat-key">{escape(key)}</span>'
        f'<span class="{klass}">{value}</span></div>'
    )


def session_strip(
    cfg: Config,
    state: str,
    state_dir: str | Path,
    source: str,
    cycle_seconds: float,
    stale: str = "",
) -> str:
    """Row 1: what is happening now, in one line each.

    **The bound directory is here and not in a settings drawer.** On the night of GATE 2's
    execution rehearsal the panel was pointed at `checkpoints/live` while the rehearsal
    wrote to `checkpoints/rehearsal`; its Co-Pilot queue rendered empty and correct, and
    criterion 2 had to be satisfied through the API instead. An empty queue and no
    recommendations are indistinguishable unless the panel says which directory it read.
    """
    colour = GAIN if state in (RUNNING, IDLE) else MUTED if state == CLOSED else ACCENT
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
        '<div class="gb-strip">'
        + f'<div class="gb-stat"><span class="gb-stat-key">SESSION</span>'
        f'<span class="gb-state" style="color:{colour};border-color:{colour}55">'
        f"{escape(state)}</span></div>"
        + _stat("LAST CYCLE", escape(age))
        + _stat("MODEL", escape(cfg.model.active.upper()), dim=True)
        + _stat("CHANNELS", escape(cfg.channels.active.upper()), dim=True)
        + _stat("UNIVERSE", f"{len(cfg.universe)} symbols", dim=True)
        + _stat("BOUND", escape(str(state_dir)), dim=True)
        + _stat("SHOWING", escape(source.upper()), dim=True)
        + _stat("CONFIG", escape(config_hash(cfg)[:10].upper()), dim=True)
        + (f'<div class="gb-stat">{stale}</div>' if stale else "")
        + "</div>"
    )


def reliability_body(reliability: Reliability | None, band: BandContext | None) -> str:
    """The decider's track record, large.

    **Not optional and not small.** A console that shows P&L while hiding how often the
    decider is right is the black box this project exists to oppose - and this one's answer
    is that it does not beat the always-long bar, which is exactly the number a reader is
    least likely to go looking for and most needs to see.
    """
    if reliability is None:
        return too_little(
            Source(BACKTEST, "not measured"),
            "NOT MEASURED - run smoke_offline --prepare-live",
        )
    gap = reliability.gap
    colour = GAIN if gap > 0 else LOSS
    return (
        f'<div class="gb-big" style="color:{colour}">{status_glyph(gap)} '
        f"{gap:+.4f}</div>"
        f'<div class="gb-sub">DIRECTION {reliability.direction:.4f} '
        f"against the always-long bar {reliability.always_long:.4f}</div>"
        f'<div class="gb-sub">OVER {reliability.folds} FOLDS &nbsp;·&nbsp; '
        f"MEASURED {escape(reliability.measured_on)} &nbsp;·&nbsp; "
        f"{escape(reliability.model.upper())}</div>"
        + (
            ""
            if band is None
            else f'<div class="gb-sub" style="margin-top:.7rem;padding-top:.6rem;'
            f'border-top:1px solid {HAIRLINE}">BAND {escape(band.summary)}</div>'
        )
    )


def activity_table(
    decisions: Sequence[DecisionRecord], trades: Sequence, limit: int = 12
) -> str:
    """Recent activity: decisions, and any closed round trips beside them.

    Both are LIVE and both come from the same state directory, so this is one source and
    not a mix. Trades are shown by exit time because that is when the row became true.
    """
    rows: list[tuple] = []
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
    for record in list(decisions)[-limit:][::-1]:
        strength = record.signal.trend_strength
        rows.append(
            (
                f"{record.as_of:%Y-%m-%d}",
                record.symbol,
                escape(record.signal.action.upper()),
                escape(provenance_label(record.provenance)),
                status_html(strength, f"{strength:+.4f}"),
            )
        )
    if not rows:
        return ""
    return table_html(
        ("WHEN", "SYMBOL", "WHAT", "DETAIL", "VALUE"),
        rows[: limit * 2],
        numeric=(4,),
        raw=(4,),
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
    st.markdown(stylesheet(), unsafe_allow_html=True)

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
            card(
                "Session equity",
                live_source,
                body,
                f"Cash {account.get('cash', math.nan):,.2f}",
            ),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            card(
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
    results = read_results()
    reference = reference_rows(results)
    daily = read_daily()
    arm = cfg.model.active
    measured = Source(BACKTEST, fold_range(reference))
    # **A card's pill states ITS OWN data's range, not the page's.** The equity area and
    # the calendar read , which can lag  by any number of
    # folds - a partial artefact from a short run had them announcing "folds 1-16" while
    # holding three. That is precisely the overclaim the source pills exist to prevent,
    # committed by the mechanism meant to prevent it.
    from_daily = Source(BACKTEST, fold_range(daily))
    missing = "report/daily_equity.csv not generated - run the study grid"

    radar_col, equity_col, bars_col = st.columns(3)
    with radar_col:
        radar = radar_svg(reference, arm)
        st.markdown(
            card(
                f"{arm.upper()} performance shape",
                measured,
                radar or too_little(measured, "no reference-condition rows"),
                "Each axis against its own reference. There is no composite score.",
            ),
            unsafe_allow_html=True,
        )
    with equity_col:
        area = cumulative_equity_svg(daily, arm)
        st.markdown(
            card(
                "Cumulative equity",
                from_daily,
                area or too_little(from_daily, missing),
                "Folds chained on returns, not concatenated on levels.",
            ),
            unsafe_allow_html=True,
        )
    with bars_col:
        bars = fold_bars_svg(reference, arm)
        st.markdown(
            card(
                "Return by fold",
                measured,
                bars or too_little(measured, "no reference-condition rows"),
            ),
            unsafe_allow_html=True,
        )

    rel_col, cal_col = st.columns(2)
    with rel_col:
        st.markdown(
            card(
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
        heat = calendar_svg(daily, arm)
        st.markdown(
            card("Daily results", from_daily, heat or too_little(from_daily, missing)),
            unsafe_allow_html=True,
        )

    # ── row 4: positions and activity ────────────────────────────────────────
    rows = position_rows(book, quantities, prices)
    closes = read_closes(cfg) if rows else {}
    pos_source = Source(LIVE, f"{len(rows)} held")
    st.markdown(
        card(
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
    activity_source = Source(LIVE, f"{len(decisions)} decisions, {len(trades)} trades")
    st.markdown(
        card(
            "Recent activity",
            activity_source,
            activity_table(decisions, trades)
            or too_little(activity_source, "NOTHING RECORDED IN THIS DIRECTORY YET"),
        ),
        unsafe_allow_html=True,
    )

    _copilot_panel(root, cfg, st)

    if not decisions:
        _refresh(cfg, st)
        return

    fresh, seen = newly_written(
        decisions, st.session_state.get("seen_decisions", set())
    )
    st.session_state["seen_decisions"] = seen

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
                    card(
                        f"{symbol} close and forecast",
                        Source(LIVE, f"{latest[symbol].as_of:%Y-%m-%d} bar"),
                        forecast_svg(
                            history,
                            price_path(
                                float(history.iloc[-1]), latest[symbol].forecast.path
                            ),
                            thresholds,
                            symbol,
                        ),
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
