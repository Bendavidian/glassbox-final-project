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

import json
import math
import time
from dataclasses import dataclass
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

HISTORY_BARS = 60
RELIABILITY_FILE = "reliability.json"

# Hebrew block, for the direction decision. See the module docstring on why `dir="auto"`
# is the wrong tool here.
HEBREW = range(0x0590, 0x0600)


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
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
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
    width: int = 720,
    height: int = 240,
) -> str:
    """Recent closes with the predicted H-day path continuing past the last bar.

    ``history`` is prices; ``path`` is the forecast **projected into prices** by the
    caller, so the chart shows one quantity on one axis rather than asking a reader to hold
    a log-return scale in their head beside a price one.

    The calibrated entry threshold is drawn as an orange dashed rule — chrome, because it
    is a decision boundary rather than a measurement. **When the band is ``never()`` the
    line is not omitted**: the chart says so in words where the rule would have been.
    Omitting it would make a session that cannot trade look like one that simply had not
    yet, which is the difference between abstaining and waiting.
    """
    left, right, top, bottom = 46, 12, 22, 26
    plot_w = width - left - right
    plot_h = height - top - bottom

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
        _text(4, top + 8, f"{high:,.2f}", MUTED),
        _text(4, top + plot_h, f"{low:,.2f}", MUTED),
        _text(left, 12, f"{symbol}  CLOSE / FORECAST", ORANGE),
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
    body.append(_text(x_at(offset) + 4, top + plot_h + 14, "NOW", ORANGE_DIM))

    if thresholds.fires:
        entry = float(history.iloc[-1]) * math.exp(thresholds.lower)
        if low <= entry <= high:
            body.append(
                _rule(left, y_at(entry), left + plot_w, y_at(entry), ORANGE, "5 4")
            )
            body.append(
                _text(
                    left + plot_w,
                    y_at(entry) - 4,
                    "ENTRY THRESHOLD",
                    ORANGE,
                    anchor="end",
                )
            )
        else:
            body.append(
                _text(
                    left + plot_w,
                    top + plot_h + 14,
                    "ENTRY THRESHOLD OFF SCALE",
                    ORANGE,
                    anchor="end",
                )
            )
    else:
        body.append(
            _text(
                left + plot_w,
                top + plot_h + 14,
                "NO CALIBRATED BAND — STOOD ASIDE",
                ORANGE,
                anchor="end",
            )
        )

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
    attribution: Attribution, width: int = 720, row_height: int = 26
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
    body.append(
        _text(
            12,
            height - 8,
            f"CANCELLATION — {survived:.1f}% OF THE GROSS VIEW SURVIVED",
            ORANGE if survived < 50.0 else MUTED,
        )
    )
    return _svg(width, height, "".join(body), "per-channel contributions")


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


def decision_rows(decisions: list[DecisionRecord]) -> pd.DataFrame:
    """The log as a table: newest first, one row per decision."""
    return pd.DataFrame(
        [
            {
                "as_of": record.as_of,
                "symbol": record.symbol,
                "action": record.signal.action,
                "trend_strength": record.signal.trend_strength,
                "forecast_total": record.attribution.forecast_total,
                "cancellation": cancellation(record.attribution),
                "order": "yes" if record.order else "",
            }
            for record in decisions
        ],
        columns=[
            "as_of",
            "symbol",
            "action",
            "trend_strength",
            "forecast_total",
            "cancellation",
            "order",
        ],
    ).sort_values(["as_of", "symbol"], ascending=[False, True])


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
      color: {ORANGE};
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
</style>
"""


def ruler_html(marks: int = 4) -> str:
    """The numbered grid rule along the top edge."""
    ticks = "".join(
        f'<span class="gb-meta">{value:02d}</span>'
        for value in range(0, 25 * marks + 1, 25)
    )
    return f'<div class="gb-ruler">{ticks}</div>'


def header_html(cfg: Config, status: str, reliability: Reliability | None) -> str:
    """Project block, status chip and the reliability panel, in one strip.

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
    return (
        f'<div class="gb-panel">'
        f'<div class="gb-meta">PROJECT: <span style="color:{ORANGE}">GLASSBOX TRADER</span>'
        f" &nbsp;·&nbsp; MODEL: {cfg.model.active.upper()}"
        f" &nbsp;·&nbsp; CHANNELS: {cfg.channels.active.upper()}"
        f" &nbsp;·&nbsp; CONFIG {config_hash(cfg)[:12]}</div>"
        f'<div style="margin-top:.5rem"><span class="gb-status">{status}</span></div>'
        f'<div style="margin-top:.6rem">{record}</div>'
        f"</div>"
    )


# ── the app ──────────────────────────────────────────────────────────────────


def main(state_dir: str | Path = "checkpoints/live") -> None:  # pragma: no cover
    """Render the dashboard. Exercised by ``streamlit run``, not by the suite.

    Every computation it performs lives in the pure functions above, which are tested. What
    is left here is layout, and layout is checked by looking at it.
    """
    import streamlit as st

    root = Path(state_dir)
    cfg = load_config()
    st.set_page_config(page_title="GlassBox Trader", layout="wide")
    st.markdown(stylesheet(), unsafe_allow_html=True)
    st.markdown(ruler_html(), unsafe_allow_html=True)

    thresholds = _thresholds(root)
    reliability = load_reliability(root / RELIABILITY_FILE)
    book = Book.load(root / "book.json")
    quantities, prices, account = _broker_view(cfg)
    hours = _in_market_hours(cfg)

    st.markdown(
        header_html(cfg, status_of(thresholds, hours, quantities), reliability),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="gb-label">POSITIONS</div>', unsafe_allow_html=True)
    rows = position_rows(book, quantities, prices)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "SYMBOL": row.symbol,
                    "QTY": f"{row.quantity:.9f}",
                    "ENTRY": f"{row.entry_price:,.2f}" if row.managed else "—",
                    "LAST": f"{row.price:,.2f}",
                    "VALUE": f"{row.market_value:,.2f}",
                    "UNREALISED": (
                        "—"
                        if not row.managed or math.isnan(row.unrealised)
                        else f"{'▲' if row.unrealised >= 0 else '▼'} "
                        f"{row.unrealised:,.2f} ({row.unrealised_pct:+.2f}%)"
                    ),
                    "STATE": "MANAGED" if row.managed else "QUARANTINED",
                }
                for row in rows
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.markdown(
        f'<div class="gb-meta">EQUITY {account.get("equity", float("nan")):,.2f}'
        f' &nbsp;·&nbsp; CASH {account.get("cash", float("nan")):,.2f}</div>',
        unsafe_allow_html=True,
    )

    decisions = _recent_decisions(root)
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
    st.dataframe(decision_rows(decisions), width="stretch", hide_index=True)
    for record in sorted(decisions, key=lambda r: (r.as_of, r.symbol), reverse=True):
        title = (
            f"{record.as_of:%Y-%m-%d}  {record.symbol}  {record.signal.action.upper()}"
        )
        with st.expander(title):
            st.markdown(narrative_html(record.narrative), unsafe_allow_html=True)
            st.markdown(contributions_svg(record.attribution), unsafe_allow_html=True)

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


def _recent_decisions(root: Path) -> list[DecisionRecord]:  # pragma: no cover - I/O
    today = pd.Timestamp.now(tz="UTC")
    return records.load_decisions(today - pd.Timedelta(days=30), today, root)


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "ASIDE",
    "CLOSED",
    "IDLE",
    "ORANGE",
    "RAMP",
    "RUNNING",
    "PositionRow",
    "Reliability",
    "channel_colour",
    "contributions_svg",
    "decision_rows",
    "forecast_svg",
    "header_html",
    "is_rtl",
    "load_reliability",
    "main",
    "narrative_html",
    "position_rows",
    "price_path",
    "status_of",
    "stylesheet",
]
