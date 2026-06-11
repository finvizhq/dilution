"""Pure-Python SVG renderer for the cash-position chart.

DilutionTracker-style: historical cash bars (blue) followed by a 3-step
right-side bridge:

  - Maroon "OpCF" bar:  prorated operating cash flow since latest report
  - Light-blue overlay: capital raised since latest report (net of fees)
  - Final blue bar:     Current Est = latest + opcf_prorated + raised

Bars carry data-* attributes consumed by an inline JS hover handler
(see `_cash_position.html`) which renders a styled tooltip card on
mouseover. The fallback `<title>` element gives a plain native tooltip
if JS is disabled.

The bridge is separated from the historical series by a 1-bar visual
gap so the maroon/light-blue/final-blue stack reads as a distinct
"current state" cluster.
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import date

from markupsafe import Markup

from dilution.cash_history import CashHistory

# DT-style palette
_BLUE = "#1f4e79"
_MAROON = "#b03060"
_LIGHT_BLUE = "#9ec5e8"
_GRID = "#e5e7eb"
_AXIS_TEXT = "#6b7280"

# Canvas geometry
_W, _H = 920, 320
_PAD_L, _PAD_R, _PAD_T, _PAD_B = 56, 16, 16, 40
_INNER_W = _W - _PAD_L - _PAD_R
_INNER_H = _H - _PAD_T - _PAD_B

# Right-side bridge layout (OpCF + Current Est). Bridge bars use a
# fixed pixel width so they remain readable regardless of how dense the
# historical series is. A small gap separates the bridge from the
# historical series; a slightly larger gap separates bridge bars from
# each other so their labels never collide.
_BRIDGE_BAR_W = 36
_BRIDGE_INTER_GAP = 14
_BRIDGE_SEPARATOR_GAP = 22


@dataclass
class _Bar:
    x: float
    y: float
    w: float
    h: float                       # in pixels after _scale_bars
    color: str
    tt_title: str
    tt_amount: str
    tt_note: str
    label: str | None = None
    raw_value_m: float = 0.0       # millions of USD; used pre-scaling
    overlay_h: float | None = None
    overlay_tt_title: str | None = None
    overlay_tt_amount: str | None = None
    overlay_tt_note: str | None = None


def render(history: CashHistory) -> Markup:
    bars = _layout_bars(history)
    if not bars:
        return Markup('<div class="cash-empty">No cash-position history available.</div>')

    # Y-range can span below zero when current_est is negative (cash
    # already exhausted by burn). We pick symmetric "nice" bounds so the
    # zero line lands on a tick.
    raw_max = max(b.raw_value_m for b in bars)
    raw_min = min(b.raw_value_m for b in bars)
    y_max, y_min = _nice_bounds(raw_max, raw_min)
    bars = _scale_bars(bars, y_max, y_min)

    parts = [f'<svg viewBox="0 0 {_W} {_H}" class="cash-chart" '
             f'xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">']
    parts.extend(_grid_and_axis(y_max, y_min))
    for b in bars:
        parts.append(_render_bar(b))
    parts.extend(_x_labels(bars, y_max, y_min))
    parts.append("</svg>")
    return Markup("".join(parts))


def _layout_bars(h: CashHistory) -> list[_Bar]:
    bridge_cols = _bridge_columns(h)
    n_hist = len(h.series)
    n_bridge = len(bridge_cols)
    if n_hist + n_bridge == 0:
        return []

    # Reserve a fixed-width strip on the right for the bridge columns.
    # Historical bars fill what's left, evenly spaced.
    bridge_zone_w = 0
    if n_bridge:
        bridge_zone_w = (_BRIDGE_SEPARATOR_GAP
                         + n_bridge * _BRIDGE_BAR_W
                         + max(n_bridge - 1, 0) * _BRIDGE_INTER_GAP)
    hist_zone_w = _INNER_W - bridge_zone_w

    bars: list[_Bar] = []
    if n_hist:
        step = hist_zone_w / n_hist
        bar_w = step * 0.72
        gap = step * 0.28
        for i, p in enumerate(h.series):
            x = _PAD_L + i * step + gap / 2
            val_m = p.value_usd / 1e6
            # DT shows just date + "Historical Cash: N.NN". For non-USD
            # FPIs we add a small italic line so the user knows the value
            # was FX-converted — surfaces our deviation from DT honestly.
            native = (f"Converted from {p.native_currency} "
                      f"{p.native_value/1e6:,.2f}M at {p.end.isoformat()} rate"
                      if p.native_currency != "USD" else "")
            bars.append(_Bar(
                x=x, y=0, w=bar_w, h=0, color=_BLUE,
                tt_title=p.end.strftime("%m/%d/%Y"),
                tt_amount=f"Historical Cash: {val_m:,.2f}",
                tt_note=native,
                label=p.end.isoformat(),
                raw_value_m=val_m,
            ))

    bridge_x0 = _PAD_L + hist_zone_w + _BRIDGE_SEPARATOR_GAP
    for j, col in enumerate(bridge_cols):
        x = bridge_x0 + j * (_BRIDGE_BAR_W + _BRIDGE_INTER_GAP)
        bars.append(_Bar(
            x=x, y=0, w=_BRIDGE_BAR_W, h=0, color=col["color"],
            tt_title=col["tt_title"],
            tt_amount=col["tt_amount"],
            tt_note=col["tt_note"],
            label=col["label"],
            raw_value_m=col["value_m"],
            overlay_h=col.get("overlay_m"),
            overlay_tt_title=col.get("overlay_tt_title"),
            overlay_tt_amount=col.get("overlay_tt_amount"),
            overlay_tt_note=col.get("overlay_tt_note"),
        ))
    return bars


def _bridge_columns(h: CashHistory) -> list[dict]:
    if h.latest_cash_usd is None or h.current_cash_est_usd is None:
        return []

    cols: list[dict] = []
    if h.op_cf_prorated_usd is not None and h.op_cf_prorated_usd < 0:
        # OpCF bar's top = latest cash + prorated burn. May be negative
        # when burn exceeds cash on hand — render extends below zero.
        after_burn_m = (h.latest_cash_usd + h.op_cf_prorated_usd) / 1e6
        cols.append({
            "value_m": after_burn_m,
            "color": _MAROON,
            "label": "OpCF",
            # Wording lifted verbatim from the DilutionTracker tooltip.
            "tt_title": "Prorated Operating Cash Flow",
            "tt_amount": f"Amount: {h.op_cf_prorated_usd/1e6:,.2f}",
            "tt_note": "Quarterly CF from operations prorated by days since "
                       "the latest reporting date",
        })

    cur_m = h.current_cash_est_usd / 1e6
    overlay_m = None
    overlay_tt = (None, None, None)
    if h.capital_raised_usd and h.capital_raised_usd > 0:
        overlay_m = h.capital_raised_usd / 1e6
        # Wording lifted verbatim from the DilutionTracker tooltip.
        overlay_tt = (
            "Capital Raise",
            f"Amount: {overlay_m:,.2f}",
            "Capital raise since last reporting date. See Completed "
            "Offerings section for details. Note that figures reported on "
            "this chart are net of underwriting fees, while the Completed "
            "Offerings section reports figures inclusive of fees.",
        )
    cols.append({
        "value_m": cur_m,
        "color": _BLUE,
        "label": "Current Est",
        # DT shows just title + amount here, no note.
        "tt_title": "Current Cash Estimate",
        "tt_amount": f"Amount: {cur_m:,.2f}",
        "tt_note": "",
        "overlay_m": overlay_m,
        "overlay_tt_title": overlay_tt[0],
        "overlay_tt_amount": overlay_tt[1],
        "overlay_tt_note": overlay_tt[2],
    })
    return cols


def _scale_bars(bars: list[_Bar], y_max: float, y_min: float) -> list[_Bar]:
    span = y_max - y_min
    if span <= 0:
        return bars
    zero_y = _y_for_value(0.0, y_max, y_min)
    for b in bars:
        v_y = _y_for_value(b.raw_value_m, y_max, y_min)
        if b.raw_value_m >= 0:
            b.y = v_y
            b.h = zero_y - v_y
        else:
            # Negative bar: hangs down from zero line to the value.
            b.y = zero_y
            b.h = v_y - zero_y
        if b.overlay_h is not None:
            if b.raw_value_m <= 0:
                # No sense drawing a "capital raise" overlay on a bar
                # that's already below zero — drop it.
                b.overlay_h = None
            else:
                # Overlay rides at the top of the (positive) bar; shrink
                # the height into pixel space using the same scale.
                b.overlay_h = (b.overlay_h / span) * _INNER_H
    return bars


def _y_for_value(v: float, y_max: float, y_min: float) -> float:
    """Pixel y-coordinate for a data value, given the chart's y-range."""
    span = y_max - y_min
    if span <= 0:
        return _PAD_T + _INNER_H
    frac = (v - y_min) / span
    return _PAD_T + _INNER_H - frac * _INNER_H


def _render_bar(b: _Bar) -> str:
    parts = [
        f'<g class="cash-bar" '
        f'data-tt-title="{html.escape(b.tt_title)}" '
        f'data-tt-amount="{html.escape(b.tt_amount)}" '
        f'data-tt-note="{html.escape(b.tt_note)}" '
        f'data-tt-color="{b.color}">'
        f'<rect x="{b.x:.1f}" y="{b.y:.1f}" width="{b.w:.1f}" '
        f'height="{b.h:.1f}" fill="{b.color}"/>'
        f'<title>{html.escape(b.tt_title)} — {html.escape(b.tt_amount)}</title>'
        f'</g>'
    ]
    if b.overlay_h and b.overlay_h > 0:
        overlay_h = min(b.overlay_h, b.h)
        parts.append(
            f'<g class="cash-bar" '
            f'data-tt-title="{html.escape(b.overlay_tt_title or "")}" '
            f'data-tt-amount="{html.escape(b.overlay_tt_amount or "")}" '
            f'data-tt-note="{html.escape(b.overlay_tt_note or "")}" '
            f'data-tt-color="{_LIGHT_BLUE}">'
            f'<rect x="{b.x:.1f}" y="{b.y:.1f}" width="{b.w:.1f}" '
            f'height="{overlay_h:.1f}" fill="{_LIGHT_BLUE}"/>'
            f'<title>{html.escape(b.overlay_tt_title or "")} — '
            f'{html.escape(b.overlay_tt_amount or "")}</title>'
            f'</g>'
        )
    return "".join(parts)


def _grid_and_axis(y_max: float, y_min: float) -> list[str]:
    parts: list[str] = []
    ticks = _axis_ticks(y_max, y_min)
    for val in ticks:
        y = _y_for_value(val, y_max, y_min)
        is_zero = abs(val) < 1e-9
        stroke = "#9ca3af" if is_zero else _GRID
        dash = "" if is_zero else ' stroke-dasharray="3,3"'
        parts.append(
            f'<line x1="{_PAD_L}" y1="{y:.1f}" x2="{_W - _PAD_R}" y2="{y:.1f}" '
            f'stroke="{stroke}" stroke-width="1"{dash}/>'
        )
        parts.append(
            f'<text x="{_PAD_L - 6}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="10" fill="{_AXIS_TEXT}">{_fmt_axis(val)}</text>'
        )
    parts.append(
        f'<text x="{_PAD_L - 44}" y="{_PAD_T + _INNER_H / 2:.1f}" '
        f'transform="rotate(-90 {_PAD_L - 44} {_PAD_T + _INNER_H / 2:.1f})" '
        f'text-anchor="middle" font-size="10" fill="{_AXIS_TEXT}">Millions of USD</text>'
    )
    return parts


def _axis_ticks(y_max: float, y_min: float) -> list[float]:
    """Ticks at every "step" between y_min and y_max inclusive. Step is
    derived from the bounds (`_nice_bounds` guarantees the span divides
    evenly into 1–4 steps). Zero always lands on a tick when both signs
    are present."""
    import math
    span = y_max - y_min
    if span <= 0:
        return [y_min, y_max]
    # Find the step that produces an integer number of intervals (≤4).
    for n in (4, 3, 2, 1):
        step = span / n
        return [y_min + step * i for i in range(n + 1)]
    return [y_min, y_max]


def _x_labels(bars: list[_Bar], y_max: float, y_min: float) -> list[str]:
    """Thin labels so they don't collide.

    Bridge labels (OpCF / Current Est) are always shown. Historical
    bars are sampled at a stride chosen to leave ~50px between labels.
    Labels sit below the chart's bottom regardless of where zero falls.
    """
    parts: list[str] = []
    y = _PAD_T + _INNER_H + 14

    # Keep all bridge labels (non-date) regardless of crowding.
    bridge_idxs = {i for i, b in enumerate(bars)
                   if b.label and not _looks_like_date(b.label)}

    # Pick historical-date labels with min ~70px between them.
    hist_idxs = [i for i, b in enumerate(bars)
                 if b.label and _looks_like_date(b.label)]
    chosen_hist: list[int] = []
    last_x = -1e9
    for i in hist_idxs:
        cx = bars[i].x + bars[i].w / 2
        if cx - last_x >= 70:
            chosen_hist.append(i)
            last_x = cx

    keep = bridge_idxs | set(chosen_hist)
    # If a bridge label is too close to the last chosen historical
    # label, drop the historical label rather than the bridge one.
    bridge_xs = [bars[i].x + bars[i].w / 2 for i in bridge_idxs]
    filtered_keep = set(keep)
    for i in chosen_hist:
        cx = bars[i].x + bars[i].w / 2
        if any(abs(cx - bx) < 50 for bx in bridge_xs):
            filtered_keep.discard(i)

    for i, b in enumerate(bars):
        if i not in filtered_keep or not b.label:
            continue
        cx = b.x + b.w / 2
        parts.append(
            f'<text x="{cx:.1f}" y="{y}" text-anchor="middle" '
            f'font-size="10" fill="{_AXIS_TEXT}">{html.escape(b.label)}</text>'
        )
    return parts


def _looks_like_date(s: str) -> bool:
    if len(s) != 10:
        return False
    try:
        date.fromisoformat(s)
        return True
    except ValueError:
        return False


def _nice_ceiling(v: float) -> float:
    if v <= 0:
        return 1.0
    import math
    exp = math.floor(math.log10(v))
    base = 10 ** exp
    for m in (1, 2, 2.5, 5, 10):
        c = m * base
        if c >= v:
            return c
    return 10 * base


def _nice_bounds(raw_max: float, raw_min: float) -> tuple[float, float]:
    """Pick (y_max, y_min) so that zero anchors the tick grid.

    Picks a tick step from a "nice" {1,2,2.5,5}·10^k ladder such that
    the positive-side ticks above zero plus the negative-side ticks
    below zero total at most 4 intervals (5 grid lines).
    """
    if raw_min >= 0:
        return _nice_ceiling(max(raw_max, 0.0)), 0.0

    import math
    pos = max(raw_max, 0.0)
    neg = abs(raw_min)
    target_step = max(pos, neg) / 3.0  # start: enough headroom for ~3 ticks on bigger side
    exp = math.floor(math.log10(target_step)) if target_step > 0 else 0
    base = 10 ** exp
    candidates: list[float] = []
    for mult in (1, 2, 2.5, 5):
        candidates.append(mult * base)
    for mult in (1, 2, 2.5, 5):
        candidates.append(mult * base * 10)
    for step in candidates:
        if step <= 0:
            continue
        n_pos = max(1, math.ceil(pos / step)) if pos > 0 else 0
        n_neg = max(1, math.ceil(neg / step))
        if n_pos + n_neg <= 4:
            return n_pos * step, -n_neg * step
    return _nice_ceiling(pos), -_nice_ceiling(neg)


def _fmt_axis(v: float) -> str:
    if abs(v) < 1e-9:
        return "0"
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1000:
        body = f"{a/1000:,.1f}K"
    elif a >= 10:
        body = f"{a:,.0f}"
    else:
        body = f"{a:,.1f}"
    return f"{sign}{body}"
