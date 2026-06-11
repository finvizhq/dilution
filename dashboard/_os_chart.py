"""Pure-Python SVG renderer for the Historical O/S & Potential Dilution chart.

DilutionTracker-style: quarterly split-adjusted shares-outstanding bars
(dark blue) followed, after a visual gap, by a single "Fully Diluted"
stacked bar — latest O/S as the dark base plus one colored segment per
potential-dilution source (warrants, convertible notes, preferred, ATM,
equity line, pending S-1).

Bars reuse the cash chart's `.cash-bar` class and data-* attributes so
`_os_chart.html` can drive the exact same hover-tooltip handler.
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import date

from markupsafe import Markup

from dilution.os_history import OsHistory, StackSegment

_BLUE = "#1f4e79"
_GRID = "#e5e7eb"
_AXIS_TEXT = "#6b7280"

# Segment palette — fixed-share paper in warm/purple tones,
# price-dependent estimates in the cooler/lighter tones.
SEG_COLORS = {
    "warrant": "#edb95e",
    "convertible": "#8e6bb8",
    "preferred": "#c46e9c",
    "atm": "#9ec5e8",
    "equity_line": "#62a87c",
    "s1": "#e08b6d",
}

# Canvas geometry (matches the cash chart).
_W, _H = 920, 320
_PAD_L, _PAD_R, _PAD_T, _PAD_B = 56, 16, 16, 40
_INNER_W = _W - _PAD_L - _PAD_R
_INNER_H = _H - _PAD_T - _PAD_B

_FD_BAR_W = 44
_FD_SEPARATOR_GAP = 26


@dataclass
class _Rect:
    x: float
    y: float
    w: float
    h: float
    color: str
    tt_title: str
    tt_amount: str
    tt_note: str


def _sh_m(x: float) -> str:
    """Share count in millions with sensible precision."""
    m = x / 1e6
    if m >= 100:
        return f"{m:,.0f}M"
    if m >= 1:
        return f"{m:,.2f}M"
    return f"{m:,.4f}M"


def render(history: OsHistory,
           latest_os: float | None,
           latest_os_note: str,
           stack: list[StackSegment]) -> Markup:
    pts = history.series
    has_fd = latest_os is not None and latest_os > 0
    if not pts and not has_fd:
        return Markup('<div class="cash-empty">No shares-outstanding '
                      'history available.</div>')

    fd_total = (latest_os + sum(s.shares for s in stack)) if has_fd else 0.0
    raw_max_m = max([p.shares / 1e6 for p in pts] + [fd_total / 1e6, 1e-6])
    y_max = _nice_ceiling(raw_max_m)

    fd_zone_w = (_FD_SEPARATOR_GAP + _FD_BAR_W) if has_fd else 0
    hist_zone_w = _INNER_W - fd_zone_w
    zero_y = _PAD_T + _INNER_H

    def px(v_shares: float) -> float:
        return (v_shares / 1e6) / y_max * _INNER_H

    rects: list[_Rect] = []
    labels: list[tuple[float, str, bool]] = []  # (center_x, text, is_fd)

    if pts:
        step = hist_zone_w / len(pts)
        bar_w = step * 0.72
        gap = step * 0.28
        for i, p in enumerate(pts):
            x = _PAD_L + i * step + gap / 2
            h = px(p.shares)
            if p.carried:
                note = (f"No filing this quarter — carried from the "
                        f"{p.form} cover dated {p.source_date.isoformat()}")
            else:
                note = f"{p.form} cover dated {p.source_date.isoformat()}"
                if p.split_adjusted:
                    note += (f"; as-reported {p.raw_shares:,.0f}, "
                             f"split-adjusted to today's share basis")
            rects.append(_Rect(
                x=x, y=zero_y - h, w=bar_w, h=h, color=_BLUE,
                tt_title=p.quarter_end.strftime("%m/%d/%Y"),
                tt_amount=f"O/S: {_sh_m(p.shares)}",
                tt_note=note,
            ))
            labels.append((x + bar_w / 2, p.quarter_end.isoformat(), False))

    if has_fd:
        x = _PAD_L + hist_zone_w + _FD_SEPARATOR_GAP
        base_h = px(latest_os)
        rects.append(_Rect(
            x=x, y=zero_y - base_h, w=_FD_BAR_W, h=base_h, color=_BLUE,
            tt_title="Fully Diluted — base",
            tt_amount=f"Latest O/S: {_sh_m(latest_os)}",
            tt_note=latest_os_note,
        ))
        y_cursor = zero_y - base_h
        for s in stack:
            h = px(s.shares)
            y_cursor -= h
            rects.append(_Rect(
                x=x, y=y_cursor, w=_FD_BAR_W, h=h,
                color=SEG_COLORS.get(s.key, "#999"),
                tt_title=f"{s.label} — potential dilution",
                tt_amount=f"+{_sh_m(s.shares)} shares",
                tt_note=s.note,
            ))
        labels.append((x + _FD_BAR_W / 2, "Fully Diluted", True))

    parts = [f'<svg viewBox="0 0 {_W} {_H}" class="cash-chart" '
             f'xmlns="http://www.w3.org/2000/svg" '
             f'preserveAspectRatio="xMidYMid meet">']
    parts.extend(_grid_and_axis(y_max))
    for r in rects:
        parts.append(
            f'<g class="cash-bar" '
            f'data-tt-title="{html.escape(r.tt_title)}" '
            f'data-tt-amount="{html.escape(r.tt_amount)}" '
            f'data-tt-note="{html.escape(r.tt_note)}" '
            f'data-tt-color="{r.color}">'
            f'<rect x="{r.x:.1f}" y="{r.y:.1f}" width="{r.w:.1f}" '
            f'height="{max(r.h, 0.0):.1f}" fill="{r.color}"/>'
            f'<title>{html.escape(r.tt_title)} — '
            f'{html.escape(r.tt_amount)}</title>'
            f'</g>'
        )
    parts.extend(_x_labels(labels))
    parts.append("</svg>")
    return Markup("".join(parts))


def _grid_and_axis(y_max: float) -> list[str]:
    parts: list[str] = []
    n_ticks = 4
    for i in range(n_ticks + 1):
        val = y_max * i / n_ticks
        y = _PAD_T + _INNER_H - (val / y_max) * _INNER_H
        is_zero = i == 0
        stroke = "#9ca3af" if is_zero else _GRID
        dash = "" if is_zero else ' stroke-dasharray="3,3"'
        parts.append(
            f'<line x1="{_PAD_L}" y1="{y:.1f}" x2="{_W - _PAD_R}" '
            f'y2="{y:.1f}" stroke="{stroke}" stroke-width="1"{dash}/>'
        )
        parts.append(
            f'<text x="{_PAD_L - 6}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="10" fill="{_AXIS_TEXT}">{_fmt_axis(val)}</text>'
        )
    parts.append(
        f'<text x="{_PAD_L - 44}" y="{_PAD_T + _INNER_H / 2:.1f}" '
        f'transform="rotate(-90 {_PAD_L - 44} {_PAD_T + _INNER_H / 2:.1f})" '
        f'text-anchor="middle" font-size="10" fill="{_AXIS_TEXT}">'
        f'Millions of Shares</text>'
    )
    return parts


def _x_labels(labels: list[tuple[float, str, bool]]) -> list[str]:
    """Date labels sampled at ~70px spacing; 'Fully Diluted' always kept,
    with the nearest date label dropped if they would collide."""
    parts: list[str] = []
    y = _PAD_T + _INNER_H + 14

    fd = [(cx, text) for cx, text, is_fd in labels if is_fd]
    dates = [(cx, text) for cx, text, is_fd in labels if not is_fd]

    chosen: list[tuple[float, str]] = []
    last_x = -1e9
    for cx, text in dates:
        if cx - last_x >= 70:
            chosen.append((cx, text))
            last_x = cx
    fd_xs = [cx for cx, _ in fd]
    chosen = [(cx, t) for cx, t in chosen
              if not any(abs(cx - fx) < 60 for fx in fd_xs)]

    for cx, text in chosen + fd:
        parts.append(
            f'<text x="{cx:.1f}" y="{y}" text-anchor="middle" '
            f'font-size="10" fill="{_AXIS_TEXT}">{html.escape(text)}</text>'
        )
    return parts


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


def _fmt_axis(v: float) -> str:
    if abs(v) < 1e-9:
        return "0"
    if v >= 1000:
        return f"{v/1000:,.1f}K"
    if v >= 10:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:,.1f}"
    return f"{v:,.2f}"
