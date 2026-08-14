"""Deterministic dilution-risk badges — the DT-style header strip.

Five badges across the top of the ticker page: a transparent 0-100
Overall Dilution Risk score plus the four drivers it blends:

  Offering Ability   can they sell new stock at will, right now?
                     (active shelf + pending S-1 raisable as % of
                      market cap; WKSI sentinel = auto-High; a live
                      ATM / equity line escalates one band)
  Overhang           investor-held paper that converts into the float
                     (warrants + convertibles + preferred, % of O/S)
  Dilution History   split-adjusted O/S growth over the past 3 years
  Cash Need          months of runway at the current operating burn

Everything is computed from data the payload build already fetches (the
card projections, finviz fundamentals, CashHistory) plus one yfinance
share-count series for the history badge. No LLM involvement — pure
arithmetic, every number traceable to a card rendered below the strip.

Where this deliberately diverges from DilutionTracker's definitions:
  * Offering Ability is sized against MARKET CAP, not absolute dollars
    (DT's $1M/$20M bands rate a $20M shelf identically for a $40M
    company and a $2B one).
  * ATM / equity-line capacity is NOT summed into capacity or overhang
    — an ATM is usually a takedown under a shelf we already counted,
    so summing double-counts. Instead a live ATM/ELOC escalates
    Offering Ability one band: at-will selling is already running.
  * The composite is transparent: fixed weights plus an explicit
    interaction bump when Cash Need and Offering Ability are both High
    (the "imminent discounted offering" setup behind overnight drops).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache

from db import get_conn

log = logging.getLogger(__name__)

# Matches shelf_status.WKSI_UNLIMITED_SHELF_CAPACITY_USD — the ledger
# sentinel for pay-as-you-go S-3ASR/F-3ASR shelves with no headline cap.
WKSI_SENTINEL_USD = 999_999_999

# ── Band thresholds ──────────────────────────────────────────────────
# Offering Ability: capacity as % of market cap, with an absolute floor
# below which no meaningful discounted offering is possible.
_OFFERING_MED_PCT = 5.0
_OFFERING_HIGH_PCT = 25.0
_OFFERING_FLOOR_USD = 1_000_000.0

# Overhang: % of shares outstanding (same bands as DT).
_OVERHANG_MED_PCT = 20.0
_OVERHANG_HIGH_PCT = 50.0

# Dilution History: 3-year split-adjusted O/S growth % (same bands as DT).
_HISTORY_MED_PCT = 30.0
_HISTORY_HIGH_PCT = 100.0
_HISTORY_YEARS = 3
# Require at least ~1.5y of share-count history before scoring a badge
# that claims to be a 3-year test — recent IPOs get "—", not a fake Low.
_HISTORY_MIN_WINDOW_DAYS = 540

# Cash Need: months of runway (same bands as DT).
_RUNWAY_LOW_MONTHS = 24.0
_RUNWAY_HIGH_MONTHS = 6.0

# Composite: weights sum to 1.0. The two forward-looking "will they /
# can they raise NOW" drivers dominate; overhang is potential supply;
# history is a backward-looking prior.
_WEIGHTS = {
    "offering_ability": 0.30,
    "overhang": 0.25,
    "history": 0.15,
    "cash_need": 0.30,
}
_INTERACTION_BUMP = 15  # Cash Need High AND Offering Ability High

# Sub-badge display band → score interval (the 0-100 driver scores stay
# consistent with the displayed Low/Medium/High by construction).
_BAND_RANGES = {"low": (0, 33), "medium": (34, 66), "high": (67, 100)}
_BAND_TEXT = {"low": "Low", "medium": "Medium", "high": "High"}

# Overall score → 5-band heat label. (min_score, css_class, label)
_OVERALL_BANDS = (
    (80, "severe", "Severe"),
    (60, "high", "High"),
    (40, "moderate", "Moderate"),
    (20, "low", "Low"),
    (0, "minimal", "Minimal"),
)


@dataclass(frozen=True)
class Badge:
    key: str
    label: str
    description: str
    band: str | None            # 'low' | 'medium' | 'high' | None
    band_text: str              # 'Low' / 'Medium' / 'High' / '—'
    score: int | None           # 0-100, feeds the composite
    detail: tuple[str, ...]     # ticker-specific driver lines
    legend: tuple[tuple[str, str, str], ...]  # (css, pill text, meaning)


@dataclass(frozen=True)
class BadgeSet:
    overall_score: int | None
    overall_band: str | None    # minimal|low|moderate|high|severe
    overall_label: str          # 'Severe' … 'Minimal' / '—'
    partial: bool               # composite missing ≥1 driver
    interaction: bool           # High×High bump applied
    description: str
    detail: tuple[str, ...]
    legend: tuple[tuple[str, str, str], ...]
    drivers: tuple[Badge, ...]


# ── formatting helpers (compact, header-strip friendly) ─────────────
def _usd(x: float | None) -> str:
    if x is None:
        return "—"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e9:
        return f"{sign}${x / 1e9:.2f}B"
    if x >= 1e6:
        return f"{sign}${x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{sign}${x / 1e3:.0f}K"
    return f"{sign}${x:.0f}"


def _sh(x: float | None) -> str:
    if x is None:
        return "—"
    if x >= 1e9:
        return f"{x / 1e9:.2f}B"
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"


def _pct0(x: float) -> str:
    return f"{x:,.0f}%"


def _band_score(band: str, frac: float) -> int:
    lo, hi = _BAND_RANGES[band]
    return int(round(lo + max(0.0, min(1.0, frac)) * (hi - lo)))


def _mk(key: str, label: str, description: str,
        band: str | None, frac: float,
        detail: list[str],
        legend: tuple[tuple[str, str, str], ...]) -> Badge:
    return Badge(
        key=key, label=label, description=description,
        band=band,
        band_text=_BAND_TEXT.get(band, "—"),
        score=_band_score(band, frac) if band is not None else None,
        detail=tuple(detail),
        legend=legend,
    )


def _is_wksi_amount(v) -> bool:
    try:
        return v is not None and int(round(float(v))) == WKSI_SENTINEL_USD
    except (TypeError, ValueError):
        return False


# ── Offering Ability ─────────────────────────────────────────────────
_OFFERING_LEGEND = (
    ("low", "Low", "No shelf / pending S-1, capacity under $1M, "
                   "or under 5% of market cap"),
    ("medium", "Medium", "Capacity 5–25% of market cap"),
    ("high", "High", "Capacity over 25% of market cap, a WKSI "
                     "automatic shelf, or escalated by a live ATM"),
)

_OFFERING_DESC = (
    "Ability to sell new shares at will via an effective shelf or a "
    "pending S-1, sized against market cap — a $20M shelf is fatal for "
    "a $40M company and trivial for a $2B one. A live ATM or equity "
    "line escalates one level (at-will selling is already running; its "
    "dollar capacity is not re-added — it usually draws down the same "
    "shelf). Higher = a discounted offering can hit at any time."
)


def _offering_ability(cards: dict, mcap: float | None) -> Badge:
    shelves = cards.get("shelf") or []
    s1s = [c for c in (cards.get("s1_offering") or [])
           if c.get("s1_status") in ("pending", "effective")]
    atms = cards.get("atm") or []
    elocs = [c for c in (cards.get("equity_line") or [])
             if not c.get("terminated")]

    wksi = any(_is_wksi_amount(c.get("current_raisable_amount"))
               or _is_wksi_amount(c.get("total_shelf_capacity"))
               for c in shelves)

    shelf_cap = 0.0
    unknown = 0
    for c in shelves:
        v = c.get("current_raisable_amount")
        if _is_wksi_amount(v):
            continue
        if v is None:
            v = c.get("total_shelf_capacity")
        if v is None:
            unknown += 1
            continue
        shelf_cap += float(v)
    s1_cap = 0.0
    for c in s1s:
        v = c.get("anticipated_deal_size")
        if v is None:
            unknown += 1
            continue
        s1_cap += float(v)
    capacity = shelf_cap + s1_cap

    # ATMs: use the IB6-capped live raisable (raisable_capped), not the
    # contractual remaining_capacity DT displays — the escalator measures
    # what can actually hit the tape. ELOCs carry no IB6 cap field.
    atm_live = (sum(float(c.get("raisable_capped",
                                c.get("remaining_capacity")) or 0)
                    for c in atms)
                + sum(float(c.get("remaining_capacity") or 0) for c in elocs))
    # Same materiality grammar as the main bands: $1M absolute floor OR
    # the 5%-of-market-cap Low/Medium boundary.
    escalator = (atm_live >= _OFFERING_FLOOR_USD
                 or (mcap and atm_live >= mcap * _OFFERING_MED_PCT / 100.0))

    detail: list[str] = []
    if shelves:
        if wksi:
            detail.append("WKSI automatic shelf (S-3ASR/F-3ASR) — "
                          "effectively unlimited capacity")
        if shelf_cap > 0 or not wksi:
            noun = "shelves" if len(shelves) > 1 else "shelf"
            detail.append(f"{len(shelves)} active {noun}: "
                          f"{_usd(shelf_cap)} raisable now")
        if any(c.get("baby_shelf_restriction") == "Yes" for c in shelves):
            detail.append("Baby-shelf (I.B.6) cap applied — raisable is "
                          "limited to ⅓ of public float per 12 months")
    if s1s:
        detail.append(f"{len(s1s)} pending S-1/F-1: {_usd(s1_cap)} anticipated")
    if not shelves and not s1s:
        detail.append("No active shelf or pending S-1 on file")
    if unknown:
        detail.append(f"{unknown} registration"
                      f"{'s' if unknown > 1 else ''} with undisclosed "
                      f"capacity (not counted)")

    if wksi:
        band, frac = "high", 1.0
    elif capacity <= 0:
        band, frac = "low", 0.0
    elif mcap:
        ratio = capacity / mcap * 100.0
        if capacity < _OFFERING_FLOOR_USD:
            band, frac = "low", min(0.5, ratio / _OFFERING_MED_PCT)
            detail.append(f"Capacity under {_usd(_OFFERING_FLOOR_USD)} — "
                          f"too small for a meaningful raise")
        elif ratio < _OFFERING_MED_PCT:
            band, frac = "low", ratio / _OFFERING_MED_PCT
        elif ratio <= _OFFERING_HIGH_PCT:
            band, frac = "medium", ((ratio - _OFFERING_MED_PCT)
                                    / (_OFFERING_HIGH_PCT - _OFFERING_MED_PCT))
        else:
            band, frac = "high", min(1.0, (ratio - _OFFERING_HIGH_PCT) / 75.0)
        detail.append(f"= {_pct0(ratio)} of {_usd(mcap)} market cap")
    else:
        band, frac = None, 0.0
        detail.append("Market cap unavailable — cannot size the capacity")

    if escalator:
        if band is None:
            band, frac = "medium", 0.5
        elif band == "low":
            band = "medium"
        elif band == "medium":
            band = "high"
        else:
            frac = min(1.0, frac + 0.33)
        detail.append(f"Live ATM/equity line, {_usd(atm_live)} remaining "
                      f"— escalated one level")

    return _mk("offering_ability", "Offering Ability", _OFFERING_DESC,
               band, frac, detail, _OFFERING_LEGEND)


# ── Overhang ─────────────────────────────────────────────────────────
_OVERHANG_LEGEND = (
    ("low", "Low", "Under 20% of shares outstanding"),
    ("medium", "Medium", "20–50% of shares outstanding"),
    ("high", "High", "Over 50% of shares outstanding"),
)

_OVERHANG_DESC = (
    "Investor-held securities that convert into the float — warrants, "
    "convertible notes, convertible preferred — as % of shares "
    "outstanding. If everything converts, this is how much your slice "
    "shrinks. Shelf / ATM / equity-line capacity is counted under "
    "Offering Ability instead (an ATM is usually a takedown of the "
    "same shelf — summing both would double-count). Pre-funded "
    "warrants are excluded."
)


def _overhang(cards: dict, latest_os: float | None) -> Badge:
    ws = cards.get("warrant") or []
    cs = cards.get("convertible") or []
    ps = cards.get("convertible_preferred") or []

    w_sh = sum(float(c.get("remaining_outstanding") or 0) for c in ws)
    c_sh = sum(float(c.get("remaining_shares_issuable") or 0) for c in cs)
    p_sh = sum(float(c.get("remaining_shares_issuable") or 0) for c in ps)
    total = w_sh + c_sh + p_sh
    # $-denominated paper with no disclosed conversion price can't be
    # share-counted; surface the exclusion instead of silently skipping.
    missing = sum(1 for c in list(cs) + list(ps)
                  if c.get("remaining_shares_issuable") is None
                  and (c.get("principal_remaining") or 0) > 0)

    detail: list[str] = []
    parts = []
    if w_sh:
        parts.append(f"warrants {_sh(w_sh)}")
    if c_sh:
        parts.append(f"convertibles {_sh(c_sh)}")
    if p_sh:
        parts.append(f"preferred {_sh(p_sh)}")

    if total <= 0:
        band, frac = "low", 0.0
        detail.append("No warrants, convertible notes, or convertible "
                      "preferred outstanding")
    elif not latest_os:
        band, frac = None, 0.0
        detail.append(" + ".join(parts)
                      + f" = {_sh(total)} potential new shares")
        detail.append("Shares outstanding unavailable — cannot compute %")
    else:
        pct = total / latest_os * 100.0
        if pct < _OVERHANG_MED_PCT:
            band, frac = "low", pct / _OVERHANG_MED_PCT
        elif pct <= _OVERHANG_HIGH_PCT:
            band, frac = "medium", ((pct - _OVERHANG_MED_PCT)
                                    / (_OVERHANG_HIGH_PCT - _OVERHANG_MED_PCT))
        else:
            band, frac = "high", min(1.0, (pct - _OVERHANG_HIGH_PCT) / 50.0)
        detail.append(" + ".join(parts)
                      + f" = {_sh(total)} potential new shares")
        detail.append(f"= {_pct0(pct)} of {_sh(latest_os)} shares outstanding")
    if missing:
        detail.append(f"{missing} instrument{'s' if missing > 1 else ''} "
                      f"with undisclosed conversion price excluded")

    return _mk("overhang", "Overhang", _OVERHANG_DESC,
               band, frac, detail, _OVERHANG_LEGEND)


# ── Dilution History ─────────────────────────────────────────────────
_HISTORY_LEGEND = (
    ("low", "Low", "Under 30% O/S growth over the past 3 years"),
    ("medium", "Medium", "30–100% O/S growth"),
    ("high", "High", "Over 100% O/S growth"),
)

_HISTORY_DESC = (
    "Split-adjusted growth in shares outstanding over the past 3 "
    "years. Reverse splits are normalized out — a 1-for-10 reverse "
    "split followed by re-dilution back to the old count is +900%, not "
    "0%. Past dilution is the best predictor of future dilution."
)


@lru_cache(maxsize=256)
def _os_growth_3y(cik: int, ticker: str, as_of_iso: str):
    """3-year split-adjusted O/S growth from yfinance's reported
    share-count series. Returns (growth_pct, anchor_date, anchor_adj,
    latest_count, latest_date, n_splits) or None when unavailable.

    yfinance's get_shares_full returns point-in-time *reported* counts
    (it jumps at splits, no retroactive adjustment), so the anchor is
    converted to the current share basis via dilution_splits: a
    1-for-10 reverse (pre=10, post=1) multiplies the old count by 1/10.
    """
    try:
        import yfinance as yf
        as_of = date.fromisoformat(as_of_iso)
        start = as_of - timedelta(days=_HISTORY_YEARS * 365 + 90)
        series = yf.Ticker(ticker).get_shares_full(start=start.isoformat())
    except Exception as e:
        log.debug("shares history failed for %s: %s", ticker, e)
        return None
    if series is None or len(series) == 0:
        return None

    pts: list[tuple[date, float]] = []
    try:
        for ts, v in series.items():
            if v is None or float(v) <= 0:
                continue
            pts.append((ts.date(), float(v)))
    except Exception as e:
        log.debug("shares series parse failed for %s: %s", ticker, e)
        return None
    if len(pts) < 2:
        return None
    pts.sort()

    target = date.fromisoformat(as_of_iso) - timedelta(days=_HISTORY_YEARS * 365)
    anchor = None
    for d, v in pts:
        if d <= target:
            anchor = (d, v)
        else:
            break
    if anchor is None:
        anchor = pts[0]
    latest = pts[-1]
    if (latest[0] - anchor[0]).days < _HISTORY_MIN_WINDOW_DAYS:
        return None

    factor, n_splits = 1.0, 0
    try:
        with get_conn() as conn:
            rows = conn.execute(
                """SELECT pre, post FROM dilution_splits
                   WHERE cik = ? AND effective_date > ?
                     AND effective_date <= ?""",
                (int(cik), anchor[0].isoformat(), latest[0].isoformat()),
            ).fetchall()
        for r in rows:
            if r["pre"]:
                factor *= float(r["post"]) / float(r["pre"])
                n_splits += 1
    except Exception as e:
        log.warning("splits lookup failed for cik %s: %s", cik, e)

    anchor_adj = anchor[1] * factor
    if anchor_adj <= 0:
        return None
    growth = (latest[1] - anchor_adj) / anchor_adj * 100.0
    return growth, anchor[0], anchor_adj, latest[1], latest[0], n_splits


def _raised_last_24mo(cik: int, as_of: date) -> float | None:
    try:
        with get_conn() as conn:
            row = conn.execute(
                """SELECT COALESCE(SUM(amount_usd), 0)
                   FROM dilution_ledger_drawdowns
                   WHERE cik = ? AND event_date > ?
                     AND amount_usd IS NOT NULL""",
                (int(cik), (as_of - timedelta(days=730)).isoformat()),
            ).fetchone()
        return float(row[0] or 0.0)
    except Exception as e:
        log.warning("raised-24mo lookup failed for cik %s: %s", cik, e)
        return None


def _dilution_history(cik: int, ticker: str | None, as_of: date) -> Badge:
    detail: list[str] = []
    g = (_os_growth_3y(int(cik), ticker, as_of.isoformat())
         if ticker else None)

    if g is None:
        band, frac = None, 0.0
        detail.append("Share-count history unavailable (or under ~18 "
                      "months listed)")
    else:
        growth, a_date, a_adj, l_cnt, l_date, n_splits = g
        if growth < _HISTORY_MED_PCT:
            band, frac = "low", max(0.0, growth) / _HISTORY_MED_PCT
        elif growth <= _HISTORY_HIGH_PCT:
            band, frac = "medium", ((growth - _HISTORY_MED_PCT)
                                    / (_HISTORY_HIGH_PCT - _HISTORY_MED_PCT))
        else:
            band, frac = "high", min(1.0, (growth - _HISTORY_HIGH_PCT) / 200.0)
        adj_note = " (split-adj)" if n_splits else ""
        detail.append(
            f"O/S {a_date.strftime('%b %Y')}: {_sh(a_adj)}{adj_note} → "
            f"{l_date.strftime('%b %Y')}: {_sh(l_cnt)} ({growth:+,.0f}%)")
        if n_splits:
            detail.append(f"Normalized for {n_splits} "
                          f"split{'s' if n_splits > 1 else ''} in the window")

    raised = _raised_last_24mo(cik, as_of)
    if raised:
        detail.append(f"{_usd(raised)} raised in the last 24 months")

    return _mk("history", "Dilution History", _HISTORY_DESC,
               band, frac, detail, _HISTORY_LEGEND)


# ── Cash Need ────────────────────────────────────────────────────────
_CASH_LEGEND = (
    ("low", "Low", "Operating cash-flow positive, or over 24 months "
                   "of runway"),
    ("medium", "Medium", "6–24 months of runway"),
    ("high", "High", "Under 6 months of runway"),
)

_CASH_DESC = (
    "Months of cash left at the current operating burn — estimated "
    "current cash (latest balance sheet, bridged with burn and capital "
    "raised since) divided by monthly burn. Companies that need cash "
    "and have the means to raise it, will."
)


def _cash_need(cash) -> Badge:
    detail: list[str] = []
    if cash is None or (cash.months_of_cash is None
                        and cash.op_cf_quarterly_usd is None):
        detail.append("Cash-flow data unavailable")
        return _mk("cash_need", "Cash Need", _CASH_DESC,
                   None, 0.0, detail, _CASH_LEGEND)

    op = cash.op_cf_quarterly_usd
    m = cash.months_of_cash
    if cash.current_cash_est_usd is not None:
        detail.append(f"Est. current cash {_usd(cash.current_cash_est_usd)}")
    if op is not None:
        detail.append(f"Operating burn {_usd(-op)}/quarter" if op < 0
                      else f"Operating CF positive: +{_usd(op)}/quarter")
    if m is not None:
        detail.append(f"≈ {m:,.1f} months of runway")
    if cash.latest_period_end:
        detail.append(f"Balance sheet as of {cash.latest_period_end}, "
                      f"bridged with burn + raises since")

    if op is not None and op >= 0:
        band, frac = "low", 0.0
    elif m is None:
        band, frac = None, 0.0
    elif m > _RUNWAY_LOW_MONTHS:
        band, frac = "low", max(0.0, min(1.0, (36.0 - m) / 12.0))
    elif m >= _RUNWAY_HIGH_MONTHS:
        band, frac = "medium", ((_RUNWAY_LOW_MONTHS - m)
                                / (_RUNWAY_LOW_MONTHS - _RUNWAY_HIGH_MONTHS))
    else:
        band, frac = "high", min(1.0, (_RUNWAY_HIGH_MONTHS - m)
                                 / _RUNWAY_HIGH_MONTHS)

    return _mk("cash_need", "Cash Need", _CASH_DESC,
               band, frac, detail, _CASH_LEGEND)


# ── Composite ────────────────────────────────────────────────────────
_OVERALL_LEGEND = (
    ("minimal", "Minimal", "0–19 — no meaningful dilution setup"),
    ("low", "Low", "20–39"),
    ("moderate", "Moderate", "40–59"),
    ("high", "High", "60–79"),
    ("severe", "Severe", "80–100 — needs cash and can print it"),
)

_OVERALL_DESC = (
    "0–100 composite of the four drivers: Offering Ability 30%, Cash "
    "Need 30%, Overhang 25%, Dilution History 15% — plus a +15 bump "
    "when the company both needs cash (under 6 months) and can raise "
    "at will, the exact setup behind surprise discounted offerings. "
    "Higher = higher probability the share count grows soon."
)


def compute_badges(cik: int, *, fund: dict | None = None,
                   latest_os: float | None = None,
                   cards: dict | None = None,
                   cash=None,
                   as_of: date | None = None) -> BadgeSet:
    """Build the five-badge strip for one ticker.

    `fund` / `latest_os` / `cards` / `cash` are the objects the ticker
    route already fetches — passed in so nothing is fetched twice.
    """
    cards = cards or {}
    as_of = as_of or date.today()
    mcap = (fund or {}).get("market_cap")
    ticker = (fund or {}).get("ticker")

    offering = _offering_ability(cards, mcap)
    overhang = _overhang(cards, latest_os)
    history = _dilution_history(cik, ticker, as_of)
    cash_need = _cash_need(cash)
    drivers = (offering, overhang, history, cash_need)

    scored = [b for b in drivers if b.score is not None]
    detail: list[str] = []
    interaction = False
    overall_score = overall_band = None
    overall_label = "—"

    if scored:
        wsum = sum(_WEIGHTS[b.key] for b in scored)
        raw = sum(_WEIGHTS[b.key] * b.score for b in scored) / wsum
        interaction = (cash_need.band == "high" and offering.band == "high")
        if interaction:
            raw = min(100.0, raw + _INTERACTION_BUMP)
        overall_score = int(round(raw))
        for mn, css, lab in _OVERALL_BANDS:
            if overall_score >= mn:
                overall_band, overall_label = css, lab
                break

        for b in drivers:
            w = int(_WEIGHTS[b.key] * 100)
            val = f"{b.band_text} ({b.score})" if b.score is not None else "—"
            detail.append(f"{b.label}: {val} · weight {w}%")
        if interaction:
            detail.append("Needs cash (<6 mo) AND can raise at will — "
                          f"+{_INTERACTION_BUMP} interaction bump")
        missing = [b.label for b in drivers if b.score is None]
        if missing:
            detail.append(f"Partial: based on {len(scored)} of 4 drivers "
                          f"({', '.join(missing)} unavailable)")
    else:
        detail.append("No drivers computable — insufficient data")

    return BadgeSet(
        overall_score=overall_score,
        overall_band=overall_band,
        overall_label=overall_label,
        partial=len(scored) < len(drivers),
        interaction=interaction,
        description=_OVERALL_DESC,
        detail=tuple(detail),
        legend=_OVERALL_LEGEND,
        drivers=drivers,
    )
