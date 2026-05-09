"""Card projection — ledger rows → DT-style dashboard cards.

Replaces the ~2650-line clustering machinery in dilution/instrument_cards.py.
Each projector is a thin SELECT + per-type field mapping; the heavy
lifting (instrument identity, amendment tracking, lifecycle) lives in
the ledger now.

Public surface (matches dashboard/app.py imports):

  warrant_cards(cik, finviz=None, latest_os=None)
  convertible_note_cards(cik)
  preferred_cards(cik)
  s1_offering_cards(cik)
  atm_cards(cik, finviz=None, latest_os=None)
  equity_line_cards(cik)
  shelf_cards(cik, finviz=None, latest_os=None)

Cross-instrument math (baby shelf, IB6, ATM utilization, % of float)
lives in this module too, reading from `dilution_ledger_drawdowns` for
fast aggregation. Narrative fields (headline, terms summary) come
from `dilution_ledger_narrative` when available; otherwise the card
falls back to a deterministic title built from terms.
"""

from __future__ import annotations

import json
import logging
from datetime import date as _d, timedelta
from typing import Any

from db import get_conn

log = logging.getLogger(__name__)


# ─── EDGAR url helper ────────────────────────────────────────────────
def _edgar_url(accession_number: str | None, cik: int | None) -> str | None:
    if not accession_number or cik is None:
        return None
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession_number.replace('-', '')}/"
    )


# ─── Row decoding ────────────────────────────────────────────────────
def _decode(row) -> dict:
    out = dict(row)
    out["terms"] = json.loads(out.get("terms_json") or "{}")
    out["outstanding"] = json.loads(out.get("outstanding_json") or "{}")
    out["history"] = json.loads(out.get("history_json") or "[]")
    return out


def _select_by_type(cik: int, type_: str,
                    statuses: tuple[str, ...] | None = None,
                    status_prefixes: tuple[str, ...] | None = None,
                    ) -> list[dict]:
    where = "cik=? AND type=?"
    args: list[Any] = [cik, type_]
    if statuses or status_prefixes:
        clauses: list[str] = []
        if statuses:
            clauses.append(f"status IN ({','.join('?' * len(statuses))})")
            args.extend(statuses)
        for pfx in status_prefixes or ():
            clauses.append("status LIKE ?")
            args.append(f"{pfx}%")
        where += f" AND ({' OR '.join(clauses)})"
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM dilution_ledger WHERE {where} "
            "ORDER BY created_at, instrument_id",
            args,
        ).fetchall()
    return [_decode(r) for r in rows]


def _format_date(s: str | None) -> str | None:
    if not s:
        return None
    return s[:10]


def _to_float(v) -> float | None:
    """Tolerant numeric coercion. Walker LLM occasionally emits a
    stringified number ("11,067,547") or a unit-bearing string
    ("8.85 million") in terms / outstanding fields, which Pydantic
    `dict` accepts as-is. Coerce defensively at read time so the
    card layer never crashes — None when uncoercible."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().lower().replace(",", "").replace("$", "")
        mult = 1.0
        for suffix, m in (
            (" billion", 1e9), ("billion", 1e9), ("bn", 1e9), ("b", 1e9),
            (" million", 1e6), ("million", 1e6), ("mm", 1e6), ("m", 1e6),
            (" thousand", 1e3), ("thousand", 1e3), ("k", 1e3),
        ):
            if s.endswith(suffix):
                s = s[: -len(suffix)].strip()
                mult = m
                break
        try:
            return float(s) * mult
        except ValueError:
            return None
    return None


# ─── Lifecycle predicates ───────────────────────────────────────────
# DT hides instruments past their economic life; the ledger keeps them
# (the walker only flips status on explicit terminator filings — RW,
# 425, etc. — which often never fire for old microcap paper).

def _date_before(iso: str | None, today: _d) -> bool:
    if not iso:
        return False
    try:
        return _d.fromisoformat(iso[:10]) < today
    except (ValueError, TypeError):
        return False


def _warrant_dead(r: dict) -> bool:
    """Past expiration → unexercisable, regardless of stated count."""
    terms = r["terms"]
    return _date_before(
        terms.get("maturity") or terms.get("expiration"), _d.today()
    )


def _convertible_dead(r: dict) -> bool:
    """Past maturity AND principal explicitly zero. None principal is
    treated as still owed — keep the card."""
    if not _date_before(r["terms"].get("maturity"), _d.today()):
        return False
    return _to_float(r["outstanding"].get("principal_remaining")) == 0


def _preferred_dead(r: dict) -> bool:
    """Past stated maturity (rare for preferreds), no shares out, no
    remaining preference."""
    if not _date_before(r["terms"].get("maturity"), _d.today()):
        return False
    out = r["outstanding"]
    count = _to_float(out.get("count")) or 0
    pr = _to_float(out.get("principal_remaining"))
    return count == 0 and pr in (None, 0)


def _eloc_atm_stale(r: dict, peers: list[dict]) -> bool:
    """Agreement ≥5yr old, never drawn, AND a newer same-type peer
    exists. Without the peer check we'd drop legitimately-active long-
    quiet ATMs (e.g. SCNI's 2020 BofA ATM is the company's only ATM)."""
    try:
        start = _d.fromisoformat((r.get("created_at") or "")[:10])
    except (ValueError, TypeError):
        return False
    if (_d.today() - start).days < 5 * 365:
        return False
    if (_to_float(r["outstanding"].get("drawn_usd")) or 0) > 0:
        return False
    own_id = r.get("instrument_id")
    own_at = r.get("created_at") or ""
    return any(
        p.get("instrument_id") != own_id
        and (p.get("created_at") or "") > own_at
        for p in peers
    )


def _select_narrative(instrument_id: str) -> dict:
    """Fetch the cached narrative row if present. Empty dict otherwise.

    Card render path is best-effort: when no narrative exists, the
    deterministic fallback renders without a headline. The project
    stage warms this cache; the dashboard only reads it.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT headline, counterparty_role, terms_summary "
            "FROM dilution_ledger_narrative WHERE instrument_id=?",
            (instrument_id,),
        ).fetchone()
    return dict(row) if row else {}


# ─── Counterparty stop-word filter ──────────────────────────────────
# The walker LLM occasionally extracts a generic narrative phrase as a
# counterparty (e.g. "promissory notes", "common stock", "warrants",
# "third party", or a bare month name). Such rows are noise — they
# describe categories, not parties, and pollute the card layer with
# duplicates of real tranches. Filter them out at projection time.
_GENERIC_COUNTERPARTIES = frozenset({
    "warrant", "warrants", "stock warrants", "outstanding warrants",
    "certain warrants", "common stock", "preferred stock",
    "promissory note", "promissory notes", "convertible note",
    "convertible notes", "note", "notes",
    "placement agent", "third party",
})

_MONTH_NAMES = frozenset({
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
})


def _is_generic_counterparty(r: dict) -> bool:
    cp = (
        (r.get("counterparty_canonical") or r.get("counterparty") or "")
        .strip().lower()
    )
    if not cp:
        return False
    if cp in _GENERIC_COUNTERPARTIES:
        return True
    parts = cp.split()
    if parts and parts[0] in _MONTH_NAMES:
        return True
    return False


# ─── Generic projector helpers ───────────────────────────────────────
def _last_update_date(r: dict) -> str | None:
    return _format_date(r.get("last_seen_date") or r.get("status_at")
                        or r.get("created_at"))


def _registered_label(r: dict, *, default: str = "Registered") -> str:
    """Default to Registered. Overridden by per-type logic when the
    instrument's history makes it clear it was unregistered (e.g.
    private-placement warrant)."""
    history = r.get("history") or []
    forms = {(h.get("form") or "").upper() for h in history}
    has_424b = any(f.startswith("424B") for f in forms)
    has_s_filing = any(f.startswith(("S-1", "S-3", "F-1", "F-3", "POS"))
                       for f in forms)
    has_8k_only = forms and all(
        not f.startswith(("424B", "S-1", "S-3", "F-1", "F-3", "POS"))
        for f in forms
    )
    if has_424b or has_s_filing:
        return "Registered"
    if has_8k_only:
        return "Not Registered"
    return default


def _known_owners(r: dict) -> list[str]:
    """Counterparty as a single-element list to match the legacy contract.
    Multi-holder lists were a future-feature; v1 ledger tracks one
    counterparty per tranche."""
    cp = r.get("counterparty_canonical") or r.get("counterparty")
    return [cp] if cp else []


def _banker(r: dict) -> str | None:
    """Resolve the bank running the offering for the underwriter card
    field. Prefers `placement_agent_canonical` / `placement_agent`
    (the new role-specific field); falls back to `counterparty_*` for
    legacy rows where the LLM hadn't been asked to separate roles."""
    return (r.get("placement_agent_canonical")
            or r.get("placement_agent")
            or r.get("counterparty_canonical")
            or r.get("counterparty"))


def _short_banker(name: str | None) -> str | None:
    """Strip common firm suffixes for short display."""
    if not name:
        return None
    out = name
    for suffix in (
        " LLC", ", LLC", " Inc.", " Inc", ", Inc", ", Inc.",
        " Corporation", " Corp.", " Corp", " Capital Corp.",
        " Securities LLC", " Securities", " & Co. LLC", " & Co.",
        " Group LLC", " Group",
    ):
        if out.endswith(suffix):
            out = out[: -len(suffix)]
    return out.strip() or name


# ─── Title rendering ─────────────────────────────────────────────────
_DESCRIPTOR_BY_KIND = {
    "warrant": "Warrants",
    "convertible": "Convertible Note",
    "convertible_preferred": "Convertible Preferred",
    "atm": "ATM",
    "equity_line": "ELOC",
    "shelf": "Shelf",
    "s1_offering": "S-1 Offering",
}


def _title(r: dict, kind: str) -> str:
    """Headline used in card-header. Prefers the walker-emitted label
    column on `dilution_ledger`; falls back to the (currently-empty)
    narrative cache; final fallback is a deterministic template built
    from counterparty + key terms + Month Year."""
    label = (r.get("label") or "").strip()
    if label:
        return label
    nar = _select_narrative(r["instrument_id"])
    if nar.get("headline"):
        return nar["headline"]
    cp = _banker(r) or ""
    terms = r.get("terms") or {}
    created_iso = (r.get("created_at") or "")[:10]
    try:
        created = _d.fromisoformat(created_iso).strftime("%B %Y")
    except (ValueError, TypeError):
        created = created_iso[:7]
    descriptor = _DESCRIPTOR_BY_KIND.get(kind, kind.replace("_", " ").title())
    series = (terms.get("series_letter") or "").strip()
    if kind == "convertible_preferred" and series:
        descriptor = f"Series {series} Preferred" if not series.lower().startswith("series") else f"{series} Preferred"
    parts: list[str] = [p for p in (created, _short_banker(cp), descriptor) if p]
    return " ".join(parts) or r["instrument_id"]


# ─── Warrant card ────────────────────────────────────────────────────
def warrant_cards(cik: int, finviz: dict | None = None,
                  latest_os: float | None = None) -> list[dict]:
    """Per-tranche warrant cards.

    Pre-funded warrants ($0.0001 strike) are filtered out — they're a
    common SPAC/microcap structuring element that DT doesn't show as a
    distinct card.
    """
    rows = _select_by_type(cik, "warrant",
                           statuses=("active",),
                           status_prefixes=("superseded:",))
    cards: list[dict] = []
    for r in rows:
        if _is_generic_counterparty(r):
            continue
        if _warrant_dead(r):
            continue
        terms = r["terms"]
        out = r["outstanding"]
        raw_strike = terms.get("strike")
        if raw_strike is None:
            raw_strike = terms.get("warrant_strike")
        strike = _to_float(raw_strike)
        if strike is not None and strike <= 0.001:
            continue
        count_now = _to_float(out.get("count")) or 0
        exercised = _to_float(out.get("exercised_to_date")) or 0
        terminated = _to_float(out.get("terminated_to_date")) or 0
        total_issued = count_now + exercised + terminated
        cards.append({
            "instrument_id": r["instrument_id"],
            "title": _title(r, "warrant"),
            "registered": _registered_label(r),
            "edgar_url": _edgar_url(
                r.get("last_seen_accession") or r.get("created_accession"),
                cik,
            ),
            "remaining_outstanding": count_now,
            "total_issued": total_issued,
            "exercise_price": strike,
            "known_owners": _known_owners(r),
            "underwriter": _short_banker(_banker(r)),
            "price_protection": terms.get("anti_dilution_type") or "undisclosed",
            "pp_clause_text": terms.get("pp_clause_text"),
            "issue_date": _format_date(r.get("created_at")),
            "exercisable_date": _format_date(
                terms.get("exercisable_date") or r.get("created_at")
            ),
            "expiration_date": _format_date(
                terms.get("maturity") or terms.get("expiration")
            ),
            "last_update_date": _last_update_date(r),
        })
    return cards


# ─── Convertible note card ──────────────────────────────────────────
def convertible_note_cards(cik: int) -> list[dict]:
    rows = _select_by_type(cik, "convertible",
                           statuses=("active",),
                           status_prefixes=("superseded:",))
    cards = []
    for r in rows:
        if _is_generic_counterparty(r):
            continue
        if _convertible_dead(r):
            continue
        terms = r["terms"]
        out = r["outstanding"]
        cv_price = _to_float(terms.get("conv_price")
                             or terms.get("conversion_price"))
        principal_total = _to_float(terms.get("principal"))
        principal_remaining = _to_float(out.get("principal_remaining"))
        rem_shares = (
            (principal_remaining / cv_price) if (cv_price and principal_remaining
                                                  and cv_price > 0) else None
        )
        total_shares = (
            (principal_total / cv_price) if (cv_price and principal_total
                                              and cv_price > 0) else None
        )
        cards.append({
            "instrument_id": r["instrument_id"],
            "title": _title(r, "convertible"),
            "registered": _registered_label(r, default="Not Registered"),
            "edgar_url": _edgar_url(
                r.get("last_seen_accession") or r.get("created_accession"),
                cik,
            ),
            "remaining_shares_issuable": rem_shares,
            "principal_remaining": principal_remaining,
            "conversion_price": cv_price,
            "total_shares_issuable": total_shares,
            "principal_total": principal_total,
            "known_owners": _known_owners(r),
            "underwriter": _short_banker(_banker(r)),
            "price_protection": terms.get("anti_dilution_type") or "undisclosed",
            "pp_clause_text": terms.get("pp_clause_text"),
            "issue_date": _format_date(r.get("created_at")),
            "convertible_date": _format_date(
                terms.get("convertible_date") or r.get("created_at")
            ),
            "maturity_date": _format_date(terms.get("maturity")),
            "last_update_date": _last_update_date(r),
        })
    return cards


# ─── Preferred card ──────────────────────────────────────────────────
def preferred_cards(cik: int) -> list[dict]:
    rows = _select_by_type(cik, "preferred",
                           statuses=("active",),
                           status_prefixes=("superseded:",))
    cards = []
    for r in rows:
        if _is_generic_counterparty(r):
            continue
        if _preferred_dead(r):
            continue
        terms = r["terms"]
        out = r["outstanding"]
        count = _to_float(out.get("count")) or 0
        liq_pref = _to_float(terms.get("liquidation_preference"))
        stated_value = _to_float(terms.get("stated_value"))
        # When liquidation_preference equals stated_value, the LLM
        # extracted it per-share. Fall through to stated_value × count
        # so the card shows the aggregate $-amount (matches DT).
        per_share_liq = (
            liq_pref is not None and stated_value
            and abs(liq_pref - stated_value) <= max(stated_value * 0.01, 0.01)
        )
        if stated_value and count and (per_share_liq or not liq_pref):
            principal_total = stated_value * count
        else:
            principal_total = _to_float(
                liq_pref
                or terms.get("principal")
                or terms.get("aggregate_value")
            )
        principal_remaining = _to_float(out.get("principal_remaining"))
        if principal_remaining is None and count and stated_value:
            principal_remaining = count * stated_value
        if not principal_total and not count:
            continue
        cv_price = _to_float(terms.get("conv_price")
                             or terms.get("conversion_price"))
        rem_shares = (
            (principal_remaining / cv_price)
            if (cv_price and principal_remaining and cv_price > 0)
            else None
        )
        total_shares = (
            (principal_total / cv_price)
            if (cv_price and principal_total and cv_price > 0)
            else None
        )
        cards.append({
            "instrument_id": r["instrument_id"],
            "title": _title(r, "convertible_preferred"),
            "registered": _registered_label(r, default="Not Registered"),
            "edgar_url": _edgar_url(
                r.get("last_seen_accession") or r.get("created_accession"),
                cik,
            ),
            "remaining_shares_issuable": rem_shares,
            "principal_remaining": principal_remaining,
            "conversion_price": cv_price,
            "total_shares_issuable": total_shares,
            "principal_total": principal_total,
            "known_owners": _known_owners(r),
            "underwriter": _short_banker(_banker(r)),
            "price_protection": terms.get("anti_dilution_type") or "undisclosed",
            "pp_clause_text": terms.get("pp_clause_text"),
            "issue_date": _format_date(r.get("created_at")),
            "convertible_date": _format_date(
                terms.get("convertible_date") or r.get("created_at")
            ),
            "maturity_date": _format_date(terms.get("maturity")),
            "last_update_date": _last_update_date(r),
        })
    return cards


# ─── S-1 offering card ──────────────────────────────────────────────
def s1_offering_cards(cik: int) -> list[dict]:
    rows = _select_by_type(cik, "s1_offering",
                           statuses=("active", "terminated", "expired"))
    cards = []
    for r in rows:
        if _is_generic_counterparty(r):
            continue
        terms = r["terms"]
        out = r["outstanding"]
        anticipated = _to_float(terms.get("anticipated_deal_size"))
        final = _to_float(out.get("drawn_usd")
                          or out.get("priced_amount_usd")
                          or terms.get("priced_amount_usd"))
        cards.append({
            "instrument_id": r["instrument_id"],
            "title": _title(r, "s1_offering"),
            "registered": "Registered",
            "edgar_url": _edgar_url(
                r.get("last_seen_accession") or r.get("created_accession"),
                cik,
            ),
            "anticipated_deal_size": anticipated,
            "status": (r.get("status") or "active").title(),
            "underwriter": _short_banker(_banker(r)),
            "filing_date": _format_date(r.get("created_at")),
            "warrant_coverage_pct":
                _to_float(terms.get("warrant_coverage_pct")),
            "final_deal_size": final,
            "final_pricing": _to_float(terms.get("final_pricing")
                                       or terms.get("ipo_price")),
            "final_shares_offered": _to_float(
                out.get("sold_to_date")
                or terms.get("final_shares_offered")
            ),
            "final_warrant_coverage_pct":
                _to_float(terms.get("final_warrant_coverage_pct")),
            "exercise_price": _to_float(terms.get("warrant_strike")),
            "last_update_date": _last_update_date(r),
        })
    return cards


# ─── ATM card ────────────────────────────────────────────────────────
def atm_cards(cik: int, finviz: dict | None = None,
              latest_os: float | None = None) -> list[dict]:
    rows = _select_by_type(cik, "atm",
                           statuses=("active",),
                           status_prefixes=("superseded:",))
    cards = []
    # Lazy-import to avoid pulling ledger.cards into baby_shelf import path.
    from .baby_shelf import ib6_remaining as _ib6
    price = (finviz or {}).get("price")
    fv = float(latest_os) if latest_os else None
    ib6 = None
    if price and fv:
        try:
            ib6 = _ib6(cik, fv, price)
        except Exception as exc:
            log.warning("ib6_remaining failed for cik=%s: %s", cik, exc)
    for r in rows:
        if _is_generic_counterparty(r):
            continue
        if _eloc_atm_stale(r, rows):
            continue
        terms = r["terms"]
        out = r["outstanding"]
        capacity = _to_float(terms.get("capacity_usd"))
        drawn = _to_float(out.get("drawn_usd")) or 0
        remaining = _to_float(out.get("remaining_capacity_usd"))
        if remaining is None and capacity is not None:
            remaining = max(0.0, capacity - drawn)
        used_pct = (drawn / capacity * 100) if (capacity and capacity > 0) else None
        ib6_cap = ib6.get("raisable_remaining_usd") if ib6 else None
        limited_label = (
            "Yes" if (ib6_cap is not None and remaining is not None
                      and ib6_cap < remaining)
            else "No" if remaining is not None
            else None
        )
        atm_remaining_capped = (
            min(ib6_cap, remaining) if (ib6_cap is not None
                                        and remaining is not None)
            else remaining
        )
        cards.append({
            "instrument_id": r["instrument_id"],
            "title": _title(r, "atm"),
            "registered": "Registered",
            "edgar_url": _edgar_url(
                r.get("last_seen_accession") or r.get("created_accession"),
                cik,
            ),
            "remaining_capacity": atm_remaining_capped,
            "total_capacity": capacity,
            "limited_by_baby_shelf": limited_label,
            "remaining_without_baby_shelf": remaining,
            "placement_agent": _short_banker(_banker(r)),
            "sales_total_usd": drawn,
            "used_pct": used_pct,
            "agreement_start_date": _format_date(r.get("created_at")),
            "agreement_end_date": _format_date(
                r.get("status_at") if r.get("status") != "active" else None
            ),
            "last_update_date": _last_update_date(r),
        })
    return cards


# ─── Equity line card ────────────────────────────────────────────────
def equity_line_cards(cik: int) -> list[dict]:
    rows = _select_by_type(cik, "equity_line",
                           statuses=("active",),
                           status_prefixes=("superseded:",))
    cards = []
    for r in rows:
        if _is_generic_counterparty(r):
            continue
        if _eloc_atm_stale(r, rows):
            continue
        terms = r["terms"]
        out = r["outstanding"]
        capacity = _to_float(terms.get("capacity_usd"))
        drawn = _to_float(out.get("drawn_usd")) or 0
        remaining = _to_float(out.get("remaining_capacity_usd"))
        if remaining is None and capacity is not None:
            remaining = max(0.0, capacity - drawn)
        used_pct = (drawn / capacity * 100) if (capacity and capacity > 0) else None
        cards.append({
            "instrument_id": r["instrument_id"],
            "title": _title(r, "equity_line"),
            "registered": _registered_label(r),
            "edgar_url": _edgar_url(
                r.get("last_seen_accession") or r.get("created_accession"),
                cik,
            ),
            "remaining_capacity": remaining,
            "total_capacity": capacity,
            "counterparty": r.get("counterparty_canonical")
                or r.get("counterparty"),
            "sales_total_usd": drawn,
            "used_pct": used_pct,
            "agreement_start_date": _format_date(r.get("created_at")),
            "agreement_end_date": _format_date(
                r.get("status_at") if r.get("status") != "active" else None
            ),
            "last_update_date": _last_update_date(r),
        })
    return cards


# ─── Shelf card ──────────────────────────────────────────────────────
def shelf_cards(cik: int, finviz: dict | None = None,
                latest_os: float | None = None) -> list[dict]:
    rows = _select_by_type(cik, "shelf",
                           statuses=("active",),
                           status_prefixes=("superseded:",))
    cards = []
    from .baby_shelf import (
        BABY_SHELF_FLOAT_VALUE_THRESHOLD_USD,
        baby_shelf_threshold_price,
        ib6_remaining,
        raised_under_ib6_last_12mo,
    )
    from .shelf_status import derive_shelf_status

    price = (finviz or {}).get("price")
    float_shares = (
        (finviz or {}).get("float_shares")
        or (finviz or {}).get("float")
        or float(latest_os) if latest_os else None
    )
    # 60-day high close — finviz_client supplies this. Lazy import to
    # avoid pulling network code into card_test paths.
    high60: float | None = None
    if finviz and finviz.get("ticker"):
        try:
            from dilution.finviz_client import highest_close
            high60 = highest_close(finviz["ticker"], bars=60)
        except Exception as exc:
            log.warning("highest_close lookup failed for %s: %s",
                        finviz.get("ticker"), exc)
    effective_price = max((price or 0), (high60 or 0)) or None

    raised_window = raised_under_ib6_last_12mo(cik)
    threshold_price = (
        baby_shelf_threshold_price(float_shares) if float_shares else None
    )
    ib6 = (
        ib6_remaining(cik, float_shares, effective_price)
        if (float_shares and effective_price) else None
    )
    is_baby_shelf = (
        bool(float_shares and effective_price
             and float_shares * effective_price
             < BABY_SHELF_FLOAT_VALUE_THRESHOLD_USD)
    )
    shelf_meta = {s["accession_number"]: s for s in derive_shelf_status(cik)}

    for r in rows:
        if _is_generic_counterparty(r):
            continue
        terms = r["terms"]
        out = r["outstanding"]
        capacity = _to_float(terms.get("capacity_usd"))
        drawn = _to_float(out.get("drawn_usd")) or 0
        remaining = _to_float(out.get("remaining_capacity_usd"))
        if remaining is None and capacity is not None:
            remaining = max(0.0, capacity - drawn)
        meta = shelf_meta.get(r.get("created_accession") or "", {})
        # Total raised under THIS shelf = drawn against this instrument.
        cards.append({
            "instrument_id": r["instrument_id"],
            "title": _title(r, "shelf"),
            "registered": (
                "Registered" if meta.get("derived_status") == "active"
                else "Pending Effect"
            ),
            "edgar_url": _edgar_url(
                r.get("last_seen_accession") or r.get("created_accession"),
                cik,
            ),
            "current_raisable_amount":
                (ib6 or {}).get("raisable_remaining_usd")
                if is_baby_shelf else remaining,
            "total_shelf_capacity": capacity,
            "baby_shelf_restriction": "Yes" if is_baby_shelf else "No",
            "total_amount_raised": drawn,
            "raised_last_12mo_under_ib6": raised_window.get("total"),
            "outstanding_shares": latest_os,
            "float": float_shares,
            "highest_60_day_close": high60,
            "price_to_exceed_baby_shelf": threshold_price,
            "ib6_float_value":
                (float_shares * effective_price)
                if (float_shares and effective_price) else None,
            "last_banker": _short_banker(
                _last_banker_for_shelf(cik, r["instrument_id"])
            ),
            "effect_date": meta.get("effect_date"),
            "expiration_date":
                _shelf_expiration(meta.get("effect_date"),
                                  r.get("created_at")),
            "last_update_date": _last_update_date(r),
        })
    return cards


def _shelf_expiration(effect_date: str | None,
                      filing_date: str | None) -> str | None:
    """S-3/F-3 shelves are good for 3 years from the effective date.
    Falls back to filing_date if no EFFECT notice has reached us yet
    (typically the first ~2 weeks after filing)."""
    anchor = effect_date or filing_date
    if not anchor:
        return None
    try:
        d = _d.fromisoformat(anchor[:10])
    except (ValueError, TypeError):
        return None
    try:
        return d.replace(year=d.year + 3).isoformat()
    except ValueError:
        return d.replace(year=d.year + 3, day=28).isoformat()


def _last_banker_for_shelf(cik: int, instrument_id: str) -> str | None:
    """Most-recent placement agent on a drawdown against this shelf.

    Pulls from the drawdown row itself (each takedown is sold by its
    own banker — Jefferies, B. Riley, etc.). Falls back to the shelf's
    own counterparty only if no drawdown carries one, which is
    typically the case before any takedown has occurred.
    """
    with get_conn() as conn:
        row = conn.execute(
            """SELECT counterparty_canonical, counterparty
                 FROM dilution_ledger_drawdowns
                WHERE cik=? AND instrument_id=?
                  AND (counterparty IS NOT NULL
                       OR counterparty_canonical IS NOT NULL)
                ORDER BY event_date DESC LIMIT 1""",
            (cik, instrument_id),
        ).fetchone()
        if row:
            return row["counterparty_canonical"] or row["counterparty"]
        # Fall back to the shelf's own counterparty (rare — most
        # shelves have None).
        row = conn.execute(
            "SELECT counterparty_canonical, counterparty "
            "FROM dilution_ledger WHERE cik=? AND instrument_id=?",
            (cik, instrument_id),
        ).fetchone()
        if row:
            return row["counterparty_canonical"] or row["counterparty"]
    return None


__all__ = [
    "atm_cards",
    "convertible_note_cards",
    "equity_line_cards",
    "preferred_cards",
    "s1_offering_cards",
    "shelf_cards",
    "warrant_cards",
]
