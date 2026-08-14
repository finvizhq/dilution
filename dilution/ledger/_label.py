"""Deterministic instrument-label builder.

The walker LLM no longer assembles labels — it only sets `descriptor`,
`placement_agent_canonical`, `counterparty_canonical`, and (where
applicable) `terms.series_letter` / `terms.is_pre_funded`. This module
composes the card headline post-call so format is uniform by
construction.

Format:  "<Month YYYY> [<qualifier>] <type-tail>"

ONE qualifier slot, picked by a per-type priority chain. Mirrors the
DilutionTracker card titles in evals/*.json (`title_contains`):

    "September 2025 Common Warrants"        ← descriptor=Common
    "October 2022 Pre-funded Warrants"      ← terms.is_pre_funded
    "November 2024 Series A Warrants"       ← terms.series_letter='A'
    "November 2020 Maxim Warrants"          ← bank (no series/descriptor)
    "December 2022 Streeterville Note"      ← counterparty (convertibles)
    "March 2024 Series 9 Preferred"         ← series letter (preferred)
    "April 2026 M2B Funding ELOC"           ← counterparty (ELOC)
    "April 2026 A.G.P. Private Placement"   ← descriptor=Private Placement
    "August 2025 Shelf"                     ← no qualifier slot
    "November 2025 TD Securities ATM"       ← bank (ATM)

The qualifier is single-slot on purpose — combining bank+descriptor
(e.g. "Dawson James Series A Warrants") was a recurring LLM mistake
and clutters the cards. The bank stays available downstream via the
`placement_agent_canonical` column.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from .mutations import extract_series_letter


# Per-type qualifier priority. The first non-empty hit wins. Each
# element is one of:
#   ('series',)        — terms.series_letter, formatted "Series X"
#   ('pre_funded',)    — terms.is_pre_funded, formatted "Pre-Funded"
#   ('descriptor',)    — m.descriptor (closed enum)
#   ('placement_agent',)  — m.placement_agent_canonical
#   ('counterparty',)  — m.counterparty_canonical
_QUALIFIER_ORDER: dict[str, tuple[str, ...]] = {
    # Warrants: series and pre-funded are the most discriminating
    # signals (multi-tranche offerings need Series A / B; pre-funded
    # is a structural distinction). Then descriptor, then bank.
    "warrant":     ("series", "pre_funded", "descriptor",
                    "placement_agent", "counterparty"),
    # Convertibles are typically named after the lender —
    # "Streeterville Note", "M2B Funding Note". Bank is rare.
    "convertible": ("counterparty", "descriptor", "placement_agent"),
    # Preferred: series letter is the canonical identifier
    # ("Series 9 Preferred"). Counterparty for direct PIPE preferreds
    # without a numbered series.
    "preferred":   ("series", "counterparty", "placement_agent",
                    "descriptor"),
    # ATMs: always named after the sales agent ("Maxim ATM",
    # "ThinkEquity ATM"). Counterparty rarely meaningful for ATMs.
    "atm":         ("placement_agent", "counterparty"),
    # ELOC / equity line: named after the funder
    # ("YA II", "Lincoln Park", "M2B Funding").
    "equity_line": ("counterparty", "placement_agent", "descriptor"),
    # Shelves are never named after a party — just "<Month> Shelf".
    "shelf":       (),
    # S-1 offerings: usually the underwriter when there is one,
    # else a generic "<Month> S-1 Offering".
    "s1_offering": ("placement_agent", "counterparty"),
    # Unregistered equity placements. Descriptor is handled
    # separately as the type-tail ("Private Placement") so the
    # qualifier slot here is purely the entity name when present.
    "equity":      ("placement_agent", "counterparty"),
}

# Type-tail. None means "the qualifier plays the type-word role"
# (equity → "Private Placement").
_TYPE_TAIL: dict[str, str | None] = {
    "warrant":     "Warrants",
    "convertible": "Convertible Note",
    "preferred":   "Preferred",
    "atm":         "ATM",
    "equity_line": "ELOC",
    "shelf":       "Shelf",
    "s1_offering": "S-1 Offering",
    "equity":      None,
}


def _resolve_slot(m: Any, slot: str) -> str | None:
    # Guard a missing or non-dict terms (the walker always emits a dict,
    # but a hand-built instrument may not) so the .get below never blows up.
    terms = getattr(m, "terms", None)
    if not isinstance(terms, dict):
        terms = {}
    if slot == "series":
        raw = terms.get("series_letter")
        s = extract_series_letter(raw) if raw else None
        return f"Series {s}" if s else None
    if slot == "pre_funded":
        return "Pre-Funded" if terms.get("is_pre_funded") else None
    if slot == "descriptor":
        return m.descriptor
    if slot == "placement_agent":
        v = m.placement_agent_canonical
        return v.strip() if isinstance(v, str) and v.strip() else None
    if slot == "counterparty":
        v = m.counterparty_canonical
        return v.strip() if isinstance(v, str) and v.strip() else None
    return None


def _pick_qualifier(m: Any) -> str | None:
    for slot in _QUALIFIER_ORDER.get(getattr(m, "type", None), ()):
        v = _resolve_slot(m, slot)
        if v:
            return v
    return None


def build_label(m: Any) -> str | None:
    """Return the assembled card label, or None when no date is
    available (caller falls back to the LLM-emitted label or the
    card layer's mechanical template).

    Date preference order: terms.issue_date (set on relabel via
    amend_warrant when a closing filing confirms the actual issuance
    date) → terms.agreement_date for atm/equity_line (DT convention:
    SPA/Sales-Agreement signing is the economic anchor, not the
    filing date) → m.event_date (the create-time date, typically the
    signing/announcement). This is how a single FPI Private Placement
    that spans March → August can be relabeled from 'March 2024' to
    'August 2024' once the closing filing posts the actual delivery
    date, matching DilutionTracker's convention.
    """
    terms = m.terms if isinstance(getattr(m, "terms", None), dict) else {}
    raw = terms.get("issue_date")
    if not raw and getattr(m, "type", None) in ("atm", "equity_line"):
        raw = terms.get("agreement_date")
    if not raw:
        raw = m.event_date
    if not raw:
        return None
    if isinstance(raw, date):
        d = raw
    else:
        try:
            d = datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
    month_year = d.strftime("%B %Y")

    qualifier = _pick_qualifier(m)
    m_type = getattr(m, "type", None)
    type_tail = _TYPE_TAIL.get(m_type)

    parts: list[str] = [month_year]
    if m_type == "equity":
        # "<Month> [<Bank/Counterparty>] <descriptor-or-default>"
        #   "April 2026 A.G.P. Private Placement"   (bank + descriptor)
        #   "April 2026 Private Placement"          (descriptor only)
        #   "April 2026 Hudson Bay Equity Issuance" (entity only)
        #   "April 2026 Equity Issuance"            (neither)
        if qualifier:
            parts.append(qualifier)
        parts.append(m.descriptor or "Equity Issuance")
    else:
        if qualifier:
            parts.append(qualifier)
        if type_tail:
            parts.append(type_tail)

    return " ".join(parts)


__all__ = ["build_label"]
