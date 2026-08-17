"""Shelf-status.

Reads `dilution_ledger.type='shelf'` rows + `dilution_filings` for
EFFECT/RW notices, plus `file_number` for SEC-canonical
registration-family linkage.

Derived statuses:
  active       — EFFECT filed within 90 days (or S-3ASR auto-effective)
  registered   — filed but no EFFECT yet (still inside the 90-day window)
  withdrawn    — an RW (registration withdrawal) was filed under the
                 same SEC file_number. Catches the case where the
                 ledger row's `status` is still `active` but the SEC
                 has formally pulled the registration.
  expired      — non-ASR S-3 filed more than 3 years ago with no
                 replacement. SEC Rule 415(a)(5) sunsets non-WKSI
                 shelves after 3 years; ASR shelves are exempt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date as _d, timedelta

from db import get_conn

from ._dates import coerce_date as _coerce_date

SHELF_FORM_PREFIXES = ("S-3", "F-3")
EFFECT_WINDOW_DAYS = 90
# SEC Rule 415(a)(5): non-ASR shelves expire 3 years after effective
# date. ASR (well-known seasoned issuer) shelves are exempt and roll
# forward indefinitely with annual updating filings.
SHELF_LIFE_YEARS = 3

# WKSI / pay-as-you-go shelf sentinel. A well-known seasoned issuer's
# S-3ASR / F-3ASR registers an INDETERMINATE amount of securities under
# Rule 457(r) — the cover page states no aggregate dollar cap (fees are
# deferred and paid per-takedown). DT renders the raisable amount as
# this sentinel ("unlimited"), decoupled from the finite total_shelf_
# capacity it shows (the cumulative registered figure) and from total_
# amount_raised. `shelf_cards` renders current_raisable_amount as this
# sentinel for any ASR shelf (form contains "ASR"), or when capacity_usd
# itself was minted at the sentinel because the filing stated no cap.
WKSI_UNLIMITED_SHELF_CAPACITY_USD = 999_999_999


def _add_days(date_str: str, days: int) -> str:
    # Route through _coerce_date so a timestamp-bearing anchor (e.g. a
    # created_at fallback like '2024-01-01T00:00:00Z') is trimmed to its
    # date head instead of failing fromisoformat and collapsing the
    # expiration to the 9999 sentinel (= never expires).
    base = _coerce_date(date_str)
    if base is None:
        return "9999-12-31"
    return (base + timedelta(days=days)).isoformat()


@dataclass(frozen=True)
class _ShelfIndex:
    """Filing-side lookups keyed the way the per-shelf derivation needs them."""
    forms: dict                       # accession → (form, filing_date)
    file_numbers: dict                # accession → SEC 333- file number
    rw_by_file_number: dict           # file number → earliest RW date
    effects_by_file_number: dict      # file number → earliest EFFECT date
    effect_dates: list                # every EFFECT date, for the no-file# path


def _load_shelf_index(cik: int) -> tuple[list, _ShelfIndex]:
    """Fetch shelf rows plus every filing-side lookup in one connection."""
    with get_conn() as conn:
        shelves = conn.execute(
            """SELECT instrument_id, terms_json, outstanding_json,
                      created_at, created_accession, status
                 FROM dilution_ledger
                WHERE cik = ? AND type = 'shelf'
                ORDER BY created_at""",
            (cik,),
        ).fetchall()
        effects = conn.execute(
            """SELECT filing_date, file_number FROM dilution_filings
                WHERE cik = ? AND form LIKE 'EFFECT%'
                ORDER BY filing_date""",
            (cik,),
        ).fetchall()
        # Match the shelf to its filing form + file_number via the
        # created_accession. file_number lets us detect RW (registration
        # withdrawal) filings under the same SEC registration family.
        forms = {}
        file_numbers = {}
        if shelves:
            placeholders = ",".join("?" * len(shelves))
            args = [cik, *(s["created_accession"] for s in shelves)]
            for row in conn.execute(
                f"""SELECT accession_number, form, filing_date, file_number
                       FROM dilution_filings
                      WHERE cik = ? AND accession_number IN ({placeholders})""",
                args,
            ).fetchall():
                forms[row["accession_number"]] = (row["form"],
                                                   row["filing_date"])
                if row["file_number"]:
                    file_numbers[row["accession_number"]] = row["file_number"]
        # RW (Registration Withdrawal) filings grouped by file_number.
        # When a company withdraws a shelf, the RW carries the shelf's
        # 333- file number. SEC honors it and stops accepting take-downs.
        rw_by_file_number: dict[str, str] = {}
        for row in conn.execute(
            """SELECT filing_date, file_number FROM dilution_filings
                WHERE cik = ? AND form = 'RW'
                  AND file_number IS NOT NULL
                ORDER BY filing_date""",
            (cik,),
        ).fetchall():
            # Keep earliest RW per file_number (the moment the
            # registration formally died).
            rw_by_file_number.setdefault(row["file_number"], row["filing_date"])
    # EFFECT notices carry the same SEC 333- file_number as the shelf
    # they declare effective — that's the canonical join. Earliest
    # EFFECT per file_number wins (subsequent ones reflect post-
    # effective amendments, not a fresh registration).
    effects_by_file_number: dict[str, str] = {}
    for e in effects:
        if e["file_number"]:
            effects_by_file_number.setdefault(e["file_number"],
                                              e["filing_date"])
    # Fallback list for shelves with no file_number on record (rare —
    # synthetic / very old EDGAR rows).
    effect_dates = [e["filing_date"] for e in effects]
    return shelves, _ShelfIndex(forms, file_numbers, rw_by_file_number,
                                effects_by_file_number, effect_dates)


def _shelf_effect_date(auto_effective: bool, filing_date: str,
                       file_number: str | None,
                       idx: _ShelfIndex) -> str | None:
    """When the registration became effective, by the sharpest join available."""
    if auto_effective:
        return filing_date
    if file_number:
        return idx.effects_by_file_number.get(file_number)
    # No file_number → can't do the canonical join. Best effort:
    # the EFFECT usually lands within ~90 days of filing.
    window_end = _add_days(filing_date, EFFECT_WINDOW_DAYS)
    for d in idx.effect_dates:
        if filing_date <= d <= window_end:
            return d
    return None


def _derive_one_shelf(s, idx: _ShelfIndex, today_iso: str) -> dict | None:
    """Derived status for one shelf row, or None if it isn't renderable."""
    terms = json.loads(s["terms_json"] or "{}")
    outstanding = json.loads(s["outstanding_json"] or "{}")
    accession = s["created_accession"]
    form, filing_date = idx.forms.get(accession, (None, s["created_at"]))
    form = (form or terms.get("form") or "").upper()
    if not any(form.startswith(p) for p in SHELF_FORM_PREFIXES):
        return None
    # remaining_capacity_usd is authoritative when present (including a
    # literal 0 = fully drawn); only fall back to the registered
    # capacity_usd when remaining is absent. A `0 or capacity_usd` here
    # would treat a drawn-to-zero shelf as still raisable.
    remaining = outstanding.get("remaining_capacity_usd")
    capacity = (remaining if remaining is not None
                else terms.get("capacity_usd"))
    if capacity is not None and capacity <= 0:
        return None
    auto_effective = "ASR" in form
    file_number = idx.file_numbers.get(accession)
    effect_date = _shelf_effect_date(auto_effective, filing_date,
                                     file_number, idx)
    base = "active" if (auto_effective or effect_date is not None) else "registered"

    # Layer terminal states on top of active/registered. Order
    # matters — withdrawn is sharper than expired (an RW is an
    # explicit SEC action, not a calendar inference).
    # An RW only counts if it post-dates the registration's filing
    # (an earlier RW belongs to a prior, unrelated registration under a
    # recycled file number). Surface the date only when it was honored,
    # so withdrawal_date is present iff derived_status == 'withdrawn'.
    rw_match = (idx.rw_by_file_number.get(file_number)
                if file_number else None)
    withdrawn = bool(rw_match and rw_match >= filing_date)
    withdrawal_date = rw_match if withdrawn else None
    # Rule 415(a)(5): 3 years from the date the registration first
    # became effective, not from filing date. Fall back to
    # filing_date only when the EFFECT hasn't arrived yet (early in
    # the registration's life) — that's a conservative under-
    # estimate of remaining life, not a misclassification.
    expiration_date = (None if auto_effective
                       else _add_days(effect_date or filing_date,
                                      365 * SHELF_LIFE_YEARS))
    if withdrawn:
        derived = "withdrawn"
    elif (not auto_effective
          and expiration_date
          and expiration_date < today_iso):
        derived = "expired"
    else:
        derived = base

    return {
        "instrument_id": s["instrument_id"],
        "accession_number": accession,
        "form": form,
        "file_number": file_number,
        "filing_date": filing_date,
        "effect_date": effect_date,
        "withdrawal_date": withdrawal_date,
        "expiration_date": expiration_date,
        "anticipated_amount_usd": terms.get("capacity_usd"),
        "reported_status": s["status"],
        "derived_status": derived,
    }


def derive_shelf_status(cik: int, today=None) -> list[dict]:
    """One entry per shelf instrument with derived status.

    Returns derived_status ∈ {active, registered, withdrawn, expired}.
    """
    today_iso = (_coerce_date(today) or _d.today()).isoformat()
    shelves, idx = _load_shelf_index(cik)
    out = []
    for s in shelves:
        card = _derive_one_shelf(s, idx, today_iso)
        if card is not None:
            out.append(card)
    return out


__all__ = ["derive_shelf_status"]
