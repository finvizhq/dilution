"""Shelf-status.

Reads `dilution_ledger.type='shelf'` rows + `dilution_filings` for
EFFECT/RW notices.

A shelf is `active` once SEC files an EFFECT notice within ~90 days
of filing (or auto-effective for S-3ASR/F-3ASR). Otherwise
`registered`. No withdrawn / superseded / expired derived state —
the ledger's `status` field already carries those when relevant.
"""

from __future__ import annotations

import json
from datetime import date as _d, timedelta

from db import get_conn

SHELF_FORM_PREFIXES = ("S-3", "F-3")
EFFECT_WINDOW_DAYS = 90


def _add_days(date_str: str, days: int) -> str:
    try:
        return (_d.fromisoformat(date_str) + timedelta(days=days)).isoformat()
    except (ValueError, TypeError):
        return "9999-12-31"


def derive_shelf_status(cik: int, today=None) -> list[dict]:
    """One entry per shelf instrument with `active` / `registered`."""
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
            """SELECT filing_date FROM dilution_filings
                WHERE cik = ? AND form LIKE 'EFFECT%'
                ORDER BY filing_date""",
            (cik,),
        ).fetchall()
        # Match the shelf to its filing form via the created_accession.
        forms = {}
        if shelves:
            placeholders = ",".join("?" * len(shelves))
            args = [cik, *(s["created_accession"] for s in shelves)]
            for row in conn.execute(
                f"""SELECT accession_number, form, filing_date
                       FROM dilution_filings
                      WHERE cik = ? AND accession_number IN ({placeholders})""",
                args,
            ).fetchall():
                forms[row["accession_number"]] = (row["form"],
                                                   row["filing_date"])
    effect_dates = [e["filing_date"] for e in effects]

    out = []
    for s in shelves:
        terms = json.loads(s["terms_json"] or "{}")
        outstanding = json.loads(s["outstanding_json"] or "{}")
        accession = s["created_accession"]
        form, filing_date = forms.get(accession, (None, s["created_at"]))
        form = (form or terms.get("form") or "").upper()
        if not any(form.startswith(p) for p in SHELF_FORM_PREFIXES):
            continue
        capacity = (outstanding.get("remaining_capacity_usd")
                    or terms.get("capacity_usd"))
        if capacity is not None and capacity <= 0:
            continue
        auto_effective = "ASR" in form
        window_end = _add_days(filing_date, EFFECT_WINDOW_DAYS)
        effect_date = None
        if auto_effective:
            effect_date = filing_date
        else:
            for d in effect_dates:
                if filing_date <= d <= window_end:
                    effect_date = d
                    break
        had_effect = effect_date is not None
        derived = "active" if (auto_effective or had_effect) else "registered"
        out.append({
            "instrument_id": s["instrument_id"],
            "accession_number": accession,
            "form": form,
            "filing_date": filing_date,
            "effect_date": effect_date,
            "anticipated_amount_usd": terms.get("capacity_usd"),
            "reported_status": s["status"],
            "derived_status": derived,
        })
    return out


__all__ = ["derive_shelf_status"]
