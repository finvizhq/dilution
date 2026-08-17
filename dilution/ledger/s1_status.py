"""S-1 status.

Mirror of shelf_status for primary S-1 / F-1 registration statements.

Reads `dilution_ledger.type='s1_offering'` rows + `dilution_filings` for
EFFECT/RW notices grouped by `file_number`, plus `dilution_ledger_drawdowns`
for pricing evidence.

Derived statuses:
  pending     — S-1 filed, no EFFECT under same file_number yet, no priced
                supplement, no RW.
  effective   — EFFECT seen under same file_number, no pricing yet.
  priced      — at least one drawdown recorded, OR walker-set
                final_pricing / final_deal_size / final_shares_offered.
  withdrawn   — RW (registration withdrawal) under same file_number.
  lapsed      — non-priced, non-withdrawn, older than 2 years. Matches the
                staleness cutoff DT uses for its S-1 cards bucket.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date as _d

from db import get_conn

from ._dates import coerce_date as _coerce_date

S1_FORM_PREFIXES = ("S-1", "F-1")
# Mirrors _S1_MAX_AGE_DAYS in cards.py — S-1 offerings that never priced
# and weren't formally withdrawn are presumed dead after two years.
S1_LAPSE_DAYS = 2 * 365


@dataclass(frozen=True)
class _S1Index:
    """Filing-side lookups keyed the way the per-offering derivation needs them."""
    filings: dict              # accession → {form, filing_date, file_number}
    effects_by_file: dict      # file number → earliest EFFECT date
    withdrawals_by_file: dict  # file number → earliest RW date
    drawdown_dates: dict       # instrument_id → first drawdown date


def _load_s1_index(cik: int) -> tuple[list, _S1Index]:
    """Fetch s1_offering rows plus every filing-side lookup in one connection."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT instrument_id, terms_json, outstanding_json,
                      created_at, created_accession, status
                 FROM dilution_ledger
                WHERE cik = ? AND type = 's1_offering'
                ORDER BY created_at""",
            (cik,),
        ).fetchall()
        if not rows:
            return [], _S1Index({}, {}, {}, {})
        # Resolve each instrument's filing form + file_number via its
        # created_accession. file_number is the SEC registration-family
        # key that links EFFECT and RW notices back to the S-1.
        placeholders = ",".join("?" * len(rows))
        args = [cik, *(r["created_accession"] for r in rows)]
        filings: dict[str, dict] = {}
        for f in conn.execute(
            f"""SELECT accession_number, form, filing_date, file_number
                   FROM dilution_filings
                  WHERE cik = ? AND accession_number IN ({placeholders})""",
            args,
        ).fetchall():
            filings[f["accession_number"]] = {
                "form": f["form"],
                "filing_date": f["filing_date"],
                "file_number": f["file_number"],
            }
        # EFFECT notices grouped by file_number — earliest wins, since
        # the SEC declares a registration effective once. Unlike S-3
        # there is no 90-day window: S-1 review can take any duration.
        effects_by_file: dict[str, str] = {}
        for row in conn.execute(
            """SELECT filing_date, file_number FROM dilution_filings
                WHERE cik = ? AND form LIKE 'EFFECT%'
                  AND file_number IS NOT NULL
                ORDER BY filing_date""",
            (cik,),
        ).fetchall():
            effects_by_file.setdefault(row["file_number"], row["filing_date"])
        # RW (Registration Withdrawal) grouped by file_number — when a
        # company pulls an S-1, the RW carries the S-1's 333- file
        # number. SEC honors it and the registration is dead.
        withdrawals_by_file: dict[str, str] = {}
        for row in conn.execute(
            """SELECT filing_date, file_number FROM dilution_filings
                WHERE cik = ? AND form = 'RW'
                  AND file_number IS NOT NULL
                ORDER BY filing_date""",
            (cik,),
        ).fetchall():
            withdrawals_by_file.setdefault(row["file_number"], row["filing_date"])
        # First drawdown per instrument — drawdown existence implies the
        # registration was effective at takedown time, even if EFFECT
        # itself was skipped by the walker.
        drawdown_dates: dict[str, str] = {}
        instrument_ids = [r["instrument_id"] for r in rows]
        dd_placeholders = ",".join("?" * len(instrument_ids))
        for row in conn.execute(
            f"""SELECT instrument_id, MIN(event_date) AS first_dd
                   FROM dilution_ledger_drawdowns
                  WHERE cik = ?
                    AND instrument_id IN ({dd_placeholders})
                  GROUP BY instrument_id""",
            [cik, *instrument_ids],
        ).fetchall():
            drawdown_dates[row["instrument_id"]] = row["first_dd"]
    return rows, _S1Index(filings, effects_by_file,
                          withdrawals_by_file, drawdown_dates)


def _derive_one_s1(r, idx: _S1Index, today_date) -> dict:
    """Derived status for one s1_offering row."""
    terms = json.loads(r["terms_json"] or "{}")
    outstanding = json.loads(r["outstanding_json"] or "{}")
    accession = r["created_accession"]
    meta = idx.filings.get(accession, {})
    form = (meta.get("form") or terms.get("form") or "").upper()
    # Trust the walker's `type='s1_offering'` classification even when
    # the created_accession is an 8-K (some s1_offering rows get seeded
    # by an offering-announcement 8-K rather than the base S-1). In
    # that case we just lack a 333- file_number, so EFFECT/RW lookup
    # is impossible and the derivation falls back to drawdown +
    # calendar evidence — which is strictly better than the old
    # behavior of treating these as Active forever.
    is_registration_filing = any(form.startswith(p)
                                 for p in S1_FORM_PREFIXES)
    filing_date = meta.get("filing_date") or r["created_at"]
    # Only honor file_number on a real S-1/F-1 — an 8-K's file_number
    # is the issuer's Exchange Act (001-xxxxx) listing number, which
    # never matches an EFFECT/RW Securities Act file_number.
    file_number = meta.get("file_number") if is_registration_filing else None
    effect_date = (idx.effects_by_file.get(file_number)
                   if file_number else None)
    # An RW only counts if it post-dates the registration's filing (an
    # earlier RW belongs to a prior, unrelated registration under a
    # recycled file number). Surface the date only when honored, so
    # withdrawal_date is present iff derived_status == 'withdrawn' —
    # mirroring derive_shelf_status.
    rw_match = (idx.withdrawals_by_file.get(file_number)
                if file_number else None)
    withdrawn = bool(rw_match and rw_match >= filing_date)
    withdrawal_date = rw_match if withdrawn else None
    first_drawdown = idx.drawdown_dates.get(r["instrument_id"])
    # Walker-extracted pricing terms are equivalent to a drawdown for
    # status purposes — final_deal_size / final_pricing / final_shares_offered
    # are only populated by the 424B pricing supplement.
    priced_evidence = bool(
        first_drawdown
        or terms.get("final_deal_size")
        or terms.get("final_pricing")
        or terms.get("final_shares_offered")
        or outstanding.get("sold_to_date")
        or outstanding.get("priced_amount_usd")
        or outstanding.get("drawn_usd")
    )

    # Precedence: withdrawn (explicit SEC action) → priced (deal
    # happened) → lapsed (calendar timeout) → effective (armed) →
    # pending (filed, awaiting EFFECT).
    if withdrawn:
        derived = "withdrawn"
    elif priced_evidence:
        derived = "priced"
    else:
        fd = _coerce_date(filing_date)
        age_days = (today_date - fd).days if fd is not None else 0
        if age_days > S1_LAPSE_DAYS:
            derived = "lapsed"
        elif effect_date:
            derived = "effective"
        else:
            derived = "pending"

    return {
        "instrument_id": r["instrument_id"],
        "accession_number": accession,
        "form": form,
        "file_number": file_number,
        "filing_date": filing_date,
        "effect_date": effect_date,
        "withdrawal_date": withdrawal_date,
        "first_drawdown_date": first_drawdown,
        "anticipated_amount_usd": terms.get("anticipated_deal_size"),
        "final_amount_usd": terms.get("final_deal_size"),
        "reported_status": r["status"],
        "derived_status": derived,
    }


def derive_s1_status(cik: int, today=None) -> list[dict]:
    """One entry per s1_offering instrument with derived status.

    Returns derived_status ∈ {pending, effective, priced, withdrawn, lapsed}.
    """
    today_date = _coerce_date(today) or _d.today()
    rows, idx = _load_s1_index(cik)
    return [_derive_one_s1(r, idx, today_date) for r in rows]


__all__ = ["derive_s1_status"]
