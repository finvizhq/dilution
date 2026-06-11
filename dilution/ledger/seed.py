"""Initial-state extraction from the earliest periodic filing.

Three cases handled (per LEDGER_REWORK_PLAN step 6):
  A. earliest periodic filing exists — extract its overhang table and
     emit one create_instrument per outstanding instrument
  B. no periodic filing in window — start with an empty ledger
  C. periodic filing exists but extraction is empty — start empty;
     anchor reconciliation (step 11) will fight to convergence

The overhang prompt is reused from `dilution.overhang` since it's
already well-tuned for periodic-filing notes-section extraction.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from datetime import date

from db import get_conn

from .anchor import _synthesize_create
from .mutations import Mutation, safe_date
from .store import apply_mutations

log = logging.getLogger(__name__)


_PERIODIC_FORMS = ("10-K", "10-K/A", "10-Q", "10-Q/A",
                   "20-F", "20-F/A", "40-F", "40-F/A")


def _summarize_row(r: dict) -> str:
    """One-line summary of an overhang row for seed-stage logging.
    Surfaces category + the highest-signal identifier so we can spot
    when extraction misses an expected instrument (eg the JMP June 2018
    warrant in CGEN that should land in this list).

    Row shape comes from _overhang_extract._clean_*_row — flat dict
    keyed by `category`, with per-category identity fields (sales_agent
    for atm, file_number for shelf, series_letter for preferred, etc.)
    and shared sizing fields (outstanding_count, principal_amount,
    strike_or_conversion_price)."""
    cat = r.get("category") or "?"
    bits: list[str] = [cat]
    if r.get("instrument_name"):
        bits.append(f"name={r['instrument_name']!r}")
    # Category-specific identity keys.
    for key in ("series_letter", "sales_agent", "investor",
                "file_number", "form"):
        v = r.get(key)
        if v not in (None, ""):
            bits.append(f"{key}={v}")
    # Date fields — issue_date is universal; ATM / equity_line also
    # carry agreement_date (primary identity for those categories), and
    # shelf carries effect_date.
    for key in ("issue_date", "agreement_date", "effect_date",
                "maturity_or_expiry"):
        v = r.get(key)
        if v not in (None, ""):
            bits.append(f"{key}={v}")
    # Sizing — first non-null wins for each economic axis.
    for key in ("outstanding_count", "common_shares_issuable",
                "strike_or_conversion_price", "principal_amount",
                "total_capacity_usd", "remaining_capacity_usd"):
        v = r.get(key)
        if v not in (None, ""):
            bits.append(f"{key}={v}")
    if r.get("is_pre_funded"):
        bits.append("pre_funded=1")
    if r.get("is_terminated"):
        bits.append("terminated=1")
    return " ".join(str(b) for b in bits)


@dataclass
class SeedSummary:
    accession: str | None
    form: str | None
    as_of_date: str | None
    instruments_created: int
    case: str  # "A_periodic" | "B_no_periodic" | "C_empty_extract"


def find_seed_filing(cik: int, since_date: str) -> dict | None:
    """Pick the earliest periodic filing in window with raw text.

    "Earliest" because we want the seed to come from the start of the
    walk window — the walker then advances forward, applying events
    that happened after that anchor.
    """
    placeholders = ",".join("?" * len(_PERIODIC_FORMS))
    with get_conn() as conn:
        row = conn.execute(
            f"""SELECT f.accession_number, f.form, f.filing_date,
                       f.report_date, f.items
                  FROM dilution_filings f
                 INNER JOIN dilution_raw r
                    ON r.accession_number = f.accession_number
                 WHERE f.cik = ?
                   AND f.filing_date >= ?
                   AND f.form IN ({placeholders})
                 GROUP BY f.accession_number
                 ORDER BY f.filing_date ASC
                 LIMIT 1""",
            (cik, since_date, *_PERIODIC_FORMS),
        ).fetchone()
    return dict(row) if row else None


async def seed_ledger(
    *, cik: int, ticker: str, since_date: str,
    client, unit_ctx: dict | None = None,
) -> SeedSummary:
    """Extract the earliest periodic filing's overhang and emit
    create_instrument mutations for each row. Returns a summary
    describing which case was hit (A / B / C)."""
    filing = find_seed_filing(cik, since_date)
    if not filing:
        log.info("seed: no periodic filing in window since %s — "
                 "starting empty (case B)", since_date)
        return SeedSummary(
            accession=None, form=None, as_of_date=None,
            instruments_created=0, case="B_no_periodic",
        )

    accession = filing["accession_number"]
    form = filing["form"]
    as_of = filing.get("report_date") or filing["filing_date"]
    log.info("seed: extracting overhang from %s %s (%s)",
             form, accession, as_of)

    from ._overhang_extract import extract_overhang_rows
    rows = await extract_overhang_rows(
        accession=accession, form=form,
        filing_date=filing["filing_date"], report_date=as_of,
        cik=cik, client=client, unit_ctx=unit_ctx,
    )

    # Per-category rollup so we can spot extractor blind spots (eg a
    # 20-F whose Item 5 mentions a shelf that the shelf specialist
    # missed). `category` is the canonical field set by the
    # _clean_*_row helpers in _overhang_extract.
    by_type: dict[str, int] = {}
    for r in rows:
        t = r.get("category") or "?"
        by_type[t] = by_type.get(t, 0) + 1
    if rows:
        log.info("seed: %s extracted %d rows (%s)",
                 accession, len(rows),
                 ", ".join(f"{t}={n}" for t, n in sorted(by_type.items())))
        for r in rows:
            log.info("seed:   row → %s", _summarize_row(r))

    if not rows:
        log.info("seed: %s extracted empty overhang — case C", accession)
        return SeedSummary(
            accession=accession, form=form, as_of_date=as_of,
            instruments_created=0, case="C_empty_extract",
        )

    # Convert each cleaned overhang row → typed Create* mutation.
    mutations: list[Mutation] = []
    for r in rows:
        m = _synthesize_create(
            r, accession=accession, filing_date=filing["filing_date"],
        )
        # Override event_date for seed provenance (issue_date if the
        # overhang row carries one, else the filing's period end).
        seed_event = safe_date(r.get("issue_date")) or as_of
        try:
            event_dt = date.fromisoformat(seed_event[:10])
        except (ValueError, TypeError):
            event_dt = m.event_date
        m = dataclasses.replace(m, event_date=event_dt)
        mutations.append(m)

    result = apply_mutations(
        cik=cik, ticker=ticker, accession=accession,
        form=form, filing_date=filing["filing_date"],
        mutations=mutations,
    )
    log.info("seed: case A — created %d instruments (rejected %d)",
             result.accepted, result.rejected)
    # Surface why rejections happened — they accumulate in
    # dilution_walk_errors during apply, so we read them back. Helps
    # diagnose "extractor saw the warrant but validator threw it out"
    # vs "extractor never saw it" gaps.
    if result.rejected:
        with get_conn() as conn:
            errs = conn.execute(
                """SELECT error_kind, message
                     FROM dilution_walk_errors
                    WHERE cik=? AND accession_number=?
                    ORDER BY id DESC LIMIT ?""",
                (cik, accession, result.rejected),
            ).fetchall()
        for e in errs:
            log.info("seed:   rejected: [%s] %s",
                     e["error_kind"], e["message"])
    return SeedSummary(
        accession=accession, form=form, as_of_date=as_of,
        instruments_created=result.accepted, case="A_periodic",
    )


__all__ = ["SeedSummary", "find_seed_filing", "seed_ledger"]
