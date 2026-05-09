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

import logging
from dataclasses import dataclass

from db import get_conn

from .anchor import _synthesize_create
from .mutations import CreateInstrument, safe_date
from .store import apply_mutations

log = logging.getLogger(__name__)


_PERIODIC_FORMS = ("10-K", "10-K/A", "10-Q", "10-Q/A",
                   "20-F", "20-F/A", "40-F", "40-F/A")


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

    if not rows:
        log.info("seed: %s extracted empty overhang — case C", accession)
        return SeedSummary(
            accession=accession, form=form, as_of_date=as_of,
            instruments_created=0, case="C_empty_extract",
        )

    # Convert each cleaned overhang row → CreateInstrument mutation.
    mutations: list[CreateInstrument] = []
    for r in rows:
        m = _synthesize_create(
            r, accession=accession, filing_date=filing["filing_date"],
        )
        # Override event_date for seed provenance (issue_date if the
        # overhang row carries one, else the filing's period end).
        m = CreateInstrument(
            kind="create_instrument",
            type=m.type,
            proposed_id=None,
            counterparty=m.counterparty,
            counterparty_canonical=m.counterparty_canonical,
            terms=m.terms,
            outstanding=m.outstanding,
            event_date=safe_date(r.get("issue_date")) or as_of,
        )
        mutations.append(m)

    result = apply_mutations(
        cik=cik, ticker=ticker, accession=accession,
        form=form, filing_date=filing["filing_date"],
        mutations=mutations,
    )
    log.info("seed: case A — created %d instruments (rejected %d)",
             result.accepted, result.rejected)
    return SeedSummary(
        accession=accession, form=form, as_of_date=as_of,
        instruments_created=result.accepted, case="A_periodic",
    )


__all__ = ["SeedSummary", "find_seed_filing", "seed_ledger"]
