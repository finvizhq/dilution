"""SEC registration-family lookup via `file_number`.

Each Securities Act registration (S-1, S-3, F-1, F-3) is assigned a
unique `333-XXXXXX` file number; every prospectus (424B*), amendment
(S-3/A), withdrawal (RW), or effective-notice (EFFECT) filed under
that registration carries the same number. This module exposes that
linkage to the walker so it can deterministically distinguish:

  primary  — a 424B / amendment filed under a shelf the company has in
             its own primary-registration ledger (i.e. a registered
             take-down by the issuer).
  resale   — a 424B / S-1 / S-3 with a 333- file number that does NOT
             match any of the company's primary registrations. These
             are typically resale prospectuses for selling holders
             (PIPE conversion shares, warrant exercise shares,
             convertible-note holders) and produce NO new issuance —
             the walker should skip them entirely.
  unknown  — coverage gap: the filing has no file_number, the company
             has no primary registrations in our ledger (first-time
             ingest), or the file_number is an Exchange Act `001-`
             number (10-K/10-Q/8-K). Fall through to the existing LLM
             rules.

Why this matters: per the cheatsheet, 424B3 alone has multiple uses
(resale vs base-prospectus supplement) and 424B5 has three; the
walker LLM has historically handled this via prose-parsing the cover
page, with non-trivial false-positive rate that pollutes
capital_raised. file_number is the SEC-canonical signal, available
deterministically at zero LLM cost.
"""
from __future__ import annotations

from typing import Literal

from db import get_conn

Attribution = Literal["primary", "resale", "unknown"]

# Forms whose primary/resale classification benefits from file_number
# pre-screening. Excludes 424B4 (IPO-final, always primary) and
# variants like 424B7 (selling-stockholder resale, always resale).
#
# S-1 / S-3 and their amendments are deliberately NOT pre-screened: a
# registration statement is the document that creates a primary shelf,
# so its file_number is by definition not yet in `primary_set` when the
# walker first sees it. Pre-screening would classify every new shelf as
# "resale" (the registration itself appears in dilution_filings, which
# is the positive-evidence trigger) and skip it — causing the shelf to
# never be created, then cascading to skip every subsequent 424B under
# that file_number too. Only 424B-family prospectuses, which are
# children of a registration, are safe to pre-screen.
PRESCREEN_FORMS = ("424B2", "424B3", "424B5", "SUPPL")

# Registration AMENDMENTS and post-effective amendments ARE safe to
# file_number-prescreen, unlike a first-time S-1/S-3 (above). The
# first-time registration is its own file_number's debut, so it can't
# be in `primary_set` yet and would self-classify resale. An /A or
# POS AM is a LATER appearance: by the time it is walked chronologically
# its parent registration has already determined the file_number's
# primary-vs-resale character. So the amendment INHERITS that verdict —
# parent created a primary shelf ⇒ file_number in primary_set ⇒
# 'primary'; parent was a resale registration (no shelf created, e.g.
# skipped 457(c)/(g)) ⇒ 'resale'. Prescreening these PROPAGATES a resale
# verdict across the family so a resale registration's amendment doesn't
# slip through and mint a phantom s1_offering / shelf (SCNI Yorkville
# SEPA F-1/A under 333-285547). Callers act ONLY on the 'resale' skip;
# 'primary'/'unknown' amendments still flow to the follow-on amend hint.
RESALE_PROPAGATION_FORMS = (
    "S-1/A", "S-3/A", "F-1/A", "F-3/A", "F-10/A", "POS AM",
)


def primary_registration_file_numbers(cik: int) -> set[str]:
    """File numbers of registrations the issuer itself uses for
    primary capital raises — shelves and S-1 offerings already
    captured in the ledger.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT DISTINCT f.file_number
                 FROM dilution_ledger l
                 JOIN dilution_filings f
                   ON f.cik = l.cik
                  AND f.accession_number = l.created_accession
                WHERE l.cik = ?
                  AND l.type IN ('shelf', 's1_offering')
                  AND f.file_number IS NOT NULL
                  AND f.file_number LIKE '333-%'""",
            (cik,),
        ).fetchall()
    return {r["file_number"] for r in rows}


# Registration-statement forms — the canonical "parent" filings of a
# 333- file_number. If we have one of these in dilution_filings for a
# given file_number, that registration is in scope and we can reason
# about it. Absent any, the file_number's parent is outside our
# ingest window and `resale` would be a guess.
_REGISTRATION_FORMS = ("S-1", "S-1/A", "S-3", "S-3/A", "S-3ASR",
                       "S-3MEF", "F-1", "F-1/A", "F-3", "F-3/A",
                       "F-3ASR", "F-3MEF",
                       "F-10", "F-10/A", "F-10EF")


def classify_424b_attribution(cik: int, accession: str) -> Attribution:
    """Decide whether a 424B / S-1 / S-3 filing is a primary issuance
    or a resale registration. See module docstring for semantics.

    A filing is classified `resale` only when there is direct evidence
    that its parent registration is NOT primary — i.e. an S-1 / S-3
    with the same file_number exists in our filings index but no
    corresponding shelf/s1_offering ledger row was created for it.

    Returns `unknown` in two coverage-gap scenarios that would
    otherwise produce false `resale` verdicts:

      1. The file_number has no S-1 / S-3 registration statement in
         our filings index. Common when the parent S-3 was filed
         before the pipeline's 6-year ingest window. Treating this
         as resale would mis-skip every take-down from a long-lived
         pre-window shelf.

      2. The ledger has no primary shelves at all yet (first-time
         ingest of a new ticker, or pipeline run before walker has
         processed any S-3s). The set is empty so every 333- file
         number would compare unequal.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT file_number FROM dilution_filings "
            "WHERE cik = ? AND accession_number = ?",
            (cik, accession),
        ).fetchone()
    file_number = row["file_number"] if row else None
    if not file_number or not file_number.startswith("333-"):
        return "unknown"
    primary_set = primary_registration_file_numbers(cik)
    if not primary_set:
        return "unknown"
    if file_number in primary_set:
        return "primary"
    # File_number doesn't match a primary. Before flagging resale,
    # require positive evidence that we have the parent registration
    # in scope — i.e. an S-1/S-3 with that file_number exists in our
    # filings index. Otherwise the parent S-3 is simply outside our
    # ingest window and resale would be a guess (see XTIA 333-223960:
    # 7 takedowns from a pre-2020 shelf with no S-3 in our window).
    with get_conn() as conn:
        registration_row = conn.execute(
            f"""SELECT 1 FROM dilution_filings
                 WHERE cik = ?
                   AND file_number = ?
                   AND form IN ({",".join(["?"] * len(_REGISTRATION_FORMS))})
                 LIMIT 1""",
            (cik, file_number, *_REGISTRATION_FORMS),
        ).fetchone()
    if not registration_row:
        return "unknown"
    return "resale"


def primary_shelf_for_filing(cik: int, accession: str) -> dict | None:
    """When a filing's file_number matches a primary shelf in the
    ledger, return that shelf's identity. Used by walker.py to pass
    a hard hint into the LLM prompt — "this 424B is a take-down from
    shelf <id>, emit a drawdown against it" — so the walker doesn't
    have to re-derive parentage by prose-parsing the prospectus
    cover page.

    Returns {instrument_id, label, file_number, accession_number} or
    None when no matching primary shelf exists (resale, unknown, or
    parent S-3 outside ingest window). Matches by file_number rather
    than exact accession so S-3/A amendments resolve to the original
    S-3's shelf row.
    """
    with get_conn() as conn:
        row = conn.execute(
            """SELECT l.instrument_id, l.label, l.created_accession,
                      f.file_number
                 FROM dilution_filings child
                 JOIN dilution_filings f
                   ON f.cik = child.cik
                  AND f.file_number = child.file_number
                 JOIN dilution_ledger l
                   ON l.cik = child.cik
                  AND l.created_accession = f.accession_number
                  AND l.type IN ('shelf', 's1_offering')
                WHERE child.cik = ?
                  AND child.accession_number = ?
                  AND child.file_number LIKE '333-%'
                ORDER BY l.created_at ASC
                LIMIT 1""",
            (cik, accession),
        ).fetchone()
    if not row:
        return None
    return {
        "instrument_id": row["instrument_id"],
        "label": row["label"],
        "file_number": row["file_number"],
        "accession_number": row["created_accession"],
    }


def family_registration_accessions(
    cik: int, accession: str,
) -> list[tuple[str, str]]:
    """Other registration-statement filings (S-1 / S-3 / F-1 + their
    amendments) sharing this filing's 333- file_number, earliest first,
    excluding the filing itself.

    Lets a registration AMENDMENT inherit its PARENT's resale verdict:
    the caller classifies these parents' fee tables, and a resale parent
    (Rule 457(c)/(g)) marks the whole family resale. Returns [] when the
    file_number is missing / non-333, or when no OTHER registration
    filing under it is in our index (parent predates the ingest window —
    in which case we must NOT assume resale)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT file_number FROM dilution_filings "
            "WHERE cik = ? AND accession_number = ?",
            (cik, accession),
        ).fetchone()
        file_number = row["file_number"] if row else None
        if not file_number or not file_number.startswith("333-"):
            return []
        rows = conn.execute(
            f"""SELECT accession_number, form FROM dilution_filings
                 WHERE cik = ?
                   AND file_number = ?
                   AND accession_number != ?
                   AND form IN ({",".join(["?"] * len(_REGISTRATION_FORMS))})
                 ORDER BY filing_date ASC, accession_number ASC""",
            (cik, file_number, accession, *_REGISTRATION_FORMS),
        ).fetchall()
    return [(r["accession_number"], r["form"]) for r in rows]


__all__ = [
    "Attribution",
    "PRESCREEN_FORMS",
    "RESALE_PROPAGATION_FORMS",
    "primary_registration_file_numbers",
    "classify_424b_attribution",
    "primary_shelf_for_filing",
    "family_registration_accessions",
]
