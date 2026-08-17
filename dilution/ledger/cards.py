"""Card projection — ledger rows → DT-style cards.

Replaces the ~2650-line clustering machinery in dilution/instrument_cards.py.
Each projector is a thin SELECT + per-type field mapping; the heavy
lifting (instrument identity, amendment tracking, lifecycle) lives in
the ledger now.

Public surface (consumed by dilution/finviz_payload.py):

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
import re
from dataclasses import dataclass
from datetime import date as _d, timedelta
from functools import lru_cache
from typing import Any

from db import get_conn

from ._counterparty_tiers import bank_tier, investor_class

log = logging.getLogger(__name__)


# ─── EDGAR url helper ────────────────────────────────────────────────
# Real SEC accession numbers look like 0001213900-24-053132. The split
# walker writes synthetic markers ("split:2026-01-29:finviz+yfinance")
# into last_seen_accession; those must never reach an EDGAR URL.
_ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")


@lru_cache(maxsize=8192)
def _primary_doc(accession_number: str | None) -> str | None:
    """Primary-document filename (or absolute URL) for an accession, used
    to build a concrete EDGAR link. Cached process-wide — a filing's
    primary document is immutable once indexed. None when unknown."""
    if not accession_number:
        return None
    with get_conn() as conn:
        row = conn.execute(
            "SELECT primary_doc, primary_doc_url FROM dilution_filings "
            "WHERE accession_number = ?",
            (accession_number,),
        ).fetchone()
    if not row:
        return None
    return row["primary_doc_url"] or row["primary_doc"]


@lru_cache(maxsize=8192)
def _file_number(accession_number: str | None) -> str | None:
    """SEC file number for an accession (e.g. '333-279901'), cached
    process-wide. Drives the shelf 'whole-registration-family' link."""
    if not accession_number:
        return None
    with get_conn() as conn:
        row = conn.execute(
            "SELECT file_number FROM dilution_filings WHERE accession_number = ?",
            (accession_number,),
        ).fetchone()
    return row["file_number"] if row else None


def _edgar_url(accession_number: str | None, cik: int | None) -> str | None:
    """Concrete EDGAR document URL for a filing.

    DilutionTracker links the primary document
    (…/<acc>/ea0207908-s1_bluejay.htm), not the filing-index directory.
    We do the same when the primary-document filename is known, falling
    back to the directory listing when it isn't.
    """
    if not accession_number or cik is None:
        return None
    if not _ACCESSION_RE.match(accession_number):
        return None
    doc = _primary_doc(accession_number)
    if doc and doc.startswith(("http://", "https://")):
        return doc
    base = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession_number.replace('-', '')}/"
    )
    return base + doc if doc else base


def _instrument_edgar_url(cik: int | None, r: dict) -> str | None:
    """EDGAR link for an instrument card: the filing that ORIGINATED the
    instrument (created_accession) — the offering 8-K / S-1 / 424B /
    certificate of designation a user expects when they click through.

    Deliberately NOT last_seen_accession: that is almost always a later
    periodic 10-K/10-Q that merely re-lists the instrument in a cap-table
    (XTIA Series 7/8 → an uninformative 2025 10-K instead of the 2021/2022
    offering 8-K), and it can also hold a synthetic "split:…" marker. When
    the instrument predates the walk window created_accession is itself the
    earliest periodic disclosure — still closer to origination than the
    latest one. last_seen is kept only as a defensive fallback for the
    (schema-impossible) case of a missing/non-real created_accession."""
    for acc in (r.get("created_accession"), r.get("last_seen_accession")):
        if acc and _ACCESSION_RE.match(acc):
            return _edgar_url(acc, cik)
    return None


def _shelf_family_url(cik: int | None, r: dict) -> str | None:
    """Shelf card link: the EDGAR file-number listing — the whole 333-
    registration family (the S-3/F-3 plus every child: 424B takedowns,
    S-3/A amendments, POS AM, RW withdrawals, EFFECT notices) — not just
    the S-3 document. A shelf *is* its file number, so the family listing
    is what a user expects to land on. Falls back to the originating-
    document link when the 333- number is unknown (e.g. a shelf created
    from a periodic filing carrying only the 001- Exchange Act number)."""
    file_no = _file_number(r.get("created_accession"))
    if file_no and file_no.startswith("333-"):
        return ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                f"&filenum={file_no}&type=&dateb=&owner=include&count=100")
    return _instrument_edgar_url(cik, r)


def _parent_shelf(cik: int, child_accession: str | None) -> dict | None:
    """Resolve a child filing back to its parent shelf via SEC file_number.

    Used by ATM / equity-line cards to expose the shelf they were taken
    down from — DilutionTracker shows this as an EDGAR sub-link on each
    child card. The linkage is SEC-canonical: every 424B / S-3 amendment
    / EFFECT notice under an S-3 carries the same `file_number` (e.g.
    "333-256827"), and the original S-3 ledger row's `created_accession`
    has it too.

    Two cases:
      1. Direct — the child's own file_number is a 333-* Securities Act
         number (i.e. it's a 424B / S-3 amendment / EFFECT). Join
         straight on file_number.
      2. Companion-filing — the ledger row was extracted from an 8-K
         that announced an ATM agreement (e.g. XTIA July 2022 Maxim
         ATM, created from an 8-K with file_number 001-36404). The
         companion 424B5 registering the same program is filed by the
         same CIK on the same day; use ITS file_number to find the
         shelf. SEC filing-agents typically file the 424B5 and 8-K
         within minutes of each other so same-day match is reliable.

    Returns None when no parent can be resolved (Reg D private
    placement with no companion registration, or parent S-3 outside
    our ingest window).
    """
    if not child_accession or cik is None:
        return None
    with get_conn() as conn:
        # Try the direct path first.
        row = conn.execute(
            """SELECT shelf_l.instrument_id, shelf_l.label,
                      shelf_f.accession_number AS shelf_accession,
                      shelf_f.file_number
                 FROM dilution_filings child
                 JOIN dilution_filings shelf_f
                   ON shelf_f.cik = child.cik
                  AND shelf_f.file_number = child.file_number
                 JOIN dilution_ledger shelf_l
                   ON shelf_l.cik = child.cik
                  AND shelf_l.type = 'shelf'
                  AND shelf_l.created_accession = shelf_f.accession_number
                WHERE child.cik = ?
                  AND child.accession_number = ?
                  AND child.file_number LIKE '333-%'
                  AND shelf_f.accession_number != child.accession_number
                ORDER BY shelf_l.created_at ASC
                LIMIT 1""",
            (cik, child_accession),
        ).fetchone()
        if row:
            return {
                "instrument_id": row["instrument_id"],
                "accession_number": row["shelf_accession"],
                "edgar_url": _edgar_url(row["shelf_accession"], cik),
                "file_number": row["file_number"],
                "title": row["label"],
            }
        # Fallback: same-day companion 424B/S-* filing.
        row = conn.execute(
            """SELECT shelf_l.instrument_id, shelf_l.label,
                      shelf_f.accession_number AS shelf_accession,
                      shelf_f.file_number
                 FROM dilution_filings child
                 JOIN dilution_filings companion
                   ON companion.cik = child.cik
                  AND companion.filing_date = child.filing_date
                  AND companion.accession_number != child.accession_number
                  AND companion.file_number LIKE '333-%'
                 JOIN dilution_filings shelf_f
                   ON shelf_f.cik = child.cik
                  AND shelf_f.file_number = companion.file_number
                 JOIN dilution_ledger shelf_l
                   ON shelf_l.cik = child.cik
                  AND shelf_l.type = 'shelf'
                  AND shelf_l.created_accession = shelf_f.accession_number
                WHERE child.cik = ?
                  AND child.accession_number = ?
                ORDER BY shelf_l.created_at ASC
                LIMIT 1""",
            (cik, child_accession),
        ).fetchone()
    if not row:
        return None
    return {
        "instrument_id": row["instrument_id"],
        "accession_number": row["shelf_accession"],
        "edgar_url": _edgar_url(row["shelf_accession"], cik),
        "file_number": row["file_number"],
        "title": row["label"],
    }


# Window in which a resale registration is typically filed after a
# Reg D / PIPE placement. The registration rights agreement usually
# caps filer at 30-60 days, but missed deadlines and amendments push
# real-world filings out to ~6 months. Beyond that the linkage is
# more likely coincidence than causation.
_RESALE_LOOKAHEAD_DAYS = 180

# Forms that register securities for resale by selling holders. The
# 424B3 supplement is the actual resale prospectus but it inherits its
# file_number from the parent S-1/S-3, so we surface the registration
# statement itself (it's the canonical "resale registration" on EDGAR).
_RESALE_REGISTRATION_FORMS = ("S-1", "S-1/A", "S-3", "S-3/A", "F-1", "F-3")


def _resale_registration(cik: int, child_accession: str | None,
                          child_created_at: str | None) -> dict | None:
    """Locate the S-1/S-3 that registered an instrument's underlying
    shares for resale.

    Pattern this exists to capture: a company sells a convertible note
    or warrants in a Reg D placement (no SEC registration on the
    primary issuance — the 8-K just notices it). The placement
    agreement contains a registration-rights clause obligating the
    company to file a resale S-1 (or S-3 if eligible) within ~30-60
    days so the holders can sell their conversion / exercise shares
    into the market. That resale registration carries its own 333-
    file_number, distinct from any primary shelf the company has.

    We don't have a "type=resale_registration" row in the ledger
    today, so we hunt the filings index directly: first non-primary
    S-1/S-3 filed by the same CIK within 180 days after the
    instrument's creation. "Non-primary" = doesn't correspond to a
    `type='shelf'` ledger row (those are primary shelves the company
    files capital under). The match is best-effort; ambiguity is fine
    — the card shows it as a "Resale registration" link, not as
    structured data the rest of the pipeline consumes.

    Returns None when nothing plausible is found.
    """
    if not child_accession or not child_created_at or cik is None:
        return None
    # created_at can carry a timestamp suffix (T..Z); the filing_date
    # column is YYYY-MM-DD, so trim before comparison.
    start_date = child_created_at[:10]
    with get_conn() as conn:
        # Exclude registrations whose file_number is already the basis
        # of any primary ledger instrument (shelf OR s1_offering). The
        # filter must match by file_number rather than by exact
        # accession, because amendments (S-3/A, S-1/A) share the
        # parent's file_number but have their own accession — they're
        # still part of the primary registration, not separate resales.
        # Examples this catches:
        #   XTIA 333-279901: S-3 (shelf SH-012) + S-3/A amendment →
        #     both excluded; only true resales (S-1 + 424B3 with
        #     different file_number) surface.
        #   XTIA 333-287989: S-1 (s1_offering S1-004) + S-1/A → same
        #     handling; primary offering, not a resale.
        row = conn.execute(
            """SELECT f.accession_number, f.form, f.filing_date,
                      f.file_number
                 FROM dilution_filings f
                WHERE f.cik = ?
                  AND f.filing_date > ?
                  AND f.filing_date <= date(?, ?)
                  AND f.form IN (""" + ",".join(["?"] * len(_RESALE_REGISTRATION_FORMS)) + """)
                  AND f.file_number LIKE '333-%'
                  AND f.file_number NOT IN (
                      SELECT primary_f.file_number
                        FROM dilution_filings primary_f
                        JOIN dilution_ledger primary_l
                          ON primary_l.cik = primary_f.cik
                         AND primary_l.created_accession = primary_f.accession_number
                         AND primary_l.type IN ('shelf', 's1_offering')
                       WHERE primary_f.cik = f.cik
                         AND primary_f.file_number IS NOT NULL
                  )
                ORDER BY f.filing_date ASC
                LIMIT 1""",
            (cik, start_date, start_date,
             f"+{_RESALE_LOOKAHEAD_DAYS} days",
             *_RESALE_REGISTRATION_FORMS),
        ).fetchone()
    if not row:
        return None
    return {
        "accession_number": row["accession_number"],
        "form": row["form"],
        "filing_date": row["filing_date"],
        "file_number": row["file_number"],
        "edgar_url": _edgar_url(row["accession_number"], cik),
    }


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


def _select_by_type_ids(cik: int, ids: tuple[str, ...]) -> list[dict]:
    if not ids:
        return []
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM dilution_ledger WHERE cik=? AND "
            f"instrument_id IN ({','.join('?' * len(ids))}) "
            "ORDER BY created_at, instrument_id",
            (cik, *ids),
        ).fetchall()
    return [_decode(r) for r in rows]


def _restate_successor_ids(cik: int, type_: str) -> set[str]:
    """Successor ids of `via: restate` supersede chains for `type_`.

    These are the LIVE heads of an amended-and-restated program: the
    predecessor was extinguished (see _supersession_extinguished) and the
    successor must render even when its own status is 'terminated', which
    the active/superseded selection would otherwise miss. Only restate_atm
    produces this marker today, so this returns {} for every type but atm
    (XTIA Maxim)."""
    out: set[str] = set()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT history_json FROM dilution_ledger "
            "WHERE cik=? AND type=? AND status LIKE 'superseded:%'",
            (cik, type_),
        ).fetchall()
    for r in rows:
        for ev in json.loads(r["history_json"] or "[]"):
            fc = ev.get("fields_changed") or {}
            if (ev.get("action") == "closed"
                    and fc.get("via") == "restate"
                    and fc.get("replaced_by")):
                out.add(fc["replaced_by"])
    return out


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


# Walker LLMs frequently omit maturity/expiration on warrants and
# convertibles from old microcap filings (the field was buried in a
# notes-section table the prompt didn't surface, or the filing rounded
# to "five-year term" without an iso date). Without that field the
# date-based dead filters can never fire, so a 2019 placement-agent
# warrant or a 2020 Streeterville note sits on the card list forever.
# DT drops these — its overhang view tops out around 3-5 years of
# history. Match that with type-specific age cutoffs for rows where
# the maturity field is missing.
#
# Convertibles are tighter (3y) because microcap notes routinely mature
# in 12-36 months; anything older without an extracted maturity has
# almost certainly settled. Warrants get 5y to preserve legitimate
# long-dated placement-agent warrants whose expiration the walker
# missed.
_WARRANT_MAX_AGE_NO_EXP_DAYS = 5 * 365
_CONVERTIBLE_MAX_AGE_NO_MAT_DAYS = 3 * 365
# Preferreds older than 5 years are effectively settled in the microcap
# universe — even when the ledger still carries a non-zero count, those
# rows almost always reflect missed conversion events the walker didn't
# catch (IQST P-001 / P-003 Nov-2020 Series A/B, XTIA P-016 Jan-2019
# Series 5). Mirrors the _warrant_dead 5-year cutoff.
_PREFERRED_MAX_AGE_DAYS = 5 * 365


# Dust threshold for convertibles. A note where principal_remaining
# has dropped to < $1,000 has been settled in practice — the residual
# is reconciliation noise from rounding the conversion math, not a
# real obligation. DT drops these. Also flag rows where < 0.5% of
# the original principal remains (relative test catches LLM-extracted
# rows whose principal_total has shifted post-amendment).
_CONVERTIBLE_DUST_ABS_USD = 1_000.0
_CONVERTIBLE_DUST_REL = 0.005


# Microcap toxic convertibles routinely run past their stated maturity in
# default / forbearance / extension while still materially outstanding —
# DilutionTracker keeps showing them, so we do too. But a note carried
# `active` with material (non-dust) principal LONG past maturity is almost
# always one the issuer settled without an explicit filing (the walker never
# flipped it to redeemed); that stale balance shouldn't inflate the card
# list. Keep a past-maturity note within this grace window; reap it beyond.
# Dust is still dropped immediately (threshold above), regardless of maturity.
_CONVERTIBLE_POST_MATURITY_GRACE_DAYS = 540


def _row_age_days(r: dict) -> int | None:
    created = (r.get("created_at") or "")[:10]
    if not created:
        return None
    try:
        return (_d.today() - _d.fromisoformat(created)).days
    except (ValueError, TypeError):
        return None


def _supersession_extinguished(r: dict) -> bool:
    """True when a `superseded:<successor_id>` status points at a row of
    a different instrument type — meaning the predecessor was exchanged
    for a different security (warrant→equity, warrant→convertible, etc.)
    and is no longer outstanding. Same-type supersession (warrant→warrant
    re-price) returns False; the row stays on the card list per DT's
    "Registered" rendering of replaced tranches.

    EXCEPTION — a same-program restatement. The store's restate_atm path
    closes the predecessor `superseded:<new>` and stamps `via: restate`
    on that close event (an amended-and-restated sales agreement minting a
    fresh successor for the SAME continuous program — XTIA Maxim, FCEL).
    DT shows ONE card for the program (the re-dated successor), so the
    restated predecessor is extinguished regardless of type-equality. A
    plain same-type supersession (no `via: restate` — a genuinely new
    distinct program on a fresh shelf, e.g. CGEN SVB→Leerink ATM) is NOT
    a restatement and falls through to the type check, staying rendered.

    Unresolved successors (the close was recorded but no counterpart
    create lives in the ledger) also return True — the walker only emits
    `replaced_by` when the predecessor is gone, so a missing successor
    row means the data trail dead-ends but the extinguishment did happen.
    """
    status = r.get("status") or ""
    if not status.startswith("superseded:"):
        return False
    if r.get("type") == "atm":
        # Per-supplement chain convention (round-3 C2): every 424B5/SUPPL
        # window of a sales agreement renders its own card (DT parity —
        # KSCP chain, FCEL amendments). The via:restate stamp is applied
        # non-deterministically across chain links (it depends on which
        # tool the LLM picked per filing — round-4 kscp-jun2024), so it
        # must not decide ATM visibility. Dead chains are still hidden
        # wholesale by _chain_head_terminated.
        return False
    successor_id = status.split(":", 1)[1]
    if not successor_id:
        return False
    for ev in (r.get("history") or []):
        fc = ev.get("fields_changed") or {}
        if (ev.get("action") == "closed"
                and fc.get("reason") == "superseded"
                and fc.get("via") == "restate"):
            return True
    with get_conn() as conn:
        row = conn.execute(
            "SELECT type FROM dilution_ledger WHERE instrument_id=?",
            (successor_id,),
        ).fetchone()
    if row is None:
        return True
    return row["type"] != r.get("type")


def _chain_head_terminated(r: dict) -> bool:
    """True when a `superseded:<succ>` row's supersede chain ends in a
    TERMINATED head — the whole program is dead.

    Counterpart to _supersession_extinguished: that handles the `via:
    restate` case. A NON-restate auto-supersede (an S-3MEF / S-3
    registration re-host via store._auto_supersede_prior_atm, which omits
    the `via: restate` marker) leaves a same-type predecessor rendered even
    though its chain terminated — the leaky FCEL 'June 2020 Jefferies' and
    XTIA 'May 2024 Maxim' extra cards. Walk status `superseded:X` to the
    first non-superseded head and check whether it is terminated.

    A chain whose head is still `active` (CGEN SVB→Leerink, where DT shows
    the predecessor as a 'Replaced' card) returns False and stays rendered."""
    status = r.get("status") or ""
    if not status.startswith("superseded:"):
        return False
    seen: set[str] = {r.get("instrument_id")}
    head_terms = "{}"
    with get_conn() as conn:
        while status.startswith("superseded:"):
            succ = status.split(":", 1)[1]
            if not succ or succ in seen:
                return False
            seen.add(succ)
            row = conn.execute(
                "SELECT status, terms_json FROM dilution_ledger"
                " WHERE instrument_id=?",
                (succ,),
            ).fetchone()
            if row is None:
                return False
            status = row["status"] or ""
            head_terms = row["terms_json"] or "{}"
    if status == "terminated":
        return True
    # A head still flagged `active` whose sales-agreement term has already
    # expired is a dead program too (XTIA Maxim ATM-2678: agreement_end
    # 2024-12-31, the walker never marked it terminated) — its restated
    # predecessors drop with it instead of leaking as stale extras.
    try:
        agree_end = json.loads(head_terms).get("agreement_end_date")
    except (ValueError, TypeError):
        agree_end = None
    return _date_before(agree_end, _d.today())


_AGREEMENT_DATE_MAX_DRIFT_DAYS = 90


def _plausible_agreement_date(agreement_date: str | None,
                              created_at: str | None) -> str | None:
    """Return `agreement_date` only when it's within ±90 days of
    `created_at`; else None (caller falls back to created_at).

    LLM sometimes copies an older ATM agreement date from prior-program
    boilerplate in the same filing — those values are years off and
    poison the card. Same-month / nearby drift is legitimate (10-K
    disclosed before the sales-agreement was actually signed)."""
    if not agreement_date or not created_at:
        return None
    try:
        ad = _d.fromisoformat(agreement_date[:10])
        cd = _d.fromisoformat(created_at[:10])
    except (ValueError, TypeError):
        return None
    if abs((ad - cd).days) > _AGREEMENT_DATE_MAX_DRIFT_DAYS:
        return None
    return agreement_date


# Periodic-report forms: a warrant-reconciliation TABLE in one of these
# lists every outstanding warrant the issuer has, so two rows sharing such a
# created_accession are co-listed, NOT a paired offering. Offering forms
# (8-K / S-1 / F-1 / 424B* / 6-K take-down / S-3) are excluded — there the
# shared accession IS a genuine multi-tranche offering. Prefix match covers
# /A amendments and the -SB small-business variants.
_PERIODIC_REPORT_PREFIXES = ("10-Q", "10-K", "20-F", "40-F")


def _is_periodic_report(form: str | None) -> bool:
    f = (form or "").strip().upper()
    return any(f.startswith(p) for p in _PERIODIC_REPORT_PREFIXES)


def _warrant_dead(r: dict) -> bool:
    """Past expiration → unexercisable, regardless of stated count.
    Also dead when the walker has no data at all (count=0 AND nothing
    was ever issued — exercised/terminated also zero), or when the row
    is >5 years old and the LLM never extracted an expiration
    (post-issuance reverse-split warrants disclosed in one-line
    footnotes that the prompt couldn't pin a date on).

    A row with count=0 but exercised_to_date>0 is a fully-exercised
    tranche. It renders (remaining=0) ONLY while a sibling warrant from
    the same offering (same created_accession) is still active — the
    paired Series A / Pre-Funded case, where dropping it collapsed
    multi-tranche offerings into a single Pre-Funded card. With no live
    sibling the offering is history and DT drops it from the live card
    list (round-4 cety-extras: eight 2022 warrants resurfaced as extras
    once bucket-B exercise accounting drove them to count=0)."""
    if _supersession_extinguished(r):
        return True
    terms = r["terms"]
    if _date_before(
        terms.get("maturity") or terms.get("expiration"), _d.today()
    ):
        return True
    out = r["outstanding"]
    count_now = _to_float(out.get("count")) or 0
    exercised = _to_float(out.get("exercised_to_date")) or 0
    terminated = _to_float(out.get("terminated_to_date")) or 0
    if count_now == 0 and exercised == 0 and terminated == 0:
        return True
    # Count-dust: a warrant exercised down to <=1 share out of a much
    # larger issuance is exhausted — the residual share is a walker
    # placeholder it couldn't fully zero (CELU W-5275 Dragasac 652,982 ->
    # 1, W-5284 535,275 -> 1, both still rendering live as extras). The
    # exercised>0 gate spares a warrant genuinely issued at <=1 share
    # (which carries no exercise activity).
    if 0 < count_now <= 1 and exercised > 0:
        return True
    if (count_now == 0 and exercised > 0
            and (r.get("status") or "") == "exercised"):
        with get_conn() as conn:
            sib = conn.execute(
                "SELECT 1 FROM dilution_ledger"
                " WHERE created_accession=? AND type='warrant'"
                "   AND status='active' AND instrument_id != ? LIMIT 1",
                (r.get("created_accession"), r.get("instrument_id")),
            ).fetchone()
            created_form = conn.execute(
                "SELECT form FROM dilution_filings WHERE accession_number=?",
                (r.get("created_accession"),),
            ).fetchone()
        # The paired-tranche keep only holds for a genuine OFFERING
        # disclosure (8-K / S-1 / 424B / 6-K take-down), where the live
        # sibling is the co-issued common/Series-A leg. Two UNRELATED
        # warrants merely co-listed in a PERIODIC warrant-reconciliation
        # table (10-Q / 10-K / 20-F) share a created_accession without
        # being a paired offering, so the sibling test wrongly resurrects
        # the dead one: CETY W-5189 (FirstFire) revived by the unrelated
        # Jefferson W-5190; ACTU W-5237 ($5.27, net-exercised into
        # Series B-1 preferred) revived by the live $10.55 W-5238 — both
        # created off a 10-Q warrant table. Drop the dead row in that case.
        form = (created_form["form"] if created_form else "") or ""
        if sib is None or _is_periodic_report(form):
            return True
    if (terms.get("maturity") is None
            and terms.get("expiration") is None):
        age = _row_age_days(r)
        if age is not None and age >= _WARRANT_MAX_AGE_NO_EXP_DAYS:
            return True
    return False


def _convertible_dead(r: dict) -> bool:
    """Decide whether a convertible note has left the cap table.

    Dead when:
      - the successor of a supersession chain is itself extinguished;
      - principal_remaining is dust (< $1,000 absolute or < 0.5% of
        principal_total): paid off in practice, the residual is rounding
        noise from conversion math — dropped regardless of maturity;
      - the note is more than `_CONVERTIBLE_POST_MATURITY_GRACE_DAYS` past
        its stated maturity: still-material principal that far past maturity
        is almost always a settlement the walker never saw an explicit filing
        for. Notes only modestly past maturity are KEPT — microcap toxic
        notes routinely run months past maturity in default / forbearance /
        extension while genuinely outstanding, and DT keeps showing them;
      - row is >=3 years old and the LLM never extracted a maturity:
        microcap convertibles run 1-3y; anything older without a maturity has
        settled in practice. Tighter than the 5y warrant cutoff because
        convertible terms are uniformly short.
    """
    if _supersession_extinguished(r):
        return True
    terms = r["terms"]
    pr = _to_float(r["outstanding"].get("principal_remaining"))
    pt = _to_float(terms.get("principal"))
    if pr is not None:
        if pr < _CONVERTIBLE_DUST_ABS_USD:
            return True
        if (pt is not None and pt > 0
                and pr / pt < _CONVERTIBLE_DUST_REL):
            return True
    if _date_before(terms.get("maturity"),
                    _d.today() - timedelta(days=_CONVERTIBLE_POST_MATURITY_GRACE_DAYS)):
        return True
    if terms.get("maturity") is None:
        age = _row_age_days(r)
        if age is not None and age >= _CONVERTIBLE_MAX_AGE_NO_MAT_DAYS:
            return True
    return False


def _preferred_dead(r: dict) -> bool:
    """Dead when past stated maturity with no shares/preference left,
    OR when the row is older than the preferred age cutoff (≥5y).

    The age cutoff fires regardless of count because old preferreds
    that still carry a non-zero count almost always reflect a missed
    conversion event — the walker re-reads the balance-sheet figure
    from a later periodic filing without seeing the original
    conversion 8-K, leaving a phantom outstanding count behind."""
    if _supersession_extinguished(r):
        return True
    age = _row_age_days(r)
    if age is not None and age >= _PREFERRED_MAX_AGE_DAYS:
        return True
    if not _date_before(r["terms"].get("maturity"), _d.today()):
        return False
    out = r["outstanding"]
    count = _to_float(out.get("count")) or 0
    pr = _to_float(out.get("principal_remaining"))
    return count == 0 and pr in (None, 0)


# DT's S-1 cards bucket cuts off at ~2 years. See _s1_offering_stale.
_S1_MAX_AGE_DAYS = 2 * 365


def _s1_offering_stale(r: dict) -> bool:
    """S-1 offerings have a natural lifecycle: filed, amended, priced
    via 424B, closed via 8-K — typically inside 12-18 months. After
    two years the offering has either completed (drawdowns recorded
    against it), been withdrawn (RW), or been superseded by a fresh
    S-1; the walker doesn't always flip status on the older row.
    Match DT's behavior by dropping any S-1 older than two years
    that isn't explicitly closed.

    Fixture coverage tops out around 18-24 months of S-1 history
    (GCTK's earliest in-fixture S-1 is Sep 2024, ~20 months old as
    of this commit), so two years is a comfortable upper bound."""
    age = _row_age_days(r)
    return age is not None and age > _S1_MAX_AGE_DAYS


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


def _eloc_terminated_displaced(r: dict, peers: list[dict]) -> bool:
    """Terminated ELOC displaced by a NEWER ACTIVE line AND terminated
    >18 months ago. DT keeps a recently-terminated line on screen (SCNI's
    March-2025 YA II, replaced Sept-2025) but drops long-dead ones once a
    successor exists (CETY's 2021 GHS line, terminated 2024-09-30). The
    newer-ACTIVE-peer condition keeps a company's only line rendered no
    matter how old (KSCP's 2022 B. Riley ELOC, terminated 2023-06-28, is
    asserted by its fixture) — a bare recency gate would regress it."""
    if r.get("status") != "terminated":
        return False
    when = (r.get("status_at") or r.get("created_at") or "")[:10]
    try:
        ended = _d.fromisoformat(when)
    except (ValueError, TypeError):
        return False
    if (_d.today() - ended).days <= 548:  # ~18 months
        return False
    own_id = r.get("instrument_id")
    own_at = r.get("created_at") or ""
    return any(
        p.get("instrument_id") != own_id
        and (p.get("created_at") or "") > own_at
        and (p.get("status") or "") == "active"
        for p in peers
    )


def _effective_conv_price(r: dict) -> float | None:
    """Effective conversion price for a variable-rate note with no stored
    fixed price, derived from the note's own latest conversion event:
    principal_converted / shares_issued. Fires only when a real conversion
    exists and no split landed after it (a pre-split conversion would give
    a stale basis). DT back-computes the same figure (CETY April-2025
    Pacific Pier: 101,904.82 / 106,097 = 0.9605)."""
    last_conv = None
    last_split = None
    for e in r.get("history") or []:
        action = e.get("action")
        if action == "converted":
            last_conv = e
        elif action == "split_applied":
            last_split = e
    if not last_conv:
        return None
    if last_split and (last_split.get("date") or "") > (last_conv.get("date") or ""):
        return None
    fc = last_conv.get("fields_changed") or {}
    shares = _to_float(fc.get("shares_issued"))
    principal = _to_float(fc.get("principal_converted"))
    if shares and principal and shares > 0 and principal > 0:
        return principal / shares
    return None


_MARKET_LOW_CACHE: dict[int, float | None] = {}


def _market_low_close(cik: int, bars: int = 10) -> float | None:
    """Lowest settled close over the recent window — the market
    reference for a variable-rate note's live effective conversion
    price ('90% of the lowest VWAP/traded price'). Approximation:
    closes stand in for VWAPs; the field is display-only and
    snapshot-exempt in the eval. Failure-safe and cached per cik for
    the process lifetime (one quote_export call per ticker)."""
    if cik in _MARKET_LOW_CACHE:
        return _MARKET_LOW_CACHE[cik]
    low: float | None = None
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT ticker FROM dilution_company WHERE cik=?",
                (cik,),
            ).fetchone()
        if row and row["ticker"]:
            from dilution.finviz_client import _client
            closes = _client().get_daily_closes(row["ticker"], bars=bars)
            vals = [float(c) for c in (closes or []) if c]
            if vals:
                low = min(vals[-bars:])
    except Exception as exc:
        log.warning("market-low lookup failed for cik=%s: %s", cik, exc)
    _MARKET_LOW_CACHE[cik] = low
    return low


def _select_narrative(instrument_id: str) -> dict:
    """Fetch the cached narrative row if present. Empty dict otherwise.

    Card render path is best-effort: when no narrative exists, the
    deterministic fallback renders without a headline. The project
    stage warms this cache; the payload build only reads it.
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
    # Date-fragment guard: when the LLM emits a counterparty like
    # "April 2024" or "october" — pure date detritus from an instrument
    # name — every token is either a month or a 4-digit year. Filter
    # only when NOTHING substantive remains. The previous rule fired on
    # `parts[0] in _MONTH_NAMES` alone, which also hid real cards: it
    # killed a hypothetical "April Capital Partners," and concretely it
    # hid XTIA's W-2680 (counterparty "october purchase", strike
    # 146250, count 154 — an exact fixture match for the October 2022
    # tranche). A surviving non-date token means the row has a real (if
    # malformed) identity behind it; let the rest of the dead/empty
    # guards in warrant_cards decide whether it's renderable.
    tokens = [
        t for t in cp.split()
        if t not in _MONTH_NAMES and not (t.isdigit() and len(t) == 4)
    ]
    if not tokens:
        return True
    return False


# ─── Generic projector helpers ───────────────────────────────────────
def _last_update_date(r: dict) -> str | None:
    return _format_date(r.get("last_seen_date") or r.get("status_at")
                        or r.get("created_at"))


def _registered_label(r: dict, *, default: str = "Registered") -> str:
    """Default to Registered. Lifecycle status wins when set; otherwise
    per-type logic falls back to registration-history inference (a
    private-placement warrant has only an 8-K so → Not Registered).

    Vocabulary per category (per DT eval fixtures):
      atm           Registered, Replaced
      equity_line   Registered, Not Registered, Terminated
      warrant       Registered, Not Registered
      convertible   Registered, Not Registered
      preferred     Registered, Not Registered

    'Replaced' is ATM-only — superseded warrants/preferred still
    render as 'Registered' in DT. 'Terminated' applies to closed
    equity-lines; closed ATMs are already filtered out of the card
    list one layer up so they never reach this label.
    """
    status = (r.get("status") or "").lower()
    type_ = (r.get("type") or "").lower()
    # 'Terminated' is equity-line-only vocabulary. A fully-drawn terminated
    # ATM that still renders (GCTK Dec-2024 Dawson) follows the ATM
    # vocabulary {Registered, Replaced} — fall through to registration
    # inference, which reads its S-3 history as Registered (DT convention).
    if status == "terminated" and type_ != "atm":
        return "Terminated"
    if status.startswith("superseded:") and type_ == "atm":
        return "Replaced"
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
    """Named investor list for the card.

    Prefer `terms.known_owners` when the walker captured a multi-
    investor purchaser table (Armistice / Sabby / Bigger Capital /
    District 2-style PIPEs). Otherwise fall back to the single-string
    counterparty for single-purchaser deals."""
    terms = r.get("terms") or {}
    owners = terms.get("known_owners")
    if isinstance(owners, list):
        names = [str(o).strip() for o in owners if o and str(o).strip()]
        if names:
            return names
    cp = r.get("counterparty_canonical") or r.get("counterparty")
    return [cp] if cp else []


def _banker(r: dict) -> str | None:
    """Resolve the bank running the offering for the underwriter card field.

    Placement-agent ONLY — never fall back to counterparty_canonical. The
    counterparty is the INVESTOR/purchaser (surfaced separately via
    known_owners / investor_class); folding it into the underwriter slot
    rendered investor names as bankers (XTLB warrant 'Alexander Rabinovich',
    SCNI preferred 'EIB'). _quality_tags already classifies bank_tier off
    placement_agent only, so dropping the fallback keeps the banker and
    investor roles consistent."""
    return r.get("placement_agent_canonical")


def _quality_tags(r: dict) -> dict:
    """Resolve the bank-tier and investor-class flags for one row.

    Banks are classified only off placement-agent fields and investors
    only off counterparty fields, so the two outputs are independent.
    """
    return {
        "bank_tier": bank_tier(r.get("placement_agent_canonical")),
        "investor_class": investor_class(r.get("counterparty_canonical")),
    }


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
                           statuses=("active", "exercised"),
                           status_prefixes=("superseded:",))
    cards: list[dict] = []
    for r in rows:
        if _is_generic_counterparty(r):
            continue
        if _warrant_dead(r):
            continue
        # PA/underwriter comp warrants are administrative — DT bundles
        # them into the primary tranche's card and so do we. The label
        # is deterministic ("<Month> Placement Agent Warrants" or
        # "<Month> Underwriter Warrants") because the LLM-emitted
        # descriptor flows through _label.py.
        label = r.get("label") or ""
        label_lc = label.lower()
        # PA / underwriter / representative comp warrants are
        # administrative — DT folds them into the primary tranche's
        # offering card. 'representative' covers ThinkEquity-style
        # Representative's Warrants (XTIA July 2025: a comp tranche at the
        # same strike as the offering warrant), which arrive labelled with
        # the verbatim narrative phrase "representative's warrants".
        if ("placement agent warrants" in label_lc
                or "underwriter warrants" in label_lc
                or "representative" in label_lc
                or "rep warrant" in label_lc):
            continue
        terms = r["terms"]
        out = r["outstanding"]
        raw_strike = terms.get("strike")
        if raw_strike is None:
            raw_strike = terms.get("warrant_strike")
        strike = _to_float(raw_strike)
        # Pre-funded suppression: trust the stable `is_pre_funded` flag
        # set at creation. The strike fallback covers legacy rows that
        # predate the flag — but `is_pre_funded` wins when present
        # because later 10-Q amends can drift the strike off $0.001
        # (e.g. when the walker pulls a weighted-average strike from a
        # warrant table) and would otherwise un-suppress the row.
        if terms.get("is_pre_funded") is True:
            continue
        # Pre-funded by series tag: the filing called this tranche
        # 'Pre-Funded' (terms.series_letter), the single most reliable
        # signal. Reverse splits divide the $0.001 strike UP (GCTK
        # W-3034: $0.001 × splits → $1.2), so neither the flag (if a
        # stale/legacy create dropped it) nor the strike<=0.001 test below
        # can be relied on alone. extract_series_letter('Pre-Funded')
        # returns None, so this never collides with a real 'Series X'.
        if (terms.get("series_letter") or "").strip().lower() == "pre-funded":
            continue
        if strike is not None and strike <= 0.001:
            continue
        count_now = _to_float(out.get("count")) or 0
        exercised = _to_float(out.get("exercised_to_date")) or 0
        terminated = _to_float(out.get("terminated_to_date")) or 0
        # Prefer the create-time issued count (split-adjusted by the
        # split path) — it's immune to amends that restate `count` to
        # the original size after exercises have already accrued. Fall
        # back to the additive view for legacy rows that pre-date the
        # initial_count field.
        initial = _to_float(out.get("initial_count"))
        total_issued = initial if initial is not None else (
            count_now + exercised + terminated
        )
        # No strike disclosed AND no shares ever issued ⇒ no
        # actionable information on the card. This is the shape of an
        # announcement 6-K that describes warrants qualitatively (e.g.
        # XTLB's Social Proxy milestone warrants — "additional
        # warrants … exercisable upon reaching certain financial
        # measured milestones") without pricing or sizing. The walker
        # prompt's announcement-without-terms rule now suppresses
        # these at extract time, but historical rows captured before
        # that rule landed still need filtering at render time.
        if strike is None and total_issued == 0:
            continue
        cards.append({
            "instrument_id": r["instrument_id"],
            "title": _title(r, "warrant"),
            "registered": _registered_label(r),
            "edgar_url": _instrument_edgar_url(cik, r),
            "parent_shelf": _parent_shelf(
                cik, r.get("registration_accession")
                or r.get("created_accession")),
            "resale_registration": _resale_registration(
                cik, r.get("created_accession"), r.get("created_at"),
            ),
            "remaining_outstanding": count_now,
            "total_issued": total_issued,
            "exercise_price": strike,
            "known_owners": _known_owners(r),
            "underwriter": _short_banker(_banker(r)),
            # terms.issue_date wins when amend_warrant set it from a
            # closing/issuance filing (FPI signing-then-closing 6-K
            # pair); otherwise fall back to the create-time date.
            "issue_date": _format_date(
                terms.get("issue_date") or r.get("created_at")
            ),
            "exercisable_date": _format_date(
                terms.get("exercisable_date")
                or terms.get("issue_date")
                or r.get("created_at")
            ),
            "expiration_date": _format_date(
                terms.get("maturity") or terms.get("expiration")
            ),
            "last_update_date": _last_update_date(r),
            **_quality_tags(r),
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
        disc = _to_float(terms.get("conv_discount_pct"))
        if disc and 0 < disc <= 1:
            # Discount-to-market note (round-3 C1 decision): a stored
            # fixed conv_price is a CAP/ceiling, not the conversion
            # economics. Prefer the realized effective price from actual
            # conversion history, else the LIVE discounted market price;
            # the fixed cap binds only when lower (round-4 cety-jan2025:
            # the split-rescaled $37.50 cap rendered as the price, 13×
            # the ~$2.86 realized conversion price DT shows). Eval-side these
            # fields are snapshot-exempt via the fixture's
            # snapshot_fields.
            eff = _effective_conv_price(r)
            if eff is None:
                low = _market_low_close(cik)
                if low:
                    eff = disc * low
            if eff:
                cv_price = min(cv_price, eff) if cv_price else eff
        if cv_price is None:
            cv_price = _effective_conv_price(r)
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
            "edgar_url": _instrument_edgar_url(cik, r),
            "resale_registration": _resale_registration(
                cik, r.get("created_accession"), r.get("created_at"),
            ),
            "remaining_shares_issuable": rem_shares,
            "principal_remaining": principal_remaining,
            "conversion_price": cv_price,
            "total_shares_issuable": total_shares,
            "principal_total": principal_total,
            "known_owners": _known_owners(r),
            "underwriter": _short_banker(_banker(r)),
            "issue_date": _format_date(r.get("created_at")),
            "convertible_date": _format_date(
                terms.get("convertible_date") or r.get("created_at")
            ),
            "maturity_date": _format_date(terms.get("maturity")),
            "last_update_date": _last_update_date(r),
            **_quality_tags(r),
        })
    return cards


# ─── Preferred card ──────────────────────────────────────────────────
@dataclass(frozen=True)
class _PreferredFace:
    """The series' dollar face, aggregate and still-outstanding."""
    count: float
    stated_value: float | None
    principal_total: float | None
    principal_remaining: float | None


def _preferred_face(terms: dict, out: dict) -> _PreferredFace:
    """Resolve the aggregate and remaining $-face of a preferred series."""
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
        # Prefer aggregate keys over liq_pref: the walker sometimes
        # stores liquidation_preference as a per-share face ($1,000)
        # rather than the series aggregate, and per-share × split-
        # adjusted stated_value × floored-count blows up. The
        # original aggregate $-amount commonly lands in
        # terms.principal_remaining for preferred extracted by the
        # walker (misnomer kept for backward-compat).
        principal_total = _to_float(
            terms.get("principal")
            or terms.get("principal_remaining")
            or terms.get("aggregate_value")
            or liq_pref
        )
    principal_remaining = _to_float(out.get("principal_remaining"))
    if principal_remaining is None:
        # When the filing never discloses a remaining principal and the
        # conversions (if any) are tracked only as share counts — never
        # as a converted-dollar figure — fall back to the aggregate face
        # so the card shows the full series value rather than a blank.
        # DT renders the full liq-pref in this case (e.g. IQST Series D:
        # split-mangled share counts, $3.546M aggregate, no $-level
        # conversion line). Guard on principal_converted_to_date so a
        # row whose conversions ARE dollar-tracked keeps the
        # count×stated_value remaining view and is never overstated.
        if (principal_total
                and out.get("principal_converted_to_date") is None):
            principal_remaining = principal_total
        elif count and stated_value:
            principal_remaining = count * stated_value
    return _PreferredFace(count, stated_value,
                          principal_total, principal_remaining)


def _preferred_issuable(terms: dict, out: dict, face: _PreferredFace
                        ) -> tuple[float | None, float | None, float | None]:
    """→ (conversion price, remaining shares issuable, total shares issuable)."""
    cv_price = _to_float(terms.get("conv_price")
                         or terms.get("conversion_price"))
    # A disclosed fixed shares-per-preferred ratio wins over the
    # $-division ONLY when the series has no dollar conversion price:
    # count × ratio is exact for fixed-rate series (SCNI EIB: 1,000 ×
    # 364 = 364,000). When conv_price IS present the series converts
    # on dollars (stated value / True-Up — IQST Series D), the stated
    # ratio is just the pre-adjustment base rate, and the $-division
    # (principal / conv_price) is the DT-matching basis (round-4
    # iqst-seriesd: ratio×live-count gave 117,362.5 vs DT 464,154).
    ratio = _to_float(terms.get("conversion_ratio"))
    if ratio and ratio > 0 and face.count and not cv_price:
        retired = ((_to_float(out.get("count_converted_to_date")) or 0)
                   + (_to_float(out.get("count_redeemed_to_date")) or 0))
        return cv_price, face.count * ratio, (face.count + retired) * ratio
    rem_shares = (
        (face.principal_remaining / cv_price)
        if (cv_price and face.principal_remaining and cv_price > 0)
        else None
    )
    total_shares = (
        (face.principal_total / cv_price)
        if (cv_price and face.principal_total and cv_price > 0)
        else None
    )
    return cv_price, rem_shares, total_shares


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
        face = _preferred_face(terms, out)
        principal_total = face.principal_total
        principal_remaining = face.principal_remaining
        count = face.count
        if not principal_total and not count:
            continue
        # A lone 1-share balance-sheet residual with no aggregate face is a
        # re-disclosure artifact (XTIA Series 4 P-181: "1 share issued and
        # outstanding" on a terminated series) — not a live instrument DT
        # tracks. count<=1 is the discriminator; legit live preferreds carry
        # real counts (1001907 P-149=280,898) or a real principal_total.
        if (principal_total in (None, 0)) and count and 0 < count <= 1:
            continue
        cv_price, rem_shares, total_shares = _preferred_issuable(
            terms, out, face)
        cards.append({
            "instrument_id": r["instrument_id"],
            "title": _title(r, "convertible_preferred"),
            "registered": _registered_label(r, default="Not Registered"),
            "edgar_url": _instrument_edgar_url(cik, r),
            "resale_registration": _resale_registration(
                cik, r.get("created_accession"), r.get("created_at"),
            ),
            "remaining_shares_issuable": rem_shares,
            "principal_remaining": principal_remaining,
            "conversion_price": cv_price,
            "total_shares_issuable": total_shares,
            "principal_total": principal_total,
            "known_owners": _known_owners(r),
            "underwriter": _short_banker(_banker(r)),
            "issue_date": _format_date(r.get("created_at")),
            "convertible_date": _format_date(
                terms.get("convertible_date") or r.get("created_at")
            ),
            "maturity_date": _format_date(terms.get("maturity")),
            "last_update_date": _last_update_date(r),
            **_quality_tags(r),
        })
    return cards


# ─── S-1 offering card ──────────────────────────────────────────────
def s1_offering_cards(cik: int) -> list[dict]:
    rows = _select_by_type(cik, "s1_offering",
                           statuses=("active",),
                           status_prefixes=("superseded:",))
    cards = []
    if not rows:
        return cards
    from .s1_status import derive_s1_status
    s1_meta = {s["accession_number"]: s for s in derive_s1_status(cik)}
    for r in rows:
        if _is_generic_counterparty(r):
            continue
        meta = s1_meta.get(r.get("created_accession") or "", {})
        derived = meta.get("derived_status")
        # Drop SEC-terminal states the way shelves drop expired/withdrawn:
        # 'withdrawn' = issuer pulled via RW, 'lapsed' = ≥2y old without
        # ever pricing. Both mean no further takedowns can occur.
        if derived in ("withdrawn", "lapsed"):
            continue
        # Belt-and-suspenders: if the projection didn't run (e.g. no
        # file_number on the filing), keep the original age-only check
        # so we never emit a 2y+ zombie card.
        if _s1_offering_stale(r):
            continue
        terms = r["terms"]
        out = r["outstanding"]
        anticipated = _to_float(terms.get("anticipated_deal_size"))
        # Walker-extracted final beats drawdown-sum: the latter
        # includes unit-priced first tranches that inflate the total.
        final = _to_float(terms.get("final_deal_size")
                          or out.get("drawn_usd")
                          or out.get("priced_amount_usd")
                          or terms.get("priced_amount_usd"))
        # Walker-extracted total offered beats the cumulative sold-to-date:
        # for a partially-drawn deal sold_to_date under-reports the deal size
        # (XTIA priced 9.143M shares, only 1.371M drawn so far). Mirrors the
        # `final` deal-size precedence above; sold_to_date only equals the
        # offered total once the deal is fully drawn.
        sold = _to_float(terms.get("final_shares_offered")
                         or out.get("sold_to_date"))
        # Status: prefer the derived projection (which sees EFFECT / RW /
        # drawdowns), fall back to the legacy Active↔Priced inference
        # only when no derived state is available.
        status_label = {
            "pending": "Pending",
            "effective": "Effective",
            "priced": "Priced",
        }.get(derived)
        if status_label is None:
            status_label = (r.get("status") or "active").title()
            if status_label == "Active" and (sold or final):
                status_label = "Priced"
        # Final pricing: walker rarely sets it; fall back to the median
        # drawdown price (robust against the occasional unit-priced
        # first tranche skewing a mean).
        final_pricing = _to_float(terms.get("final_pricing")
                                  or terms.get("ipo_price"))
        if final_pricing is None:
            final_pricing = _median_drawdown_price(cik, r["instrument_id"])
        # Walker stores warrant coverage as a decimal (1.0 = one warrant
        # per share); DT renders it as a percent (100 = 100% = one
        # warrant per share). Multiply on emit so existing decimal
        # storage matches the DT convention without a re-walk.
        coverage_pct = _to_float(terms.get("warrant_coverage_pct"))
        final_coverage_pct = _to_float(terms.get("final_warrant_coverage_pct"))
        cards.append({
            "instrument_id": r["instrument_id"],
            "title": _title(r, "s1_offering"),
            "registered": status_label,
            "s1_status": derived,
            # Originating S-1/F-1, via the shared helper — not last_seen,
            # which on a priced deal is usually the closing 8-K rather than
            # the offering document.
            "edgar_url": _instrument_edgar_url(cik, r),
            "anticipated_deal_size": anticipated,
            "status": status_label,
            "underwriter": _short_banker(
                _banker(r) or _drawdown_banker(cik, r["instrument_id"])),
            "filing_date": _format_date(r.get("created_at")),
            # Anticipated coverage falls back to the FINAL (priced) coverage
            # when the walker only captured the latter (GCTK S1-217: final
            # 2.0 stored, anticipated null). Explicit None-check, NOT `or`:
            # a legitimate 0.0 (no warrant coverage) must render 0, not be
            # overridden by the final value (S1-042/053/089 store 0.0).
            "warrant_coverage_pct": (
                (coverage_pct if coverage_pct is not None
                 else final_coverage_pct) * 100
                if (coverage_pct is not None or final_coverage_pct is not None)
                else None),
            "final_deal_size": final,
            "final_pricing": final_pricing,
            "final_shares_offered": sold,
            "final_warrant_coverage_pct":
                final_coverage_pct * 100 if final_coverage_pct is not None else None,
            "exercise_price": _to_float(terms.get("warrant_strike")),
            "last_update_date": _last_update_date(r),
            **_quality_tags(r),
        })
    return cards


def pending_s1_offerings(cik: int) -> list[dict]:
    """In-progress S-1 / F-1 offerings — filed or effective, not yet
    priced, not withdrawn. Subset of `s1_offering_cards(cik)` filtered
    to the DT 'Pending S-1 Offerings' bucket.
    """
    return [c for c in s1_offering_cards(cik)
            if c.get("s1_status") in ("pending", "effective")]


def all_pending_s1_offerings() -> list[dict]:
    """Cross-issuer roll-up of in-progress S-1 / F-1 offerings — one
    entry per pending or effective deal across every CIK that has an
    `s1_offering` row in the ledger. Powers a DT-parity sitewide
    'Pending S-1 Offerings' page.

    Each entry is a `pending_s1_offerings(cik)` card dict enriched with
    the owning `cik`. Sorted newest-filing first.
    """
    with get_conn() as conn:
        ciks = [r["cik"] for r in conn.execute(
            "SELECT DISTINCT cik FROM dilution_ledger "
            "WHERE type = 's1_offering'"
        ).fetchall()]
    out: list[dict] = []
    for cik in ciks:
        for card in pending_s1_offerings(cik):
            out.append({"cik": cik, **card})
    out.sort(key=lambda c: c.get("filing_date") or "", reverse=True)
    return out


# A reported public float is a SUBSET of shares outstanding and can
# never exceed it. Yahoo occasionally returns a wildly stale, pre-
# reverse-split float — SCNI: 12.4B vs 3.47M outstanding (3577×) — which,
# fed through the I.B.6 1/3-of-float cap, yielded a $3.36B "current
# raisable". Reject any float source exceeding outstanding by more than
# this slack (the slack absorbs reporting-date skew, where a fresher
# float post-dates the cached outstanding count) before falling through
# the Yahoo → Finviz → outstanding chain.
_FLOAT_VS_OS_MAX_RATIO = 1.5


def _resolve_float_shares(
    cik: int, finviz: dict | None, latest_os: float | None,
) -> float | None:
    """Best available public-float estimate, sanity-bounded by shares
    outstanding. Yahoo is preferred (fresher than Finviz on low-volume
    tickers — AACG: Yahoo 15.04M matches DT vs Finviz's stale 8.84M) but
    only when it doesn't exceed the outstanding-derived ceiling."""
    from dilution.share_counts import fetch_float_cached
    os_cap = (
        float(latest_os) * _FLOAT_VS_OS_MAX_RATIO if latest_os else None
    )

    def _ok(v) -> bool:
        return bool(v) and (os_cap is None or float(v) <= os_cap)

    for cand in (
        fetch_float_cached(cik).shares,
        (finviz or {}).get("float_shares"),
        (finviz or {}).get("float"),
    ):
        if _ok(cand):
            return float(cand)
    return float(latest_os) if latest_os else None


# ─── ATM card ────────────────────────────────────────────────────────
def _atm_rows(cik: int) -> list[dict]:
    """Renderable ATM rows: active/superseded, plus live restate-chain heads.

    A restate chain's head is the successor the predecessor was restated
    into. The active/superseded selection already includes any head that
    is still active or itself superseded; the only heads it misses are
    TERMINATED ones — and a chain that ends in a terminated head means
    the whole amended-and-restated program ended. DT hides ended ATMs, so
    we do NOT resurrect such a head (its restated predecessors stay
    extinguished, leaving the dead chain correctly absent). Without this,
    XTIA's tangled Maxim chain — which ends in a terminated head still
    carrying the ORIGINAL 2022 signing date — renders as a spurious
    "July 2022 Maxim ATM" card.
    """
    rows = _select_by_type(cik, "atm",
                           statuses=("active", "terminated"),
                           status_prefixes=("superseded:",))
    seen = {r["instrument_id"] for r in rows}
    heads = _restate_successor_ids(cik, "atm") - seen
    if heads:
        for h in _select_by_type_ids(cik, tuple(heads)):
            status = h.get("status") or "active"
            if status == "active" or status.startswith("superseded:"):
                rows.append(h)
    return rows


def _atm_ib6_cap(cik: int, finviz: dict | None,
                 latest_os: float | None) -> float | None:
    """The I.B.6 1/3-of-float ceiling in dollars, or None if uncapped.

    Loop-invariant for a given issuer: the cap is a property of the float
    and the price basis, not of any one ATM program.
    """
    # Lazy-import to avoid pulling ledger.cards into baby_shelf import path.
    from .baby_shelf import (
        ib6_remaining as _ib6,
        is_baby_shelf_restricted,
    )
    float_shares = _resolve_float_shares(cik, finviz, latest_os)
    high60: float | None = None
    if finviz and finviz.get("ticker"):
        try:
            from dilution.finviz_client import highest_close
            high60 = highest_close(finviz["ticker"], bars=60)
        except Exception as exc:
            log.warning("highest_close lookup failed for %s: %s",
                        finviz.get("ticker"), exc)
    # I.B.6 Instruction 1 prices the test at the last-sale price as of a
    # date of the issuer's choosing within 60 CALENDAR days prior to sale
    # (issuers rationally pick the max → highest 60-day close). Use the
    # 60-day high close alone; never max() it with the live price, which
    # would inflate effective_price (and ib6 raisable) above the closing
    # basis.
    effective_price = high60
    # Filing-time regime stamp from the prospectus cover legend when one
    # exists (C&DI 116.26 grandfathering), computed non-affiliate float
    # test as fallback — see baby_shelf.is_baby_shelf_restricted.
    is_baby_shelf = is_baby_shelf_restricted(
        cik, float_shares, latest_os, effective_price)
    ib6 = None
    if float_shares and effective_price:
        try:
            ib6 = _ib6(cik, float_shares, effective_price)
        except Exception as exc:
            log.warning("ib6_remaining failed for cik=%s: %s", cik, exc)
    # The cap only applies to baby-shelf-restricted issuers (float value
    # < $75M). Above the threshold the issuer files under I.B.1 and the
    # full ATM capacity is raisable.
    return (ib6.get("raisable_remaining_usd")
            if (ib6 and is_baby_shelf) else None)


def _atm_hidden(r: dict, rows: list[dict]) -> bool:
    """Programs DT does not surface at all, independent of economics."""
    # Drop the restated predecessor of a same-program chain (XTIA
    # Maxim): its live successor is in `rows` via _restate_successor_ids.
    if _supersession_extinguished(r):
        return True
    # Drop a non-restate auto-supersede predecessor whose chain ends in a
    # terminated head — the program is dead and DT shows nothing (FCEL
    # 'June 2020 Jefferies', XTIA 'May 2024 Maxim').
    if _chain_head_terminated(r):
        return True
    if _is_generic_counterparty(r):
        return True
    return bool(_eloc_atm_stale(r, rows))


@dataclass(frozen=True)
class _AtmEconomics:
    capacity: float | None
    drawn: float
    remaining: float | None
    used_pct: float | None


def _atm_economics(cik: int, r: dict) -> _AtmEconomics | None:
    """Capacity/drawn/remaining for one ATM, or None if the program is hidden."""
    terms = r["terms"]
    out = r["outstanding"]
    capacity = _to_float(terms.get("capacity_usd"))
    # Anchor-corrected drawn, exactly as shelf_cards does: a stated-
    # remaining checkpoint (drawn_usd_anchor/asof) already subsumes
    # every sale on or before its as-of date, so only post-asof
    # discrete draws are added on top. Reading the raw running
    # drawn_usd instead double-counts when the same period's sales
    # land BOTH as an anchor pin and as a discrete record_event
    # drawdown (FCEL Dec-2025: 42.9M pin + 56.4M discrete = 101.6M
    # vs filing-true 45.2M).
    drawn = _drawn_to_date(cik, r["instrument_id"], out)
    # DT hides ENDED ATM programs — both an explicit `terminated` status
    # and an `active` row whose sales-agreement term has already expired
    # (XTIA Maxim ATM-2678: agreement_end 2024-12-31, never marked
    # terminated) — EXCEPT one that raised its full capacity before
    # ending (GCTK Dec-2024 Dawson: drawn to within rounding of its
    # $8.23M cap). A program that ended with material capacity left was
    # abandoned mid-stream and stays hidden. DB-wide the fully-drawn
    # carve-out flags ATM-2679 alone; the expired-term skip flags the
    # XTIA Maxim chain alone.
    program_ended = ((r.get("status") or "") == "terminated"
                     or _date_before(terms.get("agreement_end_date"),
                                     _d.today()))
    if program_ended:
        if not (capacity and capacity > 0
                and 0 <= capacity - drawn < 0.005 * capacity):
            return None
    # Prefer capacity − drawn over the persisted remaining_capacity_usd
    # snapshot, which goes stale when a drawn_usd anchor amend lands
    # without refreshing it (CGEN Leerink: persisted 23.9M vs
    # filing-true 50M − 15.1M = 34.9M). Fall back to the snapshot only
    # when no capacity is known — mirrors what shelf_cards already does.
    if capacity is not None:
        # A program the filing recorded as fully drawn (stored
        # remaining 0) raised its full capacity; snap a tiny rounding
        # shortfall up so the card doesn't show a ghost residual
        # (GCTK Dawson ATM 8,217,693 → 8,230,000). See
        # _fully_drawn_clamp; mirrors the shelf-rollup path.
        drawn = _fully_drawn_clamp(drawn, capacity, out)
        remaining = max(0.0, capacity - drawn)
    else:
        remaining = _to_float(out.get("remaining_capacity_usd"))
    used_pct = (drawn / capacity * 100) if (capacity and capacity > 0) else None
    return _AtmEconomics(capacity, drawn, remaining, used_pct)


def atm_cards(cik: int, finviz: dict | None = None,
              latest_os: float | None = None) -> list[dict]:
    rows = _atm_rows(cik)
    ib6_cap = _atm_ib6_cap(cik, finviz, latest_os)
    cards = []
    for r in rows:
        if _atm_hidden(r, rows):
            continue
        econ = _atm_economics(cik, r)
        if econ is None:
            continue
        terms = r["terms"]
        capacity, drawn = econ.capacity, econ.drawn
        remaining, used_pct = econ.remaining, econ.used_pct
        # Is the program subject to the I.B.6 1/3-of-float cap? That depends
        # on whether the cap sits below the program's TOTAL raisable capacity
        # — not below how much currently REMAINS. Comparing against remaining
        # wrongly flipped fully-drawn baby-shelf ATMs (remaining=0) to "No".
        program_capacity = capacity if capacity is not None else remaining
        limited_label = (
            "Yes" if (ib6_cap is not None and program_capacity is not None
                      and ib6_cap < program_capacity)
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
            "registered": _registered_label(r),
            "edgar_url": _instrument_edgar_url(cik, r),
            "parent_shelf": _parent_shelf(
                cik, r.get("registration_accession")
                or r.get("created_accession")),
            # remaining_capacity is the CONTRACTUAL remaining (capacity −
            # drawn) — the DT display convention; the live IB6-capped
            # raisable lives in raisable_capped for dilution-pressure
            # consumers (badges / os_history / ticker_brief). Rendering the
            # IB6 cap here made an undrawn $25M ATM show $1.49M (CETY Roth).
            "remaining_capacity": remaining,
            "total_capacity": capacity,
            "limited_by_baby_shelf": limited_label,
            "remaining_without_baby_shelf": remaining,
            "raisable_capped": atm_remaining_capped,
            "placement_agent": _short_banker(_banker(r)),
            "sales_total_usd": drawn,
            "used_pct": used_pct,
            # Prefer the LLM-extracted agreement_date when present and
            # within ±90 days of the disclosure (created_at) — ATMs can
            # be disclosed in a 10-K filed weeks before the actual
            # sales-agreement signing. Outside that window the
            # agreement_date is almost always stale (LLM picked up an
            # older agreement date from prior-ATM language in the same
            # filing — FCEL ATM-013 has agreement_date='2022-07-01'
            # carried over from B. Riley boilerplate in a 2025-12-18
            # 10-K), so fall back to created_at.
            "agreement_start_date": _format_date(
                _plausible_agreement_date(terms.get("agreement_date"),
                                          r.get("created_at"))
                or r.get("created_at")
            ),
            "agreement_end_date": _format_date(
                terms.get("agreement_end_date")
                or (r.get("status_at") if r.get("status") != "active" else None)
            ),
            "last_update_date": _last_update_date(r),
            **_quality_tags(r),
        })
    return cards


# ─── Equity line card ────────────────────────────────────────────────
def equity_line_cards(cik: int) -> list[dict]:
    rows = _select_by_type(cik, "equity_line",
                           statuses=("active", "terminated"),
                           status_prefixes=("superseded:",))
    cards = []
    for r in rows:
        if _is_generic_counterparty(r):
            continue
        if _eloc_atm_stale(r, rows):
            continue
        if _eloc_terminated_displaced(r, rows):
            continue
        terms = r["terms"]
        out = r["outstanding"]
        capacity = _to_float(terms.get("capacity_usd"))
        drawn = _to_float(out.get("drawn_usd")) or 0
        remaining = _to_float(out.get("remaining_capacity_usd"))
        if remaining is None and capacity is not None:
            remaining = max(0.0, capacity - drawn)
        # Essentially-exhausted line: a sub-0.05%-of-capacity (or sub-$10)
        # residual is fee/rounding dust, not raisable capacity — DT shows
        # 0 (SCNI RK Stone: $4.18 left of $2M after the final pre-funded
        # warrant draw).
        if (remaining is not None and capacity
                and 0 < remaining < max(10.0, capacity * 5e-4)):
            remaining = 0.0
        used_pct = (drawn / capacity * 100) if (capacity and capacity > 0) else None
        cards.append({
            "instrument_id": r["instrument_id"],
            "title": _title(r, "equity_line"),
            "registered": _registered_label(r),
            "edgar_url": _instrument_edgar_url(cik, r),
            "parent_shelf": _parent_shelf(
                cik, r.get("registration_accession")
                or r.get("created_accession")),
            "remaining_capacity": remaining,
            "total_capacity": capacity,
            "counterparty": r.get("counterparty_canonical"),
            "sales_total_usd": drawn,
            "used_pct": used_pct,
            "agreement_start_date": _format_date(r.get("created_at")),
            "agreement_end_date": _format_date(
                terms.get("agreement_end_date")
                or (r.get("status_at") if r.get("status") != "active" else None)
            ),
            "terminated": r.get("status") == "terminated",
            "last_update_date": _last_update_date(r),
            **_quality_tags(r),
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
        baby_shelf_threshold_price,
        ib6_remaining,
        is_baby_shelf_restricted,
        raised_under_ib6_last_12mo,
    )
    from .shelf_status import (
        WKSI_UNLIMITED_SHELF_CAPACITY_USD,
        derive_shelf_status,
    )

    # Yahoo's floatShares is fresher than Finviz on low-volume tickers
    # (esp. ADRs — AACG: Yahoo 15.04M matches DT, Finviz 8.84M stale),
    # but is rejected when it exceeds shares outstanding (SCNI: stale
    # 12.4B pre-split). See _resolve_float_shares.
    float_shares = _resolve_float_shares(cik, finviz, latest_os)
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
    # I.B.6 float value is computed at the highest *closing* sale price in
    # the 60 days preceding the offering — not the current market price. Use
    # the 60-day high close alone; never max() it with the live price, which
    # would inflate effective_price (and ib6 raisable) above the closing basis.
    effective_price = high60

    raised_window = raised_under_ib6_last_12mo(cik)
    threshold_price = (
        baby_shelf_threshold_price(float_shares) if float_shares else None
    )
    ib6 = (
        ib6_remaining(cik, float_shares, effective_price)
        if (float_shares and effective_price) else None
    )
    # Baby-shelf classification: filing-time regime stamp from the
    # prospectus cover legend when one exists (per C&DI 116.26 the
    # regime attaches at filing and survives later float moves), with
    # the computed non-affiliate float test as fallback — see
    # baby_shelf.is_baby_shelf_restricted for the full rule. IB6 Float
    # Value below still displays the strict float × price so the
    # diligence reader can see what the strict test would say.
    is_baby_shelf = is_baby_shelf_restricted(
        cik, float_shares, latest_os, effective_price)
    shelf_meta = {s["accession_number"]: s for s in derive_shelf_status(cik)}

    for r in rows:
        if _is_generic_counterparty(r):
            continue
        meta = shelf_meta.get(r.get("created_accession") or "", {})
        # Drop shelves the SEC has effectively closed — Rule 415(a)(5)
        # gives S-3/F-3 shelves a 3-year shelf life; once derived_status
        # is 'expired' (past that window) or 'withdrawn' (issuer filed
        # an RW), no further take-downs are possible and DT doesn't
        # surface the card.
        if meta.get("derived_status") in ("expired", "withdrawn"):
            continue
        terms = r["terms"]
        out = r["outstanding"]
        capacity = _to_float(terms.get("capacity_usd"))
        # Total raised under THIS shelf = drawdowns on the shelf row
        # itself PLUS every drawdown booked against any ATM /
        # s1_offering / equity_line sibling that currently rolls up to
        # this shelf via file_number. The shelf row almost never
        # carries drawn_usd directly (takedowns are booked on the
        # child ATM); without the family rollup, total_amount_raised
        # is always 0 and current_raisable_amount stays pegged at the
        # full registered capacity.
        drawn = _drawn_to_date(cik, r["instrument_id"], out) + \
                _shelf_family_drawn(cik, r["instrument_id"])
        # Prefer capacity - drawn over the walker-emitted
        # remaining_capacity_usd: the latter is often stale (LLM writes
        # back the original capacity ignoring intervening drawdowns).
        # Fall back to remaining_capacity_usd only when capacity itself
        # isn't known.
        is_wksi = (
            "ASR" in (meta.get("form") or "")
            or capacity == WKSI_UNLIMITED_SHELF_CAPACITY_USD
        )
        if is_wksi:
            # WKSI / pay-as-you-go shelf (S-3ASR / F-3ASR, Rule 457(r)):
            # the issuer registers an indeterminate amount and can keep
            # registering more, so the raisable amount is unlimited —
            # DT shows the sentinel here, decoupled from the finite
            # total_shelf_capacity (the cumulative registered figure)
            # and from total_amount_raised. The baby-shelf I.B.6 cap, if
            # the issuer is restricted, still overrides below.
            remaining = float(WKSI_UNLIMITED_SHELF_CAPACITY_USD)
        elif capacity is not None:
            remaining = max(0.0, capacity - drawn)
        else:
            remaining = _to_float(out.get("remaining_capacity_usd"))
        last_banker_raw = _last_banker_for_shelf(
            cik, r["instrument_id"],
            effective=bool(meta.get("effect_date")))
        # Total raised under THIS shelf = drawn against this instrument.
        # Derived-status → user-facing label mapping. EXPIRED and
        # WITHDRAWN are SEC-canonical terminal states: an expired shelf
        # can no longer support take-downs (Rule 415(a)(5) 3-year limit),
        # a withdrawn one was explicitly pulled via RW filing.
        derived = meta.get("derived_status")
        registered_label = {
            "active": "Registered",
            "registered": "Pending Effect",
            "expired": "Expired",
            "withdrawn": "Withdrawn",
        }.get(derived, "Pending Effect")
        cards.append({
            "instrument_id": r["instrument_id"],
            "title": _title(r, "shelf"),
            "registered": registered_label,
            "shelf_status": derived,
            "edgar_url": _shelf_family_url(cik, r),
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
            "last_banker": _short_banker(last_banker_raw),
            # Tier follows the most recent drawdown banker (shelf rows
            # rarely carry their own placement agent). Investor class
            # stays null on shelves — drawdowns can have counterparties
            # but the shelf row itself doesn't.
            "bank_tier": bank_tier(last_banker_raw),
            "investor_class": None,
            "effect_date": meta.get("effect_date"),
            "expiration_date":
                _shelf_expiration(meta.get("effect_date"),
                                  r.get("created_at")),
            "last_update_date": _last_update_date(r),
        })
    return cards


def _median_drawdown_price(cik: int, instrument_id: str) -> float | None:
    """Median of recorded drawdown prices for an instrument.

    Used as a fallback for `final_pricing` on s1_offering cards when
    the walker didn't extract it from the prospectus. Median is more
    robust than mean against the common pattern where the first
    tranche on a 424B5 is priced as a common+warrant unit ($2.57)
    while the actual per-share offering price ($1.75) appears on the
    later tranches.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT price FROM dilution_ledger_drawdowns
                WHERE cik=? AND instrument_id=? AND price IS NOT NULL
                ORDER BY price""",
            (cik, instrument_id),
        ).fetchall()
    prices = [r["price"] for r in rows if r["price"] is not None]
    if not prices:
        return None
    n = len(prices)
    if n % 2:
        return float(prices[n // 2])
    return float((prices[n // 2 - 1] + prices[n // 2]) / 2)


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


def _shelf_family_drawn(cik: int, instrument_id: str) -> float:
    """Total raised against every sibling instrument that currently
    rolls up to this shelf via SEC file_number — ATMs, s1_offerings,
    and equity_lines registered under the same file_number. Shelf
    takedowns are almost always booked on the child instrument row
    (ATM-, S1-, EL-…) rather than on the parent shelf, so without this
    rollup the shelf's total_amount_raised stays 0 and the displayed
    remaining capacity is wrong by exactly the sum of family takedowns.
    Mirrors the file_number walk _last_banker_for_shelf already uses
    for the banker field.

    Per sibling we compute raised-to-date via `_drawn_to_date` semantics
    (see that helper): when a periodic filing has stamped a cumulative
    `drawn_usd_anchor` we trust it and add only the discrete take-downs
    dated AFTER its as-of, instead of re-summing the discrete log. This
    avoids the ATM/ELOC double-count where a quarterly aggregate
    re-reports an interim sale already booked from a prior filing (GCTK
    ATM-2183, SCNI EL-045), while still honouring the case where the
    cumulative anchor EXCEEDS the discrete log because some take-downs
    never landed as individual rows (CGEN's SVB ATM anchored to $15.1M
    vs $13.342M discrete). LEFT JOIN so a sibling whose draws arrived
    purely as amends (zero drawdown rows) still contributes.

    `post_asof_sum` is the discrete total dated strictly after the
    sibling's own `drawn_usd_asof`; `draw_sum` is its full discrete
    total — both computed once in SQL to avoid an N+1 per sibling."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT sib_l.instrument_id AS sib_id,
                      sib_l.outstanding_json AS sib_out,
                      sib_l.terms_json AS sib_terms,
                      COALESCE(SUM(d.amount_usd), 0) AS draw_sum,
                      COALESCE(SUM(CASE WHEN d.event_date >
                                   json_extract(sib_l.outstanding_json,
                                                '$.drawn_usd_asof')
                                   THEN d.amount_usd ELSE 0 END), 0)
                          AS post_asof_sum
                 FROM dilution_ledger shelf_l
                 JOIN dilution_filings shelf_f
                   ON shelf_f.cik = shelf_l.cik
                  AND shelf_f.accession_number = shelf_l.created_accession
                 JOIN dilution_filings sib_f
                   ON sib_f.cik = shelf_l.cik
                  AND sib_f.file_number = shelf_f.file_number
                 JOIN dilution_ledger sib_l
                   ON sib_l.cik = shelf_l.cik
                  AND COALESCE(sib_l.registration_accession,
                               sib_l.created_accession)
                      = sib_f.accession_number
                  AND sib_l.type IN ('atm', 's1_offering', 'equity_line')
                 LEFT JOIN dilution_ledger_drawdowns d
                   ON d.cik = sib_l.cik
                  AND d.instrument_id = sib_l.instrument_id
                WHERE shelf_l.cik = ?
                  AND shelf_l.instrument_id = ?
                GROUP BY sib_l.instrument_id""",
            (cik, instrument_id),
        ).fetchall()
    total = 0.0
    for r in rows:
        out = json.loads(r["sib_out"] or "{}")
        terms = json.loads(r["sib_terms"] or "{}")
        total += _drawn_from_parts(
            out, float(r["draw_sum"]), float(r["post_asof_sum"]),
            capacity=_to_float(terms.get("capacity_usd")),
        )
    return total


def _fully_drawn_clamp(drawn: float, capacity: float | None,
                       out: dict) -> float:
    """Snap a fully-drawn program's raised-to-date up to its capacity.

    A program the filing explicitly recorded as fully exhausted carries a
    stored ``remaining_capacity_usd == 0``; by definition it raised its
    whole capacity. The running ``drawn_usd`` / discrete-draw sum can fall
    a hair short from share-by-share rounding (GCTK Dawson ATM: 8,217,693
    vs 8,230,000 capacity), which surfaced as a $12,307 ghost remaining on
    the ATM card AND under-stated the parent shelf's total raised by the
    same $12,307. Snap to capacity in that case.

    Gated two ways so it only absorbs float noise: (1) the shortfall must
    be under 0.5% of capacity, and (2) the stored remaining must be an
    EXPLICIT 0 — never a non-zero stale-low snapshot, which capacity−drawn
    is correctly allowed to override (the CGEN Leerink case: persisted
    23.9M vs filing-true 34.9M)."""
    if (capacity and capacity > 0
            and _to_float(out.get("remaining_capacity_usd")) == 0
            and 0 <= capacity - drawn < capacity * 0.005):
        return capacity
    return drawn


def _drawn_from_parts(
    out: dict, draw_sum: float, post_asof_sum: float,
    capacity: float | None = None,
) -> float:
    """Raised-to-date for one shelf-family instrument from its
    outstanding dict plus pre-aggregated discrete sums.

    A `drawn_usd_anchor` is a cumulative checkpoint a periodic filing
    stated as of `drawn_usd_asof`; trust it and add only the discrete
    take-downs disclosed AFTER that date (post-period 8-Ks), since the
    checkpoint already subsumes everything on or before it. Without a
    checkpoint, fall back to max(discrete log, running/create-pinned
    drawn_usd) — the latter covers an ATM whose only signal is an
    LLM-pinned cumulative with no discrete rows. A fully-drawn sibling
    (stored remaining 0) snaps up to `capacity` — see _fully_drawn_clamp."""
    anchor = _to_float(out.get("drawn_usd_anchor"))
    if anchor is not None:
        base = anchor + post_asof_sum
    else:
        running = _to_float(out.get("drawn_usd")) or 0.0
        base = max(draw_sum, running)
    return _fully_drawn_clamp(base, capacity, out)


def _drawdown_sums(conn, cik: int, instrument_id: str,
                   after_date: str | None = None) -> tuple[float, float]:
    """``(raw_sum, deduped_sum)`` of an instrument's discrete take-downs.

    Same-(event_date, price) rows are collapsed to the single LARGEST
    amount, not summed: a follow-on filing that re-books an offering with
    its over-allotment exercised re-states the SAME take-down at a larger
    share count / dollar amount on the SAME pricing date — it supersedes,
    it does not add (ACTU SH-2598: $15,000,006 / 2,142,858 sh then
    $17,250,002 / 2,464,286 sh, both @ $7.00 on 2025-09-10; summing
    double-counts the $15M base). Draws on different dates or at different
    prices are summed normally. ``after_date`` restricts to rows strictly
    after it (the ``drawn_usd_anchor`` as-of path)."""
    q = ("SELECT event_date, price, amount_usd "
         "FROM dilution_ledger_drawdowns WHERE cik=? AND instrument_id=?")
    args: list[Any] = [cik, instrument_id]
    if after_date is not None:
        q += " AND event_date > ?"
        args.append(after_date)
    raw = 0.0
    groups: dict[tuple, float] = {}
    for r in conn.execute(q, args).fetchall():
        amt = float(r["amount_usd"] or 0)
        raw += amt
        price = r["price"]
        key = (r["event_date"],
               round(float(price), 6) if price is not None else None)
        groups[key] = max(groups.get(key, 0.0), amt)
    return raw, sum(groups.values())


def _drawn_to_date(cik: int, instrument_id: str, out: dict) -> float:
    """`_drawn_from_parts` for a single instrument (the shelf's OWN
    direct take-downs), querying its discrete drawdown sums directly.
    Same-date/same-price restatements are de-duped — see _drawdown_sums."""
    anchor = _to_float(out.get("drawn_usd_anchor"))
    with get_conn() as conn:
        if anchor is not None:
            _raw, deduped = _drawdown_sums(
                conn, cik, instrument_id, out.get("drawn_usd_asof") or "")
            return anchor + deduped
        raw, deduped = _drawdown_sums(conn, cik, instrument_id)
    drawn_usd = _to_float(out.get("drawn_usd")) or 0.0
    # The stored cumulative wins only when it EXCEEDS the raw discrete log
    # (an LLM-pinned total carrying take-downs that never landed as rows —
    # CGEN's SVB ATM). When it merely EQUALS the raw (double-counted)
    # discrete sum it carries the same restatement double-count, so trust
    # the de-duped discrete total instead (ACTU SH-2598: drawn_usd 32.25M
    # == raw 32.25M, de-duped 17.25M).
    if drawn_usd and abs(drawn_usd - raw) <= max(1.0, raw * 0.001):
        return deduped
    return max(deduped, drawn_usd)


def _drawdown_banker(cik: int, instrument_id: str) -> str | None:
    """Most-recent bank party recorded on a drawdown booked against
    this instrument. The walker sometimes attaches the underwriter to
    the takedown event rather than to the offering row itself (e.g.
    GCTK's Dawson James, recorded on the S-1 takedown, never promoted
    onto the S1-128 placement_agent slot). Surface it as the offering's
    underwriter when the row carries no placement agent."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT drawdown_party_canonical "
            "FROM dilution_ledger_drawdowns "
            "WHERE cik=? AND instrument_id=? "
            "AND drawdown_party_role='bank' "
            "AND drawdown_party_canonical IS NOT NULL "
            "ORDER BY event_date DESC LIMIT 1",
            (cik, instrument_id),
        ).fetchone()
        if row:
            return row["drawdown_party_canonical"]
        # Final fallback: the takedown 8-K that booked the draw (e.g.
        # GCTK Nov-2024 0001493152-24-045979) also minted sibling
        # warrant rows (W-3532..W-3537) carrying the placement agent the
        # takedown party slot lacks. Resolve the agent from a sibling
        # created by the same drawdown accession. Scoped to this cik.
        sib = conn.execute(
            "SELECT placement_agent_canonical "
            "FROM dilution_ledger "
            "WHERE cik=? AND placement_agent_canonical IS NOT NULL "
            "AND created_accession IN ("
            "  SELECT accession_number FROM dilution_ledger_drawdowns "
            "  WHERE cik=? AND instrument_id=?) "
            "LIMIT 1",
            (cik, cik, instrument_id),
        ).fetchone()
    return sib["placement_agent_canonical"] if sib else None


def _last_banker_for_shelf(cik: int, instrument_id: str,
                           effective: bool = True) -> str | None:
    """Most-recent placement agent on a drawdown against this shelf.

    Pulls from the drawdown row itself (each takedown is sold by its
    own banker — Jefferies, B. Riley, etc.). When the shelf has no
    drawdowns of its own (common pattern: drawdowns are recorded on
    the child ATM / s1_offering instrument under the same SEC
    file_number, not on the shelf row), fall back to the most recent
    sibling takedown: its named banker when the filing gave one, else
    the sibling program's standing placement agent — an ATM's sales
    agent isn't restated on every takedown, so a null-party ATM draw
    inherits the ATM's agent (the same one the ATM card shows). Final
    fallback is the shelf row's own placement agent. Siblings are
    restricted to ATM / s1_offering, so ELOC direct-investor takedowns
    never fill the banker slot.
    """
    with get_conn() as conn:
        row = conn.execute(
            """SELECT drawdown_party_canonical
                 FROM dilution_ledger_drawdowns
                WHERE cik=? AND instrument_id=?
                  AND drawdown_party_role='bank'
                  AND drawdown_party_canonical IS NOT NULL
                ORDER BY event_date DESC LIMIT 1""",
            (cik, instrument_id),
        ).fetchone()
        if row:
            return row["drawdown_party_canonical"]
        # Walk the shelf's file_number out to sibling ATM/s1 rows and
        # take the most recent takedown's banker: its named bank party
        # when one was recorded, else the sibling program's standing
        # placement agent (the null-party ATM-takedown case — the agent
        # is implied by the program, not restated per draw).
        row = conn.execute(
            """SELECT COALESCE(
                        CASE WHEN d.drawdown_party_role = 'bank'
                             THEN d.drawdown_party_canonical END,
                        sib_l.placement_agent_canonical) AS banker
                 FROM dilution_ledger shelf_l
                 JOIN dilution_filings shelf_f
                   ON shelf_f.cik = shelf_l.cik
                  AND shelf_f.accession_number = shelf_l.created_accession
                 JOIN dilution_filings sib_f
                   ON sib_f.cik = shelf_l.cik
                  AND sib_f.file_number = shelf_f.file_number
                 JOIN dilution_ledger sib_l
                   ON sib_l.cik = shelf_l.cik
                  AND COALESCE(sib_l.registration_accession,
                               sib_l.created_accession)
                      = sib_f.accession_number
                  AND sib_l.type IN ('atm', 's1_offering')
                 JOIN dilution_ledger_drawdowns d
                   ON d.cik = sib_l.cik
                  AND d.instrument_id = sib_l.instrument_id
                WHERE shelf_l.cik = ?
                  AND shelf_l.instrument_id = ?
                  AND COALESCE(
                        CASE WHEN d.drawdown_party_role = 'bank'
                             THEN d.drawdown_party_canonical END,
                        sib_l.placement_agent_canonical) IS NOT NULL
                ORDER BY d.event_date DESC LIMIT 1""",
            (cik, instrument_id),
        ).fetchone()
        if row:
            return row["banker"]
        # Drawdown-FREE sibling fallback: an undrawn standing program
        # still names its agent (CETY's Roth ATM had zero takedowns, so
        # the drawdown-JOIN tier above returns nothing and the shelf
        # showed last_banker=None against DT's 'Roth'). Gated on the
        # shelf being EFFECTIVE — a just-filed pending registration
        # (CGEN May-2026) must not inherit its sibling's standing agent
        # before any takedown program is live under it.
        if effective:
            row = conn.execute(
                """SELECT sib_l.placement_agent_canonical AS banker
                     FROM dilution_ledger shelf_l
                     JOIN dilution_filings shelf_f
                       ON shelf_f.cik = shelf_l.cik
                      AND shelf_f.accession_number = shelf_l.created_accession
                     JOIN dilution_filings sib_f
                       ON sib_f.cik = shelf_l.cik
                      AND sib_f.file_number = shelf_f.file_number
                     JOIN dilution_ledger sib_l
                       ON sib_l.cik = shelf_l.cik
                      AND COALESCE(sib_l.registration_accession,
                                   sib_l.created_accession)
                          = sib_f.accession_number
                      AND sib_l.type IN ('atm', 's1_offering')
                      -- A dead program's agent is not the shelf's "last
                      -- banker": a rolled-over-then-terminated legacy ATM
                      -- must not outrank live programs (SCNI round-6: the
                      -- terminated Oct-2020 BofA ATM, re-pointed at the
                      -- Aug-2023 F-3 by the redisclosure rollover, won
                      -- this fallback over the shelf's own Wainwright
                      -- takedowns).
                      AND sib_l.status NOT IN ('terminated', 'expired')
                    WHERE shelf_l.cik = ?
                      AND shelf_l.instrument_id = ?
                      AND sib_l.placement_agent_canonical IS NOT NULL
                    ORDER BY sib_l.created_at DESC LIMIT 1""",
                (cik, instrument_id),
            ).fetchone()
            if row:
                return row["banker"]
        # Final fallback: the shelf row's own placement agent (rare).
        row = conn.execute(
            "SELECT placement_agent_canonical "
            "FROM dilution_ledger WHERE cik=? AND instrument_id=?",
            (cik, instrument_id),
        ).fetchone()
        if row:
            return row["placement_agent_canonical"]
    return None


__all__ = [
    "all_pending_s1_offerings",
    "atm_cards",
    "convertible_note_cards",
    "equity_line_cards",
    "pending_s1_offerings",
    "preferred_cards",
    "s1_offering_cards",
    "shelf_cards",
    "warrant_cards",
]
