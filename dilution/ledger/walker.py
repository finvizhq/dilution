"""Chronological filing walker — the orchestrator.

Steps per CIK:
  1. seed_ledger() runs the earliest periodic filing through the
     overhang prompt and creates initial instruments.
  2. Walk every dilution-relevant filing chronologically. For each:
     a. Build a ledger view (open + recently-closed).
     b. Call walker_llm.walk_filing → MutationList.
     c. Validate (validate.validate_mutations).
     d. Apply (store.apply_mutations).
     e. If periodic: anchor-reconcile against the issuer's overhang
        table; record diffs and apply correction mutations.
     f. mark_walked.

Same-day filing tiebreak: periodic anchors first, then registrations,
then 424B (priced), then 8-K/6-K/FWP/425. EFFECT/RW are excluded
from the walker (handled by shelf_status / projection layer).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from db import get_conn, now_iso
from dilution.llm_provider import make_async_client

from .anchor import reconcile_against_periodic
from .mutations import MutationList, Mutation
from .seed import seed_ledger
from .store import (
    apply_mutations,
    get_drawdowns_by_instrument,
    get_open_instruments,
    get_walk_state,
    mark_walked,
    record_anchor_diffs,
    reset_walk_state,
)
from .validate import validate_mutations
from .view import render_ledger_view
from .walker_llm import pipeline_version, walk_filing

log = logging.getLogger(__name__)


# Form-priority tiebreak. Lower runs first within the same filing_date.
# Lock-step with LEDGER_REWORK_PLAN step 8.
_FORM_PRIORITY: dict[str, int] = {
    "10-K": 0, "10-K/A": 0, "20-F": 0, "20-F/A": 0,
    "40-F": 0, "40-F/A": 0,
    "10-Q": 1, "10-Q/A": 1,
    "S-3": 2, "S-3/A": 2, "S-3ASR": 2, "S-3MEF": 2,
    "F-3": 2, "F-3/A": 2, "F-3ASR": 2,
    "S-1": 2, "S-1/A": 2, "S-1MEF": 2,
    "F-1": 2, "F-1/A": 2,
    "POS AM": 2, "POSASR": 2, "S-4": 2, "F-4": 2,
    "DEF 14A": 3, "DEFM14A": 3, "DEFA14A": 3,
    "DEFC14A": 3, "DEFR14A": 3,
    "PRE 14A": 3, "PREM14A": 3, "PRER14A": 3, "PREC14A": 3,
    "424B1": 4, "424B2": 4, "424B3": 4, "424B4": 4,
    "424B5": 4, "424B7": 4, "424B8": 4,
    "8-K": 5, "8-K/A": 5, "6-K": 5, "6-K/A": 5,
    "FWP": 5, "425": 5,
    "1-A": 5, "1-A/A": 5, "1-K": 5, "1-SA": 5, "1-U": 5,
}

# Forms the walker explicitly skips — no body to process.
_SKIPPED_FORMS = frozenset({"EFFECT", "RW"})

# Periodic forms that trigger anchor reconciliation after the walker.
_PERIODIC_FORMS = frozenset({
    "10-K", "10-K/A", "10-Q", "10-Q/A",
    "20-F", "20-F/A", "40-F", "40-F/A",
})


def _form_rank(form: str | None) -> int:
    if not form:
        return 99
    f = form.upper().strip()
    if f in _FORM_PRIORITY:
        return _FORM_PRIORITY[f]
    # Prefix-match for amendments and obscure variants.
    for prefix, rank in _FORM_PRIORITY.items():
        if f.startswith(prefix):
            return rank
    return 99


@dataclass
class WalkSummary:
    cik: int
    seed_case: str
    walked: int = 0
    skipped: int = 0
    mutations_applied: int = 0
    mutations_rejected: int = 0
    instruments_created: int = 0
    redisclosures: int = 0
    drawdowns_recorded: int = 0
    anchor_diffs: int = 0
    errors: int = 0
    error_accessions: list[str] = field(default_factory=list)


# ─── DB helpers ─────────────────────────────────────────────────────
def _list_filings(cik: int, since_date: str) -> list[dict]:
    """All filings for cik with raw text in scope, ordered chronologically
    with the form_priority tiebreak. Excludes EFFECT/RW."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT f.accession_number, f.form, f.filing_date,
                      f.report_date, f.items
                 FROM dilution_filings f
                INNER JOIN dilution_raw r
                   ON r.accession_number = f.accession_number
                WHERE f.cik = ? AND f.filing_date >= ?
                GROUP BY f.accession_number
                ORDER BY f.filing_date ASC""",
            (cik, since_date),
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        if (r["form"] or "").upper() in _SKIPPED_FORMS:
            continue
        out.append(dict(r))
    out.sort(key=lambda f: (
        f["filing_date"],
        _form_rank(f["form"]),
        f["accession_number"],
    ))
    return out


def _load_filing_text(accession: str) -> str:
    """Concatenate cached doc text for one accession.

    Mirrors `dilution.extractors.base.load_filing_text` but lives here
    so the walker has zero coupling to the legacy extractor module.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT content_md FROM dilution_raw
                WHERE accession_number = ?
                ORDER BY doc_name""",
            (accession,),
        ).fetchall()
    return "\n\n".join(r["content_md"] for r in rows)


# ─── Walker ─────────────────────────────────────────────────────────
async def _walk_async(
    *, cik: int, ticker: str, since_date: str,
    force: bool, concurrency: int,
) -> WalkSummary:
    summary = WalkSummary(cik=cik, seed_case="not_run")

    if force:
        log.info("walker --force: dropping ledger for cik=%s", cik)
        reset_walk_state(cik)

    from dilution.company import get_unit_context
    unit_ctx = get_unit_context(cik)

    client = make_async_client()
    try:
        # Seed
        seed = await seed_ledger(
            cik=cik, ticker=ticker, since_date=since_date,
            client=client, unit_ctx=unit_ctx,
        )
        summary.seed_case = seed.case
        summary.instruments_created += seed.instruments_created
        if seed.accession:
            mark_walked(cik, seed.accession,
                        seed.as_of_date or since_date,
                        pipeline_version())
            log.info("seed case=%s — %s", seed.case, seed.accession)

        # Walk
        filings = _list_filings(cik, since_date)
        # When force=False, skip filings already in walk_state.
        last_acc = None
        if not force:
            state = get_walk_state(cik)
            last_acc = (state or {}).get("last_processed_accession")
        skip_until = last_acc
        log.info("walker: %d filings to walk (cik=%s)", len(filings), cik)

        sem = asyncio.Semaphore(concurrency)

        # Walker is intentionally serial on the apply path because the
        # ledger view depends on the accumulated state from prior
        # filings. The LLM call itself can be concurrent across filings,
        # but we'd then need to re-validate every result against the
        # post-batch state. v1 keeps it simple: fully serial. Concurrency
        # arg accepted for future use.
        passed_seed = (skip_until is None) or (skip_until == seed.accession)
        for f in filings:
            if not passed_seed:
                if f["accession_number"] == skip_until:
                    passed_seed = True
                continue
            if f["accession_number"] == seed.accession:
                # Already processed by seed (don't double-walk).
                continue
            if not force and skip_until and f["accession_number"] == skip_until:
                # safety
                continue
            try:
                async with sem:
                    walked = await _walk_one(
                        cik=cik, ticker=ticker, filing=f,
                        client=client, unit_ctx=unit_ctx,
                        summary=summary,
                    )
                if walked:
                    summary.walked += 1
                else:
                    summary.skipped += 1
                mark_walked(cik, f["accession_number"],
                            f["filing_date"], pipeline_version())
            except Exception as exc:
                log.exception("walker error on %s: %s",
                              f["accession_number"], exc)
                summary.errors += 1
                summary.error_accessions.append(f["accession_number"])
    finally:
        await client.close()
    return summary


async def _walk_one(
    *, cik: int, ticker: str, filing: dict, client,
    unit_ctx: dict | None, summary: WalkSummary,
) -> bool:
    """Process one filing. Returns True if any mutations were applied."""
    accession = filing["accession_number"]
    form = filing["form"]
    filing_date = filing["filing_date"]

    text = _load_filing_text(accession)
    if not text:
        log.warning("  skip %s — no raw text", accession)
        return False

    open_rows = get_open_instruments(cik)
    ledger_view = render_ledger_view(
        open_rows,
        drawdowns_by_instrument=get_drawdowns_by_instrument(cik),
    )

    from ._llm_utils import unit_preamble as _preamble
    preamble = _preamble(unit_ctx)

    mlist = await walk_filing(
        client=client, unit_preamble=preamble,
        ledger_view=ledger_view, form=form, filing_date=filing_date,
        accession=accession, items=filing.get("items"),
        period_of_report=filing.get("report_date"),
        filing_text=text,
        active_rows=open_rows,
    )
    n_mutations = len(mlist.mutations)
    log.info("  [%s] %s %s — %d mutations",
             filing_date, form, accession, n_mutations)
    if not mlist.mutations:
        return False

    # Validate against the current ledger snapshot.
    snapshot = {row["instrument_id"]: row for row in open_rows}
    report = validate_mutations(mlist.mutations, snapshot, filing_form=form)
    apply_result = apply_mutations(
        cik=cik, ticker=ticker, accession=accession, form=form,
        filing_date=filing_date, mutations=mlist.mutations,
        pre_validated_report=report,
    )
    summary.mutations_applied += apply_result.accepted
    summary.mutations_rejected += apply_result.rejected
    summary.instruments_created += len(apply_result.created_ids)
    summary.redisclosures += apply_result.redisclosures
    summary.drawdowns_recorded += apply_result.drawdowns_recorded

    # Anchor reconciliation for periodic filings
    if (form or "").upper().split("/")[0] in _PERIODIC_FORMS:
        await _anchor_one(
            cik=cik, ticker=ticker, filing=filing, client=client,
            unit_ctx=unit_ctx, summary=summary,
        )
    return True


async def _anchor_one(
    *, cik: int, ticker: str, filing: dict, client,
    unit_ctx: dict | None, summary: WalkSummary,
) -> None:
    """Run the overhang prompt against the periodic filing, diff
    against the ledger, persist diffs + apply correction mutations."""
    accession = filing["accession_number"]
    as_of = filing.get("report_date") or filing["filing_date"]
    log.info("  anchor %s as_of=%s", accession, as_of)
    from ._overhang_extract import extract_overhang_rows
    try:
        overhang_rows = await extract_overhang_rows(
            accession=accession, form=filing["form"],
            filing_date=filing["filing_date"], report_date=as_of,
            cik=cik, client=client, unit_ctx=unit_ctx,
        )
    except Exception as exc:
        log.warning("  anchor extract failed for %s: %s", accession, exc)
        return

    if not overhang_rows:
        return

    open_rows = get_open_instruments(cik)
    result = reconcile_against_periodic(
        cik=cik, accession=accession,
        filing_date=filing["filing_date"], as_of_date=as_of,
        filing_overhang=overhang_rows, ledger_open=open_rows,
    )
    if result.diffs:
        record_anchor_diffs(cik, accession, as_of,
                            [d.to_dict() for d in result.diffs])
        summary.anchor_diffs += len(result.diffs)
    if result.correction_mutations:
        apply_result = apply_mutations(
            cik=cik, ticker=ticker, accession=accession,
            form=filing["form"], filing_date=filing["filing_date"],
            mutations=result.correction_mutations,
        )
        summary.mutations_applied += apply_result.accepted
        summary.mutations_rejected += apply_result.rejected
        summary.instruments_created += len(apply_result.created_ids)
        summary.redisclosures += apply_result.redisclosures
        log.info("  anchor corrections applied=%d rejected=%d",
                 apply_result.accepted, apply_result.rejected)


# ─── Public entry points ─────────────────────────────────────────────
def walk_ticker(
    cik: int, ticker: str, *,
    since_date: str,
    force: bool = False,
    concurrency: int = 1,
) -> WalkSummary:
    """Synchronous entry point. Walks all filings for a CIK in one go.

    `force=True` drops the existing ledger first. Otherwise the walker
    resumes from `dilution_walk_state.last_processed_accession`.
    """
    from dilution.llm_provider import require_api_key
    require_api_key()
    return asyncio.run(_walk_async(
        cik=cik, ticker=ticker, since_date=since_date,
        force=force, concurrency=max(1, int(concurrency)),
    ))


__all__ = [
    "WalkSummary",
    "walk_ticker",
]
