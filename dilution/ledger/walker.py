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
import re
from dataclasses import dataclass, field
from typing import Any

from db import get_conn, now_iso
from dilution.openai_client import make_async_client

from .anchor import (
    corroborate_closes,
    extract_stated_note_balances,
    reconcile_against_periodic,
)
from datetime import date as _date, datetime as _datetime

from .mutations import (
    AmendS1Offering, AmendWarrant, ApplySplit,
    CreateConvertible, CreateS1Offering, CreateWarrant,
    MutationList, Mutation, amend_from_dict, fmt_mutation,
)
from .seed import seed_ledger
from .store import (
    apply_mutations,
    close_converted_preferred,
    close_retired_debt,
    ensure_walk_tables,
    get_drawdowns_by_instrument,
    get_open_instruments,
    get_walked_accessions,
    mark_walked,
    record_anchor_diffs,
    reopen_instruments,
    reset_walk_state,
    seed_walked_from_positional,
)
from ._llm_utils import normalize_filing_text
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
    "F-10": 2, "F-10/A": 2, "F-10EF": 2,
    "POS AM": 2, "POSASR": 2, "S-4": 2, "F-4": 2,
    "DEF 14A": 3, "DEFM14A": 3, "DEFA14A": 3,
    "DEFC14A": 3, "DEFR14A": 3,
    "PRE 14A": 3, "PREM14A": 3, "PRER14A": 3, "PREC14A": 3,
    "424B1": 4, "424B2": 4, "424B3": 4, "424B4": 4,
    "424B5": 4, "424B7": 4, "424B8": 4,
    "SUPPL": 4,  # MJDS prospectus supplement (Canadian analog of 424B5)
    "8-K": 5, "8-K/A": 5, "6-K": 5, "6-K/A": 5,
    "FWP": 5, "425": 5,
    "1-A": 5, "1-A/A": 5, "1-K": 5, "1-SA": 5, "1-U": 5,
}

# Forms the walker explicitly skips — no body to process.
_SKIPPED_FORMS = frozenset({"EFFECT", "RW"})

# Periodic forms that trigger anchor reconciliation after the walker.
# 6-K is included because FPIs disclose their interim outstanding-
# instruments tables there (there is no 10-Q analog for an FPI), and
# walker_llm already treats 6-K as periodic when lifting the create-
# dedup window — so without an anchor pass, instrument drift surfaced in
# a 6-K was suppressed-as-redisclosure yet never reconciled until the
# next annual 20-F. `_walk_one` further gates 6-K anchoring on the
# presence of interim-financial-statement markers (see
# `_sixk_carries_financials`) so press-release / investor-deck
# furnishings don't fire the six overhang specialists for nothing.
_PERIODIC_FORMS = frozenset({
    "10-K", "10-K/A", "10-Q", "10-Q/A",
    "20-F", "20-F/A", "40-F", "40-F/A",
    "6-K", "6-K/A",
})

# Interim-financial-statement markers that signal a 6-K carries an
# authoritative outstanding-instruments table worth anchor-reconciling.
# FPIs furnish a wide range of 6-Ks (press releases, investor decks,
# dividend notices); only the interim-financials variant has the
# overhang table. Requiring a positive marker keeps the anchor's six
# overhang specialists from firing on every furnishing. A missed
# financials 6-K just defers reconciliation to the next 20-F (the prior
# behavior); a false match only wastes an empty extraction.
_SIXK_FINANCIALS_MARKERS = (
    "condensed consolidated",
    "interim financial",
    "interim condensed",
    "unaudited condensed",
    "statement of financial position",
    "statements of financial position",
    "consolidated statement of operations",
    "consolidated statements of operations",
    "notes to the financial statements",
    "notes to financial statements",
)


def _sixk_carries_financials(text: str) -> bool:
    """True when a 6-K body shows interim-financial-statement markers —
    the signal that it carries an outstanding-instruments table worth
    anchor-reconciling. See `_SIXK_FINANCIALS_MARKERS`."""
    low = text.lower()
    return any(marker in low for marker in _SIXK_FINANCIALS_MARKERS)


# Full-preferred-conversion signal. A periodic filing AFFIRMS that all
# preferred stock automatically/mandatorily converted to common with NONE
# remaining outstanding (the Nasdaq-equity-compliance pattern — KSCP's
# Series A/B/M/S converted 2024-05-15, "no shares of Preferred Stock
# outstanding after the Preferred Stock Conversion Date"). The overhang LLM
# re-matches the named series in the conversion narrative without flagging
# is_terminated, so the anchor never closes them. Two gates, BOTH required,
# keep this from firing on a partial per-series conversion, a "convertible
# preferred" boilerplate description, or a conditional/pro-forma ("would be
# no preferred outstanding"):
#   (1) ZERO affirmation — "no (shares of) preferred stock ... outstanding"
#       (a live-preferred issuer states the count instead); and
#   (2) conversion ACTUALITY — "preferred stock ... automatically/mandatorily
#       converted into ... common".
# \xa0 non-breaking spaces appear in EDGAR dates, so \s (not literal space).
_PFD_ZERO_RE = re.compile(
    r"no\s+(?:shares\s+of\s+)?preferred\s+stock\b[\s\S]{0,80}?\boutstanding",
    re.I)
_PFD_CONV_RE = re.compile(
    r"preferred\s+stock\b[\s\S]{0,300}?(?:automatically|mandatorily)\s+"
    r"converted\s+into[\s\S]{0,150}?common", re.I)
_PFD_DATE_RES = (
    re.compile(r"\b([A-Z][a-z]+\s+\d{1,2},\s*\d{4})\b[\s\S]{0,60}?"
               r"[Cc]onversion Date"),
    re.compile(r"\bon\s+([A-Z][a-z]+\s+\d{1,2},\s*\d{4})\b[\s\S]{0,300}?"
               r"(?:automatically|mandatorily)\s+converted", re.I),
)


def _full_preferred_conversion_date(text: str) -> _date | None:
    """The conversion date when `text` affirms ALL preferred converted to
    common with none outstanding; None otherwise. The date scopes the close
    (only preferreds issued on/before it) so a later re-issuance is spared —
    see ``store.close_converted_preferred``."""
    if not text or not (_PFD_ZERO_RE.search(text)
                        and _PFD_CONV_RE.search(text)):
        return None
    for rx in _PFD_DATE_RES:
        m = rx.search(text)
        if not m:
            continue
        raw = m.group(1).replace("\xa0", " ").replace(",", "")
        try:
            return _datetime.strptime(raw, "%B %d %Y").date()
        except ValueError:
            continue
    return None


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
    # Subset of `skipped`: 424B / S-1 / S-3 filings short-circuited by
    # the file_number resale pre-screen (registration_family). These
    # are resale prospectuses for selling holders and the LLM would
    # have produced zero mutations anyway — skipping them saves a
    # walker round-trip per resale 424B.
    skipped_resale: int = 0
    # Subset of `skipped`: 8-K filings short-circuited by the item-code
    # pre-screen (item_classification) — no dilution-relevant EDGAR item
    # AND no substantive exhibit attached, so no cap-table event is
    # possible. Saves a walker round-trip per non-dilutive 8-K (earnings
    # without an EX-99 / officer-change / voting / listing 8-Ks).
    skipped_no_dilution: int = 0
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
    Whitespace-normalized before it reaches the prompt — see
    `normalize_filing_text` (token savings + Gemini 400 guard).
    """
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT content_md FROM dilution_raw
                WHERE accession_number = ?
                ORDER BY doc_name""",
            (accession,),
        ).fetchall()
    return normalize_filing_text(
        "\n\n".join(r["content_md"] for r in rows))


def _build_event_stream(
    filings: list[dict], splits: list[Any],
) -> list[dict]:
    """Interleave filings and externally-sourced splits into one
    chronologically-sorted event stream.

    Sort key: (event_date, kind_rank, accession). Splits get
    kind_rank=-1 so they fire BEFORE any filing on the same date —
    required so that a filing dated 2024-03-15 reporting events from
    2024-03-12 sees the 2024-03-12 split already in the ledger view.

    Each output dict carries `kind` ("filing" | "split") plus the
    fields needed by the corresponding handler.
    """
    events: list[dict] = []
    for f in filings:
        events.append({
            "kind": "filing",
            "event_date": f["filing_date"],
            "accession": f["accession_number"],
            "filing": f,
        })
    for s in splits:
        synthetic_acc = f"split:{s.effective_date}:{s.source}"
        events.append({
            "kind": "split",
            "event_date": s.effective_date,
            "accession": synthetic_acc,
            "split": s,
        })

    def sort_key(ev: dict) -> tuple:
        kind_rank = -1 if ev["kind"] == "split" else 1
        # Within filings, preserve the existing form-priority order
        # (periodic anchors → registrations → 424B → 8-K). Splits at
        # rank -1 always lead the day.
        sub_rank = (_form_rank(ev["filing"]["form"])
                    if ev["kind"] == "filing" else 0)
        return (ev["event_date"], kind_rank, sub_rank, ev["accession"])

    events.sort(key=sort_key)
    return events


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

    # Merge filings + externally-sourced splits into one event stream.
    # Splits are walker-applied at their exchange-effective date BEFORE
    # same-day filings, so any pre-existing instrument has been scaled by
    # the time downstream filings build their ledger view. Built up front
    # because both the resume migration and the walk loop need it.
    ensure_walk_tables()
    filings = _list_filings(cik, since_date)
    from dilution.splits import load_splits
    splits = load_splits(cik, since_date=since_date)
    events = _build_event_stream(filings, splits)

    # Resume: process any in-scope FILING not already in the per-accession
    # walked set. This is robust to back-filled filings — `_list_filings`
    # INNER-JOINs dilution_raw, so a filing whose raw text arrives only
    # after the first walk re-appears here at its (earlier) filing_date.
    # The old positional last_processed_accession marker skipped
    # everything sorting before the resume point, so such a filing was
    # skipped forever (only --force recovered it). force=True walks all.
    walked: set[str] = set()
    if not force:
        # One-time migration off the legacy positional marker: seed the
        # walked set with the prefix it implied, so the first incremental
        # run after this change doesn't re-walk the entire history onto a
        # non-empty ledger. No-op once the set is populated.
        seed_walked_from_positional(
            cik, [ev["accession"] for ev in events if ev["kind"] == "filing"],
        )
        walked = get_walked_accessions(cik)
    log.info("walker: %d events (filings=%d splits=%d cik=%s); "
             "%d filing(s) already walked",
             len(events), len(filings), len(splits), cik, len(walked))

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

        sem = asyncio.Semaphore(concurrency)

        # Walker is intentionally serial on the apply path because the
        # ledger view depends on the accumulated state from prior
        # filings. The LLM call itself can be concurrent across filings,
        # but we'd then need to re-validate every result against the
        # post-batch state. v1 keeps it simple: fully serial. Concurrency
        # arg accepted for future use.
        for ev in events:
            acc = ev["accession"]
            if acc == seed.accession:
                # Already processed by seed_ledger above (runs every walk).
                continue
            if ev["kind"] == "filing" and not force and acc in walked:
                continue  # already walked — resume skip
            try:
                if ev["kind"] == "split":
                    # Synthetic event — no LLM call, idempotent via
                    # `_split_already_applied`. Replayed every run (not
                    # tracked in the walked set) so a late-fetched split
                    # still lands; the store dedups same-direction splits
                    # within ±30d.
                    _walk_split(
                        cik=cik, ticker=ticker, split=ev["split"],
                        summary=summary,
                    )
                    summary.walked += 1
                else:
                    f = ev["filing"]
                    async with sem:
                        walked_flag = await _walk_one(
                            cik=cik, ticker=ticker, filing=f,
                            client=client, unit_ctx=unit_ctx,
                            summary=summary,
                        )
                    if walked_flag:
                        summary.walked += 1
                    else:
                        summary.skipped += 1
                    mark_walked(cik, acc,
                                f["filing_date"], pipeline_version())
            except Exception as exc:
                log.exception("walker error on %s: %s", acc, exc)
                summary.errors += 1
                summary.error_accessions.append(acc)
    finally:
        await client.close()
    return summary


def _build_attribution_block(parent: dict, form: str) -> str:
    """Render the file_number-derived primary-attribution hint.

    Speaks directly to the LLM in imperative form. The hint is hard
    (not advisory) because file_number is the SEC-canonical linkage —
    it cannot be wrong unless EDGAR's submissions index is corrupted,
    which doesn't happen. We make the rule explicit precisely because
    the walker's existing prompt has historically relied on the LLM
    parsing the cover page's "issued pursuant to our registration
    statement on Form S-3 (No. 333-XXXXXX)" prose, with a
    non-negligible failure rate (XTIA SH-008 has 8 of 10 take-downs
    missing from drawdowns table for exactly this reason — the LLM
    couldn't cleanly map the prospectus prose back to the ledger row).
    """
    fnum = parent.get("file_number") or "?"
    iid = parent.get("instrument_id") or "?"
    label = parent.get("label") or "?"
    form_upper = (form or "").upper()
    is_amendment = "/A" in form_upper
    if is_amendment:
        # S-3/A / S-1/A are amendments to the existing primary
        # registration. Walker should typically NOT emit
        # create_instrument for these — at most an amend_instrument
        # if capacity changes.
        return (
            f"This filing's SEC file_number is **{fnum}**, the same "
            f"registration as primary instrument **{iid}** ({label}). "
            f"This is an amendment / post-effective filing of an "
            f"already-tracked registration — do NOT emit "
            f"`create_instrument` for a new shelf or s1_offering. "
            f"At most emit `amend_instrument({iid}, …)` if the "
            f"filing changes capacity, anti-dilution terms, or "
            f"placement-agent identity."
        )
    return (
        f"This filing's SEC file_number is **{fnum}**, the same "
        f"registration as primary instrument **{iid}** ({label}). "
        f"Per SEC linkage, this 424B is a primary take-down from "
        f"that shelf — emit `record_event(drawdown, instrument_id="
        f"{iid!r}, ...)` with the proceeds, share count, and "
        f"placement agent from the cover page. Do NOT emit "
        f"`create_instrument(shelf|s1_offering|equity)` for this "
        f"take-down; the underlying shelf already exists. Warrants "
        f"or convertibles issued alongside the take-down ARE "
        f"separate instruments and should still emit their own "
        f"`create_instrument` rows."
    )


def _build_registration_followon_block(parent: dict, form: str) -> str:
    """Hard hint for S-1 / S-3 / F-1 / F-3 (and their /A, MEF, ASR
    variants) when a parent shelf or s1_offering with the same SEC
    file_number is already tracked.

    Without this hint the walker LLM emits a fresh `create_instrument`
    every time a new amendment / MEF / restatement of the same
    registration is filed — producing per-revision duplicates of the
    same logical shelf or s1_offering. file_number is the SEC-canonical
    linkage, available deterministically before the LLM call; we pass
    the verdict in as instruction so prose-parsing the cover page's
    "filed pursuant to our registration statement on Form S-3 (No.
    333-XXXXXX)" boilerplate isn't needed."""
    fnum = parent.get("file_number") or "?"
    iid = parent.get("instrument_id") or "?"
    label = parent.get("label") or "?"
    return (
        f"This filing's SEC file_number is **{fnum}**, the same "
        f"registration as primary instrument **{iid}** ({label}), "
        f"which is already in the ledger. This is a follow-on / "
        f"amendment / MEF / post-effective / restatement of an "
        f"existing registration — do NOT emit `create_instrument` for "
        f"a new shelf or s1_offering. Emit `amend_instrument({iid}, "
        f"…)` ONLY if this filing materially changes registered "
        f"capacity (S-3MEF capacity bump, S-1/A deal-size revision), "
        f"extends an expiration, modifies anti-dilution terms, or "
        f"replaces the placement agent. If the filing's effect is "
        f"purely administrative (typo correction, exhibit refresh), "
        f"emit nothing for the parent instrument. Warrants or "
        f"convertibles disclosed alongside the filing ARE separate "
        f"instruments and should still emit their own "
        f"`create_instrument` rows."
    )


def _registration_family_is_resale(cik: int, accession: str) -> bool:
    """True when an EARLIER registration filing under the same 333-
    file_number is fee-table resale (Rule 457(c)/(g)). Used to propagate
    a parent registration's resale verdict to its amendment (/A, POS AM)
    so a resale registration's amendment doesn't mint a phantom
    s1_offering/shelf. Anchored on the parent's deterministic fee table
    rather than shelf-absence, so a primary shelf whose parent S-3
    predates the ingest window is NOT wrongly skipped."""
    from .registration_family import family_registration_accessions
    from ._exhibit_provisions import FEE_TABLE_FORMS, classify_fee_table
    for acc, form in family_registration_accessions(cik, accession):
        if ((form or "").upper() in FEE_TABLE_FORMS
                and classify_fee_table(acc) == "resale"):
            return True
    return False


def _reroute_s1_create_to_amend(
    mlist: MutationList, *, cik: int, accession: str,
    form: str, snapshot: dict,
) -> MutationList:
    """Deterministically rewrite a `create_s1_offering` into
    `amend_s1_offering` when this filing's SEC file_number already maps
    to an existing s1_offering row.

    A 424B4 that prices — or an S-1/A that amends — an already-registered
    offering describes the SAME offering; minting a new s1_offering row
    spawns a duplicate card. `_build_attribution_block` already computes
    and hands the LLM this exact instruction, but the model ignores it
    run-to-run (it created a duplicate in one GCTK walk, amended in the
    next). This enforces the routing in code so it isn't an LLM judgment.

    Mirrors the post-LLM rewrite pattern in walker_llm (`_apply_guards`,
    `_propagate_banker`); runs before validate so the rewritten amend is
    validated + applied normally. Gated on a parent that resolves to an
    s1_offering — a base S-1/F-1 that first creates the row finds no
    prior parent (the row doesn't exist yet) and is left alone.
    """
    if not any(isinstance(m, CreateS1Offering) for m in mlist.mutations):
        return mlist
    from .registration_family import primary_shelf_for_filing
    parent = primary_shelf_for_filing(cik, accession)
    if not parent:
        return mlist
    parent_id = parent.get("instrument_id")
    prow = snapshot.get(parent_id)
    if prow is None or prow.get("type") != "s1_offering":
        return mlist

    is_pricing = (form or "").upper() == "424B4"
    out: list[Mutation] = []
    for m in mlist.mutations:
        if not isinstance(m, CreateS1Offering):
            out.append(m)
            continue
        if is_pricing:
            # 424B4 prices the offering → the create's amount IS the
            # final priced size; map to the final_* cover fields.
            amend = AmendS1Offering(
                instrument_id=parent_id,
                event_date=m.event_date,
                final_deal_size=m.anticipated_deal_size,
                final_warrant_coverage_pct=m.warrant_coverage_pct,
                warrant_strike=m.warrant_strike,
                placement_agent_canonical=m.placement_agent_canonical,
                sold_to_date=m.sold_to_date,
            )
        else:
            # S-1/A amendment → still anticipated (not yet priced).
            amend = AmendS1Offering(
                instrument_id=parent_id,
                event_date=m.event_date,
                anticipated_deal_size=m.anticipated_deal_size,
                warrant_strike=m.warrant_strike,
                warrant_coverage_pct=m.warrant_coverage_pct,
                placement_agent_canonical=m.placement_agent_canonical,
                sold_to_date=m.sold_to_date,
            )
        out.append(amend)
        log.info(
            "  %s — rerouted create_s1_offering → amend_s1_offering %s "
            "(file_number maps to existing offering)",
            accession, parent_id,
        )
    return MutationList(mutations=out)


def _reroute_warrant_create_to_amend(
    mlist: MutationList, *, cik: int, accession: str,
    form: str, snapshot: dict,
) -> MutationList:
    """Deterministically rewrite a `create_warrant` into `amend_warrant`
    when this filing's SEC file_number already maps to a sibling warrant
    on the same registration chain.

    An S-1/A or 424B4 that amends / prices an offering with warrant
    coverage often emits a fresh `create_warrant` carrying the new
    terms (anticipated → priced strike, updated count), when the
    predecessor S-1 / S-1/A already created the same logical warrant
    tranche under the SAME file_number. The strikes can differ by
    20-60% across the anticipated → priced step, defeating the
    post-LLM strike-dedup guard's ±2% tolerance (`walker_llm._is_dup_
    create`). Result without the reroute: one logical warrant tranche
    surfaces as two cards (XTIA June 2025 Common Warrants W-2736 +
    W-2737, 54% strike gap).

    Mirrors `_reroute_s1_create_to_amend` for the sibling-warrant case.
    Matches by pre_funded flag (common vs pre-funded) — when exactly
    ONE active sibling warrant of the same kind exists under the
    parent's file_number, reroute the create's terms onto an amend
    of that sibling. Zero siblings (new tranche) or multiple siblings
    of the same kind (ambiguous) leave the create alone for the
    existing dedup guard / regular flow to handle.
    """
    if not any(isinstance(m, CreateWarrant) for m in mlist.mutations):
        return mlist
    from .registration_family import primary_shelf_for_filing
    parent = primary_shelf_for_filing(cik, accession)
    if not parent:
        return mlist
    parent_file_number = parent.get("file_number")
    if not parent_file_number:
        return mlist

    # Find active sibling warrants under the parent's file_number,
    # EXCLUDING any created from this same accession (so a base S-1
    # that emitted two creates in one filing — its OWN pre-funded +
    # common — doesn't see itself as a sibling).
    import json as _json
    with get_conn() as conn:
        sib_rows = conn.execute(
            """SELECT l.instrument_id, l.terms_json, l.label, l.created_at
                 FROM dilution_ledger l
                 JOIN dilution_filings f
                   ON f.accession_number = l.created_accession
                WHERE l.cik = ? AND l.type = 'warrant'
                  AND l.status = 'active'
                  AND f.file_number = ?
                  AND l.created_accession != ?
                ORDER BY l.created_at ASC""",
            (cik, parent_file_number, accession),
        ).fetchall()
    if not sib_rows:
        return mlist
    common_sibs: list[str] = []
    prefunded_sibs: list[str] = []
    _COMP_LABEL_TOKENS = ("placement agent", "underwriter",
                          "representative", "rep warrant")
    for r in sib_rows:
        try:
            terms = _json.loads(r["terms_json"] or "{}")
        except (ValueError, TypeError):
            continue
        # NEVER offer a banker-compensation tranche as a reroute
        # target: folding an offering's Common-Warrant create onto a
        # 'Rep Warrants' / 'Placement Agent Warrants' sibling both
        # corrupts the comp row's economics AND hides the merged
        # common warrant behind cards.py's comp-label suppression
        # (XTIA Sept-2025: the 12.5M Common create vanished into the
        # suppressed 'March 2025 Rep Warrants' row).
        lbl = (r["label"] or "").lower()
        if any(t in lbl for t in _COMP_LABEL_TOKENS):
            continue
        strike = terms.get("strike")
        try:
            sval = float(strike) if strike is not None else None
        except (ValueError, TypeError):
            sval = None
        if terms.get("is_pre_funded") is True or (sval is not None and sval < 0.01):
            prefunded_sibs.append(r["instrument_id"])
        else:
            common_sibs.append(r["instrument_id"])

    # When a filing carries MULTIPLE non-prefunded creates they are
    # distinct instruments by construction (offering Common + PA
    # warrants in one 424B5) — rerouting them would collapse both onto
    # the single sibling, the second overwriting the first. Leave all
    # non-prefunded creates standalone in that case.
    n_common_creates = sum(
        1 for m in mlist.mutations
        if isinstance(m, CreateWarrant)
        and not (bool(m.is_pre_funded)
                 or (m.strike is not None and float(m.strike) < 0.01))
    )

    out: list[Mutation] = []
    rerouted = 0
    for m in mlist.mutations:
        if not isinstance(m, CreateWarrant):
            out.append(m)
            continue
        is_pf = bool(m.is_pre_funded) or (
            m.strike is not None and float(m.strike) < 0.01
        )
        if not is_pf and n_common_creates > 1:
            out.append(m)
            continue
        candidates = prefunded_sibs if is_pf else common_sibs
        if len(candidates) != 1:
            out.append(m)
            continue
        sibling_id = candidates[0]
        # Carry the create's terms onto the amend. Use the resolved
        # dates (the @property `terms` ran the calendar arithmetic for
        # term_months / exercise_offset_months / term_anchor); fall
        # back to None when the field wasn't set.
        from datetime import date as _date
        resolved = m.terms
        def _to_date(s):
            if not s: return None
            try: return _date.fromisoformat(s[:10])
            except (TypeError, ValueError): return None
        amend = AmendWarrant(
            instrument_id=sibling_id,
            event_date=m.event_date,
            count=float(m.count) if m.count is not None else None,
            strike=float(m.strike) if m.strike is not None else None,
            exercisable_date=_to_date(resolved.get("exercisable_date")),
            expiration=_to_date(resolved.get("expiration")),
            is_pre_funded=m.is_pre_funded,
            series_letter=m.series_letter,
            known_owners=m.known_owners,
        )
        out.append(amend)
        rerouted += 1
        log.info(
            "  %s — rerouted create_warrant → amend_warrant %s "
            "(file_number %s %s sibling; new strike=%s count=%s)",
            accession, sibling_id, parent_file_number,
            "pre-funded" if is_pf else "common",
            m.strike, m.count,
        )
    if rerouted == 0:
        return mlist
    return MutationList(mutations=out)


# An employee/officer stock-OPTION grant under an equity-incentive / ESOP
# plan is compensation, not a dilutive financing warrant. A real financing
# warrant always names itself "warrant"; an option grant filing talks
# about a plan / option / compensation committee and never says "warrant".
_ESOP_OPTION_RE = re.compile(
    r"\b(?:equity|stock)\s+incentive\s+plan\b|\bemployee\s+stock\b"
    r"|\bcompensation\s+committee\b|\bstock\s+options?\b", re.I)
_WARRANT_RE = re.compile(r"\bwarrant", re.I)
# The issuer PURCHASES a convertible note FROM another entity (it converts
# into THAT entity's shares) → an asset held, not dilution issued. An
# issuer never describes its own issuance as "purchase a ... note from".
_HELD_NOTE_RE = re.compile(
    r"purchas\w+\s+a\b[^.\n]{0,60}\bnote\s+from\b"
    r"|purchas\w+[^.\n]{0,40}\bconvertible\s+note\s+from\b", re.I)
# A convertible with NO conversion mechanism and a >10yr bullet maturity is
# a secured term loan / back-leverage facility the walker mistyped.
_CONVERTIBLE_MAX_PLAUSIBLE_TERM_MONTHS = 120


def _drop_esop_option_warrant(
    mlist: MutationList, *, accession: str, filing_text: str,
) -> MutationList:
    """Drop a create_warrant when the filing is plainly an employee/officer
    stock-OPTION grant (plan / option / compensation-committee language AND
    zero "warrant" mentions). Both conditions are required so a PIPE that
    also grants employee options is never touched (XTIA Oct-2024 W-3070:
    officer option grant under the 2018 Employee Stock Incentive Plan)."""
    if not any(isinstance(m, CreateWarrant) for m in mlist.mutations):
        return mlist
    body = filing_text or ""
    if _WARRANT_RE.search(body) or not _ESOP_OPTION_RE.search(body):
        return mlist
    out, dropped = [], 0
    for m in mlist.mutations:
        if isinstance(m, CreateWarrant):
            dropped += 1
            log.info(
                "  %s — dropped create_warrant: ESOP/officer option grant "
                "(no \"warrant\" in body; plan/option/compensation-committee "
                "language present)", accession,
            )
            continue
        out.append(m)
    return MutationList(mutations=out) if dropped else mlist


def _drop_held_asset_convertible(
    mlist: MutationList, *, accession: str, filing_text: str,
) -> MutationList:
    """Drop a create_convertible when the issuer is the HOLDER, not the
    issuer — the "purchase a ... note from <Entity>" construction, where
    the note converts into the other entity's shares and does not dilute
    the issuer (XTIA/Inpixon Oct-2023: 'Inpixon will purchase a convertible
    note from Damon')."""
    if not any(isinstance(m, CreateConvertible) for m in mlist.mutations):
        return mlist
    if not _HELD_NOTE_RE.search(filing_text or ""):
        return mlist
    out, dropped = [], 0
    for m in mlist.mutations:
        if isinstance(m, CreateConvertible):
            dropped += 1
            log.info(
                "  %s — dropped create_convertible: issuer is the HOLDER "
                "(\"purchase a ... note from\"), note is an asset not "
                "dilution", accession,
            )
            continue
        out.append(m)
    return MutationList(mutations=out) if dropped else mlist


def _drop_noncvt_term_loan(
    mlist: MutationList, *, accession: str,
) -> MutationList:
    """Drop a create_convertible with NO conversion mechanism (no conv_price,
    no conversion-start date/offset) AND a >10yr bullet maturity — the shape
    of a secured term loan / back-leverage facility the walker mistyped as a
    convertible (FCEL Aug-2023 Connecticut Green Bank $8M subordinated loan:
    conv_price=None, maturity ~2043). A real PIPE/toxic convertible always
    states a conversion price or conversion-start and matures in ≤5yr, so
    this cannot drop a genuine convertible."""
    if not any(isinstance(m, CreateConvertible) for m in mlist.mutations):
        return mlist
    out, dropped = [], 0
    for m in mlist.mutations:
        if isinstance(m, CreateConvertible):
            has_conv = (m.conv_price is not None
                        or m.convertible_date is not None
                        or m.convertible_offset_months is not None)
            term = m.maturity_months
            if term is None and m.maturity is not None:
                term = ((m.maturity.year - m.event_date.year) * 12
                        + (m.maturity.month - m.event_date.month))
            if (not has_conv and term is not None
                    and term > _CONVERTIBLE_MAX_PLAUSIBLE_TERM_MONTHS):
                dropped += 1
                log.info(
                    "  %s — dropped create_convertible: no conversion "
                    "mechanism + %d-month bullet maturity → straight term "
                    "loan, not a convertible", accession, term,
                )
                continue
        out.append(m)
    return MutationList(mutations=out) if dropped else mlist


def _walk_split(*, cik: int, ticker: str, split, summary: WalkSummary) -> None:
    """Apply one externally-sourced split via the same `apply_mutations`
    path the walker uses for LLM-emitted splits. Idempotent because
    `_split_already_applied` in store.py rejects same-direction splits
    inside a ±30-day window — re-running the walker doesn't double-apply.
    """
    accession = f"split:{split.effective_date}:{split.source}"
    mutation = ApplySplit(
        post=split.post,
        pre=split.pre,
        direction=split.direction,
        units=split.units,
        effective_date=_date.fromisoformat(split.effective_date),
    )
    log.info("  [%s] split %d-for-%d %s (%s) units=%s",
             split.effective_date, split.post, split.pre,
             split.direction, split.source, split.units)
    result = apply_mutations(
        cik=cik, ticker=ticker, accession=accession,
        form="EXTERNAL_SPLIT", filing_date=split.effective_date,
        mutations=[mutation],
    )
    summary.mutations_applied += result.accepted
    summary.mutations_rejected += result.rejected


async def _walk_one(
    *, cik: int, ticker: str, filing: dict, client,
    unit_ctx: dict | None, summary: WalkSummary,
) -> bool:
    """Process one filing. Returns True iff meaningful work happened —
    walker mutations applied OR (for periodics) the anchor pass ran."""
    accession = filing["accession_number"]
    form = filing["form"]
    filing_date = filing["filing_date"]

    # Pre-screen: 424B / S-1 / S-3 whose SEC file_number doesn't match
    # any of the company's primary registrations is a resale
    # prospectus for selling holders — no new issuance, no walker
    # mutations to extract. Short-circuit before the LLM round-trip.
    # See dilution/ledger/registration_family.py for the full
    # semantics + "unknown" fall-through.
    from .registration_family import (
        classify_424b_attribution, primary_shelf_for_filing,
        PRESCREEN_FORMS, RESALE_PROPAGATION_FORMS,
    )
    form_upper = (form or "").upper()
    attribution = "unknown"
    if any(form_upper == p or form_upper.startswith(p + "/")
           for p in PRESCREEN_FORMS):
        attribution = classify_424b_attribution(cik, accession)
        if attribution == "resale":
            log.info("  [%s] %s %s — skipped (resale, file_number "
                     "outside primary set)",
                     filing_date, form, accession)
            summary.skipped_resale += 1
            return False
    elif form_upper in RESALE_PROPAGATION_FORMS:
        # Resale-family propagation: a registration AMENDMENT (/A,
        # POS AM) inherits its PARENT registration's resale verdict. We
        # anchor on the deterministic fee-table (457(c)/(g) = resale) of
        # an EARLIER registration filing under the SAME file_number —
        # NOT on mere absence of a primary shelf, which would
        # false-positive on a primary shelf whose parent S-3 predates our
        # ingest window (its in-window /A would look shelf-less). So a
        # resale registration's amendment is skipped before it can mint a
        # phantom s1_offering/shelf (SCNI Yorkville SEPA F-1/A under
        # 333-285547, whose parent F-1 is fee-table resale), while a
        # primary shelf's amendment flows through to the follow-on amend
        # hint below, unchanged.
        if _registration_family_is_resale(cik, accession):
            log.info("  [%s] %s %s — skipped (resale-family: parent "
                     "registration under same file_number is fee-table "
                     "resale)", filing_date, form, accession)
            summary.skipped_resale += 1
            return False

    # 8-K item-code pre-screen. An 8-K whose EDGAR item codes carry no
    # dilution-relevant item AND attaches no substantive exhibit cannot
    # hold a cap-table event — skip the LLM round-trip (mirrors the
    # resale pre-screen above). The exhibit co-gate is REQUIRED: microcaps
    # under-tag, so items alone would drop real financings tagged only
    # "9.01" or bundled into an earnings 8-K's EX-99 (verified zero-miss
    # only WITH the exhibit gate). See item_classification.py.
    must_record_8k = False
    must_record_reason = ""
    if form_upper.startswith("8-K"):
        from .item_classification import classify_8k
        v8k = classify_8k(cik, accession, form)
        if v8k.skip:
            log.info("  [%s] %s %s — skipped (%s)",
                     filing_date, form, accession, v8k.reason)
            summary.skipped_no_dilution += 1
            return False
        must_record_8k = v8k.must_record
        must_record_reason = v8k.reason

    # Build the attribution hint when the filing is a primary
    # take-down. Empty string for resale/unknown/non-prescreen forms.
    attribution_block = ""
    if attribution == "primary":
        parent = primary_shelf_for_filing(cik, accession)
        if parent:
            attribution_block = _build_attribution_block(parent, form)
    else:
        # Registration-statement forms (S-3, S-1, F-3, F-1 and their
        # amendments) bypass the resale pre-screen — they're the
        # documents that CREATE primary shelves and can never be
        # resale of themselves. But when one is filed under the SAME
        # file_number as an already-tracked shelf / s1_offering, this
        # filing is a follow-on (S-3/A, S-1/A, S-3MEF, etc.) — the
        # walker should AMEND the existing instrument, not create a
        # duplicate. Without this hint the LLM reliably emits a fresh
        # `create_instrument` per amendment, producing per-revision
        # duplicates of the same logical shelf / s1_offering.
        is_registration = any(
            form_upper.startswith(p) or form_upper == p
            for p in ("S-3", "S-1", "F-3", "F-1", "F-10")
        )
        if is_registration:
            parent = primary_shelf_for_filing(cik, accession)
            if parent:
                attribution_block = _build_registration_followon_block(
                    parent, form,
                )

    # Fee-table prescreen. Deterministic primary-vs-resale classification
    # from the EX-FILING FEES exhibit (Rule 457(o)/(r) primary vs
    # 457(c)/(g) resale). Complements the file_number prescreen:
    #   - For 424B-family forms whose file_number prescreen returned
    #     "unknown" (no primary-set hit AND no positive resale evidence
    #     yet — see registration_family.py:117), a resale fee table is
    #     the deterministic tiebreak.
    #   - For S-3/F-3/S-1/F-1 (and amendments), the file_number prescreen
    #     is deliberately bypassed because these forms CREATE primary
    #     registrations — their own file_number is not yet in the
    #     primary_set when the walker first sees them. A resale-only
    #     fee table means the registration is for selling-stockholder
    #     resale of already-issued securities — the walker has no new
    #     instrument to create.
    # We don't override a positive file_number prescreen verdict
    # (attribution == "primary" is SEC-canonical evidence of a primary
    # takedown). For "primary" or "mixed" verdicts, the hint is surfaced
    # to the LLM via the user prompt's fee-table block.
    from ._exhibit_provisions import (
        FEE_TABLE_FORMS, classify_fee_table, format_fee_table_for_prompt,
    )
    fee_verdict = ("unknown" if form_upper not in FEE_TABLE_FORMS
                   else classify_fee_table(accession))
    if fee_verdict == "resale" and attribution != "primary":
        log.info("  [%s] %s %s — skipped (fee-table=resale; "
                 "457(c)/(g) only)",
                 filing_date, form, accession)
        summary.skipped_resale += 1
        return False
    fee_table_block = format_fee_table_for_prompt(fee_verdict)

    text = _load_filing_text(accession)
    if not text:
        log.warning("  skip %s — no raw text", accession)
        return False

    filing_date_obj = _date.fromisoformat(filing_date)
    open_rows = get_open_instruments(cik, today=filing_date_obj)
    ledger_view = render_ledger_view(
        open_rows,
        drawdowns_by_instrument=get_drawdowns_by_instrument(cik),
        today=filing_date_obj,
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
        attribution_block=attribution_block,
        fee_table_block=fee_table_block,
        must_record=must_record_8k,
        must_record_reason=must_record_reason,
    )
    n_mutations = len(mlist.mutations)
    log.info("  [%s] %s %s — %d mutations",
             filing_date, form, accession, n_mutations)

    form_base = (form or "").upper().split("/")[0]
    is_periodic = form_base in _PERIODIC_FORMS
    # 6-K is heterogeneous — only the interim-financials variant carries
    # an outstanding-instruments table worth anchor-reconciling. Gate on
    # financial-statement markers so press-release furnishings don't fire
    # the overhang specialists. (`text` is already loaded above and is
    # non-empty here — the no-text path returned False earlier.)
    if (form_base == "6-K" and is_periodic
            and not _sixk_carries_financials(text)):
        log.info("  [%s] 6-K %s — no interim-financials markers; "
                 "anchor reconciliation skipped", filing_date, accession)
        is_periodic = False

    # Apply walker mutations when present. A zero-mutation periodic is
    # normal ("the filing only references already-known instruments")
    # and must NOT short-circuit the anchor reconciliation below — the
    # whole point of anchor reconciliation is to verify the running
    # ledger against the filing's authoritative outstanding-instruments
    # table, and that check is most valuable precisely when the walker
    # LLM emitted nothing (no walker activity means no walker-introduced
    # drift to attribute mismatches to). An earlier version of this
    # function early-returned on empty mutations and silently skipped
    # the anchor for ~30%+ of periodic filings; that's now fixed.
    if mlist.mutations:
        snapshot = {row["instrument_id"]: row for row in open_rows}
        # Deterministic guard: a create_s1_offering on a filing whose
        # file_number already maps to an existing s1_offering is a
        # duplicate of that offering — rewrite to amend_s1_offering.
        mlist = _reroute_s1_create_to_amend(
            mlist, cik=cik, accession=accession, form=form,
            snapshot=snapshot,
        )
        # Deterministic guard: a create_warrant on a filing whose
        # file_number already maps to a sibling warrant on the same
        # registration chain is a duplicate of that warrant (the
        # anticipated → priced repricing the strike-dedup guard can't
        # bridge across 20-60% gaps) — rewrite to amend_warrant.
        mlist = _reroute_warrant_create_to_amend(
            mlist, cik=cik, accession=accession, form=form,
            snapshot=snapshot,
        )
        # Deterministic mis-classification guards (drop, not reroute —
        # there is no valid target instrument): an ESOP/officer option
        # grant minted as a warrant, an issuer-HELD note minted as an
        # issued convertible, and a non-convertible long-bullet term loan
        # minted as a convertible. Each is gated on signals absent from
        # every correctly-typed instrument's source filing.
        mlist = _drop_esop_option_warrant(
            mlist, accession=accession, filing_text=text,
        )
        mlist = _drop_held_asset_convertible(
            mlist, accession=accession, filing_text=text,
        )
        mlist = _drop_noncvt_term_loan(mlist, accession=accession)
        report = validate_mutations(
            mlist.mutations, snapshot, filing_form=form,
        )
        for vr in report.rejected:
            log.info("    reject %s — %s: %s",
                     fmt_mutation(vr.mutation),
                     vr.error_kind or "?",
                     vr.message or "")
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

    # Anchor reconciliation runs unconditionally for periodic filings.
    if is_periodic:
        # Deterministic full-preferred-conversion close: when this periodic
        # filing affirms ALL preferred automatically converted to common
        # with none remaining outstanding, close the lingering active
        # preferreds (the overhang LLM re-matches the named series in the
        # conversion narrative without flagging is_terminated, so the anchor
        # never closes them — KSCP Series A/B/M/S, converted 2024-05-15).
        # Runs BEFORE the anchor so reconciliation sees them closed; the
        # store scopes the close to preferreds issued on/before the
        # conversion date and is idempotent (active rows only).
        conv_date = _full_preferred_conversion_date(text)
        if conv_date is not None:
            closed_pfd = close_converted_preferred(
                cik, conversion_date=conv_date, accession=accession,
                form=form, filing_date=filing_date,
            )
            if closed_pfd:
                log.info(
                    "  [%s] %s affirms full preferred conversion (%s) — "
                    "closed %d phantom-active preferred(s): %s",
                    filing_date, accession, conv_date,
                    len(closed_pfd), ", ".join(closed_pfd),
                )
        # Deterministic retired-debt close: a convertible whose balance has
        # reached dust with its retired flow accounting for the face is
        # finished, but the walker only emits close_instrument when the
        # filing says so in prose — otherwise the row lingers `active` at
        # zero and only the card layer's dust filter hides it. Runs BEFORE
        # the anchor for the same reason as the preferred sweep above, so
        # reconciliation sees the closed state. Rows whose flow does NOT
        # account for the face are left active by design (see
        # store.close_retired_debt).
        closed_debt = close_retired_debt(
            cik, accession=accession, form=form, filing_date=filing_date,
        )
        if closed_debt:
            log.info(
                "  [%s] %s — closed %d retired convertible(s): %s",
                filing_date, accession, len(closed_debt),
                ", ".join(closed_debt),
            )
        await _anchor_one(
            cik=cik, ticker=ticker, filing=filing, client=client,
            unit_ctx=unit_ctx, summary=summary,
        )

    # "Walked" = any meaningful work happened (mutations applied OR an
    # anchor pass ran). Non-periodic with zero mutations returns False
    # so the caller can count it as `skipped` for telemetry.
    return bool(mlist.mutations) or is_periodic


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

    # Deterministic stated-balance scan over the filing's own prose —
    # the truncation-proof backstop for the LLM overhang read (vetoes
    # contradicted closes, pins principal_remaining; see anchor.py).
    # Runs even when the overhang extraction came back EMPTY (token-cap
    # truncation, salvage=0): the prose scans are exactly what must
    # survive that failure (round-4: FCEL's 2026 10-Q truncated and the
    # early return silently skipped the $157.1M stated-remaining pin
    # AND corroborate).
    try:
        stated_balances = extract_stated_note_balances(cik, accession)
    except Exception as exc:
        log.warning("  stated-balance scan failed for %s: %s",
                    accession, exc)
        stated_balances = []

    # Reconcile stays gated on a non-empty overhang read: with an empty
    # table every open row would diff extra_in_ledger and tier-1 could
    # close the whole book off a truncated extraction.
    if overhang_rows:
        open_rows = get_open_instruments(cik)
        result = reconcile_against_periodic(
            cik=cik, accession=accession,
            filing_date=filing["filing_date"], as_of_date=as_of,
            filing_overhang=overhang_rows, ledger_open=open_rows,
            stated_note_balances=stated_balances,
        )
        if result.diffs:
            record_anchor_diffs(cik, accession, as_of,
                                [d.to_dict() for d in result.diffs])
            summary.anchor_diffs += len(result.diffs)
        if result.correction_mutations:
            for m in result.correction_mutations:
                log.info("    anchor proposal %s", fmt_mutation(m))
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

    # Anchor-corroborated close-rejection: reopen any instrument THIS filing
    # closed that the filing's own overhang still reports as outstanding (a
    # hallucinated close contradicted by the issuer's own numbers). Complements
    # the mass-closure sweep guard, which only catches >=5-type wipes.
    # Defensive: a failure here must never break the walk.
    try:
        reopen_ids = corroborate_closes(
            cik=cik, accession=accession, filing_overhang=overhang_rows,
            stated_note_balances=stated_balances, as_of_date=as_of,
        )
        n = reopen_instruments(
            cik, reopen_ids, accession, filing["filing_date"],
        ) if reopen_ids else 0
        if n:
            summary.mutations_applied += n
            log.info(
                "  anchor reopened %d close(s) contradicted by the filing's "
                "own overhang: %s", n, reopen_ids,
            )
            # A reopened 'redeemed' convertible had its balance zeroed
            # at close — restore the filing-stated balance so the row
            # doesn't linger at $0 (which the next anchor would just
            # re-close).
            if stated_balances:
                from .anchor import _map_stated_balances
                reopened = [r for r in get_open_instruments(cik)
                            if r.get("instrument_id") in set(reopen_ids)]
                pins = _map_stated_balances(stated_balances, reopened)
                pin_muts = [
                    amend_from_dict(
                        type_="convertible", instrument_id=iid,
                        outstanding_updates={"principal_remaining": bal},
                        event_date=as_of,
                    )
                    for iid, bal in pins.items() if bal > 0
                ]
                if pin_muts:
                    pin_result = apply_mutations(
                        cik=cik, ticker=ticker, accession=accession,
                        form=filing["form"],
                        filing_date=filing["filing_date"],
                        mutations=pin_muts,
                    )
                    summary.mutations_applied += pin_result.accepted
                    log.info(
                        "  anchor pinned %d reopened note balance(s) "
                        "from stated text", pin_result.accepted,
                    )
    except Exception as exc:
        log.warning("  corroborate_closes failed for %s: %s", accession, exc)

    # Stated ATM remaining-capacity pin: "approximately $X million
    # remaining to be sold / remained available for sale under the Sales
    # Agreement" is the issuer's own window-scoped cumulative — interim
    # ATM sales often never get a discrete drawdown disclosure, so the
    # running drawn_usd undercounts by every quiet quarter (round-4
    # kscp-jul2025: $12.55M recorded vs ~$31.7M sold; fcel-dec2025). The
    # amend lands on drawn_usd_anchor/asof via the store, and the cards
    # layer adds only post-asof discrete draws on top. Defensive: never
    # break the walk.
    try:
        from .anchor import (
            extract_stated_atm_remaining, map_stated_atm_remaining,
        )
        atm_stated = extract_stated_atm_remaining(cik, accession)
        if atm_stated:
            atm_pins = map_stated_atm_remaining(
                atm_stated, get_open_instruments(cik))
            atm_pin_muts = [
                amend_from_dict(
                    type_="atm", instrument_id=iid,
                    outstanding_updates={"drawn_usd": drawn},
                    event_date=(asof or as_of),
                )
                for iid, drawn, asof in atm_pins
            ]
            if atm_pin_muts:
                atm_pin_result = apply_mutations(
                    cik=cik, ticker=ticker, accession=accession,
                    form=filing["form"],
                    filing_date=filing["filing_date"],
                    mutations=atm_pin_muts,
                )
                summary.mutations_applied += atm_pin_result.accepted
                log.info(
                    "  anchor pinned %d ATM drawn checkpoint(s) from "
                    "stated remaining-capacity text",
                    atm_pin_result.accepted,
                )
    except Exception as exc:
        log.warning("  stated ATM-remaining pin failed for %s: %s",
                    accession, exc)


# ─── Public entry points ─────────────────────────────────────────────
def walk_ticker(
    cik: int, ticker: str, *,
    since_date: str,
    force: bool = False,
    concurrency: int = 1,
) -> WalkSummary:
    """Synchronous entry point. Walks all filings for a CIK in one go.

    `force=True` drops the existing ledger first. Otherwise the walker
    resumes by skipping any filing already in the per-accession
    `dilution_walked` set — robust to filings back-filled out of date
    order (unlike the old positional last_processed_accession marker).
    """
    from dilution.openai_client import require_api_key
    require_api_key()
    return asyncio.run(_walk_async(
        cik=cik, ticker=ticker, since_date=since_date,
        force=force, concurrency=max(1, int(concurrency)),
    ))


__all__ = [
    "WalkSummary",
    "walk_ticker",
]
