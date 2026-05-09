"""Fetch and text-extract documents for filings we'll LLM-extract.

For each filing we store the primary doc PLUS dilution-relevant exhibits
as separate rows in `dilution_raw`. The card layer needs investor names,
placement agents, and price-protection clauses which often live in
exhibits (SPA, warrant agreement, certificate of designation), not in
the primary prospectus.

What gets pulled per filing:
  - Primary doc (always for non-6-K extractable forms; LLM-classified
                 for 6-K)
  - EX-1.x   — Underwriting agreements (gross spread, over-allotment,
               lock-up scope) — always pulled
  - EX-2.x   — Acquisition / merger agreements (de-SPAC share issuance,
               PIPE-with-merger mechanics) — always pulled
  - EX-3.x   — Certificate of Designation, articles (preferred terms)
               — always pulled
  - EX-4.x   — Warrant agreements, indentures, debenture forms —
               always pulled
  - EX-10.x  — Material contracts; LLM-CLASSIFIED to drop employment
               agreements / leases / supply agreements while keeping
               SPAs, RRAs, placement-agency agreements, equity-line
               agreements, equity-distribution (ATM) agreements
  - EX-99.x  — Press releases / supplemental info; LLM-CLASSIFIED

Skipped: GRAPHIC, XBRL XML (EX-101.*, EX-104), EX-21 (subs), EX-23
(consents), EX-24 (POAs).
"""

import asyncio
import logging

from edgar import get_by_accession_number, set_identity

import config
from db import get_conn, now_iso

from .exhibit_classifier import (
    classify_by_description,
    classify_exhibit,
    classify_exhibit_async,
)
from .llm_provider import make_async_client
from .html_to_text import html_to_text
from .periodic_sections import is_periodic_with_sections, select_text

log = logging.getLogger(__name__)

EXTRACTABLE_PREFIXES = (
    "8-K",
    "6-K",  # foreign private issuer analog of 8-K
    "424B",
    "S-1", "S-3", "S-3ASR", "S-4", "F-1", "F-3",
    "POS AM",
    # M&A communications (Rule 425) — usually mirrors the accompanying 8-K
    "425",
    "DEF 14A", "DEFA14A", "DEFM14A", "PRE 14A",
    "FWP",
    "10-K", "10-Q", "20-F", "40-F",
    # Regulation A+ family
    "1-A", "1-U", "1-K", "1-SA",
)

MAX_CHARS = 2_000_000  # per-row cap


def _extractable(form: str) -> bool:
    return bool(form) and any(form.startswith(p) for p in EXTRACTABLE_PREFIXES)


def _is_narrative_exhibit(doc_type: str | None, document: str | None,
                          form: str | None = None) -> str:
    """Classify an attachment by exhibit family. Return one of:
        'always'    — always pull (EX-1/2/3/4, primary)
        'classify'  — pull iff LLM classifier says relevant
        'skip'      — never pull

    EX-99.x on 8-K/6-K is always kept: by SEC convention the press
    release attached as EX-99.1 contains the substance of the
    announcement (private placement, offering closing, share
    consolidation). The classifier was over-rejecting these on
    Moonshot — fail open is the right default per the coverage rule
    in CLAUDE.md.
    """
    if not doc_type:
        return "skip"
    dt = doc_type.upper()
    doc = (document or "").lower()
    f = (form or "").strip()
    # Hard skips: graphics, XBRL, consents, subsidiary lists, POAs.
    if dt in ("GRAPHIC",) or doc.endswith((".jpg", ".jpeg", ".png", ".gif")):
        return "skip"
    if dt.startswith("EX-21") or dt.startswith("EX-23") or dt.startswith("EX-24"):
        return "skip"
    if dt.startswith("EX-101") or dt.startswith("EX-104"):
        return "skip"
    if (dt.startswith("EX-1.") or dt.startswith("EX-2.")
            or dt.startswith("EX-3.") or dt.startswith("EX-4.")):
        return "always"
    if dt.startswith("EX-99.") and (f.startswith("8-K") or f.startswith("6-K")):
        return "always"
    if dt.startswith("EX-10.") or dt.startswith("EX-99."):
        return "classify"
    return "skip"


def _attachment_markdown(attachment) -> str | None:
    """Return text/markdown for an attachment, preferring our inline-
    preserving converter on raw HTML so iXBRL-wrapped numbers stay
    adjacent to currency markers."""
    try:
        content = attachment.content
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        if content and "<" in content[:200].lower():
            md = html_to_text(content)
            if md and md.strip():
                return md
    except Exception as e:
        log.debug("  content read failed: %s", e)
    try:
        md = attachment.markdown()
        if md and md.strip():
            return md
    except Exception as e:
        log.debug("  markdown failed: %s", e)
    return None


def _store(accession: str, doc_name: str, doc_type: str, md: str) -> None:
    if len(md) > MAX_CHARS:
        md = md[:MAX_CHARS]
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO dilution_raw
                 (accession_number, doc_name, doc_type, content_md, downloaded_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(accession_number, doc_name) DO UPDATE SET
                 content_md = excluded.content_md,
                 doc_type   = excluded.doc_type,
                 downloaded_at = excluded.downloaded_at""",
            (accession, doc_name, doc_type, md, now_iso()),
        )


async def _fetch_filing_text_async(accession: str, client) -> int:
    """Async sibling of fetch_filing_text.

    Blocking SEC HTTP calls (edgartools is sync) are wrapped via
    asyncio.to_thread so the event loop stays responsive. Edgartools
    enforces SEC's 10 req/s rate limit internally on a per-process
    basis, so concurrent filings get throttled correctly without us
    having to coordinate.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT form, primary_doc FROM dilution_filings WHERE accession_number = ?",
            (accession,),
        ).fetchone()
    if not row:
        return 0

    try:
        filing = await asyncio.to_thread(get_by_accession_number, accession)
    except Exception as e:
        log.warning("  lookup failed for %s: %s", accession, e)
        return 0
    if filing is None:
        return 0

    written = 0

    form = row["form"] or ""
    is_6k = form.startswith("6-K")

    # 1) Primary document.
    md: str | None = None

    # For 10-K / 10-Q / 20-F (and /A amendments) drop obvious-noise
    # sections via edgartools' typed-object section parser before we
    # store the markdown. Risk Factors alone is typically 30-50% of a
    # 10-K. Safe-fail: if section selection returns None for any
    # reason (parsing failure, low-confidence boundaries, suspiciously
    # small result) we fall through to the full-document path.
    if is_periodic_with_sections(form):
        def _select():
            try:
                return select_text(filing, form)
            except Exception as e:  # noqa: BLE001 - boundary log
                return None, {"reason": f"select_text raised: {e}"}
        selected, stats = await asyncio.to_thread(_select)
        if selected:
            md = selected
            log.info("  %s %s sections kept=%s dropped=%d "
                     "kept_chars=%d dropped_chars=%d",
                     form, accession,
                     stats.get("kept", []),
                     len(stats.get("dropped", [])),
                     stats.get("kept_chars", 0),
                     stats.get("dropped_chars", 0))
        else:
            log.info("  %s %s section-select fallback: %s",
                     form, accession, stats.get("reason"))

    if md is None:
        def _get_html():
            try:
                return filing.html() or ""
            except Exception as e:
                log.debug("  html fetch failed for %s: %s", accession, e)
                return ""

        html = await asyncio.to_thread(_get_html)
        md = html_to_text(html) if html else None
        if not md or not md.strip():
            try:
                md = await asyncio.to_thread(filing.markdown)
            except Exception as e:
                log.warning("  markdown failed for %s: %s", accession, e)
                md = None

    primary_classified_relevant = False
    if md and md.strip() and is_6k:
        primary_classified_relevant = await classify_exhibit_async(
            client,
            doc_type="PRIMARY",
            doc_name=row["primary_doc"] or "primary",
            form=form,
            content=md,
            accession=accession,
        )

    if md and md.strip():
        _store(accession, row["primary_doc"] or "primary", row["form"], md)
        written += 1

    # 2) Narrative exhibits.
    try:
        attachments = await asyncio.to_thread(lambda: list(filing.attachments))
    except Exception as e:
        log.debug("  attachments enumeration failed for %s: %s", accession, e)
        attachments = []

    primary_doc_name = (row["primary_doc"] or "").lower()
    for a in attachments:
        doc = getattr(a, "document", None)
        dt = getattr(a, "document_type", None)
        if not doc:
            continue
        if doc.lower() == primary_doc_name:
            continue
        verdict = _is_narrative_exhibit(dt, doc, form)
        if verdict == "skip":
            continue

        # Deterministic description router runs BEFORE we fetch
        # content. Hard-DROP descriptions (employment agreements,
        # leases, pro-forma financials, etc.) skip the round-trip
        # entirely; hard-KEEP descriptions (SPA, warrant agreement,
        # ATM sales agreement, certificate of designation) skip the
        # LLM classifier. UNKNOWN descriptions fall through to the
        # LLM classifier as before.
        desc = getattr(a, "description", None)
        desc_route = "unknown"
        if verdict == "classify":
            desc_route = classify_by_description(description=desc)
            if desc_route == "drop":
                log.info("  desc-classify drop %s/%s [%s]: %r",
                         dt, doc, accession, desc)
                continue

        ex_md = await asyncio.to_thread(_attachment_markdown, a)
        if not ex_md:
            continue
        if verdict == "classify" and desc_route == "unknown":
            keep = await classify_exhibit_async(
                client,
                doc_type=dt,
                doc_name=doc,
                form=form,
                content=ex_md,
                accession=accession,
            )
            if not keep:
                continue
        elif verdict == "classify" and desc_route == "keep":
            log.info("  desc-classify keep %s/%s [%s]: %r",
                     dt, doc, accession, desc)
        _store(accession, doc, dt or "EX", ex_md)
        written += 1

    if is_6k and not primary_classified_relevant and written <= 1:
        with get_conn() as conn:
            conn.execute(
                "DELETE FROM dilution_raw WHERE accession_number = ?",
                (accession,),
            )
        log.debug("  %s — 6-K rolled back (no dilution-relevant content)",
                  accession)
        return 0

    return written


def fetch_filing_text(accession: str, client) -> int:
    """Synchronous wrapper preserved for any external callers. New code
    should use _fetch_filing_text_async via fetch_extractable_for_cik."""
    return asyncio.run(_fetch_filing_text_async(accession, client))


async def _fetch_worker(r, client, sem, counters, total):
    accession = r["accession_number"]
    async with sem:
        try:
            n = await _fetch_filing_text_async(accession, client)
        except Exception as e:
            counters["done"] += 1
            counters["errors"] += 1
            log.warning("  %s error: %s", accession, e)
            return
    counters["done"] += 1
    if n:
        counters["fetched"] += 1
        counters["docs"] += n
        log.debug("  %s %s — %d docs", r["form"], accession, n)
    if counters["done"] % 25 == 0 or counters["done"] == total:
        log.info("  fetched %d/%d filings, %d docs total (err=%d)",
                 counters["fetched"], total,
                 counters["docs"], counters["errors"])


async def _run_fetch(cik: int, since_date: str | None,
                    limit: int | None, concurrency: int) -> dict:
    set_identity(config.EDGAR_IDENTITY)

    q = """SELECT f.accession_number, f.form, f.filing_date
           FROM dilution_filings f
           LEFT JOIN dilution_raw r ON r.accession_number = f.accession_number
           WHERE f.cik = ? AND r.accession_number IS NULL"""
    args = [cik]
    if since_date:
        q += " AND f.filing_date >= ?"
        args.append(since_date)
    q += " ORDER BY f.filing_date DESC"

    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute(q, args).fetchall()]

    rows = [r for r in rows if _extractable(r["form"])]
    if limit:
        rows = rows[:limit]

    total = len(rows)
    counters = {"done": 0, "fetched": 0, "docs": 0, "errors": 0}

    if total == 0:
        log.info("  fetch done — 0/0 filings, 0 docs, 0 errors")
        return {"fetched": 0, "docs": 0, "total": 0, "errors": 0}

    log.info("  fetch concurrency=%d over %d filings", concurrency, total)
    client = make_async_client()
    try:
        sem = asyncio.Semaphore(concurrency)
        await asyncio.gather(*[
            _fetch_worker(r, client, sem, counters, total) for r in rows
        ])
    finally:
        await client.close()

    log.info("  fetch done — %d/%d filings, %d docs, %d errors",
             counters["fetched"], total, counters["docs"], counters["errors"])
    return {"fetched": counters["fetched"], "docs": counters["docs"],
            "total": total, "errors": counters["errors"]}


def fetch_extractable_for_cik(cik: int, since_date: str | None = None,
                              limit: int | None = None,
                              concurrency: int | None = None,
                              client=None) -> dict:
    """Fetch dilution-relevant filings + exhibits for `cik`.

    Filing-level concurrency bounded by `concurrency` (default
    config.LLM_CONCURRENCY). The LLM exhibit classifier is mandatory —
    raises if the active provider's API key is unset. The legacy
    `client=` kwarg is accepted but ignored (we own the async client
    lifecycle now).
    """
    from .llm_provider import require_api_key
    require_api_key()
    if concurrency is None:
        concurrency = config.LLM_CONCURRENCY
    concurrency = max(1, int(concurrency))
    return asyncio.run(_run_fetch(cik, since_date, limit, concurrency))
