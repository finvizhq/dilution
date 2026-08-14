"""Detect FPI status and ADS-to-ordinary ratio for a company.

For Foreign Private Issuers (companies that file 20-F or 40-F) listed
in the US via American Depositary Shares (ADS), the ADS — not the
underlying ordinary share — is the listed/quoted/reported instrument.
DilutionTracker shows everything in ADS units, and so do the prompts
in our extractors once this module has populated the metadata.

A non-FPI gets is_fpi=0, ads_ratio=NULL — extractors fall back to
common-share semantics.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re

from edgar import get_by_accession_number, set_identity

import config
from db import get_conn, now_iso

from .ledger._llm_utils import normalize_filing_text
from .openai_client import (
    acomplete, make_async_client, output_text, system, user,
)

log = logging.getLogger(__name__)

# Forms that mark an issuer as a Foreign Private Issuer.
FPI_ANNUAL_FORMS = ("20-F", "20-F/A", "40-F", "40-F/A")

ADS_RATIO_PROMPT = """\
You are reading the most recent annual report (Form 20-F / 40-F) of a
Foreign Private Issuer that lists in the US via American Depositary
Shares (ADS). Find the ADS-to-ordinary-share ratio currently in
effect.

Look for language like:
- "Each ADS represents [N] [ordinary | common | Class A | Class B]
  shares"
- "One ADS represents [fractional number] of an ordinary share"
- "the ratio of ADSs to [ordinary | common] shares is [N]:1"
- A 6-K-style change-in-ratio announcement that supersedes the original
  ratio (e.g. "the ADS-to-ordinary ratio has been changed from 1:100
  to 1:400") — use the LATEST stated ratio.

Return strict JSON:
{{
  "ads_ratio": <number — ordinary shares per 1 ADS>,
  "underlying_unit": "ordinary" | "common" | "Class A" | ...,
  "rationale": "<≤2 sentences quoting the verbatim filing language>"
}}

If the filing does not state a ratio, return:
{{ "ads_ratio": null, "underlying_unit": null, "rationale": "not stated" }}

Filing form: {form}
Filing date: {filing_date}

Filing text (truncated to first ~80K chars):
{text}
"""


def _is_fpi_from_filings(cik: int) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            """SELECT 1 FROM dilution_filings
               WHERE cik = ? AND (
                    form LIKE '20-F%' OR form LIKE '40-F%'
               ) LIMIT 1""",
            (cik,),
        ).fetchone()
    return row is not None


def _latest_annual_filing(cik: int) -> dict | None:
    """Most recent 20-F/40-F filing — the one with current ADS ratio."""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT accession_number, form, filing_date, primary_doc
               FROM dilution_filings
               WHERE cik = ? AND form IN ({})
               ORDER BY filing_date DESC LIMIT 1""".format(
                ",".join(["?"] * len(FPI_ANNUAL_FORMS))
            ),
            (cik, *FPI_ANNUAL_FORMS),
        ).fetchone()
    return dict(row) if row else None


def _load_text_for(accession: str, max_chars: int = 80_000) -> str | None:
    """Cached primary-doc text first; SEC fetch if missing. Trim to first
    ~80K chars — the ADS-ratio statement is on the cover page or in
    Item 12 / Description of Securities, both early in the doc."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT content_md, doc_type, LENGTH(content_md) AS len
               FROM dilution_raw
               WHERE accession_number = ?
               ORDER BY len DESC""",
            (accession,),
        ).fetchall()
    md = None
    for r in rows or []:
        dt = (r["doc_type"] or "").upper()
        if not dt.startswith("EX-"):
            md = r["content_md"]
            break
    if not md and rows:
        md = rows[0]["content_md"]

    if not md:
        # Fresh fetch — same pattern as overhang._fetch_markdown
        set_identity(config.EDGAR_IDENTITY)
        try:
            f = get_by_accession_number(accession)
        except Exception as e:
            log.warning("unit_detection: lookup failed for %s: %s", accession, e)
            return None
        if not f:
            return None
        # The $-adjacency bug in edgartools is patched at import time
        # via dilution/_edgar_patches.py (loaded by fetch_raw on first
        # pipeline import); rely on filing.markdown() directly.
        from . import _edgar_patches  # noqa: F401
        try:
            md = f.markdown() or ""
        except Exception as e:
            log.warning("unit_detection: markdown fetch failed for %s: %s",
                        accession, e)
            return None

    if md:
        # Same whitespace collapse as the walker/overhang loaders —
        # token savings + Gemini fragile-payload 400 guard. Normalize
        # BEFORE capping so the cap buys more real content.
        md = normalize_filing_text(md)
    if md and len(md) > max_chars:
        md = md[:max_chars]
    return md


def _heuristic_ads_ratio(text: str) -> float | None:
    """Pre-LLM regex heuristic. Catches the most common phrasings and
    keeps the LLM call avoidable when the answer is unambiguous.
    Returns the LATEST stated ratio if multiple are present (file order
    is roughly oldest → newest)."""
    if not text:
        return None
    lower = text.lower()
    candidates: list[float] = []

    # "Each ADS represents N ordinary shares"
    for m in re.finditer(
        r"each\s+ad[sr]\s+represents\s+([\d,\.]+)\s+(?:ordinary|common|"
        r"class\s+[a-z]\s+ordinary|class\s+[a-z])\s+shares?",
        lower,
    ):
        try:
            candidates.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass

    # "One ADS represents N ordinary shares"
    for m in re.finditer(
        r"one\s+ad[sr]\s+represents\s+([\d,\.]+)\s+(?:ordinary|common)",
        lower,
    ):
        try:
            candidates.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass

    # "ratio of ADSs to ordinary shares is N:1" (N:1 direction) or
    # "the ratio is 1:N" (1:N direction, per the docstring supersession
    # example "1:100 to 1:400"). Either way the ratio is the non-unit number.
    # The 1:N arm requires a LITERAL colon (not the loose [:\s] separator) so
    # it can't swallow distractor MD&A prose like "the coverage ratio is 1 3
    # of EBITDA" (which would otherwise yield a bogus ratio of 3 and, being
    # appended last, override the correct ADS ratio). The N:1 arm keeps the
    # historical loose separator for back-compat.
    for m in re.finditer(
        r"ratio\s+(?:of\s+ads[sr]?\s+to\s+(?:ordinary|common)\s+shares\s+)?"
        r"(?:is|of)\s+(?:1\s*:\s*([\d,\.]+)|([\d,\.]+)\s*[:\s]\s*1)",
        lower,
    ):
        try:
            candidates.append(float((m.group(1) or m.group(2)).replace(",", "")))
        except (ValueError, AttributeError):
            pass

    # Treat the LAST candidate in the doc as authoritative — change-in-
    # ratio prose typically appears later than the original ratio.
    return candidates[-1] if candidates else None


async def _llm_ads_ratio(filing: dict) -> float | None:
    text = _load_text_for(filing["accession_number"])
    if not text:
        return None
    prompt = ADS_RATIO_PROMPT.format(
        form=filing["form"],
        filing_date=filing["filing_date"],
        text=text,
    )
    client = make_async_client()
    try:
        # Routed through the shared acomplete() so this call gets the
        # same request parameters as every other extractor. Its output
        # feeds unit_preamble() into every downstream walker/overhang
        # prompt, so variance here cascades issuer-wide.
        resp = await acomplete(
            client,
            name="ads-ratio",
            messages=[
                system(
                    "You extract the ADS-to-ordinary share ratio from the "
                    "cover page or Description of Securities of a Form "
                    "20-F/40-F. Output strict JSON only — no fences, no "
                    "prose."
                ),
                user(prompt),
            ],
            max_output_tokens=1024,
            cache_key="ads-ratio",
        )
    finally:
        await client.close()

    raw = (output_text(resp) or "").strip()
    # Tolerate accidental code fences.
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", raw).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("unit_detection: LLM returned non-JSON for %s: %r",
                    filing["accession_number"], raw[:200])
        return None
    val = obj.get("ads_ratio")
    if val is None:
        return None
    try:
        ratio = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ratio) or ratio <= 0 or ratio > 100_000:
        # NaN/Infinity are as implausible as an out-of-range number and must
        # not flow into unit_preamble() — math.isfinite rejects both.
        log.warning("unit_detection: implausible ratio %r for %s",
                    ratio, filing["accession_number"])
        return None
    log.info("  ads_ratio=%g for %s — %s",
             ratio, filing["accession_number"],
             obj.get("rationale", ""))
    return ratio


def _persist(cik: int, is_fpi: int, ads_ratio: float | None) -> None:
    with get_conn() as conn:
        conn.execute(
            """UPDATE dilution_company
               SET is_fpi = ?, ads_ratio = ?, unit_detected_at = ?
               WHERE cik = ?""",
            (is_fpi, ads_ratio, now_iso(), cik),
        )


def populate_company_unit(cik: int, force: bool = False) -> dict:
    """Detect is_fpi and ads_ratio, persist to dilution_company.

    Idempotent: skips work if already detected for this cik unless
    `force=True`. Returns the populated context dict.
    """
    with get_conn() as conn:
        row = conn.execute(
            """SELECT is_fpi, ads_ratio, unit_detected_at
               FROM dilution_company WHERE cik = ?""",
            (cik,),
        ).fetchone()
    already_detected = row and row["unit_detected_at"]
    if already_detected and not force:
        log.info("  unit already detected — is_fpi=%s ads_ratio=%s",
                 row["is_fpi"], row["ads_ratio"])
        return {"is_fpi": int(row["is_fpi"] or 0),
                "ads_ratio": row["ads_ratio"],
                "reporting_unit": "ads" if row["is_fpi"] else "common"}

    is_fpi = _is_fpi_from_filings(cik)
    if not is_fpi:
        _persist(cik, 0, None)
        log.info("  is_fpi=0 (no 20-F / 40-F filings)")
        return {"is_fpi": 0, "ads_ratio": None, "reporting_unit": "common"}

    annual = _latest_annual_filing(cik)
    if not annual:
        _persist(cik, 1, None)
        log.warning("  is_fpi=1 but no 20-F/40-F filing rows — leaving "
                    "ads_ratio=NULL")
        return {"is_fpi": 1, "ads_ratio": None, "reporting_unit": "ads"}

    text = _load_text_for(annual["accession_number"])
    ratio = _heuristic_ads_ratio(text or "")
    if ratio:
        log.info("  ads_ratio=%g (heuristic) from %s",
                 ratio, annual["accession_number"])
    else:
        ratio = asyncio.run(_llm_ads_ratio(annual))

    _persist(cik, 1, ratio)
    return {"is_fpi": 1, "ads_ratio": ratio,
            "reporting_unit": "ads"}
