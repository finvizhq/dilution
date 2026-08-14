"""LLM-call helpers shared by the ledger walker and overhang extractor.

Extracted from the legacy `dilution/extractors/base.py` during the
cutover. Owns the per-issuer unit preamble, chat-creation defaults,
and finish_reason warnings — anything walker / seed / anchor needs to
make a structured-output LLM call without dragging in the legacy
two-stage extractor module.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)


# ─── Filing-text whitespace normalization ───────────────────────────
_SPACE_RUN = re.compile(r" {3,}")


def normalize_filing_text(text: str) -> str:
    """Collapse cosmetic whitespace before filing text reaches an LLM.

    Two transforms, both pure noise for extraction:
      - zero-width spaces (U+200B) — edgartools markdown carries
        thousands per proxy/10-K as table-cell filler
      - runs of 3+ spaces → 2 (markdown table alignment padding)

    Cell boundaries are pipe-delimited, so intra-cell padding carries no
    meaning; a DEF 14A proxy shrinks ~60% in chars (~15% in tokens).

    Kept vendor-independent: it earned its place on Gemini, where
    ZWSP/space-padded table rows straddling an internal request boundary
    drew an opaque 400 INVALID_ARGUMENT (FCEL DEF 14A
    0001104659-22-025374 — byte-exact and alignment-dependent: appending
    one space to a 570 KB prefix flipped accept→reject, prepending 1 KB
    of prose flipped it back; repro harness
    scripts/debug_fcel_def14a_400.py). That specific server bug is
    behind us, but the token savings stand on their own.
    """
    return _SPACE_RUN.sub("  ", text.replace("\u200b", ""))


# ─── Per-issuer unit preamble ───────────────────────────────────────
def unit_preamble(unit_ctx: dict | None) -> str:
    """Per-issuer unit instruction prepended to every extraction prompt.

    For non-FPI issuers (US listings) this is a no-op note; for FPI/ADS
    issuers the LLM must report all share-count fields in ADS units
    rather than underlying ordinary/common shares.
    """
    if not unit_ctx or not unit_ctx.get("is_fpi"):
        return (
            "This issuer is a US listing. Report every share-count field "
            "in common shares (the listed and quoted instrument). Strike "
            "and conversion prices are in USD per common share.\n"
        )
    ratio = unit_ctx.get("ads_ratio")
    ratio_clause = (
        f"One ADS represents {ratio:g} ordinary shares per the most "
        "recent 20-F. "
        if ratio else
        "The ADS-to-ordinary ratio is stated on the cover page or in "
        "the Description of Securities section of the most recent 20-F. "
    )
    return (
        "This issuer is a Foreign Private Issuer listed in the US via "
        "American Depositary Shares (ADS). The ADS — not the underlying "
        f"ordinary share — is the listed instrument. {ratio_clause}\n\n"
        "REPORTING UNITS — ADS, NOT ORDINARY SHARES. Report every "
        "share-count field in ADS units. This applies to shares, "
        "warrants_issued, common_shares_issuable, outstanding_count, "
        "principal-divided counts, and holders.shares. If the filing "
        "states '129,818 ADSs representing 12,981,800 ordinary shares', "
        "emit 129,818, never 12,981,800. If the filing gives only "
        "ordinary shares, divide by the ADS-to-ordinary ratio and round "
        "to whole ADS. Strike and conversion prices are in USD per ADS, "
        "typically already stated that way in the filing.\n\n"
        "ADS SPLITS. An ADS-ratio change (for example '1 ADS now "
        "represents 400 ordinary shares instead of 100') is a split in "
        "ADS units — emit an `apply_split` mutation with `units=\"ads\"`, "
        "`direction` set to \"reverse\" or \"forward\", and `ratio` "
        "expressed in ADS terms (a change from 1:100 to 1:400 is a "
        "1-for-4 reverse for ADS holders, so `ratio=0.25, "
        "direction=\"reverse\", units=\"ads\"`). An underlying-ordinary "
        "split where the depositary adjusts the ADS ratio proportionally "
        "so ADS holders are unaffected is a no-op in ADS units — do not "
        "emit a split event for it.\n"
    )


# ─── Response checks ────────────────────────────────────────────────
# There is deliberately no temperature/seed constant here any more. The
# gpt-5.6 family rejects temperature and top_p in reasoning mode and
# /v1/responses has no seed parameter, so the old EXTRACT_TEMPERATURE=0.0
# + EXTRACT_SEED=42 pin is unsendable — see the DETERMINISM NOTE in
# config.py and the docstring on openai_client.request_kwargs. Chat
# construction lives in openai_client.request_kwargs.
DEFAULT_MAX_TOKENS = 32_000


def check_response(response, accession: str | None = None,
                   handler: str | None = None):
    """Log WARNING when a generation did not complete. Returns the
    response unchanged so callers can chain.

    Context overflow is absent by design: OpenAI rejects an oversized
    input with a 400 rather than reporting it on the response, and the
    pre-flight MAX_INPUT_CHARS cap is what keeps us under the ceiling.
    """
    tag = f"[{handler}] {accession}" if handler else (accession or "?")
    if getattr(response, "status", None) != "incomplete":
        return response
    reason = getattr(
        getattr(response, "incomplete_details", None), "reason", None)
    if reason == "max_output_tokens":
        log.warning("%s — output truncated at max_output_tokens", tag)
    else:
        log.warning("%s — generation incomplete (reason=%r)", tag, reason)
    return response


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "check_response",
    "normalize_filing_text",
    "unit_preamble",
]
