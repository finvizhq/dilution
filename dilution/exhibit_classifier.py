"""Deterministic routing for SEC exhibit attachments.

Previously this module also held an LLM-based fallback that read an
8KB preview of each exhibit and returned a KEEP/DROP verdict. The
fallback was removed: the walker LLM re-checks every stored filing
end-to-end via tool calls (see `walker.py`), making the per-exhibit
LLM gate redundant prework. The deterministic description router
remains because it's free and prevents bulk boilerplate (employment
agreements, leases, pro-forma financials, earnings decks) from ever
hitting `dilution_raw`.

Unknown descriptions fall through as `"unknown"`; the caller treats
that as KEEP-by-default so a missing or generic description never
silently drops content.
"""

import logging

log = logging.getLogger(__name__)


# ─── Description-based routing ──────────────────────────────────────
#
# Issuers describe their EX-10 / EX-99 exhibits in the SEC filing
# index (surfaced by edgartools as ``Attachment.description``). We
# route deterministically on unambiguous substrings; ambiguous /
# generic descriptions fall through as "unknown" and the caller
# keeps them.
#
# CONSERVATIVE policy:
#   - Each phrase is a multi-word substring with a single common
#     meaning in the EX-10 / EX-99 universe. No single-word matches.
#   - On conflict (description matches both KEEP and DROP), KEEP wins.
#     Coverage > size savings (CLAUDE.md).
DESCRIPTION_DROP_PHRASES = (
    # HR / executive comp / governance — never carry dilution events.
    "EMPLOYMENT AGREEMENT",
    "EXECUTIVE EMPLOYMENT",
    "SEPARATION AGREEMENT",
    "DIRECTOR SERVICES AGREEMENT",
    "TRANSITION AGREEMENT",
    "RETIREMENT AGREEMENT",
    "OFFER LETTER",
    "INDEMNIFICATION AGREEMENT",
    "AUDIT COMMITTEE CHARTER",
    "CODE OF ETHICS",
    "INSIDER TRADING POLICY",
    # Operations contracts — never carry dilution events.
    "LEASE AGREEMENT",
    "SUPPLY AGREEMENT",
    "DISTRIBUTION AGREEMENT",
    # M&A artifacts that aren't the issuance instrument itself.
    "PRO FORMA CONDENSED",
    "PRO FORMA COMBINED",
    "PRO FORMA CONSOLIDATED",
    "FINANCIAL STATEMENTS OF",
    "AUDITED FINANCIAL STATEMENTS",
    "UNAUDITED FINANCIAL STATEMENTS",
    # Marketing / IR — earnings + investor decks.
    "INVESTOR PRESENTATION",
    "EARNINGS PRESENTATION",
    "EARNINGS CALL TRANSCRIPT",
    "CONFERENCE CALL TRANSCRIPT",
    "PREPARED REMARKS",
)

DESCRIPTION_KEEP_PHRASES = (
    # SPA family — primary dilution instrument.
    "SECURITIES PURCHASE AGREEMENT",
    "REGISTRATION RIGHTS AGREEMENT",
    "SUBSCRIPTION AGREEMENT",
    # Warrants.
    "WARRANT AGREEMENT",
    "FORM OF WARRANT",
    "FORM OF PRE-FUNDED WARRANT",
    "FORM OF PREFUNDED WARRANT",
    "FORM OF COMMON STOCK PURCHASE WARRANT",
    "FORM OF COMMON WARRANT",
    "FORM OF SERIES A WARRANT",
    "FORM OF SERIES B WARRANT",
    # Convertible debt.
    "FORM OF DEBENTURE",
    "FORM OF CONVERTIBLE NOTE",
    "FORM OF SENIOR NOTE",
    "SUPPLEMENTAL INDENTURE",
    # Underwriting / placement.
    "PLACEMENT AGENCY AGREEMENT",
    "PLACEMENT AGENT AGREEMENT",
    "UNDERWRITING AGREEMENT",
    # ATM / equity-line.
    "AT-THE-MARKET",
    "AT THE MARKET",
    "EQUITY DISTRIBUTION AGREEMENT",
    "CONTROLLED EQUITY OFFERING",
    "SALES AGREEMENT",  # standard ATM exhibit name (with placement agent)
    "STANDBY EQUITY",
    "EQUITY PURCHASE AGREEMENT",
    "COMMON STOCK PURCHASE AGREEMENT",
    # Preferred.
    "CERTIFICATE OF DESIGNATION",
)


def classify_by_description(*, description: str | None) -> str:
    """Deterministic classifier on ``Attachment.description``.

    Returns one of:
        'drop'    — substring matched DROP-list, exhibit is skipped
        'keep'    — substring matched KEEP-list, exhibit is stored
        'unknown' — generic / ambiguous, caller keeps by default

    KEEP wins on conflict so a description like
    ``"WARRANT INDEMNIFICATION AGREEMENT"`` doesn't silently drop.
    """
    if not description:
        return "unknown"
    d = description.upper()
    has_keep = any(p in d for p in DESCRIPTION_KEEP_PHRASES)
    if has_keep:
        return "keep"
    if any(p in d for p in DESCRIPTION_DROP_PHRASES):
        return "drop"
    return "unknown"
