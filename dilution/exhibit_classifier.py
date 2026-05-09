"""LLM classifier deciding whether an SEC exhibit is dilution-relevant.

Replaces the keyword sniff that previously gated EX-10.x / EX-99.x
storage and the 6-K primary doc. The keyword sniff had a known
false-negative mode: SPAs whose recitals push past the first 32KB of
content slipped through, dropping investor names / placement-agent /
price-protection clauses from the cards. The classifier reads the
doc-type label, filename, and a content preview and returns a
structured verdict.

Failure mode: ON ANY ERROR the classifier returns relevant=True. Per
CLAUDE.md, a silently-dropped filing is worse than a slow filing — we
fail open so a transient xAI error never silently drops content.
"""

import logging

from pydantic import BaseModel, Field
from .llm_provider import system, user

import config

from .ledger._llm_utils import (
    EXTRACT_SEED,
    EXTRACT_TEMPERATURE,
    check_response,
)

log = logging.getLogger(__name__)

# 8KB of content is enough to identify the document family from
# headings, definitions, and section labels. The keyword sniff used
# 32KB to survive long boilerplate recitals — the LLM doesn't need
# that runway because it understands the doc beyond literal keywords.
PREVIEW_CHARS = 8_000

CLASSIFIER_VERSION = "exhibit-classifier-v1"


# ─── Description-based routing (skip the LLM where we can) ──────────
#
# Issuers describe their EX-10 / EX-99 exhibits in the SEC filing
# index (surfaced by edgartools as ``Attachment.description``). When
# the description is unambiguously dilution-relevant or unambiguously
# not, we route deterministically and skip the LLM call — saves one
# round-trip per exhibit on a high-volume universe.
#
# CONSERVATIVE policy:
#   - Each phrase is a multi-word substring with a single common
#     meaning in the EX-10 / EX-99 universe. No single-word matches.
#   - On conflict (description matches both KEEP and DROP), KEEP wins.
#     Coverage > size savings (CLAUDE.md).
#   - Generic descriptions ("EX-10.1", "EXHIBIT 99.1") fall through
#     to the LLM classifier unchanged.
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
    """Deterministic pre-classifier on ``Attachment.description``.

    Returns one of:
        'drop'    — substring matched DROP-list, no LLM call needed
        'keep'    — substring matched KEEP-list, no LLM call needed
        'unknown' — generic / ambiguous, fall through to LLM

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


class ExhibitVerdict(BaseModel):
    relevant: bool = Field(
        description=(
            "True if this exhibit contains dilution-relevant content "
            "for an equity / convertible / warrant / preferred / ATM / "
            "equity-line / shelf offering. False for HR/comp documents, "
            "leases, supply contracts, earnings releases, business "
            "updates, governance notices."
        )
    )


SYSTEM = (
    "You are an SEC filing classifier. You decide whether an exhibit "
    "attached to a filing contains dilution-relevant content. When in "
    "doubt prefer relevant=true so downstream extraction can examine "
    "it. Output strictly conforms to the ExhibitVerdict schema."
)

PROMPT = """\
Decide whether this exhibit contains dilution-relevant disclosure.

RELEVANT (return relevant=true):
- Securities Purchase Agreement (SPA), Registration Rights Agreement (RRA)
- Warrant agreement, form of warrant, form of pre-funded warrant
- Indenture, convertible note / debenture, supplemental indenture
- Placement agency agreement, underwriting agreement
- At-The-Market / Equity Distribution / Controlled Equity Offering agreement
- Standby equity / committed equity / common-stock purchase agreement
  (equity-line / ELOC)
- Certificate of Designation for preferred stock
- Subscription agreement for a private placement
- Press release announcing an offering, private placement, registered
  direct, rights offering, share consolidation, placing (UK/AIM),
  bonus issue, cashbox, scrip dividend, open offer
- 6-K cover letter referencing any of the above

NOT RELEVANT (return relevant=false):
- Employment agreement, executive compensation plan, separation agreement
- Lease, sublease, supply agreement, license agreement, commercial
  distribution agreement
- Earnings release, business update, product announcement, governance
  notice (board changes), routine common-stock dividend declaration
- Code of ethics, audit-committee charter, insider trading policy

Filing context:
  Form: {form}
  Doc type: {doc_type}
  Filename: {doc_name}

Document preview (first {preview_len} chars):
{content}
"""


def _build_classify_prompt(*, doc_type, doc_name, form, content):
    preview = content[:PREVIEW_CHARS]
    return preview, PROMPT.format(
        form=form or "(unknown)",
        doc_type=doc_type or "(unknown)",
        doc_name=doc_name or "(unknown)",
        preview_len=len(preview),
        content=preview,
    )


def classify_exhibit(client, *, doc_type: str | None, doc_name: str | None,
                     form: str | None, content: str,
                     accession: str | None = None) -> bool:
    """Return True iff the exhibit is dilution-relevant. Fail-open on
    any exception so a transient xAI error never silently drops a
    filing."""
    if not content or not content.strip():
        return False

    _, prompt = _build_classify_prompt(doc_type=doc_type, doc_name=doc_name,
                                       form=form, content=content)

    try:
        chat = client.chat.create(
            model=config.LLM_MODEL,
            max_tokens=1024,
            temperature=EXTRACT_TEMPERATURE,
            seed=EXTRACT_SEED,
            response_format=ExhibitVerdict,
        )
        chat.append(system(SYSTEM))
        chat.append(user(prompt))
        response = check_response(chat.sample(), accession=accession,
                                  handler="exhibit_classifier")
        verdict = ExhibitVerdict.model_validate_json(response.content)
        log.info("  classify %s/%s [%s] → %s",
                 doc_type or "?", doc_name or "?", accession or "?",
                 "KEEP" if verdict.relevant else "DROP")
        return verdict.relevant
    except Exception as e:
        log.warning("  classify FAILED for %s/%s on %s — failing open: %s",
                    doc_type, doc_name, accession or "?", e)
        return True


async def classify_exhibit_async(client, *, doc_type: str | None,
                                 doc_name: str | None, form: str | None,
                                 content: str,
                                 accession: str | None = None) -> bool:
    """Async sibling of classify_exhibit. Same fail-open contract — a
    transient xAI error returns True so the exhibit is kept."""
    if not content or not content.strip():
        return False

    _, prompt = _build_classify_prompt(doc_type=doc_type, doc_name=doc_name,
                                       form=form, content=content)

    try:
        chat = client.chat.create(
            model=config.LLM_MODEL,
            max_tokens=1024,
            temperature=EXTRACT_TEMPERATURE,
            seed=EXTRACT_SEED,
            response_format=ExhibitVerdict,
        )
        chat.append(system(SYSTEM))
        chat.append(user(prompt))
        response = check_response(await chat.sample(), accession=accession,
                                  handler="exhibit_classifier")
        verdict = ExhibitVerdict.model_validate_json(response.content)
        log.info("  classify %s/%s [%s] → %s",
                 doc_type or "?", doc_name or "?", accession or "?",
                 "KEEP" if verdict.relevant else "DROP")
        return verdict.relevant
    except Exception as e:
        log.warning("  classify FAILED for %s/%s on %s — failing open: %s",
                    doc_type, doc_name, accession or "?", e)
        return True
