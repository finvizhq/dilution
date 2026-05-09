"""LLM-call helpers shared by the ledger walker and overhang extractor.

Extracted from the legacy `dilution/extractors/base.py` during the
cutover. Owns the per-issuer unit preamble, chat-creation defaults,
and finish_reason warnings — anything walker / seed / anchor needs to
make a structured-output LLM call without dragging in the legacy
two-stage extractor module.
"""

from __future__ import annotations

import logging

import config

log = logging.getLogger(__name__)


# ─── Per-issuer unit preamble ───────────────────────────────────────
def unit_preamble(unit_ctx: dict | None) -> str:
    """Per-issuer unit instruction prepended to every extraction prompt.

    For non-FPI issuers (US listings) this is a no-op note; for FPI/ADS
    issuers the LLM must report all share-count fields in ADS units
    rather than underlying ordinary/common shares.

    Avoid `ALL_CAPS:` section labels — Moonshot (json_object mode) has
    been observed echoing the preamble back as JSON keys. Plain prose
    paragraphs read the same to a strong model and don't trigger the
    echo.
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
        "Report every share-count field in ADS units rather than "
        "ordinary shares. This applies to shares, warrants_issued, "
        "common_shares_issuable, outstanding_count, principal-divided "
        "counts, and holders.shares. If the filing states '129,818 ADSs "
        "representing 12,981,800 ordinary shares', emit 129,818, never "
        "12,981,800. If the filing gives only ordinary shares, divide by "
        "the ADS-to-ordinary ratio and round to whole ADS. Strike and "
        "conversion prices are in USD per ADS, typically already stated "
        "that way in the filing.\n\n"
        "Splits work differently for ADS issuers. An ADS-ratio change "
        "(for example '1 ADS now represents 400 ordinary shares instead "
        "of 100') is a split in ADS units — emit an `apply_split` "
        "mutation with `units=\"ads\"`, `direction` set to \"reverse\" "
        "or \"forward\", and `ratio` expressed in ADS terms (a change "
        "from 1:100 to 1:400 is a 1-for-4 reverse for ADS holders, so "
        "`ratio=0.25, direction=\"reverse\", units=\"ads\"`). An "
        "underlying-ordinary split where the depositary adjusts the ADS "
        "ratio proportionally so ADS holders are unaffected is a no-op "
        "in ADS units — do not emit a split event for it.\n"
    )


# ─── Chat creation ──────────────────────────────────────────────────
EXTRACT_TEMPERATURE = 0.0
EXTRACT_SEED = 42
DEFAULT_MAX_TOKENS = 32_000


def make_chat(client, *, response_format=None,
              max_tokens: int = DEFAULT_MAX_TOKENS,
              temperature: float = EXTRACT_TEMPERATURE,
              seed: int = EXTRACT_SEED):
    """Build a chat with our standard extraction params.

    `response_format` accepts a Pydantic class. xAI binds it natively;
    Moonshot translates to json_object mode and downstream Pydantic
    validation runs in the per-call parser.

    Moonshot K2.6 enforces a fixed temperature/seed and rejects custom
    values, so we omit those kwargs on the moonshot path.
    """
    kwargs = {
        "model": config.LLM_MODEL,
        "max_tokens": max_tokens,
    }
    if config.LLM_PROVIDER == "xai":
        kwargs["temperature"] = temperature
        kwargs["seed"] = seed
    if response_format is not None:
        kwargs["response_format"] = response_format
    return client.chat.create(**kwargs)


def check_response(response, accession: str | None = None,
                   handler: str | None = None):
    """Log WARNING on truncation / context overflow / timeout. Returns
    the response unchanged so callers can chain."""
    fr = getattr(response, "finish_reason", None)
    tag = f"[{handler}] {accession}" if handler else (accession or "?")
    if fr == "REASON_MAX_LEN":
        log.warning("%s — output truncated at max_tokens", tag)
    elif fr == "REASON_MAX_CONTEXT":
        log.warning("%s — input exceeded model context window", tag)
    elif fr == "REASON_TIME_LIMIT":
        log.warning("%s — generation hit time limit", tag)
    return response


async def asample_and_check(chat, accession: str | None = None,
                            handler: str | None = None):
    """Async helper: await chat.sample() then check_response."""
    return check_response(await chat.sample(), accession=accession,
                          handler=handler)


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "asample_and_check",
    "check_response",
    "make_chat",
    "unit_preamble",
]
