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

import config

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

    This is also a hard requirement on Gemini: payloads where long runs
    of ZWSP/space-padded table rows straddle an internal request-
    processing boundary are rejected with an opaque 400 INVALID_ARGUMENT
    (FCEL DEF 14A 0001104659-22-025374 — byte-exact and alignment-
    dependent: appending one space to a 570 KB prefix flips accept→
    reject, prepending 1 KB of prose flips it back). Collapsing the pads
    removes the fragile byte patterns entirely. Repro + bisection
    harness: scripts/debug_fcel_def14a_400.py.
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


# ─── Chat creation ──────────────────────────────────────────────────
EXTRACT_TEMPERATURE = 0.0
EXTRACT_SEED = 42
DEFAULT_MAX_TOKENS = 32_000


def make_chat(client, *, response_format=None,
              tools=None, tool_choice=None,
              max_tokens: int = DEFAULT_MAX_TOKENS,
              temperature: float = EXTRACT_TEMPERATURE,
              seed: int = EXTRACT_SEED,
              model: str | None = None):
    """Build a chat with our standard extraction params.

    Two mutually-meaningful constraint modes:
      - `response_format` (Pydantic class): structured-output path used
        by callers that emit a single typed JSON value (e.g. the
        overhang extractor's typed event list). xAI binds the Pydantic
        class natively; Moonshot falls back to json_object; Gemini
        translates to json_schema. Constrains shape + Literal enums at
        decode time but cannot enforce required dict keys — the walker
        uses tool calls for that reason.
      - `tools` + `tool_choice`: function-calling path. Each tool's
        JSON-Schema parameters block IS the contract — required args,
        types, minLength, enum, additionalProperties=false enforced at
        decode time. `tool_choice="required"` forces ≥1 call.

    Both can be passed simultaneously; providers handle the combination.
    """
    kwargs = {
        "model": model if model is not None else config.LLM_MODEL,
        "max_tokens": max_tokens,
        # Temperature is sent to every provider that accepts it. Gemini's
        # adapter forwards it (llm_provider._gemini_request_kwargs); xAI
        # binds it natively; Moonshot enforces a fixed temperature and
        # silently drops this via **_dropped. This MUST stay out of the
        # xai-only gate below: gating it there is what left the active
        # Gemini provider running at the API default (1.0) instead of
        # EXTRACT_TEMPERATURE=0.0, the dominant source of run-to-run
        # walker non-determinism.
        "temperature": temperature,
    }
    if config.LLM_PROVIDER == "xai":
        # Seed stays xai-only: Gemini's OpenAI-compat endpoint 400s on
        # seed (INVALID_ARGUMENT), and Moonshot enforces a fixed seed.
        kwargs["seed"] = seed
    if response_format is not None:
        kwargs["response_format"] = response_format
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
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
    "EXTRACT_SEED",
    "EXTRACT_TEMPERATURE",
    "asample_and_check",
    "check_response",
    "make_chat",
    "unit_preamble",
]
