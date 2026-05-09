"""Walker LLM call layer.

Wraps the existing llm_provider abstraction with the walker-specific
prompt + MutationList response_format. The walker proper
(walker.py) calls into here once per filing.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

from pydantic import ValidationError

import config
from dilution.llm_provider import system, user

from ._llm_utils import (
    DEFAULT_MAX_TOKENS,
    asample_and_check,
    make_chat,
)
from .mutations import CreateInstrument, Mutation, MutationList
from .walker_prompt import SYSTEM_PROMPT, build_user_prompt

log = logging.getLogger(__name__)


WALKER_VERSION = "ledger-walker-v2"


# Filing text cap. Same as the legacy extractor. Grok-4-fast 2M-token
# context handles 2M chars (~500K tokens) plus our system + ledger view
# + mutation reference comfortably.
MAX_INPUT_CHARS = 2_000_000

# Max tokens for the walker output. A serial-diluter 8-K can emit
# 5-10 mutations; periodic filings (10-K Subsequent Events sections
# on heavy issuers) can emit dozens. 6× the default leaves headroom
# for multi-mutation filings and avoids the early STOPs we saw on
# XTIA where the model truncated mid-string at ~500 chars.
WALKER_MAX_TOKENS = DEFAULT_MAX_TOKENS * 6


async def walk_filing(
    *,
    client,
    unit_preamble: str,
    ledger_view: str,
    form: str,
    filing_date: str,
    accession: str,
    items: str | None,
    period_of_report: str | None,
    filing_text: str,
    active_rows: list[dict] | None = None,
) -> MutationList:
    """Send one filing to the LLM and return a validated MutationList.

    On parse failure, returns an empty MutationList and logs a WARNING —
    the walker continues so a single bad filing doesn't kill the run.
    The filing's accession is recorded in dilution_walk_errors at the
    walker level when the result is empty unexpectedly.

    `active_rows` is the structured ledger snapshot the walker is about
    to validate against. When supplied, post-parse guards drop
    duplicate `create_instrument` and unsafe `create_instrument(equity)`
    mutations the prompt is supposed to suppress but the LLM sometimes
    still emits.
    """
    if len(filing_text) > MAX_INPUT_CHARS:
        log.warning("%s — truncating filing %d→%d chars (dropped %d, %.1f%%)",
                    accession, len(filing_text), MAX_INPUT_CHARS,
                    len(filing_text) - MAX_INPUT_CHARS,
                    (len(filing_text) - MAX_INPUT_CHARS) / len(filing_text) * 100)
        filing_text = filing_text[:MAX_INPUT_CHARS]

    user_prompt = build_user_prompt(
        unit_preamble=unit_preamble,
        ledger_view=ledger_view,
        form=form,
        filing_date=filing_date,
        accession=accession,
        items=items,
        period_of_report=period_of_report,
        filing_text=filing_text,
    )

    chat = make_chat(client,
                     response_format=MutationList,
                     max_tokens=WALKER_MAX_TOKENS)
    chat.append(system(SYSTEM_PROMPT))
    chat.append(user(user_prompt))
    response = await asample_and_check(chat, accession=accession,
                                       handler="ledger-walker")
    finish_reason = getattr(response, "finish_reason", None)
    mlist = _parse(response.content, accession=accession,
                   finish_reason=finish_reason)
    if active_rows is not None:
        mlist = _apply_guards(mlist, active_rows=active_rows,
                              filing_date=filing_date, accession=accession)
    return mlist


_DUMP_DIR = Path(__file__).resolve().parents[2] / "walker_dumps"


def _dump_failed_response(accession: str, content: str) -> str:
    """Persist the full LLM response to walker_dumps/ for postmortem.
    Returns the path (or '<dump-failed>' on filesystem error)."""
    try:
        _DUMP_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        path = _DUMP_DIR / f"{accession}.{ts}.json"
        path.write_text(content)
        return str(path)
    except OSError as e:
        log.warning("dump failed for %s: %s", accession, e)
        return "<dump-failed>"


def _parse(content: str, *, accession: str,
           finish_reason: str | None = None) -> MutationList:
    """Validate the LLM response into a MutationList.

    Two-pass: first try the whole-list validate, then on failure parse
    JSON and fall back to row-by-row salvage so one bad mutation
    doesn't drop the whole filing's output."""
    if not content:
        log.warning("walker %s — empty response (finish_reason=%r)",
                    accession, finish_reason)
        return MutationList(mutations=[])
    try:
        return MutationList.model_validate_json(content)
    except ValidationError as exc:
        primary_err = exc

    # Salvage path
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as je:
        dump_path = _dump_failed_response(accession, content)
        log.warning("walker %s — non-JSON response (%s) "
                    "finish_reason=%r len=%d head=%r tail=%r dump=%s",
                    accession, je, finish_reason, len(content),
                    content[:300], content[-300:], dump_path)
        return MutationList(mutations=[])

    rows = raw.get("mutations") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        log.warning("walker %s — schema validation failed (no "
                    "mutations array) finish_reason=%r: %s",
                    accession, finish_reason, primary_err)
        return MutationList(mutations=[])

    salvaged: list = []
    skipped = 0
    for r in rows:
        if not isinstance(r, dict):
            skipped += 1
            continue
        try:
            single = MutationList.model_validate({"mutations": [r]})
            salvaged.extend(single.mutations)
        except ValidationError:
            skipped += 1
    if skipped:
        log.warning("walker %s — salvaged %d/%d mutations (skipped %d)",
                    accession, len(salvaged), len(rows), skipped)
    return MutationList(mutations=salvaged)


# ─── Post-parse guards ──────────────────────────────────────────────
# Two failure modes the walker prompt warns against but the LLM still
# occasionally emits. These run AFTER the LLM responds and BEFORE
# validate.py / apply_mutations, so we catch the bug regardless of
# prompt fidelity and log a metric for prompt regressions.

# strike-match tolerance: ±2% of the prompt's stated key
_STRIKE_TOL = 0.02
# created-date tolerance: ±30 days of THIS filing's date
_CREATED_TOL_DAYS = 30
# equity-while-shelf check: an active shelf vehicle within ±N days
# of the filing makes a `create_instrument(equity)` from the same
# filing presumptively a drawdown that should ride the shelf.
_SHELF_FAMILY = {"shelf", "atm", "equity_line", "s1_offering"}
_SHELF_GUARD_DAYS = 30
# Instrument types subject to (strike, created) dedup. ATM / shelf /
# equity_line / s1_offering aren't priced instruments so they don't
# go through this path; they have their own separate dedup keys
# (capacity_usd, drawdown match) handled at apply_mutations time.
_PRICED_TYPES = {"warrant", "convertible", "preferred"}


def _strike_of(row: dict) -> float | None:
    """Pull the strike-equivalent from a ledger row's terms.

    warrant     → terms.strike
    convertible → terms.conv_price
    preferred   → terms.conv_price
    """
    terms = row.get("terms") or {}
    if row.get("type") == "warrant":
        v = terms.get("strike")
    else:
        v = terms.get("conv_price")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _strike_of_create(m: CreateInstrument) -> float | None:
    if m.type == "warrant":
        v = m.terms.get("strike")
    else:
        v = m.terms.get("conv_price")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _parse_iso(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _strike_within(a: float, b: float, tol: float = _STRIKE_TOL) -> bool:
    """Symmetric ±tol relative match. Pre-funded warrants live near
    zero, so we also accept exact equality at zero."""
    if a == b:
        return True
    if a == 0 or b == 0:
        return False
    base = max(abs(a), abs(b))
    return abs(a - b) / base <= tol


def _is_dup_create(m: CreateInstrument, active_rows: list[dict],
                   filing_d: date | None) -> dict | None:
    """Return the matching ledger row if `m` re-discloses an existing
    instrument by (type, strike ±2%, created ±30d of filing_date),
    else None. Picks the earliest-created on ties — same behavior the
    prompt prescribes."""
    if m.type not in _PRICED_TYPES:
        return None
    if filing_d is None:
        return None
    m_strike = _strike_of_create(m)
    if m_strike is None:
        return None
    matches = []
    for row in active_rows:
        if row.get("type") != m.type:
            continue
        if (row.get("status") or "").startswith("superseded"):
            continue
        if row.get("status") not in (None, "active"):
            continue
        r_strike = _strike_of(row)
        if r_strike is None:
            continue
        if not _strike_within(m_strike, r_strike):
            continue
        r_created = _parse_iso(row.get("created_at"))
        if r_created is None:
            continue
        if abs((filing_d - r_created).days) > _CREATED_TOL_DAYS:
            continue
        matches.append(row)
    if not matches:
        return None
    matches.sort(key=lambda r: r.get("created_at") or "")
    return matches[0]


def _shelf_within_window(m: CreateInstrument, active_rows: list[dict],
                         filing_d: date | None) -> dict | None:
    """For `create_instrument(type='equity')`: return the active shelf-
    family row that should be carrying this issuance, or None."""
    if m.type != "equity" or filing_d is None:
        return None
    candidates = []
    for row in active_rows:
        if row.get("type") not in _SHELF_FAMILY:
            continue
        if row.get("status") not in (None, "active"):
            continue
        r_created = _parse_iso(row.get("created_at"))
        if r_created is None:
            continue
        if abs((filing_d - r_created).days) > _SHELF_GUARD_DAYS:
            continue
        candidates.append(row)
    if not candidates:
        return None
    # Pick the most-recently-created shelf-family row.
    candidates.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return candidates[0]


def _apply_guards(mlist: MutationList, *, active_rows: list[dict],
                  filing_date: str, accession: str) -> MutationList:
    """Post-parse failsafes. Drops the two creates the prompt warns
    against but the LLM sometimes still emits."""
    filing_d = _parse_iso(filing_date)
    if filing_d is None:
        return mlist

    kept: list[Mutation] = []
    dropped_dup = 0
    dropped_equity = 0
    for m in mlist.mutations:
        if isinstance(m, CreateInstrument):
            dup = _is_dup_create(m, active_rows, filing_d)
            if dup is not None:
                log.warning(
                    "walker %s — guard dropped duplicate create %s "
                    "(matches existing %s strike=%s created=%s)",
                    accession, m.type, dup.get("instrument_id"),
                    _strike_of(dup), dup.get("created_at"))
                dropped_dup += 1
                continue
            shelf = _shelf_within_window(m, active_rows, filing_d)
            if shelf is not None:
                log.warning(
                    "walker %s — guard dropped create_instrument(equity) "
                    "(active %s %s created=%s within %dd of filing)",
                    accession, shelf.get("type"),
                    shelf.get("instrument_id"), shelf.get("created_at"),
                    _SHELF_GUARD_DAYS)
                dropped_equity += 1
                continue
        kept.append(m)
    if dropped_dup or dropped_equity:
        log.info(
            "walker %s — guards dropped %d dup-creates, %d equity-on-shelf",
            accession, dropped_dup, dropped_equity)
    return MutationList(mutations=kept)


def pipeline_version() -> str:
    """Stamp recorded in dilution_walk_state.pipeline_version. Drift in
    EITHER the model OR the walker prompt triggers re-walks under
    --force semantics."""
    return f"{config.LLM_MODEL}/{WALKER_VERSION}"


__all__ = [
    "MAX_INPUT_CHARS",
    "WALKER_VERSION",
    "pipeline_version",
    "walk_filing",
]
