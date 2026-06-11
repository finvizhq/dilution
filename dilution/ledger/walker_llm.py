"""Walker LLM call layer.

Wraps the project's llm_provider abstraction with the walker-specific
system prompt + tool-call surface. The walker proper (walker.py) calls
into here once per filing.

The model is forced to emit tool calls (`tool_choice="required"`) from
a form-specific subset of the canonical tools defined in
dilution/ledger/tools/. Required-argument schema validation at decode
time eliminates the empty-terms class of bugs at its source.
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import date, datetime

import config
from dilution.llm_provider import system, user

from ._llm_utils import (
    DEFAULT_MAX_TOKENS,
    EXTRACT_SEED,
    asample_and_check,
    make_chat,
)
from .mutations import (
    AmendConvertible,
    AmendMutation,
    AmendPreferred,
    AmendWarrant,
    ApplySplit,
    CloseInstrument,
    CreateMutation,
    CreateShelf,
    CreateWarrant,
    Mutation,
    MutationList,
    NoteNoEvent,
    RecordMutation,
    RestateAtm,
    fmt_mutation,
    warrant_series_key,
)
from .tools import (
    RetryableFailure, TOOLS_FOR_FORM, build_provider_schema,
    parse_tool_calls, tools_for_form,
)
from .walker_prompt import SYSTEM_PROMPT, build_user_prompt

log = logging.getLogger(__name__)


WALKER_VERSION = "ledger-walker-v7"


# Filing-text input cap, sized to the Gemini input window
# (config.GEMINI_INPUT_TOKEN_LIMIT = 1,048,576 tokens, verified). SEC text
# tokenizes at ~3.5-4 chars/token, so 3M chars is ~750-860K tokens, leaving
# ~190-300K of the window for the system prompt, ledger view, and tool
# schemas. (Output is a separate 65,536-token budget that does not consume
# the input window.) This is an outlier guard — real filings top out ~1M+
# chars (S-1/F-1), well under — so it rarely fires; over-cap text is
# truncated with a logged warning. Heads-up: at the densest tokenization
# (~3 chars/token) a full 3M-char filing approaches the whole window, so
# ~2.8M is the guaranteed-no-overflow ceiling if margin ever matters.
MAX_INPUT_CHARS = 3_000_000

# Max tokens for the walker output (tool calls). Measured across 1,090
# eval filings: mean 1.4 calls/response, max 15 in any single response;
# 15 heavy create_instrument calls serialize to ~3.2K tokens, so 8K leaves
# ~2.5× headroom over the worst case observed (and the walker has never hit
# the old 192K cap). Break-even where 8K would truncate is ~40+ heavy calls
# in one response — never seen. Unlike the overhang path there's no salvage,
# but a truncation is logged (asample_and_check → check_response →
# REASON_MAX_LEN); if a future mega-diluter filing emits dozens, raise this.
WALKER_MAX_TOKENS = 8_000


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
    attribution_block: str = "",
    fee_table_block: str = "",
    must_record: bool = False,
    must_record_reason: str = "",
) -> MutationList:
    """Send one filing to the LLM and return a validated MutationList.

    On parse failure, returns an empty MutationList and logs a WARNING —
    the walker continues so a single bad filing doesn't kill the run.

    `active_rows` is the structured ledger snapshot the walker is about
    to validate against. When supplied, post-parse guards drop
    duplicate creates and unsafe equity creates that the prompt warns
    against but the LLM sometimes still emits.

    `attribution_block` is the file_number-derived hard hint built by
    walker._build_attribution_block. Empty string when no primary
    parent could be resolved.

    `must_record` (8-K only, from classify_8k) marks a filing whose SEC
    item/exhibit metadata asserts a dilutive instrument is disclosed. When
    set and the first pass records nothing, we take one focused second look
    (the must_record net below) before accepting silence.
    """
    if len(filing_text) > MAX_INPUT_CHARS:
        log.warning("%s — truncating filing %d→%d chars (dropped %d, %.1f%%)",
                    accession, len(filing_text), MAX_INPUT_CHARS,
                    len(filing_text) - MAX_INPUT_CHARS,
                    (len(filing_text) - MAX_INPUT_CHARS) / len(filing_text) * 100)
        filing_text = filing_text[:MAX_INPUT_CHARS]

    dedup_block = _build_dedup_candidates_block(
        active_rows, filing_date, accession=accession, form=form,
    )
    user_prompt = build_user_prompt(
        unit_preamble=unit_preamble,
        ledger_view=ledger_view,
        form=form,
        filing_date=filing_date,
        accession=accession,
        items=items,
        period_of_report=period_of_report,
        filing_text=filing_text,
        dedup_candidates_block=dedup_block,
        attribution_block=attribution_block,
        fee_table_block=fee_table_block,
    )

    tool_set = tools_for_form(form) if form in TOOLS_FOR_FORM else None
    if not tool_set:
        log.warning(
            "walker %s — no tool subset for form %r; treating as no-op",
            accession, form,
        )
        return MutationList(mutations=[])

    # Keep the unpruned set for the must_record second look (rare path): if
    # classify_8k asserts an instrument is here but the first pass is silent,
    # we re-ask with the FULL tool set so pruning can never starve recovery.
    # (Today every must_record signal also rescues its create tool, so this is
    # belt-and-suspenders against future rescue-map drift.)
    full_tool_set = tool_set

    # Per-filing create-tool pruning: drop the create_* schemas a
    # non-periodic filing can't plausibly need (~71% of an 8-K call's input
    # is tool schemas). Four-signal OR (keyword / item / exhibit / ledger),
    # zero-miss verified full-DB. No-op for periodic forms. See
    # item_classification.prune_create_tools.
    from .item_classification import prune_create_tools
    tool_set = prune_create_tools(
        tool_set, form=form, accession=accession, items=items,
        filing_text=filing_text, active_rows=active_rows,
    )

    provider_tools = [
        build_provider_schema(t, provider=config.LLM_PROVIDER)
        for t in tool_set
    ]
    log.info(
        "walker %s — form=%s (%d tools)",
        accession, form, len(tool_set),
    )

    chat = make_chat(
        client,
        tools=provider_tools,
        tool_choice="required",
        max_tokens=WALKER_MAX_TOKENS,
        seed=EXTRACT_SEED,
    )
    chat.append(system(SYSTEM_PROMPT))
    chat.append(user(user_prompt))
    response = await asample_and_check(
        chat, accession=accession, handler="ledger-walker",
    )
    log.info(
        "walker %s — finish=%r calls=%d",
        accession, response.finish_reason,
        len(response.tool_calls or []),
    )

    if not response.tool_calls:
        log.warning(
            "walker %s — 0 tool calls returned (finish_reason=%r); "
            "treating as no-op", accession, response.finish_reason,
        )
        return MutationList(mutations=[])

    failures: list[RetryableFailure] = []
    typed = parse_tool_calls(
        response.tool_calls, accession=accession,
        empty_amends=failures,
    )

    # Single retry round when at least one tool call failed in a way
    # that's worth re-asking the model about. Two flavours today:
    #   - empty_amend: model picked amend_* but provided no mutating
    #     fields (IQST Series D / P-007 — actually a create_preferred).
    #   - bad_date: a required date arg didn't parse even after the
    #     normalizer's strip + first-ISO fallback (CGEN ATM drawdown
    #     "In January and February 2025" → no single ISO date).
    # Feed the failures back with a focused prompt; the model either
    # fixes them or emits note_no_event.
    if failures:
        retry_calls = await _retry_failed_calls(
            client=client, accession=accession,
            provider_tools=provider_tools,
            user_prompt=user_prompt, failures=failures,
        )
        if retry_calls:
            retry_typed = parse_tool_calls(
                retry_calls, accession=accession,
            )
            typed.extend(retry_typed)
            _log_unresolved_empty_amends(
                failures, retry_typed, accession=accession,
            )
        elif any(f.kind == "empty_amend" for f in failures):
            for f in failures:
                if f.kind == "empty_amend":
                    log.warning(
                        "walker %s — empty_amend retry returned no calls; "
                        "%s(%s) dropped",
                        accession, f.tool_name, f.instrument_id,
                    )

    notes = [m for m in typed if isinstance(m, NoteNoEvent)]
    real = [m for m in typed if not isinstance(m, NoteNoEvent)]

    # must_record net, two triggers feeding ONE second look:
    #  (a) metadata (8-K only): classify_8k flagged SEC item/exhibit
    #      metadata (Item 3.02/2.03/5.03 or EX-3.x/4.x) asserting a
    #      dilutive instrument, yet the first pass recorded nothing.
    #  (b) content (8-K + 6-K, round-4 extension): the filing's own text
    #      announces a transaction (high-precision patterns in
    #      item_classification.expected_call_classes) and the response
    #      carries NO matching call — catches both the 6-K hole (FPIs
    #      have no item codes) and thin-but-not-silent responses, which
    #      are the dominant walk-to-walk emission-variance class (SCNI's
    #      Apr-2026 6-K: exercise recorded, three warrant creates
    #      skipped).
    # Neither forces a create — a re-disclosure of an already-tracked
    # instrument legitimately returns note_no_event, and the hint says
    # so. Exactly one re-ask per filing.
    from .item_classification import expected_call_classes
    expected = expected_call_classes(filing_text, form)
    missing_cls = {
        cls: ev for cls, ev in expected.items()
        if not any(_covers_expected_class(cls, m) for m in real)
    }
    if (must_record and not real) or missing_cls:
        mr_tools = [
            build_provider_schema(t, provider=config.LLM_PROVIDER)
            for t in full_tool_set
        ]
        if missing_cls:
            instruction = _build_expected_class_instruction(
                missing_cls, real)
            trigger = f"content:{','.join(sorted(missing_cls))}"
            if must_record and not real:
                trigger = f"{must_record_reason}; {trigger}"
        else:
            instruction = None  # _retry_must_record builds the classic one
            trigger = must_record_reason
        recovered = await _retry_must_record(
            client=client, accession=accession, provider_tools=mr_tools,
            user_prompt=user_prompt, reason=trigger,
            instruction=instruction,
        )
        rec_typed = (parse_tool_calls(recovered, accession=accession)
                     if recovered else [])
        rec_real = [m for m in rec_typed if not isinstance(m, NoteNoEvent)]
        if not real:
            # Silent first pass: the recovery is the whole answer.
            if rec_real:
                log.info(
                    "walker %s — must_record (%s): second look recovered "
                    "%d instrument(s)/event(s)",
                    accession, trigger, len(rec_real),
                )
                typed = rec_typed
                notes = [m for m in typed if isinstance(m, NoteNoEvent)]
                real = rec_real
            else:
                log.warning(
                    "walker %s — must_record (%s): recorded nothing after "
                    "second look — possible missed instrument",
                    accession, trigger,
                )
        else:
            # Thin first pass: keep what was recorded; accept ONLY calls
            # that fill the named gaps. The stateless re-ask cannot see
            # the first response, so it may re-emit calls already made —
            # a re-emitted create is absorbed by apply-time dedup, but a
            # re-emitted record_event would double-apply, hence the
            # strict gap filter.
            gap_fills = [
                m for m in rec_real
                if any(_covers_expected_class(c, m) for c in missing_cls)
            ]
            if gap_fills:
                log.info(
                    "walker %s — content-expectation second look (%s): "
                    "recovered %d gap call(s)",
                    accession, trigger, len(gap_fills),
                )
                typed = typed + gap_fills
                real = real + gap_fills
            else:
                log.info(
                    "walker %s — content-expectation second look (%s): "
                    "no gap calls returned (re-disclosure or model "
                    "declined)", accession, trigger,
                )

    if notes:
        log.info("walker %s — note_no_event reasons: %s",
                 accession, [n.reason for n in notes])
    for m in real:
        log.info("walker %s — call %s", accession, fmt_mutation(m))
    mlist = MutationList(mutations=real)

    mlist = _propagate_banker(mlist, accession=accession)
    if active_rows is not None:
        mlist = _apply_guards(
            mlist, active_rows=active_rows,
            filing_date=filing_date, accession=accession,
            form=form,
        )
    return mlist


# ─── Retry path for recoverable parse failures ──────────────────────


def _log_unresolved_empty_amends(
    failures: list[RetryableFailure],
    retry_typed: list[Mutation],
    *, accession: str,
) -> None:
    """After a retry, warn for every `empty_amend` failure that has no
    plausible recovery in the retry calls.

    The retry prompt offers the model several escape hatches besides
    "fill in the amend fields" (see _build_retry_instruction): emit a
    record_* for a partial event (f), close_instrument for a lifecycle
    end (d), apply_split for an entity-wide event (e), or a sibling
    create_* if the amend was misclassified (b). Treat each of those
    as a successful recovery for the failure — only warn when the
    model walked away with nothing applicable.
    """
    amended_ids: set[str] = set()
    recorded_ids: set[str] = set()
    closed_ids: set[str] = set()
    created_families: set[str] = set()
    has_split = False
    for m in retry_typed:
        if isinstance(m, AmendMutation):
            iid = getattr(m, "instrument_id", None)
            if iid and (m.field_updates or m.outstanding_updates):
                amended_ids.add(iid)
        elif isinstance(m, RecordMutation):
            iid = getattr(m, "instrument_id", None)
            if iid:
                recorded_ids.add(iid)
        elif isinstance(m, CloseInstrument):
            if m.instrument_id:
                closed_ids.add(m.instrument_id)
        elif isinstance(m, CreateMutation):
            fam = getattr(m, "instrument_type", None)
            if fam:
                created_families.add(fam)
        elif isinstance(m, ApplySplit):
            has_split = True
    has_note = any(isinstance(m, NoteNoEvent) for m in retry_typed)
    for f in failures:
        if f.kind != "empty_amend":
            continue
        if f.instrument_id in amended_ids:
            continue
        if f.instrument_id in recorded_ids:
            continue
        if f.instrument_id in closed_ids:
            continue
        if has_split:
            continue
        if has_note:
            continue
        family = f.tool_name.removeprefix("amend_") if f.tool_name else ""
        if family and family in created_families:
            continue
        log.warning(
            "walker %s — empty_amend retry did NOT re-emit %s(%s); "
            "model returned %d sibling call(s) instead. Amend intent "
            "lost.",
            accession, f.tool_name, f.instrument_id, len(retry_typed),
        )


def _format_failure(f: RetryableFailure) -> str:
    if f.kind == "empty_amend":
        return (
            f"- {f.tool_name}(instrument_id={f.instrument_id!r}, "
            f"event_date={f.event_date!r}) — REJECTED: no mutating "
            f"fields provided."
        )
    if f.kind == "bad_date":
        return (
            f"- {f.tool_name}(...) — REJECTED: event_date "
            f"{f.event_date!r} could not be parsed as a single "
            f"YYYY-MM-DD."
        )
    return f"- {f.tool_name}(...) — REJECTED: {f.error_message}"


def _build_retry_instruction(failures: list[RetryableFailure]) -> str:
    """Compose a focused retry instruction. Branches the guidance
    paragraph by failure kind so the model gets the right nudge."""
    has_empty = any(f.kind == "empty_amend" for f in failures)
    has_bad_date = any(f.kind == "bad_date" for f in failures)
    has_missing_price = any(
        f.kind == "drawdown_missing_price" for f in failures
    )
    lines = [
        f"Your previous response included {len(failures)} tool call(s) "
        f"that were REJECTED. Re-emit ONLY the corrected call(s); do "
        f"not repeat tool calls that were already accepted.",
        "",
        "REJECTED CALLS:",
        *(_format_failure(f) for f in failures),
        "",
    ]
    if has_empty:
        lines += [
            "EMPTY AMEND CALLS — DEFAULT ACTION: re-emit the amend with "
            "fields filled in.",
            "",
            "You called amend_* with only instrument_id + event_date "
            "and no field arguments. The rejection is a "
            "signal to fill in the fields, NOT to walk away from the "
            "amend. Look at the filing text again — it almost "
            "always contains the numeric / textual change you intended "
            "to record. Then re-emit amend_*(<id>, "
            "<field>=<value>, ...) with the matching mutating field(s):",
            "  - amend_warrant: count, strike, maturity, "
            "known_owners, issue_date",
            "  - amend_convertible: principal_remaining, conv_price, "
            "maturity",
            "  - amend_preferred: count, conv_price, "
            "liquidation_preference, dividend_rate",
            "  - amend_s1_offering: anticipated_deal_size, "
            "warrant_strike, warrant_coverage_pct, final_deal_size, "
            "final_pricing, final_shares_offered, "
            "final_warrant_coverage_pct, placement_agent_canonical, "
            "sold_to_date",
            "  - amend_atm: capacity_usd, sales_amount_remaining, "
            "agreement_date, placement_agent_canonical",
            "  - amend_shelf: capacity_usd",
            "  - amend_equity_line: capacity_usd, agreement_date, "
            "agreement_end_date, max_drawdown_amount, drawdown_floor_pct",
            "  - amend_equity: known_owners",
            "",
            "IF THE FILING HAS BOTH AN AMEND AND NEW INSTRUMENTS (a "
            "priced S-1/A cover that updates the offering frame AND "
            "issues new warrant tranches; an 8-K that re-prices an "
            "existing warrant AND issues a side letter), emit BOTH — "
            "the filled amend PLUS the create_* siblings. Sibling "
            "creates do NOT substitute for the amend.",
            "",
            "ONLY IF the preferred action genuinely doesn't fit, choose "
            "ONE of the following alternatives (and do NOT mix several "
            "of them in place of the amend):",
            "  (b) If you picked amend_X by mistake and the filing "
            "actually creates a NEW instrument of type X (fresh "
            "Certificate of Designation, new SPA, new warrant tranche "
            "with terms not in any existing row), call the matching "
            "create_* instead.",
            "  (c) If the filing only re-mentions the existing "
            "instrument without changing anything, call note_no_event.",
            "  (d) If the filing reports a LIFECYCLE ENDING — warrant "
            "expired without exercise, ATM/EDA terminated, preferred "
            "or convertible fully redeemed for cash, instrument "
            "superseded by a successor — call close_instrument with "
            "reason='expired' / 'terminated' / 'redeemed' / "
            "'converted' / 'superseded'. Do NOT use amend_warrant "
            "count=0 or amend_atm remaining_capacity_usd=0 to express "
            "an ending — those are anchor reconciliations, not "
            "closures.",
            "  (e) If the filing announces a STOCK SPLIT, REVERSE "
            "SPLIT, or ADS-RATIO CHANGE, call apply_split ONCE — not "
            "amend_* per affected instrument.",
            "  (f) If the filing reports a PARTIAL EVENT against an "
            "existing instrument — drawdown, partial conversion, "
            "partial redemption, partial termination — call the "
            "matching record_* tool.",
            "",
        ]
    if has_bad_date:
        lines += [
            "For bad_date rejections, the event_date must be a SINGLE "
            "YYYY-MM-DD string. If the source quote spans a range "
            "(e.g. 'January and February 2025') or a quarter "
            "('Q1 2025'), pick the END of the disclosure window "
            "(last day of the latest month referenced). If the filing "
            "lists multiple distinct events, emit ONE tool call per "
            "event with each event's specific date.",
            "",
        ]
    if has_missing_price:
        lines += [
            "For drawdown_missing_price rejections, re-emit "
            "record_drawdown with price_per_share = the stated GROSS "
            "per-share offering price (the figure BEFORE the placement "
            "agent's commission). It is almost always quoted verbatim "
            "next to the share count — e.g. 'sold 414,785 shares at an "
            "average offering price of $10.74 per share for net "
            "proceeds of $4,320' → price_per_share=10.74. NEVER derive "
            "the price by dividing a 'net proceeds of $X' figure by the "
            "share count (that books a net price and understates the "
            "raise). ONLY if the filing truly states no per-share price "
            "at all may you pass the GROSS aggregate in "
            "drawdown_amount_usd — never the 'net proceeds after fees' "
            "number.",
            "",
        ]
    return "\n".join(lines)


async def _retry_failed_calls(
    *, client, accession: str,
    provider_tools,
    user_prompt: str, failures: list[RetryableFailure],
):
    """One follow-up LLM call covering all retryable failures from the
    first pass. Returns the retry's tool_calls (provider-normalized)
    or [] if the retry produced nothing usable.

    Stateless re-prompt: we don't carry the previous chat over because
    the provider wrappers don't support assistant/tool message roles
    cleanly across xai/moonshot/gemini. Instead we send the original
    user prompt plus a concise retry instruction in one new turn.
    """
    retry_instruction = _build_retry_instruction(failures)
    kinds = sorted({f.kind for f in failures})
    log.info(
        "walker %s — retrying %d call(s) (kinds=%s)",
        accession, len(failures), kinds,
    )
    chat = make_chat(
        client,
        tools=provider_tools,
        tool_choice="required",
        max_tokens=WALKER_MAX_TOKENS,
        seed=EXTRACT_SEED,
    )
    chat.append(system(SYSTEM_PROMPT))
    chat.append(user(user_prompt + "\n\n" + retry_instruction))
    response = await asample_and_check(
        chat, accession=accession, handler="ledger-walker-retry",
    )
    n_calls = len(response.tool_calls or [])
    log.info(
        "walker %s — retry finish=%r calls=%d",
        accession, response.finish_reason, n_calls,
    )
    return response.tool_calls or []


def _build_must_record_instruction(reason: str) -> str:
    """Focused second-look instruction for a must_record 8-K that came back
    silent. Nudges toward the missed instrument WITHOUT forcing a create — a
    re-disclosure of an already-tracked row stays note_no_event."""
    return "\n".join([
        "RE-EXAMINE — POSSIBLE MISSED INSTRUMENT.",
        "",
        f"This 8-K's SEC metadata signals that a dilutive instrument is "
        f"disclosed here ({reason}):",
        "  - Item 3.02 = an unregistered sale of equity (PIPE / private "
        "placement / registered direct).",
        "  - Item 2.03 = a new direct financial obligation (convertible note "
        "/ debenture).",
        "  - Item 5.03 or EX-3.x = a Certificate of Designation (a new "
        "preferred series).",
        "  - EX-4.x = a warrant or note instrument document is attached.",
        "",
        "Your previous response recorded NO instrument or event. Read the "
        "filing body and the named exhibit again, then decide:",
        "  - If it discloses an instrument NOT already in the ledger above, "
        "emit the matching create_* now (or record_*/amend_* if it is an "
        "event against an existing row).",
        "  - If the instrument is ALREADY tracked above (this 8-K only "
        "re-discloses or closes a previously reported financing), that is "
        "correct — call note_no_event(reason=...) and do NOT create a "
        "duplicate.",
        "Do not invent an instrument the filing does not support.",
    ])


def _covers_expected_class(cls: str, m) -> bool:
    """True when mutation `m` satisfies content-expectation class `cls`.
    Deliberately STRICT: an expected issuance is covered only by a
    create of that type (a record_exercise against the same type does
    NOT cover it — that's exactly the SCNI thin-response failure where
    the exercise was recorded and the new warrants were skipped)."""
    if cls == "close":
        return isinstance(m, CloseInstrument)
    if cls == "atm":
        return ((isinstance(m, CreateMutation)
                 and getattr(m, "type", None) == "atm")
                or isinstance(m, RestateAtm))
    return (isinstance(m, CreateMutation)
            and getattr(m, "type", None) == cls)


_EXPECTED_CLASS_TOOL = {
    "warrant": "create_warrant",
    "convertible": "create_convertible",
    "preferred": "create_preferred",
    "atm": "create_atm / restate_atm",
    "equity_line": "create_equity_line",
    "close": "close_instrument",
}


def _build_expected_class_instruction(
    missing: dict[str, str], recorded: list,
) -> str:
    """Focused second-look instruction naming the transaction(s) the
    filing's text announces but the response did not record. Lists the
    already-recorded calls so the stateless re-ask doesn't re-emit
    them. Never forces a create — re-disclosures stay note_no_event."""
    lines = [
        "RE-EXAMINE — DISCLOSED TRANSACTION WITH NO MATCHING CALL.",
        "",
        "This filing's own text announces the following transaction(s), "
        "but your response contained no matching call:",
    ]
    for cls, ev in sorted(missing.items()):
        lines.append(
            f"  - {_EXPECTED_CLASS_TOOL.get(cls, cls)}: the filing says "
            f"“…{ev}…”"
        )
    if recorded:
        lines += [
            "",
            "Already recorded from your previous response — these are "
            "SAVED, do NOT re-emit them:",
        ]
        for m in recorded[:10]:
            lines.append(f"  - {fmt_mutation(m)}")
    lines += [
        "",
        "For each transaction listed above: if it is a NEW instrument or "
        "event not in the ledger above, emit the matching call now. If "
        "it is ALREADY tracked above (a re-disclosure of an existing "
        "row), that is correct — call note_no_event naming the existing "
        "instrument_id instead. Do not invent terms the filing does not "
        "state.",
    ]
    return "\n".join(lines)


async def _retry_must_record(
    *, client, accession: str, provider_tools,
    user_prompt: str, reason: str,
    instruction: str | None = None,
):
    """Second look when the must_record net fires (metadata-silent or
    content-expectation gap). Stateless re-prompt (mirrors
    _retry_failed_calls). `instruction` overrides the classic silent-8-K
    text when the content path built a gap-specific one. Returns the
    retry's tool_calls (provider-normalized) or []."""
    if instruction is None:
        instruction = _build_must_record_instruction(reason)
    log.info(
        "walker %s — must_record net fired (%s); re-asking",
        accession, reason,
    )
    chat = make_chat(
        client,
        tools=provider_tools,
        tool_choice="required",
        max_tokens=WALKER_MAX_TOKENS,
        seed=EXTRACT_SEED,
    )
    chat.append(system(SYSTEM_PROMPT))
    chat.append(user(user_prompt + "\n\n" + instruction))
    response = await asample_and_check(
        chat, accession=accession, handler="ledger-walker-mustrecord",
    )
    log.info(
        "walker %s — must_record retry finish=%r calls=%d",
        accession, response.finish_reason, len(response.tool_calls or []),
    )
    return response.tool_calls or []


# ─── Post-parse guards ──────────────────────────────────────────────
# Two failure modes the walker prompt warns against but the LLM still
# occasionally emits. These run AFTER the LLM responds and BEFORE
# validate.py / apply_mutations, so we catch the bug regardless of
# prompt fidelity and log a metric for prompt regressions.

# strike-match tolerance: ±2% of the prompt's stated key
_STRIKE_TOL = 0.02
# created-date tolerance: ±30 days of THIS filing's date
_CREATED_TOL_DAYS = 30
# expiration-match tolerance for the dup-create guard. Same offering's
# S-1→S-1/A→424→8-K creates compute the same expiration to the day (or
# within a 1-2 day pricing-vs-closing slip); consecutive offerings at a
# coincidentally identical strike (XTIA June + September 2025 both
# struck at $2.00, expirations 2030-06-26 vs 2030-09-12) sit 60+ days
# apart and must stay distinct. 14d is generous to lifecycle slippage
# without admitting a separate offering.
_DUP_EXPIRATION_TOL_DAYS = 14
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

# Forms whose job is to enumerate ALL outstanding instruments. For
# these, the ±30-day created window doesn't apply — a 10-Q filed 90
# days after an 8-K still re-discloses every instrument the 8-K
# created, and the walker must see all of them as dedup candidates.
# Without this, post-30d periodic re-disclosures spawn duplicate
# create_* calls; the anchor reconciliation can flag the resulting
# duplicates but cannot merge them after the fact.
_PERIODIC_FORMS = frozenset({
    "10-K", "10-K/A", "10-Q", "10-Q/A",
    "20-F", "20-F/A", "40-F", "40-F/A",
    "6-K", "6-K/A",
})


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


def _strike_of_create(m) -> float | None:
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


def _split_adjusted_strike(row: dict,
                           as_of: date | None) -> float | None:
    """Project the row's current strike BACK to what it would have read
    at `as_of`, undoing any splits whose effective_date is strictly
    after as_of.

    Use when comparing a proposed create against an existing row: the
    create's strike quotes the filing's snapshot (which may pre-date a
    later split), while the ledger holds the latest split-adjusted
    strike. Without this projection, a periodic 10-Q that reports a
    pre-split price for an instrument the walker has since split-
    adjusted produces a 15× strike mismatch and the dedup guard
    spawns a duplicate row (the CETY C-099/C-101 pattern).

    Formula: pre_strike = post_strike × ratio. Holds for both
    directions because ratio = post/pre encodes the direction (reverse
    < 1, forward > 1) and post_strike = pre_strike / ratio always.
    """
    strike = _strike_of(row)
    if strike is None or as_of is None:
        return strike
    splits = (row.get("terms") or {}).get("applied_splits") or []
    for s in splits:
        s_date = _parse_iso(s.get("date") if isinstance(s, dict) else None)
        if s_date is None or s_date <= as_of:
            continue
        try:
            ratio = float(s.get("ratio"))
        except (TypeError, ValueError):
            continue
        if ratio == 0:
            continue
        strike = strike * ratio
    return strike


def _is_dup_create(m, active_rows: list[dict],
                   filing_d: date | None,
                   form: str | None = None) -> dict | None:
    """Return the matching ledger row if `m` re-discloses an existing
    instrument by (type, strike ±2%, same-or-unspecified series, created
    within the form's window of filing_date), else None. Picks the
    earliest-created on ties — same behavior the prompt prescribes.

    The created-date window matches _build_dedup_candidates_block (the
    prompt-side block this guard backstops):
      - closing-cue / S-1-family forms (8-K, 424B, S-1/A, F-1/A) → ±180d,
        covering signing→closing and the S-1→S-1/A→424B lifecycle;
      - other non-periodic forms → ±30d;
      - periodic filings (10-K/Q, 20-F, 40-F, 6-K) lift it entirely — a
        10-Q enumerates every outstanding instrument, so any active row
        is a legitimate dedup candidate regardless of age."""
    if m.type not in _PRICED_TYPES:
        return None
    if filing_d is None:
        return None
    m_strike = _strike_of_create(m)
    if m_strike is None:
        return None
    # The mutation's event_date is what the new disclosure says about
    # WHEN this instrument existed. We use it as the as-of point for
    # split projection: any split with effective_date after event_date
    # has NOT yet happened from the disclosure's perspective, so the
    # ledger's current (post-split) strike must be projected back to
    # compare against m_strike.
    m_event_date = getattr(m, "event_date", None)
    if isinstance(m_event_date, str):
        m_event_date = _parse_iso(m_event_date)
    skip_window = form in _PERIODIC_FORMS
    matches = []
    for row in active_rows:
        if row.get("type") != m.type:
            continue
        if (row.get("status") or "").startswith("superseded"):
            continue
        if row.get("status") not in (None, "active"):
            continue
        r_strike = _split_adjusted_strike(row, m_event_date)
        if r_strike is None:
            continue
        if not _strike_within(m_strike, r_strike):
            continue
        # Distinct series at the same strike are distinct instruments
        # (e.g. an Inducement Warrant and a Series B Warrant both struck
        # at $0.55 — SCNI 2026-04-23). Mirror the store's
        # _create_keys_match guard so the widened window below can't merge
        # them. One-sided (only one side names a series) falls through to
        # strike, preserving cross-filing re-disclosure collapse where a
        # later filing drops the series tag.
        m_series = warrant_series_key((m.terms or {}).get("series_letter"))
        r_series = warrant_series_key((row.get("terms") or {}).get("series_letter"))
        if m_series and r_series and m_series != r_series:
            continue
        # Distinct end-dates ⇒ distinct instruments at the same strike.
        # The 180d window for closing-cue forms (8-K/424) covers the
        # S-1→8-K lifecycle of ONE offering, where expirations match to
        # the day. It also reaches across to a SEPARATE offering 60-120d
        # later at a coincidentally identical strike (XTIA June + Sep
        # 2025 both at $2.00, expirations 2030-06-26 vs 2030-09-12),
        # which must not collapse — the second offering's creates would
        # vanish and the first row's exercisable_date would get smeared
        # by gap-fill. The end-date is maturity-or-expiration so the
        # veto covers CONVERTIBLES too: serial toxic issuers reissue to
        # the same lender at the same conv price every few weeks, and
        # conv-price-only matching merged CETY's Feb-2025 Mast Hill note
        # (maturity 2026-02-27) onto its Sep-2024 note (maturity
        # 2025-12-31) — silently dropping the new note's conversions.
        # One-sided cases (LLM omits the end-date on a true
        # re-disclosure's second mention) fall through to strike.
        m_end = _parse_iso((m.terms or {}).get("maturity")
                           or (m.terms or {}).get("expiration"))
        r_end = _parse_iso((row.get("terms") or {}).get("maturity")
                           or (row.get("terms") or {}).get("expiration"))
        if (m_end and r_end
                and abs((m_end - r_end).days) > _DUP_EXPIRATION_TOL_DAYS):
            continue
        r_created = _parse_iso(row.get("created_at"))
        if r_created is None:
            continue
        # Match the candidate-block window this guard backstops:
        # _build_dedup_candidates_block advertises 8-K / 424B / S-1-family
        # re-disclosures within ±180d (closing-cue + S-1 lifecycle), so
        # the guard must reach the same band — otherwise the prompt tells
        # the LLM "this 90-day-old row is a dup, amend it" while a
        # re-created duplicate slips past this final safety net (a 424B5
        # closing 60-120d after the announcement 8-K). Periodic forms
        # still lift the window entirely (skip_window).
        window = (_CLOSING_CANDIDATE_WINDOW_DAYS
                  if (form or "") in _WIDE_WINDOW_FORMS else _CREATED_TOL_DAYS)
        if not skip_window and abs((filing_d - r_created).days) > window:
            continue
        matches.append(row)
    if not matches:
        return None
    matches.sort(key=lambda r: r.get("created_at") or "")
    return matches[0]


def _shelf_within_window(m, active_rows: list[dict],
                         filing_d: date | None) -> dict | None:
    """For `create_equity`: return the active shelf-family row that
    should be carrying this issuance, or None."""
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


def _propagate_banker(mlist: MutationList, *, accession: str) -> MutationList:
    """When a single placement_agent_canonical appears anywhere in the
    batch, fill it onto every Create* mutation that left it null.

    Empirically the LLM emits the banker on the drawdown event of a
    424B5 but forgets to repeat it on the warrant create from the same
    filing — they share the offering, so they share the banker. We only
    propagate when exactly one banker name is present in the batch (the
    ambiguous multi-banker case is rare and safer to leave alone).

    `CreateShelf` doesn't carry placement_agent_canonical and is
    skipped. Typed dataclasses are frozen, so we substitute via
    `dataclasses.replace` rather than mutating in place.
    """
    bankers: set[str] = set()
    for m in mlist.mutations:
        if isinstance(m, CreateMutation):
            pa = getattr(m, "placement_agent_canonical", None)
            if pa:
                bankers.add(pa)
        elif isinstance(m, RecordMutation):
            pa = m.fields.get("placement_agent_canonical")
            if pa:
                bankers.add(pa)
    if len(bankers) != 1:
        return mlist
    banker = next(iter(bankers))
    new_mutations: list = []
    filled = 0
    for m in mlist.mutations:
        if (isinstance(m, CreateMutation)
                and not isinstance(m, CreateShelf)
                and not getattr(m, "placement_agent_canonical", None)):
            new_mutations.append(
                dataclasses.replace(m, placement_agent_canonical=banker)
            )
            filled += 1
        else:
            new_mutations.append(m)
    if filled:
        log.info("walker %s — propagated banker %r to %d create_instrument(s)",
                 accession, banker, filled)
    return MutationList(mutations=new_mutations)


_AMEND_FOR_TYPE = {
    "warrant": AmendWarrant,
    "convertible": AmendConvertible,
    "preferred": AmendPreferred,
}


# Date fields a deduped create can gap-fill onto the kept row, by type.
# Keys match BOTH the create's computed `.terms` (ISO strings — set by
# CreateWarrant._resolve_dates and the convertible/preferred _offset_date)
# AND the Amend* date-field names, so they pass straight through.
_DATE_GAPFILL_FIELDS = {
    "warrant": ("exercisable_date", "expiration"),
    "convertible": ("convertible_date", "maturity"),
    "preferred": ("convertible_date", "maturity"),
}


def _gap_fill_amend(m, dup: dict):
    """When a priced create is about to be dropped as a duplicate but it
    carries information the kept row LACKS, build an Amend* that gap-fills
    the existing row instead of losing it. Salvage:

      - computed dates — exercisable_date/expiration (warrant) or
        convertible_date/maturity (convertible/preferred). The store
        computes these from term structure, but if the dated disclosure
        arrives SECOND it gets deduped against an earlier dateless row,
        silently dropping the dates (GCTK Nov-2024 Series A: expiration
        2029-11-14 → None). Fold them onto the kept row.

    Gap-fill ONLY — never override a value the kept row already carries.
    Returns None when there's nothing to fill."""
    amend_cls = _AMEND_FOR_TYPE.get(m.type)
    if amend_cls is None:
        return None
    iid = dup.get("instrument_id")
    if not iid:
        return None
    dup_terms = dup.get("terms") or {}
    m_terms = m.terms  # computed view — dates already resolved
    kwargs: dict = {}

    # date fields — fill only those the kept row is missing.
    for fld in _DATE_GAPFILL_FIELDS.get(m.type, ()):
        new_iso = m_terms.get(fld)
        if new_iso and not dup_terms.get(fld):
            parsed = _parse_iso(new_iso)
            if parsed is not None:
                kwargs[fld] = parsed

    # known_owners — a later disclosure (20-F restate, resale recap) often
    # names the holders an earlier dateless/generic create lacked. Only
    # AmendWarrant carries known_owners; gap-fill ONLY when the kept row
    # has no owners yet (never clobber a recorded list). SCNI Dec-2023
    # W-3472: the 20-F re-create named ['Armistice','Sabby'] but was
    # deduped and the owners were lost.
    if m.type == "warrant":
        new_owners = m_terms.get("known_owners")
        if new_owners and not dup_terms.get("known_owners"):
            kwargs["known_owners"] = tuple(new_owners)

    if not kwargs:
        return None
    return amend_cls(
        instrument_id=iid,
        event_date=m.event_date,
        **kwargs,
    )


def _norm_series(v) -> str | None:
    """Normalized series_letter for matching: stripped + uppercased,
    or None when absent/blank."""
    return v.strip().upper() if isinstance(v, str) and v.strip() else None


def _is_placeholder_row(row: dict) -> bool:
    """True for an UNPRICED S-1/A warrant placeholder: strike exactly 0.0
    and NOT a pre-funded warrant. Pre-funded warrants carry a nominal
    near-zero strike (~$0.0001-0.01) and an is_pre_funded flag / 'Pre-Funded'
    label, and must never be treated as an unpriced placeholder."""
    if row.get("type") != "warrant" or _strike_of(row) != 0.0:
        return False
    terms = row.get("terms") or {}
    if terms.get("is_pre_funded"):
        return False
    if "pre-funded" in (row.get("label") or "").lower():
        return False
    return True


def _placeholder_finalize_match(m, active_rows: list[dict],
                                filing_d: date | None,
                                form: str | None = None):
    """Reconcile a warrant create against an existing same-series row when
    ONE side is an unpriced placeholder (strike 0.0) and the other is
    priced — the S-1/A→424B4 pricing transition the strike-keyed dedup
    can't bridge (0.0 never matches a priced strike, so the tranche
    fragments into two rows and the wrong one survives: GCTK Nov-2024
    Series A, placeholder exp 2029-11-04 surviving over priced 2029-11-14).

    Returns (row, direction):
      "finalize": m is PRICED, row is the placeholder → caller OVERWRITES
                  the row's strike + dates with m's priced values.
      "gapfill":  m is the placeholder, row is already PRICED → caller
                  only fills the row's null fields (never clobbers the
                  good priced strike back to 0.0) and drops the create.

    Tightly scoped to avoid over-merging distinct tranches: requires
    series_letter present AND equal on both sides, a ±30d created window
    (lifted for periodic forms), and a placement_agent negative guard.
    Pre-funded warrants are excluded on both sides."""
    if m.type != "warrant" or filing_d is None:
        return None
    m_strike = _strike_of_create(m)
    if m_strike is None:
        return None
    m_series = _norm_series(m.terms.get("series_letter"))
    if m_series is None:
        return None
    m_prefunded = (bool(getattr(m, "is_pre_funded", None))
                   or getattr(m, "descriptor", None) == "Pre-Funded"
                   or "pre-funded" in (getattr(m, "label", None) or "").lower())
    if m_prefunded:
        return None
    m_pa = getattr(m, "placement_agent_canonical", None)
    skip_window = form in _PERIODIC_FORMS
    matches: list[tuple[dict, str]] = []
    for row in active_rows:
        if row.get("type") != "warrant":
            continue
        if (row.get("status") or "active") not in (None, "active"):
            continue
        if _norm_series((row.get("terms") or {}).get("series_letter")) != m_series:
            continue
        r_created = _parse_iso(row.get("created_at"))
        if r_created is None:
            continue
        if not skip_window and abs((filing_d - r_created).days) > _CREATED_TOL_DAYS:
            continue
        r_pa = row.get("placement_agent_canonical")
        if m_pa and r_pa and m_pa != r_pa:
            continue
        row_ph = _is_placeholder_row(row)
        r_strike = _strike_of(row)
        if m_strike != 0.0 and row_ph:
            matches.append((row, "finalize"))
        elif (m_strike == 0.0 and not row_ph
              and r_strike is not None and r_strike != 0.0):
            matches.append((row, "gapfill"))
    if not matches:
        return None
    matches.sort(key=lambda rd: rd[0].get("created_at") or "")
    return matches[0]


def _finalize_amend(m, placeholder_row: dict):
    """Build an AmendWarrant that finalizes an unpriced placeholder row
    with the priced create's terms. Unlike _gap_fill_amend (fills nulls
    only), this OVERWRITES the placeholder strike (0.0 → priced) and the
    priced dates. Writes the create's raw as-of-filing strike — the row's
    existing applied_splits re-project it exactly as the priced sibling
    renders today."""
    iid = placeholder_row.get("instrument_id")
    if not iid:
        return None
    m_terms = m.terms
    kwargs: dict = {"strike": _strike_of_create(m)}
    for fld in ("exercisable_date", "expiration"):
        iso = m_terms.get(fld)
        parsed = _parse_iso(iso) if iso else None
        if parsed is not None:
            kwargs[fld] = parsed
    cnt = getattr(m, "count", None)
    if cnt:
        kwargs["count"] = cnt
    return AmendWarrant(
        instrument_id=iid,
        event_date=m.event_date,
        **kwargs,
    )


def _apply_guards(mlist: MutationList, *, active_rows: list[dict],
                  filing_date: str, accession: str,
                  form: str | None = None) -> MutationList:
    """Post-parse failsafes. Drops the two creates the prompt warns
    against but the LLM sometimes still emits.

    `form` controls dedup window scope — periodic filings lift the
    ±30d created-date check (see _is_dup_create)."""
    filing_d = _parse_iso(filing_date)
    if filing_d is None:
        return mlist

    kept: list[Mutation] = []
    dropped_dup = 0
    dropped_equity = 0
    gap_filled = 0
    finalized = 0
    for m in mlist.mutations:
        if isinstance(m, CreateMutation):
            if isinstance(m, CreateWarrant):
                pf = _placeholder_finalize_match(
                    m, active_rows, filing_d, form=form)
                if pf is not None:
                    row, direction = pf
                    amend = (_finalize_amend(m, row) if direction == "finalize"
                             else _gap_fill_amend(m, row))
                    if amend is not None:
                        kept.append(amend)
                        finalized += 1
                        log.info(
                            "walker %s — placeholder-%s warrant %s via amend: "
                            "%s (create reconciled)", accession, direction,
                            row.get("instrument_id"),
                            ", ".join(sorted(amend.field_updates)))
                    else:
                        log.info(
                            "walker %s — placeholder-%s warrant %s: create "
                            "dropped (no new fields)", accession, direction,
                            row.get("instrument_id"))
                    continue
            dup = _is_dup_create(m, active_rows, filing_d, form=form)
            if dup is not None:
                upgrade = _gap_fill_amend(m, dup)
                if upgrade is not None:
                    kept.append(upgrade)
                    gap_filled += 1
                    log.info(
                        "walker %s — dedup gap-fill %s %s via amend: %s "
                        "(create deduped, info preserved)",
                        accession, upgrade.type, dup.get("instrument_id"),
                        ", ".join(sorted(upgrade.field_updates)))
                else:
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
    if dropped_dup or dropped_equity or finalized:
        log.info(
            "walker %s — guards dropped %d dup-creates "
            "(%d gap-filled via amend), %d equity-on-shelf, "
            "%d placeholder-finalized",
            accession, dropped_dup, gap_filled, dropped_equity, finalized)
    return MutationList(mutations=kept)


# ─── Pre-call dedup candidates ──────────────────────────────────────
# Mirrors the post-parse guards' algorithm but runs BEFORE the LLM call.
# The block surfaces open ledger rows whose created_at falls within ±30d
# of the filing date and renders an explicit decision rubric next to
# each, so the LLM doesn't have to apply the DEDUP CHECK rules in its
# head while reading the filing body. _apply_guards stays as a final
# safety net for cases where the LLM ignores the block.

_CANDIDATE_WINDOW_DAYS = 30

# Closing-cue forms (8-K / 424B) carry "closing of previously announced"
# disclosures whose signing/announcement filing can be up to ~180 days
# earlier — see confirm_closing's ±180d match window (record.py). The
# default ±30d would hide that signing row from the candidate set,
# exactly the relabel case the block exists to support, leaving the LLM
# to scan the raw ledger. Widen the window for these forms so the
# signing row stays a candidate. (Periodic forms already show all rows.)
_CLOSING_CANDIDATE_WINDOW_DAYS = 180
_CLOSING_CUE_FORMS = frozenset({
    "8-K", "8-K/A", "424B3", "424B4", "424B5", "424B8",
})

# An offering's S-1 → S-1/A → 424B lifecycle spans weeks to months, so
# an amendment re-disclosing its parent s1_offering routinely sits
# outside ±30d (the GCTK S-1/A filed 38d after its S-1). Widen the
# window for the S-1 family too — same 180d — so the parent offering
# stays a candidate to AMEND rather than being re-created as a
# duplicate. (424B variants are already covered as closing-cue forms.)
_WIDE_WINDOW_FORMS = _CLOSING_CUE_FORMS | frozenset({
    "S-1", "S-1/A", "F-1", "F-1/A",
})


def _strike_band(strike: float, tol: float = _STRIKE_TOL) -> str:
    """±tol band rendered as 'low-high' for the LLM. Avoids scientific
    notation across the full strike range — pre-funded warrants
    ($0.0001), ordinary dollar strikes, and post-reverse-split outliers
    (the XTIA eval has a $146,250 strike) all read sensibly."""
    if strike == 0:
        return "exactly $0"
    low = strike * (1 - tol)
    high = strike * (1 + tol)
    if abs(strike) < 0.01:
        return f"${low:.6f}-${high:.6f}"
    if abs(strike) >= 1000:
        return f"${low:,.2f}-${high:,.2f}"
    return f"${low:.4f}-${high:.4f}"


def _format_candidate(row: dict) -> str:
    """One bullet describing a dedup candidate and its decision rubric.
    Returns '' for rows where dedup-block guidance is not meaningful
    (equity, or rows missing the matching key)."""
    iid = row.get("instrument_id") or "?"
    type_ = row.get("type") or "?"
    created = (row.get("created_at") or "")[:10]
    cp = row.get("counterparty_canonical") or "—"

    if type_ in _PRICED_TYPES:
        strike = _strike_of(row)
        if strike is None:
            return ""
        band = _strike_band(strike)
        field_name = "strike" if type_ == "warrant" else "conv_price"
        terms = row.get("terms") or {}
        end_field = "expiration" if type_ == "warrant" else "maturity"
        end_raw = terms.get("maturity") or terms.get("expiration")
        end_str = f"  {end_field}={str(end_raw)[:10]}" if end_raw else ""
        # The same-{end_field} qualifier mirrors _is_dup_create's
        # end-date veto: serial issuers reuse the same price on
        # genuinely new paper every few weeks (CETY's Mast Hill ladder),
        # and a price-band-only rubric would tell the LLM to merge them.
        same_end = (
            f" AND the same {end_field} (±2 weeks)" if end_raw else ""
        )
        new_if_diff_end = (
            f"A different {end_field} (beyond ~2 weeks) means a NEW "
            f"{type_} — create it. " if end_raw else ""
        )
        return (
            f"- {iid}  {type_}  {field_name}=${strike:g}  "
            f"created={created}{end_str}  counterparty={cp}\n"
            f"  → If this filing describes a {type_} with {field_name} "
            f"in {band}{same_end}, it IS {iid}. Do NOT call "
            f"create_{type_}. {new_if_diff_end}"
            f"If THIS filing CLOSES the previously-announced tranche "
            f"('closed / consummated / completed the closing of / issued "
            f"the Units described in the [earlier] 6-K / 8-K'), call "
            f"confirm_closing({iid}, closing_date=<this filing date>) — "
            f"this rebases issue_date and slides expiration so the card "
            f"relabels by closing month. Otherwise call amend_{type_}"
            f"({iid}) only if a term genuinely changed."
        )

    if type_ in ("shelf", "atm", "equity_line"):
        terms = row.get("terms") or {}
        capacity = terms.get("capacity_usd")
        cap_str = f"${capacity:,.0f}" if capacity else "—"
        return (
            f"- {iid}  {type_}  capacity={cap_str}  "
            f"created={created}\n"
            f"  → If this filing describes a takedown/drawdown/sale "
            f"under this {type_}, call record_drawdown against "
            f"{iid}. Do NOT call create_{type_}."
        )

    if type_ == "s1_offering":
        return (
            f"- {iid}  s1_offering  created={created}  counterparty={cp}\n"
            f"  → A later filing in this offering's S-1 → S-1/A → 424B "
            f"chain re-discloses {iid}; do NOT call create_s1_offering "
            f"(the offering keeps its original S-1 date). An S-1/A with "
            f"amended terms → amend_s1_offering({iid}); a 424B that prices "
            f"the deal → amend_s1_offering({iid}) with final_* terms AND "
            f"record_drawdown against {iid}."
        )

    return ""


def _build_dedup_candidates_block(
    active_rows: list[dict] | None, filing_date: str,
    *, accession: str | None = None,
    window_days: int = _CANDIDATE_WINDOW_DAYS,
    form: str | None = None,
) -> str:
    """Body of the Dedup candidates block — '' when no candidates.

    Caller wraps the returned string with the section heading. Only
    active rows are considered; superseded/closed rows are filtered.

    For periodic filings (10-K/Q, 20-F, 40-F, 6-K) the ±window_days
    filter is lifted — these filings re-disclose every outstanding
    instrument regardless of age, so any active row is a legitimate
    dedup candidate.
    """
    filing_d = _parse_iso(filing_date)
    if filing_d is None:
        return ""

    is_periodic = form in _PERIODIC_FORMS
    # Closing-cue forms re-disclose tranches signed/announced up to
    # ~180d earlier, and S-1-family amendments re-disclose their parent
    # offering across a months-long chain; widen so the older row stays
    # a candidate.
    eff_window = (_CLOSING_CANDIDATE_WINDOW_DAYS
                  if (form or "") in _WIDE_WINDOW_FORMS else window_days)
    lines: list[str] = []
    for row in active_rows or []:
        if (row.get("status") or "active") != "active":
            continue
        r_created = _parse_iso(row.get("created_at"))
        if r_created is None:
            continue
        if not is_periodic and abs((filing_d - r_created).days) > eff_window:
            continue
        rendered = _format_candidate(row)
        if rendered:
            lines.append(rendered)

    if not lines:
        return ""

    if accession:
        log.info(
            "walker %s — dedup block: %d candidates%s",
            accession, len(lines),
            " (all active, periodic form)" if is_periodic
            else f" within ±{eff_window}d",
        )

    if is_periodic:
        intro = (
            f"Every open ledger row, as of this periodic filing "
            f"({form} dated {filing_date}). The filing's overhang table "
            f"is expected to re-disclose all of these — match each "
            f"disclosure to its existing row using the decision rubric "
            f"below, and do NOT call create_* for any instrument that "
            f"appears here.\n\n"
        )
    else:
        intro = (
            f"Open ledger rows created within ±{eff_window} days of this "
            f"filing's date ({filing_date}). These are the highest-risk dedup "
            f"matches — apply the decision rubric next to each row before "
            f"calling any create_* tool.\n\n"
        )
    return intro + "\n".join(lines) + "\n"


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
