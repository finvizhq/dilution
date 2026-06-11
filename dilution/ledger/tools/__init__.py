"""Tool-calling surface for the ledger walker.

Each Tool defines one mutation the walker can emit, with required
arguments enforced at LLM decode time (xAI strict tools; OpenAI strict
function calling; Gemini json_schema function calling). The Tool
definitions are the source of truth — provider-specific schemas, the
in-memory mutation dataclasses, and the parse-and-validate logic are
all derived from them.

Public surface:
  - Tool, ToolArg               — definitions (re-exported from ._base)
  - build_provider_schema(tool) — provider-specific Tool / function spec
  - ALL_TOOLS                   — registry of every defined tool
  - tools_for_form(form)        — tool subset for a given filing form
  - parse_tool_calls(calls)     — provider tool calls → list[Mutation]
"""

from ._base import Tool, ToolArg, build_provider_schema
from . import create as _create
from . import amend as _amend
from . import record as _record
from .parse import EmptyAmendFailure, RetryableFailure, parse_tool_calls


ALL_TOOLS: dict[str, Tool] = {
    t.name: t for t in (
        # create_*
        _create.create_atm,
        _create.create_shelf,
        _create.create_warrant,
        _create.create_convertible,
        _create.create_preferred,
        _create.create_equity_line,
        _create.create_s1_offering,
        _create.create_equity,
        # restate (ATM amendment → new card)
        _amend.restate_atm,
        # amend_*
        _amend.amend_atm,
        _amend.amend_equity_line,
        _amend.amend_shelf,
        _amend.amend_warrant,
        _amend.amend_convertible,
        _amend.amend_preferred,
        _amend.amend_s1_offering,
        _amend.amend_equity,
        # record_*
        _record.record_exercise,
        _record.record_conversion,
        _record.record_drawdown,
        _record.record_partial_redemption,
        _record.record_partial_termination,
        _record.confirm_closing,
        # misc
        _record.close_instrument,
        _record.apply_split,
        _record.note_no_event,
    )
}


_NOTE = _record.note_no_event


def tools_for_form(form: str) -> list[Tool]:
    """Return the subset of tools applicable to a given filing form.

    Membership is driven by each tool's `valid_forms` field. The safety
    valve `note_no_event` is appended to every form's list (under
    tool_choice='required' the model needs an out when nothing else
    semantically applies). Unknown / not-yet-mapped forms return just
    `[note_no_event]` — the walker logs that case and treats the
    filing as a no-op.

    /A fallback: SEC amendment forms (e.g. 20-F/A, 6-K/A, 10-K/A) carry
    the same kinds of disclosures as their parent form. When no tool's
    valid_forms includes the /A variant, we retry with the parent form
    stripped, so amendments inherit the parent's full disclosure
    surface (not just note_no_event).
    """
    matched = [t for t in ALL_TOOLS.values()
               if form in t.valid_forms]
    if not matched and form.endswith("/A"):
        parent = form[:-2]
        matched = [t for t in ALL_TOOLS.values()
                   if parent in t.valid_forms]
    # note_no_event is intentionally not in any tool's valid_forms set
    # — it's universal, appended here.
    matched.append(_NOTE)
    return matched


# Pre-computed map for the well-known forms. The walker can consult
# either this map or call tools_for_form() lazily; the map is provided
# for visibility / testability.
TOOLS_FOR_FORM: dict[str, list[Tool]] = {
    form: tools_for_form(form)
    for form in (
        # Current reports
        "8-K", "8-K/A",
        # Periodic (incl. amendments — FPI 20-F/A and 6-K/A inherit
        # via the /A fallback in tools_for_form; 40-F is the Canadian
        # MJDS annual report, analog of 20-F)
        "10-Q", "10-Q/A", "10-K", "10-K/A",
        "20-F", "20-F/A", "6-K", "6-K/A",
        "40-F", "40-F/A",
        # Registration statements
        "S-1", "S-1/A", "F-1", "F-1/A",
        "S-1MEF", "F-1MEF",
        "S-3", "S-3/A", "S-3ASR", "S-3MEF",
        "F-3", "F-3/A", "F-3ASR", "F-3MEF",
        # F-10 / F-10EF — Canadian MJDS shelf (analog of F-3 / F-3ASR)
        "F-10", "F-10/A", "F-10EF",
        # Prospectus supplements
        "424B3", "424B4", "424B5", "424B8",
        # Canadian MJDS prospectus supplement (analog of 424B5)
        "SUPPL",
        # Amendments and misc (DEFM14A is the merger-specific definitive
        # proxy — currently routes through apply_split only, broader
        # merger-consideration extraction is a separate scope)
        "POS AM", "PRE 14A", "DEF 14A", "DEFA14A", "DEFM14A",
        "EFFECT", "RW",
    )
}


__all__ = [
    "Tool", "ToolArg",
    "build_provider_schema",
    "ALL_TOOLS", "TOOLS_FOR_FORM", "tools_for_form",
    "parse_tool_calls", "RetryableFailure", "EmptyAmendFailure",
]
