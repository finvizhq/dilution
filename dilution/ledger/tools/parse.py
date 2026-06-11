"""Tool-call → mutation dispatch.

The LLM returns provider-normalized ToolCall objects (id, name,
arguments dict). This module dispatches each to the appropriate
typed dataclass in mutations.py.

The decoder already enforced types + required fields + patterns at
sample time. The builders here:
  - parse ISO date strings into datetime.date
  - apply cross-arg validation (e.g. ≥1 field set on amends — added
    in fan-out; not needed for create_atm / create_shelf)
  - log + drop malformed calls (should be rare since schema enforces)

Unknown tool names are logged + dropped (not raised) so a typo on the
LLM side doesn't crash the walker.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Callable

from ..mutations import (
    CreateAtm, CreateShelf,
    CreateWarrant, CreateConvertible, CreatePreferred,
    CreateEquityLine, CreateS1Offering, CreateEquity,
    RestateAtm,
    AmendAtm, AmendShelf, AmendEquityLine,
    AmendWarrant, AmendConvertible, AmendPreferred,
    AmendS1Offering, AmendEquity,
    RecordExercise, RecordConversion, RecordDrawdown,
    RecordPartialRedemption, RecordPartialTermination,
    ConfirmClosing,
    CloseInstrument, ApplySplit, NoteNoEvent,
    ToolMutation,
)


log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetryableFailure:
    """A tool call that failed validation in a way the walker can
    feasibly recover by re-prompting the LLM.

    Two flavours, distinguished by `kind`:
      - "empty_amend": amend_* call with instrument_id + event_date but
        no mutating fields. The model probably picked the wrong tool
        or missed the field changes. (IQST Series D / P-007.)
      - "bad_date": a required date arg failed to parse even after the
        normalizer's strip + first-ISO-date fallback. The source quote
        is usually a multi-month phrase ("January and February 2025")
        or a relative reference ("Q1 2025"). (CGEN ATM-011 $8.87M
        drawdown.)
      - "drawdown_missing_price": record_drawdown gave drawdown_shares
        but no price_per_share, so gross proceeds can't be computed and
        the model likely routed a net-of-fees aggregate into
        drawdown_amount_usd. Re-ask for the per-share GROSS offering
        price. (GCTK Q2 ATM: net $4.32M booked vs gross $4.4549M.)

    Other parse failures (missing required fields, type errors on
    numeric args) are not retried — they tend to be deeper extraction
    bugs that a single follow-up turn rarely fixes.
    """
    kind: str
    tool_name: str
    instrument_id: str
    event_date: str | None
    error_message: str = ""


# Backwards-compatible alias so callers that imported EmptyAmendFailure
# in the previous patch keep working. Will be retired once nothing
# else references it.
EmptyAmendFailure = RetryableFailure


class _DateParseError(ValueError):
    """Raised when a required date arg can't be normalized into ISO."""


# ─── Builders ─────────────────────────────────────────────────────────

def _opt_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _opt_str(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _opt_int(v) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _opt_bool(v) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v)


# Gemini frequently copies the trailing comma from sentences like
# "On January 14, 2025, we issued..." into the JSON date field, emitting
# "2025-01-14,". It also occasionally emits multi-date strings like
# "2025-02-28,2025-01-01" when the source quote spans a date range
# ("In January and February 2025, ..."). Both are cheaper to recover
# here than to re-prompt the model.
#
# Strategy:
#   1. Strip leading/trailing whitespace + punctuation.
#   2. If the result still isn't a clean YYYY-MM-DD, extract the FIRST
#      YYYY-MM-DD substring. Picking the first is a heuristic — for the
#      observed CGEN multi-date case ("2025-02-28,2025-01-01") it lands
#      on the later disclosure-window date, which matches the
#      end-of-window convention disclosures use anyway.
_DATE_STRIP_CHARS = " \t\n\r,;.\"'"
_ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _normalize_date(raw) -> str | None:
    if not raw or not isinstance(raw, str):
        return raw if raw else None
    stripped = raw.strip(_DATE_STRIP_CHARS)
    if not stripped:
        return None
    # Fast path: already a clean YYYY-MM-DD.
    if len(stripped) == 10 and stripped[4] == "-" and stripped[7] == "-":
        return stripped
    # Fallback: pull the first ISO date out of a multi-date or noisy
    # string. Logged at INFO so we can spot how often Gemini does this.
    m = _ISO_DATE_RE.search(stripped)
    if m:
        extracted = m.group(1)
        log.info("date normalizer: extracted %r from %r",
                 extracted, raw)
        return extracted
    return stripped or None


def _opt_date(args: dict, key: str) -> date | None:
    raw = _normalize_date(args.get(key))
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except (TypeError, ValueError):
        return None


def _opt_str_tuple(v) -> tuple[str, ...] | None:
    if v is None:
        return None
    if not isinstance(v, (list, tuple)):
        return None
    out = tuple(str(x).strip() for x in v if str(x).strip())
    return out or None


def _cp_from_known_owners(known: tuple[str, ...] | None) -> str | None:
    """Single-owner deals collapse to a scalar counterparty so labels
    ('Streeterville Note'), the investor-class quality tag, and the
    warrant-collapse bucket key all keep working. Multi-investor PIPEs
    leave counterparty_canonical null — the card's known_owners list
    carries the full set."""
    if known is not None and len(known) == 1:
        return known[0]
    return None


def _required_date(args: dict, key: str) -> date:
    """Required date arg. The decoder enforces the YYYY-MM-DD pattern,
    so this should normally succeed; raises _DateParseError on the
    rare miss so the caller can capture it as a retryable failure."""
    original = args.get(key)
    raw = _normalize_date(original)
    if not raw:
        raise _DateParseError(
            f"missing required date arg {key!r} (got {original!r})"
        )
    try:
        return date.fromisoformat(raw)
    except (TypeError, ValueError) as e:
        raise _DateParseError(
            f"invalid ISO date for {key!r}: {original!r} "
            f"(normalized to {raw!r}): {e}"
        ) from e


def _build_create_atm(args: dict) -> CreateAtm:
    return CreateAtm(
        capacity_usd=float(args["capacity_usd"]),
        agreement_date=_required_date(args, "agreement_date"),
        agreement_end_date=_opt_date(args, "agreement_end_date"),
        placement_agent_canonical=str(args["placement_agent_canonical"]).strip(),
        event_date=_required_date(args, "event_date"),
        remaining_capacity_usd=_opt_float(args.get("remaining_capacity_usd")),
        drawn_usd=_opt_float(args.get("drawn_usd")),
        proposed_id=args.get("proposed_id"),
    )


def _build_create_shelf(args: dict) -> CreateShelf:
    return CreateShelf(
        capacity_usd=float(args["capacity_usd"]),
        form=str(args["form"]).strip(),
        event_date=_required_date(args, "event_date"),
        remaining_capacity_usd=_opt_float(args.get("remaining_capacity_usd")),
        proposed_id=args.get("proposed_id"),
    )


def _build_create_warrant(args: dict) -> CreateWarrant:
    known = _opt_str_tuple(args.get("known_owners"))
    return CreateWarrant(
        count=float(args["count"]),
        strike=float(args["strike"]),
        event_date=_required_date(args, "event_date"),
        exercisable_date=_opt_date(args, "exercisable_date"),
        expiration=_opt_date(args, "expiration"),
        term_months=_opt_int(args.get("term_months")),
        exercise_offset_months=_opt_int(args.get("exercise_offset_months")),
        term_anchor=_opt_str(args.get("term_anchor")),
        is_pre_funded=_opt_bool(args.get("is_pre_funded")),
        units=_opt_str(args.get("units")),
        series_letter=_opt_str(args.get("series_letter")),
        counterparty_canonical=_cp_from_known_owners(known),
        placement_agent_canonical=_opt_str(args.get("placement_agent_canonical")),
        descriptor=_opt_str(args.get("descriptor")),
        known_owners=known,
        proposed_id=args.get("proposed_id"),
    )


def _build_create_convertible(args: dict) -> CreateConvertible:
    known = _opt_str_tuple(args.get("known_owners"))
    return CreateConvertible(
        principal=float(args["principal"]),
        principal_remaining=float(args["principal_remaining"]),
        event_date=_required_date(args, "event_date"),
        rate=_opt_float(args.get("rate")),
        conv_price=_opt_float(args.get("conv_price")),
        conv_discount_pct=_opt_float(args.get("conv_discount_pct")),
        convertible_date=_opt_date(args, "convertible_date"),
        maturity=_opt_date(args, "maturity"),
        convertible_offset_months=_opt_int(args.get("convertible_offset_months")),
        maturity_months=_opt_int(args.get("maturity_months")),
        oid_pct=_opt_float(args.get("oid_pct")),
        counterparty_canonical=_cp_from_known_owners(known),
        placement_agent_canonical=_opt_str(args.get("placement_agent_canonical")),
        descriptor=_opt_str(args.get("descriptor")),
        known_owners=known,
        proposed_id=args.get("proposed_id"),
    )


def _build_create_preferred(args: dict) -> CreatePreferred:
    known = _opt_str_tuple(args.get("known_owners"))
    return CreatePreferred(
        count=float(args["count"]),
        series_letter=str(args["series_letter"]).strip(),
        event_date=_required_date(args, "event_date"),
        conv_price=_opt_float(args.get("conv_price")),
        conversion_ratio=_opt_float(args.get("conversion_ratio")),
        convertible_date=_opt_date(args, "convertible_date"),
        maturity=_opt_date(args, "maturity"),
        convertible_offset_months=_opt_int(args.get("convertible_offset_months")),
        maturity_months=_opt_int(args.get("maturity_months")),
        stated_value=_opt_float(args.get("stated_value")),
        liquidation_preference=_opt_float(args.get("liquidation_preference")),
        dividend_rate=_opt_float(args.get("dividend_rate")),
        principal_remaining=_opt_float(args.get("principal_remaining")),
        counterparty_canonical=_cp_from_known_owners(known),
        placement_agent_canonical=_opt_str(args.get("placement_agent_canonical")),
        descriptor=_opt_str(args.get("descriptor")),
        known_owners=known,
        proposed_id=args.get("proposed_id"),
    )


def _build_create_equity_line(args: dict) -> CreateEquityLine:
    return CreateEquityLine(
        capacity_usd=float(args["capacity_usd"]),
        agreement_date=_required_date(args, "agreement_date"),
        agreement_end_date=_opt_date(args, "agreement_end_date"),
        term_months=_opt_int(args.get("term_months")),
        counterparty_canonical=str(args["counterparty_canonical"]).strip(),
        event_date=_required_date(args, "event_date"),
        remaining_capacity_usd=_opt_float(args.get("remaining_capacity_usd")),
        drawn_usd=_opt_float(args.get("drawn_usd")),
        placement_agent_canonical=_opt_str(args.get("placement_agent_canonical")),
        descriptor=_opt_str(args.get("descriptor")),
        proposed_id=args.get("proposed_id"),
    )


def _build_create_s1_offering(args: dict) -> CreateS1Offering:
    return CreateS1Offering(
        anticipated_deal_size=float(args["anticipated_deal_size"]),
        event_date=_required_date(args, "event_date"),
        warrant_strike=_opt_float(args.get("warrant_strike")),
        warrant_coverage_pct=_opt_float(args.get("warrant_coverage_pct")),
        sold_to_date=_opt_float(args.get("sold_to_date")),
        placement_agent_canonical=_opt_str(args.get("placement_agent_canonical")),
        proposed_id=args.get("proposed_id"),
    )


def _build_create_equity(args: dict) -> CreateEquity:
    known = _opt_str_tuple(args.get("known_owners"))
    return CreateEquity(
        count=float(args["count"]),
        price_per_share=float(args["price_per_share"]),
        event_date=_required_date(args, "event_date"),
        closing_date=_opt_date(args, "closing_date"),
        counterparty_canonical=_cp_from_known_owners(known),
        placement_agent_canonical=_opt_str(args.get("placement_agent_canonical")),
        descriptor=_opt_str(args.get("descriptor")),
        known_owners=known,
        proposed_id=args.get("proposed_id"),
    )


# ─── Restate builder ──────────────────────────────────────────────────


def _build_restate_atm(args: dict) -> RestateAtm:
    return RestateAtm(
        predecessor_id=str(args["predecessor_id"]).strip(),
        capacity_usd=float(args["capacity_usd"]),
        agreement_date=_required_date(args, "agreement_date"),
        agreement_end_date=_opt_date(args, "agreement_end_date"),
        placement_agent_canonical=str(args["placement_agent_canonical"]).strip(),
        supersede_prior=bool(_opt_bool(args.get("supersede_prior"))),
        event_date=_required_date(args, "event_date"),
        remaining_capacity_usd=_opt_float(args.get("remaining_capacity_usd")),
        proposed_id=args.get("proposed_id"),
    )


# ─── Amend builders ───────────────────────────────────────────────────


class _AmendValidationError(ValueError):
    """Raised when an amend tool emits no mutating fields."""


def _check_amend_non_empty(m, mutating_field_count: int) -> None:
    """Mirror of the legacy AmendInstrument._check_non_empty rule."""
    if mutating_field_count == 0:
        raise _AmendValidationError(
            f"{type(m).__name__} requires at least one mutating field "
            f"to be set (instrument_id={m.instrument_id!r})"
        )


def _build_amend_atm(args: dict) -> AmendAtm:
    m = AmendAtm(
        instrument_id=str(args["instrument_id"]).strip(),
        event_date=_required_date(args, "event_date"),
        capacity_usd=_opt_float(args.get("capacity_usd")),
        remaining_capacity_usd=_opt_float(args.get("remaining_capacity_usd")),
        drawn_usd=_opt_float(args.get("drawn_usd")),
        placement_agent_canonical=_opt_str(args.get("placement_agent_canonical")),
        agreement_date=_opt_date(args, "agreement_date"),
        agreement_end_date=_opt_date(args, "agreement_end_date"),
    )
    _check_amend_non_empty(
        m,
        sum(1 for v in (
            m.capacity_usd, m.remaining_capacity_usd, m.drawn_usd,
            m.placement_agent_canonical, m.agreement_date,
            m.agreement_end_date,
        ) if v is not None),
    )
    return m


def _build_amend_equity_line(args: dict) -> AmendEquityLine:
    m = AmendEquityLine(
        instrument_id=str(args["instrument_id"]).strip(),
        event_date=_required_date(args, "event_date"),
        capacity_usd=_opt_float(args.get("capacity_usd")),
        remaining_capacity_usd=_opt_float(args.get("remaining_capacity_usd")),
        drawn_usd=_opt_float(args.get("drawn_usd")),
        agreement_end_date=_opt_date(args, "agreement_end_date"),
    )
    _check_amend_non_empty(
        m,
        sum(1 for v in (m.capacity_usd, m.remaining_capacity_usd,
                        m.drawn_usd, m.agreement_end_date) if v is not None),
    )
    return m


def _build_amend_shelf(args: dict) -> AmendShelf:
    m = AmendShelf(
        instrument_id=str(args["instrument_id"]).strip(),
        event_date=_required_date(args, "event_date"),
        capacity_usd=_opt_float(args.get("capacity_usd")),
        remaining_capacity_usd=_opt_float(args.get("remaining_capacity_usd")),
    )
    _check_amend_non_empty(
        m,
        sum(1 for v in (m.capacity_usd, m.remaining_capacity_usd)
            if v is not None),
    )
    return m


def _build_amend_warrant(args: dict) -> AmendWarrant:
    m = AmendWarrant(
        instrument_id=str(args["instrument_id"]).strip(),
        event_date=_required_date(args, "event_date"),
        count=_opt_float(args.get("count")),
        strike=_opt_float(args.get("strike")),
        exercisable_date=_opt_date(args, "exercisable_date"),
        expiration=_opt_date(args, "expiration"),
        is_pre_funded=_opt_bool(args.get("is_pre_funded")),
        series_letter=_opt_str(args.get("series_letter")),
        known_owners=_opt_str_tuple(args.get("known_owners")),
        issue_date=_opt_date(args, "issue_date"),
    )
    _check_amend_non_empty(
        m,
        sum(1 for v in (
            m.count, m.strike, m.exercisable_date, m.expiration,
            m.is_pre_funded, m.series_letter,
            m.known_owners, m.issue_date,
        ) if v is not None),
    )
    return m


def _build_amend_convertible(args: dict) -> AmendConvertible:
    m = AmendConvertible(
        instrument_id=str(args["instrument_id"]).strip(),
        event_date=_required_date(args, "event_date"),
        principal_remaining=_opt_float(args.get("principal_remaining")),
        conv_price=_opt_float(args.get("conv_price")),
        conv_discount_pct=_opt_float(args.get("conv_discount_pct")),
        convertible_date=_opt_date(args, "convertible_date"),
        maturity=_opt_date(args, "maturity"),
    )
    _check_amend_non_empty(
        m,
        sum(1 for v in (
            m.principal_remaining, m.conv_price, m.conv_discount_pct,
            m.convertible_date, m.maturity,
        ) if v is not None),
    )
    return m


def _build_amend_preferred(args: dict) -> AmendPreferred:
    m = AmendPreferred(
        instrument_id=str(args["instrument_id"]).strip(),
        event_date=_required_date(args, "event_date"),
        count=_opt_float(args.get("count")),
        conv_price=_opt_float(args.get("conv_price")),
        conversion_ratio=_opt_float(args.get("conversion_ratio")),
        convertible_date=_opt_date(args, "convertible_date"),
        maturity=_opt_date(args, "maturity"),
        stated_value=_opt_float(args.get("stated_value")),
        liquidation_preference=_opt_float(args.get("liquidation_preference")),
        dividend_rate=_opt_float(args.get("dividend_rate")),
        principal_remaining=_opt_float(args.get("principal_remaining")),
    )
    _check_amend_non_empty(
        m,
        sum(1 for v in (
            m.count, m.conv_price, m.conversion_ratio,
            m.convertible_date, m.maturity,
            m.stated_value, m.liquidation_preference, m.dividend_rate,
            m.principal_remaining,
        ) if v is not None),
    )
    return m


def _build_amend_s1_offering(args: dict) -> AmendS1Offering:
    m = AmendS1Offering(
        instrument_id=str(args["instrument_id"]).strip(),
        event_date=_required_date(args, "event_date"),
        anticipated_deal_size=_opt_float(args.get("anticipated_deal_size")),
        warrant_strike=_opt_float(args.get("warrant_strike")),
        warrant_coverage_pct=_opt_float(args.get("warrant_coverage_pct")),
        sold_to_date=_opt_float(args.get("sold_to_date")),
        placement_agent_canonical=_opt_str(args.get("placement_agent_canonical")),
        # Priced-cover fields — the 424B4 / priced-S-1/A path. The schema
        # offers these and the tool description instructs the LLM to set
        # them from the final cover, but they were previously never read
        # here, so a directly-emitted amend_s1_offering silently dropped
        # the entire priced deal size (the create→amend reroute in
        # walker.py sets them; a model-emitted amend did not).
        final_deal_size=_opt_float(args.get("final_deal_size")),
        final_pricing=_opt_float(args.get("final_pricing")),
        final_shares_offered=_opt_float(args.get("final_shares_offered")),
        final_warrant_coverage_pct=_opt_float(
            args.get("final_warrant_coverage_pct")),
    )
    _check_amend_non_empty(
        m,
        sum(1 for v in (
            m.anticipated_deal_size, m.warrant_strike,
            m.warrant_coverage_pct, m.sold_to_date,
            m.placement_agent_canonical,
            m.final_deal_size, m.final_pricing,
            m.final_shares_offered, m.final_warrant_coverage_pct,
        ) if v is not None),
    )
    return m


def _build_amend_equity(args: dict) -> AmendEquity:
    m = AmendEquity(
        instrument_id=str(args["instrument_id"]).strip(),
        event_date=_required_date(args, "event_date"),
        known_owners=_opt_str_tuple(args.get("known_owners")),
    )
    _check_amend_non_empty(
        m, 1 if m.known_owners is not None else 0,
    )
    return m


# ─── Record / close / split / note builders ───────────────────────────


def _build_record_exercise(args: dict) -> RecordExercise:
    return RecordExercise(
        instrument_id=str(args["instrument_id"]).strip(),
        shares=float(args["shares"]),
        event_date=_required_date(args, "event_date"),
        price=_opt_float(args.get("price")),
        gross_proceeds=_opt_float(args.get("gross_proceeds")),
        warrants_exercised=_opt_float(args.get("warrants_exercised")),
    )


def _build_record_conversion(args: dict) -> RecordConversion:
    # Exactly-one-of {principal_converted, preferred_shares_converted}
    # is required; which one is gated by target type in validate.py
    # (note → principal_converted, preferred → preferred_shares_converted).
    # We don't fetch the row here, so the strict type-based requirement
    # is enforced downstream; this builder only ensures at least one is
    # provided so the mutation isn't empty.
    principal_converted = _opt_float(args.get("principal_converted"))
    pref_shares = _opt_float(args.get("preferred_shares_converted"))
    if principal_converted is None and pref_shares is None:
        raise ValueError(
            "record_conversion requires either `principal_converted` "
            "(convertible note) or `preferred_shares_converted` "
            "(preferred series); got neither"
        )
    return RecordConversion(
        instrument_id=str(args["instrument_id"]).strip(),
        shares_issued=float(args["shares_issued"]),
        event_date=_required_date(args, "event_date"),
        principal_converted=principal_converted,
        preferred_shares_converted=pref_shares,
        principal_remaining=_opt_float(args.get("principal_remaining")),
    )


def _build_record_drawdown(args: dict) -> RecordDrawdown:
    # Gross is computed in-store from (shares, price_per_share). The
    # aggregate drawdown_amount_usd is a fallback for takedowns that
    # state a total dollar figure with no per-share price. One of the
    # two must be present, else there's nothing to compute proceeds from.
    price = _opt_float(args.get("price_per_share"))
    amount = _opt_float(args.get("drawdown_amount_usd"))
    # A non-positive per-share price is never a real ATM / shelf takedown
    # price (only pre-funded warrant EXERCISES price near zero, and those
    # use record_exercise). Left as-is, price_per_share=0 makes the store
    # compute gross = shares × 0 = $0 and silently book a zero-proceeds
    # drawdown that ALSO slips past the drawdown_missing_price retry guard
    # (which only fires when price_per_share is None). Coerce a bogus zero
    # to None so it's handled exactly like a missing price: when an
    # aggregate drawdown_amount_usd is present it's used (and the
    # post-parse guard bounces back for the gross per-share price);
    # otherwise the call is rejected rather than booking $0.
    if price is not None and price <= 0:
        price = None
    if price is None and amount is None:
        raise ValueError(
            "record_drawdown requires price_per_share (preferred) or "
            "drawdown_amount_usd (aggregate fallback); got neither"
        )
    return RecordDrawdown(
        instrument_id=str(args["instrument_id"]).strip(),
        drawdown_shares=float(args["drawdown_shares"]),
        event_date=_required_date(args, "event_date"),
        price_per_share=price,
        drawdown_amount_usd=amount,
        placement_agent_canonical=_opt_str(args.get("placement_agent_canonical")),
    )


def _build_record_partial_redemption(args: dict) -> RecordPartialRedemption:
    # Exactly-one-of {principal_redeemed, preferred_shares_redeemed} is
    # required; which one is gated by target type in validate.py (note
    # → principal_redeemed, preferred → preferred_shares_redeemed).
    # Same shape as _build_record_conversion above.
    principal_redeemed = _opt_float(args.get("principal_redeemed"))
    pref_shares = _opt_float(args.get("preferred_shares_redeemed"))
    if principal_redeemed is None and pref_shares is None:
        raise ValueError(
            "record_partial_redemption requires either "
            "`principal_redeemed` (convertible note) or "
            "`preferred_shares_redeemed` (preferred series); got neither"
        )
    return RecordPartialRedemption(
        instrument_id=str(args["instrument_id"]).strip(),
        event_date=_required_date(args, "event_date"),
        principal_redeemed=principal_redeemed,
        preferred_shares_redeemed=pref_shares,
        cash_paid=_opt_float(args.get("cash_paid")),
    )


def _build_record_partial_termination(args: dict) -> RecordPartialTermination:
    return RecordPartialTermination(
        instrument_id=str(args["instrument_id"]).strip(),
        capacity_reduced_usd=float(args["capacity_reduced_usd"]),
        event_date=_required_date(args, "event_date"),
    )


def _build_confirm_closing(args: dict) -> ConfirmClosing:
    return ConfirmClosing(
        instrument_id=str(args["instrument_id"]).strip(),
        event_date=_required_date(args, "closing_date"),
        count_actual=_opt_float(args.get("count_actual")),
        gross_proceeds_usd=_opt_float(args.get("gross_proceeds_usd")),
    )


def _build_close_instrument(args: dict) -> CloseInstrument:
    reason = str(args["reason"]).strip()
    replaced_by = _opt_str(args.get("replaced_by"))
    if reason == "superseded" and not replaced_by:
        raise ValueError(
            "close_instrument(reason='superseded') requires replaced_by"
        )
    if reason != "superseded" and replaced_by:
        # Not fatal but suspicious — log and drop the spurious field.
        log.warning(
            "close_instrument: replaced_by=%r set with reason=%r "
            "(only meaningful when reason='superseded'); ignoring",
            replaced_by, reason,
        )
        replaced_by = None
    return CloseInstrument(
        instrument_id=str(args["instrument_id"]).strip(),
        reason=reason,
        event_date=_required_date(args, "event_date"),
        replaced_by=replaced_by,
    )


def _build_apply_split(args: dict) -> ApplySplit:
    """Resolve EITHER (post, pre, direction) OR (ads_ratio_from,
    ads_ratio_to) into the canonical ApplySplit dataclass.

    The ads_ratio_* shape exists so the LLM never divides — an FPI
    ADS-ratio change of "1:400 to 1:4,000" lands as ads_ratio_from=400 +
    ads_ratio_to=4000, and the parser computes "reverse 1-for-10". The
    LLM extracts the two literal integers from the filing's verbatim
    text; the parser owns the ratio math.
    """
    ads_from = args.get("ads_ratio_from")
    ads_to = args.get("ads_ratio_to")
    has_ratios = ads_from is not None and ads_to is not None
    has_explicit = any(k in args for k in ("post", "pre", "direction"))
    if has_ratios and has_explicit:
        raise ValueError(
            "apply_split: pass EITHER (post, pre, direction) OR "
            "(ads_ratio_from, ads_ratio_to), not both"
        )
    if has_ratios:
        try:
            af, at = float(ads_from), float(ads_to)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"apply_split: ads_ratio_* must be numeric "
                f"(from={ads_from!r}, to={ads_to!r}): {exc}"
            ) from exc
        if af <= 0 or at <= 0:
            raise ValueError(
                f"apply_split: ads_ratio_* must be positive "
                f"(from={af}, to={at})"
            )
        if af == at:
            raise ValueError(
                f"apply_split: ads_ratio_from == ads_ratio_to ({af}) "
                f"is a no-op split"
            )
        # Reverse: ratio grows (more underlying per ADS → fewer ADSs).
        # Forward: ratio shrinks. Factor must be (close to) an integer
        # — issuers don't declare fractional split ratios.
        if at > af:
            factor = at / af
            direction = "reverse"
        else:
            factor = af / at
            direction = "forward"
        if abs(factor - round(factor)) > 1e-6:
            raise ValueError(
                f"apply_split: ADS-ratio {af}→{at} implies "
                f"{direction} split with non-integer factor "
                f"{factor:.6f} — cannot derive post/pre"
            )
        n = int(round(factor))
        post, pre = (1, n) if direction == "reverse" else (n, 1)
        units = _opt_str(args.get("units")) or "ads"
    else:
        # post/pre/direction shape — all three required together
        missing = [k for k in ("post", "pre", "direction") if k not in args]
        if missing:
            raise ValueError(
                f"apply_split: missing required arg(s) {missing} "
                f"(pass either post+pre+direction OR ads_ratio_from+"
                f"ads_ratio_to)"
            )
        post = int(args["post"])
        pre = int(args["pre"])
        direction = str(args["direction"]).strip()
        units = _opt_str(args.get("units")) or "common"
    # Consistency invariants — apply regardless of input shape
    if direction == "reverse" and post >= pre:
        raise ValueError(
            f"reverse split requires post<pre; got post={post} pre={pre}"
        )
    if direction == "forward" and post <= pre:
        raise ValueError(
            f"forward split requires post>pre; got post={post} pre={pre}"
        )
    return ApplySplit(
        post=post,
        pre=pre,
        direction=direction,
        effective_date=_required_date(args, "effective_date"),
        units=units,
    )


def _build_note_no_event(args: dict) -> NoteNoEvent:
    return NoteNoEvent(reason=str(args["reason"]).strip())


# Registry: tool name → builder. Defined after all builders so the
# function references resolve.
_BUILDERS: dict[str, Callable[[dict], ToolMutation]] = {
    "create_atm":          _build_create_atm,
    "create_shelf":        _build_create_shelf,
    "create_warrant":      _build_create_warrant,
    "create_convertible":  _build_create_convertible,
    "create_preferred":    _build_create_preferred,
    "create_equity_line":  _build_create_equity_line,
    "create_s1_offering":  _build_create_s1_offering,
    "create_equity":       _build_create_equity,
    "restate_atm":         _build_restate_atm,
    "amend_atm":           _build_amend_atm,
    "amend_equity_line":   _build_amend_equity_line,
    "amend_shelf":         _build_amend_shelf,
    "amend_warrant":       _build_amend_warrant,
    "amend_convertible":   _build_amend_convertible,
    "amend_preferred":     _build_amend_preferred,
    "amend_s1_offering":   _build_amend_s1_offering,
    "amend_equity":        _build_amend_equity,
    "record_exercise":            _build_record_exercise,
    "record_conversion":          _build_record_conversion,
    "record_drawdown":            _build_record_drawdown,
    "record_partial_redemption":  _build_record_partial_redemption,
    "record_partial_termination": _build_record_partial_termination,
    "confirm_closing":            _build_confirm_closing,
    "close_instrument":           _build_close_instrument,
    "apply_split":                _build_apply_split,
    "note_no_event":              _build_note_no_event,
}


# ─── Dispatcher ───────────────────────────────────────────────────────

def parse_tool_calls(
    tool_calls, *, accession: str,
    empty_amends: list[RetryableFailure] | None = None,
) -> list[ToolMutation]:
    """Convert a list of provider-normalized ToolCall objects (from
    llm_provider.ToolCall) into typed mutation dataclasses.

    Malformed args or unknown tool names are logged + dropped — the
    walker keeps moving rather than failing the whole filing.

    When `empty_amends` is a list, retryable failures (empty amend_*
    calls, unparseable date args) are appended to it as
    `RetryableFailure` records so the caller can decide whether to
    re-prompt the LLM. The parameter name is kept for backwards
    compatibility; it now collects both flavours.
    """
    out: list[ToolMutation] = []
    if not tool_calls:
        return out
    # Runaway-loop defense: collapse byte-identical tool calls by
    # (name, sorted-args fingerprint). A healthy filing emits a handful
    # of distinct calls; degenerate responses (e.g. NUAI 0001213900-25-
    # 099168 — 906 record_exercise(shares=1, price=0) placeholders in
    # one response) emit hundreds of duplicates. Dedup before dispatch
    # so failed-parse logging and retryable-failure accumulation
    # aren't amplified by the loop either.
    seen_fps: set[tuple[str, str]] = set()
    n_dedup = 0
    for tc in tool_calls:
        try:
            fp = (tc.name, json.dumps(
                tc.arguments, sort_keys=True, default=str,
            ))
        except (TypeError, ValueError):
            fp = (tc.name, repr(tc.arguments))
        if fp in seen_fps:
            n_dedup += 1
            continue
        seen_fps.add(fp)
        builder = _BUILDERS.get(tc.name)
        if builder is None:
            log.warning(
                "walker %s — unknown tool %r (ignoring; args=%r)",
                accession, tc.name, tc.arguments,
            )
            continue
        if "__raw_arguments__" in tc.arguments:
            log.warning(
                "walker %s — %s arguments failed JSON decode (raw=%r)",
                accession, tc.name, tc.arguments["__raw_arguments__"][:200],
            )
            continue
        try:
            mutation = builder(tc.arguments)
        except _AmendValidationError as exc:
            log.warning(
                "walker %s — %s args failed validation: %s (args=%r)",
                accession, tc.name, exc, tc.arguments,
            )
            if empty_amends is not None:
                empty_amends.append(RetryableFailure(
                    kind="empty_amend",
                    tool_name=tc.name,
                    instrument_id=str(tc.arguments.get("instrument_id") or ""),
                    event_date=tc.arguments.get("event_date"),
                    error_message=str(exc),
                ))
            continue
        except _DateParseError as exc:
            log.warning(
                "walker %s — %s args failed validation: %s (args=%r)",
                accession, tc.name, exc, tc.arguments,
            )
            if empty_amends is not None:
                empty_amends.append(RetryableFailure(
                    kind="bad_date",
                    tool_name=tc.name,
                    instrument_id=str(tc.arguments.get("instrument_id") or ""),
                    event_date=tc.arguments.get("event_date"),
                    error_message=str(exc),
                ))
            continue
        except (KeyError, ValueError, TypeError) as exc:
            log.warning(
                "walker %s — %s args failed validation: %s (args=%r)",
                accession, tc.name, exc, tc.arguments,
            )
            continue
        # First-pass guard (only while collecting failures): a share-based
        # takedown that omits price_per_share almost always means the model
        # dropped the stated "$X per share" GROSS offering price and routed
        # a (frequently net-of-fees) aggregate into drawdown_amount_usd —
        # understating the raise (GCTK Q2 ATM: net $4.32M booked for a
        # gross $4.4549M sale). Bounce it back asking for the per-share
        # gross price. The retry pass runs with empty_amends=None, so this
        # guard is skipped there and a genuine no-per-share takedown that
        # the model re-confirms via the aggregate fallback still lands.
        if (empty_amends is not None
                and isinstance(mutation, RecordDrawdown)
                and mutation.price_per_share is None
                and mutation.drawdown_shares):
            empty_amends.append(RetryableFailure(
                kind="drawdown_missing_price",
                tool_name=tc.name,
                instrument_id=mutation.instrument_id,
                event_date=tc.arguments.get("event_date"),
                error_message=(
                    "record_drawdown supplied drawdown_shares without "
                    "price_per_share; provide the GROSS per-share "
                    "offering price so proceeds aren't understated by a "
                    "net-of-fees aggregate"
                ),
            ))
            continue
        out.append(mutation)
    if n_dedup:
        log.warning(
            "walker %s — collapsed %d duplicate tool call(s) "
            "(runaway-loop guard)", accession, n_dedup,
        )
    return out


__all__ = ["parse_tool_calls", "RetryableFailure", "EmptyAmendFailure"]
