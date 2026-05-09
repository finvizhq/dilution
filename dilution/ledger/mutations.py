"""Mutation vocabulary for the ledger walker.

The walker LLM emits a `MutationList` per filing — an ordered list of
ledger updates of the following kinds:

  create_instrument  — first time an instrument is disclosed
  amend_instrument   — terms change (repricing, maturity extension,
                       capacity increase, ratchet trigger, …)
  record_event       — partial state mutation: exercise / conversion /
                       partial redemption / termination / drawdown
  close_instrument   — instrument is fully consumed (exercised /
                       converted / redeemed / expired / terminated /
                       superseded by a replacement)
  apply_split        — global mutation: rewrite count + price fields
                       on every active warrant / convertible / preferred

The history-row audit trail records accession + form + action; the
LLM is not asked for a verbatim snippet. (Long verbatim strings
were tripping xAI structured-output decoder mid-string; the
accession is enough provenance — go to dilution_raw to read source.)

The discriminated union is keyed by `kind`. Both xAI structured
outputs and Moonshot json_object mode validate against MutationList
via Pydantic; pre-apply validation in validate.py enforces invariants
(id existence, type compatibility, illegal state transitions).
"""

import re
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)


# ─── Vocabulary ──────────────────────────────────────────────────────
# Keep in sync with dilution/schema.py INSTRUMENT_TYPES.
InstrumentType = Literal[
    "warrant",
    "convertible",
    "preferred",
    "atm",
    "equity_line",
    "shelf",
    "s1_offering",
    "equity",
]

# Status lifecycle. `superseded` is set with replaced_by pointing at
# the successor; the walker encodes that as status="superseded:<id>"
# in the ledger row to keep status text-only / queryable.
CloseReason = Literal[
    "exercised",
    "converted",
    "redeemed",
    "expired",
    "terminated",
    "superseded",
]

# record_event kinds. Only these are partial-state mutations; full
# closure goes through close_instrument.
EventKind = Literal[
    "exercise",          # warrant or preferred-warrant exercise
    "conversion",        # convertible-note / preferred conversion
    "partial_redemption",
    "partial_termination",
    "drawdown",          # ATM sale, equity-line draw, shelf takedown
]

SplitDirection = Literal["forward", "reverse"]
# `units` distinguishes ADS from common for FPI issuers — the walker
# applies the split only to instruments denominated in the matching
# unit, so an underlying-ordinary split that doesn't change the ADS
# ratio is a no-op for ADS-denominated warrants.
SplitUnit = Literal["common", "ads"]

# Anti-dilution classification — values mirror the prompt's
# ANTI-DILUTION CLASSIFICATION block. Keep the Literal in sync if new
# categories are added there.
AntiDilutionType = Literal[
    "Customary Anti-Dilution",
    "variable_rate",
    "full_ratchet",
    "Alternate Cashless",
    "undisclosed",
]


# ─── Date normalization ─────────────────────────────────────────────
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Accepted alternate input formats the LLM occasionally produces. We
# normalize them to YYYY-MM-DD before storage. Order matters — most
# specific first.
_DATE_ALT_FORMATS = (
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
)


def _normalize_date(value: Any) -> Any:
    """BeforeValidator for date-bearing string fields.

    Accepts None, empty string (→ None), already-ISO YYYY-MM-DD, or any
    of a few common alternate formats. Anything else raises ValueError
    so the row drops through to the walker's salvage path.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"date must be a string, got {type(value).__name__}")
    s = value.strip()
    if not s:
        return None
    if _ISO_DATE_RE.match(s):
        return s
    for fmt in _DATE_ALT_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"unrecognized date format: {value!r}")


DateStr = Annotated[str | None, BeforeValidator(_normalize_date)]


def safe_date(value: Any) -> str | None:
    """Tolerant variant of `_normalize_date`: returns the normalized
    YYYY-MM-DD string, or None when the input is unparseable. Use at
    call sites that already cascade to a fallback date — keeps a single
    bad LLM-extracted date string (e.g. year-only `'2019'`) from
    crashing the whole seed/walk path."""
    if value is None:
        return None
    try:
        return _normalize_date(value)
    except ValueError:
        return None


# ─── Per-type terms / outstanding sub-models ────────────────────────
# `extra="allow"` on every sub-model: known fields get strict type-
# checking (catches strike="$0.65", term_years=3, etc.) while novel
# keys pass through as-is so the walker's vocabulary can evolve
# without a schema change. Validation happens in CreateInstrument's
# model_validator below — terms/outstanding are still typed `dict`
# at the LLM-facing schema so xAI's structured-output decoder doesn't
# have to traverse a multi-discriminator union.
class _TermsBase(BaseModel):
    model_config = ConfigDict(extra="allow")


class WarrantTerms(_TermsBase):
    strike: float | None = None
    exercisable_date: DateStr = None
    expiration: DateStr = None
    anti_dilution_type: AntiDilutionType | None = None
    pp_clause_text: str | None = None
    is_pre_funded: bool | None = None
    units: SplitUnit | None = None


class ConvertibleTerms(_TermsBase):
    principal: float | None = None
    rate: float | None = None
    conv_price: float | None = None
    convertible_date: DateStr = None
    maturity: DateStr = None
    oid_pct: float | None = None
    anti_dilution_type: AntiDilutionType | None = None
    pp_clause_text: str | None = None


class PreferredTerms(_TermsBase):
    conv_price: float | None = None
    convertible_date: DateStr = None
    maturity: DateStr = None
    stated_value: float | None = None
    liquidation_preference: float | None = None
    dividend_rate: float | None = None
    series_letter: str | None = None
    anti_dilution_type: AntiDilutionType | None = None
    pp_clause_text: str | None = None


class ATMTerms(_TermsBase):
    capacity_usd: float | None = None


class EquityLineTerms(_TermsBase):
    capacity_usd: float | None = None


class ShelfTerms(_TermsBase):
    capacity_usd: float | None = None
    form: str | None = None


class S1OfferingTerms(_TermsBase):
    anticipated_deal_size: float | None = None
    warrant_strike: float | None = None
    warrant_coverage_pct: float | None = None


class EquityTerms(_TermsBase):
    pass


class WarrantOutstanding(_TermsBase):
    count: int | None = None


class ConvertibleOutstanding(_TermsBase):
    principal_remaining: float | None = None


class PreferredOutstanding(_TermsBase):
    count: int | None = None
    principal_remaining: float | None = None


class ATMOutstanding(_TermsBase):
    remaining_capacity_usd: float | None = None
    drawn_usd: float | None = None


class EquityLineOutstanding(_TermsBase):
    remaining_capacity_usd: float | None = None
    drawn_usd: float | None = None


class ShelfOutstanding(_TermsBase):
    remaining_capacity_usd: float | None = None


class S1OfferingOutstanding(_TermsBase):
    sold_to_date: float | None = None


class EquityOutstanding(_TermsBase):
    pass


_TERMS_BY_TYPE: dict[str, type[_TermsBase]] = {
    "warrant": WarrantTerms,
    "convertible": ConvertibleTerms,
    "preferred": PreferredTerms,
    "atm": ATMTerms,
    "equity_line": EquityLineTerms,
    "shelf": ShelfTerms,
    "s1_offering": S1OfferingTerms,
    "equity": EquityTerms,
}

_OUTSTANDING_BY_TYPE: dict[str, type[_TermsBase]] = {
    "warrant": WarrantOutstanding,
    "convertible": ConvertibleOutstanding,
    "preferred": PreferredOutstanding,
    "atm": ATMOutstanding,
    "equity_line": EquityLineOutstanding,
    "shelf": ShelfOutstanding,
    "s1_offering": S1OfferingOutstanding,
    "equity": EquityOutstanding,
}


# ─── Per-event-kind RecordEvent.fields sub-models ───────────────────
class ExerciseFields(_TermsBase):
    shares: float | None = None
    price: float | None = None
    gross_proceeds: float | None = None


class ConversionFields(_TermsBase):
    principal_converted: float | None = None
    principal_remaining: float | None = None
    shares_issued: float | None = None


class DrawdownFields(_TermsBase):
    drawdown_amount_usd: float | None = None
    drawdown_shares: float | None = None
    avg_price: float | None = None
    placement_agent: str | None = None
    placement_agent_canonical: str | None = None


class PartialRedemptionFields(_TermsBase):
    pass


class PartialTerminationFields(_TermsBase):
    pass


_FIELDS_BY_KIND: dict[str, type[_TermsBase]] = {
    "exercise": ExerciseFields,
    "conversion": ConversionFields,
    "drawdown": DrawdownFields,
    "partial_redemption": PartialRedemptionFields,
    "partial_termination": PartialTerminationFields,
}


def _validate_via(model_cls: type[_TermsBase], data: dict) -> dict:
    """Validate `data` against `model_cls` and return a plain dict.

    Keeps the walker / projector layers untouched: they still see a
    `dict`, but with normalized dates, coerced numeric types, and
    Literal-checked enum values.
    """
    validated = model_cls.model_validate(data or {})
    return validated.model_dump(exclude_unset=True)


# ─── Mutation models ─────────────────────────────────────────────────
class _MutationBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateInstrument(_MutationBase):
    kind: Literal["create_instrument"]
    type: InstrumentType
    # Walker-allocated id when None. The LLM may propose an id (e.g.
    # to recreate a known prior instrument it sees referenced) — the
    # walker rewrites colliding proposals to its next available id and
    # logs the swap.
    proposed_id: str | None = None
    # The INVESTOR / BUYER / LENDER putting capital into the issuer.
    # NOT the bank running the offering — see placement_agent. Set to
    # null when the filing only uses generic descriptors ("institutional
    # investors", "the Purchaser", etc.).
    counterparty: str | None = None
    counterparty_canonical: str | None = None
    # The BANK running the offering — underwriter / placement agent /
    # sales agent. Distinct from counterparty. E.g. "Maxim Group LLC"
    # → canonical "Maxim". Null when no bank is involved (private
    # placement, direct convertible note, internal issuance).
    placement_agent: str | None = None
    placement_agent_canonical: str | None = None
    # Clean human-readable instrument label used as the card headline.
    # E.g. "Series 9 Preferred", "Inducement Warrants", "Pre-funded
    # Warrants", "December 2022 Streeterville Note". Walker prompt
    # describes when to set it; null is acceptable and triggers the
    # mechanical card-title fallback.
    label: str | None = None
    # Type-specific fields. The LLM-facing schema is `dict` so xAI's
    # structured-output decoder doesn't need to traverse a nested
    # discriminated union, but a model_validator below routes the dict
    # through the matching per-type sub-model (WarrantTerms / etc.) so
    # known fields get strict type-checking and date normalization. New
    # fields pass through via `extra="allow"` on those sub-models.
    terms: dict = Field(default_factory=dict)
    outstanding: dict = Field(default_factory=dict)
    event_date: DateStr = None

    @model_validator(mode="after")
    def _validate_typed_dicts(self):
        terms_cls = _TERMS_BY_TYPE.get(self.type)
        if terms_cls is not None:
            self.terms = _validate_via(terms_cls, self.terms)
        out_cls = _OUTSTANDING_BY_TYPE.get(self.type)
        if out_cls is not None:
            self.outstanding = _validate_via(out_cls, self.outstanding)
        return self


class AmendInstrument(_MutationBase):
    kind: Literal["amend_instrument"]
    instrument_id: str
    # Sparse update — keys overwrite existing terms_json fields, others
    # are preserved. To clear a field, set it to None explicitly.
    field_updates: dict
    event_date: DateStr = None

    @model_validator(mode="after")
    def _normalize_field_update_dates(self):
        # Best-effort date normalization on common date keys inside
        # field_updates (the parent type isn't known here, so we apply
        # the normalizer wherever a known date key appears).
        date_keys = {"exercisable_date", "expiration",
                     "convertible_date", "maturity"}
        for k in list(self.field_updates.keys()):
            if k in date_keys:
                self.field_updates[k] = _normalize_date(
                    self.field_updates[k])
        return self


class RecordEvent(_MutationBase):
    kind: Literal["record_event"]
    instrument_id: str
    event_kind: EventKind
    # Free-form per-kind. Walker / projectors interpret. Common keys:
    #   shares, price, gross_proceeds, principal_converted,
    #   principal_remaining, drawdown_amount_usd, drawdown_shares
    fields: dict = Field(default_factory=dict)
    event_date: Annotated[str, BeforeValidator(_normalize_date)]

    @model_validator(mode="after")
    def _validate_typed_fields(self):
        cls = _FIELDS_BY_KIND.get(self.event_kind)
        if cls is not None:
            self.fields = _validate_via(cls, self.fields)
        return self


class CloseInstrument(_MutationBase):
    kind: Literal["close_instrument"]
    instrument_id: str
    reason: CloseReason
    # Required when reason="superseded"; ignored otherwise.
    replaced_by: str | None = None
    event_date: DateStr = None


class ApplySplit(_MutationBase):
    kind: Literal["apply_split"]
    # ratio = post / pre. A 1-for-10 reverse split is ratio=0.1.
    # A 2-for-1 forward split is ratio=2.0.
    ratio: float
    direction: SplitDirection
    units: SplitUnit = "common"
    effective_date: Annotated[str, BeforeValidator(_normalize_date)]


# Discriminated union by `kind`. Pydantic v2 routes to the correct
# subclass via the kind literal — ensures mutation lists from the LLM
# can mix kinds in a single response.
Mutation = Annotated[
    CreateInstrument | AmendInstrument | RecordEvent
    | CloseInstrument | ApplySplit,
    Field(discriminator="kind"),
]


class MutationList(BaseModel):
    """Walker LLM output. One per filing.

    The list ordering is meaningful: walker sorts internally to apply
    splits before creates before amends before record_events before
    closes (see validate.MUTATION_APPLY_ORDER), but a same-rank mutation
    keeps its declared order so the LLM can express "create X, then
    immediately record an exercise against X" in one filing.
    """

    model_config = ConfigDict(extra="forbid")
    mutations: list[Mutation] = Field(default_factory=list)


# ─── History entry shape ─────────────────────────────────────────────
# Stored as a JSON array on each ledger row. Not a Pydantic model
# externally — the walker constructs these when applying mutations,
# so schema is internal contract only. Documented here for reference.
#
#   {
#     "date": "YYYY-MM-DD",
#     "accession": "0001234567-25-000123",
#     "form": "8-K",
#     "action": "created" | "amended" | "exercised" | "converted"
#               | "drawn_down" | "partial_redemption"
#               | "partial_termination" | "closed" | "split_applied"
#               | "seed_from_periodic" | "anchor_reconciled",
#     "fields_changed": {...},
#   }


__all__ = [
    "ApplySplit",
    "AmendInstrument",
    "CloseInstrument",
    "CreateInstrument",
    "EventKind",
    "InstrumentType",
    "Mutation",
    "MutationList",
    "RecordEvent",
    "SplitDirection",
    "SplitUnit",
    "CloseReason",
]
