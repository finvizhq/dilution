"""Pre-apply mutation validation.

Runs after the walker LLM emits a `MutationList` but before
`store.apply_mutations` writes anything. Drops individual bad mutations
(NOT the whole filing) — one bad mutation in a multi-disclosure 8-K
shouldn't kill the adjacent good ones. Each rejection becomes a
`dilution_walk_errors` row so the failure surface is queryable.

Validation rules:

  1. Apply ordering — `apply_split` first, then `create_instrument`,
     then `amend_instrument`, then `record_event`, then
     `close_instrument`. Within a rank, declared order preserved.

  2. id existence — `amend_instrument` / `record_event` /
     `close_instrument` must reference an instrument_id that exists in
     the current ledger snapshot or was created earlier in the same
     mutation list.

  3. type compatibility — record_event.event_kind must match the
     instrument's type (only `warrant` accepts `exercise`, only
     `convertible`/`preferred` accept `conversion`, only `atm`/
     `equity_line`/`shelf` accept `drawdown`, only `convertible`/
     `preferred` accept `partial_redemption`).

  4. illegal state transitions — cannot exercise / convert / draw down
     an instrument whose status is expired / terminated / redeemed;
     cannot close an already-closed instrument unless the new reason
     is `superseded` (closed-then-superseded chain is legal).

  5. proposed_id collisions — if a CreateInstrument carries a
     proposed_id that's already in the ledger, the walker rewrites it
     to a freshly allocated id and logs the swap. (Handled in
     store.apply_mutations, not here — validate.py reports the swap
     so the audit log lines up.)

  6. capacity overflow — drawdown beyond an ATM's remaining capacity
     by more than 5% (filing-rounding tolerance) is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .mutations import (
    AmendInstrument,
    ApplySplit,
    CloseInstrument,
    CreateInstrument,
    Mutation,
    RecordEvent,
)


# Mutation kinds in apply order. Lower rank applies first.
MUTATION_APPLY_ORDER = {
    "apply_split": 0,
    "create_instrument": 1,
    "amend_instrument": 2,
    "record_event": 3,
    "close_instrument": 4,
}


# event_kind ↔ instrument type compatibility. Multi-value sets
# because `drawdown` is valid on shelf / atm / equity_line and
# `conversion` is valid on convertible / preferred.
_EVENT_KIND_TYPES: dict[str, frozenset[str]] = {
    "exercise": frozenset({"warrant", "preferred"}),
    "conversion": frozenset({"convertible", "preferred"}),
    "partial_redemption": frozenset({"convertible", "preferred"}),
    "partial_termination": frozenset(
        {"warrant", "convertible", "preferred", "atm",
         "equity_line", "shelf", "s1_offering"}
    ),
    "drawdown": frozenset({"atm", "equity_line", "shelf"}),
}

# Statuses that block further partial-state mutations. A closed
# instrument can only receive `close_instrument(reason="superseded")`
# to chain it to a successor; everything else is rejected.
_TERMINAL_STATUSES = frozenset(
    {"exercised", "converted", "redeemed", "expired", "terminated"}
)
# `superseded` is recorded as `superseded:<id>` so prefix-match it.
_SUPERSEDED_PREFIX = "superseded"

# Drawdown overflow tolerance. Filings round capacity figures (e.g.
# "approximately $25M remaining") so allow 5% slack before rejecting.
DRAWDOWN_TOLERANCE = 0.05

# Filing forms that may legitimately produce a `create_instrument(shelf)`.
# A new shelf row is born from a BASE shelf registration filing and
# nothing else — 8-Ks, 424Bs, S-3/A amendments, POS AM, periodic
# filings, and Reg A+ filings must NOT mint shelf rows. /A amendments
# go through `amend_instrument`; takedowns disclosed in 8-K/424B go
# through `record_event(drawdown)` against the existing shelf.
_SHELF_CREATE_FILING_FORMS = frozenset(
    {"S-3", "S-3ASR", "S-3MEF", "F-3", "F-3ASR", "F-3MEF"}
)


@dataclass
class ValidationResult:
    """One mutation's verdict.

    `accepted` = pass it to apply_mutations. `error_kind` + `message`
    populate dilution_walk_errors when not accepted.
    """

    mutation: Mutation
    accepted: bool
    error_kind: str | None = None
    message: str | None = None


@dataclass
class ValidationReport:
    accepted: list[Mutation] = field(default_factory=list)
    rejected: list[ValidationResult] = field(default_factory=list)


def sort_mutations(mutations: Iterable[Mutation]) -> list[Mutation]:
    """Stable-sort by apply rank. Within a rank, declared order is
    preserved so the LLM can express ordered intent (create X, then
    record event against X) in one filing."""
    return sorted(
        mutations,
        key=lambda m: MUTATION_APPLY_ORDER.get(m.kind, 99),
    )


def validate_mutations(
    mutations: Iterable[Mutation],
    ledger_snapshot: dict[str, dict],
    filing_form: str | None = None,
) -> ValidationReport:
    """Validate a filing's mutation list against the current ledger.

    `ledger_snapshot` is a dict keyed by instrument_id; values are
    ledger rows in the shape returned by `store.get_open_instruments`
    (plus closed rows the walker chose to expose). Mutations that
    create new instruments are tracked in a local pending map so
    later mutations in the same filing can reference them by
    proposed_id.

    `filing_form` is the SEC form of the filing being processed (e.g.
    "S-3ASR", "8-K", "424B5"). Used to enforce the rule that shelf
    rows can only be minted from a base shelf registration — without
    this, the LLM repeatedly emits `create_instrument(shelf)` from
    8-K takedown announcements and 424B prospectus supplements.
    """
    report = ValidationReport()
    ordered = sort_mutations(mutations)

    # Working copies so this function stays pure: the walker re-reads
    # the actual ledger inside apply_mutations.
    live_ids: set[str] = set(ledger_snapshot.keys())
    proposed_so_far: dict[str, dict] = {}

    for m in ordered:
        verdict = _validate_one(
            m, ledger_snapshot, live_ids, proposed_so_far, filing_form,
        )
        if verdict.accepted:
            report.accepted.append(m)
            # Track newly-created instruments so subsequent mutations
            # in this same filing can reference them.
            if isinstance(m, CreateInstrument) and m.proposed_id:
                proposed_so_far[m.proposed_id] = {
                    "instrument_id": m.proposed_id,
                    "type": m.type,
                    "status": "active",
                    "terms_json": _to_terms(m.terms),
                    "outstanding_json": _to_terms(m.outstanding),
                }
                live_ids.add(m.proposed_id)
        else:
            report.rejected.append(verdict)

    return report


def _validate_one(
    m: Mutation,
    ledger: dict[str, dict],
    live_ids: set[str],
    pending: dict[str, dict],
    filing_form: str | None = None,
) -> ValidationResult:
    if isinstance(m, ApplySplit):
        if m.ratio <= 0:
            return _reject(m, "invalid_split",
                           f"split ratio must be > 0; got {m.ratio}")
        return ValidationResult(mutation=m, accepted=True)

    if isinstance(m, CreateInstrument):
        # Reject obvious mis-classification: a `shelf` whose label
        # describes an EVENT against a shelf (takedown / drawdown) is
        # the LLM emitting `create_instrument` where it should have
        # emitted `record_event(drawdown)`. The walker prompt
        # explicitly bans this; the validator is the deterministic
        # safety net so polluted rows never reach the ledger.
        if m.type == "shelf" and m.label:
            lbl = m.label.lower()
            if "takedown" in lbl or "drawdown" in lbl:
                return _reject(
                    m, "shelf_takedown_misclassified",
                    f"shelf label {m.label!r} describes an event, not "
                    "an instrument; emit record_event(drawdown) instead",
                )
        # Shelves can only be minted from a base shelf registration
        # filing. The LLM often writes `terms.form="S-3"` while
        # processing an 8-K takedown announcement or 424B prospectus
        # supplement — `terms.form` is what it CLAIMS the underlying
        # shelf is, not the filing it's reading. Cross-checking against
        # the actual filing form is the only deterministic way to stop
        # 8-K/424B-sourced phantom shelves (the AMC SH-027/SH-028/
        # SH-033 case). /A amendments also fail this check, which is
        # correct — they go through amend_instrument.
        if m.type == "shelf" and filing_form:
            ff_norm = str(filing_form).upper().replace(" ", "").split("/")[0]
            if ff_norm not in _SHELF_CREATE_FILING_FORMS:
                return _reject(
                    m, "shelf_wrong_filing_form",
                    f"shelf create not allowed from filing form "
                    f"{filing_form!r}; only "
                    f"{sorted(_SHELF_CREATE_FILING_FORMS)} may mint "
                    "shelf rows. 424B / 8-K takedowns → "
                    "record_event(drawdown); /A amendments → "
                    "amend_instrument.",
                )
        # Belt-and-suspenders: even when filing_form is unknown, the
        # LLM-claimed terms.form must look like a base shelf. /A
        # suffixes (S-3/A, F-3/A) are amendments and must go through
        # amend_instrument, not create_instrument.
        if m.type == "shelf":
            form_raw = (m.terms or {}).get("form") or ""
            form_norm = str(form_raw).upper().replace(" ", "")
            _SHELF_BASE_FORMS = {"S-3", "S-3ASR", "F-3", "F-3ASR"}
            if form_norm not in _SHELF_BASE_FORMS:
                return _reject(
                    m, "shelf_misclassified",
                    f"shelf create requires terms.form in "
                    f"{sorted(_SHELF_BASE_FORMS)}; got {form_raw!r}. "
                    "A 424B5 takedown, 8-K announcement, or S-3/A "
                    "amendment is NOT a new shelf — emit "
                    "record_event(drawdown) or amend_instrument.",
                )
        # proposed_id collisions are not a validation failure — the
        # apply layer detects existing ids and reallocates atomically.
        # Validator just lets the create through; store._apply_create
        # owns id assignment, including the sequence high-water mark.
        return ValidationResult(mutation=m, accepted=True)

    # Below: amend / record_event / close — all require an existing id.
    target_id = m.instrument_id
    if target_id not in live_ids:
        return _reject(m, "missing_id",
                       f"instrument_id {target_id!r} not found in ledger")

    target = ledger.get(target_id) or pending.get(target_id)
    if target is None:
        # Created earlier in this filing but with proposed_id absent
        # (walker auto-allocated). Should be rare since the LLM is
        # told to use proposed_id when chaining; treat as missing.
        return _reject(m, "missing_id",
                       f"instrument_id {target_id!r} created in "
                       "this filing without proposed_id; cannot chain")

    target_type = target.get("type")
    target_status = (target.get("status") or "active").lower()
    is_terminal = (
        target_status in _TERMINAL_STATUSES
        or target_status.startswith(_SUPERSEDED_PREFIX)
    )

    if isinstance(m, RecordEvent):
        valid_types = _EVENT_KIND_TYPES.get(m.event_kind, frozenset())
        if target_type not in valid_types:
            return _reject(
                m, "type_mismatch",
                f"event_kind {m.event_kind!r} incompatible with "
                f"instrument type {target_type!r}",
            )
        if is_terminal:
            return _reject(
                m, "illegal_transition",
                f"cannot record_event {m.event_kind!r} on "
                f"{target_type} {target_id} in status {target_status!r}",
            )
        # Drawdown capacity overflow check.
        if m.event_kind == "drawdown":
            overflow = _drawdown_overflow(m, target)
            if overflow is not None:
                return _reject(
                    m, "capacity_overflow",
                    f"drawdown {m.fields!r} exceeds remaining "
                    f"capacity by {overflow:.1%}",
                )
        return ValidationResult(mutation=m, accepted=True)

    if isinstance(m, AmendInstrument):
        if is_terminal:
            return _reject(
                m, "illegal_transition",
                f"cannot amend {target_type} {target_id} in status "
                f"{target_status!r} (terminal)",
            )
        return ValidationResult(mutation=m, accepted=True)

    if isinstance(m, CloseInstrument):
        # Closing a closed instrument is legal only when chaining via
        # `superseded` — that's how exchange offers / repricings that
        # use new ids get linked.
        if is_terminal and m.reason != "superseded":
            return _reject(
                m, "illegal_transition",
                f"cannot close {target_id} in status {target_status!r} "
                f"with reason {m.reason!r}",
            )
        if m.reason == "superseded" and not m.replaced_by:
            return _reject(
                m, "missing_replaced_by",
                "close_instrument(reason=superseded) requires replaced_by",
            )
        return ValidationResult(mutation=m, accepted=True)

    # Unknown mutation kind — Pydantic should have caught this, but
    # defense in depth keeps the walker from crashing.
    return _reject(m, "unknown_kind",
                   f"unrecognized mutation kind {m.kind!r}")


def _drawdown_overflow(m: RecordEvent, target: dict) -> float | None:
    """Return overflow ratio if drawdown exceeds remaining capacity by
    more than DRAWDOWN_TOLERANCE; None when within tolerance / no
    capacity figure is known."""
    outstanding = _from_json_field(target, "outstanding_json")
    remaining = (
        outstanding.get("remaining_capacity_usd")
        or outstanding.get("capacity_remaining_usd")
        or outstanding.get("remaining_usd")
    )
    if remaining is None or remaining <= 0:
        return None
    requested = (
        m.fields.get("drawdown_amount_usd")
        or m.fields.get("amount_usd")
        or m.fields.get("gross_proceeds")
    )
    if requested is None or requested <= 0:
        return None
    overflow = (requested - remaining) / remaining
    return overflow if overflow > DRAWDOWN_TOLERANCE else None


def _reject(m: Mutation, kind: str, msg: str) -> ValidationResult:
    return ValidationResult(
        mutation=m, accepted=False, error_kind=kind, message=msg,
    )


def _to_terms(d) -> dict:
    """Pass-through coercion. Kept as a hook in case the walker ever
    wants to canonicalize keys / units before storing; today it's a
    no-op."""
    return dict(d) if d else {}


def _from_json_field(row: dict, key: str) -> dict:
    """Read either a parsed dict (when the row came from a snapshot
    that already pre-decoded JSON) or a JSON string (when read raw
    from sqlite). Tolerates both so callers don't have to."""
    val = row.get(key)
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    import json
    try:
        return json.loads(val) or {}
    except (TypeError, ValueError):
        return {}


__all__ = [
    "DRAWDOWN_TOLERANCE",
    "MUTATION_APPLY_ORDER",
    "ValidationReport",
    "ValidationResult",
    "sort_mutations",
    "validate_mutations",
]
