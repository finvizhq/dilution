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
     `equity_line`/`shelf`/`s1_offering` accept `drawdown`, only
     `convertible`/`preferred` accept `partial_redemption`).

  4. illegal state transitions — cannot exercise / convert / draw down
     an instrument whose status is expired / terminated / redeemed;
     cannot close an already-closed instrument unless the new reason
     is `superseded` (closed-then-superseded chain is legal).

  5. proposed_id collisions — if a Create* carries a
     proposed_id that's already in the ledger, the walker rewrites it
     to a freshly allocated id and logs the swap. (Handled in
     store.apply_mutations, not here — validate.py reports the swap
     so the audit log lines up.)

  6. capacity overflow — drawdown beyond an ATM's remaining capacity
     by more than 5% (filing-rounding tolerance) is rejected.

  7. entity-canonical sanitization — `counterparty_canonical` /
     `placement_agent_canonical` on `create_instrument` are nulled
     in-place when they contain narrative fragments, type-words,
     month-prefixed phrases, unbalanced punctuation, or all-lowercase
     strings. The create itself is accepted with a clean (if blanker)
     label.

  8. close-out cross-check — `close_instrument(reason=…)` is rejected
     when the target's outstanding state contradicts the reason: a
     `redeemed` / `converted` close requires principal_remaining=0;
     `exercised` / `terminated` on a warrant requires count=0;
     `terminated` on a convertible / preferred requires
     principal_remaining=0. Partial movements use `record_event`.

  9. stated-balance face ceiling — `amend_instrument` is rejected when
     the stated `principal_remaining` exceeds
     `PRINCIPAL_REMAINING_CEILING` × the row's face `principal`. A
     balance only moves down, so a figure that far above face is a
     cumulative-to-date or multi-note aggregate misread as a period-end
     balance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Iterable

from .mutations import (
    AmendMutation,
    ApplySplit,
    CloseInstrument,
    CreateMutation,
    Mutation,
    RecordMutation,
    RestateAtm,
)


# ─── Entity-shape rejection (LLM hallucination guard) ───────────────
# `counterparty_canonical` and `placement_agent_canonical` feed the
# deterministic label builder (_label.py). When the walker emits a
# narrative fragment instead of a real entity name (e.g.
# "warrants (november", "remaining series", "convertible promissory"),
# the assembled label becomes garbage like
# "November 2024 warrants (november Warrants". Validate the shape and
# null the field when it fails — keep the instrument, drop the bad
# qualifier. Cards.py has its own generic-CP filter as a backstop;
# this layer prevents the polluted value from reaching the label
# builder in the first place.
_GENERIC_ENTITY_TOKENS = frozenset({
    # type-words masquerading as entity names
    "warrant", "warrants", "note", "notes",
    "convertible note", "convertible notes",
    "convertible promissory", "promissory note", "promissory notes",
    "common stock", "preferred stock", "stock warrants",
    "outstanding warrants", "remaining outstanding", "remaining series",
    "certain warrants", "certain notes", "various warrants",
    # generic descriptor phrases
    "placement agent", "third party", "third parties",
    "investor", "investors", "purchaser", "purchasers",
    "holder", "holders", "lender", "lenders",
    "institutional investor", "institutional investors",
    "accredited investor", "accredited investors",
    "the investor", "the purchaser", "the purchasers",
})

_MONTH_NAME_RE = re.compile(
    r"^(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b",
    re.IGNORECASE,
)


def _canonical_entity_ok(s: str | None) -> bool:
    """Real canonical entity names are short proper-noun strings:
    'Maxim', 'Hudson Bay', 'Streeterville', 'Hybrid Capital 12 LLC'.
    Reject narrative fragments, type-words, month-prefixed phrases,
    unbalanced punctuation, and all-lowercase phrases — those are LLM
    extraction failures, not entities. None passes (null is the
    correct way to express 'no named entity')."""
    if s is None:
        return True
    if not isinstance(s, str):
        return False
    raw = s.strip()
    if not raw:
        return True
    if raw.lower() in _GENERIC_ENTITY_TOKENS:
        return False
    if _MONTH_NAME_RE.match(raw):
        return False
    # Unbalanced brackets — caught mid-extraction with no closer.
    if raw.count("(") != raw.count(")"):
        return False
    if raw.count("[") != raw.count("]"):
        return False
    if not raw[0].isalnum():
        return False
    # Real entity names have at least one uppercase letter (proper
    # noun). All-lowercase strings are narrative fragments —
    # 'remaining series', 'convertible promissory'.
    if raw == raw.lower():
        return False
    # A name made up ENTIRELY of security-type vocabulary is the LLM
    # writing the instrument description into the party slot —
    # capitalization doesn't save it ('Redeemable Convertible' on
    # ACTU's 2018 warrants, 'warrants to Warrants' labels on KSCP's
    # pre-IPO cohorts). Real entities carry at least one
    # non-vocabulary token ('Hybrid Capital 12 LLC' → 'hybrid').
    tokens = [t for t in re.split(r"[^a-z0-9]+", raw.lower()) if t]
    if tokens and all(t in _SECURITY_VOCAB_TOKENS for t in tokens):
        return False
    return True


_SECURITY_VOCAB_TOKENS = frozenset({
    "warrant", "warrants", "convertible", "redeemable", "preferred",
    "note", "notes", "stock", "series", "common", "shares", "share",
    "purchase", "to", "of", "the", "a", "an", "and", "pre", "funded",
    "prefunded", "exercisable", "outstanding", "unit", "units",
})


def _sanitize_entity_canonicals(m):
    """Null `counterparty_canonical` / `placement_agent_canonical`
    when they fail entity-shape checks. Mutations are frozen dataclasses,
    so this returns `(sanitized_mutation, sanitized_field_names)` — the
    caller must use the returned instance. The walker schema no longer
    carries verbatim `counterparty` / `placement_agent` fields — only
    the canonical forms are emitted, so this clears just those."""
    changes: dict[str, None] = {}
    if not _canonical_entity_ok(m.counterparty_canonical):
        changes["counterparty_canonical"] = None
    if not _canonical_entity_ok(m.placement_agent_canonical):
        changes["placement_agent_canonical"] = None
    if not changes:
        return m, []
    return replace(m, **changes), list(changes)


# Mutation kinds in apply order. Lower rank applies first.
MUTATION_APPLY_ORDER = {
    "apply_split": 0,
    "create_instrument": 1,
    # restate mints a successor + supersedes a predecessor; it must land
    # after plain creates (so same-filing creates exist) and before
    # amend/record/close (so in-filing drawdowns can target the successor).
    "restate_instrument": 1,
    "amend_instrument": 2,
    "record_event": 3,
    "close_instrument": 4,
}


# event_kind ↔ instrument type compatibility. Multi-value sets
# because `drawdown` is valid on shelf / atm / equity_line and
# `conversion` is valid on convertible / preferred.
_EVENT_KIND_TYPES: dict[str, frozenset[str]] = {
    "exercise": frozenset({"warrant"}),
    "conversion": frozenset({"convertible", "preferred"}),
    "partial_redemption": frozenset({"convertible", "preferred"}),
    "partial_termination": frozenset({"atm", "equity_line", "shelf"}),
    "drawdown": frozenset({"atm", "equity_line", "shelf", "s1_offering"}),
    # confirm_closing: relabel + record actual issuance of a previously
    # announced tranche (warrant/convertible/preferred), or — for
    # equity (off-shelf PIPE) — book the closing cash into the
    # drawdowns table without any relabel. ATM/shelf/equity-line use
    # record_drawdown for their closing flow instead.
    "closing": frozenset({"warrant", "convertible", "preferred", "equity"}),
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

# Ceiling on a stated `principal_remaining`, as a multiple of the
# instrument's face `principal`. Conversions and redemptions only move a
# balance DOWN, so a period-end balance above face has to come from a
# contractual accrual — and every such mechanism is bounded: PIK /
# default-interest on a 1-3y microcap note, and the conventional
# 150%-of-principal default premium. 2x admits both with headroom.
# Above it the figure is an accumulation artifact — a cumulative-to-date
# or multi-note aggregate read as a period-end balance, ratcheted higher
# by each successive amend (IPW C-119: a $1,815,976 note amended
# 1.82M → 3.82M → 5.18M → 6.69M = 3.68x face). Empirical basis: of the
# 31 convertibles carrying both figures, 29 sit at <= 1.000x and the
# only rows above are that defect plus one exact 1.500x default premium.
PRINCIPAL_REMAINING_CEILING = 2.0

# Filing forms that may legitimately produce a `create_instrument(shelf)`.
# A new shelf row is born from a BASE shelf registration filing and
# nothing else — 8-Ks, 424Bs, S-3/A amendments, POS AM, periodic
# filings, and Reg A+ filings must NOT mint shelf rows. /A amendments
# go through `amend_instrument`; takedowns disclosed in 8-K/424B go
# through `record_event(drawdown)` against the existing shelf.
_SHELF_CREATE_FILING_FORMS = frozenset(
    {"S-3", "S-3ASR", "S-3MEF", "F-3", "F-3ASR", "F-3MEF",
     # Canadian MJDS shelf (analog of F-3 / F-3ASR)
     "F-10", "F-10EF"}
)

# Filing forms whose narrative summary tables routinely reference
# convertibles and warrants WITHOUT carrying the per-instrument
# identifying terms (conv_price, maturity, strike, expiration). The
# walker LLM, faced with a 10-Q sentence like "$5.5M of convertible
# notes outstanding," tends to emit `create_instrument(principal=5.5M)`
# with everything else null — producing a malformed ledger row that
# the anchor specialist then can't match against the formal outstanding-
# instruments table, so it permanently shows as `extra_in_ledger`.
#
# Real-data measurement (IQST + GCTK + FCEL + XTIA, 2026-05-15):
#   convertible from 10-K/10-Q: 24/24 = 100% malformed
#   warrant from 10-K/10-Q:     29/32 = 91%  malformed
#   preferred from 10-K/10-Q:    0/6  = 0%   malformed
#
# The guard below rejects warrant / convertible creates from these
# forms when the identifying terms are missing. Preferred is exempt
# because filings reliably break preferred out by series (series_letter
# + conv_price always present). Legitimate cases — typically a 10-K
# Subsequent Events note disclosing a NEW deal with full SPA terms —
# pass through unchanged because the LLM has the terms available.
_PERIODIC_NARRATIVE_FORMS = frozenset(
    {"10-K", "10-Q", "20-F", "40-F"}
)

# Post-effective amendments re-register / refresh an existing shelf's
# prospectus; they do NOT terminate the sales programs hosted under that
# shelf. The walker reliably misreads the boilerplate conditional clause
# in a POS AM — "upon termination of the sales agreement, any unsold
# portion will be available for sale in other offerings" — as an actual
# termination event and closes the live ATM. FCEL 0001104659-24-132194
# (POS AM No. 2 to the Oct-2023 S-3) is the case in point: it closed
# every ATM, pointed `superseded` at a replacement it never created, and
# thereby rejected every subsequent drawdown and the Dec-2025 $200M
# capacity increase. Genuine sales-program supersession is owned by the
# store (_auto_supersede_prior_atm) when a SUCCESSOR program is created
# on a shelf-host form — never by an explicit close on a POS AM.
_POST_EFFECTIVE_AMENDMENT_FORMS = frozenset(
    {"POSAM", "POSASR", "POSEX", "POS462B", "POS462C"}
)
# Forms that may legitimately carry an ATM restatement (restate_atm).
# Mirrors the tool's valid_forms, normalized (spaces stripped, /A suffix
# dropped). An amended-and-restated sales agreement is published as a
# 424B prospectus supplement or stapled into a fresh S-3/F-3. Excluded:
# 8-K (the same amendment also lands as a 424B5, so 8-K double-mints) and
# POS AM (a post-effective amendment re-registers the host shelf — it is
# not a new sales agreement; restating off it minted a phantom ATM that
# superseded FCEL's live April-2024 row).
_RESTATE_FILING_FORMS = frozenset(
    {"S-3", "S-3ASR", "S-3MEF", "F-3", "F-3ASR", "F-3MEF",
     "424B5", "424B3", "424B4", "424B2", "SUPPL"}
)
# Shelf-hosted sales programs (mirror of store._SHELF_HOSTED_TYPES): a
# POS AM amending the host shelf must not close any of these.
_SALES_PROGRAM_TYPES = frozenset({"atm", "equity_line", "s1_offering"})


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
    # Accumulates the effect of earlier accepted Amend*s on
    # existing rows, so a subsequent CloseInstrument in the same list
    # sees the post-amend state when its outstanding-zero precondition
    # is checked. Without this, anchor reconciliation's matured-zombie
    # close (anchor emits amend(principal_remaining=0) + close(redeemed))
    # would be rejected because the snapshot still shows the pre-amend
    # principal — and the same zombie would re-flag every periodic.
    effects_overlay: dict[str, dict] = {}

    for m in ordered:
        verdict = _validate_one(
            m, ledger_snapshot, live_ids, proposed_so_far,
            effects_overlay, filing_form,
        )
        if verdict.accepted:
            # Append the VALIDATED mutation — _validate_one may have
            # rewritten it (_sanitize_entity_canonicals nulls vocab-only
            # party fragments like 'redeemable convertible' / 'warrants
            # to'). Appending the raw input discarded the sanitization
            # and let the garbage reach the store and the card labels
            # (round-4 validate-discards-sanitized-mutation).
            m = verdict.mutation if verdict.mutation is not None else m
            report.accepted.append(m)
            # Track newly-created instruments so subsequent mutations
            # in this same filing can reference them.
            if isinstance(m, CreateMutation) and m.proposed_id:
                proposed_so_far[m.proposed_id] = {
                    "instrument_id": m.proposed_id,
                    "type": m.type,
                    "status": "active",
                    "terms_json": _to_terms(m.terms),
                    "outstanding_json": _to_terms(m.outstanding),
                }
                live_ids.add(m.proposed_id)
            elif isinstance(m, RestateAtm) and m.proposed_id:
                # The restated successor is a live ATM for the rest of
                # this filing, so an in-filing drawdown can target it.
                proposed_so_far[m.proposed_id] = {
                    "instrument_id": m.proposed_id,
                    "type": "atm",
                    "status": "active",
                    "terms_json": _to_terms({"capacity_usd": m.capacity_usd}),
                    "outstanding_json": _to_terms(
                        {"remaining_capacity_usd":
                         m.remaining_capacity_usd
                         if m.remaining_capacity_usd is not None
                         else m.capacity_usd}),
                }
                live_ids.add(m.proposed_id)
            elif isinstance(m, AmendMutation):
                _accumulate_amend_effect(
                    m, ledger_snapshot, effects_overlay,
                )
            elif getattr(m, "kind", None) == "record_event":
                _accumulate_event_effect(
                    m, ledger_snapshot, effects_overlay,
                )
        else:
            report.rejected.append(verdict)

    return report


def _accumulate_amend_effect(
    m: "AmendMutation",
    snapshot: dict[str, dict],
    overlay: dict[str, dict],
) -> None:
    """Merge an accepted Amend*'s updates into the overlay so
    subsequent mutations in this same list see the post-amend state."""
    base = overlay.get(m.instrument_id) or snapshot.get(m.instrument_id)
    if base is None:
        return
    terms = _from_json_field(base, "terms_json")
    out = _from_json_field(base, "outstanding_json")
    for k, v in (m.field_updates or {}).items():
        if v is None:
            terms.pop(k, None)
        else:
            terms[k] = v
    for k, v in (m.outstanding_updates or {}).items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = v
    new_row = dict(base)
    new_row["terms_json"] = terms
    new_row["outstanding_json"] = out
    overlay[m.instrument_id] = new_row


def _accumulate_event_effect(
    m: Mutation,
    snapshot: dict[str, dict],
    overlay: dict[str, dict],
) -> None:
    """Fold an accepted record_event's count / principal decrements into
    the overlay so sibling mutations validate against the POST-event
    state. Without this, the prompt's mandated full-retirement pattern —
    record_exercise(full) + close_instrument(exercised) in ONE batch —
    is self-defeating: the close guard reads the pre-event count and
    rejects the close (ACTU W-4341: record_event 26,070 net shares, then
    close rejected with 'count=76,376'). Mirrors store._apply_event's
    decrements for the close-gating fields only (count,
    principal_remaining); never increases a balance."""
    iid = getattr(m, "instrument_id", None)
    if not iid:
        return
    base = overlay.get(iid) or snapshot.get(iid)
    if base is None:
        return
    terms = _from_json_field(base, "terms_json")
    out = _from_json_field(base, "outstanding_json")
    fields = dict(getattr(m, "fields", None) or {})
    kind = getattr(m, "event_kind", None)
    row_type = (base.get("type") or "").lower()

    def _dec(key: str, by: float) -> None:
        cur = _coerce_num(out.get(key))
        if cur is None or by <= 0:
            return
        out[key] = max(0.0, cur - by)

    if kind == "exercise":
        retired = (_coerce_num(fields.get("warrants_exercised"))
                   or _coerce_num(fields.get("shares")) or 0)
        _dec("count", retired)
    elif kind == "conversion":
        principal = _coerce_num(fields.get("principal_converted")) or 0
        if principal:
            _dec("principal_remaining", principal)
        if "principal_remaining" in fields:
            pr = _coerce_num(fields.get("principal_remaining"))
            if pr is not None:
                out["principal_remaining"] = pr
        pref_shares = (_coerce_num(
            fields.get("preferred_shares_converted")) or 0)
        if not pref_shares and principal and row_type == "preferred":
            sv = _coerce_num(terms.get("stated_value"))
            if sv:
                pref_shares = principal / sv
        if pref_shares and row_type == "preferred":
            _dec("count", pref_shares)
    elif kind == "partial_redemption":
        amount = _coerce_num(fields.get("principal_redeemed")) or 0
        if amount:
            _dec("principal_remaining", amount)
        pref_shares = (_coerce_num(
            fields.get("preferred_shares_redeemed")) or 0)
        if not pref_shares and amount and row_type == "preferred":
            sv = _coerce_num(terms.get("stated_value"))
            if sv:
                pref_shares = amount / sv
        if pref_shares and row_type == "preferred":
            _dec("count", pref_shares)
    else:
        return

    new_row = dict(base)
    new_row["terms_json"] = terms
    new_row["outstanding_json"] = out
    overlay[iid] = new_row


def _validate_one(
    m: Mutation,
    ledger: dict[str, dict],
    live_ids: set[str],
    pending: dict[str, dict],
    effects_overlay: dict[str, dict] | None = None,
    filing_form: str | None = None,
) -> ValidationResult:
    if isinstance(m, ApplySplit):
        if m.ratio <= 0:
            return _reject(m, "invalid_split",
                           f"split ratio must be > 0; got {m.ratio}")
        return ValidationResult(mutation=m, accepted=True)

    if isinstance(m, CreateMutation):
        # Sanitize entity canonicals first. Bad CP / PA values ride
        # through the label builder and produce garbage card titles
        # ('November 2024 warrants (november Warrants'); null them
        # before validation continues so the create succeeds with a
        # clean — if uninformative — label.
        m, _ = _sanitize_entity_canonicals(m)
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
            _SHELF_BASE_FORMS = {"S-3", "S-3ASR", "F-3", "F-3ASR",
                                 "F-10", "F-10EF"}
            if form_norm not in _SHELF_BASE_FORMS:
                return _reject(
                    m, "shelf_misclassified",
                    f"shelf create requires terms.form in "
                    f"{sorted(_SHELF_BASE_FORMS)}; got {form_raw!r}. "
                    "A 424B5 takedown, 8-K announcement, or S-3/A "
                    "amendment is NOT a new shelf — emit "
                    "record_event(drawdown) or amend_instrument.",
                )
        # Periodic-filing narrative creates with missing identifying
        # terms — see _PERIODIC_NARRATIVE_FORMS comment for the data
        # behind this guard. A walker prompt change asks the LLM to
        # SKIP or AMEND rather than under-specify; this is the
        # deterministic safety net for prompt drift.
        if filing_form:
            ff_base = (str(filing_form).upper()
                       .replace(" ", "").split("/")[0])
            if ff_base in _PERIODIC_NARRATIVE_FORMS:
                terms = m.terms or {}
                if m.type == "convertible" and (
                    terms.get("conv_price") is None
                    or terms.get("maturity") is None
                ):
                    return _reject(
                        m, "periodic_create_missing_terms",
                        f"create_instrument(convertible) from periodic "
                        f"form {filing_form!r} requires terms.conv_price "
                        "AND terms.maturity. Got "
                        f"conv_price={terms.get('conv_price')!r} "
                        f"maturity={terms.get('maturity')!r}. Periodic "
                        "narrative summary tables rarely carry per-note "
                        "terms — emit amend_instrument against an "
                        "existing ledger row, or skip (the anchor "
                        "specialist captures summary-only convertibles).",
                    )
                if m.type == "warrant" and (
                    terms.get("strike") is None
                    or terms.get("expiration") is None
                ):
                    return _reject(
                        m, "periodic_create_missing_terms",
                        f"create_instrument(warrant) from periodic "
                        f"form {filing_form!r} requires terms.strike AND "
                        "terms.expiration. Got "
                        f"strike={terms.get('strike')!r} "
                        f"expiration={terms.get('expiration')!r}. "
                        "Roll-forward warrant tables rarely carry per-"
                        "tranche terms — emit amend_instrument against "
                        "an existing ledger row, or skip (the anchor "
                        "specialist captures summary-only warrants).",
                    )

        # proposed_id collisions are not a validation failure — the
        # apply layer detects existing ids and reallocates atomically.
        # Validator just lets the create through; store._apply_create
        # owns id assignment, including the sequence high-water mark.
        return ValidationResult(mutation=m, accepted=True)

    if isinstance(m, RestateAtm):
        # Form gate — a restatement is announced on a deal/amendment
        # form, never minted from a periodic anchor pass.
        if filing_form:
            ff = str(filing_form).upper().replace(" ", "").split("/")[0]
            if ff not in _RESTATE_FILING_FORMS:
                return _reject(
                    m, "restate_wrong_filing_form",
                    f"restate_atm not allowed from filing form "
                    f"{filing_form!r}; an amended-and-restated sales "
                    "agreement is announced on an 8-K / 424B / S-3 / "
                    "POS AM. For a capacity figure surfaced in a 10-Q/"
                    "10-K footnote, use amend_atm.",
                )
        pred = m.predecessor_id
        overlay = effects_overlay or {}
        target = overlay.get(pred) or pending.get(pred) or ledger.get(pred)
        if pred not in live_ids or target is None:
            return _reject(
                m, "missing_predecessor",
                f"restate_atm predecessor {pred!r} not found in ledger; "
                "use create_atm for a first-time ATM with no predecessor.",
            )
        if (target.get("type") or "").lower() != "atm":
            return _reject(
                m, "type_mismatch",
                f"restate_atm predecessor {pred!r} is type "
                f"{target.get('type')!r}, not atm.",
            )
        pstatus = (target.get("status") or "active").lower()
        if (pstatus in _TERMINAL_STATUSES
                or pstatus.startswith(_SUPERSEDED_PREFIX)):
            return _reject(
                m, "illegal_transition",
                f"cannot restate predecessor {pred!r} in terminal status "
                f"{pstatus!r}; it is no longer the live program.",
            )
        return ValidationResult(mutation=m, accepted=True)

    # Below: amend / record_event / close — all require an existing id.
    target_id = m.instrument_id
    if target_id not in live_ids:
        return _reject(m, "missing_id",
                       f"instrument_id {target_id!r} not found in ledger")

    # effects_overlay carries the post-amend state of existing rows
    # accumulated from earlier accepted mutations in this same list.
    # Prefer the overlay over the raw ledger snapshot when both are
    # present so close-instrument's outstanding-zero precondition sees
    # the zeroing amend the anchor just emitted.
    overlay = effects_overlay or {}
    target = (overlay.get(target_id)
              or pending.get(target_id)
              or ledger.get(target_id))
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

    if isinstance(m, RecordMutation):
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
        # Conversion event: type-gated input field.
        #   convertible note → principal_converted (face $ converted)
        #   preferred series → preferred_shares_converted (count retired)
        # principal_* is debt-shaped and structurally meaningless for
        # equity preferred; without preferred_shares_converted the store
        # can't decrement count, the row stays at its last anchored value,
        # and the row's outstanding only converges if the next periodic's
        # overhang itemizes it (often it doesn't). IQST P-126: 8,631
        # Series D shares converted on the 2026-03-31 10-Q but count
        # stayed at 18,020 because the walker hacked principal_converted
        # =0.01 to satisfy the old required-field guard.
        if m.event_kind == "conversion":
            fields = m.fields or {}
            if target_type == "preferred":
                pref_shares = fields.get("preferred_shares_converted")
                # A debt-shaped principal_converted on a preferred is
                # translatable to a share count when the series stated_value
                # is known (loan-to-equity rollover) — the store does the
                # division. Only reject when neither the share count nor a
                # translatable $ amount is available.
                _sv, _pr = _pos_num(
                    _from_json_field(target, "terms_json").get("stated_value")
                ), _pos_num(fields.get("principal_converted"))
                if ((not pref_shares or float(pref_shares) <= 0)
                        and not (_sv and _pr)):
                    return _reject(
                        m, "preferred_shares_required",
                        f"record_conversion on preferred {target_id} "
                        "requires `preferred_shares_converted` (the count "
                        "of preferred shares retired). principal_* fields "
                        "are debt-shaped and do not move count on a "
                        "preferred.",
                    )
            elif target_type == "convertible":
                pc = fields.get("principal_converted")
                if not pc or float(pc) <= 0:
                    return _reject(
                        m, "principal_converted_required",
                        f"record_conversion on convertible {target_id} "
                        "requires `principal_converted` (face $ amount). "
                        "preferred_shares_converted applies only to "
                        "preferred series.",
                    )
        # Partial-redemption event: same per-type field dispatch as
        # conversion. Preferred carries value in `count`, so a cash
        # redemption of N preferred shares must come in via
        # preferred_shares_redeemed for the store to drop count.
        # principal_redeemed on a preferred is a debt-shaped no-op (same
        # hole the conversion path had pre-IQST-P-126 fix).
        if m.event_kind == "partial_redemption":
            fields = m.fields or {}
            if target_type == "preferred":
                pref_shares = fields.get("preferred_shares_redeemed")
                # As in the conversion gate: a debt-shaped principal_redeemed
                # on a preferred is translatable to a share count via the
                # series stated_value, so allow it through when stated_value
                # is known (SCNI EIB loan-to-equity preliminary tranche
                # retirement). Reject only when neither path is available.
                _sv, _pr = _pos_num(
                    _from_json_field(target, "terms_json").get("stated_value")
                ), _pos_num(fields.get("principal_redeemed"))
                if ((not pref_shares or float(pref_shares) <= 0)
                        and not (_sv and _pr)):
                    return _reject(
                        m, "preferred_shares_redeemed_required",
                        f"record_partial_redemption on preferred "
                        f"{target_id} requires `preferred_shares_redeemed` "
                        "(the count of preferred shares retired for "
                        "cash). principal_* fields are debt-shaped and "
                        "do not move count on a preferred.",
                    )
            elif target_type == "convertible":
                pr = fields.get("principal_redeemed")
                if not pr or float(pr) <= 0:
                    return _reject(
                        m, "principal_redeemed_required",
                        f"record_partial_redemption on convertible "
                        f"{target_id} requires `principal_redeemed` "
                        "(face $ amount called back). "
                        "preferred_shares_redeemed applies only to "
                        "preferred series.",
                    )
        return ValidationResult(mutation=m, accepted=True)

    if isinstance(m, AmendMutation):
        if is_terminal:
            return _reject(
                m, "illegal_transition",
                f"cannot amend {target_type} {target_id} in status "
                f"{target_status!r} (terminal)",
            )
        # A preferred's liquidation_preference is structurally
        # count × stated_value (the aggregate redemption face). The LLM
        # conflates it with the CONSIDERATION exchanged for the series
        # ('$29 million of debt was converted into 1,000 preferred
        # shares' → amend liq_pref 34M→29M on SCNI EIB, where 1,000 ×
        # $34,000 stated = $34M). Reject an amend that breaks the
        # identity unless stated_value / count move in the same call.
        # Face-value ceiling on a stated balance. See
        # PRINCIPAL_REMAINING_CEILING for why 2x and what the failure mode
        # looks like. Rejecting leaves the prior (credible) balance in
        # place; the alternative — clamping to face — would invent a
        # number the filing never stated. No-ops when the row carries no
        # face (`terms.principal` absent, e.g. most preferred series,
        # whose aggregate is guarded by the liq-pref identity below).
        pr_new = _coerce_num(getattr(m, "principal_remaining", None))
        if pr_new is not None and pr_new > 0:
            face = _coerce_num(
                _from_json_field(target, "terms_json").get("principal"))
            if face and face > 0 and pr_new > face * PRINCIPAL_REMAINING_CEILING:
                return _reject(
                    m, "principal_remaining_above_face",
                    f"principal_remaining={pr_new:,.0f} is "
                    f"{pr_new / face:.2f}x the ${face:,.0f} face of "
                    f"{target_id} (ceiling "
                    f"{PRINCIPAL_REMAINING_CEILING:g}x). Conversions and "
                    "redemptions only reduce a balance, so this is a "
                    "cumulative-to-date or multi-note aggregate read as a "
                    "period-end balance. Leave the balance unchanged, or "
                    "amend `principal` in the same call if the note itself "
                    "was restated.",
                )
        lp_new = getattr(m, "liquidation_preference", None)
        if (lp_new is not None and target_type == "preferred"
                and getattr(m, "stated_value", None) is None
                and getattr(m, "count", None) is None):
            t = _from_json_field(target, "terms_json")
            o = _from_json_field(target, "outstanding_json")
            sv = _coerce_num(t.get("stated_value"))
            cnt = _coerce_num(o.get("count"))
            if sv and cnt and cnt > 0:
                implied = sv * cnt
                if abs(lp_new - implied) / implied > 0.01:
                    return _reject(
                        m, "liq_pref_inconsistent",
                        f"liquidation_preference={lp_new:,.0f} breaks "
                        f"count × stated_value = {cnt:,.0f} × {sv:,.0f} "
                        f"= {implied:,.0f} on {target_id}. A '$X of "
                        "debt converted' consideration figure is NOT "
                        "the liquidation preference — leave it "
                        "unchanged, or amend stated_value/count in the "
                        "same call if the series itself was restated.",
                    )
        return ValidationResult(mutation=m, accepted=True)

    if isinstance(m, CloseInstrument):
        # A post-effective amendment re-registers the host shelf's
        # prospectus; it never terminates the sales programs hosted under
        # it. The walker misreads the standard "upon termination of the
        # sales agreement …" conditional in a POS AM as a real
        # termination and closes the live ATM, then rejects every later
        # drawdown (FCEL 0001104659-24-132194). Sales-program
        # supersession is owned by the store when a successor is CREATED
        # on a shelf-host form — not by a close emitted here.
        if filing_form and target_type in _SALES_PROGRAM_TYPES:
            ff_norm = str(filing_form).upper().replace(" ", "").split("/")[0]
            if ff_norm in _POST_EFFECTIVE_AMENDMENT_FORMS:
                return _reject(
                    m, "close_on_post_effective_amendment",
                    f"cannot close {target_type} {target_id} from a "
                    f"post-effective amendment ({filing_form!r}); a POS AM "
                    "re-registers the host shelf and does not terminate the "
                    "sales agreement. ATM/ELOC supersession happens when the "
                    "successor program is created on a shelf-host form. Emit "
                    "amend_instrument for capacity changes, or skip.",
                )
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
        # Outstanding-state cross-check. The walker hallucinates
        # redemptions / terminations / conversions on instruments with
        # non-zero remaining principal or count — closing the row
        # removes the card entirely (cards.py only renders active /
        # superseded rows), so over-eager closes erase real overhang.
        # Guard: each close reason requires the type's *outstanding*
        # field to be 0:
        #   warrant     → count (shares)
        #   preferred   → count (shares; equity-denominated)
        #   convertible → principal_remaining (debt-denominated)
        # Equity preferred carries its value in `count` × stated /
        # liquidation value; its principal_remaining is structurally
        # 0 / None even while shares remain outstanding (e.g. a debt-
        # exchange Series D issued *for* shares, never given a
        # principal). Using pr as the guard let the walker close such
        # preferreds on a *partial* conversion — IQST P-126 was closed
        # reason='converted' after only 8,631 of 18,020 Series D shares
        # converted, because pr was 0 and the guard didn't fire. See
        # anchor._confident_close_reason for the parallel rule.
        # Past-maturity alone is NOT a close-out trigger — cards.py
        # auto-greys past-expiry warrants without needing the row
        # closed. Use record_event(partial_redemption / partial_
        # termination / conversion) for non-final movements.
        if m.reason in ("redeemed", "terminated", "expired", "exercised",
                        "converted"):
            out = _from_json_field(target, "outstanding_json")
            target_type = target.get("type")
            count = _coerce_num(out.get("count"))
            pr = _coerce_num(out.get("principal_remaining"))
            # Outstanding field that gates closure for this type, plus a
            # human-readable label for the rejection message.
            if target_type in ("warrant", "preferred"):
                out_val, out_label = count, "count"
            else:
                out_val, out_label = pr, "principal_remaining"
            blocking = None
            if m.reason == "redeemed" and out_val is not None and out_val > 0:
                # Event-filing exemption. An 8-K/6-K that says a note
                # "was paid in full" IS the redemption disclosure — there
                # is no separate per-dollar event to record first, so
                # demanding outstanding==0 deadlocks the close (CETY
                # Sep-2024 Mast Hill note: the Jan-2025 8-K's
                # "$852,406.35 as payment in full" close was rejected
                # three times and the dead note kept a live card).
                # _apply_close zeroes the outstanding as the implicit
                # final repayment. Periodic filings (10-K/10-Q/20-F)
                # keep the guard — that's where the walker's
                # hallucinated mass-close batches come from.
                ff_norm = (str(filing_form).upper().replace(" ", "")
                           .split("/")[0]) if filing_form else ""
                if ff_norm not in ("8-K", "6-K"):
                    blocking = (
                        f"{out_label}={out_val} on {target_type} "
                        f"{target_id}; full redemption requires 0. "
                        "Use record_event(partial_redemption) for partial "
                        "repayments."
                    )
            elif m.reason == "converted" and out_val is not None and out_val > 0:
                blocking = (
                    f"{out_label}={out_val} on {target_type} "
                    f"{target_id}; full conversion requires 0. "
                    "Use record_event(conversion) for partial conversions."
                )
            elif m.reason == "exercised" and target_type == "warrant" \
                    and count is not None and count > 0:
                blocking = (
                    f"count={count} on warrant {target_id}; full "
                    "exercise requires 0. Use record_event(exercise) "
                    "for partial exercises."
                )
            elif m.reason == "terminated" \
                    and out_val is not None and out_val > 0:
                blocking = (
                    f"{out_label}={out_val} on {target_type} "
                    f"{target_id}; termination requires 0 outstanding."
                )
            if blocking:
                return _reject(
                    m, "close_with_outstanding",
                    f"close_instrument(reason={m.reason!r}) blocked: "
                    f"{blocking}",
                )
        return ValidationResult(mutation=m, accepted=True)

    # Unknown mutation kind — Pydantic should have caught this, but
    # defense in depth keeps the walker from crashing.
    return _reject(m, "unknown_kind",
                   f"unrecognized mutation kind {m.kind!r}")


def _drawdown_overflow(m: "RecordMutation", target: dict) -> float | None:
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
    # Single GROSS basis, matching the store. A drawdown mutation only ever
    # carries `drawdown_amount_usd` (gross — see RecordDrawdown.fields), and
    # remaining_capacity_usd is tracked gross, so the comparison is
    # like-for-like. The former `or amount_usd or gross_proceeds` fallbacks
    # were dead (no drawdown mutation emits those keys) and made the basis
    # look non-deterministic when it is not.
    requested = m.fields.get("drawdown_amount_usd")
    if requested is None or requested <= 0:
        return None
    overflow = (requested - remaining) / remaining
    return overflow if overflow > DRAWDOWN_TOLERANCE else None


def _coerce_num(v) -> float | None:
    """Best-effort numeric coerce for outstanding-state cross-checks.
    Returns None on missing / unparseable so caller can distinguish
    'unknown' from 'zero'."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _pos_num(v) -> float | None:
    """_coerce_num, but only returns the value when strictly positive
    (else None) — for 'is this a usable amount?' gates."""
    n = _coerce_num(v)
    return n if (n is not None and n > 0) else None


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
    "PRINCIPAL_REMAINING_CEILING",
    "ValidationReport",
    "ValidationResult",
    "sort_mutations",
    "validate_mutations",
]
