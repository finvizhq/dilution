"""Mutation vocabulary for the ledger walker.

One frozen dataclass per LLM-emitted tool call (create_*, amend_*,
record_*, close_instrument, apply_split, note_no_event). The walker
returns these directly; the apply layer (store.py, validate.py)
consumes them via isinstance dispatch on the `CreateMutation` /
`AmendMutation` / `RecordMutation` tuples below.

Each dataclass exposes:
  - `kind`             ClassVar — one of "create_instrument",
                       "amend_instrument", "record_event",
                       "close_instrument", "apply_split",
                       "note_no_event". Used for sort + logging.
  - `instrument_type`  ClassVar (Create*/Amend* only) — discriminator
                       between warrant/convertible/preferred/atm/...
  - `event_kind`       ClassVar (Record* only) — discriminator between
                       exercise/conversion/drawdown/...
  - `terms` / `outstanding`        @property dicts (Create*) — SQL
                       JSON-blob shapes consumed by store.py
  - `field_updates` / `outstanding_updates` @property dicts (Amend*)
                       — sparse updates merged into terms_json /
                       outstanding_json by _apply_amend
  - `fields`           @property dict (Record*) — event payload
                       merged into the event row
  - `type`             @property alias to `instrument_type` (Create*
                       / Amend* only) — kept for store.py call-site
                       compatibility

Validation lives at the JSON-Schema layer (dilution/ledger/tools/),
which enforces required args / patterns / minimums at LLM decode
time. Cross-arg validators (e.g. amend-non-empty) run in
dilution/ledger/tools/parse.py at construction time. Internal
callers (anchor.py, seed.py) construct dataclasses directly and
are trusted — they don't go through the JSON-Schema boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, ClassVar, Union

from dateutil.relativedelta import relativedelta


# ─── Utility helpers reused outside the walker ──────────────────────

_SERIES_LETTER_RE = re.compile(
    r"\bSeries\s+([A-Z]+|\d+)\b", re.IGNORECASE
)
# 'Class B Units' carries the same tranche identity as 'Series B' /
# bare 'B' (round-4 xtia-p336: an overhang 'Class B' line failed to
# unify with the stored 'B' preferred and minted a duplicate). The
# negative lookahead keeps SHARE-CLASS qualifiers ('Class A Common
# Stock', 'Class A Ordinary Shares') from masquerading as a series —
# those appear in warrant lines of every dual-class issuer and would
# otherwise poison series-based exclusions.
_CLASS_LETTER_RE = re.compile(
    r"\bClass\s+([A-Z]+|\d+)\b(?!\s+(?:common|ordinary))", re.IGNORECASE
)


def extract_series_letter(s: Any) -> str | None:
    """Pull a series identifier out of a string like 'Series D Preferred
    Stock', 'Series 9 Convertible Preferred', or just 'D' / '10'.
    Returns the uppercase letter(s) or the integer string, or None when
    no series marker is present.

    Issuers use either letters (Series A, B, ...) or integers (Series 1,
    2, ... — XTIA's "Series 9", IQST's eventual numbered series). Both
    are stable identifiers within one issuer; this helper accepts both.

    Used by anchor reconciliation and apply-time dedup so the same
    extraction logic runs everywhere a preferred row's identity is
    judged. Series identifier is unique within an issuer and
    definitively distinguishes tranches — stronger than conv_price-based
    dedup, which can drift across re-disclosures.
    """
    if not s:
        return None
    if not isinstance(s, str):
        s = str(s)
    bare = s.strip()
    if 1 <= len(bare) <= 3 and (bare.isalpha() or bare.isdigit()):
        return bare.upper()
    m = _SERIES_LETTER_RE.search(bare) or _CLASS_LETTER_RE.search(bare)
    return m.group(1).upper() if m else None


# A financial-statement footnote re-stating a warrant ladder drops the
# tranche's *closing date* ("August 23", "Sep 5") into the series_letter
# slot. That is NOT a series identity — matching the leading month name
# (full or abbreviated) followed somewhere by a digit, or a bare "23
# August". Real identifiers (A/B/C, digits, "Inducement", "Pre-Funded",
# "Purchase", "Common") and a bare month ("May") never match.
_WARRANT_DATE_SERIES_RE = re.compile(
    r"^\s*(?:(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|"
    r"Aug|Sep|Sept|Oct|Nov|Dec)\b.*\d"          # 'August 23', 'Sep 5'
    r"|\d{1,2}\s+(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December|Jan|Feb|Mar|Apr|"
    r"Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b)",     # '23 August'
    re.IGNORECASE,
)


def warrant_series_key(raw: Any) -> str:
    """Normalized warrant series_letter for identity comparison.

    Warrant series_letter is polymorphic: real identifiers are letters
    ('A','B','C'), digits ('1'), or descriptive tags ('Inducement',
    'Pre-Funded') and all stay discriminators. But a 10-Q footnote that
    re-states a ladder under financial-statement *closing dates* drops a
    date label ('August 23') into this slot (GCTK 2024-11-14 10-Q). A
    date label is NOT a series identity, so we return '' — the dedup
    guard then treats it as absent and falls through to strike, the same
    behavior as when the LLM drops the series tag on a re-disclosure.
    """
    s = (str(raw) if raw is not None else "").strip()
    if not s:
        return ""
    if _WARRANT_DATE_SERIES_RE.match(s):
        return ""
    return s.upper()


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATE_ALT_FORMATS = (
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
)


def _normalize_date(value: Any) -> str | None:
    """Best-effort ISO-string normalization. Accepts None, empty string,
    already-ISO, or one of a few common alternate formats. Raises on
    anything else. Kept for the few internal callers (anchor.py, the
    one-shot unnest_terms script) that received variably-formatted
    historical data."""
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


def safe_date(value: Any) -> str | None:
    """Tolerant variant of `_normalize_date`: returns the normalized
    YYYY-MM-DD string, or None when the input is unparseable. Use at
    call sites that already cascade to a fallback date — keeps a single
    bad date string (e.g. year-only `'2019'`) from crashing the whole
    seed/walk path."""
    if value is None:
        return None
    try:
        return _normalize_date(value)
    except ValueError:
        return None


def hoist_nested_payload(data: Any, wrapper_keys: tuple[str, ...]) -> Any:
    """Defensively un-nest LLM payload mistakes — kept for the one-shot
    `scripts/unnest_terms.py` data migration; not used by the live
    walker (the JSON-Schema layer prevents the nesting at the source).

    If `data[k]` for any k in `wrapper_keys` is a dict, merge that
    sub-dict UP, with the NESTED VALUE WINNING on key conflict.
    """
    if not isinstance(data, dict):
        return data
    out = dict(data)
    for key in wrapper_keys:
        nested = out.get(key)
        if not isinstance(nested, dict):
            continue
        out.pop(key, None)
        for k, v in nested.items():
            out[k] = v
    return out


def _iso(d: date | None) -> str | None:
    """date → ISO string, or None passthrough."""
    return d.isoformat() if d is not None else None


# ─── Create* dataclasses ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CreateAtm:
    kind: ClassVar[str] = "create_instrument"
    instrument_type: ClassVar[str] = "atm"

    capacity_usd: float
    event_date: date
    agreement_date: date | None = None
    agreement_end_date: date | None = None
    placement_agent_canonical: str | None = None
    remaining_capacity_usd: float | None = None
    drawn_usd: float | None = None
    proposed_id: str | None = None
    counterparty_canonical: str | None = None
    descriptor: str | None = None
    label: str | None = None

    @property
    def type(self) -> str:
        return self.instrument_type

    @property
    def terms(self) -> dict:
        d: dict = {"capacity_usd": self.capacity_usd}
        if self.agreement_date is not None:
            d["agreement_date"] = self.agreement_date.isoformat()
        if self.agreement_end_date is not None:
            d["agreement_end_date"] = self.agreement_end_date.isoformat()
        return d

    @property
    def outstanding(self) -> dict:
        d: dict = {}
        if self.remaining_capacity_usd is not None:
            d["remaining_capacity_usd"] = self.remaining_capacity_usd
        if self.drawn_usd is not None:
            d["drawn_usd"] = self.drawn_usd
        return d


@dataclass(frozen=True, slots=True)
class CreateShelf:
    kind: ClassVar[str] = "create_instrument"
    instrument_type: ClassVar[str] = "shelf"

    capacity_usd: float
    event_date: date
    form: str | None = None
    remaining_capacity_usd: float | None = None
    proposed_id: str | None = None
    counterparty_canonical: str | None = None
    placement_agent_canonical: str | None = None
    descriptor: str | None = None
    label: str | None = None
    # Synthetic shelves built from periodic-filing overhang carry the
    # registration's file_number so cards/anchors can resolve without
    # a dilution_filings join. Walker-emitted shelves leave this None.
    file_number: str | None = None

    @property
    def type(self) -> str:
        return self.instrument_type

    @property
    def terms(self) -> dict:
        d: dict = {"capacity_usd": self.capacity_usd}
        if self.form is not None:
            d["form"] = self.form
        if self.file_number is not None:
            d["file_number"] = self.file_number
        return d

    @property
    def outstanding(self) -> dict:
        d: dict = {}
        if self.remaining_capacity_usd is not None:
            d["remaining_capacity_usd"] = self.remaining_capacity_usd
        return d


@dataclass(frozen=True, slots=True)
class CreateWarrant:
    kind: ClassVar[str] = "create_instrument"
    instrument_type: ClassVar[str] = "warrant"

    count: float
    strike: float
    event_date: date
    exercisable_date: date | None = None
    expiration: date | None = None
    # Term STRUCTURE — the LLM extracts the term, code does the calendar
    # arithmetic (it never adds months/years to a date). An explicit
    # exercisable_date / expiration above always wins when the filing
    # prints a literal date. See _resolve_dates.
    term_months: int | None = None
    exercise_offset_months: int | None = None
    term_anchor: str | None = None  # "issuance" (default) | "exercise"
    is_pre_funded: bool | None = None
    units: str | None = None
    series_letter: str | None = None
    counterparty_canonical: str | None = None
    placement_agent_canonical: str | None = None
    descriptor: str | None = None
    known_owners: tuple[str, ...] | None = None
    proposed_id: str | None = None
    label: str | None = None

    @property
    def type(self) -> str:
        return self.instrument_type

    def _resolve_dates(self) -> tuple[date | None, date | None]:
        """Resolve absolute (exercisable_date, expiration) from the term
        structure so the LLM never does calendar arithmetic — it
        extracts the term, this adds the months.

        exercisable_date: explicit date > event_date + exercise_offset_months
                          > None (exercisability gated on an undated event).
        expiration:       explicit date > anchor + term_months > None.
                          anchor = the resolved exercisable_date when
                          term_anchor=='exercise' (the "Nth anniversary of
                          the Initial Exercise Date" case), else event_date.
        """
        exq = self.exercisable_date
        if exq is None and self.exercise_offset_months is not None:
            exq = self.event_date + relativedelta(
                months=self.exercise_offset_months)
        exp = self.expiration
        if exp is None and self.term_months is not None:
            anchor = (exq if self.term_anchor == "exercise" and exq is not None
                      else self.event_date)
            exp = anchor + relativedelta(months=self.term_months)
        return exq, exp

    @property
    def terms(self) -> dict:
        d: dict = {"strike": self.strike}
        exq, exp = self._resolve_dates()
        if exq is not None:
            d["exercisable_date"] = exq.isoformat()
        if exp is not None:
            d["expiration"] = exp.isoformat()
        # Pre-funded inference: the LLM signals "this is a pre-funded
        # warrant" two ways (the explicit flag and series_letter), and
        # also via near-zero strike. Any one is sufficient — the store
        # collapses them into is_pre_funded=True so render-time suppress
        # can rely on the stable flag. Without this, reverse splits
        # inflate strike above the 0.001 threshold (e.g. $0.001 × 1200
        # after two splits = $1.20) and the row leaks onto the card
        # surface as a "Common Warrants" tranche.
        is_pf = self.is_pre_funded
        if (self.series_letter or "").strip().lower() == "pre-funded":
            # The filing's own series tag is authoritative — promote even
            # when the model emitted is_pre_funded=False by mistake, so a
            # later reverse-split can never drift strike above the 0.001
            # threshold and un-suppress the row.
            is_pf = True
        elif is_pf is None and self.strike <= 0.001:
            is_pf = True
        if is_pf is not None:
            d["is_pre_funded"] = is_pf
        if self.units is not None:
            d["units"] = self.units
        if self.series_letter is not None:
            d["series_letter"] = self.series_letter
        if self.known_owners is not None:
            d["known_owners"] = list(self.known_owners)
        return d

    @property
    def outstanding(self) -> dict:
        # initial_count is the immutable "ever issued in this tranche"
        # number; count is the mutable "currently outstanding" view.
        # Splits adjust both; exercises/terminations only touch count.
        # cards.total_issued derives from initial_count so an amend that
        # restates count (e.g. a 20-F redisclosing the original size
        # after exercises) can't double-count against exercised_to_date.
        return {"count": self.count, "initial_count": self.count}


def _offset_date(explicit: date | None, anchor: date,
                 offset_months: int | None) -> date | None:
    """Resolve a relative date so the LLM never does calendar math: an
    explicit literal date always wins; otherwise anchor + offset_months;
    else None. Used for note/preferred convertible_date and maturity,
    both anchored to event_date (no exercise-anchor twist like warrants)."""
    if explicit is not None:
        return explicit
    if offset_months is not None:
        return anchor + relativedelta(months=offset_months)
    return None


@dataclass(frozen=True, slots=True)
class CreateConvertible:
    kind: ClassVar[str] = "create_instrument"
    instrument_type: ClassVar[str] = "convertible"

    principal: float
    principal_remaining: float
    event_date: date
    rate: float | None = None
    conv_price: float | None = None
    # Discount-to-market factor in decimal (0.90 = '90% of lowest
    # VWAP') for variable-rate notes — cards derive a live effective
    # conversion price from it when no fixed conv_price applies.
    conv_discount_pct: float | None = None
    convertible_date: date | None = None
    maturity: date | None = None
    # Term STRUCTURE — LLM extracts the term, code does the calendar
    # math. Absolute convertible_date / maturity above win when the
    # filing states a literal date. Both anchored to event_date.
    convertible_offset_months: int | None = None
    maturity_months: int | None = None
    oid_pct: float | None = None
    counterparty_canonical: str | None = None
    placement_agent_canonical: str | None = None
    descriptor: str | None = None
    known_owners: tuple[str, ...] | None = None
    proposed_id: str | None = None
    label: str | None = None

    @property
    def type(self) -> str:
        return self.instrument_type

    @property
    def terms(self) -> dict:
        d: dict = {"principal": self.principal}
        if self.rate is not None:
            d["rate"] = self.rate
        if self.conv_price is not None:
            d["conv_price"] = self.conv_price
        if self.conv_discount_pct is not None:
            d["conv_discount_pct"] = self.conv_discount_pct
        cd = _offset_date(self.convertible_date, self.event_date,
                          self.convertible_offset_months)
        if cd is not None:
            d["convertible_date"] = cd.isoformat()
        mat = _offset_date(self.maturity, self.event_date,
                           self.maturity_months)
        if mat is not None:
            d["maturity"] = mat.isoformat()
        if self.oid_pct is not None:
            d["oid_pct"] = self.oid_pct
        if self.known_owners is not None:
            d["known_owners"] = list(self.known_owners)
        return d

    @property
    def outstanding(self) -> dict:
        return {"principal_remaining": self.principal_remaining}


@dataclass(frozen=True, slots=True)
class CreatePreferred:
    kind: ClassVar[str] = "create_instrument"
    instrument_type: ClassVar[str] = "preferred"

    count: float
    series_letter: str
    event_date: date
    conv_price: float | None = None
    # Common-shares-per-preferred-share fixed ratio (e.g. 12.5 for
    # 'convertible at a rate of 12.5 shares of common stock per share').
    # When set and stated_value is present, the store derives conv_price
    # from stated_value / conversion_ratio — the LLM passes the verbatim
    # ratio, code does the division.
    conversion_ratio: float | None = None
    convertible_date: date | None = None
    maturity: date | None = None
    # Term STRUCTURE — LLM extracts the term, code does the calendar
    # math. Absolute convertible_date / maturity above win when stated
    # as a literal date. Both anchored to event_date. maturity is the
    # mandatory-redemption date — usually null (perpetual preferred).
    convertible_offset_months: int | None = None
    maturity_months: int | None = None
    stated_value: float | None = None
    liquidation_preference: float | None = None
    dividend_rate: float | None = None
    principal_remaining: float | None = None
    counterparty_canonical: str | None = None
    placement_agent_canonical: str | None = None
    descriptor: str | None = None
    known_owners: tuple[str, ...] | None = None
    proposed_id: str | None = None
    label: str | None = None

    @property
    def type(self) -> str:
        return self.instrument_type

    @property
    def terms(self) -> dict:
        d: dict = {"series_letter": self.series_letter}
        cp = self.conv_price
        if cp is None and self.conversion_ratio:
            if self.stated_value:
                cp = self.stated_value / self.conversion_ratio
            elif self.liquidation_preference and self.count:
                # Aggregate debt-exchange COD: no per-share stated value
                # is disclosed, only the aggregate face
                # (liquidation_preference) and the issued share count.
                # Per-preferred face = liq_pref / count, then conv_price
                # = that / conversion_ratio. (IQST Series D:
                # 3,546,136 / 37,110 / 12.5 = 7.6446.)
                cp = (self.liquidation_preference
                      / self.count / self.conversion_ratio)
        if cp is not None:
            d["conv_price"] = cp
        # Persist the verbatim fixed ratio too — cards prefer
        # count × conversion_ratio over principal / conv_price when
        # present (SCNI EIB: 1,000 × 364 = 364,000 exactly; the
        # $-division route compounds liq-pref drift into share counts).
        if self.conversion_ratio is not None:
            d["conversion_ratio"] = self.conversion_ratio
        cd = _offset_date(self.convertible_date, self.event_date,
                          self.convertible_offset_months)
        if cd is not None:
            d["convertible_date"] = cd.isoformat()
        mat = _offset_date(self.maturity, self.event_date,
                           self.maturity_months)
        if mat is not None:
            d["maturity"] = mat.isoformat()
        if self.stated_value is not None:
            d["stated_value"] = self.stated_value
        if self.liquidation_preference is not None:
            d["liquidation_preference"] = self.liquidation_preference
        if self.dividend_rate is not None:
            d["dividend_rate"] = self.dividend_rate
        if self.known_owners is not None:
            d["known_owners"] = list(self.known_owners)
        return d

    @property
    def outstanding(self) -> dict:
        # Seed initial_count at create (mirrors CreateWarrant): the
        # immutable "ever issued in this tranche" number. Without it the
        # upward-widen guard lazily stamps initial_count from a POST-
        # conversion balance, which can make count_converted_to_date
        # exceed initial_count (IQST Series D: 27,721 > 18,020). Not read
        # by the preferred card, so this is a ledger-coherence fix only.
        d: dict = {"count": self.count, "initial_count": self.count}
        if self.principal_remaining is not None:
            d["principal_remaining"] = self.principal_remaining
        return d


@dataclass(frozen=True, slots=True)
class CreateEquityLine:
    kind: ClassVar[str] = "create_instrument"
    instrument_type: ClassVar[str] = "equity_line"

    capacity_usd: float
    event_date: date
    agreement_date: date | None = None
    agreement_end_date: date | None = None
    term_months: int | None = None
    counterparty_canonical: str | None = None
    remaining_capacity_usd: float | None = None
    drawn_usd: float | None = None
    placement_agent_canonical: str | None = None
    descriptor: str | None = None
    proposed_id: str | None = None
    label: str | None = None

    @property
    def type(self) -> str:
        return self.instrument_type

    @property
    def terms(self) -> dict:
        d: dict = {"capacity_usd": self.capacity_usd}
        if self.agreement_date is not None:
            d["agreement_date"] = self.agreement_date.isoformat()
        # Explicit end date wins; otherwise derive from the term
        # duration (anchor = agreement_date, else event_date) so the
        # LLM never does calendar math — it extracts term_months only.
        end = _offset_date(self.agreement_end_date,
                           self.agreement_date or self.event_date,
                           self.term_months)
        if end is not None:
            d["agreement_end_date"] = end.isoformat()
        return d

    @property
    def outstanding(self) -> dict:
        d: dict = {}
        if self.remaining_capacity_usd is not None:
            d["remaining_capacity_usd"] = self.remaining_capacity_usd
        if self.drawn_usd is not None:
            d["drawn_usd"] = self.drawn_usd
        return d


@dataclass(frozen=True, slots=True)
class CreateS1Offering:
    kind: ClassVar[str] = "create_instrument"
    instrument_type: ClassVar[str] = "s1_offering"

    anticipated_deal_size: float
    event_date: date
    warrant_strike: float | None = None
    warrant_coverage_pct: float | None = None
    sold_to_date: float | None = None
    placement_agent_canonical: str | None = None
    proposed_id: str | None = None
    counterparty_canonical: str | None = None
    descriptor: str | None = None
    label: str | None = None

    @property
    def type(self) -> str:
        return self.instrument_type

    @property
    def terms(self) -> dict:
        d: dict = {"anticipated_deal_size": self.anticipated_deal_size}
        if self.warrant_strike is not None:
            d["warrant_strike"] = self.warrant_strike
        if self.warrant_coverage_pct is not None:
            d["warrant_coverage_pct"] = self.warrant_coverage_pct
        return d

    @property
    def outstanding(self) -> dict:
        d: dict = {}
        if self.sold_to_date is not None:
            d["sold_to_date"] = self.sold_to_date
        return d


@dataclass(frozen=True, slots=True)
class CreateEquity:
    kind: ClassVar[str] = "create_instrument"
    instrument_type: ClassVar[str] = "equity"

    count: float
    price_per_share: float
    event_date: date
    # Set only when the SAME filing states the placement closed/funded
    # (the common "signed and closed today" 8-K). Drives the apply
    # layer's drawdown write — cash is booked at closing, never at
    # signing (a signed-but-pending SPA must not inflate the cash
    # estimate). Later-filing closings go through confirm_closing.
    closing_date: date | None = None
    counterparty_canonical: str | None = None
    placement_agent_canonical: str | None = None
    descriptor: str | None = None
    known_owners: tuple[str, ...] | None = None
    proposed_id: str | None = None
    label: str | None = None

    @property
    def type(self) -> str:
        return self.instrument_type

    @property
    def terms(self) -> dict:
        d: dict = {"price_per_share": self.price_per_share}
        if self.closing_date is not None:
            d["closing_date"] = self.closing_date.isoformat()
        if self.known_owners is not None:
            d["known_owners"] = list(self.known_owners)
        return d

    @property
    def outstanding(self) -> dict:
        return {"count": self.count}


# ─── Amend* dataclasses ───────────────────────────────────────────────
# Sparse updates: every non-id/event/quote field is optional.
# `field_updates` / `outstanding_updates` return only the fields the
# caller set (omit-on-None semantics), which the apply layer merges
# into terms_json / outstanding_json via sparse-merge. Names match
# the legacy Pydantic AmendInstrument shape so store.py can read both
# via the same attribute path.


@dataclass(frozen=True, slots=True)
class AmendAtm:
    kind: ClassVar[str] = "amend_instrument"
    instrument_type: ClassVar[str] = "atm"

    instrument_id: str
    event_date: date
    capacity_usd: float | None = None
    remaining_capacity_usd: float | None = None
    drawn_usd: float | None = None
    placement_agent_canonical: str | None = None
    agreement_date: date | None = None
    agreement_end_date: date | None = None

    @property
    def type(self) -> str:
        return self.instrument_type

    @property
    def field_updates(self) -> dict:
        d: dict = {}
        if self.capacity_usd is not None:
            d["capacity_usd"] = self.capacity_usd
        if self.agreement_date is not None:
            d["agreement_date"] = self.agreement_date.isoformat()
        if self.agreement_end_date is not None:
            d["agreement_end_date"] = self.agreement_end_date.isoformat()
        return d

    @property
    def outstanding_updates(self) -> dict:
        d: dict = {}
        if self.remaining_capacity_usd is not None:
            d["remaining_capacity_usd"] = self.remaining_capacity_usd
        if self.drawn_usd is not None:
            d["drawn_usd"] = self.drawn_usd
        return d


@dataclass(frozen=True, slots=True)
class RestateAtm:
    """A sales-agreement amendment/restatement that DT renders as a NEW
    ATM card.

    Distinct from both `amend_atm` (in-place field updates that do NOT
    spawn a card) and `create_atm` (a first-time program with no named
    predecessor). A restatement mints a FRESH successor ATM — drawn reset
    to zero, like a create — and, when `supersede_prior` is set, marks the
    named predecessor `superseded:<new>` (DT shows it "Replaced"). When
    `supersede_prior` is False the predecessor stays active alongside the
    new card (issuers run concurrent programs — FCEL April-2024 +
    December-2025 are both live).

    This replaces the old implicit amend(capacity)→create reinterpretation
    the store performed in `_try_promote_atm_amend_to_restate`: the intent
    is now explicit in the tool call and the supersede decision is an
    evidence-gated argument rather than a heuristic.
    """
    kind: ClassVar[str] = "restate_instrument"
    instrument_type: ClassVar[str] = "atm"

    predecessor_id: str
    capacity_usd: float
    event_date: date
    agreement_date: date | None = None
    agreement_end_date: date | None = None
    placement_agent_canonical: str | None = None
    supersede_prior: bool = False
    remaining_capacity_usd: float | None = None
    proposed_id: str | None = None

    @property
    def type(self) -> str:
        return self.instrument_type


@dataclass(frozen=True, slots=True)
class AmendEquityLine:
    kind: ClassVar[str] = "amend_instrument"
    instrument_type: ClassVar[str] = "equity_line"

    instrument_id: str
    event_date: date
    capacity_usd: float | None = None
    remaining_capacity_usd: float | None = None
    drawn_usd: float | None = None
    agreement_end_date: date | None = None

    @property
    def type(self) -> str:
        return self.instrument_type

    @property
    def field_updates(self) -> dict:
        d: dict = {}
        if self.capacity_usd is not None:
            d["capacity_usd"] = self.capacity_usd
        if self.agreement_end_date is not None:
            d["agreement_end_date"] = self.agreement_end_date.isoformat()
        return d

    @property
    def outstanding_updates(self) -> dict:
        d: dict = {}
        if self.remaining_capacity_usd is not None:
            d["remaining_capacity_usd"] = self.remaining_capacity_usd
        if self.drawn_usd is not None:
            d["drawn_usd"] = self.drawn_usd
        return d


@dataclass(frozen=True, slots=True)
class AmendShelf:
    kind: ClassVar[str] = "amend_instrument"
    instrument_type: ClassVar[str] = "shelf"

    instrument_id: str
    event_date: date
    capacity_usd: float | None = None
    remaining_capacity_usd: float | None = None

    @property
    def type(self) -> str:
        return self.instrument_type

    @property
    def field_updates(self) -> dict:
        return {"capacity_usd": self.capacity_usd} if self.capacity_usd is not None else {}

    @property
    def outstanding_updates(self) -> dict:
        d: dict = {}
        if self.remaining_capacity_usd is not None:
            d["remaining_capacity_usd"] = self.remaining_capacity_usd
        return d


@dataclass(frozen=True, slots=True)
class AmendWarrant:
    kind: ClassVar[str] = "amend_instrument"
    instrument_type: ClassVar[str] = "warrant"

    instrument_id: str
    event_date: date
    count: float | None = None
    strike: float | None = None
    exercisable_date: date | None = None
    expiration: date | None = None
    is_pre_funded: bool | None = None
    series_letter: str | None = None
    known_owners: tuple[str, ...] | None = None
    issue_date: date | None = None

    @property
    def type(self) -> str:
        return self.instrument_type

    @property
    def field_updates(self) -> dict:
        d: dict = {}
        if self.strike is not None:
            d["strike"] = self.strike
        if self.exercisable_date is not None:
            d["exercisable_date"] = self.exercisable_date.isoformat()
        if self.expiration is not None:
            d["expiration"] = self.expiration.isoformat()
        if self.is_pre_funded is not None:
            d["is_pre_funded"] = self.is_pre_funded
        if self.series_letter is not None:
            d["series_letter"] = self.series_letter
        if self.known_owners is not None:
            d["known_owners"] = list(self.known_owners)
        if self.issue_date is not None:
            d["issue_date"] = self.issue_date.isoformat()
        return d

    @property
    def outstanding_updates(self) -> dict:
        return {"count": self.count} if self.count is not None else {}


@dataclass(frozen=True, slots=True)
class AmendConvertible:
    kind: ClassVar[str] = "amend_instrument"
    instrument_type: ClassVar[str] = "convertible"

    instrument_id: str
    event_date: date
    principal_remaining: float | None = None
    conv_price: float | None = None
    conv_discount_pct: float | None = None
    convertible_date: date | None = None
    maturity: date | None = None

    @property
    def type(self) -> str:
        return self.instrument_type

    @property
    def field_updates(self) -> dict:
        d: dict = {}
        if self.conv_price is not None:
            d["conv_price"] = self.conv_price
        if self.conv_discount_pct is not None:
            d["conv_discount_pct"] = self.conv_discount_pct
        if self.convertible_date is not None:
            d["convertible_date"] = self.convertible_date.isoformat()
        if self.maturity is not None:
            d["maturity"] = self.maturity.isoformat()
        return d

    @property
    def outstanding_updates(self) -> dict:
        return ({"principal_remaining": self.principal_remaining}
                if self.principal_remaining is not None else {})


@dataclass(frozen=True, slots=True)
class AmendPreferred:
    kind: ClassVar[str] = "amend_instrument"
    instrument_type: ClassVar[str] = "preferred"

    instrument_id: str
    event_date: date
    count: float | None = None
    conv_price: float | None = None
    # Common/ADS-per-preferred fixed ratio first disclosed in a LATER
    # filing. NOT persisted in terms; _apply_amend derives conv_price =
    # stated_value / conversion_ratio from the existing row (mirrors
    # CreatePreferred.terms, which also stores only the derived price).
    conversion_ratio: float | None = None
    convertible_date: date | None = None
    maturity: date | None = None
    stated_value: float | None = None
    liquidation_preference: float | None = None
    dividend_rate: float | None = None
    principal_remaining: float | None = None

    @property
    def type(self) -> str:
        return self.instrument_type

    @property
    def field_updates(self) -> dict:
        d: dict = {}
        if self.conv_price is not None:
            d["conv_price"] = self.conv_price
        if self.convertible_date is not None:
            d["convertible_date"] = self.convertible_date.isoformat()
        if self.maturity is not None:
            d["maturity"] = self.maturity.isoformat()
        if self.stated_value is not None:
            d["stated_value"] = self.stated_value
        if self.liquidation_preference is not None:
            d["liquidation_preference"] = self.liquidation_preference
        if self.dividend_rate is not None:
            d["dividend_rate"] = self.dividend_rate
        return d

    @property
    def outstanding_updates(self) -> dict:
        d: dict = {}
        if self.count is not None:
            d["count"] = self.count
        if self.principal_remaining is not None:
            d["principal_remaining"] = self.principal_remaining
        return d


@dataclass(frozen=True, slots=True)
class AmendS1Offering:
    kind: ClassVar[str] = "amend_instrument"
    instrument_type: ClassVar[str] = "s1_offering"

    instrument_id: str
    event_date: date
    anticipated_deal_size: float | None = None
    warrant_strike: float | None = None
    warrant_coverage_pct: float | None = None
    sold_to_date: float | None = None
    placement_agent_canonical: str | None = None
    # Priced-cover fields — populated when the offering closes (S-1/A
    # priced amendment, 424B4 final prospectus). The corresponding
    # card fields are surfaced by s1_offering_cards.
    final_deal_size: float | None = None
    final_pricing: float | None = None
    final_shares_offered: float | None = None
    final_warrant_coverage_pct: float | None = None

    @property
    def type(self) -> str:
        return self.instrument_type

    @property
    def field_updates(self) -> dict:
        d: dict = {}
        if self.anticipated_deal_size is not None:
            d["anticipated_deal_size"] = self.anticipated_deal_size
        if self.warrant_strike is not None:
            d["warrant_strike"] = self.warrant_strike
        if self.warrant_coverage_pct is not None:
            d["warrant_coverage_pct"] = self.warrant_coverage_pct
        if self.final_deal_size is not None:
            d["final_deal_size"] = self.final_deal_size
        if self.final_pricing is not None:
            d["final_pricing"] = self.final_pricing
        if self.final_shares_offered is not None:
            d["final_shares_offered"] = self.final_shares_offered
        if self.final_warrant_coverage_pct is not None:
            d["final_warrant_coverage_pct"] = self.final_warrant_coverage_pct
        return d

    @property
    def outstanding_updates(self) -> dict:
        return ({"sold_to_date": self.sold_to_date}
                if self.sold_to_date is not None else {})


@dataclass(frozen=True, slots=True)
class AmendEquity:
    kind: ClassVar[str] = "amend_instrument"
    instrument_type: ClassVar[str] = "equity"

    instrument_id: str
    event_date: date
    known_owners: tuple[str, ...] | None = None

    @property
    def type(self) -> str:
        return self.instrument_type

    @property
    def field_updates(self) -> dict:
        return ({"known_owners": list(self.known_owners)}
                if self.known_owners is not None else {})

    @property
    def outstanding_updates(self) -> dict:
        return {}


# ─── Record* dataclasses ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RecordExercise:
    kind: ClassVar[str] = "record_event"
    event_kind: ClassVar[str] = "exercise"

    instrument_id: str
    shares: float
    event_date: date
    price: float | None = None
    gross_proceeds: float | None = None
    # Cashless / net-share exercises surrender more warrants than the
    # common shares they deliver; when set, the store decrements the
    # outstanding warrant count by this instead of `shares` (which
    # stays the net common delivered, feeding exercised_to_date).
    warrants_exercised: float | None = None

    @property
    def fields(self) -> dict:
        d: dict = {"shares": self.shares}
        if self.price is not None:
            d["price"] = self.price
        if self.gross_proceeds is not None:
            d["gross_proceeds"] = self.gross_proceeds
        if self.warrants_exercised is not None:
            d["warrants_exercised"] = self.warrants_exercised
        return d


@dataclass(frozen=True, slots=True)
class RecordConversion:
    kind: ClassVar[str] = "record_event"
    event_kind: ClassVar[str] = "conversion"

    # Two flavors of conversion share this event:
    #   convertible note → `principal_converted` is the face $ converted;
    #                      store decrements `principal_remaining`.
    #   preferred series → `preferred_shares_converted` is the count of
    #                      preferred shares retired; store decrements
    #                      `count`. principal_* are debt-shaped and
    #                      structurally meaningless for equity-denominated
    #                      preferred (the walker used to overload
    #                      principal_converted with a share count — IQST
    #                      P-126: 8,631 preferred shares retired never
    #                      moved count, so the row stayed at 18,020
    #                      until the next overhang re-anchored it).
    # Exactly one of the two is required, gated by target type in
    # validate.py.
    instrument_id: str
    shares_issued: float
    event_date: date
    principal_converted: float | None = None
    preferred_shares_converted: float | None = None
    principal_remaining: float | None = None

    @property
    def fields(self) -> dict:
        d: dict = {"shares_issued": self.shares_issued}
        if self.principal_converted is not None:
            d["principal_converted"] = self.principal_converted
        if self.preferred_shares_converted is not None:
            d["preferred_shares_converted"] = self.preferred_shares_converted
        if self.principal_remaining is not None:
            d["principal_remaining"] = self.principal_remaining
        return d


@dataclass(frozen=True, slots=True)
class RecordDrawdown:
    kind: ClassVar[str] = "record_event"
    event_kind: ClassVar[str] = "drawdown"

    instrument_id: str
    drawdown_shares: float
    event_date: date
    # Gross dollars are COMPUTED, not trusted from the LLM. Preferred
    # input is price_per_share (the store multiplies); drawdown_amount_usd
    # is the aggregate-only fallback for takedowns that disclose a total
    # dollar figure with no per-share price. The parse builder guarantees
    # at least one of the two is set.
    price_per_share: float | None = None
    drawdown_amount_usd: float | None = None
    placement_agent_canonical: str | None = None

    @property
    def fields(self) -> dict:
        # GROSS proceeds = shares × per-share price. Computing here (the
        # apply_split "LLM never divides" pattern) means a net-vs-gross
        # slip can only enter as a wrong per-share price, never as a
        # wrong product or a net aggregate masquerading as gross. avg_price
        # is likewise derived — empirically 21% of LLM-emitted triples
        # drifted >5% (XTIA SH-007: emitted avg=$5.85 vs computed $59.25).
        # When only an aggregate dollar amount is disclosed (no per-share
        # price), fall back to it verbatim and recover avg from the pair.
        shares = self.drawdown_shares
        if self.price_per_share is not None:
            amount = shares * self.price_per_share
            avg = self.price_per_share
        else:
            amount = self.drawdown_amount_usd or 0.0
            avg = (amount / shares) if shares > 0 else None
        d: dict = {
            "drawdown_amount_usd": amount,
            "drawdown_shares": shares,
        }
        if avg is not None:
            d["avg_price"] = avg
        if self.placement_agent_canonical is not None:
            d["placement_agent_canonical"] = self.placement_agent_canonical
        return d


@dataclass(frozen=True, slots=True)
class RecordPartialRedemption:
    kind: ClassVar[str] = "record_event"
    event_kind: ClassVar[str] = "partial_redemption"

    # Mirror of RecordConversion's note-vs-preferred dichotomy:
    #   convertible note → `principal_redeemed` is the face $ called
    #                      back; store decrements `principal_remaining`.
    #   preferred series → `preferred_shares_redeemed` is the count of
    #                      preferred shares retired for cash; store
    #                      decrements `count`. principal_* fields are
    #                      debt-shaped and do not move count on a
    #                      preferred — same hole the conversion path
    #                      had before IQST P-126.
    # `cash_paid` is a dollar outflow either way (issuer pays cash for
    # whatever was retired), so it stays as-is.
    # Exactly one of {principal_redeemed, preferred_shares_redeemed}
    # is required, gated by target type in validate.py.
    instrument_id: str
    event_date: date
    principal_redeemed: float | None = None
    preferred_shares_redeemed: float | None = None
    cash_paid: float | None = None

    @property
    def fields(self) -> dict:
        d: dict = {}
        if self.principal_redeemed is not None:
            d["principal_redeemed"] = self.principal_redeemed
        if self.preferred_shares_redeemed is not None:
            d["preferred_shares_redeemed"] = self.preferred_shares_redeemed
        if self.cash_paid is not None:
            d["cash_paid"] = self.cash_paid
        return d


@dataclass(frozen=True, slots=True)
class RecordPartialTermination:
    kind: ClassVar[str] = "record_event"
    event_kind: ClassVar[str] = "partial_termination"

    instrument_id: str
    capacity_reduced_usd: float
    event_date: date

    @property
    def fields(self) -> dict:
        return {"capacity_reduced_usd": self.capacity_reduced_usd}


@dataclass(frozen=True, slots=True)
class ConfirmClosing:
    # Records the actual issuance/closing of a previously-announced
    # warrant / convertible / preferred tranche. event_date IS the
    # closing date — the apply layer uses it to relabel issue_date,
    # exercisable_date, and to slide expiration forward by the same
    # delta so the N-year term is preserved.
    kind: ClassVar[str] = "record_event"
    event_kind: ClassVar[str] = "closing"

    instrument_id: str
    event_date: date
    count_actual: float | None = None
    gross_proceeds_usd: float | None = None

    @property
    def fields(self) -> dict:
        d: dict = {}
        if self.count_actual is not None:
            d["count_actual"] = self.count_actual
        if self.gross_proceeds_usd is not None:
            d["gross_proceeds_usd"] = self.gross_proceeds_usd
        return d


# ─── close_instrument ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CloseInstrument:
    kind: ClassVar[str] = "close_instrument"

    instrument_id: str
    reason: str
    event_date: date
    replaced_by: str | None = None


# ─── apply_split ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ApplySplit:
    kind: ClassVar[str] = "apply_split"

    post: int
    pre: int
    direction: str
    effective_date: date
    units: str = "common"

    @property
    def ratio(self) -> float:
        return self.post / self.pre


# ─── note_no_event ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class NoteNoEvent:
    kind: ClassVar[str] = "note_no_event"

    reason: str


# ─── Type unions for isinstance dispatch ─────────────────────────────
# Used by store.py / validate.py / walker_llm.py to branch on mutation
# category without enumerating each typed class. Update if a new
# Create*/Amend*/Record* class is added.

CreateMutation = (
    CreateAtm, CreateShelf,
    CreateWarrant, CreateConvertible, CreatePreferred,
    CreateEquityLine, CreateS1Offering, CreateEquity,
)

AmendMutation = (
    AmendAtm, AmendShelf, AmendEquityLine,
    AmendWarrant, AmendConvertible, AmendPreferred,
    AmendS1Offering, AmendEquity,
)

RecordMutation = (
    RecordExercise, RecordConversion, RecordDrawdown,
    RecordPartialRedemption, RecordPartialTermination,
    ConfirmClosing,
)


Mutation = Union[
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
]

# Alias retained for tools/parse.py and any other caller that used
# the older name. ToolMutation == Mutation now that there's a single
# canonical mutation type.
ToolMutation = Mutation


@dataclass(slots=True)
class MutationList:
    """Thin wrapper around `list[Mutation]` — the walker returns this
    so consumers can keep their `mlist.mutations` call sites. Equivalent
    to the prior Pydantic MutationList minus the validation layer."""
    mutations: list[Mutation] = field(default_factory=list)


# ─── Logging helper ──────────────────────────────────────────────────
# Compact one-line repr of any Mutation for INFO-level logs. Pulls the
# grouping properties (terms / outstanding / fields / etc.) when the
# subclass defines them, otherwise falls back to a curated list of
# direct attrs. Long string values get truncated so the log stays
# scannable on a single line.
_FMT_GROUPING_PROPS = (
    "terms", "outstanding",
    "field_updates", "outstanding_updates",
    "fields",
)
_FMT_DIRECT_ATTRS = (
    "count", "strike", "capacity_usd", "reason",
    "post", "pre", "direction", "units",
    "replaced_by", "predecessor_id", "supersede_prior",
)


def _fmt_short(value) -> str:
    s = str(value)
    return s if len(s) <= 40 else s[:37] + "..."


def fmt_mutation(m) -> str:
    """One-line summary of a mutation for logs.

    Format: `<kind>[:<type>] <id> [date=…] [k=v k=v …]`
    """
    kind = getattr(m, "kind", type(m).__name__)
    typ = getattr(m, "instrument_type", None) or ""
    head = f"{kind}:{typ}" if typ else kind

    bits: list[str] = [head]
    iid = getattr(m, "instrument_id", None) or getattr(m, "proposed_id", None)
    if iid:
        bits.append(str(iid))

    ed = getattr(m, "event_date", None) or getattr(m, "effective_date", None)
    if ed is not None:
        bits.append(f"date={ed}")

    detail: dict = {}
    for attr in _FMT_GROUPING_PROPS:
        try:
            v = getattr(m, attr, None)
        except Exception:
            v = None
        if isinstance(v, dict) and v:
            detail.update(v)
    for attr in _FMT_DIRECT_ATTRS:
        v = getattr(m, attr, None)
        if v is not None and attr not in detail:
            detail[attr] = v

    if detail:
        bits.append(" ".join(f"{k}={_fmt_short(v)}" for k, v in detail.items()))
    return " ".join(bits)


# ─── Factory for internal callers (anchor / seed) ───────────────────
# These callers know the target instrument_type at construction time
# but assemble terms/outstanding as generic dicts (because they parse
# variably-shaped periodic-filing overhang tables). The factory
# dispatches to the right Create* / Amend* dataclass and fills the
# named-arg slots from the dicts.


def _to_date(v: Any) -> date | None:
    """str / date / None → date | None. Tolerant of malformed strings
    (returns None — caller cascades to a fallback)."""
    if v is None:
        return None
    if isinstance(v, date):
        return v
    s = safe_date(v)
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


_CREATE_BY_TYPE: dict[str, type] = {
    "atm": CreateAtm,
    "shelf": CreateShelf,
    "warrant": CreateWarrant,
    "convertible": CreateConvertible,
    "preferred": CreatePreferred,
    "equity_line": CreateEquityLine,
    "s1_offering": CreateS1Offering,
    "equity": CreateEquity,
}


def create_from_dict(
    *,
    type_: str,
    terms: dict,
    outstanding: dict,
    event_date: str | date | None,
    counterparty_canonical: str | None = None,
    placement_agent_canonical: str | None = None,
    descriptor: str | None = None,
    proposed_id: str | None = None,
    label: str | None = None,
) -> Mutation:
    """Construct the typed Create* dataclass matching `type_`, populating
    its named-arg fields from the generic terms/outstanding dicts.

    Used by anchor._synthesize_create and seed.seed_ledger — internal
    callers that materialize instruments from periodic-filing overhang
    rows rather than from LLM tool calls. The walker LLM path goes
    through `dilution/ledger/tools/parse.py` instead and constructs
    the typed Create* directly.

    Returns the typed dataclass. Raises KeyError if `type_` is
    unrecognized; ValueError if a required numeric field is missing.
    """
    cls = _CREATE_BY_TYPE.get(type_)
    if cls is None:
        raise KeyError(f"unknown instrument_type: {type_!r}")

    ev = _to_date(event_date) or date.today()
    common = {
        "event_date": ev,
        "proposed_id": proposed_id,
        "label": label,
    }
    if descriptor is not None:
        common["descriptor"] = descriptor

    if type_ == "atm":
        return CreateAtm(
            capacity_usd=float(terms.get("capacity_usd") or 0.0),
            agreement_date=_to_date(terms.get("agreement_date")),
            agreement_end_date=_to_date(terms.get("agreement_end_date")),
            placement_agent_canonical=placement_agent_canonical,
            counterparty_canonical=counterparty_canonical,
            remaining_capacity_usd=outstanding.get("remaining_capacity_usd"),
            drawn_usd=outstanding.get("drawn_usd"),
            **common,
        )
    if type_ == "shelf":
        return CreateShelf(
            capacity_usd=float(terms.get("capacity_usd") or 0.0),
            form=terms.get("form"),
            file_number=terms.get("file_number"),
            placement_agent_canonical=placement_agent_canonical,
            counterparty_canonical=counterparty_canonical,
            remaining_capacity_usd=outstanding.get("remaining_capacity_usd"),
            **common,
        )
    if type_ == "warrant":
        return CreateWarrant(
            count=float(outstanding.get("count") or 0.0),
            strike=float(terms.get("strike") or 0.0),
            exercisable_date=_to_date(terms.get("exercisable_date")),
            expiration=_to_date(terms.get("expiration")
                                or terms.get("maturity")),
            is_pre_funded=terms.get("is_pre_funded"),
            units=terms.get("units"),
            series_letter=terms.get("series_letter"),
            known_owners=tuple(terms["known_owners"])
                if isinstance(terms.get("known_owners"), (list, tuple)) else None,
            counterparty_canonical=counterparty_canonical,
            placement_agent_canonical=placement_agent_canonical,
            **common,
        )
    if type_ == "convertible":
        return CreateConvertible(
            principal=float(terms.get("principal") or 0.0),
            principal_remaining=float(
                outstanding.get("principal_remaining")
                or terms.get("principal") or 0.0
            ),
            rate=terms.get("rate"),
            conv_price=terms.get("conv_price"),
            convertible_date=_to_date(terms.get("convertible_date")),
            maturity=_to_date(terms.get("maturity")),
            oid_pct=terms.get("oid_pct"),
            counterparty_canonical=counterparty_canonical,
            placement_agent_canonical=placement_agent_canonical,
            known_owners=tuple(terms["known_owners"])
                if isinstance(terms.get("known_owners"), (list, tuple)) else None,
            **common,
        )
    if type_ == "preferred":
        return CreatePreferred(
            count=float(outstanding.get("count") or 0.0),
            series_letter=str(terms.get("series_letter") or ""),
            conv_price=terms.get("conv_price"),
            convertible_date=_to_date(terms.get("convertible_date")),
            maturity=_to_date(terms.get("maturity")),
            stated_value=terms.get("stated_value"),
            liquidation_preference=terms.get("liquidation_preference"),
            dividend_rate=terms.get("dividend_rate"),
            principal_remaining=outstanding.get("principal_remaining"),
            counterparty_canonical=counterparty_canonical,
            placement_agent_canonical=placement_agent_canonical,
            known_owners=tuple(terms["known_owners"])
                if isinstance(terms.get("known_owners"), (list, tuple)) else None,
            **common,
        )
    if type_ == "equity_line":
        return CreateEquityLine(
            capacity_usd=float(terms.get("capacity_usd") or 0.0),
            agreement_date=_to_date(terms.get("agreement_date")),
            agreement_end_date=_to_date(terms.get("agreement_end_date")),
            counterparty_canonical=counterparty_canonical,
            remaining_capacity_usd=outstanding.get("remaining_capacity_usd"),
            drawn_usd=outstanding.get("drawn_usd"),
            placement_agent_canonical=placement_agent_canonical,
            **common,
        )
    if type_ == "s1_offering":
        return CreateS1Offering(
            anticipated_deal_size=float(
                terms.get("anticipated_deal_size") or 0.0
            ),
            warrant_strike=terms.get("warrant_strike"),
            warrant_coverage_pct=terms.get("warrant_coverage_pct"),
            sold_to_date=outstanding.get("sold_to_date"),
            placement_agent_canonical=placement_agent_canonical,
            counterparty_canonical=counterparty_canonical,
            **common,
        )
    if type_ == "equity":
        return CreateEquity(
            count=float(outstanding.get("count") or 0.0),
            price_per_share=float(terms.get("price_per_share") or 0.0),
            closing_date=_to_date(terms.get("closing_date")),
            counterparty_canonical=counterparty_canonical,
            placement_agent_canonical=placement_agent_canonical,
            known_owners=tuple(terms["known_owners"])
                if isinstance(terms.get("known_owners"), (list, tuple)) else None,
            **common,
        )
    raise KeyError(f"unknown instrument_type: {type_!r}")  # unreachable


_AMEND_BY_TYPE: dict[str, type] = {
    "atm": AmendAtm,
    "shelf": AmendShelf,
    "warrant": AmendWarrant,
    "convertible": AmendConvertible,
    "preferred": AmendPreferred,
    "equity_line": AmendEquityLine,
    "s1_offering": AmendS1Offering,
    "equity": AmendEquity,
}


def amend_from_dict(
    *,
    type_: str,
    instrument_id: str,
    field_updates: dict | None = None,
    outstanding_updates: dict | None = None,
    event_date: str | date,
) -> Mutation:
    """Construct the typed Amend* dataclass for a generic sparse-update
    payload (anchor reconciliation's correction mutations). Maps each
    known dict key onto the corresponding named-arg field of the typed
    class; unknown keys are silently dropped (Amend* dataclasses use
    `extra="forbid"` semantics via dataclass field enumeration)."""
    cls = _AMEND_BY_TYPE.get(type_)
    if cls is None:
        raise KeyError(f"unknown instrument_type: {type_!r}")
    fu = field_updates or {}
    ou = outstanding_updates or {}
    ev = _to_date(event_date) or date.today()
    base = {"instrument_id": instrument_id, "event_date": ev}

    if type_ == "atm":
        return AmendAtm(
            **base,
            capacity_usd=fu.get("capacity_usd"),
            agreement_date=_to_date(fu.get("agreement_date")),
            agreement_end_date=_to_date(fu.get("agreement_end_date")),
            placement_agent_canonical=fu.get("placement_agent_canonical"),
            remaining_capacity_usd=ou.get("remaining_capacity_usd"),
            drawn_usd=ou.get("drawn_usd"),
        )
    if type_ == "shelf":
        return AmendShelf(
            **base,
            capacity_usd=fu.get("capacity_usd"),
            remaining_capacity_usd=ou.get("remaining_capacity_usd"),
        )
    if type_ == "equity_line":
        return AmendEquityLine(
            **base,
            capacity_usd=fu.get("capacity_usd"),
            agreement_end_date=_to_date(fu.get("agreement_end_date")),
            remaining_capacity_usd=ou.get("remaining_capacity_usd"),
            drawn_usd=ou.get("drawn_usd"),
        )
    if type_ == "warrant":
        return AmendWarrant(
            **base,
            count=ou.get("count"),
            strike=fu.get("strike"),
            exercisable_date=_to_date(fu.get("exercisable_date")),
            expiration=_to_date(fu.get("expiration")),
            is_pre_funded=fu.get("is_pre_funded"),
            series_letter=fu.get("series_letter"),
            known_owners=tuple(fu["known_owners"])
                if isinstance(fu.get("known_owners"), (list, tuple)) else None,
            issue_date=_to_date(fu.get("issue_date")),
        )
    if type_ == "convertible":
        return AmendConvertible(
            **base,
            principal_remaining=ou.get("principal_remaining"),
            conv_price=fu.get("conv_price"),
            convertible_date=_to_date(fu.get("convertible_date")),
            maturity=_to_date(fu.get("maturity")),
        )
    if type_ == "preferred":
        return AmendPreferred(
            **base,
            count=ou.get("count"),
            conv_price=fu.get("conv_price"),
            conversion_ratio=fu.get("conversion_ratio"),
            convertible_date=_to_date(fu.get("convertible_date")),
            maturity=_to_date(fu.get("maturity")),
            stated_value=fu.get("stated_value"),
            liquidation_preference=fu.get("liquidation_preference"),
            dividend_rate=fu.get("dividend_rate"),
            principal_remaining=ou.get("principal_remaining"),
        )
    if type_ == "s1_offering":
        return AmendS1Offering(
            **base,
            anticipated_deal_size=fu.get("anticipated_deal_size"),
            warrant_strike=fu.get("warrant_strike"),
            warrant_coverage_pct=fu.get("warrant_coverage_pct"),
            sold_to_date=ou.get("sold_to_date"),
            placement_agent_canonical=fu.get("placement_agent_canonical"),
            final_deal_size=fu.get("final_deal_size"),
            final_pricing=fu.get("final_pricing"),
            final_shares_offered=fu.get("final_shares_offered"),
            final_warrant_coverage_pct=fu.get("final_warrant_coverage_pct"),
        )
    if type_ == "equity":
        return AmendEquity(
            **base,
            known_owners=tuple(fu["known_owners"])
                if isinstance(fu.get("known_owners"), (list, tuple)) else None,
        )
    raise KeyError(f"unknown instrument_type: {type_!r}")  # unreachable


# ─── Mutation → JSON dict (for dilution_walk_errors persistence) ─────


def mutation_to_dict(m: Mutation) -> dict:
    """Serialize a typed mutation to a JSON-friendly dict.

    Used by store.py to record the originating mutation alongside an
    apply-time failure or rejection — equivalent to the prior
    Pydantic `model_dump(mode='json')`.
    """
    out: dict = {"kind": m.kind}
    if isinstance(m, CreateMutation):
        out["type"] = m.instrument_type
        out["terms"] = m.terms
        out["outstanding"] = m.outstanding
        out["event_date"] = _iso(m.event_date)
        for attr in ("proposed_id", "counterparty_canonical",
                     "placement_agent_canonical", "descriptor"):
            v = getattr(m, attr, None)
            if v is not None:
                out[attr] = v
    elif isinstance(m, RestateAtm):
        out["type"] = m.instrument_type
        out["predecessor_id"] = m.predecessor_id
        out["capacity_usd"] = m.capacity_usd
        out["supersede_prior"] = m.supersede_prior
        out["event_date"] = _iso(m.event_date)
        if m.agreement_date is not None:
            out["agreement_date"] = _iso(m.agreement_date)
        for attr in ("proposed_id", "placement_agent_canonical"):
            v = getattr(m, attr, None)
            if v is not None:
                out[attr] = v
    elif isinstance(m, AmendMutation):
        out["type"] = m.instrument_type
        out["instrument_id"] = m.instrument_id
        out["field_updates"] = m.field_updates
        out["outstanding_updates"] = m.outstanding_updates
        out["event_date"] = _iso(m.event_date)
    elif isinstance(m, RecordMutation):
        out["instrument_id"] = m.instrument_id
        out["event_kind"] = m.event_kind
        out["fields"] = m.fields
        out["event_date"] = _iso(m.event_date)
    elif isinstance(m, CloseInstrument):
        out["instrument_id"] = m.instrument_id
        out["reason"] = m.reason
        out["replaced_by"] = m.replaced_by
        out["event_date"] = _iso(m.event_date)
    elif isinstance(m, ApplySplit):
        out["post"] = m.post
        out["pre"] = m.pre
        out["direction"] = m.direction
        out["units"] = m.units
        out["effective_date"] = _iso(m.effective_date)
    elif isinstance(m, NoteNoEvent):
        out["reason"] = m.reason
    return out


__all__ = [
    # utility helpers preserved across the rewrite
    "extract_series_letter",
    "safe_date",
    "hoist_nested_payload",
    # Create* dataclasses
    "CreateAtm",
    "CreateShelf",
    "CreateWarrant",
    "CreateConvertible",
    "CreatePreferred",
    "CreateEquityLine",
    "CreateS1Offering",
    "CreateEquity",
    # Restate dataclass (ATM amendment → new card)
    "RestateAtm",
    # Amend* dataclasses
    "AmendAtm", "AmendShelf", "AmendEquityLine",
    "AmendWarrant", "AmendConvertible", "AmendPreferred",
    "AmendS1Offering", "AmendEquity",
    # Record* dataclasses
    "RecordExercise", "RecordConversion", "RecordDrawdown",
    "RecordPartialRedemption", "RecordPartialTermination",
    # Singletons
    "CloseInstrument", "ApplySplit", "NoteNoEvent",
    # Type unions
    "CreateMutation", "AmendMutation", "RecordMutation",
    "Mutation", "ToolMutation", "MutationList",
    # Serializer + factories for internal callers
    "mutation_to_dict",
    "create_from_dict",
    "amend_from_dict",
]
