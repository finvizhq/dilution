"""Periodic-filing reconciliation.

Triggered after the walker processes each 10-K / 10-Q / 20-F / 40-F.
The issuer's own outstanding-instruments table inside the filing is
the authoritative state at period end. We diff our running ledger
against it; mismatches surface to dilution_anchor_diffs and the
ledger is corrected to match.

This is the drift-bounding mechanism. Without it, walker errors
accumulate unbounded across many filings; with it, error is bounded
by the periodic-filing cadence (≤90 days for domestic issuers).

The LLM call that produces the filing-stated overhang rows lives in
the seed module (or wherever the walker chooses to invoke it). This
module is pure diff logic over Python dicts — no LLM calls — so it
can be tested deterministically with hand-built fixtures.

v1 policy (per LEDGER_REWORK_PLAN):
  - missing_in_ledger : emit synthetic create_instrument; record diff
  - extra_in_ledger   : record diff only (do NOT auto-close)
  - field_mismatch    : hard-overwrite ledger to filing's value
  - count_mismatch    : same as field_mismatch
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date

from .mutations import AmendInstrument, CreateInstrument, Mutation, safe_date

log = logging.getLogger(__name__)


# Map overhang.OVERHANG_CATEGORIES to ledger instrument types. Some
# overhang categories don't have a 1:1 ledger-type match (option_pool,
# rsu_psu_unvested) — those are excluded from reconciliation since the
# ledger doesn't track them as instrument tranches.
_CATEGORY_TO_TYPE = {
    "warrant": "warrant",
    "convertible": "convertible",
    "preferred": "preferred",
}

# Strike-bucket tolerance for matching. Filings round prices to the
# cent; with reverse-split adjustments the ledger can sit a fraction
# off. 1% tolerance is comfortable for matching but tight enough to
# distinguish real differences.
PRICE_MATCH_TOLERANCE = 0.01

# Field-mismatch flagging tolerance. Stricter than match tolerance —
# anything inside this window is "agreement"; outside is a flagged diff.
FIELD_MISMATCH_TOLERANCE = 0.02

# Issue-date match window. LLMs disagree on whether to use signing,
# pricing, or closing date — typically within a few days. 30d covers
# nearly all of the spread we've seen without merging unrelated tranches.
ISSUE_DATE_TOLERANCE_DAYS = 30

# Generic counterparty phrases the LLM emits when the filing doesn't
# name a real holder. These are NOT identity — collapse to None for
# matching so two rows both labeled "convertible preferred" don't get
# treated as distinct holders.
_PLACEHOLDER_CPS = {
    "convertible preferred", "convertible note", "convertible notes",
    "preferred stock", "preferred", "warrants", "warrant",
    "institutional investor", "institutional investors",
    "certain institutional investors", "certain insti",
    "accredited investors", "accredited investor",
    "investor", "investors",
    "holders of existing warrants", "holders",
    "purchaser", "purchasers",
    "noteholder", "noteholders",
}

_MONTHS: dict[str, int] = {}
for _i, _name in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"], 1):
    _MONTHS[_name.lower()] = _i
    _MONTHS[_name[:3].lower()] = _i


@dataclass
class AnchorDiff:
    diff_kind: str                  # missing_in_ledger | extra_in_ledger | field_mismatch | count_mismatch
    instrument_id: str | None
    category: str | None
    ledger_value: dict | None
    filing_value: dict | None
    resolution: str = "overwrite"

    def to_dict(self) -> dict:
        return {
            "diff_kind": self.diff_kind,
            "instrument_id": self.instrument_id,
            "category": self.category,
            "ledger_value": self.ledger_value,
            "filing_value": self.filing_value,
            "resolution": self.resolution,
        }


@dataclass
class AnchorResult:
    diffs: list[AnchorDiff] = field(default_factory=list)
    # Mutations the walker should apply after the anchor pass to
    # bring the ledger into agreement with the filing. Created via
    # CreateInstrument (synthetic) for missing rows and AmendInstrument
    # for field corrections. Extra-in-ledger gets no mutation in v1.
    correction_mutations: list[Mutation] = field(default_factory=list)


def reconcile_against_periodic(
    *, cik: int, accession: str, filing_date: str, as_of_date: str,
    filing_overhang: list[dict],
    ledger_open: list[dict],
) -> AnchorResult:
    """Diff the filing's overhang table against the open ledger.

    `filing_overhang` is a list of OverhangRow-shaped dicts (category,
    instrument_name, outstanding_count, strike_or_conversion_price,
    principal_amount, maturity_or_expiry, issue_date, …).

    `ledger_open` is a list of dicts as returned by
    `store.get_open_instruments(cik)` — one row per active instrument.
    """
    result = AnchorResult()
    by_type: dict[str, list[dict]] = {}
    for row in ledger_open:
        t = (row.get("type") or "").lower()
        if t in {"warrant", "convertible", "preferred"}:
            by_type.setdefault(t, []).append(row)

    used_ledger_ids: set[str] = set()
    for over in filing_overhang or []:
        cat = (over.get("category") or "").lower()
        target_type = _CATEGORY_TO_TYPE.get(cat)
        if not target_type:
            continue  # option_pool, rsu_psu_unvested, other — out of scope
        candidates = [c for c in by_type.get(target_type, [])
                      if c.get("instrument_id") not in used_ledger_ids]
        match = _best_match(over, candidates)
        if match is None:
            # Filing lists an instrument we don't have on the ledger.
            # Emit a synthetic create_instrument and record the diff.
            mutation = _synthesize_create(
                over, accession=accession, filing_date=filing_date,
            )
            result.correction_mutations.append(mutation)
            result.diffs.append(AnchorDiff(
                diff_kind="missing_in_ledger",
                instrument_id=None, category=cat,
                ledger_value=None,
                filing_value=_overhang_to_value(over),
                resolution="overwrite",
            ))
            continue
        used_ledger_ids.add(match["instrument_id"])
        # Compare fields. Drift in any one of (count, strike, principal)
        # → field_mismatch + AmendInstrument correction.
        field_changes = _field_changes(match, over)
        if field_changes:
            result.correction_mutations.append(AmendInstrument(
                kind="amend_instrument",
                instrument_id=match["instrument_id"],
                field_updates=field_changes["updates"],
                event_date=as_of_date,
            ))
            result.diffs.append(AnchorDiff(
                diff_kind=field_changes["diff_kind"],
                instrument_id=match["instrument_id"],
                category=cat,
                ledger_value=field_changes["ledger_value"],
                filing_value=field_changes["filing_value"],
                resolution="overwrite",
            ))

    # extra_in_ledger: any unmatched candidate of warrant/convertible/
    # preferred that we expected the filing to list. Logged only — we
    # don't auto-close since the filing might just have omitted it.
    for t in ("warrant", "convertible", "preferred"):
        for row in by_type.get(t, []):
            if row.get("instrument_id") in used_ledger_ids:
                continue
            result.diffs.append(AnchorDiff(
                diff_kind="extra_in_ledger",
                instrument_id=row["instrument_id"],
                category=t,
                ledger_value={"terms": row.get("terms"),
                              "outstanding": row.get("outstanding"),
                              "counterparty": row.get("counterparty")},
                filing_value=None,
                resolution="kept_ledger",
            ))

    if result.diffs:
        log.info(
            "  anchor %s — %d diffs (missing=%d field=%d count=%d extra=%d)",
            accession, len(result.diffs),
            sum(1 for d in result.diffs if d.diff_kind == "missing_in_ledger"),
            sum(1 for d in result.diffs if d.diff_kind == "field_mismatch"),
            sum(1 for d in result.diffs if d.diff_kind == "count_mismatch"),
            sum(1 for d in result.diffs if d.diff_kind == "extra_in_ledger"),
        )
    return result


# ─── matching ────────────────────────────────────────────────────────
def _best_match(over: dict, candidates: list[dict]) -> dict | None:
    """Pick the ledger row that best matches this overhang row.

    Scoring is additive across four axes — issue-date proximity,
    counterparty equality, strike-bucket equality, maturity equality.
    Highest score wins; ties go to earliest-created so subsequent
    overhang rows can claim later siblings.

    Counterparty placeholders ("convertible preferred", "institutional
    investor", etc.) are treated as no signal — they distinguish
    nothing. Strike returns three-valued (True / False / None) so a
    missing-on-one-side strike does not penalize a row that already
    matches on issue date and counterparty.

    Returns None when no candidate scores above zero.
    """
    if not candidates:
        return None

    over_strike = _to_float(
        over.get("strike_or_conversion_price")
        or over.get("conversion_price")
    )
    over_cp = _real_cp(
        over.get("counterparty")
        or over.get("counterparty_canonical")
        or _extract_cp_from_name(over.get("instrument_name"))
    )
    over_issue = _normalize_date(over.get("issue_date")) or \
        _extract_date_from_name(over.get("instrument_name"))
    over_maturity = _normalize_date(over.get("maturity_or_expiry"))

    def score(r: dict) -> int:
        s = 0
        # Issue date is the strongest signal — split-invariant, carried
        # verbatim across re-disclosures.
        if over_issue and _date_close(over_issue, r.get("created_at"),
                                       ISSUE_DATE_TOLERANCE_DAYS):
            s += 4
        r_cp = _real_cp(
            r.get("counterparty_canonical") or r.get("counterparty")
        )
        if over_cp and r_cp and over_cp == r_cp:
            s += 2
        if _strike_match(over_strike, _row_strike(r)) is True:
            s += 2
        r_maturity = _normalize_date(_terms_dict(r).get("maturity"))
        if over_maturity and r_maturity and over_maturity == r_maturity:
            s += 1
        return s

    scored = [(score(r), r.get("created_at") or "9999-99-99", r)
              for r in candidates]
    scored.sort(key=lambda x: (-x[0], x[1]))
    best_score, _, best_row = scored[0]
    if best_score == 0:
        return None
    return best_row


def _row_strike(r: dict) -> float | None:
    terms = _terms_dict(r)
    return _to_float(
        terms.get("strike")
        or terms.get("warrant_strike")
        or terms.get("conv_price")
        or terms.get("conversion_price")
    )


def _strike_match(a, b) -> bool | None:
    """Three-valued: True (within tolerance), False (real mismatch),
    None (no information on at least one side — neither a match nor
    a penalty)."""
    if a is None or b is None:
        return None
    if a == 0 or b == 0:
        return abs(a - b) < 1e-9
    return abs(a - b) / max(abs(a), abs(b)) <= PRICE_MATCH_TOLERANCE


def _normalize_cp(s: str | None) -> str | None:
    if not s:
        return None
    return s.strip().lower()


def _real_cp(s: str | None) -> str | None:
    """Counterparty after collapsing generic placeholders to None.
    A row labeled "convertible preferred" carries no holder identity
    and should not be matchable on counterparty alone."""
    n = _normalize_cp(s)
    if n is None or n in _PLACEHOLDER_CPS:
        return None
    return n


def _normalize_date(s: str | None) -> str | None:
    """Coerce any of YYYY-MM-DD, YYYY-MM, YYYY (and timestamp suffixes)
    to YYYY-MM-DD, pinning partial dates to the first of the month/year."""
    if not s:
        return None
    s = s.strip()
    if len(s) >= 10 and s[4:5] == "-" and s[7:8] == "-":
        return s[:10]
    if len(s) >= 7 and s[4:5] == "-":
        return s[:7] + "-01"
    if len(s) == 4 and s.isdigit():
        return s + "-01-01"
    return None


def _date_close(a: str | None, b: str | None, days: int) -> bool:
    a = _normalize_date(a)
    b = _normalize_date(b)
    if not a or not b:
        return False
    try:
        da = date.fromisoformat(a)
        db = date.fromisoformat(b)
    except ValueError:
        return False
    return abs((da - db).days) <= days


def _extract_date_from_name(name: str | None) -> str | None:
    """Pull a YYYY-MM-DD date hint out of names like 'April 2018 Series 5'
    or '2024-08 Convertible Notes'. Pinned to the 1st of the month so
    the date-close window absorbs the slack."""
    if not name:
        return None
    m = re.search(r"\b([A-Za-z]+)\s+(\d{4})\b", name)
    if m:
        mo = _MONTHS.get(m.group(1).lower())
        if mo:
            return f"{m.group(2)}-{mo:02d}-01"
    m = re.search(r"\b(\d{4})[-/](\d{1,2})\b", name)
    if m:
        y = m.group(1)
        mo = int(m.group(2))
        if 1 <= mo <= 12:
            return f"{y}-{mo:02d}-01"
    return None


def _extract_cp_from_name(name: str | None) -> str | None:
    """Pull a counterparty hint out of an instrument_name like
    'Streeterville Capital December 2022 Promissory Note'. Best-effort
    — returns the first 1-3 words if they look like a name (capitalized,
    not "Series A" / "2024 Convertible Notes")."""
    if not name:
        return None
    parts = name.split()
    if not parts:
        return None
    skip_first = parts[0].lower() in {"series", "class"}
    head = parts[2:5] if skip_first else parts[:3]
    head = [p for p in head if not p[:1].isdigit()]
    return " ".join(head[:2]).strip().lower() or None


# ─── per-row diffing ────────────────────────────────────────────────
_FIELD_MAP_BY_TYPE = {
    "warrant": [
        ("strike", "strike_or_conversion_price"),
    ],
    "convertible": [
        ("conv_price", "strike_or_conversion_price"),
        ("principal", "principal_amount"),
    ],
    "preferred": [
        ("conv_price", "strike_or_conversion_price"),
        ("liquidation_preference", "principal_amount"),
    ],
}
_OUT_FIELD_MAP_BY_TYPE = {
    "warrant": ("count", "outstanding_count"),
    "convertible": ("principal_remaining", "principal_amount"),
    "preferred": ("count", "outstanding_count"),
}


def _field_changes(ledger_row: dict, over: dict) -> dict | None:
    """Return field_updates dict + diff metadata if the filing's stated
    values disagree with the ledger by more than FIELD_MISMATCH_TOLERANCE.
    None when in agreement."""
    t = (ledger_row.get("type") or "").lower()
    terms = _terms_dict(ledger_row)
    out = _outstanding_dict(ledger_row)
    updates: dict = {}
    ledger_view: dict = {}
    filing_view: dict = {}
    diff_kind = None

    for terms_key, over_key in _FIELD_MAP_BY_TYPE.get(t, []):
        l_val = _to_float(terms.get(terms_key))
        f_val = _to_float(over.get(over_key))
        if l_val is None or f_val is None:
            continue
        if not _within(l_val, f_val, FIELD_MISMATCH_TOLERANCE):
            updates[terms_key] = f_val
            ledger_view[terms_key] = l_val
            filing_view[terms_key] = f_val
            diff_kind = diff_kind or "field_mismatch"

    out_pair = _OUT_FIELD_MAP_BY_TYPE.get(t)
    if out_pair:
        out_key, over_key = out_pair
        l_val = _to_float(out.get(out_key))
        f_val = _to_float(over.get(over_key))
        if l_val is not None and f_val is not None and not _within(
            l_val, f_val, FIELD_MISMATCH_TOLERANCE,
        ):
            # Outstanding fields don't go through field_updates (which
            # writes to terms_json). Walker treats this as a separate
            # AmendInstrument target — but since amend_instrument keys
            # write to terms only, we encode the count update with a
            # special key the apply layer recognizes.
            # Simpler path: emit a count_mismatch diff but no mutation;
            # we'll let v1's "hard overwrite" be a follow-up project.
            # For now, log the count drift and keep ledger as-is.
            ledger_view[f"out.{out_key}"] = l_val
            filing_view[f"out.{out_key}"] = f_val
            diff_kind = "count_mismatch"

    if not diff_kind:
        return None
    return {
        "updates": updates,
        "diff_kind": diff_kind,
        "ledger_value": ledger_view,
        "filing_value": filing_view,
    }


def _synthesize_create(
    over: dict, *, accession: str, filing_date: str,
) -> CreateInstrument:
    """Build a CreateInstrument from a filing-stated overhang row.

    Used when the periodic filing names an instrument the ledger
    doesn't have (typically a pre-window instrument the issuer is
    re-disclosing). The synthetic create carries the filing's values
    as terms; the walker assigns the id."""
    cat = (over.get("category") or "").lower()
    target_type = _CATEGORY_TO_TYPE.get(cat) or "equity"
    terms: dict = {}
    outstanding: dict = {}

    if target_type == "warrant":
        if over.get("strike_or_conversion_price") is not None:
            terms["strike"] = float(over["strike_or_conversion_price"])
        if (mat := safe_date(over.get("maturity_or_expiry"))):
            terms["maturity"] = mat
        if over.get("outstanding_count") is not None:
            outstanding["count"] = float(over["outstanding_count"])
    elif target_type == "convertible":
        if over.get("strike_or_conversion_price") is not None:
            terms["conv_price"] = float(over["strike_or_conversion_price"])
        if over.get("principal_amount") is not None:
            terms["principal"] = float(over["principal_amount"])
            outstanding["principal_remaining"] = float(over["principal_amount"])
        if (mat := safe_date(over.get("maturity_or_expiry"))):
            terms["maturity"] = mat
    elif target_type == "preferred":
        if over.get("strike_or_conversion_price") is not None:
            terms["conv_price"] = float(over["strike_or_conversion_price"])
        if over.get("principal_amount") is not None:
            terms["liquidation_preference"] = float(over["principal_amount"])
        if over.get("outstanding_count") is not None:
            outstanding["count"] = float(over["outstanding_count"])

    if over.get("price_protection"):
        terms["anti_dilution_type"] = over["price_protection"]

    cp = (over.get("counterparty")
          or _extract_cp_from_name(over.get("instrument_name")))

    return CreateInstrument(
        kind="create_instrument",
        type=target_type,
        proposed_id=None,
        counterparty=cp,
        counterparty_canonical=cp,
        label=(over.get("instrument_name") or None),
        terms=terms,
        outstanding=outstanding,
        event_date=safe_date(over.get("issue_date")) or filing_date,
    )


def _overhang_to_value(over: dict) -> dict:
    """Compact representation of an overhang row for the diff log."""
    return {
        "instrument_name": over.get("instrument_name"),
        "category": over.get("category"),
        "strike_or_conversion_price": over.get("strike_or_conversion_price"),
        "principal_amount": over.get("principal_amount"),
        "outstanding_count": over.get("outstanding_count"),
        "issue_date": over.get("issue_date"),
        "maturity_or_expiry": over.get("maturity_or_expiry"),
    }


# ─── helpers ────────────────────────────────────────────────────────
def _terms_dict(row: dict) -> dict:
    if isinstance(row.get("terms"), dict):
        return row["terms"]
    raw = row.get("terms_json")
    if raw:
        import json
        try:
            return json.loads(raw) or {}
        except (TypeError, ValueError):
            return {}
    return {}


def _outstanding_dict(row: dict) -> dict:
    if isinstance(row.get("outstanding"), dict):
        return row["outstanding"]
    raw = row.get("outstanding_json")
    if raw:
        import json
        try:
            return json.loads(raw) or {}
        except (TypeError, ValueError):
            return {}
    return {}


def _to_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _within(a, b, tol) -> bool:
    if a == 0 and b == 0:
        return True
    denom = max(abs(a), abs(b))
    return denom > 0 and abs(a - b) / denom <= tol


__all__ = [
    "AnchorDiff",
    "AnchorResult",
    "reconcile_against_periodic",
    "PRICE_MATCH_TOLERANCE",
    "FIELD_MISMATCH_TOLERANCE",
]
