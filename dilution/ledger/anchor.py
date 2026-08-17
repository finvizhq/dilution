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

v1 policy (per LEDGER_REWORK_PLAN, with Tier 1 auto-close):
  - missing_in_ledger : emit synthetic create_instrument; record diff
  - extra_in_ledger   : record diff; auto-close when an objective signal
                        confirms the row is dead (expiration / maturity
                        past as_of_date, or outstanding already 0).
                        Without such a signal, keep the row open — a
                        single anchor miss is not proof of closure
                        (the filing may simply have omitted it).
  - field_mismatch    : amend_instrument(field_updates=…) — overwrites
                        terms_json with the filing's value.
  - count_mismatch    : amend_instrument(outstanding_updates=…) —
                        overwrites outstanding_json with the filing's
                        value. This is the count-drift correction:
                        outstanding count at period end is the issuer's
                        authoritative number and the running ledger
                        must match it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date

from .mutations import (
    CloseInstrument,
    Mutation,
    amend_from_dict,
    create_from_dict,
    extract_series_letter,
    safe_date,
)
from .validate import _SECURITY_VOCAB_TOKENS as _VOCAB_ONLY_TOKENS


def _as_date(s: str | date | None) -> date:
    """Coerce an ISO string to date, falling back to today on failure.
    Anchor never constructs without an as_of/filing date in practice,
    so the fallback exists only to satisfy the typed-dataclass contract
    when a caller passes a malformed string."""
    if isinstance(s, date):
        return s
    if isinstance(s, str):
        try:
            return date.fromisoformat(s[:10])
        except ValueError:
            pass
    return date.today()

log = logging.getLogger(__name__)


# Map overhang.OVERHANG_CATEGORIES to ledger instrument types. Some
# overhang categories don't have a 1:1 ledger-type match (option_pool,
# rsu_psu_unvested) — those are excluded from reconciliation since the
# ledger doesn't track them as instrument tranches.
#
# v3 adds shelf-family types. Identity matching for these uses different
# keys than the warrant/convertible/preferred path (see _match_shelf
# and _match_atm_or_eloc) — file_number for shelves, (counterparty,
# agreement_date) for ATMs/ELOCs.
_CATEGORY_TO_TYPE = {
    "warrant": "warrant",
    "convertible": "convertible",
    "preferred": "preferred",
    "shelf": "shelf",
    "atm": "atm",
    "equity_line": "equity_line",
}

# Tranche-identity-matched ledger types. Use _best_match (strike +
# issue_date + counterparty scoring).
_TRANCHE_TYPES = frozenset({"warrant", "convertible", "preferred"})

# Capacity-matched ledger types. Use the per-type identity matchers
# (_match_shelf / _match_atm_or_eloc) keyed on file_number or
# (counterparty, agreement_date).
_CAPACITY_TYPES = frozenset({"shelf", "atm", "equity_line"})

# Agreement-date tolerance for ATM / ELOC identity matching. Two
# successive agreements with the same banker / investor are common
# (XTIA: two Maxim ATMs 11 months apart; Yorkville re-ups every 12-18
# months). 30 days is comfortably below the inter-agreement gap while
# absorbing the spread between filing's stated date and the agreement's
# actual execution date.
AGREEMENT_DATE_TOLERANCE_DAYS = 30

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

# Window of "recent drawdown activity" used to veto the periodic
# overhang's is_terminated auto-close for ATMs/ELOCs. If the instrument
# drew shares within this many days BEFORE the as_of_date, treat the
# overhang's termination flag as unreliable (likely a stale
# agreement_end_date). 270d covers two quarterly cycles plus slack —
# long enough that an actively-used ATM won't be killed by a single
# filing's stale term-end disclosure, short enough that a genuinely
# idle ATM gets its termination honored.
RECENT_ACTIVITY_DAYS = 270

# s1_offering staleness-close window. A registered S-1 takedown
# (registered direct / follow-on / IPO) is a one-shot consummated
# event: no maturity, no expiration, and no 3-year Rule 415 life, so
# none of _confident_close_reason's date checks ever fire — and it is
# not an overhang category, so the matching loop never confirms it
# either. Left untouched it sits status='active' forever (DB-wide as of
# 2026-06: 77 of 106 active S-1s unseen in a filing for >1y, oldest
# ~6y). Reap it once it has gone unmentioned this long, anchored on
# last_seen_date so an offering the walker keeps re-disclosing keeps
# resetting the clock. 540d ≈ 18 months ≈ six missed periodics:
# conservative by house style (cf. the 3-year shelf/ATM windows) since
# a false close erases a real card, while still clearing the backlog.
S1_OFFERING_STALE_CLOSE_DAYS = 540

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


# Keys already conveyed by the diff head (or too noisy for a one-line
# audit string) — dropped from the rendered value. `category` is printed
# in the head as `[category]`; `instrument_name` is long and the head
# already names the instrument by id.
_DIFF_SKIP_KEYS = frozenset({"category", "instrument_name"})


def _fmt_diff_value(v: dict | None) -> str:
    """Render a diff's ledger/filing value dict as a compact, auditable
    `key=value` string.

    Handles every shape the diff producers emit:
      - field/count mismatch → flat prefixed scalars (terms.conv_price,
        out.count) — printed verbatim.
      - missing_in_ledger     → raw overhang fields
        (strike_or_conversion_price, outstanding_count, …).
      - extra_in_ledger       → nested {terms:{…}, outstanding:{…},
        counterparty}; terms/outstanding are flattened one level to
        terms.* / out.* so they read the same as the mismatch keys.
      - is_terminated close    → {status} / {is_terminated}.

    The previous implementation filtered against a fixed allow-list that
    knew only un-prefixed keys, so the one diff kind carrying prefixed
    keys (field/count mismatch — the actual drift-bounding correction)
    always rendered as "-". That made the mechanism unauditable from the
    log; this prints whatever the diff actually carries instead.
    """
    if not v:
        return "-"
    bits: list[str] = []
    for k, val in v.items():
        if k in _DIFF_SKIP_KEYS or val is None:
            continue
        if isinstance(val, dict):
            # extra_in_ledger nests the row's terms/outstanding dicts;
            # flatten one level so they read like the mismatch keys.
            prefix = "out" if k == "outstanding" else k
            for sub_k, sub_v in val.items():
                if sub_v is not None:
                    bits.append(f"{prefix}.{sub_k}={sub_v}")
        else:
            bits.append(f"{k}={val}")
    return ", ".join(bits) if bits else "-"


def _fmt_diff(d: "AnchorDiff") -> str:
    head = f"{d.diff_kind} {d.instrument_id or '(new)'}"
    if d.category:
        head += f" [{d.category}]"
    if d.diff_kind == "missing_in_ledger":
        return f"{head} filing={_fmt_diff_value(d.filing_value)}"
    if d.diff_kind == "extra_in_ledger":
        return f"{head} ledger={_fmt_diff_value(d.ledger_value)}"
    return (f"{head} ledger={_fmt_diff_value(d.ledger_value)} "
            f"filing={_fmt_diff_value(d.filing_value)}")


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
    stated_note_balances: list[dict] | None = None,
) -> AnchorResult:
    """Diff the filing's overhang table against the open ledger.

    `filing_overhang` is a list of OverhangRow-shaped dicts (category,
    instrument_name, outstanding_count, strike_or_conversion_price,
    principal_amount, maturity_or_expiry, issue_date, …).

    `ledger_open` is a list of dicts as returned by
    `store.get_open_instruments(cik)` — one row per active instrument.
    """
    result = AnchorResult()
    # Filter out instruments minted AFTER the filing being reconciled.
    # The filing's overhang table can't describe an instrument that
    # didn't exist yet. Without this guard, incremental re-walks where
    # a later filing has already created a new tranche cause the anchor
    # matcher to map the historical disclosure to the future instrument
    # — CGEN's 14:15 re-walk amended ATM-2119 (May-2026 Leerink, $100M)
    # down to $50M because the 20-F filed 2024-03-05 was describing the
    # then-current $50M SVB ATM and matched against ATM-2119 by
    # counterparty/date proximity.
    filing_d = (filing_date or "")[:10]
    def _eligible(row: dict) -> bool:
        if not filing_d:
            return True
        created = (row.get("created_at") or "")[:10]
        return not created or created <= filing_d
    by_type: dict[str, list[dict]] = {}
    for row in ledger_open:
        if not _eligible(row):
            continue
        t = (row.get("type") or "").lower()
        # s1_offering is loaded but never reconciled against the overhang
        # (it has no overhang category); the staleness reaper after the
        # extra_in_ledger loop is its only consumer.
        if (t in _TRANCHE_TYPES or t in _CAPACITY_TYPES
                or t == "s1_offering"):
            by_type.setdefault(t, []).append(row)

    # Shelf-family identity matching also considers CLOSED rows so a
    # filing's re-disclosure of a historical ATM / shelf / ELOC doesn't
    # cause synthetic re-creation. Filings routinely discuss prior
    # terminated programs in their liquidity sections; without this,
    # each such re-disclosure would spawn a duplicate row. Matched-to-
    # closed produces no correction mutation — the closed row is frozen.
    by_type_closed: dict[str, list[dict]] = {}
    for row in _load_closed_shelf_family(cik):
        if not _eligible(row):
            continue
        t = (row.get("type") or "").lower()
        if t in _CAPACITY_TYPES:
            by_type_closed.setdefault(t, []).append(row)

    # Pre-fetch shelf file_numbers in one query (shelves carry their
    # file_number on the filings table, not in terms_json). The lookup
    # is per-create_accession; cache it for the whole reconcile pass.
    # Pool union: active + closed shelves both contribute identity.
    all_shelves = (
        by_type.get("shelf", []) + by_type_closed.get("shelf", [])
    )
    shelf_file_numbers = _load_shelf_file_numbers(cik, all_shelves)

    # Drawdowns the walker has already booked with event_date > as_of_date
    # (typically Subsequent Events disclosed in the same 10-K/10-Q whose
    # overhang table reports the FY/quarter-end balance). The running
    # ledger's drawn_usd includes these, but the filing's overhang
    # doesn't — back them out per-instrument before the drift check so
    # we don't zero out drawdowns that were correctly booked.
    post_asof_drawn = _load_post_asof_drawn(cik, as_of_date)
    # Per-instrument most-recent drawdown event_date across ALL booked
    # drawdowns (not just post-as_of). Used by the is_terminated guard
    # below — see comment there.
    last_drawdown_date = _load_max_drawdown_date(cik)

    # Tranche matching is decided GLOBALLY, before the loop. Matching each
    # overhang row in turn against whatever is still unclaimed is
    # first-fit, and first-fit strands rows: a mediocre early pairing takes
    # a candidate that a later line matches better, the later line finds
    # nothing, and the reconciler synthesizes a DUPLICATE. See
    # _assign_matches for the CELU case that motivated this. Shelf / ATM /
    # equity-line keep their own identity matchers (file_number, agreement
    # party) and stay per-row.
    tranche_assignment: dict[str, dict[int, dict]] = {}
    overs_by_type: dict[str, list[tuple[int, dict]]] = {}
    for idx, over in enumerate(filing_overhang or []):
        t = _CATEGORY_TO_TYPE.get((over.get("category") or "").lower())
        if t in _TRANCHE_TYPES:
            overs_by_type.setdefault(t, []).append((idx, over))
    for t, indexed in overs_by_type.items():
        assigned = _assign_matches([o for _, o in indexed],
                                   by_type.get(t, []))
        tranche_assignment[t] = {
            indexed[local_i][0]: row for local_i, row in assigned.items()
        }

    used_ledger_ids: set[str] = set()
    for over_index, over in enumerate(filing_overhang or []):
        cat = (over.get("category") or "").lower()
        target_type = _CATEGORY_TO_TYPE.get(cat)
        if not target_type:
            continue  # option_pool, rsu_psu_unvested, other — out of scope
        candidates = [c for c in by_type.get(target_type, [])
                      if c.get("instrument_id") not in used_ledger_ids]
        if target_type in _TRANCHE_TYPES:
            match = tranche_assignment.get(target_type, {}).get(over_index)
            # Defensive: the global pass cannot hand out the same row
            # twice, but a row consumed by the closed-family path below
            # would still be off-limits.
            if match is not None and match.get(
                    "instrument_id") in used_ledger_ids:
                match = None
        elif target_type == "shelf":
            match = _match_shelf(over, candidates, shelf_file_numbers)
        elif target_type in ("atm", "equity_line"):
            match = _match_atm_or_eloc(over, candidates, target_type)
        else:
            match = None

        # No active match — before synthesizing, check whether this
        # is a re-disclosure of a CLOSED shelf-family instrument. If
        # so, consume the id (so it doesn't fall into extra_in_ledger)
        # but emit no correction mutation. The closed row is frozen.
        if match is None and target_type in _CAPACITY_TYPES:
            closed_candidates = [
                c for c in by_type_closed.get(target_type, [])
                if c.get("instrument_id") not in used_ledger_ids
            ]
            if target_type == "shelf":
                closed_hit = _match_shelf(
                    over, closed_candidates, shelf_file_numbers,
                )
            else:
                closed_hit = _match_atm_or_eloc(
                    over, closed_candidates, target_type,
                )
            if closed_hit is not None:
                used_ledger_ids.add(closed_hit["instrument_id"])
                log.debug(
                    "anchor %s — overhang re-discloses closed %s %s; "
                    "no action",
                    accession, target_type, closed_hit["instrument_id"],
                )
                continue

        # Tranche shadow-match: when _best_match found nothing among
        # unused candidates, check whether this overhang row is a
        # sub-bucket of an already-consumed tranche (same strike, no
        # maturity contradiction). Without this guard, a periodic that
        # splits one tranche's count across multiple XBRL rows produces
        # N-1 spurious create_instrument synths after the first row
        # claims the matching ledger id.
        if match is None and target_type in _TRANCHE_TYPES:
            # Pool = rows consumed so far PLUS every row the global pass
            # has promised to another overhang line. Without the second
            # half this would be order-dependent all over again: a
            # sub-tranche line appearing BEFORE its bound sibling would
            # see an empty pool and synthesize anyway.
            claimed = set(used_ledger_ids) | {
                r.get("instrument_id")
                for r in tranche_assignment.get(target_type, {}).values()
            }
            used_pool = [r for r in by_type.get(target_type, [])
                         if r.get("instrument_id") in claimed]
            shadow = _subsumed_by_used_tranche(over, used_pool)
            if shadow is not None:
                log.debug(
                    "anchor %s — overhang sub-bucket of already-"
                    "matched %s %s; no synth",
                    accession, target_type, shadow["instrument_id"],
                )
                result.diffs.append(AnchorDiff(
                    diff_kind="missing_in_ledger",
                    instrument_id=None, category=cat,
                    ledger_value=None,
                    filing_value=_overhang_to_value(over),
                    resolution=f"redisclosure:{shadow['instrument_id']}",
                ))
                continue

        # Empty-shell guard (atm/equity_line): a syndicate filing emits an
        # itemized ATM row PLUS a bare aggregate row with no identity and
        # no economics. The itemized row binds via agent-overlap; the bare
        # row would still synthesize into an empty-shell phantom (FCEL
        # ATM-2232). Drop it — it can't represent a distinct program.
        if (match is None and target_type in ("atm", "equity_line")
                and _is_empty_capacity_overhang(over)):
            log.debug(
                "anchor %s — empty %s overhang row (no agent/date/"
                "capacity/economics); skipping synth",
                accession, target_type,
            )
            continue

        # Phantom equity_line guard: the overhang extractor occasionally
        # tags a warrant-exercise line (a "Common Stock Purchase Option"
        # already tracked as a WARRANT) as category='equity_line' with
        # capacity 0 and a "drawn" that is really warrant-exercise cash.
        # Two deterministic kills, neither of which can touch a real ELOC:
        # (a) a capacity-0/None equity_line is not a facility; (b) a
        # warrant on the same counterparty within ±30d already represents
        # it (IQST EL-048: ADI Funding, cap=0, same date as warrant).
        if match is None and target_type == "equity_line":
            cap = _to_float(over.get("total_capacity_usd"))
            over_inv = _normalize_cp(over.get("investor"))
            over_dt = (_normalize_date(over.get("agreement_date"))
                       or _normalize_date(over.get("issue_date")))
            if (not cap or cap <= 0) or _warrant_covers_same_party(
                    cik, over_inv, over_dt):
                log.info(
                    "anchor %s — suppressed phantom equity_line synth "
                    "(cap=%s investor=%s date=%s): capacity-0 or already "
                    "a warrant on same party/date",
                    accession, cap, over_inv, over_dt,
                )
                result.diffs.append(AnchorDiff(
                    diff_kind="missing_in_ledger", instrument_id=None,
                    category=cat, ledger_value=None,
                    filing_value=_overhang_to_value(over),
                    resolution="suppressed:phantom_equity_line"))
                continue

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
        # Compare fields. Drift in any one of (count, strike, principal,
        # drawn_usd, capacity_usd) → AmendInstrument correction carrying
        # terms and/or outstanding updates depending on which axis drifted.
        field_changes = _field_changes(
            match, over,
            post_asof_drawn_usd=post_asof_drawn.get(
                match["instrument_id"], 0.0,
            ),
        )
        if field_changes:
            result.correction_mutations.append(amend_from_dict(
                type_=match.get("type") or target_type,
                instrument_id=match["instrument_id"],
                field_updates=field_changes["terms_updates"],
                outstanding_updates=field_changes["out_updates"],
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

        # Filing explicitly confirms termination → close the row even
        # though it appeared in the overhang table. Applies to
        # shelf-family only (warrants / convertibles / preferreds emit
        # their close-out via the standard walker path, not via
        # is_terminated). Validate.py's close_with_outstanding guard
        # requires the relevant outstanding field to be zero, so prepend
        # a zeroing amend when needed.
        #
        # Self-contradiction guard for ATM/ELOC: refuse the periodic-
        # overhang is_terminated auto-close when recent drawdown activity
        # shows the agreement is alive. Two signals:
        #   (a) same filing booked a drawdown AFTER as_of_date
        #       (Subsequent-Events sale); or
        #   (b) any booked drawdown within RECENT_ACTIVITY_DAYS BEFORE
        #       as_of_date — the EDA was actively used during the
        #       reporting period, so a "Term: <date>" line in the
        #       overhang is almost certainly a stale agreement_end_date
        #       (extensions are routine; the LLM treats the originally-
        #       stated term-end as termination).
        # True terminations get announced in 8-Ks, which the walker
        # already processes through the standard close path; if a future
        # filing's overhang drops the row entirely, the extra_in_ledger
        # path closes it via _confident_close_reason. Skipping here is
        # safe — at worst a zombie stays open one extra cycle.
        # Observed: XTIA's FY2024 10-K and Q1-2025 10-Q both flagged the
        # ATM as is_terminated=True; the same filings booked drawdowns
        # 3-4 months apart through the reporting period.
        if (over.get("is_terminated") is True
                and target_type in ("atm", "equity_line")
                and match is not None):
            iid = match["instrument_id"]
            post_drawn_here = post_asof_drawn.get(iid, 0.0)
            last_dd = last_drawdown_date.get(iid)
            recent_here = False
            if last_dd:
                try:
                    last_d = _as_date(last_dd)
                    asof_d = _as_date(as_of_date)
                    delta = (asof_d - last_d).days
                    if -365 <= delta <= RECENT_ACTIVITY_DAYS:
                        recent_here = True
                except Exception:
                    pass
            if post_drawn_here > 0 or recent_here:
                log.info(
                    "anchor %s — overhang is_terminated=true on %s "
                    "contradicted by recent drawdown activity "
                    "(post_asof=$%.0f, last_dd=%s); skipping auto-close",
                    accession, iid, post_drawn_here, last_dd or "none",
                )
                continue
        if (over.get("is_terminated") is True
                and target_type in ("shelf", "atm", "equity_line")):
            close_reason = ("expired" if target_type == "shelf"
                            else "terminated")
            zeroing = _zero_outstanding_for_close(match, target_type)
            if zeroing:
                result.correction_mutations.append(amend_from_dict(
                    type_=match.get("type") or target_type,
                    instrument_id=match["instrument_id"],
                    outstanding_updates=zeroing,
                    event_date=as_of_date,
                ))
            result.correction_mutations.append(CloseInstrument(
                instrument_id=match["instrument_id"],
                reason=close_reason,
                event_date=_as_date(as_of_date),
            ))
            result.diffs.append(AnchorDiff(
                diff_kind="field_mismatch",
                instrument_id=match["instrument_id"],
                category=cat,
                ledger_value={"status": match.get("status")},
                filing_value={"is_terminated": True},
                resolution=f"closed:{close_reason}",
            ))

    # extra_in_ledger: any unmatched candidate of any reconciled type.
    # We auto-close ONLY when an independent signal confirms the row is
    # dead — expiration/maturity past as_of_date, outstanding zero,
    # shelf past Rule 415 3-year window, ATM/ELOC past 3y. Anything
    # else stays in the ledger as kept_ledger.
    #
    # We do NOT close rows just because the filing's overhang table
    # didn't itemize them. ~64% of periodic filings show at least one
    # extra_in_ledger warrant, and the vast majority of those are
    # benign — aggregate-summary tables ("Outstanding Warrants: 36M @
    # $1.81 weighted-avg") and partial-itemization disclosures that
    # silently drop small tranches are common in microcap 10-K/10-Q
    # narratives. Closing on filing silence was destroying live cards
    # (XTIA Q1 2026: anchor closed 15 itemized warrants because the
    # 10-Q only printed a single weighted-avg line). True terminations
    # have an explicit narrative footprint (exchange / repurchase /
    # redemption / inducement) that the walker catches from event
    # filings — that's the only place we'll accept a 'terminated'
    # reason for a warrant/convertible/preferred going forward.
    for t in ("warrant", "convertible", "preferred",
              "shelf", "atm", "equity_line"):
        for row in by_type.get(t, []):
            if row.get("instrument_id") in used_ledger_ids:
                continue
            close_reason = _confident_close_reason(row, as_of_date)
            result.diffs.append(AnchorDiff(
                diff_kind="extra_in_ledger",
                instrument_id=row["instrument_id"],
                category=t,
                ledger_value={"terms": row.get("terms"),
                              "outstanding": row.get("outstanding"),
                              "counterparty": row.get("counterparty_canonical")},
                filing_value=None,
                resolution=(f"closed:{close_reason}:tier1"
                            if close_reason else "kept_ledger"),
            ))
            if close_reason:
                # When the relevant outstanding field is still non-zero,
                # prepend a zeroing AmendInstrument. validate.py's
                # close_with_outstanding guard rejects close(redeemed)
                # while principal_remaining > 0, and walker filings rarely
                # state an explicit "principal repaid" event for an
                # instrument that matured uneventfully — without this
                # amend the close was rejected every quarter and the same
                # zombie row re-flagged forever (IQST C-003: 16 quarters
                # of "closed:redeemed" resolution with status still
                # active). effects_overlay in validate.py picks up the
                # amend so the close that follows it sees the zeroed state.
                zeroing = _zero_outstanding_for_close(row, t)
                if zeroing:
                    result.correction_mutations.append(amend_from_dict(
                        type_=t,
                        instrument_id=row["instrument_id"],
                        outstanding_updates=zeroing,
                        event_date=as_of_date,
                    ))
                result.correction_mutations.append(CloseInstrument(
                    instrument_id=row["instrument_id"],
                    reason=close_reason,
                    event_date=_as_date(as_of_date),
                ))

    # s1_offering staleness reaper. S-1 takedowns are not an overhang
    # category, so the matching loop above never touches them and the
    # extra_in_ledger loop's type list excludes them — without this they
    # accumulate as permanent 'active' rows. Close only when the age gate
    # in _confident_close_reason fires; stay silent otherwise (no
    # kept_ledger diff) so we don't write one row per S-1 per periodic.
    # No zeroing amend needed: S-1 rows carry drawn_usd/sold_to_date, not
    # principal_remaining, which is the field validate.py's terminated
    # guard checks for this type.
    for row in by_type.get("s1_offering", []):
        if row.get("instrument_id") in used_ledger_ids:
            continue
        close_reason = _confident_close_reason(row, as_of_date)
        if not close_reason:
            continue
        result.diffs.append(AnchorDiff(
            diff_kind="extra_in_ledger",
            instrument_id=row["instrument_id"],
            category="s1_offering",
            ledger_value={"terms": row.get("terms"),
                          "outstanding": row.get("outstanding"),
                          "last_seen_date": row.get("last_seen_date")},
            filing_value=None,
            resolution=f"closed:{close_reason}:tier1",
        ))
        result.correction_mutations.append(CloseInstrument(
            instrument_id=row["instrument_id"],
            reason=close_reason,
            event_date=_as_date(as_of_date),
        ))

    # ── Stated-balance reconciliation (deterministic backstop) ──────
    # The filing's own prose is ground truth for note balances. For
    # every note the text states a balance for: (a) a POSITIVE stated
    # balance VETOES any close / zeroing proposal this pass generated
    # for that note (the overhang the proposals came from is the lossy
    # LLM read; the sentence is the issuer's own number — CETY Coventry
    # was zero+closed 'redeemed' while its 10-Q stated $10,120); (b) the
    # ledger's principal_remaining is pinned to the stated balance when
    # they disagree, recovering balance updates the truncated overhang
    # dropped (CETY C-605: walked the 10-Q stating $61,597 yet stayed at
    # face $131,610).
    stated_map = _map_stated_balances(
        stated_note_balances or [], ledger_open)
    if stated_map:
        protected = {iid for iid, bal in stated_map.items() if bal > 0}
        if protected:
            kept: list[Mutation] = []
            for m in result.correction_mutations:
                iid = getattr(m, "instrument_id", None)
                if iid in protected:
                    if isinstance(m, CloseInstrument):
                        log.info(
                            "  anchor %s — veto close(%s) on %s: filing "
                            "states positive note balance $%.0f",
                            accession, m.reason, iid, stated_map[iid])
                        continue
                    upd = getattr(m, "outstanding_updates", None) or {}
                    if _to_float(upd.get("principal_remaining")) == 0:
                        log.info(
                            "  anchor %s — veto zeroing amend on %s: "
                            "filing states positive note balance $%.0f",
                            accession, iid, stated_map[iid])
                        continue
                kept.append(m)
            result.correction_mutations = kept
        by_id = {r.get("instrument_id"): r for r in ledger_open}
        for iid, bal in stated_map.items():
            row = by_id.get(iid)
            if row is None:
                continue
            cur = _to_float(_outstanding_dict(row).get(
                "principal_remaining"))
            # DECREASE-ONLY: stated 'balance of this note' figures
            # sometimes include accrued interest / default premiums and
            # EXCEED face (CETY C-603: $345,000 note, stated balance
            # $436,654) — principal_remaining is face-denominated, so
            # only a balance BELOW the ledger value is a safe principal
            # update (amortization / conversions the overhang missed).
            if cur is None or bal >= cur - max(1.0, bal * 0.001):
                continue
            result.correction_mutations.append(amend_from_dict(
                type_="convertible",
                instrument_id=iid,
                outstanding_updates={"principal_remaining": bal},
                event_date=as_of_date,
            ))
            result.diffs.append(AnchorDiff(
                diff_kind="field_mismatch",
                instrument_id=iid,
                category="convertible",
                ledger_value={"principal_remaining": cur},
                filing_value={"principal_remaining": bal},
                resolution="stated_balance_pin",
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
        for d in result.diffs:
            log.info("    diff %s", _fmt_diff(d))
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
    over_cp = _real_cp(_extract_cp_from_name(over.get("instrument_name")))
    over_issue = _normalize_date(over.get("issue_date")) or \
        _extract_date_from_name(over.get("instrument_name"))
    over_maturity = _normalize_date(over.get("maturity_or_expiry"))
    over_series = (
        extract_series_letter(over.get("series_letter"))
        or extract_series_letter(over.get("instrument_name"))
    )

    # Series letter is unique within an issuer and definitively
    # distinguishes preferred tranches. When the overhang specifies one,
    # exclude any candidate whose stored series_letter differs — series
    # B and series D are not the same instrument even at matching strike.
    # Candidates without any series_letter signal pass through and are
    # judged on the other axes.
    if over_series:
        candidates = [
            r for r in candidates
            if (r_series := (
                extract_series_letter(_terms_dict(r).get("series_letter"))
                or extract_series_letter(r.get("label"))
            )) is None or r_series == over_series
        ]
        if not candidates:
            return None

    # Pre-funded vs ordinary is a TRANCHE IDENTITY, not field drift: a
    # $0.001 pre-funded overhang line must never bind to (and then
    # overwrite) a real-strike common row sharing the issue date.
    # XTIA's truncated June-2025 anchor did exactly that — the
    # pre-funded line (785,700 @ $0.001) amended the $2.00 Common
    # Warrants row, whose card then vanished via the sub-penny render
    # filter, while the common line landed on the Rep-Warrant row.
    # Same for comp-role tranches: a line naming Common/offering
    # warrants must not bind a placement-agent/underwriter/rep row, and
    # vice versa. Both exclusions are three-valued — rows/lines with no
    # signal pass through to the scored axes.
    over_pf = _prefunded_signal(
        over.get("instrument_name"), over_strike, None)
    if over_pf is not None:
        candidates = [
            r for r in candidates
            if (r.get("type") or "") != "warrant"
            or _prefunded_signal(
                r.get("label"), _row_strike(r),
                _terms_dict(r).get("is_pre_funded")) in (None, over_pf)
        ]
        if not candidates:
            return None
    over_comp = _comp_role_signal(over.get("instrument_name"))
    if over_comp is not None:
        candidates = [
            r for r in candidates
            if (r.get("type") or "") != "warrant"
            or _comp_role_signal(r.get("label")) in (None, over_comp)
        ]
        if not candidates:
            return None

    # A line carrying an EXPLICIT issue_date that misses the candidate's
    # creation window AND a materially different count is a different
    # issuance, not field drift — it must not bind and overwrite the
    # sibling (round-4 cety-mar2025: a Jan/Feb-2025 54,594 line bound to
    # the March-2025 20,667 row on strike alone and tripled its count).
    # Both signals are required: explicit dates can lag re-disclosures
    # and counts drift with exercises, but not both at once. Exclusion
    # only — lines with no explicit date or no count pass through.
    over_count_x = _to_float(over.get("count"))
    over_issue_x = _normalize_date(over.get("issue_date"))
    if over_issue_x and over_count_x:
        def _foreign_issuance(r: dict) -> bool:
            if _date_close(over_issue_x, r.get("created_at"),
                           ISSUE_DATE_TOLERANCE_DAYS):
                return False
            r_count = _to_float(_outstanding_dict(r).get("count"))
            if not r_count:
                return False
            return (abs(r_count - over_count_x)
                    / max(r_count, over_count_x)) > 0.25
        candidates = [r for r in candidates if not _foreign_issuance(r)]
        if not candidates:
            return None

    def score(r: dict) -> int:
        return _score_axes(r, over_issue=over_issue, over_cp=over_cp,
                           over_strike=over_strike,
                           over_maturity=over_maturity,
                           over_series=over_series)

    scored = [(score(r), r.get("created_at") or "9999-99-99", r)
              for r in candidates]
    scored.sort(key=lambda x: (-x[0], x[1]))
    best_score, _, best_row = scored[0]
    if best_score == 0:
        return None
    return best_row


def _over_signals(over: dict) -> dict:
    """The five identity signals _best_match reads off an overhang row."""
    return {
        "over_strike": _to_float(
            over.get("strike_or_conversion_price")
            or over.get("conversion_price")),
        "over_cp": _real_cp(_extract_cp_from_name(over.get("instrument_name"))),
        "over_issue": (_normalize_date(over.get("issue_date"))
                       or _extract_date_from_name(over.get("instrument_name"))),
        "over_maturity": _normalize_date(over.get("maturity_or_expiry")),
        "over_series": (extract_series_letter(over.get("series_letter"))
                        or extract_series_letter(over.get("instrument_name"))),
    }


def _score_axes(r: dict, *, over_issue, over_cp, over_strike,
                over_maturity, over_series) -> int:
    """Additive match score for one ledger row. Exclusions live in
    _best_match; this is only the evidence tally, kept module-level so the
    single-winner path and the global assignment score identically."""
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
    # Warrants store their term-end under terms.expiration, NOT
    # terms.maturity (only convertibles/preferreds use 'maturity').
    # Reading only 'maturity' silently disabled this axis for every
    # warrant, so same-letter ladders (GCTK July-A vs November-A)
    # were judged on strike+issue-date alone. Use a tolerance test,
    # not exact equality (GCTK Nov-A is stored 2029-11-15 vs the
    # periodic's 2029-11-14 — 1-day drift). Additive scoring only
    # (+1), never an exclusion, so it cannot drop or mis-bind a row.
    r_maturity = _normalize_date(
        _terms_dict(r).get("maturity")
        or _terms_dict(r).get("expiration")
    )
    if (over_maturity and r_maturity
            and _date_close(over_maturity, r_maturity, 30)):
        s += 1
    if over_series:
        r_series = (
            extract_series_letter(_terms_dict(r).get("series_letter"))
            or extract_series_letter(r.get("label"))
        )
        if r_series and r_series == over_series:
            s += 5  # strong identity for preferred tranches
    return s


def _match_scores(over: dict, candidates: list[dict]) -> dict[str, int]:
    """Every eligible candidate's match score for one overhang row.

    Same exclusions and same additive axes as :func:`_best_match` — this
    is that function stopped one step earlier, before it collapses the
    ranking to a single winner. Rows scoring 0 are omitted: a zero score
    means "no positive evidence", which _best_match treats as no match.
    """
    signals = _over_signals(over)
    out: dict[str, int] = {}
    for row in candidates:
        iid = row.get("instrument_id")
        if not iid:
            continue
        # _best_match on a single-candidate pool applies the exclusion
        # rules (series letter, pre-funded vs ordinary, comp-role,
        # foreign-issuance) and returns None when this pair is ineligible
        # or scores zero — so the rules stay defined in exactly one place.
        if _best_match(over, [row]) is None:
            continue
        out[iid] = _score_axes(row, **signals)
    return out


def _assign_matches(
    overs: list[dict], candidates: list[dict],
) -> dict[int, dict]:
    """Globally assign overhang rows to ledger rows, best pairs first.

    Returns {index in `overs`: matched ledger row}. Each ledger row is
    claimed at most once, as before — what changes is the ORDER in which
    claims are granted.

    Why this exists: matching used to be greedy in overhang-row order,
    first-fit. Whoever came first claimed its best candidate, so a
    mediocre early pairing could take a row that a later line matched far
    better, stranding that later line with no candidate at all — which
    the reconciler then reported as `missing_in_ledger` and synthesized
    into a DUPLICATE instrument. Measured on CELU's Aug-2025 10-Q: 19
    overhang warrants against ~20 ledger rows produced 4 phantom creates,
    including a "March 2024 RWI Forbearance Warrants" line that shared
    both issue date and counterparty with an existing March-2024 RWI row.

    Sorting all (score, over, row) pairs descending and granting in that
    order lets the strongest evidence bind first, so a weak pairing can no
    longer strand a strong one. Ties break deterministically: higher
    score, then earliest-created ledger row (unchanged from _best_match),
    then overhang order.
    """
    pairs: list[tuple[int, str, int, dict]] = []
    for i, over in enumerate(overs):
        for iid, sc in _match_scores(over, candidates).items():
            pairs.append((sc, iid, i, over))
    by_id = {c.get("instrument_id"): c for c in candidates}
    # -score first (descending), then the candidate's created_at, then
    # overhang order — a total order, so the result cannot vary run to run.
    pairs.sort(key=lambda p: (
        -p[0], (by_id[p[1]].get("created_at") or "9999-99-99"), p[2]))

    assigned: dict[int, dict] = {}
    used: set[str] = set()
    for sc, iid, i, _over in pairs:
        if i in assigned or iid in used:
            continue
        assigned[i] = by_id[iid]
        used.add(iid)
    return assigned


def _subsumed_by_used_tranche(
    over: dict, used_rows: list[dict],
) -> dict | None:
    """Return an already-consumed ledger row this overhang row is
    plausibly a sub-bucket of, or None.

    Fires only after _best_match against unused candidates returned
    None — i.e. we're about to synthesize a brand-new row. Periodic
    filings sometimes split a single tranche's count across multiple
    XBRL rows (e.g. an 800k "investor" bucket + 450k "fund-raising"
    bucket from the same private placement). The first row claims the
    matching ledger id and consume-once locks out the second; without
    this guard the second row spawns a spurious duplicate instrument.

    The check is intentionally conservative: requires a positive
    strike-match (not the three-valued None — i.e. both sides must
    have a strike, and they must agree within tolerance), and rejects
    when both sides state a maturity and they differ. Genuinely
    distinct same-strike tranches (e.g. Series A and Series B both
    struck at $0.65 but with different maturities) keep their right
    to be synthesized.

    Two name-marked classes are additionally recognized WITHOUT the
    strike agreement, because for them a differing strike is the point
    rather than evidence of a different instrument:

      * repricing markers ("(modified)", "as amended", "repriced") — the
        filing is re-stating an existing tranche at its NEW strike, so
        requiring the old strike to match guarantees a phantom. CELU's
        "March 2023 PIPE Warrants (modified)" @ $1.00 and "April 2023
        Registered Direct Warrants (modified)" @ $0.35 both minted
        duplicates of rows still carrying the pre-amendment strike.
      * sub-tranche ordinals ("Tranche #2", "Tranche 2") — an explicitly
        numbered slice of a financing whose sibling is already bound.
        CELU's "January 2024 Bridge Loan - Tranche #2 Warrants" @ $3.076
        and "Faithstone Strategic Advisory Warrants — Tranche 2" @ $5.00.

    Both still require the issue dates to agree, so an unrelated later
    financing cannot be swallowed. Matching here suppresses the synthetic
    create only — no field is written back, deliberately: re-pointing a
    strike from an overhang line is how the conv_price ping-pong class of
    bug starts, and the walker's own amend path already owns repricing.
    """  # noqa: D205
    over_strike = _to_float(
        over.get("strike_or_conversion_price")
        or over.get("conversion_price")
    )
    over_maturity = _normalize_date(over.get("maturity_or_expiry"))
    name = over.get("instrument_name")
    relaxed = _reprice_marker(name) or _tranche_ordinal(name) is not None
    over_issue = _normalize_date(over.get("issue_date")) or \
        _extract_date_from_name(name)

    if over_strike is None and not relaxed:
        return None
    for r in used_rows:
        if relaxed:
            # Strike is allowed to differ; identity rests on the issue
            # date instead, which a repricing/sub-tranche re-disclosure
            # carries unchanged.
            if not (over_issue and _date_close(
                    over_issue, r.get("created_at"),
                    ISSUE_DATE_TOLERANCE_DAYS)):
                continue
        elif _strike_match(over_strike, _row_strike(r)) is not True:
            continue
        r_maturity = _normalize_date(_terms_dict(r).get("maturity"))
        if over_maturity and r_maturity and over_maturity != r_maturity:
            continue
        return r
    return None


# Names that mark a re-statement of an existing tranche rather than a new
# one. Deliberately narrow: only wording that explicitly says "this is the
# same instrument, changed". "new"/"inducement"/"replacement" warrants are
# NOT here — those are genuinely separate issuances.
_REPRICE_MARKER_RE = re.compile(
    r"(?i)\((?:as\s+)?(?:modified|amended|repriced)\)"
    r"|\bas\s+(?:modified|amended|repriced)\b"
    r"|\brepriced\b"
    r"|\bamendment\s+to\b"
)

# "Tranche 2", "Tranche #2", "Tranche No. 2" — an explicitly numbered
# slice. Tranche 1 counts too: it is still a slice of a financing whose
# siblings share the issue date.
_TRANCHE_ORDINAL_RE = re.compile(
    r"(?i)\btranche\s*(?:#|no\.?\s*)?(\d{1,2})\b")


def _reprice_marker(name: str | None) -> bool:
    return bool(name) and bool(_REPRICE_MARKER_RE.search(name))


def _tranche_ordinal(name: str | None) -> int | None:
    if not name:
        return None
    m = _TRANCHE_ORDINAL_RE.search(name)
    return int(m.group(1)) if m else None


def _load_closed_shelf_family(cik: int) -> list[dict]:
    """Pull closed (non-active) shelf/atm/equity_line rows for identity
    matching. Returned rows mirror `get_open_instruments`' shape so the
    same matchers work against either pool. Used to suppress synthetic
    re-creation when a filing's MD&A re-discloses a historical program.

    Local import keeps anchor.py importable without a DB at module load
    time (test fixtures use hand-built dicts).
    """
    import json
    from db import get_conn
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT instrument_id, type, created_at, created_accession,
                      counterparty_canonical, placement_agent_canonical,
                      label, status, status_at,
                      terms_json, outstanding_json
                 FROM dilution_ledger
                WHERE cik = ?
                  AND type IN ('shelf', 'atm', 'equity_line')
                  AND status != 'active'""",
            (cik,),
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        # Parse JSON columns to mirror get_open_instruments' shape so
        # _terms_dict / _outstanding_dict succeed without going through
        # terms_json/outstanding_json fallback.
        try:
            d["terms"] = json.loads(d.get("terms_json") or "{}") or {}
        except (TypeError, ValueError):
            d["terms"] = {}
        try:
            d["outstanding"] = (
                json.loads(d.get("outstanding_json") or "{}") or {}
            )
        except (TypeError, ValueError):
            d["outstanding"] = {}
        out.append(d)
    return out


def corroborate_closes(
    *, cik: int, accession: str, filing_overhang: list[dict],
    stated_note_balances: list[dict] | None = None,
    as_of_date: str | None = None,
) -> list[str]:
    """Anchor-corroborated close-rejection.

    Return instrument_ids that THIS filing closed but whose own overhang
    table still reports as outstanding — the close contradicts the filing's
    own numbers, so the row should be reopened. The walker LLM sometimes
    hallucinates a close on an instrument the same 10-Q lists as live (CETY:
    a redeem/expire sweep against notes the filing's MD&A reports as $3.6M
    outstanding). This catches the SUB-sweep cases below the mass-closure
    guard's >=5-type threshold; the two are complementary.

    Three reopen signals:
      1. Overhang re-listing (tranche types) — guarded so a FULLY-RETIRED
         row never reopens: a count-0 / principal-0 row re-listed in a
         warrant table is historical disclosure, not life (CETY's 2022
         warrants: fully exercised in 2023, re-listed by later 10-Q
         tables, reopened into phantom card extras).
      2. A positive stated note balance in the filing's own prose
         (see extract_stated_note_balances) for a convertible this
         filing closed — the issuer says money is still owed. Includes
         'converted' closes: a 10-K whose subsequent-events section
         lists post-period partial conversions can trick the walker
         into a converted-close while the SAME filing's balance prose
         says the note is live (CETY Pacific Pier: $83k+$85k January
         conversions of a $384k year-end balance → spurious converted
         close; the in-batch record_events zeroed principal so the
         validator's converted-requires-zero gate passed).
      3. ATM / equity_line closed by this filing while its booked
         drawdowns show recent activity (post-as_of sales or a draw
         within the reporting window) — mirrors the is_terminated
         auto-close guard, but for WALKER-emitted closes (GCTK: the LLM
         closed the exhausted-but-DT-rendered Dawson James ATM, which
         both dropped the card and blocked the anchor's drawn_usd pin).

    Local import keeps anchor.py importable without a DB at module load.
    """
    import json
    from db import get_conn
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT instrument_id, type, created_at, created_accession,
                      counterparty_canonical, placement_agent_canonical,
                      label, status, status_at, terms_json,
                      outstanding_json, history_json
                 FROM dilution_ledger
                WHERE cik = ? AND last_seen_accession = ?
                  AND type IN ('warrant', 'convertible', 'preferred',
                               'atm', 'equity_line')
                  AND status IN ('terminated', 'redeemed', 'expired',
                                 'converted')""",
            (cik, accession),
        ).fetchall()
    if not rows:
        return []
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        d = dict(r)
        try:
            d["terms"] = json.loads(d.get("terms_json") or "{}") or {}
        except (TypeError, ValueError):
            d["terms"] = {}
        try:
            d["outstanding"] = json.loads(d.get("outstanding_json") or "{}") or {}
        except (TypeError, ValueError):
            d["outstanding"] = {}
        by_type.setdefault(d["type"], []).append(d)

    def _retired_before_close(c: dict) -> bool:
        """True when the row's gating balance was already 0 going into
        the close — i.e. the close is corroborated by the row's own
        numbers and an overhang re-listing must not resurrect it. A
        'redeemed' close zeroes balances itself (_apply_close), so for
        those the pre-close balance is recovered from the close history
        entry's *_redeemed_at_close markers."""
        t = (c.get("type") or "").lower()
        out = c.get("outstanding") or {}
        gate = ("count" if t in ("warrant", "preferred")
                else "principal_remaining")
        bal = _to_float(out.get(gate)) or 0.0
        if (c.get("status") or "") == "redeemed":
            try:
                hist = json.loads(c.get("history_json") or "[]")
            except (TypeError, ValueError):
                hist = []
            for e in reversed(hist):
                if e.get("action") == "closed":
                    fc = e.get("fields_changed") or {}
                    bal += (_to_float(
                        fc.get("principal_redeemed_at_close")) or 0
                        if gate == "principal_remaining"
                        else _to_float(
                            fc.get("count_redeemed_at_close")) or 0)
                    break
        return bal <= 0

    reopen: list[str] = []
    used: set[str] = set()
    # Signal 1 — overhang re-listing (tranche types, retired-guarded).
    for over in filing_overhang or []:
        ttype = _CATEGORY_TO_TYPE.get((over.get("category") or "").lower())
        if ttype not in _TRANCHE_TYPES:
            continue
        cands = [c for c in by_type.get(ttype, [])
                 if c["instrument_id"] not in used
                 and not _retired_before_close(c)]
        if not cands:
            continue
        match = _best_match(over, cands)
        if match is not None:
            used.add(match["instrument_id"])
            reopen.append(match["instrument_id"])
    # Signal 2 — positive stated note balance on a closed convertible.
    closed_convs = [c for c in by_type.get("convertible", [])
                    if c["instrument_id"] not in used]
    if closed_convs and stated_note_balances:
        stated_map = _map_stated_balances(
            stated_note_balances, closed_convs)
        for iid, bal in stated_map.items():
            if bal > 0 and iid not in used:
                used.add(iid)
                reopen.append(iid)
    # Signal 3 — ATM / equity_line close contradicted by drawdowns.
    sales_rows = (by_type.get("atm", []) + by_type.get("equity_line", []))
    if sales_rows:
        post_asof_drawn = _load_post_asof_drawn(
            cik, as_of_date or "9999-12-31")
        last_drawdown_date = _load_max_drawdown_date(cik)
        for c in sales_rows:
            iid = c["instrument_id"]
            if iid in used:
                continue
            recent_here = False
            last_dd = last_drawdown_date.get(iid)
            if last_dd and as_of_date:
                try:
                    delta = (_as_date(as_of_date) - _as_date(last_dd)).days
                    if -365 <= delta <= RECENT_ACTIVITY_DAYS:
                        recent_here = True
                except Exception:
                    pass
            # An EXPLICIT termination is only contradicted by sales
            # strictly AFTER the close date — a draw on/before it was
            # already accounted-for when the program was terminated
            # (round-4 xtia-maxim: the dead Maxim chain's final
            # pre-termination draw sat inside the 270-day window and
            # resurrected the head every walk, un-hiding the whole
            # dead chain that _chain_head_terminated exists to bury).
            close_at = (c.get("status_at") or "")[:10]
            if ((c.get("status") or "") == "terminated"
                    and last_dd and close_at and last_dd[:10] <= close_at):
                recent_here = False
            if (post_asof_drawn.get(iid, 0.0) > 0) or recent_here:
                used.add(iid)
                reopen.append(iid)
    return reopen


# ─── Stated note-balance scan ────────────────────────────────────────
# Periodic filings carry stereotyped per-note balance sentences —
# "The balance of this note as of June 30, 2025, was $10,120." — in the
# same paragraph as the note's identity ("issued to Mast Hill a
# $556,000 Convertible Promissory Note"). The combined-overhang LLM
# pass is the primary reader, but it dies nondeterministically on
# runaway generations (REASON_MAX_LEN salvage keeps a prefix), silently
# dropping balance updates and letting extra_in_ledger zero+close live
# notes (CETY Coventry: closed redeemed while the same 10-Q stated
# $10,120 outstanding). This deterministic regex pass is the
# thick-core backstop: map each stated balance to its note by the
# original-principal hint in the same paragraph, then (a) pin
# principal_remaining to the stated balance and (b) veto anchor
# zero/close proposals that contradict a positive stated balance.

_NOTE_BALANCE_RE = re.compile(
    r"balance of (?:this|the) note(?:,| as amended,?)? as of "
    r"[^.$]{0,60}?(?:was|is)\s*\$\s*([\d,]+(?:\.\d+)?)",
    re.IGNORECASE)

_NOTE_PRINCIPAL_HINT_RES = (
    re.compile(r"principal amount of \$\s*([\d,]+(?:\.\d+)?)",
               re.IGNORECASE),
    re.compile(r"a \$\s*([\d,]+(?:\.\d+)?)\s+(?:[a-z\- ]{0,40}?)?"
               r"(?:convertible\s+)?promissory note", re.IGNORECASE),
    re.compile(r"promissory note [^.$]{0,80}?in the amount of "
               r"\$\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE),
)

_BALANCE_HINT_WINDOW = 2500  # chars of context preceding the sentence


def extract_stated_note_balances(
    cik: int, accession: str,
) -> list[dict]:
    """Scan the filing's cached text for per-note stated balances.

    Returns [{"principal_hint": float, "balance": float}] — one entry
    per balance sentence whose paragraph also names the note's original
    principal. Sentences with no recoverable principal hint are dropped
    (unmappable → unsafe to act on). Local DB import mirrors
    corroborate_closes.
    """
    from db import get_conn
    with get_conn() as conn:
        docs = conn.execute(
            "SELECT content_md FROM dilution_raw "
            "WHERE accession_number = ?",
            (accession,),
        ).fetchall()
    found: list[dict] = []
    for doc in docs:
        text = doc["content_md"] or ""
        for m in _NOTE_BALANCE_RE.finditer(text):
            try:
                balance = float(m.group(1).replace(",", ""))
            except ValueError:
                continue
            window = text[max(0, m.start() - _BALANCE_HINT_WINDOW):
                          m.start()]
            hint = None
            best_pos = -1
            for hint_re in _NOTE_PRINCIPAL_HINT_RES:
                for hm in hint_re.finditer(window):
                    if hm.start() > best_pos:
                        try:
                            hint = float(hm.group(1).replace(",", ""))
                            best_pos = hm.start()
                        except ValueError:
                            continue
            if hint and hint > 0:
                found.append({"principal_hint": hint, "balance": balance})
    return found


def _map_stated_balances(
    stated: list[dict], open_rows: list[dict],
) -> dict[str, float]:
    """instrument_id → stated balance, mapped by original-principal
    proximity (≤2% relative). One-to-one: a hint matching two open
    notes, or two hints fighting over one note, is dropped — acting on
    an ambiguous mapping is worse than not acting."""
    convs = [r for r in open_rows
             if (r.get("type") or "").lower() == "convertible"]
    out: dict[str, float] = {}
    claimed_rows: set[str] = set()
    for s in stated or []:
        hint = s.get("principal_hint") or 0
        if hint <= 0:
            continue
        matches = []
        for r in convs:
            principal = _to_float(_terms_dict(r).get("principal"))
            if not principal:
                continue
            if abs(principal - hint) / max(principal, hint) <= 0.02:
                matches.append(r)
        if len(matches) != 1:
            continue
        iid = matches[0]["instrument_id"]
        if iid in claimed_rows:
            # Two balance sentences mapping to the same note: keep the
            # LAST one (later in document order ≈ most recent restated
            # figure), which this overwrite achieves.
            pass
        claimed_rows.add(iid)
        out[iid] = float(s.get("balance") or 0)
    return out


# ─── Stated ATM remaining-capacity scan ──────────────────────────────
# Periodic filings state the live ATM supplement's remaining capacity
# in prose — "As of May 8, 2026, we have approximately $18.3 million
# remaining to be sold pursuant to the … prospectus supplement" (KSCP),
# "approximately $157.1 million of shares remained available for sale
# under the Sales Agreement" (FCEL). This is the issuer's own
# authoritative window-scoped figure; interim ATM sales frequently
# never appear as discrete drawdown disclosures, so without this pin
# the cumulative drawn figure undercounts by every quiet quarter (round-4
# kscp-jul2025: $12.55M recorded vs ~$31.7M actually sold).
_ATM_REMAINING_RES = (
    re.compile(
        r"(?:as of\s+(?P<asof>[A-Z][a-z]+ \d{1,2},? \d{4})[^.$]{0,80}?)?"
        r"\$\s*(?P<amt>[\d,]+(?:\.\d+)?)\s*(?P<unit>million|billion)?\s+"
        r"remain(?:ing|ed|s)?\s+(?:available\s+)?to\s+be\s+sold",
        re.IGNORECASE),
    re.compile(
        r"(?:as of\s+(?P<asof>[A-Z][a-z]+ \d{1,2},? \d{4})[^.$]{0,80}?)?"
        r"\$\s*(?P<amt>[\d,]+(?:\.\d+)?)\s*(?P<unit>million|billion)?\s+"
        r"(?:of\s+[a-z ]{0,30}?)?remain(?:ed|s|ing)?\s+available\s+for"
        r"\s+(?:sale|issuance\s+and\s+sale)",
        re.IGNORECASE),
)
# The sentence (±window) must anchor to an ATM program, not a shelf's
# generic "remained available for issuance".
_ATM_REMAINING_CONTEXT_RE = re.compile(
    r"sales agreement|prospectus supplement|at-the-market|at the market"
    r"|atm facility|open market sale", re.IGNORECASE)
_ATM_REMAINING_CONTEXT_WINDOW = 600


def _parse_prose_date(s: str | None) -> str | None:
    """'May 8, 2026' → '2026-05-08'. None when unparseable."""
    if not s:
        return None
    from datetime import datetime
    for fmt in ("%B %d, %Y", "%B %d %Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def extract_stated_atm_remaining(
    cik: int, accession: str,
) -> list[dict]:
    """Scan the filing's cached text for stated ATM remaining-capacity
    sentences. Returns [{"remaining": float, "asof": iso-date|None}].
    Sentences without ATM-program context nearby are dropped. Local DB
    import mirrors extract_stated_note_balances."""
    from db import get_conn
    with get_conn() as conn:
        docs = conn.execute(
            "SELECT content_md FROM dilution_raw "
            "WHERE accession_number = ?",
            (accession,),
        ).fetchall()
    found: list[dict] = []
    for doc in docs:
        text = doc["content_md"] or ""
        for rex in _ATM_REMAINING_RES:
            for m in rex.finditer(text):
                try:
                    amt = float(m.group("amt").replace(",", ""))
                except (ValueError, TypeError):
                    continue
                unit = (m.group("unit") or "").lower()
                if unit == "million":
                    amt *= 1e6
                elif unit == "billion":
                    amt *= 1e9
                lo = max(0, m.start() - _ATM_REMAINING_CONTEXT_WINDOW)
                hi = min(len(text), m.end() + _ATM_REMAINING_CONTEXT_WINDOW)
                if not _ATM_REMAINING_CONTEXT_RE.search(text[lo:hi]):
                    continue
                asof = _parse_prose_date(m.group("asof"))
                found.append({"remaining": amt, "asof": asof})
    return found


def map_stated_atm_remaining(
    stated: list[dict], atm_rows: list[dict],
) -> list[tuple[str, float, str | None]]:
    """(instrument_id, pinned_drawn_usd, asof) pins for ACTIVE ATM rows.

    Only unambiguous mappings act: exactly one live ATM gets the LAST
    stated figure in document order (most recent re-statement). With
    several live ATMs there is no safe text-to-row binding — skip. The
    pin is authoritative in BOTH directions (the issuer's definitional
    window figure beats summed discrete draws — FCEL's amendment-reset
    window proved increase-only wrong), but never negative and never
    above capacity."""
    live = [r for r in atm_rows
            if (r.get("type") or "") == "atm"
            and (r.get("status") or "") == "active"]
    if len(live) != 1 or not stated:
        return []
    row = live[0]
    cap = _to_float(_terms_dict(row).get("capacity_usd"))
    if not cap or cap <= 0:
        return []
    # Prefer the explicitly-dated figure with the latest as-of; undated
    # sentences (subsequent-events recaps that already fold post-period
    # draws in — FCEL's "$154.5 million") only act when nothing dated
    # exists, since pinning them at the filing's balance date would
    # double-count the post-period draws the drawdown log re-adds.
    dated = [s for s in stated if s.get("asof")]
    s = max(dated, key=lambda x: x["asof"]) if dated else stated[-1]
    remaining = _to_float(s.get("remaining"))
    if remaining is None or remaining < 0:
        return []
    if remaining > cap * (1 + FIELD_MISMATCH_TOLERANCE):
        return []  # talks about a different/old window
    drawn = max(0.0, cap - remaining)
    return [(row["instrument_id"], drawn, s.get("asof"))]


def _load_shelf_file_numbers(
    cik: int, shelves: list[dict],
) -> dict[str, str]:
    """Look up dilution_filings.file_number for each shelf's
    create_accession in one batched query. Returns instrument_id →
    file_number. Missing or non-333 entries are absent from the map.

    Imported locally to keep anchor.py importable without a DB at module
    load time (the test fixtures use hand-built dicts and don't need
    file-number lookups)."""
    if not shelves:
        return {}
    from db import get_conn
    accs_by_iid = {
        s["instrument_id"]: s.get("created_accession")
        for s in shelves
        if s.get("instrument_id") and s.get("created_accession")
    }
    if not accs_by_iid:
        return {}
    placeholders = ",".join(["?"] * len(accs_by_iid))
    with get_conn() as conn:
        rows = conn.execute(
            f"""SELECT accession_number, file_number
                  FROM dilution_filings
                 WHERE cik = ?
                   AND accession_number IN ({placeholders})""",
            (cik, *accs_by_iid.values()),
        ).fetchall()
    acc_to_fn = {r["accession_number"]: r["file_number"] for r in rows}
    return {
        iid: acc_to_fn[acc]
        for iid, acc in accs_by_iid.items()
        if acc_to_fn.get(acc) and acc_to_fn[acc].startswith("333-")
    }


def _load_post_asof_drawn(
    cik: int, as_of_date: str,
) -> dict[str, float]:
    """Per-instrument sum of booked drawdowns whose event_date is strictly
    after `as_of_date`. Anchor subtracts this from the running drawn_usd
    before comparing against the filing's overhang — without it, a 10-K
    that booked a Subsequent-Events takedown via the walker would see
    that takedown as drift against its own FY-end balance and zero it
    out. Local import follows the same DB-free-module-load pattern as
    `_load_closed_shelf_family`.
    """
    from .store import get_post_asof_drawn_by_instrument
    try:
        return get_post_asof_drawn_by_instrument(cik, as_of_date)
    except Exception as exc:
        log.warning(
            "post-as_of drawdown lookup failed cik=%s as_of=%s: %s",
            cik, as_of_date, exc,
        )
        return {}


def _load_max_drawdown_date(cik: int) -> dict[str, str]:
    """Per-instrument max(event_date) across booked drawdowns. Used by the
    is_terminated auto-close guard to detect ATMs/ELOCs that drew shares
    recently — recent activity contradicts a periodic-overhang termination
    flag derived from a stale agreement_end_date."""
    from .store import get_max_drawdown_date_by_instrument
    try:
        return get_max_drawdown_date_by_instrument(cik)
    except Exception as exc:
        log.warning(
            "max-drawdown-date lookup failed cik=%s: %s", cik, exc,
        )
        return {}


def _normalize_fn(s: str | None) -> str | None:
    """Strip stray whitespace and trailing punctuation from a 333-XXXXXX
    file number. Returns None for missing/empty."""
    if not s:
        return None
    s = s.strip().rstrip(".,;")
    return s if s.startswith("333-") else None


def _match_shelf(
    over: dict, candidates: list[dict],
    shelf_file_numbers: dict[str, str],
) -> dict | None:
    """Identity-match a shelf overhang row to a ledger shelf.

    Primary key: SEC file_number (canonical, deterministic).
    Fallback: effect_date proximity (±30d) when the filing doesn't
    quote a file_number — rare but possible in short MD&A blurbs.
    Returns None when no candidate matches.
    """
    if not candidates:
        return None
    over_fn = _normalize_fn(over.get("file_number"))
    if over_fn:
        for c in candidates:
            iid = c.get("instrument_id")
            if iid and shelf_file_numbers.get(iid) == over_fn:
                return c
        # File_number stated but no match → genuinely missing, not just
        # underspecified. Don't fall through to effect_date guess; let
        # the caller synthesize a new shelf row.
        return None

    # Effect-date fallback (only when file_number absent).
    over_effect = _normalize_date(over.get("effect_date")) or \
        _normalize_date(over.get("issue_date"))
    if not over_effect:
        return None
    for c in candidates:
        c_effect = _normalize_date(c.get("created_at"))
        if c_effect and _date_close(
            over_effect, c_effect, ISSUE_DATE_TOLERANCE_DAYS,
        ):
            return c
    return None


def _match_atm_or_eloc(
    over: dict, candidates: list[dict], target_type: str,
) -> dict | None:
    """Identity-match an ATM or equity-line overhang row to a ledger row.

    Joint key: (counterparty, agreement_date ± 30d). Counterparty key
    is `sales_agent` for ATM and `investor` for equity_line. Falls
    back to agreement_date proximity alone when the counterparty
    name is missing on one side; returns None if both axes are
    indeterminate. Two successive agreements with the same
    counterparty (re-ups) stay distinct via the date check.
    """
    if not candidates:
        return None
    cp_key = "sales_agent" if target_type == "atm" else "investor"
    over_cp = _normalize_cp(over.get(cp_key))
    over_agreement = _normalize_date(over.get("agreement_date")) or \
        _normalize_date(over.get("issue_date"))

    def _row_agreement(r: dict) -> str | None:
        t = _terms_dict(r)
        return _normalize_date(t.get("agreement_date")) or \
            _normalize_date(r.get("created_at"))

    def _row_cp(r: dict) -> str | None:
        # For atm/equity_line the canonical key on the ledger row
        # depends on type: ATM's banker lives in placement_agent_canonical,
        # ELOC's investor lives in counterparty_canonical.
        if target_type == "atm":
            return _normalize_cp(r.get("placement_agent_canonical"))
        return _normalize_cp(r.get("counterparty_canonical"))

    over_agents = _agent_set(over.get(cp_key))

    # Score each candidate: +4 exact-cp, +4 date close, +4 agent overlap
    scored: list[tuple[int, str, dict]] = []
    for c in candidates:
        s = 0
        c_cp = _row_cp(c)
        c_agreement = _row_agreement(c)
        date_close = bool(
            over_agreement and c_agreement and _date_close(
                over_agreement, c_agreement, AGREEMENT_DATE_TOLERANCE_DAYS,
            )
        )
        if over_cp and c_cp and over_cp == c_cp:
            s += 4
        if date_close:
            s += 4
        # Syndicate redisclosure: a periodic lists the ATM under its full
        # underwriting syndicate while the ledger row carries only the
        # lead bank. Set-overlap binds them — but suppress it when the
        # overhang states an agreement_date that DISAGREES with this
        # candidate, so two same-banker programs signed on different dates
        # (XTIA's three Maxim ATMs; FCEL's separate single-Jefferies 2024
        # / Dec-2025 agreements) stay distinct and are decided by the date
        # axis. The over_cp != c_cp guard avoids double-counting when the
        # lead-bank-only overhang already matched exactly on cp.
        date_contradicts = bool(
            over_agreement and c_agreement and not date_close
        )
        if over_cp != c_cp and not date_contradicts:
            c_agents = _agent_set(
                c.get("placement_agent_canonical") if target_type == "atm"
                else c.get("counterparty_canonical")
            )
            if over_agents and c_agents and _agents_overlap(
                    over_agents, c_agents):
                s += 4
        # tie-break: prefer earliest-created so the second overhang row
        # can claim the later sibling, and a syndicate redisclosure binds
        # to the original (oldest) program, not a later re-up.
        scored.append((s, c.get("created_at") or "9999-99-99", c))
    scored.sort(key=lambda x: (-x[0], x[1]))
    best_score, _, best_row = scored[0]
    if best_score < 4:
        # Require at least one strong axis to match. Two unrelated
        # rows with no cp + no date + no agent overlap should fall
        # through to synthesis, not collapse into the same id.
        return None
    return best_row


def _prefunded_signal(name: str | None, strike: float | None,
                      flag) -> bool | None:
    """Three-valued pre-funded identity: explicit flag wins, then the
    name, then a sub-penny strike (pre-funded warrants are $0.01 /
    $0.001 / $0.0001; no ordinary warrant prices there). None = no
    signal, judged on the other axes."""
    if flag is not None:
        return bool(flag)
    n = (name or "").lower()
    if "pre-funded" in n or "prefunded" in n or "pre funded" in n:
        return True
    if strike is not None:
        try:
            return float(strike) <= 0.011
        except (TypeError, ValueError):
            return None
    return None


_COMP_ROLE_TOKENS = ("placement agent", "underwriter", "representative",
                     "rep warrant")


def _comp_role_signal(name: str | None) -> bool | None:
    """Three-valued comp-warrant identity: True when the name marks a
    banker-compensation tranche, False when it explicitly names a
    Common/offering tranche, None when silent."""
    n = (name or "").lower()
    if not n:
        return None
    if any(t in n for t in _COMP_ROLE_TOKENS):
        return True
    if "common warrant" in n:
        return False
    return None


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


_AGENT_SPLIT_RE = re.compile(r"\s*(?:,|/| and )\s*", re.IGNORECASE)


def _agent_set(s: str | None) -> set[str]:
    """Split a sales-agent / investor string into normalized firm tokens.

    A periodic filing re-discloses a single ATM under its FULL
    underwriting syndicate ("Jefferies, B. Riley, Barclays, ...") while
    the live ledger row carries only the lead bank ("Jefferies"). A
    syndicate is a SUPERSET that includes the lead — set-overlap lets
    _match_atm_or_eloc recognize them as the same program. Each piece
    runs through _real_cp so generic placeholders ("investors",
    "noteholders") never create spurious overlap."""
    if not s:
        return set()
    out: set[str] = set()
    for piece in _AGENT_SPLIT_RE.split(s):
        n = _real_cp(piece)
        if n:
            out.add(n)
    return out


def _agents_overlap(a: set[str], b: set[str]) -> bool:
    """True when two agent-token sets name a common firm, tolerant of
    corporate-suffix drift: the ledger row canonicalizes to the bare lead
    name ('Jefferies', 'B. Riley') while the periodic overhang lists the
    raw filing form ('Jefferies LLC', 'B. Riley Securities'). Exact match
    always counts; a prefix match counts only when the shorter token is
    ≥4 chars, so 'jefferies'⊂'jefferies llc' binds but 3-letter tickers
    like 'bmo' match only exactly."""
    for x in a:
        for y in b:
            if x == y:
                return True
            shorter = x if len(x) <= len(y) else y
            longer = y if shorter is x else x
            if len(shorter) >= 4 and longer.startswith(shorter):
                return True
    return False


def _is_empty_capacity_overhang(over: dict) -> bool:
    """True when an atm/equity_line overhang row carries neither an
    identity key (sales_agent / investor / agreement_date) NOR any
    economics (capacity / drawn / sold / remaining). Such rows are bare
    XBRL aggregate placeholders the LLM sometimes emits alongside a real
    itemized row; synthesizing them produces empty-shell phantoms (FCEL
    ATM-2232). Dropping them is safe — a real program always carries at
    least an agent, a date, or a dollar figure."""
    has_id = (
        any(_real_cp(over.get(k)) for k in ("sales_agent", "investor"))
        or bool(_normalize_date(over.get("agreement_date"))
                or _normalize_date(over.get("issue_date")))
    )
    has_econ = any(
        _to_float(over.get(k)) not in (None, 0.0)
        for k in ("total_capacity_usd", "drawn_to_date_usd",
                  "sold_to_date_usd", "remaining_capacity_usd")
    )
    return not has_id and not has_econ


def _warrant_covers_same_party(
    cik: int, investor_cp: str | None, agreement_date: str | None,
) -> bool:
    """True when a warrant on the same counterparty within ±30d of
    `agreement_date` already represents this instrument. Used to suppress
    a phantom equity_line the overhang extractor synthesized for what is
    really a warrant-exercise line (IQST ADI Funding 'Common Stock
    Purchase Option', already tracked as a warrant). Queries warrants of
    ANY status — the matching warrant may be expired/closed by now."""
    if not investor_cp or not agreement_date:
        return False
    from db import get_conn
    with get_conn() as conn:
        ws = conn.execute(
            "SELECT counterparty_canonical, created_at "
            "FROM dilution_ledger WHERE cik=? AND type='warrant'",
            (cik,),
        ).fetchall()
    for w in ws:
        if _normalize_cp(w["counterparty_canonical"]) != investor_cp:
            continue
        if _date_close(agreement_date, w["created_at"], 30):
            return True
    return False


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
    cand = " ".join(head[:2]).strip().lower() or None
    if cand is None:
        return None
    # Never emit a "name" made entirely of security-type vocabulary —
    # 'Series B-1 Redeemable Convertible Preferred Stock Warrants' →
    # 'redeemable convertible', 'Warrants to purchase Common Stock' →
    # 'warrants to'. Those are instrument descriptions, not parties,
    # and they poison the synthesized row's label/title (round-4
    # extract-cp-from-name-emits-vocab-fragment). Belt-and-suspenders
    # with validate._sanitize_entity_canonicals downstream.
    tokens = [t for t in re.split(r"[^a-z0-9]+", cand) if t]
    if tokens and all(t in _VOCAB_ONLY_TOKENS for t in tokens):
        return None
    return cand


# ─── per-row diffing ────────────────────────────────────────────────
_FIELD_MAP_BY_TYPE = {
    "warrant": [
        ("strike", "strike_or_conversion_price"),
    ],
    "convertible": [
        ("conv_price", "strike_or_conversion_price"),
        # NOTE: principal_amount is deliberately NOT mapped to terms.principal.
        # terms.principal is the note's ORIGINAL face value (immutable,
        # set at create-time from the issuing 8-K/424B). A periodic
        # filing's principal_amount is the CURRENT outstanding balance,
        # which declines as the note converts — so the two diverge by
        # design the moment conversions start, producing a perpetual
        # field_mismatch (IQST C-192: 7 consecutive quarters). The
        # correction was also structurally un-appliable — AmendConvertible
        # carries no `principal` field, so amend_from_dict silently dropped
        # it and terms.principal never moved. The declining balance is
        # reconciled correctly via _OUT_FIELD_MAP_BY_TYPE → principal_remaining.
    ],
    "preferred": [
        ("conv_price", "strike_or_conversion_price"),
        # NOTE: principal_amount is deliberately NOT mapped to
        # terms.liquidation_preference — same class of bug as the
        # convertible principal mapping above, but WORSE because
        # AmendPreferred *does* carry liquidation_preference, so the bad
        # write persists and corrupts the card. The aggregate liq-pref is
        # an origination fact (set at create from the issuing filing, in
        # real dollars); a periodic restates it inconsistently and is
        # plagued by the "(in thousands)" problem the shelf-capacity note
        # below warns about. IQST P-160 (Series B, $64.02M): the periodic
        # stated it as "64,020" one quarter and "64,020,000" the next, so
        # the anchor ping-ponged terms.liquidation_preference between $64M
        # and $64K — leaving it 1000× wrong for an entire quarter. The
        # genuine decline (shares converting) is reconciled via the count
        # axis (_OUT_FIELD_MAP_BY_TYPE → outstanding_count), which is
        # split- and units-robust; the card derives aggregate exposure as
        # stated_value × count and treats liquidation_preference as an
        # unreliable fallback (see cards.py).
    ],
    # Shelf-family: registered dollar capacity is a REGISTRATION fact,
    # set authoritatively at create-time from the cover page of the
    # registration document (S-3/F-3/424B, in real dollars) and bumped
    # only by later registration filings the walker processes directly
    # (S-3MEF, sales-agreement amendments — none of which are periodic).
    # The anchor must NOT restate capacity_usd from a periodic financial
    # statement: those notes are routinely "(in thousands)", so trusting
    # them divides capacity by 1000 (GCTK: a $30M shelf knocked to
    # $30,000; an $8.23M ATM to $8,230). Capacity therefore has no field
    # mapping here. Drawn-to-date still reconciles (see _OUT_FIELD_MAP).
    "shelf":       [],
    "atm":         [],
    "equity_line": [],
}
_OUT_FIELD_MAP_BY_TYPE = {
    "warrant": ("count", "outstanding_count"),
    "convertible": ("principal_remaining", "principal_amount"),
    "preferred": ("count", "outstanding_count"),
    # Shelf-family count drift = drawn_usd drift. The cards layer
    # derives remaining_capacity_usd as (capacity - drawn) so we only
    # need to amend drawn_usd; the remaining figure flows through.
    "shelf":       ("drawn_usd", "drawn_to_date_usd"),
    "atm":         ("drawn_usd", "drawn_to_date_usd"),
    "equity_line": ("drawn_usd", "drawn_to_date_usd"),
}


def _field_changes(
    ledger_row: dict, over: dict, *,
    post_asof_drawn_usd: float = 0.0,
) -> dict | None:
    """Return updates + diff metadata if the filing's stated values
    disagree with the ledger by more than FIELD_MISMATCH_TOLERANCE.
    None when in agreement.

    Two parallel update channels:
      - `terms_updates` → AmendInstrument.field_updates (terms_json)
      - `out_updates`   → AmendInstrument.outstanding_updates (outstanding_json)

    `diff_kind` reflects the dominant axis: count_mismatch when only
    outstanding drifted, field_mismatch when terms drifted (with or
    without an outstanding drift on the side).

    `post_asof_drawn_usd` is the sum of drawdowns booked against this
    instrument with event_date > as_of_date — back it out of the ledger's
    drawn_usd before comparing to the filing's as-of figure (shelf-family
    only), and add it back into the amend value if drift still remains.
    """
    t = (ledger_row.get("type") or "").lower()
    terms = _terms_dict(ledger_row)
    out = _outstanding_dict(ledger_row)
    terms_updates: dict = {}
    out_updates: dict = {}
    ledger_view: dict = {}
    filing_view: dict = {}
    diff_kind: str | None = None

    for terms_key, over_key in _FIELD_MAP_BY_TYPE.get(t, []):
        l_val = _to_float(terms.get(terms_key))
        f_val = _to_float(over.get(over_key))
        if l_val is None or f_val is None:
            continue
        if not _within(l_val, f_val, FIELD_MISMATCH_TOLERANCE):
            terms_updates[terms_key] = f_val
            ledger_view[f"terms.{terms_key}"] = l_val
            filing_view[f"terms.{terms_key}"] = f_val
            diff_kind = "field_mismatch"

    out_pair = _OUT_FIELD_MAP_BY_TYPE.get(t)
    if out_pair:
        out_key, over_key = out_pair
        l_val = _to_float(out.get(out_key))
        f_val = _to_float(over.get(over_key))
        # Shelf-family drawn_usd comparison runs against an as_of-adjusted
        # ledger value: subtract Subsequent-Events drawdowns the walker
        # already booked with event_date > as_of_date so we don't read
        # them as drift.
        adjust_drawn = (
            t in {"shelf", "atm", "equity_line"}
            and out_key == "drawn_usd"
            and post_asof_drawn_usd
        )
        compare_l = (
            l_val - post_asof_drawn_usd if (adjust_drawn and l_val is not None)
            else l_val
        )
        if compare_l is not None and f_val is not None and not _within(
            compare_l, f_val, FIELD_MISMATCH_TOLERANCE,
        ):
            # Preserve int-typed warrant counts; preferred counts may be
            # fractional (see PreferredOutstanding in mutations.py).
            if t == "warrant" and out_key == "count":
                f_val = int(round(f_val))
            # Warrant count_mismatch guard: when exercises or terminations
            # have already been booked against this tranche AND the
            # filing's stated outstanding plus that event log would
            # exceed initial_count, the overhang row is restating "ever
            # issued" rather than "currently outstanding" (or the issuer
            # left a stale tabular cap-table entry). Trusting it
            # overwrites a count the exercise log has right — SCNI W-026
            # / W-031 both got their count re-inflated to the original
            # tranche size after exercises drained it. Skip the amend.
            #
            # When no events have accrued, count clarifications are
            # legitimate (8-K announces ~550K, 424B prices 626K
            # overallotment). _apply_amend will mirror onto initial_count
            # in that no-events branch — no guard needed here.
            if t == "warrant" and out_key == "count":
                exercised = _to_float(out.get("exercised_to_date")) or 0
                terminated = _to_float(out.get("terminated_to_date")) or 0
                initial = _to_float(out.get("initial_count"))
                events_booked = exercised > 0 or terminated > 0
                if (events_booked
                        and initial is not None
                        and f_val + exercised + terminated
                        > initial * (1 + FIELD_MISMATCH_TOLERANCE)):
                    log.info(
                        "anchor count_mismatch suppressed for %s: "
                        "filing=%s + exercised=%s + terminated=%s > "
                        "initial=%s — filing restates issued, not outstanding",
                        ledger_row.get("instrument_id"),
                        f_val, exercised, terminated, initial,
                    )
                else:
                    out_updates[out_key] = f_val
            elif (t in {"shelf", "atm", "equity_line"}
                    and out_key == "drawn_usd"
                    and (cap_w := _to_float(
                        terms.get("capacity_usd")
                        or terms.get("total_capacity_usd")))
                    and f_val > cap_w * (1 + FIELD_MISMATCH_TOLERANCE)):
                # A stated drawn figure EXCEEDING this row's capacity is
                # an agreement-LIFETIME / lineage cumulative, not this
                # supplement window's own sales — pinning it onto a
                # reset-fresh successor triple-books every predecessor's
                # draws into the shelf rollup (round-4 fcel-oct2023:
                # the 10-K's $423.77M lifetime figure landed on the
                # fresh $204.9M-cap successor and doubled shelf raised
                # to $715.7M). Skip the amend; window-scoped accounting
                # comes from discrete drawdowns + the stated-remaining
                # pin.
                log.info(
                    "anchor drawn_usd pin suppressed for %s: filing "
                    "drawn=%s exceeds window capacity=%s — lifetime "
                    "cumulative, not this window",
                    ledger_row.get("instrument_id"), f_val, cap_w,
                )
            elif adjust_drawn:
                # Fold post-as_of drawdowns back in so the amend doesn't
                # silently delete activity the walker booked from this
                # same filing's Subsequent Events section.
                out_updates[out_key] = f_val + post_asof_drawn_usd
            else:
                out_updates[out_key] = f_val
            if out_key in out_updates:
                ledger_view[f"out.{out_key}"] = compare_l
                filing_view[f"out.{out_key}"] = f_val
                # Only flip to count_mismatch when no terms drift was seen —
                # if both drifted, terms wins for labelling. The mutation
                # carries both updates regardless.
                if diff_kind is None:
                    diff_kind = "count_mismatch"

    if not diff_kind:
        return None
    return {
        "terms_updates": terms_updates,
        "out_updates": out_updates,
        "diff_kind": diff_kind,
        "ledger_value": ledger_view,
        "filing_value": filing_view,
    }


def _synthesize_create(
    over: dict, *, accession: str, filing_date: str,
) -> Mutation:
    """Build a typed Create* dataclass from a filing-stated overhang row.

    Used when the periodic filing names an instrument the ledger
    doesn't have (typically a pre-window instrument the issuer is
    re-disclosing). The synthetic create carries the filing's values
    as terms; the walker assigns the id."""
    cat = (over.get("category") or "").lower()
    target_type = _CATEGORY_TO_TYPE.get(cat) or "equity"
    terms: dict = {}
    outstanding: dict = {}

    # Common share counts (warrants) are whole-number; preferred
    # counts are float-typed because filings routinely state
    # `count = principal / stated_value` with a fractional result
    # (e.g. 53,197.7234 preferred shares for $53.197M @ $1000 stated).
    # We round at this entry point only for warrants — preferred
    # carries the fraction through.
    raw_count = over.get("outstanding_count")
    count_int: int | None = None
    count_float: float | None = None
    if raw_count is not None:
        try:
            count_float = float(raw_count)
            count_int = round(count_float)
        except (TypeError, ValueError):
            count_int = None
            count_float = None

    if target_type == "warrant":
        if over.get("strike_or_conversion_price") is not None:
            terms["strike"] = float(over["strike_or_conversion_price"])
        if (mat := safe_date(over.get("maturity_or_expiry"))):
            terms["maturity"] = mat
        if count_int is not None:
            outstanding["count"] = count_int
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
        if count_float is not None:
            outstanding["count"] = count_float
        # series_letter is the strongest identity key for preferreds —
        # carry it through so apply-time dedup can collapse this synth
        # onto an existing row of the same series.
        series = (
            extract_series_letter(over.get("series_letter"))
            or extract_series_letter(over.get("instrument_name"))
        )
        if series:
            terms["series_letter"] = series
    elif target_type == "shelf":
        # Synthetic shelf when the filing references a registration the
        # walker hasn't created (typical when the parent S-3 was filed
        # before the ingest window, or got skipped by the pre-screen
        # bug). Carry every field the periodic disclosure stated so
        # apply-time has a complete-enough row to be useful.
        if over.get("total_capacity_usd") is not None:
            terms["capacity_usd"] = float(over["total_capacity_usd"])
        if over.get("form"):
            terms["form"] = over["form"]
        if over.get("file_number"):
            # Stored on the ledger row's terms so cards / future anchors
            # have it without needing the dilution_filings join. The
            # walker normally relies on dilution_filings, but a synthetic
            # row created here may not have a matching filings entry.
            terms["file_number"] = over["file_number"]
        drawn = over.get("drawn_to_date_usd")
        if drawn is not None:
            outstanding["drawn_usd"] = float(drawn)
        remaining = over.get("remaining_capacity_usd")
        if remaining is not None:
            outstanding["remaining_capacity_usd"] = float(remaining)
        elif (drawn is not None
              and over.get("total_capacity_usd") is not None):
            outstanding["remaining_capacity_usd"] = max(
                0.0, float(over["total_capacity_usd"]) - float(drawn),
            )
    elif target_type == "atm":
        if over.get("total_capacity_usd") is not None:
            terms["capacity_usd"] = float(over["total_capacity_usd"])
        if over.get("agreement_date"):
            terms["agreement_date"] = safe_date(over["agreement_date"])
        drawn = over.get("drawn_to_date_usd")
        if drawn is not None:
            outstanding["drawn_usd"] = float(drawn)
        remaining = over.get("remaining_capacity_usd")
        if remaining is not None:
            outstanding["remaining_capacity_usd"] = float(remaining)
        elif (drawn is not None
              and over.get("total_capacity_usd") is not None):
            outstanding["remaining_capacity_usd"] = max(
                0.0, float(over["total_capacity_usd"]) - float(drawn),
            )
    elif target_type == "equity_line":
        if over.get("total_capacity_usd") is not None:
            terms["capacity_usd"] = float(over["total_capacity_usd"])
        if over.get("agreement_date"):
            terms["agreement_date"] = safe_date(over["agreement_date"])
        drawn = over.get("drawn_to_date_usd")
        if drawn is not None:
            outstanding["drawn_usd"] = float(drawn)
        remaining = over.get("remaining_capacity_usd")
        if remaining is not None:
            outstanding["remaining_capacity_usd"] = float(remaining)
        elif (drawn is not None
              and over.get("total_capacity_usd") is not None):
            outstanding["remaining_capacity_usd"] = max(
                0.0, float(over["total_capacity_usd"]) - float(drawn),
            )

    # Counterparty / placement-agent selection by type:
    #   shelf       — no counterparty (each takedown's banker lives on
    #                 the drawdown row, not the shelf)
    #   atm         — sales_agent is the banker → placement_agent on
    #                 the ledger row
    #   equity_line — investor is the funder → counterparty on the
    #                 ledger row
    #   others      — counterparty derived from instrument_name as
    #                 before
    cp_canonical: str | None = None
    placement_canonical: str | None = None
    if target_type == "atm":
        placement_canonical = (
            (over.get("sales_agent") or "").strip() or None
        )
    elif target_type == "equity_line":
        cp_canonical = (
            (over.get("investor") or "").strip() or None
        )
    elif target_type == "shelf":
        pass  # leave both null
    else:
        cp_canonical = _extract_cp_from_name(over.get("instrument_name"))

    return create_from_dict(
        type_=target_type,
        terms=terms,
        outstanding=outstanding,
        counterparty_canonical=cp_canonical,
        placement_agent_canonical=placement_canonical,
        label=(over.get("instrument_name") or None),
        event_date=safe_date(over.get("issue_date")) or filing_date,
    )


def _zero_outstanding_for_close(row: dict, t: str) -> dict:
    """Return the outstanding_updates dict that zeros the field the
    close_with_outstanding validator checks for type `t`. Empty dict
    when the relevant field is already zero (no amend needed)."""
    out = _outstanding_dict(row)
    updates: dict = {}
    if t == "warrant":
        count = _to_float(out.get("count"))
        if count is not None and count > 0:
            updates["count"] = 0
    elif t in ("convertible", "preferred"):
        pr = _to_float(out.get("principal_remaining"))
        if pr is not None and pr > 0:
            updates["principal_remaining"] = 0
        # Preferred also carries `count` separately; zero it too so
        # cards.py doesn't render a closed-but-counted row.
        if t == "preferred":
            count = _to_float(out.get("count"))
            if count is not None and count > 0:
                updates["count"] = 0
    elif t in ("shelf", "atm", "equity_line"):
        # Capacity-typed instruments: termination means no more
        # take-downs can occur. Zero remaining_capacity_usd; keep
        # drawn_usd as-is so the cards card still reflects what was
        # raised before termination.
        remaining = _to_float(out.get("remaining_capacity_usd"))
        if remaining is not None and remaining > 0:
            updates["remaining_capacity_usd"] = 0
    return updates


def _confident_close_reason(row: dict, as_of_date: str) -> str | None:
    """Tier 1 auto-close decision for an extra_in_ledger row.

    Returns a CloseReason ('expired' | 'redeemed' | 'terminated') when
    independent evidence confirms the row is dead, or None when the row
    might still be live and should stay open.

    Signals (any one suffices):
      * warrant      — terms.expiration on or before as_of_date → 'expired'
      * convertible  — terms.maturity on or before as_of_date  → 'redeemed'
      * preferred    — terms.maturity on or before as_of_date  → 'redeemed'
      * shelf        — effect_date + 3 years on or before as_of_date
                       (Rule 415(a)(5)) → 'expired'
      * atm, eloc    — agreement_date + 3 years past (heuristic; most
                       Sales/Purchase Agreements terminate within 3
                       years if not re-upped) → 'terminated'
      * warrant/conv — outstanding.count or .principal_remaining
                       already at 0 → 'terminated' (UNLESS the warrant
                       is unexpired / the note unmatured — a pre-term
                       zero is often the walker's own bad amend, not
                       corroborated death)
      * preferred    — outstanding.count at 0 → 'terminated'; its
                       principal_remaining is NOT a kill-signal (it is
                       structurally 0 for equity preferred even while
                       shares remain outstanding)
    Note: shelf-family rows are NOT auto-closed on
    remaining_capacity_usd == 0. An exhausted ATM / shelf is still
    legally the same Sales / Purchase / Registration Agreement until
    the issuer files a Form RW, the 3-year window elapses, or the
    filing explicitly states termination. Closing on exhaustion alone
    causes spurious closures that then prevent subsequent periodics
    from matching the same instrument (they fall into closed-pool
    re-disclosure handling, but the row is still mis-marked).

    A perpetual preferred (no maturity) and a warrant with no expiration
    fall through to None on the date checks; if their outstanding is
    also nonzero they stay open and rely on Tier 2 (consecutive misses)
    for eventual closure.
    """
    t = (row.get("type") or "").lower()
    terms = _terms_dict(row)
    out = _outstanding_dict(row)
    as_of = _normalize_date(as_of_date)

    if t == "warrant":
        exp = _normalize_date(terms.get("expiration"))
        if as_of and exp and exp <= as_of:
            return "expired"
    elif t in ("convertible", "preferred"):
        mat = _normalize_date(terms.get("maturity"))
        if as_of and mat and mat <= as_of:
            return "redeemed"
    elif t == "shelf":
        # SEC Rule 415(a)(5): shelves expire 3 years after the SEC
        # declares them effective. `effect_date` is the canonical
        # signal (from the EFFECT filing). Fall back to created_at +
        # 3y when no EFFECT has reached us, which is conservative —
        # the shelf is no later than created_at + 3 years.
        anchor_date = (
            _normalize_date(terms.get("effect_date"))
            or _normalize_date(row.get("created_at"))
        )
        if as_of and anchor_date:
            try:
                ad = date.fromisoformat(anchor_date)
                expires = ad.replace(year=ad.year + 3).isoformat()
                if expires <= as_of:
                    return "expired"
            except ValueError:
                pass
    elif t in ("atm", "equity_line"):
        # ATMs and equity lines typically terminate at the end of the
        # Sales / Purchase Agreement (often a 3-year term). If we have
        # the agreement_date and 3 years have passed, treat as
        # terminated unless the filing explicitly re-upped.
        anchor_date = (
            _normalize_date(terms.get("agreement_date"))
            or _normalize_date(row.get("created_at"))
        )
        if as_of and anchor_date:
            try:
                ad = date.fromisoformat(anchor_date)
                expires = ad.replace(year=ad.year + 3).isoformat()
                if expires <= as_of:
                    return "terminated"
            except ValueError:
                pass
    elif t == "s1_offering":
        # One-shot registered takedown — no maturity/expiration and not
        # an overhang category, so it is never matched or date-closed.
        # Close once it has gone unmentioned for S1_OFFERING_STALE_
        # CLOSE_DAYS, measured from last_seen_date (fallback created_at)
        # so continued re-disclosure resets the clock.
        anchor_date = (
            _normalize_date(row.get("last_seen_date"))
            or _normalize_date(row.get("created_at"))
        )
        if as_of and anchor_date:
            try:
                ad = date.fromisoformat(anchor_date)
                expires = date.fromordinal(
                    ad.toordinal() + S1_OFFERING_STALE_CLOSE_DAYS
                ).isoformat()
                if expires <= as_of:
                    return "terminated"
            except ValueError:
                pass

    # Tranche-typed zero-outstanding closure (warrant/convertible/preferred).
    # Shelf-family is deliberately excluded — see docstring note.
    if t in _TRANCHE_TYPES:
        count = _to_float(out.get("count"))
        if count is not None and count <= 0:
            # Symmetric with the convertible unmatured-zero guard below:
            # an UNEXPIRED warrant at count 0 is not a confident kill —
            # the zero is frequently the walker's own doing (CETY
            # round-5: the Aug-2022 Jefferson 2,894 warrant was
            # zero-amended by a 10-Q/A with no filing basis, then
            # tier-1 'terminated' here while its expiration was a year
            # out). Past expiration the zero corroborates death; before
            # it require an explicit narrative close instead.
            if t == "warrant":
                exp = _normalize_date(terms.get("expiration"))
                if exp and as_of and exp > as_of:
                    return None
            return "terminated"
        # principal_remaining is a kill-signal only for principal-denominated
        # tranches (convertible notes). Equity preferred carries its value in
        # `count` (shares outstanding) × stated/liquidation value; its
        # principal_remaining is structurally 0/None even while shares remain
        # outstanding — e.g. a debt-exchange Series D issued *for* shares, never
        # given a principal. Using it here wiped live preferreds with a positive
        # count (IQST P-125: 18,020 Series D shares closed because a conversion
        # event drove principal_remaining to 0.0). For preferred, the `count`
        # check above is the sole outstanding kill-signal.
        if t != "preferred":
            principal_remaining = _to_float(out.get("principal_remaining"))
            if principal_remaining is not None and principal_remaining <= 0:
                # An UNMATURED note sitting at 0 is not a confident
                # kill. The zero is frequently the walker's own doing —
                # a single periodic-filing amend zeroing a live note
                # (CETY round-4: the Apr-2025 and Jun-2025 Mast Hill
                # notes were self-zeroed then tier-1 'terminated' here,
                # erasing them from the cards while their maturities
                # were a year out). Past maturity the zero corroborates
                # death; before maturity require an explicit narrative
                # close from the walker instead.
                if t == "convertible":
                    maturity = _normalize_date(terms.get("maturity"))
                    if maturity and as_of and maturity > as_of:
                        return None
                return "terminated"

    return None


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
