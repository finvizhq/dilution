"""CRUD over dilution_ledger + the transactional apply_mutations entrypoint.

Layered above the validator: the walker calls validate_mutations to
get a (accepted, rejected) split, then hands the accepted list to
apply_mutations along with the rejection log. apply_mutations writes
both — accepted mutations mutate the ledger, rejections persist to
dilution_walk_errors.

Idempotence: apply_mutations is the only writer. The walker decides
which filings to process; the store doesn't dedupe at apply time
(that would silently swallow legitimate re-disclosures, which the
LLM is supposed to recognize as references-not-creates) — with two
carve-outs:

1. drawdown events. A single takedown is routinely disclosed in both
   an 8-K/6-K and the corresponding 424B (and sometimes a later
   periodic), and the LLM doesn't have ledger context to suppress
   the re-extractions. _drawdown_already_recorded short-circuits the
   second write. Two cases collapse: (a) same event_date, amount
   within 5% (or shares within 5% when amount is missing); (b)
   earlier event_date within a 180-day window, numbers within 0.5%
   — covers FPI signing-then-closing 6-K pairs and analogous 8-K
   patterns where the closing filing restates the same offering
   weeks later under a new event_date.

2. create_instrument re-disclosures. The same offering / facility /
   shelf is routinely disclosed across an announce filing, a pricing
   424B, a closing filing, and the next periodic. The walker prompt
   instructs the LLM to amend instead of re-creating, but the LLM
   gets it wrong on form-specific edge cases (F-3/A vs F-3, signing/
   closing 6-K pairs, 8-K + same-day 424B5 ATM creation). The
   _create_already_recorded gate is the deterministic safety net:
   when a new `create_instrument` matches an existing ACTIVE row by
   per-type key fields (strike for warrants, capacity_usd for ATMs,
   etc.) within an event-date window, the create is silently
   collapsed onto the existing row — last_seen_* is bumped and a
   history entry with action='redisclosed' is appended. Tolerances
   match the prompt's MATCHING KEYS (±2% price, ±5% amount, ±60d
   typical / ±180d for shelves to cover slow /A chains).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import date as _d
from typing import Any, Iterable

from db import get_conn, now_iso

from .mutations import (
    AmendInstrument,
    ApplySplit,
    CloseInstrument,
    CreateInstrument,
    Mutation,
    RecordEvent,
)
from .validate import (
    ValidationReport,
    sort_mutations,
    validate_mutations,
)

log = logging.getLogger(__name__)


# ─── ID allocator ────────────────────────────────────────────────────
# One prefix per instrument type. Mirrors the human-readable ids on
# the dashboard; the walker emits these by handing the type to
# next_instrument_id and letting the store maintain the sequence.
_TYPE_PREFIX = {
    "warrant": "W",
    "convertible": "C",
    "preferred": "P",
    "atm": "ATM",
    "equity_line": "EL",
    "shelf": "SH",
    "s1_offering": "S1",
    "equity": "EQ",
}


# Field-mapping rules used by apply_split. Walker rewrites these
# fields on every active warrant / convertible / preferred matching
# the split's units. Keys live under `terms_json` and `outstanding_json`.
_COUNT_FIELDS = (
    "count", "outstanding_count", "shares", "total_shares",
    "exercised_to_date", "terminated_to_date",
)
_PRICE_FIELDS = (
    "strike", "warrant_strike", "conv_price", "conversion_price",
    "stated_value", "liquidation_preference", "ipo_price",
)
# Split-adjusted shares are always whole; prices round to 6 dp to keep
# pre-funded $0.0001 strikes representable while squashing float drift.
_PRICE_DECIMALS = 6


@dataclass
class ApplyResult:
    """Outcome of applying one filing's mutation list."""

    accepted: int = 0
    rejected: int = 0
    created_ids: list[str] = field(default_factory=list)
    drawdowns_recorded: int = 0
    splits_applied: int = 0
    redisclosures: int = 0


# ─── Public CRUD ─────────────────────────────────────────────────────
def get_instrument(cik: int, instrument_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM dilution_ledger WHERE cik=? AND instrument_id=?",
            (cik, instrument_id),
        ).fetchone()
    return _decode_row(row)


def get_open_instruments(
    cik: int,
    *,
    types: Iterable[str] | None = None,
    include_recent_closed_days: int = 180,
    today: _d | None = None,
) -> list[dict]:
    """Open instruments + closed-but-recent. The walker shows these to
    the LLM at each filing so the prompt can route mutations correctly.

    `include_recent_closed_days` controls how far back closed
    instruments stay in the view; older closed rows are hidden by
    default (the view layer has its own further-compaction logic, but
    this is the first cut)."""
    today = today or _d.today()
    cutoff = (today.toordinal() - include_recent_closed_days)
    cutoff_iso = _d.fromordinal(cutoff).isoformat()
    type_clause = ""
    args: list[Any] = [cik]
    if types:
        types = list(types)
        type_clause = f" AND type IN ({','.join('?' * len(types))})"
        args.extend(types)
    q = f"""SELECT * FROM dilution_ledger
            WHERE cik=?{type_clause}
              AND (status='active'
                   OR (status_at IS NOT NULL AND status_at >= ?))
            ORDER BY type, created_at, instrument_id"""
    args.append(cutoff_iso)
    with get_conn() as conn:
        rows = conn.execute(q, args).fetchall()
    return [_decode_row(r) for r in rows if r]


def get_drawdowns_by_instrument(cik: int) -> dict[str, list[dict]]:
    """Per-instrument list of recorded drawdowns, newest-first.

    The walker view feeds these to the LLM so it can apply its
    `(instrument_id, ±10d, ±5%)` re-disclosure dedup rule against
    actual prior takedowns instead of an opaque `drawn_usd` total.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT instrument_id, event_date, amount_usd, shares,
                      price, counterparty, counterparty_canonical,
                      accession_number
                 FROM dilution_ledger_drawdowns
                WHERE cik=?
                ORDER BY instrument_id, event_date DESC, id DESC""",
            (cik,),
        ).fetchall()
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["instrument_id"], []).append(dict(r))
    return out


def get_walk_state(cik: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM dilution_walk_state WHERE cik=?", (cik,),
        ).fetchone()
    if not row:
        return None
    out = dict(row)
    out["next_id_seq"] = json.loads(out.get("next_id_seq_json") or "{}")
    return out


def reset_walk_state(cik: int) -> None:
    """Drop the ledger + walk state for a CIK. Used by --force re-runs."""
    with get_conn() as conn:
        conn.execute("DELETE FROM dilution_ledger_drawdowns WHERE cik=?", (cik,))
        conn.execute("DELETE FROM dilution_ledger_narrative "
                     "WHERE instrument_id IN "
                     "(SELECT instrument_id FROM dilution_ledger WHERE cik=?)",
                     (cik,))
        conn.execute("DELETE FROM dilution_ledger WHERE cik=?", (cik,))
        conn.execute("DELETE FROM dilution_walk_state WHERE cik=?", (cik,))
        conn.execute("DELETE FROM dilution_anchor_diffs WHERE cik=?", (cik,))
        conn.execute("DELETE FROM dilution_walk_errors WHERE cik=?", (cik,))


def mark_walked(
    cik: int, accession: str, filing_date: str, version: str,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO dilution_walk_state
                 (cik, last_processed_accession, last_processed_filing_date,
                  next_id_seq_json, pipeline_version, walked_at)
               VALUES (?, ?, ?, COALESCE(
                 (SELECT next_id_seq_json FROM dilution_walk_state WHERE cik=?),
                 '{}'), ?, ?)
               ON CONFLICT(cik) DO UPDATE SET
                 last_processed_accession=excluded.last_processed_accession,
                 last_processed_filing_date=excluded.last_processed_filing_date,
                 pipeline_version=excluded.pipeline_version,
                 walked_at=excluded.walked_at""",
            (cik, accession, filing_date, cik, version, now_iso()),
        )


def record_anchor_diffs(
    cik: int, accession: str, as_of_date: str, diffs: list[dict],
) -> None:
    """Persist a batch of reconciliation discrepancies. v1 always
    overwrites the ledger to match the filing — the diff rows are
    the audit trail."""
    if not diffs:
        return
    now = now_iso()
    with get_conn() as conn:
        conn.executemany(
            """INSERT INTO dilution_anchor_diffs
                 (cik, accession_number, as_of_date, diff_kind,
                  instrument_id, category, ledger_value_json,
                  filing_value_json, resolution, detected_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (cik, accession, as_of_date, d["diff_kind"],
                 d.get("instrument_id"), d.get("category"),
                 _to_json(d.get("ledger_value")),
                 _to_json(d.get("filing_value")),
                 d.get("resolution", "overwrite"), now)
                for d in diffs
            ],
        )


# ─── apply_mutations: the central transactional write ────────────────
def apply_mutations(
    *, cik: int, ticker: str, accession: str, form: str,
    filing_date: str, mutations: list[Mutation],
    pre_validated_report: ValidationReport | None = None,
) -> ApplyResult:
    """Apply a filing's mutation list against the ledger.

    Validates (or accepts a pre-computed report from the walker),
    sorts into apply order, then executes inside a single sqlite
    transaction so we either commit the whole filing's worth of
    mutations or none of them.

    Rejected mutations land in dilution_walk_errors; accepted ones
    mutate the ledger.
    """
    result = ApplyResult()

    if pre_validated_report is None:
        snapshot = {row["instrument_id"]: row
                    for row in get_open_instruments(cik)}
        report = validate_mutations(mutations, snapshot, filing_form=form)
    else:
        report = pre_validated_report

    accepted = sort_mutations(report.accepted)

    with get_conn() as conn:
        for r in report.rejected:
            conn.execute(
                """INSERT INTO dilution_walk_errors
                     (cik, accession_number, error_kind, message,
                      mutation_json, detected_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (cik, accession, r.error_kind or "unknown",
                 r.message, _to_json(_dump_mutation(r.mutation)),
                 now_iso()),
            )
            result.rejected += 1

        # Per-mutation handlers all read/write through `conn` so the
        # whole filing commits atomically.
        #
        # id_remap closes the validator/apply gap: the validator only
        # sees an LLM-emitted proposed_id, but _apply_create may
        # collapse the create onto an existing row (re-disclosure dedup)
        # or reallocate the id on collision. Either way, downstream
        # mutations in the same filing that referenced proposed_id
        # would point at a row that doesn't exist. We map proposed_id
        # → resolved_id at apply time and rewrite the references.
        seq_state = _load_seq(conn, cik)
        id_remap: dict[str, str] = {}
        for m in accepted:
            try:
                if isinstance(m, (AmendInstrument, RecordEvent,
                                  CloseInstrument)):
                    target = id_remap.get(m.instrument_id)
                    if target and target != m.instrument_id:
                        log.info(
                            "  remap %s instrument_id %s → %s",
                            m.kind, m.instrument_id, target,
                        )
                        m = m.model_copy(update={"instrument_id": target})
                if isinstance(m, CloseInstrument) and m.replaced_by:
                    rb = id_remap.get(m.replaced_by)
                    if rb and rb != m.replaced_by:
                        log.info(
                            "  remap close.replaced_by %s → %s",
                            m.replaced_by, rb,
                        )
                        m = m.model_copy(update={"replaced_by": rb})

                if isinstance(m, ApplySplit):
                    n = _apply_split(conn, cik, m, accession, form, filing_date)
                    result.splits_applied += 1
                    log.debug("  split %s ratio=%.4f touched=%d",
                              m.direction, m.ratio, n)
                elif isinstance(m, CreateInstrument):
                    new_id, was_redisclosure = _apply_create(
                        conn, cik, ticker, m, accession, form,
                        filing_date, seq_state,
                    )
                    if was_redisclosure:
                        result.redisclosures += 1
                    else:
                        result.created_ids.append(new_id)
                    if m.proposed_id and m.proposed_id != new_id:
                        id_remap[m.proposed_id] = new_id
                elif isinstance(m, AmendInstrument):
                    _apply_amend(
                        conn, cik, m, accession, form, filing_date,
                    )
                elif isinstance(m, RecordEvent):
                    drew = _apply_record_event(
                        conn, cik, m, accession, form, filing_date,
                    )
                    if drew:
                        result.drawdowns_recorded += 1
                elif isinstance(m, CloseInstrument):
                    _apply_close(
                        conn, cik, m, accession, form, filing_date,
                    )
                result.accepted += 1
            except Exception as exc:  # apply-time failure
                conn.execute(
                    """INSERT INTO dilution_walk_errors
                         (cik, accession_number, error_kind, message,
                          mutation_json, detected_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (cik, accession, "apply_error", str(exc),
                     _to_json(_dump_mutation(m)), now_iso()),
                )
                log.warning("  apply error on %s: %s", m.kind, exc)
                result.rejected += 1
        _save_seq(conn, cik, seq_state)

    return result


# ─── create_instrument re-disclosure dedup ───────────────────────────
# Tolerances mirror the walker prompt's MATCHING KEYS — strike/conv
# price within ±2%, dollar amounts (capacity, principal) within ±5%.
_CREATE_PRICE_TOLERANCE = 0.02
_CREATE_AMOUNT_TOLERANCE = 0.05

# Default event-date window. 60d covers signing-then-closing 6-K
# pairs (typically <30d apart) and 8-K + same-day-or-week 424B
# announce/price patterns without collapsing genuinely distinct
# tranches issued months apart.
_CREATE_REDISCLOSURE_WINDOW_DAYS = 60

# Shelves are a special case: a base S-3/F-3 commonly has a chain of
# /A amendments stretching 3-6 months before going effective. Wider
# window to catch those even when the LLM (despite the prompt) emits
# create_instrument for the /A.
_SHELF_REDISCLOSURE_WINDOW_DAYS = 180


def _create_already_recorded(
    conn: sqlite3.Connection, cik: int, m: CreateInstrument,
    filing_date: str, accession: str | None = None,
) -> str | None:
    """Return the matching ledger id when this create looks like a
    re-disclosure of an existing active instrument; None otherwise.

    Per-type dedup keys (in addition to type equality + active status):
      warrant       — strike within ±2%
      convertible   — conv_price within ±2%
      preferred     — conv_price OR liquidation_preference match
      shelf         — base form (S-3/F-3) match + capacity within ±5%,
                      wider 180d window for /A amendment chains
      atm           — capacity within ±5%
      equity_line   — capacity within ±5%
      s1_offering   — anticipated_deal_size within ±5%
      equity        — count within ±5% AND price within ±2% when both
                      sides have price

    Same-accession fallback: when the LLM emits two `create_instrument`
    calls of the same type from one filing AND the price-key dedup can't
    fire because both sides are missing the discriminator (conv_price /
    strike / capacity all null), collapse on series_letter equality
    (preferred) or label equality (any type). This catches LLM-side
    duplicates where a filing's "Description of securities" and
    "Subsequent events" sections describe the same instrument twice.
    """
    new_event = m.event_date or filing_date
    if not new_event:
        return None
    try:
        new_d = _d.fromisoformat(new_event[:10])
    except (ValueError, TypeError):
        return None

    window = (_SHELF_REDISCLOSURE_WINDOW_DAYS
              if m.type == "shelf"
              else _CREATE_REDISCLOSURE_WINDOW_DAYS)

    rows = conn.execute(
        """SELECT instrument_id, created_at, created_accession,
                  label, terms_json, outstanding_json
             FROM dilution_ledger
            WHERE cik=? AND type=? AND status='active'""",
        (cik, m.type),
    ).fetchall()

    new_terms = m.terms or {}
    new_out = m.outstanding or {}
    new_label = (m.label or "").strip().lower()
    new_series = (new_terms.get("series_letter") or "").strip().upper()
    for r in rows:
        try:
            existing_d = _d.fromisoformat((r["created_at"] or "")[:10])
        except (ValueError, TypeError):
            continue
        if abs((new_d - existing_d).days) > window:
            continue
        existing_terms = json.loads(r["terms_json"] or "{}")
        existing_out = json.loads(r["outstanding_json"] or "{}")
        if _create_keys_match(m.type, new_terms, new_out,
                              existing_terms, existing_out):
            return r["instrument_id"]
        # Same-accession LLM-duplicate fallback. Only fires when the
        # primary key truly can't decide — both rows from the same
        # filing missing the discriminator.
        if (accession and r["created_accession"] == accession
                and not _has_discriminator(m.type, new_terms, new_out)
                and not _has_discriminator(
                    m.type, existing_terms, existing_out)):
            existing_label = (r["label"] or "").strip().lower()
            existing_series = (
                existing_terms.get("series_letter") or "").strip().upper()
            if m.type == "preferred" and new_series and existing_series:
                if new_series == existing_series:
                    return r["instrument_id"]
            elif new_label and existing_label and new_label == existing_label:
                return r["instrument_id"]
            elif not new_label and not existing_label:
                # Both label-less rows from one filing with no
                # discriminator — treat as duplicate.
                return r["instrument_id"]
    return None


def _has_discriminator(type_: str, terms: dict, out: dict) -> bool:
    """Does this row have the price/capacity field that normally
    drives dedup? Used to decide whether to fall back to same-accession
    matching."""
    if type_ == "warrant":
        return (terms.get("strike") is not None
                or terms.get("warrant_strike") is not None)
    if type_ == "convertible":
        return (terms.get("conv_price") is not None
                or terms.get("conversion_price") is not None)
    if type_ == "preferred":
        return (terms.get("conv_price") is not None
                or terms.get("conversion_price") is not None
                or terms.get("liquidation_preference") is not None)
    if type_ in ("atm", "equity_line", "shelf"):
        return terms.get("capacity_usd") is not None
    if type_ == "s1_offering":
        return terms.get("anticipated_deal_size") is not None
    if type_ == "equity":
        return out.get("count") is not None
    return False


def _create_keys_match(
    type_: str, new_t: dict, new_o: dict,
    old_t: dict, old_o: dict,
) -> bool:
    if type_ == "warrant":
        return _close(
            new_t.get("strike") or new_t.get("warrant_strike"),
            old_t.get("strike") or old_t.get("warrant_strike"),
            _CREATE_PRICE_TOLERANCE,
        )
    if type_ == "convertible":
        return _close(
            new_t.get("conv_price") or new_t.get("conversion_price"),
            old_t.get("conv_price") or old_t.get("conversion_price"),
            _CREATE_PRICE_TOLERANCE,
        )
    if type_ == "preferred":
        if _close(
            new_t.get("conv_price") or new_t.get("conversion_price"),
            old_t.get("conv_price") or old_t.get("conversion_price"),
            _CREATE_PRICE_TOLERANCE,
        ):
            return True
        return _close(
            new_t.get("liquidation_preference"),
            old_t.get("liquidation_preference"),
            _CREATE_AMOUNT_TOLERANCE,
        )
    if type_ == "shelf":
        # Base form (S-3 vs F-3) must match — they are distinct
        # registration regimes for distinct issuer classes.
        new_base = (new_t.get("form") or "").upper().split("/")[0]
        old_base = (old_t.get("form") or "").upper().split("/")[0]
        if new_base and old_base and new_base != old_base:
            return False
        if _close(
            new_t.get("capacity_usd"), old_t.get("capacity_usd"),
            _CREATE_AMOUNT_TOLERANCE,
        ):
            return True
        # Some shelves are denominated in shares (e.g. resale registrations
        # of a fixed share count) rather than dollars. Match on
        # `capacity_shares` when both sides expose it. Without this fall-
        # back, two same-day disclosures of the same N-share registration
        # (8-K + 424B3 pair) both pass dedup with capacity_usd=None and
        # land as distinct shelf rows.
        return _close(
            new_t.get("capacity_shares"), old_t.get("capacity_shares"),
            _CREATE_AMOUNT_TOLERANCE,
        )
    if type_ in ("atm", "equity_line"):
        return _close(
            new_t.get("capacity_usd"), old_t.get("capacity_usd"),
            _CREATE_AMOUNT_TOLERANCE,
        )
    if type_ == "s1_offering":
        return _close(
            new_t.get("anticipated_deal_size"),
            old_t.get("anticipated_deal_size"),
            _CREATE_AMOUNT_TOLERANCE,
        )
    if type_ == "equity":
        if not _close(new_o.get("count"), old_o.get("count"),
                      _CREATE_AMOUNT_TOLERANCE):
            return False
        np = new_t.get("price_per_share") or new_t.get("price")
        op = old_t.get("price_per_share") or old_t.get("price")
        # Price is a strong secondary signal when both sides carry it.
        # Allow None on either side (a private placement may disclose
        # count but not price across re-disclosures).
        if np is not None and op is not None:
            return _close(np, op, _CREATE_PRICE_TOLERANCE)
        return True
    return False


def _close(a, b, tol: float) -> bool:
    """Tolerance match. Both must be present; treats None on either
    side as no-match (so absent fields don't accidentally collapse)."""
    try:
        a = float(a) if a is not None else None
        b = float(b) if b is not None else None
    except (TypeError, ValueError):
        return False
    if a is None or b is None:
        return False
    if a == 0 and b == 0:
        return True
    denom = max(abs(a), abs(b))
    return denom > 0 and abs(a - b) / denom <= tol


def _append_redisclosure(
    conn: sqlite3.Connection, instrument_id: str,
    m: CreateInstrument, accession: str, form: str, filing_date: str,
) -> None:
    """Record a re-disclosure of an existing instrument: bump
    last_seen_* and append a history entry. Terms/outstanding are
    NOT touched — the LLM should emit `amend_instrument` to update
    them; this gate is purely for collapsing duplicate creates."""
    row = conn.execute(
        "SELECT history_json FROM dilution_ledger WHERE instrument_id=?",
        (instrument_id,),
    ).fetchone()
    history = json.loads((row["history_json"] if row else None) or "[]")
    history.append({
        "date": m.event_date or filing_date,
        "accession": accession,
        "form": form,
        "action": "redisclosed",
        "fields_changed": {},
    })
    conn.execute(
        """UPDATE dilution_ledger
             SET history_json=?, last_seen_accession=?, last_seen_date=?
           WHERE instrument_id=?""",
        (_to_json(history), accession, filing_date, instrument_id),
    )


# ─── Mutation appliers ───────────────────────────────────────────────
def _apply_create(
    conn: sqlite3.Connection, cik: int, ticker: str,
    m: CreateInstrument, accession: str, form: str,
    filing_date: str, seq_state: dict[str, int],
) -> tuple[str, bool]:
    """Insert a new instrument, OR collapse onto an existing active
    row when this create looks like a re-disclosure of one already on
    the ledger.

    Returns (resolved_id, was_redisclosure). resolved_id is the actual
    ledger row this create resolved to — the new id on insert, the
    existing id on collapse, or a freshly-allocated id when the LLM's
    proposed_id collided with an existing row. apply_mutations uses
    this to remap any downstream amend/record_event/close mutations in
    the same filing that referenced m.proposed_id, since the validator
    only sees proposed_ids and can't predict the collapse/realloc.
    """
    existing_id = _create_already_recorded(
        conn, cik, m, filing_date, accession=accession,
    )
    if existing_id is not None:
        log.info(
            "  redisclosure: %s create (type=%s) collapsed onto %s",
            accession, m.type, existing_id,
        )
        _append_redisclosure(
            conn, existing_id, m, accession, form, filing_date,
        )
        return existing_id, True
    instrument_id = m.proposed_id
    if instrument_id and _exists(conn, instrument_id):
        # The LLM proposed an id that already exists. This typically
        # means the LLM thought it was creating something new but the
        # instrument is already on the ledger — likely a misclassified
        # re-disclosure. Reallocate so we don't UNIQUE-fail; the
        # orphaned record is captured for review by walk_errors.
        log.info("  proposed_id %s in use — reallocating", instrument_id)
        instrument_id = None
    if instrument_id:
        # Honor the LLM's proposed id, but bump the sequence so the
        # next walker-allocated id of this type doesn't collide.
        # Without this step, future _allocate_id calls would re-emit
        # the same number and trip a UNIQUE constraint at INSERT time
        # (the original FCEL bug).
        _bump_seq_to_match(seq_state, instrument_id)
    else:
        # Walker allocates. Sync to the global max first because
        # instrument_id is a global PK but seq_state is per-cik, so
        # without this a fresh ticker would collide with id ranges
        # owned by other tickers. Defensive loop afterwards covers
        # bumps from LLM-proposed ids honored earlier in this txn.
        _sync_seq_floor_global(conn, m.type, seq_state)
        for _ in range(100):
            instrument_id = _allocate_id(seq_state, m.type)
            if not _exists(conn, instrument_id):
                break
        else:
            raise RuntimeError(
                f"could not allocate a free instrument_id for type "
                f"{m.type!r} after 100 attempts"
            )
    history = [{
        "date": m.event_date or filing_date,
        "accession": accession,
        "form": form,
        "action": "created",
        "fields_changed": {"terms": dict(m.terms),
                           "outstanding": dict(m.outstanding)},
    }]
    conn.execute(
        """INSERT INTO dilution_ledger
             (instrument_id, ticker, cik, type, created_at,
              created_accession, counterparty, counterparty_canonical,
              placement_agent, placement_agent_canonical, label,
              terms_json, outstanding_json, status, status_at,
              history_json, last_seen_accession, last_seen_date)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)""",
        (
            instrument_id, ticker, cik, m.type,
            m.event_date or filing_date, accession,
            m.counterparty, m.counterparty_canonical,
            m.placement_agent, m.placement_agent_canonical, m.label,
            _to_json(dict(m.terms)),
            _to_json(dict(m.outstanding)),
            m.event_date or filing_date,
            _to_json(history),
            accession, filing_date,
        ),
    )
    return instrument_id, False


def _apply_amend(
    conn: sqlite3.Connection, cik: int, m: AmendInstrument,
    accession: str, form: str, filing_date: str,
) -> None:
    row = _fetch(conn, cik, m.instrument_id)
    terms = json.loads(row["terms_json"] or "{}")
    changed: dict[str, Any] = {}
    for k, v in (m.field_updates or {}).items():
        prev = terms.get(k)
        if prev != v:
            changed[k] = {"from": prev, "to": v}
        if v is None:
            terms.pop(k, None)
        else:
            terms[k] = v
    history = json.loads(row["history_json"] or "[]")
    history.append({
        "date": m.event_date or filing_date,
        "accession": accession, "form": form, "action": "amended",
        "fields_changed": changed,
    })
    conn.execute(
        """UPDATE dilution_ledger
             SET terms_json=?, history_json=?,
                 last_seen_accession=?, last_seen_date=?
           WHERE instrument_id=?""",
        (_to_json(terms), _to_json(history),
         accession, filing_date, m.instrument_id),
    )


_DRAWDOWN_DEDUP_TOLERANCE = 0.05
_DRAWDOWN_REDISCLOSURE_TOLERANCE = 0.005
_DRAWDOWN_REDISCLOSURE_WINDOW_DAYS = 180


def _drawdown_already_recorded(
    conn: sqlite3.Connection, cik: int, instrument_id: str,
    event_date: str | None, amount: float, shares: float,
) -> bool:
    """True iff the same takedown was already indexed against this
    instrument. Two collapse modes:

    - Same event_date: re-extraction or same-day cross-filing of one
      disclosure (8-K + 424B, etc). 5% tolerance covers minor LLM
      drift between filings of the same event.
    - Different event_date within a 180-day window: a later filing
      re-announcing a previously-booked offering (FPI signing-then-
      closing 6-K pair is the canonical case). Tighter 0.5% tolerance
      since exact-share-count + exact-amount across two filings is a
      strong "same offering" fingerprint, and we want to avoid
      collapsing genuinely distinct ATM/shelf takedowns that happen
      to land near each other in size.

    Without an event_date on the new mutation we can't dedupe."""
    if not event_date:
        return False
    rows = conn.execute(
        """SELECT amount_usd, shares, event_date
             FROM dilution_ledger_drawdowns
            WHERE cik = ? AND instrument_id = ?""",
        (cik, instrument_id),
    ).fetchall()
    try:
        new_d = _d.fromisoformat(event_date[:10])
    except (ValueError, TypeError):
        new_d = None
    for r in rows:
        prev_date = r["event_date"]
        if prev_date == event_date:
            tol = _DRAWDOWN_DEDUP_TOLERANCE
        elif new_d and prev_date:
            try:
                pd = _d.fromisoformat(prev_date[:10])
            except (ValueError, TypeError):
                continue
            if abs((new_d - pd).days) > _DRAWDOWN_REDISCLOSURE_WINDOW_DAYS:
                continue
            tol = _DRAWDOWN_REDISCLOSURE_TOLERANCE
        else:
            continue
        prev_amt = r["amount_usd"]
        if amount and prev_amt:
            denom = max(abs(amount), abs(prev_amt))
            if abs(amount - prev_amt) / denom <= tol:
                return True
            continue
        prev_sh = r["shares"]
        if shares and prev_sh:
            denom = max(abs(shares), abs(prev_sh))
            if abs(shares - prev_sh) / denom <= tol:
                return True
    return False


def _apply_record_event(
    conn: sqlite3.Connection, cik: int, m: RecordEvent,
    accession: str, form: str, filing_date: str,
) -> bool:
    """Return True if an aux drawdown row was recorded."""
    row = _fetch(conn, cik, m.instrument_id)
    outstanding = json.loads(row["outstanding_json"] or "{}")
    fields = dict(m.fields or {})
    drew = False

    if m.event_kind == "exercise":
        shares = float(fields.get("shares") or 0)
        if shares:
            outstanding["count"] = max(
                0.0, float(outstanding.get("count") or 0) - shares,
            )
            outstanding["exercised_to_date"] = (
                float(outstanding.get("exercised_to_date") or 0) + shares
            )
        action = "exercised"
    elif m.event_kind == "conversion":
        principal = float(fields.get("principal_converted") or 0)
        if principal:
            outstanding["principal_remaining"] = max(
                0.0, float(outstanding.get("principal_remaining") or 0)
                - principal,
            )
            outstanding["principal_converted_to_date"] = (
                float(outstanding.get("principal_converted_to_date") or 0)
                + principal
            )
        if "principal_remaining" in fields:
            outstanding["principal_remaining"] = float(
                fields["principal_remaining"])
        action = "converted"
    elif m.event_kind == "partial_redemption":
        amount = float(fields.get("principal_redeemed") or 0)
        if amount:
            outstanding["principal_remaining"] = max(
                0.0, float(outstanding.get("principal_remaining") or 0)
                - amount,
            )
        action = "partial_redemption"
    elif m.event_kind == "partial_termination":
        terminated_count = float(fields.get("count_terminated") or 0)
        if terminated_count:
            outstanding["count"] = max(
                0.0, float(outstanding.get("count") or 0) - terminated_count,
            )
        action = "partial_termination"
    elif m.event_kind == "drawdown":
        amount = float(
            fields.get("drawdown_amount_usd")
            or fields.get("amount_usd")
            or fields.get("gross_proceeds") or 0
        )
        shares = float(fields.get("drawdown_shares")
                       or fields.get("shares") or 0)
        if _drawdown_already_recorded(
            conn, cik, m.instrument_id, m.event_date, amount, shares,
        ):
            log.info(
                "drawdown re-disclosure suppressed: instrument=%s "
                "date=%s amount=%s accession=%s",
                m.instrument_id, m.event_date, amount, accession,
            )
            action = "drawdown_redisclosed"
        else:
            if amount:
                outstanding["drawn_usd"] = (
                    float(outstanding.get("drawn_usd") or 0) + amount
                )
                cap = float(outstanding.get("remaining_capacity_usd") or 0)
                if cap:
                    outstanding["remaining_capacity_usd"] = max(
                        0.0, cap - amount,
                    )
            if shares:
                outstanding["sold_to_date"] = (
                    float(outstanding.get("sold_to_date") or 0) + shares
                )
            # Index the drawdown for fast IB6 / utilization queries +
            # "Last Banker" lookups on shelf cards. Drawdown banker is
            # distinct from the parent instrument's counterparty (shelves
            # don't have a banker; each takedown does — read from the
            # mutation's fields, accepting common synonyms the LLM emits).
            cp = (
                fields.get("placement_agent")
                or fields.get("underwriter")
                or fields.get("banker")
                or fields.get("counterparty")
                or fields.get("agent")
            )
            cp_canon = (
                fields.get("placement_agent_canonical")
                or fields.get("counterparty_canonical")
            )
            conn.execute(
                """INSERT INTO dilution_ledger_drawdowns
                     (cik, instrument_id, accession_number, event_date,
                      amount_usd, shares, price, counterparty,
                      counterparty_canonical, detected_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (cik, m.instrument_id, accession, m.event_date,
                 amount or None, shares or None,
                 fields.get("price") or fields.get("avg_price"),
                 cp, cp_canon, now_iso()),
            )
            drew = True
            action = "drawn_down"
    else:
        action = m.event_kind  # forward-compat

    history = json.loads(row["history_json"] or "[]")
    history.append({
        "date": m.event_date or filing_date,
        "accession": accession, "form": form, "action": action,
        "fields_changed": fields,
    })
    conn.execute(
        """UPDATE dilution_ledger
             SET outstanding_json=?, history_json=?,
                 last_seen_accession=?, last_seen_date=?
           WHERE instrument_id=?""",
        (_to_json(outstanding), _to_json(history),
         accession, filing_date, m.instrument_id),
    )
    return drew


def _apply_close(
    conn: sqlite3.Connection, cik: int, m: CloseInstrument,
    accession: str, form: str, filing_date: str,
) -> None:
    status = (
        f"superseded:{m.replaced_by}"
        if m.reason == "superseded" and m.replaced_by
        else m.reason
    )
    row = _fetch(conn, cik, m.instrument_id)
    history = json.loads(row["history_json"] or "[]")
    history.append({
        "date": m.event_date or filing_date,
        "accession": accession, "form": form, "action": "closed",
        "fields_changed": {"reason": m.reason,
                           "replaced_by": m.replaced_by},
    })
    conn.execute(
        """UPDATE dilution_ledger
             SET status=?, status_at=?, history_json=?,
                 last_seen_accession=?, last_seen_date=?
           WHERE instrument_id=?""",
        (status, m.event_date or filing_date, _to_json(history),
         accession, filing_date, m.instrument_id),
    )


_SPLIT_DEDUP_WINDOW_DAYS = 30


def _split_already_applied(
    applied: list[dict], *,
    effective_date: str, direction: str,
) -> bool:
    """Has this split already been applied to this instrument?

    Match on direction + date within ±30 days. Ratio equality used
    to be required, but the LLM extracts the same real-world split
    with different ratios from different filings (closing 8-K says
    "1-for-13"; a later 10-Q footnote summarising a chain says
    "1-for-20"; a periodic re-disclosure rounds to 1.0). Including
    ratio in the key let those slip through and stack — every extra
    application multiplies strike by ~1/ratio, which is how a $2
    strike becomes $264 trillion after eight bogus passes.

    Failure modes traded:
      - over-apply (ratio-keyed dedup): catastrophic, strike × 10⁸+.
        Currently in production; this is what the change fixes.
      - under-apply (date-only dedup): two genuinely distinct splits
        within 30 days get treated as one. Anchor reconciliation
        against the next periodic filing's overhang catches and
        corrects the residual drift.
    """
    for a in applied:
        if a.get("direction") != direction:
            continue
        a_date = a.get("date")
        if not a_date or not effective_date:
            continue
        try:
            d1 = _d.fromisoformat(a_date[:10])
            d2 = _d.fromisoformat(effective_date[:10])
        except (ValueError, TypeError):
            continue
        if abs((d1 - d2).days) <= _SPLIT_DEDUP_WINDOW_DAYS:
            return True
    return False


def _apply_split(
    conn: sqlite3.Connection, cik: int, m: ApplySplit,
    accession: str, form: str, filing_date: str,
) -> int:
    """Walk every active warrant / convertible / preferred matching
    units and rescale counts × ratio, prices ÷ ratio. Idempotent via
    `_split_already_applied`: a split is considered already applied
    if a prior entry has the same direction and an effective_date
    within ±30 days (catches proxy-vs-8-K duplicates and quarterly
    re-disclosures with drifted ratios)."""
    rows = conn.execute(
        """SELECT instrument_id, type, terms_json, outstanding_json,
                  history_json, created_at
             FROM dilution_ledger
            WHERE cik=? AND status='active'
              AND type IN ('warrant', 'convertible', 'preferred')""",
        (cik,),
    ).fetchall()
    touched = 0
    for row in rows:
        # Splits effective on or before the instrument's creation are
        # already baked into the disclosed terms — the LLM extracted
        # post-split numbers from a filing dated after the split.
        # Re-applying inflates strike by 1/ratio per stale split,
        # which compounds catastrophically when the ledger walks a
        # multi-year history of pre-issuance splits.
        created_at = row["created_at"]
        if (created_at and m.effective_date
                and m.effective_date[:10] <= created_at[:10]):
            continue
        terms = json.loads(row["terms_json"] or "{}")
        # Filter to matching unit. ADS-denominated instruments carry
        # `units: "ads"`; everything else defaults to common. Stored
        # rows occasionally have a non-string `units` (LLM emitted a
        # share count by mistake) — fall back to "common" in that case.
        units_val = terms.get("units")
        instrument_units = (
            units_val.lower() if isinstance(units_val, str) else "common"
        )
        if instrument_units != m.units:
            continue
        applied = terms.get("applied_splits") or []
        if _split_already_applied(
            applied, effective_date=m.effective_date,
            direction=m.direction,
        ):
            continue  # already applied (within fuzzy-dedup window)
        outstanding = json.loads(row["outstanding_json"] or "{}")
        for f in _COUNT_FIELDS:
            if f in terms and isinstance(terms[f], (int, float)):
                terms[f] = round(terms[f] * m.ratio)
            if f in outstanding and isinstance(outstanding[f], (int, float)):
                outstanding[f] = round(outstanding[f] * m.ratio)
        for f in _PRICE_FIELDS:
            if f in terms and isinstance(terms[f], (int, float)) and m.ratio:
                terms[f] = round(terms[f] / m.ratio, _PRICE_DECIMALS)
        applied.append({"date": m.effective_date, "ratio": m.ratio,
                        "direction": m.direction})
        terms["applied_splits"] = applied
        history = json.loads(row["history_json"] or "[]")
        history.append({
            "date": m.effective_date,
            "accession": accession, "form": form,
            "action": "split_applied",
            "fields_changed": {"ratio": m.ratio,
                               "direction": m.direction,
                               "units": m.units},
        })
        conn.execute(
            """UPDATE dilution_ledger
                 SET terms_json=?, outstanding_json=?, history_json=?,
                     last_seen_accession=?, last_seen_date=?
               WHERE instrument_id=?""",
            (_to_json(terms), _to_json(outstanding), _to_json(history),
             accession, filing_date, row["instrument_id"]),
        )
        touched += 1
    return touched


# ─── helpers ─────────────────────────────────────────────────────────
def _exists(conn: sqlite3.Connection, instrument_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM dilution_ledger WHERE instrument_id=?",
        (instrument_id,),
    ).fetchone() is not None


def _fetch(
    conn: sqlite3.Connection, cik: int, instrument_id: str,
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM dilution_ledger WHERE cik=? AND instrument_id=?",
        (cik, instrument_id),
    ).fetchone()
    if row is None:
        # Should be unreachable: validator rejects unknown ids, and
        # apply_mutations remaps proposed_id → actual_id for ids the
        # store rewrote during _apply_create. Reaching here means one
        # of those guards regressed.
        raise KeyError(f"instrument {instrument_id!r} not found "
                       f"(cik={cik}) — id remap missed or validator "
                       f"saw a stale snapshot")
    return row


def _load_seq(conn: sqlite3.Connection, cik: int) -> dict[str, int]:
    row = conn.execute(
        "SELECT next_id_seq_json FROM dilution_walk_state WHERE cik=?",
        (cik,),
    ).fetchone()
    if not row:
        return {}
    return json.loads(row["next_id_seq_json"] or "{}")


def _save_seq(
    conn: sqlite3.Connection, cik: int, seq: dict[str, int],
) -> None:
    if not seq:
        return
    conn.execute(
        """INSERT INTO dilution_walk_state
             (cik, next_id_seq_json) VALUES (?, ?)
           ON CONFLICT(cik) DO UPDATE SET
             next_id_seq_json=excluded.next_id_seq_json""",
        (cik, _to_json(seq)),
    )


def _allocate_id(seq: dict[str, int], type_: str) -> str:
    prefix = _TYPE_PREFIX.get(type_, type_.upper()[:3])
    n = (seq.get(prefix) or 0) + 1
    seq[prefix] = n
    return f"{prefix}-{n:03d}"


def _sync_seq_floor_global(
    conn: sqlite3.Connection, type_: str, seq: dict[str, int],
) -> None:
    """Bump seq[prefix] up to the global max numeric tail for that
    prefix across the entire ledger. dilution_ledger.instrument_id is a
    global PRIMARY KEY but `seq_state` is per-cik, so a fresh ticker
    starting from seq[W]=0 would otherwise collide with W-001..W-NNN
    already owned by other tickers and exhaust the 100-attempt
    allocator. Idempotent: only raises the floor, never lowers it."""
    prefix = _TYPE_PREFIX.get(type_, type_.upper()[:3])
    pat = f"{prefix}-%"
    row = conn.execute(
        # `instrument_id GLOB '<prefix>-[0-9]*'` would be cleaner but
        # GLOB ranges aren't sargable on the PK index in SQLite; LIKE +
        # CAST is fast enough since the ledger is small.
        f"""SELECT MAX(CAST(SUBSTR(instrument_id, {len(prefix) + 2})
                            AS INTEGER)) AS max_n
              FROM dilution_ledger
             WHERE instrument_id LIKE ?
               AND SUBSTR(instrument_id, {len(prefix) + 2})
                   GLOB '[0-9]*'""",
        (pat,),
    ).fetchone()
    max_n = (row["max_n"] if row else None) or 0
    if max_n > (seq.get(prefix) or 0):
        seq[prefix] = max_n


def _bump_seq_to_match(seq: dict[str, int], instrument_id: str) -> None:
    """Push the per-type sequence high-water mark up to match an
    externally-supplied instrument_id (typically an LLM proposed_id we
    chose to honor). Without this, a subsequent _allocate_id call may
    re-emit the same number and produce a UNIQUE-constraint failure at
    INSERT time — the FCEL serial-create regression.

    Tolerant of unknown formats: if `instrument_id` doesn't match the
    standard `<PREFIX>-<NNN>` shape, this is a no-op (the caller is
    responsible for ensuring it doesn't collide via _exists)."""
    if "-" not in instrument_id:
        return
    prefix, _, tail = instrument_id.rpartition("-")
    try:
        n = int(tail)
    except ValueError:
        return
    if n > (seq.get(prefix) or 0):
        seq[prefix] = n


def _decode_row(row) -> dict | None:
    if row is None:
        return None
    out = dict(row)
    out["terms"] = json.loads(out.get("terms_json") or "{}")
    out["outstanding"] = json.loads(out.get("outstanding_json") or "{}")
    out["history"] = json.loads(out.get("history_json") or "[]")
    return out


def _to_json(obj) -> str:
    return json.dumps(obj or {}, separators=(",", ":"), ensure_ascii=False)


def _dump_mutation(m: Mutation) -> dict:
    return m.model_dump(mode="json")


__all__ = [
    "ApplyResult",
    "apply_mutations",
    "get_drawdowns_by_instrument",
    "get_instrument",
    "get_open_instruments",
    "get_walk_state",
    "mark_walked",
    "record_anchor_diffs",
    "reset_walk_state",
]
