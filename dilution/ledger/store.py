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
   are defined below as _CREATE_*_TOLERANCE / _*_WINDOW_DAYS: strike /
   conv-price within ±2%, dollar amounts within ±5%, event_date within
   ±60d typical (±180d for shelves to cover slow /A chains).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date as _d
from typing import Any, Iterable

from db import get_conn, now_iso

from ._label import build_label
from .mutations import (
    AmendMutation,
    ApplySplit,
    CloseInstrument,
    CreateAtm,
    CreateMutation,
    Mutation,
    RecordMutation,
    RestateAtm,
    extract_series_letter,
    mutation_to_dict,
    mutation_to_record,
    warrant_series_key,
)
from .validate import (
    ValidationReport,
    sort_mutations,
    validate_mutations,
)


def _ev(m, filing_date: str) -> str:
    """ISO event_date string from a typed mutation, falling back to the
    filing date when the mutation has no event_date of its own. The
    typed dataclass field is a `date` object; the SQL layer wants the
    ISO string."""
    ed = getattr(m, "event_date", None)
    if isinstance(ed, _d):
        return ed.isoformat()
    return filing_date


def _create_anchor(m, filing_date: str) -> str:
    """ISO anchor date for a create_instrument mutation.

    For atm and equity_line creates, the Sales-Agreement / SPA signing
    date (`agreement_date`) is the canonical economic start, not the
    filing date. DilutionTracker uses signing-date as the card's
    'Agreement Start Date' even when the filing lands weeks later, and
    keeping the ledger aligned to that convention makes redisclosure
    matching across primary + amendment + 6-K re-statement chains
    deterministic (the agreement date doesn't drift; the filing date
    does). All other create types — and any atm/equity_line missing
    agreement_date — fall back to event_date / filing_date via _ev.
    """
    if getattr(m, "type", None) in ("atm", "equity_line"):
        ad = getattr(m, "agreement_date", None)
        if isinstance(ad, _d):
            return ad.isoformat()
    return _ev(m, filing_date)


def _eff(m) -> str:
    """ISO effective_date string from an ApplySplit (always a date)."""
    return m.effective_date.isoformat() if isinstance(m.effective_date, _d) else str(m.effective_date)


log = logging.getLogger(__name__)


# ─── ID allocator ────────────────────────────────────────────────────
# One prefix per instrument type. Mirrors the human-readable ids on
# the cards; the walker emits these by handing the type to
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
    "exercised_to_date", "terminated_to_date", "initial_count",
)
_PRICE_FIELDS = (
    "strike", "warrant_strike", "conv_price", "conversion_price",
    "stated_value", "ipo_price",
)
# `liquidation_preference` deliberately excluded: walker extracts it as an
# aggregate $-amount for the series, not per-share. Splits don't change
# aggregates. The cards layer derives per-share from stated_value when it
# needs that view.
# Split-adjusted shares are always whole; prices round to 6 dp to keep
# pre-funded $0.0001 strikes representable while squashing float drift.
_PRICE_DECIMALS = 6


def _preferred_price_split_skip(terms: dict) -> set[str]:
    """$-price fields a common-stock split must NOT adjust on a preferred.

    ``stated_value`` (the per-share liquidation face) is ALWAYS fixed: a
    common split never moves a preferred's dollar face. ``conv_price`` /
    ``conversion_price`` are fixed ONLY when the series converts on a stated
    SHARE RATIO (``conversion_ratio`` present) — then the dollar conv_price is
    a derived/reference VWAP, the RATE absorbs the split, and dividing the
    price double-counts it (IQST Series D CoD §4(f): the rate adjusts, the
    $7.6447 VWAP is fixed; BNKK legacy NFH Series B, ratio 130). When NO
    ``conversion_ratio`` is stored the series converts on a per-common-share
    PRICE whose standard anti-dilution moves it proportionally with splits,
    exactly like a warrant strike — so the price IS split-adjusted (BNKK 2025
    Series B/C CoDs §7(a)/§6(d): $0.34 → $11.90, $0.5582 → $19.54 over the
    1-for-35 reverse split).

    Both split-handling sites (``_apply_split`` and the amend-time
    ``_rescale_stale_unit_amend``) call this so the two passes can never drift
    apart — the prior blanket exemption fixed only the divide and let a
    post-split filing re-quoting the raw pre-split price clobber it back.

    KNOWN GAP (tests/KNOWN_ISSUES.md): a fixed-RATIO series that stores a
    conv_price but NOT conversion_ratio (e.g. SCNI 'EIB' P-439, conv_price
    93.41 == $34,000 / 364) would be wrongly adjusted if it ever took a split.
    None such carries an applied split today; the durable fix is for the
    walker to stamp conversion_ratio on those series so this guard protects
    them.
    """
    ratio = terms.get("conversion_ratio")
    if (isinstance(ratio, (int, float)) and not isinstance(ratio, bool)
            and ratio > 0):
        return {"conv_price", "conversion_price", "stated_value"}
    return {"stated_value"}


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
    include_recent_closed_days: int = 365,
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
                      price, drawdown_party_canonical, drawdown_party_role,
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


def get_post_asof_drawn_by_instrument(
    cik: int, as_of_date: str,
) -> dict[str, float]:
    """Per-instrument sum of drawdown amount_usd with event_date strictly
    after `as_of_date`. Used by anchor reconciliation to back out
    Subsequent-Events drawdowns from the running `drawn_usd` before
    comparing against a periodic filing's as-of balance — otherwise the
    anchor would treat a correctly-booked subsequent takedown as drift
    and zero it out."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT instrument_id, SUM(amount_usd) AS drawn
                 FROM dilution_ledger_drawdowns
                WHERE cik=? AND event_date > ?
                  AND amount_usd IS NOT NULL
                GROUP BY instrument_id""",
            (cik, as_of_date),
        ).fetchall()
    return {r["instrument_id"]: float(r["drawn"] or 0) for r in rows}


def get_max_drawdown_date_by_instrument(
    cik: int,
) -> dict[str, str]:
    """Per-instrument max(event_date) across all booked drawdowns.
    Used by anchor reconciliation to gate the periodic-overhang
    is_terminated=true auto-close: an ATM that drew shares recently
    isn't actually terminated, regardless of what the overhang's
    stale agreement_end_date suggests."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT instrument_id, MAX(event_date) AS last_event
                 FROM dilution_ledger_drawdowns
                WHERE cik=? AND event_date IS NOT NULL
                GROUP BY instrument_id""",
            (cik,),
        ).fetchall()
    return {r["instrument_id"]: r["last_event"] for r in rows
            if r["last_event"]}


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
        ensure_walk_tables_conn(conn)
        conn.execute("DELETE FROM dilution_walked WHERE cik=?", (cik,))
        conn.execute("DELETE FROM dilution_anchor_diffs WHERE cik=?", (cik,))
        conn.execute("DELETE FROM dilution_walk_errors WHERE cik=?", (cik,))
        # The mutation log is a projection source, not history to preserve
        # across a --force: the re-walk is about to re-derive it from the
        # same filings, and stale rows would replay into a ledger that
        # mixes two extraction runs.
        ensure_mutation_log_conn(conn)
        conn.execute("DELETE FROM dilution_mutations WHERE cik=?", (cik,))


def reset_ledger_projection(cik: int) -> None:
    """Clear only what a mutation-log replay re-derives.

    The narrow counterpart to reset_walk_state, for
    scripts/rebuild_ledger.py. Two things it must NOT touch:

      * `dilution_walked` — the resume set. Clearing it would make the
        next incremental walk re-extract every filing at full LLM cost,
        which is the exact expense replay exists to avoid.
      * `dilution_mutations` — the log being replayed FROM.

    It does clear `dilution_walk_state`, because that holds
    `next_id_seq_json`: without resetting the id sequence the replayed
    creates allocate fresh ids (W-002 instead of W-001) and every logged
    amend/close then references an instrument that doesn't exist.
    """
    with get_conn() as conn:
        conn.execute("DELETE FROM dilution_ledger_drawdowns WHERE cik=?",
                     (cik,))
        conn.execute("DELETE FROM dilution_ledger_narrative "
                     "WHERE instrument_id IN "
                     "(SELECT instrument_id FROM dilution_ledger WHERE cik=?)",
                     (cik,))
        conn.execute("DELETE FROM dilution_ledger WHERE cik=?", (cik,))
        conn.execute("DELETE FROM dilution_walk_state WHERE cik=?", (cik,))
        conn.execute("DELETE FROM dilution_anchor_diffs WHERE cik=?", (cik,))
        conn.execute("DELETE FROM dilution_walk_errors WHERE cik=?", (cik,))


def ensure_walk_tables() -> None:
    """Idempotently create the per-accession walked set. The walker owns
    its own progress table, so it bootstraps it here rather than failing
    on a live DB that predates the table — `init_dilution_db()` (called
    only on a fresh DB) creates the same table from schema.py."""
    with get_conn() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS dilution_walked (
                   cik INTEGER NOT NULL,
                   accession_number TEXT NOT NULL,
                   filing_date TEXT,
                   pipeline_version TEXT,
                   walked_at TEXT NOT NULL,
                   PRIMARY KEY (cik, accession_number)
               )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dilution_walked_cik "
            "ON dilution_walked(cik)"
        )


def get_walked_accessions(cik: int) -> set[str]:
    """The set of accessions already walked for this CIK. The walker
    processes any in-scope filing NOT in this set (resume that is robust
    to back-filled filings, unlike the old positional marker)."""
    with get_conn() as conn:
        ensure_walk_tables_conn(conn)
        rows = conn.execute(
            "SELECT accession_number FROM dilution_walked WHERE cik=?",
            (cik,),
        ).fetchall()
    return {r["accession_number"] for r in rows}


def ensure_walk_tables_conn(conn: sqlite3.Connection) -> None:
    """ensure_walk_tables on an existing connection (no commit)."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS dilution_walked (
               cik INTEGER NOT NULL,
               accession_number TEXT NOT NULL,
               filing_date TEXT,
               pipeline_version TEXT,
               walked_at TEXT NOT NULL,
               PRIMARY KEY (cik, accession_number)
           )"""
    )


def seed_walked_from_positional(
    cik: int, ordered_filing_accessions: list[str],
) -> int:
    """One-time migration from the old positional resume marker.

    If the per-accession walked set is EMPTY for this CIK but
    dilution_walk_state still holds a `last_processed_accession`, this
    CIK was last walked by the old positional resumer. Record every
    filing up to AND INCLUDING that marker (in walk order) as walked —
    exactly what the positional logic implied — so the next incremental
    run does NOT re-walk the whole history onto a non-empty ledger.
    Filings that appear only on later runs (back-fills sorting before
    the marker) are absent from `ordered_filing_accessions` at seed time
    only if not yet fetched; once present they are simply not in the
    seeded set and get walked. Returns the number of rows seeded.

    No-op when the set already has rows (already on the new resumer) or
    when there is no prior marker (a fresh CIK — a full walk is correct).
    """
    with get_conn() as conn:
        ensure_walk_tables_conn(conn)
        has = conn.execute(
            "SELECT 1 FROM dilution_walked WHERE cik=? LIMIT 1", (cik,),
        ).fetchone()
        if has:
            return 0
        state = conn.execute(
            "SELECT last_processed_accession FROM dilution_walk_state "
            "WHERE cik=?", (cik,),
        ).fetchone()
        last = state["last_processed_accession"] if state else None
        if not last:
            return 0
        now = now_iso()
        rows: list[tuple] = []
        for acc in ordered_filing_accessions:
            rows.append((cik, acc, now))
            if acc == last:
                break
        if rows:
            conn.executemany(
                "INSERT OR IGNORE INTO dilution_walked "
                "(cik, accession_number, walked_at) VALUES (?, ?, ?)",
                rows,
            )
        log.info(
            "walker: seeded %d walked accessions for cik=%s from the "
            "legacy positional marker %s", len(rows), cik, last,
        )
        return len(rows)


def mark_walked(
    cik: int, accession: str, filing_date: str, version: str,
) -> None:
    with get_conn() as conn:
        ensure_walk_tables_conn(conn)
        # Per-accession walked set — the authoritative resume record.
        conn.execute(
            """INSERT INTO dilution_walked
                 (cik, accession_number, filing_date, pipeline_version,
                  walked_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(cik, accession_number) DO UPDATE SET
                 filing_date=excluded.filing_date,
                 pipeline_version=excluded.pipeline_version,
                 walked_at=excluded.walked_at""",
            (cik, accession, filing_date, version, now_iso()),
        )
        # Keep dilution_walk_state current too: it carries the
        # next_id_seq_json allocator (preserved via COALESCE) and a
        # last-walk marker retained for telemetry / backward compat.
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
# Mass-closure sweep guard. The walker LLM occasionally reads an unrelated
# "termination agreement" / "discontinued operations" / going-concern
# narrative in a periodic filing as a cap-table-wide wipe and emits
# close_instrument against most of the open ledger at once (CETY 10-Q
# 0001641172-25-011775: ~23 closes on one event_date — 2025-01-27 — spanning
# convertible + warrant + ATM + shelf + equity_line + s1_offering, while the
# filing's own text says the notes/warrants are OUTSTANDING). Reasons here are
# the issuer-ended set only; financing closes (exercised/converted/superseded)
# are legitimate mass events (a single 8-K can exercise 22 warrant tranches)
# and are NOT swept. A close batch on one date spanning >=5 distinct types is
# the wipe signature — zero false positives across the walked DB (the densest
# legitimate periodic close batch spans 4 types: VTAK/COCH).
_MASS_CLOSE_REAP_REASONS = frozenset({"terminated", "expired", "redeemed"})
_MASS_CLOSE_TYPE_SPREAD = 5


def _mass_closure_suppressions(
    conn: sqlite3.Connection, accepted: list, filing_date: str,
) -> set[str]:
    """instrument_ids whose close should be dropped as part of a single
    filing's cap-table-wide closure sweep (see the comment above)."""
    closes = [m for m in accepted
              if isinstance(m, CloseInstrument)
              and m.reason in _MASS_CLOSE_REAP_REASONS]
    if len(closes) < _MASS_CLOSE_TYPE_SPREAD:
        return set()
    ids = [m.instrument_id for m in closes]
    type_by_id = {
        r["instrument_id"]: r["type"] for r in conn.execute(
            f"SELECT instrument_id, type FROM dilution_ledger "
            f"WHERE instrument_id IN ({','.join('?' * len(ids))})", ids)
    }
    types_by_date: dict[str, set] = {}
    ids_by_date: dict[str, list] = {}
    for m in closes:
        t = type_by_id.get(m.instrument_id)
        if not t:
            continue  # close targets a row that doesn't exist — let it no-op
        d = (_ev(m, filing_date) or "")[:10]
        types_by_date.setdefault(d, set()).add(t)
        ids_by_date.setdefault(d, []).append(m.instrument_id)
    suppress: set[str] = set()
    for d, types in types_by_date.items():
        if len(types) >= _MASS_CLOSE_TYPE_SPREAD:
            suppress.update(ids_by_date[d])
            log.warning(
                "  mass-closure sweep: %d closes on %s span %d types %s — "
                "dropping them; row(s) stay active (cap-table-wide wipe, "
                "likely LLM misread of unrelated termination / discontinued-"
                "operations narrative)",
                len(ids_by_date[d]), d, len(types), sorted(types),
            )
    return suppress


def reopen_instruments(
    cik: int, instrument_ids: list[str], accession: str, filing_date: str,
) -> int:
    """Reactivate instruments wrongly closed by a filing whose own overhang
    still reports them outstanding (anchor-corroborated close-rejection — see
    anchor.corroborate_closes). Sets status back to active and logs a
    'reopened' history event. Skips rows already active. Returns the count
    reopened."""
    if not instrument_ids:
        return 0
    reopened = 0
    with get_conn() as conn:
        for iid in instrument_ids:
            row = conn.execute(
                "SELECT status, history_json FROM dilution_ledger "
                "WHERE cik=? AND instrument_id=?", (cik, iid),
            ).fetchone()
            if not row or row["status"] == "active":
                continue
            history = json.loads(row["history_json"] or "[]")
            history.append({
                "date": filing_date, "accession": accession,
                "action": "reopened",
                "fields_changed": {
                    "from": row["status"], "to": "active",
                    "why": "overhang re-lists as outstanding "
                           "(anchor-corroborated)",
                },
            })
            conn.execute(
                "UPDATE dilution_ledger SET status='active', status_at=?, "
                "history_json=? WHERE instrument_id=?",
                (filing_date, _to_json(history), iid),
            )
            reopened += 1
    return reopened


def apply_mutations(
    *, cik: int, ticker: str, accession: str, form: str,
    filing_date: str, mutations: list[Mutation],
    pre_validated_report: ValidationReport | None = None,
    log_mutations: bool = True,
) -> ApplyResult:
    """Apply a filing's mutation list against the ledger.

    Validates (or accepts a pre-computed report from the walker),
    sorts into apply order, then executes inside a single sqlite
    transaction so we either commit the whole filing's worth of
    mutations or none of them.

    Rejected mutations land in dilution_walk_errors; accepted ones
    mutate the ledger and are appended to dilution_mutations.

    `log_mutations=False` suppresses that append — for replay
    (scripts/rebuild_ledger.py), which is re-applying rows that are
    already IN the log. Leaving it on there would double the log on every
    rebuild, and the next replay would apply each create twice and
    allocate different instrument ids, so the ledger would drift further
    the more you tried to recover it.
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
        suppressed_closes = _mass_closure_suppressions(
            conn, accepted, filing_date)
        # Applied mutations are logged as they land, in the same
        # transaction, so the log and the ledger can never disagree.
        if log_mutations:
            ensure_mutation_log_conn(conn)
        pipeline_stamp = _pipeline_version() if log_mutations else None
        for seq, m in enumerate(accepted):
            try:
                # Mass-closure sweep: drop this close, leave the row active.
                if (isinstance(m, CloseInstrument)
                        and m.instrument_id in suppressed_closes):
                    continue
                if isinstance(m, (AmendMutation, RecordMutation,
                                  CloseInstrument)):
                    target = id_remap.get(m.instrument_id)
                    if target and target != m.instrument_id:
                        log.info(
                            "  remap %s instrument_id %s → %s",
                            m.kind, m.instrument_id, target,
                        )
                        m = dataclasses.replace(m, instrument_id=target)
                if isinstance(m, CloseInstrument) and m.replaced_by:
                    rb = id_remap.get(m.replaced_by)
                    if rb and rb != m.replaced_by:
                        log.info(
                            "  remap close.replaced_by %s → %s",
                            m.replaced_by, rb,
                        )
                        m = dataclasses.replace(m, replaced_by=rb)

                # Self-supersede guard: a close whose replaced_by (after
                # id_remap) resolves to its OWN instrument_id means the
                # successor create collapsed onto the very row being
                # closed — i.e. a re-registration of the same program, not
                # a replacement. dedup + shelf-rollover have already
                # repointed the row; closing it would wrongly retire the
                # live instrument and reject every later drawdown (the CGEN
                # SVB-ATM regression). Drop the close — the row stays
                # active. Validation can't catch this: it runs before
                # id_remap, when replaced_by still holds the successor's
                # proposed_id.
                if (isinstance(m, CloseInstrument) and m.replaced_by
                        and m.replaced_by == m.instrument_id):
                    log.warning(
                        "  close_instrument %s superseded by itself "
                        "(successor create collapsed onto it) — dropping "
                        "the close; row stays active", m.instrument_id,
                    )
                    continue

                # Idempotent supersede: the store auto-supersedes the
                # prior active ATM when the successor is created
                # (_auto_supersede_prior_atm, runs in the create branch
                # above, i.e. before this close). A walker-emitted
                # close_instrument(superseded) against the same row then
                # arrives second and would re-write the closure — clobbering
                # the store's resolved replaced_by id with the walker's
                # unremapped proposed-id slug and stacking a duplicate close
                # entry (the CGEN ATM-2143 double-close). The store owns ATM
                # supersession; if the row is already superseded, drop the
                # redundant walker close.
                if (isinstance(m, CloseInstrument)
                        and m.reason == "superseded"):
                    cur = conn.execute(
                        "SELECT status, type FROM dilution_ledger "
                        "WHERE instrument_id=?", (m.instrument_id,),
                    ).fetchone()
                    if cur and str(cur["status"] or "").startswith(
                            "superseded:"):
                        log.info(
                            "  close_instrument %s already superseded by the "
                            "store (%s) — dropping the redundant walker close",
                            m.instrument_id, cur["status"],
                        )
                        continue
                    # Phantom-successor guard: a superseded-close is only
                    # coherent when its successor actually exists — either
                    # a pre-existing ledger row or a same-filing create
                    # that id_remap just resolved. A replaced_by that
                    # resolves to nothing is a hallucinated supersession
                    # (CETY round-6: a routine 10-Q produced an
                    # 18-instrument mass-supersede batch with fabricated
                    # '<id>-restated' slugs; 'superseded' is exempt from
                    # the mass-closure guard because financing closes are
                    # legit, and every dangling chain rendered its
                    # predecessor as an extra). Drop the close — the row
                    # stays open; a real successor arriving later re-emits
                    # the supersession or the store's auto-supersede
                    # handles it. Shelves are EXEMPT: their close path
                    # resolves the successor itself (named live shelf,
                    # else newest active shelf — the CGEN rollover), so a
                    # stale replaced_by there is recoverable, not phantom.
                    if (m.replaced_by
                            and cur and (cur["type"] or "") != "shelf"
                            and not _exists(conn, m.replaced_by)):
                        log.warning(
                            "  close_instrument %s superseded by nonexistent "
                            "%s — dropping the close; row stays active",
                            m.instrument_id, m.replaced_by,
                        )
                        continue

                if isinstance(m, ApplySplit):
                    n = _apply_split(conn, cik, m, accession, form, filing_date)
                    result.splits_applied += 1
                    log.debug("  split %s ratio=%.4f touched=%d",
                              m.direction, m.ratio, n)
                elif isinstance(m, CreateMutation):
                    # Unchanged-program ATM re-registration: a 424B3/
                    # 424B5/shelf-host create naming the SAME agent and
                    # the SAME capacity as a still-active ATM is a
                    # prospectus refresh of the UNCHANGED agreement, not
                    # a new supplement window. Dedup alone misses it
                    # because the prompt dates supplements by their own
                    # filing date (agreement_date differs), so the
                    # create minted a re-dated duplicate card and the
                    # original went 'Replaced' (round-4 cety-roth: the
                    # Oct-2023 Roth ATM re-registered unchanged in
                    # Nov-2025). Collapse to a redisclosure — which also
                    # performs the shelf-rollover re-point when the host
                    # registration moved.
                    if m.type == "atm" and (form or "").upper() in (
                            _SHELF_HOST_FORMS | _ATM_SUPPLEMENT_FORMS
                            | {"424B3"}):
                        rid = _find_unchanged_atm_program(conn, cik, m)
                        if rid is not None:
                            _append_redisclosure(
                                conn, rid, m, accession, form,
                                filing_date,
                            )
                            result.redisclosures += 1
                            if m.proposed_id and m.proposed_id != rid:
                                id_remap[m.proposed_id] = rid
                            log.info(
                                "  create_atm collapsed onto %s — "
                                "unchanged program re-registration "
                                "(same agent, same capacity)", rid,
                            )
                            continue
                    new_id, was_redisclosure, eq_drew = _apply_create(
                        conn, cik, ticker, m, accession, form,
                        filing_date, seq_state,
                    )
                    if eq_drew:
                        result.drawdowns_recorded += 1
                    if was_redisclosure:
                        result.redisclosures += 1
                    else:
                        result.created_ids.append(new_id)
                        # A genuinely-new ATM registered on a fresh
                        # shelf-host filing supersedes the issuer's prior
                        # active ATM (DT's one-program-at-a-time rule).
                        # The store owns this — the walker is unreliable
                        # at emitting the close and dating the successor.
                        # A 424B5/SUPPL prospectus-supplement ATM create
                        # is a RE-REGISTRATION of an existing program
                        # (KSCP's four Wainwright supplements): chain it
                        # too, but only over SAME-AGENT priors — a
                        # different bank's concurrent program must
                        # survive. Plain supersede (no via:restate), so
                        # predecessors render as 'Replaced' while the
                        # chain head is live, exactly DT's per-supplement
                        # carding.
                        ff_create = (form or "").upper()
                        if m.type == "atm" and (
                                ff_create in _SHELF_HOST_FORMS
                                or (ff_create in _ATM_SUPPLEMENT_FORMS
                                    and m.placement_agent_canonical)):
                            _auto_supersede_prior_atm(
                                conn, cik, new_id, m, accession,
                                form, filing_date,
                                same_agent_only=(
                                    ff_create in _ATM_SUPPLEMENT_FORMS),
                            )
                    if m.proposed_id and m.proposed_id != new_id:
                        id_remap[m.proposed_id] = new_id
                elif isinstance(m, RestateAtm):
                    # An amended-and-restated sales agreement: mint a
                    # fresh successor ATM and (when the filing terminates
                    # the predecessor) supersede the named prior row. This
                    # is the explicit replacement for the old implicit
                    # amend(capacity)→create reinterpretation. id_remap
                    # routes in-filing follow-ups (drawdowns) onto the new
                    # row, and any reference to the superseded predecessor
                    # onto its successor.
                    new_id, prior_id = _apply_restate_atm(
                        conn, cik, ticker, m, accession, form,
                        filing_date, seq_state,
                    )
                    if new_id not in result.created_ids:
                        result.created_ids.append(new_id)
                    if m.proposed_id and m.proposed_id != new_id:
                        id_remap[m.proposed_id] = new_id
                    if prior_id and prior_id != new_id:
                        id_remap[prior_id] = new_id
                elif isinstance(m, AmendMutation):
                    _apply_amend(
                        conn, cik, m, accession, form, filing_date,
                    )
                elif isinstance(m, RecordMutation):
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
                # Creates report the id the store actually allocated
                # (which may differ from the LLM's proposed_id after
                # dedup / collision); everything else is already
                # id_remap-resolved by this point.
                if log_mutations:
                    _log_applied_mutation(
                        conn, cik=cik, accession=accession,
                        filing_date=filing_date, form=form, seq=seq,
                        mutation=m,
                        instrument_id=(
                            new_id
                            if isinstance(m, (CreateMutation, RestateAtm))
                            else getattr(m, "instrument_id", None)),
                        pipeline_version=pipeline_stamp,
                    )
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
# These tolerances are the authoritative re-disclosure dedup keys —
# strike/conv price within ±2%, dollar amounts (capacity, principal)
# within ±5%. The walker's dedup-candidates block only surfaces a
# hint; this gate is the backstop.
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

# Preferreds persist for years and routinely get re-disclosed in periodic
# filings (10-Ks, 10-Qs) months or quarters after issuance. The narrow
# 60d default would miss those re-creates. series_letter is the unique
# discriminator within an issuer (see _create_keys_match), so a wider
# window doesn't risk collapsing genuinely distinct tranches — Series B
# and Series D never share a letter.
_PREFERRED_REDISCLOSURE_WINDOW_DAYS = 365

# ATMs and equity lines live for the full Rule 415 shelf lifetime (3y)
# and get re-disclosed every quarter in 10-Qs and annually in 10-Ks. The
# 60d default routinely misses these re-creates — e.g. CETY's Roth ATM
# (agreement_date 2023-10-06) was re-disclosed across 5 quarterly filings
# at gaps of 63 / 147 / 307 / 102 days, producing 5 duplicate rows.
# agreement_date is the sole identity key (see _create_keys_match) and
# is the actual execution date of a specific signed contract, so a wide
# window cannot collapse genuinely distinct facilities — two ATMs with
# the same agreement_date and same issuer are by definition the same
# agreement.
_ATM_REDISCLOSURE_WINDOW_DAYS = 3 * 365

# Closed-row resurrection guard. A periodic balance sheet (10-K/10-Q/20-F)
# re-lists a tranche that has already been redeemed/terminated/converted/
# expired; the LLM emits create_instrument for it, and because the active-
# row dedup above only scans status='active', the create lands as a NEW
# ACTIVE row — resurrecting a dead instrument (XTIA P-447 Series 9 created
# active off a 2025 10-K balance sheet, duplicating the already-redeemed
# P-443). We match such a create against CLOSED rows too, but ONLY within
# a TIGHT window: a genuine re-disclosure carries the instrument's ORIGINAL
# issue_date, so its anchor nearly coincides with the closed row's
# created_at. The window must stay tight because series letters are NOT
# unique within every issuer (XTIA re-uses Series 4 across 2018/2024/2025
# and Series 5 across 2019/2025) — a wide window would collapse a genuinely
# NEW same-letter issuance onto an old closed tranche. 31d covers the
# pricing/closing/period-end slip without reaching a later re-issuance.
_CLOSED_REDISCLOSURE_WINDOW_DAYS = 31

# Types where re-listing a closed tranche as a new active row is the bug
# the guard fixes. Shelves/ATMs/equity-lines/s1 have their own lifecycle
# (rollover, supersession, expiry) handled elsewhere and are excluded.
_RESURRECTION_GUARD_TYPES = frozenset({"warrant", "preferred", "convertible"})

# A re-disclosure of an ATM / equity_line / s1_offering inside one of
# these forms means a brand-new shelf registration is now hosting the
# (same) underlying agreement. If the new filing's SEC file_number
# differs from the row's current host, we rewrite created_accession to
# follow the new shelf — drawdowns and parent_shelf lookups walk
# file_number, so without the rewrite they'd stay glued to the dead
# shelf for the rest of the row's life. Excludes 424B* prospectus
# supplements (those carry the host shelf's file_number, not a new one)
# and amendments (S-3/A etc. — same registration, not a new one).
_SHELF_HOST_FORMS = frozenset({
    "S-3", "S-3ASR", "S-3MEF",
    "F-3", "F-3ASR", "F-3MEF",
    "F-10", "F-10EF",
})
_SHELF_HOSTED_TYPES = frozenset({"atm", "equity_line", "s1_offering"})
# Prospectus-supplement forms that can carry an ATM RE-REGISTRATION of
# an existing sales agreement (new registered capacity, same agreement
# — KSCP's Wainwright chain). ATM creates from these forms chain onto
# same-agent priors via _auto_supersede_prior_atm(same_agent_only=True).
_ATM_SUPPLEMENT_FORMS = frozenset({"424B5", "SUPPL"})


def _agents_same(a: str | None, b: str | None) -> bool:
    """Punctuation-squashed containment match for banker names —
    'H.C. Wainwright' == 'HC Wainwright' == 'Wainwright'. Distinct
    banks ('Jefferies' vs 'Maxim') never contain each other."""
    def _norm(s: str) -> str:
        return "".join(c for c in s.lower() if c.isalnum())
    if not a or not b:
        return False
    na, nb = _norm(str(a)), _norm(str(b))
    return bool(na and nb and (na in nb or nb in na))



def _create_already_recorded(
    conn: sqlite3.Connection, cik: int, m: "CreateMutation",
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
      atm           — agreement_date exact match (no fallback), wider
                      3y window to cover the Rule 415 shelf lifetime
      equity_line   — agreement_date exact match (no fallback), wider
                      3y window to cover multi-year facilities
      s1_offering   — anticipated_deal_size within ±5%
      equity        — count within ±5% AND price within ±2% when both
                      sides have price

    Same-accession fallback: when the LLM emits two `create_instrument`
    calls of the same type from one filing AND the price-key dedup can't
    fire because both sides are missing the discriminator (conv_price /
    strike / capacity all null), collapse on series_letter equality
    (preferred) or canonical-label equality (any type). This catches
    LLM-side duplicates where a filing's "Description of securities" and
    "Subsequent events" sections describe the same instrument twice.
    The new-side label is canonicalized via build_label() so the
    comparison matches what was stored on the existing row at INSERT.
    """
    # Use the same anchor we'll store as created_at so the window check
    # compares like-to-like (existing created_at vs the date this new
    # create would land at). For atm/equity_line that's agreement_date;
    # for other types it's event_date/filing_date.
    new_event = _create_anchor(m, filing_date)
    if not new_event:
        return None
    try:
        new_d = _d.fromisoformat(new_event[:10])
    except (ValueError, TypeError):
        return None

    if m.type == "shelf":
        window = _SHELF_REDISCLOSURE_WINDOW_DAYS
    elif m.type == "preferred":
        window = _PREFERRED_REDISCLOSURE_WINDOW_DAYS
    elif m.type in ("atm", "equity_line"):
        window = _ATM_REDISCLOSURE_WINDOW_DAYS
    else:
        window = _CREATE_REDISCLOSURE_WINDOW_DAYS

    rows = conn.execute(
        """SELECT instrument_id, created_at, created_accession,
                  label, placement_agent_canonical,
                  terms_json, outstanding_json
             FROM dilution_ledger
            WHERE cik=? AND type=? AND status='active'""",
        (cik, m.type),
    ).fetchall()

    new_terms = m.terms or {}
    new_out = m.outstanding or {}
    # Compare against the canonical label that build_label produces at
    # INSERT — m.label is usually empty, and the stored label is always
    # canonical, so comparing raw m.label against r["label"] silently
    # misses every duplicate.
    new_label = (build_label(m) or m.label or "").strip().lower()
    new_series = (new_terms.get("series_letter") or "").strip().upper()
    new_pa = (m.placement_agent_canonical or "").strip()
    for r in rows:
        try:
            existing_d = _d.fromisoformat((r["created_at"] or "")[:10])
        except (ValueError, TypeError):
            continue
        if abs((new_d - existing_d).days) > window:
            continue
        existing_terms = json.loads(r["terms_json"] or "{}")
        existing_out = json.loads(r["outstanding_json"] or "{}")
        old_pa = r["placement_agent_canonical"] or ""
        if _create_keys_match(m.type, new_terms, new_out,
                              existing_terms, existing_out,
                              new_pa=new_pa, old_pa=old_pa):
            # Same-accession warrant safeguard. The LLM frequently emits
            # multiple distinct warrants from one filing at an identical
            # strike (e.g. SCNI 2026-04-23 6-K: Inducement and Series B
            # both at $0.55). Strike alone collapses them; require
            # initial_count to also agree when both sides expose it, so
            # tranches with wildly different sizes (458,621 vs 5,208,333)
            # stay distinct. Skipped cross-filing — genuine re-disclosure
            # usually omits initial_count on the second pass.
            if (m.type == "warrant" and accession
                    and r["created_accession"] == accession):
                new_ic = (new_out or {}).get("initial_count")
                old_ic = (existing_out or {}).get("initial_count")
                if (new_ic is not None and old_ic is not None
                        and not _close(new_ic, old_ic,
                                       _CREATE_AMOUNT_TOLERANCE)):
                    continue
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

        # Warrant split-via-misread-strike collapse. The strike key above
        # FAILS when the LLM read a different exercise price for the SAME
        # offering across disclosures — a 424B5 then a 10-Q re-statement of
        # the same tranche (CELU April-2023, created off both), or two
        # create calls from one filing (CELU March-2023). Serial-diluter
        # warrant ladders ratchet, so the strike the LLM copies drifts per
        # filing and can't anchor identity. A same canonical-LABEL + same
        # INITIAL_COUNT match within the window is a strong duplicate
        # signal instead: distinct tranches differ in issued size or
        # series_letter (the same-strike SCNI Inducement/Series B case is
        # already separated by initial_count), but ONE offering keeps its
        # issued count and month-year label. Collapse to the existing row
        # rather than spawn a phantom partial-count card (CELU 923,077 →
        # $7.50/435,625 + $3.50/487,451; 938,184 → $30 + $1.69/75,000).
        # series_letter must not conflict (one-sided/absent falls through).
        if m.type == "warrant":
            new_ic = (new_out or {}).get("initial_count")
            old_ic = (existing_out or {}).get("initial_count")
            old_label = (r["label"] or "").strip().lower()
            new_sl = warrant_series_key(new_terms.get("series_letter"))
            old_sl = warrant_series_key(existing_terms.get("series_letter"))
            if (new_ic is not None and old_ic is not None
                    and _close(new_ic, old_ic, _CREATE_AMOUNT_TOLERANCE)
                    and new_label and old_label and new_label == old_label
                    and not (new_sl and old_sl and new_sl != old_sl)):
                return r["instrument_id"]

    # Closed-row resurrection guard (see _CLOSED_REDISCLOSURE_WINDOW_DAYS).
    # A periodic re-disclosure of an already-CLOSED tranche must collapse
    # onto the dead row (which _append_redisclosure leaves closed), NOT
    # spawn a new active duplicate. Strong-key match only (the same
    # _create_keys_match the active path uses: series for preferred,
    # strike+non-conflicting-expiration for warrant, conv_price/principal
    # for convertible) within the TIGHT window, so a genuine new issuance
    # that re-uses a series letter never collapses onto an old tranche.
    if m.type in _RESURRECTION_GUARD_TYPES:
        closed = conn.execute(
            """SELECT instrument_id, created_at, terms_json, outstanding_json
                 FROM dilution_ledger
                WHERE cik=? AND type=? AND status!='active'
                  AND status NOT LIKE 'superseded%'""",
            (cik, m.type),
        ).fetchall()
        for r in closed:
            try:
                existing_d = _d.fromisoformat((r["created_at"] or "")[:10])
            except (ValueError, TypeError):
                continue
            if abs((new_d - existing_d).days) > _CLOSED_REDISCLOSURE_WINDOW_DAYS:
                continue
            existing_terms = json.loads(r["terms_json"] or "{}")
            existing_out = json.loads(r["outstanding_json"] or "{}")
            if _create_keys_match(m.type, new_terms, new_out,
                                  existing_terms, existing_out,
                                  new_pa=new_pa, old_pa=""):
                log.info(
                    "  resurrection-guard: %s create (type=%s) matched "
                    "CLOSED row %s — not re-creating as active",
                    accession, m.type, r["instrument_id"],
                )
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
        # principal is the secondary discriminator when conv_price is
        # absent (most non-pricing disclosures omit conv_price but list
        # the face amount).
        return (terms.get("conv_price") is not None
                or terms.get("conversion_price") is not None
                or terms.get("principal") is not None)
    if type_ == "preferred":
        return (terms.get("conv_price") is not None
                or terms.get("conversion_price") is not None
                or terms.get("liquidation_preference") is not None)
    if type_ in ("atm", "equity_line"):
        return terms.get("agreement_date") is not None
    if type_ == "shelf":
        return terms.get("capacity_usd") is not None
    if type_ == "s1_offering":
        # placement_agent is a secondary discriminator handled at the
        # row level (not in terms), so it isn't checked here — the
        # _create_keys_match fallback covers the no-deal_size case.
        return terms.get("anticipated_deal_size") is not None
    if type_ == "equity":
        return out.get("count") is not None
    return False


# End-date (maturity / expiration) tolerance for redisclosure matching.
# Re-disclosures of the SAME instrument state the same contractual
# end-date to the day (a 1-2 day pricing-vs-closing slip at most);
# 14d mirrors the walker guard's _DUP_EXPIRATION_TOL_DAYS. Distinct
# instruments from serial issuers reusing the same price sit much
# further apart: CETY's Feb-2025 Mast Hill warrant (exp 2030-02-28)
# collapsed onto the Jan-2025 warrant (exp 2030-01-16, 43d) on the
# strike-only key, and same-accession twins Mega Sincere / Noblebear
# (both conv $0.646) differ by six months of maturity.
_CREATE_END_DATE_TOL_DAYS = 14


def _end_dates_conflict(new_t: dict, old_t: dict) -> bool:
    """True when both sides state an instrument end-date (maturity or
    expiration) and they differ by more than the redisclosure
    tolerance — i.e. these are different instruments regardless of how
    well the price keys agree. One-sided absence is NOT a conflict
    (re-disclosures often omit the end-date on a later mention)."""
    def _end(t: dict) -> _d | None:
        raw = t.get("maturity") or t.get("expiration")
        if not raw:
            return None
        try:
            return _d.fromisoformat(str(raw)[:10])
        except (ValueError, TypeError):
            return None
    new_end = _end(new_t)
    old_end = _end(old_t)
    if new_end is None or old_end is None:
        return False
    return abs((new_end - old_end).days) > _CREATE_END_DATE_TOL_DAYS


def _create_keys_match(
    type_: str, new_t: dict, new_o: dict,
    old_t: dict, old_o: dict,
    *,
    new_pa: str = "", old_pa: str = "",
) -> bool:
    if type_ == "warrant":
        # series_letter as identity: when BOTH sides name one, they must
        # agree. Strike alone is not enough — a single filing can issue
        # an Inducement Warrant and a Series B Warrant at the same strike
        # (SCNI 2026-04-23 6-K: both at $0.55), and the second create
        # would otherwise collapse onto the first. warrant_series_key keeps
        # warrant series_letter's polymorphism: letters ("A","B"), digits
        # ("1"), and descriptive tags ("Inducement","Pre-Funded") stay
        # discriminators. A 10-Q footnote that re-states a ladder under
        # financial-statement *closing dates* drops a date label
        # ("August 23") here; warrant_series_key maps that to "" so this
        # guard goes one-sided and falls through to strike — the
        # re-disclosure-collapse behavior (GCTK Nov-2024 10-Q). One-sided
        # cases (LLM omits the tag in a later re-disclosure) likewise fall
        # through to strike.
        new_sl = warrant_series_key(new_t.get("series_letter"))
        old_sl = warrant_series_key(old_t.get("series_letter"))
        if new_sl and old_sl and new_sl != old_sl:
            return False
        # Distinct expirations ⇒ distinct warrants even at an identical
        # strike (CETY Feb-2025 Mast Hill @ $2.50 exp 2030-02-28 vs the
        # Jan-2025 tranche @ $2.50 exp 2030-01-16 — 43d apart, well
        # inside the 60d created-window). Mirrors walker _is_dup_create.
        if _end_dates_conflict(new_t, old_t):
            return False
        return _close(
            new_t.get("strike") or new_t.get("warrant_strike"),
            old_t.get("strike") or old_t.get("warrant_strike"),
            _CREATE_PRICE_TOLERANCE,
        )
    if type_ == "convertible":
        # Distinct maturities ⇒ distinct notes, before any price test:
        # serial toxic issuers place note after note with the SAME
        # lender at the SAME conv price (and same-accession twins exist:
        # Mega Sincere / Noblebear, both $0.646, maturities 6mo apart).
        if _end_dates_conflict(new_t, old_t):
            return False
        new_cp = new_t.get("conv_price") or new_t.get("conversion_price")
        old_cp = old_t.get("conv_price") or old_t.get("conversion_price")
        # Primary: conv_price authoritative when both sides have it.
        # Different conv_prices ⇒ different notes, period.
        if new_cp is not None and old_cp is not None:
            return _close(new_cp, old_cp, _CREATE_PRICE_TOLERANCE)
        # Fallback when conv_price missing on either side: principal
        # face amount is the next-best discriminator. Two convertibles
        # rarely share a principal within ±5% within the 60d re-disclosure
        # window unless they're the same note re-described.
        return _close(
            new_t.get("principal"), old_t.get("principal"),
            _CREATE_AMOUNT_TOLERANCE,
        )
    if type_ == "preferred":
        # series_letter is unique within an issuer — primary identity
        # key. Same letter on both sides → same tranche, regardless of
        # how conv_price has drifted across re-disclosures. Different
        # letters → definitively different tranches; do NOT fall back
        # to price-based dedup, which can cause a Series D create with
        # similar conv_price to a Series B to collapse incorrectly.
        new_series = extract_series_letter(new_t.get("series_letter"))
        old_series = extract_series_letter(old_t.get("series_letter"))
        if new_series and old_series:
            return new_series == old_series
        # When at least one side has no series_letter, fall back to
        # price/principal — the original v1 dedup behavior.
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
        # agreement_date equality is the primary identity key. Two ATMs
        # with different signing dates are DISTINCT instruments even
        # if capacity is similar. The prior capacity-similarity rule
        # collapsed multi-program issuers (XTIA had two Maxim ATMs at
        # ~$25M and ~$27M signed 11 months apart) into one row and
        # blocked valid drawdowns via capacity_overflow. No fallback —
        # if the walker fails to extract agreement_date, a duplicate
        # row is preferable to a silent collapse.
        new_ad = (new_t.get("agreement_date") or "").strip()[:10]
        old_ad = (old_t.get("agreement_date") or "").strip()[:10]
        if not (bool(new_ad) and bool(old_ad) and new_ad == old_ad):
            return False
        # agreement_date matches — but an issuer can AMEND the same
        # underlying Sales Agreement (capacity bump, banker rebrand)
        # via a fresh shelf years later. The amendment carries the
        # ORIGINAL signing date in the prospectus, which would defeat
        # the date key alone. CGEN's May-2026 F-3 amends the Jan-2023
        # SVB ATM: agreement_date still 2023-01-31, but capacity jumps
        # $50M→$100M and banker rebrands SVB→Leerink. Treating this as
        # a redisclosure throws away the new tranche; treat as a new
        # instrument so the walker_prompt's case-C supersede path can
        # close the old row.
        new_cap = new_t.get("capacity_usd")
        old_cap = old_t.get("capacity_usd")
        if (new_cap is not None and old_cap is not None
                and not _close(new_cap, old_cap, _CREATE_AMOUNT_TOLERANCE)):
            return False
        new_pa_n = (new_pa or "").strip().lower()
        old_pa_n = (old_pa or "").strip().lower()
        if new_pa_n and old_pa_n and new_pa_n != old_pa_n:
            return False
        return True
    if type_ == "s1_offering":
        new_size = new_t.get("anticipated_deal_size")
        old_size = old_t.get("anticipated_deal_size")
        # Primary: anticipated_deal_size when both sides carry it.
        if new_size is not None and old_size is not None:
            return _close(new_size, old_size, _CREATE_AMOUNT_TOLERANCE)
        # Fallback when deal_size missing: same placement agent within
        # the 60d window ⇒ same S-1. Issuers don't file two distinct
        # S-1s through the same underwriter that close together.
        np = new_pa.strip().lower()
        op = old_pa.strip().lower()
        return bool(np) and np == op
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
    m: "CreateMutation", accession: str, form: str, filing_date: str,
) -> None:
    """Record a re-disclosure of an existing instrument: bump
    last_seen_* and append a history entry. Existing non-null terms
    are NEVER overwritten (avoid LLM disagreements between filings
    stomping good data) — but missing-field backfill IS done: when
    the existing row is silent on a field that the redisclosure
    surfaces, we copy it in. Catches the common "8-K announces deal
    with conv_price; subsequent 10-Q balance sheet adds the aggregate
    liquidation_preference" pattern, which would otherwise lose the
    aggregate because the LLM emitted create rather than amend."""
    row = conn.execute(
        "SELECT type, created_accession, registration_accession, "
        "       terms_json, outstanding_json, history_json "
        "  FROM dilution_ledger WHERE instrument_id=?",
        (instrument_id,),
    ).fetchone()
    terms = json.loads((row["terms_json"] if row else None) or "{}")
    out = json.loads((row["outstanding_json"] if row else None) or "{}")
    backfilled: dict[str, Any] = {}
    for k, v in (m.terms or {}).items():
        if v is None or v == "" or v == [] or v == {}:
            continue
        if terms.get(k) in (None, "", [], {}):
            terms[k] = v
            backfilled[f"terms.{k}"] = v
    for k, v in (m.outstanding or {}).items():
        if v is None or v == "" or v == [] or v == {}:
            continue
        if out.get(k) in (None, "", [], {}):
            out[k] = v
            backfilled[f"outstanding.{k}"] = v
    history = json.loads((row["history_json"] if row else None) or "[]")
    history.append({
        "date": _ev(m, filing_date),
        "accession": accession,
        "form": form,
        "action": "redisclosed",
        "fields_changed": backfilled,
    })

    # Shelf-rollover (auto): when an ATM / equity_line / s1_offering is
    # re-disclosed inside a NEW shelf registration (different SEC
    # file_number than the row's current host), repoint the mutable
    # registration_accession at the new shelf — created_accession stays
    # immutable provenance. Without this, the row stays glued to the
    # original (often expired) shelf's file_number forever, and the
    # file_number-walk rollups (_parent_shelf, _last_banker_for_shelf,
    # the shelf-card drawdown rollup) silently attribute the agreement
    # and its takedowns to a dead registration. This is the SOLE
    # re-registration mechanism: the walker re-discloses the unchanged
    # ATM via create_atm (same date+capacity+agent → collapses here),
    # and this block repoints it to the new shelf. No LLM-facing tool.
    current_host = (
        (row["registration_accession"] or row["created_accession"])
        if row else accession
    )
    new_registration_accession = (
        row["registration_accession"] if row else None
    )
    if (row
            and (row["type"] or "") in _SHELF_HOSTED_TYPES
            and (form or "").upper() in _SHELF_HOST_FORMS
            and current_host != accession):
        fn_row = conn.execute(
            """SELECT
                 (SELECT file_number FROM dilution_filings
                   WHERE accession_number = ?) AS new_fn,
                 (SELECT file_number FROM dilution_filings
                   WHERE accession_number = ?) AS old_fn""",
            (accession, current_host),
        ).fetchone()
        new_fn = fn_row["new_fn"] if fn_row else None
        old_fn = fn_row["old_fn"] if fn_row else None
        if new_fn and old_fn and new_fn != old_fn:
            new_registration_accession = accession
            history.append({
                "date": filing_date,
                "accession": accession,
                "form": form,
                "action": "rolled_over_to_new_shelf",
                "fields_changed": {
                    "registration_accession": {
                        "from": current_host,
                        "to": accession,
                    },
                    "file_number": {"from": old_fn, "to": new_fn},
                },
            })
            log.info(
                "  shelf-rollover: %s migrated from %s (file %s) to "
                "%s (file %s)",
                instrument_id, current_host, old_fn,
                accession, new_fn,
            )

    conn.execute(
        """UPDATE dilution_ledger
             SET terms_json=?, outstanding_json=?, history_json=?,
                 registration_accession=?,
                 last_seen_accession=?, last_seen_date=?
           WHERE instrument_id=?""",
        (_to_json(terms), _to_json(out), _to_json(history),
         new_registration_accession,
         accession, filing_date, instrument_id),
    )


# Deliberately NO bare 'prospectus supplement' alternative: that phrase
# appears in generic offering boilerplate (SCNI round-6: the YA II ELOC
# announcements matched it and the ELOCs' draws rolled into the Aug-2023
# shelf's raised-to-date, 1.33M → 4.5M). Only an explicit base/shelf
# declaration marks a shelf-hosted program.
_BASE_PROSPECTUS_RE = re.compile(
    r"base\s+(?:shelf\s+)?prospectus"
    r"|shelf\s+registration\s+statement", re.IGNORECASE)


def _link_create_to_host_shelf(
    conn: sqlite3.Connection, cik: int, instrument_id: str,
    m: "CreateMutation", accession: str, form: str, filing_date: str,
) -> None:
    """Initial registration link for a shelf-hosted row created OFF its
    registration filing.

    An ATM announced via 8-K carries the 34-Act periodic file_number
    (001-/000-…), so the file_number joins that connect it to its host
    shelf (_parent_shelf, _last_banker_for_shelf, the shelf drawdown
    rollup) can never match (CETY: Oct-2023 shelf last_banker=None while
    its Roth ATM — created from an 8-K declaring sales 'pursuant to a
    prospectus supplement to the Company's base shelf prospectus' — sat
    unlinked). Mirror of the redisclosure-path shelf-rollover: when the
    creating filing is NOT a 33-Act registration, declares sale under a
    base/shelf prospectus, and exactly one live same-CIK shelf's 3-year
    window covers the agreement date, point registration_accession at
    that shelf's host filing. created_accession stays immutable
    provenance.
    """
    fn_row = conn.execute(
        "SELECT file_number FROM dilution_filings "
        " WHERE accession_number = ?", (accession,),
    ).fetchone()
    own_fn = (fn_row["file_number"] if fn_row else None) or ""
    if own_fn.startswith("333-"):
        return  # created on a registration filing — joins already work
    raw = conn.execute(
        "SELECT content_md FROM dilution_raw "
        " WHERE accession_number = ?", (accession,),
    ).fetchall()
    text = "\n".join((r["content_md"] or "") for r in raw)
    if not text or not _BASE_PROSPECTUS_RE.search(text):
        return
    agreement = (str((m.terms or {}).get("agreement_date") or "")[:10]
                 or filing_date)
    shelves = conn.execute(
        """SELECT instrument_id, created_at, created_accession,
                  registration_accession, terms_json
             FROM dilution_ledger
            WHERE cik = ? AND type = 'shelf' AND status = 'active'""",
        (cik,),
    ).fetchall()
    best_host, best_anchor = None, ""
    for s in shelves:
        try:
            s_terms = json.loads(s["terms_json"] or "{}") or {}
        except (TypeError, ValueError):
            s_terms = {}
        anchor_date = (str(s_terms.get("effect_date") or "")[:10]
                       or str(s["created_at"] or "")[:10])
        if not anchor_date or anchor_date > agreement:
            continue  # shelf postdates the agreement
        try:
            ad = _d.fromisoformat(anchor_date)
            if ad.replace(year=ad.year + 3).isoformat() < agreement:
                continue  # Rule 415(a)(5) window elapsed
        except (ValueError, TypeError):
            continue
        host = s["registration_accession"] or s["created_accession"]
        host_fn_row = conn.execute(
            "SELECT file_number FROM dilution_filings "
            " WHERE accession_number = ?", (host,),
        ).fetchone()
        host_fn = (host_fn_row["file_number"] if host_fn_row else None) or ""
        if not host_fn.startswith("333-"):
            continue
        if anchor_date > best_anchor:
            best_host, best_anchor = host, anchor_date
    if not best_host:
        return
    row = conn.execute(
        "SELECT history_json FROM dilution_ledger WHERE instrument_id=?",
        (instrument_id,),
    ).fetchone()
    history = json.loads((row["history_json"] if row else None) or "[]")
    history.append({
        "date": filing_date, "accession": accession, "form": form,
        "action": "linked_to_host_shelf",
        "fields_changed": {"registration_accession": {
            "from": None, "to": best_host}},
    })
    conn.execute(
        "UPDATE dilution_ledger SET registration_accession=?, "
        " history_json=? WHERE instrument_id=?",
        (best_host, _to_json(history), instrument_id),
    )
    log.info(
        "  host-shelf link: %s (created on %s, file %s) registered "
        "under shelf host %s",
        instrument_id, accession, own_fn or "?", best_host,
    )


_ORPHAN_ADOPT_WINDOW_DAYS = 120


def _adopt_orphans_on_shelf_create(
    conn: sqlite3.Connection, cik: int, shelf_accession: str,
    form: str, filing_date: str,
) -> None:
    """Reverse direction of _link_create_to_host_shelf: the announcing
    8-K often PREDATES the shelf registration it sells under (CETY: the
    Roth ATM 8-K was filed days before the Oct-2023 S-3), so the
    create-time link finds no shelf. When the shelf arrives, adopt
    recent unlinked shelf-hosted rows whose creating filing carried a
    34-Act file_number and declared sale under a base/shelf prospectus.
    """
    fn_row = conn.execute(
        "SELECT file_number FROM dilution_filings "
        " WHERE accession_number = ?", (shelf_accession,),
    ).fetchone()
    shelf_fn = (fn_row["file_number"] if fn_row else None) or ""
    if not shelf_fn.startswith("333-"):
        return
    try:
        fd = _d.fromisoformat(str(filing_date)[:10])
        floor = _d.fromordinal(
            fd.toordinal() - _ORPHAN_ADOPT_WINDOW_DAYS).isoformat()
    except (ValueError, TypeError):
        return
    orphans = conn.execute(
        """SELECT instrument_id, created_at, created_accession,
                  history_json
             FROM dilution_ledger
            WHERE cik = ? AND status = 'active'
              AND type IN ('atm', 'equity_line', 's1_offering')
              AND registration_accession IS NULL
              AND created_at >= ? AND created_at <= ?""",
        (cik, floor, str(filing_date)[:10]),
    ).fetchall()
    for o in orphans:
        own_fn_row = conn.execute(
            "SELECT file_number FROM dilution_filings "
            " WHERE accession_number = ?", (o["created_accession"],),
        ).fetchone()
        own_fn = (own_fn_row["file_number"] if own_fn_row else None) or ""
        if own_fn.startswith("333-"):
            continue  # already born on a registration filing
        raw = conn.execute(
            "SELECT content_md FROM dilution_raw "
            " WHERE accession_number = ?", (o["created_accession"],),
        ).fetchall()
        text = "\n".join((r["content_md"] or "") for r in raw)
        if not text or not _BASE_PROSPECTUS_RE.search(text):
            continue
        history = json.loads(o["history_json"] or "[]")
        history.append({
            "date": filing_date, "accession": shelf_accession,
            "form": form, "action": "linked_to_host_shelf",
            "fields_changed": {"registration_accession": {
                "from": None, "to": shelf_accession}},
        })
        conn.execute(
            "UPDATE dilution_ledger SET registration_accession=?, "
            " history_json=? WHERE instrument_id=?",
            (shelf_accession, _to_json(history), o["instrument_id"]),
        )
        log.info(
            "  host-shelf link (adopt): %s (created on %s, file %s) "
            "registered under new shelf %s",
            o["instrument_id"], o["created_accession"],
            own_fn or "?", shelf_accession,
        )


# ─── Mutation appliers ───────────────────────────────────────────────
def _apply_create(
    conn: sqlite3.Connection, cik: int, ticker: str,
    m: "CreateMutation", accession: str, form: str,
    filing_date: str, seq_state: dict[str, int],
    skip_dedup: bool = False,
) -> tuple[str, bool, bool]:
    """Insert a new instrument, OR collapse onto an existing active
    row when this create looks like a re-disclosure of one already on
    the ledger.

    Returns (resolved_id, was_redisclosure, drew). resolved_id is the
    actual ledger row this create resolved to — the new id on insert,
    the existing id on collapse, or a freshly-allocated id when the
    LLM's proposed_id collided with an existing row. apply_mutations
    uses this to remap any downstream amend/record_event/close
    mutations in the same filing that referenced m.proposed_id, since
    the validator only sees proposed_ids and can't predict the
    collapse/realloc. drew is True when a same-filing equity closing
    (create_equity with closing_date) booked a drawdown row.
    """
    existing_id = (None if skip_dedup else _create_already_recorded(
        conn, cik, m, filing_date, accession=accession,
    ))
    if existing_id is not None:
        log.info(
            "  redisclosure: %s create (type=%s) collapsed onto %s",
            accession, m.type, existing_id,
        )
        _append_redisclosure(
            conn, existing_id, m, accession, form, filing_date,
        )
        return existing_id, True, False
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
    anchor = _create_anchor(m, filing_date)
    out0 = dict(m.outstanding)
    # As-of stamp for asof-gated balances: the create-time balance is
    # as of the issuance. Any restatement whose as-of predates this
    # (a 10-K/A for a period BEFORE the note existed) is then
    # auto-vetoed by _stale_balance_veto.
    for _bk in _ASOF_GATED_BALANCE_FIELDS:
        if isinstance(out0.get(_bk), (int, float)) \
                and not isinstance(out0.get(_bk), bool):
            out0[f"{_bk}_asof"] = anchor[:10]
    history = [{
        "date": anchor,
        "accession": accession,
        "form": form,
        "action": "created",
        "fields_changed": {"terms": dict(m.terms),
                           "outstanding": dict(m.outstanding)},
    }]
    # Deterministic label overrides whatever the LLM emitted. Falls
    # back to the LLM's m.label when build_label can't compose one
    # (event_date missing/unparseable). The card layer has its own
    # mechanical template if both end up null.
    label = build_label(m) or m.label
    # Walker schema only carries canonical entity names; the DB schema
    # keeps the verbatim columns for backwards compatibility with old
    # rows. Populate the verbatim column from the canonical so cards.py's
    # fallback chain (canonical → verbatim) still works either way.
    conn.execute(
        """INSERT INTO dilution_ledger
             (instrument_id, ticker, cik, type, created_at,
              created_accession, counterparty_canonical,
              placement_agent_canonical, label,
              terms_json, outstanding_json, status, status_at,
              history_json, last_seen_accession, last_seen_date)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)""",
        (
            instrument_id, ticker, cik, m.type,
            anchor, accession,
            m.counterparty_canonical,
            m.placement_agent_canonical, label,
            _to_json(dict(m.terms)),
            _to_json(out0),
            anchor,
            _to_json(history),
            accession, filing_date,
        ),
    )
    # Initial registration link: an ATM/ELOC/S-1 created off its
    # registration filing (8-K announcement) needs its
    # registration_accession pointed at the host shelf or every
    # file_number-walk rollup misses it (CETY Roth/last_banker).
    if m.type in _SHELF_HOSTED_TYPES:
        try:
            _link_create_to_host_shelf(
                conn, cik, instrument_id, m, accession, form, filing_date,
            )
        except Exception:
            log.warning(
                "  host-shelf link failed for %s (%s)",
                instrument_id, accession, exc_info=True,
            )
    elif m.type == "shelf":
        try:
            _adopt_orphans_on_shelf_create(
                conn, cik, accession, form, filing_date,
            )
        except Exception:
            log.warning(
                "  host-shelf orphan adoption failed for %s (%s)",
                instrument_id, accession, exc_info=True,
            )
    # Same-filing "signed AND closed" PIPE: the create itself carries
    # the closing signal, so book the cash now. Pending placements
    # (no closing_date) book nothing until a confirm_closing arrives.
    drew = False
    if m.type == "equity" and m.closing_date is not None:
        drew = _maybe_write_equity_drawdown(
            conn, cik, instrument_id, m.counterparty_canonical,
            dict(m.terms), dict(m.outstanding), None,
            m.closing_date.isoformat(), accession,
        )
    return instrument_id, False, drew


_LABEL_MONTH_YEAR_RE = re.compile(
    r"^(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}\s+(.+)$"
)


def _relabel_with_issue_date(
    existing_label: str | None, issue_date_iso: str,
) -> str | None:
    """Swap the leading '<Month> <Year>' in an existing label for the
    one derived from issue_date_iso. Returns None when the existing
    label doesn't start with a parseable month-year (caller keeps the
    old label).

    Regex-replace preserves whatever qualifier slot was originally in
    the label (descriptor / series / pre-funded / placement-agent /
    counterparty) — those aren't all stored as separate columns, so
    rebuilding via build_label could drop the descriptor. The leading
    'Month Year' is the only piece we want to swap.
    """
    if not existing_label:
        return None
    rest_match = _LABEL_MONTH_YEAR_RE.match(existing_label.strip())
    if not rest_match:
        return None
    try:
        d = _d.fromisoformat(issue_date_iso[:10])
    except (TypeError, ValueError):
        return None
    return f"{d.strftime('%B %Y')} {rest_match.group(1)}"


def _rescale_stale_unit_amend(
    row_type: str, terms: dict, out: dict,
    field_updates: dict | None, outstanding_updates: dict | None,
    ev_iso: str,
) -> dict[str, dict[str, Any]]:
    """Normalize amend values stated in pre-split units.

    A periodic filing whose report period predates a reverse split
    restates instrument terms in the OLD units: CETY's Q3-2025 10-Q
    (period 2025-09-30) quotes $2.00/$2.50 strikes for warrants the
    2025-10-06 1-for-15 split had already adjusted to $30.00/$37.50,
    and the anchor's correction amends (dated as-of the period end)
    faithfully ping-ponged the split adjustment away. Same family as
    the conv_price 1000×-drift.

    For each numeric update on a split-scaled field of an instrument
    whose ``applied_splits`` contain entries dated AFTER the amend's
    event date, choose between the raw value and its post-split
    transform (counts × ratio, prices ÷ ratio — exactly what
    _apply_split does) by log-distance to the current ledger value:

      - a stale-unit restate lands ~1/ratio away from current and its
        transform lands on top of it → transform wins (often a no-op);
      - a value already in post-split units (mixed-unit filings restate
        counts adjusted but prices raw) stays closer raw → raw wins;
      - no current value to compare → trust the event date (a pre-split
        period discloses pre-split units) and transform.

    Mutates the update dicts in place. Returns {field: {raw, scaled}}
    for the history entry — empty when nothing was rescaled.
    """
    applied = terms.get("applied_splits") or []
    if not ev_iso:
        return {}
    cum = 1.0
    for s in applied:
        s_date = str(s.get("date") or "")[:10]
        ratio = s.get("ratio")
        if s_date and ratio and ev_iso[:10] < s_date:
            cum *= float(ratio)
    # Products of every contiguous run of already-applied splits, for
    # the stale-unit ECHO check below. A post-split filing re-quoting
    # the instrument's original (or partially-adjusted) terms emits a
    # value off from current by EXACTLY one of these factors — e.g.
    # GCTK W-4382 (round-4): a June-2025 recap re-quoted the July-2024
    # warrant with only the 60:1 adjusted, ×20 off current, and the
    # amend reverted both splits' work (5940→297, 250→5000). The event
    # date post-dated every split, so the date-vintage transform above
    # could not catch it.
    _echo_products: set[float] = set()
    _ratios = []
    for s in applied:
        try:
            r = float(s.get("ratio") or 0)
        except (TypeError, ValueError):
            r = 0.0
        if r > 0:
            _ratios.append(r)
    for i in range(len(_ratios)):
        p = 1.0
        for j in range(i, len(_ratios)):
            p *= _ratios[j]
            if p != 1.0:
                _echo_products.add(p)
    if cum == 1.0 and not _echo_products:
        return {}

    # Shared preferred $-terms split policy (see _preferred_price_split_skip):
    # stated_value always fixed; conv_price rescaled only for a price-based
    # (no conversion_ratio) series. Keeping this byte-identical to _apply_split
    # is what lets a post-split filing that re-quotes the raw pre-split
    # conv_price get echo-pinned back to the split-adjusted value below instead
    # of clobbering it (BNKK Series C: 0.5582 re-quoted vs current 19.54).
    skip_price = (
        _preferred_price_split_skip(terms)
        if row_type == "preferred" else frozenset()
    )

    def _pick(raw, scaled, current):
        """Return `raw` or `scaled` (original objects, types preserved)."""
        if raw == 0:
            return raw  # scale-invariant
        try:
            cur = float(current) if current is not None else None
        except (TypeError, ValueError):
            cur = None
        if not cur or cur <= 0 or raw < 0 or scaled <= 0:
            return scaled  # no comparable current value → date rule
        d_raw = abs(math.log(float(raw) / cur))
        d_scaled = abs(math.log(float(scaled) / cur))
        return scaled if d_scaled < d_raw else raw

    rescaled: dict[str, dict[str, Any]] = {}
    for updates, current_src in (
        (field_updates, terms),
        (outstanding_updates, out),
    ):
        if not updates:
            continue
        for k, v in list(updates.items()):
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            if k not in _COUNT_FIELDS and (
                    k not in _PRICE_FIELDS or k in skip_price):
                continue
            # Stale-unit ECHO: proposed differs from CURRENT by exactly
            # a contiguous-run product of applied splits → the filing
            # re-quoted old units; pin to current (no-op) instead of
            # letting the amend undo the split work. Runs regardless of
            # event-date vintage.
            cur_raw = current_src.get(k)
            try:
                curf = float(cur_raw) if cur_raw is not None else None
            except (TypeError, ValueError):
                curf = None
            if curf and curf > 0 and v > 0 and _echo_products:
                implied = (curf / v) if k in _COUNT_FIELDS else (v / curf)
                if any(math.isclose(implied, p, rel_tol=0.01)
                       for p in _echo_products):
                    rescaled[k] = {"raw": v, "scaled": cur_raw,
                                   "echo": True}
                    updates[k] = cur_raw
                    continue
            if cum == 1.0:
                continue
            if k in _COUNT_FIELDS:
                scaled = round(v * cum)
            else:
                scaled = round(v / cum, _PRICE_DECIMALS)
            chosen = _pick(v, scaled, cur_raw)
            if chosen != v:
                rescaled[k] = {"raw": v, "scaled": chosen}
                updates[k] = chosen
    return rescaled


# Point-in-time balance fields where "latest as-of wins" replaces
# last-write-wins. A periodic filing states these AS OF its report
# period end (the amend's event_date), so a LATER-FILED amendment
# carrying an OLDER as-of (a 10-K/A or 10-Q/A re-stating a past period)
# must not overwrite a newer balance — CETY round-6b: a FY2024 10-K/A
# (as-of 2024-12-31) zeroed a note ISSUED June-2025, then older-period
# 10-Q/As re-raised it; whichever applied last in walk order won. Same
# convention as drawn_usd_anchor/asof for ATMs.
_ASOF_GATED_BALANCE_FIELDS = ("principal_remaining",)


def _stale_balance_veto(
    out: dict, outstanding_updates: dict | None, ev_iso: str,
) -> dict[str, Any]:
    """Veto balance updates whose as-of predates the current balance's
    as-of; stamp `<field>_asof` on every accepted one. The create-time
    stamp (issue date) makes pre-issuance restatements auto-vetoed.
    Mutates outstanding_updates in place; returns {field: {...}} for
    the history entry — empty when nothing was vetoed."""
    if not outstanding_updates or not ev_iso:
        return {}
    vetoed: dict[str, Any] = {}
    for k in _ASOF_GATED_BALANCE_FIELDS:
        v = outstanding_updates.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        cur_asof = str(out.get(f"{k}_asof") or "")[:10]
        if cur_asof and ev_iso[:10] < cur_asof:
            vetoed[k] = {"raw": v, "kept": out.get(k),
                         "stale_asof": ev_iso[:10],
                         "current_asof": cur_asof}
            outstanding_updates.pop(k, None)
        else:
            # Equal as-of allowed: a later-filed amendment re-stating
            # the SAME period is a correction and wins (walk order =
            # filing order).
            outstanding_updates[f"{k}_asof"] = ev_iso[:10]
    return vetoed


def _merged_aggregate_veto(
    conn: sqlite3.Connection, cik: int, row: sqlite3.Row,
    out: dict, outstanding_updates: dict | None,
) -> dict[str, Any]:
    """Veto a count restate that equals the SUM of sibling tranches.

    A later periodic filing often re-quotes a multi-tranche issuance as
    ONE aggregate ("warrants to purchase 5,213,104 shares") and the LLM
    amends EACH tranche to that total, corrupting every sibling (SCNI
    Dec-2023 inducement: a 20-F restated both the 3-year tranche
    [292,000] and the 5.5-year tranche [229,310] to the merged 521,310).
    When the proposed count matches the split-adjusted initial_count sum
    of all same-created_accession same-type siblings (and is materially
    above this row's own initial), pin the update to the current value.
    Mutates outstanding_updates in place; returns {field: {raw, pinned}}
    for the history entry — empty when nothing was vetoed.
    """
    if not outstanding_updates:
        return {}
    proposed = outstanding_updates.get("count")
    if not isinstance(proposed, (int, float)) or isinstance(proposed, bool):
        return {}
    if not row["created_accession"]:
        return {}
    sibs = conn.execute(
        """SELECT outstanding_json FROM dilution_ledger
            WHERE cik = ? AND created_accession = ? AND type = ?""",
        (cik, row["created_accession"], row["type"]),
    ).fetchall()
    if len(sibs) < 2:
        return {}
    total = 0.0
    for s in sibs:
        try:
            s_out = json.loads(s["outstanding_json"] or "{}") or {}
        except (TypeError, ValueError):
            s_out = {}
        part = s_out.get("initial_count")
        if part is None:
            part = s_out.get("count")
        try:
            total += float(part or 0)
        except (TypeError, ValueError):
            pass
    own_initial = 0.0
    try:
        own_initial = float(out.get("initial_count")
                            or out.get("count") or 0)
    except (TypeError, ValueError):
        pass
    if (total > 0
            and math.isclose(float(proposed), total, rel_tol=0.005)
            and float(proposed) > own_initial * 1.05):
        vetoed: dict[str, Any] = {}
        for k in ("count", "initial_count"):
            v = outstanding_updates.get(k)
            if (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isclose(float(v), total, rel_tol=0.005)):
                cur = out.get(k)
                vetoed[k] = {"raw": v, "pinned": cur,
                             "sibling_total": total}
                if cur is None:
                    outstanding_updates.pop(k, None)
                else:
                    outstanding_updates[k] = cur
        return vetoed
    return {}


def _apply_amend(
    conn: sqlite3.Connection, cik: int, m: "AmendMutation",
    accession: str, form: str, filing_date: str,
) -> None:
    row = _fetch(conn, cik, m.instrument_id)
    terms = json.loads(row["terms_json"] or "{}")
    out = json.loads(row["outstanding_json"] or "{}")
    # field_updates / outstanding_updates are @property dicts rebuilt on
    # every access — capture ONCE so the stale-unit rescale below isn't
    # silently discarded by the next property call.
    field_updates = dict(m.field_updates or {})
    outstanding_updates = dict(m.outstanding_updates or {})
    # Normalize stale-unit numerics BEFORE the apply loops so every
    # downstream consumer (changed-diff, initial_count mirror, history)
    # sees post-split units.
    unit_rescaled = _rescale_stale_unit_amend(
        row["type"], terms, out,
        field_updates, outstanding_updates,
        _ev(m, filing_date),
    )
    # Merged-aggregate restate veto: a count equal to the sibling-tranche
    # SUM is the issuance total re-quoted, not this tranche's count.
    aggregate_vetoed = _merged_aggregate_veto(
        conn, cik, row, out, outstanding_updates,
    )
    # Latest-as-of-wins gate for point-in-time balances (10-K/A / 10-Q/A
    # restatements of PAST periods must not clobber newer balances).
    balance_vetoed = _stale_balance_veto(
        out, outstanding_updates, _ev(m, filing_date),
    )
    changed: dict[str, Any] = {}
    for k, v in field_updates.items():
        prev = terms.get(k)
        if prev != v:
            changed[f"terms.{k}"] = {"from": prev, "to": v}
        if v is None:
            terms.pop(k, None)
        else:
            terms[k] = v
    for k, v in outstanding_updates.items():
        prev = out.get(k)
        if prev != v:
            changed[f"outstanding.{k}"] = {"from": prev, "to": v}
        if v is None:
            out.pop(k, None)
        else:
            out[k] = v
    # Preferred conversion-ratio → conv_price derivation. A LATER filing
    # that first states the fixed common/ADS-per-preferred ratio (SCNI
    # EIB: "each Preferred Share is convertible into 364 ADSs") carries
    # no dollar conv_price. Mirror CreatePreferred.terms: derive
    # conv_price = stated_value / conversion_ratio from the EXISTING row's
    # stated_value (the amend doesn't restate it). Fire ONLY when no
    # explicit conv_price was passed (an explicit price always wins), the
    # row knows a stated_value, AND the row has no conv_price yet (never
    # clobber an existing dollar price). getattr keeps this preferred-
    # only — other amend mutations have no conversion_ratio attribute.
    _ratio = getattr(m, "conversion_ratio", None)
    if (_ratio and float(_ratio) > 0
            and field_updates.get("conv_price") is None
            and terms.get("stated_value")
            and terms.get("conv_price") is None):
        _derived_cp = float(terms["stated_value"]) / float(_ratio)
        changed["terms.conv_price"] = {
            "from": terms.get("conv_price"), "to": _derived_cp,
        }
        terms["conv_price"] = _derived_cp
    # A drawn_usd amend is a CUMULATIVE-to-date checkpoint from a
    # periodic filing (10-K/10-Q), as of its report date. Record it on
    # a DEDICATED pair of keys — distinct from the running
    # `outstanding.drawn_usd`, which _apply_record_event increments per
    # discrete take-down and is therefore unreliable as a cumulative
    # (last-writer-wins between increments and anchor sets). The card
    # layer treats drawn_usd_anchor as authoritative and adds only the
    # discrete draws dated AFTER drawn_usd_asof (post-period 8-Ks),
    # rather than re-summing draws the checkpoint already subsumes. This
    # is the fix for ATM/ELOC double-counts where a quarterly aggregate
    # re-reports an interim sale (GCTK ATM-2183 8.78M→7.96M, SCNI EL-045
    # 7.18M→5.80M).
    drawn_amend = outstanding_updates.get("drawn_usd")
    if drawn_amend is not None:
        out["drawn_usd_anchor"] = drawn_amend
        out["drawn_usd_asof"] = _ev(m, filing_date)
    # Relabel when this amend supplied a new issue_date. Used by the
    # FPI-signing-then-closing 6-K pair: the create from the signing
    # filing stamps the row 'March 2024 Warrants'; the closing filing
    # passes issue_date=2024-08-14 here and we swap the month-year to
    # 'August 2024 Warrants' so the card matches DilutionTracker's
    # closing-date convention.
    new_issue_date = field_updates.get("issue_date")
    new_label_value: str | None = None
    if new_issue_date:
        new_label_value = _relabel_with_issue_date(
            row["label"], str(new_issue_date),
        )
        if new_label_value and new_label_value != row["label"]:
            changed["label"] = {"from": row["label"], "to": new_label_value}
    # An amend that clarifies `count` UPWARD BEFORE any exercises or
    # terminations have been recorded is restating the original tranche
    # size (8-K said ~550,000, 424B prices 626,667). Mirror onto
    # initial_count so total_issued tracks the precise number. Only
    # upward moves count as a clarification: a downward amend is either
    # an exercise the walker is booking implicitly or a close-path
    # zeroing (see _zero_outstanding_for_close) — neither should erase
    # the create-time issued count.
    new_count = outstanding_updates.get("count")
    if (new_count is not None
            and not float(out.get("exercised_to_date") or 0)
            and not float(out.get("terminated_to_date") or 0)):
        initial = float(out.get("initial_count") or 0)
        if float(new_count) > initial:
            out["initial_count"] = new_count
    history = json.loads(row["history_json"] or "[]")
    ev_iso = _ev(m, filing_date)
    if unit_rescaled:
        # Audit trail: which incoming values were judged stale-unit and
        # what they became (changed[] above already reflects the
        # post-rescale value).
        changed["unit_rescale"] = unit_rescaled
    if aggregate_vetoed:
        changed["merged_aggregate_veto"] = aggregate_vetoed
    if balance_vetoed:
        changed["stale_balance_veto"] = balance_vetoed
        log.info(
            "  amend %s: stale-as-of balance restate vetoed (%s)",
            m.instrument_id,
            ", ".join(f"{k} {v['raw']} as-of {v['stale_asof']} < "
                      f"{v['current_asof']}"
                      for k, v in balance_vetoed.items()),
        )
    history.append({
        "date": ev_iso,
        "accession": accession, "form": form, "action": "amended",
        "fields_changed": changed,
    })
    # Reactivate-on-amend. A terms-changing amend dated AFTER a close means
    # the instrument is still live — an issuer does not amend the terms of
    # dead paper (CETY's October-2023 S-3 was wrongly 'terminated' by a
    # periodic 10-Q the LLM misread, then amended by an S-3/A two months
    # later; the S-3/A proves it never died). This retroactively undoes such
    # a wrongful close the moment a later filing restates the row's terms.
    # Scoped tightly to issuer-action closes: 'expired' is a date fact and
    # 'superseded'/'converted'/'exercised' are genuine end-states, so none of
    # those are ever reopened here; and only TERMS changes count (an
    # outstanding/drawdown re-disclosure on a dead row must not resurrect it).
    new_status, new_status_at = row["status"], row["status_at"]
    if (row["status"] in ("terminated", "redeemed")
            and any(k.startswith("terms.") for k in changed)
            and row["status_at"] and ev_iso
            and ev_iso[:10] > str(row["status_at"])[:10]):
        new_status, new_status_at = "active", ev_iso
        history.append({
            "date": ev_iso, "accession": accession, "form": form,
            "action": "reopened",
            "fields_changed": {"from": row["status"], "to": "active",
                               "why": "terms amended after close"},
        })
    if new_label_value and new_label_value != row["label"]:
        conn.execute(
            """UPDATE dilution_ledger
                 SET terms_json=?, outstanding_json=?, history_json=?,
                     last_seen_accession=?, last_seen_date=?, label=?,
                     status=?, status_at=?
               WHERE instrument_id=?""",
            (_to_json(terms), _to_json(out), _to_json(history),
             accession, filing_date, new_label_value,
             new_status, new_status_at, m.instrument_id),
        )
    else:
        conn.execute(
            """UPDATE dilution_ledger
                 SET terms_json=?, outstanding_json=?, history_json=?,
                     last_seen_accession=?, last_seen_date=?,
                     status=?, status_at=?
               WHERE instrument_id=?""",
            (_to_json(terms), _to_json(out), _to_json(history),
             accession, filing_date, new_status, new_status_at, m.instrument_id),
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


_PREFERRED_FULL_REDEMPTION_FRACTION = 0.95


def _preferred_shares_from_principal(
    amount: float, terms: dict, outstanding: dict,
) -> tuple[float, bool]:
    """Translate a debt-shaped $ redemption/conversion booked against a
    PREFERRED into a preferred-share count via the series stated_value.

    The walker sometimes books a loan-to-equity preferred's retirement
    with a $ amount (principal_redeemed / principal_converted) instead of
    a share count — validate lets that through when stated_value is known.
    Returns (shares_to_drop, is_full_retirement). A $ amount within
    _PREFERRED_FULL_REDEMPTION_FRACTION of the live count clamps to a full
    retirement (drop the whole count) so the preliminary tranche of a
    debt-to-equity rollover is fully extinguished (SCNI EIB preferred:
    $28,727,000 / $29,000 = 990.6 ≥ 0.95 × 1000 → count 0)."""
    try:
        sv = float(terms.get("stated_value"))
    except (TypeError, ValueError):
        return 0.0, False
    if sv <= 0:
        return 0.0, False
    cur = float(outstanding.get("count") or 0)
    if cur <= 0:
        return 0.0, False
    derived = amount / sv
    if derived >= cur * _PREFERRED_FULL_REDEMPTION_FRACTION:
        return cur, True
    return min(derived, cur), False


def _maybe_write_equity_drawdown(
    conn: sqlite3.Connection, cik: int, instrument_id: str,
    counterparty: str | None, terms: dict, outstanding: dict,
    gross_proceeds, event_iso: str, accession: str,
) -> bool:
    """Book the closing cash of an off-shelf equity placement (PIPE)
    into dilution_ledger_drawdowns — the raise-history index summed by
    capital_raised_since (cash bridge) and the dilution-history badge.

    Cash is booked once, at CLOSING: callers gate on an explicit
    closing signal (create_equity.closing_date / confirm_closing on an
    equity row), so a signed-but-pending SPA never inflates the cash
    estimate — undercounting is the safe direction here.

    Amount basis: stated gross proceeds when the filing disclosed
    them, else count × price_per_share (identical for a PIPE: gross =
    shares × purchase price). amount <= 0 (stock-for-services price=0,
    cashless closing) → no row.

    Idempotency: an equity placement closes exactly once, so ANY
    existing drawdown row on the instrument suppresses the write. This
    is deliberately stronger than _drawdown_already_recorded's
    5%/180-day tolerances (tuned for REPEATED ATM/shelf takedowns) and
    subsumes them — it closes the create-then-confirm double-count gap
    under amount drift or >180-day re-disclosures. Multi-tranche deals
    are modeled as one equity instrument per tranche, so one row per
    instrument is the correct cardinality.
    """
    count = float(outstanding.get("count") or 0)
    price = float(terms.get("price_per_share") or 0)
    amount = float(gross_proceeds or 0) or (count * price)
    if amount <= 0:
        return False
    if conn.execute(
        "SELECT 1 FROM dilution_ledger_drawdowns "
        "WHERE instrument_id = ? LIMIT 1",
        (instrument_id,),
    ).fetchone():
        return False
    conn.execute(
        """INSERT INTO dilution_ledger_drawdowns
             (cik, instrument_id, accession_number, event_date,
              amount_usd, shares, price,
              drawdown_party_canonical, drawdown_party_role,
              detected_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cik, instrument_id, accession, event_iso,
         amount, count or None, price or None,
         counterparty, "investor" if counterparty else None, now_iso()),
    )
    return True


def _apply_record_event(
    conn: sqlite3.Connection, cik: int, m: "RecordMutation",
    accession: str, form: str, filing_date: str,
) -> bool:
    """Return True if an aux drawdown row was recorded."""
    row = _fetch(conn, cik, m.instrument_id)
    terms = json.loads(row["terms_json"] or "{}")
    outstanding = json.loads(row["outstanding_json"] or "{}")
    fields = dict(m.fields or {})
    drew = False
    terms_changed = False
    pref_closed = False  # set when a preferred is fully retired below

    if m.event_kind == "exercise":
        shares = float(fields.get("shares") or 0)
        # Cashless/net-share exercises retire MORE warrants than the
        # common shares delivered (ACTU: 76,376 warrants → 26,070 net
        # shares); the count drops by the surrendered figure while
        # exercised_to_date tracks shares actually issued. Without
        # warrants_exercised the two are 1:1 (ordinary cash exercise).
        retired = float(fields.get("warrants_exercised") or 0) or shares
        # Down-round/cashless signature: shares delivered FAR above the
        # warrant's current count means the two are in different units
        # (a ratcheted or cashless formula delivered multiples of the
        # face), and a 1:1 decrement would blow through the face and
        # zero a live warrant (CETY Jan-2025 Mast Hill: 1,264,420 +
        # 195,867 shares against a 54,594-count warrant → zeroed +
        # auto-terminated). Skip the COUNT decrement — the anchor's
        # warrant-table reconciliation owns the count — but still track
        # shares in exercised_to_date. An explicit warrants_exercised
        # always wins (the issuer stated the retired figure).
        cur_count = float(outstanding.get("count") or 0)
        if (not float(fields.get("warrants_exercised") or 0)
                and cur_count > 0 and retired > cur_count * 1.5):
            fields["count_decrement_skipped"] = {
                "reason": "cashless_ratio",
                "shares": retired, "count": cur_count,
            }
            retired = 0.0
        if retired:
            outstanding["count"] = max(
                0.0, float(outstanding.get("count") or 0) - retired,
            )
        if shares:
            outstanding["exercised_to_date"] = (
                float(outstanding.get("exercised_to_date") or 0) + shares
            )
        action = "exercised"
    elif m.event_kind == "conversion":
        # Convertible notes: principal_converted decrements principal_remaining.
        principal = float(fields.get("principal_converted") or 0)
        _conv_iso = _ev(m, filing_date)[:10]
        _bal_asof = str(
            outstanding.get("principal_remaining_asof") or "")[:10]
        if principal:
            # Subsumption: a stated balance as-of X already nets every
            # conversion on or before X (same strictly-after convention
            # as _drawn_to_date for ATM draws). Decrementing again
            # double-subtracts — a 10-Q that itemizes the quarter's
            # conversions AND states the period-end balance must land
            # on the stated balance regardless of in-batch order.
            # converted_to_date still accumulates: the event is real,
            # only the balance already reflects it.
            if _bal_asof and _conv_iso <= _bal_asof:
                fields["balance_decrement_subsumed"] = {
                    "asof": _bal_asof, "event": _conv_iso}
            else:
                outstanding["principal_remaining"] = max(
                    0.0, float(outstanding.get("principal_remaining") or 0)
                    - principal,
                )
                outstanding["principal_remaining_asof"] = _conv_iso
            # Cumulative converted can never exceed the note's face — the
            # issuer cannot convert more principal than it issued. Clamp
            # and record the raw figure (same marker style as
            # `count_decrement_skipped` above). The overflow signature is
            # an aggregate multi-note conversion figure attributed to EACH
            # note separately (NUAI C-123/C-124: $6,119,409 and $6,118,243
            # booked against a combined $10M of face, summing to $12.2M).
            # Unlike `principal_remaining` this field has no anchor to
            # reconcile it — nothing downstream ever corrects it — so the
            # clamp is the only guard, and `close_retired_debt` reads it
            # as a full-retirement signal.
            _conv_td = (
                float(outstanding.get("principal_converted_to_date") or 0)
                + principal
            )
            _face = float(terms.get("principal") or 0)
            if _face > 0 and _conv_td > _face:
                fields["converted_to_date_clamped"] = {
                    "raw": _conv_td, "kept": _face,
                }
                _conv_td = _face
            outstanding["principal_converted_to_date"] = _conv_td
        if "principal_remaining" in fields:
            # An explicitly stated post-conversion balance is an as-of
            # statement at the event date — same latest-as-of-wins gate
            # as _apply_amend.
            if not (_bal_asof and _conv_iso < _bal_asof):
                outstanding["principal_remaining"] = float(
                    fields["principal_remaining"])
                outstanding["principal_remaining_asof"] = _conv_iso
            else:
                fields["stated_balance_stale_asof"] = {
                    "asof": _bal_asof, "event": _conv_iso}
        # Preferred series: preferred_shares_converted decrements `count`.
        # validate.py gates that this field is set when target is preferred,
        # so we just trust the value here. `count_converted_to_date` mirrors
        # principal_converted_to_date / exercised_to_date so downstream
        # readers (cards, anchor) can recover the running total.
        pref_shares = float(fields.get("preferred_shares_converted") or 0)
        if (not pref_shares and principal
                and (row["type"] or "").lower() == "preferred"):
            # Debt-shaped conversion ($ amount) on a preferred → translate
            # to a share count via stated_value (validate allows this when
            # stated_value is known). Full conversion extinguishes the row.
            pref_shares, pref_closed = _preferred_shares_from_principal(
                principal, terms, outstanding,
            )
        if pref_shares and (row["type"] or "").lower() == "preferred":
            outstanding["count"] = max(
                0.0, float(outstanding.get("count") or 0) - pref_shares,
            )
            outstanding["count_converted_to_date"] = (
                float(outstanding.get("count_converted_to_date") or 0)
                + pref_shares
            )
        action = "converted"
    elif m.event_kind == "partial_redemption":
        # Convertible notes: principal_redeemed decrements principal_remaining.
        amount = float(fields.get("principal_redeemed") or 0)
        if amount:
            outstanding["principal_remaining"] = max(
                0.0, float(outstanding.get("principal_remaining") or 0)
                - amount,
            )
            # Mirror principal_converted_to_date for the CASH-repayment
            # leg. Without this a note repaid in cash reaches
            # principal_remaining=0 with no flow record anywhere, so
            # `close_retired_debt` cannot tell "fully repaid" from
            # "balance is wrong" and has to leave it active. Clamped at
            # face for the same reason as the conversion accumulator.
            _red_td = (
                float(outstanding.get("principal_redeemed_to_date") or 0)
                + amount
            )
            _red_face = float(terms.get("principal") or 0)
            if _red_face > 0 and _red_td > _red_face:
                fields["redeemed_to_date_clamped"] = {
                    "raw": _red_td, "kept": _red_face,
                }
                _red_td = _red_face
            outstanding["principal_redeemed_to_date"] = _red_td
        # Preferred series: preferred_shares_redeemed decrements `count`.
        # validate.py gates that this field is set when target is preferred,
        # so we just trust the value here. `count_redeemed_to_date` mirrors
        # count_converted_to_date so cards/anchor readers can recover the
        # running total of preferred shares retired for cash.
        pref_shares = float(fields.get("preferred_shares_redeemed") or 0)
        if (not pref_shares and amount
                and (row["type"] or "").lower() == "preferred"):
            # Debt-shaped redemption ($ amount) on a preferred → translate
            # to a share count via stated_value (validate allows this when
            # stated_value is known); previously this was hard-rejected and
            # the retirement silently lost, leaving the preliminary tranche
            # of a loan-to-equity rollover live forever (SCNI EIB P-177).
            pref_shares, pref_closed = _preferred_shares_from_principal(
                amount, terms, outstanding,
            )
        if pref_shares and (row["type"] or "").lower() == "preferred":
            outstanding["count"] = max(
                0.0, float(outstanding.get("count") or 0) - pref_shares,
            )
            outstanding["count_redeemed_to_date"] = (
                float(outstanding.get("count_redeemed_to_date") or 0)
                + pref_shares
            )
        action = "partial_redemption"
    elif m.event_kind == "partial_termination":
        amount = float(fields.get("capacity_reduced_usd") or 0)
        if amount:
            cap = float(outstanding.get("remaining_capacity_usd") or 0)
            if cap:
                outstanding["remaining_capacity_usd"] = max(
                    0.0, cap - amount,
                )
        action = "partial_termination"
    elif m.event_kind == "drawdown":
        # GROSS dollars are the single capacity basis. RecordDrawdown.fields
        # always emits `drawdown_amount_usd` as gross — shares × gross
        # price_per_share when a per-share price was given, else the gross
        # aggregate fallback (the prompt + the drawdown_missing_price retry
        # steer the model to the per-share gross price precisely so a
        # net-of-fees aggregate doesn't slip in). Shelf/ATM capacity is
        # registered gross, so drawn_usd / remaining_capacity_usd below stay
        # on the same basis and the validator's overflow check compares
        # like-for-like. The former `or amount_usd or gross_proceeds`
        # fallbacks were dead — no drawdown mutation emits those keys
        # (gross_proceeds belongs to record_exercise, amount_usd to no
        # mutation at all) — and reading three interchangeable fields made
        # the gross/net basis look non-deterministic when it is not.
        amount = float(fields.get("drawdown_amount_usd") or 0)
        shares = float(fields.get("drawdown_shares")
                       or fields.get("shares") or 0)
        ev_iso = m.event_date.isoformat()
        if _drawdown_already_recorded(
            conn, cik, m.instrument_id, ev_iso, amount, shares,
        ):
            log.info(
                "drawdown re-disclosure suppressed: instrument=%s "
                "date=%s amount=%s accession=%s",
                m.instrument_id, ev_iso, amount, accession,
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
            # "Last Banker" lookups on shelf cards. Drawdown party is
            # distinct from the parent instrument's counterparty
            # (shelves don't have a banker; each takedown does).
            # placement_agent → role='bank'; counterparty → 'investor'.
            pa = fields.get("placement_agent_canonical")
            cp = fields.get("counterparty_canonical")
            if pa:
                party_canon, party_role = pa, "bank"
            elif cp:
                party_canon, party_role = cp, "investor"
            else:
                party_canon, party_role = None, None
            conn.execute(
                """INSERT INTO dilution_ledger_drawdowns
                     (cik, instrument_id, accession_number, event_date,
                      amount_usd, shares, price,
                      drawdown_party_canonical, drawdown_party_role,
                      detected_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (cik, m.instrument_id, accession, ev_iso,
                 amount or None, shares or None,
                 fields.get("price") or fields.get("avg_price"),
                 party_canon, party_role, now_iso()),
            )
            drew = True
            action = "drawn_down"
    elif m.event_kind == "closing" and (row["type"] or "").lower() == "equity":
        # Equity (off-shelf PIPE) close: book the cash into the
        # drawdowns index. NO warrant-style date-rebase and NO relabel
        # — equity has no term to preserve, and the card keeps its
        # announcement-month label.
        closing_iso = m.event_date.isoformat()
        if terms.get("closing_date") != closing_iso:
            terms["closing_date"] = closing_iso
            terms_changed = True
        # Count true-up first so the amount fallback (count × price)
        # uses the final issued count.
        count_actual = fields.get("count_actual")
        if count_actual is not None:
            outstanding["count"] = float(count_actual)
        drew = _maybe_write_equity_drawdown(
            conn, cik, m.instrument_id, row["counterparty_canonical"],
            terms, outstanding, fields.get("gross_proceeds_usd"),
            closing_iso, accession,
        )
        action = "closing_confirmed"
    elif m.event_kind == "closing":
        # Closing-relabel: a previously-announced tranche is now
        # actually issued. The signing filing's create stamped the row
        # with the announcement date; this event re-bases issue_date /
        # exercisable_date / expiration so the card matches DT's
        # closing-date convention.
        closing_iso = m.event_date.isoformat()
        # Old issue date for term-preservation math. Prefer an
        # explicit terms.issue_date the create supplied; fall back to
        # the row's created_at (the signing filing's event_date).
        old_issue_raw = terms.get("issue_date") or row["created_at"]
        try:
            old_issue = _d.fromisoformat(str(old_issue_raw)[:10])
        except (TypeError, ValueError):
            old_issue = None
        # Shift expiration by the same delta so the N-year term
        # measured "from issuance" stays intact.
        old_exp_raw = terms.get("expiration")
        if old_exp_raw and old_issue:
            try:
                old_exp = _d.fromisoformat(str(old_exp_raw)[:10])
                delta = m.event_date - old_issue
                terms["expiration"] = (old_exp + delta).isoformat()
                terms_changed = True
            except (TypeError, ValueError):
                pass
        # Shift convertible_date by the same delta — an N-month lockup
        # "commencing on the date of issuance" tracks the re-based
        # issuance, exactly like expiration above (SCNI EIB: 12-month
        # lockup anchored to the 08-13 announcement create stayed
        # frozen while issue_date re-based to the 08-21 closing).
        # ONLY a convertible_date still equal to its CREATE-time value
        # is announcement-anchored; once anything moved it (an LLM
        # amend already correcting it to the closing-based lockup, or
        # a prior closing event's own shift) re-shifting double-applies
        # the delta (round-6 EIB: amend fixed 08-13→08-21, the shift
        # then pushed it to 08-29).
        old_cd_raw = terms.get("convertible_date")
        if old_cd_raw and old_issue:
            try:
                hist0 = json.loads(row["history_json"] or "[]")
            except (TypeError, ValueError):
                hist0 = []
            create_cd = None
            for e in hist0:
                if e.get("action") == "created":
                    create_cd = ((e.get("fields_changed") or {})
                                 .get("terms") or {}).get("convertible_date")
                    break
            if create_cd and str(old_cd_raw)[:10] == str(create_cd)[:10]:
                try:
                    old_cd = _d.fromisoformat(str(old_cd_raw)[:10])
                    delta = m.event_date - old_issue
                    terms["convertible_date"] = (old_cd + delta).isoformat()
                    terms_changed = True
                except (TypeError, ValueError):
                    pass
        if terms.get("issue_date") != closing_iso:
            terms["issue_date"] = closing_iso
            terms_changed = True
        if terms.get("exercisable_date") != closing_iso:
            terms["exercisable_date"] = closing_iso
            terms_changed = True
        # Count true-up. Only widen initial_count when no exercises /
        # terminations have happened yet — same guard as _apply_amend's
        # upward-count clarification, since "closing came after the
        # tranche already started moving" is a coherence-warning case
        # the apply layer shouldn't paper over.
        count_actual = fields.get("count_actual")
        if count_actual is not None:
            new_count = float(count_actual)
            outstanding["count"] = new_count
            if (not float(outstanding.get("exercised_to_date") or 0)
                    and not float(outstanding.get("terminated_to_date") or 0)):
                initial = float(outstanding.get("initial_count") or 0)
                if new_count > initial:
                    outstanding["initial_count"] = new_count
                elif initial == 0:
                    outstanding["initial_count"] = new_count
        action = "closing_confirmed"
    else:
        action = m.event_kind  # forward-compat

    history = json.loads(row["history_json"] or "[]")
    history.append({
        "date": _ev(m, filing_date),
        "accession": accession, "form": form, "action": action,
        "fields_changed": fields,
    })
    if pref_closed:
        # A full preferred retirement (count→0) closes the row, so the
        # preliminary tranche of a loan-to-equity rollover drops off the
        # card surface — a count=0 *perpetual* preferred is not caught by
        # _preferred_dead (no maturity), so close it explicitly here.
        history.append({
            "date": _ev(m, filing_date),
            "accession": accession, "form": form, "action": "closed",
            "fields_changed": {"reason": "redeemed", "auto": True,
                               "via": "principal_redemption"},
        })
    # Closing event relabels the card by closing date. Same helper as
    # the amend_warrant(issue_date=...) path uses, so card titles
    # converge regardless of which tool the walker emitted. Equity is
    # exempt: PIPE cards keep their announcement-month label.
    new_label_value: str | None = None
    if (m.event_kind == "closing"
            and (row["type"] or "").lower() != "equity"):
        new_label_value = _relabel_with_issue_date(
            row["label"], m.event_date.isoformat(),
        )
    if new_label_value and new_label_value != row["label"]:
        conn.execute(
            """UPDATE dilution_ledger
                 SET terms_json=?, outstanding_json=?, history_json=?,
                     last_seen_accession=?, last_seen_date=?, label=?
               WHERE instrument_id=?""",
            (_to_json(terms), _to_json(outstanding), _to_json(history),
             accession, filing_date, new_label_value, m.instrument_id),
        )
    elif terms_changed:
        conn.execute(
            """UPDATE dilution_ledger
                 SET terms_json=?, outstanding_json=?, history_json=?,
                     last_seen_accession=?, last_seen_date=?
               WHERE instrument_id=?""",
            (_to_json(terms), _to_json(outstanding), _to_json(history),
             accession, filing_date, m.instrument_id),
        )
    else:
        conn.execute(
            """UPDATE dilution_ledger
                 SET outstanding_json=?, history_json=?,
                     last_seen_accession=?, last_seen_date=?
               WHERE instrument_id=?""",
            (_to_json(outstanding), _to_json(history),
             accession, filing_date, m.instrument_id),
        )
    if pref_closed:
        conn.execute(
            "UPDATE dilution_ledger SET status=?, status_at=? "
            "WHERE instrument_id=?",
            ("redeemed", filing_date, m.instrument_id),
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
    fields_changed: dict[str, Any] = {"reason": m.reason,
                                      "replaced_by": m.replaced_by}
    out = json.loads(row["outstanding_json"] or "{}")
    out_dirty = False
    # A redeemed close with material outstanding is the implicit final
    # cash repayment (validate.py only lets this through from 8-K/6-K
    # event filings — "paid in full" disclosures; the anchor's
    # amend(…=0)+close pairs arrive pre-zeroed). Zero the gating
    # balance so the dead row can't feed stale overhang to anchor
    # snapshots or get resurrected at face value, and keep the retired
    # amount in cumulative *_to_date keys + the history entry.
    # 'converted' gets the same treatment: with the as-of subsumption
    # veto, in-batch conversion decrements a stated balance already
    # covered no longer reach the store, so the validator's overlay
    # (which does its own arithmetic) can accept a converted close
    # while the store-side balance is still positive — the close is
    # the authoritative final event, so reconcile here.
    if m.reason in ("redeemed", "converted"):
        _verb = m.reason  # 'redeemed' | 'converted'
        def _pos(v):
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            return f if f > 0 else None
        pr = _pos(out.get("principal_remaining"))
        if pr is not None:
            out["principal_remaining"] = 0
            out[f"principal_{_verb}_to_date"] = (
                float(out.get(f"principal_{_verb}_to_date") or 0) + pr
            )
            fields_changed[f"principal_{_verb}_at_close"] = pr
            out_dirty = True
        if (row["type"] or "").lower() in ("warrant", "preferred"):
            cnt = _pos(out.get("count"))
            if cnt is not None:
                out["count"] = 0
                out[f"count_{_verb}_to_date"] = (
                    float(out.get(f"count_{_verb}_to_date") or 0) + cnt
                )
                fields_changed[f"count_{_verb}_at_close"] = cnt
                out_dirty = True
    history.append({
        "date": _ev(m, filing_date),
        "accession": accession, "form": form, "action": "closed",
        "fields_changed": fields_changed,
    })
    if out_dirty:
        conn.execute(
            """UPDATE dilution_ledger
                 SET status=?, status_at=?, history_json=?,
                     outstanding_json=?,
                     last_seen_accession=?, last_seen_date=?
               WHERE instrument_id=?""",
            (status, _ev(m, filing_date), _to_json(history),
             _to_json(out), accession, filing_date, m.instrument_id),
        )
    else:
        conn.execute(
            """UPDATE dilution_ledger
                 SET status=?, status_at=?, history_json=?,
                     last_seen_accession=?, last_seen_date=?
               WHERE instrument_id=?""",
            (status, _ev(m, filing_date), _to_json(history),
             accession, filing_date, m.instrument_id),
        )
    # Deterministic shelf-rollover on supersession. Previously, sibling
    # migration only happened when the LLM happened to RE-DISCLOSE the
    # ATM via create_atm on the new-shelf filing (dedup collapse →
    # _append_redisclosure → rollover). When it instead emits only
    # close_instrument(old_shelf, superseded) — as it legitimately may —
    # the still-active ATM stayed glued to the dead shelf's file_number
    # and every file_number rollup (raised, last_banker, raisable) went
    # to zero (round-4 cgen-mar2023). The store owns the migration now:
    # on a shelf superseded-close, re-point active shelf-hosted siblings
    # from the dead shelf's file_number to the issuer's live successor.
    if ((row["type"] or "").lower() == "shelf"
            and m.reason == "superseded"):
        try:
            _migrate_shelf_siblings_on_supersede(
                conn, cik, row, m.replaced_by, accession, form,
                filing_date,
            )
        except Exception:
            log.warning(
                "  shelf-sibling migration failed for %s",
                m.instrument_id, exc_info=True,
            )


def _migrate_shelf_siblings_on_supersede(
    conn: sqlite3.Connection, cik: int, closed_row: sqlite3.Row,
    replaced_by: str | None, accession: str, form: str,
    filing_date: str,
) -> None:
    """Re-point active atm/equity_line/s1_offering rows hosted on a
    just-superseded shelf's file_number at the successor shelf, so the
    file_number-walk rollups follow the live registration. Mirrors the
    rollover block in _append_redisclosure, but runs deterministically
    from the shelf close instead of relying on an LLM redisclosure."""
    old_host = (closed_row["registration_accession"]
                or closed_row["created_accession"])
    fn_row = conn.execute(
        "SELECT file_number FROM dilution_filings "
        "WHERE accession_number = ?", (old_host,),
    ).fetchone()
    old_fn = fn_row["file_number"] if fn_row else None
    if not old_fn:
        return
    # Successor: the named replaced_by when it is a live shelf, else the
    # newest active shelf on a DIFFERENT file_number.
    succ = None
    if replaced_by:
        succ = conn.execute(
            "SELECT instrument_id, created_accession, "
            "       registration_accession FROM dilution_ledger "
            "WHERE cik=? AND instrument_id=? AND type='shelf' "
            "  AND status='active'", (cik, replaced_by),
        ).fetchone()
    if succ is None:
        succ = conn.execute(
            "SELECT instrument_id, created_accession, "
            "       registration_accession FROM dilution_ledger "
            "WHERE cik=? AND type='shelf' AND status='active' "
            "  AND instrument_id != ? "
            "ORDER BY created_at DESC LIMIT 1",
            (cik, closed_row["instrument_id"]),
        ).fetchone()
    if succ is None:
        return
    new_host = succ["registration_accession"] or succ["created_accession"]
    fn_row = conn.execute(
        "SELECT file_number FROM dilution_filings "
        "WHERE accession_number = ?", (new_host,),
    ).fetchone()
    new_fn = fn_row["file_number"] if fn_row else None
    if not new_fn or new_fn == old_fn:
        return
    sibs = conn.execute(
        f"""SELECT instrument_id, type, created_accession,
                   registration_accession, history_json
              FROM dilution_ledger
             WHERE cik=? AND status='active'
               AND type IN ({','.join('?' * len(_SHELF_HOSTED_TYPES))})""",
        (cik, *_SHELF_HOSTED_TYPES),
    ).fetchall()
    for sib in sibs:
        cur_host = (sib["registration_accession"]
                    or sib["created_accession"])
        cur_fn_row = conn.execute(
            "SELECT file_number FROM dilution_filings "
            "WHERE accession_number = ?", (cur_host,),
        ).fetchone()
        cur_fn = cur_fn_row["file_number"] if cur_fn_row else None
        if cur_fn != old_fn:
            continue
        hist = json.loads(sib["history_json"] or "[]")
        hist.append({
            "date": filing_date, "accession": accession, "form": form,
            "action": "rolled_over_to_new_shelf",
            "fields_changed": {
                "registration_accession": {"from": cur_host,
                                           "to": new_host},
                "file_number": {"from": old_fn, "to": new_fn},
                "trigger": "shelf_superseded_close",
            },
        })
        conn.execute(
            "UPDATE dilution_ledger SET registration_accession=?, "
            "history_json=? WHERE instrument_id=?",
            (new_host, _to_json(hist), sib["instrument_id"]),
        )
        log.info(
            "  shelf-rollover (supersede): %s migrated from %s "
            "(file %s) to %s (file %s)",
            sib["instrument_id"], cur_host, old_fn, new_host, new_fn,
        )


def close_converted_preferred(
    cik: int, *, conversion_date: _d, accession: str, form: str,
    filing_date: str,
) -> list[str]:
    """Deterministically close every ACTIVE preferred issued on/before
    ``conversion_date`` as ``converted``.

    Called by the walker when a periodic filing AFFIRMS that all preferred
    stock automatically/mandatorily converted to common with NONE remaining
    outstanding (the Nasdaq-equity-compliance pattern: KSCP's Series
    A/B/M/S converted 2024-05-15, "no shares of Preferred Stock outstanding
    after the Preferred Stock Conversion Date"). The overhang LLM routinely
    re-matches the named series in the conversion narrative without flagging
    is_terminated, so the anchor never closes them and they linger as
    phantom-active cards. Routes through ``_apply_close(reason='converted')``
    so the count is zeroed and ``count_converted_to_date`` is tracked.

    Scoped by ``created_at <= conversion_date`` so a NEW preferred issued
    AFTER the conversion (a later re-issuance, same issuer re-using a series
    letter) is never swept up. Idempotent: only touches ``status='active'``
    rows, so re-firing on a later filing that repeats the conversion note is
    a no-op. Returns the closed instrument_ids."""
    closed: list[str] = []
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT instrument_id FROM dilution_ledger "
            "WHERE cik=? AND type='preferred' AND status='active' "
            "  AND date(created_at) <= date(?) "
            "ORDER BY instrument_id",
            (cik, conversion_date.isoformat()),
        ).fetchall()
        for r in rows:
            _apply_close(
                conn, cik,
                CloseInstrument(instrument_id=r["instrument_id"],
                                reason="converted",
                                event_date=conversion_date),
                accession, form, filing_date,
            )
            closed.append(r["instrument_id"])
    return closed


# Balance below which a debt instrument counts as retired. Mirrors
# cards._CONVERTIBLE_DUST_ABS_USD / _CONVERTIBLE_DUST_REL — deliberately
# duplicated rather than imported, because store must not depend on the
# projection layer, and the two answer different questions that happen to
# share a number (cards: "should this render?", store: "is this retired?").
_RETIRED_DUST_ABS_USD = 1_000.0
_RETIRED_DUST_REL = 0.005
# Slack on the retired-flow corroboration. Conversion / redemption figures
# are filing-rounded and accrued interest is converted alongside principal,
# so require 99% of face retired rather than an exact match.
_RETIRED_FLOW_TOLERANCE = 0.01


def close_retired_debt(
    cik: int, *, accession: str, form: str, filing_date: str,
) -> list[str]:
    """Close every ACTIVE convertible whose balance has reached dust AND
    whose retired-to-date flow corroborates full retirement.

    A row sitting at ``principal_remaining=0`` while still ``active`` is a
    lifecycle lie: the cap table says the note is live, the balance says it
    is gone. Today only the projection layer notices — ``cards.
    _convertible_dead`` drops dust rows — so the state stays wrong and
    every non-card reader (anchor, badges, the Finviz payload's card
    absence) inherits it.

    Corroboration is REQUIRED, and that is the whole design. Closing on a
    zero balance alone would close rows the anchor believes are live, and
    the anchor wins the next round: CETY C-1143 already oscillates
    ``closed redeemed → reopened "overhang re-lists as outstanding
    (anchor-corroborated)" → amended back up`` five times over. When the
    flow does NOT account for the face, the balance is the thing in doubt,
    not the status — so we log and leave the row alone rather than pick a
    fight we lose.

    Reads the two clamped accumulators (`principal_converted_to_date`,
    `principal_redeemed_to_date`); the clamps are what make them safe to
    compare against face. Idempotent — active rows only. Returns the closed
    instrument_ids."""
    closed: list[str] = []
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT instrument_id, terms_json, outstanding_json "
            "FROM dilution_ledger "
            "WHERE cik=? AND type='convertible' AND status='active' "
            "ORDER BY instrument_id",
            (cik,),
        ).fetchall()
        for r in rows:
            try:
                terms = json.loads(r["terms_json"] or "{}")
                out = json.loads(r["outstanding_json"] or "{}")
            except (TypeError, ValueError):
                continue
            remaining = out.get("principal_remaining")
            face = terms.get("principal")
            if not isinstance(remaining, (int, float)):
                continue
            if not isinstance(face, (int, float)) or face <= 0:
                continue
            is_dust = (remaining < _RETIRED_DUST_ABS_USD
                       or remaining / face < _RETIRED_DUST_REL)
            if not is_dust:
                continue
            converted = float(out.get("principal_converted_to_date") or 0)
            redeemed = float(out.get("principal_redeemed_to_date") or 0)
            if converted + redeemed < face * (1 - _RETIRED_FLOW_TOLERANCE):
                log.warning(
                    "  %s zero-balance but flow unaccounted — leaving "
                    "active: remaining=%.2f face=%.0f converted=%.0f "
                    "redeemed=%.0f (balance is the suspect, not the status)",
                    r["instrument_id"], remaining, face, converted, redeemed,
                )
                continue
            reason = "converted" if converted >= redeemed else "redeemed"
            _apply_close(
                conn, cik,
                CloseInstrument(instrument_id=r["instrument_id"],
                                reason=reason,
                                event_date=_d.fromisoformat(
                                    filing_date[:10])),
                accession, form, filing_date,
            )
            closed.append(f"{r['instrument_id']}:{reason}")
    return closed


def _find_unchanged_atm_program(
    conn: sqlite3.Connection, cik: int, m: "CreateMutation",
) -> str | None:
    """The instrument_id of a still-active ATM this create re-registers
    UNCHANGED (same agent by _agents_same, capacity within create
    tolerance), or None. Both signals are required — capacity alone
    collapsed concurrent programs historically (XTIA's two ~$25M Maxim
    ATMs), and agent alone would merge every same-bank re-up regardless
    of size. A changed capacity is a new supplement window and chains
    via _auto_supersede_prior_atm instead."""
    cap = (m.terms or {}).get("capacity_usd")
    agent = (m.placement_agent_canonical or "").strip()
    if cap is None or not agent:
        return None
    rows = conn.execute(
        "SELECT instrument_id, terms_json, placement_agent_canonical "
        "FROM dilution_ledger "
        "WHERE cik=? AND type='atm' AND status='active'",
        (cik,),
    ).fetchall()
    for r in rows:
        r_agent = (r["placement_agent_canonical"] or "").strip()
        if not r_agent or not _agents_same(r_agent, agent):
            continue
        try:
            r_cap = json.loads(r["terms_json"] or "{}").get("capacity_usd")
        except (TypeError, ValueError):
            continue
        if r_cap is None:
            continue
        if _close(cap, r_cap, _CREATE_AMOUNT_TOLERANCE):
            return r["instrument_id"]
    return None


def _auto_supersede_prior_atm(
    conn: sqlite3.Connection, cik: int, new_id: str, m: "CreateMutation",
    accession: str, form: str, filing_date: str,
    same_agent_only: bool = False,
) -> None:
    """A genuinely-new ATM registered on a fresh shelf-host filing
    supersedes the issuer's prior active ATM — DilutionTracker's
    one-ATM-program-at-a-time convention. The STORE owns this rather
    than the walker, which proved unreliable across re-walks at BOTH
    (a) remembering to emit close_instrument against the old card, and
    (b) dating the successor — it copies the embedded prospectus's
    restated ORIGINAL signing date verbatim, so the new Leerink card
    renders as 'January 2023' instead of the registration month.

    Fires only on a true new instrument (callers gate on
    was_redisclosure=False) from S-3/F-3-family forms, and only when a
    prior ATM is still active. Same-terms re-registrations collapse
    (redisclosure → never reach here) and 424B takedowns aren't
    shelf-host forms, so neither trips this path.
    """
    priors = conn.execute(
        "SELECT instrument_id, terms_json, history_json, "
        "       placement_agent_canonical "
        "  FROM dilution_ledger "
        " WHERE cik=? AND type='atm' AND status='active' "
        "   AND instrument_id != ?",
        (cik, new_id),
    ).fetchall()
    if same_agent_only:
        # Supplement-form (424B5) chaining: a re-registration replaces
        # only the SAME bank's prior tranche; an unrelated concurrent
        # program from a different agent stays live.
        priors = [p for p in priors
                  if _agents_same(p["placement_agent_canonical"],
                                  m.placement_agent_canonical)]
    if not priors:
        return
    new_ad = str((m.terms or {}).get("agreement_date") or "")[:10]
    prior_ads: list[str] = []
    for p in priors:
        hist = json.loads(p["history_json"] or "[]")
        hist.append({
            "date": filing_date, "accession": accession, "form": form,
            "action": "closed",
            "fields_changed": {"reason": "superseded",
                               "replaced_by": new_id, "auto": True},
        })
        conn.execute(
            "UPDATE dilution_ledger SET status=?, status_at=?, "
            "history_json=? WHERE instrument_id=?",
            (f"superseded:{new_id}", filing_date, _to_json(hist),
             p["instrument_id"]),
        )
        prior_ads.append(
            str(json.loads(p["terms_json"] or "{}").get("agreement_date")
                or "")[:10]
        )
        log.info(
            "  auto-supersede: ATM %s → superseded by new %s "
            "(new ATM registered on %s while prior active)",
            p["instrument_id"], new_id, form,
        )

    # A genuinely-new superseding ATM starts FRESH — it inherits no
    # outstanding from the predecessor it replaces. The LLM routinely
    # copies the program's cumulative "$X sold to date" onto the new
    # create's drawn_usd (CGEN: $15.1M pinned onto the fresh Leerink
    # ATM, showing it 85% drawn at birth). Reset to zero drawn / full
    # remaining; real takedowns book against it from here forward. Runs
    # unconditionally on supersession — the re-date below is separate and
    # conditional.
    succ = conn.execute(
        "SELECT terms_json, outstanding_json FROM dilution_ledger "
        "WHERE instrument_id=?", (new_id,),
    ).fetchone()
    if succ is not None:
        out = json.loads(succ["outstanding_json"] or "{}")
        inherited_drawn = out.get("drawn_usd")
        if inherited_drawn or out.get("sold_to_date"):
            cap = json.loads(succ["terms_json"] or "{}").get("capacity_usd")
            out["drawn_usd"] = 0.0
            out.pop("sold_to_date", None)
            # No inherited cumulative checkpoint either (a fresh program
            # has raised nothing yet) — clear it so the card layer falls
            # back to the successor's own discrete take-downs.
            out.pop("drawn_usd_anchor", None)
            out.pop("drawn_usd_asof", None)
            if cap is not None:
                out["remaining_capacity_usd"] = float(cap)
            else:
                out.pop("remaining_capacity_usd", None)
            conn.execute(
                "UPDATE dilution_ledger SET outstanding_json=? "
                "WHERE instrument_id=?",
                (_to_json(out), new_id),
            )
            log.info(
                "  auto-supersede: reset successor %s to fresh "
                "(dropped inherited drawn_usd=%s) — a superseding ATM "
                "starts with zero outstanding",
                new_id, inherited_drawn,
            )

    # Re-date the successor when it carried the restated ORIGINAL signing
    # date (<= a predecessor's) rather than the new registration date —
    # DT dates a rebrand/amendment ATM by its new registration filing.
    if new_ad and any(pa and pa >= new_ad for pa in prior_ads):
        try:
            new_label = build_label(
                dataclasses.replace(m, agreement_date=_d.fromisoformat(
                    filing_date[:10]))
            )
        except (ValueError, TypeError):
            new_label = None
        row = conn.execute(
            "SELECT terms_json, label FROM dilution_ledger "
            "WHERE instrument_id=?", (new_id,),
        ).fetchone()
        terms = json.loads(row["terms_json"] or "{}")
        terms["agreement_date"] = filing_date
        conn.execute(
            "UPDATE dilution_ledger SET terms_json=?, created_at=?, "
            "status_at=?, label=? WHERE instrument_id=?",
            (_to_json(terms), filing_date, filing_date,
             new_label or row["label"], new_id),
        )
        log.info(
            "  auto-supersede: re-dated successor %s to %s "
            "(it carried the restated original date %s)",
            new_id, filing_date, new_ad,
        )


def _apply_restate_atm(
    conn: sqlite3.Connection, cik: int, ticker: str, m: "RestateAtm",
    accession: str, form: str, filing_date: str,
    seq_state: dict[str, int],
) -> tuple[str, str | None]:
    """Apply an amended-and-restated ATM (the restate_atm tool).

    Mints a FRESH successor ATM (drawn reset to zero — the predecessor's
    cumulative sales stay on its own card, DT's convention) and, when
    ``supersede_prior`` is set and the predecessor is still active, marks
    it ``superseded:<new>``. Returns ``(new_id, prior_id)`` where
    ``prior_id`` is the superseded predecessor (or None when it stays
    live / the successor collapsed onto an existing row).

    This is the explicit successor to the old
    ``_try_promote_atm_amend_to_restate`` heuristic: the create vs amend
    decision and the supersede decision are both made by the walker and
    carried in the tool call, not inferred here.
    """
    prior = conn.execute(
        "SELECT instrument_id, status, history_json, terms_json, "
        "       placement_agent_canonical "
        "FROM dilution_ledger WHERE cik=? AND instrument_id=?",
        (cik, m.predecessor_id),
    ).fetchone()
    # Unchanged-program guard: a "restate" whose capacity equals the
    # named predecessor's (and whose agent matches) is a prospectus
    # refresh / re-registration of the SAME window, not an amendment —
    # minting a re-dated successor splits one program into a 'Replaced'
    # original plus a duplicate card (round-4 cety-roth: the Dec-2025
    # 424B3 re-registered the unchanged Oct-2023 $25M Roth agreement).
    # Dedup can't catch it because the LLM dates the restate by the new
    # filing. Treat as a redisclosure of the predecessor instead.
    if prior is not None and (prior["status"] or "active") == "active":
        try:
            _p_cap = json.loads(
                prior["terms_json"] or "{}").get("capacity_usd")
        except (TypeError, ValueError):
            _p_cap = None
        _p_agent = (prior["placement_agent_canonical"] or "").strip()
        _m_agent = (m.placement_agent_canonical or "").strip()
        if (_p_cap is not None and m.capacity_usd is not None
                and _close(float(m.capacity_usd), float(_p_cap),
                           _CREATE_AMOUNT_TOLERANCE)
                and (not _m_agent or not _p_agent
                     or _agents_same(_p_agent, _m_agent))):
            hist = json.loads(prior["history_json"] or "[]")
            hist.append({
                "date": _ev(m, filing_date), "accession": accession,
                "form": form, "action": "redisclosed",
                "fields_changed": {"via": "restate_unchanged_program"},
            })
            conn.execute(
                "UPDATE dilution_ledger SET history_json=?, "
                "last_seen_accession=?, last_seen_date=? "
                "WHERE instrument_id=?",
                (_to_json(hist), accession, filing_date,
                 m.predecessor_id),
            )
            log.info(
                "  restate_atm: %s re-registers the unchanged program "
                "(same agent, same capacity) — redisclosure, no "
                "successor minted", m.predecessor_id,
            )
            return m.predecessor_id, None
    # Re-date guard. An amended-and-restated prospectus quotes the
    # ORIGINAL agreement's signing date verbatim ("we previously entered
    # into a sales agreement dated [old date]"), and the LLM routinely
    # copies that into agreement_date — which would label the restated
    # card with the predecessor's month (FCEL: a 2024-04-10 restatement
    # mislabeled "July 2022"). When the supplied agreement_date matches
    # the predecessor's, fall back to the filing/event date so the
    # successor reads as the restatement month. DT keys ATM cards by this
    # date, so getting it right is what splits the new card from the old.
    agreement_date = m.agreement_date or m.event_date
    if prior is not None and m.agreement_date is not None:
        prior_ad = json.loads(prior["terms_json"] or "{}").get("agreement_date")
        if prior_ad and str(prior_ad)[:10] == m.agreement_date.isoformat():
            log.info(
                "  restate_atm: supplied agreement_date %s matches "
                "predecessor %s — re-dating successor to filing date %s",
                m.agreement_date.isoformat(), m.predecessor_id, m.event_date,
            )
            agreement_date = m.event_date
    new_remaining = (
        m.remaining_capacity_usd
        if m.remaining_capacity_usd is not None
        else float(m.capacity_usd)
    )
    synth = CreateAtm(
        capacity_usd=float(m.capacity_usd),
        event_date=m.event_date,
        agreement_date=agreement_date,
        agreement_end_date=m.agreement_end_date,
        placement_agent_canonical=m.placement_agent_canonical,
        remaining_capacity_usd=float(new_remaining),
        drawn_usd=0.0,
        proposed_id=m.proposed_id,
    )
    new_id, was_red, _ = _apply_create(
        conn, cik, ticker, synth, accession, form, filing_date, seq_state,
    )
    if was_red and new_id != m.predecessor_id:
        # The successor collapsed onto a row OTHER than the predecessor
        # the LLM explicitly named — the redisclosure dedup mis-keyed on
        # a stale sibling (KSCP: the Nov-2024 $25M restate collapsed
        # onto the Feb-2023 row whose capacity had been amended to $25M,
        # so no Nov-2024 instrument ever existed and the $25M landed on
        # an invisible row). The restate's named predecessor is the
        # stronger identity signal: force a real successor row and let
        # the supersede logic below chain it.
        log.info(
            "  restate_atm: dedup collapsed successor onto %s but "
            "predecessor is %s — forcing a distinct successor row",
            new_id, m.predecessor_id,
        )
        new_id, was_red, _ = _apply_create(
            conn, cik, ticker, synth, accession, form, filing_date,
            seq_state, skip_dedup=True,
        )
    elif was_red:
        # The successor collapsed onto the named predecessor itself (the
        # restatement didn't materially change the program — same agent,
        # same agreement_date, capacity within tolerance). The create
        # path already de-duped; leave the predecessor untouched.
        log.info(
            "  restate_atm: successor collapsed onto existing %s — "
            "predecessor %s left as-is", new_id, m.predecessor_id,
        )
        return new_id, None
    # supersede_prior is no longer consulted — the existence of a restate
    # pointing predecessor_id at a LIVE row is itself the supersession
    # signal (a genuinely concurrent, independent program is a create_atm,
    # never a restate of the other). The was_red short-circuit above
    # already handles the "restatement didn't materially change the
    # program" case, and the guards below protect against double-supersede
    # / missing predecessor. This is DT's one-ATM-program-at-a-time
    # convention, owned deterministically by the store.
    prior = conn.execute(
        "SELECT instrument_id, status, history_json "
        "FROM dilution_ledger WHERE cik=? AND instrument_id=?",
        (cik, m.predecessor_id),
    ).fetchone()
    if prior is None:
        log.warning(
            "  restate_atm: predecessor %s not found at apply time — "
            "successor %s stands alone", m.predecessor_id, new_id,
        )
        return new_id, None
    if (prior["status"] or "active") != "active":
        log.info(
            "  restate_atm: predecessor %s already %s — not re-superseding",
            m.predecessor_id, prior["status"],
        )
        return new_id, None
    hist = json.loads(prior["history_json"] or "[]")
    hist.append({
        "date": filing_date, "accession": accession, "form": form,
        "action": "closed",
        "fields_changed": {"reason": "superseded",
                           "replaced_by": new_id, "auto": True,
                           "via": "restate"},
    })
    conn.execute(
        "UPDATE dilution_ledger SET status=?, status_at=?, "
        "history_json=? WHERE instrument_id=?",
        (f"superseded:{new_id}", filing_date, _to_json(hist),
         m.predecessor_id),
    )
    log.info(
        "  restate_atm: %s on %s → created %s, superseded predecessor %s",
        m.predecessor_id, form, new_id, m.predecessor_id,
    )
    return new_id, m.predecessor_id



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
    # Status filter mirrors the card-projection filter in cards.py:
    # warrant/convertible/preferred cards include both active rows AND
    # superseded:* rows (multi-tranche offerings, inducement exchanges
    # — the predecessor warrant is still shown as a historical card).
    # If apply_split skipped superseded:* rows, those cards would
    # render with PRE-split counts/strikes while the live siblings
    # render post-split (SCNI September 2023 warrants drifted by 10×
    # count and ÷10 strike for exactly this reason). warrant_cards ALSO
    # renders status='exercised' rows (fully-exercised tranches DT keeps
    # on the card list), so a warrant closed BEFORE a split must still
    # be rescaled or it renders pre-split units next to post-split
    # siblings — SCNI W-4273 (exercised 2024-01-04, split 2024-05-21)
    # showed 1,146,552 @ $1.16 against DT's 114,655 @ $11.60. The
    # eff_iso <= created_at guard below still prevents re-applying a
    # split to an instrument whose terms were already disclosed in
    # post-split units.
    rows = conn.execute(
        """SELECT instrument_id, type, terms_json, outstanding_json,
                  history_json, created_at, status
             FROM dilution_ledger
            WHERE cik=?
              AND (status='active' OR status LIKE 'superseded:%'
                   OR (status='exercised' AND type='warrant'))
              AND type IN ('warrant', 'convertible', 'preferred')""",
        (cik,),
    ).fetchall()
    # Issuer-level unit default. FPI ADS issuers report every share-
    # count in ADS units (see _llm_utils.unit_preamble), but the LLM
    # historically doesn't stamp `terms.units="ads"` on every
    # create_instrument. Without this default an `apply_split(units="ads")`
    # from an ADS-ratio-change 6-K (e.g. XTLB 2026-03-20 / 1:100 →
    # 1:400) matched zero rows and silently no-op'd. Per-instrument
    # `units` still wins when explicitly set, so an FPI that issues
    # underlying-common paper can override case-by-case.
    fpi_row = conn.execute(
        "SELECT is_fpi FROM dilution_company WHERE cik=?", (cik,),
    ).fetchone()
    default_units = "ads" if (fpi_row and fpi_row["is_fpi"]) else "common"
    touched = 0
    for row in rows:
        # Splits effective on or before the instrument's creation are
        # already baked into the disclosed terms — the LLM extracted
        # post-split numbers from a filing dated after the split.
        # Re-applying inflates strike by 1/ratio per stale split,
        # which compounds catastrophically when the ledger walks a
        # multi-year history of pre-issuance splits.
        created_at = row["created_at"]
        eff_iso = _eff(m)
        if created_at and eff_iso[:10] <= created_at[:10]:
            continue
        terms = json.loads(row["terms_json"] or "{}")
        # Filter to matching unit. An explicit `terms.units` wins;
        # otherwise fall back to the issuer-level default computed
        # above (FPI → "ads", US → "common"). Non-string `units` (LLM
        # emitted a share count by mistake) is treated as missing.
        units_val = terms.get("units")
        instrument_units = (
            units_val.lower() if isinstance(units_val, str) else default_units
        )
        if instrument_units != m.units:
            continue
        applied = terms.get("applied_splits") or []
        if _split_already_applied(
            applied, effective_date=eff_iso,
            direction=m.direction,
        ):
            continue  # already applied (within fuzzy-dedup window)
        outstanding = json.loads(row["outstanding_json"] or "{}")
        for f in _COUNT_FIELDS:
            if f in terms and isinstance(terms[f], (int, float)):
                terms[f] = round(terms[f] * m.ratio)
            if f in outstanding and isinstance(outstanding[f], (int, float)):
                outstanding[f] = round(outstanding[f] * m.ratio)
        # Preferred $-terms split policy lives in one place, shared with the
        # amend-time _rescale_stale_unit_amend so the two passes can't drift:
        # stated_value is always fixed; conv_price is split-adjusted only for a
        # price-based series (no conversion_ratio). See
        # _preferred_price_split_skip for the full rationale + known gap.
        _skip_price = (
            _preferred_price_split_skip(terms)
            if row["type"] == "preferred" else frozenset()
        )
        for f in _PRICE_FIELDS:
            if f in _skip_price:
                continue
            if f in terms and isinstance(terms[f], (int, float)) and m.ratio:
                terms[f] = round(terms[f] / m.ratio, _PRICE_DECIMALS)
        applied.append({"date": eff_iso, "ratio": m.ratio,
                        "direction": m.direction})
        terms["applied_splits"] = applied
        history = json.loads(row["history_json"] or "[]")
        history.append({
            "date": eff_iso,
            "accession": accession, "form": form,
            "action": "split_applied",
            "fields_changed": {"post": m.post, "pre": m.pre,
                               "ratio": m.ratio,
                               "direction": m.direction,
                               "units": m.units},
            })
        # A split is a vendor-sourced synthetic event, not an SEC filing —
        # `accession` here is a "split:<effective_date>:<source>" marker,
        # not a real accession. Never write it to last_seen_accession: that
        # column must always hold a real accession (or NULL) or the
        # card/inspect EDGAR links resolve to "…/split:…/" → nowhere. The
        # split's provenance already lives in history_json (the
        # split_applied entry above) and the dilution_splits table.
        # last_seen_date IS still advanced to the split date so anchor.py
        # staleness / auto-close timing is unchanged by this fix.
        conn.execute(
            """UPDATE dilution_ledger
                 SET terms_json=?, outstanding_json=?, history_json=?,
                     last_seen_date=?
               WHERE instrument_id=?""",
            (_to_json(terms), _to_json(outstanding), _to_json(history),
             filing_date, row["instrument_id"]),
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
    return mutation_to_dict(m)


# ─── applied-mutation log ────────────────────────────────────────────
# The ledger is a deterministic fold of the mutations the walker applies,
# but the extraction that produces them is an LLM call: it costs money and
# re-running it yields different output. That makes the mutation stream
# source data and the ledger a projection of it. Recording the stream is
# what turns "the ledger got corrupted" from a full re-walk into a replay
# (scripts/rebuild_ledger.py).


def _pipeline_version() -> str | None:
    """The walker's version stamp, for replay provenance.

    Imported inside the function on purpose: walker_llm drags in the
    prompt and tool modules, several of which import this one, so a
    module-scope import would risk a cycle over a value that is only
    metadata. Absence is not an error — a store used outside a walk
    (tests, replay) has no walker version.
    """
    try:
        from .walker_llm import pipeline_version
        return pipeline_version()
    except Exception:
        return None


def ensure_mutation_log_conn(conn: sqlite3.Connection) -> None:
    """Idempotently create dilution_mutations on an existing connection.

    Mirrors ensure_walk_tables_conn: `init_dilution_db()` runs only on a
    fresh DB, so a live DB predating this table needs it created in place
    rather than failing the walk. Keep in sync with schema.py.
    """
    conn.execute(
        """CREATE TABLE IF NOT EXISTS dilution_mutations (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               cik INTEGER NOT NULL,
               accession_number TEXT NOT NULL,
               seq INTEGER NOT NULL,
               filing_date TEXT,
               form TEXT,
               kind TEXT NOT NULL,
               instrument_id TEXT,
               mutation_json TEXT NOT NULL,
               pipeline_version TEXT,
               applied_at TEXT NOT NULL
           )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dilution_mutations_cik "
        "ON dilution_mutations(cik, id)"
    )


def _log_applied_mutation(
    conn: sqlite3.Connection, *, cik: int, accession: str,
    filing_date: str, form: str, seq: int, mutation: Mutation,
    instrument_id: str | None, pipeline_version: str | None,
) -> None:
    """Record one mutation that HAS been applied.

    Called after the apply succeeded, never before — a log entry for a
    mutation the ledger didn't take would make a replay diverge, which is
    worse than no log at all.

    Plain INSERT — the table's autoincrement `id` is the application
    order, and that is what replay sorts on. One accession can pass
    through here in several separate apply_mutations calls (walk, then
    anchor corrections and pins), each with its own seq starting at 0, so
    (accession, seq) is NOT unique and must not be treated as a key.
    A --force re-walk clears the CIK first rather than overwriting rows.

    Note the codec: `mutation_to_record`, NOT the `mutation_to_dict` used
    for walk_errors. The latter is a flattened human-readable view with
    derived fields and is not invertible; this row exists to be replayed.
    """
    conn.execute(
        """INSERT INTO dilution_mutations
             (cik, accession_number, seq, filing_date, form, kind,
              instrument_id, mutation_json, pipeline_version, applied_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cik, accession, seq, filing_date, form, mutation.kind,
         instrument_id, _to_json(mutation_to_record(mutation)),
         pipeline_version, now_iso()),
    )


__all__ = [
    "ApplyResult",
    "apply_mutations",
    "get_drawdowns_by_instrument",
    "get_instrument",
    "get_open_instruments",
    "get_walk_state",
    "get_walked_accessions",
    "ensure_walk_tables",
    "ensure_mutation_log_conn",
    "seed_walked_from_positional",
    "mark_walked",
    "record_anchor_diffs",
    "reset_ledger_projection",
    "reset_walk_state",
]
