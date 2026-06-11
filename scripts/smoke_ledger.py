"""Spine smoke test: hand-built mutations → apply_mutations → verify state.

Runs against a temp SQLite file so it doesn't touch dilution.db. Walks
the most common shapes the walker LLM will emit:
  - apply_split before any other mutation
  - create_instrument with proposed_id (and one without)
  - amend_instrument
  - record_event drawdown / exercise / conversion
  - close_instrument (terminated, superseded chain)
  - rejection paths (missing id, illegal transition, type mismatch)

If this script prints "ALL OK" the ledger spine compiles end-to-end.
"""

import os
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Point the dilution package's DB_PATH at a temp file BEFORE importing
# anything that creates connections. config.DB_PATH is read at every
# get_conn() call, so reassigning it on the module is enough.
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()

import config  # noqa: E402
config.DB_PATH = _tmp.name

from dilution.schema import init_dilution_db  # noqa: E402
from dilution.ledger.mutations import (  # noqa: E402
    ApplySplit, CloseInstrument, ConfirmClosing,
    RecordExercise, RecordConversion, RecordDrawdown,
    amend_from_dict, create_from_dict,
)
from dilution.ledger.store import (  # noqa: E402
    apply_mutations,
    get_instrument,
    get_open_instruments,
    reset_walk_state,
)
from db import get_conn  # noqa: E402


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _build(spec: dict):
    """Translate a legacy-shape mutation dict into the typed dataclass.

    Used only by this smoke test to keep the test cases declarative.
    The walker LLM path constructs typed dataclasses directly via
    dilution/ledger/tools/parse.py — this helper exists solely so the
    smoke test can express test inputs as dicts.
    """
    kind = spec["kind"]
    if kind == "create_instrument":
        return create_from_dict(
            type_=spec["type"],
            terms=spec.get("terms") or {},
            outstanding=spec.get("outstanding") or {},
            counterparty_canonical=spec.get("counterparty_canonical"),
            placement_agent_canonical=spec.get("placement_agent_canonical"),
            descriptor=spec.get("descriptor"),
            proposed_id=spec.get("proposed_id"),
            event_date=spec.get("event_date"),
        )
    if kind == "amend_instrument":
        # Need the row's type to dispatch. The smoke test never amends a
        # row whose type isn't already known from a prior create; we hard-
        # code the dispatch by looking at field_updates / outstanding_updates
        # contents. Simpler: caller supplies `_type`.
        return amend_from_dict(
            type_=spec["_type"],
            instrument_id=spec["instrument_id"],
            field_updates=spec.get("field_updates") or {},
            outstanding_updates=spec.get("outstanding_updates") or {},
            event_date=spec.get("event_date") or "2099-01-01",
        )
    if kind == "record_event":
        ev_kind = spec["event_kind"]
        f = spec.get("fields") or {}
        ev_date = _d(spec["event_date"])
        iid = spec["instrument_id"]
        if ev_kind == "exercise":
            return RecordExercise(
                instrument_id=iid, shares=float(f.get("shares") or 0),
                event_date=ev_date,
                price=f.get("price"), gross_proceeds=f.get("gross_proceeds"),
            )
        if ev_kind == "conversion":
            pc = f.get("principal_converted")
            pref = f.get("preferred_shares_converted")
            return RecordConversion(
                instrument_id=iid,
                shares_issued=float(f.get("shares_issued") or 0),
                event_date=ev_date,
                principal_converted=float(pc) if pc is not None else None,
                preferred_shares_converted=float(pref) if pref is not None else None,
                principal_remaining=f.get("principal_remaining"),
            )
        if ev_kind == "drawdown":
            return RecordDrawdown(
                instrument_id=iid,
                drawdown_amount_usd=float(f.get("drawdown_amount_usd") or 0),
                drawdown_shares=float(f.get("drawdown_shares") or 0),
                event_date=ev_date,
                placement_agent_canonical=f.get("placement_agent_canonical"),
            )
        if ev_kind == "closing":
            ca = f.get("count_actual")
            gp = f.get("gross_proceeds_usd")
            return ConfirmClosing(
                instrument_id=iid, event_date=ev_date,
                count_actual=float(ca) if ca is not None else None,
                gross_proceeds_usd=float(gp) if gp is not None else None,
            )
        raise ValueError(f"unsupported event_kind in smoke test: {ev_kind!r}")
    if kind == "close_instrument":
        return CloseInstrument(
            instrument_id=spec["instrument_id"],
            reason=spec["reason"],
            replaced_by=spec.get("replaced_by"),
            event_date=_d(spec["event_date"]),
        )
    if kind == "apply_split":
        return ApplySplit(
            post=spec["post"], pre=spec["pre"],
            direction=spec["direction"],
            effective_date=_d(spec["effective_date"]),
            units=spec.get("units") or "common",
        )
    raise ValueError(f"unknown kind in smoke test: {kind!r}")


def _apply(cik, ticker, accession, form, dt, payload):
    mutations = [_build(spec) for spec in payload]
    return apply_mutations(
        cik=cik, ticker=ticker, accession=accession,
        form=form, filing_date=dt, mutations=mutations,
    )


def main():
    init_dilution_db()
    cik, ticker = 9999, "TEST"
    reset_walk_state(cik)

    # Filing 1 — IPO 424B: create a warrant with proposed_id
    r1 = _apply(cik, ticker, "0000-00-0001", "424B5", "2024-03-15", [
        {"kind": "create_instrument", "type": "warrant",
         "proposed_id": "W-001",
         "counterparty_canonical": "Aegis",
         "terms": {"strike": 2.50, "term_years": 5, "units": "common"},
         "outstanding": {"count": 2_000_000},
         "event_date": "2024-03-15"},
    ])
    assert r1.accepted == 1 and r1.created_ids == ["W-001"], r1

    # Filing 2 — 8-K announcing repricing
    r2 = _apply(cik, ticker, "0000-00-0002", "8-K", "2024-08-02", [
        {"kind": "amend_instrument", "_type": "warrant",
         "instrument_id": "W-001",
         "field_updates": {"strike": 1.75},
         "event_date": "2024-08-02"},
    ])
    assert r2.accepted == 1 and r2.rejected == 0, r2
    w = get_instrument(cik, "W-001")
    assert w["terms"]["strike"] == 1.75, w["terms"]

    # Filing 3 — proxy + 8-K announcing 1-for-10 reverse split.
    # apply_split sorts to first regardless of declared order.
    r3 = _apply(cik, ticker, "0000-00-0003", "8-K", "2024-08-15", [
        {"kind": "create_instrument", "type": "convertible",
         "proposed_id": "C-001",
         "terms": {"principal": 5_000_000, "rate": 0.07, "conv_price": 1.00,
                   "units": "common"},
         "outstanding": {"principal_remaining": 5_000_000},
         "event_date": "2024-08-15"},
        {"kind": "apply_split", "post": 1, "pre": 10, "direction": "reverse",
         "effective_date": "2024-08-15"},
    ])
    assert r3.accepted == 2 and r3.splits_applied == 1, r3
    w = get_instrument(cik, "W-001")
    # W-001 had strike $1.75, count 2M. Reverse 1-for-10 → strike $17.50, count 200K.
    assert abs(w["terms"]["strike"] - 17.5) < 1e-6, w["terms"]
    assert abs(w["outstanding"]["count"] - 200_000) < 1e-6, w["outstanding"]
    c = get_instrument(cik, "C-001")
    assert abs(c["terms"]["conv_price"] - 1.00) < 1e-6, c["terms"]

    # Filing 4 — re-applying the SAME split (e.g. via the next 10-Q
    # re-disclosing the split) should be idempotent.
    r4 = _apply(cik, ticker, "0000-00-0004", "10-Q", "2024-09-30", [
        {"kind": "apply_split", "post": 1, "pre": 10, "direction": "reverse",
         "effective_date": "2024-08-15"},
    ])
    assert r4.accepted == 1, r4
    w = get_instrument(cik, "W-001")
    assert abs(w["terms"]["strike"] - 17.5) < 1e-6, ("split not idempotent",
                                                      w["terms"])

    # Filing 5 — exercise + drawdown + invalid transitions
    r5 = _apply(cik, ticker, "0000-00-0005", "8-K", "2025-02-10", [
        # warrant exercise — partially consumes count
        {"kind": "record_event", "instrument_id": "W-001",
         "event_kind": "exercise",
         "fields": {"shares": 80_000, "price": 17.5,
                    "gross_proceeds": 1_400_000},
         "event_date": "2025-02-10"},
        # ATM created + draw — chained inside one filing
        {"kind": "create_instrument", "type": "atm",
         "proposed_id": "ATM-001",
         "placement_agent_canonical": "H.C. Wainwright",
         "terms": {"capacity_usd": 25_000_000,
                   "agreement_date": "2025-02-09"},
         "outstanding": {"remaining_capacity_usd": 25_000_000,
                         "drawn_usd": 0},
         "event_date": "2025-02-10"},
        {"kind": "record_event", "instrument_id": "ATM-001",
         "event_kind": "drawdown",
         "fields": {"drawdown_amount_usd": 5_600_000,
                    "drawdown_shares": 320_000},
         "event_date": "2025-02-10"},
        # type mismatch — drawdown on a warrant should be REJECTED
        {"kind": "record_event", "instrument_id": "W-001",
         "event_kind": "drawdown",
         "fields": {"drawdown_amount_usd": 100_000,
                    "drawdown_shares": 1},
         "event_date": "2025-02-10"},
        # missing id — REJECTED
        {"kind": "amend_instrument", "_type": "warrant",
         "instrument_id": "W-999",
         "field_updates": {"strike": 1.0},
         "event_date": "2025-02-10"},
    ])
    assert r5.accepted == 3, ("expected 3 accepted, got", r5)
    assert r5.rejected == 2, ("expected 2 rejected, got", r5)
    assert r5.drawdowns_recorded == 1, r5
    w = get_instrument(cik, "W-001")
    assert abs(w["outstanding"]["count"] - 120_000) < 1e-6, w["outstanding"]
    atm = get_instrument(cik, "ATM-001")
    assert abs(atm["outstanding"]["drawn_usd"] - 5_600_000) < 1e-6, atm
    assert abs(atm["outstanding"]["remaining_capacity_usd"] - 19_400_000) < 1e-6, atm

    # Verify rejections were persisted
    with get_conn() as conn:
        errs = conn.execute(
            "SELECT error_kind, COUNT(*) c FROM dilution_walk_errors "
            "WHERE cik=? GROUP BY error_kind ORDER BY error_kind",
            (cik,),
        ).fetchall()
    err_kinds = {row["error_kind"]: row["c"] for row in errs}
    assert err_kinds.get("missing_id") == 1, err_kinds
    assert err_kinds.get("type_mismatch") == 1, err_kinds

    # Verify drawdown index
    with get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) c FROM dilution_ledger_drawdowns WHERE cik=?",
            (cik,),
        ).fetchone()["c"]
    assert n == 1, ("drawdown not indexed", n)

    # Filing 6 — close W-001 (expired) in a standalone filing.
    r6 = _apply(cik, ticker, "0000-00-0006", "10-K", "2029-03-31", [
        {"kind": "close_instrument", "instrument_id": "W-001",
         "reason": "expired", "event_date": "2029-03-15"},
    ])
    assert r6.accepted == 1, r6
    w = get_instrument(cik, "W-001")
    assert w["status"] == "expired", w["status"]

    # Filing 6b — subsequent attempt to record an event on the closed
    # warrant in a LATER filing should be rejected as illegal_transition.
    r6b = _apply(cik, ticker, "0000-00-0006b", "8-K", "2029-04-01", [
        {"kind": "record_event", "instrument_id": "W-001",
         "event_kind": "exercise",
         "fields": {"shares": 1},
         "event_date": "2029-04-01"},
    ])
    assert r6b.accepted == 0 and r6b.rejected == 1, r6b

    # Filing 7 — superseded chain.
    r7 = _apply(cik, ticker, "0000-00-0007", "8-K", "2025-06-01", [
        {"kind": "create_instrument", "type": "convertible",
         "proposed_id": "C-002",
         "terms": {"principal": 5_500_000, "rate": 0.10, "conv_price": 0.50},
         "outstanding": {"principal_remaining": 5_500_000},
         "event_date": "2025-06-01"},
        {"kind": "close_instrument", "instrument_id": "C-001",
         "reason": "superseded", "replaced_by": "C-002",
         "event_date": "2025-06-01"},
    ])
    assert r7.accepted == 2, r7
    c1 = get_instrument(cik, "C-001")
    assert c1["status"] == "superseded:C-002", c1["status"]

    open_rows = get_open_instruments(cik)
    open_ids = {r["instrument_id"] for r in open_rows}
    assert "ATM-001" in open_ids and "C-002" in open_ids, open_ids

    # Filing 8 — equity (off-shelf PIPE) closings → drawdown rows.
    r8 = _apply(cik, ticker, "0000-00-0008", "8-K", "2025-07-01", [
        # 8a: signed-AND-closed in one filing — books cash at create.
        {"kind": "create_instrument", "type": "equity",
         "proposed_id": "EQ-001",
         "counterparty_canonical": "Everrise",
         "terms": {"price_per_share": 2.0, "closing_date": "2025-07-01"},
         "outstanding": {"count": 1_000_000},
         "event_date": "2025-07-01"},
        # 8b: stock-for-services (price 0) — closing stated, no cash.
        {"kind": "create_instrument", "type": "equity",
         "proposed_id": "EQ-002",
         "terms": {"price_per_share": 0.0, "closing_date": "2025-07-01"},
         "outstanding": {"count": 50_000},
         "event_date": "2025-07-01"},
        # 8c: signed-but-pending SPA — no closing_date, no cash booked.
        {"kind": "create_instrument", "type": "equity",
         "proposed_id": "EQ-003",
         "counterparty_canonical": "Hudson Bay",
         "terms": {"price_per_share": 1.5},
         "outstanding": {"count": 2_000_000},
         "event_date": "2025-07-01"},
    ])
    assert r8.accepted == 3 and r8.rejected == 0, r8
    assert r8.drawdowns_recorded == 1, r8
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT instrument_id, amount_usd, shares, "
            "       drawdown_party_canonical, drawdown_party_role "
            "FROM dilution_ledger_drawdowns "
            "WHERE cik=? AND instrument_id LIKE 'EQ-%'",
            (cik,),
        ).fetchall()
    assert len(rows) == 1 and rows[0]["instrument_id"] == "EQ-001", \
        [dict(r) for r in rows]
    assert abs(rows[0]["amount_usd"] - 2_000_000) < 1e-6, rows[0]["amount_usd"]
    assert rows[0]["drawdown_party_canonical"] == "Everrise", dict(rows[0])
    assert rows[0]["drawdown_party_role"] == "investor", dict(rows[0])

    # Filing 9 — confirm_closing on equity: ACCEPTED (validate gate),
    # books the pending PIPE's cash; re-confirming the already-booked
    # EQ-001 must NOT double-book (single-drawdown guard). The card
    # label must NOT be relabeled by closing month.
    eq1_label_before = get_instrument(cik, "EQ-001")["label"]
    r9 = _apply(cik, ticker, "0000-00-0009", "8-K", "2025-08-15", [
        {"kind": "record_event", "instrument_id": "EQ-003",
         "event_kind": "closing",
         "fields": {"gross_proceeds_usd": 3_100_000},
         "event_date": "2025-08-15"},
        {"kind": "record_event", "instrument_id": "EQ-001",
         "event_kind": "closing",
         "fields": {},
         "event_date": "2025-08-20"},
    ])
    assert r9.accepted == 2 and r9.rejected == 0, r9
    assert r9.drawdowns_recorded == 1, r9
    eq3 = get_instrument(cik, "EQ-003")
    assert eq3["terms"].get("closing_date") == "2025-08-15", eq3["terms"]
    assert get_instrument(cik, "EQ-001")["label"] == eq1_label_before, \
        "equity card must not relabel on closing"
    with get_conn() as conn:
        per_inst = {
            r["instrument_id"]: (r["c"], r["amt"])
            for r in conn.execute(
                "SELECT instrument_id, COUNT(*) c, SUM(amount_usd) amt "
                "FROM dilution_ledger_drawdowns "
                "WHERE cik=? AND instrument_id LIKE 'EQ-%' "
                "GROUP BY instrument_id",
                (cik,),
            )
        }
    assert per_inst.get("EQ-001") == (1, 2_000_000.0), per_inst
    assert per_inst.get("EQ-003") == (1, 3_100_000.0), per_inst
    assert "EQ-002" not in per_inst, per_inst

    # Cleanup
    os.unlink(_tmp.name)
    print("ALL OK — accepted=%d rejected=%d created=%s drawdowns=%d splits=%d"
          % (r1.accepted + r2.accepted + r3.accepted + r4.accepted
             + r5.accepted + r6.accepted + r6b.accepted + r7.accepted,
             r1.rejected + r2.rejected + r3.rejected + r4.rejected
             + r5.rejected + r6.rejected + r6b.rejected + r7.rejected,
             r1.created_ids + r3.created_ids + r5.created_ids + r7.created_ids,
             r5.drawdowns_recorded,
             r3.splits_applied + r4.splits_applied))


if __name__ == "__main__":
    main()
