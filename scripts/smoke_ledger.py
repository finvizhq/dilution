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
from dilution.ledger.mutations import MutationList  # noqa: E402
from dilution.ledger.store import (  # noqa: E402
    apply_mutations,
    get_instrument,
    get_open_instruments,
    reset_walk_state,
)
from db import get_conn  # noqa: E402


def _apply(cik, ticker, accession, form, date, payload):
    parsed = MutationList.model_validate({"mutations": payload})
    return apply_mutations(
        cik=cik, ticker=ticker, accession=accession,
        form=form, filing_date=date, mutations=parsed.mutations,
    )


def main():
    init_dilution_db()
    cik, ticker = 9999, "TEST"
    reset_walk_state(cik)

    # Filing 1 — IPO 424B: create a warrant with proposed_id
    r1 = _apply(cik, ticker, "0000-00-0001", "424B5", "2024-03-15", [
        {"kind": "create_instrument", "type": "warrant",
         "proposed_id": "W-001",
         "counterparty": "Aegis Capital", "counterparty_canonical": "Aegis",
         "terms": {"strike": 2.50, "term_years": 5, "units": "common"},
         "outstanding": {"count": 2_000_000},
         "event_date": "2024-03-15",
         "snippet": "2,000,000 warrants @ $2.50, 5-yr term"},
    ])
    assert r1.accepted == 1 and r1.created_ids == ["W-001"], r1

    # Filing 2 — 8-K announcing repricing
    r2 = _apply(cik, ticker, "0000-00-0002", "8-K", "2024-08-02", [
        {"kind": "amend_instrument", "instrument_id": "W-001",
         "field_updates": {"strike": 1.75},
         "event_date": "2024-08-02",
         "snippet": "warrant strike repriced to $1.75"},
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
         "snippet": "$5M senior convertible @ $1.00 conv price (post-split)"},
        {"kind": "apply_split", "ratio": 0.1, "direction": "reverse",
         "effective_date": "2024-08-15",
         "snippet": "1-for-10 reverse stock split effective Aug 15"},
    ])
    assert r3.accepted == 2 and r3.splits_applied == 1, r3
    w = get_instrument(cik, "W-001")
    # W-001 had strike $1.75, count 2M. Reverse 1-for-10 → strike $17.50, count 200K.
    assert abs(w["terms"]["strike"] - 17.5) < 1e-6, w["terms"]
    assert abs(w["outstanding"]["count"] - 200_000) < 1e-6, w["outstanding"]
    # C-001 was created in the SAME filing as the split, but split sorts FIRST.
    # The convertible was created with conv_price=$1.00 already in post-split
    # terms (per the snippet) so the split doesn't re-touch it BECAUSE
    # apply_split runs before _apply_create. Sanity-check.
    c = get_instrument(cik, "C-001")
    assert abs(c["terms"]["conv_price"] - 1.00) < 1e-6, c["terms"]

    # Filing 4 — re-applying the SAME split (e.g. via the next 10-Q
    # re-disclosing the split) should be idempotent.
    r4 = _apply(cik, ticker, "0000-00-0004", "10-Q", "2024-09-30", [
        {"kind": "apply_split", "ratio": 0.1, "direction": "reverse",
         "effective_date": "2024-08-15",
         "snippet": "reverse split disclosed again in Q1 10-Q"},
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
         "event_date": "2025-02-10",
         "snippet": "80,000 warrants exercised @ $17.50"},
        # ATM created + draw — chained inside one filing
        {"kind": "create_instrument", "type": "atm",
         "proposed_id": "ATM-001",
         "counterparty": "H.C. Wainwright",
         "terms": {"capacity_usd": 25_000_000},
         "outstanding": {"remaining_capacity_usd": 25_000_000,
                         "drawn_usd": 0},
         "snippet": "$25M ATM established"},
        {"kind": "record_event", "instrument_id": "ATM-001",
         "event_kind": "drawdown",
         "fields": {"drawdown_amount_usd": 5_600_000,
                    "drawdown_shares": 320_000, "avg_price": 17.5},
         "event_date": "2025-02-10",
         "snippet": "$5.6M drawn under ATM"},
        # type mismatch — drawdown on a warrant should be REJECTED
        {"kind": "record_event", "instrument_id": "W-001",
         "event_kind": "drawdown",
         "fields": {"drawdown_amount_usd": 100_000},
         "event_date": "2025-02-10",
         "snippet": "(invalid) drawdown on warrant"},
        # missing id — REJECTED
        {"kind": "amend_instrument", "instrument_id": "W-999",
         "field_updates": {"strike": 1.0},
         "snippet": "(invalid) amend on missing id"},
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
         "reason": "expired", "event_date": "2029-03-15",
         "snippet": "warrants expired at 5-yr term"},
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
         "event_date": "2029-04-01",
         "snippet": "(invalid) exercise after expiry"},
    ])
    assert r6b.accepted == 0 and r6b.rejected == 1, r6b

    # Filing 7 — superseded chain. Create W-002 then close W-001 with
    # superseded:W-002. (W-001 already expired here so try a fresh
    # exchange offer scenario: create C-002 superseding C-001.)
    r7 = _apply(cik, ticker, "0000-00-0007", "8-K", "2025-06-01", [
        {"kind": "create_instrument", "type": "convertible",
         "proposed_id": "C-002",
         "terms": {"principal": 5_500_000, "rate": 0.10, "conv_price": 0.50},
         "outstanding": {"principal_remaining": 5_500_000},
         "snippet": "exchange of C-001 for $5.5M new note"},
        {"kind": "close_instrument", "instrument_id": "C-001",
         "reason": "superseded", "replaced_by": "C-002",
         "event_date": "2025-06-01",
         "snippet": "C-001 exchanged for C-002"},
    ])
    assert r7.accepted == 2, r7
    c1 = get_instrument(cik, "C-001")
    assert c1["status"] == "superseded:C-002", c1["status"]

    # Open instruments query — W-001 expired (recent), C-001 superseded
    # (recent), W-001/W-002 not present, ATM-001 active, C-002 active.
    open_rows = get_open_instruments(cik)
    open_ids = {r["instrument_id"] for r in open_rows}
    assert "ATM-001" in open_ids and "C-002" in open_ids, open_ids
    # Recent closed is included by include_recent_closed_days=180.
    # But note today's date affects this — the test ran with
    # arbitrary filing dates so don't assert on closed visibility here.

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
