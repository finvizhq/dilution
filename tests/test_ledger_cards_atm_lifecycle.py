"""ATM lifecycle render rules (post-re-walk drift triage 2026-06-17).

Two convention rules DT follows for ENDED ATM programs, now mirrored in
atm_cards / _chain_head_terminated / _registered_label:

  1. A program that ENDED — `terminated` status OR an `active` row whose
     sales-agreement term already expired — is hidden, EXCEPT one that
     raised its full capacity before ending (GCTK Dec-2024 Dawson:
     status=terminated, drawn to within rounding of its $8.23M cap, which
     DT still shows). A program that ended with material capacity left was
     abandoned mid-stream and stays hidden.

  2. A fully-drawn terminated ATM that DOES render follows the ATM
     vocabulary {Registered, Replaced} — never the equity-line-only
     'Terminated' label; it falls through to registration inference.

  3. A restated-chain head still flagged `active` whose agreement term has
     expired is a dead program: its superseded predecessors drop with it
     (XTIA Maxim ATM-2678 chain — five leaked extras).
"""

import json
from datetime import date, timedelta

from dilution.ledger.cards import (
    _chain_head_terminated,
    _registered_label,
    atm_cards,
)

PAST = (date.today() - timedelta(days=400)).isoformat()
FUTURE = (date.today() + timedelta(days=400)).isoformat()


def _atm(temp_db, iid, *, status="active", capacity=1_000_000.0,
         agreement_end=None, drawn=0.0, cik=900, form="424B5"):
    acc = f"acc-{iid}"
    temp_db.add_instrument(
        iid, cik=cik, type="atm", status=status, created_accession=acc,
        terms_json=json.dumps({
            "capacity_usd": capacity,
            "agreement_end_date": agreement_end,
            "placement_agent": "Roth",
        }),
        outstanding_json="{}",
    )
    temp_db.add_filing(acc, cik=cik, form=form)
    if drawn:
        temp_db.add_drawdown(iid, cik=cik, event_date="2025-02-01",
                             amount_usd=drawn)
    return iid


def _titles(cards):
    return {c["title"] for c in cards}


def test_terminated_atm_fully_drawn_renders(temp_db):
    temp_db.add_company(900, "T900")
    _atm(temp_db, "ATM-FULL", status="terminated", capacity=1_000_000,
         drawn=1_000_000)
    cards = atm_cards(900)
    assert len(cards) == 1
    assert cards[0]["remaining_capacity"] == 0.0


def test_terminated_atm_with_capacity_left_hidden(temp_db):
    temp_db.add_company(900, "T900")
    _atm(temp_db, "ATM-PART", status="terminated", capacity=1_000_000,
         drawn=400_000)
    assert atm_cards(900) == []


def test_active_atm_expired_agreement_hidden(temp_db):
    # XTIA Maxim ATM-2678: active head, agreement term expired, partly drawn.
    temp_db.add_company(900, "T900")
    _atm(temp_db, "ATM-EXP", status="active", capacity=1_000_000,
         agreement_end=PAST, drawn=400_000)
    assert atm_cards(900) == []


def test_active_atm_expired_agreement_fully_drawn_renders(temp_db):
    temp_db.add_company(900, "T900")
    _atm(temp_db, "ATM-EXPFULL", status="active", capacity=1_000_000,
         agreement_end=PAST, drawn=1_000_000)
    assert len(atm_cards(900)) == 1


def test_active_atm_live_agreement_renders(temp_db):
    temp_db.add_company(900, "T900")
    _atm(temp_db, "ATM-LIVE", status="active", capacity=1_000_000,
         agreement_end=FUTURE, drawn=400_000)
    cards = atm_cards(900)
    assert len(cards) == 1
    assert cards[0]["remaining_capacity"] == 600_000.0


def test_chain_head_terminated_via_expired_agreement(temp_db):
    # Predecessor superseded into a still-`active` head whose agreement
    # term has expired -> the whole program is dead.
    temp_db.add_company(900, "T900")
    _atm(temp_db, "HEAD", status="active", capacity=1_000_000,
         agreement_end=PAST)
    pred = {"instrument_id": "PRED", "status": "superseded:HEAD"}
    assert _chain_head_terminated(pred) is True


def test_chain_head_active_future_agreement_not_terminated(temp_db):
    temp_db.add_company(900, "T900")
    _atm(temp_db, "HEAD2", status="active", capacity=1_000_000,
         agreement_end=FUTURE)
    pred = {"instrument_id": "PRED2", "status": "superseded:HEAD2"}
    assert _chain_head_terminated(pred) is False


def test_registered_label_terminated_atm_falls_through_to_registered():
    r = {"status": "terminated", "type": "atm",
         "history": [{"form": "S-3"}]}
    assert _registered_label(r) == "Registered"


def test_registered_label_terminated_equity_line_stays_terminated():
    r = {"status": "terminated", "type": "equity_line", "history": []}
    assert _registered_label(r) == "Terminated"
