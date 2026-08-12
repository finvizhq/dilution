"""Retired-debt lifecycle close + the two flow-accumulator clamps.

A convertible sitting at ``principal_remaining=0`` while still ``active``
is a lifecycle lie: the cap table says the note is live, the balance says
it is gone. Before this, only the projection layer noticed — ``cards.
_convertible_dead`` drops dust rows — so the ledger state stayed wrong and
every non-card reader inherited it (5 such rows in production: NUAI C-129
plus CETY C-1138/1139/1143/1155).

Three pieces, all unit-tested here:
  - ``store.close_retired_debt`` — the deterministic close, which fires
    ONLY when the retired-to-date flow accounts for the face. Closing on a
    zero balance alone would close rows the anchor believes are live, and
    the anchor wins the next round (CETY C-1143 oscillates
    closed → reopened → amended-back-up five times over).
  - the ``principal_converted_to_date`` clamp — the field has no anchor to
    reconcile it, and an aggregate multi-note conversion figure attributed
    to EACH note overflows it (NUAI C-123/C-124: $6.12M each against a
    combined $10M of face). The clamp is what makes it safe for
    ``close_retired_debt`` to read as a full-retirement signal.
  - the ``principal_redeemed_to_date`` accumulator — previously never
    written for convertibles, so a note repaid in CASH reached zero with
    no flow record anywhere and could never be corroborated.
"""

import json
from datetime import date

from dilution.ledger.mutations import RecordConversion, RecordPartialRedemption
from dilution.ledger.store import apply_mutations, close_retired_debt


CIK = 2028336  # NUAI


def _out(temp_db, iid):
    row = temp_db.execute(
        "SELECT outstanding_json FROM dilution_ledger WHERE instrument_id=?",
        (iid,))[0]
    return json.loads(row["outstanding_json"])


def _status(temp_db, iid):
    return temp_db.execute(
        "SELECT status FROM dilution_ledger WHERE instrument_id=?",
        (iid,))[0]["status"]


def _add_note(temp_db, iid, *, face, remaining, converted=None, redeemed=None,
              status="active", type="convertible"):
    out = {"principal_remaining": remaining}
    if converted is not None:
        out["principal_converted_to_date"] = converted
    if redeemed is not None:
        out["principal_redeemed_to_date"] = redeemed
    temp_db.add_instrument(
        iid, cik=CIK, type=type, status=status, created_at="2025-01-16",
        terms_json=json.dumps({"principal": face, "conv_price": 10.0}),
        outstanding_json=json.dumps(out),
    )


class TestCloseRetiredDebt:
    def test_closes_fully_converted_note(self, temp_db):
        temp_db.add_company(CIK, "NUAI")
        # NUAI C-129: $5,000,000 face, converted exactly 5,000,000.
        _add_note(temp_db, "C-129", face=5_000_000.0, remaining=0.0,
                  converted=5_000_000.0)
        closed = close_retired_debt(
            CIK, accession="acc-1", form="10-K", filing_date="2026-03-13")
        assert closed == ["C-129:converted"]
        assert _status(temp_db, "C-129") == "converted"

    def test_closes_fully_redeemed_note_as_redeemed(self, temp_db):
        temp_db.add_company(CIK, "NUAI")
        _add_note(temp_db, "C-200", face=700_000.0, remaining=0.0,
                  redeemed=700_000.0)
        closed = close_retired_debt(
            CIK, accession="acc-1", form="10-K", filing_date="2026-03-13")
        assert closed == ["C-200:redeemed"]
        assert _status(temp_db, "C-200") == "redeemed"

    def test_mixed_flow_closes_on_dominant_leg(self, temp_db):
        temp_db.add_company(CIK, "NUAI")
        _add_note(temp_db, "C-201", face=1_000_000.0, remaining=0.0,
                  converted=300_000.0, redeemed=700_000.0)
        closed = close_retired_debt(
            CIK, accession="acc-1", form="10-K", filing_date="2026-03-13")
        assert closed == ["C-201:redeemed"]

    def test_zero_balance_with_unaccounted_flow_left_active(self, temp_db):
        """The CETY class: a periodic amend zeroed the balance but no
        conversion or redemption was ever recorded. The balance is the
        suspect, not the status — closing here fights the anchor."""
        temp_db.add_company(CIK, "CETY")
        _add_note(temp_db, "C-1138", face=3_600_000.0, remaining=0.0)
        closed = close_retired_debt(
            CIK, accession="acc-1", form="10-K", filing_date="2026-03-13")
        assert closed == []
        assert _status(temp_db, "C-1138") == "active"

    def test_partial_flow_left_active(self, temp_db):
        temp_db.add_company(CIK, "NUAI")
        # Half the face retired — a zero balance is not yet explained.
        _add_note(temp_db, "C-202", face=1_000_000.0, remaining=0.0,
                  converted=500_000.0)
        closed = close_retired_debt(
            CIK, accession="acc-1", form="10-K", filing_date="2026-03-13")
        assert closed == []
        assert _status(temp_db, "C-202") == "active"

    def test_flow_within_tolerance_closes(self, temp_db):
        temp_db.add_company(CIK, "NUAI")
        # 99.5% of face retired — filing rounding, still a full retirement.
        _add_note(temp_db, "C-203", face=1_000_000.0, remaining=0.0,
                  converted=995_000.0)
        closed = close_retired_debt(
            CIK, accession="acc-1", form="10-K", filing_date="2026-03-13")
        assert closed == ["C-203:converted"]

    def test_live_balance_untouched(self, temp_db):
        temp_db.add_company(CIK, "NUAI")
        _add_note(temp_db, "C-204", face=1_000_000.0, remaining=400_000.0,
                  converted=600_000.0)
        closed = close_retired_debt(
            CIK, accession="acc-1", form="10-K", filing_date="2026-03-13")
        assert closed == []
        assert _status(temp_db, "C-204") == "active"

    def test_relative_dust_counts_as_retired(self, temp_db):
        temp_db.add_company(CIK, "NUAI")
        # $2,000 residual on a $10M note is 0.02% — dust by the relative
        # rule even though it clears the $1,000 absolute floor.
        _add_note(temp_db, "C-205", face=10_000_000.0, remaining=2_000.0,
                  converted=10_000_000.0)
        closed = close_retired_debt(
            CIK, accession="acc-1", form="10-K", filing_date="2026-03-13")
        assert closed == ["C-205:converted"]

    def test_idempotent(self, temp_db):
        temp_db.add_company(CIK, "NUAI")
        _add_note(temp_db, "C-129", face=5_000_000.0, remaining=0.0,
                  converted=5_000_000.0)
        close_retired_debt(
            CIK, accession="acc-1", form="10-K", filing_date="2026-03-13")
        again = close_retired_debt(
            CIK, accession="acc-2", form="10-Q", filing_date="2026-05-15")
        assert again == []

    def test_only_touches_convertibles(self, temp_db):
        temp_db.add_company(CIK, "NUAI")
        _add_note(temp_db, "P-300", face=5_000_000.0, remaining=0.0,
                  converted=5_000_000.0, type="preferred")
        closed = close_retired_debt(
            CIK, accession="acc-1", form="10-K", filing_date="2026-03-13")
        assert closed == []
        assert _status(temp_db, "P-300") == "active"

    def test_skips_row_without_face(self, temp_db):
        temp_db.add_company(CIK, "NUAI")
        temp_db.add_instrument(
            "C-206", cik=CIK, type="convertible", status="active",
            created_at="2025-01-16", terms_json="{}",
            outstanding_json='{"principal_remaining": 0.0}')
        assert close_retired_debt(
            CIK, accession="acc-1", form="10-K",
            filing_date="2026-03-13") == []

    def test_skips_row_without_balance(self, temp_db):
        temp_db.add_company(CIK, "NUAI")
        temp_db.add_instrument(
            "C-207", cik=CIK, type="convertible", status="active",
            created_at="2025-01-16",
            terms_json='{"principal": 1000000.0}', outstanding_json="{}")
        assert close_retired_debt(
            CIK, accession="acc-1", form="10-K",
            filing_date="2026-03-13") == []


class TestConvertedToDateClamp:
    def test_aggregate_overflow_clamped_to_face(self, temp_db):
        """NUAI C-124: a $6,118,243 aggregate conversion figure booked
        against a $3,000,000 note."""
        temp_db.add_company(CIK, "NUAI")
        temp_db.add_filing("acc-1", cik=CIK, form="10-Q",
                           filing_date="2025-11-14")
        _add_note(temp_db, "C-124", face=3_000_000.0, remaining=3_000_000.0)
        apply_mutations(
            cik=CIK, ticker="NUAI", accession="acc-1", form="10-Q",
            filing_date="2025-11-14",
            mutations=[RecordConversion(
                instrument_id="C-124", event_date=date(2025, 9, 30),
                shares_issued=6_125_002.0,
                principal_converted=6_118_243.0)],
        )
        out = _out(temp_db, "C-124")
        assert out["principal_converted_to_date"] == 3_000_000.0

    def test_within_face_not_clamped(self, temp_db):
        temp_db.add_company(CIK, "NUAI")
        temp_db.add_filing("acc-1", cik=CIK, form="10-Q",
                           filing_date="2025-11-14")
        _add_note(temp_db, "C-300", face=3_000_000.0, remaining=3_000_000.0)
        apply_mutations(
            cik=CIK, ticker="NUAI", accession="acc-1", form="10-Q",
            filing_date="2025-11-14",
            mutations=[RecordConversion(
                instrument_id="C-300", event_date=date(2025, 9, 30),
                shares_issued=120_000.0,
                principal_converted=1_200_000.0)],
        )
        assert _out(temp_db, "C-300")["principal_converted_to_date"] == 1_200_000.0

    def test_clamp_records_raw_figure_in_history(self, temp_db):
        temp_db.add_company(CIK, "NUAI")
        temp_db.add_filing("acc-1", cik=CIK, form="10-Q",
                           filing_date="2025-11-14")
        _add_note(temp_db, "C-301", face=1_000_000.0, remaining=1_000_000.0)
        apply_mutations(
            cik=CIK, ticker="NUAI", accession="acc-1", form="10-Q",
            filing_date="2025-11-14",
            mutations=[RecordConversion(
                instrument_id="C-301", event_date=date(2025, 9, 30),
                shares_issued=400_000.0,
                principal_converted=4_000_000.0)],
        )
        hist = json.loads(temp_db.execute(
            "SELECT history_json FROM dilution_ledger WHERE "
            "instrument_id='C-301'")[0]["history_json"])
        clamps = [e for e in hist
                  if "converted_to_date_clamped" in str(e.get("fields_changed"))]
        assert clamps, "clamp marker must be recorded in history"


class TestRedeemedToDateAccumulator:
    def test_partial_redemption_accumulates(self, temp_db):
        temp_db.add_company(CIK, "NUAI")
        temp_db.add_filing("acc-1", cik=CIK, form="8-K",
                           filing_date="2025-11-14")
        _add_note(temp_db, "C-400", face=1_000_000.0, remaining=1_000_000.0)
        apply_mutations(
            cik=CIK, ticker="NUAI", accession="acc-1", form="8-K",
            filing_date="2025-11-14",
            mutations=[RecordPartialRedemption(
                instrument_id="C-400", event_date=date(2025, 10, 1),
                principal_redeemed=250_000.0)],
        )
        out = _out(temp_db, "C-400")
        assert out["principal_redeemed_to_date"] == 250_000.0
        assert out["principal_remaining"] == 750_000.0

    def test_full_cash_repayment_becomes_corroborable(self, temp_db):
        """The gap this closes: a cash-repaid note previously reached zero
        with no flow record, so close_retired_debt could never close it."""
        temp_db.add_company(CIK, "NUAI")
        temp_db.add_filing("acc-1", cik=CIK, form="8-K",
                           filing_date="2025-11-14")
        _add_note(temp_db, "C-401", face=700_000.0, remaining=700_000.0)
        apply_mutations(
            cik=CIK, ticker="NUAI", accession="acc-1", form="8-K",
            filing_date="2025-11-14",
            mutations=[RecordPartialRedemption(
                instrument_id="C-401", event_date=date(2025, 10, 1),
                principal_redeemed=700_000.0)],
        )
        assert close_retired_debt(
            CIK, accession="acc-2", form="10-K",
            filing_date="2026-03-13") == ["C-401:redeemed"]

    def test_overflow_clamped_to_face(self, temp_db):
        temp_db.add_company(CIK, "NUAI")
        temp_db.add_filing("acc-1", cik=CIK, form="8-K",
                           filing_date="2025-11-14")
        _add_note(temp_db, "C-402", face=500_000.0, remaining=500_000.0)
        apply_mutations(
            cik=CIK, ticker="NUAI", accession="acc-1", form="8-K",
            filing_date="2025-11-14",
            mutations=[RecordPartialRedemption(
                instrument_id="C-402", event_date=date(2025, 10, 1),
                principal_redeemed=1_500_000.0)],
        )
        assert _out(temp_db, "C-402")["principal_redeemed_to_date"] == 500_000.0
