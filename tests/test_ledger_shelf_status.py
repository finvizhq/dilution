"""Unit tests for ``dilution/ledger/shelf_status.py``.

Covers the two testable units:

* ``_add_days`` — a pure ISO-date arithmetic helper with a
  ``'9999-12-31'`` sentinel on malformed/None/empty input.
* ``derive_shelf_status`` — a DB-backed status machine that joins
  ``dilution_ledger.type='shelf'`` rows to ``dilution_filings``
  EFFECT/RW notices (via SEC ``file_number``) and derives
  ``derived_status ∈ {active, registered, withdrawn, expired}`` while
  applying the 90-day EFFECT window and the Rule 415(a)(5) 3-year
  (1095 calendar day) sunset (ASR-exempt).

Determinism: every ``derive_shelf_status`` test passes ``today=`` as a
``datetime.date`` object (the function does ``(today or _d.today())``
then ``.isoformat()`` — a string ``today`` would raise, so a ``date``
is required).

DB staging uses the autouse ``temp_db`` fixture from conftest (it
reroutes ``db.get_conn`` to a throwaway SQLite file with the full
production schema). NOTE: ``dilution_ledger.terms_json`` /
``outstanding_json`` / ``status`` are NOT NULL in the schema, and
``instrument_id`` / ``accession_number`` are globally unique, so each
test uses unique ids and never stages a true NULL json (only the
empty-string ``''`` path is reachable through the schema).
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from dilution.ledger.shelf_status import (
    _add_days,
    derive_shelf_status,
    EFFECT_WINDOW_DAYS,
    SHELF_LIFE_YEARS,
)


# ── helpers ──────────────────────────────────────────────────────────
def _stage_shelf(
    temp_db,
    *,
    cik: int,
    instrument_id: str,
    created_accession: str,
    created_at: str = "2025-01-01",
    terms: dict | None = None,
    outstanding: dict | None = None,
    status: str = "active",
    # filing for the created_accession (omit to exercise the
    # "missing filing" fallback path)
    filing_form: str | None = "S-3",
    filing_date: str | None = "2025-01-01",
    file_number: str | None = "333-DEFAULT",
    ticker: str = "TEST",
) -> None:
    """Stage one shelf instrument plus (optionally) its created filing.

    ``terms``/``outstanding`` default to ``{"capacity_usd": 1_000_000}``
    / ``{}`` so the row survives the capacity filter unless overridden.
    Pass ``filing_form=None`` to skip the created_accession filing row
    entirely (forces the form-from-terms / created_at fallback).
    """
    if terms is None:
        terms = {"capacity_usd": 1_000_000}
    if outstanding is None:
        outstanding = {}
    # Ensure the company exists (FK / visibility hygiene).
    try:
        temp_db.add_company(cik, ticker=ticker)
    except Exception:
        pass  # already added by a prior call in the same test
    if filing_form is not None:
        temp_db.add_filing(
            created_accession,
            cik,
            form=filing_form,
            filing_date=filing_date,
            file_number=file_number,
        )
    temp_db.add_instrument(
        instrument_id,
        cik=cik,
        ticker=ticker,
        type="shelf",
        created_at=created_at,
        created_accession=created_accession,
        terms_json=json.dumps(terms),
        outstanding_json=json.dumps(outstanding),
        status=status,
    )


# ─────────────────────────────────────────────────────────────────────
# _add_days
# ─────────────────────────────────────────────────────────────────────
class TestAddDays:
    def test_normal_positive_add_crosses_month_boundary(self):
        assert _add_days("2025-01-01", 90) == "2025-04-01"

    def test_days_zero_returns_same_date(self):
        assert _add_days("2025-01-01", 0) == "2025-01-01"

    def test_negative_days_goes_backwards(self):
        # 2025-01-01 minus 10 days -> 2024-12-22
        assert _add_days("2025-01-01", -10) == "2024-12-22"

    def test_1095_days_is_calendar_days_not_calendar_years(self):
        # The caller uses 365 * SHELF_LIFE_YEARS = 1095 *calendar* days,
        # which drifts off true 3-calendar-years across leap years.
        assert 365 * SHELF_LIFE_YEARS == 1095
        # 2020 + 1095d lands on 2022-12-31, NOT 2023-01-01 (2020 leap yr).
        assert _add_days("2020-01-01", 1095) == "2022-12-31"
        # A non-leap-spanning base lands cleanly off-by-one earlier too.
        assert _add_days("2022-06-15", 1095) == "2025-06-14"

    def test_leap_day_boundary(self):
        assert _add_days("2024-02-29", 1) == "2024-03-01"

    @pytest.mark.parametrize(
        "bad",
        [
            "not-a-date",
            "",
            "2025-13-01",   # invalid month
            "2025-1-1",     # non-zero-padded -> fromisoformat rejects
            "01/01/2025",   # wrong format
        ],
    )
    def test_malformed_string_returns_sentinel(self, bad):
        assert _add_days(bad, 90) == "9999-12-31"

    def test_none_input_returns_sentinel(self):
        # TypeError path.
        assert _add_days(None, 90) == "9999-12-31"

    def test_window_constant_used_by_caller(self):
        assert EFFECT_WINDOW_DAYS == 90


# ─────────────────────────────────────────────────────────────────────
# derive_shelf_status — basic structure / empties
# ─────────────────────────────────────────────────────────────────────
class TestDeriveBasics:
    def test_no_shelf_rows_returns_empty_list(self, temp_db):
        temp_db.add_company(9999, ticker="EMPTY")
        assert derive_shelf_status(9999, today=date(2025, 3, 1)) == []

    def test_unknown_cik_returns_empty_list(self, temp_db):
        assert derive_shelf_status(424242, today=date(2025, 3, 1)) == []

    def test_non_shelf_type_is_ignored(self, temp_db):
        temp_db.add_company(2200, ticker="WARR")
        temp_db.add_filing("t1", 2200, form="S-3",
                           filing_date="2025-01-01", file_number="F22")
        temp_db.add_instrument(
            "w-2200", cik=2200, type="warrant",
            created_at="2025-01-01", created_accession="t1",
            terms_json=json.dumps({"capacity_usd": 1_000_000}),
        )
        assert derive_shelf_status(2200, today=date(2025, 3, 1)) == []

    def test_reported_status_passes_through_untouched(self, temp_db):
        _stage_shelf(temp_db, cik=2100, instrument_id="i-2100",
                     created_accession="s-2100", file_number="F21",
                     status="terminated")
        out = derive_shelf_status(2100, today=date(2025, 3, 1))
        assert len(out) == 1
        assert out[0]["reported_status"] == "terminated"
        assert out[0]["instrument_id"] == "i-2100"

    def test_output_dict_shape(self, temp_db):
        _stage_shelf(temp_db, cik=10, instrument_id="i-10",
                     created_accession="a-10", file_number="333-10")
        row = derive_shelf_status(10, today=date(2025, 3, 1))[0]
        assert set(row) == {
            "instrument_id", "accession_number", "form", "file_number",
            "filing_date", "effect_date", "withdrawal_date",
            "expiration_date", "anticipated_amount_usd", "reported_status",
            "derived_status",
        }


# ─────────────────────────────────────────────────────────────────────
# derive_shelf_status — ASR (auto-effective) shelves
# ─────────────────────────────────────────────────────────────────────
class TestASRShelves:
    def test_s3asr_is_active_effect_equals_filing_no_expiration(self, temp_db):
        _stage_shelf(temp_db, cik=100, instrument_id="i-100",
                     created_accession="a-100", filing_form="S-3ASR",
                     filing_date="2018-01-01", created_at="2018-01-01",
                     file_number="333-1",
                     terms={"capacity_usd": 5_000_000})
        # today far in the future — an ASR can never expire.
        row = derive_shelf_status(100, today=date(2026, 1, 1))[0]
        assert row["form"] == "S-3ASR"
        assert row["derived_status"] == "active"
        assert row["effect_date"] == "2018-01-01"   # == filing_date
        assert row["expiration_date"] is None
        assert row["anticipated_amount_usd"] == 5_000_000

    def test_f3asr_is_active_asr(self, temp_db):
        _stage_shelf(temp_db, cik=1500, instrument_id="i-1500",
                     created_accession="f-1500", filing_form="F-3ASR",
                     filing_date="2018-01-01", created_at="2018-01-01",
                     file_number="F3")
        row = derive_shelf_status(1500, today=date(2026, 1, 1))[0]
        assert row["derived_status"] == "active"
        assert row["expiration_date"] is None
        assert row["effect_date"] == "2018-01-01"

    def test_asr_never_expires_even_when_ancient(self, temp_db):
        # Filed in 2005 — would be expired many times over if non-ASR.
        _stage_shelf(temp_db, cik=101, instrument_id="i-101",
                     created_accession="a-101", filing_form="S-3ASR",
                     filing_date="2005-01-01", created_at="2005-01-01",
                     file_number="333-101")
        row = derive_shelf_status(101, today=date(2026, 1, 1))[0]
        assert row["derived_status"] == "active"
        assert row["expiration_date"] is None


# ─────────────────────────────────────────────────────────────────────
# derive_shelf_status — EFFECT join by file_number
# ─────────────────────────────────────────────────────────────────────
class TestEffectByFileNumber:
    def test_matching_effect_makes_active_with_effect_date(self, temp_db):
        _stage_shelf(temp_db, cik=200, instrument_id="i-200",
                     created_accession="a-200", filing_date="2025-01-01",
                     file_number="333-2")
        temp_db.add_filing("e-200", 200, form="EFFECT",
                           filing_date="2025-02-01", file_number="333-2")
        row = derive_shelf_status(200, today=date(2025, 3, 1))[0]
        assert row["derived_status"] == "active"
        assert row["effect_date"] == "2025-02-01"
        # expiration is effect_date + 1095 days. Independently derived
        # literal (2025-02-01 + 1095d = 2028-02-01), cross-checked against
        # the helper so a regression in _add_days also surfaces here.
        assert row["expiration_date"] == "2028-02-01"
        assert row["expiration_date"] == _add_days("2025-02-01", 1095)

    def test_no_effect_within_window_is_registered(self, temp_db):
        _stage_shelf(temp_db, cik=201, instrument_id="i-201",
                     created_accession="a-201", filing_date="2025-01-01",
                     file_number="333-201")
        row = derive_shelf_status(201, today=date(2025, 3, 1))[0]
        assert row["derived_status"] == "registered"
        assert row["effect_date"] is None
        # falls back to filing_date + 1095 for expiration. Independently
        # derived literal (2025-01-01 + 1095d = 2028-01-01).
        assert row["expiration_date"] == "2028-01-01"
        assert row["expiration_date"] == _add_days("2025-01-01", 1095)

    def test_effect_under_different_file_number_does_not_match(self, temp_db):
        _stage_shelf(temp_db, cik=202, instrument_id="i-202",
                     created_accession="a-202", filing_date="2025-01-01",
                     file_number="333-202")
        # EFFECT exists but under an unrelated file number.
        temp_db.add_filing("e-202", 202, form="EFFECT",
                           filing_date="2025-02-01", file_number="333-OTHER")
        row = derive_shelf_status(202, today=date(2025, 3, 1))[0]
        assert row["derived_status"] == "registered"
        assert row["effect_date"] is None

    def test_multiple_effects_same_file_number_earliest_wins(self, temp_db):
        _stage_shelf(temp_db, cik=1100, instrument_id="i-1100",
                     created_accession="m-1100", filing_date="2025-01-01",
                     file_number="333-X")
        # Insert the later EFFECT first to prove ordering, not insert order.
        temp_db.add_filing("me1", 1100, form="EFFECT",
                           filing_date="2025-03-01", file_number="333-X")
        temp_db.add_filing("me2", 1100, form="EFFECT",
                           filing_date="2025-02-01", file_number="333-X")
        row = derive_shelf_status(1100, today=date(2025, 4, 1))[0]
        assert row["effect_date"] == "2025-02-01"

    def test_file_number_effect_join_ignores_90_day_window(self, temp_db):
        # When the shelf HAS a file_number, the EFFECT join is purely by
        # file_number — the 90-day date window only governs the *fallback*
        # path (file_number is None). So an EFFECT under the same
        # file_number that lands 200 days after filing STILL makes the
        # shelf active. This distinguishes the two join branches.
        _stage_shelf(temp_db, cik=5000, instrument_id="i-5000",
                     created_accession="a-5000", filing_date="2025-01-01",
                     file_number="333-5000")
        temp_db.add_filing("e-5000", 5000, form="EFFECT",
                           filing_date="2025-07-20",  # ~200 days later
                           file_number="333-5000")
        row = derive_shelf_status(5000, today=date(2025, 8, 1))[0]
        assert row["derived_status"] == "active"
        assert row["effect_date"] == "2025-07-20"
        # And expiration is measured from that (late) EFFECT, not filing.
        assert row["expiration_date"] == _add_days("2025-07-20", 1095)

    def test_file_number_effect_before_filing_still_makes_active(self, temp_db):
        # The date window (which would reject a pre-filing EFFECT) does NOT
        # apply on the file_number path: an EFFECT dated BEFORE filing but
        # carrying the same file_number is still honored as the effect_date.
        _stage_shelf(temp_db, cik=5001, instrument_id="i-5001",
                     created_accession="a-5001", filing_date="2025-01-01",
                     file_number="333-5001")
        temp_db.add_filing("e-5001", 5001, form="EFFECT",
                           filing_date="2024-12-01",  # before filing
                           file_number="333-5001")
        row = derive_shelf_status(5001, today=date(2025, 3, 1))[0]
        assert row["derived_status"] == "active"
        assert row["effect_date"] == "2024-12-01"

    def test_effect_prefix_match_eg_effectiveness(self, temp_db):
        # Form is matched by LIKE 'EFFECT%' — any EFFECT* form counts.
        _stage_shelf(temp_db, cik=203, instrument_id="i-203",
                     created_accession="a-203", filing_date="2025-01-01",
                     file_number="333-203")
        temp_db.add_filing("e-203", 203, form="EFFECTIVENESS",
                           filing_date="2025-02-01", file_number="333-203")
        row = derive_shelf_status(203, today=date(2025, 3, 1))[0]
        assert row["derived_status"] == "active"
        assert row["effect_date"] == "2025-02-01"


# ─────────────────────────────────────────────────────────────────────
# derive_shelf_status — EFFECT date-window fallback (no file_number)
# ─────────────────────────────────────────────────────────────────────
class TestEffectWindowFallback:
    def test_effect_inside_window_at_upper_bound_is_active(self, temp_db):
        # filing 2025-01-01, window_end = +90 = 2025-04-01 (inclusive).
        _stage_shelf(temp_db, cik=600, instrument_id="i-600",
                     created_accession="a-600", filing_date="2025-01-01",
                     file_number=None)
        temp_db.add_filing("e-600", 600, form="EFFECT",
                           filing_date="2025-04-01", file_number=None)
        row = derive_shelf_status(600, today=date(2025, 5, 1))[0]
        assert row["derived_status"] == "active"
        assert row["effect_date"] == "2025-04-01"

    def test_effect_exactly_on_filing_date_lower_bound_is_active(self, temp_db):
        _stage_shelf(temp_db, cik=1900, instrument_id="i-1900",
                     created_accession="a-1900", filing_date="2025-01-01",
                     file_number=None)
        temp_db.add_filing("e-1900", 1900, form="EFFECT",
                           filing_date="2025-01-01", file_number=None)
        row = derive_shelf_status(1900, today=date(2025, 3, 1))[0]
        assert row["derived_status"] == "active"
        assert row["effect_date"] == "2025-01-01"

    def test_effect_outside_window_day_91_is_registered(self, temp_db):
        # day 91 = 2025-04-02 is past window_end 2025-04-01.
        _stage_shelf(temp_db, cik=700, instrument_id="i-700",
                     created_accession="a-700", filing_date="2025-01-01",
                     file_number=None)
        temp_db.add_filing("e-700", 700, form="EFFECT",
                           filing_date="2025-04-02", file_number=None)
        row = derive_shelf_status(700, today=date(2025, 5, 1))[0]
        assert row["derived_status"] == "registered"
        assert row["effect_date"] is None

    def test_effect_before_filing_date_is_ignored(self, temp_db):
        # EFFECT predates filing -> outside [filing_date, +90] -> registered.
        _stage_shelf(temp_db, cik=701, instrument_id="i-701",
                     created_accession="a-701", filing_date="2025-01-01",
                     file_number=None)
        temp_db.add_filing("e-701", 701, form="EFFECT",
                           filing_date="2024-12-31", file_number=None)
        row = derive_shelf_status(701, today=date(2025, 3, 1))[0]
        assert row["derived_status"] == "registered"
        assert row["effect_date"] is None

    def test_first_in_window_wins_given_ordered_effect_dates(self, temp_db):
        # Two EFFECTs both inside the window: effects are ordered by
        # filing_date, so the earliest one inside the window is picked.
        _stage_shelf(temp_db, cik=2000, instrument_id="i-2000",
                     created_accession="a-2000", filing_date="2025-01-01",
                     file_number=None)
        temp_db.add_filing("qe1", 2000, form="EFFECT",
                           filing_date="2025-02-15", file_number=None)
        temp_db.add_filing("qe2", 2000, form="EFFECT",
                           filing_date="2025-02-01", file_number=None)
        row = derive_shelf_status(2000, today=date(2025, 4, 1))[0]
        assert row["effect_date"] == "2025-02-01"


# ─────────────────────────────────────────────────────────────────────
# derive_shelf_status — expiration (Rule 415(a)(5) sunset)
# ─────────────────────────────────────────────────────────────────────
class TestExpiration:
    def test_expiration_date_is_filing_plus_1095_when_no_effect(self, temp_db):
        _stage_shelf(temp_db, cik=300, instrument_id="i-300",
                     created_accession="a-300", filing_date="2020-01-01",
                     file_number="333-3")
        row = derive_shelf_status(300, today=date(2020, 6, 1))[0]
        assert row["expiration_date"] == "2022-12-31"  # 2020-01-01 + 1095

    def test_exactly_at_boundary_is_not_expired(self, temp_db):
        # expiration uses strict '<' so today == expiration is NOT expired.
        _stage_shelf(temp_db, cik=301, instrument_id="i-301",
                     created_accession="a-301", filing_date="2020-01-01",
                     file_number="333-301")
        # expiration_date == 2022-12-31; today == that day.
        row = derive_shelf_status(301, today=date(2022, 12, 31))[0]
        assert row["expiration_date"] == "2022-12-31"
        assert row["derived_status"] == "registered"

    def test_one_day_past_boundary_is_expired(self, temp_db):
        _stage_shelf(temp_db, cik=302, instrument_id="i-302",
                     created_accession="a-302", filing_date="2020-01-01",
                     file_number="333-302")
        row = derive_shelf_status(302, today=date(2023, 1, 1))[0]
        assert row["derived_status"] == "expired"

    def test_within_window_is_registered_not_expired(self, temp_db):
        _stage_shelf(temp_db, cik=303, instrument_id="i-303",
                     created_accession="a-303", filing_date="2025-01-01",
                     file_number="333-303")
        row = derive_shelf_status(303, today=date(2025, 6, 1))[0]
        assert row["derived_status"] == "registered"

    def test_expiration_measured_from_effect_date_when_present(self, temp_db):
        # Effect arrives later than filing -> expiration extends from it.
        _stage_shelf(temp_db, cik=304, instrument_id="i-304",
                     created_accession="a-304", filing_date="2020-01-01",
                     file_number="333-304")
        temp_db.add_filing("e-304", 304, form="EFFECT",
                           filing_date="2020-06-01", file_number="333-304")
        row = derive_shelf_status(304, today=date(2026, 1, 1))[0]
        assert row["effect_date"] == "2020-06-01"
        # expiration = 2020-06-01 + 1095 calendar days = 2023-06-01
        # (independently re-derived; well before the 2026 today).
        assert row["expiration_date"] == "2023-06-01"
        assert row["expiration_date"] == _add_days("2020-06-01", 1095)
        assert row["derived_status"] == "expired"


# ─────────────────────────────────────────────────────────────────────
# derive_shelf_status — withdrawn (RW) and precedence
# ─────────────────────────────────────────────────────────────────────
class TestWithdrawn:
    def test_rw_under_same_file_number_makes_withdrawn(self, temp_db):
        _stage_shelf(temp_db, cik=401, instrument_id="i-401",
                     created_accession="a-401", filing_date="2025-01-01",
                     file_number="333-401")
        temp_db.add_filing("rw-401", 401, form="RW",
                           filing_date="2025-06-01", file_number="333-401")
        row = derive_shelf_status(401, today=date(2025, 7, 1))[0]
        assert row["derived_status"] == "withdrawn"
        assert row["withdrawal_date"] == "2025-06-01"

    def test_withdrawn_takes_precedence_over_expired(self, temp_db):
        # Old enough to expire AND has a valid RW -> withdrawn wins.
        _stage_shelf(temp_db, cik=400, instrument_id="i-400",
                     created_accession="a-400", filing_date="2020-01-01",
                     file_number="333-4")
        temp_db.add_filing("rw-400", 400, form="RW",
                           filing_date="2021-01-01", file_number="333-4")
        row = derive_shelf_status(400, today=date(2026, 1, 1))[0]
        assert row["derived_status"] == "withdrawn"
        assert row["withdrawal_date"] == "2021-01-01"

    def test_rw_filed_before_filing_date_does_not_withdraw(self, temp_db):
        # withdrawal_date < filing_date fails the >= guard.
        _stage_shelf(temp_db, cik=500, instrument_id="i-500",
                     created_accession="a-500", filing_date="2020-06-01",
                     file_number="333-5")
        temp_db.add_filing("rw-500", 500, form="RW",
                           filing_date="2020-01-01", file_number="333-5")
        row = derive_shelf_status(500, today=date(2020, 7, 1))[0]
        assert row["derived_status"] == "registered"
        # FIXED (was bug B#7): an un-honored RW (predating filing) is no longer
        # echoed — withdrawal_date is present iff derived_status == 'withdrawn'.
        assert row["withdrawal_date"] is None

    def test_rw_on_exactly_filing_date_does_withdraw(self, temp_db):
        # withdrawal_date == filing_date satisfies '>=' (inclusive).
        _stage_shelf(temp_db, cik=501, instrument_id="i-501",
                     created_accession="a-501", filing_date="2025-01-01",
                     file_number="333-501")
        temp_db.add_filing("rw-501", 501, form="RW",
                           filing_date="2025-01-01", file_number="333-501")
        row = derive_shelf_status(501, today=date(2025, 3, 1))[0]
        assert row["derived_status"] == "withdrawn"

    def test_multiple_rws_same_file_number_earliest_wins(self, temp_db):
        _stage_shelf(temp_db, cik=1200, instrument_id="i-1200",
                     created_accession="w-1200", filing_date="2020-01-01",
                     file_number="333-Y")
        temp_db.add_filing("wr1", 1200, form="RW",
                           filing_date="2021-06-01", file_number="333-Y")
        temp_db.add_filing("wr2", 1200, form="RW",
                           filing_date="2021-02-01", file_number="333-Y")
        row = derive_shelf_status(1200, today=date(2026, 1, 1))[0]
        assert row["withdrawal_date"] == "2021-02-01"

    def test_shelf_without_file_number_ignores_rw(self, temp_db):
        # file_number=None on the shelf -> RW cannot be matched.
        _stage_shelf(temp_db, cik=1800, instrument_id="i-1800",
                     created_accession="n-1800", filing_date="2025-01-01",
                     file_number=None)
        temp_db.add_filing("nr-1800", 1800, form="RW",
                           filing_date="2025-02-01", file_number="333-N")
        row = derive_shelf_status(1800, today=date(2025, 3, 1))[0]
        assert row["derived_status"] == "registered"
        assert row["withdrawal_date"] is None

    def test_rw_without_file_number_is_skipped_by_query(self, temp_db):
        # RW with NULL file_number is excluded by the SQL filter, so even
        # a shelf with a file_number sees no withdrawal.
        _stage_shelf(temp_db, cik=502, instrument_id="i-502",
                     created_accession="a-502", filing_date="2025-01-01",
                     file_number="333-502")
        temp_db.add_filing("rw-502", 502, form="RW",
                           filing_date="2025-06-01", file_number=None)
        row = derive_shelf_status(502, today=date(2025, 7, 1))[0]
        assert row["derived_status"] == "registered"
        assert row["withdrawal_date"] is None


# ─────────────────────────────────────────────────────────────────────
# derive_shelf_status — capacity filtering
# ─────────────────────────────────────────────────────────────────────
class TestCapacityFiltering:
    def test_negative_remaining_capacity_drops_row(self, temp_db):
        _stage_shelf(temp_db, cik=820, instrument_id="i-820",
                     created_accession="a-820", file_number="333-820",
                     terms={}, outstanding={"remaining_capacity_usd": -5})
        assert derive_shelf_status(820, today=date(2025, 3, 1)) == []

    def test_negative_remaining_with_terms_capacity_still_drops(self, temp_db):
        # remaining=-5 is truthy in the `or`, so it wins over terms.
        _stage_shelf(temp_db, cik=830, instrument_id="i-830",
                     created_accession="a-830", file_number="333-830",
                     terms={"capacity_usd": 1_000_000},
                     outstanding={"remaining_capacity_usd": -5})
        assert derive_shelf_status(830, today=date(2025, 3, 1)) == []

    def test_terms_capacity_zero_only_drops_row(self, temp_db):
        # capacity = (None or 0) = 0; 0 is not None and 0 <= 0 -> dropped.
        _stage_shelf(temp_db, cik=850, instrument_id="i-850",
                     created_accession="a-850", file_number="333-850",
                     terms={"capacity_usd": 0}, outstanding={})
        assert derive_shelf_status(850, today=date(2025, 3, 1)) == []

    def test_capacity_absent_both_keys_keeps_row(self, temp_db):
        # capacity is None -> the `<= 0` guard is skipped -> kept.
        _stage_shelf(temp_db, cik=840, instrument_id="i-840",
                     created_accession="a-840", file_number="333-840",
                     terms={}, outstanding={})
        out = derive_shelf_status(840, today=date(2025, 3, 1))
        assert len(out) == 1
        assert out[0]["anticipated_amount_usd"] is None

    def test_positive_remaining_capacity_keeps_row(self, temp_db):
        _stage_shelf(temp_db, cik=860, instrument_id="i-860",
                     created_accession="a-860", file_number="333-860",
                     terms={"capacity_usd": 5_000_000},
                     outstanding={"remaining_capacity_usd": 2_000_000})
        out = derive_shelf_status(860, today=date(2025, 3, 1))
        assert len(out) == 1

    def test_negative_remaining_takes_precedence_over_positive_terms(
            self, temp_db):
        # Precedence: a truthy `remaining_capacity_usd` wins the `or`, so a
        # NEGATIVE remaining drops the row even though terms.capacity_usd is
        # a healthy positive figure. (The complementary 0-remaining case is
        # the documented BUG below.)
        _stage_shelf(temp_db, cik=870, instrument_id="i-870",
                     created_accession="a-870", file_number="333-870",
                     terms={"capacity_usd": 9_999_999},
                     outstanding={"remaining_capacity_usd": -1})
        assert derive_shelf_status(870, today=date(2025, 3, 1)) == []

    # FIXED (was bug B#7): a fully-drawn shelf (remaining_capacity_usd == 0)
    # is now dropped. remaining is authoritative when present, so a literal 0
    # no longer falls through to the (positive) registered terms.capacity_usd.
    def test_zero_remaining_drops_row(self, temp_db):
        _stage_shelf(temp_db, cik=801, instrument_id="i-801",
                     created_accession="a-801", file_number="333-801",
                     terms={"capacity_usd": 5_000_000},
                     outstanding={"remaining_capacity_usd": 0})
        assert derive_shelf_status(801, today=date(2025, 3, 1)) == []

    def test_zero_remaining_no_terms_capacity_drops_row(self, temp_db):
        # FIXED (was bug B#7): remaining=0 is authoritative even with no
        # terms.capacity_usd — a fully-drawn shelf is dropped consistently.
        _stage_shelf(temp_db, cik=810, instrument_id="i-810",
                     created_accession="a-810", file_number="333-810",
                     terms={}, outstanding={"remaining_capacity_usd": 0})
        assert derive_shelf_status(810, today=date(2025, 3, 1)) == []


# ─────────────────────────────────────────────────────────────────────
# derive_shelf_status — form filtering
# ─────────────────────────────────────────────────────────────────────
class TestFormFiltering:
    @pytest.mark.parametrize("bad_form", ["424B5", "S-1", "8-K", "10-K", "6-K"])
    def test_non_shelf_form_is_skipped(self, temp_db, bad_form):
        cik = 900 + hash(bad_form) % 1000
        _stage_shelf(temp_db, cik=cik, instrument_id=f"i-{bad_form}",
                     created_accession=f"a-{bad_form}", filing_form=bad_form,
                     file_number="333-X")
        assert derive_shelf_status(cik, today=date(2025, 3, 1)) == []

    @pytest.mark.parametrize("good_form", ["S-3", "F-3", "S-3ASR", "F-3ASR",
                                           "s-3", "f-3"])
    def test_shelf_prefix_forms_are_kept(self, temp_db, good_form):
        cik = 9100 + hash(good_form) % 800
        _stage_shelf(temp_db, cik=cik, instrument_id=f"i-{good_form}",
                     created_accession=f"a-{good_form}", filing_form=good_form,
                     filing_date="2025-01-01", created_at="2025-01-01",
                     file_number="333-G")
        out = derive_shelf_status(cik, today=date(2025, 3, 1))
        assert len(out) == 1
        assert out[0]["form"] == good_form.upper()

    def test_form_from_terms_when_filing_missing(self, temp_db):
        # No created_accession filing row -> form falls back to terms.form,
        # filing_date falls back to created_at, file_number is None.
        _stage_shelf(temp_db, cik=1000, instrument_id="i-1000",
                     created_accession="MISSING", filing_form=None,
                     created_at="2025-01-01",
                     terms={"capacity_usd": 1_000_000, "form": "S-3"})
        row = derive_shelf_status(1000, today=date(2025, 3, 1))[0]
        assert row["form"] == "S-3"
        assert row["filing_date"] == "2025-01-01"
        assert row["file_number"] is None
        # No file number + no EFFECT in window -> registered.
        assert row["derived_status"] == "registered"

    def test_empty_form_everywhere_is_skipped(self, temp_db):
        # Filing missing AND terms has no form -> form == '' -> skipped.
        _stage_shelf(temp_db, cik=1001, instrument_id="i-1001",
                     created_accession="GONE", filing_form=None,
                     terms={"capacity_usd": 1_000_000})
        assert derive_shelf_status(1001, today=date(2025, 3, 1)) == []

    def test_filing_form_wins_over_terms_form(self, temp_db):
        # Filing says S-3 (kept); terms.form says 424B5 (would be skipped).
        _stage_shelf(temp_db, cik=1002, instrument_id="i-1002",
                     created_accession="a-1002", filing_form="S-3",
                     file_number="333-1002",
                     terms={"capacity_usd": 1_000_000, "form": "424B5"})
        out = derive_shelf_status(1002, today=date(2025, 3, 1))
        assert len(out) == 1
        assert out[0]["form"] == "S-3"


# ─────────────────────────────────────────────────────────────────────
# derive_shelf_status — JSON robustness, ordering, anticipated amount
# ─────────────────────────────────────────────────────────────────────
class TestMiscellaneous:
    def test_empty_string_json_does_not_crash(self, temp_db):
        # terms_json/outstanding_json are NOT NULL in the schema, but the
        # empty-string '' path exercises json.loads('' or '{}').
        temp_db.add_company(1300, ticker="EMPTYJSON")
        temp_db.add_filing("z1", 1300, form="S-3",
                           filing_date="2025-01-01", file_number="333-Z")
        temp_db.add_instrument(
            "i-1300", cik=1300, type="shelf", created_at="2025-01-01",
            created_accession="z1", terms_json="", outstanding_json="",
            status="active",
        )
        row = derive_shelf_status(1300, today=date(2025, 3, 1))[0]
        # form comes from the FILING (S-3), not terms.
        assert row["form"] == "S-3"
        assert row["derived_status"] == "registered"
        assert row["anticipated_amount_usd"] is None

    def test_results_ordered_by_created_at(self, temp_db):
        temp_db.add_company(1400, ticker="ORDER")
        temp_db.add_filing("o1", 1400, form="S-3",
                           filing_date="2025-01-01", file_number="F1")
        temp_db.add_filing("o2", 1400, form="S-3",
                           filing_date="2025-01-01", file_number="F2")
        # Insert LATER first; ORDER BY created_at must still front EARLIER.
        temp_db.add_instrument(
            "LATER", cik=1400, type="shelf", created_at="2025-06-01",
            created_accession="o2",
            terms_json=json.dumps({"capacity_usd": 1_000_000}),
        )
        temp_db.add_instrument(
            "EARLIER", cik=1400, type="shelf", created_at="2025-01-01",
            created_accession="o1",
            terms_json=json.dumps({"capacity_usd": 1_000_000}),
        )
        ids = [r["instrument_id"]
               for r in derive_shelf_status(1400, today=date(2025, 7, 1))]
        assert ids == ["EARLIER", "LATER"]

    def test_anticipated_amount_reflects_terms_not_remaining(self, temp_db):
        # anticipated_amount_usd == terms.capacity_usd, even though the
        # capacity FILTER used outstanding.remaining_capacity_usd.
        _stage_shelf(temp_db, cik=1600, instrument_id="i-1600",
                     created_accession="a-1600", file_number="333-1600",
                     terms={"capacity_usd": 5_000_000},
                     outstanding={"remaining_capacity_usd": 2_000_000})
        row = derive_shelf_status(1600, today=date(2025, 3, 1))[0]
        assert row["anticipated_amount_usd"] == 5_000_000  # terms, not 2e6

    def test_today_none_defaults_to_real_today_no_crash(self, temp_db):
        # We can't assert the boundary deterministically, but a recent
        # filing is comfortably inside its 3-year window, so default today
        # yields 'registered' (no EFFECT). Just confirm no crash + sane val.
        _stage_shelf(temp_db, cik=1700, instrument_id="i-1700",
                     created_accession="a-1700",
                     filing_date="2025-06-01", created_at="2025-06-01",
                     file_number="333-1700")
        out = derive_shelf_status(1700)  # today=None
        assert len(out) == 1
        assert out[0]["derived_status"] in {"registered", "active",
                                            "withdrawn", "expired"}

    def test_today_as_string_is_accepted(self, temp_db):
        # FIXED (mirror of bug A#4 in s1_status): a string `today` is now
        # normalized via _coerce_date instead of raising AttributeError on
        # (today or _d.today()).isoformat(). It yields the same result as the
        # equivalent date.
        _stage_shelf(temp_db, cik=1750, instrument_id="i-1750",
                     created_accession="a-1750", filing_date="2025-01-01",
                     file_number="333-1750")
        as_str = derive_shelf_status(1750, today="2025-03-01")
        as_date = derive_shelf_status(1750, today=date(2025, 3, 1))
        assert as_str == as_date
        assert as_str[0]["derived_status"] == "registered"

    def test_timestamp_anchor_does_not_collapse_expiration(self, temp_db):
        # FIXED (mirror of bug A#4): a timestamp-bearing filing_date used as
        # the expiration anchor (via _add_days) is trimmed to its date head
        # instead of failing fromisoformat and collapsing to the 9999
        # never-expires sentinel. A >3-year-old non-ASR shelf correctly
        # reads 'expired'.
        _stage_shelf(temp_db, cik=1751, instrument_id="i-1751",
                     created_accession="a-1751", filing_form="S-3",
                     filing_date="2018-01-01T00:00:00Z",
                     created_at="2018-01-01T00:00:00Z", file_number="333-1751")
        out = derive_shelf_status(1751, today=date(2026, 1, 1))
        assert out[0]["derived_status"] == "expired"
        assert out[0]["expiration_date"] != "9999-12-31"

    def test_two_shelves_independent_statuses(self, temp_db):
        # One active (ASR), one expired non-ASR, under one cik.
        temp_db.add_company(2300, ticker="MULTI")
        temp_db.add_filing("asr", 2300, form="S-3ASR",
                           filing_date="2024-01-01", file_number="333-A")
        temp_db.add_filing("old", 2300, form="S-3",
                           filing_date="2018-01-01", file_number="333-B")
        temp_db.add_instrument(
            "shelf-asr", cik=2300, type="shelf", created_at="2024-01-01",
            created_accession="asr",
            terms_json=json.dumps({"capacity_usd": 1_000_000}),
        )
        temp_db.add_instrument(
            "shelf-old", cik=2300, type="shelf", created_at="2018-01-01",
            created_accession="old",
            terms_json=json.dumps({"capacity_usd": 1_000_000}),
        )
        out = {r["instrument_id"]: r["derived_status"]
               for r in derive_shelf_status(2300, today=date(2026, 1, 1))}
        assert out["shelf-asr"] == "active"
        assert out["shelf-old"] == "expired"
