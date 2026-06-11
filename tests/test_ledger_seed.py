"""Unit tests for dilution/ledger/seed.py — initial-ledger seeding.

Covers the three testable units called out in the survey slice:

  * ``_summarize_row``  — PURE one-line log formatter with a fixed key
    order and a deliberate "0 is not skipped" boundary.
  * ``find_seed_filing`` — DB-backed earliest-periodic-filing picker that
    INNER-JOINs dilution_raw (so a filing with no raw text is invisible).
  * ``seed_ledger``     — async orchestrator (cases A/B/C). The LLM
    extraction is lazy-imported from ``dilution.ledger._overhang_extract``;
    we monkeypatch that module attribute (NOT ``seed.extract_overhang_rows``,
    which does not exist at module level).

No network / LLM / vendor call is ever made: ``extract_overhang_rows`` is
always stubbed before ``seed_ledger`` is invoked. The autouse ``temp_db``
fixture (conftest.py) reroutes every ``get_conn()`` to a throwaway SQLite
DB, so ``find_seed_filing`` and the rejection read-back run against real
tables in isolation.
"""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from dilution.ledger import seed
from dilution.ledger.mutations import CreateWarrant
from dilution.ledger.seed import (
    SeedSummary,
    _PERIODIC_FORMS,
    _summarize_row,
    find_seed_filing,
)


# ─────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────

def _stage_filing(temp_db, *, cik, acc, form, filing_date,
                  report_date=None, with_raw=True, ticker="TEST"):
    """Stage a company (idempotent-ish), a filing, and optionally a
    matching dilution_raw row (required for find_seed_filing's INNER
    JOIN to surface the filing)."""
    # add_company once per cik — ignore duplicate-PK errors so callers
    # can stage several filings for the same issuer.
    try:
        temp_db.add_company(cik, ticker)
    except Exception:
        pass
    temp_db.add_filing(acc, cik, form=form, filing_date=filing_date,
                       report_date=report_date)
    if with_raw:
        temp_db.execute(
            """INSERT INTO dilution_raw
                 (accession_number, doc_name, doc_type, content_md,
                  downloaded_at)
               VALUES (?, ?, ?, ?, ?)""",
            (acc, "primary.htm", form, "body text", "2026-01-01T00:00:00Z"),
        )


class _FakeResult:
    """Stand-in for store.ApplyResult — only .accepted / .rejected are
    read by seed_ledger."""

    def __init__(self, accepted=0, rejected=0):
        self.accepted = accepted
        self.rejected = rejected


def _patch_extract(monkeypatch, rows):
    """Replace the LAZY-imported extract_overhang_rows with an async stub
    returning ``rows``. Must patch the module attribute, since seed_ledger
    does ``from ._overhang_extract import extract_overhang_rows`` inside
    the function body."""
    import dilution.ledger._overhang_extract as oe

    async def _fake(**kwargs):
        return rows

    monkeypatch.setattr(oe, "extract_overhang_rows", _fake)


def _patch_extract_capture(monkeypatch, rows):
    """Like ``_patch_extract`` but ALSO captures the kwargs seed_ledger
    hands the extractor, so a test can assert the LLM-seam contract
    (accession/form/filing_date/report_date/cik/client/unit_ctx) rather
    than just that *something* was awaited. Returns the dict that the
    stub fills in on call."""
    import dilution.ledger._overhang_extract as oe

    captured: dict = {}

    async def _fake(**kwargs):
        captured.update(kwargs)
        return rows

    monkeypatch.setattr(oe, "extract_overhang_rows", _fake)
    return captured


# ─────────────────────────────────────────────────────────────────────
# _summarize_row  (PURE)
# ─────────────────────────────────────────────────────────────────────

class TestSummarizeRow:
    def test_empty_dict_returns_bare_question_mark(self):
        assert _summarize_row({}) == "?"

    def test_category_none_falls_back_to_question_mark(self):
        assert _summarize_row({"category": None}) == "?"

    def test_category_empty_string_falls_back_to_question_mark(self):
        # '' is falsy -> `r.get("category") or "?"` yields '?'.
        assert _summarize_row({"category": ""}) == "?"

    def test_category_present_used_verbatim(self):
        assert _summarize_row({"category": "warrant"}) == "warrant"

    def test_instrument_name_uses_repr_quoting(self):
        # name is emitted via !r, so a str value is quoted.
        out = _summarize_row({"category": "warrant", "instrument_name": "ABC"})
        assert out == "warrant name='ABC'"

    def test_instrument_name_empty_string_omitted(self):
        # `if r.get("instrument_name")` is falsy for '' -> no name= bit.
        assert _summarize_row({"category": "warrant",
                               "instrument_name": ""}) == "warrant"

    def test_none_valued_key_is_omitted(self):
        out = _summarize_row({"category": "shelf", "file_number": None})
        assert out == "shelf"

    def test_empty_string_valued_key_is_omitted(self):
        # '' is explicitly in the (None, '') skip set.
        out = _summarize_row({"category": "atm", "sales_agent": ""})
        assert out == "atm"

    def test_zero_outstanding_count_is_included(self):
        # BOUNDARY: 0 is NOT in (None, '') so a literal 0 IS surfaced.
        out = _summarize_row({"category": "atm", "outstanding_count": 0})
        assert out == "atm outstanding_count=0"

    def test_zero_strike_is_included(self):
        out = _summarize_row({"category": "warrant",
                              "strike_or_conversion_price": 0})
        assert out == "warrant strike_or_conversion_price=0"

    def test_identity_keys_fixed_order_regardless_of_insertion(self):
        # Stage identity keys in REVERSE of source order; output must
        # follow the source-coded order: series_letter, sales_agent,
        # investor, file_number, form.
        r = {
            "form": "S-3",
            "file_number": "333-1",
            "investor": "Acme",
            "sales_agent": "Wainwright",
            "series_letter": "D",
            "category": "preferred",
        }
        out = _summarize_row(r)
        assert out == ("preferred series_letter=D sales_agent=Wainwright "
                       "investor=Acme file_number=333-1 form=S-3")

    def test_date_keys_fixed_order(self):
        r = {
            "category": "warrant",
            "maturity_or_expiry": "2030-01-01",
            "effect_date": "2021-01-01",
            "agreement_date": "2020-06-01",
            "issue_date": "2020-01-01",
        }
        out = _summarize_row(r)
        assert out == ("warrant issue_date=2020-01-01 "
                       "agreement_date=2020-06-01 effect_date=2021-01-01 "
                       "maturity_or_expiry=2030-01-01")

    def test_sizing_keys_fixed_order(self):
        r = {
            "category": "convertible",
            "remaining_capacity_usd": 6,
            "total_capacity_usd": 5,
            "principal_amount": 4,
            "strike_or_conversion_price": 3,
            "common_shares_issuable": 2,
            "outstanding_count": 1,
        }
        out = _summarize_row(r)
        assert out == ("convertible outstanding_count=1 "
                       "common_shares_issuable=2 "
                       "strike_or_conversion_price=3 principal_amount=4 "
                       "total_capacity_usd=5 remaining_capacity_usd=6")

    def test_pre_funded_flag_appended_when_truthy(self):
        out = _summarize_row({"category": "warrant", "is_pre_funded": True})
        assert out == "warrant pre_funded=1"

    def test_pre_funded_flag_omitted_when_falsy(self):
        out = _summarize_row({"category": "warrant", "is_pre_funded": False})
        assert out == "warrant"

    def test_pre_funded_flag_omitted_when_absent(self):
        out = _summarize_row({"category": "warrant"})
        assert out == "warrant"

    def test_terminated_flag_appended_when_truthy(self):
        out = _summarize_row({"category": "atm", "is_terminated": 1})
        assert out == "atm terminated=1"

    def test_both_flags_pre_funded_before_terminated(self):
        out = _summarize_row({"category": "warrant",
                              "is_pre_funded": True, "is_terminated": True})
        assert out == "warrant pre_funded=1 terminated=1"

    def test_non_string_sizing_value_coerced_via_str(self):
        # float 1.5 is joined via str(b) in the final join.
        out = _summarize_row({"category": "convertible",
                              "principal_amount": 1.5})
        assert out == "convertible principal_amount=1.5"

    def test_full_row_end_to_end_ordering(self):
        # Out-of-order full row exercising every section together.
        r = {
            "principal_amount": 1.5,
            "category": "preferred",
            "is_terminated": True,
            "outstanding_count": 0,
            "instrument_name": "ABC",
            "series_letter": "D",
            "issue_date": "2020-01-01",
            "strike_or_conversion_price": 2.5,
            "is_pre_funded": True,
            "sales_agent": "Wainwright",
        }
        out = _summarize_row(r)
        assert out == (
            "preferred name='ABC' series_letter=D sales_agent=Wainwright "
            "issue_date=2020-01-01 outstanding_count=0 "
            "strike_or_conversion_price=2.5 principal_amount=1.5 "
            "pre_funded=1 terminated=1"
        )


# ─────────────────────────────────────────────────────────────────────
# _PERIODIC_FORMS constant
# ─────────────────────────────────────────────────────────────────────

class TestPeriodicFormsConstant:
    def test_contains_core_periodic_and_amended_and_fpi_forms(self):
        for form in ("10-K", "10-K/A", "10-Q", "10-Q/A",
                     "20-F", "20-F/A", "40-F", "40-F/A"):
            assert form in _PERIODIC_FORMS

    def test_excludes_non_periodic_forms(self):
        for form in ("8-K", "S-1", "424B5", "6-K", "S-3"):
            assert form not in _PERIODIC_FORMS


# ─────────────────────────────────────────────────────────────────────
# find_seed_filing  (DB-backed)
# ─────────────────────────────────────────────────────────────────────

class TestFindSeedFiling:
    def test_no_filings_returns_none(self, temp_db):
        temp_db.add_company(99, "T")
        assert find_seed_filing(99, "2000-01-01") is None

    def test_filing_without_raw_row_is_excluded(self, temp_db):
        # INNER JOIN on dilution_raw: a periodic filing with no raw row
        # must NOT be returned.
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2024-01-01", with_raw=False)
        assert find_seed_filing(99, "2000-01-01") is None

    def test_filing_with_raw_row_is_returned(self, temp_db):
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2024-01-01", report_date="2023-12-31")
        row = find_seed_filing(99, "2000-01-01")
        assert row is not None
        assert row["accession_number"] == "A1"
        assert row["form"] == "10-K"
        assert row["filing_date"] == "2024-01-01"
        assert row["report_date"] == "2023-12-31"

    def test_filing_date_before_since_is_excluded(self, temp_db):
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2023-12-31")
        # since strictly after the filing -> excluded.
        assert find_seed_filing(99, "2024-01-01") is None

    def test_filing_date_equal_to_since_is_inclusive(self, temp_db):
        # >= boundary: filing_date == since_date IS included.
        _stage_filing(temp_db, cik=99, acc="A1", form="10-Q",
                      filing_date="2024-04-01")
        row = find_seed_filing(99, "2024-04-01")
        assert row is not None and row["accession_number"] == "A1"

    @pytest.mark.parametrize("form", ["8-K", "S-1", "424B5", "6-K"])
    def test_non_periodic_form_excluded_even_with_raw(self, temp_db, form):
        _stage_filing(temp_db, cik=99, acc="A1", form=form,
                      filing_date="2024-01-01")
        assert find_seed_filing(99, "2000-01-01") is None

    @pytest.mark.parametrize("form", ["10-K/A", "10-Q/A", "20-F/A", "40-F/A"])
    def test_amended_periodic_forms_included(self, temp_db, form):
        _stage_filing(temp_db, cik=99, acc="A1", form=form,
                      filing_date="2024-01-01")
        row = find_seed_filing(99, "2000-01-01")
        assert row is not None and row["form"] == form

    @pytest.mark.parametrize("form", ["20-F", "40-F"])
    def test_fpi_periodic_forms_qualify(self, temp_db, form):
        # FPI forms (relevant for the QTEX FPI fixture work) qualify.
        _stage_filing(temp_db, cik=99, acc="A1", form=form,
                      filing_date="2024-01-01")
        row = find_seed_filing(99, "2000-01-01")
        assert row is not None and row["form"] == form

    def test_earliest_filing_date_wins(self, temp_db):
        # Three qualifying filings; ORDER BY filing_date ASC LIMIT 1 must
        # pick the EARLIEST, not the most recent.
        _stage_filing(temp_db, cik=99, acc="LATE", form="10-Q",
                      filing_date="2024-04-01")
        _stage_filing(temp_db, cik=99, acc="EARLY", form="10-K",
                      filing_date="2024-01-01")
        _stage_filing(temp_db, cik=99, acc="MID", form="10-Q",
                      filing_date="2024-02-15")
        row = find_seed_filing(99, "2000-01-01")
        assert row["accession_number"] == "EARLY"

    def test_wrong_cik_excluded(self, temp_db):
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2024-01-01")
        assert find_seed_filing(1234, "2000-01-01") is None

    def test_report_date_null_present_as_none_key(self, temp_db):
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2024-01-01", report_date=None)
        row = find_seed_filing(99, "2000-01-01")
        assert "report_date" in row and row["report_date"] is None

    def test_items_key_present(self, temp_db):
        # The SELECT carries f.items; absent -> None key.
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2024-01-01")
        row = find_seed_filing(99, "2000-01-01")
        assert "items" in row

    def test_multiple_raw_docs_collapse_to_single_dict(self, temp_db):
        # A multi-document filing has SEVERAL dilution_raw rows for one
        # accession. The INNER JOIN would fan out to N rows, but the
        # query GROUP BYs f.accession_number -> exactly one row, and the
        # outer dict(row) must not choke. (Regression guard: a missing
        # GROUP BY would either dup the seed or raise on .fetchone()
        # silently dropping docs.)
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2024-01-01", report_date="2023-12-31")
        # Add a SECOND raw doc for the same accession.
        temp_db.execute(
            """INSERT INTO dilution_raw
                 (accession_number, doc_name, doc_type, content_md,
                  downloaded_at)
               VALUES (?, ?, ?, ?, ?)""",
            ("A1", "exhibit99.htm", "10-K", "ex text",
             "2026-01-01T00:00:00Z"),
        )
        row = find_seed_filing(99, "2000-01-01")
        assert row is not None
        assert row["accession_number"] == "A1"
        # Sanity: there really were 2 raw rows staged.
        assert len(temp_db.execute(
            "SELECT 1 FROM dilution_raw WHERE accession_number='A1'")) == 2

    def test_returns_plain_dict_not_sqlite_row(self, temp_db):
        # find_seed_filing wraps the Row in dict(...) so downstream
        # .get('report_date') works; assert the type contract.
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2024-01-01")
        row = find_seed_filing(99, "2000-01-01")
        assert type(row) is dict


# ─────────────────────────────────────────────────────────────────────
# seed_ledger  (async, io_mockable)
# ─────────────────────────────────────────────────────────────────────

class TestSeedLedgerCaseB:
    def test_no_periodic_filing_returns_case_b(self, temp_db):
        # No filings staged at all -> find_seed_filing None -> case B.
        # extract_overhang_rows is never reached (no need to stub).
        summary = asyncio.run(seed.seed_ledger(
            cik=4321, ticker="T", since_date="2020-01-01", client=None))
        assert isinstance(summary, SeedSummary)
        assert summary.case == "B_no_periodic"
        assert summary.instruments_created == 0
        assert summary.accession is None
        assert summary.form is None
        assert summary.as_of_date is None

    def test_periodic_filing_without_raw_falls_to_case_b(self, temp_db):
        # A periodic filing exists but no raw row -> find_seed_filing None.
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2024-01-01", with_raw=False)
        summary = asyncio.run(seed.seed_ledger(
            cik=99, ticker="T", since_date="2020-01-01", client=None))
        assert summary.case == "B_no_periodic"


class TestSeedLedgerCaseC:
    def test_empty_extract_returns_case_c(self, temp_db, monkeypatch):
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2024-01-01", report_date="2023-12-31")
        _patch_extract(monkeypatch, [])  # extractor returns nothing
        summary = asyncio.run(seed.seed_ledger(
            cik=99, ticker="T", since_date="2020-01-01", client=None))
        assert summary.case == "C_empty_extract"
        assert summary.instruments_created == 0
        # case C still carries the source filing identity + as_of.
        assert summary.accession == "A1"
        assert summary.form == "10-K"
        assert summary.as_of_date == "2023-12-31"

    def test_case_c_as_of_falls_back_to_filing_date(self, temp_db,
                                                    monkeypatch):
        # report_date NULL -> as_of = filing_date.
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2024-01-01", report_date=None)
        _patch_extract(monkeypatch, [])
        summary = asyncio.run(seed.seed_ledger(
            cik=99, ticker="T", since_date="2020-01-01", client=None))
        assert summary.case == "C_empty_extract"
        assert summary.as_of_date == "2024-01-01"


class TestSeedLedgerExtractorContract:
    """seed_ledger feeds the LLM extractor a specific kwarg contract. The
    extractor is the only LLM/network seam, so pinning what it RECEIVES
    (not just that it is awaited) is the most valuable seam assertion."""

    def test_extractor_called_with_filing_identity_and_client(
            self, temp_db, monkeypatch):
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2024-01-01", report_date="2023-12-31")
        captured = _patch_extract_capture(monkeypatch, [])
        client = object()  # opaque sentinel — must pass through untouched
        unit_ctx = {"shares_in": "thousands"}
        asyncio.run(seed.seed_ledger(
            cik=99, ticker="T", since_date="2020-01-01",
            client=client, unit_ctx=unit_ctx))
        assert captured["accession"] == "A1"
        assert captured["form"] == "10-K"
        assert captured["filing_date"] == "2024-01-01"
        assert captured["cik"] == 99
        # client / unit_ctx are forwarded by identity.
        assert captured["client"] is client
        assert captured["unit_ctx"] is unit_ctx

    def test_extractor_report_date_kwarg_is_resolved_as_of_not_raw(
            self, temp_db, monkeypatch):
        # report_date kwarg = as_of = report_date OR filing_date. With a
        # NULL report_date the extractor must receive the FILING date,
        # proving seed resolves as_of BEFORE the extract call (not the
        # raw NULL).
        _stage_filing(temp_db, cik=99, acc="A1", form="10-Q",
                      filing_date="2024-05-15", report_date=None)
        captured = _patch_extract_capture(monkeypatch, [])
        asyncio.run(seed.seed_ledger(
            cik=99, ticker="T", since_date="2020-01-01", client=None))
        assert captured["report_date"] == "2024-05-15"

    def test_extractor_report_date_kwarg_prefers_report_date(
            self, temp_db, monkeypatch):
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2024-01-01", report_date="2023-12-31")
        captured = _patch_extract_capture(monkeypatch, [])
        asyncio.run(seed.seed_ledger(
            cik=99, ticker="T", since_date="2020-01-01", client=None))
        assert captured["report_date"] == "2023-12-31"


class TestSeedLedgerCaseA:
    def test_real_apply_creates_instrument(self, temp_db, monkeypatch):
        # End-to-end against the REAL store: a single clean preferred row
        # should create one instrument. (A bare warrant would be rejected
        # by the store's periodic_create_missing_terms rule — see the
        # rejection test below — so we use a preferred, which the store
        # accepts with just series_letter + count.)
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2024-01-01", report_date="2023-12-31")
        rows = [{
            "category": "preferred", "series_letter": "D",
            "outstanding_count": 50, "strike_or_conversion_price": 1.0,
            "issue_date": "2022-05-01",
        }]
        _patch_extract(monkeypatch, rows)
        summary = asyncio.run(seed.seed_ledger(
            cik=99, ticker="T", since_date="2020-01-01", client=None))
        assert summary.case == "A_periodic"
        assert summary.accession == "A1"
        assert summary.form == "10-K"
        assert summary.as_of_date == "2023-12-31"
        assert summary.instruments_created == 1
        # Confirm it actually landed in the ledger.
        rows_db = temp_db.execute(
            "SELECT instrument_id, type FROM dilution_ledger WHERE cik=?",
            (99,))
        assert len(rows_db) == 1
        assert rows_db[0]["type"] == "preferred"

    def test_real_apply_case_a_as_of_falls_back_to_filing_date(
            self, temp_db, monkeypatch):
        # Parity with case C: when report_date is NULL, summary.as_of_date
        # falls back to the filing_date even on the success (case A) path.
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2024-01-01", report_date=None)
        rows = [{
            "category": "preferred", "series_letter": "D",
            "outstanding_count": 50, "strike_or_conversion_price": 1.0,
            "issue_date": "2022-05-01",
        }]
        _patch_extract(monkeypatch, rows)
        summary = asyncio.run(seed.seed_ledger(
            cik=99, ticker="T", since_date="2020-01-01", client=None))
        assert summary.case == "A_periodic"
        assert summary.as_of_date == "2024-01-01"
        assert summary.instruments_created == 1

    def test_real_apply_multiple_clean_rows_create_each(
            self, temp_db, monkeypatch):
        # Two distinct, valid preferred series -> the REAL store accepts
        # both; instruments_created == 2 and two ledger rows land. Guards
        # against an accept/len-rows confusion AND a dedup over-collapse
        # (different series must NOT unify).
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2024-01-01", report_date="2023-12-31")
        rows = [
            {"category": "preferred", "series_letter": "C",
             "outstanding_count": 10, "strike_or_conversion_price": 1.0,
             "issue_date": "2021-01-01"},
            {"category": "preferred", "series_letter": "D",
             "outstanding_count": 20, "strike_or_conversion_price": 2.0,
             "issue_date": "2022-01-01"},
        ]
        _patch_extract(monkeypatch, rows)
        summary = asyncio.run(seed.seed_ledger(
            cik=99, ticker="T", since_date="2020-01-01", client=None))
        assert summary.case == "A_periodic"
        assert summary.instruments_created == 2
        types = temp_db.execute(
            "SELECT type FROM dilution_ledger WHERE cik=?", (99,))
        assert [r["type"] for r in types] == ["preferred", "preferred"]

    def test_real_apply_warrant_with_expiry_creates_instrument(
            self, temp_db, monkeypatch):
        # A warrant DOES create when the overhang row carries an explicit
        # maturity_or_expiry (mapped to terms.expiration by
        # _synthesize_create), satisfying the periodic-create rule.
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2024-01-01", report_date="2023-12-31")
        rows = [{
            "category": "warrant", "instrument_name": "Common Warrants",
            "outstanding_count": 1000, "strike_or_conversion_price": 2.5,
            "maturity_or_expiry": "2030-01-01", "issue_date": "2022-05-01",
        }]
        _patch_extract(monkeypatch, rows)
        summary = asyncio.run(seed.seed_ledger(
            cik=99, ticker="T", since_date="2020-01-01", client=None))
        assert summary.instruments_created == 1
        rows_db = temp_db.execute(
            "SELECT type FROM dilution_ledger WHERE cik=?", (99,))
        assert rows_db[0]["type"] == "warrant"

    def test_real_apply_rejects_bare_warrant_and_reads_back_errors(
            self, temp_db, monkeypatch):
        # End-to-end rejection path against the REAL store: a warrant
        # overhang row with NO expiration is rejected by the
        # periodic_create_missing_terms validator. seed_ledger then reads
        # the error back from dilution_walk_errors (rejected>0 branch).
        # instruments_created reflects 0 accepted.
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2024-01-01", report_date="2023-12-31")
        rows = [{
            "category": "warrant", "instrument_name": "Common Warrants",
            "outstanding_count": 1000, "strike_or_conversion_price": 2.5,
            "issue_date": "2022-05-01",
        }]
        _patch_extract(monkeypatch, rows)
        summary = asyncio.run(seed.seed_ledger(
            cik=99, ticker="T", since_date="2020-01-01", client=None))
        assert summary.case == "A_periodic"
        assert summary.instruments_created == 0
        # The store persisted the rejection; seed_ledger's read-back ran.
        errs = temp_db.execute(
            "SELECT error_kind FROM dilution_walk_errors WHERE cik=?", (99,))
        assert len(errs) == 1
        assert errs[0]["error_kind"] == "periodic_create_missing_terms"
        # Ledger stayed empty.
        assert temp_db.execute(
            "SELECT 1 FROM dilution_ledger WHERE cik=?", (99,)) == []

    def test_instruments_created_mirrors_accepted_not_len_rows(
            self, temp_db, monkeypatch):
        # Stub apply_mutations to claim accepted=2, rejected=1 for THREE
        # rows -> instruments_created must be 2 (result.accepted), NOT 3.
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2024-01-01", report_date="2023-12-31")
        rows = [
            {"category": "warrant", "issue_date": "2022-05-01",
             "outstanding_count": 10, "strike_or_conversion_price": 1.0},
            {"category": "warrant", "issue_date": "2022-06-01",
             "outstanding_count": 20, "strike_or_conversion_price": 2.0},
            {"category": "warrant", "issue_date": "2022-07-01",
             "outstanding_count": 30, "strike_or_conversion_price": 3.0},
        ]
        _patch_extract(monkeypatch, rows)
        monkeypatch.setattr(
            seed, "apply_mutations",
            lambda **kw: _FakeResult(accepted=2, rejected=1))
        summary = asyncio.run(seed.seed_ledger(
            cik=99, ticker="T", since_date="2020-01-01", client=None))
        assert summary.case == "A_periodic"
        assert summary.instruments_created == 2

    def test_rejection_readback_does_not_crash(self, temp_db, monkeypatch):
        # rejected>0 triggers a re-query of dilution_walk_errors. With
        # temp_db the table exists (empty) so the read-back is harmless.
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2024-01-01", report_date="2023-12-31")
        rows = [{"category": "warrant", "issue_date": "2022-05-01",
                 "outstanding_count": 10, "strike_or_conversion_price": 1.0}]
        _patch_extract(monkeypatch, rows)
        monkeypatch.setattr(
            seed, "apply_mutations",
            lambda **kw: _FakeResult(accepted=0, rejected=3))
        # Must not raise even though no error rows exist.
        summary = asyncio.run(seed.seed_ledger(
            cik=99, ticker="T", since_date="2020-01-01", client=None))
        assert summary.instruments_created == 0
        assert summary.case == "A_periodic"


class TestSeedLedgerEventDateOverride:
    """The seed-provenance event_date override: safe_date(issue_date) or
    as_of, parsed via date.fromisoformat(seed_event[:10]); on
    ValueError/TypeError it falls back to the synthesized mutation's own
    event_date. We stub apply_mutations and capture the `mutations=`
    kwarg to assert each mutation's resolved event_date."""

    def _capture(self, temp_db, monkeypatch, rows, *, report_date="2023-12-31"):
        _stage_filing(temp_db, cik=99, acc="A1", form="10-K",
                      filing_date="2024-01-01", report_date=report_date)
        _patch_extract(monkeypatch, rows)
        captured = {}

        def _fake_apply(**kw):
            captured.update(kw)
            return _FakeResult(accepted=len(kw.get("mutations", [])))

        monkeypatch.setattr(seed, "apply_mutations", _fake_apply)
        asyncio.run(seed.seed_ledger(
            cik=99, ticker="T", since_date="2020-01-01", client=None))
        return captured["mutations"]

    def test_valid_issue_date_used(self, temp_db, monkeypatch):
        rows = [{"category": "warrant", "issue_date": "2022-05-01",
                 "outstanding_count": 10, "strike_or_conversion_price": 1.0}]
        muts = self._capture(temp_db, monkeypatch, rows)
        assert muts[0].event_date == date(2022, 5, 1)

    def test_alternate_format_issue_date_normalized(self, temp_db,
                                                    monkeypatch):
        # safe_date normalizes 'May 1, 2022' -> '2022-05-01' before parse.
        rows = [{"category": "warrant", "issue_date": "May 1, 2022",
                 "outstanding_count": 10, "strike_or_conversion_price": 1.0}]
        muts = self._capture(temp_db, monkeypatch, rows)
        assert muts[0].event_date == date(2022, 5, 1)

    def test_none_issue_date_uses_as_of(self, temp_db, monkeypatch):
        # issue_date None -> safe_date None -> `or as_of` -> report_date.
        rows = [{"category": "preferred", "series_letter": "D",
                 "issue_date": None, "outstanding_count": 50,
                 "strike_or_conversion_price": 1.0}]
        muts = self._capture(temp_db, monkeypatch, rows)
        assert muts[0].event_date == date(2023, 12, 31)

    def test_missing_issue_date_uses_as_of(self, temp_db, monkeypatch):
        # No issue_date key at all -> r.get(...) None -> as_of.
        rows = [{"category": "preferred", "series_letter": "D",
                 "outstanding_count": 50,
                 "strike_or_conversion_price": 1.0}]
        muts = self._capture(temp_db, monkeypatch, rows)
        assert muts[0].event_date == date(2023, 12, 31)

    def test_unparseable_issue_date_uses_as_of(self, temp_db, monkeypatch):
        # 'garbage' -> safe_date None -> `or as_of`. (Not the ValueError
        # branch: safe_date already swallowed it.)
        rows = [{"category": "warrant", "issue_date": "garbage",
                 "outstanding_count": 10, "strike_or_conversion_price": 1.0}]
        muts = self._capture(temp_db, monkeypatch, rows)
        assert muts[0].event_date == date(2023, 12, 31)

    def test_iso_shaped_invalid_date_falls_back_to_mutation_event_date(
            self, temp_db, monkeypatch):
        # '2022-13-45' matches the YYYY-MM-DD shape regex so safe_date
        # returns it VERBATIM (no calendar validation), then
        # date.fromisoformat raises ValueError -> the override falls back
        # to the synthesized mutation's own event_date. We stub
        # _synthesize_create to plant a deterministic sentinel so the
        # fallback is observable (the real default would be date.today()).
        sentinel = date(1999, 9, 9)

        def _fake_synth(over, *, accession, filing_date):
            return CreateWarrant(count=1, strike=1.0, event_date=sentinel)

        monkeypatch.setattr(seed, "_synthesize_create", _fake_synth)
        rows = [{"category": "warrant", "issue_date": "2022-13-45"}]
        muts = self._capture(temp_db, monkeypatch, rows)
        assert muts[0].event_date == sentinel

    def test_multiple_rows_each_resolve_independently(self, temp_db,
                                                      monkeypatch):
        rows = [
            {"category": "warrant", "issue_date": "2021-03-04",
             "outstanding_count": 10, "strike_or_conversion_price": 1.0},
            {"category": "preferred", "series_letter": "D",
             "issue_date": None, "outstanding_count": 50,
             "strike_or_conversion_price": 1.0},
        ]
        muts = self._capture(temp_db, monkeypatch, rows)
        assert muts[0].event_date == date(2021, 3, 4)
        assert muts[1].event_date == date(2023, 12, 31)  # as_of fallback


class TestSeedSummaryDataclass:
    def test_construct_and_field_access(self):
        s = SeedSummary(accession="A1", form="10-K", as_of_date="2024-01-01",
                        instruments_created=3, case="A_periodic")
        assert s.accession == "A1"
        assert s.form == "10-K"
        assert s.as_of_date == "2024-01-01"
        assert s.instruments_created == 3
        assert s.case == "A_periodic"
