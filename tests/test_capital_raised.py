"""Unit tests for dilution/capital_raised.py.

Target: capital_raised_since(cik, since) -> float | None

Sums USD proceeds from dilution_ledger_drawdowns rows whose event_date is
strictly after `since` (textual ISO comparison), for the given CIK, where
amount_usd IS NOT NULL. COALESCE collapses a NULL SUM to 0, so the function
returns 0.0 (not None) when there are no qualifying rows, and None only when a
DB exception is raised (logged as a warning).

These tests use the autouse `temp_db` fixture from conftest.py, which reroutes
db.get_conn() to a fresh per-test SQLite DB with the production schema. Because
dilution_ledger_drawdowns has a FK to dilution_ledger(instrument_id) and the
conftest helper enables PRAGMA foreign_keys=ON, every drawdown needs a parent
instrument (and we add a company for good measure).
"""

from __future__ import annotations

import logging
from datetime import date

import pytest

from dilution.capital_raised import capital_raised_since


# The `since` used for the strict-> boundary throughout the boundary tests.
SINCE = date(2026, 1, 1)
CIK = 320193


def _stage_parent(temp_db, cik=CIK, instrument_id="ATM-1", ticker="AAPL"):
    """Create a company + instrument so drawdown FK inserts succeed."""
    temp_db.add_company(cik, ticker)
    temp_db.add_instrument(instrument_id, cik=cik, ticker=ticker, type="atm")
    return instrument_id


class TestEmptyAndNull:
    def test_no_drawdowns_returns_zero_float_not_none(self, temp_db):
        # COALESCE(SUM(NULL),0) -> 0; the docstring says 0.0 means "no raises".
        result = capital_raised_since(CIK, SINCE)
        assert result == 0.0
        assert result is not None
        assert isinstance(result, float)

    def test_no_rows_for_this_cik_other_cik_present(self, temp_db):
        _stage_parent(temp_db, cik=111, instrument_id="OTHER-1")
        temp_db.add_drawdown("OTHER-1", cik=111,
                             event_date="2026-06-01", amount_usd=5000.0)
        # Querying a CIK with no rows of its own -> 0.0, isolated from CIK 111.
        result = capital_raised_since(CIK, SINCE)
        assert result == 0.0
        assert isinstance(result, float)

    def test_all_matching_rows_null_amount_returns_zero(self, temp_db):
        inst = _stage_parent(temp_db)
        # Both rows are after `since` but have NULL amount_usd -> SUM is NULL
        # -> COALESCE -> 0.0 (the IS NOT NULL guard also drops them).
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-02-01",
                             amount_usd=None)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-03-01",
                             amount_usd=None)
        result = capital_raised_since(CIK, SINCE)
        assert result == 0.0
        assert isinstance(result, float)

    def test_null_amount_row_does_not_error_or_contribute(self, temp_db):
        inst = _stage_parent(temp_db)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-02-01",
                             amount_usd=1000.0)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-03-01",
                             amount_usd=None)  # excluded by IS NOT NULL guard
        result = capital_raised_since(CIK, SINCE)
        assert result == pytest.approx(1000.0)


class TestBoundary:
    @pytest.mark.parametrize(
        "event_date, included",
        [
            ("2025-12-31", False),  # strictly before since -> excluded
            ("2026-01-01", False),  # exactly equal to since -> excluded (strict >)
            ("2026-01-02", True),   # since + 1 day -> included
            ("2026-06-01", True),   # well after -> included
        ],
    )
    def test_strict_greater_than_boundary(self, temp_db, event_date, included):
        inst = _stage_parent(temp_db)
        temp_db.add_drawdown(inst, cik=CIK, event_date=event_date,
                             amount_usd=777.0)
        result = capital_raised_since(CIK, SINCE)
        assert result == pytest.approx(777.0 if included else 0.0)

    def test_equal_excluded_but_next_day_included_together(self, temp_db):
        inst = _stage_parent(temp_db)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-01-01",
                             amount_usd=100.0)  # equal -> excluded
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-01-02",
                             amount_usd=200.0)  # after -> included
        temp_db.add_drawdown(inst, cik=CIK, event_date="2025-12-31",
                             amount_usd=400.0)  # before -> excluded
        result = capital_raised_since(CIK, SINCE)
        assert result == pytest.approx(200.0)


class TestSummationAndSigns:
    def test_multiple_qualifying_rows_exact_sum(self, temp_db):
        inst = _stage_parent(temp_db)
        for d, amt in [("2026-02-01", 100.0),
                       ("2026-03-01", 250.5),
                       ("2026-04-01", 49.5)]:
            temp_db.add_drawdown(inst, cik=CIK, event_date=d, amount_usd=amt)
        result = capital_raised_since(CIK, SINCE)
        assert result == pytest.approx(400.0)

    def test_negative_amount_summed_as_is_no_flooring(self, temp_db):
        inst = _stage_parent(temp_db)
        # A correction/reversal row: net can drop below a single positive row.
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-02-01",
                             amount_usd=200.0)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-03-01",
                             amount_usd=-50.0)
        result = capital_raised_since(CIK, SINCE)
        assert result == pytest.approx(150.0)

    def test_all_negative_net_is_negative(self, temp_db):
        inst = _stage_parent(temp_db)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-02-01",
                             amount_usd=-30.0)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-03-01",
                             amount_usd=-70.0)
        result = capital_raised_since(CIK, SINCE)
        assert result == pytest.approx(-100.0)

    def test_result_is_always_float_when_not_none(self, temp_db):
        inst = _stage_parent(temp_db)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-02-01",
                             amount_usd=12345)  # int in DB -> float() out
        result = capital_raised_since(CIK, SINCE)
        assert isinstance(result, float)
        assert result == pytest.approx(12345.0)


class TestCikIsolationAndCoercion:
    def test_only_target_cik_summed(self, temp_db):
        _stage_parent(temp_db, cik=CIK, instrument_id="ATM-1", ticker="AAPL")
        _stage_parent(temp_db, cik=999, instrument_id="ATM-2", ticker="OTHR")
        temp_db.add_drawdown("ATM-1", cik=CIK, event_date="2026-02-01",
                             amount_usd=1000.0)
        temp_db.add_drawdown("ATM-2", cik=999, event_date="2026-02-01",
                             amount_usd=8888.0)  # different CIK -> excluded
        result = capital_raised_since(CIK, SINCE)
        assert result == pytest.approx(1000.0)

    @pytest.mark.parametrize("cik_arg", ["320193", 320193.0, 320193])
    def test_cik_string_or_float_coerced_to_int_matches(self, temp_db, cik_arg):
        inst = _stage_parent(temp_db)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-02-01",
                             amount_usd=500.0)
        # int() coercion of "320193"/320193.0 still matches the stored int cik.
        result = capital_raised_since(cik_arg, SINCE)
        assert result == pytest.approx(500.0)

    def test_float_cik_with_fraction_truncates(self, temp_db):
        # int(320193.9) == 320193 -> still matches the stored integer cik.
        inst = _stage_parent(temp_db)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-02-01",
                             amount_usd=500.0)
        result = capital_raised_since(320193.9, SINCE)
        assert result == pytest.approx(500.0)


class TestTextualComparison:
    def test_well_formed_iso_dates_compare_chronologically(self, temp_db):
        # With zero-padded YYYY-MM-DD, lexicographic == chronological order.
        inst = _stage_parent(temp_db)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-01-09",
                             amount_usd=9.0)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-01-10",
                             amount_usd=10.0)
        # since = 2026-01-09 (compared as the string "2026-01-09"); strict >
        # so 01-09 excluded, 01-10 included.
        result = capital_raised_since(CIK, date(2026, 1, 9))
        assert result == pytest.approx(10.0)

    def test_comparison_is_textual_not_date_typed(self, temp_db):
        # Document that the WHERE clause compares event_date textually against
        # since.isoformat(). A datetime-with-time string sorts AFTER the bare
        # date string for the same calendar day under lexicographic comparison
        # ("2026-01-01T..." > "2026-01-01"), so such a row would be INCLUDED
        # even though the calendar day equals `since`. We assert this textual
        # behavior with a well-formed extra-suffix value.
        inst = _stage_parent(temp_db)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-01-01T12:00:00",
                             amount_usd=42.0)
        result = capital_raised_since(CIK, SINCE)  # since.isoformat() == "2026-01-01"
        # "2026-01-01T12:00:00" > "2026-01-01" lexicographically -> included.
        assert result == pytest.approx(42.0)


class TestAggregationDepth:
    def test_mixed_positive_null_and_negative_in_one_query(self, temp_db):
        # Combines all three behaviors at once: the IS NOT NULL guard drops
        # the NULL row, the negative row is summed as-is (no flooring), and
        # the positive row contributes fully. 1000 + (NULL->skip) + (-200) = 800.
        inst = _stage_parent(temp_db)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-02-01",
                             amount_usd=1000.0)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-03-01",
                             amount_usd=None)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-04-01",
                             amount_usd=-200.0)
        result = capital_raised_since(CIK, SINCE)
        assert result == pytest.approx(800.0)

    def test_sums_across_multiple_instruments_for_same_cik(self, temp_db):
        # The query filters by cik, not instrument_id, so drawdowns from
        # several instruments under the same issuer all roll up together.
        _stage_parent(temp_db, cik=CIK, instrument_id="ATM-1", ticker="AAPL")
        temp_db.add_instrument("PIPE-2", cik=CIK, ticker="AAPL", type="equity")
        temp_db.add_drawdown("ATM-1", cik=CIK, event_date="2026-02-01",
                             amount_usd=1000.0)
        temp_db.add_drawdown("PIPE-2", cik=CIK, event_date="2026-03-01",
                             amount_usd=2500.0)
        result = capital_raised_since(CIK, SINCE)
        assert result == pytest.approx(3500.0)

    def test_since_far_in_past_includes_all_qualifying_rows(self, temp_db):
        inst = _stage_parent(temp_db)
        for d, amt in [("2024-05-01", 10.0), ("2025-01-01", 20.0),
                       ("2026-06-01", 30.0)]:
            temp_db.add_drawdown(inst, cik=CIK, event_date=d, amount_usd=amt)
        # since well before every row -> everything strictly after -> all summed.
        result = capital_raised_since(CIK, date(2000, 1, 1))
        assert result == pytest.approx(60.0)

    def test_since_far_in_future_excludes_everything(self, temp_db):
        inst = _stage_parent(temp_db)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-06-01",
                             amount_usd=999.0)
        # No event_date is strictly after a far-future since -> 0.0, not None.
        result = capital_raised_since(CIK, date(2099, 1, 1))
        assert result == 0.0
        assert isinstance(result, float)


class TestErrorPath:
    def test_successful_call_does_not_log_warning(self, temp_db, caplog):
        # Guard against a spurious warning on the happy path: the warning is
        # reserved for the DB-error branch only.
        inst = _stage_parent(temp_db)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-02-01",
                             amount_usd=500.0)
        with caplog.at_level(logging.WARNING, logger="dilution.capital_raised"):
            result = capital_raised_since(CIK, SINCE)
        assert result == pytest.approx(500.0)
        assert not any("capital_raised_since failed" in r.getMessage()
                       for r in caplog.records)

    def test_db_error_returns_none_and_logs_warning(self, temp_db, monkeypatch,
                                                    caplog):
        import dilution.capital_raised as cr

        def boom():
            raise RuntimeError("db down")

        # The module did `from db import get_conn`, so patch the name it bound.
        monkeypatch.setattr(cr, "get_conn", boom)
        with caplog.at_level(logging.WARNING, logger="dilution.capital_raised"):
            result = cr.capital_raised_since(CIK, SINCE)
        assert result is None
        assert any("capital_raised_since failed" in r.getMessage()
                   for r in caplog.records)

    def test_db_error_does_not_propagate(self, temp_db, monkeypatch):
        import dilution.capital_raised as cr

        class BadCtx:
            def __enter__(self):
                raise sqlite_err()

            def __exit__(self, *a):
                return False

        def sqlite_err():
            import sqlite3
            return sqlite3.OperationalError("no such table")

        # get_conn() itself returns a context manager whose __enter__ raises.
        monkeypatch.setattr(cr, "get_conn", lambda: BadCtx())
        # Must swallow the exception and return None, never raise.
        assert cr.capital_raised_since(CIK, SINCE) is None

    def test_execute_failure_inside_with_block_returns_none(self, temp_db,
                                                            monkeypatch, caplog):
        # The realistic DB-error shape is not get_conn() raising, but a query
        # blowing up *after* the connection is opened (e.g. "no such table",
        # a locked DB). The try/except wraps the whole `with` block, so the
        # exception raised by conn.execute() must still be swallowed -> None,
        # and __exit__ must run so the connection is released. We assert both:
        # None is returned and the warning carries the CIK.
        import sqlite3

        import dilution.capital_raised as cr

        closed = {"exited": False}

        class FakeConn:
            def execute(self, *a, **k):
                raise sqlite3.OperationalError("no such table: drawdowns")

        class FakeCtx:
            def __enter__(self):
                return FakeConn()

            def __exit__(self, *a):
                closed["exited"] = True
                return False  # do not suppress; the function's try/except does

        monkeypatch.setattr(cr, "get_conn", lambda: FakeCtx())
        with caplog.at_level(logging.WARNING, logger="dilution.capital_raised"):
            result = cr.capital_raised_since(CIK, SINCE)
        assert result is None
        # __exit__ ran -> the `with` unwound cleanly even though execute raised.
        assert closed["exited"] is True
        assert any("capital_raised_since failed" in r.getMessage()
                   for r in caplog.records)

    def test_warning_message_includes_the_cik(self, temp_db, monkeypatch,
                                              caplog):
        # The log call is log.warning(..., cik, e); prove the CIK is actually
        # interpolated into the emitted record (guards against a future edit
        # that drops the %s arg or logs a static string).
        import dilution.capital_raised as cr

        def boom():
            raise RuntimeError("db down")

        monkeypatch.setattr(cr, "get_conn", boom)
        with caplog.at_level(logging.WARNING, logger="dilution.capital_raised"):
            cr.capital_raised_since(CIK, SINCE)
        msgs = [r.getMessage() for r in caplog.records]
        assert any(str(CIK) in m and "capital_raised_since failed" in m
                   for m in msgs), msgs


class TestMalformedDateTextualHazard:
    """The WHERE clause compares event_date as TEXT against since.isoformat(),
    so non-zero-padded / malformed date strings can sort in counterintuitive
    ways. These are NOT bugs in the function (it honestly documents textual
    comparison) but pin the current, surprising behavior so an integrator who
    relies on it is warned, and a future switch to real date typing would
    visibly break these and force a deliberate update.
    """

    def test_non_zero_padded_date_sorts_after_padded_since_and_is_included(
            self, temp_db):
        # "2026-1-2" (no zero padding) is lexicographically GREATER than the
        # zero-padded "2026-01-01" because at index 5 the char '1' (start of
        # the un-padded month) compares against '0', and '1' > '0'. So this
        # row is INCLUDED even though Jan 2 is only one day after `since`.
        # (We confirmed: "2026-1-2" > "2026-01-01" is True in SQLite.)
        inst = _stage_parent(temp_db)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-1-2",
                             amount_usd=55.0)
        result = capital_raised_since(CIK, SINCE)  # since == "2026-01-01"
        assert result == pytest.approx(55.0)

    def test_three_digit_year_typo_sorts_after_and_is_wrongly_included(
            self, temp_db):
        # A 3-digit-year typo "226-01-01" (year 226 AD, far in the PAST) sorts
        # AFTER "2026-01-01" lexicographically: index 1 compares '2' (from the
        # truncated "226") against '0' (from "2026"), and '2' > '0'. So this
        # ancient/garbage date is INCLUDED -- a real overcount hazard the
        # docstring's "undercounts are safer" intent would not want, but which
        # the textual comparison produces. Pinned as current behavior.
        inst = _stage_parent(temp_db)
        temp_db.add_drawdown(inst, cik=CIK, event_date="226-01-01",
                             amount_usd=77.0)
        result = capital_raised_since(CIK, SINCE)
        assert result == pytest.approx(77.0)

    def test_slash_formatted_date_sorts_after_dash_since_and_is_included(
            self, temp_db):
        # "2026/06/01": at index 4 '/' (0x2F) > '-' (0x2D), so any slash-format
        # date sorts after the dash-format since regardless of the actual month.
        inst = _stage_parent(temp_db)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026/06/01",
                             amount_usd=88.0)
        result = capital_raised_since(CIK, SINCE)
        assert result == pytest.approx(88.0)


class TestDrawdownCikVsParentCik:
    def test_drawdown_cik_column_drives_filter_not_parent_instrument_cik(
            self, temp_db):
        # The WHERE filters on the drawdown row's own cik column, NOT the
        # parent instrument's cik. Stage a parent under a DIFFERENT cik but
        # write the drawdown row with cik=CIK; it must still be summed for CIK.
        # (FK is on instrument_id, which is satisfied; cik is free.)
        temp_db.add_company(555, "OTHR")
        temp_db.add_instrument("INST-X", cik=555, ticker="OTHR", type="atm")
        temp_db.add_drawdown("INST-X", cik=CIK, event_date="2026-02-01",
                             amount_usd=4321.0)
        result = capital_raised_since(CIK, SINCE)
        assert result == pytest.approx(4321.0)
        # And querying the parent's cik 555 finds nothing for this row.
        assert capital_raised_since(555, SINCE) == pytest.approx(0.0)


class TestPrecisionAndMagnitude:
    def test_large_realistic_offering_sum_no_precision_loss(self, temp_db):
        # Realistic ATM/shelf magnitudes (tens of millions) sum exactly under
        # float; assert with Decimal-equivalent approx to guard a future switch
        # to a lossy accumulator.
        inst = _stage_parent(temp_db)
        for d, amt in [("2026-02-01", 12_500_000.0),
                       ("2026-03-01", 7_250_000.50),
                       ("2026-04-01", 333.33)]:
            temp_db.add_drawdown(inst, cik=CIK, event_date=d, amount_usd=amt)
        result = capital_raised_since(CIK, SINCE)
        assert result == pytest.approx(19_750_333.83)

    def test_positive_and_negative_cancel_to_exact_zero_returns_float_zero(
            self, temp_db):
        # A draw fully reversed nets to 0.0 -- distinct from the no-rows 0.0,
        # but both must be a float 0.0 (never None). Guards the COALESCE-vs-sum
        # paths producing the same observable type.
        inst = _stage_parent(temp_db)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-02-01",
                             amount_usd=1000.0)
        temp_db.add_drawdown(inst, cik=CIK, event_date="2026-03-01",
                             amount_usd=-1000.0)
        result = capital_raised_since(CIK, SINCE)
        assert result == 0.0
        assert isinstance(result, float)
