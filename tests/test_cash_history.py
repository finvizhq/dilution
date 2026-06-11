"""Unit tests for dilution/cash_history.py.

This module is db_backed=NONE: it reads EDGAR XBRL facts (duck-typed) and
converts non-USD via dilution.fx. We never touch the network/EDGAR/fx —
every seam is monkeypatched. The autouse temp_db fixture from conftest is
present but unused here (no dilution_* tables are read).
"""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import pytest

from dilution import cash_history as ch


# ──────────────────────────────────────────────────────────────────────
# Fake EDGAR facts plumbing
# ──────────────────────────────────────────────────────────────────────
class FakeQuery:
    """Mimics edgar facts.query() -> .by_concept(c, exact=True).execute()."""

    def __init__(self, facts_by_concept, call_log=None, raise_for=None):
        self._facts_by_concept = facts_by_concept
        self._concept = None
        self._call_log = call_log if call_log is not None else []
        self._raise_for = raise_for or set()

    def by_concept(self, concept, exact=False):
        # The production code always passes exact=True; assert that contract.
        assert exact is True, "by_concept must be called with exact=True"
        self._concept = concept
        self._call_log.append((concept, exact))
        return self

    def execute(self):
        if self._concept in self._raise_for:
            raise RuntimeError(f"boom for {self._concept}")
        return self._facts_by_concept.get(self._concept, [])


class FakeFacts:
    def __init__(self, facts_by_concept, raise_for=None):
        self._facts_by_concept = facts_by_concept
        self.call_log = []
        self._raise_for = raise_for or set()

    def query(self):
        return FakeQuery(self._facts_by_concept, self.call_log, self._raise_for)


def mkfact(*, period_end, filing_date="2025-04-10", numeric_value=100.0,
           unit="USD", fiscal_year=None, fiscal_period=None,
           accession="acc-1", form_type="10-K", period_start=None):
    return SimpleNamespace(
        period_end=period_end,
        period_start=period_start,
        filing_date=filing_date,
        numeric_value=numeric_value,
        unit=unit,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        accession=accession,
        form_type=form_type,
    )


CASH = ch._CASH_CONCEPTS[0]
OPCF = ch._OPCF_CONCEPTS[0]


@pytest.fixture(autouse=True)
def _clear_cache_and_identity(monkeypatch):
    """Reset module-global state that leaks across tests.

    - _cached lru_cache persists -> clear it so call-count asserts are clean.
    - _IDENTITY_SET guards set_identity -> reset so tests can re-assert.
    - Neutralize set_identity so no real edgar identity work happens.
    """
    ch._cached.cache_clear()
    monkeypatch.setattr(ch, "_IDENTITY_SET", False)
    monkeypatch.setattr(ch, "set_identity", lambda *a, **k: None)
    yield
    ch._cached.cache_clear()


# ──────────────────────────────────────────────────────────────────────
# _as_date
# ──────────────────────────────────────────────────────────────────────
class TestAsDate:
    def test_none_returns_none(self):
        assert ch._as_date(None) is None

    def test_bare_date_passes_through_unchanged(self):
        d = date(2025, 3, 31)
        out = ch._as_date(d)
        assert out == d
        # The guard `not isinstance(v, datetime)` means a bare date is
        # returned as-is (it is NOT funneled through the datetime branch).
        assert type(out) is date

    def test_datetime_drops_time_component(self):
        out = ch._as_date(datetime(2025, 3, 31, 12, 34, 56))
        assert out == date(2025, 3, 31)
        assert type(out) is date

    def test_iso_date_string(self):
        assert ch._as_date("2025-03-31") == date(2025, 3, 31)

    def test_iso_datetime_string(self):
        assert ch._as_date("2025-03-31T00:00:00") == date(2025, 3, 31)

    def test_malformed_string_returns_none(self):
        assert ch._as_date("not-a-date") is None

    def test_empty_string_returns_none(self):
        assert ch._as_date("") is None

    @pytest.mark.parametrize("bad", [12345, 2025, 1.5])
    def test_unparseable_numeric_returns_none(self, bad):
        # str(12345) etc. are not valid ISO datetimes -> ValueError caught.
        assert ch._as_date(bad) is None

    def test_int_basic_iso_form_parses_on_py312(self):
        # BUG/quirk note: the survey predicted str(20250331) would raise
        # ValueError -> None. On Python 3.12, datetime.fromisoformat accepts
        # the "basic" YYYYMMDD form, so this actually parses to a date.
        # Asserting the ACTUAL current behavior.
        assert ch._as_date(20250331) == date(2025, 3, 31)


# ──────────────────────────────────────────────────────────────────────
# _query_first_with_facts
# ──────────────────────────────────────────────────────────────────────
class TestQueryFirstWithFacts:
    def test_first_concept_with_facts_short_circuits(self):
        concepts = ("c1", "c2", "c3")
        facts = FakeFacts({"c1": [mkfact(period_end="2025-03-31")]})
        out = ch._query_first_with_facts(facts, concepts)
        assert len(out) == 1
        # later concepts never queried
        assert facts.call_log == [("c1", True)]

    def test_empty_first_falls_through_to_next(self):
        concepts = ("c1", "c2", "c3")
        facts = FakeFacts({"c1": [], "c2": [mkfact(period_end="2025-03-31")]})
        out = ch._query_first_with_facts(facts, concepts)
        assert len(out) == 1
        assert facts.call_log == [("c1", True), ("c2", True)]

    def test_raising_concept_swallowed_and_continues(self, caplog):
        concepts = ("c1", "c2")
        facts = FakeFacts({"c2": [mkfact(period_end="2025-03-31")]},
                          raise_for={"c1"})
        out = ch._query_first_with_facts(facts, concepts)
        assert len(out) == 1
        assert facts.call_log == [("c1", True), ("c2", True)]

    def test_all_empty_or_raising_returns_empty(self):
        concepts = ("c1", "c2")
        facts = FakeFacts({"c1": []}, raise_for={"c2"})
        assert ch._query_first_with_facts(facts, concepts) == []

    def test_exact_true_kwarg_passed(self):
        # FakeQuery.by_concept asserts exact is True; if the code ever
        # dropped the kwarg this test would error.
        facts = FakeFacts({"c1": [mkfact(period_end="2025-03-31")]})
        ch._query_first_with_facts(facts, ("c1",))
        assert facts.call_log[0] == ("c1", True)

    def test_probes_real_cash_concepts_in_order(self):
        # us-gaap probed before IFRS: only the IFRS concept has facts, so the
        # probe must fall all the way through the us-gaap concepts to it.
        ifrs_cash = ch._CASH_CONCEPTS[3]  # ifrs-full:CashAndCashEquivalents
        assert ifrs_cash.startswith("ifrs-full:")
        facts = FakeFacts({ifrs_cash: [mkfact(period_end="2025-03-31",
                                              numeric_value=42.0)]})
        out = ch._query_first_with_facts(facts, ch._CASH_CONCEPTS)
        assert len(out) == 1 and float(out[0].numeric_value) == 42.0
        # Every preceding us-gaap concept was tried, in declared order.
        assert [c for c, _ in facts.call_log] == list(ch._CASH_CONCEPTS[:4])

    def test_non_list_truthy_result_returned_as_is(self):
        # The code returns `r` on the first truthy result without coercing to
        # a list; a non-empty tuple is returned unchanged.
        facts = FakeFacts({"c1": ("only-one",)})
        out = ch._query_first_with_facts(facts, ("c1",))
        assert out == ("only-one",)


# ──────────────────────────────────────────────────────────────────────
# _build_series
# ──────────────────────────────────────────────────────────────────────
class TestBuildSeries:
    AS_OF = date(2026, 6, 15)  # mid-year month avoids day=1 cutoff confusion

    def test_no_concept_with_facts(self):
        facts = FakeFacts({})
        pts, fxf = ch._build_series(facts, self.AS_OF)
        assert pts == []
        assert fxf is False

    def test_all_unparseable_period_end(self):
        facts = FakeFacts({CASH: [
            mkfact(period_end=None),
            mkfact(period_end="not-a-date"),
        ]})
        pts, fxf = ch._build_series(facts, self.AS_OF)
        assert pts == []
        assert fxf is False

    def test_restatement_later_filing_wins(self):
        facts = FakeFacts({CASH: [
            mkfact(period_end="2025-03-31", filing_date="2025-04-10",
                   numeric_value=100.0),
            mkfact(period_end="2025-03-31", filing_date="2025-08-10",
                   numeric_value=999.0),
        ]})
        pts, _ = ch._build_series(facts, self.AS_OF)
        assert len(pts) == 1
        assert pts[0].value_usd == 999.0

    def test_equal_filing_date_keeps_first_seen(self):
        # '>' is strict, so an equal filing_date does NOT replace the
        # already-seen fact: input ordering is load-bearing here.
        facts = FakeFacts({CASH: [
            mkfact(period_end="2025-03-31", filing_date="2025-04-10",
                   numeric_value=111.0),
            mkfact(period_end="2025-03-31", filing_date="2025-04-10",
                   numeric_value=222.0),
        ]})
        pts, _ = ch._build_series(facts, self.AS_OF)
        assert len(pts) == 1
        assert pts[0].value_usd == 111.0

    def test_duplicate_end_none_filing_date_raises_typeerror(self):
        # BUG/quirk: dedup compares `_as_date(f.filing_date) > _as_date(
        # existing.filing_date)`. When a DUPLICATE-period_end fact carries a
        # None filing_date, _as_date(None) is None and `None > date(...)`
        # raises TypeError. The first fact is seen fine; the second (None,
        # same end) trips the unguarded comparison. Recorded as current
        # behavior (only fires on a same-end collision, not a lone None).
        facts = FakeFacts({CASH: [
            mkfact(period_end="2025-03-31", filing_date="2025-04-10",
                   numeric_value=111.0),
            mkfact(period_end="2025-03-31", filing_date=None,
                   numeric_value=222.0),
        ]})
        with pytest.raises(TypeError):
            ch._build_series(facts, self.AS_OF)

    def test_lone_none_filing_date_does_not_compare(self):
        # A single fact with a None filing_date never hits the comparison
        # branch (existing is None), so it is kept without error.
        facts = FakeFacts({CASH: [
            mkfact(period_end="2025-03-31", filing_date=None,
                   numeric_value=333.0),
        ]})
        pts, _ = ch._build_series(facts, self.AS_OF)
        assert len(pts) == 1 and pts[0].value_usd == 333.0

    def test_cutoff_boundary_at_cutoff_kept(self):
        # cutoff = date(as_of.year-10, as_of.month, 1) = 2016-06-01.
        # A period exactly == cutoff is NOT < cutoff, so it is KEPT.
        facts = FakeFacts({CASH: [
            mkfact(period_end="2016-06-01", numeric_value=42.0),
        ]})
        pts, _ = ch._build_series(facts, self.AS_OF)
        assert len(pts) == 1
        assert pts[0].end == date(2016, 6, 1)

    def test_cutoff_boundary_before_cutoff_dropped(self):
        # 2016-05-31 < 2016-06-01 cutoff -> dropped.
        facts = FakeFacts({CASH: [
            mkfact(period_end="2016-05-31", numeric_value=42.0),
        ]})
        pts, _ = ch._build_series(facts, self.AS_OF)
        assert pts == []

    def test_cutoff_constructed_for_year_start_month(self):
        # as_of with a January month still constructs a valid cutoff date.
        as_of = date(2026, 1, 15)  # cutoff = 2016-01-01
        facts = FakeFacts({CASH: [
            mkfact(period_end="2016-01-01", numeric_value=7.0),   # kept
            mkfact(period_end="2015-12-31", numeric_value=8.0),   # dropped
        ]})
        pts, _ = ch._build_series(facts, as_of)
        assert [p.end for p in pts] == [date(2016, 1, 1)]

    def test_unit_none_defaults_usd_no_fx(self, monkeypatch):
        called = []
        monkeypatch.setattr(ch.fx, "to_usd",
                            lambda *a, **k: called.append(a) or 1.0)
        facts = FakeFacts({CASH: [
            mkfact(period_end="2025-03-31", numeric_value=500.0, unit=None),
        ]})
        pts, fxf = ch._build_series(facts, self.AS_OF)
        assert called == []  # no fx call for USD
        assert pts[0].value_usd == 500.0
        assert pts[0].native_currency == "USD"
        assert fxf is False

    def test_unit_lowercase_usd_no_fx(self, monkeypatch):
        called = []
        monkeypatch.setattr(ch.fx, "to_usd",
                            lambda *a, **k: called.append(a) or 1.0)
        facts = FakeFacts({CASH: [
            mkfact(period_end="2025-03-31", numeric_value=500.0, unit="usd"),
        ]})
        pts, _ = ch._build_series(facts, self.AS_OF)
        assert called == []
        assert pts[0].native_currency == "USD"

    def test_unit_lowercase_eur_uppercased_and_converted(self, monkeypatch):
        seen = {}

        def fake(amount, currency, on):
            seen["args"] = (amount, currency, on)
            return amount * 1.1

        monkeypatch.setattr(ch.fx, "to_usd", fake)
        facts = FakeFacts({CASH: [
            mkfact(period_end="2025-03-31", numeric_value=100.0, unit="eur"),
        ]})
        pts, fxf = ch._build_series(facts, self.AS_OF)
        # uppercased to EUR and passed to fx with the period end-date
        assert seen["args"] == (100.0, "EUR", date(2025, 3, 31))
        assert pts[0].value_usd == pytest.approx(110.0)
        assert pts[0].native_currency == "EUR"
        assert pts[0].native_value == 100.0
        assert fxf is False

    def test_fx_failure_skips_point_and_flips_flag(self, monkeypatch):
        # EUR fails -> skipped + fx_failed True; USD point still included.
        def fake(amount, currency, on):
            return None if currency == "EUR" else amount

        monkeypatch.setattr(ch.fx, "to_usd", fake)
        facts = FakeFacts({CASH: [
            mkfact(period_end="2025-03-31", numeric_value=100.0, unit="EUR"),
            mkfact(period_end="2025-06-30", numeric_value=200.0, unit="USD"),
        ]})
        pts, fxf = ch._build_series(facts, self.AS_OF)
        assert fxf is True
        assert len(pts) == 1
        assert pts[0].end == date(2025, 6, 30)
        assert pts[0].value_usd == 200.0

    def test_fiscal_year_falsy_defaults_to_end_year(self):
        facts = FakeFacts({CASH: [
            mkfact(period_end="2025-03-31", fiscal_year=0),
            mkfact(period_end="2024-12-31", fiscal_year=None,
                   filing_date="2025-02-01"),
        ]})
        pts, _ = ch._build_series(facts, self.AS_OF)
        by_end = {p.end: p for p in pts}
        assert by_end[date(2025, 3, 31)].fy == 2025
        assert by_end[date(2024, 12, 31)].fy == 2024

    def test_fiscal_year_truthy_used(self):
        facts = FakeFacts({CASH: [
            mkfact(period_end="2025-03-31", fiscal_year=2024),
        ]})
        pts, _ = ch._build_series(facts, self.AS_OF)
        assert pts[0].fy == 2024

    def test_fiscal_period_none_coerced_to_empty(self):
        facts = FakeFacts({CASH: [
            mkfact(period_end="2025-03-31", fiscal_period=None),
        ]})
        pts, _ = ch._build_series(facts, self.AS_OF)
        assert pts[0].fp == ""

    def test_fiscal_period_value_used(self):
        facts = FakeFacts({CASH: [
            mkfact(period_end="2025-03-31", fiscal_period="Q1"),
        ]})
        pts, _ = ch._build_series(facts, self.AS_OF)
        assert pts[0].fp == "Q1"

    def test_accession_and_form_coerced_to_str(self):
        facts = FakeFacts({CASH: [
            mkfact(period_end="2025-03-31", accession=None, form_type=None),
        ]})
        pts, _ = ch._build_series(facts, self.AS_OF)
        assert pts[0].accession == ""
        assert pts[0].form == ""

    def test_output_sorted_ascending_by_end(self):
        facts = FakeFacts({CASH: [
            mkfact(period_end="2025-06-30", numeric_value=3),
            mkfact(period_end="2024-12-31", numeric_value=1,
                   filing_date="2025-02-01"),
            mkfact(period_end="2025-03-31", numeric_value=2),
        ]})
        pts, _ = ch._build_series(facts, self.AS_OF)
        assert [p.end for p in pts] == [
            date(2024, 12, 31), date(2025, 3, 31), date(2025, 6, 30),
        ]

    def test_numeric_value_string_cast_via_float(self):
        facts = FakeFacts({CASH: [
            mkfact(period_end="2025-03-31", numeric_value="123.0"),
        ]})
        pts, _ = ch._build_series(facts, self.AS_OF)
        assert pts[0].value_usd == 123.0
        assert pts[0].native_value == 123.0

    def test_none_numeric_value_raises_typeerror(self):
        # Current code does float(f.numeric_value) with no guard -> None raises.
        # Documenting the unguarded path (survey gotcha #4).
        facts = FakeFacts({CASH: [
            mkfact(period_end="2025-03-31", numeric_value=None),
        ]})
        with pytest.raises(TypeError):
            ch._build_series(facts, self.AS_OF)


# ──────────────────────────────────────────────────────────────────────
# _latest_quarterly_opcf
# ──────────────────────────────────────────────────────────────────────
class TestLatestQuarterlyOpcf:
    PERIOD_END = date(2025, 6, 30)

    def test_no_opcf_facts(self):
        facts = FakeFacts({})
        assert ch._latest_quarterly_opcf(facts, self.PERIOD_END) is None

    def test_all_candidates_after_period_end(self):
        facts = FakeFacts({OPCF: [
            mkfact(period_start="2025-07-01", period_end="2025-09-30",
                   numeric_value=-100.0),
        ]})
        assert ch._latest_quarterly_opcf(facts, self.PERIOD_END) is None

    def test_candidate_exactly_at_period_end_included(self):
        # end == period_end is not skipped (only `end > period_end` skips).
        facts = FakeFacts({OPCF: [
            mkfact(period_start="2025-04-01", period_end="2025-06-30",
                   numeric_value=-90.0),
        ]})
        out = ch._latest_quarterly_opcf(facts, self.PERIOD_END)
        days = (date(2025, 6, 30) - date(2025, 4, 1)).days
        assert out == pytest.approx(-90.0 * 90.0 / days)

    def test_missing_start_skipped(self):
        facts = FakeFacts({OPCF: [
            mkfact(period_start=None, period_end="2025-06-30",
                   numeric_value=-90.0),
        ]})
        assert ch._latest_quarterly_opcf(facts, self.PERIOD_END) is None

    def test_missing_end_skipped(self):
        facts = FakeFacts({OPCF: [
            mkfact(period_start="2025-04-01", period_end=None,
                   numeric_value=-90.0),
        ]})
        assert ch._latest_quarterly_opcf(facts, self.PERIOD_END) is None

    def test_zero_or_negative_days_skipped(self):
        # start == end -> days 0 (div-by-zero guard); start after end -> negative.
        facts = FakeFacts({OPCF: [
            mkfact(period_start="2025-06-30", period_end="2025-06-30",
                   numeric_value=-90.0),
            mkfact(period_start="2025-06-30", period_end="2025-05-30",
                   numeric_value=-90.0),
        ]})
        assert ch._latest_quarterly_opcf(facts, self.PERIOD_END) is None

    def test_90_day_quarter_unchanged(self):
        facts = FakeFacts({OPCF: [
            mkfact(period_start="2025-04-01", period_end="2025-06-30",
                   numeric_value=-90.0),
        ]})
        days = (date(2025, 6, 30) - date(2025, 4, 1)).days
        out = ch._latest_quarterly_opcf(facts, self.PERIOD_END)
        assert out == pytest.approx(-90.0 * 90.0 / days)

    def test_365_day_fy_quartered(self):
        facts = FakeFacts({OPCF: [
            mkfact(period_start="2024-07-01", period_end="2025-06-30",
                   numeric_value=-365.0),
        ]})
        days = (date(2025, 6, 30) - date(2024, 7, 1)).days  # 364
        out = ch._latest_quarterly_opcf(facts, self.PERIOD_END)
        assert out == pytest.approx(-365.0 * 90.0 / days)

    def test_negative_burn_stays_negative(self):
        facts = FakeFacts({OPCF: [
            mkfact(period_start="2025-04-01", period_end="2025-06-30",
                   numeric_value=-1000.0),
        ]})
        out = ch._latest_quarterly_opcf(facts, self.PERIOD_END)
        assert out < 0

    def test_tie_on_end_prefers_longer_fy(self):
        # FIXED (was bug B#6): on a same-end tie the sort key is now (end, days)
        # so candidates[-1] picks the LONGER (FY) period, matching the comment.
        facts = FakeFacts({OPCF: [
            mkfact(period_start="2025-04-01", period_end="2025-06-30",
                   numeric_value=-90.0),    # Q
            mkfact(period_start="2024-07-01", period_end="2025-06-30",
                   numeric_value=-400.0),   # FY (longer)
        ]})
        out = ch._latest_quarterly_opcf(facts, self.PERIOD_END)
        fy_days = (date(2025, 6, 30) - date(2024, 7, 1)).days
        assert out == pytest.approx(-400.0 * 90.0 / fy_days)

    def test_latest_end_wins_over_earlier(self):
        facts = FakeFacts({OPCF: [
            mkfact(period_start="2024-10-01", period_end="2024-12-31",
                   numeric_value=-50.0),
            mkfact(period_start="2025-04-01", period_end="2025-06-30",
                   numeric_value=-90.0),
        ]})
        out = ch._latest_quarterly_opcf(facts, self.PERIOD_END)
        days = (date(2025, 6, 30) - date(2025, 4, 1)).days
        assert out == pytest.approx(-90.0 * 90.0 / days)  # the later quarter

    def test_future_candidate_filtered_earlier_valid_used(self):
        # A future-dated period (end > period_end) is dropped; the latest
        # surviving past period is chosen. Guards against the filter and the
        # candidates[-1] pick interacting wrongly.
        facts = FakeFacts({OPCF: [
            mkfact(period_start="2025-07-01", period_end="2025-09-30",
                   numeric_value=-9999.0),  # future -> excluded
            mkfact(period_start="2025-01-01", period_end="2025-03-31",
                   numeric_value=-50.0),    # past Q
            mkfact(period_start="2025-04-01", period_end="2025-06-30",
                   numeric_value=-90.0),    # latest valid past Q (==period_end)
        ]})
        out = ch._latest_quarterly_opcf(facts, self.PERIOD_END)
        days = (date(2025, 6, 30) - date(2025, 4, 1)).days
        assert out == pytest.approx(-90.0 * 90.0 / days)

    def test_non_usd_calls_fx_with_chosen_end(self, monkeypatch):
        seen = {}

        def fake(amount, currency, on):
            seen["args"] = (amount, currency, on)
            return amount * 1.2

        monkeypatch.setattr(ch.fx, "to_usd", fake)
        facts = FakeFacts({OPCF: [
            mkfact(period_start="2025-04-01", period_end="2025-06-30",
                   numeric_value=-100.0, unit="eur"),
        ]})
        out = ch._latest_quarterly_opcf(facts, self.PERIOD_END)
        days = (date(2025, 6, 30) - date(2025, 4, 1)).days
        assert seen["args"] == (-100.0, "EUR", date(2025, 6, 30))
        assert out == pytest.approx(-100.0 * 1.2 * 90.0 / days)

    def test_non_usd_fx_failure_returns_none(self, monkeypatch):
        monkeypatch.setattr(ch.fx, "to_usd", lambda *a, **k: None)
        facts = FakeFacts({OPCF: [
            mkfact(period_start="2025-04-01", period_end="2025-06-30",
                   numeric_value=-100.0, unit="EUR"),
        ]})
        assert ch._latest_quarterly_opcf(facts, self.PERIOD_END) is None

    def test_unit_none_defaults_usd_no_fx(self, monkeypatch):
        called = []
        monkeypatch.setattr(ch.fx, "to_usd",
                            lambda *a, **k: called.append(a) or 1.0)
        facts = FakeFacts({OPCF: [
            mkfact(period_start="2025-04-01", period_end="2025-06-30",
                   numeric_value=-90.0, unit=None),
        ]})
        out = ch._latest_quarterly_opcf(facts, self.PERIOD_END)
        assert called == []
        assert out is not None and out < 0


# ──────────────────────────────────────────────────────────────────────
# _ensure_identity
# ──────────────────────────────────────────────────────────────────────
class TestEnsureIdentity:
    def test_sets_identity_once(self, monkeypatch):
        calls = []
        monkeypatch.setattr(ch, "_IDENTITY_SET", False)
        monkeypatch.setattr(ch, "set_identity",
                            lambda ident: calls.append(ident))
        ch._ensure_identity()
        ch._ensure_identity()  # idempotent
        assert len(calls) == 1

    def test_already_set_is_noop(self, monkeypatch):
        calls = []
        monkeypatch.setattr(ch, "_IDENTITY_SET", True)
        monkeypatch.setattr(ch, "set_identity",
                            lambda ident: calls.append(ident))
        ch._ensure_identity()
        assert calls == []

    def test_uses_config_identity_when_present(self, monkeypatch):
        calls = []
        monkeypatch.setattr(ch, "_IDENTITY_SET", False)
        monkeypatch.setattr(ch, "set_identity",
                            lambda ident: calls.append(ident))
        monkeypatch.setattr(ch.config, "EDGAR_IDENTITY",
                            "me me@example.com", raising=False)
        ch._ensure_identity()
        assert calls == ["me me@example.com"]


# ──────────────────────────────────────────────────────────────────────
# fetch_cash_history — the deterministic bridge math
# ──────────────────────────────────────────────────────────────────────
class TestFetchCashHistoryBridge:
    """Isolate the arithmetic bridge by patching the helper seams.

    We stub Company.get_facts (returns a sentinel), _build_series (returns a
    fixed series), and _latest_quarterly_opcf (returns a fixed op_cf). This
    keeps EDGAR/fx entirely out of the picture.
    """

    @staticmethod
    def _series(end, value):
        return [ch.CashPoint(end=end, value_usd=value, fy=end.year, fp="FY",
                             accession="acc", form="10-K",
                             native_currency="USD", native_value=value)]

    @staticmethod
    def _patch(monkeypatch, *, series, op_cf, fx_failed=False):
        monkeypatch.setattr(ch, "_IDENTITY_SET", True)
        monkeypatch.setattr(ch, "Company",
                            lambda cik: SimpleNamespace(
                                get_facts=lambda: SimpleNamespace()))
        monkeypatch.setattr(ch, "_build_series",
                            lambda facts, as_of: (series, fx_failed))
        monkeypatch.setattr(ch, "_latest_quarterly_opcf",
                            lambda facts, end: op_cf)

    def test_get_facts_failure_returns_empty(self, monkeypatch):
        monkeypatch.setattr(ch, "_IDENTITY_SET", True)

        def boom(cik):
            raise RuntimeError("network down")

        monkeypatch.setattr(ch, "Company", boom)
        as_of = date(2025, 6, 30)
        out = ch.fetch_cash_history(1, as_of=as_of)
        assert out.series == []
        assert out.as_of == as_of
        assert out.latest_cash_usd is None
        assert out.current_cash_est_usd is None
        assert out.fx_failed is False

    def test_empty_series_carries_fx_failed(self, monkeypatch):
        self._patch(monkeypatch, series=[], op_cf=None, fx_failed=True)
        as_of = date(2025, 6, 30)
        out = ch.fetch_cash_history(1, as_of=as_of)
        assert out.series == []
        assert out.fx_failed is True
        assert out.as_of == as_of
        assert out.current_cash_est_usd is None

    def test_opcf_none_no_proration(self, monkeypatch):
        latest_end = date(2025, 3, 31)
        self._patch(monkeypatch,
                    series=self._series(latest_end, 1_000_000.0), op_cf=None)
        out = ch.fetch_cash_history(1, as_of=date(2025, 6, 30))
        assert out.op_cf_prorated_usd is None
        assert out.current_cash_est_usd == 1_000_000.0
        assert out.months_of_cash is None
        # stale_days set even when op_cf None
        assert out.stale_days == (date(2025, 6, 30) - latest_end).days

    def test_opcf_zero_no_runway(self, monkeypatch):
        latest_end = date(2025, 3, 31)
        self._patch(monkeypatch,
                    series=self._series(latest_end, 500.0), op_cf=0.0)
        out = ch.fetch_cash_history(1, as_of=date(2025, 6, 30))
        # zero op_cf is prorated (to 0) but months stays None (0 is not < 0)
        assert out.op_cf_prorated_usd == 0.0
        assert out.months_of_cash is None
        assert out.current_cash_est_usd == 500.0

    def test_opcf_positive_no_runway(self, monkeypatch):
        latest_end = date(2025, 3, 31)
        as_of = date(2025, 6, 30)  # ~90 days later
        self._patch(monkeypatch,
                    series=self._series(latest_end, 1000.0), op_cf=300.0)
        out = ch.fetch_cash_history(1, as_of=as_of)
        days = (as_of - latest_end).days
        assert out.op_cf_prorated_usd == pytest.approx(300.0 * days / 90.0)
        # cash-generating -> no runway figure
        assert out.months_of_cash is None
        assert out.current_cash_est_usd == pytest.approx(
            1000.0 + 300.0 * days / 90.0)

    def test_opcf_negative_runway_computed(self, monkeypatch):
        latest_end = date(2025, 3, 31)
        as_of = date(2025, 4, 30)  # 30 days later
        op_cf = -900.0  # quarterly burn
        self._patch(monkeypatch,
                    series=self._series(latest_end, 9000.0), op_cf=op_cf)
        out = ch.fetch_cash_history(1, as_of=as_of)
        days = (as_of - latest_end).days  # 30
        prorated = op_cf * days / 90.0
        current = 9000.0 + prorated
        monthly_burn = -op_cf / 3.0  # 300
        assert out.op_cf_prorated_usd == pytest.approx(prorated)
        assert out.current_cash_est_usd == pytest.approx(current)
        assert out.months_of_cash == pytest.approx(current / monthly_burn)

    def test_days_since_zero_proration_zero(self, monkeypatch):
        latest_end = date(2025, 3, 31)
        self._patch(monkeypatch,
                    series=self._series(latest_end, 5000.0), op_cf=-600.0)
        out = ch.fetch_cash_history(1, as_of=latest_end)  # as_of == latest.end
        assert out.stale_days == 0
        assert out.op_cf_prorated_usd == pytest.approx(0.0)
        assert out.current_cash_est_usd == pytest.approx(5000.0)

    def test_days_since_45_half_quarter(self, monkeypatch):
        latest_end = date(2025, 3, 31)
        as_of = latest_end.replace() + (date(2025, 5, 15) - latest_end)
        # use explicit 45-day offset
        as_of = date(2025, 5, 15)
        days = (as_of - latest_end).days
        op_cf = -900.0
        self._patch(monkeypatch,
                    series=self._series(latest_end, 5000.0), op_cf=op_cf)
        out = ch.fetch_cash_history(1, as_of=as_of)
        assert out.op_cf_prorated_usd == pytest.approx(op_cf * days / 90.0)

    def test_days_since_90_full_quarter(self, monkeypatch):
        latest_end = date(2025, 3, 31)
        as_of = date(2025, 6, 29)  # exactly 90 days later
        assert (as_of - latest_end).days == 90
        op_cf = -900.0
        self._patch(monkeypatch,
                    series=self._series(latest_end, 5000.0), op_cf=op_cf)
        out = ch.fetch_cash_history(1, as_of=as_of)
        assert out.op_cf_prorated_usd == pytest.approx(op_cf)  # full quarter

    def test_as_of_before_latest_end_negative_proration(self, monkeypatch):
        # stale/future-dated data -> negative days_since -> negative proration.
        latest_end = date(2025, 6, 30)
        as_of = date(2025, 6, 15)  # before latest end
        op_cf = -900.0
        self._patch(monkeypatch,
                    series=self._series(latest_end, 5000.0), op_cf=op_cf)
        out = ch.fetch_cash_history(1, as_of=as_of)
        days = (as_of - latest_end).days  # negative
        assert days < 0
        assert out.stale_days == days
        assert out.op_cf_prorated_usd == pytest.approx(op_cf * days / 90.0)
        # negative * negative -> positive proration adds cash
        assert out.op_cf_prorated_usd > 0

    def test_capital_raised_none_not_added(self, monkeypatch):
        latest_end = date(2025, 3, 31)
        self._patch(monkeypatch,
                    series=self._series(latest_end, 1000.0), op_cf=None)
        out = ch.fetch_cash_history(1, as_of=latest_end,
                                    capital_raised_usd=None)
        assert out.capital_raised_usd is None
        assert out.current_cash_est_usd == 1000.0

    @pytest.mark.parametrize("raised", [0.0, 250.0, -250.0])
    def test_capital_raised_added(self, monkeypatch, raised):
        latest_end = date(2025, 3, 31)
        self._patch(monkeypatch,
                    series=self._series(latest_end, 1000.0), op_cf=None)
        out = ch.fetch_cash_history(1, as_of=latest_end,
                                    capital_raised_usd=raised)
        assert out.capital_raised_usd == raised
        assert out.current_cash_est_usd == pytest.approx(1000.0 + raised)

    def test_negative_current_est_not_clamped(self, monkeypatch):
        # latest tiny, big burn proration, big negative raise -> sub-zero est.
        latest_end = date(2025, 3, 31)
        as_of = date(2025, 6, 29)  # 90 days
        op_cf = -2000.0
        self._patch(monkeypatch,
                    series=self._series(latest_end, 1000.0), op_cf=op_cf)
        out = ch.fetch_cash_history(1, as_of=as_of,
                                    capital_raised_usd=-500.0)
        # 1000 + (-2000) + (-500) = -1500
        assert out.current_cash_est_usd == pytest.approx(-1500.0)
        # months allowed to be negative (current_est < 0, burn > 0)
        monthly_burn = -op_cf / 3.0
        assert out.months_of_cash == pytest.approx(-1500.0 / monthly_burn)
        assert out.months_of_cash < 0

    def test_fx_failed_propagated_with_full_series(self, monkeypatch):
        latest_end = date(2025, 3, 31)
        self._patch(monkeypatch,
                    series=self._series(latest_end, 1000.0), op_cf=None,
                    fx_failed=True)
        out = ch.fetch_cash_history(1, as_of=latest_end)
        assert out.fx_failed is True
        assert out.latest_cash_usd == 1000.0

    def test_latest_is_last_in_series(self, monkeypatch):
        # latest used for the bridge is series[-1].
        s = (self._series(date(2024, 12, 31), 100.0)
             + self._series(date(2025, 3, 31), 200.0))
        monkeypatch.setattr(ch, "_IDENTITY_SET", True)
        monkeypatch.setattr(ch, "Company",
                            lambda cik: SimpleNamespace(
                                get_facts=lambda: SimpleNamespace()))
        monkeypatch.setattr(ch, "_build_series",
                            lambda facts, as_of: (s, False))
        monkeypatch.setattr(ch, "_latest_quarterly_opcf",
                            lambda facts, end: None)
        out = ch.fetch_cash_history(1, as_of=date(2025, 3, 31))
        assert out.latest_period_end == date(2025, 3, 31)
        assert out.latest_cash_usd == 200.0

    def test_as_of_none_defaults_to_today(self, monkeypatch):
        latest_end = date(2025, 3, 31)
        sentinel = date(2025, 7, 1)

        class FakeDate(date):
            @classmethod
            def today(cls):
                return sentinel

        monkeypatch.setattr(ch, "date", FakeDate)
        self._patch(monkeypatch,
                    series=self._series(latest_end, 1000.0), op_cf=None)
        out = ch.fetch_cash_history(1)  # no as_of
        assert out.as_of == sentinel
        assert out.stale_days == (sentinel - latest_end).days

    def test_ensure_identity_called(self, monkeypatch):
        calls = []
        monkeypatch.setattr(ch, "_IDENTITY_SET", False)
        monkeypatch.setattr(ch, "set_identity",
                            lambda ident: calls.append(ident))
        self._patch(monkeypatch,
                    series=self._series(date(2025, 3, 31), 1.0), op_cf=None)
        # _patch sets _IDENTITY_SET True; undo so identity actually fires
        monkeypatch.setattr(ch, "_IDENTITY_SET", False)
        ch.fetch_cash_history(1, as_of=date(2025, 3, 31))
        assert len(calls) == 1

    def test_cik_coerced_to_int_for_company(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(ch, "_IDENTITY_SET", True)

        def fake_company(cik):
            seen["cik"] = cik
            return SimpleNamespace(get_facts=lambda: SimpleNamespace())

        monkeypatch.setattr(ch, "Company", fake_company)
        monkeypatch.setattr(ch, "_build_series",
                            lambda facts, as_of: ([], False))
        ch.fetch_cash_history("0000320193", as_of=date(2025, 3, 31))
        assert seen["cik"] == 320193
        assert isinstance(seen["cik"], int)


# ──────────────────────────────────────────────────────────────────────
# fetch_cash_history — END-TO-END through the REAL helpers
# ──────────────────────────────────────────────────────────────────────
class TestFetchCashHistoryIntegration:
    """Drive fetch_cash_history through the genuine _build_series and
    _latest_quarterly_opcf — only Company().get_facts() is faked. The
    isolated bridge tests stub both helpers, so they cannot catch a wiring
    bug (wrong arg order, latest.end not threaded into the opcf lookup,
    fx_failed not propagated). This test exercises that real wiring.
    """

    @staticmethod
    def _facts(facts_by_concept):
        ff = FakeFacts(facts_by_concept)
        return SimpleNamespace(get_facts=lambda: ff)

    def test_full_path_cash_plus_burn(self, monkeypatch):
        monkeypatch.setattr(ch, "_IDENTITY_SET", True)
        facts_obj = self._facts({
            CASH: [mkfact(period_end="2025-03-31", numeric_value=10_000.0,
                          fiscal_period="Q1")],
            OPCF: [mkfact(period_start="2025-01-01", period_end="2025-03-31",
                          numeric_value=-900.0)],
        })
        monkeypatch.setattr(ch, "Company", lambda cik: facts_obj)
        as_of = date(2025, 6, 29)  # exactly 90 days after the latest end
        out = ch.fetch_cash_history(7, as_of=as_of)

        assert out.latest_period_end == date(2025, 3, 31)
        assert out.latest_cash_usd == 10_000.0
        # opcf normalized to a 90-day quarter from the real Q1 duration.
        q_days = (date(2025, 3, 31) - date(2025, 1, 1)).days
        exp_opcf = -900.0 * 90.0 / q_days
        assert out.op_cf_quarterly_usd == pytest.approx(exp_opcf)
        # stale 90 days -> full-quarter proration.
        assert out.stale_days == 90
        assert out.op_cf_prorated_usd == pytest.approx(exp_opcf)
        assert out.current_cash_est_usd == pytest.approx(10_000.0 + exp_opcf)
        monthly_burn = -exp_opcf / 3.0
        assert out.months_of_cash == pytest.approx(
            (10_000.0 + exp_opcf) / monthly_burn)
        assert out.fx_failed is False

    def test_opcf_uses_latest_period_end_not_an_earlier_one(self, monkeypatch):
        # Two cash points; latest is 2025-06-30. An opcf fact ending AFTER
        # 2025-06-30 must be excluded by the real helper (period_end is the
        # latest cash end, threaded by fetch_cash_history), leaving the
        # 2025-06-30 quarter.
        monkeypatch.setattr(ch, "_IDENTITY_SET", True)
        facts_obj = self._facts({
            CASH: [
                mkfact(period_end="2025-03-31", numeric_value=8_000.0),
                mkfact(period_end="2025-06-30", numeric_value=5_000.0,
                       filing_date="2025-08-01"),
            ],
            OPCF: [
                mkfact(period_start="2025-04-01", period_end="2025-06-30",
                       numeric_value=-300.0),
                mkfact(period_start="2025-07-01", period_end="2025-09-30",
                       numeric_value=-9_999.0),  # after latest cash end
            ],
        })
        monkeypatch.setattr(ch, "Company", lambda cik: facts_obj)
        out = ch.fetch_cash_history(7, as_of=date(2025, 9, 30))
        assert out.latest_period_end == date(2025, 6, 30)
        q_days = (date(2025, 6, 30) - date(2025, 4, 1)).days
        assert out.op_cf_quarterly_usd == pytest.approx(-300.0 * 90.0 / q_days)

    def test_fx_failure_propagates_end_to_end(self, monkeypatch):
        # A non-USD cash fact whose fx conversion fails is skipped and sets
        # fx_failed; a USD point survives. Verifies the flag is carried out
        # of the real _build_series into the CashHistory.
        monkeypatch.setattr(ch, "_IDENTITY_SET", True)
        monkeypatch.setattr(ch.fx, "to_usd",
                            lambda amt, cur, on: None if cur == "EUR" else amt)
        facts_obj = self._facts({
            CASH: [
                mkfact(period_end="2025-03-31", numeric_value=100.0,
                       unit="EUR"),
                mkfact(period_end="2025-06-30", numeric_value=200.0,
                       unit="USD", filing_date="2025-08-01"),
            ],
        })
        monkeypatch.setattr(ch, "Company", lambda cik: facts_obj)
        out = ch.fetch_cash_history(7, as_of=date(2025, 7, 1))
        assert out.fx_failed is True
        assert len(out.series) == 1
        assert out.latest_cash_usd == 200.0

    def test_no_cash_facts_returns_empty_history(self, monkeypatch):
        monkeypatch.setattr(ch, "_IDENTITY_SET", True)
        facts_obj = self._facts({})  # no cash concept yields facts
        monkeypatch.setattr(ch, "Company", lambda cik: facts_obj)
        out = ch.fetch_cash_history(7, as_of=date(2025, 7, 1))
        assert out.series == []
        assert out.latest_cash_usd is None
        assert out.current_cash_est_usd is None


# ──────────────────────────────────────────────────────────────────────
# fetch_cash_history_cached / _cached
# ──────────────────────────────────────────────────────────────────────
class TestFetchCashHistoryCached:
    def test_same_args_calls_underlying_once(self, monkeypatch):
        calls = []

        def fake(cik, *, as_of=None, capital_raised_usd=None):
            calls.append((cik, as_of, capital_raised_usd))
            return ch.CashHistory(as_of=as_of)

        monkeypatch.setattr(ch, "fetch_cash_history", fake)
        ch._cached.cache_clear()
        as_of = date(2025, 6, 30)
        ch.fetch_cash_history_cached(1, as_of=as_of)
        ch.fetch_cash_history_cached(1, as_of=as_of)
        assert len(calls) == 1

    def test_str_and_int_cik_share_cache_entry(self, monkeypatch):
        calls = []

        def fake(cik, *, as_of=None, capital_raised_usd=None):
            calls.append(cik)
            return ch.CashHistory(as_of=as_of)

        monkeypatch.setattr(ch, "fetch_cash_history", fake)
        ch._cached.cache_clear()
        as_of = date(2025, 6, 30)
        ch.fetch_cash_history_cached("42", as_of=as_of)
        ch.fetch_cash_history_cached(42, as_of=as_of)
        # cik normalized to int -> single underlying call
        assert calls == [42]

    def test_different_capital_raised_distinct_entries(self, monkeypatch):
        calls = []

        def fake(cik, *, as_of=None, capital_raised_usd=None):
            calls.append(capital_raised_usd)
            return ch.CashHistory(as_of=as_of)

        monkeypatch.setattr(ch, "fetch_cash_history", fake)
        ch._cached.cache_clear()
        as_of = date(2025, 6, 30)
        ch.fetch_cash_history_cached(1, as_of=as_of, capital_raised_usd=100.0)
        ch.fetch_cash_history_cached(1, as_of=as_of, capital_raised_usd=200.0)
        assert calls == [100.0, 200.0]

    def test_as_of_none_uses_today_isoformat_key(self, monkeypatch):
        seen = {}
        sentinel = date(2025, 7, 1)

        class FakeDate(date):
            @classmethod
            def today(cls):
                return sentinel

        def fake(cik, *, as_of=None, capital_raised_usd=None):
            seen["as_of"] = as_of
            return ch.CashHistory(as_of=as_of)

        monkeypatch.setattr(ch, "date", FakeDate)
        monkeypatch.setattr(ch, "fetch_cash_history", fake)
        ch._cached.cache_clear()
        ch.fetch_cash_history_cached(1)
        # _cached reconstructs the date from the iso key
        assert seen["as_of"] == sentinel

    def test_caches_returned_value(self, monkeypatch):
        result = ch.CashHistory(as_of=date(2025, 6, 30), latest_cash_usd=7.0)

        def fake(cik, *, as_of=None, capital_raised_usd=None):
            return result

        monkeypatch.setattr(ch, "fetch_cash_history", fake)
        ch._cached.cache_clear()
        out1 = ch.fetch_cash_history_cached(1, as_of=date(2025, 6, 30))
        out2 = ch.fetch_cash_history_cached(1, as_of=date(2025, 6, 30))
        assert out1 is out2
        assert out1.latest_cash_usd == 7.0


# ──────────────────────────────────────────────────────────────────────
# Dataclass defaults (CashHistory / CashPoint)
# ──────────────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_cash_history_defaults(self):
        ch_obj = ch.CashHistory(as_of=date(2025, 6, 30))
        assert ch_obj.series == []
        assert ch_obj.latest_period_end is None
        assert ch_obj.latest_cash_usd is None
        assert ch_obj.fx_failed is False
        assert ch_obj.stale_days is None

    def test_cash_point_is_frozen(self):
        p = ch.CashPoint(end=date(2025, 3, 31), value_usd=1.0, fy=2025,
                         fp="FY", accession="a", form="10-K",
                         native_currency="USD", native_value=1.0)
        with pytest.raises(Exception):
            p.value_usd = 2.0  # frozen dataclass

    def test_cash_history_series_default_is_independent(self):
        a = ch.CashHistory(as_of=date(2025, 1, 1))
        b = ch.CashHistory(as_of=date(2025, 1, 1))
        a.series.append("x")
        assert b.series == []  # default_factory => not shared
