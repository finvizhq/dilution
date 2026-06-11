"""Unit tests for dilution/share_counts.py.

Covers:
  * _parse_date (pure)
  * _company_unit (db_backed)
  * _query_latest_period_facts (io_mockable, fake xbrl)
  * _latest_periodic_filing (io_mockable, fake Company)
  * _yahoo_shares / _yahoo_float (io_mockable, fake yfinance in sys.modules)
  * fetch_implied_outstanding (io_mockable, full decision tree)
  * fetch_float (io_mockable)
  * fetch_implied_outstanding_cached / fetch_float_cached (lru_cache wrappers)
  * _ensure_identity (module-global guard)

No network / SEC / yfinance call is ever made: every external seam is
monkeypatched. DB access is rerouted by the autouse temp_db fixture.
"""

from __future__ import annotations

import sys
import types
from datetime import date

import pytest

import dilution.share_counts as sc


# ── shared fakes ───────────────────────────────────────────────────────


class FakeFiling:
    """Stub for an edgartools filing object."""

    def __init__(self, *, form, accession_no, xbrl_obj=None, xbrl_exc=None):
        self.form = form
        self.accession_no = accession_no
        self._xbrl_obj = xbrl_obj
        self._xbrl_exc = xbrl_exc

    def xbrl(self):
        if self._xbrl_exc is not None:
            raise self._xbrl_exc
        return self._xbrl_obj


class FakeQuery:
    """query() -> by_concept() -> execute() chain returning a fact list."""

    def __init__(self, facts_by_concept, raise_on_concept=None):
        self._facts_by_concept = facts_by_concept
        self._raise_on_concept = raise_on_concept
        self._concept = None

    def by_concept(self, concept):
        self._concept = concept
        return self

    def execute(self):
        if (self._raise_on_concept is not None
                and self._concept == self._raise_on_concept):
            raise RuntimeError("boom")
        return self._facts_by_concept.get(self._concept, [])


class FakeFacts:
    def __init__(self, facts_by_concept, raise_on_concept=None):
        self._facts_by_concept = facts_by_concept
        self._raise_on_concept = raise_on_concept

    def query(self):
        return FakeQuery(self._facts_by_concept, self._raise_on_concept)


class FakeXbrl:
    def __init__(self, facts_by_concept, raise_on_concept=None):
        self.facts = FakeFacts(facts_by_concept, raise_on_concept)


def fact(period, value, *, dim_label=None, label=None):
    """Build an XBRL fact dict in the shape the module reads."""
    d = {"period_instant": period, "numeric_value": value}
    if dim_label is not None:
        d["dimension_member_label"] = dim_label
    if label is not None:
        d["label"] = label
    return d


@pytest.fixture(autouse=True)
def _clear_caches_and_identity(monkeypatch):
    """Reset module-level caches and the identity guard around every test."""
    sc._cached.cache_clear()
    sc._cached_float.cache_clear()
    # Never let _ensure_identity reach the real set_identity.
    monkeypatch.setattr(sc, "set_identity", lambda *a, **k: None)
    sc._IDENTITY_SET = True  # default: don't care; tests that assert reset it
    yield
    sc._cached.cache_clear()
    sc._cached_float.cache_clear()


# Clean up any fake yfinance module a test injects.
@pytest.fixture
def restore_yfinance():
    saved = sys.modules.get("yfinance", "__missing__")
    yield
    if saved == "__missing__":
        sys.modules.pop("yfinance", None)
    else:
        sys.modules["yfinance"] = saved


AS_OF = date(2026, 6, 10)


# ── _parse_date ─────────────────────────────────────────────────────────


class TestParseDate:
    def test_none_returns_none(self):
        assert sc._parse_date(None) is None

    def test_empty_string_returns_none(self):
        assert sc._parse_date("") is None

    def test_date_only(self):
        assert sc._parse_date("2026-06-10") == date(2026, 6, 10)

    def test_datetime_string_drops_time(self):
        assert sc._parse_date("2026-06-10T00:00:00") == date(2026, 6, 10)

    def test_datetime_with_nonzero_time_still_drops_time(self):
        assert sc._parse_date("2026-06-10T13:45:59") == date(2026, 6, 10)

    def test_malformed_returns_none(self):
        assert sc._parse_date("not-a-date") is None

    def test_non_iso_slash_format_returns_none(self):
        assert sc._parse_date("06/10/2026") is None

    def test_int_basic_iso_format_coerced_via_str(self):
        # str(20260610) == "20260610" which is a valid ISO basic date in
        # Python 3.11+. The function wraps input in str() before parsing.
        assert sc._parse_date(20260610) == date(2026, 6, 10)

    def test_int_zero_is_falsy_returns_none(self):
        # 0 is falsy -> short-circuits to None before any parse attempt.
        assert sc._parse_date(0) is None


# ── _company_unit ───────────────────────────────────────────────────────


class TestCompanyUnit:
    def test_missing_row_returns_defaults(self, temp_db):
        assert sc._company_unit(999999) == (False, None, None)

    def test_fpi_true(self, temp_db):
        temp_db.add_company(cik=111, ticker="AACG", is_fpi=1, ads_ratio=2.0)
        is_fpi, ratio, ticker = sc._company_unit(111)
        assert is_fpi is True
        assert ratio == 2.0
        assert ticker == "AACG"

    def test_fpi_false(self, temp_db):
        temp_db.add_company(cik=222, ticker="GENK", is_fpi=0)
        is_fpi, ratio, ticker = sc._company_unit(222)
        assert is_fpi is False
        assert ratio is None
        assert ticker == "GENK"

    def test_ads_ratio_null_stays_none(self, temp_db):
        temp_db.add_company(cik=333, ticker="X", is_fpi=1, ads_ratio=None)
        _, ratio, _ = sc._company_unit(333)
        assert ratio is None

    def test_ads_ratio_fractional_float(self, temp_db):
        temp_db.add_company(cik=444, ticker="Y", is_fpi=1, ads_ratio=0.5)
        _, ratio, _ = sc._company_unit(444)
        assert ratio == pytest.approx(0.5)
        assert isinstance(ratio, float)

    def test_cik_passed_as_str_is_coerced(self, temp_db):
        temp_db.add_company(cik=555, ticker="Z", is_fpi=0)
        # int(cik) coercion in the query param.
        assert sc._company_unit("555") == (False, None, "Z")


# ── _query_latest_period_facts ──────────────────────────────────────────


class TestQueryLatestPeriodFacts:
    CONCEPT = "EntityCommonStockSharesOutstanding"

    def test_query_raises_returns_empty(self):
        xbrl = FakeXbrl({}, raise_on_concept=self.CONCEPT)
        assert sc._query_latest_period_facts(xbrl, self.CONCEPT) == ([], None)

    def test_empty_facts_returns_empty(self):
        xbrl = FakeXbrl({self.CONCEPT: []})
        assert sc._query_latest_period_facts(xbrl, self.CONCEPT) == ([], None)

    def test_all_facts_lack_period_instant(self):
        facts = [{"numeric_value": 5.0}, {"numeric_value": 6.0}]
        xbrl = FakeXbrl({self.CONCEPT: facts})
        assert sc._query_latest_period_facts(xbrl, self.CONCEPT) == ([], None)

    def test_returns_latest_period_by_string_compare(self):
        f_old = fact("2025-12-31", 10.0)
        f_new = fact("2026-03-31", 20.0)
        xbrl = FakeXbrl({self.CONCEPT: [f_old, f_new]})
        out, period = sc._query_latest_period_facts(xbrl, self.CONCEPT)
        assert period == "2026-03-31"
        assert out == [f_new]

    def test_empty_string_period_skipped(self):
        f_empty = fact("", 10.0)
        f_good = fact("2026-03-31", 20.0)
        xbrl = FakeXbrl({self.CONCEPT: [f_empty, f_good]})
        out, period = sc._query_latest_period_facts(xbrl, self.CONCEPT)
        assert period == "2026-03-31"
        assert out == [f_good]

    def test_none_period_skipped(self):
        f_none = {"period_instant": None, "numeric_value": 1.0}
        f_good = fact("2026-03-31", 20.0)
        xbrl = FakeXbrl({self.CONCEPT: [f_none, f_good]})
        out, period = sc._query_latest_period_facts(xbrl, self.CONCEPT)
        assert period == "2026-03-31"
        assert out == [f_good]

    def test_multiple_facts_same_latest_period_all_returned(self):
        a = fact("2026-03-31", 30.0, dim_label="Class A")
        b = fact("2026-03-31", 5.0, dim_label="Class B")
        older = fact("2025-12-31", 99.0)
        xbrl = FakeXbrl({self.CONCEPT: [a, older, b]})
        out, period = sc._query_latest_period_facts(xbrl, self.CONCEPT)
        assert period == "2026-03-31"
        assert out == [a, b]

    def test_only_none_period_facts_returns_empty(self):
        f_none = {"period_instant": None, "numeric_value": 1.0}
        xbrl = FakeXbrl({self.CONCEPT: [f_none]})
        assert sc._query_latest_period_facts(xbrl, self.CONCEPT) == ([], None)

    def test_missing_period_key_skipped_among_valid(self):
        # A fact with NO period_instant key at all (f.get -> None -> "")
        # is dropped; only the keyed fact survives and sets the period.
        no_key = {"numeric_value": 7.0}
        good = fact("2026-03-31", 20.0)
        xbrl = FakeXbrl({self.CONCEPT: [no_key, good]})
        out, period = sc._query_latest_period_facts(xbrl, self.CONCEPT)
        assert period == "2026-03-31"
        assert out == [good]

    @pytest.mark.parametrize(
        "periods, expected",
        [
            # Same-length ISO strings: lexical max == chronological max.
            (["2024-01-31", "2024-12-01", "2024-06-15"], "2024-12-01"),
            (["2025-01-01", "2026-01-01"], "2026-01-01"),
            # Cross-year day-vs-month: lexical still tracks chronology for ISO.
            (["2025-12-31", "2026-01-01"], "2026-01-01"),
        ],
    )
    def test_latest_period_is_lexical_max_over_sweep(self, periods, expected):
        facts = [fact(p, float(i)) for i, p in enumerate(periods)]
        xbrl = FakeXbrl({self.CONCEPT: facts})
        out, period = sc._query_latest_period_facts(xbrl, self.CONCEPT)
        assert period == expected
        # The returned facts are exactly those whose period == the max.
        assert out == [f for f in facts if f["period_instant"] == expected]


# ── _latest_periodic_filing ─────────────────────────────────────────────


class FakeFilingsResult:
    """Return of get_filings(form=...): supports .head(n) and indexing."""

    def __init__(self, filings):
        self._filings = filings

    def head(self, n):
        return FakeFilingsResult(self._filings[:n])

    def __len__(self):
        return len(self._filings)

    def __getitem__(self, i):
        return self._filings[i]


class FakeCompany:
    """Stub for edgar.Company; maps form -> filings (or an exception)."""

    def __init__(self, by_form):
        self._by_form = by_form

    def get_filings(self, form):
        v = self._by_form.get(form)
        if isinstance(v, Exception):
            raise v
        return FakeFilingsResult(v or [])


class DatedFiling:
    def __init__(self, form, filing_date, accession_no="acc"):
        self.form = form
        self.filing_date = filing_date
        self.accession_no = accession_no


class TestLatestPeriodicFiling:
    def _patch_company(self, monkeypatch, by_form):
        monkeypatch.setattr(sc, "Company", lambda cik: FakeCompany(by_form))

    def test_no_filings_returns_none(self, monkeypatch):
        self._patch_company(monkeypatch, {})
        assert sc._latest_periodic_filing(1) is None

    def test_picks_max_filing_date_across_forms(self, monkeypatch):
        # SCNI regression guard: stale 2023 10-Q vs current 2026 20-F.
        stale_q = DatedFiling("10-Q", "2023-08-01", "q-acc")
        cur_f = DatedFiling("20-F", "2026-04-15", "f-acc")
        self._patch_company(monkeypatch, {
            "10-Q": [stale_q],
            "20-F": [cur_f],
        })
        out = sc._latest_periodic_filing(1)
        assert out is cur_f

    def test_form_that_raises_is_skipped(self, monkeypatch):
        good = DatedFiling("10-K", "2026-02-01", "k-acc")
        self._patch_company(monkeypatch, {
            "10-Q": RuntimeError("edgar down"),
            "10-K": [good],
        })
        out = sc._latest_periodic_filing(1)
        assert out is good

    def test_only_head_one_considered(self, monkeypatch):
        # head(1) keeps only the first per form; the second 10-Q (a later
        # date) must NOT be considered.
        first = DatedFiling("10-Q", "2026-01-01", "a")
        second = DatedFiling("10-Q", "2026-09-09", "b")
        self._patch_company(monkeypatch, {"10-Q": [first, second]})
        out = sc._latest_periodic_filing(1)
        assert out is first

    def test_equal_dates_picks_first_form_in_probe_order(self, monkeypatch):
        # max() is stable: on a date tie it returns the FIRST candidate,
        # i.e. the one from the earlier form in _PERIODIC_FORMS (10-Q
        # before 10-K). Documents the (rare) tie-break ordering.
        q = DatedFiling("10-Q", "2026-04-15", "q-acc")
        k = DatedFiling("10-K", "2026-04-15", "k-acc")
        self._patch_company(monkeypatch, {"10-Q": [q], "10-K": [k]})
        out = sc._latest_periodic_filing(1)
        assert out is q

    def test_filing_date_compared_lexically_as_string(self, monkeypatch):
        # Comparison is str(filing_date); a date object that stringifies to
        # ISO ('2026-...') still orders correctly against an ISO string.
        from datetime import date as _date
        new = DatedFiling("10-K", _date(2026, 2, 1), "k-acc")
        old = DatedFiling("10-Q", "2025-11-30", "q-acc")
        self._patch_company(monkeypatch, {"10-Q": [old], "10-K": [new]})
        out = sc._latest_periodic_filing(1)
        assert out is new

    def test_only_one_form_present(self, monkeypatch):
        only = DatedFiling("40-F", "2026-03-01", "f-acc")
        self._patch_company(monkeypatch, {"40-F": [only]})
        out = sc._latest_periodic_filing(1)
        assert out is only


# ── _yahoo_shares ───────────────────────────────────────────────────────


def make_fake_yf(*, info, shares_full=None, info_raises=False,
                 shares_full_raises=False):
    """Build a fake `yfinance` module object."""

    class FakeSeries:
        def __init__(self, dates):
            self._dates = dates
            self.index = [_IdxItem(d) for d in dates]

        def __len__(self):
            return len(self._dates)

    class _IdxItem:
        def __init__(self, d):
            self._d = d

        def date(self):
            return self._d

    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        @property
        def info(self):
            if info_raises:
                raise RuntimeError("info boom")
            return info

        def get_shares_full(self, start=None):
            if shares_full_raises:
                raise RuntimeError("history boom")
            if shares_full is None:
                return None
            return FakeSeries(shares_full)

    mod = types.ModuleType("yfinance")
    mod.Ticker = FakeTicker
    mod.FakeSeries = FakeSeries
    return mod


class TestYahooShares:
    def test_import_error_returns_none(self, monkeypatch, restore_yfinance):
        # Importing None raises ImportError inside the function.
        monkeypatch.setitem(sys.modules, "yfinance", None)
        assert sc._yahoo_shares("AACG") == (None, None)

    def test_info_raises_returns_none(self, monkeypatch, restore_yfinance):
        mod = make_fake_yf(info={}, info_raises=True)
        monkeypatch.setitem(sys.modules, "yfinance", mod)
        assert sc._yahoo_shares("AACG") == (None, None)

    def test_info_none_treated_as_empty(self, monkeypatch, restore_yfinance):
        mod = make_fake_yf(info=None)
        monkeypatch.setitem(sys.modules, "yfinance", mod)
        assert sc._yahoo_shares("AACG") == (None, None)

    def test_uses_implied_fallback(self, monkeypatch, restore_yfinance):
        mod = make_fake_yf(
            info={"impliedSharesOutstanding": 1234.0},
            shares_full=[date(2026, 5, 1)],
        )
        monkeypatch.setitem(sys.modules, "yfinance", mod)
        shares, as_of = sc._yahoo_shares("AACG")
        assert shares == pytest.approx(1234.0)
        assert as_of == date(2026, 5, 1)

    def test_shares_outstanding_preferred_over_implied(
            self, monkeypatch, restore_yfinance):
        mod = make_fake_yf(
            info={"sharesOutstanding": 5000.0,
                  "impliedSharesOutstanding": 9999.0},
            shares_full=None,
        )
        monkeypatch.setitem(sys.modules, "yfinance", mod)
        shares, _ = sc._yahoo_shares("AACG")
        assert shares == pytest.approx(5000.0)

    @pytest.mark.parametrize("bad", [0, -5, 0.0])
    def test_zero_or_negative_returns_none(
            self, monkeypatch, restore_yfinance, bad):
        mod = make_fake_yf(info={"sharesOutstanding": bad})
        monkeypatch.setitem(sys.modules, "yfinance", mod)
        assert sc._yahoo_shares("AACG") == (None, None)

    def test_shares_present_history_raises_keeps_none_date(
            self, monkeypatch, restore_yfinance):
        mod = make_fake_yf(info={"sharesOutstanding": 100.0},
                           shares_full_raises=True)
        monkeypatch.setitem(sys.modules, "yfinance", mod)
        shares, as_of = sc._yahoo_shares("AACG")
        assert shares == pytest.approx(100.0)
        assert as_of is None

    def test_empty_series_keeps_none_date(
            self, monkeypatch, restore_yfinance):
        mod = make_fake_yf(info={"sharesOutstanding": 100.0}, shares_full=[])
        monkeypatch.setitem(sys.modules, "yfinance", mod)
        shares, as_of = sc._yahoo_shares("AACG")
        assert shares == pytest.approx(100.0)
        assert as_of is None

    def test_series_last_index_used(self, monkeypatch, restore_yfinance):
        mod = make_fake_yf(
            info={"sharesOutstanding": 100.0},
            shares_full=[date(2026, 1, 1), date(2026, 4, 30)],
        )
        monkeypatch.setitem(sys.modules, "yfinance", mod)
        shares, as_of = sc._yahoo_shares("AACG")
        assert shares == pytest.approx(100.0)
        assert as_of == date(2026, 4, 30)

    def test_zero_shares_outstanding_falls_through_to_implied(
            self, monkeypatch, restore_yfinance):
        # `info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")`
        # — a falsy 0 sharesOutstanding must short-circuit to the implied
        # fallback, NOT be treated as the (then-rejected) value.
        mod = make_fake_yf(
            info={"sharesOutstanding": 0, "impliedSharesOutstanding": 4321.0},
            shares_full=[date(2026, 5, 5)],
        )
        monkeypatch.setitem(sys.modules, "yfinance", mod)
        shares, as_of = sc._yahoo_shares("AACG")
        assert shares == pytest.approx(4321.0)
        assert as_of == date(2026, 5, 5)

    def test_empty_info_returns_none(self, monkeypatch, restore_yfinance):
        # info present but no share keys at all -> (None, None).
        mod = make_fake_yf(info={})
        monkeypatch.setitem(sys.modules, "yfinance", mod)
        assert sc._yahoo_shares("AACG") == (None, None)


# ── _yahoo_float ────────────────────────────────────────────────────────


class TestYahooFloat:
    def test_import_error_returns_none(self, monkeypatch, restore_yfinance):
        monkeypatch.setitem(sys.modules, "yfinance", None)
        assert sc._yahoo_float("AACG") == (None, None)

    def test_info_raises_returns_none(self, monkeypatch, restore_yfinance):
        mod = make_fake_yf(info={}, info_raises=True)
        monkeypatch.setitem(sys.modules, "yfinance", mod)
        assert sc._yahoo_float("AACG") == (None, None)

    @pytest.mark.parametrize("bad", [None, 0, -1])
    def test_missing_or_nonpositive_returns_none(
            self, monkeypatch, restore_yfinance, bad):
        info = {} if bad is None else {"floatShares": bad}
        mod = make_fake_yf(info=info)
        monkeypatch.setitem(sys.modules, "yfinance", mod)
        assert sc._yahoo_float("AACG") == (None, None)

    def test_reads_float_shares_not_shares_outstanding(
            self, monkeypatch, restore_yfinance):
        # floatShares must be the field read; sharesOutstanding must NOT
        # be substituted.
        mod = make_fake_yf(
            info={"floatShares": 700.0, "sharesOutstanding": 999.0},
            shares_full=[date(2026, 3, 3)],
        )
        monkeypatch.setitem(sys.modules, "yfinance", mod)
        shares, as_of = sc._yahoo_float("AACG")
        assert shares == pytest.approx(700.0)
        assert as_of == date(2026, 3, 3)

    def test_present_history_raises_keeps_none_date(
            self, monkeypatch, restore_yfinance):
        mod = make_fake_yf(info={"floatShares": 700.0},
                           shares_full_raises=True)
        monkeypatch.setitem(sys.modules, "yfinance", mod)
        shares, as_of = sc._yahoo_float("AACG")
        assert shares == pytest.approx(700.0)
        assert as_of is None


# ── fetch_implied_outstanding ───────────────────────────────────────────


def _patch_filing(monkeypatch, filing):
    monkeypatch.setattr(sc, "_latest_periodic_filing", lambda cik: filing)


class TestFetchImpliedOutstanding:
    def test_company_missing_non_fpi_returns_native_total(
            self, temp_db, monkeypatch):
        # No company row -> is_fpi False, ticker None -> non-FPI path.
        facts = {"EntityCommonStockSharesOutstanding":
                 [fact("2026-03-31", 1000.0)]}
        filing = FakeFiling(form="10-Q", accession_no="acc-1",
                            xbrl_obj=FakeXbrl(facts))
        _patch_filing(monkeypatch, filing)
        out = sc.fetch_implied_outstanding(404404, as_of=AS_OF)
        assert out.is_fpi is False
        assert out.native_total == pytest.approx(1000.0)
        assert out.total == pytest.approx(1000.0)
        assert out.source_form == "10-Q"
        assert out.source_accession == "acc-1"
        assert out.source_concept == "EntityCommonStockSharesOutstanding"

    def test_filings_lookup_failed(self, temp_db, monkeypatch, caplog):
        def boom(cik):
            raise RuntimeError("edgar exploded")
        monkeypatch.setattr(sc, "_latest_periodic_filing", boom)
        with caplog.at_level("WARNING", logger="dilution.share_counts"):
            out = sc.fetch_implied_outstanding(42, as_of=AS_OF)
        assert out.total is None
        assert out.native_total is None
        assert out.warnings == ("filings_lookup_failed",)
        # The failure is logged at WARNING with the offending CIK + cause.
        assert any("get_filings failed for CIK 42" in r.getMessage()
                   and "edgar exploded" in r.getMessage()
                   for r in caplog.records)

    def test_filings_lookup_failed_preserves_company_unit(
            self, temp_db, monkeypatch):
        # FPI/ads_ratio surfaced from the DB even though the filing lookup
        # blew up before any XBRL was read.
        temp_db.add_company(cik=909, ticker="AACG", is_fpi=1, ads_ratio=3.0)

        def boom(cik):
            raise RuntimeError("edgar exploded")
        monkeypatch.setattr(sc, "_latest_periodic_filing", boom)
        out = sc.fetch_implied_outstanding(909, as_of=AS_OF)
        assert out.warnings == ("filings_lookup_failed",)
        assert out.is_fpi is True
        assert out.ads_ratio == pytest.approx(3.0)

    def test_no_periodic_filing(self, temp_db, monkeypatch):
        _patch_filing(monkeypatch, None)
        out = sc.fetch_implied_outstanding(1, as_of=AS_OF)
        assert out.total is None
        assert out.warnings == ("no_periodic_filing",)

    def test_xbrl_load_failed_keeps_source(self, temp_db, monkeypatch, caplog):
        filing = FakeFiling(form="10-K", accession_no="acc-9",
                            xbrl_exc=RuntimeError("xbrl boom"))
        _patch_filing(monkeypatch, filing)
        with caplog.at_level("WARNING", logger="dilution.share_counts"):
            out = sc.fetch_implied_outstanding(1, as_of=AS_OF)
        assert out.total is None
        assert out.warnings == ("xbrl_load_failed",)
        assert out.source_form == "10-K"
        assert out.source_accession == "acc-9"
        # accession (not CIK) identifies the bad filing in the log line.
        assert any("xbrl() failed for acc-9" in r.getMessage()
                   for r in caplog.records)

    def test_xbrl_concept_missing(self, temp_db, monkeypatch):
        # No facts for either concept.
        filing = FakeFiling(form="10-Q", accession_no="acc-2",
                            xbrl_obj=FakeXbrl({}))
        _patch_filing(monkeypatch, filing)
        out = sc.fetch_implied_outstanding(1, as_of=AS_OF)
        assert out.total is None
        assert out.warnings == ("xbrl_concept_missing",)
        assert out.source_form == "10-Q"
        assert out.source_accession == "acc-2"

    def test_second_concept_used_when_first_empty(self, temp_db, monkeypatch):
        facts = {
            "EntityCommonStockSharesOutstanding": [],
            "CommonStockSharesOutstanding": [fact("2026-03-31", 42.0)],
        }
        filing = FakeFiling(form="10-Q", accession_no="acc-3",
                            xbrl_obj=FakeXbrl(facts))
        _patch_filing(monkeypatch, filing)
        out = sc.fetch_implied_outstanding(1, as_of=AS_OF)
        assert out.source_concept == "CommonStockSharesOutstanding"
        assert out.native_total == pytest.approx(42.0)

    def test_first_concept_query_raises_recovers_via_second(
            self, temp_db, monkeypatch):
        # _query_latest_period_facts swallows the per-concept exception and
        # returns ([], None); the concept loop then keeps going and picks up
        # the second concept. A raised first concept must NOT abort the fetch.
        facts = {
            "CommonStockSharesOutstanding": [fact("2026-03-31", 77.0)],
        }
        xbrl = FakeXbrl(
            facts, raise_on_concept="EntityCommonStockSharesOutstanding")
        filing = FakeFiling(form="10-Q", accession_no="acc-r2", xbrl_obj=xbrl)
        _patch_filing(monkeypatch, filing)
        out = sc.fetch_implied_outstanding(1, as_of=AS_OF)
        assert out.source_concept == "CommonStockSharesOutstanding"
        assert out.native_total == pytest.approx(77.0)
        assert out.warnings == ()

    def test_none_numeric_value_skipped(self, temp_db, monkeypatch):
        facts = {"EntityCommonStockSharesOutstanding": [
            fact("2026-03-31", None, dim_label="Class A"),
            fact("2026-03-31", 50.0, dim_label="Class B"),
        ]}
        filing = FakeFiling(form="10-Q", accession_no="acc-4",
                            xbrl_obj=FakeXbrl(facts))
        _patch_filing(monkeypatch, filing)
        out = sc.fetch_implied_outstanding(1, as_of=AS_OF)
        # None fact not summed, not added to classes.
        assert out.native_total == pytest.approx(50.0)
        assert len(out.classes) == 1
        assert out.classes[0].label == "Class B"

    def test_multi_class_summation_and_label_fallback(
            self, temp_db, monkeypatch):
        facts = {"EntityCommonStockSharesOutstanding": [
            fact("2026-03-31", 30.0, dim_label="Class A Common"),
            fact("2026-03-31", 3.0, label="Class B Common"),
            fact("2026-03-31", 1.0),  # no label at all -> 'Common Stock'
        ]}
        filing = FakeFiling(form="10-Q", accession_no="acc-5",
                            xbrl_obj=FakeXbrl(facts))
        _patch_filing(monkeypatch, filing)
        out = sc.fetch_implied_outstanding(1, as_of=AS_OF)
        assert out.native_total == pytest.approx(34.0)
        labels = [c.label for c in out.classes]
        assert labels == ["Class A Common", "Class B Common", "Common Stock"]

    def test_class_ratio_exactly_100_does_not_warn(
            self, temp_db, monkeypatch):
        facts = {"EntityCommonStockSharesOutstanding": [
            fact("2026-03-31", 100.0, dim_label="A"),
            fact("2026-03-31", 1.0, dim_label="B"),
        ]}
        filing = FakeFiling(form="10-Q", accession_no="acc-6",
                            xbrl_obj=FakeXbrl(facts))
        _patch_filing(monkeypatch, filing)
        out = sc.fetch_implied_outstanding(1, as_of=AS_OF)
        assert "unequal_class_ratio" not in out.warnings

    def test_class_ratio_above_100_warns(self, temp_db, monkeypatch):
        # Berkshire-like 2770x.
        facts = {"EntityCommonStockSharesOutstanding": [
            fact("2026-03-31", 2770.0, dim_label="A"),
            fact("2026-03-31", 1.0, dim_label="B"),
        ]}
        filing = FakeFiling(form="10-Q", accession_no="acc-7",
                            xbrl_obj=FakeXbrl(facts))
        _patch_filing(monkeypatch, filing)
        out = sc.fetch_implied_outstanding(1, as_of=AS_OF)
        assert "unequal_class_ratio" in out.warnings

    def test_single_class_never_warns_ratio(self, temp_db, monkeypatch):
        facts = {"EntityCommonStockSharesOutstanding": [
            fact("2026-03-31", 5000.0),
        ]}
        filing = FakeFiling(form="10-Q", accession_no="acc-8",
                            xbrl_obj=FakeXbrl(facts))
        _patch_filing(monkeypatch, filing)
        out = sc.fetch_implied_outstanding(1, as_of=AS_OF)
        assert "unequal_class_ratio" not in out.warnings

    def test_zero_negative_values_filtered_from_ratio(
            self, temp_db, monkeypatch):
        # Ratio calc uses only c.value > 0. With one positive value left,
        # len(values) < 2 -> no ratio warning even though a huge gap.
        facts = {"EntityCommonStockSharesOutstanding": [
            fact("2026-03-31", 1000000.0, dim_label="A"),
            fact("2026-03-31", 0.0, dim_label="B"),
            fact("2026-03-31", -5.0, dim_label="C"),
        ]}
        filing = FakeFiling(form="10-Q", accession_no="acc-zn",
                            xbrl_obj=FakeXbrl(facts))
        _patch_filing(monkeypatch, filing)
        out = sc.fetch_implied_outstanding(1, as_of=AS_OF)
        assert "unequal_class_ratio" not in out.warnings
        # native_total still sums everything (incl. 0 and -5).
        assert out.native_total == pytest.approx(1000000.0 - 5.0)

    def test_staleness_exactly_180_no_warn(self, temp_db, monkeypatch):
        # period = as_of - 180 days = 2025-12-12.
        period = (date(2025, 12, 12)).isoformat()
        facts = {"EntityCommonStockSharesOutstanding": [
            fact(period, 100.0),
        ]}
        filing = FakeFiling(form="10-Q", accession_no="acc-s1",
                            xbrl_obj=FakeXbrl(facts))
        _patch_filing(monkeypatch, filing)
        out = sc.fetch_implied_outstanding(1, as_of=AS_OF)
        assert out.stale_days == 180
        assert not any(w.startswith("stale_") for w in out.warnings)

    def test_staleness_181_warns(self, temp_db, monkeypatch):
        period = (date(2025, 12, 11)).isoformat()  # 181 days before AS_OF
        facts = {"EntityCommonStockSharesOutstanding": [
            fact(period, 100.0),
        ]}
        filing = FakeFiling(form="10-Q", accession_no="acc-s2",
                            xbrl_obj=FakeXbrl(facts))
        _patch_filing(monkeypatch, filing)
        out = sc.fetch_implied_outstanding(1, as_of=AS_OF)
        assert out.stale_days == 181
        assert "stale_181d" in out.warnings

    def test_period_none_no_staleness(self, temp_db, monkeypatch):
        # All facts lack period_instant -> concept query yields nothing for
        # that period grouping. To get facts but None period we must have a
        # fact list that produces facts yet no period. The module only
        # returns facts WITH a period, so to exercise period_date None we
        # rely on _query returning ([],None) -> concept_missing. Instead,
        # patch _query_latest_period_facts to hand back facts + None period.
        facts_list = [fact("ignored", 100.0)]
        monkeypatch.setattr(
            sc, "_query_latest_period_facts",
            lambda xbrl, concept: (facts_list, None))
        filing = FakeFiling(form="10-Q", accession_no="acc-pn",
                            xbrl_obj=FakeXbrl({}))
        _patch_filing(monkeypatch, filing)
        out = sc.fetch_implied_outstanding(1, as_of=AS_OF)
        assert out.stale_days is None
        assert out.as_of is None
        assert not any(w.startswith("stale_") for w in out.warnings)
        assert out.native_total == pytest.approx(100.0)

    def test_result_as_of_is_period_for_non_fpi(self, temp_db, monkeypatch):
        facts = {"EntityCommonStockSharesOutstanding": [
            fact("2026-03-31", 100.0),
        ]}
        filing = FakeFiling(form="10-Q", accession_no="acc-ra",
                            xbrl_obj=FakeXbrl(facts))
        _patch_filing(monkeypatch, filing)
        out = sc.fetch_implied_outstanding(1, as_of=AS_OF)
        assert out.as_of == date(2026, 3, 31)

    # ── FPI branches ────────────────────────────────────────────────

    def _fpi_filing(self, monkeypatch, period="2025-06-30", value=200.0):
        facts = {"EntityCommonStockSharesOutstanding": [fact(period, value)]}
        filing = FakeFiling(form="20-F", accession_no="fpi-acc",
                            xbrl_obj=FakeXbrl(facts))
        _patch_filing(monkeypatch, filing)

    def test_fpi_yahoo_returns_shares(self, temp_db, monkeypatch):
        temp_db.add_company(cik=700, ticker="AACG", is_fpi=1, ads_ratio=4.0)
        self._fpi_filing(monkeypatch)
        monkeypatch.setattr(
            sc, "_yahoo_shares",
            lambda t: (15040000.0, date(2026, 5, 20)))
        out = sc.fetch_implied_outstanding(700, as_of=AS_OF)
        assert out.is_fpi is True
        assert out.total == pytest.approx(15040000.0)
        assert out.source_form == "yahoo"
        assert out.source_accession is None
        assert out.source_concept == "yfinance.sharesOutstanding"
        assert out.as_of == date(2026, 5, 20)
        # native_total stays the XBRL number.
        assert out.native_total == pytest.approx(200.0)

    def test_fpi_yahoo_strips_then_recomputes_staleness(
            self, temp_db, monkeypatch):
        # XBRL period is very stale (would warn), but yahoo override gives
        # a fresh date -> pre-existing stale_* stripped, recomputed fresh.
        temp_db.add_company(cik=701, ticker="AACG", is_fpi=1, ads_ratio=4.0)
        self._fpi_filing(monkeypatch, period="2024-01-01")  # very stale
        monkeypatch.setattr(
            sc, "_yahoo_shares",
            lambda t: (999.0, date(2026, 6, 1)))  # 9 days stale -> no warn
        out = sc.fetch_implied_outstanding(701, as_of=AS_OF)
        assert not any(w.startswith("stale_") for w in out.warnings)
        assert out.stale_days == 9

    def test_fpi_yahoo_override_keeps_fresh_stale_warning(
            self, temp_db, monkeypatch):
        # yahoo date itself is >180d stale -> a fresh stale_ warning added.
        temp_db.add_company(cik=702, ticker="AACG", is_fpi=1, ads_ratio=4.0)
        self._fpi_filing(monkeypatch, period="2026-03-31")
        monkeypatch.setattr(
            sc, "_yahoo_shares",
            lambda t: (999.0, date(2025, 1, 1)))  # 525 days before AS_OF
        out = sc.fetch_implied_outstanding(702, as_of=AS_OF)
        stale_warns = [w for w in out.warnings if w.startswith("stale_")]
        assert len(stale_warns) == 1
        assert stale_warns[0] == f"stale_{(AS_OF - date(2025,1,1)).days}d"

    def test_fpi_yahoo_override_preserves_non_stale_warning(
            self, temp_db, monkeypatch):
        # The yahoo override strips ONLY stale_* warnings, not the whole
        # list: a co-existing unequal_class_ratio warning must survive.
        temp_db.add_company(cik=706, ticker="AACG", is_fpi=1, ads_ratio=4.0)
        facts = {"EntityCommonStockSharesOutstanding": [
            fact("2026-03-31", 2770.0, dim_label="A"),  # >100x ratio
            fact("2026-03-31", 1.0, dim_label="B"),
        ]}
        filing = FakeFiling(form="20-F", accession_no="fpi-acc-r",
                            xbrl_obj=FakeXbrl(facts))
        _patch_filing(monkeypatch, filing)
        monkeypatch.setattr(
            sc, "_yahoo_shares",
            lambda t: (999.0, date(2026, 6, 1)))  # fresh, no stale warn
        out = sc.fetch_implied_outstanding(706, as_of=AS_OF)
        assert out.total == pytest.approx(999.0)
        assert "unequal_class_ratio" in out.warnings
        assert not any(w.startswith("stale_") for w in out.warnings)
        # native_total is unaffected by the yahoo override.
        assert out.native_total == pytest.approx(2771.0)

    def test_fpi_yahoo_none_with_ads_ratio_fallback(
            self, temp_db, monkeypatch):
        temp_db.add_company(cik=703, ticker="AACG", is_fpi=1, ads_ratio=4.0)
        self._fpi_filing(monkeypatch, value=800.0)
        monkeypatch.setattr(sc, "_yahoo_shares", lambda t: (None, None))
        out = sc.fetch_implied_outstanding(703, as_of=AS_OF)
        assert out.total == pytest.approx(800.0 / 4.0)
        assert "yahoo_unavailable_xbrl_fallback" in out.warnings
        # source remains the XBRL filing, not yahoo.
        assert out.source_form == "20-F"

    def test_fpi_yahoo_none_ads_ratio_none(self, temp_db, monkeypatch):
        temp_db.add_company(cik=704, ticker="AACG", is_fpi=1, ads_ratio=None)
        self._fpi_filing(monkeypatch)
        monkeypatch.setattr(sc, "_yahoo_shares", lambda t: (None, None))
        out = sc.fetch_implied_outstanding(704, as_of=AS_OF)
        assert out.total is None
        assert "fpi_ads_ratio_missing" in out.warnings

    def test_fpi_yahoo_none_ads_ratio_zero(self, temp_db, monkeypatch):
        temp_db.add_company(cik=705, ticker="AACG", is_fpi=1, ads_ratio=0.0)
        self._fpi_filing(monkeypatch)
        monkeypatch.setattr(sc, "_yahoo_shares", lambda t: (None, None))
        out = sc.fetch_implied_outstanding(705, as_of=AS_OF)
        assert out.total is None
        assert "fpi_ads_ratio_missing" in out.warnings

    def test_non_fpi_ignores_yahoo_and_ads_ratio(self, temp_db, monkeypatch):
        temp_db.add_company(cik=707, ticker="GENK", is_fpi=0, ads_ratio=5.0)
        facts = {"EntityCommonStockSharesOutstanding": [
            fact("2026-03-31", 333.0),
        ]}
        filing = FakeFiling(form="10-Q", accession_no="acc-nf",
                            xbrl_obj=FakeXbrl(facts))
        _patch_filing(monkeypatch, filing)

        def _should_not_be_called(t):
            raise AssertionError("yahoo should not be called for non-FPI")
        monkeypatch.setattr(sc, "_yahoo_shares", _should_not_be_called)
        out = sc.fetch_implied_outstanding(707, as_of=AS_OF)
        assert out.total == pytest.approx(333.0)  # native, not /5.0

    def test_fpi_no_ticker_skips_yahoo_and_falls_to_ads_ratio(
            self, temp_db, monkeypatch):
        # Defensive `_yahoo_shares(ticker) if ticker else (None, None)` guard:
        # an FPI with ticker=None must NOT call yahoo and must drop straight
        # into the ads_ratio fallback. This DB state (is_fpi + NULL ticker)
        # is impossible against the NOT NULL schema, but the code branch is
        # real -- exercised here by faking _company_unit, not staging a row.
        monkeypatch.setattr(sc, "_company_unit", lambda cik: (True, 4.0, None))
        self._fpi_filing(monkeypatch, value=800.0)

        def _should_not_be_called(t):
            raise AssertionError("yahoo must be skipped when ticker is None")
        monkeypatch.setattr(sc, "_yahoo_shares", _should_not_be_called)
        out = sc.fetch_implied_outstanding(1, as_of=AS_OF)
        assert out.is_fpi is True
        assert out.total == pytest.approx(800.0 / 4.0)
        assert "yahoo_unavailable_xbrl_fallback" in out.warnings

    def test_fpi_no_ticker_no_ads_ratio_returns_none(
            self, temp_db, monkeypatch):
        # Same defensive guard, but no ads_ratio either -> total None,
        # 'fpi_ads_ratio_missing'. Locks in that the ticker=None ternary
        # short-circuits BEFORE the yahoo call.
        monkeypatch.setattr(sc, "_company_unit", lambda cik: (True, None, None))
        self._fpi_filing(monkeypatch, value=800.0)

        def _should_not_be_called(t):
            raise AssertionError("yahoo must be skipped when ticker is None")
        monkeypatch.setattr(sc, "_yahoo_shares", _should_not_be_called)
        out = sc.fetch_implied_outstanding(1, as_of=AS_OF)
        assert out.total is None
        assert "fpi_ads_ratio_missing" in out.warnings

    def test_facts_present_all_numeric_value_none_zero_total(
            self, temp_db, monkeypatch):
        # Facts exist for the latest period but EVERY numeric_value is None.
        # `facts` is truthy (so NOT xbrl_concept_missing), the None values are
        # skipped, and native_total stays 0.0 -> total 0.0, empty classes.
        # A subtle boundary: an empty-but-present fact set yields a real 0,
        # not a failure code.
        facts = {"EntityCommonStockSharesOutstanding": [
            fact("2026-03-31", None, dim_label="Class A"),
            fact("2026-03-31", None, dim_label="Class B"),
        ]}
        filing = FakeFiling(form="10-Q", accession_no="acc-an",
                            xbrl_obj=FakeXbrl(facts))
        _patch_filing(monkeypatch, filing)
        out = sc.fetch_implied_outstanding(1, as_of=AS_OF)
        assert out.native_total == pytest.approx(0.0)
        assert out.total == pytest.approx(0.0)
        assert out.classes == ()
        assert "xbrl_concept_missing" not in out.warnings
        # period still parsed -> as_of and (non-stale) stale_days populated.
        assert out.as_of == date(2026, 3, 31)

    def test_as_of_defaults_to_today(self, temp_db, monkeypatch):
        # Use a recent period; with default today() it should not be stale.
        recent = date.today().isoformat()
        facts = {"EntityCommonStockSharesOutstanding": [fact(recent, 10.0)]}
        filing = FakeFiling(form="10-Q", accession_no="acc-td",
                            xbrl_obj=FakeXbrl(facts))
        _patch_filing(monkeypatch, filing)
        out = sc.fetch_implied_outstanding(1)  # no as_of
        assert out.stale_days == 0
        assert not any(w.startswith("stale_") for w in out.warnings)


# ── fetch_float ─────────────────────────────────────────────────────────


class TestFetchFloat:
    def test_no_ticker(self, temp_db, monkeypatch):
        # No company row -> ticker None.
        out = sc.fetch_float(123, as_of=AS_OF)
        assert out.shares is None
        assert out.warnings == ("no_ticker",)

    def test_yahoo_unavailable(self, temp_db, monkeypatch):
        temp_db.add_company(cik=800, ticker="AACG")
        monkeypatch.setattr(sc, "_yahoo_float", lambda t: (None, None))
        out = sc.fetch_float(800, as_of=AS_OF)
        assert out.shares is None
        assert out.warnings == ("yahoo_unavailable",)

    def test_yahoo_returns_shares(self, temp_db, monkeypatch):
        temp_db.add_company(cik=801, ticker="AACG")
        monkeypatch.setattr(
            sc, "_yahoo_float", lambda t: (15040000.0, date(2026, 5, 1)))
        out = sc.fetch_float(801, as_of=AS_OF)
        assert out.shares == pytest.approx(15040000.0)
        assert out.as_of == date(2026, 5, 1)
        assert out.source == "yfinance.floatShares"
        assert out.stale_days == (AS_OF - date(2026, 5, 1)).days
        assert out.warnings == ()

    def test_src_as_of_none_no_stale(self, temp_db, monkeypatch):
        temp_db.add_company(cik=802, ticker="AACG")
        monkeypatch.setattr(sc, "_yahoo_float", lambda t: (500.0, None))
        out = sc.fetch_float(802, as_of=AS_OF)
        assert out.shares == pytest.approx(500.0)
        assert out.stale_days is None
        assert out.warnings == ()

    def test_stale_exactly_180_no_warn(self, temp_db, monkeypatch):
        temp_db.add_company(cik=803, ticker="AACG")
        src = date(2025, 12, 12)  # exactly 180 days before AS_OF
        monkeypatch.setattr(sc, "_yahoo_float", lambda t: (1.0, src))
        out = sc.fetch_float(803, as_of=AS_OF)
        assert out.stale_days == 180
        assert out.warnings == ()

    def test_stale_181_warns(self, temp_db, monkeypatch):
        temp_db.add_company(cik=804, ticker="AACG")
        src = date(2025, 12, 11)  # 181 days before AS_OF
        monkeypatch.setattr(sc, "_yahoo_float", lambda t: (1.0, src))
        out = sc.fetch_float(804, as_of=AS_OF)
        assert out.stale_days == 181
        assert out.warnings == ("stale_181d",)

    def test_future_src_as_of_negative_stale_no_warn(
            self, temp_db, monkeypatch):
        # A source date AFTER as_of yields a negative stale_days. -21 > 180
        # is False, so no stale warning is emitted (guards a sign bug).
        temp_db.add_company(cik=806, ticker="AACG")
        monkeypatch.setattr(sc, "_yahoo_float",
                            lambda t: (1000.0, date(2026, 7, 1)))
        out = sc.fetch_float(806, as_of=AS_OF)
        assert out.stale_days == (AS_OF - date(2026, 7, 1)).days
        assert out.stale_days < 0
        assert out.warnings == ()

    def test_as_of_defaults_to_today(self, temp_db, monkeypatch):
        temp_db.add_company(cik=805, ticker="AACG")
        monkeypatch.setattr(
            sc, "_yahoo_float", lambda t: (1.0, date.today()))
        out = sc.fetch_float(805)
        assert out.stale_days == 0
        assert out.warnings == ()


# ── cached wrappers ─────────────────────────────────────────────────────


class TestCachedWrappers:
    def test_implied_cache_hit_calls_once(self, temp_db, monkeypatch):
        calls = []

        def fake_fetch(cik, *, as_of=None):
            calls.append((cik, as_of))
            return sc.ImpliedOutstanding(total=1.0, native_total=1.0)
        monkeypatch.setattr(sc, "fetch_implied_outstanding", fake_fetch)
        sc._cached.cache_clear()
        a = sc.fetch_implied_outstanding_cached(1, as_of=AS_OF)
        b = sc.fetch_implied_outstanding_cached(1, as_of=AS_OF)
        assert a is b
        assert len(calls) == 1

    def test_implied_different_as_of_separate_entries(
            self, temp_db, monkeypatch):
        calls = []

        def fake_fetch(cik, *, as_of=None):
            calls.append(as_of)
            return sc.ImpliedOutstanding(total=1.0, native_total=1.0)
        monkeypatch.setattr(sc, "fetch_implied_outstanding", fake_fetch)
        sc._cached.cache_clear()
        sc.fetch_implied_outstanding_cached(1, as_of=date(2026, 6, 10))
        sc.fetch_implied_outstanding_cached(1, as_of=date(2026, 6, 9))
        assert len(calls) == 2

    def test_implied_as_of_none_uses_today_iso(self, temp_db, monkeypatch):
        seen = []

        def fake_fetch(cik, *, as_of=None):
            seen.append(as_of)
            return sc.ImpliedOutstanding(total=1.0, native_total=1.0)
        monkeypatch.setattr(sc, "fetch_implied_outstanding", fake_fetch)
        sc._cached.cache_clear()
        sc.fetch_implied_outstanding_cached(1)
        # _cached coerces the iso string back to a date == today.
        assert seen == [date.today()]

    def test_float_cache_hit_calls_once(self, temp_db, monkeypatch):
        calls = []

        def fake_fetch(cik, *, as_of=None):
            calls.append((cik, as_of))
            return sc.ImpliedFloat(shares=2.0)
        monkeypatch.setattr(sc, "fetch_float", fake_fetch)
        sc._cached_float.cache_clear()
        a = sc.fetch_float_cached(5, as_of=AS_OF)
        b = sc.fetch_float_cached(5, as_of=AS_OF)
        assert a is b
        assert len(calls) == 1

    def test_float_as_of_none_uses_today(self, temp_db, monkeypatch):
        seen = []

        def fake_fetch(cik, *, as_of=None):
            seen.append(as_of)
            return sc.ImpliedFloat(shares=2.0)
        monkeypatch.setattr(sc, "fetch_float", fake_fetch)
        sc._cached_float.cache_clear()
        sc.fetch_float_cached(5)
        assert seen == [date.today()]

    def test_implied_cache_distinct_cik_separate_entries(
            self, temp_db, monkeypatch):
        # Different CIK with the same as_of must NOT collide in the cache.
        calls = []

        def fake_fetch(cik, *, as_of=None):
            calls.append(cik)
            return sc.ImpliedOutstanding(total=float(cik), native_total=1.0)
        monkeypatch.setattr(sc, "fetch_implied_outstanding", fake_fetch)
        sc._cached.cache_clear()
        a = sc.fetch_implied_outstanding_cached(11, as_of=AS_OF)
        b = sc.fetch_implied_outstanding_cached(22, as_of=AS_OF)
        assert a.total == pytest.approx(11.0)
        assert b.total == pytest.approx(22.0)
        assert calls == [11, 22]

    def test_implied_cache_returns_genuine_fetched_result(
            self, temp_db, monkeypatch):
        # End-to-end through the REAL fetch_implied_outstanding (only the
        # edgar/db seams faked): proves the cache hands back the actual
        # computed object, not a sentinel, and keys correctly on cik.
        temp_db.add_company(cik=950, ticker="GENK", is_fpi=0)
        facts = {"EntityCommonStockSharesOutstanding": [
            fact("2026-03-31", 30.0, dim_label="Class A"),
            fact("2026-03-31", 3.0, dim_label="Class B"),
        ]}
        filing = FakeFiling(form="10-Q", accession_no="acc-cache",
                            xbrl_obj=FakeXbrl(facts))
        monkeypatch.setattr(sc, "_latest_periodic_filing",
                            lambda cik: filing)
        sc._cached.cache_clear()
        a = sc.fetch_implied_outstanding_cached(950, as_of=AS_OF)
        b = sc.fetch_implied_outstanding_cached(950, as_of=AS_OF)
        assert a is b  # cache hit returns the identical object
        assert a.native_total == pytest.approx(33.0)
        assert a.source_accession == "acc-cache"


# ── _ensure_identity ────────────────────────────────────────────────────


class TestEnsureIdentity:
    def test_first_call_invokes_then_idempotent(self, monkeypatch):
        captured = []
        monkeypatch.setattr(sc, "set_identity",
                            lambda ident: captured.append(ident))
        sc._IDENTITY_SET = False
        try:
            sc._ensure_identity()
            sc._ensure_identity()
            assert len(captured) == 1  # idempotent: second call no-ops
        finally:
            sc._IDENTITY_SET = True

    def test_uses_config_edgar_identity_when_present(self, monkeypatch):
        captured = []
        monkeypatch.setattr(sc, "set_identity",
                            lambda ident: captured.append(ident))
        monkeypatch.setattr(sc.config, "EDGAR_IDENTITY",
                            "me me@example.com", raising=False)
        sc._IDENTITY_SET = False
        try:
            sc._ensure_identity()
            assert captured == ["me me@example.com"]
        finally:
            sc._IDENTITY_SET = True

    def test_uses_default_when_config_missing(self, monkeypatch):
        captured = []
        monkeypatch.setattr(sc, "set_identity",
                            lambda ident: captured.append(ident))
        monkeypatch.delattr(sc.config, "EDGAR_IDENTITY", raising=False)
        sc._IDENTITY_SET = False
        try:
            sc._ensure_identity()
            assert captured == ["dilution-tracker contact@example.com"]
        finally:
            sc._IDENTITY_SET = True


# ── dataclass models ────────────────────────────────────────────────────


class TestDataclasses:
    def test_implied_outstanding_defaults(self):
        o = sc.ImpliedOutstanding(total=1.0, native_total=2.0)
        assert o.classes == ()
        assert o.as_of is None
        assert o.warnings == ()
        assert o.is_fpi is False
        assert o.ads_ratio is None

    def test_implied_outstanding_frozen(self):
        import dataclasses
        o = sc.ImpliedOutstanding(total=1.0, native_total=2.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            o.total = 3.0  # frozen dataclass

    def test_class_count_fields(self):
        c = sc.ClassCount(label="Class A", value=10.0)
        assert c.label == "Class A"
        assert c.value == pytest.approx(10.0)

    def test_implied_float_defaults(self):
        f = sc.ImpliedFloat(shares=None)
        assert f.as_of is None
        assert f.source is None
        assert f.stale_days is None
        assert f.warnings == ()
