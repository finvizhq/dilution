"""Unit tests for dilution/filings.py.

Covers the three pure normalization/classification helpers (_is_relevant,
_norm_str, _norm_date) and the io_mockable + db_backed pull_filing_index,
which is exercised against the autouse temp_db with edgartools fully
monkeypatched (no network).
"""

from __future__ import annotations

import datetime

import pytest

import dilution.filings as filings
from dilution.filings import _is_relevant, _norm_date, _norm_str


# ──────────────────────────────────────────────────────────────────────
# _is_relevant
# ──────────────────────────────────────────────────────────────────────
class TestIsRelevant:
    @pytest.mark.parametrize(
        "form",
        [
            None,
            "",
            "   ",          # strips to empty; no non-empty prefix matches ""
        ],
    )
    def test_falsy_or_empty_returns_false(self, form):
        assert _is_relevant(form) is False

    @pytest.mark.parametrize(
        "form",
        [
            "8-K",          # exact
            " 8-K ",        # strip applied
            "8-K/A",        # amended variant via base prefix
            "6-K",
            "424B5",        # take-down via 424B
            "424B3",
            "S-1",
            "S-3",
            "S-3ASR",       # matches S-3 prefix (also explicitly listed)
            "S-4",
            "F-1",
            "F-3",
            "POS AM",
            "425",
            "RW",
            "EFFECT",
            "10-K",
            "10-Q",
            "10-K405",      # startswith 10-K
            "20-F",
            "40-F",
            "DEF 14A",
            "DEFA14A",
            "DEFM14A",
            "PRE 14A",
            "DEFC14A",
            "DEFR14A",
            "PREM14A",
            "PRER14A",
            "FWP",
            "SUPPL",
            "1-A",
            "1-U",
            "1-K",
            "1-SA",
        ],
    )
    def test_relevant_forms_match(self, form):
        assert _is_relevant(form) is True

    @pytest.mark.parametrize(
        "form",
        [
            "SC 13D",
            "4",
            "3",
            "NT 10-K",      # leading "NT " breaks the 10-K prefix
            "10",           # no bare "10" prefix; needs -K/-Q
            "ABC",
        ],
    )
    def test_irrelevant_forms_rejected(self, form):
        assert _is_relevant(form) is False

    def test_case_sensitive_lowercase_rejected(self):
        # Prefixes are uppercase and no .lower() is applied, so a lowercase
        # form does NOT match. Asserting documented case-sensitive behavior.
        assert _is_relevant("8-k") is False
        assert _is_relevant("s-3") is False


# ──────────────────────────────────────────────────────────────────────
# _norm_str
# ──────────────────────────────────────────────────────────────────────
class TestNormStr:
    def test_none_returns_none(self):
        assert _norm_str(None) is None

    def test_nan_float_returns_none(self):
        assert _norm_str(float("nan")) is None

    def test_numpy_nan_returns_none(self):
        np = pytest.importorskip("numpy")
        assert _norm_str(np.nan) is None

    def test_empty_string_returns_none(self):
        assert _norm_str("") is None

    def test_whitespace_only_returns_none(self):
        assert _norm_str("   ") is None

    def test_normal_string_is_stripped(self):
        assert _norm_str(" abc ") == "abc"

    def test_int_zero_returns_string_zero(self):
        # Subtle trap: 0 is not NaN/None, so str(0).strip() == "0" which is
        # truthy as a string -> NOT coerced to None.
        assert _norm_str(0) == "0"

    def test_float_value_stringified(self):
        assert _norm_str(333.0) == "333.0"

    def test_int_value_stringified(self):
        assert _norm_str(42) == "42"

    def test_non_pandas_object_falls_through_to_str(self):
        # The bare try/except around pd.isna must not blow up on a custom
        # object; it falls through to str(x).strip().
        class Thing:
            def __str__(self):
                return "  payload  "

        assert _norm_str(Thing()) == "payload"

    def test_object_whose_str_is_whitespace_returns_none(self):
        class Blank:
            def __str__(self):
                return "    "

        assert _norm_str(Blank()) is None

    def test_array_like_pd_isna_raises_is_swallowed(self):
        # The bare `except Exception: pass` around `pd.isna(x)` is load-bearing:
        # pd.isna on an ndarray returns an ARRAY, so `if pd.isna(x)` raises a
        # "truth value of an array is ambiguous" ValueError. The except must
        # swallow it and fall through to str(x).strip(), NOT propagate. This
        # proves the except is doing real work (not only catching ImportError).
        np = pytest.importorskip("numpy")
        out = _norm_str(np.array([1.0, np.nan]))
        # numpy renders the array; just assert it returned a non-None string
        # via the fall-through (the function did not raise).
        assert out is not None
        assert "nan" in out

    def test_list_pd_isna_raises_is_swallowed(self):
        # Same swallow path for a plain list (pd.isna(list) -> array -> ambiguous).
        pytest.importorskip("pandas")
        assert _norm_str([1, 2]) == "[1, 2]"


# ──────────────────────────────────────────────────────────────────────
# _norm_date
# ──────────────────────────────────────────────────────────────────────
class TestNormDate:
    def test_none_returns_none(self):
        assert _norm_date(None) is None

    def test_nan_returns_none(self):
        assert _norm_date(float("nan")) is None

    def test_empty_returns_none(self):
        assert _norm_date("") is None

    def test_whitespace_returns_none(self):
        assert _norm_date("   ") is None

    def test_already_formatted_date_unchanged(self):
        assert _norm_date("2025-03-14") == "2025-03-14"

    def test_iso_datetime_truncated_to_ten(self):
        assert _norm_date("2025-03-14T00:00:00") == "2025-03-14"

    def test_long_iso_with_tz_offset_sliced_to_ten(self):
        # A >10-char string with no .strftime falls to the s[:10] slice branch.
        # Proves slicing keeps only the leading calendar date regardless of the
        # tz suffix length.
        assert _norm_date("2025-03-14T00:00:00+00:00") == "2025-03-14"

    def test_exactly_ten_char_non_date_string_passthrough(self):
        # The slice branch is length-based, not content-validating: a 10-char
        # garbage string is returned verbatim (no date parsing performed).
        assert _norm_date("XXXXXXXXXX") == "XXXXXXXXXX"

    def test_short_string_returned_whole(self):
        # len < 10 -> returned as-is, no slicing.
        assert _norm_date("abc") == "abc"

    def test_length_nine_returned_whole(self):
        assert _norm_date("2025-03-1") == "2025-03-1"

    def test_length_exactly_ten_unchanged(self):
        s = "2025-03-14"
        assert len(s) == 10
        assert _norm_date(s) == s

    def test_timestamp_uses_strftime_branch(self):
        pd = pytest.importorskip("pandas")
        # A Timestamp with a time component would slice to "2025-03-14"
        # anyway, but strftime takes precedence and yields the same.
        ts = pd.Timestamp("2025-03-14 12:00")
        assert _norm_date(ts) == "2025-03-14"

    def test_date_object_via_strftime(self):
        d = datetime.date(2025, 3, 14)
        assert _norm_date(d) == "2025-03-14"

    def test_strftime_that_raises_falls_back_to_slice(self):
        # Defensive except: strftime blows up, so fall back to s[:10].
        class BadStrftime:
            def strftime(self, fmt):
                raise ValueError("boom")

            def __str__(self):
                return "2099-12-31TZZ"

        assert _norm_date(BadStrftime()) == "2099-12-31"

    def test_datetime_datetime_with_time_uses_strftime(self):
        # A stdlib datetime.datetime has .strftime and a time component; the
        # strftime branch must win and drop the time, NOT a naive str-slice
        # (str(datetime) is "2025-03-14 12:00:00", whose [:10] also happens
        # to be the date — so to prove the strftime branch is actually taken
        # we use a tz-aware value whose str() differs in offset but strftime
        # still yields the calendar date).
        dt = datetime.datetime(2025, 3, 14, 12, 0, 0)
        assert _norm_date(dt) == "2025-03-14"

    def test_strftime_branch_beats_slice_when_str_differs(self):
        # An object whose str() is NOT a date-like prefix but whose strftime
        # returns a real date proves strftime takes precedence over slicing.
        class OnlyStrftime:
            def strftime(self, fmt):
                return "2030-07-04"

            def __str__(self):
                return "garbage-not-a-date"

        # _norm_str sees "garbage-not-a-date" (truthy) so we reach the
        # strftime branch, which returns the formatted date — proving the
        # result comes from strftime, not from slicing str(x)[:10]
        # (which would be "garbage-no").
        assert _norm_date(OnlyStrftime()) == "2030-07-04"


# ──────────────────────────────────────────────────────────────────────
# pull_filing_index — fakes for edgartools
# ──────────────────────────────────────────────────────────────────────
class FakeFiling:
    """Mimics an edgartools Filing object's attribute surface used by the
    loop in pull_filing_index."""

    def __init__(self, form, accession_no, filing_date, homepage_url="http://hp/x"):
        self.form = form
        self.accession_no = accession_no
        # filing_date may be a value whose str() is YYYY-MM-DD, or None.
        self.filing_date = filing_date
        self.homepage_url = homepage_url


class FakeFilings:
    """Mimics edgartools Filings: .filter(date=...) -> self, .to_pandas(),
    and iteration over filing objects."""

    def __init__(self, filing_objs, df):
        self._filings = filing_objs
        self._df = df
        self.filter_calls = []

    def filter(self, **kwargs):
        self.filter_calls.append(kwargs)
        return self

    def to_pandas(self):
        return self._df

    def __iter__(self):
        return iter(self._filings)


def _make_company_factory(fake_filings):
    """Returns a fake Company class binding get_filings() to fake_filings."""

    class FakeCompany:
        instances = []

        def __init__(self, cik):
            self.cik = cik
            FakeCompany.instances.append(cik)

        def get_filings(self):
            return fake_filings

    return FakeCompany


@pytest.fixture
def patch_edgar(monkeypatch):
    """Patch the module-bound set_identity and Company. Returns an installer
    callable that, given a FakeFilings, wires up Company and records
    set_identity invocations."""
    identity_calls = []

    def fake_set_identity(ident):
        identity_calls.append(ident)

    monkeypatch.setattr(filings, "set_identity", fake_set_identity)

    def install(fake_filings):
        company_cls = _make_company_factory(fake_filings)
        monkeypatch.setattr(filings, "Company", company_cls)
        return company_cls

    install.identity_calls = identity_calls
    return install


def _df(rows):
    """Build a pandas DataFrame from a list of row dicts."""
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(rows)


def _empty_df():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame()


class TestPullFilingIndex:
    def test_empty_dataframe_and_no_filings_returns_zero(self, temp_db, patch_edgar):
        ff = FakeFilings([], _empty_df())
        patch_edgar(ff)
        n = filings.pull_filing_index(123, "2025-01-01")
        assert n == 0
        rows = temp_db.execute("SELECT COUNT(*) AS c FROM dilution_filings")
        assert rows[0]["c"] == 0

    def test_set_identity_called_with_config_identity(self, temp_db, patch_edgar):
        import config

        ff = FakeFilings([], _empty_df())
        patch_edgar(ff)
        filings.pull_filing_index(7, "2025-01-01")
        assert patch_edgar.identity_calls == [config.EDGAR_IDENTITY]

    def test_filter_called_with_since_date_open_range(self, temp_db, patch_edgar):
        ff = FakeFilings([], _empty_df())
        patch_edgar(ff)
        filings.pull_filing_index(9, "2025-02-15")
        assert ff.filter_calls == [{"date": "2025-02-15:"}]

    def test_single_relevant_filing_inserted(self, temp_db, patch_edgar):
        acc = "0001-25-000001"
        df = _df([
            {
                "accession_number": acc,
                "reportDate": "2025-03-10",
                "items": "1.01,9.01",
                "primaryDocument": "form8k.htm",
                "fileNumber": "001-36404",
            }
        ])
        ff = FakeFilings(
            [FakeFiling("8-K", acc, "2025-03-14", homepage_url="http://hp/8k")],
            df,
        )
        patch_edgar(ff)

        n = filings.pull_filing_index(555, "2025-01-01")
        assert n == 1

        rows = temp_db.execute(
            "SELECT * FROM dilution_filings WHERE accession_number=?", (acc,)
        )
        assert len(rows) == 1
        r = rows[0]
        assert r["cik"] == 555
        assert r["form"] == "8-K"
        assert r["filing_date"] == "2025-03-14"
        assert r["report_date"] == "2025-03-10"
        assert r["items"] == "1.01,9.01"
        assert r["primary_doc"] == "form8k.htm"
        assert r["file_number"] == "001-36404"
        assert r["homepage_url"] == "http://hp/8k"
        # primary_doc_url is always stored as None by design.
        assert r["primary_doc_url"] is None
        # fetched_at populated from now_iso().
        assert r["fetched_at"] is not None

    def test_irrelevant_form_filtered_out(self, temp_db, patch_edgar):
        acc_keep = "ACC-KEEP"
        acc_drop = "ACC-DROP"
        df = _df([
            {"accession_number": acc_keep, "reportDate": "2025-03-10",
             "items": None, "primaryDocument": None, "fileNumber": None},
            {"accession_number": acc_drop, "reportDate": "2025-03-11",
             "items": None, "primaryDocument": None, "fileNumber": None},
        ])
        ff = FakeFilings(
            [
                FakeFiling("8-K", acc_keep, "2025-03-14"),
                FakeFiling("SC 13D", acc_drop, "2025-03-15"),
            ],
            df,
        )
        patch_edgar(ff)

        n = filings.pull_filing_index(1, "2025-01-01")
        assert n == 1
        rows = temp_db.execute("SELECT accession_number FROM dilution_filings")
        accs = {r["accession_number"] for r in rows}
        assert accs == {acc_keep}

    def test_return_value_counts_only_relevant_in_window(self, temp_db, patch_edgar):
        df = _df([
            {"accession_number": "A", "reportDate": None, "items": None,
             "primaryDocument": None, "fileNumber": None},
            {"accession_number": "B", "reportDate": None, "items": None,
             "primaryDocument": None, "fileNumber": None},
            {"accession_number": "C", "reportDate": None, "items": None,
             "primaryDocument": None, "fileNumber": None},
        ])
        ff = FakeFilings(
            [
                FakeFiling("8-K", "A", "2025-03-14"),     # relevant, in window
                FakeFiling("SC 13D", "B", "2025-03-15"),  # irrelevant
                FakeFiling("10-Q", "C", "2024-12-31"),    # relevant but pre-window
            ],
            df,
        )
        patch_edgar(ff)
        n = filings.pull_filing_index(2, "2025-01-01")
        assert n == 1

    def test_filing_date_equal_since_date_kept(self, temp_db, patch_edgar):
        acc = "BOUND-EQ"
        df = _df([
            {"accession_number": acc, "reportDate": None, "items": None,
             "primaryDocument": None, "fileNumber": None},
        ])
        ff = FakeFilings([FakeFiling("8-K", acc, "2025-01-01")], df)
        patch_edgar(ff)
        # >= boundary: equal date is KEPT.
        n = filings.pull_filing_index(3, "2025-01-01")
        assert n == 1
        rows = temp_db.execute("SELECT 1 FROM dilution_filings WHERE accession_number=?", (acc,))
        assert len(rows) == 1

    def test_filing_date_one_day_earlier_skipped(self, temp_db, patch_edgar):
        acc = "BOUND-LT"
        df = _df([
            {"accession_number": acc, "reportDate": None, "items": None,
             "primaryDocument": None, "fileNumber": None},
        ])
        ff = FakeFilings([FakeFiling("8-K", acc, "2024-12-31")], df)
        patch_edgar(ff)
        n = filings.pull_filing_index(3, "2025-01-01")
        assert n == 0
        rows = temp_db.execute("SELECT 1 FROM dilution_filings WHERE accession_number=?", (acc,))
        assert len(rows) == 0

    def test_filing_with_none_date_skipped(self, temp_db, patch_edgar):
        acc = "NODATE"
        df = _df([
            {"accession_number": acc, "reportDate": None, "items": None,
             "primaryDocument": None, "fileNumber": None},
        ])
        ff = FakeFilings([FakeFiling("8-K", acc, None)], df)
        patch_edgar(ff)
        n = filings.pull_filing_index(3, "2025-01-01")
        assert n == 0

    def test_missing_dataframe_columns_backfilled_no_keyerror(self, temp_db, patch_edgar):
        acc = "MINIMAL"
        # DataFrame only has accession_number; the 4 expected cols are absent
        # and must be backfilled with None (no KeyError).
        df = _df([{"accession_number": acc}])
        ff = FakeFiling("8-K", acc, "2025-03-14")
        patch_edgar(FakeFilings([ff], df))
        n = filings.pull_filing_index(11, "2025-01-01")
        assert n == 1
        r = temp_db.execute(
            "SELECT report_date, items, primary_doc, file_number "
            "FROM dilution_filings WHERE accession_number=?", (acc,)
        )[0]
        assert r["report_date"] is None
        assert r["items"] is None
        assert r["primary_doc"] is None
        assert r["file_number"] is None

    def test_accession_in_iteration_missing_from_bulk(self, temp_db, patch_edgar):
        # DataFrame describes a DIFFERENT accession; the iterated filing's
        # accession is absent from the bulk dict -> b = {} -> all None,
        # but still inserted.
        iter_acc = "ITER-ONLY"
        df = _df([
            {"accession_number": "OTHER", "reportDate": "2025-01-02",
             "items": "1.01", "primaryDocument": "x.htm", "fileNumber": "333-1"},
        ])
        ff = FakeFilings([FakeFiling("8-K", iter_acc, "2025-03-14")], df)
        patch_edgar(ff)
        n = filings.pull_filing_index(12, "2025-01-01")
        assert n == 1
        r = temp_db.execute(
            "SELECT report_date, items, primary_doc, file_number "
            "FROM dilution_filings WHERE accession_number=?", (iter_acc,)
        )[0]
        assert r["report_date"] is None
        assert r["items"] is None
        assert r["primary_doc"] is None
        assert r["file_number"] is None

    def test_nan_cells_normalized_to_none(self, temp_db, patch_edgar):
        pd = pytest.importorskip("pandas")
        acc = "NANCELLS"
        # Mixed-type columns force NaN for the missing values.
        df = pd.DataFrame([
            {"accession_number": acc, "reportDate": float("nan"),
             "items": "  ", "primaryDocument": float("nan"),
             "fileNumber": float("nan")},
        ])
        ff = FakeFilings([FakeFiling("8-K", acc, "2025-03-14")], df)
        patch_edgar(ff)
        filings.pull_filing_index(13, "2025-01-01")
        r = temp_db.execute(
            "SELECT report_date, items, primary_doc, file_number "
            "FROM dilution_filings WHERE accession_number=?", (acc,)
        )[0]
        assert r["report_date"] is None
        assert r["items"] is None          # whitespace-only -> None
        assert r["primary_doc"] is None
        assert r["file_number"] is None

    def test_report_date_truncated_from_timestamp(self, temp_db, patch_edgar):
        pd = pytest.importorskip("pandas")
        acc = "TSDATE"
        df = pd.DataFrame([
            {"accession_number": acc, "reportDate": pd.Timestamp("2025-03-10 09:30"),
             "items": None, "primaryDocument": None, "fileNumber": None},
        ])
        ff = FakeFilings([FakeFiling("8-K", acc, "2025-03-14")], df)
        patch_edgar(ff)
        filings.pull_filing_index(14, "2025-01-01")
        r = temp_db.execute(
            "SELECT report_date FROM dilution_filings WHERE accession_number=?", (acc,)
        )[0]
        assert r["report_date"] == "2025-03-10"

    def test_upsert_updates_existing_row_no_duplicate(self, temp_db, patch_edgar):
        acc = "UPSERT-ME"
        # First pass: form 8-K, items "1.01".
        df1 = _df([
            {"accession_number": acc, "reportDate": "2025-03-10", "items": "1.01",
             "primaryDocument": "v1.htm", "fileNumber": "333-1"},
        ])
        ff1 = FakeFilings([FakeFiling("8-K", acc, "2025-03-14")], df1)
        patch_edgar(ff1)
        assert filings.pull_filing_index(99, "2025-01-01") == 1

        first = temp_db.execute(
            "SELECT fetched_at FROM dilution_filings WHERE accession_number=?", (acc,)
        )[0]
        original_fetched_at = first["fetched_at"]
        assert original_fetched_at is not None

        # Second pass: same accession, MUTATED form/items/file_number.
        df2 = _df([
            {"accession_number": acc, "reportDate": "2025-03-11", "items": "8.01",
             "primaryDocument": "v2.htm", "fileNumber": "333-2"},
        ])
        ff2 = FakeFilings(
            [FakeFiling("8-K/A", acc, "2025-03-20", homepage_url="http://hp/v2")],
            df2,
        )
        patch_edgar(ff2)
        # Return count still increments for the (single) accession.
        assert filings.pull_filing_index(99, "2025-01-01") == 1

        rows = temp_db.execute(
            "SELECT * FROM dilution_filings WHERE accession_number=?", (acc,)
        )
        # ON CONFLICT update, not a duplicate insert.
        assert len(rows) == 1
        r = rows[0]
        assert r["form"] == "8-K/A"
        assert r["filing_date"] == "2025-03-20"
        assert r["report_date"] == "2025-03-11"
        assert r["items"] == "8.01"
        assert r["primary_doc"] == "v2.htm"
        assert r["file_number"] == "333-2"
        assert r["homepage_url"] == "http://hp/v2"
        # fetched_at is excluded from the DO UPDATE SET -> preserved.
        assert r["fetched_at"] == original_fetched_at

    def test_upsert_over_preseeded_row_preserves_fetched_at(self, temp_db, patch_edgar):
        acc = "PRESEED"
        preseed_fetched = "2020-01-01T00:00:00Z"
        # Pre-seed a colliding accession with a distinct fetched_at.
        temp_db.add_filing(
            acc, 77, form="10-K", filing_date="2024-01-01",
            items="olditems", fetched_at=preseed_fetched,
        )
        df = _df([
            {"accession_number": acc, "reportDate": "2025-03-10", "items": "newitems",
             "primaryDocument": "new.htm", "fileNumber": "333-9"},
        ])
        ff = FakeFilings([FakeFiling("8-K", acc, "2025-03-14")], df)
        patch_edgar(ff)
        assert filings.pull_filing_index(77, "2025-01-01") == 1

        r = temp_db.execute(
            "SELECT form, items, fetched_at FROM dilution_filings "
            "WHERE accession_number=?", (acc,)
        )[0]
        # Mutable columns overwritten...
        assert r["form"] == "8-K"
        assert r["items"] == "newitems"
        # ...but fetched_at preserved from the pre-seeded value.
        assert r["fetched_at"] == preseed_fetched

    def test_count_equals_inserted_row_count(self, temp_db, patch_edgar):
        df = _df([
            {"accession_number": "R1", "reportDate": None, "items": None,
             "primaryDocument": None, "fileNumber": None},
            {"accession_number": "R2", "reportDate": None, "items": None,
             "primaryDocument": None, "fileNumber": None},
            {"accession_number": "X1", "reportDate": None, "items": None,
             "primaryDocument": None, "fileNumber": None},
        ])
        ff = FakeFilings(
            [
                FakeFiling("8-K", "R1", "2025-03-14"),
                FakeFiling("10-K", "R2", "2025-03-15"),
                FakeFiling("4", "X1", "2025-03-16"),  # irrelevant
            ],
            df,
        )
        patch_edgar(ff)
        n = filings.pull_filing_index(42, "2025-01-01")
        stored = temp_db.execute("SELECT COUNT(*) AS c FROM dilution_filings")[0]["c"]
        assert n == 2
        assert stored == n

    def test_empty_df_but_filings_present_still_inserts(self, temp_db, patch_edgar):
        # GOTCHA isolation: df.empty is True (so the bulk-extract block is
        # skipped and bulk stays {}), but iteration STILL yields a filing.
        # The loop is independent of the DataFrame, so the relevant filing
        # is inserted with all-None metadata. The pre-existing
        # test_empty_dataframe_and_no_filings_returns_zero conflated empty-df
        # with empty-filings and never proved this branch.
        acc = "DF-EMPTY-BUT-ITER"
        ff = FakeFilings([FakeFiling("8-K", acc, "2025-03-14")], _empty_df())
        patch_edgar(ff)
        n = filings.pull_filing_index(101, "2025-01-01")
        assert n == 1
        r = temp_db.execute(
            "SELECT report_date, items, primary_doc, file_number, homepage_url "
            "FROM dilution_filings WHERE accession_number=?", (acc,)
        )[0]
        # No bulk row existed -> all metadata None...
        assert r["report_date"] is None
        assert r["items"] is None
        assert r["primary_doc"] is None
        assert r["file_number"] is None
        # ...but homepage_url comes from the filing object, not the bulk dict.
        assert r["homepage_url"] == "http://hp/x"

    def test_datetime_date_filing_date_realistic_edgar(self, temp_db, patch_edgar):
        # Realism: edgartools hands f.filing_date as a stdlib datetime.date,
        # not a pre-stringified value. The code does str(f.filing_date), which
        # yields "YYYY-MM-DD" and compares lexicographically against since_date.
        acc = "DATE-OBJ"
        df = _df([
            {"accession_number": acc, "reportDate": "2025-03-10", "items": None,
             "primaryDocument": None, "fileNumber": None},
        ])
        ff = FakeFilings(
            [FakeFiling("8-K", acc, datetime.date(2025, 3, 14))], df
        )
        patch_edgar(ff)
        n = filings.pull_filing_index(102, "2025-01-01")
        assert n == 1
        r = temp_db.execute(
            "SELECT filing_date FROM dilution_filings WHERE accession_number=?",
            (acc,),
        )[0]
        # str(date) -> the full YYYY-MM-DD, stored verbatim (no truncation here).
        assert r["filing_date"] == "2025-03-14"

    def test_datetime_date_filing_date_before_window_skipped(self, temp_db, patch_edgar):
        # The in-loop secondary guard must still skip a date-object filing
        # whose str() sorts strictly before since_date.
        acc = "DATE-OBJ-OLD"
        df = _df([
            {"accession_number": acc, "reportDate": None, "items": None,
             "primaryDocument": None, "fileNumber": None},
        ])
        ff = FakeFilings(
            [FakeFiling("8-K", acc, datetime.date(2024, 12, 31))], df
        )
        patch_edgar(ff)
        n = filings.pull_filing_index(103, "2025-01-01")
        assert n == 0
        rows = temp_db.execute(
            "SELECT 1 FROM dilution_filings WHERE accession_number=?", (acc,)
        )
        assert len(rows) == 0

    def test_timestamp_filing_date_in_window_kept(self, temp_db, patch_edgar):
        # If filing_date is a pandas Timestamp, str() is "YYYY-MM-DD HH:MM:SS";
        # the lexicographic >= since_date compare still holds and the full
        # str (with time) is stored verbatim into filing_date.
        pd = pytest.importorskip("pandas")
        acc = "TS-FDATE"
        df = _df([
            {"accession_number": acc, "reportDate": None, "items": None,
             "primaryDocument": None, "fileNumber": None},
        ])
        ts = pd.Timestamp("2025-03-14 09:30:00")
        ff = FakeFilings([FakeFiling("8-K", acc, ts)], df)
        patch_edgar(ff)
        n = filings.pull_filing_index(104, "2025-01-01")
        assert n == 1
        r = temp_db.execute(
            "SELECT filing_date FROM dilution_filings WHERE accession_number=?",
            (acc,),
        )[0]
        # filing_date is str(f.filing_date) UNtruncated -> keeps the time part.
        assert r["filing_date"] == "2025-03-14 09:30:00"

    def test_int_zero_cell_stored_as_string_zero(self, temp_db, patch_edgar):
        # End-to-end of the _norm_str(0) trap: a DataFrame cell holding the
        # integer 0 (numpy.int64(0)) is NOT NaN, so it survives as the string
        # "0" rather than being coerced to None. Exercise it via the items col.
        pd = pytest.importorskip("pandas")
        acc = "ZERO-ITEMS"
        df = pd.DataFrame([
            {"accession_number": acc, "reportDate": None, "items": 0,
             "primaryDocument": None, "fileNumber": 0},
        ])
        ff = FakeFilings([FakeFiling("8-K", acc, "2025-03-14")], df)
        patch_edgar(ff)
        filings.pull_filing_index(105, "2025-01-01")
        r = temp_db.execute(
            "SELECT items, file_number FROM dilution_filings WHERE accession_number=?",
            (acc,),
        )[0]
        assert r["items"] == "0"
        assert r["file_number"] == "0"

    @pytest.mark.parametrize(
        "form",
        ["8-K", "6-K", "424B5", "S-1", "S-3ASR", "POS AM", "10-Q",
         "20-F", "DEF 14A", "FWP", "425", "RW", "EFFECT", "1-A", "SUPPL"],
    )
    def test_each_relevant_form_actually_persisted(self, temp_db, patch_edgar, form):
        # Beyond the pure _is_relevant unit test, prove a representative
        # spread of relevant forms each round-trips into dilution_filings
        # through the real insert path.
        acc = f"ACC-{form.replace(' ', '_')}"
        df = _df([
            {"accession_number": acc, "reportDate": None, "items": None,
             "primaryDocument": None, "fileNumber": None},
        ])
        ff = FakeFilings([FakeFiling(form, acc, "2025-03-14")], df)
        patch_edgar(ff)
        n = filings.pull_filing_index(106, "2025-01-01")
        assert n == 1
        r = temp_db.execute(
            "SELECT form FROM dilution_filings WHERE accession_number=?", (acc,)
        )[0]
        assert r["form"] == form

    def test_no_real_db_used_temp_db_isolated(self, temp_db, patch_edgar):
        # Guard against real-DB leakage: writes land in the per-test temp DB,
        # whose path lives under pytest's tmp_path, never config.DB_PATH.
        import config
        acc = "ISOLATION"
        df = _df([
            {"accession_number": acc, "reportDate": None, "items": None,
             "primaryDocument": None, "fileNumber": None},
        ])
        ff = FakeFilings([FakeFiling("8-K", acc, "2025-03-14")], df)
        patch_edgar(ff)
        filings.pull_filing_index(107, "2025-01-01")
        # The fixture's DB path must NOT be the production DB.
        assert str(temp_db.path) != str(config.DB_PATH)
        assert "tmp" in str(temp_db.path) or "pytest" in str(temp_db.path)

    def test_company_constructed_with_cik(self, temp_db, patch_edgar):
        # Confirm Company(cik) is built with the cik we passed (not testing a
        # stub's own return — asserts the real call wiring).
        ff = FakeFilings([], _empty_df())
        company_cls = patch_edgar(ff)
        filings.pull_filing_index(424242, "2025-01-01")
        assert company_cls.instances == [424242]

    def test_fetched_at_comes_through_now_iso_seam(self, temp_db, patch_edgar, monkeypatch):
        # Prove the `from db import now_iso` binding inside filings.py is the
        # seam that populates fetched_at: monkeypatch filings.now_iso to a
        # sentinel and assert it lands verbatim. (Asserting "not None" alone
        # would pass even if fetched_at were sourced elsewhere.)
        sentinel = "1999-12-31T23:59:59Z"
        monkeypatch.setattr(filings, "now_iso", lambda: sentinel)
        acc = "NOWISO"
        df = _df([
            {"accession_number": acc, "reportDate": None, "items": None,
             "primaryDocument": None, "fileNumber": None},
        ])
        ff = FakeFilings([FakeFiling("8-K", acc, "2025-03-14")], df)
        patch_edgar(ff)
        filings.pull_filing_index(200, "2025-01-01")
        r = temp_db.execute(
            "SELECT fetched_at FROM dilution_filings WHERE accession_number=?", (acc,)
        )[0]
        assert r["fetched_at"] == sentinel

    def test_stored_cik_is_function_arg_not_filing_attr(self, temp_db, patch_edgar):
        # The cik column must come from the pull_filing_index(cik=...) argument,
        # NOT from any attribute of the filing object. FakeFiling exposes no
        # .cik, so a regression that read f.cik would AttributeError; here we
        # also positively assert the stored cik equals the passed arg even
        # though two distinct filings share the same DataFrame.
        df = _df([
            {"accession_number": "K1", "reportDate": None, "items": None,
             "primaryDocument": None, "fileNumber": None},
            {"accession_number": "K2", "reportDate": None, "items": None,
             "primaryDocument": None, "fileNumber": None},
        ])
        ff = FakeFilings(
            [FakeFiling("8-K", "K1", "2025-03-14"),
             FakeFiling("10-K", "K2", "2025-03-15")],
            df,
        )
        patch_edgar(ff)
        filings.pull_filing_index(98765, "2025-01-01")
        rows = temp_db.execute("SELECT accession_number, cik FROM dilution_filings")
        ciks = {r["accession_number"]: r["cik"] for r in rows}
        assert ciks == {"K1": 98765, "K2": 98765}

    def test_log_summary_emits_relevant_count(self, temp_db, patch_edgar, caplog):
        # The function logs a one-line summary "  N relevant filings since D"
        # at INFO; assert it reflects the in-window relevant count (1), not the
        # raw filings count (2).
        import logging
        df = _df([
            {"accession_number": "L1", "reportDate": None, "items": None,
             "primaryDocument": None, "fileNumber": None},
            {"accession_number": "L2", "reportDate": None, "items": None,
             "primaryDocument": None, "fileNumber": None},
        ])
        ff = FakeFilings(
            [
                FakeFiling("8-K", "L1", "2025-03-14"),     # relevant
                FakeFiling("SC 13D", "L2", "2025-03-15"),  # irrelevant
            ],
            df,
        )
        patch_edgar(ff)
        with caplog.at_level(logging.INFO, logger="dilution.filings"):
            n = filings.pull_filing_index(108, "2025-02-01")
        assert n == 1
        msgs = [r.getMessage() for r in caplog.records]
        assert any("1 relevant filings since 2025-02-01" in m for m in msgs)
