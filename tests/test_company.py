"""Unit tests for dilution/company.py.

Covers the three edgar-mockable functions (resolve, _set_identity_once,
ensure_company) and the three DB-backed functions (upsert_company,
get_company_by_ticker, get_unit_context).

Network/edgar is NEVER touched: ``Company`` and ``set_identity`` were imported
by name into the ``dilution.company`` namespace, so we monkeypatch them there.
DB access is rerouted to a fresh per-test SQLite file by the autouse
``temp_db`` fixture in conftest.py.
"""

from __future__ import annotations

import pytest

import config
import dilution.company as company


# ── test doubles ────────────────────────────────────────────────────────

class FakeCompany:
    """Stand-in for edgar.Company.

    Captures the positional arg it was constructed with on the class so a
    test can assert int-vs-upper routing, and returns an object exposing
    .cik / .tickers / .name.
    """

    last_arg = None  # set on the class each time the factory is invoked

    def __init__(self, cik="0001499961", tickers=None, name="Fake Corp",
                 *, drop_tickers=False):
        self._cik = cik
        self._tickers = tickers
        self.name = name
        self._drop_tickers = drop_tickers
        if not drop_tickers:
            self.tickers = tickers

    @property
    def cik(self):
        return self._cik


def make_factory(cik="0001499961", tickers=None, name="Fake Corp",
                 drop_tickers=False):
    """Build a callable usable as the patched ``Company`` symbol.

    Records the construction argument into ``captured['arg']`` so the test
    can assert which branch (int vs str.upper) was taken.
    """
    captured = {"arg": None, "calls": 0}

    def factory(arg):
        captured["arg"] = arg
        captured["calls"] += 1
        return FakeCompany(cik=cik, tickers=tickers, name=name,
                           drop_tickers=drop_tickers)

    factory.captured = captured
    return factory


@pytest.fixture
def identity_recorder(monkeypatch):
    """Patch set_identity with a recorder and return the call-arg list."""
    calls = []
    monkeypatch.setattr(company, "set_identity",
                        lambda ident: calls.append(ident))
    return calls


# ── resolve ─────────────────────────────────────────────────────────────

class TestResolve:
    def test_numeric_string_takes_isdigit_branch(self, monkeypatch,
                                                  identity_recorder):
        # '1499961'.isdigit() -> Company(int(...)) not Company(str)
        factory = make_factory(cik="1499961", tickers=["BINI"])
        monkeypatch.setattr(company, "Company", factory)
        out = company.resolve("1499961")
        assert factory.captured["arg"] == 1499961
        assert isinstance(factory.captured["arg"], int)
        assert out == {"cik": 1499961, "ticker": "BINI", "name": "Fake Corp"}

    def test_lowercase_ticker_uppercased_before_company(self, monkeypatch,
                                                        identity_recorder):
        factory = make_factory(tickers=["BINI"])
        monkeypatch.setattr(company, "Company", factory)
        company.resolve("bini")
        assert factory.captured["arg"] == "BINI"

    def test_surrounding_whitespace_stripped(self, monkeypatch,
                                             identity_recorder):
        factory = make_factory(tickers=["BINI"])
        monkeypatch.setattr(company, "Company", factory)
        company.resolve("  bini  ")
        assert factory.captured["arg"] == "BINI"

    def test_int_identifier_accepted_and_routed_numeric(self, monkeypatch,
                                                        identity_recorder):
        # int identifier coerced via str(identifier) then isdigit()
        factory = make_factory(cik="1499961", tickers=["BINI"])
        monkeypatch.setattr(company, "Company", factory)
        out = company.resolve(1499961)
        assert factory.captured["arg"] == 1499961
        assert isinstance(factory.captured["arg"], int)
        assert out["cik"] == 1499961

    def test_tickers_empty_list_falls_back_to_upper_ident(self, monkeypatch,
                                                          identity_recorder):
        factory = make_factory(cik="42", tickers=[])
        monkeypatch.setattr(company, "Company", factory)
        out = company.resolve("abc")
        assert out["ticker"] == "ABC"

    def test_tickers_none_falls_back_to_upper_ident(self, monkeypatch,
                                                    identity_recorder):
        factory = make_factory(cik="42", tickers=None)
        monkeypatch.setattr(company, "Company", factory)
        out = company.resolve("abc")
        assert out["ticker"] == "ABC"

    def test_numeric_branch_fallback_uses_digit_string(self, monkeypatch,
                                                       identity_recorder):
        # GOTCHA(2): on the numeric branch with no tickers, the fallback
        # is ident.upper() which is just the digit string itself.
        factory = make_factory(cik="1499961", tickers=None)
        monkeypatch.setattr(company, "Company", factory)
        out = company.resolve("1499961")
        assert out["ticker"] == "1499961"

    def test_multiple_tickers_first_chosen(self, monkeypatch,
                                           identity_recorder):
        factory = make_factory(tickers=["ABC", "DEF"])
        monkeypatch.setattr(company, "Company", factory)
        out = company.resolve("abc")
        assert out["ticker"] == "ABC"

    def test_cik_string_coerced_to_int(self, monkeypatch, identity_recorder):
        # zero-padded CIK string -> int 1499961
        factory = make_factory(cik="0001499961", tickers=["BINI"])
        monkeypatch.setattr(company, "Company", factory)
        out = company.resolve("bini")
        assert out["cik"] == 1499961
        assert isinstance(out["cik"], int)

    def test_missing_tickers_attribute_uses_getattr_default(self, monkeypatch,
                                                            identity_recorder):
        # Company object missing the 'tickers' attribute entirely.
        factory = make_factory(cik="42", drop_tickers=True)
        monkeypatch.setattr(company, "Company", factory)
        out = company.resolve("abc")
        assert out["ticker"] == "ABC"

    def test_set_identity_invoked_once_with_config_identity(self, monkeypatch,
                                                            identity_recorder):
        factory = make_factory(tickers=["BINI"])
        monkeypatch.setattr(company, "Company", factory)
        company.resolve("bini")
        assert identity_recorder == [config.EDGAR_IDENTITY]

    def test_name_passed_through(self, monkeypatch, identity_recorder):
        factory = make_factory(tickers=["BINI"], name="Binah Labs Inc")
        monkeypatch.setattr(company, "Company", factory)
        out = company.resolve("bini")
        assert out["name"] == "Binah Labs Inc"

    def test_set_identity_called_before_company_construction(self, monkeypatch):
        # Documented behavior: resolve() sets the EDGAR identity *before* it
        # constructs Company(...) (which would otherwise hit the network with
        # no UA). Pin the ordering, not just the call count.
        order = []
        monkeypatch.setattr(company, "set_identity",
                            lambda ident: order.append(("identity", ident)))

        def factory(arg):
            order.append(("company", arg))
            return FakeCompany(cik="42", tickers=["BINI"])

        monkeypatch.setattr(company, "Company", factory)
        company.resolve("bini")
        assert order == [("identity", config.EDGAR_IDENTITY),
                         ("company", "BINI")]

    def test_empty_identifier_routes_to_ticker_branch(self, monkeypatch,
                                                      identity_recorder):
        # Boundary: a whitespace-only identifier strips to "" which is NOT
        # .isdigit(), so it takes the str branch -> Company("") and the
        # ticker fallback is "".upper() == "". (No raise; pins the actual
        # current behavior at the empty boundary.)
        factory = make_factory(cik="42", tickers=[])
        monkeypatch.setattr(company, "Company", factory)
        out = company.resolve("   ")
        assert factory.captured["arg"] == ""
        assert out == {"cik": 42, "ticker": "", "name": "Fake Corp"}

    def test_cik_returned_as_int_already_passes_through_int(self, monkeypatch,
                                                            identity_recorder):
        # int(c.cik) is idempotent when edgar hands back an int cik.
        factory = make_factory(cik=1499961, tickers=["BINI"])
        monkeypatch.setattr(company, "Company", factory)
        out = company.resolve("bini")
        assert out["cik"] == 1499961
        assert isinstance(out["cik"], int)

    def test_non_numeric_cik_raises_valueerror(self, monkeypatch,
                                               identity_recorder):
        # Failure path: resolve hard-coerces cik via int(c.cik). If edgar
        # hands back a non-numeric cik the call propagates ValueError rather
        # than swallowing it — pins the strict coercion contract.
        factory = make_factory(cik="NOTANUMBER", tickers=["BINI"])
        monkeypatch.setattr(company, "Company", factory)
        with pytest.raises(ValueError):
            company.resolve("bini")

    def test_set_identity_fires_exactly_once_per_call(self, monkeypatch):
        # Complements the arg-content check: pin the call *count* at exactly
        # one so a refactor adding a second identity call is caught.
        calls = []
        monkeypatch.setattr(company, "set_identity",
                            lambda ident: calls.append(ident))
        factory = make_factory(tickers=["BINI"])
        monkeypatch.setattr(company, "Company", factory)
        company.resolve("bini")
        assert len(calls) == 1


# ── _set_identity_once ──────────────────────────────────────────────────

class TestSetIdentityOnce:
    def test_passes_config_identity_verbatim(self, identity_recorder):
        company._set_identity_once()
        assert identity_recorder == [config.EDGAR_IDENTITY]

    def test_no_memoization_fires_every_call(self, identity_recorder):
        # GOTCHA(1): 'once' is a misnomer; there is no guard/caching.
        company._set_identity_once()
        company._set_identity_once()
        company._set_identity_once()
        assert identity_recorder == [config.EDGAR_IDENTITY] * 3


# ── ensure_company ──────────────────────────────────────────────────────

class TestEnsureCompany:
    def test_returns_resolve_dict_unchanged(self, monkeypatch, temp_db,
                                            identity_recorder):
        factory = make_factory(cik="1499961", tickers=["BINI"],
                               name="Binah Labs Inc")
        monkeypatch.setattr(company, "Company", factory)
        out = company.ensure_company("bini")
        assert out == {"cik": 1499961, "ticker": "BINI",
                       "name": "Binah Labs Inc"}

    def test_upsert_called_with_cik_ticker_name_in_order(self, monkeypatch,
                                                         temp_db,
                                                         identity_recorder):
        factory = make_factory(cik="1499961", tickers=["BINI"],
                               name="Binah Labs Inc")
        monkeypatch.setattr(company, "Company", factory)
        # Capture the raw positional args (and any kwargs) so the test proves
        # *positional ordering* (cik, ticker, name), not merely that three
        # named params each received some value. ensure_company calls
        # upsert_company(info["cik"], info["ticker"], info["name"]) positionally.
        seen = {}
        monkeypatch.setattr(
            company, "upsert_company",
            lambda *args, **kwargs: seen.update(args=args, kwargs=kwargs))
        company.ensure_company("bini")
        assert seen["kwargs"] == {}
        assert seen["args"] == (1499961, "BINI", "Binah Labs Inc")

    def test_roundtrip_persists_row(self, monkeypatch, temp_db,
                                    identity_recorder):
        factory = make_factory(cik="1499961", tickers=["BINI"],
                               name="Binah Labs Inc")
        monkeypatch.setattr(company, "Company", factory)
        company.ensure_company("bini")
        back = company.get_company_by_ticker("BINI")
        assert back == {"cik": 1499961, "ticker": "BINI",
                        "name": "Binah Labs Inc"}

    def test_upsert_not_called_if_resolve_raises(self, monkeypatch, temp_db,
                                                 identity_recorder):
        def boom(arg):
            raise RuntimeError("edgar down")
        monkeypatch.setattr(company, "Company", boom)
        calls = []
        monkeypatch.setattr(company, "upsert_company",
                            lambda *a, **k: calls.append(a))
        with pytest.raises(RuntimeError):
            company.ensure_company("bini")
        assert calls == []

    def test_logs_resolution_at_info(self, monkeypatch, temp_db,
                                     identity_recorder, caplog):
        # ensure_company emits an INFO log with the resolved CIK/ticker/name.
        factory = make_factory(cik="1499961", tickers=["BINI"],
                               name="Binah Labs Inc")
        monkeypatch.setattr(company, "Company", factory)
        with caplog.at_level("INFO", logger="dilution.company"):
            company.ensure_company("bini")
        msgs = [r.getMessage() for r in caplog.records
                if r.name == "dilution.company"]
        assert any("1499961" in m and "BINI" in m
                   and "Binah Labs Inc" in m for m in msgs)


# ── upsert_company ──────────────────────────────────────────────────────

class TestUpsertCompany:
    def test_fresh_cik_inserts_row_with_added_at(self, temp_db):
        company.upsert_company(1499961, "BINI", "Binah Labs Inc")
        rows = temp_db.execute(
            "SELECT cik, ticker, name, added_at FROM dilution_company "
            "WHERE cik = ?", (1499961,))
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["cik"] == 1499961
        assert row["ticker"] == "BINI"
        assert row["name"] == "Binah Labs Inc"
        assert row["added_at"]  # populated (now_iso())

    def test_conflict_preserves_added_at_updates_ticker_name(self, temp_db):
        temp_db.add_company(1499961, ticker="OLD", name="Old Name",
                            added_at="2020-05-05T00:00:00Z")
        company.upsert_company(1499961, "NEW", "New Name")
        row = dict(temp_db.execute(
            "SELECT * FROM dilution_company WHERE cik = ?", (1499961,))[0])
        assert row["ticker"] == "NEW"
        assert row["name"] == "New Name"
        # added_at is NOT in the UPDATE SET -> original preserved.
        assert row["added_at"] == "2020-05-05T00:00:00Z"

    def test_conflict_does_not_touch_is_fpi_or_ads_ratio(self, temp_db):
        temp_db.add_company(1499961, ticker="OLD", name="Old Name",
                            is_fpi=1, ads_ratio=100.0)
        company.upsert_company(1499961, "NEW", "New Name")
        row = dict(temp_db.execute(
            "SELECT * FROM dilution_company WHERE cik = ?", (1499961,))[0])
        assert row["is_fpi"] == 1
        assert row["ads_ratio"] == 100.0
        assert row["ticker"] == "NEW"

    def test_rename_overwrites_old_ticker(self, temp_db):
        company.upsert_company(7, "AAA", "Co")
        company.upsert_company(7, "BBB", "Co")
        rows = temp_db.execute(
            "SELECT ticker FROM dilution_company WHERE cik = ?", (7,))
        assert len(rows) == 1
        assert rows[0]["ticker"] == "BBB"

    def test_two_distinct_ciks_two_rows(self, temp_db):
        company.upsert_company(1, "AAA", "First")
        company.upsert_company(2, "BBB", "Second")
        rows = temp_db.execute(
            "SELECT cik, ticker FROM dilution_company ORDER BY cik")
        assert [(r["cik"], r["ticker"]) for r in rows] == [
            (1, "AAA"), (2, "BBB")]

    def test_fresh_insert_leaves_unit_columns_at_schema_defaults(self, temp_db):
        # The INSERT lists only (cik, ticker, name, added_at); is_fpi/ads_ratio
        # are never written, so a fresh company falls back to the schema
        # defaults (is_fpi=0 NOT NULL DEFAULT 0, ads_ratio NULL). Proven via
        # the get_unit_context round-trip so the cross-function contract holds:
        # an upserted-but-not-yet-unit-staged company reads as US/common.
        company.upsert_company(7, "AAA", "Co")
        row = dict(temp_db.execute(
            "SELECT is_fpi, ads_ratio FROM dilution_company WHERE cik = ?",
            (7,))[0])
        assert row["is_fpi"] == 0
        assert row["ads_ratio"] is None
        assert company.get_unit_context(7) == {
            "is_fpi": 0, "ads_ratio": None, "reporting_unit": "common"}


# ── get_company_by_ticker ───────────────────────────────────────────────

class TestGetCompanyByTicker:
    def test_numeric_string_looks_up_by_cik(self, temp_db):
        temp_db.add_company(1499961, ticker="BINI", name="Binah Labs Inc")
        out = company.get_company_by_ticker("1499961")
        assert out == {"cik": 1499961, "ticker": "BINI",
                       "name": "Binah Labs Inc"}

    def test_lowercase_ticker_matches_stored_uppercase(self, temp_db):
        temp_db.add_company(1, ticker="TEST", name="Test Co")
        out = company.get_company_by_ticker("test")
        assert out is not None
        assert out["ticker"] == "TEST"

    def test_whitespace_stripped_before_match(self, temp_db):
        temp_db.add_company(1, ticker="TEST", name="Test Co")
        out = company.get_company_by_ticker("  test  ")
        assert out is not None
        assert out["cik"] == 1

    def test_no_match_returns_none(self, temp_db):
        out = company.get_company_by_ticker("NOPE")
        assert out is None

    def test_numeric_no_match_returns_none(self, temp_db):
        out = company.get_company_by_ticker("999999")
        assert out is None

    def test_empty_string_routes_to_ticker_branch_returns_none(self, temp_db):
        # Boundary: "".strip().isdigit() is False -> ticker branch, querying
        # WHERE ticker = '' against an empty/unmatched table -> None, no raise.
        temp_db.add_company(1, ticker="TEST", name="Test Co")
        out = company.get_company_by_ticker("")
        assert out is None

    def test_stored_lowercase_ticker_does_not_match_upper_query(self, temp_db):
        # GOTCHA(3): only the query is upper()'d, not stored data. A
        # lowercase-stored ticker is a non-case because the convention is
        # uppercase storage -> upper query 'TEST' won't find stored 'test'.
        temp_db.add_company(1, ticker="test", name="Test Co")
        out = company.get_company_by_ticker("test")  # -> query 'TEST'
        assert out is None

    def test_selects_only_cik_ticker_name(self, temp_db):
        temp_db.add_company(1, ticker="TEST", name="Test Co", is_fpi=1,
                            ads_ratio=100.0)
        out = company.get_company_by_ticker("TEST")
        assert set(out.keys()) == {"cik", "ticker", "name"}

    def test_int_arg_coerced_and_routed_through_isdigit(self, temp_db):
        temp_db.add_company(1499961, ticker="BINI", name="Binah Labs Inc")
        out = company.get_company_by_ticker(1499961)
        assert out is not None
        assert out["cik"] == 1499961

    def test_alphanumeric_ticker_takes_ticker_branch(self, temp_db):
        # 'AB12'.isdigit() is False -> ticker branch, uppercased lookup.
        # A ticker that merely *contains* digits must NOT be treated as a CIK.
        temp_db.add_company(5, ticker="AB12", name="Mixed Co")
        out = company.get_company_by_ticker("ab12")
        assert out == {"cik": 5, "ticker": "AB12", "name": "Mixed Co"}

    def test_numeric_string_with_whitespace_routes_to_cik(self, temp_db):
        # '  42  '.strip().isdigit() is True -> cik branch with int(42).
        temp_db.add_company(42, ticker="ANS", name="Answer Co")
        out = company.get_company_by_ticker("  42  ")
        assert out is not None
        assert out["cik"] == 42

    def test_ticker_match_returns_dict_not_row_object(self, temp_db):
        # The function wraps the sqlite3.Row in dict(); a caller doing
        # `out.get(...)` or `==` against a plain dict must work.
        temp_db.add_company(1, ticker="TEST", name="Test Co")
        out = company.get_company_by_ticker("TEST")
        assert type(out) is dict


# ── get_unit_context ────────────────────────────────────────────────────

class TestGetUnitContext:
    def test_unknown_cik_safe_default(self, temp_db):
        out = company.get_unit_context(424242)
        assert out == {"is_fpi": 0, "ads_ratio": None,
                       "reporting_unit": "common"}

    def test_fpi_with_ads_ratio(self, temp_db):
        temp_db.add_company(1, ticker="QTEX", name="Q", is_fpi=1,
                            ads_ratio=100.0)
        out = company.get_unit_context(1)
        assert out["is_fpi"] == 1
        assert out["reporting_unit"] == "ads"
        assert out["ads_ratio"] == pytest.approx(100.0)

    def test_us_issuer_is_common(self, temp_db):
        temp_db.add_company(1, ticker="US", name="U", is_fpi=0)
        out = company.get_unit_context(1)
        assert out["is_fpi"] == 0
        assert out["reporting_unit"] == "common"
        assert out["ads_ratio"] is None

    def test_is_fpi_zero_collapses_via_or(self, temp_db):
        # GOTCHA(4): the `or 0` collapses a falsy stored is_fpi. The column
        # is NOT NULL DEFAULT 0 in the schema, so a stored 0 (not a true
        # NULL) is the realistic falsy case; the no-row NULL path is covered
        # by test_unknown_cik_safe_default.
        temp_db.add_company(1, ticker="X", name="X", is_fpi=0)
        out = company.get_unit_context(1)
        assert out["is_fpi"] == 0
        assert isinstance(out["is_fpi"], int)
        assert out["reporting_unit"] == "common"

    def test_is_fpi_stored_as_real_yields_int(self, temp_db):
        # A REAL/float-ish stored is_fpi still yields an int via int(...).
        temp_db.add_company(1, ticker="X", name="X")
        with temp_db.conn() as c:
            c.execute("UPDATE dilution_company SET is_fpi = 1.0 WHERE cik = 1")
        out = company.get_unit_context(1)
        assert out["is_fpi"] == 1
        assert isinstance(out["is_fpi"], int)
        assert out["reporting_unit"] == "ads"

    def test_fpi_with_null_ads_ratio(self, temp_db):
        # FPI flagged before unit stage filled the ratio.
        temp_db.add_company(1, ticker="X", name="X", is_fpi=1, ads_ratio=None)
        out = company.get_unit_context(1)
        assert out["is_fpi"] == 1
        assert out["reporting_unit"] == "ads"
        assert out["ads_ratio"] is None

    def test_us_issuer_ads_ratio_passed_through_verbatim(self, temp_db):
        # GOTCHA: the `if row else None` only guards the no-row case. When a
        # row exists, ads_ratio is returned verbatim even for a US issuer.
        temp_db.add_company(1, ticker="X", name="X", is_fpi=0, ads_ratio=5.0)
        out = company.get_unit_context(1)
        assert out["is_fpi"] == 0
        assert out["reporting_unit"] == "common"
        assert out["ads_ratio"] == pytest.approx(5.0)

    @pytest.mark.parametrize("is_fpi,expected_unit", [
        (1, "ads"),
        (0, "common"),
    ])
    def test_reporting_unit_follows_is_fpi(self, temp_db, is_fpi,
                                           expected_unit):
        temp_db.add_company(1, ticker="X", name="X", is_fpi=is_fpi)
        out = company.get_unit_context(1)
        assert out["reporting_unit"] == expected_unit

    @pytest.mark.parametrize("stored_ratio", [1.0, 4.0, 100.0, 400.0])
    def test_ads_ratio_passthrough_various_fpi_ratios(self, temp_db,
                                                      stored_ratio):
        # ads_ratio is returned verbatim (no rounding/normalization). Sweep
        # realistic ordinary-per-ADS ratios from the schema comment (e.g. 100
        # / 400 for XTLB) plus boundary 1.0.
        temp_db.add_company(1, ticker="X", name="X", is_fpi=1,
                            ads_ratio=stored_ratio)
        out = company.get_unit_context(1)
        assert out["ads_ratio"] == pytest.approx(stored_ratio)
        assert out["reporting_unit"] == "ads"

    def test_truthy_real_is_fpi_floors_to_int_one(self, temp_db):
        # A stored REAL is_fpi of 2.0 is truthy -> int(2.0) == 2 (NOT clamped
        # to 1); reporting_unit is 'ads' because the value is truthy. Pins the
        # actual int() coercion semantics, not a 0/1-only assumption.
        temp_db.add_company(1, ticker="X", name="X")
        with temp_db.conn() as c:
            c.execute("UPDATE dilution_company SET is_fpi = 2.0 WHERE cik = 1")
        out = company.get_unit_context(1)
        assert out["is_fpi"] == 2
        assert isinstance(out["is_fpi"], int)
        assert out["reporting_unit"] == "ads"
