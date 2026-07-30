"""Unit tests for dilution/finviz_client.py.

The Finviz client has no DB and no pydantic surface, so the autouse
``temp_db`` fixture is irrelevant here. The only real I/O seam is
``requests.Session.get`` inside ``FinvizClient._get_csv``; every other
unit is tested by patching a higher seam (``_get_csv``,
``get_daily_closes``, ``highest_close``, ``_client``) or a controllable
clock (``_market_now`` / ``time.monotonic``).

Module-level globals ``_fund_cache`` and ``_default_client`` persist
across tests with NO autouse reset, so ``reset_module_globals`` clears
them before each test.
"""

from __future__ import annotations

from datetime import date, datetime, timezone, timedelta

import pytest
import requests

from dilution import finviz_client as fc


# ── shared fixtures ───────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_module_globals():
    """Clear the module-level fundamentals cache and singleton client.

    These persist across tests; without this any test touching
    ``fundamentals()`` / ``_client()`` would see stale state.
    """
    fc._fund_cache.clear()
    fc._default_client = None
    yield
    fc._fund_cache.clear()
    fc._default_client = None


class _FakeResponse:
    """Minimal stand-in for requests.Response used by _get_csv tests."""

    def __init__(self, *, text="", content=None, raise_exc=None):
        self.text = text
        # content defaults to text-bytes so "truthy content" mirrors text
        self.content = content if content is not None else text.encode()
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc is not None:
            raise self._raise_exc


def _make_client(api_key="key", base_url="https://finviz.test"):
    """Construct a client with explicit auth so config is never read."""
    return fc.FinvizClient(api_key=api_key, base_url=base_url)


# ── _parse_num ────────────────────────────────────────────────────────


class TestParseNum:
    @pytest.mark.parametrize("value", [None, "", "   ", "-", "—", "N/A"])
    def test_blank_and_sentinels_to_none(self, value):
        assert fc._parse_num(value) is None

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("12.34M", 12_340_000.0),
            ("1.23B", 1_230_000_000.0),
            ("5K", 5_000.0),
            ("2T", 2e12),
            ("12.3m", 12_300_000.0),  # lowercase suffix upper()'d
        ],
    )
    def test_scale_suffixes(self, value, expected):
        assert fc._parse_num(value) == pytest.approx(expected)

    def test_commas_removed(self):
        assert fc._parse_num("1,234,567") == pytest.approx(1_234_567.0)

    def test_percent_stripped_not_divided(self):
        # NOTE: percent is stripped but NOT divided by 100 -> '12.3%' == 12.3
        assert fc._parse_num("12.3%") == pytest.approx(12.3)

    def test_dollar_stripped(self):
        assert fc._parse_num("$45.20") == pytest.approx(45.2)

    def test_parenthesized_negative(self):
        assert fc._parse_num("(123)") == pytest.approx(-123.0)

    def test_negative_with_dollar_and_suffix(self):
        assert fc._parse_num("($1.5M)") == pytest.approx(-1_500_000.0)

    def test_zero_is_valid(self):
        result = fc._parse_num("0")
        assert result == 0.0
        assert result is not None

    def test_negative_literal(self):
        assert fc._parse_num("-5") == pytest.approx(-5.0)

    @pytest.mark.parametrize("value", ["abc", "M", "1.2.3"])
    def test_unparseable_to_none(self, value):
        # 'M' -> s[:-1] == '' -> float('') raises ValueError -> None
        assert fc._parse_num(value) is None

    def test_numeric_input_passthrough(self):
        # str(value) is applied; an int/float still parses
        assert fc._parse_num(42) == pytest.approx(42.0)
        assert fc._parse_num(3.5) == pytest.approx(3.5)

    def test_parenthesized_double_negative_flips_to_positive(self):
        # neg flag set by the parens; stripping them leaves '-5'; float('-5')
        # == -5.0; the `-v if neg` then negates AGAIN -> +5.0. Verified
        # against the source: parens + inner minus cancel.
        assert fc._parse_num("(-5)") == pytest.approx(5.0)

    def test_empty_parens_to_none(self):
        # '()' -> neg=True, strip('()') -> '' -> float('') ValueError -> None
        assert fc._parse_num("()") is None

    def test_lowercase_b_suffix(self):
        assert fc._parse_num("1.5b") == pytest.approx(1_500_000_000.0)

    def test_lone_open_paren_to_none(self):
        # '(' is not endswith ')' so neg=False; strip('()') -> '' -> None
        assert fc._parse_num("(") is None


# ── _market_now ───────────────────────────────────────────────────────


class TestMarketNow:
    def test_happy_path_is_tz_aware_new_york(self):
        now = fc._market_now()
        assert isinstance(now, datetime)
        # zoneinfo available in this env -> tz-aware America/New_York
        assert now.tzinfo is not None
        assert "New_York" in str(now.tzinfo)

    def test_fallback_to_naive_when_zoneinfo_raises(self, monkeypatch):
        # Force the ZoneInfo lookup inside _market_now to blow up by
        # poisoning the zoneinfo module's ZoneInfo. _market_now imports it
        # lazily, so patching the real module is what it resolves.
        import zoneinfo

        def _boom(*a, **k):
            raise RuntimeError("no tzdata")

        monkeypatch.setattr(zoneinfo, "ZoneInfo", _boom)
        now = fc._market_now()
        assert isinstance(now, datetime)
        # fallback path: naive local datetime, no exception escapes
        assert now.tzinfo is None


# ── _current_session_date ─────────────────────────────────────────────


class TestCurrentSessionDate:
    def test_zero_padded_mm_dd_yyyy(self, monkeypatch):
        fixed = datetime(2026, 6, 5, 14, 30, tzinfo=timezone.utc)
        monkeypatch.setattr(fc, "_market_now", lambda: fixed)
        assert fc._current_session_date() == "06/05/2026"

    def test_reflects_market_now_value(self, monkeypatch):
        fixed = datetime(2025, 12, 31, 9, 0)
        monkeypatch.setattr(fc, "_market_now", lambda: fixed)
        assert fc._current_session_date() == "12/31/2025"

    def test_uses_new_york_date_for_tz_aware(self, monkeypatch):
        # A tz-aware dt in NY is formatted with its own date, not UTC.
        try:
            from zoneinfo import ZoneInfo
            ny = ZoneInfo("America/New_York")
        except Exception:
            pytest.skip("zoneinfo unavailable")
        # 2026-06-05 00:30 NY (which is 04:30 UTC) -> NY date is 06/05
        fixed = datetime(2026, 6, 5, 0, 30, tzinfo=ny)
        monkeypatch.setattr(fc, "_market_now", lambda: fixed)
        assert fc._current_session_date() == "06/05/2026"


# ── FinvizClient.__init__ ─────────────────────────────────────────────


class TestInit:
    def test_explicit_args_override_config(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "FINVIZ_API_KEY", "config-key",
                            raising=False)
        monkeypatch.setattr(config, "FINVIZ_BASE_URL", "https://config",
                            raising=False)
        c = fc.FinvizClient(api_key="arg-key", base_url="https://arg")
        assert c.api_key == "arg-key"
        assert c.base_url == "https://arg"

    def test_falls_back_to_config(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "FINVIZ_API_KEY", "config-key",
                            raising=False)
        monkeypatch.setattr(config, "FINVIZ_BASE_URL", "https://config",
                            raising=False)
        c = fc.FinvizClient()
        assert c.api_key == "config-key"
        assert c.base_url == "https://config"

    def test_base_url_trailing_slash_stripped(self):
        c = fc.FinvizClient(api_key="k", base_url="https://x.com/")
        assert c.base_url == "https://x.com"

    def test_user_agent_header(self):
        c = _make_client()
        assert c.session.headers["User-Agent"] == "FinvizApps/Dilution"

    def test_no_api_key_logs_warning(self, monkeypatch, caplog):
        import config
        monkeypatch.setattr(config, "FINVIZ_API_KEY", "", raising=False)
        monkeypatch.setattr(config, "FINVIZ_BASE_URL", "https://config",
                            raising=False)
        with caplog.at_level("WARNING", logger="dilution.finviz_client"):
            c = fc.FinvizClient()
        assert not c.api_key
        assert any("without API key" in r.message for r in caplog.records)

    def test_default_timeout(self):
        c = _make_client()
        assert c.timeout == fc.DEFAULT_TIMEOUT


# ── FinvizClient._get_csv ─────────────────────────────────────────────


class TestGetCsv:
    def test_no_api_key_returns_none_without_http(self, monkeypatch):
        # __init__ does `api_key or config.FINVIZ_API_KEY`, so to get a
        # truly keyless client we must also blank the config fallback.
        import config
        monkeypatch.setattr(config, "FINVIZ_API_KEY", "", raising=False)
        c = fc.FinvizClient(api_key="", base_url="https://x")
        assert not c.api_key  # confirm the fallback didn't repopulate it
        called = {"n": 0}

        def _spy(*a, **k):
            called["n"] += 1
            raise AssertionError("should not be called")

        monkeypatch.setattr(c.session, "get", _spy)
        assert c._get_csv("export", {"t": "AAPL"}) is None
        assert called["n"] == 0

    def test_url_join_lstrips_leading_slash(self, monkeypatch):
        c = _make_client(base_url="https://finviz.test")
        captured = {}

        def _get(url, params=None, timeout=None):
            captured["url"] = url
            captured["params"] = params
            return _FakeResponse(text="A\n1\n")

        monkeypatch.setattr(c.session, "get", _get)
        c._get_csv("/export", {"t": "AAPL"})
        # base/export, never base//export
        assert captured["url"] == "https://finviz.test/export"

    def test_auth_param_injected(self, monkeypatch):
        c = _make_client(api_key="secret")
        captured = {}

        def _get(url, params=None, timeout=None):
            captured["params"] = params
            captured["timeout"] = timeout
            return _FakeResponse(text="A\n1\n")

        monkeypatch.setattr(c.session, "get", _get)
        c._get_csv("export", {"t": "AAPL", "v": "152"})
        assert captured["params"]["auth"] == "secret"
        assert captured["params"]["t"] == "AAPL"
        assert captured["params"]["v"] == "152"
        assert captured["timeout"] == fc.DEFAULT_TIMEOUT

    def test_request_exception_returns_none_and_warns(self, monkeypatch,
                                                      caplog):
        c = _make_client()

        def _get(*a, **k):
            raise requests.RequestException("boom")

        monkeypatch.setattr(c.session, "get", _get)
        with caplog.at_level("WARNING", logger="dilution.finviz_client"):
            assert c._get_csv("export", {}) is None
        assert any("failed" in r.message for r in caplog.records)

    def test_http_error_in_raise_for_status_returns_none(self, monkeypatch):
        c = _make_client()
        resp = _FakeResponse(text="ignored",
                             raise_exc=requests.HTTPError("404"))
        monkeypatch.setattr(c.session, "get", lambda *a, **k: resp)
        assert c._get_csv("export", {}) is None

    def test_empty_content_returns_none(self, monkeypatch):
        c = _make_client()
        resp = _FakeResponse(text="", content=b"")
        monkeypatch.setattr(c.session, "get", lambda *a, **k: resp)
        assert c._get_csv("export", {}) is None

    def test_csv_parse_error_returns_none_and_warns(self, monkeypatch,
                                                    caplog):
        # Survey-listed failure path: a malformed body that makes
        # csv.DictReader raise csv.Error must fail soft to None + warn.
        # A single field larger than csv.field_size_limit() is the
        # reliable way to provoke csv.Error during the list() materialize.
        import csv as _csv
        oversized = "x" * (_csv.field_size_limit() + 50)
        c = _make_client()
        resp = _FakeResponse(text="Ticker\n" + oversized + "\n")
        monkeypatch.setattr(c.session, "get", lambda *a, **k: resp)
        with caplog.at_level("WARNING", logger="dilution.finviz_client"):
            assert c._get_csv("export", {}) is None
        assert any("CSV parse failed" in r.message for r in caplog.records)

    def test_valid_csv_single_row(self, monkeypatch):
        c = _make_client()
        resp = _FakeResponse(text="Ticker,Price\nAAPL,100.5\n")
        monkeypatch.setattr(c.session, "get", lambda *a, **k: resp)
        rows = c._get_csv("export", {})
        assert rows == [{"Ticker": "AAPL", "Price": "100.5"}]

    def test_multi_row_preserves_order(self, monkeypatch):
        c = _make_client()
        resp = _FakeResponse(
            text="Ticker,Price\nAAA,1\nBBB,2\nCCC,3\n")
        monkeypatch.setattr(c.session, "get", lambda *a, **k: resp)
        rows = c._get_csv("export", {})
        assert [r["Ticker"] for r in rows] == ["AAA", "BBB", "CCC"]


# ── FinvizClient.get_fundamentals ─────────────────────────────────────


class TestGetFundamentals:
    @pytest.mark.parametrize("ticker", ["", None])
    def test_falsy_ticker_returns_none_no_fetch(self, ticker, monkeypatch):
        c = _make_client()
        called = {"n": 0}

        def _spy(*a, **k):
            called["n"] += 1
            return None

        monkeypatch.setattr(c, "_get_csv", _spy)
        assert c.get_fundamentals(ticker) is None
        assert called["n"] == 0

    def test_ticker_uppercased_in_params(self, monkeypatch):
        c = _make_client()
        captured = {}

        def _get_csv(endpoint, params):
            captured["endpoint"] = endpoint
            captured["params"] = params
            return [{"Ticker": "AAPL", "Price": "100"}]

        monkeypatch.setattr(c, "_get_csv", _get_csv)
        c.get_fundamentals("aapl")
        assert captured["endpoint"] == "export"
        assert captured["params"]["t"] == "AAPL"

    def test_full_export_params_built(self, monkeypatch):
        # The view/filter/columns params are load-bearing for the endpoint;
        # assert the complete request shape, not just the ticker.
        c = _make_client()
        captured = {}

        def _get_csv(endpoint, params):
            captured["params"] = params
            return [{"Ticker": "X"}]

        monkeypatch.setattr(c, "_get_csv", _get_csv)
        c.get_fundamentals("x")
        assert captured["params"]["v"] == "152"
        assert captured["params"]["ft"] == "4"
        # c= is the comma-joined FUNDAMENTALS_COLS tuple, in order
        assert captured["params"]["c"] == ",".join(fc.FUNDAMENTALS_COLS)

    def test_end_to_end_through_real_csv_parse(self, monkeypatch):
        # Integration: exercise get_fundamentals -> _get_csv -> real
        # csv.DictReader, so a HEADER_TO_KEY / SCALE_MULTIPLIERS regression
        # is caught through the actual parse path (all other tests in this
        # class stub _get_csv and bypass the CSV reader entirely).
        c = _make_client()
        header = ("Ticker,Company,Sector,Industry,Country,Exchange,"
                  "Market Cap,Shares Outstanding,Shares Float,"
                  "Institutional Ownership,Short Float,Average Volume,Price")
        row = ("AAPL,Apple Inc,Tech,Consumer,USA,NASD,3500000,15000,14000,"
               "60.5%,0.8%,80000,200.25")
        resp = _FakeResponse(text=header + "\n" + row + "\n")
        monkeypatch.setattr(c.session, "get", lambda *a, **k: resp)
        out = c.get_fundamentals("aapl")
        # millions-scaled
        assert out["market_cap"] == pytest.approx(3_500_000 * 1e6)
        assert out["shares_outstanding"] == pytest.approx(15_000 * 1e6)
        assert out["float_shares"] == pytest.approx(14_000 * 1e6)
        # thousands-scaled
        assert out["avg_volume"] == pytest.approx(80_000 * 1e3)
        # unscaled percent / price
        assert out["institutional_ownership_pct"] == pytest.approx(60.5)
        assert out["short_interest_pct"] == pytest.approx(0.8)
        assert out["price"] == pytest.approx(200.25)
        # string passthrough survives the round-trip
        assert out["ticker"] == "AAPL"
        assert out["company"] == "Apple Inc"

    def test_get_csv_none_returns_none(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(c, "_get_csv", lambda *a, **k: None)
        assert c.get_fundamentals("AAPL") is None

    def test_get_csv_empty_list_returns_none(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(c, "_get_csv", lambda *a, **k: [])
        assert c.get_fundamentals("AAPL") is None

    def test_market_cap_scaled_by_million(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(
            c, "_get_csv",
            lambda *a, **k: [{"Ticker": "X", "Market Cap": "500"}])
        out = c.get_fundamentals("X")
        assert out["market_cap"] == pytest.approx(500_000_000.0)

    def test_shares_and_float_scaled_by_million(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(
            c, "_get_csv",
            lambda *a, **k: [{"Shares Outstanding": "12.5",
                              "Shares Float": "10"}])
        out = c.get_fundamentals("X")
        assert out["shares_outstanding"] == pytest.approx(12_500_000.0)
        assert out["float_shares"] == pytest.approx(10_000_000.0)

    def test_avg_volume_scaled_by_thousand(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(
            c, "_get_csv",
            lambda *a, **k: [{"Average Volume": "1234"}])
        out = c.get_fundamentals("X")
        assert out["avg_volume"] == pytest.approx(1_234_000.0)

    def test_short_float_percent_not_scaled(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(
            c, "_get_csv",
            lambda *a, **k: [{"Short Float": "5.2%"}])
        out = c.get_fundamentals("X")
        # percent not scaled, not divided by 100
        assert out["short_interest_pct"] == pytest.approx(5.2)

    def test_institutional_ownership_not_scaled(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(
            c, "_get_csv",
            lambda *a, **k: [{"Institutional Ownership": "33.3%"}])
        out = c.get_fundamentals("X")
        assert out["institutional_ownership_pct"] == pytest.approx(33.3)

    def test_price_parsed_not_scaled(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(
            c, "_get_csv",
            lambda *a, **k: [{"Price": "$45.20"}])
        out = c.get_fundamentals("X")
        assert out["price"] == pytest.approx(45.2)

    def test_missing_header_absent_from_out(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(
            c, "_get_csv",
            lambda *a, **k: [{"Ticker": "X", "Price": "10"}])
        out = c.get_fundamentals("X")
        # no Market Cap header in row -> key absent, no KeyError
        assert "market_cap" not in out
        assert "shares_outstanding" not in out

    def test_numeric_dash_parsed_none_scale_skipped(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(
            c, "_get_csv",
            lambda *a, **k: [{"Market Cap": "-"}])
        out = c.get_fundamentals("X")
        # '-' -> None; scale step skips None (does not multiply)
        assert out["market_cap"] is None

    def test_string_passthrough_fields(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(
            c, "_get_csv",
            lambda *a, **k: [{"Ticker": "X", "Company": "Acme Inc",
                              "Sector": "Tech", "Industry": "Soft",
                              "Country": "USA", "Exchange": "NASD"}])
        out = c.get_fundamentals("X")
        assert out["ticker"] == "X"
        assert out["company"] == "Acme Inc"
        assert out["sector"] == "Tech"
        assert out["industry"] == "Soft"
        assert out["country"] == "USA"
        assert out["exchange"] == "NASD"

    def test_uses_only_first_row(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(
            c, "_get_csv",
            lambda *a, **k: [{"Ticker": "FIRST", "Price": "1"},
                             {"Ticker": "SECOND", "Price": "2"}])
        out = c.get_fundamentals("X")
        assert out["ticker"] == "FIRST"
        assert out["price"] == pytest.approx(1.0)


# ── FinvizClient.get_daily_closes ─────────────────────────────────────


class TestGetDailyCloses:
    @pytest.mark.parametrize("ticker", ["", None])
    def test_falsy_ticker_returns_none(self, ticker, monkeypatch):
        c = _make_client()
        # both falsy variants short-circuit before any _get_csv call
        called = {"n": 0}

        def _spy(*a, **k):
            called["n"] += 1
            return [{"Date": "06/04/2026", "Close": "1.0"}]

        monkeypatch.setattr(c, "_get_csv", _spy)
        assert c.get_daily_closes(ticker) is None
        assert called["n"] == 0

    def test_live_bar_date_with_whitespace_is_stripped_and_dropped(
            self, monkeypatch):
        # The live-bar comparison does (Date or "").strip(); a Date with
        # surrounding whitespace that matches today must still be dropped.
        c = _make_client()
        rows = [
            {"Date": "06/04/2026", "Close": "2.0"},
            {"Date": "  06/05/2026  ", "Close": "99.0"},  # padded live bar
        ]
        monkeypatch.setattr(c, "_get_csv", lambda *a, **k: list(rows))
        monkeypatch.setattr(fc, "_current_session_date",
                            lambda: "06/05/2026")
        out = c.get_daily_closes("X", bars=60)
        assert out == [2.0]  # whitespace stripped -> matched today -> dropped

    @pytest.mark.parametrize("bars,expected", [(2, [1.0, 2.0]), (1, [2.0])])
    def test_live_drop_then_truncate_yields_bars_settled_closes(
            self, bars, expected, monkeypatch):
        # The documented bars+1 design: request one extra, drop the live
        # bar, then keep the most-recent `bars` settled closes. Exercise
        # the drop and the rows[-bars:] truncation together (every other
        # test isolates one or the other).
        c = _make_client()
        rows = [
            {"Date": "06/03/2026", "Close": "1.0"},
            {"Date": "06/04/2026", "Close": "2.0"},
            {"Date": "06/05/2026", "Close": "99.0"},  # live bar -> dropped
        ]
        monkeypatch.setattr(c, "_get_csv", lambda *a, **k: list(rows))
        monkeypatch.setattr(fc, "_current_session_date",
                            lambda: "06/05/2026")
        out = c.get_daily_closes("X", bars=bars)
        assert out == expected

    def test_requests_bars_plus_one(self, monkeypatch):
        c = _make_client()
        captured = {}

        def _get_csv(endpoint, params):
            captured["endpoint"] = endpoint
            captured["params"] = params
            return [{"Date": "01/01/2026", "Close": "1"}]

        monkeypatch.setattr(c, "_get_csv", _get_csv)
        monkeypatch.setattr(fc, "_current_session_date",
                            lambda: "12/31/2025")
        c.get_daily_closes("X", bars=60)
        assert captured["endpoint"] == "quote_export"
        assert captured["params"]["barsCount"] == "61"
        assert captured["params"]["t"] == "X"
        assert captured["params"]["r"] == "d1"

    def test_no_rows_returns_none(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(c, "_get_csv", lambda *a, **k: None)
        assert c.get_daily_closes("X") is None

    def test_live_bar_dropped_when_last_date_is_today(self, monkeypatch):
        c = _make_client()
        rows = [
            {"Date": "06/03/2026", "Close": "1.0"},
            {"Date": "06/04/2026", "Close": "2.0"},
            {"Date": "06/05/2026", "Close": "99.0"},  # live bar
        ]
        monkeypatch.setattr(c, "_get_csv", lambda *a, **k: list(rows))
        monkeypatch.setattr(fc, "_current_session_date",
                            lambda: "06/05/2026")
        out = c.get_daily_closes("X", bars=60)
        assert out == [1.0, 2.0]  # live bar (99.0) dropped

    def test_no_drop_when_last_date_not_today(self, monkeypatch):
        c = _make_client()
        rows = [
            {"Date": "06/03/2026", "Close": "1.0"},
            {"Date": "06/04/2026", "Close": "2.0"},
        ]
        monkeypatch.setattr(c, "_get_csv", lambda *a, **k: list(rows))
        monkeypatch.setattr(fc, "_current_session_date",
                            lambda: "06/05/2026")
        out = c.get_daily_closes("X", bars=60)
        assert out == [1.0, 2.0]

    def test_missing_date_on_last_row_no_drop(self, monkeypatch):
        c = _make_client()
        rows = [
            {"Close": "1.0"},  # no Date key
            {"Close": "2.0"},  # no Date key on last either
        ]
        monkeypatch.setattr(c, "_get_csv", lambda *a, **k: list(rows))
        monkeypatch.setattr(fc, "_current_session_date",
                            lambda: "06/05/2026")
        out = c.get_daily_closes("X", bars=60)
        # missing Date -> "" != today -> no drop
        assert out == [1.0, 2.0]

    def test_within_calendar_days_none_no_filter(self, monkeypatch):
        c = _make_client()
        rows = [
            {"Date": "01/01/2020", "Close": "1.0"},  # very old
            {"Date": "06/04/2026", "Close": "2.0"},
        ]
        monkeypatch.setattr(c, "_get_csv", lambda *a, **k: list(rows))
        monkeypatch.setattr(fc, "_current_session_date",
                            lambda: "06/05/2026")
        out = c.get_daily_closes("X", bars=60, within_calendar_days=None)
        assert out == [1.0, 2.0]  # old row kept

    def test_within_calendar_days_filters_old_keeps_boundary(self,
                                                             monkeypatch):
        c = _make_client()
        # _market_now used by cutoff; fix it
        now = datetime(2026, 6, 5, 12, 0)
        monkeypatch.setattr(fc, "_market_now", lambda: now)
        monkeypatch.setattr(fc, "_current_session_date",
                            lambda: "06/05/2026")
        cutoff = (now - timedelta(days=10)).date()  # 05/26/2026
        rows = [
            {"Date": "05/25/2026", "Close": "1.0"},  # before cutoff -> drop
            {"Date": "05/26/2026", "Close": "2.0"},  # == cutoff -> KEEP
            {"Date": "06/04/2026", "Close": "3.0"},  # within -> keep
        ]
        monkeypatch.setattr(c, "_get_csv", lambda *a, **k: list(rows))
        out = c.get_daily_closes("X", bars=60, within_calendar_days=10)
        assert cutoff == datetime(2026, 5, 26).date()
        assert out == [2.0, 3.0]  # 05/25 dropped, 05/26 boundary kept

    def test_malformed_date_excluded_by_window_filter(self, monkeypatch):
        c = _make_client()
        now = datetime(2026, 6, 5, 12, 0)
        monkeypatch.setattr(fc, "_market_now", lambda: now)
        monkeypatch.setattr(fc, "_current_session_date",
                            lambda: "06/05/2026")
        rows = [
            {"Date": "garbage", "Close": "1.0"},  # ValueError -> excluded
            {"Date": "06/04/2026", "Close": "2.0"},
        ]
        monkeypatch.setattr(c, "_get_csv", lambda *a, **k: list(rows))
        out = c.get_daily_closes("X", bars=60, within_calendar_days=10)
        assert out == [2.0]

    def test_truncation_keeps_most_recent_bars(self, monkeypatch):
        c = _make_client()
        # 5 settled rows + a non-today last row; bars=3 keeps last 3
        rows = [
            {"Date": "06/01/2026", "Close": "1.0"},
            {"Date": "06/02/2026", "Close": "2.0"},
            {"Date": "06/03/2026", "Close": "3.0"},
            {"Date": "06/04/2026", "Close": "4.0"},
            {"Date": "06/05/2026", "Close": "5.0"},
        ]
        monkeypatch.setattr(c, "_get_csv", lambda *a, **k: list(rows))
        monkeypatch.setattr(fc, "_current_session_date",
                            lambda: "06/09/2026")  # not today -> no drop
        out = c.get_daily_closes("X", bars=3)
        assert out == [3.0, 4.0, 5.0]  # last 3, oldest-first

    def test_unparseable_close_skipped(self, monkeypatch):
        c = _make_client()
        rows = [
            {"Date": "06/03/2026", "Close": "1.0"},
            {"Date": "06/04/2026", "Close": "-"},     # skipped
            {"Date": "06/05/2026", "Close": "3.0"},
        ]
        monkeypatch.setattr(c, "_get_csv", lambda *a, **k: list(rows))
        monkeypatch.setattr(fc, "_current_session_date",
                            lambda: "06/09/2026")
        out = c.get_daily_closes("X", bars=60)
        assert out == [1.0, 3.0]  # dash row dropped, not None in list

    def test_all_unparseable_returns_none(self, monkeypatch):
        c = _make_client()
        rows = [
            {"Date": "06/03/2026", "Close": "-"},
            {"Date": "06/04/2026", "Close": "N/A"},
        ]
        monkeypatch.setattr(c, "_get_csv", lambda *a, **k: list(rows))
        monkeypatch.setattr(fc, "_current_session_date",
                            lambda: "06/09/2026")
        assert c.get_daily_closes("X", bars=60) is None

    def test_zero_rows_after_window_returns_none(self, monkeypatch):
        c = _make_client()
        now = datetime(2026, 6, 5, 12, 0)
        monkeypatch.setattr(fc, "_market_now", lambda: now)
        monkeypatch.setattr(fc, "_current_session_date",
                            lambda: "06/05/2026")
        rows = [{"Date": "01/01/2020", "Close": "1.0"}]  # all too old
        monkeypatch.setattr(c, "_get_csv", lambda *a, **k: list(rows))
        assert c.get_daily_closes("X", bars=60,
                                  within_calendar_days=10) is None

    def test_output_is_oldest_first(self, monkeypatch):
        c = _make_client()
        rows = [
            {"Date": "06/01/2026", "Close": "10.0"},
            {"Date": "06/02/2026", "Close": "20.0"},
            {"Date": "06/03/2026", "Close": "30.0"},
        ]
        monkeypatch.setattr(c, "_get_csv", lambda *a, **k: list(rows))
        monkeypatch.setattr(fc, "_current_session_date",
                            lambda: "06/09/2026")
        out = c.get_daily_closes("X", bars=60)
        assert out == [10.0, 20.0, 30.0]


# ── FinvizClient.highest_close ────────────────────────────────────────


class TestHighestCloseMethod:
    def test_default_within_calendar_days_is_60(self, monkeypatch):
        c = _make_client()
        captured = {}

        def _gdc(ticker, bars=60, within_calendar_days=None):
            captured["bars"] = bars
            captured["within"] = within_calendar_days
            return [1.0, 2.0]

        monkeypatch.setattr(c, "get_daily_closes", _gdc)
        c.highest_close("X")
        assert captured["within"] == 60
        assert captured["bars"] == 60

    def test_returns_max(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(c, "get_daily_closes",
                            lambda *a, **k: [1.0, 2.18, 6.6, 0.5])
        assert c.highest_close("X") == pytest.approx(6.6)

    def test_none_closes_returns_none(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(c, "get_daily_closes", lambda *a, **k: None)
        assert c.highest_close("X") is None

    def test_empty_list_returns_none(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(c, "get_daily_closes", lambda *a, **k: [])
        assert c.highest_close("X") is None

    def test_single_element(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(c, "get_daily_closes", lambda *a, **k: [4.2])
        assert c.highest_close("X") == pytest.approx(4.2)


# ── _client ───────────────────────────────────────────────────────────


class TestClientSingleton:
    def test_returns_same_instance(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "FINVIZ_API_KEY", "k", raising=False)
        monkeypatch.setattr(config, "FINVIZ_BASE_URL", "https://x",
                            raising=False)
        a = fc._client()
        b = fc._client()
        assert a is b
        assert isinstance(a, fc.FinvizClient)

    def test_reset_global_yields_new_instance(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "FINVIZ_API_KEY", "k", raising=False)
        monkeypatch.setattr(config, "FINVIZ_BASE_URL", "https://x",
                            raising=False)
        a = fc._client()
        fc._default_client = None
        b = fc._client()
        assert a is not b


# ── fundamentals (module-level cached) ────────────────────────────────


class _StubClient:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def get_fundamentals(self, ticker):
        self.calls += 1
        return self.result


class TestFundamentals:
    def test_cache_key_uppercased(self, monkeypatch):
        stub = _StubClient({"ticker": "AAPL"})
        monkeypatch.setattr(fc, "_client", lambda: stub)
        monkeypatch.setattr(fc.time, "monotonic", lambda: 1000.0)
        fc.fundamentals("aapl")
        # 'AAPL' shares the same cache entry -> no second call
        fc.fundamentals("AAPL")
        assert stub.calls == 1
        assert "AAPL" in fc._fund_cache

    def test_first_call_populates_and_calls_once(self, monkeypatch):
        stub = _StubClient({"ticker": "X"})
        monkeypatch.setattr(fc, "_client", lambda: stub)
        monkeypatch.setattr(fc.time, "monotonic", lambda: 0.0)
        result = fc.fundamentals("X")
        assert result == {"ticker": "X"}
        assert stub.calls == 1

    def test_second_call_within_ttl_uses_cache(self, monkeypatch):
        stub = _StubClient({"ticker": "X"})
        monkeypatch.setattr(fc, "_client", lambda: stub)
        clock = {"t": 0.0}
        monkeypatch.setattr(fc.time, "monotonic", lambda: clock["t"])
        fc.fundamentals("X")
        clock["t"] = fc._FUND_TTL - 1  # still within window
        fc.fundamentals("X")
        assert stub.calls == 1

    def test_expired_positive_entry_refetches(self, monkeypatch):
        stub = _StubClient({"ticker": "X"})
        monkeypatch.setattr(fc, "_client", lambda: stub)
        clock = {"t": 0.0}
        monkeypatch.setattr(fc.time, "monotonic", lambda: clock["t"])
        fc.fundamentals("X")
        clock["t"] = fc._FUND_TTL  # now - ts >= TTL -> refetch
        fc.fundamentals("X")
        assert stub.calls == 2

    def test_none_result_uses_short_neg_ttl(self, monkeypatch):
        stub = _StubClient(None)
        monkeypatch.setattr(fc, "_client", lambda: stub)
        clock = {"t": 0.0}
        monkeypatch.setattr(fc.time, "monotonic", lambda: clock["t"])
        assert fc.fundamentals("X") is None
        # within 30s -> cached, no refetch
        clock["t"] = fc._FUND_TTL_NEG - 1
        assert fc.fundamentals("X") is None
        assert stub.calls == 1
        # at/after 30s -> refetch
        clock["t"] = fc._FUND_TTL_NEG
        fc.fundamentals("X")
        assert stub.calls == 2

    def test_none_not_held_for_full_positive_ttl(self, monkeypatch):
        # A None result must NOT be cached for the full 600s positive TTL.
        stub = _StubClient(None)
        monkeypatch.setattr(fc, "_client", lambda: stub)
        clock = {"t": 0.0}
        monkeypatch.setattr(fc.time, "monotonic", lambda: clock["t"])
        fc.fundamentals("X")
        clock["t"] = fc._FUND_TTL_NEG + 5  # past neg TTL, well under pos TTL
        fc.fundamentals("X")
        assert stub.calls == 2


# ── highest_close (module-level) ──────────────────────────────────────


class TestHighestCloseModule:
    def test_delegates_to_client(self, monkeypatch):
        captured = {}

        class _C:
            def highest_close(self, ticker, bars=60,
                              within_calendar_days=None):
                captured["ticker"] = ticker
                captured["bars"] = bars
                captured["within"] = within_calendar_days
                return 3.5

        monkeypatch.setattr(fc, "_client", lambda: _C())
        out = fc.highest_close("X")
        assert out == pytest.approx(3.5)
        assert captured["ticker"] == "X"
        assert captured["bars"] == 60
        assert captured["within"] == 60  # module default forwards 60


# ── ib6_effective_price ───────────────────────────────────────────────


class TestIb6EffectivePrice:
    def test_both_none_returns_none(self, monkeypatch):
        monkeypatch.setattr(fc, "highest_close", lambda *a, **k: None)
        assert fc.ib6_effective_price("X", current_price=None) is None

    def test_current_none_high_present(self, monkeypatch):
        monkeypatch.setattr(fc, "highest_close", lambda *a, **k: 2.18)
        assert fc.ib6_effective_price("X", current_price=None) == \
            pytest.approx(2.18)

    def test_high_none_current_present(self, monkeypatch):
        monkeypatch.setattr(fc, "highest_close", lambda *a, **k: None)
        assert fc.ib6_effective_price("X", current_price=3.0) == \
            pytest.approx(3.0)

    def test_high_wins_pump_case(self, monkeypatch):
        monkeypatch.setattr(fc, "highest_close", lambda *a, **k: 3.08)
        assert fc.ib6_effective_price("X", current_price=1.6) == \
            pytest.approx(3.08)

    def test_current_wins(self, monkeypatch):
        monkeypatch.setattr(fc, "highest_close", lambda *a, **k: 2.0)
        assert fc.ib6_effective_price("X", current_price=5.0) == \
            pytest.approx(5.0)

    def test_equal_values(self, monkeypatch):
        monkeypatch.setattr(fc, "highest_close", lambda *a, **k: 4.0)
        assert fc.ib6_effective_price("X", current_price=4.0) == \
            pytest.approx(4.0)

    def test_current_zero_high_none_not_coerced(self, monkeypatch):
        # 0.0 is not None and must NOT become None
        monkeypatch.setattr(fc, "highest_close", lambda *a, **k: None)
        result = fc.ib6_effective_price("X", current_price=0.0)
        assert result == 0.0
        assert result is not None

    def test_forwards_bars_60_to_highest_close(self, monkeypatch):
        captured = {}

        def _hc(ticker, bars=60):
            captured["ticker"] = ticker
            captured["bars"] = bars
            return 1.0

        monkeypatch.setattr(fc, "highest_close", _hc)
        fc.ib6_effective_price("X", current_price=None)
        assert captured["bars"] == 60
        assert captured["ticker"] == "X"


# ── FinvizClient.daily_bars ───────────────────────────────────────────


class TestDailyBars:
    """The dated form of get_daily_closes. The drop-the-live-bar and
    window rules are covered by TestGetDailyCloses (which now delegates
    here); these tests pin the date pairing and the parse-failure paths
    that only the pair form can express."""

    def test_pairs_dates_with_closes(self, monkeypatch):
        c = _make_client()
        rows = [
            {"Date": "06/03/2026", "Close": "1.0"},
            {"Date": "06/04/2026", "Close": "2.0"},
            {"Date": "06/05/2026", "Close": "99.0"},  # live bar -> dropped
        ]
        monkeypatch.setattr(c, "_get_csv", lambda *a, **k: list(rows))
        monkeypatch.setattr(fc, "_current_session_date", lambda: "06/05/2026")
        assert c.daily_bars("X", bars=60) == [
            (date(2026, 6, 3), 1.0), (date(2026, 6, 4), 2.0)]

    def test_unparseable_date_still_yields_its_close(self, monkeypatch):
        # Behaviour parity with get_daily_closes, which never parsed Date
        # outside the window filter: a bad Date must not drop the bar.
        c = _make_client()
        monkeypatch.setattr(c, "_get_csv", lambda *a, **k: [
            {"Date": "not-a-date", "Close": "3.0"}])
        monkeypatch.setattr(fc, "_current_session_date", lambda: "06/05/2026")
        assert c.daily_bars("X", bars=60) == [(None, 3.0)]

    def test_unparseable_close_drops_the_bar(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(c, "_get_csv", lambda *a, **k: [
            {"Date": "06/03/2026", "Close": "—"},
            {"Date": "06/04/2026", "Close": "2.0"}])
        monkeypatch.setattr(fc, "_current_session_date", lambda: "06/05/2026")
        assert c.daily_bars("X", bars=60) == [(date(2026, 6, 4), 2.0)]

    def test_get_daily_closes_is_the_close_only_projection(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(c, "daily_bars", lambda *a, **k: [
            (date(2026, 6, 3), 1.0), (None, 2.0)])
        assert c.get_daily_closes("X") == [1.0, 2.0]

    @pytest.mark.parametrize("pairs", [None, []])
    def test_get_daily_closes_none_when_no_bars(self, pairs, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(c, "daily_bars", lambda *a, **k: pairs)
        assert c.get_daily_closes("X") is None


# ── _parse_session_date ───────────────────────────────────────────────


class TestParseSessionDate:
    def test_parses_finviz_format(self):
        assert fc._parse_session_date("06/04/2026") == date(2026, 6, 4)

    def test_strips_padding(self):
        assert fc._parse_session_date("  06/04/2026 ") == date(2026, 6, 4)

    @pytest.mark.parametrize("value", [None, "", "2026-06-04", "junk", 7])
    def test_bad_input_is_none(self, value):
        assert fc._parse_session_date(value) is None


# ── latest_settled_close ──────────────────────────────────────────────


class TestLatestSettledClose:
    def test_returns_the_most_recent_settled_pair(self, monkeypatch):
        captured = {}

        class _C:
            def daily_bars(self, ticker, bars=60, within_calendar_days=None):
                captured["bars"] = bars
                captured["within"] = within_calendar_days
                return [(date(2026, 7, 24), 0.31), (date(2026, 7, 27), 0.304)]

        monkeypatch.setattr(fc, "_client", lambda: _C())
        assert fc.latest_settled_close("X") == (date(2026, 7, 27), 0.304)
        # No calendar window: the last settled session may be days back
        # over a holiday weekend or on a thinly-traded ticker.
        assert captured["within"] is None
        assert captured["bars"] == 1

    @pytest.mark.parametrize("pairs", [None, []])
    def test_none_when_export_unavailable(self, pairs, monkeypatch):
        monkeypatch.setattr(fc, "_client",
                            lambda: type("C", (), {
                                "daily_bars": lambda *a, **k: pairs})())
        assert fc.latest_settled_close("X") is None
