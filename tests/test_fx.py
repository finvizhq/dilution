"""Unit tests for dilution/fx.py — currency → USD conversion.

The module never touches the DB; the autouse temp_db fixture from
conftest.py is harmless and simply ignored here. All filesystem isolation
is achieved by monkeypatching the module-level Path constants
(_CACHE_DIR / _FINVIZ_DIR / _FRANK_DIR) onto tmp_path, and all network is
mocked at the exact seams (_finviz_quote and urllib.request.urlopen).
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

import dilution.fx as fx


# ──────────────────────────────────────────────────────────────────────
# Filesystem isolation: point all on-disk cache dirs at a tmp_path so we
# never read or write the real ~/.cache/dilution/fx.
# ──────────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def isolate_cache_dirs(tmp_path, monkeypatch):
    cache = tmp_path / "fx"
    finviz = cache / "finviz"
    frank = cache / "frankfurter"
    monkeypatch.setattr(fx, "_CACHE_DIR", cache)
    monkeypatch.setattr(fx, "_FINVIZ_DIR", finviz)
    monkeypatch.setattr(fx, "_FRANK_DIR", frank)
    return cache


# ══════════════════════════════════════════════════════════════════════
# to_usd
# ══════════════════════════════════════════════════════════════════════
class TestToUsd:
    ON = date(2024, 1, 5)

    def test_usd_returns_amount_verbatim_no_rate_lookup(self, monkeypatch):
        # _rate must NOT be called for USD. Sabotage it to prove it.
        def boom(*a, **k):
            raise AssertionError("_rate should not be called for USD")
        monkeypatch.setattr(fx, "_rate", boom)
        assert fx.to_usd(123.45, "USD", self.ON) == 123.45

    def test_usd_zero_short_circuits(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("_rate should not be called for USD")
        monkeypatch.setattr(fx, "_rate", boom)
        assert fx.to_usd(0, "USD", self.ON) == 0.0

    def test_usd_negative_returns_verbatim(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("_rate should not be called for USD")
        monkeypatch.setattr(fx, "_rate", boom)
        assert fx.to_usd(-50.0, "USD", self.ON) == -50.0

    def test_lowercase_usd_treated_as_usd(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("_rate should not be called for USD")
        monkeypatch.setattr(fx, "_rate", boom)
        assert fx.to_usd(99.0, "usd", self.ON) == 99.0

    def test_lowercase_currency_uppercased_before_rate(self, monkeypatch):
        seen = {}

        def fake_rate(currency, on):
            seen["currency"] = currency
            seen["on"] = on
            return 1.1
        monkeypatch.setattr(fx, "_rate", fake_rate)
        result = fx.to_usd(100.0, "eur", self.ON)
        assert seen["currency"] == "EUR"   # uppercased
        assert seen["on"] == self.ON
        assert result == pytest.approx(110.0)

    def test_rate_none_returns_none(self, monkeypatch):
        monkeypatch.setattr(fx, "_rate", lambda c, o: None)
        assert fx.to_usd(100.0, "EUR", self.ON) is None

    def test_rate_float_multiplies(self, monkeypatch):
        monkeypatch.setattr(fx, "_rate", lambda c, o: 1.1)
        assert fx.to_usd(100.0, "EUR", self.ON) == pytest.approx(110.0)

    def test_zero_amount_non_usd_with_rate_is_zero(self, monkeypatch):
        monkeypatch.setattr(fx, "_rate", lambda c, o: 1.1)
        assert fx.to_usd(0, "EUR", self.ON) == 0.0

    def test_negative_amount_propagates_sign(self, monkeypatch):
        monkeypatch.setattr(fx, "_rate", lambda c, o: 2.0)
        assert fx.to_usd(-25.0, "GBP", self.ON) == pytest.approx(-50.0)


# ══════════════════════════════════════════════════════════════════════
# _rate — source selection
# ══════════════════════════════════════════════════════════════════════
class TestRate:
    ON = date(2024, 1, 5)

    def test_major_finviz_hit_short_circuits_frankfurter(self, monkeypatch):
        called = {"frank": False}
        monkeypatch.setattr(fx, "_finviz_rate", lambda c, o: 1.08)

        def fake_frank(c, o):
            called["frank"] = True
            return 9.99
        monkeypatch.setattr(fx, "_frankfurter_rate", fake_frank)

        assert fx._rate("EUR", self.ON) == 1.08
        assert called["frank"] is False  # never consulted on a Finviz hit

    def test_major_finviz_miss_falls_back_to_frankfurter(self, monkeypatch):
        monkeypatch.setattr(fx, "_finviz_rate", lambda c, o: None)
        monkeypatch.setattr(fx, "_frankfurter_rate", lambda c, o: 1.07)
        assert fx._rate("EUR", self.ON) == 1.07

    def test_non_major_skips_finviz(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("_finviz_rate should not be called for non-major")
        monkeypatch.setattr(fx, "_finviz_rate", boom)
        monkeypatch.setattr(fx, "_frankfurter_rate", lambda c, o: 0.14)
        assert fx._rate("CNY", self.ON) == 0.14

    def test_non_major_sgd_uses_frankfurter(self, monkeypatch):
        monkeypatch.setattr(fx, "_finviz_rate",
                            lambda c, o: pytest.fail("finviz called for SGD"))
        monkeypatch.setattr(fx, "_frankfurter_rate", lambda c, o: 0.74)
        assert fx._rate("SGD", self.ON) == 0.74

    def test_both_sources_none_returns_none(self, monkeypatch):
        monkeypatch.setattr(fx, "_finviz_rate", lambda c, o: None)
        monkeypatch.setattr(fx, "_frankfurter_rate", lambda c, o: None)
        assert fx._rate("EUR", self.ON) is None

    @pytest.mark.parametrize("ccy", ["EUR", "GBP", "AUD", "NZD", "JPY", "CHF", "CAD"])
    def test_every_finviz_pair_routes_through_finviz_first(self, ccy, monkeypatch):
        finviz_called = {"v": False}

        def fake_finviz(c, o):
            finviz_called["v"] = True
            return 2.5
        monkeypatch.setattr(fx, "_finviz_rate", fake_finviz)
        monkeypatch.setattr(fx, "_frankfurter_rate",
                            lambda c, o: pytest.fail("frankfurter called on finviz hit"))
        assert fx._rate(ccy, self.ON) == 2.5
        assert finviz_called["v"] is True

    def test_sek_is_not_a_finviz_pair(self, monkeypatch):
        # SEK is a major but not in _FINVIZ_PAIRS -> straight to frankfurter.
        monkeypatch.setattr(fx, "_finviz_rate",
                            lambda c, o: pytest.fail("finviz called for SEK"))
        monkeypatch.setattr(fx, "_frankfurter_rate", lambda c, o: 0.095)
        assert fx._rate("SEK", self.ON) == 0.095


# ══════════════════════════════════════════════════════════════════════
# _finviz_rate
# ══════════════════════════════════════════════════════════════════════
class TestFinvizRate:
    def _series(self):
        return [
            (date(2024, 1, 2), 1.05),
            (date(2024, 1, 3), 1.06),
            (date(2024, 1, 5), 1.08),
        ]

    def test_series_none_returns_none(self, monkeypatch):
        monkeypatch.setattr(fx, "_finviz_series", lambda c: None)
        assert fx._finviz_rate("EUR", date(2024, 1, 5)) is None

    def test_series_empty_returns_none(self, monkeypatch):
        monkeypatch.setattr(fx, "_finviz_series", lambda c: [])
        assert fx._finviz_rate("EUR", date(2024, 1, 5)) is None

    def test_on_before_earliest_returns_none(self, monkeypatch):
        monkeypatch.setattr(fx, "_finviz_series", lambda c: self._series())
        # earliest is 2024-01-02; ask for 2024-01-01 -> out of window
        assert fx._finviz_rate("EUR", date(2024, 1, 1)) is None

    def test_on_exactly_earliest_is_in_window(self, monkeypatch):
        monkeypatch.setattr(fx, "_finviz_series", lambda c: self._series())
        # boundary: on == earliest (uses < not <=) -> that close
        assert fx._finviz_rate("EUR", date(2024, 1, 2)) == pytest.approx(1.05)

    def test_non_invert_pair_returns_close_directly(self, monkeypatch):
        monkeypatch.setattr(fx, "_finviz_series", lambda c: self._series())
        assert fx._finviz_rate("EUR", date(2024, 1, 5)) == pytest.approx(1.08)

    def test_invert_pair_jpy_returns_reciprocal(self, monkeypatch):
        # USDJPY base; close=150 -> USD per JPY = 1/150
        monkeypatch.setattr(fx, "_finviz_series",
                            lambda c: [(date(2024, 1, 2), 150.0)])
        rate = fx._finviz_rate("JPY", date(2024, 1, 5))
        assert rate == pytest.approx(1.0 / 150.0)

    @pytest.mark.parametrize("ccy,close", [("CHF", 0.9), ("CAD", 1.35)])
    def test_invert_pairs_chf_cad(self, ccy, close, monkeypatch):
        monkeypatch.setattr(fx, "_finviz_series",
                            lambda c: [(date(2024, 1, 2), close)])
        rate = fx._finviz_rate(ccy, date(2024, 1, 5))
        assert rate == pytest.approx(1.0 / close)

    def test_on_after_last_bar_forward_fills(self, monkeypatch):
        monkeypatch.setattr(fx, "_finviz_series", lambda c: self._series())
        # 2024-02-01 is after last bar (2024-01-05) -> uses last close
        assert fx._finviz_rate("EUR", date(2024, 2, 1)) == pytest.approx(1.08)

    def test_between_bars_uses_on_or_before(self, monkeypatch):
        monkeypatch.setattr(fx, "_finviz_series", lambda c: self._series())
        # 2024-01-04 is between 01-03 and 01-05 -> uses 01-03 close
        assert fx._finviz_rate("EUR", date(2024, 1, 4)) == pytest.approx(1.06)

    def test_nearest_close_none_branch_returns_none(self, monkeypatch):
        # Defensive branch (fx.py: `if close is None: return None`). It is
        # unreachable through _finviz_series because the `on < earliest`
        # window guard fires first, so drive it by forcing _nearest_close to
        # None while passing the earliest guard (on == earliest).
        monkeypatch.setattr(fx, "_finviz_series", lambda c: self._series())
        monkeypatch.setattr(fx, "_nearest_close", lambda series, target: None)
        # earliest is 2024-01-02, so on==earliest clears the window guard and
        # we reach the _nearest_close-None branch instead of the invert path.
        assert fx._finviz_rate("EUR", date(2024, 1, 2)) is None

    def test_nearest_close_none_branch_skips_invert(self, monkeypatch):
        # Same branch on an invert pair: proves the None short-circuits BEFORE
        # the `1.0 / close` inversion (a None close would otherwise TypeError).
        monkeypatch.setattr(fx, "_finviz_series",
                            lambda c: [(date(2024, 1, 2), 150.0)])
        monkeypatch.setattr(fx, "_nearest_close", lambda series, target: None)
        assert fx._finviz_rate("JPY", date(2024, 1, 5)) is None


# ══════════════════════════════════════════════════════════════════════
# _finviz_series — cache + fetch + parsing
# ══════════════════════════════════════════════════════════════════════
class TestFinvizSeries:
    def _write_cache(self, fetched_at, series_iso):
        fx._FINVIZ_DIR.mkdir(parents=True, exist_ok=True)
        cache = fx._FINVIZ_DIR / "EUR.json"
        cache.write_text(json.dumps({
            "fetched_at": fetched_at,
            "pair": "EURUSD",
            "series": series_iso,
        }))
        return cache

    def test_fresh_cache_used_without_fetch(self, monkeypatch):
        import time
        self._write_cache(time.time(), [["2024-01-02", 1.05], ["2024-01-05", 1.08]])

        def boom(pair):
            raise AssertionError("_finviz_quote should not be called on a fresh cache")
        monkeypatch.setattr(fx, "_finviz_quote", boom)

        series = fx._finviz_series("EUR")
        assert series == [(date(2024, 1, 2), 1.05), (date(2024, 1, 5), 1.08)]

    def test_fresh_cache_dates_parsed_via_fromisoformat(self, monkeypatch):
        import time
        self._write_cache(time.time(), [["2024-03-04", 1.10]])
        monkeypatch.setattr(fx, "_finviz_quote",
                            lambda p: pytest.fail("should not fetch"))
        series = fx._finviz_series("EUR")
        assert series[0][0] == date(2024, 3, 4)

    def test_stale_cache_triggers_refetch(self, monkeypatch):
        # fetched_at=0 is far past -> stale -> re-fetch.
        self._write_cache(0, [["2020-01-01", 1.00]])
        monkeypatch.setattr(fx, "_finviz_quote",
                            lambda p: [{"Date": "01/05/2024", "Close": "1.08"}])
        series = fx._finviz_series("EUR")
        assert series == [(date(2024, 1, 5), 1.08)]

    def test_missing_fetched_at_is_instantly_stale(self, monkeypatch):
        fx._FINVIZ_DIR.mkdir(parents=True, exist_ok=True)
        cache = fx._FINVIZ_DIR / "EUR.json"
        cache.write_text(json.dumps({"pair": "EURUSD",
                                     "series": [["2020-01-01", 1.0]]}))
        called = {"v": False}

        def fake_quote(pair):
            called["v"] = True
            return [{"Date": "01/05/2024", "Close": "1.08"}]
        monkeypatch.setattr(fx, "_finviz_quote", fake_quote)
        fx._finviz_series("EUR")
        assert called["v"] is True

    def test_malformed_cache_json_falls_through_to_fetch(self, monkeypatch):
        fx._FINVIZ_DIR.mkdir(parents=True, exist_ok=True)
        cache = fx._FINVIZ_DIR / "EUR.json"
        cache.write_text("{not valid json")
        monkeypatch.setattr(fx, "_finviz_quote",
                            lambda p: [{"Date": "01/05/2024", "Close": "1.08"}])
        series = fx._finviz_series("EUR")
        assert series == [(date(2024, 1, 5), 1.08)]

    def test_cache_missing_series_key_falls_through(self, monkeypatch):
        import time
        fx._FINVIZ_DIR.mkdir(parents=True, exist_ok=True)
        cache = fx._FINVIZ_DIR / "EUR.json"
        cache.write_text(json.dumps({"fetched_at": time.time(), "pair": "EURUSD"}))
        monkeypatch.setattr(fx, "_finviz_quote",
                            lambda p: [{"Date": "01/05/2024", "Close": "1.08"}])
        series = fx._finviz_series("EUR")
        assert series == [(date(2024, 1, 5), 1.08)]

    def test_cache_non_numeric_close_falls_through(self, monkeypatch):
        import time
        self._write_cache(time.time(), [["2024-01-02", "notanumber"]])
        monkeypatch.setattr(fx, "_finviz_quote",
                            lambda p: [{"Date": "01/05/2024", "Close": "1.08"}])
        series = fx._finviz_series("EUR")
        assert series == [(date(2024, 1, 5), 1.08)]

    def test_cache_null_close_falls_through_to_fetch(self, monkeypatch):
        # FIXED (was bug A#2): a JSON `null` close is malformed and now falls
        # through to a fresh re-fetch like the "notanumber" row above —
        # TypeError was added to the except guard so float(None) no longer
        # escapes uncaught.
        import time
        self._write_cache(time.time(), [["2024-01-02", None]])
        monkeypatch.setattr(fx, "_finviz_quote",
                            lambda p: [{"Date": "01/05/2024", "Close": "1.08"}])
        series = fx._finviz_series("EUR")
        assert series == [(date(2024, 1, 5), 1.08)]

    def test_fetch_none_returns_none_no_cache_write(self, monkeypatch):
        monkeypatch.setattr(fx, "_finviz_quote", lambda p: None)
        assert fx._finviz_series("EUR") is None
        assert not (fx._FINVIZ_DIR / "EUR.json").exists()

    def test_bad_rows_skipped_good_rows_kept(self, monkeypatch):
        rows = [
            {"Date": "bad-date", "Close": "1.0"},     # unparseable date
            {"Close": "1.0"},                          # missing Date
            {"Date": "01/05/2024"},                    # missing Close
            {"Date": "01/06/2024", "Close": "notnum"}, # non-float close
            {"Date": "01/07/2024", "Close": "1.09"},   # GOOD
        ]
        monkeypatch.setattr(fx, "_finviz_quote", lambda p: rows)
        series = fx._finviz_series("EUR")
        assert series == [(date(2024, 1, 7), 1.09)]

    def test_all_rows_malformed_returns_none_no_write(self, monkeypatch):
        rows = [{"Date": "bad", "Close": "x"}, {"foo": "bar"}]
        monkeypatch.setattr(fx, "_finviz_quote", lambda p: rows)
        assert fx._finviz_series("EUR") is None
        assert not (fx._FINVIZ_DIR / "EUR.json").exists()

    def test_rows_sorted_ascending(self, monkeypatch):
        rows = [
            {"Date": "01/07/2024", "Close": "3"},
            {"Date": "01/02/2024", "Close": "1"},
            {"Date": "01/05/2024", "Close": "2"},
        ]
        monkeypatch.setattr(fx, "_finviz_quote", lambda p: rows)
        series = fx._finviz_series("EUR")
        assert series == [
            (date(2024, 1, 2), 1.0),
            (date(2024, 1, 5), 2.0),
            (date(2024, 1, 7), 3.0),
        ]

    def test_date_parsed_month_day_year(self, monkeypatch):
        # '03/04/2024' is March 4 (US m/d/y), NOT April 3.
        monkeypatch.setattr(fx, "_finviz_quote",
                            lambda p: [{"Date": "03/04/2024", "Close": "1.1"}])
        series = fx._finviz_series("EUR")
        assert series[0][0] == date(2024, 3, 4)

    def test_successful_fetch_writes_roundtrippable_cache(self, monkeypatch):
        fixed = 1700000000.0
        monkeypatch.setattr(fx.time, "time", lambda: fixed)
        monkeypatch.setattr(fx, "_finviz_quote",
                            lambda p: [{"Date": "01/05/2024", "Close": "1.08"}])
        fx._finviz_series("EUR")
        cache = fx._FINVIZ_DIR / "EUR.json"
        assert cache.exists()
        payload = json.loads(cache.read_text())
        assert payload["fetched_at"] == fixed
        assert payload["pair"] == "EURUSD"
        assert payload["series"] == [["2024-01-05", 1.08]]

    def test_correct_pair_passed_to_quote(self, monkeypatch):
        seen = {}

        def fake_quote(pair):
            seen["pair"] = pair
            return [{"Date": "01/05/2024", "Close": "1.35"}]
        monkeypatch.setattr(fx, "_finviz_quote", fake_quote)
        fx._finviz_series("CAD")
        assert seen["pair"] == "USDCAD"


# ══════════════════════════════════════════════════════════════════════
# _nearest_close — pure forward-fill scan
# ══════════════════════════════════════════════════════════════════════
class TestNearestClose:
    SERIES = [
        (date(2024, 1, 2), 1.0),
        (date(2024, 1, 4), 2.0),
        (date(2024, 1, 6), 3.0),
    ]

    def test_target_one_day_before_first_returns_none(self):
        # SERIES[0] is 2024-01-02; the immediately-preceding day is out.
        assert fx._nearest_close(self.SERIES, date(2024, 1, 1)) is None

    def test_target_far_before_first_returns_none(self):
        # A target years before the series is also None (distinct input from
        # the one-day-before boundary, not a copy of it).
        assert fx._nearest_close(self.SERIES, date(2020, 6, 30)) is None

    def test_target_exactly_a_date_includes_it(self):
        # boundary: uses > to break, so equal is included
        assert fx._nearest_close(self.SERIES, date(2024, 1, 4)) == 2.0

    def test_target_between_dates_uses_earlier(self):
        assert fx._nearest_close(self.SERIES, date(2024, 1, 5)) == 2.0

    def test_target_after_last_returns_last(self):
        assert fx._nearest_close(self.SERIES, date(2024, 6, 1)) == 3.0

    def test_single_element_target_equals(self):
        assert fx._nearest_close([(date(2024, 1, 2), 7.0)], date(2024, 1, 2)) == 7.0

    def test_single_element_target_before(self):
        assert fx._nearest_close([(date(2024, 1, 2), 7.0)], date(2024, 1, 1)) is None

    def test_empty_series_returns_none(self):
        assert fx._nearest_close([], date(2024, 1, 5)) is None

    def test_duplicate_dates_last_match_wins(self):
        series = [
            (date(2024, 1, 2), 1.0),
            (date(2024, 1, 4), 2.0),
            (date(2024, 1, 4), 2.5),  # later duplicate
            (date(2024, 1, 6), 3.0),
        ]
        # target == 2024-01-04: equal dates don't break, so last dup (2.5) wins
        assert fx._nearest_close(series, date(2024, 1, 4)) == 2.5

    def test_first_element_exact_match(self):
        assert fx._nearest_close(self.SERIES, date(2024, 1, 2)) == 1.0


# ══════════════════════════════════════════════════════════════════════
# _frankfurter_rate — cache-then-fetch orchestration
# ══════════════════════════════════════════════════════════════════════
class TestFrankfurterRate:
    ON = date(2024, 1, 5)

    def test_cache_hit_short_circuits_fetch(self, monkeypatch):
        monkeypatch.setattr(fx, "_frankfurter_read", lambda c, o: 1.23)
        monkeypatch.setattr(fx, "_frankfurter_fetch",
                            lambda c, o: pytest.fail("fetch called on cache hit"))
        monkeypatch.setattr(fx, "_frankfurter_write",
                            lambda c, o, r: pytest.fail("write called on cache hit"))
        assert fx._frankfurter_rate("CNY", self.ON) == 1.23

    def test_cache_miss_fetch_hit_writes_and_returns(self, monkeypatch):
        written = {}
        monkeypatch.setattr(fx, "_frankfurter_read", lambda c, o: None)
        monkeypatch.setattr(fx, "_frankfurter_fetch", lambda c, o: 0.14)

        def fake_write(c, o, r):
            written["args"] = (c, o, r)
        monkeypatch.setattr(fx, "_frankfurter_write", fake_write)

        assert fx._frankfurter_rate("CNY", self.ON) == 0.14
        assert written["args"] == ("CNY", self.ON, 0.14)

    def test_cache_miss_fetch_none_does_not_write(self, monkeypatch):
        monkeypatch.setattr(fx, "_frankfurter_read", lambda c, o: None)
        monkeypatch.setattr(fx, "_frankfurter_fetch", lambda c, o: None)
        monkeypatch.setattr(fx, "_frankfurter_write",
                            lambda c, o, r: pytest.fail("write called on None fetch"))
        assert fx._frankfurter_rate("CNY", self.ON) is None

    def test_cached_zero_is_a_hit_not_a_miss(self, monkeypatch):
        # 0.0 is not None -> short circuits fetch and returns 0.0
        monkeypatch.setattr(fx, "_frankfurter_read", lambda c, o: 0.0)
        monkeypatch.setattr(fx, "_frankfurter_fetch",
                            lambda c, o: pytest.fail("fetch called when cache is 0.0"))
        assert fx._frankfurter_rate("CNY", self.ON) == 0.0


# ══════════════════════════════════════════════════════════════════════
# _frankfurter_fetch — network walk-back over urllib
# ══════════════════════════════════════════════════════════════════════
class _FakeResp:
    """Context-manager fake for urllib.request.urlopen."""
    def __init__(self, body: str):
        self._body = body.encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


class TestFrankfurterFetch:
    ON = date(2024, 1, 5)

    def test_first_day_hit_only_one_request(self, monkeypatch):
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(req.full_url)
            return _FakeResp(json.dumps({"rates": {"USD": 0.14}}))
        monkeypatch.setattr(fx.urllib.request, "urlopen", fake_urlopen)

        assert fx._frankfurter_fetch("CNY", self.ON) == pytest.approx(0.14)
        assert len(calls) == 1
        assert "2024-01-05" in calls[0]
        assert "from=CNY" in calls[0]
        assert "to=USD" in calls[0]

    def test_walks_back_until_hit_and_decrements_date(self, monkeypatch):
        # First 2 days have no USD rate, 3rd day hits.
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(req.full_url)
            if len(calls) < 3:
                return _FakeResp(json.dumps({"rates": {}}))
            return _FakeResp(json.dumps({"rates": {"USD": 0.15}}))
        monkeypatch.setattr(fx.urllib.request, "urlopen", fake_urlopen)

        assert fx._frankfurter_fetch("CNY", self.ON) == pytest.approx(0.15)
        assert len(calls) == 3
        # URL date decrements: on, on-1, on-2
        assert "2024-01-05" in calls[0]
        assert "2024-01-04" in calls[1]
        assert "2024-01-03" in calls[2]

    def test_all_attempts_empty_returns_none_and_warns(self, monkeypatch, caplog):
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(req.full_url)
            return _FakeResp(json.dumps({"rates": {}}))
        monkeypatch.setattr(fx.urllib.request, "urlopen", fake_urlopen)

        with caplog.at_level("WARNING"):
            assert fx._frankfurter_fetch("CNY", self.ON) is None
        # _FRANK_MAX_FALLBACK_DAYS=5 -> range(6) -> 6 attempts (back=0..5)
        assert len(calls) == fx._FRANK_MAX_FALLBACK_DAYS + 1 == 6
        assert any("no frankfurter rate" in r.message for r in caplog.records)
        # last attempt should be on - 5 days
        last_date = (self.ON - timedelta(days=5)).isoformat()
        assert last_date in calls[-1]

    def test_missing_rates_key_keeps_walking(self, monkeypatch):
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(req.full_url)
            if len(calls) == 1:
                return _FakeResp(json.dumps({}))  # no 'rates' at all
            return _FakeResp(json.dumps({"rates": {"USD": 0.16}}))
        monkeypatch.setattr(fx.urllib.request, "urlopen", fake_urlopen)
        assert fx._frankfurter_fetch("CNY", self.ON) == pytest.approx(0.16)
        assert len(calls) == 2

    @pytest.mark.parametrize("rate_val,expected", [
        ("0.14", 0.14),   # string coerced via float()
        (0.14, 0.14),     # native float
        (1, 1.0),         # int coerced
    ])
    def test_rate_coerced_via_float(self, rate_val, expected, monkeypatch):
        monkeypatch.setattr(
            fx.urllib.request, "urlopen",
            lambda req, timeout=None: _FakeResp(json.dumps({"rates": {"USD": rate_val}})),
        )
        result = fx._frankfurter_fetch("CNY", self.ON)
        assert result == pytest.approx(expected)
        assert isinstance(result, float)

    def test_urlerror_aborts_immediately_no_walk(self, monkeypatch, caplog):
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(req.full_url)
            raise fx.urllib.error.URLError("boom")
        monkeypatch.setattr(fx.urllib.request, "urlopen", fake_urlopen)

        with caplog.at_level("WARNING"):
            assert fx._frankfurter_fetch("CNY", self.ON) is None
        # Does NOT continue the loop — exactly one attempt.
        assert len(calls) == 1
        assert any("frankfurter fetch failed" in r.message for r in caplog.records)

    def test_timeout_error_aborts_immediately(self, monkeypatch):
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(1)
            raise TimeoutError("slow")
        monkeypatch.setattr(fx.urllib.request, "urlopen", fake_urlopen)
        assert fx._frankfurter_fetch("CNY", self.ON) is None
        assert len(calls) == 1

    def test_json_decode_error_aborts_immediately(self, monkeypatch):
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(1)
            return _FakeResp("not json at all {{{")
        monkeypatch.setattr(fx.urllib.request, "urlopen", fake_urlopen)
        assert fx._frankfurter_fetch("CNY", self.ON) is None
        assert len(calls) == 1

    def test_request_carries_user_agent_header(self, monkeypatch):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["headers"] = req.headers
            return _FakeResp(json.dumps({"rates": {"USD": 0.14}}))
        monkeypatch.setattr(fx.urllib.request, "urlopen", fake_urlopen)
        fx._frankfurter_fetch("CNY", self.ON)
        # urllib normalizes header keys to title-case
        assert captured["headers"].get("User-agent") == "dilution-tracker/1.0"

    def test_url_construction_format(self, monkeypatch):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            return _FakeResp(json.dumps({"rates": {"USD": 0.14}}))
        monkeypatch.setattr(fx.urllib.request, "urlopen", fake_urlopen)
        fx._frankfurter_fetch("ILS", date(2024, 1, 5))
        assert captured["url"] == (
            "https://api.frankfurter.dev/v1/2024-01-05?from=ILS&to=USD"
        )


# ══════════════════════════════════════════════════════════════════════
# _frankfurter_path — pure path builder
# ══════════════════════════════════════════════════════════════════════
class TestFrankfurterPath:
    def test_currency_uppercased_in_dir(self):
        p = fx._frankfurter_path("cny", date(2024, 1, 5))
        assert p.parent.name == "CNY"

    def test_filename_is_iso_date_json(self):
        p = fx._frankfurter_path("CNY", date(2024, 1, 5))
        assert p.name == "2024-01-05.json"

    def test_rooted_at_frank_dir(self):
        p = fx._frankfurter_path("CNY", date(2024, 1, 5))
        assert p.parent.parent == fx._FRANK_DIR

    def test_full_path_shape(self):
        p = fx._frankfurter_path("eur", date(2023, 12, 31))
        assert p == fx._FRANK_DIR / "EUR" / "2023-12-31.json"


# ══════════════════════════════════════════════════════════════════════
# _frankfurter_read
# ══════════════════════════════════════════════════════════════════════
class TestFrankfurterRead:
    ON = date(2024, 1, 5)

    def _write(self, content: str):
        p = fx._frankfurter_path("CNY", self.ON)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def test_missing_file_returns_none(self):
        assert fx._frankfurter_read("CNY", self.ON) is None

    def test_valid_file_returns_rate(self):
        self._write(json.dumps({"rate": 1.23, "currency": "CNY",
                                "date": "2024-01-05"}))
        assert fx._frankfurter_read("CNY", self.ON) == pytest.approx(1.23)

    def test_malformed_json_returns_none(self):
        self._write("{not json")
        assert fx._frankfurter_read("CNY", self.ON) is None

    def test_missing_rate_key_returns_none(self):
        self._write(json.dumps({"currency": "CNY"}))
        assert fx._frankfurter_read("CNY", self.ON) is None

    def test_non_numeric_rate_returns_none(self):
        self._write(json.dumps({"rate": "notanumber"}))
        assert fx._frankfurter_read("CNY", self.ON) is None

    def test_null_rate_returns_none(self):
        # FIXED (was bug A#1): a JSON `null` rate is malformed and now returns
        # None like the "notanumber" row above — TypeError was added to the
        # except guard so float(None) no longer escapes uncaught.
        self._write(json.dumps({"rate": None}))
        assert fx._frankfurter_read("CNY", self.ON) is None

    def test_rate_string_coerced_to_float(self):
        self._write(json.dumps({"rate": "1.23"}))
        result = fx._frankfurter_read("CNY", self.ON)
        assert result == pytest.approx(1.23)
        assert isinstance(result, float)

    def test_rate_zero_returns_zero_not_none(self):
        self._write(json.dumps({"rate": 0}))
        result = fx._frankfurter_read("CNY", self.ON)
        assert result == 0.0
        assert result is not None


# ══════════════════════════════════════════════════════════════════════
# _frankfurter_write
# ══════════════════════════════════════════════════════════════════════
class TestFrankfurterWrite:
    ON = date(2024, 1, 5)

    def test_creates_nested_parent_dirs(self):
        # _FRANK_DIR/CNY does not exist yet
        assert not (fx._FRANK_DIR / "CNY").exists()
        fx._frankfurter_write("CNY", self.ON, 0.14)
        assert fx._frankfurter_path("CNY", self.ON).exists()

    def test_written_json_roundtrips(self):
        fx._frankfurter_write("CNY", self.ON, 0.14)
        payload = json.loads(fx._frankfurter_path("CNY", self.ON).read_text())
        assert payload == {"rate": 0.14, "currency": "CNY", "date": "2024-01-05"}

    def test_overwrites_existing_file(self):
        fx._frankfurter_write("CNY", self.ON, 0.14)
        fx._frankfurter_write("CNY", self.ON, 0.99)
        payload = json.loads(fx._frankfurter_path("CNY", self.ON).read_text())
        assert payload["rate"] == 0.99

    def test_write_then_read_roundtrip(self):
        fx._frankfurter_write("CNY", self.ON, 0.14)
        assert fx._frankfurter_read("CNY", self.ON) == pytest.approx(0.14)


# ══════════════════════════════════════════════════════════════════════
# End-to-end-ish: to_usd through real _rate with mocked I/O seams only
# ══════════════════════════════════════════════════════════════════════
class TestIntegrationLight:
    def test_eur_through_finviz_series_to_usd(self, monkeypatch):
        import time
        # Stage a fresh finviz cache so no network is touched.
        fx._FINVIZ_DIR.mkdir(parents=True, exist_ok=True)
        (fx._FINVIZ_DIR / "EUR.json").write_text(json.dumps({
            "fetched_at": time.time(),
            "pair": "EURUSD",
            "series": [["2024-01-02", 1.05], ["2024-01-05", 1.08]],
        }))
        monkeypatch.setattr(fx, "_finviz_quote",
                            lambda p: pytest.fail("should not fetch"))
        # 100 EUR on 2024-01-05 at close 1.08 -> 108 USD
        assert fx.to_usd(100.0, "eur", date(2024, 1, 5)) == pytest.approx(108.0)

    def test_cny_through_frankfurter_cache_to_usd(self, monkeypatch):
        # Stage a frankfurter cache hit so no network is touched.
        on = date(2024, 1, 5)
        fx._frankfurter_write("CNY", on, 0.14)
        monkeypatch.setattr(fx.urllib.request, "urlopen",
                            lambda *a, **k: pytest.fail("should not hit network"))
        assert fx.to_usd(1000.0, "CNY", on) == pytest.approx(140.0)

    def test_jpy_invert_path_through_public_api(self, monkeypatch):
        # The invert (USD-base) path is only unit-tested on _finviz_rate;
        # drive it through the public to_usd over the real _rate/_finviz_rate/
        # _finviz_series stack so the 1/close inversion is exercised
        # end-to-end. USDJPY=155 -> 1,000,000 yen ~= 6451.61 USD.
        import time
        fx._FINVIZ_DIR.mkdir(parents=True, exist_ok=True)
        (fx._FINVIZ_DIR / "JPY.json").write_text(json.dumps({
            "fetched_at": time.time(),
            "pair": "USDJPY",
            "series": [["2024-01-02", 150.0], ["2024-01-05", 155.0]],
        }))
        monkeypatch.setattr(fx, "_finviz_quote",
                            lambda p: pytest.fail("should not fetch"))
        result = fx.to_usd(1_000_000.0, "jpy", date(2024, 1, 5))
        assert result == pytest.approx(1_000_000.0 / 155.0)

    def test_cny_through_frankfurter_network_walk_to_usd(self, monkeypatch):
        # CNY is not a Finviz pair: with an empty disk cache, to_usd must
        # drive the real _rate -> _frankfurter_rate -> _frankfurter_fetch
        # network seam (urlopen) and then write the cache. Asserts the
        # full non-major stack, not just an isolated fetch.
        on = date(2024, 1, 5)

        def fake_urlopen(req, timeout=None):
            return _FakeResp(json.dumps({"rates": {"USD": 0.14}}))
        monkeypatch.setattr(fx.urllib.request, "urlopen", fake_urlopen)

        assert fx.to_usd(1000.0, "CNY", on) == pytest.approx(140.0)
        # fetch result was persisted to the cache by _frankfurter_rate.
        assert fx._frankfurter_path("CNY", on).exists()
        assert fx._frankfurter_read("CNY", on) == pytest.approx(0.14)

    def test_major_finviz_miss_falls_back_to_frankfurter_none_returns_none(
            self, monkeypatch):
        # EUR IS a Finviz major, but with no series available _finviz_rate
        # returns None and _rate must fall through to Frankfurter; when that
        # also yields nothing the public to_usd returns None (not 0, not
        # an exception). Exercises the real fallback chain end-to-end.
        monkeypatch.setattr(fx, "_finviz_quote", lambda p: None)

        def empty_urlopen(req, timeout=None):
            return _FakeResp(json.dumps({"rates": {}}))
        monkeypatch.setattr(fx.urllib.request, "urlopen", empty_urlopen)

        assert fx.to_usd(100.0, "EUR", date(2024, 1, 5)) is None

    def test_frankfurter_dir_uppercased_end_to_end(self, monkeypatch):
        # Lowercase currency input must both route correctly AND land under
        # the uppercased cache directory after a network fetch.
        on = date(2024, 1, 5)
        monkeypatch.setattr(
            fx.urllib.request, "urlopen",
            lambda req, timeout=None: _FakeResp(json.dumps({"rates": {"USD": 0.74}})),
        )
        assert fx.to_usd(10.0, "sgd", on) == pytest.approx(7.4)
        assert (fx._FRANK_DIR / "SGD" / "2024-01-05.json").exists()


# ══════════════════════════════════════════════════════════════════════
# _frankfurter_fetch — explicit-None rate value keeps walking
# ══════════════════════════════════════════════════════════════════════
class TestFrankfurterFetchExtra:
    ON = date(2024, 1, 5)

    def test_explicit_null_usd_rate_keeps_walking(self, monkeypatch):
        # rates present but USD is JSON null -> .get('USD') is None -> walk on.
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(req.full_url)
            if len(calls) == 1:
                return _FakeResp(json.dumps({"rates": {"USD": None}}))
            return _FakeResp(json.dumps({"rates": {"USD": 0.17}}))
        monkeypatch.setattr(fx.urllib.request, "urlopen", fake_urlopen)
        assert fx._frankfurter_fetch("CNY", self.ON) == pytest.approx(0.17)
        assert len(calls) == 2

    def test_negative_rate_is_returned_verbatim(self, monkeypatch):
        # A negative number is not None, so it is accepted as a hit and
        # coerced via float() (no validity filtering in the fetch path).
        monkeypatch.setattr(
            fx.urllib.request, "urlopen",
            lambda req, timeout=None: _FakeResp(json.dumps({"rates": {"USD": -0.5}})),
        )
        assert fx._frankfurter_fetch("CNY", self.ON) == pytest.approx(-0.5)
