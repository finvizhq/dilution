"""Unit tests for dilution/splits.py.

Covers pure merge/tolerance logic, the io_mockable vendor fetchers
(network + lazy-yfinance seams monkeypatched), the top-level
fetch_and_persist_splits orchestration, and the DB-backed _persist /
load_splits round-trip against the autouse temp_db.

No test makes a real network / yfinance / SEC call — the seams
(`dilution.splits.requests.get`, `sys.modules['yfinance']`, the vendor
fetchers) are all monkeypatched.
"""

from __future__ import annotations

import dataclasses
import logging
import sys

import pytest

import dilution.splits as splits
from dilution.splits import (
    SplitEvent,
    SplitFetchError,
    _persist,
    _within_tolerance,
    fetch_and_persist_splits,
    fetch_finviz_splits,
    fetch_yfinance_splits,
    load_splits,
    merge_split_sources,
)

LOGGER_NAME = "dilution.splits"


# ── helpers ──────────────────────────────────────────────────────────
def ev(date_, post, pre, *, direction=None, units="common", source="finviz"):
    """Build a SplitEvent with sensible direction default."""
    if direction is None:
        direction = "forward" if post > pre else "reverse"
    return SplitEvent(
        effective_date=date_, pre=pre, post=post,
        direction=direction, units=units, source=source,
    )


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, payload, *, json_exc=None, status_exc=None):
        self._payload = payload
        self._json_exc = json_exc
        self._status_exc = status_exc

    def raise_for_status(self):
        if self._status_exc is not None:
            raise self._status_exc

    def json(self):
        if self._json_exc is not None:
            raise self._json_exc
        return self._payload


# ─── SplitEvent dataclass ─────────────────────────────────────────────
class TestSplitEvent:
    def test_frozen(self):
        e = ev("2024-03-13", 1, 100)
        # Tightened: assert the specific FrozenInstanceError rather than a
        # bare Exception (a broad `Exception` would also pass on an
        # unrelated AttributeError/TypeError, masking a regression where
        # the dataclass stopped being frozen but raised for another reason).
        with pytest.raises(dataclasses.FrozenInstanceError):
            e.pre = 5  # type: ignore[misc]
        # value is unchanged
        assert e.pre == 100

    def test_fields_roundtrip(self):
        e = SplitEvent(
            effective_date="2024-03-13", pre=100, post=1,
            direction="reverse", units="ads", source="yfinance",
        )
        assert e.effective_date == "2024-03-13"
        assert e.pre == 100
        assert e.post == 1
        assert e.direction == "reverse"
        assert e.units == "ads"
        assert e.source == "yfinance"

    def test_equality_value_based(self):
        assert ev("2024-03-13", 1, 100) == ev("2024-03-13", 1, 100)
        assert ev("2024-03-13", 1, 100) != ev("2024-03-14", 1, 100)


# ─── _within_tolerance ────────────────────────────────────────────────
class TestWithinTolerance:
    def test_identical_dates(self):
        assert _within_tolerance("2024-03-13", "2024-03-13") is True

    @pytest.mark.parametrize("a,b,expected", [
        ("2024-03-13", "2024-03-20", True),    # exactly 7 days -> <= boundary
        ("2024-03-20", "2024-03-13", True),    # order independent
        ("2024-03-13", "2024-03-21", False),   # 8 days -> outside
        ("2024-03-21", "2024-03-13", False),   # order independent (8 days)
        ("2024-03-13", "2024-03-14", True),    # 1 day
        ("2024-03-01", "2024-04-01", False),   # ~31 days
    ])
    def test_day_boundaries(self, a, b, expected):
        assert _within_tolerance(a, b) is expected

    def test_seven_day_inclusive_eight_day_exclusive(self):
        # Explicit boundary assertions: the comparison is `<= 7`.
        assert _within_tolerance("2024-01-01", "2024-01-08") is True
        assert _within_tolerance("2024-01-01", "2024-01-09") is False

    def test_malformed_one_side_falls_back_to_equality_false(self):
        # ValueError path -> returns a == b
        assert _within_tolerance("garbage", "2024-03-13") is False

    def test_malformed_both_identical_strings_true(self):
        assert _within_tolerance("garbage", "garbage") is True

    def test_malformed_both_different_false(self):
        assert _within_tolerance("garbage", "rubbish") is False

    def test_datetime_with_time_vs_date_only_within_window(self):
        # fromisoformat parses the time component; comparison is by .date()
        assert _within_tolerance("2024-03-13T00:00:00", "2024-03-15") is True
        assert _within_tolerance("2024-03-13T23:59:59", "2024-03-25") is False


# ─── merge_split_sources ──────────────────────────────────────────────
class TestMergeSplitSources:
    def test_both_empty(self):
        assert merge_split_sources([], []) == []

    def test_only_finviz_passthrough(self):
        fv = [ev("2024-03-13", 1, 100, source="finviz")]
        out = merge_split_sources(fv, [])
        assert len(out) == 1
        assert out[0].source == "finviz"
        assert out[0] == fv[0]

    def test_only_yfinance_passthrough(self):
        yf = [ev("2024-03-13", 1, 100, source="yfinance")]
        out = merge_split_sources([], yf)
        assert len(out) == 1
        assert out[0].source == "yfinance"
        assert out[0] == yf[0]

    def test_exact_match_same_ratio_dedups_to_one(self):
        fv = [ev("2024-03-12", 1, 100, source="finviz")]
        yf = [ev("2024-03-13", 1, 100, source="yfinance")]
        out = merge_split_sources(fv, yf)
        assert len(out) == 1
        m = out[0]
        assert m.source == "finviz+yfinance"
        # On agreement, yfinance's date/direction/units win.
        assert m.effective_date == "2024-03-13"
        assert m.post == 1 and m.pre == 100
        assert m.direction == "reverse"

    def test_seven_days_apart_same_ratio_still_merges(self):
        fv = [ev("2024-03-13", 4, 1, source="finviz")]
        yf = [ev("2024-03-20", 4, 1, source="yfinance")]  # exactly 7 days
        out = merge_split_sources(fv, yf)
        assert len(out) == 1
        assert out[0].source == "finviz+yfinance"
        assert out[0].effective_date == "2024-03-20"

    def test_eight_days_apart_not_merged(self):
        fv = [ev("2024-03-13", 4, 1, source="finviz")]
        yf = [ev("2024-03-21", 4, 1, source="yfinance")]  # 8 days
        out = merge_split_sources(fv, yf)
        assert len(out) == 2
        sources = {e.source for e in out}
        assert sources == {"finviz", "yfinance"}
        # ascending order
        assert [e.effective_date for e in out] == ["2024-03-13", "2024-03-21"]

    def test_ratio_disagreement_yfinance_wins_with_warning(self, caplog):
        fv = [ev("2024-12-15", 1, 100, source="finviz")]
        yf = [ev("2024-12-15", 1017, 1000, source="yfinance")]
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            out = merge_split_sources(fv, yf)
        assert len(out) == 1
        kept = out[0]
        assert kept.source == "yfinance"
        assert kept.post == 1017 and kept.pre == 1000
        assert any("disagreement" in r.message for r in caplog.records)

    def test_greedy_first_unused_match_no_double_consume(self):
        # Two finviz events both within tolerance of one yfinance event.
        # Only the first finviz claims it; the second is passed through.
        fv = [
            ev("2024-03-13", 4, 1, source="finviz"),
            ev("2024-03-14", 4, 1, source="finviz"),
        ]
        yf = [ev("2024-03-13", 4, 1, source="yfinance")]
        out = merge_split_sources(fv, yf)
        assert len(out) == 2
        srcs = sorted(e.source for e in out)
        # one merged row + one leftover finviz row
        assert srcs == ["finviz", "finviz+yfinance"]
        # The leftover finviz is the second (2024-03-14) one.
        leftover = [e for e in out if e.source == "finviz"][0]
        assert leftover.effective_date == "2024-03-14"

    def test_output_sorted_ascending_from_reverse_input(self):
        fv = [
            ev("2024-09-01", 2, 1, source="finviz"),
            ev("2024-01-01", 3, 1, source="finviz"),
            ev("2024-05-01", 4, 1, source="finviz"),
        ]
        out = merge_split_sources(fv, [])
        assert [e.effective_date for e in out] == [
            "2024-01-01", "2024-05-01", "2024-09-01",
        ]

    def test_multiple_agreeing_splits_all_merged(self):
        fv = [
            ev("2023-01-10", 1, 100, source="finviz"),
            ev("2024-06-15", 4, 1, source="finviz"),
        ]
        yf = [
            ev("2023-01-10", 1, 100, source="yfinance"),
            ev("2024-06-15", 4, 1, source="yfinance"),
        ]
        out = merge_split_sources(fv, yf)
        assert len(out) == 2
        assert all(e.source == "finviz+yfinance" for e in out)
        assert [e.effective_date for e in out] == ["2023-01-10", "2024-06-15"]

    def test_unmatched_yfinance_passes_through(self):
        fv = [ev("2024-03-13", 4, 1, source="finviz")]
        yf = [
            ev("2024-03-13", 4, 1, source="yfinance"),   # matches fv
            ev("2024-09-01", 2, 1, source="yfinance"),   # unmatched
        ]
        out = merge_split_sources(fv, yf)
        assert len(out) == 2
        merged = [e for e in out if e.source == "finviz+yfinance"]
        leftover = [e for e in out if e.source == "yfinance"]
        assert len(merged) == 1 and len(leftover) == 1
        assert leftover[0].effective_date == "2024-09-01"

    def test_two_same_date_yf_entries_not_double_consumed(self):
        # finviz has one event; yfinance has two on the same date.
        # The first yfinance is consumed by the match; the second is
        # left unmatched and passed through (greedy, no double-consume).
        fv = [ev("2024-03-13", 4, 1, source="finviz")]
        yf = [
            ev("2024-03-13", 4, 1, source="yfinance"),
            ev("2024-03-13", 2, 1, source="yfinance"),
        ]
        out = merge_split_sources(fv, yf)
        assert len(out) == 2
        assert sorted(e.source for e in out) == ["finviz+yfinance", "yfinance"]
        leftover = [e for e in out if e.source == "yfinance"][0]
        assert leftover.post == 2 and leftover.pre == 1

    def test_merge_does_not_mutate_inputs(self):
        fv = [ev("2024-03-13", 4, 1, source="finviz")]
        yf = [ev("2024-03-13", 4, 1, source="yfinance")]
        fv_copy, yf_copy = list(fv), list(yf)
        merge_split_sources(fv, yf)
        assert fv == fv_copy and yf == yf_copy

    def test_disagreement_at_seven_day_boundary_still_merges_yf_wins(self, caplog):
        # Boundary tolerance (7 days) AND ratio disagreement combined:
        # they DO bucket together (<=7), so yfinance wins and finviz is
        # dropped — exactly ONE row, not two, and a warning is logged.
        fv = [ev("2024-03-13", 1, 100, source="finviz")]
        yf = [ev("2024-03-20", 1017, 1000, source="yfinance")]  # 7 days
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            out = merge_split_sources(fv, yf)
        assert len(out) == 1
        assert out[0].source == "yfinance"   # kept as-is, NOT re-wrapped
        assert out[0].post == 1017 and out[0].pre == 1000
        assert out[0].effective_date == "2024-03-20"
        assert any("disagreement" in r.message for r in caplog.records)

    def test_no_warning_logged_when_ratios_agree(self, caplog):
        # The disagreement warning must NOT fire on an agreeing match.
        fv = [ev("2024-03-13", 4, 1, source="finviz")]
        yf = [ev("2024-03-13", 4, 1, source="yfinance")]
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            out = merge_split_sources(fv, yf)
        assert len(out) == 1
        assert not any("disagreement" in r.message for r in caplog.records)

    def test_interleaved_mixed_match_and_passthrough_sorted(self):
        # A realistic mix: one agreeing pair, one finviz-only, one
        # yfinance-only — assert final ascending order and per-row source.
        fv = [
            ev("2024-06-15", 4, 1, source="finviz"),     # agrees with yf
            ev("2023-01-10", 1, 100, source="finviz"),   # finviz only
        ]
        yf = [
            ev("2024-06-15", 4, 1, source="yfinance"),   # agrees with fv
            ev("2025-09-01", 2, 1, source="yfinance"),   # yfinance only
        ]
        out = merge_split_sources(fv, yf)
        assert [(e.effective_date, e.source) for e in out] == [
            ("2023-01-10", "finviz"),
            ("2024-06-15", "finviz+yfinance"),
            ("2025-09-01", "yfinance"),
        ]


# ─── fetch_finviz_splits ──────────────────────────────────────────────
class TestFetchFinvizSplits:
    def test_no_key_short_circuits_no_http(self, monkeypatch):
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "")
        called = {"n": 0}

        def boom(*a, **k):
            called["n"] += 1
            raise AssertionError("requests.get should not be called")

        monkeypatch.setattr(splits.requests, "get", boom)
        assert fetch_finviz_splits("XYZ") == []
        assert called["n"] == 0

    def test_request_exception_returns_empty(self, monkeypatch, caplog):
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "k")

        def raiser(*a, **k):
            raise splits.requests.RequestException("network down")

        monkeypatch.setattr(splits.requests, "get", raiser)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            assert fetch_finviz_splits("XYZ") == []
        assert any("fetch failed" in r.message for r in caplog.records)

    def test_json_value_error_returns_empty(self, monkeypatch):
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "k")
        resp = FakeResponse(None, json_exc=ValueError("bad json"))
        monkeypatch.setattr(splits.requests, "get", lambda *a, **k: resp)
        assert fetch_finviz_splits("XYZ") == []

    def test_raise_for_status_error_returns_empty(self, monkeypatch):
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "k")
        resp = FakeResponse(
            [], status_exc=splits.requests.RequestException("500"),
        )
        monkeypatch.setattr(splits.requests, "get", lambda *a, **k: resp)
        assert fetch_finviz_splits("XYZ") == []

    def test_payload_not_a_list_returns_empty(self, monkeypatch):
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "k")
        resp = FakeResponse({"error": "nope"})
        monkeypatch.setattr(splits.requests, "get", lambda *a, **k: resp)
        assert fetch_finviz_splits("XYZ") == []

    def test_case_insensitive_ticker_match(self, monkeypatch):
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "k")
        payload = [{"ticker": "xyz", "exdate": "2024-03-13",
                    "factorFrom": 4, "factorTo": 1}]
        monkeypatch.setattr(splits.requests, "get",
                            lambda *a, **k: FakeResponse(payload))
        out = fetch_finviz_splits("XYZ")
        assert len(out) == 1
        assert out[0].effective_date == "2024-03-13"

    def test_ticker_mismatch_returns_empty(self, monkeypatch):
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "k")
        payload = [{"ticker": "AAA", "exdate": "2024-03-13",
                    "factorFrom": 4, "factorTo": 1}]
        monkeypatch.setattr(splits.requests, "get",
                            lambda *a, **k: FakeResponse(payload))
        assert fetch_finviz_splits("ZZZ") == []

    def test_non_dict_entry_skipped(self, monkeypatch):
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "k")
        payload = ["not-a-dict",
                   {"ticker": "XYZ", "exdate": "2024-03-13",
                    "factorFrom": 4, "factorTo": 1}]
        monkeypatch.setattr(splits.requests, "get",
                            lambda *a, **k: FakeResponse(payload))
        out = fetch_finviz_splits("XYZ")
        assert len(out) == 1

    def test_missing_exdate_skipped(self, monkeypatch):
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "k")
        payload = [{"ticker": "XYZ", "factorFrom": 4, "factorTo": 1}]
        monkeypatch.setattr(splits.requests, "get",
                            lambda *a, **k: FakeResponse(payload))
        assert fetch_finviz_splits("XYZ") == []

    def test_float_factor_skipped(self, monkeypatch):
        # isinstance(post, int) guard rejects float factors.
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "k")
        payload = [{"ticker": "XYZ", "exdate": "2024-03-13",
                    "factorFrom": 4.0, "factorTo": 1}]
        monkeypatch.setattr(splits.requests, "get",
                            lambda *a, **k: FakeResponse(payload))
        assert fetch_finviz_splits("XYZ") == []

    def test_string_factor_skipped(self, monkeypatch):
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "k")
        payload = [{"ticker": "XYZ", "exdate": "2024-03-13",
                    "factorFrom": "4", "factorTo": "1"}]
        monkeypatch.setattr(splits.requests, "get",
                            lambda *a, **k: FakeResponse(payload))
        assert fetch_finviz_splits("XYZ") == []

    def test_bool_factor_passes_isinstance_int_guard(self, monkeypatch):
        # Documenting current behavior: bool is a subclass of int, so a
        # JSON `true`/`false` factor passes the isinstance(post, int)
        # guard. Here factorFrom=True (==1), factorTo=2 -> reverse split,
        # post(=1) != pre(=2), both >= 1, so it is ACCEPTED.
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "k")
        payload = [{"ticker": "XYZ", "exdate": "2024-03-13",
                    "factorFrom": True, "factorTo": 2}]
        monkeypatch.setattr(splits.requests, "get",
                            lambda *a, **k: FakeResponse(payload))
        out = fetch_finviz_splits("XYZ")
        # bool True is accepted by isinstance(.., int) -> entry kept.
        assert len(out) == 1
        assert out[0].post == 1  # True coerces to 1 in the SplitEvent
        assert out[0].pre == 2
        assert out[0].direction == "reverse"

    def test_noop_split_skipped(self, monkeypatch):
        # post == pre (1:1) -> skipped by `post != pre` guard.
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "k")
        payload = [{"ticker": "XYZ", "exdate": "2024-03-13",
                    "factorFrom": 1, "factorTo": 1}]
        monkeypatch.setattr(splits.requests, "get",
                            lambda *a, **k: FakeResponse(payload))
        assert fetch_finviz_splits("XYZ") == []

    def test_factor_below_one_skipped(self, monkeypatch):
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "k")
        payload = [{"ticker": "XYZ", "exdate": "2024-03-13",
                    "factorFrom": 0, "factorTo": 1}]
        monkeypatch.setattr(splits.requests, "get",
                            lambda *a, **k: FakeResponse(payload))
        assert fetch_finviz_splits("XYZ") == []

    def test_pre_below_one_skipped(self, monkeypatch):
        # The guard is `post >= 1 and pre >= 1` — exercise the PRE side
        # too (factorTo=0), distinct from the factorFrom=0 case above.
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "k")
        payload = [{"ticker": "XYZ", "exdate": "2024-03-13",
                    "factorFrom": 4, "factorTo": 0}]
        monkeypatch.setattr(splits.requests, "get",
                            lambda *a, **k: FakeResponse(payload))
        assert fetch_finviz_splits("XYZ") == []

    @pytest.mark.parametrize("post,pre", [(-1, 1), (1, -5)])
    def test_negative_factor_skipped(self, monkeypatch, post, pre):
        # Negative ints fail `post >= 1`/`pre >= 1`.
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "k")
        payload = [{"ticker": "XYZ", "exdate": "2024-03-13",
                    "factorFrom": post, "factorTo": pre}]
        monkeypatch.setattr(splits.requests, "get",
                            lambda *a, **k: FakeResponse(payload))
        assert fetch_finviz_splits("XYZ") == []

    def test_multiple_valid_entries_for_ticker_all_returned(self, monkeypatch):
        # The dump can contain >1 split for one ticker; all valid ones are
        # emitted (output order follows the input dump order, since
        # fetch_finviz_splits does not sort — only merge does).
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "k")
        payload = [
            {"ticker": "XYZ", "exdate": "2024-03-13",
             "factorFrom": 4, "factorTo": 1},
            {"ticker": "AAA", "exdate": "2024-03-13",   # other ticker
             "factorFrom": 2, "factorTo": 1},
            {"ticker": "XYZ", "exdate": "2020-01-01",
             "factorFrom": 1, "factorTo": 10},
        ]
        monkeypatch.setattr(splits.requests, "get",
                            lambda *a, **k: FakeResponse(payload))
        out = fetch_finviz_splits("XYZ")
        assert len(out) == 2
        # input order preserved (no sort inside the fetcher)
        assert [e.effective_date for e in out] == ["2024-03-13", "2020-01-01"]
        assert {e.direction for e in out} == {"forward", "reverse"}

    def test_forward_split_mapping(self, monkeypatch):
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "k")
        payload = [{"ticker": "XYZ", "exdate": "2024-03-13",
                    "factorFrom": 4, "factorTo": 1}]
        monkeypatch.setattr(splits.requests, "get",
                            lambda *a, **k: FakeResponse(payload))
        out = fetch_finviz_splits("XYZ")
        assert len(out) == 1
        e = out[0]
        assert e.post == 4 and e.pre == 1
        assert e.direction == "forward"
        assert e.units == "common"
        assert e.source == "finviz"

    def test_reverse_split_mapping(self, monkeypatch):
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "k")
        payload = [{"ticker": "XYZ", "exdate": "2024-03-13",
                    "factorFrom": 1, "factorTo": 100}]
        monkeypatch.setattr(splits.requests, "get",
                            lambda *a, **k: FakeResponse(payload))
        out = fetch_finviz_splits("XYZ")
        assert len(out) == 1
        e = out[0]
        assert e.post == 1 and e.pre == 100
        assert e.direction == "reverse"

    def test_http_params_headers_and_url(self, monkeypatch):
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "secret-key")
        monkeypatch.setattr(splits.config, "FINVIZ_BASE_URL",
                            "https://example.com/")  # trailing slash
        captured = {}

        def capture(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return FakeResponse([])

        monkeypatch.setattr(splits.requests, "get", capture)
        fetch_finviz_splits("XYZ")
        # trailing slash stripped, endpoint appended
        assert captured["url"] == "https://example.com/export/split-history"
        assert captured["params"] == {"auth": "secret-key"}
        assert captured["timeout"] == splits._HTTP_TIMEOUT
        assert captured["headers"] == {"User-Agent": "FinvizApps/Dilution"}


# ─── fetch_yfinance_splits ────────────────────────────────────────────
class FakeSeries:
    """Stand-in for a pandas Series exposing .items() and len()."""

    def __init__(self, pairs):
        self._pairs = list(pairs)

    def __len__(self):
        return len(self._pairs)

    def items(self):
        return iter(self._pairs)


class FakeTimestamp:
    """Timestamp with a strftime."""

    def __init__(self, s):
        self._s = s

    def strftime(self, fmt):
        # The code always passes '%Y-%m-%d'; just return our canned str.
        return self._s


class FakeTicker:
    def __init__(self, series_or_exc):
        self._sx = series_or_exc

    @property
    def splits(self):
        if isinstance(self._sx, Exception):
            raise self._sx
        return self._sx


def _install_fake_yfinance(monkeypatch, ticker_factory):
    """Inject a fake yfinance module whose Ticker(sym) is built by
    ticker_factory(sym)."""
    import types

    fake = types.ModuleType("yfinance")
    fake.Ticker = ticker_factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", fake)


class TestFetchYfinanceSplits:
    def test_import_error_returns_empty_with_warning(self, monkeypatch, caplog):
        # setting sys.modules['yfinance'] = None makes `import yfinance`
        # raise ImportError.
        monkeypatch.setitem(sys.modules, "yfinance", None)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            assert fetch_yfinance_splits("XYZ") == []
        assert any("not installed" in r.message for r in caplog.records)

    def test_splits_property_raises_returns_empty_with_warning(
        self, monkeypatch, caplog,
    ):
        _install_fake_yfinance(
            monkeypatch, lambda sym: FakeTicker(RuntimeError("boom")),
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            assert fetch_yfinance_splits("XYZ") == []
        assert any("fetch failed" in r.message for r in caplog.records)

    def test_series_none_returns_empty(self, monkeypatch):
        _install_fake_yfinance(monkeypatch, lambda sym: FakeTicker(None))
        assert fetch_yfinance_splits("XYZ") == []

    def test_empty_series_returns_empty(self, monkeypatch):
        _install_fake_yfinance(
            monkeypatch, lambda sym: FakeTicker(FakeSeries([])),
        )
        assert fetch_yfinance_splits("XYZ") == []

    def test_non_coercible_ratio_skipped(self, monkeypatch):
        series = FakeSeries([
            (FakeTimestamp("2024-03-13"), object()),       # float() raises TypeError
            (FakeTimestamp("2024-06-15"), 4.0),            # valid
        ])
        _install_fake_yfinance(
            monkeypatch, lambda sym: FakeTicker(series),
        )
        out = fetch_yfinance_splits("XYZ")
        assert len(out) == 1
        assert out[0].effective_date == "2024-06-15"

    @pytest.mark.parametrize("ratio", [0.0, -1.0, 1.0])
    def test_zero_negative_or_unit_ratio_skipped(self, monkeypatch, ratio):
        series = FakeSeries([(FakeTimestamp("2024-03-13"), ratio)])
        _install_fake_yfinance(
            monkeypatch, lambda sym: FakeTicker(series),
        )
        assert fetch_yfinance_splits("XYZ") == []

    def test_forward_4_to_1(self, monkeypatch):
        series = FakeSeries([(FakeTimestamp("2024-03-13"), 4.0)])
        _install_fake_yfinance(
            monkeypatch, lambda sym: FakeTicker(series),
        )
        out = fetch_yfinance_splits("XYZ")
        assert len(out) == 1
        e = out[0]
        assert e.post == 4 and e.pre == 1
        assert e.direction == "forward"
        assert e.source == "yfinance"
        assert e.units == "common"

    def test_reverse_1_to_100(self, monkeypatch):
        series = FakeSeries([(FakeTimestamp("2024-03-13"), 0.01)])
        _install_fake_yfinance(
            monkeypatch, lambda sym: FakeTicker(series),
        )
        out = fetch_yfinance_splits("XYZ")
        assert len(out) == 1
        e = out[0]
        assert e.post == 1 and e.pre == 100
        assert e.direction == "reverse"

    def test_noisy_float_1017_1000(self, monkeypatch):
        series = FakeSeries([(FakeTimestamp("2025-12-15"), 1.017)])
        _install_fake_yfinance(
            monkeypatch, lambda sym: FakeTicker(series),
        )
        out = fetch_yfinance_splits("XYZ")
        assert len(out) == 1
        e = out[0]
        assert e.post == 1017 and e.pre == 1000
        assert e.direction == "forward"

    def test_timestamp_without_strftime_falls_back_to_str(self, monkeypatch):
        # A plain string has no .strftime -> AttributeError -> str(ts)[:10]
        series = FakeSeries([("2024-03-13T00:00:00", 4.0)])
        _install_fake_yfinance(
            monkeypatch, lambda sym: FakeTicker(series),
        )
        out = fetch_yfinance_splits("XYZ")
        assert len(out) == 1
        assert out[0].effective_date == "2024-03-13"

    def test_near_unit_ratio_reduces_to_post_eq_pre_skipped(self, monkeypatch):
        # SURVEY edge missed by the original suite: a ratio that is NOT
        # exactly 1.0 (so it survives the `ratio == 1.0` early skip) but
        # whose Fraction(ratio).limit_denominator(10000) collapses to 1/1
        # -> post == pre -> skipped by the second guard. Observed: [].
        series = FakeSeries([(FakeTimestamp("2024-03-13"), 1.00001)])
        _install_fake_yfinance(
            monkeypatch, lambda sym: FakeTicker(series),
        )
        assert fetch_yfinance_splits("XYZ") == []

    def test_multiple_ratios_mixed_valid_and_skipped(self, monkeypatch):
        # Two valid splits plus one unit ratio interleaved; only the two
        # valid ones are emitted, in series order.
        series = FakeSeries([
            (FakeTimestamp("2023-01-10"), 0.01),   # reverse 1:100
            (FakeTimestamp("2023-06-01"), 1.0),    # skipped (==1.0)
            (FakeTimestamp("2024-06-15"), 4.0),    # forward 4:1
        ])
        _install_fake_yfinance(
            monkeypatch, lambda sym: FakeTicker(series),
        )
        out = fetch_yfinance_splits("XYZ")
        assert [e.effective_date for e in out] == ["2023-01-10", "2024-06-15"]
        assert [(e.post, e.pre) for e in out] == [(1, 100), (4, 1)]

    def test_ticker_uppercased_in_lookup(self, monkeypatch):
        # yf.Ticker is called with ticker.upper(); capture the symbol.
        seen = {}

        def factory(sym):
            seen["sym"] = sym
            return FakeTicker(FakeSeries([(FakeTimestamp("2024-03-13"), 4.0)]))

        _install_fake_yfinance(monkeypatch, factory)
        fetch_yfinance_splits("xyz")
        assert seen["sym"] == "XYZ"


# ─── fetch_and_persist_splits ─────────────────────────────────────────
class TestFetchAndPersistSplits:
    def test_both_empty_no_raise_persist_called(self, monkeypatch, temp_db):
        monkeypatch.setattr(splits, "fetch_finviz_splits", lambda t: [])
        monkeypatch.setattr(splits, "fetch_yfinance_splits", lambda t: [])
        captured = {}
        monkeypatch.setattr(
            splits, "_persist",
            lambda cik, events: captured.update(cik=cik, events=list(events)),
        )
        out = fetch_and_persist_splits(123, "XYZ")
        assert out == []
        # _persist still called (clears the CIK's rows)
        assert captured == {"cik": 123, "events": []}

    def test_finviz_raises_yfinance_succeeds(self, monkeypatch):
        def boom(t):
            raise RuntimeError("finviz down")

        yf_events = [ev("2024-03-13", 4, 1, source="yfinance")]
        monkeypatch.setattr(splits, "fetch_finviz_splits", boom)
        monkeypatch.setattr(splits, "fetch_yfinance_splits", lambda t: yf_events)
        monkeypatch.setattr(splits, "_persist", lambda cik, events: None)
        out = fetch_and_persist_splits(1, "XYZ")
        assert len(out) == 1
        assert out[0].source == "yfinance"

    def test_yfinance_raises_finviz_succeeds(self, monkeypatch):
        def boom(t):
            raise RuntimeError("yf down")

        fv_events = [ev("2024-03-13", 4, 1, source="finviz")]
        monkeypatch.setattr(splits, "fetch_finviz_splits", lambda t: fv_events)
        monkeypatch.setattr(splits, "fetch_yfinance_splits", boom)
        monkeypatch.setattr(splits, "_persist", lambda cik, events: None)
        out = fetch_and_persist_splits(1, "XYZ")
        assert len(out) == 1
        assert out[0].source == "finviz"

    def test_both_raise_split_fetch_error_persist_not_reached(self, monkeypatch):
        def boom(t):
            raise RuntimeError("down")

        monkeypatch.setattr(splits, "fetch_finviz_splits", boom)
        monkeypatch.setattr(splits, "fetch_yfinance_splits", boom)

        def persist_guard(cik, events):
            raise AssertionError("_persist should not be reached")

        monkeypatch.setattr(splits, "_persist", persist_guard)
        with pytest.raises(SplitFetchError) as exc:
            fetch_and_persist_splits(987, "ZZZ")
        msg = str(exc.value)
        assert "ZZZ" in msg
        assert "987" in msg

    def test_is_fpi_rewrites_units_to_ads(self, monkeypatch):
        fv_events = [ev("2024-03-13", 1, 100, source="finviz")]
        monkeypatch.setattr(splits, "fetch_finviz_splits", lambda t: fv_events)
        monkeypatch.setattr(splits, "fetch_yfinance_splits", lambda t: [])
        captured = {}
        monkeypatch.setattr(
            splits, "_persist",
            lambda cik, events: captured.update(events=list(events)),
        )
        out = fetch_and_persist_splits(1, "XYZ", is_fpi=True)
        assert len(out) == 1
        e = out[0]
        assert e.units == "ads"
        # other fields preserved
        assert e.post == 1 and e.pre == 100
        assert e.direction == "reverse"
        assert e.source == "finviz"
        # persisted rows carry the ads unit too
        assert captured["events"][0].units == "ads"

    def test_is_fpi_false_keeps_common(self, monkeypatch):
        fv_events = [ev("2024-03-13", 1, 100, source="finviz")]
        monkeypatch.setattr(splits, "fetch_finviz_splits", lambda t: fv_events)
        monkeypatch.setattr(splits, "fetch_yfinance_splits", lambda t: [])
        monkeypatch.setattr(splits, "_persist", lambda cik, events: None)
        out = fetch_and_persist_splits(1, "XYZ")
        assert out[0].units == "common"

    def test_persist_receives_merged_events_via_db(self, monkeypatch, temp_db):
        # Let _persist actually run against temp_db; assert DB rows.
        fv_events = [ev("2024-03-13", 4, 1, source="finviz")]
        yf_events = [ev("2024-03-13", 4, 1, source="yfinance")]
        monkeypatch.setattr(splits, "fetch_finviz_splits", lambda t: fv_events)
        monkeypatch.setattr(splits, "fetch_yfinance_splits", lambda t: yf_events)
        out = fetch_and_persist_splits(555, "XYZ")
        assert len(out) == 1
        assert out[0].source == "finviz+yfinance"
        rows = temp_db.execute(
            "SELECT cik, effective_date, source FROM dilution_splits "
            "WHERE cik = ?", (555,),
        )
        assert len(rows) == 1
        assert rows[0]["source"] == "finviz+yfinance"

    def test_vendor_returning_empty_without_raising_marks_ok(self, monkeypatch):
        # Both legitimately return [] -> no raise (proves _ok set even on []).
        monkeypatch.setattr(splits, "fetch_finviz_splits", lambda t: [])
        monkeypatch.setattr(splits, "fetch_yfinance_splits", lambda t: [])
        monkeypatch.setattr(splits, "_persist", lambda cik, events: None)
        # No SplitFetchError expected.
        assert fetch_and_persist_splits(1, "XYZ") == []

    def test_is_fpi_persists_ads_units_to_db_end_to_end(self, monkeypatch, temp_db):
        # End-to-end: let the REAL _persist run against temp_db and assert
        # the ads-rewrite actually lands in the table (the unit-test above
        # only checked the in-memory _persist capture).
        fv_events = [ev("2024-03-13", 1, 100, source="finviz")]
        monkeypatch.setattr(splits, "fetch_finviz_splits", lambda t: fv_events)
        monkeypatch.setattr(splits, "fetch_yfinance_splits", lambda t: [])
        fetch_and_persist_splits(321, "XYZ", is_fpi=True)
        rows = temp_db.execute(
            "SELECT units, direction, source FROM dilution_splits WHERE cik = ?",
            (321,),
        )
        assert len(rows) == 1
        assert rows[0]["units"] == "ads"
        assert rows[0]["direction"] == "reverse"
        assert rows[0]["source"] == "finviz"

    def test_split_fetch_error_is_runtimeerror_subclass(self):
        # Documented contract: SplitFetchError extends RuntimeError so a
        # generic `except RuntimeError` at the call site catches it.
        assert issubclass(SplitFetchError, RuntimeError)


# ─── _persist (db_backed) ─────────────────────────────────────────────
class TestPersist:
    def test_empty_events_clears_cik_rows(self, temp_db):
        temp_db.add_split(7, "2024-03-13", pre=100, post=1)
        temp_db.add_split(7, "2023-01-01", pre=1, post=4)
        _persist(7, [])
        rows = temp_db.execute(
            "SELECT * FROM dilution_splits WHERE cik = ?", (7,))
        assert rows == []

    def test_replace_semantics(self, temp_db):
        # Stage an old row, persist a different set, assert full replacement.
        temp_db.add_split(7, "2020-01-01", pre=1, post=2, source="finviz")
        _persist(7, [ev("2024-03-13", 1, 100, source="finviz+yfinance")])
        rows = temp_db.execute(
            "SELECT effective_date, source FROM dilution_splits WHERE cik = ? "
            "ORDER BY effective_date", (7,))
        assert len(rows) == 1
        assert rows[0]["effective_date"] == "2024-03-13"
        assert rows[0]["source"] == "finviz+yfinance"

    def test_other_ciks_untouched(self, temp_db):
        temp_db.add_split(8, "2022-05-05", pre=1, post=3, source="yfinance")
        _persist(7, [ev("2024-03-13", 4, 1, source="finviz")])
        rows8 = temp_db.execute(
            "SELECT * FROM dilution_splits WHERE cik = ?", (8,))
        assert len(rows8) == 1  # survivor
        rows7 = temp_db.execute(
            "SELECT * FROM dilution_splits WHERE cik = ?", (7,))
        assert len(rows7) == 1

    def test_all_columns_persisted(self, temp_db):
        e = SplitEvent(
            effective_date="2024-03-13", pre=100, post=1,
            direction="reverse", units="ads", source="finviz+yfinance",
        )
        _persist(42, [e])
        rows = temp_db.execute(
            "SELECT cik, effective_date, pre, post, direction, units, "
            "source, fetched_at FROM dilution_splits WHERE cik = ?", (42,))
        assert len(rows) == 1
        r = rows[0]
        assert r["cik"] == 42
        assert r["effective_date"] == "2024-03-13"
        assert r["pre"] == 100
        assert r["post"] == 1
        assert r["direction"] == "reverse"
        assert r["units"] == "ads"
        assert r["source"] == "finviz+yfinance"
        assert r["fetched_at"]  # non-null from now_iso()

    def test_duplicate_dates_violate_primary_key(self, temp_db):
        import sqlite3
        dup = [
            ev("2024-03-13", 4, 1, source="finviz"),
            ev("2024-03-13", 1, 100, source="yfinance"),
        ]
        with pytest.raises(sqlite3.IntegrityError):
            _persist(7, dup)


# ─── load_splits (db_backed) ──────────────────────────────────────────
class TestLoadSplits:
    def test_no_rows_returns_empty(self, temp_db):
        assert load_splits(999) == []

    def test_since_date_none_returns_all(self, temp_db):
        temp_db.add_split(7, "2023-01-01", pre=1, post=4)
        temp_db.add_split(7, "2024-06-15", pre=100, post=1)
        out = load_splits(7)
        assert len(out) == 2

    def test_since_date_boundary_inclusive(self, temp_db):
        temp_db.add_split(7, "2023-01-01", pre=1, post=4)
        temp_db.add_split(7, "2024-06-15", pre=100, post=1)
        temp_db.add_split(7, "2024-09-01", pre=1, post=2)
        out = load_splits(7, since_date="2024-06-15")
        dates = [e.effective_date for e in out]
        # row exactly equal to since_date is included (>=); earlier excluded
        assert dates == ["2024-06-15", "2024-09-01"]

    def test_ordering_ascending(self, temp_db):
        temp_db.add_split(7, "2024-09-01", pre=1, post=2)
        temp_db.add_split(7, "2023-01-01", pre=1, post=4)
        temp_db.add_split(7, "2024-06-15", pre=100, post=1)
        out = load_splits(7)
        dates = [e.effective_date for e in out]
        assert dates == ["2023-01-01", "2024-06-15", "2024-09-01"]

    def test_cik_isolation(self, temp_db):
        temp_db.add_split(7, "2024-03-13", pre=1, post=4)
        temp_db.add_split(8, "2024-03-13", pre=100, post=1)
        out7 = load_splits(7)
        assert len(out7) == 1
        assert out7[0].post == 4 and out7[0].pre == 1

    def test_splitevent_reconstructed_six_fields(self, temp_db):
        temp_db.add_split(
            7, "2024-03-13", pre=100, post=1, direction="reverse",
            units="ads", source="finviz+yfinance",
        )
        out = load_splits(7)
        assert len(out) == 1
        e = out[0]
        assert e.effective_date == "2024-03-13"
        assert e.pre == 100
        assert e.post == 1
        assert e.direction == "reverse"
        assert e.units == "ads"
        assert e.source == "finviz+yfinance"
        # fetched_at is NOT a SplitEvent field
        assert not hasattr(e, "fetched_at")

    def test_persist_then_load_roundtrip(self, temp_db):
        events = [
            ev("2023-01-10", 1, 100, source="finviz+yfinance"),
            ev("2024-06-15", 4, 1, source="finviz+yfinance"),
        ]
        _persist(7, events)
        out = load_splits(7)
        assert [e.effective_date for e in out] == ["2023-01-10", "2024-06-15"]
        assert all(e.source == "finviz+yfinance" for e in out)

    def test_since_date_excludes_all_rows_returns_empty(self, temp_db):
        # since_date strictly after every persisted row -> [] (the >= filter
        # excludes everything). Distinct from the inclusive-boundary test:
        # this proves the empty-result path, not just trimming.
        temp_db.add_split(7, "2023-01-01", pre=1, post=4)
        temp_db.add_split(7, "2024-06-15", pre=100, post=1)
        assert load_splits(7, since_date="2025-01-01") == []

    def test_since_date_excludes_strictly_earlier_only(self, temp_db):
        # since_date one day AFTER the earlier row's date -> earlier row
        # dropped, later row kept (lexicographic ISO comparison).
        temp_db.add_split(7, "2023-01-01", pre=1, post=4)
        temp_db.add_split(7, "2024-06-15", pre=100, post=1)
        out = load_splits(7, since_date="2023-01-02")
        assert [e.effective_date for e in out] == ["2024-06-15"]


# ─── Cross-function integration & survey-gap coverage ─────────────────
class TestFinvizEntryShape:
    def test_entry_missing_ticker_key_entirely_skipped(self, monkeypatch):
        # SURVEY GAP: the filter is `(entry.get("ticker") or "").upper()`,
        # so an entry with NO ticker key at all yields "" and never matches
        # the requested ticker — distinct from a present-but-different
        # ticker. Observed: [].
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "k")
        payload = [{"exdate": "2024-03-13", "factorFrom": 4, "factorTo": 1}]
        monkeypatch.setattr(splits.requests, "get",
                            lambda *a, **k: FakeResponse(payload))
        assert fetch_finviz_splits("XYZ") == []

    def test_ticker_key_none_skipped(self, monkeypatch):
        # `entry.get("ticker")` is None -> `None or ""` -> "" -> no match.
        monkeypatch.setattr(splits.config, "FINVIZ_API_KEY", "k")
        payload = [{"ticker": None, "exdate": "2024-03-13",
                    "factorFrom": 4, "factorTo": 1}]
        monkeypatch.setattr(splits.requests, "get",
                            lambda *a, **k: FakeResponse(payload))
        assert fetch_finviz_splits("XYZ") == []


class TestYfinanceTimestampFallback:
    def test_str_fallback_truncates_to_first_ten_chars(self, monkeypatch):
        # SURVEY GAP: ts with no .strftime -> AttributeError -> str(ts)[:10].
        # Prove the [:10] slice precisely: a long ISO string with sub-second
        # precision must be cut to exactly the YYYY-MM-DD prefix.
        series = FakeSeries([("2024-03-13T09:30:00.123456+00:00", 4.0)])
        _install_fake_yfinance(
            monkeypatch, lambda sym: FakeTicker(series),
        )
        out = fetch_yfinance_splits("XYZ")
        assert len(out) == 1
        assert out[0].effective_date == "2024-03-13"  # exactly 10 chars

    def test_str_fallback_short_string_not_padded(self, monkeypatch):
        # A non-strftime object whose str() is SHORTER than 10 chars: [:10]
        # is a no-op (no padding/truncation). Observed: the raw string.
        series = FakeSeries([("2024", 4.0)])
        _install_fake_yfinance(
            monkeypatch, lambda sym: FakeTicker(series),
        )
        out = fetch_yfinance_splits("XYZ")
        assert len(out) == 1
        assert out[0].effective_date == "2024"


class TestFetchPersistLoadIntegration:
    def test_full_roundtrip_merged_through_real_persist_and_load(
        self, monkeypatch, temp_db,
    ):
        # End-to-end: vendor fetchers mocked, but merge -> real _persist ->
        # real load_splits all run. Proves the persisted rows reload as the
        # same canonical merged sequence the function returned.
        fv = [
            ev("2023-01-10", 1, 100, source="finviz"),
            ev("2024-06-15", 4, 1, source="finviz"),
        ]
        yf = [
            ev("2023-01-10", 1, 100, source="yfinance"),   # agrees -> merged
            ev("2025-09-01", 2, 1, source="yfinance"),      # yf-only
        ]
        monkeypatch.setattr(splits, "fetch_finviz_splits", lambda t: fv)
        monkeypatch.setattr(splits, "fetch_yfinance_splits", lambda t: yf)
        returned = fetch_and_persist_splits(2024, "XYZ")
        reloaded = load_splits(2024)
        # what was returned == what reloads (value equality of SplitEvents)
        assert returned == reloaded
        assert [(e.effective_date, e.source) for e in reloaded] == [
            ("2023-01-10", "finviz+yfinance"),
            ("2024-06-15", "finviz"),
            ("2025-09-01", "yfinance"),
        ]

    def test_disagreement_winner_persists_yfinance_end_to_end(
        self, monkeypatch, temp_db, caplog,
    ):
        # Disagreement within window: yfinance wins in merge AND that is what
        # actually lands in the DB (real _persist), with a warning logged.
        fv = [ev("2025-12-15", 1, 100, source="finviz")]
        yf = [ev("2025-12-15", 1017, 1000, source="yfinance")]
        monkeypatch.setattr(splits, "fetch_finviz_splits", lambda t: fv)
        monkeypatch.setattr(splits, "fetch_yfinance_splits", lambda t: yf)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            fetch_and_persist_splits(4242, "IQST")
        rows = temp_db.execute(
            "SELECT post, pre, source FROM dilution_splits WHERE cik = ?",
            (4242,),
        )
        assert len(rows) == 1
        assert rows[0]["post"] == 1017 and rows[0]["pre"] == 1000
        assert rows[0]["source"] == "yfinance"
        assert any("disagreement" in r.message for r in caplog.records)

    def test_replace_semantics_across_two_persist_calls_end_to_end(
        self, monkeypatch, temp_db,
    ):
        # A second fetch_and_persist for the same cik REPLACES the prior
        # rows (vendor data is authoritative). Stage via a first call, then
        # a second call with a different (smaller) set; assert no orphans.
        monkeypatch.setattr(
            splits, "fetch_finviz_splits",
            lambda t: [ev("2023-01-10", 1, 100, source="finviz"),
                       ev("2024-06-15", 4, 1, source="finviz")])
        monkeypatch.setattr(splits, "fetch_yfinance_splits", lambda t: [])
        fetch_and_persist_splits(900, "XYZ")
        assert len(load_splits(900)) == 2
        # second walk: only one split survives in the vendor dump now
        monkeypatch.setattr(
            splits, "fetch_finviz_splits",
            lambda t: [ev("2024-06-15", 4, 1, source="finviz")])
        fetch_and_persist_splits(900, "XYZ")
        out = load_splits(900)
        assert [e.effective_date for e in out] == ["2024-06-15"]


class TestMergeGreedyOrderingStress:
    def test_unsorted_inputs_sorted_before_greedy_matching(self):
        # merge sorts BOTH inputs by date before the greedy walk. Feed both
        # vendors out of order with one agreeing pair and one finviz-only
        # earlier date; assert the earlier finviz-only is emitted first and
        # the pair merges (proves the internal sort governs match order, not
        # input order).
        fv = [
            ev("2024-06-15", 4, 1, source="finviz"),     # agrees w/ yf
            ev("2022-02-02", 1, 10, source="finviz"),    # earliest, fv-only
        ]
        yf = [ev("2024-06-15", 4, 1, source="yfinance")]
        out = merge_split_sources(fv, yf)
        assert [(e.effective_date, e.source) for e in out] == [
            ("2022-02-02", "finviz"),
            ("2024-06-15", "finviz+yfinance"),
        ]

    def test_three_finviz_one_yf_only_earliest_matches(self):
        # Three finviz events all within 7 days of a single yf event; the
        # date-sorted FIRST one claims the match, the other two pass through
        # unmatched (used_yf prevents re-consumption).
        fv = [
            ev("2024-03-16", 4, 1, source="finviz"),
            ev("2024-03-13", 4, 1, source="finviz"),   # earliest -> matches
            ev("2024-03-19", 4, 1, source="finviz"),
        ]
        yf = [ev("2024-03-14", 4, 1, source="yfinance")]
        out = merge_split_sources(fv, yf)
        srcs = [(e.effective_date, e.source) for e in out]
        # exactly one merged row; it carries yf's date (2024-03-14)
        merged = [s for s in srcs if s[1] == "finviz+yfinance"]
        leftover = sorted(s[0] for s in srcs if s[1] == "finviz")
        assert merged == [("2024-03-14", "finviz+yfinance")]
        assert leftover == ["2024-03-16", "2024-03-19"]
        # final output ascending overall
        assert [s[0] for s in srcs] == sorted(s[0] for s in srcs)
