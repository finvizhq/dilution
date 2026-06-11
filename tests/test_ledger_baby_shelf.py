"""Unit tests for dilution/ledger/baby_shelf.py.

Covers the baby-shelf / Form S-3 General Instruction I.B.6 math:
  - pure threshold / max-raise arithmetic
  - the share-basis chooser (single vs multi-class)
  - the durable upward-crossing exit override (margin + persistence)
  - eligibility derivation, trailing-12mo raise sum, unsold-ATM math
  - the three-tier baby-shelf classifier
  - the assembled 'Current Raisable Amount' dict

All cross-module dependencies are LAZY imports inside the function
bodies, so each is patched on its SOURCE module. The DB seam is the
autouse ``temp_db`` fixture from conftest.py.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

import dilution.ledger.baby_shelf as bs
from dilution.ledger.baby_shelf import (
    BABY_SHELF_EXIT_MARGIN_MULT,
    BABY_SHELF_FLOAT_VALUE_THRESHOLD_USD,
    baby_shelf_threshold_price,
    has_eligible_shelf,
    ib6_basis_shares,
    ib6_max_raise,
    ib6_remaining,
    is_baby_shelf_restricted,
    raised_under_ib6_last_12mo,
    _durably_exited_baby_shelf,
    _unsold_live_atm_usd,
)


THRESHOLD = 75_000_000  # the $75M float-value line


@pytest.fixture(autouse=True)
def _clear_exit_cache():
    """The module-level _BABY_EXIT_CLOSES_CACHE persists across calls AND
    tests, and caches negative results ([]) too. Clear before & after each
    test for determinism."""
    bs._BABY_EXIT_CLOSES_CACHE.clear()
    yield
    bs._BABY_EXIT_CLOSES_CACHE.clear()


# ── helper stubs ──────────────────────────────────────────────────────────


class _FakeClient:
    """Stub for dilution.finviz_client._client(); records invocation."""

    def __init__(self, closes, raise_exc=False):
        self._closes = closes
        self._raise = raise_exc
        self.called = False

    def get_daily_closes(self, ticker, bars=None, within_calendar_days=None):
        self.called = True
        if self._raise:
            raise RuntimeError("price feed down")
        return self._closes


class _FakeImplied:
    def __init__(self, n_classes):
        self.classes = tuple(range(n_classes))


def _patch_client(monkeypatch, client):
    """Patch the finviz _client factory at its source module."""
    import dilution.finviz_client as fc
    monkeypatch.setattr(fc, "_client", lambda: client, raising=True)


def _patch_shelf_status(monkeypatch, statuses):
    import dilution.ledger.shelf_status as ss
    monkeypatch.setattr(
        ss, "derive_shelf_status",
        lambda cik, today=None: statuses, raising=True)


def _patch_shelf_status_spy(monkeypatch, statuses, spy):
    import dilution.ledger.shelf_status as ss

    def _fake(cik, today=None):
        spy["cik"] = cik
        spy["today"] = today
        return statuses

    monkeypatch.setattr(ss, "derive_shelf_status", _fake, raising=True)


def _patch_implied(monkeypatch, n_classes=None, raise_exc=False):
    import dilution.share_counts as sc

    def _fake(cik):
        if raise_exc:
            raise RuntimeError("share-counts unavailable")
        return _FakeImplied(n_classes)

    monkeypatch.setattr(
        sc, "fetch_implied_outstanding_cached", _fake, raising=True)


def _patch_regime(monkeypatch, regime, raise_exc=False):
    import dilution.ib6_cover as ic

    def _fake(cik):
        if raise_exc:
            raise RuntimeError("regime scan blew up")
        return {"regime": regime}

    monkeypatch.setattr(ic, "ib6_regime", _fake, raising=True)


# ════════════════════════════════════════════════════════════════════════
# baby_shelf_threshold_price  (pure)
# ════════════════════════════════════════════════════════════════════════
class TestBabyShelfThresholdPrice:
    @pytest.mark.parametrize("falsy", [None, 0, 0.0])
    def test_none_or_zero_returns_none(self, falsy):
        assert baby_shelf_threshold_price(falsy) is None

    @pytest.mark.parametrize("neg", [-1, -75_000_000, -0.5])
    def test_negative_returns_none(self, neg):
        assert baby_shelf_threshold_price(neg) is None

    def test_exact_threshold_share_count_is_one_dollar(self):
        assert baby_shelf_threshold_price(75_000_000) == pytest.approx(1.0)

    def test_half_share_count_is_two_dollars(self):
        assert baby_shelf_threshold_price(37_500_000) == pytest.approx(2.0)

    def test_very_small_float_no_overflow(self):
        assert baby_shelf_threshold_price(1) == pytest.approx(75_000_000.0)

    def test_int_and_float_both_yield_float(self):
        as_int = baby_shelf_threshold_price(50_000_000)
        as_float = baby_shelf_threshold_price(50_000_000.0)
        assert isinstance(as_int, float)
        assert isinstance(as_float, float)
        assert as_int == pytest.approx(as_float)


# ════════════════════════════════════════════════════════════════════════
# ib6_max_raise  (pure)
# ════════════════════════════════════════════════════════════════════════
class TestIb6MaxRaise:
    @pytest.mark.parametrize("fs,price", [
        (None, None),
        (None, 5.0),
        (5.0, None),
        (0, 5.0),
        (5.0, 0),
        (-3, 5.0),
        (5.0, -3),
    ])
    def test_missing_or_nonpositive_returns_none(self, fs, price):
        assert ib6_max_raise(fs, price) is None

    def test_exact_third_division(self):
        assert ib6_max_raise(3, 1) == pytest.approx(1.0)

    def test_ninety_million_at_one_dollar(self):
        assert ib6_max_raise(90_000_000, 1) == pytest.approx(30_000_000.0)

    def test_result_is_float(self):
        assert isinstance(ib6_max_raise(3, 1), float)


# ════════════════════════════════════════════════════════════════════════
# ib6_basis_shares  (io_mockable: fetch_implied_outstanding_cached)
# ════════════════════════════════════════════════════════════════════════
class TestIb6BasisShares:
    def test_single_class_uses_float_primary(self, monkeypatch):
        _patch_implied(monkeypatch, n_classes=1)
        assert ib6_basis_shares(1, 40_000_000, 99_000_000) == pytest.approx(
            40_000_000.0)

    def test_single_class_float_none_falls_back_to_os(self, monkeypatch):
        _patch_implied(monkeypatch, n_classes=1)
        assert ib6_basis_shares(1, None, 99_000_000) == pytest.approx(
            99_000_000.0)

    def test_single_class_both_none_returns_none(self, monkeypatch):
        _patch_implied(monkeypatch, n_classes=1)
        assert ib6_basis_shares(1, None, None) is None

    def test_multi_class_uses_os_primary(self, monkeypatch):
        _patch_implied(monkeypatch, n_classes=2)
        # multi-class prefers latest_os over float
        assert ib6_basis_shares(1, 40_000_000, 99_000_000) == pytest.approx(
            99_000_000.0)

    def test_multi_class_os_none_falls_back_to_float(self, monkeypatch):
        _patch_implied(monkeypatch, n_classes=3)
        assert ib6_basis_shares(1, 40_000_000, None) == pytest.approx(
            40_000_000.0)

    def test_exception_treated_as_single_class(self, monkeypatch):
        _patch_implied(monkeypatch, raise_exc=True)
        # single-class branch => float is primary
        assert ib6_basis_shares(1, 40_000_000, 99_000_000) == pytest.approx(
            40_000_000.0)

    def test_classes_len_one_is_single_class(self, monkeypatch):
        _patch_implied(monkeypatch, n_classes=1)
        assert ib6_basis_shares(1, 12_345, 999) == pytest.approx(12_345.0)

    def test_primary_zero_falls_through_to_fallback(self, monkeypatch):
        # single-class: primary=float_shares=0 (falsy) -> fallback=latest_os
        _patch_implied(monkeypatch, n_classes=1)
        assert ib6_basis_shares(1, 0, 88_000_000) == pytest.approx(
            88_000_000.0)

    def test_int_inputs_coerced_to_float(self, monkeypatch):
        _patch_implied(monkeypatch, n_classes=1)
        out = ib6_basis_shares(1, 50_000_000, None)
        assert isinstance(out, float)


# ════════════════════════════════════════════════════════════════════════
# _durably_exited_baby_shelf  (io_mockable: finviz _client + _ticker_for_cik)
# ════════════════════════════════════════════════════════════════════════
class TestDurablyExitedBabyShelf:
    def test_basis_none_returns_false_no_feed(self, monkeypatch):
        client = _FakeClient([100.0] * 90)
        _patch_client(monkeypatch, client)
        assert _durably_exited_baby_shelf(1, None, 5.0) is False
        assert client.called is False

    def test_price_none_returns_false_no_feed(self, monkeypatch):
        client = _FakeClient([100.0] * 90)
        _patch_client(monkeypatch, client)
        assert _durably_exited_baby_shelf(1, 50_000_000, None) is False
        assert client.called is False

    def test_margin_just_below_returns_false_no_feed(
            self, monkeypatch, temp_db):
        temp_db.add_company(1, "TEST")
        client = _FakeClient([100.0] * 90)
        _patch_client(monkeypatch, client)
        # value just under 1.2 * 75M = 90M
        basis = 1.0
        price = BABY_SHELF_EXIT_MARGIN_MULT * THRESHOLD - 1.0
        assert _durably_exited_baby_shelf(1, basis, price) is False
        # margin check is before the feed call
        assert client.called is False

    def test_margin_exactly_at_threshold_passes_margin_gate(
            self, monkeypatch, temp_db):
        # basis*price == 1.2 * 75M exactly => passes the inverted '<' gate.
        # Provide persistent closes so persistence also passes => True.
        temp_db.add_company(1, "TEST")
        basis = 1.0
        price = BABY_SHELF_EXIT_MARGIN_MULT * THRESHOLD  # exactly 90M
        # each close c must clear bare 75M: c >= 75M
        client = _FakeClient([THRESHOLD * 2.0] * 90)
        _patch_client(monkeypatch, client)
        assert _durably_exited_baby_shelf(1, basis, price) is True

    def test_persistence_exactly_80pct_returns_true(
            self, monkeypatch, temp_db):
        temp_db.add_company(1, "TEST")
        basis = 1.0
        price = float(THRESHOLD * 2)  # clears margin
        # 8 above, 2 below => 80% exactly => >= boundary => True
        above = [float(THRESHOLD)] * 8        # basis*c == 75M => >= passes
        below = [float(THRESHOLD - 1)] * 2    # below bare threshold
        client = _FakeClient(above + below)
        _patch_client(monkeypatch, client)
        assert _durably_exited_baby_shelf(1, basis, price) is True

    def test_persistence_79pct_returns_false(self, monkeypatch, temp_db):
        temp_db.add_company(1, "TEST")
        basis = 1.0
        price = float(THRESHOLD * 2)
        # 79 above, 21 below => 0.79 < 0.80 => False
        above = [float(THRESHOLD)] * 79
        below = [float(THRESHOLD - 1)] * 21
        client = _FakeClient(above + below)
        _patch_client(monkeypatch, client)
        assert _durably_exited_baby_shelf(1, basis, price) is False

    def test_fewer_than_two_closes_returns_false(self, monkeypatch, temp_db):
        temp_db.add_company(1, "TEST")
        basis = 1.0
        price = float(THRESHOLD * 2)
        client = _FakeClient([float(THRESHOLD * 3)])  # single close
        _patch_client(monkeypatch, client)
        assert _durably_exited_baby_shelf(1, basis, price) is False

    def test_empty_closes_returns_false_and_caches_empty(
            self, monkeypatch, temp_db):
        temp_db.add_company(1, "TEST")
        client = _FakeClient([])
        _patch_client(monkeypatch, client)
        assert _durably_exited_baby_shelf(1, 1.0, float(THRESHOLD * 2)) is False
        assert bs._BABY_EXIT_CLOSES_CACHE[1] == []

    def test_feed_exception_caches_empty_and_returns_false(
            self, monkeypatch, temp_db):
        temp_db.add_company(1, "TEST")
        client = _FakeClient(None, raise_exc=True)
        _patch_client(monkeypatch, client)
        assert _durably_exited_baby_shelf(1, 1.0, float(THRESHOLD * 2)) is False
        assert client.called is True
        # negative result cached
        assert bs._BABY_EXIT_CLOSES_CACHE[1] == []

    def test_cache_hit_skips_feed(self, monkeypatch, temp_db):
        temp_db.add_company(1, "TEST")
        # pre-seed cache so feed is never consulted
        bs._BABY_EXIT_CLOSES_CACHE[7] = [float(THRESHOLD)] * 10
        client = _FakeClient(None, raise_exc=True)  # would raise if called
        _patch_client(monkeypatch, client)
        assert _durably_exited_baby_shelf(
            7, 1.0, float(THRESHOLD * 2)) is True
        assert client.called is False

    def test_none_and_zero_closes_filtered(self, monkeypatch, temp_db):
        temp_db.add_company(1, "TEST")
        basis = 1.0
        price = float(THRESHOLD * 2)
        # Two real closes clear, plus None/0 noise that must be filtered out.
        client = _FakeClient(
            [float(THRESHOLD), None, 0, 0.0, float(THRESHOLD)])
        _patch_client(monkeypatch, client)
        # after filtering: 2 vals, both >= 75M => 100% => True
        assert _durably_exited_baby_shelf(1, basis, price) is True
        assert bs._BABY_EXIT_CLOSES_CACHE[1] == [
            float(THRESHOLD), float(THRESHOLD)]

    def test_persistence_uses_bare_threshold_not_margin(
            self, monkeypatch, temp_db):
        # closes that clear bare 75M but NOT 1.2x. Margin gate uses the
        # caller price (passes); persistence must count these as 'above'.
        temp_db.add_company(1, "TEST")
        basis = 1.0
        price = float(THRESHOLD * 2)  # passes margin
        # each close == exactly 75M -> clears bare, below 1.2x margin
        client = _FakeClient([float(THRESHOLD)] * 5)
        _patch_client(monkeypatch, client)
        assert _durably_exited_baby_shelf(1, basis, price) is True

    def test_no_company_row_ticker_none_returns_false(
            self, monkeypatch, temp_db):
        # No company => _ticker_for_cik None => vals stay [] => False
        client = _FakeClient([float(THRESHOLD * 3)] * 90)
        _patch_client(monkeypatch, client)
        assert _durably_exited_baby_shelf(
            999, 1.0, float(THRESHOLD * 2)) is False
        assert client.called is False  # never reached the feed
        assert bs._BABY_EXIT_CLOSES_CACHE[999] == []

    def test_per_close_strictly_below_bare_threshold_not_counted(
            self, monkeypatch, temp_db):
        # Persistence counts a close only when basis*c >= bare $75M.
        # A close one cent under the per-share line that hits exactly $75M
        # must NOT be counted as 'above'. basis=1.0 so the line is c==75M.
        temp_db.add_company(1, "TEST")
        basis = 1.0
        price = float(THRESHOLD * 2)  # clears the margin gate comfortably
        # 5 closes each just under 75M => 0/5 above => 0.0 < 0.80 => False,
        # proving the gate is a real >= on the float value, not >.
        client = _FakeClient([float(THRESHOLD) - 0.01] * 5)
        _patch_client(monkeypatch, client)
        assert _durably_exited_baby_shelf(1, basis, price) is False

    def test_basis_scales_persistence_count(self, monkeypatch, temp_db):
        # Persistence multiplies the wired-through basis_shares by each close.
        # Holding the close fixed at $40, a basis of 2M shares yields a float
        # value of 2M*40 = $80M (>= bare $75M => above), while a basis of 1M
        # yields only $40M (< $75M => below). Same close stream, opposite
        # verdict => proves the multiply is on basis, not the raw close.
        temp_db.add_company(1, "TEST")
        close = 40.0
        feed = [close] * 10
        # margin gate uses basis * effective_price; pick effective_price so
        # the 60-day-high clears 1.2*75M=90M for both bases under test.
        price = 100.0  # 2M*100=200M and 1M*100=100M both >= 90M
        _patch_client(monkeypatch, _FakeClient(feed))
        assert _durably_exited_baby_shelf(1, 2_000_000.0, price) is True
        bs._BABY_EXIT_CLOSES_CACHE.clear()
        _patch_client(monkeypatch, _FakeClient(feed))
        assert _durably_exited_baby_shelf(1, 1_000_000.0, price) is False


# ════════════════════════════════════════════════════════════════════════
# has_eligible_shelf  (io_mockable: derive_shelf_status)
# ════════════════════════════════════════════════════════════════════════
class TestHasEligibleShelf:
    def test_empty_status_list_false(self, monkeypatch):
        _patch_shelf_status(monkeypatch, [])
        assert has_eligible_shelf(1, today=date(2026, 1, 1)) is False

    def test_lowercase_s3_effective_true(self, monkeypatch):
        _patch_shelf_status(
            monkeypatch,
            [{"form": "s-3", "derived_status": "effective"}])
        assert has_eligible_shelf(1, today=date(2026, 1, 1)) is True

    def test_s3asr_prefix_match_true(self, monkeypatch):
        _patch_shelf_status(
            monkeypatch,
            [{"form": "S-3ASR", "derived_status": "active"}])
        assert has_eligible_shelf(1, today=date(2026, 1, 1)) is True

    def test_f3_active_true(self, monkeypatch):
        _patch_shelf_status(
            monkeypatch,
            [{"form": "F-3", "derived_status": "active"}])
        assert has_eligible_shelf(1, today=date(2026, 1, 1)) is True

    def test_s1_effective_false_wrong_prefix(self, monkeypatch):
        _patch_shelf_status(
            monkeypatch,
            [{"form": "S-1", "derived_status": "effective"}])
        assert has_eligible_shelf(1, today=date(2026, 1, 1)) is False

    @pytest.mark.parametrize("status", ["withdrawn", "expired", "registered"])
    def test_s3_noneligible_status_false(self, monkeypatch, status):
        _patch_shelf_status(
            monkeypatch,
            [{"form": "S-3", "derived_status": status}])
        assert has_eligible_shelf(1, today=date(2026, 1, 1)) is False

    def test_form_none_skipped_no_crash(self, monkeypatch):
        _patch_shelf_status(
            monkeypatch,
            [{"form": None, "derived_status": "effective"}])
        assert has_eligible_shelf(1, today=date(2026, 1, 1)) is False

    def test_missing_derived_status_key_not_eligible(self, monkeypatch):
        _patch_shelf_status(
            monkeypatch, [{"form": "S-3"}])
        assert has_eligible_shelf(1, today=date(2026, 1, 1)) is False

    def test_multiple_statuses_one_eligible_true(self, monkeypatch):
        _patch_shelf_status(monkeypatch, [
            {"form": "S-1", "derived_status": "effective"},
            {"form": "S-3", "derived_status": "withdrawn"},
            {"form": "F-3", "derived_status": "active"},
        ])
        assert has_eligible_shelf(1, today=date(2026, 1, 1)) is True

    def test_today_forwarded_to_derive_shelf_status(self, monkeypatch):
        spy = {}
        _patch_shelf_status_spy(monkeypatch, [], spy)
        d = date(2025, 7, 4)
        has_eligible_shelf(42, today=d)
        assert spy["cik"] == 42
        assert spy["today"] == d


# ════════════════════════════════════════════════════════════════════════
# raised_under_ib6_last_12mo  (db_backed)
# ════════════════════════════════════════════════════════════════════════
class TestRaisedUnderIb6Last12mo:
    TODAY = date(2026, 1, 1)

    def _eligible(self, monkeypatch):
        _patch_shelf_status(
            monkeypatch,
            [{"form": "S-3", "derived_status": "effective"}])

    def _ineligible(self, monkeypatch):
        _patch_shelf_status(monkeypatch, [])

    def test_ineligible_returns_zero_result(self, monkeypatch, temp_db):
        self._ineligible(monkeypatch)
        # stage a drawdown that would otherwise count; must be ignored
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-1",
            event_date="2025-12-01", amount_usd=1_000_000.0)
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        assert out["eligible"] is False
        assert out["total"] == 0.0
        assert out["rows"] == []

    def test_basic_shelf_drawdown_summed(self, monkeypatch, temp_db):
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-1",
            event_date="2025-12-01", amount_usd=2_000_000.0,
            price=5.0, drawdown_party_canonical="Jefferies")
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        assert out["eligible"] is True
        assert out["total"] == pytest.approx(2_000_000.0)
        assert len(out["rows"]) == 1
        row = out["rows"][0]
        assert row["instrument_id"] == "SH-001"
        assert row["type"] == "shelf"
        assert row["proceeds"] == pytest.approx(2_000_000.0)
        assert row["counterparty"] == "Jefferies"
        assert row["accession"] == "acc-1"
        assert row["date"] == "2025-12-01"

    @pytest.mark.parametrize("amount", [None, 0, -50_000.0])
    def test_nonpositive_amount_skipped(self, monkeypatch, temp_db, amount):
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-1",
            event_date="2025-12-01", amount_usd=amount)
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        assert out["total"] == 0.0
        assert out["rows"] == []

    def test_dedupe_same_accession_within_5pct(self, monkeypatch, temp_db):
        # ATM + shelf double-log of one offering within 5% -> counted once.
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        temp_db.add_instrument("ATM-001", cik=1, type="atm")
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-dup",
            event_date="2025-11-01", amount_usd=1_000_000.0)
        temp_db.add_drawdown(
            "ATM-001", cik=1, accession_number="acc-dup",
            event_date="2025-11-02", amount_usd=1_040_000.0)  # +4% => dup
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        assert len(out["rows"]) == 1
        assert out["total"] == pytest.approx(1_000_000.0)

    def test_different_accessions_same_amount_both_counted(
            self, monkeypatch, temp_db):
        # Dedupe is keyed on accession_number, NOT amount: two distinct
        # accessions with IDENTICAL amounts are both genuine raises and must
        # both count (the dedupe `if prev_acc != acc: continue` short-circuits).
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-A",
            event_date="2025-06-01", amount_usd=1_000_000.0)
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-B",
            event_date="2025-07-01", amount_usd=1_000_000.0)
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        assert len(out["rows"]) == 2
        assert out["total"] == pytest.approx(2_000_000.0)

    def test_same_accession_over_5pct_both_counted(self, monkeypatch, temp_db):
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        temp_db.add_instrument("ATM-001", cik=1, type="atm")
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-x",
            event_date="2025-11-01", amount_usd=1_000_000.0)
        temp_db.add_drawdown(
            "ATM-001", cik=1, accession_number="acc-x",
            event_date="2025-11-02", amount_usd=1_100_000.0)  # +10% => both
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        assert len(out["rows"]) == 2
        assert out["total"] == pytest.approx(2_100_000.0)

    def test_equity_line_type_excluded(self, monkeypatch, temp_db):
        # only shelf/atm joined; 'eline' must be excluded.
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("EL-001", cik=1, type="eline")
        temp_db.add_drawdown(
            "EL-001", cik=1, accession_number="acc-el",
            event_date="2025-12-01", amount_usd=3_000_000.0)
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        assert out["total"] == 0.0
        assert out["rows"] == []

    def test_drawdown_on_cutoff_date_included(self, monkeypatch, temp_db):
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        # cutoff = today - 365 = 2025-01-01 (2026 not a leap year boundary)
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-c",
            event_date="2025-01-01", amount_usd=500_000.0)
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        assert out["total"] == pytest.approx(500_000.0)

    def test_drawdown_day_before_cutoff_excluded(self, monkeypatch, temp_db):
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-old",
            event_date="2024-12-31", amount_usd=500_000.0)
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        assert out["total"] == 0.0

    def test_drawdown_on_today_included(self, monkeypatch, temp_db):
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-now",
            event_date="2026-01-01", amount_usd=700_000.0)
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        assert out["total"] == pytest.approx(700_000.0)

    def test_warrant_underlying_added(self, monkeypatch, temp_db):
        # warrant created by a contributing accession + a valid price ->
        # adds count*price underlying row.
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-unit",
            event_date="2025-10-01", amount_usd=1_000_000.0, price=4.0)
        temp_db.add_instrument(
            "W-001", cik=1, type="warrant", created_accession="acc-unit",
            outstanding_json=json.dumps({"initial_count": 100_000}))
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        # 1M cash + 100k * $4 = 1.4M
        assert out["total"] == pytest.approx(1_400_000.0)
        wrow = [r for r in out["rows"] if r["type"] == "warrant_underlying"]
        assert len(wrow) == 1
        assert wrow[0]["proceeds"] == pytest.approx(400_000.0)
        assert wrow[0]["instrument_id"] == "W-001"
        assert wrow[0]["accession"] == "acc-unit"
        assert wrow[0]["date"] is None

    def test_warrant_no_price_skipped(self, monkeypatch, temp_db):
        # contributing drawdown carries no price => by_acc_price empty =>
        # warrant skipped.
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-np",
            event_date="2025-10-01", amount_usd=1_000_000.0, price=None)
        temp_db.add_instrument(
            "W-001", cik=1, type="warrant", created_accession="acc-np",
            outstanding_json=json.dumps({"initial_count": 100_000}))
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        assert out["total"] == pytest.approx(1_000_000.0)
        assert all(r["type"] != "warrant_underlying" for r in out["rows"])

    def test_warrant_missing_count_skipped(self, monkeypatch, temp_db):
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-mc",
            event_date="2025-10-01", amount_usd=1_000_000.0, price=4.0)
        temp_db.add_instrument(
            "W-001", cik=1, type="warrant", created_accession="acc-mc",
            outstanding_json=json.dumps({"exercise_price": 5.0}))
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        assert out["total"] == pytest.approx(1_000_000.0)

    def test_warrant_malformed_json_skipped(self, monkeypatch, temp_db):
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-bad",
            event_date="2025-10-01", amount_usd=1_000_000.0, price=4.0)
        temp_db.add_instrument(
            "W-001", cik=1, type="warrant", created_accession="acc-bad",
            outstanding_json="{not valid json")
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        assert out["total"] == pytest.approx(1_000_000.0)

    @pytest.mark.parametrize("count", [0, -10])
    def test_warrant_nonpositive_count_skipped(
            self, monkeypatch, temp_db, count):
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-z",
            event_date="2025-10-01", amount_usd=1_000_000.0, price=4.0)
        temp_db.add_instrument(
            "W-001", cik=1, type="warrant", created_accession="acc-z",
            outstanding_json=json.dumps({"initial_count": count}))
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        assert out["total"] == pytest.approx(1_000_000.0)

    def test_warrant_initial_count_preferred_over_count(
            self, monkeypatch, temp_db):
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-pref",
            event_date="2025-10-01", amount_usd=1_000_000.0, price=2.0)
        temp_db.add_instrument(
            "W-001", cik=1, type="warrant", created_accession="acc-pref",
            outstanding_json=json.dumps(
                {"initial_count": 50_000, "count": 999_999}))
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        # uses initial_count: 50k * $2 = 100k
        assert out["total"] == pytest.approx(1_100_000.0)

    def test_warrant_on_non_contributing_accession_excluded(
            self, monkeypatch, temp_db):
        # C&DI 116.24 only links a warrant whose created_accession matches a
        # CONTRIBUTING drawdown's accession. A warrant minted by some other
        # accession (here 'acc-OTHER') is excluded by the SQL `IN (...)` filter
        # even though a valid priced contributing drawdown exists.
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-good",
            event_date="2025-06-01", amount_usd=1_000_000.0, price=4.0)
        temp_db.add_instrument(
            "W-001", cik=1, type="warrant", created_accession="acc-OTHER",
            outstanding_json=json.dumps({"initial_count": 100_000}))
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        assert out["total"] == pytest.approx(1_000_000.0)
        assert all(r["type"] != "warrant_underlying" for r in out["rows"])

    def test_warrant_count_used_when_initial_count_absent(
            self, monkeypatch, temp_db):
        # `out.get("initial_count") or out.get("count")` -> with no
        # initial_count, the plain `count` key supplies the share count.
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-c",
            event_date="2025-06-01", amount_usd=1_000_000.0, price=3.0)
        temp_db.add_instrument(
            "W-001", cik=1, type="warrant", created_accession="acc-c",
            outstanding_json=json.dumps({"count": 20_000}))
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        # 1M cash + 20k * $3 = 1.06M
        assert out["total"] == pytest.approx(1_060_000.0)
        wrow = [r for r in out["rows"] if r["type"] == "warrant_underlying"][0]
        assert wrow["proceeds"] == pytest.approx(60_000.0)

    def test_no_contributing_accessions_no_warrant_block(
            self, monkeypatch, temp_db):
        # No qualifying drawdowns (all out of window) => contributing empty
        # => warrant block skipped even though a warrant exists.
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-old2",
            event_date="2020-01-01", amount_usd=1_000_000.0, price=4.0)
        temp_db.add_instrument(
            "W-001", cik=1, type="warrant", created_accession="acc-old2",
            outstanding_json=json.dumps({"initial_count": 100_000}))
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        assert out["total"] == 0.0
        assert out["rows"] == []

    def test_tiny_equal_positive_amounts_dedupe_no_zerodivision(
            self, monkeypatch, temp_db):
        # Two positive same-accession sub-dollar amounts exercise the dedupe
        # division (`abs(a-b)/max(|a|,|b|)`) without tripping the denom==0
        # guard (both pass the amount>0 filter, so denom is always > 0).
        # Equal amounts => 0% diff <= 5% => second collapses into the first.
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        temp_db.add_instrument("ATM-001", cik=1, type="atm")
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-tiny",
            event_date="2025-11-01", amount_usd=0.01)
        temp_db.add_drawdown(
            "ATM-001", cik=1, accession_number="acc-tiny",
            event_date="2025-11-02", amount_usd=0.01)
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        assert len(out["rows"]) == 1
        assert out["total"] == pytest.approx(0.01)

    def test_by_acc_price_takes_first_nonnull_by_event_date(
            self, monkeypatch, temp_db):
        # Same accession: first (by event_date) drawdown has no price, later
        # one does. by_acc_price should take the FIRST non-null encountered
        # in the query's event_date ordering.
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        # earliest non-null price = 3.0; a later 9.0 must NOT override it
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-pp",
            event_date="2025-09-01", amount_usd=1_000_000.0, price=3.0)
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-pp",
            event_date="2025-10-01", amount_usd=2_000_000.0, price=9.0)
        temp_db.add_instrument(
            "W-001", cik=1, type="warrant", created_accession="acc-pp",
            outstanding_json=json.dumps({"initial_count": 10_000}))
        out = raised_under_ib6_last_12mo(1, today=self.TODAY)
        # cash: 1M + 2M (>5% apart => both) = 3M; warrant 10k * $3 = 30k
        assert out["total"] == pytest.approx(3_030_000.0)
        wrow = [r for r in out["rows"] if r["type"] == "warrant_underlying"][0]
        assert wrow["proceeds"] == pytest.approx(30_000.0)


# ════════════════════════════════════════════════════════════════════════
# _unsold_live_atm_usd  (db_backed)
# ════════════════════════════════════════════════════════════════════════
class TestUnsoldLiveAtmUsd:
    def test_no_active_atm_returns_zero(self, temp_db):
        temp_db.add_company(1, "TEST")
        assert _unsold_live_atm_usd(1) == 0.0

    def test_explicit_remaining_capacity_added(self, temp_db):
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument(
            "ATM-001", cik=1, type="atm", status="active",
            outstanding_json=json.dumps({"remaining_capacity_usd": 5_000_000}))
        assert _unsold_live_atm_usd(1) == pytest.approx(5_000_000.0)

    def test_fallback_cap_minus_drawn(self, temp_db):
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument(
            "ATM-001", cik=1, type="atm", status="active",
            terms_json=json.dumps({"capacity_usd": 10_000_000}),
            outstanding_json=json.dumps({"drawn_usd": 4_000_000}))
        assert _unsold_live_atm_usd(1) == pytest.approx(6_000_000.0)

    def test_cap_present_drawn_none_treated_as_cap_minus_zero(self, temp_db):
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument(
            "ATM-001", cik=1, type="atm", status="active",
            terms_json=json.dumps({"capacity_usd": 8_000_000}),
            outstanding_json="{}")
        assert _unsold_live_atm_usd(1) == pytest.approx(8_000_000.0)

    def test_explicit_remaining_zero_does_not_fall_back_to_cap(self, temp_db):
        # The fallback to capacity_usd - drawn_usd only triggers when
        # remaining_capacity_usd is *absent* (None). An EXPLICIT 0 is `is not
        # None`, so the code keeps the 0 and the `remaining > 0` filter drops
        # the row entirely -- it must NOT silently fall back to cap-drawn (10M-1M).
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument(
            "ATM-001", cik=1, type="atm", status="active",
            terms_json=json.dumps({"capacity_usd": 10_000_000}),
            outstanding_json=json.dumps(
                {"remaining_capacity_usd": 0, "drawn_usd": 1_000_000}))
        assert _unsold_live_atm_usd(1) == 0.0

    def test_cap_none_no_remaining_contributes_nothing(self, temp_db):
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument(
            "ATM-001", cik=1, type="atm", status="active",
            terms_json="{}", outstanding_json="{}")
        assert _unsold_live_atm_usd(1) == 0.0

    def test_remaining_nonpositive_excluded(self, temp_db):
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument(
            "ATM-001", cik=1, type="atm", status="active",
            terms_json=json.dumps({"capacity_usd": 3_000_000}),
            outstanding_json=json.dumps({"drawn_usd": 3_000_000}))  # net 0
        assert _unsold_live_atm_usd(1) == 0.0

    @pytest.mark.parametrize("status", ["terminated", "inactive"])
    def test_inactive_atm_excluded(self, temp_db, status):
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument(
            "ATM-001", cik=1, type="atm", status=status,
            outstanding_json=json.dumps({"remaining_capacity_usd": 5_000_000}))
        assert _unsold_live_atm_usd(1) == 0.0

    def test_non_atm_type_excluded(self, temp_db):
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument(
            "SH-001", cik=1, type="shelf", status="active",
            outstanding_json=json.dumps({"remaining_capacity_usd": 5_000_000}))
        assert _unsold_live_atm_usd(1) == 0.0

    def test_malformed_json_row_skipped(self, temp_db):
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument(
            "ATM-bad", cik=1, type="atm", status="active",
            terms_json="{not json", outstanding_json="{}")
        temp_db.add_instrument(
            "ATM-ok", cik=1, type="atm", status="active",
            outstanding_json=json.dumps({"remaining_capacity_usd": 2_000_000}))
        # bad row skipped, good row counted
        assert _unsold_live_atm_usd(1) == pytest.approx(2_000_000.0)

    def test_multiple_active_atms_summed(self, temp_db):
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument(
            "ATM-1", cik=1, type="atm", status="active",
            outstanding_json=json.dumps({"remaining_capacity_usd": 1_000_000}))
        temp_db.add_instrument(
            "ATM-2", cik=1, type="atm", status="active",
            terms_json=json.dumps({"capacity_usd": 5_000_000}),
            outstanding_json=json.dumps({"drawn_usd": 2_000_000}))
        assert _unsold_live_atm_usd(1) == pytest.approx(4_000_000.0)

    def test_result_is_float(self, temp_db):
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument(
            "ATM-001", cik=1, type="atm", status="active",
            outstanding_json=json.dumps({"remaining_capacity_usd": 1_000_000}))
        assert isinstance(_unsold_live_atm_usd(1), float)


# ════════════════════════════════════════════════════════════════════════
# is_baby_shelf_restricted  (io_mockable: ib6_regime + override + basis)
# ════════════════════════════════════════════════════════════════════════
class TestIsBabyShelfRestricted:
    def test_regime_baby_no_durable_exit_true(self, monkeypatch):
        _patch_regime(monkeypatch, "baby")
        monkeypatch.setattr(
            bs, "_durably_exited_baby_shelf",
            lambda cik, basis, price: False, raising=True)
        _patch_implied(monkeypatch, n_classes=1)
        assert is_baby_shelf_restricted(1, 40_000_000, 40_000_000, 1.0) is True

    def test_regime_baby_durable_exit_overrides_to_false(self, monkeypatch):
        _patch_regime(monkeypatch, "baby")
        monkeypatch.setattr(
            bs, "_durably_exited_baby_shelf",
            lambda cik, basis, price: True, raising=True)
        _patch_implied(monkeypatch, n_classes=1)
        assert is_baby_shelf_restricted(
            1, 40_000_000, 40_000_000, 5.0) is False

    def test_regime_unrestricted_always_false(self, monkeypatch):
        _patch_regime(monkeypatch, "unrestricted")
        # even with a tiny float*price well under $75M
        assert is_baby_shelf_restricted(1, 1_000_000, 1_000_000, 1.0) is False

    def test_regime_none_uses_computed_fallback_restricted(self, monkeypatch):
        _patch_regime(monkeypatch, None)
        _patch_implied(monkeypatch, n_classes=1)
        # basis*price = 40M*1 = 40M < 75M => restricted
        assert is_baby_shelf_restricted(1, 40_000_000, None, 1.0) is True

    def test_ib6_regime_exception_caught_falls_through(
            self, monkeypatch, caplog):
        _patch_regime(monkeypatch, None, raise_exc=True)
        _patch_implied(monkeypatch, n_classes=1)
        # must not propagate; logs a warning; falls through to computed test
        with caplog.at_level("WARNING"):
            out = is_baby_shelf_restricted(1, 40_000_000, None, 1.0)
        assert out is True  # 40M < 75M
        assert any("ib6_regime failed" in r.message for r in caplog.records)

    def test_fallback_exactly_at_threshold_not_restricted(self, monkeypatch):
        # strict '<' => value == 75M is NOT restricted
        _patch_regime(monkeypatch, None)
        _patch_implied(monkeypatch, n_classes=1)
        assert is_baby_shelf_restricted(
            1, 75_000_000, None, 1.0) is False

    def test_fallback_just_below_threshold_restricted(self, monkeypatch):
        _patch_regime(monkeypatch, None)
        _patch_implied(monkeypatch, n_classes=1)
        assert is_baby_shelf_restricted(
            1, 74_999_999, None, 1.0) is True

    def test_fallback_basis_none_returns_false(self, monkeypatch):
        # ib6_basis_shares returns None (both float and os None) => value None
        _patch_regime(monkeypatch, None)
        _patch_implied(monkeypatch, n_classes=1)
        assert is_baby_shelf_restricted(1, None, None, 1.0) is False

    def test_fallback_price_none_returns_false(self, monkeypatch):
        _patch_regime(monkeypatch, None)
        _patch_implied(monkeypatch, n_classes=1)
        assert is_baby_shelf_restricted(1, 40_000_000, None, None) is False

    def test_baby_branch_computes_basis_for_override_even_with_no_price(
            self, monkeypatch):
        # baby stamp + missing price: override needs basis but price None =>
        # _durably_exited returns False => stays True (restricted).
        _patch_regime(monkeypatch, "baby")
        _patch_implied(monkeypatch, n_classes=1)
        # real override, no company row, price None => False => stays baby
        assert is_baby_shelf_restricted(1, 40_000_000, None, None) is True

    def test_baby_real_override_fires_via_price_feed(
            self, monkeypatch, temp_db):
        # Integration: drive the ACTUAL _durably_exited_baby_shelf (no stub)
        # through the public classifier. baby stamp + single-class so basis =
        # float_shares (50M) at effective_price 2.0 => 100M >= 90M margin, and
        # every close yields 50M*2=100M >= 75M => persistence 100% => override
        # fires => unrestricted (False).
        temp_db.add_company(1, "TEST")
        _patch_regime(monkeypatch, "baby")
        _patch_implied(monkeypatch, n_classes=1)
        _patch_client(monkeypatch, _FakeClient([2.0] * 90))
        assert is_baby_shelf_restricted(1, 50_000_000, None, 2.0) is False

    def test_baby_real_override_persistence_fails_stays_restricted(
            self, monkeypatch, temp_db):
        # Same wiring but the closes sit below the bare threshold on a
        # supermajority of days => persistence fails => stamp stands (True).
        # Margin still passes off the 60-day-high effective_price (2.0).
        temp_db.add_company(1, "TEST")
        _patch_regime(monkeypatch, "baby")
        _patch_implied(monkeypatch, n_classes=1)
        # 50M * 1.0 = 50M < 75M on every close => 0% persistence
        _patch_client(monkeypatch, _FakeClient([1.0] * 90))
        assert is_baby_shelf_restricted(1, 50_000_000, None, 2.0) is True


# ════════════════════════════════════════════════════════════════════════
# ib6_remaining  (db_backed + io)
# ════════════════════════════════════════════════════════════════════════
class TestIb6Remaining:
    TODAY = date(2026, 1, 1)

    def _eligible(self, monkeypatch):
        _patch_shelf_status(
            monkeypatch,
            [{"form": "S-3", "derived_status": "effective"}])

    def _ineligible(self, monkeypatch):
        _patch_shelf_status(monkeypatch, [])

    def test_cap_none_returns_none_early(self, monkeypatch, temp_db):
        # float falsy => cap None => None before any eligibility check
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        assert ib6_remaining(1, None, 5.0, today=self.TODAY) is None
        assert ib6_remaining(1, 10_000_000, 0, today=self.TODAY) is None

    def test_no_eligible_shelf_returns_none(self, monkeypatch, temp_db):
        self._ineligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        assert ib6_remaining(1, 90_000_000, 1.0, today=self.TODAY) is None

    def test_basic_assembly(self, monkeypatch, temp_db):
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        # no drawdowns => raised total 0
        out = ib6_remaining(1, 90_000_000, 1.0, today=self.TODAY)
        assert out is not None
        # cap = 90M * 1 / 3 = 30M
        assert out["ib6_capacity_usd"] == pytest.approx(30_000_000.0)
        assert out["raised_last_12mo_usd"] == pytest.approx(0.0)
        assert out["raisable_remaining_usd"] == pytest.approx(30_000_000.0)
        assert out["float_value_usd"] == pytest.approx(90_000_000.0)
        assert out["float_shares"] == 90_000_000
        assert out["price"] == 1.0
        # 90M >= 75M => not baby
        assert out["is_baby_shelf"] is False
        # threshold echoes baby_shelf_threshold_price
        assert out["threshold_price_to_exit_baby_shelf"] == pytest.approx(
            baby_shelf_threshold_price(90_000_000))
        assert out["raised_rows"] == []

    def test_raised_over_cap_clamped_to_zero(self, monkeypatch, temp_db):
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        # cap = 30M; raise 40M => remaining clamped to 0
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-big",
            event_date="2025-12-01", amount_usd=40_000_000.0)
        out = ib6_remaining(1, 90_000_000, 1.0, today=self.TODAY)
        assert out["raised_last_12mo_usd"] == pytest.approx(40_000_000.0)
        assert out["raisable_remaining_usd"] == 0.0

    def test_unsold_atm_clamps_new_takedown(self, monkeypatch, temp_db):
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        # cap=30M, no raises => remaining 30M; unsold ATM 50M => new takedown 0
        temp_db.add_instrument(
            "ATM-001", cik=1, type="atm", status="active",
            outstanding_json=json.dumps(
                {"remaining_capacity_usd": 50_000_000}))
        out = ib6_remaining(1, 90_000_000, 1.0, today=self.TODAY)
        assert out["unsold_live_atm_usd"] == pytest.approx(50_000_000.0)
        assert out["raisable_remaining_usd"] == pytest.approx(30_000_000.0)
        assert out["raisable_new_takedown_usd"] == 0.0

    def test_unsold_atm_below_remaining_leaves_positive_takedown(
            self, monkeypatch, temp_db):
        # When unsold ATM (5M) < remaining (30M), the new-takedown figure is
        # the positive remainder, NOT clamped: 30M - 5M = 25M.
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument(
            "ATM-001", cik=1, type="atm", status="active",
            outstanding_json=json.dumps(
                {"remaining_capacity_usd": 5_000_000}))
        out = ib6_remaining(1, 90_000_000, 1.0, today=self.TODAY)
        assert out["unsold_live_atm_usd"] == pytest.approx(5_000_000.0)
        assert out["raisable_remaining_usd"] == pytest.approx(30_000_000.0)
        assert out["raisable_new_takedown_usd"] == pytest.approx(25_000_000.0)

    def test_is_baby_shelf_true_below_threshold(self, monkeypatch, temp_db):
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        # float*price = 60M*1 = 60M < 75M => baby
        out = ib6_remaining(1, 60_000_000, 1.0, today=self.TODAY)
        assert out["is_baby_shelf"] is True
        assert out["float_value_usd"] == pytest.approx(60_000_000.0)

    def test_is_baby_shelf_false_exactly_at_threshold(
            self, monkeypatch, temp_db):
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        # float*price == 75M => not baby (strict <)
        out = ib6_remaining(1, 75_000_000, 1.0, today=self.TODAY)
        assert out["is_baby_shelf"] is False

    def test_raised_rows_passed_through(self, monkeypatch, temp_db):
        self._eligible(monkeypatch)
        temp_db.add_company(1, "TEST")
        temp_db.add_instrument("SH-001", cik=1, type="shelf")
        temp_db.add_drawdown(
            "SH-001", cik=1, accession_number="acc-r",
            event_date="2025-12-01", amount_usd=1_000_000.0,
            drawdown_party_canonical="ATM Agent")
        out = ib6_remaining(1, 90_000_000, 1.0, today=self.TODAY)
        assert len(out["raised_rows"]) == 1
        assert out["raised_rows"][0]["accession"] == "acc-r"
        assert out["raisable_remaining_usd"] == pytest.approx(29_000_000.0)

    def test_today_forwarded(self, monkeypatch, temp_db):
        spy = {}
        _patch_shelf_status_spy(
            monkeypatch,
            [{"form": "S-3", "derived_status": "effective"}], spy)
        temp_db.add_company(1, "TEST")
        d = date(2025, 3, 15)
        ib6_remaining(1, 90_000_000, 1.0, today=d)
        # has_eligible_shelf forwards today to derive_shelf_status
        assert spy["today"] == d
