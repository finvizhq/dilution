"""Unit tests for dilution/os_history.py.

Covers the pure date/split/stack helpers, the two DB-backed lookups
(_company_unit, _split_factors) against the autouse temp_db, and the
io_mockable orchestration (fetch_os_history / _query_first_with_facts /
fetch_os_history_cached / _ensure_identity) with the edgar Company seam
monkeypatched. NO network / edgar / LLM call is ever made.
"""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace as NS

import pytest

from dilution import os_history as oh


# ── shared fake-facts scaffolding (edgar seam) ───────────────────────
class FakeFacts:
    """Chainable query().by_concept(c, exact=True).execute() stub.

    ``results_by_concept`` maps a concept string to the list returned by
    execute(); a missing concept yields []. ``raise_for`` is a set of
    concepts whose by_concept() raises, exercising the swallow path.
    """

    def __init__(self, results_by_concept=None, raise_for=()):
        self._r = results_by_concept or {}
        self._raise_for = set(raise_for)
        self._c = None

    def query(self):
        return self

    def by_concept(self, c, exact=True):
        self._c = c
        if c in self._raise_for:
            raise RuntimeError(f"boom for {c}")
        return self

    def execute(self):
        return self._r.get(self._c, [])


def _make_company(facts_obj, recorder=None):
    """Return a fake edgar.Company class yielding ``facts_obj``."""

    class _Company:
        def __init__(self, cik):
            if recorder is not None:
                recorder.append(int(cik))

        def get_facts(self):
            return facts_obj

    return _Company


def _fact(period_end, numeric_value, filing_date, form_type="10-Q"):
    return NS(period_end=period_end, numeric_value=numeric_value,
              filing_date=filing_date, form_type=form_type)


@pytest.fixture
def no_identity(monkeypatch):
    """Neutralise the real set_identity and the once-only guard."""
    monkeypatch.setattr(oh, "set_identity", lambda *a, **k: None)
    monkeypatch.setattr(oh, "_IDENTITY_SET", True, raising=False)


@pytest.fixture(autouse=True)
def _clear_cache():
    """The module-level lru_cache + identity global persist across tests."""
    oh._cached.cache_clear()
    yield
    oh._cached.cache_clear()


# ── _quarter_end ─────────────────────────────────────────────────────
class TestQuarterEnd:
    @pytest.mark.parametrize("d,expected", [
        (date(2025, 1, 1), date(2025, 3, 31)),    # Q1 first day
        (date(2025, 2, 15), date(2025, 3, 31)),   # Q1 mid
        (date(2025, 3, 31), date(2025, 3, 31)),   # Q1 last day (idempotent)
        (date(2025, 4, 1), date(2025, 6, 30)),    # Q2 first day
        (date(2025, 5, 20), date(2025, 6, 30)),   # Q2 mid
        (date(2025, 6, 30), date(2025, 6, 30)),   # Q2 last (idempotent)
        (date(2025, 7, 1), date(2025, 9, 30)),    # Q3 first
        (date(2025, 8, 31), date(2025, 9, 30)),   # Q3 mid
        (date(2025, 9, 30), date(2025, 9, 30)),   # Q3 last (idempotent)
        (date(2025, 10, 1), date(2025, 12, 31)),  # Q4 first (month==12 branch)
        (date(2025, 11, 15), date(2025, 12, 31)),  # Q4 mid
        (date(2025, 12, 31), date(2025, 12, 31)),  # Q4 last (idempotent)
    ])
    def test_buckets_to_quarter_end(self, d, expected):
        assert oh._quarter_end(d) == expected

    def test_leap_year_feb_29_maps_to_mar_31(self):
        assert oh._quarter_end(date(2024, 2, 29)) == date(2024, 3, 31)

    @pytest.mark.parametrize("qe", [
        date(2025, 3, 31), date(2025, 6, 30),
        date(2025, 9, 30), date(2025, 12, 31),
    ])
    def test_quarter_ends_are_idempotent(self, qe):
        assert oh._quarter_end(qe) == qe


# ── _next_quarter_end ────────────────────────────────────────────────
class TestNextQuarterEnd:
    @pytest.mark.parametrize("q,expected", [
        (date(2025, 3, 31), date(2025, 6, 30)),
        (date(2025, 6, 30), date(2025, 9, 30)),
        (date(2025, 9, 30), date(2025, 12, 31)),
        (date(2025, 12, 31), date(2026, 3, 31)),   # year rollover
    ])
    def test_advances_one_quarter(self, q, expected):
        assert oh._next_quarter_end(q) == expected

    def test_non_quarter_end_input_still_advances(self):
        # Mar 31 + 1 day = Apr 1 -> Jun 30
        assert oh._next_quarter_end(date(2025, 3, 31)) == date(2025, 6, 30)


# ── _adjustment ──────────────────────────────────────────────────────
class TestAdjustment:
    def test_empty_splits_is_unity(self):
        assert oh._adjustment([], date(2025, 1, 1)) == 1.0

    def test_all_splits_on_or_before_d_unity(self):
        splits = [(date(2024, 1, 1), 0.004), (date(2024, 6, 1), 2.0)]
        assert oh._adjustment(splits, date(2025, 1, 1)) == 1.0

    def test_split_effective_exactly_on_d_excluded_strict(self):
        # eff > d is strict: a split effective ON d is NOT applied.
        splits = [(date(2024, 1, 1), 0.004)]
        assert oh._adjustment(splits, date(2024, 1, 1)) == 1.0

    def test_single_later_reverse_split(self):
        splits = [(date(2024, 1, 1), 0.004)]  # 1-for-250 reverse
        assert oh._adjustment(splits, date(2023, 1, 1)) == pytest.approx(0.004)

    def test_multiple_later_splits_multiply(self):
        splits = [(date(2024, 1, 1), 0.5), (date(2024, 6, 1), 0.1)]
        assert oh._adjustment(splits, date(2023, 1, 1)) == pytest.approx(0.05)

    def test_forward_split_multiplies_up(self):
        splits = [(date(2024, 1, 1), 3.0)]
        assert oh._adjustment(splits, date(2023, 1, 1)) == pytest.approx(3.0)

    def test_mixed_before_and_after_d(self):
        # only the 2024-06-01 (0.1) split is after 2024-03-01.
        splits = [(date(2024, 1, 1), 0.5), (date(2024, 6, 1), 0.1)]
        assert oh._adjustment(splits, date(2024, 3, 1)) == pytest.approx(0.1)


# ── _as_date ─────────────────────────────────────────────────────────
class TestAsDate:
    def test_none_returns_none(self):
        assert oh._as_date(None) is None

    def test_datetime_returns_its_date(self):
        v = datetime(2025, 6, 30, 12, 34, 56)
        out = oh._as_date(v)
        assert out == date(2025, 6, 30)
        # the datetime branch yields a pure date, not a datetime
        assert type(out) is date

    def test_plain_date_unchanged(self):
        v = date(2025, 6, 30)
        assert oh._as_date(v) is v

    def test_iso_date_string(self):
        assert oh._as_date("2025-06-30") == date(2025, 6, 30)

    def test_iso_datetime_string_returns_pure_date(self):
        # The survey speculated this returns a datetime, but the code calls
        # .date() on the parse result for the str path too -> pure date.
        out = oh._as_date("2025-06-30T12:00:00")
        assert out == date(2025, 6, 30)
        assert type(out) is date

    def test_malformed_string_returns_none(self):
        assert oh._as_date("not-a-date") is None

    def test_empty_string_returns_none(self):
        assert oh._as_date("") is None

    def test_integer_str_non_iso_returns_none(self):
        # str(12345) = "12345" is not a valid ISO date -> ValueError caught.
        assert oh._as_date(12345) is None


# ── _usd0 ────────────────────────────────────────────────────────────
class TestUsd0:
    @pytest.mark.parametrize("x,expected", [
        (0, "$0"),
        (1234567, "$1,234,567"),
        (1234.6, "$1,235"),     # rounds up
        (1234.5, "$1,234"),     # banker's rounding (round-half-to-even)
        (-1000, "$-1,000"),     # negative format
    ])
    def test_format(self, x, expected):
        assert oh._usd0(x) == expected


# ── build_fd_stack ───────────────────────────────────────────────────
class TestBuildFdStack:
    def test_empty_cards_is_empty(self):
        assert oh.build_fd_stack({}, 2.50) == []

    def test_warrant_segment_basic(self):
        cards = {"warrant": [{"remaining_outstanding": 1000},
                             {"remaining_outstanding": 500}]}
        segs = oh.build_fd_stack(cards, None)
        assert len(segs) == 1
        seg = segs[0]
        assert seg.key == "warrant"
        assert seg.label == "Warrants"
        assert seg.shares == 1500.0
        assert seg.price_based is False
        assert "2 warrant cards" in seg.note

    def test_warrant_pluralization_single_card(self):
        segs = oh.build_fd_stack(
            {"warrant": [{"remaining_outstanding": 10}]}, None)
        assert "1 warrant card," in segs[0].note
        assert "warrant cards" not in segs[0].note

    def test_warrant_all_none_or_zero_no_segment(self):
        cards = {"warrant": [{"remaining_outstanding": None},
                             {"remaining_outstanding": 0}]}
        assert oh.build_fd_stack(cards, None) == []

    def test_warrant_missing_key_coerced_to_zero(self):
        cards = {"warrant": [{}, {}]}  # no remaining_outstanding key
        assert oh.build_fd_stack(cards, None) == []

    def test_convertible_segment_and_key(self):
        cards = {"convertible": [{"remaining_shares_issuable": 500}]}
        segs = oh.build_fd_stack(cards, None)
        assert segs[0].key == "convertible"
        assert segs[0].label == "Convertible Notes"
        assert segs[0].shares == 500.0
        assert "1 note card" in segs[0].note

    def test_convertible_preferred_maps_to_preferred_key(self):
        cards = {"convertible_preferred": [{"remaining_shares_issuable": 200}]}
        segs = oh.build_fd_stack(cards, None)
        assert segs[0].key == "preferred"
        assert segs[0].label == "Convertible Preferred"
        assert segs[0].shares == 200.0
        assert "1 preferred card" in segs[0].note

    def test_convertible_missing_price_note(self):
        cards = {"convertible": [
            {"remaining_shares_issuable": 500},
            {"remaining_shares_issuable": None, "principal_remaining": 1000},
        ]}
        segs = oh.build_fd_stack(cards, None)
        assert segs[0].shares == 500.0
        assert "1 with undisclosed conversion price excluded" in segs[0].note

    def test_convertible_missing_only_when_principal_positive(self):
        # remaining None but principal 0 -> not counted as missing.
        cards = {"convertible": [
            {"remaining_shares_issuable": 500},
            {"remaining_shares_issuable": None, "principal_remaining": 0},
        ]}
        segs = oh.build_fd_stack(cards, None)
        assert "undisclosed conversion price" not in segs[0].note

    def test_convertible_all_missing_no_segment(self):
        # sh == 0 -> no segment, so the missing note is never shown.
        cards = {"convertible": [
            {"remaining_shares_issuable": None, "principal_remaining": 1000},
        ]}
        assert oh.build_fd_stack(cards, None) == []

    def test_price_none_suppresses_price_based(self):
        cards = {
            "warrant": [{"remaining_outstanding": 100}],
            "atm": [{"remaining_capacity": 5_000_000}],
            "equity_line": [{"remaining_capacity": 1_000_000}],
            "s1_offering": [{"anticipated_deal_size": 1_000_000,
                             "s1_status": "pending"}],
        }
        keys = [s.key for s in oh.build_fd_stack(cards, None)]
        assert keys == ["warrant"]

    @pytest.mark.parametrize("price", [0, -1.5])
    def test_non_positive_price_suppresses_price_based(self, price):
        cards = {"atm": [{"remaining_capacity": 5_000_000}]}
        assert oh.build_fd_stack(cards, price) == []

    def test_atm_division(self):
        cards = {"atm": [{"remaining_capacity": 5_000_000}]}
        segs = oh.build_fd_stack(cards, 2.50)
        assert segs[0].key == "atm"
        assert segs[0].shares == pytest.approx(2_000_000)
        assert segs[0].price_based is True
        assert "$5,000,000 remaining ATM capacity" in segs[0].note
        assert "$2.50" in segs[0].note

    def test_atm_raisable_capped_overrides_remaining_capacity(self):
        # IB6-cap precedence: raisable_capped present overrides remaining.
        cards = {"atm": [{"raisable_capped": 1_000_000,
                          "remaining_capacity": 5_000_000}]}
        segs = oh.build_fd_stack(cards, 2.50)
        assert segs[0].shares == pytest.approx(400_000)
        assert "$1,000,000 remaining ATM capacity" in segs[0].note

    def test_atm_falls_back_to_remaining_capacity_when_no_capped(self):
        cards = {"atm": [{"remaining_capacity": 5_000_000}]}
        segs = oh.build_fd_stack(cards, 2.50)
        assert segs[0].shares == pytest.approx(2_000_000)

    def test_atm_raisable_capped_present_but_none_suppresses_segment(self):
        # SUBTLETY: production uses dict.get("raisable_capped",
        # c.get("remaining_capacity")) — a PRESENT key wins even when its
        # value is None. get returns None (not the default), float(None or 0)
        # == 0, so the ATM segment vanishes despite a large remaining_capacity.
        # This is the dict.get(key, default)-vs-(key) or default distinction
        # the survey flagged; observed against current code.
        cards = {"atm": [{"raisable_capped": None,
                          "remaining_capacity": 5_000_000}]}
        assert oh.build_fd_stack(cards, 2.50) == []

    def test_atm_raisable_capped_zero_suppresses_segment(self):
        # raisable_capped == 0 (IB6 fully capped) -> float(0 or 0) == 0,
        # no ATM segment even with remaining_capacity present.
        cards = {"atm": [{"raisable_capped": 0,
                          "remaining_capacity": 5_000_000}]}
        assert oh.build_fd_stack(cards, 2.50) == []

    def test_atm_multiple_cards_summed_then_divided(self):
        # Aggregation across ATM cards (untested path): mixed capped and
        # uncapped cards sum BEFORE the single ÷price. 1,000,000 (capped) +
        # 5,000,000 (remaining fallback) = 6,000,000; /2.50 = 2,400,000.
        cards = {"atm": [
            {"raisable_capped": 1_000_000, "remaining_capacity": 9_999_999},
            {"remaining_capacity": 5_000_000},
        ]}
        segs = oh.build_fd_stack(cards, 2.50)
        assert len(segs) == 1
        assert segs[0].shares == pytest.approx(2_400_000)
        assert "$6,000,000 remaining ATM capacity" in segs[0].note

    def test_equity_line_multiple_active_cards_summed(self):
        # Two non-terminated ELOC cards aggregate before ÷price; a third
        # terminated one is dropped. (300_000 + 200_000) / 2.50 = 200_000.
        cards = {"equity_line": [
            {"remaining_capacity": 300_000},
            {"remaining_capacity": 200_000, "terminated": False},
            {"remaining_capacity": 9_000_000, "terminated": True},
        ]}
        segs = oh.build_fd_stack(cards, 2.50)
        assert segs[0].shares == pytest.approx(200_000)
        assert "$500,000 remaining equity-line capacity" in segs[0].note

    def test_s1_multiple_cards_only_active_statuses_summed(self):
        # pending + effective summed; withdrawn dropped.
        # (250_000 + 250_000) / 2.50 = 200_000.
        cards = {"s1_offering": [
            {"anticipated_deal_size": 250_000, "s1_status": "pending"},
            {"anticipated_deal_size": 250_000, "s1_status": "effective"},
            {"anticipated_deal_size": 9_000_000, "s1_status": "withdrawn"},
        ]}
        segs = oh.build_fd_stack(cards, 2.50)
        assert segs[0].shares == pytest.approx(200_000)
        assert "$500,000 anticipated deal size" in segs[0].note

    def test_convertible_two_missing_pluralizes_count(self):
        # TWO cards with remaining None AND principal>0 -> "2 with undisclosed
        # conversion price excluded" (the missing counter, distinct from the
        # single-missing case already covered).
        cards = {"convertible": [
            {"remaining_shares_issuable": 500},
            {"remaining_shares_issuable": None, "principal_remaining": 1000},
            {"remaining_shares_issuable": None, "principal_remaining": 2000},
        ]}
        segs = oh.build_fd_stack(cards, None)
        assert segs[0].shares == pytest.approx(500.0)
        assert "2 with undisclosed conversion price excluded" in segs[0].note

    def test_warrant_and_convertible_preferred_two_cards_plural_note(self):
        # convertible_preferred pluralizes its noun ("2 preferred cards").
        cards = {"convertible_preferred": [
            {"remaining_shares_issuable": 100},
            {"remaining_shares_issuable": 200},
        ]}
        segs = oh.build_fd_stack(cards, None)
        assert segs[0].shares == pytest.approx(300.0)
        assert "2 preferred cards" in segs[0].note

    def test_string_numeric_card_values_coerced_via_float(self):
        # float("1500") works (the code does float(c.get(...) or 0)); a numeric
        # string contributes its value rather than raising.
        cards = {"warrant": [{"remaining_outstanding": "1500"}]}
        segs = oh.build_fd_stack(cards, None)
        assert segs[0].shares == pytest.approx(1500.0)

    def test_equity_line_excludes_terminated(self):
        cards = {"equity_line": [
            {"remaining_capacity": 500_000},
            {"remaining_capacity": 999_999, "terminated": True},
        ]}
        segs = oh.build_fd_stack(cards, 2.50)
        assert segs[0].key == "equity_line"
        assert segs[0].shares == pytest.approx(200_000)
        assert "$500,000 remaining equity-line capacity" in segs[0].note

    def test_equity_line_all_terminated_no_segment(self):
        cards = {"equity_line": [
            {"remaining_capacity": 999_999, "terminated": True},
        ]}
        assert oh.build_fd_stack(cards, 2.50) == []

    @pytest.mark.parametrize("terminated", [False, None, 0, ""])
    def test_equity_line_falsy_terminated_is_included(self, terminated):
        # `if not c.get("terminated")` -> any falsy terminated value keeps
        # the card. Only a truthy flag drops it.
        cards = {"equity_line": [
            {"remaining_capacity": 500_000, "terminated": terminated},
        ]}
        segs = oh.build_fd_stack(cards, 2.50)
        assert segs[0].shares == pytest.approx(200_000)

    @pytest.mark.parametrize("status,expected_keys", [
        ("pending", ["s1"]),
        ("effective", ["s1"]),
        ("withdrawn", []),
        ("closed", []),
        (None, []),
    ])
    def test_s1_status_filter(self, status, expected_keys):
        cards = {"s1_offering": [
            {"anticipated_deal_size": 250_000, "s1_status": status},
        ]}
        keys = [s.key for s in oh.build_fd_stack(cards, 2.50)]
        assert keys == expected_keys

    def test_s1_division_and_note(self):
        cards = {"s1_offering": [
            {"anticipated_deal_size": 500_000, "s1_status": "effective"},
        ]}
        segs = oh.build_fd_stack(cards, 2.50)
        assert segs[0].key == "s1"
        assert segs[0].label == "S-1 Offering"
        assert segs[0].shares == pytest.approx(200_000)
        assert "$500,000 anticipated deal size" in segs[0].note

    def test_full_ordering(self):
        cards = {
            "warrant": [{"remaining_outstanding": 100}],
            "convertible": [{"remaining_shares_issuable": 200}],
            "convertible_preferred": [{"remaining_shares_issuable": 300}],
            "atm": [{"remaining_capacity": 2_500}],
            "equity_line": [{"remaining_capacity": 2_500}],
            "s1_offering": [{"anticipated_deal_size": 2_500,
                             "s1_status": "pending"}],
        }
        keys = [s.key for s in oh.build_fd_stack(cards, 2.50)]
        assert keys == ["warrant", "convertible", "preferred",
                        "atm", "equity_line", "s1"]


# ── _company_unit (db_backed) ────────────────────────────────────────
class TestCompanyUnit:
    def test_no_row_returns_false_none(self, temp_db):
        assert oh._company_unit(999) == (False, None)

    def test_not_fpi_null_ratio(self, temp_db):
        temp_db.add_company(10, is_fpi=0, ads_ratio=None)
        assert oh._company_unit(10) == (False, None)

    def test_fpi_with_ratio(self, temp_db):
        temp_db.add_company(11, is_fpi=1, ads_ratio=10.0)
        is_fpi, ratio = oh._company_unit(11)
        assert is_fpi is True
        assert ratio == pytest.approx(10.0)

    def test_fpi_without_ratio(self, temp_db):
        temp_db.add_company(12, is_fpi=1, ads_ratio=None)
        assert oh._company_unit(12) == (True, None)

    def test_is_fpi_coerced_to_bool(self, temp_db):
        temp_db.add_company(13, is_fpi=1, ads_ratio=4.0)
        temp_db.add_company(14, is_fpi=0, ads_ratio=4.0)
        assert oh._company_unit(13)[0] is True
        assert oh._company_unit(14)[0] is False

    def test_ratio_float_coercion(self, temp_db):
        temp_db.add_company(15, is_fpi=1, ads_ratio=2)
        ratio = oh._company_unit(15)[1]
        assert isinstance(ratio, float)
        assert ratio == pytest.approx(2.0)

    def test_ratio_zero_returned_as_float_zero(self, temp_db):
        # ads_ratio stored as 0 is NOT None, so it round-trips as 0.0 (the
        # `is not None` guard keeps it). fetch_os_history's separate
        # `if is_fpi and not ads_ratio` falsy-check is what later rejects it.
        temp_db.add_company(16, is_fpi=1, ads_ratio=0)
        is_fpi, ratio = oh._company_unit(16)
        assert is_fpi is True
        assert ratio == pytest.approx(0.0)
        assert isinstance(ratio, float)


# ── _split_factors (db_backed) ───────────────────────────────────────
class TestSplitFactors:
    def test_no_splits_empty(self, temp_db):
        assert oh._split_factors(700) == []

    def test_reverse_split_factor(self, temp_db):
        temp_db.add_split(701, effective_date="2024-01-01", pre=250, post=1)
        out = oh._split_factors(701)
        assert out == [(date(2024, 1, 1), pytest.approx(0.004))]

    def test_forward_split_factor(self, temp_db):
        temp_db.add_split(702, effective_date="2024-01-01", pre=1, post=2)
        out = oh._split_factors(702)
        assert out == [(date(2024, 1, 1), pytest.approx(2.0))]

    def test_ads_rows_excluded(self, temp_db):
        temp_db.add_split(703, effective_date="2024-01-01", pre=250, post=1)
        temp_db.add_split(703, effective_date="2024-03-01", pre=10, post=5,
                          units="ads")
        out = oh._split_factors(703)
        assert out == [(date(2024, 1, 1), pytest.approx(0.004))]

    def test_pre_zero_skipped(self, temp_db):
        # add_split derives direction from post>pre; pre=0 is falsy so the
        # row is skipped in _split_factors (div-by-zero guard).
        temp_db.add_split(704, effective_date="2024-01-01", pre=0, post=1)
        assert oh._split_factors(704) == []

    def test_ascending_by_effective_date(self, temp_db):
        # insert out of order
        temp_db.add_split(705, effective_date="2025-06-01", pre=1, post=2)
        temp_db.add_split(705, effective_date="2024-01-01", pre=250, post=1)
        temp_db.add_split(705, effective_date="2024-09-01", pre=2, post=1)
        out = oh._split_factors(705)
        dates = [d for d, _ in out]
        assert dates == sorted(dates)
        assert dates == [date(2024, 1, 1), date(2024, 9, 1), date(2025, 6, 1)]

    def test_malformed_effective_date_skipped(self, temp_db):
        # A bad date string is skipped via except continue, good row kept.
        temp_db.execute(
            """INSERT INTO dilution_splits
                 (cik, effective_date, pre, post, direction, units,
                  source, fetched_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (706, "garbage", 250, 1, "reverse", "common", "f",
             "2026-01-01T00:00:00Z"))
        temp_db.add_split(706, effective_date="2024-01-01", pre=1, post=2)
        out = oh._split_factors(706)
        assert out == [(date(2024, 1, 1), pytest.approx(2.0))]

    def test_mix_common_and_ads_only_common(self, temp_db):
        temp_db.add_split(707, effective_date="2024-01-01", pre=1, post=2)
        temp_db.add_split(707, effective_date="2024-02-01", pre=5, post=1,
                          units="ads")
        temp_db.add_split(707, effective_date="2024-03-01", pre=10, post=1)
        out = oh._split_factors(707)
        assert [d for d, _ in out] == [date(2024, 1, 1), date(2024, 3, 1)]

    def test_multiple_common_splits_all_returned(self, temp_db):
        temp_db.add_split(708, effective_date="2024-01-01", pre=2, post=1)
        temp_db.add_split(708, effective_date="2024-06-01", pre=1, post=3)
        out = oh._split_factors(708)
        assert len(out) == 2
        assert out[0] == (date(2024, 1, 1), pytest.approx(0.5))
        assert out[1] == (date(2024, 6, 1), pytest.approx(3.0))


# ── _query_first_with_facts (io_mockable) ────────────────────────────
class TestQueryFirstWithFacts:
    def test_first_concept_wins(self):
        facts = FakeFacts({oh._OS_CONCEPTS[0]: [1, 2, 3]})
        result, concept = oh._query_first_with_facts(facts, oh._OS_CONCEPTS)
        assert result == [1, 2, 3]
        assert concept == oh._OS_CONCEPTS[0]

    def test_first_empty_second_wins(self):
        facts = FakeFacts({oh._OS_CONCEPTS[1]: [9]})
        result, concept = oh._query_first_with_facts(facts, oh._OS_CONCEPTS)
        assert result == [9]
        assert concept == oh._OS_CONCEPTS[1]

    def test_first_raises_falls_through_to_second(self):
        facts = FakeFacts({oh._OS_CONCEPTS[1]: [5]},
                          raise_for={oh._OS_CONCEPTS[0]})
        result, concept = oh._query_first_with_facts(facts, oh._OS_CONCEPTS)
        assert result == [5]
        assert concept == oh._OS_CONCEPTS[1]

    def test_all_empty_returns_none(self):
        facts = FakeFacts({})
        assert oh._query_first_with_facts(facts, oh._OS_CONCEPTS) == ([], None)

    def test_all_raise_returns_none(self):
        facts = FakeFacts({}, raise_for=set(oh._OS_CONCEPTS))
        assert oh._query_first_with_facts(facts, oh._OS_CONCEPTS) == ([], None)


# ── fetch_os_history (io_mockable) ───────────────────────────────────
class TestFetchOsHistory:
    def test_fpi_missing_ratio_early_return(self, temp_db, no_identity,
                                            monkeypatch):
        temp_db.add_company(800, is_fpi=1, ads_ratio=None)

        def _boom(cik):
            raise AssertionError("Company must not be called on early return")

        monkeypatch.setattr(oh, "Company", _boom)
        h = oh.fetch_os_history(800, as_of=date(2025, 12, 31))
        assert h.warnings == ("fpi_ads_ratio_missing",)
        assert h.series == []

    def test_facts_fetch_failed(self, temp_db, no_identity, monkeypatch):
        class _BadCompany:
            def __init__(self, cik):
                pass

            def get_facts(self):
                raise RuntimeError("network down")

        monkeypatch.setattr(oh, "Company", _BadCompany)
        h = oh.fetch_os_history(801, as_of=date(2025, 12, 31))
        assert h.warnings == ("facts_fetch_failed",)
        assert h.concept is None

    def test_concept_missing(self, temp_db, no_identity, monkeypatch):
        monkeypatch.setattr(oh, "Company", _make_company(FakeFacts({})))
        h = oh.fetch_os_history(802, as_of=date(2025, 12, 31))
        assert h.warnings == ("concept_missing",)
        assert h.concept is None

    def test_no_facts_in_window(self, temp_db, no_identity, monkeypatch):
        # facts exist for the concept but every period_end is out of window.
        facts_obj = FakeFacts({oh._OS_CONCEPTS[0]: [
            _fact("2000-03-31", 1000, "2000-04-10"),
        ]})
        monkeypatch.setattr(oh, "Company", _make_company(facts_obj))
        h = oh.fetch_os_history(803, as_of=date(2025, 12, 31))
        assert h.warnings == ("no_facts_in_window",)
        # concept was found, so it is recorded even though no rows in window.
        assert h.concept == oh._OS_CONCEPTS[0]

    def test_dedup_restatement_latest_filing_wins(self, temp_db, no_identity,
                                                  monkeypatch):
        facts_obj = FakeFacts({oh._OS_CONCEPTS[0]: [
            _fact("2025-03-31", 1000, "2025-04-10", "10-Q"),
            _fact("2025-03-31", 1100, "2025-05-10", "10-Q/A"),
        ]})
        monkeypatch.setattr(oh, "Company", _make_company(facts_obj))
        h = oh.fetch_os_history(804, as_of=date(2025, 3, 31))
        assert len(h.series) == 1
        assert h.series[0].raw_shares == 1100.0
        assert h.series[0].form == "10-Q/A"

    def test_latest_cover_date_in_quarter_wins(self, temp_db, no_identity,
                                               monkeypatch):
        # Two facts landing in the SAME calendar quarter (Q1-2025) but with
        # different cover (period_end) dates. The bucket loop iterates
        # sorted(by_end) and OVERWRITES, so the later cover date (Mar 20)
        # is the surviving representative for that quarter — NOT the earlier
        # Jan 15 one, even though Jan 15 has a later filing_date here. This
        # is the "latest cover date inside the quarter wins" rule and is
        # distinct from the period_end-restatement dedup.
        facts_obj = FakeFacts({oh._OS_CONCEPTS[0]: [
            _fact("2025-01-15", 1000, "2025-09-01", "10-K"),
            _fact("2025-03-20", 2000, "2025-04-01", "10-Q"),
        ]})
        monkeypatch.setattr(oh, "Company", _make_company(facts_obj))
        h = oh.fetch_os_history(821, as_of=date(2025, 3, 31))
        assert len(h.series) == 1
        p = h.series[0]
        assert p.quarter_end == date(2025, 3, 31)
        assert p.raw_shares == 2000.0
        assert p.source_date == date(2025, 3, 20)
        assert p.form == "10-Q"

    def test_fpi_zero_ads_ratio_treated_as_missing(self, temp_db, no_identity,
                                                   monkeypatch):
        # ads_ratio stored as 0 is falsy, so `if is_fpi and not ads_ratio`
        # fires the same early return as a NULL ratio — and Company is never
        # constructed (no facts fetch).
        temp_db.add_company(822, is_fpi=1, ads_ratio=0)

        def _boom(cik):
            raise AssertionError("Company must not be called on early return")

        monkeypatch.setattr(oh, "Company", _boom)
        h = oh.fetch_os_history(822, as_of=date(2025, 12, 31))
        assert h.warnings == ("fpi_ads_ratio_missing",)
        assert h.series == []

    def test_multiple_quarters_distinct_facts_no_carry(self, temp_db,
                                                       no_identity, monkeypatch):
        # Consecutive quarters each with their own fact -> every point is
        # real (carried=False), and split_adjusted defaults False with no
        # splits. Sanity check on the non-carry main path.
        facts_obj = FakeFacts({oh._OS_CONCEPTS[0]: [
            _fact("2025-03-31", 1000, "2025-04-10"),
            _fact("2025-06-30", 1100, "2025-07-10"),
        ]})
        monkeypatch.setattr(oh, "Company", _make_company(facts_obj))
        h = oh.fetch_os_history(823, as_of=date(2025, 6, 30))
        assert [(p.quarter_end, p.raw_shares, p.carried) for p in h.series] == [
            (date(2025, 3, 31), 1000.0, False),
            (date(2025, 6, 30), 1100.0, False),
        ]

    @pytest.mark.parametrize("bad_value", [0, -5, None, "abc"])
    def test_nonpositive_or_bad_numeric_skipped(self, temp_db, no_identity,
                                                 monkeypatch, bad_value):
        # one good fact in a later quarter, one bad fact earlier. The bad one
        # is skipped, leaving only the good quarter (no carry-forward before
        # the first real point).
        facts_obj = FakeFacts({oh._OS_CONCEPTS[0]: [
            _fact("2025-03-31", bad_value, "2025-04-10"),
            _fact("2025-06-30", 2000, "2025-07-10"),
        ]})
        monkeypatch.setattr(oh, "Company", _make_company(facts_obj))
        h = oh.fetch_os_history(805, as_of=date(2025, 6, 30))
        # series starts at the first VALID bucket (Q2)
        assert [p.quarter_end for p in h.series] == [date(2025, 6, 30)]
        assert h.series[0].raw_shares == 2000.0

    def test_period_end_none_skipped(self, temp_db, no_identity, monkeypatch):
        facts_obj = FakeFacts({oh._OS_CONCEPTS[0]: [
            _fact(None, 1000, "2025-04-10"),
            _fact("2025-06-30", 2000, "2025-07-10"),
        ]})
        monkeypatch.setattr(oh, "Company", _make_company(facts_obj))
        h = oh.fetch_os_history(806, as_of=date(2025, 6, 30))
        assert [p.quarter_end for p in h.series] == [date(2025, 6, 30)]

    def test_carry_forward_fills_empty_quarters(self, temp_db, no_identity,
                                                monkeypatch):
        facts_obj = FakeFacts({oh._OS_CONCEPTS[0]: [
            _fact("2025-03-31", 1000, "2025-04-10", "10-Q"),
            _fact("2025-09-30", 2000, "2025-10-10", "10-Q"),
        ]})
        monkeypatch.setattr(oh, "Company", _make_company(facts_obj))
        h = oh.fetch_os_history(807, as_of=date(2025, 12, 31))
        rows = [(p.quarter_end, p.shares, p.carried) for p in h.series]
        assert rows == [
            (date(2025, 3, 31), 1000.0, False),
            (date(2025, 6, 30), 1000.0, True),   # carried from Q1
            (date(2025, 9, 30), 2000.0, False),
            (date(2025, 12, 31), 2000.0, True),  # carried from Q3
        ]

    def test_split_adjustment_applied(self, temp_db, no_identity, monkeypatch):
        # 1-for-250 reverse effective AFTER the cover date divides by 250.
        temp_db.add_split(808, effective_date="2025-06-01", pre=250, post=1)
        facts_obj = FakeFacts({oh._OS_CONCEPTS[0]: [
            _fact("2025-03-31", 250_000, "2025-04-10"),
        ]})
        monkeypatch.setattr(oh, "Company", _make_company(facts_obj))
        h = oh.fetch_os_history(808, as_of=date(2025, 3, 31))
        p = h.series[0]
        assert p.raw_shares == 250_000.0
        assert p.shares == pytest.approx(1000.0)
        assert p.split_adjusted is True

    def test_no_split_means_split_adjusted_false(self, temp_db, no_identity,
                                                 monkeypatch):
        facts_obj = FakeFacts({oh._OS_CONCEPTS[0]: [
            _fact("2025-03-31", 1000, "2025-04-10"),
        ]})
        monkeypatch.setattr(oh, "Company", _make_company(facts_obj))
        h = oh.fetch_os_history(809, as_of=date(2025, 3, 31))
        assert h.series[0].split_adjusted is False

    def test_fpi_division_by_ads_ratio(self, temp_db, no_identity, monkeypatch):
        temp_db.add_company(810, is_fpi=1, ads_ratio=10.0)
        facts_obj = FakeFacts({oh._OS_CONCEPTS[0]: [
            _fact("2025-03-31", 1000, "2025-04-10", "20-F"),
        ]})
        monkeypatch.setattr(oh, "Company", _make_company(facts_obj))
        h = oh.fetch_os_history(810, as_of=date(2025, 3, 31))
        assert h.ads_ratio == pytest.approx(10.0)
        assert h.series[0].raw_shares == 1000.0
        assert h.series[0].shares == pytest.approx(100.0)

    def test_non_fpi_ads_ratio_none_on_history(self, temp_db, no_identity,
                                               monkeypatch):
        facts_obj = FakeFacts({oh._OS_CONCEPTS[0]: [
            _fact("2025-03-31", 1000, "2025-04-10"),
        ]})
        monkeypatch.setattr(oh, "Company", _make_company(facts_obj))
        h = oh.fetch_os_history(811, as_of=date(2025, 3, 31))
        assert h.ads_ratio is None

    def test_concept_probe_order_dei_first(self, temp_db, no_identity,
                                           monkeypatch):
        # both concepts have data; dei must win.
        facts_obj = FakeFacts({
            oh._OS_CONCEPTS[0]: [_fact("2025-03-31", 1000, "2025-04-10")],
            oh._OS_CONCEPTS[1]: [_fact("2025-03-31", 9999, "2025-04-10")],
        })
        monkeypatch.setattr(oh, "Company", _make_company(facts_obj))
        h = oh.fetch_os_history(812, as_of=date(2025, 3, 31))
        assert h.concept == oh._OS_CONCEPTS[0]
        assert h.series[0].raw_shares == 1000.0

    def test_us_gaap_used_when_dei_missing(self, temp_db, no_identity,
                                           monkeypatch):
        facts_obj = FakeFacts({
            oh._OS_CONCEPTS[1]: [_fact("2025-03-31", 7777, "2025-04-10")],
        })
        monkeypatch.setattr(oh, "Company", _make_company(facts_obj))
        h = oh.fetch_os_history(813, as_of=date(2025, 3, 31))
        assert h.concept == oh._OS_CONCEPTS[1]
        assert h.series[0].raw_shares == 7777.0

    def test_cutoff_10_years_excludes_older(self, temp_db, no_identity,
                                            monkeypatch):
        # as_of 2025-12-31 -> cutoff date(2015, 12, 1). A fact at 2015-09-30
        # is before cutoff (excluded); 2025-12-31 is in window.
        facts_obj = FakeFacts({oh._OS_CONCEPTS[0]: [
            _fact("2015-09-30", 100, "2015-10-10"),
            _fact("2025-12-31", 500, "2026-01-10"),
        ]})
        monkeypatch.setattr(oh, "Company", _make_company(facts_obj))
        h = oh.fetch_os_history(814, as_of=date(2025, 12, 31))
        assert [p.quarter_end for p in h.series] == [date(2025, 12, 31)]

    def test_cutoff_boundary_fact_exactly_at_cutoff_kept(self, temp_db,
                                                         no_identity, monkeypatch):
        # cutoff = date(as_of.year-10, as_of.month, 1). With as_of 2025-12-31
        # the cutoff is 2015-12-01; a fact dated EXACTLY on the cutoff is kept
        # (the gate is `end < cutoff`, strict). It buckets to Q4-2015.
        facts_obj = FakeFacts({oh._OS_CONCEPTS[0]: [
            _fact("2015-12-01", 100, "2015-12-15"),
            _fact("2025-12-31", 500, "2026-01-10"),
        ]})
        monkeypatch.setattr(oh, "Company", _make_company(facts_obj))
        h = oh.fetch_os_history(815, as_of=date(2025, 12, 31))
        # first real point is the cutoff-quarter; carry-forward then fills
        # every quarter through Q4-2025, so the series spans 2015Q4..2025Q4.
        assert h.series[0].quarter_end == date(2015, 12, 31)
        assert h.series[0].raw_shares == 100.0
        assert h.series[-1].quarter_end == date(2025, 12, 31)
        assert h.series[-1].raw_shares == 500.0

    def test_fact_after_as_of_excluded(self, temp_db, no_identity, monkeypatch):
        # the window gate is `end < cutoff or end > as_of`: a fact whose
        # period_end lands AFTER as_of is dropped (the upper half of the gate,
        # which the older cutoff test never exercised).
        facts_obj = FakeFacts({oh._OS_CONCEPTS[0]: [
            _fact("2025-03-31", 1000, "2025-04-10"),
            _fact("2026-09-30", 5000, "2026-10-10"),  # after as_of -> dropped
        ]})
        monkeypatch.setattr(oh, "Company", _make_company(facts_obj))
        h = oh.fetch_os_history(816, as_of=date(2025, 6, 30))
        # only Q1 is real; Q2 is carried from Q1; the future point never lands.
        assert [(p.quarter_end, p.raw_shares, p.carried) for p in h.series] == [
            (date(2025, 3, 31), 1000.0, False),
            (date(2025, 6, 30), 1000.0, True),
        ]

    def test_unity_split_factor_leaves_split_adjusted_false(self, temp_db,
                                                            no_identity, monkeypatch):
        # A 1-for-1 split (pre=1, post=1) effective AFTER the cover date is a
        # no-op factor of 1.0; _adjustment stays 1.0, so split_adjusted is
        # False even though a dilution_splits row exists and the shares are
        # numerically unchanged. Boundary of the `adj != 1.0` flag.
        temp_db.add_split(817, effective_date="2025-06-01", pre=1, post=1)
        facts_obj = FakeFacts({oh._OS_CONCEPTS[0]: [
            _fact("2025-03-31", 1000, "2025-04-10"),
        ]})
        monkeypatch.setattr(oh, "Company", _make_company(facts_obj))
        h = oh.fetch_os_history(817, as_of=date(2025, 3, 31))
        assert h.series[0].shares == pytest.approx(1000.0)
        assert h.series[0].split_adjusted is False

    def test_split_before_cover_date_not_applied(self, temp_db, no_identity,
                                                 monkeypatch):
        # A split effective ON-OR-BEFORE the cover date is NOT applied
        # (_adjustment uses strict eff > d). The point is unadjusted.
        temp_db.add_split(818, effective_date="2025-03-31", pre=250, post=1)
        facts_obj = FakeFacts({oh._OS_CONCEPTS[0]: [
            _fact("2025-03-31", 1000, "2025-04-10"),
        ]})
        monkeypatch.setattr(oh, "Company", _make_company(facts_obj))
        h = oh.fetch_os_history(818, as_of=date(2025, 3, 31))
        assert h.series[0].shares == pytest.approx(1000.0)
        assert h.series[0].split_adjusted is False

    def test_dedup_collision_none_filing_date_does_not_crash(self, temp_db,
                                                             no_identity, monkeypatch):
        # FIXED (was bug A#3): when two facts share a period_end and the
        # later-iterated one has a None filing_date, it is treated as oldest
        # and the dated restatement is kept, instead of crashing on
        # `None > date(...)`.
        facts_obj = FakeFacts({oh._OS_CONCEPTS[0]: [
            _fact("2025-03-31", 1000, "2025-04-10"),
            _fact("2025-03-31", 1100, None),
        ]})
        monkeypatch.setattr(oh, "Company", _make_company(facts_obj))
        h = oh.fetch_os_history(820, as_of=date(2025, 3, 31))
        # the dated restatement (1000 @ 2025-04-10) should be retained over
        # the undated 1100; either way the call must not raise.
        assert len(h.series) == 1


# ── fetch_os_history_cached (io_mockable) ────────────────────────────
class TestFetchOsHistoryCached:
    def test_second_identical_call_is_cached(self, monkeypatch):
        counter = {"n": 0}

        def _fake_fetch(cik, *, as_of=None):
            counter["n"] += 1
            return oh.OsHistory(as_of=as_of)

        monkeypatch.setattr(oh, "fetch_os_history", _fake_fetch)
        a = oh.fetch_os_history_cached(900, as_of=date(2025, 1, 1))
        b = oh.fetch_os_history_cached(900, as_of=date(2025, 1, 1))
        assert counter["n"] == 1
        assert a is b

    def test_different_as_of_separate_entries(self, monkeypatch):
        counter = {"n": 0}

        def _fake_fetch(cik, *, as_of=None):
            counter["n"] += 1
            return oh.OsHistory(as_of=as_of)

        monkeypatch.setattr(oh, "fetch_os_history", _fake_fetch)
        oh.fetch_os_history_cached(901, as_of=date(2025, 1, 1))
        oh.fetch_os_history_cached(901, as_of=date(2025, 2, 1))
        assert counter["n"] == 2

    def test_as_of_none_defaults_to_today(self, monkeypatch):
        seen = {}

        def _fake_fetch(cik, *, as_of=None):
            seen["as_of"] = as_of
            return oh.OsHistory(as_of=as_of)

        monkeypatch.setattr(oh, "fetch_os_history", _fake_fetch)
        before = date.today()
        oh.fetch_os_history_cached(902)
        after = date.today()
        # _cached resolves None -> today().isoformat() before the call, so the
        # downstream as_of is a real date (today). Bracket the call to stay
        # robust across a midnight rollover instead of calling today() again.
        assert isinstance(seen["as_of"], date)
        assert before <= seen["as_of"] <= after


# ── _ensure_identity (io_mockable) ───────────────────────────────────
class TestEnsureIdentity:
    def test_called_once_then_noop(self, monkeypatch):
        calls = []
        monkeypatch.setattr(oh, "set_identity",
                            lambda ident: calls.append(ident))
        monkeypatch.setattr(oh, "_IDENTITY_SET", False, raising=False)
        oh._ensure_identity()
        oh._ensure_identity()
        assert len(calls) == 1
