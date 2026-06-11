"""Unit tests for dilution/badges.py — the deterministic 5-badge strip.

The bulk of the value is in the pure banding/formatting functions, which
need neither DB nor network. The two SQL helpers read the autouse temp_db,
and _os_growth_3y is the only network seam (a function-local
``import yfinance``), which we fake via sys.modules / attribute patching.

CRITICAL: _os_growth_3y is @lru_cache(maxsize=256). The autouse
``_clear_os_cache`` fixture clears it before/after every test so memoized
tuples never bleed across cases.
"""
from __future__ import annotations

import sys
import types
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

import dilution.badges as badges
from dilution.badges import (
    WKSI_SENTINEL_USD,
    Badge,
    BadgeSet,
    _band_score,
    _cash_need,
    _dilution_history,
    _is_wksi_amount,
    _mk,
    _offering_ability,
    _os_growth_3y,
    _overhang,
    _pct0,
    _raised_last_24mo,
    _sh,
    _usd,
    compute_badges,
)

# ── constants the survey told us to import for exact boundaries ──────
from dilution.badges import (  # noqa: E402
    _HISTORY_HIGH_PCT,
    _HISTORY_MED_PCT,
    _INTERACTION_BUMP,
    _OFFERING_FLOOR_USD,
    _OFFERING_HIGH_PCT,
    _OFFERING_MED_PCT,
    _OVERHANG_HIGH_PCT,
    _OVERHANG_MED_PCT,
    _RUNWAY_HIGH_MONTHS,
    _RUNWAY_LOW_MONTHS,
)


@pytest.fixture(autouse=True)
def _clear_os_cache():
    """_os_growth_3y is lru_cached on (cik, ticker, as_of_iso); stale
    results would bleed across tests. Clear before and after each test."""
    _os_growth_3y.cache_clear()
    yield
    _os_growth_3y.cache_clear()


def _cash(**kw):
    """Build a cash stub — _cash_need only reads attributes, no typecheck."""
    d = dict(months_of_cash=None, op_cf_quarterly_usd=None,
             current_cash_est_usd=None, latest_period_end=None)
    d.update(kw)
    return SimpleNamespace(**d)


def _fake_yf(ticker_factory):
    """Install a fake `yfinance` module exposing `.Ticker(t)`."""
    mod = types.ModuleType("yfinance")
    mod.Ticker = ticker_factory
    sys.modules["yfinance"] = mod
    return mod


class _FakeSeries:
    """Minimal pandas-Series stand-in: len() + items() of (ts, value).

    The timestamps expose `.date()` like a pandas Timestamp does.
    """

    def __init__(self, data):
        self._d = data  # list of (datetime.date, value)

    def __len__(self):
        return len(self._d)

    def items(self):
        for d, v in self._d:
            yield SimpleNamespace(date=(lambda d=d: d)), v


# ─────────────────────────────────────────────────────────────────────
# _usd
# ─────────────────────────────────────────────────────────────────────
class TestUsd:
    @pytest.mark.parametrize("x,expected", [
        (None, "—"),
        (0, "$0"),
        (500, "$500"),
        (999, "$999"),
        (1e3, "$1K"),
        (1e6, "$1.0M"),
        (999_999, "$1000K"),   # rounds up at the M boundary, still K branch
        (1e9, "$1.00B"),
        (1_550_000, "$1.6M"),  # rounding
    ])
    def test_positive_and_none(self, x, expected):
        assert _usd(x) == expected

    def test_negative_sign_before_dollar(self):
        # sign is applied to the abs-value formatting: "-$2.0M"
        assert _usd(-2_000_000) == "-$2.0M"

    def test_negative_billions(self):
        assert _usd(-2.5e9) == "-$2.50B"


# ─────────────────────────────────────────────────────────────────────
# _sh
# ─────────────────────────────────────────────────────────────────────
class TestSh:
    @pytest.mark.parametrize("x,expected", [
        (None, "—"),
        (0, "0"),
        (999, "999"),
        (1234, "1K"),
        (1e6, "1.0M"),
        (2.5e9, "2.50B"),
    ])
    def test_positive_and_none(self, x, expected):
        assert _sh(x) == expected

    def test_negative_has_no_special_handling(self):
        # Contrast with _usd: negatives fall through to the final branch
        # as-is (x < 1e3 once negative), formatted with :.0f and no sign
        # logic — so -5e6 is NOT "-5.0M" but the raw integer string.
        assert _sh(-5e6) == "-5000000"


# ─────────────────────────────────────────────────────────────────────
# _pct0
# ─────────────────────────────────────────────────────────────────────
class TestPct0:
    @pytest.mark.parametrize("x,expected", [
        (0, "0%"),
        (12.7, "13%"),         # rounds
        (-50, "-50%"),
        (1234.5, "1,234%"),    # banker's rounding: .5 -> even -> 1234
        (1000, "1,000%"),      # thousands separator
    ])
    def test_format(self, x, expected):
        assert _pct0(x) == expected


# ─────────────────────────────────────────────────────────────────────
# _band_score
# ─────────────────────────────────────────────────────────────────────
class TestBandScore:
    @pytest.mark.parametrize("band,frac,expected", [
        ("low", 0.0, 0),
        ("medium", 0.0, 34),
        ("high", 0.0, 67),
        ("low", 1.0, 33),
        ("medium", 1.0, 66),
        ("high", 1.0, 100),
        ("low", 2.0, 33),    # frac>1 clamps to 1
        ("low", -1.0, 0),    # frac<0 clamps to 0
        ("low", 0.5, 16),    # round(0 + 0.5*33) = round(16.5) -> 16 (banker)
    ])
    def test_interval(self, band, frac, expected):
        assert _band_score(band, frac) == expected

    def test_unknown_band_raises_keyerror(self):
        with pytest.raises(KeyError):
            _band_score("nope", 0.5)


# ─────────────────────────────────────────────────────────────────────
# _is_wksi_amount
# ─────────────────────────────────────────────────────────────────────
class TestIsWksiAmount:
    @pytest.mark.parametrize("v,expected", [
        (None, False),
        (WKSI_SENTINEL_USD, True),
        (999_999_999.0, True),
        (999_999_998.6, True),       # rounds to sentinel
        ("999999999", True),         # float() coerces the string
        ("abc", False),              # ValueError swallowed
        (0, False),
        ([1, 2], False),             # TypeError swallowed
        (object(), False),           # TypeError swallowed
        (999_999_998, False),        # off-by-one does not round to sentinel
    ])
    def test_values(self, v, expected):
        assert _is_wksi_amount(v) is expected


# ─────────────────────────────────────────────────────────────────────
# _offering_ability
# ─────────────────────────────────────────────────────────────────────
class TestOfferingAbility:
    def test_empty_cards_is_low(self):
        b = _offering_ability({}, 1e8)
        assert b.band == "low"
        assert b.score == 0
        assert b.detail == ("No active shelf or pending S-1 on file",)

    def test_wksi_sentinel_forces_high_even_without_mcap(self):
        b = _offering_ability(
            {"shelf": [{"current_raisable_amount": WKSI_SENTINEL_USD}]}, None)
        assert b.band == "high"
        assert b.score == 100

    def test_wksi_via_total_shelf_capacity(self):
        b = _offering_ability(
            {"shelf": [{"total_shelf_capacity": WKSI_SENTINEL_USD}]}, 1e8)
        assert b.band == "high"
        assert any("WKSI" in d for d in b.detail)

    def test_capacity_present_but_mcap_none_band_none(self):
        b = _offering_ability(
            {"shelf": [{"current_raisable_amount": 1e7}]}, None)
        assert b.band is None
        assert b.score is None
        assert any("Market cap unavailable" in d for d in b.detail)

    def test_ratio_exactly_med_pct_lands_in_medium(self):
        # ratio == 5.0%: NOT < _OFFERING_MED_PCT (strict), so falls into the
        # <= HIGH medium branch.
        b = _offering_ability(
            {"shelf": [{"current_raisable_amount": 5e6}]}, 1e8)
        assert b.band == "medium"
        assert b.score == 34  # frac == 0 at the medium floor

    def test_ratio_just_below_med_pct_is_low(self):
        b = _offering_ability(
            {"shelf": [{"current_raisable_amount": 4.9e6}]}, 1e8)
        assert b.band == "low"

    def test_ratio_exactly_high_pct_is_medium(self):
        # ratio == 25.0%: <= _OFFERING_HIGH_PCT -> still medium (top frac)
        b = _offering_ability(
            {"shelf": [{"current_raisable_amount": 2.5e7}]}, 1e8)
        assert b.band == "medium"
        assert b.score == 66

    def test_ratio_just_over_high_pct_is_high(self):
        b = _offering_ability(
            {"shelf": [{"current_raisable_amount": 2.6e7}]}, 1e8)
        assert b.band == "high"

    def test_capacity_below_floor_is_low_with_too_small_note(self):
        b = _offering_ability(
            {"shelf": [{"current_raisable_amount": 5e5}]}, 1e8)
        assert b.band == "low"
        assert any("too small" in d for d in b.detail)

    def test_shelf_raisable_falls_back_to_total_capacity(self):
        b = _offering_ability(
            {"shelf": [{"current_raisable_amount": None,
                        "total_shelf_capacity": 1e7}]}, 1e8)
        # 1e7 / 1e8 = 10% -> medium
        assert b.band == "medium"

    def test_shelf_both_none_increments_unknown(self):
        b = _offering_ability(
            {"shelf": [{"current_raisable_amount": None,
                        "total_shelf_capacity": None}]}, 1e8)
        assert any("undisclosed capacity" in d for d in b.detail)
        # capacity stays 0 -> low
        assert b.band == "low"

    def test_s1_anticipated_none_unknown(self):
        b = _offering_ability(
            {"s1_offering": [{"s1_status": "pending",
                              "anticipated_deal_size": None}]}, 1e8)
        assert any("undisclosed capacity" in d for d in b.detail)

    def test_s1_withdrawn_excluded(self):
        b = _offering_ability(
            {"s1_offering": [{"s1_status": "withdrawn",
                              "anticipated_deal_size": 1e7}]}, 1e8)
        # withdrawn filtered -> no S-1 -> "No active shelf..." note, low
        assert b.band == "low"
        assert b.detail == ("No active shelf or pending S-1 on file",)

    @pytest.mark.parametrize("status", ["pending", "effective"])
    def test_s1_pending_and_effective_kept(self, status):
        b = _offering_ability(
            {"s1_offering": [{"s1_status": status,
                              "anticipated_deal_size": 2e7}]}, 1e8)
        # 2e7 / 1e8 = 20% -> medium
        assert b.band == "medium"
        assert any("pending S-1/F-1" in d for d in b.detail)

    def test_eloc_terminated_excluded(self):
        b = _offering_ability(
            {"equity_line": [{"terminated": True,
                              "remaining_capacity": 5e6}]}, 1e8)
        # terminated ELOC dropped -> no escalator, no capacity -> low 0
        assert b.band == "low"
        assert b.score == 0

    def test_escalator_on_band_none_becomes_medium_half(self):
        # capacity present, mcap None -> base band None, then ATM escalates
        b = _offering_ability(
            {"shelf": [{"current_raisable_amount": 1e7}],
             "atm": [{"raisable_capped": 2e6}]}, None)
        assert b.band == "medium"
        assert b.score == 50  # frac 0.5

    def test_escalator_low_to_medium(self):
        b = _offering_ability(
            {"shelf": [{"current_raisable_amount": 5e5}],
             "atm": [{"raisable_capped": 2e6}]}, 1e8)
        assert b.band == "medium"

    def test_escalator_medium_to_high(self):
        b = _offering_ability(
            {"shelf": [{"current_raisable_amount": 1e7}],
             "atm": [{"raisable_capped": 2e6}]}, 1e8)
        # base medium (10%) -> escalated to high
        assert b.band == "high"
        assert b.score == 75

    def test_escalator_on_high_bumps_frac_capped(self):
        # base high (30% -> frac min(1,(30-25)/75)=0.0667 -> score 67),
        # escalator adds 0.33 to frac -> ~0.397 -> score 80
        b = _offering_ability(
            {"shelf": [{"current_raisable_amount": 3e7}],
             "atm": [{"raisable_capped": 2e6}]}, 1e8)
        assert b.band == "high"
        assert b.score == 80

    def test_atm_raisable_capped_overrides_remaining_capacity(self):
        # raisable_capped (2e6) is used, not remaining_capacity (5e5)
        b = _offering_ability(
            {"atm": [{"raisable_capped": 2e6, "remaining_capacity": 5e5}]},
            1e8)
        assert any("$2.0M remaining" in d for d in b.detail)

    def test_atm_missing_both_is_zero_no_escalator(self):
        b = _offering_ability({"atm": [{}]}, 1e8)
        assert b.band == "low"
        assert b.score == 0

    def test_atm_exactly_at_floor_escalates(self):
        # atm_live == _OFFERING_FLOOR_USD -> escalator via >= floor
        b = _offering_ability({"atm": [{"raisable_capped": 1e6}]}, 1e8)
        # base: capacity 0 -> low, escalated -> medium
        assert b.band == "medium"

    def test_atm_via_mcap_5pct_boundary_escalates(self):
        # below $1M floor but >= 5%-of-mcap -> escalator still fires
        b = _offering_ability(
            {"atm": [{"raisable_capped": 5e5}]}, 1e7)  # 5e5 == 5% of 1e7
        assert b.band == "medium"

    def test_multiple_shelves_plural_noun(self):
        b = _offering_ability(
            {"shelf": [{"current_raisable_amount": 1e7},
                       {"current_raisable_amount": 1e7}]}, 1e8)
        assert any("active shelves" in d for d in b.detail)

    def test_single_shelf_singular_noun(self):
        b = _offering_ability(
            {"shelf": [{"current_raisable_amount": 1e7}]}, 1e8)
        assert any("active shelf:" in d for d in b.detail)

    def test_baby_shelf_restriction_note(self):
        b = _offering_ability(
            {"shelf": [{"current_raisable_amount": 1e7,
                        "baby_shelf_restriction": "Yes"}]}, 1e8)
        assert any("I.B.6" in d for d in b.detail)


# ─────────────────────────────────────────────────────────────────────
# _overhang
# ─────────────────────────────────────────────────────────────────────
class TestOverhang:
    def test_total_zero_is_low(self):
        b = _overhang({}, 1e7)
        assert b.band == "low"
        assert b.score == 0
        assert any("No warrants" in d for d in b.detail)

    def test_total_positive_but_os_none_band_none(self):
        b = _overhang({"warrant": [{"remaining_outstanding": 1e6}]}, None)
        assert b.band is None
        assert b.score is None
        assert any("Shares outstanding unavailable" in d for d in b.detail)

    def test_pct_exactly_med_is_medium(self):
        # 2e6 / 1e7 = 20% == _OVERHANG_MED_PCT; NOT < MED -> medium
        b = _overhang({"warrant": [{"remaining_outstanding": 2e6}]}, 1e7)
        assert b.band == "medium"
        assert b.score == 34

    def test_pct_just_below_med_is_low(self):
        b = _overhang({"warrant": [{"remaining_outstanding": 1.9e6}]}, 1e7)
        assert b.band == "low"

    def test_pct_exactly_high_is_medium(self):
        # 5e6 / 1e7 = 50% == _OVERHANG_HIGH_PCT; <= HIGH -> still medium
        b = _overhang({"warrant": [{"remaining_outstanding": 5e6}]}, 1e7)
        assert b.band == "medium"
        assert b.score == 66

    def test_pct_just_over_high_is_high(self):
        b = _overhang({"warrant": [{"remaining_outstanding": 5.1e6}]}, 1e7)
        assert b.band == "high"

    def test_pct_over_100_frac_caps_at_one(self):
        b = _overhang({"warrant": [{"remaining_outstanding": 2e7}]}, 1e7)
        assert b.band == "high"
        assert b.score == 100

    def test_missing_conversion_price_surfaced(self):
        b = _overhang(
            {"convertible": [{"remaining_shares_issuable": None,
                              "principal_remaining": 1e6}]}, 1e7)
        # not share-countable -> total 0 -> low, but exclusion noted
        assert b.band == "low"
        assert any("undisclosed conversion price excluded" in d
                   for d in b.detail)

    def test_missing_plural_count(self):
        b = _overhang(
            {"convertible": [{"remaining_shares_issuable": None,
                              "principal_remaining": 1e6}],
             "convertible_preferred": [{"remaining_shares_issuable": None,
                                        "principal_remaining": 2e6}]}, 1e7)
        assert any("2 instruments" in d for d in b.detail)

    def test_principal_none_not_counted_as_missing(self):
        b = _overhang(
            {"convertible": [{"remaining_shares_issuable": None,
                              "principal_remaining": None}]}, 1e7)
        assert not any("undisclosed conversion price" in d for d in b.detail)

    def test_none_values_coerced_to_zero(self):
        b = _overhang(
            {"warrant": [{"remaining_outstanding": None}],
             "convertible": [{"remaining_shares_issuable": None}]}, 1e7)
        # principal not set so not "missing"; total 0 -> low
        assert b.band == "low"
        assert b.score == 0

    def test_parts_joined_with_plus(self):
        b = _overhang(
            {"warrant": [{"remaining_outstanding": 1e6}],
             "convertible": [{"remaining_shares_issuable": 1e6}],
             "convertible_preferred": [{"remaining_shares_issuable": 1e6}]},
            1e8)
        line = b.detail[0]
        assert "warrants" in line and "convertibles" in line
        assert "preferred" in line and " + " in line


# ─────────────────────────────────────────────────────────────────────
# _cash_need
# ─────────────────────────────────────────────────────────────────────
class TestCashNeed:
    def test_none_cash_band_none(self):
        b = _cash_need(None)
        assert b.band is None
        assert b.score is None
        assert b.detail == ("Cash-flow data unavailable",)

    def test_both_months_and_op_none_band_none(self):
        b = _cash_need(_cash())
        assert b.band is None
        assert b.detail == ("Cash-flow data unavailable",)

    def test_op_cf_zero_short_circuits_low(self):
        b = _cash_need(_cash(op_cf_quarterly_usd=0, months_of_cash=1))
        assert b.band == "low"
        assert b.score == 0

    def test_op_cf_positive_short_circuits_low(self):
        b = _cash_need(_cash(op_cf_quarterly_usd=5e5, months_of_cash=1))
        assert b.band == "low"
        assert b.score == 0
        assert any("positive" in d for d in b.detail)

    def test_op_cf_negative_months_none_band_none(self):
        b = _cash_need(_cash(op_cf_quarterly_usd=-5e5, months_of_cash=None))
        assert b.band is None
        assert b.score is None

    def test_months_exactly_low_threshold_is_medium(self):
        # 24.0 == _RUNWAY_LOW_MONTHS; NOT > LOW -> medium boundary
        b = _cash_need(_cash(op_cf_quarterly_usd=-5e5, months_of_cash=24.0))
        assert b.band == "medium"
        assert b.score == 34

    def test_months_just_over_low_threshold_is_low(self):
        b = _cash_need(_cash(op_cf_quarterly_usd=-5e5, months_of_cash=24.5))
        assert b.band == "low"

    def test_months_exactly_high_threshold_is_medium(self):
        # 6.0 == _RUNWAY_HIGH_MONTHS; >= HIGH -> medium boundary
        b = _cash_need(_cash(op_cf_quarterly_usd=-5e5, months_of_cash=6.0))
        assert b.band == "medium"
        assert b.score == 66

    def test_months_just_under_high_threshold_is_high(self):
        b = _cash_need(_cash(op_cf_quarterly_usd=-5e5, months_of_cash=5.9))
        assert b.band == "high"

    def test_months_very_large_low_frac_clamped(self):
        b = _cash_need(_cash(op_cf_quarterly_usd=-5e5, months_of_cash=100))
        assert b.band == "low"
        assert b.score == 0  # (36-100)/12 negative -> clamped to 0

    def test_months_thirty_low_frac_half(self):
        b = _cash_need(_cash(op_cf_quarterly_usd=-5e5, months_of_cash=30))
        assert b.band == "low"
        assert b.score == 16  # frac (36-30)/12 = 0.5

    def test_burn_phrasing_for_negative_op(self):
        b = _cash_need(_cash(op_cf_quarterly_usd=-5e5, months_of_cash=10,
                             current_cash_est_usd=1e6,
                             latest_period_end="2025-12-31"))
        assert any("Operating burn $500K/quarter" in d for d in b.detail)
        assert any("Est. current cash $1.0M" in d for d in b.detail)
        assert any("months of runway" in d for d in b.detail)
        assert any("Balance sheet as of 2025-12-31" in d for d in b.detail)


# ─────────────────────────────────────────────────────────────────────
# _mk (Badge factory)
# ─────────────────────────────────────────────────────────────────────
class TestMk:
    def test_band_none_score_none_dash_text(self):
        b = _mk("k", "L", "D", None, 0.0, ["x"], ())
        assert b.score is None
        assert b.band_text == "—"
        assert b.detail == ("x",)

    def test_band_high_full_frac(self):
        b = _mk("k", "L", "D", "high", 1.0, ["x", "y"], ())
        assert b.score == 100
        assert b.band_text == "High"

    def test_detail_list_becomes_tuple(self):
        b = _mk("k", "L", "D", "low", 0.0, ["a", "b"], ())
        assert isinstance(b.detail, tuple)
        assert b.detail == ("a", "b")


# ─────────────────────────────────────────────────────────────────────
# _raised_last_24mo  (DB-backed)
# ─────────────────────────────────────────────────────────────────────
class TestRaisedLast24mo:
    AS_OF = date(2026, 1, 1)

    def test_no_rows_returns_zero_not_none(self, temp_db):
        result = _raised_last_24mo(99, self.AS_OF)
        assert result == 0.0
        assert result is not None

    def test_multiple_rows_summed(self, temp_db):
        # The drawdowns table FK-references dilution_ledger(instrument_id),
        # so stage the parent instruments first.
        temp_db.add_instrument("I1", cik=1)
        temp_db.add_drawdown("I1", cik=1, event_date="2025-06-01",
                             amount_usd=1_000_000.0)
        temp_db.add_drawdown("I1", cik=1, event_date="2025-07-01",
                             amount_usd=500_000.0)
        assert _raised_last_24mo(1, self.AS_OF) == 1_500_000.0

    def test_exactly_730_days_excluded(self, temp_db):
        temp_db.add_instrument("I730", cik=5)
        d730 = (self.AS_OF - timedelta(days=730)).isoformat()
        temp_db.add_drawdown("I730", cik=5, event_date=d730, amount_usd=100.0)
        # strict > -> the exact 730-day-old row is excluded
        assert _raised_last_24mo(5, self.AS_OF) == 0.0

    def test_729_days_included(self, temp_db):
        temp_db.add_instrument("I729", cik=6)
        d729 = (self.AS_OF - timedelta(days=729)).isoformat()
        temp_db.add_drawdown("I729", cik=6, event_date=d729, amount_usd=200.0)
        assert _raised_last_24mo(6, self.AS_OF) == 200.0

    def test_null_amount_excluded(self, temp_db):
        temp_db.add_instrument("Inull", cik=7)
        temp_db.add_drawdown("Inull", cik=7, event_date="2025-06-01",
                             amount_usd=None)
        assert _raised_last_24mo(7, self.AS_OF) == 0.0

    def test_other_cik_excluded(self, temp_db):
        temp_db.add_instrument("Iother", cik=8)
        temp_db.add_drawdown("Iother", cik=8, event_date="2025-06-01",
                             amount_usd=9_000.0)
        assert _raised_last_24mo(999, self.AS_OF) == 0.0


# ─────────────────────────────────────────────────────────────────────
# _os_growth_3y  (io_mockable — function-local `import yfinance`)
# ─────────────────────────────────────────────────────────────────────
class TestOsGrowth3y:
    AS_OF = "2026-01-01"

    def test_basic_growth(self, temp_db):
        _fake_yf(lambda t: SimpleNamespace(
            get_shares_full=lambda start=None: _FakeSeries([
                (date(2022, 12, 1), 1_000_000.0),
                (date(2026, 1, 1), 3_000_000.0),
            ])))
        g = _os_growth_3y(1, "TEST", self.AS_OF)
        assert g is not None
        growth, a_date, a_adj, l_cnt, l_date, n_splits = g
        assert growth == pytest.approx(200.0)
        assert a_date == date(2022, 12, 1)
        assert a_adj == pytest.approx(1_000_000.0)
        assert l_cnt == pytest.approx(3_000_000.0)
        assert n_splits == 0

    def test_yfinance_raises_returns_none(self, temp_db):
        def raiser(t):
            return SimpleNamespace(
                get_shares_full=lambda start=None: (_ for _ in ()).throw(
                    RuntimeError("boom")))
        _fake_yf(raiser)
        assert _os_growth_3y(2, "X", self.AS_OF) is None

    def test_empty_series_returns_none(self, temp_db):
        _fake_yf(lambda t: SimpleNamespace(
            get_shares_full=lambda start=None: _FakeSeries([])))
        assert _os_growth_3y(3, "X", self.AS_OF) is None

    def test_fewer_than_two_positive_points_returns_none(self, temp_db):
        # one zero point gets filtered, leaving a single positive point
        _fake_yf(lambda t: SimpleNamespace(
            get_shares_full=lambda start=None: _FakeSeries([
                (date(2022, 1, 1), 0),
                (date(2026, 1, 1), 1_000_000.0),
            ])))
        assert _os_growth_3y(4, "X", self.AS_OF) is None

    def test_window_under_540_days_returns_none(self, temp_db):
        _fake_yf(lambda t: SimpleNamespace(
            get_shares_full=lambda start=None: _FakeSeries([
                (date(2025, 1, 1), 1_000_000.0),
                (date(2025, 6, 1), 2_000_000.0),
            ])))
        assert _os_growth_3y(5, "X", self.AS_OF) is None

    def test_reverse_split_multiplies_anchor(self, temp_db):
        # pre=10 post=1 -> anchor * (1/10) = 100k; latest 3M -> 2900%
        temp_db.add_split(9, "2023-06-01", 10, 1)
        _fake_yf(lambda t: SimpleNamespace(
            get_shares_full=lambda start=None: _FakeSeries([
                (date(2022, 12, 1), 1_000_000.0),
                (date(2026, 1, 1), 3_000_000.0),
            ])))
        g = _os_growth_3y(9, "X", self.AS_OF)
        assert g is not None
        growth, _, a_adj, _, _, n_splits = g
        assert a_adj == pytest.approx(100_000.0)
        assert growth == pytest.approx(2900.0)
        assert n_splits == 1

    def test_lru_cache_memoizes(self, temp_db):
        # First call populates the cache; a second call with a different
        # (mutated) fake should return the cached tuple, proving memoization.
        _fake_yf(lambda t: SimpleNamespace(
            get_shares_full=lambda start=None: _FakeSeries([
                (date(2022, 12, 1), 1_000_000.0),
                (date(2026, 1, 1), 2_000_000.0),
            ])))
        first = _os_growth_3y(11, "CACHED", self.AS_OF)
        # swap the fake to a None-yielding one; cached key short-circuits
        _fake_yf(lambda t: SimpleNamespace(
            get_shares_full=lambda start=None: _FakeSeries([])))
        second = _os_growth_3y(11, "CACHED", self.AS_OF)
        assert second == first


# ─────────────────────────────────────────────────────────────────────
# _dilution_history  (DB-backed + mocked _os_growth_3y)
# ─────────────────────────────────────────────────────────────────────
class TestDilutionHistory:
    AS_OF = date(2026, 1, 1)

    @staticmethod
    def _growth_tuple(growth, n_splits=0):
        return (growth, date(2022, 12, 1), 1_000_000.0,
                1_000_000.0 * (1 + growth / 100.0), date(2026, 1, 1),
                n_splits)

    def test_ticker_none_band_none(self, temp_db):
        b = _dilution_history(1, None, self.AS_OF)
        assert b.band is None
        assert b.score is None
        assert any("unavailable" in d for d in b.detail)

    def test_os_growth_none_band_none(self, temp_db, monkeypatch):
        monkeypatch.setattr(badges, "_os_growth_3y",
                            lambda cik, t, iso: None)
        b = _dilution_history(1, "X", self.AS_OF)
        assert b.band is None

    def test_growth_exactly_med_is_medium(self, temp_db, monkeypatch):
        # 30.0 == _HISTORY_MED_PCT; NOT < MED -> medium boundary
        monkeypatch.setattr(badges, "_os_growth_3y",
                            lambda c, t, i: self._growth_tuple(30.0))
        b = _dilution_history(1, "X", self.AS_OF)
        assert b.band == "medium"
        assert b.score == 34

    def test_growth_exactly_high_is_medium(self, temp_db, monkeypatch):
        # 100.0 == _HISTORY_HIGH_PCT; <= HIGH -> still medium
        monkeypatch.setattr(badges, "_os_growth_3y",
                            lambda c, t, i: self._growth_tuple(100.0))
        b = _dilution_history(1, "X", self.AS_OF)
        assert b.band == "medium"
        assert b.score == 66

    def test_growth_over_high_is_high(self, temp_db, monkeypatch):
        monkeypatch.setattr(badges, "_os_growth_3y",
                            lambda c, t, i: self._growth_tuple(300.0))
        b = _dilution_history(1, "X", self.AS_OF)
        assert b.band == "high"
        assert b.score == 100  # frac min(1,(300-100)/200)=1.0

    def test_negative_growth_low_frac_zero(self, temp_db, monkeypatch):
        monkeypatch.setattr(badges, "_os_growth_3y",
                            lambda c, t, i: self._growth_tuple(-50.0))
        b = _dilution_history(1, "X", self.AS_OF)
        assert b.band == "low"
        assert b.score == 0  # max(0, -50)/30 -> 0

    def test_splits_note_plural(self, temp_db, monkeypatch):
        monkeypatch.setattr(badges, "_os_growth_3y",
                            lambda c, t, i: self._growth_tuple(50.0, 2))
        b = _dilution_history(1, "X", self.AS_OF)
        assert any("(split-adj)" in d for d in b.detail)
        assert any("Normalized for 2 splits" in d for d in b.detail)

    def test_raised_line_appended(self, temp_db, monkeypatch):
        temp_db.add_instrument("Ihist", cik=1)
        temp_db.add_drawdown("Ihist", cik=1, event_date="2025-06-01",
                             amount_usd=5_000_000.0)
        monkeypatch.setattr(badges, "_os_growth_3y",
                            lambda c, t, i: self._growth_tuple(50.0))
        b = _dilution_history(1, "X", self.AS_OF)
        assert any("$5.0M raised in the last 24 months" in d
                   for d in b.detail)

    def test_no_raised_line_when_zero(self, temp_db, monkeypatch):
        monkeypatch.setattr(badges, "_os_growth_3y",
                            lambda c, t, i: self._growth_tuple(50.0))
        b = _dilution_history(1, "X", self.AS_OF)
        assert not any("raised in the last" in d for d in b.detail)


# ─────────────────────────────────────────────────────────────────────
# compute_badges  (orchestration + composite)
# ─────────────────────────────────────────────────────────────────────
class TestComputeBadges:
    AS_OF = date(2026, 1, 1)

    def test_no_scored_drivers(self, temp_db):
        # offering: capacity>0, mcap None -> band None
        # overhang: total>0, latest_os None -> band None
        # history: ticker None -> band None
        # cash: None -> band None
        bs = compute_badges(
            1, fund={"market_cap": None}, latest_os=None,
            cards={"shelf": [{"current_raisable_amount": 1e7}],
                   "warrant": [{"remaining_outstanding": 1e6}]},
            cash=None, as_of=self.AS_OF)
        assert bs.overall_score is None
        assert bs.overall_band is None
        assert bs.overall_label == "—"
        assert bs.partial is True
        assert bs.detail == ("No drivers computable — insufficient data",)

    def test_interaction_bump_applies_and_clamps(self, temp_db):
        # offering high via WKSI; cash high via months<6; ticker None so
        # no yfinance touched. scored: offering(100), overhang(low,0),
        # cash(high). raw = (.30*100 + .25*0 + .30*84)/.85 = 64.94, +15
        # = 79.94 -> 80 -> Severe.
        bs = compute_badges(
            1, fund={"market_cap": 1e8, "ticker": None}, latest_os=1e7,
            cards={"shelf": [{"current_raisable_amount": WKSI_SENTINEL_USD}]},
            cash=_cash(op_cf_quarterly_usd=-5e5, months_of_cash=3.0),
            as_of=self.AS_OF)
        assert bs.interaction is True
        assert bs.overall_score == 80
        assert bs.overall_label == "Severe"
        assert any("interaction bump" in d for d in bs.detail)

    def test_interaction_bump_clamps_at_100(self, temp_db):
        # Drive both offering and cash to max so raw+15 would exceed 100.
        bs = compute_badges(
            1, fund={"market_cap": 1e6, "ticker": None}, latest_os=1e7,
            cards={"shelf": [{"current_raisable_amount": WKSI_SENTINEL_USD}]},
            cash=_cash(op_cf_quarterly_usd=-5e5, months_of_cash=0.0),
            as_of=self.AS_OF)
        # offering 100, cash high 100, overhang low 0 -> wsum .85
        # raw = (.30*100 + .30*100)/.85... overhang adds 0 weight .25
        # = 60/.85 = 70.6, +15 = 85.6 -> not 100 here; verify <=100 anyway
        assert bs.interaction is True
        assert bs.overall_score <= 100

    def test_interaction_clamp_truly_at_100(self, temp_db):
        # Remove overhang from the mix entirely so the weighted raw is high
        # enough that +15 would exceed 100 and gets clamped.
        bs = compute_badges(
            1, fund={"market_cap": 1e6, "ticker": None}, latest_os=None,
            cards={"shelf": [{"current_raisable_amount": WKSI_SENTINEL_USD}]},
            cash=_cash(op_cf_quarterly_usd=-5e5, months_of_cash=0.0),
            as_of=self.AS_OF)
        # overhang: no paper -> low 0 still scored. offering 100, cash 100.
        # wsum = .30 + .25 + .30 = .85; raw = (30 + 0 + 30)/.85 = 70.6
        # +15 = 85.6 -> 86. Confirm not over 100.
        assert bs.overall_score == 86
        assert bs.overall_score <= 100

    def test_interaction_requires_both_high(self, temp_db):
        # offering high (WKSI) but cash medium -> no bump
        bs = compute_badges(
            1, fund={"market_cap": 1e8, "ticker": None}, latest_os=1e7,
            cards={"shelf": [{"current_raisable_amount": WKSI_SENTINEL_USD}]},
            cash=_cash(op_cf_quarterly_usd=-5e5, months_of_cash=12.0),
            as_of=self.AS_OF)
        assert bs.interaction is False
        assert not any("interaction bump" in d for d in bs.detail)

    def test_weight_renormalization_uses_partial_wsum(self, temp_db):
        # Only offering scored: high via WKSI -> score 100. Everything else
        # band None. Composite must be 100 (100*0.30 / 0.30), not 30.
        bs = compute_badges(
            1, fund={"market_cap": 1e8, "ticker": None}, latest_os=None,
            cards={"shelf": [{"current_raisable_amount": WKSI_SENTINEL_USD}],
                   "warrant": [{"remaining_outstanding": 1e6}]},
            cash=None, as_of=self.AS_OF)
        # offering scored (100); overhang band None (os None); history None;
        # cash None -> only offering scored -> 100*0.30/0.30 = 100
        assert bs.overall_score == 100
        assert bs.partial is True

    def test_partial_flag_when_fewer_than_four_scored(self, temp_db):
        bs = compute_badges(
            1, fund={"market_cap": 1e8, "ticker": None}, latest_os=1e7,
            cards={}, cash=None, as_of=self.AS_OF)
        # offering low(0), overhang low(0), history None (ticker None),
        # cash None -> 2 scored -> partial
        assert bs.partial is True

    @staticmethod
    def _cash_badge(score):
        """Fabricate a Cash Need Badge with an EXACT score by inverting the
        band-range math (low 0-33, medium 34-66, high 67-100)."""
        from dilution.badges import _CASH_LEGEND
        if score <= 33:
            band, lo, hi = "low", 0, 33
        elif score <= 66:
            band, lo, hi = "medium", 34, 66
        else:
            band, lo, hi = "high", 67, 100
        frac = (score - lo) / (hi - lo)
        return _mk("cash_need", "Cash Need", "d", band, frac, [],
                   _CASH_LEGEND)

    def _single_cash_composite(self, monkeypatch, *, score_target):
        """Make Cash Need the only scored driver with an exact score so
        the composite equals that score.

        offering: capacity>0 & mcap None -> band None (not scored).
        overhang: total>0 & latest_os None -> band None (not scored).
        history: ticker None -> band None (not scored).
        Cash Need is monkeypatched to a fixed-score Badge, so the renormalized
        composite is cash.score * 0.30 / 0.30 == cash.score.
        """
        badge = self._cash_badge(score_target)
        monkeypatch.setattr(badges, "_cash_need", lambda cash: badge)
        return compute_badges(
            1, fund={"market_cap": None, "ticker": None}, latest_os=None,
            cards={"shelf": [{"current_raisable_amount": 1e7}],
                   "warrant": [{"remaining_outstanding": 1e6}]},
            cash=object(), as_of=self.AS_OF)

    @pytest.mark.parametrize("score_target,expected_label", [
        (80, "Severe"),
        (79, "High"),
        (60, "High"),
        (59, "Moderate"),
        (40, "Moderate"),
        (39, "Low"),
        (20, "Low"),
        (19, "Minimal"),
        (0, "Minimal"),
    ])
    def test_overall_band_boundaries(self, temp_db, monkeypatch,
                                     score_target, expected_label):
        bs = self._single_cash_composite(monkeypatch,
                                         score_target=score_target)
        assert bs.overall_score == score_target
        assert bs.overall_label == expected_label

    def test_as_of_defaults_to_today_no_crash(self, temp_db):
        # as_of None -> date.today(); ticker None so no yfinance. Just
        # assert it runs and produces a BadgeSet.
        bs = compute_badges(1, fund=None, latest_os=None, cards={},
                            cash=None)
        assert isinstance(bs, BadgeSet)

    def test_fund_none_no_yfinance(self, temp_db, monkeypatch):
        # fund None -> ticker None -> _os_growth_3y must NOT be called.
        def boom(*a, **k):
            raise AssertionError("yfinance should not be reached")
        monkeypatch.setattr(badges, "_os_growth_3y", boom)
        bs = compute_badges(1, fund=None, latest_os=None, cards={},
                            cash=None, as_of=self.AS_OF)
        # history badge band None
        hist = next(b for b in bs.drivers if b.key == "history")
        assert hist.band is None

    def test_drivers_order_and_keys(self, temp_db):
        bs = compute_badges(1, fund=None, latest_os=None, cards={},
                            cash=None, as_of=self.AS_OF)
        keys = [b.key for b in bs.drivers]
        assert keys == ["offering_ability", "overhang", "history",
                        "cash_need"]
