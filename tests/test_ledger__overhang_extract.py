"""Unit tests for dilution/ledger/_overhang_extract.py.

Covers the pure normalization helpers (_num / _scale_factor / _scaled /
_apply_ads_normalization / _form_family / all six _clean_*_row), the
truncated-JSON salvage machinery (_salvage_truncated_json / _row_count),
the prompt-assembly helpers (_type_instruction_block /
_build_combined_prompt), the io_mockable parser (_parse_overhang_response,
driven with a SimpleNamespace fake response — no network), the db_backed
loader (_load_filing_text against the autouse temp_db), and the pydantic
row/list schemas.

The async orchestration glue (extract_overhang_rows /
_extract_per_specialist_lists) wraps the real LLM seam and is out of scope
per the survey slice — its deterministic substance is exercised by the
pure helper tests here.
"""

from __future__ import annotations

import types

import pytest
from pydantic import ValidationError

from conftest import response_stub
from dilution.ledger import _overhang_extract as m


# ════════════════════════════════════════════════════════════════════
# _num
# ════════════════════════════════════════════════════════════════════
class TestNum:
    def test_none_maps_to_none(self):
        assert m._num(None) is None

    def test_empty_string_maps_to_none(self):
        assert m._num("") is None

    def test_comma_stripping(self):
        assert m._num("1,234,567") == 1234567.0

    def test_dollar_and_comma_stripping(self):
        assert m._num("$30,000") == 30000.0

    def test_already_float_passthrough(self):
        assert m._num(12.5) == 12.5

    def test_unparseable_string_swallowed_to_none(self):
        assert m._num("abc") is None

    @pytest.mark.parametrize("val", ["0", 0, 0.0])
    def test_zero_is_zero_not_none(self, val):
        # Only None and '' map to None; falsy 0/'0' must coerce to 0.0.
        assert m._num(val) == 0.0
        assert m._num(val) is not None

    def test_negative_string(self):
        assert m._num("-5") == -5.0

    def test_negative_dollar_string(self):
        # Both '$' and the leading '-' survive the strip; float('-5') == -5.0.
        assert m._num("$-5") == -5.0

    def test_scientific_notation(self):
        assert m._num("1.5e3") == 1500.0

    def test_whitespace_only_is_none(self):
        # float('   ') raises ValueError (no comma/$ to strip) -> swallowed.
        assert m._num("   ") is None


# ════════════════════════════════════════════════════════════════════
# _scale_factor
# ════════════════════════════════════════════════════════════════════
class TestScaleFactor:
    def test_none_is_one(self):
        assert m._scale_factor(None) == 1.0

    def test_empty_string_is_one(self):
        assert m._scale_factor("") == 1.0

    def test_thousands(self):
        assert m._scale_factor("thousands") == 1000.0

    def test_millions(self):
        assert m._scale_factor("millions") == 1_000_000.0

    def test_ones(self):
        assert m._scale_factor("ones") == 1.0

    @pytest.mark.parametrize("tok", ["THOUSANDS", " Thousands ", "thOUSands"])
    def test_case_and_whitespace_insensitive(self, tok):
        assert m._scale_factor(tok) == 1000.0

    @pytest.mark.parametrize("tok", ["billions", "kajillions", "k", "1000"])
    def test_unknown_token_defaults_to_one(self, tok):
        assert m._scale_factor(tok) == 1.0


# ════════════════════════════════════════════════════════════════════
# _scaled
# ════════════════════════════════════════════════════════════════════
class TestScaled:
    def test_none_value_passthrough(self):
        assert m._scaled(None, 1000.0) is None

    def test_zero_value_not_short_circuited(self):
        # Only None short-circuits; 0.0 * factor == 0.0 (not None).
        assert m._scaled(0.0, 1000.0) == 0.0
        assert m._scaled(0.0, 1000.0) is not None

    def test_basic_multiply(self):
        assert m._scaled(30000, 1000.0) == 30_000_000.0

    def test_negative_preserves_sign(self):
        assert m._scaled(-5.0, 1000.0) == -5000.0


# ════════════════════════════════════════════════════════════════════
# _apply_ads_normalization
# ════════════════════════════════════════════════════════════════════
class TestApplyAdsNormalization:
    def test_ads_ratio_none_unchanged(self):
        assert m._apply_ads_normalization(200.0, 1000.0, 10.0, None) == 200.0

    def test_ads_ratio_below_two_unchanged(self):
        assert m._apply_ads_normalization(200.0, 1000.0, 10.0, 1.0) == 200.0

    def test_ratio_exactly_two_active(self):
        # implied_ads = 1000/10 = 100; csi=200 -> 200/100=2.0 == ratio
        # within band -> divide back -> 200/2 = 100.
        assert m._apply_ads_normalization(200.0, 1000.0, 10.0, 2.0) == 100.0

    def test_csi_none_unchanged(self):
        assert m._apply_ads_normalization(None, 1000.0, 10.0, 2.0) is None

    def test_pa_none_unchanged(self):
        assert m._apply_ads_normalization(200.0, None, 10.0, 2.0) == 200.0

    def test_cp_none_unchanged(self):
        assert m._apply_ads_normalization(200.0, 1000.0, None, 2.0) == 200.0

    def test_cp_nonpositive_unchanged(self):
        assert m._apply_ads_normalization(200.0, 1000.0, 0.0, 2.0) == 200.0
        assert m._apply_ads_normalization(200.0, 1000.0, -3.0, 2.0) == 200.0

    def test_consistent_within_band_divides(self):
        # csi == (pa/cp)*ads_ratio exactly -> divide back by ratio.
        assert m._apply_ads_normalization(200.0, 1000.0, 10.0, 2.0) == pytest.approx(100.0)

    def test_just_inside_5pct_band_divides(self):
        # implied=100, ratio 2.0; csi=209.8 -> csi/implied=2.098, off=0.049 (<0.05)
        assert m._apply_ads_normalization(209.8, 1000.0, 10.0, 2.0) == pytest.approx(104.9)

    def test_just_outside_5pct_band_unchanged(self):
        # csi=210.2 -> 2.102, off=0.051 (>0.05) -> unchanged.
        assert m._apply_ads_normalization(210.2, 1000.0, 10.0, 2.0) == pytest.approx(210.2)

    def test_csi_already_in_ads_unchanged(self):
        # csi == pa/cp (ratio ~1 vs ads_ratio 2) -> way outside band -> unchanged.
        assert m._apply_ads_normalization(100.0, 1000.0, 10.0, 2.0) == 100.0

    def test_implied_ads_nonpositive_unchanged(self):
        # Survey edge: implied_ads = pa/cp must be > 0. A negative principal
        # with cp>0 yields implied_ads < 0, so the band test never fires and
        # csi is returned unchanged (guards against a divide that would flip
        # sign / mis-normalize).
        assert m._apply_ads_normalization(200.0, -1000.0, 10.0, 2.0) == 200.0


# ════════════════════════════════════════════════════════════════════
# _form_family
# ════════════════════════════════════════════════════════════════════
class TestFormFamily:
    @pytest.mark.parametrize("form", ["", None])
    def test_falsy_defaults_us_periodic(self, form):
        assert m._form_family(form) == "us_periodic"

    @pytest.mark.parametrize("form", ["20-F", "40-F", "6-K"])
    def test_fpi_forms(self, form):
        assert m._form_family(form) == "fpi_annual"

    def test_amendment_suffix_dropped(self):
        assert m._form_family("20-F/A") == "fpi_annual"

    @pytest.mark.parametrize("form", ["10-K", "10-Q", "10-Q/A"])
    def test_us_periodic_forms(self, form):
        assert m._form_family(form) == "us_periodic"

    def test_lowercase_uppercased(self):
        assert m._form_family("10-k") == "us_periodic"

    def test_lowercase_fpi_uppercased(self):
        assert m._form_family("20-f") == "fpi_annual"

    def test_spaces_stripped(self):
        assert m._form_family(" 20-F ") == "fpi_annual"

    def test_8k_is_us_periodic_not_6k(self):
        # The key boundary mistake: 8-K is US periodic, only 6-K is FPI.
        assert m._form_family("8-K") == "us_periodic"

    @pytest.mark.parametrize("form", ["NT 10-K", "S-1", "junkform"])
    def test_unknown_defaults_us_periodic(self, form):
        assert m._form_family(form) == "us_periodic"


# ════════════════════════════════════════════════════════════════════
# _clean_warrant_row
# ════════════════════════════════════════════════════════════════════
class TestCleanWarrantRow:
    def test_count_none_both_none(self):
        out = m._clean_warrant_row(m.WarrantOverhangRow(), None)
        assert out["outstanding_count"] is None
        assert out["common_shares_issuable"] is None

    def test_csi_equals_count_one_to_one(self):
        out = m._clean_warrant_row(
            m.WarrantOverhangRow(outstanding_count=1234), None)
        assert out["outstanding_count"] == 1234.0
        assert out["common_shares_issuable"] == 1234.0

    def test_blank_instrument_name_to_none(self):
        out = m._clean_warrant_row(
            m.WarrantOverhangRow(instrument_name="   "), None)
        assert out["instrument_name"] is None

    def test_blank_expiry_to_none(self):
        out = m._clean_warrant_row(m.WarrantOverhangRow(expiry="  "), None)
        assert out["maturity_or_expiry"] is None

    def test_expiry_to_maturity_or_expiry(self):
        out = m._clean_warrant_row(
            m.WarrantOverhangRow(expiry="2027-06-01"), None)
        assert out["maturity_or_expiry"] == "2027-06-01"

    @pytest.mark.parametrize("pf", [True, False, None])
    def test_is_pre_funded_carried_verbatim(self, pf):
        out = m._clean_warrant_row(
            m.WarrantOverhangRow(is_pre_funded=pf), None)
        assert out["is_pre_funded"] is pf

    def test_principal_always_none(self):
        out = m._clean_warrant_row(
            m.WarrantOverhangRow(outstanding_count=10), None)
        assert out["principal_amount"] is None

    def test_strike_mapped(self):
        out = m._clean_warrant_row(
            m.WarrantOverhangRow(strike_price=2.5), None)
        assert out["strike_or_conversion_price"] == 2.5

    def test_category(self):
        out = m._clean_warrant_row(m.WarrantOverhangRow(), None)
        assert out["category"] == "warrant"

    def test_ads_ratio_ignored(self):
        # Warrant counts must NOT be ADS-normalized regardless of ratio.
        out = m._clean_warrant_row(
            m.WarrantOverhangRow(outstanding_count=1000), 5.0)
        assert out["outstanding_count"] == 1000.0
        assert out["common_shares_issuable"] == 1000.0


# ════════════════════════════════════════════════════════════════════
# _clean_convertible_row
# ════════════════════════════════════════════════════════════════════
class TestCleanConvertibleRow:
    def test_thousands_scales_principal_not_conversion_price(self):
        out = m._clean_convertible_row(
            m.ConvertibleOverhangRow(
                principal_amount=30000, conversion_price=10,
                dollar_scale="thousands"),
            None)
        assert out["principal_amount"] == 30_000_000.0
        assert out["strike_or_conversion_price"] == 10.0  # untouched

    def test_csi_derived_from_scaled_principal(self):
        out = m._clean_convertible_row(
            m.ConvertibleOverhangRow(
                principal_amount=30000, conversion_price=10,
                dollar_scale="thousands"),
            None)
        # 30,000,000 / 10
        assert out["common_shares_issuable"] == 3_000_000.0

    def test_csi_explicit_used_as_is(self):
        out = m._clean_convertible_row(
            m.ConvertibleOverhangRow(
                principal_amount=1000, conversion_price=10,
                common_shares_issuable=42),
            None)
        # Explicitly stated CSI is used, not the derived 1000/10=100.
        assert out["common_shares_issuable"] == 42.0

    def test_csi_none_when_cp_none(self):
        out = m._clean_convertible_row(
            m.ConvertibleOverhangRow(principal_amount=1000), None)
        assert out["common_shares_issuable"] is None

    def test_csi_none_when_cp_zero(self):
        out = m._clean_convertible_row(
            m.ConvertibleOverhangRow(principal_amount=1000, conversion_price=0),
            None)
        assert out["common_shares_issuable"] is None

    def test_principal_none_csi_none(self):
        out = m._clean_convertible_row(
            m.ConvertibleOverhangRow(conversion_price=10), None)
        assert out["principal_amount"] is None
        assert out["common_shares_issuable"] is None

    def test_outstanding_count_always_none(self):
        out = m._clean_convertible_row(
            m.ConvertibleOverhangRow(principal_amount=1000, conversion_price=10),
            None)
        assert out["outstanding_count"] is None

    def test_ads_normalization_applied(self):
        # Convertible DOES call _apply_ads_normalization.
        # pa=1000 cp=10 -> implied=100; csi explicit 200, ratio 200/100=2.0
        # within band -> divided back to 100.
        out = m._clean_convertible_row(
            m.ConvertibleOverhangRow(
                principal_amount=1000, conversion_price=10,
                common_shares_issuable=200),
            2.0)
        assert out["common_shares_issuable"] == pytest.approx(100.0)

    def test_blank_maturity_to_none(self):
        out = m._clean_convertible_row(
            m.ConvertibleOverhangRow(maturity="  "), None)
        assert out["maturity_or_expiry"] is None

    def test_explicit_csi_not_dollar_scaled(self):
        # Load-bearing asymmetry: dollar_scale rescales the AGGREGATE dollar
        # principal, never the share-count CSI. An explicit CSI passes
        # through verbatim even when dollar_scale='thousands' triples the
        # principal by 1000x.
        out = m._clean_convertible_row(
            m.ConvertibleOverhangRow(
                principal_amount=30000, conversion_price=10,
                common_shares_issuable=42, dollar_scale="thousands"),
            None)
        assert out["principal_amount"] == 30_000_000.0   # scaled
        assert out["common_shares_issuable"] == 42.0       # NOT scaled

    def test_category(self):
        out = m._clean_convertible_row(m.ConvertibleOverhangRow(), None)
        assert out["category"] == "convertible"


# ════════════════════════════════════════════════════════════════════
# _clean_preferred_row
# ════════════════════════════════════════════════════════════════════
class TestCleanPreferredRow:
    def test_csi_derived_from_scaled_alp(self):
        out = m._clean_preferred_row(
            m.PreferredOverhangRow(
                aggregate_liquidation_preference=1000, conversion_price=10),
            None)
        assert out["common_shares_issuable"] == 100.0

    def test_count_proxy_fallback_when_no_alp(self):
        out = m._clean_preferred_row(
            m.PreferredOverhangRow(outstanding_count=53197.0, conversion_price=5),
            None)
        # No alp -> falls back to count as a lower-bound proxy.
        assert out["common_shares_issuable"] == 53197.0

    def test_csi_none_when_cp_none(self):
        out = m._clean_preferred_row(
            m.PreferredOverhangRow(
                aggregate_liquidation_preference=1000, outstanding_count=10),
            None)
        assert out["common_shares_issuable"] is None

    def test_csi_none_when_cp_zero(self):
        out = m._clean_preferred_row(
            m.PreferredOverhangRow(
                aggregate_liquidation_preference=1000, conversion_price=0,
                outstanding_count=10),
            None)
        assert out["common_shares_issuable"] is None

    def test_fractional_count_preserved(self):
        out = m._clean_preferred_row(
            m.PreferredOverhangRow(outstanding_count=53197.5), None)
        assert out["outstanding_count"] == 53197.5

    def test_blank_series_letter_to_none(self):
        out = m._clean_preferred_row(
            m.PreferredOverhangRow(series_letter="  "), None)
        assert out["series_letter"] is None

    def test_series_letter_preserved(self):
        out = m._clean_preferred_row(
            m.PreferredOverhangRow(series_letter="A"), None)
        assert out["series_letter"] == "A"

    def test_dollar_scale_scales_alp_not_cp(self):
        out = m._clean_preferred_row(
            m.PreferredOverhangRow(
                aggregate_liquidation_preference=30000, conversion_price=10,
                dollar_scale="thousands"),
            None)
        assert out["principal_amount"] == 30_000_000.0
        assert out["strike_or_conversion_price"] == 10.0

    def test_no_ads_normalization(self):
        # Asymmetry vs convertible: preferred does NOT call ADS norm.
        # alp=1000 cp=10 -> csi=100; an ads_ratio of 2 would (if applied)
        # be irrelevant here since derived csi already equals implied,
        # so prove it via an EXPLICIT csi that ADS norm WOULD have divided.
        out = m._clean_preferred_row(
            m.PreferredOverhangRow(
                aggregate_liquidation_preference=1000, conversion_price=10,
                common_shares_issuable=200),
            2.0)
        # If ADS norm were applied, 200 -> 100. It is NOT, so csi stays 200.
        assert out["common_shares_issuable"] == 200.0

    def test_maturity_always_none(self):
        out = m._clean_preferred_row(m.PreferredOverhangRow(), None)
        assert out["maturity_or_expiry"] is None

    def test_category(self):
        out = m._clean_preferred_row(m.PreferredOverhangRow(), None)
        assert out["category"] == "preferred"


# ════════════════════════════════════════════════════════════════════
# _clean_shelf_row
# ════════════════════════════════════════════════════════════════════
class TestCleanShelfRow:
    def test_thousands_scales_all_three_capacities(self):
        out = m._clean_shelf_row(
            m.ShelfOverhangRow(
                total_capacity_usd=30000, drawn_to_date_usd=5000,
                remaining_capacity_usd=25000, dollar_scale="thousands"),
            None)
        assert out["total_capacity_usd"] == 30_000_000.0
        assert out["drawn_to_date_usd"] == 5_000_000.0
        assert out["remaining_capacity_usd"] == 25_000_000.0

    def test_file_number_preserved(self):
        out = m._clean_shelf_row(
            m.ShelfOverhangRow(file_number="333-123456"), None)
        assert out["file_number"] == "333-123456"

    def test_blank_file_number_to_none(self):
        out = m._clean_shelf_row(m.ShelfOverhangRow(file_number="  "), None)
        assert out["file_number"] is None

    @pytest.mark.parametrize("term", [True, False, None])
    def test_is_terminated_verbatim(self, term):
        out = m._clean_shelf_row(
            m.ShelfOverhangRow(is_terminated=term), None)
        assert out["is_terminated"] is term

    def test_none_capacity_stays_none(self):
        out = m._clean_shelf_row(
            m.ShelfOverhangRow(dollar_scale="thousands"), None)
        assert out["total_capacity_usd"] is None
        assert out["drawn_to_date_usd"] is None
        assert out["remaining_capacity_usd"] is None

    def test_core_slots_null(self):
        out = m._clean_shelf_row(
            m.ShelfOverhangRow(total_capacity_usd=100), None)
        assert out["outstanding_count"] is None
        assert out["common_shares_issuable"] is None
        assert out["principal_amount"] is None
        assert out["strike_or_conversion_price"] is None

    def test_blank_form_to_none(self):
        out = m._clean_shelf_row(m.ShelfOverhangRow(form="  "), None)
        assert out["form"] is None

    def test_category(self):
        out = m._clean_shelf_row(m.ShelfOverhangRow(), None)
        assert out["category"] == "shelf"


# ════════════════════════════════════════════════════════════════════
# _clean_atm_row
# ════════════════════════════════════════════════════════════════════
class TestCleanAtmRow:
    def test_sold_to_date_renamed_to_drawn_to_date(self):
        out = m._clean_atm_row(
            m.ATMOverhangRow(sold_to_date_usd=500), None)
        assert out["drawn_to_date_usd"] == 500.0
        # The input field name must NOT survive in the output dict.
        assert "sold_to_date_usd" not in out

    def test_dollar_scale_applied_to_all(self):
        out = m._clean_atm_row(
            m.ATMOverhangRow(
                total_capacity_usd=1000, sold_to_date_usd=500,
                remaining_capacity_usd=500, dollar_scale="thousands"),
            None)
        assert out["total_capacity_usd"] == 1_000_000.0
        assert out["drawn_to_date_usd"] == 500_000.0
        assert out["remaining_capacity_usd"] == 500_000.0

    def test_blank_sales_agent_to_none(self):
        out = m._clean_atm_row(m.ATMOverhangRow(sales_agent="  "), None)
        assert out["sales_agent"] is None

    def test_sales_agent_preserved(self):
        out = m._clean_atm_row(m.ATMOverhangRow(sales_agent="Maxim"), None)
        assert out["sales_agent"] == "Maxim"

    def test_blank_agreement_date_to_none(self):
        out = m._clean_atm_row(m.ATMOverhangRow(agreement_date="  "), None)
        assert out["agreement_date"] is None

    @pytest.mark.parametrize("term", [True, False, None])
    def test_is_terminated_passthrough(self, term):
        out = m._clean_atm_row(m.ATMOverhangRow(is_terminated=term), None)
        assert out["is_terminated"] is term

    def test_category(self):
        out = m._clean_atm_row(m.ATMOverhangRow(), None)
        assert out["category"] == "atm"


# ════════════════════════════════════════════════════════════════════
# _clean_equity_line_row
# ════════════════════════════════════════════════════════════════════
class TestCleanEquityLineRow:
    def test_drawn_to_date_keeps_name(self):
        # Contrast with ATM: equity_line keeps drawn_to_date_usd (no rename).
        out = m._clean_equity_line_row(
            m.EquityLineOverhangRow(drawn_to_date_usd=500), None)
        assert out["drawn_to_date_usd"] == 500.0

    def test_blank_investor_to_none(self):
        out = m._clean_equity_line_row(
            m.EquityLineOverhangRow(investor="  "), None)
        assert out["investor"] is None

    def test_investor_preserved(self):
        out = m._clean_equity_line_row(
            m.EquityLineOverhangRow(investor="Yorkville"), None)
        assert out["investor"] == "Yorkville"

    def test_dollar_scale_applied_to_all(self):
        out = m._clean_equity_line_row(
            m.EquityLineOverhangRow(
                total_capacity_usd=1000, drawn_to_date_usd=400,
                remaining_capacity_usd=600, dollar_scale="thousands"),
            None)
        assert out["total_capacity_usd"] == 1_000_000.0
        assert out["drawn_to_date_usd"] == 400_000.0
        assert out["remaining_capacity_usd"] == 600_000.0

    @pytest.mark.parametrize("term", [True, False, None])
    def test_is_terminated_passthrough(self, term):
        out = m._clean_equity_line_row(
            m.EquityLineOverhangRow(is_terminated=term), None)
        assert out["is_terminated"] is term

    def test_category(self):
        out = m._clean_equity_line_row(m.EquityLineOverhangRow(), None)
        assert out["category"] == "equity_line"


# ════════════════════════════════════════════════════════════════════
# _salvage_truncated_json
# ════════════════════════════════════════════════════════════════════
class TestSalvageTruncatedJson:
    def test_complete_valid_object(self):
        assert m._salvage_truncated_json(
            '{"warrants": [{"outstanding_count": 10}]}'
        ) == {"warrants": [{"outstanding_count": 10}]}

    def test_truncated_mid_row_keeps_completed(self):
        raw = ('{"warrants": [{"outstanding_count": 10}, '
               '{"outstanding_count": 999')
        # The half-written second row is dropped; the first survives.
        assert m._salvage_truncated_json(raw) == {
            "warrants": [{"outstanding_count": 10}]
        }

    def test_no_closer_returns_none(self):
        assert m._salvage_truncated_json('{"warrants": [{"outstanding') is None

    def test_empty_string_returns_none(self):
        assert m._salvage_truncated_json("") is None

    def test_list_root_returns_none(self):
        # Closes to a list, not a dict -> isinstance(dict) check fails.
        assert m._salvage_truncated_json("[1, 2, 3]") is None

    def test_runaway_20_digits_zeroed(self):
        out = m._salvage_truncated_json('{"a": 12345678901234567890, "b": 5}')
        assert out == {"a": 0, "b": 5}

    def test_19_digits_preserved(self):
        # Boundary: exactly 19 digits is below the >=20 runaway threshold.
        out = m._salvage_truncated_json('{"a": 1234567890123456789}')
        assert out == {"a": 1234567890123456789}

    def test_string_with_braces_not_counted(self):
        # Braces/brackets inside a JSON string must not affect the stack.
        out = m._salvage_truncated_json('{"a": "{[", "b": 1}')
        assert out == {"a": "{[", "b": 1}

    def test_escaped_quote_does_not_end_string(self):
        out = m._salvage_truncated_json(r'{"a": "he said \"hi\"", "b": 1}')
        assert out == {"a": 'he said "hi"', "b": 1}

    def test_unbalanced_extra_closer_breaks(self):
        # Second '}' hits empty stack -> break; prefix up to first close parses.
        out = m._salvage_truncated_json('{"a": 1}}')
        assert out == {"a": 1}

    def test_repaired_still_invalid_returns_none(self):
        # A closer IS seen (the trailing '}' sets cut), but the prefix
        # raw[:cut] = '{"a": 1,}' has a dangling comma -> json.loads raises
        # ValueError -> caught -> None.
        out = m._salvage_truncated_json('{"a": 1,}')
        assert out is None


# ════════════════════════════════════════════════════════════════════
# _row_count
# ════════════════════════════════════════════════════════════════════
class TestRowCount:
    def test_empty_dict(self):
        assert m._row_count({}) == 0

    def test_scalar_values_ignored(self):
        assert m._row_count({"x": 5, "y": "str", "z": None}) == 0

    def test_sum_of_list_lengths(self):
        assert m._row_count(
            {"warrants": [1, 2], "convertibles": [1], "preferreds": []}
        ) == 3

    def test_empty_list_is_zero(self):
        assert m._row_count({"warrants": []}) == 0

    def test_nested_dict_value_ignored(self):
        assert m._row_count({"a": {"nested": 1}, "b": [1, 2, 3]}) == 3


# ════════════════════════════════════════════════════════════════════
# _type_instruction_block
# ════════════════════════════════════════════════════════════════════
class TestTypeInstructionBlock:
    def test_no_filing_text_trailer(self):
        blk = m._type_instruction_block(
            m._WARRANT_PROMPT, form="10-K", cik=123, as_of="2025-12-31",
            anchor="ANCHOR_X")
        assert "Filing text:" not in blk

    def test_no_global_rules_text(self):
        blk = m._type_instruction_block(
            m._WARRANT_PROMPT, form="10-K", cik=123, as_of="2025-12-31",
            anchor="ANCHOR_X")
        assert "General rules:" not in blk

    def test_anchor_substituted(self):
        blk = m._type_instruction_block(
            m._WARRANT_PROMPT, form="10-K", cik=123, as_of="2025-12-31",
            anchor="ANCHOR_SENTINEL_XYZ")
        assert "ANCHOR_SENTINEL_XYZ" in blk

    def test_form_cik_as_of_substituted(self):
        blk = m._type_instruction_block(
            m._WARRANT_PROMPT, form="20-F", cik=999, as_of="2024-09-30",
            anchor="A")
        assert "20-F" in blk
        assert "999" in blk
        assert "2024-09-30" in blk

    def test_warrant_template_without_dollar_scale_renders(self):
        # The warrant template has no {dollar_scale_rule}; str.format must
        # ignore the extra kwarg without error.
        blk = m._type_instruction_block(
            m._WARRANT_PROMPT, form="10-K", cik=1, as_of="2025", anchor="A")
        assert blk  # rendered non-empty

    def test_trailing_whitespace_stripped(self):
        blk = m._type_instruction_block(
            m._WARRANT_PROMPT, form="10-K", cik=1, as_of="2025", anchor="A")
        assert blk == blk.rstrip()

    def test_dollar_scale_template_blanks_rule(self):
        # Convertible template references {dollar_scale_rule}; rendered with
        # "" so the rule body must be absent from the instruction block.
        blk = m._type_instruction_block(
            m._CONVERTIBLE_PROMPT, form="10-K", cik=1, as_of="2025",
            anchor="A")
        assert "- dollar_scale:" not in blk


# ════════════════════════════════════════════════════════════════════
# _build_combined_prompt
# ════════════════════════════════════════════════════════════════════
class TestBuildCombinedPrompt:
    def _build_us(self, text="FILING_BODY"):
        return m._build_combined_prompt(
            form="10-K", cik=123, as_of="2025-12-31",
            family="us_periodic", text=text)

    @pytest.mark.parametrize(
        "cat", ["WARRANT", "CONVERTIBLE", "PREFERRED", "SHELF", "ATM",
                "EQUITY_LINE"])
    def test_all_six_headers_present(self, cat):
        p = self._build_us()
        assert f"════════ {cat} ════════" in p

    def test_filing_text_appears_once(self):
        p = self._build_us(text="UNIQUE_BODY_SENTINEL")
        assert p.count("UNIQUE_BODY_SENTINEL") == 1
        # And it's at the end, after the trailer.
        assert p.rstrip().endswith("UNIQUE_BODY_SENTINEL")

    def test_global_rules_once(self):
        p = self._build_us()
        assert p.count("General rules:") == 1

    def test_dollar_scale_rule_once(self):
        p = self._build_us()
        assert p.count("- dollar_scale:") == 1

    def test_filing_text_trailer_once(self):
        p = self._build_us()
        assert p.count("Filing text:") == 1

    def test_fpi_family_selects_20f_anchors(self):
        p = m._build_combined_prompt(
            form="20-F", cik=1, as_of="2025", family="fpi_annual", text="X")
        # An fpi-specific phrase present, absent from us_periodic anchors.
        assert "Item 10" in p

    def test_us_family_omits_fpi_phrase(self):
        p = self._build_us()
        assert "Item 10" not in p

    def test_unknown_family_raises_keyerror(self):
        with pytest.raises(KeyError):
            m._build_combined_prompt(
                form="10-K", cik=1, as_of="2025",
                family="not_a_family", text="X")

    def test_exactly_six_block_headers(self):
        # No category header should appear twice (one block per type).
        p = self._build_us()
        for cat in ("WARRANT", "CONVERTIBLE", "PREFERRED", "SHELF", "ATM",
                    "EQUITY_LINE"):
            assert p.count(f"════════ {cat} ════════") == 1

    def test_intro_and_general_rules_banner_present(self):
        p = self._build_us()
        # Intro line from _COMBINED_INTRO and the trailing rules banner.
        assert "sorting each into the correct one of six lists" in p
        assert "GENERAL RULES (apply to all lists)" in p

    def test_fpi_and_us_prompts_differ(self):
        # The family selects different section anchors -> different prompts.
        us = self._build_us(text="X")
        fpi = m._build_combined_prompt(
            form="20-F", cik=123, as_of="2025-12-31",
            family="fpi_annual", text="X")
        assert us != fpi
        # A US-specific anchor phrase present only in us_periodic.
        assert "notes 8/9" in us
        assert "notes 8/9" not in fpi


# ════════════════════════════════════════════════════════════════════
# _parse_overhang_response (io_mockable — fake response object)
# ════════════════════════════════════════════════════════════════════
class TestParseOverhangResponse:
    def _resp(self, content, truncated=False):
        # Truncation is status="incomplete" + incomplete_details.reason on
        # /v1/responses, not a finish_reason string.
        return response_stub(
            text=content,
            status="incomplete" if truncated else "completed",
            incomplete_reason="max_output_tokens" if truncated else None,
        )

    def test_valid_json(self):
        resp = self._resp('{"warrants": [{"outstanding_count": 5}]}')
        parsed = m._parse_overhang_response(
            resp, m.CombinedOverhangList, accession="A", handler="h")
        assert parsed is not None
        assert parsed.warrants[0].outstanding_count == 5.0

    def test_invalid_json_not_truncated_returns_none(self, caplog):
        resp = self._resp("{bad json")
        with caplog.at_level("WARNING"):
            parsed = m._parse_overhang_response(
                resp, m.CombinedOverhangList, accession="ACC1", handler="h")
        assert parsed is None
        assert "parse failed" in caplog.text

    def test_invalid_json_truncated_salvageable(self):
        # Valid prefix (one warrant + start of a convertible) at the cap.
        content = ('{"warrants": [{"outstanding_count": 5}], '
                   '"convertibles": [{"principal_amount": 999')
        resp = self._resp(content, truncated=True)
        parsed = m._parse_overhang_response(
            resp, m.CombinedOverhangList, accession="A", handler="h")
        assert parsed is not None
        assert parsed.warrants[0].outstanding_count == 5.0
        assert parsed.convertibles == []  # the truncated row was dropped

    def test_truncated_unsalvageable_returns_none(self, caplog):
        resp = self._resp('{"warrants": [{"outstanding', truncated=True)
        with caplog.at_level("WARNING"):
            parsed = m._parse_overhang_response(
                resp, m.CombinedOverhangList, accession="A", handler="h")
        assert parsed is None
        assert "salvage found no valid prefix" in caplog.text

    def test_truncated_salvaged_prefix_still_invalid_returns_none(self, caplog):
        # Salvages to {"bogus_key": [{"x": 1}]} which model_validate rejects
        # (extra='forbid'); the warrants row was mid-write and dropped.
        content = ('{"bogus_key": [{"x": 1}], '
                   '"warrants": [{"outstanding_count": 5')
        resp = self._resp(content, truncated=True)
        with caplog.at_level("WARNING"):
            parsed = m._parse_overhang_response(
                resp, m.CombinedOverhangList, accession="A", handler="h")
        assert parsed is None
        assert "salvaged prefix" in caplog.text

    def test_missing_status_attr_treated_not_truncated(self):
        class NoStatus:
            output_text = "{bad"
        parsed = m._parse_overhang_response(
            NoStatus(), m.CombinedOverhangList, accession="A", handler="h")
        assert parsed is None

    def test_per_type_list_model_accepted(self):
        # The fallback path validates against per-type list models, not just
        # CombinedOverhangList — exercise WarrantOverhangList as `model`.
        resp = self._resp('{"warrants": [{"outstanding_count": 9}]}')
        parsed = m._parse_overhang_response(
            resp, m.WarrantOverhangList, accession="A", handler="h")
        assert isinstance(parsed, m.WarrantOverhangList)
        assert parsed.warrants[0].outstanding_count == 9.0

    def test_salvage_emits_recovered_row_count_info_log(self, caplog):
        # The salvage success path logs an INFO with the recovered row count
        # (via _row_count) — assert the count threads through.
        content = ('{"warrants": [{"outstanding_count": 5}, '
                   '{"outstanding_count": 6}], '
                   '"convertibles": [{"principal_amount": 999')
        resp = self._resp(content, truncated=True)
        with caplog.at_level("INFO"):
            parsed = m._parse_overhang_response(
                resp, m.CombinedOverhangList, accession="ACC9", handler="h")
        # Both completed warrants survive; the half-written convertible drops.
        assert len(parsed.warrants) == 2
        assert parsed.convertibles == []
        assert "salvaged 2 row(s)" in caplog.text


# ════════════════════════════════════════════════════════════════════
# _load_filing_text (db_backed via autouse temp_db)
# ════════════════════════════════════════════════════════════════════
def _insert_raw(temp_db, accession, doc_name, doc_type, content):
    # dilution_raw.accession_number FKs to dilution_filings, so the parent
    # filing must exist first (PRAGMA foreign_keys=ON in the temp_db conn).
    temp_db.execute(
        "INSERT OR IGNORE INTO dilution_filings "
        "(accession_number, cik, form, filing_date) VALUES (?,?,?,?)",
        (accession, 1, "10-K", "2025-01-01"),
    )
    temp_db.execute(
        "INSERT INTO dilution_raw "
        "(accession_number, doc_name, doc_type, content_md, downloaded_at) "
        "VALUES (?,?,?,?,?)",
        (accession, doc_name, doc_type, content, "2025-01-01T00:00:00Z"),
    )


class TestLoadFilingText:
    def test_no_rows_returns_none(self, temp_db):
        assert m._load_filing_text("MISSING-ACC") is None

    def test_single_ex_row_falls_back_to_longest(self, temp_db):
        # Only an EX-* row exists -> the loop finds no primary, falls back
        # to rows[0] (the single, longest row).
        _insert_raw(temp_db, "ACC-EX", "ex.htm", "EX-99.1", "EXHIBIT_BODY")
        assert m._load_filing_text("ACC-EX") == "EXHIBIT_BODY"

    def test_non_ex_preferred_over_longer_ex(self, temp_db):
        # Rows sort by LENGTH DESC; the long EX-99 sorts first, but the
        # loop skips EX-* and returns the first non-EX row (the 10-K body).
        _insert_raw(temp_db, "ACC-1", "ex.htm", "EX-99.1", "E" * 200)
        _insert_raw(temp_db, "ACC-1", "body.htm", "10-K", "TEN_K_BODY")
        assert m._load_filing_text("ACC-1") == "TEN_K_BODY"

    def test_doc_type_none_counts_as_primary(self, temp_db):
        # NULL doc_type -> ("" or "").upper() == "" -> not EX-* -> primary.
        _insert_raw(temp_db, "ACC-N", "doc.htm", None, "NULL_TYPE_BODY")
        assert m._load_filing_text("ACC-N") == "NULL_TYPE_BODY"

    def test_lowercase_ex_uppercased_and_skipped(self, temp_db):
        # 'ex-99' uppercases to 'EX-99' -> recognized & skipped; the 10-K
        # body wins even though it's shorter.
        _insert_raw(temp_db, "ACC-L", "ex.htm", "ex-99", "X" * 100)
        _insert_raw(temp_db, "ACC-L", "body.htm", "10-K", "SHORTBODY")
        assert m._load_filing_text("ACC-L") == "SHORTBODY"

    def test_result_is_whitespace_normalized(self, temp_db):
        # normalize_filing_text strips zero-width spaces and collapses
        # runs of 3+ spaces to 2.
        _insert_raw(temp_db, "ACC-W", "body.htm", "10-K",
                    "A​B    C")  # ZWSP + 4 spaces
        assert m._load_filing_text("ACC-W") == "AB  C"


# ════════════════════════════════════════════════════════════════════
# Pydantic row schemas
# ════════════════════════════════════════════════════════════════════
class TestPydanticRowSchemas:
    @pytest.mark.parametrize("cls", [
        m.WarrantOverhangRow, m.ConvertibleOverhangRow, m.PreferredOverhangRow,
        m.ShelfOverhangRow, m.ATMOverhangRow, m.EquityLineOverhangRow,
    ])
    def test_extra_key_forbidden(self, cls):
        with pytest.raises(ValidationError):
            cls(this_key_does_not_exist=1)

    @pytest.mark.parametrize("cls", [
        m.WarrantOverhangRow, m.ConvertibleOverhangRow, m.PreferredOverhangRow,
        m.ShelfOverhangRow, m.ATMOverhangRow, m.EquityLineOverhangRow,
    ])
    def test_all_fields_optional(self, cls):
        # Construct with no args -> valid (all-optional schema).
        cls()

    def test_instrument_name_max_length(self):
        with pytest.raises(ValidationError):
            m.WarrantOverhangRow(instrument_name="x" * 201)

    def test_instrument_name_at_limit_ok(self):
        assert m.WarrantOverhangRow(instrument_name="x" * 200).instrument_name

    def test_notes_max_length(self):
        with pytest.raises(ValidationError):
            m.WarrantOverhangRow(notes="x" * 801)

    def test_numeric_string_coerced_to_float(self):
        r = m.WarrantOverhangRow(outstanding_count="100")
        assert r.outstanding_count == 100.0
        assert isinstance(r.outstanding_count, float)

    @pytest.mark.parametrize("val", [True, False, None])
    def test_is_pre_funded_accepts_bool_and_none(self, val):
        assert m.WarrantOverhangRow(is_pre_funded=val).is_pre_funded is val


# ════════════════════════════════════════════════════════════════════
# CombinedOverhangList
# ════════════════════════════════════════════════════════════════════
class TestCombinedOverhangList:
    def test_empty_dict_yields_six_empty_lists(self):
        c = m.CombinedOverhangList.model_validate({})
        assert c.warrants == []
        assert c.convertibles == []
        assert c.preferreds == []
        assert c.shelves == []
        assert c.atms == []
        assert c.equity_lines == []

    def test_extra_top_level_key_forbidden(self):
        with pytest.raises(ValidationError):
            m.CombinedOverhangList.model_validate({"bogus": []})

    def test_row_with_extra_inner_key_propagates_error(self):
        with pytest.raises(ValidationError):
            m.CombinedOverhangList.model_validate(
                {"warrants": [{"bogus_inner": 1}]})

    def test_partial_dict_defaults_others(self):
        c = m.CombinedOverhangList.model_validate(
            {"warrants": [{"outstanding_count": 7}]})
        assert c.warrants[0].outstanding_count == 7.0
        assert c.convertibles == []
        assert c.preferreds == []
        assert c.shelves == []
        assert c.atms == []
        assert c.equity_lines == []


# ════════════════════════════════════════════════════════════════════
# extract_overhang_rows (async orchestration — LLM seam monkeypatched,
# NO network). Drives the full merge->clean and merge-fail->fallback
# wiring that the pure helper tests cannot reach. Two module-level seams
# are patched: acomplete (the single async network entry) and
# _load_filing_text (the DB read). temp_db autouse guarantees no real DB
# is touched even though we also patch the loader.
# ════════════════════════════════════════════════════════════════════
import asyncio


def _resp(content, truncated=False):
    return response_stub(
        text=content,
        status="incomplete" if truncated else "completed",
        incomplete_reason="max_output_tokens" if truncated else None,
    )


class TestExtractOverhangRowsOrchestration:
    def _patch_seam(self, monkeypatch, *, body="FILING BODY", asample):
        """Wire the two module seams: the supplied async call stub and a
        fixed filing body. Returns nothing — patches in place.

        The stub is called as acomplete(client, name=..., messages=...,
        ...), so it receives the request kwargs the production code built.
        """
        async def _acomplete(client, *, name=None, **kw):
            return await asample(kw, accession=kw.get("accession"),
                                 handler=name)

        monkeypatch.setattr(m, "acomplete", _acomplete)
        monkeypatch.setattr(m, "_load_filing_text", lambda acc: body)

    def _run(self, **kw):
        defaults = dict(
            accession="A", form="10-K", filing_date="2025-01-01",
            report_date=None, cik=123, client=object(), unit_ctx=None,
        )
        defaults.update(kw)
        return asyncio.run(m.extract_overhang_rows(**defaults))

    def test_empty_body_short_circuits_no_llm_call(self, monkeypatch):
        calls = []

        async def asample(chat, accession=None, handler=None):
            calls.append(handler)
            return _resp("{}")

        self._patch_seam(monkeypatch, body="", asample=asample)
        out = self._run()
        assert out == []
        assert calls == []   # no body -> no LLM call at all

    def test_merge_happy_path_cleans_all_rows(self, monkeypatch):
        async def asample(chat, accession=None, handler=None):
            # one warrant + one atm (sold_to_date) in a single merged response
            return _resp('{"warrants": [{"outstanding_count": 10}], '
                         '"atms": [{"sold_to_date_usd": 500}]}')

        self._patch_seam(monkeypatch, asample=asample)
        out = self._run()
        cats = [r["category"] for r in out]
        assert cats == ["warrant", "atm"]
        w = out[0]
        assert w["outstanding_count"] == 10.0
        assert w["common_shares_issuable"] == 10.0   # 1:1
        a = out[1]
        # The ATM rename survives the full pipeline.
        assert a["drawn_to_date_usd"] == 500.0
        assert "sold_to_date_usd" not in a

    def test_merge_fail_falls_back_to_six_specialists_in_order(
            self, monkeypatch):
        seen = []

        async def asample(chat, accession=None, handler=None):
            seen.append(handler)
            if handler == "overhang-combined":
                return _resp("{NOT JSON")
            payloads = {
                "overhang-warrant": '{"warrants": [{"outstanding_count": 7}]}',
                "overhang-convertible": '{"convertibles": []}',
                "overhang-preferred": '{"preferreds": []}',
                "overhang-shelf": '{"shelves": []}',
                "overhang-atm": '{"atms": []}',
                "overhang-equity-line": '{"equity_lines": []}',
            }
            return _resp(payloads[handler])

        self._patch_seam(monkeypatch, asample=asample)
        out = self._run()
        # The merged call ran first, THEN all six specialists dispatched.
        assert seen[0] == "overhang-combined"
        assert set(seen[1:]) == {
            "overhang-warrant", "overhang-convertible", "overhang-preferred",
            "overhang-shelf", "overhang-atm", "overhang-equity-line",
        }
        # Only the warrant specialist returned a row -> one cleaned warrant.
        assert [r["category"] for r in out] == ["warrant"]
        assert out[0]["outstanding_count"] == 7.0

    def test_merge_call_exception_triggers_fallback(self, monkeypatch):
        # A transport error on the merged call (not a parse failure) must
        # also route to the per-specialist fallback, not crash.
        seen = []

        async def asample(chat, accession=None, handler=None):
            seen.append(handler)
            if handler == "overhang-combined":
                raise RuntimeError("transport boom")
            return _resp('{"%s": []}' % {
                "overhang-warrant": "warrants",
                "overhang-convertible": "convertibles",
                "overhang-preferred": "preferreds",
                "overhang-shelf": "shelves",
                "overhang-atm": "atms",
                "overhang-equity-line": "equity_lines",
            }[handler])

        self._patch_seam(monkeypatch, asample=asample)
        out = self._run()
        assert out == []   # all specialists empty
        assert seen[0] == "overhang-combined"
        assert len(seen) == 7   # combined + six specialists

    def test_ads_ratio_threaded_from_unit_ctx(self, monkeypatch):
        async def asample(chat, accession=None, handler=None):
            # explicit csi 200, pa 1000, cp 10 -> ADS norm (ratio 2) -> 100
            return _resp('{"convertibles": [{"principal_amount": 1000, '
                         '"conversion_price": 10, '
                         '"common_shares_issuable": 200}]}')

        self._patch_seam(monkeypatch, asample=asample)
        out = self._run(unit_ctx={"ads_ratio": 2.0})
        assert out[0]["common_shares_issuable"] == pytest.approx(100.0)

    def test_no_ads_ratio_leaves_csi_unnormalized(self, monkeypatch):
        async def asample(chat, accession=None, handler=None):
            return _resp('{"convertibles": [{"principal_amount": 1000, '
                         '"conversion_price": 10, '
                         '"common_shares_issuable": 200}]}')

        self._patch_seam(monkeypatch, asample=asample)
        out = self._run(unit_ctx=None)
        # No ads_ratio -> ADS norm guard short-circuits -> explicit csi kept.
        assert out[0]["common_shares_issuable"] == 200.0

    def test_oversize_text_truncated_to_cap(self, monkeypatch):
        # Body longer than MAX_INPUT_CHARS must be truncated before it
        # reaches the prompt; capture the request the seam actually sees.
        captured = {}
        big = "Z" * (m.MAX_INPUT_CHARS + 50)

        async def asample(kw, accession=None, handler=None):
            captured["user_text"] = kw["messages"][-1]["content"]
            return _resp("{}")

        self._patch_seam(monkeypatch, body=big, asample=asample)
        self._run()
        # The Z-run in the assembled prompt is capped at MAX_INPUT_CHARS.
        assert captured["user_text"].count("Z") == m.MAX_INPUT_CHARS


# ════════════════════════════════════════════════════════════════════
# Specialist fallback: shared cacheable prefix
# ════════════════════════════════════════════════════════════════════
class TestSpecialistPrefixSharing:
    """The six specialists read the same filing body, so [system + text]
    must be a byte-identical prefix across all six calls and land in one
    prompt-cache bucket. Measured effect on a 164K-char 10-K: 90% of the
    fallback's input tokens served from cache instead of billed fresh.
    """

    def _capture(self, monkeypatch, body="THE FILING BODY"):
        seen = []

        async def _acomplete(client, *, name=None, **kw):
            seen.append({"name": name, **kw})
            return _resp('{"warrants": []}')

        monkeypatch.setattr(m, "acomplete", _acomplete)
        asyncio.run(m._extract_per_specialist_lists(
            client=object(), accession="A", preamble="PRE", form="10-K",
            cik=1, as_of="2025-01-01", family=m._form_family("10-K"),
            text=body))
        return seen

    def test_all_six_specialists_run(self, monkeypatch):
        assert len(self._capture(monkeypatch)) == 6

    def test_system_message_is_identical_across_specialists(self, monkeypatch):
        systems = {c["messages"][0]["content"] for c in self._capture(monkeypatch)}
        assert len(systems) == 1, "a per-specialist system message breaks the prefix"

    def test_filing_text_is_the_second_message_and_identical(self, monkeypatch):
        seen = self._capture(monkeypatch, body="UNIQUE-BODY-XYZ")
        texts = {c["messages"][1]["content"] for c in seen}
        assert len(texts) == 1
        assert "UNIQUE-BODY-XYZ" in texts.pop()

    def test_instruction_comes_after_the_text(self, monkeypatch):
        # Document first, instructions last — both the cache prefix and
        # instruction-following depend on this order.
        for c in self._capture(monkeypatch):
            assert len(c["messages"]) == 3
            assert c["messages"][1]["content"].startswith("Filing text:")
            assert "PRE" in c["messages"][2]["content"]

    def test_body_is_not_duplicated_into_the_instruction(self, monkeypatch):
        # _fmt_args now passes text="" and _instruction_only strips the
        # trailer; a regression here would send the document twice.
        for c in self._capture(monkeypatch, body="UNIQUE-BODY-XYZ"):
            assert "UNIQUE-BODY-XYZ" not in c["messages"][2]["content"]
            assert "Filing text:" not in c["messages"][2]["content"]

    def test_one_shared_cache_key(self, monkeypatch):
        keys = {c["cache_key"] for c in self._capture(monkeypatch)}
        assert keys == {"overhang-specialist"}, "per-specialist keys defeat sharing"

    def test_first_call_is_awaited_before_the_others_fan_out(self, monkeypatch):
        # A prefix only caches once a request carrying it COMPLETES, so
        # firing all six at once yields six misses. Assert the first call
        # finishes before the second starts.
        order = []

        async def _acomplete(client, *, name=None, **kw):
            order.append(("start", name))
            await asyncio.sleep(0)
            order.append(("end", name))
            return _resp('{"warrants": []}')

        monkeypatch.setattr(m, "acomplete", _acomplete)
        asyncio.run(m._extract_per_specialist_lists(
            client=object(), accession="A", preamble="", form="10-K", cik=1,
            as_of="2025-01-01", family=m._form_family("10-K"), text="B"))
        assert order[0][0] == "start" and order[1][0] == "end", (
            "the warm-up call must complete before the fan-out")
        assert order[1][1] == order[0][1]

    def test_results_keep_positional_order(self, monkeypatch):
        # Caller unpacks (warrants, convertibles, preferreds, shelves,
        # atms, equity_lines); the warm-up call must not reshuffle that.
        payloads = {
            "overhang-warrant": '{"warrants": [{"outstanding_count": 1}]}',
            "overhang-convertible": '{"convertibles": [{"principal_amount": 2}]}',
            "overhang-preferred": '{"preferreds": [{"outstanding_count": 3}]}',
            "overhang-shelf": '{"shelves": [{"drawn_to_date_usd": 4}]}',
            "overhang-atm": '{"atms": [{"remaining_capacity_usd": 5}]}',
            "overhang-equity-line": '{"equity_lines": [{"drawn_to_date_usd": 6}]}',
        }

        async def _acomplete(client, *, name=None, **kw):
            return _resp(payloads[name])

        monkeypatch.setattr(m, "acomplete", _acomplete)
        w, c, p, s, a, e = asyncio.run(m._extract_per_specialist_lists(
            client=object(), accession="A", preamble="", form="10-K", cik=1,
            as_of="2025-01-01", family=m._form_family("10-K"), text="B"))
        assert [len(x) for x in (w, c, p, s, a, e)] == [1, 1, 1, 1, 1, 1]
        assert w[0].outstanding_count == 1.0
        assert c[0].principal_amount == 2.0
        assert p[0].outstanding_count == 3.0
