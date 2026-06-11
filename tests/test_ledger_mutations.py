"""Unit tests for dilution/ledger/mutations.py.

Pure mutation-vocabulary module: frozen dataclasses with @property dict
shapers, regex/date helpers, two dict->dataclass factories, and a JSON
serializer. No DB / I/O / LLM. The autouse temp_db fixture from
conftest.py runs but is never referenced here.

Determinism: every date is passed explicitly so date.today() is never
relied upon, except in the two factory today()-fallback tests which assert
against an explicitly-captured date.today() value.
"""

from __future__ import annotations

import json
from datetime import date, datetime

import pytest

from dilution.ledger import mutations as M
from dilution.ledger.mutations import (
    AmendAtm,
    AmendConvertible,
    AmendEquity,
    AmendEquityLine,
    AmendPreferred,
    AmendS1Offering,
    AmendShelf,
    AmendWarrant,
    ApplySplit,
    CloseInstrument,
    ConfirmClosing,
    CreateAtm,
    CreateConvertible,
    CreateEquity,
    CreateEquityLine,
    CreatePreferred,
    CreateS1Offering,
    CreateShelf,
    CreateWarrant,
    NoteNoEvent,
    RecordConversion,
    RecordDrawdown,
    RecordExercise,
    RecordPartialRedemption,
    RecordPartialTermination,
    RestateAtm,
    amend_from_dict,
    create_from_dict,
    extract_series_letter,
    hoist_nested_payload,
    mutation_to_dict,
    safe_date,
    warrant_series_key,
)

ED = date(2024, 1, 15)  # canonical event_date used across tests


# ───────────────────────── extract_series_letter ─────────────────────────


class TestExtractSeriesLetter:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, None),
            ("", None),
            (5, "5"),  # non-str coerced; bare digit
            ("D", "D"),
            ("d", "D"),  # lowercase bare -> uppercased
            ("10", "10"),  # 2-char bare digit
            ("Series D Preferred Stock", "D"),
            ("series 9 convertible preferred", "9"),  # case-insensitive
            ("Class B Units", "B"),  # Class regex
            ("Class A Common Stock", None),  # negative lookahead blocks 'common'
            ("Class A Ordinary Shares", None),  # blocks 'ordinary'
            ("ABCD", None),  # len-4 bare falls through, no Series/Class
            ("A1", None),  # not isalpha and not isdigit -> skip bare, no regex
            ("Series AA", "AA"),  # multi-letter
            ("Common Warrant", None),  # no series marker
        ],
    )
    def test_extraction(self, raw, expected):
        assert extract_series_letter(raw) == expected

    def test_3char_bare_alpha(self):
        # boundary: len 3 alpha is accepted as bare token
        assert extract_series_letter("ABC") == "ABC"

    def test_4char_bare_alpha_falls_through(self):
        # len 4 alpha not bare-eligible, regex finds no Series/Class
        assert extract_series_letter("ABCD") is None


# ───────────────────────── warrant_series_key ─────────────────────────


class TestWarrantSeriesKey:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, ""),
            ("", ""),
            ("   ", ""),  # whitespace only -> ''
            ("A", "A"),
            ("pre-funded", "PRE-FUNDED"),  # uppercased
            ("Inducement", "INDUCEMENT"),
            ("August 23", ""),  # full month + digit -> date label
            ("Sep 5", ""),  # abbrev month + digit
            ("23 August", ""),  # digit + month
            ("May", "MAY"),  # bare month, NO digit -> NOT suppressed
            ("Aug", "AUG"),  # bare abbrev, no digit -> not suppressed
            ("Sept 5", ""),  # 4-char 'Sept' alt in regex
            ("august 23", ""),  # case-insensitive
            ("  August 23", ""),  # leading whitespace tolerated
            ("1", "1"),  # digit identifier, not a date
            (5, "5"),  # non-str coerced
        ],
    )
    def test_normalization(self, raw, expected):
        assert warrant_series_key(raw) == expected


# ───────────────────────── _normalize_date ─────────────────────────


class TestNormalizeDate:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, None),
            ("", None),
            ("   ", None),
            ("2024-01-15", "2024-01-15"),  # already ISO passthrough
            ("2024/01/15", "2024-01-15"),
            ("01/15/2024", "2024-01-15"),
            ("January 15, 2024", "2024-01-15"),
            ("Jan 15, 2024", "2024-01-15"),
            ("15 January 2024", "2024-01-15"),
            ("15 Jan 2024", "2024-01-15"),
        ],
    )
    def test_valid(self, raw, expected):
        assert M._normalize_date(raw) == expected

    @pytest.mark.parametrize("bad", ["2019", "not a date", "13/13/2024"])
    def test_unrecognized_raises_valueerror(self, bad):
        with pytest.raises(ValueError):
            M._normalize_date(bad)

    @pytest.mark.parametrize("nonstr", [5, date(2024, 1, 1), 3.14, ["x"]])
    def test_nonstr_raises_valueerror_naming_type(self, nonstr):
        with pytest.raises(ValueError, match="date must be a string"):
            M._normalize_date(nonstr)

    def test_iso_shaped_but_invalid_returned_verbatim(self):
        # GOTCHA: ISO branch is a regex shape check only, no calendar
        # validation -> '2024-13-99' passes through unvalidated.
        assert M._normalize_date("2024-13-99") == "2024-13-99"


# ───────────────────────── safe_date ─────────────────────────


class TestSafeDate:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, None),
            ("", None),
            ("2019", None),  # ValueError swallowed
            ("2024-01-15", "2024-01-15"),
            ("01/15/2024", "2024-01-15"),
            ("garbage", None),
            (5, None),  # non-str raises ValueError inside, caught -> None
        ],
    )
    def test_tolerant(self, raw, expected):
        assert safe_date(raw) == expected


# ───────────────────────── hoist_nested_payload ─────────────────────────


class TestHoistNestedPayload:
    @pytest.mark.parametrize("nondict", [["a"], None, "str", 5])
    def test_non_dict_returned_unchanged(self, nondict):
        assert hoist_nested_payload(nondict, ("terms",)) is nondict

    def test_wrapper_absent_returns_copy_no_mutation(self):
        data = {"a": 1}
        out = hoist_nested_payload(data, ("terms",))
        assert out == {"a": 1}
        assert out is not data  # a fresh dict() copy
        assert data == {"a": 1}  # input not mutated

    def test_wrapper_value_not_dict_skipped_key_retained(self):
        data = {"terms": "a string", "a": 1}
        out = hoist_nested_payload(data, ("terms",))
        assert out == {"terms": "a string", "a": 1}

    def test_nested_value_wins_on_conflict(self):
        data = {"x": 1, "terms": {"x": 99, "y": 2}}
        out = hoist_nested_payload(data, ("terms",))
        assert out == {"x": 99, "y": 2}
        assert "terms" not in out

    def test_multiple_keys_some_present(self):
        data = {"terms": {"a": 1}, "outstanding": "notdict", "z": 5}
        out = hoist_nested_payload(data, ("terms", "outstanding", "missing"))
        assert out == {"a": 1, "outstanding": "notdict", "z": 5}

    def test_empty_wrapper_keys_returns_copy(self):
        data = {"a": 1}
        out = hoist_nested_payload(data, ())
        assert out == {"a": 1}
        assert out is not data

    def test_empty_nested_dict_pops_key_adds_nothing(self):
        data = {"terms": {}, "a": 1}
        out = hoist_nested_payload(data, ("terms",))
        assert out == {"a": 1}

    def test_input_immutability(self):
        data = {"terms": {"x": 1}}
        hoist_nested_payload(data, ("terms",))
        assert data == {"terms": {"x": 1}}  # original untouched


# ───────────────────────── _iso ─────────────────────────


class TestIso:
    def test_none(self):
        assert M._iso(None) is None

    def test_date(self):
        assert M._iso(date(2024, 1, 5)) == "2024-01-05"


# ───────────────────────── _offset_date ─────────────────────────


class TestOffsetDate:
    ANCHOR = date(2024, 1, 15)

    def test_explicit_wins_over_offset(self):
        explicit = date(2025, 6, 1)
        assert M._offset_date(explicit, self.ANCHOR, 12) == explicit

    def test_offset_only(self):
        assert M._offset_date(None, self.ANCHOR, 12) == date(2025, 1, 15)

    def test_offset_zero_is_not_none(self):
        # 0 is not None -> anchor + 0 months == anchor
        assert M._offset_date(None, self.ANCHOR, 0) == self.ANCHOR

    def test_both_none(self):
        assert M._offset_date(None, self.ANCHOR, None) is None

    def test_negative_offset(self):
        assert M._offset_date(None, self.ANCHOR, -2) == date(2023, 11, 15)

    def test_month_overflow_clamps(self):
        # relativedelta clamps Jan 31 + 1 month to Feb 29 (2024 is leap)
        assert M._offset_date(None, date(2024, 1, 31), 1) == date(2024, 2, 29)


# ───────────────────────── CreateWarrant ─────────────────────────


class TestCreateWarrantResolveDates:
    def test_explicit_exercisable_wins_over_offset(self):
        w = CreateWarrant(count=1, strike=1.0, event_date=ED,
                          exercisable_date=date(2024, 3, 1),
                          exercise_offset_months=6)
        exq, _ = w._resolve_dates()
        assert exq == date(2024, 3, 1)

    def test_exercisable_from_offset(self):
        w = CreateWarrant(count=1, strike=1.0, event_date=ED,
                          exercise_offset_months=6)
        exq, _ = w._resolve_dates()
        assert exq == date(2024, 7, 15)

    def test_exercisable_none_when_no_offset(self):
        w = CreateWarrant(count=1, strike=1.0, event_date=ED)
        exq, _ = w._resolve_dates()
        assert exq is None
        assert "exercisable_date" not in w.terms

    def test_explicit_expiration_wins_over_term_months(self):
        w = CreateWarrant(count=1, strike=1.0, event_date=ED,
                          expiration=date(2030, 1, 1), term_months=12)
        _, exp = w._resolve_dates()
        assert exp == date(2030, 1, 1)

    def test_term_anchor_exercise_with_resolved_exq(self):
        # "Nth anniversary of the Initial Exercise Date"
        w = CreateWarrant(count=1, strike=1.0, event_date=ED,
                          exercise_offset_months=6, term_months=60,
                          term_anchor="exercise")
        exq, exp = w._resolve_dates()
        assert exq == date(2024, 7, 15)
        assert exp == date(2029, 7, 15)  # exq + 60 months

    def test_term_anchor_exercise_but_exq_none_falls_back_to_event(self):
        w = CreateWarrant(count=1, strike=1.0, event_date=ED,
                          term_months=60, term_anchor="exercise")
        exq, exp = w._resolve_dates()
        assert exq is None
        assert exp == date(2029, 1, 15)  # event_date + 60 months

    def test_term_anchor_issuance_default(self):
        w = CreateWarrant(count=1, strike=1.0, event_date=ED, term_months=12)
        _, exp = w._resolve_dates()
        assert exp == date(2025, 1, 15)


class TestCreateWarrantTerms:
    def test_pre_funded_series_promotes_even_when_flag_false(self):
        w = CreateWarrant(count=1, strike=5.0, event_date=ED,
                          is_pre_funded=False, series_letter=" Pre-Funded ")
        # filing's own series tag authoritative -> True
        assert w.terms["is_pre_funded"] is True

    def test_strike_exactly_threshold_promotes(self):
        w = CreateWarrant(count=1, strike=0.001, event_date=ED)
        assert w.terms["is_pre_funded"] is True

    def test_strike_above_threshold_not_promoted(self):
        w = CreateWarrant(count=1, strike=0.0011, event_date=ED)
        assert "is_pre_funded" not in w.terms

    def test_explicit_false_high_strike_emits_false(self):
        w = CreateWarrant(count=1, strike=5.0, event_date=ED,
                          is_pre_funded=False)
        assert w.terms["is_pre_funded"] is False

    def test_strike_threshold_does_not_override_explicit_false(self):
        # is_pf is not None (it's False), so the strike<=0.001 branch
        # is gated behind `is_pf is None` and does NOT fire.
        w = CreateWarrant(count=1, strike=0.001, event_date=ED,
                          is_pre_funded=False)
        assert w.terms["is_pre_funded"] is False

    def test_known_owners_tuple_to_list(self):
        w = CreateWarrant(count=1, strike=1.0, event_date=ED,
                          known_owners=("Alice", "Bob"))
        assert w.terms["known_owners"] == ["Alice", "Bob"]

    def test_known_owners_none_omitted(self):
        w = CreateWarrant(count=1, strike=1.0, event_date=ED)
        assert "known_owners" not in w.terms

    def test_units_and_series_emitted_when_set(self):
        w = CreateWarrant(count=1, strike=1.0, event_date=ED,
                          units="common", series_letter="A")
        assert w.terms["units"] == "common"
        assert w.terms["series_letter"] == "A"

    def test_outstanding_seeds_initial_count(self):
        w = CreateWarrant(count=500.0, strike=1.0, event_date=ED)
        assert w.outstanding == {"count": 500.0, "initial_count": 500.0}

    def test_type_alias(self):
        w = CreateWarrant(count=1, strike=1.0, event_date=ED)
        assert w.type == "warrant"


# ───────────────────────── CreateConvertible ─────────────────────────


class TestCreateConvertible:
    def test_only_principal(self):
        c = CreateConvertible(principal=500000.0, principal_remaining=500000.0,
                              event_date=ED)
        assert c.terms == {"principal": 500000.0}

    def test_conv_discount_pct_emitted(self):
        c = CreateConvertible(principal=1.0, principal_remaining=1.0,
                              event_date=ED, conv_discount_pct=0.90)
        assert c.terms["conv_discount_pct"] == 0.90

    def test_convertible_offset_drives_date(self):
        c = CreateConvertible(principal=1.0, principal_remaining=1.0,
                              event_date=ED, convertible_offset_months=6)
        assert c.terms["convertible_date"] == "2024-07-15"

    def test_maturity_months_drives_maturity(self):
        c = CreateConvertible(principal=1.0, principal_remaining=1.0,
                              event_date=ED, maturity_months=12)
        assert c.terms["maturity"] == "2025-01-15"

    def test_known_owners_tuple_to_list(self):
        c = CreateConvertible(principal=1.0, principal_remaining=1.0,
                              event_date=ED, known_owners=("X",))
        assert c.terms["known_owners"] == ["X"]

    def test_rate_zero_still_emitted(self):
        # GOTCHA: 0.0 is not None so it IS emitted
        c = CreateConvertible(principal=1.0, principal_remaining=1.0,
                              event_date=ED, rate=0.0)
        assert c.terms["rate"] == 0.0

    def test_oid_pct_emitted(self):
        c = CreateConvertible(principal=1.0, principal_remaining=1.0,
                              event_date=ED, oid_pct=0.10)
        assert c.terms["oid_pct"] == 0.10

    def test_outstanding(self):
        c = CreateConvertible(principal=100.0, principal_remaining=42.0,
                              event_date=ED)
        assert c.outstanding == {"principal_remaining": 42.0}


# ───────────────────────── CreatePreferred ─────────────────────────


class TestCreatePreferred:
    def test_conv_price_explicit_used_ratio_persisted(self):
        p = CreatePreferred(count=100, series_letter="D", event_date=ED,
                            conv_price=5.0, conversion_ratio=12.5,
                            stated_value=100)
        # explicit conv_price used as-is; derivation NOT applied
        assert p.terms["conv_price"] == 5.0
        # but ratio still persisted verbatim
        assert p.terms["conversion_ratio"] == 12.5

    def test_stated_value_path(self):
        p = CreatePreferred(count=100, series_letter="D", event_date=ED,
                            conversion_ratio=12.5, stated_value=100)
        assert p.terms["conv_price"] == pytest.approx(8.0)  # 100 / 12.5

    def test_aggregate_liq_pref_path_iqst(self):
        # IQST Series D: 3,546,136 / 37,110 / 12.5
        p = CreatePreferred(count=37110, series_letter="D", event_date=ED,
                            conversion_ratio=12.5,
                            liquidation_preference=3546136)
        assert p.terms["conv_price"] == pytest.approx(7.6445939, rel=1e-6)

    def test_no_conv_price_when_both_falsy(self):
        # ratio truthy but stated_value falsy AND liq_pref falsy
        p = CreatePreferred(count=100, series_letter="D", event_date=ED,
                            conversion_ratio=12.5)
        assert "conv_price" not in p.terms
        assert p.terms["conversion_ratio"] == 12.5  # ratio still persisted

    def test_conversion_ratio_zero_skips_derivation_but_persists(self):
        # ratio == 0: `if cp is None and self.conversion_ratio` is falsy
        # so derivation skipped (no ZeroDivision); but `is not None` is
        # True so 0 IS persisted.
        p = CreatePreferred(count=100, series_letter="D", event_date=ED,
                            conversion_ratio=0, stated_value=100)
        assert "conv_price" not in p.terms
        assert p.terms["conversion_ratio"] == 0

    def test_count_zero_skips_liq_pref_branch(self):
        # count == 0 -> `self.count` falsy -> liq_pref branch skipped, no
        # ZeroDivision.
        p = CreatePreferred(count=0, series_letter="D", event_date=ED,
                            conversion_ratio=12.5,
                            liquidation_preference=3546136)
        assert "conv_price" not in p.terms

    def test_stated_value_emitted(self):
        p = CreatePreferred(count=1, series_letter="D", event_date=ED,
                            stated_value=1000)
        assert p.terms["stated_value"] == 1000

    def test_liquidation_preference_emitted(self):
        p = CreatePreferred(count=1, series_letter="D", event_date=ED,
                            liquidation_preference=5000)
        assert p.terms["liquidation_preference"] == 5000

    def test_dividend_rate_emitted(self):
        p = CreatePreferred(count=1, series_letter="D", event_date=ED,
                            dividend_rate=0.08)
        assert p.terms["dividend_rate"] == 0.08

    def test_known_owners_tuple_to_list(self):
        p = CreatePreferred(count=1, series_letter="D", event_date=ED,
                            known_owners=("X", "Y"))
        assert p.terms["known_owners"] == ["X", "Y"]

    def test_series_letter_always_emitted(self):
        p = CreatePreferred(count=1, series_letter="A", event_date=ED)
        assert p.terms["series_letter"] == "A"

    def test_outstanding_equal_at_create(self):
        p = CreatePreferred(count=10.0, series_letter="A", event_date=ED)
        assert p.outstanding == {"count": 10.0, "initial_count": 10.0}

    def test_outstanding_principal_remaining_added_when_set(self):
        p = CreatePreferred(count=10.0, series_letter="A", event_date=ED,
                            principal_remaining=5.0)
        assert p.outstanding == {"count": 10.0, "initial_count": 10.0,
                                 "principal_remaining": 5.0}


# ───────────────────────── CreateEquityLine ─────────────────────────


class TestCreateEquityLine:
    def test_explicit_end_wins(self):
        e = CreateEquityLine(capacity_usd=1e6, event_date=ED,
                             agreement_date=date(2024, 2, 1),
                             agreement_end_date=date(2026, 1, 1),
                             term_months=12)
        assert e.terms["agreement_end_date"] == "2026-01-01"

    def test_end_anchored_on_agreement_date(self):
        e = CreateEquityLine(capacity_usd=1e6, event_date=ED,
                             agreement_date=date(2024, 2, 1), term_months=12)
        assert e.terms["agreement_end_date"] == "2025-02-01"

    def test_end_anchored_on_event_date_when_no_agreement(self):
        e = CreateEquityLine(capacity_usd=1e6, event_date=ED, term_months=12)
        assert e.terms["agreement_end_date"] == "2025-01-15"

    def test_only_capacity(self):
        e = CreateEquityLine(capacity_usd=1e6, event_date=ED)
        assert e.terms == {"capacity_usd": 1e6}

    def test_drawn_zero_emitted(self):
        e = CreateEquityLine(capacity_usd=1e6, event_date=ED, drawn_usd=0.0)
        assert e.outstanding["drawn_usd"] == 0.0

    def test_remaining_none_omitted(self):
        e = CreateEquityLine(capacity_usd=1e6, event_date=ED)
        assert "remaining_capacity_usd" not in e.outstanding


# ───────────────────────── CreateAtm / CreateShelf / S1 / Equity ──────────


class TestCreateAtm:
    def test_only_capacity(self):
        a = CreateAtm(capacity_usd=5e6, event_date=ED)
        assert a.terms == {"capacity_usd": 5e6}

    def test_agreement_dates_isoformatted(self):
        a = CreateAtm(capacity_usd=5e6, event_date=ED,
                      agreement_date=date(2024, 2, 1),
                      agreement_end_date=date(2027, 2, 1))
        assert a.terms["agreement_date"] == "2024-02-01"
        assert a.terms["agreement_end_date"] == "2027-02-01"

    def test_drawn_zero_emitted(self):
        a = CreateAtm(capacity_usd=5e6, event_date=ED, drawn_usd=0.0)
        assert a.outstanding["drawn_usd"] == 0.0

    def test_outstanding_empty_when_none(self):
        a = CreateAtm(capacity_usd=5e6, event_date=ED)
        assert a.outstanding == {}


class TestCreateShelf:
    def test_form_and_file_number_omitted_when_none(self):
        s = CreateShelf(capacity_usd=1e8, event_date=ED)
        assert s.terms == {"capacity_usd": 1e8}

    def test_form_and_file_number_included(self):
        s = CreateShelf(capacity_usd=1e8, event_date=ED, form="S-3",
                        file_number="333-12345")
        assert s.terms == {"capacity_usd": 1e8, "form": "S-3",
                           "file_number": "333-12345"}

    def test_outstanding_remaining(self):
        s = CreateShelf(capacity_usd=1e8, event_date=ED,
                        remaining_capacity_usd=5e7)
        assert s.outstanding == {"remaining_capacity_usd": 5e7}


class TestCreateS1Offering:
    def test_only_deal_size(self):
        s = CreateS1Offering(anticipated_deal_size=2e7, event_date=ED)
        assert s.terms == {"anticipated_deal_size": 2e7}
        assert s.outstanding == {}

    def test_warrant_fields_omitted_when_none(self):
        s = CreateS1Offering(anticipated_deal_size=2e7, event_date=ED)
        assert "warrant_strike" not in s.terms
        assert "warrant_coverage_pct" not in s.terms

    def test_warrant_fields_included(self):
        s = CreateS1Offering(anticipated_deal_size=2e7, event_date=ED,
                             warrant_strike=1.5, warrant_coverage_pct=0.5)
        assert s.terms["warrant_strike"] == 1.5
        assert s.terms["warrant_coverage_pct"] == 0.5

    def test_sold_to_date_zero_emitted(self):
        s = CreateS1Offering(anticipated_deal_size=2e7, event_date=ED,
                             sold_to_date=0.0)
        assert s.outstanding == {"sold_to_date": 0.0}


class TestCreateEquity:
    def test_closing_date_isoformat(self):
        e = CreateEquity(count=1000, price_per_share=2.5, event_date=ED,
                         closing_date=date(2024, 1, 20))
        assert e.terms["closing_date"] == "2024-01-20"

    def test_closing_date_none_omitted(self):
        e = CreateEquity(count=1000, price_per_share=2.5, event_date=ED)
        assert e.terms == {"price_per_share": 2.5}

    def test_known_owners_tuple_to_list(self):
        e = CreateEquity(count=1000, price_per_share=2.5, event_date=ED,
                         known_owners=("X",))
        assert e.terms["known_owners"] == ["X"]

    def test_outstanding_count(self):
        e = CreateEquity(count=1000.0, price_per_share=2.5, event_date=ED)
        assert e.outstanding == {"count": 1000.0}


# ───────────────────────── Amend* properties ─────────────────────────


class TestAmendAtm:
    def test_all_none_empty(self):
        a = AmendAtm(instrument_id="A-1", event_date=ED)
        assert a.field_updates == {}
        assert a.outstanding_updates == {}

    def test_capacity_to_field_updates(self):
        a = AmendAtm(instrument_id="A-1", event_date=ED, capacity_usd=1e6)
        assert a.field_updates == {"capacity_usd": 1e6}

    def test_dates_isoformatted(self):
        a = AmendAtm(instrument_id="A-1", event_date=ED,
                     agreement_date=date(2024, 2, 1),
                     agreement_end_date=date(2027, 2, 1))
        assert a.field_updates["agreement_date"] == "2024-02-01"
        assert a.field_updates["agreement_end_date"] == "2027-02-01"

    def test_balances_to_outstanding(self):
        a = AmendAtm(instrument_id="A-1", event_date=ED,
                     remaining_capacity_usd=5e5, drawn_usd=5e5)
        assert a.outstanding_updates == {"remaining_capacity_usd": 5e5,
                                         "drawn_usd": 5e5}


class TestAmendShelf:
    def test_capacity_none_empty(self):
        s = AmendShelf(instrument_id="S-1", event_date=ED)
        assert s.field_updates == {}

    def test_capacity_set(self):
        s = AmendShelf(instrument_id="S-1", event_date=ED, capacity_usd=1e8)
        assert s.field_updates == {"capacity_usd": 1e8}

    def test_remaining(self):
        s = AmendShelf(instrument_id="S-1", event_date=ED,
                       remaining_capacity_usd=5e7)
        assert s.outstanding_updates == {"remaining_capacity_usd": 5e7}


class TestAmendEquityLine:
    def test_all_none_empty(self):
        e = AmendEquityLine(instrument_id="E-1", event_date=ED)
        assert e.field_updates == {}
        assert e.outstanding_updates == {}

    def test_end_date_isoformatted(self):
        e = AmendEquityLine(instrument_id="E-1", event_date=ED,
                            agreement_end_date=date(2026, 1, 1))
        assert e.field_updates["agreement_end_date"] == "2026-01-01"


class TestAmendWarrant:
    def test_count_to_outstanding_strike_to_fields(self):
        w = AmendWarrant(instrument_id="W-1", event_date=ED, count=500,
                         strike=3.0)
        assert w.outstanding_updates == {"count": 500}
        assert w.field_updates == {"strike": 3.0}

    def test_known_owners_to_list_and_dates(self):
        w = AmendWarrant(instrument_id="W-1", event_date=ED,
                         known_owners=("X", "Y"),
                         issue_date=date(2024, 1, 1),
                         exercisable_date=date(2024, 2, 1),
                         expiration=date(2029, 2, 1))
        fu = w.field_updates
        assert fu["known_owners"] == ["X", "Y"]
        assert fu["issue_date"] == "2024-01-01"
        assert fu["exercisable_date"] == "2024-02-01"
        assert fu["expiration"] == "2029-02-01"

    def test_is_pre_funded_false_emitted(self):
        w = AmendWarrant(instrument_id="W-1", event_date=ED,
                         is_pre_funded=False)
        assert w.field_updates == {"is_pre_funded": False}

    def test_all_none_empty(self):
        w = AmendWarrant(instrument_id="W-1", event_date=ED)
        assert w.field_updates == {}
        assert w.outstanding_updates == {}


class TestAmendConvertible:
    def test_principal_remaining_zero_emitted(self):
        # boundary for fully-converted notes: 0.0 is not None
        c = AmendConvertible(instrument_id="C-1", event_date=ED,
                             principal_remaining=0.0)
        assert c.outstanding_updates == {"principal_remaining": 0.0}

    def test_fields_dates_isoformatted(self):
        c = AmendConvertible(instrument_id="C-1", event_date=ED,
                             conv_price=2.0, conv_discount_pct=0.9,
                             convertible_date=date(2024, 6, 1),
                             maturity=date(2026, 1, 1))
        fu = c.field_updates
        assert fu["conv_price"] == 2.0
        assert fu["conv_discount_pct"] == 0.9
        assert fu["convertible_date"] == "2024-06-01"
        assert fu["maturity"] == "2026-01-01"

    def test_all_none_empty(self):
        c = AmendConvertible(instrument_id="C-1", event_date=ED)
        assert c.field_updates == {}
        assert c.outstanding_updates == {}


class TestAmendPreferred:
    def test_conversion_ratio_NOT_in_field_updates(self):
        # GOTCHA: conversion_ratio deliberately omitted from field_updates;
        # only the conv_price-derived value persists downstream.
        p = AmendPreferred(instrument_id="P-1", event_date=ED,
                           conversion_ratio=12.5)
        assert "conversion_ratio" not in p.field_updates
        assert p.field_updates == {}

    def test_count_and_principal_to_outstanding(self):
        p = AmendPreferred(instrument_id="P-1", event_date=ED, count=50,
                           principal_remaining=100)
        assert p.outstanding_updates == {"count": 50, "principal_remaining": 100}

    def test_fields_emitted(self):
        p = AmendPreferred(instrument_id="P-1", event_date=ED, conv_price=5.0,
                           convertible_date=date(2024, 6, 1),
                           maturity=date(2030, 1, 1), stated_value=1000,
                           liquidation_preference=5000, dividend_rate=0.08)
        fu = p.field_updates
        assert fu["conv_price"] == 5.0
        assert fu["convertible_date"] == "2024-06-01"
        assert fu["maturity"] == "2030-01-01"
        assert fu["stated_value"] == 1000
        assert fu["liquidation_preference"] == 5000
        assert fu["dividend_rate"] == 0.08

    def test_all_none_empty(self):
        p = AmendPreferred(instrument_id="P-1", event_date=ED)
        assert p.field_updates == {}
        assert p.outstanding_updates == {}


class TestAmendS1Offering:
    def test_final_fields_emitted_only_when_set(self):
        s = AmendS1Offering(instrument_id="S-1", event_date=ED,
                            final_deal_size=2e7, final_pricing=1.5,
                            final_shares_offered=1e7,
                            final_warrant_coverage_pct=0.5)
        fu = s.field_updates
        assert fu["final_deal_size"] == 2e7
        assert fu["final_pricing"] == 1.5
        assert fu["final_shares_offered"] == 1e7
        assert fu["final_warrant_coverage_pct"] == 0.5

    def test_sold_to_date_to_outstanding(self):
        s = AmendS1Offering(instrument_id="S-1", event_date=ED,
                            sold_to_date=1e6)
        assert s.outstanding_updates == {"sold_to_date": 1e6}

    def test_all_none_empty(self):
        s = AmendS1Offering(instrument_id="S-1", event_date=ED)
        assert s.field_updates == {}
        assert s.outstanding_updates == {}


class TestAmendEquity:
    def test_known_owners_none_empty(self):
        e = AmendEquity(instrument_id="E-1", event_date=ED)
        assert e.field_updates == {}
        assert e.outstanding_updates == {}

    def test_known_owners_set_to_list(self):
        e = AmendEquity(instrument_id="E-1", event_date=ED,
                        known_owners=("X", "Y"))
        assert e.field_updates == {"known_owners": ["X", "Y"]}

    def test_outstanding_always_empty(self):
        e = AmendEquity(instrument_id="E-1", event_date=ED,
                        known_owners=("X",))
        assert e.outstanding_updates == {}


# ───────────────────────── RecordDrawdown ─────────────────────────


class TestRecordDrawdown:
    def test_price_per_share_path(self):
        r = RecordDrawdown(instrument_id="A-1", drawdown_shares=100,
                           event_date=ED, price_per_share=5.0)
        f = r.fields
        assert f["drawdown_amount_usd"] == 500.0
        assert f["avg_price"] == 5.0
        assert f["drawdown_shares"] == 100

    def test_aggregate_path(self):
        r = RecordDrawdown(instrument_id="A-1", drawdown_shares=200,
                           event_date=ED, drawdown_amount_usd=1000)
        f = r.fields
        assert f["drawdown_amount_usd"] == 1000
        assert f["avg_price"] == 5.0  # 1000/200

    def test_no_price_no_amount(self):
        # amount = drawdown_amount_usd or 0.0 -> 0.0; avg = 0.0/100 = 0.0
        r = RecordDrawdown(instrument_id="A-1", drawdown_shares=100,
                           event_date=ED)
        f = r.fields
        assert f["drawdown_amount_usd"] == 0.0
        assert f["avg_price"] == 0.0

    def test_shares_zero_no_avg(self):
        # shares=0 in aggregate branch -> avg None, no ZeroDivision, key omitted
        r = RecordDrawdown(instrument_id="A-1", drawdown_shares=0,
                           event_date=ED, drawdown_amount_usd=1000)
        f = r.fields
        assert "avg_price" not in f
        assert f["drawdown_amount_usd"] == 1000

    def test_shares_negative_no_avg(self):
        r = RecordDrawdown(instrument_id="A-1", drawdown_shares=-5,
                           event_date=ED, drawdown_amount_usd=1000)
        assert "avg_price" not in r.fields

    def test_zero_amount_coalesced(self):
        # drawdown_amount_usd=0 (falsy) + price None -> `0 or 0.0` -> 0.0
        r = RecordDrawdown(instrument_id="A-1", drawdown_shares=100,
                           event_date=ED, drawdown_amount_usd=0)
        assert r.fields["drawdown_amount_usd"] == 0.0

    def test_placement_agent_emitted(self):
        r = RecordDrawdown(instrument_id="A-1", drawdown_shares=100,
                           event_date=ED, price_per_share=5.0,
                           placement_agent_canonical="HCW")
        assert r.fields["placement_agent_canonical"] == "HCW"

    def test_placement_agent_omitted(self):
        r = RecordDrawdown(instrument_id="A-1", drawdown_shares=100,
                           event_date=ED, price_per_share=5.0)
        assert "placement_agent_canonical" not in r.fields

    def test_price_takes_precedence_over_aggregate(self):
        # GOTCHA: both set -> price wins, aggregate ignored
        r = RecordDrawdown(instrument_id="A-1", drawdown_shares=100,
                           event_date=ED, price_per_share=5.0,
                           drawdown_amount_usd=99999)
        f = r.fields
        assert f["drawdown_amount_usd"] == 500.0  # 100*5, not 99999
        assert f["avg_price"] == 5.0


# ───────────────────────── other Record* fields ─────────────────────────


class TestRecordExercise:
    def test_only_shares(self):
        r = RecordExercise(instrument_id="W-1", shares=100, event_date=ED)
        assert r.fields == {"shares": 100}

    def test_cashless_warrants_exercised(self):
        r = RecordExercise(instrument_id="W-1", shares=80, event_date=ED,
                           warrants_exercised=100, price=1.0,
                           gross_proceeds=80.0)
        f = r.fields
        assert f["shares"] == 80
        assert f["warrants_exercised"] == 100
        assert f["price"] == 1.0
        assert f["gross_proceeds"] == 80.0


class TestRecordConversion:
    def test_note_path_only_principal(self):
        r = RecordConversion(instrument_id="C-1", shares_issued=1000,
                             event_date=ED, principal_converted=50000)
        f = r.fields
        assert f["shares_issued"] == 1000
        assert f["principal_converted"] == 50000
        assert "preferred_shares_converted" not in f

    def test_preferred_path(self):
        r = RecordConversion(instrument_id="P-1", shares_issued=1000,
                             event_date=ED, preferred_shares_converted=10)
        f = r.fields
        assert f["preferred_shares_converted"] == 10
        assert "principal_converted" not in f

    def test_principal_remaining_zero_emitted(self):
        # fully-converted boundary: 0.0 is not None
        r = RecordConversion(instrument_id="C-1", shares_issued=1000,
                             event_date=ED, principal_converted=50000,
                             principal_remaining=0.0)
        assert r.fields["principal_remaining"] == 0.0


class TestRecordPartialRedemption:
    def test_all_none_empty(self):
        r = RecordPartialRedemption(instrument_id="C-1", event_date=ED)
        assert r.fields == {}

    def test_note_redemption(self):
        r = RecordPartialRedemption(instrument_id="C-1", event_date=ED,
                                    principal_redeemed=10000, cash_paid=10500)
        assert r.fields == {"principal_redeemed": 10000, "cash_paid": 10500}

    def test_preferred_redemption(self):
        r = RecordPartialRedemption(instrument_id="P-1", event_date=ED,
                                    preferred_shares_redeemed=5)
        assert r.fields == {"preferred_shares_redeemed": 5}


class TestRecordPartialTermination:
    def test_always_emits_capacity(self):
        r = RecordPartialTermination(instrument_id="A-1",
                                     capacity_reduced_usd=1e6, event_date=ED)
        assert r.fields == {"capacity_reduced_usd": 1e6}

    def test_zero_capacity_emitted(self):
        r = RecordPartialTermination(instrument_id="A-1",
                                     capacity_reduced_usd=0.0, event_date=ED)
        assert r.fields == {"capacity_reduced_usd": 0.0}


class TestConfirmClosing:
    def test_all_none_empty(self):
        c = ConfirmClosing(instrument_id="W-1", event_date=ED)
        assert c.fields == {}

    def test_count_actual_zero_emitted(self):
        c = ConfirmClosing(instrument_id="W-1", event_date=ED,
                           count_actual=0.0)
        assert c.fields == {"count_actual": 0.0}

    def test_both_emitted(self):
        c = ConfirmClosing(instrument_id="W-1", event_date=ED,
                           count_actual=1000, gross_proceeds_usd=5000)
        assert c.fields == {"count_actual": 1000, "gross_proceeds_usd": 5000}


# ───────────────────────── ApplySplit.ratio ─────────────────────────


class TestApplySplitRatio:
    def test_reverse(self):
        s = ApplySplit(post=1, pre=10, direction="reverse",
                       effective_date=ED)
        assert s.ratio == pytest.approx(0.1)

    def test_forward(self):
        s = ApplySplit(post=2, pre=1, direction="forward",
                       effective_date=ED)
        assert s.ratio == 2.0

    def test_pre_zero_raises(self):
        # GOTCHA: no zero guard exists
        s = ApplySplit(post=1, pre=0, direction="reverse", effective_date=ED)
        with pytest.raises(ZeroDivisionError):
            _ = s.ratio

    def test_units_default(self):
        s = ApplySplit(post=2, pre=1, direction="forward", effective_date=ED)
        assert s.units == "common"


# ───────────────────────── _fmt_short ─────────────────────────


class TestFmtShort:
    def test_short_unchanged(self):
        assert M._fmt_short("abc") == "abc"

    def test_len_40_unchanged(self):
        s = "x" * 40
        assert M._fmt_short(s) == s

    def test_len_41_truncated(self):
        s = "x" * 41
        out = M._fmt_short(s)
        assert out == "x" * 37 + "..."
        assert len(out) == 40

    def test_non_str_coerced(self):
        assert M._fmt_short(5) == "5"
        assert M._fmt_short({"a": 1}) == "{'a': 1}"


# ───────────────────────── fmt_mutation ─────────────────────────


class TestFmtMutation:
    def test_create_with_type(self):
        w = CreateWarrant(count=10, strike=1.0, event_date=ED,
                          proposed_id="W-1", series_letter="A")
        s = M.fmt_mutation(w)
        assert s.startswith("create_instrument:warrant W-1 date=2024-01-15")
        assert "series_letter=A" in s

    def test_note_no_event(self):
        s = M.fmt_mutation(NoteNoEvent(reason="no dilution events found"))
        assert s == "note_no_event reason=no dilution events found"

    def test_apply_split_uses_effective_date(self):
        s = M.fmt_mutation(ApplySplit(post=1, pre=10, direction="reverse",
                                      effective_date=date(2024, 2, 1)))
        assert "date=2024-02-01" in s
        assert "post=1" in s and "pre=10" in s
        assert "direction=reverse" in s and "units=common" in s

    def test_proposed_id_shown_when_no_instrument_id(self):
        a = CreateAtm(capacity_usd=5e6, event_date=ED, proposed_id="ATM-9")
        assert "ATM-9" in M.fmt_mutation(a)

    def test_no_id_segment(self):
        # NoteNoEvent has neither instrument_id nor proposed_id
        s = M.fmt_mutation(NoteNoEvent(reason="r"))
        # head then detail, no id between
        assert s == "note_no_event reason=r"

    def test_grouping_prop_raising_treated_as_none(self):
        class Stub:
            kind = "x"
            instrument_id = "S-1"
            event_date = date(2024, 1, 1)

            @property
            def terms(self):
                raise RuntimeError("boom")

        s = M.fmt_mutation(Stub())
        # property raised -> caught -> treated as None; no crash
        assert "S-1" in s
        assert "date=2024-01-01" in s

    def test_empty_grouping_prop_contributes_nothing(self):
        # CreateAtm with no optionals: outstanding == {} contributes nothing
        a = CreateAtm(capacity_usd=5e6, event_date=ED)
        s = M.fmt_mutation(a)
        assert "remaining_capacity_usd" not in s

    def test_direct_attr_not_overwriting_grouping_value(self):
        # 'capacity_usd' comes from terms; the direct-attr loop must NOT
        # overwrite it (attr-not-in-detail guard) -> appears exactly once.
        a = CreateAtm(capacity_usd=5e6, event_date=ED)
        s = M.fmt_mutation(a)
        assert s.count("capacity_usd=") == 1

    def test_long_value_truncated_in_detail(self):
        # a known_owners list rendered > 40 chars gets truncated
        owners = tuple(f"Investor{i}" for i in range(20))
        w = CreateWarrant(count=1, strike=1.0, event_date=ED,
                          known_owners=owners)
        s = M.fmt_mutation(w)
        assert "..." in s


# ───────────────────────── _to_date ─────────────────────────


class TestToDate:
    def test_none(self):
        assert M._to_date(None) is None

    def test_date_passthrough(self):
        d = date(2024, 1, 15)
        assert M._to_date(d) is d

    def test_iso_string(self):
        assert M._to_date("2024-01-15") == date(2024, 1, 15)

    def test_alt_format_via_safe_date(self):
        assert M._to_date("01/15/2024") == date(2024, 1, 15)

    def test_garbage_none(self):
        assert M._to_date("garbage") is None

    def test_year_only_none(self):
        assert M._to_date("2019") is None

    def test_datetime_passthrough(self):
        # GOTCHA: datetime is a subclass of date -> returned as-is
        dt = datetime(2024, 1, 15, 10, 30)
        result = M._to_date(dt)
        assert result is dt
        assert isinstance(result, datetime)


# ───────────────────────── create_from_dict ─────────────────────────


class TestCreateFromDict:
    def test_unknown_type_raises_keyerror(self):
        with pytest.raises(KeyError):
            create_from_dict(type_="foo", terms={}, outstanding={},
                             event_date="2024-01-15")

    def test_warrant_full(self):
        w = create_from_dict(
            type_="warrant",
            terms={"strike": 2.5, "series_letter": "A",
                   "exercisable_date": "2024-03-01",
                   "known_owners": ["Alice", "Bob"]},
            outstanding={"count": 1000},
            event_date="2024-01-15",
        )
        assert isinstance(w, CreateWarrant)
        assert w.strike == 2.5
        assert w.count == 1000.0
        assert w.exercisable_date == date(2024, 3, 1)
        assert w.known_owners == ("Alice", "Bob")

    def test_missing_strike_coalesces_to_zero(self):
        # GOTCHA: 'or 0.0' coalesces; does NOT raise ValueError despite docstring
        w = create_from_dict(type_="warrant", terms={}, outstanding={},
                             event_date="2024-01-15")
        assert w.strike == 0.0
        assert w.count == 0.0

    def test_event_date_none_today_fallback(self):
        # nondeterministic source replaced by explicit capture
        today = date.today()
        w = create_from_dict(type_="warrant", terms={"strike": 1},
                             outstanding={"count": 1}, event_date=None)
        assert w.event_date == today

    def test_event_date_unparseable_today_fallback(self):
        today = date.today()
        w = create_from_dict(type_="warrant", terms={"strike": 1},
                             outstanding={"count": 1}, event_date="2019")
        assert w.event_date == today

    def test_convertible_principal_remaining_falls_back_to_principal(self):
        c = create_from_dict(type_="convertible",
                             terms={"principal": 500000}, outstanding={},
                             event_date="2024-01-15")
        assert isinstance(c, CreateConvertible)
        assert c.principal == 500000.0
        assert c.principal_remaining == 500000.0

    def test_warrant_expiration_from_maturity_key(self):
        w = create_from_dict(type_="warrant",
                             terms={"strike": 1, "maturity": "2030-01-01"},
                             outstanding={"count": 1}, event_date="2024-01-15")
        assert w.expiration == date(2030, 1, 1)

    def test_known_owners_list_to_tuple(self):
        w = create_from_dict(type_="warrant",
                             terms={"strike": 1, "known_owners": ["X"]},
                             outstanding={"count": 1}, event_date="2024-01-15")
        assert w.known_owners == ("X",)

    def test_known_owners_string_to_none(self):
        w = create_from_dict(type_="warrant",
                             terms={"strike": 1, "known_owners": "Alice"},
                             outstanding={"count": 1}, event_date="2024-01-15")
        assert w.known_owners is None

    def test_descriptor_none_not_added(self):
        # descriptor None -> not passed as kwarg (default None on dataclass)
        w = create_from_dict(type_="warrant", terms={"strike": 1},
                             outstanding={"count": 1}, event_date="2024-01-15",
                             descriptor=None)
        assert w.descriptor is None

    def test_descriptor_set_added(self):
        w = create_from_dict(type_="warrant", terms={"strike": 1},
                             outstanding={"count": 1}, event_date="2024-01-15",
                             descriptor="some note")
        assert w.descriptor == "some note"

    def test_preferred_series_letter_absent_empty_string(self):
        p = create_from_dict(type_="preferred", terms={},
                             outstanding={"count": 10}, event_date="2024-01-15")
        assert isinstance(p, CreatePreferred)
        assert p.series_letter == ""

    def test_equity_closing_date_parsed(self):
        e = create_from_dict(type_="equity",
                             terms={"price_per_share": 2.5,
                                    "closing_date": "2024-01-20"},
                             outstanding={"count": 1000},
                             event_date="2024-01-15")
        assert isinstance(e, CreateEquity)
        assert e.closing_date == date(2024, 1, 20)

    def test_atm_dispatch(self):
        a = create_from_dict(type_="atm", terms={"capacity_usd": 5e6},
                             outstanding={"drawn_usd": 1e6},
                             event_date="2024-01-15")
        assert isinstance(a, CreateAtm)
        assert a.capacity_usd == 5e6
        assert a.drawn_usd == 1e6

    def test_shelf_dispatch(self):
        s = create_from_dict(type_="shelf",
                             terms={"capacity_usd": 1e8, "form": "S-3",
                                    "file_number": "333-1"},
                             outstanding={}, event_date="2024-01-15")
        assert isinstance(s, CreateShelf)
        assert s.form == "S-3"
        assert s.file_number == "333-1"

    def test_equity_line_dispatch(self):
        e = create_from_dict(type_="equity_line",
                             terms={"capacity_usd": 1e6},
                             outstanding={}, event_date="2024-01-15")
        assert isinstance(e, CreateEquityLine)

    def test_s1_offering_dispatch(self):
        s = create_from_dict(type_="s1_offering",
                             terms={"anticipated_deal_size": 2e7},
                             outstanding={}, event_date="2024-01-15")
        assert isinstance(s, CreateS1Offering)
        assert s.anticipated_deal_size == 2e7


# ───────────────────────── amend_from_dict ─────────────────────────


class TestAmendFromDict:
    def test_unknown_type_raises_keyerror(self):
        with pytest.raises(KeyError):
            amend_from_dict(type_="foo", instrument_id="X",
                            event_date="2024-01-15")

    def test_none_updates_treated_as_empty(self):
        a = amend_from_dict(type_="atm", instrument_id="A-1",
                            field_updates=None, outstanding_updates=None,
                            event_date="2024-01-15")
        assert isinstance(a, AmendAtm)
        assert a.capacity_usd is None

    def test_unknown_key_silently_dropped(self):
        w = amend_from_dict(type_="warrant", instrument_id="W-1",
                            field_updates={"strike": 3.0, "bogus": 99},
                            outstanding_updates={"count": 500},
                            event_date="2024-01-15")
        assert w.strike == 3.0
        assert w.count == 500

    def test_warrant_count_read_from_outstanding_updates(self):
        # count lives in outstanding_updates, not field_updates
        w = amend_from_dict(type_="warrant", instrument_id="W-1",
                            field_updates={"count": 999},  # wrong slot, ignored
                            outstanding_updates={"count": 500},
                            event_date="2024-01-15")
        assert w.count == 500

    def test_preferred_conversion_ratio_passed_to_attr(self):
        p = amend_from_dict(type_="preferred", instrument_id="P-1",
                            field_updates={"conversion_ratio": 12.5},
                            event_date="2024-01-15")
        assert isinstance(p, AmendPreferred)
        assert p.conversion_ratio == 12.5
        # but it won't reach field_updates property
        assert p.field_updates == {}

    def test_event_date_none_today_fallback(self):
        today = date.today()
        a = amend_from_dict(type_="atm", instrument_id="A-1",
                            event_date=None)
        assert a.event_date == today

    def test_event_date_unparseable_today_fallback(self):
        today = date.today()
        a = amend_from_dict(type_="atm", instrument_id="A-1",
                            event_date="2019")
        assert a.event_date == today

    def test_known_owners_list_to_tuple(self):
        e = amend_from_dict(type_="equity", instrument_id="E-1",
                            field_updates={"known_owners": ["X", "Y"]},
                            event_date="2024-01-15")
        assert e.known_owners == ("X", "Y")

    def test_known_owners_non_list_to_none(self):
        e = amend_from_dict(type_="equity", instrument_id="E-1",
                            field_updates={"known_owners": "Alice"},
                            event_date="2024-01-15")
        assert e.known_owners is None

    def test_date_string_keys_parsed(self):
        w = amend_from_dict(
            type_="warrant", instrument_id="W-1",
            field_updates={"issue_date": "2024-01-01",
                           "exercisable_date": "2024-02-01",
                           "expiration": "2029-02-01"},
            event_date="2024-01-15",
        )
        assert w.issue_date == date(2024, 1, 1)
        assert w.exercisable_date == date(2024, 2, 1)
        assert w.expiration == date(2029, 2, 1)

    def test_convertible_dispatch(self):
        c = amend_from_dict(type_="convertible", instrument_id="C-1",
                            field_updates={"convertible_date": "2024-06-01",
                                           "maturity": "2026-01-01",
                                           "conv_price": 2.0},
                            outstanding_updates={"principal_remaining": 0},
                            event_date="2024-01-15")
        assert isinstance(c, AmendConvertible)
        assert c.convertible_date == date(2024, 6, 1)
        assert c.maturity == date(2026, 1, 1)
        assert c.principal_remaining == 0

    def test_shelf_dispatch(self):
        s = amend_from_dict(type_="shelf", instrument_id="S-1",
                            field_updates={"capacity_usd": 1e8},
                            outstanding_updates={"remaining_capacity_usd": 5e7},
                            event_date="2024-01-15")
        assert isinstance(s, AmendShelf)
        assert s.capacity_usd == 1e8

    def test_equity_line_dispatch(self):
        e = amend_from_dict(type_="equity_line", instrument_id="E-1",
                            field_updates={"agreement_end_date": "2026-01-01"},
                            event_date="2024-01-15")
        assert isinstance(e, AmendEquityLine)
        assert e.agreement_end_date == date(2026, 1, 1)

    def test_s1_offering_dispatch(self):
        s = amend_from_dict(type_="s1_offering", instrument_id="S-1",
                            field_updates={"final_pricing": 1.5},
                            outstanding_updates={"sold_to_date": 1e6},
                            event_date="2024-01-15")
        assert isinstance(s, AmendS1Offering)
        assert s.final_pricing == 1.5
        assert s.sold_to_date == 1e6


# ───────────────────────── mutation_to_dict ─────────────────────────


def _assert_json_roundtrip(d: dict) -> dict:
    """Confirm JSON-serializable and return the round-tripped dict."""
    return json.loads(json.dumps(d))


class TestMutationToDict:
    def test_create_branch(self):
        w = CreateWarrant(count=10, strike=1.0, event_date=ED,
                          proposed_id="W-1", counterparty_canonical="Inv")
        d = mutation_to_dict(w)
        assert d["kind"] == "create_instrument"
        assert d["type"] == "warrant"
        assert d["terms"]["strike"] == 1.0
        assert d["outstanding"] == {"count": 10, "initial_count": 10}
        assert d["event_date"] == "2024-01-15"
        assert d["proposed_id"] == "W-1"
        assert d["counterparty_canonical"] == "Inv"
        # descriptor None -> omitted
        assert "descriptor" not in d
        _assert_json_roundtrip(d)

    def test_restate_atm_branch(self):
        r = RestateAtm(predecessor_id="ATM-1", capacity_usd=1e7,
                       event_date=ED, supersede_prior=True,
                       agreement_date=date(2024, 2, 1),
                       placement_agent_canonical="HCW")
        d = mutation_to_dict(r)
        assert d["kind"] == "restate_instrument"
        assert d["type"] == "atm"
        assert d["predecessor_id"] == "ATM-1"
        assert d["capacity_usd"] == 1e7
        assert d["supersede_prior"] is True
        assert d["event_date"] == "2024-01-15"
        assert d["agreement_date"] == "2024-02-01"
        assert d["placement_agent_canonical"] == "HCW"
        _assert_json_roundtrip(d)

    def test_restate_atm_no_agreement_date_omitted(self):
        r = RestateAtm(predecessor_id="ATM-1", capacity_usd=1e7, event_date=ED)
        d = mutation_to_dict(r)
        assert "agreement_date" not in d

    def test_amend_branch(self):
        a = AmendWarrant(instrument_id="W-1", event_date=ED, strike=3.0,
                         count=500)
        d = mutation_to_dict(a)
        assert d["kind"] == "amend_instrument"
        assert d["type"] == "warrant"
        assert d["instrument_id"] == "W-1"
        assert d["field_updates"] == {"strike": 3.0}
        assert d["outstanding_updates"] == {"count": 500}
        assert d["event_date"] == "2024-01-15"
        _assert_json_roundtrip(d)

    def test_record_branch(self):
        r = RecordExercise(instrument_id="W-1", shares=100, event_date=ED,
                           price=1.0)
        d = mutation_to_dict(r)
        assert d["kind"] == "record_event"
        assert d["instrument_id"] == "W-1"
        assert d["event_kind"] == "exercise"
        assert d["fields"] == {"shares": 100, "price": 1.0}
        assert d["event_date"] == "2024-01-15"
        _assert_json_roundtrip(d)

    def test_close_instrument_includes_replaced_by_none(self):
        c = CloseInstrument(instrument_id="X-1", reason="matured",
                            event_date=ED)
        d = mutation_to_dict(c)
        assert d["kind"] == "close_instrument"
        assert d["instrument_id"] == "X-1"
        assert d["reason"] == "matured"
        # GOTCHA: replaced_by always set, even when None
        assert "replaced_by" in d
        assert d["replaced_by"] is None
        assert d["event_date"] == "2024-01-15"
        _assert_json_roundtrip(d)

    def test_apply_split_branch_no_instrument_id(self):
        s = ApplySplit(post=1, pre=10, direction="reverse",
                       effective_date=ED, units="preferred")
        d = mutation_to_dict(s)
        assert d["kind"] == "apply_split"
        assert d["post"] == 1
        assert d["pre"] == 10
        assert d["direction"] == "reverse"
        assert d["units"] == "preferred"
        assert d["effective_date"] == "2024-01-15"
        assert "instrument_id" not in d
        _assert_json_roundtrip(d)

    def test_note_no_event_branch(self):
        d = mutation_to_dict(NoteNoEvent(reason="nothing"))
        assert d == {"kind": "note_no_event", "reason": "nothing"}
        _assert_json_roundtrip(d)

    def test_all_create_subtypes_roundtrip(self):
        creates = [
            CreateAtm(capacity_usd=5e6, event_date=ED),
            CreateShelf(capacity_usd=1e8, event_date=ED),
            CreateConvertible(principal=1.0, principal_remaining=1.0,
                              event_date=ED),
            CreatePreferred(count=1, series_letter="A", event_date=ED),
            CreateEquityLine(capacity_usd=1e6, event_date=ED),
            CreateS1Offering(anticipated_deal_size=2e7, event_date=ED),
            CreateEquity(count=1, price_per_share=1.0, event_date=ED),
        ]
        for m in creates:
            d = mutation_to_dict(m)
            assert d["kind"] == "create_instrument"
            _assert_json_roundtrip(d)
