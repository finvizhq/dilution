"""Unit tests for dilution/ledger/tools/parse.py.

Pure module: tool-call → typed-mutation dispatch + arg coercion +
cross-arg validation + dedup + retryable-failure routing. No DB, no
network, no LLM. The autouse ``temp_db`` fixture from conftest still
runs (it's harmless) but is never referenced here.

ToolCall objects are faked with ``types.SimpleNamespace(name=...,
arguments=...)`` — parse_tool_calls only reads ``.name`` and
``.arguments`` so we avoid importing the provider module.
"""

from __future__ import annotations

import types
from datetime import date

import pytest

from dilution.ledger.tools import parse
from dilution.ledger.tools.parse import (
    RetryableFailure,
    EmptyAmendFailure,
    _DateParseError,
    _AmendValidationError,
)
from dilution.ledger.mutations import (
    CreateAtm, CreateWarrant, CreateConvertible, CreatePreferred,
    CreateEquity,
    AmendAtm, AmendEquity, AmendS1Offering,
    RecordConversion, RecordDrawdown, RecordPartialRedemption,
    CloseInstrument, ApplySplit, NoteNoEvent,
)


def tc(name, arguments):
    """Build a fake provider ToolCall (only .name / .arguments read)."""
    return types.SimpleNamespace(name=name, arguments=arguments)


# ─────────────────────────────────────────────────────────────────────
# _normalize_date
# ─────────────────────────────────────────────────────────────────────
class TestNormalizeDate:
    def test_none_returns_none(self):
        assert parse._normalize_date(None) is None

    def test_empty_string_returns_none(self):
        assert parse._normalize_date("") is None

    def test_int_truthy_passes_through_unchanged(self):
        # BUG-adjacent contract: guard is `not raw or not isinstance(raw,str)`
        # -> a truthy non-str (int 123) returns raw unchanged. Documented.
        assert parse._normalize_date(123) == 123

    def test_false_returns_none(self):
        assert parse._normalize_date(False) is None

    def test_zero_int_returns_none(self):
        assert parse._normalize_date(0) is None

    def test_clean_iso_fast_path(self):
        assert parse._normalize_date("2025-01-14") == "2025-01-14"

    def test_trailing_comma_stripped_then_fast_path(self):
        assert parse._normalize_date("2025-01-14,") == "2025-01-14"

    def test_leading_and_trailing_quotes_stripped(self):
        assert parse._normalize_date('"2025-01-14"') == "2025-01-14"

    def test_multi_date_picks_first_regex_match_not_later(self):
        # The docstring claims "lands on the later disclosure-window date"
        # but the code uses _ISO_DATE_RE.search which returns the FIRST
        # match. Assert the actual first-match behavior.
        assert parse._normalize_date("2025-02-28,2025-01-01") == "2025-02-28"

    def test_noisy_string_extracts_embedded_iso(self):
        assert parse._normalize_date("On 2025-03-05 we...") == "2025-03-05"

    def test_only_strip_chars_returns_none(self):
        assert parse._normalize_date(" ,.;") is None

    def test_non_iso_phrase_returns_stripped_leftover(self):
        assert parse._normalize_date("Q1 2025") == "Q1 2025"

    def test_short_non_iso_no_match_returns_stripped(self):
        assert parse._normalize_date("soon") == "soon"

    def test_info_log_on_regex_extraction_branch(self, caplog):
        with caplog.at_level("INFO", logger="dilution.ledger.tools.parse"):
            parse._normalize_date("On 2025-03-05 we...")
        assert any("date normalizer: extracted" in r.message
                   for r in caplog.records)

    def test_no_info_log_on_fast_path(self, caplog):
        with caplog.at_level("INFO", logger="dilution.ledger.tools.parse"):
            parse._normalize_date("2025-01-14")
        assert not any("date normalizer" in r.message
                       for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────
# _required_date
# ─────────────────────────────────────────────────────────────────────
class TestRequiredDate:
    def test_valid(self):
        assert parse._required_date({"d": "2025-01-14"}, "d") == date(2025, 1, 14)

    def test_missing_key_raises(self):
        with pytest.raises(_DateParseError):
            parse._required_date({}, "d")

    def test_none_value_raises(self):
        with pytest.raises(_DateParseError):
            parse._required_date({"d": None}, "d")

    def test_empty_string_raises(self):
        with pytest.raises(_DateParseError):
            parse._required_date({"d": ""}, "d")

    def test_non_iso_phrase_raises(self):
        with pytest.raises(_DateParseError):
            parse._required_date({"d": "Q1 2025"}, "d")

    def test_out_of_range_components_raise(self):
        # Regex matches the YYYY-MM-DD pattern but fromisoformat rejects
        # month 13 / day 40.
        with pytest.raises(_DateParseError):
            parse._required_date({"d": "2025-13-40"}, "d")

    def test_multi_date_picks_first(self):
        assert parse._required_date(
            {"d": "2025-02-28,2025-01-01"}, "d") == date(2025, 2, 28)

    def test_error_message_contains_original_and_normalized(self):
        with pytest.raises(_DateParseError) as ei:
            parse._required_date({"d": "Q1 2025"}, "d")
        msg = str(ei.value)
        assert "Q1 2025" in msg

    def test_dateparseerror_is_valueerror_subclass(self):
        with pytest.raises(ValueError):
            parse._required_date({}, "d")


# ─────────────────────────────────────────────────────────────────────
# _opt_date
# ─────────────────────────────────────────────────────────────────────
class TestOptDate:
    def test_missing_key(self):
        assert parse._opt_date({}, "k") is None

    def test_none_value(self):
        assert parse._opt_date({"k": None}, "k") is None

    def test_empty_string(self):
        assert parse._opt_date({"k": ""}, "k") is None

    def test_valid(self):
        assert parse._opt_date({"k": "2025-01-14"}, "k") == date(2025, 1, 14)

    def test_not_a_date_swallowed_to_none(self):
        assert parse._opt_date({"k": "not-a-date"}, "k") is None

    def test_out_of_range_swallowed_to_none(self):
        assert parse._opt_date({"k": "2025-99-99"}, "k") is None

    def test_multi_date_routes_through_normalizer_first_match(self):
        # _opt_date normalizes before fromisoformat, so a noisy multi-date
        # string yields the FIRST embedded ISO date, not None.
        assert parse._opt_date(
            {"k": "2025-02-28,2025-01-01"}, "k") == date(2025, 2, 28)

    def test_noisy_prefixed_iso_extracted(self):
        assert parse._opt_date(
            {"k": "On 2025-03-05 we filed"}, "k") == date(2025, 3, 5)


# ─────────────────────────────────────────────────────────────────────
# _opt_float
# ─────────────────────────────────────────────────────────────────────
class TestOptFloat:
    def test_none(self):
        assert parse._opt_float(None) is None

    def test_zero_is_kept_as_float(self):
        # Load-bearing: 0 is falsey but a valid 'set' value.
        assert parse._opt_float(0) == 0.0
        assert parse._opt_float(0) is not None

    def test_numeric_string(self):
        assert parse._opt_float("3.5") == pytest.approx(3.5)

    def test_negative_string(self):
        assert parse._opt_float("-5") == pytest.approx(-5.0)

    def test_uncoercible_string_to_none(self):
        assert parse._opt_float("abc") is None

    @pytest.mark.parametrize("v", [[], {}])
    def test_typeerror_to_none(self, v):
        assert parse._opt_float(v) is None

    def test_bool_true_coerces_to_one(self):
        assert parse._opt_float(True) == 1.0


# ─────────────────────────────────────────────────────────────────────
# _opt_int
# ─────────────────────────────────────────────────────────────────────
class TestOptInt:
    def test_none(self):
        assert parse._opt_int(None) is None

    def test_zero_kept(self):
        assert parse._opt_int(0) == 0
        assert parse._opt_int(0) is not None

    def test_numeric_string(self):
        assert parse._opt_int("12") == 12

    def test_float_truncates(self):
        assert parse._opt_int(12.9) == 12

    def test_numeric_string_with_decimal_to_none(self):
        # int('12.9') raises ValueError -> None
        assert parse._opt_int("12.9") is None

    def test_uncoercible_to_none(self):
        assert parse._opt_int("abc") is None

    def test_list_to_none(self):
        assert parse._opt_int([]) is None


# ─────────────────────────────────────────────────────────────────────
# _opt_str
# ─────────────────────────────────────────────────────────────────────
class TestOptStr:
    def test_none(self):
        assert parse._opt_str(None) is None

    def test_strips_surrounding_whitespace(self):
        assert parse._opt_str("  hi  ") == "hi"

    def test_whitespace_only_to_none(self):
        assert parse._opt_str("   ") is None

    def test_empty_to_none(self):
        assert parse._opt_str("") is None

    def test_int_stringified(self):
        assert parse._opt_str(123) == "123"

    def test_zero_kept_as_string(self):
        # str(0) == '0' is truthy, so zero survives.
        assert parse._opt_str(0) == "0"


# ─────────────────────────────────────────────────────────────────────
# _opt_bool
# ─────────────────────────────────────────────────────────────────────
class TestOptBool:
    def test_none(self):
        assert parse._opt_bool(None) is None

    def test_real_bools_pass_through(self):
        assert parse._opt_bool(True) is True
        assert parse._opt_bool(False) is False

    @pytest.mark.parametrize("s", ["true", "TRUE", "Yes", "1"])
    def test_truthy_strings(self, s):
        assert parse._opt_bool(s) is True

    @pytest.mark.parametrize("s", ["false", "no", "0", ""])
    def test_falsey_strings(self, s):
        assert parse._opt_bool(s) is False

    def test_two_string_not_in_allow_set(self):
        assert parse._opt_bool("2") is False

    @pytest.mark.parametrize("v,expected", [(0, False), (1, True),
                                            ([], False), (["x"], True)])
    def test_non_string_bool_fallback(self, v, expected):
        assert parse._opt_bool(v) is expected


# ─────────────────────────────────────────────────────────────────────
# _opt_str_tuple
# ─────────────────────────────────────────────────────────────────────
class TestOptStrTuple:
    def test_none(self):
        assert parse._opt_str_tuple(None) is None

    def test_str_input_is_not_list_or_tuple(self):
        # A bare string is not list/tuple -> None (would otherwise iterate
        # characters).
        assert parse._opt_str_tuple("foo") is None

    def test_empty_list_to_none(self):
        assert parse._opt_str_tuple([]) is None

    def test_blanks_dropped(self):
        assert parse._opt_str_tuple(["  a ", "", "  "]) == ("a",)

    def test_normal_list(self):
        assert parse._opt_str_tuple(["a", "b"]) == ("a", "b")

    def test_elements_stringified(self):
        assert parse._opt_str_tuple([1, 2]) == ("1", "2")

    def test_tuple_input_stripped(self):
        assert parse._opt_str_tuple((" x ",)) == ("x",)

    def test_all_blank_to_none(self):
        assert parse._opt_str_tuple(["   "]) is None


# ─────────────────────────────────────────────────────────────────────
# _cp_from_known_owners
# ─────────────────────────────────────────────────────────────────────
class TestCpFromKnownOwners:
    def test_none(self):
        assert parse._cp_from_known_owners(None) is None

    def test_empty_tuple(self):
        assert parse._cp_from_known_owners(()) is None

    def test_single_owner_collapses_to_scalar(self):
        assert parse._cp_from_known_owners(("Streeterville",)) == "Streeterville"

    def test_multi_owner_leaves_null(self):
        assert parse._cp_from_known_owners(("A", "B")) is None


# ─────────────────────────────────────────────────────────────────────
# _check_amend_non_empty
# ─────────────────────────────────────────────────────────────────────
class _FakeAmend:
    instrument_id = "INST-1"


class TestCheckAmendNonEmpty:
    def test_zero_raises(self):
        with pytest.raises(_AmendValidationError) as ei:
            parse._check_amend_non_empty(_FakeAmend(), 0)
        msg = str(ei.value)
        assert "_FakeAmend" in msg
        assert "INST-1" in msg

    def test_one_does_not_raise(self):
        parse._check_amend_non_empty(_FakeAmend(), 1)

    def test_negative_does_not_raise(self):
        # Only ==0 raises; negative (which shouldn't happen) is a no-op.
        parse._check_amend_non_empty(_FakeAmend(), -1)

    def test_amendvalidationerror_is_valueerror(self):
        with pytest.raises(ValueError):
            parse._check_amend_non_empty(_FakeAmend(), 0)


# ─────────────────────────────────────────────────────────────────────
# _build_apply_split
# ─────────────────────────────────────────────────────────────────────
ED = {"effective_date": "2025-01-14"}


class TestBuildApplySplit:
    def test_both_shapes_present_raises(self):
        with pytest.raises(ValueError, match="not both"):
            parse._build_apply_split(
                {"post": 1, "pre": 10, "direction": "reverse",
                 "ads_ratio_from": 400, "ads_ratio_to": 4000, **ED})

    def test_neither_shape_lists_missing(self):
        with pytest.raises(ValueError, match="missing required arg"):
            parse._build_apply_split({**ED})

    def test_ads_non_numeric_raises(self):
        with pytest.raises(ValueError, match="numeric"):
            parse._build_apply_split(
                {"ads_ratio_from": "x", "ads_ratio_to": 4000, **ED})

    @pytest.mark.parametrize("af,at", [(0, 4000), (400, 0), (-1, 4000)])
    def test_ads_non_positive_raises(self, af, at):
        with pytest.raises(ValueError, match="positive"):
            parse._build_apply_split(
                {"ads_ratio_from": af, "ads_ratio_to": at, **ED})

    def test_ads_equal_is_noop_raises(self):
        with pytest.raises(ValueError, match="no-op"):
            parse._build_apply_split(
                {"ads_ratio_from": 400, "ads_ratio_to": 400, **ED})

    def test_ads_reverse_400_to_4000(self):
        m = parse._build_apply_split(
            {"ads_ratio_from": 400, "ads_ratio_to": 4000, **ED})
        assert (m.post, m.pre, m.direction) == (1, 10, "reverse")
        assert m.units == "ads"
        assert m.effective_date == date(2025, 1, 14)

    def test_ads_forward_4000_to_400(self):
        m = parse._build_apply_split(
            {"ads_ratio_from": 4000, "ads_ratio_to": 400, **ED})
        assert (m.post, m.pre, m.direction) == (10, 1, "forward")

    def test_ads_non_integer_factor_raises(self):
        with pytest.raises(ValueError, match="non-integer"):
            parse._build_apply_split(
                {"ads_ratio_from": 100, "ads_ratio_to": 150, **ED})

    def test_ads_integer_within_tolerance(self):
        m = parse._build_apply_split(
            {"ads_ratio_from": 3, "ads_ratio_to": 9, **ED})
        assert (m.post, m.pre, m.direction) == (1, 3, "reverse")

    def test_ads_units_arg_overrides_default(self):
        m = parse._build_apply_split(
            {"ads_ratio_from": 400, "ads_ratio_to": 4000,
             "units": "common", **ED})
        assert m.units == "common"

    def test_explicit_reverse_ok(self):
        m = parse._build_apply_split(
            {"post": 1, "pre": 10, "direction": "reverse", **ED})
        assert (m.post, m.pre, m.direction) == (1, 10, "reverse")
        assert m.units == "common"

    def test_explicit_forward_ok(self):
        m = parse._build_apply_split(
            {"post": 10, "pre": 1, "direction": "forward", **ED})
        assert (m.post, m.pre, m.direction) == (10, 1, "forward")

    def test_explicit_reverse_post_ge_pre_raises(self):
        with pytest.raises(ValueError, match="reverse split requires post<pre"):
            parse._build_apply_split(
                {"post": 10, "pre": 1, "direction": "reverse", **ED})

    def test_explicit_forward_post_le_pre_raises(self):
        with pytest.raises(ValueError, match="forward split requires post>pre"):
            parse._build_apply_split(
                {"post": 1, "pre": 10, "direction": "forward", **ED})

    def test_explicit_units_override(self):
        m = parse._build_apply_split(
            {"post": 1, "pre": 10, "direction": "reverse",
             "units": "ads", **ED})
        assert m.units == "ads"

    def test_numeric_string_post_pre_coerced(self):
        m = parse._build_apply_split(
            {"post": "1", "pre": "10", "direction": "reverse", **ED})
        assert (m.post, m.pre) == (1, 10)

    def test_non_int_post_raises_valueerror(self):
        with pytest.raises(ValueError):
            parse._build_apply_split(
                {"post": "x", "pre": "10", "direction": "reverse", **ED})

    def test_missing_effective_date_raises_dateparseerror(self):
        # _required_date error propagates uncaught from this builder.
        with pytest.raises(_DateParseError):
            parse._build_apply_split(
                {"post": 1, "pre": 10, "direction": "reverse"})

    def test_unknown_direction_bypasses_both_invariants(self):
        # The post/pre-vs-direction invariants only guard the literal
        # "reverse"/"forward" branches; an unrecognized direction string
        # matches neither and is built verbatim with no validation.
        # Pins the (intended) absence of a direction whitelist here.
        m = parse._build_apply_split(
            {"post": 10, "pre": 1, "direction": "sideways", **ED})
        assert m.direction == "sideways"
        assert (m.post, m.pre) == (10, 1)

    def test_ads_path_ignores_explicit_direction_string(self):
        # ads_ratio_* fully derives direction; a stray 'direction' key is
        # not part of the explicit-shape trigger only because post/pre are
        # absent here, but its presence still trips the "not both" guard.
        with pytest.raises(ValueError, match="not both"):
            parse._build_apply_split(
                {"ads_ratio_from": 400, "ads_ratio_to": 4000,
                 "direction": "reverse", **ED})


# ─────────────────────────────────────────────────────────────────────
# _build_close_instrument
# ─────────────────────────────────────────────────────────────────────
class TestBuildCloseInstrument:
    def test_superseded_without_replaced_by_raises(self):
        with pytest.raises(ValueError, match="requires replaced_by"):
            parse._build_close_instrument(
                {"instrument_id": "X", "reason": "superseded",
                 "event_date": "2025-01-14"})

    def test_superseded_with_replaced_by_kept(self):
        m = parse._build_close_instrument(
            {"instrument_id": "X", "reason": "superseded",
             "replaced_by": "Y", "event_date": "2025-01-14"})
        assert m.reason == "superseded"
        assert m.replaced_by == "Y"

    def test_non_superseded_with_replaced_by_dropped_and_warns(self, caplog):
        with caplog.at_level("WARNING", logger="dilution.ledger.tools.parse"):
            m = parse._build_close_instrument(
                {"instrument_id": "X", "reason": "matured",
                 "replaced_by": "Y", "event_date": "2025-01-14"})
        assert m.reason == "matured"
        assert m.replaced_by is None
        assert any("replaced_by" in r.message and "ignoring" in r.message
                   for r in caplog.records)

    def test_matured_without_replaced_by_ok(self):
        m = parse._build_close_instrument(
            {"instrument_id": "X", "reason": "matured",
             "event_date": "2025-01-14"})
        assert m.reason == "matured"
        assert m.replaced_by is None

    def test_reason_whitespace_stripped(self):
        m = parse._build_close_instrument(
            {"instrument_id": "X", "reason": "  matured  ",
             "event_date": "2025-01-14"})
        assert m.reason == "matured"

    def test_missing_reason_raises_keyerror(self):
        with pytest.raises(KeyError):
            parse._build_close_instrument(
                {"instrument_id": "X", "event_date": "2025-01-14"})

    def test_missing_event_date_raises_dateparseerror(self):
        with pytest.raises(_DateParseError):
            parse._build_close_instrument(
                {"instrument_id": "X", "reason": "matured"})


# ─────────────────────────────────────────────────────────────────────
# _build_note_no_event
# ─────────────────────────────────────────────────────────────────────
class TestBuildNoteNoEvent:
    def test_reason_kept(self):
        m = parse._build_note_no_event({"reason": "no dilutive events"})
        assert isinstance(m, NoteNoEvent)
        assert m.reason == "no dilutive events"

    def test_reason_whitespace_stripped(self):
        assert parse._build_note_no_event(
            {"reason": "  done  "}).reason == "done"

    def test_missing_reason_raises_keyerror(self):
        # No event_date arg at all — proves note_no_event takes no date.
        with pytest.raises(KeyError):
            parse._build_note_no_event({})

    def test_int_reason_stringified(self):
        # str(args["reason"]) coerces a non-string scalar.
        assert parse._build_note_no_event({"reason": 5}).reason == "5"


# ─────────────────────────────────────────────────────────────────────
# _build_record_drawdown
# ─────────────────────────────────────────────────────────────────────
DRAW_BASE = {"instrument_id": "X", "drawdown_shares": 100,
             "event_date": "2025-01-14"}


class TestBuildRecordDrawdown:
    def test_price_kept_when_positive(self):
        m = parse._build_record_drawdown({**DRAW_BASE, "price_per_share": 5})
        assert m.price_per_share == 5.0

    def test_amount_only_ok(self):
        m = parse._build_record_drawdown(
            {**DRAW_BASE, "drawdown_amount_usd": 1000})
        assert m.price_per_share is None
        assert m.drawdown_amount_usd == 1000.0

    def test_zero_price_coerced_none_with_amount(self):
        m = parse._build_record_drawdown(
            {**DRAW_BASE, "price_per_share": 0, "drawdown_amount_usd": 1000})
        assert m.price_per_share is None
        assert m.drawdown_amount_usd == 1000.0

    def test_negative_price_coerced_none_with_amount(self):
        m = parse._build_record_drawdown(
            {**DRAW_BASE, "price_per_share": -1, "drawdown_amount_usd": 1000})
        assert m.price_per_share is None

    def test_zero_price_no_amount_raises(self):
        with pytest.raises(ValueError, match="got neither"):
            parse._build_record_drawdown({**DRAW_BASE, "price_per_share": 0})

    def test_neither_price_nor_amount_raises(self):
        with pytest.raises(ValueError, match="got neither"):
            parse._build_record_drawdown({**DRAW_BASE})

    def test_missing_drawdown_shares_raises_keyerror(self):
        with pytest.raises(KeyError):
            parse._build_record_drawdown(
                {"instrument_id": "X", "price_per_share": 5,
                 "event_date": "2025-01-14"})

    def test_price_and_amount_both_kept(self):
        m = parse._build_record_drawdown(
            {**DRAW_BASE, "price_per_share": 5, "drawdown_amount_usd": 600})
        assert m.price_per_share == 5.0
        assert m.drawdown_amount_usd == 600.0


# ─────────────────────────────────────────────────────────────────────
# _build_record_conversion
# ─────────────────────────────────────────────────────────────────────
CONV_BASE = {"instrument_id": "X", "shares_issued": 100,
             "event_date": "2025-01-14"}


class TestBuildRecordConversion:
    def test_neither_raises(self):
        with pytest.raises(ValueError, match="got neither"):
            parse._build_record_conversion({**CONV_BASE})

    def test_principal_only(self):
        m = parse._build_record_conversion(
            {**CONV_BASE, "principal_converted": 5000})
        assert m.principal_converted == 5000.0
        assert m.preferred_shares_converted is None

    def test_preferred_only(self):
        m = parse._build_record_conversion(
            {**CONV_BASE, "preferred_shares_converted": 12})
        assert m.preferred_shares_converted == 12.0
        assert m.principal_converted is None

    def test_both_set_ok(self):
        # Builder only requires >=1; exclusivity gated in validate.py.
        m = parse._build_record_conversion(
            {**CONV_BASE, "principal_converted": 5, "preferred_shares_converted": 3})
        assert m.principal_converted == 5.0
        assert m.preferred_shares_converted == 3.0

    def test_missing_shares_issued_raises_keyerror(self):
        with pytest.raises(KeyError):
            parse._build_record_conversion(
                {"instrument_id": "X", "principal_converted": 5,
                 "event_date": "2025-01-14"})

    def test_principal_zero_passes_guard(self):
        # _opt_float keeps 0.0, so 0 is not-None and satisfies the guard.
        m = parse._build_record_conversion(
            {**CONV_BASE, "principal_converted": 0})
        assert m.principal_converted == 0.0


# ─────────────────────────────────────────────────────────────────────
# _build_record_partial_redemption
# ─────────────────────────────────────────────────────────────────────
REDEEM_BASE = {"instrument_id": "X", "event_date": "2025-01-14"}


class TestBuildRecordPartialRedemption:
    def test_neither_raises(self):
        with pytest.raises(ValueError, match="got neither"):
            parse._build_record_partial_redemption({**REDEEM_BASE})

    def test_principal_only(self):
        m = parse._build_record_partial_redemption(
            {**REDEEM_BASE, "principal_redeemed": 5000})
        assert m.principal_redeemed == 5000.0

    def test_preferred_only(self):
        m = parse._build_record_partial_redemption(
            {**REDEEM_BASE, "preferred_shares_redeemed": 7})
        assert m.preferred_shares_redeemed == 7.0

    def test_zero_value_passes_guard(self):
        m = parse._build_record_partial_redemption(
            {**REDEEM_BASE, "principal_redeemed": 0})
        assert m.principal_redeemed == 0.0

    def test_cash_paid_optional_none(self):
        m = parse._build_record_partial_redemption(
            {**REDEEM_BASE, "principal_redeemed": 5})
        assert m.cash_paid is None

    def test_cash_paid_kept(self):
        m = parse._build_record_partial_redemption(
            {**REDEEM_BASE, "principal_redeemed": 5, "cash_paid": 1234})
        assert m.cash_paid == 1234.0


# ─────────────────────────────────────────────────────────────────────
# _build_amend_atm (representative of all amend_* builders)
# ─────────────────────────────────────────────────────────────────────
class TestBuildAmendAtm:
    def test_no_mutating_fields_raises(self):
        with pytest.raises(_AmendValidationError):
            parse._build_amend_atm(
                {"instrument_id": "X", "event_date": "2025-01-14"})

    def test_one_mutating_field_ok(self):
        m = parse._build_amend_atm(
            {"instrument_id": "X", "event_date": "2025-01-14",
             "capacity_usd": 1})
        assert m.capacity_usd == 1.0

    def test_drawn_zero_counts_as_set(self):
        # 0.0 is not None -> satisfies non-empty.
        m = parse._build_amend_atm(
            {"instrument_id": "X", "event_date": "2025-01-14",
             "drawn_usd": 0})
        assert m.drawn_usd == 0.0

    def test_instrument_id_whitespace_stripped(self):
        m = parse._build_amend_atm(
            {"instrument_id": "  X  ", "event_date": "2025-01-14",
             "capacity_usd": 1})
        assert m.instrument_id == "X"

    def test_missing_event_date_raises_dateparseerror(self):
        with pytest.raises(_DateParseError):
            parse._build_amend_atm(
                {"instrument_id": "X", "capacity_usd": 1})

    def test_agreement_date_counts_as_mutating(self):
        m = parse._build_amend_atm(
            {"instrument_id": "X", "event_date": "2025-01-14",
             "agreement_date": "2025-01-01"})
        assert m.agreement_date == date(2025, 1, 1)


# ─────────────────────────────────────────────────────────────────────
# _build_amend_equity (smallest amend — only known_owners is mutating)
# ─────────────────────────────────────────────────────────────────────
class TestBuildAmendEquity:
    def test_no_known_owners_raises(self):
        with pytest.raises(_AmendValidationError):
            parse._build_amend_equity(
                {"instrument_id": "X", "event_date": "2025-01-14"})

    def test_empty_list_does_not_count(self):
        # _opt_str_tuple([]) -> None -> empty amend.
        with pytest.raises(_AmendValidationError):
            parse._build_amend_equity(
                {"instrument_id": "X", "event_date": "2025-01-14",
                 "known_owners": []})

    def test_owner_present_ok(self):
        m = parse._build_amend_equity(
            {"instrument_id": "X", "event_date": "2025-01-14",
             "known_owners": ["Alice"]})
        assert m.known_owners == ("Alice",)


# ─────────────────────────────────────────────────────────────────────
# _build_amend_s1_offering (priced-cover fields regression)
# ─────────────────────────────────────────────────────────────────────
class TestBuildAmendS1Offering:
    def test_only_final_deal_size_not_rejected(self):
        m = parse._build_amend_s1_offering(
            {"instrument_id": "X", "event_date": "2025-01-14",
             "final_deal_size": 1000})
        assert m.final_deal_size == 1000.0

    def test_only_final_pricing_ok(self):
        m = parse._build_amend_s1_offering(
            {"instrument_id": "X", "event_date": "2025-01-14",
             "final_pricing": 5})
        assert m.final_pricing == 5.0

    def test_none_of_nine_raises(self):
        with pytest.raises(_AmendValidationError):
            parse._build_amend_s1_offering(
                {"instrument_id": "X", "event_date": "2025-01-14"})

    def test_final_zero_counts_as_set(self):
        m = parse._build_amend_s1_offering(
            {"instrument_id": "X", "event_date": "2025-01-14",
             "final_deal_size": 0})
        assert m.final_deal_size == 0.0


# ─────────────────────────────────────────────────────────────────────
# _build_create_atm (representative create builder)
# ─────────────────────────────────────────────────────────────────────
CREATE_ATM_BASE = {"capacity_usd": 1_000_000,
                   "placement_agent_canonical": "HCW",
                   "agreement_date": "2025-01-01",
                   "event_date": "2025-01-14"}


class TestBuildCreateAtm:
    def test_missing_capacity_raises_keyerror(self):
        args = {k: v for k, v in CREATE_ATM_BASE.items()
                if k != "capacity_usd"}
        with pytest.raises(KeyError):
            parse._build_create_atm(args)

    def test_capacity_non_numeric_raises_valueerror(self):
        with pytest.raises(ValueError):
            parse._build_create_atm({**CREATE_ATM_BASE, "capacity_usd": "abc"})

    def test_missing_placement_agent_raises_keyerror(self):
        args = {k: v for k, v in CREATE_ATM_BASE.items()
                if k != "placement_agent_canonical"}
        with pytest.raises(KeyError):
            parse._build_create_atm(args)

    def test_missing_event_date_raises_dateparseerror(self):
        args = {k: v for k, v in CREATE_ATM_BASE.items()
                if k != "event_date"}
        with pytest.raises(_DateParseError):
            parse._build_create_atm(args)

    def test_agreement_end_date_optional_none(self):
        m = parse._build_create_atm({**CREATE_ATM_BASE})
        assert m.agreement_end_date is None

    def test_placement_agent_whitespace_stripped(self):
        m = parse._build_create_atm(
            {**CREATE_ATM_BASE, "placement_agent_canonical": "  HCW  "})
        assert m.placement_agent_canonical == "HCW"
        assert m.capacity_usd == 1_000_000.0
        assert m.agreement_date == date(2025, 1, 1)


# ─────────────────────────────────────────────────────────────────────
# _build_create_warrant (known_owners collapse path)
# ─────────────────────────────────────────────────────────────────────
WARRANT_BASE = {"count": 100, "strike": 1.5, "event_date": "2025-01-14"}


class TestBuildCreateWarrant:
    def test_single_owner_collapses_to_counterparty(self):
        m = parse._build_create_warrant(
            {**WARRANT_BASE, "known_owners": ["Solo"]})
        assert m.counterparty_canonical == "Solo"
        assert m.known_owners == ("Solo",)

    def test_multi_owner_counterparty_none(self):
        m = parse._build_create_warrant(
            {**WARRANT_BASE, "known_owners": ["A", "B"]})
        assert m.counterparty_canonical is None
        assert m.known_owners == ("A", "B")

    def test_no_known_owners_both_none(self):
        m = parse._build_create_warrant({**WARRANT_BASE})
        assert m.counterparty_canonical is None
        assert m.known_owners is None

    def test_missing_count_raises_keyerror(self):
        with pytest.raises(KeyError):
            parse._build_create_warrant(
                {"strike": 1.5, "event_date": "2025-01-14"})

    def test_missing_strike_raises_keyerror(self):
        with pytest.raises(KeyError):
            parse._build_create_warrant(
                {"count": 100, "event_date": "2025-01-14"})

    def test_is_pre_funded_yes_coerced_true(self):
        m = parse._build_create_warrant(
            {**WARRANT_BASE, "is_pre_funded": "yes"})
        assert m.is_pre_funded is True


# ─────────────────────────────────────────────────────────────────────
# parse_tool_calls — dispatcher
# ─────────────────────────────────────────────────────────────────────
class TestParseToolCalls:
    def test_empty_returns_empty_list(self):
        assert parse.parse_tool_calls([], accession="ACC") == []

    def test_none_returns_empty_list(self):
        assert parse.parse_tool_calls(None, accession="ACC") == []

    def test_byte_identical_calls_deduped(self, caplog):
        calls = [tc("note_no_event", {"reason": "a"}),
                 tc("note_no_event", {"reason": "a"})]
        with caplog.at_level("WARNING", logger="dilution.ledger.tools.parse"):
            out = parse.parse_tool_calls(calls, accession="ACC")
        assert len(out) == 1
        assert any("collapsed 1 duplicate" in r.message for r in caplog.records)

    def test_dedup_ignores_key_order(self):
        # Same args, different insertion order -> the fingerprint uses
        # json.dumps(sort_keys=True), so the two calls collapse to one.
        # note_no_event reads only 'reason' (the extra 'instrument_id' is
        # ignored), so the survivor builds successfully into exactly one
        # NoteNoEvent. (Previously this only asserted len <= 1, which a
        # broken dedup that dropped BOTH calls would also satisfy.)
        a = {"instrument_id": "X", "reason": "a"}
        b = {"reason": "a", "instrument_id": "X"}
        calls = [tc("note_no_event", a), tc("note_no_event", b)]
        out = parse.parse_tool_calls(calls, accession="ACC")
        assert len(out) == 1
        assert isinstance(out[0], NoteNoEvent)
        assert out[0].reason == "a"

    def test_dedup_repr_fallback_on_unserializable_args(self, caplog):
        # A self-referential dict makes json.dumps raise ValueError; the
        # fingerprint falls back to repr(args) and still dedups.
        circ: dict = {}
        circ["self"] = circ
        calls = [tc("note_no_event", circ), tc("note_no_event", circ)]
        with caplog.at_level("WARNING", logger="dilution.ledger.tools.parse"):
            out = parse.parse_tool_calls(calls, accession="ACC")
        # Both fail to build (no 'reason' key) but the dedup must collapse
        # them to a single attempt.
        assert out == []
        assert any("collapsed 1 duplicate" in r.message for r in caplog.records)

    def test_unknown_tool_dropped_and_warns(self, caplog):
        with caplog.at_level("WARNING", logger="dilution.ledger.tools.parse"):
            out = parse.parse_tool_calls(
                [tc("frobnicate", {})], accession="ACC")
        assert out == []
        assert any("unknown tool 'frobnicate'" in r.message
                   for r in caplog.records)

    def test_raw_arguments_marker_dropped_and_warns(self, caplog):
        with caplog.at_level("WARNING", logger="dilution.ledger.tools.parse"):
            out = parse.parse_tool_calls(
                [tc("create_atm", {"__raw_arguments__": "broken json"})],
                accession="ACC")
        assert out == []
        assert any("failed JSON decode" in r.message for r in caplog.records)

    def test_empty_amend_appends_retryable_failure(self):
        ea: list[RetryableFailure] = []
        out = parse.parse_tool_calls(
            [tc("amend_atm", {"instrument_id": "X",
                              "event_date": "2025-01-14"})],
            accession="ACC", empty_amends=ea)
        assert out == []
        assert len(ea) == 1
        assert ea[0].kind == "empty_amend"
        assert ea[0].tool_name == "amend_atm"
        assert ea[0].instrument_id == "X"
        assert ea[0].event_date == "2025-01-14"

    def test_empty_amend_with_none_drops_no_crash(self):
        out = parse.parse_tool_calls(
            [tc("amend_atm", {"instrument_id": "X",
                              "event_date": "2025-01-14"})],
            accession="ACC", empty_amends=None)
        assert out == []

    def test_bad_date_appends_retryable_failure(self):
        ea: list[RetryableFailure] = []
        out = parse.parse_tool_calls(
            [tc("create_atm", {"capacity_usd": 1e6,
                               "placement_agent_canonical": "A",
                               "agreement_date": "2025-01-14",
                               "event_date": "Q1 2025"})],
            accession="ACC", empty_amends=ea)
        assert out == []
        assert len(ea) == 1
        assert ea[0].kind == "bad_date"
        assert ea[0].tool_name == "create_atm"

    def test_bad_date_event_date_none_captured(self):
        # amend_atm with a mutating field but missing event_date -> bad_date,
        # and the captured event_date is None.
        ea: list[RetryableFailure] = []
        parse.parse_tool_calls(
            [tc("amend_atm", {"instrument_id": "X", "capacity_usd": 1})],
            accession="ACC", empty_amends=ea)
        assert len(ea) == 1
        assert ea[0].kind == "bad_date"
        assert ea[0].event_date is None

    def test_generic_valueerror_dropped_not_appended(self):
        # record_drawdown neither price nor amount -> plain ValueError ->
        # silently dropped, NOT routed to empty_amends.
        ea: list[RetryableFailure] = []
        out = parse.parse_tool_calls(
            [tc("record_drawdown", {"instrument_id": "X",
                                    "drawdown_shares": 100,
                                    "event_date": "2025-01-14"})],
            accession="ACC", empty_amends=ea)
        assert out == []
        assert ea == []

    def test_drawdown_missing_price_guard_first_pass(self):
        ea: list[RetryableFailure] = []
        out = parse.parse_tool_calls(
            [tc("record_drawdown", {"instrument_id": "X",
                                    "drawdown_shares": 100,
                                    "drawdown_amount_usd": 1000,
                                    "event_date": "2025-01-14"})],
            accession="ACC", empty_amends=ea)
        assert out == []
        assert len(ea) == 1
        assert ea[0].kind == "drawdown_missing_price"
        assert ea[0].instrument_id == "X"
        assert ea[0].event_date == "2025-01-14"

    def test_drawdown_missing_price_emitted_on_retry_pass(self):
        # Identical input, empty_amends=None (retry pass) -> guard skipped,
        # the mutation IS emitted. Proves the asymmetry.
        out = parse.parse_tool_calls(
            [tc("record_drawdown", {"instrument_id": "X",
                                    "drawdown_shares": 100,
                                    "drawdown_amount_usd": 1000,
                                    "event_date": "2025-01-14"})],
            accession="ACC", empty_amends=None)
        assert len(out) == 1
        assert isinstance(out[0], RecordDrawdown)
        assert out[0].price_per_share is None

    def test_drawdown_with_price_no_guard(self):
        ea: list[RetryableFailure] = []
        out = parse.parse_tool_calls(
            [tc("record_drawdown", {"instrument_id": "X",
                                    "drawdown_shares": 100,
                                    "price_per_share": 5,
                                    "event_date": "2025-01-14"})],
            accession="ACC", empty_amends=ea)
        assert len(out) == 1
        assert isinstance(out[0], RecordDrawdown)
        assert ea == []

    def test_drawdown_zero_shares_no_price_skips_guard_first_pass(self):
        # The guard requires a TRUTHY mutation.drawdown_shares. With
        # drawdown_shares=0 (falsey) on the first pass, the guard does NOT
        # fire even though price_per_share is None — the aggregate-only
        # mutation is emitted instead of being bounced to retry.
        ea: list[RetryableFailure] = []
        out = parse.parse_tool_calls(
            [tc("record_drawdown", {"instrument_id": "X",
                                    "drawdown_shares": 0,
                                    "drawdown_amount_usd": 1000,
                                    "event_date": "2025-01-14"})],
            accession="ACC", empty_amends=ea)
        assert len(out) == 1
        assert isinstance(out[0], RecordDrawdown)
        assert out[0].drawdown_shares == 0.0
        assert out[0].price_per_share is None
        assert ea == []

    def test_valid_call_returns_typed_mutation(self):
        out = parse.parse_tool_calls(
            [tc("note_no_event", {"reason": "no dilutive events"})],
            accession="ACC")
        assert len(out) == 1
        assert isinstance(out[0], NoteNoEvent)
        assert out[0].reason == "no dilutive events"

    def test_mixed_batch_preserves_input_order_of_valid(self):
        calls = [
            tc("create_atm", {"capacity_usd": 5e6,
                              "placement_agent_canonical": "HCW",
                              "agreement_date": "2025-01-01",
                              "event_date": "2025-01-02"}),
            tc("frobnicate", {}),                                  # unknown
            tc("record_drawdown", {"instrument_id": "X",
                                   "drawdown_shares": 1,
                                   "event_date": "2025-01-03"}),   # ValueError
            tc("note_no_event", {"reason": "done"}),
            tc("note_no_event", {"reason": "done"}),               # dup
        ]
        out = parse.parse_tool_calls(calls, accession="ACC")
        assert len(out) == 2
        assert isinstance(out[0], CreateAtm)
        assert isinstance(out[1], NoteNoEvent)

    def test_accession_appears_in_log(self, caplog):
        with caplog.at_level("WARNING", logger="dilution.ledger.tools.parse"):
            parse.parse_tool_calls(
                [tc("frobnicate", {})], accession="MY-UNIQUE-ACC")
        assert any("MY-UNIQUE-ACC" in r.message for r in caplog.records)

    def test_close_instrument_routes_keyerror_drop(self):
        # Missing reason -> KeyError -> dropped silently, nothing appended.
        ea: list[RetryableFailure] = []
        out = parse.parse_tool_calls(
            [tc("close_instrument", {"instrument_id": "X",
                                     "event_date": "2025-01-14"})],
            accession="ACC", empty_amends=ea)
        assert out == []
        assert ea == []


# ─────────────────────────────────────────────────────────────────────
# RetryableFailure dataclass + aliases
# ─────────────────────────────────────────────────────────────────────
class TestRetryableFailure:
    def test_default_error_message_empty(self):
        f = RetryableFailure(kind="empty_amend", tool_name="amend_atm",
                             instrument_id="X", event_date=None)
        assert f.error_message == ""

    def test_frozen(self):
        f = RetryableFailure(kind="bad_date", tool_name="create_atm",
                             instrument_id="", event_date=None)
        with pytest.raises(Exception):
            f.kind = "other"  # type: ignore[misc]

    def test_emptyamendfailure_alias_is_retryablefailure(self):
        assert EmptyAmendFailure is RetryableFailure


# ─────────────────────────────────────────────────────────────────────
# apply_split through the full dispatcher (integration of split builder)
# ─────────────────────────────────────────────────────────────────────
class TestApplySplitViaDispatcher:
    def test_valid_ads_split_emits_applysplit(self):
        out = parse.parse_tool_calls(
            [tc("apply_split", {"ads_ratio_from": 400, "ads_ratio_to": 4000,
                                "effective_date": "2025-01-14"})],
            accession="ACC")
        assert len(out) == 1
        m = out[0]
        assert isinstance(m, ApplySplit)
        assert (m.post, m.pre, m.direction, m.units) == (1, 10, "reverse", "ads")

    def test_invalid_split_dropped_silently(self):
        # both shapes -> ValueError -> dropped, not retryable.
        ea: list[RetryableFailure] = []
        out = parse.parse_tool_calls(
            [tc("apply_split", {"post": 1, "pre": 10, "direction": "reverse",
                                "ads_ratio_from": 400, "ads_ratio_to": 4000,
                                "effective_date": "2025-01-14"})],
            accession="ACC", empty_amends=ea)
        assert out == []
        assert ea == []
