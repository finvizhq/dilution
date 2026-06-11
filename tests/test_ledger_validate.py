"""Unit tests for dilution/ledger/validate.py.

The module is PURE: no I/O, no DB, no LLM, no config-at-import. The
autouse ``temp_db`` fixture from conftest.py still runs (it reroutes
db.get_conn for safety) but is never referenced here — nothing in this
module touches the database.

Tests construct frozen mutation dataclasses from
``dilution.ledger.mutations`` directly and pass plain dicts as the
ledger snapshot / target rows (the shape returned by
``store.get_open_instruments``: keys ``type`` / ``status`` /
``terms_json`` / ``outstanding_json``).
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from dilution.ledger import validate
from dilution.ledger.validate import (
    DRAWDOWN_TOLERANCE,
    MUTATION_APPLY_ORDER,
    ValidationReport,
    ValidationResult,
    _accumulate_amend_effect,
    _accumulate_event_effect,
    _canonical_entity_ok,
    _coerce_num,
    _drawdown_overflow,
    _from_json_field,
    _pos_num,
    _sanitize_entity_canonicals,
    _validate_one,
    sort_mutations,
    validate_mutations,
)
from dilution.ledger.mutations import (
    AmendConvertible,
    AmendPreferred,
    AmendWarrant,
    ApplySplit,
    CloseInstrument,
    ConfirmClosing,
    CreateConvertible,
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
)


D = date(2024, 1, 1)


def _warrant_row(count=10.0, status="active"):
    return {
        "type": "warrant",
        "status": status,
        "terms_json": {},
        "outstanding_json": {"count": count},
    }


def _convertible_row(principal_remaining=1000.0, status="active", terms=None):
    return {
        "type": "convertible",
        "status": status,
        "terms_json": terms or {},
        "outstanding_json": {"principal_remaining": principal_remaining},
    }


def _preferred_row(count=100.0, status="active", terms=None):
    return {
        "type": "preferred",
        "status": status,
        "terms_json": terms or {},
        "outstanding_json": {"count": count},
    }


def _atm_row(remaining=100.0, status="active"):
    return {
        "type": "atm",
        "status": status,
        "terms_json": {},
        "outstanding_json": {"remaining_capacity_usd": remaining},
    }


def _equity_line_row(remaining=100.0, status="active"):
    return {
        "type": "equity_line",
        "status": status,
        "terms_json": {},
        "outstanding_json": {"remaining_capacity_usd": remaining},
    }


def _s1_offering_row(status="active"):
    return {
        "type": "s1_offering",
        "status": status,
        "terms_json": {},
        "outstanding_json": {"sold_to_date": 0.0},
    }


def _shelf_row(remaining=100.0, status="active"):
    return {
        "type": "shelf",
        "status": status,
        "terms_json": {},
        "outstanding_json": {"remaining_capacity_usd": remaining},
    }


# ═══════════════════════════════════════════════════════════════════════
class TestCanonicalEntityOk:
    def test_none_passes(self):
        assert _canonical_entity_ok(None) is True

    @pytest.mark.parametrize("s", ["", "   ", "\t\n "])
    def test_empty_or_whitespace_passes(self, s):
        assert _canonical_entity_ok(s) is True

    @pytest.mark.parametrize(
        "s", ["Maxim", "Hudson Bay", "Hybrid Capital 12 LLC", "Streeterville"]
    )
    def test_real_proper_nouns_pass(self, s):
        # Each has at least one non-vocabulary token.
        assert _canonical_entity_ok(s) is True

    @pytest.mark.parametrize(
        "s", ["warrants", "Investors", "The Purchaser", "investor", "holders"]
    )
    def test_generic_tokens_rejected_case_insensitive(self, s):
        assert _canonical_entity_ok(s) is False

    def test_uppercase_generic_token_rejected_via_lower(self):
        assert _canonical_entity_ok("WARRANTS") is False

    @pytest.mark.parametrize(
        "s", ["November 2024 warrants", "May", "August deal", "January Notes"]
    )
    def test_month_prefixed_rejected(self, s):
        # _MONTH_NAME_RE uses \b so bare 'May' matches.
        assert _canonical_entity_ok(s) is False

    @pytest.mark.parametrize("s", ["Maxim (Group", "Maxim [Group"])
    def test_unbalanced_brackets_rejected(self, s):
        assert _canonical_entity_ok(s) is False

    def test_balanced_paren_with_alnum_lead_passes(self):
        # Balanced and other checks pass.
        assert _canonical_entity_ok("Maxim (Group)") is True

    def test_leading_non_alnum_rejected(self):
        # '(Maxim)' is balanced but starts with '(' (non-alnum).
        assert _canonical_entity_ok("(Maxim)") is False

    @pytest.mark.parametrize(
        "s", ["remaining series", "convertible promissory", "redeemable convertible"]
    )
    def test_all_lowercase_rejected(self, s):
        assert _canonical_entity_ok(s) is False

    @pytest.mark.parametrize(
        "s", ["Redeemable Convertible", "Warrants To Warrants"]
    )
    def test_all_vocab_tokens_rejected_even_capitalized(self, s):
        assert _canonical_entity_ok(s) is False

    @pytest.mark.parametrize("v", [123, 4.5, ["x"], {"a": 1}])
    def test_non_str_rejected(self, v):
        assert _canonical_entity_ok(v) is False

    def test_one_vocab_one_nonvocab_passes(self):
        # 'Hybrid Warrants' is NOT all-vocab ('hybrid' is non-vocab).
        assert _canonical_entity_ok("Hybrid Warrants") is True


# ═══════════════════════════════════════════════════════════════════════
class TestSanitizeEntityCanonicals:
    def test_both_clean_returns_same_identity(self):
        m = CreateWarrant(
            count=1, strike=1, event_date=D,
            counterparty_canonical="Maxim",
            placement_agent_canonical="Hudson Bay",
        )
        out, changes = _sanitize_entity_canonicals(m)
        assert out is m
        assert changes == []

    def test_bad_counterparty_only(self):
        m = CreateWarrant(
            count=1, strike=1, event_date=D,
            counterparty_canonical="warrants to",
            placement_agent_canonical="Maxim",
        )
        out, changes = _sanitize_entity_canonicals(m)
        assert out is not m  # frozen dataclass → replace returns new instance
        assert out.counterparty_canonical is None
        assert out.placement_agent_canonical == "Maxim"
        assert changes == ["counterparty_canonical"]
        # original untouched
        assert m.counterparty_canonical == "warrants to"

    def test_bad_placement_agent_only(self):
        m = CreateWarrant(
            count=1, strike=1, event_date=D,
            counterparty_canonical="Maxim",
            placement_agent_canonical="investors",
        )
        out, changes = _sanitize_entity_canonicals(m)
        assert out is not m
        assert out.placement_agent_canonical is None
        assert out.counterparty_canonical == "Maxim"
        assert changes == ["placement_agent_canonical"]

    def test_both_bad(self):
        m = CreateWarrant(
            count=1, strike=1, event_date=D,
            counterparty_canonical="warrants",
            placement_agent_canonical="holders",
        )
        out, changes = _sanitize_entity_canonicals(m)
        assert out.counterparty_canonical is None
        assert out.placement_agent_canonical is None
        assert set(changes) == {"counterparty_canonical",
                                "placement_agent_canonical"}

    def test_field_already_none_not_in_changes(self):
        # None passes _canonical_entity_ok, so it stays None and is NOT
        # reported as a change.
        m = CreateWarrant(
            count=1, strike=1, event_date=D,
            counterparty_canonical=None,
            placement_agent_canonical=None,
        )
        out, changes = _sanitize_entity_canonicals(m)
        assert out is m
        assert changes == []


# ═══════════════════════════════════════════════════════════════════════
class TestSortMutations:
    def test_empty(self):
        assert sort_mutations([]) == []

    def test_single_unchanged(self):
        m = CreateWarrant(count=1, strike=1, event_date=D)
        assert sort_mutations([m]) == [m]

    def test_mixed_kinds_ordered(self):
        close = CloseInstrument(instrument_id="X", reason="redeemed", event_date=D)
        create = CreateWarrant(count=1, strike=1, event_date=D)
        split = ApplySplit(post=2, pre=1, direction="forward", effective_date=D)
        amend = AmendWarrant(instrument_id="X", event_date=D, count=5)
        rec = RecordExercise(instrument_id="X", shares=1, event_date=D)
        out = sort_mutations([close, rec, amend, create, split])
        kinds = [m.kind for m in out]
        assert kinds == [
            "apply_split", "create_instrument", "amend_instrument",
            "record_event", "close_instrument",
        ]

    def test_create_and_restate_same_rank_stable(self):
        create = CreateWarrant(count=1, strike=1, event_date=D)
        restate = RestateAtm(predecessor_id="A", capacity_usd=1e6, event_date=D)
        # both rank 1; declared order preserved
        out = sort_mutations([restate, create])
        assert out == [restate, create]
        out2 = sort_mutations([create, restate])
        assert out2 == [create, restate]

    def test_same_kind_stable(self):
        a = CreateWarrant(count=1, strike=1, event_date=D, proposed_id="A")
        b = CreateWarrant(count=2, strike=2, event_date=D, proposed_id="B")
        assert sort_mutations([a, b]) == [a, b]
        assert sort_mutations([b, a]) == [b, a]

    def test_note_no_event_sorts_last(self):
        # NoteNoEvent kind not in MUTATION_APPLY_ORDER → rank 99.
        note = NoteNoEvent(reason="nothing")
        create = CreateWarrant(count=1, strike=1, event_date=D)
        out = sort_mutations([note, create])
        assert out == [create, note]

    def test_generator_input_consumed(self):
        gen = (m for m in [
            CloseInstrument(instrument_id="X", reason="redeemed", event_date=D),
            CreateWarrant(count=1, strike=1, event_date=D),
        ])
        out = sort_mutations(gen)
        assert [m.kind for m in out] == ["create_instrument", "close_instrument"]

    def test_apply_order_constants(self):
        assert MUTATION_APPLY_ORDER["apply_split"] == 0
        assert MUTATION_APPLY_ORDER["create_instrument"] == 1
        assert MUTATION_APPLY_ORDER["restate_instrument"] == 1
        assert MUTATION_APPLY_ORDER["amend_instrument"] == 2
        assert MUTATION_APPLY_ORDER["record_event"] == 3
        assert MUTATION_APPLY_ORDER["close_instrument"] == 4


# ═══════════════════════════════════════════════════════════════════════
class TestDrawdownOverflow:
    def _dd(self, amount):
        return RecordDrawdown(
            instrument_id="A", drawdown_shares=1, event_date=D,
            price_per_share=None, drawdown_amount_usd=amount,
        )

    def test_remaining_none(self):
        assert _drawdown_overflow(
            self._dd(200), {"outstanding_json": {}}) is None

    def test_remaining_zero(self):
        assert _drawdown_overflow(
            self._dd(200),
            {"outstanding_json": {"remaining_capacity_usd": 0}}) is None

    def test_remaining_negative(self):
        assert _drawdown_overflow(
            self._dd(200),
            {"outstanding_json": {"remaining_capacity_usd": -50}}) is None

    def test_requested_none(self):
        # price_per_share=None and drawdown_amount_usd=None → fields amount 0.0
        m = RecordDrawdown(instrument_id="A", drawdown_shares=1, event_date=D)
        assert _drawdown_overflow(
            m, {"outstanding_json": {"remaining_capacity_usd": 100}}) is None

    def test_requested_zero(self):
        assert _drawdown_overflow(
            self._dd(0),
            {"outstanding_json": {"remaining_capacity_usd": 100}}) is None

    def test_exactly_at_tolerance_returns_none(self):
        # 105 vs 100: overflow == 0.05, comparison is strictly > → None.
        assert _drawdown_overflow(
            self._dd(105.0),
            {"outstanding_json": {"remaining_capacity_usd": 100.0}}) is None

    def test_just_over_tolerance_returns_ratio(self):
        out = _drawdown_overflow(
            self._dd(105.01),
            {"outstanding_json": {"remaining_capacity_usd": 100.0}})
        assert out == pytest.approx(0.0501, abs=1e-4)

    def test_far_over(self):
        out = _drawdown_overflow(
            self._dd(200.0),
            {"outstanding_json": {"remaining_capacity_usd": 100.0}})
        assert out == pytest.approx(1.0)

    def test_under_remaining(self):
        # 50 vs 100 → overflow negative, not > tolerance → None.
        assert _drawdown_overflow(
            self._dd(50.0),
            {"outstanding_json": {"remaining_capacity_usd": 100.0}}) is None

    def test_key_precedence_zero_falls_through(self):
        # remaining_capacity_usd=0 (falsy) → falls to capacity_remaining_usd.
        out = _drawdown_overflow(
            self._dd(200.0),
            {"outstanding_json": {
                "remaining_capacity_usd": 0,
                "capacity_remaining_usd": 100.0}})
        assert out == pytest.approx(1.0)

    def test_third_key_remaining_usd(self):
        out = _drawdown_overflow(
            self._dd(200.0),
            {"outstanding_json": {"remaining_usd": 100.0}})
        assert out == pytest.approx(1.0)

    def test_json_string_outstanding(self):
        # _from_json_field path: outstanding_json as a JSON string.
        out = _drawdown_overflow(
            self._dd(200.0),
            {"outstanding_json": '{"remaining_capacity_usd": 100.0}'})
        assert out == pytest.approx(1.0)

    def test_price_per_share_computes_amount(self):
        # drawdown_amount_usd computed from shares * price.
        m = RecordDrawdown(
            instrument_id="A", drawdown_shares=100, event_date=D,
            price_per_share=2.0)  # → 200
        out = _drawdown_overflow(
            m, {"outstanding_json": {"remaining_capacity_usd": 100.0}})
        assert out == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════
class TestCoerceNum:
    def test_none(self):
        assert _coerce_num(None) is None

    @pytest.mark.parametrize("v", [True, False])
    def test_bool_returns_none(self, v):
        # bool guard precedes int branch (bool is int subclass).
        assert _coerce_num(v) is None

    def test_zero_is_float_not_none(self):
        # distinguishes zero from unknown
        out = _coerce_num(0)
        assert out == 0.0
        assert out is not None

    @pytest.mark.parametrize("v,expected", [(5, 5.0), (5.5, 5.5), (-3, -3.0)])
    def test_int_float(self, v, expected):
        assert _coerce_num(v) == expected

    def test_string_comma(self):
        assert _coerce_num("1,234.5") == 1234.5

    def test_string_whitespace(self):
        assert _coerce_num("  42 ") == 42.0

    @pytest.mark.parametrize("v", ["abc", "", "  "])
    def test_unparseable_string(self, v):
        assert _coerce_num(v) is None

    @pytest.mark.parametrize("v", [[1, 2], {"a": 1}])
    def test_collections_return_none(self, v):
        assert _coerce_num(v) is None


# ═══════════════════════════════════════════════════════════════════════
class TestPosNum:
    @pytest.mark.parametrize("v", [0, 0.0])
    def test_zero_returns_none(self, v):
        assert _pos_num(v) is None

    @pytest.mark.parametrize("v", [-1, -0.5, "-3"])
    def test_negative_returns_none(self, v):
        assert _pos_num(v) is None

    @pytest.mark.parametrize("v,expected", [(5, 5.0), (5.5, 5.5), ("12", 12.0)])
    def test_positive(self, v, expected):
        assert _pos_num(v) == expected

    @pytest.mark.parametrize("v", [None, True, False, "abc", [1]])
    def test_invalid_returns_none(self, v):
        assert _pos_num(v) is None


# ═══════════════════════════════════════════════════════════════════════
class TestFromJsonField:
    def test_key_absent(self):
        assert _from_json_field({}, "terms_json") == {}

    def test_value_none(self):
        assert _from_json_field({"terms_json": None}, "terms_json") == {}

    def test_already_dict_returned_as_is(self):
        d = {"a": 1}
        out = _from_json_field({"terms_json": d}, "terms_json")
        assert out is d

    def test_json_string(self):
        out = _from_json_field({"terms_json": '{"a": 1}'}, "terms_json")
        assert out == {"a": 1}

    def test_malformed_json(self):
        assert _from_json_field({"terms_json": "not json"}, "terms_json") == {}

    def test_json_null(self):
        # json.loads('null') is None → `or {}` → {}
        assert _from_json_field({"terms_json": "null"}, "terms_json") == {}

    def test_json_non_dict_list_returned(self):
        # No type guard: a JSON list is returned as-is. Current behavior.
        out = _from_json_field({"terms_json": "[1, 2]"}, "terms_json")
        assert out == [1, 2]


# ═══════════════════════════════════════════════════════════════════════
class TestAccumulateAmendEffect:
    def test_id_missing_both_noop(self):
        overlay = {}
        m = AmendConvertible(instrument_id="ZZZ", event_date=D, conv_price=2.0)
        _accumulate_amend_effect(m, {}, overlay)
        assert overlay == {}

    def test_normal_set_into_terms_and_outstanding(self):
        snap = {"C1": _convertible_row(principal_remaining=500.0,
                                       terms={"conv_price": 1.0})}
        overlay = {}
        m = AmendConvertible(instrument_id="C1", event_date=D,
                             conv_price=3.0, principal_remaining=200.0)
        _accumulate_amend_effect(m, snap, overlay)
        assert overlay["C1"]["terms_json"]["conv_price"] == 3.0
        assert overlay["C1"]["outstanding_json"]["principal_remaining"] == 200.0

    def test_overlay_already_present_chains(self):
        snap = {"C1": _convertible_row(principal_remaining=500.0)}
        overlay = {
            "C1": {
                "type": "convertible", "status": "active",
                "terms_json": {"conv_price": 9.0},
                "outstanding_json": {"principal_remaining": 100.0},
            }
        }
        m = AmendConvertible(instrument_id="C1", event_date=D, conv_price=3.0)
        _accumulate_amend_effect(m, snap, overlay)
        # uses overlay base, not snapshot (chained amend)
        assert overlay["C1"]["terms_json"]["conv_price"] == 3.0
        assert overlay["C1"]["outstanding_json"]["principal_remaining"] == 100.0

    def test_json_string_base_merged(self):
        snap = {"C1": {
            "type": "convertible", "status": "active",
            "terms_json": '{"conv_price": 2.0}',
            "outstanding_json": '{"principal_remaining": 500.0}',
        }}
        overlay = {}
        m = AmendConvertible(instrument_id="C1", event_date=D,
                             conv_price=3.0, principal_remaining=400.0)
        _accumulate_amend_effect(m, snap, overlay)
        assert overlay["C1"]["terms_json"] == {"conv_price": 3.0}
        assert overlay["C1"]["outstanding_json"] == {"principal_remaining": 400.0}

    def test_none_pop_branch_via_handbuilt(self):
        # Amend* props omit None fields, so the pop branch is only
        # reachable by a hand-built object whose update dicts literally
        # contain None values.
        snap = {"C1": {
            "type": "convertible", "status": "active",
            "terms_json": {"conv_price": 2.0},
            "outstanding_json": {"principal_remaining": 500.0},
        }}
        overlay = {}
        fake = SimpleNamespace(
            instrument_id="C1",
            field_updates={"conv_price": None},
            outstanding_updates={"principal_remaining": None},
        )
        _accumulate_amend_effect(fake, snap, overlay)
        assert "conv_price" not in overlay["C1"]["terms_json"]
        assert "principal_remaining" not in overlay["C1"]["outstanding_json"]


# ═══════════════════════════════════════════════════════════════════════
class TestAccumulateEventEffect:
    def test_no_instrument_id_attr_early_return(self):
        overlay = {}
        fake = SimpleNamespace(event_kind="exercise", fields={})
        # getattr(m, 'instrument_id', None) → None → early return
        _accumulate_event_effect(fake, {}, overlay)
        assert overlay == {}

    def test_id_missing_early_return(self):
        overlay = {}
        m = RecordExercise(instrument_id="ZZZ", shares=1, event_date=D)
        _accumulate_event_effect(m, {}, overlay)
        assert overlay == {}

    def test_exercise_dec_by_warrants_exercised(self):
        snap = {"W1": _warrant_row(count=100.0)}
        overlay = {}
        m = RecordExercise(instrument_id="W1", shares=30, event_date=D,
                           warrants_exercised=40)
        _accumulate_event_effect(m, snap, overlay)
        assert overlay["W1"]["outstanding_json"]["count"] == 60.0

    def test_exercise_dec_by_shares_when_no_warrants_exercised(self):
        snap = {"W1": _warrant_row(count=100.0)}
        overlay = {}
        m = RecordExercise(instrument_id="W1", shares=30, event_date=D)
        _accumulate_event_effect(m, snap, overlay)
        assert overlay["W1"]["outstanding_json"]["count"] == 70.0

    def test_dec_when_current_none_skipped(self):
        # count absent → _coerce_num(None) is None → _dec skips.
        snap = {"W1": {"type": "warrant", "status": "active",
                       "terms_json": {}, "outstanding_json": {}}}
        overlay = {}
        m = RecordExercise(instrument_id="W1", shares=30, event_date=D)
        _accumulate_event_effect(m, snap, overlay)
        assert "count" not in overlay["W1"]["outstanding_json"]

    def test_exercise_floors_at_zero(self):
        snap = {"W1": _warrant_row(count=10.0)}
        overlay = {}
        m = RecordExercise(instrument_id="W1", shares=100, event_date=D)
        _accumulate_event_effect(m, snap, overlay)
        assert overlay["W1"]["outstanding_json"]["count"] == 0.0

    def test_conversion_convertible_dec_principal(self):
        snap = {"C1": _convertible_row(principal_remaining=1000.0)}
        overlay = {}
        m = RecordConversion(instrument_id="C1", shares_issued=10, event_date=D,
                             principal_converted=400)
        _accumulate_event_effect(m, snap, overlay)
        assert overlay["C1"]["outstanding_json"]["principal_remaining"] == 600.0

    def test_conversion_explicit_principal_remaining_overrides(self):
        snap = {"C1": _convertible_row(principal_remaining=1000.0)}
        overlay = {}
        m = RecordConversion(instrument_id="C1", shares_issued=10, event_date=D,
                             principal_converted=400, principal_remaining=123)
        _accumulate_event_effect(m, snap, overlay)
        # decrement applied then absolute set wins
        assert overlay["C1"]["outstanding_json"]["principal_remaining"] == 123.0

    def test_conversion_preferred_dec_by_pref_shares(self):
        snap = {"P1": _preferred_row(count=50.0)}
        overlay = {}
        m = RecordConversion(instrument_id="P1", shares_issued=10, event_date=D,
                             preferred_shares_converted=8)
        _accumulate_event_effect(m, snap, overlay)
        assert overlay["P1"]["outstanding_json"]["count"] == 42.0

    def test_conversion_preferred_derive_pref_shares_from_stated_value(self):
        snap = {"P1": _preferred_row(count=50.0, terms={"stated_value": 1000.0})}
        overlay = {}
        m = RecordConversion(instrument_id="P1", shares_issued=10, event_date=D,
                             principal_converted=10000)  # → 10 pref shares
        _accumulate_event_effect(m, snap, overlay)
        assert overlay["P1"]["outstanding_json"]["count"] == 40.0

    def test_preferred_derivation_skipped_when_not_preferred_row(self):
        # convertible row: principal decremented, count not touched.
        snap = {"C1": _convertible_row(principal_remaining=10000.0,
                                       terms={"stated_value": 1000.0})}
        overlay = {}
        m = RecordConversion(instrument_id="C1", shares_issued=10, event_date=D,
                             principal_converted=2000)
        _accumulate_event_effect(m, snap, overlay)
        assert overlay["C1"]["outstanding_json"]["principal_remaining"] == 8000.0
        assert "count" not in overlay["C1"]["outstanding_json"]

    def test_partial_redemption_convertible(self):
        snap = {"C1": _convertible_row(principal_remaining=1000.0)}
        overlay = {}
        m = RecordPartialRedemption(instrument_id="C1", event_date=D,
                                    principal_redeemed=250)
        _accumulate_event_effect(m, snap, overlay)
        assert overlay["C1"]["outstanding_json"]["principal_remaining"] == 750.0

    def test_partial_redemption_preferred_derive_from_stated_value(self):
        snap = {"P1": _preferred_row(count=50.0, terms={"stated_value": 1000.0})}
        overlay = {}
        m = RecordPartialRedemption(instrument_id="P1", event_date=D,
                                    principal_redeemed=5000)  # → 5 pref shares
        _accumulate_event_effect(m, snap, overlay)
        assert overlay["P1"]["outstanding_json"]["count"] == 45.0

    def test_drawdown_is_noop(self):
        snap = {"A1": _atm_row(remaining=100.0)}
        overlay = {}
        m = RecordDrawdown(instrument_id="A1", drawdown_shares=10, event_date=D,
                           price_per_share=2.0)
        _accumulate_event_effect(m, snap, overlay)
        # else branch returns before mutating overlay
        assert overlay == {}


# ═══════════════════════════════════════════════════════════════════════
class TestValidateOneApplySplit:
    @pytest.mark.parametrize("post,pre", [(0, 1), (-1, 1)])
    def test_ratio_non_positive_rejected(self, post, pre):
        m = ApplySplit(post=post, pre=pre, direction="forward",
                       effective_date=D)
        r = _validate_one(m, {}, set(), {})
        assert not r.accepted
        assert r.error_kind == "invalid_split"

    def test_ratio_positive_accepted(self):
        m = ApplySplit(post=2, pre=1, direction="forward", effective_date=D)
        r = _validate_one(m, {}, set(), {})
        assert r.accepted


class TestValidateOneCreateShelf:
    def test_takedown_label_rejected(self):
        m = CreateShelf(capacity_usd=1e6, event_date=D, form="S-3",
                        label="2024 Takedown")
        r = _validate_one(m, {}, set(), {})
        assert r.error_kind == "shelf_takedown_misclassified"

    def test_drawdown_label_rejected(self):
        m = CreateShelf(capacity_usd=1e6, event_date=D, form="S-3",
                        label="Q2 Drawdown notice")
        r = _validate_one(m, {}, set(), {})
        assert r.error_kind == "shelf_takedown_misclassified"

    @pytest.mark.parametrize("ff", ["8-K", "424B5", "POSAM"])
    def test_wrong_filing_form_rejected(self, ff):
        m = CreateShelf(capacity_usd=1e6, event_date=D, form="S-3")
        r = _validate_one(m, {}, set(), {}, None, ff)
        assert r.error_kind == "shelf_wrong_filing_form"

    def test_filing_form_slash_a_suffix_stripped_passes_form_gate(self):
        # filing_form 'S-3/A' normalizes to 'S-3' (split on '/'), which IS
        # an allowed shelf-create form — so the wrong-filing-form gate does
        # NOT fire. With a valid terms.form='S-3' the whole create passes.
        # (The /A amendment intent is caught instead by terms.form='S-3/A'
        # → shelf_misclassified, covered separately.)
        m = CreateShelf(capacity_usd=1e6, event_date=D, form="S-3")
        r = _validate_one(m, {}, set(), {}, None, "S-3/A")
        assert r.accepted

    @pytest.mark.parametrize("ff", ["S-3", "F-3ASR", "f-10", "S-3ASR"])
    def test_allowed_filing_forms_pass(self, ff):
        m = CreateShelf(capacity_usd=1e6, event_date=D, form="S-3")
        r = _validate_one(m, {}, set(), {}, None, ff)
        assert r.accepted

    def test_terms_form_misclassified_even_when_filing_form_ok(self):
        # filing_form S-3 passes the wrong-form gate, but terms.form S-3/A
        # (→ 'S-3A') is not a base shelf form.
        m = CreateShelf(capacity_usd=1e6, event_date=D, form="S-3/A")
        r = _validate_one(m, {}, set(), {}, None, "S-3")
        assert r.error_kind == "shelf_misclassified"

    def test_no_filing_form_still_checks_terms_form(self):
        m = CreateShelf(capacity_usd=1e6, event_date=D, form="424B5")
        r = _validate_one(m, {}, set(), {}, None, None)
        assert r.error_kind == "shelf_misclassified"

    def test_valid_shelf_accepted(self):
        m = CreateShelf(capacity_usd=1e6, event_date=D, form="S-3")
        r = _validate_one(m, {}, set(), {}, None, "S-3")
        assert r.accepted


class TestValidateOnePeriodicCreate:
    def test_convertible_missing_terms_rejected(self):
        m = CreateConvertible(principal=1e6, principal_remaining=1e6,
                              event_date=D)
        r = _validate_one(m, {}, set(), {}, None, "10-Q")
        assert r.error_kind == "periodic_create_missing_terms"

    def test_convertible_missing_maturity_only_rejected(self):
        m = CreateConvertible(principal=1e6, principal_remaining=1e6,
                              event_date=D, conv_price=2.0)
        r = _validate_one(m, {}, set(), {}, None, "10-K")
        assert r.error_kind == "periodic_create_missing_terms"

    def test_convertible_both_present_passes(self):
        m = CreateConvertible(principal=1e6, principal_remaining=1e6,
                              event_date=D, conv_price=2.0,
                              maturity=date(2026, 1, 1))
        r = _validate_one(m, {}, set(), {}, None, "10-Q")
        assert r.accepted

    def test_warrant_missing_terms_rejected(self):
        m = CreateWarrant(count=1, strike=1, event_date=D)
        r = _validate_one(m, {}, set(), {}, None, "20-F")
        assert r.error_kind == "periodic_create_missing_terms"

    def test_warrant_both_present_passes(self):
        m = CreateWarrant(count=1, strike=1, event_date=D,
                          expiration=date(2026, 1, 1))
        r = _validate_one(m, {}, set(), {}, None, "10-Q")
        assert r.accepted

    def test_preferred_exempt_from_periodic_gate(self):
        m = CreatePreferred(count=1, series_letter="A", event_date=D)
        r = _validate_one(m, {}, set(), {}, None, "10-K")
        assert r.accepted

    def test_non_periodic_form_skips_gate(self):
        # 8-K is not a periodic narrative form; create passes the term gate
        # (a warrant create from 8-K with no strike is still allowed here).
        m = CreateWarrant(count=1, strike=1, event_date=D)
        r = _validate_one(m, {}, set(), {}, None, "8-K")
        assert r.accepted


class TestValidateOneRestateAtm:
    def test_wrong_filing_form_rejected(self):
        m = RestateAtm(predecessor_id="A0", capacity_usd=1e6, event_date=D)
        r = _validate_one(m, {"A0": _atm_row()}, {"A0"}, {}, None, "8-K")
        assert r.error_kind == "restate_wrong_filing_form"

    def test_posam_form_rejected(self):
        m = RestateAtm(predecessor_id="A0", capacity_usd=1e6, event_date=D)
        r = _validate_one(m, {"A0": _atm_row()}, {"A0"}, {}, None, "POSAM")
        assert r.error_kind == "restate_wrong_filing_form"

    def test_424b5_form_passes_to_pred_check(self):
        m = RestateAtm(predecessor_id="A0", capacity_usd=1e6, event_date=D)
        r = _validate_one(m, {"A0": _atm_row()}, {"A0"}, {}, None, "424B5")
        assert r.accepted

    def test_missing_predecessor_rejected(self):
        m = RestateAtm(predecessor_id="A0", capacity_usd=1e6, event_date=D)
        r = _validate_one(m, {}, set(), {}, None, "424B5")
        assert r.error_kind == "missing_predecessor"

    def test_predecessor_wrong_type_rejected(self):
        m = RestateAtm(predecessor_id="A0", capacity_usd=1e6, event_date=D)
        snap = {"A0": _warrant_row()}
        r = _validate_one(m, snap, {"A0"}, {}, None, "424B5")
        assert r.error_kind == "type_mismatch"

    @pytest.mark.parametrize("status", ["terminated", "superseded:NEW"])
    def test_predecessor_terminal_rejected(self, status):
        m = RestateAtm(predecessor_id="A0", capacity_usd=1e6, event_date=D)
        snap = {"A0": _atm_row(status=status)}
        r = _validate_one(m, snap, {"A0"}, {}, None, "424B5")
        assert r.error_kind == "illegal_transition"

    def test_valid_restate_accepted(self):
        m = RestateAtm(predecessor_id="A0", capacity_usd=1e6, event_date=D)
        snap = {"A0": _atm_row(status="active")}
        r = _validate_one(m, snap, {"A0"}, {}, None, "424B5")
        assert r.accepted


class TestValidateOneIdExistence:
    def test_missing_id(self):
        m = CloseInstrument(instrument_id="ZZZ", reason="redeemed", event_date=D)
        r = _validate_one(m, {}, set(), {})
        assert r.error_kind == "missing_id"
        assert "not found in ledger" in r.message

    def test_in_live_ids_but_target_none(self):
        m = CloseInstrument(instrument_id="ZZZ", reason="redeemed", event_date=D)
        r = _validate_one(m, {}, {"ZZZ"}, {})
        assert r.error_kind == "missing_id"
        assert "without proposed_id" in r.message


class TestValidateOneRecordEvent:
    def test_exercise_on_non_warrant_type_mismatch(self):
        snap = {"C1": _convertible_row()}
        m = RecordExercise(instrument_id="C1", shares=1, event_date=D)
        r = _validate_one(m, snap, {"C1"}, {})
        assert r.error_kind == "type_mismatch"

    def test_exercise_on_warrant_ok(self):
        snap = {"W1": _warrant_row(count=10.0)}
        m = RecordExercise(instrument_id="W1", shares=1, event_date=D)
        r = _validate_one(m, snap, {"W1"}, {})
        assert r.accepted

    def test_conversion_on_atm_type_mismatch(self):
        snap = {"A1": _atm_row()}
        m = RecordConversion(instrument_id="A1", shares_issued=1, event_date=D,
                             principal_converted=100)
        r = _validate_one(m, snap, {"A1"}, {})
        assert r.error_kind == "type_mismatch"

    def test_conversion_on_convertible_ok(self):
        snap = {"C1": _convertible_row()}
        m = RecordConversion(instrument_id="C1", shares_issued=1, event_date=D,
                             principal_converted=100)
        r = _validate_one(m, snap, {"C1"}, {})
        assert r.accepted

    def test_record_event_on_terminal_rejected(self):
        snap = {"W1": _warrant_row(count=10.0, status="expired")}
        m = RecordExercise(instrument_id="W1", shares=1, event_date=D)
        r = _validate_one(m, snap, {"W1"}, {})
        assert r.error_kind == "illegal_transition"

    def test_drawdown_capacity_overflow_rejected(self):
        snap = {"A1": _atm_row(remaining=100.0)}
        m = RecordDrawdown(instrument_id="A1", drawdown_shares=1, event_date=D,
                           drawdown_amount_usd=200.0)
        r = _validate_one(m, snap, {"A1"}, {})
        assert r.error_kind == "capacity_overflow"

    def test_drawdown_within_capacity_ok(self):
        snap = {"A1": _atm_row(remaining=100.0)}
        m = RecordDrawdown(instrument_id="A1", drawdown_shares=1, event_date=D,
                           drawdown_amount_usd=50.0)
        r = _validate_one(m, snap, {"A1"}, {})
        assert r.accepted

    def test_conversion_preferred_without_shares_rejected(self):
        snap = {"P1": _preferred_row(count=100.0)}
        m = RecordConversion(instrument_id="P1", shares_issued=1, event_date=D)
        r = _validate_one(m, snap, {"P1"}, {})
        assert r.error_kind == "preferred_shares_required"

    def test_conversion_preferred_with_stated_value_and_principal_ok(self):
        snap = {"P1": _preferred_row(count=100.0,
                                     terms={"stated_value": 1000.0})}
        m = RecordConversion(instrument_id="P1", shares_issued=1, event_date=D,
                             principal_converted=5000)
        r = _validate_one(m, snap, {"P1"}, {})
        assert r.accepted

    def test_conversion_preferred_with_pref_shares_ok(self):
        snap = {"P1": _preferred_row(count=100.0)}
        m = RecordConversion(instrument_id="P1", shares_issued=1, event_date=D,
                             preferred_shares_converted=5)
        r = _validate_one(m, snap, {"P1"}, {})
        assert r.accepted

    def test_conversion_convertible_without_principal_rejected(self):
        snap = {"C1": _convertible_row()}
        m = RecordConversion(instrument_id="C1", shares_issued=1, event_date=D)
        r = _validate_one(m, snap, {"C1"}, {})
        assert r.error_kind == "principal_converted_required"

    def test_partial_redemption_preferred_without_shares_rejected(self):
        snap = {"P1": _preferred_row(count=100.0)}
        m = RecordPartialRedemption(instrument_id="P1", event_date=D)
        r = _validate_one(m, snap, {"P1"}, {})
        assert r.error_kind == "preferred_shares_redeemed_required"

    def test_partial_redemption_preferred_stated_value_translatable_ok(self):
        snap = {"P1": _preferred_row(count=100.0,
                                     terms={"stated_value": 1000.0})}
        m = RecordPartialRedemption(instrument_id="P1", event_date=D,
                                    principal_redeemed=5000)
        r = _validate_one(m, snap, {"P1"}, {})
        assert r.accepted

    def test_partial_redemption_convertible_without_principal_rejected(self):
        snap = {"C1": _convertible_row()}
        m = RecordPartialRedemption(instrument_id="C1", event_date=D)
        r = _validate_one(m, snap, {"C1"}, {})
        assert r.error_kind == "principal_redeemed_required"

    def test_partial_redemption_convertible_with_principal_ok(self):
        snap = {"C1": _convertible_row()}
        m = RecordPartialRedemption(instrument_id="C1", event_date=D,
                                    principal_redeemed=100)
        r = _validate_one(m, snap, {"C1"}, {})
        assert r.accepted


class TestValidateOneAmend:
    def test_amend_on_terminal_rejected(self):
        snap = {"W1": _warrant_row(status="terminated")}
        m = AmendWarrant(instrument_id="W1", event_date=D, count=5)
        r = _validate_one(m, snap, {"W1"}, {})
        assert r.error_kind == "illegal_transition"

    def test_amend_on_superseded_rejected(self):
        snap = {"W1": _warrant_row(status="superseded:NEW")}
        m = AmendWarrant(instrument_id="W1", event_date=D, count=5)
        r = _validate_one(m, snap, {"W1"}, {})
        assert r.error_kind == "illegal_transition"

    def test_liq_pref_inconsistent_rejected(self):
        # count 1000 * stated 34000 = 34M; amend liq_pref to 29M (>1% off).
        snap = {"P1": _preferred_row(count=1000.0,
                                     terms={"stated_value": 34000.0})}
        m = AmendPreferred(instrument_id="P1", event_date=D,
                           liquidation_preference=29_000_000)
        r = _validate_one(m, snap, {"P1"}, {})
        assert r.error_kind == "liq_pref_inconsistent"

    def test_liq_pref_within_one_percent_ok(self):
        snap = {"P1": _preferred_row(count=1000.0,
                                     terms={"stated_value": 34000.0})}
        m = AmendPreferred(instrument_id="P1", event_date=D,
                           liquidation_preference=34_100_000)  # ~0.29% off
        r = _validate_one(m, snap, {"P1"}, {})
        assert r.accepted

    def test_liq_pref_check_skipped_when_count_moves(self):
        snap = {"P1": _preferred_row(count=1000.0,
                                     terms={"stated_value": 34000.0})}
        m = AmendPreferred(instrument_id="P1", event_date=D,
                           liquidation_preference=29_000_000, count=853)
        r = _validate_one(m, snap, {"P1"}, {})
        assert r.accepted

    def test_liq_pref_check_skipped_when_stated_value_missing(self):
        snap = {"P1": _preferred_row(count=1000.0, terms={})}
        m = AmendPreferred(instrument_id="P1", event_date=D,
                           liquidation_preference=29_000_000)
        r = _validate_one(m, snap, {"P1"}, {})
        assert r.accepted

    def test_liq_pref_check_skipped_when_count_zero(self):
        snap = {"P1": _preferred_row(count=0.0,
                                     terms={"stated_value": 34000.0})}
        m = AmendPreferred(instrument_id="P1", event_date=D,
                           liquidation_preference=29_000_000)
        r = _validate_one(m, snap, {"P1"}, {})
        assert r.accepted


class TestValidateOneClose:
    def test_close_sales_program_from_posam_rejected(self):
        snap = {"A1": _atm_row()}
        m = CloseInstrument(instrument_id="A1", reason="terminated",
                            event_date=D)
        r = _validate_one(m, snap, {"A1"}, {}, None, "POS AM")
        assert r.error_kind == "close_on_post_effective_amendment"

    def test_close_terminal_non_superseded_rejected(self):
        snap = {"W1": _warrant_row(count=0.0, status="redeemed")}
        m = CloseInstrument(instrument_id="W1", reason="expired", event_date=D)
        r = _validate_one(m, snap, {"W1"}, {})
        assert r.error_kind == "illegal_transition"

    def test_close_terminal_superseded_allowed(self):
        snap = {"W1": _warrant_row(count=0.0, status="redeemed")}
        m = CloseInstrument(instrument_id="W1", reason="superseded",
                            event_date=D, replaced_by="W2")
        r = _validate_one(m, snap, {"W1"}, {})
        assert r.accepted

    def test_close_superseded_without_replaced_by_rejected(self):
        snap = {"W1": _warrant_row(count=0.0)}
        m = CloseInstrument(instrument_id="W1", reason="superseded",
                            event_date=D)
        r = _validate_one(m, snap, {"W1"}, {})
        assert r.error_kind == "missing_replaced_by"

    def test_redeemed_with_outstanding_blocked(self):
        snap = {"C1": _convertible_row(principal_remaining=500.0)}
        m = CloseInstrument(instrument_id="C1", reason="redeemed", event_date=D)
        r = _validate_one(m, snap, {"C1"}, {}, None, "10-Q")
        assert r.error_kind == "close_with_outstanding"

    def test_converted_with_outstanding_blocked(self):
        snap = {"C1": _convertible_row(principal_remaining=500.0)}
        m = CloseInstrument(instrument_id="C1", reason="converted", event_date=D)
        r = _validate_one(m, snap, {"C1"}, {})
        assert r.error_kind == "close_with_outstanding"

    def test_exercised_warrant_with_count_blocked(self):
        snap = {"W1": _warrant_row(count=5.0)}
        m = CloseInstrument(instrument_id="W1", reason="exercised", event_date=D)
        r = _validate_one(m, snap, {"W1"}, {})
        assert r.error_kind == "close_with_outstanding"

    def test_terminated_with_outstanding_blocked(self):
        snap = {"C1": _convertible_row(principal_remaining=500.0)}
        m = CloseInstrument(instrument_id="C1", reason="terminated", event_date=D)
        r = _validate_one(m, snap, {"C1"}, {})
        assert r.error_kind == "close_with_outstanding"

    @pytest.mark.parametrize("ff", ["8-K", "6-K"])
    def test_redeemed_with_outstanding_exempt_from_event_filing(self, ff):
        snap = {"C1": _convertible_row(principal_remaining=852.0)}
        m = CloseInstrument(instrument_id="C1", reason="redeemed", event_date=D)
        r = _validate_one(m, snap, {"C1"}, {}, None, ff)
        assert r.accepted

    def test_redeemed_with_outstanding_periodic_still_blocked(self):
        snap = {"C1": _convertible_row(principal_remaining=852.0)}
        m = CloseInstrument(instrument_id="C1", reason="redeemed", event_date=D)
        r = _validate_one(m, snap, {"C1"}, {}, None, "10-K")
        assert r.error_kind == "close_with_outstanding"

    def test_converted_still_blocked_from_8k(self):
        # Exemption only applies to reason='redeemed', not 'converted'.
        snap = {"C1": _convertible_row(principal_remaining=500.0)}
        m = CloseInstrument(instrument_id="C1", reason="converted", event_date=D)
        r = _validate_one(m, snap, {"C1"}, {}, None, "8-K")
        assert r.error_kind == "close_with_outstanding"

    def test_close_with_zero_outstanding_ok(self):
        snap = {"C1": _convertible_row(principal_remaining=0.0)}
        m = CloseInstrument(instrument_id="C1", reason="redeemed", event_date=D)
        r = _validate_one(m, snap, {"C1"}, {}, None, "10-Q")
        assert r.accepted

    def test_preferred_uses_count_as_gate(self):
        # preferred carries value in count; redeemed with count>0 blocked.
        snap = {"P1": _preferred_row(count=50.0)}
        m = CloseInstrument(instrument_id="P1", reason="redeemed", event_date=D)
        r = _validate_one(m, snap, {"P1"}, {}, None, "10-Q")
        assert r.error_kind == "close_with_outstanding"
        assert "count=" in r.message


class TestValidateOneUnknownKind:
    def test_unknown_kind_with_existing_id(self):
        # An object with instrument_id that exists but isn't an
        # Apply/Create/Restate/Record/Amend/Close instance reaches the
        # final unknown_kind branch.
        fake = SimpleNamespace(kind="weird", instrument_id="X")
        snap = {"X": _warrant_row()}
        r = _validate_one(fake, snap, {"X"}, {})
        assert r.error_kind == "unknown_kind"
        assert "weird" in r.message

    def test_note_no_event_crashes_before_unknown_kind(self):
        # BUG / characterization: NoteNoEvent has NO instrument_id attr, so
        # _validate_one raises AttributeError at `m.instrument_id` (line 670)
        # instead of returning the documented `unknown_kind` rejection. The
        # survey lists "NoteNoEvent -> reject unknown_kind" as intended, but
        # NoteNoEvent never reaches that branch — it is filtered out of the
        # mutation stream upstream (kind 'note_no_event'), so validate is
        # never asked to judge it. Asserting actual behavior here.
        m = NoteNoEvent(reason="nothing")
        with pytest.raises(AttributeError):
            _validate_one(m, {}, set(), {})


# ═══════════════════════════════════════════════════════════════════════
class TestValidateMutations:
    def test_empty(self):
        rep = validate_mutations([], {})
        assert isinstance(rep, ValidationReport)
        assert rep.accepted == []
        assert rep.rejected == []

    def test_ledger_snapshot_keys_seed_live_ids(self):
        snap = {"W1": _warrant_row(count=10.0)}
        m = RecordExercise(instrument_id="W1", shares=1, event_date=D)
        rep = validate_mutations([m], snap)
        assert len(rep.accepted) == 1
        assert rep.rejected == []

    def test_proposed_id_chaining(self):
        # accepted create with proposed_id → later record_event passes.
        create = CreateWarrant(count=10, strike=1, event_date=D,
                               expiration=date(2026, 1, 1), proposed_id="WNEW")
        rec = RecordExercise(instrument_id="WNEW", shares=1, event_date=D)
        rep = validate_mutations([create, rec], {})
        assert len(rep.accepted) == 2
        assert rep.rejected == []

    def test_restate_successor_drawdown_chaining(self):
        # accepted RestateAtm with proposed_id → in-filing drawdown passes.
        snap = {"A0": _atm_row(remaining=1e6)}
        restate = RestateAtm(predecessor_id="A0", capacity_usd=1e6,
                             event_date=D, proposed_id="ANEW",
                             remaining_capacity_usd=1e6)
        dd = RecordDrawdown(instrument_id="ANEW", drawdown_shares=1,
                            event_date=D, drawdown_amount_usd=1000.0)
        rep = validate_mutations([restate, dd], snap, filing_form="424B5")
        assert len(rep.accepted) == 2
        assert rep.rejected == []

    def test_sanitization_carry_through(self):
        # accepted create's stored mutation holds the NULLED CP, not raw.
        create = CreateWarrant(count=1, strike=1, event_date=D,
                               counterparty_canonical="warrants to")
        rep = validate_mutations([create], {})
        assert len(rep.accepted) == 1
        assert rep.accepted[0].counterparty_canonical is None

    def test_close_before_amend_reordered(self):
        # close emitted before its amend; amend rank 2 < close rank 4 so the
        # overlay zeroing is visible → close passes.
        snap = {"C1": _convertible_row(principal_remaining=500.0)}
        close = CloseInstrument(instrument_id="C1", reason="redeemed",
                                event_date=D)
        amend = AmendConvertible(instrument_id="C1", event_date=D,
                                 principal_remaining=0)
        rep = validate_mutations([close, amend], snap, filing_form="10-Q")
        assert len(rep.accepted) == 2
        assert rep.rejected == []

    def test_full_exercise_then_close(self):
        # record full exercise + close(exercised) in one batch → overlay
        # decrements count to 0 so close passes (ACTU W-4341 scenario).
        snap = {"W1": _warrant_row(count=76.0)}
        ex = RecordExercise(instrument_id="W1", shares=76, event_date=D)
        close = CloseInstrument(instrument_id="W1", reason="exercised",
                                event_date=D)
        rep = validate_mutations([close, ex], snap)
        assert len(rep.accepted) == 2
        assert rep.rejected == []

    def test_anchor_zombie_amend_then_close(self):
        # anchor pattern: amend(principal_remaining=0) + close(redeemed)
        # passes despite stale snapshot.
        snap = {"C1": _convertible_row(principal_remaining=1000.0)}
        amend = AmendConvertible(instrument_id="C1", event_date=D,
                                 principal_remaining=0)
        close = CloseInstrument(instrument_id="C1", reason="redeemed",
                                event_date=D)
        rep = validate_mutations([amend, close], snap, filing_form="10-Q")
        assert len(rep.accepted) == 2
        assert rep.rejected == []

    def test_one_bad_does_not_kill_siblings(self):
        snap = {
            "W1": _warrant_row(count=10.0),
            "C1": _convertible_row(principal_remaining=500.0),
        }
        good = RecordExercise(instrument_id="W1", shares=1, event_date=D)
        bad = CloseInstrument(instrument_id="C1", reason="redeemed",
                              event_date=D)  # outstanding>0, periodic-ish
        rep = validate_mutations([good, bad], snap, filing_form="10-Q")
        assert len(rep.accepted) == 1
        assert len(rep.rejected) == 1
        assert rep.rejected[0].error_kind == "close_with_outstanding"
        # the surviving accepted is the good exercise
        assert rep.accepted[0] is good

    def test_filing_form_none_skips_form_gates_but_keeps_terms_form(self):
        # filing_form None → shelf wrong-filing-form gate skipped, but the
        # terms.form-based shelf_misclassified check still applies.
        bad_shelf = CreateShelf(capacity_usd=1e6, event_date=D, form="424B5")
        rep = validate_mutations([bad_shelf], {}, filing_form=None)
        assert len(rep.rejected) == 1
        assert rep.rejected[0].error_kind == "shelf_misclassified"

    def test_rejected_carries_error_kind_and_message(self):
        snap = {"C1": _convertible_row(principal_remaining=500.0)}
        bad = CloseInstrument(instrument_id="C1", reason="converted",
                              event_date=D)
        rep = validate_mutations([bad], snap)
        assert len(rep.rejected) == 1
        v = rep.rejected[0]
        assert isinstance(v, ValidationResult)
        assert v.accepted is False
        assert v.error_kind == "close_with_outstanding"
        assert v.message


# ═══════════════════════════════════════════════════════════════════════
class TestDataclassesAndConstants:
    def test_drawdown_tolerance_constant(self):
        assert DRAWDOWN_TOLERANCE == 0.05

    def test_validation_result_defaults(self):
        m = NoteNoEvent(reason="x")
        r = ValidationResult(mutation=m, accepted=True)
        assert r.error_kind is None
        assert r.message is None

    def test_validation_report_defaults(self):
        rep = ValidationReport()
        assert rep.accepted == []
        assert rep.rejected == []


# ═══════════════════════════════════════════════════════════════════════
# Coverage added by adversarial review: event kinds and close paths that
# the original suite skipped. Every expected value below was re-derived
# from validate.py source and confirmed by observing the function.
# ═══════════════════════════════════════════════════════════════════════
class TestValidateOnePartialTermination:
    """`partial_termination` is valid on atm/equity_line/shelf only
    (_EVENT_KIND_TYPES). The original suite never exercised it."""

    @pytest.mark.parametrize(
        "row", [_atm_row(), _equity_line_row(), _shelf_row()]
    )
    def test_accepted_on_capacity_program(self, row):
        m = RecordPartialTermination(instrument_id="X",
                                     capacity_reduced_usd=10.0, event_date=D)
        r = _validate_one(m, {"X": row}, {"X"}, {})
        assert r.accepted

    def test_type_mismatch_on_warrant(self):
        m = RecordPartialTermination(instrument_id="W1",
                                     capacity_reduced_usd=10.0, event_date=D)
        r = _validate_one(m, {"W1": _warrant_row()}, {"W1"}, {})
        assert r.error_kind == "type_mismatch"

    def test_type_mismatch_on_convertible(self):
        m = RecordPartialTermination(instrument_id="C1",
                                     capacity_reduced_usd=10.0, event_date=D)
        r = _validate_one(m, {"C1": _convertible_row()}, {"C1"}, {})
        assert r.error_kind == "type_mismatch"

    def test_partial_termination_is_event_noop_in_overlay(self):
        # _accumulate_event_effect only folds exercise/conversion/
        # partial_redemption; partial_termination hits the else branch.
        snap = {"A1": _atm_row(remaining=100.0)}
        overlay = {}
        m = RecordPartialTermination(instrument_id="A1",
                                     capacity_reduced_usd=10.0, event_date=D)
        _accumulate_event_effect(m, snap, overlay)
        assert overlay == {}


class TestValidateOneConfirmClosing:
    """`closing` (ConfirmClosing) is valid on warrant/convertible/
    preferred/equity. Untested by the original suite."""

    def test_closing_on_warrant_accepted(self):
        m = ConfirmClosing(instrument_id="W1", event_date=D, count_actual=5)
        r = _validate_one(m, {"W1": _warrant_row()}, {"W1"}, {})
        assert r.accepted

    def test_closing_on_convertible_accepted(self):
        m = ConfirmClosing(instrument_id="C1", event_date=D)
        r = _validate_one(m, {"C1": _convertible_row()}, {"C1"}, {})
        assert r.accepted

    def test_closing_on_preferred_accepted(self):
        m = ConfirmClosing(instrument_id="P1", event_date=D)
        r = _validate_one(m, {"P1": _preferred_row()}, {"P1"}, {})
        assert r.accepted

    def test_closing_on_atm_type_mismatch(self):
        # atm is NOT in the closing-compatible set.
        m = ConfirmClosing(instrument_id="A1", event_date=D)
        r = _validate_one(m, {"A1": _atm_row()}, {"A1"}, {})
        assert r.error_kind == "type_mismatch"

    def test_closing_is_event_noop_in_overlay(self):
        snap = {"W1": _warrant_row(count=100.0)}
        overlay = {}
        m = ConfirmClosing(instrument_id="W1", event_date=D, count_actual=5)
        _accumulate_event_effect(m, snap, overlay)
        # 'closing' is not a decrementing kind → else branch returns.
        assert overlay == {}


class TestValidateOneClosePrograms:
    """Close-out behavior on capacity programs (atm/equity_line/shelf/
    s1_offering) — the original suite only covered atm-from-POSAM and
    the warrant/preferred/convertible outstanding gate."""

    @pytest.mark.parametrize(
        "row,ff",
        [
            (_equity_line_row(), "POSAM"),
            (_s1_offering_row(), "POS AM"),
            (_atm_row(), "POSASR"),
        ],
    )
    def test_sales_program_close_from_post_effective_blocked(self, row, ff):
        m = CloseInstrument(instrument_id="X", reason="terminated",
                            event_date=D)
        r = _validate_one(m, {"X": row}, {"X"}, {}, None, ff)
        assert r.error_kind == "close_on_post_effective_amendment"

    def test_shelf_not_a_sales_program_passes_posam_gate(self):
        # shelf is NOT in _SALES_PROGRAM_TYPES, so the POS-AM gate does
        # not fire; the outstanding gate uses principal_remaining (None
        # on a shelf row) so the terminated close is accepted.
        m = CloseInstrument(instrument_id="SH1", reason="terminated",
                            event_date=D)
        r = _validate_one(m, {"SH1": _shelf_row(remaining=100.0)},
                          {"SH1"}, {}, None, "POSAM")
        assert r.accepted

    def test_atm_terminated_with_remaining_capacity_accepted(self):
        # BUG-ADJACENT / characterization: the outstanding-zero gate keys
        # off principal_remaining for non-warrant/preferred types. An ATM
        # tracks value in remaining_capacity_usd, NOT principal_remaining,
        # so principal_remaining is None and the `terminated` block never
        # fires — the close passes even with $100 of capacity left. This
        # is intentional: ATM capacity is not a debt balance, and program
        # supersession/termination is owned elsewhere. Asserting the
        # actual current behavior so a future regression here is visible.
        m = CloseInstrument(instrument_id="A1", reason="terminated",
                            event_date=D)
        r = _validate_one(m, {"A1": _atm_row(remaining=100.0)},
                          {"A1"}, {}, None, "8-K")
        assert r.accepted

    def test_atm_close_from_non_posam_not_blocked_by_program_gate(self):
        # A plain 424B5 / 8-K is not a post-effective amendment, so the
        # sales-program gate is silent.
        m = CloseInstrument(instrument_id="A1", reason="terminated",
                            event_date=D)
        r = _validate_one(m, {"A1": _atm_row()}, {"A1"}, {}, None, "424B5")
        assert r.accepted


class TestValidateOneCloseExpiredReason:
    """`expired` appears in the close reason tuple at validate.py L901 but
    has NO blocking branch (unlike redeemed/converted/exercised/
    terminated). Past-maturity is not a close-out trigger per the source
    comment, so an `expired` close with outstanding>0 is accepted."""

    def test_expired_warrant_with_count_accepted(self):
        m = CloseInstrument(instrument_id="W1", reason="expired", event_date=D)
        r = _validate_one(m, {"W1": _warrant_row(count=5.0)},
                          {"W1"}, {}, None, "10-K")
        assert r.accepted

    def test_expired_convertible_with_principal_accepted(self):
        m = CloseInstrument(instrument_id="C1", reason="expired", event_date=D)
        r = _validate_one(m, {"C1": _convertible_row(principal_remaining=500.0)},
                          {"C1"}, {}, None, "10-K")
        assert r.accepted


class TestLiqPrefBoundary:
    """The liq-pref identity guard rejects when the relative deviation is
    strictly > 0.01 (1%). The original suite tested 0.29% (pass) and a
    large miss (reject) but not the exact boundary."""

    def test_exactly_one_percent_off_accepted(self):
        # implied = 1000 * 34000 = 34,000,000; +1% = 34,340,000.
        # deviation == 0.01, NOT > 0.01 → accepted.
        snap = {"P1": _preferred_row(count=1000.0,
                                     terms={"stated_value": 34000.0})}
        m = AmendPreferred(instrument_id="P1", event_date=D,
                           liquidation_preference=34_340_000)
        r = _validate_one(m, snap, {"P1"}, {})
        assert r.accepted

    def test_just_over_one_percent_off_rejected(self):
        snap = {"P1": _preferred_row(count=1000.0,
                                     terms={"stated_value": 34000.0})}
        m = AmendPreferred(instrument_id="P1", event_date=D,
                           liquidation_preference=34_350_000)  # ~1.03%
        r = _validate_one(m, snap, {"P1"}, {})
        assert r.error_kind == "liq_pref_inconsistent"

    def test_liq_pref_on_non_preferred_skipped(self):
        # The identity check only applies to preferred. A liq_pref-shaped
        # amend on a non-preferred never reaches it (AmendConvertible has
        # no liquidation_preference attr → getattr None → check skipped).
        snap = {"C1": _convertible_row(principal_remaining=1000.0)}
        m = AmendConvertible(instrument_id="C1", event_date=D, conv_price=2.0)
        r = _validate_one(m, snap, {"C1"}, {})
        assert r.accepted


class TestAccumulateEventEffectChaining:
    """Overlay-base chaining for record_event effects (the amend path had
    a chaining test; the event path did not)."""

    def test_second_exercise_decrements_from_overlay_not_snapshot(self):
        snap = {"W1": _warrant_row(count=100.0)}
        overlay = {"W1": {"type": "warrant", "status": "active",
                          "terms_json": {},
                          "outstanding_json": {"count": 50.0}}}
        m = RecordExercise(instrument_id="W1", shares=20, event_date=D)
        _accumulate_event_effect(m, snap, overlay)
        # 50 (overlay) - 20, NOT 100 (snapshot) - 20.
        assert overlay["W1"]["outstanding_json"]["count"] == 30.0

    def test_conversion_zero_principal_no_decrement(self):
        # principal_converted=0 → _dec by<=0 short-circuits; no explicit
        # principal_remaining in fields → value unchanged.
        snap = {"C1": _convertible_row(principal_remaining=1000.0)}
        overlay = {}
        m = RecordConversion(instrument_id="C1", shares_issued=1, event_date=D,
                             principal_converted=0)
        _accumulate_event_effect(m, snap, overlay)
        assert overlay["C1"]["outstanding_json"]["principal_remaining"] == 1000.0

    def test_conversion_explicit_principal_remaining_zero_sets_absolute(self):
        # Anchor matured-zombie shape: principal_remaining=0 set absolutely
        # (the close gate then sees 0 and accepts a redeemed close).
        snap = {"C1": _convertible_row(principal_remaining=1000.0)}
        overlay = {}
        m = RecordConversion(instrument_id="C1", shares_issued=1, event_date=D,
                             principal_converted=200, principal_remaining=0)
        _accumulate_event_effect(m, snap, overlay)
        assert overlay["C1"]["outstanding_json"]["principal_remaining"] == 0.0


class TestValidateMutationsRejectedRestateCascade:
    """When a RestateAtm is rejected, its proposed_id successor is NOT
    seeded into live_ids, so an in-filing drawdown against it cascades to
    missing_id. The accept path of this is covered by
    test_restate_successor_drawdown_chaining; this is the reject path."""

    def test_rejected_restate_breaks_successor_drawdown(self):
        snap = {"A0": _atm_row(remaining=1e6)}
        restate = RestateAtm(predecessor_id="A0", capacity_usd=1e6,
                             event_date=D, proposed_id="ANEW",
                             remaining_capacity_usd=1e6)
        dd = RecordDrawdown(instrument_id="ANEW", drawdown_shares=1,
                            event_date=D, drawdown_amount_usd=1000.0)
        # 8-K is not an allowed restate form → restate rejected.
        rep = validate_mutations([restate, dd], snap, filing_form="8-K")
        assert rep.accepted == []
        kinds = sorted(r.error_kind for r in rep.rejected)
        assert kinds == ["missing_id", "restate_wrong_filing_form"]


class TestSortMutationsStabilityDeep:
    def test_three_same_rank_records_keep_order(self):
        a = RecordExercise(instrument_id="A", shares=1, event_date=D)
        b = RecordConversion(instrument_id="B", shares_issued=1, event_date=D,
                             principal_converted=10)
        c = RecordPartialRedemption(instrument_id="C", event_date=D,
                                    principal_redeemed=10)
        out = sort_mutations([c, a, b])
        # all rank 3 (record_event) → declared order [c, a, b] preserved.
        assert out == [c, a, b]

    def test_interleaved_full_pipeline_orders_by_rank_then_declared(self):
        split = ApplySplit(post=2, pre=1, direction="forward",
                           effective_date=D)
        c1 = CreateWarrant(count=1, strike=1, event_date=D, proposed_id="C1")
        c2 = CreateConvertible(principal=1, principal_remaining=1, event_date=D,
                               proposed_id="C2")
        rec = RecordExercise(instrument_id="C1", shares=1, event_date=D)
        close = CloseInstrument(instrument_id="C2", reason="redeemed",
                                event_date=D)
        out = sort_mutations([close, rec, c2, c1, split])
        # rank order: split(0) < c2,c1(1) < rec(3) < close(4); within
        # rank 1 the declared order is c2 then c1.
        assert out == [split, c2, c1, rec, close]
