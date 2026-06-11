"""Unit tests for dilution/ledger/view.py.

This is a PURE module: no DB, no network, no LLM, no filesystem. We build
plain dict rows by hand (mirroring store.get_open_instruments output) and
assert exact rendered strings / derived values. The autouse temp_db fixture
from conftest.py is present but unused here.

Determinism: every test that touches bucketing passes an explicit
today=date(2026, 6, 10) so the RECENT_CLOSED_DAYS / ARCHIVE_CLOSED_DAYS
cutoffs are stable. No module-level caches to clear.
"""

from __future__ import annotations

from datetime import date

import pytest

from dilution.ledger import view
from dilution.ledger.view import (
    ARCHIVE_CLOSED_DAYS,
    DEFAULT_MAX_CHARS,
    RECENT_CLOSED_DAYS,
    _bucket,
    _collapse_warrants,
    _fmt_takedown,
    _fmt_val,
    _format_cell,
    _format_status,
    _get_field,
    _render_archive_row,
    _render_buckets,
    _render_section,
    _row_flags,
    _short_counterparty,
    _takedown_line,
    _to_dict,
    render_ledger_view,
)

TODAY = date(2026, 6, 10)
# Cutoffs derived exactly the way the module derives them.
CUTOFF_RECENT = date.fromordinal(TODAY.toordinal() - RECENT_CLOSED_DAYS).isoformat()
CUTOFF_ARCHIVE = date.fromordinal(TODAY.toordinal() - ARCHIVE_CLOSED_DAYS).isoformat()


# ───────────────────────────── _fmt_val ──────────────────────────────
class TestFmtVal:
    def test_bool_true_is_Y(self):
        # bool must be checked BEFORE int branch (bool is an int subclass)
        assert _fmt_val(True) == "Y"

    def test_bool_false_is_N(self):
        assert _fmt_val(False) == "N"

    @pytest.mark.parametrize("z", [0, 0.0])
    def test_exact_zero_short_circuits(self, z):
        assert _fmt_val(z) == "0"

    def test_one_million_comma_grouped(self):
        assert _fmt_val(1_000_000) == "1,000,000"

    def test_one_thousand_comma_grouped(self):
        # >= 1000 branch inside the 1 <= abs region
        assert _fmt_val(1000) == "1,000"

    def test_just_below_thousand_uses_4g(self):
        assert _fmt_val(999.5) == "999.5"

    def test_just_below_one_uses_4g(self):
        assert _fmt_val(0.001) == "0.001"

    def test_simple_decimal(self):
        assert _fmt_val(1.25) == "1.25"

    def test_four_sig_figs_rounding(self):
        assert _fmt_val(1.23456) == "1.235"

    def test_negative_million_preserves_sign(self):
        # abs() selects the tier but the sign is preserved in output
        assert _fmt_val(-1_500_000) == "-1,500,000"

    def test_negative_small_preserves_sign(self):
        assert _fmt_val(-0.5) == "-0.5"

    def test_large_float_with_fraction_rounds_no_decimals(self):
        # 500000.7 is in the >=1000 comma branch -> %,.0f rounds to int
        assert _fmt_val(500000.7) == "500,001"

    def test_4g_rounds_up_across_thousand_boundary(self):
        # 999.95 is < 1000 so uses %.4g, which rounds to 1000
        # (no comma since formatting is %.4g, not %,.0f). Observed behavior.
        assert _fmt_val(999.95) == "1000"

    def test_list_compact_json(self):
        assert _fmt_val([1, 2]) == "[1,2]"

    def test_dict_compact_json(self):
        assert _fmt_val({"a": 1}) == '{"a":1}'

    def test_non_numeric_string_passthrough(self):
        assert _fmt_val("hello") == "hello"

    def test_plain_int_one(self):
        assert _fmt_val(1) == "1"

    def test_value_near_one_four_g(self):
        assert _fmt_val(0.9999) == "0.9999"


# ───────────────────────────── _to_dict ──────────────────────────────
class TestToDict:
    def test_already_dict_returned_as_is(self):
        d = {"a": 1}
        assert _to_dict({"terms": d}, "terms") == {"a": 1}

    def test_json_string_parsed(self):
        assert _to_dict({"terms_json": '{"a": 1}'}, "terms") == {"a": 1}

    def test_malformed_json_returns_empty(self):
        assert _to_dict({"terms_json": "{bad"}, "terms") == {}

    def test_json_null_returns_empty(self):
        # json.loads('null') -> None -> 'or {}' -> {}
        assert _to_dict({"terms_json": "null"}, "terms") == {}

    def test_empty_string_json_returns_empty(self):
        # falsy raw -> skip json.loads -> {}
        assert _to_dict({"terms_json": ""}, "terms") == {}

    def test_neither_key_present_returns_empty(self):
        assert _to_dict({}, "terms") == {}

    def test_outstanding_key_decoded(self):
        assert _to_dict({"outstanding_json": '{"count": 5}'}, "outstanding") == {
            "count": 5
        }


# ──────────────────────────── _get_field ─────────────────────────────
class TestGetField:
    def test_counterparty_from_canonical(self):
        assert _get_field({"counterparty_canonical": "Foo"}, "counterparty") == "Foo"

    def test_counterparty_empty_string_is_none(self):
        assert _get_field({"counterparty_canonical": ""}, "counterparty") is None

    def test_placement_agent_from_canonical(self):
        assert (
            _get_field({"placement_agent_canonical": "Bar"}, "placement_agent") == "Bar"
        )

    def test_placement_agent_falsy_is_none(self):
        assert _get_field({"placement_agent_canonical": ""}, "placement_agent") is None

    def test_terms_beats_outstanding(self):
        row = {"terms": {"strike": 1.5}, "outstanding": {"strike": 2.0}}
        assert _get_field(row, "strike") == 1.5

    def test_terms_none_falls_through_to_outstanding(self):
        row = {"terms": {"strike": None}, "outstanding": {"strike": 2.0}}
        assert _get_field(row, "strike") == 2.0

    def test_outstanding_only(self):
        assert _get_field({"outstanding": {"count": 100}}, "count") == 100

    def test_key_in_neither_returns_none(self):
        assert _get_field({}, "strike") is None

    def test_zero_in_terms_is_returned(self):
        # guard is `is not None`, not truthiness — 0 propagates
        assert _get_field({"terms": {"strike": 0}}, "strike") == 0


# ──────────────────────────── _row_flags ─────────────────────────────
class TestRowFlags:
    def test_no_flags_returns_none(self):
        assert _row_flags({}) is None

    def test_pre_funded_flag(self):
        assert _row_flags({"terms": {"is_pre_funded": True}}) == "pre-funded"

    def test_splits_flag_with_count(self):
        assert _row_flags({"terms": {"applied_splits": [1, 2, 3]}}) == "splits×3"

    def test_empty_splits_list_no_flag(self):
        assert _row_flags({"terms": {"applied_splits": []}}) is None

    def test_splits_none_no_flag(self):
        assert _row_flags({"terms": {"applied_splits": None}}) is None

    def test_tranche_count_float_int_coerced(self):
        assert _row_flags({"outstanding": {"tranche_count": 3.0}}) == "×3 tranches"

    def test_tranche_count_zero_no_flag(self):
        assert _row_flags({"outstanding": {"tranche_count": 0}}) is None

    def test_tranche_count_none_no_flag(self):
        assert _row_flags({"outstanding": {"tranche_count": None}}) is None

    def test_multiple_flags_joined_in_order(self):
        row = {
            "terms": {"is_pre_funded": True, "applied_splits": [1, 2]},
            "outstanding": {"tranche_count": 3.0},
        }
        assert _row_flags(row) == "pre-funded, splits×2, ×3 tranches"


# ──────────────────────────── _fmt_takedown ──────────────────────────
class TestFmtTakedown:
    def test_missing_event_date_question_mark(self):
        assert _fmt_takedown({}) == "?"

    def test_full_takedown(self):
        t = {
            "event_date": "2025-06-01",
            "amount_usd": 1_000_000,
            "shares": 5000,
            "price": 2.5,
            "drawdown_party_canonical": "  Maxim Group LLC  ",
        }
        assert _fmt_takedown(t) == "2025-06-01 $1,000,000 (5,000 sh @$2.5) [Maxim Group LLC]"

    def test_zero_amount_omitted(self):
        # falsy amount guard -> '$' omitted
        assert _fmt_takedown({"event_date": "2025-06-01", "amount_usd": 0}) == "2025-06-01"

    def test_shares_only(self):
        assert _fmt_takedown({"event_date": "2025-06-01", "shares": 5000}) == (
            "2025-06-01 (5,000 sh)"
        )

    def test_price_only(self):
        assert _fmt_takedown({"event_date": "2025-06-01", "price": 2.5}) == (
            "2025-06-01 (@$2.5)"
        )

    def test_date_only_no_parens(self):
        # no amount/shares/price/party -> just the date, no parens, no '$'
        assert _fmt_takedown({"event_date": "2025-06-01"}) == "2025-06-01"

    def test_amount_only_no_share_parens(self):
        # amount present but no shares/price -> '$amt' with no (..) group
        assert _fmt_takedown({"event_date": "d", "amount_usd": 500}) == "d $500"

    def test_zero_shares_and_price_omit_parens(self):
        # falsy shares & price -> no parens; only the (truthy) amount renders
        assert _fmt_takedown(
            {"event_date": "d", "amount_usd": 1000, "shares": 0, "price": 0}
        ) == "d $1,000"

    def test_party_exactly_24_chars_not_truncated(self):
        # condition is > 24, so exactly 24 is kept whole
        p = "X" * 24
        assert _fmt_takedown({"event_date": "d", "drawdown_party_canonical": p}) == (
            f"d [{p}]"
        )

    def test_party_25_chars_truncated_to_24(self):
        p = "Y" * 25
        assert _fmt_takedown({"event_date": "d", "drawdown_party_canonical": p}) == (
            "d [" + "Y" * 24 + "]"
        )

    def test_party_whitespace_stripped(self):
        assert _fmt_takedown(
            {"event_date": "d", "drawdown_party_canonical": "  Acme  "}
        ) == "d [Acme]"

    def test_string_numeric_values_coerced(self):
        # amount/shares/price coerced via float()
        t = {"event_date": "d", "amount_usd": "1000", "shares": "200", "price": "1.5"}
        assert _fmt_takedown(t) == "d $1,000 (200 sh @$1.5)"


# ──────────────────────────── _takedown_line ─────────────────────────
class TestTakedownLine:
    def test_missing_instrument_id_empty(self):
        assert _takedown_line({}, {"X": [{"event_date": "d"}]}) == ""

    def test_no_entry_for_id_empty(self):
        assert _takedown_line({"instrument_id": "X"}, {}) == ""

    def test_empty_items_empty(self):
        assert _takedown_line({"instrument_id": "X"}, {"X": []}) == ""

    def test_exactly_7_items_no_earlier_suffix(self):
        dd = {"S": [{"event_date": f"2025-01-{i + 1:02d}"} for i in range(7)]}
        line = _takedown_line({"instrument_id": "S"}, dd)
        assert "earlier" not in line
        assert line.startswith("  ↳ S takedowns:")

    def test_8_items_shows_plus_one_earlier(self):
        dd = {"S": [{"event_date": f"2025-01-{i + 1:02d}"} for i in range(8)]}
        line = _takedown_line({"instrument_id": "S"}, dd)
        assert line.endswith("+1 earlier\n")
        # only the first 7 dates appear before the tail count
        assert "2025-01-08" not in line.split("+1 earlier")[0]

    def test_id_prefixed_output(self):
        dd = {"SH-1": [{"event_date": "2025-11-17", "amount_usd": 100}]}
        line = _takedown_line({"instrument_id": "SH-1"}, dd)
        assert line == "  ↳ SH-1 takedowns: 2025-11-17 $100\n"


# ──────────────────────────── _format_status ─────────────────────────
class TestFormatStatus:
    def test_missing_status_is_active(self):
        assert _format_status({}) == "active"

    def test_explicit_active(self):
        assert _format_status({"status": "active"}) == "active"

    def test_closed_with_status_at(self):
        assert _format_status(
            {"status": "exercised", "status_at": "2025-02-10"}
        ) == "exercised(2025-02-10)"

    def test_closed_without_status_at(self):
        assert _format_status({"status": "exercised", "status_at": ""}) == "exercised"


# ─────────────────────────── _short_counterparty ─────────────────────
class TestShortCounterparty:
    def test_missing_is_none(self):
        assert _short_counterparty({}) is None

    def test_empty_string_is_none(self):
        assert _short_counterparty({"counterparty_canonical": ""}) is None

    def test_whitespace_stripped(self):
        assert _short_counterparty({"counterparty_canonical": "  X  "}) == "X"


# ─────────────────────────── _render_archive_row ─────────────────────
class TestRenderArchiveRow:
    def test_all_fields_missing_uses_defaults(self):
        # '- ? ? — closed ' with trailing space (empty status_at) + newline
        assert _render_archive_row({}) == "- ? ? — closed \n"

    def test_full_row(self):
        r = {
            "instrument_id": "W-1",
            "type": "warrant",
            "counterparty_canonical": " Foo ",
            "status": "exercised",
            "status_at": "2025-01-01",
        }
        assert _render_archive_row(r) == "- W-1 warrant Foo exercised 2025-01-01\n"

    def test_counterparty_stripped(self):
        r = {"instrument_id": "X", "counterparty_canonical": "  Acme  "}
        # type missing -> '?', status missing -> 'closed', status_at '' trailing
        assert _render_archive_row(r) == "- X ? Acme closed \n"


# ──────────────────────────── _format_cell ───────────────────────────
class TestFormatCell:
    def test_id_missing_question_mark(self):
        assert _format_cell({}, "id") == "?"

    def test_id_present(self):
        assert _format_cell({"instrument_id": "W-1"}, "id") == "W-1"

    def test_created_none_empty_string(self):
        assert _format_cell({"created_at": None}, "created") == ""

    def test_created_full_timestamp_truncated_to_date(self):
        assert _format_cell({"created_at": "2025-01-01T12:00:00Z"}, "created") == (
            "2025-01-01"
        )

    def test_flags_none_returns_empty_not_dash(self):
        assert _format_cell({}, "flags") == ""

    def test_arbitrary_key_none_returns_dash(self):
        assert _format_cell({}, "strike") == "—"

    def test_arbitrary_key_value_formatted(self):
        assert _format_cell({"terms": {"strike": 1.5}}, "strike") == "1.5"

    def test_status_cell(self):
        assert _format_cell(
            {"status": "exercised", "status_at": "2025-01-01"}, "status"
        ) == "exercised(2025-01-01)"


# ─────────────────────────────── _bucket ─────────────────────────────
class TestBucket:
    def test_active_statuses_and_default(self):
        rows = [
            {"instrument_id": "A", "type": "warrant", "status": "active"},
            {"instrument_id": "B", "type": "warrant", "status": None},  # default active
            {"instrument_id": "C", "type": "warrant", "status": "ACTIVE"},  # lowercased
        ]
        actives, recent, archived = _bucket(rows, TODAY)
        assert [r["instrument_id"] for r in actives] == ["A", "B", "C"]
        assert recent == []
        assert archived == []

    def test_recent_boundary_inclusive(self):
        # status_at exactly == cutoff_recent -> recent (>= is inclusive)
        rows = [
            {
                "instrument_id": "R",
                "type": "warrant",
                "status": "exercised",
                "status_at": CUTOFF_RECENT,
            }
        ]
        actives, recent, archived = _bucket(rows, TODAY)
        assert [r["instrument_id"] for r in recent] == ["R"]
        assert actives == [] and archived == []

    def test_archive_boundary_inclusive(self):
        # status_at exactly == cutoff_archive -> archived
        rows = [
            {
                "instrument_id": "AR",
                "type": "warrant",
                "status": "exercised",
                "status_at": CUTOFF_ARCHIVE,
            }
        ]
        actives, recent, archived = _bucket(rows, TODAY)
        assert [r["instrument_id"] for r in archived] == ["AR"]
        assert recent == []

    def test_just_before_recent_lands_in_archive(self):
        one_before_recent = date.fromordinal(
            TODAY.toordinal() - RECENT_CLOSED_DAYS - 1
        ).isoformat()
        rows = [
            {
                "instrument_id": "AR",
                "type": "warrant",
                "status": "exercised",
                "status_at": one_before_recent,
            }
        ]
        actives, recent, archived = _bucket(rows, TODAY)
        assert recent == []
        assert [r["instrument_id"] for r in archived] == ["AR"]

    def test_older_than_archive_dropped(self):
        rows = [
            {
                "instrument_id": "OLD",
                "type": "warrant",
                "status": "exercised",
                "status_at": "2010-01-01",
            }
        ]
        actives, recent, archived = _bucket(rows, TODAY)
        assert actives == [] and recent == [] and archived == []

    def test_empty_status_at_dropped(self):
        # '' < any real cutoff so a closed row with empty status_at is dropped
        rows = [
            {
                "instrument_id": "EMPTY",
                "type": "warrant",
                "status": "exercised",
                "status_at": "",
            }
        ]
        actives, recent, archived = _bucket(rows, TODAY)
        assert actives == [] and recent == [] and archived == []

    def test_actives_sort_by_type_created_id(self):
        rows = [
            {"instrument_id": "W-2", "type": "warrant", "created_at": "2025-01-02"},
            {"instrument_id": "S-1", "type": "shelf", "created_at": "2025-01-01"},
            {"instrument_id": "W-1", "type": "warrant", "created_at": "2025-01-01"},
            {"instrument_id": "W-0", "type": "warrant", "created_at": "2025-01-01"},
        ]
        actives, _, _ = _bucket(rows, TODAY)
        # shelf < warrant alphabetically; within warrant tie on created_at -> id
        assert [r["instrument_id"] for r in actives] == ["S-1", "W-0", "W-1", "W-2"]

    def test_recent_closed_sorted_newest_first(self):
        rows = [
            {"instrument_id": "R1", "type": "w", "status": "exercised", "status_at": "2025-07-01"},
            {"instrument_id": "R2", "type": "w", "status": "exercised", "status_at": "2025-09-01"},
            {"instrument_id": "R3", "type": "w", "status": "exercised", "status_at": "2025-08-01"},
        ]
        _, recent, _ = _bucket(rows, TODAY)
        assert [r["instrument_id"] for r in recent] == ["R2", "R3", "R1"]

    def test_archived_sorted_newest_first(self):
        rows = [
            {"instrument_id": "A1", "type": "w", "status": "exercised", "status_at": "2024-01-01"},
            {"instrument_id": "A2", "type": "w", "status": "exercised", "status_at": "2024-06-01"},
        ]
        _, _, archived = _bucket(rows, TODAY)
        assert [r["instrument_id"] for r in archived] == ["A2", "A1"]

    def test_none_instrument_id_sorts_first_no_crash(self):
        # FIXED (was bug A#10): the actives third sort key now has an `or ""`
        # fallback, so a None instrument_id coerces to "" and sorts ahead of a
        # string id instead of crashing the in-place sort with a TypeError.
        rows = [
            {"instrument_id": None, "type": "warrant", "created_at": "2025-01-01"},
            {"instrument_id": "W", "type": "warrant", "created_at": "2025-01-01"},
        ]
        actives, _, _ = _bucket(rows, TODAY)
        assert [r.get("instrument_id") for r in actives] == [None, "W"]


# ───────────────────────────── _render_section ───────────────────────
class TestRenderSection:
    def test_empty_rows_returns_empty(self):
        assert _render_section("warrant", [], include_status=False, drawdowns={}) == ""

    def test_warrant_table_drops_null_columns_keeps_populated(self):
        rows = [
            {
                "instrument_id": "W-1",
                "type": "warrant",
                "created_at": "2020-11-30",
                "terms": {"strike": 1.25},
                "outstanding": {"count": 8_000_000},
            },
            {
                "instrument_id": "W-2",
                "type": "warrant",
                "created_at": "2020-11-30",
                "terms": {"strike": 0.001, "is_pre_funded": True},
                "outstanding": {"count": 3_000_000},
            },
        ]
        out = _render_section("warrant", rows, include_status=False, drawdowns={})
        assert "### Warrants" in out
        # counterparty & placement_agent are null on every row -> dropped
        assert "counterparty" not in out
        assert "placement_agent" not in out
        # flags kept because W-2 has 'pre-funded'
        assert "flags" in out
        assert "pre-funded" in out
        assert "8,000,000" in out

    def test_unknown_type_uses_default_columns_and_title_label(self):
        rows = [
            {
                "instrument_id": "X",
                "type": "mystery_type",
                "created_at": "2025-01-01",
                "counterparty_canonical": "Acme",
            }
        ]
        out = _render_section("mystery_type", rows, include_status=False, drawdowns={})
        assert "### Mystery Type" in out
        assert "counterparty" in out
        assert "Acme" in out

    def test_anchor_columns_always_kept_even_when_empty(self):
        # id is present but created_at empty; id/created (idx 0,1) always kept
        rows = [{"instrument_id": "W-1", "type": "warrant", "created_at": ""}]
        out = _render_section("warrant", rows, include_status=False, drawdowns={})
        header = out.splitlines()[1]
        assert "id" in header
        assert "created" in header

    def test_null_column_kept_when_one_row_has_value(self):
        rows = [
            {"instrument_id": "W-1", "type": "warrant", "created_at": "2025-01-01",
             "counterparty_canonical": "NamedInvestor", "terms": {"strike": 1.0}},
            {"instrument_id": "W-2", "type": "warrant", "created_at": "2025-01-01",
             "terms": {"strike": 1.0}},
        ]
        out = _render_section("warrant", rows, include_status=False, drawdowns={})
        assert "counterparty" in out
        assert "NamedInvestor" in out

    def test_include_status_appends_status_column(self):
        rows = [
            {
                "instrument_id": "W-1",
                "type": "warrant",
                "created_at": "2024-08-02",
                "status": "exercised",
                "status_at": "2025-02-10",
                "terms": {"strike": 1.1},
            }
        ]
        out = _render_section("warrant", rows, include_status=True, drawdowns={})
        assert "status" in out.splitlines()[1]
        assert "exercised(2025-02-10)" in out

    def test_column_width_padded_to_header(self):
        # single narrow row: 'count' cell '5' is shorter than header 'count'(5)
        rows = [
            {"instrument_id": "W", "type": "warrant", "created_at": "2025-01-01",
             "terms": {"strike": 1.0}, "outstanding": {"count": 5}}
        ]
        out = _render_section("warrant", rows, include_status=False, drawdowns={})
        # the data line cell for count should pad '5' to the 'count' header width
        data_line = [ln for ln in out.splitlines() if ln.startswith("| W ")][0]
        assert "count" in out.splitlines()[1]
        # 'count' header is 5 chars; cell '5' padded to 5
        assert "| 5     |" in data_line

    def test_separator_row_format(self):
        rows = [{"instrument_id": "W", "type": "warrant", "created_at": "2025-01-01",
                 "terms": {"strike": 1.0}}]
        out = _render_section("warrant", rows, include_status=False, drawdowns={})
        sep = out.splitlines()[2]
        # separator uses '-'*(w+2) per column, pipe-delimited
        assert sep.startswith("|")
        assert set(sep) <= {"|", "-"}

    def test_takedowns_emitted_for_shelf(self):
        rows = [
            {"instrument_id": "SH-1", "type": "shelf", "created_at": "2024-01-01",
             "terms": {"form": "S-3", "capacity_usd": 80_000_000}}
        ]
        dd = {"SH-1": [{"event_date": "2025-11-17", "amount_usd": 8_854_039,
                        "shares": 11_067_547, "price": 0.8,
                        "drawdown_party_canonical": "Maxim"}]}
        out = _render_section("shelf", rows, include_status=False, drawdowns=dd)
        assert "↳ SH-1 takedowns:" in out
        assert "[Maxim]" in out

    @pytest.mark.parametrize("type_", ["atm", "equity_line"])
    def test_takedowns_emitted_for_atm_and_equity_line(self, type_):
        rows = [{"instrument_id": "I-1", "type": type_, "created_at": "2024-01-01"}]
        dd = {"I-1": [{"event_date": "2025-01-01", "amount_usd": 100}]}
        out = _render_section(type_, rows, include_status=False, drawdowns=dd)
        assert "↳ I-1 takedowns:" in out

    def test_takedowns_not_emitted_for_warrant(self):
        rows = [{"instrument_id": "W-1", "type": "warrant", "created_at": "2024-01-01",
                 "terms": {"strike": 1.0}}]
        out = _render_section("warrant", rows, include_status=False,
                              drawdowns={"W-1": [{"event_date": "d"}]})
        assert "↳" not in out

    def test_known_type_label_from_table(self):
        rows = [{"instrument_id": "P-1", "type": "preferred", "created_at": "2025-01-01",
                 "terms": {"series_letter": "A"}}]
        out = _render_section("preferred", rows, include_status=False, drawdowns={})
        assert "### Preferred stock" in out

    def test_raw_json_terms_and_outstanding_form(self):
        # store rows can arrive with raw *_json strings instead of decoded
        # dicts; _to_dict tolerates both, so the rendered cells must match.
        rows = [{
            "instrument_id": "W-1", "type": "warrant", "created_at": "2025-01-01",
            "terms_json": '{"strike": 1.5}', "outstanding_json": '{"count": 42}',
        }]
        out = _render_section("warrant", rows, include_status=False, drawdowns={})
        assert "1.5" in out
        assert "42" in out

    def test_created_at_iso_timestamp_truncated_in_table(self):
        # full ISO timestamp in created_at is truncated to the date in the cell
        rows = [{"instrument_id": "W-1", "type": "warrant",
                 "created_at": "2025-01-01T12:34:56Z", "terms": {"strike": 1.0}}]
        out = _render_section("warrant", rows, include_status=False, drawdowns={})
        data_line = [ln for ln in out.splitlines() if ln.startswith("| W-1 ")][0]
        assert "2025-01-01" in data_line
        assert "T12:34:56Z" not in data_line


# ────────────────────────── _collapse_warrants ───────────────────────
class TestCollapseWarrants:
    def test_non_warrant_passes_through(self):
        rows = [{"instrument_id": "SH-1", "type": "shelf", "created_at": "2025-01-03"}]
        out = _collapse_warrants(rows)
        assert len(out) == 1
        assert out[0]["instrument_id"] == "SH-1"

    def test_single_warrant_not_collapsed(self):
        rows = [{"instrument_id": "W-9", "type": "warrant", "status": "active",
                 "terms": {"strike": 1.0}, "outstanding": {"count": 5}}]
        out = _collapse_warrants(rows)
        assert out[0]["instrument_id"] == "W-9"

    def test_two_same_group_merged(self):
        rows = [
            {"instrument_id": "W-2", "type": "warrant", "status": "active",
             "created_at": "2025-01-02", "terms": {"strike": 1.25},
             "outstanding": {"count": 100}},
            {"instrument_id": "W-1", "type": "warrant", "status": "active",
             "created_at": "2025-01-01", "terms": {"strike": 1.25},
             "outstanding": {"count": 200}},
        ]
        out = _collapse_warrants(rows)
        assert len(out) == 1
        merged = out[0]
        # members sorted by id: first..last
        assert merged["instrument_id"] == "W-1..W-2"
        assert merged["outstanding"]["count"] == pytest.approx(300.0)
        assert merged["outstanding"]["tranche_count"] == 2
        assert merged["terms"]["strike"] == 1.25

    def test_strike_rounding_groups_within_4dp(self):
        rows = [
            {"instrument_id": "A", "type": "warrant", "status": "active",
             "terms": {"strike": 1.25001}, "outstanding": {"count": 1}},
            {"instrument_id": "B", "type": "warrant", "status": "active",
             "terms": {"strike": 1.25004}, "outstanding": {"count": 1}},
        ]
        out = _collapse_warrants(rows)
        assert len(out) == 1
        assert out[0]["instrument_id"] == "A..B"

    def test_strike_rounding_separates_at_4dp(self):
        rows = [
            {"instrument_id": "A", "type": "warrant", "status": "active",
             "terms": {"strike": 1.2500}, "outstanding": {"count": 1}},
            {"instrument_id": "B", "type": "warrant", "status": "active",
             "terms": {"strike": 1.2501}, "outstanding": {"count": 1}},
        ]
        out = _collapse_warrants(rows)
        assert {r["instrument_id"] for r in out} == {"A", "B"}

    def test_warrant_strike_fallback(self):
        rows = [
            {"instrument_id": "A", "type": "warrant", "status": "active",
             "terms": {"warrant_strike": 3.0}, "outstanding": {"count": 1}},
            {"instrument_id": "B", "type": "warrant", "status": "active",
             "terms": {"warrant_strike": 3.0}, "outstanding": {"count": 1}},
        ]
        out = _collapse_warrants(rows)
        assert len(out) == 1
        assert out[0]["instrument_id"] == "A..B"
        # collapse stamps a 'strike' key from the resolved value
        assert out[0]["terms"]["strike"] == 3.0

    def test_none_strike_groups_together(self):
        rows = [
            {"instrument_id": "A", "type": "warrant", "status": "active",
             "terms": {}, "outstanding": {"count": 1}},
            {"instrument_id": "B", "type": "warrant", "status": "active",
             "terms": {}, "outstanding": {"count": 2}},
        ]
        out = _collapse_warrants(rows)
        assert len(out) == 1
        assert out[0]["instrument_id"] == "A..B"

    def test_none_counterparty_uses_dash_key(self):
        # anonymous warrants (no counterparty) collapse together
        rows = [
            {"instrument_id": "A", "type": "warrant", "status": "active",
             "terms": {"strike": 1.0}, "outstanding": {"count": 1}},
            {"instrument_id": "B", "type": "warrant", "status": "active",
             "terms": {"strike": 1.0}, "outstanding": {"count": 1}},
        ]
        out = _collapse_warrants(rows)
        assert len(out) == 1
        assert out[0]["counterparty_canonical"] == "—"

    def test_differing_status_splits_groups(self):
        rows = [
            {"instrument_id": "A", "type": "warrant", "status": "active",
             "terms": {"strike": 1.0}, "outstanding": {"count": 1}},
            {"instrument_id": "B", "type": "warrant", "status": "exercised",
             "terms": {"strike": 1.0}, "outstanding": {"count": 1}},
        ]
        out = _collapse_warrants(rows)
        # both singletons in their own status group -> not merged
        assert {r["instrument_id"] for r in out} == {"A", "B"}

    def test_missing_count_treated_as_zero(self):
        rows = [
            {"instrument_id": "A", "type": "warrant", "status": "active",
             "terms": {"strike": 1.0}, "outstanding": {}},
            {"instrument_id": "B", "type": "warrant", "status": "active",
             "terms": {"strike": 1.0}, "outstanding": {"count": 50}},
        ]
        out = _collapse_warrants(rows)
        assert out[0]["outstanding"]["count"] == pytest.approx(50.0)

    def test_output_resorted_with_mixed_types(self):
        rows = [
            {"instrument_id": "W-1", "type": "warrant", "status": "active",
             "created_at": "2025-02-01", "terms": {"strike": 9.0},
             "outstanding": {"count": 1}},
            {"instrument_id": "SH-1", "type": "shelf", "created_at": "2025-01-01"},
        ]
        out = _collapse_warrants(rows)
        # sorted by (type, created_at, id): shelf < warrant
        assert [r["instrument_id"] for r in out] == ["SH-1", "W-1"]

    def test_resort_with_none_created_at_no_crash(self):
        rows = [
            {"instrument_id": "SH-1", "type": "shelf", "created_at": None},
            {"instrument_id": "W-1", "type": "warrant", "created_at": "2025-01-01",
             "terms": {"strike": 1.0}, "outstanding": {"count": 1}},
        ]
        out = _collapse_warrants(rows)
        assert [r["instrument_id"] for r in out] == ["SH-1", "W-1"]


# ────────────────────────── render_ledger_view ───────────────────────
class TestRenderLedgerView:
    def test_none_rows_no_active_instruments(self):
        assert render_ledger_view(None, today=TODAY) == (
            "## Open ledger\n(no active instruments)\n"
        )

    def test_empty_rows_no_active_instruments(self):
        assert render_ledger_view([], today=TODAY) == (
            "## Open ledger\n(no active instruments)\n"
        )

    def test_drawdowns_default_none_no_crash(self):
        rows = [{"instrument_id": "SH-1", "type": "shelf", "created_at": "2024-01-01",
                 "terms": {"form": "S-3"}}]
        out = render_ledger_view(rows, today=TODAY, drawdowns_by_instrument=None)
        assert "### Shelves" in out
        assert "↳" not in out  # no takedown lines

    def test_recent_boundary_row_in_recently_closed(self):
        # a row dated exactly RECENT_CLOSED_DAYS ago lands in recent_closed
        rows = [
            {"instrument_id": "R-1", "type": "warrant", "status": "exercised",
             "status_at": CUTOFF_RECENT, "created_at": "2024-01-01",
             "terms": {"strike": 1.1}},
        ]
        out = render_ledger_view(rows, today=TODAY, max_chars=10**9)
        assert "## Recently closed (last 365 days)" in out
        assert "R-1" in out

    def test_body_exactly_max_chars_unchanged(self):
        full = render_ledger_view([], today=TODAY, max_chars=10**9)
        # condition is <=, so body == max_chars returns unchanged
        boundary = render_ledger_view([], today=TODAY, max_chars=len(full))
        assert boundary == full

    def test_stage1_drops_archived_keeps_actives_and_recent(self):
        actives = [{"instrument_id": "W-1", "type": "warrant", "created_at": "2024-01-01",
                    "terms": {"strike": 1.0}, "outstanding": {"count": 100}}]
        archived = [
            {"instrument_id": f"OLD-{i}", "type": "warrant", "status": "exercised",
             "status_at": "2024-01-01"}
            for i in range(20)
        ]
        rows = actives + archived
        full = render_ledger_view(rows, today=TODAY, max_chars=10**9)
        assert "Older closed" in full
        out = render_ledger_view(rows, today=TODAY, max_chars=len(full) - 50)
        assert "Older closed" not in out  # archived dropped
        assert "W-1" in out  # actives intact

    def test_stage3_collapses_active_warrants_to_range(self):
        # cap between collapsed-active size and full size -> '..' range appears
        actives = [
            {"instrument_id": f"W-{i:02d}", "type": "warrant", "created_at": "2024-01-01",
             "status": "active", "terms": {"strike": 1.0}, "outstanding": {"count": 100}}
            for i in range(10)
        ]
        full = render_ledger_view(actives, today=TODAY, max_chars=10**9)
        collapsed = _render_buckets(_collapse_warrants(actives), [], [], {})
        cap = (len(full) + len(collapsed)) // 2
        out = render_ledger_view(actives, today=TODAY, max_chars=cap)
        assert "W-00..W-09" in out
        assert "×10 tranches" in out

    def test_stage2_collapses_closed_warrants_to_range(self):
        actives = [{"instrument_id": "A-1", "type": "shelf", "created_at": "2024-01-01",
                    "status": "active", "terms": {"form": "S-3"}}]
        recent = [
            {"instrument_id": f"W-{i:02d}", "type": "warrant", "status": "exercised",
             "status_at": "2025-07-01", "created_at": "2024-01-01",
             "terms": {"strike": 1.0}, "outstanding": {"count": 100}}
            for i in range(10)
        ]
        rows = actives + recent
        full = render_ledger_view(rows, today=TODAY, max_chars=10**9)
        body2 = _render_buckets(actives, _collapse_warrants(recent), [], {})
        cap = (len(full) + len(body2)) // 2
        out = render_ledger_view(rows, today=TODAY, max_chars=cap)
        assert "W-00..W-09" in out
        assert "A-1" in out  # active shelf preserved

    def test_stage4_truncates_oldest_first_and_terminates(self, caplog):
        # tiny cap smaller than any single active row -> while-pop empties the
        # list; the loop must terminate and not IndexError. Result body becomes
        # the '(no active instruments)' string.
        actives = [
            {"instrument_id": f"W-{i}", "type": "warrant",
             "created_at": f"2024-01-{i + 1:02d}", "status": "active",
             "terms": {"strike": float(i)}, "outstanding": {"count": 100}}
            for i in range(5)
        ]
        out = render_ledger_view(actives, today=TODAY, max_chars=50)
        assert out == "## Open ledger\n(no active instruments)\n"

    def test_stage4_keeps_newest_drops_oldest(self):
        # find a cap that keeps exactly one active warrant (the newest by
        # created_at) after stage-4 truncation
        # collapse_warrants would merge same strike/cp/status warrants; to
        # force the hard-truncate path we need distinct strikes so collapse
        # cannot shrink them.
        actives = [
            {"instrument_id": f"W-{i}", "type": "warrant",
             "created_at": f"2024-01-{i + 1:02d}", "status": "active",
             "terms": {"strike": float(i + 1)}, "outstanding": {"count": 100}}
            for i in range(3)
        ]
        single_newest = _render_buckets([actives[2]], [], [], {})
        out = render_ledger_view(actives, today=TODAY, max_chars=len(single_newest))
        # the newest (W-2, 2024-01-03) is kept; older ones dropped
        assert "W-2" in out
        assert "W-0" not in out

    def test_stage4_none_created_at_no_typeerror(self):
        # created_at=None in stage-4 sort uses '0000-00-00' fallback, no crash.
        # The sort is reverse=True on created_at, so the None row ('0000-00-00')
        # sorts LAST (oldest) and is the first to be popped. Distinct strikes
        # prevent _collapse_warrants from merging, forcing the hard-truncate path.
        actives = [
            {"instrument_id": "W-1", "type": "warrant", "created_at": None,
             "status": "active", "terms": {"strike": 1.0}},
            {"instrument_id": "W-2", "type": "warrant", "created_at": "2024-01-01",
             "status": "active", "terms": {"strike": 2.0}},
        ]
        # Cap that fits exactly the single newest (real-dated) row: the None-dated
        # row is dropped, proving the fallback orders it as oldest without raising.
        single_w2 = _render_buckets([actives[1]], [], [], {})
        out = render_ledger_view(actives, today=TODAY, max_chars=len(single_w2))
        assert isinstance(out, str)
        assert "W-2" in out      # newest (real date) kept
        assert "W-1" not in out  # None-dated row treated as oldest, dropped first

    def test_default_max_chars_constant(self):
        assert DEFAULT_MAX_CHARS == 60_000

    def test_golden_full_warrant_table(self):
        # Exact end-to-end render of the docstring's canonical two-warrant
        # example: column selection (counterparty/placement_agent dropped as
        # all-null, flags kept for pre-funded), content-sized widths, the
        # '-'*(w+2) separator, and the pre-funded flag all in one string.
        rows = [
            {"instrument_id": "W-005", "type": "warrant", "created_at": "2020-11-30",
             "status": "active", "terms": {"strike": 1.25},
             "outstanding": {"count": 8_000_000}},
            {"instrument_id": "W-006", "type": "warrant", "created_at": "2020-11-30",
             "status": "active", "terms": {"strike": 0.001, "is_pre_funded": True},
             "outstanding": {"count": 3_000_000}},
        ]
        out = render_ledger_view(rows, today=TODAY, max_chars=10**9)
        assert out == (
            "## Open ledger\n\n"
            "### Warrants\n"
            "| id    | created    | strike | count     | flags      |\n"
            "|-------|------------|--------|-----------|------------|\n"
            "| W-005 | 2020-11-30 | 1.25   | 8,000,000 |            |\n"
            "| W-006 | 2020-11-30 | 0.001  | 3,000,000 | pre-funded |\n"
            "\n"
        )

    def test_closed_empty_status_at_dropped_end_to_end(self):
        # A closed row with status_at='' is dropped from ALL buckets; only the
        # active instrument survives. (status_at '' < every real cutoff.)
        rows = [
            {"instrument_id": "W-1", "type": "warrant", "created_at": "2024-01-01",
             "status": "active", "terms": {"strike": 1.0}},
            {"instrument_id": "GONE", "type": "warrant", "status": "exercised",
             "status_at": "", "terms": {"strike": 2.0}},
        ]
        out = render_ledger_view(rows, today=TODAY, max_chars=10**9)
        assert "W-1" in out
        assert "GONE" not in out
        assert "Recently closed" not in out
        assert "Older closed" not in out

    def test_recent_and_archive_buckets_render_both_headers(self):
        # one active, one recent-closed, one archived -> all three sections.
        one_before_recent = date.fromordinal(
            TODAY.toordinal() - RECENT_CLOSED_DAYS - 1
        ).isoformat()
        rows = [
            {"instrument_id": "W-A", "type": "warrant", "created_at": "2025-01-01",
             "status": "active", "terms": {"strike": 1.0}},
            {"instrument_id": "W-R", "type": "warrant", "status": "exercised",
             "status_at": CUTOFF_RECENT, "created_at": "2024-01-01",
             "terms": {"strike": 2.0}},
            {"instrument_id": "W-OLD", "type": "warrant", "status": "exercised",
             "status_at": one_before_recent, "created_at": "2022-01-01",
             "counterparty_canonical": "Acme"},
        ]
        out = render_ledger_view(rows, today=TODAY, max_chars=10**9)
        assert "## Open ledger" in out
        assert "## Recently closed (last 365 days)" in out
        assert "## Older closed (summary)" in out
        # archived row uses the one-line summary form, not a table
        assert "- W-OLD warrant Acme exercised" in out


# ─────────────────────────── module sanity ───────────────────────────
class TestModuleSanity:
    def test_constants(self):
        assert RECENT_CLOSED_DAYS == 365
        assert ARCHIVE_CLOSED_DAYS == 365 * 3

    def test_import_is_side_effect_free(self):
        # re-import must not raise (no get_conn call at import time)
        import importlib
        importlib.reload(view)
