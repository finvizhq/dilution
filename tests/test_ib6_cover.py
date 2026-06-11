"""Unit tests for dilution/ib6_cover.py.

Covers the pure legend-extraction tier (_num, _in, _parse_legend_date,
_extract_float_fields, parse_ib6_legend, Ib6Legend), the DB-backed regime
walk (_candidate_docs, _accession_text, ib6_cover_status + the lru_cache
wrapper), and the io-mockable tiers (entity_public_float_latest, ib6_regime).

All DB access flows through the autouse ``temp_db`` fixture from conftest.py
(reroutes db.get_conn to a throwaway SQLite file). No network/SEC/LLM calls:
``requests`` is injected as a fake module since ib6_cover imports it lazily.
"""
from __future__ import annotations

import sys
import types
from datetime import date, timedelta

import dataclasses
import pytest

import dilution.ib6_cover as ib6
from dilution.ib6_cover import (
    Ib6Legend,
    _accession_text,
    _candidate_docs,
    _extract_float_fields,
    _in,
    _num,
    _parse_legend_date,
    entity_public_float_latest,
    ib6_cover_status,
    ib6_cover_status_cached,
    ib6_regime,
    parse_ib6_legend,
)


# ── shared helpers ────────────────────────────────────────────────────

# The canonical legend from the module docstring; the golden case used as a
# baseline that individual tests mutate one clause at a time.
CANONICAL_LEGEND = (
    "As of August 29, 2025, the aggregate market value of our "
    "outstanding common stock held by non-affiliates, or public float, "
    "was approximately $61.7 million, based on 7,113,902 shares of "
    "outstanding common stock held by non-affiliates at a price of "
    "$8.67 per share, which was the closing price of our common stock "
    "on Nasdaq on August 12, 2025. We have not offered any securities "
    "pursuant to General Instruction I.B.6 of Form S-3 during the "
    "prior 12-calendar-month period."
)


def _stage_raw(temp_db, accession, content, doc_name="d.htm"):
    """Insert a dilution_raw row (content_md is NOT NULL in the schema)."""
    temp_db.execute(
        "INSERT INTO dilution_raw(accession_number, doc_name, content_md, "
        "downloaded_at) VALUES(?,?,?,?)",
        (accession, doc_name, content, "2026-01-01"),
    )


def _clear_lru_caches():
    # Some tests monkeypatch these names with plain lambdas, which have no
    # cache_clear; guard so teardown never errors.
    for name in ("_cached_status", "_cached_public_float"):
        fn = getattr(ib6, name, None)
        clearer = getattr(fn, "cache_clear", None)
        if clearer is not None:
            clearer()


@pytest.fixture(autouse=True)
def _clear_caches():
    """The lru_cache wrappers are keyed on (cik, today_iso) and would leak
    state staged by an earlier test. Clear before AND after each test."""
    _clear_lru_caches()
    yield
    _clear_lru_caches()


# ════════════════════════════════════════════════════════════════════════
# PURE TIER
# ════════════════════════════════════════════════════════════════════════

class TestNum:
    def test_million_with_commas(self):
        assert _num("1,234.5", "million") == 1234500000.0

    def test_unit_none_no_multiplier(self):
        assert _num("100") == 100.0

    def test_mixed_case_million_scales(self):
        assert _num("2", "Million") == 2_000_000.0

    def test_billion_scales(self):
        assert _num("3", "billion") == 3e9

    def test_empty_unit_multiplier_one(self):
        assert _num("3", "") == 3.0

    def test_unknown_unit_no_scaling(self):
        # "thousand" is not in _MULT -> multiplier 1.0
        assert _num("4", "thousand") == 4.0

    def test_non_numeric_returns_none(self):
        # ValueError swallowed
        assert _num("abc") is None

    def test_none_input_returns_none(self):
        # AttributeError on .replace swallowed
        assert _num(None) is None

    def test_zero_is_not_none(self):
        # "0" parses to 0.0, distinct from None
        result = _num("0")
        assert result == 0.0
        assert result is not None


class TestIn:
    def test_none_is_false(self):
        assert _in(None, (1, 10)) is False

    def test_lower_bound_inclusive(self):
        assert _in(1, (1, 10)) is True

    def test_upper_bound_inclusive(self):
        assert _in(10, (1, 10)) is True

    @pytest.mark.parametrize("v,expected", [
        (0.99, False),   # just below lo
        (10.01, False),  # just above hi
        (-5, False),     # negative vs positive bounds
        (5, True),       # interior
    ])
    def test_boundary_sweep(self, v, expected):
        assert _in(v, (1, 10)) is expected

    def test_degenerate_bound_lo_equals_hi(self):
        # lo == hi: only the exact point is inside (lo <= v <= hi collapses).
        assert _in(5, (5, 5)) is True
        assert _in(5.0001, (5, 5)) is False
        assert _in(4.9999, (5, 5)) is False

    def test_zero_value_with_zero_lower_bound(self):
        # 0.0 is a real value (not None) -> the None guard does not fire,
        # and 0 is within [0, 10].
        assert _in(0.0, (0, 10)) is True


class TestParseLegendDate:
    def test_canonical_full_month(self):
        assert _parse_legend_date("August 29, 2025") == "2025-08-29"

    def test_single_digit_day(self):
        assert _parse_legend_date("July 4, 2024") == "2024-07-04"

    @pytest.mark.parametrize("bad", [
        "Aug 29, 2025",     # abbreviated month not accepted by %B
        "not a date",
        "",
        "29 August 2025",   # wrong order
    ])
    def test_unparseable_returns_none(self, bad):
        assert _parse_legend_date(bad) is None


class TestExtractFloatFields:
    def test_value_plus_shares_derives_price(self):
        # $1,000,000 / 500,000 -> price 2.0
        out = _extract_float_fields(
            "aggregate market value held by non-affiliates was $1,000,000, "
            "based on 500,000 shares held by non-affiliates."
        )
        assert out["float_value_usd"] == 1_000_000.0
        assert out["non_affiliate_shares"] == 500_000.0
        assert out["price_usd"] == 2.0

    def test_value_plus_price_derives_shares(self):
        # value $1,000,000 + styled price $2.00 -> shares = round(1e6/2) = 500000
        out = _extract_float_fields(
            "aggregate market value of common stock held by non-affiliates "
            "was $1,000,000 at a price of $2.00 per share."
        )
        assert out["float_value_usd"] == 1_000_000.0
        assert out["price_usd"] == 2.0
        assert out["non_affiliate_shares"] == 500_000

    def test_shares_plus_price_derives_value(self):
        # 500,000 * 2.00 -> value 1,000,000
        out = _extract_float_fields(
            "based on 500,000 shares held by non-affiliates at a price of "
            "$2.00 per share"
        )
        assert out["non_affiliate_shares"] == 500_000.0
        assert out["price_usd"] == 2.0
        assert out["float_value_usd"] == 1_000_000.0

    def test_threshold_sentence_nulls_value_but_loose_dollar_is_price(self):
        # The threshold guard rejects "below $75 million" as the float value,
        # but the loose _DOLLAR_RE still surfaces 75.0 and (no shares/value)
        # stores it as price_usd. Documented quirk.
        out = _extract_float_fields(
            "the public float remains below $75 million as required."
        )
        assert "float_value_usd" not in out
        assert out == {"price_usd": 75.0}

    def test_astc_last_held_by_wins(self):
        # shares should be the computation-clause number (16,608,155), not the
        # outstanding-shares number (18,557,754) — the LAST held-by wins.
        out = _extract_float_fields(
            "based on 18,557,754 shares of outstanding common stock, of which "
            "approximately 16,608,155 shares are held by non-affiliates, and a "
            "price of $2.68 per share"
        )
        assert out["non_affiliate_shares"] == 16_608_155.0
        assert out["price_usd"] == 2.68
        assert out["float_value_usd"] == pytest.approx(44_509_855.4)

    def test_no_held_by_mention(self):
        # No "held by non-affiliates" -> no shares from that path. value+price
        # then derive shares.
        out = _extract_float_fields(
            "the aggregate market value was $5,000,000 at a price of $10.00 "
            "per share"
        )
        assert out["float_value_usd"] == 5_000_000.0
        assert out["price_usd"] == 10.0
        # value+price -> derived shares = 500000
        assert out["non_affiliate_shares"] == 500_000

    def test_price_within_2pct_accepted(self):
        # value 1e6, shares 500000 implies 2.00; styled $2.01 -> 500000*2.01 =
        # 1,005,000, off by 0.5% < 2% -> accepted as-is.
        out = _extract_float_fields(
            "aggregate market value held by non-affiliates was $1,000,000, "
            "based on 500,000 shares held by non-affiliates at a price of "
            "$2.01 per share"
        )
        assert out["price_usd"] == 2.01

    def test_price_beyond_2pct_rejected_then_derived(self):
        # styled $9.99 is wildly inconsistent (500000*9.99 vs 1e6) -> rejected,
        # price derived from value/shares = 2.0 instead.
        out = _extract_float_fields(
            "aggregate market value held by non-affiliates was $1,000,000, "
            "based on 500,000 shares held by non-affiliates at a price of "
            "$9.99 per share"
        )
        assert out["price_usd"] == 2.0

    def test_value_out_of_bounds_nulled(self):
        # "(was) $5" -> 5.0 not in _VALUE_BOUNDS (1e5,5e9) -> not stored as
        # value; the loose $ then becomes price_usd=5.0.
        out = _extract_float_fields("the value was $5 held by non-affiliates")
        assert "float_value_usd" not in out
        assert out == {"price_usd": 5.0}

    def test_shares_below_lower_bound_not_used(self):
        # plain "5000" is not a candidate share count (below _SHARES_BOUNDS
        # 1e4 AND not matched by _NUMBER_RE without a comma group), so with no
        # other usable number the result is empty.
        out = _extract_float_fields("based on 5000 shares held by non-affiliates.")
        assert out == {}

    def test_derived_price_out_of_bounds_not_stored(self):
        # value 4e9 + shares 1e5 -> derived price 40000 > _PRICE_BOUNDS hi
        # (10000) -> price left absent (no crash).
        out = _extract_float_fields(
            "aggregate market value held by non-affiliates was $4,000,000,000, "
            "based on 100,000 shares held by non-affiliates."
        )
        assert out["float_value_usd"] == 4_000_000_000.0
        assert out["non_affiliate_shares"] == 100_000.0
        assert "price_usd" not in out

    def test_present_tense_is_form_parses(self):
        # _FLOAT_VALUE_RE accepts "(was|is) $X"; the present-tense "is"
        # variant must parse identically to "was".
        out = _extract_float_fields(
            "aggregate market value held by non-affiliates is $2,000,000, "
            "based on 1,000,000 shares held by non-affiliates."
        )
        assert out["float_value_usd"] == 2_000_000.0
        assert out["non_affiliate_shares"] == 1_000_000.0
        # derived price = 2e6 / 1e6 = 2.0
        assert out["price_usd"] == 2.0

    def test_empty_sentence(self):
        assert _extract_float_fields("") == {}


class TestParseIb6Legend:
    def test_empty_string_returns_none(self):
        assert parse_ib6_legend("") is None

    def test_none_input_returns_none(self):
        # `if not text` short-circuits on None (and on '') before any regex.
        assert parse_ib6_legend(None) is None

    def test_no_anchor_returns_none(self):
        assert parse_ib6_legend(
            "This is a normal prospectus with no instruction reference."
        ) is None

    def test_anchor_trailing_dot_and_missing_general_still_parses(self):
        # _ANCHOR_RE makes "General" optional and tolerates a trailing dot
        # after the instruction reference ("Instruction I.B.6.").
        leg = parse_ib6_legend(
            "baby shelf. Pursuant to Instruction I.B.6. the aggregate market "
            "value of our common stock held by non-affiliates was $10,000,000 "
            "based on 5,000,000 shares held by non-affiliates."
        )
        assert leg is not None
        assert leg.present is True
        assert leg.float_value_usd == 10_000_000.0
        assert leg.non_affiliate_shares == 5_000_000.0

    def test_anchor_without_context_returns_none(self):
        # I.B.5 anchor present but no baby-shelf context within 4000 chars.
        assert parse_ib6_legend(
            "Pursuant to General Instruction I.B.5 we may offer securities. "
            + "x" * 100
        ) is None

    def test_canonical_full_legend(self):
        leg = parse_ib6_legend(CANONICAL_LEGEND)
        assert leg == Ib6Legend(
            present=True,
            float_value_usd=61_700_000.0,
            non_affiliate_shares=7_113_902.0,
            price_usd=8.67,
            price_date="2025-08-12",
            as_of_date="2025-08-29",
            sold_12mo_usd=0.0,
        )

    def test_have_not_offered_any_yields_explicit_zero(self):
        leg = parse_ib6_legend(CANONICAL_LEGEND)
        # sold_12mo_usd == 0.0 (explicit not-offered) distinct from None
        assert leg.sold_12mo_usd == 0.0
        assert leg.sold_12mo_usd is not None

    def test_sold_amount_forward_order(self):
        leg = parse_ib6_legend(
            "held by non-affiliates one-third. We have offered and sold "
            "$5 million of securities pursuant to General Instruction I.B.6 "
            "of Form S-3."
        )
        assert leg is not None
        assert leg.sold_12mo_usd == 5_000_000.0

    def test_sold_amount_reversed_order(self):
        # matched by the second _SOLD_AMOUNT_RES (pursuant ... then sold)
        leg = parse_ib6_legend(
            "baby shelf. Pursuant to General Instruction I.B.6 of Form S-3, "
            "during the prior 12 months we have sold $3,000,000 of common stock."
        )
        assert leg is not None
        assert leg.sold_12mo_usd == 3_000_000.0

    def test_sold_amount_over_1e10_rejected(self):
        leg = parse_ib6_legend(
            "baby shelf. We have offered and sold $15,000,000,000 of "
            "securities pursuant to General Instruction I.B.6 of Form S-3."
        )
        assert leg is not None
        assert leg.sold_12mo_usd is None

    def test_ib5_fpi_anchor_with_context_parses(self):
        # I.B.5 (F-3 / FPI) anchor with baby-shelf context still parses the
        # legend (anchor regex matches [56]).
        leg = parse_ib6_legend(
            "baby shelf context. Pursuant to General Instruction I.B.5 of "
            "Form F-3, the aggregate market value of our common stock held by "
            "non-affiliates was $10,000,000 based on 5,000,000 shares held by "
            "non-affiliates."
        )
        assert leg is not None
        assert leg.present is True
        assert leg.float_value_usd == 10_000_000.0
        assert leg.non_affiliate_shares == 5_000_000.0

    def test_as_of_captured_from_prefix_window(self):
        # "As of <date>," precedes the float sentence; captured from the
        # 80-char prefix lookback.
        leg = parse_ib6_legend(CANONICAL_LEGEND)
        assert leg.as_of_date == "2025-08-29"

    def test_present_true_even_when_value_fields_absent(self):
        # A legend with an anchor + context but a deviating float sentence is
        # still present=True (a valid stamp) with None value fields.
        leg = parse_ib6_legend(
            "baby shelf. Pursuant to General Instruction I.B.6 of Form S-3, "
            "no calculation is reproduced here."
        )
        assert leg is not None
        assert leg.present is True
        assert leg.float_value_usd is None
        assert leg.non_affiliate_shares is None

    def test_picks_candidate_with_most_fields_not_nearest(self):
        # A cap/one-third sentence (no numbers) sits between the anchor and the
        # real calculation sentence; the calculation must still be chosen.
        text = (
            "We may not sell securities exceeding one-third of the aggregate "
            "market value of our common stock held by non-affiliates in any "
            "12-month period under General Instruction I.B.6. As of August 29, "
            "2025, the aggregate market value of our common stock held by "
            "non-affiliates was approximately $61.7 million, based on 7,113,902 "
            "shares held by non-affiliates at a price of $8.67 per share."
        )
        leg = parse_ib6_legend(text)
        assert leg is not None
        assert leg.float_value_usd == 61_700_000.0
        assert leg.non_affiliate_shares == 7_113_902.0
        assert leg.price_usd == 8.67


class TestIb6Legend:
    def test_frozen_assignment_raises(self):
        leg = Ib6Legend(present=True)
        with pytest.raises(dataclasses.FrozenInstanceError):
            leg.present = False

    def test_present_only_all_none_valid(self):
        leg = Ib6Legend(present=True)
        assert leg.present is True
        assert leg.float_value_usd is None
        assert leg.non_affiliate_shares is None
        assert leg.price_usd is None
        assert leg.price_date is None
        assert leg.as_of_date is None
        assert leg.sold_12mo_usd is None

    def test_explicit_zero_distinct_from_none(self):
        zero = Ib6Legend(present=True, sold_12mo_usd=0.0)
        none = Ib6Legend(present=True, sold_12mo_usd=None)
        assert zero.sold_12mo_usd == 0.0
        assert zero != none

    def test_value_equality_and_hashable(self):
        a = Ib6Legend(present=True, float_value_usd=1.0)
        b = Ib6Legend(present=True, float_value_usd=1.0)
        assert a == b
        assert hash(a) == hash(b)


# ════════════════════════════════════════════════════════════════════════
# DB TIER
# ════════════════════════════════════════════════════════════════════════

class TestCandidateDocs:
    def test_only_docs_with_raw_text_appear(self, temp_db):
        # A 424B5 with NO dilution_raw row is dropped by the INNER JOIN.
        temp_db.add_company(100, "AAA")
        temp_db.add_filing("with_raw", 100, form="424B5",
                           filing_date="2026-01-01")
        _stage_raw(temp_db, "with_raw", "txt")
        temp_db.add_filing("no_raw", 100, form="424B5",
                           filing_date="2026-02-01")
        accs = [d["accession_number"] for d in _candidate_docs(100)]
        assert accs == ["with_raw"]

    def test_newest_first_and_424b3_excluded_and_groupby_collapse(self, temp_db):
        temp_db.add_company(100, "AAA")
        temp_db.add_filing("A", 100, form="S-3", filing_date="2026-01-01")
        _stage_raw(temp_db, "A", "a", doc_name="d1")
        temp_db.add_filing("B", 100, form="424B5", filing_date="2026-03-01")
        _stage_raw(temp_db, "B", "b1", doc_name="d1")
        _stage_raw(temp_db, "B", "b2", doc_name="d2")  # multi-doc accession
        # 424B3 is a resale form not in PRIMARY_PROSPECTUS_FORMS
        temp_db.add_filing("C", 100, form="424B3", filing_date="2026-04-01")
        _stage_raw(temp_db, "C", "c", doc_name="d1")
        docs = _candidate_docs(100)
        # newest filing_date first; B (multi-doc) collapsed to one row; 424B3 gone
        assert [d["accession_number"] for d in docs] == ["B", "A"]

    def test_424b_off_s1_only_excluded(self, temp_db):
        temp_db.add_company(200, "BBB")
        temp_db.add_filing("b1", 200, form="424B5", filing_date="2026-02-01",
                           file_number="333-S1")
        _stage_raw(temp_db, "b1", "txt")
        temp_db.add_filing("s1f", 200, form="S-1", filing_date="2026-01-01",
                           file_number="333-S1")
        assert _candidate_docs(200) == []

    def test_424b_off_both_s1_and_s3_kept_s3_wins(self, temp_db):
        temp_db.add_company(300, "CCC")
        temp_db.add_filing("b1", 300, form="424B5", filing_date="2026-02-01",
                           file_number="333-BOTH")
        _stage_raw(temp_db, "b1", "txt")
        temp_db.add_filing("s1f", 300, form="S-1", filing_date="2026-01-01",
                           file_number="333-BOTH")
        temp_db.add_filing("s3f", 300, form="S-3", filing_date="2026-01-02",
                           file_number="333-BOTH")
        docs = _candidate_docs(300)
        assert len(docs) == 1
        assert docs[0]["accession_number"] == "b1"
        assert docs[0]["registration_known"] is True

    def test_424b_unknown_file_number_kept_reg_unknown(self, temp_db):
        temp_db.add_company(400, "DDD")
        temp_db.add_filing("b1", 400, form="424B5", filing_date="2026-02-01",
                           file_number=None)
        _stage_raw(temp_db, "b1", "txt")
        docs = _candidate_docs(400)
        assert len(docs) == 1
        assert docs[0]["registration_known"] is False

    def test_s3_form_always_registration_known(self, temp_db):
        temp_db.add_company(500, "EEE")
        temp_db.add_filing("s3", 500, form="S-3", filing_date="2026-02-01",
                           file_number=None)
        _stage_raw(temp_db, "s3", "txt")
        docs = _candidate_docs(500)
        assert docs[0]["registration_known"] is True

    @pytest.mark.parametrize("form", ["S-3", "S-3/A", "F-3", "F-3/A"])
    def test_registration_form_always_known_regardless_of_file_number(
            self, temp_db, form):
        # Every non-424B primary-prospectus form is registration_known True
        # even with a NULL file_number (the `not startswith 424B` leg).
        temp_db.add_company(700, "GGG")
        temp_db.add_filing("reg", 700, form=form, filing_date="2026-02-01",
                           file_number=None)
        _stage_raw(temp_db, "reg", "txt")
        docs = _candidate_docs(700)
        assert len(docs) == 1
        assert docs[0]["form"] == form
        assert docs[0]["registration_known"] is True

    def test_424b_off_f1_only_excluded(self, temp_db):
        # The S-1/F-1 exclusion fires on the F-1 prefix too, not just S-1.
        temp_db.add_company(800, "HHH")
        temp_db.add_filing("b1", 800, form="424B4", filing_date="2026-02-01",
                           file_number="333-F1")
        _stage_raw(temp_db, "b1", "txt")
        temp_db.add_filing("f1f", 800, form="F-1", filing_date="2026-01-01",
                           file_number="333-F1")
        assert _candidate_docs(800) == []

    def test_no_matching_filings_empty(self, temp_db):
        temp_db.add_company(600, "FFF")
        assert _candidate_docs(600) == []


class TestAccessionText:
    def test_multiple_rows_joined_with_newline(self, temp_db):
        temp_db.add_company(100, "AAA")
        temp_db.add_filing("acc", 100, form="424B5")
        _stage_raw(temp_db, "acc", "b1", doc_name="d1")
        _stage_raw(temp_db, "acc", "b2", doc_name="d2")
        assert _accession_text("acc") == "b1\nb2"

    def test_no_rows_returns_empty_string(self, temp_db):
        # Not None, an empty string.
        assert _accession_text("missing") == ""

    def test_empty_content_row_contributes_empty_string(self, temp_db):
        # content_md is NOT NULL in the schema (a literal NULL is rejected by
        # SQLite, so the docstring's "NULL content_md -> ''" case is
        # unreachable via normal staging). The `r["content_md"] or ""`
        # coalesce IS still exercised by an empty-string row: it joins as an
        # empty segment, so a real row + an empty row -> "real\n".
        temp_db.add_company(100, "AAA")
        temp_db.add_filing("acc", 100, form="424B5")
        _stage_raw(temp_db, "acc", "real", doc_name="d1")
        _stage_raw(temp_db, "acc", "", doc_name="d2")
        # rows come back in PK (doc_name) order: d1 then d2.
        assert _accession_text("acc") == "real\n"

    def test_single_row_verbatim(self, temp_db):
        temp_db.add_company(100, "AAA")
        temp_db.add_filing("acc", 100, form="424B5")
        _stage_raw(temp_db, "acc", "the only content")
        assert _accession_text("acc") == "the only content"


class TestIb6CoverStatus:
    TODAY = date(2026, 6, 1)

    def test_legend_present_yields_baby(self, temp_db):
        temp_db.add_company(100, "AAA")
        temp_db.add_filing("acc", 100, form="424B5", filing_date="2026-05-01")
        _stage_raw(temp_db, "acc", CANONICAL_LEGEND)
        st = ib6_cover_status(100, today=self.TODAY)
        assert st["regime"] == "baby"
        assert st["legend"] is not None
        assert st["legend"].present is True
        assert st["source_accession"] == "acc"
        assert st["source_form"] == "424B5"
        assert st["source_filing_date"] == "2026-05-01"

    def test_no_legend_known_registration_yields_unrestricted(self, temp_db):
        temp_db.add_company(100, "AAA")
        temp_db.add_filing("acc", 100, form="S-3", filing_date="2026-05-01")
        _stage_raw(temp_db, "acc", "no legend here at all")
        st = ib6_cover_status(100, today=self.TODAY)
        assert st["regime"] == "unrestricted"
        assert st["legend"] is None
        assert st["source_accession"] == "acc"

    def test_no_legend_unknown_424b_skipped_falls_to_next(self, temp_db):
        temp_db.add_company(100, "AAA")
        # newest: 424B5 unknown registration, no legend -> skipped
        temp_db.add_filing("new", 100, form="424B5", filing_date="2026-05-01",
                           file_number=None)
        _stage_raw(temp_db, "new", "no legend at all")
        # older: S-3 (reg known), no legend -> unrestricted
        temp_db.add_filing("old", 100, form="S-3", filing_date="2026-04-01")
        _stage_raw(temp_db, "old", "no legend either")
        st = ib6_cover_status(100, today=self.TODAY)
        assert st["regime"] == "unrestricted"
        assert st["source_accession"] == "old"

    def test_resale_three_mentions_skipped(self, temp_db):
        temp_db.add_company(100, "AAA")
        temp_db.add_filing("r1", 100, form="S-3", filing_date="2026-05-01")
        _stage_raw(
            temp_db, "r1",
            "This prospectus relates to selling stockholders. The selling "
            "stockholders may sell. These selling stockholders.",
        )
        st = ib6_cover_status(100, today=self.TODAY)
        # resale-only -> skipped -> nothing else -> all-None
        assert st["regime"] is None

    def test_resale_two_mentions_not_skipped(self, temp_db):
        temp_db.add_company(100, "AAA")
        temp_db.add_filing("r1", 100, form="S-3", filing_date="2026-05-01")
        _stage_raw(
            temp_db, "r1",
            "selling stockholders. The selling stockholders may sell.",
        )
        st = ib6_cover_status(100, today=self.TODAY)
        # 2 < _RESALE_MENTIONS_MIN (3) -> not resale -> unrestricted
        assert st["regime"] == "unrestricted"

    def test_doc_exactly_at_cutoff_included(self, temp_db):
        temp_db.add_company(100, "AAA")
        cutoff = (self.TODAY - timedelta(days=540)).isoformat()
        temp_db.add_filing("acc", 100, form="S-3", filing_date=cutoff)
        _stage_raw(temp_db, "acc", "no legend")
        # filing_date == cutoff is NOT strictly less than cutoff -> included
        st = ib6_cover_status(100, today=self.TODAY)
        assert st["regime"] == "unrestricted"

    def test_doc_one_day_older_than_cutoff_breaks(self, temp_db):
        temp_db.add_company(100, "AAA")
        older = (self.TODAY - timedelta(days=541)).isoformat()
        temp_db.add_filing("acc", 100, form="S-3", filing_date=older)
        _stage_raw(temp_db, "acc", "no legend")
        st = ib6_cover_status(100, today=self.TODAY)
        assert st["regime"] is None
        assert st["source_accession"] is None

    def test_empty_candidate_list_all_none(self, temp_db):
        temp_db.add_company(100, "AAA")
        st = ib6_cover_status(100, today=self.TODAY)
        assert st == {
            "regime": None, "legend": None, "source_accession": None,
            "source_form": None, "source_filing_date": None,
        }


class TestIb6CoverStatusCached:
    TODAY = date(2026, 6, 1)

    def test_caches_result_across_db_changes(self, temp_db):
        temp_db.add_company(100, "AAA")
        temp_db.add_filing("acc", 100, form="S-3", filing_date="2026-05-01")
        _stage_raw(temp_db, "acc", "no legend")
        first = ib6_cover_status_cached(100, today=self.TODAY)
        assert first["regime"] == "unrestricted"
        # Mutate the DB underneath the cache: drop everything for cik 100.
        temp_db.execute("DELETE FROM dilution_raw WHERE accession_number=?",
                        ("acc",))
        temp_db.execute("DELETE FROM dilution_filings WHERE cik=?", (100,))
        # Cached result is returned despite the DB change.
        second = ib6_cover_status_cached(100, today=self.TODAY)
        assert second["regime"] == "unrestricted"
        # After clearing, the fresh (now-empty) DB yields None.
        ib6._cached_status.cache_clear()
        third = ib6_cover_status_cached(100, today=self.TODAY)
        assert third["regime"] is None

    def test_str_castable_cik_keyed_as_int(self, temp_db):
        temp_db.add_company(100, "AAA")
        temp_db.add_filing("acc", 100, form="S-3", filing_date="2026-05-01")
        _stage_raw(temp_db, "acc", "no legend")
        as_int = ib6_cover_status_cached(100, today=self.TODAY)
        as_str = ib6_cover_status_cached("100", today=self.TODAY)
        assert as_int == as_str

    def test_today_none_resolves_to_today_and_cik_int_cast(self, monkeypatch):
        # The wrapper resolves today=None to date.today().isoformat() and
        # int()-casts the cik BEFORE building the lru_cache key. Capture the
        # exact args _cached_status is called with (no DB, no date.today
        # dependence in the assertion: compare to the same call's today()).
        captured = {}

        def fake(cik, today_iso):
            captured["cik"] = cik
            captured["today_iso"] = today_iso
            return {"regime": None}

        monkeypatch.setattr(ib6, "_cached_status", fake)
        before = date.today().isoformat()
        ib6_cover_status_cached("100")  # str cik, today=None
        after = date.today().isoformat()
        assert captured["cik"] == 100
        assert isinstance(captured["cik"], int)
        # The resolved iso must be the run-day (tolerate a midnight rollover
        # between the two date.today() reads).
        assert captured["today_iso"] in {before, after}


# ════════════════════════════════════════════════════════════════════════
# IO TIER (entity_public_float_latest)
# ════════════════════════════════════════════════════════════════════════

class _FakeResp:
    def __init__(self, *, status=200, payload=None, raises=None):
        self.status_code = status
        self._payload = payload or {}
        self._raises = raises

    def raise_for_status(self):
        if self._raises is not None:
            raise self._raises

    def json(self):
        return self._payload


@pytest.fixture
def fake_requests(monkeypatch):
    """Inject a fake ``requests`` module so the lazy `import requests` inside
    entity_public_float_latest resolves to it. Returns a setter closure that
    installs a `.get` for the next call."""
    mod = types.ModuleType("requests")

    def set_get(fn):
        mod.get = fn

    set_get(lambda *a, **k: _FakeResp())
    monkeypatch.setitem(sys.modules, "requests", mod)
    return set_get


class TestEntityPublicFloatLatest:
    def test_404_returns_none(self, fake_requests):
        fake_requests(lambda *a, **k: _FakeResp(status=404))
        assert entity_public_float_latest(123) is None

    def test_request_exception_returns_none_and_warns(self, fake_requests, caplog):
        def raiser(*a, **k):
            raise RuntimeError("timeout")
        fake_requests(raiser)
        with caplog.at_level("WARNING"):
            assert entity_public_float_latest(123) is None
        assert any("EntityPublicFloat fetch failed" in r.message
                   for r in caplog.records)

    def test_500_raise_for_status_caught(self, fake_requests):
        fake_requests(lambda *a, **k: _FakeResp(
            status=500, raises=RuntimeError("server error")))
        assert entity_public_float_latest(123) is None

    def test_all_below_min_returns_none(self, fake_requests):
        payload = {"units": {"USD": [
            {"val": 50_000, "end": "2025-06-30", "filed": "2025-08-01"},
            {"val": 0, "end": "2025-06-30", "filed": "2025-08-01"},
        ]}}
        fake_requests(lambda *a, **k: _FakeResp(payload=payload))
        assert entity_public_float_latest(123) is None

    def test_exactly_min_inclusive(self, fake_requests):
        payload = {"units": {"USD": [
            {"val": 1e5, "end": "2025-06-30", "filed": "2025-08-01",
             "form": "10-K"},
        ]}}
        fake_requests(lambda *a, **k: _FakeResp(payload=payload))
        result = entity_public_float_latest(123)
        assert result is not None
        assert result["value"] == 100_000.0

    def test_picks_max_by_end_then_filed(self, fake_requests):
        payload = {"units": {"USD": [
            {"val": 1_000_000, "end": "2024-06-30", "filed": "2024-08-01",
             "form": "10-K"},
            {"val": 2_000_000, "end": "2025-06-30", "filed": "2025-08-01",
             "form": "10-K"},
            {"val": 9_000_000, "end": "2025-06-30", "filed": "2024-12-01",
             "form": "10-K/A"},  # same end, older filed -> loses tiebreak
        ]}}
        fake_requests(lambda *a, **k: _FakeResp(payload=payload))
        result = entity_public_float_latest(123)
        assert result == {
            "value": 2_000_000.0, "as_of": "2025-06-30",
            "filed": "2025-08-01", "form": "10-K",
        }

    def test_missing_units_usd_key_returns_none(self, fake_requests):
        fake_requests(lambda *a, **k: _FakeResp(payload={"units": {}}))
        assert entity_public_float_latest(123) is None

    def test_value_cast_to_float_missing_meta_is_none(self, fake_requests):
        # Qualifying value but no end/filed/form keys -> still returned with
        # None metadata, value coerced to float.
        payload = {"units": {"USD": [{"val": 5_000_000}]}}
        fake_requests(lambda *a, **k: _FakeResp(payload=payload))
        result = entity_public_float_latest(123)
        assert result == {
            "value": 5_000_000.0, "as_of": None, "filed": None, "form": None,
        }
        assert isinstance(result["value"], float)

    def test_negative_and_null_val_filtered(self, fake_requests):
        # `(f.get("val") or 0) >= 1e5`: a negative val fails the >= check and
        # a null val coerces to 0 -> both dropped, leaving only the 200k fact.
        payload = {"units": {"USD": [
            {"val": -10_000_000, "end": "2025-06-30", "filed": "2025-08-01"},
            {"val": None, "end": "2025-06-30", "filed": "2025-08-01"},
            {"val": 200_000, "end": "2024-06-30", "filed": "2024-08-01",
             "form": "10-K"},
        ]}}
        fake_requests(lambda *a, **k: _FakeResp(payload=payload))
        result = entity_public_float_latest(123)
        assert result is not None
        assert result["value"] == 200_000.0

    def test_request_targets_zero_padded_cik_url_with_identity_header(
            self, fake_requests):
        # Pin the exact SEC seam: a 10-digit zero-padded CIK in the URL and the
        # config.EDGAR_IDENTITY User-Agent. Also proves no real network egress
        # (the call is captured, never issued).
        import config
        captured = {}

        def capture(url, *a, **k):
            captured["url"] = url
            captured["headers"] = k.get("headers")
            captured["timeout"] = k.get("timeout")
            return _FakeResp(status=404)

        fake_requests(capture)
        assert entity_public_float_latest(1837493) is None
        assert captured["url"] == (
            "https://data.sec.gov/api/xbrl/companyconcept/"
            "CIK0001837493/dei/EntityPublicFloat.json"
        )
        assert captured["headers"]["User-Agent"] == config.EDGAR_IDENTITY
        assert captured["timeout"] == 10


# ════════════════════════════════════════════════════════════════════════
# ib6_regime (tiered) — monkeypatch the two seams the slice names
# ════════════════════════════════════════════════════════════════════════

class TestIb6Regime:
    TODAY = date(2026, 6, 1)

    def test_stamp_supersedes_and_no_float_fetch(self, monkeypatch):
        stamp = {"regime": "baby", "legend": None, "source_accession": "a",
                 "source_form": "424B5", "source_filing_date": "2026-01-01"}
        monkeypatch.setattr(ib6, "ib6_cover_status_cached",
                            lambda cik, today=None: stamp)

        def _boom(*a, **k):  # the 10k_float branch must never run
            raise AssertionError("public float fetched despite stamp")
        monkeypatch.setattr(ib6, "_cached_public_float", _boom)
        result = ib6_regime(444, today=self.TODAY)
        assert result["source"] == "stamp"
        assert result["regime"] == "baby"
        # The full stamp dict must be carried through and merged with the
        # source tag — not just regime/source. The source_* provenance fields
        # are load-bearing for the caller (which doc stamped the regime).
        assert result == {**stamp, "source": "stamp"}
        assert result["source_accession"] == "a"
        assert result["source_form"] == "424B5"
        assert result["source_filing_date"] == "2026-01-01"

    def test_stamp_unrestricted_also_supersedes(self, monkeypatch):
        # The supersede check is `st["regime"] is not None`, so an
        # 'unrestricted' stamp (legend-absent primary prospectus) ALSO wins
        # over the 10-K float, never consulting the public-float fetch.
        stamp = {"regime": "unrestricted", "legend": None,
                 "source_accession": "s3acc", "source_form": "S-3",
                 "source_filing_date": "2026-02-01"}
        monkeypatch.setattr(ib6, "ib6_cover_status_cached",
                            lambda cik, today=None: stamp)

        def _boom(*a, **k):
            raise AssertionError("public float fetched despite stamp")
        monkeypatch.setattr(ib6, "_cached_public_float", _boom)
        result = ib6_regime(444, today=self.TODAY)
        assert result == {**stamp, "source": "stamp"}

    def _no_stamp(self, monkeypatch):
        monkeypatch.setattr(
            ib6, "ib6_cover_status_cached",
            lambda cik, today=None: {
                "regime": None, "legend": None, "source_accession": None,
                "source_form": None, "source_filing_date": None},
        )

    def test_float_below_75m_is_baby(self, monkeypatch):
        self._no_stamp(monkeypatch)
        filed = (self.TODAY - timedelta(days=100)).isoformat()
        monkeypatch.setattr(
            ib6, "_cached_public_float",
            lambda cik, iso: {"value": 50_000_000.0, "as_of": "2025-06-30",
                              "filed": filed, "form": "10-K"},
        )
        result = ib6_regime(444, today=self.TODAY)
        assert result["source"] == "10k_float"
        assert result["regime"] == "baby"
        assert result["public_float_usd"] == 50_000_000.0
        assert result["public_float_as_of"] == "2025-06-30"
        assert result["source_form"] == "10-K"
        # The 10k_float branch carries no prospectus provenance: legend and
        # source_accession are None (the regime came from the XBRL fact, not
        # a cover legend), and source_filing_date is the fact's filed date.
        assert result["legend"] is None
        assert result["source_accession"] is None
        assert result["source_filing_date"] == filed

    def test_float_exactly_75m_is_unrestricted(self, monkeypatch):
        self._no_stamp(monkeypatch)
        filed = (self.TODAY - timedelta(days=100)).isoformat()
        monkeypatch.setattr(
            ib6, "_cached_public_float",
            lambda cik, iso: {"value": 75_000_000.0, "as_of": "x",
                              "filed": filed, "form": "10-K"},
        )
        # strict < for baby -> exactly 75M is unrestricted
        assert ib6_regime(444, today=self.TODAY)["regime"] == "unrestricted"

    def test_float_above_75m_is_unrestricted(self, monkeypatch):
        self._no_stamp(monkeypatch)
        filed = (self.TODAY - timedelta(days=100)).isoformat()
        monkeypatch.setattr(
            ib6, "_cached_public_float",
            lambda cik, iso: {"value": 200_000_000.0, "as_of": "x",
                              "filed": filed, "form": "10-K"},
        )
        assert ib6_regime(444, today=self.TODAY)["regime"] == "unrestricted"

    def test_filed_age_exactly_540_included(self, monkeypatch):
        self._no_stamp(monkeypatch)
        filed = (self.TODAY - timedelta(days=540)).isoformat()
        monkeypatch.setattr(
            ib6, "_cached_public_float",
            lambda cik, iso: {"value": 10_000_000.0, "as_of": "x",
                              "filed": filed, "form": "10-K"},
        )
        # age <= _PUBLIC_FLOAT_STALE_DAYS (540) -> included
        result = ib6_regime(444, today=self.TODAY)
        assert result["regime"] == "baby"
        assert result["source"] == "10k_float"

    def test_filed_age_541_falls_through_to_none(self, monkeypatch):
        self._no_stamp(monkeypatch)
        filed = (self.TODAY - timedelta(days=541)).isoformat()
        monkeypatch.setattr(
            ib6, "_cached_public_float",
            lambda cik, iso: {"value": 10_000_000.0, "as_of": "x",
                              "filed": filed, "form": "10-K"},
        )
        result = ib6_regime(444, today=self.TODAY)
        assert result["regime"] is None
        assert result["source"] is None

    def test_no_stamp_no_float_returns_none(self, monkeypatch):
        self._no_stamp(monkeypatch)
        monkeypatch.setattr(ib6, "_cached_public_float",
                            lambda cik, iso: None)
        result = ib6_regime(444, today=self.TODAY)
        assert result == {
            "regime": None, "legend": None, "source": None,
            "source_accession": None, "source_form": None,
            "source_filing_date": None,
        }

    def test_pf_missing_filed_falls_through(self, monkeypatch):
        self._no_stamp(monkeypatch)
        # pf present but no 'filed' key -> branch not taken
        monkeypatch.setattr(
            ib6, "_cached_public_float",
            lambda cik, iso: {"value": 10_000_000.0, "as_of": "x"},
        )
        assert ib6_regime(444, today=self.TODAY)["regime"] is None
