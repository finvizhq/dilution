"""Unit tests for dilution/ledger/item_classification.py.

Covers the deterministic 8-K dilution-relevance gate (classify_8k), the
per-filing create-tool pruning (prune_create_tools + the pure
_keep_create_type), and the content-side expected-call detector
(expected_call_classes).

The autouse ``temp_db`` fixture (conftest.py) reroutes db.get_conn() to a
fresh per-test SQLite DB with the production schema, so the DB-backed
functions work with no monkeypatching — we just stage rows. No network /
LLM / vendor calls are made anywhere in this file.

Schema note: dilution_raw has a NOT-NULL doc_name / content_md and a FK to
dilution_filings(accession_number). Helpers below always stage a parent
filing row before inserting raw exhibit rows.
"""

from __future__ import annotations

import types

import pytest

from dilution.ledger import item_classification as ic
from dilution.ledger.item_classification import (
    Item8KVerdict,
    classify_8k,
    expected_call_classes,
    prune_create_tools,
    _keep_create_type,
)


# ── small helpers ──────────────────────────────────────────────────────
def _stub_tool(name):
    """A lightweight stand-in for an LLM tool schema exposing ``.name``."""
    return types.SimpleNamespace(name=name)


def _add_raw(temp_db, accession, cik, doc_type, *, doc_name=None,
             form="8-K", items=None, with_filing=True):
    """Stage one dilution_raw row (and its required parent filing).

    dilution_raw.accession_number is a FK to dilution_filings, and
    doc_name/content_md are NOT NULL, so we always create a filing first
    unless the caller already created one (with_filing=False).
    """
    if with_filing:
        # Use a single shared filing per accession; ignore duplicate inserts.
        try:
            temp_db.add_filing(accession, cik, form=form, items=items)
        except Exception:
            pass
    temp_db.execute(
        "INSERT INTO dilution_raw "
        "(accession_number, doc_name, doc_type, content_md, downloaded_at) "
        "VALUES (?,?,?,?,?)",
        (accession, doc_name or f"{doc_type}.htm", doc_type, "body",
         "2026-01-01T00:00:00Z"),
    )


# ════════════════════════════════════════════════════════════════════════
# classify_8k  (db_backed three-way verdict)
# ════════════════════════════════════════════════════════════════════════
class TestClassify8kFormGating:
    @pytest.mark.parametrize("form", ["6-K", "S-3", "424B5", "S-1", "10-K"])
    def test_non_8k_forms_fail_open_to_process(self, form, temp_db):
        # Even with a fully-populated 8-K-shaped row present, a non-8-K form
        # returns _PROCESS without consulting the DB.
        temp_db.add_filing("ACC-NON", 11, form="8-K", items="3.02")
        v = classify_8k(11, "ACC-NON", form)
        assert v == Item8KVerdict(skip=False, must_record=False, reason="")

    def test_form_none_fails_open(self, temp_db):
        v = classify_8k(11, "ACC-NON", None)
        assert v.skip is False and v.must_record is False

    def test_form_empty_string_fails_open(self, temp_db):
        v = classify_8k(11, "ACC-NON", "")
        assert v.skip is False and v.must_record is False

    @pytest.mark.parametrize("form", ["8-k", "8-K", "8-K/A", "8-K ", "8-KX"])
    def test_8k_variants_are_gated(self, form, temp_db):
        # All 8-K* forms (upper().startswith('8-K')) ARE gated. We stage a
        # filing that produces skip=True so a gated form is observably
        # different from the fail-open process verdict.
        temp_db.add_filing("ACC-9", 11, form="8-K", items="9.01")
        v = classify_8k(11, "ACC-9", form)
        assert v.skip is True


class TestClassify8kFailOpen:
    def test_no_filing_and_no_docs_fails_open(self, temp_db):
        v = classify_8k(99, "MISSING-ACC", "8-K")
        assert v == Item8KVerdict(skip=False, must_record=False, reason="")

    def test_filing_row_but_null_items_with_doc_still_evaluated(self, temp_db):
        # items NULL coerces to '' so the verdict rests on doc_types alone.
        # EX-4.1 is substantive AND must-record → must_record (not fail-open).
        temp_db.add_filing("ACC-NULLITEMS", 11, form="8-K", items=None)
        _add_raw(temp_db, "ACC-NULLITEMS", 11, "EX-4.1", with_filing=False)
        v = classify_8k(11, "ACC-NULLITEMS", "8-K")
        assert v.skip is False
        assert v.must_record is True


class TestClassify8kSkip:
    def test_items_9_01_only_no_substantive_exhibit_skips(self, temp_db):
        # 9.01 is the exhibit index, NOT in DILUTIVE_ITEMS. With an EX-101
        # (non-substantive) doc the filing has neither a dilutive item nor a
        # substantive exhibit → skip.
        temp_db.add_filing("ACC-901", 11, form="8-K", items="9.01")
        _add_raw(temp_db, "ACC-901", 11, "EX-101.INS", with_filing=False)
        v = classify_8k(11, "ACC-901", "8-K")
        assert v.skip is True
        assert v.must_record is False
        assert "9.01" in v.reason

    def test_earnings_with_only_nonsubstantive_doc_skips(self, temp_db):
        temp_db.add_filing("ACC-202", 11, form="8-K", items="2.02")
        _add_raw(temp_db, "ACC-202", 11, "EX-101.INS", with_filing=False)
        v = classify_8k(11, "ACC-202", "8-K")
        assert v.skip is True

    def test_skip_reason_shows_empty_glyph_when_items_blank(self, temp_db):
        # items empty + only a non-substantive exhibit → skip; the reason
        # renders the ∅ glyph for the empty items list.
        temp_db.add_filing("ACC-EMPTY", 11, form="8-K", items="")
        _add_raw(temp_db, "ACC-EMPTY", 11, "EX-101.INS", with_filing=False)
        v = classify_8k(11, "ACC-EMPTY", "8-K")
        assert v.skip is True
        assert "∅" in v.reason

    def test_skip_short_circuits_must_record(self, temp_db):
        # A 9.01-only / no-exhibit filing is skip=True and MUST NOT be
        # must_record — even though no must-record item is present, the gate
        # ordering returns early before the must-record block.
        temp_db.add_filing("ACC-SC", 11, form="8-K", items="9.01")
        _add_raw(temp_db, "ACC-SC", 11, "EX-101.INS", with_filing=False)
        v = classify_8k(11, "ACC-SC", "8-K")
        assert v.skip is True and v.must_record is False


class TestClassify8kProcess:
    def test_earnings_with_substantive_ex99_is_process(self, temp_db):
        # 2.02 (earnings, not dilutive) + EX-99.1 (substantive) → NOT skip,
        # but EX-99 is not a must-record prefix and 2.02 is not must-record →
        # process.
        temp_db.add_filing("ACC-EARN", 11, form="8-K", items="2.02,9.01")
        _add_raw(temp_db, "ACC-EARN", 11, "EX-99.1", with_filing=False)
        v = classify_8k(11, "ACC-EARN", "8-K")
        assert v == Item8KVerdict(skip=False, must_record=False, reason="")

    def test_dilutive_item_1_01_no_mustrecord_is_process(self, temp_db):
        # 1.01 is dilutive (clears the skip gate) but not in MUST_RECORD_ITEMS
        # and there's no must-record exhibit → process.
        temp_db.add_filing("ACC-101", 11, form="8-K", items="1.01")
        v = classify_8k(11, "ACC-101", "8-K")
        assert v == Item8KVerdict(skip=False, must_record=False, reason="")

    @pytest.mark.parametrize(
        "item",
        # Every DILUTIVE_ITEMS code that is NOT a MUST_RECORD_ITEMS code. Each
        # clears the skip gate (so not skip) yet is not a hard issuance signal
        # (so not must_record) → process. Locks the DILUTIVE-minus-MUST set
        # through the public function rather than asserting the constants.
        ["1.01", "1.02", "2.01", "2.04", "3.03", "5.01", "7.01", "8.01"],
    )
    def test_dilutive_nonmust_items_are_process(self, item, temp_db):
        temp_db.add_filing("ACC-DIL", 11, form="8-K", items=item)
        v = classify_8k(11, "ACC-DIL", "8-K")
        assert v == Item8KVerdict(skip=False, must_record=False, reason="")

    def test_dilutive_item_with_nonsubstantive_exhibit_is_process(self, temp_db):
        # 1.01 (dilutive) clears the skip gate ON ITS OWN; the only attached
        # doc is non-substantive (EX-101.INS). The skip co-gate is an AND
        # (no dilutive item AND no substantive exhibit), so a dilutive item
        # alone prevents skip even with no substantive exhibit. Not must-record
        # (1.01 not a hard signal, EX-101 not a must-record prefix) → process.
        temp_db.add_filing("ACC-DILNS", 11, form="8-K", items="1.01")
        _add_raw(temp_db, "ACC-DILNS", 11, "EX-101.INS", with_filing=False)
        v = classify_8k(11, "ACC-DILNS", "8-K")
        assert v == Item8KVerdict(skip=False, must_record=False, reason="")

    def test_ex99_alone_empty_items_is_process(self, temp_db):
        # EX-99.1 is substantive (clears skip) but is NOT a must-record prefix,
        # and items is empty → process. Isolates the EX-99 substantive-but-not
        # -must-record path without any dilutive item present.
        temp_db.add_filing("ACC-EX99", 11, form="8-K", items="")
        _add_raw(temp_db, "ACC-EX99", 11, "EX-99.1", with_filing=False)
        v = classify_8k(11, "ACC-EX99", "8-K")
        assert v == Item8KVerdict(skip=False, must_record=False, reason="")

    def test_ex10_substantive_alone_is_process(self, temp_db):
        # EX-10.1 (SPA) is substantive → clears skip, but EX-10 is not a
        # must-record prefix and items is empty → process (neither skip nor
        # must_record). Guards the "substantive but not must-record exhibit"
        # middle ground.
        temp_db.add_filing("ACC-EX10P", 11, form="8-K", items="")
        _add_raw(temp_db, "ACC-EX10P", 11, "EX-10.1", with_filing=False)
        v = classify_8k(11, "ACC-EX10P", "8-K")
        assert v == Item8KVerdict(skip=False, must_record=False, reason="")


class TestClassify8kMustRecord:
    def test_item_3_02_must_record(self, temp_db):
        temp_db.add_filing("ACC-302", 11, form="8-K", items="3.02")
        v = classify_8k(11, "ACC-302", "8-K")
        assert v.must_record is True
        assert v.skip is False
        assert "items=3.02" in v.reason

    def test_item_5_03_alone_must_record(self, temp_db):
        # 5.03 = charter amendment / Certificate of Designation → preferred.
        temp_db.add_filing("ACC-503", 11, form="8-K", items="5.03")
        v = classify_8k(11, "ACC-503", "8-K")
        assert v.must_record is True
        assert "items=5.03" in v.reason

    def test_item_2_03_must_record(self, temp_db):
        temp_db.add_filing("ACC-203", 11, form="8-K", items="2.03")
        v = classify_8k(11, "ACC-203", "8-K")
        assert v.must_record is True
        assert "items=2.03" in v.reason

    def test_ex4_exhibit_empty_items_must_record(self, temp_db):
        # EX-4.1 is substantive (clears skip) AND in MUST_RECORD prefixes.
        temp_db.add_filing("ACC-EX4", 11, form="8-K", items="")
        _add_raw(temp_db, "ACC-EX4", 11, "EX-4.1", with_filing=False)
        v = classify_8k(11, "ACC-EX4", "8-K")
        assert v.must_record is True
        assert v.skip is False
        assert "exhibits=EX-4.1" in v.reason

    def test_ex3_with_item_503_reason_joins_both(self, temp_db):
        temp_db.add_filing("ACC-BOTH", 11, form="8-K", items="5.03")
        _add_raw(temp_db, "ACC-BOTH", 11, "EX-3.1", with_filing=False)
        v = classify_8k(11, "ACC-BOTH", "8-K")
        assert v.must_record is True
        assert v.reason == "items=5.03; exhibits=EX-3.1"

    def test_must_items_lexically_sorted(self, temp_db):
        # items='5.03,2.03,3.02' → reason lists must items in lexical sort.
        temp_db.add_filing("ACC-SORT", 11, form="8-K", items="5.03,2.03,3.02")
        v = classify_8k(11, "ACC-SORT", "8-K")
        assert v.must_record is True
        assert v.reason == "items=2.03,3.02,5.03"

    def test_must_exhibits_dedup_and_sort(self, temp_db):
        # Two EX-4.1 rows + one EX-3.2 → set-dedup then sorted: EX-3.2,EX-4.1.
        temp_db.add_filing("ACC-EXDD", 11, form="8-K", items="")
        _add_raw(temp_db, "ACC-EXDD", 11, "EX-4.1",
                 doc_name="a.htm", with_filing=False)
        _add_raw(temp_db, "ACC-EXDD", 11, "EX-4.1",
                 doc_name="b.htm", with_filing=False)
        _add_raw(temp_db, "ACC-EXDD", 11, "EX-3.2",
                 doc_name="c.htm", with_filing=False)
        v = classify_8k(11, "ACC-EXDD", "8-K")
        assert v.must_record is True
        assert v.reason == "exhibits=EX-3.2,EX-4.1"

    def test_dilutive_nonmust_item_with_mustrecord_exhibit_reason_exhibit_only(
            self, temp_db):
        # 1.01 (dilutive, NOT in MUST_RECORD_ITEMS) clears the skip gate, and
        # the EX-4.1 doc is the ONLY must-record signal. The reason's
        # filter(None, [...]) must drop the empty items= segment and emit
        # 'exhibits=EX-4.1' alone (no leading 'items=;'). This isolates the
        # join branch where must_items is empty yet skip was cleared by a
        # non-must dilutive item — distinct from the empty-items branch which
        # clears skip via the exhibit itself.
        temp_db.add_filing("ACC-MIX", 11, form="8-K", items="1.01")
        _add_raw(temp_db, "ACC-MIX", 11, "EX-4.1", with_filing=False)
        v = classify_8k(11, "ACC-MIX", "8-K")
        assert v.skip is False
        assert v.must_record is True
        assert v.reason == "exhibits=EX-4.1"  # no 'items=' segment


class TestClassify8kItemParsing:
    def test_whitespace_in_items_stripped(self, temp_db):
        # '1.01, 3.02 ,9.01' must strip to {'1.01','3.02','9.01'} so 3.02 is
        # detected as must-record.
        temp_db.add_filing("ACC-WS", 11, form="8-K", items="1.01, 3.02 ,9.01")
        v = classify_8k(11, "ACC-WS", "8-K")
        assert v.must_record is True
        assert v.reason == "items=3.02"

    def test_duplicate_and_empty_fragments_deduped(self, temp_db):
        # '1.01,,1.01' → {'1.01'}: dilutive (clears skip) but not must-record.
        temp_db.add_filing("ACC-DUP", 11, form="8-K", items="1.01,,1.01")
        v = classify_8k(11, "ACC-DUP", "8-K")
        assert v == Item8KVerdict(skip=False, must_record=False, reason="")

    def test_null_doc_type_coerced_not_matched(self, temp_db):
        # A NULL doc_type coerces to '' and is not a substantive prefix. With
        # only items='9.01' the filing skips (NULL doc doesn't rescue it).
        temp_db.add_filing("ACC-NULLDOC", 11, form="8-K", items="9.01")
        temp_db.execute(
            "INSERT INTO dilution_raw "
            "(accession_number, doc_name, doc_type, content_md, downloaded_at)"
            " VALUES (?,?,?,?,?)",
            ("ACC-NULLDOC", "d.htm", None, "body", "2026-01-01T00:00:00Z"),
        )
        v = classify_8k(11, "ACC-NULLDOC", "8-K")
        assert v.skip is True

    def test_filing_matched_by_cik_and_accession(self, temp_db):
        # A filing row exists under a DIFFERENT cik, plus a raw doc under the
        # queried accession. classify_8k(cik) won't find the filing row (so
        # items='') but DOES find the raw doc (matched by accession alone),
        # so the doc_types path governs the verdict.
        temp_db.add_filing("ACC-XCIK", 777, form="8-K", items="3.02")
        _add_raw(temp_db, "ACC-XCIK", 777, "EX-4.1", with_filing=False)
        v = classify_8k(11, "ACC-XCIK", "8-K")  # cik 11 != 777
        # items not found (''), but EX-4.1 doc is → must_record on exhibit.
        assert v.must_record is True
        assert v.reason == "exhibits=EX-4.1"


# ════════════════════════════════════════════════════════════════════════
# _keep_create_type  (pure: OR of four signals)
# ════════════════════════════════════════════════════════════════════════
class TestKeepCreateTypeEmpty:
    @pytest.mark.parametrize("typ",
                             ["warrant", "preferred", "convertible", "atm"])
    def test_all_empty_inputs_drop_every_type(self, typ):
        assert _keep_create_type(
            typ, filing_text="", item_codes=set(),
            doc_types=[], open_types=set()) is False

    def test_filing_text_none_does_not_raise(self):
        # 'or ""' coercion guard — regression check.
        assert _keep_create_type(
            "warrant", filing_text=None, item_codes=set(),
            doc_types=[], open_types=set()) is False


class TestKeepCreateTypeKeywords:
    @pytest.mark.parametrize("typ,text", [
        ("warrant", "WARRANT"),
        ("warrant", "the Warrant Agreement"),
        ("convertible", "Convertible Debenture"),
        ("convertible", "Promissory Note"),
        ("convertible", "a convertible note"),
        ("preferred", "preference shares"),   # FPI plural
        ("preferred", "preference share"),    # singular via s?
        ("preferred", "Preferred Stock"),
        ("atm", "At-the-Market"),
        ("atm", "at the market"),
        ("atm", "Sales Agreement"),
        ("atm", "Equity Distribution"),
    ])
    def test_keyword_matches_keep_case_insensitive(self, typ, text):
        assert _keep_create_type(
            typ, filing_text=text, item_codes=set(),
            doc_types=[], open_types=set()) is True

    def test_convertible_does_not_fire_on_bare_note(self):
        # 'note' without convertible/debenture/promissory must NOT keep.
        assert _keep_create_type(
            "convertible", filing_text="see note above", item_codes=set(),
            doc_types=[], open_types=set()) is False

    def test_atm_does_not_fire_on_generic_distribution(self):
        assert _keep_create_type(
            "atm", filing_text="a cash distribution to holders",
            item_codes=set(), doc_types=[], open_types=set()) is False


class TestKeepCreateTypeItemRescue:
    def test_preferred_rescued_by_5_03(self):
        assert _keep_create_type(
            "preferred", filing_text="", item_codes={"5.03"},
            doc_types=[], open_types=set()) is True

    def test_convertible_rescued_by_2_03(self):
        assert _keep_create_type(
            "convertible", filing_text="", item_codes={"2.03"},
            doc_types=[], open_types=set()) is True

    @pytest.mark.parametrize("codes", [{"5.03"}, {"2.03"}, {"3.02"}])
    def test_warrant_has_no_item_rescue(self, codes):
        # warrant is absent from _CREATE_ITEM_RESCUE → items never keep it.
        assert _keep_create_type(
            "warrant", filing_text="", item_codes=codes,
            doc_types=[], open_types=set()) is False

    def test_atm_has_no_item_rescue(self):
        assert _keep_create_type(
            "atm", filing_text="", item_codes={"5.03", "2.03"},
            doc_types=[], open_types=set()) is False


class TestKeepCreateTypeExhibitRescue:
    @pytest.mark.parametrize("typ,doc", [
        ("warrant", "EX-4.1"),
        ("preferred", "EX-3.1"),
        ("convertible", "EX-4.1"),
        ("convertible", "EX-4."),
        ("atm", "EX-1.1"),
    ])
    def test_type_specific_exhibit_rescue(self, typ, doc):
        assert _keep_create_type(
            typ, filing_text="", item_codes=set(),
            doc_types=[doc], open_types=set()) is True

    @pytest.mark.parametrize("typ",
                             ["warrant", "preferred", "convertible", "atm"])
    def test_cross_type_ex10_rescues_all(self, typ):
        # EX-10 (an SPA can issue any type) keeps every one of the four.
        assert _keep_create_type(
            typ, filing_text="", item_codes=set(),
            doc_types=["EX-10.1"], open_types=set()) is True

    def test_ex40_does_not_trigger_warrant(self):
        # Prefix 'EX-4.' includes the trailing dot, so 'EX-40.1' must NOT
        # match the warrant exhibit rescue (boundary guard).
        assert _keep_create_type(
            "warrant", filing_text="", item_codes=set(),
            doc_types=["EX-40.1"], open_types=set()) is False

    def test_preferred_not_rescued_by_ex4(self):
        # EX-4 is warrant/convertible territory; preferred wants EX-3.
        assert _keep_create_type(
            "preferred", filing_text="", item_codes=set(),
            doc_types=["EX-4.1"], open_types=set()) is False

    def test_atm_not_rescued_by_ex4(self):
        assert _keep_create_type(
            "atm", filing_text="", item_codes=set(),
            doc_types=["EX-4.1"], open_types=set()) is False

    @pytest.mark.parametrize("typ",
                             ["warrant", "preferred", "convertible", "atm"])
    def test_ex100_does_not_trigger_cross_rescue(self, typ):
        # Cross-type prefix 'EX-10.' includes the trailing dot, so 'EX-100.1'
        # must NOT match it for any type (boundary guard mirroring EX-40).
        assert _keep_create_type(
            typ, filing_text="", item_codes=set(),
            doc_types=["EX-100.1"], open_types=set()) is False

    def test_ex30_does_not_trigger_preferred(self):
        # 'EX-3.' prefix includes the dot, so 'EX-30.1' must NOT rescue
        # preferred.
        assert _keep_create_type(
            "preferred", filing_text="", item_codes=set(),
            doc_types=["EX-30.1"], open_types=set()) is False


class TestKeepCreateTypeOpenTypes:
    def test_open_type_keeps_regardless_of_other_signals(self):
        assert _keep_create_type(
            "atm", filing_text="", item_codes=set(),
            doc_types=[], open_types={"atm", "warrant"}) is True

    def test_not_in_open_types_does_not_keep(self):
        assert _keep_create_type(
            "convertible", filing_text="", item_codes=set(),
            doc_types=[], open_types={"atm", "warrant"}) is False


# ════════════════════════════════════════════════════════════════════════
# prune_create_tools  (db_backed)
# ════════════════════════════════════════════════════════════════════════
_FOUR_CREATES = ["create_warrant", "create_preferred",
                 "create_convertible", "create_atm"]
_NON_PRUNABLE = ["create_equity", "create_equity_line", "amend_atm",
                 "close_instrument", "note_no_event", "apply_split"]


def _all_tools():
    return [_stub_tool(n) for n in (_FOUR_CREATES + _NON_PRUNABLE)]


class TestPruneCreateToolsFormGate:
    @pytest.mark.parametrize("form", ["10-K", "S-1", "S-3", "F-3", "10-Q"])
    def test_non_prune_forms_return_input_unchanged(self, form, temp_db):
        tools = _all_tools()
        out = prune_create_tools(
            tools, form=form, accession="ACC-X", items=None,
            filing_text="", active_rows=None)
        # No DB hit, no drop: SAME object returned.
        assert out is tools

    @pytest.mark.parametrize("form", [None, ""])
    def test_form_none_or_empty_unchanged(self, form, temp_db):
        tools = _all_tools()
        out = prune_create_tools(
            tools, form=form, accession="ACC-X", items=None,
            filing_text="", active_rows=None)
        assert out is tools

    def test_amendment_suffix_is_prunable(self, temp_db):
        # '8-K/A' → form_base '8-K' (split on '/') → prunable. With all-empty
        # signals every create is dropped, so the result differs from input.
        tools = _all_tools()
        out = prune_create_tools(
            tools, form="8-K/A", accession="ACC-EMPTY", items=None,
            filing_text="", active_rows=None)
        names = {t.name for t in out}
        assert names.isdisjoint(_FOUR_CREATES)

    def test_424b5_amendment_suffix_prunable(self, temp_db):
        tools = _all_tools()
        out = prune_create_tools(
            tools, form="424B5/A", accession="ACC-EMPTY", items=None,
            filing_text="", active_rows=None)
        assert {t.name for t in out}.isdisjoint(_FOUR_CREATES)

    def test_lowercase_form_upper_matches(self, temp_db):
        tools = _all_tools()
        out = prune_create_tools(
            tools, form="8-k", accession="ACC-EMPTY", items=None,
            filing_text="", active_rows=None)
        assert {t.name for t in out}.isdisjoint(_FOUR_CREATES)

    @pytest.mark.parametrize(
        "form",
        ["424B2", "424B3", "424B4", "424B5", "424B8", "FWP", "SUPPL", "425"],
    )
    def test_all_event_prune_forms_are_gated(self, form, temp_db):
        # Every non-8-K member of _PRUNE_FORMS prunes too. With all-empty
        # signals the four creates drop, observably differing from input.
        tools = _all_tools()
        out = prune_create_tools(
            tools, form=form, accession="ACC-EMPTY", items=None,
            filing_text="", active_rows=None)
        assert {t.name for t in out}.isdisjoint(_FOUR_CREATES)

    def test_non_prune_form_keeps_creates_despite_signals(self, temp_db):
        # A periodic form is never pruned even when the filing text names every
        # instrument — the full tool set is returned (identity), and no DB read
        # happens (no dilution_raw staged).
        tools = _all_tools()
        out = prune_create_tools(
            tools, form="10-K", accession="ACC-NONE", items="5.03,2.03",
            filing_text="warrant convertible preferred sales agreement",
            active_rows=[{"type": "atm"}])
        assert out is tools

    def test_non_prune_form_does_not_touch_db(self, temp_db, monkeypatch):
        # Prove the form gate returns BEFORE any get_conn() call: patch the
        # module's get_conn to explode, then confirm a non-prune form still
        # returns the input unchanged without raising. (The prune path would
        # invoke get_conn and surface the boom.)
        def _boom(*a, **k):
            raise AssertionError("get_conn must not be called for non-prune "
                                 "forms")
        monkeypatch.setattr(ic, "get_conn", _boom)
        tools = _all_tools()
        out = prune_create_tools(
            tools, form="S-1", accession="ACC-NONE", items="3.02",
            filing_text="warrant", active_rows=[{"type": "atm"}])
        assert out is tools

    def test_prune_form_does_touch_db(self, temp_db, monkeypatch):
        # Mirror assertion: a prunable form DOES reach get_conn (the boom
        # fires), confirming the previous test is not vacuously green.
        def _boom(*a, **k):
            raise RuntimeError("queried")
        monkeypatch.setattr(ic, "get_conn", _boom)
        with pytest.raises(RuntimeError, match="queried"):
            prune_create_tools(
                _all_tools(), form="8-K", accession="ACC-NONE", items=None,
                filing_text="", active_rows=None)


class TestPruneCreateToolsDropping:
    def test_all_signals_empty_drops_all_four_keeps_rest(self, temp_db):
        tools = _all_tools()
        out = prune_create_tools(
            tools, form="8-K", accession="ACC-NONE", items=None,
            filing_text="", active_rows=None)
        names = [t.name for t in out]
        # All four creates gone; every non-prunable tool survives in order.
        assert names == _NON_PRUNABLE

    def test_partial_drop_returns_new_list(self, temp_db):
        tools = _all_tools()
        # filing_text mentions only 'warrant' → warrant kept, other 3 dropped.
        out = prune_create_tools(
            tools, form="8-K", accession="ACC-NONE", items=None,
            filing_text="issued a warrant to the holder", active_rows=None)
        assert out is not tools
        names = [t.name for t in out]
        assert "create_warrant" in names
        assert "create_preferred" not in names
        assert "create_convertible" not in names
        assert "create_atm" not in names
        # non-prunable tools preserved.
        for n in _NON_PRUNABLE:
            assert n in names

    def test_all_kept_returns_same_object(self, temp_db):
        # Mentioning all four instrument nouns keeps every create → no drop →
        # identical input object returned (early return path).
        tools = _all_tools()
        text = ("issued a warrant, convertible note, and preferred stock "
                "under an at-the-market sales agreement")
        out = prune_create_tools(
            tools, form="8-K", accession="ACC-NONE", items=None,
            filing_text=text, active_rows=None)
        assert out is tools

    def test_ex10_doc_rescues_all_nothing_dropped(self, temp_db):
        _add_raw(temp_db, "ACC-EX10", 11, "EX-10.1")
        tools = _all_tools()
        out = prune_create_tools(
            tools, form="8-K", accession="ACC-EX10", items=None,
            filing_text="", active_rows=None)
        # EX-10 cross-rescues all four → no drop → same object.
        assert out is tools

    def test_ex4_doc_keeps_warrant_and_convertible_only(self, temp_db):
        _add_raw(temp_db, "ACC-EX4ONLY", 11, "EX-4.1")
        tools = _all_tools()
        out = prune_create_tools(
            tools, form="8-K", accession="ACC-EX4ONLY", items=None,
            filing_text="", active_rows=None)
        names = [t.name for t in out]
        assert "create_warrant" in names
        assert "create_convertible" in names
        assert "create_preferred" not in names
        assert "create_atm" not in names

    def test_items_rescue_keeps_preferred(self, temp_db):
        # items='5.03' rescues preferred; nothing else fires → other 3 drop.
        tools = _all_tools()
        out = prune_create_tools(
            tools, form="8-K", accession="ACC-NONE", items="5.03",
            filing_text="", active_rows=None)
        names = [t.name for t in out]
        assert "create_preferred" in names
        assert "create_warrant" not in names
        assert "create_convertible" not in names
        assert "create_atm" not in names

    def test_active_rows_open_type_keeps_atm(self, temp_db):
        tools = _all_tools()
        out = prune_create_tools(
            tools, form="8-K", accession="ACC-NONE", items=None,
            filing_text="", active_rows=[{"type": "atm"}])
        names = [t.name for t in out]
        assert "create_atm" in names
        assert "create_warrant" not in names

    def test_active_rows_row_with_none_type_no_crash(self, temp_db):
        tools = _all_tools()
        out = prune_create_tools(
            tools, form="8-K", accession="ACC-NONE", items=None,
            filing_text="", active_rows=[{"type": None}])
        # type None → '' open type, keeps nothing extra → all four dropped.
        assert {t.name for t in out} == set(_NON_PRUNABLE)

    def test_active_rows_none_treated_as_empty(self, temp_db):
        tools = _all_tools()
        out = prune_create_tools(
            tools, form="8-K", accession="ACC-NONE", items=None,
            filing_text="", active_rows=None)
        assert {t.name for t in out} == set(_NON_PRUNABLE)

    def test_db_rows_for_other_accession_not_matched(self, temp_db):
        # A doc exists under a DIFFERENT accession; the queried one has none →
        # doc_types empty → all four dropped (no rescue from foreign rows).
        _add_raw(temp_db, "OTHER-ACC", 11, "EX-10.1")
        tools = _all_tools()
        out = prune_create_tools(
            tools, form="8-K", accession="ACC-NONE", items=None,
            filing_text="", active_rows=None)
        assert {t.name for t in out} == set(_NON_PRUNABLE)

    def test_only_named_creates_are_droppable(self, temp_db):
        # create_equity / create_equity_line look like creates but are never
        # in _CREATE_TOOL_BY_TYPE, so they survive an all-empty prune.
        tools = _all_tools()
        out = prune_create_tools(
            tools, form="8-K", accession="ACC-NONE", items=None,
            filing_text="", active_rows=None)
        names = [t.name for t in out]
        assert "create_equity" in names
        assert "create_equity_line" in names

    def test_items_2_03_rescues_convertible_only(self, temp_db):
        # items='2.03' rescues convertible via _CREATE_ITEM_RESCUE; the other
        # three creates drop (no other signal).
        tools = _all_tools()
        out = prune_create_tools(
            tools, form="8-K", accession="ACC-NONE", items="2.03",
            filing_text="", active_rows=None)
        names = [t.name for t in out]
        assert "create_convertible" in names
        assert "create_warrant" not in names
        assert "create_preferred" not in names
        assert "create_atm" not in names

    def test_items_whitespace_and_dups_parsed(self, temp_db):
        # prune parses items with the same strip/dedup as classify_8k, so
        # ' 5.03 ,5.03' rescues preferred.
        tools = _all_tools()
        out = prune_create_tools(
            tools, form="8-K", accession="ACC-NONE", items=" 5.03 ,5.03",
            filing_text="", active_rows=None)
        names = [t.name for t in out]
        assert "create_preferred" in names
        assert "create_warrant" not in names

    def test_ex3_doc_keeps_preferred_only(self, temp_db):
        # EX-3.1 (Cert of Designation) rescues preferred only; warrant/
        # convertible (EX-4) and atm (EX-1) drop.
        _add_raw(temp_db, "ACC-EX3ONLY", 11, "EX-3.1")
        tools = _all_tools()
        out = prune_create_tools(
            tools, form="8-K", accession="ACC-EX3ONLY", items=None,
            filing_text="", active_rows=None)
        names = [t.name for t in out]
        assert "create_preferred" in names
        assert "create_warrant" not in names
        assert "create_convertible" not in names
        assert "create_atm" not in names

    def test_ex100_doc_does_not_rescue_any(self, temp_db):
        # EX-100.1 must not match the EX-10. cross-rescue prefix (boundary) →
        # all four create tools drop just as with an empty doc set.
        _add_raw(temp_db, "ACC-EX100", 11, "EX-100.1")
        tools = _all_tools()
        out = prune_create_tools(
            tools, form="8-K", accession="ACC-EX100", items=None,
            filing_text="", active_rows=None)
        assert {t.name for t in out} == set(_NON_PRUNABLE)

    def test_kept_order_preserved(self, temp_db):
        # Order of surviving tools matches the original list order.
        tools = _all_tools()
        out = prune_create_tools(
            tools, form="8-K", accession="ACC-NONE", items=None,
            filing_text="issued a warrant", active_rows=None)
        names = [t.name for t in out]
        # original order had create_warrant first, then the non-prunables.
        assert names == ["create_warrant"] + _NON_PRUNABLE

    def test_drop_logs_sorted_dropped_names(self, temp_db, caplog):
        # On a partial drop the module logs an INFO line naming the accession,
        # the drop count, and the dropped tools in SORTED order. Asserts the
        # observable side effect, not just the return value.
        tools = _all_tools()
        with caplog.at_level("INFO", logger="dilution.ledger.item_classification"):
            prune_create_tools(
                tools, form="8-K", accession="ACC-LOG", items=None,
                filing_text="issued a warrant", active_rows=None)
        msgs = [r.getMessage() for r in caplog.records]
        assert any(
            "ACC-LOG" in m
            and "pruned 3 create tool(s)" in m
            and "create_atm,create_convertible,create_preferred" in m
            for m in msgs
        ), msgs

    def test_noop_drop_emits_no_log(self, temp_db, caplog):
        # When nothing is dropped (all four kept) the early return happens
        # BEFORE the log.info call → no pruning log line is emitted.
        tools = _all_tools()
        text = ("issued a warrant, convertible note, and preferred stock "
                "under an at-the-market sales agreement")
        with caplog.at_level("INFO", logger="dilution.ledger.item_classification"):
            out = prune_create_tools(
                tools, form="8-K", accession="ACC-NOLOG", items=None,
                filing_text=text, active_rows=None)
        assert out is tools
        assert not any("pruned" in r.getMessage() for r in caplog.records)


# ════════════════════════════════════════════════════════════════════════
# expected_call_classes  (pure: content-side detection)
# ════════════════════════════════════════════════════════════════════════
class TestExpectedCallClassesFormGate:
    @pytest.mark.parametrize("form", ["S-1", "424B5", "10-Q", "10-K", "S-3"])
    def test_non_event_forms_return_empty(self, form):
        text = "agreed to issue warrants to purchase up to 1,500,000 shares"
        assert expected_call_classes(text, form) == {}

    def test_8k_amendment_split_to_8k(self):
        text = "agreed to issue warrants to purchase up to 1,500,000 shares"
        assert "warrant" in expected_call_classes(text, "8-K/A")

    def test_6k_is_event_form(self):
        text = "agreed to issue warrants to purchase up to 1,500,000 shares"
        assert "warrant" in expected_call_classes(text, "6-K")

    def test_empty_text_returns_empty(self):
        assert expected_call_classes("", "8-K") == {}

    def test_none_text_returns_empty(self):
        assert expected_call_classes(None, "8-K") == {}


class TestExpectedCallClassesWarrant:
    def test_valid_warrant_issuance(self):
        text = ("The company agreed to issue warrants to purchase up to "
                "1,500,000 shares of common stock.")
        out = expected_call_classes(text, "8-K")
        assert out == {"warrant": "warrants to purchase up to 1,500,000"}

    def test_two_digit_count_does_not_match(self):
        # count regex requires >=3 digits ('[\\d,]{3,}'); '50' is too short.
        text = "agreed to issue warrants to purchase 50 shares"
        assert expected_call_classes(text, "8-K") == {}

    def test_three_digit_count_matches(self):
        text = "agreed to issue warrants to purchase up to 750 shares"
        out = expected_call_classes(text, "8-K")
        assert out.get("warrant") == "warrants to purchase up to 750"

    @pytest.mark.parametrize("count,matches", [
        ("9", False),     # 1 char
        ("99", False),    # 2 chars
        ("12", False),    # 2 chars
        ("100", True),    # 3 chars
        ("999", True),    # 3 chars
        # `[\d,]{3,}` counts CHARACTERS in the [digit-or-comma] class, not
        # digits: a comma-bearing 2-digit value spans 3 chars and matches.
        # Pins the (slightly surprising) char-count vs digit-count semantics.
        ("1,2", True),
    ])
    def test_count_threshold_is_three_chars_not_digits(self, count, matches):
        text = f"agreed to issue warrants to purchase up to {count} shares"
        out = expected_call_classes(text, "8-K")
        assert ("warrant" in out) is matches
        if matches:
            assert out["warrant"] == f"warrants to purchase up to {count}"

    def test_no_issuance_verb_excluded(self):
        # A warrant phrase with no issuance verb in the +-240 window is gated.
        text = ("The warrants to purchase 1,000,000 shares remain "
                "exercisable for five years.")
        assert expected_call_classes(text, "8-K") == {}

    def test_recap_window_excluded(self):
        text = "previously issued warrants to purchase 1,000,000 shares"
        assert expected_call_classes(text, "8-K") == {}

    def test_currently_outstanding_recap_excluded(self):
        text = ("issued warrants to purchase 1,000,000 shares that are "
                "currently outstanding")
        assert expected_call_classes(text, "8-K") == {}

    def test_recap_first_real_later_captures_real(self):
        # First warrant occurrence is a recap (excluded), a second occurrence
        # far enough away is a real issuance → the loop continues past the
        # recap and captures the real match (it does NOT break on the recap).
        pad = " filler text. " * 50  # ~700 chars, keeps windows disjoint
        text = ("previously issued warrants to purchase 1,000,000 shares."
                + pad +
                "the company agreed to issue warrants to purchase up to "
                "2,000,000 shares.")
        out = expected_call_classes(text, "8-K")
        assert out.get("warrant") == "warrants to purchase up to 2,000,000"

    def test_first_valid_match_only(self):
        # Two valid warrant disclosures → exactly one key, the first snippet.
        text = ("agreed to issue warrants to purchase up to 1,111,000 shares. "
                "Also agreed to issue warrants to purchase up to 2,222,000 "
                "shares.")
        out = expected_call_classes(text, "8-K")
        assert list(out.keys()) == ["warrant"]
        assert out["warrant"] == "warrants to purchase up to 1,111,000"

    def test_noverb_first_real_later_captures_real(self):
        # Companion to test_recap_first_real_later_captures_real but for the
        # OTHER continue branch: the first warrant occurrence has NO issuance
        # verb in its ±240 window (continue on the verb gate), a later
        # occurrence DOES → the loop must continue, not break, and capture the
        # real one. Proves the verb-gate `continue` (not just the recap one)
        # advances finditer rather than abandoning the class.
        pad = " filler text. " * 50  # ~700 chars, keeps windows disjoint
        text = ("the warrants to purchase 1,000,000 shares remain exercisable."
                + pad +
                "the company agreed to issue warrants to purchase up to "
                "2,000,000 shares.")
        out = expected_call_classes(text, "8-K")
        assert out.get("warrant") == "warrants to purchase up to 2,000,000"


class TestExpectedCallClassesConvertible:
    def test_branch_one_convertible_notes_principal(self):
        text = ("The company issued convertible notes in the aggregate "
                "principal amount of $5,000,000.")
        out = expected_call_classes(text, "8-K")
        assert out.get("convertible") == (
            "convertible notes in the aggregate principal amount of $5,000,000")

    def test_branch_two_principal_then_promissory_note(self):
        # alternation branch: 'aggregate principal amount of $X ... note'
        # (within 120 chars), promissory.
        text = ("issued a note in the aggregate principal amount of "
                "$2,000,000 under a promissory note")
        out = expected_call_classes(text, "8-K")
        assert out.get("convertible") == (
            "aggregate principal amount of $2,000,000 under a promissory note")

    def test_convertible_pattern_matches_but_no_verb_excluded(self):
        # The convertible pattern matches, but with NO issuance verb in the
        # ±240 window the non-close verb gate drops it. Proves the verb gate
        # applies to convertible, not just warrant.
        text = ("convertible notes in the aggregate principal amount of "
                "$5,000,000 that will mature in 2030.")
        assert expected_call_classes(text, "8-K") == {}


class TestExpectedCallClassesPreferred:
    def test_series_preferred_with_purchase_agreement(self):
        text = ("issued 1,000 shares of Series A Convertible Preferred Stock "
                "pursuant to a purchase agreement")
        out = expected_call_classes(text, "8-K")
        assert "preferred" in out
        assert out["preferred"].startswith(
            "shares of Series A Convertible Preferred Stock")

    def test_bare_series_preferred_without_tail_excluded(self):
        # No purchase/exchange/stated-value tail → no match.
        text = "issued shares of Series A Preferred Stock to investors"
        assert expected_call_classes(text, "8-K") == {}

    def test_preferred_pattern_matches_but_no_verb_excluded(self):
        # Pattern matches (Series A ... purchase agreement) but no issuance
        # verb in the window → verb gate excludes it. Proves the verb gate is
        # enforced for preferred too.
        text = ("shares of Series A Convertible Preferred Stock under a "
                "purchase agreement that matured early.")
        assert expected_call_classes(text, "8-K") == {}

    def test_preferred_exchange_agreement_tail(self):
        # The 'exchange agreement' alternative in the preferred tail matches.
        text = ("issued shares of newly designated Series B Preferred Stock "
                "pursuant to an exchange agreement with the holder")
        out = expected_call_classes(text, "8-K")
        assert "preferred" in out
        assert out["preferred"].startswith(
            "shares of newly designated Series B Preferred Stock")

    def test_preferred_stated_value_tail(self):
        # The 'stated value' alternative in the preferred tail matches.
        text = ("issued shares of Series C Preferred Stock with a stated "
                "value of $1,000 per share")
        out = expected_call_classes(text, "8-K")
        assert "preferred" in out


class TestExpectedCallClassesAtmAndEquityLine:
    def test_atm_open_market_sale_agreement(self):
        text = ("The company entered into an Open Market Sale Agreement "
                "providing for sales of up to $50,000,000.")
        out = expected_call_classes(text, "8-K")
        assert "atm" in out
        assert out["atm"].startswith("entered into an Open Market Sale "
                                     "Agreement")

    def test_equity_line_standby_equity_purchase_agreement(self):
        text = ("The company entered into a Standby Equity Purchase Agreement "
                "for up to $25,000,000.")
        out = expected_call_classes(text, "8-K")
        assert "equity_line" in out
        assert out["equity_line"].startswith(
            "entered into a Standby Equity Purchase Agreement")


class TestExpectedCallClassesClose:
    def test_close_does_not_require_issuance_verb(self):
        # 'close' class skips the issuance-verb requirement. (Use a target
        # name with NO dots — '[^.]{0,80}?' cannot span a period, so
        # 'H.C. Wainwright' would break the match.)
        text = "The company terminated the Wainwright Sales Agreement today."
        out = expected_call_classes(text, "8-K")
        assert out.get("close") == "terminated the Wainwright Sales Agreement"

    def test_close_at_the_market_target(self):
        text = "terminated the at-the-market offering program"
        out = expected_call_classes(text, "8-K")
        assert out.get("close") == "terminated the at-the-market"

    @pytest.mark.parametrize("phrase,expected", [
        ("terminates the Sales Agreement now",
         "terminates the Sales Agreement"),
        ("the termination of the equity distribution agreement was announced",
         "termination of the equity distribution agreement"),
        ("terminated the purchase agreement with the investor",
         "terminated the purchase agreement"),
        ("terminated the sale agreement effective immediately",
         "terminated the sale agreement"),
    ])
    def test_close_verb_and_target_variants(self, phrase, expected):
        # 'terminat(?:ed|es|ion of)' across all three verb inflections, and
        # each agreement-shaped target alternative. No issuance verb is needed
        # for the close class.
        out = expected_call_classes(phrase, "8-K")
        assert out.get("close") == expected

    def test_close_still_subject_to_recap(self):
        # close skips the verb gate but NOT the recap gate.
        text = ("previously disclosed and terminated the Sales Agreement "
                "with the agent.")
        assert expected_call_classes(text, "8-K") == {}

    def test_close_boilerplate_warrant_not_matched(self):
        # 'termination of this Warrant' is exhibit boilerplate, not an event:
        # the close pattern only targets agreement-shaped names.
        text = ("Upon termination of this Warrant, the holder shall have no "
                "further rights.")
        assert expected_call_classes(text, "8-K") == {}

    def test_close_target_with_dots_in_name_not_matched(self):
        # Documents the dot-sensitivity: '[^.]{0,80}?' cannot cross a period,
        # so a dotted agent name ('H.C. Wainwright') blocks the close match.
        text = ("The company terminated the H.C. Wainwright Sales Agreement "
                "effective today.")
        assert expected_call_classes(text, "8-K") == {}


class TestExpectedCallClassesMultiAndNormalization:
    def test_multiple_classes_all_present(self):
        text = ("The company agreed to issue warrants to purchase up to "
                "1,500,000 shares. It also issued convertible notes in the "
                "aggregate principal amount of $5,000,000.")
        out = expected_call_classes(text, "8-K")
        assert set(out.keys()) == {"warrant", "convertible"}

    def test_snippet_whitespace_collapsed(self):
        # multi-space / newline / tab inside the matched span collapses to
        # single spaces (the atm lazy span [^.]{0,120}? captures them).
        text = ("The company entered into an   Open Market Sale Agreement "
                "providing\nfor\tsales of up to $50,000,000.")
        out = expected_call_classes(text, "8-K")
        assert out.get("atm") == (
            "entered into an Open Market Sale Agreement providing for sales "
            "of up to")

    def test_snippet_truncated_to_200_chars(self):
        # A very long matched count is truncated to 200 chars.
        digits = "1" + ",123" * 80
        text = "agreed to issue warrants to purchase up to " + digits + " shares"
        out = expected_call_classes(text, "8-K")
        assert len(out["warrant"]) == 200

    def test_window_boundary_match_near_start_no_raise(self):
        # match at the very start: lo clamps to 0, must not raise. There is
        # no issuance verb here so the result is {} but the call is safe.
        text = "warrants to purchase up to 1,500,000 shares"
        assert expected_call_classes(text, "8-K") == {}

    def test_window_boundary_match_near_end_no_raise(self):
        # match at the very end: hi clamps to len, must not raise.
        text = "the company agreed to issue warrants to purchase up to 1,500,000"
        out = expected_call_classes(text, "8-K")
        assert out.get("warrant") == "warrants to purchase up to 1,500,000"


class TestExpectedCallClassesIssuanceVerbs:
    @pytest.mark.parametrize("verb_phrase", [
        "issued", "issuance of", "agreed to issue", "will issue",
        "sold", "sale of", "entered into the agreement and granted",
        "closing of",
    ])
    def test_issuance_verbs_satisfy_warrant_gate(self, verb_phrase):
        # Each issuance verb in the window unlocks the warrant class. Place
        # the verb phrase before the warrant count phrase, same window.
        text = (f"The company {verb_phrase} warrants to purchase up to "
                f"1,250,000 shares.")
        out = expected_call_classes(text, "8-K")
        assert out.get("warrant") == "warrants to purchase up to 1,250,000"


# ════════════════════════════════════════════════════════════════════════
# Item8KVerdict  (frozen dataclass smoke test)
# ════════════════════════════════════════════════════════════════════════
class TestItem8KVerdict:
    def test_equality_and_fields(self):
        a = Item8KVerdict(skip=True, must_record=False, reason="x")
        b = Item8KVerdict(skip=True, must_record=False, reason="x")
        assert a == b
        assert a.skip is True
        assert a.must_record is False
        assert a.reason == "x"

    def test_frozen_is_immutable(self):
        import dataclasses
        v = Item8KVerdict(skip=False, must_record=False, reason="")
        with pytest.raises(dataclasses.FrozenInstanceError):
            v.skip = True
