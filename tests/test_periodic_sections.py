"""Unit tests for :mod:`dilution.periodic_sections`.

This module is pure logic — no DB, no network, no LLM, no config-at-import.
It only imports ``logging`` and ``typing``. The DB scaffolding from
``conftest.py`` (the autouse ``temp_db`` fixture) is irrelevant here and is
simply ignored; we never reference it.

``select_text`` reads form / declared_items from a ``filing`` object that the
caller passes in. edgartools is never imported in this file, so there is no
import-time seam to monkeypatch; instead we hand-roll tiny fake filing / obj /
section objects. The key subtlety the fakes must respect:

  * ``sec.text()`` is a *callable method* (invoked as ``sec.text()``).
  * ``confidence`` / ``part`` / ``item`` / ``title`` are *plain attributes*,
    read via ``getattr``. The confidence default is 1.0 when the attribute is
    entirely absent, but ``0.0`` (via ``... or 0.0``) when present-but-None —
    those two cases diverge and both are tested.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest

from dilution.periodic_sections import (
    KEEP_SECTIONS,
    MIN_KEPT_CHARS,
    MIN_SECTION_CONFIDENCE,
    _MIN_KEPT_CHARS_BY_FORM,
    _declared_keep_keys,
    _normalize_form,
    is_periodic_with_sections,
    select_text,
)


# ── fake filing scaffolding ────────────────────────────────────────────────
class FakeSection:
    """A stand-in for an edgartools Section.

    ``text`` is a callable; ``confidence`` / ``part`` / ``item`` / ``title``
    are plain attributes. Pass ``confidence=_ABSENT`` to omit the attribute
    entirely (exercising the getattr default of 1.0). Any attribute passed as
    ``None`` is set to None (which is a *different* code path from absent for
    confidence).
    """

    _ABSENT = object()

    def __init__(self, text, confidence=0.95, *,
                 part=_ABSENT, item=_ABSENT, title=_ABSENT):
        self._text = text
        if confidence is not FakeSection._ABSENT:
            self.confidence = confidence
        if part is not FakeSection._ABSENT:
            self.part = part
        if item is not FakeSection._ABSENT:
            self.item = item
        if title is not FakeSection._ABSENT:
            self.title = title

    def text(self):
        return self._text


class RaisingTextSection:
    """Section whose ``.text()`` raises — exercises the sec.text() guard."""

    confidence = 0.95

    def __init__(self, exc):
        self._exc = exc

    def text(self):
        raise self._exc


class FakeObj:
    def __init__(self, sections):
        self.sections = sections


class FakeObjNoSections:
    """An obj without a ``sections`` attribute at all (getattr -> None)."""


class FakeFiling:
    def __init__(self, obj):
        self._obj = obj

    def obj(self):
        return self._obj


class RaisingObjFiling:
    def __init__(self, exc):
        self._exc = exc

    def obj(self):
        raise self._exc


class NoneObjFiling:
    def obj(self):
        return None


class BadSectionsMapping:
    """Truthy mapping whose ``.items()`` raises."""

    def __bool__(self):
        return True

    def items(self):
        raise TypeError("items blew up")


def _filing_with(sections):
    return FakeFiling(FakeObj(sections))


# ── _normalize_form ─────────────────────────────────────────────────────────
class TestNormalizeForm:
    @pytest.mark.parametrize("raw, expected", [
        ("10-K", "10-K"),
        ("10-Q", "10-Q"),
        ("20-F", "20-F"),
        ("8-K", "8-K"),
        ("10-k", "10-K"),               # case-insensitive via .upper()
        ("8-k", "8-K"),
        ("10-K/A", "10-K"),             # amendment suffix stripped
        ("8-K/A", "8-K"),
        ("  10-Q  ", "10-Q"),           # surrounding whitespace
        ("10-K/A/foo", "10-K"),         # only [0] of the split is used
    ])
    def test_supported_forms_normalize(self, raw, expected):
        assert _normalize_form(raw) == expected

    @pytest.mark.parametrize("raw", ["40-F", "6-K", "S-3", "424B5", "1-K", "F-1"])
    def test_unsupported_forms_return_none(self, raw):
        assert _normalize_form(raw) is None

    def test_empty_string_returns_none(self):
        assert _normalize_form("") is None

    def test_none_returns_none(self):
        # str annotation, but the `if not form` guard handles a falsy None.
        assert _normalize_form(None) is None

    def test_returns_canonical_keep_key(self):
        # The returned value must be a literal key of KEEP_SECTIONS.
        assert _normalize_form("10-k/a") in KEEP_SECTIONS


# ── _declared_keep_keys ──────────────────────────────────────────────────────
class TestDeclaredKeepKeys:
    KEEP_8K = KEEP_SECTIONS["8-K"]

    @pytest.mark.parametrize("declared", [None, "", [], (), set()])
    def test_falsy_input_returns_empty_set(self, declared):
        assert _declared_keep_keys(declared, self.KEEP_8K) == set()

    def test_comma_string_drops_item_901(self):
        # 9.01 is the exhibit index and is explicitly excluded.
        assert _declared_keep_keys("1.01,3.02,9.01", self.KEEP_8K) == {
            "item_101", "item_302",
        }

    def test_iterable_of_strings(self):
        assert _declared_keep_keys(["1.01", "3.02"], self.KEEP_8K) == {
            "item_101", "item_302",
        }

    def test_iterable_of_non_strings_coerced_via_str(self):
        # str(101) == "101"; no dot to remove -> "item_101".
        assert _declared_keep_keys([101, 302], self.KEEP_8K) == {
            "item_101", "item_302",
        }

    def test_whitespace_after_comma_is_stripped(self):
        assert _declared_keep_keys("1.01, 3.02", self.KEEP_8K) == {
            "item_101", "item_302",
        }

    def test_item_901_explicitly_skipped_when_alone(self):
        assert _declared_keep_keys("9.01", self.KEEP_8K) == set()

    def test_trailing_comma_blank_code_skipped(self):
        # "1.01," -> ["1.01", ""]; the blank is dropped by `if not code`.
        assert _declared_keep_keys("1.01,", self.KEEP_8K) == {"item_101"}

    def test_code_not_in_keep_set_excluded(self):
        # 2.02 (earnings) is not in the 8-K keep-list -> intersection drops it.
        assert _declared_keep_keys("2.02", self.KEEP_8K) == set()

    @pytest.mark.parametrize("code, key", [
        ("1.01", "item_101"),
        ("5.03", "item_503"),
        ("3.02", "item_302"),
        ("8.01", "item_801"),
    ])
    def test_dot_removal_mapping(self, code, key):
        assert _declared_keep_keys(code, self.KEEP_8K) == {key}

    def test_duplicate_codes_dedup(self):
        assert _declared_keep_keys("1.01,1.01", self.KEEP_8K) == {"item_101"}

    def test_empty_keep_set_yields_empty_regardless(self):
        assert _declared_keep_keys("1.01,3.02", set()) == set()

    def test_result_is_a_set(self):
        assert isinstance(_declared_keep_keys("1.01", self.KEEP_8K), set)

    def test_int_901_is_not_excluded(self):
        # The 9.01 skip is an exact STRING compare (`code == "9.01"`). An
        # integer 901 stringifies to "901" (no dot), maps to "item_901"
        # (which IS in the 8-K keep-list), and is therefore NOT skipped —
        # diverging from the dotted-string "9.01" case below.
        assert _declared_keep_keys([901], self.KEEP_8K) == {"item_901"}

    def test_float_901_is_excluded_via_str_coercion(self):
        # str(9.01) == "9.01" which exactly matches the skip literal, so the
        # float form behaves like the dotted string and IS dropped.
        assert _declared_keep_keys([9.01], self.KEEP_8K) == set()

    def test_string_901_excluded_but_int_901_kept_diverge(self):
        # Pin the divergence explicitly: same numeric item, two encodings.
        assert _declared_keep_keys("9.01", self.KEEP_8K) == set()
        assert _declared_keep_keys([901], self.KEEP_8K) == {"item_901"}

    def test_int_101_maps_like_dotted_string(self):
        # str(101).replace('.', '') == "101" -> "item_101" (in keep).
        assert _declared_keep_keys([101], self.KEEP_8K) == {"item_101"}

    def test_mixed_string_iterable_intersects_keep(self):
        # 1.01 + 5.02 (officer changes, NOT in keep) + 3.02 -> only the
        # two keep-list members survive the intersection.
        assert _declared_keep_keys(["1.01", "5.02", "3.02"], self.KEEP_8K) == {
            "item_101", "item_302",
        }


# ── is_periodic_with_sections ────────────────────────────────────────────────
class TestIsPeriodicWithSections:
    @pytest.mark.parametrize("form", ["10-K", "10-K/A", "10-Q", "20-F",
                                      "8-K", "8-k", "  10-Q  "])
    def test_keep_list_forms_true(self, form):
        assert is_periodic_with_sections(form) is True

    @pytest.mark.parametrize("form", ["40-F", "6-K", "424B5", "S-3", "1-K",
                                      "", None])
    def test_non_keep_forms_false(self, form):
        assert is_periodic_with_sections(form) is False

    def test_returns_bool_not_truthy_object(self):
        # Thin wrapper must coerce to a real bool, not a key string.
        assert isinstance(is_periodic_with_sections("10-K"), bool)
        assert isinstance(is_periodic_with_sections("40-F"), bool)


# ── module constants ──────────────────────────────────────────────────────────
class TestConstants:
    def test_min_kept_chars_value(self):
        assert MIN_KEPT_CHARS == 5_000

    def test_min_section_confidence_value(self):
        assert MIN_SECTION_CONFIDENCE == 0.7

    def test_per_form_floors(self):
        assert _MIN_KEPT_CHARS_BY_FORM == {
            "10-K": 5_000, "10-Q": 5_000, "20-F": 5_000, "8-K": 300,
        }

    def test_all_keep_forms_have_a_floor(self):
        # The .get(..., MIN_KEPT_CHARS) default is effectively dead because
        # every keep-list form is in the floor dict; assert that invariant.
        assert set(KEEP_SECTIONS) <= set(_MIN_KEPT_CHARS_BY_FORM)

    def test_keep_sections_8k_excludes_dropped_items(self):
        # Sanity: dropped items (earnings, officer changes) are not keepers.
        assert "item_202" not in KEEP_SECTIONS["8-K"]
        assert "item_502" not in KEEP_SECTIONS["8-K"]

    def test_8k_keys_use_item_convention(self):
        # 8-K (CurrentReport) keys are dotless item_<digits>; this is what
        # _declared_keep_keys produces, so a convention mismatch would
        # silently make the declared-items guard never match.
        assert all(k.startswith("item_") for k in KEEP_SECTIONS["8-K"])

    @pytest.mark.parametrize("form", ["10-K", "10-Q", "20-F"])
    def test_periodic_keys_use_part_convention(self, form):
        # Periodic forms use part_<roman>_item_<n> keys.
        assert all(k.startswith("part_") for k in KEEP_SECTIONS[form])

    def test_declared_keep_keys_output_subset_of_8k_keep(self):
        # The full 8-K item code list maps cleanly into the keep-set keys
        # (i.e. the dot-removal convention agrees with the configured keys).
        codes = "1.01,1.02,2.03,2.04,3.02,3.03,5.01,5.03,7.01,8.01"
        out = _declared_keep_keys(codes, KEEP_SECTIONS["8-K"])
        assert out == KEEP_SECTIONS["8-K"] - {"item_901"}


# ── select_text: early guards ────────────────────────────────────────────────
class TestSelectTextGuards:
    def test_form_not_in_keep_list(self):
        text, stats = select_text(object(), "40-F")
        assert text is None
        assert stats == {"reason": "form '40-F' not in keep-list"}

    def test_none_form(self):
        text, stats = select_text(object(), None)
        assert text is None
        assert stats["reason"] == "form None not in keep-list"

    def test_obj_raises(self):
        text, stats = select_text(RaisingObjFiling(ValueError("nope")), "8-K")
        assert text is None
        assert stats == {"reason": "obj() failed: nope"}

    def test_obj_returns_none(self):
        text, stats = select_text(NoneObjFiling(), "8-K")
        assert text is None
        assert stats == {"reason": "obj() returned None"}

    def test_obj_without_sections_attribute(self):
        text, stats = select_text(FakeFiling(FakeObjNoSections()), "8-K")
        assert text is None
        assert stats == {"reason": "no sections parsed"}

    def test_empty_sections(self):
        text, stats = select_text(_filing_with({}), "8-K")
        assert text is None
        assert stats == {"reason": "no sections parsed"}

    def test_falsy_sections(self):
        # sections present but falsy (None) hits the `if not sections` guard.
        text, stats = select_text(FakeFiling(FakeObj(None)), "8-K")
        assert text is None
        assert stats == {"reason": "no sections parsed"}

    def test_sections_items_raises(self):
        text, stats = select_text(
            FakeFiling(FakeObj(BadSectionsMapping())), "8-K")
        assert text is None
        assert stats == {"reason": "sections.items() failed: items blew up"}

    def test_sec_text_raises(self):
        secs = {"item_101": RaisingTextSection(RuntimeError("boom"))}
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is None
        assert stats == {"reason": "sec.text() failed on item_101: boom"}

    def test_sec_text_raises_on_later_section_discards_prior_work(self):
        # A good keep section is processed first, then a later section's
        # text() raises -> the whole filing bails with ONLY a reason key;
        # the partially-collected kept/dropped lists are discarded.
        secs = OrderedDict([
            ("item_101", FakeSection("z" * 350, confidence=0.95)),
            ("item_302", RaisingTextSection(RuntimeError("late boom"))),
        ])
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is None
        assert stats == {"reason": "sec.text() failed on item_302: late boom"}
        assert "kept" not in stats


# ── select_text: 8-K declared-items guard ─────────────────────────────────────
class TestSelectTextDeclaredItemsGuard:
    def test_declared_item_not_parsed_bails(self):
        # Declared 1.01 + 3.02, but only item_302 was parsed -> 1.01 undetected.
        secs = {"item_302": FakeSection("z" * 350)}
        text, stats = select_text(
            _filing_with(secs), "8-K", declared_items="1.01,3.02,9.01")
        assert text is None
        assert stats["reason"] == "declared items not detected: ['item_101']"
        assert stats["parsed"] == ["item_302"]

    def test_undetected_list_is_sorted(self):
        # Declare 8.01 and 1.01; neither parsed -> sorted output.
        secs = {"item_302": FakeSection("z" * 350)}
        text, stats = select_text(
            _filing_with(secs), "8-K", declared_items="8.01,1.01")
        assert text is None
        assert stats["reason"] == (
            "declared items not detected: ['item_101', 'item_801']")

    def test_parsed_list_is_sorted(self):
        secs = OrderedDict([
            ("item_801", FakeSection("a" * 200)),
            ("item_302", FakeSection("b" * 200)),
        ])
        text, stats = select_text(
            _filing_with(secs), "8-K", declared_items="1.01")
        # 1.01 is declared but not parsed -> bail, with parsed keys sorted.
        assert text is None
        assert stats["parsed"] == ["item_302", "item_801"]

    def test_declared_901_only_guard_passes(self):
        # 9.01 is excluded from the required set, so the guard never fires.
        secs = {"item_101": FakeSection("z" * 350, part="1", item="1.01")}
        text, stats = select_text(
            _filing_with(secs), "8-K", declared_items="9.01")
        assert text is not None
        assert "reason" not in stats
        assert stats["kept"] == ["item_101"]

    def test_declared_item_present_guard_passes(self):
        secs = {"item_101": FakeSection("z" * 350, part="1", item="1.01")}
        text, stats = select_text(
            _filing_with(secs), "8-K", declared_items="1.01")
        assert text is not None
        assert stats["kept"] == ["item_101"]

    def test_declared_excluded_item_2_02_not_required(self):
        # 2.02 isn't in the keep-list so it never becomes a "required" key.
        secs = {"item_101": FakeSection("z" * 350)}
        text, stats = select_text(
            _filing_with(secs), "8-K", declared_items="1.01,2.02")
        assert text is not None
        assert "reason" not in stats

    def test_guard_only_runs_for_8k(self):
        # declared_items is ignored for periodic forms (no guard branch).
        secs = {"part_ii_item_7": FakeSection("z" * 5000, confidence=0.9)}
        text, stats = select_text(
            _filing_with(secs), "10-K", declared_items="1.01")
        assert text is not None
        assert "reason" not in stats
        # The 'parsed' key is only emitted on the 8-K declared-bail branch.
        assert "parsed" not in stats

    def test_declared_guard_bails_with_iterable_input(self):
        # The guard path must work when declared_items is an iterable (not
        # just a comma string): list ["1.01","3.02"] with only 3.02 parsed.
        secs = {"item_302": FakeSection("z" * 350, confidence=0.9)}
        text, stats = select_text(
            _filing_with(secs), "8-K", declared_items=["1.01", "3.02"])
        assert text is None
        assert stats["reason"] == "declared items not detected: ['item_101']"
        assert stats["parsed"] == ["item_302"]

    def test_declared_int_items_pass_when_parsed(self):
        # Integer item codes (101) are coerced via str() and must match the
        # parsed key item_101 so the guard passes.
        secs = {"item_101": FakeSection("z" * 350, confidence=0.9)}
        text, stats = select_text(
            _filing_with(secs), "8-K", declared_items=[101])
        assert text is not None
        assert stats["kept"] == ["item_101"]

    def test_success_8k_omits_parsed_key(self):
        # Even when declared_items drives a (passing) guard, success stats
        # carry only kept/dropped/kept_chars/dropped_chars.
        secs = {"item_101": FakeSection("z" * 350, confidence=0.9)}
        text, stats = select_text(
            _filing_with(secs), "8-K", declared_items="1.01")
        assert text is not None
        assert set(stats) == {"kept", "dropped", "kept_chars", "dropped_chars"}


# ── select_text: confidence handling ──────────────────────────────────────────
class TestSelectTextConfidence:
    def test_confidence_below_floor_bails(self):
        secs = {"item_101": FakeSection("z" * 350, confidence=0.5)}
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is None
        assert stats["reason"].startswith("keep-section low confidence:")
        assert "('item_101', 0.5)" in stats["reason"]
        assert stats["kept"] == []          # never added to kept_keys
        assert stats["dropped"] == []

    def test_confidence_exactly_at_floor_is_kept(self):
        # Strict `<` means 0.7 is NOT low -> section kept.
        secs = {"item_101": FakeSection("z" * 350, confidence=0.7)}
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is not None
        assert "reason" not in stats
        assert stats["kept"] == ["item_101"]

    def test_confidence_attr_missing_defaults_to_kept(self):
        # getattr default 1.0 -> kept.
        secs = {"item_101": FakeSection("z" * 350, confidence=FakeSection._ABSENT)}
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is not None
        assert stats["kept"] == ["item_101"]

    def test_confidence_present_but_none_is_treated_as_low(self):
        # `... or 0.0` turns None into 0.0 -> low-confidence bail. This is
        # the important divergence from a *missing* attribute (which is 1.0).
        secs = {"item_101": FakeSection("z" * 350, confidence=None)}
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is None
        assert "('item_101', 0.0)" in stats["reason"]

    def test_confidence_zero_bails(self):
        secs = {"item_101": FakeSection("z" * 350, confidence=0.0)}
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is None
        assert stats["reason"].startswith("keep-section low confidence:")

    def test_confidence_string_below_floor_is_coerced_and_bails(self):
        # confidence is coerced via float(...): a numeric STRING "0.5" parses
        # to 0.5 < 0.7 -> low-confidence bail. Pins the float() coercion.
        secs = {"item_101": FakeSection("z" * 350, confidence="0.5")}
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is None
        assert "('item_101', 0.5)" in stats["reason"]

    def test_confidence_string_at_floor_is_coerced_and_kept(self):
        # "0.7" -> 0.7, strict `<` keeps it.
        secs = {"item_101": FakeSection("z" * 350, confidence="0.7")}
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is not None
        assert stats["kept"] == ["item_101"]

    def test_confidence_above_one_is_kept(self):
        # Confidence is only floor-checked; values >1.0 (e.g. a buggy parser)
        # are still >= 0.7 so they pass rather than being clamped/rejected.
        secs = {"item_101": FakeSection("z" * 350, confidence=1.5)}
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is not None
        assert stats["kept"] == ["item_101"]

    def test_low_confidence_short_circuits_other_valid_sections(self):
        # Order matters: low-confidence bail is checked AFTER the loop and
        # returns before the kept_keys / floor checks, so a single low-conf
        # keep section short-circuits everything even with other good ones.
        secs = OrderedDict([
            ("item_101", FakeSection("z" * 350, confidence=0.95)),
            ("item_302", FakeSection("y" * 350, confidence=0.1)),
        ])
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is None
        assert stats["reason"].startswith("keep-section low confidence:")
        # item_101 was collected into kept_keys before item_302 tripped it.
        assert stats["kept"] == ["item_101"]


# ── select_text: kept-chars floor ─────────────────────────────────────────────
class TestSelectTextFloor:
    def test_8k_at_floor_300_passes(self):
        secs = {"item_101": FakeSection("z" * 300, confidence=0.9)}
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is not None              # strict `<`: equal passes
        assert stats["kept_chars"] == 300

    def test_8k_one_below_floor_bails(self):
        secs = {"item_101": FakeSection("z" * 299, confidence=0.9)}
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is None
        assert stats["reason"] == "kept_chars 299 below floor 300"
        assert stats["kept"] == ["item_101"]
        assert stats["dropped"] == []

    def test_10k_at_floor_5000_passes(self):
        secs = {"part_ii_item_7": FakeSection("z" * 5000, confidence=0.9)}
        text, stats = select_text(_filing_with(secs), "10-K")
        assert text is not None
        assert stats["kept_chars"] == 5000

    def test_10k_one_below_floor_bails(self):
        secs = {"part_ii_item_7": FakeSection("z" * 4999, confidence=0.9)}
        text, stats = select_text(_filing_with(secs), "10-K")
        assert text is None
        assert stats["reason"] == "kept_chars 4999 below floor 5000"

    @pytest.mark.parametrize("form, floor", [
        ("10-K", 5000), ("10-Q", 5000), ("20-F", 5000), ("8-K", 300),
    ])
    def test_floor_matches_per_form_dict(self, form, floor):
        # Drive one keep-list section per form just below its floor and
        # assert the reason names the right floor number.
        key = sorted(KEEP_SECTIONS[form])[0]
        secs = {key: FakeSection("z" * (floor - 1), confidence=0.9)}
        text, stats = select_text(_filing_with(secs), form)
        assert text is None
        assert stats["reason"] == f"kept_chars {floor - 1} below floor {floor}"


# ── select_text: dropped / no-keep handling ──────────────────────────────────
class TestSelectTextDropped:
    def test_non_keep_section_counted_in_dropped(self):
        secs = OrderedDict([
            ("item_101", FakeSection("z" * 350, confidence=0.9)),
            ("item_202", FakeSection("y" * 123, confidence=0.9)),  # not kept
        ])
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is not None
        assert stats["kept"] == ["item_101"]
        assert stats["dropped"] == ["item_202"]
        assert stats["kept_chars"] == 350
        assert stats["dropped_chars"] == 123
        # dropped section text never reaches the output.
        assert "y" not in text

    def test_multiple_dropped_sections_accumulate_in_order(self):
        # Two non-keep sections straddling a keep section: dropped list keeps
        # iteration order and dropped_chars sums across both.
        secs = OrderedDict([
            ("item_202", FakeSection("a" * 100, confidence=0.9)),  # earnings
            ("item_101", FakeSection("z" * 350, confidence=0.9)),  # keep
            ("item_402", FakeSection("b" * 50, confidence=0.9)),   # restatement
        ])
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is not None
        assert stats["kept"] == ["item_101"]
        assert stats["dropped"] == ["item_202", "item_402"]
        assert stats["dropped_chars"] == 150
        assert stats["kept_chars"] == 350

    def test_no_keep_sections_present(self):
        # Only a dropped (non-keep) section exists -> "no keep sections present".
        secs = {"item_202": FakeSection("y" * 350, confidence=0.9)}
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is None
        assert stats["reason"] == "no keep sections present"
        assert stats["dropped"] == ["item_202"]
        assert "kept" not in stats

    def test_section_text_none_coerced_to_empty(self):
        # text() -> None becomes '' (n=0). With only this section kept_chars
        # is 0, which is below the floor -> bail with kept_chars 0.
        secs = {"item_101": FakeSection(None, confidence=0.9)}
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is None
        assert stats["reason"] == "kept_chars 0 below floor 300"
        assert stats["kept"] == ["item_101"]


# ── select_text: whitespace-only latent surprise ─────────────────────────────
class TestSelectTextWhitespaceOnly:
    def test_whitespace_only_long_section_emits_empty_text(self):
        # A whitespace-only kept section longer than the floor increments
        # kept_chars and kept_keys, clears the floor, BUT is not appended to
        # kept_parts because of the `if text.strip()` guard. Result: success
        # stats with a nonempty `kept` list yet an EMPTY output string. This
        # is a latent surprise worth pinning explicitly.
        secs = {"item_101": FakeSection(" " * 350, confidence=0.9)}
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text == ""                     # nothing emitted
        assert "reason" not in stats          # but treated as success
        assert stats["kept"] == ["item_101"]  # key still counted
        assert stats["kept_chars"] == 350     # chars still counted

    def test_whitespace_only_short_section_hits_floor_first(self):
        # Below the floor the floor check fires before the empty-text quirk.
        secs = {"item_101": FakeSection("   ", confidence=0.9)}
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is None
        assert stats["reason"] == "kept_chars 3 below floor 300"


# ── select_text: heading synthesis & assembly ────────────────────────────────
class TestSelectTextHeadings:
    def test_part_and_item_heading(self):
        secs = {"item_101": FakeSection("z" * 350, confidence=0.9,
                                        part="1", item="1.01")}
        text, _ = select_text(_filing_with(secs), "8-K")
        assert text.startswith("## Part 1, Item 1.01\n\n")

    def test_title_fallback_when_part_or_item_missing(self):
        secs = {"item_101": FakeSection("z" * 350, confidence=0.9,
                                        title="Material Agreement")}
        text, _ = select_text(_filing_with(secs), "8-K")
        assert text.startswith("## Material Agreement\n\n")

    def test_key_fallback_when_no_part_item_or_title(self):
        secs = {"item_101": FakeSection("z" * 350, confidence=0.9)}
        text, _ = select_text(_filing_with(secs), "8-K")
        assert text.startswith("## item_101\n\n")

    def test_title_none_falls_back_to_key(self):
        # title=None -> `getattr(...) or key` -> key.
        secs = {"item_101": FakeSection("z" * 350, confidence=0.9,
                                        title=None, part=None, item=None)}
        text, _ = select_text(_filing_with(secs), "8-K")
        assert text.startswith("## item_101\n\n")

    def test_part_present_item_none_falls_back(self):
        # Only one of part/item present -> not the "Part X, Item Y" branch.
        secs = {"item_101": FakeSection("z" * 350, confidence=0.9,
                                        part="1", item=None,
                                        title="Some Title")}
        text, _ = select_text(_filing_with(secs), "8-K")
        assert text.startswith("## Some Title\n\n")

    def test_leading_newlines_stripped_internal_separators_preserved(self):
        secs = OrderedDict([
            ("item_101", FakeSection("AAA", confidence=0.9,
                                     part="1", item="1.01")),
            ("item_302", FakeSection("B" * 350, confidence=0.9,
                                     part="3", item="3.02")),
        ])
        text, _ = select_text(_filing_with(secs), "8-K")
        # The very first '\n\n## ' had its leading newlines stripped.
        assert not text.startswith("\n")
        assert text.startswith("## Part 1, Item 1.01\n\nAAA")
        # Internal section separator is preserved verbatim.
        assert "\n\n## Part 3, Item 3.02\n\n" in text


# ── select_text: ordering & success stats shape ──────────────────────────────
class TestSelectTextOrderingAndStats:
    def test_kept_keys_preserve_iteration_order(self):
        secs = OrderedDict([
            ("item_801", FakeSection("a" * 200, confidence=0.9,
                                     part="8", item="8.01")),
            ("item_101", FakeSection("b" * 200, confidence=0.9,
                                     part="1", item="1.01")),
        ])
        text, stats = select_text(_filing_with(secs), "8-K")
        assert stats["kept"] == ["item_801", "item_101"]
        # body order mirrors iteration order too.
        assert text.index("8.01") < text.index("1.01")

    def test_success_stats_shape(self):
        secs = {"item_101": FakeSection("z" * 350, confidence=0.9,
                                        part="1", item="1.01")}
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is not None
        assert "reason" not in stats
        assert set(stats) == {"kept", "dropped", "kept_chars", "dropped_chars"}
        assert stats["kept"] == ["item_101"]
        assert stats["dropped"] == []
        assert stats["kept_chars"] == 350
        assert stats["dropped_chars"] == 0

    def test_failure_stats_always_carry_reason(self):
        # Every safe-fail path carries a 'reason' key.
        for filing, form, declared in [
            (object(), "40-F", None),
            (RaisingObjFiling(ValueError("x")), "8-K", None),
            (NoneObjFiling(), "8-K", None),
            (FakeFiling(FakeObjNoSections()), "8-K", None),
            (_filing_with({"item_101": FakeSection("z" * 350, confidence=0.1)}),
             "8-K", None),
            (_filing_with({"item_202": FakeSection("z" * 350, confidence=0.9)}),
             "8-K", None),
            (_filing_with({"item_101": FakeSection("z" * 10, confidence=0.9)}),
             "8-K", None),
        ]:
            text, stats = select_text(filing, form, declared_items=declared)
            assert text is None
            assert "reason" in stats

    def test_amendment_form_uses_base_keep_list(self):
        # 8-K/A normalizes to 8-K and uses the 8-K keep-list + floor.
        secs = {"item_101": FakeSection("z" * 350, confidence=0.9,
                                        part="1", item="1.01")}
        text, stats = select_text(_filing_with(secs), "8-K/A")
        assert text is not None
        assert stats["kept"] == ["item_101"]

    def test_multiple_kept_sections_sum_chars(self):
        secs = OrderedDict([
            ("item_101", FakeSection("a" * 200, confidence=0.9,
                                     part="1", item="1.01")),
            ("item_302", FakeSection("b" * 200, confidence=0.9,
                                     part="3", item="3.02")),
        ])
        text, stats = select_text(_filing_with(secs), "8-K")
        assert stats["kept"] == ["item_101", "item_302"]
        assert stats["kept_chars"] == 400


# ── adversarial-review additions ──────────────────────────────────────────────
# The original suite covered confidence/floor branches almost exclusively on the
# 8-K path (cheap because the floor is 300). These add the symmetric periodic
# coverage and a few boundaries the survey slice called out but the suite only
# partially exercised. Every expected value below was re-derived from the source
# and observed against the live function before being asserted.
class TestSelectTextConfidencePeriodic:
    def test_periodic_low_confidence_bails(self):
        # Confidence floor applies to periodic forms too, not just 8-K. A 10-K
        # keep section above the 5000 char floor but below 0.7 confidence still
        # bails to the full-document fallback.
        secs = {"part_ii_item_7": FakeSection("z" * 5000, confidence=0.5)}
        text, stats = select_text(_filing_with(secs), "10-K")
        assert text is None
        assert stats["reason"].startswith("keep-section low confidence:")
        assert "('part_ii_item_7', 0.5)" in stats["reason"]

    def test_periodic_missing_confidence_defaults_to_kept(self):
        # On a periodic form the getattr default (1.0) must apply identically:
        # a 10-K keep section with NO confidence attribute is kept.
        secs = {"part_ii_item_7": FakeSection("z" * 5000,
                                              confidence=FakeSection._ABSENT)}
        text, stats = select_text(_filing_with(secs), "10-K")
        assert text is not None
        assert stats["kept"] == ["part_ii_item_7"]

    @pytest.mark.parametrize("conf", [-0.3, -1.0])
    def test_negative_confidence_bails(self, conf):
        # Negative confidence is below the floor (no abs/clamp) -> low-conf bail
        # and the raw negative value is echoed verbatim in the reason tuple.
        secs = {"item_101": FakeSection("z" * 350, confidence=conf)}
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is None
        assert f"('item_101', {conf})" in stats["reason"]

    def test_all_keep_sections_low_confidence_yields_empty_kept(self):
        # When EVERY keep section is low-confidence they all divert to
        # low_confidence_keep and NONE reach kept_keys, so the bail carries
        # kept == []. (Contrast test_low_confidence_short_circuits_* where one
        # good section was collected before a later one tripped the bail.)
        secs = OrderedDict([
            ("item_101", FakeSection("z" * 350, confidence=0.2)),
            ("item_302", FakeSection("y" * 350, confidence=0.3)),
        ])
        text, stats = select_text(_filing_with(secs), "8-K")
        assert text is None
        # Both low-conf tuples accumulate in iteration order.
        assert stats["reason"] == (
            "keep-section low confidence: "
            "[('item_101', 0.2), ('item_302', 0.3)]")
        assert stats["kept"] == []


class TestDeclaredKeepKeysWhitespace:
    KEEP_8K = KEEP_SECTIONS["8-K"]

    def test_surrounding_whitespace_around_single_code_stripped(self):
        # " 1.01 " -> strip -> "1.01" -> "item_101"; pins the per-code .strip().
        assert _declared_keep_keys(" 1.01 ", self.KEEP_8K) == {"item_101"}

    @pytest.mark.parametrize("declared", [" ", "  ,  ", ","])
    def test_whitespace_only_or_blank_codes_yield_empty(self, declared):
        # A truthy-but-whitespace string is NOT caught by the `if not
        # declared_items` guard; every individual code strips to "" and is
        # dropped by the per-code `if not code` check -> empty set.
        assert _declared_keep_keys(declared, self.KEEP_8K) == set()


class TestSelectTextDeclaredGuardExtra:
    def test_non_keep_declared_item_never_required(self):
        # 5.02 (officer changes) is declared and NOT parsed, but it isn't a
        # keep-list item so it never enters the required set -> the guard does
        # NOT fire even though 5.02 is absent from the parsed sections.
        secs = {"item_101": FakeSection("z" * 350, confidence=0.9)}
        text, stats = select_text(
            _filing_with(secs), "8-K", declared_items="1.01,5.02")
        assert text is not None
        assert "reason" not in stats
        assert stats["kept"] == ["item_101"]

    def test_whitespace_only_declared_items_skips_guard(self):
        # declared_items=" " is truthy but yields an empty required set, so the
        # guard is a no-op and selection proceeds normally.
        secs = {"item_101": FakeSection("z" * 350, confidence=0.9)}
        text, stats = select_text(
            _filing_with(secs), "8-K", declared_items=" ")
        assert text is not None
        assert stats["kept"] == ["item_101"]
