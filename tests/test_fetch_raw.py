"""Unit tests for dilution/fetch_raw.py.

Covers the deterministic classifiers (`_extractable`, `_is_narrative_exhibit`),
the DB-backed UPSERT/truncation logic in `_store`, and the async fetch
orchestration (`_run_fetch` / `fetch_extractable_for_cik`) exercised through a
monkeypatched edgartools seam — never a real SEC call.

Per the survey slice, the autouse `temp_db` fixture (tests/conftest.py)
reroutes `db.get_conn()` to a fresh per-test SQLite DB with the full schema,
so the production DB is never touched. `db.get_conn()` enables
`PRAGMA foreign_keys=ON`, and `dilution_raw.accession_number` has an FK to
`dilution_filings`, so DB-backed tests stage the parent filing first.
"""

from __future__ import annotations

import pytest

from dilution import fetch_raw
from dilution.fetch_raw import (
    MAX_CHARS,
    _extractable,
    _is_narrative_exhibit,
    _store,
)


# ─────────────────────────────────────────────────────────────────────────
# _extractable — pure prefix-match classifier
# ─────────────────────────────────────────────────────────────────────────
class TestExtractable:
    @pytest.mark.parametrize(
        "form",
        [
            "8-K",
            "6-K",
            "424B",
            "424B5",
            "424B3",
            "S-1",
            "S-3",
            "S-3ASR",
            "S-4",
            "F-1",
            "F-3",
            "SUPPL",
            "POS AM",
            "425",
            "DEF 14A",
            "DEFA14A",
            "DEFM14A",
            "PRE 14A",
            "FWP",
            "10-K",
            "10-Q",
            "20-F",
            "40-F",
            "1-A",
            "1-U",
            "1-K",
            "1-SA",
            "EFFECT",
            "RW",
        ],
    )
    def test_exact_prefix_forms_are_extractable(self, form):
        assert _extractable(form) is True

    @pytest.mark.parametrize(
        "form,expected",
        [
            ("8-K", True),  # exact match
            ("8-K/A", True),  # amendment — startswith '8-K'
            ("S-3ASR", True),  # startswith 'S-3'
            ("S-1/A", True),  # startswith 'S-1'
            ("424B5", True),  # startswith '424B'
            ("10-K405", True),  # legacy 10-K variant, startswith '10-K'
            ("10-Q/A", True),
            ("6-K/A", True),
        ],
    )
    def test_prefix_match_amendments_and_variants(self, form, expected):
        assert _extractable(form) is expected

    @pytest.mark.parametrize(
        "form",
        [
            "SC 13D",
            "SC 13G",
            "3",
            "4",
            "5",  # insider forms
            "NT 10-K",  # does NOT start with '10-K'
            "NT 10-Q",
            "D",
            "13F-HR",
            "CORRESP",
            "UPLOAD",
        ],
    )
    def test_non_extractable_forms(self, form):
        assert _extractable(form) is False

    def test_empty_string_is_not_extractable(self):
        # Guarded by bool(form) — empty string is falsy.
        assert _extractable("") is False

    def test_none_does_not_crash_and_returns_false(self):
        # bool(None) is False; the `and` short-circuits before any
        # .startswith() call, so None is safe.
        assert _extractable(None) is False

    def test_lowercase_form_is_not_extractable(self):
        # Case-sensitivity asymmetry: _extractable applies NO .upper(),
        # unlike _is_narrative_exhibit. A lowercase form misses every
        # prefix.
        assert _extractable("8-k") is False
        assert _extractable("s-3") is False

    def test_overlapping_prefixes_still_true(self):
        # Both 'S-3' and 'S-3ASR' live in the prefix tuple; any()
        # short-circuits on the first match — result is still True.
        assert _extractable("S-3ASR") is True

    def test_return_type_is_bool(self):
        # any(...) over a non-empty form yields a real bool, and the
        # bool(form) guard ensures the False branch is also a bool.
        assert _extractable("8-K") is True
        assert _extractable("nope") is False


# ─────────────────────────────────────────────────────────────────────────
# _is_narrative_exhibit — pure routing classifier
# ─────────────────────────────────────────────────────────────────────────
class TestIsNarrativeExhibit:
    # --- first-guard skips ------------------------------------------------
    def test_none_doc_type_skips(self):
        assert _is_narrative_exhibit(None, "x.htm", "8-K") == "skip"

    def test_empty_doc_type_skips(self):
        assert _is_narrative_exhibit("", "x.htm", "8-K") == "skip"

    def test_graphic_doc_type_skips(self):
        assert _is_narrative_exhibit("GRAPHIC", "x.htm", "8-K") == "skip"

    @pytest.mark.parametrize(
        "document",
        ["logo.jpg", "logo.jpeg", "logo.png", "logo.gif", "logo.PNG", "IMG.JPG"],
    )
    def test_image_documents_skip_case_insensitive(self, document):
        # document is .lower()'d before the .endswith() check, so any
        # casing of an image extension routes to skip.
        assert _is_narrative_exhibit("EX-99.1", document, "8-K") == "skip"

    # --- hard-skip exhibit families --------------------------------------
    @pytest.mark.parametrize(
        "doc_type",
        ["EX-21", "EX-21.1", "EX-23", "EX-23.1", "EX-24", "EX-24.1"],
    )
    def test_subs_consents_poas_skip(self, doc_type):
        assert _is_narrative_exhibit(doc_type, "x.htm", "8-K") == "skip"

    def test_ex231_false_positive_skips_via_startswith(self):
        # BUG-ADJACENT (documented intent): startswith('EX-23') means a
        # hypothetical 'EX-231' also matches the consent-skip branch.
        # This is the documented startswith collision — assert it.
        assert _is_narrative_exhibit("EX-231", "x.htm", "8-K") == "skip"

    @pytest.mark.parametrize("doc_type", ["EX-101", "EX-101.INS", "EX-104"])
    def test_xbrl_skips(self, doc_type):
        assert _is_narrative_exhibit(doc_type, "x.htm", "8-K") == "skip"

    # --- 'always' families: EX-1/2/3/4 (note trailing dot) ----------------
    @pytest.mark.parametrize(
        "doc_type",
        ["EX-1.1", "EX-2.1", "EX-3.1", "EX-4.1", "EX-1.", "EX-3.2"],
    )
    def test_ex_1234_dot_always(self, doc_type):
        assert _is_narrative_exhibit(doc_type, "x.htm", "10-K") == "always"

    def test_ex10_does_not_hit_ex1_always_branch(self):
        # The trailing dot in 'EX-1.' guards against 'EX-10.1' matching;
        # EX-10 must route to classify, not always.
        assert _is_narrative_exhibit("EX-10.1", "x.htm", "10-K") == "classify"

    @pytest.mark.parametrize("doc_type", ["EX-FILING", "EX-FILING.FEES", "EX-107", "EX-107.1"])
    def test_filing_fee_table_always(self, doc_type):
        assert _is_narrative_exhibit(doc_type, "x.htm", "S-3") == "always"

    # --- EX-99 on 8-K/6-K = always ---------------------------------------
    @pytest.mark.parametrize("form", ["8-K", "8-K/A", "6-K", "6-K/A"])
    def test_ex99_on_8k_6k_always(self, form):
        assert _is_narrative_exhibit("EX-99.1", "pr.htm", form) == "always"

    def test_ex99_lowercase_form_falls_to_classify(self):
        # Case-sensitivity asymmetry: form is only .strip()'d, never
        # uppercased, so lowercase '8-k' fails the startswith('8-K')
        # guard and falls through to the EX-99 classify branch.
        assert _is_narrative_exhibit("EX-99.1", "pr.htm", "8-k") == "classify"

    @pytest.mark.parametrize("form", ["10-K", "S-3", None, ""])
    def test_ex99_on_non_8k_6k_classifies(self, form):
        # The 8-K/6-K always-guard fails, so EX-99 routes to classify.
        assert _is_narrative_exhibit("EX-99.1", "pr.htm", form) == "classify"

    def test_ex99_on_425_classifies_not_always(self):
        # REVIEW-ADDED: '425' (Rule 425 M&A communication) is extractable
        # but is NEITHER 8-K NOR 6-K, so an EX-99 press release on a 425
        # does NOT hit the always-branch — it falls through to classify.
        # Guards against a regression that widened the always-guard.
        assert _is_narrative_exhibit("EX-99.1", "pr.htm", "425") == "classify"

    def test_ex99_default_form_none_classifies(self):
        # REVIEW-ADDED: form is an optional 3rd param defaulting to None.
        # Omitting it entirely must behave like form=None → the 8-K/6-K
        # always-guard fails (f == "") → EX-99 routes to classify.
        assert _is_narrative_exhibit("EX-99.1", "pr.htm") == "classify"

    def test_always_family_default_form_none(self):
        # REVIEW-ADDED: EX-3.x and EX-FILING are 'always' regardless of
        # form, so the default-None form arg must not change the verdict.
        assert _is_narrative_exhibit("EX-3.1", "pr.htm") == "always"
        assert _is_narrative_exhibit("EX-FILING.FEES", "x.htm") == "always"

    def test_form_with_whitespace_is_stripped(self):
        # f = (form or '').strip() — surrounding whitespace must not
        # break the startswith('8-K') guard.
        assert _is_narrative_exhibit("EX-99.1", "pr.htm", "  8-K  ") == "always"

    # --- classify families ------------------------------------------------
    @pytest.mark.parametrize("form", ["8-K", "10-K", "S-3", None])
    def test_ex10_always_classifies(self, form):
        # EX-10 is description-routed regardless of form.
        assert _is_narrative_exhibit("EX-10.1", "x.htm", form) == "classify"

    # --- fall-through skip ------------------------------------------------
    @pytest.mark.parametrize("doc_type", ["EX-5.1", "EX-8.1", "EX-95", "COVER"])
    def test_unknown_families_skip(self, doc_type):
        assert _is_narrative_exhibit(doc_type, "x.htm", "8-K") == "skip"

    # --- casing of doc_type ----------------------------------------------
    def test_doc_type_is_uppercased(self):
        # dt = doc_type.upper(); a lowercase 'ex-3.1' still routes always.
        assert _is_narrative_exhibit("ex-3.1", "x.htm", "8-K") == "always"
        assert _is_narrative_exhibit("ex-10.1", "x.htm", "8-K") == "classify"

    def test_none_document_uses_empty_string_no_crash(self):
        # document=None -> doc = '' (no .png ending). With a valid
        # always-doc_type the verdict is still 'always'.
        assert _is_narrative_exhibit("EX-99.1", None, "8-K") == "always"
        assert _is_narrative_exhibit("EX-3.1", None, "8-K") == "always"

    def test_return_is_one_of_three_literals(self):
        for verdict in (
            _is_narrative_exhibit("EX-1.1", "x", "8-K"),
            _is_narrative_exhibit("EX-10.1", "x", "8-K"),
            _is_narrative_exhibit("GRAPHIC", "x", "8-K"),
        ):
            assert verdict in ("always", "classify", "skip")


# ─────────────────────────────────────────────────────────────────────────
# _store — db-backed UPSERT + truncation
# ─────────────────────────────────────────────────────────────────────────
class TestStore:
    def _stage_parent(self, temp_db, accession="ACC-1", cik=7):
        # dilution_raw.accession_number FKs to dilution_filings, and
        # get_conn() enables PRAGMA foreign_keys=ON, so the parent must
        # exist before _store inserts a child row.
        temp_db.add_filing(accession, cik=cik, form="8-K", primary_doc="p.htm")

    def test_fresh_insert(self, temp_db):
        self._stage_parent(temp_db)
        _store("ACC-1", "p.htm", "8-K", "primary body")
        rows = temp_db.execute(
            "SELECT accession_number, doc_name, doc_type, content_md, "
            "downloaded_at FROM dilution_raw"
        )
        assert len(rows) == 1
        r = rows[0]
        assert r["accession_number"] == "ACC-1"
        assert r["doc_name"] == "p.htm"
        assert r["doc_type"] == "8-K"
        assert r["content_md"] == "primary body"
        # downloaded_at set via now_iso() — non-null ISO string with Z.
        assert r["downloaded_at"] and r["downloaded_at"].endswith("Z")

    def test_upsert_same_key_updates_not_duplicates(self, temp_db):
        self._stage_parent(temp_db)
        _store("ACC-1", "p.htm", "8-K", "first version")
        _store("ACC-1", "p.htm", "EX-99.1", "second version")  # same PK
        rows = temp_db.execute(
            "SELECT doc_type, content_md FROM dilution_raw "
            "WHERE accession_number=? AND doc_name=?",
            ("ACC-1", "p.htm"),
        )
        # Row count stays 1; content_md AND doc_type overwritten.
        assert len(rows) == 1
        assert rows[0]["content_md"] == "second version"
        assert rows[0]["doc_type"] == "EX-99.1"

    def test_different_doc_name_same_accession_two_rows(self, temp_db):
        self._stage_parent(temp_db)
        _store("ACC-1", "p.htm", "8-K", "primary")
        _store("ACC-1", "ex99.htm", "EX-99.1", "exhibit")
        rows = temp_db.execute(
            "SELECT doc_name FROM dilution_raw WHERE accession_number=? "
            "ORDER BY doc_name",
            ("ACC-1",),
        )
        assert [r["doc_name"] for r in rows] == ["ex99.htm", "p.htm"]

    def test_empty_md_stored_as_empty_string(self, temp_db):
        # content_md is NOT NULL but '' satisfies that constraint.
        self._stage_parent(temp_db)
        _store("ACC-1", "p.htm", "8-K", "")
        rows = temp_db.execute(
            "SELECT content_md FROM dilution_raw WHERE accession_number=?",
            ("ACC-1",),
        )
        assert len(rows) == 1
        assert rows[0]["content_md"] == ""

    def test_md_exactly_max_chars_stored_unchanged(self, temp_db, monkeypatch):
        # Keep the test fast by shrinking the cap.
        monkeypatch.setattr(fetch_raw, "MAX_CHARS", 100)
        self._stage_parent(temp_db)
        md = "x" * 100
        _store("ACC-1", "p.htm", "8-K", md)
        rows = temp_db.execute(
            "SELECT content_md FROM dilution_raw WHERE accession_number=?",
            ("ACC-1",),
        )
        assert len(rows[0]["content_md"]) == 100

    def test_md_over_max_chars_truncated_to_boundary(self, temp_db, monkeypatch):
        monkeypatch.setattr(fetch_raw, "MAX_CHARS", 100)
        self._stage_parent(temp_db)
        md = "y" * 101  # MAX_CHARS + 1
        _store("ACC-1", "p.htm", "8-K", md)
        rows = temp_db.execute(
            "SELECT content_md FROM dilution_raw WHERE accession_number=?",
            ("ACC-1",),
        )
        stored = rows[0]["content_md"]
        assert len(stored) == 100
        assert stored == "y" * 100

    def test_real_max_chars_constant_value(self):
        # Sanity-check the production cap matches the documented value.
        assert MAX_CHARS == 2_000_000

    def test_truncation_with_real_constant(self, temp_db):
        # Exercise the real (un-patched) MAX_CHARS truncation boundary
        # once to prove the production path, not just the shrunk one.
        self._stage_parent(temp_db)
        md = "z" * (MAX_CHARS + 1)
        _store("ACC-1", "p.htm", "8-K", md)
        rows = temp_db.execute(
            "SELECT LENGTH(content_md) AS n FROM dilution_raw "
            "WHERE accession_number=?",
            ("ACC-1",),
        )
        assert rows[0]["n"] == MAX_CHARS

    def test_missing_parent_filing_raises_fk_error(self, temp_db):
        # FK enforcement is ON in get_conn(); storing a child for a
        # nonexistent filing must raise, not silently insert.
        import sqlite3

        with pytest.raises(sqlite3.IntegrityError):
            _store("NO-PARENT", "p.htm", "8-K", "body")


# ─────────────────────────────────────────────────────────────────────────
# Fetch orchestration — _run_fetch / fetch_extractable_for_cik
#   monkeypatch the edgartools seam; never a real SEC call.
# ─────────────────────────────────────────────────────────────────────────
class _FakeAttachment:
    def __init__(self, document, document_type, description=None, md="exhibit body"):
        self.document = document
        self.document_type = document_type
        self.description = description
        self._md = md

    def markdown(self):
        return self._md


class _FakeFiling:
    """Minimal stand-in for an edgartools Filing.

    Only exposes .markdown() and .attachments — enough for the primary +
    narrative-exhibit fetch loop. Deliberately lacks .obj so the periodic
    section-select path falls through to the full-document path.
    """

    def __init__(self, *, primary_md="primary body", attachments=None):
        self._primary_md = primary_md
        self.attachments = attachments if attachments is not None else []

    def markdown(self):
        return self._primary_md


@pytest.fixture
def patch_edgar(monkeypatch):
    """Patch set_identity to a no-op and route get_by_accession_number to
    a per-accession registry. Returns a dict the test fills with
    {accession: FakeFiling | None | Exception}.
    """
    registry: dict[str, object] = {}

    monkeypatch.setattr(fetch_raw, "set_identity", lambda *a, **k: None)

    def _lookup(accession):
        val = registry.get(accession)
        if isinstance(val, Exception):
            raise val
        return val

    monkeypatch.setattr(fetch_raw, "get_by_accession_number", _lookup)
    return registry


class TestFetchOrchestration:
    def test_empty_cik_returns_zero_counters(self, temp_db, patch_edgar):
        # No filings at all → early-return total==0 path.
        res = fetch_raw.fetch_extractable_for_cik(12345)
        assert res == {"fetched": 0, "docs": 0, "total": 0, "errors": 0}

    def test_non_extractable_forms_filtered_out(self, temp_db, patch_edgar):
        # SC 13D is not in EXTRACTABLE_PREFIXES → post-filter drops it,
        # leaving total==0 even though a filing row exists.
        temp_db.add_filing("SC-1", cik=50, form="SC 13D", primary_doc="p.htm")
        patch_edgar["SC-1"] = _FakeFiling()
        res = fetch_raw.fetch_extractable_for_cik(50)
        assert res["total"] == 0
        assert res["fetched"] == 0

    def test_already_fetched_filing_excluded_by_left_join(self, temp_db, patch_edgar):
        # A filing that already has a dilution_raw row is excluded by the
        # LEFT JOIN ... IS NULL clause.
        temp_db.add_filing("A-3", cik=60, form="6-K", primary_doc="p.htm")
        temp_db.execute(
            "INSERT INTO dilution_raw (accession_number, doc_name, doc_type, "
            "content_md, downloaded_at) VALUES (?,?,?,?,?)",
            ("A-3", "p.htm", "6-K", "already", "2025-01-01T00:00:00Z"),
        )
        patch_edgar["A-3"] = _FakeFiling()
        res = fetch_raw.fetch_extractable_for_cik(60)
        assert res["total"] == 0

    def test_primary_plus_exhibit_stored_and_counted(self, temp_db, patch_edgar):
        # 6-K avoids the periodic section-select path. Primary doc +
        # one EX-99.1 (always on 6-K) → 2 docs.
        temp_db.add_filing("A-1", cik=70, form="6-K", primary_doc="p.htm")
        patch_edgar["A-1"] = _FakeFiling(
            primary_md="primary body",
            attachments=[
                _FakeAttachment("ex991.htm", "EX-99.1", md="press release"),
            ],
        )
        res = fetch_raw.fetch_extractable_for_cik(70)
        assert res == {"fetched": 1, "docs": 2, "total": 1, "errors": 0}
        rows = temp_db.execute(
            "SELECT doc_name, content_md FROM dilution_raw "
            "WHERE accession_number=? ORDER BY doc_name",
            ("A-1",),
        )
        names = {r["doc_name"]: r["content_md"] for r in rows}
        assert names == {"ex991.htm": "press release", "p.htm": "primary body"}

    def test_skip_verdict_exhibit_not_stored(self, temp_db, patch_edgar):
        # A GRAPHIC attachment routes to skip → only the primary doc is
        # stored (1 doc).
        temp_db.add_filing("A-2", cik=71, form="6-K", primary_doc="p.htm")
        patch_edgar["A-2"] = _FakeFiling(
            attachments=[_FakeAttachment("logo.png", "GRAPHIC")],
        )
        res = fetch_raw.fetch_extractable_for_cik(71)
        assert res["docs"] == 1
        rows = temp_db.execute(
            "SELECT doc_name FROM dilution_raw WHERE accession_number=?",
            ("A-2",),
        )
        assert [r["doc_name"] for r in rows] == ["p.htm"]

    def test_primary_doc_attachment_deduped(self, temp_db, patch_edgar):
        # An attachment whose document matches primary_doc (case-insensitive)
        # is skipped to avoid double-storing the primary.
        temp_db.add_filing("A-4", cik=72, form="6-K", primary_doc="P.htm")
        patch_edgar["A-4"] = _FakeFiling(
            attachments=[_FakeAttachment("p.htm", "EX-99.1", md="dup primary")],
        )
        res = fetch_raw.fetch_extractable_for_cik(72)
        # Only the primary stored (the attachment was the primary again).
        assert res["docs"] == 1

    def test_lookup_returns_none_counts_as_zero_docs_no_error(self, temp_db, patch_edgar):
        # get_by_accession_number returns None → 0 written, not an error.
        temp_db.add_filing("A-5", cik=73, form="6-K", primary_doc="p.htm")
        patch_edgar["A-5"] = None
        res = fetch_raw.fetch_extractable_for_cik(73)
        assert res == {"fetched": 0, "docs": 0, "total": 1, "errors": 0}

    def test_lookup_exception_counted_as_zero_docs_no_worker_error(
        self, temp_db, patch_edgar
    ):
        # _fetch_filing_text_async catches the lookup exception internally
        # and returns 0, so the worker sees n==0 (not an error counter).
        temp_db.add_filing("A-6", cik=74, form="6-K", primary_doc="p.htm")
        patch_edgar["A-6"] = RuntimeError("SEC 503")
        res = fetch_raw.fetch_extractable_for_cik(74)
        assert res == {"fetched": 0, "docs": 0, "total": 1, "errors": 0}

    def test_limit_slices_after_date_sort(self, temp_db, patch_edgar):
        # ORDER BY filing_date DESC then [:limit] → only the newest
        # filing is fetched.
        temp_db.add_filing("OLD", cik=75, form="6-K", filing_date="2025-01-01",
                           primary_doc="old.htm")
        temp_db.add_filing("NEW", cik=75, form="6-K", filing_date="2025-12-01",
                           primary_doc="new.htm")
        patch_edgar["OLD"] = _FakeFiling(primary_md="old body")
        patch_edgar["NEW"] = _FakeFiling(primary_md="new body")
        res = fetch_raw.fetch_extractable_for_cik(75, limit=1)
        assert res["total"] == 1
        rows = temp_db.execute("SELECT accession_number FROM dilution_raw")
        assert [r["accession_number"] for r in rows] == ["NEW"]

    def test_since_date_filters_older_filings(self, temp_db, patch_edgar):
        temp_db.add_filing("OLD", cik=76, form="6-K", filing_date="2024-06-01",
                           primary_doc="old.htm")
        temp_db.add_filing("NEW", cik=76, form="6-K", filing_date="2025-06-01",
                           primary_doc="new.htm")
        patch_edgar["OLD"] = _FakeFiling()
        patch_edgar["NEW"] = _FakeFiling()
        res = fetch_raw.fetch_extractable_for_cik(76, since_date="2025-01-01")
        assert res["total"] == 1
        rows = temp_db.execute("SELECT accession_number FROM dilution_raw")
        assert {r["accession_number"] for r in rows} == {"NEW"}

    def test_blank_primary_markdown_not_stored(self, temp_db, patch_edgar):
        # md.strip() falsy → primary not stored; no exhibits → 0 docs.
        temp_db.add_filing("A-7", cik=77, form="6-K", primary_doc="p.htm")
        patch_edgar["A-7"] = _FakeFiling(primary_md="   ")
        res = fetch_raw.fetch_extractable_for_cik(77)
        assert res["docs"] == 0
        assert res["fetched"] == 0

    def test_periodic_form_falls_through_to_full_markdown(self, temp_db, patch_edgar):
        # 8-K triggers is_periodic_with_sections; on the fake (no .obj)
        # select_text returns (None, {"reason": ...}) — verified directly,
        # it does NOT raise — so `selected` is falsy, the section-select
        # branch is skipped, md stays None, and the code falls through to
        # filing.markdown(). Either way the observable outcome is the same:
        # the full primary body is stored.
        temp_db.add_filing("A-8", cik=78, form="8-K", primary_doc="p.htm")
        patch_edgar["A-8"] = _FakeFiling(primary_md="full 8-K body")
        res = fetch_raw.fetch_extractable_for_cik(78)
        assert res["docs"] == 1
        rows = temp_db.execute(
            "SELECT content_md FROM dilution_raw WHERE accession_number=?",
            ("A-8",),
        )
        assert rows[0]["content_md"] == "full 8-K body"

    def test_concurrency_clamped_to_at_least_one(self, temp_db, patch_edgar):
        # concurrency=0 must be clamped to 1 (asyncio.Semaphore(0) would
        # deadlock); with no filings it returns the zero-counter dict.
        res = fetch_raw.fetch_extractable_for_cik(999, concurrency=0)
        assert res == {"fetched": 0, "docs": 0, "total": 0, "errors": 0}

    def test_concurrency_clamp_still_processes_with_zero(self, temp_db, patch_edgar):
        # Prove the clamp actually lets work proceed (not just early-out).
        temp_db.add_filing("A-9", cik=80, form="6-K", primary_doc="p.htm")
        patch_edgar["A-9"] = _FakeFiling(primary_md="body")
        res = fetch_raw.fetch_extractable_for_cik(80, concurrency=0)
        assert res["fetched"] == 1
        assert res["total"] == 1

    def test_default_concurrency_from_config(self, temp_db, patch_edgar, monkeypatch):
        # concurrency=None → pulled from config.LLM_CONCURRENCY; should
        # not crash and should process the one filing.
        temp_db.add_filing("A-10", cik=81, form="6-K", primary_doc="p.htm")
        patch_edgar["A-10"] = _FakeFiling(primary_md="body")
        res = fetch_raw.fetch_extractable_for_cik(81)  # concurrency=None
        assert res["fetched"] == 1

    @pytest.mark.parametrize("concurrency", [-1, -5])
    def test_negative_concurrency_clamped_and_processes(
        self, temp_db, patch_edgar, concurrency
    ):
        # REVIEW-ADDED: failure-path boundary. max(1, int(concurrency))
        # clamps a negative semaphore size to 1 (asyncio.Semaphore(-1)
        # would raise ValueError). Work must still complete.
        temp_db.add_filing("NEG-1", cik=84, form="6-K", primary_doc="p.htm")
        patch_edgar["NEG-1"] = _FakeFiling(primary_md="body")
        res = fetch_raw.fetch_extractable_for_cik(84, concurrency=concurrency)
        assert res == {"fetched": 1, "docs": 1, "total": 1, "errors": 0}

    def test_set_identity_invoked_through_run_fetch(
        self, temp_db, monkeypatch
    ):
        # REVIEW-ADDED: prove the SEC-identity seam is actually wired into
        # _run_fetch (and patched away from real edgar global state). The
        # patch_edgar fixture no-ops it; here we capture the call to assert
        # it fires with config.EDGAR_IDENTITY before any fetch work.
        import config

        seen = []
        monkeypatch.setattr(fetch_raw, "set_identity",
                            lambda ident=None, *a, **k: seen.append(ident))
        monkeypatch.setattr(fetch_raw, "get_by_accession_number",
                            lambda acc: _FakeFiling(primary_md="body"))
        temp_db.add_filing("ID-1", cik=85, form="6-K", primary_doc="p.htm")
        res = fetch_raw.fetch_extractable_for_cik(85)
        assert res["fetched"] == 1
        # set_identity ran exactly once with the configured identity.
        assert seen == [config.EDGAR_IDENTITY]

    def test_set_identity_invoked_even_with_zero_filings(
        self, temp_db, monkeypatch
    ):
        # REVIEW-ADDED: set_identity is called at the TOP of _run_fetch,
        # before the total==0 early-return, so it fires even when there is
        # nothing to fetch (the seam is unconditionally exercised).
        import config

        seen = []
        monkeypatch.setattr(fetch_raw, "set_identity",
                            lambda ident=None, *a, **k: seen.append(ident))
        monkeypatch.setattr(fetch_raw, "get_by_accession_number",
                            lambda acc: None)
        res = fetch_raw.fetch_extractable_for_cik(99999)
        assert res == {"fetched": 0, "docs": 0, "total": 0, "errors": 0}
        assert seen == [config.EDGAR_IDENTITY]

    def test_exhibit_markdown_empty_not_stored(self, temp_db, patch_edgar):
        # _attachment_markdown returns None for blank markdown → exhibit
        # skipped, only primary counted.
        temp_db.add_filing("A-11", cik=82, form="6-K", primary_doc="p.htm")
        patch_edgar["A-11"] = _FakeFiling(
            attachments=[_FakeAttachment("ex991.htm", "EX-99.1", md="   ")],
        )
        res = fetch_raw.fetch_extractable_for_cik(82)
        assert res["docs"] == 1
        rows = temp_db.execute(
            "SELECT doc_name FROM dilution_raw WHERE accession_number=?",
            ("A-11",),
        )
        assert [r["doc_name"] for r in rows] == ["p.htm"]

    def test_attachment_without_document_skipped(self, temp_db, patch_edgar):
        # An attachment with document=None is skipped (the `if not doc`
        # guard) before classification.
        temp_db.add_filing("A-12", cik=83, form="6-K", primary_doc="p.htm")
        patch_edgar["A-12"] = _FakeFiling(
            attachments=[_FakeAttachment(None, "EX-99.1")],
        )
        res = fetch_raw.fetch_extractable_for_cik(83)
        assert res["docs"] == 1

    def test_multiple_filings_aggregate_counters(self, temp_db, patch_edgar):
        for i, acc in enumerate(["M-1", "M-2", "M-3"]):
            temp_db.add_filing(acc, cik=90, form="6-K",
                               filing_date=f"2025-0{i+1}-01",
                               primary_doc=f"p{i}.htm")
            patch_edgar[acc] = _FakeFiling(primary_md=f"body {i}")
        res = fetch_raw.fetch_extractable_for_cik(90)
        assert res["total"] == 3
        assert res["fetched"] == 3
        assert res["docs"] == 3
        assert res["errors"] == 0


# ─────────────────────────────────────────────────────────────────────────
# fetch_filing_text — synchronous wrapper
# ─────────────────────────────────────────────────────────────────────────
class TestFetchFilingTextWrapper:
    def test_missing_filing_returns_zero(self, temp_db, patch_edgar):
        # No dilution_filings row for the accession → returns 0 without
        # ever calling get_by_accession_number.
        assert fetch_raw.fetch_filing_text("MISSING") == 0

    def test_stores_primary_and_returns_count(self, temp_db, patch_edgar):
        temp_db.add_filing("S-1ACC", cik=100, form="6-K", primary_doc="p.htm")
        patch_edgar["S-1ACC"] = _FakeFiling(primary_md="body")
        written = fetch_raw.fetch_filing_text("S-1ACC")
        assert written == 1
        rows = temp_db.execute(
            "SELECT content_md FROM dilution_raw WHERE accession_number=?",
            ("S-1ACC",),
        )
        assert rows[0]["content_md"] == "body"


# ─────────────────────────────────────────────────────────────────────────
# Description-router (`classify` verdict) path through _run_fetch.
#
# REVIEW-ADDED: the original suite exercised only the 'always' verdict
# (EX-99 on a 6-K) and the 'skip' verdict (GRAPHIC). The 'classify'
# branch — the deterministic description router at lines 239-247 of
# fetch_raw.py that calls classify_by_description and DROPs / KEEPs /
# fail-opens an EX-10 exhibit — had ZERO integration coverage. EX-10 is
# always 'classify' regardless of form, so a 6-K (which avoids the
# periodic section-select path) reaches the router cleanly.
#
# Expected values were derived by running fetch_extractable_for_cik once
# against constructed fakes and observing: DROP-desc EX-10 → only primary
# stored (docs=1); KEEP-desc and unknown/None-desc EX-10 → exhibit also
# stored (docs=2, fail-open).
# ─────────────────────────────────────────────────────────────────────────
class TestDescriptionRouterPath:
    def _stage(self, temp_db, acc, cik, form="6-K", primary="p.htm"):
        temp_db.add_filing(acc, cik=cik, form=form, primary_doc=primary)

    def test_ex10_drop_description_exhibit_not_stored(self, temp_db, patch_edgar):
        # EX-10 → 'classify' → classify_by_description('EMPLOYMENT
        # AGREEMENT') returns 'drop' → the exhibit is skipped before the
        # markdown round-trip. Only the primary doc survives.
        self._stage(temp_db, "DR-1", 200)
        patch_edgar["DR-1"] = _FakeFiling(
            attachments=[
                _FakeAttachment("emp.htm", "EX-10.1",
                                description="EMPLOYMENT AGREEMENT",
                                md="employment body"),
            ],
        )
        res = fetch_raw.fetch_extractable_for_cik(200)
        assert res == {"fetched": 1, "docs": 1, "total": 1, "errors": 0}
        rows = temp_db.execute(
            "SELECT doc_name FROM dilution_raw WHERE accession_number=?",
            ("DR-1",),
        )
        assert [r["doc_name"] for r in rows] == ["p.htm"]

    def test_ex10_keep_description_exhibit_stored(self, temp_db, patch_edgar):
        # 'SECURITIES PURCHASE AGREEMENT' is a KEEP phrase → router 'keep'
        # → exhibit stored alongside the primary (docs=2).
        self._stage(temp_db, "KP-1", 201)
        patch_edgar["KP-1"] = _FakeFiling(
            attachments=[
                _FakeAttachment("spa.htm", "EX-10.1",
                                description="SECURITIES PURCHASE AGREEMENT",
                                md="spa body"),
            ],
        )
        res = fetch_raw.fetch_extractable_for_cik(201)
        assert res == {"fetched": 1, "docs": 2, "total": 1, "errors": 0}
        rows = temp_db.execute(
            "SELECT doc_name, content_md FROM dilution_raw "
            "WHERE accession_number=? ORDER BY doc_name",
            ("KP-1",),
        )
        names = {r["doc_name"]: r["content_md"] for r in rows}
        assert names == {"p.htm": "primary body", "spa.htm": "spa body"}

    @pytest.mark.parametrize("description", [None, "", "QUARTERLY UPDATE DECK"])
    def test_ex10_unknown_description_fails_open_and_stores(
        self, temp_db, patch_edgar, description
    ):
        # No / generic description → classify_by_description returns
        # 'unknown' → fail-open per CLAUDE.md coverage rule → exhibit
        # stored (docs=2).
        self._stage(temp_db, "UN-1", 202)
        patch_edgar["UN-1"] = _FakeFiling(
            attachments=[
                _FakeAttachment("mat.htm", "EX-10.1",
                                description=description, md="material body"),
            ],
        )
        res = fetch_raw.fetch_extractable_for_cik(202)
        assert res["docs"] == 2
        rows = temp_db.execute(
            "SELECT doc_name FROM dilution_raw WHERE accession_number=? "
            "ORDER BY doc_name",
            ("UN-1",),
        )
        assert [r["doc_name"] for r in rows] == ["mat.htm", "p.htm"]

    def test_ex10_drop_emits_desc_classify_log(self, temp_db, patch_edgar, caplog):
        # The drop branch logs an INFO 'desc-classify drop' line carrying
        # the doc_type, doc name and accession. Assert the observable log.
        import logging

        self._stage(temp_db, "LG-1", 203)
        patch_edgar["LG-1"] = _FakeFiling(
            attachments=[
                _FakeAttachment("emp.htm", "EX-10.1",
                                description="EXECUTIVE EMPLOYMENT",
                                md="emp body"),
            ],
        )
        with caplog.at_level(logging.INFO, logger="dilution.fetch_raw"):
            fetch_raw.fetch_extractable_for_cik(203)
        assert any("desc-classify drop" in r.message for r in caplog.records)
        # The dropped doc name and accession appear in the log payload.
        drop_rec = next(r for r in caplog.records
                        if "desc-classify drop" in r.message)
        msg = drop_rec.getMessage()
        assert "emp.htm" in msg and "LG-1" in msg

    def test_ex10_keep_emits_desc_classify_log(self, temp_db, patch_edgar, caplog):
        # REVIEW-ADDED: the KEEP branch (lines 245-247) logs an INFO
        # 'desc-classify keep' line — the original suite asserted only the
        # symmetric DROP log. Assert the observable keep-log too.
        import logging

        self._stage(temp_db, "LK-1", 206)
        patch_edgar["LK-1"] = _FakeFiling(
            attachments=[
                _FakeAttachment("spa.htm", "EX-10.1",
                                description="SECURITIES PURCHASE AGREEMENT",
                                md="spa body"),
            ],
        )
        with caplog.at_level(logging.INFO, logger="dilution.fetch_raw"):
            fetch_raw.fetch_extractable_for_cik(206)
        keep_recs = [r for r in caplog.records
                     if "desc-classify keep" in r.message]
        assert keep_recs, "expected a 'desc-classify keep' INFO log"
        msg = keep_recs[0].getMessage()
        assert "spa.htm" in msg and "LK-1" in msg

    def test_mixed_verdicts_aggregate_one_kept_exhibit(self, temp_db, patch_edgar):
        # always(EX-99.1 on 6-K) + classify-drop(EX-10 employment) +
        # skip(GRAPHIC) → primary + the EX-99 only = 2 docs.
        self._stage(temp_db, "MX-1", 204)
        patch_edgar["MX-1"] = _FakeFiling(
            attachments=[
                _FakeAttachment("pr.htm", "EX-99.1", md="press release"),
                _FakeAttachment("emp.htm", "EX-10.1",
                                description="EMPLOYMENT AGREEMENT",
                                md="emp body"),
                _FakeAttachment("logo.png", "GRAPHIC"),
            ],
        )
        res = fetch_raw.fetch_extractable_for_cik(204)
        assert res == {"fetched": 1, "docs": 2, "total": 1, "errors": 0}
        rows = temp_db.execute(
            "SELECT doc_name FROM dilution_raw WHERE accession_number=? "
            "ORDER BY doc_name",
            ("MX-1",),
        )
        assert [r["doc_name"] for r in rows] == ["p.htm", "pr.htm"]

    def test_exhibit_markdown_raises_skips_that_exhibit(self, temp_db, patch_edgar):
        # _attachment_markdown swallows the exception from attachment
        # .markdown() and returns None → the exhibit is skipped (not
        # counted, not an error counter). Primary still stored.
        self._stage(temp_db, "RM-1", 205)
        att = _FakeAttachment("ex991.htm", "EX-99.1", md="ignored")

        def _boom():
            raise RuntimeError("parse failure")

        att.markdown = _boom  # exhibit-level markdown raises
        patch_edgar["RM-1"] = _FakeFiling(attachments=[att])
        res = fetch_raw.fetch_extractable_for_cik(205)
        assert res == {"fetched": 1, "docs": 1, "total": 1, "errors": 0}
        rows = temp_db.execute(
            "SELECT doc_name FROM dilution_raw WHERE accession_number=?",
            ("RM-1",),
        )
        assert [r["doc_name"] for r in rows] == ["p.htm"]


# ─────────────────────────────────────────────────────────────────────────
# _store — additional UPSERT-side observations the original suite missed.
# ─────────────────────────────────────────────────────────────────────────
class TestStoreUpsertExtra:
    def _stage_parent(self, temp_db, accession="ACC-U", cik=8):
        temp_db.add_filing(accession, cik=cik, form="8-K", primary_doc="p.htm")

    def test_upsert_refreshes_downloaded_at(self, temp_db, monkeypatch):
        # ON CONFLICT ... DO UPDATE SET downloaded_at = excluded.downloaded_at
        # means a re-store with a later now_iso() overwrites the timestamp.
        # Pin now_iso() deterministically to prove the update path.
        self._stage_parent(temp_db)
        monkeypatch.setattr(fetch_raw, "now_iso",
                            lambda: "2026-02-02T00:00:00Z")
        _store("ACC-U", "p.htm", "8-K", "v1")
        monkeypatch.setattr(fetch_raw, "now_iso",
                            lambda: "2026-03-03T00:00:00Z")
        _store("ACC-U", "p.htm", "8-K", "v2")
        rows = temp_db.execute(
            "SELECT downloaded_at, content_md FROM dilution_raw "
            "WHERE accession_number=? AND doc_name=?",
            ("ACC-U", "p.htm"),
        )
        assert len(rows) == 1
        assert rows[0]["downloaded_at"] == "2026-03-03T00:00:00Z"
        assert rows[0]["content_md"] == "v2"

    def test_md_one_below_max_chars_unchanged(self, temp_db, monkeypatch):
        # Boundary sweep companion to the ==MAX and ==MAX+1 cases: a
        # string strictly below the cap is stored verbatim.
        monkeypatch.setattr(fetch_raw, "MAX_CHARS", 100)
        self._stage_parent(temp_db)
        _store("ACC-U", "p.htm", "8-K", "q" * 99)
        rows = temp_db.execute(
            "SELECT LENGTH(content_md) AS n FROM dilution_raw "
            "WHERE accession_number=?",
            ("ACC-U",),
        )
        assert rows[0]["n"] == 99
