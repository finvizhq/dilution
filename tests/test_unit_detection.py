"""Unit tests for dilution/unit_detection.py.

Covers:
  - _heuristic_ads_ratio  (pure regex; latest-ratio supersession)
  - _is_fpi_from_filings   (db, LIKE prefix match)
  - _latest_annual_filing  (db, exact IN-list match)
  - _load_text_for         (db + SEC-fetch fallback; EX- skip + truncation)
  - _llm_ads_ratio         (async; LLM seam mocked, plausibility bounds)
  - populate_company_unit  (orchestrator; idempotency + branch routing)

No test makes a real network/SEC/LLM call. The SEC-fetch and LLM seams are
monkeypatched at the module namespace (dilution.unit_detection.*).
The autouse `temp_db` fixture (conftest.py) reroutes db.get_conn() to a
fresh per-test SQLite DB, so the real dilution.db is never touched.
"""

from __future__ import annotations

import asyncio

import pytest

import dilution.unit_detection as ud


# ── tiny fakes for the async LLM seam ───────────────────────────────────


class _FakeResp:
    def __init__(self, content):
        self.content = content


class _FakeChat:
    """Mimics make_chat(...)'s return: .append() no-ops, async .sample()."""

    def __init__(self, content):
        self._content = content
        self.appended = []

    def append(self, msg):
        self.appended.append(msg)

    async def sample(self):
        return _FakeResp(self._content)


class _RaisingChat:
    def __init__(self):
        self.appended = []

    def append(self, msg):
        self.appended.append(msg)

    async def sample(self):
        raise RuntimeError("boom")


class _FakeClient:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


def _patch_llm(monkeypatch, *, content=None, chat=None, text="some filing text"):
    """Wire the LLM seam. Returns the FakeClient so close() can be asserted."""
    client = _FakeClient()
    monkeypatch.setattr(ud, "make_async_client", lambda: client)
    the_chat = chat if chat is not None else _FakeChat(content)
    monkeypatch.setattr(ud, "make_chat", lambda c, **k: the_chat)
    monkeypatch.setattr(ud, "_load_text_for", lambda acc: text)
    return client


def _filing(acc="acc-1", form="20-F", filing_date="2024-01-01"):
    return {"accession_number": acc, "form": form, "filing_date": filing_date}


# ── _heuristic_ads_ratio (pure) ─────────────────────────────────────────


class TestHeuristicAdsRatio:
    def test_none_input(self):
        assert ud._heuristic_ads_ratio(None) is None

    def test_empty_string(self):
        assert ud._heuristic_ads_ratio("") is None

    def test_no_match(self):
        assert ud._heuristic_ads_ratio("nothing relevant here") is None

    def test_each_ads_represents_basic(self):
        assert ud._heuristic_ads_ratio(
            "Each ADS represents 10 ordinary shares"
        ) == 10.0

    def test_comma_formatted_number(self):
        assert ud._heuristic_ads_ratio(
            "Each ADS represents 1,000 ordinary shares"
        ) == 1000.0

    def test_decimal_fractional_ratio(self):
        assert ud._heuristic_ads_ratio(
            "Each ADS represents 0.5 ordinary shares"
        ) == 0.5

    def test_multiple_ratios_returns_last(self):
        # original 100 then change-to 400; supersession => LAST wins.
        text = ("each ADS represents 100 ordinary shares. Later, "
                "each ADS represents 400 ordinary shares.")
        assert ud._heuristic_ads_ratio(text) == 400.0

    def test_case_insensitive(self):
        assert ud._heuristic_ads_ratio(
            "EACH ADS REPRESENTS 8 ORDINARY SHARES"
        ) == 8.0

    def test_adr_spelling_variant(self):
        # regex `ad[sr]` matches both ADS and ADR.
        assert ud._heuristic_ads_ratio(
            "Each ADR represents 12 ordinary shares"
        ) == 12.0

    def test_one_ads_represents_common_branch(self):
        assert ud._heuristic_ads_ratio("One ADS represents 5 common") == 5.0

    def test_ratio_of_adss_to_ordinary_branch(self):
        assert ud._heuristic_ads_ratio(
            "the ratio of ADSs to ordinary shares is 4:1"
        ) == 4.0

    def test_class_a_ordinary_wording(self):
        assert ud._heuristic_ads_ratio(
            "each ADS represents 3 Class A ordinary shares"
        ) == 3.0

    def test_class_a_bare_share_wording(self):
        # "class\s+[a-z]\s+shares" alternation (no 'ordinary').
        assert ud._heuristic_ads_ratio(
            "each ADS represents 7 Class B shares"
        ) == 7.0

    def test_whitespace_and_newlines_between_tokens(self):
        # regex uses \s+ so embedded newlines / multiple spaces still match.
        text = "each   ADS\nrepresents\t15\nordinary  shares"
        assert ud._heuristic_ads_ratio(text) == 15.0

    def test_malformed_dots_does_not_crash(self):
        # weird dots survive the char class but float() of e.g. "1.2.3"
        # raises ValueError -> swallowed; no candidate, no crash.
        out = ud._heuristic_ads_ratio(
            "each ADS represents 1.2.3 ordinary shares"
        )
        assert out is None

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Each ADS represents 1 ordinary share", 1.0),  # singular 'share'
            ("each ads represents 2 common shares", 2.0),
            ("one ADR represents 0.25 ordinary", 0.25),
        ],
    )
    def test_parametrized_branches(self, text, expected):
        assert ud._heuristic_ads_ratio(text) == expected

    def test_returns_float_type(self):
        out = ud._heuristic_ads_ratio("Each ADS represents 10 ordinary shares")
        assert isinstance(out, float)

    # ── supersession is BRANCH-order, not file-order (subtle) ───────────
    #
    # The docstring claims "Returns the LATEST stated ratio if multiple are
    # present (file order is roughly oldest → newest)" and `candidates[-1]`
    # is taken. But candidates are appended branch-by-branch: ALL branch-1
    # ("each ADS represents") matches, THEN all branch-2 ("one ADS
    # represents"), THEN all branch-3 ("ratio ... is N:1"). So `[-1]` is the
    # last match of the HIGHEST-numbered branch that fired, regardless of
    # where it appears in the document. The "file order" wording only holds
    # WITHIN a single branch. These tests lock that real behavior so a
    # future refactor that (legitimately) switches to true file-order
    # ordering is forced to update them deliberately.

    def test_within_branch1_last_wins_regardless_of_position(self):
        # two branch-1 matches; the textually-last (400) wins.
        text = ("each ADS represents 100 ordinary shares ... "
                "each ADS represents 400 ordinary shares")
        assert ud._heuristic_ads_ratio(text) == 400.0

    def test_within_branch2_multiple_last_wins(self):
        # branch-2 ("one ADS represents") was previously only tested with a
        # single match; lock the within-branch supersession here too.
        text = "one ADS represents 3 common ... one ADS represents 7 common"
        assert ud._heuristic_ads_ratio(text) == 7.0

    def test_branch3_wins_over_earlier_branch1_in_file(self):
        # branch-1 ("each ADS ... 10") appears FIRST in the text, branch-3
        # ("ratio is 4:1") appears LAST → file-order would say 4, branch-
        # order ALSO says 4 here. Symmetric to the next test.
        text = "Each ADS represents 10 ordinary shares. the ratio is 4:1."
        assert ud._heuristic_ads_ratio(text) == 4.0

    def test_branch3_wins_even_when_earlier_in_file(self):
        # KEY divergence: branch-3 ("ratio is 4:1") appears FIRST, branch-1
        # ("each ADS ... 10") appears LAST. True file-order would return 10;
        # the code returns 4 because branch-3 candidates are appended AFTER
        # branch-1 candidates. Locks the branch-order-over-file-order rule.
        text = "the ratio is 4:1. Each ADS represents 10 ordinary shares."
        assert ud._heuristic_ads_ratio(text) == 4.0

    def test_branch2_appended_after_branch1(self):
        # branch-1 (10) appears first, branch-2 (3) second → both file-order
        # and branch-order agree on 3, but this pins that branch-2 outranks
        # branch-1 when both fire.
        text = "Each ADS represents 10 ordinary shares. one ADS represents 3 common."
        assert ud._heuristic_ads_ratio(text) == 3.0

    def test_branch2_outranks_branch1_even_when_earlier_in_file(self):
        # branch-2 (3) appears FIRST in the file, branch-1 (10) appears LAST.
        # File-order would return 10; branch-order returns 3 (branch-2 < ...
        # wait: branch-2 is appended after branch-1, so branch-2 wins).
        text = "one ADS represents 3 common. Each ADS represents 10 ordinary shares."
        assert ud._heuristic_ads_ratio(text) == 3.0

    # ── branch-3 directionality + loose separator (lock the surprises) ──

    def test_branch3_freeform_supersession_prose_still_falls_to_llm(self):
        # Branch-3 now captures the "ratio is 1:N" form (see
        # test_branch3_1_colon_N_capture), but a free-form supersession
        # sentence with no "ratio is/of N" structure ("...changed from 1:100
        # to 1:400") still has no anchor for the heuristic and falls through
        # to the LLM. Locks that boundary.
        assert ud._heuristic_ads_ratio(
            "the ADS-to-ordinary ratio has been changed from 1:100 to 1:400"
        ) is None

    def test_branch3_1_colon_N_capture(self):
        # FIXED (was bug B#9): the docstring's 1:N direction is now captured by
        # the heuristic (the regex matches either 1:N or N:1).
        assert ud._heuristic_ads_ratio("the ratio is 1:400") == 400.0

    @pytest.mark.parametrize("distractor", [
        "the leverage ratio is 1 5",
        "the coverage ratio is 1 3 of EBITDA",
        "the current ratio is 1 8 times",
    ])
    def test_branch3_1_space_N_distractor_not_captured(self, distractor):
        # GUARD (B#9 follow-up): the 1:N arm requires a LITERAL colon, so a
        # space-separated "ratio is 1 N" financial-prose distractor is NOT
        # captured (it would otherwise yield a bogus ADS ratio that, being
        # appended last, overrides the correct branch-1/2 value).
        assert ud._heuristic_ads_ratio(distractor) is None

    def test_branch3_1_space_N_distractor_does_not_override_real_ratio(self):
        # End-to-end: a correct branch-1 ADS ratio (8) must survive even when a
        # distractor "ratio is 1 N" sentence follows it in the same document.
        doc = ("Each ADS represents 8 ordinary shares. "
               "Separately, the coverage ratio is 1 3 of EBITDA.")
        assert ud._heuristic_ads_ratio(doc) == 8.0

    def test_branch3_optional_leading_1_colon(self):
        # The "(?:1\s*[:\s]\s*)?" prefix is optional, so "1:8:1" → the leading
        # "1:" is consumed and "8:1" matches → 8.
        assert ud._heuristic_ads_ratio("the ratio is 1:8:1") == 8.0

    def test_branch3_space_separator_is_loose(self):
        # The separator char class is [:\s], so a plain space works in place
        # of the colon: "the ratio is 5 1" matches with N=5. This is a loose
        # (arguably over-eager) match — locked here so it can't silently
        # change.
        assert ud._heuristic_ads_ratio("the ratio is 5 1") == 5.0

    def test_branch3_spaced_colon(self):
        assert ud._heuristic_ads_ratio(
            "ratio of ADSs to ordinary shares is 4 : 1"
        ) == 4.0

    def test_one_ads_ordinary_singular(self):
        # branch-2 "one ad[sr] represents N (ordinary|common)" — no trailing
        # "shares" needed; "0.25 ordinary" matches.
        assert ud._heuristic_ads_ratio("one ADR represents 0.25 ordinary") == 0.25

    # ── near-miss NON-matches (lock that the heuristic is not over-eager) ──

    def test_unqualified_shares_does_not_match(self):
        # "represents N shares" with NO ordinary/common/class qualifier is
        # NOT a recognized ADS-ratio phrasing → None (would otherwise be a
        # false positive). Distinguishes the heuristic from a bare number grab.
        assert ud._heuristic_ads_ratio("each ADS represents 10 shares") is None

    def test_preferred_shares_not_in_alternation(self):
        # "preferred" is deliberately outside the (ordinary|common|class …)
        # alternation; an ADS over preferred shares is not the ordinary-share
        # ratio this heuristic targets → None.
        assert ud._heuristic_ads_ratio(
            "each ADS represents 10 preferred shares"
        ) is None

    def test_leading_minus_breaks_the_match(self):
        # The number char class is [\d,\.] with no '-', and the preceding
        # token is \s+, so "represents -5 ordinary" does NOT match (the '-'
        # sits between the space and the digit). No negative ratio leaks in.
        assert ud._heuristic_ads_ratio(
            "each ADS represents -5 ordinary shares"
        ) is None

    def test_represents_with_no_number(self):
        # "represents ordinary shares" — the [\d,\.]+ group requires at least
        # one char, so a missing count means no match (no crash, no candidate).
        assert ud._heuristic_ads_ratio(
            "each ADS represents ordinary shares"
        ) is None


# ── _is_fpi_from_filings (db) ───────────────────────────────────────────


class TestIsFpiFromFilings:
    def test_no_filings(self, temp_db):
        assert ud._is_fpi_from_filings(123) is False

    @pytest.mark.parametrize("form", ["20-F", "20-F/A", "40-F", "40-F/A"])
    def test_fpi_forms_match(self, temp_db, form):
        temp_db.add_filing("a1", 123, form=form)
        assert ud._is_fpi_from_filings(123) is True

    def test_amendment_via_wildcard(self, temp_db):
        # explicit assert that the % wildcard catches amendments.
        temp_db.add_filing("a1", 123, form="20-F/A")
        assert ud._is_fpi_from_filings(123) is True

    @pytest.mark.parametrize("form", ["10-K", "8-K", "10-Q", "S-1"])
    def test_non_fpi_forms_dont_match(self, temp_db, form):
        temp_db.add_filing("a1", 123, form=form)
        assert ud._is_fpi_from_filings(123) is False

    def test_filings_for_different_cik(self, temp_db):
        temp_db.add_filing("a1", 999, form="20-F")
        assert ud._is_fpi_from_filings(123) is False

    def test_420f_substring_does_not_match(self, temp_db):
        # LIKE '20-F%' anchors at the start, so '420-F' must NOT match.
        temp_db.add_filing("a1", 123, form="420-F")
        assert ud._is_fpi_from_filings(123) is False

    def test_20fr_prefix_matches(self, temp_db):
        # '20-FR' matches LIKE '20-F%' (prefix) even though it is NOT in
        # the exact FPI_ANNUAL_FORMS tuple — divergence with _latest.
        temp_db.add_filing("a1", 123, form="20-FR")
        assert ud._is_fpi_from_filings(123) is True


# ── _latest_annual_filing (db) ──────────────────────────────────────────


class TestLatestAnnualFiling:
    def test_none_when_no_annual(self, temp_db):
        assert ud._latest_annual_filing(123) is None

    def test_only_non_annual_returns_none(self, temp_db):
        temp_db.add_filing("a1", 123, form="10-K")
        assert ud._latest_annual_filing(123) is None

    def test_returns_latest_by_date(self, temp_db):
        temp_db.add_filing("old", 123, form="20-F", filing_date="2023-01-01")
        temp_db.add_filing("new", 123, form="20-F", filing_date="2025-01-01")
        out = ud._latest_annual_filing(123)
        assert out is not None
        assert out["accession_number"] == "new"

    def test_mixed_forms_latest_wins(self, temp_db):
        temp_db.add_filing("f20", 123, form="20-F", filing_date="2024-01-01")
        temp_db.add_filing("f40", 123, form="40-F", filing_date="2025-06-01")
        out = ud._latest_annual_filing(123)
        assert out["accession_number"] == "f40"
        assert out["form"] == "40-F"

    def test_amendment_form_eligible(self, temp_db):
        temp_db.add_filing("amd", 123, form="20-F/A", filing_date="2025-01-01")
        out = ud._latest_annual_filing(123)
        assert out is not None
        assert out["form"] == "20-F/A"

    def test_20fr_disagrees_with_is_fpi(self, temp_db):
        # By design: _is_fpi (LIKE) sees it; _latest (IN-list) does not.
        temp_db.add_filing("a1", 123, form="20-FR", filing_date="2025-01-01")
        assert ud._is_fpi_from_filings(123) is True
        assert ud._latest_annual_filing(123) is None

    def test_wrong_cik_filtered_out(self, temp_db):
        temp_db.add_filing("a1", 999, form="20-F")
        assert ud._latest_annual_filing(123) is None

    def test_returned_dict_keys(self, temp_db):
        temp_db.add_filing("a1", 123, form="20-F", filing_date="2025-01-01",
                           primary_doc="doc.htm")
        out = ud._latest_annual_filing(123)
        assert set(out.keys()) == {
            "accession_number", "form", "filing_date", "primary_doc"
        }
        assert out["primary_doc"] == "doc.htm"


# ── _load_text_for (db + fetch fallback) ────────────────────────────────


def _stage_raw(temp_db, acc, doc_name, doc_type, content, *, with_filing=True):
    if with_filing:
        # FK references dilution_filings(accession_number); stage once.
        existing = temp_db.execute(
            "SELECT 1 FROM dilution_filings WHERE accession_number=?", (acc,)
        )
        if not existing:
            temp_db.add_filing(acc, 1, form="20-F")
    temp_db.execute(
        """INSERT INTO dilution_raw
             (accession_number, doc_name, doc_type, content_md, downloaded_at)
           VALUES (?,?,?,?,?)""",
        (acc, doc_name, doc_type, content, "2025-01-01"),
    )


class TestLoadTextFor:
    def test_no_rows_fetch_returns_none(self, temp_db, monkeypatch):
        # No cached rows -> SEC-fetch path; mock get_by_accession_number=None.
        monkeypatch.setattr(ud, "set_identity", lambda *a, **k: None)
        monkeypatch.setattr(ud, "get_by_accession_number", lambda acc: None)
        assert ud._load_text_for("missing-acc") is None

    def test_ex_skip_beats_length_sort(self, temp_db, monkeypatch):
        # Longer EX-99.1 row exists, but the non-EX NT 20-F is chosen.
        monkeypatch.setattr(ud, "normalize_filing_text", lambda s: s)
        _stage_raw(temp_db, "t1", "long.htm", "EX-99.1", "X" * 5000)
        _stage_raw(temp_db, "t1", "short.htm", "NT 20-F", "primary")
        assert ud._load_text_for("t1") == "primary"

    def test_all_ex_falls_back_to_longest(self, temp_db, monkeypatch):
        monkeypatch.setattr(ud, "normalize_filing_text", lambda s: s)
        _stage_raw(temp_db, "t2", "d1.htm", "EX-99.1", "longerexhibit")
        _stage_raw(temp_db, "t2", "d2.htm", "ex-10.1", "short")
        # ORDER BY len DESC => rows[0] is the longer one.
        assert ud._load_text_for("t2") == "longerexhibit"

    def test_lowercase_ex_is_skipped(self, temp_db, monkeypatch):
        # doc_type uppercased before startswith('EX-') => lowercase ex- skipped.
        monkeypatch.setattr(ud, "normalize_filing_text", lambda s: s)
        _stage_raw(temp_db, "t3", "ex.htm", "ex-99", "X" * 100)
        _stage_raw(temp_db, "t3", "main.htm", "NT 20-F", "real-primary")
        assert ud._load_text_for("t3") == "real-primary"

    def test_null_doc_type_treated_as_primary(self, temp_db, monkeypatch):
        monkeypatch.setattr(ud, "normalize_filing_text", lambda s: s)
        _stage_raw(temp_db, "t4", "d.htm", None, "nulltype")
        assert ud._load_text_for("t4") == "nulltype"

    def test_truncation_boundary_equal_not_truncated(self, temp_db, monkeypatch):
        monkeypatch.setattr(ud, "normalize_filing_text", lambda s: s)
        _stage_raw(temp_db, "m1", "d.htm", "NT 20-F", "abcde")  # len 5
        out = ud._load_text_for("m1", max_chars=5)
        assert out == "abcde"  # strict > => not truncated at exactly max

    def test_truncation_boundary_over_truncated(self, temp_db, monkeypatch):
        monkeypatch.setattr(ud, "normalize_filing_text", lambda s: s)
        _stage_raw(temp_db, "m2", "d.htm", "NT 20-F", "abcdef")  # len 6
        out = ud._load_text_for("m2", max_chars=5)
        assert out == "abcde"
        assert len(out) == 5

    def test_small_max_chars(self, temp_db, monkeypatch):
        monkeypatch.setattr(ud, "normalize_filing_text", lambda s: s)
        _stage_raw(temp_db, "m5", "d.htm", "NT 20-F", "abcdefghij")
        assert ud._load_text_for("m5", max_chars=3) == "abc"

    def test_normalize_applied_before_cap(self, temp_db):
        # normalize collapses runs of 3+ spaces to 2 + strips ZWSP.
        # Use real normalize_filing_text: stage padded content and assert
        # the collapse happened (content shrinks vs raw).
        padded = "each​ ADS    represents     10 ordinary shares"
        _stage_raw(temp_db, "m6", "d.htm", "NT 20-F", padded)
        out = ud._load_text_for("m6")
        # ZWSP removed; "    "(4) and "     "(5) runs collapsed to 2 spaces.
        assert "​" not in out
        assert "    " not in out  # no run of 3+ spaces survives
        assert "represents  10" in out  # 5-space run -> 2 spaces

    def test_empty_primary_md_falls_to_fetch(self, temp_db, monkeypatch):
        # Empty content_md primary leaves `md` falsy -> fetch path.
        monkeypatch.setattr(ud, "set_identity", lambda *a, **k: None)
        monkeypatch.setattr(ud, "get_by_accession_number", lambda acc: None)
        _stage_raw(temp_db, "m3", "d.htm", "NT 20-F", "")
        assert ud._load_text_for("m3") is None

    def test_fetch_lookup_raises_returns_none(self, temp_db, monkeypatch):
        # get_by_accession_number raises -> caught, returns None.
        monkeypatch.setattr(ud, "set_identity", lambda *a, **k: None)

        def _boom(acc):
            raise RuntimeError("network down")

        monkeypatch.setattr(ud, "get_by_accession_number", _boom)
        out = ud._load_text_for("no-rows-acc")
        assert out is None

    def test_fetch_success_path(self, temp_db, monkeypatch):
        # No cached rows; fake filing with .markdown(); identity is no-op.
        monkeypatch.setattr(ud, "set_identity", lambda *a, **k: None)
        monkeypatch.setattr(ud, "normalize_filing_text", lambda s: s)

        class _FakeFiling:
            def markdown(self):
                return "Each ADS represents 9 ordinary shares"

        monkeypatch.setattr(ud, "get_by_accession_number",
                            lambda acc: _FakeFiling())
        out = ud._load_text_for("fetched-acc")
        assert out == "Each ADS represents 9 ordinary shares"

    def test_whitespace_only_primary_is_kept_not_fetched(self, temp_db, monkeypatch):
        # SUBTLE: only an *empty* string is falsy. A whitespace-only primary
        # survives `if not md` (it is truthy), real normalize_filing_text does
        # NOT strip leading/trailing whitespace to empty, so it is returned —
        # the fetch path is NOT taken. Guard against a regression that would
        # silently start hitting the network on whitespace docs.
        def _boom(acc):  # would fire only if the fetch path were taken
            raise AssertionError("fetch path taken for whitespace-only primary")

        monkeypatch.setattr(ud, "get_by_accession_number", _boom)
        _stage_raw(temp_db, "ws1", "d.htm", "NT 20-F", "   \n\t  ")
        out = ud._load_text_for("ws1")
        assert out is not None
        assert out.strip() == ""  # whitespace preserved, not fetched

    def test_fetch_markdown_raises_returns_none(self, temp_db, monkeypatch):
        monkeypatch.setattr(ud, "set_identity", lambda *a, **k: None)

        class _FakeFiling:
            def markdown(self):
                raise ValueError("bad doc")

        monkeypatch.setattr(ud, "get_by_accession_number",
                            lambda acc: _FakeFiling())
        assert ud._load_text_for("md-raise-acc") is None


# ── _llm_ads_ratio (async, io_mockable) ─────────────────────────────────


class TestLlmAdsRatio:
    def test_none_text_short_circuits(self, monkeypatch):
        # _load_text_for -> None: return None before any LLM construction.
        sentinel = {"called": False}

        def _client():
            sentinel["called"] = True
            return _FakeClient()

        monkeypatch.setattr(ud, "make_async_client", _client)
        monkeypatch.setattr(ud, "_load_text_for", lambda acc: None)
        out = asyncio.run(ud._llm_ads_ratio(_filing()))
        assert out is None
        assert sentinel["called"] is False  # client never built

    def test_valid_json(self, monkeypatch):
        client = _patch_llm(monkeypatch, content='{"ads_ratio": 8}')
        out = asyncio.run(ud._llm_ads_ratio(_filing()))
        assert out == 8.0
        assert client.closed is True

    def test_fenced_json(self, monkeypatch):
        _patch_llm(monkeypatch, content='```json\n{"ads_ratio": 5}\n```')
        out = asyncio.run(ud._llm_ads_ratio(_filing()))
        assert out == 5.0

    def test_fenced_json_no_lang(self, monkeypatch):
        _patch_llm(monkeypatch, content='```\n{"ads_ratio": 6}\n```')
        out = asyncio.run(ud._llm_ads_ratio(_filing()))
        assert out == 6.0

    def test_non_json_garbage(self, monkeypatch, caplog):
        _patch_llm(monkeypatch, content="this is not json")
        with caplog.at_level("WARNING"):
            out = asyncio.run(ud._llm_ads_ratio(_filing()))
        assert out is None
        assert any("non-JSON" in r.message for r in caplog.records)

    def test_ads_ratio_null(self, monkeypatch):
        _patch_llm(monkeypatch, content='{"ads_ratio": null}')
        assert asyncio.run(ud._llm_ads_ratio(_filing())) is None

    def test_ads_ratio_missing_key(self, monkeypatch):
        _patch_llm(monkeypatch, content='{"underlying_unit": "ordinary"}')
        assert asyncio.run(ud._llm_ads_ratio(_filing())) is None

    def test_non_numeric_string(self, monkeypatch):
        _patch_llm(monkeypatch, content='{"ads_ratio": "eight"}')
        assert asyncio.run(ud._llm_ads_ratio(_filing())) is None

    def test_numeric_string_coerced(self, monkeypatch):
        _patch_llm(monkeypatch, content='{"ads_ratio": "400"}')
        assert asyncio.run(ud._llm_ads_ratio(_filing())) == 400.0

    @pytest.mark.parametrize("val", [0, -1, -100.5])
    def test_ratio_at_or_below_zero_rejected(self, monkeypatch, val):
        _patch_llm(monkeypatch, content=f'{{"ads_ratio": {val}}}')
        assert asyncio.run(ud._llm_ads_ratio(_filing())) is None

    def test_ratio_just_over_max_rejected(self, monkeypatch):
        _patch_llm(monkeypatch, content='{"ads_ratio": 100001}')
        assert asyncio.run(ud._llm_ads_ratio(_filing())) is None

    def test_ratio_exactly_max_accepted(self, monkeypatch):
        # boundary: condition is `> 100_000`, so exactly 100_000 is kept.
        _patch_llm(monkeypatch, content='{"ads_ratio": 100000}')
        assert asyncio.run(ud._llm_ads_ratio(_filing())) == 100000.0

    def test_smallest_positive_accepted(self, monkeypatch):
        _patch_llm(monkeypatch, content='{"ads_ratio": 0.001}')
        assert asyncio.run(ud._llm_ads_ratio(_filing())) == pytest.approx(0.001)

    def test_client_closed_even_when_sample_raises(self, monkeypatch):
        client = _FakeClient()
        monkeypatch.setattr(ud, "make_async_client", lambda: client)
        monkeypatch.setattr(ud, "make_chat", lambda c, **k: _RaisingChat())
        monkeypatch.setattr(ud, "_load_text_for", lambda acc: "text")
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(ud._llm_ads_ratio(_filing()))
        assert client.closed is True  # finally awaited close()

    def test_empty_response_content(self, monkeypatch):
        # resp.content None -> "" -> not JSON -> None.
        _patch_llm(monkeypatch, content=None)
        assert asyncio.run(ud._llm_ads_ratio(_filing())) is None

    def test_infinity_rejected_by_upper_bound(self, monkeypatch, caplog):
        # stdlib json.loads accepts the non-standard `Infinity` token ->
        # float('inf'). inf > 100_000 is True -> implausible -> None (+warn).
        _patch_llm(monkeypatch, content='{"ads_ratio": Infinity}')
        with caplog.at_level("WARNING"):
            out = asyncio.run(ud._llm_ads_ratio(_filing()))
        assert out is None
        assert any("implausible" in r.message for r in caplog.records)

    def test_bool_true_coerces_to_one(self, monkeypatch):
        # JSON `true` -> float(True) == 1.0, which is in-bounds (0 < 1 <= 1e5)
        # and so is ACCEPTED as a ratio of 1.0. Documents the (lenient) bool
        # coercion path.
        _patch_llm(monkeypatch, content='{"ads_ratio": true}')
        assert asyncio.run(ud._llm_ads_ratio(_filing())) == 1.0

    def test_nan_is_rejected(self, monkeypatch):
        # FIXED (was bug B#8): NaN is non-finite and now rejected like Infinity
        # (math.isfinite guard) instead of flowing into unit_preamble().
        _patch_llm(monkeypatch, content='{"ads_ratio": NaN}')
        assert asyncio.run(ud._llm_ads_ratio(_filing())) is None

    def test_loaded_text_actually_flows_into_prompt(self, monkeypatch):
        # Defeats the over-mock trap: _FakeChat.append no-ops, so a valid-JSON
        # test alone can't prove the function ever USED the loaded text. Here
        # we capture the appended user message and assert the _load_text_for
        # text + the filing form/date are interpolated into ADS_RATIO_PROMPT.
        chat = _FakeChat('{"ads_ratio": 7}')
        _patch_llm(monkeypatch, chat=chat, text="UNIQUE-FILING-BODY-XYZ")
        out = asyncio.run(
            ud._llm_ads_ratio(_filing(form="40-F", filing_date="2023-09-09"))
        )
        assert out == 7.0
        # append() got the system message then the user prompt.
        assert len(chat.appended) == 2
        kind, sys_msg = chat.appended[0]
        assert kind == "system"
        user_kind, user_msg = chat.appended[1]
        assert user_kind == "user"
        assert "UNIQUE-FILING-BODY-XYZ" in user_msg
        assert "40-F" in user_msg
        assert "2023-09-09" in user_msg

    def test_fenced_json_with_trailing_text_after_close(self, monkeypatch):
        # The fence-strip regex only removes a closing ```\n at the very end
        # ("\n```$"). If anything follows the closing fence the strip leaves
        # the fence in place -> json.loads fails -> None. Locks that the
        # salvage is intentionally narrow.
        _patch_llm(monkeypatch, content='```json\n{"ads_ratio": 5}\n```\ntrailing')
        assert asyncio.run(ud._llm_ads_ratio(_filing())) is None

    def test_leading_whitespace_then_fence_is_stripped(self, monkeypatch):
        # raw = (content or "").strip() runs BEFORE the startswith('```')
        # check, so surrounding whitespace around the fenced block is fine.
        _patch_llm(monkeypatch, content='   \n```json\n{"ads_ratio": 6}\n```  \n')
        assert asyncio.run(ud._llm_ads_ratio(_filing())) == 6.0


# ── populate_company_unit (orchestrator, db) ────────────────────────────


class TestPopulateCompanyUnit:
    def _mark_detected(self, temp_db, cik, at="2025-12-31T00:00:00Z"):
        temp_db.execute(
            "UPDATE dilution_company SET unit_detected_at=? WHERE cik=?",
            (at, cik),
        )

    def test_already_detected_returns_cached_no_recompute(
        self, temp_db, monkeypatch
    ):
        temp_db.add_company(700, is_fpi=1, ads_ratio=8.0)
        self._mark_detected(temp_db, 700)

        # Prove _is_fpi_from_filings is NOT called on the cached path.
        def _should_not_run(cik):
            raise AssertionError("recomputed despite cached result")

        monkeypatch.setattr(ud, "_is_fpi_from_filings", _should_not_run)
        # _persist must NOT run on the cached path either — guard against a
        # re-stamp of unit_detected_at (the early return precedes _persist).
        def _no_persist(*a, **k):
            raise AssertionError("_persist called on cached path")

        monkeypatch.setattr(ud, "_persist", _no_persist)
        out = ud.populate_company_unit(700, force=False)
        assert out == {"is_fpi": 1, "ads_ratio": 8.0, "reporting_unit": "ads"}
        # DB row untouched: unit_detected_at is exactly what we staged.
        row = temp_db.execute(
            "SELECT unit_detected_at, is_fpi, ads_ratio "
            "FROM dilution_company WHERE cik=700"
        )[0]
        assert row["unit_detected_at"] == "2025-12-31T00:00:00Z"
        assert row["is_fpi"] == 1
        assert row["ads_ratio"] == 8.0

    def test_already_detected_force_recomputes(self, temp_db, monkeypatch):
        # was is_fpi=1; force re-run with no filings flips to non-FPI.
        temp_db.add_company(701, is_fpi=1, ads_ratio=8.0)
        self._mark_detected(temp_db, 701)
        out = ud.populate_company_unit(701, force=True)
        assert out == {"is_fpi": 0, "ads_ratio": None,
                       "reporting_unit": "common"}
        row = temp_db.execute(
            "SELECT is_fpi, ads_ratio FROM dilution_company WHERE cik=701"
        )[0]
        assert row["is_fpi"] == 0
        assert row["ads_ratio"] is None

    def test_cached_is_fpi_zero_is_common(self, temp_db):
        # Cached is_fpi=0 -> int(0 or 0)=0, reporting_unit 'common'.
        # NOTE: the `or 0` guard in populate_company_unit defends against a
        # NULL is_fpi, but the schema declares the column INTEGER NOT NULL
        # DEFAULT 0, so a NULL can never actually be stored — the coercion
        # is effectively defensive/unreachable. We exercise the reachable
        # is_fpi=0 case here.
        temp_db.add_company(702, is_fpi=0, ads_ratio=None)
        self._mark_detected(temp_db, 702)
        out = ud.populate_company_unit(702, force=False)
        assert out["is_fpi"] == 0
        assert out["ads_ratio"] is None
        assert out["reporting_unit"] == "common"

    def test_non_fpi_persists_zero(self, temp_db, monkeypatch):
        monkeypatch.setattr(ud, "now_iso", lambda: "2026-06-10T00:00:00Z")
        temp_db.add_company(703)
        out = ud.populate_company_unit(703)
        assert out == {"is_fpi": 0, "ads_ratio": None,
                       "reporting_unit": "common"}
        row = temp_db.execute(
            "SELECT is_fpi, ads_ratio, unit_detected_at "
            "FROM dilution_company WHERE cik=703"
        )[0]
        assert row["is_fpi"] == 0
        assert row["ads_ratio"] is None
        assert row["unit_detected_at"] == "2026-06-10T00:00:00Z"

    def test_fpi_no_annual_filing_persists_one_null(self, temp_db):
        # _is_fpi True via 20-FR (LIKE), but _latest (IN-list) returns None.
        temp_db.add_company(704)
        temp_db.add_filing("a1", 704, form="20-FR", filing_date="2025-01-01")
        out = ud.populate_company_unit(704)
        assert out == {"is_fpi": 1, "ads_ratio": None,
                       "reporting_unit": "ads"}
        row = temp_db.execute(
            "SELECT is_fpi, ads_ratio FROM dilution_company WHERE cik=704"
        )[0]
        assert row["is_fpi"] == 1
        assert row["ads_ratio"] is None

    def test_fpi_heuristic_hits_skips_llm(self, temp_db, monkeypatch):
        temp_db.add_company(705)
        temp_db.add_filing("acc-h", 705, form="20-F", filing_date="2025-01-01")
        monkeypatch.setattr(ud, "_load_text_for", lambda acc: "ignored")
        monkeypatch.setattr(ud, "_heuristic_ads_ratio", lambda t: 4.0)

        async def _must_not_run(filing):
            raise AssertionError("LLM invoked despite heuristic hit")

        monkeypatch.setattr(ud, "_llm_ads_ratio", _must_not_run)
        out = ud.populate_company_unit(705)
        assert out == {"is_fpi": 1, "ads_ratio": 4.0, "reporting_unit": "ads"}
        row = temp_db.execute(
            "SELECT ads_ratio FROM dilution_company WHERE cik=705"
        )[0]
        assert row["ads_ratio"] == 4.0

    def test_fpi_heuristic_miss_uses_llm(self, temp_db, monkeypatch):
        temp_db.add_company(706)
        temp_db.add_filing("acc-l", 706, form="20-F", filing_date="2025-01-01")
        monkeypatch.setattr(ud, "_load_text_for", lambda acc: "ignored")
        monkeypatch.setattr(ud, "_heuristic_ads_ratio", lambda t: None)

        async def _fake_llm(filing):
            return 13.0

        monkeypatch.setattr(ud, "_llm_ads_ratio", _fake_llm)
        out = ud.populate_company_unit(706)
        assert out == {"is_fpi": 1, "ads_ratio": 13.0, "reporting_unit": "ads"}
        row = temp_db.execute(
            "SELECT ads_ratio FROM dilution_company WHERE cik=706"
        )[0]
        assert row["ads_ratio"] == 13.0

    def test_fpi_heuristic_and_llm_both_miss_persists_one_null(
        self, temp_db, monkeypatch
    ):
        # FPI with an annual filing, but neither the heuristic nor the LLM
        # finds a ratio → persists (is_fpi=1, ads_ratio=None) and reports ads.
        temp_db.add_company(709)
        temp_db.add_filing("acc-n", 709, form="20-F", filing_date="2025-01-01")
        monkeypatch.setattr(ud, "_load_text_for", lambda acc: "no ratio here")
        monkeypatch.setattr(ud, "_heuristic_ads_ratio", lambda t: None)

        async def _fake_llm(filing):
            return None

        monkeypatch.setattr(ud, "_llm_ads_ratio", _fake_llm)
        out = ud.populate_company_unit(709)
        assert out == {"is_fpi": 1, "ads_ratio": None, "reporting_unit": "ads"}
        row = temp_db.execute(
            "SELECT is_fpi, ads_ratio FROM dilution_company WHERE cik=709"
        )[0]
        assert row["is_fpi"] == 1
        assert row["ads_ratio"] is None

    def test_heuristic_zero_falls_through_to_llm(self, temp_db, monkeypatch):
        # Heuristic 0.0 is falsy -> `if ratio:` is False -> LLM path taken.
        temp_db.add_company(707)
        temp_db.add_filing("acc-z", 707, form="40-F", filing_date="2025-01-01")
        monkeypatch.setattr(ud, "_load_text_for", lambda acc: "ignored")
        monkeypatch.setattr(ud, "_heuristic_ads_ratio", lambda t: 0.0)

        called = {"llm": False}

        async def _fake_llm(filing):
            called["llm"] = True
            return 20.0

        monkeypatch.setattr(ud, "_llm_ads_ratio", _fake_llm)
        out = ud.populate_company_unit(707)
        assert called["llm"] is True
        assert out["ads_ratio"] == 20.0

    def test_no_company_row_is_silent_noop(self, temp_db, monkeypatch):
        # _persist is UPDATE-only; with no company row the write is a no-op
        # but the function still returns the computed dict.
        out = ud.populate_company_unit(99999)
        assert out == {"is_fpi": 0, "ads_ratio": None,
                       "reporting_unit": "common"}
        rows = temp_db.execute(
            "SELECT * FROM dilution_company WHERE cik=99999"
        )
        assert rows == []  # nothing persisted (no pre-existing row)

    def test_fpi_with_heuristic_uses_real_text_path(self, temp_db):
        # End-to-end through the real heuristic (no LLM): stage raw text
        # containing a ratio phrase and assert it is extracted + persisted.
        temp_db.add_company(708)
        temp_db.add_filing("acc-e2e", 708, form="20-F",
                           filing_date="2025-01-01")
        _stage_raw(temp_db, "acc-e2e", "main.htm", "NT 20-F",
                   "Each ADS represents 25 ordinary shares.",
                   with_filing=False)  # filing already staged above
        out = ud.populate_company_unit(708)
        assert out == {"is_fpi": 1, "ads_ratio": 25.0,
                       "reporting_unit": "ads"}
