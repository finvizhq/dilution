"""Unit tests for dilution/ledger/_exhibit_provisions.py.

The module is a deterministic regex classifier for the EX-FILING FEES
exhibit (SEC Rule 457(x) codes) plus a prompt-hint renderer:

  * ``_classify_fee_table_text`` — PURE two-pass regex classification.
  * ``classify_fee_table``       — DB-backed thin selector + delegate.
  * ``format_fee_table_for_prompt`` — PURE verdict -> markdown hint.

No network / LLM / vendor seams exist in this module, so nothing is
monkeypatched here. DB-backed tests use the autouse ``temp_db`` fixture
and stage rows directly into ``dilution_raw`` (no add_raw helper exists).
"""

from __future__ import annotations

import pytest

from dilution.ledger._exhibit_provisions import (
    FEE_TABLE_FORMS,
    classify_fee_table,
    format_fee_table_for_prompt,
    _classify_fee_table_text,
)


# Helper to stage an EX-FILING FEES raw doc for an accession. The FK on
# dilution_raw -> dilution_filings(accession_number) requires the filing
# row to exist first.
def _stage_raw(temp_db, accession, content_md, *, doc_type="EX-FILING FEES",
               cik=1, form="S-3", doc_name="fees.htm"):
    temp_db.add_filing(accession, cik, form=form, filing_date="2026-01-01")
    temp_db.execute(
        """INSERT INTO dilution_raw
             (accession_number, doc_name, doc_type, content_md, downloaded_at)
           VALUES (?,?,?,?,?)""",
        (accession, doc_name, doc_type, content_md, "2026-01-01"),
    )


class TestClassifyFeeTableText:
    """Pure two-pass regex classifier."""

    # ── empty / no-signal branches ──────────────────────────────────
    def test_empty_string_is_unknown(self):
        assert _classify_fee_table_text("") == "unknown"

    def test_falsy_none_guard(self):
        # Signature says str, but `if not text` guards. Callers pass ''
        # for None content; None itself is also caught by the falsy guard.
        assert _classify_fee_table_text(None) == "unknown"  # type: ignore[arg-type]

    def test_no_457_code_anywhere_is_unknown(self):
        assert _classify_fee_table_text(
            "| Security Type | Amount | Fee |\n| Common | 100 | 1.00 |"
        ) == "unknown"

    def test_prose_without_any_rule_code_is_unknown(self):
        assert _classify_fee_table_text(
            "This registration statement registers shares of common stock."
        ) == "unknown"

    # ── cell-anchored primary ───────────────────────────────────────
    def test_cell_rule_457o_is_primary(self):
        assert _classify_fee_table_text("| Rule 457(o) |") == "primary"

    def test_cell_457o_without_rule_prefix_is_primary(self):
        # "Rule " is an optional group.
        assert _classify_fee_table_text("| 457(o) |") == "primary"

    # ── cell-anchored resale ────────────────────────────────────────
    def test_cell_457c_without_rule_prefix_is_resale(self):
        assert _classify_fee_table_text("| 457(c) |") == "resale"

    def test_cell_rule_457g_is_resale(self):
        assert _classify_fee_table_text("| Rule 457(g) |") == "resale"

    # ── cell-anchored mixed ─────────────────────────────────────────
    def test_cell_both_primary_and_resale_is_mixed(self):
        assert _classify_fee_table_text("| 457(o) |\n| 457(c) |") == "mixed"

    # ── case-insensitivity (re.IGNORECASE) ──────────────────────────
    def test_cell_uppercase_rule_letter_o_is_primary(self):
        assert _classify_fee_table_text("| rule 457(O) |") == "primary"

    def test_cell_uppercase_letter_c_is_resale(self):
        assert _classify_fee_table_text("| 457(C) |") == "resale"

    # ── whitespace inside parens ([\s\n] / \s* in regex) ────────────
    def test_cell_inner_spaces_around_letter_primary(self):
        assert _classify_fee_table_text("| 457( o ) |") == "primary"

    def test_cell_newlines_around_letter_resale(self):
        assert _classify_fee_table_text("| 457(\nc\n) |") == "resale"

    def test_pipe_with_newline_before_code(self):
        # _RE_*_CELL allows [\s\n]* between the pipe and the code.
        assert _classify_fee_table_text("|\n457(o) |") == "primary"

    # ── cell pass short-circuits before loose pass ──────────────────
    def test_cell_primary_beats_loose_resale(self):
        # Cell-anchored primary present AND loose-only resale (no pipe)
        # in prose -> cell pass returns 'primary' before loose runs.
        # This is the documented gotcha: NOT 'mixed'.
        text = "| Rule 457(o) |\nSee also resale registration under 457(c)."
        assert _classify_fee_table_text(text) == "primary"

    def test_cell_resale_beats_loose_primary(self):
        # Symmetric: cell-anchored resale + loose-only primary -> resale.
        text = "| 457(c) |\nNote: primary shelf historically used 457(o)."
        assert _classify_fee_table_text(text) == "resale"

    def test_cell_single_primary_suppresses_loose_both_mixed(self):
        # STRONGEST short-circuit proof: the cell pass finds exactly ONE
        # class (primary). The loose-pass, were it reached, WOULD see both
        # 457(o) AND 457(c) and return 'mixed'. Because the cell pass
        # returns first, the verdict is the single class 'primary' -- this
        # is what makes the two-pass design observable (loose never runs).
        text = "| 457(o) |\nfootnote prose mentions 457(o) and 457(c)"
        assert _classify_fee_table_text(text) == "primary"

    def test_cell_single_resale_suppresses_loose_both_mixed(self):
        # Symmetric: cell-resale present; loose would find both -> still
        # 'resale', not 'mixed'. Proves the short-circuit in both directions
        # against a loose-pass that contains the *opposite* class too.
        text = "| 457(c) |\nfootnote prose mentions 457(o) and 457(c)"
        assert _classify_fee_table_text(text) == "resale"

    def test_trailing_pipe_only_falls_through_to_loose(self):
        # The cell regex requires the pipe to PRECEDE the code. A code with
        # only a TRAILING pipe ('457(o) |') has no anchoring leading pipe,
        # so the cell pass misses and the loose pass classifies it primary.
        assert _classify_fee_table_text("457(o) |") == "primary"

    # ── loose fallback when no cell match ───────────────────────────
    def test_loose_only_resale_footnote_is_resale(self):
        # 457(g) in prose with NO pipe -> loose fallback -> resale.
        assert _classify_fee_table_text(
            "Pursuant to Rule 457(g) under the Securities Act."
        ) == "resale"

    def test_loose_only_primary_prose_is_primary(self):
        assert _classify_fee_table_text(
            "Calculated in accordance with Rule 457(o)."
        ) == "primary"

    def test_loose_both_primary_and_resale_is_mixed(self):
        # Prose mentions both 457(o) and 457(c), no pipe cells -> mixed.
        assert _classify_fee_table_text(
            "The fee was computed under 457(o) and 457(c) respectively."
        ) == "mixed"

    def test_pipe_absent_falls_through_to_loose_primary(self):
        # No '|' precedes the code, so cell pass misses; loose still
        # catches it -> primary (verifies cell-vs-loose path selection).
        assert _classify_fee_table_text("Footnote text 457(o) here.") == "primary"

    # ── neutral marker 457(p) must never classify ───────────────────
    def test_neutral_marker_457p_alone_is_unknown(self):
        # 457(p) is in NO char class -> stays unknown.
        assert _classify_fee_table_text("| 457(p) |") == "unknown"

    def test_neutral_marker_457p_does_not_force_mixed(self):
        # 457(p) alongside one real primary rule must NOT become mixed.
        assert _classify_fee_table_text("| 457(p) |\n| 457(o) |") == "primary"

    def test_neutral_marker_457p_alongside_resale(self):
        assert _classify_fee_table_text("| 457(p) |\n| 457(c) |") == "resale"

    # ── boundary char-class correctness ─────────────────────────────
    def test_stray_457x_is_unknown(self):
        assert _classify_fee_table_text("| 457(x) |") == "unknown"

    def test_stray_457d_is_unknown(self):
        # 'd' is in neither [oraibhfi] nor [cg].
        assert _classify_fee_table_text("| 457(d) |") == "unknown"

    @pytest.mark.parametrize("letter", list("oraibhfi"))
    def test_all_primary_char_class_letters(self, letter):
        assert _classify_fee_table_text(f"| 457({letter}) |") == "primary"

    @pytest.mark.parametrize("letter", list("cg"))
    def test_all_resale_char_class_letters(self, letter):
        assert _classify_fee_table_text(f"| 457({letter}) |") == "resale"

    @pytest.mark.parametrize("letter", list("dejklmnpqstuvwxyz"))
    def test_letters_outside_both_classes_are_unknown(self, letter):
        # Sweep every lowercase letter not in [oraibhfi] or [cg]; each
        # alone must classify as unknown. (Includes the neutral 'p'.)
        assert _classify_fee_table_text(f"| 457({letter}) |") == "unknown"

    # ── word-boundary / \\b anchor guards (loose regex) ──────────────
    def test_loose_code_embedded_in_word_does_not_match(self):
        # _RE_*_LOOSE is \\b-anchored. 'pre457(c)post' — the char before
        # '4' is the word-char 'e', so there is NO word boundary and the
        # loose regex must NOT fire. Verdict stays 'unknown'.
        assert _classify_fee_table_text("pre457(c)post") == "unknown"

    def test_loose_code_with_leading_digit_does_not_match(self):
        # '1457(o)' — a digit before '457' is also a word char, so no
        # boundary; the spurious longer number must not be misread.
        assert _classify_fee_table_text("x1457(o)x") == "unknown"

    def test_loose_standalone_code_does_match(self):
        # Contrast: a properly word-bounded standalone code DOES match.
        assert _classify_fee_table_text("see 457(c) here") == "resale"

    # ── char class is exactly one letter, then a close paren ─────────
    def test_double_letter_inside_parens_is_unknown(self):
        # The class matches a single letter that must be followed by
        # (optional ws and) ')'. '457(oo)' has a second 'o' before the
        # paren, so neither cell nor loose regex completes -> unknown.
        assert _classify_fee_table_text("| 457(oo) |") == "unknown"

    def test_space_between_457_and_paren_is_unknown(self):
        # The regex expects '457(' with no gap. '457 (o)' has a space, so
        # it does not match either pass -> unknown.
        assert _classify_fee_table_text("| 457 (o) |") == "unknown"

    def test_rule_glued_to_code_is_unknown(self):
        # 'Rule' is an optional group requiring trailing \\s+. 'Rule457(o)'
        # has no space, so the 'Rule' branch fails AND the bare-'457'
        # branch sees a word char ('e') before it -> no boundary -> unknown.
        assert _classify_fee_table_text("| Rule457(o) |") == "unknown"

    # ── cell-pass tab / uppercase RULE whitespace tolerance ──────────
    def test_cell_uppercase_rule_word_is_primary(self):
        # 'Rule' match is IGNORECASE, so all-caps 'RULE' still anchors.
        assert _classify_fee_table_text("| RULE 457(o) |") == "primary"

    def test_cell_tab_after_pipe_is_primary(self):
        # [\\s\\n]* between pipe and code includes a tab.
        assert _classify_fee_table_text("|\t457(o) |") == "primary"

    # ── mixed on a single pipe-delimited row (both cells, one line) ──
    def test_cell_both_codes_same_line_is_mixed(self):
        assert _classify_fee_table_text("| 457(o) | 457(c) |") == "mixed"

    def test_cell_mixed_resale_before_primary_order_independent(self):
        # Order of the two cells must not change the 'mixed' verdict.
        assert _classify_fee_table_text("| 457(c) |\n| 457(o) |") == "mixed"

    def test_cell_resale_g_and_primary_o_is_mixed(self):
        # Using the OTHER resale letter (g) + a primary letter -> mixed.
        assert _classify_fee_table_text("| 457(g) |\n| 457(o) |") == "mixed"

    # ── loose-pass mixed using the second letters of each class ──────
    def test_loose_mixed_resale_g_and_primary_r(self):
        assert _classify_fee_table_text("under 457(g) and 457(r)") == "mixed"

    # ── truthy-but-codeless content (the `if not text` guard) ────────
    @pytest.mark.parametrize("blank", ["   ", "\n\n", "\t"])
    def test_whitespace_only_text_is_unknown(self, blank):
        # Non-empty (truthy) whitespace passes the `if not text` guard but
        # contains no rule code, so both passes miss -> unknown.
        assert _classify_fee_table_text(blank) == "unknown"


class TestClassifyFeeTable:
    """DB-backed thin selector + delegate to the pure classifier."""

    def test_no_raw_row_is_unknown(self, temp_db):
        # Accession with no dilution_raw row at all -> 'unknown' no-op.
        assert classify_fee_table("0000-none") == "unknown"

    def test_accession_with_filing_but_no_raw_docs_is_unknown(self, temp_db):
        temp_db.add_filing("0000-bare", 1, form="S-3")
        assert classify_fee_table("0000-bare") == "unknown"

    def test_wrong_doc_type_not_selected(self, temp_db):
        # A row exists but doc_type='EX-10.1' -> filter excludes it.
        _stage_raw(temp_db, "0000-ex10", "| Rule 457(o) |",
                   doc_type="EX-10.1")
        assert classify_fee_table("0000-ex10") == "unknown"

    def test_doc_type_10k_not_selected(self, temp_db):
        _stage_raw(temp_db, "0000-10k", "| 457(c) |", doc_type="10-K")
        assert classify_fee_table("0000-10k") == "unknown"

    def test_doc_type_match_is_case_sensitive(self, temp_db):
        # SQL '=' is exact; lowercase doc_type must NOT match.
        _stage_raw(temp_db, "0000-lc", "| 457(o) |",
                   doc_type="ex-filing fees")
        assert classify_fee_table("0000-lc") == "unknown"

    def test_fee_table_primary(self, temp_db):
        _stage_raw(temp_db, "0000-pri", "| Rule 457(o) |")
        assert classify_fee_table("0000-pri") == "primary"

    def test_fee_table_resale(self, temp_db):
        _stage_raw(temp_db, "0000-res", "| 457(c) |")
        assert classify_fee_table("0000-res") == "resale"

    def test_fee_table_mixed(self, temp_db):
        _stage_raw(temp_db, "0000-mix", "| 457(o) |\n| 457(c) |")
        assert classify_fee_table("0000-mix") == "mixed"

    def test_empty_content_md_is_unknown(self, temp_db):
        # Schema marks content_md NOT NULL, so stage '' to exercise the
        # `content_md or ''` empty branch.
        _stage_raw(temp_db, "0000-empty", "")
        assert classify_fee_table("0000-empty") == "unknown"

    def test_only_ex_filing_fees_row_read_among_multiple(self, temp_db):
        # Two raw rows for same accession: one EX-FILING FEES (resale)
        # and one EX-10.1 (primary code). Only the EX-FILING FEES row is
        # selected, so the verdict must be 'resale'.
        temp_db.add_filing("0000-two", 1, form="S-3")
        temp_db.execute(
            """INSERT INTO dilution_raw
                 (accession_number, doc_name, doc_type, content_md, downloaded_at)
               VALUES (?,?,?,?,?)""",
            ("0000-two", "fees.htm", "EX-FILING FEES", "| 457(c) |", "2026-01-01"),
        )
        temp_db.execute(
            """INSERT INTO dilution_raw
                 (accession_number, doc_name, doc_type, content_md, downloaded_at)
               VALUES (?,?,?,?,?)""",
            ("0000-two", "ex10.htm", "EX-10.1", "| 457(o) |", "2026-01-01"),
        )
        assert classify_fee_table("0000-two") == "resale"

    def test_loose_only_prose_classifies_through_db(self, temp_db):
        # The DB path must delegate to the SAME two-pass classifier, so a
        # loose-only (no-pipe) prose code in the stored exhibit resolves.
        _stage_raw(temp_db, "0000-loose",
                   "Computed pursuant to Rule 457(g) under the Act.")
        assert classify_fee_table("0000-loose") == "resale"

    def test_whitespace_only_content_md_is_unknown(self, temp_db):
        # Truthy but codeless content stored verbatim -> unknown (guards
        # against a false 'primary'/'resale' on blank exhibits).
        _stage_raw(temp_db, "0000-ws", "   \n  ")
        assert classify_fee_table("0000-ws") == "unknown"

    def test_two_ex_filing_fees_rows_limit_one(self, temp_db):
        # PK is (accession_number, doc_name); stage TWO EX-FILING FEES
        # rows (distinct doc_names) for one accession. The query uses
        # LIMIT 1, so the verdict comes from whichever single row the DB
        # returns. SQLite's PK ordering returns the lexicographically
        # first doc_name ('fees_a.htm' -> resale) deterministically.
        temp_db.add_filing("0000-dup", 1, form="S-3")
        temp_db.execute(
            """INSERT INTO dilution_raw
                 (accession_number, doc_name, doc_type, content_md, downloaded_at)
               VALUES (?,?,?,?,?)""",
            ("0000-dup", "fees_a.htm", "EX-FILING FEES", "| 457(c) |", "2026-01-01"),
        )
        temp_db.execute(
            """INSERT INTO dilution_raw
                 (accession_number, doc_name, doc_type, content_md, downloaded_at)
               VALUES (?,?,?,?,?)""",
            ("0000-dup", "fees_b.htm", "EX-FILING FEES", "| 457(o) |", "2026-01-01"),
        )
        # Only one row is read; the verdict is a single non-'mixed' class.
        verdict = classify_fee_table("0000-dup")
        assert verdict in ("resale", "primary")
        assert verdict != "mixed"

    def test_db_mixed_round_trip(self, temp_db):
        # Both codes present in the single stored exhibit -> 'mixed' via DB.
        _stage_raw(temp_db, "0000-dbmix", "| 457(o) |\n| 457(c) |")
        assert classify_fee_table("0000-dbmix") == "mixed"

    def test_doc_type_filter_wins_over_alphabetical_doc_name(self, temp_db):
        # Two rows: a non-fee-table EX-10.1 whose doc_name ('aaa.htm') sorts
        # FIRST, and the real EX-FILING FEES row ('zzz.htm') with a primary
        # code. The WHERE doc_type=? filter must select the fee-table row
        # regardless of doc_name ordering -> 'primary', NOT the EX-10.1's
        # resale code. (Proves the filter, not LIMIT-1 row order, drives it.)
        temp_db.add_filing("0000-filt", 1, form="S-3")
        temp_db.execute(
            """INSERT INTO dilution_raw
                 (accession_number, doc_name, doc_type, content_md, downloaded_at)
               VALUES (?,?,?,?,?)""",
            ("0000-filt", "aaa.htm", "EX-10.1", "| 457(c) |", "2026-01-01"),
        )
        temp_db.execute(
            """INSERT INTO dilution_raw
                 (accession_number, doc_name, doc_type, content_md, downloaded_at)
               VALUES (?,?,?,?,?)""",
            ("0000-filt", "zzz.htm", "EX-FILING FEES", "| 457(o) |", "2026-01-01"),
        )
        assert classify_fee_table("0000-filt") == "primary"

    def test_db_loose_only_primary_prose_classifies(self, temp_db):
        # Symmetric to the resale loose-prose DB test: a no-pipe primary
        # 457(r) (the WKSI auto-shelf code) in stored prose -> 'primary'.
        _stage_raw(temp_db, "0000-loosep",
                   "Fee deferred per Rule 457(r) of the Securities Act.")
        assert classify_fee_table("0000-loosep") == "primary"

    def test_db_neutral_457p_only_is_unknown(self, temp_db):
        # A fee-table exhibit that mentions ONLY the neutral 457(p) offset
        # reference -> 'unknown' through the DB path (no spurious verdict).
        _stage_raw(temp_db, "0000-pcode", "| 457(p) |")
        assert classify_fee_table("0000-pcode") == "unknown"


class TestFormatFeeTableForPrompt:
    """Pure verdict -> markdown hint renderer."""

    def test_primary_block_mentions_primary_and_create_shelf(self):
        out = format_fee_table_for_prompt("primary")
        assert out != ""
        assert "## Fee-table classification" in out
        assert "PRIMARY" in out
        assert "create_shelf" in out
        assert "create_s1_offering" in out
        # The primary block is an affirmative "proceed" instruction; it must
        # NOT carry the resale block's note_no_event directive (guards against
        # a copy-paste cross-contamination of the verdict-specific prose).
        assert "note_no_event" not in out
        assert "RESALE" not in out

    def test_resale_block_mentions_resale_and_note_no_event(self):
        out = format_fee_table_for_prompt("resale")
        assert out != ""
        assert "RESALE" in out
        assert "note_no_event" in out
        # NB: the resale block legitimately contains the substring
        # "create_shelf" inside "Do NOT call create_shelf or
        # create_s1_offering", so we deliberately do NOT assert its absence.
        # It must, however, not announce itself as PRIMARY/COMBINED.
        assert "**PRIMARY**" not in out
        assert "COMBINED" not in out

    def test_mixed_block_mentions_combined(self):
        out = format_fee_table_for_prompt("mixed")
        assert out != ""
        assert "COMBINED" in out
        # The mixed block instructs emitting create_shelf for the primary
        # section and deriving the resale section from the file_number.
        assert "create_shelf" in out
        assert "file_number" in out

    def test_unknown_is_empty_string(self):
        assert format_fee_table_for_prompt("unknown") == ""

    @pytest.mark.parametrize("garbage", ["", "garbage", "PRIMARY", "Primary"])
    def test_out_of_domain_falls_through_to_empty(self, garbage):
        # Anything not exactly primary/resale/mixed -> defensive '' default.
        assert format_fee_table_for_prompt(garbage) == ""  # type: ignore[arg-type]

    def test_three_real_verdicts_are_distinct(self):
        blocks = {
            format_fee_table_for_prompt("primary"),
            format_fee_table_for_prompt("resale"),
            format_fee_table_for_prompt("mixed"),
        }
        assert len(blocks) == 3


class TestModuleConstants:
    """Sanity on the exported gating constant."""

    def test_fee_table_forms_is_frozenset_with_known_members(self):
        assert isinstance(FEE_TABLE_FORMS, frozenset)
        # Spot-check a representative primary-shelf form and a takedown.
        assert "S-3" in FEE_TABLE_FORMS
        assert "424B5" in FEE_TABLE_FORMS
        # A periodic form is intentionally absent (no fee table).
        assert "10-K" not in FEE_TABLE_FORMS
        assert "8-K" not in FEE_TABLE_FORMS
