"""Unit tests for dilution/ledger/_llm_utils.py.

Covers the shared LLM-call helpers: filing-text whitespace normalization,
the per-issuer unit preamble, and the incomplete-generation warnings
(check_response).

Request assembly lives in dilution/openai_client.py now (see
tests/test_openai_client.py) — this module no longer builds chats.

No DB, no network, no real LLM.
"""

from __future__ import annotations

import logging
import types

import pytest

import dilution.ledger._llm_utils as m


ZWSP = "​"
LOGGER_NAME = "dilution.ledger._llm_utils"


# ════════════════════════════════════════════════════════════════════
# normalize_filing_text
# ════════════════════════════════════════════════════════════════════
class TestNormalizeFilingText:
    def test_empty_string(self):
        assert m.normalize_filing_text("") == ""

    def test_plain_text_identity(self):
        # No ZWSP and no >=3 space runs -> returned unchanged.
        s = "The quick brown fox jumped over the lazy dog."
        assert m.normalize_filing_text(s) == s

    def test_single_zwsp_removed(self):
        assert m.normalize_filing_text(ZWSP) == ""

    def test_zwsp_removed_everywhere_including_mid_word(self):
        assert m.normalize_filing_text(f"a{ZWSP}b") == "ab"

    def test_consecutive_zwsp_all_removed(self):
        assert m.normalize_filing_text(f"a{ZWSP}{ZWSP}{ZWSP}b") == "ab"

    def test_zwsp_at_edges_removed(self):
        assert m.normalize_filing_text(f"{ZWSP}hello{ZWSP}") == "hello"

    def test_interior_single_spaces_preserved(self):
        assert m.normalize_filing_text("a b c d") == "a b c d"

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("a b", "a b"),        # 1 space — untouched
            ("a  b", "a  b"),      # 2 spaces — BELOW threshold, untouched
            ("a   b", "a  b"),     # 3 spaces — AT threshold -> 2
            ("a    b", "a  b"),    # 4 spaces -> 2
            ("a     b", "a  b"),   # 5 spaces -> 2
            ("a          b", "a  b"),  # 10 spaces -> 2
        ],
    )
    def test_space_run_threshold_sweep(self, raw, expected):
        assert m.normalize_filing_text(raw) == expected

    def test_only_spaces_three_collapse_to_two(self):
        assert m.normalize_filing_text("   ") == "  "

    def test_only_spaces_many_collapse_to_two(self):
        assert m.normalize_filing_text(" " * 50) == "  "

    def test_only_two_spaces_untouched(self):
        assert m.normalize_filing_text("  ") == "  "

    def test_tabs_not_touched(self):
        # Regex matches literal space only; tabs are left intact.
        assert m.normalize_filing_text("a\t\t\t\tb") == "a\t\t\t\tb"

    def test_newlines_not_touched(self):
        assert m.normalize_filing_text("a\n\n\n\nb") == "a\n\n\n\nb"

    def test_mixed_tabs_and_spaces(self):
        # The space run collapses but the tab interrupts so each side is
        # evaluated independently; here only the 4-space run collapses.
        assert m.normalize_filing_text("a    \tb") == "a  \tb"

    def test_zwsp_strip_before_space_collapse_interaction(self):
        # The load-bearing ordering case: ZWSPs sit between single spaces
        # ("a␣ZWSP␣ZWSP␣b"). Stripping ZWSP FIRST merges the single
        # spaces into a 3-run, which then collapses to 2. If the order
        # were reversed the spaces would never reach the >=3 threshold.
        raw = f"a{ZWSP} {ZWSP} {ZWSP} b"
        # After strip: "a   b" (a + 3 spaces + b), wait — count spaces.
        # raw spaces: " " after a's ZWSP? Construct explicitly below.
        assert m.normalize_filing_text(raw) == "a  b"

    def test_zwsp_merge_exact_three_spaces(self):
        # "a" + space + ZWSP + space + ZWSP + space + "b"
        # Strip ZWSP -> "a   b" (3 spaces) -> collapse -> "a  b".
        raw = "a " + ZWSP + " " + ZWSP + " b"
        assert m.normalize_filing_text(raw) == "a  b"

    def test_zwsp_merge_two_spaces_stays_below_threshold(self):
        # "a" + space + ZWSP + space + "b" -> strip -> "a  b" (2 spaces)
        # -> below threshold -> unchanged.
        raw = "a " + ZWSP + " b"
        assert m.normalize_filing_text(raw) == "a  b"

    def test_combined_zwsp_and_long_runs(self):
        raw = f"col1{ZWSP}     col2{ZWSP}{ZWSP}      col3"
        assert m.normalize_filing_text(raw) == "col1  col2  col3"

    def test_returns_str(self):
        assert isinstance(m.normalize_filing_text("x"), str)


# ════════════════════════════════════════════════════════════════════
# unit_preamble
# ════════════════════════════════════════════════════════════════════
class TestUnitPreamble:
    US_MARKER = "This issuer is a US listing."
    FPI_MARKER = "REPORTING UNITS — ADS, NOT ORDINARY SHARES."

    # ── US-listing (non-FPI) branch ────────────────────────────────
    @pytest.mark.parametrize(
        "ctx",
        [
            None,
            {},
            {"is_fpi": False},
            {"is_fpi": None},
            {"is_fpi": 0},
            {"is_fpi": ""},
            {"foo": "bar"},  # truthy dict but no is_fpi key
        ],
    )
    def test_us_branch_for_falsy_or_missing_fpi(self, ctx):
        out = m.unit_preamble(ctx)
        assert self.US_MARKER in out
        # US branch must NOT mention ADS at all.
        assert "ADS" not in out
        assert self.FPI_MARKER not in out

    def test_none_does_not_raise(self):
        # The `not unit_ctx` guard short-circuits before any .get on None.
        assert m.unit_preamble(None).startswith("This issuer is a US listing.")

    def test_us_branch_mentions_common_shares_and_usd(self):
        out = m.unit_preamble(None)
        assert "common shares" in out
        assert "USD per common share" in out

    def test_us_branch_ends_with_newline(self):
        assert m.unit_preamble(None).endswith("\n")

    # ── FPI branch — generic ratio clause (falsy ads_ratio) ────────
    @pytest.mark.parametrize(
        "ctx",
        [
            {"is_fpi": True},                    # no ads_ratio key
            {"is_fpi": True, "ads_ratio": None}, # explicit None
            {"is_fpi": True, "ads_ratio": 0},    # 0 is falsy
            {"is_fpi": True, "ads_ratio": 0.0},  # 0.0 falsy
        ],
    )
    def test_fpi_branch_generic_ratio_clause(self, ctx):
        out = m.unit_preamble(ctx)
        assert self.FPI_MARKER in out
        # Generic clause references the cover page / Description section.
        assert "cover page" in out
        assert "Description of Securities section" in out
        # Must NOT render a numeric "One ADS represents N ordinary".
        assert "One ADS represents" not in out

    def test_fpi_branch_with_ratio_renders_number(self):
        out = m.unit_preamble({"is_fpi": True, "ads_ratio": 100})
        assert "One ADS represents 100 ordinary shares" in out
        assert self.FPI_MARKER in out

    @pytest.mark.parametrize(
        "ratio, rendered",
        [
            (100, "100"),
            (100.0, "100"),     # :g trims trailing .0
            (0.25, "0.25"),
            (12.5, "12.5"),
            (1, "1"),
            (400, "400"),
        ],
    )
    def test_fpi_ratio_g_formatting(self, ratio, rendered):
        out = m.unit_preamble({"is_fpi": True, "ads_ratio": ratio})
        assert f"One ADS represents {rendered} ordinary shares" in out

    def test_fpi_branch_contains_apply_split_guidance(self):
        out = m.unit_preamble({"is_fpi": True, "ads_ratio": 100})
        assert "apply_split" in out
        assert 'units="ads"' in out
        assert "ADS SPLITS." in out

    def test_fpi_branch_contains_reporting_units_marker(self):
        assert self.FPI_MARKER in m.unit_preamble({"is_fpi": True})

    def test_fpi_branch_ends_with_newline(self):
        assert m.unit_preamble({"is_fpi": True}).endswith("\n")
        assert m.unit_preamble({"is_fpi": True, "ads_ratio": 7}).endswith("\n")

    def test_fpi_branch_example_share_count(self):
        # The verbatim worked example should be present.
        out = m.unit_preamble({"is_fpi": True})
        assert "129,818" in out
        assert "12,981,800" in out

    def test_returns_str(self):
        assert isinstance(m.unit_preamble(None), str)
        assert isinstance(m.unit_preamble({"is_fpi": True}), str)

    # ── added: truthy non-bool is_fpi mirrors the falsy-int US sweep ──
    @pytest.mark.parametrize("flag", [1, "yes", [0], {"k": "v"}])
    def test_fpi_branch_for_truthy_non_bool_is_fpi(self, flag):
        # The branch lever is `not unit_ctx.get("is_fpi")`; any truthy
        # value (not just bool True) selects the FPI/ADS block.
        out = m.unit_preamble({"is_fpi": flag})
        assert self.FPI_MARKER in out
        assert self.US_MARKER not in out

    # ── added: ratio truthiness boundary — negative is truthy, renders ─
    def test_fpi_negative_ratio_is_truthy_and_renders(self):
        # A negative ratio is truthy, so it takes the numeric clause (the
        # generic cover-page clause is reserved for falsy 0/None/missing).
        out = m.unit_preamble({"is_fpi": True, "ads_ratio": -5})
        assert "One ADS represents -5 ordinary shares" in out
        assert "cover page" not in out

    # ── added: failure-path — numeric `:g` format rejects a str ratio ──
    def test_fpi_string_ratio_raises_value_error(self):
        # ads_ratio is documented as numeric; f"{ratio:g}" cannot format a
        # str. A truthy non-numeric ratio surfaces a ValueError rather than
        # silently mis-rendering. Pins the numeric contract.
        with pytest.raises(ValueError):
            m.unit_preamble({"is_fpi": True, "ads_ratio": "4"})


# ════════════════════════════════════════════════════════════════════
# Module constants
# ════════════════════════════════════════════════════════════════════
class TestConstants:
    def test_constant_values(self):
        assert m.DEFAULT_MAX_TOKENS == 32_000

    def test_no_sampling_constants_survive(self):
        # The gpt-5.6 family rejects temperature/top_p in reasoning mode
        # and /v1/responses has no seed parameter, so these constants were
        # deleted rather than left lying around to be "helpfully" re-wired
        # into a request — which would 400 every call in the pipeline.
        assert not hasattr(m, "EXTRACT_TEMPERATURE")
        assert not hasattr(m, "EXTRACT_SEED")

    def test_chat_builders_are_gone(self):
        # make_chat / asample_and_check belonged to the chat-completions
        # era; request assembly is openai_client.request_kwargs.
        assert not hasattr(m, "make_chat")
        assert not hasattr(m, "asample_and_check")


# ════════════════════════════════════════════════════════════════════
# check_response
# ════════════════════════════════════════════════════════════════════
def _resp(status=None, reason=None, **extra):
    """Minimal Responses-shaped stub: status + incomplete_details.reason."""
    ns = types.SimpleNamespace(status=status, **extra)
    if reason is not None:
        ns.incomplete_details = types.SimpleNamespace(reason=reason)
    return ns


class TestCheckResponse:
    def test_no_status_attr_no_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = object()  # no status attribute at all
        out = m.check_response(resp)
        assert out is resp
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_status_none_no_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = _resp(status=None)
        out = m.check_response(resp)
        assert out is resp
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    @pytest.mark.parametrize("status", ["completed", "COMPLETED", "in_progress",
                                        "queued", "failed", "INCOMPLETE", ""])
    def test_non_incomplete_status_no_warning(self, caplog, status):
        # Match is exact and case-sensitive — only the literal "incomplete"
        # means the generation stopped early with usable partial output.
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = _resp(status=status, reason="max_output_tokens")
        out = m.check_response(resp)
        assert out is resp
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_max_output_tokens_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = _resp(status="incomplete", reason="max_output_tokens")
        out = m.check_response(resp)
        assert out is resp
        warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warns) == 1
        assert "output truncated at max_output_tokens" in warns[0].getMessage()

    def test_other_incomplete_reason_warns_generically(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = _resp(status="incomplete", reason="content_filter")
        out = m.check_response(resp)
        assert out is resp
        warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warns) == 1
        msg = warns[0].getMessage()
        assert "generation incomplete" in msg
        assert "content_filter" in msg

    def test_incomplete_without_details_still_warns(self, caplog):
        # An incomplete response missing incomplete_details must not blow
        # up on the attribute walk — reason falls through as None.
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = _resp(status="incomplete")
        out = m.check_response(resp)
        assert out is resp
        warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warns) == 1
        assert "reason=None" in warns[0].getMessage()

    def test_returns_same_object_identity_for_warning_branch(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = _resp(status="incomplete", reason="max_output_tokens", payload=1)
        assert m.check_response(resp) is resp

    # ── tag formatting ─────────────────────────────────────────────
    def test_tag_with_handler_and_accession(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = _resp(status="incomplete", reason="max_output_tokens")
        m.check_response(resp, accession="0001104659-22-025374",
                         handler="walker")
        msg = caplog.records[-1].getMessage()
        assert "[walker] 0001104659-22-025374" in msg

    def test_tag_accession_only(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = _resp(status="incomplete", reason="max_output_tokens")
        m.check_response(resp, accession="acc-123", handler=None)
        msg = caplog.records[-1].getMessage()
        assert "acc-123" in msg
        assert "[" not in msg.split("—")[0]  # no handler bracket

    def test_tag_question_mark_when_both_none(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = _resp(status="incomplete", reason="max_output_tokens")
        m.check_response(resp, accession=None, handler=None)
        msg = caplog.records[-1].getMessage()
        assert msg.startswith("? ")

    def test_handler_set_accession_none(self, caplog):
        # handler is truthy so tag is "[handler] None".
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = _resp(status="incomplete", reason="max_output_tokens")
        m.check_response(resp, accession=None, handler="seed")
        msg = caplog.records[-1].getMessage()
        assert "[seed] None" in msg

    def test_default_args_no_warning_when_clean(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = _resp(status="completed")
        out = m.check_response(resp)
        assert out is resp
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
