"""Unit tests for dilution/ledger/_llm_utils.py.

Covers the shared LLM-call helpers: filing-text whitespace normalization,
the per-issuer unit preamble, chat-kwargs assembly (make_chat), the
finish_reason truncation warnings (check_response), and the thin async
wrapper (asample_and_check).

No DB, no network, no real LLM. The only seam touched is the module-level
`config` symbol (monkeypatched per test for make_chat) — config.py only
reads os.environ, so importing the target is side-effect-free.
"""

from __future__ import annotations

import asyncio
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
# make_chat
# ════════════════════════════════════════════════════════════════════
def _fake_client():
    """A client whose chat.create returns the exact kwargs dict it got."""
    return types.SimpleNamespace(
        chat=types.SimpleNamespace(create=lambda **kw: kw)
    )


@pytest.fixture
def patch_config(monkeypatch):
    """Helper to set config.LLM_PROVIDER / LLM_MODEL on the module's
    `config` symbol. make_chat reads these at CALL time."""
    def _set(provider="gemini", model="cfg-model"):
        monkeypatch.setattr(m.config, "LLM_PROVIDER", provider, raising=False)
        monkeypatch.setattr(m.config, "LLM_MODEL", model, raising=False)
    return _set


class TestMakeChat:
    def test_base_keys_default_flow(self, patch_config):
        patch_config(provider="gemini", model="cfg-model")
        kw = m.make_chat(_fake_client())
        # model from config, plus the two always-present keys.
        assert kw["model"] == "cfg-model"
        assert kw["max_tokens"] == m.DEFAULT_MAX_TOKENS == 32_000
        assert kw["temperature"] == m.EXTRACT_TEMPERATURE == 0.0
        # No optional keys, and (non-xai) no seed.
        assert set(kw) == {"model", "max_tokens", "temperature"}

    def test_temperature_always_present_for_gemini(self, patch_config):
        # Regression guard for the gemini temp-1.0 determinism bug: the
        # temperature key must NOT be gated behind the xai provider.
        patch_config(provider="gemini")
        kw = m.make_chat(_fake_client())
        assert "temperature" in kw
        assert kw["temperature"] == 0.0

    def test_temperature_present_for_all_providers(self, patch_config):
        for provider in ("xai", "gemini", "moonshot", "whatever"):
            patch_config(provider=provider)
            kw = m.make_chat(_fake_client())
            assert "temperature" in kw, provider

    def test_seed_present_only_for_xai(self, patch_config):
        patch_config(provider="xai")
        kw = m.make_chat(_fake_client())
        assert "seed" in kw
        assert kw["seed"] == m.EXTRACT_SEED == 42

    @pytest.mark.parametrize("provider", ["gemini", "moonshot", "other", ""])
    def test_seed_absent_for_non_xai(self, patch_config, provider):
        patch_config(provider=provider)
        kw = m.make_chat(_fake_client())
        assert "seed" not in kw

    def test_model_none_uses_config(self, patch_config):
        patch_config(model="sentinel-config-model")
        kw = m.make_chat(_fake_client(), model=None)
        assert kw["model"] == "sentinel-config-model"

    def test_explicit_model_overrides_config(self, patch_config):
        patch_config(model="cfg-model")
        kw = m.make_chat(_fake_client(), model="explicit-foo")
        assert kw["model"] == "explicit-foo"

    def test_empty_string_model_forwarded_not_replaced(self, patch_config):
        # Code uses `model is not None`, so an empty-string model is a
        # legitimate explicit value and must NOT fall back to config.
        patch_config(model="cfg-model")
        kw = m.make_chat(_fake_client(), model="")
        assert kw["model"] == ""

    def test_optional_keys_absent_when_none(self, patch_config):
        patch_config(provider="gemini")
        kw = m.make_chat(_fake_client(),
                         response_format=None, tools=None, tool_choice=None)
        assert "response_format" not in kw
        assert "tools" not in kw
        assert "tool_choice" not in kw

    def test_response_format_identity_pass_through(self, patch_config):
        patch_config()
        sentinel = type("MyModel", (), {})  # a class object
        kw = m.make_chat(_fake_client(), response_format=sentinel)
        assert kw["response_format"] is sentinel

    def test_tools_and_tool_choice_both_present(self, patch_config):
        patch_config()
        tools = [{"type": "function", "function": {"name": "x"}}]
        kw = m.make_chat(_fake_client(), tools=tools, tool_choice="required")
        assert kw["tools"] is tools
        assert kw["tool_choice"] == "required"

    def test_tools_alone_without_tool_choice(self, patch_config):
        # tools provided but tool_choice left None: tools key present,
        # tool_choice key ABSENT (each optional key is gated independently).
        patch_config(provider="gemini")
        tools = [{"type": "function", "function": {"name": "y"}}]
        kw = m.make_chat(_fake_client(), tools=tools)
        assert kw["tools"] is tools
        assert "tool_choice" not in kw
        assert "response_format" not in kw

    def test_response_format_alone_without_tools(self, patch_config):
        # Mirror: response_format alone leaves tools/tool_choice absent.
        patch_config(provider="gemini")
        sentinel = type("OnlyModel", (), {})
        kw = m.make_chat(_fake_client(), response_format=sentinel)
        assert kw["response_format"] is sentinel
        assert "tools" not in kw
        assert "tool_choice" not in kw

    def test_seed_added_for_xai_even_when_zero(self, patch_config):
        # The seed gate is provider=='xai', NOT truthiness of seed: a
        # falsy seed=0 must still be forwarded for xai. Guards against a
        # naive `if seed:` regression.
        patch_config(provider="xai")
        kw = m.make_chat(_fake_client(), seed=0)
        assert "seed" in kw
        assert kw["seed"] == 0

    def test_response_format_and_tools_combined(self, patch_config):
        patch_config()
        sentinel = type("MyModel", (), {})
        tools = [{"type": "function"}]
        kw = m.make_chat(_fake_client(), response_format=sentinel,
                         tools=tools, tool_choice="auto")
        assert kw["response_format"] is sentinel
        assert kw["tools"] is tools
        assert kw["tool_choice"] == "auto"

    def test_defaults_flow_through(self, patch_config):
        patch_config(provider="xai")
        kw = m.make_chat(_fake_client())
        assert kw["max_tokens"] == 32_000
        assert kw["temperature"] == 0.0
        assert kw["seed"] == 42

    def test_explicit_max_tokens_and_temperature_override(self, patch_config):
        patch_config(provider="xai")
        kw = m.make_chat(_fake_client(), max_tokens=512,
                         temperature=0.7, seed=99)
        assert kw["max_tokens"] == 512
        assert kw["temperature"] == 0.7
        assert kw["seed"] == 99

    def test_seed_override_ignored_for_non_xai(self, patch_config):
        # Even an explicit seed arg is dropped when provider isn't xai.
        patch_config(provider="gemini")
        kw = m.make_chat(_fake_client(), seed=99)
        assert "seed" not in kw

    def test_return_value_is_pass_through(self, patch_config):
        patch_config()
        sentinel = object()
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(create=lambda **kw: sentinel)
        )
        assert m.make_chat(client) is sentinel

    def test_create_called_with_keyword_args(self, patch_config):
        # Verify create receives **kwargs (keyword), not positional.
        patch_config(provider="xai", model="cfg-model")
        captured = {}

        def create(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return kwargs

        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(create=create)
        )
        m.make_chat(client, tools=[1], tool_choice="required")
        assert captured["args"] == ()  # nothing positional
        assert captured["kwargs"]["model"] == "cfg-model"
        assert captured["kwargs"]["tools"] == [1]
        assert captured["kwargs"]["tool_choice"] == "required"


# ════════════════════════════════════════════════════════════════════
# Module constants
# ════════════════════════════════════════════════════════════════════
class TestConstants:
    def test_constant_values(self):
        assert m.EXTRACT_TEMPERATURE == 0.0
        assert m.EXTRACT_SEED == 42
        assert m.DEFAULT_MAX_TOKENS == 32_000


# ════════════════════════════════════════════════════════════════════
# check_response
# ════════════════════════════════════════════════════════════════════
class TestCheckResponse:
    def test_no_finish_reason_attr_no_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = object()  # no finish_reason attribute at all
        out = m.check_response(resp)
        assert out is resp
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_finish_reason_none_no_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = types.SimpleNamespace(finish_reason=None)
        out = m.check_response(resp)
        assert out is resp
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    @pytest.mark.parametrize("reason", ["stop", "STOP", "reason_max_len",
                                        "length", "tool_calls", ""])
    def test_unrecognized_reason_no_warning(self, caplog, reason):
        # Match is exact and case-sensitive; lowercase variant must NOT fire.
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = types.SimpleNamespace(finish_reason=reason)
        out = m.check_response(resp)
        assert out is resp
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_reason_max_len_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = types.SimpleNamespace(finish_reason="REASON_MAX_LEN")
        out = m.check_response(resp)
        assert out is resp
        warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warns) == 1
        assert "output truncated at max_tokens" in warns[0].getMessage()

    def test_reason_max_context_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = types.SimpleNamespace(finish_reason="REASON_MAX_CONTEXT")
        out = m.check_response(resp)
        assert out is resp
        warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warns) == 1
        assert "input exceeded model context window" in warns[0].getMessage()

    def test_reason_time_limit_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = types.SimpleNamespace(finish_reason="REASON_TIME_LIMIT")
        out = m.check_response(resp)
        assert out is resp
        warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warns) == 1
        assert "generation hit time limit" in warns[0].getMessage()

    def test_returns_same_object_identity_for_warning_branch(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = types.SimpleNamespace(finish_reason="REASON_MAX_LEN", payload=1)
        assert m.check_response(resp) is resp

    # ── tag formatting ─────────────────────────────────────────────
    def test_tag_with_handler_and_accession(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = types.SimpleNamespace(finish_reason="REASON_MAX_LEN")
        m.check_response(resp, accession="0001104659-22-025374",
                         handler="walker")
        msg = caplog.records[-1].getMessage()
        assert "[walker] 0001104659-22-025374" in msg

    def test_tag_accession_only(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = types.SimpleNamespace(finish_reason="REASON_MAX_CONTEXT")
        m.check_response(resp, accession="acc-123", handler=None)
        msg = caplog.records[-1].getMessage()
        assert "acc-123" in msg
        assert "[" not in msg.split("—")[0]  # no handler bracket

    def test_tag_question_mark_when_both_none(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = types.SimpleNamespace(finish_reason="REASON_TIME_LIMIT")
        m.check_response(resp, accession=None, handler=None)
        msg = caplog.records[-1].getMessage()
        assert msg.startswith("? ")

    def test_handler_set_accession_none(self, caplog):
        # handler is truthy so tag is "[handler] None".
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = types.SimpleNamespace(finish_reason="REASON_MAX_LEN")
        m.check_response(resp, accession=None, handler="seed")
        msg = caplog.records[-1].getMessage()
        assert "[seed] None" in msg

    def test_default_args_no_warning_when_clean(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = types.SimpleNamespace(finish_reason="stop")
        out = m.check_response(resp)
        assert out is resp
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ════════════════════════════════════════════════════════════════════
# asample_and_check
# ════════════════════════════════════════════════════════════════════
class _FakeChat:
    def __init__(self, response, *, record):
        self._response = response
        self._record = record

    async def sample(self):
        self._record["calls"] += 1
        return self._response


class TestAsampleAndCheck:
    def test_awaits_sample_once_and_passes_through(self):
        resp = types.SimpleNamespace(finish_reason="stop")
        record = {"calls": 0}
        chat = _FakeChat(resp, record=record)
        out = asyncio.run(m.asample_and_check(chat))
        assert out is resp
        assert record["calls"] == 1

    def test_return_value_equals_response(self):
        resp = object()
        record = {"calls": 0}
        # object() has no finish_reason -> no warning, plain pass-through.
        chat = types.SimpleNamespace()

        async def sample():
            record["calls"] += 1
            return resp

        chat.sample = sample
        out = asyncio.run(m.asample_and_check(chat))
        assert out is resp
        assert record["calls"] == 1

    def test_forwards_accession_and_handler_to_check_response(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = types.SimpleNamespace(finish_reason="REASON_MAX_LEN")
        record = {"calls": 0}
        chat = _FakeChat(resp, record=record)
        out = asyncio.run(
            m.asample_and_check(chat, accession="acc-xyz", handler="overhang")
        )
        assert out is resp
        warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warns) == 1
        msg = warns[0].getMessage()
        assert "[overhang] acc-xyz" in msg
        assert "output truncated at max_tokens" in msg

    def test_no_warning_for_clean_finish(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        resp = types.SimpleNamespace(finish_reason="stop")
        chat = _FakeChat(resp, record={"calls": 0})
        out = asyncio.run(m.asample_and_check(chat, accession="a", handler="h"))
        assert out is resp
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_sample_exception_propagates_and_skips_check(self, caplog):
        # If chat.sample() raises, the exception bubbles out and
        # check_response is never reached (no warning emitted).
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

        class _Boom:
            async def sample(self):
                raise RuntimeError("sample blew up")

        with pytest.raises(RuntimeError, match="sample blew up"):
            asyncio.run(m.asample_and_check(_Boom(), accession="a"))
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
