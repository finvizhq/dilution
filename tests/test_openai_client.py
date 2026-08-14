"""Unit tests for dilution/openai_client.py.

This module is the whole vendor surface, so its contract is worth pinning
hard. Three things here are live production hazards rather than style
points, and each has a test that fails loudly if it regresses:

  1. `tool_calls()` must FILTER `resp.output` by item type. With reasoning
     on, the API interleaves `reasoning` items among the `function_call`
     items; positional access silently drops calls and the walk looks
     like a clean no-op filing.
  2. `request_kwargs()` must emit NO temperature / top_p / seed. The
     gpt-5.6 family rejects the first two in reasoning mode and
     /v1/responses has no seed parameter, so a well-meaning
     "determinism fix" 400s every call in the pipeline.
  3. The json_schema translation must set `strict: False` EXPLICITLY.
     /v1/responses defaults `text.format` strict to true (chat completions
     defaulted it false), and strict demands every property appear in
     `required` — something our Optional-bearing Pydantic models violate
     by construction, so omitting the key 400s every overhang call.

No network: every test either builds a payload or reads a stub response.
"""

from __future__ import annotations

import types

import pytest
from pydantic import BaseModel

import config
import dilution.openai_client as oc
from conftest import response_stub


def _msgs():
    return [oc.system("sys"), oc.user("usr")]


def _kwargs(**over):
    base = dict(messages=_msgs(), max_output_tokens=64)
    base.update(over)
    return oc.request_kwargs(**base)


# ════════════════════════════════════════════════════════════════════
# message helpers
# ════════════════════════════════════════════════════════════════════
class TestMessageHelpers:
    def test_system_shape(self):
        assert oc.system("hello") == {"role": "system", "content": "hello"}

    def test_user_shape(self):
        assert oc.user("hello") == {"role": "user", "content": "hello"}

    def test_helpers_return_dicts_not_tuples(self):
        # The old provider layer used ("role", text) tuples; /v1/responses
        # wants dicts and would reject a tuple outright.
        assert isinstance(oc.system("x"), dict)
        assert isinstance(oc.user("x"), dict)


# ════════════════════════════════════════════════════════════════════
# request_kwargs
# ════════════════════════════════════════════════════════════════════
class TestRequestKwargs:
    def test_base_payload(self):
        kw = _kwargs()
        assert kw["model"] == config.LLM_MODEL
        assert kw["input"] == _msgs()
        assert kw["max_output_tokens"] == 64
        assert kw["reasoning"] == {"effort": config.OPENAI_REASONING_EFFORT}
        assert kw["service_tier"] == config.OPENAI_SERVICE_TIER
        assert kw["store"] is False

    def test_no_sampling_parameters(self):
        # THE regression guard. Any of these three keys makes the API 400.
        kw = _kwargs()
        for banned in ("temperature", "top_p", "seed"):
            assert banned not in kw, banned

    def test_uses_max_output_tokens_not_chat_completions_names(self):
        kw = _kwargs()
        assert "max_output_tokens" in kw
        assert "max_tokens" not in kw
        assert "max_completion_tokens" not in kw

    def test_input_key_not_messages(self):
        kw = _kwargs()
        assert "input" in kw
        assert "messages" not in kw

    def test_model_defaults_to_config(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_MODEL", "sentinel-model")
        assert _kwargs()["model"] == "sentinel-model"

    def test_explicit_model_overrides_config(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_MODEL", "cfg")
        assert _kwargs(model="explicit")["model"] == "explicit"

    def test_empty_string_model_forwarded_not_replaced(self, monkeypatch):
        # The check is `model is not None`, so "" is an explicit value.
        monkeypatch.setattr(config, "LLM_MODEL", "cfg")
        assert _kwargs(model="")["model"] == ""

    def test_reasoning_effort_read_from_config_at_call_time(self, monkeypatch):
        monkeypatch.setattr(config, "OPENAI_REASONING_EFFORT", "medium")
        assert _kwargs()["reasoning"] == {"effort": "medium"}

    def test_service_tier_read_from_config_at_call_time(self, monkeypatch):
        monkeypatch.setattr(config, "OPENAI_SERVICE_TIER", "default")
        assert _kwargs()["service_tier"] == "default"

    def test_optional_keys_absent_when_none(self):
        kw = _kwargs(response_format=None, tools=None, tool_choice=None,
                     cache_key=None)
        for key in ("text", "tools", "tool_choice", "prompt_cache_key"):
            assert key not in kw, key

    def test_tools_and_tool_choice_forwarded(self):
        tools = [{"type": "function", "name": "x"}]
        kw = _kwargs(tools=tools, tool_choice="required")
        assert kw["tools"] is tools
        assert kw["tool_choice"] == "required"

    def test_tools_alone_leaves_tool_choice_absent(self):
        kw = _kwargs(tools=[{"type": "function", "name": "y"}])
        assert "tools" in kw
        assert "tool_choice" not in kw

    def test_cache_key_forwarded_as_prompt_cache_key(self):
        assert _kwargs(cache_key="walker-v8")["prompt_cache_key"] == "walker-v8"

    def test_empty_cache_key_is_dropped(self):
        # Gate is truthiness: an empty key would be a useless cache bucket.
        assert "prompt_cache_key" not in _kwargs(cache_key="")


# ════════════════════════════════════════════════════════════════════
# response_format → text.format translation
# ════════════════════════════════════════════════════════════════════
class Sample(BaseModel):
    required_field: str
    optional_field: int | None = None


class TestTextFormat:
    def test_pydantic_class_becomes_json_schema(self):
        fmt = _kwargs(response_format=Sample)["text"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["name"] == "Sample"
        assert fmt["schema"] == Sample.model_json_schema()

    def test_strict_is_explicitly_false(self):
        # NOT merely absent: /v1/responses defaults text.format strict to
        # TRUE, and strict demands every property appear in `required`, so
        # an omitted key 400s every structured-output call. Regression
        # guard for exactly that outage.
        fmt = _kwargs(response_format=Sample)["text"]["format"]
        assert fmt["strict"] is False

    def test_optional_field_is_not_required_in_emitted_schema(self):
        # Pins WHY strict is unusable, so nobody "fixes" it later.
        schema = _kwargs(response_format=Sample)["text"]["format"]["schema"]
        assert "required_field" in schema["required"]
        assert "optional_field" not in schema.get("required", [])

    def test_dict_response_format_passed_through_verbatim(self):
        raw = {"format": {"type": "json_object"}}
        assert _kwargs(response_format=raw)["text"] is raw

    def test_object_without_schema_method_falls_back_to_json_object(self):
        fmt = _kwargs(response_format=type("Bare", (), {}))["text"]["format"]
        assert fmt == {"type": "json_object"}


# ════════════════════════════════════════════════════════════════════
# tool_calls
# ════════════════════════════════════════════════════════════════════
class TestToolCalls:
    def test_extracts_function_calls(self):
        resp = response_stub(calls=[("create_warrant", '{"count": 5}')])
        calls = oc.tool_calls(resp)
        assert len(calls) == 1
        assert calls[0].name == "create_warrant"
        assert calls[0].arguments == {"count": 5}
        assert calls[0].id == "call_create_warrant"

    def test_skips_interleaved_reasoning_items(self):
        # response_stub prepends a reasoning item by default — exactly what
        # the live API does with reasoning on. Anything that reads output
        # positionally breaks here.
        resp = response_stub(calls=[("a", "{}"), ("b", "{}")])
        assert [c.name for c in oc.tool_calls(resp)] == ["a", "b"]

    def test_skips_message_items(self):
        resp = response_stub(calls=[("a", "{}")], text="some prose")
        assert [c.name for c in oc.tool_calls(resp)] == ["a"]

    def test_preserves_call_order(self):
        resp = response_stub(calls=[("first", "{}"), ("second", "{}"),
                                    ("third", "{}")])
        assert [c.name for c in oc.tool_calls(resp)] == [
            "first", "second", "third"]

    def test_empty_when_no_calls(self):
        assert oc.tool_calls(response_stub(text="prose only")) == []

    def test_empty_when_output_missing(self):
        assert oc.tool_calls(types.SimpleNamespace()) == []

    def test_empty_when_output_none(self):
        assert oc.tool_calls(types.SimpleNamespace(output=None)) == []

    def test_malformed_json_falls_back_to_raw(self):
        # A truncated / poisoned arguments blob must not sink the whole
        # filing — parse.py sees the sentinel and logs a drop instead.
        resp = response_stub(calls=[("create_warrant", '{"count": ')])
        args = oc.tool_calls(resp)[0].arguments
        assert args == {"__raw_arguments__": '{"count": '}

    def test_empty_arguments_string_becomes_empty_dict(self):
        resp = response_stub(calls=[("note_no_event", "")])
        assert oc.tool_calls(resp)[0].arguments == {}

    def test_falls_back_to_id_when_call_id_absent(self):
        item = types.SimpleNamespace(type="function_call", name="x",
                                     arguments="{}", id="fc_123")
        calls = oc.tool_calls(types.SimpleNamespace(output=[item]))
        assert calls[0].id == "fc_123"

    def test_tool_call_is_frozen(self):
        call = oc.tool_calls(response_stub(calls=[("a", "{}")]))[0]
        with pytest.raises(Exception):
            call.name = "mutated"


# ════════════════════════════════════════════════════════════════════
# output_text
# ════════════════════════════════════════════════════════════════════
class TestOutputText:
    def test_prefers_sdk_convenience_property(self):
        resp = types.SimpleNamespace(output_text="from-property", output=[])
        assert oc.output_text(resp) == "from-property"

    def test_assembles_from_message_items_when_property_missing(self):
        item = types.SimpleNamespace(
            type="message",
            content=[types.SimpleNamespace(text="part1"),
                     types.SimpleNamespace(text="part2")],
        )
        resp = types.SimpleNamespace(output=[item])
        assert oc.output_text(resp) == "part1part2"

    def test_skips_non_message_items_when_assembling(self):
        reasoning = types.SimpleNamespace(type="reasoning", summary=[])
        msg = types.SimpleNamespace(
            type="message", content=[types.SimpleNamespace(text="only-this")])
        resp = types.SimpleNamespace(output=[reasoning, msg])
        assert oc.output_text(resp) == "only-this"

    def test_empty_string_when_tools_only(self):
        assert oc.output_text(response_stub(calls=[("a", "{}")])) == ""

    def test_empty_string_when_no_output(self):
        assert oc.output_text(types.SimpleNamespace()) == ""


# ════════════════════════════════════════════════════════════════════
# truncated
# ════════════════════════════════════════════════════════════════════
class TestTruncated:
    def test_true_only_for_max_output_tokens(self):
        assert oc.truncated(response_stub(
            text="x", status="incomplete",
            incomplete_reason="max_output_tokens")) is True

    def test_false_for_other_incomplete_reasons(self):
        assert oc.truncated(response_stub(
            text="x", status="incomplete",
            incomplete_reason="content_filter")) is False

    def test_false_for_completed(self):
        assert oc.truncated(response_stub(text="x")) is False

    def test_false_when_incomplete_details_missing(self):
        assert oc.truncated(
            types.SimpleNamespace(status="incomplete")) is False

    def test_false_when_status_missing(self):
        assert oc.truncated(types.SimpleNamespace()) is False


# ════════════════════════════════════════════════════════════════════
# max_input_chars
# ════════════════════════════════════════════════════════════════════
class TestMaxInputChars:
    def test_derived_from_config(self, monkeypatch):
        monkeypatch.setattr(config, "OPENAI_MAX_INPUT_TOKENS", 1000)
        monkeypatch.setattr(config, "CHARS_PER_TOKEN_FLOOR", 3)
        assert oc.max_input_chars() == 3000

    def test_production_value(self):
        # 922,000 input tokens × 3 chars/token floor.
        assert oc.max_input_chars() == 2_766_000

    def test_uses_the_floor_not_a_typical_ratio(self, monkeypatch):
        # SEC markdown really runs 3.5-4 chars/token; the cap deliberately
        # assumes the densest case so it can never overshoot the window.
        monkeypatch.setattr(config, "OPENAI_MAX_INPUT_TOKENS", 100)
        monkeypatch.setattr(config, "CHARS_PER_TOKEN_FLOOR", 3)
        assert oc.max_input_chars() == 300


# ════════════════════════════════════════════════════════════════════
# require_api_key / clients
# ════════════════════════════════════════════════════════════════════
class TestRequireApiKey:
    def test_raises_when_empty(self, monkeypatch):
        monkeypatch.setattr(config, "OPENAI_API_KEY", "")
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            oc.require_api_key()

    def test_passes_when_set(self, monkeypatch):
        monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
        assert oc.require_api_key() is None

    def test_client_factories_check_the_key(self, monkeypatch):
        monkeypatch.setattr(config, "OPENAI_API_KEY", "")
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            oc.make_async_client()
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            oc.make_sync_client()


class TestClientKwargs:
    def test_timeout_and_retries_set(self, monkeypatch):
        monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
        kw = oc._client_kwargs()
        assert kw["api_key"] == "sk-test"
        assert kw["max_retries"] == oc._MAX_RETRIES
        assert kw["timeout"] is oc._HTTP_TIMEOUT

    def test_base_url_omitted_when_blank(self, monkeypatch):
        monkeypatch.setattr(config, "OPENAI_BASE_URL", "")
        assert "base_url" not in oc._client_kwargs()

    def test_base_url_forwarded_when_set(self, monkeypatch):
        monkeypatch.setattr(config, "OPENAI_BASE_URL", "https://proxy.local/v1")
        assert oc._client_kwargs()["base_url"] == "https://proxy.local/v1"


# ════════════════════════════════════════════════════════════════════
# complete / acomplete
# ════════════════════════════════════════════════════════════════════
class _FakeResponses:
    def __init__(self, resp):
        self._resp = resp
        self.calls: list[dict] = []

    def create(self, **kw):
        self.calls.append(kw)
        return self._resp


class _FakeAsyncResponses(_FakeResponses):
    async def create(self, **kw):
        self.calls.append(kw)
        return self._resp


def _fake_client(resp, *, is_async=False):
    cls = _FakeAsyncResponses if is_async else _FakeResponses
    responses = cls(resp)
    return types.SimpleNamespace(responses=responses), responses


class TestComplete:
    def test_sync_passes_payload_and_returns_response(self):
        resp = response_stub(text="ok")
        client, responses = _fake_client(resp)
        out = oc.complete(client, name="brief", messages=_msgs(),
                          max_output_tokens=32)
        assert out is resp
        assert len(responses.calls) == 1
        assert responses.calls[0]["input"] == _msgs()
        assert responses.calls[0]["max_output_tokens"] == 32

    def test_async_passes_payload_and_returns_response(self):
        import asyncio
        resp = response_stub(calls=[("a", "{}")])
        client, responses = _fake_client(resp, is_async=True)
        out = asyncio.run(oc.acomplete(client, name="walker",
                                       messages=_msgs(),
                                       max_output_tokens=32))
        assert out is resp
        assert len(responses.calls) == 1

    def test_name_is_not_forwarded_to_the_api(self):
        # `name` labels the Langfuse span; the API has no such parameter.
        resp = response_stub(text="ok")
        client, responses = _fake_client(resp)
        oc.complete(client, name="brief", messages=_msgs(),
                    max_output_tokens=32)
        assert "name" not in responses.calls[0]

    def test_tier_downgrade_warns(self, caplog, monkeypatch):
        monkeypatch.setattr(config, "OPENAI_SERVICE_TIER", "flex")
        client, _ = _fake_client(
            response_stub(text="ok", service_tier="default"))
        with caplog.at_level("WARNING",
                             logger="dilution.openai_client"):
            oc.complete(client, name="brief", messages=_msgs(),
                        max_output_tokens=32)
        assert "service_tier downgrade" in caplog.text

    def test_matching_tier_is_quiet(self, caplog, monkeypatch):
        monkeypatch.setattr(config, "OPENAI_SERVICE_TIER", "flex")
        client, _ = _fake_client(response_stub(text="ok", service_tier="flex"))
        with caplog.at_level("WARNING",
                             logger="dilution.openai_client"):
            oc.complete(client, name="brief", messages=_msgs(),
                        max_output_tokens=32)
        assert "service_tier downgrade" not in caplog.text

    def test_missing_tier_on_response_is_quiet(self, caplog, monkeypatch):
        # A stub or a future API shape without the echo must not warn.
        monkeypatch.setattr(config, "OPENAI_SERVICE_TIER", "flex")
        client, _ = _fake_client(types.SimpleNamespace(
            output=[], output_text="", status="completed", usage=None))
        with caplog.at_level("WARNING",
                             logger="dilution.openai_client"):
            oc.complete(client, name="brief", messages=_msgs(),
                        max_output_tokens=32)
        assert "service_tier downgrade" not in caplog.text
