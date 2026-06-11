"""LLM provider abstraction.

Wraps xAI, Moonshot, and Gemini under one interface so the extractors
don't care which model is running. Switch by editing config.LLM_PROVIDER.

The wrapper mimics xai-sdk's client shape since the extractors were
written against it:

    client = make_async_client()
    chat = client.chat.create(model=..., max_tokens=..., response_format=Pydantic)
    chat.append(system("..."))
    chat.append(user("..."))
    response = await chat.sample()
    response.content        # JSON string
    response.finish_reason  # "REASON_MAX_LEN" / "REASON_NORMAL" / ...

Provider-specific quirks hidden in the wrapper:
  - Moonshot K2.6 enforces fixed temperature and seed; the Moonshot
    branch silently drops temperature/seed kwargs from make_chat().
  - Moonshot doesn't accept a Pydantic class as response_format; the
    branch translates to {"type": "json_object"}. Pydantic validation
    happens downstream in parse_event_list / parse_overhang_list, which
    already returns [] on schema-validation failure.
  - Moonshot's finish_reason ("stop"/"length") is translated to xAI's
    enum names ("REASON_NORMAL"/"REASON_MAX_LEN") so check_response()
    works for both providers.
  - Moonshot thinking mode is set from config.MOONSHOT_THINKING.
  - Gemini speaks OpenAI-compatible at GEMINI_BASE_URL. It accepts
    temperature (passed through), but rejects seed with HTTP 400
    INVALID_ARGUMENT, so the branch silently drops seed kwargs. Unlike
    Moonshot, Gemini supports strict json_schema response_format (as of
    Nov 2025 anyOf / $ref / enum / null / additionalProperties are all
    honored), so we translate a Pydantic class via model_json_schema()
    and pass it through. This enforces Literal enums and required
    fields at decode time, matching xAI's structured-output behavior.
    finish_reason translation reuses the OpenAI→xAI map. When
    config.GEMINI_SERVICE_TIER is set to "flex" (default), the branch
    forwards service_tier via extra_body so we get the 50% Flex
    discount; openai-python's built-in 429/503 retry handles the
    variable-latency / sheddable-capacity tradeoff.
"""

import json
from dataclasses import dataclass

import httpx

import config

from .observability import (
    _openai_usage,
    _xai_usage,
    llm_generation,
)


# ─── HTTP timeouts for OpenAI-compat providers ────────────────────────
# Without an explicit timeout, openai-python uses its built-in default,
# which combined with no application-level cap let a single wedged call
# stall a walker silently for tens of minutes. We set explicit values
# per Google's Flex-tier guidance:
#   https://ai.google.dev/gemini-api/docs/flex-inference
# Flex requests may queue, so the client timeout must be generous —
# Google's docs recommend 10 min or more (their examples cite 15 min for
# non-streaming). We use 600s for Flex, 120s for non-Flex/Moonshot where
# requests don't queue. max_retries is bumped from the SDK default of 2
# to 6: 503 UNAVAILABLE from Gemini Flex during demand spikes was
# leaking past the SDK's retry budget and surfacing as walker errors.
# Backoff interval stays at the SDK default (INITIAL=0.5s, MAX=8s, so
# 6 retries cover ~23s total) — Google warns against aggressive
# retries on Flex, and retry-count is the cheaper knob than interval.
_HTTP_TIMEOUT_FLEX = httpx.Timeout(600.0, connect=10.0)
_HTTP_TIMEOUT_STANDARD = httpx.Timeout(120.0, connect=10.0)
_MAX_RETRIES = 6


def _gemini_http_timeout() -> httpx.Timeout:
    """Pick the per-request HTTP timeout for the Gemini client based on
    the configured service tier. Flex queues; standard does not."""
    tier = (config.GEMINI_SERVICE_TIER or "standard").lower()
    return _HTTP_TIMEOUT_FLEX if tier == "flex" else _HTTP_TIMEOUT_STANDARD


# ─── Provider-neutral message helpers ─────────────────────────────────
# Tuples flow through; each branch translates at append time.
def system(text: str) -> tuple[str, str]:
    return ("system", text)


def user(text: str) -> tuple[str, str]:
    return ("user", text)


# ─── Provider-neutral tool-call shape ─────────────────────────────────
# Each provider returns tool calls in its own protobuf / dict shape;
# both are normalized to this dataclass so the walker doesn't branch on
# provider when iterating calls.
@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


# ─── Response wrapper ─────────────────────────────────────────────────
class _Response:
    __slots__ = ("content", "finish_reason", "tool_calls")

    def __init__(self, content: str, finish_reason: str | None,
                 tool_calls: list[ToolCall] | None = None):
        self.content = content
        self.finish_reason = finish_reason
        # None when the call did not use tool calling; empty list when
        # the model declined to call any tool despite tools being
        # offered. Distinguishes "no tools offered" from "tools offered
        # but model chose prose".
        self.tool_calls = tool_calls


def _parse_xai_tool_calls(proto_calls) -> list[ToolCall]:
    """xAI returns tool_calls as protobuf ToolCall messages with a
    nested .function.name + .function.arguments (JSON string)."""
    out: list[ToolCall] = []
    for tc in proto_calls:
        fn = tc.function
        try:
            args = json.loads(fn.arguments) if fn.arguments else {}
        except json.JSONDecodeError:
            args = {"__raw_arguments__": fn.arguments}
        out.append(ToolCall(id=tc.id, name=fn.name, arguments=args))
    return out


def _parse_openai_tool_calls(message_tool_calls) -> list[ToolCall]:
    """OpenAI-compat (Moonshot, Gemini) returns tool_calls on the
    message object with .id, .function.name, .function.arguments
    (JSON string)."""
    if not message_tool_calls:
        return []
    out: list[ToolCall] = []
    for tc in message_tool_calls:
        fn = tc.function
        try:
            args = json.loads(fn.arguments) if fn.arguments else {}
        except json.JSONDecodeError:
            args = {"__raw_arguments__": fn.arguments}
        out.append(ToolCall(id=tc.id, name=fn.name, arguments=args))
    return out


# ─── xAI branch ───────────────────────────────────────────────────────
class _XaiSyncChat:
    def __init__(self, xai_chat, *, model: str | None = None):
        self._chat = xai_chat
        self._model = model
        self._messages: list[tuple[str, str]] = []

    def append(self, msg: tuple[str, str]):
        from xai_sdk.chat import system as _xs, user as _xu
        role, content = msg
        self._chat.append((_xs if role == "system" else _xu)(content))
        self._messages.append(msg)

    def sample(self):
        with llm_generation(
            name="xai-chat", model=self._model, messages=self._messages,
        ) as gen:
            r = self._chat.sample()
            if gen is not None:
                gen.update(
                    output=r.content,
                    usage_details=_xai_usage(r),
                    metadata={"finish_reason": getattr(r, "finish_reason", None)},
                )
        raw_calls = getattr(r, "tool_calls", None) or None
        return _Response(
            r.content,
            getattr(r, "finish_reason", None),
            tool_calls=_parse_xai_tool_calls(raw_calls) if raw_calls else None,
        )


class _XaiAsyncChat:
    def __init__(self, xai_chat, *, model: str | None = None):
        self._chat = xai_chat
        self._model = model
        self._messages: list[tuple[str, str]] = []

    def append(self, msg: tuple[str, str]):
        from xai_sdk.chat import system as _xs, user as _xu
        role, content = msg
        self._chat.append((_xs if role == "system" else _xu)(content))
        self._messages.append(msg)

    async def sample(self):
        with llm_generation(
            name="xai-chat", model=self._model, messages=self._messages,
        ) as gen:
            r = await self._chat.sample()
            if gen is not None:
                gen.update(
                    output=r.content,
                    usage_details=_xai_usage(r),
                    metadata={"finish_reason": getattr(r, "finish_reason", None)},
                )
        raw_calls = getattr(r, "tool_calls", None) or None
        return _Response(
            r.content,
            getattr(r, "finish_reason", None),
            tool_calls=_parse_xai_tool_calls(raw_calls) if raw_calls else None,
        )


class _XaiChatFactory:
    def __init__(self, xai_client, is_async: bool):
        self._client = xai_client
        self._is_async = is_async

    def create(self, **kwargs):
        chat = self._client.chat.create(**kwargs)
        model = kwargs.get("model")
        cls = _XaiAsyncChat if self._is_async else _XaiSyncChat
        return cls(chat, model=model)


class _XaiSyncClient:
    def __init__(self, xai_client):
        self._client = xai_client
        self.chat = _XaiChatFactory(xai_client, is_async=False)


class _XaiAsyncClient:
    def __init__(self, xai_client):
        self._client = xai_client
        self.chat = _XaiChatFactory(xai_client, is_async=True)

    async def close(self):
        close = getattr(self._client, "close", None)
        if close is None:
            return
        result = close()
        if hasattr(result, "__await__"):
            await result


# ─── Moonshot branch ──────────────────────────────────────────────────
_FINISH_TRANSLATE = {
    "stop": "REASON_NORMAL",
    "length": "REASON_MAX_LEN",
    "content_filter": "REASON_CONTENT_FILTER",
    "tool_calls": "REASON_TOOL_CALLS",
}


def _translate_finish_reason(reason):
    if reason is None:
        return None
    return _FINISH_TRANSLATE.get(reason, reason)


def _moonshot_request_kwargs(model, max_tokens, response_format,
                              tools, tool_choice):
    kwargs = {"model": model}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        # Pydantic class → json_object mode. Schema enforcement happens
        # downstream via Pydantic in parse_event_list / parse_overhang_list.
        kwargs["response_format"] = {"type": "json_object"}
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    kwargs["extra_body"] = {
        "thinking": {"type": config.MOONSHOT_THINKING},
    }
    return kwargs


class _MoonshotSyncChat:
    def __init__(self, openai_client, *, model, max_tokens=None,
                 response_format=None, tools=None, tool_choice=None,
                 **_dropped):
        # _dropped absorbs temperature / seed — Moonshot rejects custom
        # values and we want a quiet drop, not an error.
        self._oai = openai_client
        self._model = model
        self._kwargs = _moonshot_request_kwargs(
            model, max_tokens, response_format, tools, tool_choice)
        self._messages: list[dict] = []

    def append(self, msg: tuple[str, str]):
        role, content = msg
        self._messages.append({"role": role, "content": content})

    def sample(self):
        with llm_generation(
            name="moonshot-chat", model=self._model, messages=self._messages,
        ) as gen:
            resp = self._oai.chat.completions.create(
                messages=self._messages, **self._kwargs)
            choice = resp.choices[0]
            if gen is not None:
                gen.update(
                    output=choice.message.content,
                    usage_details=_openai_usage(resp),
                    metadata={"finish_reason": choice.finish_reason},
                )
        raw_calls = getattr(choice.message, "tool_calls", None)
        return _Response(
            choice.message.content,
            _translate_finish_reason(choice.finish_reason),
            tool_calls=_parse_openai_tool_calls(raw_calls) if raw_calls else None,
        )


class _MoonshotAsyncChat(_MoonshotSyncChat):
    async def sample(self):
        with llm_generation(
            name="moonshot-chat", model=self._model, messages=self._messages,
        ) as gen:
            resp = await self._oai.chat.completions.create(
                messages=self._messages, **self._kwargs)
            choice = resp.choices[0]
            if gen is not None:
                gen.update(
                    output=choice.message.content,
                    usage_details=_openai_usage(resp),
                    metadata={"finish_reason": choice.finish_reason},
                )
        raw_calls = getattr(choice.message, "tool_calls", None)
        return _Response(
            choice.message.content,
            _translate_finish_reason(choice.finish_reason),
            tool_calls=_parse_openai_tool_calls(raw_calls) if raw_calls else None,
        )


class _MoonshotChatFactory:
    def __init__(self, openai_client, is_async: bool):
        self._client = openai_client
        self._is_async = is_async

    def create(self, **kwargs):
        cls = _MoonshotAsyncChat if self._is_async else _MoonshotSyncChat
        return cls(self._client, **kwargs)


class _MoonshotSyncClient:
    def __init__(self, openai_client):
        self._client = openai_client
        self.chat = _MoonshotChatFactory(openai_client, is_async=False)


class _MoonshotAsyncClient:
    def __init__(self, openai_client):
        self._client = openai_client
        self.chat = _MoonshotChatFactory(openai_client, is_async=True)

    async def close(self):
        close = getattr(self._client, "close", None)
        if close is None:
            return
        result = close()
        if hasattr(result, "__await__"):
            await result


# ─── Gemini branch ────────────────────────────────────────────────────
# Same OpenAI-compatible surface as Moonshot, but Gemini accepts
# temperature (Moonshot rejects it) and rejects seed (Moonshot accepts
# fixed only). No thinking extra_body. Unlike Moonshot, Gemini's
# OpenAI-compat endpoint supports json_schema mode (since Nov 2025 it
# accepts anyOf, $ref, enum, null types, additionalProperties), so we
# translate a Pydantic class to a strict json_schema response_format —
# this enforces required fields and Literal enums at decode time.
def _gemini_response_format(response_format):
    """Convert a Pydantic class (or dict) to Gemini-compatible response_format."""
    if response_format is None:
        return None
    if isinstance(response_format, dict):
        return response_format
    # Pydantic class → json_schema. Falls back to json_object if the
    # class doesn't expose model_json_schema (defensive — shouldn't hit).
    if hasattr(response_format, "model_json_schema"):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "schema": response_format.model_json_schema(),
            },
        }
    return {"type": "json_object"}


def _gemini_request_kwargs(model, max_tokens, response_format, temperature,
                            tools, tool_choice):
    kwargs = {"model": model}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    rf = _gemini_response_format(response_format)
    if rf is not None:
        kwargs["response_format"] = rf
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    # Service tier (flex = 50% off, variable latency, 503/429 risk).
    # Goes via extra_body so the OpenAI-compat layer forwards it
    # untouched into the REST request body, where Gemini reads it.
    tier = config.GEMINI_SERVICE_TIER
    if tier and tier != "standard":
        kwargs["extra_body"] = {"service_tier": tier}
    return kwargs


class _GeminiSyncChat:
    def __init__(self, openai_client, *, model, max_tokens=None,
                 response_format=None, temperature=None,
                 tools=None, tool_choice=None, **_dropped):
        # _dropped absorbs seed — Gemini's OpenAI-compat endpoint rejects
        # it with 400 INVALID_ARGUMENT. The walker's per-attempt seed
        # perturbation (walker_llm.py) becomes a no-op on this provider;
        # determinism is best-effort via temperature=0 only.
        self._oai = openai_client
        self._model = model
        self._kwargs = _gemini_request_kwargs(
            model, max_tokens, response_format, temperature, tools, tool_choice)
        self._messages: list[dict] = []

    def append(self, msg: tuple[str, str]):
        role, content = msg
        self._messages.append({"role": role, "content": content})

    def sample(self):
        with llm_generation(
            name="gemini-chat", model=self._model, messages=self._messages,
        ) as gen:
            resp = self._oai.chat.completions.create(
                messages=self._messages, **self._kwargs)
            choice = resp.choices[0]
            if gen is not None:
                gen.update(
                    output=choice.message.content,
                    usage_details=_openai_usage(resp),
                    metadata={"finish_reason": choice.finish_reason},
                )
        raw_calls = getattr(choice.message, "tool_calls", None)
        return _Response(
            choice.message.content,
            _translate_finish_reason(choice.finish_reason),
            tool_calls=_parse_openai_tool_calls(raw_calls) if raw_calls else None,
        )


class _GeminiAsyncChat(_GeminiSyncChat):
    async def sample(self):
        with llm_generation(
            name="gemini-chat", model=self._model, messages=self._messages,
        ) as gen:
            resp = await self._oai.chat.completions.create(
                messages=self._messages, **self._kwargs)
            choice = resp.choices[0]
            if gen is not None:
                gen.update(
                    output=choice.message.content,
                    usage_details=_openai_usage(resp),
                    metadata={"finish_reason": choice.finish_reason},
                )
        raw_calls = getattr(choice.message, "tool_calls", None)
        return _Response(
            choice.message.content,
            _translate_finish_reason(choice.finish_reason),
            tool_calls=_parse_openai_tool_calls(raw_calls) if raw_calls else None,
        )


class _GeminiChatFactory:
    def __init__(self, openai_client, is_async: bool):
        self._client = openai_client
        self._is_async = is_async

    def create(self, **kwargs):
        cls = _GeminiAsyncChat if self._is_async else _GeminiSyncChat
        return cls(self._client, **kwargs)


class _GeminiSyncClient:
    def __init__(self, openai_client):
        self._client = openai_client
        self.chat = _GeminiChatFactory(openai_client, is_async=False)


class _GeminiAsyncClient:
    def __init__(self, openai_client):
        self._client = openai_client
        self.chat = _GeminiChatFactory(openai_client, is_async=True)

    async def close(self):
        close = getattr(self._client, "close", None)
        if close is None:
            return
        result = close()
        if hasattr(result, "__await__"):
            await result


# ─── Public factories ─────────────────────────────────────────────────
def make_async_client():
    require_api_key()
    if config.LLM_PROVIDER == "moonshot":
        from openai import AsyncOpenAI
        return _MoonshotAsyncClient(AsyncOpenAI(
            api_key=config.MOONSHOT_API_KEY,
            base_url=config.MOONSHOT_BASE_URL,
            timeout=_HTTP_TIMEOUT_STANDARD,
            max_retries=_MAX_RETRIES,
        ))
    if config.LLM_PROVIDER == "gemini":
        from openai import AsyncOpenAI
        return _GeminiAsyncClient(AsyncOpenAI(
            api_key=config.GEMINI_API_KEY,
            base_url=config.GEMINI_BASE_URL,
            timeout=_gemini_http_timeout(),
            max_retries=_MAX_RETRIES,
        ))
    from xai_sdk.aio.client import Client as XaiAsyncClient
    return _XaiAsyncClient(XaiAsyncClient(api_key=config.XAI_API_KEY))


def make_sync_client():
    require_api_key()
    if config.LLM_PROVIDER == "moonshot":
        from openai import OpenAI
        return _MoonshotSyncClient(OpenAI(
            api_key=config.MOONSHOT_API_KEY,
            base_url=config.MOONSHOT_BASE_URL,
            timeout=_HTTP_TIMEOUT_STANDARD,
            max_retries=_MAX_RETRIES,
        ))
    if config.LLM_PROVIDER == "gemini":
        from openai import OpenAI
        return _GeminiSyncClient(OpenAI(
            api_key=config.GEMINI_API_KEY,
            base_url=config.GEMINI_BASE_URL,
            timeout=_gemini_http_timeout(),
            max_retries=_MAX_RETRIES,
        ))
    from xai_sdk import Client as XaiSyncClient
    return _XaiSyncClient(XaiSyncClient(api_key=config.XAI_API_KEY))


def require_api_key() -> None:
    """Raise if the active provider's API key is missing."""
    if config.LLM_PROVIDER == "moonshot":
        if not config.MOONSHOT_API_KEY:
            raise RuntimeError(
                "MOONSHOT_API_KEY not set (config.LLM_PROVIDER=moonshot)")
    elif config.LLM_PROVIDER == "gemini":
        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY not set (config.LLM_PROVIDER=gemini)")
    elif config.LLM_PROVIDER == "xai":
        if not config.XAI_API_KEY:
            raise RuntimeError(
                "XAI_API_KEY not set (config.LLM_PROVIDER=xai)")
    else:
        raise RuntimeError(
            f"unknown config.LLM_PROVIDER={config.LLM_PROVIDER!r} "
            f"— expected 'xai', 'moonshot', or 'gemini'")
