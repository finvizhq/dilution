"""LLM provider abstraction.

Wraps both xAI and Moonshot under one interface so the extractors don't
care which model is running. Switch by editing config.LLM_PROVIDER.

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
"""

import config

from .observability import (
    _openai_usage,
    _xai_usage,
    llm_generation,
)


# ─── Provider-neutral message helpers ─────────────────────────────────
# Tuples flow through; each branch translates at append time.
def system(text: str) -> tuple[str, str]:
    return ("system", text)


def user(text: str) -> tuple[str, str]:
    return ("user", text)


# ─── Response wrapper ─────────────────────────────────────────────────
class _Response:
    __slots__ = ("content", "finish_reason")

    def __init__(self, content: str, finish_reason: str | None):
        self.content = content
        self.finish_reason = finish_reason


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
        return _Response(r.content, getattr(r, "finish_reason", None))


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
        return _Response(r.content, getattr(r, "finish_reason", None))


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


def _moonshot_request_kwargs(model, max_tokens, response_format):
    kwargs = {"model": model}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        # Pydantic class → json_object mode. Schema enforcement happens
        # downstream via Pydantic in parse_event_list / parse_overhang_list.
        kwargs["response_format"] = {"type": "json_object"}
    kwargs["extra_body"] = {
        "thinking": {"type": config.MOONSHOT_THINKING},
    }
    return kwargs


class _MoonshotSyncChat:
    def __init__(self, openai_client, *, model, max_tokens=None,
                 response_format=None, **_dropped):
        # _dropped absorbs temperature / seed — Moonshot rejects custom
        # values and we want a quiet drop, not an error.
        self._oai = openai_client
        self._model = model
        self._kwargs = _moonshot_request_kwargs(
            model, max_tokens, response_format)
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
        return _Response(
            choice.message.content,
            _translate_finish_reason(choice.finish_reason))


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
        return _Response(
            choice.message.content,
            _translate_finish_reason(choice.finish_reason))


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


# ─── Public factories ─────────────────────────────────────────────────
def make_async_client():
    require_api_key()
    if config.LLM_PROVIDER == "moonshot":
        from openai import AsyncOpenAI
        return _MoonshotAsyncClient(AsyncOpenAI(
            api_key=config.MOONSHOT_API_KEY,
            base_url=config.MOONSHOT_BASE_URL,
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
        ))
    from xai_sdk import Client as XaiSyncClient
    return _XaiSyncClient(XaiSyncClient(api_key=config.XAI_API_KEY))


def require_api_key() -> None:
    """Raise if the active provider's API key is missing."""
    if config.LLM_PROVIDER == "moonshot":
        if not config.MOONSHOT_API_KEY:
            raise RuntimeError(
                "MOONSHOT_API_KEY not set (config.LLM_PROVIDER=moonshot)")
    elif config.LLM_PROVIDER == "xai":
        if not config.XAI_API_KEY:
            raise RuntimeError(
                "XAI_API_KEY not set (config.LLM_PROVIDER=xai)")
    else:
        raise RuntimeError(
            f"unknown config.LLM_PROVIDER={config.LLM_PROVIDER!r} "
            f"— expected 'xai' or 'moonshot'")
