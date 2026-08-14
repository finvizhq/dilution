"""OpenAI client + request/response plumbing for every LLM call.

One vendor, one endpoint: `/v1/responses`. This module owns the parts
that must not be re-derived per call site — client construction, the
request payload, tool-call normalization, truncation detection, and the
Langfuse span — and nothing else. Call sites build a list of message
dicts, call `acomplete()` / `complete()`, and read the raw SDK
`Response` through the small readers below.

Why Responses and not Chat Completions: the walker is entirely
tool-driven, and on chat completions function tools plus any reasoning
effort above "none" is rejected outright —

    400 Function tools with reasoning_effort are not supported for
        gpt-5.6-luna in /v1/chat/completions. To use function tools,
        use /v1/responses or set reasoning_effort to 'none'.

Shape differences from the chat-completions world, all load-bearing:
  - request: `input` (not `messages`), flat tool objects (no nested
    "function" key), `text.format` (not `response_format`),
    `max_output_tokens` (not `max_tokens` / `max_completion_tokens`)
  - response: an `output` LIST whose items are interleaved `reasoning`,
    `function_call` and `message` entries — never index it positionally
  - truncation: `status="incomplete"` + `incomplete_details.reason`,
    not `finish_reason="length"`
  - usage: `input_tokens` / `output_tokens` with cached and reasoning
    counts nested in `*_tokens_details`
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import httpx

import config

from .observability import _openai_usage, llm_generation

log = logging.getLogger(__name__)


# ─── HTTP timeouts ────────────────────────────────────────────────────
# Without an explicit timeout, openai-python's built-in default combined
# with no application-level cap let a single wedged call stall a walker
# silently for tens of minutes. Flex requests are queued and sheddable,
# so the client timeout has to be generous — 600s covers the long tail
# without hiding a wedge forever. max_retries is raised from the SDK
# default of 2 to 6 because 503 UNAVAILABLE during flex demand spikes
# was leaking past the SDK's retry budget and surfacing as walker
# errors. Backoff interval stays at the SDK default (0.5s → 8s, so 6
# retries cover ~23s): retry count is the cheaper knob than interval,
# and hammering a shedding tier faster does not help.
_HTTP_TIMEOUT = httpx.Timeout(600.0, connect=10.0)
_MAX_RETRIES = 6


# ─── Tool-call shape ──────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


# ─── Clients ──────────────────────────────────────────────────────────
def _client_kwargs() -> dict:
    kwargs = {
        "api_key": config.OPENAI_API_KEY,
        "timeout": _HTTP_TIMEOUT,
        "max_retries": _MAX_RETRIES,
    }
    if config.OPENAI_BASE_URL:
        kwargs["base_url"] = config.OPENAI_BASE_URL
    return kwargs


def make_async_client():
    """Async client. Callers own its lifecycle (`await client.close()`)."""
    require_api_key()
    from openai import AsyncOpenAI
    return AsyncOpenAI(**_client_kwargs())


def make_sync_client():
    require_api_key()
    from openai import OpenAI
    return OpenAI(**_client_kwargs())


def require_api_key() -> None:
    """Raise if the API key is missing."""
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")


# ─── Message helpers ──────────────────────────────────────────────────
def system(text: str) -> dict:
    return {"role": "system", "content": text}


def user(text: str) -> dict:
    return {"role": "user", "content": text}


# ─── Input sizing ─────────────────────────────────────────────────────
def max_input_chars() -> int:
    """Filing-text char cap that cannot overflow the input window.

    Both tiers share the same 922,000-token input ceiling, so this is a
    single number rather than a per-model lookup, and no filing has to
    escalate between tiers to fit. Derived at the 3-chars/token floor so
    the densest possible SEC markdown still lands inside the window.
    """
    return config.OPENAI_MAX_INPUT_TOKENS * config.CHARS_PER_TOKEN_FLOOR


# ─── Request payload ──────────────────────────────────────────────────
def _text_format(response_format) -> dict | None:
    """Pydantic class (or pre-built dict) → Responses `text` param.

    `strict: False` MUST be explicit here. On /v1/responses `text.format`
    defaults strict to TRUE — the opposite of chat completions, where
    omitting the key meant non-strict — and strict mode requires every key
    in `properties` to appear in `required`, which our Pydantic models
    violate by design (Optional fields with defaults). Omitting it fails
    every overhang call with

        400 Invalid schema for response_format 'CombinedOverhangList':
            'required' is required to be supplied and to be an array
            including every key in properties. Missing 'instrument_name'.

    Non-strict json_schema still constrains shape and Literal enums at
    decode time, and parse_overhang_list / model_validate_json validate
    downstream anyway, returning [] on a schema miss.
    """
    if response_format is None:
        return None
    if isinstance(response_format, dict):
        return response_format
    if hasattr(response_format, "model_json_schema"):
        return {
            "format": {
                "type": "json_schema",
                "name": response_format.__name__,
                "schema": response_format.model_json_schema(),
                "strict": False,
            }
        }
    return {"format": {"type": "json_object"}}


def request_kwargs(*, messages: list[dict], model: str | None = None,
                   max_output_tokens: int,
                   response_format=None,
                   tools=None, tool_choice=None,
                   cache_key: str | None = None) -> dict:
    """Build the /v1/responses payload with our standard parameters.

    NO temperature, top_p or seed — not an oversight. In reasoning mode
    the gpt-5.6 family rejects `temperature` and `top_p` ("not supported
    with this model"), and `seed` is not a Responses parameter at all
    ("Unknown parameter: 'seed'"). Adding any of them 400s every call in
    the pipeline. See the DETERMINISM NOTE in config.py.

    Two mutually-compatible constraint modes:
      - `response_format` (Pydantic class): structured-output path for
        callers emitting one typed JSON value (the overhang extractor's
        event lists, the ticker brief). Constrains shape + Literal enums
        but cannot enforce required dict keys.
      - `tools` + `tool_choice`: function calling. Each tool's
        JSON-Schema parameters block IS the contract (required args,
        types, minLength, enum, additionalProperties=false), enforced at
        decode time. `tool_choice="required"` forces ≥1 call. This is
        why the walker uses tools rather than structured output.
    """
    kwargs: dict = {
        "model": model if model is not None else config.LLM_MODEL,
        "input": messages,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": config.OPENAI_REASONING_EFFORT},
        "service_tier": config.OPENAI_SERVICE_TIER,
        # Nothing downstream reads the stored response, and the walker
        # ships filing text we have no reason to leave on a vendor.
        "store": False,
    }
    text = _text_format(response_format)
    if text is not None:
        kwargs["text"] = text
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if cache_key:
        # Steers OpenAI's automatic prefix cache. Our system prompt +
        # tool schemas are a large stable prefix and cached input is 10×
        # cheaper, so a stable key per call class is real money.
        kwargs["prompt_cache_key"] = cache_key
    return kwargs


# ─── Calls ────────────────────────────────────────────────────────────
def _check_tier(resp) -> None:
    """Warn when a response did not run on the tier we asked for."""
    got = getattr(resp, "service_tier", None)
    want = config.OPENAI_SERVICE_TIER
    if got and want and got != want:
        log.warning("service_tier downgrade — asked %r, billed %r",
                    want, got)


def complete(client, *, name: str, **kwargs):
    """Sync /v1/responses call wrapped in a Langfuse generation span."""
    payload = request_kwargs(**kwargs)
    with llm_generation(name=name, model=payload["model"],
                        messages=payload["input"]) as gen:
        resp = client.responses.create(**payload)
        if gen is not None:
            gen.update(output=output_text(resp),
                       usage_details=_openai_usage(resp),
                       metadata={"status": resp.status})
    _check_tier(resp)
    return resp


async def acomplete(client, *, name: str, **kwargs):
    """Async /v1/responses call wrapped in a Langfuse generation span."""
    payload = request_kwargs(**kwargs)
    with llm_generation(name=name, model=payload["model"],
                        messages=payload["input"]) as gen:
        resp = await client.responses.create(**payload)
        if gen is not None:
            gen.update(output=output_text(resp),
                       usage_details=_openai_usage(resp),
                       metadata={"status": resp.status})
    _check_tier(resp)
    return resp


# ─── Response readers ─────────────────────────────────────────────────
def tool_calls(resp) -> list[ToolCall]:
    """Normalized function calls from a Responses `output` list.

    Filtering on item type is mandatory, not cosmetic: with reasoning
    enabled the model interleaves `reasoning` items among the
    `function_call` items, so anything that assumes a fixed position
    silently drops calls.
    """
    out: list[ToolCall] = []
    for item in (getattr(resp, "output", None) or []):
        if getattr(item, "type", None) != "function_call":
            continue
        raw = getattr(item, "arguments", None)
        try:
            args = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            args = {"__raw_arguments__": raw}
        out.append(ToolCall(
            id=getattr(item, "call_id", None) or getattr(item, "id", ""),
            name=getattr(item, "name", ""),
            arguments=args,
        ))
    return out


def output_text(resp) -> str:
    """Concatenated assistant text, or "" when the model only called tools."""
    text = getattr(resp, "output_text", None)
    if text:
        return text
    # Hand-assemble for stubs / older payloads that lack the SDK's
    # convenience property.
    parts: list[str] = []
    for item in (getattr(resp, "output", None) or []):
        if getattr(item, "type", None) != "message":
            continue
        for chunk in (getattr(item, "content", None) or []):
            piece = getattr(chunk, "text", None)
            if piece:
                parts.append(piece)
    return "".join(parts)


def truncated(resp) -> bool:
    """True when generation stopped at max_output_tokens.

    The Responses equivalent of finish_reason == "length", and the
    trigger for the overhang extractor's truncated-JSON salvage.
    """
    if getattr(resp, "status", None) != "incomplete":
        return False
    details = getattr(resp, "incomplete_details", None)
    return getattr(details, "reason", None) == "max_output_tokens"


__all__ = [
    "ToolCall",
    "acomplete",
    "complete",
    "make_async_client",
    "make_sync_client",
    "max_input_chars",
    "output_text",
    "request_kwargs",
    "require_api_key",
    "system",
    "tool_calls",
    "truncated",
    "user",
]
