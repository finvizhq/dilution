"""Langfuse observability bootstrap for the dilution pipeline.

Idempotent. Safe when Langfuse credentials are missing — every helper
becomes a no-op so callers can wrap unconditionally without branching.

Tracing surface:
  * Every LLM ``sample()`` in ``dilution.llm_provider`` emits a
    Langfuse generation observation (model, input messages, output
    content, token usage, finish reason). xAI proto and OpenAI usage
    shapes are both translated to Langfuse's ``usage_details``.
  * ``run_dilution.py`` wraps each pipeline run in a top-level span and
    propagates the ticker as ``session_id`` so every run for the same
    issuer groups together in the Sessions view.
  * Per-stage spans (``index``, ``unit``, ``fetch``, ``walk``) nest
    under the pipeline span — find the slow / failing stage at a
    glance.

Call ``setup_observability()`` once at startup and
``flush_observability()`` in a finally block before exit; otherwise
queued events never leave the box (Langfuse SDK v4 ships in the
background).
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

_INITIALIZED = False
_ENABLED = False


def setup_observability() -> bool:
    """Bootstrap the Langfuse singleton. Returns True iff tracing active.

    Reads ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` /
    ``LANGFUSE_BASE_URL`` (or ``LANGFUSE_HOST``) from the environment —
    ``config.py`` loads ``.env`` into ``os.environ`` before this runs.
    """
    global _INITIALIZED, _ENABLED
    if _INITIALIZED:
        return _ENABLED
    _INITIALIZED = True

    if not (os.environ.get("LANGFUSE_PUBLIC_KEY")
            and os.environ.get("LANGFUSE_SECRET_KEY")):
        log.info(
            "langfuse: LANGFUSE_PUBLIC_KEY/SECRET_KEY not set — tracing disabled")
        return False

    try:
        from langfuse import get_client
    except ImportError:
        log.info(
            "langfuse: package not installed — tracing disabled "
            "(`pip install langfuse` to enable)")
        return False

    try:
        client = get_client()
        if client.auth_check():
            host = (os.environ.get("LANGFUSE_BASE_URL")
                    or os.environ.get("LANGFUSE_HOST")
                    or "https://cloud.langfuse.com")
            log.info("langfuse: tracing enabled (host=%s)", host)
            _ENABLED = True
        else:
            log.warning("langfuse: auth_check failed — tracing disabled")
    except Exception as exc:
        log.warning("langfuse: setup failed (%s) — tracing disabled", exc)

    return _ENABLED


def is_enabled() -> bool:
    return _ENABLED


def flush_observability() -> None:
    """Block until pending traces are uploaded. Required at script exit
    because the SDK ships events in the background."""
    if not _ENABLED:
        return
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception as exc:
        log.warning("langfuse: flush failed: %s", exc)


@contextlib.contextmanager
def pipeline_session(ticker: str, *, name: str = "dilution-pipeline",
                     metadata: dict[str, Any] | None = None):
    """Wrap a pipeline run in a top-level span and propagate the ticker
    as ``session_id`` so child observations group together by issuer.

    No-op when tracing is disabled — callers don't need to branch.
    """
    if not _ENABLED:
        yield None
        return

    from langfuse import get_client, propagate_attributes

    with get_client().start_as_current_observation(
        as_type="span",
        name=name,
        input={"ticker": ticker.upper()},
        metadata=metadata or {},
    ) as span:
        with propagate_attributes(session_id=ticker.upper()):
            yield span


@contextlib.contextmanager
def stage(name: str, *, input: Any = None,
          metadata: dict[str, Any] | None = None):
    """Wrap a pipeline stage in a nested span. No-op when disabled."""
    if not _ENABLED:
        yield None
        return

    from langfuse import get_client

    with get_client().start_as_current_observation(
        as_type="span", name=name, input=input,
        metadata=metadata or {},
    ) as span:
        yield span


# ─── LLM-call instrumentation helpers (used by llm_provider) ─────────

def _xai_usage(response) -> dict[str, int] | None:
    """Translate xai_sdk Response.usage proto → Langfuse usage_details.

    Maps ``cached_prompt_text_tokens`` to ``cache_read_input_tokens``
    so the cache-hit savings show up in the Langfuse cost view.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    out = {
        "input": getattr(usage, "prompt_tokens", 0) or 0,
        "output": getattr(usage, "completion_tokens", 0) or 0,
        "total": getattr(usage, "total_tokens", 0) or 0,
    }
    cached = getattr(usage, "cached_prompt_text_tokens", 0) or 0
    if cached:
        out["cache_read_input_tokens"] = cached
    reasoning = getattr(usage, "reasoning_tokens", 0) or 0
    if reasoning:
        out["reasoning"] = reasoning
    return out


def _openai_usage(response) -> dict[str, int] | None:
    """Translate openai ChatCompletion.usage → Langfuse usage_details."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "input": getattr(usage, "prompt_tokens", 0) or 0,
        "output": getattr(usage, "completion_tokens", 0) or 0,
        "total": getattr(usage, "total_tokens", 0) or 0,
    }


@contextlib.contextmanager
def llm_generation(*, name: str, model: str | None,
                   messages: list[tuple[str, str]] | list[dict]):
    """Wrap an LLM call in a Langfuse generation observation.

    Yields the observation handle (or None when disabled). Caller is
    responsible for calling ``handle.update(output=..., usage_details=...)``
    after the model returns; on exception we mark the observation as
    ERROR and re-raise.
    """
    if not _ENABLED:
        yield None
        return

    from langfuse import get_client

    # Normalize tuple form to dicts for the UI.
    norm_messages = [
        {"role": m[0], "content": m[1]} if isinstance(m, tuple) else m
        for m in messages
    ]

    with get_client().start_as_current_observation(
        as_type="generation",
        name=name,
        model=model,
        input=norm_messages,
    ) as gen:
        try:
            yield gen
        except Exception as exc:
            try:
                gen.update(level="ERROR", status_message=str(exc))
            except Exception:
                pass
            raise
