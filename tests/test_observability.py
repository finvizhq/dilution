"""Unit tests for dilution/observability.py.

This module is Langfuse glue. It touches no DB tables, so no temp_db
helpers are used (the autouse fixture still runs harmlessly). The two
pure usage translators are the highest-value targets; the rest are
io_mockable seams exercised by injecting a fake ``langfuse`` module into
sys.modules and/or toggling the module-global ``_ENABLED`` / ``_INITIALIZED``
flags.

Determinism: ``_INITIALIZED`` and ``_ENABLED`` are module-level booleans.
``reset_obs_state`` (autouse) restores both to False before every test and
removes any fake ``langfuse`` module we injected, so tests never leak state
into one another.
"""

from __future__ import annotations

import contextlib
import sys
import types

import pytest

from dilution import observability as o


# ─── shared scaffolding ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_obs_state(monkeypatch):
    """Restore module-global flags and purge any injected fake langfuse.

    Both flags are restored via the real module objects so a test that
    sets them directly cannot bleed into the next test. We snapshot and
    restore sys.modules['langfuse'] too.
    """
    monkeypatch.setattr(o, "_INITIALIZED", False, raising=False)
    monkeypatch.setattr(o, "_ENABLED", False, raising=False)
    saved = sys.modules.get("langfuse")
    yield
    # monkeypatch restores the module globals; clean up sys.modules.
    if saved is None:
        sys.modules.pop("langfuse", None)
    else:
        sys.modules["langfuse"] = saved


def usage(**kw):
    """Build a duck-typed usage object exposing exactly the given attrs."""
    return types.SimpleNamespace(**kw)


def response_with(usage_obj):
    return types.SimpleNamespace(usage=usage_obj)


class _RecordingGen:
    """Fake generation/span handle that records .update() calls."""

    def __init__(self, *, update_raises: bool = False):
        self.update_calls: list[dict] = []
        self._update_raises = update_raises

    def update(self, **kwargs):
        self.update_calls.append(kwargs)
        if self._update_raises:
            raise RuntimeError("update boom")


class _FakeClient:
    """Fake langfuse client. Captures every observation kwarg dict and the
    handles it hands out, and records flush()/auth_check() invocations."""

    def __init__(self, *, auth_result=True, auth_raises=False,
                 gen_update_raises=False):
        self.obs_kwargs: list[dict] = []
        self.handles: list[_RecordingGen] = []
        self.flush_calls = 0
        self.auth_calls = 0
        self._auth_result = auth_result
        self._auth_raises = auth_raises
        self._gen_update_raises = gen_update_raises

    def auth_check(self):
        self.auth_calls += 1
        if self._auth_raises:
            raise RuntimeError("auth boom")
        return self._auth_result

    def flush(self):
        self.flush_calls += 1

    def start_as_current_observation(self, **kwargs):
        self.obs_kwargs.append(kwargs)
        handle = _RecordingGen(update_raises=self._gen_update_raises)
        self.handles.append(handle)

        @contextlib.contextmanager
        def _cm():
            yield handle

        return _cm()


def install_fake_langfuse(client=None, *, record_propagate=None,
                          get_client_raises=False):
    """Inject a fake ``langfuse`` module exposing get_client and
    propagate_attributes. Returns the client so tests can assert captures.
    """
    client = client if client is not None else _FakeClient()
    mod = types.ModuleType("langfuse")

    def _get_client():
        if get_client_raises:
            raise RuntimeError("get_client boom")
        return client

    mod.get_client = _get_client

    @contextlib.contextmanager
    def _propagate_attributes(**kwargs):
        if record_propagate is not None:
            record_propagate.append(kwargs)
        yield

    mod.propagate_attributes = _propagate_attributes
    sys.modules["langfuse"] = mod
    return client


# ─── _xai_usage ──────────────────────────────────────────────────────

class TestXaiUsage:
    def test_none_when_no_usage_attr(self):
        # object() has no .usage -> getattr default None -> returns None.
        assert o._xai_usage(object()) is None

    def test_none_when_usage_is_none(self):
        assert o._xai_usage(response_with(None)) is None

    def test_missing_token_attrs_default_to_zero(self):
        # usage present but no token attrs at all.
        r = response_with(usage())
        assert o._xai_usage(r) == {"input": 0, "output": 0, "total": 0}

    def test_none_token_values_coalesce_to_zero(self):
        r = response_with(usage(prompt_tokens=None, completion_tokens=None,
                                total_tokens=None))
        out = o._xai_usage(r)
        assert out == {"input": 0, "output": 0, "total": 0}
        # Specifically 0 (int), not None.
        assert out["input"] == 0 and out["input"] is not None

    def test_cache_key_absent_when_zero(self):
        r = response_with(usage(prompt_tokens=10, completion_tokens=5,
                                total_tokens=15, cached_prompt_text_tokens=0))
        out = o._xai_usage(r)
        assert "cache_read_input_tokens" not in out

    def test_cache_key_present_when_positive(self):
        r = response_with(usage(prompt_tokens=10, completion_tokens=5,
                                total_tokens=15, cached_prompt_text_tokens=4))
        out = o._xai_usage(r)
        assert out["cache_read_input_tokens"] == 4

    def test_reasoning_key_absent_when_zero(self):
        r = response_with(usage(prompt_tokens=10, completion_tokens=5,
                                total_tokens=15, reasoning_tokens=0))
        assert "reasoning" not in o._xai_usage(r)

    def test_reasoning_key_present_when_positive(self):
        r = response_with(usage(prompt_tokens=10, completion_tokens=5,
                                total_tokens=15, reasoning_tokens=7))
        assert o._xai_usage(r)["reasoning"] == 7

    def test_both_zero_yields_exactly_three_keys(self):
        r = response_with(usage(prompt_tokens=10, completion_tokens=5,
                                total_tokens=15, cached_prompt_text_tokens=0,
                                reasoning_tokens=0))
        out = o._xai_usage(r)
        assert set(out) == {"input", "output", "total"}

    def test_fully_populated_exact_equality(self):
        r = response_with(usage(prompt_tokens=10, completion_tokens=5,
                                total_tokens=15, cached_prompt_text_tokens=3,
                                reasoning_tokens=2))
        assert o._xai_usage(r) == {
            "input": 10, "output": 5, "total": 15,
            "cache_read_input_tokens": 3, "reasoning": 2,
        }

    @pytest.mark.parametrize("cached,reasoning,expect_cache,expect_reason", [
        (0, 0, False, False),
        (1, 0, True, False),
        (0, 1, False, True),
        (5, 9, True, True),
    ])
    def test_conditional_key_boundary_sweep(self, cached, reasoning,
                                            expect_cache, expect_reason):
        r = response_with(usage(prompt_tokens=1, completion_tokens=1,
                                total_tokens=2,
                                cached_prompt_text_tokens=cached,
                                reasoning_tokens=reasoning))
        out = o._xai_usage(r)
        assert ("cache_read_input_tokens" in out) is expect_cache
        assert ("reasoning" in out) is expect_reason

    def test_none_cached_does_not_add_key(self):
        # cached_prompt_text_tokens present but None -> `or 0` -> 0 -> absent.
        r = response_with(usage(prompt_tokens=2, completion_tokens=1,
                                total_tokens=3, cached_prompt_text_tokens=None,
                                reasoning_tokens=None))
        out = o._xai_usage(r)
        assert out == {"input": 2, "output": 1, "total": 3}

    def test_conditional_keys_present_even_when_base_tokens_missing(self):
        # Base token attrs absent (default 0) but a positive cached value is
        # still surfaced — the conditional gate is independent of the base.
        r = response_with(usage(cached_prompt_text_tokens=4, reasoning_tokens=6))
        assert o._xai_usage(r) == {
            "input": 0, "output": 0, "total": 0,
            "cache_read_input_tokens": 4, "reasoning": 6,
        }

    def test_negative_cached_is_truthy_and_included(self):
        # The gate is `if cached:` (truthiness), so a negative int is truthy
        # and the key IS emitted with the raw (negative) value verbatim.
        r = response_with(usage(prompt_tokens=1, completion_tokens=1,
                                total_tokens=2, cached_prompt_text_tokens=-3,
                                reasoning_tokens=-1))
        out = o._xai_usage(r)
        assert out["cache_read_input_tokens"] == -3
        assert out["reasoning"] == -1


# ─── _openai_usage ───────────────────────────────────────────────────

class TestOpenaiUsage:
    def test_none_when_no_usage_attr(self):
        assert o._openai_usage(object()) is None

    def test_none_when_usage_is_none(self):
        assert o._openai_usage(response_with(None)) is None

    def test_missing_token_attrs_default_to_zero(self):
        assert o._openai_usage(response_with(usage())) == {
            "input": 0, "output": 0, "total": 0}

    def test_none_token_values_coalesce_to_zero(self):
        out = o._openai_usage(response_with(
            usage(prompt_tokens=None, completion_tokens=None,
                  total_tokens=None)))
        assert out == {"input": 0, "output": 0, "total": 0}

    def test_fully_populated_exact_equality(self):
        out = o._openai_usage(response_with(
            usage(prompt_tokens=7, completion_tokens=3, total_tokens=10)))
        assert out == {"input": 7, "output": 3, "total": 10}

    def test_never_emits_cache_or_reasoning_keys(self):
        # Even when the usage object carries xAI-style attrs, the OpenAI
        # translator ignores them and returns exactly the 3 base keys.
        out = o._openai_usage(response_with(
            usage(prompt_tokens=7, completion_tokens=3, total_tokens=10,
                  cached_prompt_text_tokens=99, reasoning_tokens=42)))
        assert set(out) == {"input", "output", "total"}
        assert "cache_read_input_tokens" not in out
        assert "reasoning" not in out


# ─── setup_observability ─────────────────────────────────────────────

class TestSetupObservability:
    def test_already_initialized_short_circuits(self, monkeypatch):
        # Sentinel-style: _INITIALIZED True -> returns cached _ENABLED
        # without reading env or importing langfuse.
        monkeypatch.setattr(o, "_INITIALIZED", True)
        monkeypatch.setattr(o, "_ENABLED", True)
        # Even with both keys missing, the short-circuit wins.
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        assert o.setup_observability() is True

    def test_short_circuit_returns_cached_false(self, monkeypatch):
        monkeypatch.setattr(o, "_INITIALIZED", True)
        monkeypatch.setattr(o, "_ENABLED", False)
        assert o.setup_observability() is False

    def test_only_public_key_disabled(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        assert o.setup_observability() is False
        assert o.is_enabled() is False
        # Side effect: it marked initialized.
        assert o._INITIALIZED is True

    def test_only_secret_key_disabled(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        assert o.setup_observability() is False
        assert o.is_enabled() is False

    def test_no_keys_disabled(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        assert o.setup_observability() is False

    def test_import_error_disabled(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        # Force `from langfuse import get_client` to raise ImportError by
        # planting a module without the attribute (lazy import inside fn).
        broken = types.ModuleType("langfuse")  # no get_client attr
        monkeypatch.setitem(sys.modules, "langfuse", broken)
        assert o.setup_observability() is False
        assert o.is_enabled() is False

    def test_auth_check_false_disabled(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        client = install_fake_langfuse(_FakeClient(auth_result=False))
        assert o.setup_observability() is False
        assert o.is_enabled() is False
        assert client.auth_calls == 1

    def test_auth_check_true_enabled(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        client = install_fake_langfuse(_FakeClient(auth_result=True))
        assert o.setup_observability() is True
        assert o.is_enabled() is True
        assert client.auth_calls == 1

    def test_arbitrary_exception_swallowed(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        # get_client() raises -> caught, returns False, not propagated.
        install_fake_langfuse(get_client_raises=True)
        assert o.setup_observability() is False
        assert o.is_enabled() is False

    def test_auth_check_raises_swallowed(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        install_fake_langfuse(_FakeClient(auth_raises=True))
        assert o.setup_observability() is False

    @pytest.mark.parametrize("envvar", ["LANGFUSE_BASE_URL", "LANGFUSE_HOST"])
    def test_host_fallback_does_not_change_return(self, monkeypatch, envvar):
        # The host only affects the log line; enabled return is unchanged.
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
        monkeypatch.delenv("LANGFUSE_HOST", raising=False)
        monkeypatch.setenv(envvar, "https://example.test")
        install_fake_langfuse(_FakeClient(auth_result=True))
        assert o.setup_observability() is True

    def test_host_log_line_uses_base_url(self, monkeypatch, caplog):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://from-base-url")
        monkeypatch.setenv("LANGFUSE_HOST", "https://from-host")
        install_fake_langfuse(_FakeClient(auth_result=True))
        with caplog.at_level("INFO", logger="dilution.observability"):
            assert o.setup_observability() is True
        assert "https://from-base-url" in caplog.text

    def test_idempotent_only_calls_auth_once(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        client = install_fake_langfuse(_FakeClient(auth_result=True))
        assert o.setup_observability() is True
        # Second call short-circuits; auth_check not re-invoked.
        assert o.setup_observability() is True
        assert client.auth_calls == 1

    def test_default_host_when_no_url_envs(self, monkeypatch, caplog):
        # Neither BASE_URL nor HOST set -> log line uses the hard-coded
        # cloud.langfuse.com default (return value unaffected).
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
        monkeypatch.delenv("LANGFUSE_HOST", raising=False)
        install_fake_langfuse(_FakeClient(auth_result=True))
        with caplog.at_level("INFO", logger="dilution.observability"):
            assert o.setup_observability() is True
        assert "https://cloud.langfuse.com" in caplog.text

    def test_host_falls_back_to_host_env_when_base_url_absent(
            self, monkeypatch, caplog):
        # Only LANGFUSE_HOST set -> it is used (second link of the chain).
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
        monkeypatch.setenv("LANGFUSE_HOST", "https://only-host")
        install_fake_langfuse(_FakeClient(auth_result=True))
        with caplog.at_level("INFO", logger="dilution.observability"):
            assert o.setup_observability() is True
        assert "https://only-host" in caplog.text
        assert "cloud.langfuse.com" not in caplog.text

    def test_enabled_path_sets_both_module_flags(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        install_fake_langfuse(_FakeClient(auth_result=True))
        assert o.setup_observability() is True
        assert o._INITIALIZED is True
        assert o._ENABLED is True

    @pytest.mark.parametrize("make_client,expect_warn", [
        (lambda: _FakeClient(auth_result=False), "auth_check failed"),
        (lambda: _FakeClient(auth_raises=True), "setup failed"),
    ])
    def test_failed_setup_marks_initialized_but_not_enabled(
            self, monkeypatch, caplog, make_client, expect_warn):
        # The idempotency contract: a failed setup still flips _INITIALIZED
        # so it is NOT retried, and leaves _ENABLED False. Also emits a
        # WARNING on the appropriate branch.
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        install_fake_langfuse(make_client())
        with caplog.at_level("WARNING", logger="dilution.observability"):
            assert o.setup_observability() is False
        assert o._INITIALIZED is True
        assert o._ENABLED is False
        assert expect_warn in caplog.text

    def test_missing_keys_logs_info_not_warning(self, monkeypatch, caplog):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        with caplog.at_level("INFO", logger="dilution.observability"):
            assert o.setup_observability() is False
        assert "tracing disabled" in caplog.text
        # No WARNING-level record on the clean "creds absent" path.
        assert not [r for r in caplog.records if r.levelname == "WARNING"]


# ─── flush_observability ─────────────────────────────────────────────

class TestFlushObservability:
    def test_disabled_does_not_import_langfuse(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", False)
        # Plant a fake whose get_client would raise if called; disabled
        # path must early-return without touching it.
        client = install_fake_langfuse(get_client_raises=True)
        # Should not raise (proves get_client never invoked).
        assert o.flush_observability() is None
        assert client.flush_calls == 0

    def test_enabled_calls_flush_once(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", True)
        client = install_fake_langfuse(_FakeClient())
        o.flush_observability()
        assert client.flush_calls == 1

    def test_flush_exception_swallowed(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", True)

        class BoomClient(_FakeClient):
            def flush(self):
                raise RuntimeError("flush boom")

        install_fake_langfuse(BoomClient())
        # No propagation.
        assert o.flush_observability() is None


# ─── pipeline_session ────────────────────────────────────────────────

class TestPipelineSession:
    def test_disabled_yields_none_no_import(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", False)
        client = install_fake_langfuse(get_client_raises=True)
        with o.pipeline_session("x") as s:
            assert s is None
        # Proves langfuse.get_client never touched.
        assert client.obs_kwargs == []

    def test_enabled_uppercases_ticker_in_input_and_session(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", True)
        propagate = []
        client = install_fake_langfuse(_FakeClient(),
                                       record_propagate=propagate)
        with o.pipeline_session("fcel") as span:
            assert span is client.handles[0]
        obs = client.obs_kwargs[0]
        assert obs["input"] == {"ticker": "FCEL"}
        assert obs["as_type"] == "span"
        assert obs["name"] == "dilution-pipeline"
        assert propagate == [{"session_id": "FCEL"}]

    def test_metadata_none_coerced_to_empty_dict(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", True)
        client = install_fake_langfuse(_FakeClient(), record_propagate=[])
        with o.pipeline_session("abc"):
            pass
        assert client.obs_kwargs[0]["metadata"] == {}

    def test_metadata_passed_through(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", True)
        client = install_fake_langfuse(_FakeClient(), record_propagate=[])
        md = {"k": "v"}
        with o.pipeline_session("abc", name="custom", metadata=md):
            pass
        assert client.obs_kwargs[0]["metadata"] == md
        assert client.obs_kwargs[0]["name"] == "custom"


# ─── stage ───────────────────────────────────────────────────────────

class TestStage:
    def test_disabled_yields_none_no_import(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", False)
        client = install_fake_langfuse(get_client_raises=True)
        with o.stage("walk") as s:
            assert s is None
        assert client.obs_kwargs == []

    def test_enabled_forwards_name_and_input(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", True)
        client = install_fake_langfuse(_FakeClient())
        payload = {"docs": 3}
        with o.stage("fetch", input=payload) as span:
            assert span is client.handles[0]
        obs = client.obs_kwargs[0]
        assert obs["name"] == "fetch"
        assert obs["input"] == payload
        assert obs["as_type"] == "span"

    def test_metadata_none_coerced_to_empty_dict(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", True)
        client = install_fake_langfuse(_FakeClient())
        with o.stage("index"):
            pass
        assert client.obs_kwargs[0]["metadata"] == {}

    def test_input_defaults_to_none(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", True)
        client = install_fake_langfuse(_FakeClient())
        with o.stage("index"):
            pass
        assert client.obs_kwargs[0]["input"] is None

    def test_metadata_passed_through(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", True)
        client = install_fake_langfuse(_FakeClient())
        md = {"stage_meta": 1}
        with o.stage("walk", metadata=md):
            pass
        assert client.obs_kwargs[0]["metadata"] == md


# ─── llm_generation ──────────────────────────────────────────────────

class TestLlmGeneration:
    def test_disabled_yields_none_no_import(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", False)
        client = install_fake_langfuse(get_client_raises=True)
        with o.llm_generation(name="n", model="m",
                              messages=[("system", "a")]) as g:
            assert g is None
        assert client.obs_kwargs == []

    def test_tuple_messages_normalized_to_dicts(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", True)
        client = install_fake_langfuse(_FakeClient())
        with o.llm_generation(name="walk", model="gpt",
                              messages=[("system", "a"), ("user", "b")]):
            pass
        obs = client.obs_kwargs[0]
        assert obs["input"] == [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        assert obs["as_type"] == "generation"
        assert obs["name"] == "walk"
        assert obs["model"] == "gpt"

    def test_dict_messages_passed_through_unchanged(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", True)
        client = install_fake_langfuse(_FakeClient())
        d1 = {"role": "system", "content": "x"}
        d2 = {"role": "user", "content": "y", "extra": 1}
        with o.llm_generation(name="n", model="m", messages=[d1, d2]):
            pass
        captured = client.obs_kwargs[0]["input"]
        # Identity preserved — dicts are not copied.
        assert captured[0] is d1
        assert captured[1] is d2

    def test_mixed_tuple_and_dict(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", True)
        client = install_fake_langfuse(_FakeClient())
        d = {"role": "assistant", "content": "z"}
        with o.llm_generation(name="n", model="m",
                              messages=[("system", "a"), d]):
            pass
        captured = client.obs_kwargs[0]["input"]
        assert captured[0] == {"role": "system", "content": "a"}
        assert captured[1] is d

    def test_empty_messages_normalize_to_empty_list(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", True)
        client = install_fake_langfuse(_FakeClient())
        with o.llm_generation(name="n", model="m", messages=[]):
            pass
        assert client.obs_kwargs[0]["input"] == []

    def test_none_model_forwarded(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", True)
        client = install_fake_langfuse(_FakeClient())
        with o.llm_generation(name="n", model=None, messages=[]):
            pass
        assert client.obs_kwargs[0]["model"] is None

    def test_exception_marks_error_and_reraises(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", True)
        client = install_fake_langfuse(_FakeClient())
        with pytest.raises(ValueError, match="boom"):
            with o.llm_generation(name="n", model="m", messages=[]):
                raise ValueError("boom")
        handle = client.handles[0]
        assert handle.update_calls == [
            {"level": "ERROR", "status_message": "boom"}
        ]

    def test_update_failure_swallowed_original_reraised(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", True)
        # gen.update() itself raises; inner try/except swallows it but the
        # ORIGINAL exception must still propagate.
        client = install_fake_langfuse(_FakeClient(gen_update_raises=True))
        with pytest.raises(KeyError, match="orig"):
            with o.llm_generation(name="n", model="m", messages=[]):
                raise KeyError("orig")
        # update was attempted exactly once before raising internally.
        assert len(client.handles[0].update_calls) == 1

    def test_no_exception_does_not_call_update(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", True)
        client = install_fake_langfuse(_FakeClient())
        with o.llm_generation(name="n", model="m", messages=[]):
            pass
        assert client.handles[0].update_calls == []

    def test_generation_observation_carries_no_metadata_kwarg(self, monkeypatch):
        # Unlike pipeline_session/stage (span helpers), the generation
        # observation is opened with as_type/name/model/input only — no
        # metadata kwarg is forwarded.
        monkeypatch.setattr(o, "_ENABLED", True)
        client = install_fake_langfuse(_FakeClient())
        with o.llm_generation(name="n", model="m", messages=[]):
            pass
        obs = client.obs_kwargs[0]
        assert set(obs) == {"as_type", "name", "model", "input"}
        assert "metadata" not in obs

    def test_status_message_uses_str_of_exception(self, monkeypatch):
        # A non-string exception payload is stringified via str(exc), not
        # repr or the raw arg — assert the exact rendered message.
        monkeypatch.setattr(o, "_ENABLED", True)
        client = install_fake_langfuse(_FakeClient())
        with pytest.raises(ValueError):
            with o.llm_generation(name="n", model="m", messages=[]):
                raise ValueError(404, "not found")
        assert client.handles[0].update_calls == [
            {"level": "ERROR", "status_message": "(404, 'not found')"},
        ]

    def test_exception_in_body_opens_exactly_one_observation(self, monkeypatch):
        # The observation must be opened (and torn down) exactly once even
        # on the error path — no duplicate/leaked observation.
        monkeypatch.setattr(o, "_ENABLED", True)
        client = install_fake_langfuse(_FakeClient())
        with pytest.raises(RuntimeError):
            with o.llm_generation(name="n", model="m",
                                  messages=[("user", "hi")]):
                raise RuntimeError("x")
        assert len(client.obs_kwargs) == 1
        assert len(client.handles) == 1


# ─── is_enabled ──────────────────────────────────────────────────────

class TestIsEnabled:
    def test_reflects_module_flag(self, monkeypatch):
        monkeypatch.setattr(o, "_ENABLED", False)
        assert o.is_enabled() is False
        monkeypatch.setattr(o, "_ENABLED", True)
        assert o.is_enabled() is True
