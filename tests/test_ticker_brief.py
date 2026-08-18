"""ensure_brief — the freshness gate and its fallbacks (no LLM, no
network: `generate` is always stubbed; the mutation-log rule runs
against the real temp DB)."""
import pytest

from db import get_conn
from dilution import ticker_brief as tb
from dilution.ledger.store import ensure_mutation_log_conn


def _cache_row(cik=1, summary="cached prose",
               generated_at="2026-08-01T00:00:00Z"):
    with get_conn() as conn:
        tb._ensure_table(conn)
        conn.execute(
            "INSERT INTO dilution_ticker_brief "
            "(cik, facts_hash, summary, generated_at, model) "
            "VALUES (?, 'fh', ?, ?, 'test-model')",
            (cik, summary, generated_at))


def _mutation(cik=1, applied_at="2026-08-02T00:00:00Z"):
    with get_conn() as conn:
        ensure_mutation_log_conn(conn)
        conn.execute(
            "INSERT INTO dilution_mutations "
            "(cik, accession_number, seq, kind, mutation_json, applied_at) "
            "VALUES (?, 'acc-1', 0, 'create', '{}', ?)",
            (cik, applied_at))


# ensure_brief's keyword bundle — the tests stub `generate`, so the
# facts built from these empty objects never reach an LLM.
_ARGS = dict(name="Test Co", fund=None, latest_os=None, cards={},
             cash=None, raised=None, badges=None)


class TestIsFresh:
    def test_nothing_cached_is_not_fresh(self):
        assert tb.is_fresh(1) is False

    def test_cached_with_no_later_mutation_is_fresh(self):
        _cache_row()
        assert tb.is_fresh(1) is True

    def test_mutation_after_generation_ages_the_brief(self):
        _cache_row(generated_at="2026-08-01T00:00:00Z")
        _mutation(applied_at="2026-08-02T00:00:00Z")
        assert tb.is_fresh(1) is False

    def test_mutation_before_generation_does_not(self):
        _cache_row(generated_at="2026-08-03T00:00:00Z")
        _mutation(applied_at="2026-08-02T00:00:00Z")
        assert tb.is_fresh(1) is True

    def test_another_tickers_mutation_does_not_age_it(self):
        _cache_row(cik=1)
        _mutation(cik=2, applied_at="2030-01-01T00:00:00Z")
        assert tb.is_fresh(1) is True


class TestEnsureBrief:
    def test_fresh_cache_short_circuits_the_llm(self, monkeypatch):
        _cache_row(summary="cached prose")

        def _boom(*a, **k):
            raise AssertionError("generate must not be called")
        monkeypatch.setattr(tb, "generate", _boom)
        brief = tb.ensure_brief(1, "TEST", **_ARGS)
        assert brief["summary"] == "cached prose"

    def test_stale_cache_regenerates(self, monkeypatch):
        _cache_row(summary="old prose")
        _mutation()
        monkeypatch.setattr(
            tb, "generate",
            lambda cik, ticker, facts: {"summary": "new prose"})
        brief = tb.ensure_brief(1, "TEST", **_ARGS)
        assert brief["summary"] == "new prose"

    def test_missing_cache_generates(self, monkeypatch):
        monkeypatch.setattr(
            tb, "generate",
            lambda cik, ticker, facts: {"summary": "new prose"})
        brief = tb.ensure_brief(1, "TEST", **_ARGS)
        assert brief["summary"] == "new prose"

    def test_generation_failure_falls_back_to_cached(self, monkeypatch):
        _cache_row(summary="old prose")
        _mutation()

        def _boom(*a, **k):
            raise RuntimeError("LLM down")
        monkeypatch.setattr(tb, "generate", _boom)
        brief = tb.ensure_brief(1, "TEST", **_ARGS)
        assert brief["summary"] == "old prose"

    def test_never_briefed_and_generation_failing_returns_none(
            self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("LLM down")
        monkeypatch.setattr(tb, "generate", _boom)
        assert tb.ensure_brief(1, "TEST", **_ARGS) is None
