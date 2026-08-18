"""Unit tests for dilution/finviz_push.py.

No DB and no network here: the I/O seams are ``requests.post`` /
``requests.get``, patched per test, and ``config.FINVIZ_INGEST_TOKEN``,
set by the autouse ``ingest_token`` fixture so no test can accidentally
depend on a real credential.

``_sleep`` is neutralized suite-wide — the retry tests assert on the
*sequence* of attempts and the delays handed to the sleeper, never by
actually waiting.

The stakes here are unusual and worth restating: per FINVIZ_API_CONTRACT.md
§3.5 the ingest server does not validate ``data`` and every accepted POST
is a full replace, so a document that should not have been sent destroys
live data rather than being rejected. ``validate_snapshot`` is the only
guard, hence the density of its tests below.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest
import requests

import config
from dilution import finviz_push as fp


# ── shared fixtures / helpers ────────────────────────────────────────


@pytest.fixture(autouse=True)
def ingest_token(monkeypatch):
    """Every test runs with a known fake token."""
    monkeypatch.setattr(config, "FINVIZ_INGEST_TOKEN", "test-token")
    monkeypatch.setattr(config, "FINVIZ_BASE_URL", "https://finviz.test")


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Record backoff delays instead of sleeping through them."""
    slept: list[float] = []
    monkeypatch.setattr(fp, "_sleep", slept.append)
    return slept


class _FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code=200, *, json_body=None, text="",
                 headers=None):
        self.status_code = status_code
        self._json = json_body
        self.text = text or (json.dumps(json_body) if json_body else "")
        self.headers = headers or {}

    def json(self):
        if self._json is None:
            raise ValueError("no json body")
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _snapshot(**overrides) -> dict:
    """A minimal snapshot that passes validation.

    ``_filler`` pads past the MIN_BODY_BYTES floor so a test about (say)
    a bad ``as_of`` fails for that reason alone and not incidentally on
    body size. It stands in for the ~20-30 KB a real snapshot carries.
    """
    snap = {
        "schema_version": fp.SCHEMA_VERSION,
        "ticker": "TEST",
        "cik": 1234567,
        "company_name": "Test Co",
        "as_of": "2026-06-01",
        "generated_at": "2026-06-02T01:02:03Z",
        "company": {"shares_outstanding": 1e6, "cash": {"months_of_cash": 3.0}},
        "badges": {"overall": {"score": 50, "label": "Moderate"}},
        "cards": {"warrant": [{"source_ref": "W-001", "title": "Warrants"}]},
        "brief": {"summary": "s",
                  "generated_at": "2026-06-02T00:00:00Z"},
        "_filler": "x" * 4096,
    }
    snap.update(overrides)
    return snap


def _envelope(snapshot=None, ticker="TEST") -> dict:
    snap = snapshot if snapshot is not None else _snapshot()
    return {"ticker": ticker, "data": snap}


def _celu_example() -> dict | None:
    """The committed real payload, when present — a regression anchor
    against the validator drifting away from the actual producer."""
    path = (Path(__file__).resolve().parent.parent
            / "examples" / "finviz_payload_CELU.json")
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ── content_digest ───────────────────────────────────────────────────


class TestContentDigest:
    def test_stable_across_calls(self):
        snap = _snapshot()
        assert fp.content_digest(snap) == fp.content_digest(snap)

    def test_top_level_generated_at_is_ignored(self):
        """The whole point: the build stamp must not read as a change,
        or nothing would ever be skipped."""
        a = _snapshot(generated_at="2026-06-02T01:02:03Z")
        b = _snapshot(generated_at="2030-12-31T23:59:59Z")
        assert fp.content_digest(a) == fp.content_digest(b)

    def test_nested_brief_generated_at_is_a_change(self):
        """Deliberately NOT stripped — a regenerated brief is new prose
        the consumer should receive."""
        a = _snapshot()
        b = _snapshot(brief={**_snapshot()["brief"],
                             "generated_at": "2030-01-01T00:00:00Z"})
        assert fp.content_digest(a) != fp.content_digest(b)

    def test_as_of_is_a_change(self):
        """A new trading day must publish: `as_of` is what the consumer
        displays and what §10's staleness rule flags on."""
        assert (fp.content_digest(_snapshot(as_of="2026-06-01"))
                != fp.content_digest(_snapshot(as_of="2026-06-02")))

    def test_key_order_does_not_matter(self):
        snap = _snapshot()
        reordered = {k: snap[k] for k in reversed(list(snap))}
        assert fp.content_digest(snap) == fp.content_digest(reordered)

    def test_card_field_change_is_a_change(self):
        a = _snapshot()
        b = _snapshot(cards={"warrant": [{"source_ref": "W-001",
                                          "title": "Warrants",
                                          "strike": 1.25}]})
        assert fp.content_digest(a) != fp.content_digest(b)

    def test_nested_value_change_is_a_change(self):
        a = _snapshot()
        b = _snapshot(company={"shares_outstanding": 2e6,
                               "cash": {"months_of_cash": 3.0}})
        assert fp.content_digest(a) != fp.content_digest(b)

    def test_real_payload_digest_is_reproducible(self):
        doc = _celu_example()
        if doc is None:
            pytest.skip("examples/finviz_payload_CELU.json not present")
        first = fp.content_digest(doc["data"])
        second = fp.content_digest(json.loads(json.dumps(doc))["data"])
        assert first == second


# ── validate_snapshot ────────────────────────────────────────────────


class TestValidateSnapshot:
    def test_good_envelope_passes(self):
        assert fp.validate_snapshot(_envelope()) == []

    def test_real_committed_payload_passes(self):
        """If this fails, the validator and the producer disagree — which
        would block every real push."""
        doc = _celu_example()
        if doc is None:
            pytest.skip("examples/finviz_payload_CELU.json not present")
        assert fp.validate_snapshot(doc) == []

    def test_non_dict_envelope(self):
        assert fp.validate_snapshot("nope")

    @pytest.mark.parametrize("ticker", [None, "", 123])
    def test_missing_envelope_ticker(self, ticker):
        errs = fp.validate_snapshot({"ticker": ticker, "data": _snapshot()})
        assert any("ticker" in e for e in errs)

    def test_lowercase_envelope_ticker(self):
        errs = fp.validate_snapshot({"ticker": "test", "data": _snapshot()})
        assert any("upper-case" in e for e in errs)

    def test_envelope_ticker_mismatch(self):
        errs = fp.validate_snapshot(
            {"ticker": "OTHER", "data": _snapshot(ticker="TEST")})
        assert any("!=" in e for e in errs)

    @pytest.mark.parametrize("data", ["x", 5, None, [1, 2]])
    def test_non_dict_data_is_refused(self, data):
        """§3.5's proven silent-wipe shape: `{"ticker": "X", "data": "x"}`
        returned 200 and replaced the stored document."""
        errs = fp.validate_snapshot({"ticker": "TEST", "data": data})
        assert errs and any("`data`" in e for e in errs)

    @pytest.mark.parametrize(
        "key", ["schema_version", "ticker", "cik", "as_of", "generated_at"])
    def test_required_keys(self, key):
        snap = _snapshot()
        del snap[key]
        errs = fp.validate_snapshot({"ticker": "TEST", "data": snap})
        assert any(key in e for e in errs)

    def test_schema_version_mismatch(self):
        errs = fp.validate_snapshot(
            _envelope(_snapshot(schema_version=fp.SCHEMA_VERSION + 1)))
        assert any("schema_version" in e for e in errs)

    def test_unparseable_as_of(self):
        errs = fp.validate_snapshot(_envelope(_snapshot(as_of="June 1st")))
        assert any("as_of" in e for e in errs)

    def test_future_as_of(self):
        future = (date.today() + timedelta(days=30)).isoformat()
        errs = fp.validate_snapshot(_envelope(_snapshot(as_of=future)))
        assert any("future" in e for e in errs)

    def test_today_as_of_is_fine(self):
        assert fp.validate_snapshot(
            _envelope(_snapshot(as_of=date.today().isoformat()))) == []

    def test_unparseable_generated_at(self):
        errs = fp.validate_snapshot(
            _envelope(_snapshot(generated_at="last tuesday")))
        assert any("generated_at" in e for e in errs)

    def test_cards_must_be_a_dict(self):
        errs = fp.validate_snapshot(_envelope(_snapshot(cards=[])))
        assert any("cards" in e for e in errs)

    def test_card_arrays_must_be_lists(self):
        errs = fp.validate_snapshot(
            _envelope(_snapshot(cards={"warrant": {"not": "a list"}})))
        assert any("cards.warrant" in e for e in errs)

    def test_truncated_body_is_refused(self):
        snap = _snapshot()
        del snap["_filler"]
        errs = fp.validate_snapshot({"ticker": "TEST", "data": snap})
        assert any("floor" in e for e in errs)

    def test_empty_build_tripwire(self):
        """No cards + no badges + no cash is what a mid-write DB or a
        blanket fetcher failure looks like."""
        errs = fp.validate_snapshot(_envelope(_snapshot(
            cards={}, badges=None, company={"shares_outstanding": None})))
        assert any("failed build" in e for e in errs)

    def test_empty_build_allowed_explicitly(self):
        assert fp.validate_snapshot(
            _envelope(_snapshot(cards={}, badges=None,
                                company={"shares_outstanding": None})),
            allow_empty=True) == []

    def test_empty_cards_but_real_badges_is_not_tripped(self):
        """A genuinely no-paper issuer still has badges/cash, so the
        tripwire must not fire on cards alone."""
        errs = fp.validate_snapshot(_envelope(_snapshot(cards={})))
        assert errs == []


# ── fetch_snapshot ───────────────────────────────────────────────────


class TestFetchSnapshot:
    def test_returns_unwrapped_data(self, monkeypatch):
        """§3.3 returns the inner `data`, NOT the envelope — the digest
        comparison depends on this asymmetry being handled here."""
        snap = _snapshot()
        monkeypatch.setattr(fp.requests, "get",
                            lambda *a, **k: _FakeResponse(200, json_body=snap))
        assert fp.fetch_snapshot("test") == snap

    def test_404_is_none(self, monkeypatch):
        monkeypatch.setattr(fp.requests, "get",
                            lambda *a, **k: _FakeResponse(404))
        assert fp.fetch_snapshot("TEST") is None

    def test_401_raises_config_error(self, monkeypatch):
        monkeypatch.setattr(fp.requests, "get",
                            lambda *a, **k: _FakeResponse(401))
        with pytest.raises(fp.FinvizPushError):
            fp.fetch_snapshot("TEST")

    def test_500_raises(self, monkeypatch):
        monkeypatch.setattr(fp.requests, "get",
                            lambda *a, **k: _FakeResponse(500))
        with pytest.raises(requests.HTTPError):
            fp.fetch_snapshot("TEST")

    def test_ticker_is_upper_cased_in_url(self, monkeypatch):
        seen = {}

        def fake_get(url, **kwargs):
            seen["url"] = url
            seen["params"] = kwargs.get("params")
            return _FakeResponse(404)

        monkeypatch.setattr(fp.requests, "get", fake_get)
        fp.fetch_snapshot("celu")
        assert seen["url"].endswith("/api/dilution/CELU")
        assert seen["params"] == {"auth": "test-token"}

    def test_missing_token_raises_before_request(self, monkeypatch):
        monkeypatch.setattr(config, "FINVIZ_INGEST_TOKEN", "")

        def boom(*a, **k):
            raise AssertionError("must not reach the network")

        monkeypatch.setattr(fp.requests, "get", boom)
        with pytest.raises(fp.FinvizPushError):
            fp.fetch_snapshot("TEST")


# ── push_snapshot ────────────────────────────────────────────────────


class _Recorder:
    """Collects POSTs and replays a scripted list of responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        resp = self.responses.pop(0) if self.responses else _FakeResponse(200)
        if isinstance(resp, Exception):
            raise resp
        return resp


@pytest.fixture
def no_get(monkeypatch):
    """Read-back returns 404 (never published) so pushes go through."""
    monkeypatch.setattr(fp.requests, "get",
                        lambda *a, **k: _FakeResponse(404))


class TestPushSnapshotValidation:
    def test_invalid_document_sends_nothing_at_all(self, monkeypatch):
        """Not even the read-back GET — validation runs first so a bad
        build costs zero requests."""
        def boom(*a, **k):
            raise AssertionError("must not touch the network")

        monkeypatch.setattr(fp.requests, "post", boom)
        monkeypatch.setattr(fp.requests, "get", boom)
        result = fp.push_snapshot({"ticker": "TEST", "data": "junk"})
        assert result.status == "skipped_invalid"
        assert result.errors

    def test_empty_build_refused_by_default(self, monkeypatch):
        monkeypatch.setattr(fp.requests, "post",
                            lambda *a, **k: pytest.fail("should not POST"))
        monkeypatch.setattr(fp.requests, "get",
                            lambda *a, **k: _FakeResponse(404))
        result = fp.push_snapshot(_envelope(_snapshot(
            cards={}, badges=None, company={})))
        assert result.status == "skipped_invalid"


class TestPushSnapshotChangeDetection:
    def test_unchanged_live_content_is_not_resent(self, monkeypatch):
        """The behavior the whole feature exists for: a fresh
        `generated_at` must not defeat the digest."""
        doc = _envelope()
        live = dict(doc["data"], generated_at="2020-01-01T00:00:00Z")
        monkeypatch.setattr(fp.requests, "get",
                            lambda *a, **k: _FakeResponse(200, json_body=live))
        recorder = _Recorder()
        monkeypatch.setattr(fp.requests, "post", recorder)

        result = fp.push_snapshot(doc)
        assert result.status == "skipped_unchanged"
        assert recorder.calls == []

    def test_changed_live_content_is_pushed(self, monkeypatch):
        doc = _envelope()
        live = dict(doc["data"], as_of="2020-01-01")
        monkeypatch.setattr(fp.requests, "get",
                            lambda *a, **k: _FakeResponse(200, json_body=live))
        recorder = _Recorder(_FakeResponse(
            200, json_body={"success": True, "ticker": "TEST"}))
        monkeypatch.setattr(fp.requests, "post", recorder)

        result = fp.push_snapshot(doc)
        assert result.status == "pushed"
        assert len(recorder.calls) == 1

    def test_never_published_is_pushed(self, monkeypatch, no_get):
        recorder = _Recorder(_FakeResponse(200))
        monkeypatch.setattr(fp.requests, "post", recorder)
        assert fp.push_snapshot(_envelope()).status == "pushed"
        assert len(recorder.calls) == 1

    def test_read_back_failure_fails_open(self, monkeypatch):
        """An unreachable GET must not silently stop publishing; a
        redundant POST is idempotent, an unpublished ticker is not."""
        def bad_get(*a, **k):
            raise requests.ConnectionError("read-back down")

        monkeypatch.setattr(fp.requests, "get", bad_get)
        recorder = _Recorder(_FakeResponse(200))
        monkeypatch.setattr(fp.requests, "post", recorder)

        result = fp.push_snapshot(_envelope())
        assert result.status == "pushed"
        assert len(recorder.calls) == 1

    def test_if_changed_false_skips_the_read_back(self, monkeypatch):
        """--force-push: publish without asking what's live."""
        def boom(*a, **k):
            raise AssertionError("should not read back")

        monkeypatch.setattr(fp.requests, "get", boom)
        recorder = _Recorder(_FakeResponse(200))
        monkeypatch.setattr(fp.requests, "post", recorder)

        result = fp.push_snapshot(_envelope(), if_changed=False)
        assert result.status == "pushed"
        assert len(recorder.calls) == 1

    def test_force_push_resends_identical_content(self, monkeypatch):
        doc = _envelope()
        monkeypatch.setattr(
            fp.requests, "get",
            lambda *a, **k: _FakeResponse(200, json_body=doc["data"]))
        recorder = _Recorder(_FakeResponse(200))
        monkeypatch.setattr(fp.requests, "post", recorder)

        assert fp.push_snapshot(doc, if_changed=False).status == "pushed"
        assert len(recorder.calls) == 1


class TestPushSnapshotTransport:
    def test_content_type_header_is_exact(self, monkeypatch, no_get):
        """§2: mandatory, and its absence yields a misleading 400 about
        form key length."""
        recorder = _Recorder(_FakeResponse(200))
        monkeypatch.setattr(fp.requests, "post", recorder)
        fp.push_snapshot(_envelope())
        assert (recorder.calls[0]["headers"]["Content-Type"]
                == "application/json; charset=utf-8")

    def test_body_is_not_compressed(self, monkeypatch, no_get):
        """§2: gzip is rejected with a 400 (the magic byte reaches the
        JSON parser)."""
        recorder = _Recorder(_FakeResponse(200))
        monkeypatch.setattr(fp.requests, "post", recorder)
        fp.push_snapshot(_envelope())
        call = recorder.calls[0]
        assert "Content-Encoding" not in call["headers"]
        assert json.loads(call["data"].decode("utf-8"))["ticker"] == "TEST"

    def test_auth_is_a_query_param(self, monkeypatch, no_get):
        """Not an Authorization header (§2)."""
        recorder = _Recorder(_FakeResponse(200))
        monkeypatch.setattr(fp.requests, "post", recorder)
        fp.push_snapshot(_envelope())
        call = recorder.calls[0]
        assert call["params"] == {"auth": "test-token"}
        assert "Authorization" not in call["headers"]
        assert call["url"].endswith("/api/dilution/set")

    def test_400_is_not_retried(self, monkeypatch, no_get):
        """A producer bug — the identical body would fail identically."""
        recorder = _Recorder(_FakeResponse(
            400, json_body={"error": "Data is required",
                            "traceId": "00-abc-01"}))
        monkeypatch.setattr(fp.requests, "post", recorder)
        result = fp.push_snapshot(_envelope())
        assert result.status == "failed"
        assert result.http_status == 400
        assert "00-abc-01" in result.reason
        assert len(recorder.calls) == 1

    def test_401_raises(self, monkeypatch, no_get):
        recorder = _Recorder(_FakeResponse(401))
        monkeypatch.setattr(fp.requests, "post", recorder)
        with pytest.raises(fp.FinvizPushError):
            fp.push_snapshot(_envelope())

    def test_500_retries_then_succeeds(self, monkeypatch, no_get,
                                       no_real_sleep):
        recorder = _Recorder(_FakeResponse(500), _FakeResponse(503),
                             _FakeResponse(200))
        monkeypatch.setattr(fp.requests, "post", recorder)
        result = fp.push_snapshot(_envelope())
        assert result.status == "pushed"
        assert len(recorder.calls) == 3
        assert len(no_real_sleep) == 2

    def test_network_error_is_retried(self, monkeypatch, no_get):
        recorder = _Recorder(requests.ConnectionError("boom"),
                             _FakeResponse(200))
        monkeypatch.setattr(fp.requests, "post", recorder)
        assert fp.push_snapshot(_envelope()).status == "pushed"
        assert len(recorder.calls) == 2

    def test_retries_exhaust_to_failed(self, monkeypatch, no_get,
                                       no_real_sleep):
        """Never raises on transport failure — a batch over 66 tickers
        must not abort on one bad ticker.

        The delay sequence must CROSS RETRY_CEILING_SECONDS, not sit under
        it. `_sleep` is neutralized suite-wide (autouse `no_real_sleep`),
        so nothing advances the loop's `elapsed`; a constant sub-ceiling
        delay leaves `elapsed + delay > RETRY_CEILING_SECONDS` permanently
        false and `while True` spins at CPU speed, emitting a log.warning
        per iteration that pytest's log capture retains — ~32 GB in two
        minutes, which the kernel OOM killer ends. Here the first delay
        fits the budget (so a real retry happens) and the second blows it,
        pinning the exhaust path at exactly two POSTs.
        """
        recorder = _Recorder(_FakeResponse(500), _FakeResponse(500))
        monkeypatch.setattr(fp.requests, "post", recorder)
        delays = iter([600.0, fp.RETRY_CEILING_SECONDS + 100.0])
        monkeypatch.setattr(fp, "_backoff_seconds",
                            lambda attempt: next(delays))
        result = fp.push_snapshot(_envelope())
        assert result.status == "failed"
        assert "exhausted" in result.reason
        assert result.http_status == 500
        assert len(recorder.calls) == 2      # one retry, then gave up
        assert no_real_sleep == [600.0]      # only the in-budget delay slept

    def test_429_honors_retry_after(self, monkeypatch, no_get,
                                    no_real_sleep):
        recorder = _Recorder(
            _FakeResponse(429, headers={"Retry-After": "7"}),
            _FakeResponse(200))
        monkeypatch.setattr(fp.requests, "post", recorder)
        result = fp.push_snapshot(_envelope())
        assert result.status == "pushed"
        assert no_real_sleep == [7.0]

    def test_429_without_retry_after_uses_backoff(self, monkeypatch, no_get,
                                                  no_real_sleep):
        recorder = _Recorder(_FakeResponse(429), _FakeResponse(200))
        monkeypatch.setattr(fp.requests, "post", recorder)
        monkeypatch.setattr(fp, "_backoff_seconds", lambda attempt: 1.5)
        fp.push_snapshot(_envelope())
        assert no_real_sleep == [1.5]

    def test_429_with_http_date_falls_back_to_backoff(self, monkeypatch,
                                                      no_get, no_real_sleep):
        recorder = _Recorder(
            _FakeResponse(429, headers={
                "Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
            _FakeResponse(200))
        monkeypatch.setattr(fp.requests, "post", recorder)
        monkeypatch.setattr(fp, "_backoff_seconds", lambda attempt: 2.5)
        fp.push_snapshot(_envelope())
        assert no_real_sleep == [2.5]

    def test_dry_run_issues_no_post(self, monkeypatch, no_get):
        def boom(*a, **k):
            raise AssertionError("dry run must not POST")

        monkeypatch.setattr(fp.requests, "post", boom)
        result = fp.push_snapshot(_envelope(), dry_run=True)
        assert result.status == "pushed"
        assert "dry run" in result.reason
        assert result.body_bytes and result.body_bytes > 0

    def test_missing_token_raises_before_any_request(self, monkeypatch):
        monkeypatch.setattr(config, "FINVIZ_INGEST_TOKEN", "")

        def boom(*a, **k):
            raise AssertionError("must not touch the network")

        monkeypatch.setattr(fp.requests, "post", boom)
        monkeypatch.setattr(fp.requests, "get", boom)
        with pytest.raises(fp.FinvizPushError):
            fp.push_snapshot(_envelope())

    def test_token_is_never_logged(self, monkeypatch, no_get, caplog):
        recorder = _Recorder(_FakeResponse(500), _FakeResponse(200))
        monkeypatch.setattr(fp.requests, "post", recorder)
        with caplog.at_level("DEBUG"):
            fp.push_snapshot(_envelope())
        assert "test-token" not in caplog.text

    def test_redacted_strips_query_string(self):
        assert (fp._redacted("https://x.test/api/dilution/set?auth=secret")
                == "https://x.test/api/dilution/set")


class TestPushResult:
    @pytest.mark.parametrize("status,expected", [
        ("pushed", True), ("skipped_unchanged", True),
        ("skipped_invalid", False), ("failed", False),
    ])
    def test_ok_property(self, status, expected):
        assert fp.PushResult(ticker="T", status=status).ok is expected
