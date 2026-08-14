"""Publisher for the Finviz ingest API — transport, safety gate, and
change detection for the payload `dilution/finviz_payload.py` builds.

Wire contract: FINVIZ_API_CONTRACT.md §2 (transport), §3.1 (publish),
§3.3 (read-back), §10 (ordering + retries).

This module deliberately does NOT share `finviz_client.py`'s fail-soft
posture. There, a failed fetch returns None because market data is
optional and the pipeline runs without it. Here a silent failure means
either unpublished data or — worse — wiped data, so nothing is swallowed:
configuration problems raise, and per-ticker transport failures come back
as an explicit `PushResult(status="failed")` the caller must handle.

The two rules that shape everything below:

  1. A bad POST is DESTRUCTIVE, not rejected. Per §3.5 the server does
     not validate `data`, and every accepted POST replaces the whole
     stored document: posting `{"schema_version": 1}` returns 200 and
     wipes the live snapshot. So `validate_snapshot` runs first and a
     document that fails it is never sent.
  2. `generated_at` is re-stamped on every build, so "has anything
     changed?" cannot be a whole-document comparison — see
     `content_digest`.

Usage:

    from dilution.finviz_push import push_snapshot
    from dilution.finviz_payload import build_payload

    result = push_snapshot(build_payload("CELU"))
    result.status   # pushed | skipped_unchanged | skipped_invalid | failed
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import requests

import config
from dilution.finviz_payload import SCHEMA_VERSION

log = logging.getLogger(__name__)

PUBLISH_PATH = "/api/dilution/set"
FETCH_PATH = "/api/dilution"

# Observed single-push latency is ~200 ms (§10) and bodies are tens of KB,
# so this is a generous ceiling for a hung connection rather than a
# realistic duration.
DEFAULT_TIMEOUT = 30  # seconds

# §2: mandatory. Without it the server attempts *form* parsing and fails
# with a 400 about "Form key length limit 2048 exceeded" — which reads
# like a body-size problem and is not one.
JSON_CONTENT_TYPE = "application/json; charset=utf-8"

# §10 retry envelope: exponential backoff with jitter, 1 s base, ×2,
# capped per-sleep, giving up after roughly 15 minutes.
RETRY_BASE_SECONDS = 1.0
RETRY_CAP_SECONDS = 60.0
RETRY_CEILING_SECONDS = 900.0

# A real snapshot is 20-30 KB compact (§3.1). Anything this small is a
# truncated or stub build, not a sparse issuer.
MIN_BODY_BYTES = 2048

# Fields re-stamped on every build, excluded from the change digest.
# Top-level ONLY — `brief.generated_at` is deliberately kept, see
# content_digest.
_VOLATILE_TOP_LEVEL = ("generated_at",)

# Envelope keys §4 requires inside `data`; also what we ask Finviz to
# start enforcing server-side (§3.5 ask #1).
_REQUIRED_SNAPSHOT_KEYS = (
    "schema_version", "ticker", "cik", "as_of", "generated_at",
)


class FinvizPushError(RuntimeError):
    """Configuration or programmer error that must stop the run.

    Transport failures do NOT raise this — they come back as
    PushResult(status="failed") so a batch over many tickers isn't
    aborted by one bad ticker.
    """


@dataclass
class PushResult:
    """Outcome of one publish attempt.

    status:
      pushed             POST returned 200; the snapshot is live.
      skipped_unchanged  Finviz already holds this exact content.
      skipped_invalid    Local validation refused it; nothing was sent.
      failed             Transport/HTTP failure; nothing is known to be live.
    """
    ticker: str
    status: str
    reason: str = ""
    digest: str | None = None
    body_bytes: int | None = None
    http_status: int | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the ticker ended in a good state — either we
        published it or Finviz already had it. Both mean the live
        document is current; only `failed` / `skipped_invalid` don't."""
        return self.status in ("pushed", "skipped_unchanged")


# ── change detection ─────────────────────────────────────────────────

def content_digest(snapshot: dict) -> str:
    """Stable SHA-256 over a snapshot's *content*, ignoring the build stamp.

    Three choices here are load-bearing:

    * Only the TOP-LEVEL `generated_at` is stripped. The nested
      `brief.generated_at` stays in the digest on purpose: it moves only
      when the brief is regenerated, and new prose is a real content
      change Finviz should receive.
    * `as_of` stays in the digest, so the first push of a new trading day
      always goes out. `as_of` is what the consumer displays and what
      §10's staleness rule flags a ticker on — skipping a push because
      "only the market data moved" would let a current ticker age into
      being hidden while we sit on fresher numbers.
    * `sort_keys` + tight separators make this independent of dict
      insertion order, so a reshuffled builder doesn't read as a change.

    Takes the inner snapshot (the `data` block), NOT the envelope — which
    is also exactly what `fetch_snapshot` returns, so the two sides of the
    comparison are symmetric.
    """
    body = {k: v for k, v in snapshot.items() if k not in _VOLATILE_TOP_LEVEL}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── local validation (the §3.5 gate) ─────────────────────────────────

def _parse_iso_date(value) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_iso_datetime(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def validate_snapshot(doc: dict, *, allow_empty: bool = False) -> list[str]:
    """Check a publish envelope, returning a list of reasons it must not
    be sent (empty list == publishable).

    This is the only thing standing between a bad build and destroyed
    live data (§3.5), so it errs toward refusing. Pure and offline —
    no network, no DB.
    """
    errors: list[str] = []

    if not isinstance(doc, dict):
        return [f"envelope is {type(doc).__name__}, expected dict"]

    ticker = doc.get("ticker")
    if not isinstance(ticker, str) or not ticker:
        errors.append("envelope `ticker` is missing or not a non-empty string")
    elif ticker != ticker.upper():
        errors.append(f"envelope `ticker` is not upper-case: {ticker!r}")

    snapshot = doc.get("data")
    if not isinstance(snapshot, dict):
        # §3.5's exact failure: a non-object `data` is accepted by the
        # server with a 200 and replaces the stored document.
        errors.append(
            f"`data` is {type(snapshot).__name__}, expected dict")
        return errors

    for key in _REQUIRED_SNAPSHOT_KEYS:
        if snapshot.get(key) in (None, ""):
            errors.append(f"`data.{key}` is missing")

    if isinstance(ticker, str) and snapshot.get("ticker") != ticker:
        errors.append(
            f"envelope ticker {ticker!r} != data.ticker "
            f"{snapshot.get('ticker')!r}")

    version = snapshot.get("schema_version")
    if version is not None and version != SCHEMA_VERSION:
        # A mismatch means a half-migrated build, not a new contract:
        # the producer bumps SCHEMA_VERSION and the payload together.
        errors.append(
            f"schema_version {version!r} != producer's {SCHEMA_VERSION}")

    as_of = _parse_iso_date(snapshot.get("as_of"))
    if snapshot.get("as_of") is not None and as_of is None:
        errors.append(f"`as_of` is not an ISO date: {snapshot.get('as_of')!r}")
    elif as_of is not None:
        # One day of slack: `as_of` is a US session date while this may run
        # on a box whose local date is behind ET.
        if as_of > date.today() + timedelta(days=1):
            errors.append(f"`as_of` {as_of.isoformat()} is in the future")

    if (snapshot.get("generated_at") is not None
            and _parse_iso_datetime(snapshot.get("generated_at")) is None):
        errors.append("`generated_at` is not an ISO datetime: "
                      f"{snapshot.get('generated_at')!r}")

    cards = snapshot.get("cards")
    if not isinstance(cards, dict):
        errors.append(f"`cards` is {type(cards).__name__}, expected dict")
        cards = {}
    else:
        for key, value in cards.items():
            if not isinstance(value, list):
                errors.append(
                    f"`cards.{key}` is {type(value).__name__}, expected list")

    body_bytes = len(json.dumps(doc, ensure_ascii=False).encode("utf-8"))
    if body_bytes < MIN_BODY_BYTES:
        errors.append(f"body is {body_bytes} bytes, under the "
                      f"{MIN_BODY_BYTES}-byte floor — truncated build?")

    # Emptiness tripwire. A ticker with no paper at all is legitimate but
    # rare; a build with no cards AND no badges AND no cash is what a
    # mid-write DB or a blanket fetcher failure produces, and it is
    # precisely the document that would silently wipe a good snapshot.
    if not allow_empty:
        company = snapshot.get("company") or {}
        if (not any(cards.get(k) for k in cards)
                and not snapshot.get("badges")
                and not company.get("cash")):
            errors.append(
                "no cards, no badges and no cash — looks like a failed "
                "build, not a no-paper issuer (pass allow_empty to publish)")

    return errors


# ── transport ────────────────────────────────────────────────────────

def _token() -> str:
    token = config.FINVIZ_INGEST_TOKEN
    if not token:
        raise FinvizPushError(
            "FINVIZ_INGEST_TOKEN is not set — refusing to contact the "
            "ingest API. This is the write credential and is deliberately "
            "separate from FINVIZ_API_KEY (the Elite /export read key).")
    return token


def _redacted(url: str) -> str:
    """URL safe to log — the auth token lives in the query string."""
    return url.split("?", 1)[0]


def _retry_after_seconds(resp: requests.Response) -> float | None:
    """§10: honor Retry-After on 429. Integer-seconds form only; the
    HTTP-date form falls through to normal backoff."""
    raw = (resp.headers.get("Retry-After") or "").strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with full jitter. Separate function so tests
    can neutralize the sleep without touching retry logic."""
    ceiling = min(RETRY_CAP_SECONDS, RETRY_BASE_SECONDS * (2 ** attempt))
    return random.uniform(ceiling / 2, ceiling)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _trace_id(resp: requests.Response) -> str:
    """ASP.NET problem-details carry a traceId; §2 asks us to quote it
    when reporting an ingest failure to Finviz infra."""
    try:
        payload = resp.json()
    except ValueError:
        return ""
    if isinstance(payload, dict):
        return str(payload.get("traceId") or "")
    return ""


def fetch_snapshot(ticker: str) -> dict | None:
    """Read back what Finviz currently holds for `ticker` (§3.3).

    Returns the stored snapshot as the INNER `data` object, unwrapped —
    not the §4 envelope — so it compares directly against
    `build_payload(...)["data"]`. None when the ticker isn't published
    (404). Raises on transport failure so callers can distinguish "not
    published" from "couldn't tell".
    """
    ticker = ticker.upper()
    url = f"{config.FINVIZ_BASE_URL.rstrip('/')}{FETCH_PATH}/{ticker}"
    resp = requests.get(url, params={"auth": _token()},
                        timeout=DEFAULT_TIMEOUT)
    if resp.status_code == 404:
        log.info("finviz read-back %s: not published", ticker)
        return None
    if resp.status_code == 401:
        raise FinvizPushError(
            f"401 from {_redacted(url)} — FINVIZ_INGEST_TOKEN rejected")
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError as exc:
        raise FinvizPushError(
            f"read-back for {ticker} was not JSON: {exc}") from exc


def push_snapshot(doc: dict, *, if_changed: bool = True,
                  dry_run: bool = False,
                  allow_empty: bool = False) -> PushResult:
    """Publish one ticker's envelope to `POST /api/dilution/set` (§3.1).

    Steps, in this order for a reason:

      1. Validate locally. An invalid document is never sent — under
         §3.5 sending it would replace good live data with junk. Doing
         this first also means a bad build costs zero requests.
      2. If `if_changed`, read back what Finviz holds and compare
         content digests; identical content is not re-sent. A 404 (never
         published) publishes. A read-back *error* also publishes —
         failing open, because an unreachable GET must not silently stop
         publishing, and a redundant POST is idempotent.
      3. POST, retrying 5xx and network errors per §10.

    Never raises on an HTTP failure; returns status="failed" so a batch
    can carry on. Raises FinvizPushError only for misconfiguration.
    """
    ticker = str(doc.get("ticker") or (doc.get("data") or {}).get("ticker")
                 or "?").upper()

    errors = validate_snapshot(doc, allow_empty=allow_empty)
    if errors:
        log.error("push %s refused by local validation: %s",
                  ticker, "; ".join(errors))
        return PushResult(ticker=ticker, status="skipped_invalid",
                          reason=errors[0], errors=errors)

    snapshot = doc["data"]
    digest = content_digest(snapshot)
    body = json.dumps(doc, ensure_ascii=False).encode("utf-8")

    if if_changed:
        try:
            live = fetch_snapshot(ticker)
        except FinvizPushError:
            raise
        except Exception as exc:
            # Fail open: publish rather than risk never publishing.
            log.warning("push %s: read-back failed (%s) — publishing anyway",
                        ticker, exc)
        else:
            if live is not None and content_digest(live) == digest:
                log.info("push %s: unchanged (%s), not re-sending",
                         ticker, digest[:12])
                return PushResult(ticker=ticker, status="skipped_unchanged",
                                  reason=f"digest {digest[:12]} already live",
                                  digest=digest, body_bytes=len(body))

    if dry_run:
        log.info("push %s: DRY RUN — would send %d bytes (digest %s)",
                 ticker, len(body), digest[:12])
        return PushResult(ticker=ticker, status="pushed",
                          reason=f"dry run, {len(body)} bytes not sent",
                          digest=digest, body_bytes=len(body))

    url = f"{config.FINVIZ_BASE_URL.rstrip('/')}{PUBLISH_PATH}"
    # §2: the content type is mandatory, and gzip is rejected — so the
    # body goes out as plain UTF-8 JSON with no Content-Encoding.
    headers = {"Content-Type": JSON_CONTENT_TYPE}

    started = time.monotonic()
    attempt = 0
    last_reason = ""
    last_status: int | None = None

    while True:
        retry_after: float | None = None
        try:
            resp = requests.post(url, params={"auth": _token()},
                                 data=body, headers=headers,
                                 timeout=DEFAULT_TIMEOUT)
        except requests.RequestException as exc:
            last_reason, last_status = f"network error: {exc}", None
        else:
            last_status = resp.status_code
            if resp.status_code == 200:
                log.info("push %s: published %d bytes (digest %s)",
                         ticker, len(body), digest[:12])
                return PushResult(ticker=ticker, status="pushed",
                                  reason="200 OK", digest=digest,
                                  body_bytes=len(body), http_status=200)
            if resp.status_code == 400:
                # §3.1: envelope validation failed. Producer bug — a
                # retry sends the identical body and fails identically.
                trace = _trace_id(resp)
                reason = ("400 from ingest (producer bug, not retried)"
                          + (f" traceId={trace}" if trace else ""))
                log.error("push %s: %s — body %s", ticker, reason,
                          resp.text[:400])
                return PushResult(ticker=ticker, status="failed",
                                  reason=reason, digest=digest,
                                  body_bytes=len(body), http_status=400)
            if resp.status_code == 401:
                raise FinvizPushError(
                    f"401 from {_redacted(url)} — FINVIZ_INGEST_TOKEN "
                    f"rejected")
            retry_after = (_retry_after_seconds(resp)
                           if resp.status_code == 429 else None)
            last_reason = f"HTTP {resp.status_code}"

        delay = retry_after if retry_after is not None else _backoff_seconds(attempt)
        elapsed = time.monotonic() - started
        if elapsed + delay > RETRY_CEILING_SECONDS:
            log.error("push %s: giving up after %.0fs — %s",
                      ticker, elapsed, last_reason)
            return PushResult(ticker=ticker, status="failed",
                              reason=f"{last_reason} (retries exhausted)",
                              digest=digest, body_bytes=len(body),
                              http_status=last_status)
        log.warning("push %s: %s — retrying in %.1fs (attempt %d)",
                    ticker, last_reason, delay, attempt + 1)
        _sleep(delay)
        attempt += 1
