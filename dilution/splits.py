"""Stock-split lookup — vendor-sourced, persisted, walker-fed.

The walker historically extracted `apply_split` mutations directly from
SEC filings. That works when the 8-K body states the ratio in plain
text, but breaks on the common pattern where the Certificate of
Amendment is filed as a scanned-image exhibit (EX-3.x .jpg). XTIA's
2024-03-13 1-for-100 split is the canonical failure case: the body
says "Reverse Stock Split" without a ratio, EX-3.2 is image-only, and
the post-merger 10-K's plain-text mention used to be ignored by the
periodic-filing hint.

This module sources splits from market-data vendors (where they're
authoritative — exchanges record splits via CUSIP changes) and feeds
them into the walker as synthetic mutations. Walker-extracted splits
remain a defensive backstop via `_split_already_applied` dedup.

Two vendors, both treated as primary:
  - Finviz: GET /export/split-history?auth=<KEY> — JSON array of
    {ticker, exdate, factorFrom, factorTo} entries, full universe.
    factorFrom:factorTo is the post:pre ratio (4-for-1 forward
    = factorFrom=4 factorTo=1; 1-for-100 reverse = 1:100).
  - yfinance: yf.Ticker(symbol).splits — pandas Series of
    {effective_date: post/pre ratio} as floats.

Merge rules:
  - Bucket events from both sources by (date ± 7 days) — vendors can
    disagree by a day or two on whether the date is announcement vs
    ex-date.
  - Both sources agree on (post, pre) → emit one row (source =
    "finviz+yfinance").
  - Both sources present but ratios disagree → yfinance wins, log
    warning. yfinance has consistently lined up with walker-extracted
    8-K text in the cases we've audited (e.g. IQST 2025-12-15 where
    Finviz reports 173:10000 reverse, yfinance + 8-K both say 1017:1000
    forward).
  - Only one source has it → take it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from fractions import Fraction
from typing import Literal

import requests

import config
from db import get_conn, now_iso

log = logging.getLogger(__name__)

# Vendors can drift by a day or two on date conventions (filing date
# vs ex-date vs effective date). When matching across vendors we
# bucket within this window.
_VENDOR_DATE_TOLERANCE_DAYS = 7

# yfinance returns ratios as floats — round-trip through Fraction with
# this denominator cap to recover integer (post, pre). All real splits
# we've seen reduce within this cap; XTIA's 1-for-250 and IQST's
# 1017:1000 both work.
_FRACTION_DENOM_CAP = 10000

_HTTP_TIMEOUT = 30  # seconds — Finviz endpoint is ~600KB, can be slow


@dataclass(frozen=True)
class SplitEvent:
    """One executed split. `pre`/`post` mirror the ApplySplit
    Pydantic model so callers can build mutations directly."""
    effective_date: str          # YYYY-MM-DD
    pre: int                     # 100 for a 1-for-100 reverse
    post: int                    # 1
    direction: Literal["forward", "reverse"]
    units: Literal["common", "ads"]
    source: str                  # "finviz" | "yfinance" | "finviz+yfinance"


# ─── Vendor fetchers ────────────────────────────────────────────────
def fetch_finviz_splits(ticker: str) -> list[SplitEvent]:
    """Fetch the Finviz split-history dump and filter to one ticker.

    Returns [] on any failure (missing key, network error, parse
    error, ticker not in dump). `units` is left as "common"; the
    splits-stage merge layer rewrites to "ads" for FPI issuers.
    """
    if not config.FINVIZ_API_KEY:
        log.debug("finviz: no FINVIZ_API_KEY configured")
        return []
    base = config.FINVIZ_BASE_URL.rstrip("/")
    url = f"{base}/export/split-history"
    try:
        resp = requests.get(
            url, params={"auth": config.FINVIZ_API_KEY},
            timeout=_HTTP_TIMEOUT,
            headers={"User-Agent": "FinvizApps/Dilution"},
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        log.warning("finviz split-history fetch failed: %s", e)
        return []
    if not isinstance(data, list):
        log.warning("finviz split-history: expected list, got %s",
                    type(data).__name__)
        return []

    t = ticker.upper()
    out: list[SplitEvent] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if (entry.get("ticker") or "").upper() != t:
            continue
        exdate = entry.get("exdate")
        post = entry.get("factorFrom")
        pre = entry.get("factorTo")
        if not (exdate and isinstance(post, int) and isinstance(pre, int)
                and post >= 1 and pre >= 1 and post != pre):
            log.debug("finviz: skipping malformed entry %s", entry)
            continue
        direction = "forward" if post > pre else "reverse"
        out.append(SplitEvent(
            effective_date=str(exdate),
            pre=pre, post=post,
            direction=direction,
            units="common",
            source="finviz",
        ))
    return out


def fetch_yfinance_splits(ticker: str) -> list[SplitEvent]:
    """Fetch yfinance's split history for one ticker.

    yfinance returns floats; we round-trip through Fraction to
    recover the original integer pre/post. Failures (missing module,
    network, no splits) return [].
    """
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed — yfinance split source unavailable")
        return []
    try:
        series = yf.Ticker(ticker.upper()).splits
    except Exception as e:
        log.warning("yfinance splits fetch failed for %s: %s", ticker, e)
        return []
    if series is None or len(series) == 0:
        return []

    out: list[SplitEvent] = []
    for ts, ratio in series.items():
        try:
            ratio = float(ratio)
        except (TypeError, ValueError):
            continue
        if ratio <= 0 or ratio == 1.0:
            continue
        # post/pre as integers. limit_denominator preserves exact
        # ratios for typical splits (1:100, 4:1, 1017:1000) while
        # collapsing yfinance's float representation noise.
        frac = Fraction(ratio).limit_denominator(_FRACTION_DENOM_CAP)
        post, pre = frac.numerator, frac.denominator
        if post == pre or post < 1 or pre < 1:
            continue
        direction = "forward" if post > pre else "reverse"
        try:
            effective_date = ts.strftime("%Y-%m-%d")
        except AttributeError:
            effective_date = str(ts)[:10]
        out.append(SplitEvent(
            effective_date=effective_date,
            pre=pre, post=post,
            direction=direction,
            units="common",
            source="yfinance",
        ))
    return out


# ─── Merge logic ─────────────────────────────────────────────────────
def _within_tolerance(a: str, b: str) -> bool:
    try:
        da = datetime.fromisoformat(a).date()
        db = datetime.fromisoformat(b).date()
    except ValueError:
        return a == b
    return abs((da - db).days) <= _VENDOR_DATE_TOLERANCE_DAYS


def merge_split_sources(
    finviz: list[SplitEvent], yfinance: list[SplitEvent],
) -> list[SplitEvent]:
    """Combine vendor lists into one canonical sequence.

    Same-event matching by date ± 7 days. Yfinance wins on ratio
    disagreements (logged as warning). Output is sorted by
    effective_date.
    """
    finviz_sorted = sorted(finviz, key=lambda s: s.effective_date)
    yfinance_sorted = sorted(yfinance, key=lambda s: s.effective_date)
    used_yf: set[int] = set()
    merged: list[SplitEvent] = []

    for fv in finviz_sorted:
        match_idx = None
        for i, yf in enumerate(yfinance_sorted):
            if i in used_yf:
                continue
            if _within_tolerance(fv.effective_date, yf.effective_date):
                match_idx = i
                break
        if match_idx is None:
            merged.append(fv)
            continue
        yf = yfinance_sorted[match_idx]
        used_yf.add(match_idx)
        if fv.post == yf.post and fv.pre == yf.pre:
            # Agreement — emit one row, prefer the more reliable
            # date (yfinance's ex-date matches the exchange's
            # CUSIP-change effective date).
            merged.append(SplitEvent(
                effective_date=yf.effective_date,
                pre=yf.pre, post=yf.post,
                direction=yf.direction,
                units=yf.units,
                source="finviz+yfinance",
            ))
        else:
            log.warning(
                "split source disagreement on %s: finviz=%d-for-%d "
                "yfinance=%d-for-%d (using yfinance)",
                fv.effective_date, fv.post, fv.pre, yf.post, yf.pre,
            )
            merged.append(yf)

    for i, yf in enumerate(yfinance_sorted):
        if i not in used_yf:
            merged.append(yf)

    merged.sort(key=lambda s: s.effective_date)
    return merged


# ─── Top-level API ──────────────────────────────────────────────────
class SplitFetchError(RuntimeError):
    """Raised when ALL vendor sources fail. Walk should abort —
    proceeding without splits silently produces a corrupted ledger
    for any issuer that had a reverse split during the window."""


def fetch_and_persist_splits(
    cik: int, ticker: str, *, is_fpi: bool = False,
) -> list[SplitEvent]:
    """Fetch from both vendors, merge, persist to dilution_splits,
    return the merged list.

    Raises `SplitFetchError` only when BOTH vendors raise — an
    individual vendor returning [] (legitimate: ticker has no splits)
    is fine. Determining "no splits at all" vs "all vendors are down"
    is impossible without trying both, so we capture per-vendor
    success flags.

    `is_fpi` rewrites units to "ads" so ADS-issuer splits flow into
    the right `_apply_split` branch. (The store also has an
    issuer-level default but explicit is better than implicit here.)
    """
    finviz_ok = False
    yfinance_ok = False
    finviz: list[SplitEvent] = []
    yfinance: list[SplitEvent] = []

    try:
        finviz = fetch_finviz_splits(ticker)
        finviz_ok = True
    except Exception as e:
        log.warning("finviz vendor errored: %s", e)
    try:
        yfinance = fetch_yfinance_splits(ticker)
        yfinance_ok = True
    except Exception as e:
        log.warning("yfinance vendor errored: %s", e)

    if not (finviz_ok or yfinance_ok):
        raise SplitFetchError(
            f"all vendors failed for {ticker} (cik={cik}) — refusing "
            f"to proceed without splits"
        )

    merged = merge_split_sources(finviz, yfinance)
    if is_fpi:
        merged = [SplitEvent(
            effective_date=s.effective_date, pre=s.pre, post=s.post,
            direction=s.direction, units="ads", source=s.source,
        ) for s in merged]

    _persist(cik, merged)
    log.info(
        "  splits — finviz=%d yfinance=%d merged=%d",
        len(finviz), len(yfinance), len(merged),
    )
    return merged


def _persist(cik: int, events: list[SplitEvent]) -> None:
    """Replace this CIK's rows with the merged set. Vendor data is
    authoritative; if a row disappears from both vendors (rare —
    typically a vendor data fix) we drop it here too."""
    ts = now_iso()
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM dilution_splits WHERE cik = ?", (cik,),
        )
        conn.executemany(
            """INSERT INTO dilution_splits
                 (cik, effective_date, pre, post, direction, units,
                  source, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [(cik, e.effective_date, e.pre, e.post, e.direction,
              e.units, e.source, ts) for e in events],
        )


def load_splits(cik: int, since_date: str | None = None) -> list[SplitEvent]:
    """Read persisted splits back out for the walker. Filters to
    `effective_date >= since_date` when supplied — splits before the
    walker's window can't affect any instrument it creates."""
    sql = """SELECT effective_date, pre, post, direction, units, source
               FROM dilution_splits
              WHERE cik = ?"""
    params: list = [cik]
    if since_date:
        sql += " AND effective_date >= ?"
        params.append(since_date)
    sql += " ORDER BY effective_date ASC"
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [SplitEvent(
        effective_date=r["effective_date"], pre=r["pre"], post=r["post"],
        direction=r["direction"], units=r["units"], source=r["source"],
    ) for r in rows]
