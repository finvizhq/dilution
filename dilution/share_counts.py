"""Implied (fully-exchanged) common-share count for a company.

Pulls shares-outstanding facts directly from XBRL on the latest periodic
filing. For dual-class issuers (Up-C structures like GENK / Shake Shack
/ Sweetgreen, or multi-class economic structures like META / GOOG /
SNAP), this sums all common-stock classes — that's the right denominator
for dilution math and matches DilutionTracker's "Outstanding Shares"
line.

Why not Finviz: Finviz reports the SEC cover-page number for a single
class (Class A only on Up-C tickers). For GENK that's ~5M instead of
the ~33M implied by Class A + Class B → every %-dilution number on the
count was 6.6× overstated and the Baby Shelf classification was
wrong.

Why not edgartools' high-level Company.get_facts(): it silently drops
dimensioned facts. The per-class breakdown only surfaces via the
filing-level XBRL query path used here.

For Foreign Private Issuers (20-F filers), the XBRL cover-page count
reflects the *close of the period covered* — not "latest practicable
date" like a 10-K does. That means anything issued between period end
and now (RDOs, ATM, option exercises) is missing, and on tickers like
AACG that gap can be 35%+. We sidestep this by sourcing the FPI count
from yfinance, which aggregates post-period disclosures and reports in
the listed instrument (ADS). XBRL native_total / ads_ratio remains the
fallback when Yahoo is unavailable. FPIs without a known ratio AND no
Yahoo data return total=None.

Returns total=None on any failure — callers should fall through to
Finviz's number as a degraded-but-better-than-nothing fallback.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache

from edgar import Company, set_identity

import config
from db import get_conn

log = logging.getLogger(__name__)

# Probe order. dei: is the cover-page concept (dated near filing_date,
# typically ~1 week after period_end, so freshest). us-gaap: is the
# balance-sheet concept (dated at period_end). Either is acceptable.
_CONCEPTS = (
    "EntityCommonStockSharesOutstanding",
    "CommonStockSharesOutstanding",
)

# All forms that carry a cover-page share count.
_PERIODIC_FORMS = ("10-Q", "10-K", "20-F", "40-F")

# Above this max:min ratio between any two classes, direct summation is
# dimensionally suspect. Empirically Up-C and pure-economic multi-class
# (META/GOOG/SNAP) stay <100x. Berkshire's Class A:B sits at ~2770x
# because Class B has a 1/1500 economic claim, and naive summation
# produces a nonsense total. Flag, don't silently lie.
_CLASS_RATIO_WARN = 100.0

# Stop trusting XBRL once the as-of date is older than this — common
# for delisted / dark issuers that stopped filing. Caller falls back.
_STALE_DAYS = 180

_IDENTITY_SET = False


@dataclass(frozen=True)
class ClassCount:
    label: str
    value: float


@dataclass(frozen=True)
class ImpliedOutstanding:
    total: float | None              # payload units (ADS for FPI, common else)
    native_total: float | None       # issuer-reported units (ordinary for FPI)
    classes: tuple[ClassCount, ...] = ()
    as_of: date | None = None
    source_form: str | None = None
    source_accession: str | None = None
    source_concept: str | None = None
    ads_ratio: float | None = None
    is_fpi: bool = False
    stale_days: int | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImpliedFloat:
    """Float-share count, sourced from Yahoo.

    Finviz's `float_shares` lags badly on low-volume tickers (especially
    ADRs — AACG shows 8.84M on Finviz vs 15.04M on DilutionTracker /
    Yahoo). Yahoo reports for the listed instrument, so for FPIs the
    value is already in ADS units — matches how `cards.py` consumes
    float (price × float in payload currency).

    Returns shares=None on any failure; callers fall through to Finviz.
    """
    shares: float | None
    as_of: date | None = None
    source: str | None = None
    stale_days: int | None = None
    warnings: tuple[str, ...] = ()


def _ensure_identity() -> None:
    global _IDENTITY_SET
    if _IDENTITY_SET:
        return
    set_identity(getattr(config, "EDGAR_IDENTITY",
                         "dilution-tracker contact@example.com"))
    _IDENTITY_SET = True


def _company_unit(cik: int) -> tuple[bool, float | None, str | None]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT is_fpi, ads_ratio, ticker FROM dilution_company WHERE cik = ?",
            (int(cik),),
        ).fetchone()
    if not row:
        return False, None, None
    return (
        bool(row["is_fpi"]),
        float(row["ads_ratio"]) if row["ads_ratio"] is not None else None,
        row["ticker"],
    )


def _yahoo_shares(ticker: str) -> tuple[float | None, date | None]:
    """Latest shares-outstanding from Yahoo for an ADS-listed FPI.

    Yahoo reports for the listed instrument, so for AACG/XTLB/etc. the
    value is already in ADS — no division by ads_ratio needed.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None, None
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception as e:
        log.debug("yfinance info failed for %s: %s", ticker, e)
        return None, None
    shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
    if not shares or shares <= 0:
        return None, None
    as_of: date | None = None
    try:
        start = (date.today() - timedelta(days=180)).isoformat()
        series = t.get_shares_full(start=start)
        if series is not None and len(series) > 0:
            as_of = series.index[-1].date()
    except Exception as e:
        log.debug("yfinance shares-history failed for %s: %s", ticker, e)
    return float(shares), as_of


def _latest_periodic_filing(cik: int):
    """Most recent periodic filing across all forms.

    Iterating forms and returning the first match (the old behavior)
    silently picks a stale 10-Q over a current 20-F on issuers that
    flipped between domestic and FPI status (e.g. SCNI's 2023 10-Q
    still exists alongside its 2026 20-F).
    """
    c = Company(int(cik))
    candidates = []
    for form in _PERIODIC_FORMS:
        try:
            fs = c.get_filings(form=form).head(1)
            if len(fs) > 0:
                candidates.append(fs[0])
        except Exception:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda f: str(f.filing_date))


def _query_latest_period_facts(xbrl, concept: str):
    try:
        facts = xbrl.facts.query().by_concept(concept).execute()
    except Exception as e:
        log.debug("xbrl query %s failed: %s", concept, e)
        return [], None
    if not facts:
        return [], None
    by_period: dict[str, list] = {}
    for f in facts:
        p = f.get("period_instant") or ""
        if p:
            by_period.setdefault(p, []).append(f)
    if not by_period:
        return [], None
    latest = max(by_period)
    return by_period[latest], latest


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s)).date()
    except ValueError:
        return None


def fetch_implied_outstanding(cik: int, *, as_of: date | None = None
                              ) -> ImpliedOutstanding:
    _ensure_identity()
    as_of = as_of or date.today()
    is_fpi, ads_ratio, ticker = _company_unit(cik)

    try:
        filing = _latest_periodic_filing(int(cik))
    except Exception as e:
        log.warning("get_filings failed for CIK %s: %s", cik, e)
        return ImpliedOutstanding(
            total=None, native_total=None, is_fpi=is_fpi, ads_ratio=ads_ratio,
            warnings=("filings_lookup_failed",))

    if filing is None:
        return ImpliedOutstanding(
            total=None, native_total=None, is_fpi=is_fpi, ads_ratio=ads_ratio,
            warnings=("no_periodic_filing",))

    try:
        xbrl = filing.xbrl()
    except Exception as e:
        log.warning("xbrl() failed for %s: %s", filing.accession_no, e)
        return ImpliedOutstanding(
            total=None, native_total=None,
            source_form=filing.form, source_accession=filing.accession_no,
            is_fpi=is_fpi, ads_ratio=ads_ratio,
            warnings=("xbrl_load_failed",))

    facts: list = []
    used_concept: str | None = None
    used_period: str | None = None
    for concept in _CONCEPTS:
        facts, used_period = _query_latest_period_facts(xbrl, concept)
        if facts:
            used_concept = concept
            break
    if not facts:
        return ImpliedOutstanding(
            total=None, native_total=None,
            source_form=filing.form, source_accession=filing.accession_no,
            is_fpi=is_fpi, ads_ratio=ads_ratio,
            warnings=("xbrl_concept_missing",))

    classes: list[ClassCount] = []
    native_total = 0.0
    for f in facts:
        lbl = (f.get("dimension_member_label")
               or f.get("label")
               or "Common Stock")
        v = f.get("numeric_value")
        if v is None:
            continue
        v = float(v)
        classes.append(ClassCount(label=lbl, value=v))
        native_total += v

    warnings: list[str] = []
    values = sorted((c.value for c in classes if c.value > 0), reverse=True)
    if len(values) >= 2 and values[0] / values[-1] > _CLASS_RATIO_WARN:
        warnings.append("unequal_class_ratio")

    period_date = _parse_date(used_period)
    stale = (as_of - period_date).days if period_date else None
    if stale is not None and stale > _STALE_DAYS:
        warnings.append(f"stale_{stale}d")

    source_form = filing.form
    source_accession = filing.accession_no
    source_concept = used_concept
    result_as_of = period_date

    if is_fpi:
        y_shares, y_as_of = _yahoo_shares(ticker) if ticker else (None, None)
        if y_shares is not None:
            total: float | None = y_shares
            source_form = "yahoo"
            source_accession = None
            source_concept = "yfinance.sharesOutstanding"
            result_as_of = y_as_of
            stale = (as_of - y_as_of).days if y_as_of else None
            warnings = [w for w in warnings if not w.startswith("stale_")]
            if stale is not None and stale > _STALE_DAYS:
                warnings.append(f"stale_{stale}d")
        elif ads_ratio and ads_ratio > 0:
            total = native_total / ads_ratio
            warnings.append("yahoo_unavailable_xbrl_fallback")
        else:
            warnings.append("fpi_ads_ratio_missing")
            total = None
    else:
        total = native_total

    return ImpliedOutstanding(
        total=total,
        native_total=native_total,
        classes=tuple(classes),
        as_of=result_as_of,
        source_form=source_form,
        source_accession=source_accession,
        source_concept=source_concept,
        is_fpi=is_fpi,
        ads_ratio=ads_ratio,
        stale_days=stale,
        warnings=tuple(warnings),
    )


@lru_cache(maxsize=256)
def _cached(cik: int, as_of_iso: str) -> ImpliedOutstanding:
    return fetch_implied_outstanding(int(cik),
                                     as_of=date.fromisoformat(as_of_iso))


def fetch_implied_outstanding_cached(
        cik: int, *, as_of: date | None = None) -> ImpliedOutstanding:
    return _cached(int(cik), (as_of or date.today()).isoformat())


def _yahoo_float(ticker: str) -> tuple[float | None, date | None]:
    """Latest float-share count from Yahoo.

    Yahoo doesn't publish a separate `as_of` for float — it refreshes
    on the same cadence as `sharesOutstanding`, so we proxy the date
    via `get_shares_full`, matching `_yahoo_shares`.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None, None
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception as e:
        log.debug("yfinance info failed for %s: %s", ticker, e)
        return None, None
    shares = info.get("floatShares")
    if not shares or shares <= 0:
        return None, None
    as_of: date | None = None
    try:
        start = (date.today() - timedelta(days=180)).isoformat()
        series = t.get_shares_full(start=start)
        if series is not None and len(series) > 0:
            as_of = series.index[-1].date()
    except Exception as e:
        log.debug("yfinance shares-history failed for %s: %s", ticker, e)
    return float(shares), as_of


def fetch_float(cik: int, *, as_of: date | None = None) -> ImpliedFloat:
    as_of = as_of or date.today()
    _, _, ticker = _company_unit(cik)
    if not ticker:
        return ImpliedFloat(shares=None, warnings=("no_ticker",))
    shares, src_as_of = _yahoo_float(ticker)
    if shares is None:
        return ImpliedFloat(shares=None, warnings=("yahoo_unavailable",))
    stale = (as_of - src_as_of).days if src_as_of else None
    warnings: tuple[str, ...] = ()
    if stale is not None and stale > _STALE_DAYS:
        warnings = (f"stale_{stale}d",)
    return ImpliedFloat(
        shares=shares,
        as_of=src_as_of,
        source="yfinance.floatShares",
        stale_days=stale,
        warnings=warnings,
    )


@lru_cache(maxsize=256)
def _cached_float(cik: int, as_of_iso: str) -> ImpliedFloat:
    return fetch_float(int(cik), as_of=date.fromisoformat(as_of_iso))


def fetch_float_cached(cik: int, *, as_of: date | None = None) -> ImpliedFloat:
    return _cached_float(int(cik), (as_of or date.today()).isoformat())
