"""Minimal Finviz Elite client — fundamentals fetch only.

This is a slimmed-down, dilution-only fork of why-moving's
src/finviz_client.py. Why-moving's client supports news, screener,
earnings, transcripts, futures, crypto, and forex — none of which
this repo needs. We only need a single-ticker fundamentals snapshot
to fill the market-data fields DilutionTracker shows in its headline
panel (price, market cap, float, short interest, institutional
ownership), plus what we need for the baby-shelf calc.

Auth: API key in `auth=` query param. Endpoint: GET /export
(returns CSV). The response shape depends on the columns requested
via the `c=` parameter — see FUNDAMENTALS_COLS below.

Designed to fail soft: any HTTP / parse / missing-ticker error
returns None. Callers MUST handle None, since the dilution pipeline
is functional without market data.
"""

import csv
import io
import logging
import os
import re
import time
from typing import Optional

import requests

import config

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10  # seconds

# Column-code list for the /export endpoint with view 152 (Custom).
# Each integer maps to a specific data field in Finviz's screener grid.
# Source: https://elite.finviz.com/screener?v=152&ft=4&c=...
FUNDAMENTALS_COLS = (
    "1",     # Ticker
    "2",     # Company
    "3",     # Sector
    "4",     # Industry
    "5",     # Country
    "129",   # Exchange
    "6",     # Market Cap (millions of USD)
    "24",    # Shares Outstanding (millions)
    "25",    # Shares Float (millions)
    "28",    # Institutional Ownership (%)
    "30",    # Short Float (% of float)
    "63",    # Average Volume (thousands of shares)
    "65",    # Price (USD)
)

# Header→output-key map. Finviz's CSV header strings are stable for a
# given column set; we read by header name (not positional) so the code
# is robust to column-order changes in `FUNDAMENTALS_COLS`.
HEADER_TO_KEY = {
    "Ticker": "ticker",
    "Company": "company",
    "Sector": "sector",
    "Industry": "industry",
    "Country": "country",
    "Exchange": "exchange",
    "Price": "price",
    "Market Cap": "market_cap",
    "Average Volume": "avg_volume",
    "Shares Outstanding": "shares_outstanding",
    "Shares Float": "float_shares",
    "Short Float": "short_interest_pct",
    "Institutional Ownership": "institutional_ownership_pct",
}

# Unit normalization. Finviz's CSV export reports market cap and share
# counts in MILLIONS, average daily volume in THOUSANDS. We rescale to
# raw units so callers can do arithmetic without mental conversion.
SCALE_MULTIPLIERS = {
    "market_cap": 1e6,
    "shares_outstanding": 1e6,
    "float_shares": 1e6,
    "avg_volume": 1e3,
}

NUMERIC_KEYS = (
    "price", "market_cap", "avg_volume",
    "shares_outstanding", "float_shares",
    "short_interest_pct", "institutional_ownership_pct",
)

_SCALE_SUFFIXES = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def _parse_num(value) -> Optional[float]:
    """Parse Finviz's text-formatted numbers.

    Handles "12.34M" / "1.23B" / "1,234,567" / "12.3%" / "-" / "" /
    parenthesized negatives. Returns None for unparseable input.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s in ("-", "—", "N/A"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    s = s.replace(",", "").replace("$", "").replace("%", "")
    s = s.strip()
    mult = 1.0
    if s and s[-1].upper() in _SCALE_SUFFIXES:
        mult = _SCALE_SUFFIXES[s[-1].upper()]
        s = s[:-1]
    try:
        v = float(s) * mult
        return -v if neg else v
    except ValueError:
        return None


def _market_now():
    """Current datetime in US market time. Falls back to local time if
    zoneinfo is unavailable."""
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return datetime.now()


def _current_session_date() -> str:
    """Today's date in US market time as MM/DD/YYYY (matches Finviz's
    daily-export Date column). Used to identify and drop the current,
    still-trading session, whose Close is the live price, not a close."""
    return _market_now().strftime("%m/%d/%Y")


class FinvizClient:
    """Minimal Finviz Elite client. Single-ticker fundamentals only."""

    def __init__(self, api_key: str | None = None,
                 base_url: str | None = None,
                 timeout: float = DEFAULT_TIMEOUT):
        self.api_key = api_key or config.FINVIZ_API_KEY
        self.base_url = (base_url or config.FINVIZ_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "FinvizApps/Dilution"
        if not self.api_key:
            log.warning("FinvizClient instantiated without API key — "
                        "all calls will fail soft (return None).")

    def _get_csv(self, endpoint: str, params: dict) -> list[dict] | None:
        """GET an endpoint, parse as CSV, return list of row-dicts.

        Returns None on any failure (network, HTTP, parse, empty body).
        """
        if not self.api_key:
            return None
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        merged = {**params, "auth": self.api_key}
        try:
            resp = self.session.get(url, params=merged, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.warning("finviz GET %s failed: %s", endpoint, e)
            return None
        if not resp.content:
            return None
        try:
            reader = csv.DictReader(io.StringIO(resp.text))
            return list(reader)
        except csv.Error as e:
            log.warning("finviz CSV parse failed for %s: %s", endpoint, e)
            return None

    def get_fundamentals(self, ticker: str) -> dict | None:
        """Fetch a single ticker's fundamentals.

        Returns dict with keys from `HEADER_TO_KEY` values plus a
        `ticker` field. Numeric fields are parsed to floats (None when
        unparseable). Returns None if the ticker has no Finviz row,
        the API is unreachable, or the API key is unset.
        """
        if not ticker:
            return None
        params = {
            "v": "152",
            "t": ticker.upper(),
            "ft": "4",
            "c": ",".join(FUNDAMENTALS_COLS),
        }
        rows = self._get_csv("export", params)
        if not rows:
            return None
        row = rows[0]

        out: dict = {}
        for header, key in HEADER_TO_KEY.items():
            if header in row:
                out[key] = row[header]
        for k in NUMERIC_KEYS:
            if k in out:
                out[k] = _parse_num(out[k])
        for k, mult in SCALE_MULTIPLIERS.items():
            if out.get(k) is not None:
                out[k] = out[k] * mult
        return out

    def get_daily_closes(self, ticker: str, bars: int = 60,
                         within_calendar_days: int | None = None,
                         ) -> list[float] | None:
        """Fetch daily *settled* close prices for the trailing `bars` sessions.

        Endpoint: /quote_export with r=d1. Returns oldest-first list of
        floats, or None on any failure / empty response.

        The export's last bar is the current, still-trading session, whose
        Close column carries the *live* market price (it changes intraday
        and is not a settled close — e.g. BJDX mid-spike returned ~6.6 live
        vs a true 60-day high close of 2.18). We drop that bar and request
        one extra so the caller still gets `bars` settled closes.

        `within_calendar_days` additionally drops bars older than N
        calendar days (US market time). `bars` counts trading sessions
        (~1.45 calendar days each), so a bare bars=60 spans ~87 calendar
        days — too wide for windows the SEC defines in calendar days.
        """
        if not ticker:
            return None
        rows = self._get_csv("quote_export", {
            "t": ticker.upper(),
            "r": "d1",
            "barsCount": str(bars + 1),
        })
        if not rows:
            return None
        today = _current_session_date()
        if (rows[-1].get("Date") or "").strip() == today:
            rows = rows[:-1]
        if within_calendar_days is not None:
            from datetime import datetime, timedelta
            cutoff = (_market_now()
                      - timedelta(days=within_calendar_days)).date()

            def _in_window(r: dict) -> bool:
                try:
                    return datetime.strptime(
                        (r.get("Date") or "").strip(), "%m/%d/%Y",
                    ).date() >= cutoff
                except ValueError:
                    return False
            rows = [r for r in rows if _in_window(r)]
        out = []
        for r in rows[-bars:]:
            v = _parse_num(r.get("Close"))
            if v is not None:
                out.append(v)
        return out or None

    def highest_close(self, ticker: str, bars: int = 60,
                      within_calendar_days: int | None = 60,
                      ) -> float | None:
        """Highest settled close for the baby-shelf calc. Form S-3
        General Instruction I.B.6, Instruction 1 prices the float-value
        test at the last-sale price as of a date of the issuer's
        choosing "within 60 days prior to the date of sale" — calendar
        days, and issuers rationally choose the highest close. Hence
        the default 60-calendar-day bound; bars=60 alone would scan ~87
        calendar days and could surface a high the issuer can't use."""
        closes = self.get_daily_closes(
            ticker, bars=bars, within_calendar_days=within_calendar_days)
        return max(closes) if closes else None


_default_client: FinvizClient | None = None


def _client() -> FinvizClient:
    global _default_client
    if _default_client is None:
        _default_client = FinvizClient()
    return _default_client


# Cache fundamentals so a public /t/<ticker> page can't turn anonymous
# traffic into unbounded Finviz Elite API calls on our key. Market data is
# point-in-time and tolerates a short TTL. Failures are cached briefly so a
# transient outage doesn't get hammered, but recover quickly. Tunable via env.
_FUND_TTL = int(os.environ.get("FINVIZ_FUNDAMENTALS_TTL", "600"))  # seconds
_FUND_TTL_NEG = 30  # seconds — negative (None) result cache
_fund_cache: dict[str, tuple[float, Optional[dict]]] = {}


def fundamentals(ticker: str) -> dict | None:
    """Module-level convenience: lazy-singleton FinvizClient.

    Use from any caller that needs one-off market data. The shared
    client keeps HTTP keep-alive on a single Session across calls,
    so multiple lookups in the same Python process are cheaper.

    Results are memoized for _FUND_TTL seconds (None for _FUND_TTL_NEG)
    so repeated page loads don't each spend a Finviz Elite API call.
    """
    key = ticker.upper()
    now = time.monotonic()
    hit = _fund_cache.get(key)
    if hit is not None:
        ts, val = hit
        ttl = _FUND_TTL if val is not None else _FUND_TTL_NEG
        if now - ts < ttl:
            return val
    result = _client().get_fundamentals(ticker)
    _fund_cache[key] = (now, result)
    return result


def highest_close(ticker: str, bars: int = 60,
                  within_calendar_days: int | None = 60) -> float | None:
    """Highest settled close within 60 calendar days, for IB6 baby-shelf
    math (see FinvizClient.highest_close)."""
    return _client().highest_close(
        ticker, bars=bars, within_calendar_days=within_calendar_days)


def ib6_effective_price(ticker: str,
                        current_price: float | None = None) -> float | None:
    """Per SEC General Instruction I.B.6, an issuer may compute float
    market value using the higher of (current price, highest close in
    the past 60 calendar days). Returns the larger of the two, or None
    when neither is available.
    """
    high60 = highest_close(ticker, bars=60)
    if current_price is None and high60 is None:
        return None
    if current_price is None:
        return high60
    if high60 is None:
        return current_price
    return max(current_price, high60)
