"""Historical FX rates → USD.

Two-tier source strategy:

  1. Finviz `quote_export?instrument=forex` for the seven G10 majors
     we trade against USD: EUR/JPY/GBP/CHF/AUD/CAD/NZD. Finviz returns
     ~1200 daily bars (~4.5 years) which covers most of a cash-history
     chart's recent end.

  2. Frankfurter (ECB reference rates) for everything else: CNY, SGD,
     ILS, HKD, INR, KRW, and any older period outside Finviz's window.
     Frankfurter is free, no key, covers ~30 currencies, supports
     single-day lookups going back to 1999.

Daily Finviz series are cached on disk as JSON under
~/.cache/dilution/fx/finviz/<CCY>.json. Frankfurter rates are cached
per (currency, date) under .../fx/frankfurter/<CCY>/<DATE>.json.

`to_usd(amount, currency, on)` is the only public entry point — it
returns USD or None on total failure.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "dilution" / "fx"
_FINVIZ_DIR = _CACHE_DIR / "finviz"
_FRANK_DIR = _CACHE_DIR / "frankfurter"
_FRANK_TIMEOUT = 8.0
_FRANK_MAX_FALLBACK_DAYS = 5
# Finviz daily snapshot is refreshed after the trading day closes. Cache
# is good for 24h before we re-fetch the series.
_FINVIZ_TTL_SECS = 24 * 3600

# Finviz-supported USD-cross majors. Pair convention is base→quote.
# For (CCY → USD) we use "<CCY>USD" if the major is the base, else
# "USD<CCY>" and invert. JPY/CHF/CAD are USD-base on Finviz.
_FINVIZ_PAIRS: dict[str, tuple[str, bool]] = {
    "EUR": ("EURUSD", False),
    "GBP": ("GBPUSD", False),
    "AUD": ("AUDUSD", False),
    "NZD": ("NZDUSD", False),
    "JPY": ("USDJPY", True),   # invert: USD per JPY = 1/quote
    "CHF": ("USDCHF", True),
    "CAD": ("USDCAD", True),
}


def to_usd(amount: float, currency: str, on: date) -> float | None:
    """Convert `amount` of `currency` to USD using the rate on `on`."""
    currency = currency.upper()
    if currency == "USD":
        return amount
    rate = _rate(currency, on)
    if rate is None:
        return None
    return amount * rate


# ─── source selection ────────────────────────────────────────────────

def _rate(currency: str, on: date) -> float | None:
    """Resolve a CCY→USD rate for `on`.

    Finviz takes precedence when the currency is a supported major AND
    the date falls within Finviz's ~4.5-year window. Frankfurter is the
    fallback in every other case.
    """
    if currency in _FINVIZ_PAIRS:
        r = _finviz_rate(currency, on)
        if r is not None:
            return r
    return _frankfurter_rate(currency, on)


# ─── Finviz path ─────────────────────────────────────────────────────

def _finviz_rate(currency: str, on: date) -> float | None:
    series = _finviz_series(currency)
    if not series:
        return None
    earliest = series[0][0]
    if on < earliest:
        # Outside Finviz's window — let frankfurter handle it.
        return None
    close = _nearest_close(series, on)
    if close is None:
        return None
    pair, invert = _FINVIZ_PAIRS[currency]
    return (1.0 / close) if invert else close


def _finviz_series(currency: str) -> list[tuple[date, float]] | None:
    """Daily (date, close) for a Finviz forex pair, oldest first."""
    pair, _invert = _FINVIZ_PAIRS[currency]
    cache = _FINVIZ_DIR / f"{currency}.json"
    if cache.exists():
        try:
            payload = json.loads(cache.read_text())
            if time.time() - payload.get("fetched_at", 0) < _FINVIZ_TTL_SECS:
                return [(date.fromisoformat(d), float(c))
                        for d, c in payload["series"]]
        except (OSError, ValueError, KeyError, TypeError):
            # TypeError covers a JSON `null` close (float(None)); a malformed
            # cache row of any kind falls through to a fresh re-fetch.
            pass

    rows = _finviz_quote(pair)
    if rows is None:
        return None
    series: list[tuple[date, float]] = []
    for r in rows:
        try:
            d = datetime.strptime(r["Date"], "%m/%d/%Y").date()
            series.append((d, float(r["Close"])))
        except (KeyError, ValueError):
            continue
    series.sort(key=lambda t: t[0])
    if not series:
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "fetched_at": time.time(),
        "pair": pair,
        "series": [(d.isoformat(), c) for d, c in series],
    }))
    return series


def _finviz_quote(pair: str) -> list[dict] | None:
    # Imported lazily — finviz_client pulls in config + requests at module load.
    from dilution.finviz_client import _client
    return _client()._get_csv(
        "quote_export",
        {"t": pair, "instrument": "forex", "r": "d1", "barsCount": "1200"},
    )


def _nearest_close(series: list[tuple[date, float]],
                   target: date) -> float | None:
    """Last close on or before `target`. Returns None if `target` precedes
    the series. Linear scan is fine — ~1200 entries, called per period."""
    last = None
    for d, c in series:
        if d > target:
            break
        last = c
    return last


# ─── Frankfurter path ────────────────────────────────────────────────

def _frankfurter_rate(currency: str, on: date) -> float | None:
    cached = _frankfurter_read(currency, on)
    if cached is not None:
        return cached
    rate = _frankfurter_fetch(currency, on)
    if rate is not None:
        _frankfurter_write(currency, on, rate)
    return rate


def _frankfurter_fetch(currency: str, on: date) -> float | None:
    for back in range(_FRANK_MAX_FALLBACK_DAYS + 1):
        d = on - timedelta(days=back)
        url = f"https://api.frankfurter.dev/v1/{d.isoformat()}?from={currency}&to=USD"
        req = urllib.request.Request(url, headers={"User-Agent": "dilution-tracker/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=_FRANK_TIMEOUT) as resp:
                payload = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            log.warning("frankfurter fetch failed for %s @ %s: %s", currency, d, e)
            return None
        rate = payload.get("rates", {}).get("USD")
        if rate is not None:
            return float(rate)
    log.warning("no frankfurter rate for %s within %d days of %s",
                currency, _FRANK_MAX_FALLBACK_DAYS, on)
    return None


def _frankfurter_path(currency: str, on: date) -> Path:
    return _FRANK_DIR / currency.upper() / f"{on.isoformat()}.json"


def _frankfurter_read(currency: str, on: date) -> float | None:
    p = _frankfurter_path(currency, on)
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())["rate"])
    except (OSError, ValueError, KeyError, TypeError):
        # TypeError covers a JSON `null` rate (float(None)) — a malformed
        # cache file returns None per the "unreadable -> None" contract.
        return None


def _frankfurter_write(currency: str, on: date, rate: float) -> None:
    p = _frankfurter_path(currency, on)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"rate": rate, "currency": currency, "date": on.isoformat()}))
