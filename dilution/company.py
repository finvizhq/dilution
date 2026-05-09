"""Ticker/CIK resolution via edgartools, persisted into dilution_company.

Accepts either a ticker (e.g. "BINI") or a numeric CIK string (e.g. "1499961").
Some tracked companies are delisted/renamed — in that case pass the CIK directly.
"""

import logging

from edgar import Company, set_identity

import config
from db import get_conn, now_iso

log = logging.getLogger(__name__)


def _set_identity_once():
    set_identity(config.EDGAR_IDENTITY)


def resolve(identifier: str) -> dict:
    """Resolve ticker or CIK → {cik, ticker, name}. Identifier may be:
      - a ticker string (e.g. "BINI")
      - a numeric CIK (as string or int)
    """
    _set_identity_once()
    ident = str(identifier).strip()
    if ident.isdigit():
        c = Company(int(ident))
    else:
        c = Company(ident.upper())
    ticker = (c.tickers[0] if getattr(c, "tickers", None) else ident.upper())
    return {"cik": int(c.cik), "ticker": ticker, "name": c.name}


def upsert_company(cik: int, ticker: str, name: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO dilution_company (cik, ticker, name, added_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(cik) DO UPDATE SET
                 ticker = excluded.ticker,
                 name = excluded.name""",
            (cik, ticker, name, now_iso()),
        )


def ensure_company(identifier: str) -> dict:
    info = resolve(identifier)
    upsert_company(info["cik"], info["ticker"], info["name"])
    log.info("resolved %s → CIK %d %s (%s)",
             identifier, info["cik"], info["ticker"], info["name"])
    return info


def get_company_by_ticker(identifier: str) -> dict | None:
    ident = str(identifier).strip()
    with get_conn() as conn:
        if ident.isdigit():
            row = conn.execute(
                "SELECT cik, ticker, name FROM dilution_company WHERE cik = ?",
                (int(ident),),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT cik, ticker, name FROM dilution_company WHERE ticker = ?",
                (ident.upper(),),
            ).fetchone()
    return dict(row) if row else None


def get_unit_context(cik: int) -> dict:
    """Reporting-unit metadata for a company. Used by every extractor
    prompt to tell the LLM which unit to report share counts in.

    Returns:
        is_fpi: 0 | 1
        ads_ratio: ordinary shares per 1 ADS, or None
        reporting_unit: 'ads' (FPI) | 'common' (US issuer)

    Defaults safely to common-share semantics when the unit stage hasn't
    run yet."""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT is_fpi, ads_ratio FROM dilution_company WHERE cik = ?""",
            (cik,),
        ).fetchone()
    is_fpi = int((row and row["is_fpi"]) or 0)
    ads_ratio = (row["ads_ratio"] if row else None)
    return {
        "is_fpi": is_fpi,
        "ads_ratio": ads_ratio,
        "reporting_unit": "ads" if is_fpi else "common",
    }
