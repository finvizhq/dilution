"""5-year filing index via edgartools Company.get_filings().

We grab all filings and keep only dilution-relevant forms; stored into
dilution_filings so later stages can look them up by accession without
hitting SEC again.
"""

import logging

from edgar import Company, set_identity

import config
from db import get_conn, now_iso

log = logging.getLogger(__name__)

RELEVANT_PREFIXES = (
    "8-K",
    "6-K",  # foreign private issuer analog of 8-K (XTLB, BNTX, NVS, etc.)
    "424B",
    "S-1", "S-3", "S-3ASR", "S-4", "F-1", "F-3",
    "POS AM",
    # Form 425 — written communications related to a merger / business
    # combination filed under Rule 425 of the Securities Act. Per the
    # DilutionTracker cheatsheet, content is close to identical to the
    # accompanying 8-K, so we route it through the 8-K handler.
    "425",
    # Status-change signals for active shelves/S-1s:
    #   RW         — registration withdrawal request (filer pulls it)
    #   EFFECT     — SEC notice of effectiveness for a shelf
    "RW",
    "EFFECT",
    "10-K", "10-Q",
    "20-F", "40-F",
    "DEF 14A", "DEFA14A", "DEFM14A", "PRE 14A",
    # Less-common proxy variants the dispatcher already routes:
    #   DEFC14A — definitive contested proxy
    #   DEFR14A — definitive revised proxy
    #   PREM14A — preliminary merger proxy
    #   PRER14A — preliminary revised proxy
    "DEFC14A", "DEFR14A", "PREM14A", "PRER14A",
    "FWP",
    # Regulation A+ (Tier 1 / Tier 2) filings. Microcap dilution path
    # commonly used by very low-quality issuers per the cheatsheet:
    #   1-A    — offering circular (S-1 analog for Reg A+)
    #   1-U    — current report (8-K analog)
    #   1-K    — annual report (10-K analog)
    #   1-SA   — semi-annual report (10-Q analog; only one mid-year)
    "1-A", "1-U", "1-K", "1-SA",
)


def _is_relevant(form: str) -> bool:
    if not form:
        return False
    f = form.strip()
    return any(f.startswith(p) for p in RELEVANT_PREFIXES)


def pull_filing_index(cik: int, since_date: str) -> int:
    set_identity(config.EDGAR_IDENTITY)
    c = Company(cik)
    filings = c.get_filings().filter(date=f"{since_date}:")

    # Bulk-extract the EDGAR submissions JSON columns we need. These come
    # from a single cached HTTP fetch the Filings object already made;
    # accessing them per-filing via `f.period_of_report` triggers a
    # per-filing SGML round-trip (~480ms × N filings). The DataFrame
    # path gives the same data for free.
    df = filings.to_pandas()
    bulk: dict[str, dict] = {}
    if not df.empty:
        for col in ("reportDate", "items", "primaryDocument"):
            if col not in df.columns:
                df[col] = None
        for _, r in df.iterrows():
            bulk[r["accession_number"]] = {
                "report_date": _norm_date(r.get("reportDate")),
                "items": _norm_str(r.get("items")),
                "primary_doc": _norm_str(r.get("primaryDocument")),
            }

    n = 0
    with get_conn() as conn:
        for f in filings:
            form = f.form
            if not _is_relevant(form):
                continue
            accession = f.accession_no
            filing_date = str(f.filing_date) if f.filing_date else None
            if not filing_date or filing_date < since_date:
                continue

            b = bulk.get(accession, {})
            report_date = b.get("report_date")
            items = b.get("items")
            primary_doc_name = b.get("primary_doc")

            homepage = f.homepage_url
            # filing_url and document_url both trigger SGML fetches.
            # Store None — fetch_raw resolves the doc itself when needed.
            primary_doc_url = None

            conn.execute(
                """INSERT INTO dilution_filings
                     (accession_number, cik, form, filing_date, report_date,
                      primary_doc, primary_doc_url, homepage_url, items, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(accession_number) DO UPDATE SET
                     form = excluded.form,
                     filing_date = excluded.filing_date,
                     report_date = excluded.report_date,
                     primary_doc = excluded.primary_doc,
                     primary_doc_url = excluded.primary_doc_url,
                     homepage_url = excluded.homepage_url,
                     items = excluded.items""",
                (
                    accession, cik, form, filing_date, report_date,
                    primary_doc_name, primary_doc_url, homepage, items, now_iso(),
                ),
            )
            n += 1

    log.info("  %d relevant filings since %s", n, since_date)
    return n


def _norm_str(x) -> str | None:
    """DataFrame cells may be NaN, empty string, or real values."""
    if x is None:
        return None
    try:
        import pandas as pd
        if pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x).strip()
    return s or None


def _norm_date(x) -> str | None:
    s = _norm_str(x)
    if not s:
        return None
    # Already YYYY-MM-DD from EDGAR; defensive in case Pandas hands us a Timestamp
    if hasattr(x, "strftime"):
        try:
            return x.strftime("%Y-%m-%d")
        except Exception:
            pass
    return s[:10] if len(s) >= 10 else s
