"""Dashboard Flask app — cards-only build.

Two routes:
  /             tracked-ticker index
  /t/<TICKER>   per-ticker view, just the DT-style instrument cards

No charts, KPIs, risk, events table. Cards are the product.
"""

import logging
from pathlib import Path

from flask import Flask, abort, render_template

from db import get_conn
from dilution.finviz_client import fundamentals as finviz_fundamentals
from dilution.ledger.cards import (
    atm_cards,
    convertible_note_cards,
    equity_line_cards,
    preferred_cards,
    s1_offering_cards,
    shelf_cards,
    warrant_cards,
)

log = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)


def _fmt_shares(x):
    if x is None:
        return "—"
    return f"{int(round(x)):,}"


def _fmt_usd(x):
    if x is None:
        return "—"
    sign = "-" if x < 0 else ""
    return f"{sign}${int(round(abs(x))):,}"


def _fmt_pct(x):
    if x is None:
        return "—"
    return f"{x:.1f}%"


def _edgar_url(accession_number, cik):
    if not accession_number or cik is None:
        return None
    return (f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{accession_number.replace('-', '')}/")


app.jinja_env.filters["shares"] = _fmt_shares
app.jinja_env.filters["usd"] = _fmt_usd
app.jinja_env.filters["pct"] = _fmt_pct
app.jinja_env.globals["edgar_url"] = _edgar_url


@app.route("/")
def index():
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT c.cik, c.ticker, c.name,
                      (SELECT COUNT(*) FROM dilution_ledger l
                          WHERE l.cik=c.cik AND l.status='active') events,
                      (SELECT COUNT(*) FROM dilution_filings f WHERE f.cik=c.cik) filings,
                      (SELECT MAX(filing_date) FROM dilution_filings f WHERE f.cik=c.cik) latest_filing
               FROM dilution_company c
               ORDER BY c.ticker""",
        ).fetchall()
    tickers = [dict(r) for r in rows]
    return render_template("index.html", tickers=tickers)


@app.route("/t/<ticker>")
def ticker_detail(ticker: str):
    ticker = ticker.upper()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT cik, ticker, name FROM dilution_company WHERE ticker = ?",
            (ticker,),
        ).fetchone()
    if not row:
        abort(404)
    cik = row["cik"]

    fund = finviz_fundamentals(ticker)
    latest_os = (fund or {}).get("shares_outstanding")

    cards = {
        "s1_offering": s1_offering_cards(cik),
        "warrant": warrant_cards(cik),
        "convertible": convertible_note_cards(cik),
        "convertible_preferred": preferred_cards(cik),
        "atm": atm_cards(cik, fund, latest_os),
        "equity_line": equity_line_cards(cik),
        "shelf": shelf_cards(cik, fund, latest_os),
    }

    return render_template(
        "ticker.html",
        ticker=ticker,
        name=row["name"],
        cik=cik,
        finviz=fund,
        cards=cards,
    )
