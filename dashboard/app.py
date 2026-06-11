"""Dashboard Flask app — cards-only build.

Two routes:
  /             tracked-ticker index
  /t/<TICKER>   per-ticker view, just the DT-style instrument cards

No charts, KPIs, risk, events table. Cards are the product.
"""

import logging
import os
import time
from collections import deque
from datetime import date
from pathlib import Path

from flask import Flask, abort, render_template, request

from dashboard._cash_chart import render as render_cash_chart
from dashboard._os_chart import SEG_COLORS, render as render_os_chart
from db import get_conn
from dilution.badges import compute_badges
from dilution.capital_raised import capital_raised_since
from dilution.cash_history import fetch_cash_history_cached
from dilution.finviz_client import fundamentals as finviz_fundamentals
from dilution.os_history import build_fd_stack, fetch_os_history_cached
from dilution.share_counts import fetch_implied_outstanding_cached
from dilution import ticker_brief as brief_mod
from dilution.ledger.cards import (
    all_pending_s1_offerings,
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

from dashboard.inspect import inspect_bp  # noqa: E402

app.register_blueprint(inspect_bp)


# --- Per-IP rate limit (defense-in-depth for the public dev server) --------
# Fixed-window counter. Static assets are exempt so a normal page load
# (HTML + CSS) doesn't burn the budget. The dev server is single-process,
# so an in-memory map is sufficient. Tunable via env.
_RL_MAX = int(os.environ.get("DASHBOARD_RATE_MAX", "90"))      # requests
_RL_WINDOW = int(os.environ.get("DASHBOARD_RATE_WINDOW", "60"))  # seconds
_rl_hits: dict[str, deque] = {}


@app.before_request
def _rate_limit():
    if request.path.startswith("/static/"):
        return
    now = time.monotonic()
    cutoff = now - _RL_WINDOW
    ip = request.remote_addr or "?"
    dq = _rl_hits.setdefault(ip, deque())
    while dq and dq[0] <= cutoff:
        dq.popleft()
    if len(dq) >= _RL_MAX:
        abort(429)
    dq.append(now)
    # Opportunistic sweep so a flood of unique IPs can't grow the map forever.
    if len(_rl_hits) > 10000:
        for k in list(_rl_hits.keys()):
            d = _rl_hits[k]
            while d and d[0] <= cutoff:
                d.popleft()
            if not d:
                del _rl_hits[k]


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


def _fmt_usd_m(x):
    """Format a USD amount as a signed $X.XXM string for the cash header."""
    if x is None:
        return "—"
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x)/1e6:,.2f}M"


# Concrete primary-document URLs (…/<acc>/<doc>.htm), matching DT. Shared
# with the card layer so the directory-vs-document logic lives in one place.
from dilution.ledger.cards import _edgar_url

# Compact share-count formatter (23.5K / 1.1M / 2.30B) shared with the
# badge strip so the O/S-chart legend chips match the badge wording.
from dilution.badges import _sh as _sh_compact


app.jinja_env.filters["shares"] = _fmt_shares
app.jinja_env.filters["usd"] = _fmt_usd
app.jinja_env.filters["usd_m"] = _fmt_usd_m
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


def _cards_for(cik: int, fund: dict | None, latest_os) -> dict:
    """The seven card arrays — shared by the ticker page and the brief
    regeneration endpoint so both see the identical projection."""
    return {
        "s1_offering": s1_offering_cards(cik),
        "warrant": warrant_cards(cik),
        "convertible": convertible_note_cards(cik),
        "convertible_preferred": preferred_cards(cik),
        "atm": atm_cards(cik, fund, latest_os),
        "equity_line": equity_line_cards(cik),
        "shelf": shelf_cards(cik, fund, latest_os),
    }


def _cash_for(cik: int):
    """(cash, raised_since_report) via the two-phase fetch: probe XBRL
    for the latest reporting date, compute capital-raised-since, then
    build the bridged history. (None, None) when XBRL is unavailable."""
    try:
        probe = fetch_cash_history_cached(cik)
        raised = (capital_raised_since(cik, probe.latest_period_end)
                  if probe.latest_period_end else None)
        return fetch_cash_history_cached(cik, capital_raised_usd=raised), raised
    except Exception:
        log.exception("cash fetch failed for cik=%s", cik)
        return None, None


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
    # Prefer XBRL-derived fully-exchanged count (Class A + Class B for
    # Up-C tickers; matches DilutionTracker). Falls back to Finviz's
    # cover-page Class-A-only number when XBRL is missing/stale/an FPI
    # has no known ADS ratio. See dilution.share_counts for the rules.
    implied = fetch_implied_outstanding_cached(cik)
    latest_os = (implied.total
                 if implied.total is not None
                 else (fund or {}).get("shares_outstanding"))

    cards = _cards_for(cik, fund, latest_os)

    # Historical O/S & Potential Dilution chart. The FD bar's dark base
    # is the same latest_os shown elsewhere on the page; the stacked
    # segments are derived from the cards rendered below it.
    os_chart_svg = ""
    os_legend = []
    osh = None
    price = (fund or {}).get("price")
    try:
        osh = fetch_os_history_cached(cik)
        stack = build_fd_stack(cards, price)
        if osh.series or (latest_os and stack):
            if implied.total is not None:
                latest_note = (f"{implied.source_form} XBRL"
                               + (f", a/o {implied.as_of.isoformat()}"
                                  if implied.as_of else ""))
            else:
                latest_note = "Finviz cover-page count (XBRL unavailable)"
            os_chart_svg = render_os_chart(osh, latest_os, latest_note, stack)
            os_legend = [{
                "label": s.label,
                "color": SEG_COLORS.get(s.key, "#999"),
                "shares_m": _sh_compact(s.shares),
            } for s in stack]
    except Exception:
        log.exception("O/S chart failed for %s", ticker)
        os_chart_svg, os_legend = "", []

    cash, raised = _cash_for(cik)
    cash_chart_svg = ""
    try:
        if cash and cash.series:
            cash_chart_svg = render_cash_chart(cash)
    except Exception:
        log.exception("cash chart failed for %s", ticker)

    try:
        badges = compute_badges(cik, fund=fund, latest_os=latest_os,
                                cards=cards, cash=cash)
    except Exception:
        log.exception("badges failed for %s", ticker)
        badges = None

    # AI dilution brief — cached headline + bullets generated from the
    # same deterministic objects rendered above. Display-only: the
    # cache is populated by scripts/run_brief_all.py (skip-if-fresh
    # batch) or ticker_brief_test.py. Staleness = a filing arrived
    # after generation (the facts hash is too churny for this — it
    # embeds the live price via fundamentals and badge detail lines,
    # so it flips intraday without any new information).
    brief, brief_stale = None, False
    try:
        brief = brief_mod.get_cached(cik)
        if brief:
            with get_conn() as conn:
                latest_filing = conn.execute(
                    "SELECT MAX(filing_date) d FROM dilution_filings "
                    "WHERE cik = ?", (cik,),
                ).fetchone()["d"]
            brief_stale = bool(latest_filing
                               and latest_filing > brief["generated_at"][:10])
    except Exception:
        log.exception("brief lookup failed for %s", ticker)

    return render_template(
        "ticker.html",
        ticker=ticker,
        name=row["name"],
        cik=cik,
        finviz=fund,
        cards=cards,
        cash=cash,
        cash_chart_svg=cash_chart_svg,
        badges=badges,
        brief=brief,
        brief_stale=brief_stale,
        os_chart_svg=os_chart_svg,
        os_legend=os_legend,
        os_price=price,
        os_price_asof=date.today().strftime("%b %-d, %Y"),
        os_ads_ratio=(osh.ads_ratio if (osh and os_chart_svg) else None),
    )


@app.route("/pending-s1")
def pending_s1():
    """Cross-issuer DT-parity view of in-progress S-1 / F-1 offerings."""
    rows = all_pending_s1_offerings()
    if rows:
        ciks = tuple({r["cik"] for r in rows})
        placeholders = ",".join("?" * len(ciks))
        with get_conn() as conn:
            company_rows = conn.execute(
                f"SELECT cik, ticker, name FROM dilution_company "
                f"WHERE cik IN ({placeholders})",
                ciks,
            ).fetchall()
        company = {r["cik"]: dict(r) for r in company_rows}
    else:
        company = {}
    for r in rows:
        info = company.get(r["cik"], {})
        r["ticker"] = info.get("ticker")
        r["company_name"] = info.get("name")
    return render_template("pending_s1.html", offerings=rows)
