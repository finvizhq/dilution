"""Inspection / debug blueprint — the raw truth behind the product cards.

The main dashboard (app.py) shows the curated DT-style cards. This view
is for *inspecting* a ticker while debugging the walker: it dumps every
ledger row regardless of status (active / closed / superseded), the full
terms / outstanding / history JSON, drawdowns, the filing index with a
raw-markdown viewer, anchor-reconciliation diffs, dropped-mutation walk
errors, splits, and walk state. Nothing is filtered or prettified away.

Routes:
  /inspect                         ticker picker + jump box
  /inspect/<ticker>                everything for one issuer
  /inspect/<ticker>/raw/<accession>[?doc=]   raw filing markdown
"""

import json
import os

from flask import Blueprint, abort, render_template, request

from db import get_conn
from dilution.ledger.store import (
    get_drawdowns_by_instrument,
    get_open_instruments,
)
from dilution.ledger.view import render_ledger_view

inspect_bp = Blueprint("inspect", __name__)


# This blueprint dumps internal pipeline state (walk errors, anchor diffs,
# LLM narratives) and raw filing text — not for the public internet. Restrict
# to loopback + the VPN exit IPs; override with INSPECT_ALLOW_IPS (comma-sep).
# No reverse proxy fronts the dev server, so request.remote_addr is the true
# client IP. Respond 404 (not 403) so the endpoint isn't advertised to scanners.
_INSPECT_ALLOW = {
    ip.strip()
    for ip in os.environ.get(
        "INSPECT_ALLOW_IPS",
        "127.0.0.1,::1,178.41.101.134,217.138.206.6",
    ).split(",")
    if ip.strip()
}


@inspect_bp.before_request
def _restrict_inspect():
    if request.remote_addr not in _INSPECT_ALLOW:
        abort(404)


@inspect_bp.app_template_filter("pjson")
def _pjson(val):
    """Pretty-print a JSON string or dict/list; pass through plain text."""
    if val is None or val == "":
        return ""
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except (ValueError, TypeError):
            return val
    try:
        return json.dumps(val, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(val)


def _loads(val, default):
    if not val:
        return default
    try:
        return json.loads(val)
    except (ValueError, TypeError):
        return default


def _resolve_company(conn, ident):
    ident = ident.strip()
    if ident.isdigit():
        return conn.execute(
            "SELECT * FROM dilution_company WHERE cik = ?", (int(ident),)
        ).fetchone()
    return conn.execute(
        "SELECT * FROM dilution_company WHERE ticker = ?", (ident.upper(),)
    ).fetchone()


@inspect_bp.route("/inspect")
def inspect_index():
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT c.cik, c.ticker, c.name,
                      (SELECT COUNT(*) FROM dilution_ledger l WHERE l.cik=c.cik) ledger_rows,
                      (SELECT COUNT(*) FROM dilution_ledger l
                          WHERE l.cik=c.cik AND l.status='active') active_rows,
                      (SELECT COUNT(*) FROM dilution_walk_errors e WHERE e.cik=c.cik) errors,
                      (SELECT MAX(walked_at) FROM dilution_walk_state w WHERE w.cik=c.cik) walked_at
               FROM dilution_company c
               ORDER BY c.ticker""",
        ).fetchall()
    return render_template("inspect_index.html", tickers=[dict(r) for r in rows])


@inspect_bp.route("/inspect/<ident>")
def inspect_ticker(ident: str):
    with get_conn() as conn:
        company = _resolve_company(conn, ident)
        if not company:
            abort(404)
        cik = company["cik"]

        walk_state = conn.execute(
            "SELECT * FROM dilution_walk_state WHERE cik = ?", (cik,)
        ).fetchone()

        ledger_rows = conn.execute(
            """SELECT * FROM dilution_ledger WHERE cik = ?
               ORDER BY type, created_at, instrument_id""",
            (cik,),
        ).fetchall()

        drawdowns = conn.execute(
            """SELECT * FROM dilution_ledger_drawdowns WHERE cik = ?
               ORDER BY event_date, id""",
            (cik,),
        ).fetchall()

        narratives = conn.execute(
            """SELECT n.* FROM dilution_ledger_narrative n
               JOIN dilution_ledger l ON l.instrument_id = n.instrument_id
               WHERE l.cik = ?""",
            (cik,),
        ).fetchall()

        filings = conn.execute(
            """SELECT f.*,
                      (SELECT COUNT(*) FROM dilution_raw r
                          WHERE r.accession_number = f.accession_number) raw_docs
               FROM dilution_filings f WHERE f.cik = ?
               ORDER BY f.filing_date DESC, f.accession_number DESC""",
            (cik,),
        ).fetchall()

        anchor_diffs = conn.execute(
            """SELECT * FROM dilution_anchor_diffs WHERE cik = ?
               ORDER BY detected_at DESC, id DESC""",
            (cik,),
        ).fetchall()

        walk_errors = conn.execute(
            """SELECT * FROM dilution_walk_errors WHERE cik = ?
               ORDER BY detected_at DESC, id DESC""",
            (cik,),
        ).fetchall()

        splits = conn.execute(
            "SELECT * FROM dilution_splits WHERE cik = ? ORDER BY effective_date",
            (cik,),
        ).fetchall()

    # Decode ledger rows: unpack JSON blobs and attach drawdowns + narrative.
    dd_by_instr = {}
    for d in drawdowns:
        dd_by_instr.setdefault(d["instrument_id"], []).append(dict(d))
    narr_by_instr = {n["instrument_id"]: dict(n) for n in narratives}

    instruments = []
    status_counts = {}
    type_counts = {}
    for r in ledger_rows:
        row = dict(r)
        row["terms"] = _loads(row.get("terms_json"), {})
        row["outstanding"] = _loads(row.get("outstanding_json"), {})
        row["history"] = _loads(row.get("history_json"), [])
        row["drawdowns"] = dd_by_instr.get(row["instrument_id"], [])
        row["narrative"] = narr_by_instr.get(row["instrument_id"])
        instruments.append(row)
        base_status = (row.get("status") or "").split(":", 1)[0]
        status_counts[base_status] = status_counts.get(base_status, 0) + 1
        type_counts[row["type"]] = type_counts.get(row["type"], 0) + 1

    # The exact text block the walker feeds the LLM at each filing, rendered
    # against the CURRENT ledger — i.e. what the model would see if the next
    # filing were walked right now. Uses the same get_open_instruments +
    # render_ledger_view path as walker.py (no historical reconstruction;
    # the per-filing snapshots the LLM actually saw are not persisted).
    try:
        llm_open_rows = get_open_instruments(cik)
        llm_ledger_view = render_ledger_view(
            llm_open_rows,
            drawdowns_by_instrument=get_drawdowns_by_instrument(cik),
        )
    except Exception as exc:  # noqa: BLE001 — debug view must never 500
        llm_open_rows = []
        llm_ledger_view = f"(failed to render ledger view: {exc!r})"

    return render_template(
        "inspect.html",
        llm_ledger_view=llm_ledger_view,
        llm_view_rows=len(llm_open_rows),
        company=dict(company),
        cik=cik,
        walk_state=dict(walk_state) if walk_state else None,
        instruments=instruments,
        status_counts=status_counts,
        type_counts=type_counts,
        drawdowns=[dict(d) for d in drawdowns],
        filings=[dict(f) for f in filings],
        anchor_diffs=[dict(a) for a in anchor_diffs],
        walk_errors=[dict(e) for e in walk_errors],
        splits=[dict(s) for s in splits],
    )


@inspect_bp.route("/inspect/<ident>/raw/<path:accession>")
def inspect_raw(ident: str, accession: str):
    want_doc = request.args.get("doc")
    with get_conn() as conn:
        company = _resolve_company(conn, ident)
        if not company:
            abort(404)
        filing = conn.execute(
            "SELECT * FROM dilution_filings WHERE accession_number = ?",
            (accession,),
        ).fetchone()
        docs = conn.execute(
            """SELECT doc_name, doc_type, length(content_md) AS len
               FROM dilution_raw WHERE accession_number = ?
               ORDER BY doc_name""",
            (accession,),
        ).fetchall()
        docs = [dict(d) for d in docs]
        content = None
        chosen = None
        if docs:
            chosen = want_doc or docs[0]["doc_name"]
            row = conn.execute(
                """SELECT content_md FROM dilution_raw
                   WHERE accession_number = ? AND doc_name = ?""",
                (accession, chosen),
            ).fetchone()
            content = row["content_md"] if row else None

    return render_template(
        "inspect_raw.html",
        company=dict(company),
        ticker=company["ticker"],
        accession=accession,
        filing=dict(filing) if filing else None,
        docs=docs,
        chosen=chosen,
        content=content,
    )
