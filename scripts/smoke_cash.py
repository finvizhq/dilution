"""Smoke test: dump cash-position bridge + render SVG for sample tickers.

Usage:
    python3 scripts/smoke_cash.py

Picks one US issuer and one FPI from the tracked-companies table. For
each: prints the historical series, the bridge components (latest cash,
prorated OpCF, capital raised, current cash estimate, months of cash),
and writes the rendered SVG to /tmp for visual inspection.

Eyeball-compare the bars against DilutionTracker's "Cash Position" card
on the same ticker.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard._cash_chart import render as render_cash_chart  # noqa: E402
from db import get_conn  # noqa: E402
from dilution.capital_raised import capital_raised_since  # noqa: E402
from dilution.cash_history import fetch_cash_history  # noqa: E402


def _pick_targets():
    with get_conn() as c:
        rows = c.execute(
            "SELECT ticker, cik, is_fpi FROM dilution_company ORDER BY ticker"
        ).fetchall()
    us = next((dict(r) for r in rows if not r["is_fpi"]), None)
    fpi = next((dict(r) for r in rows if r["is_fpi"]), None)
    targets = [t for t in (us, fpi) if t]
    if not targets:
        raise SystemExit("No tracked companies in dilution_company.")
    return targets


def _dump(target: dict) -> None:
    ticker = target["ticker"]
    cik = target["cik"]
    print(f"\n=== {ticker} (CIK {cik}, is_fpi={target['is_fpi']}) ===")

    # Probe once to learn the latest reporting date, then compute raises.
    probe = fetch_cash_history(cik)
    raised = None
    if probe.latest_period_end:
        raised = capital_raised_since(cik, probe.latest_period_end)
        print(f"capital raised since {probe.latest_period_end}: "
              f"${(raised or 0)/1e6:,.2f}M")

    h = fetch_cash_history(cik, capital_raised_usd=raised)
    print(f"series len:       {len(h.series)}")
    if h.series:
        print("last 5 periods:")
        for p in h.series[-5:]:
            native = (f"  ({p.native_currency} {p.native_value/1e6:,.2f}M)"
                      if p.native_currency != "USD" else "")
            print(f"  {p.end} {p.fp:>3}/{p.fy}  "
                  f"USD {p.value_usd/1e6:>9,.2f}M{native}  form={p.form}")
    print(f"latest period:    {h.latest_period_end}")
    print(f"latest cash:      ${(h.latest_cash_usd or 0)/1e6:,.2f}M")
    print(f"op CF quarterly:  ${(h.op_cf_quarterly_usd or 0)/1e6:,.2f}M")
    print(f"op CF prorated:   ${(h.op_cf_prorated_usd or 0)/1e6:,.2f}M")
    print(f"capital raised:   ${(h.capital_raised_usd or 0)/1e6:,.2f}M")
    print(f"current cash est: ${(h.current_cash_est_usd or 0)/1e6:,.2f}M")
    print(f"months of cash:   "
          f"{h.months_of_cash:.2f}" if h.months_of_cash is not None else
          "months of cash:   n/a (positive operating CF or no OpCF data)")
    print(f"stale days:       {h.stale_days}")
    print(f"fx_failed:        {h.fx_failed}")

    if h.series:
        svg = render_cash_chart(h)
        out = Path(f"/tmp/cash_{ticker}.svg")
        full = (
            f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{ticker} cash</title>"
            f"<style>body{{font-family:sans-serif;padding:20px}}"
            f"svg.cash-chart{{border:1px solid #ddd;background:#fff;"
            f"max-width:920px;width:100%}}</style></head>"
            f"<body><h2>{ticker} — Cash Position</h2>{svg}</body></html>"
        )
        out.write_text(full)
        print(f"wrote {out}")


def main() -> int:
    for t in _pick_targets():
        _dump(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
