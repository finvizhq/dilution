"""Audit ledger rows whose status is 'active' but functionally aren't.

Flags three categories per CIK:
  * matured: convertible/note past terms.maturity
  * expired: warrant past terms.expiration
  * zero:    outstanding count/principal_remaining/shares ≤ 0

Read-only — does not mutate the ledger.

Usage:
    python -m scripts.audit_stale_active                # whole DB
    python -m scripts.audit_stale_active --cik 1527702  # one CIK, with rows
    python -m scripts.audit_stale_active --top 10       # show top N by hits
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import date as _d
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_conn  # noqa: E402


def _to_date(s):
    if not s or not isinstance(s, str):
        return None
    try:
        return _d.fromisoformat(s[:10])
    except ValueError:
        return None


def _zero_outstanding(typ: str, outstanding: dict) -> bool:
    """True if the row's outstanding numbers indicate nothing is left."""
    if typ in ("convertible", "note"):
        v = outstanding.get("principal_remaining")
        return v is not None and float(v) <= 0
    if typ in ("warrant", "preferred"):
        v = outstanding.get("count")
        return v is not None and float(v) <= 0
    if typ == "equity":
        v = outstanding.get("shares")
        return v is not None and float(v) <= 0
    if typ == "equity_line":
        v = outstanding.get("remaining_capacity_usd")
        return v is not None and float(v) <= 0
    if typ == "shelf":
        cap = outstanding.get("capacity_remaining_usd")
        return cap is not None and float(cap) <= 0
    return False


def classify(row: dict, today: _d) -> list[str]:
    """Return zero or more flag tags for an active row."""
    flags = []
    typ = row["type"]
    terms = json.loads(row["terms_json"] or "{}")
    outstanding = json.loads(row["outstanding_json"] or "{}")

    if typ in ("convertible", "note"):
        m = _to_date(terms.get("maturity"))
        if m and m < today:
            flags.append("matured")

    if typ == "warrant":
        e = _to_date(terms.get("expiration"))
        if e and e < today:
            flags.append("expired")

    if _zero_outstanding(typ, outstanding):
        flags.append("zero")

    return flags


def audit_cik(cik: int, today: _d) -> dict:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT instrument_id, type, label, counterparty,
                      created_at, terms_json, outstanding_json
                 FROM dilution_ledger
                WHERE cik=? AND status='active'
                ORDER BY type, created_at, instrument_id""",
            (cik,),
        ).fetchall()
    flagged = []
    by_flag = Counter()
    by_type = Counter()
    for r in rows:
        d = dict(r)
        tags = classify(d, today)
        if tags:
            d["flags"] = tags
            flagged.append(d)
            for t in tags:
                by_flag[t] += 1
            by_type[d["type"]] += 1
    return {
        "cik": cik,
        "active_total": len(rows),
        "flagged": flagged,
        "by_flag": dict(by_flag),
        "by_type": dict(by_type),
    }


def _fmt_pct(n, d):
    return f"{(100.0 * n / d):4.1f}%" if d else "  —  "


def print_summary(per_cik: list[dict], top: int | None) -> None:
    total_active = sum(c["active_total"] for c in per_cik)
    total_flagged = sum(len(c["flagged"]) for c in per_cik)
    overall = Counter()
    for c in per_cik:
        overall.update(c["by_flag"])

    print(f"=== ledger stale-active audit ===")
    print(f"CIKs scanned:    {len(per_cik)}")
    print(f"active rows:     {total_active}")
    print(f"flagged rows:    {total_flagged}  "
          f"({_fmt_pct(total_flagged, total_active)})")
    print(f"  matured:       {overall.get('matured', 0)}")
    print(f"  expired:       {overall.get('expired', 0)}")
    print(f"  zero:          {overall.get('zero', 0)}")
    print()

    ranked = sorted(per_cik, key=lambda c: len(c["flagged"]), reverse=True)
    if top:
        ranked = ranked[:top]
    print(f"{'cik':>10}  {'active':>6}  {'flagged':>7}  {'%':>5}  "
          f"{'matured':>7}  {'expired':>7}  {'zero':>4}")
    for c in ranked:
        f = len(c["flagged"])
        if not f:
            continue
        print(f"{c['cik']:>10}  {c['active_total']:>6}  {f:>7}  "
              f"{_fmt_pct(f, c['active_total']):>5}  "
              f"{c['by_flag'].get('matured', 0):>7}  "
              f"{c['by_flag'].get('expired', 0):>7}  "
              f"{c['by_flag'].get('zero', 0):>4}")


def print_cik_detail(report: dict) -> None:
    print(f"=== cik={report['cik']} — "
          f"{len(report['flagged'])}/{report['active_total']} "
          f"active rows flagged ===")
    print(f"by flag: {report['by_flag']}")
    print(f"by type: {report['by_type']}")
    print()
    print(f"{'id':<10} {'type':<12} {'created':<10} {'flags':<22} "
          f"{'detail'}")
    for r in report["flagged"]:
        terms = json.loads(r["terms_json"] or "{}")
        outstanding = json.loads(r["outstanding_json"] or "{}")
        bits = []
        if "matured" in r["flags"]:
            bits.append(f"mat={terms.get('maturity')}")
        if "expired" in r["flags"]:
            bits.append(f"exp={terms.get('expiration')}")
        if "zero" in r["flags"]:
            for k in ("principal_remaining", "count", "shares",
                      "remaining_capacity_usd"):
                if k in outstanding:
                    bits.append(f"{k}={outstanding[k]}")
                    break
        print(f"{r['instrument_id']:<10} {r['type']:<12} "
              f"{r['created_at'][:10]:<10} {','.join(r['flags']):<22} "
              f"{' '.join(bits)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cik", type=int,
                    help="audit a single CIK and print every flagged row")
    ap.add_argument("--top", type=int, default=20,
                    help="show top N CIKs by flagged-row count (default 20)")
    args = ap.parse_args()

    today = _d.today()

    if args.cik:
        report = audit_cik(args.cik, today)
        print_cik_detail(report)
        return

    with get_conn() as conn:
        ciks = [r["cik"] for r in conn.execute(
            "SELECT DISTINCT cik FROM dilution_ledger "
            "WHERE status='active' ORDER BY cik"
        ).fetchall()]

    per_cik = [audit_cik(c, today) for c in ciks]
    print_summary(per_cik, top=args.top)


if __name__ == "__main__":
    main()
