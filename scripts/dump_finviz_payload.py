#!/usr/bin/env python3
"""Dump the Finviz ingest payload (FINVIZ_API_CONTRACT.md) for one or
more tickers — the exact JSON body the push job would PUT, wrapper
included: `{"ticker": ..., "data": {...}}`.

    python scripts/dump_finviz_payload.py GCTK
    python scripts/dump_finviz_payload.py GCTK FCEL KSCP --out examples
    python scripts/dump_finviz_payload.py --all --out examples
    python scripts/dump_finviz_payload.py GCTK --dummy-brief   # placeholder §8
    python scripts/dump_finviz_payload.py GCTK --live          # what's published

Writes examples/finviz_payload_<TICKER>.json (pretty-printed, UTF-8) and
prints a one-line shape summary per ticker. With --stdout, prints the
document instead of writing a file.

`--live` reads the published snapshot back from Finviz (§3.3) instead of
building one, in the same output shape — so the two are directly
comparable, which is how you check what is actually live:

    python scripts/dump_finviz_payload.py CELU --stdout > local.json
    python scripts/dump_finviz_payload.py CELU --live --stdout > live.json
    diff <(jq -S .data local.json) <(jq -S .data live.json)

Needs a walked ledger (dilution.db) plus network: Finviz Elite for
market data and SEC XBRL for the cash / shares-outstanding history.
`--live` needs only FINVIZ_INGEST_TOKEN.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dilution.badges import _sh, _usd                # noqa: E402
from dilution.finviz_payload import (                # noqa: E402
    all_tracked_tickers,
    build_payload,
    payload_summary,
)
from dilution.finviz_push import fetch_snapshot      # noqa: E402


# ── --dummy-brief ────────────────────────────────────────────────────
# Test scaffolding, deliberately kept out of dilution/finviz_payload.py:
# the producer must never invent a brief. This fills the §8 block for
# consumers that need the shape while the real generator is unavailable
# (the brief LLM 503s in streaks) or has only stale prose cached.
#
# The sentences are templated from the snapshot's OWN numbers rather than
# canned filler, so a placeholder brief never contradicts the cards
# rendered beside it — which is the failure mode stale real prose has.
_DUMMY_MARKER = ("Placeholder brief, templated from this snapshot's own "
                 "numbers — real briefs are model-written prose.")


def _dummy_brief(snapshot: dict) -> dict:
    company = snapshot.get("company") or {}
    cash = company.get("cash") or {}
    cards = snapshot.get("cards") or {}
    badges = snapshot.get("badges") or {}
    overall = badges.get("overall") or {}
    os_chart = company.get("os_chart") or {}
    outstanding = company.get("shares_outstanding")

    months = cash.get("months_of_cash")
    runway = f"{months:.1f} months" if months is not None else "unknown"
    raisable = sum(c["current_raisable_amount"] for c in cards.get("shelf") or []
                   if c.get("current_raisable_amount"))
    unlimited = any(c.get("unlimited") for c in cards.get("shelf") or [])
    at_will = sum(c["remaining_capacity"]
                  for group in ("atm", "equity_line")
                  for c in cards.get(group) or []
                  if c.get("remaining_capacity") and not c.get("terminated"))
    overhang = sum(s["shares"] for s in os_chart.get("fd_stack") or []
                   if not s.get("price_based") and s.get("shares"))

    # op_cf is signed (negative = burn); the sentence supplies the sign in
    # words, so pass the magnitude or it reads "burn of -$30.6M".
    op_cf = cash.get("op_cf_quarterly_usd")
    burn = _usd(abs(op_cf)) if op_cf is not None else "—"
    clauses = []
    if unlimited:
        clauses.append("an unlimited (WKSI) shelf is on file")
    elif raisable:
        clauses.append(f"{_usd(raisable)} of shelf capacity is raisable "
                       f"today"
                       + (" after the baby-shelf I.B.6 cap"
                          if company.get("is_baby_shelf_restricted")
                          else ""))
    if at_will:
        clauses.append(f"{_usd(at_will)} of at-will ATM / equity-line "
                       f"capacity is live")
    if overhang:
        pct = (f" ({overhang / outstanding * 100:.0f}% of O/S)"
               if outstanding else "")
        clauses.append(f"{_sh(overhang)} potential shares{pct} hang over "
                       f"the stock")
    takeaway = (f" — composite dilution-risk is {overall['score']}/100 "
                f"({overall.get('label')})"
                if overall.get("score") is not None else "")
    summary = (
        f"With estimated cash of {_usd(cash.get('current_cash_est_usd'))} "
        f"against a quarterly operating "
        f"{'burn' if (op_cf or 0) < 0 else 'inflow'} of {burn}, "
        f"{snapshot.get('ticker')}'s runway is {runway}"
        + "".join(f", and {c}" for c in clauses)
        + f"{takeaway}. [{_DUMMY_MARKER}]"
    )

    return {
        "summary": summary,
        "generated_at": snapshot.get("generated_at"),
    }


# `_all_tickers` / `_summary` moved to dilution.finviz_payload as
# `all_tracked_tickers` / `payload_summary` so scripts/push_finviz.py
# shares one definition of "the universe" and one output format.


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="*", help="tickers to dump")
    ap.add_argument("--all", action="store_true",
                    help="every tracked ticker in the ledger")
    ap.add_argument("--out", default="examples",
                    help="output directory (default: examples)")
    ap.add_argument("--stdout", action="store_true",
                    help="print the document instead of writing a file")
    ap.add_argument("--dummy-brief", action="store_true",
                    help="replace the §8 brief with a placeholder templated "
                         "from the snapshot itself (testing aid — the real "
                         "generator is an LLM call and 503s in streaks); "
                         "omit it for tickers whose cached prose is current")
    ap.add_argument("--live", action="store_true",
                    help="show the snapshot Finviz currently holds (read-back "
                         "GET, §3.3) instead of building one locally — the "
                         "way to answer 'what is actually published'")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s")

    if args.live and args.dummy_brief:
        ap.error("--live shows what is published; --dummy-brief rewrites a "
                 "local build. Pick one.")

    tickers = [t.upper() for t in args.tickers]
    if args.all:
        tickers = all_tracked_tickers()
    if not tickers:
        ap.error("pass one or more tickers, or --all")

    out_dir = Path(args.out)
    if not args.stdout:
        out_dir.mkdir(parents=True, exist_ok=True)

    failed = 0
    for ticker in tickers:
        try:
            if args.live:
                # §3.3 hands back the inner `data` unwrapped, so wrap it to
                # keep this script's output shape identical either way —
                # that is what makes build-vs-live diffable.
                snapshot = fetch_snapshot(ticker)
                if snapshot is None:
                    print(f"{ticker}: not published", file=sys.stderr)
                    continue
                doc = {"ticker": ticker, "data": snapshot}
            else:
                doc = build_payload(ticker)
        except Exception as exc:
            failed += 1
            print(f"{ticker}: FAILED — {exc}", file=sys.stderr)
            continue
        if args.dummy_brief:
            # Unconditional: old cached prose can contradict the cards
            # (GCTK's June prose claimed no warrants with ten cards live).
            # Dump without the flag for tickers whose cached prose you
            # know is current.
            doc["data"]["brief"] = _dummy_brief(doc["data"])
        text = json.dumps(doc, indent=2, ensure_ascii=False)
        if args.stdout:
            print(text)
            continue
        suffix = "_live" if args.live else ""
        path = out_dir / f"finviz_payload_{ticker}{suffix}.json"
        path.write_text(text + "\n", encoding="utf-8")
        print(f"{ticker}: {len(text):>8,} bytes → {path}  "
              f"{payload_summary(doc)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
