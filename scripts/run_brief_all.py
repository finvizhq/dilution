"""Generate / refresh the AI dilution brief for every tracked ticker.

Usage:
    python3 scripts/run_brief_all.py              # all stale/missing briefs
    python3 scripts/run_brief_all.py --force      # regenerate everything
    python3 scripts/run_brief_all.py AACG CETY    # just these tickers

Skip rule (without --force): a ticker is skipped when its cached brief
is newer than its latest filing — the same staleness test the dashboard
shows, so a nightly run only pays for tickers that actually changed.

Sequential by design: each ticker is one facts build (cached fetchers —
XBRL / finviz may go to network on a cold cache) plus one LLM call on
the flex tier. 65 tickers ≈ 15-25 min cold, much less warm. Failures
are logged and skipped — one bad ticker doesn't abort the batch.

Results land in dilution_ticker_brief (read by the dashboard panel) —
nothing else is written.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402  (loads .env)
from db import get_conn  # noqa: E402
from dilution import ticker_brief  # noqa: E402
from dilution.badges import compute_badges  # noqa: E402
from dilution.capital_raised import capital_raised_since  # noqa: E402
from dilution.cash_history import fetch_cash_history_cached  # noqa: E402
from dilution.finviz_client import fundamentals as finviz_fundamentals  # noqa: E402
from dilution.ledger.cards import (  # noqa: E402
    atm_cards, convertible_note_cards, equity_line_cards, preferred_cards,
    s1_offering_cards, shelf_cards, warrant_cards,
)
from dilution.llm_provider import require_api_key  # noqa: E402
from dilution.share_counts import fetch_implied_outstanding_cached  # noqa: E402


def _is_fresh(cik: int) -> bool:
    """True when the cached brief postdates the ticker's latest filing
    — the dashboard's staleness rule, inverted."""
    cached = ticker_brief.get_cached(cik)
    if not cached:
        return False
    with get_conn() as conn:
        latest = conn.execute(
            "SELECT MAX(filing_date) d FROM dilution_filings WHERE cik = ?",
            (cik,),
        ).fetchone()["d"]
    return not (latest and latest > cached["generated_at"][:10])


def _generate_one(cik: int, ticker: str, name: str) -> None:
    fund = finviz_fundamentals(ticker)
    implied = fetch_implied_outstanding_cached(cik)
    latest_os = (implied.total if implied.total is not None
                 else (fund or {}).get("shares_outstanding"))
    cards = {
        "s1_offering": s1_offering_cards(cik),
        "warrant": warrant_cards(cik),
        "convertible": convertible_note_cards(cik),
        "convertible_preferred": preferred_cards(cik),
        "atm": atm_cards(cik, fund, latest_os),
        "equity_line": equity_line_cards(cik),
        "shelf": shelf_cards(cik, fund, latest_os),
    }
    probe = fetch_cash_history_cached(cik)
    raised = (capital_raised_since(cik, probe.latest_period_end)
              if probe.latest_period_end else None)
    cash = fetch_cash_history_cached(cik, capital_raised_usd=raised)
    badges = compute_badges(cik, fund=fund, latest_os=latest_os,
                            cards=cards, cash=cash)
    facts = ticker_brief.build_facts(
        ticker=ticker, name=name, fund=fund, latest_os=latest_os,
        cards=cards, cash=cash, raised=raised, badges=badges)
    brief = ticker_brief.generate(cik, ticker, facts)
    print(f"  {brief['headline']}")


def main() -> int:
    force = "--force" in sys.argv
    only = {a.upper() for a in sys.argv[1:] if not a.startswith("-")}
    require_api_key()

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT cik, ticker, name FROM dilution_company ORDER BY ticker",
        ).fetchall()
    if only:
        rows = [r for r in rows if r["ticker"] in only]
        missing = only - {r["ticker"] for r in rows}
        if missing:
            print(f"not tracked, skipping: {', '.join(sorted(missing))}")

    done = skipped = failed = 0
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        tag = f"[{i}/{len(rows)}] {r['ticker']}"
        if not force and _is_fresh(r["cik"]):
            print(f"{tag}: fresh — skipped")
            skipped += 1
            continue
        print(f"{tag}: generating…")
        try:
            _generate_one(r["cik"], r["ticker"], r["name"])
            done += 1
        except Exception as e:  # keep the batch going
            print(f"  FAILED: {e}")
            failed += 1
    print(f"\n{done} generated, {skipped} fresh-skipped, {failed} failed "
          f"in {time.time() - t0:.0f}s "
          f"(model={config.LLM_MODEL})")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
