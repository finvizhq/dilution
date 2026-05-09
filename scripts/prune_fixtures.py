#!/usr/bin/env python3
"""Drop fixture cards whose key date is before the ticker's filing-coverage start.

Coverage = MIN(filing_date) in dilution_filings for that CIK. The walker
cannot extract events from filings it has never ingested, so any fixture
card dated before that point is unscoreable noise.

Cards with no key date (e.g. legacy preferreds with issue_date=null) are
left in place — those are a separate matcher issue, not a coverage issue.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_conn

DATE_KEY = {
    "warrant": "issue_date",
    "convertible": "issue_date",
    "convertible_preferred": "issue_date",
    "atm": "agreement_start_date",
    "equity_line": "agreement_start_date",
    "shelf": "effect_date",
    "s1_offering": "filing_date",
}

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "evals"


def coverage_start(ticker: str) -> str | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MIN(f.filing_date) AS d "
            "FROM dilution_filings f "
            "JOIN dilution_company c ON c.cik = f.cik "
            "WHERE c.ticker = ?",
            (ticker,),
        ).fetchone()
    return row["d"] if row and row["d"] else None


def prune_fixture(path: Path, dry_run: bool) -> tuple[int, int, list[str]]:
    fixture = json.loads(path.read_text())
    ticker = fixture["ticker"]
    cutoff = coverage_start(ticker)
    if not cutoff:
        return 0, 0, [f"{ticker}: no coverage data, skipping"]

    dropped_log: list[str] = []
    total_before = 0
    total_after = 0

    for category, cards in fixture["cards"].items():
        date_key = DATE_KEY.get(category, "issue_date")
        kept = []
        for card in cards:
            total_before += 1
            d = card.get(date_key) or card.get("issue_date")
            if d and d < cutoff:
                label = card.get("title_contains") or d
                dropped_log.append(
                    f"  [{category}] {label} ({d}) < cutoff {cutoff}"
                )
                continue
            kept.append(card)
            total_after += 1
        fixture["cards"][category] = kept

    if not dry_run and total_before != total_after:
        path.write_text(json.dumps(fixture, indent=2) + "\n")

    return total_before, total_after, [f"{ticker}: cutoff {cutoff}"] + dropped_log


def main():
    dry_run = "--apply" not in sys.argv
    fixtures = sorted(FIXTURE_DIR.glob("*.json"))
    grand_before = grand_after = 0
    for f in fixtures:
        before, after, log = prune_fixture(f, dry_run=dry_run)
        grand_before += before
        grand_after += after
        for line in log:
            print(line)
        print(f"  {before} → {after} cards "
              f"({'-' + str(before - after) if before != after else 'no change'})")
        print()

    mode = "DRY RUN" if dry_run else "APPLIED"
    print(f"=== {mode}: {grand_before} → {grand_after} cards "
          f"({grand_before - grand_after} dropped) ===")
    if dry_run:
        print("Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
