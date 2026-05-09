#!/usr/bin/env python3
"""Compare actual cards against the eval fixture for a ticker.

    python run_eval.py XTLB

Fixtures live in evals/<TICKER>.json. Each fixture lists the expected
cards (per category) with the fields that should match. Matching is
keyed on `issue_date` (or `agreement_start_date` / `effect_date` /
`filing_date` depending on the category).

Comparison is field-level. Numeric fields use a small relative
tolerance; strings are case-insensitive substring; lists (known_owners,
underwriters) use fuzzy substring containment so the LLM's longer form
("Hudson Bay Capital Management") still matches the screenshot's short
form ("Hudson Bay").
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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


# Per-category date key used to match expected ↔ actual cards.
DATE_KEY = {
    "warrant": "issue_date",
    "convertible": "issue_date",
    "convertible_preferred": "issue_date",
    "atm": "agreement_start_date",
    "equity_line": "agreement_start_date",
    "shelf": "effect_date",
    "s1_offering": "filing_date",
}

# Numeric tolerance — relative. 0.1% covers float-rounding noise (e.g.
# 1,777.78 vs 1,778) without papering over real value differences.
NUMERIC_TOLERANCE = 0.001


def _resolve_cik(ticker: str) -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT cik FROM dilution_company WHERE ticker = ?",
            (ticker,),
        ).fetchone()
    if not row:
        sys.exit(f"ticker {ticker} not in dilution_company — "
                 f"run `python run_dilution.py {ticker}` first")
    return row["cik"]


def _build_actual(cik: int) -> dict:
    fund = finviz_fundamentals(_ticker_for(cik))
    latest_os = (fund or {}).get("shares_outstanding")
    return {
        "warrant": warrant_cards(cik),
        "convertible": convertible_note_cards(cik),
        "convertible_preferred": preferred_cards(cik),
        "atm": atm_cards(cik, fund, latest_os),
        "equity_line": equity_line_cards(cik),
        "shelf": shelf_cards(cik, fund, latest_os),
        "s1_offering": s1_offering_cards(cik),
    }


def _ticker_for(cik: int) -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT ticker FROM dilution_company WHERE cik = ?", (cik,),
        ).fetchone()
    return row["ticker"] if row else ""


def _close_num(a, b, tol=NUMERIC_TOLERANCE) -> bool:
    if a is None or b is None:
        return a == b
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if a == 0 and b == 0:
        return True
    return abs(a - b) / max(abs(a), abs(b)) <= tol


def _str_contains(actual, expected) -> bool:
    """Case-insensitive substring: expected ⊆ actual (or vice versa).
    Punctuation is squashed so banker names like 'H.C. Wainwright' and
    'HC Wainwright' compare equal — DT and SEC filings use different
    conventions and we don't maintain an alias map."""
    if not expected:
        return not actual
    if not actual:
        return False
    def _norm(s: str) -> str:
        return "".join(c for c in s.lower() if c.isalnum() or c.isspace())
    a, e = _norm(str(actual)), _norm(str(expected))
    return e in a or a in e


def _list_subset(actual_list, expected_list) -> tuple[bool, list]:
    """Every expected name should fuzzy-match some actual name."""
    if not expected_list:
        return True, []
    actual_str = ", ".join(str(x) for x in (actual_list or [])).lower()
    missing = [e for e in expected_list if e.lower() not in actual_str]
    return (not missing), missing


_DATE_TOLERANCE_DAYS = 14

# Field-level date comparison tolerance. Fixtures use one date convention
# (typically the SEC effective / cover-page date); the walker emits another
# (the contractual closing / agreement-signing date). Both are defensible —
# the gap is 1–6 days in practice. We accept any date within ±7 days as a
# match; misses outside the window are real disagreements.
_DATE_FIELD_TOLERANCE_DAYS = 7

# Fields whose values are YYYY-MM-DD strings and should be compared with
# ±_DATE_FIELD_TOLERANCE_DAYS slack rather than substring equality.
_DATE_FIELDS = frozenset({
    "exercisable_date", "expiration_date", "convertible_date",
    "maturity_date", "agreement_start_date", "agreement_end_date",
    "effect_date", "last_update_date", "filing_date",
})


def _date_diff_days(a: str | None, b: str | None) -> int | None:
    if not a or not b:
        return None
    try:
        from datetime import date as _date
        return abs((_date.fromisoformat(a[:10])
                    - _date.fromisoformat(b[:10])).days)
    except (ValueError, TypeError):
        return None


def _match_card(expected: dict, actual_list: list[dict],
                date_key: str) -> dict | None:
    """Match expected card to an actual card by date proximity.
    DT uses closing date, system often uses pricing/agreement date —
    same instrument, dates 1–7 days apart. Tolerance is 14 days. Strike
    disambiguates same-month tranches (placement-agent comp warrants)."""
    target = expected.get("issue_date") or expected.get(date_key)
    if not target:
        return None

    def _date_score(card: dict) -> int | None:
        diffs = [d for d in (
            _date_diff_days(card.get(date_key), target),
            _date_diff_days(card.get("issue_date"), target),
        ) if d is not None]
        if not diffs:
            return None
        m = min(diffs)
        return m if m <= _DATE_TOLERANCE_DAYS else None

    scored = [(c, s) for c in actual_list
              if (s := _date_score(c)) is not None]
    if not scored:
        return None

    exp_strike = (expected.get("exercise_price")
                  or expected.get("conversion_price"))
    if exp_strike is not None and len(scored) > 1:
        # Rank by absolute strike distance so a candidate with strike $4.80
        # beats one with strike None when expected is $4.00. None sinks last.
        def _strike_dist(card: dict) -> float:
            s = card.get("exercise_price") or card.get("conversion_price")
            if s is None:
                return float("inf")
            try:
                return abs(float(s) - float(exp_strike))
            except (TypeError, ValueError):
                return float("inf")
        scored.sort(key=lambda cs: (_strike_dist(cs[0]), cs[1]))
        return scored[0][0]
    scored.sort(key=lambda cs: cs[1])
    return scored[0][0]


def _compare_card(expected: dict, actual: dict) -> list[str]:
    diffs = []
    for field, exp_val in expected.items():
        if field in ("title_contains", "issue_date"):
            if field == "title_contains":
                if exp_val.lower() not in (actual.get("title") or "").lower():
                    diffs.append(
                        f"title: expected substring '{exp_val}', got '{actual.get('title')}'"
                    )
            continue
        act_val = actual.get(field)
        if isinstance(exp_val, (int, float)) and not isinstance(exp_val, bool):
            if not _close_num(act_val, exp_val):
                diffs.append(f"{field}: expected {exp_val!r}, got {act_val!r}")
        elif isinstance(exp_val, list):
            ok, missing = _list_subset(act_val, exp_val)
            if not ok:
                diffs.append(
                    f"{field}: expected to contain {missing}, got {act_val!r}"
                )
        elif isinstance(exp_val, str):
            if field in _DATE_FIELDS:
                d = _date_diff_days(act_val, exp_val)
                if d is None or d > _DATE_FIELD_TOLERANCE_DAYS:
                    diffs.append(f"{field}: expected ~'{exp_val}', got {act_val!r}")
            elif not _str_contains(act_val, exp_val):
                diffs.append(f"{field}: expected ~'{exp_val}', got {act_val!r}")
        else:
            if act_val != exp_val:
                diffs.append(f"{field}: expected {exp_val!r}, got {act_val!r}")
    return diffs


def evaluate(ticker: str) -> int:
    fixture_path = Path(__file__).parent / "evals" / f"{ticker}.json"
    if not fixture_path.exists():
        sys.exit(f"no fixture at {fixture_path}")
    fixture = json.loads(fixture_path.read_text())

    cik = _resolve_cik(ticker)
    actual = _build_actual(cik)

    total_expected = 0
    total_found = 0          # card located by date+strike
    total_exact = 0          # found AND every field matches
    total_field_checks = 0
    total_field_passes = 0
    pass_lines, fail_lines, missing_lines, extra_lines = [], [], [], []

    for category, expected_cards in fixture["cards"].items():
        actual_cards = actual.get(category, [])
        date_key = DATE_KEY.get(category, "issue_date")
        seen_actual = set()
        for exp in expected_cards:
            total_expected += 1
            act = _match_card(exp, actual_cards, date_key)
            if act is None:
                missing_lines.append(
                    f"  [{category}] {exp.get('title_contains') or exp.get('issue_date')}: NO MATCH"
                )
                continue
            seen_actual.add(id(act))
            total_found += 1
            diffs = _compare_card(exp, act)
            # Count fields: every comparable expected key
            n_fields = sum(1 for k in exp.keys()
                           if k not in ("title_contains", "issue_date"))
            total_field_checks += n_fields
            total_field_passes += n_fields - len(diffs)
            label = exp.get("title_contains") or exp.get("issue_date")
            if not diffs:
                total_exact += 1
                pass_lines.append(f"  [{category}] {label}: ✓ all {n_fields} fields exact")
            else:
                fail_lines.append(
                    f"  [{category}] {label}: {n_fields - len(diffs)}/{n_fields} fields exact"
                )
                for d in diffs:
                    fail_lines.append(f"      ✗ {d}")
        for c in actual_cards:
            if id(c) not in seen_actual:
                extra_lines.append(
                    f"  [{category}] {c.get('title')} ({c.get(date_key)}): unexpected actual card"
                )

    print(f"=== Eval: {ticker} (cik {cik}) ===")
    print(f"  cards exact     : {total_exact}/{total_expected}")
    print(f"  cards found     : {total_found}/{total_expected}")
    print(f"  fields exact    : {total_field_passes}/{total_field_checks}"
          + (f"  ({total_field_passes/total_field_checks:.0%})"
             if total_field_checks else ""))
    print(f"  unexpected extra: {len(extra_lines)}")
    if pass_lines:
        print("\nPASS")
        for l in pass_lines:
            print(l)
    if fail_lines:
        print("\nFIELD MISMATCHES (exact-value comparison)")
        for l in fail_lines:
            print(l)
    if missing_lines:
        print("\nMISSING (in fixture, not in actual)")
        for l in missing_lines:
            print(l)
    if extra_lines:
        print("\nEXTRA (in actual, not in fixture)")
        for l in extra_lines:
            print(l)

    return 0 if (total_exact == total_expected and not extra_lines) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    args = ap.parse_args()
    sys.exit(evaluate(args.ticker.upper()))


if __name__ == "__main__":
    main()
