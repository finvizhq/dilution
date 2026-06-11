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
import re
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
        raise LookupError(
            f"ticker {ticker} not in dilution_company — "
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


# Corporate-RENAME aliases. The matcher deliberately maintains no general
# alias map (punctuation squashing bridges format differences), but a
# rename can't be bridged by any string transform: SVB Securities LLC
# became Leerink Partners LLC mid-2023, so a fixture transcribed from a
# 2023-era DT screenshot says 'SVB' while the walker reads 'Leerink
# Partners' from the current filing — same bank, both names correct for
# their date. Canonicalize before comparing. Longest key first; keep this
# list to VERIFIED renames only (DB-wide only CGEN carries this pair).
_RENAME_ALIASES = [
    ("svb securities", "leerink partners"),
    ("svb leerink", "leerink partners"),
    ("svb", "leerink partners"),
]


def _alias(s: str) -> str:
    """Apply whole-token rename aliases to an already-lowercased string."""
    for old, new in _RENAME_ALIASES:
        s = re.sub(rf"\b{old}\b", new, s)
    return s


def _str_contains(actual, expected) -> bool:
    """Case-insensitive substring: expected ⊆ actual (or vice versa).
    Punctuation is squashed so banker names like 'H.C. Wainwright' and
    'HC Wainwright' compare equal — DT and SEC filings use different
    conventions and we don't maintain an alias map (corporate RENAMES
    are the one exception, see _RENAME_ALIASES)."""
    if not expected:
        return not actual
    if not actual:
        return False
    def _norm(s: str) -> str:
        return "".join(c for c in s.lower() if c.isalnum() or c.isspace())
    a, e = _alias(_norm(str(actual))), _alias(_norm(str(expected)))
    return e in a or a in e


def _list_subset(actual_list, expected_list) -> tuple[bool, list]:
    """Every expected name should fuzzy-match some actual name.

    Substring match first; for short all-caps expected entries (e.g.
    'EIB'), also try acronym match against the word initials of each
    actual entry ('European Investment Bank' → 'EIB'). This bridges
    the gap between fixture short-forms and the walker's canonical
    long-form entity names without maintaining an alias map."""
    if not expected_list:
        return True, []
    actuals = [str(x) for x in (actual_list or [])]
    actual_str = ", ".join(actuals).lower()
    missing = []
    for e in expected_list:
        el = e.lower()
        if el in actual_str:
            continue
        if e.isupper() and 2 <= len(e) <= 4 and any(
                _initials(a) == e for a in actuals):
            continue
        # Truncation: the walker stores a clipped entity name ('M2B' for
        # 'M2B Funding Corp'). Accept when an actual token (≥3 chars) is a
        # leading WHOLE-WORD prefix of the expected name — the trailing
        # space gate means 'M2B' matches 'M2B Funding' but 'Can' never
        # matches 'Cantor'. The ≥3 gate keeps a stray 2-char token from
        # binding an unrelated long name.
        if any(len(a) >= 3 and el.startswith(a.lower() + " ")
               for a in actuals):
            continue
        missing.append(e)
    return (not missing), missing


def _initials(s: str) -> str:
    """Uppercase first-letter-of-each-word, alphabetic only.
    'European Investment Bank' → 'EIB'; 'H.C. Wainwright' → 'HW'."""
    return "".join(w[0].upper() for w in s.split() if w and w[0].isalpha())


def _acronym_match(actual, expected) -> bool:
    """Scalar-string analogue of the `_list_subset` acronym rule: a
    short all-caps expected token (e.g. 'AGP') matches a long-form
    actual whose word-initials spell it ('Alliance Global Partners' →
    'AGP'). Banker fields carry the walker's canonical long-form while
    fixtures use the trading short-form; this bridges them without an
    alias map. Mirrors the 2–4-char uppercase guard in `_list_subset`."""
    if not actual or not expected:
        return False
    e = str(expected)
    return (e.isupper() and 2 <= len(e) <= 4
            and _initials(str(actual)) == e)


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


# Baby-shelf I.B.6 fields are computed as-of TODAY — a rolling trailing-12-
# month drawdown window, the live 60-day-high price, and the live float — but
# the fixtures are frozen point-in-time DT snapshots with no `as_of`. They
# therefore drift out of tolerance purely with the passage of time and price
# movement, not because of any correctness defect in the window math. Treat
# them as informational (excluded from scoring) for baby-shelf-restricted
# shelves. Scoped to actual cards with baby_shelf_restriction=='Yes' so
# non-baby shelves (CGEN, FCEL, XTIA) keep grading current_raisable_amount,
# where a divergence is a genuine extraction/attribution bug.
_SNAPSHOT_RELATIVE_BABY_SHELF_FIELDS = frozenset({
    "raised_last_12mo_under_ib6",
    "current_raisable_amount",
    "ib6_float_value",
})


def _exempt_fields(category: str, actual: dict) -> set:
    """Expected fields to skip scoring for this matched card (see
    _SNAPSHOT_RELATIVE_BABY_SHELF_FIELDS)."""
    if (category == "shelf"
            and str(actual.get("baby_shelf_restriction") or "").lower() == "yes"):
        return set(_SNAPSHOT_RELATIVE_BABY_SHELF_FIELDS)
    return set()


_SERIES_RE = re.compile(r"\bseries\s+([a-z0-9]+)\b", re.IGNORECASE)


def _series_token(text: str | None) -> str | None:
    """Lowercased 'series X' if `text` names one, else None.

    Used as a hard filter in `_match_card`: when a fixture asks for
    'Series B', a same-date 'Inducement' or 'Pre-Funded' actual is
    a different instrument that happens to share a date, not a field
    discrepancy — refuse the bind rather than report misleading
    field mismatches."""
    if not text:
        return None
    m = _SERIES_RE.search(text)
    return f"series {m.group(1).lower()}" if m else None


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
                date_key: str,
                excluded: set | None = None) -> dict | None:
    """Match expected card to an actual card by date proximity.
    DT uses closing date, system often uses pricing/agreement date —
    same instrument, dates 1–7 days apart. Tolerance is 14 days.

    When multiple same-day candidates remain, rank by:
      1. title_contains substring (most discriminating — e.g. "Series A"
         vs "Series B" when actual strikes are both NULL),
      2. strike distance (None sinks last via inf),
      3. date proximity.

    `excluded` carries the id()s of actual cards already bound to a
    prior expected card in this category, so two fixtures with the
    same date can't collapse onto a single actual."""
    excluded = excluded or set()
    target = expected.get("issue_date") or expected.get(date_key)
    exp_series = _series_token(expected.get("title_contains"))

    def _series_ok(card: dict) -> bool:
        # When the fixture names a series, the actual must name the
        # same one — otherwise we'd bind "Series B" to a same-date
        # "Series A" / "Inducement" / "Pre-Funded" tranche on strike
        # proximity alone.
        if not exp_series:
            return True
        return _series_token(card.get("title")) == exp_series

    if not target:
        # Pending-effect shelves (and similar null-date fixtures)
        # can't match by date proximity. Fall back to title_contains
        # substring matching against same-null-date actuals — this is
        # the only signal available, and it's enough to distinguish
        # "May 2026" from "March 2023" on the shelf surface.
        exp_title = expected.get("title_contains") or ""
        if not exp_title:
            return None
        for c in actual_list:
            if id(c) in excluded:
                continue
            if c.get(date_key):
                # Skip actuals that DO have a date — the fixture
                # explicitly wants the null-date one.
                continue
            if not _series_ok(c):
                continue
            if _str_contains(c.get("title"), exp_title):
                return c
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
              if id(c) not in excluded
              and _series_ok(c)
              and (s := _date_score(c)) is not None]
    if not scored:
        return None

    # A snapshot-exempt strike is a FROZEN DT market value (ratcheted /
    # discount-to-market), not a stable identity key — the comparator
    # already exempts it from scoring, so the matcher must not rank on
    # it either. Ranking on it paired CETY's July-2025 FirstFire fixture
    # to a same-month 1800 Diagonal note 12 days away while the 0-day
    # FirstFire card sat in extras.
    _snap = set(expected.get("snapshot_fields") or [])
    exp_strike = None
    if not ({"exercise_price", "conversion_price"} & _snap):
        exp_strike = (expected.get("exercise_price")
                      or expected.get("conversion_price"))
    exp_title = expected.get("title_contains") or ""
    exp_owners = [str(o) for o in (expected.get("known_owners") or [])]

    def _title_dist(card: dict) -> int:
        if not exp_title:
            return 0
        return 0 if _str_contains(card.get("title"), exp_title) else 1

    def _owner_dist(card: dict) -> int:
        # Counterparty beats any price-derived signal for breaking a
        # same-month tie. Same subset semantics as the downstream
        # known_owners comparison so the matcher never prefers a card
        # the comparator would then fail.
        if not exp_owners:
            return 0
        ok, _ = _list_subset(card.get("known_owners"), exp_owners)
        return 0 if ok else 1

    def _strike_dist(card: dict) -> float:
        if exp_strike is None:
            return 0.0
        s = card.get("exercise_price") or card.get("conversion_price")
        if s is None:
            return float("inf")
        try:
            return abs(float(s) - float(exp_strike))
        except (TypeError, ValueError):
            return float("inf")

    scored.sort(key=lambda cs: (
        _title_dist(cs[0]), _owner_dist(cs[0]), _strike_dist(cs[0]), cs[1]))
    return scored[0][0]


def _compare_card(expected: dict, actual: dict) -> list[str]:
    diffs = []
    for field, exp_val in expected.items():
        if field in ("title_contains", "issue_date"):
            if field == "title_contains":
                # Token-subset, not contiguous substring: the card is already
                # bound by date/series/strike before we get here, so the title
                # is a secondary check. A benign inserted qualifier ('Common'
                # in 'October 2022 Common Warrants') shouldn't fail a fixture
                # asking for 'October 2022 Warrants' when every token is
                # present. Any value that passed the old substring test still
                # passes (a contiguous substring's tokens are all present).
                act_title = _alias((actual.get("title") or "").lower())
                if not set(_alias(exp_val.lower()).split()).issubset(
                        act_title.split()):
                    diffs.append(
                        f"title: expected tokens of '{exp_val}', got '{actual.get('title')}'"
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
            elif (not _str_contains(act_val, exp_val)
                  and not _acronym_match(act_val, exp_val)):
                diffs.append(f"{field}: expected ~'{exp_val}', got {act_val!r}")
        else:
            if act_val != exp_val:
                diffs.append(f"{field}: expected {exp_val!r}, got {act_val!r}")
    return diffs


def score(ticker: str) -> dict:
    """Score one fixture against the live DB. Pure — no printing.

    Returns a metrics dict: the headline counts plus the human-readable
    detail-line lists that `evaluate()` prints. Raises FileNotFoundError
    when no fixture exists and LookupError when the ticker has not been
    walked into the DB yet, so a batch runner can mark those tickers
    "not scored" instead of crashing the whole sweep."""
    fixture_path = Path(__file__).parent / "evals" / f"{ticker}.json"
    if not fixture_path.exists():
        raise FileNotFoundError(f"no fixture at {fixture_path}")
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
            act = _match_card(exp, actual_cards, date_key,
                              excluded=seen_actual)
            if act is None:
                missing_lines.append(
                    f"  [{category}] {exp.get('title_contains') or exp.get('issue_date')}: NO MATCH"
                )
                continue
            seen_actual.add(id(act))
            total_found += 1
            # Drop snapshot-relative baby-shelf fields from BOTH the diff
            # check and the field count, so they neither pass nor fail.
            # A fixture card can also self-declare snapshot-relative fields
            # via "snapshot_fields": [...] — used for values DT derives from
            # market data at ITS OWN card-update date (variable-rate
            # convertible effective prices, post-ratchet warrant strikes):
            # no filing-derived computation can match a frozen market
            # snapshot, so the value is recorded for documentation but not
            # scored (round-3 C1 decision, 2026-06-05).
            exempt = _exempt_fields(category, act)
            exempt |= set(exp.get("snapshot_fields") or [])
            exempt.add("snapshot_fields")
            exp = {k: v for k, v in exp.items() if k not in exempt}
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

    return {
        "ticker": ticker,
        "cik": cik,
        "expected": total_expected,
        "found": total_found,
        "exact": total_exact,
        "field_checks": total_field_checks,
        "field_passes": total_field_passes,
        "extra": len(extra_lines),
        "missing": len(missing_lines),
        "passed": total_exact == total_expected and not extra_lines,
        "pass_lines": pass_lines,
        "fail_lines": fail_lines,
        "missing_lines": missing_lines,
        "extra_lines": extra_lines,
    }


def evaluate(ticker: str) -> int:
    """Score `ticker` and print the per-card report. Returns a process
    exit code (0 = perfect, 1 = any miss/extra)."""
    m = score(ticker)
    fc, fp = m["field_checks"], m["field_passes"]
    print(f"=== Eval: {ticker} (cik {m['cik']}) ===")
    print(f"  cards exact     : {m['exact']}/{m['expected']}")
    print(f"  cards found     : {m['found']}/{m['expected']}")
    print(f"  fields exact    : {fp}/{fc}"
          + (f"  ({fp/fc:.0%})" if fc else ""))
    print(f"  unexpected extra: {m['extra']}")
    if m["pass_lines"]:
        print("\nPASS")
        for l in m["pass_lines"]:
            print(l)
    if m["fail_lines"]:
        print("\nFIELD MISMATCHES (exact-value comparison)")
        for l in m["fail_lines"]:
            print(l)
    if m["missing_lines"]:
        print("\nMISSING (in fixture, not in actual)")
        for l in m["missing_lines"]:
            print(l)
    if m["extra_lines"]:
        print("\nEXTRA (in actual, not in fixture)")
        for l in m["extra_lines"]:
            print(l)
    return 0 if m["passed"] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    args = ap.parse_args()
    try:
        sys.exit(evaluate(args.ticker.upper()))
    except (FileNotFoundError, LookupError) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
