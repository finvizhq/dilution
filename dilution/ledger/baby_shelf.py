"""Baby-shelf math.

Reads from `dilution_ledger_drawdowns` (populated by every
record_event(drawdown)).

What we compute deterministically from SEC data alone:
  - raised_under_ib6_last_12mo  (sum of drawdowns against shelf
                                 instruments + ATM sales — anything
                                 that registered as a primary cash
                                 raise off an S-3/F-3)
  - baby_shelf_threshold_price  (parameterized by float)

What needs an external price feed (caller passes in):
  - ib6_max_raise(float, price)
  - ib6_remaining(cik, float, price)

Eligibility: I.B.6 only attaches to issuers with an effective S-3/F-3
on file. Without one, the rule is moot — return eligible=False.
"""

from __future__ import annotations

from datetime import date as _d, timedelta

from db import get_conn

BABY_SHELF_FLOAT_VALUE_THRESHOLD_USD = 75_000_000

# Forms whose effective registration triggers I.B.6 / I.B.5.
IB6_ELIGIBLE_FORM_PREFIXES = ("S-3", "F-3")


def has_eligible_shelf(cik: int, today: _d | None = None) -> bool:
    """True when the issuer has an S-3 / S-3ASR / F-3 / F-3ASR shelf
    in `effective` or `active` derived status."""
    from .shelf_status import derive_shelf_status

    today = today or _d.today()
    eligible_statuses = ("effective", "active")
    for s in derive_shelf_status(cik, today=today):
        form = (s.get("form") or "").upper()
        if not any(form.startswith(p) for p in IB6_ELIGIBLE_FORM_PREFIXES):
            continue
        if s.get("derived_status") in eligible_statuses:
            return True
    return False


def raised_under_ib6_last_12mo(cik: int,
                               today: _d | None = None) -> dict:
    """Sum gross proceeds from primary registered cash raises in the
    rolling 12-month window. Source: dilution_ledger_drawdowns,
    filtered to drawdowns against shelf or ATM ledger instruments
    (the ATM lives under a shelf — we double-count ATM and shelf
    drawdowns separately because the indexer logs both as drawdowns
    on their respective ids; dedupe by accession). Equity-line
    drawdowns don't count toward IB6 since equity lines are NOT
    primary registered offerings under I.B.6.
    """
    today = today or _d.today()
    cutoff = (today - timedelta(days=365)).isoformat()
    today_iso = today.isoformat()
    if not has_eligible_shelf(cik, today=today):
        return {
            "as_of": today_iso, "window_start": cutoff,
            "total": 0.0, "rows": [], "eligible": False,
        }
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT d.accession_number, d.event_date, d.amount_usd,
                      d.instrument_id, d.shares, d.price,
                      l.type, l.counterparty
                 FROM dilution_ledger_drawdowns d
                 JOIN dilution_ledger l
                   ON l.instrument_id = d.instrument_id
                WHERE d.cik = ?
                  AND d.event_date >= ?
                  AND d.event_date <= ?
                  AND l.type IN ('shelf', 'atm')
                ORDER BY d.event_date""",
            (cik, cutoff, today_iso),
        ).fetchall()
    # Dedupe: an offering can register one drawdown against the shelf
    # (e.g. SH-001) and another against the ATM (ATM-001) for the same
    # accession. Aggregate by accession + amount within tolerance.
    seen: list[tuple[str, float, str]] = []
    contributing = []
    total = 0.0
    for r in rows:
        amount = r["amount_usd"]
        if not amount or amount <= 0:
            continue
        acc = r["accession_number"]
        date = r["event_date"]
        is_dup = False
        for prev_acc, prev_amt, prev_date in seen:
            if prev_acc != acc:
                continue
            denom = max(abs(amount), abs(prev_amt))
            if denom == 0 or abs(amount - prev_amt) / denom <= 0.05:
                is_dup = True
                break
        if is_dup:
            continue
        seen.append((acc, amount, date))
        total += amount
        contributing.append({
            "date": date,
            "instrument_id": r["instrument_id"],
            "type": r["type"],
            "proceeds": amount,
            "counterparty": r["counterparty"],
            "accession": acc,
        })
    return {
        "as_of": today_iso, "window_start": cutoff,
        "total": total, "rows": contributing, "eligible": True,
    }


def baby_shelf_threshold_price(float_shares: float | None) -> float | None:
    """Price at which `float_shares × price` exceeds $75M, removing the
    issuer from baby-shelf restriction."""
    if not float_shares or float_shares <= 0:
        return None
    return BABY_SHELF_FLOAT_VALUE_THRESHOLD_USD / float_shares


def ib6_max_raise(float_shares: float | None,
                  price: float | None) -> float | None:
    """Maximum raisable under IB6 in any 12-month window: 1/3 of float
    market value."""
    if not float_shares or not price or float_shares <= 0 or price <= 0:
        return None
    return float_shares * price / 3.0


def ib6_remaining(cik: int, float_shares: float | None,
                  price: float | None,
                  today: _d | None = None) -> dict | None:
    """Combine max-raise and last-12mo raises into a single 'remaining
    raisable under IB6' figure, mirroring DT's 'Current Raisable Amount'."""
    cap = ib6_max_raise(float_shares, price)
    if cap is None:
        return None
    if not has_eligible_shelf(cik, today=today):
        return None
    raised = raised_under_ib6_last_12mo(cik, today=today)
    threshold = baby_shelf_threshold_price(float_shares)
    return {
        "float_shares": float_shares,
        "price": price,
        "float_value_usd": float_shares * price,
        "is_baby_shelf":
            float_shares * price < BABY_SHELF_FLOAT_VALUE_THRESHOLD_USD,
        "ib6_capacity_usd": cap,
        "raised_last_12mo_usd": raised["total"],
        "raisable_remaining_usd": max(0.0, cap - raised["total"]),
        "threshold_price_to_exit_baby_shelf": threshold,
        "raised_rows": raised["rows"],
    }


__all__ = [
    "BABY_SHELF_FLOAT_VALUE_THRESHOLD_USD",
    "IB6_ELIGIBLE_FORM_PREFIXES",
    "baby_shelf_threshold_price",
    "has_eligible_shelf",
    "ib6_max_raise",
    "ib6_remaining",
    "raised_under_ib6_last_12mo",
]
