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

    Per Instruction 2 to General Instruction I.B.6 (and C&DI 116.24),
    warrants sold in a unit offering consume cap at the market value of
    their UNDERLYING shares — even when not exercisable within 12
    months — valued at the same per-share price as the offering. We
    approximate via warrants whose created_accession matches a
    contributing drawdown, valued at that drawdown's price. Reported
    separately in rows (type='warrant_underlying') and included in
    total.
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
                      l.type, d.drawdown_party_canonical
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
        for prev_acc, prev_amt, _prev_date in seen:
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
            "counterparty": r["drawdown_party_canonical"],
            "accession": acc,
        })
    # C&DI 116.24: unit-offering warrants consume cap at underlying-
    # share value. Link warrants minted by the same accession as a
    # contributing drawdown; value = share count × that drawdown's
    # per-share price (the Instruction-1 price proxy at sale time).
    by_acc_price: dict[str, float] = {}
    for r in rows:
        if r["price"] and r["accession_number"] not in by_acc_price:
            by_acc_price[r["accession_number"]] = float(r["price"])
    contributing_accs = {c["accession"] for c in contributing}
    if contributing_accs:
        qmarks = ",".join("?" * len(contributing_accs))
        with get_conn() as conn:
            wrows = conn.execute(
                f"""SELECT instrument_id, created_accession,
                           outstanding_json
                      FROM dilution_ledger
                     WHERE cik = ? AND type = 'warrant'
                       AND created_accession IN ({qmarks})""",
                (cik, *sorted(contributing_accs)),
            ).fetchall()
        import json as _json
        for w in wrows:
            price = by_acc_price.get(w["created_accession"])
            if not price:
                continue
            try:
                out = _json.loads(w["outstanding_json"] or "{}")
            except ValueError:
                continue
            count = out.get("initial_count") or out.get("count")
            if not count or count <= 0:
                continue
            underlying = float(count) * price
            total += underlying
            contributing.append({
                "date": None,
                "instrument_id": w["instrument_id"],
                "type": "warrant_underlying",
                "proceeds": underlying,
                "counterparty": None,
                "accession": w["created_accession"],
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


def ib6_basis_shares(cik: int, float_shares: float | None,
                     latest_os: float | None) -> float | None:
    """Share count whose market value feeds the I.B.6 $75M test.

    Form S-3 General Instruction I.B.6(a) measures the "aggregate
    market value of the voting and non-voting common equity held by
    NON-AFFILIATES" — the public float, not total shares outstanding.
    For a single-class issuer the strict float is therefore the basis.
    ACTU is the discriminating case: ~57% affiliate-held, so OS×price
    ($79M) and float×price ($34M) straddle the $75M line — the old
    os-basis misclassified it as unrestricted while its 424B covers
    carry the I.B.6 calculation.

    Up-C / multi-class issuers (GENK / SHAK / SG) are the exception:
    the float feed covers only the listed Class A, while Class B / LLC
    units are exchangeable common equity those issuers count when
    self-determining I.B.1 vs I.B.6. Per-class non-affiliate splits
    aren't observable in XBRL, so the implied fully-exchanged total
    remains the best approximation there (it matches their actual
    I.B.1 filings; strict Class-A float would misclassify them as
    baby-restricted).
    """
    try:
        from dilution.share_counts import fetch_implied_outstanding_cached
        multi_class = (
            len(fetch_implied_outstanding_cached(int(cik)).classes) >= 2
        )
    except Exception:
        multi_class = False
    primary, fallback = (
        (latest_os, float_shares) if multi_class
        else (float_shares, latest_os)
    )
    if primary:
        return float(primary)
    return float(fallback) if fallback else None


# Upward-crossing exit override (General Instruction I.B.6, Instruction 3):
# once non-affiliate float value reaches $75M on any date AFTER a
# registration's effective date, the 1/3 cap stops and the offering is
# treated as I.B.1 (unrestricted) — a permanent, one-way ratchet. The
# filing-time cover-legend STAMP cannot see a float rise that postdates the
# issuer's last primary prospectus, and FPIs have no dei:EntityPublicFloat
# 10-K fallback to catch it (QTEX/cik 1837493: stamped 'baby' off a Feb-2026
# 424B5 at $40.5M float, since risen to ~$151.9M via a PP + price gain). This
# override flips a 'baby' stamp to unrestricted ONLY on a MARGIN + PERSISTENCE
# signal, so it never flaps on a transient spike near the threshold — the
# exact noise the stamp model was adopted to avoid. Conservative by
# construction: when any input is missing it keeps the stamp (understating
# raise capacity is the safe error).
BABY_SHELF_EXIT_MARGIN_MULT = 1.20            # 60-day-high value >= 1.2 x $75M
BABY_SHELF_EXIT_PERSIST_WINDOW_DAYS = 90      # trailing calendar-day window
BABY_SHELF_EXIT_PERSIST_MIN_FRACTION = 0.80   # >=80% of window closes >= $75M

# Process-lifetime cache of the daily-close window per cik (mirrors the
# convention of cards._MARKET_LOW_CACHE) so shelf_cards + atm_cards don't
# each re-hit the quote feed within one render.
_BABY_EXIT_CLOSES_CACHE: dict[int, list[float]] = {}


def _ticker_for_cik(cik: int) -> str | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT ticker FROM dilution_company WHERE cik = ?", (cik,),
        ).fetchone()
    return row["ticker"] if row else None


def _durably_exited_baby_shelf(cik: int, basis_shares: float | None,
                               effective_price: float | None) -> bool:
    """True when a 'baby' regime STAMP should be overridden to unrestricted
    under Instruction 3 — the non-affiliate float value has crossed $75M
    with margin AND held there persistently.

    Margin gates on the 60-day-high float value (caller-supplied
    `effective_price`); persistence gates on the share of trailing
    90-calendar-day closes whose float value clears the bare $75M
    threshold. Failure-safe: any missing input or price-history gap
    returns False, leaving the filing-time stamp in force."""
    if not basis_shares or not effective_price:
        return False
    # Margin: the 60-day-high float value must clear the threshold by the
    # configured headroom, so we never un-baby an issuer sitting at the line.
    if (basis_shares * effective_price
            < BABY_SHELF_EXIT_MARGIN_MULT
            * BABY_SHELF_FLOAT_VALUE_THRESHOLD_USD):
        return False
    # Persistence: float value must have held above $75M on a supermajority
    # of recent trading days — same daily-close feed the IB6 price uses.
    if cik in _BABY_EXIT_CLOSES_CACHE:
        vals = _BABY_EXIT_CLOSES_CACHE[cik]
    else:
        vals = []
        ticker = _ticker_for_cik(cik)
        if ticker:
            try:
                from dilution.finviz_client import _client
                closes = _client().get_daily_closes(
                    ticker,
                    bars=BABY_SHELF_EXIT_PERSIST_WINDOW_DAYS,
                    within_calendar_days=BABY_SHELF_EXIT_PERSIST_WINDOW_DAYS)
                vals = [float(c) for c in (closes or []) if c]
            except Exception:
                vals = []
        _BABY_EXIT_CLOSES_CACHE[cik] = vals
    if len(vals) < 2:
        return False
    above = sum(1 for c in vals
                if basis_shares * c >= BABY_SHELF_FLOAT_VALUE_THRESHOLD_USD)
    return (above / len(vals)) >= BABY_SHELF_EXIT_PERSIST_MIN_FRACTION


def is_baby_shelf_restricted(cik: int,
                             float_shares: float | None,
                             latest_os: float | None,
                             effective_price: float | None) -> bool:
    """Single home of the baby-shelf classification.

    Primary signal: the FILING-TIME regime stamp from the issuer's
    latest primary prospectus cover (dilution.ib6_cover). Instruction 7
    of General Instruction I.B.6 forces every I.B.6 prospectus to carry
    the calculation legend, so legend presence/absence states the
    regime under counsel's liability — and per C&DI 116.26 (2026-03-19)
    the regime attaches at the prospectus-supplement filing, surviving
    later float moves (KSCP: legend-free $50M ATM supplement stays
    uncapped while its live float value sits below $75M). A live
    float×price test cannot reproduce that, and flaps with price near
    the threshold besides.

    Second tier (no stamp): the latest dei:EntityPublicFloat 10-K cover
    fact vs $75M — issuer-computed and dated, but annual-only, and it
    never overrides a stamp (KSCP/XTIA fixtures prove DT keeps the
    stamp when a newer 10-K disagrees).

    Fallback (neither available): the computed test — non-affiliate
    float basis per ib6_basis_shares × the 60-calendar-day high close
    vs the $75M threshold.
    """
    try:
        from dilution.ib6_cover import ib6_regime
        regime = ib6_regime(cik)["regime"]
    except Exception as exc:  # regime scan must never break card render
        import logging
        logging.getLogger(__name__).warning(
            "ib6_regime failed for cik=%s: %s", cik, exc)
        regime = None
    if regime == "baby":
        # Instruction-3 upward ratchet: a durable post-effective-date float
        # crossing past $75M stops the 1/3 cap, overriding a stale 'baby'
        # stamp that the filing-time legend (and, for FPIs, the absent 10-K
        # float) cannot see. Margin + persistence guard against flapping.
        basis = ib6_basis_shares(cik, float_shares, latest_os)
        if _durably_exited_baby_shelf(cik, basis, effective_price):
            return False
        return True
    if regime == "unrestricted":
        return False
    basis = ib6_basis_shares(cik, float_shares, latest_os)
    value = (basis * effective_price
             if (basis and effective_price) else None)
    return bool(value is not None
                and value < BABY_SHELF_FLOAT_VALUE_THRESHOLD_USD)


def ib6_max_raise(float_shares: float | None,
                  price: float | None) -> float | None:
    """Maximum raisable under IB6 in any 12-month window: 1/3 of float
    market value."""
    if not float_shares or not price or float_shares <= 0 or price <= 0:
        return None
    return float_shares * price / 3.0


def _unsold_live_atm_usd(cik: int) -> float:
    """Unsold remaining capacity across the issuer's ACTIVE ATM
    programs. Per C&DI 116.23, securities still being offered in a live
    continuous offering count against the 1/3 cap for any NEW takedown
    — only actual sales count for the trailing window itself."""
    import json as _json
    total = 0.0
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT terms_json, outstanding_json FROM dilution_ledger
                WHERE cik = ? AND type = 'atm' AND status = 'active'""",
            (cik,),
        ).fetchall()
    for r in rows:
        try:
            terms = _json.loads(r["terms_json"] or "{}")
            out = _json.loads(r["outstanding_json"] or "{}")
        except ValueError:
            continue
        remaining = out.get("remaining_capacity_usd")
        if remaining is None:
            cap = terms.get("capacity_usd")
            drawn = out.get("drawn_usd") or 0
            remaining = (cap - drawn) if cap is not None else None
        if remaining and remaining > 0:
            total += float(remaining)
    return total


def ib6_remaining(cik: int, float_shares: float | None,
                  price: float | None,
                  today: _d | None = None) -> dict | None:
    """Combine max-raise and last-12mo raises into a single 'remaining
    raisable under IB6' figure, mirroring DT's 'Current Raisable Amount'.

    `raisable_remaining_usd` deliberately matches DT's semantics (cap −
    actual trailing sales). `raisable_new_takedown_usd` additionally
    subtracts unsold live-ATM capacity per C&DI 116.23 — the legally
    available size of a NEW prospectus supplement today — and is
    informational (fixtures never assert it; DT doesn't display it)."""
    cap = ib6_max_raise(float_shares, price)
    if cap is None:
        return None
    if not has_eligible_shelf(cik, today=today):
        return None
    raised = raised_under_ib6_last_12mo(cik, today=today)
    threshold = baby_shelf_threshold_price(float_shares)
    unsold_atm = _unsold_live_atm_usd(cik)
    remaining = max(0.0, cap - raised["total"])
    return {
        "float_shares": float_shares,
        "price": price,
        "float_value_usd": float_shares * price,
        "is_baby_shelf":
            float_shares * price < BABY_SHELF_FLOAT_VALUE_THRESHOLD_USD,
        "ib6_capacity_usd": cap,
        "raised_last_12mo_usd": raised["total"],
        "raisable_remaining_usd": remaining,
        "unsold_live_atm_usd": unsold_atm,
        "raisable_new_takedown_usd": max(0.0, remaining - unsold_atm),
        "threshold_price_to_exit_baby_shelf": threshold,
        "raised_rows": raised["rows"],
    }


__all__ = [
    "BABY_SHELF_FLOAT_VALUE_THRESHOLD_USD",
    "BABY_SHELF_EXIT_MARGIN_MULT",
    "BABY_SHELF_EXIT_PERSIST_WINDOW_DAYS",
    "BABY_SHELF_EXIT_PERSIST_MIN_FRACTION",
    "IB6_ELIGIBLE_FORM_PREFIXES",
    "baby_shelf_threshold_price",
    "has_eligible_shelf",
    "ib6_basis_shares",
    "ib6_max_raise",
    "is_baby_shelf_restricted",
    "ib6_remaining",
    "raised_under_ib6_last_12mo",
]
