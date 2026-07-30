"""Producer for the Finviz ingest payload — the wire format defined in
FINVIZ_API_CONTRACT.md.

One JSON-serializable document per ticker: the same projection the
internal dashboard renders (`/t/<TICKER>`), translated into the
contract's public vocabulary. Nothing here computes business math — it
reads the card / badge / chart objects the rest of the package already
produces and reshapes them. Every deviation between an internal field
name and its contract name is deliberate and commented.

What this layer is responsible for (§4 conventions):

  * internal `instrument_id` → opaque `source_ref` (never a key)
  * "Yes"/"No" label strings → real JSON booleans
  * `date` / `datetime` objects → "YYYY-MM-DD" / RFC 3339 strings
  * an explicit per-type field whitelist, so internal-only fields
    (`raisable_capped`, duplicate `status`, sub-object `instrument_id`s)
    never reach the consumer
  * producer-side filtering the contract promises: expired / withdrawn
    shelves are dropped here (the dashboard renders them; Finviz never
    sees them). Pre-funded / placement-agent warrants and withdrawn /
    lapsed S-1s are already filtered by the card layer.
  * the cached AI brief read (never generated) at push time, with the
    staleness flag the dashboard shows
  * a SETTLED market basis: `as_of` and the price behind the price-based
    O/S-chart segments are the last settled close, never the live price
    (contract §5.2 + open question #3 — a pushed snapshot must not carry
    an intraday-varying number).

Usage:

    from dilution.finviz_payload import build_payload
    doc = build_payload("GCTK")     # {"ticker": "GCTK", "data": {...}}
    snap = build_snapshot("GCTK")   # just the inner document

`scripts/dump_finviz_payload.py` is the CLI wrapper.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import date, datetime, timezone

from db import get_conn
from dilution.badges import compute_badges
from dilution.capital_raised import capital_raised_since
from dilution.cash_history import fetch_cash_history_cached
from dilution.finviz_client import (
    fundamentals as finviz_fundamentals,
    highest_close,
    latest_settled_close,
)
from dilution.os_history import build_fd_stack, fetch_os_history_cached
from dilution.share_counts import fetch_implied_outstanding_cached
from dilution import ticker_brief as brief_mod
from dilution.ledger.baby_shelf import (
    baby_shelf_threshold_price,
    is_baby_shelf_restricted,
)
from dilution.ledger.cards import (
    _resolve_float_shares,
    atm_cards,
    convertible_note_cards,
    equity_line_cards,
    preferred_cards,
    s1_offering_cards,
    shelf_cards,
    warrant_cards,
)
from dilution.ledger.shelf_status import WKSI_UNLIMITED_SHELF_CAPACITY_USD

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Contract §7.1: expired / withdrawn shelves are filtered producer-side
# and never pushed. The dashboard shows them (an expired shelf is useful
# history internally); the consumer's card list is live paper only.
_DROPPED_SHELF_STATUSES = ("expired", "withdrawn")


# ── scalar normalizers (§4 conventions) ──────────────────────────────
def _iso(value):
    """date/datetime → "YYYY-MM-DD"; pass through strings; else None."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _bool(value):
    """The card layer carries display labels ("Yes"/"No") where the
    contract wants a real boolean. None stays None — "not known" and
    "false" are different claims (§4)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("yes", "true", "1"):
        return True
    if text in ("no", "false", "0"):
        return False
    return None


def _num(value):
    """Numbers ride raw (the consumer owns rounding — open question #8);
    this only coerces away Decimal/str stragglers from the ledger."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _strs(value) -> list[str]:
    """known_owners and friends: always an array, never null (§7.4)."""
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


# ── shared card sub-objects (§7.0) ───────────────────────────────────
def _sub_ref(obj, keys: tuple[str, ...]) -> dict | None:
    """parent_shelf / resale_registration: whitelist only, dropping the
    internal instrument_id the card layer includes."""
    if not obj:
        return None
    out = {k: obj.get(k) for k in keys}
    if "filing_date" in out:
        out["filing_date"] = _iso(out["filing_date"])
    return out


def _parent_shelf(card) -> dict | None:
    return _sub_ref(card.get("parent_shelf"),
                    ("title", "file_number", "accession_number", "edgar_url"))


def _resale(card) -> dict | None:
    return _sub_ref(card.get("resale_registration"),
                    ("form", "filing_date", "file_number",
                     "accession_number", "edgar_url"))


def _head(card) -> dict:
    """Fields common to every card type (§7.0), in contract order."""
    return {
        # Opaque debug handle. Reassigned on every re-walk — the contract
        # forbids the consumer joining on it (§7.0).
        "source_ref": card.get("instrument_id"),
        "title": card.get("title"),
        "registered": card.get("registered"),
        "edgar_url": card.get("edgar_url"),
    }


def _tail(card) -> dict:
    return {
        "last_update_date": _iso(card.get("last_update_date")),
        "bank_tier": card.get("bank_tier"),
        "investor_class": card.get("investor_class"),
    }


# ── per-type card serializers (§7.1 – §7.7) ──────────────────────────
def _shelf_card(card) -> dict:
    capacity = _num(card.get("total_shelf_capacity"))
    raisable = _num(card.get("current_raisable_amount"))
    # WKSI / pay-as-you-go shelf (Rule 457(r)): the ledger carries an
    # indeterminate capacity as a sentinel. The wire format states the
    # fact as a boolean instead and nulls the amount — a consumer that
    # rendered the sentinel would print "$999,999,999 raisable".
    # A baby-shelf-restricted WKSI keeps its real I.B.6 raisable here
    # (the sentinel never survives that path), so `unlimited` correctly
    # stays false and the capped number is what renders.
    unlimited = (raisable == WKSI_UNLIMITED_SHELF_CAPACITY_USD
                 or capacity == WKSI_UNLIMITED_SHELF_CAPACITY_USD)
    return {
        **_head(card),
        "shelf_status": card.get("shelf_status"),
        "total_shelf_capacity":
            None if capacity == WKSI_UNLIMITED_SHELF_CAPACITY_USD else capacity,
        "current_raisable_amount": None if unlimited else raisable,
        "unlimited": unlimited,
        # Internal name is the display label `baby_shelf_restriction`.
        "is_baby_shelf_restricted": _bool(card.get("baby_shelf_restriction")),
        "total_amount_raised": _num(card.get("total_amount_raised")),
        "raised_last_12mo_under_ib6": _num(card.get("raised_last_12mo_under_ib6")),
        "outstanding_shares": _num(card.get("outstanding_shares")),
        # Internal card key is the bare `float`.
        "float_shares": _num(card.get("float")),
        "highest_60_day_close": _num(card.get("highest_60_day_close")),
        "price_to_exceed_baby_shelf": _num(card.get("price_to_exceed_baby_shelf")),
        "ib6_float_value": _num(card.get("ib6_float_value")),
        "last_banker": card.get("last_banker"),
        "effect_date": _iso(card.get("effect_date")),
        "expiration_date": _iso(card.get("expiration_date")),
        **_tail(card),
    }


def _atm_card(card) -> dict:
    # Naming flip, deliberate (§7.2): the contract's headline
    # `remaining_capacity` is the I.B.6-CAPPED raisable, which the card
    # layer keeps in `raisable_capped` (its own `remaining_capacity` is
    # the contractual remaining, the DT display convention). Shipping the
    # card's raw `remaining_capacity` as the headline would overstate what
    # a baby-shelf-restricted issuer can actually sell.
    contractual = _num(card.get("remaining_without_baby_shelf"))
    capped = card.get("raisable_capped")
    return {
        **_head(card),
        "parent_shelf": _parent_shelf(card),
        "total_capacity": _num(card.get("total_capacity")),
        "remaining_capacity": _num(capped) if capped is not None else contractual,
        "remaining_without_baby_shelf": contractual,
        "limited_by_baby_shelf": _bool(card.get("limited_by_baby_shelf")),
        "sales_total_usd": _num(card.get("sales_total_usd")),
        "used_pct": _num(card.get("used_pct")),
        "placement_agent": card.get("placement_agent"),
        "agreement_start_date": _iso(card.get("agreement_start_date")),
        "agreement_end_date": _iso(card.get("agreement_end_date")),
        **_tail(card),
    }


def _equity_line_card(card) -> dict:
    return {
        **_head(card),
        "parent_shelf": _parent_shelf(card),
        "total_capacity": _num(card.get("total_capacity")),
        "remaining_capacity": _num(card.get("remaining_capacity")),
        "sales_total_usd": _num(card.get("sales_total_usd")),
        "used_pct": _num(card.get("used_pct")),
        "counterparty": card.get("counterparty"),
        "agreement_start_date": _iso(card.get("agreement_start_date")),
        "agreement_end_date": _iso(card.get("agreement_end_date")),
        "terminated": bool(card.get("terminated")),
        **_tail(card),
    }


def _warrant_card(card) -> dict:
    return {
        **_head(card),
        "parent_shelf": _parent_shelf(card),
        "resale_registration": _resale(card),
        "total_issued": _num(card.get("total_issued")),
        "remaining_outstanding": _num(card.get("remaining_outstanding")),
        "exercise_price": _num(card.get("exercise_price")),
        "known_owners": _strs(card.get("known_owners")),
        "underwriter": card.get("underwriter"),
        "issue_date": _iso(card.get("issue_date")),
        "exercisable_date": _iso(card.get("exercisable_date")),
        "expiration_date": _iso(card.get("expiration_date")),
        **_tail(card),
    }


def _convertible_card(card) -> dict:
    """§7.5 convertible notes and §7.6 preferred share one field set;
    on preferred, principal_* is aggregate stated value.

    No `parent_shelf` here: notes and preferred are privately placed, so
    the card layer never links one (only warrants / ATMs / equity lines
    draw from a shelf) and the contract doesn't define the field for
    these types.
    """
    return {
        **_head(card),
        "resale_registration": _resale(card),
        "principal_total": _num(card.get("principal_total")),
        "principal_remaining": _num(card.get("principal_remaining")),
        "conversion_price": _num(card.get("conversion_price")),
        "total_shares_issuable": _num(card.get("total_shares_issuable")),
        "remaining_shares_issuable": _num(card.get("remaining_shares_issuable")),
        "known_owners": _strs(card.get("known_owners")),
        "underwriter": card.get("underwriter"),
        "issue_date": _iso(card.get("issue_date")),
        "convertible_date": _iso(card.get("convertible_date")),
        "maturity_date": _iso(card.get("maturity_date")),
        **_tail(card),
    }


def _s1_card(card) -> dict:
    # The card carries both `registered` and a duplicate `status` display
    # string; only the machine enum `s1_status` crosses the wire.
    return {
        **_head(card),
        "s1_status": card.get("s1_status"),
        "anticipated_deal_size": _num(card.get("anticipated_deal_size")),
        "final_deal_size": _num(card.get("final_deal_size")),
        "final_pricing": _num(card.get("final_pricing")),
        "final_shares_offered": _num(card.get("final_shares_offered")),
        "warrant_coverage_pct": _num(card.get("warrant_coverage_pct")),
        "final_warrant_coverage_pct": _num(card.get("final_warrant_coverage_pct")),
        "exercise_price": _num(card.get("exercise_price")),
        "underwriter": card.get("underwriter"),
        "filing_date": _iso(card.get("filing_date")),
        **_tail(card),
    }


def _cards_block(cards: dict) -> dict:
    """§7 — all seven keys always present, producer display order kept.
    `convertible_preferred` is the internal key; the wire name is
    `preferred`."""
    shelves = [c for c in cards.get("shelf") or []
               if c.get("shelf_status") not in _DROPPED_SHELF_STATUSES]
    return {
        "shelf": [_shelf_card(c) for c in shelves],
        "atm": [_atm_card(c) for c in cards.get("atm") or []],
        "equity_line": [_equity_line_card(c)
                        for c in cards.get("equity_line") or []],
        "warrant": [_warrant_card(c) for c in cards.get("warrant") or []],
        "convertible": [_convertible_card(c)
                        for c in cards.get("convertible") or []],
        "preferred": [_convertible_card(c)
                      for c in cards.get("convertible_preferred") or []],
        "s1_offering": [_s1_card(c) for c in cards.get("s1_offering") or []],
    }


# ── §5.1 cash ────────────────────────────────────────────────────────
def _fiscal_label(point: dict) -> str | None:
    """"2026 Q1" / "2025 FY" when the XBRL fiscal tags are trustworthy.

    `fy`/`fp` come off the reporting FACT, so they describe the filing's
    own fiscal context, not the period the cash balance belongs to. A
    2016-12-31 balance can arrive tagged fy=2018/fp=Q3, and a comparative
    prior-period balance carries the LATER filing's tags (GCTK's
    2025-12-31 year-end balance restated in the fy=2026/Q1 10-Q). Both
    would label the bar with the wrong period, so anything whose fiscal
    year doesn't match the period-end year is dropped to null and the
    consumer falls back to `period_end` (§5.1 rendering notes). That also
    nulls the legitimate off-calendar year-end (a November FY-end tags
    fy = end.year + 1) — a missing label beats a wrong one.
    """
    end, fy, fp = point.get("end"), point.get("fy"), point.get("fp")
    fp = (fp or "").strip().upper()
    if not isinstance(end, date) or not fy or fp not in ("FY", "Q1", "Q2",
                                                        "Q3", "Q4"):
        return None
    if int(fy) != end.year:
        return None
    return f"{int(fy)} {fp}"


def _cash_block(cash) -> dict | None:
    """§5.1. Reported bars + at most one trailing estimate bar, plot-ready
    and self-contained: the consumer never derives a bar."""
    if cash is None or not getattr(cash, "series", None):
        return None
    data = asdict(cash)
    bars = []
    for point in data.get("series") or []:
        bars.append({
            "kind": "reported",
            "period_end": _iso(point.get("end")),
            "fiscal": _fiscal_label(point),
            "form": point.get("form") or None,
            "cash_usd": _num(point.get("value_usd")),
            "overlay_usd": None,
        })
    current = _num(data.get("current_cash_est_usd"))
    if current is not None:
        raised = _num(data.get("capital_raised_usd"))
        bars.append({
            "kind": "estimate",
            "period_end": None,
            "fiscal": None,
            "form": None,
            "cash_usd": current,
            # Non-null on the estimate bar only: the stacked
            # "raised since the last report" segment.
            "overlay_usd": raised or None,
        })
    return {
        "latest_period_end": _iso(data.get("latest_period_end")),
        # Internal name is latest_cash_usd.
        "latest_reported_cash_usd": _num(data.get("latest_cash_usd")),
        "op_cf_quarterly_usd": _num(data.get("op_cf_quarterly_usd")),
        "capital_raised_since_usd": _num(data.get("capital_raised_usd")),
        "current_cash_est_usd": current,
        "months_of_cash": _num(data.get("months_of_cash")),
        "stale_days": _num(data.get("stale_days")),
        "fx_failed": bool(data.get("fx_failed")),
        "chart": {"bars": bars},
    }


# ── §5.2 os_chart ────────────────────────────────────────────────────
def _os_chart_block(osh, latest_os, latest_note, stack, price_basis):
    """§5.2. Historical quarterly O/S bars + the fully-diluted bar
    (current O/S base + potential-dilution segments)."""
    if osh is None:
        return None
    bars = [{
        "quarter_end": _iso(p.quarter_end),
        "shares": _num(p.shares),
        "raw_shares": _num(p.raw_shares),
        "source_date": _iso(p.source_date),
        "form": p.form or None,
        "carried": bool(p.carried),
        "split_adjusted": bool(p.split_adjusted),
    } for p in osh.series or []]
    if not bars and not (latest_os and stack):
        return None
    fd_stack = []
    for seg in stack:
        # StackSegment keeps only the share count; the contract also ships
        # the dollar capacity behind a price-based segment as hard data
        # (§5.2 / open question #3), and shares × basis inverts the
        # capacity ÷ basis the segment was built from exactly.
        capacity = (_num(seg.shares) * price_basis
                    if (seg.price_based and price_basis) else None)
        fd_stack.append({
            "key": seg.key,
            "label": seg.label,
            "shares": _num(seg.shares),
            "price_based": bool(seg.price_based),
            "capacity_usd": capacity,
            "note": seg.note,
        })
    return {
        "ads_ratio": _num(osh.ads_ratio),
        "price_basis": price_basis,
        "bars": bars,
        "latest": {"shares": _num(latest_os), "source": latest_note},
        "fd_stack": fd_stack,
    }


# ── §6 badges ────────────────────────────────────────────────────────
def _legend(rows) -> list[dict]:
    """Internal legend tuples are (band, pill text, meaning)."""
    return [{"band": band, "pill": pill, "meaning": meaning}
            for band, pill, meaning in rows or ()]


def _badges_block(badges) -> dict | None:
    """§6. The whole block is nullable when nothing was computable."""
    if badges is None:
        return None
    return {
        "overall": {
            "score": badges.overall_score,
            "band": badges.overall_band,
            "label": badges.overall_label,
            "partial": bool(badges.partial),
            "description": badges.description,
            "detail": list(badges.detail or ()),
            "legend": _legend(badges.legend),
        },
        "drivers": [{
            "key": d.key,
            "label": d.label,
            "score": d.score,
            "band": d.band,
            "band_text": d.band_text,
            "description": d.description,
            "detail": list(d.detail or ()),
            "legend": _legend(d.legend),
        } for d in badges.drivers or ()],
    }


# ── §8 brief ─────────────────────────────────────────────────────────
def _latest_filing_date(cik: int) -> str | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MAX(filing_date) d FROM dilution_filings WHERE cik = ?",
            (int(cik),),
        ).fetchone()
    return row["d"] if row else None


def _brief_block(cik: int) -> dict | None:
    """§8. The cached AI dilution brief, or null when none exists.

    Display-only prose generated from the same deterministic facts the
    rest of the payload carries (cards / badges / cash). The producer
    only *reads* the cache here — generation is a pipeline job
    (scripts/run_brief_all.py), so a push never blocks on an LLM call
    and never mints text mid-request.

    `stale` is the dashboard's rule: a filing arrived after the brief
    was generated, so the prose may not mention it. It does NOT catch
    prose that went stale because the LEDGER changed under it (a re-walk
    with no new filing), which is why `generated_at` ships too — a
    consumer that wants to suppress old commentary can age it out.
    """
    try:
        cached = brief_mod.get_cached(cik)
    except Exception:
        log.exception("brief lookup failed for cik=%s", cik)
        return None
    if not cached:
        return None
    generated_at = cached.get("generated_at")
    latest_filing = None
    try:
        latest_filing = _latest_filing_date(cik)
    except Exception:
        log.exception("latest-filing lookup failed for cik=%s", cik)
    stale = bool(latest_filing and generated_at
                 and latest_filing > generated_at[:10])
    return {
        "headline": cached.get("headline"),
        "bullets": _strs(cached.get("bullets")),
        # Dated forward-looking items (expiries, maturities, lock-up
        # ends). Frequently empty — not every issuer has any.
        "watch": _strs(cached.get("watch")),
        "generated_at": generated_at,
        "stale": stale,
        # The filing that makes it stale, so the consumer can say what it
        # predates instead of just flagging it. Null when not stale.
        "stale_since_filing_date": latest_filing if stale else None,
    }


# ── assembly ─────────────────────────────────────────────────────────
def _company_row(ticker: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT cik, ticker, name FROM dilution_company WHERE ticker = ?",
            (ticker.upper(),),
        ).fetchone()
    return dict(row) if row else None


def _internal_cards(cik: int, fund: dict | None, latest_os) -> dict:
    """The same seven arrays the dashboard renders, internal keys — see
    dashboard.app._cards_for."""
    return {
        "s1_offering": s1_offering_cards(cik),
        "warrant": warrant_cards(cik),
        "convertible": convertible_note_cards(cik),
        "convertible_preferred": preferred_cards(cik),
        "atm": atm_cards(cik, fund, latest_os),
        "equity_line": equity_line_cards(cik),
        "shelf": shelf_cards(cik, fund, latest_os),
    }


def _cash_and_raised(cik: int):
    """Two-phase cash fetch (probe for the latest period end, price the
    raises since it, then bridge) — dashboard.app._cash_for."""
    try:
        probe = fetch_cash_history_cached(cik)
        raised = (capital_raised_since(cik, probe.latest_period_end)
                  if probe.latest_period_end else None)
        return fetch_cash_history_cached(cik, capital_raised_usd=raised), raised
    except Exception:
        log.exception("cash fetch failed for cik=%s", cik)
        return None, None


def build_snapshot(ticker: str, *, generated_at: datetime | None = None) -> dict:
    """The snapshot itself — what rides under `data` (§4).

    Raises LookupError when the ticker isn't tracked. Sub-blocks that
    depend on external data (cash, O/S history, badges, brief) degrade to
    null/omitted rather than failing the whole document — same fail-soft
    posture as the dashboard route.
    """
    ticker = ticker.upper()
    row = _company_row(ticker)
    if not row:
        raise LookupError(f"{ticker} is not a tracked ticker")
    cik = int(row["cik"])

    fund = finviz_fundamentals(ticker)
    implied = fetch_implied_outstanding_cached(cik)
    latest_os = (implied.total if implied.total is not None
                 else (fund or {}).get("shares_outstanding"))

    cards = _internal_cards(cik, fund, latest_os)

    # Settled market basis. `as_of` is the trading date every
    # market-derived number reflects; the same close prices the
    # price-based O/S segments (never the live price — §5.2).
    settled = None
    try:
        settled = latest_settled_close(ticker)
    except Exception:
        log.exception("settled close lookup failed for %s", ticker)
    as_of, price_basis = (settled or (None, None))
    as_of = as_of or date.today()

    # Company-level I.B.6 context, computed exactly as the ATM/shelf
    # cards compute it so the mirrored card fields agree by construction.
    float_shares = _resolve_float_shares(cik, fund, latest_os)
    high60 = None
    try:
        high60 = highest_close(ticker, bars=60)
    except Exception:
        log.exception("highest_close lookup failed for %s", ticker)
    try:
        baby = is_baby_shelf_restricted(cik, float_shares, latest_os, high60)
    except Exception:
        log.exception("baby-shelf test failed for %s", ticker)
        baby = None

    cash, _raised = _cash_and_raised(cik)

    osh = None
    os_block = None
    try:
        osh = fetch_os_history_cached(cik)
        if implied.total is not None:
            latest_note = ("%s XBRL%s" % (
                implied.source_form,
                f", a/o {implied.as_of.isoformat()}" if implied.as_of else ""))
        else:
            latest_note = "Finviz cover-page count (XBRL unavailable)"
        os_block = _os_chart_block(
            osh, latest_os, latest_note,
            build_fd_stack(cards, price_basis), price_basis)
    except Exception:
        log.exception("O/S chart block failed for %s", ticker)

    try:
        badges = compute_badges(cik, fund=fund, latest_os=latest_os,
                                cards=cards, cash=cash)
    except Exception:
        log.exception("badges failed for %s", ticker)
        badges = None

    company = {
        "shares_outstanding": _num(latest_os),
        "float_shares": _num(float_shares),
        "highest_60_day_close": _num(high60),
        "price_to_exceed_baby_shelf":
            _num(baby_shelf_threshold_price(float_shares)),
        "is_baby_shelf_restricted": baby,
    }
    cash_block = _cash_block(cash)
    if cash_block is not None:
        company["cash"] = cash_block
    if os_block is not None:
        company["os_chart"] = os_block

    stamp = (generated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return {
        "schema_version": SCHEMA_VERSION,
        "ticker": ticker,
        "cik": cik,
        "company_name": row["name"],
        "as_of": _iso(as_of),
        "generated_at": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "company": company,
        "badges": _badges_block(badges),
        "cards": _cards_block(cards),
        "brief": _brief_block(cik),
    }


def build_payload(ticker: str, *, generated_at: datetime | None = None) -> dict:
    """One ticker's complete PUT body: `{"ticker": ..., "data": {...}}`.

    The wrapper is the wire contract (§4); `data` carries the snapshot
    verbatim, ticker included, so a detached snapshot still identifies
    itself.
    """
    snapshot = build_snapshot(ticker, generated_at=generated_at)
    return {"ticker": snapshot["ticker"], "data": snapshot}
