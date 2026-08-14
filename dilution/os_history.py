"""Historical shares outstanding + the potential-dilution stack.

Powers the DilutionTracker-style "Historical O/S & Potential Dilution"
chart. Ships as data points in payload §5.2; Finviz draws it:

  dark-blue bars   split-adjusted cover-page shares outstanding, one
                   bar per calendar quarter (dei XBRL facts from each
                   periodic filing; quarters with no filing carry the
                   previous count forward)
  final bar        latest O/S (dark base) plus stacked potential
                   dilution from the instrument cards: warrants /
                   convertible notes / preferred at their share counts,
                   ATM / equity-line / S-1 capacity ÷ current price

Methodology matches DT's published rules (knowledge.dilutiontracker.com,
"How do I interpret the O/S chart?"):
  * ATM / equity-line / S-1 dollars are divided by the CURRENT market
    price — an assumption, not a forecast (real offerings price lower,
    so real dilution is higher).
  * Warrants count at outstanding share counts whether in or out of
    the money. Our warrant cards already exclude pre-funded and
    placement-agent warrants, so the segment inherits those rules.
  * Shelf capacity is excluded entirely — headline shelf sizes dwarf
    any realistic single raise (it's scored under Offering Ability).

Split adjustment: cover-page counts are as-reported on their date, so
each point is multiplied by Π(post/pre) over all later dilution_splits
rows with units='common' (a 1-for-250 reverse divides old counts by
250). FPI ordinary-share counts convert to ADS at the CURRENT ratio —
ADS-ratio changes cancel out because the ordinary count is
ratio-independent: today's-ADS-equivalent = ordinary_adj / ratio_now.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache

from edgar import Company, set_identity

import config
from db import get_conn

log = logging.getLogger(__name__)

# Probe order. dei: is the cover page ("as of the latest practicable
# date" — what DT plots); us-gaap: is the balance-sheet count, used
# only when the dei concept is missing entirely.
_OS_CONCEPTS = (
    "dei:EntityCommonStockSharesOutstanding",
    "us-gaap:CommonStockSharesOutstanding",
)

_MAX_YEARS = 10
_IDENTITY_SET = False


@dataclass(frozen=True)
class OsPoint:
    quarter_end: date        # calendar-quarter bucket the bar sits in
    shares: float            # split-adjusted, payload units (ADS for FPI)
    raw_shares: float        # as-reported on the cover page
    source_date: date        # the fact's instant date (cover date)
    form: str
    carried: bool            # no filing this quarter — carried forward
    split_adjusted: bool


@dataclass(frozen=True)
class OsHistory:
    series: list[OsPoint] = field(default_factory=list)
    as_of: date = field(default_factory=date.today)
    concept: str | None = None
    ads_ratio: float | None = None   # set when FPI conversion applied
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class StackSegment:
    key: str          # warrant|convertible|preferred|atm|equity_line|s1
    label: str
    shares: float
    note: str         # tooltip provenance line
    price_based: bool # True when computed as $-capacity ÷ price


def _ensure_identity() -> None:
    global _IDENTITY_SET
    if _IDENTITY_SET:
        return
    set_identity(getattr(config, "EDGAR_IDENTITY",
                         "dilution-tracker contact@example.com"))
    _IDENTITY_SET = True


def _company_unit(cik: int) -> tuple[bool, float | None]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT is_fpi, ads_ratio FROM dilution_company WHERE cik = ?",
            (int(cik),),
        ).fetchone()
    if not row:
        return False, None
    return (bool(row["is_fpi"]),
            float(row["ads_ratio"]) if row["ads_ratio"] is not None else None)


def _split_factors(cik: int) -> list[tuple[date, float]]:
    """(effective_date, post/pre) for common-stock splits, ascending.

    units='ads' rows (FPI ratio changes) are deliberately excluded —
    see the module docstring: ordinary counts divided by the current
    ratio already land on today's ADS basis.
    """
    try:
        with get_conn() as conn:
            rows = conn.execute(
                """SELECT effective_date, pre, post FROM dilution_splits
                   WHERE cik = ? AND units = 'common'
                   ORDER BY effective_date""",
                (int(cik),),
            ).fetchall()
    except Exception as e:
        log.warning("splits lookup failed for cik %s: %s", cik, e)
        return []
    out = []
    for r in rows:
        try:
            if r["pre"]:
                out.append((date.fromisoformat(r["effective_date"]),
                            float(r["post"]) / float(r["pre"])))
        except (TypeError, ValueError):
            continue
    return out


def _adjustment(splits: list[tuple[date, float]], d: date) -> float:
    """Cumulative factor putting a count reported on `d` on today's basis."""
    f = 1.0
    for eff, fac in splits:
        if eff > d:
            f *= fac
    return f


def _quarter_end(d: date) -> date:
    last_month = ((d.month - 1) // 3) * 3 + 3
    if last_month == 12:
        return date(d.year, 12, 31)
    return date(d.year, last_month + 1, 1) - timedelta(days=1)


def _next_quarter_end(q: date) -> date:
    return _quarter_end(q + timedelta(days=1))


def _as_date(v) -> date | None:
    if v is None:
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    try:
        return datetime.fromisoformat(str(v)).date()
    except ValueError:
        return None


def _query_first_with_facts(facts, concepts) -> tuple[list, str | None]:
    for c in concepts:
        try:
            r = facts.query().by_concept(c, exact=True).execute()
        except Exception as e:
            log.debug("query failed for %s: %s", c, e)
            continue
        if r:
            return r, c
    return [], None


def fetch_os_history(cik: int, *, as_of: date | None = None) -> OsHistory:
    """Quarterly split-adjusted shares-outstanding series from XBRL.

    Known limitation: edgartools' Company.get_facts() drops dimensioned
    facts, so multi-class issuers (Up-C) surface only the undimensioned
    class here — the FD bar's base (share_counts) handles classes
    properly, the history may understate. Single-class microcaps (the
    universe this pipeline targets) are unaffected.
    """
    _ensure_identity()
    as_of = as_of or date.today()
    is_fpi, ads_ratio = _company_unit(cik)
    if is_fpi and not ads_ratio:
        return OsHistory(as_of=as_of, warnings=("fpi_ads_ratio_missing",))

    try:
        facts = Company(int(cik)).get_facts()
    except Exception as e:
        log.warning("get_facts failed for CIK %s: %s", cik, e)
        return OsHistory(as_of=as_of, warnings=("facts_fetch_failed",))

    raw, concept = _query_first_with_facts(facts, _OS_CONCEPTS)
    if not raw:
        return OsHistory(as_of=as_of, warnings=("concept_missing",))

    # Dedup by instant date keeping the latest filing (restatements).
    by_end: dict[date, object] = {}
    for f in raw:
        end = _as_date(f.period_end)
        if end is None or f.numeric_value is None:
            continue
        try:
            if float(f.numeric_value) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        cur = by_end.get(end)
        if cur is None:
            by_end[end] = f
        else:
            new_fd = _as_date(f.filing_date)
            cur_fd = _as_date(cur.filing_date)
            # A missing filing_date sorts oldest: keep the dated restatement
            # rather than crashing on `None > date(...)`.
            if new_fd is not None and (cur_fd is None or new_fd > cur_fd):
                by_end[end] = f

    cutoff = date(as_of.year - _MAX_YEARS, as_of.month, 1)
    splits = _split_factors(cik)

    # One representative fact per calendar quarter: the latest cover
    # date inside the quarter wins.
    buckets: dict[date, object] = {}
    for end in sorted(by_end):
        if end < cutoff or end > as_of:
            continue
        buckets[_quarter_end(end)] = by_end[end]

    if not buckets:
        return OsHistory(as_of=as_of, concept=concept,
                         warnings=("no_facts_in_window",))

    series: list[OsPoint] = []
    q = min(buckets)
    last_q = _quarter_end(as_of)
    prev: OsPoint | None = None
    while q <= last_q:
        f = buckets.get(q)
        if f is not None:
            src = _as_date(f.period_end)
            raw_val = float(f.numeric_value)
            adj = _adjustment(splits, src)
            val = raw_val * adj
            if is_fpi and ads_ratio:
                val /= ads_ratio
            prev = OsPoint(
                quarter_end=q,
                shares=val,
                raw_shares=raw_val,
                source_date=src,
                form=str(f.form_type or ""),
                carried=False,
                split_adjusted=(adj != 1.0),
            )
            series.append(prev)
        elif prev is not None:
            series.append(OsPoint(
                quarter_end=q,
                shares=prev.shares,
                raw_shares=prev.raw_shares,
                source_date=prev.source_date,
                form=prev.form,
                carried=True,
                split_adjusted=prev.split_adjusted,
            ))
        q = _next_quarter_end(q)

    return OsHistory(series=series, as_of=as_of, concept=concept,
                     ads_ratio=ads_ratio if is_fpi else None)


@lru_cache(maxsize=256)
def _cached(cik: int, as_of_iso: str) -> OsHistory:
    return fetch_os_history(int(cik), as_of=date.fromisoformat(as_of_iso))


def fetch_os_history_cached(cik: int, *, as_of: date | None = None) -> OsHistory:
    return _cached(int(cik), (as_of or date.today()).isoformat())


# ── Fully-diluted stack ──────────────────────────────────────────────
def _usd0(x: float) -> str:
    return f"${x:,.0f}"


def build_fd_stack(cards: dict, price: float | None) -> list[StackSegment]:
    """Potential-dilution segments stacked on top of the latest O/S.

    Reads the same card dicts the page renders (and badges.py scores),
    so every segment is auditable against a card below the chart.
    Returned in display order: fixed-share paper first, price-dependent
    estimates on top.
    """
    segs: list[StackSegment] = []

    ws = cards.get("warrant") or []
    w_sh = sum(float(c.get("remaining_outstanding") or 0) for c in ws)
    if w_sh > 0:
        segs.append(StackSegment(
            "warrant", "Warrants", w_sh,
            f"Remaining outstanding across {len(ws)} warrant card"
            f"{'s' if len(ws) != 1 else ''}, in or out of the money "
            f"(pre-funded and placement-agent warrants excluded)",
            False))

    for key, label, noun in (("convertible", "Convertible Notes", "note"),
                             ("convertible_preferred", "Convertible Preferred",
                              "preferred")):
        cs = cards.get(key) or []
        sh = sum(float(c.get("remaining_shares_issuable") or 0) for c in cs)
        missing = sum(1 for c in cs
                      if c.get("remaining_shares_issuable") is None
                      and (c.get("principal_remaining") or 0) > 0)
        if sh > 0:
            note = (f"Shares issuable from remaining principal at stated "
                    f"conversion prices ({len(cs)} {noun} card"
                    f"{'s' if len(cs) != 1 else ''})")
            if missing:
                note += (f"; {missing} with undisclosed conversion price "
                         f"excluded")
            segs.append(StackSegment(
                key if key == "convertible" else "preferred",
                label, sh, note, False))

    if price and price > 0:
        # raisable_capped = IB6-capped live raisable; remaining_capacity is
        # the contractual remaining (DT display convention) and would
        # over-count dilution pressure on baby-shelf-restricted issuers.
        atm_usd = sum(float(c.get("raisable_capped",
                                  c.get("remaining_capacity")) or 0)
                      for c in (cards.get("atm") or []))
        if atm_usd > 0:
            segs.append(StackSegment(
                "atm", "ATM", atm_usd / price,
                f"{_usd0(atm_usd)} remaining ATM capacity ÷ ${price:.2f} "
                f"(I.B.6-capped where applicable)",
                True))

        eloc_usd = sum(float(c.get("remaining_capacity") or 0)
                       for c in (cards.get("equity_line") or [])
                       if not c.get("terminated"))
        if eloc_usd > 0:
            segs.append(StackSegment(
                "equity_line", "Equity Line", eloc_usd / price,
                f"{_usd0(eloc_usd)} remaining equity-line capacity "
                f"÷ ${price:.2f}",
                True))

        s1_usd = sum(float(c.get("anticipated_deal_size") or 0)
                     for c in (cards.get("s1_offering") or [])
                     if c.get("s1_status") in ("pending", "effective"))
        if s1_usd > 0:
            segs.append(StackSegment(
                "s1", "S-1 Offering", s1_usd / price,
                f"{_usd0(s1_usd)} anticipated deal size ÷ ${price:.2f}",
                True))

    return segs
