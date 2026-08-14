"""Historical cash position + bridge to current cash estimate.

Powers the DilutionTracker-style "Cash Position" chart. Ships as data
points in payload §5.1; Finviz draws it:

  blue bars     historical cash & equivalents per fiscal period
  maroon bar    prorated operating cash flow since latest period end
  light-blue    capital raised since latest period end (net of fees)
  final bar     current cash estimate = latest + opcf_prorated + raised

Concept resolution probes us-gaap first, then IFRS so 20-F filers
(foreign private issuers) are supported. Non-USD facts are converted
via dilution.fx at each period-end date.

The series is sorted ascending by period_end, deduplicated by end date
(keeping the most-recently-filed accession to absorb restatements), and
clamped to ~10 years for legibility.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import lru_cache

from edgar import Company, set_identity

import config
from dilution import fx

log = logging.getLogger(__name__)

# Probe order. First concept with ≥1 fact wins.
_CASH_CONCEPTS = (
    "us-gaap:CashAndCashEquivalentsAtCarryingValue",
    "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    "us-gaap:Cash",
    "ifrs-full:CashAndCashEquivalents",
    "ifrs-full:Cash",
)

_OPCF_CONCEPTS = (
    "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    "ifrs-full:CashFlowsFromUsedInOperatingActivities",
)

_MAX_YEARS = 10
_IDENTITY_SET = False


@dataclass(frozen=True)
class CashPoint:
    end: date
    value_usd: float
    fy: int
    fp: str          # 'FY' / 'Q1' / 'Q2' / 'Q3'
    accession: str
    form: str
    native_currency: str
    native_value: float


@dataclass(frozen=True)
class CashHistory:
    series: list[CashPoint] = field(default_factory=list)
    latest_period_end: date | None = None
    latest_cash_usd: float | None = None
    op_cf_quarterly_usd: float | None = None   # negative = burn
    op_cf_prorated_usd: float | None = None    # prorated to today
    capital_raised_usd: float | None = None    # since latest period end
    current_cash_est_usd: float | None = None
    months_of_cash: float | None = None
    as_of: date = field(default_factory=date.today)
    stale_days: int | None = None              # days since latest_period_end
    fx_failed: bool = False


def fetch_cash_history(cik: int, *, as_of: date | None = None,
                       capital_raised_usd: float | None = None) -> CashHistory:
    """Pull historical cash + compute the bridge to a current-cash estimate.

    `capital_raised_usd` is an optional override for raises since the
    latest reporting date (computed elsewhere from the dilution ledger).
    """
    _ensure_identity()
    as_of = as_of or date.today()

    try:
        facts = Company(int(cik)).get_facts()
    except Exception as e:
        log.warning("get_facts failed for CIK %s: %s", cik, e)
        return CashHistory(as_of=as_of)

    series, fx_failed = _build_series(facts, as_of)
    if not series:
        return CashHistory(as_of=as_of, fx_failed=fx_failed)

    latest = series[-1]
    op_cf_q = _latest_quarterly_opcf(facts, latest.end)

    days_since = (as_of - latest.end).days
    op_cf_prorated = None
    if op_cf_q is not None:
        # 90-day quarter as the proration base, matching DT's tooltip text
        # ("Quarterly CF from operations prorated by days since the latest
        # reporting date").
        op_cf_prorated = op_cf_q * (days_since / 90.0)

    current_est = latest.value_usd
    if op_cf_prorated is not None:
        current_est = current_est + op_cf_prorated
    if capital_raised_usd is not None:
        current_est = current_est + capital_raised_usd
    # Intentionally NOT clamped to 0 — a negative current_est is the
    # most important runway signal and DT shows it as a sub-zero bar.

    months = None
    if op_cf_q is not None and op_cf_q < 0:
        monthly_burn = -op_cf_q / 3.0
        if monthly_burn > 0:
            # Allow negative months when current_est is already below zero —
            # matches DT's "-10.3 months of cash left" wording.
            months = current_est / monthly_burn

    return CashHistory(
        series=series,
        latest_period_end=latest.end,
        latest_cash_usd=latest.value_usd,
        op_cf_quarterly_usd=op_cf_q,
        op_cf_prorated_usd=op_cf_prorated,
        capital_raised_usd=capital_raised_usd,
        current_cash_est_usd=current_est,
        months_of_cash=months,
        as_of=as_of,
        stale_days=days_since,
        fx_failed=fx_failed,
    )


def _ensure_identity() -> None:
    global _IDENTITY_SET
    if _IDENTITY_SET:
        return
    set_identity(getattr(config, "EDGAR_IDENTITY", "dilution-tracker contact@example.com"))
    _IDENTITY_SET = True


def _build_series(facts, as_of: date) -> tuple[list[CashPoint], bool]:
    raw = _query_first_with_facts(facts, _CASH_CONCEPTS)
    if not raw:
        return [], False

    # Dedup by period_end keeping latest filing_date (restatements).
    by_end: dict[date, object] = {}
    for f in raw:
        end = _as_date(f.period_end)
        if end is None:
            continue
        existing = by_end.get(end)
        if existing is None or _as_date(f.filing_date) > _as_date(existing.filing_date):
            by_end[end] = f

    cutoff = date(as_of.year - _MAX_YEARS, as_of.month, 1)
    fx_failed = False
    points: list[CashPoint] = []
    for end in sorted(by_end.keys()):
        if end < cutoff:
            continue
        f = by_end[end]
        native = float(f.numeric_value)
        unit = (f.unit or "USD").upper()
        usd = native if unit == "USD" else fx.to_usd(native, unit, end)
        if usd is None:
            fx_failed = True
            continue
        points.append(CashPoint(
            end=end,
            value_usd=usd,
            fy=int(f.fiscal_year) if f.fiscal_year else end.year,
            fp=str(f.fiscal_period or ""),
            accession=str(f.accession or ""),
            form=str(f.form_type or ""),
            native_currency=unit,
            native_value=native,
        ))
    return points, fx_failed


def _latest_quarterly_opcf(facts, period_end: date) -> float | None:
    """Get the most recent ~quarterly operating cash flow value in USD.

    OpCF is a duration concept. Annual (FY) facts cover ~365 days and we
    divide by 4 to approximate a quarter; quarterly facts (period_length
    ~90d) are used as-is. We pick the most recent period that ends on or
    before `period_end` so the "burn rate" lines up with the latest cash
    balance.
    """
    raw = _query_first_with_facts(facts, _OPCF_CONCEPTS)
    if not raw:
        return None

    candidates = []
    for f in raw:
        end = _as_date(f.period_end)
        start = _as_date(f.period_start)
        if end is None or start is None or end > period_end:
            continue
        days = (end - start).days
        if days <= 0:
            continue
        candidates.append((end, days, f))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]))  # latest end first; prefer longer (FY) on tie
    end, days, f = candidates[-1]

    native = float(f.numeric_value)
    unit = (f.unit or "USD").upper()
    usd = native if unit == "USD" else fx.to_usd(native, unit, end)
    if usd is None:
        return None
    # Normalize to a 90-day quarter.
    return usd * (90.0 / days)


def _query_first_with_facts(facts, concepts) -> list:
    for c in concepts:
        try:
            r = facts.query().by_concept(c, exact=True).execute()
        except Exception as e:
            log.debug("query failed for %s: %s", c, e)
            continue
        if r:
            return r
    return []


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


@lru_cache(maxsize=256)
def _cached(cik: int, as_of_iso: str, capital_raised_usd: float | None) -> CashHistory:
    return fetch_cash_history(int(cik), as_of=date.fromisoformat(as_of_iso),
                              capital_raised_usd=capital_raised_usd)


def fetch_cash_history_cached(cik: int, *, as_of: date | None = None,
                              capital_raised_usd: float | None = None) -> CashHistory:
    return _cached(int(cik), (as_of or date.today()).isoformat(), capital_raised_usd)
