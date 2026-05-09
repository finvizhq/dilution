"""Render the open ledger as compact text for the walker LLM prompt.

The walker needs to show the LLM what the issuer's cap table looks
like *right now* so the model can correctly emit `amend_instrument`
and `record_event` against existing ids rather than duplicating with
fresh `create_instrument` calls. The view is the contract between
the ledger (state) and the LLM (which produces mutations).

Constraints:
  - Always show every active instrument (those are the ones the LLM
    might mutate).
  - Show recently-closed instruments (default 180d) at full detail —
    a closed warrant from 6 months ago might still be referenced by
    name in the new filing.
  - Show older closed as one-line summaries so the LLM has the
    historical context without burning tokens.
  - Hard cap on total chars; if exceeded, collapse same-strike same-
    counterparty warrant tranches into a `[summary]` row with id range.
  - NEVER truncate active rows. If active alone exceeds the cap, log
    ERROR and truncate oldest-first; this only trips for the worst
    serial diluters.

The Grok-4-fast 2M-token context easily holds 60K of view + 1.5M of
filing text + system, so the cap is a prompt-tidiness floor, not a
context-window concern.
"""

from __future__ import annotations

import json
import logging
from datetime import date as _d
from typing import Iterable

log = logging.getLogger(__name__)


# Width budget for one rendered row. Keeping rows under ~200 chars
# means a ledger of 100 instruments fits in ~20K, well under the cap.
ROW_BUDGET = 200

DEFAULT_MAX_CHARS = 60_000

# Recently-closed window for detailed inclusion. Older closed rows
# appear as one-line summaries; closed > ~3y ago drop entirely.
RECENT_CLOSED_DAYS = 180
ARCHIVE_CLOSED_DAYS = 365 * 3


def render_ledger_view(
    rows: Iterable[dict],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    today: _d | None = None,
    drawdowns_by_instrument: dict[str, list[dict]] | None = None,
) -> str:
    """Render rows (dicts as returned by store.get_open_instruments) into
    the prompt text block.

    Layout:
        ## Open ledger
        type        id       created     counterparty  terms                                           outstanding                          status
        warrant     W-001    2024-03-15  Aegis         strike=2.50 term=5y units=common               count=2,000,000                      active
        shelf       SH-002   2024-04-25  —             capacity_usd=80,000,000 form=F-3                drawn_usd=8,854,039                  active
            takedowns: 2025-11-17 $8,854,039 (11,067,547 sh) [Maxim]; 2024-08-02 $2,500,000
        ...
        ## Recently closed
        warrant     W-002    2024-08-02  Maxim         strike=1.10 term=5y                            count=0 exercised_to_date=1,200,000  exercised(2025-02-10)
        ## Older closed (summary)
        - W-005..W-008: 4 warrants, all expired between 2022-06 and 2023-04
    """
    today = today or _d.today()
    rows = list(rows or [])
    dd = drawdowns_by_instrument or {}

    actives, recent_closed, archived = _bucket(rows, today)
    body = _render_buckets(actives, recent_closed, archived, dd)
    if len(body) <= max_chars:
        return body

    # First compaction: drop archived bucket entirely.
    log.info("ledger view %d chars > cap %d — dropping archived",
             len(body), max_chars)
    body = _render_buckets(actives, recent_closed, [], dd)
    if len(body) <= max_chars:
        return body

    # Second compaction: collapse same-counterparty same-strike warrant
    # tranches in the recent_closed bucket into id-range summaries.
    log.info("ledger view %d chars > cap %d — collapsing closed warrants",
             len(body), max_chars)
    collapsed_recent = _collapse_warrants(recent_closed)
    body = _render_buckets(actives, collapsed_recent, [], dd)
    if len(body) <= max_chars:
        return body

    # Third compaction: collapse active warrants too (more aggressive).
    log.info("ledger view %d chars > cap %d — collapsing active warrants",
             len(body), max_chars)
    collapsed_active = _collapse_warrants(actives)
    body = _render_buckets(collapsed_active, collapsed_recent, [], dd)
    if len(body) <= max_chars:
        return body

    # Last resort: hard-truncate oldest-first within the active bucket.
    # Should be vanishingly rare. If we hit this, the prompt is going
    # to omit instruments and the walker may emit duplicate creates.
    log.error("ledger view active alone exceeds cap %d — truncating "
              "oldest-first; walker may misclassify instruments", max_chars)
    actives_sorted = sorted(
        collapsed_active,
        key=lambda r: r.get("created_at") or "0000-00-00",
        reverse=True,  # newest first; we cut from the back
    )
    while actives_sorted and len(
        _render_buckets(actives_sorted, collapsed_recent, [], dd)
    ) > max_chars:
        actives_sorted.pop()
    return _render_buckets(actives_sorted, collapsed_recent, [], dd)


def _bucket(rows, today):
    actives, recent_closed, archived = [], [], []
    cutoff_recent = _d.fromordinal(
        today.toordinal() - RECENT_CLOSED_DAYS).isoformat()
    cutoff_archive = _d.fromordinal(
        today.toordinal() - ARCHIVE_CLOSED_DAYS).isoformat()
    for r in rows:
        status = (r.get("status") or "active").lower()
        if status == "active":
            actives.append(r)
            continue
        status_at = r.get("status_at") or ""
        if status_at >= cutoff_recent:
            recent_closed.append(r)
        elif status_at >= cutoff_archive:
            archived.append(r)
        # else: too old — drop entirely
    actives.sort(key=lambda r: (r.get("type") or "",
                                r.get("created_at") or "",
                                r.get("instrument_id")))
    recent_closed.sort(key=lambda r: (r.get("status_at") or ""), reverse=True)
    archived.sort(key=lambda r: (r.get("status_at") or ""), reverse=True)
    return actives, recent_closed, archived


def _render_buckets(actives, recent_closed, archived, drawdowns=None) -> str:
    drawdowns = drawdowns or {}
    parts = []
    if actives:
        parts.append("## Open ledger\n")
        parts.append(_header())
        for r in actives:
            parts.append(_render_row(r))
            tline = _render_takedowns(r, drawdowns)
            if tline:
                parts.append(tline)
    else:
        parts.append("## Open ledger\n(no active instruments)\n")
    if recent_closed:
        parts.append("\n## Recently closed (last 180 days)\n")
        parts.append(_header())
        for r in recent_closed:
            parts.append(_render_row(r))
            tline = _render_takedowns(r, drawdowns)
            if tline:
                parts.append(tline)
    if archived:
        parts.append("\n## Older closed (summary)\n")
        for r in archived:
            parts.append(_render_archive_row(r))
    return "".join(parts)


def _header() -> str:
    return ("type        id                     created     "
            "counterparty   terms                              "
            "outstanding                  status\n")


def _render_row(r: dict) -> str:
    """One detail row.

    Format keeps to one line, ~200 chars, so 100 instruments fit in
    ~20K. The terms / outstanding fields are flattened key=value to
    keep the LLM's parsing trivial.
    """
    type_ = (r.get("type") or "")[:11].ljust(11)
    iid = (r.get("instrument_id") or "")[:22].ljust(22)
    created = (r.get("created_at") or "")[:10].ljust(10)
    cp = _short_counterparty(r) or "—"
    cp_disp = cp[:13].ljust(13)
    terms = _flatten_kv(_to_dict(r, "terms"))[:34].ljust(34)
    outstanding = _flatten_kv(_to_dict(r, "outstanding"))[:28].ljust(28)
    status_disp = _format_status(r)
    return (f"{type_} {iid} {created} {cp_disp} {terms} {outstanding} "
            f"{status_disp}\n")


# Per-instrument cap for inline takedowns. Older entries are summarized.
_TAKEDOWN_HEAD = 7


def _render_takedowns(r: dict, drawdowns: dict) -> str:
    """Continuation line of recorded takedowns, newest-first.

    Only emitted for instruments that can have drawdowns (shelf, ATM,
    equity_line). The walker uses these to dedup re-disclosures of the
    same takedown across multiple filings (signing 6-K, closing 6-K,
    quarterly recap).
    """
    if (r.get("type") or "") not in ("shelf", "atm", "equity_line"):
        return ""
    iid = r.get("instrument_id")
    items = drawdowns.get(iid) if iid else None
    if not items:
        return ""
    head = items[:_TAKEDOWN_HEAD]
    tail_n = len(items) - len(head)
    rendered = "; ".join(_fmt_takedown(t) for t in head)
    if tail_n > 0:
        rendered += f"; +{tail_n} earlier"
    return f"    takedowns: {rendered}\n"


def _fmt_takedown(t: dict) -> str:
    """Compact `YYYY-MM-DD $amount (Nsh @price) [counterparty]`."""
    parts = [t.get("event_date") or "?"]
    amt = t.get("amount_usd")
    if amt:
        parts.append(f"${_fmt_val(float(amt))}")
    sh = t.get("shares")
    px = t.get("price")
    sh_px = []
    if sh:
        sh_px.append(f"{_fmt_val(float(sh))} sh")
    if px:
        sh_px.append(f"@${_fmt_val(float(px))}")
    if sh_px:
        parts.append(f"({' '.join(sh_px)})")
    cp = t.get("counterparty_canonical") or t.get("counterparty")
    if cp:
        cp = cp.strip()
        if len(cp) > 24:
            cp = cp[:24]
        parts.append(f"[{cp}]")
    return " ".join(parts)


def _render_archive_row(r: dict) -> str:
    iid = r.get("instrument_id") or "?"
    type_ = r.get("type") or "?"
    closed_at = r.get("status_at") or ""
    status = r.get("status") or "closed"
    cp = _short_counterparty(r) or "—"
    return f"- {iid} {type_} {cp} {status} {closed_at}\n"


def _format_status(r: dict) -> str:
    status = r.get("status") or "active"
    if status == "active":
        return "active"
    when = r.get("status_at") or ""
    return f"{status}({when})" if when else status


def _short_counterparty(r: dict) -> str | None:
    """Prefer the canonical short label; fall back to the raw name."""
    cp = r.get("counterparty_canonical") or r.get("counterparty")
    if not cp:
        return None
    cp = cp.strip()
    return cp


_TERMS_KEY_ORDER = (
    "strike", "warrant_strike", "conv_price", "conversion_price",
    "term_years", "maturity",
    "principal", "rate", "coupon", "oid_pct",
    "capacity_usd", "capacity_shares",
    "stated_value", "liquidation_preference", "dividend_rate",
    "anti_dilution_type", "is_pre_funded",
    "units",
)
_OUTSTANDING_KEY_ORDER = (
    "count", "principal_remaining", "remaining_capacity_usd",
    "drawn_usd", "sold_to_date",
    "exercised_to_date", "principal_converted_to_date",
)


def _flatten_kv(d: dict) -> str:
    """Render a small dict as `k=v k=v` in stable order. Numbers are
    formatted compactly; strings are passed through; lists/dicts
    stringified to JSON. Keys are emitted in a curated order so the
    most-relevant fields show first regardless of dict order."""
    if not d:
        return "—"
    seen: set[str] = set()
    parts: list[str] = []
    for k in _TERMS_KEY_ORDER + _OUTSTANDING_KEY_ORDER:
        if k in d and d[k] is not None:
            parts.append(f"{k}={_fmt_val(d[k])}")
            seen.add(k)
    # Spill remaining keys (non-canonical) so the LLM still sees them.
    for k, v in d.items():
        if k in seen or v is None:
            continue
        if k == "applied_splits":
            # Compress the splits list to a count to keep the row tidy.
            parts.append(f"applied_splits={len(v)}")
            continue
        parts.append(f"{k}={_fmt_val(v)}")
    return " ".join(parts) if parts else "—"


def _fmt_val(v) -> str:
    if isinstance(v, bool):
        return "Y" if v else "N"
    if isinstance(v, (int, float)):
        if v == 0:
            return "0"
        ax = abs(v)
        if ax >= 1_000_000:
            return f"{v:,.0f}"
        if ax >= 1:
            # 4 sig figs for prices, full int for counts ≥ 1000
            if ax >= 1000:
                return f"{v:,.0f}"
            return f"{v:.4g}"
        return f"{v:.4g}"
    if isinstance(v, (list, dict)):
        return json.dumps(v, separators=(",", ":"))
    return str(v)


def _to_dict(r: dict, key: str) -> dict:
    """Decoded dict for the row's terms/outstanding. Tolerates both
    pre-decoded snapshots (key="terms") and raw rows (key+"_json")."""
    val = r.get(key)
    if isinstance(val, dict):
        return val
    raw = r.get(f"{key}_json")
    if raw:
        try:
            return json.loads(raw) or {}
        except (TypeError, ValueError):
            return {}
    return {}


def _collapse_warrants(rows: list[dict]) -> list[dict]:
    """Fold same-counterparty + same-strike warrant tranches into a
    single summary row. Used as a compaction step when the rendered
    view exceeds the cap. Non-warrant rows pass through unchanged."""
    bucket: dict[tuple, list[dict]] = {}
    other: list[dict] = []
    for r in rows:
        if (r.get("type") or "") != "warrant":
            other.append(r)
            continue
        terms = _to_dict(r, "terms")
        strike = terms.get("strike") or terms.get("warrant_strike")
        cp = _short_counterparty(r) or "—"
        key = (round(float(strike), 4) if strike else None, cp,
               r.get("status") or "active")
        bucket.setdefault(key, []).append(r)
    out = list(other)
    for (strike, cp, status), members in bucket.items():
        if len(members) == 1:
            out.append(members[0])
            continue
        # Collapse — pick id range, sum counts
        members.sort(key=lambda m: m.get("instrument_id") or "")
        first = members[0]["instrument_id"]
        last = members[-1]["instrument_id"]
        total_count = sum(
            float(_to_dict(m, "outstanding").get("count") or 0)
            for m in members
        )
        merged = dict(members[0])
        merged["instrument_id"] = f"{first}..{last}"
        merged["counterparty_canonical"] = cp
        merged_terms = dict(_to_dict(members[0], "terms"))
        if strike is not None:
            merged_terms["strike"] = strike
        merged["terms"] = merged_terms
        merged["outstanding"] = {"count": total_count,
                                 "tranche_count": len(members)}
        merged["status"] = status
        out.append(merged)
    out.sort(key=lambda r: (r.get("type") or "",
                            r.get("created_at") or "",
                            r.get("instrument_id") or ""))
    return out


__all__ = [
    "DEFAULT_MAX_CHARS",
    "render_ledger_view",
]
