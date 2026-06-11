"""Render the open ledger as compact text for the walker LLM prompt.

The walker needs to show the LLM what the issuer's cap table looks
like *right now* so the model can correctly emit `amend_instrument`
and `record_event` against existing ids rather than duplicating with
fresh `create_instrument` calls. The view is the contract between
the ledger (state) and the LLM (which produces mutations).

Layout: each instrument type renders as its own markdown table
under an H3 header (`### Warrants`, `### Preferred stock`, …) in
INSTRUMENT_TYPES order. Per-section columns are chosen from a
per-type priority list, and any column that is null on every row in
the section is dropped — that keeps warrants tables free of an
always-empty counterparty column (NULLING RULE: most warrants are
issued to anonymous "institutional investors") without breaking the
case where one warrant does name a counterparty.

The previous fixed-width renderer silently truncated `terms` at 34
chars and `outstanding` at 28 chars, which lost long field values
(rendered truncated mid-word) and similar. The new layout has no
per-field cap; column widths size to content.

Constraints:
  - Always show every active instrument.
  - Show recently-closed instruments (default 365d) at full detail.
  - Show older closed as one-line summaries.
  - Hard cap on total chars; if exceeded, collapse same-strike
    same-counterparty warrant tranches into a `[summary]` row.
  - NEVER truncate a field value mid-render; only drop whole rows
    (oldest-first) as a last resort when the active bucket alone
    exceeds cap.

The Grok-4-fast 2M-token context easily holds 60K of view + 1.5M of
filing text + system, so the cap is a prompt-tidiness floor, not a
context-window concern.
"""

from __future__ import annotations

import json
import logging
from datetime import date as _d
from typing import Iterable

from ..schema import INSTRUMENT_TYPES

log = logging.getLogger(__name__)


DEFAULT_MAX_CHARS = 60_000

# Recently-closed window for detailed inclusion. Older closed rows
# appear as one-line summaries; closed > ~3y ago drop entirely.
RECENT_CLOSED_DAYS = 365
ARCHIVE_CLOSED_DAYS = 365 * 3


# H3 label per instrument type. Plural to match 10-Q footnote idiom
# ("Note 13 — Warrants"), the closest training distribution for the
# model.
_TYPE_LABELS = {
    "warrant": "Warrants",
    "convertible": "Convertibles",
    "preferred": "Preferred stock",
    "atm": "ATMs",
    "equity_line": "Equity lines",
    "shelf": "Shelves",
    "s1_offering": "S-1 offerings",
    "equity": "Equity placements",
}

# Columns per instrument type, in display order. Anchor columns
# (id, created) are always rendered first; status is appended only
# when the section is the "Recently closed" bucket. A column whose
# value is None on every row in the section is dropped.
_TYPE_COLUMNS = {
    "warrant":     ["strike", "count",
                    "counterparty", "placement_agent", "flags"],
    "convertible": ["conv_price", "rate", "maturity",
                    "principal_remaining",
                    "counterparty", "flags"],
    "preferred":   ["conv_price", "series_letter",
                    "count", "principal_remaining", "counterparty",
                    "flags"],
    "atm":         ["agreement_date", "capacity_usd", "drawn_usd",
                    "remaining_capacity_usd", "placement_agent"],
    "equity_line": ["agreement_date", "capacity_usd", "drawn_usd",
                    "remaining_capacity_usd", "counterparty"],
    "shelf":       ["form", "capacity_usd", "drawn_usd",
                    "remaining_capacity_usd"],
    "s1_offering": ["anticipated_deal_size", "warrant_strike",
                    "warrant_coverage_pct", "sold_to_date",
                    "placement_agent"],
    "equity":      ["counterparty", "sold_to_date"],
}

# Fallback if a new type ships before this table is updated. Render
# the row anyway with minimal columns so it isn't silently dropped.
_DEFAULT_COLUMNS = ["counterparty"]

# Display header per column key. Keep narrow to keep table width
# down for human reviewers (the LLM tolerates either).
_COLUMN_HEADERS = {
    "id": "id",
    "created": "created",
    "strike": "strike",
    "conv_price": "conv_price",
    "principal_remaining": "principal_rem",
    "remaining_capacity_usd": "remaining_usd",
    "anticipated_deal_size": "deal_size",
    "warrant_coverage_pct": "wt_coverage",
    "warrant_strike": "wt_strike",
    "series_letter": "series",
    "placement_agent": "placement_agent",
    "counterparty": "counterparty",
    "flags": "flags",
    "form": "form",
    "rate": "rate",
    "maturity": "maturity",
    "agreement_date": "agreement",
    "capacity_usd": "capacity_usd",
    "drawn_usd": "drawn_usd",
    "sold_to_date": "sold_to_date",
    "exercised_to_date": "exercised",
    "count": "count",
    "status": "status",
}


def render_ledger_view(
    rows: Iterable[dict],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    today: _d | None = None,
    drawdowns_by_instrument: dict[str, list[dict]] | None = None,
) -> str:
    """Render rows (dicts as returned by store.get_open_instruments) into
    the prompt text block.

    Layout (one section per instrument type, in INSTRUMENT_TYPES order):
        ## Open ledger

        ### Warrants
        | id    | created    | strike | count      | flags      |
        |-------|------------|--------|------------|------------|
        | W-005 | 2020-11-30 |   1.25 |  8,000,000 |            |
        | W-006 | 2020-11-30 |  0.001 |  3,000,000 | pre-funded |

        ### Shelves
        | id     | created    | form | capacity_usd | drawn_usd | remaining_usd |
        |--------|------------|------|--------------|-----------|---------------|
        | SH-002 | 2024-04-25 | F-3  |   80,000,000 | 8,854,039 |    71,145,961 |

          ↳ SH-002 takedowns: 2025-11-17 $8,854,039 (11,067,547 sh) [Maxim]; …

        ## Recently closed (last 365 days)
        ### Warrants
        | id    | created    | strike | count | status              |
        | W-002 | 2024-08-02 |   1.10 |     0 | exercised(2025-02-10) |

        ## Older closed (summary)
        - W-005 warrant — exercised 2022-06-15
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
                                r.get("instrument_id") or ""))
    recent_closed.sort(key=lambda r: (r.get("status_at") or ""), reverse=True)
    archived.sort(key=lambda r: (r.get("status_at") or ""), reverse=True)
    return actives, recent_closed, archived


def _render_buckets(actives, recent_closed, archived, drawdowns=None) -> str:
    drawdowns = drawdowns or {}
    parts: list[str] = []
    if actives:
        parts.append("## Open ledger\n\n")
        for type_ in INSTRUMENT_TYPES:
            section_rows = [r for r in actives
                            if (r.get("type") or "") == type_]
            section = _render_section(
                type_, section_rows,
                include_status=False, drawdowns=drawdowns,
            )
            if section:
                parts.append(section)
    else:
        parts.append("## Open ledger\n(no active instruments)\n")
    if recent_closed:
        parts.append("\n## Recently closed (last 365 days)\n\n")
        for type_ in INSTRUMENT_TYPES:
            section_rows = [r for r in recent_closed
                            if (r.get("type") or "") == type_]
            section = _render_section(
                type_, section_rows,
                include_status=True, drawdowns=drawdowns,
            )
            if section:
                parts.append(section)
    if archived:
        parts.append("\n## Older closed (summary)\n")
        for r in archived:
            parts.append(_render_archive_row(r))
    return "".join(parts)


def _render_section(type_: str, rows: list[dict], *,
                    include_status: bool,
                    drawdowns: dict) -> str:
    """Render one type's rows as a markdown table. Empty rows list
    returns ''. Columns null across every row are dropped before
    rendering, so the table width tracks actually-populated columns.

    The takedowns continuation line for shelf/atm/equity_line rows
    is emitted below the table, id-prefixed so the per-row
    association survives the table sitting between them. The
    `takedowns:` marker word from the legacy format is preserved so
    the walker_prompt reference at walker_prompt.py:926 still holds."""
    if not rows:
        return ""
    type_cols = list(_TYPE_COLUMNS.get(type_) or _DEFAULT_COLUMNS)
    if include_status:
        type_cols.append("status")
    column_keys = ["id", "created"] + type_cols

    # Compute every cell up front. We re-use these to (a) drop empty
    # columns and (b) render the final table without recomputing.
    cell_rows = [[_format_cell(r, k) for k in column_keys] for r in rows]

    # Drop columns where every non-anchor cell is empty. id/created
    # are always kept (anchor columns at indices 0 and 1).
    keep_idx: list[int] = [0, 1]
    for i in range(2, len(column_keys)):
        if any(row[i] not in ("", "—") for row in cell_rows):
            keep_idx.append(i)
    kept_keys = [column_keys[i] for i in keep_idx]
    kept_cells = [[row[i] for i in keep_idx] for row in cell_rows]

    headers = [_COLUMN_HEADERS.get(k, k) for k in kept_keys]
    widths = [
        max(len(h), max((len(row[i]) for row in kept_cells), default=0))
        for i, h in enumerate(headers)
    ]

    parts: list[str] = []
    label = _TYPE_LABELS.get(type_, type_.replace("_", " ").title())
    parts.append(f"### {label}\n")
    parts.append(
        "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |\n"
    )
    parts.append(
        "|" + "|".join("-" * (w + 2) for w in widths) + "|\n"
    )
    for row_cells in kept_cells:
        parts.append(
            "| " + " | ".join(c.ljust(w) for c, w in zip(row_cells, widths))
            + " |\n"
        )

    # Takedowns continuation lives only on the three drawdown-bearing
    # types. id-prefixed so the per-row association is unambiguous.
    if type_ in ("shelf", "atm", "equity_line"):
        for r in rows:
            tline = _takedown_line(r, drawdowns)
            if tline:
                parts.append(tline)

    parts.append("\n")
    return "".join(parts)


def _format_cell(row: dict, key: str) -> str:
    if key == "id":
        return row.get("instrument_id") or "?"
    if key == "created":
        return (row.get("created_at") or "")[:10]
    if key == "status":
        return _format_status(row)
    if key == "flags":
        return _row_flags(row) or ""
    v = _get_field(row, key)
    if v is None:
        return "—"
    return _fmt_val(v)


def _get_field(row: dict, key: str):
    """Resolve a column key to its value. Top-level for synthesized
    columns (counterparty, placement_agent), then terms, then
    outstanding. Returns None for absent or null."""
    if key == "counterparty":
        return row.get("counterparty_canonical") or None
    if key == "placement_agent":
        return row.get("placement_agent_canonical") or None
    terms = _to_dict(row, "terms")
    if key in terms and terms[key] is not None:
        return terms[key]
    out = _to_dict(row, "outstanding")
    if key in out and out[key] is not None:
        return out[key]
    return None


def _row_flags(row: dict) -> str | None:
    """Synthesize a compact flags column from per-row signals that
    don't merit their own column."""
    terms = _to_dict(row, "terms")
    out = _to_dict(row, "outstanding")
    flags: list[str] = []
    if terms.get("is_pre_funded"):
        flags.append("pre-funded")
    n_splits = len(terms.get("applied_splits") or [])
    if n_splits:
        flags.append(f"splits×{n_splits}")
    tranche_count = out.get("tranche_count")
    if tranche_count:
        flags.append(f"×{int(tranche_count)} tranches")
    return ", ".join(flags) if flags else None


# Per-instrument cap for inline takedowns. Older entries are summarized.
_TAKEDOWN_HEAD = 7


def _takedown_line(r: dict, drawdowns: dict) -> str:
    """Continuation line of recorded takedowns for shelf/ATM/equity-line,
    newest-first. Emitted below the parent section's markdown table,
    id-prefixed so the walker (and prompt) can still tie a
    `takedowns:` listing back to its specific row even with the
    table sitting between them."""
    iid = r.get("instrument_id")
    items = drawdowns.get(iid) if iid else None
    if not items:
        return ""
    head = items[:_TAKEDOWN_HEAD]
    tail_n = len(items) - len(head)
    rendered = "; ".join(_fmt_takedown(t) for t in head)
    if tail_n > 0:
        rendered += f"; +{tail_n} earlier"
    return f"  ↳ {iid} takedowns: {rendered}\n"


def _fmt_takedown(t: dict) -> str:
    """Compact `YYYY-MM-DD $amount (Nsh @price) [party]`."""
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
    cp = t.get("drawdown_party_canonical")
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
    cp = r.get("counterparty_canonical")
    if not cp:
        return None
    return cp.strip()


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
