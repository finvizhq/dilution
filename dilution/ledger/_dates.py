"""Shared date coercion for the ledger status derivers.

``derive_s1_status`` and ``derive_shelf_status`` are mirror status machines
that compare ISO date strings and accept a caller-supplied ``today``. The
date fields they read arrive in several shapes: a bare ``'YYYY-MM-DD'``
string, a full ISO timestamp (the schema's ``created_at`` default
``'...T00:00:00Z'``), a ``date``, or a ``datetime``. This normalizes all of
them to a calendar :class:`datetime.date` so the lapse / expiry math is
robust regardless of the source shape — keeping the two mirrors consistent.
"""

from __future__ import annotations

from datetime import date as _date, datetime as _datetime


def coerce_date(value):
    """Best-effort calendar date from a date, datetime, or ISO string.

    Trims a trailing ``'Z'`` and any time component (so a timestamp like
    ``'2026-01-01T00:00:00Z'`` parses to its date head). Returns ``None``
    when the value can't be parsed, so callers can fall back
    deterministically. ``datetime`` is checked first since it subclasses
    ``date``.
    """
    if value is None:
        return None
    if isinstance(value, _datetime):
        return value.date()
    if isinstance(value, _date):
        return value
    s = str(value).strip().replace("Z", "")
    if not s:
        return None
    try:
        return _date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


__all__ = ["coerce_date"]
