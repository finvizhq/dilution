"""Capital raised since a given date, summed from the dilution ledger.

Powers the light-blue overlay on the cash-position chart: how much cash
flowed in from offerings (ATM draws, shelf takedowns, equity-line sales,
registered directs) between the latest balance-sheet date and today.
The walker records each takedown in `dilution_ledger_drawdowns` with a
USD amount; we just sum. Off-shelf equity placements (PIPEs) write a
drawdown row at CLOSING (create_equity.closing_date / confirm_closing)
— a signed-but-pending SPA is deliberately NOT counted until the
closing is disclosed.

S-1 closings live in the ledger's history_json rather than the drawdowns
table, so we miss those for v1. Undercounts are safer than overcounts
here — an inflated current-cash estimate would understate runway risk.
"""
from __future__ import annotations

import logging
from datetime import date

from db import get_conn

log = logging.getLogger(__name__)


def capital_raised_since(cik: int, since: date) -> float | None:
    """Sum USD proceeds from drawdowns with event_date strictly after `since`.

    Returns 0.0 if no raises have happened. Returns None on DB error so
    the chart can omit the overlay rather than show a misleading zero.
    """
    try:
        with get_conn() as conn:
            row = conn.execute(
                """SELECT COALESCE(SUM(amount_usd), 0)
                   FROM dilution_ledger_drawdowns
                   WHERE cik = ? AND event_date > ? AND amount_usd IS NOT NULL""",
                (int(cik), since.isoformat()),
            ).fetchone()
    except Exception as e:
        log.warning("capital_raised_since failed for CIK %s: %s", cik, e)
        return None
    return float(row[0] or 0.0)
