"""Fee-table primary-vs-resale classifier.

Deterministic regex classification of the EX-FILING FEES exhibit
attached to S-1 / S-3 / F-1 / F-3 (and 424B-family) filings.
"""
from __future__ import annotations

import re

from db import get_conn


# ─── Fee-table primary-vs-resale classifier ─────────────────────────
# Deterministic regex classification of the EX-FILING FEES exhibit
# attached to S-1 / S-3 / F-1 / F-3 (and 424B-family) filings. The fee
# table's "Fee Calculation Rule" cells carry an SEC rule code that
# unambiguously identifies primary vs resale:
#
#   Primary (issuer raises new capital):
#     457(o) — aggregate-cap shelf (most common)
#     457(r) — WKSI auto-effective shelf, fee deferred to takedown
#     457(a) — specific-share registration at a fixed price
#     457(b)/(f)/(h)/(i) — variants (carry-forward, M&A, ESOP, convertible)
#   Resale (selling stockholders only — issuer gets no proceeds):
#     457(c) — selling-stockholder registration at market price
#     457(g) — warrant-on-warrant resale
#   Neutral marker (don't classify):
#     457(p) — fee-offset reference (appears alongside other rules)
#
# Complements registration_family.classify_424b_attribution: file_number
# classifies 424B takedowns by SEC-canonical linkage to a primary set;
# fee-table classifies registration statements (S-3/F-3/S-1/F-1) the
# file_number can't pre-screen (those create primary shelves so their
# own file_number isn't in the primary_set yet), and fills the file_
# number prescreen's "unknown" gap for 424B variants.
from typing import Literal as _Literal

FeeTableVerdict = _Literal["primary", "resale", "mixed", "unknown"]

# Cell-anchored: rule code appears in a markdown table cell ("| Rule
# 457(o) |"). Definitive — that IS the line item's classification.
_RE_RES_CELL = re.compile(
    r"\|[\s\n]*(?:Rule\s+)?457\(\s*[cg]\s*\)", re.IGNORECASE
)
_RE_PRI_CELL = re.compile(
    r"\|[\s\n]*(?:Rule\s+)?457\(\s*[oraibhfi]\s*\)", re.IGNORECASE
)
# Loose: rule code anywhere in the exhibit (including footnote prose).
# Used only as a fallback when the cell regex finds nothing — some fee
# tables format the rule on a continuation line a pipe doesn't anchor.
_RE_RES_LOOSE = re.compile(r"\b457\(\s*[cg]\s*\)", re.IGNORECASE)
_RE_PRI_LOOSE = re.compile(r"\b457\(\s*[oraibhfi]\s*\)", re.IGNORECASE)

_FEE_TABLE_DOC_TYPE = "EX-FILING FEES"

# Forms that meaningfully carry a fee table. Gating the DB query to
# these forms avoids an extra roundtrip on every periodic / 8-K filing.
FEE_TABLE_FORMS = frozenset({
    "S-1", "S-1/A", "S-1MEF",
    "F-1", "F-1/A", "F-1MEF",
    "S-3", "S-3/A", "S-3ASR", "S-3MEF",
    "F-3", "F-3/A", "F-3ASR", "F-3MEF",
    "S-4", "S-4/A", "F-4", "F-4/A",
    "F-10", "F-10/A", "F-10EF",
    "424B3", "424B4", "424B5", "424B8", "SUPPL",
    "POS AM", "POSASR",
    "S-11", "S-11/A",
})


def _classify_fee_table_text(text: str) -> FeeTableVerdict:
    """Pure-text classification — separated for testability.

    Two-pass: prefer cell-anchored rule mentions (definitive table-row
    classification); fall back to loose match when no cell-anchored
    rule is found (covers exhibits whose continuation-line cell format
    a pipe lookbehind doesn't catch).
    """
    if not text:
        return "unknown"
    cell_r = bool(_RE_RES_CELL.search(text))
    cell_p = bool(_RE_PRI_CELL.search(text))
    if cell_r and cell_p:
        return "mixed"
    if cell_r:
        return "resale"
    if cell_p:
        return "primary"
    loose_r = bool(_RE_RES_LOOSE.search(text))
    loose_p = bool(_RE_PRI_LOOSE.search(text))
    if loose_r and loose_p:
        return "mixed"
    if loose_r:
        return "resale"
    if loose_p:
        return "primary"
    return "unknown"


def classify_fee_table(accession: str) -> FeeTableVerdict:
    """Classify a filing as primary / resale / mixed / unknown based on
    its EX-FILING FEES exhibit's rule codes.

    Returns "unknown" when no fee-table exhibit is attached (typical for
    periodics, 8-Ks, and other non-registration forms) — callers should
    treat "unknown" as a no-op, not as evidence of anything.
    """
    with get_conn() as conn:
        row = conn.execute(
            """SELECT content_md FROM dilution_raw
                WHERE accession_number = ? AND doc_type = ?
                LIMIT 1""",
            (accession, _FEE_TABLE_DOC_TYPE),
        ).fetchone()
    if row is None:
        return "unknown"
    return _classify_fee_table_text(row["content_md"] or "")


def format_fee_table_for_prompt(verdict: FeeTableVerdict) -> str:
    """Render the fee-table verdict as a markdown hint block for the
    walker user prompt. Returns "" for `unknown` so the section is
    suppressed when there's no signal to convey."""
    if verdict == "primary":
        return (
            "## Fee-table classification\n\n"
            "This filing's EX-FILING FEES exhibit uses Rule "
            "457(o)/(r)/(a)/... — **PRIMARY** registration. The issuer "
            "is raising new capital; proceed with create_shelf / "
            "create_s1_offering / record_drawdown as appropriate.\n"
        )
    if verdict == "resale":
        return (
            "## Fee-table classification\n\n"
            "This filing's EX-FILING FEES exhibit uses Rule 457(c)/(g) — "
            "**RESALE** registration. The shares already exist on the "
            "ledger as the underlying instrument; the issuer raises no "
            "new capital. Do NOT call create_shelf or create_s1_offering. "
            "Call note_no_event(reason=\"resale registration\").\n"
        )
    if verdict == "mixed":
        return (
            "## Fee-table classification\n\n"
            "This filing's EX-FILING FEES exhibit uses BOTH primary "
            "(457(o)/(r)/...) AND resale (457(c)/(g)) rules — "
            "**COMBINED** primary+resale registration. Emit create_shelf "
            "(or create_s1_offering) for the primary section's newly-"
            "registered securities; the resale section's selling-"
            "stockholder shares are derived downstream from the "
            "file_number — do not call create_* for them.\n"
        )
    return ""


__all__ = [
    "FeeTableVerdict",
    "classify_fee_table",
    "format_fee_table_for_prompt",
    "FEE_TABLE_FORMS",
]
