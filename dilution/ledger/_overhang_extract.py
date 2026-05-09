"""Periodic-filing overhang extraction for the seed and anchor passes.

Extracted from the legacy `dilution/overhang.py` during the cutover.
The walker invokes this for each 10-K / 10-Q / 20-F / 40-F:

  - seed.py runs it on the earliest periodic filing to populate the
    ledger with initial outstanding instruments.
  - walker.py runs it after each subsequent periodic filing and
    diffs the result against the ledger via anchor.reconcile_against_periodic.

Output shape is a list of cleaned dicts (one per outstanding
instrument) with keys: category, instrument_name, outstanding_count,
common_shares_issuable, strike_or_conversion_price, principal_amount,
maturity_or_expiry, issue_date, price_protection, pp_clause_text,
notes.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from db import get_conn
from dilution.llm_provider import system, user

from ._llm_utils import asample_and_check, make_chat

log = logging.getLogger(__name__)


# Cap matches walker_llm.MAX_INPUT_CHARS — periodic filings can be
# large (S-1/F-1 can hit 1M+ chars). Grok-4-fast handles this comfortably.
MAX_INPUT_CHARS = 2_000_000

# Stamped into provenance / version tracking. Bump when the prompt changes.
HANDLER_VERSION = "ledger-overhang-v1"


# ─── Schema (Pydantic) ──────────────────────────────────────────────
OVERHANG_CATEGORIES = (
    "warrant",
    "convertible",
    "preferred",
    "option_pool",
    "rsu_psu_unvested",
    "other",
)

_PRICE_PROTECTION = (
    "Customary Anti-Dilution",
    "variable_rate",
    "full_ratchet",
    "Alternate Cashless",
    "undisclosed",
)


class OverhangRow(BaseModel):
    """One outstanding instrument with potential dilution exposure."""

    model_config = ConfigDict(use_enum_values=True)

    category: str
    instrument_name: str | None = Field(None, max_length=200)
    outstanding_count: float | None = None
    common_shares_issuable: float | None = None
    strike_or_conversion_price: float | None = None
    principal_amount: float | None = None
    maturity_or_expiry: str | None = Field(None, max_length=30)
    issue_date: str | None = Field(None, max_length=30)
    price_protection: str | None = None
    pp_clause_text: str | None = Field(None, max_length=800)
    notes: str | None = Field(None, max_length=800)


class OverhangList(BaseModel):
    overhang: list[OverhangRow] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data):
        """Tolerate the wrapper shapes Moonshot occasionally returns
        (bare list, single bare row, alias key) instead of the canonical
        ``{"overhang": [...]}`` xAI structured outputs enforce."""
        if isinstance(data, list):
            return {"overhang": data}
        if not isinstance(data, dict) or "overhang" in data:
            return data
        if "category" in data:
            return {"overhang": [data]}
        list_fields = [v for v in data.values() if isinstance(v, list)]
        if len(list_fields) == 1:
            return {"overhang": list_fields[0]}
        return {"overhang": []}


# ─── Prompt ─────────────────────────────────────────────────────────
PROMPT = """\
You are extracting POTENTIAL DILUTION OVERHANG from an SEC periodic filing.

Context: this is the {form} for CIK {cik}, period of report {as_of_date}.

Extract one row per distinct outstanding instrument that could dilute
common stock if exercised/converted/vested.

Return a JSON object with this exact shape:

{{
  "overhang": [
    {{
      "category": "warrant",
      "instrument_name": "Series A Common Stock Warrants",
      "outstanding_count": 4500000,
      "common_shares_issuable": 4500000,
      "strike_or_conversion_price": 3.10,
      "maturity_or_expiry": "2029-08-12",
      "issue_date": "2024-08-12",
      "price_protection": "Customary Anti-Dilution",
      "pp_clause_text": null,
      "notes": null
    }}
  ]
}}

Return {{"overhang": []}} when the filing discloses no overhang.

Each row must have these keys (use null if not disclosed):
- category: one of {categories}
- instrument_name: descriptive name from the filing. When the filing
  identifies a lender / holder of a convertible note family, INCLUDE
  the lender name (e.g. "Streeterville Capital December 2022 Promissory
  Note") so downstream code can merge tranches across periods.
- outstanding_count: explicit count. For PREFERRED STOCK, extract
  SHARES OUTSTANDING (not shares authorized).
- common_shares_issuable: populate ONLY if the filing explicitly states
  the aggregate number of common shares (or ADSs) issuable on full
  exercise/conversion. Otherwise null — code derives it from
  principal_amount / strike_or_conversion_price.
- strike_or_conversion_price: USD per common share. Extract the
  CURRENT (as-of-period-end) adjusted price, NOT the initial price.
- principal_amount:
  * Convertible debt: principal outstanding.
  * Convertible preferred with stated aggregate redemption / liquidation
    value: that aggregate dollar amount.
  * Warrants / option pool / RSUs: null.
- maturity_or_expiry: YYYY-MM-DD or YYYY-only.
- issue_date: when first issued (YYYY-MM-DD).
- price_protection: apply these rules in order, take the FIRST match:
    1. Filing describes any reset / VWAP-linked formula / lookback /
       alternate-conversion-on-default / "lower of $X and Y% of VWAP" /
       floor below initial strike / repayable-in-shares-at-recent-price
       → "variable_rate". Set notes to a verbatim ≤2-sentence excerpt.
    2. Filing describes a full reset of strike / conversion price to
       the price of any subsequent dilutive issuance
       → "full_ratchet".
    3. Filing describes a net-share-settle / cashless formula NOT tied
       to Black-Scholes (e.g. "cashless after 60 days for 0.5 shares
       per warrant")
       → "Alternate Cashless".
    4. Filing describes ordinary adjustments for stock splits, stock
       dividends, recapitalizations, or similar — with no reset / VWAP
       / floor mechanism
       → "Customary Anti-Dilution".
    5. Filing is SILENT on anti-dilution provisions for this instrument
       → "undisclosed".
  "Customary Anti-Dilution" requires the filing to actually describe
  the standard adjustments. It is NOT a default for ambiguity.
  "undisclosed" is reserved for genuine silence — do NOT use it when
  the filing describes adjustments you could classify under rules 1-4.
- pp_clause_text: verbatim ≤2-sentence excerpt of the unusual clause
  (variable_rate / full_ratchet / Alternate Cashless). Leave null for
  Customary Anti-Dilution and undisclosed.
- notes: short context (floor/ceiling, call features, conversion-price
  adjustments).

Rules:
- Extract numbers VERBATIM. Never multiply, divide, sum, or transform.
- If multiple warrant tranches are listed separately, emit each as
  its own row.
- Distinguish CONVERTIBLE PREFERRED from accompanying WARRANTS. A
  common transaction issues both — emit TWO rows, never collapsed.
- Skip already-exercised / converted / cancelled instruments.
- If the filing discloses no overhang (rare for 10-K, possible for
  10-Q), return {{"overhang": []}}.

Filing text:
{text}
"""


# ─── Helpers ───────────────────────────────────────────────────────
def _load_filing_text(accession: str) -> str | None:
    """Pull the periodic filing's primary doc out of dilution_raw.
    Prefers the form-typed row over EX-* exhibits, falls back to longest."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT content_md, doc_type, LENGTH(content_md) AS len
                 FROM dilution_raw
                WHERE accession_number = ?
                ORDER BY len DESC""",
            (accession,),
        ).fetchall()
    if not rows:
        return None
    for r in rows:
        dt = (r["doc_type"] or "").upper()
        if not dt.startswith("EX-"):
            return r["content_md"]
    return rows[0]["content_md"]


def _parse_overhang_list(content: str, *, accession: str | None = None):
    """Whole-list validate; fall back to row-by-row salvage on
    ValidationError so one bad row doesn't drop the rest."""
    try:
        return OverhangList.model_validate_json(content).overhang
    except ValidationError as exc:
        primary = exc
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        log.warning("overhang %s — non-JSON response: %r",
                    accession or "?", content[:500])
        return []
    rows = None
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        if "overhang" in raw and isinstance(raw["overhang"], list):
            rows = raw["overhang"]
        else:
            list_fields = [v for v in raw.values() if isinstance(v, list)]
            if len(list_fields) == 1:
                rows = list_fields[0]
    if not rows:
        log.warning("overhang %s — schema validation failed: %s",
                    accession or "?", primary)
        return []
    salvaged: list[OverhangRow] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            salvaged.append(OverhangRow.model_validate(r))
        except ValidationError:
            continue
    if len(salvaged) < len(rows):
        log.warning("overhang %s — salvaged %d/%d rows",
                    accession or "?", len(salvaged), len(rows))
    return salvaged


def _num(x):
    if x is None or x == "":
        return None
    try:
        return float(str(x).replace(",", "").replace("$", ""))
    except ValueError:
        return None


def _clean_row(row: OverhangRow, ads_ratio: float | None = None) -> dict | None:
    """Normalize fields + apply ADS-unit fix-up + canonicalize
    common_shares_issuable from algebraic identities. Returns None for
    rows whose category isn't in the controlled vocab."""
    if row.category not in OVERHANG_CATEGORIES:
        return None
    pp = row.price_protection
    if pp not in (None, *_PRICE_PROTECTION):
        pp = None
    out = {
        "category": row.category,
        "instrument_name": (row.instrument_name or "").strip() or None,
        "outstanding_count": _num(row.outstanding_count),
        "common_shares_issuable": _num(row.common_shares_issuable),
        "strike_or_conversion_price": _num(row.strike_or_conversion_price),
        "principal_amount": _num(row.principal_amount),
        "maturity_or_expiry": (row.maturity_or_expiry or "").strip() or None,
        "issue_date": (row.issue_date or "").strip() or None,
        "price_protection": pp,
        "pp_clause_text": (row.pp_clause_text or "").strip() or None,
        "notes": (row.notes or "").strip() or None,
    }

    # ADS unit normalization. When the LLM mixed units (count in ADS
    # but csi in ordinary shares), restore consistency.
    if (ads_ratio and ads_ratio >= 2
            and out["outstanding_count"] and out["common_shares_issuable"]):
        cnt = out["outstanding_count"]
        csi = out["common_shares_issuable"]
        if cnt > 0 and abs((csi / cnt) - ads_ratio) / ads_ratio < 0.05:
            out["common_shares_issuable"] = csi / ads_ratio
            note = (f"normalized csi {csi:,.0f} → ADS {csi/ads_ratio:,.0f}")
            out["notes"] = (
                (out["notes"] + " | " + note) if out["notes"] else note
            )

    # Derive common_shares_issuable when the filing didn't state it.
    iss = out["common_shares_issuable"]
    cp = out["strike_or_conversion_price"]
    pa = out["principal_amount"]
    cnt = out["outstanding_count"]
    if iss is None:
        if (out["category"] in ("convertible", "preferred")
                and pa and cp and cp > 0):
            out["common_shares_issuable"] = pa / cp
        elif (out["category"] in ("warrant", "preferred", "option_pool",
                                   "rsu_psu_unvested") and cnt):
            out["common_shares_issuable"] = cnt
    return out


# ─── Public entry point ─────────────────────────────────────────────
async def extract_overhang_rows(
    *, accession: str, form: str, filing_date: str,
    report_date: str | None, cik: int, client,
    unit_ctx: dict | None = None,
) -> list[dict]:
    """Run the overhang prompt over one periodic filing's primary doc.

    Returns a list of cleaned dicts. Empty list if the filing has no
    body cached, the LLM returns nothing, or schema validation fails."""
    text = _load_filing_text(accession)
    if not text:
        return []
    if len(text) > MAX_INPUT_CHARS:
        log.warning("overhang %s — truncated %d→%d chars",
                    accession, len(text), MAX_INPUT_CHARS)
        text = text[:MAX_INPUT_CHARS]

    from ._llm_utils import unit_preamble
    prompt = unit_preamble(unit_ctx) + "\n" + PROMPT.format(
        form=form, cik=cik,
        as_of_date=report_date or filing_date,
        categories=list(OVERHANG_CATEGORIES),
        text=text,
    )

    chat = make_chat(client, response_format=OverhangList,
                     max_tokens=384_000)
    chat.append(system(
        "You extract outstanding dilution overhang (warrants, "
        "convertibles, preferred, option pool, unvested equity) from "
        "SEC 10-K/10-Q filings. Output strictly conforms to the "
        "OverhangList schema."
    ))
    chat.append(user(prompt))
    response = await asample_and_check(
        chat, accession=accession, handler="ledger-overhang"
    )

    rows = _parse_overhang_list(response.content, accession=accession)
    if not rows:
        return []
    ads_ratio = (unit_ctx or {}).get("ads_ratio")
    cleaned: list[dict] = []
    for r in rows:
        c = _clean_row(r, ads_ratio=ads_ratio)
        if c:
            cleaned.append(c)
    return cleaned


__all__ = [
    "HANDLER_VERSION",
    "MAX_INPUT_CHARS",
    "OVERHANG_CATEGORIES",
    "OverhangList",
    "OverhangRow",
    "extract_overhang_rows",
]
