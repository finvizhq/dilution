"""Per-ticker AI dilution brief — headline + scannable bullets + watch items.

Answers "what's the dilution situation on this ticker RIGHT NOW" in a
shape a user can scan in five seconds. Thin-prompt/thick-core: every
number is computed deterministically by the existing projections
(cards, badges, cash) and handed to the LLM as a facts block — the
model only writes prose around them and is instructed to use no
outside numbers.

Caching: one row per cik in `dilution_ticker_brief`. The payload is
display-only — it shows the cached brief and marks it stale when a
filing arrived after generation (NOT via the stored facts_hash — that
embeds live-price-derived numbers and flips intraday). The cache is
populated by scripts/run_brief_all.py (skip-if-fresh batch over all
tracked tickers) or scripts/ticker_brief_test.py (one ticker). This
module owns the working copy of the DDL (schema.py carries a copy for
reset_db completeness — keep in sync) and self-bootstraps on first use.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

import config
from db import get_conn, now_iso

from .ledger._llm_utils import make_chat
from .llm_provider import make_sync_client, system, user

log = logging.getLogger(__name__)

# Thinking models (gemini-3.5-flash) spend reasoning tokens from the
# same budget as content — a tight cap truncates the JSON mid-emission
# (the overhang-specialist REASON_MAX_LEN lesson). Keep headroom.
MAX_OUTPUT_TOKENS = 8_192

_DDL = """
CREATE TABLE IF NOT EXISTS dilution_ticker_brief (
    cik INTEGER PRIMARY KEY,
    facts_hash TEXT NOT NULL,
    headline TEXT NOT NULL,
    bullets_json TEXT NOT NULL,
    watch_json TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    model TEXT
);
"""


class TickerBrief(BaseModel):
    headline: str
    bullets: list[str]
    watch: list[str]


SYSTEM_PROMPT = (
    "You write compact dilution briefs for a stock-research site aimed at "
    "small-cap traders. You are given a FACTS block computed from SEC "
    "filings. Use ONLY those facts — never invent or recall numbers from "
    "memory. Output strict JSON only."
)

PROMPT = """\
Write a dilution brief for {ticker} from the FACTS below.

Return strict JSON:
{{
  "headline": "<≤90 chars — the dilution story in one line, lead with the
               single most important fact>",
  "bullets": ["<4-7 bullets, ≤110 chars each, ordered most→least important.
               Each bullet = one concrete fact with its number(s).
               Cover: share overhang, live raise capacity (ATM/shelf/ELOC),
               recent dilution events, cash runway — when facts exist>"],
  "watch": ["<0-3 forward-looking items with dates/triggers from the facts,
             e.g. lock-up expiries, exchangeability dates, pending votes.
             ONLY items with a concrete date or trigger in the FACTS —
             if none exist, return []. Never pad with generic risks>"]
}}

Style: plain trader language, no hedging, no filler. Abbreviate dollars
and share counts (e.g. $7.4M, 15.3M shs). Every number must appear in
the FACTS block. Never join two facts into a composition claim
("including", "of which", "driven by") unless the FACTS state that
relationship explicitly — listing two numbers side by side is fine.

FACTS:
{facts}
"""


def _f(v):  # compact floats for the facts block
    if isinstance(v, float) and v == int(v):
        return int(v)
    return v


def _compact(d: dict, keys: tuple[str, ...]) -> dict:
    return {k: _f(d[k]) for k in keys if d.get(k) not in (None, "", [], 0)}


_CARD_KEYS = {
    "warrant": ("title", "remaining_outstanding", "exercise_price",
                "expiration_date", "known_owners"),
    "convertible": ("title", "principal_remaining", "conversion_price",
                    "remaining_shares_issuable", "maturity_date",
                    "known_owners"),
    "convertible_preferred": ("title", "principal_remaining",
                              "conversion_price", "remaining_shares_issuable",
                              "known_owners"),
    "atm": ("title", "total_capacity", "raisable_capped",
            "limited_by_baby_shelf", "sales_total_usd", "placement_agent"),
    "equity_line": ("title", "total_capacity", "remaining_capacity",
                    "counterparty"),
    "shelf": ("title", "total_shelf_capacity", "current_raisable_amount",
              "unlimited", "total_amount_raised", "expiration_date"),
    "s1_offering": ("title", "status", "anticipated_deal_size",
                    "final_deal_size", "final_pricing", "underwriter"),
}


def build_facts(*, ticker: str, name: str, fund: dict | None,
                latest_os: float | None, cards: dict, cash,
                raised: float | None, badges,
                include_recent: bool = True) -> dict:
    """Assemble the deterministic facts block from objects the ticker
    route has already fetched (same pattern as compute_badges — nothing
    is fetched twice).

    `include_recent` mixes in per-filing narrative headlines from
    walker_dumps/narrative_<ticker>.jsonl when the prototype script has
    produced them — optional texture, the brief works without it.
    """
    facts: dict = {
        "company": {
            "name": name,
            "price": (fund or {}).get("price"),
            "market_cap": _f((fund or {}).get("market_cap")),
            "shares_outstanding": _f(latest_os),
            "float": _f((fund or {}).get("float_shares")),
        },
        "instruments": {
            t: [_compact(c, _CARD_KEYS[t]) for c in lst]
            for t, lst in cards.items() if lst and t in _CARD_KEYS
        },
    }
    if badges is not None:
        facts["dilution_risk"] = {
            "overall": f"{badges.overall_label} ({badges.overall_score})",
            "drivers": {
                b.label: {"band": b.band_text, "score": b.score,
                          "facts": list(b.detail)}
                for b in badges.drivers
            },
        }
    if cash is not None:
        facts["cash"] = {
            "current_cash_est_usd": _f(cash.current_cash_est_usd),
            "quarterly_op_cf_usd": _f(cash.op_cf_quarterly_usd),
            "months_of_cash": cash.months_of_cash,
            "as_of_report": cash.latest_period_end,
            "raised_since_report_usd": _f(raised),
        }

    if include_recent:
        npath = (Path(config.BASE_DIR) / "walker_dumps"
                 / f"narrative_{ticker.lower()}.jsonl")
        if npath.exists():
            recent = []
            for line in npath.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (r.get("ok") and (r.get("dilution_impact") or 0) >= 20
                        and "recap" not in (r.get("tags") or [])):
                    recent.append({
                        "date": r["filing_date"], "form": r["form"],
                        "impact": r["dilution_impact"],
                        "what": r["headline"],
                    })
            if recent:
                facts["recent_filings"] = recent[:12]
    return facts


def facts_hash(facts: dict) -> str:
    blob = json.dumps(facts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _ensure_table(conn) -> None:
    conn.executescript(_DDL)


def get_cached(cik: int) -> dict | None:
    with get_conn() as conn:
        _ensure_table(conn)
        row = conn.execute(
            "SELECT * FROM dilution_ticker_brief WHERE cik = ?",
            (int(cik),),
        ).fetchone()
    if not row:
        return None
    return {
        "headline": row["headline"],
        "bullets": json.loads(row["bullets_json"]),
        "watch": json.loads(row["watch_json"]),
        "facts_hash": row["facts_hash"],
        "generated_at": row["generated_at"],
        "model": row["model"],
    }


def generate(cik: int, ticker: str, facts: dict) -> dict:
    """One sync LLM call → validated brief → upsert cache → return it.

    Raises RuntimeError on a non-conforming response (no partial rows
    are written) — the caller surfaces the message.
    """
    client = make_sync_client()
    chat = make_chat(client, response_format=TickerBrief,
                     max_tokens=MAX_OUTPUT_TOKENS)
    chat.append(system(SYSTEM_PROMPT))
    chat.append(user(PROMPT.format(
        ticker=ticker,
        facts=json.dumps(facts, indent=1, ensure_ascii=False, default=str),
    )))
    resp = chat.sample()
    try:
        brief = TickerBrief.model_validate_json(resp.content or "")
    except Exception as e:
        raise RuntimeError(
            f"brief generation returned non-conforming JSON: {e}"
        ) from e
    fhash = facts_hash(facts)
    with get_conn() as conn:
        _ensure_table(conn)
        conn.execute(
            """INSERT INTO dilution_ticker_brief
                 (cik, facts_hash, headline, bullets_json, watch_json,
                  generated_at, model)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(cik) DO UPDATE SET
                 facts_hash=excluded.facts_hash,
                 headline=excluded.headline,
                 bullets_json=excluded.bullets_json,
                 watch_json=excluded.watch_json,
                 generated_at=excluded.generated_at,
                 model=excluded.model""",
            (int(cik), fhash, brief.headline,
             json.dumps(brief.bullets, ensure_ascii=False),
             json.dumps(brief.watch, ensure_ascii=False),
             now_iso(), config.LLM_MODEL),
        )
    log.info("ticker brief generated for cik=%s (%s)", cik, ticker)
    return {
        "headline": brief.headline,
        "bullets": brief.bullets,
        "watch": brief.watch,
        "facts_hash": fhash,
        "generated_at": now_iso(),
        "model": config.LLM_MODEL,
    }
