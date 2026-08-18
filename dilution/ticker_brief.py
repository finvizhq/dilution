"""Per-ticker AI dilution brief — the current situation in one sentence.

Answers "what's the dilution situation on this ticker RIGHT NOW" in a
shape a user can scan in five seconds. Thin-prompt/thick-core: every
number is computed deterministically by the existing projections
(cards, badges, cash) and handed to the LLM as a facts block — the
model only writes prose around them and is instructed to use no
outside numbers.

Caching: one row per cik in `dilution_ticker_brief`, kept fresh by
`ensure_brief` — the pipeline entry point the payload assembly
(dilution/finviz_payload.py §8) calls with the objects it has already
fetched, so building a payload IS the brief pipeline; there is no
separate batch job. A cached brief is reused until a ledger mutation
is applied after its generation (NOT the stored facts_hash — that
embeds live-price-derived numbers and flips intraday; and NOT bare
filing dates — most filings mutate nothing); only then does one LLM
call regenerate it. This module owns the working copy of the DDL
(schema.py carries a copy for reset_db completeness — keep in sync)
and self-bootstraps on first use.
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

from .ledger.store import ensure_mutation_log_conn
from .observability import pipeline_session
from .openai_client import complete, make_sync_client, output_text, system, user

log = logging.getLogger(__name__)

# Reasoning tokens bill from the same budget as content, so a tight cap
# truncates the JSON mid-emission (the overhang-specialist lesson).
# Measured draw at effort="low" is ~200 tokens against a brief that
# serializes to well under 1K, so this keeps ample headroom.
MAX_OUTPUT_TOKENS = 8_192

_DDL = """
CREATE TABLE IF NOT EXISTS dilution_ticker_brief (
    cik INTEGER PRIMARY KEY,
    facts_hash TEXT NOT NULL,
    summary TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    model TEXT
);
"""


class TickerBrief(BaseModel):
    summary: str


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
  "summary": "<ONE flowing sentence, ≤320 chars — the whole dilution
              story, readable at a glance. Lead with the single most
              important fact, weave in the 2-4 facts that matter most
              (cash vs. quarterly burn / runway, live raise capacity,
              share overhang, risk-driver levels; a dated trigger like
              a maturity or expiry only if it is imminent and material),
              then close with an em-dash takeaway in plain words, e.g.
              '— the company is nearly out of cash with every mechanism
              in place to dilute heavily, almost immediately'>"
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
    cols = {r[1] for r in conn.execute(
        "PRAGMA table_info(dilution_ticker_brief)")}
    if "summary" not in cols or "headline" in cols or "watch_json" in cols:
        # Pre-2026-08 headline+bullets+watch shape. Pure regenerable
        # cache — drop it, the next payload builds refill, no migration.
        conn.execute("DROP TABLE dilution_ticker_brief")
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
        "summary": row["summary"],
        "facts_hash": row["facts_hash"],
        "generated_at": row["generated_at"],
        "model": row["model"],
    }


def is_fresh(cik: int) -> bool:
    """True when no ledger mutation has been applied since the cached
    brief was generated. Both timestamps come from db.now_iso(), so
    plain string comparison is exact.

    Render-time card changes that mint no mutation (a note aging past
    maturity, the s1 540-day reaper) can lag in the prose until the
    ticker's next real mutation — accepted: they are calendar effects
    whose dates the prose already carries."""
    cached = get_cached(cik)
    if not cached:
        return False
    with get_conn() as conn:
        ensure_mutation_log_conn(conn)
        row = conn.execute(
            "SELECT 1 FROM dilution_mutations "
            "WHERE cik = ? AND applied_at > ? LIMIT 1",
            (int(cik), cached["generated_at"]),
        ).fetchone()
    return row is None


def ensure_brief(cik: int, ticker: str, *, name: str, fund: dict | None,
                 latest_os: float | None, cards: dict, cash,
                 raised: float | None, badges) -> dict | None:
    """Pipeline entry point: return this ticker's brief, fresh.

    Reuses the cache when no ledger mutation postdates it; otherwise
    builds the facts block from the objects the caller has already
    fetched (the payload assembly fetches them all anyway — nothing is
    fetched twice) and regenerates with one LLM call. Any regeneration
    failure falls back to whatever is cached — dated prose beats a
    blank §8 — so only a ticker that has never been briefed returns
    None on failure.
    """
    if is_fresh(cik):
        return get_cached(cik)
    try:
        facts = build_facts(ticker=ticker, name=name, fund=fund,
                            latest_os=latest_os, cards=cards, cash=cash,
                            raised=raised, badges=badges)
        with pipeline_session(ticker, name="dilution-brief",
                              metadata={"cik": cik,
                                        "llm_model": config.LLM_MODEL}):
            return generate(cik, ticker, facts)
    except Exception:
        log.exception("brief regeneration failed for %s — serving cached",
                      ticker)
        return get_cached(cik)


def generate(cik: int, ticker: str, facts: dict) -> dict:
    """One sync LLM call → validated brief → upsert cache → return it.

    Raises RuntimeError on a non-conforming response (no partial rows
    are written) — the caller surfaces the message.
    """
    client = make_sync_client()
    resp = complete(
        client,
        name="ticker-brief",
        messages=[system(SYSTEM_PROMPT), user(PROMPT.format(
            ticker=ticker,
            facts=json.dumps(facts, indent=1, ensure_ascii=False,
                             default=str),
        ))],
        response_format=TickerBrief,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        cache_key="ticker-brief",
    )
    try:
        brief = TickerBrief.model_validate_json(output_text(resp) or "")
    except Exception as e:
        raise RuntimeError(
            f"brief generation returned non-conforming JSON: {e}"
        ) from e
    # Record what actually answered, not what config nominates — the
    # aliases float, so a stored "gpt-5.6-luna" tells you less than the
    # resolved model the API echoes back.
    model_used = getattr(resp, "model", None) or config.LLM_MODEL
    fhash = facts_hash(facts)
    with get_conn() as conn:
        _ensure_table(conn)
        conn.execute(
            """INSERT INTO dilution_ticker_brief
                 (cik, facts_hash, summary, generated_at, model)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(cik) DO UPDATE SET
                 facts_hash=excluded.facts_hash,
                 summary=excluded.summary,
                 generated_at=excluded.generated_at,
                 model=excluded.model""",
            (int(cik), fhash, brief.summary, now_iso(), model_used),
        )
    log.info("ticker brief generated for cik=%s (%s)", cik, ticker)
    return {
        "summary": brief.summary,
        "facts_hash": fhash,
        "generated_at": now_iso(),
        "model": model_used,
    }
