"""Dilution sqlite schema. Cards-only build — ledger architecture.

Tables:
  dilution_company       — ticker / CIK / name
  dilution_filings       — filing index (form, date, accession)
  dilution_raw           — raw markdown of narrative filings (walker input)
  dilution_ledger        — capitalization table: one row per instrument tranche
  dilution_walk_state    — per-ticker id-sequence allocator + last-walk marker
  dilution_walked        — per-accession walked set (resume / back-fill safe)
  dilution_anchor_diffs  — periodic-filing reconciliation discrepancies
  dilution_walk_errors   — dropped-mutation audit trail

init_dilution_db() is only called from scripts/reset_db.py (after the DB
file has been deleted). The pipeline workers assume the schema is
already in place — running them against a missing/old DB is a user
error, not something to silently migrate around.
"""

from db import get_conn

SCHEMA = """
CREATE TABLE IF NOT EXISTS dilution_company (
    cik INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    name TEXT NOT NULL,
    added_at TEXT NOT NULL,
    -- Foreign Private Issuer flag (filed 20-F/40-F). Drives ADS-vs-
    -- ordinary unit handling end-to-end: extractors are told to report
    -- in the listed instrument (ADS for FPI), and the card layer
    -- displays in that same unit.
    is_fpi INTEGER NOT NULL DEFAULT 0,
    -- Ordinary shares per 1 ADS, e.g. 100 for XTLB pre 2026-03-25 or
    -- 400 after. NULL for non-FPI issuers.
    ads_ratio REAL,
    unit_detected_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_dilution_company_ticker ON dilution_company(ticker);

CREATE TABLE IF NOT EXISTS dilution_filings (
    accession_number TEXT PRIMARY KEY,
    cik INTEGER NOT NULL,
    form TEXT NOT NULL,
    filing_date TEXT NOT NULL,
    report_date TEXT,
    primary_doc TEXT,
    primary_doc_url TEXT,
    homepage_url TEXT,
    items TEXT,
    -- SEC Securities Act file number (e.g. "333-256827") OR Exchange
    -- Act file number ("001-36404"). The 333- prefix is the canonical
    -- linkage between a registration statement (S-1, S-3) and all its
    -- children (424B prospectus take-downs, S-3/A amendments, POS AM,
    -- RW withdrawals, EFFECT notices). Lets us deterministically group
    -- shelf families, distinguish primary vs resale 424Bs, and detect
    -- withdrawn / expired registrations without LLM prose-parsing.
    -- Exchange Act forms (10-K/10-Q/8-K) carry the 001- listing number
    -- which is per-ticker, not per-registration — useful but separate.
    file_number TEXT,
    fetched_at TEXT,
    -- Set when the two-stage extractor has run on this accession.
    -- Distinguishes "filing has 0 events because none were found"
    -- from "filing has not been processed yet" — the previous
    -- pipeline conflated these and re-extracted empty filings every
    -- run. Format: "<llm_model>/stage1-vN" so handler-version drift
    -- triggers re-extraction.
    extracted_at TEXT,
    extracted_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_dilution_filings_cik ON dilution_filings(cik);
CREATE INDEX IF NOT EXISTS idx_dilution_filings_form ON dilution_filings(form);
CREATE INDEX IF NOT EXISTS idx_dilution_filings_date ON dilution_filings(filing_date);
CREATE INDEX IF NOT EXISTS idx_dilution_filings_file_number
    ON dilution_filings(cik, file_number);

CREATE TABLE IF NOT EXISTS dilution_raw (
    accession_number TEXT NOT NULL REFERENCES dilution_filings(accession_number),
    doc_name TEXT NOT NULL,
    doc_type TEXT,
    content_md TEXT NOT NULL,
    downloaded_at TEXT NOT NULL,
    PRIMARY KEY (accession_number, doc_name)
);

-- ─── Ledger architecture ─────────────────────────────────────────────
-- One row per instrument tranche the issuer has outstanding (or had,
-- in the case of closed instruments kept for history). Replaces the
-- per-filing event log + post-hoc clustering with a stateful cap table
-- mutated chronologically as filings arrive.
CREATE TABLE IF NOT EXISTS dilution_ledger (
    instrument_id TEXT PRIMARY KEY,             -- "W-001" / "C-001" / "ATM-001" / "P-001" / "EL-001" / "SH-001" / "S1-001"
    ticker TEXT NOT NULL,
    cik INTEGER NOT NULL,
    type TEXT NOT NULL,                         -- warrant|convertible|preferred|atm|equity_line|shelf|s1_offering|equity
    created_at TEXT NOT NULL,                   -- filing_date the instrument was first disclosed
    created_accession TEXT NOT NULL,            -- IMMUTABLE provenance: the filing that birthed this instrument. Never repointed.
    -- Mutable registration pointer. NULL until the instrument is re-
    -- registered onto a successor shelf via the redisclosure shelf-
    -- rollover path (store._append_redisclosure). When set, it holds the
    -- accession of the shelf-host filing (S-3/F-3) the instrument now
    -- lives under, and the file_number-walk rollups (_parent_shelf,
    -- _shelf_family_drawn, _last_banker_for_shelf) join on
    -- COALESCE(registration_accession, created_accession) so takedowns
    -- credit the LIVE shelf instead of the original (often expired) one.
    -- See the CGEN SVB ATM: created under the 2020 F-3 (333-240183),
    -- re-registered under the 2023 F-3 (333-270985).
    registration_accession TEXT,
    counterparty_canonical TEXT,                -- the INVESTOR/BUYER/LENDER putting capital into the issuer (e.g. "Streeterville", "Hudson Bay")
    counterparty_status TEXT,                   -- named|generic|absent — distinguishes "filing named an entity" (named, canonical populated) from "filing used a generic descriptor like 'institutional investor' so NULLING RULE applied" (generic, canonical null) from "filing didn't mention a counterparty at all" (absent, canonical null). Lets the view render "n/a" vs "—" instead of conflating both into a single em-dash.
    placement_agent_canonical TEXT,             -- the BANK running the offering (e.g. "Maxim", "ThinkEquity"). Distinct from counterparty.
    label TEXT,                                 -- clean human-readable instrument label, e.g. "Series 9 Preferred", "Inducement Warrants", "December 2022 Streeterville Note"
    terms_json TEXT NOT NULL,                   -- type-specific fields (strike, conv_price, principal, capacity, maturity, …)
    outstanding_json TEXT NOT NULL,             -- count, principal_remaining, sold_to_date, drawn_usd, etc.
    status TEXT NOT NULL,                       -- active|exercised|converted|redeemed|expired|terminated|superseded:<id>
    status_at TEXT,                             -- date the current status was set
    history_json TEXT NOT NULL,                 -- append-only array of {date, accession, form, action, fields_changed, snippet}
    last_seen_accession TEXT,
    last_seen_date TEXT,
    -- Anchor-reconciliation consecutive-miss counter. Incremented each
    -- time a periodic-filing overhang pass does NOT match this row;
    -- reset to 0 when the row IS matched. After 2 consecutive misses,
    -- warrants/convertibles/preferreds without an objective dead-signal
    -- (Tier 1) are auto-closed by anchor.py — this is "Tier 2" closure
    -- for warrants whose stated expiration is far enough out that the
    -- date-based Tier 1 check would never fire even though the issuer
    -- has clearly stopped disclosing the row. Shelves / ATMs / equity
    -- lines are excluded from Tier 2 — they have legal duration and
    -- close on date-based Tier 1 (3y agreement / Rule 415 3y) only.
    anchor_miss_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_dilution_ledger_cik
    ON dilution_ledger(cik);
CREATE INDEX IF NOT EXISTS idx_dilution_ledger_cik_type
    ON dilution_ledger(cik, type);
CREATE INDEX IF NOT EXISTS idx_dilution_ledger_cik_status
    ON dilution_ledger(cik, status);

-- Per-issuer walker progress + id sequence allocator. One row per CIK.
-- next_id_seq_json holds the high-water mark for each instrument-type
-- prefix so id allocation is monotonic across walks.
CREATE TABLE IF NOT EXISTS dilution_walk_state (
    cik INTEGER PRIMARY KEY,
    last_processed_accession TEXT,
    last_processed_filing_date TEXT,
    next_id_seq_json TEXT NOT NULL DEFAULT '{}',
    pipeline_version TEXT,
    walked_at TEXT
);

-- Per-accession walker progress. One row per (cik, accession) the
-- walker has processed. Replaces the single positional
-- last_processed_accession marker in dilution_walk_state for resume
-- decisions: that marker skipped any filing sorting BEFORE the resume
-- point on a later run, so a back-filled filing whose raw text arrived
-- after the first walk (dilution_raw is INNER-JOINed in _list_filings)
-- never got picked up without a --force re-walk. With this set the
-- walker processes any in-scope filing NOT already recorded here,
-- regardless of sort position. (next_id_seq_json stays in
-- dilution_walk_state — id sequencing is independent of resume.)
CREATE TABLE IF NOT EXISTS dilution_walked (
    cik INTEGER NOT NULL,
    accession_number TEXT NOT NULL,
    filing_date TEXT,
    pipeline_version TEXT,
    walked_at TEXT NOT NULL,
    PRIMARY KEY (cik, accession_number)
);

CREATE INDEX IF NOT EXISTS idx_dilution_walked_cik
    ON dilution_walked(cik);

-- Stock splits sourced from market-data vendors (Finviz, yfinance).
-- Walked-in to the ledger as synthetic `apply_split` mutations BEFORE
-- the chronologically corresponding filings, so any pre-existing
-- instrument is scaled by the time the filing is processed.
--   pre / post: as in the ApplySplit Pydantic model — 1-for-100 is
--               post=1, pre=100; 4-for-1 is post=4, pre=1.
--   units:      "common" for US issuers, "ads" for FPIs (the store's
--               _apply_split filters by this so an ADS-ratio change
--               doesn't rescale underlying-common warrants).
--   source:     "finviz", "yfinance", or "finviz+yfinance" when both
--               vendors agreed; lets us audit conflict-resolution.
CREATE TABLE IF NOT EXISTS dilution_splits (
    cik INTEGER NOT NULL,
    effective_date TEXT NOT NULL,
    pre INTEGER NOT NULL,
    post INTEGER NOT NULL,
    direction TEXT NOT NULL,
    units TEXT NOT NULL,
    source TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (cik, effective_date)
);

CREATE INDEX IF NOT EXISTS idx_dilution_splits_cik
    ON dilution_splits(cik);

-- Periodic-filing anchor reconciliation log. After processing each
-- 10-K/10-Q/20-F/40-F, the walker diffs the ledger against the
-- issuer's own outstanding-instruments table; mismatches land here.
-- v1 always overwrites the ledger to match the filing — these rows
-- are the audit trail of where the walker drifted.
CREATE TABLE IF NOT EXISTS dilution_anchor_diffs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cik INTEGER NOT NULL,
    accession_number TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    diff_kind TEXT NOT NULL,                    -- missing_in_ledger|extra_in_ledger|field_mismatch|count_mismatch
    instrument_id TEXT,
    category TEXT,                              -- warrant|convertible|preferred|...
    ledger_value_json TEXT,
    filing_value_json TEXT,
    resolution TEXT,                            -- overwrite|kept_ledger|noop  (v1: always overwrite)
    detected_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dilution_anchor_diffs_cik_acc
    ON dilution_anchor_diffs(cik, accession_number);

-- Mutations the walker rejected at apply time (missing id, illegal
-- transition, capacity overflow, etc.). Drops are localized to the
-- bad mutation; the rest of the filing's mutation list still applies.
-- Surfaces the worst-case failures of the LLM walker for inspection.
CREATE TABLE IF NOT EXISTS dilution_walk_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cik INTEGER NOT NULL,
    accession_number TEXT NOT NULL,
    error_kind TEXT NOT NULL,                   -- missing_id|illegal_transition|capacity_overflow|type_mismatch|...
    message TEXT,
    mutation_json TEXT,
    detected_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dilution_walk_errors_cik_acc
    ON dilution_walk_errors(cik, accession_number);

-- Auxiliary index of drawdowns against shelf / ATM / equity-line
-- instruments. Populated synchronously by apply_mutations every time a
-- record_event(drawdown) lands on an eligible ledger row. Powers IB6
-- "raised in last 12 months" without having to JSON-walk history_json.
CREATE TABLE IF NOT EXISTS dilution_ledger_drawdowns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cik INTEGER NOT NULL,
    instrument_id TEXT NOT NULL REFERENCES dilution_ledger(instrument_id),
    accession_number TEXT NOT NULL,
    event_date TEXT NOT NULL,
    amount_usd REAL,
    shares REAL,
    price REAL,
    -- Canonical name of the party that sold the shares on this
    -- takedown. For ATM / shelf / s1_offering this is the placement
    -- agent / underwriter (Jefferies, B. Riley); for equity_line it's
    -- the named investor buying direct (Yorkville, M2B). The role
    -- column distinguishes the two so the "Last Banker" shelf card
    -- query can filter to 'bank' without conflating direct-investor
    -- takedowns into the banker slot.
    drawdown_party_canonical TEXT,
    drawdown_party_role TEXT,  -- 'bank' | 'investor' | NULL
    detected_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dilution_ledger_drawdowns_cik_date
    ON dilution_ledger_drawdowns(cik, event_date);
CREATE INDEX IF NOT EXISTS idx_dilution_ledger_drawdowns_instrument
    ON dilution_ledger_drawdowns(instrument_id);

-- Per-instrument narrative cache for the dashboard. Generated lazily
-- by the project stage; key is a hash of the ledger row's terms +
-- status so the LLM only re-runs when the underlying state changes.
CREATE TABLE IF NOT EXISTS dilution_ledger_narrative (
    instrument_id TEXT PRIMARY KEY REFERENCES dilution_ledger(instrument_id),
    terms_hash TEXT NOT NULL,
    headline TEXT,
    counterparty_role TEXT,                     -- bank|investor|strategic|undisclosed
    terms_summary TEXT,
    generated_at TEXT NOT NULL,
    model TEXT
);

-- Per-ticker AI dilution brief cache (headline + bullets + watch
-- items) for the dashboard. dilution/ticker_brief.py owns the working
-- copy of this DDL (keep in sync) and self-bootstraps on first use;
-- the copy here only exists so reset_db.py produces a complete
-- schema. Keyed by a hash of the deterministic facts block so the
-- dashboard can flag a cached brief as stale without an LLM call.
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


# Controlled vocabulary — keep in sync with extractor prompts.
EVENT_TYPES = [
    "shelf_registration",
    "atm_program_established",
    "atm_sale",
    "equity_line_established",
    "equity_line_sale",
    "registered_direct_offering",
    "underwritten_offering",
    "private_placement",
    "convertible_note_issuance",
    "convertible_note_conversion",
    "warrant_issuance",
    "warrant_exercise",
    "preferred_issuance",
    "preferred_conversion",
    "reverse_split",
    "forward_split",
    "authorized_share_increase",
    "equity_plan_increase",
    "share_issuance_other",
    "offering_effective",
    "shelf_withdrawn",
    # Stock-paid / scrip / PIK dividends and UK "bonus issues" — the
    # company issues new shares to existing holders pro-rata. Dilutive
    # in absolute share count even though pro-rata.
    "stock_dividend",
    # US "rights offering" / UK "rights issue" / "open offer" — pro-rata
    # subscription right to buy new shares at a fixed price. Distinct
    # from underwritten / registered_direct since holders are
    # explicitly identified by record date.
    "rights_offering",
    # Anti-dilutive: company buys back shares (open-market or tender).
    # Tracked so the cards can net repurchases against issuances.
    "share_repurchase",
    "other",
]


# Ledger instrument-type vocabulary. Keep in sync with mutations.py
# CreateInstrument.type Literal and the per-type card projectors.
INSTRUMENT_TYPES = (
    "warrant",
    "convertible",
    "preferred",
    "atm",
    "equity_line",
    "shelf",
    "s1_offering",
    "equity",
)

# Lifecycle states an instrument can be in. `superseded` carries a
# reference to the replacing instrument id encoded as "superseded:<id>".
INSTRUMENT_STATUSES = (
    "active",
    "exercised",
    "converted",
    "redeemed",
    "expired",
    "terminated",
    "superseded",
)


def init_dilution_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
