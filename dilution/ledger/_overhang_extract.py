"""Periodic-filing overhang extraction for the seed and anchor passes.

Originally extracted from the legacy `dilution/overhang.py`. v2 split
the single mixed-category prompt into category-specialist prompts
(warrant / convertible / preferred, later + shelf / atm / equity_line)
dispatched in parallel — six calls, each re-sending the full filing
text.

v5 MERGES those six specialists back into ONE call. The six parallel
calls each re-sent the entire periodic body (median ~106K chars, up to
~940K) — so a 10-K was shipped to the model six times. The merge sends
it once and asks for all six instrument types in a single structured
response (~83% fewer input tokens on the periodic path).

The per-type guidance that made the specialists accurate is preserved
verbatim: `extract_overhang_rows` composes the merged prompt from the
SAME six `_*_PROMPT` templates (each contributing its section anchors,
field semantics, and type-exclusion rules), with the filing text and
shared global rules factored out to appear exactly once. The
type-exclusion language ("X is extracted separately") becomes the
list-routing rule ("X belongs in its own list"), so combo transactions
(preferred + attached warrant) still emit one row per list.

Fault isolation is preserved via a fallback (v6): if the merged call's
JSON fails to parse — large/complex filings occasionally produce
malformed combined output (FCEL 10-Ks) — `extract_overhang_rows` falls
back to the six independent specialist calls for that one filing. So the
merge is the fast path for ~95% of filings (one call) and the robust
six-call path catches the rest, where smaller per-type schemas and
outputs parse far more reliably. Eval-confirmed: eval LLM calls −43%,
accuracy unchanged (anchor.reconcile_against_periodic also degrades
safely on an empty overhang — it never closes ledger rows on overhang
silence, only on independent maturity/expiry signals).

Public surface is unchanged: `extract_overhang_rows()` returns a list
of cleaned dicts keyed by `category`. The non-core categories from v1
(`option_pool`, `rsu_psu_unvested`, `other`) are dropped — anchor and
seed only consume the three core categories, so extracting the others
was dead weight that diluted prompt attention.

The walker invokes this for each 10-K / 10-Q / 20-F / 40-F, and for
6-K interim furnishings that carry financial statements:
  - seed.py runs it on the earliest periodic filing.
  - walker.py runs it after each subsequent periodic filing and
    diffs the result against the ledger via anchor.reconcile_against_periodic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError

import config
from db import get_conn
from dilution.openai_client import (
    acomplete, max_input_chars, output_text, system, truncated, user,
)

from ._llm_utils import (
    check_response, normalize_filing_text, unit_preamble,
)

log = logging.getLogger(__name__)


# Cap matches walker_llm.MAX_INPUT_CHARS — derived from the model input
# ceiling (config.OPENAI_MAX_INPUT_TOKENS = 922,000) at the 3-chars/token
# floor. Periodic filings can be large (S-1/F-1 can hit 1M+ chars) but
# rarely exceed this guard; over-cap text is truncated with a logged
# warning.
MAX_INPUT_CHARS = max_input_chars()

# Output cap for the overhang call, shared by reasoning and content
# tokens (on /v1/responses reasoning bills from this budget). A
# legitimate combined response is tiny: across 202 real eval responses
# the mean was 2.7 instrument rows, and a maxed-out 17-row worst case
# (the densest ever observed) serializes to ~5.5K chars ≈ ~1.4-1.9K
# tokens; reasoning at effort="low" measured ~200 tokens on real
# filings. 16K leaves several times that headroom while staying ~1/8 of
# the model's hard 128,000-token output limit. Keeping it low tightly
# bounds the blast radius of a degenerate numeric repetition loop: the
# model occasionally death-spirals on a numeric field (e.g. a
# preferred's common_shares_issuable) and emits digits until it hits
# this ceiling. The old 384K cap let that run ~5 min/filing (and again
# on the per-specialist fallback). Truncated output is salvaged (see
# _salvage_truncated_json), so even a rare legitimate over-cap response
# still yields its completed rows rather than nothing.
OVERHANG_MAX_TOKENS = 16_000

# Stamped into provenance / version tracking. Bump when the prompt
# or schema changes. v3 = shelf-family specialists added (shelf / atm /
# equity_line) alongside the original warrant / convertible / preferred.
# v4 = global rule barring counts/principal/capacity sourced from
# beneficial-ownership / major-shareholder tables (XTLB: a 20-F Item-7
# footnote listing one holder's 80M ordinary-share warrants got read as
# the tranche total and shrank the ledger count 1.5M→800k).
# v5 = six parallel specialists merged into ONE call (filing text sent
# once, not 6×); per-type prompts reused verbatim to build the merged
# prompt. Eval-confirmed: eval LLM calls −43%, accuracy unchanged.
# v6 = per-specialist fallback when the merged JSON fails to parse
# (large filings — FCEL 10-Ks); restores fault isolation for the ~5%
# of filings that need it.
# v7 = output cap cut 384K→8K (OVERHANG_MAX_TOKENS) + truncated-JSON
# salvage. A degenerate numeric repetition loop ran the model to the old
# 384K ceiling (~5 min/filing, twice when the fallback re-looped); the
# smaller cap bounds the loop and salvage recovers the rows emitted before
# the poisoned one, so most truncations no longer trigger the 6× fallback.
HANDLER_VERSION = "ledger-overhang-v7"


# ─── Output vocabulary (unchanged for downstream consumers) ─────────
# v3 adds shelf-family categories. anchor.py / cards.py consume them via
# the unified dict shape produced by `_clean_*_row`.
OVERHANG_CATEGORIES = (
    "warrant", "convertible", "preferred",
    "shelf", "atm", "equity_line",
)


# ─── Per-category narrow schemas ────────────────────────────────────
# Each schema declares only the fields the category actually populates.
# Narrower schemas help xAI's structured-output decoder stay focused —
# fewer fields to mis-fill, fewer null-vs-value decisions per row.
# All three convert to the unified dict shape in `_clean_row` so the
# downstream consumers (anchor.py, seed.py) don't need to change.

class _OverhangBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instrument_name: str | None = Field(None, max_length=200)
    issue_date: str | None = Field(None, max_length=30)
    notes: str | None = Field(None, max_length=800)


class WarrantOverhangRow(_OverhangBase):
    """One outstanding warrant tranche."""
    outstanding_count: float | None = None
    strike_price: float | None = None
    expiry: str | None = Field(None, max_length=30)
    # Pre-funded warrants price near zero ($0.0001 typical) and behave
    # economically like common shares — surface explicitly so the
    # walker's downstream cards layer can treat them as such.
    is_pre_funded: bool | None = None


class WarrantOverhangList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    warrants: list[WarrantOverhangRow] = Field(default_factory=list)


class ConvertibleOverhangRow(_OverhangBase):
    """One outstanding convertible note tranche."""
    principal_amount: float | None = None
    conversion_price: float | None = None
    # Scale token for principal_amount only — convertible-note balances
    # are routinely tabulated "(in thousands)". conversion_price is a
    # per-share price (always full dollars) and is NOT scaled. See
    # _scale_factor / _DOLLAR_SCALE_RULE.
    dollar_scale: str | None = Field(None, max_length=12)
    # Common-shares-issuable when the filing explicitly states the
    # aggregate. Otherwise null — code derives it as
    # principal_amount / conversion_price.
    common_shares_issuable: float | None = None
    maturity: str | None = Field(None, max_length=30)


class ConvertibleOverhangList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    convertibles: list[ConvertibleOverhangRow] = Field(default_factory=list)


class PreferredOverhangRow(_OverhangBase):
    """One outstanding preferred-stock series."""
    # Series identifier is the strongest identity key for preferreds —
    # see extract_series_letter in mutations.py. Surfacing it as a
    # structured field (vs. burying in instrument_name) tightens
    # anchor matching considerably.
    series_letter: str | None = None
    outstanding_count: float | None = None
    conversion_price: float | None = None
    # Aggregate liquidation preference / redemption value for the
    # series, NOT per-share. Walker stores this on terms.liquidation_preference.
    aggregate_liquidation_preference: float | None = None
    # Scale token for aggregate_liquidation_preference only — preferred
    # liquidation-preference figures are often tabulated "(in
    # thousands)". conversion_price (per-share) is NOT scaled. See
    # _scale_factor / _DOLLAR_SCALE_RULE.
    dollar_scale: str | None = Field(None, max_length=12)
    # Common-shares-issuable when the filing explicitly states the
    # aggregate. Otherwise null — code derives it as
    # aggregate_liquidation_preference / conversion_price.
    common_shares_issuable: float | None = None


class PreferredOverhangList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preferreds: list[PreferredOverhangRow] = Field(default_factory=list)


# Shelf-family schemas — extracted as period-end snapshots of capacity
# utilization. Identity keys differ per type:
#   shelf       — SEC file_number (333-XXXXXX), canonical and stable
#   atm         — (sales_agent, agreement_date) — two Maxim ATMs filed
#                 ~11 months apart are TWO distinct instruments, not
#                 amendments of one (see ATMTerms.agreement_date)
#   equity_line — (investor, agreement_date) — Yorkville / M2B re-up
#                 every 12-18 months
#
# All three reuse _OverhangBase for instrument_name / issue_date /
# notes.
class ShelfOverhangRow(_OverhangBase):
    """One active shelf registration disclosed in a periodic filing."""
    file_number: str | None = Field(None, max_length=20)
    form: str | None = Field(None, max_length=20)
    total_capacity_usd: float | None = None
    drawn_to_date_usd: float | None = None
    remaining_capacity_usd: float | None = None
    dollar_scale: str | None = Field(None, max_length=12)
    effect_date: str | None = Field(None, max_length=30)
    is_terminated: bool | None = None


class ShelfOverhangList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shelves: list[ShelfOverhangRow] = Field(default_factory=list)


class ATMOverhangRow(_OverhangBase):
    """One ATM (At-the-Market) sales program."""
    sales_agent: str | None = Field(None, max_length=120)
    agreement_date: str | None = Field(None, max_length=30)
    total_capacity_usd: float | None = None
    sold_to_date_usd: float | None = None
    remaining_capacity_usd: float | None = None
    dollar_scale: str | None = Field(None, max_length=12)
    is_terminated: bool | None = None


class ATMOverhangList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    atms: list[ATMOverhangRow] = Field(default_factory=list)


class EquityLineOverhangRow(_OverhangBase):
    """One equity-line / ELOC / standby purchase facility."""
    investor: str | None = Field(None, max_length=120)
    agreement_date: str | None = Field(None, max_length=30)
    total_capacity_usd: float | None = None
    drawn_to_date_usd: float | None = None
    remaining_capacity_usd: float | None = None
    dollar_scale: str | None = Field(None, max_length=12)
    is_terminated: bool | None = None


class EquityLineOverhangList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    equity_lines: list[EquityLineOverhangRow] = Field(default_factory=list)


class CombinedOverhangList(BaseModel):
    """Single-call merged schema (v5) — all six instrument types in one
    structured response. Reuses the exact per-type row models so the
    `_clean_*_row` functions and every downstream consumer are unchanged;
    only the number of LLM calls drops (6 → 1)."""
    model_config = ConfigDict(extra="forbid")
    warrants: list[WarrantOverhangRow] = Field(default_factory=list)
    convertibles: list[ConvertibleOverhangRow] = Field(default_factory=list)
    preferreds: list[PreferredOverhangRow] = Field(default_factory=list)
    shelves: list[ShelfOverhangRow] = Field(default_factory=list)
    atms: list[ATMOverhangRow] = Field(default_factory=list)
    equity_lines: list[EquityLineOverhangRow] = Field(default_factory=list)


# ─── Form-family section anchors ────────────────────────────────────
# 10-K/10-Q and 20-F/40-F are structurally different documents. A 10-K's
# notes-to-financial-statements live in the main body with sequential
# numbering (Notes 1-N). A 20-F's notes live in Part III, Item 18,
# numbered separately from the main 20-F body, and the equity-class
# disclosures often sit in Item 10 "Additional Information" rather than
# in the financial-statement notes. Empirical signal: all three FPI
# tickers in the 2026-05-15 eval (SCNI / AACG / XTLB) returned empty
# overhang on their seed 20-F under the 10-K-tuned section anchors.
#
# `_form_family()` maps the filing form to a category, and per-(family,
# category) section anchors below get injected into each specialist
# prompt via `{section_anchor}` substitution.

def _form_family(form: str) -> str:
    """us_periodic for 10-K/10-Q variants, fpi_annual for 20-F/40-F and
    6-K. A 6-K is an FPI interim furnishing whose IFRS-style
    financial-statement notes match the fpi_annual section anchors far
    better than the 10-K/10-Q-tuned us_periodic ones. Defaults to
    us_periodic for unknown forms — the older behavior and the most
    common in our corpus."""
    if not form:
        return "us_periodic"
    base = form.upper().replace(" ", "").split("/")[0]
    if base in ("20-F", "40-F", "6-K"):
        return "fpi_annual"
    return "us_periodic"


_SECTION_ANCHORS = {
    ("us_periodic", "warrant"): (
        "Locate the warrants in: notes 8/9 (or equivalent) of the "
        "10-K/10-Q, the equity / capital-stock note, and any "
        "\"Warrants\" subsection of the long-term-debt note. Cap "
        "tables and roll-forward tables (\"Beginning balance / issued "
        "/ exercised / outstanding\") are authoritative — period-end "
        "\"outstanding\" is the row to extract."
    ),
    ("fpi_annual", "warrant"): (
        "Locate the warrants in a 20-F / 40-F: (a) Item 10 "
        "\"Additional Information\" / \"Description of Securities\" "
        "for warrant-class summaries, and (b) the Notes to "
        "Financial Statements in Part III (Item 18). The "
        "financial-statement notes are numbered separately from the "
        "main 20-F body — look for notes titled \"Warrants,\" "
        "\"Equity-linked Instruments,\" \"Warrant Liabilities,\" or "
        "\"Derivative Liabilities.\" Under IFRS, warrants are often "
        "classified on the balance sheet as a liability and "
        "fair-valued each period; the relevant balance-sheet line is "
        "typically \"Warrant liabilities\" or \"Derivative warrant "
        "liabilities.\" Per-tranche detail (strike, expiration, "
        "count) lives in the warrant note's table, not in the "
        "balance-sheet line. The cover page states the ADS-to-"
        "ordinary ratio — warrant counts may be quoted in ordinary "
        "shares; the unit preamble at the top of this prompt tells "
        "you how to convert to ADS."
    ),
    ("us_periodic", "convertible"): (
        "Locate the notes in: the long-term-debt note (typically "
        "notes 6-8 of a 10-K), any \"Convertible Notes\" or \"Notes "
        "Payable\" subsection, and the liquidity / going-concern "
        "section. Tranche tables (\"Issued / discount / accrued "
        "interest / principal outstanding\") are authoritative — use "
        "the period-end \"principal outstanding\" value."
    ),
    ("fpi_annual", "convertible"): (
        "Locate the convertible notes in a 20-F / 40-F: (a) the "
        "Notes to Financial Statements in Part III (Item 18) — look "
        "for notes titled \"Convertible Loan Notes,\" \"Convertible "
        "Bonds,\" \"Convertible Loans,\" \"Loans Payable,\" or "
        "\"Debt.\" IFRS issuers commonly carve convertibles into "
        "host-debt and embedded-derivative components; the principal "
        "outstanding for our purposes is the FACE / NOMINAL amount, "
        "not the carrying value net of discount or bifurcated "
        "derivative liability. (b) Item 5 \"Operating and Financial "
        "Review — Liquidity\" sometimes summarizes convertible "
        "balances and recent activity. The financial-statement notes "
        "are numbered separately from the main 20-F body. If the "
        "issuer's reporting currency is non-USD, use the period-end "
        "USD convenience translation when stated, otherwise the "
        "USD-translated balance from the convenience-translation "
        "table on the financial-statement cover."
    ),
    ("us_periodic", "preferred"): (
        "Locate the preferred in: the balance-sheet face (line item "
        "\"Preferred Stock, par $X, N shares authorized / M shares "
        "outstanding\"), the equity / capital-stock note (one "
        "subsection per series), and any \"Series A/B/C...\" "
        "narrative. Use the period-end OUTSTANDING count, NOT shares "
        "authorized."
    ),
    ("fpi_annual", "preferred"): (
        "Locate the preferred in a 20-F / 40-F: (a) Item 10 "
        "\"Additional Information\" — \"Memorandum and Articles of "
        "Association\" or \"Description of Capital Stock\" describes "
        "the share-class structure (ordinary, deferred, preferred, "
        "founder, etc.), and (b) the share-capital / equity note in "
        "Part III financial statements. Many FPIs reserve preferred "
        "or \"deferred\" shares in their constitutional documents "
        "without ever issuing them — extract ONLY series where the "
        "filing states an OUTSTANDING count (not just AUTHORIZED). "
        "The balance sheet may have a separate \"Preference Shares\" "
        "or \"Preferred Shares\" line; use the period-end count. "
        "Series identifiers in FPI filings are sometimes by name "
        "rather than letter (e.g. \"Series Seed,\" \"Series Founders\") "
        "— extract whatever stable label the filing uses."
    ),
    ("us_periodic", "shelf"): (
        "Locate the shelf registrations in: MD&A \"Liquidity and "
        "Capital Resources\", the equity / capital-stock note, the "
        "subsequent-events note, and any cover-page reference to a "
        "Form S-3 / F-3 file number (333-XXXXXX). The filing typically "
        "states the original SHELF CAPACITY (e.g. \"$350 million "
        "shelf registration statement on Form S-3\"), the CUMULATIVE "
        "AMOUNT RAISED to date under that shelf, and the AMOUNT "
        "REMAINING AVAILABLE for future take-downs. SEC Rule 415(a)(5) "
        "caps shelves at 3 years from the effective date; mention of "
        "an expired / unavailable shelf indicates termination."
    ),
    ("fpi_annual", "shelf"): (
        "Locate the shelf registrations in a 20-F / 40-F: (a) Item 5 "
        "\"Operating and Financial Review — Liquidity and Capital "
        "Resources\", (b) Item 10 \"Additional Information\", and (c) "
        "the share-capital / equity note. FPI issuers use Form F-3 "
        "(or F-3ASR for well-known seasoned issuers); the file number "
        "still has the 333-XXXXXX format. Convenience-translation to "
        "USD applies if the issuer reports in another currency — use "
        "the USD amounts when stated."
    ),
    ("us_periodic", "atm"): (
        "Locate the At-the-Market programs in: the equity / capital-"
        "stock note (often titled \"At-the-Market Offering\" or "
        "\"Equity Sales Agreement\"), the cash-flow-statement "
        "footnotes (look for \"proceeds from sales under ATM facility\" "
        "or \"sales agent commission\"), and MD&A liquidity. Each "
        "Sales Agreement with a named bank (Maxim, B. Riley, "
        "Jefferies, H.C. Wainwright, Cantor, etc.) is its own program "
        "— two agreements with the SAME bank but DIFFERENT signing "
        "dates are TWO DISTINCT programs, not amendments of one. The "
        "filing should state total capacity (often equal to the parent "
        "shelf available), cumulative sales-to-date, and remaining "
        "capacity. Terminations are usually disclosed in Subsequent "
        "Events or in the equity-note narrative."
    ),
    ("fpi_annual", "atm"): (
        "Locate ATM facilities in a 20-F / 40-F equity / share-capital "
        "note or Item 5 liquidity discussion. ATMs are rare for FPIs "
        "but do occur (typically with US-based sales agents). Use the "
        "same (sales_agent, agreement_date) identity convention. "
        "Convenience-translation to USD applies when applicable."
    ),
    ("us_periodic", "equity_line"): (
        "Locate equity-line / ELOC / standby purchase facilities in: "
        "the equity / capital-stock note (often titled \"Standby "
        "Equity Purchase Agreement,\" \"Common Stock Purchase "
        "Agreement,\" \"Equity Line of Credit,\" or \"Pre-Paid "
        "Advance Agreement\"), MD&A liquidity, and the cash-flow "
        "footnotes. The named investor is typically Yorkville, M2B "
        "Funding, Lincoln Park, Tumim Stone, Williams Trading, "
        "Tysadco, or a similar microcap-focused private fund. Each "
        "Purchase Agreement is its own facility — successive "
        "agreements with the same investor (re-ups) are DISTINCT "
        "instruments, not amendments. Extract total committed "
        "capacity, drawn-to-date, and termination status."
    ),
    ("fpi_annual", "equity_line"): (
        "Locate equity-line / standby-purchase facilities in a 20-F / "
        "40-F equity note or Item 5 liquidity discussion. Less common "
        "for FPIs but does occur. Same (investor, agreement_date) "
        "identity convention. Convenience-translation to USD applies."
    ),
}


# ─── Prompt scaffolding shared by all three specialists ─────────────
_GLOBAL_RULES = """\
General rules:
- Extract numbers VERBATIM — never multiply, divide, or otherwise
  transform a stated figure. Exception: the cumulative "to-date"
  capacity fields for shelves / ATMs / equity lines (drawn / sold to
  date) may be accumulated across periods when the filing reports them
  per-period — those field semantics say so explicitly.
- NEVER source counts, principal, or capacity from a BENEFICIAL-
  OWNERSHIP / MAJOR-SHAREHOLDER / insider-ownership table or its
  footnotes (20-F Items 6–7 "Directors, Senior Management and
  Employees" / "Major Shareholders"; 10-K or proxy "Security Ownership
  of Certain Beneficial Owners and Management"). Those rows state ONE
  holder's position — a single shareholder's warrant / option /
  convertible / preferred block, often quoted in underlying ordinary
  shares — and are a FRACTION of the tranche, not its total. They
  frequently carry tranche-like detail (strike, expiry) that makes them
  look authoritative; they are not. Take outstanding totals only from
  issuer-level disclosures: the capital-stock / securities note, the
  warrant / debt / equity roll-forward and cap tables, and the
  balance-sheet face.
- Use null for any field the filing does not state.
- Dates: YYYY-MM-DD or YYYY-only.
- notes: short context (floor/ceiling, call features, conversion-price
  adjustments). Keep ≤120 characters; this is a tag, not a quotation.
- Skip already-exercised / already-converted / cancelled / redeemed
  instruments. Only emit rows that remain OUTSTANDING as of period end.
"""


# Per-row units declaration for shelf-family dollar fields. The
# financial-statement notes that disclose shelf / ATM / equity-line
# capacity are routinely tabulated "(in thousands)" (sometimes "(in
# millions)"), so a verbatim read of "30,000" is really $30,000,000.
# We do NOT ask the model to multiply (the verbatim rule above guards
# against flaky mental math); it only REPORTS the scale it sees, and
# code rescales deterministically.
_DOLLAR_SCALE_RULE = """\
- dollar_scale: the units in which THIS row's AGGREGATE dollar figures
  (capacity / principal / aggregate liquidation preference / drawn /
  remaining — whichever this row has) are stated, read from the
  governing column or statement header. One of: "ones" (figures are
  full dollars, e.g. cover-page or prose "$30,000,000" / "$30.0
  million"), "thousands" (table headed "(in thousands)" — a stated
  "30,000" means $30,000,000), or "millions" (headed "(in millions)").
  This applies to AGGREGATE dollar columns ONLY — never to per-share
  prices (conversion_price / strike_price) or share counts, which are
  always read verbatim. Report the scale of the figure you actually
  pulled; do NOT convert the number yourself. Default "ones" only when
  no scale is indicated.
"""


# ─── Warrant prompt ─────────────────────────────────────────────────
_WARRANT_PROMPT = """\
You are extracting OUTSTANDING WARRANTS from an SEC periodic filing.

Context: this is the {form} for CIK {cik}, period of report {as_of_date}.

WARRANTS only. Convertible notes and preferred stock are extracted by
separate specialists — do NOT emit rows for them here, even when they
appear in the same transaction as a warrant. Common combo patterns:
  - Convertible preferred + accompanying warrants → emit the warrants
    here, the preferred elsewhere.
  - Convertible note + warrant coverage → emit the warrants here, the
    note elsewhere.

{section_anchor}

Field semantics:
- instrument_name: descriptive name from the filing. Include any
  identifying qualifier the filing uses ("November 2023 Common
  Warrants", "Placement Agent Warrants — March 2024 Offering",
  "Inducement Warrants").
- outstanding_count: explicit period-end count for this tranche.
- strike_price: USD per common share (or per ADS for FPI issuers).
  Extract the CURRENT (as-of-period-end) ADJUSTED price, not the
  initial issuance price.
- expiry: YYYY-MM-DD or YYYY-only.
- issue_date: when first issued.
- is_pre_funded: true when the filing identifies the tranche as
  "pre-funded warrants" (strike at or near $0.0001 / $0.001). Else null.
- notes: short tag; e.g. "Black-Scholes settlement on FCC",
  "subject to 4.99% beneficial-ownership cap".

{global_rules}

Multiple-tranche rule: if multiple warrant tranches are listed
separately (different strikes, different issue dates, different
counterparties), emit EACH as its own row. Never collapse into a
single "warrants outstanding: N" row.

Emit {{"warrants": []}} when the filing discloses no outstanding
warrants.

Filing text:
{text}
"""


# ─── Convertible prompt ─────────────────────────────────────────────
_CONVERTIBLE_PROMPT = """\
You are extracting OUTSTANDING CONVERTIBLE NOTES from an SEC periodic
filing.

Context: this is the {form} for CIK {cik}, period of report {as_of_date}.

CONVERTIBLE NOTES only. Warrants and preferred stock are extracted by
separate specialists — do NOT emit rows for them here, even when they
appear in the same transaction as a convertible note. If a note has
attached warrant coverage, emit the note here and let the warrant
specialist emit the warrants.

CONVERTIBLE PREFERRED STOCK is NOT a convertible note. It is preferred,
not debt, and is extracted by the preferred specialist. Filings often
use "convertible preferred" wording — that goes elsewhere.

{section_anchor}

Field semantics:
- instrument_name: descriptive name including the LENDER when the
  filing names one ("Streeterville Capital December 2022 Promissory
  Note", "April 2024 Senior Secured Convertible Note"). Lender
  identity is the key to merging tranches across periods.
- principal_amount: principal OUTSTANDING as of period end, in USD.
  Not the original face value if partial repayments / conversions
  have occurred. If the filing presents "principal outstanding"
  separately from "carrying value net of debt discount", USE the
  principal outstanding.
- conversion_price: USD per common share (or per ADS for FPI issuers).
  Extract the CURRENT (as-of-period-end) ADJUSTED price.
- common_shares_issuable: populate ONLY when the filing explicitly
  states the aggregate number of common shares (or ADSs) issuable on
  full conversion. Otherwise null — the downstream code derives it
  from principal_amount / conversion_price.
- maturity: YYYY-MM-DD or YYYY-only.
- issue_date: when first issued.
- notes: short tag; e.g. "10% OID", "secured by all assets",
  "subject to 9.99% beneficial-ownership cap".
{dollar_scale_rule}
{global_rules}

Multiple-tranche rule: if multiple convertible notes are listed
separately (different lenders, different conversion prices, different
maturities), emit EACH as its own row. Roll-forward tables that show
"December 2022 Note" and "April 2024 Note" as separate lines should
emit two rows even when they share a lender.

Emit {{"convertibles": []}} when the filing discloses no outstanding
convertible notes.

Filing text:
{text}
"""


# ─── Preferred prompt ───────────────────────────────────────────────
_PREFERRED_PROMPT = """\
You are extracting OUTSTANDING PREFERRED STOCK from an SEC periodic
filing.

Context: this is the {form} for CIK {cik}, period of report {as_of_date}.

PREFERRED STOCK only. Warrants and convertible NOTES are extracted by
separate specialists — do NOT emit rows for them here, even when they
appear in the same transaction as the preferred. Combo patterns:
  - Preferred + accompanying warrants → emit the preferred here, the
    warrants elsewhere.

"Convertible preferred stock" IS preferred and belongs here. Only
ACTUAL debt (notes payable / convertible notes / debentures) belongs
to the convertible specialist.

{section_anchor}

Field semantics:
- instrument_name: descriptive name from the filing. Include the
  series identifier ("Series A Convertible Preferred Stock",
  "Series 9 Preferred").
- series_letter: the series identifier alone — "A", "B", "9", etc.
  Extract this even when also encoded in instrument_name; downstream
  code uses it as the strongest tranche-identity key.
- outstanding_count: explicit period-end shares outstanding for this
  series. Fractional values are real (filings often state
  count = aggregate / stated_value, e.g. $53,197,000 / $1,000 =
  53,197.000 preferred shares) — preserve the fraction.
- conversion_price: USD per common share (or per ADS for FPI issuers).
  Extract the CURRENT (as-of-period-end) ADJUSTED price.
- aggregate_liquidation_preference: TOTAL dollar liquidation
  preference for the entire series outstanding — not per-share.
  Includes stated value × shares plus any cumulative accrued
  dividends if the filing states an aggregate.
- common_shares_issuable: populate ONLY when the filing explicitly
  states the aggregate number of common shares (or ADSs) issuable on
  full conversion. Otherwise null — the downstream code derives it.
- issue_date: when first issued.
- notes: short tag; e.g. "mandatorily redeemable 2027",
  "8% cumulative dividend", "perpetual".
{dollar_scale_rule}
{global_rules}

Multiple-series rule: each series (A, B, C, …) is its own row. NEVER
collapse two series into one row even when they share economic terms.

Emit {{"preferreds": []}} when the filing discloses no outstanding
preferred stock.

Filing text:
{text}
"""


# ─── Shelf prompt ───────────────────────────────────────────────────
_SHELF_PROMPT = """\
You are extracting ACTIVE SHELF REGISTRATIONS from an SEC periodic
filing.

Context: this is the {form} for CIK {cik}, period of report {as_of_date}.

SHELVES only (Form S-3, S-3/A, S-3ASR, S-3MEF, F-3, F-3/A, F-3ASR, or
the Reg A+ offering circular Form 1-A). ATM programs and equity-line
facilities are extracted by separate specialists — do NOT emit rows
for them here, even when they share a parent shelf. Warrants /
convertibles / preferreds are also separate.

{section_anchor}

Field semantics:
- instrument_name: descriptive name from the filing ("June 2021 $350M
  Shelf", "April 2024 F-3 Shelf"). Optional.
- file_number: the SEC Securities Act file number in the form
  "333-XXXXXX" (six digits typical). This is the canonical identity
  key and is usually quoted on the cover page or in the registration
  statement reference. Required when the filing names it.
- form: the registration form ("S-3", "F-3", "S-3ASR", "F-3ASR",
  "S-3MEF", "1-A").
- total_capacity_usd: the original DOLLAR CAPACITY registered under
  this shelf. Verbatim from the filing — do NOT subtract proceeds.
- drawn_to_date_usd: CUMULATIVE dollar amount the issuer has actually
  RAISED under this shelf through period end. Sum of all take-downs to
  date. If the filing only states the remaining-available figure,
  leave this null — the downstream code derives it.
- remaining_capacity_usd: AMOUNT REMAINING AVAILABLE for future
  take-downs. Verbatim if stated.
- effect_date: date the shelf became effective (YYYY-MM-DD). Usually
  3 weeks after the S-3 filing. Optional.
- issue_date: filing date of the S-3 itself (YYYY-MM-DD). Optional.
- is_terminated: true ONLY when the filing explicitly states the shelf
  has been withdrawn (Form RW filed), expired, or is otherwise no
  longer available. SEC Rule 415(a)(5) caps shelves at 3 years from
  effective date but the filing's own statement is the authoritative
  signal — if it's silent on termination, leave null.
- notes: short tag; e.g. "ASR auto-effective", "baby-shelf restricted",
  "subject to Form S-3 General Instruction I.B.6 limitation".
{dollar_scale_rule}
{global_rules}

Multiple-shelf rule: if the issuer has multiple active shelves
(different file numbers — common when a new S-3 is filed before the
prior one expires), emit EACH as its own row. Do NOT collapse.

ONLY shelves the filing CURRENTLY describes as outstanding /
available. Skip shelves that the filing explicitly states are
withdrawn / expired / fully consumed UNLESS the filing also gives
the file_number and capacity — in which case emit the row with
is_terminated=true so the ledger can record the closure.

Emit {{"shelves": []}} when the filing discloses no shelf
registrations.

Filing text:
{text}
"""


# ─── ATM prompt ─────────────────────────────────────────────────────
_ATM_PROMPT = """\
You are extracting AT-THE-MARKET (ATM) SALES PROGRAMS from an SEC
periodic filing.

Context: this is the {form} for CIK {cik}, period of report {as_of_date}.

ATM PROGRAMS only. A SHELF is the parent registration capacity; an
ATM program is a specific Sales Agreement with a named bank that
draws against a shelf (or is registered standalone). They are
distinct instruments — extract the ATM here and the shelf elsewhere.
Equity-line facilities (Yorkville / M2B / Lincoln Park / etc.) are
NOT ATMs — those are extracted by the equity_line specialist.

{section_anchor}

Field semantics:
- instrument_name: descriptive name ("July 2022 Maxim ATM",
  "Jefferies Common Stock Sales Agreement"). Optional.
- sales_agent: the bank acting as sales agent / underwriter on the
  Sales Agreement. Short canonical form: "Maxim", "Jefferies",
  "B. Riley", "Cantor", "H.C. Wainwright", "Roth", "Ladenburg",
  "ThinkEquity", "JMP Securities", "Brookline", "Aegis", etc. Required
  when the filing names one.
- agreement_date: the date the Sales Agreement was signed / executed.
  YYYY-MM-DD. PRIMARY IDENTITY KEY together with sales_agent — two
  Maxim ATMs signed 11 months apart are TWO DISTINCT programs. If the
  filing states only a month / year, pin to the 1st of the month.
- total_capacity_usd: total dollar capacity under the Sales Agreement
  (often equal to the parent shelf's available capacity, but stated
  separately). Verbatim.
- sold_to_date_usd: CUMULATIVE dollar amount sold under THIS Sales
  Agreement through period end. Common phrasing: "during the year
  ended ... we sold $X under the [Maxim] ATM" — accumulate across
  periods if the filing presents per-period numbers.
- remaining_capacity_usd: amount remaining under THIS agreement.
  Verbatim if stated.
- is_terminated: true ONLY when the filing explicitly states the
  Sales Agreement has been terminated. Silence → null.
- issue_date: same as agreement_date; redundant but the schema accepts
  it for consistency.
- notes: short tag; e.g. "sales-agent commission 3.0%", "limited to
  baby-shelf cap", "drew from June 2021 shelf".
{dollar_scale_rule}
{global_rules}

Multiple-program rule: each Sales Agreement is its own row. NEVER
collapse two Sales Agreements into one row even when they share the
same sales agent.

Emit {{"atms": []}} when the filing discloses no ATM programs.

Filing text:
{text}
"""


# ─── Equity-line prompt ─────────────────────────────────────────────
_EQUITY_LINE_PROMPT = """\
You are extracting EQUITY LINES OF CREDIT (ELOC) and STANDBY EQUITY
PURCHASE FACILITIES from an SEC periodic filing.

Context: this is the {form} for CIK {cik}, period of report {as_of_date}.

EQUITY-LINE FACILITIES only. ATM programs (with a named SALES AGENT
bank) are extracted by a separate specialist — equity lines have a
NAMED INVESTOR (a private fund), not a sales-agent bank. If the
filing uses both labels (rare), the presence of a named purchaser
investor and a fixed pricing formula tied to VWAP indicates an
equity line.

Common equity-line investors: Yorkville (YA II PN), M2B Funding,
Lincoln Park Capital, Tumim Stone Capital, Williams Trading,
Tysadco Partners, Crom Cortana, B. Riley Principal Capital. Common
agreement names: "Standby Equity Purchase Agreement," "Common Stock
Purchase Agreement," "Equity Line of Credit," "Pre-Paid Advance
Agreement," "ChEF (Committed Equity Financing Facility)".

{section_anchor}

Field semantics:
- instrument_name: descriptive name ("March 2025 Yorkville ELOC",
  "April 2026 M2B Funding Standby"). Optional.
- investor: the named PURCHASER / funder. Short canonical form:
  "Yorkville", "M2B Funding", "Lincoln Park", "Tumim Stone",
  "B. Riley", etc. Required when the filing names one.
- agreement_date: date the Purchase Agreement was signed / executed.
  YYYY-MM-DD. PRIMARY IDENTITY KEY together with investor — successive
  agreements with the same investor (re-ups every 12-18 months are
  common) are DISTINCT instruments.
- total_capacity_usd: total dollar commitment under the agreement.
  Verbatim.
- drawn_to_date_usd: CUMULATIVE dollar amount drawn / advanced under
  THIS agreement through period end.
- remaining_capacity_usd: amount remaining under THIS agreement.
- is_terminated: true ONLY when the filing explicitly states the
  Purchase Agreement has been terminated. Silence → null.
- issue_date: same as agreement_date; redundant but accepted.
- notes: short tag; e.g. "97% of VWAP", "11.99% beneficial-ownership
  cap", "drew from December 2024 shelf".
{dollar_scale_rule}
{global_rules}

Multiple-facility rule: each Purchase Agreement is its own row.

Emit {{"equity_lines": []}} when the filing discloses no equity-line
facilities.

Filing text:
{text}
"""


# ─── Helpers ───────────────────────────────────────────────────────
def _load_filing_text(accession: str) -> str | None:
    """Pull the periodic filing's primary doc out of dilution_raw.
    Prefers the form-typed row over EX-* exhibits, falls back to longest.
    Whitespace-normalized before it reaches the prompt — see
    `normalize_filing_text` (token savings + Gemini 400 guard)."""
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
            return normalize_filing_text(r["content_md"])
    return normalize_filing_text(rows[0]["content_md"])


def _num(x):
    if x is None or x == "":
        return None
    try:
        return float(str(x).replace(",", "").replace("$", ""))
    except ValueError:
        return None


_SCALE_FACTORS = {"ones": 1.0, "thousands": 1_000.0, "millions": 1_000_000.0}


def _scale_factor(dollar_scale: str | None) -> float:
    """Resolve a model-reported `dollar_scale` token into a multiplier.

    Periodic financial-statement tables disclose shelf-family capacity
    "(in thousands)" far more often than in full dollars; reading them
    verbatim divides every dollar figure by 1000 (GCTK: a $30M shelf
    booked as $30,000). The model reports the units header it saw and
    we apply the multiplier here. Unknown / null → 1.0 (no-op)."""
    if not dollar_scale:
        return 1.0
    return _SCALE_FACTORS.get(dollar_scale.strip().lower(), 1.0)


def _scaled(value: float | None, factor: float) -> float | None:
    return value * factor if value is not None else None


def _clean_warrant_row(
    row: WarrantOverhangRow, ads_ratio: float | None,
) -> dict:
    cnt = _num(row.outstanding_count)
    out = {
        "category": "warrant",
        "instrument_name": (row.instrument_name or "").strip() or None,
        "outstanding_count": cnt,
        "common_shares_issuable": cnt,  # 1:1 by construction for warrants
        "strike_or_conversion_price": _num(row.strike_price),
        "principal_amount": None,
        "maturity_or_expiry": (row.expiry or "").strip() or None,
        "issue_date": (row.issue_date or "").strip() or None,
        "notes": (row.notes or "").strip() or None,
        # Structured carry-through for downstream is_pre_funded detection.
        "is_pre_funded": row.is_pre_funded,
    }
    return out


def _clean_convertible_row(
    row: ConvertibleOverhangRow, ads_ratio: float | None,
) -> dict:
    # principal_amount is an aggregate dollar figure and is routinely
    # tabulated "(in thousands)"; rescale it before deriving CSI.
    # conversion_price (per-share) and common_shares_issuable (a share
    # count) are read verbatim — dollar_scale never applies to them.
    sf = _scale_factor(row.dollar_scale)
    pa = _scaled(_num(row.principal_amount), sf)
    cp = _num(row.conversion_price)
    csi = _num(row.common_shares_issuable)
    # Derive when not stated (from the already-scaled principal).
    if csi is None and pa is not None and cp and cp > 0:
        csi = pa / cp
    csi = _apply_ads_normalization(csi, pa, cp, ads_ratio)
    return {
        "category": "convertible",
        "instrument_name": (row.instrument_name or "").strip() or None,
        "outstanding_count": None,
        "common_shares_issuable": csi,
        "strike_or_conversion_price": cp,
        "principal_amount": pa,
        "maturity_or_expiry": (row.maturity or "").strip() or None,
        "issue_date": (row.issue_date or "").strip() or None,
        "notes": (row.notes or "").strip() or None,
    }


def _clean_preferred_row(
    row: PreferredOverhangRow, ads_ratio: float | None,
) -> dict:
    # aggregate_liquidation_preference is an aggregate dollar figure
    # (commonly "(in thousands)" on the balance sheet); rescale it before
    # deriving CSI. outstanding_count (shares) and conversion_price
    # (per-share) are read verbatim — dollar_scale never applies.
    sf = _scale_factor(row.dollar_scale)
    cnt = _num(row.outstanding_count)
    cp = _num(row.conversion_price)
    alp = _scaled(_num(row.aggregate_liquidation_preference), sf)
    csi = _num(row.common_shares_issuable)
    if csi is None:
        if alp is not None and cp and cp > 0:
            csi = alp / cp
        elif cnt is not None and cp and cp > 0:
            # Some filings state per-share economics but not aggregate.
            # Fall back to count as a lower-bound proxy for CSI.
            csi = cnt
    return {
        "category": "preferred",
        "instrument_name": (row.instrument_name or "").strip() or None,
        "series_letter": (row.series_letter or "").strip() or None,
        "outstanding_count": cnt,
        "common_shares_issuable": csi,
        "strike_or_conversion_price": cp,
        "principal_amount": alp,
        "maturity_or_expiry": None,
        "issue_date": (row.issue_date or "").strip() or None,
        "notes": (row.notes or "").strip() or None,
    }


def _clean_shelf_row(
    row: ShelfOverhangRow, ads_ratio: float | None,
) -> dict:
    """Normalize a ShelfOverhangRow into the unified anchor dict shape.
    Shelf-family rows reuse `outstanding_count` / `principal_amount` /
    `strike_or_conversion_price` slots — they're null for shelves. The
    relevant fields live under shelf-specific keys consumed by
    anchor.py's shelf matcher / field_changes path.
    """
    sf = _scale_factor(row.dollar_scale)
    return {
        "category": "shelf",
        "instrument_name": (row.instrument_name or "").strip() or None,
        "outstanding_count": None,
        "common_shares_issuable": None,
        "strike_or_conversion_price": None,
        "principal_amount": None,
        "maturity_or_expiry": None,
        "issue_date": (row.issue_date or "").strip() or None,
        "notes": (row.notes or "").strip() or None,
        # Shelf-family specific identity + capacity fields.
        "file_number": (row.file_number or "").strip() or None,
        "form": (row.form or "").strip() or None,
        "total_capacity_usd": _scaled(_num(row.total_capacity_usd), sf),
        "drawn_to_date_usd": _scaled(_num(row.drawn_to_date_usd), sf),
        "remaining_capacity_usd": _scaled(_num(row.remaining_capacity_usd), sf),
        "effect_date": (row.effect_date or "").strip() or None,
        "is_terminated": row.is_terminated,
    }


def _clean_atm_row(
    row: ATMOverhangRow, ads_ratio: float | None,
) -> dict:
    sf = _scale_factor(row.dollar_scale)
    return {
        "category": "atm",
        "instrument_name": (row.instrument_name or "").strip() or None,
        "outstanding_count": None,
        "common_shares_issuable": None,
        "strike_or_conversion_price": None,
        "principal_amount": None,
        "maturity_or_expiry": None,
        "issue_date": (row.issue_date or "").strip() or None,
        "notes": (row.notes or "").strip() or None,
        "sales_agent": (row.sales_agent or "").strip() or None,
        "agreement_date": (row.agreement_date or "").strip() or None,
        "total_capacity_usd": _scaled(_num(row.total_capacity_usd), sf),
        "drawn_to_date_usd": _scaled(_num(row.sold_to_date_usd), sf),
        "remaining_capacity_usd": _scaled(_num(row.remaining_capacity_usd), sf),
        "is_terminated": row.is_terminated,
    }


def _clean_equity_line_row(
    row: EquityLineOverhangRow, ads_ratio: float | None,
) -> dict:
    sf = _scale_factor(row.dollar_scale)
    return {
        "category": "equity_line",
        "instrument_name": (row.instrument_name or "").strip() or None,
        "outstanding_count": None,
        "common_shares_issuable": None,
        "strike_or_conversion_price": None,
        "principal_amount": None,
        "maturity_or_expiry": None,
        "issue_date": (row.issue_date or "").strip() or None,
        "notes": (row.notes or "").strip() or None,
        "investor": (row.investor or "").strip() or None,
        "agreement_date": (row.agreement_date or "").strip() or None,
        "total_capacity_usd": _scaled(_num(row.total_capacity_usd), sf),
        "drawn_to_date_usd": _scaled(_num(row.drawn_to_date_usd), sf),
        "remaining_capacity_usd": _scaled(_num(row.remaining_capacity_usd), sf),
        "is_terminated": row.is_terminated,
    }


def _apply_ads_normalization(
    csi: float | None, pa: float | None, cp: float | None,
    ads_ratio: float | None,
) -> float | None:
    """ADS unit fix-up. When the LLM mixed units (principal in USD,
    conversion price per ADS but CSI in ordinary shares), restore
    consistency by dividing CSI back to ADS."""
    if not (ads_ratio and ads_ratio >= 2 and csi and pa and cp and cp > 0):
        return csi
    implied_ads = pa / cp
    if implied_ads > 0 and abs((csi / implied_ads) - ads_ratio) / ads_ratio < 0.05:
        return csi / ads_ratio
    return csi


# ─── Merged-prompt assembly (v5) ────────────────────────────────────
# One call extracts all six types. The merged prompt is composed from
# the SAME six `_*_PROMPT` templates so each type's section anchors /
# field semantics / exclusion rules survive verbatim; the filing text
# and the shared global rules are factored out to appear exactly once.

_COMBINED_SYS_MSG = (
    "You extract ALL outstanding dilutive instruments — warrants, "
    "convertible notes, preferred stock, shelf registrations, ATM "
    "programs, and equity-line facilities — from one SEC periodic "
    "filing in a single pass. Output strictly conforms to the "
    "CombinedOverhangList schema: every instrument goes in exactly one "
    "of the six lists."
)

_COMBINED_INTRO = """\
You are extracting EVERY outstanding dilutive instrument from one SEC
periodic filing, sorting each into the correct one of six lists:
warrants, convertibles (notes), preferreds, shelves, atms, equity_lines.

This is the {form} for CIK {cik}, period of report {as_of_date}.

Put each instrument in EXACTLY ONE list by its type. The six
type-specific instruction blocks below were each written as a
standalone specialist; where a block says another type is "extracted
by a separate specialist," read that as "that instrument belongs in
its OWN list" — route it there, do not drop it. Type boundaries that
matter most:
  - "Convertible preferred stock" is PREFERRED (preferreds list), NOT a
    convertible note. Only actual debt (notes payable / debentures /
    convertible notes) goes in convertibles.
  - A SHELF is the parent registration (Form S-3/F-3 capacity); an ATM
    is a Sales Agreement with a named SALES-AGENT bank drawing on it;
    an EQUITY LINE has a named private-fund INVESTOR (Yorkville, M2B,
    Lincoln Park…) with a VWAP-linked price. Three distinct lists.
  - A note/preferred with attached warrant coverage emits the warrants
    in `warrants` AND the host instrument in its own list.

Emit an empty list for any type the filing does not disclose. The
general rules and dollar-scale rule at the end apply to every block.

"""


def _type_instruction_block(
    template: str, *, form, cik: int, as_of: str, anchor: str,
) -> str:
    """Render one specialist template with the shared global rules,
    dollar-scale rule, and trailing filing-text stripped — leaving only
    that type's instructions (intro, exclusions, section anchor, field
    semantics, multiple-tranche rule, empty-list note). str.format
    ignores the extra kwargs a given template doesn't reference (e.g.
    the warrant template has no {dollar_scale_rule})."""
    rendered = template.format(
        form=form, cik=cik, as_of_date=as_of, section_anchor=anchor,
        global_rules="", dollar_scale_rule="", text="",
    )
    # Everything before the "Filing text:" trailer is the instruction body.
    return rendered.partition("\nFiling text:")[0].rstrip()


_MERGE_SPECS = (
    (_WARRANT_PROMPT, "warrant"),
    (_CONVERTIBLE_PROMPT, "convertible"),
    (_PREFERRED_PROMPT, "preferred"),
    (_SHELF_PROMPT, "shelf"),
    (_ATM_PROMPT, "atm"),
    (_EQUITY_LINE_PROMPT, "equity_line"),
)


def _build_combined_prompt(
    *, form: str, cik: int, as_of: str, family: str, text: str,
) -> str:
    """Assemble the single merged user prompt: intro + six per-type
    instruction blocks + shared rules (once) + filing text (once)."""
    blocks = []
    for template, cat in _MERGE_SPECS:
        anchor = _SECTION_ANCHORS[(family, cat)]
        header = f"════════ {cat.upper()} ════════"
        blocks.append(
            header + "\n"
            + _type_instruction_block(
                template, form=form, cik=cik, as_of=as_of, anchor=anchor,
            )
        )
    return (
        _COMBINED_INTRO.format(form=form, cik=cik, as_of_date=as_of)
        + "\n\n".join(blocks)
        + "\n\n════════ GENERAL RULES (apply to all lists) ════════\n"
        + _DOLLAR_SCALE_RULE + "\n" + _GLOBAL_RULES
        + "\n\nFiling text:\n" + text
    )


# ─── Truncated-output salvage (v7) ──────────────────────────────────
# A degenerate numeric repetition loop drives the model to the token cap
# mid-number, so the structured response is cut off partway through a row
# (and the runaway integer also trips json.loads' 4300-digit ceiling).
# Rather than discard the whole — expensive — response and trigger the 6×
# per-specialist fallback (which re-loops on the same number), recover the
# largest structurally-valid prefix: every row emitted before the poisoned
# one. anchor.reconcile_against_periodic degrades safely on the missing
# tail (it never closes ledger rows on overhang silence), so a partial
# list is strictly better than an empty one.

# Any contiguous run of this many digits is a runaway, not a real figure:
# the largest legitimate value we extract is an aggregate dollar amount
# (~$1e12, 13 digits) or a share count (~1e10, 11 digits). Neutralize such
# runs to 0 so json.loads can't choke on the int-string-length limit even
# when a complete-but-absurd number sits inside the salvaged prefix.
_RUNAWAY_DIGITS = re.compile(r"\d{20,}")


def _salvage_truncated_json(raw: str) -> dict | None:
    """Recover the largest valid JSON-object prefix from truncated output.

    Walks `raw` tracking string state and the container stack, recording
    the position just past every closed container (`}` / `]`). The last
    such position ends the last fully-emitted element; everything after it
    (a half-written row, or a runaway number that ran to the token cap) is
    dropped, the still-open containers are closed, and the repaired text is
    parsed. Returns the parsed dict, or None if there is no closeable
    prefix or the repair still won't parse.
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    cut: int | None = None              # exclusive index past last closer
    cut_stack: tuple[str, ...] = ()     # containers still open at `cut`
    for i, ch in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                break                   # unbalanced — stop scanning
            stack.pop()
            cut = i + 1
            cut_stack = tuple(stack)
    if cut is None:
        return None
    closers = "".join("}" if c == "{" else "]" for c in reversed(cut_stack))
    repaired = _RUNAWAY_DIGITS.sub("0", raw[:cut]) + closers
    try:
        parsed = json.loads(repaired)
    except (ValueError, RecursionError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _row_count(payload: dict) -> int:
    """Total rows across the list-valued fields of a parsed overhang dict."""
    return sum(len(v) for v in payload.values() if isinstance(v, list))


def _parse_overhang_response(response, model, *, accession: str, handler: str):
    """Parse a structured overhang response into `model`.

    Fast path: json.loads + model_validate. On failure, if the model
    truncated the output at the token cap — the signature of a numeric
    repetition loop running to the ceiling — salvage the largest valid
    prefix and validate that instead, recovering every row emitted before
    the poisoned one. Returns the parsed model, or None if even salvage
    yields nothing usable.
    """
    raw = output_text(response)
    try:
        return model.model_validate(json.loads(raw))
    except (ValueError, ValidationError) as exc:
        if not truncated(response):
            log.warning("%s %s — parse failed (%s)", handler, accession, exc)
            return None
        salvaged = _salvage_truncated_json(raw)
        if salvaged is None:
            log.warning("%s %s — truncated at token cap; salvage found no "
                        "valid prefix (%s)", handler, accession, exc)
            return None
        try:
            parsed = model.model_validate(salvaged)
        except (ValueError, ValidationError) as exc2:
            log.warning("%s %s — truncated at token cap; salvaged prefix "
                        "still invalid (%s)", handler, accession, exc2)
            return None
        log.info("%s %s — truncated at token cap; salvaged %d row(s) from "
                 "the valid prefix", handler, accession, _row_count(salvaged))
        return parsed


async def _extract_per_specialist_lists(
    *, client, accession: str, preamble: str, form: str, cik: int,
    as_of: str, family: str, text: str,
) -> tuple[list, list, list, list, list, list]:
    """Fallback path (the pre-v5 behavior): run the six category
    specialists in parallel, each on its own smaller prompt + schema.

    Used ONLY when the merged single-call extraction fails to parse —
    large/complex filings occasionally produce malformed combined JSON,
    and a failure there would otherwise drop all six categories for the
    filing. Six small independent calls parse far more reliably and fail
    in isolation (one specialist returning [] doesn't sink the others).

    Returns the six lists in (warrants, convertibles, preferreds,
    shelves, atms, equity_lines) order — the same shape the merged path
    unpacks — so the caller's cleaning loop is identical on both paths.
    """
    def _fmt_args(category: str) -> dict:
        return dict(
            form=form, cik=cik, as_of_date=as_of,
            section_anchor=_SECTION_ANCHORS[(family, category)],
            global_rules=_GLOBAL_RULES,
            dollar_scale_rule=_DOLLAR_SCALE_RULE,
            text=text,
        )

    async def run_one(
        *, prompt: str, response_model: type[BaseModel],
        list_attr: str, handler_label: str, sys_msg: str,
    ):
        try:
            response = check_response(
                await acomplete(
                    client,
                    name=handler_label,
                    messages=[system(sys_msg),
                              user(preamble + "\n" + prompt)],
                    response_format=response_model,
                    max_output_tokens=OVERHANG_MAX_TOKENS,
                    model=config.LLM_MODEL_PERIODIC,
                    cache_key=f"overhang-{list_attr}",
                ),
                accession=accession, handler=handler_label,
            )
        except Exception as exc:
            # One specialist's transport error must not cancel its siblings
            # (asyncio.gather with return_exceptions=False would propagate).
            # Degrade this category to empty; others stand.
            log.warning("%s %s — specialist call failed (%s): %s",
                        handler_label, accession, type(exc).__name__, exc)
            return []
        # Shared parse path also salvages a truncated specialist response —
        # the poisoned filing that broke the merged call tends to re-loop
        # here too, and salvage recovers the rows before the runaway.
        parsed = _parse_overhang_response(
            response, response_model,
            accession=accession, handler=handler_label,
        )
        if parsed is None:
            return []
        return getattr(parsed, list_attr, [])

    return await asyncio.gather(
        run_one(
            prompt=_WARRANT_PROMPT.format(**_fmt_args("warrant")),
            response_model=WarrantOverhangList, list_attr="warrants",
            handler_label="overhang-warrant",
            sys_msg=("You extract outstanding WARRANTS from SEC periodic "
                     "filings. Output strictly conforms to the "
                     "WarrantOverhangList schema."),
        ),
        run_one(
            prompt=_CONVERTIBLE_PROMPT.format(**_fmt_args("convertible")),
            response_model=ConvertibleOverhangList, list_attr="convertibles",
            handler_label="overhang-convertible",
            sys_msg=("You extract outstanding CONVERTIBLE NOTES from SEC "
                     "periodic filings. Output strictly conforms to the "
                     "ConvertibleOverhangList schema."),
        ),
        run_one(
            prompt=_PREFERRED_PROMPT.format(**_fmt_args("preferred")),
            response_model=PreferredOverhangList, list_attr="preferreds",
            handler_label="overhang-preferred",
            sys_msg=("You extract outstanding PREFERRED STOCK from SEC "
                     "periodic filings. Output strictly conforms to the "
                     "PreferredOverhangList schema."),
        ),
        run_one(
            prompt=_SHELF_PROMPT.format(**_fmt_args("shelf")),
            response_model=ShelfOverhangList, list_attr="shelves",
            handler_label="overhang-shelf",
            sys_msg=("You extract ACTIVE SHELF REGISTRATIONS from SEC "
                     "periodic filings. Output strictly conforms to the "
                     "ShelfOverhangList schema."),
        ),
        run_one(
            prompt=_ATM_PROMPT.format(**_fmt_args("atm")),
            response_model=ATMOverhangList, list_attr="atms",
            handler_label="overhang-atm",
            sys_msg=("You extract AT-THE-MARKET (ATM) SALES PROGRAMS "
                     "from SEC periodic filings. Output strictly conforms "
                     "to the ATMOverhangList schema."),
        ),
        run_one(
            prompt=_EQUITY_LINE_PROMPT.format(**_fmt_args("equity_line")),
            response_model=EquityLineOverhangList, list_attr="equity_lines",
            handler_label="overhang-equity-line",
            sys_msg=("You extract EQUITY-LINE / STANDBY EQUITY PURCHASE "
                     "facilities from SEC periodic filings. Output "
                     "strictly conforms to the EquityLineOverhangList "
                     "schema."),
        ),
        return_exceptions=False,
    )


# ─── Public entry point ─────────────────────────────────────────────
async def extract_overhang_rows(
    *, accession: str, form: str, filing_date: str,
    report_date: str | None, cik: int, client,
    unit_ctx: dict | None = None,
) -> list[dict]:
    """Extract ALL six instrument types from one periodic filing.

    Fast path (v5): ONE merged call returns all six types. If that call's
    output fails to parse — malformed combined JSON, which happens on
    very large filings (FCEL 10-Ks) — fall back to the six independent
    specialist calls for this filing (v6), preserving the pre-v5 fault
    isolation exactly where it's needed. Returns the unified
    OverhangRow-shaped dict list anchor.py and seed.py consume; [] only
    when the filing has no cached body.
    """
    text = _load_filing_text(accession)
    if not text:
        return []
    if len(text) > MAX_INPUT_CHARS:
        log.warning("overhang %s — truncated %d→%d chars",
                    accession, len(text), MAX_INPUT_CHARS)
        text = text[:MAX_INPUT_CHARS]

    preamble = unit_preamble(unit_ctx)
    as_of = report_date or filing_date
    family = _form_family(form)
    log.debug("overhang %s form=%s family=%s", accession, form, family)

    prompt = _build_combined_prompt(
        form=form, cik=cik, as_of=as_of, family=family, text=text,
    )
    # Parse with stdlib json (not pydantic's jiter) because LLMs
    # occasionally emit runaway-decimal floats like 0.0505050505...
    # repeated for thousands of chars (per-share dividend ratios on
    # FCEL/XTIA preferreds). jiter hard-rejects those as "number out of
    # range"; stdlib silently rounds to float64. _parse_overhang_response
    # additionally salvages a valid prefix when a runaway integer truncates
    # the output at the token cap, so a truncated merge usually still yields
    # its completed rows instead of falling all the way back.
    parsed = None
    try:
        response = check_response(
            await acomplete(
                client,
                name="overhang-combined",
                messages=[system(_COMBINED_SYS_MSG),
                          user(preamble + "\n" + prompt)],
                response_format=CombinedOverhangList,
                max_output_tokens=OVERHANG_MAX_TOKENS,
                model=config.LLM_MODEL_PERIODIC,
                cache_key="overhang-combined",
            ),
            accession=accession, handler="overhang-combined",
        )
    except Exception as exc:
        response = None
        log.warning("overhang-combined %s — merged call failed (%s: %s); "
                    "falling back to per-specialist extraction",
                    accession, type(exc).__name__, exc)
    if response is not None:
        parsed = _parse_overhang_response(
            response, CombinedOverhangList,
            accession=accession, handler="overhang-combined",
        )
        if parsed is None:
            log.warning("overhang-combined %s — merged parse failed; "
                        "falling back to per-specialist extraction",
                        accession)

    if parsed is not None:
        warrants = parsed.warrants
        convertibles = parsed.convertibles
        preferreds = parsed.preferreds
        shelves = parsed.shelves
        atms = parsed.atms
        equity_lines = parsed.equity_lines
        log.info(
            "overhang-combined %s — w=%d c=%d p=%d s=%d a=%d e=%d",
            accession, len(warrants), len(convertibles), len(preferreds),
            len(shelves), len(atms), len(equity_lines),
        )
    else:
        # Merge unusable (typically malformed combined JSON on a very
        # large filing). Fall back to the six independent specialist
        # calls: costs 6× the input on the ~5% of filings that hit this,
        # vs losing all overhang reconciliation for the filing.
        (warrants, convertibles, preferreds, shelves,
         atms, equity_lines) = await _extract_per_specialist_lists(
            client=client, accession=accession, preamble=preamble,
            form=form, cik=cik, as_of=as_of, family=family, text=text,
        )

    ads_ratio = (unit_ctx or {}).get("ads_ratio")
    cleaned: list[dict] = []
    for r in warrants:
        cleaned.append(_clean_warrant_row(r, ads_ratio))
    for r in convertibles:
        cleaned.append(_clean_convertible_row(r, ads_ratio))
    for r in preferreds:
        cleaned.append(_clean_preferred_row(r, ads_ratio))
    for r in shelves:
        cleaned.append(_clean_shelf_row(r, ads_ratio))
    for r in atms:
        cleaned.append(_clean_atm_row(r, ads_ratio))
    for r in equity_lines:
        cleaned.append(_clean_equity_line_row(r, ads_ratio))
    return cleaned


__all__ = [
    "HANDLER_VERSION",
    "MAX_INPUT_CHARS",
    "OVERHANG_CATEGORIES",
    "WarrantOverhangRow",
    "WarrantOverhangList",
    "ConvertibleOverhangRow",
    "ConvertibleOverhangList",
    "PreferredOverhangRow",
    "PreferredOverhangList",
    "ShelfOverhangRow",
    "ShelfOverhangList",
    "ATMOverhangRow",
    "ATMOverhangList",
    "EquityLineOverhangRow",
    "EquityLineOverhangList",
    "CombinedOverhangList",
    "extract_overhang_rows",
]
