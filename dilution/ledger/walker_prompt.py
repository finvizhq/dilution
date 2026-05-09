"""The single ledger-aware extraction prompt.

One prompt, one MutationList output schema, branched internally by
form bucket via a hint paragraph. NOT N form-specific prompts — see
LEDGER_REWORK_PLAN step 3 for the rationale.

Inputs the walker passes in:
  - issuer/ticker + unit context (FPI, ADS ratio)
  - rendered ledger view (from view.render_ledger_view)
  - filing metadata (form, filing_date, period_of_report, items, accession)
  - the filing's full text (capped at MAX_INPUT_CHARS in walker_llm)

Output:
  - MutationList — pydantic-validated; xAI structured outputs enforce
    natively, Moonshot json_object mode validates downstream.
"""

from __future__ import annotations


# ─── Form-bucket hints ──────────────────────────────────────────────
# One short paragraph per form family pointing the LLM at where in the
# filing the cap-table-relevant disclosures usually live. The walker
# selects the matching hint and inlines it.
_PERIODIC_HINT = (
    "Periodic-filing hint (10-K / 10-Q / 20-F / 40-F): the "
    "definitive overhang table is typically in the Notes to "
    "Financial Statements (Notes 8-12 commonly). Read those "
    "notes for the AS-OF-PERIOD-END outstanding warrants, "
    "convertible notes, preferred shares, and ATM/equity-line "
    "remaining capacity. The cover page may state shares "
    "outstanding. Do NOT re-create instruments that already "
    "exist in the ledger view — the walker will run a separate "
    "anchor reconciliation pass against the issuer's overhang "
    "table after this filing. Your job in the periodic filing "
    "is to record events that happened during the period (Item "
    "5 unregistered sales, Subsequent Events, Item 9B) — "
    "exercises, conversions, drawdowns, cancellations."
)

_FORM_HINTS: dict[str, str] = {
    "8-K": (
        "Form 8-K hint: dilution news lives in Items 1.01 (material "
        "agreement), 3.02 (unregistered sale of equity securities), "
        "3.03 (material modification to rights), 5.03 (amendments to "
        "charter/bylaws — usually the Series X Certificate of "
        "Designation), 7.01/8.01 (regulation FD / other), and the "
        "exhibits (EX-3.1 Cert of Designation, EX-4.x warrant or note "
        "form, EX-10.x securities purchase / underwriting / placement "
        "agency / ATM sales / equity-line agreement, EX-99.1 press "
        "release).\n\n"
        "ANNOUNCEMENT vs CLOSING: an 8-K may announce an offering at "
        "PRICING (terms set, deal not yet closed) or at CLOSING (deal "
        "actually funded). Both are real events — but if the offering "
        "was already priced in an earlier 8-K or 424B that's in the "
        "ledger view as a `drawn_down` entry, this 8-K is a "
        "re-disclosure of the SAME event. Emit NO mutations in that "
        "case. Specifically, an 8-K announcing 'closing of previously "
        "announced registered direct offering' against a shelf row "
        "that already shows the drawdown is a NO-OP — do NOT create "
        "an `s1_offering`, `equity`, or any other new instrument for "
        "it. The shelf+drawdown is the complete representation.\n\n"
        "If a same-day 424B was already processed (visible in ledger "
        "as a drawdown), the 8-K is a no-op — see 424B hint's DEDUP "
        "RULE."
    ),
    "424B": (
        "Form 424B hint: prospectus supplement carrying the priced "
        "terms of an offering. 424B5 is the most common (shelf "
        "takedown — registered direct or underwritten); 424B3 is "
        "often a resale prospectus or a base-prospectus supplement; "
        "424B4 is a final supplement at IPO; 424B7 sets terms for a "
        "selling-stockholder resale. Look at the cover page for "
        "shares × price = gross_proceeds, the underwriter / placement-"
        "agent name, any warrant coverage. For shelf-takedown 424Bs, "
        "the prior shelf instrument should already be in the ledger "
        "view — emit `record_event(drawdown)` against that shelf id "
        "rather than creating a new instrument. RESALE 424Bs do NOT "
        "create new dilution and should produce no mutations.\n\n"
        "ONE EVENT, ONE MUTATION. A shelf-takedown 424B5 is "
        "represented by EXACTLY ONE `record_event(drawdown)` against "
        "the shelf. Do NOT also emit `create_instrument(equity)` for "
        "the shares being sold — the share count and proceeds are "
        "tracked on the drawdown event itself. Do NOT emit "
        "`create_instrument(s1_offering)` either. The shelf row + "
        "drawdown is the complete representation. Only emit a "
        "separate `create_instrument` when the 424B5 also issues "
        "warrants or convertible notes alongside the common shares "
        "(those ARE distinct instruments — one warrant create per "
        "tranche).\n\n"
        "NEVER CREATE A SHELF CALLED 'TAKEDOWN'/'DRAWDOWN'. A takedown "
        "is an EVENT against a shelf, not an instrument. Labels like "
        "'September 2021 Shelf Takedown' or 'Shelf Drawdown' on a "
        "`create_instrument(type='shelf')` are ALWAYS WRONG. If the "
        "filing describes a takedown but the ledger view shows no "
        "prior shelf, emit `record_event(drawdown)` against the "
        "most-recently-created shelf id of any form (S-3 / S-3ASR / "
        "F-3) — DO NOT create a new shelf to host the drawdown. If "
        "no shelf exists in the ledger AT ALL, emit no mutations and "
        "let a later periodic-filing anchor reconcile the gap.\n\n"
        "DEDUP RULE: if the ledger view already shows a `drawn_down` "
        "history entry on a shelf/ATM whose date is within ~10 days "
        "of this 424B's filing date AND whose amount matches this "
        "424B's offering size within ~5%, this 424B is the priced "
        "supplement for the SAME offering an earlier 8-K/6-K already "
        "recorded — emit NO mutations (or at most an `amend_instrument` "
        "against the existing drawdown history if pricing terms are "
        "more precise here). Do NOT emit a second drawdown for the "
        "same offering."
    ),
    "S-1": (
        "Form S-1 hint: primary registration of a new offering. "
        "Create an `s1_offering` ledger row with anticipated_deal_size "
        "from the cover; the underwriter from the table of contents; "
        "and any inducement / pre-funded warrants typically issued in "
        "the same offering (each as its own warrant ledger row)."
    ),
    "F-1": (
        "Form F-1 hint: foreign-issuer analog of S-1. Same handling — "
        "create an `s1_offering` row + accompanying warrant rows. ADS "
        "issuers report all share counts in ADS units."
    ),
    "S-3": (
        "Form S-3 hint: shelf registration. A NEW BASE S-3 (no /A "
        "suffix) creates a NEW `shelf` ledger row with capacity_usd "
        "from the cover, form='S-3' (or 'S-3ASR' if auto-effective). "
        "EACH new BASE S-3 filing is a SEPARATE shelf, even when "
        "the issuer is renewing or replacing an expiring older "
        "shelf — DO NOT amend the old shelf row.\n\n"
        "S-3/A AND POS AM are AMENDMENTS to the prior shelf — emit "
        "`amend_instrument` against the most-recently-created shelf "
        "row of matching form (S-3 family). Use field_updates to "
        "carry any updated capacity_usd or other terms. Do NOT "
        "create a second `shelf` ledger row for an /A or POS AM. A "
        "single base S-3 commonly has 3-5 /A amendments before "
        "going effective; all of them collapse onto the one base "
        "shelf row.\n\n"
        "CAPACITY CHANGES on an /A are STILL AMENDMENTS — not new "
        "shelves. Issuers routinely file a base S-3 at one capacity "
        "(e.g. $100M) and then bump it via an /A before the SEC "
        "declares the registration effective (e.g. /A raises to "
        "$200M). The /A is amending the SAME registration statement, "
        "not registering a parallel shelf. Emit "
        "`amend_instrument(field_updates={capacity_usd: 200000000})` "
        "against the base S-3's ledger id; do NOT emit a fresh "
        "`create_instrument` with the new capacity.\n\n"
        "Subsequent 424B5 takedowns are drawdowns against the "
        "most-recently-created shelf id of matching form."
    ),
    "F-3": (
        "Form F-3 hint: foreign analog of S-3. A NEW BASE F-3 (no "
        "/A suffix) creates a NEW `shelf` ledger row with form='F-3'. "
        "EACH BASE F-3 filing is its own shelf; do NOT amend the "
        "prior shelf even when the new F-3 'replaces' or 'renews' "
        "it. ADS-denominated shelves carry units='ads'.\n\n"
        "F-3/A AND POS AM are AMENDMENTS — emit `amend_instrument` "
        "against the most-recently-created shelf row of matching "
        "form (F-3 family), NOT a new `create_instrument`. A single "
        "base F-3 commonly has 3-5 /A amendments before going "
        "effective; all of them collapse onto the one base shelf row.\n\n"
        "CAPACITY CHANGES on an /A are STILL AMENDMENTS — not new "
        "shelves. A base F-3 at $100M followed by an F-3/A at $200M "
        "is the SAME shelf with capacity bumped during the pre-"
        "effectiveness review. Emit `amend_instrument(field_updates="
        "{capacity_usd: 200000000})` against the base F-3's ledger "
        "id; do NOT emit a fresh `create_instrument` with the new "
        "capacity."
    ),
    "S-4": (
        "Form S-4 hint: M&A registration. Often results in newly-"
        "issued shares to the target's holders + sometimes assumed "
        "warrants/preferred. Create equity / warrant / preferred "
        "rows for the consideration."
    ),
    "DEF 14A": (
        "Proxy hint: shareholder votes on share-authorized increases, "
        "reverse splits, or compensation-plan capacity changes. A "
        "reverse-split proxy emits an `apply_split` mutation with "
        "the effective_date the proxy schedules (or with no date "
        "until a follow-up 8-K confirms — in that case, omit). "
        "Equity-incentive-plan increases don't create ledger rows in "
        "v1 (option_pool is out-of-scope for the cards we render)."
    ),
    "DEFM14A": (
        "M&A proxy hint: same handling as S-4 once the deal closes. "
        "Don't create instruments from the proxy itself — wait for "
        "the closing 8-K / S-4 effectiveness."
    ),
    "FWP": (
        "Free Writing Prospectus hint: marketing material for an "
        "offering. The pricing terms in an FWP supersede prior "
        "indications; treat the FWP as the canonical pricing input "
        "if no 424B has been seen yet, otherwise no-op."
    ),
    "POS AM": (
        "Post-effective amendment hint: usually adjusts an existing "
        "shelf or S-1. Emit `amend_instrument` against the prior "
        "registration's ledger id rather than creating a new one."
    ),
    "10-K": _PERIODIC_HINT,
    "10-Q": _PERIODIC_HINT,
    "20-F": _PERIODIC_HINT,
    "40-F": _PERIODIC_HINT,
    "6-K": (
        "Form 6-K hint: foreign-issuer current report. Treat like an "
        "8-K — look for offering announcements, exhibits with "
        "agreements/warrants, and pricing tables. Apply the same "
        "ANNOUNCEMENT vs CLOSING rule and DEDUP RULE — a 6-K "
        "announcing 'closing' of an offering already drawn-down on "
        "the ledger is a no-op."
    ),
    "EFFECT": (
        "EFFECT hint: SEC notice that a registration is effective. "
        "No body to extract from. The walker normally skips these; "
        "if you are seeing one, emit no mutations."
    ),
    "RW": (
        "RW hint: registration withdrawal. The walker normally skips "
        "these; if you are seeing one, emit no mutations."
    ),
}


def form_hint(form: str | None) -> str:
    """Pick the best-fitting hint for a form. Prefix-match so that "S-3/A"
    picks up the S-3 hint, "424B5" picks up 424B, etc."""
    if not form:
        return ("General hint: this is an SEC filing without a "
                "recognized form prefix. Read end-to-end for any "
                "dilution disclosure.")
    f = form.upper().strip()
    for prefix, hint in _FORM_HINTS.items():
        if f.startswith(prefix):
            return hint
    return ("General hint: unknown form prefix; read the document for "
            "warrant / convertible / preferred / ATM / equity-line / "
            "shelf disclosures and emit mutations as appropriate.")


# ─── Mutation vocabulary reference ──────────────────────────────────
# Concatenated into the system prompt below so the stable rulebook is
# cached across every filing call. (Moonshot's json_object mode also
# doesn't surface the response_format schema to the model, so the
# inline reference is doubly necessary.)
_MUTATION_REFERENCE = """\
Mutation kinds (emit any in any order; the walker sorts internally):

  create_instrument    — first time an instrument is disclosed
    fields:
      type       : "warrant" | "convertible" | "preferred" | "atm"
                   | "equity_line" | "shelf" | "s1_offering" | "equity"
      proposed_id: optional string like "W-001" — only set when you
                   want to link a later mutation in the same filing
                   to this create. Otherwise leave null.
      counterparty            : the INVESTOR / BUYER / LENDER /
                                HOLDER — the party putting capital
                                INTO the company (e.g. "Streeterville
                                Capital", "Hudson Bay Master Fund").
                                NOT the bank — see placement_agent.
                                Set to null when only generic
                                descriptors appear ("institutional
                                investors", "the Purchaser") — see
                                COUNTERPARTY NULLING RULE below.
      counterparty_canonical  : a short canonical form of the investor
                                (e.g. "Hudson Bay" for "Hudson Bay
                                Master Fund Ltd"). Same null rule.
      placement_agent         : the BANK running the offering —
                                underwriter / placement agent / sales
                                agent VERBATIM (e.g. "Maxim Group
                                LLC", "ThinkEquity LLC", "Joseph
                                Gunnar & Co. LLC"). DISTINCT from
                                counterparty — set BOTH when the
                                filing names both a bank and an
                                investor. Null when no bank is
                                involved (private placement,
                                convertible debenture issued direct).
      placement_agent_canonical: short canonical form of the bank
                                (e.g. "Maxim", "ThinkEquity",
                                "Joseph Gunnar", "Aegis").
      label                   : clean human-readable instrument label
                                used as the dashboard card headline.
                                See INSTRUMENT LABEL section below for
                                the format.
      terms      : type-specific dict. Common keys:
                     warrant       — strike, exercisable_date,
                                     expiration, anti_dilution_type,
                                     pp_clause_text, is_pre_funded,
                                     units
                     convertible   — principal, rate, conv_price,
                                     convertible_date, maturity,
                                     oid_pct, anti_dilution_type,
                                     pp_clause_text
                     preferred     — conv_price, convertible_date,
                                     maturity, stated_value,
                                     liquidation_preference,
                                     dividend_rate, series_letter
                     atm           — capacity_usd
                     equity_line   — capacity_usd
                     shelf         — capacity_usd, form (e.g. "S-3")
                     s1_offering   — anticipated_deal_size,
                                     warrant_strike,
                                     warrant_coverage_pct
      outstanding: type-specific dict. Common keys:
                     warrant       — count
                     convertible   — principal_remaining
                     preferred     — count, principal_remaining
                     atm           — remaining_capacity_usd, drawn_usd
                     equity_line   — remaining_capacity_usd, drawn_usd
                     shelf         — remaining_capacity_usd
                     s1_offering   — sold_to_date
      event_date : YYYY-MM-DD when the disclosure happened

  amend_instrument     — terms change on an existing ledger row
    fields:
      instrument_id  : the existing id from the ledger view
      field_updates  : sparse dict of terms keys → new values (set a
                       value to null to clear)
      event_date     : when the amendment happened

  record_event         — partial-state mutation
    fields:
      instrument_id  : the existing id
      event_kind     : "exercise" | "conversion" | "partial_redemption"
                       | "partial_termination" | "drawdown"
      fields         : kind-specific dict. Common keys:
                         exercise   — shares, price, gross_proceeds
                         conversion — principal_converted,
                                      principal_remaining, shares_issued
                         drawdown   — drawdown_amount_usd,
                                      drawdown_shares, avg_price,
                                      placement_agent (banker /
                                        underwriter on this takedown,
                                        verbatim from filing — e.g.
                                        "Jefferies LLC"),
                                      placement_agent_canonical
                                        (short form — e.g. "Jefferies")
      event_date     : when it happened

  close_instrument     — instrument is fully consumed
    fields:
      instrument_id  : the existing id
      reason         : "exercised" | "converted" | "redeemed"
                       | "expired" | "terminated" | "superseded"
      replaced_by    : when reason="superseded", the id of the
                       replacement
      event_date     : when it happened

  apply_split          — global mutation across all warrants/converts/preferreds
    fields:
      ratio          : post / pre. 1-for-10 reverse = 0.1; 2-for-1
                       forward = 2.0
      direction      : "reverse" | "forward"
      units          : "common" | "ads" (default "common")
      effective_date : YYYY-MM-DD when the split took effect

Rules:
  - Extract numbers VERBATIM from the filing — never sum/multiply/
    average. Code does that downstream.
  - If the filing re-discloses an instrument that already exists in
    the ledger view, emit no mutation OR an `amend_instrument` if
    terms changed. NEVER create a duplicate.
  - DEDUP CHECK before any `create_instrument` for warrant /
    convertible / preferred: scan the open ledger for an active
    row of the same type whose strike (or conv_price) is within
    ±2% of this filing's value AND whose `created` is within ±30
    days of this filing's date (or of an "issued / closed / dated
    as of" date in the filing text). If one matches, emit no
    `create_instrument` — the instrument already exists. Counter-
    party labels are NOT a matching signal; see CORE PRINCIPLE.
  - Resale 424Bs / S-1 resale prospectuses do NOT create new ledger
    rows — the underlying instruments were already issued.
  - Pre-funded warrants ($0.0001 strike) ARE warrants — emit a
    create_instrument for them with is_pre_funded=true in terms.
  - The `equity` type is RESERVED for UNREGISTERED private
    placements of common shares — Reg D, Reg S, Section 4(a)(2) —
    where there is NO shelf, ATM, equity-line, or S-1 vehicle
    carrying the issuance. A registered direct offering taken down
    under a shelf is NOT an equity instrument; it is a drawdown
    against the shelf. An ATM sale is not equity; it is a drawdown
    against the ATM. An IPO / follow-on registered on S-1 is not
    equity; it is the s1_offering itself plus drawdowns. Common
    shares issued in any of those structures should NEVER produce
    a `create_instrument(equity)` — only `record_event(drawdown)`
    on the carrying instrument.

ANTI-DILUTION CLASSIFICATION — when emitting `create_instrument` or
`amend_instrument` for warrant / convertible / preferred, set
`terms.anti_dilution_type` by applying the rules below in order and
taking the FIRST match:

  1. Filing describes ANY reset / VWAP-linked formula / lookback /
     alternate-conversion-on-default / "lower of $X and Y% of VWAP" /
     floor price below initial strike / repayable-in-shares-at-recent-
     price language
     → "variable_rate". Set pp_clause_text to a verbatim ≤2-sentence
       excerpt of the trigger clause.

  2. Filing describes a full reset of strike / conversion price to
     the price of any subsequent dilutive issuance
     → "full_ratchet". Set pp_clause_text verbatim.

  3. Filing describes a net-share-settle / cashless formula NOT tied
     to Black-Scholes (e.g. "cashless after 60 days for 0.5 shares
     per warrant")
     → "Alternate Cashless". Set pp_clause_text verbatim.

  4. Filing describes ordinary adjustments for stock splits, stock
     dividends, recapitalizations, or similar — with no reset / VWAP
     / floor mechanism
     → "Customary Anti-Dilution". pp_clause_text = null.

  5. Filing is SILENT on anti-dilution provisions for this instrument
     → "undisclosed". pp_clause_text = null.

"Customary Anti-Dilution" requires the filing to actually describe
the standard adjustments. It is NOT a default for ambiguity.
"undisclosed" is reserved for genuine silence — do NOT use it when
the filing describes adjustments you could classify under rules 1-4.

DATE FIELDS — emit ABSOLUTE YYYY-MM-DD dates in `terms`, not relative
durations. The dashboard renders the dates verbatim and never
back-computes them from a term length.

  warrant:
    * exercisable_date — when the warrant FIRST becomes exercisable.
      Usually the issuance / closing date, but can be later when the
      filing imposes a vesting period (e.g. "exercisable beginning
      six months after issuance" → issuance_date + 6 months) or
      shareholder-approval gate (e.g. "exercisable upon receipt of
      stockholder approval").
    * expiration — when the warrant expires. Compute from "5-year
      warrants issued September 15, 2025" → "2030-09-15". Never
      emit `term_years` on its own; always resolve to an absolute
      expiration date.

  convertible:
    * convertible_date — when the note FIRST becomes convertible.
      Usually the issuance date for OID convertibles; can be later
      for notes that are convertible only at maturity or after a
      specified seasoning period.
    * maturity — when the note matures (mandatory repayment date).

  preferred:
    * convertible_date — when the preferred FIRST becomes
      convertible. Usually the issuance date.
    * maturity — mandatory redemption date when one exists; null
      for perpetual preferred.

  atm / equity_line / shelf:
    * No additional date keys needed. The card uses `created_at`
      (the filing/event_date) for agreement_start_date and `status_at`
      for agreement_end_date.

When the filing states a relative term ("5-year warrants",
"2-year note"), compute the absolute date from the issuance date
the filing also states. When only a relative term is given with no
issuance date, use `event_date` (this filing's date) as the
issuance reference.

COUNTERPARTY vs PLACEMENT AGENT — every typical financing has up to
two parties the cap-table cares about. Track them in SEPARATE fields:

  * counterparty            — the INVESTOR / BUYER / LENDER / HOLDER
                              (capital flows IN from them). Examples:
                              "Streeterville Capital" (note lender),
                              "Hudson Bay Master Fund" (PIPE buyer),
                              "Anson Funds" (warrant holder),
                              "3AM Investments" (preferred buyer),
                              "Empery Asset Management".
  * placement_agent         — the BANK running the offering — the
                              underwriter / placement agent / sales
                              agent. Examples: "Maxim Group LLC"
                              (canonical "Maxim"), "ThinkEquity LLC"
                              (canonical "ThinkEquity"), "Joseph
                              Gunnar & Co. LLC", "Aegis Capital",
                              "Roth Capital Partners", "H.C.
                              Wainwright", "Jefferies", "B. Riley".

Both can be set on the same `create_instrument`. Common patterns:
  - Maxim-underwritten registered direct to Hudson Bay → both set.
  - Streeterville convertible note (no bank) → counterparty only.
  - Underwritten public offering with no named investor →
    placement_agent only, counterparty null.
  - 8-K describing "issued to certain institutional investors" with no
    bank named → BOTH null (counterparty by NULLING RULE; no bank in
    the filing).

NEVER put a bank into counterparty (Maxim is not an investor). NEVER
put an investor into placement_agent (Streeterville is not a bank).

COUNTERPARTY NULLING RULE — when the filing does NOT name a specific
investor, set BOTH `counterparty` and `counterparty_canonical` to
null. Generic descriptors are NOT investors — never put them in these
fields:

  * "institutional investor" / "institutional investors" / "certain
    institutional investors"
  * "the Investor" / "the Purchaser" / "the Purchasers"
  * "Holders" / "Holders of Existing Warrants" / "various investors"
  * "accredited investors"
  * the instrument category as a phrase: "convertible preferred",
    "outstanding warrants", "remaining outstanding", "preferred stock"

Only populate counterparty when the filing names a specific entity
(Streeterville, Hudson Bay, Anson, Empery, Sabby, etc.). The same
rule applies to `placement_agent` — only set it when a bank is named
by name; null otherwise.

INSTRUMENT LABEL — set `label` on every `create_instrument` for
warrant / convertible / preferred / atm / equity_line / shelf /
s1_offering. The label is what appears as the card headline on the
dashboard.

STRICT FORMAT: "<Month> <Year> <Bank-or-Series-or-Descriptor> <Type>"

  * The label MUST start with the ISSUANCE Month and Year — the date
    the instrument was first issued / agreement was signed / shelf
    became effective. NOT the expiration year, NOT the maturity year,
    NOT the period-of-report date. A 5-year warrant issued in
    September 2021 is "September 2021 ...", NOT "September 2026 ..."
    (its 2026 expiration is irrelevant to the label).
  * Word order is fixed: Month-Year FIRST, then optional bank or
    series letter, then the type. Do NOT put the bank name first —
    "Streeterville Capital December 2022 Promissory Note" is WRONG;
    correct is "December 2022 Streeterville Promissory Note".
  * Do NOT include generic descriptors like "outstanding", "remaining",
    "convertible preferred", "preferred stock", "warrants" twice. The
    type word at the end is the only category descriptor needed.
  * Do NOT include strike, principal, capacity, or any dollar amount
    in the label — those render in their own card fields.

Examples that match how DilutionTracker labels these (mirror this
format exactly):

  * warrant            : "September 2025 Common Warrants"
                         "December 2023 Inducement Warrants"
                         "October 2022 Pre-funded Warrants"
                         "August 2019 Underwriter Warrants"
                         "November 2020 Maxim Warrants"
  * convertible        : "December 2022 Streeterville Promissory Note"
                         "May 2024 Origin Group Convertible Note"
                         "April 2022 Inpixon OID Debenture"
  * preferred          : "March 2024 Series 9 Preferred"
                         "January 2019 Series 5 Convertible Preferred"
                         "November 2025 Series 10 Convertible Preferred"
  * atm                : "July 2022 Maxim ATM"
                         "May 2024 ThinkEquity ATM"
  * equity_line        : "March 2023 Lincoln Park ELOC"
  * shelf              : "August 2025 Shelf"
                         "June 2021 Shelf"
  * s1_offering        : "June 2025 S-1 Offering"
                         "December 2023 S-1 Offering"

Leave `label` null only when no useful descriptor exists; the card
will fall back to a deterministic template.
"""


# ─── System + user prompt assembly ──────────────────────────────────
# The system prompt holds the entire stable rulebook so it stays
# identical across filings and benefits from prompt caching. The user
# message carries only per-filing variables (ledger, metadata, form
# hint, ADS preamble, filing text).
_SYSTEM_PROMPT_CORE = """\
You are an analyst maintaining a capitalization table for a
publicly-traded issuer. Your job is to read one SEC filing and emit
mutations that update the cap-table state accordingly.

Output format: a JSON object {"mutations": [...]} matching the
MutationList schema. Return {"mutations": []} when the filing
contains no cap-table-relevant disclosure.

CORE PRINCIPLE — be CONSERVATIVE about duplicate disclosures.
The same offering / amendment / exercise is typically disclosed
across multiple filings:

  - announcement 8-K → pricing 424B → closing 8-K → next 10-Q
    confirms outstanding count → 20-F restates at year-end

If the open ledger already contains an instrument that matches
what THIS filing describes, emit NO mutation (or at most one
`amend_instrument` when a term genuinely changed). The earlier
filing already captured the instrument; this filing is
re-disclosure.

MATCHING KEYS for warrant / convertible / preferred re-disclosure,
in order of strength:
  1. strike (warrant) or conversion price (convertible / preferred)
     agrees within ±2%.
  2. the ledger row's `created` column (= original disclosure date)
     is within ±30 days of THIS filing's date OR of any "issued on
     / issuance date / closed on / dated as of" date stated in the
     filing text.

If both keys agree on an active ledger row of the same type, it
IS the same instrument — emit no `create_instrument`. If multiple
ledger rows match, pick the earliest-created and amend it; do NOT
add another row.

Counterparty labels are NOT a matching signal. Filings describe
the same purchaser many ways — "institutional investor", "certain
institutional investors", "the Investor", "Holders of Existing
Warrants", "January Purchaser", "Hudson Bay", "Hudson Bay Master
Fund Ltd", "Series 5 Holders". Different labels across filings
are NOT evidence of different tranches.

Instrument-NAME labels are also NOT a matching signal. The same
tranche is described in different filings as "Inducement
Warrants", "New Warrants", "Series A Warrants", "the January
2027 Warrants", "the 3-Year Warrants" — depending on whether the
filing emphasizes purpose, recency, series letter, expiry, or
term. None of those re-namings create a new instrument. Match on
strike + date.

WORKED EXAMPLE — same tranche, different labels.

  Filing 1 (8-K, 2024-01-02) announces an inducement: holders
  exercise old warrants and receive "New Inducement Warrants"
  at $0.65 strike, 3-year term, dated 2023-12-29.
  → Emit create_instrument(warrant), strike=0.65, term_years=3,
    event_date=2023-12-29.
  → Ledger now has W-007 strike=0.65 created=2023-12-29.

  Filing 2 (424B3, 2024-01-02) is the resale prospectus for the
  same offering. It calls them "the New Warrants" issued in the
  "December 2023 Inducement" — $0.65 strike, 3-year term.

    DEDUP CHECK before any create_instrument:
      strike ±2%?  0.65 == 0.65. ✓
      created ±30d? 2023-12-29 is within 30d of 2024-01-02. ✓
    → It IS W-007. Emit NO create_instrument. If the 424B has a
      more precise maturity than W-007 carries, emit ONE
      amend_instrument against W-007 to refine that field.

  Filing 3 (6-K, 2024-01-02) describes the same warrants again,
  this time by expiry — "Warrants expiring January 2027".
  → Same dedup keys still match W-007. NO mutation.

  Filing 4 (6-K, 2024-01-02) is the closing 6-K, describing them
  as "the 3-Year Warrants".
  → Same tranche. NO mutation.

The wrong outcome is four filings producing W-007, W-008, W-009,
W-010 because the model latched onto the LABEL each filing used
("Inducement", "New", "Jan 2027", "3-Year"). They are ONE
tranche.

Outstanding-count drift is NOT evidence of a new tranche. An 8-K
announcing "approximately 550,000 warrants" and a 424B pricing
"626,667 warrants" are the SAME tranche — overallotment was
exercised between announcement and closing. Match on strike +
date, not on count.

For shelf / ATM / equity-line drawdowns: re-disclosure dedup keys
are (instrument_id, date ±10d, gross_proceeds ±5%) — see the
per-form hints below. Each shelf/ATM/equity-line row in "Today's
ledger" is followed by a `takedowns:` line listing recorded prior
drawdowns (date, amount, shares, banker). Compare the filing's
takedown against that list before emitting a new drawdown. A
subscription-agreement-signed 6-K and the later "closing of
previously announced" 6-K describe the SAME takedown — emit it
once (on the signing date is fine), not twice. Same for an
earnings/20-F/10-K recap of an already-recorded takedown: no-op.

The walker has anchor-reconciliation against periodic filings to
catch instruments you missed; it has NO mechanism to merge
duplicates you created by mistake. When in doubt, emit nothing.

Numeric values must be NUMBERS, not strings. Write 8850000, not
"8.85 million" or "8,850,000". Write 0.85, not "$0.85".
"""


SYSTEM_PROMPT = _SYSTEM_PROMPT_CORE + "\n\n" + _MUTATION_REFERENCE


def build_user_prompt(
    *,
    unit_preamble: str,
    ledger_view: str,
    form: str,
    filing_date: str,
    accession: str,
    items: str | None,
    period_of_report: str | None,
    filing_text: str,
) -> str:
    """Assemble the user message. Filing text is the last block so the
    LLM doesn't lose ledger context if the body is long."""
    parts = [
        unit_preamble.rstrip() + "\n",
        "## Today's ledger\n",
        ledger_view.rstrip() + "\n",
        "## Filing metadata\n",
        f"- form: {form}\n"
        f"- filing_date: {filing_date}\n"
        f"- accession: {accession}\n",
    ]
    if period_of_report:
        parts.append(f"- period_of_report: {period_of_report}\n")
    if items:
        parts.append(f"- items: {items}\n")
    parts.append("\n## Form hint\n")
    parts.append(form_hint(form).rstrip() + "\n")
    parts.append("\n## Filing text\n")
    parts.append(filing_text)
    parts.append("\n\nEmit a MutationList for this filing.")
    return "".join(parts)


__all__ = [
    "SYSTEM_PROMPT",
    "build_user_prompt",
    "form_hint",
]
