"""create_* tool definitions.

Initial vertical: create_atm + create_shelf. The remaining 6 creates
(warrant, convertible, preferred, equity_line, s1_offering, equity)
get added in the fan-out phase.

Tool descriptions absorb the relevant guidance from walker_prompt.py
so the LLM sees the rule at the moment it's deciding which tool to
call. The system prompt no longer needs to carry the per-tool prose.
"""

from __future__ import annotations

from ._base import Tool, ToolArg, ISO_DATE_PATTERN, PROPOSED_ID_PATTERN


# ─── create_atm ───────────────────────────────────────────────────────

_CREATE_ATM_DESCRIPTION = (
    "Call this when the filing establishes a NEW at-the-market sales "
    "agreement. Recognize by: an 'Equity Distribution Agreement', "
    "'At Market Issuance Sales Agreement', 'ATM Sales Agreement', or "
    "'Sales Agreement' that names one or more Sales Agents, typically "
    "attached as EX-1.x on an S-3 or referenced in 8-K Item 1.01.\n\n"
    "DO NOT call this for:\n"
    "  - a takedown / drawdown FROM an existing ATM already in the "
    "ledger view (use record_drawdown instead)\n"
    "  - a baby-shelf re-registration that down-sizes an existing ATM's "
    "capacity at the I.B.5 ceiling when the same agreement_date and "
    "Sales Agent already exist in the ledger (use amend_atm instead)\n"
    "  - a Form S-3 that also registers a base shelf — call create_shelf "
    "SEPARATELY for the base prospectus headline cap. This tool's "
    "capacity_usd is the smaller carve-out cap stated in the Sales "
    "Agreement prospectus, NOT the base shelf cap.\n\n"
    "AGREEMENT vs RE-REGISTRATION: the agreement_date is the execution "
    "date stated on the Sales Agreement itself, not the filing date of "
    "the S-3 that registers it. Two agreements with the same Sales "
    "Agent but different execution dates are DISTINCT instruments, not "
    "amendments — separate create_atm calls."
)

create_atm = Tool(
    name="create_atm",
    description=_CREATE_ATM_DESCRIPTION,
    mutation_kind="create_instrument",
    instrument_type="atm",
    event_kind=None,
    valid_forms=frozenset({
        "S-3", "S-3ASR", "S-3/A", "F-3", "F-3ASR", "F-3/A",
        "S-3MEF", "F-3MEF",
        "F-10", "F-10/A", "F-10EF",
        "8-K", "8-K/A",
        "6-K", "20-F", "40-F",
        "424B5", "SUPPL",
    }),
    args=(
        ToolArg(
            name="capacity_usd",
            type="number",
            required=True,
            min_value=1.0,
            description=(
                "Dollar cap stated IN THE SALES AGREEMENT — e.g. "
                "'aggregate offering price of up to $15,300,000'. This "
                "is the ATM's own carve-out cap. If the parent S-3 "
                "registers a larger base prospectus number (e.g. "
                "$200,000,000), do NOT use that here; call create_shelf "
                "separately."
            ),
        ),
        ToolArg(
            name="agreement_date",
            type="date",
            required=True,
            description=(
                "Execution date of the Sales Agreement itself (ISO "
                "YYYY-MM-DD). Primary identity key for the ATM program. "
                "Look for the EX-1.x cover or 8-K Item 1.01 sentence "
                "'On [DATE], we entered into an Equity Distribution "
                "Agreement…'.\n\n"
                "OVERRIDE — supersession via fresh S-3 / F-3. When THIS "
                "create is paired with a close_instrument(reason="
                "'superseded') against an existing ATM, use the NEW "
                "filing date here, NOT the verbatim '[old date]' the "
                "supplement restates. The new card must render under the "
                "current month/banker. ONLY use this override when the "
                "new cover shows a banker rebrand OR a capacity change "
                "vs the existing card (see the system prompt's 'fresh "
                "S-3 / F-3 supersedes an embedded ATM' worked example) — "
                "a fresh shelf with the same agent and same capacity is "
                "re-registration, NOT supersession, and the agreement_date "
                "stays at the verbatim signing date with no paired close."
            ),
        ),
        ToolArg(
            name="placement_agent_canonical",
            type="string",
            required=True,
            min_length=2,
            description=(
                "Short canonical name of the Sales Agent (e.g. "
                "'Canaccord', 'Wainwright', 'Maxim', 'ThinkEquity'). "
                "Use current entities for rebrands ('Leerink Partners' "
                "not 'SVB Securities', 'TD Cowen' not 'Cowen'). For "
                "multi-agent EDAs name the lead. NOT the issuer's "
                "counterparty — ATMs don't have a counterparty."
            ),
        ),
        ToolArg(
            name="event_date",
            type="date",
            required=True,
            description=(
                "Filing date for this disclosure (YYYY-MM-DD), used as "
                "the ledger history timestamp. Usually equals "
                "agreement_date for first-time registration."
            ),
        ),        ToolArg(
            name="remaining_capacity_usd",
            type="number",
            required=False,
            description=(
                "Capacity not yet drawn. Omit on first registration "
                "(implicitly equals capacity_usd). Set only when the "
                "filing explicitly states a residual cap, typically on "
                "anchor reconciliations from a 10-Q footnote like "
                "'$X remained available under the agreement as of …'."
            ),
        ),
        ToolArg(
            name="drawn_usd",
            type="number",
            required=False,
            description=(
                "Cumulative dollars already drawn under THIS agreement "
                "at event_date. Omit if zero or unstated."
            ),
        ),
        ToolArg(
            name="agreement_end_date",
            type="date",
            required=False,
            description=(
                "Contractual term-end of the Sales Agreement (ISO "
                "YYYY-MM-DD) when the agreement specifies a fixed "
                "expiry — e.g. 'the Equity Distribution Agreement will "
                "terminate on the earliest of … December 31, 2024'. "
                "Omit when the agreement runs until cap is sold or is "
                "open-ended."
            ),
        ),
        ToolArg(
            name="proposed_id",
            type="string",
            required=False,
            pattern=PROPOSED_ID_PATTERN,
            description=(
                "Set ONLY when this create is paired with a "
                "close_instrument(reason='superseded') against a "
                "predecessor ATM — proposed_id on the create and "
                "replaced_by on the close must match so the store can "
                "resolve them to the same actual id at apply time. "
                "Format: lowercase-kebab, e.g. 'atm-fresh-tranche'."
            ),
        ),
    ),
)


# ─── create_shelf ─────────────────────────────────────────────────────

_CREATE_SHELF_DESCRIPTION = (
    "Call this when an S-3 / F-3 / S-3ASR / F-3ASR (or Canadian MJDS "
    "F-10 / F-10EF) registration statement first declares a primary "
    "shelf. The capacity_usd is the headline aggregate offering amount "
    "on the prospectus cover page (e.g. 'aggregate offering price not "
    "exceeding $200,000,000').\n\n"
    "WKSI / UNLIMITED SHELF — an S-3ASR / F-3ASR (or any base shelf) "
    "filed by a well-known seasoned issuer registers an INDETERMINATE "
    "amount of securities under Rule 457(r): the cover states no dollar "
    "cap (language like 'an indeterminate amount' / 'an unspecified "
    "number of shares', fees deferred to each takedown). This IS a "
    "primary shelf — DO create it. Since no headline figure exists, "
    "pass capacity_usd=999999999 (the unlimited-shelf sentinel). Do NOT "
    "skip it and do NOT invent a number from the fee table.\n\n"
    "DO NOT call this for:\n"
    "  - a resale S-3 / F-3 registering shares for selling stockholders "
    "(cover says 'resale from time-to-time…by the Selling Stockholders' "
    "and 'we will not receive any proceeds'). That's a secondary "
    "registration of shares already issued (or issuable under existing "
    "warrants) — it does not give the issuer new raisable capacity. The "
    "dollar figure on the fee-table is a Rule 457 max-aggregate-offering "
    "price for fee calculation, not a primary headline. Skip the call.\n"
    "  - a 424B prospectus takedown from an existing shelf (use "
    "record_drawdown instead, against the parent shelf's instrument_id)\n"
    "  - an S-3/A that amends an existing shelf's capacity (use "
    "amend_shelf instead)\n"
    "  - an S-3MEF that bolts additional capacity onto an existing "
    "shelf (use amend_shelf — MEF is mechanically an extension, not a "
    "new shelf)\n"
    "  - a 10-K / 10-Q footnote describing an existing shelf — that's "
    "re-disclosure, not a new registration\n\n"
    "If the same S-3 also registers an embedded at-the-market sales "
    "agreement (an 'Equity Distribution Agreement' prospectus stapled "
    "inside the base S-3), ALSO call create_atm separately for the "
    "carve-out. This tool's capacity_usd is the larger base headline; "
    "create_atm's capacity_usd is the smaller carve-out."
)

create_shelf = Tool(
    name="create_shelf",
    description=_CREATE_SHELF_DESCRIPTION,
    mutation_kind="create_instrument",
    instrument_type="shelf",
    event_kind=None,
    valid_forms=frozenset({
        "S-3", "S-3ASR", "F-3", "F-3ASR",
        "F-10", "F-10/A", "F-10EF",
    }),
    args=(
        ToolArg(
            name="capacity_usd",
            type="number",
            required=True,
            min_value=1.0,
            description=(
                "Aggregate offering amount stated on the prospectus "
                "cover page, e.g. 'aggregate offering price not "
                "exceeding $200,000,000' → 200000000. This is the "
                "base headline cap registered for sale across all "
                "instrument types (common, preferred, warrants, units, "
                "debt). Use the largest dollar figure on the cover, "
                "not the smaller per-tranche / per-prospectus amount. "
                "For a WKSI S-3ASR / F-3ASR that registers an "
                "indeterminate amount under Rule 457(r) (no dollar cap "
                "stated), pass 999999999 — the unlimited-shelf sentinel."
            ),
        ),
        ToolArg(
            name="form",
            type="string",
            required=True,
            enum_values=("S-3", "S-3ASR", "F-3", "F-3ASR",
                         "F-10", "F-10EF"),
            description=(
                "Registration form symbol. Must match the filing's "
                "actual form."
            ),
        ),
        ToolArg(
            name="event_date",
            type="date",
            required=True,
            description=(
                "Filing date of this S-3 / F-3 / F-10 (YYYY-MM-DD)."
            ),
        ),        ToolArg(
            name="remaining_capacity_usd",
            type="number",
            required=False,
            description=(
                "Capacity not yet drawn. Omit on first registration "
                "(implicitly equals capacity_usd). Set only on anchor "
                "reconciliations from a 10-Q footnote."
            ),
        ),
        ToolArg(
            name="proposed_id",
            type="string",
            required=False,
            pattern=PROPOSED_ID_PATTERN,
            description=(
                "Set ONLY when this create is paired with a "
                "close_instrument(reason='superseded') against a "
                "predecessor shelf. Format: lowercase-kebab."
            ),
        ),
    ),
)


# ─── create_warrant ───────────────────────────────────────────────────

_DESCRIPTOR_ENUM = (
    "Pre-Funded",        # warrants priced near zero strike
    "Inducement",        # exchange/inducement warrants in a re-pricing
    "Common",            # warrants attached to a common-stock offering
    "Underwriter",       # warrants issued to the underwriter as comp
    "Placement Agent",   # warrants issued to the placement agent
    "Rep",               # representative warrants (post-IPO)
    "Purchase",          # purchase warrants (broker-attached)
    "Private Placement", # equity issued in an unregistered placement
    "ELOC",              # equity-line-of-credit / equity_line label
)

_UNITS_ENUM = ("common", "ads")


_COMMON_PARTY_ARGS = (
    ToolArg(
        name="placement_agent_canonical",
        type="string",
        required=False,
        description=(
            "The BANK running the offering — underwriter / placement "
            "agent / sales agent. Short canonical form ('Maxim', "
            "'ThinkEquity', 'H.C. Wainwright'). Null when no bank is "
            "involved (private placement, direct convertible, internal "
            "issuance). Investors / buyers / lenders go in "
            "known_owners, not here. NEGATIVE-COVENANT TRAP: a broker "
            "named only as the EXCEPTION inside a SPA's 'No Brokers' / "
            "'No Solicitation' representation ('Except with respect to "
            "J.H. Darbie & Co., a registered broker-dealer…') is NOT "
            "the offering's placement agent — leave null; a Section "
            "4(a)(2) private placement has no underwriter."
        ),
    ),
    ToolArg(
        name="descriptor",
        type="string",
        required=False,
        enum_values=_DESCRIPTOR_ENUM,
        description=(
            "Short qualifier the walker uses to assemble the card "
            "label. Pick the best match for the filing's wording. "
            "Leave unset when the instrument type alone describes the "
            "instrument."
        ),
    ),
    ToolArg(
        name="known_owners",
        type="array",
        required=False,
        items_type="string",
        description=(
            "Named investor(s) / buyer(s) / lender(s) putting capital "
            "into the issuer, in the EXACT short form the filing uses. "
            "ALWAYS populate when ANY named purchaser is identified — "
            "both single-investor deals (Reg D notes from one fund, "
            "EIB preferred, Board-approved warrant) and multi-investor "
            "PIPEs (Purchaser / SPA table listing 2+ funds). Examples:\n"
            "  - 4-investor PIPE: ['Armistice', 'Sabby', 'Bigger "
            "Capital', 'District 2']\n"
            "  - Single PIPE warrant: ['Armistice']\n"
            "  - Streeterville note: ['Streeterville']\n"
            "  - EIB Finance Contract preferred: ['EIB']\n"
            "  - Yorkville SEPA single-investor: ['YA II'] (or "
            "['Yorkville'] if that's the form the SPA uses)\n"
            "Use the SHORT form from the filing's prose, NOT the legal "
            "entity name ('YA II', not 'YA II PN, Ltd.'; 'EIB', not "
            "'European Investment Bank'; 'Armistice', not 'Armistice "
            "Capital Master Fund Ltd.'). Leave null ONLY when the "
            "filing uses purely generic descriptors ('an institutional "
            "investor', 'the Purchaser', 'certain investors') without "
            "ever naming a specific entity. Underwriters and placement "
            "agents do NOT go here — they go in "
            "placement_agent_canonical. The card's investor headline "
            "(e.g. 'December 2022 Streeterville Note') and investor "
            "quality tag are both derived from known_owners[0] when "
            "the array has exactly one entry."
        ),
    ),
)


_PROPOSED_ID_ARG = ToolArg(
    name="proposed_id",
    type="string",
    required=False,
    pattern=PROPOSED_ID_PATTERN,
    description=(
        "Set ONLY when this create is paired with a "
        "close_instrument(reason='superseded') against a predecessor "
        "of the same instrument type. Format: lowercase-kebab."
    ),
)


create_warrant = Tool(
    name="create_warrant",
    description=(
        "Call when a filing newly discloses a warrant tranche. Set "
        "`count` to the number of warrants issued, `strike` to the "
        "per-share exercise price. `is_pre_funded=true` when the "
        "strike is near zero (pre-funded warrants are economically "
        "common stock). For warrants attached to a common-stock "
        "offering (an underwritten S-1 or 8-K PIPE), set descriptor "
        "to 'Common'; for inducement warrants emitted in a "
        "warrant-repricing exchange, set 'Inducement'.\n\n"
        "NOT A WARRANT — do NOT call create_warrant for:\n"
        "  • EMPLOYEE/OFFICER STOCK OPTIONS. Options granted to "
        "officers/directors/employees under an equity-incentive / ESOP "
        "plan (e.g. 'options to purchase ... under the 2018 Stock "
        "Incentive Plan', granted by the Compensation Committee) are "
        "compensation, not dilutive financing — even when the same "
        "filing also discusses real warrants/ATMs. This INCLUDES option "
        "grants made under an executive/director EMPLOYMENT or new-hire "
        "agreement (e.g. time-based + milestone/performance options to "
        "purchase shares/ADSs granted to an incoming officer/director AS "
        "COMPENSATION under that agreement), even when the grant carries "
        "a per-share exercise price, an exercisable date and an "
        "expiration date and superficially resembles a warrant, and even "
        "when it is 'subject to approval of the Board'. Such compensation "
        "options are NOT a warrant tranche. This does NOT apply to "
        "PIPE / inducement / underwritten FINANCING warrants held by "
        "named individuals — those remain real create_warrant events. "
        "Call note_no_event "
        "for the option grant.\n"
        "  • CANCELLED/RETURNED warrants in an exchange. When holders "
        "RETURN warrants for cancellation in exchange for common stock "
        "or a different security ('issued X shares ... in exchange for "
        "the return and cancellation of the [...] warrants'), those "
        "warrants NO LONGER EXIST — close the existing warrant "
        "(close_instrument) and record the shares via create_equity; "
        "do NOT mint a new warrant for the cancelled tranche. This holds "
        "even when a later registration statement (S-1/A, S-3/A) "
        "re-narrates the past exchange — it is describing securities "
        "already on the ledger, not issuing new ones.\n\n"
        "LABEL agent-comp warrants correctly: placement-agent / "
        "underwriter / representative compensation warrants must carry "
        "that descriptor (e.g. descriptor='Placement Agent'), NOT an "
        "invented 'Series A/B/C'. The system folds agent-comp warrants "
        "into the offering card; an invented Series label defeats that "
        "and surfaces them as spurious extra cards. Only use Series "
        "A/B/C when the filing itself designates the warrants that way."
    ),
    mutation_kind="create_instrument",
    instrument_type="warrant",
    event_kind=None,
    valid_forms=frozenset({
        "8-K", "8-K/A", "S-1", "S-1/A", "S-3", "S-3ASR", "S-3/A",
        "F-1", "F-3", "F-3ASR", "F-10", "F-10/A", "F-10EF",
        "424B5", "424B3", "424B4", "SUPPL",
        "10-Q", "10-K",  # exhibit-attached warrants disclosed in periodic
        "6-K", "20-F", "40-F",  # FPI event + annual reports
    }),
    args=(
        ToolArg(
            name="count",
            type="number",
            required=True,
            min_value=1.0,
            description="Number of warrants in this tranche.",
        ),
        ToolArg(
            name="strike",
            type="number",
            required=True,
            min_value=0.0,
            description=(
                "Per-share exercise price in USD. Set to 0 (or near-"
                "zero) for pre-funded warrants and set is_pre_funded=true."
            ),
        ),
        ToolArg(
            name="event_date",
            type="date",
            required=True,
            description="Filing or issuance date (YYYY-MM-DD).",
        ),        ToolArg(
            name="exercisable_date",
            type="date",
            required=False,
            description=(
                "Absolute first-exercisable date (YYYY-MM-DD) — set ONLY "
                "when the filing prints a literal calendar date. For the "
                "dominant RELATIVE phrasings, do NOT compute a date: set "
                "exercise_offset_months instead and the store derives "
                "this from event_date. An absolute date here overrides "
                "the computed value."
            ),
        ),
        ToolArg(
            name="expiration",
            type="date",
            required=False,
            description=(
                "Absolute expiration date (YYYY-MM-DD) — set ONLY when "
                "the filing prints a literal calendar date. For "
                "TERM-based language ('five-year term', 'fifth "
                "anniversary of issuance'), do NOT compute a date: set "
                "term_months (and term_anchor) and the store derives "
                "this. An absolute date here overrides the computed "
                "value."
            ),
        ),
        ToolArg(
            name="term_months",
            type="integer",
            required=False,
            min_value=1,
            description=(
                "Warrant life in MONTHS, extracted from the term "
                "language — the store adds it to the anchor date so you "
                "never do calendar math. Convert years to months: "
                "'five-year term' → 60, 'three-year' → 36, 'seven-year' "
                "→ 84, 'ten-year' → 120, 'thirty-month term' → 30, "
                "'three-and-a-half year' → 42. Set whenever the filing "
                "states a term; leave null only for genuinely perpetual "
                "warrants (rare)."
            ),
        ),
        ToolArg(
            name="exercise_offset_months",
            type="integer",
            required=False,
            min_value=0,
            description=(
                "Months from issuance until the warrant first becomes "
                "exercisable — the store adds it to event_date. Read "
                "the integer N straight out of the filing for any "
                "'N months from / after issuance' phrasing: 0 for "
                "'exercisable immediately upon issuance' / 'on the "
                "Closing Date' / 'on or after the date of issuance'; "
                "6 for 'six months after issuance'; 12 for 'twelve "
                "months from its issuance' / 'on the first anniversary "
                "of the issue date'; 24 for 'two years after issuance'. "
                "These are examples, not an enum — set whatever N the "
                "filing prints. Leave null ONLY when exercisability is "
                "gated on an undated future event ('upon stockholder "
                "approval'). Set this even when expiration is given as "
                "an absolute date — the two are independent."
            ),
        ),
        ToolArg(
            name="term_anchor",
            type="string",
            required=False,
            enum_values=("issuance", "exercise"),
            description=(
                "What term_months is measured FROM. 'issuance' "
                "(default) → expiration = event_date + term_months, for "
                "'five years from issuance' / 'fifth anniversary of the "
                "date of issuance'. 'exercise' → expiration = the "
                "initial exercise date + term_months, for 'expire on the "
                "fifth anniversary of the Initial Exercise Date' (pair "
                "with exercise_offset_months). Omit when measured from "
                "issuance."
            ),
        ),
        ToolArg(
            name="is_pre_funded",
            type="boolean",
            required=False,
            description=(
                "True when the warrant's strike is at or near zero "
                "(pre-funded warrants are economically common stock)."
            ),
        ),
        ToolArg(
            name="units",
            type="string",
            required=False,
            enum_values=_UNITS_ENUM,
            description=(
                "Denomination of the warrant: 'common' (default for US "
                "listings) or 'ads' (FPI issuers with American "
                "Depositary Shares). Set to 'ads' only when the filing "
                "explicitly expresses the count in ADS units."
            ),
        ),
        ToolArg(
            name="series_letter",
            type="string",
            required=False,
            min_length=1,
            description=(
                "Series identifier when a single offering issues MULTIPLE "
                "warrant tranches distinguished only by a series tag — "
                "e.g. 'Series A Warrants' (exercise price $0.48) and "
                "'Series B Warrants' (exercise price $0.55) issued on "
                "the same date in the same SPA, or 'Series 1' / "
                "'Series 2' / 'Pre-Funded'. Copy the letter/number "
                "form used in the filing ('A', 'B', '1', '2'). The "
                "card title resolves to '<Month> Series A Warrants' "
                "when set, so MISSING this on a multi-tranche offering "
                "causes BOTH tranches to render under one collapsed "
                "card. Omit for single-tranche warrant issuances and "
                "for placement-agent/underwriter comp warrants — those "
                "use descriptor='Placement Agent' / 'Underwriter' "
                "instead."
            ),
        ),
        *_COMMON_PARTY_ARGS,
        _PROPOSED_ID_ARG,
    ),
)


# ─── create_convertible ───────────────────────────────────────────────

create_convertible = Tool(
    name="create_convertible",
    description=(
        "Call when a filing newly issues a convertible note or "
        "convertible debt. `principal` is the face amount, "
        "`principal_remaining` is what's outstanding now (equals "
        "principal at issuance). `conv_price` is the per-share "
        "conversion price (or null if variable-rate with no "
        "scalar).\n\n"
        "NOT A CONVERTIBLE — straight debt with NO conversion "
        "mechanism. A term loan / back-leverage / subordinated "
        "project-finance facility (e.g. a secured back-leverage loan "
        "from a lender such as a green bank, with a fixed coupon and a "
        "long bullet maturity but NO clause letting the holder convert "
        "principal into the issuer's shares — no conversion price, no "
        "conversion ratio, no 'convertible into shares' language) is "
        "NOT a convertible and does not dilute. Do NOT call this tool "
        "and do NOT invent a maturity; call note_no_event. Only call "
        "create_convertible when the instrument is explicitly "
        "convertible into the issuer's own equity."
    ),
    mutation_kind="create_instrument",
    instrument_type="convertible",
    event_kind=None,
    valid_forms=frozenset({
        "8-K", "8-K/A", "S-1", "S-3", "S-3ASR", "F-3", "F-3ASR",
        "F-10", "F-10/A", "F-10EF",
        "424B5", "SUPPL", "10-Q", "10-K",
        "6-K", "20-F", "40-F",
    }),
    args=(
        ToolArg(
            name="principal",
            type="number",
            required=True,
            min_value=1.0,
            description=(
                "Face amount of the note in USD — CONVERTIBLE PORTION "
                "only. When the filing states that just a slice of a "
                "larger facility is convertible ('the Lenders may "
                "convert up to 30% of the amount of the loan "
                "disbursed'), set principal to that convertible slice "
                "(30% x face), NOT the full facility face: the card "
                "models dilution and only the convertible portion can "
                "become shares. A fully-convertible note's principal is "
                "its face amount as before."
            ),
        ),
        ToolArg(
            name="principal_remaining",
            type="number",
            required=True,
            min_value=0.0,
            description=(
                "Principal currently outstanding. Equals principal at "
                "first issuance; lower after partial conversions / "
                "redemptions. Same convertible-portion basis as "
                "`principal`."
            ),
        ),
        ToolArg(
            name="event_date",
            type="date",
            required=True,
            description=(
                "AGREEMENT date — when the SPA / note was signed and "
                "terms became binding (usually the 8-K filing date). "
                "NOT the closing / funding date when cash settled — "
                "those are typically 5-10 business days later and are "
                "not the date DT uses."
            ),
        ),        ToolArg(
            name="rate",
            type="number",
            required=False,
            description="Coupon / interest rate in decimal (e.g. 0.10 = 10%).",
        ),
        ToolArg(
            name="conv_price",
            type="number",
            required=False,
            description=(
                "Per-share conversion price in USD. For a fixed-price "
                "note this is the stated conversion price. For a "
                "variable / Qualified-Financing / VWAP-linked note that "
                "states a FLOOR PRICE (a stated minimum conversion price "
                "the conversion can never go below), set conv_price to "
                "that floor (the floor is what gets displayed and what "
                "sizes the share count). Leave null ONLY for a purely "
                "floating note (X% of VWAP) with NO stated floor."
            ),
        ),
        ToolArg(
            name="conv_discount_pct",
            type="number",
            required=False,
            min_value=0.01,
            description=(
                "Discount-to-market factor in DECIMAL for variable-rate "
                "notes: 0.90 for '90% of the lowest VWAP', 0.85 for "
                "'85% of the lowest traded price'. Set it WHENEVER the "
                "conversion formula references market price — alongside "
                "conv_price when the note is 'lesser of $X or Y% of "
                "VWAP' (conv_price carries the $X cap/floor, this "
                "carries Y), or alone for a purely floating note. The "
                "store renders a live effective conversion price from "
                "it; never compute the multiplication yourself. Leave "
                "null for fixed-price notes."
            ),
        ),
        ToolArg(
            name="convertible_date",
            type="date",
            required=False,
            description=(
                "Absolute first-convertible date (YYYY-MM-DD) — set "
                "ONLY when the filing prints a literal date. For the "
                "typical 'convertible from issuance' note set "
                "convertible_offset_months=0; for a hold period set the "
                "offset. The store derives this from event_date."
            ),
        ),
        ToolArg(
            name="convertible_offset_months",
            type="integer",
            required=False,
            min_value=0,
            description=(
                "Months from issuance until the note first becomes "
                "convertible — the store adds it to event_date so you "
                "never do calendar math. Use 0 for the typical "
                "'convertible from issuance' note; 6 for 'convertible "
                "six months after issuance'. Leave null when no "
                "conversion-start rule is stated."
            ),
        ),
        ToolArg(
            name="maturity",
            type="date",
            required=False,
            description=(
                "Absolute maturity date (YYYY-MM-DD) — set ONLY when the "
                "filing prints a literal date. For TERM-based language "
                "('eighteen-month term', 'matures on the second "
                "anniversary'), set maturity_months instead and the "
                "store derives this. An absolute date here overrides the "
                "computed value."
            ),
        ),
        ToolArg(
            name="maturity_months",
            type="integer",
            required=False,
            min_value=1,
            description=(
                "Note term in MONTHS, extracted from the filing — the "
                "store adds it to event_date so you never do calendar "
                "math. Convert years to months: 'eighteen-month term' → "
                "18, 'second anniversary' / 'two-year' → 24, 'one-year "
                "term' → 12, 'five-year' → 60. Common: 12-24 for "
                "Streeterville / Mast Hill toxic notes, 36-60 for senior "
                "secured. Leave null only for demand-redemption / "
                "perpetual structures with no stated maturity (rare)."
            ),
        ),
        ToolArg(
            name="oid_pct",
            type="number",
            required=False,
            description=(
                "Original issue discount in decimal (e.g. 0.10 for a "
                "10% OID — note issued at 90% of principal)."
            ),
        ),
        *_COMMON_PARTY_ARGS,
        _PROPOSED_ID_ARG,
    ),
)


# ─── create_preferred ─────────────────────────────────────────────────

create_preferred = Tool(
    name="create_preferred",
    description=(
        "Call when a filing newly issues a series of preferred stock. "
        "`series_letter` (e.g. 'D', '9') is the primary identity key "
        "within an issuer — distinguishes Series D from Series A even "
        "if conv_price drifts. `count` is the share count issued. "
        "`stated_value` is the per-share face value; "
        "`liquidation_preference` is the aggregate face amount across "
        "the tranche. When the filing attaches a Certificate of "
        "Designation (or summarises one), ALWAYS scan its Conversion "
        "Rights / Conversion section and populate the conversion "
        "fields too: `conversion_ratio` when the COD states a fixed "
        "common-per-preferred ratio ('at a rate of N shares of common "
        "stock per share'), or `conv_price` when it states a dollar "
        "price; `convertible_offset_months` when the COD states a "
        "hold period ('Following N months from the issuance date', "
        "'no earlier than the Mth anniversary'). These belong on the "
        "creation call — they are NOT separate amend events."
    ),
    mutation_kind="create_instrument",
    instrument_type="preferred",
    event_kind=None,
    valid_forms=frozenset({
        "8-K", "8-K/A", "S-1", "S-3", "S-3ASR", "F-3", "F-3ASR",
        "F-10", "F-10/A", "F-10EF",
        "424B5", "SUPPL", "10-Q", "10-K",
        "6-K", "20-F", "40-F",
    }),
    args=(
        ToolArg(
            name="count",
            type="number",
            required=True,
            min_value=0.0,
            description=(
                "Preferred-share count issued. Fractional values are "
                "valid (count = principal / stated_value when the "
                "division isn't integer — common with "
                "Inpixon/Streeterville-style series)."
            ),
        ),
        ToolArg(
            name="series_letter",
            type="string",
            required=True,
            min_length=1,
            description=(
                "Series identifier — letter ('A', 'B', 'D') or integer "
                "string ('9', '10'). Both are valid; copy the form "
                "used in the filing."
            ),
        ),
        ToolArg(
            name="event_date",
            type="date",
            required=True,
            description=(
                "AGREEMENT date — when the SPA / Certificate of "
                "Designation was signed and terms became binding "
                "(usually the 8-K filing date). NOT the closing / "
                "funding date when cash settled — those are typically "
                "5-10 business days later and are not the date DT uses."
            ),
        ),        ToolArg(
            name="conv_price",
            type="number",
            required=False,
            description=(
                "Per-share conversion price into common (USD). Set ONLY "
                "when the filing states a price in dollars. When the "
                "filing instead states a fixed ratio ('N shares of "
                "common stock per share' of preferred), pass "
                "conversion_ratio instead and leave this null — the "
                "store derives conv_price = stated_value / "
                "conversion_ratio. Leave both null for variable-rate "
                "series."
            ),
        ),
        ToolArg(
            name="conversion_ratio",
            type="number",
            required=False,
            min_value=0.0,
            description=(
                "Number of COMMON shares issuable per ONE share of "
                "preferred under the Certificate of Designation's "
                "fixed conversion mechanism. Example: 'convertible "
                "into common stock at a rate of 12.5 shares of common "
                "stock per share' → 12.5. The store derives "
                "conv_price = stated_value / conversion_ratio so you "
                "do NOT also pass conv_price. Leave null for "
                "variable-rate / VWAP-linked / lookback conversion "
                "mechanisms."
            ),
        ),
        ToolArg(
            name="convertible_date",
            type="date",
            required=False,
            description=(
                "Absolute first-convertible date (YYYY-MM-DD) — set "
                "ONLY when the filing prints a literal date. For the "
                "typical 'convertible from issuance' series set "
                "convertible_offset_months=0; for a hold period set the "
                "offset. The store derives this from event_date."
            ),
        ),
        ToolArg(
            name="convertible_offset_months",
            type="integer",
            required=False,
            min_value=0,
            description=(
                "Months from issuance until the series first becomes "
                "convertible — the store adds it to event_date. Use 0 "
                "for the typical 'convertible from issuance' series; set "
                "a positive offset whenever the Certificate of "
                "Designation states a hold period. Concrete phrasings: "
                "'Following three months from the issuance date, the "
                "Series D Preferred Stock is convertible into common "
                "stock' → 3; 'convertible after a 6-month hold' → 6; "
                "'no earlier than the first anniversary of issuance' "
                "→ 12. Leave null only when no conversion-start rule "
                "is stated."
            ),
        ),
        ToolArg(
            name="maturity",
            type="date",
            required=False,
            description=(
                "Absolute mandatory-redemption date (YYYY-MM-DD) — set "
                "ONLY when the Certificate of Designation prints a "
                "literal 'redeemed for cash on [DATE]'. For a stated "
                "redemption TERM ('mandatory redemption on the third "
                "anniversary'), set maturity_months instead. An absolute "
                "date here overrides the computed value."
            ),
        ),
        ToolArg(
            name="maturity_months",
            type="integer",
            required=False,
            min_value=1,
            description=(
                "Mandatory-redemption term in MONTHS — the store adds it "
                "to event_date. MOST preferred series have NO mandatory "
                "redemption (perpetual until converted), so leave this "
                "null in the common case. Set ONLY when the Certificate "
                "of Designation states a redemption-by term: 'mandatory "
                "redemption on the third anniversary' → 36. Do NOT "
                "hallucinate a term for ordinary convertible preferred — "
                "perpetual is the default. Not the conversion-window "
                "expiry (a different, rarely-stated concept)."
            ),
        ),
        ToolArg(
            name="stated_value",
            type="number",
            required=False,
            description="Per-share stated value / face amount.",
        ),
        ToolArg(
            name="liquidation_preference",
            type="number",
            required=False,
            description=(
                "Aggregate liquidation preference (USD) across the "
                "tranche, typically count × stated_value."
            ),
        ),
        ToolArg(
            name="dividend_rate",
            type="number",
            required=False,
            description=(
                "Dividend rate in decimal (e.g. 0.08 for 8%). Null "
                "when the dividend is described as a formula rather "
                "than a scalar."
            ),
        ),
        ToolArg(
            name="principal_remaining",
            type="number",
            required=False,
            description=(
                "Aggregate face amount of preferred still outstanding "
                "(typically count × stated_value). Used when partial "
                "redemptions have reduced the balance."
            ),
        ),
        *_COMMON_PARTY_ARGS,
        _PROPOSED_ID_ARG,
    ),
)


# ─── create_equity_line ───────────────────────────────────────────────

create_equity_line = Tool(
    name="create_equity_line",
    description=(
        "Call when a filing executes a new Standby Equity Purchase "
        "Agreement (SEPA), equity line of credit (ELOC), or similar "
        "discretionary common-stock purchase commitment with a named "
        "investor (YA II, M2B Funding, Hudson Bay, Tumim, etc.). NOT the "
        "same as an ATM — an ATM uses a Sales Agent to sell into the "
        "open market; an equity_line has a single COUNTERPARTY who "
        "buys directly from the issuer at a market-linked price.\n\n"
        "DO NOT use for ATM sales agreements (call create_atm). "
        "DO NOT use for new draws on an existing equity line — that's "
        "record_drawdown against the existing instrument_id."
    ),
    mutation_kind="create_instrument",
    instrument_type="equity_line",
    event_kind=None,
    valid_forms=frozenset({
        "8-K", "8-K/A", "S-1", "S-3", "S-3ASR", "F-3", "F-3ASR",
        "F-10", "F-10/A", "F-10EF",
        "424B5", "SUPPL",
        "6-K", "20-F", "40-F",
    }),
    args=(
        ToolArg(
            name="capacity_usd",
            type="number",
            required=True,
            min_value=1.0,
            description=(
                "Dollar cap stated in the purchase agreement (e.g. "
                "'up to $25,000,000 of our common stock')."
            ),
        ),
        ToolArg(
            name="agreement_date",
            type="date",
            required=True,
            description=(
                "Execution date of the purchase agreement. Primary "
                "identity key — successive purchase agreements with "
                "the same investor (Yorkville re-ups every 12-18 mo) "
                "are DISTINCT instruments."
            ),
        ),
        ToolArg(
            name="counterparty_canonical",
            type="string",
            required=True,
            min_length=2,
            description=(
                "The named CONTRACTING INVESTOR / buyer ('YA II', "
                "'Hudson Bay', 'M2B Funding'). Required — equity lines "
                "always have a named counterparty. Use the contracting "
                "Investor entity, NOT the fund-manager nickname: a "
                "Yorkville SEPA is signed by 'YA II PN, LTD.' (the "
                "Investor) even though the 6-K prose abbreviates it as "
                "'Yorkville' (the manager) — emit 'YA II' here, not "
                "'Yorkville'."
            ),
        ),
        ToolArg(
            name="event_date",
            type="date",
            required=True,
            description="Filing date (YYYY-MM-DD).",
        ),        ToolArg(
            name="remaining_capacity_usd",
            type="number",
            required=False,
            description="Capacity not yet drawn. Omit on first registration.",
        ),
        ToolArg(
            name="drawn_usd",
            type="number",
            required=False,
            description="Cumulative drawn USD to date.",
        ),
        ToolArg(
            name="placement_agent_canonical",
            type="string",
            required=False,
            description="Bank brokering the agreement, if any.",
        ),
        ToolArg(
            name="agreement_end_date",
            type="date",
            required=False,
            description=(
                "Contractual term-end of the purchase agreement (ISO "
                "YYYY-MM-DD) when the agreement specifies a fixed "
                "expiry — e.g. 'this Agreement will terminate on … "
                "September 11, 2027' (Yorkville/M2B SEPAs typically "
                "have 2–5 year terms). Omit when open-ended."
            ),
        ),
        ToolArg(
            name="term_months",
            type="integer",
            required=False,
            min_value=1,
            description=(
                "Agreement term in MONTHS when the SEPA/ELOC states a "
                "duration instead of a literal end date — the store "
                "adds it to agreement_date so you never do calendar "
                "math. Convert: '24-month term from the Commencement "
                "Date' → 24, 'five-year term' → 60. Use this whenever "
                "the term is a duration; use agreement_end_date instead "
                "only when the filing prints an explicit calendar date. "
                "Omit for open-ended agreements."
            ),
        ),
        ToolArg(
            name="descriptor",
            type="string",
            required=False,
            enum_values=_DESCRIPTOR_ENUM,
            description=(
                "Usually 'ELOC' for equity lines; leave null otherwise."
            ),
        ),
        _PROPOSED_ID_ARG,
    ),
)


# ─── create_s1_offering ───────────────────────────────────────────────

create_s1_offering = Tool(
    name="create_s1_offering",
    description=(
        "Call when an S-1 / F-1 / S-1/A registers a new public "
        "offering (typically an IPO or follow-on). "
        "`anticipated_deal_size` is the expected gross proceeds; for "
        "preliminary S-1s with no priced cover, use the registration "
        "fee table's maximum aggregate offering price. Attached "
        "warrants (per share + per common-share warrant) get a "
        "SEPARATE create_warrant call — this tool only records the "
        "offering frame.\n\n"
        "DO NOT use for resale S-1s. Cover-page tells: the prospectus "
        "says it 'relates to the resale of … shares … by the selling "
        "stockholder', and 'We are not selling any securities under "
        "this prospectus and will not receive any of the proceeds from "
        "the resale by the selling stockholder'. This includes:\n"
        "  - resale of shares already held by named investors / "
        "placement agents from a prior private placement,\n"
        "  - resale registrations for an equity line of credit (ELOC) / "
        "committed equity facility: the cover lists 'Purchase Shares' "
        "and 'Commitment Shares' to be issued from time to time to the "
        "ELOC counterparty (e.g. Sixth Borough, Lincoln Park, B. Riley). "
        "The ELOC itself is a separate create_equity_line call — the "
        "S-1 just registers the resale path and is NOT a new offering.\n"
        "These are not new dilution events; skip the call.\n\n"
        "DO NOT use when this filing re-discloses an offering ALREADY in "
        "the ledger as an s1_offering — an S-1/A amends the same offering "
        "(narrowed share count / pricing range, weeks after the S-1) and "
        "a 424B4 / pricing supplement prices the takedown. Both keep the "
        "offering's ORIGINAL S-1 date; call amend_s1_offering against the "
        "existing row (the 424B4 ALSO needs record_drawdown). Creating "
        "here mints a duplicate offering card."
    ),
    mutation_kind="create_instrument",
    instrument_type="s1_offering",
    event_kind=None,
    valid_forms=frozenset({
        "S-1", "S-1/A", "F-1", "F-1/A", "424B4",
    }),
    args=(
        ToolArg(
            name="anticipated_deal_size",
            type="number",
            required=True,
            min_value=1.0,
            description=(
                "Expected gross proceeds in USD. For priced offerings "
                "this is shares × price; for preliminary S-1s use the "
                "fee-table maximum aggregate offering price."
            ),
        ),
        ToolArg(
            name="event_date",
            type="date",
            required=True,
            description="Filing date.",
        ),        ToolArg(
            name="warrant_strike",
            type="number",
            required=False,
            description=(
                "Strike of attached warrants (if any), per common "
                "share. The warrant itself is registered via a sibling "
                "create_warrant call — this field is a cover-page hint."
            ),
        ),
        ToolArg(
            name="warrant_coverage_pct",
            type="number",
            required=False,
            description=(
                "Warrant coverage as a decimal (e.g. 1.0 = one warrant "
                "per common share, 0.5 = half-warrant)."
            ),
        ),
        ToolArg(
            name="sold_to_date",
            type="number",
            required=False,
            description=(
                "Cumulative gross proceeds sold to date when the S-1/A "
                "or subsequent 424B4 reports actual sales below the "
                "registered maximum."
            ),
        ),
        ToolArg(
            name="placement_agent_canonical",
            type="string",
            required=False,
            description=(
                "Lead underwriter / book-runner. Short canonical name."
            ),
        ),
        _PROPOSED_ID_ARG,
    ),
)


# ─── create_equity ────────────────────────────────────────────────────

create_equity = Tool(
    name="create_equity",
    description=(
        "Call when common stock is ISSUED directly outside the other "
        "instrument frames — typically a private placement (PIPE), "
        "an inducement issuance, a direct registered offering of a "
        "fixed share count without warrants, or a stock-for-services "
        "issuance.\n\n"
        "DO NOT use for: warrant exercises (record_exercise), "
        "preferred conversions (record_conversion), ATM/equity-line "
        "drawdowns (record_drawdown), or share issuances against an "
        "existing S-1 / shelf — those are drawdowns, not new equity "
        "creates."
    ),
    mutation_kind="create_instrument",
    instrument_type="equity",
    event_kind=None,
    valid_forms=frozenset({
        "8-K", "8-K/A", "S-1", "S-3", "S-3ASR", "F-3", "F-3ASR",
        "F-10", "F-10/A", "F-10EF",
        "424B5", "424B3", "SUPPL",
        "6-K", "20-F", "40-F",
    }),
    args=(
        ToolArg(
            name="count",
            type="number",
            required=True,
            min_value=1.0,
            description="Number of common shares issued.",
        ),
        ToolArg(
            name="price_per_share",
            type="number",
            required=True,
            min_value=0.0,
            description=(
                "Price per share in USD. 0 for stock-for-services or "
                "stock dividends; otherwise the offering price."
            ),
        ),
        ToolArg(
            name="event_date",
            type="date",
            required=True,
            description="Filing or issuance date.",
        ),
        ToolArg(
            name="closing_date",
            type="date",
            required=False,
            description=(
                "Set ONLY when THIS filing states the placement "
                "closed / was consummated / funded (the common "
                "'signed and closed' 8-K). Books the cash proceeds "
                "(count × price_per_share) as of this date. Leave "
                "unset for a signed-but-pending SPA — no cash is "
                "counted until a later closing filing confirms it "
                "via confirm_closing."
            ),
        ),
        *_COMMON_PARTY_ARGS,
        _PROPOSED_ID_ARG,
    ),
)


__all__ = [
    "create_atm", "create_shelf",
    "create_warrant", "create_convertible", "create_preferred",
    "create_equity_line", "create_s1_offering", "create_equity",
]
