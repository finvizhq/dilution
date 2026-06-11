"""amend_* tool definitions.

Each amend tool targets one instrument type and exposes only the
fields the apply layer (store.py:_apply_amend) actually knows how to
update for that type. Sparse updates: every non-id/event/quote field
is optional. Cross-arg validator (in parse.py) enforces ≥1 mutating
field set — matches the current AmendInstrument._check_non_empty rule.

`field_updates` and `outstanding_updates` from the legacy Pydantic
AmendInstrument get auto-split at parse time: each typed dataclass
exposes terms_updates() / outstanding_updates() helpers that route
fields to the right side of the SQL JSON blob.

Used by: anchor reconciliation (capacity drift, count drift),
agreement-date corrections, repricings, partial-redemption
write-downs, capacity bumps via S-3/A or S-3MEF.
"""

from __future__ import annotations

from ._base import Tool, ToolArg, PROPOSED_ID_PATTERN


_INSTRUMENT_ID_ARG = ToolArg(
    name="instrument_id",
    type="string",
    required=True,
    min_length=3,
    description=(
        "Existing ledger row to amend. Use the id shown in the current "
        "ledger view (e.g. 'ATM-012', 'W-007'). DO NOT invent — if the "
        "target isn't in the ledger, call create_<type> instead."
    ),
)

_EVENT_DATE_ARG = ToolArg(
    name="event_date",
    type="date",
    required=True,
    description=(
        "Date this amendment takes effect (YYYY-MM-DD). For anchor "
        "reconciliations use the periodic filing's as_of date; for "
        "capacity bumps use the S-3/A or 424B5 filing date."
    ),
)


# ─── amend_atm ────────────────────────────────────────────────────────

amend_atm = Tool(
    name="amend_atm",
    description=(
        "Update an existing ATM IN PLACE — no new card is created. "
        "Common cases:\n"
        "  - baby-shelf down-size at the I.B.5 ceiling (set capacity_usd "
        "AND remaining_capacity_usd to the new cap)\n"
        "  - anchor reconciliation from a 10-Q footnote stating "
        "remaining capacity or cumulative drawn\n"
        "  - placement-agent rebrand mid-program (Cowen → TD Cowen)\n\n"
        "DO NOT use for an Amended-and-Restated Equity Distribution / "
        "Sales Agreement that materially restates the program (new "
        "aggregate capacity, new agent, or 'Amendment No. N' that DT "
        "would show as a SEPARATE card) — call restate_atm instead. "
        "DO NOT use for new ATM agreements with no predecessor — call "
        "create_atm. DO NOT use for drawdowns (call record_drawdown)."
    ),
    mutation_kind="amend_instrument",
    instrument_type="atm",
    event_kind=None,
    valid_forms=frozenset({
        "S-3", "S-3/A", "S-3ASR", "S-3MEF", "F-3", "F-3/A", "F-3ASR",
        "F-3MEF", "F-10", "F-10/A", "F-10EF",
        "8-K", "8-K/A", "10-Q", "10-K", "10-Q/A", "10-K/A",
        "20-F", "40-F", "6-K",
        "424B5", "SUPPL",
    }),
    args=(
        _INSTRUMENT_ID_ARG,
        _EVENT_DATE_ARG,
        ToolArg(
            name="capacity_usd", type="number", required=False,
            description="New cap. Set when an Amended and Restated EDA expands or contracts the headline.",
        ),
        ToolArg(
            name="remaining_capacity_usd", type="number", required=False,
            description="Cap remaining post-amendment. Pair with capacity_usd on baby-shelf down-sizes.",
        ),
        ToolArg(
            name="drawn_usd", type="number", required=False,
            description="Cumulative drawn to date. Set on anchor reconciliations.",
        ),
        ToolArg(
            name="placement_agent_canonical", type="string", required=False,
            description="New canonical Sales Agent name after a banker rebrand.",
        ),
        ToolArg(
            name="agreement_date", type="date", required=False,
            description=(
                "RARE — only when correcting a wrong agreement_date "
                "from a prior create. Different agreement_date "
                "normally indicates a different instrument."
            ),
        ),
        ToolArg(
            name="agreement_end_date", type="date", required=False,
            description=(
                "Contractual term-end after an amendment extends or "
                "shortens the EDA term — e.g. 'Amendment No. 2 … "
                "extended the term of the Equity Distribution Agreement "
                "until … December 31, 2024'. Set this on Amended-and-"
                "Restated EDAs and amendment letters that move the "
                "term-end. Do NOT use this for a terminated EDA — call "
                "close_instrument(reason='terminated') instead."
            ),
        ),
    ),
)


# ─── restate_atm ──────────────────────────────────────────────────────

restate_atm = Tool(
    name="restate_atm",
    description=(
        "An Amended-and-Restated Equity Distribution / Sales Agreement "
        "(or 'Amendment No. N') that DilutionTracker tracks as a NEW ATM "
        "card. Use this — NOT amend_atm — whenever an existing ATM is "
        "materially restated: a new aggregate capacity, a new/changed "
        "sales-agent line-up, or a fresh amendment that opens new "
        "selling capacity. It mints a fresh successor ATM (drawn reset "
        "to zero — the prior program's cumulative sales stay on the "
        "predecessor card) and marks the named predecessor superseded "
        "('Replaced'). Pointing predecessor_id at a live ATM IS the "
        "statement that it is being restated.\n\n"
        "predecessor_id is the existing ATM row this restates (from the "
        "current ledger view). capacity_usd / placement_agent_canonical / "
        "agreement_date describe the RESTATED program as of this filing.\n\n"
        "If two ATMs are genuinely CONCURRENT and independent (issuers do "
        "run more than one program at once), they are NOT a restate pair — "
        "emit a separate create_atm for the new one and do NOT point "
        "predecessor_id at the other.\n\n"
        "DO NOT use for an in-place tweak with no new card (use "
        "amend_atm), a first-time ATM with no predecessor (use "
        "create_atm), or a drawdown (use record_drawdown)."
    ),
    mutation_kind="restate_instrument",
    instrument_type="atm",
    event_kind=None,
    # 424B5 / S-3 carriers only. NOT 8-K — an amendment announced on an
    # 8-K also lands as a 424B5 supplement, so admitting it double-mints
    # the card (XTIA Amendment No. 3). NOT POS AM — a post-effective
    # amendment RE-REGISTERS the host shelf's prospectus; it is not a new
    # sales agreement and must not spawn a card (FCEL's 2025-03-05 POS AM
    # wrongly minted a phantom restated ATM that then superseded the live
    # April-2024 row). The 424B/S-3 carriers catch FCEL's April-2024 and
    # Dec-2025 cases (both filed under 333-274971).
    valid_forms=frozenset({
        "S-3", "S-3/A", "S-3ASR", "S-3MEF", "F-3", "F-3/A", "F-3ASR",
        "F-3MEF", "424B5", "424B3", "424B4", "424B2", "SUPPL",
    }),
    args=(
        ToolArg(
            name="predecessor_id", type="string", required=True,
            min_length=3,
            description=(
                "The existing ATM row being restated (e.g. 'ATM-012'), "
                "from the current ledger view. DO NOT invent — if no "
                "prior ATM exists, call create_atm instead."
            ),
        ),
        ToolArg(
            name="capacity_usd", type="number", required=True,
            min_value=1.0,
            description=(
                "Aggregate offering amount of the RESTATED agreement — "
                "the new headline cap, not the predecessor's."
            ),
        ),
        ToolArg(
            name="placement_agent_canonical", type="string", required=True,
            description=(
                "Canonical sales-agent name(s) under the restated "
                "agreement. State the agent(s) named in THIS filing — do "
                "not assume they match the predecessor's."
            ),
        ),
        ToolArg(
            name="agreement_date", type="date", required=True,
            description=(
                "Signing/effective date of the amended-and-restated "
                "agreement (YYYY-MM-DD). DT keys ATM cards by this date."
            ),
        ),
        ToolArg(
            name="supersede_prior", type="boolean", required=False,
            description=(
                "Deprecated / ignored — a restate always supersedes the "
                "predecessor you name; the store owns this decision. Kept "
                "only for backward-compat with older tool transcripts."
            ),
        ),
        _EVENT_DATE_ARG,
        ToolArg(
            name="remaining_capacity_usd", type="number", required=False,
            description=(
                "Capacity not yet drawn under the restated agreement. "
                "Omit on a fresh restatement (defaults to capacity_usd); "
                "set only when the filing states a partial remaining."
            ),
        ),
        ToolArg(
            name="agreement_end_date", type="date", required=False,
            description="Contractual term-end of the restated agreement, if stated.",
        ),
        ToolArg(
            name="proposed_id", type="string", required=False,
            pattern=PROPOSED_ID_PATTERN,
            description=(
                "Optional stable slug for the successor so later "
                "mutations in THIS filing (e.g. a drawdown) can reference "
                "it. Lowercase-kebab."
            ),
        ),
    ),
)


# ─── amend_equity_line ────────────────────────────────────────────────

amend_equity_line = Tool(
    name="amend_equity_line",
    description=(
        "Update an existing equity_line (SEPA/ELOC). Used for capacity "
        "bumps on amended purchase agreements with the same investor, "
        "anchor reconciliations from 10-Q footnotes stating remaining "
        "capacity or cumulative draws, and term-end extensions in "
        "amended-and-restated SEPAs."
    ),
    mutation_kind="amend_instrument",
    instrument_type="equity_line",
    event_kind=None,
    valid_forms=frozenset({
        "S-3", "S-3/A", "8-K", "8-K/A", "10-Q", "10-K", "10-Q/A", "10-K/A",
        "20-F", "40-F", "6-K",
        "424B5", "SUPPL",
    }),
    args=(
        _INSTRUMENT_ID_ARG,
        _EVENT_DATE_ARG,
        ToolArg(
            name="capacity_usd", type="number", required=False,
            description="New cap.",
        ),
        ToolArg(
            name="remaining_capacity_usd", type="number", required=False,
            description="Cap remaining post-amendment.",
        ),
        ToolArg(
            name="drawn_usd", type="number", required=False,
            description="Cumulative drawn to date.",
        ),
        ToolArg(
            name="agreement_end_date", type="date", required=False,
            description=(
                "New contractual term-end after the amendment moves it. "
                "Do NOT use this for a terminated SEPA — call "
                "close_instrument(reason='terminated') instead."
            ),
        ),
    ),
)


# ─── amend_shelf ──────────────────────────────────────────────────────

amend_shelf = Tool(
    name="amend_shelf",
    description=(
        "Update an existing primary shelf. Used by:\n"
        "  - S-3/A capacity changes (e.g. /A raises cap from $100M to "
        "$200M after a higher-fee table)\n"
        "  - S-3MEF amendments that bolt additional capacity onto an "
        "existing S-3 ($200M base + $50M MEF = $250M total — call "
        "amend_shelf with the new total, not create_shelf)\n"
        "  - anchor reconciliations from 10-Q footnotes stating "
        "remaining shelf capacity\n\n"
        "DO NOT use for 424B prospectus takedowns — those are "
        "record_drawdown against the shelf's instrument_id."
    ),
    mutation_kind="amend_instrument",
    instrument_type="shelf",
    event_kind=None,
    valid_forms=frozenset({
        "S-3", "S-3/A", "S-3ASR", "S-3MEF", "F-3", "F-3/A", "F-3ASR",
        "F-3MEF", "F-10", "F-10/A", "F-10EF",
        "POS AM", "10-Q", "10-K", "10-Q/A", "10-K/A",
        "20-F", "40-F", "6-K",
    }),
    args=(
        _INSTRUMENT_ID_ARG,
        _EVENT_DATE_ARG,
        ToolArg(
            name="capacity_usd", type="number", required=False,
            description="New headline cap.",
        ),
        ToolArg(
            name="remaining_capacity_usd", type="number", required=False,
            description="Cap remaining post-amendment.",
        ),
    ),
)


# ─── amend_warrant ────────────────────────────────────────────────────

amend_warrant = Tool(
    name="amend_warrant",
    description=(
        "Update an existing warrant tranche that is STILL ALIVE — "
        "term has shifted but the instrument continues to exist. "
        "Common cases:\n"
        "  - strike repricing in an inducement / exchange offer\n"
        "  - expiration-date EXTENSION (e.g. 'the expiry of the "
        "Series A Warrants has been extended from 2026-01-15 to "
        "2028-01-15') — sets `expiration` to the new later date\n"
        "  - anti-dilution clause classification update\n"
        "  - count adjustment (rare — usually a partial exercise is "
        "record_exercise, not amend_warrant)\n"
        "  - known_owners disclosed in a later 13G after an SPA was "
        "originally generic\n\n"
        "DO NOT use for: a warrant that EXPIRED without exercise "
        "(use close_instrument(reason='expired') — do NOT set "
        "count=0 or count=actual_outstanding here, the ledger "
        "needs the explicit closure to flip status). A warrant "
        "tranche fully exchanged for shares or replaced by a new "
        "tranche (use close_instrument(reason='superseded') paired "
        "with the successor create_*). A 1-for-N reverse split or "
        "ADS-ratio change affecting every active warrant (use "
        "apply_split ONCE — the store rewrites all matching "
        "warrants in one pass)."
    ),
    mutation_kind="amend_instrument",
    instrument_type="warrant",
    event_kind=None,
    valid_forms=frozenset({
        "8-K", "8-K/A", "S-1", "S-1/A", "S-3", "S-3/A", "F-3", "F-3/A",
        "F-10", "F-10/A", "F-10EF",
        "424B5", "SUPPL", "10-Q", "10-K", "10-Q/A", "10-K/A",
        "20-F", "40-F", "6-K",
        "PRE 14A", "DEF 14A", "DEFA14A", "DEFM14A",
    }),
    args=(
        _INSTRUMENT_ID_ARG,
        _EVENT_DATE_ARG,
        ToolArg(
            name="count", type="number", required=False,
            description=(
                "Updated CURRENTLY-OUTSTANDING count — the number of "
                "warrants still alive, NOT the original issuance size. "
                "If a 10-Q footnote says 'as of period end, 1,778 "
                "warrants remain outstanding from the July 2024 "
                "tranche', emit 1778 here. If the same footnote also "
                "says 'an aggregate of 2,500 warrants were issued', "
                "DO NOT emit 2500 — that's the initial size, which "
                "lives in initial_count and never changes.\n\n"
                "Prefer record_exercise / close_instrument over this "
                "field for known events: a partial exercise that "
                "consumed the missing shares is record_exercise (drops "
                "count by `shares`); a full exercise / expiry / "
                "termination is close_instrument. Use amend.count ONLY "
                "when the filing restates the current outstanding "
                "without a discrete consuming event (e.g. anchor "
                "reconciliation against a periodic table, or a 13G "
                "that reveals a count we didn't have)."
            ),
        ),
        ToolArg(
            name="strike", type="number", required=False,
            description=(
                "Updated per-share strike for THIS tranche only — "
                "repricings (inducement letter, ratchet trigger, "
                "exchange offer at a lower strike). The store keeps "
                "the original strike in history but cards show the "
                "current value.\n\n"
                "DO NOT use for a stock split or ADS-ratio change "
                "that mechanically rewrites every active tranche — "
                "that's apply_split, which the store applies ONCE "
                "and propagates to every matching warrant / "
                "convertible / preferred. Emitting amend.strike per "
                "tranche after a split would double-apply the ratio."
            ),
        ),
        ToolArg(
            name="exercisable_date", type="date", required=False,
            description=(
                "Updated first-exercisable date. Use when a later "
                "filing reveals or refines this date and the existing "
                "ledger row has it null or wrong. See "
                "create_warrant.exercisable_date for the derivation "
                "patterns ('exercisable immediately upon issuance' → "
                "event_date, etc.) — the same rules apply here when "
                "the amending filing uses term-based language."
            ),
        ),
        ToolArg(
            name="expiration", type="date", required=False,
            description=(
                "Updated expiration. Use when a later filing reveals "
                "or refines the term and the existing ledger row has "
                "it null or wrong. See create_warrant.expiration for "
                "the derivation patterns ('five-year term' → "
                "event_date + 5y, anchored-to-Initial-Exercise-Date "
                "variants, etc.)."
            ),
        ),
        ToolArg(
            name="is_pre_funded", type="boolean", required=False,
            description="Flip when the strike resets to near-zero.",
        ),
        ToolArg(
            name="series_letter", type="string", required=False,
            min_length=1,
            description=(
                "Clarify the warrant's series identifier when a later "
                "filing (resale S-1, 13G) discloses 'Series A' / "
                "'Series B' for a previously-generic tranche. Used to "
                "split a single collapsed card into per-series cards."
            ),
        ),
        ToolArg(
            name="known_owners", type="array", required=False,
            items_type="string",
            description=(
                "Named holders disclosed in a LATER filing (resale "
                "S-1's selling-stockholder table, 13G/13D, 10-K "
                "subsequent-events footnote) when the original create "
                "captured only a generic counterparty. Short form, "
                "preserve the filing's exact entity tag ('Armistice', "
                "not 'Armistice Capital Master Fund Ltd.')."
            ),
        ),
        ToolArg(
            name="issue_date",
            type="date",
            required=False,
            description=(
                "Canonical issuance date for card labeling. For the "
                "STANDARD signing-then-closing flow (FPI 6-K pair, "
                "8-K announce → 8-K close), call CONFIRM_CLOSING "
                "instead — it relabels issue_date AND exercisable_date "
                "AND slides expiration so the N-year term is preserved "
                "atomically. This `issue_date` arg is an edge-case "
                "escape valve for corrections that don't fit the "
                "closing-event shape — e.g. a 10-K footnote retroactively "
                "discloses an earlier 'issued on' date than the create "
                "captured, with no closing semantics.\n\n"
                "DO NOT use for: any signing-then-closing pair (use "
                "confirm_closing); same-filing same-day signing+closing "
                "(the create's event_date already equals issuance); "
                "an exercise / partial conversion (use record_exercise "
                "/ record_conversion); a strike repricing (use "
                "`strike`)."
            ),
        ),
    ),
)


# ─── amend_convertible ────────────────────────────────────────────────

amend_convertible = Tool(
    name="amend_convertible",
    description=(
        "Update an existing convertible. Used for repricings "
        "(conv_price drops on a ratchet trigger), maturity extensions, "
        "principal write-downs from partial redemption (or use "
        "record_partial_redemption — same effect via outstanding "
        "update)."
    ),
    mutation_kind="amend_instrument",
    instrument_type="convertible",
    event_kind=None,
    valid_forms=frozenset({
        "8-K", "8-K/A", "S-3", "S-3/A", "10-Q", "10-K",
        "10-Q/A", "10-K/A", "20-F", "40-F", "6-K",
    }),
    args=(
        _INSTRUMENT_ID_ARG,
        _EVENT_DATE_ARG,
        ToolArg(
            name="principal_remaining", type="number", required=False,
            description=(
                "Updated CURRENTLY-OUTSTANDING principal — the face "
                "amount the issuer still owes on this note, NOT the "
                "original face value. If a 10-Q says 'as of period "
                "end, $3,000,000 of principal remains under the note', "
                "emit 3000000. The original principal lives "
                "immutably on terms.principal — do not touch it here.\n\n"
                "Prefer record_conversion (drops principal_remaining "
                "by principal_converted) and "
                "record_partial_redemption (drops by "
                "principal_redeemed) for known events. Use "
                "amend.principal_remaining ONLY when the filing "
                "restates the outstanding without naming a specific "
                "consuming event (anchor reconciliation, "
                "forbearance-letter restatement, etc.)."
            ),
        ),
        ToolArg(
            name="conv_price", type="number", required=False,
            description="Updated conversion price (repricing / ratchet trigger).",
        ),
        ToolArg(
            name="conv_discount_pct", type="number", required=False,
            min_value=0.01,
            description=(
                "Updated discount-to-market factor in DECIMAL (0.90 = "
                "'90% of lowest VWAP') when an amendment changes the "
                "variable-rate formula, or when a later filing first "
                "discloses it. Same semantics as on create_convertible."
            ),
        ),
        ToolArg(
            name="convertible_date", type="date", required=False,
            description="Updated first-convertible date.",
        ),
        ToolArg(
            name="maturity", type="date", required=False,
            description=(
                "Updated note maturity. Use when a later filing extends "
                "the term (forbearance / amendment) and the existing "
                "ledger row has it null or wrong. See "
                "create_convertible.maturity for the derivation "
                "patterns ('eighteen-month term' → event_date + 18m, "
                "etc.). Common case: a forbearance 8-K pushes maturity "
                "out by 6-12 months."
            ),
        ),
    ),
)


# ─── amend_preferred ──────────────────────────────────────────────────

amend_preferred = Tool(
    name="amend_preferred",
    description=(
        "Update an existing preferred series. Repricings, "
        "stated-value resets, partial-redemption count drops, dividend "
        "rate changes after a forbearance, etc."
    ),
    mutation_kind="amend_instrument",
    instrument_type="preferred",
    event_kind=None,
    valid_forms=frozenset({
        "8-K", "8-K/A", "S-3", "S-3/A", "10-Q", "10-K",
        "10-Q/A", "10-K/A", "20-F", "40-F", "6-K",
    }),
    args=(
        _INSTRUMENT_ID_ARG,
        _EVENT_DATE_ARG,
        ToolArg(
            name="count", type="number", required=False,
            description=(
                "Updated CURRENTLY-OUTSTANDING preferred share count — "
                "the number of preferred shares still alive, NOT the "
                "original issuance size. If the 10-Q's preferred-stock "
                "footnote says 'as of period end, 425 shares of "
                "Series D remain outstanding', emit 425. The original "
                "issuance size lives in initial_count and never moves.\n\n"
                "Prefer record_conversion / record_partial_redemption / "
                "close_instrument over this field for known events. "
                "Use amend.count ONLY for anchor reconciliation against "
                "a periodic table or restatement that doesn't tie to a "
                "specific consuming event."
            ),
        ),
        ToolArg(
            name="conv_price", type="number", required=False,
            description=(
                "Updated conversion price into common (USD). Set ONLY "
                "when the later filing states a price in dollars. When "
                "it instead states a fixed ratio ('convertible into N "
                "common shares / ADSs per preferred share'), pass "
                "conversion_ratio instead and leave this null — the "
                "store derives conv_price = stated_value / "
                "conversion_ratio from the existing row."
            ),
        ),
        ToolArg(
            name="conversion_ratio", type="number", required=False,
            min_value=0.0,
            description=(
                "Number of COMMON shares (or ADSs) issuable per ONE "
                "preferred share, when a LATER filing first discloses "
                "the fixed conversion mechanism that the create filing "
                "lacked. Example: 'each Preferred Share is convertible "
                "into 364 ADSs' → 364. The store derives "
                "conv_price = stated_value / conversion_ratio using the "
                "row's existing stated_value, so do NOT also pass "
                "conv_price. Leave null for variable / VWAP-linked "
                "mechanisms. (On the original issuance use "
                "create_preferred.conversion_ratio instead.)"
            ),
        ),
        ToolArg(
            name="convertible_date", type="date", required=False,
            description="Updated first-convertible date.",
        ),
        ToolArg(
            name="maturity", type="date", required=False,
            description=(
                "Updated mandatory-redemption maturity. Use when an "
                "amendment to the Certificate of Designation extends "
                "(or first establishes) the redemption-by date. See "
                "create_preferred.maturity — most preferreds remain "
                "perpetual; do not populate this just because a 10-Q "
                "mentions the series."
            ),
        ),
        ToolArg(
            name="stated_value", type="number", required=False,
            description="Updated per-share stated value.",
        ),
        ToolArg(
            name="liquidation_preference", type="number", required=False,
            description="Updated aggregate liquidation preference.",
        ),
        ToolArg(
            name="dividend_rate", type="number", required=False,
            description="Updated dividend rate (decimal).",
        ),
        ToolArg(
            name="principal_remaining", type="number", required=False,
            description=(
                "Updated CURRENTLY-OUTSTANDING aggregate face amount "
                "across the preferred series (USD). If the 10-Q says "
                "'as of period end, $5,500,000 of Series D remains "
                "outstanding', emit 5500000. The series' original "
                "aggregate face amount is captured at create time and "
                "doesn't move here.\n\n"
                "Prefer record_conversion / record_partial_redemption "
                "for known events. Use this field ONLY for anchor "
                "reconciliation against a periodic table when no "
                "discrete consuming event ties to the change."
            ),
        ),
    ),
)


# ─── amend_s1_offering ────────────────────────────────────────────────

amend_s1_offering = Tool(
    name="amend_s1_offering",
    description=(
        "Update an existing S-1 offering. Two main cases:\n"
        "  - S-1/A with a priced cover narrowing the anticipated_deal_size "
        "and fixing warrant_strike / warrant_coverage_pct.\n"
        "  - 424B4 final prospectus that prices the offering: set the "
        "final_* fields (final_deal_size, final_pricing, "
        "final_shares_offered, final_warrant_coverage_pct) from the "
        "cover's '$X per share' / 'aggregate offering price' / "
        "'shares offered' figures. Also set warrant_strike on the 424B4 "
        "if the cover discloses a cash-warrant strike that differs from "
        "any prior placeholder. ALWAYS set placement_agent_canonical "
        "when the cover names the placement agent / book-runner / lead "
        "underwriter — don't leave it null just because a prior call "
        "did. A 424B4 against an existing S-1 family is NOT a no-op "
        "redisclosure; it's the pricing event that closes the offering. "
        "Pair it with record_drawdown against the same s1_offering to "
        "book the proceeds, and do NOT call create_s1_offering — the "
        "offering already exists."
    ),
    mutation_kind="amend_instrument",
    instrument_type="s1_offering",
    event_kind=None,
    valid_forms=frozenset({
        "S-1", "S-1/A", "F-1", "F-1/A", "S-1MEF", "F-1MEF", "424B4",
    }),
    args=(
        _INSTRUMENT_ID_ARG,
        _EVENT_DATE_ARG,
        ToolArg(
            name="anticipated_deal_size", type="number", required=False,
            description="Updated expected gross proceeds.",
        ),
        ToolArg(
            name="warrant_strike", type="number", required=False,
            description=(
                "Cash strike of attached common warrants once priced "
                "(e.g. Series A / Series B common warrants on a 424B4). "
                "Do NOT use the $0.001 pre-funded warrant strike here — "
                "that's a sibling pre-funded tranche, not the cash "
                "warrant strike for the offering frame."
            ),
        ),
        ToolArg(
            name="warrant_coverage_pct", type="number", required=False,
            description=(
                "Updated warrant coverage as a decimal (1.0 = one "
                "warrant per share, 2.0 = two warrants per share, "
                "e.g. paired Series A + Series B)."
            ),
        ),
        ToolArg(
            name="sold_to_date", type="number", required=False,
            description="Cumulative gross proceeds sold.",
        ),
        ToolArg(
            name="placement_agent_canonical", type="string", required=False,
            description=(
                "Lead underwriter / placement agent / book-runner short "
                "canonical name (e.g. 'Dawson James', 'ThinkEquity', "
                "'Maxim'). Pull from the cover-page 'We have engaged "
                "X to act as our exclusive placement agent' or 'X is "
                "acting as the sole book-running manager' sentence."
            ),
        ),
        ToolArg(
            name="final_deal_size", type="number", required=False,
            description=(
                "Final gross proceeds at pricing in USD — shares × "
                "combined offering price on the 424B4 cover. Example: "
                "7,194,240 shares × $1.39 → 9999993."
            ),
        ),
        ToolArg(
            name="final_pricing", type="number", required=False,
            description=(
                "Combined public offering price per share on the 424B4 "
                "cover (e.g. 'The combined public offering price for "
                "each share of Common Stock and accompanying Common "
                "Warrants is $1.39' → 1.39)."
            ),
        ),
        ToolArg(
            name="final_shares_offered", type="number", required=False,
            description=(
                "Total shares offered at pricing — common shares plus "
                "any pre-funded warrant shares in lieu thereof. Example: "
                "2,437,340 common + 4,756,900 pre-funded = 7,194,240."
            ),
        ),
        ToolArg(
            name="final_warrant_coverage_pct", type="number", required=False,
            description=(
                "Final warrant coverage at pricing as a decimal (1.0 = "
                "one warrant per share). Paired Series A + Series B "
                "warrants both at 1× coverage → 2.0."
            ),
        ),
    ),
)


# ─── amend_equity ─────────────────────────────────────────────────────

amend_equity = Tool(
    name="amend_equity",
    description=(
        "Update an existing equity issuance. Rare — typically used "
        "when a later 13G / 13D discloses named holders that were "
        "anonymous in the original SPA / PIPE table."
    ),
    mutation_kind="amend_instrument",
    instrument_type="equity",
    event_kind=None,
    valid_forms=frozenset({
        "8-K", "8-K/A", "10-Q", "10-K", "10-Q/A", "10-K/A",
        "20-F", "40-F", "6-K",
    }),
    args=(
        _INSTRUMENT_ID_ARG,
        _EVENT_DATE_ARG,
        ToolArg(
            name="known_owners", type="array", required=False,
            items_type="string",
            description=(
                "Named holders disclosed in a LATER filing (resale "
                "S-1's selling-stockholder table, 13G/13D, 10-K "
                "subsequent-events footnote) when the original create "
                "captured only a generic counterparty. Short form, "
                "preserve the filing's exact entity tag ('Armistice', "
                "not 'Armistice Capital Master Fund Ltd.')."
            ),
        ),
    ),
)


__all__ = [
    "amend_atm", "restate_atm", "amend_equity_line", "amend_shelf",
    "amend_warrant", "amend_convertible", "amend_preferred",
    "amend_s1_offering", "amend_equity",
]
