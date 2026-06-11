"""record_*, close_instrument, apply_split, note_no_event tools.

record_event tools encode partial-state mutations against existing
instruments (warrant exercise, conv conversion, ATM drawdown, partial
redemption / termination). close_instrument is the full-closure
sibling. apply_split is the global mutation that rewrites every
active warrant / convertible / preferred. note_no_event is the safety
valve that forces the model to commit to a REASON when a filing
contains nothing dilutive — the tool_choice='required' enforcement
needs a no-op tool to fall back on.

Drawdown proceeds are COMPUTED, not trusted from the LLM. The model
supplies drawdown_shares + a per-share GROSS price; the store computes
gross = shares × price (and avg_price = price). Empirically ~21% of
LLM-emitted (amount, shares, avg_price) triples drifted >5% from the
arithmetic identity, so neither the product nor avg_price is taken
from the model. drawdown_amount_usd remains as an aggregate-only
fallback for takedowns that state a total dollar figure with no
per-share price.
"""

from __future__ import annotations

from ._base import Tool, ToolArg, PROPOSED_ID_PATTERN


_INSTRUMENT_ID_ARG = ToolArg(
    name="instrument_id",
    type="string",
    required=True,
    min_length=3,
    description=(
        "Existing ledger row this event applies to. Use the id from "
        "the current ledger view; do not invent."
    ),
)

_EVENT_DATE_ARG = ToolArg(
    name="event_date",
    type="date",
    required=True,
    description="Date the event occurred (YYYY-MM-DD).",
)

# ─── record_exercise ──────────────────────────────────────────────────

record_exercise = Tool(
    name="record_exercise",
    description=(
        "Call when a warrant tranche is partially exercised (cashless "
        "or cash). `shares` is the share count exercised. `price` is "
        "the per-share strike paid (use the strike on the warrant; "
        "set to 0 for cashless exercise). For full exercise of the "
        "whole tranche, ALSO call close_instrument with "
        "reason='exercised' as a sibling — record_exercise just drops "
        "the outstanding count by `shares`.\n\n"
        "CASHLESS / NET-SHARE exercises retire MORE warrants than the "
        "shares they deliver ('76,376 warrants were automatically "
        "exercised on a cashless basis … converted into 26,070 shares "
        "of common stock'): pass the NET common shares delivered in "
        "`shares` and the SURRENDERED warrant count in "
        "`warrants_exercised` — the store decrements the outstanding "
        "count by warrants_exercised, not shares. Omitting it on a "
        "cashless full exercise leaves a phantom residual that blocks "
        "the close."
    ),
    mutation_kind="record_event",
    instrument_type=None,
    event_kind="exercise",
    # Warrant exercises are 8-K Item 3.02 events (or summarized in the
    # next periodic). A 424B prospectus supplement that mentions an
    # exercise is wrapping the underlying 8-K — the standalone 8-K is
    # the canonical source, so 424B3/424B5/SUPPL must NOT expose
    # record_exercise. Without this guard the LLM, faced with a 424B
    # wrapping an 8-K-style event (conversion, redemption, exercise)
    # and lacking the matching record_* tool, grabs record_exercise as
    # the closest semantic match — producing type_mismatch (exercise
    # on convertible/preferred) and, under tool_choice='required',
    # degenerating into runaway-loop responses (NUAI 0001213900-25-
    # 099168 emitted 906 record_exercise placeholders for this reason).
    valid_forms=frozenset({
        "8-K", "8-K/A", "10-Q", "10-K", "10-Q/A", "10-K/A",
        "20-F", "40-F", "6-K",
    }),
    args=(
        _INSTRUMENT_ID_ARG,
        ToolArg(
            name="shares", type="number", required=True, min_value=1.0,
            description="Share count exercised in this event.",
        ),
        _EVENT_DATE_ARG,
        ToolArg(
            name="price", type="number", required=False,
            description=(
                "Per-share strike actually paid. 0 for cashless. Omit "
                "when filing doesn't state — it equals the warrant's "
                "strike for cash exercises."
            ),
        ),
        ToolArg(
            name="gross_proceeds", type="number", required=False,
            description="Total cash received from the exercise.",
        ),
        ToolArg(
            name="warrants_exercised", type="number", required=False,
            min_value=1.0,
            description=(
                "Warrant count SURRENDERED in this event when it "
                "differs from the shares delivered (cashless / "
                "net-share exercise). The store decrements the "
                "outstanding warrant count by this value instead of "
                "`shares`. Omit for ordinary cash exercises where one "
                "warrant yields one share."
            ),
        ),
    ),
)


# ─── record_conversion ────────────────────────────────────────────────

record_conversion = Tool(
    name="record_conversion",
    description=(
        "Call when a convertible note or preferred series is partially "
        "or fully converted into common stock. `shares_issued` is the "
        "common-share count delivered to the holder. The OTHER input "
        "depends on the instrument type:\n"
        "  • Convertible NOTE → pass `principal_converted` (the face $ "
        "amount that converted); the store decrements principal_remaining.\n"
        "  • PREFERRED series → pass `preferred_shares_converted` (the "
        "count of preferred shares retired in this conversion); the "
        "store decrements `count`. principal_* fields are debt-shaped "
        "and do NOT move count on a preferred — passing a share count "
        "in principal_converted just buries it in principal_converted_"
        "to_date and leaves count stale.\n"
        "When the conversion fully retires the instrument, ALSO call "
        "close_instrument with reason='converted'."
    ),
    mutation_kind="record_event",
    instrument_type=None,
    event_kind="conversion",
    valid_forms=frozenset({
        "8-K", "8-K/A", "10-Q", "10-K", "10-Q/A", "10-K/A",
        "20-F", "40-F", "6-K",
    }),
    args=(
        _INSTRUMENT_ID_ARG,
        ToolArg(
            name="shares_issued", type="number", required=True,
            min_value=1.0,
            description="Common shares delivered to the holder.",
        ),
        ToolArg(
            name="principal_converted", type="number", required=False,
            min_value=0.01,
            description=(
                "Face $ amount of principal converted in this event. "
                "Required for convertible notes; omit for preferred "
                "(use preferred_shares_converted instead)."
            ),
        ),
        ToolArg(
            name="preferred_shares_converted", type="number", required=False,
            min_value=0.01,
            description=(
                "Count of PREFERRED shares retired in this conversion "
                "(not the common shares issued — that's `shares_issued`). "
                "Required for preferred conversions; omit for notes."
            ),
        ),
        _EVENT_DATE_ARG,
        ToolArg(
            name="principal_remaining", type="number", required=False,
            description=(
                "Principal left on the note after this conversion. Omit "
                "to let the apply layer subtract `principal_converted` "
                "from the existing outstanding. Notes only — has no "
                "effect on a preferred."
            ),
        ),
    ),
)


# ─── record_drawdown ──────────────────────────────────────────────────

record_drawdown = Tool(
    name="record_drawdown",
    description=(
        "Call when shares are sold under an existing ATM / equity-line / "
        "shelf / S-1 offering. Provide `drawdown_shares` (count "
        "delivered) and `price_per_share` (the per-share GROSS offering "
        "price). The store computes gross proceeds as shares × price — "
        "you never multiply, and avg_price is derived.\n\n"
        "PER-SHARE PRICE IS GROSS, NOT NET — use the stated offering "
        "price per share (e.g. '$10.74 per share'), the figure BEFORE "
        "the placement-agent commission. ATM/ELOC footnotes often lead "
        "with net: 'sold 414,785 shares at $10.74/share for net proceeds "
        "of $4,320K after fees' — pass price_per_share=10.74 (gross), "
        "NOT a net-derived per-share number. Cards and capacity math run "
        "in gross dollars, matching the instrument's `capacity_usd`.\n\n"
        "AGGREGATE FALLBACK — only when the filing discloses a total "
        "dollar figure with NO per-share price, omit price_per_share and "
        "pass the GROSS aggregate in `drawdown_amount_usd` (never the "
        "'net proceeds after deducting fees' number — that understates "
        "capacity drawn).\n\n"
        "DO NOT call this for: a new sales agreement (use create_atm "
        "or create_equity_line); a 424B5 that registers a primary "
        "shelf takedown WITHOUT actual sales yet (that's just "
        "registration, not a draw).\n\n"
        "PRE-FUNDED WARRANTS SOLD FOR CASH count toward the takedown: "
        "when a shelf offering sells common shares AND pre-funded "
        "warrants under the same prospectus supplement ('400,000 ADSs "
        "at $1.16 and pre-funded warrants to purchase 746,552 ADSs at "
        "$1.159'), the drawdown is the FULL offering gross (use the "
        "offering table's Total), not just the common-share line — "
        "the pre-funded cash was raised under the shelf too. Still "
        "create the pre-funded warrant instrument separately."
    ),
    mutation_kind="record_event",
    instrument_type=None,
    event_kind="drawdown",
    valid_forms=frozenset({
        "8-K", "8-K/A", "10-Q", "10-K", "10-Q/A", "10-K/A",
        "20-F", "40-F", "6-K",
        "424B5", "424B3", "424B4", "424B8", "SUPPL",
    }),
    args=(
        _INSTRUMENT_ID_ARG,
        ToolArg(
            name="drawdown_shares", type="number", required=True,
            min_value=1.0,
            description="Common shares delivered in this drawdown.",
        ),
        ToolArg(
            name="price_per_share", type="number", required=False,
            min_value=0.0,
            description=(
                "Per-share GROSS offering price in USD — the price the "
                "shares were sold at, BEFORE commission. PREFERRED input: "
                "the store computes gross = drawdown_shares × "
                "price_per_share, so you never do the multiplication. "
                "Provide this whenever the filing states a per-share "
                "price. Must be gross, not a net-of-fees per-share figure."
            ),
        ),
        ToolArg(
            name="drawdown_amount_usd", type="number", required=False,
            min_value=0.01,
            description=(
                "Aggregate GROSS dollar proceeds. FALLBACK ONLY — use "
                "when the filing gives a total dollar figure with no "
                "per-share price. Omit when you provide price_per_share. "
                "Must be gross (pre-commission), never the 'net proceeds "
                "after deducting fees' number."
            ),
        ),
        _EVENT_DATE_ARG,
        ToolArg(
            name="placement_agent_canonical", type="string", required=False,
            description=(
                "Sales agent / bank for this specific drawdown when "
                "the filing identifies a placement agent distinct from "
                "the parent ATM's banker (rare). Leave null on a "
                "one-off registered-direct / best-efforts SHELF "
                "takedown — the underwriter belongs to the offering's "
                "own card (s1/equity), not the shelf; stamping it here "
                "wrongly brands the whole shelf with that banker."
            ),
        ),
    ),
)


# ─── record_partial_redemption ────────────────────────────────────────

record_partial_redemption = Tool(
    name="record_partial_redemption",
    description=(
        "Call when an issuer redeems part of a convertible note or "
        "preferred series for cash (NOT converted into common — use "
        "record_conversion for that). `cash_paid` is the dollar outflow "
        "(typically the face/stated amount × 1.05 to 1.30 for the "
        "redemption premium). The OTHER input depends on the instrument "
        "type:\n"
        "  • Convertible NOTE → pass `principal_redeemed` (the face $ "
        "called back); the store decrements principal_remaining.\n"
        "  • PREFERRED series → pass `preferred_shares_redeemed` (the "
        "count of preferred shares retired); the store decrements "
        "`count`. principal_* fields are debt-shaped and do NOT move "
        "count on a preferred — passing a share count in "
        "principal_redeemed leaves count stale."
    ),
    mutation_kind="record_event",
    instrument_type=None,
    event_kind="partial_redemption",
    valid_forms=frozenset({
        "8-K", "8-K/A", "10-Q", "10-K", "10-Q/A", "10-K/A",
        "20-F", "40-F", "6-K",
    }),
    args=(
        _INSTRUMENT_ID_ARG,
        ToolArg(
            name="principal_redeemed", type="number", required=False,
            min_value=0.01,
            description=(
                "Face $ amount of principal redeemed in this event. "
                "Required for convertible notes; omit for preferred "
                "(use preferred_shares_redeemed instead)."
            ),
        ),
        ToolArg(
            name="preferred_shares_redeemed", type="number", required=False,
            min_value=0.01,
            description=(
                "Count of PREFERRED shares retired for cash in this "
                "event. Required for preferred redemptions; omit for "
                "notes."
            ),
        ),
        _EVENT_DATE_ARG,
        ToolArg(
            name="cash_paid", type="number", required=False,
            description="Cash outflow paid to the holder (includes premium).",
        ),
    ),
)


# ─── record_partial_termination ───────────────────────────────────────

record_partial_termination = Tool(
    name="record_partial_termination",
    description=(
        "Call when the issuer partially terminates an ATM, equity "
        "line, or shelf — reducing the cap without using it. Distinct "
        "from a drawdown (which sells shares against the cap) and "
        "from a full close_instrument (which marks the whole "
        "instrument terminated)."
    ),
    mutation_kind="record_event",
    instrument_type=None,
    event_kind="partial_termination",
    valid_forms=frozenset({
        "8-K", "8-K/A", "10-Q", "10-K", "10-Q/A", "10-K/A",
        "20-F", "40-F", "6-K",
    }),
    args=(
        _INSTRUMENT_ID_ARG,
        ToolArg(
            name="capacity_reduced_usd", type="number", required=True,
            min_value=0.01,
            description="Dollars of capacity removed from the cap.",
        ),
        _EVENT_DATE_ARG,
    ),
)


# ─── confirm_closing ──────────────────────────────────────────────────

confirm_closing = Tool(
    name="confirm_closing",
    description=(
        "Call when a CLOSING filing confirms the actual issuance of a "
        "previously-announced warrant / convertible / preferred / "
        "equity (PIPE common-stock) tranche already in the ledger. "
        "For an EQUITY target the apply layer books the closing cash "
        "(gross_proceeds_usd, else count × price) into the raise "
        "history — no relabel, no date-rebase. For the other types, "
        "atomically:\n"
        "  1. relabels the tranche by closing date — the apply layer "
        "sets issue_date and exercisable_date to closing_date and "
        "slides expiration by the same delta so the N-year term is "
        "preserved (e.g. 5-year warrant signed 2024-03-20 "
        "exp 2029-03-20 → closed 2024-08-14 exp 2029-08-14). Card's "
        "month-year label flips from 'March 2024 Warrants' to "
        "'August 2024 Warrants', matching DilutionTracker's "
        "closing-date convention.\n"
        "  2. records a 'closing' event on the instrument for audit.\n"
        "  3. optionally true-ups count if the closing filing's final "
        "number differs from the signing filing's anticipated number "
        "(overallotment exercised → count UP, tranche downsized at "
        "pricing → count DOWN).\n\n"
        "WHEN TO CALL — the signing-then-closing pair shows up most "
        "often as:\n"
        "  - FPI 6-K announce → 6-K close (Israeli or Chinese ADR "
        "issuers often need shareholder approval, e.g. for related-"
        "party PIPE participation, which delays the close by months).\n"
        "  - 8-K SPA announcement → 8-K closing (typical for US PIPEs "
        "where definitive agreement post-dates the term sheet).\n"
        "  - 424B3/424B4 pricing → 8-K closing (registered direct "
        "with separate closing notice).\n\n"
        "Filing-text cues — any one is sufficient when paired with a "
        "dedup hit on an open ledger tranche:\n"
        "  - 'closed the previously announced [PIPE / private "
        "placement / offering]'\n"
        "  - 'consummated its previously announced [...]'\n"
        "  - 'completed the closing of the [...]'\n"
        "  - 'issued the Units described in the [date] 6-K / 8-K'\n\n"
        "DEDUP CHECK before calling — the target tranche MUST already "
        "exist in the ledger. Match on (instrument_type, strike ±2%, "
        "created within ±180 days of THIS filing). If no match, the "
        "signing filing was missed upstream — call create_<type> "
        "instead with event_date=<closing_date> (same-filing create "
        "implicitly captures the closing).\n\n"
        "DO NOT use for: same-filing same-day signing+closing (the "
        "create's event_date already equals issuance; for an equity "
        "PIPE use create_equity(closing_date=...) instead); "
        "re-disclosure of an ALREADY-closed tranche in a later "
        "10-Q / 20-F (note_no_event); ATM / equity-line / shelf "
        "takedowns (record_drawdown); exercise / conversion of a "
        "closed tranche (record_exercise / record_conversion)."
    ),
    mutation_kind="record_event",
    instrument_type=None,           # validator routes by event_kind
    event_kind="closing",
    valid_forms=frozenset({
        "8-K", "8-K/A", "6-K", "6-K/A",
        "424B3", "424B4", "424B5", "SUPPL",
    }),
    args=(
        _INSTRUMENT_ID_ARG,
        ToolArg(
            name="closing_date",
            type="date",
            required=True,
            description=(
                "Actual issuance / delivery date of the tranche "
                "(YYYY-MM-DD) — typically the filing date for a "
                "same-day closing 6-K/8-K, or the explicit "
                "'On [date] the Company closed' date from the filing "
                "body. Becomes the new issue_date AND exercisable_date "
                "on the instrument; expiration shifts by the same "
                "delta to preserve the original term. (Equity targets: "
                "no date-rebase — stamps terms.closing_date and dates "
                "the cash booking.)"
            ),
        ),
        ToolArg(
            name="count_actual", type="number", required=False,
            min_value=1.0,
            description=(
                "Final issued count IF the closing filing states a "
                "different number than the announcement. Common "
                "reasons: overallotment exercised between signing and "
                "close (count UP), tranche downsized at pricing (count "
                "DOWN), some investors dropped (count DOWN). Omit when "
                "the closing filing re-states the signing count."
            ),
        ),
        ToolArg(
            name="gross_proceeds_usd", type="number", required=False,
            min_value=0.0,
            description=(
                "Cash received at closing, BEFORE placement-agent "
                "commissions (gross, not net). On an EQUITY (PIPE) "
                "close this is the amount booked into the raise "
                "history — state it whenever the filing discloses it. "
                "Informational on a warrant/convertible/preferred "
                "close — the associated equity issuance from a unit "
                "deal is emitted separately as create_equity. Omit "
                "for cashless / inducement closings where no new "
                "cash changes hands."
            ),
        ),
    ),
)


# ─── close_instrument ─────────────────────────────────────────────────

_CLOSE_REASONS = (
    "exercised",   # warrants fully exercised
    "converted",   # convertibles/preferreds fully converted
    "redeemed",    # paid off in cash
    "expired",     # past expiration without exercise/conversion
    "terminated",  # contract terminated (ATM agreement ends, etc.)
    "superseded",  # replaced by a successor instrument (requires replaced_by)
)


close_instrument = Tool(
    name="close_instrument",
    description=(
        "Call when an existing instrument is fully consumed. "
        "Use 'exercised' for warrants fully exercised, 'converted' "
        "for fully-converted notes/preferred, 'redeemed' for cash "
        "payoffs, 'expired' for past-expiration without action, "
        "'terminated' for contract terminations (e.g. Sales "
        "Agreement ends), 'superseded' when a successor instrument "
        "replaces this one (then ALSO set replaced_by). The store "
        "encodes successor relationships via the predecessor's "
        "history + the successor's proposed_id link."
    ),
    mutation_kind="close_instrument",
    instrument_type=None,
    event_kind=None,
    valid_forms=frozenset({
        "8-K", "8-K/A", "10-Q", "10-K", "10-Q/A", "10-K/A",
        "20-F", "40-F", "6-K",
        "S-3", "S-3/A", "S-3ASR", "F-3", "F-3/A", "F-3ASR",
        "F-10", "F-10/A", "F-10EF",
        "POS AM", "RW",
    }),
    args=(
        _INSTRUMENT_ID_ARG,
        ToolArg(
            name="reason", type="string", required=True,
            enum_values=_CLOSE_REASONS,
            description=(
                "Closure cause. The value picks the downstream status "
                "and gates which validators fire — getting it wrong is "
                "not cosmetic.\n"
                "  - 'exercised': warrant fully exercised. Pair with "
                "record_exercise booking the residual shares — "
                "record_exercise drops count, close_instrument flips "
                "status. Notes/preferred are NEVER 'exercised'.\n"
                "  - 'converted': convertible note or preferred series "
                "fully converted into common. Pair with record_conversion "
                "for the final tranche. Warrants are NEVER 'converted'.\n"
                "  - 'redeemed': paid off in cash (full redemption / "
                "maturity payoff). Distinct from 'converted' — no "
                "shares were issued, holder got cash.\n"
                "  - 'expired': the expiration / maturity date passed "
                "without exercise / conversion / redemption. Warrants "
                "and notes only. The filing's footnote will read "
                "'the warrants expired in <month>' or 'the note "
                "matured and was not converted'.\n"
                "  - 'terminated': contract / agreement ended without "
                "being fully consumed. Used for ATM agreements, "
                "equity-line / ELOC agreements, shelves explicitly "
                "withdrawn (RW filing), and warrant tranches "
                "explicitly cancelled by the issuer without exercise "
                "or expiry. Do NOT use 'terminated' as a catch-all "
                "for warrants whose footnote simply stops mentioning "
                "them — that's the anchor's job, not the walker's.\n"
                "  - 'superseded': replaced by a successor instrument "
                "(warrant exchange, note restructured into preferred, "
                "etc.). REQUIRES the sibling create_* to set "
                "proposed_id, and you must set replaced_by here to "
                "the same proposed_id."
            ),
        ),
        _EVENT_DATE_ARG,
        ToolArg(
            name="replaced_by", type="string", required=False,
            pattern=PROPOSED_ID_PATTERN,
            description=(
                "REQUIRED iff reason='superseded'. Must match the "
                "proposed_id on the sibling create_* mutation for the "
                "successor instrument."
            ),
        ),
    ),
)


# ─── apply_split ──────────────────────────────────────────────────────

apply_split = Tool(
    name="apply_split",
    description=(
        "Call when an issuer declares a stock split (forward or "
        "reverse) or an FPI changes its ADS ratio. The store derives "
        "the split ratio so the LLM never has to divide.\n\n"
        "TWO INPUT SHAPES — pass ONE, never both:\n"
        "  • For an ordinary common stock split, pass `post` + `pre` "
        "+ `direction`. For a 1-for-N REVERSE split emit post=1, "
        "pre=N. For an N-for-1 FORWARD split emit post=N, pre=1.\n"
        "  • For an FPI ADS-ratio change, pass `ads_ratio_from` + "
        "`ads_ratio_to` — the per-ADS underlying-share count BEFORE "
        "and AFTER the change. The parser derives post/pre/direction "
        "(e.g. 400 → 4,000 → reverse 1-for-10). `units` defaults to "
        "'ads' for this shape.\n\n"
        "The store rewrites count and per-share price fields on every "
        "active warrant / convertible / preferred denominated in "
        "matching `units`."
    ),
    mutation_kind="apply_split",
    instrument_type=None,
    event_kind=None,
    valid_forms=frozenset({
        "8-K", "8-K/A", "PRE 14A", "DEF 14A", "DEFA14A", "DEFM14A",
        "6-K", "6-K/A", "20-F", "40-F",
    }),
    args=(
        ToolArg(
            name="post", type="integer", required=False, min_value=1,
            description=(
                "POST-split numerator (common-split shape). For "
                "'1-for-60 reverse' emit post=1; for '4-for-1 "
                "forward' emit post=4. Omit when passing ads_ratio_*."
            ),
        ),
        ToolArg(
            name="pre", type="integer", required=False, min_value=1,
            description=(
                "PRE-split denominator (common-split shape). For "
                "'1-for-60 reverse' emit pre=60; for '4-for-1 "
                "forward' emit pre=1. Omit when passing ads_ratio_*."
            ),
        ),
        ToolArg(
            name="direction", type="string", required=False,
            enum_values=("forward", "reverse"),
            description=(
                "Direction (common-split shape). The store rejects "
                "post>=pre on reverse and post<=pre on forward. Omit "
                "when passing ads_ratio_* — derived from the ratio."
            ),
        ),
        ToolArg(
            name="ads_ratio_from", type="number", required=False,
            min_value=0.0001,
            description=(
                "ADS-ratio shape: ordinary shares represented by one "
                "ADS BEFORE the ratio change. E.g. for a filing "
                "stating 'increasing from 400 to 4,000', emit "
                "ads_ratio_from=400."
            ),
        ),
        ToolArg(
            name="ads_ratio_to", type="number", required=False,
            min_value=0.0001,
            description=(
                "ADS-ratio shape: ordinary shares represented by one "
                "ADS AFTER the ratio change. E.g. for a filing "
                "stating 'increasing from 400 to 4,000', emit "
                "ads_ratio_to=4000. Parser computes the implied "
                "post/pre/direction (4000>400 → reverse 1-for-10)."
            ),
        ),
        ToolArg(
            name="effective_date", type="date", required=True,
            description="Date the split takes effect (YYYY-MM-DD).",
        ),
        ToolArg(
            name="units", type="string", required=False,
            enum_values=("common", "ads"),
            description=(
                "Denomination of the split. Default 'common' for the "
                "post/pre shape, 'ads' for the ads_ratio_* shape. "
                "Use 'ads' only for FPI issuers whose ADS ratio "
                "changes — the walker applies the split only to "
                "instruments denominated in matching units."
            ),
        ),
    ),
)


# ─── note_no_event ────────────────────────────────────────────────────

note_no_event = Tool(
    name="note_no_event",
    description=(
        "Call when the filing contains NO dilutive instrument and NO "
        "dilutive event applicable to this issuer's ledger. The "
        "`reason` argument is a short justification (typically the "
        "filing's subject matter — e.g. 'earnings release', 'officer "
        "change', 'proxy materials only', 'auditor change'). Required "
        "under tool_choice='required' so the model commits to a "
        "rationale instead of returning prose or silence."
    ),
    mutation_kind="note_no_event",
    instrument_type=None,
    event_kind=None,
    # Every form is valid for note_no_event — it's the safety valve.
    # Walker exposes this on every form's tool list.
    valid_forms=frozenset(),
    args=(
        ToolArg(
            name="reason", type="string", required=True, min_length=10,
            description=(
                "Short rationale for emitting no other tool calls. "
                "What is this filing primarily about?"
            ),
        ),
    ),
)


__all__ = [
    "record_exercise", "record_conversion", "record_drawdown",
    "record_partial_redemption", "record_partial_termination",
    "confirm_closing",
    "close_instrument", "apply_split", "note_no_event",
]
