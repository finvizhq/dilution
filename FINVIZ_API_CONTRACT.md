# Finviz Dilution Data — Ingest API Contract (DRAFT v1)

Status: **draft for review** — nothing here is implemented on either side yet.
Producer: Peter's dilution pipeline. Consumer: Finviz.

---

## 1. Overview

The dilution pipeline extracts dilution instruments (shelves, ATMs, equity
lines, warrants, convertibles, preferreds, S-1 offerings) from SEC filings
into a ledger, then projects them into display-ready "cards", a dilution-risk
badge strip, and two charts (historical cash position; historical shares
outstanding + potential-dilution stack) — the same elements the internal
dashboard renders today. This contract defines how that data gets into
Finviz.

**Model: push, full-replace per ticker.**

```
pipeline (source of truth)                        Finviz
──────────────────────────                        ──────
EDGAR → walker → ledger
            │
   market data (settled closes,
   float, shares outstanding)
            │
            ▼
   card/badge/cash projection → push job ──HTTPS PUT──▶ ingest API → table → UI
```

Design principles the contract encodes:

1. **Finviz stores a rendering projection, not a database of record.** The
   pipeline can rebuild Finviz's entire table at any time with one
   full-universe push. Finviz never writes to it except via this API.
2. **One JSON document per ticker, replaced atomically.** No per-card
   updates, no diffs, no deletes of individual cards. A push is the complete
   new truth for that ticker; anything not in it is gone.
3. **No stable cross-push identifiers.** Internal instrument IDs are
   reassigned when the pipeline re-processes a ticker. The consumer must
   never join on, store, or build URLs from any ID in the payload.
4. **Data on the wire, never presentation.** No SVG, no HTML, no CSS class
   names. All business math (baby-shelf classification, IB6 remaining,
   badge scoring, runway estimates, dead-instrument filtering) happens
   producer-side; all visual rendering (charts, pills, tooltips, number
   formatting) happens consumer-side in Finviz's own UI stack.
5. **Unknown fields are ignored, never errors.** This is what lets the
   producer ship additive changes without coordination.

**Deliberately out of scope:** the page header (live price, market cap,
exchange, sector) — Finviz already owns that data natively and should render
it from its own systems.

---

## 2. Transport & authentication

- HTTPS only.
- Auth: static API key in `Authorization: Bearer <key>` (issued by Finviz,
  one key per environment). IP allowlisting optional on top.
- Content type: `application/json; charset=utf-8`.
- Compression: producer sends `Content-Encoding: gzip` on bodies > 16 KB;
  server must accept it.
- Two environments requested: **staging** and **production**, identical
  contract, separate keys.

---

## 3. Endpoints

### 3.1 `PUT /v1/dilution/tickers/{ticker}` — publish / replace snapshot

Body: the snapshot envelope (§4). Replaces the entire stored document for
`{ticker}` atomically — readers must never observe a half-applied snapshot.

Responses:

| Code | Meaning |
|------|---------|
| 200  | Stored (also for an idempotent re-send of the same `generated_at`). |
| 400  | Body failed envelope validation. Response body lists the violations. Producer treats this as a bug, not a retry case. |
| 401/403 | Bad/missing key. |
| 409  | `generated_at` is **older** than the currently stored snapshot's. Stored data unchanged. Protects against out-of-order retries rolling a ticker back. Producer treats as success-of-a-stale-write and moves on. |
| 413  | Body too large (limit: 2 MB — generous; real payloads are 10–100 KB). |
| 429  | Rate limited; `Retry-After` honored by producer. |
| 5xx  | Producer retries with exponential backoff + jitter (see §9). |

### 3.2 `DELETE /v1/dilution/tickers/{ticker}` — unpublish

Removes the ticker from the public projection immediately. Used when the
pipeline detects bad data for a ticker and wants it offline before a fix is
re-pushed, or when a ticker leaves the covered universe. 200 even if the
ticker wasn't present (idempotent).

### 3.3 `GET /v1/dilution/tickers/{ticker}` — read-back

Returns the stored envelope verbatim. Producer uses it to verify what is
actually live when debugging. Not a public/UI endpoint.

### 3.4 `GET /v1/dilution/tickers` — published index

Returns `[{ "ticker", "cik", "as_of", "generated_at", "schema_version" }]`
for every published ticker. Producer runs a nightly reconciliation: diff
this list against the intended universe, re-push or DELETE the drift.

---

## 4. Snapshot envelope

```jsonc
{
  "schema_version": 1,            // int; see §10 for evolution rules
  "ticker": "GCTK",               // upper-case; must match the URL path
  "cik": 1506983,                 // SEC CIK, integer
  "company_name": "GlucoTrack, Inc.",

  // The settled-close trading date all market-derived numbers reflect.
  // Filing-derived facts can be newer than this (intraday filing pushes).
  "as_of": "2026-06-02",

  // Producer-side build timestamp; strictly increasing per ticker.
  // The server's ordering guard (409) keys on this.
  "generated_at": "2026-06-02T21:10:43Z",

  "company": { ... },             // §5 — market context + cash position/chart
  "badges":  { ... },             // §6 — dilution-risk score strip
  "cards":   { ... }              // §7 — the seven instrument-card arrays
}
```

Conventions, everywhere in the payload:

| Kind | Encoding |
|------|----------|
| Money | raw USD as JSON number (`12500000`, not thousands/millions). |
| Share counts | raw count as JSON number. |
| Percentages | `0–100` scale number (`37.5` = 37.5%). |
| Dates | `"YYYY-MM-DD"` strings. |
| Timestamps | RFC 3339 UTC. |
| Unknown / not disclosed | `null`. Render as "—", never as 0. |
| Booleans | real JSON booleans (the producer normalizes its internal "Yes"/"No" strings before sending). |

`null` is semantically meaningful: SEC filings frequently don't disclose a
field (e.g. a warrant with no extracted expiration). A null must not be
rendered as zero — "$0 remaining" and "remaining unknown" are opposite
claims.

---

## 5. `company` block

Per-ticker market/fundamental context plus the two chart sections (cash
position §5.1, O/S & potential dilution §5.2). The market values also appear
on shelf cards (cards are deliberately self-contained).

```jsonc
"company": {
  "shares_outstanding": 3470000,        // XBRL-derived implied total (incl. Class B / units for Up-C)
  "float_shares": 2100000,              // sanity-bounded public float
  "highest_60_day_close": 2.18,         // settled closes only — never includes the in-progress session
  "price_to_exceed_baby_shelf": 35.71,  // close needed for float value ≥ $75M; null if float unknown
  "is_baby_shelf_restricted": true,     // implied common-equity value < $75M (SEC S-3 I.B.6)

  "cash":     { ... },                  // §5.1 — cash position & chart; omitted when XBRL cash unavailable
  "os_chart": { ... }                   // §5.2 — O/S & potential dilution; omitted when no O/S history
}
```

Notes:

- `highest_60_day_close` is the SEC I.B.6 basis (highest **closing** price
  of the preceding 60 trading days). It is intentionally *not* the live
  price and *not* max(close, live). It changes at most once per trading day.
- Any of these may be `null` for thinly-covered tickers; cards and badges
  that depend on them will carry nulls / `partial` flags accordingly.

### 5.1 `cash` — cash position & chart

```jsonc
"cash": {
  // ── summary facts (Finviz composes the sentence above the chart from these) ──
  "latest_period_end": "2026-03-31",
  "latest_reported_cash_usd": 4120000,   // last XBRL-reported cash & equivalents
  "op_cf_quarterly_usd": -2210000,       // latest quarterly operating cash flow; negative = burn
  "capital_raised_since_usd": 1200000,   // offerings/ATM draws since latest_period_end; null/0 = none
  "current_cash_est_usd": 3650000,       // reported − prorated burn + capital raised = the "today" estimate
  "months_of_cash": 5.6,                 // runway at current burn; may be negative; null when no burn
  "stale_days": 63,                      // days since latest_period_end
  "fx_failed": false,                    // true = some historical points dropped (FX conversion failure)

  // ── chart: explicit, plot-ready bar list (ascending, up to ~10 years) ──
  "chart": {
    "bars": [
      { "kind": "reported", "period_end": "2024-09-30", "fiscal": "2024 Q3", "form": "10-Q", "cash_usd": 7400000, "overlay_usd": null },
      { "kind": "reported", "period_end": "2024-12-31", "fiscal": "2024 FY", "form": "10-K", "cash_usd": 6000000, "overlay_usd": null },
      // ...
      { "kind": "estimate", "period_end": null, "fiscal": null, "form": null, "cash_usd": 3650000, "overlay_usd": 1200000 }
    ]
  }
}
```

**Rendering (consumer):** the bars arrive as explicit data points — the
consumer plots the array as-is and never derives, inserts, or recomputes a
bar:

- X axis: array order. Reported bars are labelled from `fiscal` /
  `period_end`; the `kind: "estimate"` bar (at most one, always last) is
  "today".
- Y axis: `cash_usd`. May be negative — render below the axis.
- `kind: "estimate"`: style distinctly — it's the producer's estimate
  (reported cash − prorated burn + raises), not a filing. Its `cash_usd`
  equals `current_cash_est_usd` by construction; the bar list is
  self-contained on purpose.
- `overlay_usd` (non-null on the estimate bar only): stacked segment =
  capital raised since the last report.
- Tooltip content, colors, axis formatting, and the summary sentence above
  the chart are Finviz's to compose — from the bar fields and the summary
  fact fields, in house style.

### 5.2 `os_chart` — shares outstanding & potential dilution

Historical split-adjusted shares-outstanding bars plus a final
fully-diluted bar: the current O/S as the base with potential-dilution
segments (warrants, converts, ATM capacity, …) stacked on top. Every
segment is derived from the cards in §7, so the chart is auditable against
the cards rendered below it.

```jsonc
"os_chart": {
  "ads_ratio": null,                    // FPI ADS conversion factor when applied; null for domestic issuers
  "price_basis": 2.18,                  // settled close used for price-based stack segments (as_of close)

  // ── historical O/S: quarterly, split-adjusted (ascending, up to ~10 years) ──
  "bars": [
    { "quarter_end": "2025-09-30", "shares": 2890000, "raw_shares": 2890000,
      "source_date": "2025-11-12", "form": "10-Q", "carried": false, "split_adjusted": true },
    { "quarter_end": "2025-12-31", "shares": 3120000, "raw_shares": 3120000,
      "source_date": "2026-03-30", "form": "10-K", "carried": false, "split_adjusted": false },
    { "quarter_end": "2026-03-31", "shares": 3120000, "raw_shares": 3120000,
      "source_date": "2026-03-30", "form": "10-K", "carried": true,  "split_adjusted": false }
    // ...
  ],

  // ── the final fully-diluted bar: current O/S base + stacked segments ──
  "latest": {
    "shares": 3470000,                  // = company.shares_outstanding (the base of the FD bar)
    "source": "10-Q XBRL, a/o 2026-05-12"  // provenance line (content — not derivable from other fields)
  },
  "fd_stack": [                         // display order: fixed-share paper first, price-based estimates on top
    { "key": "warrant", "label": "Warrants", "shares": 480000,
      "price_based": false, "capacity_usd": null,
      "note": "Remaining outstanding across 1 warrant card (pre-funded and placement-agent warrants excluded)" },
    { "key": "atm", "label": "ATM", "shares": 700000,
      "price_based": true, "capacity_usd": 1526000,
      "note": "$1.5M remaining ATM capacity ÷ $2.18 (I.B.6-capped where applicable)" }
    // ...
  ]
}
```

**Rendering (consumer):** plot the arrays as-is — no consumer-side
derivation:

- X axis: `quarter_end`, array order. Y axis: `shares`.
- `carried: true` = no filing that quarter, value carried forward from the
  prior one — typically styled ghosted/muted.
- The final (fully-diluted) bar = `latest.shares` base + `fd_stack`
  segments stacked in array order; one color per segment `key` (consumer
  palette), legend from `label` + `shares`.
- `price_based: true` segments were computed as `capacity_usd ÷
  price_basis` at the settled close. `capacity_usd` ships as hard data so
  these segments *could* be recomputed against the live price if open
  question #3 is answered yes; the snapshot values stay canonical.
- `note` is a per-segment provenance line (content, not derivable from
  other payload fields) — usable as tooltip body or caption; presentation
  is the consumer's.
- Segment `key` enum: `"warrant"` \| `"convertible"` \| `"preferred"` \|
  `"atm"` \| `"equity_line"` \| `"s1"` — additive (§10): render unknown
  keys with a fallback color rather than erroring.

---

## 6. `badges` block

The dilution-risk strip: one overall 0–100 composite plus the four drivers
it blends. All scoring happens producer-side; the consumer renders pills and
tooltips. Badges depend on price and cash, so they change daily — they ride
the same snapshot, no separate feed.

```jsonc
"badges": {
  "overall": {
    "score": 72,                  // 0–100; null when no driver was scorable
    "band": "high",               // "minimal"|"low"|"moderate"|"high"|"severe"|null
    "label": "High",              // display text for the band; "—" when null
    "partial": true,              // composite is missing ≥1 driver — render an asterisk
    "description": "0–100 composite of the four drivers: Offering Ability 30%, ...",
    "detail": [                   // ticker-specific tooltip lines, pre-formatted
      "Offering Ability 81 (weight 30%)",
      "Cash Need 88 (weight 30%)"
    ],
    "legend": [                   // band → meaning rows for the tooltip
      { "band": "severe",   "pill": "80–100", "meaning": "…" },
      { "band": "high",     "pill": "60–79",  "meaning": "…" }
      // ...
    ]
  },
  "drivers": [                    // render in array order
    {
      "key": "offering_ability",  // "offering_ability"|"overhang"|"history"|"cash_need"
      "label": "Offering Ability",
      "score": 81,                // 0–100; feeds the composite; null = unscorable
      "band": "high",             // "low"|"medium"|"high"|null
      "band_text": "High",        // display text; "—" when null
      "description": "…",         // static definition copy for the tooltip
      "detail": [                 // ticker-specific facts, pre-formatted
        "Active ATM: $1.5M raisable under I.B.6 cap",
        "Baby-shelf restricted (float value $4.6M)"
      ],
      "legend": [ { "band": "low", "pill": "Low", "meaning": "…" } /* … */ ]
    }
    // … overhang, history, cash_need
  ]
}
```

Contract decisions worth calling out:

- **`detail` lines arrive pre-formatted.** The driver formulas and their
  explanation wording iterate frequently producer-side; shipping finished
  strings means a formula change never needs a Finviz deploy. They are
  plain text (no markup), ≤ ~80 chars each, render as a bulleted list.
- **`description` and `legend` are static copy but shipped anyway** — they
  are tiny, and shipping them keeps tooltip copy editable without consumer
  releases.
- **The driver list is additive** (§10): a fifth driver appearing one day is
  not a breaking change; render whatever arrives, in order.
- The whole `badges` block may be `null` when nothing was computable.

---

## 7. `cards` block

```jsonc
"cards": {
  "shelf":        [ ... ],
  "atm":          [ ... ],
  "equity_line":  [ ... ],
  "warrant":      [ ... ],
  "convertible":  [ ... ],
  "preferred":    [ ... ],
  "s1_offering":  [ ... ]
}
```

All seven keys are always present; an empty category is `[]`. Array order is
the producer's display order (creation order) — preserve it.

### 7.0 Fields common to every card

| Field | Type | Notes |
|-------|------|-------|
| `source_ref` | string | Opaque producer-side debug handle (e.g. `"W-3034"`). **Not stable across pushes** — never store as a key, join on it, or expose it in a URL. Useful only when reporting a data problem back to the producer. |
| `title` | string | Display headline, e.g. `"November 2024 Dawson James S-1 Offering"`. |
| `registered` | string | Per-type label vocabulary (see each card type). Render verbatim. |
| `edgar_url` | string\|null | Deep link to the originating SEC document (or registration-family listing for shelves). |
| `last_update_date` | date\|null | Last filing that touched this instrument. |
| `bank_tier` | string\|null | `"bulge_bracket"` \| `"middle_market"` \| `"boutique"` \| `"pump_trifecta"` — classification of the placement agent. |
| `investor_class` | string\|null | `"long_term_informed"` \| `"eloc_funder"` \| `"toxic_lender"` \| `"pipe_flipper"` — classification of the counterparty. |

Two shared sub-objects:

```jsonc
"parent_shelf": {                  // nullable — the S-3 this program draws from
  "title": "March 2024 $100M Shelf",
  "file_number": "333-279901",
  "accession_number": "0001213900-24-053132",
  "edgar_url": "https://www.sec.gov/..."
}

"resale_registration": {           // nullable — S-1/S-3 registering underlying shares for resale
  "form": "S-1",
  "filing_date": "2024-12-20",
  "file_number": "333-284001",
  "accession_number": "0001213900-24-061002",
  "edgar_url": "https://www.sec.gov/..."
}
```

### 7.1 `shelf`

`registered` vocabulary: `Registered` · `Pending Effect`. (Expired/withdrawn
shelves are filtered producer-side and never pushed.)

| Field | Type | Notes |
|-------|------|-------|
| `shelf_status` | string\|null | Machine enum: `"active"` \| `"registered"` (pending effect). |
| `total_shelf_capacity` | money\|null | Registered dollar capacity. |
| `current_raisable_amount` | money\|null | What can still be raised **today**: baby-shelf-capped IB6 remaining when restricted, else capacity − raised. `null` when `unlimited` is true. |
| `unlimited` | bool | WKSI / pay-as-you-go shelf (S-3ASR, Rule 457(r)) — capacity is indeterminate. When true, render "Unlimited" for raisable amount. |
| `is_baby_shelf_restricted` | bool\|null | Mirrors `company.is_baby_shelf_restricted`. |
| `total_amount_raised` | money | Raised under this shelf incl. child ATM / S-1 / ELOC takedowns (file-number family rollup). |
| `raised_last_12mo_under_ib6` | money\|null | Trailing-12-month IB6-relevant raises. |
| `outstanding_shares` | number\|null | = `company.shares_outstanding`. |
| `float_shares` | number\|null | = `company.float_shares`. |
| `highest_60_day_close` | number\|null | = `company.highest_60_day_close`. |
| `price_to_exceed_baby_shelf` | number\|null | = `company.price_to_exceed_baby_shelf`. |
| `ib6_float_value` | money\|null | float × 60-day-high close (the strict I.B.6 test value). |
| `last_banker` | string\|null | Most recent takedown banker. |
| `effect_date` | date\|null | SEC EFFECT notice date. |
| `expiration_date` | date\|null | effect_date + 3 years (Rule 415(a)(5)). |

### 7.2 `atm`

`registered` vocabulary: `Registered` · `Replaced`.

| Field | Type | Notes |
|-------|------|-------|
| `parent_shelf` | object\|null | See §7.0. |
| `total_capacity` | money\|null | Program size per the sales agreement. |
| `remaining_capacity` | money\|null | Remaining, **after** the IB6 baby-shelf cap when the issuer is restricted. This is the headline number. |
| `remaining_without_baby_shelf` | money\|null | Remaining ignoring the IB6 cap (contractual remaining). |
| `limited_by_baby_shelf` | bool\|null | True when the IB6 cap sits below total program capacity. |
| `sales_total_usd` | money | Sold under the program to date. |
| `used_pct` | pct\|null | sales / capacity × 100. |
| `placement_agent` | string\|null | Sales agent (short form, e.g. `"H.C. Wainwright"`). |
| `agreement_start_date` | date\|null | |
| `agreement_end_date` | date\|null | |

### 7.3 `equity_line`

`registered` vocabulary: `Registered` · `Not Registered` · `Terminated`.

| Field | Type | Notes |
|-------|------|-------|
| `parent_shelf` | object\|null | |
| `total_capacity` | money\|null | |
| `remaining_capacity` | money\|null | No IB6 interaction on ELOCs. |
| `sales_total_usd` | money | |
| `used_pct` | pct\|null | |
| `counterparty` | string\|null | The ELOC funder (e.g. `"Lincoln Park"`, `"Yorkville"`). |
| `agreement_start_date` | date\|null | |
| `agreement_end_date` | date\|null | |
| `terminated` | bool | |

### 7.4 `warrant`

`registered` vocabulary: `Registered` · `Not Registered`.
Pre-funded warrants and placement-agent/underwriter compensation warrants
are filtered producer-side and never appear.

| Field | Type | Notes |
|-------|------|-------|
| `parent_shelf` | object\|null | |
| `resale_registration` | object\|null | See §7.0. |
| `total_issued` | number\|null | Tranche size at issuance (split-adjusted). |
| `remaining_outstanding` | number\|null | Unexercised count. May legitimately be `0` (fully-exercised tranche kept for context). |
| `exercise_price` | number\|null | Per-share strike, USD. |
| `known_owners` | string[] | Named holders (may be empty). |
| `underwriter` | string\|null | Placement agent, short form. |
| `issue_date` | date\|null | |
| `exercisable_date` | date\|null | |
| `expiration_date` | date\|null | |

### 7.5 `convertible`

`registered` vocabulary: `Registered` · `Not Registered`.

| Field | Type | Notes |
|-------|------|-------|
| `resale_registration` | object\|null | |
| `principal_total` | money\|null | Face at issuance. |
| `principal_remaining` | money\|null | |
| `conversion_price` | number\|null | Per-share, USD. Variable-rate notes carry the latest known value — display, don't compute on it. |
| `total_shares_issuable` | number\|null | principal_total / conversion_price (producer-computed). |
| `remaining_shares_issuable` | number\|null | principal_remaining / conversion_price. |
| `known_owners` | string[] | |
| `underwriter` | string\|null | |
| `issue_date` | date\|null | |
| `convertible_date` | date\|null | First conversion date. |
| `maturity_date` | date\|null | |

### 7.6 `preferred`

Identical field set to `convertible` (§7.5); `title` carries the series
(e.g. `"Series C Preferred"`). `principal_*` here means aggregate stated
value / liquidation preference.

### 7.7 `s1_offering`

`registered` vocabulary: `Pending` · `Effective` · `Priced`.
(Withdrawn / lapsed offerings are filtered producer-side.)

| Field | Type | Notes |
|-------|------|-------|
| `s1_status` | string\|null | Machine enum: `"pending"` \| `"effective"` \| `"priced"`. |
| `anticipated_deal_size` | money\|null | From the registration statement. |
| `final_deal_size` | money\|null | Once priced. |
| `final_pricing` | number\|null | Per-share offering price. |
| `final_shares_offered` | number\|null | |
| `warrant_coverage_pct` | pct\|null | Anticipated warrant coverage (100 = 1 warrant per share). |
| `final_warrant_coverage_pct` | pct\|null | |
| `exercise_price` | number\|null | Strike of attached warrants. |
| `underwriter` | string\|null | |
| `filing_date` | date\|null | |

---

## 8. Suggested consumer-side storage

The contract doesn't require any particular schema, but the natural minimum
is one row per ticker:

```sql
CREATE TABLE dilution_snapshots (
  ticker          TEXT PRIMARY KEY,
  cik             BIGINT NOT NULL,
  schema_version  INT NOT NULL,
  as_of           DATE NOT NULL,
  generated_at    TIMESTAMPTZ NOT NULL,   -- ordering guard (§3.1, 409)
  payload         JSONB NOT NULL,         -- the envelope, verbatim
  received_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Page render = one primary-key lookup + template over `payload`. If Finviz
later wants screener integration ("all tickers with an active ATM", "overall
dilution risk ≥ 60"), populate a derived table from `payload` on ingest —
that's a consumer-side projection and needs no contract change, **as long as
it tolerates fields being absent** (see §10).

---

## 9. Cadence, ordering, retries

**Push cadence (producer):**

1. **Daily batch** — after the settled close is available (~18:00 ET;
   exact time TBD), recompute market-derived fields (60-day-high-dependent
   card fields, badge scores, cash estimate proration) for the full
   universe and push every ticker. These change at most once per trading
   day by construction.
2. **Filing-driven** — within the EDGAR polling cycle, push only tickers
   whose ledger changed. Market inputs reuse the last settled close, so
   `as_of` stays at the prior trading date while filing facts are current.
3. **Pipeline releases** — full-universe re-push. May change every
   `source_ref` and reshape cards within a category. This is normal and is
   exactly why §1 principle 3 exists.

**Ordering:** `generated_at` is strictly increasing per ticker on the
producer side. The server rejects regressions with 409 (§3.1). Last write
wins; there is no merge.

**Retries (producer):** 5xx and network errors → exponential backoff with
jitter (1s base, ×2, cap 60s, give up after ~15 min and alert). 429 →
honor `Retry-After`. 400 → no retry, page the producer. A retried PUT is
safe by construction (idempotent full replace + ordering guard).

**Throughput expectations:** steady state is a trickle (filing-driven pushes
for a handful of tickers per cycle) plus one daily burst of the full
universe — O(few thousand) PUTs of 10–100 KB. Producer will run ≤8
concurrent requests during the burst; if that's too hot, tell us a rate and
we'll match it, or expose a bulk endpoint and we'll use it.

**Staleness (consumer):** display `as_of` ("data as of Jun 2 close") on the
page. Recommended: flag or hide a ticker whose `as_of` is more than 3
trading days old — that means the producer feed is down and the data is
drifting.

---

## 10. Schema evolution

- `schema_version` is a contract-breaking-change counter, starting at 1.
- **Additive changes do NOT bump it**: new fields, new card-type arrays, new
  badge drivers, new enum values in `bank_tier` / `investor_class` /
  `*_status`. The consumer must ignore unknown fields and render unknown
  enum values as null/absent.
- **Breaking changes DO bump it**: renaming/removing a field, changing a
  type or unit, changing the meaning of an existing field. Producer gives
  notice, consumer acks support for version N+1 before producer flips —
  during transition the producer can send N to prod and N+1 to staging.
- The consumer never needs a deploy for an additive change. That asymmetry
  is deliberate: the card and badge models are still evolving weekly on the
  producer side.

---

## 11. Failure & data-quality handling

- **Bad data discovered in a published ticker:** producer either pushes a
  corrected snapshot or `DELETE`s the ticker. Both take effect on the next
  page load. No consumer action needed.
- **Producer down:** nothing changes on Finviz; pages serve the last
  snapshot with an aging `as_of`. The §9 staleness rule is the safety net.
- **Consumer ingest down:** producer retries, then alerts and keeps state;
  next successful daily batch heals everything (full replace).
- **Reconciliation:** nightly, producer GETs the published index (§3.4) and
  diffs against its intended universe; re-push / DELETE any drift.
- **Ticker rename (same CIK):** producer PUTs the new ticker and DELETEs the
  old one in that order. CIK continuity is visible in the payload.
- **Contact:** data-content complaints (wrong numbers on a card/badge) go to
  the producer with `ticker` + `source_ref` (or badge `key`) +
  `generated_at`; ingest/transport issues go to Finviz infra.

---

## 12. Example payload (abridged)

```json
{
  "schema_version": 1,
  "ticker": "GCTK",
  "cik": 1506983,
  "company_name": "GlucoTrack, Inc.",
  "as_of": "2026-06-02",
  "generated_at": "2026-06-02T21:10:43Z",
  "company": {
    "shares_outstanding": 3470000,
    "float_shares": 2100000,
    "highest_60_day_close": 2.18,
    "price_to_exceed_baby_shelf": 35.71,
    "is_baby_shelf_restricted": true,
    "cash": {
      "latest_period_end": "2026-03-31",
      "latest_reported_cash_usd": 4120000,
      "op_cf_quarterly_usd": -2210000,
      "capital_raised_since_usd": 1200000,
      "current_cash_est_usd": 3650000,
      "months_of_cash": 5.6,
      "stale_days": 63,
      "fx_failed": false,
      "chart": {
        "bars": [
          { "kind": "reported", "period_end": "2025-09-30", "fiscal": "2025 Q3", "form": "10-Q", "cash_usd": 7400000, "overlay_usd": null },
          { "kind": "reported", "period_end": "2025-12-31", "fiscal": "2025 FY", "form": "10-K", "cash_usd": 6000000, "overlay_usd": null },
          { "kind": "reported", "period_end": "2026-03-31", "fiscal": "2026 Q1", "form": "10-Q", "cash_usd": 4120000, "overlay_usd": null },
          { "kind": "estimate", "period_end": null, "fiscal": null, "form": null, "cash_usd": 3650000, "overlay_usd": 1200000 }
        ]
      }
    },
    "os_chart": {
      "ads_ratio": null,
      "price_basis": 2.18,
      "bars": [
        { "quarter_end": "2025-12-31", "shares": 3120000, "raw_shares": 3120000, "source_date": "2026-03-30", "form": "10-K", "carried": false, "split_adjusted": false },
        { "quarter_end": "2026-03-31", "shares": 3470000, "raw_shares": 3470000, "source_date": "2026-05-12", "form": "10-Q", "carried": false, "split_adjusted": false }
      ],
      "latest": {
        "shares": 3470000,
        "source": "10-Q XBRL, a/o 2026-05-12"
      },
      "fd_stack": [
        { "key": "warrant", "label": "Warrants", "shares": 480000, "price_based": false, "capacity_usd": null,
          "note": "Remaining outstanding across 1 warrant card (pre-funded and placement-agent warrants excluded)" },
        { "key": "atm", "label": "ATM", "shares": 700000, "price_based": true, "capacity_usd": 1526000,
          "note": "$1.5M remaining ATM capacity ÷ $2.18 (I.B.6-capped where applicable)" }
      ]
    }
  },
  "badges": {
    "overall": {
      "score": 72,
      "band": "high",
      "label": "High",
      "partial": false,
      "description": "0–100 composite of the four drivers: Offering Ability 30%, Cash Need 30%, Overhang 25%, Dilution History 15%.",
      "detail": [
        "Offering Ability 81 (weight 30%)",
        "Cash Need 88 (weight 30%)",
        "Overhang 54 (weight 25%)",
        "Dilution History 60 (weight 15%)"
      ],
      "legend": [
        { "band": "severe",   "pill": "80–100", "meaning": "Imminent, large-scale dilution likely" },
        { "band": "high",     "pill": "60–79",  "meaning": "Strong dilution pressure" },
        { "band": "moderate", "pill": "40–59",  "meaning": "Meaningful but manageable" },
        { "band": "low",      "pill": "20–39",  "meaning": "Limited near-term risk" },
        { "band": "minimal",  "pill": "0–19",   "meaning": "Little dilution capability or need" }
      ]
    },
    "drivers": [
      {
        "key": "offering_ability",
        "label": "Offering Ability",
        "score": 81,
        "band": "high",
        "band_text": "High",
        "description": "How much the company can raise right now through live programs and shelf capacity.",
        "detail": [
          "Active ATM: $1.5M raisable under I.B.6 cap",
          "Baby-shelf restricted (float value $4.6M)"
        ],
        "legend": [
          { "band": "low",    "pill": "Low",    "meaning": "Little or no live capacity" },
          { "band": "medium", "pill": "Medium", "meaning": "Some capacity, constrained" },
          { "band": "high",   "pill": "High",   "meaning": "Large ready-to-use capacity" }
        ]
      }
    ]
  },
  "cards": {
    "shelf": [
      {
        "source_ref": "SH-012",
        "title": "March 2024 $100M Shelf",
        "registered": "Registered",
        "shelf_status": "active",
        "edgar_url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&filenum=333-279901&type=&dateb=&owner=include&count=100",
        "total_shelf_capacity": 100000000,
        "current_raisable_amount": 1526000,
        "unlimited": false,
        "is_baby_shelf_restricted": true,
        "total_amount_raised": 11420000,
        "raised_last_12mo_under_ib6": 980000,
        "outstanding_shares": 3470000,
        "float_shares": 2100000,
        "highest_60_day_close": 2.18,
        "price_to_exceed_baby_shelf": 35.71,
        "ib6_float_value": 4578000,
        "last_banker": "Dawson James",
        "bank_tier": "boutique",
        "investor_class": null,
        "effect_date": "2024-04-02",
        "expiration_date": "2027-04-02",
        "last_update_date": "2026-05-14"
      }
    ],
    "atm": [
      {
        "source_ref": "ATM-2183",
        "title": "June 2024 Maxim ATM",
        "registered": "Registered",
        "edgar_url": "https://www.sec.gov/Archives/edgar/data/1506983/000121390024053132/ea0207908-424b5_glucotrack.htm",
        "parent_shelf": {
          "title": "March 2024 $100M Shelf",
          "file_number": "333-279901",
          "accession_number": "0001213900-24-053132",
          "edgar_url": "https://www.sec.gov/..."
        },
        "total_capacity": 5000000,
        "remaining_capacity": 1526000,
        "remaining_without_baby_shelf": 2840000,
        "limited_by_baby_shelf": true,
        "sales_total_usd": 2160000,
        "used_pct": 43.2,
        "placement_agent": "Maxim",
        "bank_tier": "pump_trifecta",
        "investor_class": null,
        "agreement_start_date": "2024-06-11",
        "agreement_end_date": null,
        "last_update_date": "2026-05-14"
      }
    ],
    "equity_line": [],
    "warrant": [
      {
        "source_ref": "W-3034",
        "title": "November 2024 Warrants",
        "registered": "Registered",
        "edgar_url": "https://www.sec.gov/...",
        "parent_shelf": null,
        "resale_registration": {
          "form": "S-1",
          "filing_date": "2024-12-20",
          "file_number": "333-284001",
          "accession_number": "0001213900-24-061002",
          "edgar_url": "https://www.sec.gov/..."
        },
        "total_issued": 480000,
        "remaining_outstanding": 480000,
        "exercise_price": 1.20,
        "known_owners": ["Armistice Capital"],
        "underwriter": "Dawson James",
        "bank_tier": "boutique",
        "investor_class": "pipe_flipper",
        "issue_date": "2024-11-19",
        "exercisable_date": "2024-11-19",
        "expiration_date": "2029-11-19",
        "last_update_date": "2026-03-31"
      }
    ],
    "convertible": [],
    "preferred": [],
    "s1_offering": []
  }
}
```

---

## 13. Open questions for the Finviz team

1. **Screener integration** — wanted at launch, or later? If at launch, we
   should agree on the ~10 query-stable fields worth extracting into typed
   columns (e.g. has-active-ATM, total remaining raisable, baby-shelf flag,
   overall badge score). Blob-only keeps v1 trivially simple; the producer
   recommends blob-only first.
2. **Permalinks / per-card URLs** — does the UI need to link to an
   individual card? If yes we need a deterministic card identity
   (content-hash based, not `source_ref`), which is producer work — say so
   early.
3. **Live intraday fields** — should anything react to the live price (e.g.
   "raisable at current price", or the `price_based` O/S-chart segments
   recomputed as `capacity_usd ÷ live price`)? The payload already carries
   the hard inputs (`float_shares`, `price_to_exceed_baby_shelf`,
   capacities, `fd_stack[].capacity_usd`), so Finviz could compute
   display-only live deltas from its own quote feed; the daily snapshot
   stays canonical. Not in v1 unless wanted.
4. **Chart & badge rendering** — the contract ships plot-ready data (§5
   chart bars with X/Y values, §6 badge scores/bands); Finviz renders the
   bar chart and pill strip in its own charting/UI stack and owns all
   tooltip copy, colors, and number formatting. Confirm that's the
   preference vs. receiving pre-rendered SVG (not recommended: clashes with
   site theming and responsive layout).
5. **Universe** — initial ticker list size and admission rule (who decides a
   ticker is covered, producer or Finviz?).
6. **Rate limits / bulk endpoint** — is ≤8 concurrent PUTs in a nightly
   burst acceptable, or should we define `PUT /v1/dilution/bulk`?
7. **Daily batch timing** — earliest time Finviz's settled daily bars are
   final (the producer's 60-day-high basis must match Finviz's own close
   data to avoid visible discrepancies on the same site).
8. **Display rounding** — producer sends raw precision; Finviz owns
   formatting (M/B abbreviations, decimal places). Confirm.
