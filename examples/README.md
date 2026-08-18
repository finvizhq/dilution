# Example ingest payloads

Real generated output of the producer — the exact JSON document
`PUT /v1/dilution/{ticker}` would carry, per
[`../FINVIZ_API_CONTRACT.md`](../FINVIZ_API_CONTRACT.md). Build a
front-end against these files; the §13 example in the contract is
abridged and hand-written, these are not.

Everything the producer ships is in here — cards, badge
strip, cash chart, O/S & potential-dilution chart, AI brief — except the
page header (live price, market cap, exchange, sector), which §1 leaves
to Finviz's own quote data.

Regenerate (needs a walked `dilution.db` plus Finviz Elite + SEC XBRL
network access):

```bash
python scripts/dump_finviz_payload.py GCTK FCEL XTIA CELU SCNI --out examples
python scripts/dump_finviz_payload.py --all --out examples       # whole universe
python scripts/dump_finviz_payload.py KSCP --stdout | jq .
```

Producer: `dilution/finviz_payload.py` — `build_payload(ticker)` returns
the wrapped body, `build_snapshot(ticker)` just the inner document.

## Why these five

Between them they cover every branch a consumer has to render:

| File | Exercises |
|------|-----------|
| `finviz_payload_GCTK.json` | baby-shelf-restricted microcap; six of seven card types (shelf, ATM, equity line, 10 warrants, convertible, priced S-1); fully-drawn ATM (`remaining_capacity: 0`, `used_pct: 100`); **negative** cash estimate and negative `months_of_cash` |
| `finviz_payload_FCEL.json` | large, unrestricted issuer (`is_baby_shelf_restricted: false`); both **unlimited-shelf** shapes — `unlimited: true` with a finite `total_shelf_capacity`, and one with both amounts `null`; 6 ATM cards; no warrants/converts at all (empty arrays) |
| `finviz_payload_XTIA.json` | two live shelves with 9-figure capacity; `$55M` equity line; a warrant with a `parent_shelf` sub-object and one fully-exercised (`remaining_outstanding: 0`) tranche; near-max badge score |
| `finviz_payload_CELU.json` | the long tail — 20 warrant cards, 13 of them carrying a `resale_registration` sub-object; preferred and convertible; **no shelf and no ATM at all** (two empty arrays) |
| `finviz_payload_SCNI.json` | FPI: non-null `os_chart.ads_ratio` (ordinary→ADS conversion), split-adjusted history; a **terminated** equity line alongside two live ones; preferred with a two-figure conversion price against a $0.50 close |

## Shape

Each file is the complete PUT body — a thin wrapper around the snapshot:

```json
{ "ticker": "CELU", "data": { "schema_version": 1, "ticker": "CELU", … } }
```

`data` is always present and always complete; everything described below
lives inside it. `ticker` is repeated inside `data` so a stored or
forwarded snapshot still identifies itself.

## Things worth knowing before you render

- **`null` ≠ `0`.** Filings frequently don't disclose a field. Render
  `null` as "—"; "$0 remaining" and "remaining unknown" are opposite
  claims. This is the single most important convention (§4).
- **No stable identifiers.** `source_ref` is reassigned on every re-walk.
  Don't key on it, store it, or put it in a URL — it exists so a data
  complaint can be traced back.
- **All seven `cards` keys are always present**; an empty category is
  `[]`. Array order is display order.
- **`company.cash` and `company.os_chart` may be absent**, and `badges`
  and `brief` may be `null`, when the upstream data isn't available.
  Everything else in the envelope is always there.
- **The `brief` is the one piece of generated prose** (§8) and the one
  piece written on a different clock: `brief.generated_at` lags the
  envelope's by days, and `brief.stale` says a filing has landed since.
  Cards win any disagreement.
- **Charts arrive plot-ready.** Don't derive, insert, or recompute a bar.
  The cash chart's last bar is `kind: "estimate"` (style it distinctly);
  the O/S chart's final fully-diluted bar is `latest.shares` as the base
  with `fd_stack` segments stacked in array order.
- **Numbers are raw** — full precision, raw USD, raw share counts,
  percentages on a 0–100 scale. Formatting is yours (open question #8).
- **`as_of` is a settled trading date**, not today: every market-derived
  number (`highest_60_day_close`, `os_chart.price_basis`, the price-based
  `fd_stack` segments) reflects that session's close, never the live
  price. `generated_at` is the build timestamp and is what the ordering
  guard keys on.
- **Money is never in thousands or millions**, including on cards whose
  source filing reported in thousands.

## Caveats on the values

These come from the current development ledger, so treat the **shapes**
as production-true and the **numbers** as a dev snapshot: they move with
each re-walk, and a few fields are still being reconciled against
DilutionTracker. Empty `known_owners`, a missing warrant expiration, or a
null `bank_tier` in these files are genuine extraction gaps — good test
cases for your null rendering, not bugs in the format.

**The `brief` blocks are mostly placeholders.** The real generator is an
LLM call that was returning 503s while these were dumped, and the prose
still cached from June contradicted the current cards (GCTK's claimed no
outstanding warrants while its own `cards.warrant` array holds ten). So
four of the five were dumped with `--dummy-brief`, which templates the
block from the snapshot's own numbers — correct shape, coherent with the
cards, and every one says so in its closing sentence.

`CELU` is the exception: its brief is genuine model prose, generated
today. Use that one to judge tone and length; use the other four to build
the panel. The brief regenerates inside the payload build itself when
the ledger changed, so re-dumping is enough once the generator is
healthy (delete the ticker's `dilution_ticker_brief` row first to force
fresh prose):

```bash
python scripts/dump_finviz_payload.py GCTK FCEL XTIA SCNI --out examples
```

Note that none of the five currently exercises `brief: null` — build
that state from the §8 spec, or point `--dummy-brief` at a ticker with
no cached brief.
