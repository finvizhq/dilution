# Eval-Defect Root-Cause Report — 2026-06-11

> ## IMPLEMENTATION OUTCOME (2026-06-11, appended)
> **Eval: 44→56 cards exact · 526/617 (85.3%) → 551/623 (88.0%) fields · 43→40 extras · pytest 4060 green · zero regressions.**
> Landed (all verified, no re-walk needed — fixture + render-time only):
> - **Fixture corrections** across 10 tickers (§3) — every value filing-verified; "asserts-to-fail" cases (FCEL Dec-2025 ATM=$0.5M, CETY C-1065=$384K, BNKK strike=$11.55) deliberately kept filing-true over walker-matching, per the codebase convention.
> - **`_warrant_dead` periodic-form guard** (cards.py): sibling-revival now skips warrants created off a 10-Q/10-K/20-F table → drops CETY W-5189 + ACTU W-5237 ghosts (−2 extras), keeps all 8 legitimate offering-form pre-funded pairs.
> - **`_fully_drawn_clamp`** (cards.py): a program with stored `remaining_capacity_usd==0` snaps a <0.5% float residual up to capacity, on both the ATM card and the shelf rollup → GCTK Dawson ATM + Sept-2024 shelf both flip exact (+3 fields, +2 cards).
>
> **NOT applied — the 3 proposed walk-time "low-risk" fixes are MIS-DIAGNOSED** (real stored data contradicts the report's fix-layer; verified case-by-case):
> - **#2 GCTK term-date** — the store's `_resolve_dates` math is *correct*. The Ballantyne warrants were created right (exercisable 2025-07-30, exp 2034-07-30) then **a spurious LLM amend on 2024-08-19** shifted both +10yr. An LLM amend-extraction error, not `parse.py`. No clean deterministic fix.
> - **#7 ACTU greenshoe** — `dilution_ledger_drawdowns` is **empty** for ACTU; the +$15M lives in the shelf row's stored `drawn_usd=32,250,008` (=$15M base + $17.25M with-over-allotment, both booked for one offering). `_drawdown_already_recorded` (proposed layer) never sees it. A walk-time shelf-drawn booking issue.
> - **#9 GCTK s1-anchor** — there are **two** s1 rows, both created from the *same* S-1/A (2024-11-08); needs registration-family date linkage to the original S-1, not a `create.py` anchor flip.
>
> These three need prompt work + a re-walk (nondeterministic), not localized edits — deferred as a separate, higher-risk effort. Lesson: the investigation verifiers confirmed each DEFECT, but did not validate the proposed FIX LAYER against stored data; always re-check the ledger/history before implementing a proposed fix.
>
> ### Follow-up (2026-06-11): Cluster D duplicate-instrument dedup
> Implemented the **closed-row resurrection guard** in `store._create_already_recorded` (+ `_CLOSED_REDISCLOSURE_WINDOW_DAYS=31`, `_RESURRECTION_GUARD_TYPES`): the dedup previously scanned only `status='active'`, so a periodic balance-sheet re-disclosure of an already redeemed/terminated/converted tranche spawned a NEW active duplicate (XTIA Series-9 P-447 resurrected the redeemed P-443). The guard now also matches CLOSED rows, on a STRONG key (series for preferred, strike+non-conflicting-expiration for warrant) within a TIGHT 31-day window — so a genuinely new same-letter issuance (XTIA re-uses Series 4/5 across years) never collapses onto an old tranche. **8 unit tests + full suite 4068 passed.**
>
> **Scope reality:** investigating the 14 "duplicate" extras against stored data showed they are NOT one cluster, and a *safe* dedup cleanly fixes only the XTIA-type phantom (1 eval extra). The rest are: active-row LLM strike/date inconsistencies (BNKK Jan-2023 re-disclosed at strike 35.0 vs original 32.62; QTEX/SCNI), no-sibling rows (GCTK W-5207), or stale-active-should-be-closed (Cluster H — KSCP's 4 converted preferreds, BNKK P-437). A dedup aggressive enough to catch those would over-merge distinct unit-offering tranches. So the report's "Cluster D clears ~14 extras" was over-stated; the guard is a real **DB-wide correctness fix** but its eval reach is 1 extra.
>
> **Re-walk outcome (XTIA):** the guard verifiably fired (2×) and eliminated the Series-9 phantom. BUT the verification re-walk DRIFTED: LLM nondeterminism + 2 deterministic fragile-payload-400 errors (8-K 0001213900-25-108855, [[gemini-fragile-payload-400]]) left reaping events unprocessed, adding 3 spurious extras (a $0 May-2026 ATM, an FC Imperial ELOC, a Feb-2024 warrant the morning walk had reaped). Net XTIA 3→5 extras; suite 40→42 extras, exact/fields flat (56 / 551-623 / 88%). The drift is transient (next full suite re-walk supersedes it); the clean expected state with the guard is XTIA 3→2, suite 39. Lesson reinforced: single-ticker --force re-walks are nondeterministic and can drift currently-passing state — prefer full-suite re-walks and accept the guard's value is realized on the next regular walk.
>
> ### Cluster H — KSCP close-on-conversion (IMPLEMENTED 2026-06-11, activates on next full re-walk)
> Root cause (verified): KSCP's 4 Nov-2023 preferreds (Series A/B/M/S) **automatically converted to common on 2024-05-15**. The Q2-2024 10-Q (acc 0001558370-24-012268) states verbatim: "the Automatic Conversion … As a result … there were **no shares of Preferred Stock outstanding** after the Preferred Stock Conversion Date." The walker read that 10-Q but the overhang extractor **matched the named series in the conversion narrative without setting `is_terminated`**, so the anchor's `is_terminated` auto-close never fired → 4 phantom-active preferreds (4 extras).
> **No safe render-time fix:** absence/staleness is ambiguous. SCNI's *live, in-fixture, 10-field-exact* EIB preferred (P-439) is **560 days** stale (issuer doesn't re-itemize it) vs KSCP's 630 — any last_seen / anchor-miss-count reaper that kills KSCP also kills SCNI's EIB. The only robust signal is the **affirmative "zero preferred outstanding"** statement, not absence.
> **Fix design (deterministic, walk-time):** a pure detector `_full_preferred_conversion(text)` gated on BOTH (a) automatic/mandatory conversion of preferred→common AND (b) an explicit "no shares of preferred stock … outstanding" affirmation (tight enough that per-series partial conversions and "convertible preferred" boilerplate don't match); when it fires on a periodic filing, close ALL active preferreds for the cik via `record_conversion`(preferred_shares_converted=count — required because validate rejects a `converted` close with principal_remaining≠0) + `close_instrument(reason='converted')`, wired as a deterministic per-filing pass beside `_reroute_*`/`_drop_*` in `walker._walk_async`.
> **Implemented (uncommitted):** `walker._full_preferred_conversion_date(text)` (two-gate detector: ZERO affirmation + conversion ACTUALITY; extracts the conversion date; \s handles EDGAR \xa0 nbsp) + `store.close_converted_preferred(cik, conversion_date, …)` (closes active preferreds with `created_at <= conversion_date` via `_apply_close(reason='converted')`, zeroing count; idempotent), wired into `walker._walk_async`'s periodic branch before `_anchor_one`. **12 unit tests + full suite 4080 passed.** Detector verified against KSCP's REAL cached Q2-2024 10-Q text → 2024-05-15; false-positives (partial per-series, boilerplate, conditional, zero-without-conversion, conversion-without-zero) all return None; the `created_at<=conversion_date` scope spares a post-conversion re-issuance.
> **Activation:** walk-time → takes effect on the next full re-walk (NOT done as a single-ticker re-walk, per the XTIA drift lesson). Current eval UNCHANGED until then; on the next full re-walk KSCP's 4 phantom preferreds close → KSCP 9→5 extras, suite toward ~35. DB-wide it also closes any other issuer's automatic-conversion phantoms. Verify suite-wide on that re-walk (a false-positive would need a filing that both affirms "no preferred outstanding" AND states an automatic conversion to common while live preferred exists — contradictory; the detector + created_at scope make it very unlikely, but the full re-walk is the integration check).
>
> ### Cluster A — BNKK preferred conv_price split (IMPLEMENTED 2026-06-12, activates on next full re-walk)
> The §A highest-leverage code lever, now landed (uncommitted) after a 7-agent grounding+verify workflow. **Shared `store._preferred_price_split_skip(terms)`** wired into BOTH split-handling sites (`_apply_split` and the amend-time `_rescale_stale_unit_amend` — the original single-site plan was insufficient: BNKK Series C's post-split amend re-quotes the raw 0.5582 and clobbered it back). Rule: `stated_value` always split-skipped; `conv_price`/`conversion_price` adjusted only when **no `conversion_ratio`** is stored (price-based ⇒ adjust; ratio-bearing ⇒ skip, protecting IQST D + BNKK legacy P-437). The proposed FCEL "band-guard" was a phantom (dropped). **19 unit tests + full suite 4099 passed.** Expected on next full re-walk: BNKK Series B 0.34→11.90, Series C 0.5582→19.54 + their as-converted share counts (~4 graded fields). Series A ($153.77) NOT recovered (walker stores no conv_price for it — separate extraction gap). Residual SCNI-style risk + a deferred preferred-COUNT split bug documented in the §A block below + `tests/KNOWN_ISSUES.md #14`.
>
> ---

# Eval-Defect Root-Cause Report — 2026-06-11

Walk: commit `70721d2` (dirty), all 14 fixtures re-walked 2026-06-11 11:47–13:43.
Method: 23-agent orchestrated investigation (1 deep root-cause + 1 adversarial verifier per
failing ticker + synthesis). Every verdict grounded in the cached primary-source filing text
in `dilution_raw`. CETY's verifier died on a socket error mid-run and was recovered + re-verified
separately (13/14 upheld). Reports: `logs/eval_reports/<TICKER>.txt`.

## 1. Headline

**State:** 3 clean (AACG, IQST, XTLB) · 11 failing · **526/617** field checks (85%) ·
**44/96** cards exact (46%) · **43 extras**.

**Takeaway:** the eval is **not bottlenecked on extraction quality** — it's bottlenecked on a
handful of *systemic store/cards bugs* plus *stale DT fixtures the walker is more current than*.
Highest-leverage **code** fix: the **preferred conv_price split-skip exemption (BNKK, ~9 fields)**.
Highest-leverage **non-code** action: **refresh stale fixtures** (CGEN, FCEL, ACTU, XTIA, VRM, CETY)
that postdate the DT screenshot — ~15+ fields and ~3 extras cleared at **zero regression risk**.
Of 43 extras, **~18 are "expected" (real instruments DT curates out)** — not bugs.

---

## 2. Systemic clusters (UPHELD findings, by impact)

### A — Preferred conv_price split-skip · BNKK · ~9 fields · **highest impact**
`store.py:1807-1812` hard-skips split adjustment of `conv_price/conversion_price/stated_value`
for `type=='preferred'` (the IQST §4(f) fixed-VWAP precedent). BNKK's Certificates of Designation
**say the conversion price adjusts for splits**, so the 1-for-35 reverse split is never applied →
every preferred conv_price is 35× too low, as-converted shares 35× too high (Series A/B/C).
**Proof:** the 2026-05-13 10-Q reports an *actual* Series B conversion 5,399 pref → 340,273 common
= effective $11.90 = $0.34 × 35 (not $0.34). The 10-Q narrative $0.5582 / $0.34 / $4.3935 are stale
pre-split boilerplate.
**Fix:** make the preferred split-skip **conditional** — apply split adjustment when the CoD/terms
say the conversion *price* adjusts for splits; keep the exemption scoped to fixed-ratio/fixed-VWAP
series so **IQST does not regress**. `store.py`.

> **CLUSTER A — IMPLEMENTED 2026-06-12 (uncommitted, walk-time; activates on next full re-walk).**
> Grounded via a 7-agent workflow (DB rows + verbatim CoD text + full regression surface + adversarial
> verify). Two corrections to the original plan, both forced by *re-walk* (not static-DB) semantics:
> (1) **There are TWO skip sites, not one** — `_apply_split` (the split divide, was line ~3462) **and**
> `_rescale_stale_unit_amend` (amend-time stale-unit normalization, was line ~1865). Fixing only the
> first left BNKK Series C's post-split 10-Q amend (re-quotes raw 0.5582) clobbering the divided 19.537
> straight back to 0.5582 → no fix. Both sites now share one helper so they can't drift (mirror-module
> lesson). (2) The "stated_value band-guard" the synthesis proposed for an alleged FCEL regression is a
> **phantom**: at FCEL's split event conv_price is the as-created 1692 (stated_value still null), so the
> guard never executes; 1692 ÷ (1/30) = 50760 = the filer's retroactively-adjusted value anyway. FCEL
> was never a regression — dropped the band-guard entirely.
>
> **Final design — shared `store._preferred_price_split_skip(terms)`:** `stated_value` ALWAYS skipped
> (liquidation face is split-invariant); `conv_price`/`conversion_price` split-adjusted for a preferred
> ONLY when **no `conversion_ratio`** is stored (price-based ⇒ adjust; ratio-bearing ⇒ the rate absorbs
> the split, conv_price is a fixed VWAP reference ⇒ skip — IQST D ratio 12.5, BNKK legacy NFH P-437
> ratio 130). The existing echo/distance machinery in `_rescale_stale_unit_amend` then handles **both**
> conventions correctly once the preferred exemption is lifted: BNKK (raw pre-split re-quote, off by
> exactly the split ratio) → echo-pinned to 19.54; FCEL (retro-adjusted post-split re-quote == current)
> → left as-is. CoD basis (verbatim): BNKK Series A §7(a) acc 0001641172-25-009075, Series B §7(a) acc
> 0001641172-25-018310, Series C §6(d) acc 0001641172-25-024802 all subject the Conversion Price to
> ×(shares before/after) on a combination; FCEL 10-K acc 0001104659-25-122302 "conversion prices …
> prior to the reverse stock split have been adjusted retroactively."
>
> **Blast radius (full DB):** 13 preferred rows have conv_price + an applied split; 7 are
> terminated/converted (excluded by `_apply_split`'s WHERE + the preferred-card status filter →
> eval-neutral). Of the 6 active: BNKK B (0.34→**11.90**) + C (0.5582→**19.54**) FIXED; IQST P-441/P-450
> + BNKK legacy P-437 protected by the ratio-guard (unchanged); FCEL P-434 neutral (50760 either way).
> **BNKK Series A ($153.77) is NOT recovered** — P-445 stores *no* conv_price at all (walker never
> extracted the Core 4 exchange price); separate walker-extraction gap, out of scope. **Net expected on
> next full re-walk: BNKK ~+2 fields (Series B/C conv_price + their as-converted share counts; ~4
> graded fields).**
>
> **Residual risk (documented, no current impact):** the ratio-absent discriminator is a proxy — a
> fixed-ratio series storing conv_price but no `conversion_ratio` (e.g. SCNI EIB P-439, conv 93.41 =
> $34,000/364) would be wrongly divided IF it ever took a split; none does today. Logged as
> `tests/KNOWN_ISSUES.md #14`; durable fix is for the walker to stamp `conversion_ratio` on those series.
>
> **Tests:** `tests/test_ledger_preferred_conv_price_split.py` — 19 tests pinning the helper + both call
> sites (BNKK B/C adjust, IQST/legacy skip, stated_value always skip, FCEL no-op, warrant unaffected,
> idempotency, pre-creation skip, the load-bearing site-#2 echo-pin). Full suite **4099 passed, 0
> failed, 0 xfail**.
>
> **Also surfaced (DEFERRED, out of scope):** a *preferred-count* split bug — `_apply_split`'s
> `_COUNT_FIELDS` loop multiplies a preferred SHARE count by the common-split ratio (no `type=='preferred'`
> guard); BNKK Series A count went 39,993 → 1,143 (÷35) via a unit_rescale, IQST Series A 10,000 → 127.
> A common split does not combine preferred shares. Symmetric companion to this fix; needs its own
> regression pass.

### B — Stale fixtures (walker is more current) · CGEN, FCEL, ACTU, XTIA, VRM, CETY · ~15 fields + ~3 extras · **zero code risk**
DT fixtures transcribed at an earlier snapshot than the walk (DB walked through 2026-06-08); the
walker correctly ingested later EFFECT notices / 10-Q subsequent-events / FY2025 10-K & 10-Q/A. See §3
for exact JSON edits. CETY's FY2025 10-K + 10-Q/A batch (filed 2026-06-05) postdates DT's 2026-06-02
screenshot — four CETY fixture fields are stale against it.

### C — Drawdown / balance-restate accounting · ACTU, GCTK, KSCP, SCNI, CETY · ~10 fields · medium-high
Shared theme (multiple mechanisms): the store mishandles period-fragment / aggregate / restatement
balances.
- **ACTU** shelf total +$15M greenshoe double-book → same instrument + same event_date + same price
  should keep **max, not sum**. `store.py _drawdown_already_recorded`.
- **GCTK** ATM $12,307 residual (2 fields) + shelf rollup inherits it (1 field) → honor stored
  `remaining_capacity_usd=0` / clamp residual to 0 at ≥99.8% used. `cards.py / capital_raised.py`
  (one fix clears both).
- **KSCP** Riley ELOC: single-period draw overwrites cumulative → accumulate across periods.
  `capital_raised.py`.
- **SCNI** YA II SEPA missed a $4.2M NOTE-11 advance (1 field); RK Stone ELOC full-draw misread as
  termination (3 fields) → recognize NOTE-11 "raised $X through drawdowns" + pre-funded-warrant ELOC
  settlement as draws. `walker_prompt.py / store.py`.

### C′ — CETY FY2025 periodic-batch reconciliation coverage gap · CETY · ~4 fields · high
*(re-verdict from the recovered CETY verifier — supersedes the original "stale_balance_veto" claim,
which was overturned: that veto fires correctly on a genuinely stale Q1 balance.)*
The walker processed the FY2025 10-K (`…027409`) / 10-Q/A (`…027406`) balance-restate + conversion
events for **some** notes but not others. C-1060 (Jan-2025 Mast Hill) never references `…027406` at
all → its 4 Mast Hill conversion notices were never recorded as `converted` events, so
`cards._effective_conv_price` fell back to discount×market (0.6471) instead of the realized $2.86, and
the $702,581 year-end balance was never picked up (card stuck at $416,452). C-1065 likewise missed the
10-K restate to $384,000 (siblings C-1062/C-1069 *were* updated).
**Fix:** anchor the periodic balance-restate + conversion-extraction reconciliation by
**owner + issue-date** so it covers **every** note in the batch, not a subset. `anchor.py / store.py`
+ `tools/record.py` (record the per-note conversion notices for **all** counterparties).

### D — Duplicate instrument / restate-from-balance · BNKK, GCTK, QTEX, SCNI, XTIA, CETY · ~10 extras + ~6 fields · high
Periodic filings (10-K/10-Q/20-F balance-sheet & warrant-reconciliation tables) re-emit an
already-tracked instrument as a **new** row, or an amend binds the **wrong** table row.
- BNKK Jan-2023 ×2 (10-K re-create) + SCC double-book; GCTK W-5207 (June-2024 re-create) + S1-203
  ($30M fee-table aggregate); QTEX W-5115/W-5141 (20-F NOTE-11 table-row re-create — also drives the
  July-2021 count/underwriter mismatches); SCNI W-5132/W-5133 inducement double-count (drives title +
  expiration mismatch + 1 extra); XTIA P-447 Series 9 / P-449 Series 5 phantoms.
- **CETY W-5190** total_issued 133,333 vs 2,894: a 10-K/A amend **mis-bound a 2,000,000-share
  March-2024 subscription warrant onto the Aug-2022 Jefferson warrant** (cross-instrument
  contamination); the same 10-K/A line states the Jefferson warrant = 43,403 shares.
**Fix:** dedup/restate keyed on **(strike, expiration)** for warrants and **(series_letter, issue_date)**
for preferred; never create a row whose key already maps to a live instrument; a count restatement
binds only when the disclosed share figure/date is consistent with the existing instrument; preserve
the original create's `initial_count` as `total_issued`. `store.py` + `anchor.py`.
**Caution (XTIA):** series_letter-alone would wrongly merge XTIA's three Series-4 / two Series-5 — use
the composite key, or prefer the lower-risk render-time residual/zero-carrying-value floor.

### E — known_owners empty · BNKK, QTEX, SCNI · ~3 fields · **needs-design**
Resale F-3/F-1 selling-shareholder tables are hard-skipped at the resale-classification gate before
the LLM known_owners back-fill can fire. SCNI Dec-2023 + Sep-2023 (Armistice/Sabby/Bigger/District 2)
and QTEX Dec-2024 (5 holders) are named **only** in the linked resale prospectus. (BNKK Jan-2023 owners
are **not** recoverable — names live only in litigation prose / DT's cap-table, not warrant terms.)
**Fix:** route resale prospectuses to a **known_owners-only amend pass** when they name selling
shareholders for ledger instruments with empty owners — scoped so **no `create_*` tools are exposed**
on resale filings (else phantom-create regression). `item_classification.py / walker resale gate`.

### F — Split-adjust (warrant, non-BNKK) · GCTK · ~2 fields
Series B strike $2,172 vs $36.20 (one split under-applied despite both in `applied_splits`); July-2024
strike $5,940 vs $5,490 (a 10-K **digit-transposition typo** "$5,490.00" overwrote the split-correct
value). **Fix:** store guard — an amend whose strike is inconsistent with the recorded split chain
re-applies outstanding splits or rejects the stale/typo value; sibling-consistency for co-issued
Series A/B. `store.py / splits.py`. *(July-2024 is partly a faithful-but-wrong extraction of a filing
typo — see §7.)*

### G — Warrant term/date math · GCTK · 6 fields
July-2024 Ballantyne ×3: "exercisable N months after issuance, term M years" wrongly computed as
exercisable = issue + M and expiration = exercisable + M (double-stacked the 10-yr term → exercisable
2034 instead of 2025, expiration 2044 instead of 2034). **Fix:** exercisable = issue + N months;
expiration = issue + M years. `tools/parse.py`. Flips all three cards exact.

### G′ — Warrant exercisability-trigger extraction · CETY · 1 field
CETY Aug-2022 Jefferson: filing says "exercisable on or after February 2, 2023"; walker defaulted
`exercisable_date` to the issue date (2022-08-12). **Fix:** when a warrant states a future
exercisability trigger, capture it rather than defaulting to issue_date.
`walker_prompt.py / tools/create.py`.

### H — Walker over-closes / misses conversion · KSCP, GCTK, BNKK · ~5 extras + 2 fields
KSCP 4 Nov-2023 preferreds converted May-15-2024 but never processed (still active → 4 extras); GCTK
Sept-2024 spurious count-amends fabricating exercises (2 fields); BNKK P-437 stale 2021 Series B;
BNKK S1-205 resale-S-1 misclassified as primary offering. **Fix:** process explicit
conversion/redemption disclosures to close; reconcile against stated totals; classify resale-only S-1
as non-offering. `anchor.py / store.py / item_classification.py`.

### I — Shelf-family rollup linkage · FCEL · 1 field (code + fixture)
FCEL Oct-2023 shelf total drops the ~$200M Dec-2025 ATM (created from an 8-K with NULL
`registration_accession`; `_shelf_family_drawn` lacks the same-day-companion-424B5 fallback that
`_parent_shelf` has). **Fix:** mirror the `_parent_shelf` companion-424B5 file_number fallback into
`_shelf_family_drawn`. `cards.py:1962`. (Fixture is *also* stale — re-pin to ~$498.99M after the code
fix.)

### J — Cards render: `_warrant_dead` sibling-keep too loose · ACTU, CETY · 1 extra each
A fully-exercised count=0 warrant is resurrected when a sibling with the **same `created_accession`**
is active. CETY W-5189 (FirstFire, dead) is revived because the unrelated Jefferson W-5190 shares the
same source 10-Q accession. ACTU W-5237 (net-exercised, $5.27) is similarly kept.
**Fix:** restrict sibling-revival to genuine same-offering pairs (same SPA/8-K event, or matching
counterparty + issue_date), not any instrument sharing a periodic-filing accession.
`cards.py _warrant_dead ~640-650`. **Verifier caveat (ACTU):** the originally-proposed
"same issue_date AND strike family" predicate would **not** drop the ACTU ghost — needs the
same-offering-event predicate.

### K — Shelf last_banker rollup · SCNI · 1 field
SCNI Aug-2023 shelf `last_banker` None: a Wainwright takedown was recorded as a direct/warrant
offering with NULL `drawdown_party` fields; `_last_banker_for_shelf` only joins ATM/s1 siblings.
**Fix:** capture takedown banker into drawdown_party, or broaden `_last_banker_for_shelf` to consider
`placement_agent` on direct/warrant takedowns. `store.py / cards.py`.

---

## 3. Fixture corrections (upheld `fixture_error` — edit the JSON)

| Ticker | Card | Field | Current | Correct | Filing cite |
|---|---|---|---|---|---|
| ACTU | Mar-2025 Riley equity_line | remaining_capacity | 46,254,494 | **46,199,535** (drawn $3,800,465) | FY2025 10-K 0001683168-26-002257 "net proceeds of $3,800,465" |
| CETY | Mar-2025 warrant | title | "March 2025" | **"February 2025"** | 8-K 0001493152-25-008949 "closed on February 28, 2025"; warrants dated Feb 27, 2025 |
| CETY | Jan-2025 warrant | remaining_outstanding | 672,309 | **add to `snapshot_fields`** (DT post-ratchet snapshot; = 2,132,596 − 1,460,287 exercised, not walker-derivable) | 8-K 0001493152-25-025780 (cashless exercises) |
| CETY | Apr-2025 conv (C-1062) | principal_remaining | 49,835 | **188,558** | FY2025 10-K 0001493152-26-027409 "balance…as of December 31, 2025, was $188,558" |
| CETY | Jul-2025 FirstFire conv (C-1069) | principal_remaining | 83,572 | **120,750** | FY2025 10-K …027409 "balance…as of December 31, 2025, was $120,750" |
| CETY | Apr-2025 conv (C-1065) | principal_remaining | 256,000 | **384,000** (re-pin *after* code coverage fix; neither side currently right) | FY2025 10-K …027409 "balance…as of December 31, 2025, was $384,000" |
| CGEN | May-2026 Shelf (333-295989) | effect_date / registered / expiration / last_banker | null / "Pending Effect" / 2029-05-18 / null | **2026-06-08 / Registered / 2029-06-08 / Leerink** | EFFECT 9999999995-26-001920 ("Effectiveness Date: June 8, 2026"); F-3 0001178913-26-002774 |
| FCEL | Dec-2025 Jefferies ATM | remaining_capacity / …without_baby_shelf | 154,799,000 | **~500,000 (or 0)** | 10-Q 0001104659-26-071183 "~$0.5 million…remained available" |
| FCEL | Oct-2023 Shelf (333-274971) | total_amount_raised | 368,779,594.92 | **~498,990,000** (re-pin after rollup fix §I) | file-scoped cumulative 95.1M+203.8M+200.07M |
| FCEL | June-2026 Shelf (333-296607) | (absent) | **add** (optional) | S-3ASR 0001104659-26-071419 (WKSI, filed 2026-06-08) |
| GCTK | Nov-2024 Series A warrant | expiration_date | 2029-11-14 | **2030-01-03** | 424B4 0001493152-24-045365 (5yr from Jan-3-2025 stockholder-approval IEAD) |
| GCTK | Nov-2024 Series B warrant | expiration_date | 2027-05-14 | **2027-07-03** | same 424B4 (2.5yr from Jan-3-2025 IEAD) |
| KSCP | Apr-2025 Shelf (333-286404) | total_amount_raised / current_raisable | 44,367,186 / 55,632,814 | **31,700,000 / 68,300,000** | 2026 10-Q 0001104659-26-062505 "$18.3 million remaining" on the $50M Jul-2025 supplement; DT conflated old shelf 333-269493 |
| KSCP | Jun-2024 Wainwright ATM | remaining_capacity | 1,347,000 | **not filing-supportable** (1,347,000 = the *Oct-2024* supplement cap mis-keyed) — correct/drop | Jun-2024 424B5 $11.66M; Oct-2024 424B5 $1.347M |
| QTEX | Jul-2021 warrant | known_owners | [Armistice, Sabby, Warberg] | **remove** (0 hits in any QTEX filing; firm-commitment IPO names no buyers) | 424B4 0001213900-21-037050 |
| VRM | Aug-2025 convertible | maturity_date | 2030-08-29 | **2030-10-01** | 10-Q 0001193125-25-274273 "will mature on October 1, 2030" |
| XTIA | Mar-2022 Series 8 preferred | (whole entry MISSING) | listed (out=0) | **drop** (walker correctly hides terminated/redeemed) | 8-K 0001213900-22-065092 (terminated 2022-09-30) |
| XTIA | May-2024 Shelf | total_amount_raised / current_raisable | 43,700,032 / 306,299,968 | **~49,000,040 / ~300,999,961** *(med conf)* | 8-K 0001213900-25-006029 ("$20,000,000 @ $13.75" + "~$25M ATM") |
| BNKK | Jan-2025 warrant | exercise_price | 11.9 | **15.218** *(med conf)* | 8-K 0001493152-25-003643 ($0.4348×35); $11.90 = Series B conv cross-contamination |
| BNKK | May-2021 Note Warrant | (whole card MISSING) | listed live | **drop / mark expired** (5-yr term → expired 2026-05-11) | 8-K 0001264931-21-000049 |

> NOTE — BNKK Jan-2025 `known_owners` was **overturned to a walker bug** (not a fixture error): the
> warrant went to Bigger Capital (EX-4.1); the walker over-propagated the *note's* Trajan/Fried
> assignment onto it. Fixture `['Bigger Capital']` is correct.

---

## 4. Scorer / matcher artifacts (run_eval.py)

| Ticker | Defect | Mechanism | Fix |
|---|---|---|---|
| CGEN | May-2026 shelf MISSING+EXTRA (same SH-2596) | shelf keyed on `effect_date`; the null-date fixture fallback (`run_eval.py:305-308`) **skips dated actuals**, so a null fixture can't bind a now-dated actual | **Fixture refresh (§3) is the primary fix.** Optional/risky: let a unique-title null-date fixture bind a sole dated candidate (verifier flags this masks real misses — prefer fixture). |
| GCTK | Sept-2024 S-1 MISSING + Nov-2024 priced S1-204 EXTRA | s1_offering keyed on `filing_date ±14d`; walker anchored S1-204 to the S-1/A (2024-11-08), 53d after the original S-1 (2024-09-16) | Anchor s1_offering `filing_date` to the **original S-1**, not the latest S-1/A (`tools/create.py`); recovers 1 found card, removes 1 extra, ~7 fields rebind. |
| BNKK | Apr-2022 twin warrants (Greentree EXTRA) | matcher bound the single two-owner fixture card to the L&H twin by date proximity (0d vs 2d), leaving Greentree extra | **Originally-proposed owner-subset tie-break REFUTED** (both twins are owner-equal). Real cause = fixture combines two real twins into one card. Split the fixture or accept. |

---

## 5. Expected extras — NO action (real instruments DT curates out)

| Ticker | # | Instruments |
|---|---|---|
| ACTU | 2 | $10.55 W-5238 (Series B) + $9.42 W-5243 (Series C) — both live per Q1-2026 10-Q |
| BNKK | 3 | Jul-2021 W-5120 ($500M Aegis) + Apr-2025 W-5187 (Core 4) + Aug-2023 W-5155 ($43.05 PP) |
| CETY | 4 | Nov-2022 W-5196 + Mar-2023 W-5206 (Mast Hill warrants) + May/Jul-2025 1800 Diagonal convert-on-default notes C-1066/C-1070 |
| FCEL | 4 | Jun-2020 / Jun-2021 / Jul-2022 Jefferies ATM chapters (dead predecessors) + Jun-2026 S-3ASR |
| KSCP | 5 | May-2022 warrants ×2 (distinct cohorts) + Aug-2023/Apr-2024/Oct-2024 Wainwright ATM supplements (re-registrations of the one Feb-2023 agreement) |
| XTIA | 1 | Jul-2023 W-5162 (pre-merger private XTI warrant) |
| VRM | 1 | Dec-2023 Virtu ATM ($50M, expired shelf) — DT itself displays it flagged "Shelf Expired"; do **not** reap |

**~18 of 43 extras are expected.** The remaining ~25 are addressed by clusters D/H/J.

---

## 6. Prioritized fix plan

### Code fixes (ranked by impact / risk)
| # | Fix | Layer | Tickers | Fields/extras | Risk | Conf |
|---|---|---|---|---|---|---|
| 1 | ✅ **DONE 2026-06-12** Conditional preferred conv_price split — shared `_preferred_price_split_skip`, BOTH sites (`_apply_split` + `_rescale_stale_unit_amend`); adjust when no `conversion_ratio` | store.py | BNKK | ~4 fields (B/C; Series A needs extraction) | Med (keep conditional) | High |
| 2 | Warrant term/date math (exercisable=issue+N mo, exp=issue+M yr) | tools/parse.py | GCTK | 6 fields | Low | High |
| 3 | Preferred/warrant dedup on composite key + restate-not-create + count-consistency on amend | store.py, anchor.py, cards.py | BNKK, GCTK, QTEX, SCNI, XTIA, CETY | ~10 extras + 6 fields | Med (XTIA same-letter series) | High |
| 4 | CETY FY2025 periodic-batch coverage: reconcile balance-restate + conversions over **every** note (owner+issue-date) | anchor.py, store.py, tools/record.py | CETY | ~4 fields (+ C-1065 w/ fixture) | Med | High |
| 5 | Recognize NOTE-11 advances + pre-funded-warrant ELOC settlement as drawdowns | walker_prompt.py, store.py | SCNI | 4 fields | Low-Med | High |
| 6 | Honor stored remaining_capacity_usd=0 / clamp residual at ≥99.8% used | cards.py, capital_raised.py | GCTK | 3 fields | Low | High |
| 7 | Greenshoe restatement dedup (same instr+date+price ⇒ max, not add) | store.py _drawdown_already_recorded | ACTU | 1 field | Low | High |
| 8 | Process automatic-conversion to close converted preferred; reconcile to totals | anchor.py, store.py | KSCP, BNKK | 4+ extras | Med | High |
| 9 | s1_offering anchor to original S-1 (not S-1/A) + resale-S-1 = non-offering | tools/create.py, item_classification.py | GCTK, BNKK | 1 found + ~7 rebind + 1 extra | Low-Med | High |
| 10 | `_warrant_dead` sibling-keep → same-offering-event predicate (not shared accession) | cards.py 640-650 | ACTU, CETY | 2 extras | Low | High |
| 11 | Warrant exercisability-trigger extraction (don't default to issue_date) | walker_prompt.py, tools/create.py | CETY | 1 field | Low | High |
| 12 | Shelf-family rollup companion-424B5 fallback | cards.py _shelf_family_drawn 1962 | FCEL | 1 field (+ fixture) | Low | High |
| 13 | Cross-period ELOC drawdown accumulation | capital_raised.py | KSCP | 1 field | Low | High |
| 14 | Split-consistency guard on amend (reject/re-apply stale/typo strike) | store.py, splits.py | GCTK | 2 fields | Med | Med-High |
| 15 | Shelf last_banker from direct/warrant takedown placement_agent | store.py, cards.py | SCNI | 1 field | Low | High |
| 16 | s1 fee-table-aggregate is not a separate offering (de-dup $30M ceiling) | store.py, tools/create.py | GCTK | 1 extra | Low | High |
| 17 | known_owners amend-only pass from resale F-3 (scoped, no create_*) | item_classification.py, walker_prompt.py | SCNI, QTEX | 3 fields | **Med-High (needs-design — §7.6)** | High |

### Fixture fixes (zero code risk — values in §3)
CGEN (3/4→4/4, −1 extra) · FCEL Dec-2025 ATM + add Jun-2026 (+2, −1) · ACTU equity_line (+1) ·
VRM maturity (+1, 3/4→4/4) · GCTK Series A/B exp (+2) · KSCP Apr-2025 shelf + Jun ATM (+2) ·
XTIA shelf + drop Series 8 (+3, −1 missing) · BNKK Jan-2025 strike + drop expired May-2021 (+2, −1) ·
QTEX remove Jul-2021 known_owners (+1) · CETY title + snapshot_fields + C-1062/C-1069 principal (+4).

---

## 7. Open / needs-design

1. **ATM-program segmentation (FCEL, KSCP)** — DT collapses one evolving Jefferies/Wainwright sales
   agreement into a single continuously-updated card; the walker freezes each amendment/re-registration
   as a separate superseded chapter. FCEL Apr-2024 ($1.1M vs frozen $204.9M, both filing-true at their
   as-of dates) + Mar-2025 extra and KSCP's 3 ATM-supplement extras are the **same** question. **Decide
   once, suite-wide:** adopt DT's single-program-card convention, or keep per-supplement chapters and
   align fixtures? No clean bug.
2. **GCTK Nov-2024 concurrent-PP 2,201 tranches (2 MISSING)** — fixture notes say "STILL-FAILING BY
   DESIGN…do NOT collapse rows." Emit the concurrent-PP tranche as a separate same-series row, or fold
   it / drop the fixture cards?
3. **KSCP Nov-2024 ATM remaining target is non-derivable** — DT's $3,117,034 is in no filing (only the
   annual aggregate exists). The walker's "dump annual total on newest supplement" is a real
   mis-attribution, but an exact match is unachievable from primary source. Chase a per-period heuristic
   or accept?
4. **XTIA source-convention conflicts (verifier overturns):**
   - June-2025 warrant remaining (6,483,892): **walker staleness** — W-5204 never re-walked the
     Q1-2026 exercises (10,447,300 − 3,963,408). Re-walk or accept?
   - warrant_coverage_pct (0 vs 100) & last_banker (Maxim vs ThinkEquity): registration statement
     (walker) vs FWP/standing-agent (DT) — which source convention wins?
   - s1 anticipated_deal_size ($18.4M fee-table vs $20M FWP headline) — prefer FWP headline?
5. **GCTK July-2024 strike $5,940 vs $5,490** — the 2026 10-K literally states "$5,490.00" (94↔49
   transposition typo); the walker faithfully extracted a wrong filing value. Accept, or add the
   split-consistency guard (fix #14)?
6. **known_owners resale-F-3 un-skip (SCNI, QTEX — fix #17)** — verdicts are sound, but un-skipping
   resale prospectuses risks reintroducing phantom `create_*` emissions. Can the amend-only pass be
   scoped tightly (no create_* tools on resale filings), or is a deterministic selling-shareholder-table
   parser safer?
7. **BNKK April-2022 twin-warrant fixture** — one fixture card represents two real twins (Greentree +
   L&H). Split the fixture into two cards, or accept the extra?
