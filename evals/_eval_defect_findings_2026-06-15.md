# Eval-Defect Root-Cause Report — 2026-06-15

> ## VERIFIED ADDENDUM (orchestrated 23-agent investigation, 2026-06-15)
> **Method:** per-ticker deep root-cause + per-ticker adversarial verifier (source-grounded in
> `dilution_raw`) + synthesis, over the 11 FAILING fixtures. 147 findings total; verifier verdicts
> 125 upheld / 18 revised / 4 refuted. Raw per-ticker findings + verdicts:
> `evals/_eval_defect_findings_2026-06-15_bundle.json`.
>
> **Canonical aggregate (15-fixture suite, scored on the live DB walked 06-11/06-12):**
> **640/728 fields = 88% · 63/112 cards exact · 107/112 found · 55 extras · 4/15 fixtures pass**
> (AACG, CGEN, IQST, XTLB). The body below reports the 11-fixture SUBSET it investigated
> (578/666 = 86.8%); add the 4 passing fixtures (62/62 fields, 9 exact) for the suite total.
>
> **Timing context that frames everything:** all fixtures were re-walked 06-11 (CELU 06-12). The
> RENDER-time cards.py fixes (`_warrant_dead` periodic guard, `_fully_drawn_clamp`) ARE active in
> this eval. The WALK-time store/walker fixes were NOT active during those walks (the conditional
> preferred conv_price split landed 06-12, after the BNKK 06-11 walk; Cluster H + the resurrection
> guard were 06-11 but flagged "activates on next re-walk"). So some defects are "fixed-in-tree,
> pending a fresh re-walk" (Bucket A) — but FAR FEWER than the prior 06-11 report assumed.
>
> **LOAD-BEARING CORRECTION (independently re-verified at the code level, 2026-06-15):**
> The prior 06-12 report claimed the conditional conv_price split fix banks BNKK **Series C**
> (0.5582→19.54) via the `_rescale_stale_unit_amend` echo-pin. **This is FALSE — the fix is a
> verified NO-OP for Series C.** Proof: with a single 1-for-35 split, `_echo_products = {1/35 ≈
> 0.02857}`; the echo-pin only re-pins a re-quoted value `v` to current when `v/current ≈ 0.02857`.
> Series C's real filing chain (P-446 history_json) re-quotes an INTERMEDIATE `1.081` at the
> 2025-12-31 10-K: `1.081/19.537 = 0.0553 ≈ 2/35`, NOT a clean echo → no pin; the amend post-dates
> the split so `cum==1.0` → no date-rescale → it CLOBBERS 19.537 down to 1.081; the next 10-Q then
> re-quotes 0.5582 (`0.5582/1.081 = 0.516`, not an echo) → clobbers to 0.5582. Final = 0.5582.
> The 06-12 unit test only exercised the clean 0.5582↔19.537 echo and missed the 1.081 intermediate.
> **Series C is OPEN (Bucket B9: amend-precedence so ANY post-split raw pre-split re-quote is
> re-split-adjusted), not Bucket A.** BNKK **Series B** DOES clear (its chain has no intermediate
> re-quote; split→11.90 sticks). KSCP's 4 phantom preferreds (Cluster H) DO clear (verifier executed
> the detector on the real Q2-2024 10-Q → 2024-05-15). So Bucket A = KSCP −4 extras + BNKK Series B
> +2 fields → suite ~640/728 (88%)→~642/728, extras 55→51. Everything else needs code or fixture work.
>
> ---

# Dilution Eval Suite — Consolidated Root-Cause & Action Report
*Synthesis of 11 per-ticker investigations (ACTU, BNKK, CELU, CETY, FCEL, GCTK, KSCP, QTEX, SCNI, VRM, XTIA), each adversarially verified. Findings used = UPHELD or REVISED (revision applied); REFUTED findings dropped.*

---

## 1. HEADLINE

**Live aggregate (11 fixtures re-run 2026-06-15):** **54/103 cards exact · 578/666 fields = 86.8% · 55 unexpected extras · 0/11 fixtures fully pass** (gate = all cards exact AND zero extras; the per-ticker "PASS" banner is only a section header — every fixture currently has either an extra or a non-exact card).

> Note on scope vs. the brief: the prompt's 640/728=88% / 4-pass figure is the **15-fixture** suite total; the numbers above are the **11 fixtures investigated here** (the other 4 — IQST/SCNI-adjacent etc. — were not re-run). Use 86.8% as the verified baseline for these 11.

**The single most important takeaway:** *Almost nothing clears on a plain re-walk.* The headline "in-tree walk-time fix" — the conditional preferred conv_price split-skip — was **REFUTED for its biggest target (BNKK Series C)** by the verifier: a post-split 10-K re-quote of an intermediate price ($1.081) defeats the echo-pin, so the fix is a verified no-op there. Only **KSCP's 4 phantom preferreds (Cluster H)** and **BNKK Series B (2 fields)** genuinely bank on re-walk. The real eval levers are **(a) fixture corrections** (zero risk, ~20 fields across CELU/BNKK/FCEL/KSCP/SCNI) and **(b) a small set of store/cards code fixes** (greenshoe dedup, shelf-family companion-424B5 rollup, warrant count-dust floor). The dominant *defect class* by volume is **DT-curated expected-extras + segmentation/convention disagreements** that are not bugs at all.

**Disagreement flagged up front:** The root-causer claimed `clears_on_rewalk` for BNKK-17/18/19 (Series C). The verifier **REFUTED** all three (stays 0.5582, fix is a no-op due to the $1.081 intermediate re-quote). This report follows the verifier — Series C is OPEN, not Bucket A.

---

## 2. BUCKET A — Clears on a fresh full re-walk (banked for free)

In-tree walk-time fixes that did NOT activate in the 06-11/06-12 walks and *will* fire on a fresh full re-walk. Verifier-confirmed firing only.

| Ticker | Defect | In-tree fix that clears it | Confirmed fires? | Predicted post-rewalk value | Eval delta |
|---|---|---|---|---|---|
| KSCP-1 | P-453 Nov-2023 Series A preferred extra | `walker._full_preferred_conversion_date` → `store.close_converted_preferred` (Cluster H) | **YES** — verifier *executed* the detector on the real Q2-2024 10-Q (acc 0001558370-24-012268), returns 2024-05-15; P-453 created 2023-11-13 ≤ that date → status `converted` → excluded by `preferred_cards` | card absent | −1 extra |
| KSCP-2 | P-454 Series B preferred extra | same | YES (same sweep) | card absent | −1 extra |
| KSCP-3 | P-455 Series M preferred extra | same | YES | card absent | −1 extra |
| KSCP-4 | P-456 Series S preferred extra | same | YES | card absent | −1 extra |
| BNKK-20 | Series B 2025 conversion_price 0.34 (should be 11.90) | `store._preferred_price_split_skip` (no conversion_ratio → split-adjust) | **YES** — verifier confirmed pref-series-b has NO conversion_ratio AND no intermediate post-split re-quote; even a re-emitted 0.34 echo-pins exactly (0.34/11.90 = 0.02857 = split ratio) | 11.90 → **matches fixture** | +1 field |
| BNKK-21 | Series B 2025 total_shares_issuable | derived from BNKK-20 (principal_total = liq_pref 5,409,000 / 11.90) | YES (rides BNKK-20) | 454,538 → **matches fixture** | +1 field |

**Predicted post-rewalk aggregate (Bucket A only):** extras **55 → 51**, fields **578 → 580 (87.1%)**. Cards-exact unchanged (the cleared items are extras/sub-fields, not whole new exact cards; BNKK Series B card has other open fields so it does not flip to exact).

**Re-walk caveat (critical):** Per the codebase memory, single-ticker `--force` re-walks drift (XTIA 3→5 extras observed) and Gemini fragile-payload-400s can drop events. Run a **full-suite parallel re-walk** (`scripts/run_eval_pipeline.sh`, PARALLEL=5), not per-ticker, and re-score only on a fully-walked DB. Several currently-passing cards (e.g. XTIA-6/8/9 walker fabrications) could equally *re-appear or shift*; the +4 extras / +2 fields are the floor, not a guarantee of net-positive if drift regresses elsewhere.

---

## 3. BUCKET B — Genuinely OPEN code fixes

Ranked by (fields+extras cleared)/(risk·effort). No in-tree fix exists or the in-tree fix is inert/wrong.

| # | Ticker(s) | Defect | Layer / function | Exact change | Gain | Effort | Risk | Conf |
|---|---|---|---|---|---|---|---|---|
| B1 | ACTU-1 | Shelf greenshoe double-count: $15.0M base (424B5) + $17.25M with-over-allotment (10-Q) both booked same-date/same-price on SH-2598 → total 33.38M vs 18.38M | `store._drawdown_already_recorded` (line ~2283, `_DRAWDOWN_DEDUP_TOLERANCE`=0.05) | Add same-date/same-price **superset-collapse keyed on share-count** (2,142,858 ⊂ 2,464,286 at equal $7.00) → keep MAX, not sum. 13.04% gap exceeds the 5% tol so current code sums them. Verifier reproduced 33,382,848.74 render and confirmed no superset-collapse exists | +1 field (→ exact-match 18,382,842.74) | S | med | high |
| B2 | FCEL-5 | Oct-2023 shelf rollup drops the $200M Dec-2025 ATM (atm-amendment-3 created off an 001-* 8-K, no registration_accession) | `cards._shelf_family_drawn` (~line 1997/2017) | Mirror the `_parent_shelf` same-day-companion-424B5 fallback (lines 194-217) so a sibling ATM created from an 001-* 8-K rolls up via its same-day 333-* companion (424B5 0001104659-25-125273, file# 333-274971). Verifier ran both helpers and proved the asymmetry. **Also re-pin the fixture** total_amount_raised (post-fix ≈498.99M, fixture's stale 368.78M won't match) | +1 field + render correctness | M | low | high |
| B3 | CELU-E01, E05 | Count=1 cashless-residual "dust" warrants render as live (W-5275, W-5284) | `cards._warrant_dead` | Add a count-dust floor: drop active warrant when `count <= 1 AND exercised_to_date > 0`. Verifier corrected E05's mechanism (it's count=1 dust, NOT the sibling-keep) | −2 extras | S | low | high |
| B4 | CETY-3/4/5/6/7 (+side CETY-1) | FY2025 10-K periodic-batch coverage gap: walker restated C-1062/C-1069 but skipped C-1060/C-1065 → stale balances + fallback discount×market conv_price | `anchor.py`/`store.py`/`tools/record.py` | Per-note owner+issue-date reconciliation over EVERY note in the 10-K batch (report fix #4), not the LLM subset. Clears conv_price (→$2.86), principal, and 2 derived fields on C-1060; principal_remaining on C-1065 (→$384,000) | ~5 fields (CETY-3/4/7 + derived 5/6) | M | med | high |
| B5 | QTEX-2/3/4/6 (+QTEX-5) | Duplicate-instrument: 20-F/A NOTE-11 warrant table RE-CREATES tracked tranches (W-5115 dup of W-5103; W-5141 dup of suppressed W-5104) + wrong same-strike row binding; compounded by 2 hallucinated W-5103 exercises | `store.py` (Cluster D dedup) + `walker.py` (exercise guard) | (a) Periodic warrant-table re-disclosure must AMEND not CREATE, binding the same-strike row by matching **expiration**; (b) walker must not emit exercise events from 6-Ks lacking warrant terms (both QTEX-5 exercises hallucinated from presentation/results-only 6-Ks) | 3 fields + 1 extra | L | med | high |
| B6 | BNKK-31/32 | Two Jan-2023 10-K warrant duplicates (W-5164/65) | `store.py` dedup | Verifier corrected the mechanism: created at event_date 2023-01-19 (4d inside the 60d window, NOT 434d outside); dedup fails on strike-key drift (1.0 vs amended 0.932 = 6.8% > 2%) + label mismatch ("Common" vs "Aegis"). Need composite expiration+initial_count dedup tolerant of strike drift | −2 extras | L | med | high |
| B7 | BNKK-24..28 | Series A (P-445) preferred all-null: no conv_price ($4.3935), no stated_value ($750), count wrongly ÷35 (39,993→1,143) | `walker.py` extraction + `store._apply_split` count guard | Extract Series A terms from 10-Q CoD; add `type=='preferred'` guard so `_apply_split` does NOT divide preferred share counts by the common-split ratio (verified: lines ~3516-3520 rescale all types). The split-skip fix can't help — nothing stored to divide | 5 fields | M | med | high |
| B8 | BNKK-22/23 | Series B principal_remaining = garbage 206 (balance-sheet $2-thousands mis-extraction) | `walker.py`/`cards.py` | Don't write a tiny carrying value as principal_remaining for preferred; or derive count×stated_value when stored value is implausibly small. Then rem_shares = 1,359,750/11.90 = 114,265 | 2 fields | M | med | high |
| B9 | BNKK-17/18/19 (Series C) **[was claimed Bucket A — REFUTED]** | conv_price 0.5582 should be 19.54 | `store.py` amend handling | Verifier-proven the split-skip fix is a NO-OP: post-split FY2025 10-K re-quotes intermediate $1.081 (1.081/19.537≠split ratio → echo misses → clobbers), then 10-Q re-quotes 0.5582. Need amend-precedence so a post-split filing's raw pre-split conv_price re-quote is re-split-adjusted | 3 fields | M | med | high |
| B10 | CELU-E13 | Helena convertible note C-1082 left active after the 2026-05-21 settlement closed the paired preferred P-464 | `walker.py`/`store.py` settlement-close | Extend settlement/termination handling to close ALL instruments named in a settlement-and-release (preferred + Exchange Note) for the same holder. `close_converted_preferred` is auto-conversion-only, doesn't cover cash settlement | −1 extra | M | med | high |
| B11 | SCNI-2/5, QTEX-1, BNKK-6/10 | Resale-prospectus known_owners gap (Armistice/Sabby/Bigger/District 2; Avatar/I9/LIA/Xylo/YA II) | `walker.py` resale gate (line ~1007 **fee-table** prescreen — verifier corrected: NOT line 909) | Fix #17: route resale F-3/F-1 selling-shareholder tables to a known_owners-ONLY amend pass, scoped so NO `create_*` tools are exposed (avoid phantom-create regression). Un-gate at the **fee-table** skip path | ~6 fields across 3 tickers | M | med-high | high |
| B12 | SCNI-7/8/9 | RK Stone ELOC: $2M pre-funded-warrant draw never booked + closed as "terminated" not "completed" | `walker.py`/`store.py` | Recognize pre-funded-warrant ELOC settlement as a full $2M drawdown (→ remaining $0 via existing dust clamp); don't classify a fully-drawn-at-expiry line as "terminated" | 3 fields | M | med | high |
| B13 | KSCP-13, SCNI-6 (REVISED→walker), CELU-F13 | Cross-period / missed ELOC drawdown under-counts | `capital_raised.py`/drawdown booking | Accumulate cross-period ELOC sales (KSCP Riley: only Q1-2023 $1.3M booked, missing FY2022 $2.9M); book SCNI YA II July/Aug-2025 $4.2M; net CELU SEPA $3.15M pre-paid advance | ~3 fields | M | low-med | high |
| B14 | XTIA-2/3 | s1 anticipated_deal_size + warrant_coverage_pct overwritten by S-1/A fee-table (should reflect original S-1/FWP: $20M, 0%) | `walker.py`/`tools/create.py` | Preserve create-time anticipated values; don't let an S-1/A amend overwrite anticipated (route to FINAL fields). Same class as GCTK s1 anchor | 2 fields | M | med | high |
| B15 | GCTK-11/14/15 | s1 anchored to S-1/A (2024-11-08) not original S-1 (2024-09-16) → Sept fixture MISSING + 2 extras; + $30M fee-table phantom | `store._create_anchor` + s1 dedup | Anchor s1 created_at to earliest `family_registration_accessions`; fold S-1 fee-table aggregate into the priced row (don't mint $30M phantom). Recovers 1 found card + ~7 fields + −2 extras | 1 card + ~7 fields − 2 extras | M | med | high |
| B16 | CELU-F09, plus E03/E09-11/M01/M02 | Warrant offering-splitting + missed June-2025 $2.50 / July-2025 $2.84 repricings + Faithstone over-split + missed July-2023 RD & SPAC Sponsor warrants | `walker.py` amend/segmentation | Apply repricings from the Q3-2025 10-Q master table; collapse Faithstone advisory tranches; create the missed July-2023 RD (857,143) & Sponsor (849,999) warrants. In-tree collapse fix can't fire (differing initial_counts) | ~1 field + extras/missing | M-L | med | high |
| B17 | GCTK-1/8 | Warrant strike split-inconsistency (Series B $36.20 should be $2,172; Hybrid $5,490 should be $5,940) | `store.py`/`splits.py` | Echo-pin/reject an amend strike that is not a contiguous split-product of the create strike, or enforce sibling Series A/B consistency. Effort downgraded: GCTK-8 evidence partly unverifiable (10-K cache truncated) but create-chain math is exact | 2 fields | M | med | high |
| B18 | GCTK-2..7 | Six Ballantyne warrant term-dates faithfully copy a DEF 14A "2034" typo (+10yr) | `walker.py` amend guard | Reject a date amend pushing exercisable_date past expiration/sane horizon. One amend-guard fixes all 6 date fields | 6 fields | M | med | high |
| B19 | CETY-1 | Aug-2022 warrant total_issued contaminated (133,333 vs 2,894) via amend→initial_count mirror | `store.py` (lines 2208-2214) | An amend that mirrors count→initial_count must reject a value inconsistent with the create-time figure. **Verifier REVISED the evidence** (the "2,000,000 warrant shares" quote is fabricated; real text = "2,000,000 **units**" subscription) — diagnosis/fix unchanged | 1 field | M | med | high |
| B20 | KSCP-11/12 | Nov-2024 ATM remaining mis-attributes the FY2024 *annual program aggregate* to one supplement | `store.py` drawdown attribution | Apportion annual "issued N shares under the ATM program" across the program-family, not the newest supplement. **Caveat: fixture target 3,117,034 is non-derivable from source** (grep = 0 hits) — even a correct fix won't exact-match; recommend snapshot-exempt the field | 0 reliable (non-derivable) | L | med | med |

---

## 4. BUCKET C — Fixture corrections (JSON edits, zero code risk)

| Ticker | Card | Field | Current | Correct | Filing cite |
|---|---|---|---|---|---|
| BNKK | Jan-2025 warrant (W-5181) | known_owners | `['Bigger Capital']` | `['Trajan Holdings','Fried']` | Amendment 0001641172-25-018310 recital: Purchasers (Trajan + Fried) "purchased … a warrant dated January 20, 2025" — **refutes** fixture note #15 |
| BNKK | Sept-2024 warrant (W-5168) | title | `September 2024` | `August 2024` | 8-K 0001493152-24-035089: "dated as of August 30, 2024 … Core 4 Capital Corp." |
| BNKK | Sept-2024 warrant (W-5168) | known_owners | `['Insiders']` | `['Core 4 Capital']` | same 8-K names Core 4 Capital Corp. as Purchaser |
| BNKK | April-2022 warrant | (whole card) | one merged Greentree+L&H card | **split into two twin cards** (Greentree W-5139 + L&H W-5146, each total_issued 31,429) | W-5139 from 8-K 0001493152-22-010686; W-5146 from 10-Q; merged card matches neither twin → clears the W-5139 extra too |
| BNKK | Jan-2023 warrants (×2) | exercise_price | `32.55` | `32.62` (optional) | 424B5 **0001493152-23-032382** (verifier-corrected accession): "modify the exercise price of the Warrants to $0.932 per share" × 35 = 32.62 |
| CELU | 2021-07-16 warrant | exercisable_date | `2021-09-14` | `2021-08-15` | S-1/A 0001213900-21-040814: "exercisable on August 15, 2021" |
| CELU | Jan-2024 RWI T2 (W-5290) | exercise_price | `2.99` | **`2.844`** (NOT 2.84) | Q3-2025 10-Q 0001493152-25-023467 rounds "$2.84" but walker stores 2.844; 2.84 still fails 0.1% tol (verifier-corrected) |
| CELU | Jan-2024 RWI T2 | exercisable_date | `2024-01-16` | `2024-07-15` | 10-Q 0000950170-24-133799: "became exercisable on July 15, 2024" |
| CELU | Jan-2024 RWI T2 | expiration_date | `2029-01-16` | `2029-07-15` | Q3-2025 master table |
| CELU | April-2023 RD (W-5281) | remaining_outstanding | `923077` | `435625` | Q3-2025 master table $7.50 tranche |
| CELU | May-2022 PIPE (W-5278) | expiration_date | `2027-05-20` | `2028-10-10` | Q3-2025 10-Q: extended to "October 10, 2028" |
| CELU | June-2023 RWI (W-5283) | exercise_price | `8.1` | **`2.844`** (NOT 2.84) | Q3-2025 master table July-24-2025 RWI reprice; same precision trap as above |
| FCEL | June-2026 Shelf | (add card) | absent | add S-3ASR 333-296607 (effect 2026-06-08, Registered, capacity 999999999, raisable 999999999, baby='No', raised 0) | S-3ASR 0001104659-26-071419; borderline expected_extra — equally defensible to leave |
| FCEL | Oct-2023 Shelf | total_amount_raised | `368779594.92` | `≈498,990,000` (after B2 code fix) | re-pin to post-fix rollup (298.91M + 200.07M) |
| KSCP | Nov-2022 warrant (W-5224) | title_contains | `November 2022` | `October 2022` | **Verifier REVISED to fixture_error**: closing 8-K 0001104659-22-110300 "On October 20, 2022 … Closing"; DT's "November" tracks only the Nov-10 resale S-1 |
| KSCP | April-2022 Riley ELOC | remaining_capacity | `97,100,000` | `95,800,000` | FY2022 10-K $2.9M + Q1-2023 $1.3M = $4.2M drawn; **fixture itself was wrong** (paired with B13 code fix) |
| KSCP | Nov-2024 ATM | remaining_capacity/_without_baby_shelf | scored | mark snapshot-exempt | target 3,117,034 non-derivable (grep=0); see B20 |

**Drop refuted/stale fixture prose (no scoring impact, cosmetic):** CELU note #11 (P-464 "active walker bug" — it's correctly redeemed, reaper passes); BNKK note #15 (warrant stayed with Bigger); ACTU note #6 (warrant "supersession" framing — both warrants converted-and-stayed-live); SCNI YA II note (the asserted $4.2M *third* advance does exist after all — **do NOT drop SCNI-6 as fixture error**, see Bucket B13/verifier reversal).

---

## 5. BUCKET D — Scorer / matcher artifacts (run_eval.py)

| Defect | Mis-pairing | Minimal matcher fix |
|---|---|---|
| CELU-F05 | No-owner fixture March-2023 PIPE card binds to lower-strike Starr W-5280 (|1.69−2.50|<|30−2.50|) instead of the real PIPE W-5279 | Add an initial_count/total_issued tie-break **before** strike-proximity when fixture has no owner — *but high risk*; cleaner path is the CELU-E03 fixture re-pin (bind to 938,184@$30 Robert Hariri). Largely a segmentation problem, not purely matcher |
| CELU-F06 | April-2023 blended $4.50 fixture binds to $7.50 tranche W-5281 | Same segmentation root; no clean matcher-only fix |
| GCTK-11/14 | s1 fixture keyed 2024-09-16 can't bind actuals anchored 2024-11-08 (53d > ±14d window) | This is really the **store anchor bug B15**, not a matcher bug — fix the anchor, not the matcher window |

The matcher itself is largely sound; most "scorer_artifact" findings are downstream of store anchoring or segmentation and should be fixed there. No matcher code change is high-value/low-risk enough to recommend standalone.

---

## 6. BUCKET E — Expected extras / snapshot drift / convention (NO ACTION)

These are real instruments DT curates out of its live screenshot, or convention differences — chasing them regresses other fixtures. Listed so they are not re-investigated:

- **ACTU-2, ACTU-3** — W-5238 ($10.55 Series B) & W-5243 ($9.42 Series C) warrants, both proven live in the Q1-2026 10-Q NOTE 7 table; DT live-only view omits them.
- **BNKK-29, BNKK-33, BNKK-34** — Jul-2021 Aegis IPO (W-5120), Aug-2023 PP $43.05 (W-5155), Apr-2025 Core 4 (W-5187) warrants, all genuinely live.
- **BNKK-37 (P-437), BNKK-39 (S1-205)** — open-but-no-fix: NFH 2021 Series B stale-active (needs a preferred absence-reaper, ambiguous per SCNI EIB precedent); 333-284689 S-1 settlement-exhibit misclassified as offering. Defer (needs design).
- **CELU-E04, E06, E07, E08, E12** — June-2023/March-2024/Nov-2024/Feb-2025/July-2025 warrants, all in the authoritative Q3-2025 master table; DT omits micro-tranches.
- **CELU-NOTE-P464** — reaper PASSES; the fixture's "live walker bug" note is stale.
- **CETY-8/9/10/11** — Nov-2022/March-2023 Mast Hill warrants + May/July-2025 1800 Diagonal convert-on-default notes, all FY2025 10-K-confirmed live, DT-curated-out.
- **FCEL-3/4/6/7/8/9** — the single continuous Jefferies Open Market Sale Agreement program (June-2020 → Dec-2025 chapters); walker freezes each amendment, DT collapses to 2 cards. **Do NOT land a date-partition rollup** — fixture round-3 note proves it regresses SCNI.
- **GCTK-12/13** — Nov-2024 Series A/B concurrent-PP 2,201 sub-tranches; segmentation convention, "do NOT collapse rows in code (refuted)".
- **KSCP-5/6** (two genuine pre-IPO Alto preferred-warrant cohorts, extended to 2027-12-31) **and KSCP-7/8/9** (three Wainwright ATM-supplement re-registrations of the one Feb-2023 agreement) — DT collapses; per-supplement carding is the open §7.1 suite-wide decision.
- **VRM-1** — Dec-2023 Virtu ATM ($50M, drawn $2.5M, superseded→"Replaced"). **This is the only fixture-edit in Bucket E**: VRM is otherwise 4/4 cards, 26/26 fields; adding the "Replaced" predecessor card (start 2023-12-01, registered "Replaced", total 50M, remaining 47.5M, agent Virtu) makes VRM the **most likely fixture to flip to a true PASS**. The fixture note's "walker reaped it" premise is wrong (superseded, not closed). Matches FCEL/KSCP "Replaced" convention.
- **XTIA-4 (last_banker ThinkEquity vs Maxim — convention), XTIA-5 (legacy pre-merger XTI warrant, 1 share), XTIA-7 (126-share Series 5 preferred, live)** — no action. *Note XTIA-6/8/9 are genuine walker fabrications → Bucket B-class open, not expected-extra.*
- **XTIA-1** snapshot drift (June-2025 warrant count stale; fixture's 6,483,892 itself not cleanly filing-derivable).

---

## 7. RECOMMENDED ACTION SEQUENCE

**Order for max gain at min risk:**

1. **Land all Bucket C fixture edits first** (zero code risk, ~20 fields + likely flips VRM to PASS). Critical precision note: **CELU F02/F11 must edit to 2.844, NOT 2.84** (the 0.1% tolerance rejects 2.84 — verifier-caught trap that would silently waste both edits). Re-score on the *current* DB (no re-walk) — these are render/static gains.
2. **Land the two cleanest store/cards code fixes** that don't need a re-walk: **B1 (ACTU greenshoe superset-collapse, S/med)** and **B2 (FCEL shelf-family companion-424B5 rollup, M/low)** and **B3 (CELU warrant count-dust floor, S/low)**. All three are render-time or static-DB-correctable; re-score to bank +2 fields −2 extras + the FCEL rollup correctness.
3. **Then do ONE full-suite parallel re-walk** (`scripts/run_eval_pipeline.sh`, PARALLEL=5) to activate Bucket A (KSCP −4 extras, BNKK Series B +2 fields → 51 extras / 87.1%). Do this *after* steps 1-2 so you can attribute deltas cleanly, and re-score only on the fully-walked DB.
4. **Batch the high-ROI walk-time code fixes for the *next* re-walk cycle**, prioritizing B11 (resale known_owners, ~6 fields/3 tickers), B4 (CETY per-note batch, ~5 fields), B7 (BNKK Series A, 5 fields), B18 (GCTK 6 date fields, one guard), B12 (SCNI RK Stone, 3 fields). These need code + a re-walk to verify, so land them together, not piecemeal.
5. **Defer** the L-effort / med-risk items (B5 QTEX duplicate-instrument, B6/B16 warrant dedup-with-strike-drift, B20 non-derivable KSCP target) and the needs-design preferred absence-reaper (BNKK-37) until the cheaper banks are taken.

**Honest re-walk drift / nondeterminism risk:** Per codebase memory, **full-suite re-walks > single-ticker** (single `--force` walks drift — XTIA went 3→5 extras). LLM nondeterminism plus Gemini fragile-payload-400s can *regress currently-passing state* (e.g. XTIA's fabricated extras XTIA-6/8/9 may re-spawn or shift; CETY's per-note batch coverage is nondeterministic — last walk caught C-1062 but not C-1060). The Bucket A +4 extras/+2 fields is a **floor**, not a guaranteed net gain. Mitigate by: (a) doing fixture/render gains *before* re-walking so they're locked in independently; (b) re-walking the whole suite in one pass; (c) re-scoring only on a fully-walked DB (never mid-walk — the scorer false-positives mid-walk). The single biggest banked-on-re-walk verdict (BNKK Series C, 3 fields) was **REFUTED** — do not count on it; it needs the B9 amend-precedence code fix.