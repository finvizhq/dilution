All file references confirmed. The structure is `cards.<type>[index]`. I have everything I need to write the report accurately.

# Post-Re-Walk Drift Triage — Prioritized Action Plan (DB walked 2026-06-16)

## 1. Verdict on the re-walk — KEEP the DB

**Net: a marginal win, KEEP it.** The headline metrics moved the right direction on the dimension that matters most for product trust:

- **Extras 51 → 44 (-7)** — the structural win. KSCP Cluster H cleared (4 phantom converted preferreds gone), BNKK Series B `conv_price` corrected to 11.90 (Cluster A fired and is verified by S-3 0001493152-26-017674).
- **Field % 90.4 → 91 (+0.6).**

The losses are real but **none are corruption** — they are all isolated, diagnosed, walk-time **cross-contamination/date-attribution regressions**, not a broken pipeline:
- **cards exact 72 → 68 (-4)**, **found 109 → 106 (-3)**, **fixtures-passing 5 → 4 (-1, VRM)**.

The single concerning item is **VRM losing its PASS** to one field (warrant `exercisable_date` drifted to a hallucinated 2026-01-15 / 2030-01-15 — fixture is filing-true 2025-01-14). It is **not recoverable now** (bad value baked into `terms_json`; render fallback is pre-empted; `created_at`=8-K filing date 2025-01-15 ≠ 01-14). It is a clean, well-understood walk-time fix and does not impugn the DB. No item is bad enough to discard the walk; the extras reduction is durable and the regressions are recoverable on the next cycle.

---

## 2. APPLY NOW — fixture edits (upheld, no re-walk)

Fixtures are structured as `cards.<type>[index]`. All edits below are where the **walker is now filing-TRUE and the fixture is stale**, verified by the adversarial pass.

| File / key | New value | Filing justification | Eval delta |
|---|---|---|---|
| `evals/GCTK.json` `cards.warrant[1].exercise_price` (Nov 2024 Series B) | `2172.0 → 36.2` | Latest 10-Q 0001493152-26-023205: "Series B...exercise price of $36.20" (distinct reset regime; prior $2,172 was a restated Monte-Carlo copy error) | **+1 field** |
| `evals/GCTK.json` `cards.warrant[2].exercise_price` (Nov 2024 Series B, 2201 tranche) | `2172.0 → 36.2` | Same filing; keep instrument internally consistent | 0 (unmatched card, no regression) |
| `evals/GCTK.json` `cards.warrant[1].total_issued` (Nov 2024 Series A) | `5995 → 8359` | 10-Qs 0001493152-26-023205 / -25-022299 / -25-011981 all state "aggregate of 8,359 Series A Warrants"; **"5,995" appears in NO GCTK filing** (fabricated). Walker 8358 is within 0.1% tol | **+1 field** |
| `evals/CETY.json` `cards.convertible[]` ADD | New card: May-2025 1800 Diagonal — `principal_total 131610`, `principal_remaining 29247`, owner `1800 Diagonal Lending`, `issue 2025-05-08`, `maturity 2026-02-15`, registered `Not Registered`; **omit conversion_price** | FY2025 10-K 0001493152-26-027409: "$131,610...balance...$29,247". Walker C-1153 filing-exact | **-1 extra, +1 card found/exact (+6 field-checks)** |
| `evals/CETY.json` `cards.convertible[]` ADD | New card: July-2025 1800 Diagonal — `principal_total 151800`, `principal_remaining 91957`, owner `1800 Diagonal Lending`, `issue 2025-07-30`, **`maturity 2026-05-30`** (NOT 2026-02-15), `Not Registered`; omit conversion_price | FY2025 10-K: "$151,800...balance...$91,957". **maturity per binding note exhibit 0001641172-25-022038 ("May 30, 2026")** — the 10-K prose "Feb 15" is a carry-over typo; walker stored 2026-05-30 correctly | **-1 extra, +1 card exact (+6 field-checks)** |

**Adversarially-REJECTED fixture edits — DO NOT apply:**
- **KSCP Riley ELOC 97.1M → 95.8M**: 95.8M is filing-true, but the walker emits **98.7M** either way, so the field stays failing — **eval delta 0**. Apply only as an honest fixture-accuracy improvement (resolves note 14), not as a recovery lever. Classification is *fixture-stale + walker-bug*, not walker-correct.
- **GCTK Nov-2024 collapse to 8359/8359**: REJECTED — walker renders Series B `total_issued=5996`, not 8359; symmetric collapse regresses Series B (currently passes at 5995 vs 5996) for **net ~0**. Standing "do NOT collapse rows" decision; accept 2 MISSING as drift.
- **SCNI Dec-2023 expiration 2029-01-03 → 2027-01-04**: REJECTED — 521,310 is the **combined** 3yr+5.5yr leg; 2027 retires the live 5.5-yr book ~2.5yr early. Real defect is the walker offering-split/aggregation bug (walk-time).

**VRM PASS-recovery: does NOT qualify as a fixture edit.** The fixture is already filing-correct (2025-01-14); the walker drifted. **Not recoverable now** — requires a walk-time §1145/"from and after the Original Issue Date" anchor rule (see §4). This is the one PASS that cannot be bought back this cycle.

---

## 3. APPLY NOW — render-time cards.py fixes (upheld)

| Fix | Function / logic | Delta | Regression-safety |
|---|---|---|---|
| **GCTK Dec-2024 ATM render** | `atm_cards` (cards.py ~1562-1564): add `"terminated"` to statuses tuple, **gated to fully-drawn** via existing `_fully_drawn_clamp` semantics: include iff `capacity>0 AND stored remaining==0 AND 0<=capacity-drawn<0.5%*capacity` | **+1 card found, +1 exact, up to +7 fields** | **Provably zero-regression**: DB-wide scan of all 52 terminated ATMs — this exact gate flags **only ATM-2679**. Do NOT use the loose "has-drawdowns/remaining=0" wording (resurrects XTIA ATM-2657/2664, FCEL ATM-2658, CGEN/SCYX/CLIR/etc.) |
| **BNKK Series B `principal_remaining`** | Preferred block (cards.py ~1320-1333): set `principal_remaining = count*stated_value` **only on a positive store conversion/redemption signal** — `out.count_converted_to_date` or `out.count_redeemed_to_date` present-and-positive (Series B: count 3606→1813). NOT the proposed `count*stated_value < liq_pref / count<initial_count` trigger | **+2 fields** (principal_remaining 5.4M→1.36M, cascades remaining_shares →114,265) | Corrected trigger avoids firing on the BNKK May-2025/Series A-1 row (count÷split-corrupted) and CELU P-467; leaves split-bug rows on the existing liq_pref fallback (no value regression) |
| **GCTK s1 `warrant_coverage_pct`** | s1 card (cards.py ~1462): `coverage_pct = coverage_pct if coverage_pct is not None else final_coverage_pct` — **explicit None check, NOT `or`** | **+1 field** (renders 200, GCTK 90/91→91/91) | The literal `or` regresses 3 live rows storing legitimate `0.0` (ELUT/IPW/ZYBT → None). None-check preserves them; XTIA S1-218 (init=1.0) untouched |
| **SCNI Jan-2024 Inducement extra** | `_warrant_dead`: drop a `count=0/exercised` row that shares `created_accession + label + issue_date` with a live sibling and contributes 0 remaining | **-1 extra** (bonus: BNKK extras 10→9) | DB-wide enum: only SCNI W-5336 + BNKK W-5307 match; the 6 other paired-keep cases have different labels and are untouched; dust tests unaffected |
| **XTIA Maxim ATM chain (5 extras)** | **TWO-part fix** (corrected): (a) extend `_chain_head_terminated` to treat a head as dead when `terms.agreement_end_date < today`; **AND** (b) add a head-level skip in `atm_cards` for an active row whose own `terms.agreement_end_date < today` (ATM-2678 is status=`active`, not `superseded:`, so the cascade alone won't suppress it) | **-5 extras** | The proposal's single per-row skip drops only 3 (ATM-2675/2676 have no end_date). Both parts needed. XTIA is the **only** ATM head with a past `agreement_end_date` across all 99 rows → suite-wide no-op elsewhere |

**FCEL Oct-2023 423.79M overshoot — NOT cleanly recoverable at render.** Both the new walker value (423.79M) and the fixture (368.78M) are wrong vs filing-truth **$298.9M**. Root cause is `cards._shelf_family_drawn/_drawn_from_parts` double-counting ATM-2673's FY2025 cumulative anchor (190.4M) against the overlapping Nov-6 8-K discrete draw (138.27M, straddles the Oct-31 as-of). The proposed `min(drawn, total_shelf_capacity)` clamp caps only to **405M ≠ 298.9M** (insufficient), and a per-sibling residual-capacity clamp is fragile (nested slices). This is the **deferred B2 ATM-family incremental-accounting redesign** (walk-store) — do NOT chase a render clamp. The interim clamp is SAFE for SCNI (family sums << cap) but buys nothing for the failing field. **Leave for §4; re-pin fixture to 298.9M only after the redesign.**

Run `pytest tests/ -q` after these edits (0 failed / 0 xfail gate).

---

## 4. NEXT WALK CYCLE — walk-time fixes (need a re-walk), ranked by eval ROI

**Group A — Warrant date/offering cross-contamination & coverage gaps (HIGHEST ROI: ~30+ fields).**
This is the dominant source of this cycle's regressions.
- **XTIA missing Sept-2025 warrant (+7 fields)** — walker booked the offering only as an SH-2611 shelf takedown, dropped the 12.5M $2.00 common-warrant leg; that drop then contaminated the June-2025 warrant (mis-attributed $2.00 exercises → remaining/expiration wrong, +3 more fields) and the Mar-2025 warrant carries Jan-2025's dates (+2). Fix: emit `record_warrant` for the dropped leg; date-attribute warrants to the **issuance-event** date, not the form-filing 8-K.
- **CELU missing July-2023 RD (+7), missing Sponsor (+6), July-2025 PIPE date 16d off >14d matcher tol (simultaneous missing+extra, +6)** — same date-attribution + coverage-gap family. Largest single-ticker lever.
- **GCTK July-2024 Ballantyne ×3 `exercisable_date` (+3)** — walker copied the termination date into `exercisable_date`; must compute `issue + 12mo`. Plus June-27 Warrant strike `5490 → 5940` digit transposition (+1).
- **VRM §1145 warrant (+1, restores the lost PASS)** — store rule: for "from and after the Original Issue Date"/immediately-exercisable warrants, anchor `exercisable_date := issue_date`, `expiration := issue_date + term`, overriding LLM absolute dates.
- **QTEX July-2021 duplicate-instrument (+8)** — collapse W-5250/W-5253 to one common-warrant row (5.50, total 3,345,455, remaining 1,640,455, exp 2026-07-16); the Aegis Representative warrant must attribute to July-2021 + be labeled "Representative/Underwriter" so render-suppression folds it.

**Group B — Split-adjustment / repricing propagation (HIGH ROI: ~15+ fields).**
- **BNKK preferred `conv_price` split-skip (Series C +3, May-2025/Series A-1 +5)** — apply the split to `conv_price` at create for no-`conversion_ratio` preferreds so "current"=post-split BEFORE any echo-pin (Series C 0.5582→19.54). Echo-pin NO-OP confirmed as predicted. **Also the preferred-COUNT-÷-by-split bug** (May-2025 count 39993→1141) — preferred share counts must NOT be divided by the common-split ratio.
- **CELU warrant repricing propagation** — May-2023 PIPE $10→$2.50, March-2023 PIPE split $30/$2.50 (June-2025 SPA); walker keeps original issuance strikes.
- **BNKK Jan-2025 warrant** — missed July-2-2025 SPA reprice $0.4348→$0.33 AND split-rescale to 11.55.
- **GCTK** — covered in Group A (strike transposition is split-derived).

**Group C — Owner extraction from resale registrations (MEDIUM ROI: ~8 fields).**
- **BNKK** (Jan-2023 6-holder cluster, Jan-2025 Trajan/Fried, April-2022 GreenTree+L&H), **QTEX** (Dec-2024 5 holders), **SCNI** (Dec-2023 / Sept-2023 Armistice etc.), **CELU** (Oct-2025 Helena). Single shared fix: a **resale-F-3/424B3 `known_owners` amend pass** that harvests selling-shareholder tables and maps to the underlying instrument by share count, instead of hard-skipping the resale gate.

**Group D — Segmentation / fabrication / lifecycle close (MEDIUM ROI: ~10 extras + fields).**
- **CELU** — drop phantom Sept-2023 Inducement (a March-2023-PIPE reprice, not a new instrument); preferred-exchange close path for P-467 (Helena surrender, count!=0 blocked close).
- **ACTU** — extinguish Sep-2018 + Jun-2023 preferred-stock warrants at the Aug-14-2024 IPO net-exercise (10-K: "no...Preferred Stock Warrants outstanding").
- **BNKK** — dedup the Silverback liability-settlement double-book (emitted as BOTH convertible C-1092 AND equity_line EL-204).
- **CETY** — process Q3 10-Q/A conversion notices for periodic-balance restate (Jan-2025 conv_price, Aug-2025 principal 388,888→206,745); remove spurious Jefferson Aug-2022 exercised_to_date.

**Group E — Store accounting (LOWER ROI, harder).**
- **FCEL** ATM-family incremental-accounting redesign (deferred B2) — the Oct-2023 423.79M overshoot + per-supplement remaining-propagation ($204.9M→$1.1M). Then re-pin fixture to 298.9M.
- **XTIA** May-2024 shelf anchor+discrete double-count (85.3M→49M; +2 fields).
- **KSCP** June-2024 ATM cumulative-draw mis-attribution + FY2022 Riley draw under-count (then fixture→95.8M passes).
- **SCNI** YA-II SEPA draw under-count (B13; fixture 4.31M is correct target), shelf `last_banker` storage on the takedown row.
- **CELU** YA-II SEPA $3.15M Initial-Advance netting.

---

## 5. ACCEPT AS DRIFT / NO ACTION

- **KSCP** — May-2022 warrants ×2 (genuinely distinct Series m-3 vs Series S cohorts, fixture note 7), 3 Wainwright ATM extras (per-supplement re-registrations of the ONE Feb-1-2023 agreement; deeper-than-DT segmentation, memory `kscp-atm-one-agreement`).
- **FCEL** — 4 superseded ATM-chapter extras (Jun-2020/Jun-2021/Jul-2022/Mar-2025); leak only because the Dec-2025 head is active; clearing them needs a "one-agreement collapse" convention decision, not a render hack.
- **XTIA** — Mar-2022 (W-5276) + Jul-2023 (W-5309) warrant fabrications (DT-curated-out legacy/merger-pro-forma); FC Imperial ELOC (proposed/unconsummated Reg-FD press release); s1 deal-size $20M-vs-$18.4M and warrant_coverage 0-vs-100 (pre-flagged FWP-vs-fee-table source-convention).
- **BNKK** — Series B `convertible_date` July-2 vs Aug-26 (both filing-grounded); per-tranche warrant counts (DT-proprietary, not filing-provable, only the 685,254 aggregate is).
- **CELU** — Mar-2024 RWI, Nov-2024, July-2025 Starr ×2, July-2025 RWI extras (real instruments DT scopes out / walker-beats-DT).
- **GCTK** — Nov-2024 2201-tranche MISSING ×2 (segmentation, standing "do not collapse rows").
- **CETY** — Mar-2023 Mast Hill partial-exercise warrant (live, DT below-threshold).
- **ACTU** — Aug-2024 underwriters' warrants (161,000 @ $10, real comp warrant DT curates out).
- **SCNI** — Dec-2023 title December-vs-January (offer-date vs close-date convention, eval matches by date).

---

## 6. Net eval projection if §2 + §3 applied now

Starting point (post-re-walk): **91% fields, 44 extras, 68 exact, 4 fixtures passing.**

**Fixture edits (§2, upheld only):** GCTK +2 fields (Series B strike, Series A total_issued); CETY -2 extras / +2 cards exact / +12 field-checks (two 1800 Diagonal adds). KSCP ELOC edit = 0 (apply for accuracy, not score).

**Render fixes (§3, upheld only):** GCTK ATM +1 card found/exact / +up to 7 fields, s1 +1 field; BNKK Series B +2 fields; SCNI -1 extra; XTIA -5 extras (two-part fix).

**Projected:**
- **Extras: 44 → ~36** (-8: CETY -2, SCNI -1, XTIA -5).
- **Exact cards: 68 → ~71-72** (GCTK ATM + 2 CETY notes).
- **Fields: 91% → ~92%** (GCTK +~11, BNKK +2, plus the new CETY exact cards).
- **Fixtures passing: 4 → likely 5** (GCTK is the strongest candidate to flip with ATM render + 2 fixture fields + s1 coverage all landing on the same ticker; verify with `run_eval_all.py` on the walked DB).

**VRM cannot be recovered this cycle** (needs the §4 Group-A §1145 walk-time rule) — so the lost VRM PASS is the one durable casualty until the next re-walk. The biggest deferred lever remains **FCEL Oct-2023 (298.9M)**, which is blocked on the B2 ATM-family redesign and is *not* render-recoverable. Run `pytest tests/ -q` (0 failed/0 xfail) and `run_eval_all.py` against the current walked DB to confirm before committing.