# Known issues surfaced by the unit tests

These are divergences between a module's documented intent and its actual
behavior, found while writing the test suite.

The **robustness (A)** and **logic (B)** bugs below have been **FIXED** —
production code was corrected and each corresponding test now asserts the
intended behavior (search the tests for `# FIXED (was bug …)`). The
**cosmetic / latent (C)** items remain open, captured as characterization
tests (`# BUG:` notes) that pin current behavior.

Run `pytest tests/ -q` — the suite is fully green with no `xfail` markers.

## Robustness — uncaught exceptions on malformed/edge input — FIXED

1. **`fx._frankfurter_read`** (`dilution/fx.py`) — a cache file
   `{"rate": null}` made `float(None)` raise `TypeError`, which was **not**
   in the `except (OSError, ValueError, KeyError)` guard, so it escaped
   instead of returning `None`. **Fix:** added `TypeError` to the except
   tuple. *Test:* `test_fx.py::TestFrankfurterRead::test_null_rate_returns_none`.

2. **`fx._finviz_series`** (`dilution/fx.py`) — same class: a JSON `null`
   close in the cached series raised an uncaught `TypeError` instead of
   falling through to a re-fetch. **Fix:** same. *Test:*
   `test_fx.py::TestFinvizSeries::test_cache_null_close_falls_through_to_fetch`.

3. **`os_history.fetch_os_history`** — restatement dedup crashed with an
   uncaught `TypeError` when two facts shared a `period_end` and the
   later-iterated one had `filing_date=None` (`None > date(...)`). **Fix:**
   treat a missing filing_date as oldest (keep the dated restatement).
   *Test:* `test_os_history.py::TestFetchOsHistory::test_dedup_collision_none_filing_date_does_not_crash`.

4. **`s1_status.derive_s1_status(today=...)`** (`dilution/ledger/s1_status.py`) —
   the `today=` param only worked for a `datetime.date`: a `datetime.datetime`
   silently disabled the lapse check, a plain ISO string raised
   `AttributeError`, and a `created_at`/`filing_date` stored as a full ISO
   timestamp (the schema default `…T00:00:00Z`) was rejected by
   `date.fromisoformat`. **Fix:** added a `_coerce_date()` helper that
   normalizes date / datetime / ISO-string (trimming a timestamp to its date
   head) for both `today` and the filing-date comparison. The helper was
   then extracted to a shared `dilution/ledger/_dates.py::coerce_date` and
   the **mirror module `shelf_status.py`** (which had the identical latent
   bug — a string/`datetime` `today`, and a timestamp anchor collapsing the
   expiration to the `9999` never-expires sentinel via `_add_days`) was
   routed through it too, keeping the two mirrors consistent.
   *Tests:* `test_ledger_s1_status.py::TestTodayParam::*`,
   `test_ledger_shelf_status.py::TestMiscellaneous::test_today_as_string_is_accepted`,
   `::test_timestamp_anchor_does_not_collapse_expiration`.

5. **`_label._resolve_slot` / `build_label`** (`dilution/ledger/_label.py`) —
   `_resolve_slot` read `m.terms` unguarded (only `build_label` guarded it),
   a truthy non-dict `terms` (list/str) survived `or {}` then exploded on
   `.get`, and `build_label` dereferenced `m.type` directly. **Fix:**
   `_resolve_slot` reads `terms` via `getattr` and coerces a non-dict to
   `{}`; `_pick_qualifier`/`_TYPE_TAIL` read `m.type` via `getattr`. A
   typeless instrument degrades to a bare `<Month YYYY>` label.
   *Tests:* `test_ledger__label.py::TestMissingTermsAttrBug`,
   `::TestResolveSlotTermsNonDict`, `::TestBuildLabelMissingTypeAttr`.

## Logic — behavior contradicted the code's own comment/docstring — FIXED

6. **`cash_history._latest_quarterly_opcf`** — the tie-break comment said
   "prefer longer (FY) on tie", but the sort key `(end, -days)` + `[-1]`
   picked the **shorter** period. **Fix:** sort key is now `(end, days)`.
   *Test:* `test_cash_history.py::TestLatestQuarterlyOpcf::test_tie_on_end_prefers_longer_fy`.

7. **`shelf_status.derive_shelf_status`** — two issues. (a) The capacity
   filter `(remaining_capacity_usd or capacity_usd)` treated `0` as falsy,
   so a fully-drawn (`remaining == 0`) shelf fell through to the registered
   `capacity_usd` and was kept. (b) An `RW` predating `filing_date` was
   correctly **not** marked withdrawn, yet the output still echoed a
   `withdrawal_date`. **Fix:** remaining is authoritative when present (a
   literal 0 drops the shelf); `withdrawal_date` is now populated iff
   `derived_status == 'withdrawn'`. (Verified no downstream consumer reads
   the field — cards.py only reads `derived_status`/`effect_date`.) The same
   un-honored-RW echo (b) was mirrored-fixed in the sibling **`s1_status.py`**
   so both mirrors share the "present iff withdrawn" contract.
   *Tests:* `test_ledger_shelf_status.py::TestCapacityFiltering`,
   `::TestWithdrawn::test_rw_filed_before_filing_date_does_not_withdraw`,
   `test_ledger_s1_status.py::TestWithdrawn::test_rw_strictly_before_filing_date_not_withdrawn`.

8. **`unit_detection._llm_ads_ratio`** — `NaN` slipped the plausibility
   guard: both `nan <= 0` and `nan > 100_000` are `False`, so a `NaN` ratio
   was returned (unlike `Infinity`) and would feed `unit_preamble()`.
   **Fix:** added a `math.isfinite` check. *Test:*
   `test_unit_detection.py::TestLlmAdsRatio::test_nan_is_rejected`.

9. **`unit_detection`** heuristic branch-3 — the docstring example
   ("1:100 to 1:400") is the `1:N` direction, but the regex only matched the
   trailing `:1` (`N:1`) direction. **Fix:** the regex now matches either
   `1:N` or `N:1` and captures the non-unit number. The `1:N` arm requires a
   **literal colon** (not the loose `[:\s]` separator) — adversarial review
   showed a loose separator let distractor MD&A prose ("the coverage ratio
   is 1 3 of EBITDA") yield a bogus ratio that, appended last, overrode the
   correct ADS ratio. (Free-form supersession *prose* with no "ratio is/of N"
   structure still falls to the LLM.)
   *Tests:* `test_unit_detection.py::TestHeuristicAdsRatio::test_branch3_1_colon_N_capture`,
   `::test_branch3_1_space_N_distractor_not_captured`,
   `::test_branch3_1_space_N_distractor_does_not_override_real_ratio`,
   `::test_branch3_freeform_supersession_prose_still_falls_to_llm`.

10. **`view._bucket`** (`dilution/ledger/view.py`) — the actives sort key
    `(type or '', created_at or '', instrument_id)` lacked a `None`-fallback
    on the third element, so two active rows tying on `(type, created_at)`
    with a `None` `instrument_id` raised on the tuple comparison. **Fix:**
    added `or ""` (None sorts first). *Test:*
    `test_ledger_view.py::test_none_instrument_id_sorts_first_no_crash`.

## Minor / cosmetic / latent — OPEN (characterization tests)

11. **`_label.build_label`** — a whitespace-only `descriptor` on
    `type=='equity'` bypasses the "Equity Issuance" default (`m.descriptor
    or default` keeps truthy whitespace), yielding a label with trailing
    whitespace. *(LLM emits clean descriptors in practice.)*

12. **`mutations.create_from_dict`** — docstring claims "ValueError if a
    required numeric field is missing", but `float(terms.get('strike') or
    0.0)` silently defaults to `0.0` instead of raising.

13. **`_counterparty_tiers._norm`** — the `if not name` guard only catches
    falsy input; a truthy non-`str` (int/list/dict) would reach the string
    ops and raise. Never fires in production (callers pass strings).

14. **`store._preferred_price_split_skip`** — the discriminator for "should a
    common split adjust this preferred's conv_price?" is *presence of
    `conversion_ratio`* (absent ⇒ price-based ⇒ adjust). This is correct for
    every preferred that carries an applied split today, but it is a proxy,
    not the true signal (which lives only in filing prose: did the filer
    retroactively split-adjust the conversion price?). A fixed-RATIO series
    that stores a conv_price but NOT a `conversion_ratio` — e.g. SCNI `EIB`
    (P-439, conv_price 93.41 == $34,000 / 364, a derived reference) — would be
    *wrongly* divided if it ever took a split. None such carries an applied
    split now (SCNI has no split in-window), so there is no current eval
    impact. **Durable fix:** have the walker stamp `conversion_ratio` on such
    series, at which point this guard protects them. *Pinned by:*
    `test_ledger_preferred_conv_price_split.py` (the ratio-present cases show
    a stored ratio is what protects a series).
