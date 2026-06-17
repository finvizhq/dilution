"""s1_offering warrant_coverage_pct fallback (drift triage 2026-06-17).

When the walker captured only the FINAL (priced) warrant coverage and not
the anticipated one, the anticipated `warrant_coverage_pct` falls back to
the final value (GCTK S1-217: final 2.0 stored, anticipated null -> render
200). The fallback is an explicit None-check, NOT `or`: a legitimate 0.0
(no warrant coverage) must render 0, never be overridden by the final
value (S1-042/053/089 store 0.0).
"""

import json

from dilution.ledger.cards import s1_offering_cards


def _s1(temp_db, iid, terms, *, cik=901):
    acc = f"acc-{iid}"
    temp_db.add_instrument(
        iid, cik=cik, type="s1_offering", status="active",
        created_accession=acc, created_at="2025-06-01",
        terms_json=json.dumps(terms),
        outstanding_json=json.dumps({"sold_to_date": 5_000_000}),
    )
    temp_db.add_filing(acc, cik=cik, form="424B4")
    return s1_offering_cards(cik)[0]


def test_anticipated_coverage_falls_back_to_final(temp_db):
    temp_db.add_company(901, "S1P")
    card = _s1(temp_db, "S1-FB", {
        "anticipated_deal_size": 10_000_000,
        "final_warrant_coverage_pct": 2.0,   # 200% priced; anticipated null
    })
    assert card["warrant_coverage_pct"] == 200.0
    assert card["final_warrant_coverage_pct"] == 200.0


def test_zero_anticipated_coverage_preserved(temp_db):
    # 0.0 = no warrant coverage; must stay 0, NOT be overridden by final.
    temp_db.add_company(902, "S1Z")
    card = _s1(temp_db, "S1-ZERO", {
        "anticipated_deal_size": 10_000_000,
        "warrant_coverage_pct": 0.0,
        "final_warrant_coverage_pct": 2.0,
    }, cik=902)
    assert card["warrant_coverage_pct"] == 0.0


def test_both_absent_renders_none(temp_db):
    temp_db.add_company(903, "S1N")
    card = _s1(temp_db, "S1-NONE", {
        "anticipated_deal_size": 10_000_000,
    }, cik=903)
    assert card["warrant_coverage_pct"] is None
