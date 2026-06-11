"""Unit tests for dilution/ledger/s1_status.py :: derive_s1_status.

derive_s1_status is a single db_backed status machine. The autouse
``temp_db`` fixture from conftest reroutes ``db.get_conn()`` to a fresh
per-test SQLite DB, so we just stage rows and call the function — no
monkeypatching of any seam is required (there is no network/LLM/fs seam).

Determinism: every lapse-sensitive test passes an explicit ``today=``
(a ``datetime.date``). ``today`` also accepts a ``datetime.datetime`` or
an ISO string — ``_coerce_date`` normalizes all three (see TestTodayParam).
"""

from __future__ import annotations

import datetime
import json

import pytest

from dilution.ledger.s1_status import (
    S1_FORM_PREFIXES,
    S1_LAPSE_DAYS,
    derive_s1_status,
)

CIK = 100


# ── helpers ──────────────────────────────────────────────────────────
def _stage_s1(
    temp_db,
    *,
    cik: int = CIK,
    instrument_id: str = "S1-1",
    created_accession: str = "acc-s1",
    created_at: str = "2024-01-01",
    filing_form: str | None = "S-1",
    filing_date: str = "2024-01-01",
    file_number: str | None = "333-100000",
    terms: dict | None = None,
    outstanding: dict | None = None,
    status: str = "active",
    add_company: bool = True,
    add_filing: bool = True,
):
    """Stage a company + (optionally) an S-1 base filing + the
    s1_offering ledger row. Returns the instrument_id."""
    if add_company:
        # add_company tolerates being called once; guard for multi-instrument
        try:
            temp_db.add_company(cik, "AAA", "A Co")
        except Exception:
            pass
    if add_filing and filing_form is not None:
        temp_db.add_filing(
            created_accession,
            cik,
            form=filing_form,
            filing_date=filing_date,
            file_number=file_number,
        )
    temp_db.add_instrument(
        instrument_id,
        cik=cik,
        type="s1_offering",
        created_at=created_at,
        created_accession=created_accession,
        terms_json=json.dumps(terms or {}),
        outstanding_json=json.dumps(outstanding or {}),
        status=status,
    )
    return instrument_id


def _only(temp_db, **kw):
    """Stage one s1_offering and return its single derived dict."""
    today = kw.pop("today", datetime.date(2024, 6, 1))
    _stage_s1(temp_db, **kw)
    res = derive_s1_status(kw.get("cik", CIK), today=today)
    assert len(res) == 1
    return res[0]


# ── empty / type-filtering ───────────────────────────────────────────
class TestNoRows:
    def test_no_rows_for_cik_returns_empty(self, temp_db):
        temp_db.add_company(CIK)
        assert derive_s1_status(CIK, today=datetime.date(2024, 6, 1)) == []

    def test_unknown_cik_returns_empty(self, temp_db):
        # nothing staged at all
        assert derive_s1_status(99999, today=datetime.date(2024, 6, 1)) == []

    def test_non_s1_types_are_excluded(self, temp_db):
        temp_db.add_company(CIK)
        temp_db.add_filing("acc-w", CIK, form="S-1",
                           filing_date="2024-01-01", file_number="333-1")
        # warrant + shelf + atm rows must all be ignored
        temp_db.add_instrument("W-1", cik=CIK, type="warrant",
                               created_accession="acc-w")
        temp_db.add_instrument("SH-1", cik=CIK, type="shelf",
                               created_accession="acc-w")
        temp_db.add_instrument("ATM-1", cik=CIK, type="atm",
                               created_accession="acc-w")
        assert derive_s1_status(CIK, today=datetime.date(2024, 6, 1)) == []

    def test_only_s1_offering_selected_among_mixed(self, temp_db):
        temp_db.add_company(CIK)
        temp_db.add_filing("acc-s1", CIK, form="S-1",
                           filing_date="2024-01-01", file_number="333-1")
        temp_db.add_instrument("W-1", cik=CIK, type="warrant",
                               created_accession="acc-s1")
        temp_db.add_instrument("S1-1", cik=CIK, type="s1_offering",
                               created_accession="acc-s1")
        res = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))
        assert [d["instrument_id"] for d in res] == ["S1-1"]


# ── pending baseline ─────────────────────────────────────────────────
class TestPending:
    def test_pure_pending(self, temp_db):
        d = _only(temp_db)
        assert d["derived_status"] == "pending"
        assert d["effect_date"] is None
        assert d["withdrawal_date"] is None
        assert d["first_drawdown_date"] is None

    def test_output_shape_and_echoed_fields(self, temp_db):
        d = _only(
            temp_db,
            terms={"anticipated_deal_size": 7_500_000, "final_deal_size": None},
            status="active",
        )
        assert d["instrument_id"] == "S1-1"
        assert d["accession_number"] == "acc-s1"
        assert d["form"] == "S-1"
        assert d["file_number"] == "333-100000"
        assert d["filing_date"] == "2024-01-01"
        assert d["anticipated_amount_usd"] == 7_500_000
        assert d["final_amount_usd"] is None
        assert d["reported_status"] == "active"
        assert d["derived_status"] == "pending"

    def test_form_is_uppercased_in_output(self, temp_db):
        d = _only(temp_db, filing_form="s-1")
        assert d["form"] == "S-1"
        # lowercase s-1 still treated as a registration filing → file_number honored
        assert d["file_number"] == "333-100000"
        assert d["derived_status"] == "pending"


# ── effective ────────────────────────────────────────────────────────
class TestEffective:
    def test_effect_under_matching_file_number(self, temp_db):
        _stage_s1(temp_db, file_number="333-100000")
        temp_db.add_filing("acc-eff", CIK, form="EFFECT",
                           filing_date="2024-03-01", file_number="333-100000")
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        assert d["derived_status"] == "effective"
        assert d["effect_date"] == "2024-03-01"

    def test_effect_form_like_prefix_match(self, temp_db):
        # SEC sometimes files 'EFFECT' variations; query uses LIKE 'EFFECT%'
        _stage_s1(temp_db, file_number="333-100000")
        temp_db.add_filing("acc-eff", CIK, form="EFFECTAMENDMENT",
                           filing_date="2024-03-01", file_number="333-100000")
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        assert d["effect_date"] == "2024-03-01"
        assert d["derived_status"] == "effective"

    def test_effect_under_different_file_number_not_matched(self, temp_db):
        _stage_s1(temp_db, file_number="333-100000")
        temp_db.add_filing("acc-eff", CIK, form="EFFECT",
                           filing_date="2024-03-01", file_number="333-999999")
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        assert d["effect_date"] is None
        assert d["derived_status"] == "pending"

    def test_effect_with_null_file_number_excluded(self, temp_db):
        _stage_s1(temp_db, file_number="333-100000")
        temp_db.add_filing("acc-eff", CIK, form="EFFECT",
                           filing_date="2024-03-01", file_number=None)
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        assert d["effect_date"] is None
        assert d["derived_status"] == "pending"

    def test_earliest_effect_wins_among_multiple(self, temp_db):
        _stage_s1(temp_db, file_number="333-100000")
        temp_db.add_filing("acc-eff2", CIK, form="EFFECT",
                           filing_date="2024-05-01", file_number="333-100000")
        temp_db.add_filing("acc-eff1", CIK, form="EFFECT",
                           filing_date="2024-03-01", file_number="333-100000")
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        assert d["effect_date"] == "2024-03-01"  # earliest by filing_date


# ── priced ───────────────────────────────────────────────────────────
class TestPriced:
    def test_first_drawdown_yields_priced(self, temp_db):
        _stage_s1(temp_db)
        temp_db.add_drawdown("S1-1", cik=CIK, event_date="2024-07-01",
                             amount_usd=1_000_000)
        d = derive_s1_status(CIK, today=datetime.date(2024, 8, 1))[0]
        assert d["derived_status"] == "priced"
        assert d["first_drawdown_date"] == "2024-07-01"

    def test_first_drawdown_date_is_min_event_date(self, temp_db):
        _stage_s1(temp_db)
        temp_db.add_drawdown("S1-1", cik=CIK, event_date="2024-09-01",
                             amount_usd=2_000_000)
        temp_db.add_drawdown("S1-1", cik=CIK, event_date="2024-07-15",
                             amount_usd=1_000_000)
        d = derive_s1_status(CIK, today=datetime.date(2024, 10, 1))[0]
        assert d["first_drawdown_date"] == "2024-07-15"  # MIN()
        assert d["derived_status"] == "priced"

    @pytest.mark.parametrize("term_key", [
        "final_deal_size",
        "final_pricing",
        "final_shares_offered",
    ])
    def test_terms_pricing_evidence(self, temp_db, term_key):
        d = _only(temp_db, terms={term_key: 12345})
        assert d["derived_status"] == "priced"
        assert d["first_drawdown_date"] is None

    def test_final_deal_size_echoed_as_final_amount_usd(self, temp_db):
        # final_amount_usd echoes terms['final_deal_size'] verbatim (not just
        # the truthiness used for the priced decision).
        d = _only(temp_db, terms={"final_deal_size": 9_250_000})
        assert d["final_amount_usd"] == 9_250_000
        assert d["derived_status"] == "priced"

    def test_final_pricing_does_not_populate_final_amount_usd(self, temp_db):
        # only final_deal_size feeds final_amount_usd; final_pricing prices the
        # deal but leaves final_amount_usd None.
        d = _only(temp_db, terms={"final_pricing": 1.25})
        assert d["derived_status"] == "priced"
        assert d["final_amount_usd"] is None

    @pytest.mark.parametrize("out_key", [
        "sold_to_date",
        "priced_amount_usd",
        "drawn_usd",
    ])
    def test_outstanding_pricing_evidence(self, temp_db, out_key):
        d = _only(temp_db, outstanding={out_key: 555})
        assert d["derived_status"] == "priced"

    def test_zero_valued_pricing_terms_are_falsy_not_priced(self, temp_db):
        # bool(0) is False → no pricing evidence
        d = _only(temp_db, terms={"final_deal_size": 0},
                  outstanding={"sold_to_date": 0})
        assert d["derived_status"] == "pending"

    def test_priced_beats_lapsed(self, temp_db):
        # very old filing WITH pricing evidence → priced, not lapsed
        _stage_s1(temp_db, filing_date="2018-01-01", created_at="2018-01-01",
                  terms={"final_deal_size": 5_000_000})
        d = derive_s1_status(CIK, today=datetime.date(2026, 1, 1))[0]
        assert d["derived_status"] == "priced"

    def test_priced_beats_effective(self, temp_db):
        _stage_s1(temp_db, file_number="333-100000")
        temp_db.add_filing("acc-eff", CIK, form="EFFECT",
                           filing_date="2024-03-01", file_number="333-100000")
        temp_db.add_drawdown("S1-1", cik=CIK, event_date="2024-07-01",
                             amount_usd=1_000_000)
        d = derive_s1_status(CIK, today=datetime.date(2024, 8, 1))[0]
        assert d["effect_date"] == "2024-03-01"
        assert d["derived_status"] == "priced"


# ── withdrawn ────────────────────────────────────────────────────────
class TestWithdrawn:
    def test_rw_after_filing_date_yields_withdrawn(self, temp_db):
        _stage_s1(temp_db, filing_date="2024-01-01", file_number="333-100000")
        temp_db.add_filing("acc-rw", CIK, form="RW",
                           filing_date="2024-04-01", file_number="333-100000")
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        assert d["derived_status"] == "withdrawn"
        assert d["withdrawal_date"] == "2024-04-01"

    def test_rw_exactly_equal_to_filing_date_is_withdrawn(self, temp_db):
        # boundary: withdrawal_date >= filing_date uses '>='
        _stage_s1(temp_db, filing_date="2024-01-01", file_number="333-100000")
        temp_db.add_filing("acc-rw", CIK, form="RW",
                           filing_date="2024-01-01", file_number="333-100000")
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        assert d["derived_status"] == "withdrawn"

    def test_rw_strictly_before_filing_date_not_withdrawn(self, temp_db):
        # stale/unrelated RW (predates the filing) is ignored → falls through
        _stage_s1(temp_db, filing_date="2024-01-01", file_number="333-100000")
        temp_db.add_filing("acc-rw", CIK, form="RW",
                           filing_date="2023-12-31", file_number="333-100000")
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        # An un-honored RW (predating filing) is no longer echoed — like
        # derive_shelf_status, withdrawal_date is present iff withdrawn.
        assert d["withdrawal_date"] is None
        assert d["derived_status"] == "pending"

    def test_withdrawn_beats_priced(self, temp_db):
        # stage BOTH a drawdown AND an RW → withdrawn wins
        _stage_s1(temp_db, filing_date="2024-01-01", file_number="333-100000")
        temp_db.add_drawdown("S1-1", cik=CIK, event_date="2024-02-01",
                             amount_usd=1_000_000)
        temp_db.add_filing("acc-rw", CIK, form="RW",
                           filing_date="2024-04-01", file_number="333-100000")
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        assert d["first_drawdown_date"] == "2024-02-01"
        assert d["derived_status"] == "withdrawn"

    def test_rw_under_different_file_number_not_matched(self, temp_db):
        _stage_s1(temp_db, file_number="333-100000")
        temp_db.add_filing("acc-rw", CIK, form="RW",
                           filing_date="2024-04-01", file_number="333-222222")
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        assert d["withdrawal_date"] is None
        assert d["derived_status"] == "pending"

    def test_rw_with_null_file_number_excluded(self, temp_db):
        _stage_s1(temp_db, file_number="333-100000")
        temp_db.add_filing("acc-rw", CIK, form="RW",
                           filing_date="2024-04-01", file_number=None)
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        assert d["withdrawal_date"] is None
        assert d["derived_status"] == "pending"

    def test_earliest_rw_wins_among_multiple(self, temp_db):
        _stage_s1(temp_db, filing_date="2024-01-01", file_number="333-100000")
        temp_db.add_filing("acc-rw2", CIK, form="RW",
                           filing_date="2024-06-01", file_number="333-100000")
        temp_db.add_filing("acc-rw1", CIK, form="RW",
                           filing_date="2024-04-01", file_number="333-100000")
        d = derive_s1_status(CIK, today=datetime.date(2024, 8, 1))[0]
        assert d["withdrawal_date"] == "2024-04-01"
        assert d["derived_status"] == "withdrawn"


# ── lapse / calendar ─────────────────────────────────────────────────
class TestLapse:
    def test_s1_lapse_days_constant(self):
        assert S1_LAPSE_DAYS == 730

    @pytest.mark.parametrize("age_days,expected", [
        (0, "pending"),
        (729, "pending"),
        (730, "pending"),   # strict '>' → 730 is NOT lapsed
        (731, "lapsed"),    # first day past the boundary
        (1000, "lapsed"),
    ])
    def test_lapse_boundary_sweep(self, temp_db, age_days, expected):
        fd = datetime.date(2024, 1, 1)
        _stage_s1(temp_db, filing_date=fd.isoformat(),
                  created_at=fd.isoformat())
        today = fd + datetime.timedelta(days=age_days)
        d = derive_s1_status(CIK, today=today)[0]
        assert d["derived_status"] == expected

    def test_effective_loses_to_lapsed(self, temp_db):
        # old filing (>730d) with EFFECT but no pricing → lapsed
        # (lapsed checked before effective in precedence)
        fd = datetime.date(2022, 1, 1)
        _stage_s1(temp_db, filing_date=fd.isoformat(),
                  created_at=fd.isoformat(), file_number="333-100000")
        temp_db.add_filing("acc-eff", CIK, form="EFFECT",
                           filing_date="2022-03-01", file_number="333-100000")
        d = derive_s1_status(CIK, today=datetime.date(2026, 1, 1))[0]
        assert d["effect_date"] == "2022-03-01"  # still echoed
        assert d["derived_status"] == "lapsed"

    def test_malformed_filing_date_falls_back_age_zero(self, temp_db):
        # non-ISO filing_date → except clause sets age_days=0 → not lapsed
        _stage_s1(temp_db, filing_date="not-a-date")
        d = derive_s1_status(CIK, today=datetime.date(2030, 1, 1))[0]
        assert d["filing_date"] == "not-a-date"
        assert d["derived_status"] == "pending"

    def test_timestamp_style_filing_date_lapses(self, temp_db):
        # FIXED (was bug A#4): a filing_date carrying a time component (e.g.
        # the schema default '2026-01-01T00:00:00Z') is now trimmed to its
        # date head by _coerce_date, so a years-old filing correctly lapses.
        _stage_s1(temp_db, filing_date="2018-01-01T00:00:00Z",
                  created_at="2018-01-01T00:00:00Z")
        d = derive_s1_status(CIK, today=datetime.date(2030, 1, 1))[0]
        assert d["filing_date"] == "2018-01-01T00:00:00Z"
        assert d["derived_status"] == "lapsed"


# ── 8-K seeded branch (file_number forced None) ──────────────────────
class TestEightKSeeded:
    def test_8k_seeded_ignores_effect_and_rw(self, temp_db):
        # s1_offering whose created_accession is an 8-K. Even though an
        # EFFECT and RW exist under the 8-K's (Exchange Act) file_number,
        # file_number is forced None → EFFECT/RW lookups are skipped.
        temp_db.add_company(CIK)
        temp_db.add_filing("acc-8k", CIK, form="8-K",
                           filing_date="2024-01-01", file_number="001-39999")
        temp_db.add_filing("acc-eff", CIK, form="EFFECT",
                           filing_date="2024-03-01", file_number="001-39999")
        temp_db.add_filing("acc-rw", CIK, form="RW",
                           filing_date="2024-04-01", file_number="001-39999")
        temp_db.add_instrument("S1-8k", cik=CIK, type="s1_offering",
                               created_at="2024-01-01",
                               created_accession="acc-8k")
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        assert d["form"] == "8-K"
        assert d["file_number"] is None
        assert d["effect_date"] is None
        assert d["withdrawal_date"] is None
        assert d["derived_status"] == "pending"

    def test_8k_seeded_still_priced_by_drawdown(self, temp_db):
        # 8-K-seeded falls back to drawdown + calendar evidence
        temp_db.add_company(CIK)
        temp_db.add_filing("acc-8k", CIK, form="8-K",
                           filing_date="2024-01-01", file_number="001-39999")
        temp_db.add_instrument("S1-8k", cik=CIK, type="s1_offering",
                               created_at="2024-01-01",
                               created_accession="acc-8k")
        temp_db.add_drawdown("S1-8k", cik=CIK, event_date="2024-02-01",
                             amount_usd=900_000)
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        assert d["file_number"] is None
        assert d["first_drawdown_date"] == "2024-02-01"
        assert d["derived_status"] == "priced"

    def test_8k_seeded_can_lapse_on_calendar(self, temp_db):
        temp_db.add_company(CIK)
        temp_db.add_filing("acc-8k", CIK, form="8-K",
                           filing_date="2022-01-01", file_number="001-39999")
        temp_db.add_instrument("S1-8k", cik=CIK, type="s1_offering",
                               created_at="2022-01-01",
                               created_accession="acc-8k")
        d = derive_s1_status(CIK, today=datetime.date(2026, 1, 1))[0]
        assert d["file_number"] is None
        assert d["derived_status"] == "lapsed"


# ── filing-meta fallbacks ────────────────────────────────────────────
class TestFilingMetaFallback:
    def test_missing_filing_meta_uses_terms_form_and_created_at(self, temp_db):
        # created_accession has NO matching dilution_filings row.
        # form falls back to terms.get('form'); filing_date to created_at;
        # file_number forced None (no meta).
        temp_db.add_company(CIK)
        temp_db.add_instrument(
            "S1-m", cik=CIK, type="s1_offering",
            created_at="2024-02-15", created_accession="missing-acc",
            terms_json=json.dumps({"form": "s-1",
                                    "anticipated_deal_size": 5_000_000}),
        )
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        assert d["form"] == "S-1"  # from terms, uppercased
        assert d["filing_date"] == "2024-02-15"  # fallback to created_at
        assert d["file_number"] is None
        assert d["anticipated_amount_usd"] == 5_000_000
        assert d["derived_status"] == "pending"

    def test_missing_meta_and_no_terms_form_yields_empty_form(self, temp_db):
        temp_db.add_company(CIK)
        temp_db.add_instrument(
            "S1-m", cik=CIK, type="s1_offering",
            created_at="2024-02-15", created_accession="missing-acc",
            terms_json="{}",
        )
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        assert d["form"] == ""
        assert d["file_number"] is None
        assert d["derived_status"] == "pending"


# ── NULL / empty JSON tolerance ──────────────────────────────────────
class TestJsonTolerance:
    def test_empty_string_json_columns_no_crash(self, temp_db):
        temp_db.add_company(CIK)
        temp_db.add_filing("acc-s1", CIK, form="S-1",
                           filing_date="2024-01-01", file_number="333-100000")
        # terms_json / outstanding_json as '' → `(value or "{}")` path
        temp_db.add_instrument("S1-1", cik=CIK, type="s1_offering",
                               created_at="2024-01-01",
                               created_accession="acc-s1",
                               terms_json="", outstanding_json="")
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        assert d["anticipated_amount_usd"] is None
        assert d["final_amount_usd"] is None
        assert d["derived_status"] == "pending"


# ── F-1 prefix ───────────────────────────────────────────────────────
class TestF1Prefix:
    def test_f1_form_honors_file_number(self, temp_db):
        assert S1_FORM_PREFIXES == ("S-1", "F-1")
        _stage_s1(temp_db, filing_form="F-1", file_number="333-100000")
        temp_db.add_filing("acc-eff", CIK, form="EFFECT",
                           filing_date="2024-03-01", file_number="333-100000")
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        assert d["form"] == "F-1"
        assert d["file_number"] == "333-100000"
        assert d["derived_status"] == "effective"

    def test_f1_amendment_form_prefix(self, temp_db):
        # 'F-1/A' starts with 'F-1' → treated as registration filing
        _stage_s1(temp_db, filing_form="F-1/A", file_number="333-100000")
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        assert d["form"] == "F-1/A"
        assert d["file_number"] == "333-100000"


# ── multi-instrument: ordering + no cross contamination ──────────────
class TestMultipleInstruments:
    def test_ordered_by_created_at(self, temp_db):
        temp_db.add_company(CIK)
        temp_db.add_filing("acc-b", CIK, form="S-1",
                           filing_date="2024-02-01", file_number="333-2")
        temp_db.add_filing("acc-a", CIK, form="S-1",
                           filing_date="2024-01-01", file_number="333-1")
        # insert out of created_at order
        temp_db.add_instrument("S1-late", cik=CIK, type="s1_offering",
                               created_at="2024-02-01",
                               created_accession="acc-b")
        temp_db.add_instrument("S1-early", cik=CIK, type="s1_offering",
                               created_at="2024-01-01",
                               created_accession="acc-a")
        res = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))
        assert [d["instrument_id"] for d in res] == ["S1-early", "S1-late"]

    def test_drawdown_grouping_not_cross_contaminated(self, temp_db):
        temp_db.add_company(CIK)
        temp_db.add_filing("acc-a", CIK, form="S-1",
                           filing_date="2024-01-01", file_number="333-1")
        temp_db.add_filing("acc-b", CIK, form="S-1",
                           filing_date="2024-01-02", file_number="333-2")
        temp_db.add_instrument("S1-a", cik=CIK, type="s1_offering",
                               created_at="2024-01-01",
                               created_accession="acc-a")
        temp_db.add_instrument("S1-b", cik=CIK, type="s1_offering",
                               created_at="2024-01-02",
                               created_accession="acc-b")
        # only S1-a has a drawdown
        temp_db.add_drawdown("S1-a", cik=CIK, event_date="2024-03-01",
                             amount_usd=1_000_000)
        res = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))
        by_id = {d["instrument_id"]: d for d in res}
        assert by_id["S1-a"]["first_drawdown_date"] == "2024-03-01"
        assert by_id["S1-a"]["derived_status"] == "priced"
        assert by_id["S1-b"]["first_drawdown_date"] is None
        assert by_id["S1-b"]["derived_status"] == "pending"

    def test_cik_isolation(self, temp_db):
        # drawdown / EFFECT for a different cik must not bleed in
        temp_db.add_company(CIK)
        temp_db.add_company(200, ticker="BBB", name="B Co")
        temp_db.add_filing("acc-s1", CIK, form="S-1",
                           filing_date="2024-01-01", file_number="333-1")
        temp_db.add_instrument("S1-1", cik=CIK, type="s1_offering",
                               created_at="2024-01-01",
                               created_accession="acc-s1")
        # EFFECT under cik 200 with the SAME file_number — must not match
        temp_db.add_filing("acc-eff-other", 200, form="EFFECT",
                           filing_date="2024-03-01", file_number="333-1")
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        assert d["effect_date"] is None
        assert d["derived_status"] == "pending"

    def test_rw_cik_isolation(self, temp_db):
        # the RW lookup is also cik-scoped: an RW under another cik with the
        # SAME file_number must not withdraw our registration.
        temp_db.add_company(CIK)
        temp_db.add_company(200, ticker="BBB", name="B Co")
        temp_db.add_filing("acc-s1", CIK, form="S-1",
                           filing_date="2024-01-01", file_number="333-1")
        temp_db.add_instrument("S1-1", cik=CIK, type="s1_offering",
                               created_at="2024-01-01",
                               created_accession="acc-s1")
        temp_db.add_filing("acc-rw-other", 200, form="RW",
                           filing_date="2024-04-01", file_number="333-1")
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        assert d["withdrawal_date"] is None
        assert d["derived_status"] == "pending"

    def test_drawdown_cik_isolation(self, temp_db):
        # the drawdown query carries an explicit WHERE cik=? guard; a drawdown
        # booked under another cik must not price our registration. (instrument_id
        # is a global PRIMARY KEY so the two ciks must use distinct ids — the
        # cik guard is what this test pins.)
        temp_db.add_company(CIK)
        temp_db.add_company(200, ticker="BBB", name="B Co")
        temp_db.add_filing("acc-s1", CIK, form="S-1",
                           filing_date="2024-01-01", file_number="333-1")
        temp_db.add_instrument("S1-1", cik=CIK, type="s1_offering",
                               created_at="2024-01-01",
                               created_accession="acc-s1")
        temp_db.add_filing("acc-other", 200, form="S-1",
                           filing_date="2024-01-01", file_number="333-9")
        temp_db.add_instrument("S1-other", cik=200, type="s1_offering",
                               created_at="2024-01-01",
                               created_accession="acc-other")
        temp_db.add_drawdown("S1-other", cik=200, event_date="2024-02-01",
                             amount_usd=1_000_000)
        d = derive_s1_status(CIK, today=datetime.date(2024, 6, 1))[0]
        assert d["instrument_id"] == "S1-1"
        assert d["first_drawdown_date"] is None
        assert d["derived_status"] == "pending"


# ── today= param semantics ───────────────────────────────────────────
class TestTodayParam:
    def test_explicit_date_object(self, temp_db):
        fd = datetime.date(2024, 1, 1)
        _stage_s1(temp_db, filing_date=fd.isoformat(),
                  created_at=fd.isoformat())
        # well past the lapse window
        d = derive_s1_status(CIK, today=datetime.date(2027, 1, 1))[0]
        assert d["derived_status"] == "lapsed"

    def test_default_today_uses_wall_clock(self, temp_db):
        # passing today=None → uses _d.today(); a filing from far in the
        # past must lapse regardless of the actual wall-clock date.
        _stage_s1(temp_db, filing_date="2000-01-01", created_at="2000-01-01")
        d = derive_s1_status(CIK)[0]  # no today=
        assert d["derived_status"] == "lapsed"

    def test_equivalent_date_does_lapse(self, temp_db):
        # Contrast: the same instant as a plain date DOES lapse.
        fd = datetime.date(2024, 1, 1)
        _stage_s1(temp_db, filing_date=fd.isoformat(),
                  created_at=fd.isoformat())
        d = derive_s1_status(CIK, today=datetime.date(2027, 1, 1))[0]
        assert d["derived_status"] == "lapsed"

    def test_datetime_today_priced_branch(self, temp_db):
        # A datetime today is normalized by _coerce_date; pricing evidence
        # still resolves to 'priced' (the priced branch precedes the lapse
        # check). Kept as a cross-check on the priced precedence.
        fd = datetime.date(2018, 1, 1)
        _stage_s1(temp_db, filing_date=fd.isoformat(),
                  created_at=fd.isoformat(),
                  terms={"final_deal_size": 5_000_000})
        d = derive_s1_status(CIK, today=datetime.datetime(2027, 1, 1, 9, 0))[0]
        assert d["derived_status"] == "priced"

    def test_datetime_today_lapses(self, temp_db):
        # FIXED (was bug A#4): a datetime today is normalized to its date by
        # _coerce_date, so a 3-year-old filing correctly lapses.
        fd = datetime.date(2024, 1, 1)
        _stage_s1(temp_db, filing_date=fd.isoformat(),
                  created_at=fd.isoformat())
        d = derive_s1_status(CIK, today=datetime.datetime(2027, 1, 1, 12, 30))[0]
        assert d["derived_status"] == "lapsed"

    def test_string_today_is_accepted(self, temp_db):
        # FIXED (was bug A#4): a raw ISO string today is now parsed by
        # _coerce_date instead of raising AttributeError. The default-staged
        # S-1 (filed 2024-01-01) is ~5 months old under a 2024-06-01 today,
        # so it is still pending (not yet lapsed).
        _stage_s1(temp_db)
        d = derive_s1_status(CIK, today="2024-06-01")[0]
        assert d["derived_status"] == "pending"
