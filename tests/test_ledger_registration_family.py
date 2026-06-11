"""Unit tests for dilution/ledger/registration_family.py.

All four functions are pure SQL over the autouse ``temp_db`` fixture
(conftest reroutes ``db.get_conn()`` to a fresh schema-initialized
SQLite DB). No network / LLM / filesystem seams exist, so no
monkeypatching is needed — tests just stage rows and call.

Staging recipe (the join semantics are the gotcha):
  * A PRIMARY shelf requires a coordinated PAIR: a filing AND a ledger
    row of type shelf/s1_offering whose ``created_accession`` ==
    filing.accession_number AND same cik.
  * A RESALE registration is a registration-form filing whose
    file_number has NO shelf/s1_offering ledger row.
"""

from __future__ import annotations

import pytest

from dilution.ledger import registration_family as rf


# ── helpers ───────────────────────────────────────────────────────────
def _stage_primary_shelf(
    temp_db,
    *,
    cik: int = 1,
    ticker: str = "TEST",
    shelf_acc: str = "S3-1",
    file_number: str = "333-100000",
    instrument_id: str = "SH-001",
    created_at: str = "2025-01-01",
    label: str | None = "Shelf 1",
    form: str = "S-3",
    filing_date: str = "2025-01-01",
    itype: str = "shelf",
):
    """Stage a coordinated (filing, shelf-ledger) pair = a primary shelf."""
    temp_db.add_filing(
        shelf_acc, cik, form=form, file_number=file_number,
        filing_date=filing_date,
    )
    temp_db.add_instrument(
        instrument_id, cik=cik, ticker=ticker, type=itype,
        created_at=created_at, created_accession=shelf_acc, label=label,
    )
    return instrument_id


# ──────────────────────────────────────────────────────────────────────
class TestModuleConstants:
    """Pin the form-code allowlists; SQL IN-clauses & callers depend on
    their exact contents."""

    def test_prescreen_forms_contents(self):
        assert rf.PRESCREEN_FORMS == ("424B2", "424B3", "424B5", "SUPPL")

    def test_first_time_s1_s3_absent_from_prescreen(self):
        # S-1 / S-3 (no /A) would self-classify resale if prescreened.
        assert "S-1" not in rf.PRESCREEN_FORMS
        assert "S-3" not in rf.PRESCREEN_FORMS

    def test_prescreen_excludes_424b4_and_424b7(self):
        # 424B4 IPO-final = always primary; 424B7 selling-holder = resale.
        assert "424B4" not in rf.PRESCREEN_FORMS
        assert "424B7" not in rf.PRESCREEN_FORMS

    def test_resale_propagation_forms_contents(self):
        assert rf.RESALE_PROPAGATION_FORMS == (
            "S-1/A", "S-3/A", "F-1/A", "F-3/A", "F-10/A", "POS AM",
        )

    def test_registration_forms_contents(self):
        assert rf._REGISTRATION_FORMS == (
            "S-1", "S-1/A", "S-3", "S-3/A", "S-3ASR",
            "S-3MEF", "F-1", "F-1/A", "F-3", "F-3/A",
            "F-3ASR", "F-3MEF",
            "F-10", "F-10/A", "F-10EF",
        )

    @pytest.mark.parametrize(
        "form", ["S-3ASR", "S-3MEF", "F-3ASR", "F-3MEF", "F-10", "F-10EF"],
    )
    def test_registration_forms_includes_asr_mef_ef_f10(self, form):
        assert form in rf._REGISTRATION_FORMS

    def test_pos_am_diverges_between_lists(self):
        # POS AM is a propagation form but NOT a registration-parent form.
        # Pin so a future merge of the two tuples is caught.
        assert "POS AM" in rf.RESALE_PROPAGATION_FORMS
        assert "POS AM" not in rf._REGISTRATION_FORMS

    def test_attribution_literal_values(self):
        # Sanity: the Attribution Literal documents the three verdicts.
        assert rf.Attribution.__args__ == ("primary", "resale", "unknown")


# ──────────────────────────────────────────────────────────────────────
class TestPrimaryRegistrationFileNumbers:

    def test_no_ledger_rows_returns_empty_set(self, temp_db):
        temp_db.add_company(cik=1)
        result = rf.primary_registration_file_numbers(1)
        assert result == set()
        assert isinstance(result, set)

    def test_shelf_pair_yields_file_number(self, temp_db):
        temp_db.add_company(cik=1)
        _stage_primary_shelf(temp_db, file_number="333-100000")
        assert rf.primary_registration_file_numbers(1) == {"333-100000"}

    def test_s1_offering_type_included(self, temp_db):
        temp_db.add_company(cik=1)
        _stage_primary_shelf(
            temp_db, file_number="333-111111", itype="s1_offering",
            shelf_acc="S1-acc", instrument_id="S1-001",
        )
        assert rf.primary_registration_file_numbers(1) == {"333-111111"}

    @pytest.mark.parametrize("noncore_type", ["warrant", "atm", "convertible",
                                              "preferred", "equity_line"])
    def test_noncore_ledger_type_excluded(self, temp_db, noncore_type):
        # A warrant/atm created under a 333 file_number is NOT a primary
        # registration, even though its created_accession carries a 333.
        temp_db.add_company(cik=1)
        temp_db.add_filing("F-acc", 1, form="424B5", file_number="333-100000")
        temp_db.add_instrument(
            "X-001", cik=1, type=noncore_type, created_accession="F-acc",
        )
        assert rf.primary_registration_file_numbers(1) == set()

    def test_created_accession_without_matching_filing_contributes_nothing(
        self, temp_db,
    ):
        # Inner join: ledger row whose created_accession has no filing row.
        temp_db.add_company(cik=1)
        temp_db.add_instrument(
            "SH-001", cik=1, type="shelf", created_accession="missing-acc",
        )
        assert rf.primary_registration_file_numbers(1) == set()

    def test_matching_filing_null_file_number_excluded(self, temp_db):
        temp_db.add_company(cik=1)
        temp_db.add_filing("S3-1", 1, form="S-3", file_number=None)
        temp_db.add_instrument(
            "SH-001", cik=1, type="shelf", created_accession="S3-1",
        )
        assert rf.primary_registration_file_numbers(1) == set()

    def test_non_333_file_number_excluded(self, temp_db):
        # 001- Exchange Act number filtered by LIKE '333-%'.
        temp_db.add_company(cik=1)
        temp_db.add_filing("S3-1", 1, form="S-3", file_number="001-39000")
        temp_db.add_instrument(
            "SH-001", cik=1, type="shelf", created_accession="S3-1",
        )
        assert rf.primary_registration_file_numbers(1) == set()

    def test_two_shelves_one_file_number_distinct_collapses(self, temp_db):
        # S-3 + S-3/A both create shelves under the same file_number.
        temp_db.add_company(cik=1)
        temp_db.add_filing("S3-1", 1, form="S-3", file_number="333-100000")
        temp_db.add_filing("S3-A", 1, form="S-3/A", file_number="333-100000")
        temp_db.add_instrument(
            "SH-001", cik=1, type="shelf", created_accession="S3-1",
        )
        temp_db.add_instrument(
            "SH-002", cik=1, type="shelf", created_accession="S3-A",
        )
        result = rf.primary_registration_file_numbers(1)
        assert result == {"333-100000"}
        assert len(result) == 1

    def test_multiple_distinct_file_numbers(self, temp_db):
        temp_db.add_company(cik=1)
        _stage_primary_shelf(
            temp_db, file_number="333-100000", shelf_acc="S3-1",
            instrument_id="SH-001",
        )
        _stage_primary_shelf(
            temp_db, file_number="333-200000", shelf_acc="S3-2",
            instrument_id="SH-002",
        )
        assert rf.primary_registration_file_numbers(1) == {
            "333-100000", "333-200000",
        }

    def test_cross_cik_isolation(self, temp_db):
        # cik 1 shelf, cik 2 shelf sharing same file_number — must not leak.
        temp_db.add_company(cik=1, ticker="AAA")
        temp_db.add_company(cik=2, ticker="BBB")
        _stage_primary_shelf(
            temp_db, cik=1, file_number="333-100000", shelf_acc="A-S3",
            instrument_id="A-SH",
        )
        _stage_primary_shelf(
            temp_db, cik=2, file_number="333-100000", shelf_acc="B-S3",
            instrument_id="B-SH", ticker="BBB",
        )
        assert rf.primary_registration_file_numbers(1) == {"333-100000"}
        assert rf.primary_registration_file_numbers(2) == {"333-100000"}

    def test_ledger_filing_cik_mismatch_excluded(self, temp_db):
        # Join requires f.cik = l.cik. A ledger row for cik 1 whose
        # created_accession only matches a filing under cik 2 must not join.
        temp_db.add_company(cik=1)
        temp_db.add_company(cik=2)
        temp_db.add_filing("S3-X", 2, form="S-3", file_number="333-100000")
        temp_db.add_instrument(
            "SH-001", cik=1, type="shelf", created_accession="S3-X",
        )
        # cik 1's ledger row's created_accession matches only a cik-2 filing.
        assert rf.primary_registration_file_numbers(1) == set()

    def test_mixed_rows_only_valid_333_shelf_survives(self, temp_db):
        # Three ledger rows for one cik: a valid 333 shelf, a warrant on a
        # 333 file_number (wrong type), and a shelf on a 001- file_number
        # (wrong prefix). Only the first should survive every filter.
        temp_db.add_company(cik=1)
        temp_db.add_filing("S3-1", 1, form="S-3", file_number="333-100000")
        temp_db.add_filing("W-acc", 1, form="424B5", file_number="333-222222")
        temp_db.add_filing("S3-001", 1, form="S-3", file_number="001-39999")
        temp_db.add_instrument("SH-1", cik=1, type="shelf",
                               created_accession="S3-1")
        temp_db.add_instrument("W-1", cik=1, type="warrant",
                               created_accession="W-acc")
        temp_db.add_instrument("SH-001", cik=1, type="shelf",
                               created_accession="S3-001")
        assert rf.primary_registration_file_numbers(1) == {"333-100000"}

    def test_closed_status_shelf_still_counted(self, temp_db):
        # The query has NO status filter: a closed/terminated shelf is still
        # a primary registration the issuer used. Pin this behavior.
        temp_db.add_company(cik=1)
        temp_db.add_filing("S3-1", 1, form="S-3", file_number="333-100000")
        temp_db.add_instrument(
            "SH-001", cik=1, type="shelf", created_accession="S3-1",
            status="closed",
        )
        assert rf.primary_registration_file_numbers(1) == {"333-100000"}


# ──────────────────────────────────────────────────────────────────────
class TestClassify424bAttribution:

    def test_accession_absent_returns_unknown(self, temp_db):
        temp_db.add_company(cik=1)
        _stage_primary_shelf(temp_db)  # primary_set non-empty
        assert rf.classify_424b_attribution(1, "nonexistent") == "unknown"

    def test_filing_with_null_file_number_returns_unknown(self, temp_db):
        temp_db.add_company(cik=1)
        _stage_primary_shelf(temp_db)
        temp_db.add_filing("B5-1", 1, form="424B5", file_number=None)
        assert rf.classify_424b_attribution(1, "B5-1") == "unknown"

    def test_exchange_act_001_number_returns_unknown(self, temp_db):
        temp_db.add_company(cik=1)
        _stage_primary_shelf(temp_db)
        temp_db.add_filing("K-1", 1, form="10-K", file_number="001-39000")
        assert rf.classify_424b_attribution(1, "K-1") == "unknown"

    def test_primary_set_empty_returns_unknown(self, temp_db):
        # 333- file_number but the ledger has zero shelf/s1_offering rows.
        temp_db.add_company(cik=1)
        temp_db.add_filing("B5-1", 1, form="424B5", file_number="333-100000")
        assert rf.classify_424b_attribution(1, "B5-1") == "unknown"

    def test_file_number_in_primary_set_returns_primary(self, temp_db):
        temp_db.add_company(cik=1)
        _stage_primary_shelf(temp_db, file_number="333-100000")
        temp_db.add_filing("B5-1", 1, form="424B5", file_number="333-100000")
        assert rf.classify_424b_attribution(1, "B5-1") == "primary"

    def test_333_prefix_boundary_proceeds_to_set_comparison(self, temp_db):
        # Exactly the '333-' prefix (XTIA-style file number) -> not unknown
        # on the prefix guard; flows to the primary-set comparison.
        temp_db.add_company(cik=1)
        _stage_primary_shelf(temp_db, file_number="333-223960")
        temp_db.add_filing("B5-1", 1, form="424B5", file_number="333-223960")
        assert rf.classify_424b_attribution(1, "B5-1") == "primary"

    def test_resale_when_registration_evidence_present(self, temp_db):
        # Primary shelf under a DIFFERENT file_number (primary_set non-empty),
        # plus an S-1 registration filing under the queried file_number with
        # NO shelf ledger row => resale.
        temp_db.add_company(cik=1)
        _stage_primary_shelf(temp_db, file_number="333-100000")
        # resale registration: an S-1 under 333-200000, NO ledger shelf.
        temp_db.add_filing("S1-resale", 1, form="S-1", file_number="333-200000")
        # the 424B child under the resale file_number.
        temp_db.add_filing("B5-r", 1, form="424B5", file_number="333-200000")
        assert rf.classify_424b_attribution(1, "B5-r") == "resale"

    def test_no_registration_evidence_falls_to_unknown(self, temp_db):
        # The XTIA 333-223960 pre-window-shelf scenario: 333- file_number not
        # in primary_set AND no registration-form filing indexed -> unknown.
        temp_db.add_company(cik=1)
        _stage_primary_shelf(temp_db, file_number="333-100000")
        temp_db.add_filing("B5-pw", 1, form="424B5", file_number="333-223960")
        assert rf.classify_424b_attribution(1, "B5-pw") == "unknown"

    def test_evidence_form_not_in_registration_forms_stays_unknown(
        self, temp_db,
    ):
        # A 424B5 carrying the file_number is NOT a registration parent;
        # it must not satisfy the evidence query -> unknown.
        temp_db.add_company(cik=1)
        _stage_primary_shelf(temp_db, file_number="333-100000")
        # Another 424B5 (NOT a registration form) under the queried number.
        temp_db.add_filing("B5-sib", 1, form="424B5", file_number="333-200000")
        temp_db.add_filing("B5-q", 1, form="424B5", file_number="333-200000")
        assert rf.classify_424b_attribution(1, "B5-q") == "unknown"

    def test_pos_am_sibling_does_not_satisfy_resale_evidence(self, temp_db):
        # POS AM is in RESALE_PROPAGATION_FORMS but NOT _REGISTRATION_FORMS,
        # so a POS AM sibling does NOT satisfy the resale-evidence check.
        temp_db.add_company(cik=1)
        _stage_primary_shelf(temp_db, file_number="333-100000")
        temp_db.add_filing(
            "POSAM-sib", 1, form="POS AM", file_number="333-200000",
        )
        temp_db.add_filing("B5-q", 1, form="424B5", file_number="333-200000")
        assert rf.classify_424b_attribution(1, "B5-q") == "unknown"

    def test_primary_shelves_under_different_file_number(self, temp_db):
        # primary_set non-empty but no match for the queried file_number;
        # without registration evidence -> unknown.
        temp_db.add_company(cik=1)
        _stage_primary_shelf(temp_db, file_number="333-100000")
        temp_db.add_filing("B5-q", 1, form="424B5", file_number="333-999999")
        assert rf.classify_424b_attribution(1, "B5-q") == "unknown"

    def test_wrong_cik_evidence_does_not_leak(self, temp_db):
        # An S-1 registration under cik 2 sharing the file_number must not
        # satisfy cik 1's evidence query.
        temp_db.add_company(cik=1)
        temp_db.add_company(cik=2)
        _stage_primary_shelf(temp_db, cik=1, file_number="333-100000")
        # cik 2 holds the registration parent under the queried file_number.
        temp_db.add_filing("S1-cik2", 2, form="S-1", file_number="333-200000")
        # cik 1's 424B child under that file_number.
        temp_db.add_filing("B5-q", 1, form="424B5", file_number="333-200000")
        assert rf.classify_424b_attribution(1, "B5-q") == "unknown"

    def test_wrong_cik_primary_does_not_leak(self, temp_db):
        # Only cik 2 has a primary shelf; querying cik 1 finds an empty
        # primary_set -> unknown (not 'primary').
        temp_db.add_company(cik=1)
        temp_db.add_company(cik=2)
        _stage_primary_shelf(temp_db, cik=2, file_number="333-100000")
        temp_db.add_filing("B5-q", 1, form="424B5", file_number="333-100000")
        assert rf.classify_424b_attribution(1, "B5-q") == "unknown"

    @pytest.mark.parametrize(
        "reg_form",
        ["S-1", "S-3", "S-3/A", "S-3ASR", "F-1", "F-3", "F-10EF"],
    )
    def test_resale_for_each_registration_form(self, temp_db, reg_form):
        temp_db.add_company(cik=1)
        _stage_primary_shelf(temp_db, file_number="333-100000")
        temp_db.add_filing(
            "REG-r", 1, form=reg_form, file_number="333-200000",
        )
        temp_db.add_filing("B5-r", 1, form="424B5", file_number="333-200000")
        assert rf.classify_424b_attribution(1, "B5-r") == "resale"

    def test_empty_string_file_number_returns_unknown(self, temp_db):
        # Empty-string file_number hits the same `not file_number` guard as
        # NULL (falsy), short-circuiting to unknown before the 333- check.
        temp_db.add_company(cik=1)
        _stage_primary_shelf(temp_db, file_number="333-100000")
        temp_db.add_filing("B5-e", 1, form="424B5", file_number="")
        assert rf.classify_424b_attribution(1, "B5-e") == "unknown"

    def test_substring_333_not_prefix_returns_unknown(self, temp_db):
        # An Exchange-Act-style number that CONTAINS '333-' but does not
        # START with it (e.g. '001-333-99999') must fail the anchored
        # startswith('333-') guard -> unknown. Pins that the guard is a
        # prefix check, not a substring/`in` check.
        temp_db.add_company(cik=1)
        _stage_primary_shelf(temp_db, file_number="333-100000")
        temp_db.add_filing("B5-x", 1, form="424B5", file_number="001-333-99999")
        assert rf.classify_424b_attribution(1, "B5-x") == "unknown"

    def test_in_primary_set_short_circuits_over_resale_evidence(self, temp_db):
        # When the file_number IS in primary_set, the function returns
        # 'primary' and NEVER reaches the registration-evidence query — even
        # if a registration sibling (which would otherwise look like resale
        # evidence) also exists under that SAME file_number. Pins the
        # `if file_number in primary_set: return 'primary'` short-circuit.
        temp_db.add_company(cik=1)
        _stage_primary_shelf(temp_db, file_number="333-100000")
        # A bare S-3/A registration sibling under the SAME primary file_number
        # with no ledger shelf of its own — would be resale-evidence if the
        # in-set branch did not short-circuit first.
        temp_db.add_filing("S3-A-sib", 1, form="S-3/A",
                           file_number="333-100000")
        temp_db.add_filing("B5-1", 1, form="424B5", file_number="333-100000")
        assert rf.classify_424b_attribution(1, "B5-1") == "primary"


# ──────────────────────────────────────────────────────────────────────
class TestPrimaryShelfForFiling:

    def test_child_absent_returns_none(self, temp_db):
        temp_db.add_company(cik=1)
        _stage_primary_shelf(temp_db)
        assert rf.primary_shelf_for_filing(1, "nope") is None

    def test_child_null_file_number_returns_none(self, temp_db):
        temp_db.add_company(cik=1)
        _stage_primary_shelf(temp_db)
        temp_db.add_filing("B5-1", 1, form="424B5", file_number=None)
        assert rf.primary_shelf_for_filing(1, "B5-1") is None

    def test_child_non_333_file_number_returns_none(self, temp_db):
        temp_db.add_company(cik=1)
        _stage_primary_shelf(temp_db)
        temp_db.add_filing("K-1", 1, form="10-K", file_number="001-39000")
        assert rf.primary_shelf_for_filing(1, "K-1") is None

    def test_no_sibling_shelf_returns_none(self, temp_db):
        # 333- file_number but no sibling filing's accession created a shelf.
        temp_db.add_company(cik=1)
        # A primary shelf exists under a DIFFERENT file_number so the DB
        # isn't trivially empty.
        _stage_primary_shelf(temp_db, file_number="333-100000")
        temp_db.add_filing("B5-q", 1, form="424B5", file_number="333-200000")
        assert rf.primary_shelf_for_filing(1, "B5-q") is None

    def test_424b_resolves_to_family_shelf(self, temp_db):
        temp_db.add_company(cik=1)
        _stage_primary_shelf(
            temp_db, file_number="333-100000", shelf_acc="S3-1",
            instrument_id="SH-001", label="2025 Shelf",
        )
        temp_db.add_filing("B5-1", 1, form="424B5", file_number="333-100000")
        result = rf.primary_shelf_for_filing(1, "B5-1")
        assert result == {
            "instrument_id": "SH-001",
            "label": "2025 Shelf",
            "file_number": "333-100000",
            # KEY: accession_number is the SHELF's created_accession,
            # NOT the child accession (B5-1) we passed in.
            "accession_number": "S3-1",
        }

    def test_shelf_self_match(self, temp_db):
        # The shelf-creating filing resolving to its own shelf (join by
        # file_number, not accession inequality).
        temp_db.add_company(cik=1)
        _stage_primary_shelf(
            temp_db, file_number="333-100000", shelf_acc="S3-1",
            instrument_id="SH-001",
        )
        result = rf.primary_shelf_for_filing(1, "S3-1")
        assert result is not None
        assert result["instrument_id"] == "SH-001"
        assert result["accession_number"] == "S3-1"

    def test_amendment_resolves_to_original_shelf(self, temp_db):
        # S-3/A amendment under the same file_number resolves to original.
        temp_db.add_company(cik=1)
        _stage_primary_shelf(
            temp_db, file_number="333-100000", shelf_acc="S3-1",
            instrument_id="SH-001",
        )
        temp_db.add_filing("S3-A", 1, form="S-3/A", file_number="333-100000")
        result = rf.primary_shelf_for_filing(1, "S3-A")
        assert result is not None
        assert result["instrument_id"] == "SH-001"
        assert result["accession_number"] == "S3-1"

    def test_multiple_shelves_returns_earliest_by_created_at(self, temp_db):
        # Two shelves under one file_number; earliest created_at wins.
        temp_db.add_company(cik=1)
        temp_db.add_filing("S3-early", 1, form="S-3", file_number="333-100000")
        temp_db.add_filing("S3-late", 1, form="S-3/A", file_number="333-100000")
        temp_db.add_instrument(
            "SH-late", cik=1, type="shelf", created_accession="S3-late",
            created_at="2025-06-01", label="late",
        )
        temp_db.add_instrument(
            "SH-early", cik=1, type="shelf", created_accession="S3-early",
            created_at="2025-01-01", label="early",
        )
        temp_db.add_filing("B5-1", 1, form="424B5", file_number="333-100000")
        result = rf.primary_shelf_for_filing(1, "B5-1")
        assert result["instrument_id"] == "SH-early"
        assert result["accession_number"] == "S3-early"

    def test_label_none_carries_through(self, temp_db):
        temp_db.add_company(cik=1)
        _stage_primary_shelf(
            temp_db, file_number="333-100000", shelf_acc="S3-1",
            instrument_id="SH-001", label=None,
        )
        temp_db.add_filing("B5-1", 1, form="424B5", file_number="333-100000")
        result = rf.primary_shelf_for_filing(1, "B5-1")
        assert result["label"] is None

    def test_warrant_under_same_file_number_does_not_match(self, temp_db):
        # ledger type must be shelf/s1_offering; a warrant must NOT match.
        temp_db.add_company(cik=1)
        temp_db.add_filing("W-acc", 1, form="424B5", file_number="333-100000")
        temp_db.add_instrument(
            "W-001", cik=1, type="warrant", created_accession="W-acc",
        )
        temp_db.add_filing("B5-1", 1, form="424B5", file_number="333-100000")
        assert rf.primary_shelf_for_filing(1, "B5-1") is None

    def test_s1_offering_type_matches(self, temp_db):
        temp_db.add_company(cik=1)
        _stage_primary_shelf(
            temp_db, file_number="333-100000", shelf_acc="S1-acc",
            instrument_id="S1-001", itype="s1_offering", form="S-1",
        )
        temp_db.add_filing("B5-1", 1, form="424B5", file_number="333-100000")
        result = rf.primary_shelf_for_filing(1, "B5-1")
        assert result is not None
        assert result["instrument_id"] == "S1-001"

    def test_cross_cik_isolation(self, temp_db):
        # cik 2's shelf under the same file_number must not resolve for cik 1.
        temp_db.add_company(cik=1)
        temp_db.add_company(cik=2)
        _stage_primary_shelf(
            temp_db, cik=2, file_number="333-100000", shelf_acc="S3-2",
            instrument_id="SH-2",
        )
        temp_db.add_filing("B5-1", 1, form="424B5", file_number="333-100000")
        assert rf.primary_shelf_for_filing(1, "B5-1") is None

    def test_identical_created_at_returns_single_deterministic_row(
        self, temp_db,
    ):
        # Two shelves under one file_number with IDENTICAL created_at. The
        # LIMIT 1 must still yield exactly one dict (not raise, not two rows).
        temp_db.add_company(cik=1)
        temp_db.add_filing("S3-a", 1, form="S-3", file_number="333-100000")
        temp_db.add_filing("S3-b", 1, form="S-3/A", file_number="333-100000")
        temp_db.add_instrument(
            "SH-a", cik=1, type="shelf", created_accession="S3-a",
            created_at="2025-01-01",
        )
        temp_db.add_instrument(
            "SH-b", cik=1, type="shelf", created_accession="S3-b",
            created_at="2025-01-01",
        )
        temp_db.add_filing("B5-1", 1, form="424B5", file_number="333-100000")
        result = rf.primary_shelf_for_filing(1, "B5-1")
        assert isinstance(result, dict)
        assert result["instrument_id"] in ("SH-a", "SH-b")
        assert result["accession_number"] in ("S3-a", "S3-b")

    def test_closed_status_shelf_still_resolves(self, temp_db):
        # No status filter in the SQL: a closed shelf still resolves as the
        # primary hint for a take-down under its file_number.
        temp_db.add_company(cik=1)
        temp_db.add_filing("S3-1", 1, form="S-3", file_number="333-100000")
        temp_db.add_instrument(
            "SH-001", cik=1, type="shelf", created_accession="S3-1",
            status="terminated",
        )
        temp_db.add_filing("B5-1", 1, form="424B5", file_number="333-100000")
        result = rf.primary_shelf_for_filing(1, "B5-1")
        assert result is not None
        assert result["instrument_id"] == "SH-001"

    def test_substring_333_not_prefix_returns_none(self, temp_db):
        # file_number LIKE '333-%' is an anchored prefix; a child whose
        # number merely contains '333-' ('001-333-99999') must NOT match.
        temp_db.add_company(cik=1)
        _stage_primary_shelf(temp_db, file_number="333-100000")
        # a sibling shelf-creating filing carrying the substring number,
        # so the only thing keeping the result None is the prefix guard.
        temp_db.add_filing("S3-x", 1, form="S-3", file_number="001-333-99999")
        temp_db.add_instrument(
            "SH-x", cik=1, type="shelf", created_accession="S3-x",
        )
        temp_db.add_filing("B5-x", 1, form="424B5", file_number="001-333-99999")
        assert rf.primary_shelf_for_filing(1, "B5-x") is None

    def test_self_match_with_sibling_424b_returns_shelf_not_child(
        self, temp_db,
    ):
        # The shelf-creating S-3 is queried directly while a sibling 424B
        # under the same file_number also exists. The join is by
        # file_number (no accession-inequality), so the S-3 resolves to its
        # OWN shelf row — the returned accession_number is the shelf's
        # created_accession, never the unrelated 424B sibling.
        temp_db.add_company(cik=1)
        _stage_primary_shelf(
            temp_db, file_number="333-100000", shelf_acc="S3-1",
            instrument_id="SH-001",
        )
        temp_db.add_filing("B5-sib", 1, form="424B5", file_number="333-100000")
        result = rf.primary_shelf_for_filing(1, "S3-1")
        assert result is not None
        assert result["instrument_id"] == "SH-001"
        assert result["accession_number"] == "S3-1"

    def test_returned_dict_has_exactly_four_keys(self, temp_db):
        # Pin the public contract: callers (walker.py) destructure these.
        temp_db.add_company(cik=1)
        _stage_primary_shelf(temp_db, file_number="333-100000")
        temp_db.add_filing("B5-1", 1, form="424B5", file_number="333-100000")
        result = rf.primary_shelf_for_filing(1, "B5-1")
        assert set(result.keys()) == {
            "instrument_id", "label", "file_number", "accession_number",
        }


# ──────────────────────────────────────────────────────────────────────
class TestFamilyRegistrationAccessions:

    def test_child_absent_returns_empty(self, temp_db):
        temp_db.add_company(cik=1)
        assert rf.family_registration_accessions(1, "nope") == []

    def test_null_file_number_returns_empty(self, temp_db):
        temp_db.add_company(cik=1)
        temp_db.add_filing("X-1", 1, form="S-3/A", file_number=None)
        assert rf.family_registration_accessions(1, "X-1") == []

    def test_non_333_file_number_returns_empty(self, temp_db):
        temp_db.add_company(cik=1)
        temp_db.add_filing("X-1", 1, form="S-3/A", file_number="001-39000")
        assert rf.family_registration_accessions(1, "X-1") == []

    def test_only_self_under_file_number_returns_empty(self, temp_db):
        # The only registration under the file_number is the filing itself.
        temp_db.add_company(cik=1)
        temp_db.add_filing("S3-1", 1, form="S-3", file_number="333-100000")
        assert rf.family_registration_accessions(1, "S3-1") == []

    def test_returns_sibling_registration(self, temp_db):
        temp_db.add_company(cik=1)
        temp_db.add_filing(
            "S3-1", 1, form="S-3", file_number="333-100000",
            filing_date="2025-01-01",
        )
        temp_db.add_filing(
            "S3-A", 1, form="S-3/A", file_number="333-100000",
            filing_date="2025-02-01",
        )
        result = rf.family_registration_accessions(1, "S3-1")
        assert result == [("S3-A", "S-3/A")]
        # 2-tuples, not dicts.
        assert isinstance(result[0], tuple) and len(result[0]) == 2

    def test_non_registration_siblings_excluded(self, temp_db):
        # 424B5, EFFECT, RW sharing the file_number are excluded.
        temp_db.add_company(cik=1)
        temp_db.add_filing("S3-1", 1, form="S-3", file_number="333-100000")
        temp_db.add_filing("B5", 1, form="424B5", file_number="333-100000")
        temp_db.add_filing("EFF", 1, form="EFFECT", file_number="333-100000")
        temp_db.add_filing("RW1", 1, form="RW", file_number="333-100000")
        assert rf.family_registration_accessions(1, "S3-1") == []

    def test_pos_am_sibling_not_returned(self, temp_db):
        # POS AM is in RESALE_PROPAGATION_FORMS but NOT _REGISTRATION_FORMS.
        temp_db.add_company(cik=1)
        temp_db.add_filing("S3-1", 1, form="S-3", file_number="333-100000")
        temp_db.add_filing("POSAM", 1, form="POS AM", file_number="333-100000")
        assert rf.family_registration_accessions(1, "S3-1") == []

    def test_ordering_by_filing_date_asc(self, temp_db):
        # Stage out-of-order filing_dates; assert earliest-first.
        temp_db.add_company(cik=1)
        temp_db.add_filing(
            "S3-1", 1, form="S-3", file_number="333-100000",
            filing_date="2025-01-01",
        )
        temp_db.add_filing(
            "S3-C", 1, form="S-3/A", file_number="333-100000",
            filing_date="2025-12-01",
        )
        temp_db.add_filing(
            "S3-B", 1, form="S-3/A", file_number="333-100000",
            filing_date="2025-06-01",
        )
        result = rf.family_registration_accessions(1, "S3-1")
        assert result == [("S3-B", "S-3/A"), ("S3-C", "S-3/A")]

    def test_tie_break_by_accession_asc(self, temp_db):
        # Equal filing_date -> tie-break by accession_number ASC.
        temp_db.add_company(cik=1)
        temp_db.add_filing(
            "S3-self", 1, form="S-3", file_number="333-100000",
            filing_date="2025-01-01",
        )
        temp_db.add_filing(
            "ZZZ", 1, form="S-3/A", file_number="333-100000",
            filing_date="2025-06-01",
        )
        temp_db.add_filing(
            "AAA", 1, form="S-3/A", file_number="333-100000",
            filing_date="2025-06-01",
        )
        result = rf.family_registration_accessions(1, "S3-self")
        assert result == [("AAA", "S-3/A"), ("ZZZ", "S-3/A")]

    def test_cross_cik_isolation(self, temp_db):
        # Same-file_number registration under another cik must not appear.
        temp_db.add_company(cik=1)
        temp_db.add_company(cik=2)
        temp_db.add_filing("S3-1", 1, form="S-3", file_number="333-100000")
        temp_db.add_filing(
            "S3-cik2", 2, form="S-3/A", file_number="333-100000",
        )
        assert rf.family_registration_accessions(1, "S3-1") == []

    @pytest.mark.parametrize(
        "form", ["S-1", "S-1/A", "S-3", "S-3/A", "S-3ASR", "S-3MEF",
                 "F-1", "F-1/A", "F-3", "F-3/A", "F-3ASR", "F-3MEF",
                 "F-10", "F-10/A", "F-10EF"],
    )
    def test_each_registration_form_is_returned(self, temp_db, form):
        temp_db.add_company(cik=1)
        # The queried filing itself (excluded by self != check).
        temp_db.add_filing(
            "QUERY", 1, form="S-3", file_number="333-100000",
            filing_date="2025-01-01",
        )
        temp_db.add_filing(
            "SIB", 1, form=form, file_number="333-100000",
            filing_date="2025-02-01",
        )
        result = rf.family_registration_accessions(1, "QUERY")
        assert result == [("SIB", form)]

    def test_non_registration_self_still_returns_reg_siblings(self, temp_db):
        # The function does NOT gate on the queried filing's own form — only
        # on its file_number. A 424B child (non-registration) still returns
        # the registration siblings under its file_number.
        temp_db.add_company(cik=1)
        temp_db.add_filing(
            "B5-q", 1, form="424B5", file_number="333-100000",
            filing_date="2025-03-01",
        )
        temp_db.add_filing(
            "S3-sib", 1, form="S-3", file_number="333-100000",
            filing_date="2025-01-01",
        )
        assert rf.family_registration_accessions(1, "B5-q") == [
            ("S3-sib", "S-3"),
        ]

    def test_empty_string_file_number_returns_empty(self, temp_db):
        # Empty-string file_number is falsy -> early [] (same guard as NULL).
        temp_db.add_company(cik=1)
        temp_db.add_filing("X-1", 1, form="S-3/A", file_number="")
        # A would-be sibling under empty file_number must not be matched
        # (the guard returns before the sibling query runs).
        temp_db.add_filing("X-2", 1, form="S-3", file_number="")
        assert rf.family_registration_accessions(1, "X-1") == []

    def test_substring_333_not_prefix_returns_empty(self, temp_db):
        # Same anchored-prefix guard as the sibling functions: a queried
        # filing whose file_number contains '333-' but does not start with
        # it returns [] even when a genuine registration sibling shares
        # that exact (non-333) number.
        temp_db.add_company(cik=1)
        temp_db.add_filing("Q", 1, form="424B5", file_number="001-333-99999")
        temp_db.add_filing("SIB", 1, form="S-3", file_number="001-333-99999")
        assert rf.family_registration_accessions(1, "Q") == []

    def test_return_container_is_list_of_tuples(self, temp_db):
        # Pin the container type: a list (ordered, indexable) of 2-tuples,
        # not a set or a list of dicts. Callers iterate and destructure.
        temp_db.add_company(cik=1)
        temp_db.add_filing("S3-self", 1, form="S-3", file_number="333-100000",
                           filing_date="2025-01-01")
        temp_db.add_filing("S3-a", 1, form="S-3/A", file_number="333-100000",
                           filing_date="2025-02-01")
        temp_db.add_filing("S3-b", 1, form="S-3/A", file_number="333-100000",
                           filing_date="2025-03-01")
        result = rf.family_registration_accessions(1, "S3-self")
        assert isinstance(result, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result)
        assert result == [("S3-a", "S-3/A"), ("S3-b", "S-3/A")]

    def test_multiple_distinct_dates_full_ordering(self, temp_db):
        # Three siblings across three distinct dates plus an excluded 424B —
        # asserts the full ASC ordering and the non-registration exclusion
        # together. The queried self is the earliest registration.
        temp_db.add_company(cik=1)
        temp_db.add_filing("S3-self", 1, form="S-3", file_number="333-100000",
                           filing_date="2025-01-01")
        temp_db.add_filing("S3-mid", 1, form="S-3/A", file_number="333-100000",
                           filing_date="2025-05-01")
        temp_db.add_filing("S3-old", 1, form="S-3/A", file_number="333-100000",
                           filing_date="2025-02-01")
        temp_db.add_filing("S3-new", 1, form="S-3MEF",
                           file_number="333-100000", filing_date="2025-09-01")
        temp_db.add_filing("B5-skip", 1, form="424B5",
                           file_number="333-100000", filing_date="2025-03-01")
        result = rf.family_registration_accessions(1, "S3-self")
        assert result == [
            ("S3-old", "S-3/A"),
            ("S3-mid", "S-3/A"),
            ("S3-new", "S-3MEF"),
        ]
