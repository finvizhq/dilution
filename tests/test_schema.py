"""Unit tests for ``dilution/schema.py``.

This module is almost entirely DDL plus three controlled-vocabulary
constants. The only callable is :func:`init_dilution_db`, a thin
pass-through to ``conn.executescript(SCHEMA)``. The autouse ``temp_db``
fixture in ``conftest.py`` already invokes it once during setup, so these
tests mostly assert observable structural facts (table/index set,
idempotency, return value) and pin the public vocabulary constants.

No network / SEC / LLM / vendor calls are made. The ``temp_db`` fixture
reroutes ``db.get_conn()`` to a throwaway per-test SQLite file, so the
real ``dilution.db`` is never touched.
"""

from __future__ import annotations

import sqlite3

import pytest

import dilution.schema as schema


# The 12 dilution_* tables the SCHEMA literal creates. (The module
# docstring header only enumerates 8 of these, and the survey slice
# enumerated 12 while calling them "11" — the authoritative list is the
# set of CREATE TABLE statements in SCHEMA, which is these 12.)
EXPECTED_TABLES = {
    "dilution_company",
    "dilution_filings",
    "dilution_raw",
    "dilution_ledger",
    "dilution_walk_state",
    "dilution_walked",
    "dilution_splits",
    "dilution_anchor_diffs",
    "dilution_walk_errors",
    "dilution_ledger_drawdowns",
    "dilution_ledger_narrative",
    "dilution_ticker_brief",
}

# Named indexes declared in SCHEMA (every "CREATE INDEX ... idx_...").
EXPECTED_INDEXES = {
    "idx_dilution_company_ticker",
    "idx_dilution_filings_cik",
    "idx_dilution_filings_form",
    "idx_dilution_filings_date",
    "idx_dilution_filings_file_number",
    "idx_dilution_ledger_cik",
    "idx_dilution_ledger_cik_type",
    "idx_dilution_ledger_cik_status",
    "idx_dilution_walked_cik",
    "idx_dilution_splits_cik",
    "idx_dilution_anchor_diffs_cik_acc",
    "idx_dilution_walk_errors_cik_acc",
    "idx_dilution_ledger_drawdowns_cik_date",
    "idx_dilution_ledger_drawdowns_instrument",
}


def _tables(temp_db) -> set[str]:
    rows = temp_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    return {r[0] for r in rows}


def _indexes(temp_db) -> set[str]:
    rows = temp_db.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    )
    return {r[0] for r in rows}


class TestInitDilutionDb:
    """``init_dilution_db()`` builds the full schema via executescript."""

    def test_returns_none(self, temp_db):
        # No return value: pure side-effect pass-through.
        assert schema.init_dilution_db() is None

    def test_all_expected_tables_exist(self, temp_db):
        # The autouse fixture already ran init once; assert the full set.
        present = _tables(temp_db)
        missing = EXPECTED_TABLES - present
        assert not missing, f"missing tables: {sorted(missing)}"

    def test_exactly_twelve_dilution_tables(self, temp_db):
        dilution_tables = {t for t in _tables(temp_db)
                           if t.startswith("dilution_")}
        assert dilution_tables == EXPECTED_TABLES

    def test_no_unexpected_user_tables(self, temp_db):
        # Every non-internal table must be one of the 12 declared ones.
        # SQLite's own bookkeeping table (sqlite_sequence, created
        # because several tables use AUTOINCREMENT) is the only allowed
        # non-dilution_ table; anything else is accidental creep.
        present = {t for t in _tables(temp_db)
                   if not t.startswith("sqlite_")}
        assert present == EXPECTED_TABLES

    def test_all_expected_indexes_exist(self, temp_db):
        present = _indexes(temp_db)
        missing = EXPECTED_INDEXES - present
        assert not missing, f"missing indexes: {sorted(missing)}"

    def test_exactly_fourteen_named_indexes(self, temp_db):
        # The subset check above tolerates stray/extra indexes; pin the
        # full named-index set so a renamed or dropped index is caught.
        # (SQLite also creates internal sqlite_autoindex_* entries for
        # composite/AUTOINCREMENT PKs; exclude those — only the 14
        # explicitly-declared idx_dilution_* indexes are the contract.)
        present = {i for i in _indexes(temp_db)
                   if i.startswith("idx_dilution_")}
        assert present == EXPECTED_INDEXES
        assert len(EXPECTED_INDEXES) == 14

    def test_idempotent_double_call_does_not_raise(self, temp_db):
        # SCHEMA uses CREATE TABLE/INDEX IF NOT EXISTS, so re-running is a
        # no-op. The fixture already called it once; call twice more.
        schema.init_dilution_db()
        schema.init_dilution_db()
        # Schema still intact and complete after repeated calls.
        assert EXPECTED_TABLES <= _tables(temp_db)
        assert EXPECTED_INDEXES <= _indexes(temp_db)

    def test_idempotent_does_not_wipe_existing_rows(self, temp_db):
        # Re-running the DDL must not drop/recreate populated tables.
        temp_db.add_company(1234, ticker="AAA", name="Alpha")
        schema.init_dilution_db()
        rows = temp_db.execute(
            "SELECT ticker FROM dilution_company WHERE cik=?", (1234,)
        )
        assert [r[0] for r in rows] == ["AAA"]

    def test_idempotent_preserves_rows_across_related_tables(self, temp_db):
        # A second init must not truncate ANY table — stage a small
        # company -> ledger -> drawdown chain (which only survives a
        # re-run if the FK-linked rows are all left intact) and assert
        # every row is still present after re-running the DDL twice.
        temp_db.add_company(99, ticker="ZZZ", name="Zeta")
        temp_db.add_instrument("ATM-001", cik=99, type="atm")
        temp_db.add_drawdown("ATM-001", cik=99, amount_usd=2_500_000.0)
        schema.init_dilution_db()
        schema.init_dilution_db()
        assert temp_db.execute(
            "SELECT COUNT(*) FROM dilution_company WHERE cik=99"
        )[0][0] == 1
        assert temp_db.execute(
            "SELECT COUNT(*) FROM dilution_ledger WHERE instrument_id='ATM-001'"
        )[0][0] == 1
        amt = temp_db.execute(
            "SELECT amount_usd FROM dilution_ledger_drawdowns "
            "WHERE instrument_id='ATM-001'"
        )[0][0]
        assert amt == pytest.approx(2_500_000.0)

    def test_uses_get_conn_seam(self, temp_db, monkeypatch):
        # init_dilution_db must go through the (already-rerouted)
        # get_conn seam, not connect to any hard-coded path. We confirm
        # by spying on schema.get_conn and asserting it is invoked.
        calls = {"n": 0}
        real = schema.get_conn

        def spy():
            calls["n"] += 1
            return real()

        monkeypatch.setattr(schema, "get_conn", spy)
        schema.init_dilution_db()
        assert calls["n"] == 1


class TestSchemaStructure:
    """Column-level guards on a few load-bearing table definitions."""

    def test_ledger_primary_key_is_instrument_id(self, temp_db):
        rows = temp_db.execute("PRAGMA table_info(dilution_ledger)")
        pk_cols = [r["name"] for r in rows if r["pk"]]
        assert pk_cols == ["instrument_id"]

    def test_ledger_has_anchor_miss_count_default_zero(self, temp_db):
        # Insert a row without specifying anchor_miss_count and confirm
        # the schema default of 0 lands.
        temp_db.add_instrument("W-001")
        rows = temp_db.execute(
            "SELECT anchor_miss_count FROM dilution_ledger "
            "WHERE instrument_id='W-001'"
        )
        assert rows[0][0] == 0

    def test_company_is_fpi_default_zero(self, temp_db):
        # add_company in conftest passes is_fpi explicitly; assert the
        # column default via a raw insert that omits it.
        temp_db.execute(
            "INSERT INTO dilution_company (cik, ticker, name, added_at) "
            "VALUES (?,?,?,?)",
            (777, "DEF", "Default Co", "2026-01-01T00:00:00Z"),
        )
        rows = temp_db.execute(
            "SELECT is_fpi, ads_ratio FROM dilution_company WHERE cik=777"
        )
        assert rows[0]["is_fpi"] == 0
        assert rows[0]["ads_ratio"] is None

    def test_walk_state_next_id_seq_default(self, temp_db):
        temp_db.execute(
            "INSERT INTO dilution_walk_state (cik) VALUES (?)", (42,)
        )
        rows = temp_db.execute(
            "SELECT next_id_seq_json FROM dilution_walk_state WHERE cik=42"
        )
        assert rows[0][0] == "{}"

    def test_anchor_diffs_autoincrement_id(self, temp_db):
        # AUTOINCREMENT means inserts assign monotonic ids automatically.
        ins = (
            "INSERT INTO dilution_anchor_diffs "
            "(cik, accession_number, as_of_date, diff_kind, detected_at) "
            "VALUES (?,?,?,?,?)"
        )
        temp_db.execute(
            ins, (1, "acc-1", "2025-01-01", "missing_in_ledger", "2026-01-01")
        )
        temp_db.execute(
            ins, (1, "acc-2", "2025-01-02", "extra_in_ledger", "2026-01-01")
        )
        ids = [r[0] for r in temp_db.execute(
            "SELECT id FROM dilution_anchor_diffs ORDER BY id"
        )]
        assert ids == [1, 2]

    @pytest.mark.parametrize(
        "table,cols,values",
        [
            (
                "dilution_anchor_diffs",
                "(cik, accession_number, as_of_date, diff_kind, detected_at)",
                (1, "acc", "2025-01-01", "missing_in_ledger", "2026-01-01"),
            ),
            (
                "dilution_walk_errors",
                "(cik, accession_number, error_kind, detected_at)",
                (1, "acc", "missing_id", "2026-01-01"),
            ),
        ],
    )
    def test_autoincrement_does_not_reuse_deleted_id(
        self, temp_db, table, cols, values
    ):
        # A plain ``INTEGER PRIMARY KEY`` (ROWID alias) would *reuse* the
        # max rowid after the top row is deleted; a true ``AUTOINCREMENT``
        # column never reissues a freed id. Re-derived from source: both
        # of these tables declare ``id INTEGER PRIMARY KEY AUTOINCREMENT``.
        # The old ``[1, 2]`` assertion alone could not distinguish the two
        # — this delete-then-reinsert sequence is what pins AUTOINCREMENT.
        ph = ",".join("?" * len(values))
        ins = f"INSERT INTO {table} {cols} VALUES ({ph})"
        temp_db.execute(ins, values)
        temp_db.execute(ins, values)
        temp_db.execute(f"DELETE FROM {table} WHERE id=2")
        temp_db.execute(ins, values)
        ids = [r[0] for r in temp_db.execute(
            f"SELECT id FROM {table} ORDER BY id"
        )]
        # 2 was retired by AUTOINCREMENT; the third insert gets 3, not 2.
        assert ids == [1, 3]

    def test_narrative_foreign_key_to_ledger_enforced(self, temp_db):
        # dilution_ledger_narrative.instrument_id REFERENCES dilution_ledger;
        # a narrative row for a ghost instrument must be rejected, and
        # lands once the parent ledger row exists. (The conftest conn()
        # turns PRAGMA foreign_keys=ON for every connection.)
        with pytest.raises(sqlite3.IntegrityError):
            temp_db.execute(
                "INSERT INTO dilution_ledger_narrative "
                "(instrument_id, terms_hash, generated_at) "
                "VALUES (?,?,?)",
                ("GHOST-1", "h", "2026-01-01"),
            )
        temp_db.add_instrument("W-001")
        temp_db.execute(
            "INSERT INTO dilution_ledger_narrative "
            "(instrument_id, terms_hash, generated_at) "
            "VALUES (?,?,?)",
            ("W-001", "h", "2026-01-01"),
        )
        n = temp_db.execute(
            "SELECT COUNT(*) FROM dilution_ledger_narrative "
            "WHERE instrument_id='W-001'"
        )[0][0]
        assert n == 1

    def test_ledger_not_null_columns_reject_null(self, temp_db):
        # type is NOT NULL: a raw insert leaving it NULL must fail.
        with pytest.raises(sqlite3.IntegrityError):
            temp_db.execute(
                "INSERT INTO dilution_ledger "
                "(instrument_id, ticker, cik, type, created_at, "
                " created_accession, terms_json, outstanding_json, "
                " status, history_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                ("X-1", "TEST", 1, None, "2025-01-01", "acc",
                 "{}", "{}", "active", "[]"),
            )

    def test_splits_composite_primary_key(self, temp_db):
        # PRIMARY KEY (cik, effective_date): duplicate must be rejected.
        temp_db.add_split(1, "2025-01-01", pre=10, post=1)
        with pytest.raises(sqlite3.IntegrityError):
            temp_db.add_split(1, "2025-01-01", pre=20, post=1)
        # Different effective_date for same cik is fine.
        temp_db.add_split(1, "2025-02-01", pre=20, post=1)
        n = temp_db.execute(
            "SELECT COUNT(*) FROM dilution_splits WHERE cik=1"
        )[0][0]
        assert n == 2

    def test_filings_primary_key_rejects_duplicate_accession(self, temp_db):
        # accession_number is the PK; a second row with the same
        # accession must be rejected.
        temp_db.add_filing("ACC-1", cik=1, form="8-K")
        with pytest.raises(sqlite3.IntegrityError):
            temp_db.add_filing("ACC-1", cik=2, form="10-K")

    def test_walked_composite_primary_key(self, temp_db):
        # PRIMARY KEY (cik, accession_number): same pair rejected, but a
        # different accession for the same cik is allowed.
        temp_db.execute(
            "INSERT INTO dilution_walked (cik, accession_number, walked_at) "
            "VALUES (?,?,?)",
            (1, "A-1", "2026-01-01"),
        )
        with pytest.raises(sqlite3.IntegrityError):
            temp_db.execute(
                "INSERT INTO dilution_walked (cik, accession_number, walked_at) "
                "VALUES (?,?,?)",
                (1, "A-1", "2026-02-01"),
            )
        temp_db.execute(
            "INSERT INTO dilution_walked (cik, accession_number, walked_at) "
            "VALUES (?,?,?)",
            (1, "A-2", "2026-02-01"),
        )
        n = temp_db.execute(
            "SELECT COUNT(*) FROM dilution_walked WHERE cik=1"
        )[0][0]
        assert n == 2

    def test_raw_composite_primary_key(self, temp_db):
        # PRIMARY KEY (accession_number, doc_name): the same document on
        # the same filing cannot be inserted twice, but a second doc on
        # the same filing is fine.
        temp_db.add_filing("ACC-1", cik=1)
        temp_db.execute(
            "INSERT INTO dilution_raw "
            "(accession_number, doc_name, content_md, downloaded_at) "
            "VALUES (?,?,?,?)",
            ("ACC-1", "doc.htm", "md", "2026-01-01"),
        )
        with pytest.raises(sqlite3.IntegrityError):
            temp_db.execute(
                "INSERT INTO dilution_raw "
                "(accession_number, doc_name, content_md, downloaded_at) "
                "VALUES (?,?,?,?)",
                ("ACC-1", "doc.htm", "different md", "2026-01-02"),
            )
        temp_db.execute(
            "INSERT INTO dilution_raw "
            "(accession_number, doc_name, content_md, downloaded_at) "
            "VALUES (?,?,?,?)",
            ("ACC-1", "doc2.htm", "md2", "2026-01-02"),
        )
        n = temp_db.execute(
            "SELECT COUNT(*) FROM dilution_raw WHERE accession_number='ACC-1'"
        )[0][0]
        assert n == 2

    def test_raw_foreign_key_to_filings_enforced(self, temp_db):
        # dilution_raw.accession_number REFERENCES dilution_filings; the
        # conftest conn() turns PRAGMA foreign_keys=ON, so an orphan row
        # (no matching filing) must be rejected.
        with pytest.raises(sqlite3.IntegrityError):
            temp_db.execute(
                "INSERT INTO dilution_raw "
                "(accession_number, doc_name, content_md, downloaded_at) "
                "VALUES (?,?,?,?)",
                ("no-such-filing", "doc.htm", "md", "2026-01-01"),
            )
        # With the parent filing present, the same insert succeeds.
        temp_db.add_filing("real-acc", cik=1)
        temp_db.execute(
            "INSERT INTO dilution_raw "
            "(accession_number, doc_name, content_md, downloaded_at) "
            "VALUES (?,?,?,?)",
            ("real-acc", "doc.htm", "md", "2026-01-01"),
        )
        n = temp_db.execute(
            "SELECT COUNT(*) FROM dilution_raw WHERE accession_number='real-acc'"
        )[0][0]
        assert n == 1

    def test_drawdowns_foreign_key_to_ledger_enforced(self, temp_db):
        # dilution_ledger_drawdowns.instrument_id REFERENCES dilution_ledger;
        # a drawdown against a ghost instrument must be rejected.
        with pytest.raises(sqlite3.IntegrityError):
            temp_db.execute(
                "INSERT INTO dilution_ledger_drawdowns "
                "(cik, instrument_id, accession_number, event_date, detected_at) "
                "VALUES (?,?,?,?,?)",
                (1, "GHOST-1", "acc", "2025-01-01", "2026-01-01"),
            )
        # With the parent ledger row present, the drawdown lands.
        temp_db.add_instrument("ATM-001", type="atm")
        temp_db.add_drawdown("ATM-001", amount_usd=1_000_000.0)
        n = temp_db.execute(
            "SELECT COUNT(*) FROM dilution_ledger_drawdowns "
            "WHERE instrument_id='ATM-001'"
        )[0][0]
        assert n == 1


class TestVocabularyConstants:
    """Public controlled-vocabulary constants exposed by the module."""

    def test_instrument_types_is_immutable_tuple(self):
        assert isinstance(schema.INSTRUMENT_TYPES, tuple)

    def test_instrument_statuses_is_immutable_tuple(self):
        assert isinstance(schema.INSTRUMENT_STATUSES, tuple)

    def test_event_types_is_list(self):
        # EVENT_TYPES is intentionally a list (the two are tuples).
        assert isinstance(schema.EVENT_TYPES, list)

    def test_instrument_types_exact_membership(self):
        assert schema.INSTRUMENT_TYPES == (
            "warrant",
            "convertible",
            "preferred",
            "atm",
            "equity_line",
            "shelf",
            "s1_offering",
            "equity",
        )

    def test_instrument_statuses_exact_membership(self):
        assert schema.INSTRUMENT_STATUSES == (
            "active",
            "exercised",
            "converted",
            "redeemed",
            "expired",
            "terminated",
            "superseded",
        )

    def test_event_types_contains_known_anchors(self):
        # Spot-check representative members across the spectrum.
        for ev in (
            "shelf_registration",
            "atm_sale",
            "warrant_exercise",
            "preferred_conversion",
            "reverse_split",
            "stock_dividend",
            "rights_offering",
            "share_repurchase",
            "other",
        ):
            assert ev in schema.EVENT_TYPES

    def test_event_types_count(self):
        assert len(schema.EVENT_TYPES) == 25

    def test_event_types_exact_ordered_list(self):
        # Re-derived directly from the SCHEMA module source. A spot-check
        # + count cannot catch a rename that keeps the length the same;
        # pinning the full ordered list does. This is the controlled
        # vocabulary the extractor prompts must stay in sync with.
        assert schema.EVENT_TYPES == [
            "shelf_registration",
            "atm_program_established",
            "atm_sale",
            "equity_line_established",
            "equity_line_sale",
            "registered_direct_offering",
            "underwritten_offering",
            "private_placement",
            "convertible_note_issuance",
            "convertible_note_conversion",
            "warrant_issuance",
            "warrant_exercise",
            "preferred_issuance",
            "preferred_conversion",
            "reverse_split",
            "forward_split",
            "authorized_share_increase",
            "equity_plan_increase",
            "share_issuance_other",
            "offering_effective",
            "shelf_withdrawn",
            "stock_dividend",
            "rights_offering",
            "share_repurchase",
            "other",
        ]

    def test_no_duplicate_vocab_entries(self):
        for vocab in (schema.INSTRUMENT_TYPES,
                      schema.INSTRUMENT_STATUSES,
                      schema.EVENT_TYPES):
            assert len(vocab) == len(set(vocab))

    @pytest.mark.parametrize("status", schema.INSTRUMENT_STATUSES)
    def test_each_status_is_lowercase_token(self, status):
        # Vocab tokens are bare lowercase identifiers (the "superseded:<id>"
        # form is constructed at use-site, not stored in the vocab).
        assert status.islower()
        assert ":" not in status

    @pytest.mark.parametrize("itype", schema.INSTRUMENT_TYPES)
    def test_each_type_is_lowercase_token(self, itype):
        assert itype.islower()

    @pytest.mark.parametrize("ev", schema.EVENT_TYPES)
    def test_each_event_type_is_lowercase_token(self, ev):
        # Event vocab tokens are bare lowercase snake_case identifiers
        # (no whitespace, no colon, no uppercase) since they are emitted
        # verbatim by the extractor and matched literally downstream.
        assert ev.islower()
        assert " " not in ev
        assert ":" not in ev


class TestSchemaConstant:
    """The SCHEMA string itself is a CREATE-only DDL script."""

    def test_schema_is_str(self):
        assert isinstance(schema.SCHEMA, str)

    def test_schema_has_no_destructive_statements(self):
        upper = schema.SCHEMA.upper()
        # A reset-from-scratch DDL must never DROP or DELETE.
        assert "DROP TABLE" not in upper
        assert "DELETE FROM" not in upper

    def test_every_create_uses_if_not_exists(self):
        # Idempotency is guaranteed only if every CREATE is guarded.
        import re
        creates = re.findall(r"CREATE (?:TABLE|INDEX)([^\n(]*)",
                             schema.SCHEMA)
        for tail in creates:
            assert "IF NOT EXISTS" in tail.upper(), tail

    def test_create_table_count_matches_expected(self):
        import re
        names = re.findall(
            r"CREATE TABLE IF NOT EXISTS (\w+)", schema.SCHEMA
        )
        assert set(names) == EXPECTED_TABLES
        assert len(names) == 12
