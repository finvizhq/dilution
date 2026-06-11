"""Shared pytest fixtures for the dilution test suite.

The cardinal rule of this suite: **no test may ever touch the real
``dilution.db``** (it is ~1.2 GB of production data). ``db.get_conn()``
resolves ``DB_PATH`` from the ``db`` module's globals *at call time*, so
redirecting ``db.DB_PATH`` to a throwaway file reroutes every caller —
including code that did ``from db import get_conn`` at import time.

The :func:`temp_db` fixture below is **autouse**, so that redirect happens
for *every* test whether it asks for it or not. Pure-logic tests simply
ignore the yielded helper; DB-backed tests accept the ``temp_db`` argument
and use its inserter helpers to stage fixture rows.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest


# Anything not passed to the inserter helpers falls back to these so a
# test only has to specify the columns it actually cares about.
_NOW = "2026-01-01T00:00:00Z"


@dataclass
class DBHelper:
    """Thin wrapper around the per-test temp SQLite DB.

    Exposes a connection context plus convenience inserters that mirror
    the production schema (``dilution/schema.py``). Every inserter accepts
    keyword overrides for any column and applies schema-consistent
    defaults for the rest, so a test can write::

        temp_db.add_instrument("ATM-001", type="atm",
                               terms_json='{"capacity_usd": 5e6}')

    without having to spell out all 18 ledger columns.
    """

    path: str

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def execute(self, sql: str, params: tuple = ()):  # convenience
        with self.conn() as c:
            cur = c.execute(sql, params)
            return cur.fetchall()

    # ── inserters ──────────────────────────────────────────────────
    def add_company(self, cik: int, ticker: str = "TEST",
                    name: str = "Test Co", *, is_fpi: int = 0,
                    ads_ratio: float | None = None,
                    added_at: str = _NOW) -> int:
        with self.conn() as c:
            c.execute(
                """INSERT INTO dilution_company
                     (cik, ticker, name, added_at, is_fpi, ads_ratio)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (cik, ticker, name, added_at, is_fpi, ads_ratio),
            )
        return cik

    def add_filing(self, accession: str, cik: int, *, form: str = "8-K",
                   filing_date: str = "2025-01-01",
                   report_date: str | None = None,
                   file_number: str | None = None,
                   items: str | None = None,
                   primary_doc: str | None = None,
                   primary_doc_url: str | None = None,
                   homepage_url: str | None = None,
                   fetched_at: str | None = _NOW,
                   extracted_at: str | None = None,
                   extracted_by: str | None = None) -> str:
        with self.conn() as c:
            c.execute(
                """INSERT INTO dilution_filings
                     (accession_number, cik, form, filing_date, report_date,
                      primary_doc, primary_doc_url, homepage_url, items,
                      file_number, fetched_at, extracted_at, extracted_by)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (accession, cik, form, filing_date, report_date,
                 primary_doc, primary_doc_url, homepage_url, items,
                 file_number, fetched_at, extracted_at, extracted_by),
            )
        return accession

    def add_instrument(self, instrument_id: str, *, cik: int = 1,
                       ticker: str = "TEST", type: str = "warrant",
                       created_at: str = "2025-01-01",
                       created_accession: str = "0000-acc",
                       registration_accession: str | None = None,
                       counterparty_canonical: str | None = None,
                       counterparty_status: str | None = None,
                       placement_agent_canonical: str | None = None,
                       label: str | None = None,
                       terms_json: str = "{}",
                       outstanding_json: str = "{}",
                       status: str = "active",
                       status_at: str | None = None,
                       history_json: str = "[]",
                       last_seen_accession: str | None = None,
                       last_seen_date: str | None = None,
                       anchor_miss_count: int = 0) -> str:
        with self.conn() as c:
            c.execute(
                """INSERT INTO dilution_ledger
                     (instrument_id, ticker, cik, type, created_at,
                      created_accession, registration_accession,
                      counterparty_canonical, counterparty_status,
                      placement_agent_canonical, label, terms_json,
                      outstanding_json, status, status_at, history_json,
                      last_seen_accession, last_seen_date, anchor_miss_count)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (instrument_id, ticker, cik, type, created_at,
                 created_accession, registration_accession,
                 counterparty_canonical, counterparty_status,
                 placement_agent_canonical, label, terms_json,
                 outstanding_json, status, status_at, history_json,
                 last_seen_accession, last_seen_date, anchor_miss_count),
            )
        return instrument_id

    def add_drawdown(self, instrument_id: str, *, cik: int = 1,
                     accession_number: str = "0000-acc",
                     event_date: str = "2025-06-01",
                     amount_usd: float | None = None,
                     shares: float | None = None,
                     price: float | None = None,
                     drawdown_party_canonical: str | None = None,
                     drawdown_party_role: str | None = None,
                     detected_at: str = _NOW) -> None:
        with self.conn() as c:
            c.execute(
                """INSERT INTO dilution_ledger_drawdowns
                     (cik, instrument_id, accession_number, event_date,
                      amount_usd, shares, price, drawdown_party_canonical,
                      drawdown_party_role, detected_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (cik, instrument_id, accession_number, event_date,
                 amount_usd, shares, price, drawdown_party_canonical,
                 drawdown_party_role, detected_at),
            )

    def add_split(self, cik: int, effective_date: str, pre: int, post: int,
                  *, direction: str | None = None, units: str = "common",
                  source: str = "finviz", fetched_at: str = _NOW) -> None:
        if direction is None:
            direction = "forward" if post > pre else "reverse"
        with self.conn() as c:
            c.execute(
                """INSERT INTO dilution_splits
                     (cik, effective_date, pre, post, direction, units,
                      source, fetched_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (cik, effective_date, pre, post, direction, units, source,
                 fetched_at),
            )


# Modules that memoize results keyed on cik/ticker/date via @lru_cache or a
# module-level dict. Because those keys do NOT include the DB file, a value
# computed against one test's temp_db would leak into another test that reuses
# the same cik. We clear them before every test so suite order can never cause
# a stale-cache cross-file failure. Importing is best-effort — a module that
# can't import (heavy optional dep) is simply skipped.
_CACHE_MODULES = (
    "dilution.cash_history",
    "dilution.share_counts",
    "dilution.os_history",
    "dilution.ib6_cover",
    "dilution.badges",
    "dilution.ledger.baby_shelf",
    "dilution.ledger.cards",
)
# Module-level dict caches that must be emptied (cleared in place so any
# code holding the same dict object sees the reset).
_DICT_CACHES = (
    ("dilution.ledger.baby_shelf", "_BABY_EXIT_CLOSES_CACHE"),
    ("dilution.ledger.cards", "_MARKET_LOW_CACHE"),
)


@pytest.fixture(autouse=True)
def _reset_caches():
    """Clear every process-global memo before each test for isolation."""
    import importlib

    def _clear():
        for name in _CACHE_MODULES:
            try:
                mod = importlib.import_module(name)
            except Exception:
                continue
            for attr in vars(mod).values():
                cc = getattr(attr, "cache_clear", None)
                if callable(cc):
                    try:
                        cc()
                    except Exception:
                        pass
        for mod_name, attr_name in _DICT_CACHES:
            try:
                mod = importlib.import_module(mod_name)
                cache = getattr(mod, attr_name, None)
                if isinstance(cache, dict):
                    cache.clear()
            except Exception:
                pass

    _clear()   # before the test
    yield
    _clear()   # and after, so a test's own writes don't outlive it


@pytest.fixture(scope="session")
def _schema_template(tmp_path_factory):
    """Build the production schema ONCE per session into a template DB.

    Running ``init_dilution_db()`` per test costs ~250 ms (the
    ``PRAGMA journal_mode=WAL`` fsync + the full DDL ``executescript``);
    across ~4k tests that dominated the whole run. The per-test fixture
    below instead copies this prebuilt file (~0.2 ms). Built on a plain
    (non-WAL) connection so the single ``.db`` file is fully self-contained
    and safe to ``copyfile``.
    """
    import sqlite3
    import dilution.schema as schema

    path = tmp_path_factory.mktemp("schema") / "template.db"
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(schema.SCHEMA)
        conn.commit()
    finally:
        conn.close()
    return path


@pytest.fixture(autouse=True)
def temp_db(_schema_template, tmp_path, monkeypatch):
    """Redirect ALL db access to a fresh per-test SQLite file (a copy of
    the session schema template) and reroute ``db.get_conn()`` to it.
    Autouse → every test is isolated from the real DB even if it never
    references this fixture.

    Yields a :class:`DBHelper` for tests that want to stage rows.
    """
    import shutil
    import db

    db_file = tmp_path / "dilution_test.db"
    shutil.copyfile(_schema_template, db_file)
    # db.get_conn reads DB_PATH from the db module's globals at call time,
    # so this one patch reroutes every consumer of get_conn().
    monkeypatch.setattr(db, "DB_PATH", db_file, raising=False)
    yield DBHelper(str(db_file))
