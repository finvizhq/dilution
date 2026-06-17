"""Closed-row resurrection guard in ``store._create_already_recorded``.

A periodic balance sheet (10-K/10-Q/20-F) re-lists a tranche that has
already been redeemed/terminated/converted/expired. The LLM emits
``create_instrument`` for it; because the active-row dedup only scans
``status='active'``, the create would otherwise land as a NEW active row,
resurrecting a dead instrument (the live XTIA P-447 Series-9 bug, which
duplicated the already-redeemed P-443). The guard collapses such a create
onto the closed row — but only on a STRONG identity key within a TIGHT
date window, so a genuinely new same-letter issuance (XTIA re-uses Series
4/5 across years) never collapses onto an old closed tranche.
"""

from datetime import date

import db
from dilution.ledger.mutations import CreatePreferred, CreateWarrant
from dilution.ledger.store import _create_already_recorded


CIK = 1529113  # XTIA, for verisimilitude — temp_db is empty regardless.


def _redisclose(m, *, filing_date: str, accession: str = "new-acc"):
    with db.get_conn() as conn:
        return _create_already_recorded(conn, CIK, m, filing_date,
                                        accession=accession)


class TestPreferredResurrectionGuard:
    def test_redeemed_preferred_redisclosed_collapses_onto_closed_row(
            self, temp_db):
        """Series 9 redeemed in 2024; a 2025 10-K re-lists it carrying the
        ORIGINAL issue_date → must collapse onto the closed row, not spawn
        a new active duplicate."""
        temp_db.add_company(CIK, "XTIA")
        temp_db.add_instrument(
            "P-443", cik=CIK, type="preferred", status="redeemed",
            created_at="2024-03-12", created_accession="orig-acc",
            terms_json='{"series_letter": "9"}',
            outstanding_json='{"count": 0}',
        )
        m = CreatePreferred(count=1000, series_letter="9",
                            event_date=date(2024, 3, 12))
        assert _redisclose(m, filing_date="2025-04-15") == "P-443"

    def test_new_same_letter_issuance_outside_window_is_not_merged(
            self, temp_db):
        """XTIA re-uses series letters: a terminated Series 4 (2024-08) must
        NOT swallow a genuinely new Series 4 issued ~9 months later — the
        tight window keeps them distinct."""
        temp_db.add_company(CIK, "XTIA")
        temp_db.add_instrument(
            "P-444", cik=CIK, type="preferred", status="terminated",
            created_at="2024-08-14", created_accession="orig-acc",
            terms_json='{"series_letter": "4"}',
            outstanding_json='{"count": 0}',
        )
        m = CreatePreferred(count=500, series_letter="4",
                            event_date=date(2025, 5, 19))
        assert _redisclose(m, filing_date="2025-05-19") is None

    def test_different_series_does_not_match_closed_row(self, temp_db):
        temp_db.add_company(CIK, "XTIA")
        temp_db.add_instrument(
            "P-443", cik=CIK, type="preferred", status="redeemed",
            created_at="2024-03-12", created_accession="orig-acc",
            terms_json='{"series_letter": "9"}',
            outstanding_json='{"count": 0}',
        )
        m = CreatePreferred(count=1000, series_letter="10",
                            event_date=date(2024, 3, 12))
        assert _redisclose(m, filing_date="2025-04-15") is None

    def test_superseded_row_is_excluded(self, temp_db):
        """A superseded row has a live successor the active scan already
        matches; the guard must NOT collapse onto the superseded predecessor."""
        temp_db.add_company(CIK, "XTIA")
        temp_db.add_instrument(
            "P-443", cik=CIK, type="preferred", status="superseded:P-999",
            created_at="2024-03-12", created_accession="orig-acc",
            terms_json='{"series_letter": "9"}',
            outstanding_json='{"count": 0}',
        )
        m = CreatePreferred(count=1000, series_letter="9",
                            event_date=date(2024, 3, 12))
        assert _redisclose(m, filing_date="2025-04-15") is None


class TestWarrantResurrectionGuard:
    def test_expired_warrant_redisclosed_collapses_on_strike_and_expiration(
            self, temp_db):
        temp_db.add_company(CIK, "XTIA")
        temp_db.add_instrument(
            "W-100", cik=CIK, type="warrant", status="expired",
            created_at="2021-01-01", created_accession="orig-acc",
            terms_json='{"strike": 5.0, "expiration": "2026-01-01"}',
            outstanding_json='{"count": 0}',
        )
        m = CreateWarrant(count=10000, strike=5.0, event_date=date(2021, 1, 1),
                          expiration=date(2026, 1, 1))
        assert _redisclose(m, filing_date="2025-08-14") == "W-100"

    def test_distinct_strike_does_not_collapse(self, temp_db):
        temp_db.add_company(CIK, "XTIA")
        temp_db.add_instrument(
            "W-100", cik=CIK, type="warrant", status="exercised",
            created_at="2021-01-01", created_accession="orig-acc",
            terms_json='{"strike": 5.0, "expiration": "2026-01-01"}',
            outstanding_json='{"count": 0}',
        )
        m = CreateWarrant(count=10000, strike=12.0, event_date=date(2021, 1, 1),
                          expiration=date(2026, 1, 1))
        assert _redisclose(m, filing_date="2025-08-14") is None

    def test_conflicting_expiration_does_not_collapse(self, temp_db):
        """Same strike but a clearly different expiration ⇒ a distinct
        warrant, even against a closed row (mirrors the active-path
        _end_dates_conflict guard)."""
        temp_db.add_company(CIK, "XTIA")
        temp_db.add_instrument(
            "W-100", cik=CIK, type="warrant", status="expired",
            created_at="2021-01-01", created_accession="orig-acc",
            terms_json='{"strike": 5.0, "expiration": "2026-01-01"}',
            outstanding_json='{"count": 0}',
        )
        m = CreateWarrant(count=10000, strike=5.0, event_date=date(2021, 1, 10),
                          expiration=date(2028, 6, 1))
        assert _redisclose(m, filing_date="2025-08-14") is None


class TestGuardScope:
    def test_active_row_still_takes_priority(self, temp_db):
        """When an ACTIVE row matches, the existing active-path dedup wins —
        the guard is only a fallback for closed rows."""
        temp_db.add_company(CIK, "XTIA")
        temp_db.add_instrument(
            "P-500", cik=CIK, type="preferred", status="active",
            created_at="2024-03-12", created_accession="orig-acc",
            terms_json='{"series_letter": "9"}',
            outstanding_json='{"count": 1000}',
        )
        m = CreatePreferred(count=1000, series_letter="9",
                            event_date=date(2024, 3, 12))
        assert _redisclose(m, filing_date="2025-04-15") == "P-500"
