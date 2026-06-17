"""Warrant split-via-misread-strike dedup in ``store._create_already_recorded``.

Serial-diluter warrant ladders ratchet, so the LLM copies a DIFFERENT
exercise price for the SAME offering across disclosures — a 424B5 then a
10-Q re-statement of the same tranche, or two ``create_warrant`` calls from
one filing. The strike-keyed dedup then fails (strikes are >2% apart) and a
phantom partial-count duplicate is born (the CELU 2026-06-12 finding:
April-2023 923,077 split into $7.50/435,625 + $3.50/487,451; March-2023
938,184 into $30 + $1.69/75,000). The fix collapses such a create onto the
existing active row on a same canonical-LABEL + same INITIAL_COUNT match
within the window — distinct tranches differ in issued size or
series_letter, but one offering keeps its issued count and month-year label.
"""

from datetime import date

import db
from dilution.ledger.mutations import CreateWarrant
from dilution.ledger.store import _create_already_recorded

CIK = 1752828  # CELU, for verisimilitude — temp_db is empty regardless.


def _recorded(m, *, filing_date: str, accession: str):
    with db.get_conn() as conn:
        return _create_already_recorded(conn, CIK, m, filing_date,
                                        accession=accession)


def _seed_april2023(temp_db, *, strike=7.5, initial_count=923077,
                    label="April 2023 Common Warrants",
                    accession="424B5-acc", series_letter=None):
    temp_db.add_company(CIK, "CELU")
    terms = {"strike": strike}
    if series_letter is not None:
        terms["series_letter"] = series_letter
    import json
    temp_db.add_instrument(
        "W-1", cik=CIK, type="warrant", status="active",
        created_at="2023-04-07", created_accession=accession,
        label=label,
        terms_json=json.dumps(terms),
        outstanding_json=json.dumps(
            {"count": initial_count, "initial_count": initial_count}),
    )


class TestWarrantSplitDedup:
    def test_misread_strike_same_label_same_count_collapses_same_filing(
            self, temp_db):
        """Two create_warrant calls from ONE filing for one offering, at
        divergent (mis-read) strikes — collapse on label+initial_count."""
        _seed_april2023(temp_db, strike=7.5, accession="one-acc")
        # Same offering, different strike read; SAME filing/accession.
        m = CreateWarrant(count=923077, strike=3.5, event_date=date(2023, 4, 10),
                          descriptor="Common")
        assert m.outstanding["initial_count"] == 923077  # guard the premise
        assert _recorded(m, filing_date="2023-04-10",
                         accession="one-acc") == "W-1"

    def test_misread_strike_cross_filing_redisclosure_collapses(self, temp_db):
        """A later 10-Q re-states the SAME April-2023 tranche at a drifted
        strike; the walker dates it to the original offering. Must collapse
        rather than spawn a duplicate (the CELU W-5277/W-5280 bug)."""
        _seed_april2023(temp_db, strike=7.5, accession="424B5-acc")
        m = CreateWarrant(count=923077, strike=3.5, event_date=date(2023, 4, 10),
                          descriptor="Common")
        # Different accession (the 10-Q), filed months later.
        assert _recorded(m, filing_date="2023-08-14",
                         accession="10Q-acc") == "W-1"

    def test_different_initial_count_stays_distinct(self, temp_db):
        """Genuine distinct tranches differ in size even at the same label
        (the SCNI Inducement/Series-B precedent) — do NOT collapse."""
        _seed_april2023(temp_db, strike=7.5, initial_count=923077)
        m = CreateWarrant(count=250000, strike=3.5, event_date=date(2023, 4, 10),
                          descriptor="Common")
        assert _recorded(m, filing_date="2023-04-10",
                         accession="other-acc") is None

    def test_conflicting_series_letter_stays_distinct(self, temp_db):
        """Same count + same month-label but a hard series_letter conflict
        (A vs B) means two real tranches — must not collapse."""
        _seed_april2023(temp_db, strike=7.5, initial_count=923077,
                        series_letter="A")
        m = CreateWarrant(count=923077, strike=3.5, event_date=date(2023, 4, 10),
                          descriptor="Common", series_letter="B")
        assert _recorded(m, filing_date="2023-04-10",
                         accession="other-acc") is None

    def test_missing_existing_initial_count_does_not_collapse(self, temp_db):
        """When the existing row exposes no initial_count there is nothing to
        compare — fall through rather than collapse on label alone."""
        import json
        temp_db.add_company(CIK, "CELU")
        temp_db.add_instrument(
            "W-1", cik=CIK, type="warrant", status="active",
            created_at="2023-04-07", created_accession="424B5-acc",
            label="April 2023 Common Warrants",
            terms_json=json.dumps({"strike": 7.5}),
            outstanding_json=json.dumps({"count": 923077}),  # no initial_count
        )
        m = CreateWarrant(count=923077, strike=3.5, event_date=date(2023, 4, 10),
                          descriptor="Common")
        assert _recorded(m, filing_date="2023-04-10",
                         accession="10Q-acc") is None

    def test_different_month_label_does_not_collapse(self, temp_db):
        """Same issued count but a different month-year label is a different
        offering — the label gate keeps them apart."""
        _seed_april2023(temp_db, strike=7.5, initial_count=923077)
        m = CreateWarrant(count=923077, strike=3.5, event_date=date(2023, 7, 31),
                          descriptor="Common")  # → "July 2023 Common Warrants"
        assert _recorded(m, filing_date="2023-07-31",
                         accession="other-acc") is None

    def test_matching_strike_still_collapses_via_primary_key(self, temp_db):
        """Sanity: when the strike DOES match, the primary strike key still
        fires (the new fallback doesn't disturb the existing path)."""
        _seed_april2023(temp_db, strike=4.5, initial_count=923077)
        m = CreateWarrant(count=923077, strike=4.5, event_date=date(2023, 4, 10),
                          descriptor="Common")
        assert _recorded(m, filing_date="2023-04-10",
                         accession="other-acc") == "W-1"
