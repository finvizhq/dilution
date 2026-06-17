"""Full-preferred-conversion close (KSCP Cluster H).

When a periodic filing affirms that ALL preferred stock automatically
converted to common with none remaining outstanding, the walker
deterministically closes the lingering active preferreds — the overhang
LLM re-matches the named series in the conversion narrative without
flagging is_terminated, so the anchor never closes them (KSCP Series
A/B/M/S, converted 2024-05-15, left phantom-active).

Two pieces, both unit-tested here:
  - ``walker._full_preferred_conversion_date`` — the pure detector;
  - ``store.close_converted_preferred`` — the deterministic close.
"""

from datetime import date

import db
from dilution.ledger.store import close_converted_preferred
from dilution.ledger.walker import _full_preferred_conversion_date


# Mirrors KSCP's Q2-2024 10-Q wording, INCLUDING the EDGAR \xa0 non-breaking
# space inside the date that broke a literal-space regex during development.
KSCP_TEXT = (
    "On May\xa015, 2024 (the “Preferred Stock Conversion Date”), "
    "pursuant to the terms of the Certificate of Incorporation, each share "
    "of the Company’s Super Voting Preferred Stock (as defined in the "
    "Certificate of Incorporation) was automatically converted into "
    "fully-paid, non-assessable shares of Class B common stock and each "
    "share of the Company’s Ordinary Preferred Stock was automatically "
    "converted into Class A common stock. As a result of the Automatic "
    "Conversion, there were no shares of Preferred Stock outstanding after "
    "the Preferred Stock Conversion Date."
)


class TestDetector:
    def test_kscp_positive_returns_conversion_date(self):
        assert _full_preferred_conversion_date(KSCP_TEXT) == date(2024, 5, 15)

    def test_date_anchored_to_conversion_verb_when_no_conversion_date_label(self):
        t = ("On June 1, 2023, all of the Company's preferred stock was "
             "mandatorily converted into shares of common stock; as a result "
             "no shares of preferred stock remained outstanding thereafter.")
        assert _full_preferred_conversion_date(t) == date(2023, 6, 1)

    def test_partial_per_series_does_not_fire(self):
        # Names a single series, not the general "preferred stock ... none
        # outstanding" affirmation → must not fire (other series may be live).
        t = ("No Series A Preferred Stock was outstanding. The Series A "
             "Preferred Stock was automatically converted into common stock "
             "on May 1, 2024.")
        assert _full_preferred_conversion_date(t) is None

    def test_convertible_boilerplate_does_not_fire(self):
        t = ("The convertible preferred stock is convertible into common "
             "stock at the holder's option. 1,000 shares of preferred stock "
             "were issued and outstanding as of the period end.")
        assert _full_preferred_conversion_date(t) is None

    def test_conditional_pro_forma_does_not_fire(self):
        # No conversion ACTUALITY ("would be") → the conv gate fails.
        t = ("After giving effect to the offering, there would be no shares "
             "of preferred stock outstanding.")
        assert _full_preferred_conversion_date(t) is None

    def test_zero_affirmation_without_conversion_does_not_fire(self):
        t = ("The Company had no shares of preferred stock outstanding as of "
             "December 31, 2024.")
        assert _full_preferred_conversion_date(t) is None

    def test_conversion_without_zero_affirmation_does_not_fire(self):
        t = ("On May 15, 2024, 5,000 shares of preferred stock were "
             "automatically converted into common stock.")
        assert _full_preferred_conversion_date(t) is None

    def test_empty_text(self):
        assert _full_preferred_conversion_date("") is None


class TestCloseConvertedPreferred:
    CIK = 1600983  # KSCP

    def _stage(self, temp_db):
        temp_db.add_company(self.CIK, "KSCP")
        # 4 pre-conversion preferreds (issued 2023-11-13) — should close.
        for iid, series, cnt in (("P-453", "A", 28368), ("P-454", "B", 69977),
                                 ("P-455", "M", 35512), ("P-456", "S", 52405)):
            temp_db.add_instrument(
                iid, cik=self.CIK, type="preferred", status="active",
                created_at="2023-11-13",
                terms_json=f'{{"series_letter": "{series}"}}',
                outstanding_json=f'{{"count": {cnt}}}',
            )

    def test_closes_pre_conversion_preferreds_and_zeros_count(self, temp_db):
        self._stage(temp_db)
        closed = close_converted_preferred(
            self.CIK, conversion_date=date(2024, 5, 15),
            accession="acc-q2", form="10-Q", filing_date="2024-08-14")
        assert set(closed) == {"P-453", "P-454", "P-455", "P-456"}
        rows = temp_db.execute(
            "SELECT instrument_id, status, outstanding_json FROM "
            "dilution_ledger WHERE cik=? AND type='preferred'", (self.CIK,))
        import json
        for r in rows:
            assert r["status"] == "converted"
            out = json.loads(r["outstanding_json"])
            assert out["count"] == 0
            assert out.get("count_converted_to_date", 0) > 0

    def test_spares_preferred_issued_after_conversion(self, temp_db):
        self._stage(temp_db)
        # A NEW Series A issued AFTER the conversion (re-used letter) — the
        # created_at scope must NOT sweep it up.
        temp_db.add_instrument(
            "P-500", cik=self.CIK, type="preferred", status="active",
            created_at="2025-02-01", terms_json='{"series_letter": "A"}',
            outstanding_json='{"count": 9999}')
        closed = close_converted_preferred(
            self.CIK, conversion_date=date(2024, 5, 15),
            accession="acc-q2", form="10-Q", filing_date="2024-08-14")
        assert "P-500" not in closed
        row = temp_db.execute(
            "SELECT status FROM dilution_ledger WHERE instrument_id='P-500'")[0]
        assert row["status"] == "active"

    def test_idempotent_on_already_closed(self, temp_db):
        self._stage(temp_db)
        close_converted_preferred(
            self.CIK, conversion_date=date(2024, 5, 15),
            accession="acc-q2", form="10-Q", filing_date="2024-08-14")
        # Re-fire (a later filing repeats the conversion note) → no-op.
        again = close_converted_preferred(
            self.CIK, conversion_date=date(2024, 5, 15),
            accession="acc-q3", form="10-Q", filing_date="2024-11-14")
        assert again == []

    def test_only_touches_preferred_type(self, temp_db):
        self._stage(temp_db)
        temp_db.add_instrument(
            "W-1", cik=self.CIK, type="warrant", status="active",
            created_at="2023-01-01",
            terms_json='{"strike": 5.0}', outstanding_json='{"count": 100}')
        close_converted_preferred(
            self.CIK, conversion_date=date(2024, 5, 15),
            accession="acc", form="10-Q", filing_date="2024-08-14")
        w = temp_db.execute(
            "SELECT status FROM dilution_ledger WHERE instrument_id='W-1'")[0]
        assert w["status"] == "active"
