"""Unit tests for dilution/finviz_payload.py.

Covers the wire-format translation layer only — the scalar normalizers,
the per-type card serializers (every internal→contract rename and the
producer-side filtering the contract promises), the cash / O/S-chart /
badge block shapes, and `build_payload`'s assembly with every fetch seam
monkeypatched. NO network / edgar / Finviz call is ever made.

The contract these assertions encode is FINVIZ_API_CONTRACT.md; section
numbers in the test names refer to it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import pytest

from dilution import finviz_payload as fp
from dilution.ledger.shelf_status import WKSI_UNLIMITED_SHELF_CAPACITY_USD


# ── §4 scalar conventions ────────────────────────────────────────────
class TestScalars:
    @pytest.mark.parametrize("value,expected", [
        (date(2026, 3, 31), "2026-03-31"),
        (datetime(2026, 3, 31, 14, 5), "2026-03-31"),
        ("2026-03-31", "2026-03-31"),
        (None, None),
        ("", None),
    ])
    def test_iso(self, value, expected):
        assert fp._iso(value) == expected

    @pytest.mark.parametrize("value,expected", [
        ("Yes", True), ("yes", True), ("No", False), ("no", False),
        (True, True), (False, False),
        # None must survive as None — "not disclosed" is not "false" (§4).
        (None, None),
        ("—", None), ("maybe", None),
    ])
    def test_bool(self, value, expected):
        assert fp._bool(value) is expected

    def test_num_coerces_and_rejects(self):
        assert fp._num("12.5") == 12.5
        assert fp._num(7) == 7
        assert fp._num(None) is None
        assert fp._num("n/a") is None
        # bools are ints in Python; they must not leak in as numbers.
        assert fp._num(True) is None

    def test_strs_always_a_list(self):
        assert fp._strs(None) == []
        assert fp._strs([]) == []
        assert fp._strs("Alpha Fund") == ["Alpha Fund"]
        assert fp._strs(["A", "B"]) == ["A", "B"]


# ── §7.0 shared sub-objects ──────────────────────────────────────────
class TestSubObjects:
    def test_parent_shelf_drops_internal_id(self):
        out = fp._parent_shelf({"parent_shelf": {
            "instrument_id": "shelf-1", "title": "March 2024 Shelf",
            "file_number": "333-1", "accession_number": "0001-24-1",
            "edgar_url": "https://sec.gov/x",
        }})
        assert out == {"title": "March 2024 Shelf", "file_number": "333-1",
                       "accession_number": "0001-24-1",
                       "edgar_url": "https://sec.gov/x"}

    def test_parent_shelf_absent_is_none(self):
        assert fp._parent_shelf({}) is None
        assert fp._parent_shelf({"parent_shelf": None}) is None

    def test_resale_registration_normalizes_date(self):
        out = fp._resale({"resale_registration": {
            "form": "S-1", "filing_date": date(2024, 12, 20),
            "file_number": "333-2", "accession_number": "0001-24-2",
            "edgar_url": "https://sec.gov/y", "instrument_id": "nope",
        }})
        assert out["filing_date"] == "2024-12-20"
        assert "instrument_id" not in out

    def test_head_maps_instrument_id_to_source_ref(self):
        head = fp._head({"instrument_id": "W-3034", "title": "T",
                         "registered": "Registered", "edgar_url": None})
        assert head["source_ref"] == "W-3034"
        assert "instrument_id" not in head


# ── §7.1 shelf ───────────────────────────────────────────────────────
class TestShelfCard:
    def _card(self, **over):
        card = {
            "instrument_id": "shelf-1", "title": "September 2024 Shelf",
            "registered": "Registered", "shelf_status": "active",
            "edgar_url": "https://sec.gov/s",
            "current_raisable_amount": 1_374_528.4,
            "total_shelf_capacity": 30_000_000.0,
            "baby_shelf_restriction": "Yes",
            "total_amount_raised": 11_263_748.3,
            "raised_last_12mo_under_ib6": 0.0,
            "outstanding_shares": 6_259_279.0,
            "float": 6_200_880.0,
            "highest_60_day_close": 0.665,
            "price_to_exceed_baby_shelf": 12.095,
            "ib6_float_value": 4_123_585.2,
            "last_banker": "Dawson James",
            "effect_date": date(2024, 10, 3),
            "expiration_date": date(2027, 10, 3),
            "last_update_date": "2026-03-30",
            "bank_tier": "boutique", "investor_class": None,
        }
        card.update(over)
        return card

    def test_renames_and_booleans(self):
        out = fp._shelf_card(self._card())
        assert out["is_baby_shelf_restricted"] is True
        assert out["float_shares"] == 6_200_880.0
        assert out["unlimited"] is False
        assert out["current_raisable_amount"] == 1_374_528.4
        assert "baby_shelf_restriction" not in out
        assert "float" not in out

    def test_wksi_sentinel_raisable_becomes_unlimited(self):
        """An ASR shelf carries the sentinel in raisable while its
        capacity stays the finite cumulative registered figure (FCEL
        October 2023): unlimited flips true, the amount nulls, the
        capacity survives."""
        out = fp._shelf_card(self._card(
            current_raisable_amount=float(WKSI_UNLIMITED_SHELF_CAPACITY_USD),
            total_shelf_capacity=405_000_000.0,
            baby_shelf_restriction="No"))
        assert out["unlimited"] is True
        assert out["current_raisable_amount"] is None
        assert out["total_shelf_capacity"] == 405_000_000.0

    def test_wksi_sentinel_capacity_nulls_both(self):
        out = fp._shelf_card(self._card(
            current_raisable_amount=float(WKSI_UNLIMITED_SHELF_CAPACITY_USD),
            total_shelf_capacity=float(WKSI_UNLIMITED_SHELF_CAPACITY_USD)))
        assert out["unlimited"] is True
        assert out["total_shelf_capacity"] is None
        assert out["current_raisable_amount"] is None

    def test_baby_shelf_capped_wksi_keeps_its_number(self):
        """A restricted issuer's I.B.6 raisable is a real number and must
        render — the sentinel never reaches this path, so `unlimited`
        stays false."""
        out = fp._shelf_card(self._card(current_raisable_amount=468_275.5,
                                        total_shelf_capacity=150_000_000.0))
        assert out["unlimited"] is False
        assert out["current_raisable_amount"] == 468_275.5


# ── §7.2 atm ─────────────────────────────────────────────────────────
class TestAtmCard:
    def _card(self, **over):
        card = {
            "instrument_id": "ATM-1", "title": "December 2024 ATM",
            "registered": "Registered", "edgar_url": None,
            "parent_shelf": None,
            "total_capacity": 8_230_000.0,
            # Card-layer semantics: remaining_capacity == the CONTRACTUAL
            # remaining; raisable_capped == the I.B.6-capped raisable.
            "remaining_capacity": 5_000_000.0,
            "remaining_without_baby_shelf": 5_000_000.0,
            "raisable_capped": 1_400_000.0,
            "limited_by_baby_shelf": "Yes",
            "placement_agent": "Dawson James",
            "sales_total_usd": 3_230_000.0,
            "used_pct": 39.2,
            "agreement_start_date": "2024-12-17",
            "agreement_end_date": None,
            "last_update_date": "2026-03-30",
            "bank_tier": "boutique", "investor_class": None,
        }
        card.update(over)
        return card

    def test_headline_remaining_is_the_ib6_capped_number(self):
        out = fp._atm_card(self._card())
        assert out["remaining_capacity"] == 1_400_000.0
        assert out["remaining_without_baby_shelf"] == 5_000_000.0
        assert out["limited_by_baby_shelf"] is True
        assert "raisable_capped" not in out

    def test_zero_capped_remaining_is_not_treated_as_missing(self):
        """A fully-drawn baby-shelf ATM has raisable_capped == 0.0; a
        truthiness fallback would resurrect the contractual remaining."""
        out = fp._atm_card(self._card(raisable_capped=0.0))
        assert out["remaining_capacity"] == 0.0

    def test_missing_capped_falls_back_to_contractual(self):
        out = fp._atm_card(self._card(raisable_capped=None))
        assert out["remaining_capacity"] == 5_000_000.0


# ── §7.3–§7.7 remaining card types ───────────────────────────────────
class TestOtherCards:
    def test_equity_line_terminated_is_a_bool(self):
        out = fp._equity_line_card({
            "instrument_id": "EL-1", "title": "March 2025 YA II ELOC",
            "registered": "Terminated", "terminated": True,
            "remaining_capacity": 6_928_097.7, "total_capacity": 10_000_000.0,
            "counterparty": "Yorkville", "sales_total_usd": 3_071_902.3,
            "used_pct": 30.7,
        })
        assert out["terminated"] is True
        assert out["counterparty"] == "Yorkville"

    def test_warrant_owners_and_dates(self):
        out = fp._warrant_card({
            "instrument_id": "W-1", "title": "July 2024 Common Warrants",
            "registered": "Not Registered", "known_owners": ["Hybrid 12"],
            "total_issued": 250.0, "remaining_outstanding": 250.0,
            "exercise_price": 5490.0, "issue_date": date(2024, 7, 1),
            "expiration_date": "2029-07-01",
        })
        assert out["known_owners"] == ["Hybrid 12"]
        assert out["issue_date"] == "2024-07-01"
        assert out["exercisable_date"] is None

    def test_convertible_and_preferred_share_one_field_set(self):
        card = {"instrument_id": "C-1", "title": "Series A Preferred",
                "principal_total": 3_600_000.0,
                "principal_remaining": 2_012_000.0,
                "conversion_price": 4.87,
                "total_shares_issuable": 739_219.7,
                "remaining_shares_issuable": 413_141.7}
        out = fp._convertible_card(card)
        assert out["principal_remaining"] == 2_012_000.0
        assert out["remaining_shares_issuable"] == 413_141.7
        assert set(out) >= {"conversion_price", "maturity_date",
                            "convertible_date", "known_owners"}
        # Privately placed paper never links a shelf, and the contract
        # doesn't define the field for these types (§7.5/§7.6).
        assert "parent_shelf" not in out

    def test_s1_drops_the_duplicate_display_status(self):
        out = fp._s1_card({
            "instrument_id": "S1-1", "title": "September 2024 S-1 Offering",
            "registered": "Priced", "status": "Priced", "s1_status": "priced",
            "anticipated_deal_size": 10_000_000.0, "final_pricing": 1.39,
            "warrant_coverage_pct": 200.0, "filing_date": "2024-09-16",
        })
        assert out["s1_status"] == "priced"
        assert out["registered"] == "Priced"
        assert "status" not in out


# ── §7 cards block ───────────────────────────────────────────────────
class TestCardsBlock:
    def test_seven_keys_always_present_and_preferred_renamed(self):
        block = fp._cards_block({"convertible_preferred": [
            {"instrument_id": "P-1", "title": "Series C Preferred"}]})
        assert set(block) == {"shelf", "atm", "equity_line", "warrant",
                              "convertible", "preferred", "s1_offering"}
        assert block["preferred"][0]["title"] == "Series C Preferred"
        assert block["warrant"] == []

    def test_expired_and_withdrawn_shelves_are_never_pushed(self):
        block = fp._cards_block({"shelf": [
            {"instrument_id": "s1", "title": "Live", "shelf_status": "active"},
            {"instrument_id": "s2", "title": "Pending",
             "shelf_status": "registered"},
            {"instrument_id": "s3", "title": "Old", "shelf_status": "expired"},
            {"instrument_id": "s4", "title": "Pulled",
             "shelf_status": "withdrawn"},
        ]})
        assert [c["title"] for c in block["shelf"]] == ["Live", "Pending"]

    def test_display_order_is_preserved(self):
        block = fp._cards_block({"warrant": [
            {"instrument_id": f"W-{i}", "title": f"T{i}"} for i in range(4)]})
        assert [c["source_ref"] for c in block["warrant"]] == [
            "W-0", "W-1", "W-2", "W-3"]


# ── §5.1 cash ────────────────────────────────────────────────────────
@dataclass
class FakeCash:
    series: list = field(default_factory=list)
    latest_period_end: date | None = None
    latest_cash_usd: float | None = None
    op_cf_quarterly_usd: float | None = None
    op_cf_prorated_usd: float | None = None
    capital_raised_usd: float | None = None
    current_cash_est_usd: float | None = None
    months_of_cash: float | None = None
    as_of: date = date(2026, 7, 27)
    stale_days: int | None = None
    fx_failed: bool = False


def _cash_point(end, value, fy, fp_, form="10-Q"):
    return {"end": end, "value_usd": value, "fy": fy, "fp": fp_,
            "accession": "0001-1", "form": form,
            "native_currency": "USD", "native_value": value}


class TestFiscalLabel:
    def test_matching_year_is_labelled(self):
        assert fp._fiscal_label(
            _cash_point(date(2025, 9, 30), 1.0, 2025, "Q3")) == "2025 Q3"

    def test_comparative_balance_from_a_later_filing_is_dropped(self):
        # GCTK: the 2025-12-31 year-end balance restated in the fy2026 Q1
        # 10-Q would otherwise render as "2026 Q1".
        assert fp._fiscal_label(
            _cash_point(date(2025, 12, 31), 1.0, 2026, "Q1")) is None

    def test_wildly_off_tags_are_dropped(self):
        assert fp._fiscal_label(
            _cash_point(date(2016, 12, 31), 1.0, 2018, "Q3")) is None

    @pytest.mark.parametrize("fp_", ["", "H1", None, "FY2025"])
    def test_unrecognized_period_is_dropped(self, fp_):
        assert fp._fiscal_label(
            _cash_point(date(2025, 9, 30), 1.0, 2025, fp_)) is None


class TestCashBlock:
    def test_none_and_empty_series_are_omitted(self):
        assert fp._cash_block(None) is None
        assert fp._cash_block(FakeCash(series=[])) is None

    def test_bars_end_with_a_single_estimate_carrying_the_overlay(self):
        cash = FakeCash(
            series=[_cash_point(date(2026, 3, 31), 4_120_000.0, 2026, "Q1")],
            latest_period_end=date(2026, 3, 31),
            latest_cash_usd=4_120_000.0, op_cf_quarterly_usd=-2_210_000.0,
            capital_raised_usd=1_200_000.0, current_cash_est_usd=3_650_000.0,
            months_of_cash=5.6, stale_days=63)
        block = fp._cash_block(cash)
        assert block["latest_reported_cash_usd"] == 4_120_000.0
        assert block["capital_raised_since_usd"] == 1_200_000.0
        bars = block["chart"]["bars"]
        assert [b["kind"] for b in bars] == ["reported", "estimate"]
        assert bars[0] == {"kind": "reported", "period_end": "2026-03-31",
                           "fiscal": "2026 Q1", "form": "10-Q",
                           "cash_usd": 4_120_000.0, "overlay_usd": None}
        # The estimate bar equals current_cash_est_usd by construction.
        assert bars[-1]["cash_usd"] == block["current_cash_est_usd"]
        assert bars[-1]["overlay_usd"] == 1_200_000.0
        assert bars[-1]["period_end"] is None

    def test_no_raises_means_no_overlay_segment(self):
        cash = FakeCash(
            series=[_cash_point(date(2026, 3, 31), 1.0, 2026, "Q1")],
            capital_raised_usd=0.0, current_cash_est_usd=-1_483_494.4)
        bars = fp._cash_block(cash)["chart"]["bars"]
        assert bars[-1]["overlay_usd"] is None
        # A negative estimate is legitimate (burn exceeds cash) — it must
        # survive as-is for the consumer to plot below the axis.
        assert bars[-1]["cash_usd"] == -1_483_494.4

    def test_estimate_bar_omitted_when_unestimable(self):
        cash = FakeCash(
            series=[_cash_point(date(2026, 3, 31), 1.0, 2026, "Q1")],
            current_cash_est_usd=None)
        assert [b["kind"] for b in fp._cash_block(cash)["chart"]["bars"]] == [
            "reported"]


# ── §5.2 os_chart ────────────────────────────────────────────────────
@dataclass
class FakeOsPoint:
    quarter_end: date
    shares: float
    raw_shares: float
    source_date: date
    form: str
    carried: bool
    split_adjusted: bool


@dataclass
class FakeOsHistory:
    series: list = field(default_factory=list)
    as_of: date = date(2026, 7, 27)
    concept: str | None = None
    ads_ratio: float | None = None
    warnings: tuple = ()


@dataclass
class FakeSeg:
    key: str
    label: str
    shares: float
    note: str
    price_based: bool


class TestOsChartBlock:
    def _osh(self):
        return FakeOsHistory(series=[FakeOsPoint(
            date(2026, 3, 31), 3_470_000.0, 3_470_000.0, date(2026, 5, 12),
            "10-Q", False, True)])

    def test_bars_and_latest(self):
        block = fp._os_chart_block(self._osh(), 3_470_000.0,
                                   "10-Q XBRL, a/o 2026-05-12", [], 2.18)
        assert block["bars"] == [{
            "quarter_end": "2026-03-31", "shares": 3_470_000.0,
            "raw_shares": 3_470_000.0, "source_date": "2026-05-12",
            "form": "10-Q", "carried": False, "split_adjusted": True}]
        assert block["latest"] == {"shares": 3_470_000.0,
                                   "source": "10-Q XBRL, a/o 2026-05-12"}
        assert block["price_basis"] == 2.18

    def test_price_based_segment_ships_its_dollar_capacity(self):
        """capacity_usd must invert the capacity ÷ basis the segment was
        built from, so the consumer can recompute against a live price."""
        segs = [FakeSeg("warrant", "Warrants", 480_000.0, "n1", False),
                FakeSeg("atm", "ATM", 700_000.0, "n2", True)]
        stack = fp._os_chart_block(self._osh(), 3_470_000.0, "src",
                                   segs, 2.18)["fd_stack"]
        assert stack[0]["capacity_usd"] is None
        assert stack[0]["price_based"] is False
        assert stack[1]["capacity_usd"] == pytest.approx(1_526_000.0)
        assert stack[1]["key"] == "atm"

    def test_missing_price_basis_leaves_capacity_null(self):
        segs = [FakeSeg("atm", "ATM", 700_000.0, "n", True)]
        stack = fp._os_chart_block(self._osh(), 1.0, "src",
                                   segs, None)["fd_stack"]
        assert stack[0]["capacity_usd"] is None

    def test_omitted_when_nothing_to_plot(self):
        assert fp._os_chart_block(None, 1.0, "s", [], 2.0) is None
        assert fp._os_chart_block(FakeOsHistory(), None, "s", [], 2.0) is None

    def test_stack_alone_is_enough_to_render(self):
        segs = [FakeSeg("warrant", "Warrants", 10.0, "n", False)]
        block = fp._os_chart_block(FakeOsHistory(), 100.0, "s", segs, 2.0)
        assert block["bars"] == []
        assert len(block["fd_stack"]) == 1


# ── §6 badges ────────────────────────────────────────────────────────
@dataclass
class FakeBadge:
    key: str
    label: str
    description: str
    band: str | None
    band_text: str
    score: int | None
    detail: tuple = ()
    legend: tuple = ()


@dataclass
class FakeBadgeSet:
    overall_score: int | None = 72
    overall_band: str | None = "high"
    overall_label: str = "High"
    partial: bool = False
    interaction: bool = True
    description: str = "0–100 composite"
    detail: tuple = ("Offering Ability 81 (weight 30%)",)
    legend: tuple = (("high", "High", "60–79"),)
    drivers: tuple = ()


class TestBadgesBlock:
    def test_null_when_uncomputable(self):
        assert fp._badges_block(None) is None

    def test_overall_and_drivers(self):
        badges = FakeBadgeSet(drivers=(FakeBadge(
            "offering_ability", "Offering Ability", "desc", "high", "High", 81,
            detail=("Active ATM: $1.5M raisable",),
            legend=(("low", "Low", "no capacity"),)),))
        block = fp._badges_block(badges)
        assert block["overall"]["score"] == 72
        assert block["overall"]["legend"] == [
            {"band": "high", "pill": "High", "meaning": "60–79"}]
        assert block["overall"]["detail"] == [
            "Offering Ability 81 (weight 30%)"]
        driver = block["drivers"][0]
        assert driver["key"] == "offering_ability"
        assert driver["band_text"] == "High"
        assert driver["legend"] == [
            {"band": "low", "pill": "Low", "meaning": "no capacity"}]
        # The internal High×High interaction flag is not part of the
        # contract — the bump is already inside the score and detail.
        assert "interaction" not in block["overall"]

    def test_unscorable_partial_composite(self):
        block = fp._badges_block(FakeBadgeSet(
            overall_score=None, overall_band=None, overall_label="—",
            partial=True))
        assert block["overall"]["score"] is None
        assert block["overall"]["partial"] is True


# ── §8 brief ─────────────────────────────────────────────────────────
_CACHED_BRIEF = {
    "headline": "CELU faces severe dilution",
    "bullets": ["Cash is $531K.", "A pending S-1 seeks $17.9M."],
    "watch": ["October 16, 2026: Maturity of the Helena note."],
    "facts_hash": "abc123",
    "generated_at": "2026-06-04T14:33:01Z",
    "model": "gemini-3.5-flash",
}


class TestBriefBlock:
    @pytest.fixture
    def cached(self, monkeypatch):
        monkeypatch.setattr(fp.brief_mod, "get_cached",
                            lambda cik: dict(_CACHED_BRIEF))

    def test_null_when_nothing_cached(self, monkeypatch):
        monkeypatch.setattr(fp.brief_mod, "get_cached", lambda cik: None)
        assert fp._brief_block(1) is None

    def test_null_when_the_lookup_blows_up(self, monkeypatch):
        def _boom(_cik):
            raise RuntimeError("no such table")
        monkeypatch.setattr(fp.brief_mod, "get_cached", _boom)
        assert fp._brief_block(1) is None

    def test_fresh_brief(self, cached, monkeypatch):
        monkeypatch.setattr(fp, "_latest_filing_date", lambda cik: "2026-06-01")
        block = fp._brief_block(1)
        assert block["headline"] == "CELU faces severe dilution"
        assert block["bullets"] == _CACHED_BRIEF["bullets"]
        assert block["watch"] == _CACHED_BRIEF["watch"]
        assert block["generated_at"] == "2026-06-04T14:33:01Z"
        assert block["stale"] is False
        assert block["stale_since_filing_date"] is None
        # Cache-internal fields stay internal.
        assert "facts_hash" not in block
        assert "model" not in block

    def test_filing_after_generation_is_stale(self, cached, monkeypatch):
        monkeypatch.setattr(fp, "_latest_filing_date", lambda cik: "2026-06-12")
        block = fp._brief_block(1)
        assert block["stale"] is True
        assert block["stale_since_filing_date"] == "2026-06-12"

    def test_same_day_filing_is_not_stale(self, cached, monkeypatch):
        # The rule compares dates, not timestamps: a filing on the
        # generation date is assumed to have been in the facts.
        monkeypatch.setattr(fp, "_latest_filing_date", lambda cik: "2026-06-04")
        assert fp._brief_block(1)["stale"] is False

    def test_no_filings_at_all_is_not_stale(self, cached, monkeypatch):
        monkeypatch.setattr(fp, "_latest_filing_date", lambda cik: None)
        assert fp._brief_block(1)["stale"] is False

    def test_filing_lookup_failure_degrades_to_fresh(self, cached,
                                                     monkeypatch):
        def _boom(_cik):
            raise RuntimeError("db gone")
        monkeypatch.setattr(fp, "_latest_filing_date", _boom)
        block = fp._brief_block(1)
        assert block["stale"] is False
        assert block["headline"] == "CELU faces severe dilution"

    def test_missing_watch_becomes_an_empty_list(self, monkeypatch):
        row = dict(_CACHED_BRIEF)
        row["watch"] = None
        row["bullets"] = None
        monkeypatch.setattr(fp.brief_mod, "get_cached", lambda cik: row)
        monkeypatch.setattr(fp, "_latest_filing_date", lambda cik: None)
        block = fp._brief_block(1)
        assert block["watch"] == []
        assert block["bullets"] == []


# ── §4 assembly ──────────────────────────────────────────────────────
class TestBuildPayload:
    """Every fetch seam is stubbed: assembly only, no I/O."""

    @pytest.fixture
    def stubs(self, monkeypatch):
        monkeypatch.setattr(fp, "_company_row", lambda t: {
            "cik": 1506983, "ticker": "GCTK", "name": "Glucotrack, Inc."})
        monkeypatch.setattr(fp, "finviz_fundamentals", lambda t: {
            "ticker": t, "price": 0.29, "shares_outstanding": 7_720_000.0,
            "float_shares": 6_270_000.0})
        monkeypatch.setattr(fp, "fetch_implied_outstanding_cached", lambda cik: (
            type("Implied", (), {"total": 6_259_279.0, "source_form": "10-Q",
                                 "as_of": date(2026, 5, 14)})()))
        monkeypatch.setattr(fp, "_internal_cards", lambda cik, f, os_: {
            "s1_offering": [], "warrant": [
                {"instrument_id": "W-1", "title": "W", "registered": "—",
                 "remaining_outstanding": 100.0}],
            "convertible": [], "convertible_preferred": [], "atm": [],
            "equity_line": [], "shelf": []})
        monkeypatch.setattr(fp, "latest_settled_close",
                            lambda t: (date(2026, 7, 27), 0.304))
        monkeypatch.setattr(fp, "_resolve_float_shares",
                            lambda cik, f, os_: 6_200_880.0)
        monkeypatch.setattr(fp, "highest_close", lambda t, bars=60: 0.665)
        monkeypatch.setattr(fp, "is_baby_shelf_restricted",
                            lambda *a, **k: True)
        monkeypatch.setattr(fp, "_cash_and_raised", lambda cik: (None, None))
        monkeypatch.setattr(fp, "fetch_os_history_cached",
                            lambda cik: FakeOsHistory())
        monkeypatch.setattr(fp, "build_fd_stack", lambda cards, price: [])
        monkeypatch.setattr(fp, "compute_badges", lambda *a, **k: None)
        monkeypatch.setattr(fp, "_brief_block", lambda cik: None)

    def test_wire_body_is_the_ticker_plus_data_wrapper(self, stubs):
        doc = fp.build_payload("gctk")
        assert set(doc) == {"ticker", "data"}
        assert doc["ticker"] == "GCTK"          # upper-cased
        # The snapshot repeats it, so a detached `data` still identifies
        # itself (§4).
        assert doc["data"]["ticker"] == "GCTK"

    def test_snapshot_envelope(self, stubs):
        snap = fp.build_snapshot(
            "gctk", generated_at=datetime(2026, 7, 28, 11, 34, 55,
                                          tzinfo=timezone.utc))
        assert snap["schema_version"] == 1
        assert snap["ticker"] == "GCTK"
        assert snap["cik"] == 1506983
        assert snap["company_name"] == "Glucotrack, Inc."
        assert snap["as_of"] == "2026-07-27"    # settled session, not today
        assert snap["generated_at"] == "2026-07-28T11:34:55Z"
        assert set(snap) == {"schema_version", "ticker", "cik", "company_name",
                             "as_of", "generated_at", "company", "badges",
                             "cards", "brief"}

    def test_company_block_and_optional_sections(self, stubs):
        snap = fp.build_snapshot("GCTK")
        company = snap["company"]
        assert company["shares_outstanding"] == 6_259_279.0
        assert company["float_shares"] == 6_200_880.0
        assert company["highest_60_day_close"] == 0.665
        assert company["is_baby_shelf_restricted"] is True
        assert company["price_to_exceed_baby_shelf"] == pytest.approx(
            75_000_000.0 / 6_200_880.0)
        # cash is omitted (no XBRL), os_chart present but bar-less.
        assert "cash" not in company
        assert snap["badges"] is None
        assert snap["brief"] is None
        assert snap["cards"]["warrant"][0]["source_ref"] == "W-1"

    def test_json_serializable(self, stubs):
        json.dumps(fp.build_payload("GCTK"))

    def test_unknown_ticker_raises(self, stubs, monkeypatch):
        monkeypatch.setattr(fp, "_company_row", lambda t: None)
        with pytest.raises(LookupError):
            fp.build_payload("NOPE")

    def test_settled_close_failure_falls_back_to_today(self, stubs,
                                                       monkeypatch):
        def _boom(_ticker):
            raise RuntimeError("finviz down")
        monkeypatch.setattr(fp, "latest_settled_close", _boom)
        assert fp.build_snapshot("GCTK")["as_of"] == date.today().isoformat()

    def test_badge_failure_degrades_to_null(self, stubs, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("badges down")
        monkeypatch.setattr(fp, "compute_badges", _boom)
        assert fp.build_snapshot("GCTK")["badges"] is None

    def test_brief_rides_inside_data(self, stubs, monkeypatch):
        monkeypatch.setattr(fp, "_brief_block", lambda cik: {
            "headline": "h", "bullets": [], "watch": [],
            "generated_at": "2026-06-04T14:33:01Z", "stale": True,
            "stale_since_filing_date": "2026-06-12"})
        doc = fp.build_payload("GCTK")
        assert doc["data"]["brief"]["stale"] is True
        assert "brief" not in doc
