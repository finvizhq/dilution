"""Conditional preferred conv_price split-adjustment (BNKK Cluster A).

The store has TWO sites that adjust split-scaled $-terms:
  - ``_apply_split``               — the split-event divide;
  - ``_rescale_stale_unit_amend``  — amend-time stale-unit normalization.

Both formerly skipped {conv_price, conversion_price, stated_value} for EVERY
preferred (the IQST Series D §4(f) precedent: a fixed conversion RATE absorbs
the split, the dollar conv_price is a fixed VWAP reference). That blanket
exemption is wrong for a PRICE-based preferred whose CoD subjects the
conversion price to standard split anti-dilution — BNKK's 2025 Series B/C
(no conversion_ratio) stored the raw pre-split $0.34 / $0.5582 and rendered
as-converted shares ~35x inflated over the 1-for-35 reverse split.

The fix (shared ``_preferred_price_split_skip``, wired into both sites):
stated_value is ALWAYS fixed; conv_price/conversion_price are split-adjusted
for a preferred ONLY when no conversion_ratio is stored. These tests pin the
helper and both call sites, including the load-bearing interaction where a
post-split filing re-quotes the raw pre-split price and the amend pass must
echo-pin it back to the split-adjusted value (not clobber it).
"""

from datetime import date

import db
from dilution.ledger.mutations import ApplySplit
from dilution.ledger.store import (
    _apply_split,
    _preferred_price_split_skip,
    _rescale_stale_unit_amend,
)

import json

# 1-for-35 reverse split (BNKK, 2025-12-11): ratio = post/pre = 1/35.
_REVERSE_35 = ApplySplit(
    post=1, pre=35, direction="reverse",
    effective_date=date(2025, 12, 11), units="common",
)


# ─── the shared policy helper ─────────────────────────────────────────
class TestPreferredPriceSplitSkip:
    def test_ratio_present_skips_all_three_dollar_terms(self):
        # Rate-driven preferred (IQST D ratio 12.5): conv_price is a fixed
        # reference -> all three $-terms are split-invariant.
        assert _preferred_price_split_skip({"conversion_ratio": 12.5}) == {
            "conv_price", "conversion_price", "stated_value"}

    def test_ratio_absent_skips_only_stated_value(self):
        # Price-based preferred (BNKK B/C): conv_price moves with the split,
        # stated_value (liquidation face) stays fixed.
        assert _preferred_price_split_skip({"conv_price": 0.5582}) == {
            "stated_value"}

    def test_empty_terms_skips_only_stated_value(self):
        assert _preferred_price_split_skip({}) == {"stated_value"}

    def test_zero_ratio_treated_as_absent(self):
        assert _preferred_price_split_skip({"conversion_ratio": 0}) == {
            "stated_value"}

    def test_negative_ratio_treated_as_absent(self):
        assert _preferred_price_split_skip({"conversion_ratio": -5}) == {
            "stated_value"}

    def test_none_ratio_treated_as_absent(self):
        assert _preferred_price_split_skip({"conversion_ratio": None}) == {
            "stated_value"}

    def test_bool_ratio_not_mistaken_for_number(self):
        # bool is an int subclass; True must NOT count as a real ratio.
        assert _preferred_price_split_skip({"conversion_ratio": True}) == {
            "stated_value"}


# ─── site #1: _apply_split ────────────────────────────────────────────
class TestApplySplitPreferred:
    CIK = 1760903  # BNKK

    def _split_one(self, temp_db, *, terms, status="active",
                   created_at="2025-08-08", type="preferred", units=None):
        temp_db.add_company(self.CIK, "BNKK", is_fpi=0)
        t = dict(terms)
        if units is not None:
            t["units"] = units
        temp_db.add_instrument(
            "P-X", cik=self.CIK, type=type, status=status,
            created_at=created_at, terms_json=json.dumps(t),
            outstanding_json='{"count": 135000}')
        with db.get_conn() as conn:
            _apply_split(conn, self.CIK, _REVERSE_35,
                         "split:2025-12-11:finviz", "SPLIT", "2025-12-11")
        row = temp_db.execute(
            "SELECT terms_json FROM dilution_ledger WHERE instrument_id='P-X'"
        )[0]
        return json.loads(row["terms_json"])

    def test_price_based_preferred_conv_price_is_split_adjusted(self, temp_db):
        # BNKK Series C: 0.5582 (raw pre-split) x 35 = 19.537 ~= fixture 19.54.
        t = self._split_one(
            temp_db, terms={"series_letter": "C", "conv_price": 0.5582,
                            "stated_value": 1000})
        assert round(t["conv_price"], 3) == 19.537
        # stated_value (liquidation face) untouched.
        assert t["stated_value"] == 1000

    def test_conversion_price_field_also_adjusted(self, temp_db):
        # The alternate field name `conversion_price` follows the same rule.
        t = self._split_one(
            temp_db, terms={"series_letter": "B", "conversion_price": 0.34})
        assert round(t["conversion_price"], 2) == 11.90

    def test_ratio_bearing_preferred_conv_price_untouched(self, temp_db):
        # IQST-style: conversion_ratio present -> conv_price is a fixed
        # reference and must NOT be divided (the rate absorbs the split).
        t = self._split_one(
            temp_db, terms={"series_letter": "D", "conv_price": 7.6445952,
                            "conversion_ratio": 12.5, "stated_value": 95.55744})
        assert t["conv_price"] == 7.6445952
        assert t["conversion_ratio"] == 12.5
        assert t["stated_value"] == 95.55744

    def test_stated_value_never_adjusted_even_without_ratio(self, temp_db):
        t = self._split_one(
            temp_db, terms={"series_letter": "C", "conv_price": 0.5582,
                            "stated_value": 1000})
        assert t["stated_value"] == 1000  # liquidation face is split-invariant

    def test_warrant_strike_still_divides(self, temp_db):
        # Non-preferred is unaffected by the preferred policy: a warrant
        # strike divides as before (regression guard for the shared helper).
        t = self._split_one(
            temp_db, type="warrant",
            terms={"strike": 0.4348})
        assert round(t["strike"], 4) == round(0.4348 * 35, 4)  # 15.218

    def test_idempotent_when_split_already_applied(self, temp_db):
        # applied_splits already records this split -> no second divide.
        t = self._split_one(
            temp_db,
            terms={"series_letter": "C", "conv_price": 19.537,
                   "stated_value": 1000,
                   "applied_splits": [{"date": "2025-12-11",
                                       "ratio": 1 / 35,
                                       "direction": "reverse"}]})
        assert t["conv_price"] == 19.537  # unchanged (already applied)

    def test_split_on_or_before_creation_is_skipped(self, temp_db):
        # Split effective on/before the instrument's creation is already baked
        # into the disclosed (post-split) terms -> not re-applied.
        t = self._split_one(
            temp_db, created_at="2025-12-11",
            terms={"series_letter": "C", "conv_price": 19.537,
                   "stated_value": 1000})
        assert t["conv_price"] == 19.537


# ─── site #2: _rescale_stale_unit_amend ───────────────────────────────
class TestRescaleStaleUnitAmendPreferred:
    """A post-split filing that re-quotes the raw pre-split conv_price must
    be echo-pinned back to the split-adjusted current value — the behavior
    the old blanket preferred-skip blocked, causing BNKK Series C to revert
    to 0.5582 after _apply_split had correctly produced 19.537."""

    _SPLIT = [{"date": "2025-12-11", "ratio": 1 / 35, "direction": "reverse"}]

    def test_price_based_amend_echo_pinned_to_split_adjusted(self):
        # BNKK Series C: 2026-03-31 10-Q re-quotes 0.5582; current ledger is
        # the split-adjusted 19.537 -> implied factor 0.5582/19.537 == split
        # ratio -> echo pins it back to 19.537 (no clobber).
        terms = {"conv_price": 19.537, "stated_value": 1000,
                 "applied_splits": self._SPLIT}
        field_updates = {"conv_price": 0.5582}
        rescaled = _rescale_stale_unit_amend(
            "preferred", terms, {}, field_updates, {}, "2026-03-31")
        assert field_updates["conv_price"] == 19.537
        assert rescaled["conv_price"]["echo"] is True

    def test_ratio_bearing_preferred_amend_left_raw(self):
        # conversion_ratio present -> conv_price stays in skip_price -> the
        # amend value is NOT rescaled (preserves the IQST-style exemption).
        terms = {"conv_price": 19.537, "conversion_ratio": 12.5,
                 "applied_splits": self._SPLIT}
        field_updates = {"conv_price": 0.5582}
        rescaled = _rescale_stale_unit_amend(
            "preferred", terms, {}, field_updates, {}, "2026-03-31")
        assert field_updates["conv_price"] == 0.5582  # untouched
        assert "conv_price" not in rescaled

    def test_stated_value_amend_never_rescaled(self):
        # stated_value is always skipped at both sites.
        terms = {"stated_value": 1000, "applied_splits": self._SPLIT}
        field_updates = {"stated_value": 28.571}
        rescaled = _rescale_stale_unit_amend(
            "preferred", terms, {}, field_updates, {}, "2026-03-31")
        assert field_updates["stated_value"] == 28.571
        assert "stated_value" not in rescaled

    def test_already_post_split_amend_left_as_is(self):
        # FCEL convention: the filer retro-adjusted the conv_price, so the
        # post-split 10-K re-quotes the POST-split 50760 == current -> implied
        # factor 1.0 (not the split ratio) -> no echo, no divide -> stays.
        terms = {"conv_price": 50760,
                 "applied_splits": [{"date": "2024-11-08", "ratio": 1 / 30,
                                     "direction": "reverse"}]}
        field_updates = {"conv_price": 50760}
        rescaled = _rescale_stale_unit_amend(
            "preferred", terms, {}, field_updates, {}, "2025-10-31")
        assert field_updates["conv_price"] == 50760
        assert "conv_price" not in rescaled

    def test_warrant_conv_price_still_rescaled(self):
        # Non-preferred regression guard: a warrant's pre-split-quoted price
        # is still echo-pinned (CETY $2.00 quoted vs current $30 over 1:15).
        terms = {"conv_price": 30.0,
                 "applied_splits": [{"date": "2025-10-06", "ratio": 1 / 15,
                                     "direction": "reverse"}]}
        field_updates = {"conv_price": 2.0}
        rescaled = _rescale_stale_unit_amend(
            "warrant", terms, {}, field_updates, {}, "2025-09-30")
        assert field_updates["conv_price"] == 30.0
        assert rescaled["conv_price"]["echo"] is True
