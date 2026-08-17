"""Unit tests for the anchor's overhang→ledger assignment.

The reconciler pairs each overhang row from a periodic filing with a ledger
row, and synthesizes a create for anything left unpaired. That synthesis is
where duplicate instruments come from, so HOW the pairing is decided is the
whole ballgame.

It used to be greedy first-fit in overhang-row order: whoever came first
claimed its best remaining candidate. A weak early pairing could take a row
that a later line matched far better, leaving the later line with nothing —
reported `missing_in_ledger` and synthesized into a duplicate. Measured on
CELU's Aug-2025 10-Q: 19 overhang warrants against ~20 ledger rows produced
4 phantom creates, one of which ("March 2024 RWI Forbearance Warrants")
shared both issue date and counterparty with an existing March-2024 RWI row.

`_assign_matches` decides globally instead — all candidate pairs sorted by
score, strongest binding first.

No DB, no network: rows are shaped exactly as the reconciler receives them
(terms/outstanding as JSON strings, strike under `terms.strike`).
"""

from __future__ import annotations

import json

from dilution.ledger.anchor import _assign_matches, _best_match, _match_scores


def row(iid, *, created, strike=None, cp=None, label=None, expiration=None,
        count=None):
    return {
        "instrument_id": iid, "type": "warrant", "status": "active",
        "created_at": created, "counterparty_canonical": cp, "label": label,
        "terms_json": json.dumps(
            {k: v for k, v in (("strike", strike),
                               ("expiration", expiration)) if v is not None}),
        "outstanding_json": json.dumps(
            {"count": count} if count is not None else {}),
    }


def over(name, *, issue=None, strike=None, expiry=None, count=None):
    o = {"category": "warrant", "instrument_name": name}
    if issue is not None:
        o["issue_date"] = issue
    if strike is not None:
        o["strike_or_conversion_price"] = strike
    if expiry is not None:
        o["maturity_or_expiry"] = expiry
    if count is not None:
        o["count"] = count
    return o


def first_fit(overs, rows):
    """The old behaviour, reproduced for contrast."""
    used, out = set(), {}
    for i, o in enumerate(overs):
        pool = [r for r in rows if r["instrument_id"] not in used]
        m = _best_match(o, pool)
        if m is not None:
            out[i] = m
            used.add(m["instrument_id"])
    return out


class TestScoring:
    def test_strike_is_read_from_terms_strike(self):
        # Guard on the row contract: the accessor reads terms.strike, so a
        # test (or caller) using `strike_price` silently loses the axis.
        r = row("W-1", created="2024-03-15", strike=5.9)
        assert _match_scores(over("X", strike=5.9), [r]) == {"W-1": 2}

    def test_issue_date_outweighs_strike(self):
        r = row("W-1", created="2024-03-15", strike=5.9)
        date_only = _match_scores(over("X", issue="2024-03-13"), [r])["W-1"]
        strike_only = _match_scores(over("X", strike=5.9), [r])["W-1"]
        assert date_only > strike_only

    def test_zero_score_candidates_are_omitted(self):
        r = row("W-1", created="2020-01-01", strike=1.0)
        assert _match_scores(over("Unrelated", issue="2026-01-01"), [r]) == {}


class TestGlobalAssignment:
    def test_strong_pairing_wins_the_contested_row(self):
        """THE regression. Both lines want W-A; the weaker one is first."""
        rows = [row("W-A", created="2024-03-15", strike=5.9,
                    expiration="2028-06-20", label="March 2024 Warrants")]
        weak = over("Common Warrants", strike=5.9)              # strike only
        strong = over("March 2024 Warrants", issue="2024-03-13",
                      strike=5.9, expiry="2028-06-20")          # date+strike+expiry
        overs = [weak, strong]

        # Old behaviour: the weak line, being first, takes W-A and the
        # strong line is stranded → duplicate create.
        old = first_fit(overs, rows)
        assert old.get(0) is not None and old.get(1) is None

        # New behaviour: the strong line binds; the weak line is the one
        # left over, which is the better candidate for a genuine create.
        new = _assign_matches(overs, rows)
        assert new[1]["instrument_id"] == "W-A"
        assert 0 not in new

    def test_no_double_claiming(self):
        rows = [row("W-A", created="2024-03-15", strike=5.9)]
        overs = [over("A", issue="2024-03-13", strike=5.9),
                 over("B", issue="2024-03-14", strike=5.9)]
        assigned = _assign_matches(overs, rows)
        assert len(assigned) == 1
        assert len({r["instrument_id"] for r in assigned.values()}) == 1

    def test_each_row_matched_when_pairing_is_unambiguous(self):
        rows = [row("W-A", created="2024-03-15", strike=5.9),
                row("W-B", created="2023-01-10", strike=1.25)]
        overs = [over("later", issue="2024-03-13", strike=5.9),
                 over("earlier", issue="2023-01-11", strike=1.25)]
        assigned = _assign_matches(overs, rows)
        assert assigned[0]["instrument_id"] == "W-A"
        assert assigned[1]["instrument_id"] == "W-B"

    def test_unmatchable_row_yields_no_assignment(self):
        rows = [row("W-A", created="2020-01-01", strike=1.0)]
        assigned = _assign_matches([over("New", issue="2026-05-05",
                                         strike=9.99)], rows)
        assert assigned == {}

    def test_empty_inputs(self):
        assert _assign_matches([], []) == {}
        assert _assign_matches([over("X", strike=1.0)], []) == {}
        assert _assign_matches([], [row("W-A", created="2024-01-01")]) == {}

    def test_result_is_order_independent(self):
        # A total sort order means the outcome cannot depend on how the
        # overhang happened to be ordered, which is what made the old
        # behaviour fragile in the first place.
        rows = [row("W-A", created="2024-03-15", strike=5.9,
                    expiration="2028-06-20"),
                row("W-B", created="2024-01-16", strike=2.99,
                    expiration="2029-07-15")]
        a = over("March", issue="2024-03-13", strike=5.9, expiry="2028-06-20")
        b = over("January", issue="2024-01-16", strike=2.99,
                 expiry="2029-07-15")
        fwd = _assign_matches([a, b], rows)
        rev = _assign_matches([b, a], rows)
        assert fwd[0]["instrument_id"] == rev[1]["instrument_id"] == "W-A"
        assert fwd[1]["instrument_id"] == rev[0]["instrument_id"] == "W-B"

    def test_exclusions_still_apply(self):
        # Pre-funded vs ordinary is a tranche identity, not field drift —
        # the global pass must not bind across it just to raise a score.
        rows = [row("W-A", created="2025-02-13", strike=2.0,
                    label="February 2025 Common Warrants")]
        pf = over("February 2025 Pre-Funded Warrants", issue="2025-02-13",
                  strike=0.001)
        assert _assign_matches([pf], rows) == {}


class TestRepriceAndTrancheMarkers:
    """Overhang lines that RE-STATE an existing tranche must not mint a new
    row. Two name-marked classes, both of which legitimately carry a
    different strike than the stored row — so the shadow-match cannot
    require strike agreement for them.
    """

    def test_reprice_markers_detected(self):
        from dilution.ledger.anchor import _reprice_marker
        for name in ("March 2023 PIPE Warrants (modified)",
                     "April 2023 Registered Direct Warrants (modified)",
                     "Series A Warrants as amended",
                     "Warrants (as modified)", "repriced 2024 warrants"):
            assert _reprice_marker(name), name

    def test_genuinely_new_issuances_are_not_reprice_markers(self):
        # These words all describe a SEPARATE instrument. Treating them as
        # re-statements would silently swallow real dilution.
        from dilution.ledger.anchor import _reprice_marker
        for name in ("December 2023 Inducement New Warrants",
                     "November 2024 Placement Agent Warrants",
                     "January 2024 Pre-Funded Warrants",
                     "Replacement Warrants", "New Warrants"):
            assert not _reprice_marker(name), name

    def test_tranche_ordinals_detected(self):
        from dilution.ledger.anchor import _tranche_ordinal
        assert _tranche_ordinal("Bridge Loan - Tranche #2 Warrants") == 2
        assert _tranche_ordinal("Advisory Warrants — Tranche 2") == 2
        assert _tranche_ordinal("Tranche No. 3 Warrants") == 3
        assert _tranche_ordinal("Series A Warrants") is None
        assert _tranche_ordinal(None) is None

    def test_reprice_line_binds_despite_a_changed_strike(self):
        from dilution.ledger.anchor import _subsumed_by_used_tranche
        # Stored row still carries the PRE-amendment strike.
        used = [row("W-A", created="2023-03-24", strike=2.5,
                    label="March 2023 PIPE Warrants")]
        over_mod = over("March 2023 PIPE Warrants (modified)",
                        issue="2023-03-27", strike=1.0)
        assert _subsumed_by_used_tranche(over_mod, used) is not None

    def test_tranche_line_binds_despite_a_changed_strike(self):
        from dilution.ledger.anchor import _subsumed_by_used_tranche
        used = [row("W-A", created="2024-01-16", strike=2.49,
                    label="January 2024 Bridge Loan Warrants")]
        over_t2 = over("January 2024 Bridge Loan - Tranche #2 Warrants",
                       issue="2024-01-16", strike=3.076)
        assert _subsumed_by_used_tranche(over_t2, used) is not None

    def test_relaxed_matching_still_requires_the_issue_date_to_agree(self):
        # A later, unrelated financing that happens to say "tranche 2"
        # must NOT be swallowed by an old row.
        from dilution.ledger.anchor import _subsumed_by_used_tranche
        used = [row("W-A", created="2021-01-05", strike=2.49)]
        far = over("Tranche 2 Warrants", issue="2025-08-30", strike=3.0)
        assert _subsumed_by_used_tranche(far, used) is None

    def test_unmarked_line_still_needs_strike_agreement(self):
        # The original conservative contract is unchanged for names that
        # carry neither marker.
        from dilution.ledger.anchor import _subsumed_by_used_tranche
        used = [row("W-A", created="2024-01-16", strike=2.49)]
        plain = over("January 2024 Warrants", issue="2024-01-16", strike=9.99)
        assert _subsumed_by_used_tranche(plain, used) is None
        same = over("January 2024 Warrants", issue="2024-01-16", strike=2.49)
        assert _subsumed_by_used_tranche(same, used) is not None

    def test_maturity_contradiction_still_vetoes(self):
        from dilution.ledger.anchor import _subsumed_by_used_tranche
        used = [row("W-A", created="2024-01-16", strike=2.49)]
        used[0]["terms_json"] = json.dumps(
            {"strike": 2.49, "maturity": "2029-01-16"})
        clash = over("January 2024 Warrants (modified)", issue="2024-01-16",
                     strike=3.0, expiry="2031-12-31")
        assert _subsumed_by_used_tranche(clash, used) is None
