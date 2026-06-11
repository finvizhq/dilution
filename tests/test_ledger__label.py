"""Unit tests for dilution/ledger/_label.py.

The module is a deterministic, side-effect-free, post-LLM card-label
builder. There is NO I/O, NO DB, NO network — every test constructs a
duck-typed ``types.SimpleNamespace`` stub for the instrument ``m`` and
asserts the composed headline string.

Tested units:
  * build_label(m)      — top-level entry; date resolution + assembly
  * _pick_qualifier(m)  — per-type single-slot qualifier priority chain
  * _resolve_slot(m, s) — one named slot value

``extract_series_letter`` (imported by ``_resolve_slot``) is exercised
for real per the survey guidance — it is cheap and deterministic.
"""

from __future__ import annotations

import types
from datetime import date

import pytest

from dilution.ledger._label import (
    build_label,
    _pick_qualifier,
    _resolve_slot,
)


# ── stub factory ───────────────────────────────────────────────────
def make_m(**overrides):
    """Build an instrument stub with every attribute build_label /
    _resolve_slot touch defaulted to a benign value.

    Defaults: type=None, terms={}, descriptor=None,
    placement_agent_canonical=None, counterparty_canonical=None,
    event_date=None. Override only what a case cares about.
    """
    base = dict(
        type=None,
        terms={},
        descriptor=None,
        placement_agent_canonical=None,
        counterparty_canonical=None,
        event_date=None,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


# ════════════════════════════════════════════════════════════════════
#  _resolve_slot
# ════════════════════════════════════════════════════════════════════
class TestResolveSlotSeries:
    def test_series_full_phrase(self):
        m = make_m(terms={"series_letter": "Series D Preferred Stock"})
        assert _resolve_slot(m, "series") == "Series D"

    def test_series_bare_letter_uppercased(self):
        m = make_m(terms={"series_letter": "d"})
        assert _resolve_slot(m, "series") == "Series D"

    def test_series_numeric_bare(self):
        m = make_m(terms={"series_letter": "9"})
        assert _resolve_slot(m, "series") == "Series 9"

    def test_series_numeric_phrase(self):
        m = make_m(terms={"series_letter": "Series 9 Convertible Preferred"})
        assert _resolve_slot(m, "series") == "Series 9"

    def test_series_stuffed_closing_date_yields_none(self):
        # 'August 23' is not a series identifier — extract_series_letter
        # returns None, and the slot must NOT emit 'Series None'.
        m = make_m(terms={"series_letter": "August 23"})
        assert _resolve_slot(m, "series") is None

    def test_series_empty_string_short_circuits(self):
        # falsy series_letter never reaches extract_series_letter
        m = make_m(terms={"series_letter": ""})
        assert _resolve_slot(m, "series") is None

    def test_series_none_short_circuits(self):
        m = make_m(terms={"series_letter": None})
        assert _resolve_slot(m, "series") is None

    def test_series_absent_key(self):
        m = make_m(terms={})
        assert _resolve_slot(m, "series") is None

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Series A Warrants", "Series A"),
            ("a", "Series A"),
            ("B", "Series B"),
            ("Series 10 Preferred", "Series 10"),
            ("Class B Units", "Series B"),
        ],
    )
    def test_series_sweep(self, raw, expected):
        m = make_m(terms={"series_letter": raw})
        assert _resolve_slot(m, "series") == expected


class TestResolveSlotPreFunded:
    def test_pre_funded_true(self):
        m = make_m(terms={"is_pre_funded": True})
        assert _resolve_slot(m, "pre_funded") == "Pre-Funded"

    def test_pre_funded_false(self):
        m = make_m(terms={"is_pre_funded": False})
        assert _resolve_slot(m, "pre_funded") is None

    def test_pre_funded_absent(self):
        m = make_m(terms={})
        assert _resolve_slot(m, "pre_funded") is None

    def test_pre_funded_truthy_nonbool(self):
        m = make_m(terms={"is_pre_funded": 1})
        assert _resolve_slot(m, "pre_funded") == "Pre-Funded"


class TestResolveSlotDescriptor:
    def test_descriptor_verbatim(self):
        m = make_m(descriptor="Common")
        assert _resolve_slot(m, "descriptor") == "Common"

    def test_descriptor_none_passthrough(self):
        # descriptor is NOT guarded inside _resolve_slot — returns raw None
        m = make_m(descriptor=None)
        assert _resolve_slot(m, "descriptor") is None

    def test_descriptor_empty_string_passthrough(self):
        # '' is returned verbatim here; the caller's `if v:` check drops it
        m = make_m(descriptor="")
        assert _resolve_slot(m, "descriptor") == ""


class TestResolveSlotPlacementAgent:
    def test_agent_plain(self):
        m = make_m(placement_agent_canonical="Maxim")
        assert _resolve_slot(m, "placement_agent") == "Maxim"

    def test_agent_stripped(self):
        m = make_m(placement_agent_canonical=" Maxim ")
        assert _resolve_slot(m, "placement_agent") == "Maxim"

    def test_agent_empty_string(self):
        m = make_m(placement_agent_canonical="")
        assert _resolve_slot(m, "placement_agent") is None

    def test_agent_whitespace_only(self):
        m = make_m(placement_agent_canonical="   ")
        assert _resolve_slot(m, "placement_agent") is None

    def test_agent_none(self):
        m = make_m(placement_agent_canonical=None)
        assert _resolve_slot(m, "placement_agent") is None

    def test_agent_non_str_number(self):
        # isinstance(v, str) guard fails for a number -> None
        m = make_m(placement_agent_canonical=123)
        assert _resolve_slot(m, "placement_agent") is None


class TestResolveSlotCounterparty:
    def test_counterparty_plain(self):
        m = make_m(counterparty_canonical="Streeterville")
        assert _resolve_slot(m, "counterparty") == "Streeterville"

    def test_counterparty_stripped(self):
        m = make_m(counterparty_canonical=" Streeterville ")
        assert _resolve_slot(m, "counterparty") == "Streeterville"

    @pytest.mark.parametrize("blank", ["", "   ", "\t", None])
    def test_counterparty_blank_guards(self, blank):
        m = make_m(counterparty_canonical=blank)
        assert _resolve_slot(m, "counterparty") is None

    def test_counterparty_non_str(self):
        m = make_m(counterparty_canonical=42)
        assert _resolve_slot(m, "counterparty") is None


class TestResolveSlotMisc:
    def test_unknown_slot_name(self):
        assert _resolve_slot(make_m(), "no_such_slot") is None

    def test_terms_none_series(self):
        # m.terms None -> treated as {} so no AttributeError
        m = make_m(terms=None, descriptor=None)
        assert _resolve_slot(m, "series") is None

    def test_terms_none_pre_funded(self):
        m = make_m(terms=None)
        assert _resolve_slot(m, "pre_funded") is None


# ════════════════════════════════════════════════════════════════════
#  _pick_qualifier
# ════════════════════════════════════════════════════════════════════
class TestPickQualifierWarrant:
    def test_series_beats_pre_funded(self):
        m = make_m(type="warrant",
                   terms={"series_letter": "A", "is_pre_funded": True})
        assert _pick_qualifier(m) == "Series A"

    def test_pre_funded_when_no_series(self):
        m = make_m(type="warrant", terms={"is_pre_funded": True})
        assert _pick_qualifier(m) == "Pre-Funded"

    def test_pre_funded_beats_descriptor(self):
        m = make_m(type="warrant", terms={"is_pre_funded": True},
                   descriptor="Common")
        assert _pick_qualifier(m) == "Pre-Funded"

    def test_descriptor_when_no_series_or_prefunded(self):
        m = make_m(type="warrant", descriptor="Common")
        assert _pick_qualifier(m) == "Common"

    def test_descriptor_beats_agent(self):
        m = make_m(type="warrant", descriptor="Common",
                   placement_agent_canonical="Maxim")
        assert _pick_qualifier(m) == "Common"

    def test_agent_when_only_agent(self):
        m = make_m(type="warrant", placement_agent_canonical="Maxim")
        assert _pick_qualifier(m) == "Maxim"

    def test_agent_beats_counterparty(self):
        m = make_m(type="warrant", placement_agent_canonical="Maxim",
                   counterparty_canonical="Lender")
        assert _pick_qualifier(m) == "Maxim"

    def test_counterparty_last_resort(self):
        m = make_m(type="warrant", counterparty_canonical="Lender")
        assert _pick_qualifier(m) == "Lender"

    def test_all_empty_returns_none(self):
        m = make_m(type="warrant")
        assert _pick_qualifier(m) is None


class TestPickQualifierConvertible:
    def test_counterparty_first(self):
        m = make_m(type="convertible", counterparty_canonical="Streeterville",
                   descriptor="Note", placement_agent_canonical="Maxim")
        assert _pick_qualifier(m) == "Streeterville"

    def test_descriptor_when_no_counterparty(self):
        m = make_m(type="convertible", descriptor="Note",
                   placement_agent_canonical="Maxim")
        assert _pick_qualifier(m) == "Note"

    def test_agent_when_only_agent(self):
        m = make_m(type="convertible", placement_agent_canonical="Maxim")
        assert _pick_qualifier(m) == "Maxim"

    def test_series_ignored_for_convertible(self):
        # convertible order has no 'series' slot
        m = make_m(type="convertible", terms={"series_letter": "A"})
        assert _pick_qualifier(m) is None


class TestPickQualifierPreferred:
    def test_series_beats_counterparty(self):
        m = make_m(type="preferred", terms={"series_letter": "9"},
                   counterparty_canonical="PIPE Co")
        assert _pick_qualifier(m) == "Series 9"

    def test_counterparty_when_no_series(self):
        m = make_m(type="preferred", counterparty_canonical="PIPE Co")
        assert _pick_qualifier(m) == "PIPE Co"

    def test_counterparty_beats_agent_and_descriptor(self):
        m = make_m(type="preferred", counterparty_canonical="PIPE Co",
                   placement_agent_canonical="Maxim", descriptor="X")
        assert _pick_qualifier(m) == "PIPE Co"

    def test_descriptor_is_lowest(self):
        m = make_m(type="preferred", descriptor="X")
        assert _pick_qualifier(m) == "X"


class TestPickQualifierAtm:
    def test_agent_only(self):
        m = make_m(type="atm", placement_agent_canonical="TD Securities")
        assert _pick_qualifier(m) == "TD Securities"

    def test_counterparty_fallback(self):
        m = make_m(type="atm", counterparty_canonical="CP")
        assert _pick_qualifier(m) == "CP"

    def test_agent_beats_counterparty(self):
        m = make_m(type="atm", placement_agent_canonical="TD",
                   counterparty_canonical="CP")
        assert _pick_qualifier(m) == "TD"

    def test_descriptor_and_series_ignored(self):
        # ATM order is (placement_agent, counterparty) only
        m = make_m(type="atm", descriptor="X", terms={"series_letter": "A"})
        assert _pick_qualifier(m) is None


class TestPickQualifierEquityLine:
    def test_counterparty_first(self):
        m = make_m(type="equity_line", counterparty_canonical="M2B Funding",
                   placement_agent_canonical="Maxim", descriptor="X")
        assert _pick_qualifier(m) == "M2B Funding"

    def test_agent_when_no_counterparty(self):
        m = make_m(type="equity_line", placement_agent_canonical="Maxim",
                   descriptor="X")
        assert _pick_qualifier(m) == "Maxim"

    def test_descriptor_last(self):
        m = make_m(type="equity_line", descriptor="X")
        assert _pick_qualifier(m) == "X"


class TestPickQualifierOther:
    def test_shelf_always_none(self):
        m = make_m(type="shelf", placement_agent_canonical="Maxim",
                   counterparty_canonical="CP", descriptor="X",
                   terms={"series_letter": "A", "is_pre_funded": True})
        assert _pick_qualifier(m) is None

    def test_unknown_type_none(self):
        m = make_m(type="zzz", placement_agent_canonical="Maxim")
        assert _pick_qualifier(m) is None

    def test_none_type_none(self):
        m = make_m(type=None, placement_agent_canonical="Maxim")
        assert _pick_qualifier(m) is None

    def test_s1_offering_agent_then_counterparty(self):
        assert _pick_qualifier(
            make_m(type="s1_offering",
                   placement_agent_canonical="Maxim",
                   counterparty_canonical="CP")) == "Maxim"
        assert _pick_qualifier(
            make_m(type="s1_offering",
                   counterparty_canonical="CP")) == "CP"

    def test_equity_agent_then_counterparty(self):
        assert _pick_qualifier(
            make_m(type="equity",
                   placement_agent_canonical="A.G.P.",
                   counterparty_canonical="Hudson Bay")) == "A.G.P."
        assert _pick_qualifier(
            make_m(type="equity",
                   counterparty_canonical="Hudson Bay")) == "Hudson Bay"


# ════════════════════════════════════════════════════════════════════
#  build_label — date resolution
# ════════════════════════════════════════════════════════════════════
class TestBuildLabelDateResolution:
    def test_no_date_anywhere_returns_none(self):
        m = make_m(type="warrant", terms={}, event_date=None)
        assert build_label(m) is None

    def test_no_date_terms_none(self):
        m = make_m(type="warrant", terms=None, event_date=None)
        assert build_label(m) is None

    def test_issue_date_overrides_event_date(self):
        # FPI March->August relabel path: issue_date wins.
        m = make_m(type="warrant",
                   terms={"issue_date": "2024-08-15"},
                   event_date="2024-03-01",
                   descriptor="Common")
        assert build_label(m) == "August 2024 Common Warrants"

    def test_event_date_fallback_when_no_issue_date(self):
        m = make_m(type="warrant", terms={},
                   event_date="2024-03-01", descriptor="Common")
        assert build_label(m) == "March 2024 Common Warrants"

    def test_atm_uses_agreement_date_when_no_issue_date(self):
        m = make_m(type="atm",
                   terms={"agreement_date": "2025-11-01"},
                   event_date="2025-03-01",
                   placement_agent_canonical="TD Securities")
        assert build_label(m) == "November 2025 TD Securities ATM"

    def test_equity_line_uses_agreement_date(self):
        m = make_m(type="equity_line",
                   terms={"agreement_date": "2026-04-01"},
                   event_date="2026-01-01",
                   counterparty_canonical="M2B Funding")
        assert build_label(m) == "April 2026 M2B Funding ELOC"

    def test_atm_without_agreement_date_falls_to_event_date(self):
        m = make_m(type="atm", terms={},
                   event_date="2025-03-01",
                   placement_agent_canonical="TD Securities")
        assert build_label(m) == "March 2025 TD Securities ATM"

    def test_agreement_date_ignored_for_non_atm_type(self):
        # agreement_date branch is gated to atm/equity_line; a warrant
        # with agreement_date but no issue_date falls through to
        # event_date.
        m = make_m(type="warrant",
                   terms={"agreement_date": "2025-11-01"},
                   event_date="2025-03-01")
        assert build_label(m) == "March 2025 Warrants"

    def test_agreement_date_non_atm_no_event_date_is_none(self):
        # confirms agreement_date is truly ignored: with no event_date
        # there is no usable date at all.
        m = make_m(type="warrant",
                   terms={"agreement_date": "2025-11-01"},
                   event_date=None)
        assert build_label(m) is None

    def test_raw_date_object_used_directly(self):
        m = make_m(type="warrant", event_date=date(2025, 9, 1),
                   descriptor="Common")
        assert build_label(m) == "September 2025 Common Warrants"

    def test_issue_date_as_date_object(self):
        m = make_m(type="shelf", terms={"issue_date": date(2025, 8, 1)})
        assert build_label(m) == "August 2025 Shelf"

    def test_yyyy_mm_dd_string_parsed(self):
        m = make_m(type="shelf", terms={"issue_date": "2025-08-15"})
        assert build_label(m) == "August 2025 Shelf"

    def test_long_iso_string_sliced_to_10_chars(self):
        m = make_m(type="warrant", event_date="2025-09-01T12:34:56",
                   descriptor="Common")
        assert build_label(m) == "September 2025 Common Warrants"

    @pytest.mark.parametrize("bad", ["not-a-date", "2024-13-99", "",
                                     "2025/09/01", "Sept 2025"])
    def test_malformed_date_string_returns_none(self, bad):
        m = make_m(type="warrant", event_date=bad)
        assert build_label(m) is None

    def test_terms_missing_attr_shelf_no_attribute_error(self):
        # build_label guards `terms` via hasattr; shelf has an empty
        # qualifier order so _resolve_slot (which DOES read m.terms
        # unguarded) is never reached.
        class NoTerms:
            type = "shelf"
            descriptor = None
            placement_agent_canonical = None
            counterparty_canonical = None
            event_date = "2025-09-01"

        assert build_label(NoTerms()) == "September 2025 Shelf"

    @pytest.mark.parametrize(
        "month_str,expected_month",
        [
            ("2025-01-15", "January"),
            ("2025-02-15", "February"),
            ("2025-09-15", "September"),
            ("2025-12-15", "December"),
        ],
    )
    def test_full_english_month_names(self, month_str, expected_month):
        # %B uses full English names under the C/en CI locale.
        m = make_m(type="shelf", terms={"issue_date": month_str})
        assert build_label(m) == f"{expected_month} 2025 Shelf"


# ════════════════════════════════════════════════════════════════════
#  build_label — assembly per type
# ════════════════════════════════════════════════════════════════════
class TestBuildLabelAssembly:
    def test_warrant_with_qualifier(self):
        m = make_m(type="warrant", terms={"issue_date": "2025-09-01"},
                   descriptor="Common")
        assert build_label(m) == "September 2025 Common Warrants"

    def test_warrant_pre_funded(self):
        m = make_m(type="warrant", terms={"issue_date": "2022-10-01",
                                           "is_pre_funded": True})
        assert build_label(m) == "October 2022 Pre-Funded Warrants"

    def test_warrant_series(self):
        m = make_m(type="warrant", terms={"issue_date": "2024-11-01",
                                           "series_letter": "A"})
        assert build_label(m) == "November 2024 Series A Warrants"

    def test_warrant_no_qualifier(self):
        m = make_m(type="warrant", terms={"issue_date": "2025-08-01"})
        assert build_label(m) == "August 2025 Warrants"

    def test_convertible_counterparty(self):
        m = make_m(type="convertible", terms={"issue_date": "2022-12-01"},
                   counterparty_canonical="Streeterville")
        assert build_label(m) == "December 2022 Streeterville Convertible Note"

    def test_convertible_no_qualifier(self):
        m = make_m(type="convertible", terms={"issue_date": "2022-12-01"})
        assert build_label(m) == "December 2022 Convertible Note"

    def test_preferred_series(self):
        m = make_m(type="preferred", terms={"issue_date": "2024-03-01",
                                             "series_letter": "9"})
        assert build_label(m) == "March 2024 Series 9 Preferred"

    def test_atm_no_agent(self):
        m = make_m(type="atm", terms={"issue_date": "2025-08-01"})
        assert build_label(m) == "August 2025 ATM"

    def test_equity_line_no_qualifier(self):
        m = make_m(type="equity_line", terms={"issue_date": "2026-04-01"})
        assert build_label(m) == "April 2026 ELOC"

    def test_shelf_never_takes_qualifier(self):
        # Even with agent/descriptor/series set, shelf stays bare.
        m = make_m(type="shelf", terms={"issue_date": "2025-08-01",
                                        "series_letter": "A"},
                   descriptor="X", placement_agent_canonical="Maxim")
        assert build_label(m) == "August 2025 Shelf"

    def test_s1_offering_no_qualifier(self):
        m = make_m(type="s1_offering", terms={"issue_date": "2025-08-01"})
        assert build_label(m) == "August 2025 S-1 Offering"

    def test_s1_offering_with_agent(self):
        m = make_m(type="s1_offering", terms={"issue_date": "2025-08-01"},
                   placement_agent_canonical="Maxim")
        assert build_label(m) == "August 2025 Maxim S-1 Offering"

    def test_unknown_type_bare_month_year(self):
        # type not in _TYPE_TAIL / _QUALIFIER_ORDER -> no tail, no
        # qualifier -> just the month/year. Descriptor is NOT appended
        # because the type is not 'equity'.
        m = make_m(type="mystery", terms={"issue_date": "2025-09-01"},
                   descriptor="X", placement_agent_canonical="Maxim")
        assert build_label(m) == "September 2025"


# ════════════════════════════════════════════════════════════════════
#  build_label — the 'equity' special branch
# ════════════════════════════════════════════════════════════════════
class TestBuildLabelEquityBranch:
    def test_bank_plus_descriptor(self):
        m = make_m(type="equity", terms={"issue_date": "2026-04-01"},
                   descriptor="Private Placement",
                   placement_agent_canonical="A.G.P.")
        assert build_label(m) == "April 2026 A.G.P. Private Placement"

    def test_descriptor_only(self):
        m = make_m(type="equity", terms={"issue_date": "2026-04-01"},
                   descriptor="Private Placement")
        assert build_label(m) == "April 2026 Private Placement"

    def test_entity_only(self):
        m = make_m(type="equity", terms={"issue_date": "2026-04-01"},
                   counterparty_canonical="Hudson Bay")
        assert build_label(m) == "April 2026 Hudson Bay Equity Issuance"

    def test_neither_defaults_to_equity_issuance(self):
        m = make_m(type="equity", terms={"issue_date": "2026-04-01"})
        assert build_label(m) == "April 2026 Equity Issuance"

    def test_empty_descriptor_falls_back_to_default(self):
        # m.descriptor or "Equity Issuance": '' is falsy -> default tail.
        m = make_m(type="equity", terms={"issue_date": "2026-04-01"},
                   descriptor="")
        assert build_label(m) == "April 2026 Equity Issuance"

    def test_agent_plus_default_when_descriptor_empty(self):
        m = make_m(type="equity", terms={"issue_date": "2026-04-01"},
                   descriptor="", placement_agent_canonical="A.G.P.")
        assert build_label(m) == "April 2026 A.G.P. Equity Issuance"

    def test_agent_preferred_over_counterparty_in_equity(self):
        # equity qualifier order is (placement_agent, counterparty)
        m = make_m(type="equity", terms={"issue_date": "2026-04-01"},
                   placement_agent_canonical="A.G.P.",
                   counterparty_canonical="Hudson Bay",
                   descriptor="Private Placement")
        assert build_label(m) == "April 2026 A.G.P. Private Placement"


# ════════════════════════════════════════════════════════════════════
#  FIXED (was bug A#5): _resolve_slot now reads `m.terms` via getattr and
#  coerces a missing/non-dict terms to {}, and build_label reads `m.type`
#  via getattr. An instrument lacking a `terms` (or `type`) attribute now
#  builds its label gracefully instead of raising AttributeError, for every
#  type — not just the empty-qualifier-order ones (e.g. shelf).
# ════════════════════════════════════════════════════════════════════
class TestMissingTermsAttrBug:
    def test_atm_missing_terms_attr_builds_label(self):
        class NoTerms:
            type = "atm"
            descriptor = None
            placement_agent_canonical = "TD"
            counterparty_canonical = None
            event_date = "2025-09-01"

        # FIXED: _resolve_slot reads terms via getattr, so a missing terms
        # attr defaults to {} and the atm label builds from placement_agent.
        assert build_label(NoTerms()) == "September 2025 TD ATM"

    def test_warrant_missing_terms_attr_builds_label(self):
        class NoTerms:
            type = "warrant"
            descriptor = "Common"
            placement_agent_canonical = None
            counterparty_canonical = None
            event_date = "2025-09-01"

        assert build_label(NoTerms()) == "September 2025 Common Warrants"


# ════════════════════════════════════════════════════════════════════
#  Reviewer-added coverage: gaps from the survey edge-case slice that
#  the original suite did not exercise. All assertions were derived from
#  the source logic and confirmed by observing the real functions.
# ════════════════════════════════════════════════════════════════════
class TestResolveSlotTermsNonDict:
    # FIXED (was bug A#5): _resolve_slot now coerces a non-dict terms
    # (list/str) to {} before calling .get, so a non-Mapping terms resolves
    # to None instead of exploding. Latent (terms is always a dict in prod).
    def test_terms_is_list_resolves_series_to_none(self):
        m = make_m(terms=["not", "a", "dict"])
        assert _resolve_slot(m, "series") is None

    def test_terms_is_string_resolves_pre_funded_to_none(self):
        m = make_m(terms="oops")
        assert _resolve_slot(m, "pre_funded") is None

    def test_terms_is_list_does_not_affect_descriptor_slot(self):
        # The `.get` blowup is SLOT-SPECIFIC, not at the top: line 92's
        # `terms = m.terms or {}` only ASSIGNS — it never calls `.get`.
        # Only the series/pre_funded branches dereference terms, so the
        # descriptor branch (which just returns m.descriptor) is immune to
        # a non-Mapping terms and returns the descriptor verbatim.
        m = make_m(terms=["x"], descriptor="Common")
        assert _resolve_slot(m, "descriptor") == "Common"


class TestPickQualifierEmptyStringSkips:
    # _pick_qualifier walks the chain with `if v:` — a slot that resolves
    # to a falsy value (e.g. '' from a blank descriptor) is skipped and
    # the chain continues to the next slot. Proves the falsy-skip wiring,
    # not just the truthy-stop wiring the original suite covered.
    def test_warrant_empty_descriptor_skips_to_agent(self):
        m = make_m(type="warrant", descriptor="",
                   placement_agent_canonical="Maxim")
        assert _pick_qualifier(m) == "Maxim"

    def test_warrant_empty_descriptor_and_blank_agent_skips_to_counterparty(self):
        m = make_m(type="warrant", descriptor="",
                   placement_agent_canonical="   ",
                   counterparty_canonical="Lender")
        assert _pick_qualifier(m) == "Lender"

    def test_equity_line_blank_counterparty_skips_to_agent(self):
        m = make_m(type="equity_line", counterparty_canonical="   ",
                   placement_agent_canonical="Maxim")
        assert _pick_qualifier(m) == "Maxim"

    def test_preferred_none_series_skips_to_counterparty(self):
        # series_letter='August 23' -> extract_series_letter None -> slot
        # None -> chain falls through to the counterparty slot.
        m = make_m(type="preferred",
                   terms={"series_letter": "August 23"},
                   counterparty_canonical="PIPE Co")
        assert _pick_qualifier(m) == "PIPE Co"


class TestBuildLabelTermsNoneOnAgreementPath:
    # build_label guards terms with hasattr, so terms=None coerces to {}
    # and the agreement_date lookup safely misses, falling to event_date.
    # The atm/equity_line agreement branch must still work when terms=None.
    def test_atm_terms_none_falls_to_event_date(self):
        m = make_m(type="atm", terms=None, event_date="2025-03-01",
                   placement_agent_canonical="TD")
        assert build_label(m) == "March 2025 TD ATM"

    def test_equity_line_terms_none_falls_to_event_date(self):
        m = make_m(type="equity_line", terms=None, event_date="2026-01-01",
                   counterparty_canonical="M2B Funding")
        assert build_label(m) == "January 2026 M2B Funding ELOC"


class TestBuildLabelDateBranchEdges:
    def test_datetime_object_used_directly(self):
        # datetime.datetime subclasses datetime.date, so isinstance(raw,
        # date) is True and the strptime branch is skipped; the time
        # component is dropped by %B %Y.
        import datetime as _dt
        m = make_m(type="shelf",
                   terms={"issue_date": _dt.datetime(2025, 8, 1, 12, 30)})
        assert build_label(m) == "August 2025 Shelf"

    def test_whitespace_only_issue_date_short_circuits_no_fallback(self):
        # A truthy whitespace-only issue_date string is NOT empty, so the
        # `if not raw` fallback to event_date never fires; strptime then
        # fails on the blank -> None, even though event_date is valid.
        m = make_m(type="shelf", terms={"issue_date": "   "},
                   event_date="2025-01-01")
        assert build_label(m) is None

    def test_series_letter_whitespace_padded_stripped_in_label(self):
        # End-to-end: extract_series_letter strips the padding so the
        # composed label carries the clean 'Series A'.
        m = make_m(type="warrant",
                   terms={"issue_date": "2024-11-01",
                          "series_letter": "  A  "})
        assert build_label(m) == "November 2024 Series A Warrants"

    def test_preferred_no_qualifier_bare_tail(self):
        # No series, no counterparty/agent/descriptor -> bare type tail.
        m = make_m(type="preferred", terms={"issue_date": "2024-03-01"})
        assert build_label(m) == "March 2024 Preferred"


class TestBuildLabelEquityWhitespaceDescriptorQuirk:
    # A whitespace-only descriptor '   ' is TRUTHY, so `m.descriptor or
    # "Equity Issuance"` keeps the whitespace instead of defaulting, and
    # it also fills the qualifier slot is irrelevant (descriptor is not in
    # the equity qualifier order). Result: trailing/embedded whitespace,
    # NOT the 'Equity Issuance' default. Latent quirk; LLM emits clean
    # descriptors in practice.
    def test_whitespace_descriptor_is_not_defaulted(self):
        m = make_m(type="equity", terms={"issue_date": "2026-04-01"},
                   descriptor="   ")
        # BUG: truthy whitespace descriptor bypasses the default tail.
        assert build_label(m) == "April 2026    "

    def test_empty_string_descriptor_with_counterparty_qualifier(self):
        # '' descriptor -> default tail 'Equity Issuance'; counterparty is
        # the equity qualifier, prepended ahead of the tail.
        m = make_m(type="equity", terms={"issue_date": "2026-04-01"},
                   descriptor="", counterparty_canonical="Hudson Bay")
        assert build_label(m) == "April 2026 Hudson Bay Equity Issuance"


# ════════════════════════════════════════════════════════════════════
#  Reviewer-added (round 2): further gaps from the survey edge slice
#  not yet exercised. Every expected value was OBSERVED against the live
#  functions before being asserted (see the adversarial-review trace),
#  never blind-guessed.
# ════════════════════════════════════════════════════════════════════
class TestResolveSlotSeriesBoundaries:
    def test_multi_letter_series_kept(self):
        # The bare branch accepts 1..3 alpha chars verbatim, so a real
        # two-letter series ('AB') survives -> 'Series AB'. Boundary of the
        # `1 <= len(bare) <= 3` gate in extract_series_letter.
        m = make_m(terms={"series_letter": "AB"})
        assert _resolve_slot(m, "series") == "Series AB"

    def test_four_char_alpha_not_bare_falls_to_regex_none(self):
        # len > 3 skips the bare branch; 'ABCD' has no 'Series'/'Class'
        # marker so the regexes miss -> None (not 'Series ABCD').
        m = make_m(terms={"series_letter": "ABCD"})
        assert _resolve_slot(m, "series") is None

    def test_integer_series_letter_coerced_to_str(self):
        # series_letter is truthy non-str int 9; extract_series_letter does
        # `str(s)` then the 1..3 digit bare branch -> 'Series 9'. Proves the
        # raw truthiness gate (`if raw`) admits a non-string and the helper
        # coerces it rather than raising.
        m = make_m(terms={"series_letter": 9})
        assert _resolve_slot(m, "series") == "Series 9"

    def test_zero_series_letter_is_falsy_short_circuit(self):
        # int 0 is falsy -> `if raw else None` short-circuits BEFORE
        # extract_series_letter, so the slot is None (not 'Series 0').
        m = make_m(terms={"series_letter": 0})
        assert _resolve_slot(m, "series") is None


class TestPickQualifierChainExhaustion:
    # Survey edge ('all candidate slots empty/None -> None') for EVERY
    # type with a non-empty order, proving the full chain falls through
    # to None when nothing resolves (the original suite only covered the
    # warrant exhaustion case).
    @pytest.mark.parametrize(
        "typ",
        ["warrant", "convertible", "preferred", "atm",
         "equity_line", "s1_offering", "equity"],
    )
    def test_empty_everything_returns_none(self, typ):
        m = make_m(type=typ)  # all qualifier sources None / {}
        assert _pick_qualifier(m) is None


class TestBuildLabelNonStringDateFailurePaths:
    def test_integer_event_date_not_a_date_returns_none(self):
        # An int event_date is truthy, so it reaches the parse branch;
        # str(20250901)[:10] = '20250901' fails strptime('%Y-%m-%d')
        # -> ValueError -> None. It is NOT silently coerced to a date.
        m = make_m(type="warrant", event_date=20250901)
        assert build_label(m) is None

    def test_zero_event_date_is_falsy_returns_none(self):
        # int 0 is falsy: `if not raw` fires and there is no other source,
        # so build_label returns None without ever reaching strptime.
        m = make_m(type="warrant", event_date=0)
        assert build_label(m) is None

    def test_float_event_date_returns_none(self):
        # A float likewise has no '%Y-%m-%d' shape after str()[:10].
        m = make_m(type="warrant", event_date=2025.09)
        assert build_label(m) is None


class TestBuildLabelMissingTypeAttr:
    # FIXED (was bug A#5): _pick_qualifier and the _TYPE_TAIL lookup now read
    # m.type via getattr too, so an object with a usable date but NO 'type'
    # attribute degrades to a bare "<Month YYYY>" label (no qualifier, no
    # type-tail) instead of raising AttributeError. Latent (real instruments
    # always carry a type).
    def test_missing_type_attr_builds_bare_month_year(self):
        class NoType:
            terms = {}
            descriptor = None
            placement_agent_canonical = None
            counterparty_canonical = None
            event_date = "2025-09-01"

        assert build_label(NoType()) == "September 2025"
