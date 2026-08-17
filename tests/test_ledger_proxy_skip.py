"""Unit tests for the proxy-statement pre-screen in walker.py.

Proxies reach the walker with only amend_warrant + apply_split +
note_no_event available, and neither lever is reachable ONLY from a proxy:
every split in the DB came from market data (fetch_and_persist_splits) and
across the corpus no ledger row was ever created by a proxy. These tests
pin the skip set and the reasons it can be trusted.
"""

from __future__ import annotations

from dilution.ledger.tools import TOOLS_FOR_FORM, tools_for_form
from dilution.ledger.walker import _PROXY_SKIP_FORMS, _SKIPPED_FORMS


class TestProxySkipSet:
    def test_covers_the_measured_barren_proxy_forms(self):
        # 205 walked, 93-100% barren across the corpus.
        for form in ("DEF 14A", "DEFA14A", "PRE 14A"):
            assert form in _PROXY_SKIP_FORMS, form

    def test_merger_proxies_stay_in_the_llm_path(self):
        # Merger-consideration share issuance is real dilution and broader
        # extraction there is open scope — skipping DEFM14A/PREM14A would
        # foreclose it silently.
        for form in ("DEFM14A", "PREM14A"):
            assert form not in _PROXY_SKIP_FORMS, form

    def test_disjoint_from_the_no_body_skip_set(self):
        # _SKIPPED_FORMS (EFFECT/RW) is skipped for a different reason —
        # no body to process — and is applied at a different stage.
        assert not (_PROXY_SKIP_FORMS & _SKIPPED_FORMS)

    def test_no_periodic_or_current_report_is_ever_proxy_skipped(self):
        for form in ("10-K", "10-Q", "20-F", "8-K", "6-K", "S-1", "S-3",
                     "424B5", "POS AM"):
            assert form not in _PROXY_SKIP_FORMS, form

    def test_skipped_forms_only_ever_had_the_two_inert_levers(self):
        # The justification for skipping: nothing else was on the table.
        # If a future tool is added to a proxy's set, this fails and the
        # skip has to be re-argued.
        allowed = {"amend_warrant", "apply_split", "note_no_event"}
        for form in _PROXY_SKIP_FORMS:
            if form not in TOOLS_FOR_FORM:
                continue          # unknown forms already short-circuit
            names = {t.name for t in tools_for_form(form)}
            assert names <= allowed, (form, names - allowed)
