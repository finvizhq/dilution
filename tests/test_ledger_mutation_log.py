"""Unit tests for the applied-mutation log (dilution_mutations) and the
lossless codec behind it.

Why this log matters: LLM extraction is the only step in the pipeline that
costs money AND is nondeterministic, so the mutations it produces are
source data while the ledger is a fold of them. The log makes that fold
replayable (scripts/rebuild_ledger.py). An INCOMPLETE log is worse than no
log — it would replay most of history and look plausible — so these tests
concentrate on completeness and exactness:

  * every mutation class round-trips through the codec bit-for-bit
  * accepted mutations are logged, rejected ones are not
  * the several apply_mutations calls that share ONE accession each get
    their own rows (the bug an (accession, seq) key would have hidden)
  * a --force reset clears the log rather than mixing extraction runs
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from dilution.ledger import mutations as mut
from dilution.ledger.mutations import (
    AmendAtm,
    AmendConvertible,
    AmendEquity,
    AmendEquityLine,
    AmendPreferred,
    AmendS1Offering,
    AmendShelf,
    AmendWarrant,
    ApplySplit,
    CloseInstrument,
    ConfirmClosing,
    CreateAtm,
    CreateConvertible,
    CreateEquity,
    CreateEquityLine,
    CreatePreferred,
    CreateS1Offering,
    CreateShelf,
    CreateWarrant,
    NoteNoEvent,
    RecordConversion,
    RecordDrawdown,
    RecordExercise,
    RecordPartialRedemption,
    RecordPartialTermination,
    RestateAtm,
    mutation_from_record,
    mutation_to_record,
)
from dilution.ledger.store import apply_mutations, reset_walk_state

CIK = 1752828
ED = date(2024, 1, 15)


def _log(temp_db, cik: int = CIK) -> list[dict]:
    return [dict(r) for r in temp_db.execute(
        "SELECT * FROM dilution_mutations WHERE cik=? ORDER BY id", (cik,))]


# ── the codec ────────────────────────────────────────────────────────


# One instance per mutation class, so the registry can be asserted
# exhaustive below. Values are deliberately non-default (dates, tuples,
# floats, None) — a codec that only handles the empty case is useless.
_CASES = [
    CreateAtm(capacity_usd=1e7, event_date=ED, agreement_date=date(2024, 2, 1),
              placement_agent_canonical="HCW", proposed_id="ATM-1",
              label="Feb 2024 ATM"),
    CreateShelf(capacity_usd=5e7, event_date=ED, proposed_id="SH-1"),
    CreateWarrant(count=10_000, strike=1.25, event_date=ED,
                  proposed_id="W-1", label="Feb Warrants",
                  counterparty_canonical="Hudson Bay"),
    CreateConvertible(principal=3e6, principal_remaining=3e6, event_date=ED,
                      conv_price=0.5, proposed_id="C-1"),
    CreatePreferred(count=1000, series_letter="B", event_date=ED,
                    conv_price=11.90, proposed_id="P-1"),
    CreateEquityLine(capacity_usd=2.5e7, event_date=ED,
                     counterparty_canonical="Yorkville", proposed_id="EL-1"),
    CreateS1Offering(anticipated_deal_size=1.5e7, event_date=ED,
                     proposed_id="S1-1"),
    CreateEquity(count=500_000, price_per_share=1.10, event_date=ED),
    RestateAtm(predecessor_id="ATM-1", capacity_usd=2e7, event_date=ED,
               supersede_prior=True, agreement_date=date(2024, 3, 1),
               placement_agent_canonical="Maxim"),
    AmendAtm(instrument_id="ATM-1", event_date=ED, capacity_usd=1.5e7),
    AmendWarrant(instrument_id="W-1", event_date=ED, strike=0.9, count=9_000),
    AmendShelf(instrument_id="SH-1", event_date=ED,
               remaining_capacity_usd=4e7),
    AmendConvertible(instrument_id="C-1", event_date=ED, conv_price=0.35,
                     convertible_date=date(2024, 6, 1)),
    AmendPreferred(instrument_id="P-1", event_date=ED, conv_price=11.90,
                   conversion_ratio=100.0),
    AmendEquityLine(instrument_id="EL-1", event_date=ED, drawn_usd=1.2e6),
    AmendS1Offering(instrument_id="S1-1", event_date=ED, sold_to_date=3e6),
    AmendEquity(instrument_id="EQ-1", event_date=ED,
                known_owners=("Armistice",)),
    RecordExercise(instrument_id="W-1", shares=500, event_date=ED, price=1.25,
                   warrants_exercised=600),
    RecordConversion(instrument_id="C-1", shares_issued=1000, event_date=ED,
                     principal_converted=250_000.0),
    # Both RecordDrawdown flavors: `fields` DERIVES drawdown_amount_usd from
    # shares × price and never emits price_per_share, which is exactly why
    # the log cannot use mutation_to_dict.
    RecordDrawdown(instrument_id="ATM-1", drawdown_shares=1000, event_date=ED,
                   price_per_share=2.50, placement_agent_canonical="HCW"),
    RecordDrawdown(instrument_id="ATM-1", drawdown_shares=1000, event_date=ED,
                   drawdown_amount_usd=5000.0),
    RecordPartialRedemption(instrument_id="C-1", event_date=ED,
                            principal_redeemed=100_000.0),
    RecordPartialTermination(instrument_id="ATM-1", event_date=ED,
                             capacity_reduced_usd=1e6),
    ConfirmClosing(instrument_id="ATM-1", event_date=ED),
    CloseInstrument(instrument_id="W-1", reason="expired", event_date=ED),
    CloseInstrument(instrument_id="ATM-1", reason="superseded",
                    replaced_by="ATM-2", event_date=ED),
    ApplySplit(pre=100, post=1, direction="reverse", units="common",
               effective_date=ED),
    NoteNoEvent(reason="no dilutive event"),
]


class TestCodec:
    @pytest.mark.parametrize("m", _CASES, ids=lambda m: type(m).__name__)
    def test_round_trip_is_exact(self, m):
        """Through real JSON, not just the dict — the log stores text."""
        restored = mutation_from_record(
            json.loads(json.dumps(mutation_to_record(m))))
        assert restored == m
        assert type(restored) is type(m)

    def test_dates_survive_as_dates(self):
        """A date must not come back as a string: apply_mutations does
        date arithmetic on event_date."""
        restored = mutation_from_record(
            json.loads(json.dumps(mutation_to_record(_CASES[0]))))
        assert isinstance(restored.event_date, date)
        assert restored.agreement_date == date(2024, 2, 1)

    def test_every_registered_class_is_covered(self):
        """Guards the parametrized list above from drifting behind a newly
        added mutation type — which would leave that type silently
        unreplayable."""
        covered = {type(m).__name__ for m in _CASES}
        registered = {c.__name__ for c in mut._MUTATION_CLASSES}
        assert registered - covered == set(), (
            f"mutation classes with no codec test: {registered - covered}")

    def test_unknown_class_raises(self):
        """A log row from a pipeline with a type this one lacks must fail
        loudly — skipping it would yield a ledger that replayed most of
        history and looked fine."""
        with pytest.raises(KeyError):
            mutation_from_record({"class": "CreateSomethingNew",
                                  "fields": {}})

    def test_unknown_field_is_dropped_not_fatal(self):
        """Forward compatibility: a row written after a field was removed
        still replays."""
        record = mutation_to_record(_CASES[2])
        record["fields"]["a_field_that_no_longer_exists"] = 1
        assert mutation_from_record(record) == _CASES[2]

    def test_tuple_fields_survive_as_tuples(self):
        """known_owners is a tuple; a list would break equality and any
        downstream `is`-typed handling."""
        m = CreateWarrant(count=10, strike=1.0, event_date=ED,
                          known_owners=("Hudson Bay", "Armistice"))
        restored = mutation_from_record(
            json.loads(json.dumps(mutation_to_record(m))))
        assert restored.known_owners == ("Hudson Bay", "Armistice")
        assert isinstance(restored.known_owners, tuple)


# ── the log write path ───────────────────────────────────────────────


class TestLogWrites:
    def test_accepted_mutation_is_logged(self, temp_db):
        temp_db.add_company(CIK, "CELU")
        temp_db.add_filing("acc-1", cik=CIK, form="8-K",
                           filing_date="2024-01-16")
        result = apply_mutations(
            cik=CIK, ticker="CELU", accession="acc-1", form="8-K",
            filing_date="2024-01-16",
            mutations=[CreateWarrant(count=10_000, strike=1.25,
                                     event_date=ED, proposed_id="W-1")],
        )
        assert result.accepted == 1
        rows = _log(temp_db)
        assert len(rows) == 1
        assert rows[0]["kind"] == "create_instrument"
        assert rows[0]["accession_number"] == "acc-1"
        assert rows[0]["form"] == "8-K"
        assert rows[0]["filing_date"] == "2024-01-16"

    def test_logged_row_replays_to_the_same_mutation(self, temp_db):
        temp_db.add_company(CIK, "CELU")
        temp_db.add_filing("acc-1", cik=CIK, form="8-K",
                           filing_date="2024-01-16")
        original = CreateWarrant(count=10_000, strike=1.25, event_date=ED,
                                 proposed_id="W-1", label="Feb Warrants")
        apply_mutations(cik=CIK, ticker="CELU", accession="acc-1", form="8-K",
                        filing_date="2024-01-16", mutations=[original])
        stored = json.loads(_log(temp_db)[0]["mutation_json"])
        assert mutation_from_record(stored) == original

    def test_logged_instrument_id_is_the_resolved_id(self, temp_db):
        """Creates record the id the store ALLOCATED, which may differ
        from the LLM's proposed_id after dedup or collision."""
        temp_db.add_company(CIK, "CELU")
        temp_db.add_filing("acc-1", cik=CIK, form="8-K",
                           filing_date="2024-01-16")
        apply_mutations(
            cik=CIK, ticker="CELU", accession="acc-1", form="8-K",
            filing_date="2024-01-16",
            mutations=[CreateWarrant(count=10_000, strike=1.25,
                                     event_date=ED, proposed_id="W-1")],
        )
        row = _log(temp_db)[0]
        assert row["instrument_id"]
        assert row["instrument_id"].startswith("W-")

    def test_rejected_mutation_is_not_logged(self, temp_db):
        """An amend against a nonexistent instrument. The log must carry
        only what the ledger actually took, or a replay diverges."""
        temp_db.add_company(CIK, "CELU")
        temp_db.add_filing("acc-1", cik=CIK, form="8-K",
                           filing_date="2024-01-16")
        result = apply_mutations(
            cik=CIK, ticker="CELU", accession="acc-1", form="8-K",
            filing_date="2024-01-16",
            mutations=[AmendWarrant(instrument_id="W-999", event_date=ED,
                                    strike=2.0)],
        )
        assert result.accepted == 0
        assert result.rejected >= 1
        assert _log(temp_db) == []
        assert temp_db.execute(
            "SELECT COUNT(*) c FROM dilution_walk_errors")[0]["c"] >= 1

    def test_repeated_calls_on_one_accession_all_persist(self, temp_db):
        """The real shape of a walk: ONE accession passes through
        apply_mutations up to four times (the walk, then anchor
        corrections, note pins, ATM pins), each restarting seq at 0. Keying
        the log on (accession, seq) would have silently dropped all but the
        last pass.
        """
        temp_db.add_company(CIK, "CELU")
        temp_db.add_filing("acc-1", cik=CIK, form="8-K",
                           filing_date="2024-01-16")
        common = dict(cik=CIK, ticker="CELU", accession="acc-1", form="8-K",
                      filing_date="2024-01-16")
        apply_mutations(**common, mutations=[
            CreateWarrant(count=10_000, strike=1.25, event_date=ED,
                          proposed_id="W-1")])
        first = _log(temp_db)
        assert len(first) == 1, "first pass was not logged"
        apply_mutations(**common, mutations=[
            AmendWarrant(instrument_id=first[0]["instrument_id"],
                         event_date=ED, strike=0.90)])
        rows = _log(temp_db)
        assert len(rows) == 2, "second pass overwrote the first"
        assert [r["kind"] for r in rows] == ["create_instrument",
                                            "amend_instrument"]
        # Both passes numbered their own mutation 0 — proof that seq is
        # not unique per accession and `id` is what orders a replay.
        assert [r["seq"] for r in rows] == [0, 0]
        assert rows[0]["id"] < rows[1]["id"]

    def test_multiple_mutations_get_increasing_ids(self, temp_db):
        temp_db.add_company(CIK, "CELU")
        temp_db.add_filing("acc-1", cik=CIK, form="8-K",
                           filing_date="2024-01-16")
        apply_mutations(
            cik=CIK, ticker="CELU", accession="acc-1", form="8-K",
            filing_date="2024-01-16",
            mutations=[
                CreateWarrant(count=10_000, strike=1.25, event_date=ED,
                              proposed_id="W-1"),
                CreateWarrant(count=20_000, strike=2.50, event_date=ED,
                              proposed_id="W-2"),
            ],
        )
        rows = _log(temp_db)
        assert len(rows) == 2
        assert rows[0]["id"] < rows[1]["id"]

    def test_reset_walk_state_clears_the_log(self, temp_db):
        """--force is about to re-derive these rows; keeping the old ones
        would replay a ledger mixing two extraction runs."""
        temp_db.add_company(CIK, "CELU")
        temp_db.add_filing("acc-1", cik=CIK, form="8-K",
                           filing_date="2024-01-16")
        apply_mutations(
            cik=CIK, ticker="CELU", accession="acc-1", form="8-K",
            filing_date="2024-01-16",
            mutations=[CreateWarrant(count=10_000, strike=1.25,
                                     event_date=ED, proposed_id="W-1")],
        )
        assert _log(temp_db)
        reset_walk_state(CIK)
        assert _log(temp_db) == []

    def test_reset_leaves_other_ciks_alone(self, temp_db):
        temp_db.add_company(CIK, "CELU")
        temp_db.add_company(999999, "OTHER")
        temp_db.add_filing("acc-1", cik=CIK, form="8-K",
                           filing_date="2024-01-16")
        temp_db.add_filing("acc-2", cik=999999, form="8-K",
                           filing_date="2024-01-16")
        for cik, ticker, acc in ((CIK, "CELU", "acc-1"),
                                 (999999, "OTHER", "acc-2")):
            apply_mutations(
                cik=cik, ticker=ticker, accession=acc, form="8-K",
                filing_date="2024-01-16",
                mutations=[CreateWarrant(count=10_000, strike=1.25,
                                         event_date=ED, proposed_id="W-1")],
            )
        reset_walk_state(CIK)
        assert _log(temp_db, CIK) == []
        assert len(_log(temp_db, 999999)) == 1

    def test_split_mutations_are_logged(self, temp_db):
        """Splits enter as synthetic mutations under EXTERNAL_SPLIT, and a
        replay that skipped them would mis-scale every instrument."""
        temp_db.add_company(CIK, "CELU")
        apply_mutations(
            cik=CIK, ticker="CELU", accession="split-2024-01-15",
            form="EXTERNAL_SPLIT", filing_date="2024-01-15",
            mutations=[ApplySplit(pre=100, post=1, direction="reverse",
                                  units="common", effective_date=ED)],
        )
        rows = _log(temp_db)
        assert len(rows) == 1
        assert rows[0]["kind"] == "apply_split"
        assert rows[0]["form"] == "EXTERNAL_SPLIT"


# ── replay (scripts/rebuild_ledger.py) ───────────────────────────────


def _rebuild_module():
    """Load the CLI by path — scripts/ is not an importable package."""
    import importlib.util
    path = (Path(__file__).resolve().parent.parent
            / "scripts" / "rebuild_ledger.py")
    spec = importlib.util.spec_from_file_location("rebuild_ledger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_a_history(temp_db) -> None:
    """A small but non-trivial walk: a split, two creates, an amend, a
    drawdown and a close — enough that a replay bug shows up as a diff."""
    temp_db.add_company(CIK, "CELU")
    for acc, d in (("acc-1", "2024-01-16"), ("acc-2", "2024-03-20"),
                   ("acc-3", "2024-06-11")):
        temp_db.add_filing(acc, cik=CIK, form="8-K", filing_date=d)

    apply_mutations(
        cik=CIK, ticker="CELU", accession="split-2024-01-10",
        form="EXTERNAL_SPLIT", filing_date="2024-01-10",
        mutations=[ApplySplit(pre=10, post=1, direction="reverse",
                              units="common", effective_date=date(2024, 1, 10))])
    apply_mutations(
        cik=CIK, ticker="CELU", accession="acc-1", form="8-K",
        filing_date="2024-01-16",
        mutations=[
            CreateAtm(capacity_usd=1e7, event_date=date(2024, 1, 16),
                      placement_agent_canonical="HCW", proposed_id="ATM-1"),
            CreateWarrant(count=10_000, strike=1.25,
                          event_date=date(2024, 1, 16), proposed_id="W-1",
                          label="Jan Warrants"),
        ])
    atm_id = [r["instrument_id"] for r in _log(temp_db)
              if r["kind"] == "create_instrument"][0]
    warrant_id = [r["instrument_id"] for r in _log(temp_db)
                  if r["kind"] == "create_instrument"][1]
    apply_mutations(
        cik=CIK, ticker="CELU", accession="acc-2", form="8-K",
        filing_date="2024-03-20",
        mutations=[RecordDrawdown(instrument_id=atm_id,
                                  drawdown_shares=100_000,
                                  event_date=date(2024, 3, 19),
                                  price_per_share=2.50)])
    # Second pass on the SAME accession, as anchor corrections do.
    apply_mutations(
        cik=CIK, ticker="CELU", accession="acc-2", form="8-K",
        filing_date="2024-03-20",
        mutations=[AmendWarrant(instrument_id=warrant_id,
                                event_date=date(2024, 3, 20), strike=0.90)])
    apply_mutations(
        cik=CIK, ticker="CELU", accession="acc-3", form="8-K",
        filing_date="2024-06-11",
        mutations=[CloseInstrument(instrument_id=warrant_id, reason="expired",
                                  event_date=date(2024, 6, 11))])


class TestReplay:
    def test_dry_run_reproduces_the_ledger_exactly(self, temp_db):
        """The check the whole mutation log exists for. If this diffs, the
        log is incomplete and cannot be relied on for recovery."""
        _seed_a_history(temp_db)
        module = _rebuild_module()
        ok, message = module._rebuild_one("CELU", write=False)
        assert ok, message
        assert "identical" in message

    def test_dry_run_does_not_touch_the_live_db(self, temp_db):
        _seed_a_history(temp_db)
        before = temp_db.execute(
            "SELECT instrument_id, terms_json, outstanding_json, status "
            "FROM dilution_ledger ORDER BY instrument_id")
        module = _rebuild_module()
        module._rebuild_one("CELU", write=False)
        after = temp_db.execute(
            "SELECT instrument_id, terms_json, outstanding_json, status "
            "FROM dilution_ledger ORDER BY instrument_id")
        assert [dict(r) for r in before] == [dict(r) for r in after]

    def test_write_mode_rebuilds_in_place_and_keeps_the_log(self, temp_db):
        """--write resets the ledger before replaying, so it must restore
        the log it is replaying from (reset_walk_state clears it)."""
        _seed_a_history(temp_db)
        expected = [dict(r) for r in temp_db.execute(
            "SELECT instrument_id, terms_json, outstanding_json, status "
            "FROM dilution_ledger ORDER BY instrument_id")]
        log_before = len(_log(temp_db))

        module = _rebuild_module()
        ok, message = module._rebuild_one("CELU", write=True)
        assert ok, message

        after = [dict(r) for r in temp_db.execute(
            "SELECT instrument_id, terms_json, outstanding_json, status "
            "FROM dilution_ledger ORDER BY instrument_id")]
        assert after == expected
        assert len(_log(temp_db)) == log_before, "replay lost the log"

    def test_replay_is_idempotent(self, temp_db):
        """Rebuilding twice must land in the same place — otherwise
        recovery depends on how many times you ran it."""
        _seed_a_history(temp_db)
        module = _rebuild_module()
        module._rebuild_one("CELU", write=True)
        once = [dict(r) for r in temp_db.execute(
            "SELECT instrument_id, terms_json, outstanding_json, status "
            "FROM dilution_ledger ORDER BY instrument_id")]
        module._rebuild_one("CELU", write=True)
        twice = [dict(r) for r in temp_db.execute(
            "SELECT instrument_id, terms_json, outstanding_json, status "
            "FROM dilution_ledger ORDER BY instrument_id")]
        assert once == twice

    def test_write_mode_preserves_the_walk_resume_set(self, temp_db):
        """The invariant that makes replay cheap. If a rebuild cleared
        dilution_walked, the next incremental walk would re-extract every
        filing at full LLM cost — the exact expense replay exists to
        avoid, silently reintroduced by a recovery step."""
        _seed_a_history(temp_db)
        with __import__("db").get_conn() as conn:
            for acc in ("acc-1", "acc-2", "acc-3"):
                conn.execute(
                    "INSERT INTO dilution_walked "
                    "(cik, accession_number, filing_date, walked_at) "
                    "VALUES (?,?,?,?)",
                    (CIK, acc, "2024-01-16", "2026-01-01T00:00:00Z"))

        module = _rebuild_module()
        ok, message = module._rebuild_one("CELU", write=True)
        assert ok, message
        walked = temp_db.execute(
            "SELECT accession_number FROM dilution_walked WHERE cik=?", (CIK,))
        assert {r["accession_number"] for r in walked} == {
            "acc-1", "acc-2", "acc-3"}

    def test_no_log_is_reported_not_silently_ok(self, temp_db):
        """A ticker walked before the log existed must NOT read as
        'reproduced' — that would imply a recovery path it doesn't have."""
        temp_db.add_company(CIK, "CELU")
        module = _rebuild_module()
        ok, message = module._rebuild_one("CELU", write=False)
        assert not ok
        assert "no logged mutations" in message

    def test_unknown_ticker(self, temp_db):
        module = _rebuild_module()
        ok, message = module._rebuild_one("NOPE", write=False)
        assert not ok
        assert "not a tracked ticker" in message

    def test_diff_is_reported_when_the_ledger_was_tampered_with(self, temp_db):
        """Proves the comparison has teeth: corrupt one field and the
        rebuild must notice."""
        _seed_a_history(temp_db)
        temp_db.execute(
            "UPDATE dilution_ledger SET status='terminated' "
            "WHERE type='atm'")
        module = _rebuild_module()
        ok, message = module._rebuild_one("CELU", write=False)
        assert not ok
        assert "status" in message
