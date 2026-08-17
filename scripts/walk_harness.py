#!/usr/bin/env python3
"""Prove a walk-time refactor changed nothing — without paying for a re-walk.

    python scripts/walk_harness.py                     # print digests
    python scripts/walk_harness.py --save base.txt     # record a baseline
    python scripts/walk_harness.py --check base.txt    # compare (exit 1 on drift)
    python scripts/walk_harness.py --coverage          # what each probe exercised
    python scripts/walk_harness.py --probes ab         # subset

Why this exists: render-time code (cards, badges, status derivers) is
provable for free — `run_eval_all.py --no-log` re-scores against the
already-walked DB and an identical table means an identical projection.
Walk-time code has no such oracle. store.py / validate.py / anchor.py only
execute DURING a walk, and a re-walk re-runs a nondeterministic model, so
an A/B across a walk measures model drift, not your change.

So this drives those modules directly and hashes what they produce:

  A  synthetic mutation corpus -> temp DB -> digest the resulting ledger
     (every event_kind x instrument type through apply_mutations, so
      _apply_create/_apply_amend/_apply_record_event/_apply_close/
      _apply_split/_apply_restate_atm all execute)
  B  validate._validate_one over a mutation x filing-form grid
  C  pure-ish store helpers driven by REAL production ledger rows
  D  anchor.reconcile_against_periodic over REAL rows, with the overhang
     re-derived FROM those rows at several perturbations so the match,
     drift-correct, synthesize and close paths all fire

Typical use around a refactor:

    python scripts/walk_harness.py --save /tmp/base.txt
    ...edit store.py / anchor.py / validate.py...
    python scripts/walk_harness.py --check /tmp/base.txt

A moved digest is not automatically a bug — if you intended a semantic
change, it SHOULD move, and --coverage plus a manual diff tells you
whether it moved the way you meant. An unchanged digest across a pure
restructuring is the thing worth having.

Caveats worth knowing:
  * --coverage is not decoration. A corpus where everything is rejected
    hashes just as stably as a corpus that exercises every branch, and
    proves nothing. Check the mix before trusting a match.
  * Probes C and D read the production DB read-only and are skipped when
    it is absent (a fresh clone gets A and B only).
  * Wall-clock columns are excluded from the digest — otherwise two runs
    a second apart disagree, which looks alarmingly like real drift.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import shutil
import sqlite3
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                              # noqa: E402,F401
import db                                                  # noqa: E402
import dilution.schema as schema                           # noqa: E402

log = logging.getLogger("walk_harness")

PROD_DB = ROOT / "dilution.db"

# Columns stamped with wall-clock time. They vary run to run and would
# mask every real difference behind noise; the error/drawdown CONTENT
# they sit next to is still digested.
_VOLATILE_COLS = frozenset({"detected_at", "created_ts", "logged_at"})

# Ledger state the digest covers. history_json is included deliberately:
# the action labels a refactor might silently rename are the point.
_LEDGER_COLS = (
    "instrument_id", "cik", "type", "label", "status", "status_at",
    "terms_json", "outstanding_json", "history_json", "created_at",
    "created_accession", "last_seen_accession", "last_seen_date",
)

_HARNESS_CIK = 999_001
_HARNESS_TICKER = "HARN"


def _digest(rows) -> str:
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, default=str).encode()
    ).hexdigest()


def _fresh_db(tmp: Path) -> str:
    """A throwaway DB on the production schema, wired into db.get_conn()."""
    path = tmp / "harness.db"
    conn = sqlite3.connect(path)
    conn.executescript(schema.SCHEMA)
    conn.commit()
    conn.close()
    db.DB_PATH = str(path)
    return str(path)


def _dump_ledger(path: str) -> list:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows = []
    for r in conn.execute(
        f"SELECT {', '.join(_LEDGER_COLS)} FROM dilution_ledger "
        "ORDER BY instrument_id"
    ):
        rows.append({k: r[k] for k in r.keys()})
    for table, order in (("dilution_ledger_drawdowns", "instrument_id, event_date"),
                         ("dilution_walk_errors", "rowid")):
        try:
            for r in conn.execute(f"SELECT * FROM {table} ORDER BY {order}"):
                rows.append({k: r[k] for k in r.keys()
                             if k not in _VOLATILE_COLS})
        except sqlite3.Error:
            pass
    conn.close()
    return rows


def _prod_rows(sql: str, params: tuple = ()) -> list[dict]:
    """Read-only fetch from the production DB."""
    conn = sqlite3.connect(f"file:{PROD_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()


# ── Probe A: synthetic mutation corpus ───────────────────────────────
def probe_a() -> list:
    """Apply a broad mutation corpus to a temp DB; return final ledger state."""
    from dilution.ledger import mutations as M
    from dilution.ledger.store import apply_mutations

    tmp = Path(tempfile.mkdtemp(prefix="walk-harness-"))
    try:
        path = _fresh_db(tmp)
        cik, ticker = _HARNESS_CIK, _HARNESS_TICKER
        d = date(2024, 3, 15)

        creates = [
            M.CreateWarrant(count=1_000_000, strike=2.50, event_date=d,
                            expiration=date(2029, 3, 15),
                            counterparty_canonical="Armistice Capital"),
            M.CreateWarrant(count=250_000, strike=0.0001, event_date=d,
                            expiration=date(2029, 3, 15), is_pre_funded=True,
                            counterparty_canonical="Hudson Bay Capital"),
            M.CreateConvertible(principal=5_000_000,
                                principal_remaining=5_000_000,
                                conv_price=1.25, event_date=d,
                                maturity=date(2027, 3, 15), rate=8.0,
                                counterparty_canonical="Yorkville Advisors"),
            M.CreatePreferred(count=10_000, series_letter="B", event_date=d,
                              stated_value=1000.0, conv_price=2.0,
                              counterparty_canonical="Lincoln Park Capital"),
            M.CreateAtm(capacity_usd=25_000_000, event_date=d,
                        agreement_date=d,
                        placement_agent_canonical="H.C. Wainwright"),
            M.CreateShelf(capacity_usd=100_000_000, event_date=d, form="S-3"),
            M.CreateEquityLine(capacity_usd=15_000_000, event_date=d,
                               agreement_date=d,
                               counterparty_canonical="Yorkville Advisors"),
            M.CreateS1Offering(anticipated_deal_size=8_000_000, event_date=d),
            M.CreateEquity(count=2_000_000, price_per_share=1.10, event_date=d,
                           counterparty_canonical="Armistice Capital"),
        ]
        apply_mutations(cik=cik, ticker=ticker, accession="0001", form="8-K",
                        filing_date="2024-03-16", mutations=creates)

        # Ids are allocated by the store, so read them back before
        # building events that reference them.
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        by_type = {r["type"]: r["instrument_id"] for r in conn.execute(
            "SELECT type, instrument_id FROM dilution_ledger WHERE cik=? "
            "ORDER BY instrument_id", (cik,))}
        warrants = [r["instrument_id"] for r in conn.execute(
            "SELECT instrument_id FROM dilution_ledger "
            "WHERE cik=? AND type='warrant' ORDER BY instrument_id", (cik,))]
        conn.close()

        w1 = warrants[0]
        w2 = warrants[1] if len(warrants) > 1 else warrants[0]
        cv, pf = by_type.get("convertible"), by_type.get("preferred")
        atm, sh = by_type.get("atm"), by_type.get("shelf")
        el, s1 = by_type.get("equity_line"), by_type.get("s1_offering")
        eq = by_type.get("equity")

        ev, amend_d, close_d = (date(2024, 6, 1), date(2024, 9, 1),
                                date(2024, 12, 1))
        events = [
            # Ordinary cash exercise, then a cashless one where warrants
            # surrendered exceeds shares delivered.
            M.RecordExercise(instrument_id=w1, shares=100_000,
                             event_date=ev, price=2.50),
            M.RecordExercise(instrument_id=w1, shares=26_070, event_date=ev,
                             price=2.50, warrants_exercised=76_376),
            M.RecordConversion(instrument_id=cv, shares_issued=800_000,
                               event_date=ev, principal_converted=1_000_000),
            M.RecordPartialRedemption(instrument_id=cv, event_date=ev,
                                      principal_redeemed=500_000,
                                      cash_paid=550_000),
            # Preferred conversions/redemptions travel as share counts.
            M.RecordConversion(instrument_id=pf, shares_issued=1_000_000,
                               event_date=ev, preferred_shares_converted=2_000),
            M.RecordPartialRedemption(instrument_id=pf, event_date=ev,
                                      preferred_shares_redeemed=1_000),
            # Per-share priced draw, then the aggregate-only fallback.
            M.RecordDrawdown(instrument_id=atm, drawdown_shares=500_000,
                             event_date=ev, price_per_share=1.20),
            M.RecordDrawdown(instrument_id=atm, drawdown_shares=100_000,
                             event_date=ev, drawdown_amount_usd=125_000),
            M.RecordPartialTermination(instrument_id=atm,
                                       capacity_reduced_usd=1_000_000,
                                       event_date=ev),
            M.RecordDrawdown(instrument_id=sh, drawdown_shares=1_000_000,
                             event_date=ev, price_per_share=1.05),
            M.RecordDrawdown(instrument_id=el, drawdown_shares=250_000,
                             event_date=ev, price_per_share=0.95),
            # Equity closing books cash; warrant closing re-bases dates.
            M.ConfirmClosing(instrument_id=eq, event_date=ev,
                             count_actual=1_950_000,
                             gross_proceeds_usd=2_145_000),
            M.ConfirmClosing(instrument_id=w2, event_date=ev),
        ]
        amends = [
            M.AmendWarrant(instrument_id=w1, event_date=amend_d, strike=1.75,
                           expiration=date(2030, 3, 15)),
            M.AmendConvertible(instrument_id=cv, event_date=amend_d,
                               conv_price=0.80),
            M.AmendPreferred(instrument_id=pf, event_date=amend_d,
                             conv_price=1.10),
            M.AmendAtm(instrument_id=atm, event_date=amend_d,
                       capacity_usd=40_000_000),
            M.AmendShelf(instrument_id=sh, event_date=amend_d,
                         capacity_usd=150_000_000),
            M.AmendEquityLine(instrument_id=el, event_date=amend_d,
                              capacity_usd=20_000_000),
            M.AmendS1Offering(instrument_id=s1, event_date=amend_d,
                              final_deal_size=7_500_000),
        ]
        steps = [
            ("0002", "10-Q", "2024-06-05", events),
            ("0003", "10-K", "2024-09-01", amends),
            ("0004", "424B5", "2024-10-01",
             [M.RestateAtm(predecessor_id=atm, capacity_usd=60_000_000,
                           event_date=date(2024, 10, 1))]),
            # A reverse split rescales every live instrument.
            ("0005", "8-K", "2024-11-01",
             [M.ApplySplit(post=1, pre=20, direction="reverse",
                           effective_date=date(2024, 11, 1))]),
            ("0006", "8-K", "2024-12-01",
             [M.CloseInstrument(instrument_id=s1, reason="withdrawn",
                                event_date=close_d),
              M.CloseInstrument(instrument_id=el, reason="terminated",
                                event_date=close_d),
              M.CloseInstrument(instrument_id=w2, reason="expired",
                                event_date=close_d)]),
        ]
        for accession, form, filing_date, muts in steps:
            apply_mutations(cik=cik, ticker=ticker, accession=accession,
                            form=form, filing_date=filing_date, mutations=muts)
        return _dump_ledger(path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def coverage_a(rows: list) -> str:
    actions = collections.Counter()
    instruments = 0
    for r in rows:
        if "history_json" in r:
            instruments += 1
            for h in json.loads(r.get("history_json") or "[]"):
                actions[h.get("action")] += 1
    return (f"{instruments} instruments, "
            f"{sum(actions.values())} history events: "
            f"{dict(actions.most_common())}")


# ── Probe B: validate._validate_one grid ─────────────────────────────
def probe_b() -> list:
    """Run every mutation shape past the validator under every filing form."""
    from dilution.ledger import mutations as M
    from dilution.ledger.validate import _validate_one

    d = date(2024, 5, 1)
    ledger = {
        "W-001": {"instrument_id": "W-001", "type": "warrant",
                  "status": "active",
                  "terms_json": json.dumps({"strike": 2.0, "count": 1e6}),
                  "outstanding_json": json.dumps({"count": 1e6})},
        "CV-001": {"instrument_id": "CV-001", "type": "convertible",
                   "status": "active",
                   "terms_json": json.dumps({"principal": 5e6,
                                             "conv_price": 1.0}),
                   "outstanding_json": json.dumps({"principal_remaining": 5e6})},
        "P-001": {"instrument_id": "P-001", "type": "preferred",
                  "status": "active",
                  "terms_json": json.dumps({"stated_value": 1000.0}),
                  "outstanding_json": json.dumps({"count": 5000})},
        "ATM-001": {"instrument_id": "ATM-001", "type": "atm",
                    "status": "active",
                    "terms_json": json.dumps({"capacity_usd": 2e7}),
                    "outstanding_json": json.dumps({"drawn_usd": 5e6})},
        "SH-001": {"instrument_id": "SH-001", "type": "shelf",
                   "status": "active",
                   "terms_json": json.dumps({"capacity_usd": 1e8}),
                   "outstanding_json": json.dumps({})},
        # Terminal row: every mutation against it must be rejected.
        "DEAD-1": {"instrument_id": "DEAD-1", "type": "warrant",
                   "status": "expired", "terms_json": json.dumps({}),
                   "outstanding_json": json.dumps({})},
    }
    live = {"W-001", "CV-001", "P-001", "ATM-001", "SH-001"}

    candidates = [
        M.ApplySplit(post=1, pre=20, direction="reverse", effective_date=d),
        M.ApplySplit(post=0, pre=20, direction="reverse", effective_date=d),
        M.ApplySplit(post=2, pre=1, direction="forward", effective_date=d),
        M.CreateWarrant(count=1e5, strike=1.0, event_date=d),
        M.CreateShelf(capacity_usd=5e7, event_date=d, form="S-3"),
        M.CreateShelf(capacity_usd=None, event_date=d, form="S-3"),
        M.CreateShelf(capacity_usd=5e7, event_date=d, form="424B5"),
        M.CreateAtm(capacity_usd=1e7, event_date=d, agreement_date=d),
        M.CreateConvertible(principal=1e6, principal_remaining=1e6,
                            conv_price=1.0, event_date=d),
        M.CreatePreferred(count=100, series_letter="A", event_date=d,
                          stated_value=1000.0),
        M.CreateEquity(count=1e6, price_per_share=1.0, event_date=d),
        M.CreateEquityLine(capacity_usd=1e7, event_date=d),
        M.CreateS1Offering(anticipated_deal_size=5e6, event_date=d),
        M.RestateAtm(predecessor_id="ATM-001", capacity_usd=3e7, event_date=d),
        M.RestateAtm(predecessor_id="SH-001", capacity_usd=3e7, event_date=d),
        M.RestateAtm(predecessor_id="NOPE-1", capacity_usd=3e7, event_date=d),
        M.RestateAtm(predecessor_id="DEAD-1", capacity_usd=3e7, event_date=d),
        M.RecordExercise(instrument_id="W-001", shares=1e5, event_date=d),
        M.RecordExercise(instrument_id="DEAD-1", shares=1e5, event_date=d),
        M.RecordExercise(instrument_id="NOPE-1", shares=1e5, event_date=d),
        M.RecordDrawdown(instrument_id="ATM-001", drawdown_shares=1e5,
                         event_date=d, price_per_share=1.0),
        # Overflows the program's capacity — must trip the overflow guard.
        M.RecordDrawdown(instrument_id="ATM-001", drawdown_shares=1e9,
                         event_date=d, price_per_share=100.0),
        M.RecordDrawdown(instrument_id="W-001", drawdown_shares=1e5,
                         event_date=d, price_per_share=1.0),
        M.RecordDrawdown(instrument_id="SH-001", drawdown_shares=1e5,
                         event_date=d, drawdown_amount_usd=1e5),
        M.RecordConversion(instrument_id="CV-001", shares_issued=1e6,
                           event_date=d, principal_converted=1e6),
        M.RecordConversion(instrument_id="CV-001", shares_issued=1e6,
                           event_date=d, principal_converted=9e9),
        M.RecordConversion(instrument_id="P-001", shares_issued=1e6,
                           event_date=d, preferred_shares_converted=1e3),
        M.RecordConversion(instrument_id="P-001", shares_issued=1e6,
                           event_date=d, preferred_shares_converted=9e9),
        M.RecordPartialRedemption(instrument_id="CV-001", event_date=d,
                                  principal_redeemed=1e6),
        M.RecordPartialRedemption(instrument_id="P-001", event_date=d,
                                  preferred_shares_redeemed=9e9),
        M.RecordPartialTermination(instrument_id="ATM-001",
                                   capacity_reduced_usd=1e6, event_date=d),
        M.ConfirmClosing(instrument_id="W-001", event_date=d,
                         count_actual=9e5),
        M.AmendWarrant(instrument_id="W-001", event_date=d, strike=1.0),
        M.AmendWarrant(instrument_id="DEAD-1", event_date=d, strike=1.0),
        M.AmendConvertible(instrument_id="CV-001", event_date=d,
                           principal_remaining=1e6),
        M.AmendConvertible(instrument_id="CV-001", event_date=d,
                           principal_remaining=-5.0),
        M.AmendPreferred(instrument_id="P-001", event_date=d,
                         liquidation_preference=5e6),
        M.AmendAtm(instrument_id="ATM-001", event_date=d, capacity_usd=3e7),
        M.AmendShelf(instrument_id="SH-001", event_date=d, capacity_usd=2e8),
        M.CloseInstrument(instrument_id="W-001", reason="expired",
                          event_date=d),
        M.CloseInstrument(instrument_id="W-001", reason="superseded",
                          event_date=d),
        M.CloseInstrument(instrument_id="W-001", reason="superseded",
                          event_date=d, replaced_by="CV-001"),
        M.CloseInstrument(instrument_id="ATM-001", reason="terminated",
                          event_date=d),
        M.CloseInstrument(instrument_id="SH-001", reason="expired",
                          event_date=d),
        M.CloseInstrument(instrument_id="DEAD-1", reason="expired",
                          event_date=d),
        M.CloseInstrument(instrument_id="NOPE-1", reason="expired",
                          event_date=d),
        M.CloseInstrument(instrument_id="P-001", reason="redeemed",
                          event_date=d),
        M.CloseInstrument(instrument_id="CV-001", reason="exercised",
                          event_date=d),
    ]
    # Several rules are form-conditional (shelf creates, ATM restates,
    # closes on post-effective amendments), so every candidate runs
    # against every form.
    forms = [None, "8-K", "10-K", "10-Q", "424B5", "S-3", "POS AM", "RW"]

    out = []
    for m in candidates:
        for form in forms:
            try:
                r = _validate_one(m, ledger, live, {}, None, form)
                out.append([type(m).__name__, form, r.accepted,
                            getattr(r, "error_kind", None),
                            getattr(r, "error", None)])
            except Exception as exc:      # the failure shape must be stable too
                out.append([type(m).__name__, form, "EXC",
                            type(exc).__name__, str(exc)])
    return out


def coverage_b(rows: list) -> str:
    verdicts = collections.Counter(r[2] for r in rows)
    kinds = collections.Counter(r[3] for r in rows if r[2] is False)
    return (f"{len(rows)} cases, accepted={verdicts.get(True, 0)} "
            f"rejected={verdicts.get(False, 0)} exc={verdicts.get('EXC', 0)}; "
            f"{len(kinds)} rejection kinds: {dict(kinds.most_common(6))}")


# ── Probe C: pure-ish store helpers over REAL rows ───────────────────
def probe_c() -> list:
    """Drive the identity/discriminator helpers with real ledger rows."""
    from dilution.ledger import store as S

    rows = _prod_rows("SELECT * FROM dilution_ledger "
                      "ORDER BY instrument_id LIMIT 400")
    out = []
    for r in rows:
        terms = json.loads(r.get("terms_json") or "{}")
        outstanding = json.loads(r.get("outstanding_json") or "{}")
        for label, fn in (
            ("discriminator",
             lambda: S._has_discriminator(r["type"] or "", terms, outstanding)),
            ("split_skip",
             lambda: sorted(S._preferred_price_split_skip(terms))),
        ):
            try:
                out.append([label, r["instrument_id"], fn()])
            except Exception as exc:
                out.append([label, r["instrument_id"],
                            f"EXC:{type(exc).__name__}"])

    # Pairwise identity checks within each type — capped so the probe
    # stays a few seconds rather than quadratic over the whole ledger.
    by_type: dict = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)
    for _t, group in sorted(by_type.items()):
        group = group[:40]
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                ta = json.loads(a.get("terms_json") or "{}")
                tb = json.loads(b.get("terms_json") or "{}")
                try:
                    out.append(["end_dates", a["instrument_id"],
                                b["instrument_id"], S._end_dates_conflict(ta, tb)])
                except Exception as exc:
                    out.append(["end_dates", a["instrument_id"],
                                b["instrument_id"], f"EXC:{type(exc).__name__}"])
    return out


def coverage_c(rows: list) -> str:
    kinds = collections.Counter(r[0] for r in rows)
    excs = sum(1 for r in rows if any(isinstance(x, str)
                                      and x.startswith("EXC") for x in r))
    return f"{len(rows)} calls {dict(kinds)}, {excs} exceptions"


# ── Probe D: anchor reconcile over REAL rows ─────────────────────────
def _overhang_from_row(row: dict, *, perturb: float = 1.0) -> dict:
    """Shape a real ledger row back into an OverhangRow-ish dict.

    Deriving the filing side FROM the ledger side is what makes this
    probe bite: at perturb=1.0 everything should match cleanly, and at
    0.85 every matched row should produce a field-drift correction.
    """
    t = (row.get("type") or "").lower()
    terms = (row.get("terms") if isinstance(row.get("terms"), dict)
             else json.loads(row.get("terms_json") or "{}"))
    out = (row.get("outstanding") if isinstance(row.get("outstanding"), dict)
           else json.loads(row.get("outstanding_json") or "{}"))
    over = {"category": t, "instrument_name": row.get("label") or "",
            "issue_date": (row.get("created_at") or "")[:10]}

    count = out.get("count")
    if count:
        over["outstanding_count"] = float(count) * perturb
        over["count"] = float(count) * perturb
    price = terms.get("strike") or terms.get("conv_price")
    if price:
        over["strike_or_conversion_price"] = float(price) * perturb
        over["conversion_price"] = float(price) * perturb
    principal = out.get("principal_remaining") or terms.get("principal")
    if principal:
        over["principal_amount"] = float(principal) * perturb
    expiry = terms.get("expiration") or terms.get("maturity")
    if expiry:
        over["maturity_or_expiry"] = str(expiry)[:10]
    capacity = terms.get("capacity_usd")
    if capacity:
        over["total_capacity_usd"] = float(capacity) * perturb
    remaining = out.get("remaining_capacity_usd")
    if remaining is not None:
        over["remaining_capacity_usd"] = float(remaining) * perturb
    drawn = out.get("drawn_usd")
    if drawn is not None:
        over["drawn_to_date_usd"] = float(drawn) * perturb
    if terms.get("series_letter"):
        over["series_letter"] = terms["series_letter"]
    if terms.get("counterparty_canonical"):
        over["investor"] = terms["counterparty_canonical"]
    if terms.get("placement_agent_canonical"):
        over["sales_agent"] = terms["placement_agent_canonical"]
    return over


def probe_d() -> list:
    """Reconcile real ledgers against overhang derived from themselves."""
    import dilution.ledger.anchor as anchor
    from dilution.ledger.store import get_open_instruments

    db.DB_PATH = str(PROD_DB)
    ciks = [r["cik"] for r in _prod_rows(
        "SELECT cik, COUNT(*) n FROM dilution_ledger WHERE status='active' "
        "GROUP BY cik ORDER BY n DESC LIMIT 10")]

    out = []
    for cik in ciks:
        try:
            rows = get_open_instruments(cik)
        except Exception as exc:
            out.append([cik, f"LOADEXC:{type(exc).__name__}"])
            continue
        if not rows:
            continue
        scenarios = {
            # clean agreement; every row should match, nothing to correct
            "exact": [_overhang_from_row(r) for r in rows],
            # 15% drift on every numeric axis -> field corrections
            "drift": [_overhang_from_row(r, perturb=0.85) for r in rows],
            # half the rows itemised -> extra_in_ledger handling
            "partial": [_overhang_from_row(r)
                        for r in rows[: max(1, len(rows) // 2)]],
            # silent filing -> the reapers and close paths
            "empty": [],
        }
        for name, overhang in sorted(scenarios.items()):
            for as_of in ("2025-06-30", "2026-03-31"):
                try:
                    res = anchor.reconcile_against_periodic(
                        cik=cik, accession="harness", filing_date=as_of,
                        as_of_date=as_of, filing_overhang=overhang,
                        ledger_open=rows,
                    )
                    out.append([
                        cik, name, as_of,
                        sorted(str(d) for d in (res.diffs or [])),
                        sorted(f"{type(m).__name__}:"
                               f"{getattr(m, 'instrument_id', '')}"
                               for m in (res.correction_mutations or [])),
                    ])
                except Exception as exc:
                    out.append([cik, name, as_of,
                                f"EXC:{type(exc).__name__}:{exc}"])
    return out


def coverage_d(rows: list) -> str:
    excs = [r for r in rows
            if any(isinstance(x, str) and x.startswith(("EXC", "LOADEXC"))
                   for x in r)]
    ok = [r for r in rows if r not in excs and len(r) > 4]
    diffs = sum(len(r[3]) for r in ok)
    muts = collections.Counter(m.split(":")[0] for r in ok for m in r[4])
    return (f"{len(rows)} scenario runs, {len(excs)} exceptions, "
            f"{diffs} diffs, {sum(muts.values())} corrections "
            f"across {len(muts)} mutation kinds")


PROBES = {
    "a": ("synthetic-corpus", probe_a, coverage_a, False),
    "b": ("validate-grid", probe_b, coverage_b, False),
    "c": ("store-helpers-real", probe_c, coverage_c, True),
    "d": ("anchor-reconcile", probe_d, coverage_d, True),
}


def run(selected: str, show_coverage: bool) -> dict[str, str]:
    results: dict[str, str] = {}
    for key in selected:
        title, fn, cov, needs_prod = PROBES[key]
        if needs_prod and not PROD_DB.exists():
            print(f"{key.upper()} {title:<20} SKIPPED (no {PROD_DB.name})")
            continue
        rows = fn()
        results[key] = _digest(rows)
        print(f"{key.upper()} {title:<20} {results[key]}")
        if show_coverage:
            print(f"    coverage: {cov(rows)}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Digest walk-time behaviour so a refactor can be "
                    "proven behaviour-preserving without a re-walk.")
    ap.add_argument("--probes", default="abcd",
                    help="subset of probes to run (default: abcd)")
    ap.add_argument("--save", metavar="PATH",
                    help="write the digests to PATH as a baseline")
    ap.add_argument("--check", metavar="PATH",
                    help="compare against a saved baseline; exit 1 on drift")
    ap.add_argument("--coverage", action="store_true",
                    help="report what each probe actually exercised — an "
                         "all-rejected corpus hashes stably and proves nothing")
    args = ap.parse_args()

    selected = [c for c in args.probes.lower() if c in PROBES]
    if not selected:
        ap.error(f"--probes must name some of {''.join(PROBES)}")

    results = run(selected, args.coverage)

    if args.save:
        Path(args.save).write_text(
            "".join(f"{k} {v}\n" for k, v in sorted(results.items())))
        print(f"\nbaseline written to {args.save}")

    if args.check:
        baseline = {}
        for line in Path(args.check).read_text().splitlines():
            if line.strip():
                k, v = line.split()
                baseline[k] = v
        drifted = [k for k, v in results.items()
                   if k in baseline and baseline[k] != v]
        missing = [k for k in results if k not in baseline]
        if drifted:
            print("\nDRIFT:")
            for k in drifted:
                print(f"  {PROBES[k][0]}: {baseline[k]} -> {results[k]}")
            print("\nIf the change was meant to be behaviour-preserving, this "
                  "is a regression. If it was semantic, re-baseline.")
            return 1
        if missing:
            print(f"\nnot in baseline (not compared): {', '.join(missing)}")
        print("\nno drift — every compared probe matches the baseline")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
