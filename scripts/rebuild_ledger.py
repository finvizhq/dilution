#!/usr/bin/env python3
"""Rebuild a ledger by replaying its logged mutations — no LLM, no EDGAR.

    python scripts/rebuild_ledger.py --ticker CELU --dry-run   # compare only
    python scripts/rebuild_ledger.py --ticker CELU --write     # rebuild in place
    python scripts/rebuild_ledger.py --all --dry-run

Why this exists: LLM extraction is the one step in the pipeline that costs
money and is not reproducible. Everything downstream — the ledger, the
cards, the Finviz payload — is a deterministic fold of the mutations the
walker applied, and those are recorded in `dilution_mutations`. So the
ledger is a projection, not an asset, and this script is the projector.

Two uses:

  1. **Recovery.** A corrupted or lost ledger replays in seconds at zero
     extraction cost, instead of a full re-walk.
  2. **Deterministic re-projection.** A --force re-walk reassigns every
     instrument_id and re-runs a nondeterministic model, so it cannot tell
     you whether a store.py or cards.py change did what you intended.
     Replay holds the extraction fixed and varies only the code, which
     makes a store-layer diff readable.

--dry-run never touches dilution.db: it replays into a scratch database
and diffs. --write resets the ticker's ledger rows and replays in place.

A rebuilt ledger differing from the live one is not automatically a bug —
if store semantics changed since the walk, the difference is the point.
`pipeline_version` on each logged row tells you which fold produced what.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                              # noqa: E402,F401
import db                                                  # noqa: E402
import dilution.schema as schema                           # noqa: E402
from dilution.finviz_payload import all_tracked_tickers    # noqa: E402
from dilution.ledger.mutations import (                    # noqa: E402
    mutation_from_record,
)

log = logging.getLogger("rebuild_ledger")

# Tables the replay needs to READ but does not re-derive. Copied into the
# scratch DB for a --dry-run. dilution_raw is deliberately absent: filing
# text is only an LLM input, and not needing it is the whole point.
_SEED_TABLES = (
    "dilution_company",
    "dilution_filings",
    "dilution_splits",
    "dilution_mutations",
)

# Ledger columns compared between the live and rebuilt rows. Excludes
# last_seen_* and anchor_miss_count, which track walk bookkeeping rather
# than instrument economics.
_COMPARE_COLUMNS = (
    "type", "created_at", "created_accession", "registration_accession",
    "counterparty_canonical", "counterparty_status",
    "placement_agent_canonical", "label", "terms_json",
    "outstanding_json", "status", "status_at",
)


def _cik_for(ticker: str) -> int | None:
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT cik FROM dilution_company WHERE ticker = ?",
            (ticker.upper(),),
        ).fetchone()
    return int(row["cik"]) if row else None


def _load_log(cik: int) -> list[sqlite3.Row]:
    """Every logged mutation for a CIK, in application order.

    Ordered by `id`, not (filing_date, accession, seq): one accession can
    pass through apply_mutations several times (walk, anchor corrections,
    pins), so only the insertion sequence reflects what actually happened.
    """
    with db.get_conn() as conn:
        try:
            return conn.execute(
                "SELECT * FROM dilution_mutations WHERE cik = ? ORDER BY id",
                (cik,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []


def _ledger_rows(conn: sqlite3.Connection, cik: int) -> dict[str, sqlite3.Row]:
    rows = conn.execute(
        "SELECT * FROM dilution_ledger WHERE cik = ? ORDER BY instrument_id",
        (cik,),
    ).fetchall()
    return {r["instrument_id"]: r for r in rows}


def _snapshot_live_ledger(cik: int) -> dict[str, dict]:
    with db.get_conn() as conn:
        return {k: dict(v) for k, v in _ledger_rows(conn, cik).items()}


def _build_scratch_db(cik: int) -> Path:
    """A throwaway DB carrying the schema plus this CIK's read-only inputs."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="rebuild_ledger_"))
    scratch = tmp_dir / "scratch.db"
    conn = sqlite3.connect(str(scratch))
    try:
        conn.executescript(schema.SCHEMA)
        conn.commit()
    finally:
        conn.close()

    live = str(db.DB_PATH)
    conn = sqlite3.connect(str(scratch))
    try:
        conn.execute("ATTACH DATABASE ? AS live", (live,))
        for table in _SEED_TABLES:
            cols = [r[1] for r in conn.execute(
                f"PRAGMA table_info({table})").fetchall()]
            live_cols = [r[1] for r in conn.execute(
                f"PRAGMA live.table_info({table})").fetchall()]
            shared = [c for c in cols if c in live_cols]
            if not shared:
                continue
            col_list = ", ".join(shared)
            conn.execute(
                f"INSERT INTO {table} ({col_list}) "
                f"SELECT {col_list} FROM live.{table} WHERE cik = ?",
                (cik,))
        conn.commit()
        conn.execute("DETACH DATABASE live")
    finally:
        conn.close()
    return scratch


def _replay(cik: int, ticker: str, rows: list[sqlite3.Row]) -> tuple[int, int]:
    """Apply every logged mutation in order. Returns (applied, failed).

    Each row goes through its own apply_mutations call, matching how it was
    originally applied — batching them would change validation context and
    id_remap scope, which is exactly what must not vary.

    `log_mutations=False` is essential, not an optimization: these rows are
    already in the log, so re-logging them would double it, and the next
    replay would apply every create twice and allocate different ids.
    """
    from dilution.ledger.store import apply_mutations

    applied = failed = 0
    for row in rows:
        try:
            mutation = mutation_from_record(json.loads(row["mutation_json"]))
        except Exception as exc:
            # An unreplayable row means the log is incomplete, which is
            # worse than not having it — surface it, never skip silently.
            log.error("id=%s %s: cannot decode — %s",
                      row["id"], row["kind"], exc)
            failed += 1
            continue
        result = apply_mutations(
            cik=cik, ticker=ticker,
            accession=row["accession_number"],
            form=row["form"] or "",
            filing_date=row["filing_date"] or "",
            mutations=[mutation],
            log_mutations=False,
        )
        applied += result.accepted
        if result.rejected:
            failed += result.rejected
            log.warning("id=%s %s on %s: rejected at replay",
                        row["id"], row["kind"], row["instrument_id"])
    return applied, failed


def _diff(live: dict[str, dict], rebuilt: dict[str, dict]) -> list[str]:
    diffs: list[str] = []
    for iid in sorted(set(live) - set(rebuilt)):
        diffs.append(f"  MISSING from rebuild: {iid} "
                     f"({live[iid]['type']}, {live[iid]['status']})")
    for iid in sorted(set(rebuilt) - set(live)):
        diffs.append(f"  EXTRA in rebuild:     {iid} "
                     f"({rebuilt[iid]['type']}, {rebuilt[iid]['status']})")
    for iid in sorted(set(live) & set(rebuilt)):
        for col in _COMPARE_COLUMNS:
            a, b = live[iid].get(col), rebuilt[iid].get(col)
            if a != b:
                diffs.append(f"  {iid}.{col}: live={a!r} rebuilt={b!r}")
    return diffs


def _rebuild_one(ticker: str, *, write: bool) -> tuple[bool, str]:
    """Returns (ok, message). ok=False means a real problem: no log, an
    undecodable row, or (in --dry-run) a ledger that did not reproduce."""
    cik = _cik_for(ticker)
    if cik is None:
        return False, "not a tracked ticker"

    rows = _load_log(cik)
    if not rows:
        return False, ("no logged mutations — this ticker was walked before "
                       "the mutation log existed, so it cannot be replayed")

    live = _snapshot_live_ledger(cik)
    original_path = db.DB_PATH

    if write:
        # NOT reset_walk_state: that clears dilution_walked (forcing the
        # next incremental walk to re-extract every filing at full LLM
        # cost) and dilution_mutations (the log we are replaying FROM).
        from dilution.ledger.store import reset_ledger_projection
        reset_ledger_projection(cik)
        applied, failed = _replay(cik, ticker, rows)
        rebuilt = _snapshot_live_ledger(cik)
    else:
        scratch = _build_scratch_db(cik)
        try:
            # db.get_conn() resolves DB_PATH at call time, so redirecting
            # it reroutes the whole store layer at once — the same seam
            # the test suite uses to stay off the production DB.
            db.DB_PATH = scratch
            applied, failed = _replay(cik, ticker, rows)
            with db.get_conn() as conn:
                rebuilt = {k: dict(v)
                           for k, v in _ledger_rows(conn, cik).items()}
        finally:
            db.DB_PATH = original_path
            shutil.rmtree(scratch.parent, ignore_errors=True)

    diffs = _diff(live, rebuilt)
    detail = (f"{len(rows)} logged, {applied} applied, {failed} failed; "
              f"{len(live)} live rows vs {len(rebuilt)} rebuilt")
    if failed:
        return False, f"{detail} — REPLAY FAILURES"
    if write:
        return True, detail
    if diffs:
        return False, (f"{detail}; {len(diffs)} difference(s)\n"
                       + "\n".join(diffs[:40])
                       + ("\n  ..." if len(diffs) > 40 else ""))
    return True, f"{detail}; identical"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", action="append", default=[],
                    help="ticker to rebuild (repeatable)")
    ap.add_argument("--all", action="store_true",
                    help="every tracked ticker")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true",
                     help="replay into a scratch DB and diff; dilution.db is "
                          "not modified")
    mode.add_argument("--write", action="store_true",
                     help="reset the ticker's ledger and replay in place")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s")

    tickers = [t.upper() for t in args.ticker]
    if args.all:
        tickers = all_tracked_tickers()
    if not tickers:
        ap.error("pass --ticker TICKER (repeatable) or --all")

    bad = 0
    for ticker in tickers:
        ok, message = _rebuild_one(ticker, write=args.write)
        print(f"{ticker:<8} {'OK  ' if ok else 'DIFF'} {message}")
        if not ok:
            bad += 1

    print(f"\n{len(tickers) - bad}/{len(tickers)} reproduced",
          file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
