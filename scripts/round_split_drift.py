#!/usr/bin/env python3
"""One-shot backfill: re-round existing ledger rows whose split math
left fractional counts / float-drifted prices in `terms_json` and
`outstanding_json`.

Idempotent — running twice changes nothing on the second run.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_conn
from dilution.ledger.store import _COUNT_FIELDS, _PRICE_FIELDS, _PRICE_DECIMALS


def round_dict(d: dict, count_fields: tuple[str, ...],
               price_fields: tuple[str, ...]) -> bool:
    """Mutate d in place. Return True if anything changed."""
    changed = False
    for f in count_fields:
        v = d.get(f)
        if isinstance(v, float):
            new = round(v)
            if new != v:
                d[f] = new
                changed = True
    for f in price_fields:
        v = d.get(f)
        if isinstance(v, float):
            new = round(v, _PRICE_DECIMALS)
            if new != v:
                d[f] = new
                changed = True
    return changed


def main():
    apply = "--apply" in sys.argv
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT instrument_id, terms_json, outstanding_json
                 FROM dilution_ledger
                WHERE type IN ('warrant', 'convertible', 'preferred')"""
        ).fetchall()

        touched = 0
        samples = []
        for r in rows:
            terms = json.loads(r["terms_json"] or "{}")
            outstanding = json.loads(r["outstanding_json"] or "{}")
            t_changed = round_dict(terms, _COUNT_FIELDS, _PRICE_FIELDS)
            o_changed = round_dict(outstanding, _COUNT_FIELDS, _PRICE_FIELDS)
            if t_changed or o_changed:
                touched += 1
                if len(samples) < 8:
                    samples.append((r["instrument_id"],
                                    terms, outstanding))
                if apply:
                    conn.execute(
                        "UPDATE dilution_ledger "
                        "SET terms_json=?, outstanding_json=? "
                        "WHERE instrument_id=?",
                        (json.dumps(terms), json.dumps(outstanding),
                         r["instrument_id"]),
                    )
        if apply:
            conn.commit()

    print(f"{'APPLIED' if apply else 'DRY RUN'}: "
          f"{touched} rows would be / were updated.")
    for iid, t, o in samples:
        print(f"  {iid}: terms={t} outstanding={o}")
    if not apply:
        print("Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
