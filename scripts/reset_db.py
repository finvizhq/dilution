"""Erase dilution.db and recreate empty schema.

Usage:
    python scripts/reset_db.py            # prompts for confirmation
    python scripts/reset_db.py --yes      # skip prompt
    python scripts/reset_db.py --no-backup
"""

import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DB_PATH
from dilution.schema import init_dilution_db


def _abort_if_active(db: Path) -> None:
    """Refuse to reset when another process has the DB open with pending
    state. Catches the orphan-pipeline race where reset_db.py wipes the
    file mid-fetch and the still-running worker's next `_store` insert
    hits FOREIGN KEY constraint failed against the freshly-empty schema.

    Detection: try BEGIN EXCLUSIVE with a 0.5s timeout. SQLite's
    exclusive lock conflicts with ANY other connection's read or write
    transaction, so this fails fast when a `run_dilution.py` worker is
    active. Clean idle state acquires the lock instantly and releases it.
    """
    if not db.exists():
        return
    try:
        conn = sqlite3.connect(str(db), timeout=0.5)
    except sqlite3.OperationalError as exc:
        raise SystemExit(f"refusing to reset: cannot open {db} ({exc})")
    try:
        conn.execute("BEGIN EXCLUSIVE")
        conn.rollback()
    except sqlite3.OperationalError as exc:
        raise SystemExit(
            f"refusing to reset: {db} is in use ({exc}). Kill any active "
            "run_dilution.py / eval pipeline processes first — try "
            "`pgrep -fa run_dilution` to find them."
        )
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Wipe dilution.db and recreate empty schema.")
    ap.add_argument("--yes", "-y", action="store_true", help="skip confirmation prompt")
    ap.add_argument("--no-backup", action="store_true", help="don't keep a timestamped backup")
    args = ap.parse_args()

    db = Path(DB_PATH)
    print(f"Target DB: {db}")

    if not args.yes:
        resp = input("Erase this database? [y/N]: ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborted.")
            return 1

    _abort_if_active(db)

    if db.exists():
        if not args.no_backup:
            backup = db.with_name(f"{db.name}.bak-{int(time.time())}")
            shutil.copy2(db, backup)
            print(f"Backup written: {backup}")
        for sidecar in (db, db.with_suffix(db.suffix + "-wal"), db.with_suffix(db.suffix + "-shm")):
            if sidecar.exists():
                sidecar.unlink()
                print(f"Removed: {sidecar}")
    else:
        print("DB did not exist; creating fresh.")

    init_dilution_db()
    print(f"Initialized empty schema at {db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
