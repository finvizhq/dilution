"""Erase dilution.db and recreate empty schema.

Usage:
    python scripts/reset_db.py            # prompts for confirmation
    python scripts/reset_db.py --yes      # skip prompt
    python scripts/reset_db.py --no-backup
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DB_PATH
from dilution.schema import init_dilution_db


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
