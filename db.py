"""SQLite connection helper for the dilution pipeline.

Schema lives in dilution/schema.py — call init_dilution_db() there to
create tables. This module only owns the connection context and a
shared timestamp helper.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime

from config import DB_PATH


@contextmanager
def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
