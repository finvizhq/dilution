#!/usr/bin/env python3
"""Run the dilution pipeline on the Open Access ticker set.

Usage:
    scripts/run_open_access.py                # default tickers
    scripts/run_open_access.py --years 3      # extra args forwarded to run_dilution.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

TICKERS = ["SMX", "DJT", "ARBE", "BYND", "AMC", "GNS", "GME", "QUBT", "MSTR"]

REPO_ROOT = Path(__file__).resolve().parent.parent

if sys.stdout.isatty():
    BOLD, DIM = "\033[1m", "\033[2m"
    RED, GREEN, YELLOW, BLUE = "\033[31m", "\033[32m", "\033[33m", "\033[34m"
    RESET = "\033[0m"
else:
    BOLD = DIM = RED = GREEN = YELLOW = BLUE = RESET = ""


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"{DIM}[{ts()}]{RESET} {msg}", flush=True)


def main() -> int:
    os.chdir(REPO_ROOT)
    extra_args = sys.argv[1:]

    total = len(TICKERS)
    log(f"{BOLD}Running dilution pipeline for {total} tickers:{RESET} {' '.join(TICKERS)}")
    if extra_args:
        log(f"forwarding extra args: {' '.join(extra_args)}")
    batch_start = time.monotonic()

    ok_list: list[str] = []
    failed_list: list[str] = []

    for i, t in enumerate(TICKERS, start=1):
        print(f"\n{BOLD}{BLUE}=== [{i}/{total}] {t} ==={RESET}", flush=True)
        log(f"starting {t}")
        t_start = time.monotonic()

        rc = subprocess.run(
            [sys.executable, "run_dilution.py", t, *extra_args]
        ).returncode
        elapsed = int(time.monotonic() - t_start)

        if rc == 0:
            log(f"{GREEN}✓ {t} finished in {elapsed}s{RESET}")
            ok_list.append(f"{t} ({elapsed}s)")
        else:
            log(f"{RED}✗ {t} failed (exit {rc}) after {elapsed}s{RESET}")
            failed_list.append(f"{t} (exit {rc}, {elapsed}s)")

    batch_elapsed = int(time.monotonic() - batch_start)
    mins, secs = divmod(batch_elapsed, 60)

    print(f"\n{BOLD}{BLUE}=== Summary ==={RESET}", flush=True)
    log(f"total runtime: {mins}m {secs}s")
    log(f"{GREEN}succeeded: {len(ok_list)}/{total}{RESET}")
    for s in ok_list:
        print(f"  {GREEN}✓{RESET} {s}", flush=True)

    if failed_list:
        log(f"{RED}failed: {len(failed_list)}/{total}{RESET}")
        for s in failed_list:
            print(f"  {RED}✗{RESET} {s}", flush=True)
        return 1

    log(f"{GREEN}{BOLD}all tickers completed successfully{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
