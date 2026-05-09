#!/usr/bin/env python3
"""Launch the dilution dashboard.

Usage:
    python run_dashboard.py              # http://127.0.0.1:5050/
    python run_dashboard.py --port 8000  # custom port
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DASHBOARD_LOG_PATH, setup_logging

setup_logging(DASHBOARD_LOG_PATH)

from dashboard.app import app


def main():
    ap = argparse.ArgumentParser(description="Dilution dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
