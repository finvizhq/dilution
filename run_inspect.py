#!/usr/bin/env python3
"""Launch the walker inspection tool.

    python run_inspect.py                # http://127.0.0.1:5050/inspect
    python run_inspect.py --port 8000

This is a DEBUGGING tool, not the product. The product is the JSON
snapshot pushed to Finviz (dilution/finviz_payload.py +
scripts/push_finviz.py); this serves the raw truth behind it — every
ledger row regardless of status, full terms / outstanding / history JSON,
drawdowns, anchor-reconciliation diffs, dropped-mutation walk errors,
splits, walk state, and the filing index with a raw-markdown viewer.

Binds to loopback only. It dumps internal pipeline state and raw filing
text, so reach it over an SSH tunnel rather than opening a port:

    ssh -L 5050:127.0.0.1:5050 user@vps
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DASHBOARD_LOG_PATH, setup_logging

setup_logging(DASHBOARD_LOG_PATH)

from flask import Flask  # noqa: E402

from dashboard.inspect import inspect_bp  # noqa: E402
from dilution.ledger.cards import _edgar_url  # noqa: E402

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "dashboard" / "templates"),
    static_folder=str(Path(__file__).parent / "dashboard" / "static"),
)
app.register_blueprint(inspect_bp)

# The only Jinja helper the inspect templates use (8 call sites in
# inspect.html, 1 in inspect_raw.html). The product dashboard's
# shares/usd/pct filters went with it — they were never referenced here.
app.jinja_env.globals["edgar_url"] = _edgar_url


def main() -> None:
    ap = argparse.ArgumentParser(description="Walker inspection tool")
    # Loopback default, unlike the old dashboard: this is now the only
    # HTTP surface in the repo and it is for one operator, not the public.
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
