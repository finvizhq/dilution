#!/bin/bash
# Set up the dilution backend service on a VPS.
# Run from /opt/dilution after rsyncing the repo:
#
#   rsync -avz --progress \
#     --exclude='*.bak-*' --exclude='__pycache__' --exclude='.git' \
#     --exclude='walker_dumps' --exclude='logs' --exclude='evals' \
#     --exclude='knowledge' --exclude='.venv' --exclude='*.log' \
#     --exclude='dilution.db' \
#     /home/peter/finviz/dilution/ user@vps:/opt/dilution/
#
#   ssh user@vps
#   cd /opt/dilution && ./deploy.sh
#
# There is no web server any more. This installs a venv and a nightly
# systemd timer that walks EDGAR, refreshes briefs, and pushes changed
# snapshots to Finviz. See DEPLOY.md for the full procedure, including the
# one-time database seed.
#
# Prerequisites on the VPS: python3, python3-venv, sqlite3, flock (util-linux).

set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Creating venv..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt -q
mkdir -p logs
chmod +x scripts/nightly.sh

echo ""
echo "=== Environment check ==="

fail=0
check_env() {
    # Read from .env without sourcing it (values may contain characters
    # that a shell would interpret).
    if grep -qE "^$1=.+" .env 2>/dev/null; then
        echo "  ok      $1 is set"
    else
        echo "  MISSING $1 — $2"
        fail=1
    fi
}

if [ ! -f .env ]; then
    echo "  MISSING .env entirely. Create it (chmod 600) with the keys below."
    fail=1
else
    check_env FINVIZ_INGEST_TOKEN "write credential for POST /api/dilution/set"
    check_env FINVIZ_API_KEY      "Elite /export read key (market data)"
    check_env OPENAI_API_KEY      "walker LLM key"
fi

if [ ! -f dilution.db ]; then
    echo "  MISSING dilution.db — the pipeline has no ledger to build from."
    echo "          A cold start would re-walk every ticker over the whole"
    echo "          config.HISTORY_YEARS window of"
    echo "          filings and re-pay the entire historical LLM cost. Seed it"
    echo "          once from a machine that already has one (DEPLOY.md step 3)."
    fail=1
else
    tables=$(sqlite3 dilution.db \
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name LIKE 'dilution_%';" \
        2>/dev/null || echo 0)
    rows=$(sqlite3 dilution.db \
        "SELECT COUNT(*) FROM dilution_company;" 2>/dev/null || echo 0)
    echo "  ok      dilution.db present ($tables tables, $rows tracked tickers)"
fi

echo ""
echo "=== Self-check: build and validate a snapshot, send nothing ==="
if [ "$fail" -eq 0 ]; then
    ticker=$(sqlite3 dilution.db \
        "SELECT ticker FROM dilution_company WHERE ticker IS NOT NULL LIMIT 1;")
    if [ -n "$ticker" ]; then
        # --dry-run validates the document and change-checks it against
        # what Finviz holds without POSTing. A POST is a destructive full
        # replace, so this never writes.
        python scripts/push_finviz.py "$ticker" --dry-run || {
            echo "  self-check FAILED for $ticker — do not enable the timer yet"
            fail=1
        }
    fi
else
    echo "  skipped (fix the environment problems above first)"
fi

echo ""
if [ "$fail" -ne 0 ]; then
    echo "=== Setup INCOMPLETE — resolve the items above ==="
    exit 1
fi

echo "=== Setup complete ==="
echo ""
echo "Install the nightly timer (needs sudo):"
echo "  sudo cp deploy/dilution-nightly.service deploy/dilution-nightly.timer \\"
echo "       /etc/systemd/system/"
echo "  sudo sed -i \"s/^User=CHANGEME/User=\$USER/\" \\"
echo "       /etc/systemd/system/dilution-nightly.service"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable --now dilution-nightly.timer"
echo ""
echo "Verify before trusting the schedule:"
echo "  systemctl list-timers dilution-nightly"
echo "  ./scripts/nightly.sh --dry-run     # full run, publishes nothing"
echo ""
echo "Watch a real run:"
echo "  journalctl -u dilution-nightly -f"
echo "  tail -f logs/nightly_\$(date +%F).log"
echo ""
echo "Confirm what Finviz actually holds:"
echo "  python scripts/dump_finviz_payload.py <TICKER> --live --stdout"
