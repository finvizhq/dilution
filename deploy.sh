#!/bin/bash
# Set up the dilution dashboard on a VPS.
# Run this from /opt/dilution on the VPS after rsyncing the repo:
#
#   rsync -avz --progress \
#     --exclude='*.bak-*' --exclude='__pycache__' --exclude='.git' \
#     --exclude='walker_dumps' --exclude='logs' --exclude='evals' \
#     --exclude='knowledge' --exclude='.venv' --exclude='*.log' \
#     /home/peter/finviz/dilution/ user@vps:/opt/dilution/
#
#   ssh user@vps
#   cd /opt/dilution && ./deploy.sh
#
# Prerequisites on VPS: python3, python3-venv

set -e

cd "$(dirname "$0")"

# Create venv if needed
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt -q

echo ""
echo "=== Setup complete ==="
echo ""
echo "Start the dashboard (bound to all interfaces on port 5050):"
echo "  cd /opt/dilution && source .venv/bin/activate"
echo "  nohup python run_dashboard.py --host 0.0.0.0 --port 5050 > server.log 2>&1 &"
echo ""
echo "Then open http://<vps-ip>:5050/"
echo ""
echo "Note: per-ticker pages call the Finviz Elite API, so .env must contain"
echo "      FINVIZ_API_KEY=... for fundamentals to load."
