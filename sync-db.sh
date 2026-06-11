#!/bin/bash
# Push the freshly-built dilution.db from this machine to the VPS and
# bounce the dashboard so SQLite reopens the new file.
#
# Usage: ./sync-db.sh user@your-vps

set -e

if [ -z "$1" ]; then
    echo "Usage: ./sync-db.sh user@your-vps"
    exit 1
fi

VPS=$1
REMOTE_DIR="/opt/dilution"

cd "$(dirname "$0")"

# Flush WAL into the main DB file so the rsync'd snapshot is consistent
# (otherwise uncommitted pages stay in dilution.db-wal and don't travel).
echo "Checkpointing WAL..."
sqlite3 dilution.db "PRAGMA wal_checkpoint(TRUNCATE);" > /dev/null

echo "Rsyncing dilution.db -> $VPS:$REMOTE_DIR/ ..."
rsync -avz --progress dilution.db "$VPS:$REMOTE_DIR/"

echo "Bouncing dashboard on VPS..."
ssh "$VPS" << 'REMOTE'
cd /opt/dilution
pkill -f run_dashboard.py || true
sleep 1
source .venv/bin/activate
nohup python run_dashboard.py --host 0.0.0.0 --port 5050 > server.log 2>&1 &
disown
echo "Dashboard restarted, pid $!"
REMOTE

echo ""
echo "Done."
