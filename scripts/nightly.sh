#!/usr/bin/env bash
# The nightly dilution service: one run_dilution.py pipeline per ticker —
# walk new filings, rebuild the snapshot, publish it if it changed.
#
#   scripts/nightly.sh              # full run
#   scripts/nightly.sh --dry-run    # walk + build, but publish nothing
#
# There is no separate brief or publish step: each run_dilution.py walks
# its ticker, rebuilds the snapshot (regenerating the AI brief inline
# when the walk mutated the ledger — finviz_payload §8), validates it
# locally, digest-compares it against what Finviz already holds, and
# POSTs only when the content changed. --dry-run propagates to every
# child: everything except the POST, so a malformed payload still
# surfaces in the log the night it appears.
#
# Exit codes: 0 all tickers clean; 1 a ticker failed (see the log).

set -u -o pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
export PYTHON

PARALLEL="${PARALLEL:-4}"
export PARALLEL

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y-%m-%d)
LOG_FILE="$LOG_DIR/nightly_${STAMP}.log"

DRY_RUN=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run" ;;
        *) echo "unknown argument: $arg" >&2; exit 64 ;;
    esac
done
export DRY_RUN

# A run can outlast the timer interval (a cold walk over the universe is
# hours). Overlapping runs would have two processes mutating one SQLite
# ledger and racing on per-ticker pushes, so a second invocation exits
# immediately rather than queueing.
LOCK_FILE="${LOCK_FILE:-/tmp/dilution-nightly.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "$(date -Is) another nightly run holds $LOCK_FILE — exiting" \
        | tee -a "$LOG_FILE"
    exit 0
fi

log() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG_FILE"; }

rc=0
started=$SECONDS

log "=== nightly start (parallel=$PARALLEL dry_run=${DRY_RUN:-no}) ==="

# The universe is tickers.txt (one ticker per line, '#' comments) — an
# explicit list, not whatever accumulated in dilution.db. Each ticker is
# one full run_dilution.py pipeline (walk → build → publish-if-changed),
# $PARALLEL at a time. Each child is capped at EDGAR_RATE_LIMIT_PER_SEC
# (the edgartools throttle is per-process) so the aggregate stays under
# SEC's 10 req/s edge — tripping it is a ~10-min IP ban.
# A failed ticker doesn't stop the batch: it lands in $FAILED and the
# run continues, so the survivors still get published. A failed ticker's
# last-known-good snapshot simply stays live on Finviz.
export EDGAR_RATE_LIMIT_PER_SEC="${EDGAR_RATE_LIMIT_PER_SEC:-2}"
FAILED=$(mktemp)
export FAILED
log "walking + publishing tickers.txt"
sed 's/#.*//' tickers.txt | tr '[:lower:]' '[:upper:]' | grep -oE '[A-Z0-9.-]+' \
    | xargs -P "$PARALLEL" -I{} -- sh -c \
        '"$PYTHON" -u run_dilution.py "$1" $DRY_RUN >"logs/walk_$1.log" 2>&1 \
            || echo "$1" >>"$FAILED"' _ {}
if [[ -s "$FAILED" ]]; then
    rc=1
    log "failed for: $(xargs <"$FAILED") — see logs/walk_<TICKER>.log"
fi
rm -f "$FAILED"

elapsed=$((SECONDS - started))
log "=== nightly done in $((elapsed / 60))m $((elapsed % 60))s (exit $rc) ==="
exit "$rc"
