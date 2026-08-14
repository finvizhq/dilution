#!/usr/bin/env bash
# The nightly dilution service: walk new filings, refresh briefs, publish
# whatever changed to Finviz.
#
#   scripts/nightly.sh              # full run
#   scripts/nightly.sh --dry-run    # walk + briefs, but publish nothing
#
# Ordering is deliberate and is the reason this wrapper exists rather than
# just enabling per-walk auto-push:
#
#   1. walk every tracked ticker      (run_dilution.py --no-push)
#   2. refresh stale briefs           (scripts/run_brief_all.py)
#   3. publish what changed           (scripts/push_finviz.py --all)
#
# run_dilution.py can publish a ticker itself at the end of its walk, and
# for an ad-hoc single-ticker run that is what you want. But
# run_brief_all.py regenerates exactly the briefs whose ticker just
# received new filings, so a per-walk push would ship the OLD brief
# alongside the NEW cards and the fresh prose would wait a full day for
# the next run. Hence --no-push in step 1 and one publish pass at the end.
#
# Step 3 is cheap despite covering the whole universe: push_finviz.py
# digest-compares each build against what Finviz already holds and POSTs
# only the differences.
#
# Exit codes: 0 all steps clean; 1 a step failed (see the log); 2 the
# blast-radius gate refused to publish (an implausible share of the
# universe changed — inspect before forcing with --yes).

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

# ── 1. walk ──────────────────────────────────────────────────────────
# ALL_TRACKED reads the universe from dilution_company rather than a
# hardcoded list. --no-push: publishing happens in step 3.
log "step 1/3 — walking all tracked tickers"
if ALL_TRACKED=1 scripts/run_open_access.sh --no-push >>"$LOG_FILE" 2>&1; then
    log "step 1/3 — walk OK"
else
    # Non-fatal by design: run_open_access.sh already continues past a
    # single bad ticker, so a non-zero code means "some ticker failed",
    # not "nothing was walked". The tickers that did succeed should still
    # get their briefs refreshed and published.
    rc=1
    log "step 1/3 — walk reported failures (continuing; see log)"
fi

# ── 2. briefs ────────────────────────────────────────────────────────
# Skips any ticker whose cached brief already postdates its latest filing,
# so this only pays for tickers step 1 actually changed.
log "step 2/3 — refreshing stale briefs"
if "$PYTHON" scripts/run_brief_all.py >>"$LOG_FILE" 2>&1; then
    log "step 2/3 — briefs OK"
else
    rc=1
    log "step 2/3 — brief refresh reported failures (continuing)"
fi

# ── 3. publish ───────────────────────────────────────────────────────
log "step 3/3 — publishing changed snapshots"
set +e
"$PYTHON" scripts/push_finviz.py --all $DRY_RUN >>"$LOG_FILE" 2>&1
push_rc=$?
set -e
case "$push_rc" in
    0) log "step 3/3 — publish OK" ;;
    2) rc=2
       log "step 3/3 — REFUSED by the blast-radius gate: an implausible" \
          "share of the universe changed. Nothing was published." ;;
    *) rc=1
       log "step 3/3 — publish reported failures (exit $push_rc)" ;;
esac

elapsed=$((SECONDS - started))
log "=== nightly done in $((elapsed / 60))m $((elapsed % 60))s (exit $rc) ==="
exit "$rc"
