#!/usr/bin/env bash
# Run the full dilution pipeline against every ticker that has an eval fixture.
# Runs $PARALLEL tickers concurrently (default 5), interleaving their output
# with a [TICKER] prefix on each line and capturing each ticker's full
# stdout/stderr to logs/eval_pipeline_<TICKER>.log so failures are debuggable
# without terminal scrollback. Continues past failures so one bad ticker
# doesn't abort the batch. Tickers are scheduled longest-job-first (LPT) by
# filing count (from dilution.db) so the slowest ticker starts in the first
# wave and the makespan is shortest.
#
# Override parallelism with: PARALLEL=3 scripts/run_eval_pipeline.sh
#
# SEC's hard limit is 10 req/s globally; edgartools' throttle is per-process,
# so we cap each child at EDGAR_RATE_LIMIT_PER_SEC=2 → 5 procs * 2 = 10 req/s
# aggregate, right at SEC's edge. Tripping 429 results in a ~10-min IP ban,
# so prefer to keep PARALLEL * EDGAR_RATE_LIMIT_PER_SEC <= 10.

set -u -o pipefail
cd "$(dirname "$0")/.."

PARALLEL="${PARALLEL:-5}"
export EDGAR_RATE_LIMIT_PER_SEC="${EDGAR_RATE_LIMIT_PER_SEC:-2}"

mkdir -p logs

if [[ -t 1 ]]; then
    BOLD=$'\e[1m'; DIM=$'\e[2m'; RED=$'\e[31m'; GREEN=$'\e[32m'
    YELLOW=$'\e[33m'; BLUE=$'\e[34m'; RESET=$'\e[0m'
else
    BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; RESET=""
fi

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { printf "%s[%s]%s %s\n" "$DIM" "$(ts)" "$RESET" "$*"; }

# LPT scheduling: order tickers by filing count desc so the longest job
# enters the first wave and finishes last (this minimizes makespan when
# job sizes vary). Counts come from dilution_filings if the DB has been
# populated; tickers without DB rows fall back to fixture card-count *
# 30 (rough proxy — keeps unmeasured tickers in the same scale as
# measured ones rather than sinking to last). Ties broken alphabetically.
read -r -d '' _ORDER_PY <<'PY' || true
import json, sqlite3, sys
from pathlib import Path

fixtures = {p.stem: p for p in Path("evals").glob("*.json")}
if not fixtures:
    sys.exit("no fixtures")

counts = {}
db_path = Path("dilution.db")
if db_path.exists():
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        placeholders = ",".join("?" * len(fixtures))
        rows = conn.execute(
            "SELECT c.ticker, COUNT(*) AS n "
            "FROM dilution_filings f JOIN dilution_company c USING(cik) "
            f"WHERE c.ticker IN ({placeholders}) GROUP BY c.ticker",
            list(fixtures),
        ).fetchall()
        counts = {t: n for t, n in rows}
        conn.close()
    except sqlite3.Error:
        pass

def card_count(p):
    try:
        d = json.loads(p.read_text())
        return sum(len(v) for v in (d.get("cards") or {}).values())
    except Exception:
        return 0

def weight(t):
    if t in counts:
        return counts[t], "filings"
    return card_count(fixtures[t]) * 30, "cards*30"

ranked = sorted(fixtures, key=lambda t: (-weight(t)[0], t))
for t in ranked:
    w, src = weight(t)
    print(f"{t}\t{w}\t{src}")
PY

ranking=$(python3 -c "$_ORDER_PY") || {
    log "${RED}Failed to compute ticker ordering${RESET}"
    exit 1
}

tickers=()
declare -A ticker_weight
declare -A ticker_wsource
while IFS=$'\t' read -r t w src; do
    [[ -z "$t" ]] && continue
    tickers+=("$t")
    ticker_weight["$t"]="$w"
    ticker_wsource["$t"]="$src"
done <<< "$ranking"

total=${#tickers[@]}
if (( total == 0 )); then
    log "${RED}No eval fixtures found in evals/*.json${RESET}"
    exit 1
fi

results_dir=$(mktemp -d)
trap 'rm -rf "$results_dir"' EXIT

run_one() {
    local t="$1"
    local t_start=$SECONDS
    local rc=0
    local logfile="logs/eval_pipeline_${t}.log"
    log "${BOLD}${BLUE}▶ starting${RESET} $t (weight=${ticker_weight[$t]} ${ticker_wsource[$t]}, log: $logfile)"
    # python -u keeps output line-buffered; tee fans out to a per-ticker
    # log file (so post-mortems don't depend on terminal scrollback);
    # sed -u prefixes each live line with [TICKER]. pipefail makes the
    # pipeline's exit code = python's, since tee/sed never fail.
    if python -u run_dilution.py "$t" 2>&1 \
            | tee "$logfile" \
            | sed -u "s/^/${BLUE}[$t]${RESET} /"; then
        rc=0
    else
        rc=$?
    fi
    local elapsed=$((SECONDS - t_start))
    if (( rc == 0 )); then
        log "${GREEN}✓ $t finished in ${elapsed}s${RESET}"
        printf "OK\t%s\t%s\n" "$t" "$elapsed" > "$results_dir/$t"
    else
        log "${RED}✗ $t failed (exit $rc) after ${elapsed}s — see $logfile${RESET}"
        printf "FAIL\t%s\t%s\t%s\n" "$t" "$rc" "$elapsed" > "$results_dir/$t"
    fi
}

log "${BOLD}Running dilution pipeline for $total tickers (parallel=$PARALLEL, edgar_rps=$EDGAR_RATE_LIMIT_PER_SEC):${RESET}"
log "${BOLD}LPT order (longest first):${RESET}"
for t in "${tickers[@]}"; do
    log "  $t  weight=${ticker_weight[$t]}  (${ticker_wsource[$t]})"
done
batch_start=$SECONDS

running=0
for t in "${tickers[@]}"; do
    if (( running >= PARALLEL )); then
        wait -n
        running=$((running - 1))
    fi
    run_one "$t" &
    running=$((running + 1))
done
wait

batch_elapsed=$((SECONDS - batch_start))
mins=$((batch_elapsed / 60))
secs=$((batch_elapsed % 60))

declare -a ok_list=()
declare -a failed_list=()
for t in "${tickers[@]}"; do
    if [[ ! -s "$results_dir/$t" ]]; then
        failed_list+=("$t (no result file)")
        continue
    fi
    IFS=$'\t' read -r status _name f1 f2 < "$results_dir/$t"
    if [[ "$status" == "OK" ]]; then
        ok_list+=("$t (${f1}s)")
    else
        failed_list+=("$t (exit $f1, ${f2}s)")
    fi
done

printf "\n%s%s=== Summary ===%s\n" "$BOLD" "$BLUE" "$RESET"
log "total runtime: ${mins}m ${secs}s (parallel=$PARALLEL)"
log "${GREEN}succeeded: ${#ok_list[@]}/${total}${RESET}"
for s in "${ok_list[@]}"; do printf "  %s✓%s %s\n" "$GREEN" "$RESET" "$s"; done

if (( ${#failed_list[@]} )); then
    log "${RED}failed: ${#failed_list[@]}/${total}${RESET}"
    for s in "${failed_list[@]}"; do printf "  %s✗%s %s\n" "$RED" "$RESET" "$s"; done
    exit 1
fi

log "${GREEN}${BOLD}all tickers completed successfully${RESET}"
