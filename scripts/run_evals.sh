#!/usr/bin/env bash
# Run run_eval.py against every fixture in evals/*.json.
#
# Pure read-only — compares the CURRENT ledger state in the DB against
# the hand-transcribed DT screenshots, without re-walking anything. Use
# scripts/run_eval_pipeline.sh first when you want fresh data; this
# script just scores what's already there.
#
# Output: per-ticker section + a final PASS/FAIL summary line. Per-ticker
# exit codes are collected but the script always exits 0 so a single bad
# fixture doesn't mask the rest.

set -u -o pipefail
cd "$(dirname "$0")/.."

if [[ -t 1 ]]; then
    BOLD=$'\e[1m'; DIM=$'\e[2m'; RED=$'\e[31m'; GREEN=$'\e[32m'
    BLUE=$'\e[34m'; RESET=$'\e[0m'
else
    BOLD=""; DIM=""; RED=""; GREEN=""; BLUE=""; RESET=""
fi

tickers=()
for f in evals/*.json; do
    [[ -e "$f" ]] || continue
    t=$(basename "$f" .json)
    tickers+=("$t")
done

if (( ${#tickers[@]} == 0 )); then
    echo "${RED}no fixtures found in evals/*.json${RESET}" >&2
    exit 1
fi

pass=0
fail=0
for t in "${tickers[@]}"; do
    printf "\n%s=== %s ===%s\n" "$BOLD$BLUE" "$t" "$RESET"
    if python run_eval.py "$t"; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
    fi
done

total=${#tickers[@]}
printf "\n%s=== Summary ===%s  %s%d/%d PASS%s" \
    "$BOLD" "$RESET" "$GREEN" "$pass" "$total" "$RESET"
if (( fail > 0 )); then
    printf "  %s%d FAIL%s" "$RED" "$fail" "$RESET"
fi
printf "\n"
