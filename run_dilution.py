#!/usr/bin/env python3
"""Single-ticker dilution tracker — cards-only build.

Runs the bare pipeline that feeds the DT-style instrument cards:

Usage:
    python run_dilution.py MULN                     # full pipeline, 6y
    python run_dilution.py MULN --years 3
    python run_dilution.py MULN --stage extract --limit 5
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from config import PIPELINE_LOG_PATH, set_log_ticker, setup_logging
from dilution.company import ensure_company, get_company_by_ticker
from dilution.fetch_raw import fetch_extractable_for_cik
from dilution.filings import pull_filing_index
from dilution.ledger.walker import walk_ticker
from dilution.observability import (
    flush_observability,
    pipeline_session,
    setup_observability,
    stage,
)
from dilution.unit_detection import populate_company_unit

setup_logging(PIPELINE_LOG_PATH)
log = logging.getLogger("run_dilution")

STAGES = ("resolve", "index", "unit", "fetch", "walk", "all")


def _since(years: int) -> str:
    return (date.today() - timedelta(days=365 * years + 1)).isoformat()


def main():
    ap = argparse.ArgumentParser(description="Single-ticker SEC dilution tracker")
    ap.add_argument("ticker", help="e.g. MULN")
    ap.add_argument("--stage", choices=STAGES, default="all")
    ap.add_argument("--years", type=int, default=6,
                    help="history window in years (default 6)")
    ap.add_argument("--limit", type=int, default=None,
                    help="limit filings processed (for extract, debugging)")
    ap.add_argument("--concurrency", type=int, default=config.LLM_CONCURRENCY,
                    help=f"max concurrent xAI calls in extract/overhang "
                         f"(default {config.LLM_CONCURRENCY}, set 1 for serial)")
    ap.add_argument("--force", action="store_true",
                    help="re-extract every filing even when its model/"
                         "handler version matches the current pipeline. "
                         "Use after card-layer or schema fixes when you "
                         "want a fresh extraction without bumping "
                         "STAGE1_VERSION/STAGE2_VERSION.")
    args = ap.parse_args()

    set_log_ticker(args.ticker)
    setup_observability()
    since = _since(args.years)
    log.info("Dilution tracker — ticker=%s stage=%s events_since=%s",
             args.ticker, args.stage, since)

    stages = STAGES[:-1] if args.stage == "all" else [args.stage]

    try:
        with pipeline_session(
            args.ticker,
            metadata={
                "stage": args.stage,
                "years": args.years,
                "concurrency": args.concurrency,
                "force": args.force,
                "llm_provider": config.LLM_PROVIDER,
                "llm_model": config.LLM_MODEL,
            },
        ):
            if "resolve" in stages or args.stage == "all":
                company = ensure_company(args.ticker)
            else:
                company = (get_company_by_ticker(args.ticker)
                           or ensure_company(args.ticker))
            set_log_ticker(company["ticker"])
            cik = company["cik"]

            if "index" in stages:
                log.info("STAGE index")
                with stage("index", input={"cik": cik, "since": since}):
                    pull_filing_index(cik, since_date=since)

            if "unit" in stages:
                log.info("STAGE unit")
                with stage("unit", input={"cik": cik}):
                    ctx = populate_company_unit(cik)
                log.info("  unit ctx — is_fpi=%s ads_ratio=%s reporting_unit=%s",
                         ctx["is_fpi"], ctx["ads_ratio"], ctx["reporting_unit"])

            if "fetch" in stages:
                log.info("STAGE fetch")
                with stage("fetch", input={"cik": cik, "since": since}):
                    fetch_extractable_for_cik(
                        cik, since_date=since,
                        concurrency=args.concurrency)

            if "walk" in stages:
                log.info("STAGE walk")
                with stage("walk", input={
                        "cik": cik, "since": since, "force": args.force}):
                    summary = walk_ticker(
                        cik=cik, ticker=company["ticker"], since_date=since,
                        force=args.force, concurrency=args.concurrency,
                    )
                log.info(
                    "  walk done — seed=%s walked=%d skipped=%d "
                    "applied=%d rejected=%d created=%d drawdowns=%d "
                    "anchor_diffs=%d errors=%d",
                    summary.seed_case, summary.walked, summary.skipped,
                    summary.mutations_applied, summary.mutations_rejected,
                    summary.instruments_created, summary.drawdowns_recorded,
                    summary.anchor_diffs, summary.errors,
                )
    finally:
        flush_observability()


if __name__ == "__main__":
    main()
