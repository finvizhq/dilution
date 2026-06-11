#!/usr/bin/env python3
"""Single-ticker dilution tracker — cards-only build.

End-to-end pipeline: resolve ticker → CIK, pull SEC filing index, detect
unit context (FPI / ADS ratio), pull split history, fetch filing text,
walk the event stream with the LLM extractor.

Usage:
    python run_dilution.py MULN              # 6-year default window
    python run_dilution.py MULN --years 3
    python run_dilution.py MULN --force      # re-walk from scratch
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from config import PIPELINE_LOG_PATH, set_log_ticker, setup_logging
from dilution.company import ensure_company, get_unit_context
from dilution.fetch_raw import fetch_extractable_for_cik
from dilution.filings import pull_filing_index
from dilution.ledger.walker import walk_ticker
from dilution.observability import (
    flush_observability,
    pipeline_session,
    setup_observability,
)
from dilution.splits import fetch_and_persist_splits
from dilution.unit_detection import populate_company_unit

setup_logging(PIPELINE_LOG_PATH)
log = logging.getLogger("run_dilution")


def _since(years: int) -> str:
    return (date.today() - timedelta(days=365 * years + 1)).isoformat()


def main():
    ap = argparse.ArgumentParser(description="Single-ticker SEC dilution tracker")
    ap.add_argument("ticker", help="e.g. MULN")
    ap.add_argument("--years", type=int, default=6,
                    help="history window in years (default 6)")
    ap.add_argument("--concurrency", type=int, default=config.LLM_CONCURRENCY,
                    help=f"max concurrent LLM calls (default {config.LLM_CONCURRENCY})")
    ap.add_argument("--force", action="store_true",
                    help="drop ledger and re-walk every filing from scratch")
    args = ap.parse_args()

    set_log_ticker(args.ticker)
    setup_observability()
    since = _since(args.years)
    log.info("Dilution tracker — ticker=%s since=%s "
             "llm_provider=%s llm_model=%s llm_model_periodic=%s",
             args.ticker, since, config.LLM_PROVIDER,
             config.LLM_MODEL, config.LLM_MODEL_PERIODIC)

    try:
        with pipeline_session(
            args.ticker,
            metadata={
                "years": args.years,
                "concurrency": args.concurrency,
                "force": args.force,
                "llm_provider": config.LLM_PROVIDER,
                "llm_model": config.LLM_MODEL,
                "llm_model_periodic": config.LLM_MODEL_PERIODIC,
            },
        ):
            company = ensure_company(args.ticker)
            set_log_ticker(company["ticker"])
            cik = company["cik"]

            pull_filing_index(cik, since_date=since)

            ctx = populate_company_unit(cik)
            log.info("  unit ctx — is_fpi=%s ads_ratio=%s reporting_unit=%s",
                     ctx["is_fpi"], ctx["ads_ratio"], ctx["reporting_unit"])

            unit_ctx_for_splits = get_unit_context(cik)
            events = fetch_and_persist_splits(
                cik=cik, ticker=company["ticker"],
                is_fpi=bool(unit_ctx_for_splits.get("is_fpi")),
            )
            log.info("  splits — %d events persisted", len(events))

            fetch_extractable_for_cik(
                cik, since_date=since, concurrency=args.concurrency)

            summary = walk_ticker(
                cik=cik, ticker=company["ticker"], since_date=since,
                force=args.force, concurrency=args.concurrency,
            )
            log.info(
                "  walk done — seed=%s walked=%d skipped=%d "
                "(resale=%d empty8k=%d) "
                "applied=%d rejected=%d created=%d drawdowns=%d "
                "anchor_diffs=%d errors=%d",
                summary.seed_case, summary.walked, summary.skipped,
                summary.skipped_resale, summary.skipped_no_dilution,
                summary.mutations_applied, summary.mutations_rejected,
                summary.instruments_created, summary.drawdowns_recorded,
                summary.anchor_diffs, summary.errors,
            )
    finally:
        flush_observability()


if __name__ == "__main__":
    main()
