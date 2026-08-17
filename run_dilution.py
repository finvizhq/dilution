#!/usr/bin/env python3
"""Single-ticker dilution tracker — cards-only build.

End-to-end pipeline: resolve ticker → CIK, pull SEC filing index, detect
unit context (FPI / ADS ratio), pull split history, fetch filing text,
walk the event stream with the LLM extractor.

Usage:
    python run_dilution.py MULN              # config.HISTORY_YEARS window
    python run_dilution.py MULN --force      # re-walk from scratch
    python run_dilution.py MULN --dry-run    # walk + build, send nothing
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
from dilution.finviz_payload import build_payload
from dilution.finviz_push import push_snapshot
from dilution.ledger.walker import walk_ticker
from dilution.openai_client import cache_stats
from dilution.observability import (
    flush_observability,
    pipeline_session,
    setup_observability,
)
from dilution.splits import fetch_and_persist_splits
from dilution.unit_detection import populate_company_unit

setup_logging(PIPELINE_LOG_PATH)
log = logging.getLogger("run_dilution")


def _since() -> str:
    """Start of the filing-history window. Length is a pipeline-wide
    constant (config.HISTORY_YEARS), not a per-run choice: the window
    determines which filings the seed anchors on, so varying it between
    walks of the same ticker makes two ledgers that cannot be compared."""
    return (date.today()
            - timedelta(days=365 * config.HISTORY_YEARS + 1)).isoformat()


def _publish(ticker: str, *, dry_run: bool) -> bool:
    """Build and publish this ticker's snapshot. Returns False on failure.

    Deliberately swallows its own exceptions: the walk has already
    committed the ledger, which is the expensive part of this run, and a
    publish problem must not present as a failed walk. The caller turns a
    False into a non-zero exit so a nightly job still notices.

    A skipped push still pays for one build — you cannot know whether
    content changed without building it — but that is cheap next to a walk.
    """
    try:
        result = push_snapshot(build_payload(ticker), dry_run=dry_run)
    except Exception:
        log.exception("  push — FAILED to publish %s", ticker)
        return False
    log.info("  push — %s: %s", result.status, result.reason)
    return result.ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Single-ticker SEC dilution tracker")
    ap.add_argument("ticker", help="e.g. MULN")
    ap.add_argument("--concurrency", type=int, default=config.LLM_CONCURRENCY,
                    help=f"max concurrent LLM calls (default {config.LLM_CONCURRENCY})")
    ap.add_argument("--force", action="store_true",
                    help="drop ledger and re-walk every filing from scratch")
    ap.add_argument("--fresh-llm", action="store_true",
                    help="bypass the local LLM response cache and re-sample "
                         "every call. Required when measuring walk-to-walk "
                         "drift — a cache hit replays the previous answer, "
                         "which is exactly what a drift check must not do")
    ap.add_argument("--dry-run", action="store_true",
                    help="do everything except the POST: build the snapshot, "
                         "validate the envelope, read back what Finviz "
                         "currently holds and report whether the content "
                         "changed. Nothing is sent. The nightly job runs this "
                         "way and publishes in one pass at the end, after "
                         "briefs are refreshed")
    args = ap.parse_args()

    set_log_ticker(args.ticker)
    if args.fresh_llm:
        config.LLM_CACHE_ENABLED = False
    setup_observability()
    since = _since()
    log.info("Dilution tracker — ticker=%s since=%s "
             "llm_model=%s llm_model_periodic=%s reasoning=%s tier=%s cache=%s",
             args.ticker, since,
             config.LLM_MODEL, config.LLM_MODEL_PERIODIC,
             config.OPENAI_REASONING_EFFORT, config.OPENAI_SERVICE_TIER,
             "on" if config.LLM_CACHE_ENABLED else "off")

    # A dry run still sets this from the build+validate result, so a
    # malformed payload is reported even when nothing is sent.
    published = True

    try:
        with pipeline_session(
            args.ticker,
            metadata={
                "years": config.HISTORY_YEARS,
                "concurrency": args.concurrency,
                "force": args.force,
                "llm_model": config.LLM_MODEL,
                "llm_model_periodic": config.LLM_MODEL_PERIODIC,
                "reasoning_effort": config.OPENAI_REASONING_EFFORT,
                "service_tier": config.OPENAI_SERVICE_TIER,
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

            stats = cache_stats()
            if stats["hit"] or stats["miss"]:
                total = stats["hit"] + stats["miss"]
                log.info("  llm cache — %d hit / %d miss (%.0f%% replayed)",
                         stats["hit"], stats["miss"],
                         stats["hit"] / total * 100)

            published = _publish(company["ticker"], dry_run=args.dry_run)
    finally:
        flush_observability()

    # Non-zero on a publish failure only — the walk itself succeeded and
    # its ledger writes stand, but a nightly job needs to see that the
    # snapshot did not reach Finviz.
    return 0 if published else 1


if __name__ == "__main__":
    raise SystemExit(main())
