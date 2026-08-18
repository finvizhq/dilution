#!/usr/bin/env python3
"""Publish dilution snapshots to the Finviz ingest API.

    python scripts/push_finviz.py CELU                # one ticker
    python scripts/push_finviz.py CELU GCTK XTIA      # several
    python scripts/push_finviz.py --all --dry-run     # validate, send nothing
    python scripts/push_finviz.py --all               # publish what changed
    python scripts/push_finviz.py CELU --force-push   # skip the change check

Only tickers whose content actually changed are sent: push_snapshot
digest-compares each build against what Finviz currently holds
(read-back GET), so a walk that changed nothing costs one GET and no
POST. See dilution/finviz_push.content_digest for what counts as a
change — note `as_of` (the settled-close date) is deliberately part of
it, so the first push of a new trading day always goes out.

One safety behavior worth knowing before running this against prod: a
POST is a DESTRUCTIVE full replace and the server does not validate
`data` (FINVIZ_API_CONTRACT.md §3.5), so every document is validated
locally first and a bad build is skipped rather than sent.

Needs a walked ledger (dilution.db), FINVIZ_INGEST_TOKEN, and network:
Finviz Elite for market data and SEC XBRL for cash / shares history.
A build also regenerates the AI brief inline (one LLM call, needs
OPENAI_API_KEY) when the ledger mutated since the prose was written —
on failure the build degrades to the cached prose rather than failing.
To force-regenerate all prose: `sqlite3 dilution.db "DELETE FROM
dilution_ticker_brief"` and re-push.
"""
import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                            # noqa: E402,F401
from dilution.finviz_payload import (                     # noqa: E402
    all_tracked_tickers,
    build_payload,
    payload_summary,
)
from dilution.finviz_push import (                        # noqa: E402
    FinvizPushError,
    PushResult,
    push_snapshot,
)
from dilution.observability import (                      # noqa: E402
    flush_observability,
    setup_observability,
)

log = logging.getLogger("push_finviz")

# §10 caps the producer at 8 concurrent requests during a burst.
MAX_CONCURRENCY = 8


def _build(ticker: str) -> dict | None:
    try:
        return build_payload(ticker)
    except Exception as exc:
        log.error("%s: build failed — %s", ticker, exc)
        return None


def _push_one(doc: dict, *, if_changed: bool, dry_run: bool,
              allow_empty: bool) -> PushResult:
    try:
        return push_snapshot(doc, if_changed=if_changed, dry_run=dry_run,
                             allow_empty=allow_empty)
    except FinvizPushError:
        raise
    except Exception as exc:                       # pragma: no cover
        log.exception("%s: unexpected push failure", doc.get("ticker"))
        return PushResult(ticker=str(doc.get("ticker")), status="failed",
                          reason=f"unexpected error: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="*", help="tickers to publish")
    ap.add_argument("--all", action="store_true",
                    help="every tracked ticker in the ledger")
    ap.add_argument("--dry-run", action="store_true",
                    help="build, validate and change-check, but send no POST")
    ap.add_argument("--force-push", action="store_true",
                    help="publish even when Finviz already holds identical "
                         "content (skips the read-back comparison)")
    ap.add_argument("--allow-empty", action="store_true",
                    help="permit a snapshot with no cards, badges or cash — "
                         "only for a genuinely no-paper issuer, since that "
                         "shape is normally a failed build")
    ap.add_argument("--concurrency", type=int, default=4,
                    help=f"parallel tickers, capped at {MAX_CONCURRENCY} "
                         f"(default 4)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s")

    tickers = [t.upper() for t in args.tickers]
    if args.all:
        tickers = all_tracked_tickers()
    if not tickers:
        ap.error("pass one or more tickers, or --all")

    concurrency = max(1, min(args.concurrency, MAX_CONCURRENCY))

    # A build can regenerate a stale brief (one LLM call) — trace it the
    # same way the walker traces its calls.
    setup_observability()

    # Build everything first, so a wide build regression is visible in
    # one place before anything is published.
    print(f"Building {len(tickers)} snapshot(s)...", file=sys.stderr)
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            docs = list(pool.map(_build, tickers))
    finally:
        flush_observability()

    built = [(t, d) for t, d in zip(tickers, docs) if d is not None]
    build_failed = [t for t, d in zip(tickers, docs) if d is None]

    # Push. push_snapshot's own read-back comparison (`if_changed`) is
    # the change gate: an unchanged ticker comes back as
    # skipped_unchanged, costing one GET and no POST.
    results: list[PushResult] = []
    if built:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(
                lambda pair: _push_one(
                    pair[1],
                    if_changed=not args.force_push,
                    dry_run=args.dry_run,
                    allow_empty=args.allow_empty),
                built))

    verb = "would publish" if args.dry_run else "published"
    by_ticker = {t: d for t, d in built}
    for result in results:
        doc = by_ticker.get(result.ticker)
        extra = f"  {payload_summary(doc)}" if doc else ""
        print(f"{result.ticker:<8} {result.status:<18} {result.reason}{extra}")
    for ticker in build_failed:
        print(f"{ticker:<8} {'build_failed':<18} see log")

    pushed = [r for r in results if r.status == "pushed"]
    skipped_unchanged = [r for r in results
                         if r.status == "skipped_unchanged"]
    invalid = [r for r in results if r.status == "skipped_invalid"]
    failed = [r for r in results if r.status == "failed"]

    print(f"\n{verb} {len(pushed)}  "
          f"unchanged {len(skipped_unchanged)}  "
          f"invalid {len(invalid)}  "
          f"failed {len(failed) + len(build_failed)}", file=sys.stderr)

    return 1 if (invalid or failed or build_failed) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FinvizPushError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(3)
