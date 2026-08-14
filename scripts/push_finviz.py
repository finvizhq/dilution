#!/usr/bin/env python3
"""Publish dilution snapshots to the Finviz ingest API.

    python scripts/push_finviz.py CELU                # one ticker
    python scripts/push_finviz.py CELU GCTK XTIA      # several
    python scripts/push_finviz.py --all --dry-run     # validate, send nothing
    python scripts/push_finviz.py --all               # publish what changed
    python scripts/push_finviz.py CELU --force-push   # skip the change check

Only tickers whose content actually changed are sent: each build is
digest-compared against what Finviz currently holds (read-back GET), so a
walk that changed nothing costs one GET and no POST. See
dilution/finviz_push.content_digest for what counts as a change.

Two safety behaviors worth knowing before running this against prod:

  * A POST is a DESTRUCTIVE full replace and the server does not validate
    `data` (FINVIZ_API_CONTRACT.md §3.5), so every document is validated
    locally first and a bad build is skipped rather than sent.
  * `--all` refuses to publish when an implausible share of the universe
    comes up changed (--max-changed), because that shape means a code or
    prompt regression reshaped every payload rather than the market
    moving. Pass --yes when a pipeline release makes it expected.

Needs a walked ledger (dilution.db), FINVIZ_INGEST_TOKEN, and network:
Finviz Elite for market data and SEC XBRL for cash / shares history.
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
    content_digest,
    fetch_snapshot,
    push_snapshot,
)

log = logging.getLogger("push_finviz")

# §10 caps the producer at 8 concurrent requests during a burst.
MAX_CONCURRENCY = 8

# Default blast-radius ceiling as a share of the universe. A daily batch
# legitimately changes many tickers (`as_of` rolls for everything on a new
# trading day), so this is deliberately loose — it exists to catch "every
# card in every ticker got reshaped", not ordinary churn.
DEFAULT_MAX_CHANGED_FRACTION = 0.25


def _build(ticker: str) -> dict | None:
    try:
        return build_payload(ticker)
    except Exception as exc:
        log.error("%s: build failed — %s", ticker, exc)
        return None


def _changed(doc: dict) -> bool | None:
    """True when Finviz's copy differs from this build.

    None means "couldn't tell" (read-back failed) — the caller treats
    that as changed, matching push_snapshot's fail-open rule.
    """
    ticker = doc["ticker"]
    try:
        live = fetch_snapshot(ticker)
    except FinvizPushError:
        raise
    except Exception as exc:
        log.warning("%s: read-back failed (%s) — assuming changed",
                    ticker, exc)
        return None
    if live is None:
        return True
    return content_digest(live) != content_digest(doc["data"])


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
    # NB: no literal '%' in any help string — argparse %-formats them.
    ap.add_argument("--max-changed", type=int, default=None,
                    help="refuse to publish if more than N tickers changed "
                         f"(default: {DEFAULT_MAX_CHANGED_FRACTION:.2f} of "
                         f"the batch, minimum 4). Only applies to a "
                         f"multi-ticker run.")
    ap.add_argument("--yes", action="store_true",
                    help="acknowledge a wide change set and publish anyway")
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

    # Build everything first. This is also the change-check input, so a
    # wide regression is visible BEFORE anything is published — which is
    # the entire point of the blast-radius gate.
    print(f"Building {len(tickers)} snapshot(s)...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        docs = list(pool.map(_build, tickers))

    built = [(t, d) for t, d in zip(tickers, docs) if d is not None]
    build_failed = [t for t, d in zip(tickers, docs) if d is None]

    # The blast-radius gate needs to know what would change, which means
    # asking Finviz. Skipped entirely for --force-push (nothing to
    # compare against) and for a single ticker (no "radius" to speak of).
    gated = len(built) > 1 and not args.force_push
    changed: list[tuple[str, dict]] = []
    unchanged: list[str] = []
    if gated:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            verdicts = list(pool.map(lambda pair: _changed(pair[1]), built))
        for (ticker, doc), verdict in zip(built, verdicts):
            if verdict is False:
                unchanged.append(ticker)
            else:
                changed.append((ticker, doc))

        limit = args.max_changed
        if limit is None:
            limit = max(4, int(len(built) * DEFAULT_MAX_CHANGED_FRACTION))
        if len(changed) > limit and not args.yes:
            print(f"\nREFUSING TO PUBLISH: {len(changed)} of {len(built)} "
                  f"tickers changed (limit {limit}).", file=sys.stderr)
            print("That usually means a code or prompt change reshaped every "
                  "payload rather than the market moving.", file=sys.stderr)
            print("Inspect a diff first:", file=sys.stderr)
            example = changed[0][0]
            print(f"  python scripts/dump_finviz_payload.py {example} "
                  f"--stdout > /tmp/local.json", file=sys.stderr)
            print(f"  python scripts/dump_finviz_payload.py {example} --live "
                  f"--stdout > /tmp/live.json", file=sys.stderr)
            print("Re-run with --yes if the change is intended.",
                  file=sys.stderr)
            print(f"\nchanged: {', '.join(t for t, _ in changed)}",
                  file=sys.stderr)
            return 2
    else:
        changed = built

    # Push. `if_changed` is False here when the gate already did the
    # comparison — re-asking would double the GETs for no new information.
    results: list[PushResult] = []
    if changed:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(
                lambda pair: _push_one(
                    pair[1],
                    if_changed=not (gated or args.force_push),
                    dry_run=args.dry_run,
                    allow_empty=args.allow_empty),
                changed))

    verb = "would publish" if args.dry_run else "published"
    by_ticker = {t: d for t, d in changed}
    for result in results:
        doc = by_ticker.get(result.ticker)
        extra = f"  {payload_summary(doc)}" if doc else ""
        print(f"{result.ticker:<8} {result.status:<18} {result.reason}{extra}")
    for ticker in unchanged:
        print(f"{ticker:<8} {'unchanged':<18} already live")
    for ticker in build_failed:
        print(f"{ticker:<8} {'build_failed':<18} see log")

    pushed = [r for r in results if r.status == "pushed"]
    skipped_unchanged = [r for r in results
                         if r.status == "skipped_unchanged"]
    invalid = [r for r in results if r.status == "skipped_invalid"]
    failed = [r for r in results if r.status == "failed"]

    print(f"\n{verb} {len(pushed)}  "
          f"unchanged {len(unchanged) + len(skipped_unchanged)}  "
          f"invalid {len(invalid)}  "
          f"failed {len(failed) + len(build_failed)}", file=sys.stderr)

    return 1 if (invalid or failed or build_failed) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FinvizPushError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(3)
