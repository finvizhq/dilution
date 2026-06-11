#!/usr/bin/env python3
"""Speedometer for the dilution eval suite.

Scores every fixture in evals/*.json against the current DB, prints the
aggregate plus the delta since the last recorded run, and appends today's
datapoint to evals/_trend.jsonl so you can watch the line move over time.

    python run_eval_all.py            # score all, record a datapoint, show trend
    python run_eval_all.py --no-log   # score + show, but don't record
    python run_eval_all.py --trend    # just print the recorded history

Each datapoint is tied to the current git commit (and a dirty flag) so a
score change can be attributed to the change that caused it. The point is
to answer one question a per-ticker snapshot can't: is the line moving?
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from db import get_conn
from dilution.ledger.walker import _SKIPPED_FORMS
from run_eval import score

EVALS_DIR = ROOT / "evals"
TREND_PATH = EVALS_DIR / "_trend.jsonl"


# ─── helpers ─────────────────────────────────────────────────────────
def _git_state() -> tuple[str, bool]:
    """(short-commit, dirty?). Returns ('unknown', False) outside git."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip())
        return commit, dirty
    except Exception:
        return "unknown", False


def _discover() -> list[str]:
    """Fixture tickers, sorted. Files prefixed '_' (e.g. _trend) skipped."""
    return sorted(p.stem for p in EVALS_DIR.glob("*.json")
                  if not p.stem.startswith("_"))


def _walk_status(tickers: list[str]) -> dict[str, tuple[str, str | None, str | None]]:
    """Per-ticker walk completeness: 'complete' | 'partial' | 'absent'.

    'complete' means the walker's last_processed_filing_date has reached
    the latest walkable filing on record for that CIK — the ledger
    reflects every known filing, so scoring it is meaningful. 'partial'
    means a walk is in progress or was interrupted (last_processed lags
    the latest filing), so the ledger is half-built and any score is
    noise. 'absent' means never walked / not in the DB. Returns
    (status, last_processed_date, latest_filing_date) per ticker.

    Forms the walker never processes (EFFECT/RW — walker._SKIPPED_FORMS)
    are excluded from the high-water mark: the watermark can never reach
    them, so counting them would flag a finished walk as mid-walk
    forever."""
    out: dict[str, tuple[str, str | None, str | None]] = {
        t: ("absent", None, None) for t in tickers
    }
    if not tickers:
        return out
    qmarks = ",".join("?" * len(tickers))
    skipped = sorted(_SKIPPED_FORMS)
    skip_qmarks = ",".join("?" * len(skipped))
    with get_conn() as conn:
        rows = conn.execute(
            f"""SELECT co.ticker AS ticker,
                       ws.last_processed_filing_date AS lp,
                       (SELECT MAX(filing_date) FROM dilution_filings f
                         WHERE f.cik = co.cik
                           AND UPPER(COALESCE(f.form, ''))
                               NOT IN ({skip_qmarks})) AS latest
                  FROM dilution_company co
                  LEFT JOIN dilution_walk_state ws ON ws.cik = co.cik
                 WHERE co.ticker IN ({qmarks})""",
            [*skipped, *tickers],
        ).fetchall()
    for r in rows:
        lp, latest = r["lp"], r["latest"]
        if not lp or not latest:
            status = "absent"
        elif lp < latest:
            status = "partial"
        else:
            status = "complete"
        out[r["ticker"]] = (status, lp, latest)
    return out


def _load_trend() -> list[dict]:
    if not TREND_PATH.exists():
        return []
    out = []
    for line in TREND_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _pct(n, d) -> float:
    return (100.0 * n / d) if d else 0.0


def _delta(cur, prev, prec=1, suffix="") -> str:
    """Signed delta vs the previous run, or '' when there's no baseline."""
    if prev is None:
        return ""
    d = cur - prev
    if abs(d) < 0.5 * 10 ** -prec:
        return "  ="
    return f"  {'▲' if d > 0 else '▼'}{abs(d):.{prec}f}{suffix}"


def _sparkline(values: list[float]) -> str:
    bars = "▁▂▃▄▅▆▇█"
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi == lo:
        return bars[len(bars) // 2] * len(values)
    return "".join(
        bars[min(len(bars) - 1, int((v - lo) / (hi - lo) * (len(bars) - 1)))]
        for v in values)


# ─── rendering ───────────────────────────────────────────────────────
def _print_report(entry: dict, prev: dict | None,
                  partial: dict, errored: dict) -> None:
    agg = entry["agg"]
    prev_tk = (prev or {}).get("tickers", {})
    prev_agg = (prev or {}).get("agg")

    print("=" * 58)
    print(" DILUTION EVAL SPEEDOMETER")
    print("=" * 58)
    print(f" {'ticker':<7}{'fields':>9}{'Δ vs last':>12}  {'cards':>7}  result")
    print(" " + "-" * 52)
    for t in sorted(entry["tickers"]):
        r = entry["tickers"][t]
        cur = _pct(r["fp"], r["fc"])
        p = prev_tk.get(t)
        d = _delta(cur, _pct(p["fp"], p["fc"]) if p else None)
        if t in partial:
            result = f"⚠ mid-walk (@{partial[t][1]})"
        elif r["pass"]:
            result = "PASS"
        elif r["extra"]:
            result = f"{r['extra']} extra"
        else:
            result = "fail"
        flds = f"{r['fp']}/{r['fc']}"
        cards = f"{r['exact']}/{r['exp']}"
        print(f" {t:<7}{flds:>9}{d:>12}  {cards:>7}  {result}")
    for t in sorted(errored):
        print(f" {t:<7}{'—':>9}{'':>12}  {'—':>7}  not walked")
    print(" " + "-" * 52)

    excl = len(partial) + len(errored)
    excl_note = (f"   (excl. {len(partial)} partial, {len(errored)} not-walked)"
                 if excl else "")
    if agg["field_checks"]:
        cur_field = _pct(agg["field_passes"], agg["field_checks"])
        prev_field = (_pct(prev_agg["field_passes"], prev_agg["field_checks"])
                      if prev_agg else None)
        print(f" TOTAL   fields {agg['field_passes']}/{agg['field_checks']} = "
              f"{cur_field:.0f}%{_delta(cur_field, prev_field, suffix=' pts')}")
    else:
        print(" TOTAL   fields — (no fully-walked fixtures to aggregate)")
    print(f"         fixtures passed {agg['fixtures_passed']}/{agg['fixtures_scored']}"
          f"{_delta(agg['fixtures_passed'], prev_agg['fixtures_passed'] if prev_agg else None, prec=0)}"
          f"{excl_note}")
    print(f"         cards exact {agg['cards_exact']}/{agg['cards_expected']}, "
          f"found {agg['cards_found']}/{agg['cards_expected']}, extra {agg['extra']}")
    when = "first datapoint — no baseline yet" if prev is None else \
        f"baseline: {prev['commit']}{'*' if prev.get('dirty') else ''} @ {prev['ts'][:10]}"
    print(f"         ({when})")


def _print_trend(history: list[dict], tail: int = 12) -> None:
    if not history:
        print(" (no recorded datapoints yet — run without --trend to log one)")
        return
    series = [_pct(h["agg"]["field_passes"], h["agg"]["field_checks"])
              for h in history]
    print(f"\n TREND  field%: {_sparkline(series)}   "
          f"{series[0]:.0f}% → {series[-1]:.0f}% over {len(history)} run"
          f"{'s' if len(history) != 1 else ''}")
    print(f" {'date':<11} {'commit':<9} {'field%':>7} {'fixtures':>9}")
    for h in history[-tail:]:
        a = h["agg"]
        tag = h["commit"] + ("*" if h.get("dirty") else "")
        print(f" {h['ts'][:10]:<11} {tag:<9} "
              f"{_pct(a['field_passes'], a['field_checks']):>6.0f}% "
              f"{a['fixtures_passed']:>4}/{a['fixtures_scored']}")


# ─── main ────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-log", action="store_true",
                    help="score and display but don't record a datapoint")
    ap.add_argument("--trend", action="store_true",
                    help="print recorded history only; don't score")
    ap.add_argument("--force", action="store_true",
                    help="record a datapoint even if some fixtures are "
                         "mid-walk / not walked (normally refused)")
    args = ap.parse_args()

    history = _load_trend()

    if args.trend:
        _print_trend(history, tail=10**9)
        return

    tickers = _discover()
    walk = _walk_status(tickers)
    results, errored, partial = {}, {}, {}
    for t in tickers:
        try:
            r = score(t)
        except (FileNotFoundError, LookupError) as e:
            errored[t] = str(e)
            continue
        results[t] = r
        if walk[t][0] == "partial":
            partial[t] = walk[t]

    if not results:
        sys.exit("no fixtures could be scored — is the DB walked? "
                 + ("; ".join(errored.values()) if errored else ""))

    # Aggregate over fully-walked fixtures only — a partial ledger's
    # numbers are noise and would poison the trend.
    complete = {t: r for t, r in results.items() if t not in partial}
    agg_src = complete.values()
    commit, dirty = _git_state()
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit,
        "dirty": dirty,
        "agg": {
            "field_passes": sum(r["field_passes"] for r in agg_src),
            "field_checks": sum(r["field_checks"] for r in complete.values()),
            "fixtures_passed": sum(1 for r in complete.values() if r["passed"]),
            "fixtures_scored": len(complete),
            "cards_exact": sum(r["exact"] for r in complete.values()),
            "cards_found": sum(r["found"] for r in complete.values()),
            "cards_expected": sum(r["expected"] for r in complete.values()),
            "extra": sum(r["extra"] for r in complete.values()),
        },
        "tickers": {
            t: {"fp": r["field_passes"], "fc": r["field_checks"],
                "pass": r["passed"], "extra": r["extra"],
                "found": r["found"], "exp": r["expected"], "exact": r["exact"]}
            for t, r in results.items()
        },
        "partial": sorted(partial),
        "errored": sorted(errored),
    }

    _print_report(entry, history[-1] if history else None, partial, errored)

    blockers = sorted(set(partial) | set(errored))
    if args.no_log:
        print("\n (--no-log: datapoint NOT recorded)")
        _print_trend(history)
        return
    if blockers and not args.force:
        print(f"\n ⚠ NOT recorded — {len(blockers)} fixture(s) not fully "
              f"walked: {', '.join(blockers)}")
        print("   A walk is in progress or incomplete, so the DB is "
              "unstable. Re-run when it finishes, or pass --force.")
        _print_trend(history)
        return
    if not complete:
        sys.exit("\n no fully-walked fixtures — nothing meaningful to record.")
    if blockers:
        entry["forced"] = True
    with TREND_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\n recorded → evals/{TREND_PATH.name}"
          + ("  (FORCED — excludes the partial/not-walked fixtures above)"
             if entry.get("forced") else ""))
    _print_trend(history + [entry])


if __name__ == "__main__":
    main()
