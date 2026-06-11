# Dilution test suite

Deterministic `pytest` unit tests for the dilution-tracker logic layer.
No test ever touches the network, an LLM provider, or the real
`dilution.db` — see "Isolation" below.

## Running

```bash
# from the repo root (/home/peter/finviz/dilution)
/home/peter/.venvs/dilution/bin/python3 -m pytest tests/ -q

# one module
/home/peter/.venvs/dilution/bin/python3 -m pytest tests/test_splits.py -q

# show the documented-bug expected-failures
/home/peter/.venvs/dilution/bin/python3 -m pytest tests/ -q -rx
```

`pytest.ini` (repo root) sets `pythonpath = .` so the tests import
`config`, `db`, and `dilution.*` exactly as production does.

## Isolation (conftest.py)

Two **autouse** fixtures run for every test:

- `temp_db` — points `db.get_conn()` at a fresh per-test SQLite file with
  the full production schema, so the ~1.2 GB real DB is never opened.
  `db.get_conn()` resolves `DB_PATH` from the `db` module's globals at call
  time, so one `monkeypatch.setattr(db, "DB_PATH", ...)` reroutes every
  caller. It yields a `DBHelper` with schema-accurate row inserters:
  `add_company`, `add_filing`, `add_instrument`, `add_drawdown`,
  `add_split`, `execute`, `conn`.
- `_reset_caches` — clears every `@lru_cache` and module-level dict cache
  (`baby_shelf._BABY_EXIT_CLOSES_CACHE`, `cards._MARKET_LOW_CACHE`, the
  `_cached*` memos in `share_counts`/`cash_history`/`os_history`/
  `ib6_cover`/`badges`) before and after each test. Those memos key on
  `cik`/`ticker`/date but not on the DB file, so without this a value
  computed against one test's `temp_db` could leak into another.

Network/LLM/vendor/filesystem seams are `monkeypatch`-ed at the boundary
in each test (`requests.get`, `urllib.request.urlopen`, `yfinance`,
edgartools, the finviz client, `open`).

## Coverage

33 modules of the deterministic logic layer (split math, FX, baby-shelf /
IB6 regulatory model, shelf/S-1 status machines, ledger mutations & tool
arg-parsing, validation, overhang extraction, card view projection,
badges, OS/cash history, unit detection, exhibit/item classification,
registration-family linkage, …). The pure declarative walker tool specs
(`tools/create.py|amend.py|record.py`) have no logic and are intentionally
not tested directly — their schema codegen is exercised via
`test_ledger_tools__base.py`.

## Bugs surfaced by the suite

Writing the suite surfaced a set of divergences between a module's
documented intent and its actual behavior (see `tests/KNOWN_ISSUES.md`).

- **Robustness (A) and logic (B) bugs are now FIXED** — the production code
  was corrected and every corresponding test now asserts the intended
  behavior (no `xfail` markers remain). Each fixed test carries a
  `# FIXED (was bug …)` note pointing back to the catalogue entry.
- A few **cosmetic / latent (C)** divergences remain, captured as
  characterization tests with a `# BUG:` note (they pin current behavior
  rather than asserting a fix). See `KNOWN_ISSUES.md` for the catalogue
  with file/line references.
