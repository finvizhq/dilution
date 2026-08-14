# Deploying the dilution backend service

This is a headless service. There is no web UI: the pipeline walks SEC
filings into a ledger and pushes one JSON snapshot per ticker to Finviz,
which renders the product. The only HTTP surface is `run_inspect.py`, a
loopback-only debug view you reach over an SSH tunnel.

**Nightly shape** — one systemd timer runs `scripts/nightly.sh`:

1. walk every tracked ticker (`--no-push`)
2. refresh briefs whose ticker got new filings
3. publish the snapshots whose content changed

Step 3 is one pass at the end rather than a push per walk, because step 2
regenerates exactly the briefs the walk invalidated — pushing per-walk
would ship the old brief beside the new cards.

## What lives where

| | |
|---|---|
| Source of truth for published data | **Finviz.** `GET /api/dilution/<TICKER>?auth=` is how you see what's live |
| Source of truth for extraction | **`dilution_mutations`** in the local DB — the applied-mutation log |
| `dilution.db` | Local working state. Not synced, not served |
| Irreplaceable | The mutation log (~5 MB across the universe). Everything else re-derives |

`dilution.db` is ~1.2 GB, but 99.3% of that is `dilution_raw`, a cache of
EDGAR filing text that can be re-fetched. The ledger is a deterministic
fold of the mutation log, so a corrupted or lost ledger replays in seconds
at zero LLM cost:

```bash
python scripts/rebuild_ledger.py --ticker CELU --dry-run
```

That is the recovery story — **not** "restore a backup". What does deserve
backing up is the mutation log, because LLM extraction costs money and is
not reproducible.

---

## 1. One-time: prepare the VPS

```bash
ssh user@VPS_IP
sudo apt update && sudo apt install -y python3-venv python3-pip sqlite3
sudo mkdir -p /opt/dilution
sudo chown $USER:$USER /opt/dilution
exit
```

No firewall rule is needed. If you are migrating from the old dashboard,
close its port — `run_inspect.py` binds to loopback now:

```bash
sudo ufw delete allow 5050/tcp
```

## 2. Push the code

From your laptop. Note `--exclude='dilution.db'`: the database is not
shipped by this command (step 3 seeds it once, deliberately).

```bash
cd /home/peter/finviz/dilution
rsync -avz --progress \
  --exclude='*.bak-*' --exclude='__pycache__' --exclude='.git' \
  --exclude='walker_dumps' --exclude='logs' --exclude='evals' \
  --exclude='knowledge' --exclude='.venv' --exclude='*.log' \
  --exclude='dilution.db' \
  ./ user@VPS_IP:/opt/dilution/
```

## 3. One-time: seed the database

**Do this once, and understand why it is not a sync.** A cold start would
re-walk 66 tickers across six years of filings and re-pay the entire
historical LLM bill. Copying an existing DB avoids that. Nothing ships it
again afterwards — the VPS's copy diverges from your laptop's from here on,
by design, and neither is authoritative for the other.

```bash
cd /home/peter/finviz/dilution
sqlite3 dilution.db "PRAGMA wal_checkpoint(TRUNCATE);"
rsync -avz --progress dilution.db user@VPS_IP:/opt/dilution/
```

## 4. Install the service

```bash
ssh user@VPS_IP
cd /opt/dilution
```

Create `.env` by hand — it is gitignored and never rsynced:

```bash
install -m 600 /dev/null .env
$EDITOR .env
```

It needs:

| key | purpose |
|---|---|
| `FINVIZ_INGEST_TOKEN` | **write** credential for `POST /api/dilution/set` |
| `FINVIZ_API_KEY` | Elite `/export` **read** key (market data) |
| `OPENAI_API_KEY` | the walker's LLM |
| `LANGFUSE_*` | optional tracing |

The two Finviz values are different credentials and there is no fallback
between them: a missing `FINVIZ_INGEST_TOKEN` raises before any request
rather than 401-ing mid-batch.

Then:

```bash
./deploy.sh
```

`deploy.sh` builds the venv, installs requirements, checks the
environment, and dry-run-publishes one ticker as a self-check (validating
and change-checking without POSTing). It refuses to declare success if
anything is missing. Follow the systemd commands it prints:

```bash
sudo cp deploy/dilution-nightly.service deploy/dilution-nightly.timer \
     /etc/systemd/system/
sudo sed -i "s/^User=CHANGEME/User=$USER/" \
     /etc/systemd/system/dilution-nightly.service
sudo systemctl daemon-reload
sudo systemctl enable --now dilution-nightly.timer
```

`User=CHANGEME` is a required edit. Left as-is the unit runs as root and
writes root-owned files through `/opt/dilution`, which then breaks the
next rsync.

## 5. Verify before trusting the schedule

```bash
systemctl list-timers dilution-nightly     # next fire time
./scripts/nightly.sh --dry-run             # full run, publishes nothing
```

The dry run walks and refreshes briefs for real, then validates and
change-checks every snapshot without POSTing. Check the log shows all
three steps in order.

Then let the timer fire once and confirm what landed:

```bash
journalctl -u dilution-nightly -n 50
python scripts/dump_finviz_payload.py CELU --live --stdout | jq .as_of
```

`as_of` should be the most recent settled trading date.

---

## Timing matters for correctness

The timer fires at **23:15 UTC** — after the US settled close.

This is not a scheduling preference. `as_of` and every market-derived
field (`highest_60_day_close`, badge scores, baby-shelf status) come from
`latest_settled_close`, and Finviz's daily export carries the **live**
price in its last bar until the session settles. A run that starts before
the close publishes an intraday price as a settled one.

23:15 UTC is 18:15 ET in winter and 19:15 ET in summer, clearing the 16:00
ET close either way. The timer is stated in UTC on purpose: `OnCalendar`
follows the system timezone, so with a UTC host and an ET market the DST
shift moves the margin, not the deadline.

---

## Operating it

```bash
journalctl -u dilution-nightly -f          # live
tail -f logs/nightly_$(date +%F).log       # this run's detail
ls logs/open_access_*.log                  # per-ticker walk logs
```

**Exit codes.** 0 clean; 1 a step reported failures; **2 the blast-radius
gate declined to publish.** Exit 2 means an implausible share of the
universe came up changed, which normally means a code or prompt change
reshaped every payload rather than the market moving. Nothing was
published. Inspect a diff:

```bash
python scripts/dump_finviz_payload.py CELU --stdout > /tmp/local.json
python scripts/dump_finviz_payload.py CELU --live --stdout > /tmp/live.json
diff <(jq -S .data /tmp/local.json) <(jq -S .data /tmp/live.json)
```

If the change is intended (a pipeline release), publish deliberately:

```bash
python scripts/push_finviz.py --all --yes
```

The unit treats exit 2 as success (`SuccessExitStatus=2`) so it reads as
"held for review" rather than paging on every release.

**Publishing by hand:**

```bash
python scripts/push_finviz.py CELU                 # one ticker
python scripts/push_finviz.py --all --dry-run      # validate the universe
python scripts/push_finviz.py CELU --force-push    # ignore the change check
```

Unchanged tickers cost a GET and no POST, so re-running `--all` is cheap.

**The debug view**, over a tunnel — never a public port:

```bash
ssh -L 5050:127.0.0.1:5050 user@VPS_IP
# on the VPS:
cd /opt/dilution && source .venv/bin/activate && python run_inspect.py
# then open http://127.0.0.1:5050/inspect locally
```

## 6. Ongoing: shipping a code change

```bash
cd /home/peter/finviz/dilution
rsync -avz --progress \
  --exclude='*.bak-*' --exclude='__pycache__' --exclude='.git' \
  --exclude='walker_dumps' --exclude='logs' --exclude='evals' \
  --exclude='knowledge' --exclude='.venv' --exclude='*.log' \
  --exclude='dilution.db' \
  ./ user@VPS_IP:/opt/dilution/
ssh user@VPS_IP 'cd /opt/dilution && ./deploy.sh'
```

No service to bounce — the timer picks up the new code on its next fire.
A release that reshapes payloads will trip the blast-radius gate on that
first run; that is the intended checkpoint.

---

## Backing up what matters

The DB is disposable; the mutation log inside it is not. It is a few MB and
compresses well:

```bash
sqlite3 dilution.db ".backup /tmp/dilution-backup.db"
sqlite3 /tmp/dilution-backup.db \
  "DELETE FROM dilution_raw; VACUUM;"     # drop the re-fetchable cache
gzip /tmp/dilution-backup.db
```

Copy that off-box. For continuous replication instead, litestream streams
SQLite to object storage with no application changes.

**Known gap:** the mutation log only covers walks performed after it was
introduced. Tickers walked before then have a ledger but no replayable
history — `rebuild_ledger.py` reports that explicitly rather than claiming
success. Until the universe has been walked once with logging on, the
seeded DB from step 3 is the pre-log baseline, so keep a copy of it.

## Troubleshooting

- **Timer never fires:** `systemctl status dilution-nightly.timer`, then
  `systemctl list-timers`. A masked or non-enabled timer shows here.
- **Unit fails immediately:** almost always `.env` or `User=CHANGEME`.
  `journalctl -u dilution-nightly -n 30`.
- **`FINVIZ_INGEST_TOKEN is not set`:** the unit reads `.env` via
  `EnvironmentFile`; confirm the key has a non-empty value and no
  surrounding quotes.
- **Pushes 401:** the ingest token is a different credential from
  `FINVIZ_API_KEY`. Check you did not paste the read key.
- **Pushes 400:** a producer bug, never retried. The log carries the
  ASP.NET `traceId` — quote it to Finviz infra.
- **A run seems stuck:** a cold walk is genuinely hours. Check per-ticker
  progress in `logs/open_access_<TICKER>.log`. `TimeoutStartSec=8h` is the
  outer bound.
- **Two runs at once:** they cannot. `nightly.sh` takes an advisory lock
  (`/opt/dilution/.nightly.lock`) and a second invocation exits 0
  immediately.
- **SEC 429 / IP ban (~10 min):** `PARALLEL * EDGAR_RATE_LIMIT_PER_SEC`
  must stay ≤ 10. Defaults are 4 × 2 = 8.
- **A ticker looks wrong on Finviz:** compare live against a fresh build
  with `--live` (above). The cards are authoritative; report `ticker` +
  `source_ref` + `generated_at`.
