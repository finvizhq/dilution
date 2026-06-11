# Deploying the dilution dashboard to a VPS

End-to-end. Step 1 runs once, step 2 runs on first deploy, step 3 runs each
time you want to push a fresh DB.

## 1. One-time: prep the VPS

SSH in and make sure Python is installed:

```bash
ssh user@VPS_IP
sudo apt update && sudo apt install -y python3-venv python3-pip sqlite3
sudo mkdir -p /opt/dilution
sudo chown $USER:$USER /opt/dilution
exit
```

Open the port (both layers — Vultr's cloud firewall in their web console,
and ufw if it's running):

```bash
sudo ufw allow 5050/tcp   # only if ufw is active on the VPS
```

Plus add an inbound TCP 5050 rule in the Vultr web console → your instance
→ Firewall group, if you've attached one.

## 2. First deploy

From your laptop:

```bash
cd /home/peter/finviz/dilution

# 1. Push code + current DB
rsync -avz --progress \
  --exclude='*.bak-*' --exclude='__pycache__' --exclude='.git' \
  --exclude='walker_dumps' --exclude='logs' --exclude='evals' \
  --exclude='knowledge' --exclude='.venv' --exclude='*.log' \
  ./ user@VPS_IP:/opt/dilution/

# 2. Run the setup script on the VPS
ssh user@VPS_IP
cd /opt/dilution
./deploy.sh
```

`deploy.sh` creates the venv and installs requirements. When it finishes
it prints the start command — run it:

```bash
source .venv/bin/activate
nohup python run_dashboard.py --host 0.0.0.0 --port 5050 > server.log 2>&1 &
disown
exit
```

Open `http://VPS_IP:5050/` in a browser. You're live.

## 3. Ongoing: push a fresh DB after a local pipeline run

After `run_dilution.py` finishes locally, from your laptop:

```bash
cd /home/peter/finviz/dilution
./sync-db.sh user@VPS_IP
```

That checkpoints the WAL, rsyncs `dilution.db`, and bounces the dashboard.
~2 sec of downtime; takes maybe 30 sec end-to-end on a typical connection.

## 4. When the code changes (not just data)

Repeat the first-deploy rsync, then SSH in and bounce manually:

```bash
ssh user@VPS_IP
cd /opt/dilution
./deploy.sh                            # picks up any new requirements
pkill -f run_dashboard.py || true
sleep 1
source .venv/bin/activate
nohup python run_dashboard.py --host 0.0.0.0 --port 5050 > server.log 2>&1 &
disown
exit
```

## Troubleshooting

- **Dashboard won't come up:** `ssh user@VPS_IP 'tail -50 /opt/dilution/server.log'`
- **Page loads but `/ticker/XYZ` is blank:** missing `FINVIZ_API_KEY` in `/opt/dilution/.env`. Add it and bounce.
- **Port 5050 unreachable from outside:** Vultr cloud firewall, not ufw — check the web console.
- **`sync-db.sh` ships a stale-looking DB:** local pipeline was running during the sync. Wait for it to finish, then re-run.
