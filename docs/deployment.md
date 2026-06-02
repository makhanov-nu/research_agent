# Deployment

The reference deployment runs the bot on a small always-on VPS (no GPU), with
Postgres + pgvector in Docker, under a systemd service. Heavy compute (if any) is
offloaded to a separate GPU node via the [experiment runner](experiments.md).

## Server prep

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 git python3-venv python3-pip
sudo systemctl enable --now docker
```

## Deploy

```bash
git clone https://github.com/makhanov-nu/research_agent.git
cd research_agent
docker compose up -d                  # Postgres + pgvector (localhost-only)
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[memory]"
cp .env.example .env                  # then fill in the secrets
```

## systemd service

```ini
# /etc/systemd/system/research-agent.service
[Unit]
Description=Research Agent Discord bot
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser/research_agent
ExecStart=/home/youruser/research_agent/.venv/bin/python -m research_agent.main
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now research-agent
journalctl -u research-agent -f       # watch for "Logged in as ..."
```

!!! tip "Updating"
    Pull (or rsync) the latest code, `pip install -e ".[memory]"` to pick up new
    deps, then `sudo systemctl restart research-agent`. The bot reads `.env` on
    start; the database schema migrates itself.

## Discord setup

1. Create an app at the [Discord Developer Portal](https://discord.com/developers/applications).
2. Under **Bot**, enable the **Message Content Intent** (required) and copy the token.
3. Invite the bot with the `bot` scope and send/read-message permissions.

## Notes

- Postgres binds to `127.0.0.1` only — never expose it publicly.
- A small box (≈2 vCPU / 4 GB) comfortably hosts the bot + database; it is **not**
  sized for model training — use the experiment runner's GPU node for that.
