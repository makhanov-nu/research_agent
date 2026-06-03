# Deploying the web UI (on the bot VPS, behind Cloudflare)

The web UI runs on the **same machine as the bot** (`research-vps`), sharing its
Postgres + `outputs/`. It binds `127.0.0.1:8800`; **Caddy** fronts it for TLS, and
**Cloudflare** proxies the public domain `airesearchagent.uk`.

## One-time

```bash
# 1) deps (web + observability extras), in the existing venv
cd ~/research_agent && source .venv/bin/activate
pip install -e ".[memory,web,obs]"

# 2) ship the prebuilt SPA (Node isn't installed on the VPS) — from your laptop:
#    (build once: cd src/research_agent/web/frontend && npm install && npm run build)
rsync -az src/research_agent/web/frontend/dist/ \
  research-vps:~/research_agent/src/research_agent/web/frontend/dist/

# 3) .env additions (see below)

# 4) Phoenix (local trace UI)
docker compose up -d phoenix

# 5) web service
sudo cp deploy/research-agent-web.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now research-agent-web

# 6) Caddy (reverse proxy / TLS) — pick the block in deploy/Caddyfile that
#    matches your Cloudflare SSL mode, then:
sudo apt install -y caddy
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo systemctl restart caddy
```

Open the GCP firewall for 80/443 (Cloudflare → origin).

## `.env` additions

```
WEB_HOST=127.0.0.1
WEB_PORT=8800
WEB_BASE_URL=https://airesearchagent.uk
WORKOS_REDIRECT_URI=https://airesearchagent.uk/auth/callback
WEB_SESSION_SECRET=<generated>
WEB_ALLOWED_EMAILS=makhanov91@gmail.com
WORKOS_API_KEY=<set>
WORKOS_CLIENT_ID=<set>
PHOENIX_ENABLED=true
```

## Updating later

```bash
rsync the dist/ again (after rebuilding) and: sudo systemctl restart research-agent-web
```
