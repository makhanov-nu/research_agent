# Research Agent — web frontend

React (Vite) SPA for the web UI: browse projects + their artifacts, read files,
and watch the task dashboard live (SSE). It's served by the FastAPI app
(`research-agent-web`) from `dist/`.

## Develop

```bash
cd src/research_agent/web/frontend
npm install
npm run dev          # http://localhost:5173 (proxies /api and /auth to :8800)
```

Run the API alongside it: `research-agent-web` (needs `pip install -e ".[web]"`).

## Build (for serving in production)

```bash
npm install
npm run build        # outputs ./dist, which FastAPI serves at /
```

The FastAPI app auto-detects `dist/` and serves the SPA; without it, `/` returns
a JSON placeholder.
