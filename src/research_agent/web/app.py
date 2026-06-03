"""FastAPI app: JSON API + SSE task feed behind WorkOS AuthKit, serving the SPA.

Runs as its own process (`research-agent-web`) over the same Postgres DB and
`outputs/` folder as the bot — a read-mostly viewer. The live task feed is SSE
backed by polling the tasks table, so the web app stays decoupled from the bot.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from ..config import settings
from . import auth

logger = logging.getLogger(__name__)


def _safe_path(rel: str) -> Path:
    """Resolve `rel` under the outputs dir, rejecting traversal/absolute paths."""
    root = Path(settings.output_dir).resolve()
    target = (root / rel).resolve()
    if target != root and root not in target.parents:
        raise ValueError("path escapes outputs root")
    return target


def _file_tree(base: Path) -> list[dict]:
    if not base.exists():
        return []
    out = []
    for p in sorted(base.rglob("*")):
        if p.is_file():
            out.append({
                "rel": str(p.resolve().relative_to(Path(settings.output_dir).resolve())),
                "name": p.name,
                "size": p.stat().st_size,
            })
    return out


def create_app():
    from fastapi import Depends, FastAPI, HTTPException, Query, Request
    from fastapi.responses import (
        FileResponse,
        JSONResponse,
        PlainTextResponse,
        RedirectResponse,
    )

    from ..agents.task_store import TaskStore
    from ..db import open_pool
    from ..projects.store import ProjectStore

    app = FastAPI(title="Research Agent", docs_url=None, redoc_url=None)

    @app.on_event("startup")
    async def _startup() -> None:
        app.state.pool = await open_pool()
        app.state.projects = ProjectStore(app.state.pool, settings.output_dir)
        app.state.tasks = TaskStore(app.state.pool)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        pool = getattr(app.state, "pool", None)
        if pool is not None:
            await pool.close()

    # --- auth routes ---

    @app.get("/auth/login")
    async def login():
        if not auth.configured():
            raise HTTPException(503, "WorkOS AuthKit is not configured")
        return RedirectResponse(auth.authorization_url())

    @app.get("/auth/callback")
    async def callback(code: str = "", error: str = ""):
        if error or not code:
            return RedirectResponse("/?auth_error=1")
        try:
            user = auth.authenticate_code(code)
        except PermissionError:
            return RedirectResponse("/?auth_error=forbidden")
        except Exception:  # noqa: BLE001
            logger.exception("Auth callback failed")
            return RedirectResponse("/?auth_error=1")
        resp = RedirectResponse("/")
        resp.set_cookie(
            auth.COOKIE_NAME, auth.seal_session(user), max_age=auth.SESSION_MAX_AGE,
            httponly=True, samesite="lax", secure=settings.web_is_https,
        )
        return resp

    @app.get("/auth/logout")
    async def logout():
        resp = RedirectResponse("/")
        resp.delete_cookie(auth.COOKIE_NAME)
        return resp

    @app.get("/api/me")
    async def me(user: dict = Depends(auth.require_user)):
        return user

    # --- data API (all require auth) ---

    @app.get("/api/projects")
    async def list_projects(user: dict = Depends(auth.require_user)):
        return await app.state.projects.list_projects()

    @app.get("/api/projects/{project_id}")
    async def project_detail(project_id: int, user: dict = Depends(auth.require_user)):
        projects = app.state.projects
        rows = await projects.list_projects()
        proj = next((p for p in rows if p["id"] == project_id), None)
        if proj is None:
            raise HTTPException(404, "project not found")
        artifacts = await projects.list_artifacts(project_id)
        tree = _file_tree(projects.project_dir(proj["slug"]))
        return {"project": proj, "artifacts": artifacts, "files": tree}

    @app.get("/api/file")
    async def read_file(path: str = Query(...), user: dict = Depends(auth.require_user)):
        try:
            target = _safe_path(path)
        except ValueError:
            raise HTTPException(400, "invalid path")
        if not target.is_file():
            raise HTTPException(404, "file not found")
        # Render text inline; offer binary as a download.
        if target.suffix.lower() in {".tex", ".bib", ".md", ".txt", ".json", ".py", ".log", ".csv"}:
            return PlainTextResponse(target.read_text(errors="replace"))
        return FileResponse(target, filename=target.name)

    async def _attach_projects(rows: list[dict]) -> list[dict]:
        """Annotate task rows with their project (name/id) by channel."""
        cache: dict[str, dict | None] = {}
        for r in rows:
            chan = r.get("channel_id")
            if chan and chan not in cache:
                cache[chan] = await app.state.projects.get_by_channel(chan)
            proj = cache.get(chan)
            r["project_id"] = proj["id"] if proj else None
            r["project_name"] = proj["name"] if proj else None
        return rows

    @app.get("/api/tasks")
    async def list_tasks(limit: int = 30, user: dict = Depends(auth.require_user)):
        return await _attach_projects(await app.state.tasks.list_recent(limit=limit))

    @app.get("/api/tasks/{task_id}")
    async def task_detail(task_id: int, user: dict = Depends(auth.require_user)):
        row = await app.state.tasks.get(task_id)
        if row is None:
            raise HTTPException(404, "task not found")
        (await _attach_projects([row]))
        return row

    @app.get("/api/tasks/stream")
    async def task_stream(request: Request, user: dict = Depends(auth.require_user)):
        from sse_starlette.sse import EventSourceResponse

        async def gen():
            last = None
            while True:
                if await request.is_disconnected():
                    break
                rows = await app.state.tasks.list_recent(limit=30)
                payload = json.dumps(rows, default=str)
                if payload != last:
                    last = payload
                    yield {"event": "tasks", "data": payload}
                await asyncio.sleep(2)

        return EventSourceResponse(gen())

    # --- Phoenix trace UI, reverse-proxied behind auth ---
    # Phoenix runs locally (PHOENIX_HOST_ROOT_PATH=/phoenix) and isn't exposed
    # publicly; we proxy it under /phoenix so the frontend can embed it and it
    # inherits the WorkOS session. Registered before the SPA catch-all.
    _HOP_BY_HOP = {
        "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
        "te", "trailers", "transfer-encoding", "upgrade", "content-encoding",
        "content-length",
    }

    @app.api_route(
        "/phoenix/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def phoenix_proxy(path: str, request: Request, user: dict = Depends(auth.require_user)):
        import httpx
        from fastapi.responses import Response

        root = settings.phoenix_root_path.strip("/")
        target = f"{settings.phoenix_internal_url.rstrip('/')}/{root}/{path}"
        url = httpx.URL(target, query=request.url.query.encode("utf-8"))
        fwd = {
            k: v for k, v in request.headers.items()
            if k.lower() not in _HOP_BY_HOP and k.lower() != "host"
        }
        body = await request.body()
        try:
            async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client:
                upstream = await client.request(
                    request.method, url, headers=fwd, content=body
                )
        except httpx.HTTPError:
            raise HTTPException(502, "Phoenix is not reachable (is it running?)")
        out = {
            k: v for k, v in upstream.headers.items()
            if k.lower() not in _HOP_BY_HOP
            # allow embedding the proxied UI in our SPA iframe
            and k.lower() not in {"x-frame-options", "content-security-policy"}
        }
        return Response(content=upstream.content, status_code=upstream.status_code, headers=out)

    # --- SPA static (built React app), if present ---

    dist = Path(__file__).resolve().parent / "frontend" / "dist"
    if dist.exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{full_path:path}")
        async def spa(full_path: str):
            # Serve real files; otherwise fall back to index.html (client routing).
            candidate = (dist / full_path).resolve()
            if full_path and candidate.is_file() and dist in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")
    else:
        @app.get("/")
        async def placeholder():
            return JSONResponse({
                "status": "ok",
                "note": "API up. Build the SPA (web/frontend) to serve the UI.",
            })

    return app
