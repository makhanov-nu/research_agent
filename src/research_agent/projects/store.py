"""Projects: treat each chat (Discord channel) as a project.

A project owns a folder tree under `outputs/projects/<slug>/` with a subfolder per
artifact kind, and a Postgres record so the web frontend can list projects and
their artifacts. Every produced artifact (lit review, council proposal,
methodology, experiment, paper) is saved into the project folder AND registered
in the `artifacts` table.

Degrades to folder-only (no registry) when there's no DB pool; folder paths are
always returned so callers can save regardless.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Artifact kinds == subfolders under a project.
KINDS = ("lit_review", "council", "methodology", "experiments", "paper")

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id          BIGSERIAL PRIMARY KEY,
    channel_id  TEXT UNIQUE,
    slug        TEXT UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS artifacts (
    id          BIGSERIAL PRIMARY KEY,
    project_id  BIGINT NOT NULL REFERENCES projects(id),
    kind        TEXT NOT NULL,
    title       TEXT,
    rel_path    TEXT NOT NULL,
    meta        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS artifacts_project_idx ON artifacts (project_id, created_at DESC);
"""


def slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (s[:max_len].rstrip("-")) or "project"


class ProjectStore:
    def __init__(self, pool, output_dir: str):
        self.pool = pool
        self.root = Path(output_dir) / "projects"

    @property
    def enabled(self) -> bool:
        return self.pool is not None

    async def setup(self) -> None:
        if not self.enabled:
            return
        async with self.pool.connection() as conn:
            await conn.execute(SCHEMA)
        logger.info("Project store schema ready.")

    # -- folders --

    def project_dir(self, slug: str) -> Path:
        return self.root / slug

    def kind_dir(self, slug: str, kind: str) -> Path:
        d = self.project_dir(slug) / kind
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _make_dirs(self, slug: str) -> None:
        for kind in KINDS:
            (self.project_dir(slug) / kind).mkdir(parents=True, exist_ok=True)

    # -- projects --

    async def ensure(self, channel_id: str | None, name: str | None = None) -> dict:
        """Get or create the project for a channel; always returns a project dict.

        Without a DB this returns a lightweight folder-only project keyed by the
        channel so artifacts still organize on disk.
        """
        default_name = name or (f"project-{channel_id}" if channel_id else "project")
        if not self.enabled:
            slug = slugify(default_name)
            self._make_dirs(slug)
            return {"id": None, "channel_id": channel_id, "slug": slug, "name": default_name}

        existing = await self.get_by_channel(channel_id)
        if existing:
            return existing

        slug = await self._unique_slug(slugify(default_name))
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO projects (channel_id, slug, name) VALUES (%s, %s, %s) "
                "RETURNING id, channel_id, slug, name",
                (channel_id, slug, default_name),
            )
            row = await cur.fetchone()
        self._make_dirs(row["slug"])
        logger.info("Created project %s (%s)", row["name"], row["slug"])
        return dict(row)

    async def _unique_slug(self, base: str) -> str:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT slug FROM projects WHERE slug LIKE %s", (base + "%",)
            )
            taken = {r["slug"] for r in await cur.fetchall()}
        if base not in taken:
            return base
        i = 2
        while f"{base}-{i}" in taken:
            i += 1
        return f"{base}-{i}"

    async def get_by_channel(self, channel_id: str | None) -> Optional[dict]:
        if not self.enabled or channel_id is None:
            return None
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id, channel_id, slug, name FROM projects WHERE channel_id=%s",
                (channel_id,),
            )
            row = await cur.fetchone()
        return dict(row) if row else None

    async def rename(self, channel_id: str | None, name: str) -> Optional[dict]:
        if not self.enabled:
            return None
        proj = await self.ensure(channel_id)
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE projects SET name=%s WHERE id=%s", (name, proj["id"])
            )
        proj["name"] = name
        return proj

    async def list_projects(self) -> list[dict]:
        if not self.enabled:
            return []
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id, channel_id, slug, name, created_at FROM projects "
                "ORDER BY created_at DESC"
            )
            return [dict(r) for r in await cur.fetchall()]

    # -- artifacts --

    async def add_artifact(
        self, project_id: int | None, kind: str, title: str, rel_path: str,
        meta: dict | None = None,
    ) -> Optional[int]:
        if not self.enabled or project_id is None:
            return None
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO artifacts (project_id, kind, title, rel_path, meta) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (project_id, kind, title, rel_path, json.dumps(meta or {})),
            )
            row = await cur.fetchone()
            return row["id"] if row else None

    async def list_artifacts(
        self, project_id: int, kind: str | None = None
    ) -> list[dict]:
        if not self.enabled:
            return []
        async with self.pool.connection() as conn:
            if kind:
                cur = await conn.execute(
                    "SELECT id, kind, title, rel_path, meta, created_at FROM artifacts "
                    "WHERE project_id=%s AND kind=%s ORDER BY created_at DESC",
                    (project_id, kind),
                )
            else:
                cur = await conn.execute(
                    "SELECT id, kind, title, rel_path, meta, created_at FROM artifacts "
                    "WHERE project_id=%s ORDER BY created_at DESC",
                    (project_id,),
                )
            return [dict(r) for r in await cur.fetchall()]
