"""Task registry: the cross-subagent dashboard (Postgres).

Every delegation becomes a task row tracking who was assigned, status, the final
result, and the FULL trace (reasoning + tool calls, start to end) in a separate
JSONB column for research/validation. Degrades to no-ops without a pool.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id           BIGSERIAL PRIMARY KEY,
    parent_id    BIGINT REFERENCES tasks(id),
    channel_id   TEXT,
    agent        TEXT NOT NULL,
    input        TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',  -- pending|running|done|failed|cancelled
    result       TEXT,
    trace        JSONB NOT NULL DEFAULT '[]'::jsonb,
    error        TEXT,
    quality      TEXT,            -- user verdict: good|bad (the training label)
    feedback     TEXT,            -- the user's correction/note, if any
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at   TIMESTAMPTZ,
    finished_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS tasks_channel_idx ON tasks (channel_id, created_at DESC);
CREATE INDEX IF NOT EXISTS tasks_status_idx ON tasks (status);
-- Backfill the label columns on installs created before this change.
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS quality TEXT;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS feedback TEXT;
"""


class TaskStore:
    def __init__(self, pool):
        self.pool = pool

    @property
    def enabled(self) -> bool:
        return self.pool is not None

    async def setup(self) -> None:
        if not self.enabled:
            return
        async with self.pool.connection() as conn:
            await conn.execute(SCHEMA)
        logger.info("Task store schema ready.")

    async def create(self, agent: str, input: str, channel_id: str | None = None,
                     parent_id: int | None = None) -> Optional[int]:
        if not self.enabled:
            return None
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO tasks (agent, input, channel_id, parent_id, status) "
                "VALUES (%s, %s, %s, %s, 'pending') RETURNING id",
                (agent, input, channel_id, parent_id),
            )
            row = await cur.fetchone()
            return row["id"] if row else None

    async def mark_running(self, task_id: int | None) -> None:
        if not self.enabled or task_id is None:
            return
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE tasks SET status='running', started_at=now() WHERE id=%s",
                (task_id,),
            )

    async def finish(self, task_id: int | None, result: str, trace: list) -> None:
        if not self.enabled or task_id is None:
            return
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE tasks SET status='done', result=%s, trace=%s, finished_at=now() "
                "WHERE id=%s",
                (result, json.dumps(trace), task_id),
            )

    async def fail(self, task_id: int | None, error: str, trace: list) -> None:
        if not self.enabled or task_id is None:
            return
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE tasks SET status='failed', error=%s, trace=%s, finished_at=now() "
                "WHERE id=%s",
                (error, json.dumps(trace), task_id),
            )

    async def get(self, task_id: int) -> Optional[dict]:
        if not self.enabled:
            return None
        async with self.pool.connection() as conn:
            cur = await conn.execute("SELECT * FROM tasks WHERE id=%s", (task_id,))
            return await cur.fetchone()

    async def set_feedback(
        self, task_id: int, quality: str, feedback: str | None = None
    ) -> bool:
        """Attach a user verdict (good|bad) + optional note to a task.

        This is the training-quality label: it turns the logged (input, result,
        trace) into a labeled example for later fine-tuning/distillation, and
        lets you filter the corpus to only the trajectories you approved.
        """
        if not self.enabled:
            return False
        quality = (quality or "").lower()
        if quality not in {"good", "bad"}:
            raise ValueError(f"quality must be 'good' or 'bad', got {quality!r}")
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "UPDATE tasks SET quality=%s, feedback=%s WHERE id=%s RETURNING id",
                (quality, feedback, task_id),
            )
            return await cur.fetchone() is not None

    async def list_for_export(
        self, *, agents=None, quality=None, since=None, limit: int = 100_000
    ) -> list[dict]:
        """Completed, result-bearing tasks for the training exporter.

        Optional filters: `agents` (roles), `quality` (e.g. ("good",)), and
        `since` (created_at lower bound). Returns oldest-first so exports are
        stable/append-friendly.
        """
        if not self.enabled:
            return []
        clauses = ["status = 'done'", "result IS NOT NULL", "result <> ''"]
        params: list = []
        if agents:
            clauses.append("agent = ANY(%s)")
            params.append(list(agents))
        if quality:
            clauses.append("quality = ANY(%s)")
            params.append(list(quality))
        if since:
            clauses.append("created_at >= %s")
            params.append(since)
        params.append(limit)
        sql = (
            "SELECT id, agent, input, result, trace, quality, feedback, created_at "
            "FROM tasks WHERE " + " AND ".join(clauses) +
            " ORDER BY created_at, id LIMIT %s"
        )
        async with self.pool.connection() as conn:
            cur = await conn.execute(sql, params)
            return await cur.fetchall()

    async def list_recent(self, limit: int = 15, channel_id: str | None = None) -> list[dict]:
        if not self.enabled:
            return []
        async with self.pool.connection() as conn:
            if channel_id is None:
                cur = await conn.execute(
                    "SELECT id, agent, status, input, channel_id, created_at FROM tasks "
                    "ORDER BY created_at DESC LIMIT %s", (limit,)
                )
            else:
                cur = await conn.execute(
                    "SELECT id, agent, status, input, channel_id, created_at FROM tasks "
                    "WHERE channel_id=%s ORDER BY created_at DESC LIMIT %s",
                    (channel_id, limit),
                )
            return await cur.fetchall()
