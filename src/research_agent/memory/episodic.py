"""Episodic memory: the agent's lab notebook (Postgres).

Stores *experiences* — what the agent did and what happened:

- agent_actions     : a log of notable actions (searches, reads, summaries)
- conversations     : per-channel activity + rolling summary + archive state
- experiments       : the experiment registry (config, metrics, status, artifacts)

All writes degrade to no-ops when no pool is configured, so the rest of the
system runs unchanged without a database.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Columns that update_experiment is allowed to write, and which of them are
# JSONB (so values must be json-encoded). Anything outside this set is rejected
# before any SQL is built, so untrusted field names can never reach the query.
UPDATABLE_EXPERIMENT_FIELDS = frozenset(
    {"title", "hypothesis", "config", "code_ref", "dataset",
     "status", "metrics", "artifacts", "notes"}
)
JSON_EXPERIMENT_FIELDS = frozenset({"config", "metrics", "artifacts"})

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    channel_id     TEXT PRIMARY KEY,
    last_activity  TIMESTAMPTZ NOT NULL DEFAULT now(),
    summary        TEXT NOT NULL DEFAULT '',
    archived       BOOLEAN NOT NULL DEFAULT FALSE,
    total_tokens   BIGINT NOT NULL DEFAULT 0,
    last_nudge_tokens BIGINT NOT NULL DEFAULT 0,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_actions (
    id          BIGSERIAL PRIMARY KEY,
    channel_id  TEXT,
    kind        TEXT NOT NULL,
    summary     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_actions_channel_idx ON agent_actions (channel_id, created_at);

CREATE TABLE IF NOT EXISTS experiments (
    id          BIGSERIAL PRIMARY KEY,
    channel_id  TEXT,
    title       TEXT NOT NULL,
    hypothesis  TEXT,
    config      JSONB NOT NULL DEFAULT '{}'::jsonb,
    code_ref    TEXT,
    dataset     TEXT,
    status      TEXT NOT NULL DEFAULT 'planned',
    metrics     JSONB NOT NULL DEFAULT '{}'::jsonb,
    artifacts   JSONB NOT NULL DEFAULT '[]'::jsonb,
    notes       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS experiments_channel_idx ON experiments (channel_id, created_at);
"""


class EpisodicStore:
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
        logger.info("Episodic store schema ready.")

    # --- conversation activity / archival state -------------------------------

    async def touch_channel(self, channel_id: str, total_tokens: int) -> None:
        """Record that a channel was just used (un-archives it)."""
        if not self.enabled:
            return
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO conversations (channel_id, last_activity, total_tokens, archived)
                VALUES (%s, now(), %s, FALSE)
                ON CONFLICT (channel_id) DO UPDATE
                  SET last_activity = now(),
                      total_tokens = EXCLUDED.total_tokens,
                      archived = FALSE
                """,
                (channel_id, total_tokens),
            )

    async def get_conversation(self, channel_id: str) -> Optional[dict]:
        if not self.enabled:
            return None
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM conversations WHERE channel_id = %s", (channel_id,)
            )
            return await cur.fetchone()

    async def set_summary(self, channel_id: str, summary: str) -> None:
        if not self.enabled:
            return
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO conversations (channel_id, summary)
                VALUES (%s, %s)
                ON CONFLICT (channel_id) DO UPDATE SET summary = EXCLUDED.summary
                """,
                (channel_id, summary),
            )

    async def set_last_nudge_tokens(self, channel_id: str, tokens: int) -> None:
        if not self.enabled:
            return
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO conversations (channel_id, last_nudge_tokens)
                VALUES (%s, %s)
                ON CONFLICT (channel_id) DO UPDATE
                  SET last_nudge_tokens = EXCLUDED.last_nudge_tokens
                """,
                (channel_id, tokens),
            )

    async def list_idle_channels(self, idle_days: int) -> list[dict]:
        """Active (non-archived) channels with no activity for idle_days."""
        if not self.enabled:
            return []
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT * FROM conversations
                WHERE archived = FALSE
                  AND last_activity < now() - make_interval(days => %s)
                """,
                (idle_days,),
            )
            return await cur.fetchall()

    async def mark_archived(self, channel_id: str, summary: str) -> None:
        if not self.enabled:
            return
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                UPDATE conversations
                   SET archived = TRUE, summary = %s
                 WHERE channel_id = %s
                """,
                (summary, channel_id),
            )

    # --- action log -----------------------------------------------------------

    async def log_action(
        self, kind: str, summary: str, channel_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        async with self.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO agent_actions (channel_id, kind, summary, metadata) "
                "VALUES (%s, %s, %s, %s)",
                (channel_id, kind, summary, json.dumps(metadata or {})),
            )

    async def recent_actions(self, since: datetime, limit: int = 200) -> list[dict]:
        if not self.enabled:
            return []
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM agent_actions WHERE created_at >= %s "
                "ORDER BY created_at DESC LIMIT %s",
                (since, limit),
            )
            return await cur.fetchall()

    # --- experiment registry --------------------------------------------------

    async def create_experiment(
        self, title: str, channel_id: str | None = None, hypothesis: str = "",
        config: dict | None = None, dataset: str = "", code_ref: str = "",
    ) -> Optional[int]:
        if not self.enabled:
            return None
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                INSERT INTO experiments (channel_id, title, hypothesis, config, dataset, code_ref)
                VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
                """,
                (channel_id, title, hypothesis, json.dumps(config or {}), dataset, code_ref),
            )
            row = await cur.fetchone()
            return row["id"] if row else None

    async def update_experiment(self, experiment_id: int, **fields) -> None:
        # Validate before anything else (and before the enabled short-circuit)
        # so unknown columns are always rejected, never silently ignored.
        unknown = set(fields) - UPDATABLE_EXPERIMENT_FIELDS
        if unknown:
            raise ValueError(
                f"Unknown experiment field(s): {', '.join(sorted(unknown))}. "
                f"Allowed: {', '.join(sorted(UPDATABLE_EXPERIMENT_FIELDS))}."
            )
        if not self.enabled or not fields:
            return
        sets, values = [], []
        for col, val in fields.items():
            sets.append(f"{col} = %s")
            values.append(json.dumps(val) if col in JSON_EXPERIMENT_FIELDS else val)
        values.append(experiment_id)
        async with self.pool.connection() as conn:
            await conn.execute(
                f"UPDATE experiments SET {', '.join(sets)}, updated_at = now() WHERE id = %s",
                values,
            )

    async def list_experiments(self, channel_id: str | None = None, limit: int = 20) -> list[dict]:
        if not self.enabled:
            return []
        async with self.pool.connection() as conn:
            if channel_id is None:
                cur = await conn.execute(
                    "SELECT * FROM experiments ORDER BY created_at DESC LIMIT %s", (limit,)
                )
            else:
                cur = await conn.execute(
                    "SELECT * FROM experiments WHERE channel_id = %s "
                    "ORDER BY created_at DESC LIMIT %s",
                    (channel_id, limit),
                )
            return await cur.fetchall()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
