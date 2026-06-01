"""Procedural memory: instructions the agent has learned to follow.

This is the editable layer on top of the static system prompt — durable
preferences and reusable procedures ("how the user likes things done", "how to
run an HF fine-tune"). Retrieved each turn and prepended to the system prompt.

Stored in Postgres so it survives restarts and can be consolidated over time.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS procedures (
    id          BIGSERIAL PRIMARY KEY,
    kind        TEXT NOT NULL DEFAULT 'preference',  -- preference | procedure
    content     TEXT NOT NULL,
    weight      INT NOT NULL DEFAULT 1,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class ProceduralMemory:
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
        logger.info("Procedural store schema ready.")

    async def add(self, content: str, kind: str = "preference") -> None:
        if not self.enabled:
            return
        async with self.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO procedures (kind, content) VALUES (%s, %s)",
                (kind, content),
            )

    async def instructions_block(self, limit: int = 25) -> str:
        """Return learned instructions formatted for the system prompt."""
        if not self.enabled:
            return ""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT kind, content FROM procedures "
                "ORDER BY weight DESC, created_at DESC LIMIT %s",
                (limit,),
            )
            rows = await cur.fetchall()
        if not rows:
            return ""
        return "\n".join(f"- ({r['kind']}) {r['content']}" for r in rows)
