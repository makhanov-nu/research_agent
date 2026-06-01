"""Postgres connection pool and the durable LangGraph checkpointer.

A single async pool is shared by the checkpointer (working memory) and the
episodic store. psycopg3 needs autocommit + dict rows for the checkpointer's
migrations and reads.
"""

from __future__ import annotations

import logging

from .config import settings

logger = logging.getLogger(__name__)


async def open_pool():
    """Open and return an AsyncConnectionPool, or None if no DB is configured."""
    if not settings.memory_enabled:
        return None

    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        max_size=20,
        open=False,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    await pool.open()
    logger.info("Opened Postgres connection pool.")
    return pool


async def build_checkpointer(pool):
    """Build a durable Postgres checkpointer over an open pool, else in-memory."""
    if pool is None:
        from langgraph.checkpoint.memory import MemorySaver

        logger.warning("No database configured; using in-memory checkpointer.")
        return MemorySaver()

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    saver = AsyncPostgresSaver(pool)
    await saver.setup()  # idempotent: creates checkpoint tables if missing
    return saver
