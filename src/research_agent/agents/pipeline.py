"""Linear per-project research pipelines.

A pipeline is an ordered list of (agent, task) stages that run one after the
other, where each stage automatically receives the previous stage's result via
input_tasks piping.  The orchestrator creates a pipeline once and then stands
back; the dispatcher advances it stage-by-stage as each task completes.

Why not a DAG?  The overwhelming majority of research flows are linear:
literature review → methodology → paper.  A full DAG scheduler would
complicate the state machine significantly for a use-case that has never been
requested.  Branches can be modelled as separate pipelines.

Database schema follows the same ALTER-safe setup() pattern used by TaskStore
and ProjectStore so it can be applied on top of an existing schema without
manual migration steps.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS pipelines (
    id            SERIAL PRIMARY KEY,
    channel_id    TEXT,
    name          TEXT NOT NULL,
    stages        JSONB NOT NULL DEFAULT '[]'::jsonb,
    current_stage INT NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'queued',  -- queued|running|failed|done
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS pipelines_channel_idx ON pipelines (channel_id, created_at DESC);
CREATE INDEX IF NOT EXISTS pipelines_status_idx  ON pipelines (status);
-- Safe to re-run on an existing install.
ALTER TABLE pipelines ADD COLUMN IF NOT EXISTS channel_id TEXT;
ALTER TABLE pipelines ADD COLUMN IF NOT EXISTS name TEXT;
"""


class PipelineStore:
    """Postgres-backed store for linear stage pipelines.

    Degrades gracefully when pool is None: all mutating methods return None or
    False; ``enabled`` is False.  The dispatcher checks this before calling any
    pipeline helpers.
    """

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
        logger.info("Pipeline store schema ready.")

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    async def create(
        self,
        name: str,
        stages: list[dict],
        channel_id: str | None = None,
    ) -> Optional[int]:
        """Insert a new pipeline row and return its id."""
        if not self.enabled:
            return None
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO pipelines (name, stages, channel_id, status, current_stage) "
                "VALUES (%s, %s, %s, 'queued', 0) RETURNING id",
                (name, json.dumps(stages), channel_id),
            )
            row = await cur.fetchone()
            return row["id"] if row else None

    async def set_status(self, pipeline_id: int, status: str) -> None:
        if not self.enabled:
            return
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE pipelines SET status=%s WHERE id=%s",
                (status, pipeline_id),
            )

    async def advance(self, pipeline_id: int, next_stage: int, task_id: int) -> None:
        """Record that ``next_stage`` has been dispatched as ``task_id``.

        This updates the JSONB stages array in-place using jsonb_set so we never
        lose the other stage entries.
        """
        if not self.enabled:
            return
        async with self.pool.connection() as conn:
            # Fetch the current stages, patch, and write back.
            cur = await conn.execute(
                "SELECT stages FROM pipelines WHERE id=%s", (pipeline_id,)
            )
            row = await cur.fetchone()
            if row is None:
                return
            stages: list = list(row["stages"])
            if next_stage < len(stages):
                stages[next_stage] = {**stages[next_stage], "task_id": task_id}
            await conn.execute(
                "UPDATE pipelines SET stages=%s, current_stage=%s, status='running' "
                "WHERE id=%s",
                (json.dumps(stages), next_stage, pipeline_id),
            )

    async def record_stage_task(
        self, pipeline_id: int, stage_index: int, task_id: int
    ) -> None:
        """Store the task_id for a specific stage slot (used for stage 0)."""
        await self.advance(pipeline_id, stage_index, task_id)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    async def get(self, pipeline_id: int) -> Optional[dict]:
        if not self.enabled:
            return None
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id, channel_id, name, stages, current_stage, status, created_at "
                "FROM pipelines WHERE id=%s",
                (pipeline_id,),
            )
            row = await cur.fetchone()
            return dict(row) if row else None

    async def find_by_task(self, task_id: int) -> Optional[dict]:
        """Return the pipeline that owns ``task_id`` (any stage), or None."""
        if not self.enabled:
            return None
        async with self.pool.connection() as conn:
            # JSONB containment search: stages array has an element whose
            # task_id equals the given int.
            cur = await conn.execute(
                "SELECT id, channel_id, name, stages, current_stage, status "
                "FROM pipelines "
                "WHERE stages @> %s::jsonb",
                (json.dumps([{"task_id": task_id}]),),
            )
            row = await cur.fetchone()
            return dict(row) if row else None


# ---------------------------------------------------------------------------
# Pipeline advancement helpers — called by TaskDispatcher._run after finish/fail
# ---------------------------------------------------------------------------

async def on_stage_success(
    pipeline: dict,
    completed_task_id: int,
    dispatch_fn,            # async (agent, task, channel_id, input_tasks) -> int
) -> None:
    """Advance a pipeline after one of its stages succeeds.

    ``dispatch_fn`` is a coroutine that accepts (agent, task, channel_id,
    input_tasks) and returns the new task id.  We pass it as a parameter so
    pipeline.py has no direct import dependency on TaskDispatcher (avoids a
    circular import).
    """
    store: PipelineStore = pipeline["_store"]
    stages: list[dict] = list(pipeline["stages"])
    current: int = pipeline["current_stage"]
    next_stage = current + 1
    channel_id = pipeline.get("channel_id")
    pipeline_id: int = pipeline["id"]

    if next_stage >= len(stages):
        # All stages done.
        await store.set_status(pipeline_id, "done")
        logger.info("Pipeline #%d completed all %d stages.", pipeline_id, len(stages))
        return

    stage = stages[next_stage]
    try:
        new_task_id = await dispatch_fn(
            stage["agent"],
            stage["task"],
            channel_id,
            [completed_task_id],   # input_tasks
        )
        await store.advance(pipeline_id, next_stage, new_task_id)
        logger.info(
            "Pipeline #%d: advanced to stage %d (task #%d).",
            pipeline_id, next_stage, new_task_id,
        )
    except Exception:
        logger.exception(
            "Pipeline #%d: failed to dispatch stage %d — marking pipeline failed.",
            pipeline_id, next_stage,
        )
        await store.set_status(pipeline_id, "failed")


async def on_stage_failure(pipeline: dict) -> None:
    """Mark the pipeline failed when one of its stages fails."""
    store: PipelineStore = pipeline["_store"]
    pipeline_id: int = pipeline["id"]
    await store.set_status(pipeline_id, "failed")
    logger.info(
        "Pipeline #%d halted after stage %d failed.",
        pipeline_id, pipeline["current_stage"],
    )
