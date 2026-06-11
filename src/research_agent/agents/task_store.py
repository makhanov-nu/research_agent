"""Task registry: the cross-subagent dashboard (Postgres).

Every delegation becomes a task row tracking who was assigned, status, the final
result, and the FULL trace (reasoning + tool calls, start to end) in a separate
JSONB column for research/validation. Degrades to no-ops without a pool.

Quality labeling has three tiers (highest priority wins):
  1. User quality (``quality`` column) — set via ``!feedback <id> good|bad``.
  2. Auto quality (``auto_quality`` column) — derived at finish() time from the
     trace: all final verifier verdicts valid AND no missing citations → "good";
     any final verdict invalid → "bad"; otherwise NULL.
  3. Judge score (``judge_score`` column) — set by the batch LLM judge tool
     (research-agent-judge).  Score ≥ 4 → "good", ≤ 2 → "bad".
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
-- Auto-label columns: rule-based signal derived from the trace at finish time.
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS auto_quality TEXT;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS auto_signals JSONB;
-- LLM-judge columns: set by the batch research-agent-judge tool.
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS judge_score INT;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS judge_rationale TEXT;
"""


def derive_auto_label(trace: list) -> tuple[str | None, dict]:
    """Derive a rule-based quality label from a task's trace.

    Pure function — no I/O, safe to unit-test in isolation.

    Algorithm:
    - Collect the FINAL critique entry per verifier (highest round number).
    - Read missing_citations count from the LAST artifact entry.
    - All final verdicts "valid" AND missing_citations == 0  → "good".
    - Any final verdict "invalid"                            → "bad".
    - No signals at all (no critiques, no artifact)         → (None, {}).
    - "error" verdicts are ignored (treated as absent).

    Returns:
        (label, signals) where label is "good" | "bad" | None and signals is
        a dict like {"verdicts": {"citation_check": "valid"}, "missing_citations": 0}.
    """
    # Collect the final critique entry per verifier — highest round number wins.
    # Track (best_round, verdict) so we can compare cheaply in a single pass.
    _best: dict[str, tuple[int, str]] = {}  # verifier → (round, verdict)
    for entry in trace:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "critique":
            continue
        verifier = entry.get("verifier") or "unknown"
        verdict = entry.get("verdict") or "error"
        if verdict == "error":
            continue  # errors don't contribute a signal
        rnd = int(entry.get("round") or 0)
        prev = _best.get(verifier)
        if prev is None or rnd >= prev[0]:
            _best[verifier] = (rnd, verdict)
    final_by_verifier: dict[str, str] = {v: info[1] for v, info in _best.items()}

    # Read missing_citations from the last artifact entry.
    missing_citations: int | None = None
    for entry in reversed(trace):
        if isinstance(entry, dict) and entry.get("type") == "artifact":
            mc = entry.get("missing_citations")
            missing_citations = len(mc) if isinstance(mc, list) else 0
            break

    signals: dict = {}
    if final_by_verifier:
        signals["verdicts"] = dict(final_by_verifier)
    if missing_citations is not None:
        signals["missing_citations"] = missing_citations

    if not signals:
        return None, {}

    # Any invalid verdict → bad.
    if "invalid" in final_by_verifier.values():
        return "bad", signals

    # All verdicts valid (possibly empty dict) + missing_citations is 0 → good.
    all_valid = all(v == "valid" for v in final_by_verifier.values())
    no_missing = missing_citations is not None and missing_citations == 0
    has_signals = bool(final_by_verifier) or missing_citations is not None

    if has_signals and all_valid and no_missing:
        return "good", signals

    # Some verdicts valid but no artifact info, or missing_citations > 0.
    if missing_citations is not None and missing_citations > 0:
        return "bad", signals

    # verdicts present and all valid but no artifact data.
    if all_valid and final_by_verifier:
        return "good", signals

    return None, signals


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
        # Best-effort auto-label: never let labeling break the finish call.
        auto_quality: str | None = None
        auto_signals: dict | None = None
        try:
            auto_quality, auto_signals = derive_auto_label(trace)
        except Exception:  # noqa: BLE001
            logger.exception("derive_auto_label failed for task %s; skipping", task_id)
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE tasks SET status='done', result=%s, trace=%s, finished_at=now(), "
                "auto_quality=%s, auto_signals=%s "
                "WHERE id=%s",
                (
                    result,
                    json.dumps(trace),
                    auto_quality,
                    json.dumps(auto_signals) if auto_signals else None,
                    task_id,
                ),
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
            "SELECT id, agent, input, result, trace, quality, feedback, "
            "auto_quality, auto_signals, judge_score, judge_rationale, created_at "
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
