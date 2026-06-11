"""Lesson usage and outcome statistics stored in Postgres.

Tracks how often each mem0 lesson is recalled and whether the jobs that used it
turned out good or bad — giving us a noisy soft prior for re-ranking recall.

WHY: The lesson store grows monotonically. Without any signal about which lessons
actually helped, top-K recall degrades as noise accumulates. This table gives us
a lightweight quality signal: Laplace-smoothed good/(good+bad) re-ranks lessons
so better-performing ones float up. It is *deliberately* noisy — we inject 5
lessons per job with no per-lesson attribution, so the counts are a loose
aggregate, not causal attribution. The score is only used to re-rank recall, never
to hard-delete lessons.

Schema:
    lesson_stats (lesson_id TEXT PRIMARY KEY, kind TEXT, times_used INT,
                  good INT, bad INT, last_used TIMESTAMPTZ)

The `kind` column mirrors the mem0 lesson's `kind` metadata so maintenance
consolidation can scope counts per agent kind.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS lesson_stats (
    lesson_id   TEXT PRIMARY KEY,
    kind        TEXT NOT NULL DEFAULT '',
    times_used  INT NOT NULL DEFAULT 0,
    good        INT NOT NULL DEFAULT 0,
    bad         INT NOT NULL DEFAULT 0,
    last_used   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS lesson_stats_kind_idx ON lesson_stats (kind);
"""


class LessonStats:
    """Postgres-backed lesson quality counters.

    All methods are best-effort (never raise) and no-op when pool is None,
    so callers can use this unconditionally regardless of DB availability.
    """

    def __init__(self, pool):
        self.pool = pool

    @property
    def enabled(self) -> bool:
        return self.pool is not None

    async def setup(self) -> None:
        if not self.enabled:
            return
        try:
            async with self.pool.connection() as conn:
                await conn.execute(SCHEMA)
            logger.info("LessonStats schema ready.")
        except Exception:  # noqa: BLE001
            logger.exception("LessonStats schema setup failed.")

    async def record_used(self, lesson_ids: list[str], kind: str = "") -> None:
        """Increment times_used for lessons that were returned to a caller."""
        if not self.enabled or not lesson_ids:
            return
        try:
            async with self.pool.connection() as conn:
                for lid in lesson_ids:
                    await conn.execute(
                        """
                        INSERT INTO lesson_stats (lesson_id, kind, times_used, last_used)
                        VALUES (%s, %s, 1, now())
                        ON CONFLICT (lesson_id) DO UPDATE
                          SET times_used = lesson_stats.times_used + 1,
                              last_used  = now()
                        """,
                        (lid, kind),
                    )
        except Exception:  # noqa: BLE001
            logger.exception("LessonStats.record_used failed.")

    async def credit(self, lesson_ids: list[str], outcome: str) -> None:
        """Increment good or bad counts for a set of recalled lessons.

        `outcome` should be "good" or "bad"; anything else is ignored silently.
        This is a NOISY signal — all recalled lessons get the same credit even
        though the outcome is a job-level signal, not lesson-level.
        """
        if not self.enabled or not lesson_ids or outcome not in ("good", "bad"):
            return
        # Explicit allowlist mapping outcome -> column, so the f-string SQL
        # below can never interpolate anything else.
        col = {"good": "good", "bad": "bad"}[outcome]
        try:
            async with self.pool.connection() as conn:
                for lid in lesson_ids:
                    await conn.execute(
                        f"""
                        INSERT INTO lesson_stats (lesson_id, kind, {col})
                        VALUES (%s, '', 1)
                        ON CONFLICT (lesson_id) DO UPDATE
                          SET {col} = lesson_stats.{col} + 1
                        """,
                        (lid,),
                    )
        except Exception:  # noqa: BLE001
            logger.exception("LessonStats.credit failed (outcome=%s).", outcome)

    async def get_scores(self, lesson_ids: list[str]) -> dict[str, float]:
        """Return Laplace-smoothed quality score for each id.

        Score = (good + 1) / (good + bad + 2) — neutral 0.5 when unused.
        Lessons absent from the table are treated as unused (score = 0.5).
        """
        if not self.enabled or not lesson_ids:
            return {lid: 0.5 for lid in lesson_ids}
        try:
            async with self.pool.connection() as conn:
                placeholders = ", ".join(["%s"] * len(lesson_ids))
                cur = await conn.execute(
                    f"SELECT lesson_id, good, bad FROM lesson_stats "
                    f"WHERE lesson_id IN ({placeholders})",
                    lesson_ids,
                )
                rows = await cur.fetchall()
        except Exception:  # noqa: BLE001
            logger.exception("LessonStats.get_scores failed.")
            return {lid: 0.5 for lid in lesson_ids}

        scores: dict[str, float] = {lid: 0.5 for lid in lesson_ids}
        for row in rows or []:
            g = row["good"] if isinstance(row, dict) else row[1]
            b = row["bad"] if isinstance(row, dict) else row[2]
            lid = row["lesson_id"] if isinstance(row, dict) else row[0]
            scores[lid] = (g + 1) / (g + b + 2)
        return scores

    async def aggregate_onto(
        self, target_id: str, source_ids: list[str], kind: str = ""
    ) -> None:
        """Merge stats from source_ids onto target_id (used after consolidation).

        After the LLM replaces N near-duplicate lessons with one merged lesson,
        we sum their good/bad/times_used counts onto the merged lesson's row and
        delete the source rows, so the quality signal is preserved.
        """
        if not self.enabled or not source_ids:
            return
        try:
            async with self.pool.connection() as conn:
                placeholders = ", ".join(["%s"] * len(source_ids))
                cur = await conn.execute(
                    f"SELECT COALESCE(SUM(times_used),0), COALESCE(SUM(good),0), "
                    f"COALESCE(SUM(bad),0) FROM lesson_stats "
                    f"WHERE lesson_id IN ({placeholders})",
                    source_ids,
                )
                row = await cur.fetchone()
                tu = row[0] if row else 0
                g = row[1] if row else 0
                b = row[2] if row else 0
                await conn.execute(
                    """
                    INSERT INTO lesson_stats (lesson_id, kind, times_used, good, bad)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (lesson_id) DO UPDATE
                      SET times_used = lesson_stats.times_used + EXCLUDED.times_used,
                          good       = lesson_stats.good       + EXCLUDED.good,
                          bad        = lesson_stats.bad        + EXCLUDED.bad
                    """,
                    (target_id, kind, int(tu), int(g), int(b)),
                )
                await conn.execute(
                    f"DELETE FROM lesson_stats WHERE lesson_id IN ({placeholders})",
                    source_ids,
                )
        except Exception:  # noqa: BLE001
            logger.exception("LessonStats.aggregate_onto failed.")
