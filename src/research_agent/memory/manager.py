"""Wires the three memory stores together behind one interface.

The graph and the Discord bot talk to a single MemoryManager. Synchronous mem0
calls are pushed to worker threads so they never block the event loop.
"""

from __future__ import annotations

import asyncio
import logging

from ..config import settings
from .episodic import EpisodicStore
from .lesson_stats import LessonStats
from .procedural import ProceduralMemory
from .semantic import SemanticMemory

logger = logging.getLogger(__name__)

_REFLECT_SYSTEM = (
    "You are the reflection step of a self-improving research agent. A '{kind}' "
    "subagent just finished a job. From the TASK and its RESULT, extract at most "
    "{n} DURABLE, reusable lessons that would make the NEXT '{kind}' job better: "
    "sources/datasets/methods that worked, pitfalls to avoid, effective structure "
    "or phrasing, and any apparent user preferences. Each lesson must be one "
    "self-contained sentence that GENERALIZES beyond this specific task — not a "
    "summary of this result. If nothing is durably useful, reply with exactly "
    "NONE. Output one lesson per line, no numbering or preamble."
)

# Used when the outcome was explicitly bad: extract pitfalls/causes so the NEXT
# job avoids them, not lessons about what worked.
_REFLECT_BAD_SYSTEM = (
    "You are the reflection step of a self-improving research agent. A '{kind}' "
    "subagent just finished a job that was judged POOR or INVALID. From the TASK "
    "and its RESULT, extract at most {n} DURABLE pitfalls or failure causes that "
    "the NEXT '{kind}' job should AVOID. Each pitfall must be one self-contained "
    "sentence phrased as a warning (e.g. 'Avoid X because Y') that GENERALIZES "
    "beyond this specific task. If nothing is durably useful, reply with NONE. "
    "Output one pitfall per line, no numbering or preamble."
)


def _flatten(content) -> str:
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return content if isinstance(content, str) else str(content)


class MemoryManager:
    def __init__(self, pool):
        self.pool = pool
        self.semantic = SemanticMemory()
        self.episodic = EpisodicStore(pool)
        self.procedural = ProceduralMemory(pool)
        self.lesson_stats = LessonStats(pool)

    async def setup(self) -> None:
        await self.episodic.setup()
        await self.procedural.setup()
        await self.lesson_stats.setup()
        # mem0 init is synchronous and does I/O; keep it off the loop.
        await asyncio.to_thread(self.semantic.setup)

    async def build_context(self, query: str) -> str:
        """Assemble recalled facts + learned procedures for the system prompt."""
        sections: list[str] = []

        procedures = await self.procedural.instructions_block()
        if procedures:
            sections.append("Learned preferences & procedures:\n" + procedures)

        if query and self.semantic.enabled:
            facts = await asyncio.to_thread(self.semantic.recall, query)
            if facts:
                sections.append("Relevant facts from long-term memory:\n" + facts)

        return "\n\n".join(sections)

    async def recall_lessons(
        self,
        query: str,
        limit: int | None = None,
        *,
        kind: str | None = None,
    ) -> str:
        """Recall consolidated lessons relevant to a task (e.g. past failures).

        `kind` scopes recall to one agent's lessons (e.g. "literature") so a
        worker is primed with its own past experience, not every agent's.

        Returns only the formatted text block; lesson ids are discarded here.
        Callers that need ids for credit assignment should use
        recall_lessons_with_ids().
        """
        text, _ = await self.recall_lessons_with_ids(query, limit, kind=kind)
        return text

    async def recall_lessons_with_ids(
        self,
        query: str,
        limit: int | None = None,
        *,
        kind: str | None = None,
    ) -> tuple[str, list[str]]:
        """Like recall_lessons() but also returns the mem0 ids of recalled items.

        The ids let the caller credit the lessons after the job finishes (via
        credit_lessons). Re-ranks the raw vector candidates with a
        Laplace-smoothed quality prior from LessonStats before returning, so
        lessons from consistently good jobs surface ahead of noisy ones.

        The re-ranking is a SOFT prior only (equal-weight blend of vector
        relevance and quality score) — it never hard-deletes anything.
        """
        if not (query and self.semantic.enabled):
            return "", []
        limit = limit or settings.lesson_recall_limit
        # Over-fetch so re-ranking has candidates to work with.
        oversample = settings.lesson_recall_oversample
        fetch_limit = limit * oversample

        # Get candidate ids to look up scores (we need them for re-ranking).
        _, candidate_ids = await asyncio.to_thread(
            self.semantic.recall_with_ids,
            query, fetch_limit, "lesson", kind,
        )
        score_map = await self.lesson_stats.get_scores(candidate_ids)

        # Now fetch again with the score_map so recall_with_ids can re-rank.
        text_block, ids = await asyncio.to_thread(
            self.semantic.recall_with_ids,
            query, limit, "lesson", kind, score_map,
        )
        # Record usage for the ids we actually surfaced.
        if ids:
            await self.lesson_stats.record_used(ids, kind=kind or "")
        return text_block, ids

    async def credit_lessons(
        self, lesson_ids: list[str], outcome: str
    ) -> None:
        """Increment good/bad counts for a set of previously recalled lessons.

        Called after a job finishes with a known outcome so that lessons
        contributing to good jobs float up in future re-ranking.  This is a
        NOISY signal — all recalled lessons get the same credit regardless of
        individual contribution — and is intentionally only used for re-ranking,
        never for hard deletion.
        """
        if not lesson_ids or outcome not in ("good", "bad"):
            return
        try:
            await self.lesson_stats.credit(lesson_ids, outcome)
        except Exception:  # noqa: BLE001
            logger.exception("credit_lessons failed for outcome=%s", outcome)

    async def reflect_and_record(
        self,
        agent_kind: str,
        task: str,
        result: str,
        *,
        channel_id: str | None = None,
        project: str | None = None,
        outcome: str | None = None,
    ) -> int:
        """Distill durable lessons from a finished job and store them.

        Runs a cheap reflection model over (task -> result); each extracted lesson
        is recorded (episodic + semantic, tagged with `agent_kind` and `project`)
        so future `recall_lessons(kind=agent_kind)` surfaces it. Returns the number
        of lessons stored. Best-effort: never raises.

        `outcome` ("good" | "bad" | None) controls which prompt variant is used:
        - "bad"  → pitfall/avoid-phrasing prompt so future jobs steer clear
        - "good" / None → normal what-worked prompt (existing behavior)
        The outcome is also stored in each lesson's metadata for filtering.
        """
        if not self.semantic.enabled or not (task and result):
            return 0
        from langchain_core.messages import HumanMessage, SystemMessage

        from ..llm import build_reflection_llm

        n_max = settings.reflection_max_lessons
        if outcome == "bad":
            system_tmpl = _REFLECT_BAD_SYSTEM
        else:
            system_tmpl = _REFLECT_SYSTEM
        system = system_tmpl.format(kind=agent_kind, n=n_max)
        human = f"TASK:\n{task[:2000]}\n\nRESULT:\n{result[:4000]}"
        try:
            resp = await build_reflection_llm().ainvoke(
                [SystemMessage(content=system), HumanMessage(content=human)]
            )
        except Exception:  # noqa: BLE001
            logger.exception("Reflection LLM call failed for %s", agent_kind)
            return 0

        text = _flatten(resp.content).strip()
        if not text or text.upper() == "NONE":
            return 0
        n = 0
        for line in text.splitlines():
            if n >= n_max:  # honor the cap (incl. 0) before persisting anything
                break
            lesson = line.strip(" \t-•*").rstrip()
            if len(lesson) < 8 or lesson.upper() == "NONE":
                continue
            await self.record_lesson(
                lesson, kind=agent_kind, channel_id=channel_id,
                project=project, outcome=outcome,
            )
            n += 1
        return n

    async def record_lesson(
        self,
        text: str,
        *,
        kind: str,
        channel_id: str | None = None,
        status: str | None = None,
        project: str | None = None,
        outcome: str | None = None,
    ) -> str | None:
        """Persist a durable lesson to episodic (action log) + semantic (mem0).

        `kind` groups lessons (e.g. "experiment", "council"); the semantic copy is
        tagged type=lesson so recall_lessons can retrieve it for future runs.
        `outcome` ("good"|"bad"|None) is stored in metadata for filtering and
        stats; the lesson text itself uses pitfall phrasing when outcome="bad".

        Returns the mem0 memory id on success (None if disabled or error).
        """
        mem_id: str | None = None
        try:
            await self.episodic.log_action(
                f"lesson_{kind}", text[:280], channel_id=channel_id,
                metadata={"status": status, "project": project, "outcome": outcome},
            )
            if self.semantic.enabled:
                meta: dict = {"type": "lesson", "kind": kind}
                if status:
                    meta["status"] = status
                if project:
                    meta["project"] = project
                if outcome:
                    meta["outcome"] = outcome
                mem_id = await asyncio.to_thread(
                    self.semantic.add_fact, text, f"lesson:{kind}", meta
                )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to record %s lesson", kind)
        return mem_id

    async def log_experience(
        self, kind: str, summary: str, channel_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Log a notable subagent experience (council session, experiment outcome)."""
        try:
            await self.episodic.log_action(kind, summary[:500], channel_id, metadata)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to log experience %s", kind)

    async def remember(
        self, channel_id: str, user_text: str, agent_text: str,
        cumulative_tokens: int = 0, source: str | None = None,
        summary: str | None = None,
    ) -> None:
        """Persist one exchange: activity, durable summary, facts, action log.

        Runs as a fire-and-forget background task, so it logs and swallows its
        own failures rather than surfacing them as unobserved task exceptions.
        """
        try:
            await self.episodic.touch_channel(channel_id, cumulative_tokens)
            # Keep the durable episodic summary in sync with auto-summarization
            # so idle archival never reads an empty summary.
            if summary:
                await self.episodic.set_summary(channel_id, summary)
            await self.episodic.log_action(
                "exchange", user_text[:280], channel_id=channel_id
            )
            if self.semantic.enabled:
                await asyncio.to_thread(
                    self.semantic.remember, user_text, agent_text, source
                )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Background memory persistence failed for channel %s", channel_id
            )
