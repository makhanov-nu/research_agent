"""Wires the three memory stores together behind one interface.

The graph and the Discord bot talk to a single MemoryManager. Synchronous mem0
calls are pushed to worker threads so they never block the event loop.
"""

from __future__ import annotations

import asyncio
import logging

from ..config import settings
from .episodic import EpisodicStore
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

    async def setup(self) -> None:
        await self.episodic.setup()
        await self.procedural.setup()
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
        self, query: str, limit: int | None = None, *, kind: str | None = None
    ) -> str:
        """Recall consolidated lessons relevant to a task (e.g. past failures).

        `kind` scopes recall to one agent's lessons (e.g. "literature") so a
        worker is primed with its own past experience, not every agent's.
        """
        if not (query and self.semantic.enabled):
            return ""
        limit = limit or settings.lesson_recall_limit
        return await asyncio.to_thread(
            self.semantic.recall, query, limit, "lesson", kind
        )

    async def reflect_and_record(
        self, agent_kind: str, task: str, result: str, *,
        channel_id: str | None = None, project: str | None = None,
    ) -> int:
        """Distill durable lessons from a finished job and store them.

        Runs a cheap reflection model over (task -> result); each extracted lesson
        is recorded (episodic + semantic, tagged with `agent_kind` and `project`)
        so future `recall_lessons(kind=agent_kind)` surfaces it. Returns the number
        of lessons stored. Best-effort: never raises.
        """
        if not self.semantic.enabled or not (task and result):
            return 0
        from langchain_core.messages import HumanMessage, SystemMessage

        from ..llm import build_reflection_llm

        n_max = settings.reflection_max_lessons
        system = _REFLECT_SYSTEM.format(kind=agent_kind, n=n_max)
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
                lesson, kind=agent_kind, channel_id=channel_id, project=project,
            )
            n += 1
        return n

    async def record_lesson(
        self, text: str, *, kind: str, channel_id: str | None = None,
        status: str | None = None, project: str | None = None,
    ) -> None:
        """Persist a durable lesson to episodic (action log) + semantic (mem0).

        `kind` groups lessons (e.g. "experiment", "council"); the semantic copy is
        tagged type=lesson so recall_lessons can retrieve it for future runs.
        """
        try:
            await self.episodic.log_action(
                f"lesson_{kind}", text[:280], channel_id=channel_id,
                metadata={"status": status, "project": project},
            )
            if self.semantic.enabled:
                meta = {"type": "lesson", "kind": kind}
                if status:
                    meta["status"] = status
                if project:
                    meta["project"] = project
                await asyncio.to_thread(
                    self.semantic.add_fact, text, f"lesson:{kind}", meta
                )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to record %s lesson", kind)

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
