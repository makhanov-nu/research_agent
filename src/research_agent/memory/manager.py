"""Wires the three memory stores together behind one interface.

The graph and the Discord bot talk to a single MemoryManager. Synchronous mem0
calls are pushed to worker threads so they never block the event loop.
"""

from __future__ import annotations

import asyncio
import logging

from .episodic import EpisodicStore
from .procedural import ProceduralMemory
from .semantic import SemanticMemory

logger = logging.getLogger(__name__)


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

    async def recall_lessons(self, query: str, limit: int = 5) -> str:
        """Recall consolidated lessons relevant to a task (e.g. past failures)."""
        if not (query and self.semantic.enabled):
            return ""
        return await asyncio.to_thread(
            self.semantic.recall, query, limit, "lesson"
        )

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
