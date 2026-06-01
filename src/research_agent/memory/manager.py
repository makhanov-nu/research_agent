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

    async def remember(
        self, channel_id: str, user_text: str, agent_text: str,
        cumulative_tokens: int = 0, source: str | None = None,
    ) -> None:
        """Persist one exchange: facts (mem0), activity, and an action log entry."""
        await self.episodic.touch_channel(channel_id, cumulative_tokens)
        await self.episodic.log_action(
            "exchange", user_text[:280], channel_id=channel_id
        )
        if self.semantic.enabled:
            await asyncio.to_thread(
                self.semantic.remember, user_text, agent_text, source
            )
