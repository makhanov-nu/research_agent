"""Discord client that bridges messages to the research agent graph.

The bot replies in DMs and when @-mentioned in a server channel. Each Discord
channel maps to its own conversation thread (LangGraph checkpoint). On startup
it opens the Postgres pool (if configured), builds a durable checkpointer + the
memory manager, and launches the background maintenance loop.

Commands:
  !checkpoint / !summarize  - summarize this thread to long-term memory + reset
  !remember <text>          - store a durable preference/instruction
  !help                     - show commands
"""

from __future__ import annotations

import asyncio
import logging

import discord

from ..agent import build_graph
from ..config import settings
from ..db import build_checkpointer, open_pool
from ..llm import get_llm
from ..memory.maintenance import run_loop
from ..memory.manager import MemoryManager
from ..memory.summarize import summarize_messages

logger = logging.getLogger(__name__)

DISCORD_MAX_CHARS = 2000

HELP_TEXT = (
    "**Commands**\n"
    "`!checkpoint` (or `!summarize`) — summarize this thread to long-term "
    "memory and reset the live context\n"
    "`!remember <text>` — store a durable preference/instruction\n"
    "`!help` — this message\n\n"
    "Otherwise, just talk to me — DM or @-mention."
)


def _chunk(text: str, size: int = DISCORD_MAX_CHARS) -> list[str]:
    """Split text into Discord-sized chunks, preferring line boundaries."""
    if not text:
        return ["(empty response)"]

    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        while len(line) > size:  # a single very long line
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:size])
            line = line[size:]
        if len(current) + len(line) > size:
            chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return chunks


def _flatten(content) -> str:
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return content if isinstance(content, str) else str(content)


class ResearchBot(discord.Client):
    def __init__(self, **kwargs):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, **kwargs)
        self.graph = None
        self.memory: MemoryManager | None = None
        self.llm = None
        self._pool = None
        self._maintenance_task: asyncio.Task | None = None

    async def setup_hook(self) -> None:
        self.llm = get_llm()
        self._pool = await open_pool()
        checkpointer = await build_checkpointer(self._pool)

        if settings.memory_enabled and self._pool is not None:
            self.memory = MemoryManager(self._pool)
            await self.memory.setup()
            self._maintenance_task = self.loop.create_task(
                run_loop(self.memory, self.llm)
            )

        self.graph = await build_graph(checkpointer, self.memory)
        logger.info("Research agent ready (memory=%s).", bool(self.memory))

    async def close(self) -> None:
        if self._maintenance_task:
            self._maintenance_task.cancel()
        if self._pool is not None:
            await self._pool.close()
        await super().close()

    async def on_ready(self) -> None:
        logger.info("Logged in as %s (id=%s)", self.user, self.user.id)

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user or message.author.bot:
            return

        is_dm = message.guild is None
        mentioned = self.user in message.mentions
        if not (is_dm or mentioned):
            return

        content = message.content
        if self.user:
            for token in (f"<@{self.user.id}>", f"<@!{self.user.id}>"):
                content = content.replace(token, "")
        content = content.strip()

        thread_id = str(message.channel.id)
        config = {"configurable": {"thread_id": thread_id}}

        if content.startswith("!"):
            await self._handle_command(message, content, config)
            return
        if not content:
            await message.channel.send("Hi — what are we researching?")
            return

        await self._handle_chat(message, content, config, thread_id)

    async def _handle_chat(self, message, content, config, thread_id) -> None:
        try:
            async with message.channel.typing():
                result = await self.graph.ainvoke(
                    {"messages": [("user", content)]}, config=config
                )
            reply = _flatten(result["messages"][-1].content)
            cumulative = result.get("cumulative_tokens", 0)
        except Exception:  # noqa: BLE001
            logger.exception("Error handling message")
            await message.channel.send(
                "Something went wrong while I was thinking. Check the logs."
            )
            return

        for chunk in _chunk(reply):
            await message.channel.send(chunk)

        # Persist the exchange to memory without blocking the reply.
        if self.memory is not None:
            self.loop.create_task(
                self.memory.remember(thread_id, content, reply, cumulative)
            )

    async def _handle_command(self, message, content, config) -> None:
        cmd, _, arg = content[1:].partition(" ")
        cmd = cmd.lower()

        if cmd == "help":
            await message.channel.send(HELP_TEXT)
        elif cmd in {"checkpoint", "summarize"}:
            await self._checkpoint(message, config)
        elif cmd == "remember":
            arg = arg.strip()
            if not arg:
                await message.channel.send("Usage: `!remember <text>`")
            elif self.memory is not None:
                await self.memory.procedural.add(arg, kind="preference")
                await message.channel.send("Noted — I'll remember that.")
            else:
                await message.channel.send("Memory isn't configured, so I can't store that.")
        else:
            await message.channel.send(f"Unknown command `!{cmd}`.\n\n{HELP_TEXT}")

    async def _checkpoint(self, message, config) -> None:
        thread_id = config["configurable"]["thread_id"]
        snapshot = await self.graph.aget_state(config)
        messages = snapshot.values.get("messages", []) if snapshot else []
        if not messages:
            await message.channel.send("Nothing to checkpoint yet.")
            return

        async with message.channel.typing():
            summary = await summarize_messages(
                self.llm, messages, snapshot.values.get("summary", "")
            )
            from langchain_core.messages import RemoveMessage

            removes = [RemoveMessage(id=m.id) for m in messages if getattr(m, "id", None)]
            await self.graph.aupdate_state(config, {"summary": summary, "messages": removes})

            if self.memory is not None:
                await self.memory.episodic.set_summary(thread_id, summary)
                await asyncio.to_thread(
                    self.memory.semantic.remember,
                    "Conversation checkpoint:",
                    summary,
                    f"channel:{thread_id}",
                )

        await message.channel.send(
            "Checkpointed. Summary saved to long-term memory and the live "
            "context was reset."
        )
