"""Discord client that bridges messages to the research agent graph.

The bot replies in DMs and when @-mentioned in a server channel. Each Discord
channel maps to its own conversation thread (LangGraph checkpoint), so context
is preserved per channel.
"""

from __future__ import annotations

import logging

import discord
from langgraph.checkpoint.memory import MemorySaver

from ..agent import build_graph

logger = logging.getLogger(__name__)

DISCORD_MAX_CHARS = 2000


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


class ResearchBot(discord.Client):
    def __init__(self, **kwargs):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, **kwargs)
        self.graph = None

    async def setup_hook(self) -> None:
        # Build the graph (loads MCP tools) once, before the gateway connects.
        self.graph = await build_graph(MemorySaver())
        logger.info("Research agent graph ready.")

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
        if self.user:  # strip the bot mention from the text
            for token in (f"<@{self.user.id}>", f"<@!{self.user.id}>"):
                content = content.replace(token, "")
        content = content.strip()
        if not content:
            await message.channel.send("Hi — what are we researching?")
            return

        thread_id = str(message.channel.id)
        config = {"configurable": {"thread_id": thread_id}}

        try:
            async with message.channel.typing():
                result = await self.graph.ainvoke(
                    {"messages": [("user", content)]}, config=config
                )
            reply = result["messages"][-1].content
            if isinstance(reply, list):  # some models return content blocks
                reply = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in reply
                )
        except Exception:  # noqa: BLE001 — surface failures to the user
            logger.exception("Error handling message")
            reply = "Something went wrong while I was thinking. Check the logs."

        for chunk in _chunk(reply):
            await message.channel.send(chunk)
