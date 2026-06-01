"""Local terminal REPL for testing the agent without Discord.

    python -m research_agent.cli

Uses the same durable checkpointer + memory as the bot when DATABASE_URL is set,
otherwise falls back to in-process state.
"""

from __future__ import annotations

import asyncio
import logging

from .agent import build_graph
from .db import build_checkpointer, open_pool
from .memory.manager import MemoryManager


async def _run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    pool = await open_pool()
    checkpointer = await build_checkpointer(pool)
    memory = None
    if pool is not None:
        memory = MemoryManager(pool)
        await memory.setup()

    graph = await build_graph(checkpointer, memory)
    config = {"configurable": {"thread_id": "cli"}}
    print("Research agent ready. Type a message (Ctrl-D / 'exit' to quit).\n")

    try:
        while True:
            try:
                user = input("you> ").strip()
            except EOFError:
                break
            if user.lower() in {"exit", "quit"}:
                break
            if not user:
                continue
            result = await graph.ainvoke({"messages": [("user", user)]}, config=config)
            reply = result["messages"][-1].content
            if isinstance(reply, list):
                reply = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in reply
                )
            print(f"\nagent> {reply}\n")
            if memory is not None:
                await memory.remember("cli", user, reply, result.get("cumulative_tokens", 0))
    finally:
        if pool is not None:
            await pool.close()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
