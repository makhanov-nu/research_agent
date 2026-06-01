"""Local terminal REPL for testing the agent without Discord.

    python -m research_agent.cli
"""

from __future__ import annotations

import asyncio
import logging

from langgraph.checkpoint.memory import MemorySaver

from .agent import build_graph


async def _run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    graph = await build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "cli"}}
    print("Research agent ready. Type a message (Ctrl-D / 'exit' to quit).\n")

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


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
