"""Helper to wrap a specialized ReAct subagent as a single delegation tool.

The returned tool takes one `task` string, runs the subagent to completion in
isolation, and returns only its final text. The subagent's intermediate messages
never reach the orchestrator, which is what keeps the orchestrator's context (and
token cost) small.
"""

from __future__ import annotations

import logging

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)


def flatten_content(content) -> str:
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return content if isinstance(content, str) else str(content)


def build_subagent_tool(
    *, name: str, description: str, system_prompt: str, tools, llm,
    recursion_limit: int = 40,
) -> BaseTool:
    """Build a delegation tool that runs a fresh, stateless subagent per call."""
    agent = create_react_agent(llm, tools, prompt=system_prompt)

    @tool(name, description=description)
    async def _delegate(task: str) -> str:
        try:
            result = await agent.ainvoke(
                {"messages": [("user", task)]},
                config={"recursion_limit": recursion_limit},
            )
        except Exception as exc:  # noqa: BLE001 — report failure to the orchestrator
            logger.exception("Subagent %s failed", name)
            return f"[{name} could not complete the task: {exc}]"
        return flatten_content(result["messages"][-1].content)

    return _delegate
