"""Assemble the orchestrator's delegation tools from available resources.

This is the single place to register subagents. To add a new specialized agent,
write its builder and append its delegation tool here.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from ..writing.tools import build_writing_tools
from .consortium_tool import build_consortium_tool
from .literature import build_literature_agent_tool


def build_delegated_tools(
    *, llm, mcp_tools, reviewer, experiment_runner=None, consortium=None,
    task_store=None,
) -> list[BaseTool]:
    tools: list[BaseTool] = []

    # Literature research subagent (owns the paperclip/MCP tools so the
    # orchestrator never sees raw search output).
    if mcp_tools:
        tools.append(build_literature_agent_tool(llm, mcp_tools, task_store))

    # LaTeX literature-review writer (itself a subagent).
    tools += build_writing_tools(reviewer, task_store=task_store)

    # Multi-model ideation consortium.
    if consortium is not None:
        tools.append(build_consortium_tool(consortium, task_store=task_store))

    # Experiment tools (lightweight, return concise status strings).
    if experiment_runner is not None and getattr(experiment_runner, "enabled", False):
        from ..experiments.tools import build_experiment_tools

        tools += build_experiment_tools(experiment_runner)

    return tools
