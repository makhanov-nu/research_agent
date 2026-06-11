"""Assemble the orchestrator's delegation tools from available resources.

This is the single place to register subagents. To add a new specialized agent,
write its builder and append its delegation tool here.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from ..writing.tools import build_writing_tools
from .code_reader import build_code_reader_tool
from .consortium_tool import build_consortium_tool
from .literature import build_literature_agent_tool


def build_delegated_tools(
    *, llm, mcp_tools, writers, experiment_runner=None, consortium=None,
    task_store=None, projects=None, memory=None,
) -> list[BaseTool]:
    tools: list[BaseTool] = []

    # Literature research subagent (owns the paperclip/MCP tools so the
    # orchestrator never sees raw search output). `memory` makes it learn.
    if mcp_tools:
        tools.append(
            build_literature_agent_tool(llm, mcp_tools, task_store, memory, projects)
        )
        # Code reader subagent: fetches and analyses GitHub repositories.
        tools.append(
            build_code_reader_tool(llm, mcp_tools, task_store, memory, projects)
        )

    # LaTeX writers: literature review, methodology, paper draft (each a subagent).
    tools += build_writing_tools(
        writers, task_store=task_store, projects=projects, memory=memory
    )

    # Multi-model ideation consortium.
    if consortium is not None:
        tools.append(
            build_consortium_tool(
                consortium, task_store=task_store, projects=projects, memory=memory
            )
        )

    # Experiment tools (lightweight, return concise status strings).
    if experiment_runner is not None and getattr(experiment_runner, "enabled", False):
        from ..config import settings
        from ..experiments.coder import ExperimentCoder
        from ..experiments.tools import build_experiment_tools

        coder = None
        if settings.openrouter_api_key:
            from ..llm import build_openrouter_chat

            coder = ExperimentCoder(
                build_openrouter_chat(
                    settings.experiment_coder_model, temperature=0.2, max_tokens=16384
                )
            )
        tools += build_experiment_tools(experiment_runner, coder=coder)

    return tools
