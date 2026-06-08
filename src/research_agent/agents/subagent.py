"""Wrap a specialized subagent as a single delegation tool.

The tool takes one `task` string, runs the subagent (via langchain's
`create_agent` + middleware) to completion in isolation, records the task and its
full trace to the task store, and returns ONLY the final result text. The
subagent's intermediate reasoning/tool-calls never reach the orchestrator — that
isolation is the token/memory win, and a TaskRecorderMiddleware persists the full
trace separately for research/validation.
"""

from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

logger = logging.getLogger(__name__)


def flatten_content(content) -> str:
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return content if isinstance(content, str) else str(content)


def _channel(config: RunnableConfig | None) -> str | None:
    if not config:
        return None
    return (config.get("configurable") or {}).get("thread_id")


async def run_subagent(
    *, system_prompt: str, tools, model, task: str, recursion_limit: int = 40,
    memory=None, agent_kind: str | None = None, channel_id: str | None = None,
    project: str | None = None,
) -> tuple[str, list]:
    """Run a fresh, traced subagent to completion; return (result, trace).

    Shared by the synchronous delegation tools and the background dispatcher. When
    `memory` + `agent_kind` are given, the task is primed with relevant past
    lessons before the run, and the finished job is reflected into new lessons
    afterward (in the background) — so the worker accumulates experience.
    """
    # Imported here so the package imports without langchain installed.
    from langchain.agents import create_agent

    from ..memory.lessons import prime_with_lessons, schedule_reflection
    from .middleware import TaskRecorderMiddleware

    primed = await prime_with_lessons(memory, agent_kind, task, project=project)
    recorder = TaskRecorderMiddleware()
    agent = create_agent(
        model, tools, system_prompt=system_prompt, middleware=[recorder]
    )
    state = await agent.ainvoke(
        {"messages": [("user", primed)]},
        config={"recursion_limit": recursion_limit},
    )
    result = flatten_content(state["messages"][-1].content)
    schedule_reflection(
        memory, agent_kind, task, result, channel_id=channel_id, project=project
    )
    return result, recorder.trace


def build_subagent_tool(
    *, name: str, description: str, system_prompt: str, tools, model,
    task_store=None, recursion_limit: int = 40, memory=None,
    agent_kind: str | None = None,
) -> BaseTool:
    """Build a delegation tool that runs a fresh, traced subagent per call.

    When `memory` is supplied the subagent learns: it's primed with past lessons
    (tagged `agent_kind`, defaulting to the tool name) and reflects each finished
    job into new ones.
    """
    kind = agent_kind or name

    @tool(name, description=description)
    async def _delegate(task: str, config: RunnableConfig = None) -> str:
        channel = _channel(config)
        task_id = None
        if task_store is not None:
            task_id = await task_store.create(name, task, channel)
            await task_store.mark_running(task_id)
        try:
            result, trace = await run_subagent(
                system_prompt=system_prompt, tools=tools, model=model,
                task=task, recursion_limit=recursion_limit,
                memory=memory, agent_kind=kind, channel_id=channel,
            )
        except Exception as exc:  # noqa: BLE001 — report failure up, record it
            logger.exception("Subagent %s failed", name)
            if task_store is not None:
                await task_store.fail(task_id, str(exc), [])
            return f"[{name} could not complete the task: {exc}]"

        if task_store is not None:
            await task_store.finish(task_id, result, trace)
        return result  # only the result crosses back to the orchestrator

    return _delegate
