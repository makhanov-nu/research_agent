"""Agent middleware for subagents.

`TaskRecorderMiddleware` captures the full run — every model message (reasoning +
content) and every tool call/result — into a structured trace via the after_agent
hook, for persistence to the task store. The orchestrator never receives this
trace; it only gets the final result string (see build_subagent_tool).

This is also the seam for other cross-cutting concerns the codebase should grow
into: before_model/after_model (token accounting, trimming), wrap_tool_call
(per-tool policy), and HumanInTheLoopMiddleware for tool approval (HITL).
"""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware

from .subagent import flatten_content


def serialize_messages(messages) -> list[dict]:
    """Turn an agent's message history into a JSON-able step-by-step trace."""
    trace: list[dict] = []
    for m in messages:
        entry: dict = {
            "type": getattr(m, "type", m.__class__.__name__),
            "content": flatten_content(getattr(m, "content", "")),
        }
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = [
                {"name": tc.get("name"), "args": tc.get("args")} for tc in tool_calls
            ]
        if getattr(m, "name", None):
            entry["name"] = m.name
        if getattr(m, "tool_call_id", None):
            entry["tool_call_id"] = m.tool_call_id
        trace.append(entry)
    return trace


class TaskRecorderMiddleware(AgentMiddleware):
    """Records the full message trace at the end of an agent run.

    A fresh instance is used per delegation so `trace` is isolated (safe for
    concurrent/parallel subagent runs).
    """

    def __init__(self):
        super().__init__()
        self.trace: list[dict] = []

    def _capture(self, state) -> None:
        self.trace = serialize_messages(state.get("messages", []))

    def after_agent(self, state, runtime):  # sync path
        self._capture(state)
        return None

    async def aafter_agent(self, state, runtime):  # async path
        self._capture(state)
        return None
