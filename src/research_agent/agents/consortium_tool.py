"""Delegation tool wrapping the ideation consortium.

Returns only the synthesized ideas to the orchestrator; the full multi-model
transcript is saved to disk (retrievable via !getfile) and the task is recorded
in the dashboard, so the orchestrator's context stays small.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool


def _channel(config: RunnableConfig | None) -> str | None:
    if not config:
        return None
    return (config.get("configurable") or {}).get("thread_id")


def build_consortium_tool(consortium, task_store=None) -> BaseTool:
    @tool("brainstorm_research_ideas")
    async def brainstorm_research_ideas(
        topic: str, focus: str = "", config: RunnableConfig = None
    ) -> str:
        """Convene the multi-model consortium to debate and propose Q1-level
        research ideas on a topic (panel discusses in a shared session, then a
        chair synthesizes). Returns the synthesized ideas; the full transcript is
        saved to disk.
        """
        task_id = None
        if task_store is not None:
            task_id = await task_store.create("consortium", topic, _channel(config))
            await task_store.mark_running(task_id)
        try:
            result = await consortium.ideate(topic, focus=focus)
        except Exception as exc:  # noqa: BLE001
            if task_store is not None:
                await task_store.fail(task_id, str(exc), [])
            return f"[consortium could not complete: {exc}]"

        out = (
            result["ideas"]
            + f"\n\n(Full shared-session transcript saved: {result['rel_path']})"
        )
        if task_store is not None:
            await task_store.finish(
                task_id, result["ideas"],
                [{"type": "transcript", "path": result["rel_path"]}],
            )
        return out

    return brainstorm_research_ideas
