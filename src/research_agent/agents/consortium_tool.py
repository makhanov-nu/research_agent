"""Delegation tool wrapping the ideation consortium.

Returns only the synthesized ideas to the orchestrator; the validated proposal is
saved into the project's council folder (for the methodology writer to pick up)
and the full transcript is saved to disk, while the task is recorded in the
dashboard — so the orchestrator's context stays small.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from .. import projects as projects_pkg
from ..consortium import capture_council
from ..projects import save_council_proposal


def _channel(config: RunnableConfig | None) -> str | None:
    if not config:
        return None
    return (config.get("configurable") or {}).get("thread_id")


def build_consortium_tool(consortium, task_store=None, projects=None, memory=None) -> BaseTool:
    @tool("brainstorm_research_ideas")
    async def brainstorm_research_ideas(
        topic: str, focus: str = "", config: RunnableConfig = None
    ) -> str:
        """Convene the multi-model consortium to debate and propose Q1-level
        research ideas on a topic (panel discusses in a shared session, then a
        chair synthesizes). Returns the synthesized ideas; the validated proposal
        is saved into the project's council folder for the methodology writer.
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

        # Save the ready-to-hand proposal into the project's council folder.
        project = await projects_pkg.resolve_project(projects, config)
        council_rel = await save_council_proposal(
            projects, project, topic, result["ideas"]
        )
        # Capture the session to memory so future ideation builds on it.
        await capture_council(
            memory, _channel(config), topic, result["ideas"], result["rel_path"]
        )

        out = result["ideas"]
        if council_rel:
            out += (
                f"\n\n(Validated proposal saved for methodology: `!getfile {council_rel}`)"
            )
        out += f"\n(Full transcript: `!getfile {result['rel_path']}`)"
        if task_store is not None:
            await task_store.finish(
                task_id, result["ideas"],
                [{"type": "transcript", "path": result["rel_path"]},
                 {"type": "council", "path": council_rel}],
            )
        return out

    return brainstorm_research_ideas
