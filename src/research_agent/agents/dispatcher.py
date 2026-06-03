"""Background task dispatcher: run subagents off the conversation turn.

The orchestrator submits jobs via the dispatch tools; each runs as a background
asyncio task (bounded by a concurrency limit), records its result + trace to the
task store, and posts the result to the originating channel when done. This lets
the orchestrator fan out parallel work and keep chatting while it runs.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

logger = logging.getLogger(__name__)

# The dispatcher is GLOBAL (one pool across all projects), so a runner is given
# the originating channel and routes its output into THAT channel's project.
# A runner maps (task, channel_id) -> (result, trace).
Runner = Callable[[str, "str | None"], Awaitable[tuple[str, list]]]
# Fired when a task reaches a terminal state. This is a pure TRIGGER: it carries
# only the task id, agent, status, and channel — NOT the result. The result and
# error already live in the task dashboard; the handler reads them back from
# there (single source of truth): (task_id, agent, status, channel_id).
OnComplete = Callable[[int, str, str, "str | None"], Awaitable[None]]


def build_runners(*, model, mcp_tools, writers, consortium, projects=None,
                  memory=None) -> dict[str, Runner]:
    """Assemble the dispatchable subagent runners from available resources.

    Runners are project-aware: each resolves the project from the originating
    channel and saves its artifact into that project's folder (registering it),
    so several projects can run agents concurrently without colliding.
    """
    from .literature import build_literature_runner

    runners: dict[str, Runner] = {}

    if mcp_tools:
        lit = build_literature_runner(model, mcp_tools)

        async def _literature(task: str, channel_id: str | None) -> tuple[str, list]:
            return await lit(task)  # research only; produces no saved artifact

        runners["research_literature"] = _literature

    def _rel(path: str) -> str:
        from pathlib import Path

        from ..config import settings

        try:
            return str(Path(path).resolve().relative_to(Path(settings.output_dir).resolve()))
        except ValueError:
            return Path(path).name

    def _artifact_runner(writer, label, kind):
        async def _run(task: str, channel_id: str | None) -> tuple[str, list]:
            project = await projects.ensure(channel_id) if projects is not None else None
            dirpath = None
            if projects is not None and project is not None:
                dirpath = projects.kind_dir(project["slug"], kind)
            r = await writer.draft(task, dirpath=dirpath)
            rel = _rel(r["tex_path"])
            tag = f" (project: {project['name']})" if project else ""
            if projects is not None and project is not None and project.get("id"):
                from pathlib import Path

                await projects.add_artifact(
                    project["id"], kind, Path(r["tex_path"]).stem, rel,
                    {"n_refs": r["n_refs"]},
                )
            summary = (
                f"Wrote a LaTeX {label} with {r['n_refs']} references{tag}: "
                f"`!getfile {rel}`"
            )
            return summary, [{"type": "artifact", "tex": r["tex_path"], "bib": r["bib_path"]}]

        return _run

    runners["literature_review"] = _artifact_runner(writers.reviewer, "literature review", "lit_review")
    runners["methodology"] = _artifact_runner(writers.methodologist, "methodology", "methodology")
    runners["paper_draft"] = _artifact_runner(writers.paper_writer, "paper draft", "paper")

    if consortium is not None:
        async def _consortium(task: str, channel_id: str | None) -> tuple[str, list]:
            from ..consortium import capture_council
            from ..projects import save_council_proposal

            r = await consortium.ideate(task)
            extra = []
            if projects is not None:
                project = await projects.ensure(channel_id)
                council_rel = await save_council_proposal(projects, project, task, r["ideas"])
                if council_rel:
                    extra = [{"type": "council", "path": council_rel}]
            # Capture to memory so future ideation builds on this session.
            await capture_council(memory, channel_id, task, r["ideas"], r["rel_path"])
            return r["ideas"], [{"type": "transcript", "path": r["rel_path"]}, *extra]

        runners["consortium"] = _consortium

    return runners


class TaskDispatcher:
    def __init__(self, runners: dict[str, Runner], task_store, on_complete: OnComplete,
                 max_parallel: int = 4):
        self._runners = runners
        self.task_store = task_store
        self._on_complete = on_complete
        self._sem = asyncio.Semaphore(max_parallel)
        self._running: dict[int, asyncio.Task] = {}

    @property
    def agents(self) -> list[str]:
        return list(self._runners)

    async def dispatch(self, agent: str, task: str, channel_id: str | None) -> int:
        if agent not in self._runners:
            raise ValueError(
                f"Unknown agent {agent!r}. Available: {', '.join(self.agents)}."
            )
        task_id = await self.task_store.create(agent, task, channel_id)
        bg = asyncio.create_task(self._run(task_id, agent, task, channel_id))
        if task_id is not None:
            self._running[task_id] = bg
        return task_id

    async def _run(self, task_id, agent, task, channel_id) -> None:
        async with self._sem:
            await self.task_store.mark_running(task_id)
            try:
                result, trace = await self._runners[agent](task, channel_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Dispatched task %s (%s) failed", task_id, agent)
                # Record the failure to the dashboard, THEN trigger the handler.
                await self.task_store.fail(task_id, str(exc), [])
                await self._fire(task_id, agent, "failed", channel_id)
                return
            finally:
                self._running.pop(task_id, None)
            # Write the result to the dashboard FIRST so the handler can read it.
            await self.task_store.finish(task_id, result, trace)
            await self._fire(task_id, agent, "done", channel_id)

    async def _fire(self, task_id, agent, status, channel_id) -> None:
        """Trigger the completion handler (never raises). Carries no result —
        the handler reads the result/error back from the task store."""
        try:
            await self._on_complete(task_id, agent, status, channel_id)
        except Exception:  # noqa: BLE001
            logger.exception("Task completion handler failed for #%s", task_id)

    async def join(self) -> None:
        """Await all in-flight tasks (used in tests / graceful shutdown)."""
        if self._running:
            await asyncio.gather(*self._running.values(), return_exceptions=True)

    async def shutdown(self) -> None:
        for t in list(self._running.values()):
            t.cancel()


def build_dispatch_tools(dispatcher: TaskDispatcher) -> list[BaseTool]:
    agents = ", ".join(dispatcher.agents) or "(none)"

    def _channel(config: RunnableConfig | None) -> str | None:
        if not config:
            return None
        return (config.get("configurable") or {}).get("thread_id")

    @tool(
        "dispatch_task",
        description=(
            "Run a subagent in the BACKGROUND so you can keep talking instead of "
            f"waiting. `agent` must be one of: {agents}. `task` is a COMPLETE, "
            "self-contained instruction (for consortium/literature_review, the "
            "topic). Returns a task id immediately and runs asynchronously. You do "
            "NOT poll for the result: when the task finishes, its result is "
            "delivered back to you automatically as a '[BACKGROUND TASK COMPLETE]' "
            "event for you to process. Use this for heavy or multiple parallel "
            "jobs (dispatch several to run them at once); for a quick single "
            "lookup, call the direct tool instead."
        ),
    )
    async def dispatch_task(agent: str, task: str, config: RunnableConfig = None) -> str:
        try:
            task_id = await dispatcher.dispatch(agent, task, _channel(config))
        except ValueError as exc:
            return str(exc)
        return (
            f"Dispatched task #{task_id} to `{agent}` in the background. "
            f"Its result will be delivered to you automatically when it finishes."
        )

    return [dispatch_task]
