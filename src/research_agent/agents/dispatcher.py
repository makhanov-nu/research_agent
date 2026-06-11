"""Background task dispatcher: run subagents off the conversation turn.

The orchestrator submits jobs via the dispatch tools; each runs as a background
asyncio task (bounded by a concurrency limit), records its result + trace to the
task store, and posts the result to the originating channel when done. This lets
the orchestrator fan out parallel work and keep chatting while it runs.

Each artifact runner applies a bounded draft → critique → revise loop via
review_loop.run_review_loop.  Citation gaps are caught by a rule-based check
(no LLM cost); the methodology writer additionally runs the LLM methodology
validator; the paper writer additionally runs the paper claims verifier.  Every
verifier pass is appended to the task trace as a ``{"type": "critique", ...}``
entry, giving us preference-pair training data (rejected draft + critique +
accepted revision).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from ..config import settings

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
                  memory=None, task_store=None, model_for_role=None) -> dict[str, Runner]:
    """Assemble the dispatchable subagent runners from available resources.

    Runners are project-aware: each resolves the project from the originating
    channel and saves its artifact into that project's folder (registering it),
    so several projects can run agents concurrently without colliding.

    Args:
        model: The default LLM for roles without overrides.
        model_for_role: Optional callable(role: str) -> BaseChatModel to resolve
            per-role models. Defaults to a lambda returning model for all roles.
    """
    if model_for_role is None:
        model_for_role = lambda role: model  # noqa: E731

    from .code_reader import build_code_reader_runner
    from .literature import build_literature_runner

    runners: dict[str, Runner] = {}

    if mcp_tools:
        lit = build_literature_runner(model_for_role("research_literature"), mcp_tools,
                                      memory=memory)

        async def _literature(task: str, channel_id: str | None) -> tuple[str, list]:
            return await lit(task, channel_id)  # research only; no saved artifact

        runners["research_literature"] = _literature

        code = build_code_reader_runner(model_for_role("code_reader"), mcp_tools,
                                        memory=memory)

        async def _code_reader(task: str, channel_id: str | None) -> tuple[str, list]:
            return await code(task, channel_id)  # analysis only; no saved artifact

        runners["code_reader"] = _code_reader

    def _rel(path: str) -> str:
        from pathlib import Path

        from ..config import settings

        try:
            return str(Path(path).resolve().relative_to(Path(settings.output_dir).resolve()))
        except ValueError:
            return Path(path).name

    def _artifact_runner(writer, label, kind, agent) -> Runner:
        """Build a runner that drafts, then applies a citation-critique loop."""
        async def _run(task: str, channel_id: str | None) -> tuple[str, list]:
            from .review_loop import citation_critique, run_review_loop

            project = await projects.ensure(channel_id) if projects is not None else None
            proj_slug = project["slug"] if project else None
            dirpath = None
            if projects is not None and project is not None:
                dirpath = projects.kind_dir(project["slug"], kind)
            # Prime with relevant lessons from past jobs of this kind.
            lessons = ""
            if memory is not None and settings.lessons_enabled:
                try:
                    lessons = await memory.recall_lessons(task, kind=agent)
                except Exception:  # noqa: BLE001 — recall must not break the job
                    logger.exception("Lesson recall failed for %s", agent)

            full_trace: list = []

            # Wrap writer.draft so the review loop can re-invoke it with a
            # revised task while keeping dirpath and lessons constant.
            async def _draft(*, task: str) -> dict:
                return await writer.draft(task, dirpath=dirpath, lessons=lessons)

            r = await run_review_loop(
                original_task=task,
                draft_fn=_draft,
                critique_fn=citation_critique,
                trace=full_trace,
            )

            rel = _rel(r["tex_path"])
            tag = f" (project: {project['name']})" if project else ""
            if projects is not None and project is not None and project.get("id"):
                from pathlib import Path

                await projects.add_artifact(
                    project["id"], kind, Path(r["tex_path"]).stem, rel,
                    {"n_refs": r["n_refs"]},
                )
            missing = r.get("missing_citations") or []
            warn = (
                f" ⚠ {len(missing)} undefined cite(s): "
                f"{', '.join(missing[:8])}{'…' if len(missing) > 8 else ''}"
                if missing else ""
            )
            summary = (
                f"Wrote a LaTeX {label} with {r['n_refs']} references{tag}: "
                f"`!getfile {rel}`{warn}"
            )
            # Record the writer's full reasoning/tool-call trace to the dashboard,
            # with critique entries and the saved artifact appended.
            full_trace = (r.get("trace") or []) + full_trace + [
                {"type": "artifact", "tex": r["tex_path"], "bib": r["bib_path"],
                 "missing_citations": missing}
            ]
            # Reflect the finished draft into durable lessons (background).
            if memory is not None:
                from ..memory.lessons import schedule_reflection

                schedule_reflection(
                    memory, agent, task, r.get("latex", ""),
                    channel_id=channel_id, project=proj_slug,
                )
            return summary, full_trace

        return _run

    runners["literature_review"] = _artifact_runner(writers.reviewer, "literature review", "lit_review", "literature_review")

    # paper_draft gets an extra paper_verifier pass on top of citation critique.
    async def _paper_draft(task: str, channel_id: str | None) -> tuple[str, list]:
        from .review_loop import citation_critique, run_review_loop

        project = await projects.ensure(channel_id) if projects is not None else None
        proj_slug = project["slug"] if project else None
        dirpath = projects.kind_dir(project["slug"], "paper") if (
            projects is not None and project is not None) else None

        lessons = ""
        if memory is not None and settings.lessons_enabled:
            try:
                lessons = await memory.recall_lessons(task, kind="paper_draft")
            except Exception:  # noqa: BLE001
                logger.exception("Lesson recall failed for paper_draft")

        full_trace: list = []

        async def _draft(*, task: str) -> dict:
            return await writers.paper_writer.draft(task, dirpath=dirpath, lessons=lessons)

        r = await run_review_loop(
            original_task=task,
            draft_fn=_draft,
            critique_fn=citation_critique,
            trace=full_trace,
        )

        # Paper claims verifier — LLM-based, records its own task row.
        if mcp_tools and settings.validation_rounds >= 1:
            from pathlib import Path

            from .paper_verifier import verify_paper

            paper_text = r.get("latex") or Path(r["tex_path"]).read_text(errors="replace")
            # material was injected into the task string; pass the original task
            # as brief so the verifier knows what was requested.
            if task_store is not None:
                pv_task_id = None
                try:
                    pv_task_id = await task_store.create("paper_verifier", task, channel_id)
                    await task_store.mark_running(pv_task_id)
                    is_valid, feedback = await verify_paper(
                        model_for_role("paper_verifier"), mcp_tools, brief=task,
                        material="", paper_text=paper_text, memory=memory,
                    )
                    pv_result = (
                        "VALID — no fabricated claims detected."
                        if is_valid else
                        f"Issues found:\n{feedback}"
                    )
                    await task_store.finish(pv_task_id, pv_result, [])
                    full_trace.append({
                        "type": "critique",
                        "round": 1,
                        "verifier": "paper_verifier",
                        "verdict": "valid" if is_valid else "invalid",
                        "feedback": feedback,
                        "superseded_draft": None if is_valid else r.get("latex", ""),
                    })
                    if not is_valid and settings.validation_rounds >= 2:
                        from .review_loop import _REVISION_PREFIX
                        revision_task = _REVISION_PREFIX.format(
                            feedback=feedback, task=task
                        )
                        try:
                            r = await _draft(task=revision_task)
                        except Exception:  # noqa: BLE001
                            logger.exception(
                                "Paper revision after verifier failed; keeping draft"
                            )
                except Exception:  # noqa: BLE001
                    logger.exception("Paper verification failed")
                    if task_store is not None and pv_task_id is not None:
                        await task_store.fail(pv_task_id, "Verifier raised an exception", [])
            else:
                # No task_store — inline verdict only.
                try:
                    is_valid, feedback = await verify_paper(
                        model_for_role("paper_verifier"), mcp_tools, brief=task,
                        material="", paper_text=paper_text, memory=memory,
                    )
                    full_trace.append({
                        "type": "critique",
                        "round": 1,
                        "verifier": "paper_verifier",
                        "verdict": "valid" if is_valid else "invalid",
                        "feedback": feedback,
                        "superseded_draft": None if is_valid else r.get("latex", ""),
                    })
                except Exception:  # noqa: BLE001
                    logger.exception("Paper verification failed (no task_store)")

        rel = _rel(r["tex_path"])
        tag = f" (project: {project['name']})" if project else ""
        if projects is not None and project is not None and project.get("id"):
            from pathlib import Path

            await projects.add_artifact(
                project["id"], "paper", Path(r["tex_path"]).stem, rel,
                {"n_refs": r["n_refs"]},
            )
        missing = r.get("missing_citations") or []
        warn = (
            f" ⚠ {len(missing)} undefined cite(s): "
            f"{', '.join(missing[:8])}{'…' if len(missing) > 8 else ''}"
            if missing else ""
        )
        summary = (
            f"Wrote a LaTeX paper draft with {r['n_refs']} references{tag}: "
            f"`!getfile {rel}`{warn}"
        )
        full_trace = (r.get("trace") or []) + full_trace + [
            {"type": "artifact", "tex": r["tex_path"], "bib": r["bib_path"],
             "missing_citations": missing}
        ]
        if memory is not None:
            from ..memory.lessons import schedule_reflection
            schedule_reflection(
                memory, "paper_draft", task, r.get("latex", ""),
                channel_id=channel_id, project=proj_slug,
            )
        return summary, full_trace

    runners["paper_draft"] = _paper_draft

    async def _methodology(task: str, channel_id: str | None) -> tuple[str, list]:
        from pathlib import Path

        from .methodology_validator import validate_methodology
        from .review_loop import citation_critique, run_review_loop

        project = await projects.ensure(channel_id) if projects is not None else None
        proj_slug = project["slug"] if project else None
        dirpath = projects.kind_dir(project["slug"], "methodology") if (
            projects is not None and project is not None) else None

        lessons = ""
        if memory is not None and settings.lessons_enabled:
            try:
                lessons = await memory.recall_lessons(task, kind="methodology")
            except Exception:  # noqa: BLE001
                logger.exception("Lesson recall failed for methodology")

        full_trace: list = []

        async def _draft(*, task: str) -> dict:
            return await writers.methodologist.draft(task, dirpath=dirpath, lessons=lessons)

        # Citation pass first (rule-based, cheap).
        r = await run_review_loop(
            original_task=task,
            draft_fn=_draft,
            critique_fn=citation_critique,
            trace=full_trace,
        )

        # LLM methodology validator — each pass is its own tracked dashboard task.
        # Track how many validator tasks we create so the summary string is accurate.
        if mcp_tools and settings.validation_rounds >= 1:
            methodology_text = r.get("latex") or Path(r["tex_path"]).read_text(errors="replace")
            last_val_task_id = None
            revisions_done = 0
            final_is_valid = False

            for val_round in range(1, settings.validation_rounds + 1):
                val_task_id = None
                try:
                    if task_store is not None:
                        val_task_id = await task_store.create(
                            "methodology_validator", task, channel_id
                        )
                        await task_store.mark_running(val_task_id)
                    is_valid, feedback = await validate_methodology(
                        model_for_role("methodology_validator"), mcp_tools, task,
                        methodology_text, memory=memory,
                    )
                    val_result = (
                        "VALID — methodology is sound and addresses the original task."
                        if is_valid else
                        f"Issues found:\n{feedback}"
                    )
                    if task_store is not None and val_task_id is not None:
                        await task_store.finish(val_task_id, val_result, [])
                    last_val_task_id = val_task_id
                    final_is_valid = is_valid

                    full_trace.append({
                        "type": "critique",
                        "round": val_round,
                        "verifier": "methodology_validator",
                        "verdict": "valid" if is_valid else "invalid",
                        "feedback": feedback,
                        "superseded_draft": None if is_valid else r.get("latex", ""),
                    })

                    if is_valid or val_round >= settings.validation_rounds:
                        break

                    # Issues found and rounds remain — revise.
                    from .review_loop import _REVISION_PREFIX
                    revision_task = _REVISION_PREFIX.format(feedback=feedback, task=task)
                    try:
                        r = await _draft(task=revision_task)
                        methodology_text = r.get("latex") or Path(r["tex_path"]).read_text(errors="replace")
                        revisions_done += 1
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "Methodology revision failed on round %d; keeping draft", val_round
                        )
                        break

                except Exception:  # noqa: BLE001
                    logger.exception("Methodology validation round %d failed", val_round)
                    if task_store is not None and val_task_id is not None:
                        await task_store.fail(
                            val_task_id, "Validator encountered an error", []
                        )
                    break  # validator error — accept current draft

            # Build the suffix that goes into the main task summary.
            if final_is_valid:
                if revisions_done == 0:
                    val_suffix = " ✓ validated"
                else:
                    val_suffix = f" ✓ validated after {revisions_done} revision{'s' if revisions_done != 1 else ''}"
            elif last_val_task_id is not None:
                val_suffix = f" (see validator task #{last_val_task_id} for issues)"
            else:
                val_suffix = " (validation: issues remain)"
        else:
            val_suffix = ""

        rel = _rel(r["tex_path"])
        tag = f" (project: {project['name']})" if project else ""
        if projects is not None and project is not None and project.get("id"):
            await projects.add_artifact(
                project["id"], "methodology", Path(r["tex_path"]).stem, rel,
                {"n_refs": r["n_refs"]},
            )

        missing = r.get("missing_citations") or []
        warn = (
            f" ⚠ {len(missing)} undefined cite(s): "
            f"{', '.join(missing[:8])}{'…' if len(missing) > 8 else ''}"
            if missing else ""
        )
        summary = (
            f"Wrote a LaTeX methodology with {r['n_refs']} references{tag}: "
            f"`!getfile {rel}`{warn}{val_suffix}"
        )
        full_trace = (r.get("trace") or []) + full_trace + [
            {"type": "artifact", "tex": r["tex_path"], "bib": r["bib_path"],
             "missing_citations": missing}
        ]

        if memory is not None:
            from ..memory.lessons import schedule_reflection
            schedule_reflection(
                memory, "methodology", task, r.get("latex", ""),
                channel_id=channel_id, project=proj_slug,
            )
        return summary, full_trace

    runners["methodology"] = _methodology

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
            trace = (r.get("trace") or []) + [
                {"type": "transcript", "path": r["rel_path"]}, *extra
            ]
            return r["ideas"], trace

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
