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

Artifact / task piping
----------------------
dispatch_task (and TaskDispatcher.dispatch) accept two optional reference lists:

  * input_artifacts — paths relative to output_dir; each file is read and
    appended to the task string before the runner is invoked.
  * input_tasks — ids of COMPLETED task rows whose ``result`` is appended.

Content is injected with clearly-labelled delimiters and a shared character
budget (``settings.dispatch_input_budget_chars``).  Missing files or non-done
tasks are rejected before dispatch so the orchestrator can react early rather
than letting a half-injected job silently proceed.

The original (un-augmented) task string is stored as the task row's ``input``
column; the injected content is only in the runner's call, so the dashboard
stays clean.  A ``{"type": "inputs", ...}`` entry is appended to the final
trace for auditability.

Linear pipelines
----------------
TaskDispatcher accepts an optional ``pipelines`` PipelineStore.  After every
task reaches a terminal state the dispatcher checks whether the task belongs to
a pipeline and, if so, calls the pipeline advancement helpers from pipeline.py.
The dispatcher itself stays thin; all pipeline logic lives in pipeline.py.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from ..config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input resolution helpers (artifact + task piping)
# ---------------------------------------------------------------------------

def _resolve_artifact_path(rel_path: str, output_dir: str) -> Path | str:
    """Resolve an artifact path relative to output_dir; reject traversals.

    Returns a Path if valid, or an error string if the path is rejected or the
    file does not exist.
    """
    base = Path(output_dir).resolve()
    try:
        target = (base / rel_path).resolve()
        target.relative_to(base)  # raises ValueError on traversal
    except (ValueError, Exception) as exc:
        return f"[piping] Path rejected (traversal or invalid): {rel_path!r} — {exc}"
    if not target.exists():
        return f"[piping] Artifact not found: {rel_path!r}"
    return target


def _build_injected_task(
    original_task: str,
    input_artifacts: list[str] | None,
    input_tasks_data: list[tuple[int, str]] | None,
    budget: int,
) -> tuple[str, list[str], list[int]]:
    """Compose the augmented task string from original task + piped inputs.

    Returns (augmented_task, artifact_paths_used, task_ids_used).
    Blocks are appended in artifact order then task order.
    Content that would exceed the budget is truncated with an explicit marker.
    """
    blocks: list[str] = []
    used_artifacts: list[str] = []
    used_task_ids: list[int] = []
    remaining = budget

    for path in (input_artifacts or []):
        header = f"\n\n=== INPUT (artifact: {path}) ===\n"
        cap = remaining - len(header)
        if cap <= 0:
            blocks.append(header + "…[truncated: input budget exhausted]")
            used_artifacts.append(path)
            remaining = 0
            break
        target = _resolve_artifact_path(path, settings.output_dir)
        if isinstance(target, str):
            # Already an error string — caller should have caught this; skip silently.
            continue
        try:
            content = target.read_text(errors="replace")
        except Exception as exc:
            content = f"[piping] Could not read file: {exc}"
        if len(content) > cap:
            content = content[:cap] + f"\n…[truncated; {len(content)} chars total]"
            remaining = 0
        else:
            remaining -= len(content)
        blocks.append(header + content)
        used_artifacts.append(path)
        if remaining <= 0:
            break

    for tid, result in (input_tasks_data or []):
        header = f"\n\n=== INPUT (task #{tid} result) ===\n"
        cap = remaining - len(header)
        if cap <= 0:
            blocks.append(header + "…[truncated: input budget exhausted]")
            used_task_ids.append(tid)
            remaining = 0
            break
        if len(result) > cap:
            result = result[:cap] + f"\n…[truncated; {len(result)} chars total]"
            remaining = 0
        else:
            remaining -= len(result)
        blocks.append(header + result)
        used_task_ids.append(tid)
        if remaining <= 0:
            break

    augmented = original_task + "".join(blocks)
    return augmented, used_artifacts, used_task_ids


def _outcome_from_trace(trace: list, missing_citations: list) -> str | None:
    """Derive a job outcome signal from the critique trace and citation gaps.

    Rules (in priority order):
    - Any critique entry with verdict "invalid" → "bad" (the job had verifiable
      problems; pitfall extraction will help future jobs avoid them).
    - No invalid verdicts AND no missing citations → "good" (clean pass).
    - No critique entries at all → None (no signal; use neutral prompt).

    This is a CONSERVATIVE heuristic: when in doubt we return None so the
    normal "what worked" prompt is used rather than incorrectly labelling a
    job bad.
    """
    critique_entries = [e for e in trace if e.get("type") == "critique"]
    if not critique_entries:
        return None
    if any(e.get("verdict") == "invalid" for e in critique_entries):
        return "bad"
    if not missing_citations:
        return "good"
    # Citations missing but no invalid verdict — ambiguous; no signal.
    return None


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
            lesson_ids: list[str] = []
            if memory is not None and settings.lessons_enabled:
                try:
                    lessons, lesson_ids = await memory.recall_lessons_with_ids(task, kind=agent)
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
            # Derive outcome from citation-check trace: valid + no missing cites
            # → "good"; final verdict invalid → "bad"; otherwise no signal.
            outcome = _outcome_from_trace(full_trace, missing)
            # Reflect the finished draft into durable lessons (background).
            if memory is not None:
                from ..memory.lessons import schedule_reflection

                schedule_reflection(
                    memory, agent, task, r.get("latex", ""),
                    channel_id=channel_id, project=proj_slug,
                    outcome=outcome, lesson_ids=lesson_ids,
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
        paper_lesson_ids: list[str] = []
        if memory is not None and settings.lessons_enabled:
            try:
                lessons, paper_lesson_ids = await memory.recall_lessons_with_ids(task, kind="paper_draft")
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
        paper_outcome = _outcome_from_trace(full_trace, missing)
        if memory is not None:
            from ..memory.lessons import schedule_reflection
            schedule_reflection(
                memory, "paper_draft", task, r.get("latex", ""),
                channel_id=channel_id, project=proj_slug,
                outcome=paper_outcome, lesson_ids=paper_lesson_ids,
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
        meth_lesson_ids: list[str] = []
        if memory is not None and settings.lessons_enabled:
            try:
                lessons, meth_lesson_ids = await memory.recall_lessons_with_ids(task, kind="methodology")
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

        meth_outcome = _outcome_from_trace(full_trace, missing)
        if memory is not None:
            from ..memory.lessons import schedule_reflection
            schedule_reflection(
                memory, "methodology", task, r.get("latex", ""),
                channel_id=channel_id, project=proj_slug,
                outcome=meth_outcome, lesson_ids=meth_lesson_ids,
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
                 max_parallel: int = 4, pipelines=None):
        self._runners = runners
        self.task_store = task_store
        self._on_complete = on_complete
        self._sem = asyncio.Semaphore(max_parallel)
        self._running: dict[int, asyncio.Task] = {}
        # Optional PipelineStore — when present, task completions advance pipelines.
        self._pipelines = pipelines

    @property
    def agents(self) -> list[str]:
        return list(self._runners)

    async def dispatch(
        self,
        agent: str,
        task: str,
        channel_id: str | None,
        input_artifacts: list[str] | None = None,
        input_tasks: list[int] | None = None,
    ) -> int:
        """Dispatch a task to a runner, optionally piping artifact/task inputs.

        Args:
            agent: runner key (must exist in self._runners).
            task: the original task string (stored as-is in the task row).
            channel_id: originating channel.
            input_artifacts: paths relative to output_dir whose content is
                prepended to the runner's task string.
            input_tasks: ids of COMPLETED task rows whose result is prepended.

        Returns:
            The task row id.

        Raises:
            ValueError: unknown agent, unresolvable artifact, or non-done task.
        """
        if agent not in self._runners:
            raise ValueError(
                f"Unknown agent {agent!r}. Available: {', '.join(self.agents)}."
            )

        # --- validate and pre-read inputs ---
        artifact_errors: list[str] = []
        task_inputs_data: list[tuple[int, str]] = []

        if input_artifacts:
            for rel_path in input_artifacts:
                result = _resolve_artifact_path(rel_path, settings.output_dir)
                if isinstance(result, str):
                    artifact_errors.append(result)
            if artifact_errors:
                raise ValueError(
                    "Cannot dispatch: " + "; ".join(artifact_errors)
                )

        if input_tasks:
            for tid in input_tasks:
                row = await self.task_store.get(tid)
                if row is None:
                    raise ValueError(
                        f"Cannot dispatch: task #{tid} not found in the dashboard."
                    )
                if row.get("status") != "done":
                    raise ValueError(
                        f"Cannot dispatch: task #{tid} is not done "
                        f"(status={row.get('status')!r})."
                    )
                task_inputs_data.append((tid, row.get("result") or ""))

        # Store the ORIGINAL task string in the row (clean dashboard).
        task_id = await self.task_store.create(agent, task, channel_id)

        # Build the augmented task that the runner will actually see.
        augmented_task, used_artifacts, used_task_ids = _build_injected_task(
            task,
            input_artifacts,
            task_inputs_data,
            budget=settings.dispatch_input_budget_chars,
        )

        bg = asyncio.create_task(
            self._run(task_id, agent, augmented_task, channel_id,
                      used_artifacts, used_task_ids)
        )
        if task_id is not None:
            self._running[task_id] = bg
        return task_id

    async def _run(
        self, task_id, agent, task, channel_id,
        used_artifacts=None, used_task_ids=None,
    ) -> None:
        async with self._sem:
            await self.task_store.mark_running(task_id)
            try:
                result, trace = await self._runners[agent](task, channel_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Dispatched task %s (%s) failed", task_id, agent)
                await self.task_store.fail(task_id, str(exc), [])
                await self._advance_pipeline(task_id, "failed", channel_id)
                await self._fire(task_id, agent, "failed", channel_id)
                return
            finally:
                self._running.pop(task_id, None)
            # Append an inputs trace entry when content was piped in.
            if used_artifacts or used_task_ids:
                trace = list(trace) + [{
                    "type": "inputs",
                    "artifacts": used_artifacts or [],
                    "tasks": used_task_ids or [],
                }]
            # Write the result to the dashboard FIRST so the handler can read it.
            await self.task_store.finish(task_id, result, trace)
            await self._advance_pipeline(task_id, "done", channel_id)
            await self._fire(task_id, agent, "done", channel_id)

    async def _advance_pipeline(
        self, task_id: int | None, status: str, channel_id: str | None
    ) -> None:
        """Hook into the pipeline store after a task reaches a terminal state.

        Best-effort: any exception is logged and swallowed so pipeline errors
        never interfere with normal task completion.
        """
        if task_id is None or self._pipelines is None:
            return
        try:
            pipeline = await self._pipelines.find_by_task(task_id)
            if pipeline is None:
                return
            # Attach the store reference so the helper can call it.
            pipeline["_store"] = self._pipelines

            from .pipeline import on_stage_failure, on_stage_success

            if status == "done":
                async def _dispatch_next(agent, task, ch_id, input_task_ids):
                    return await self.dispatch(
                        agent, task, ch_id, input_tasks=input_task_ids
                    )
                await on_stage_success(pipeline, task_id, _dispatch_next)
            else:
                await on_stage_failure(pipeline)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Pipeline advancement failed after task #%s (%s)", task_id, status
            )

    async def _fire(self, task_id, agent, status, channel_id) -> None:
        """Trigger the completion handler (never raises). Carries no result —
        the handler reads the result/error back from the task store."""
        try:
            await self._on_complete(task_id, agent, status, channel_id)
        except Exception:  # noqa: BLE001
            logger.exception("Task completion handler failed for #%s", task_id)

    async def cancel(self, task_id: int) -> bool:
        """Cancel a running or pending task.

        Marks the DB row cancelled and cancels the backing asyncio task when
        it is still in flight. Returns True if the asyncio task was found and
        cancelled, False if the task had already finished or was never running.
        """
        await self.task_store.cancel(task_id)
        bg = self._running.pop(task_id, None)
        if bg is not None:
            bg.cancel()
            return True
        return False

    async def join(self) -> None:
        """Await all in-flight tasks (used in tests / graceful shutdown)."""
        if self._running:
            await asyncio.gather(*self._running.values(), return_exceptions=True)

    async def shutdown(self) -> None:
        for t in list(self._running.values()):
            t.cancel()


def build_dispatch_tools(dispatcher: TaskDispatcher) -> list[BaseTool]:
    """Build the orchestrator tools that expose the background dispatcher.

    Returns dispatch_task, and — when the dispatcher has a pipeline store —
    also run_pipeline and pipeline_status.
    """
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
            "lookup, call the direct tool instead.\n\n"
            "ARTIFACT / TASK PIPING — instead of copy-pasting prior outputs into "
            "the task string, pass references:\n"
            "  • input_artifacts: list of paths relative to the output dir "
            "(e.g. ['projects/my-project/methodology/design.tex']) — the file "
            "contents are injected into the task before the runner is called.\n"
            "  • input_tasks: list of integer task ids whose results should be "
            "piped in (those tasks MUST be done; pass their ids from earlier "
            "'[BACKGROUND TASK COMPLETE]' events).\n"
            "Both are optional. For a known multi-stage flow "
            "(lit review → methodology → paper) prefer run_pipeline instead."
        ),
    )
    async def dispatch_task(
        agent: str,
        task: str,
        input_artifacts: list[str] | None = None,
        input_tasks: list[int] | None = None,
        config: RunnableConfig = None,
    ) -> str:
        try:
            task_id = await dispatcher.dispatch(
                agent, task, _channel(config),
                input_artifacts=input_artifacts,
                input_tasks=input_tasks,
            )
        except ValueError as exc:
            return str(exc)
        pipes = []
        if input_artifacts:
            pipes.append(f"{len(input_artifacts)} artifact(s)")
        if input_tasks:
            pipes.append(f"result(s) of task(s) {input_tasks}")
        piped = f" (piping {', '.join(pipes)})" if pipes else ""
        return (
            f"Dispatched task #{task_id} to `{agent}` in the background{piped}. "
            f"Its result will be delivered to you automatically when it finishes."
        )

    @tool(
        "cancel_task",
        description=(
            "Cancel a running or pending background task by id. "
            "Use when the user asks to stop a task or it is no longer needed. "
            "Has no effect on tasks that are already done, failed, or cancelled."
        ),
    )
    async def cancel_task(task_id: int) -> str:
        row = await dispatcher.task_store.get(task_id)
        if row is None:
            return f"Task #{task_id} not found."
        status = row.get("status", "")
        if status not in ("pending", "running"):
            return f"Task #{task_id} is already {status} — nothing to cancel."
        found = await dispatcher.cancel(task_id)
        agent = row.get("agent", "?")
        if found:
            return f"Task #{task_id} ({agent}) cancelled."
        return f"Task #{task_id} ({agent}) marked cancelled (it may have just finished)."

    tools: list[BaseTool] = [dispatch_task, cancel_task]

    # Pipeline tools — only when the dispatcher has a pipeline store.
    if dispatcher._pipelines is not None:
        pipelines = dispatcher._pipelines

        @tool(
            "run_pipeline",
            description=(
                "Create and start a LINEAR multi-stage pipeline. Each stage runs "
                "after the previous one completes, automatically receiving the "
                "previous stage's result. Use this for known sequential flows such "
                "as 'literature review → methodology → paper' instead of manually "
                "chaining dispatch_task calls.\n\n"
                f"`stages` is a JSON list of objects with 'agent' (one of: {agents}) "
                "and 'task' (the instruction for that stage). Stage 0 is dispatched "
                "immediately; subsequent stages are dispatched automatically.\n\n"
                "Returns the pipeline id and the stage-0 task id. Use "
                "pipeline_status(pipeline_id) to inspect progress."
            ),
        )
        async def run_pipeline(
            name: str, stages: list[dict], config: RunnableConfig = None
        ) -> str:
            if not pipelines.enabled:
                return (
                    "Pipelines require DATABASE_URL to be configured. "
                    "Use dispatch_task for individual steps instead."
                )
            # Validate every agent up front.
            unknown = [
                s.get("agent", "") for s in stages
                if s.get("agent", "") not in dispatcher.agents
            ]
            if unknown:
                return (
                    f"Unknown agent(s) in pipeline stages: {unknown}. "
                    f"Available: {', '.join(dispatcher.agents)}."
                )
            if not stages:
                return "Pipeline must have at least one stage."

            channel_id = _channel(config)
            # Build the stage list with task_id=null placeholders.
            stage_rows = [
                {"agent": s["agent"], "task": s["task"], "task_id": None}
                for s in stages
            ]
            pipeline_id = await pipelines.create(name, stage_rows, channel_id)
            if pipeline_id is None:
                return "Failed to create pipeline (database error)."

            # Dispatch stage 0 immediately (no inputs yet).
            try:
                task_id = await dispatcher.dispatch(
                    stages[0]["agent"], stages[0]["task"], channel_id
                )
            except ValueError as exc:
                await pipelines.set_status(pipeline_id, "failed")
                return f"Failed to dispatch stage 0: {exc}"

            await pipelines.record_stage_task(pipeline_id, 0, task_id)
            return (
                f"Pipeline #{pipeline_id} '{name}' started with {len(stages)} stage(s). "
                f"Stage 0 dispatched as task #{task_id}. "
                f"Use pipeline_status({pipeline_id}) to track progress."
            )

        @tool(
            "pipeline_status",
            description=(
                "Return the current status of a pipeline created with run_pipeline. "
                "Shows each stage (agent, task, dispatched task id, completion) and "
                "the overall pipeline status (queued|running|failed|done)."
            ),
        )
        async def pipeline_status(pipeline_id: int) -> str:
            if not pipelines.enabled:
                return "Pipelines require DATABASE_URL."
            row = await pipelines.get(pipeline_id)
            if row is None:
                return f"Pipeline #{pipeline_id} not found."
            stages = row.get("stages") or []
            lines = [
                f"Pipeline #{pipeline_id} '{row['name']}' — status: {row['status']}",
                f"Current stage: {row['current_stage']}",
                "",
                "Stages:",
            ]
            for i, stage in enumerate(stages):
                tid = stage.get("task_id")
                tid_str = f"task #{tid}" if tid is not None else "not yet dispatched"
                lines.append(
                    f"  [{i}] agent={stage.get('agent')} | {tid_str} | "
                    f"task={stage.get('task', '')[:80]}"
                )
            return "\n".join(lines)

        tools += [run_pipeline, pipeline_status]

    return tools
