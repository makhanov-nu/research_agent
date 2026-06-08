"""The reflect-and-recall loop that lets subagents accumulate experience.

Around each subagent job we:

1. RECALL — pull the most relevant lessons from past jobs of the same kind and
   prime the task with them (retrieval-augmented: a top-K vector search, so the
   prompt stays bounded no matter how large the lesson store grows — you never
   "load all of memory").
2. REFLECT — after the job, distill durable lessons from (task -> result) and
   record them for next time. This runs in the BACKGROUND so it never slows the
   response or blocks the caller.

This is the same mechanism the experiment runner already uses for failures,
generalized so the literature and writing subagents grow too. Lessons are tagged
by `agent_kind` (so recall stays relevant) and optionally by `project`.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# asyncio keeps only weak references to bare tasks, so a fire-and-forget task can
# be garbage-collected mid-flight. Holding a strong ref here until it completes
# prevents that.
_pending: set[asyncio.Task] = set()


async def prime_with_lessons(memory, agent_kind: str | None, task: str) -> str:
    """Prepend relevant past lessons to a task. No-op without memory/lessons.

    Recall is global (kind-scoped, not project-scoped) by design — knowledge
    compounds across projects; the `project` tag lives on the *stored* lesson for
    optional filtering, not on recall.
    """
    from ..config import settings

    if memory is None or not agent_kind or not settings.lessons_enabled:
        return task
    try:
        lessons = await memory.recall_lessons(task, kind=agent_kind)
    except Exception:  # noqa: BLE001 — recall must never break the job
        logger.exception("Lesson recall failed for %s", agent_kind)
        return task
    if not lessons:
        return task
    return (
        f"{task}\n\n=== Lessons from past {agent_kind} jobs "
        "(apply these; avoid repeating past mistakes) ===\n"
        f"{lessons}"
    )


def schedule_reflection(memory, agent_kind: str | None, task: str, result: str, *,
                        channel_id: str | None = None,
                        project: str | None = None) -> None:
    """Fire-and-forget: distill + store lessons from a finished job.

    Runs on the current event loop without blocking the caller; all exceptions
    are logged, never raised. No-op without memory/lessons or an empty result.
    """
    from ..config import settings

    if memory is None or not agent_kind or not result or not settings.lessons_enabled:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # no running loop (e.g. a sync context) — skip silently
    task_ = loop.create_task(
        _reflect(memory, agent_kind, task, result, channel_id, project)
    )
    _pending.add(task_)
    task_.add_done_callback(_pending.discard)


async def _reflect(memory, agent_kind, task, result, channel_id, project) -> None:
    try:
        await memory.reflect_and_record(
            agent_kind, task, result, channel_id=channel_id, project=project,
        )
    except Exception:  # noqa: BLE001 — reflection is best-effort
        logger.exception("Reflection failed for %s", agent_kind)
