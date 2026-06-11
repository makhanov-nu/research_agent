"""The reflect-and-recall loop that lets subagents accumulate experience.

Around each subagent job we:

1. RECALL — pull the most relevant lessons from past jobs of the same kind and
   prime the task with them (retrieval-augmented: a top-K vector search, so the
   prompt stays bounded no matter how large the lesson store grows — you never
   "load all of memory"). The recalled lesson ids are returned alongside the
   primed task so the job can credit them after completion.
2. REFLECT — after the job, distill durable lessons from (task -> result) and
   record them for next time. This runs in the BACKGROUND so it never slows the
   response or blocks the caller.
3. CREDIT — when the outcome is known ("good" or "bad"), update lesson_stats for
   the lessons that were recalled for this job, so the quality prior stays fresh.

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


async def prime_with_lessons(
    memory, agent_kind: str | None, task: str
) -> tuple[str, list[str]]:
    """Prepend relevant past lessons to a task.

    Returns (primed_task, lesson_ids) where lesson_ids is the list of mem0 ids
    for the lessons that were injected. lesson_ids is empty when memory is off,
    lessons_enabled is False, or recall fails.

    Recall is global (kind-scoped, not project-scoped) by design — knowledge
    compounds across projects; the `project` tag lives on the *stored* lesson for
    optional filtering, not on recall.
    """
    from ..config import settings

    if memory is None or not agent_kind or not settings.lessons_enabled:
        return task, []
    try:
        lessons, ids = await memory.recall_lessons_with_ids(task, kind=agent_kind)
    except Exception:  # noqa: BLE001 — recall must never break the job
        logger.exception("Lesson recall failed for %s", agent_kind)
        return task, []
    if not lessons:
        return task, []
    primed = (
        f"{task}\n\n=== Lessons from past {agent_kind} jobs "
        "(apply these; avoid repeating past mistakes) ===\n"
        f"{lessons}"
    )
    return primed, ids


def schedule_reflection(
    memory,
    agent_kind: str | None,
    task: str,
    result: str,
    *,
    channel_id: str | None = None,
    project: str | None = None,
    outcome: str | None = None,
    lesson_ids: list[str] | None = None,
) -> None:
    """Fire-and-forget: distill + store lessons from a finished job.

    Runs on the current event loop without blocking the caller; all exceptions
    are logged, never raised. No-op without memory/lessons or an empty result.

    `outcome` ("good" | "bad" | None) determines the reflection prompt variant
    used and is stored in each lesson's metadata.

    `lesson_ids` is the list of mem0 ids returned by prime_with_lessons(); when
    provided and outcome is known, those lessons' good/bad counts are updated so
    the quality prior improves over time. This is a NOISY signal (all recalled
    lessons get the same credit with no per-lesson attribution) and is used only
    to re-rank recall, never to hard-delete.
    """
    from ..config import settings

    if memory is None or not agent_kind or not result or not settings.lessons_enabled:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # no running loop (e.g. a sync context) — skip silently
    task_ = loop.create_task(
        _reflect(memory, agent_kind, task, result, channel_id, project,
                 outcome, lesson_ids or [])
    )
    _pending.add(task_)
    task_.add_done_callback(_pending.discard)


async def _reflect(
    memory,
    agent_kind: str,
    task: str,
    result: str,
    channel_id: str | None,
    project: str | None,
    outcome: str | None,
    lesson_ids: list[str],
) -> None:
    try:
        await memory.reflect_and_record(
            agent_kind, task, result,
            channel_id=channel_id, project=project, outcome=outcome,
        )
    except Exception:  # noqa: BLE001 — reflection is best-effort
        logger.exception("Reflection failed for %s", agent_kind)
    # Credit the lessons that were recalled for this job (noisy soft prior).
    if lesson_ids and outcome in ("good", "bad"):
        try:
            await memory.credit_lessons(lesson_ids, outcome)
        except Exception:  # noqa: BLE001
            logger.exception("Lesson credit failed for %s outcome=%s", agent_kind, outcome)
