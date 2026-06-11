"""Background maintenance: idle-channel archival + consolidation/reflection.

Runs on a periodic loop (see settings.maintenance_interval_seconds):

1. Archival      - channels idle > archive_idle_days get their rolling summary
                   written into long-term semantic memory, then marked archived.
2. Reflection    - recent experiments/actions are distilled into a semantic
                   "insight" so episodic experience is promoted to durable facts.
3. Consolidation - per-kind lesson deduplication: when the number of stored
                   lessons for an agent kind exceeds `max_lessons_per_kind`,
                   near-duplicate lessons are merged in batches via LLM and the
                   originals are deleted. The merged lesson's lesson_stats row
                   accumulates the source rows' good/bad/times_used counts.
                   Gated by `lesson_consolidation_enabled`; never raises.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from ..config import settings
from .episodic import utcnow

logger = logging.getLogger(__name__)


async def archive_idle_channels(memory) -> int:
    idle = await memory.episodic.list_idle_channels(settings.archive_idle_days)
    for conv in idle:
        channel_id = conv["channel_id"]
        summary = conv.get("summary") or ""
        if summary and memory.semantic.enabled:
            await asyncio.to_thread(
                memory.semantic.remember,
                f"Archived research thread {channel_id} summary:",
                summary,
                f"channel:{channel_id}",
            )
        await memory.episodic.mark_archived(channel_id, summary)
        logger.info("Archived idle channel %s", channel_id)
    return len(idle)


async def reflect(memory, llm) -> None:
    """Promote recent episodic experience into semantic insights."""
    since = utcnow() - timedelta(seconds=settings.maintenance_interval_seconds)
    actions = await memory.episodic.recent_actions(since)
    experiments = await memory.episodic.list_experiments(limit=20)
    if not actions and not experiments:
        return

    exp_lines = [
        f"- {e['title']} [{e['status']}] metrics={e.get('metrics')}"
        for e in experiments
    ]
    body = (
        f"Recent agent actions: {len(actions)}.\n"
        f"Experiments:\n" + "\n".join(exp_lines)
    )
    from langchain_core.messages import HumanMessage, SystemMessage

    resp = await llm.ainvoke(
        [
            SystemMessage(
                content="Distill the recent research activity into 1-5 concise, "
                "durable insights worth remembering long-term. If nothing is "
                "noteworthy, reply with 'NONE'."
            ),
            HumanMessage(content=body),
        ]
    )
    insight = getattr(resp, "content", "")
    if isinstance(insight, list):
        insight = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in insight)
    insight = insight.strip()
    if insight and insight.upper() != "NONE" and memory.semantic.enabled:
        await asyncio.to_thread(
            memory.semantic.remember,
            "Consolidated research insight:",
            insight,
            "reflection",
        )
        logger.info("Stored consolidated insight.")


_CONSOLIDATE_SYSTEM = (
    "You are a knowledge-distillation step. You will receive a numbered list of "
    "similar lessons from past '{kind}' agent jobs. Merge them into ONE concise, "
    "durable lesson that captures everything non-redundant. The merged lesson must "
    "be a single sentence (or two at most) and must generalise beyond any specific "
    "task. Output ONLY the merged lesson — no preamble, no numbering."
)


async def consolidate_lessons(memory, llm) -> int:
    """Merge near-duplicate lessons per kind when over the configured cap.

    For each agent kind whose lesson count exceeds `max_lessons_per_kind`, we
    pull all lesson texts from mem0 (via a kind-scoped recall with a high limit),
    batch them into groups of 10, and ask the LLM to merge each batch into one
    lesson. The merged lesson is stored via record_lesson(); the originals are
    deleted via semantic.delete_fact() and their lesson_stats rows are aggregated
    onto the merged row. Best-effort: any exception is logged and the pass
    continues with the next batch/kind.

    Returns the total number of lessons removed (originals − merged replacements).
    """
    if not settings.lesson_consolidation_enabled:
        return 0
    if not memory.semantic.enabled:
        return 0

    from langchain_core.messages import HumanMessage, SystemMessage

    # Known agent kinds — expand as new agents are added.
    kinds = [
        "literature", "literature_review", "paper_draft",
        "methodology", "experiment", "council",
    ]
    total_removed = 0

    for kind in kinds:
        try:
            removed = await _consolidate_kind(memory, llm, kind)
            total_removed += removed
        except Exception:  # noqa: BLE001
            logger.exception("Lesson consolidation failed for kind=%s", kind)

    return total_removed


async def _consolidate_kind(memory, llm, kind: str) -> int:
    """Consolidate lessons for a single agent kind. Returns lessons removed."""
    from langchain_core.messages import HumanMessage, SystemMessage

    cap = settings.max_lessons_per_kind
    batch_size = 10

    # Recall up to 3× the cap so we can tell if we're over the limit.
    check_limit = cap + 1
    raw = await asyncio.to_thread(
        memory.semantic.recall_with_ids,
        kind,           # query — kind name gives good recall for kind-filtered items
        check_limit,
        "lesson",       # only_type
        kind,           # only_kind
        None,           # no score_map — this is maintenance, not job recall
    )
    _text_block, all_ids = raw
    if len(all_ids) <= cap:
        return 0

    # We're over cap — process in batches of batch_size.
    # Collect (id, text) pairs by re-fetching with a high limit so we have the text.
    raw_full = await asyncio.to_thread(
        memory.semantic.recall_with_ids,
        kind, cap * 3, "lesson", kind, None,
    )
    full_text_block, full_ids = raw_full
    # Parse the text block back into per-lesson lines.
    lines = [
        line.lstrip("- ").strip()
        for line in full_text_block.splitlines() if line.strip()
    ]
    # Zip ids with lines; fall back gracefully if counts differ.
    pairs = list(zip(full_ids, lines, strict=False))
    if not pairs:
        return 0

    total_removed = 0
    system_msg = SystemMessage(content=_CONSOLIDATE_SYSTEM.format(kind=kind))

    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start: batch_start + batch_size]
        if len(batch) < 2:
            # Single lesson — nothing to merge; skip.
            continue
        numbered = "\n".join(f"{i+1}. {text}" for i, (_, text) in enumerate(batch))
        try:
            resp = await llm.ainvoke([
                system_msg,
                HumanMessage(content=numbered),
            ])
            merged_text = getattr(resp, "content", "")
            if isinstance(merged_text, list):
                merged_text = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in merged_text
                )
            merged_text = merged_text.strip()
            if not merged_text or merged_text.upper() == "NONE":
                continue
        except Exception:  # noqa: BLE001
            logger.exception(
                "Consolidation LLM call failed for kind=%s batch=%d", kind, batch_start
            )
            continue

        # Store the merged lesson; it gets its own mem0 id.
        source_ids = [mid for mid, _ in batch]
        merged_id = await memory.record_lesson(merged_text, kind=kind)

        # Aggregate stats from source ids onto merged id (best-effort).
        if merged_id:
            try:
                await memory.lesson_stats.aggregate_onto(merged_id, source_ids, kind=kind)
            except Exception:  # noqa: BLE001
                logger.exception("Stats aggregation failed after consolidation (kind=%s)", kind)

        # Delete originals from mem0 (best-effort per id).
        for mid in source_ids:
            try:
                await asyncio.to_thread(memory.semantic.delete_fact, mid)
            except Exception:  # noqa: BLE001
                logger.exception("delete_fact failed for id=%s", mid)

        total_removed += len(batch) - 1  # net reduction
        logger.info(
            "Consolidated %d lessons → 1 for kind=%s (merged id=%s)",
            len(batch), kind, merged_id,
        )

    return total_removed


async def run_once(memory, llm) -> None:
    try:
        n = await archive_idle_channels(memory)
        await reflect(memory, llm)
        removed = await consolidate_lessons(memory, llm)
        logger.info(
            "Maintenance pass complete (archived %d channel(s), removed %d duplicate lessons).",
            n, removed,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Maintenance pass failed.")


async def run_loop(memory, llm) -> None:
    while True:
        await run_once(memory, llm)
        await asyncio.sleep(settings.maintenance_interval_seconds)
