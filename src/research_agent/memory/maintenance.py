"""Background maintenance: idle-channel archival + consolidation/reflection.

Runs on a periodic loop (see settings.maintenance_interval_seconds):

1. Archival   - channels idle > archive_idle_days get their rolling summary
                written into long-term semantic memory, then marked archived.
2. Reflection - recent experiments/actions are distilled into a semantic
                "insight" so episodic experience is promoted to durable facts.
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


async def run_once(memory, llm) -> None:
    try:
        n = await archive_idle_channels(memory)
        await reflect(memory, llm)
        logger.info("Maintenance pass complete (archived %d channel(s)).", n)
    except Exception:  # noqa: BLE001
        logger.exception("Maintenance pass failed.")


async def run_loop(memory, llm) -> None:
    while True:
        await run_once(memory, llm)
        await asyncio.sleep(settings.maintenance_interval_seconds)
