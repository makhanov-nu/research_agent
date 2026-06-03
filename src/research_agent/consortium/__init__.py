"""Ideation consortium: a shared multi-model round-table debate.

Several frontier models (via OpenRouter) discuss in one shared session — each
reads the running transcript before speaking, so they hear and react to each
other — through propose -> debate -> chair synthesis, converging on a few
Q1-level research ideas grounded in the literature.
"""

from .consortium import Consortium, ConsortiumSession

__all__ = ["Consortium", "ConsortiumSession", "capture_council"]


async def capture_council(
    memory, channel_id, topic: str, ideas: str, rel_path: str = "", rounds=None
) -> None:
    """Record a finished council session to memory (episodic + a durable lesson).

    Shared by the interactive `!ideate done` path, the orchestrator's
    brainstorm tool, and the background dispatcher, so every council run feeds
    future ideation (recall seeds round 1). No-op without a memory manager.
    """
    if memory is None:
        return
    note = f"Consortium on '{topic}'"
    if rounds is not None:
        note += f" converged after {rounds} round(s)"
    await memory.log_experience(
        "council_session", note, channel_id,
        {"topic": topic, "rel_path": rel_path},
    )
    await memory.record_lesson(
        f"[council:{topic}] {ideas[:1500]}", kind="council", channel_id=channel_id,
    )
