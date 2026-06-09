"""Ideation consortium: a two-track, scored multi-model panel.

Frontier reasoning models (via OpenRouter) generate ideas through two isolated
tracks — an INDEPENDENT track (each works alone, for diversity) and a DEBATED
track (a separate shared conversation, for emergent synthesis). The merged,
anonymized pool is scored 0-10 by every panelist under an anti-neutrality rubric;
the chair ranks them (normalizing across raters) into a top 5. The researcher
picks; the panel then polishes the chosen idea(s) on both tracks and votes. The
debate transcripts are saved so future sessions' debate track recalls them.
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
