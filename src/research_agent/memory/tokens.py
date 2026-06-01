"""Lightweight token accounting and the 20k-token nudge boundary logic.

We use a cheap char-based estimate (~4 chars/token) rather than a tokenizer:
it needs no extra dependency and no network round-trip, and it's accurate
enough to drive summarization thresholds and user nudges.
"""

from __future__ import annotations

from typing import Iterable

_CHARS_PER_TOKEN = 4


def _content_to_text(content) -> str:
    """Flatten a message's content (str or list of content blocks) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", "") or "")
            else:
                parts.append(str(block))
        return " ".join(parts)
    return str(content)


def estimate_text_tokens(text: str) -> int:
    return max(0, len(text) // _CHARS_PER_TOKEN)


def estimate_message_tokens(messages: Iterable) -> int:
    """Estimate tokens for a list of LangChain messages or (role, text) tuples."""
    total = 0
    for m in messages:
        content = m[1] if isinstance(m, tuple) else getattr(m, "content", "")
        total += estimate_text_tokens(_content_to_text(content))
    return total


def crossed_nudge_boundary(current: int, last_nudged: int, step: int) -> bool:
    """True when `current` has entered a new `step`-sized band beyond the last
    band we already nudged at.

    e.g. step=20000: nudges fire once as we pass 20k, 40k, 60k, ...
    """
    if step <= 0:
        return False
    return current // step > last_nudged // step
