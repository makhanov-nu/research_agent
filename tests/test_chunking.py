"""Tests for Discord message chunking."""

from __future__ import annotations

from research_agent.discord_bot.bot import DISCORD_MAX_CHARS, _chunk


def test_short_text_single_chunk():
    assert _chunk("hello") == ["hello"]


def test_empty_text():
    assert _chunk("") == ["(empty response)"]


def test_respects_max_size():
    text = "\n".join(f"line {i}" for i in range(1000))
    chunks = _chunk(text)
    assert all(len(c) <= DISCORD_MAX_CHARS for c in chunks)
    # No content lost (joining is lossless here since we keep line endings).
    assert "".join(chunks) == text


def test_splits_single_overlong_line():
    text = "x" * (DISCORD_MAX_CHARS * 2 + 5)
    chunks = _chunk(text)
    assert all(len(c) <= DISCORD_MAX_CHARS for c in chunks)
    assert "".join(chunks) == text
