"""Tests for token estimation and the nudge-boundary logic."""

from __future__ import annotations

from research_agent.memory.tokens import (
    crossed_nudge_boundary,
    estimate_message_tokens,
    estimate_text_tokens,
    _content_to_text,
)


def test_estimate_text_tokens():
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("a" * 40) == 10  # ~4 chars/token


def test_content_to_text_handles_blocks():
    blocks = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
    assert _content_to_text(blocks) == "hello world"
    assert _content_to_text("plain") == "plain"
    assert _content_to_text(None) == ""


def test_estimate_message_tokens_tuples_and_objects():
    msgs = [("user", "a" * 40), ("assistant", "b" * 80)]
    assert estimate_message_tokens(msgs) == 10 + 20


def test_nudge_boundary_fires_once_per_band():
    step = 20000
    # crossing into the first band
    assert crossed_nudge_boundary(20001, 0, step) is True
    # same band again -> no repeat
    assert crossed_nudge_boundary(25000, 20001, step) is False
    # next band
    assert crossed_nudge_boundary(40001, 20001, step) is True
    # below first band -> never
    assert crossed_nudge_boundary(19999, 0, step) is False


def test_nudge_boundary_disabled_when_step_nonpositive():
    assert crossed_nudge_boundary(99999, 0, 0) is False
