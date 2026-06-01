"""Tests for summarization helpers (pure logic only)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from research_agent.memory.summarize import messages_to_drop


def _msgs():
    return [
        HumanMessage(content="a", id="1"),
        AIMessage(content="b", id="2"),
        HumanMessage(content="c", id="3"),
        AIMessage(content="d", id="4"),
    ]


def test_drops_all_but_last_n():
    drops = messages_to_drop(_msgs(), keep_last=2)
    assert all(isinstance(d, RemoveMessage) for d in drops)
    assert [d.id for d in drops] == ["1", "2"]


def test_drops_nothing_when_under_keep():
    assert messages_to_drop(_msgs(), keep_last=10) == []


def test_keep_last_zero_drops_everything():
    assert len(messages_to_drop(_msgs(), keep_last=0)) == 4
