"""Tests for the orchestrator delegation layer."""

from __future__ import annotations

from research_agent.agents.consortium_tool import build_consortium_tool
from research_agent.agents.subagent import flatten_content


def test_flatten_content():
    assert flatten_content("plain") == "plain"
    assert flatten_content([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    assert flatten_content([{"text": "x"}, "y"]) == "xy"


class _FakeConsortium:
    def __init__(self):
        self.calls = []

    async def ideate(self, topic, focus=""):
        self.calls.append((topic, focus))
        return {"ideas": f"3 ideas for {topic}", "rel_path": "ideas/x.md"}


async def test_consortium_tool_returns_only_ideas_and_path():
    fake = _FakeConsortium()
    tool = build_consortium_tool(fake)
    out = await tool.ainvoke({"topic": "speculative decoding", "focus": "training-free"})
    assert "3 ideas for speculative decoding" in out
    assert "ideas/x.md" in out
    assert fake.calls == [("speculative decoding", "training-free")]
