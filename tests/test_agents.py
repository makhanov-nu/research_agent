"""Tests for the orchestrator delegation layer."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from research_agent.agents.consortium_tool import build_consortium_tool
from research_agent.agents.middleware import serialize_messages
from research_agent.agents.subagent import flatten_content


def test_serialize_messages_captures_reasoning_tools_and_results():
    msgs = [
        HumanMessage(content="find papers on X"),
        AIMessage(
            content="searching",
            tool_calls=[{"name": "search", "args": {"q": "X"}, "id": "tc1"}],
        ),
        ToolMessage(content="paper A; paper B", tool_call_id="tc1", name="search"),
        AIMessage(content="here is the synthesis"),
    ]
    trace = serialize_messages(msgs)
    assert trace[0]["content"] == "find papers on X"
    assert trace[1]["tool_calls"][0]["name"] == "search"
    assert trace[1]["tool_calls"][0]["args"] == {"q": "X"}
    assert trace[2]["tool_call_id"] == "tc1"
    assert trace[2]["name"] == "search"
    assert trace[3]["content"] == "here is the synthesis"


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
