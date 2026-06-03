"""Tests for the experience->lesson->reuse memory loop."""

from __future__ import annotations

import pytest

from research_agent.memory.manager import MemoryManager
from research_agent.memory.semantic import SemanticMemory


class _FakeMem:
    def __init__(self, items):
        self._items = items
        self.added = []

    def search(self, query, filters=None, limit=5):
        return {"results": self._items}

    def add(self, text, user_id=None, metadata=None, infer=False):
        self.added.append({"text": text, "metadata": metadata, "infer": infer})


def _semantic_with(items):
    sm = SemanticMemory()
    sm._enabled = True
    sm._mem = _FakeMem(items)
    return sm


def test_recall_only_type_filters_to_lessons():
    sm = _semantic_with([
        {"memory": "a fact", "metadata": {"type": "fact"}},
        {"memory": "a lesson", "metadata": {"type": "lesson"}},
    ])
    out = sm.recall("q", only_type="lesson")
    assert "a lesson" in out and "a fact" not in out


def test_recall_without_type_returns_all():
    sm = _semantic_with([{"memory": "x", "metadata": {}}])
    assert "x" in sm.recall("q")


def test_add_fact_stores_verbatim_with_metadata():
    sm = _semantic_with([])
    sm.add_fact("OOM at batch>32 → use grad accumulation", "lesson:experiment",
                {"type": "lesson", "kind": "experiment"})
    assert sm._mem.added[0]["infer"] is False
    assert sm._mem.added[0]["metadata"]["type"] == "lesson"


# --- MemoryManager routing (fake stores, no DB/mem0) -------------------------

class _FakeEpisodic:
    enabled = True

    def __init__(self):
        self.actions = []

    async def log_action(self, kind, summary, channel_id=None, metadata=None):
        self.actions.append((kind, summary, channel_id, metadata))


class _FakeSemantic:
    enabled = True

    def __init__(self):
        self.facts = []
        self.searched = []

    def add_fact(self, text, source=None, metadata=None, infer=False):
        self.facts.append((text, metadata))

    def recall(self, query, limit=5, only_type=None):
        self.searched.append((query, only_type))
        return "- a past lesson" if only_type == "lesson" else "- a fact"


def _manager():
    m = MemoryManager.__new__(MemoryManager)  # bypass __init__ (no pool/mem0)
    m.pool = None
    m.episodic = _FakeEpisodic()
    m.semantic = _FakeSemantic()
    m.procedural = None
    return m


@pytest.mark.asyncio
async def test_record_lesson_writes_episodic_and_semantic():
    m = _manager()
    await m.record_lesson("don't OOM", kind="experiment", channel_id="c", status="failed")
    assert m.episodic.actions[0][0] == "lesson_experiment"
    text, meta = m.semantic.facts[0]
    assert meta["type"] == "lesson" and meta["kind"] == "experiment" and meta["status"] == "failed"


@pytest.mark.asyncio
async def test_recall_lessons_queries_lesson_type():
    m = _manager()
    out = await m.recall_lessons("fine-tune llama")
    assert out == "- a past lesson"
    assert m.semantic.searched[0][1] == "lesson"
