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


def test_recall_filters_by_kind():
    sm = _semantic_with([
        {"memory": "lit lesson", "metadata": {"type": "lesson", "kind": "literature"}},
        {"memory": "meth lesson", "metadata": {"type": "lesson", "kind": "methodology"}},
    ])
    out = sm.recall("q", only_type="lesson", only_kind="literature")
    assert "lit lesson" in out and "meth lesson" not in out


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

    def recall(self, query, limit=5, only_type=None, only_kind=None):
        self.searched.append((query, only_type, only_kind))
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


@pytest.mark.asyncio
async def test_recall_lessons_scopes_by_kind():
    m = _manager()
    await m.recall_lessons("survey diffusion models", kind="literature")
    assert m.semantic.searched[0] == ("survey diffusion models", "lesson", "literature")


@pytest.mark.asyncio
async def test_reflect_and_record_distills_and_tags(monkeypatch):
    """A finished job is distilled into kind/project-tagged lessons."""
    import research_agent.llm as llm_mod

    class _Resp:
        content = "Prefer primary sources.\nAlways cite arXiv ids.\nNONE"

    class _LLM:
        async def ainvoke(self, messages):
            return _Resp()

    monkeypatch.setattr(llm_mod, "build_reflection_llm", lambda: _LLM())

    m = _manager()
    n = await m.reflect_and_record(
        "literature", "find work on X", "a synthesis", project="p1"
    )
    assert n == 2  # two real lessons; the literal "NONE" line is dropped
    assert {meta["kind"] for _, meta in m.semantic.facts} == {"literature"}
    assert all(meta.get("project") == "p1" for _, meta in m.semantic.facts)


@pytest.mark.asyncio
async def test_reflect_respects_max_lessons_cap(monkeypatch):
    """The cap is honored before persisting (so it can't over-record by one)."""
    import research_agent.llm as llm_mod
    from research_agent.config import settings

    monkeypatch.setattr(settings, "reflection_max_lessons", 1)

    class _Resp:
        content = "Lesson one here.\nLesson two here."

    class _LLM:
        async def ainvoke(self, messages):
            return _Resp()

    monkeypatch.setattr(llm_mod, "build_reflection_llm", lambda: _LLM())

    m = _manager()
    n = await m.reflect_and_record("literature", "task", "result")
    assert n == 1
    assert len(m.semantic.facts) == 1


# --- the reusable prime/reflect helpers --------------------------------------

class _LoopMem:
    def __init__(self, lessons=""):
        self._lessons = lessons
        self.last_recall = None
        self.reflected = []

    async def recall_lessons(self, query, kind=None, limit=None):
        self.last_recall = (query, kind)
        return self._lessons

    async def reflect_and_record(self, kind, task, result, *, channel_id=None, project=None):
        self.reflected.append((kind, task, result, channel_id, project))


@pytest.mark.asyncio
async def test_prime_with_lessons_appends_relevant_block():
    from research_agent.memory.lessons import prime_with_lessons

    mem = _LoopMem("- prefer primary sources")
    out = await prime_with_lessons(mem, "literature", "survey X")
    assert "survey X" in out
    assert "Lessons from past literature jobs" in out
    assert "prefer primary sources" in out
    assert mem.last_recall == ("survey X", "literature")


@pytest.mark.asyncio
async def test_prime_with_lessons_noop_without_memory_or_lessons():
    from research_agent.memory.lessons import prime_with_lessons

    assert await prime_with_lessons(None, "literature", "t") == "t"
    assert await prime_with_lessons(_LoopMem(""), "literature", "t") == "t"


@pytest.mark.asyncio
async def test_schedule_reflection_runs_in_background():
    import asyncio

    from research_agent.memory.lessons import schedule_reflection

    mem = _LoopMem()
    schedule_reflection(
        mem, "literature", "task text", "result text", channel_id="c1", project="p1"
    )
    # Poll for the fire-and-forget task instead of a fixed sleep (CI-robust).
    for _ in range(200):  # up to ~1s
        if mem.reflected:
            break
        await asyncio.sleep(0.005)
    assert mem.reflected == [("literature", "task text", "result text", "c1", "p1")]


@pytest.mark.asyncio
async def test_schedule_reflection_noop_without_result():
    import asyncio

    from research_agent.memory.lessons import schedule_reflection

    mem = _LoopMem()
    schedule_reflection(mem, "literature", "task", "")  # empty result -> skip
    await asyncio.sleep(0.01)
    assert mem.reflected == []
