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
        self._next_id = 0

    def add_fact(self, text, source=None, metadata=None, infer=False):
        self._next_id += 1
        self.facts.append((text, metadata))
        return f"fake-id-{self._next_id}"

    def recall(self, query, limit=5, only_type=None, only_kind=None):
        self.searched.append((query, only_type, only_kind))
        return "- a past lesson" if only_type == "lesson" else "- a fact"

    def recall_with_ids(self, query, limit=5, only_type=None, only_kind=None, score_map=None):
        self.searched.append((query, only_type, only_kind))
        text = "- a past lesson" if only_type == "lesson" else "- a fact"
        return text, ["fake-id-1"]


class _FakeLessonStats:
    """No-op lesson stats for unit tests (no DB)."""
    enabled = False

    async def get_scores(self, lesson_ids):
        return {lid: 0.5 for lid in lesson_ids}

    async def record_used(self, lesson_ids, kind=""):
        pass

    async def credit(self, lesson_ids, outcome):
        pass


def _manager():
    m = MemoryManager.__new__(MemoryManager)  # bypass __init__ (no pool/mem0)
    m.pool = None
    m.episodic = _FakeEpisodic()
    m.semantic = _FakeSemantic()
    m.procedural = None
    m.lesson_stats = _FakeLessonStats()
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
        self.credited = []

    async def recall_lessons(self, query, kind=None, limit=None):
        self.last_recall = (query, kind)
        return self._lessons

    async def recall_lessons_with_ids(self, query, kind=None, limit=None):
        self.last_recall = (query, kind)
        return self._lessons, ["id-1", "id-2"]

    async def reflect_and_record(self, kind, task, result, *, channel_id=None,
                                 project=None, outcome=None):
        self.reflected.append((kind, task, result, channel_id, project))

    async def credit_lessons(self, lesson_ids, outcome):
        self.credited.append((lesson_ids, outcome))


@pytest.mark.asyncio
async def test_prime_with_lessons_appends_relevant_block():
    from research_agent.memory.lessons import prime_with_lessons

    mem = _LoopMem("- prefer primary sources")
    out, ids = await prime_with_lessons(mem, "literature", "survey X")
    assert "survey X" in out
    assert "Lessons from past literature jobs" in out
    assert "prefer primary sources" in out
    assert mem.last_recall == ("survey X", "literature")
    # ids should be the ones returned by recall_lessons_with_ids
    assert ids == ["id-1", "id-2"]


@pytest.mark.asyncio
async def test_prime_with_lessons_noop_without_memory_or_lessons():
    from research_agent.memory.lessons import prime_with_lessons

    task, ids = await prime_with_lessons(None, "literature", "t")
    assert task == "t" and ids == []
    task, ids = await prime_with_lessons(_LoopMem(""), "literature", "t")
    assert task == "t" and ids == []


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


# =============================================================================
# New tests for outcome-scored lessons feature
# =============================================================================

# --- Outcome threading -------------------------------------------------------

@pytest.mark.asyncio
async def test_reflect_and_record_uses_bad_prompt_for_bad_outcome(monkeypatch):
    """outcome='bad' must produce a pitfall-phrased prompt (AVOID wording)."""
    import research_agent.llm as llm_mod
    from research_agent.memory.manager import _REFLECT_BAD_SYSTEM, _REFLECT_SYSTEM

    captured_messages = []

    class _Resp:
        content = "Avoid skipping baseline checks."

    class _LLM:
        async def ainvoke(self, messages):
            captured_messages.extend(messages)
            return _Resp()

    monkeypatch.setattr(llm_mod, "build_reflection_llm", lambda: _LLM())

    m = _manager()
    n = await m.reflect_and_record(
        "experiment", "run benchmark", "failed: missing baseline", outcome="bad"
    )
    assert n == 1
    system_content = captured_messages[0].content
    # Bad outcome must use the pitfall template, not the normal one.
    assert "AVOID" in system_content or "avoid" in system_content
    # Normal template must NOT be used.
    assert "what worked" not in system_content.lower()


@pytest.mark.asyncio
async def test_reflect_and_record_normal_prompt_for_good_outcome(monkeypatch):
    """outcome='good' must use the normal (what-worked) prompt."""
    import research_agent.llm as llm_mod

    captured_messages = []

    class _Resp:
        content = "Always validate inputs first."

    class _LLM:
        async def ainvoke(self, messages):
            captured_messages.extend(messages)
            return _Resp()

    monkeypatch.setattr(llm_mod, "build_reflection_llm", lambda: _LLM())

    m = _manager()
    await m.reflect_and_record(
        "experiment", "run benchmark", "success", outcome="good"
    )
    system_content = captured_messages[0].content
    # Must NOT be the pitfall template.
    assert "AVOID" not in system_content


@pytest.mark.asyncio
async def test_reflect_and_record_outcome_stored_in_metadata(monkeypatch):
    """outcome must be persisted in the lesson's semantic metadata."""
    import research_agent.llm as llm_mod

    class _Resp:
        content = "Avoid over-fitting on small datasets."

    class _LLM:
        async def ainvoke(self, messages):
            return _Resp()

    monkeypatch.setattr(llm_mod, "build_reflection_llm", lambda: _LLM())

    m = _manager()
    await m.reflect_and_record("experiment", "task", "result", outcome="bad")
    _, meta = m.semantic.facts[0]
    assert meta.get("outcome") == "bad"


@pytest.mark.asyncio
async def test_schedule_reflection_credits_lessons_on_good_outcome():
    """schedule_reflection must call credit_lessons when outcome is good."""
    import asyncio

    from research_agent.memory.lessons import schedule_reflection

    mem = _LoopMem()
    schedule_reflection(
        mem, "literature", "task", "result",
        outcome="good", lesson_ids=["id-a", "id-b"],
    )
    for _ in range(200):
        if mem.credited:
            break
        await asyncio.sleep(0.005)
    assert mem.credited == [(["id-a", "id-b"], "good")]


@pytest.mark.asyncio
async def test_schedule_reflection_credits_lessons_on_bad_outcome():
    """schedule_reflection must call credit_lessons when outcome is bad."""
    import asyncio

    from research_agent.memory.lessons import schedule_reflection

    mem = _LoopMem()
    schedule_reflection(
        mem, "literature", "task", "result",
        outcome="bad", lesson_ids=["id-x"],
    )
    for _ in range(200):
        if mem.credited:
            break
        await asyncio.sleep(0.005)
    assert mem.credited == [(["id-x"], "bad")]


@pytest.mark.asyncio
async def test_schedule_reflection_no_credit_without_outcome():
    """schedule_reflection must NOT call credit_lessons when outcome is None."""
    import asyncio

    from research_agent.memory.lessons import schedule_reflection

    mem = _LoopMem()
    schedule_reflection(
        mem, "literature", "task", "result",
        outcome=None, lesson_ids=["id-z"],
    )
    # Wait long enough for the background task to complete.
    for _ in range(200):
        if mem.reflected:
            break
        await asyncio.sleep(0.005)
    assert mem.credited == []


# --- Re-ranking math ---------------------------------------------------------

def test_recall_with_ids_reranks_by_score_map():
    """Higher score_map values should float lessons to the top.

    Blended score = 0.5 * relevance + 0.5 * quality.
    "low" has high relevance (0.9) but low quality (0.0) → blend = 0.45
    "high" has low relevance (0.1) but high quality (1.0) → blend = 0.55
    So "high" wins.
    """
    items = [
        {"id": "low", "memory": "low-quality lesson", "metadata": {"type": "lesson", "kind": "k"},
         "score": 0.9},
        {"id": "high", "memory": "high-quality lesson", "metadata": {"type": "lesson", "kind": "k"},
         "score": 0.1},
    ]
    sm = _semantic_with(items)
    # Extreme quality scores so the blend decisively flips the order.
    score_map = {"low": 0.0, "high": 1.0}
    text, ids = sm.recall_with_ids("q", limit=2, only_type="lesson", only_kind="k",
                                   score_map=score_map)
    # blend("high") = 0.5*0.1 + 0.5*1.0 = 0.55 > blend("low") = 0.5*0.9 + 0.5*0.0 = 0.45
    assert ids[0] == "high"


def test_recall_with_ids_without_score_map_uses_relevance_order():
    """Without a score_map, the original relevance order is preserved."""
    items = [
        {"id": "first", "memory": "first lesson", "metadata": {"type": "lesson"}, "score": 0.9},
        {"id": "second", "memory": "second lesson", "metadata": {"type": "lesson"}, "score": 0.5},
    ]
    sm = _semantic_with(items)
    text, ids = sm.recall_with_ids("q", limit=2, only_type="lesson")
    assert ids == ["first", "second"]


def test_recall_with_ids_returns_ids_alongside_text():
    """recall_with_ids must return both a text block and a non-empty id list."""
    items = [
        {"id": "abc", "memory": "a lesson", "metadata": {"type": "lesson"}},
    ]
    sm = _semantic_with(items)
    text, ids = sm.recall_with_ids("q", limit=5, only_type="lesson")
    assert "a lesson" in text
    assert "abc" in ids


def test_recall_with_ids_disabled_returns_empty():
    """recall_with_ids must return ('', []) when semantic memory is disabled."""
    sm = SemanticMemory()  # disabled by default in tests (no env vars)
    assert sm.recall_with_ids("q") == ("", [])


# --- Credit assignment -------------------------------------------------------

@pytest.mark.asyncio
async def test_credit_lessons_increments_good_count():
    """credit_lessons with outcome='good' must call lesson_stats.credit."""
    credited = []

    class _Stats:
        enabled = True
        async def credit(self, ids, outcome):
            credited.append((ids, outcome))

    m = _manager()
    m.lesson_stats = _Stats()
    await m.credit_lessons(["id-1"], "good")
    assert credited == [(["id-1"], "good")]


@pytest.mark.asyncio
async def test_credit_lessons_increments_bad_count():
    """credit_lessons with outcome='bad' must call lesson_stats.credit."""
    credited = []

    class _Stats:
        enabled = True
        async def credit(self, ids, outcome):
            credited.append((ids, outcome))

    m = _manager()
    m.lesson_stats = _Stats()
    await m.credit_lessons(["id-2"], "bad")
    assert credited == [(["id-2"], "bad")]


@pytest.mark.asyncio
async def test_credit_lessons_noop_for_unknown_outcome():
    """credit_lessons must silently skip unknown outcome strings."""
    credited = []

    class _Stats:
        enabled = True
        async def credit(self, ids, outcome):
            credited.append((ids, outcome))

    m = _manager()
    m.lesson_stats = _Stats()
    await m.credit_lessons(["id-3"], "meh")
    assert credited == []


@pytest.mark.asyncio
async def test_recall_lessons_with_ids_increments_usage():
    """recall_lessons_with_ids must call lesson_stats.record_used for returned ids."""
    used = []

    class _Stats:
        enabled = True
        async def get_scores(self, ids):
            return {lid: 0.5 for lid in ids}
        async def record_used(self, ids, kind=""):
            used.append((ids, kind))

    m = _manager()
    m.lesson_stats = _Stats()
    text, ids = await m.recall_lessons_with_ids("some query", kind="literature")
    assert used  # record_used was called
    assert ids == ["fake-id-1"]  # from _FakeSemantic


# --- Graceful no-ops when memory disabled ------------------------------------

@pytest.mark.asyncio
async def test_recall_lessons_with_ids_noop_when_disabled():
    """recall_lessons_with_ids must return ('', []) when semantic is disabled."""
    m = MemoryManager.__new__(MemoryManager)
    m.pool = None
    m.episodic = _FakeEpisodic()

    class _DisabledSemantic:
        enabled = False

    m.semantic = _DisabledSemantic()
    m.procedural = None
    m.lesson_stats = _FakeLessonStats()

    text, ids = await m.recall_lessons_with_ids("query")
    assert text == "" and ids == []


@pytest.mark.asyncio
async def test_credit_lessons_noop_for_empty_ids():
    """credit_lessons with empty id list must not crash."""
    m = _manager()
    # Should complete without error even with empty list.
    await m.credit_lessons([], "good")


# --- Consolidation pass ------------------------------------------------------

@pytest.mark.asyncio
async def test_consolidation_disabled_when_flag_off(monkeypatch):
    """consolidate_lessons must be a no-op when lesson_consolidation_enabled=False."""
    from research_agent.config import settings
    from research_agent.memory.maintenance import consolidate_lessons

    monkeypatch.setattr(settings, "lesson_consolidation_enabled", False)
    m = _manager()
    removed = await consolidate_lessons(m, llm=None)
    assert removed == 0


@pytest.mark.asyncio
async def test_consolidation_merges_batch_and_deletes_originals(monkeypatch):
    """consolidate_lessons must merge batches via LLM and delete originals."""
    import research_agent.llm as llm_mod
    from research_agent.config import settings
    from research_agent.memory.maintenance import consolidate_lessons

    monkeypatch.setattr(settings, "lesson_consolidation_enabled", True)
    monkeypatch.setattr(settings, "max_lessons_per_kind", 2)

    # Build a memory manager where recall_with_ids returns 5 lessons for "literature"
    # (over the cap of 2), so consolidation triggers.
    deleted_ids = []
    added_merged = []

    class _Stats:
        enabled = True
        async def aggregate_onto(self, target, sources, kind=""):
            pass

    class _MergeSemantic:
        enabled = True

        def recall_with_ids(self, query, limit, only_type=None, only_kind=None, score_map=None):
            if only_kind == "literature":
                ids = [f"id-{i}" for i in range(5)]
                lines = "\n".join(f"- lesson {i}" for i in range(5))
                return lines, ids
            return "", []

        def add_fact(self, text, source=None, metadata=None, infer=False):
            added_merged.append(text)
            return "merged-id"

        def delete_fact(self, mid):
            deleted_ids.append(mid)

    class _LLM:
        async def ainvoke(self, messages):
            class _R:
                content = "Merged: always validate data before processing."
            return _R()

    m = MemoryManager.__new__(MemoryManager)
    m.pool = None
    m.episodic = _FakeEpisodic()
    m.semantic = _MergeSemantic()
    m.procedural = None
    m.lesson_stats = _Stats()

    removed = await consolidate_lessons(m, llm=_LLM())
    # 5 originals → 1 merged = 4 removed (net reduction per batch)
    assert removed > 0
    # The merged lesson text should have been passed to add_fact.
    assert any("validate" in t for t in added_merged)
    # Originals should have been scheduled for deletion.
    assert deleted_ids  # at least some deletions happened


@pytest.mark.asyncio
async def test_consolidation_noop_when_under_cap(monkeypatch):
    """consolidate_lessons must not modify anything if count <= cap."""
    from research_agent.config import settings
    from research_agent.memory.maintenance import consolidate_lessons

    monkeypatch.setattr(settings, "lesson_consolidation_enabled", True)
    monkeypatch.setattr(settings, "max_lessons_per_kind", 200)

    class _SmallSemantic:
        enabled = True

        def recall_with_ids(self, query, limit, only_type=None, only_kind=None, score_map=None):
            # Only 2 lessons — well under the cap of 200.
            return "- lesson a\n- lesson b", ["a", "b"]

    m = MemoryManager.__new__(MemoryManager)
    m.pool = None
    m.episodic = _FakeEpisodic()
    m.semantic = _SmallSemantic()
    m.procedural = None
    m.lesson_stats = _FakeLessonStats()

    removed = await consolidate_lessons(m, llm=None)
    assert removed == 0


# --- _outcome_from_trace helper (dispatcher) ---------------------------------

def test_outcome_from_trace_valid_no_missing_is_good():
    from research_agent.agents.dispatcher import _outcome_from_trace

    trace = [
        {"type": "critique", "verifier": "citation_check", "verdict": "valid",
         "feedback": "ok", "round": 1, "superseded_draft": None},
    ]
    assert _outcome_from_trace(trace, []) == "good"


def test_outcome_from_trace_invalid_verdict_is_bad():
    from research_agent.agents.dispatcher import _outcome_from_trace

    trace = [
        {"type": "critique", "verifier": "paper_verifier", "verdict": "invalid",
         "feedback": "fabricated claims", "round": 1, "superseded_draft": "old"},
    ]
    assert _outcome_from_trace(trace, []) == "bad"


def test_outcome_from_trace_no_critiques_is_none():
    from research_agent.agents.dispatcher import _outcome_from_trace

    trace = [{"type": "artifact", "tex": "x.tex", "bib": "x.bib", "missing_citations": []}]
    assert _outcome_from_trace(trace, []) is None


def test_outcome_from_trace_valid_with_missing_is_none():
    """Valid critiques but missing citations → ambiguous → None."""
    from research_agent.agents.dispatcher import _outcome_from_trace

    trace = [
        {"type": "critique", "verifier": "citation_check", "verdict": "valid",
         "feedback": "", "round": 1, "superseded_draft": None},
    ]
    assert _outcome_from_trace(trace, ["smith2020"]) is None
