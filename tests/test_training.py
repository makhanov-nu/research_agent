"""Tests for the training-data exporter (task trajectories -> JSONL)."""

from __future__ import annotations

import json

from research_agent.training.export import (
    export_dataset,
    system_prompt_for,
    task_to_example,
)


def test_task_to_example_builds_chat_triplet():
    row = {
        "id": 1, "agent": "methodology", "input": "design X",
        "result": "```latex\n\\section{Methodology}", "quality": "good",
        "feedback": None, "trace": [{"type": "ai"}], "created_at": "2026-06-08",
    }
    ex = task_to_example(row)
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]
    assert ex["messages"][1]["content"] == "design X"
    assert ex["messages"][2]["content"].startswith("```latex")
    assert ex["metadata"]["agent"] == "methodology"
    assert ex["metadata"]["quality"] == "good"
    assert "trace" not in ex  # excluded by default


def test_task_to_example_include_trace():
    row = {"id": 2, "agent": "literature_review", "input": "a", "result": "b",
           "trace": [{"type": "tool", "name": "search"}]}
    ex = task_to_example(row, include_trace=True)
    assert ex["trace"] == [{"type": "tool", "name": "search"}]


def test_task_to_example_skips_empty_input_or_result():
    assert task_to_example({"agent": "x", "input": "", "result": "y"}) is None
    assert task_to_example({"agent": "x", "input": "a", "result": ""}) is None


def test_safe_filename_sanitizes_agent_names():
    from research_agent.training.export import _safe_filename

    assert _safe_filename("research_literature") == "research_literature"
    assert _safe_filename("../etc/passwd") == "etc_passwd"      # no traversal
    assert _safe_filename("a/b\\c") == "a_b_c"                   # no path separators
    assert _safe_filename("") == "unknown"
    assert _safe_filename("...") == "unknown"


def test_system_prompt_for_known_and_fallback():
    # Known role -> its real system prompt.
    assert "methodolog" in system_prompt_for("methodology").lower()
    assert "literature" in system_prompt_for("research_literature").lower()
    # Unknown role -> a non-empty fallback, never an error.
    assert system_prompt_for("nope")


class _FakeStore:
    """Stands in for TaskStore.list_for_export with canned rows."""

    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    async def list_for_export(self, *, agents=None, quality=None, since=None, limit=100_000):
        self.calls.append({"agents": agents, "quality": quality, "since": since})
        rows = self._rows
        if quality:
            rows = [r for r in rows if r.get("quality") in quality]
        if agents:
            rows = [r for r in rows if r.get("agent") in agents]
        return rows


async def test_export_dataset_writes_jsonl_per_role(tmp_path):
    store = _FakeStore([
        {"id": 1, "agent": "methodology", "input": "i1", "result": "o1", "quality": "good"},
        {"id": 2, "agent": "methodology", "input": "i2", "result": "o2", "quality": None},
        {"id": 3, "agent": "literature_review", "input": "i3", "result": "o3", "quality": "good"},
        {"id": 4, "agent": "methodology", "input": "", "result": "skip"},  # dropped
    ])
    manifest = await export_dataset(store, tmp_path)

    assert manifest["methodology"]["count"] == 2  # the empty-input row is dropped
    assert manifest["literature_review"]["count"] == 1

    lines = (tmp_path / "methodology.jsonl").read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["messages"][0]["role"] == "system"
    assert (tmp_path / "manifest.json").exists()


async def test_export_dataset_good_only_filters(tmp_path):
    store = _FakeStore([
        {"id": 1, "agent": "methodology", "input": "i1", "result": "o1", "quality": "good"},
        {"id": 2, "agent": "methodology", "input": "i2", "result": "o2", "quality": None},
    ])
    manifest = await export_dataset(store, tmp_path, good_only=True)
    assert manifest["methodology"]["count"] == 1
    assert store.calls[0]["quality"] == ("good",)  # filter pushed down to the query
