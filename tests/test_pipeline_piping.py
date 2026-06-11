"""Tests for artifact/task piping and linear pipelines.

Covers:
  - artifact input resolution (valid, traversal rejection, missing file)
  - task input resolution (done task piped, not-done task rejected, missing task rejected)
  - budget truncation (artifacts and tasks)
  - _build_injected_task composes blocks correctly
  - dispatch_task tool returns error string on bad inputs
  - pipeline advance on success (stage 1 dispatched with input_tasks)
  - pipeline halted on failure (status set to failed)
  - pipeline done after last stage
  - run_pipeline rejects unknown agents
  - run_pipeline degrades gracefully when pipelines store is disabled
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from research_agent.agents.dispatcher import (
    TaskDispatcher,
    _build_injected_task,
    _resolve_artifact_path,
    build_dispatch_tools,
)
from research_agent.agents.pipeline import (
    PipelineStore,
    on_stage_failure,
    on_stage_success,
)


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------

class _FakeTaskStore:
    def __init__(self):
        self.rows: dict[int, dict] = {}
        self._next = 1

    async def create(self, agent, input, channel_id=None, parent_id=None):
        tid = self._next
        self._next += 1
        self.rows[tid] = {
            "id": tid, "agent": agent, "input": input, "channel_id": channel_id,
            "status": "pending", "result": None, "trace": [], "error": None,
        }
        return tid

    async def mark_running(self, tid):
        if tid is not None:
            self.rows[tid]["status"] = "running"

    async def finish(self, tid, result, trace):
        if tid is not None:
            self.rows[tid].update(status="done", result=result, trace=trace)

    async def fail(self, tid, error, trace):
        if tid is not None:
            self.rows[tid].update(status="failed", error=error, trace=trace)

    async def get(self, tid):
        return self.rows.get(tid)


class _FakePipelineStore:
    """In-memory pipeline store that matches the PipelineStore interface."""

    def __init__(self):
        self.rows: dict[int, dict] = {}
        self._next = 1

    @property
    def enabled(self):
        return True

    async def setup(self):
        pass

    async def create(self, name, stages, channel_id=None):
        pid = self._next
        self._next += 1
        self.rows[pid] = {
            "id": pid, "name": name, "stages": [dict(s) for s in stages],
            "current_stage": 0, "status": "queued", "channel_id": channel_id,
        }
        return pid

    async def set_status(self, pipeline_id, status):
        self.rows[pipeline_id]["status"] = status

    async def advance(self, pipeline_id, next_stage, task_id):
        row = self.rows[pipeline_id]
        row["stages"][next_stage]["task_id"] = task_id
        row["current_stage"] = next_stage
        row["status"] = "running"

    async def record_stage_task(self, pipeline_id, stage_index, task_id):
        await self.advance(pipeline_id, stage_index, task_id)

    async def get(self, pipeline_id):
        return dict(self.rows.get(pipeline_id, {})) or None

    async def find_by_task(self, task_id):
        for row in self.rows.values():
            for stage in row["stages"]:
                if stage.get("task_id") == task_id:
                    r = dict(row)
                    r["stages"] = [dict(s) for s in row["stages"]]
                    return r
        return None


def _make_dispatcher(runners, events, pipelines=None):
    async def on_complete(task_id, agent, status, channel_id):
        events.append((task_id, agent, status, channel_id))

    return TaskDispatcher(
        runners, _FakeTaskStore(), on_complete, max_parallel=4,
        pipelines=pipelines,
    )


async def _good_runner(task, channel_id=None):
    return f"done: {task[:30]}", []


# ---------------------------------------------------------------------------
# _resolve_artifact_path
# ---------------------------------------------------------------------------

def test_resolve_valid_artifact(tmp_path, monkeypatch):
    import research_agent.config as cfg_mod
    monkeypatch.setattr(cfg_mod.settings, "output_dir", str(tmp_path))
    f = tmp_path / "report.tex"
    f.write_text("hello")
    result = _resolve_artifact_path("report.tex", str(tmp_path))
    assert isinstance(result, Path)
    assert result == f.resolve()


def test_resolve_traversal_rejected(tmp_path, monkeypatch):
    import research_agent.config as cfg_mod
    monkeypatch.setattr(cfg_mod.settings, "output_dir", str(tmp_path))
    result = _resolve_artifact_path("../../etc/passwd", str(tmp_path))
    assert isinstance(result, str)
    assert "rejected" in result.lower()


def test_resolve_missing_file(tmp_path, monkeypatch):
    import research_agent.config as cfg_mod
    monkeypatch.setattr(cfg_mod.settings, "output_dir", str(tmp_path))
    result = _resolve_artifact_path("does_not_exist.tex", str(tmp_path))
    assert isinstance(result, str)
    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# _build_injected_task
# ---------------------------------------------------------------------------

def test_build_injected_task_no_inputs():
    task, arts, tids = _build_injected_task("original", None, None, budget=1000)
    assert task == "original"
    assert arts == []
    assert tids == []


def test_build_injected_task_with_task_result():
    task, arts, tids = _build_injected_task(
        "do X",
        None,
        [(7, "prior result here")],
        budget=1000,
    )
    assert "=== INPUT (task #7 result) ===" in task
    assert "prior result here" in task
    assert tids == [7]


def test_build_injected_task_budget_truncates():
    long_result = "A" * 2000
    task, arts, tids = _build_injected_task(
        "task",
        None,
        [(1, long_result)],
        budget=100,
    )
    assert "[truncated" in task
    assert len(task) < 2200  # did not paste the full 2000 chars


def test_build_injected_task_artifact_and_task(tmp_path, monkeypatch):
    import research_agent.config as cfg_mod
    monkeypatch.setattr(cfg_mod.settings, "output_dir", str(tmp_path))
    f = tmp_path / "file.tex"
    f.write_text("artifact content")
    task, arts, tids = _build_injected_task(
        "base task",
        ["file.tex"],
        [(3, "task result")],
        budget=5000,
    )
    assert "artifact content" in task
    assert "task result" in task
    assert arts == ["file.tex"]
    assert tids == [3]


# ---------------------------------------------------------------------------
# TaskDispatcher.dispatch — input validation
# ---------------------------------------------------------------------------

async def test_dispatch_rejects_missing_artifact(tmp_path, monkeypatch):
    import research_agent.config as cfg_mod
    monkeypatch.setattr(cfg_mod.settings, "output_dir", str(tmp_path))

    events = []
    disp = _make_dispatcher({"lit": _good_runner}, events)
    with pytest.raises(ValueError, match="not found"):
        await disp.dispatch("lit", "task", "c1", input_artifacts=["missing.tex"])


async def test_dispatch_rejects_traversal_artifact(tmp_path, monkeypatch):
    import research_agent.config as cfg_mod
    monkeypatch.setattr(cfg_mod.settings, "output_dir", str(tmp_path))

    events = []
    disp = _make_dispatcher({"lit": _good_runner}, events)
    with pytest.raises(ValueError, match="rejected"):
        await disp.dispatch("lit", "task", "c1", input_artifacts=["../../etc/passwd"])


async def test_dispatch_rejects_not_done_task():
    events = []
    disp = _make_dispatcher({"lit": _good_runner}, events)
    # Create a pending task in the store.
    ts = disp.task_store
    tid = await ts.create("lit", "earlier job", "c1")
    # It is still "pending" — dispatch should refuse.
    with pytest.raises(ValueError, match="not done"):
        await disp.dispatch("lit", "next job", "c1", input_tasks=[tid])


async def test_dispatch_rejects_missing_task_id():
    events = []
    disp = _make_dispatcher({"lit": _good_runner}, events)
    with pytest.raises(ValueError, match="not found"):
        await disp.dispatch("lit", "task", "c1", input_tasks=[9999])


async def test_dispatch_pipes_done_task(tmp_path):
    """A done task's result is injected into the runner's task string."""
    received_tasks: list[str] = []

    async def capturing_runner(task, channel_id=None):
        received_tasks.append(task)
        return "output", []

    events = []
    disp = _make_dispatcher({"lit": capturing_runner}, events)
    ts = disp.task_store
    # Pre-create and finish a task so it is in "done" state.
    prior_id = await ts.create("lit", "prior", "c1")
    await ts.mark_running(prior_id)
    await ts.finish(prior_id, "the prior result", [])

    new_id = await disp.dispatch("lit", "new task", "c1", input_tasks=[prior_id])
    await disp.join()

    assert disp.task_store.rows[new_id]["status"] == "done"
    assert "the prior result" in received_tasks[0]
    assert f"task #{prior_id}" in received_tasks[0]


async def test_dispatch_artifact_piped_content(tmp_path, monkeypatch):
    """File content is injected into the runner's task string."""
    import research_agent.config as cfg_mod
    monkeypatch.setattr(cfg_mod.settings, "output_dir", str(tmp_path))

    (tmp_path / "notes.tex").write_text("some LaTeX content")

    received: list[str] = []

    async def capturing_runner(task, channel_id=None):
        received.append(task)
        return "ok", []

    events = []
    disp = _make_dispatcher({"lit": capturing_runner}, events)
    await disp.dispatch("lit", "base", "c1", input_artifacts=["notes.tex"])
    await disp.join()

    assert "some LaTeX content" in received[0]
    assert "artifact: notes.tex" in received[0]


async def test_original_task_stored_in_row_not_augmented(tmp_path, monkeypatch):
    """The task row's 'input' column should hold the original task, not the augmented one."""
    import research_agent.config as cfg_mod
    monkeypatch.setattr(cfg_mod.settings, "output_dir", str(tmp_path))

    (tmp_path / "doc.tex").write_text("EXTRA CONTENT")

    events = []
    disp = _make_dispatcher({"lit": _good_runner}, events)
    tid = await disp.dispatch("lit", "clean task", "c1", input_artifacts=["doc.tex"])
    await disp.join()

    row = disp.task_store.rows[tid]
    assert row["input"] == "clean task"
    # The augmented content must NOT appear in the stored input.
    assert "EXTRA CONTENT" not in row["input"]


async def test_inputs_trace_entry_appended(tmp_path):
    """When piping, a {"type": "inputs"} entry is appended to the trace."""
    events = []
    disp = _make_dispatcher({"lit": _good_runner}, events)
    ts = disp.task_store

    prior_id = await ts.create("lit", "prior", "c1")
    await ts.mark_running(prior_id)
    await ts.finish(prior_id, "result", [])

    tid = await disp.dispatch("lit", "task", "c1", input_tasks=[prior_id])
    await disp.join()

    trace = disp.task_store.rows[tid]["trace"]
    inputs_entries = [e for e in trace if e.get("type") == "inputs"]
    assert len(inputs_entries) == 1
    assert inputs_entries[0]["tasks"] == [prior_id]


# ---------------------------------------------------------------------------
# dispatch_task tool — error string on bad input
# ---------------------------------------------------------------------------

async def test_dispatch_tool_returns_error_on_missing_artifact(tmp_path, monkeypatch):
    import research_agent.config as cfg_mod
    monkeypatch.setattr(cfg_mod.settings, "output_dir", str(tmp_path))

    events = []
    disp = _make_dispatcher({"lit": _good_runner}, events)
    tools = build_dispatch_tools(disp)
    dispatch_tool = next(t for t in tools if t.name == "dispatch_task")

    msg = await dispatch_tool.ainvoke({
        "agent": "lit",
        "task": "q",
        "input_artifacts": ["ghost.tex"],
    })
    # Should return an error string, not raise.
    assert "not found" in msg.lower() or "rejected" in msg.lower()
    assert "Dispatched" not in msg


async def test_dispatch_tool_returns_error_on_not_done_task():
    events = []
    disp = _make_dispatcher({"lit": _good_runner}, events)
    ts = disp.task_store
    prior_id = await ts.create("lit", "old", "c1")  # pending

    tools = build_dispatch_tools(disp)
    dispatch_tool = next(t for t in tools if t.name == "dispatch_task")

    msg = await dispatch_tool.ainvoke({
        "agent": "lit",
        "task": "q",
        "input_tasks": [prior_id],
    })
    assert "not done" in msg.lower()


# ---------------------------------------------------------------------------
# Pipeline advancement helpers (unit tests on on_stage_success / on_stage_failure)
# ---------------------------------------------------------------------------

async def test_on_stage_success_dispatches_next_stage():
    store = _FakePipelineStore()
    pid = await store.create(
        "test-pipe",
        [
            {"agent": "lit", "task": "stage0", "task_id": 10},
            {"agent": "meth", "task": "stage1", "task_id": None},
        ],
    )
    # Mark stage 0 dispatched.
    store.rows[pid]["stages"][0]["task_id"] = 10
    store.rows[pid]["current_stage"] = 0
    store.rows[pid]["status"] = "running"

    dispatched_calls: list[tuple] = []

    async def _dispatch(agent, task, channel_id, input_task_ids):
        dispatched_calls.append((agent, task, input_task_ids))
        return 99  # fake new task id

    pipeline = {**store.rows[pid], "_store": store}
    await on_stage_success(pipeline, 10, _dispatch)

    assert len(dispatched_calls) == 1
    assert dispatched_calls[0][0] == "meth"
    assert dispatched_calls[0][2] == [10]  # input_tasks = [completed_task_id]
    # Pipeline should have advanced to stage 1.
    assert store.rows[pid]["current_stage"] == 1
    assert store.rows[pid]["stages"][1]["task_id"] == 99
    assert store.rows[pid]["status"] == "running"


async def test_on_stage_success_done_after_last_stage():
    store = _FakePipelineStore()
    pid = await store.create(
        "pipe",
        [{"agent": "lit", "task": "only stage", "task_id": 5}],
    )
    store.rows[pid]["stages"][0]["task_id"] = 5
    store.rows[pid]["current_stage"] = 0
    store.rows[pid]["status"] = "running"

    called = []

    async def _dispatch(*args, **kwargs):
        called.append(args)
        return 99

    pipeline = {**store.rows[pid], "_store": store}
    await on_stage_success(pipeline, 5, _dispatch)

    assert not called  # no next stage
    assert store.rows[pid]["status"] == "done"


async def test_on_stage_failure_marks_pipeline_failed():
    store = _FakePipelineStore()
    pid = await store.create(
        "pipe",
        [
            {"agent": "lit", "task": "s0", "task_id": 1},
            {"agent": "meth", "task": "s1", "task_id": None},
        ],
    )
    store.rows[pid]["stages"][0]["task_id"] = 1
    store.rows[pid]["current_stage"] = 0
    store.rows[pid]["status"] = "running"

    pipeline = {**store.rows[pid], "_store": store}
    await on_stage_failure(pipeline)

    assert store.rows[pid]["status"] == "failed"


# ---------------------------------------------------------------------------
# Full dispatcher integration: pipeline auto-advancement
# ---------------------------------------------------------------------------

async def test_pipeline_advances_through_all_stages():
    """Two-stage pipeline: stage 0 succeeds → stage 1 is dispatched automatically."""
    store = _FakePipelineStore()
    pid = await store.create(
        "chain",
        [
            {"agent": "lit", "task": "stage0 task", "task_id": None},
            {"agent": "lit", "task": "stage1 task", "task_id": None},
        ],
    )

    events = []
    disp = _make_dispatcher({"lit": _good_runner}, events, pipelines=store)

    # Dispatch stage 0 and record it in the pipeline.
    t0 = await disp.dispatch("lit", "stage0 task", "c1")
    await store.record_stage_task(pid, 0, t0)
    await disp.join()

    # After t0 finishes, the dispatcher should have advanced the pipeline
    # by dispatching stage 1 automatically.
    assert store.rows[pid]["current_stage"] == 1 or store.rows[pid]["status"] == "done"
    stage1_tid = store.rows[pid]["stages"][1].get("task_id")
    # stage1 task was dispatched as a new background task.
    assert stage1_tid is not None
    # Let stage 1 complete too.
    await disp.join()
    assert store.rows[pid]["status"] == "done"


async def test_pipeline_halts_on_stage_failure():
    """When stage 0 fails, the pipeline is marked failed and stage 1 is not dispatched."""

    async def failing_runner(task, channel_id=None):
        raise RuntimeError("boom")

    store = _FakePipelineStore()
    pid = await store.create(
        "doomed",
        [
            {"agent": "bad", "task": "s0", "task_id": None},
            {"agent": "bad", "task": "s1", "task_id": None},
        ],
    )

    events = []
    disp = _make_dispatcher({"bad": failing_runner}, events, pipelines=store)

    t0 = await disp.dispatch("bad", "s0", "c1")
    await store.record_stage_task(pid, 0, t0)
    await disp.join()

    assert store.rows[pid]["status"] == "failed"
    # Stage 1 should NOT have been dispatched.
    assert store.rows[pid]["stages"][1].get("task_id") is None


# ---------------------------------------------------------------------------
# run_pipeline tool
# ---------------------------------------------------------------------------

async def test_run_pipeline_tool_rejects_unknown_agent():
    store = _FakePipelineStore()
    events = []
    disp = _make_dispatcher({"lit": _good_runner}, events, pipelines=store)
    tools = build_dispatch_tools(disp)
    run_pipe = next(t for t in tools if t.name == "run_pipeline")

    msg = await run_pipe.ainvoke({
        "name": "test",
        "stages": [
            {"agent": "lit", "task": "step 1"},
            {"agent": "nonexistent", "task": "step 2"},
        ],
    })
    assert "unknown" in msg.lower() or "nonexistent" in msg.lower()
    # No pipeline should have been created.
    assert len(store.rows) == 0


async def test_run_pipeline_tool_disabled_without_db():
    """Without a pipeline store (pool=None), run_pipeline returns a clear message."""
    disabled_store = PipelineStore(pool=None)  # enabled = False

    events = []
    disp = _make_dispatcher({"lit": _good_runner}, events, pipelines=disabled_store)
    tools = build_dispatch_tools(disp)
    run_pipe = next(t for t in tools if t.name == "run_pipeline")

    msg = await run_pipe.ainvoke({
        "name": "pipe",
        "stages": [{"agent": "lit", "task": "go"}],
    })
    assert "database_url" in msg.lower() or "require" in msg.lower()


async def test_run_pipeline_tool_happy_path():
    store = _FakePipelineStore()
    events = []
    disp = _make_dispatcher({"lit": _good_runner}, events, pipelines=store)
    tools = build_dispatch_tools(disp)
    run_pipe = next(t for t in tools if t.name == "run_pipeline")

    msg = await run_pipe.ainvoke({
        "name": "lit-meth",
        "stages": [
            {"agent": "lit", "task": "step 1"},
            {"agent": "lit", "task": "step 2"},
        ],
    })
    assert "Pipeline #" in msg
    assert "stage 0" in msg.lower() or "task #" in msg.lower()
    assert len(store.rows) == 1


async def test_pipeline_status_tool():
    store = _FakePipelineStore()
    pid = await store.create(
        "my-pipe",
        [{"agent": "lit", "task": "do research", "task_id": 3}],
    )
    store.rows[pid]["stages"][0]["task_id"] = 3
    store.rows[pid]["status"] = "running"

    events = []
    disp = _make_dispatcher({"lit": _good_runner}, events, pipelines=store)
    tools = build_dispatch_tools(disp)
    status_tool = next(t for t in tools if t.name == "pipeline_status")

    msg = await status_tool.ainvoke({"pipeline_id": pid})
    assert "my-pipe" in msg
    assert "running" in msg
    assert "task #3" in msg
