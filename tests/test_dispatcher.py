"""Tests for the background task dispatcher."""

from __future__ import annotations

import asyncio

import pytest

from research_agent.agents.dispatcher import TaskDispatcher, build_dispatch_tools


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
        self.rows[tid]["status"] = "running"

    async def finish(self, tid, result, trace):
        self.rows[tid].update(status="done", result=result, trace=trace)

    async def fail(self, tid, error, trace):
        self.rows[tid].update(status="failed", error=error, trace=trace)

    async def get(self, tid):
        return self.rows.get(tid)


def _dispatcher(runners, events):
    # The completion trigger carries no result — only the task id/agent/status/
    # channel. The result lives in the task store, read back by the handler.
    async def on_complete(task_id, agent, status, channel_id):
        events.append((task_id, agent, status, channel_id))

    return TaskDispatcher(runners, _FakeTaskStore(), on_complete, max_parallel=2)


async def test_dispatch_runs_in_background_and_pushes_event():
    async def good_runner(task, channel_id=None):
        return f"done: {task}", [{"type": "ai", "content": task}]

    events = []
    disp = _dispatcher({"lit": good_runner}, events)
    tid = await disp.dispatch("lit", "find X", channel_id="c1")
    assert disp.task_store.rows[tid]["status"] in {"pending", "running"}  # not blocking
    await disp.join()
    row = disp.task_store.rows[tid]
    assert row["status"] == "done"
    # The result + trace are written to the dashboard (the single source of truth)...
    assert row["result"] == "done: find X"
    assert row["trace"] == [{"type": "ai", "content": "find X"}]
    # ...while the completion event is a pure trigger (no result payload).
    assert events == [(tid, "lit", "done", "c1")]


async def test_dispatch_failure_is_recorded_and_pushed():
    async def bad_runner(task, channel_id=None):
        raise RuntimeError("boom")

    events = []
    disp = _dispatcher({"lit": bad_runner}, events)
    tid = await disp.dispatch("lit", "x", channel_id="c1")
    await disp.join()
    assert disp.task_store.rows[tid]["status"] == "failed"
    # The error is recorded in the dashboard; the trigger just signals "failed".
    assert "boom" in disp.task_store.rows[tid]["error"]
    assert events and events[0][2] == "failed"


async def test_dispatch_unknown_agent_raises():
    disp = _dispatcher({"lit": None}, [])
    with pytest.raises(ValueError, match="Unknown agent"):
        await disp.dispatch("nope", "x", channel_id="c1")


async def test_concurrency_capped_by_semaphore():
    active = 0
    peak = 0

    async def slow_runner(task, channel_id=None):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return "ok", []

    events = []
    disp = _dispatcher({"s": slow_runner}, events)  # max_parallel=2
    for _ in range(5):
        await disp.dispatch("s", "t", channel_id="c")
    await disp.join()
    assert peak <= 2


async def test_build_runners_route_artifacts_into_project(tmp_path):
    """A dispatched artifact run saves into the originating channel's project."""
    from research_agent.agents.dispatcher import build_runners
    from research_agent.projects.store import ProjectStore

    class _FakeWriter:
        async def draft(self, task, dirpath=None):
            p = dirpath / "out.tex"
            p.write_text("\\section{X}")
            return {"tex_path": str(p), "bib_path": "", "n_refs": 0, "latex": ""}

    class _Writers:
        reviewer = _FakeWriter()
        methodologist = _FakeWriter()
        paper_writer = _FakeWriter()

    projects = ProjectStore(pool=None, output_dir=str(tmp_path))
    runners = build_runners(
        model=None, mcp_tools=[], writers=_Writers(), consortium=None, projects=projects,
    )
    summary, trace = await runners["literature_review"]("a topic", "chan-9")
    saved = list((tmp_path / "projects").rglob("lit_review/out.tex"))
    assert saved, "artifact not saved into a project folder"
    assert "project" in summary


async def test_dispatch_tool_is_the_only_tool_and_submits():
    async def good_runner(task, channel_id=None):
        return "the answer", []

    events = []
    disp = _dispatcher({"lit": good_runner}, events)
    tools = build_dispatch_tools(disp)
    assert [t.name for t in tools] == ["dispatch_task"]  # no polling tool

    msg = await tools[0].ainvoke(
        {"agent": "lit", "task": "q", "config": {"configurable": {"thread_id": "c1"}}}
    )
    assert "Dispatched task #1" in msg and "automatically" in msg
    await disp.join()
    assert events and events[0][2] == "done"
