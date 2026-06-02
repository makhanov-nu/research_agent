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
    async def on_complete(task_id, agent, status, payload, channel_id):
        events.append((task_id, agent, status, payload, channel_id))

    return TaskDispatcher(runners, _FakeTaskStore(), on_complete, max_parallel=2)


async def test_dispatch_runs_in_background_and_pushes_event():
    async def good_runner(task):
        return f"done: {task}", [{"type": "ai", "content": task}]

    events = []
    disp = _dispatcher({"lit": good_runner}, events)
    tid = await disp.dispatch("lit", "find X", channel_id="c1")
    assert disp.task_store.rows[tid]["status"] in {"pending", "running"}  # not blocking
    await disp.join()
    row = disp.task_store.rows[tid]
    assert row["status"] == "done"
    assert row["result"] == "done: find X"
    assert row["trace"] == [{"type": "ai", "content": "find X"}]
    # completion is pushed (task_id, agent, status, payload, channel)
    assert events == [(tid, "lit", "done", "done: find X", "c1")]


async def test_dispatch_failure_is_recorded_and_pushed():
    async def bad_runner(task):
        raise RuntimeError("boom")

    events = []
    disp = _dispatcher({"lit": bad_runner}, events)
    tid = await disp.dispatch("lit", "x", channel_id="c1")
    await disp.join()
    assert disp.task_store.rows[tid]["status"] == "failed"
    assert "boom" in disp.task_store.rows[tid]["error"]
    assert events and events[0][2] == "failed" and "boom" in events[0][3]


async def test_dispatch_unknown_agent_raises():
    disp = _dispatcher({"lit": None}, [])
    with pytest.raises(ValueError, match="Unknown agent"):
        await disp.dispatch("nope", "x", channel_id="c1")


async def test_concurrency_capped_by_semaphore():
    active = 0
    peak = 0

    async def slow_runner(task):
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


async def test_dispatch_tool_is_the_only_tool_and_submits():
    async def good_runner(task):
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
