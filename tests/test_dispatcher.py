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


def _dispatcher(runners, notifications):
    async def notify(channel_id, message):
        notifications.append((channel_id, message))

    return TaskDispatcher(runners, _FakeTaskStore(), notify, max_parallel=2)


async def test_dispatch_runs_in_background_and_notifies():
    async def good_runner(task):
        return f"done: {task}", [{"type": "ai", "content": task}]

    notes = []
    disp = _dispatcher({"lit": good_runner}, notes)
    tid = await disp.dispatch("lit", "find X", channel_id="c1")
    assert disp.task_store.rows[tid]["status"] in {"pending", "running"}  # not blocking
    await disp.join()
    row = disp.task_store.rows[tid]
    assert row["status"] == "done"
    assert row["result"] == "done: find X"
    assert row["trace"] == [{"type": "ai", "content": "find X"}]
    assert notes and notes[0][0] == "c1" and "finished" in notes[0][1]


async def test_dispatch_failure_is_recorded_and_notified():
    async def bad_runner(task):
        raise RuntimeError("boom")

    notes = []
    disp = _dispatcher({"lit": bad_runner}, notes)
    tid = await disp.dispatch("lit", "x", channel_id="c1")
    await disp.join()
    assert disp.task_store.rows[tid]["status"] == "failed"
    assert "boom" in disp.task_store.rows[tid]["error"]
    assert notes and "failed" in notes[0][1]


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

    notes = []
    disp = _dispatcher({"s": slow_runner}, notes)  # max_parallel=2
    for _ in range(5):
        await disp.dispatch("s", "t", channel_id="c")
    await disp.join()
    assert peak <= 2


async def test_get_task_result_tool_reports_status_and_result():
    async def good_runner(task):
        return "the answer", []

    notes = []
    disp = _dispatcher({"lit": good_runner}, notes)
    dispatch_tool, get_result_tool = build_dispatch_tools(disp)

    msg = await dispatch_tool.ainvoke(
        {"agent": "lit", "task": "q", "config": {"configurable": {"thread_id": "c1"}}}
    )
    assert "Dispatched task #1" in msg
    await disp.join()
    out = await get_result_tool.ainvoke({"task_id": 1})
    assert out == "the answer"
