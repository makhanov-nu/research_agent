"""Tests for the experiment runner's pure logic and orchestration flow."""

from __future__ import annotations

import json

import pytest

from research_agent.experiments.runner import ExperimentRunner, read_latest_metrics
from research_agent.experiments.ssh_docker import (
    build_docker_run_command,
    parse_inspect_status,
)
from research_agent.experiments.types import JobHandle, JobState, JobStatus
from research_agent.experiments.workspace import Workspace


# --- docker command builder ----------------------------------------------------

def test_build_docker_run_command_basics():
    argv = build_docker_run_command(
        name="ra_exp_1", image="img:tag", command=["python", "train.py"],
        workspace_remote="/w", output_remote="/o", gpus="all", memory="8g",
        pids_limit=512, env_file_remote="/w/.env.job",
    )
    assert argv[:5] == ["docker", "run", "-d", "--name", "ra_exp_1"]
    assert "--gpus" in argv and "all" in argv
    assert "--memory" in argv and "8g" in argv
    assert "--env-file" in argv
    assert "/w:/workspace:ro" in argv
    assert "/o:/output" in argv
    # image precedes its command
    assert argv.index("img:tag") < argv.index("python")


def test_build_docker_run_command_omits_optional_flags():
    argv = build_docker_run_command(
        name="x", image="img", command=["run"], workspace_remote="/w",
        output_remote="/o",
    )
    assert "--gpus" not in argv
    assert "--memory" not in argv
    assert "--env-file" not in argv


# --- docker inspect status parsing ---------------------------------------------

@pytest.mark.parametrize(
    "raw,expected_state,expected_code",
    [
        ("running;0", JobState.RUNNING, 0),
        ("created;0", JobState.RUNNING, 0),
        ("exited;0", JobState.SUCCEEDED, 0),
        ("exited;1", JobState.FAILED, 1),
        ("dead;137", JobState.FAILED, 137),
        ("", JobState.UNKNOWN, None),
    ],
)
def test_parse_inspect_status(raw, expected_state, expected_code):
    status = parse_inspect_status(raw)
    assert status.state is expected_state
    assert status.exit_code == expected_code


# --- workspace path safety -----------------------------------------------------

def test_workspace_writes_files(tmp_path):
    ws = Workspace(str(tmp_path))
    written = ws.write_files(7, {"train.py": "print(1)", "sub/util.py": "x=1"})
    assert written == ["sub/util.py", "train.py"]
    assert (tmp_path / "exp_7" / "train.py").read_text() == "print(1)"


def test_workspace_rejects_traversal(tmp_path):
    ws = Workspace(str(tmp_path))
    with pytest.raises(ValueError, match="Unsafe"):
        ws.write_files(7, {"../escape.py": "bad"})


def test_workspace_rejects_absolute(tmp_path):
    ws = Workspace(str(tmp_path))
    with pytest.raises(ValueError, match="Unsafe"):
        ws.write_files(7, {"/etc/passwd": "bad"})


# --- metrics parsing -----------------------------------------------------------

def test_read_latest_metrics_returns_last_object(tmp_path):
    d = tmp_path / "exp_1"
    d.mkdir()
    (d / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "loss": 0.5}) + "\n" + json.dumps({"step": 2, "loss": 0.2}) + "\n"
    )
    assert read_latest_metrics(str(d)) == {"step": 2, "loss": 0.2}


def test_read_latest_metrics_missing_file(tmp_path):
    assert read_latest_metrics(str(tmp_path)) == {}


# --- runner orchestration (fakes, no DB/SSH) -----------------------------------

class _FakeEpisodic:
    enabled = True

    def __init__(self):
        self.rows: dict[int, dict] = {}
        self._next = 1

    async def create_experiment(self, title, channel_id=None, hypothesis="", config=None, dataset="", code_ref=""):
        exp_id = self._next
        self._next += 1
        self.rows[exp_id] = {
            "id": exp_id, "title": title, "channel_id": channel_id,
            "hypothesis": hypothesis, "config": config or {}, "status": "planned",
            "metrics": {}, "artifacts": [],
        }
        return exp_id

    async def update_experiment(self, experiment_id, **fields):
        self.rows[experiment_id].update(fields)

    async def get_experiment(self, experiment_id):
        return self.rows.get(experiment_id)

    async def list_active_experiments(self):
        # Mirror production: only 'running' experiments are still in flight.
        return [r for r in self.rows.values() if r["status"] == "running"]

    async def log_action(self, *a, **k):
        pass


class _FakeBackend:
    name = "fake"
    configured = True  # stands in for an attached GPU box

    def __init__(self):
        self.submitted = []
        self.cancelled = []
        self.state = JobState.RUNNING
        self.host, self.user, self.port, self.key = "h", "u", 22, ""

    def set_target(self, host, user=None, port=None, key=None):
        self.host = host
        if user:
            self.user = user

    async def submit(self, spec, workspace_local):
        self.submitted.append(spec)
        return JobHandle(backend=self.name, container_id="deadbeef", raw={"output_remote": "/o"})

    async def status(self, handle):
        return JobStatus(self.state, exit_code=0 if self.state is JobState.SUCCEEDED else None)

    async def logs(self, handle, tail=200):
        return "log line"

    async def cancel(self, handle):
        self.cancelled.append(handle)

    async def fetch_artifacts(self, handle, dest_local):
        return []


def _runner(tmp_path, episodic, backend):
    return ExperimentRunner(
        episodic, backend, Workspace(str(tmp_path / "ws")), str(tmp_path / "art")
    )


async def test_launch_requires_approval_then_runs(tmp_path, monkeypatch):
    from research_agent import config as cfg
    monkeypatch.setattr(cfg.settings, "experiment_require_approval", True)
    monkeypatch.setattr(cfg.settings, "compute_base_image", "img:1")
    monkeypatch.setattr(cfg.settings, "compute_default_gpus", "all")

    ep, be = _FakeEpisodic(), _FakeBackend()
    runner = _runner(tmp_path, ep, be)

    exp_id = await runner.propose("T", "H", "P", channel_id="c1")
    await runner.write_code(exp_id, {"train.py": "print(1)"})

    msg = await runner.request_launch(exp_id, "python train.py")
    assert "awaiting approval" in msg
    assert ep.rows[exp_id]["status"] == "pending_approval"
    assert not be.submitted  # not launched yet

    out = await runner.approve_and_launch(exp_id)
    assert "launched" in out
    assert ep.rows[exp_id]["status"] == "running"
    assert be.submitted and be.submitted[0].command == ["python", "train.py"]


async def test_launch_blocked_without_code(tmp_path):
    ep, be = _FakeEpisodic(), _FakeBackend()
    runner = _runner(tmp_path, ep, be)
    exp_id = await runner.propose("T", "H", "P", channel_id=None)
    msg = await runner.request_launch(exp_id, "python train.py")
    assert "no code" in msg
    assert ep.rows[exp_id]["status"] == "planned"


async def test_poll_active_reports_completion(tmp_path):
    ep, be = _FakeEpisodic(), _FakeBackend()
    runner = _runner(tmp_path, ep, be)
    exp_id = await runner.propose("T", "H", "P", channel_id="c1")
    await runner.write_code(exp_id, {"train.py": "x"})
    # mark running with a handle, as approve_and_launch would have
    ep.rows[exp_id]["status"] = "running"
    ep.rows[exp_id]["config"]["handle"] = JobHandle(
        be.name, "deadbeef", {"output_remote": "/o"}
    ).to_dict()

    be.state = JobState.SUCCEEDED
    changes = await runner.poll_active()
    assert len(changes) == 1
    assert changes[0].experiment_id == exp_id
    assert changes[0].state == "succeeded"
    assert ep.rows[exp_id]["status"] == "succeeded"


def _mark_running(ep, be, exp_id, *, minutes_ago, limit):
    """Put an experiment into the running state, launched `minutes_ago` with a
    `limit`-minute wall-clock budget."""
    from datetime import datetime, timedelta, timezone

    ep.rows[exp_id]["status"] = "running"
    ep.rows[exp_id]["config"]["handle"] = JobHandle(
        be.name, "deadbeef", {"output_remote": "/o"}
    ).to_dict()
    ep.rows[exp_id]["config"]["resources"] = {"time_limit_minutes": limit}
    ep.rows[exp_id]["config"]["started_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    ).isoformat()


async def test_poll_enforces_time_limit(tmp_path):
    """A still-running job past its wall-clock budget is stopped and failed."""
    ep, be = _FakeEpisodic(), _FakeBackend()
    runner = _runner(tmp_path, ep, be)
    exp_id = await runner.propose("T", "H", "P", channel_id="c1")
    await runner.write_code(exp_id, {"train.py": "x"})
    _mark_running(ep, be, exp_id, minutes_ago=30, limit=5)
    be.state = JobState.RUNNING  # backend still reports it alive

    changes = await runner.poll_active()
    assert len(changes) == 1
    assert changes[0].state == "failed"
    assert "timed out" in changes[0].message
    assert ep.rows[exp_id]["status"] == "failed"
    assert be.cancelled, "the runaway container should have been stopped"


async def test_poll_leaves_job_running_within_budget(tmp_path):
    """A running job still inside its budget is left alone (no false-positive kill)."""
    ep, be = _FakeEpisodic(), _FakeBackend()
    runner = _runner(tmp_path, ep, be)
    exp_id = await runner.propose("T", "H", "P", channel_id="c1")
    await runner.write_code(exp_id, {"train.py": "x"})
    _mark_running(ep, be, exp_id, minutes_ago=1, limit=60)
    be.state = JobState.RUNNING

    assert await runner.poll_active() == []
    assert not be.cancelled
    assert ep.rows[exp_id]["status"] == "running"


async def test_poll_keeps_running_when_timeout_cancel_fails(tmp_path):
    """If stopping a timed-out run fails, it stays running so the next poll retries
    (rather than being marked failed and dropping out of the active filter -> a
    container that keeps holding the GPU)."""

    class _FlakyBackend(_FakeBackend):
        cancel_should_fail = True

        async def cancel(self, handle):
            if self.cancel_should_fail:
                raise RuntimeError("ssh down")
            await super().cancel(handle)

    ep, be = _FakeEpisodic(), _FlakyBackend()
    runner = _runner(tmp_path, ep, be)
    exp_id = await runner.propose("T", "H", "P", channel_id="c1")
    await runner.write_code(exp_id, {"train.py": "x"})
    _mark_running(ep, be, exp_id, minutes_ago=30, limit=5)
    be.state = JobState.RUNNING

    # First poll: cancel fails -> not terminal, still running, note recorded.
    assert await runner.poll_active() == []
    assert not be.cancelled
    assert ep.rows[exp_id]["status"] == "running"
    assert "cancellation failed" in (ep.rows[exp_id].get("notes") or "")

    # Next poll: cancel succeeds -> the run is stopped, failed, and reported.
    be.cancel_should_fail = False
    changes = await runner.poll_active()
    assert len(changes) == 1 and changes[0].state == "failed"
    assert be.cancelled
    assert ep.rows[exp_id]["status"] == "failed"
