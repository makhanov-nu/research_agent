"""Tests for the bounded auto-retry feature in ExperimentRunner.

Fakes mirror the patterns in test_experiments.py (no DB / SSH / LLM).
"""

from __future__ import annotations

import pytest

from research_agent.experiments.runner import ExperimentRunner
from research_agent.experiments.types import JobHandle, JobState, JobStatus
from research_agent.experiments.workspace import Workspace


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------

class _FakeEpisodic:
    enabled = True

    def __init__(self):
        self.rows: dict[int, dict] = {}
        self._next = 1
        self.logged_actions: list[tuple] = []

    async def create_experiment(self, title, channel_id=None, hypothesis="",
                                config=None, dataset="", code_ref=""):
        exp_id = self._next
        self._next += 1
        self.rows[exp_id] = {
            "id": exp_id, "title": title, "channel_id": channel_id,
            "hypothesis": hypothesis, "config": config or {}, "status": "planned",
            "metrics": {}, "artifacts": [],
        }
        return exp_id

    async def update_experiment(self, experiment_id, **fields):
        row = self.rows[experiment_id]
        for k, v in fields.items():
            row[k] = v

    async def get_experiment(self, experiment_id):
        return self.rows.get(experiment_id)

    async def list_active_experiments(self):
        return [r for r in self.rows.values() if r["status"] == "running"]

    async def log_action(self, *args, **kwargs):
        self.logged_actions.append((args, kwargs))


class _FakeBackend:
    name = "fake"
    configured = True

    def __init__(self, initial_state: JobState = JobState.FAILED):
        self.submitted: list = []
        self.state = initial_state
        self.host, self.user, self.port, self.key = "h", "u", 22, ""
        self._submit_count = 0

    def set_target(self, host, user=None, port=None, key=None):
        self.host = host
        if user:
            self.user = user

    async def submit(self, spec, workspace_local):
        self._submit_count += 1
        self.submitted.append(spec)
        cid = f"deadbeef{self._submit_count:02d}"
        return JobHandle(backend=self.name, container_id=cid, raw={"output_remote": "/o"})

    async def status(self, handle):
        return JobStatus(self.state,
                         exit_code=0 if self.state is JobState.SUCCEEDED else 1)

    async def logs(self, handle, tail=200):
        return "ImportError: No module named 'foo'"

    async def cancel(self, handle):
        pass

    async def fetch_artifacts(self, handle, dest_local):
        return []


class _FakeCoder:
    """Records calls; returns a fixed patched file set."""

    def __init__(self):
        self.revise_calls: list[dict] = []

    async def revise(self, spec, files, logs, lessons=""):
        self.revise_calls.append(
            {"spec": spec, "files": files, "logs": logs, "lessons": lessons}
        )
        # Return a "fixed" version of whatever files were passed in
        return {path: content + "# patched\n" for path, content in files.items()}


# ---------------------------------------------------------------------------
# Helper: build a runner and put an experiment into a running+failed state
# ---------------------------------------------------------------------------

def _make_runner(tmp_path, episodic, backend, coder=None):
    return ExperimentRunner(
        episodic, backend,
        Workspace(str(tmp_path / "ws")),
        str(tmp_path / "art"),
        coder=coder,
    )


async def _setup_failed_exp(tmp_path, ep, be, coder=None):
    """Create an experiment, write code, put it in a failed running state."""
    runner = _make_runner(tmp_path, ep, be, coder=coder)
    exp_id = await runner.propose("TestExp", "H", "plan text", channel_id="c1")
    await runner.write_code(exp_id, {"train.py": "print('hello')\n"})
    # Simulate a completed-but-failed run (has a handle, status=running so
    # list_active_experiments returns it, and the fake backend returns FAILED).
    ep.rows[exp_id]["status"] = "running"
    ep.rows[exp_id]["config"]["handle"] = JobHandle(
        be.name, "deadbeef00", {"output_remote": "/o"}
    ).to_dict()
    ep.rows[exp_id]["config"]["image"] = "research-agent/experiment:latest"
    ep.rows[exp_id]["config"]["command"] = ["python", "train.py"]
    ep.rows[exp_id]["config"]["resources"] = {
        "gpus": "all", "memory": "", "pids_limit": 0, "time_limit_minutes": 0
    }
    return runner, exp_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_auto_retry_on_failure_triggers_revise_and_relaunch(tmp_path, monkeypatch):
    """A failed run must invoke coder.revise() and relaunch; counter increments."""
    from research_agent import config as cfg
    monkeypatch.setattr(cfg.settings, "experiment_auto_retry", 2)
    monkeypatch.setattr(cfg.settings, "mlflow_enabled", False)
    monkeypatch.setattr(cfg.settings, "hf_token", "")

    coder = _FakeCoder()
    ep, be = _FakeEpisodic(), _FakeBackend(JobState.FAILED)
    runner, exp_id = await _setup_failed_exp(tmp_path, ep, be, coder=coder)

    changes = await runner.poll_active()

    # A retry was launched — poller returns ONE StateChange with "auto-retry" in msg.
    assert len(changes) == 1
    assert "auto-retry" in changes[0].message
    assert changes[0].state == "auto_retry"

    # The coder was called once with the failure logs.
    assert len(coder.revise_calls) == 1
    assert "ImportError" in coder.revise_calls[0]["logs"]

    # The backend submitted exactly one container (the retry): the original
    # run's handle was injected directly into the config, not via submit().
    assert len(be.submitted) == 1
    assert be._submit_count == 1

    # Counter incremented in config JSONB.
    assert ep.rows[exp_id]["config"]["auto_retry_count"] == 1

    # Experiment status is "running" again (retry in flight).
    assert ep.rows[exp_id]["status"] == "running"

    # An auto-retry action was logged.
    logged_names = [a[0][0] for a in ep.logged_actions]
    assert "experiment_auto_retry" in logged_names


async def test_auto_retry_cap_respected(tmp_path, monkeypatch):
    """When retry counter equals the cap, no retry is attempted."""
    from research_agent import config as cfg
    monkeypatch.setattr(cfg.settings, "experiment_auto_retry", 2)
    monkeypatch.setattr(cfg.settings, "mlflow_enabled", False)
    monkeypatch.setattr(cfg.settings, "hf_token", "")

    coder = _FakeCoder()
    ep, be = _FakeEpisodic(), _FakeBackend(JobState.FAILED)
    runner, exp_id = await _setup_failed_exp(tmp_path, ep, be, coder=coder)

    # Pre-seed the counter at the cap.
    ep.rows[exp_id]["config"]["auto_retry_count"] = 2

    changes = await runner.poll_active()

    # Falls through to normal failure — no retry StateChange.
    assert len(changes) == 1
    assert changes[0].state == "failed"
    assert "auto-retry" not in changes[0].message
    assert coder.revise_calls == []
    assert be._submit_count == 0  # no new container submitted


async def test_auto_retry_disabled_at_zero(tmp_path, monkeypatch):
    """experiment_auto_retry=0 means the feature is completely off."""
    from research_agent import config as cfg
    monkeypatch.setattr(cfg.settings, "experiment_auto_retry", 0)
    monkeypatch.setattr(cfg.settings, "mlflow_enabled", False)
    monkeypatch.setattr(cfg.settings, "hf_token", "")

    coder = _FakeCoder()
    ep, be = _FakeEpisodic(), _FakeBackend(JobState.FAILED)
    runner, exp_id = await _setup_failed_exp(tmp_path, ep, be, coder=coder)

    changes = await runner.poll_active()

    assert len(changes) == 1
    assert changes[0].state == "failed"
    assert coder.revise_calls == []
    assert be._submit_count == 0


async def test_cancelled_run_never_retried(tmp_path, monkeypatch):
    """A CANCELLED run must not trigger auto-retry even when budget remains."""
    from research_agent import config as cfg
    monkeypatch.setattr(cfg.settings, "experiment_auto_retry", 3)
    monkeypatch.setattr(cfg.settings, "mlflow_enabled", False)
    monkeypatch.setattr(cfg.settings, "hf_token", "")

    coder = _FakeCoder()
    ep, be = _FakeEpisodic(), _FakeBackend(JobState.CANCELLED)
    runner, exp_id = await _setup_failed_exp(tmp_path, ep, be, coder=coder)

    changes = await runner.poll_active()

    assert len(changes) == 1
    assert changes[0].state == "cancelled"
    assert coder.revise_calls == []
    assert be._submit_count == 0


async def test_revise_exception_falls_through_to_normal_failure(tmp_path, monkeypatch):
    """If coder.revise() raises, the normal failure path still runs."""
    from research_agent import config as cfg
    monkeypatch.setattr(cfg.settings, "experiment_auto_retry", 2)
    monkeypatch.setattr(cfg.settings, "mlflow_enabled", False)
    monkeypatch.setattr(cfg.settings, "hf_token", "")

    class _BrokenCoder:
        async def revise(self, *args, **kwargs):
            raise RuntimeError("LLM is down")

    ep, be = _FakeEpisodic(), _FakeBackend(JobState.FAILED)
    runner, exp_id = await _setup_failed_exp(tmp_path, ep, be, coder=_BrokenCoder())

    # Must not propagate the exception; must report failure.
    changes = await runner.poll_active()

    assert len(changes) == 1
    assert changes[0].state == "failed"
    assert be._submit_count == 0  # no retry launched


async def test_auto_retry_state_change_message_contains_keyword(tmp_path, monkeypatch):
    """StateChange.message for a retry must contain 'auto-retry'."""
    from research_agent import config as cfg
    monkeypatch.setattr(cfg.settings, "experiment_auto_retry", 1)
    monkeypatch.setattr(cfg.settings, "mlflow_enabled", False)
    monkeypatch.setattr(cfg.settings, "hf_token", "")

    coder = _FakeCoder()
    ep, be = _FakeEpisodic(), _FakeBackend(JobState.FAILED)
    runner, exp_id = await _setup_failed_exp(tmp_path, ep, be, coder=coder)

    changes = await runner.poll_active()

    assert len(changes) == 1
    assert "auto-retry" in changes[0].message
    assert "1/1" in changes[0].message


async def test_no_coder_skips_retry(tmp_path, monkeypatch):
    """When no coder is available and build_default_coder returns None, skip retry."""
    from research_agent import config as cfg
    monkeypatch.setattr(cfg.settings, "experiment_auto_retry", 2)
    monkeypatch.setattr(cfg.settings, "mlflow_enabled", False)
    monkeypatch.setattr(cfg.settings, "hf_token", "")
    # Ensure build_default_coder returns None (no openrouter key).
    monkeypatch.setattr(cfg.settings, "openrouter_api_key", "")

    ep, be = _FakeEpisodic(), _FakeBackend(JobState.FAILED)
    # coder=None → runner will try build_default_coder() which returns None
    runner, exp_id = await _setup_failed_exp(tmp_path, ep, be, coder=None)

    changes = await runner.poll_active()

    assert len(changes) == 1
    assert changes[0].state == "failed"
    assert be._submit_count == 0


async def test_get_status_shows_retry_count(tmp_path, monkeypatch):
    """experiment_status includes auto_retries=N when N > 0."""
    from research_agent import config as cfg
    monkeypatch.setattr(cfg.settings, "mlflow_enabled", False)

    ep, be = _FakeEpisodic(), _FakeBackend()
    runner = _make_runner(tmp_path, ep, be)
    exp_id = await runner.propose("RetryExp", "H", "P", channel_id=None)
    ep.rows[exp_id]["config"]["auto_retry_count"] = 1

    status = await runner.get_status(exp_id)
    assert "auto_retries=1" in status


async def test_get_status_hides_retry_count_at_zero(tmp_path, monkeypatch):
    """experiment_status does NOT show auto_retries when count is 0."""
    from research_agent import config as cfg
    monkeypatch.setattr(cfg.settings, "mlflow_enabled", False)

    ep, be = _FakeEpisodic(), _FakeBackend()
    runner = _make_runner(tmp_path, ep, be)
    exp_id = await runner.propose("CleanExp", "H", "P", channel_id=None)

    status = await runner.get_status(exp_id)
    assert "auto_retries" not in status
