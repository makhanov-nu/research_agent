"""Tests for the experiment-runner additions: MLflow, coder, ephemeral host."""

from __future__ import annotations

from research_agent.experiments.coder import parse_files
from research_agent.experiments.mlflow import (
    api_url,
    best_run,
    build_mlflow_server_command,
    parse_metrics,
    parse_params,
    run_name,
)
from research_agent.experiments.runner import _launch_env
from research_agent.experiments.ssh_docker import (
    SSHDockerBackend,
    build_docker_run_command,
)

# --- MLflow helpers -----------------------------------------------------------


def test_build_mlflow_server_command():
    argv = build_mlflow_server_command(
        container="ra-mlflow", image="mlflow:img", network="ra-net",
        port=5000, volume="ra-mlflow-data",
    )
    assert argv[:3] == ["docker", "run", "-d"]
    assert "--network" in argv and "ra-net" in argv
    assert "127.0.0.1:5000:5000" in argv
    assert "ra-mlflow-data:/mlflow" in argv
    assert argv[-2:] == ["sqlite:////mlflow/mlflow.db", "/mlflow/artifacts"] or (
        "mlflow" in argv and "server" in argv
    )


def test_api_url_and_run_name():
    assert api_url(5000, "runs/search") == "http://127.0.0.1:5000/api/2.0/mlflow/runs/search"
    assert api_url(5000, "/runs/get").endswith("/runs/get")
    assert run_name(7) == "exp_7"


def test_parse_metrics_and_params():
    run = {"data": {
        "metrics": [{"key": "acc", "value": 0.9}, {"key": "loss", "value": 0.1}],
        "params": [{"key": "lr", "value": "0.01"}],
    }}
    assert parse_metrics(run) == {"acc": 0.9, "loss": 0.1}
    assert parse_params(run) == {"lr": "0.01"}
    assert parse_metrics({}) == {}


def test_best_run_max_and_min():
    resp = {"runs": [
        {"info": {"run_id": "a", "run_name": "t1"}, "data": {"metrics": [{"key": "acc", "value": 0.8}]}},
        {"info": {"run_id": "b", "run_name": "t2"}, "data": {"metrics": [{"key": "acc", "value": 0.95}]}},
        {"info": {"run_id": "c", "run_name": "t3"}, "data": {"metrics": []}},  # ignored
    ]}
    assert best_run(resp, "acc", "max")["run_id"] == "b"
    assert best_run(resp, "acc", "min")["run_id"] == "a"
    assert best_run({"runs": []}, "acc") is None
    assert best_run(resp, "missing") is None


# --- coder file parsing -------------------------------------------------------


def test_parse_files_splits_marked_blocks():
    text = (
        "preamble\n"
        "=== FILE: train.py ===\n"
        "import os\nprint('hi')\n"
        "=== FILE: requirements.txt ===\n"
        "optuna\nmlflow\n"
    )
    files = parse_files(text)
    assert set(files) == {"train.py", "requirements.txt"}
    assert files["train.py"].startswith("import os")
    assert "optuna" in files["requirements.txt"]


def test_parse_files_empty_when_no_markers():
    assert parse_files("just prose, no files") == {}


# --- docker run with network + extra volumes ----------------------------------


def test_docker_run_includes_network_and_volumes():
    argv = build_docker_run_command(
        name="ra_exp_1", image="img", command=["python", "train.py"],
        workspace_remote="/w", output_remote="/o", gpus="all",
        network="ra-net", volumes=["ra-hf-cache:/root/.cache/huggingface"],
    )
    assert "--network" in argv and "ra-net" in argv
    assert "ra-hf-cache:/root/.cache/huggingface" in argv
    # default (no network/volumes) stays clean
    bare = build_docker_run_command(
        name="x", image="img", command=["run"], workspace_remote="/w",
        output_remote="/o",
    )
    assert "--network" not in bare


# --- launch env injection -----------------------------------------------------


def test_launch_env_injects_mlflow_and_respects_overrides(monkeypatch):
    from research_agent.experiments import runner as runner_mod

    monkeypatch.setattr(runner_mod.settings, "mlflow_enabled", True)
    monkeypatch.setattr(runner_mod.settings, "hf_token", "hf_xxx")
    env = _launch_env(5, {"MLFLOW_RUN_NAME": "custom", "FOO": "bar"})
    assert env["MLFLOW_TRACKING_URI"].startswith("http://")
    assert env["HF_TOKEN"] == "hf_xxx"
    assert env["FOO"] == "bar"
    assert env["MLFLOW_RUN_NAME"] == "custom"  # explicit env overrides default


# --- ephemeral host target ----------------------------------------------------


def test_set_target_parses_user_at_host():
    b = SSHDockerBackend()
    b.set_target("ubuntu@1.2.3.4")
    assert b.user == "ubuntu" and b.host == "1.2.3.4"
    assert b.configured
    b.set_target("5.6.7.8", user="root")
    assert b.user == "root" and b.host == "5.6.7.8"
