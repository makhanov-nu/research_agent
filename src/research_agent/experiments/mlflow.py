"""MLflow tracking integration (server runs on the GPU box).

A single MLflow tracking server runs as a Docker container on the compute node,
bound to 127.0.0.1 and joined to the shared `compute_network` so experiment
containers reach it by name (`http://ra-mlflow:5000`). The bot reads runs/metrics
back through the MLflow REST API, tunnelled over SSH (see `SSHDockerBackend`).

Everything here is pure (command construction, URL building, JSON parsing) so it
is unit-tested without a live server.
"""

from __future__ import annotations

from typing import Any, Optional


def build_mlflow_server_command(
    *, container: str, image: str, network: str, port: int, volume: str
) -> list[str]:
    """argv for a detached MLflow tracking server backed by a named volume.

    sqlite backend store + local artifact root both live in `/mlflow` (the named
    volume), so the server is self-contained and survives restarts.
    """
    return [
        "docker", "run", "-d",
        "--name", container,
        "--network", network,
        "--restart", "unless-stopped",
        "-p", f"127.0.0.1:{port}:{port}",
        "-v", f"{volume}:/mlflow",
        image,
        "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--backend-store-uri", "sqlite:////mlflow/mlflow.db",
        "--artifacts-destination", "/mlflow/artifacts",
    ]


def run_name(experiment_id: int) -> str:
    """Stable MLflow run name the runner uses to find an experiment's run."""
    return f"exp_{experiment_id}"


def api_url(port: int, path: str) -> str:
    """Build a localhost MLflow REST URL (called on the remote, over SSH)."""
    return f"http://127.0.0.1:{port}/api/2.0/mlflow/{path.lstrip('/')}"


def parse_metrics(run: dict) -> dict[str, float]:
    """Latest value per metric key from a `runs/get` (or search) run object."""
    data = (run or {}).get("data") or {}
    out: dict[str, float] = {}
    for m in data.get("metrics") or []:
        key = m.get("key")
        if key is not None:
            out[key] = m.get("value")
    return out


def parse_params(run: dict) -> dict[str, str]:
    data = (run or {}).get("data") or {}
    return {
        p.get("key"): p.get("value")
        for p in (data.get("params") or [])
        if p.get("key") is not None
    }


def _run_metric(run: dict, metric: str) -> Optional[float]:
    return parse_metrics(run).get(metric)


def best_run(
    search_response: dict, metric: str, mode: str = "max"
) -> Optional[dict[str, Any]]:
    """Pick the best run (by `metric`) from a `runs/search` response.

    Returns `{"run_id", "run_name", "value", "metrics", "params"}` or None.
    Runs missing the metric are ignored; ties keep the first seen.
    """
    runs = (search_response or {}).get("runs") or []
    best: Optional[dict] = None
    best_val: Optional[float] = None
    for r in runs:
        val = _run_metric(r, metric)
        if val is None:
            continue
        better = (
            best_val is None
            or (mode == "max" and val > best_val)
            or (mode == "min" and val < best_val)
        )
        if better:
            best_val = val
            info = r.get("info") or {}
            best = {
                "run_id": info.get("run_id") or info.get("run_uuid"),
                "run_name": info.get("run_name"),
                "value": val,
                "metrics": parse_metrics(r),
                "params": parse_params(r),
            }
    return best
