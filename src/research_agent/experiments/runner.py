"""Orchestrates experiments: registry <-> workspace <-> compute backend.

State (status, JobSpec, backend handle, metrics, artifacts) lives in the
`experiments` registry row's columns + `config` JSONB, so nothing is lost across
restarts. The Discord bot drives this via tools and a periodic poller.
"""

from __future__ import annotations

import json
import logging
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import settings
from .types import (
    TERMINAL_STATES,
    JobHandle,
    JobSpec,
    JobState,
    Resources,
)

logger = logging.getLogger(__name__)


@dataclass
class StateChange:
    """A run that reached a terminal state since the last poll (for reporting)."""

    experiment_id: int
    channel_id: str | None
    title: str
    state: str
    message: str


def _launch_env(experiment_id: int, base_env: dict) -> dict:
    """Merge MLflow + HuggingFace env into the job's env (user values win)."""
    from . import mlflow

    injected: dict[str, str] = {}
    if settings.mlflow_enabled:
        injected["MLFLOW_TRACKING_URI"] = settings.mlflow_tracking_uri_internal
        injected["MLFLOW_EXPERIMENT_NAME"] = settings.mlflow_experiment_name
        injected["MLFLOW_RUN_NAME"] = mlflow.run_name(experiment_id)
    if settings.hf_token:
        injected["HF_TOKEN"] = settings.hf_token
        injected["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token
    injected.update(base_env or {})  # explicit per-experiment env overrides defaults
    return injected


def read_latest_metrics(artifact_dir: str) -> dict:
    """Return the last JSON object from a metrics.jsonl in the artifact dir."""
    path = Path(artifact_dir) / "metrics.jsonl"
    if not path.exists():
        return {}
    last = ""
    for line in path.read_text(errors="replace").splitlines():
        if line.strip():
            last = line
    if not last:
        return {}
    try:
        obj = json.loads(last)
        return obj if isinstance(obj, dict) else {"value": obj}
    except json.JSONDecodeError:
        return {}


class ExperimentRunner:
    def __init__(self, episodic, backend, workspace, artifacts_dir: str, projects=None):
        self.episodic = episodic
        self.backend = backend
        self.workspace = workspace
        self.artifacts_dir = artifacts_dir
        self.projects = projects

    @property
    def enabled(self) -> bool:
        # Tools are available whenever the registry is; a GPU box can be attached
        # at runtime (ephemeral, per-experiment) via `set_compute`.
        return self.backend is not None and getattr(self.episodic, "enabled", False)

    @property
    def attached(self) -> bool:
        """Whether a GPU box is currently attached (host configured)."""
        return bool(getattr(self.backend, "configured", False))

    # -- ephemeral GPU box (a fresh IP per experiment) --

    async def set_compute(self, host: str, key: str | None = None) -> str:
        self.backend.set_target(host, key=key)
        return f"Attached compute box `{self.backend.user}@{self.backend.host}`."

    async def provision(self) -> str:
        if not self.attached:
            return "No GPU box attached. Use `set_compute` / `!gpu <user@ip>` first."
        return await self.backend.provision()

    async def survey(self) -> str:
        if not self.attached:
            return "No GPU box attached."
        return await self.backend.survey()

    def _apply_target(self, cfg: dict) -> bool:
        """Point the backend at the box an experiment was launched on, if recorded."""
        target = cfg.get("compute")
        if target and target.get("host"):
            self.backend.set_target(
                target["host"], user=target.get("user"),
                port=target.get("port"), key=target.get("key"),
            )
            return True
        return self.attached

    # -- registry helpers --

    async def _config(self, experiment_id: int) -> dict | None:
        exp = await self.episodic.get_experiment(experiment_id)
        if exp is None:
            return None
        cfg = exp.get("config") or {}
        return cfg if isinstance(cfg, dict) else {}

    # -- lifecycle --

    async def propose(
        self, title: str, hypothesis: str, plan: str, channel_id: str | None
    ) -> int | None:
        exp_id = await self.episodic.create_experiment(
            title=title, channel_id=channel_id, hypothesis=hypothesis,
            config={"plan": plan},
        )
        await self.episodic.log_action(
            "experiment_proposed", title, channel_id=channel_id,
            metadata={"experiment_id": exp_id},
        )
        return exp_id

    async def write_code(self, experiment_id: int, files: dict[str, str]) -> list[str]:
        written = self.workspace.write_files(experiment_id, files)
        await self.episodic.log_action(
            "experiment_code_written", f"exp {experiment_id}: {', '.join(written)}",
            metadata={"experiment_id": experiment_id, "files": written},
        )
        return written

    async def request_launch(
        self, experiment_id: int, command: str, image: str = "",
        gpus: str | None = None, memory: str = "", time_limit_minutes: int = 0,
    ) -> str:
        cfg = await self._config(experiment_id)
        if cfg is None:
            return f"Experiment #{experiment_id} not found."
        if not self.attached:
            return (
                "No GPU box is attached. Attach one with `!gpu <user@ip>` "
                "(it'll be provisioned automatically), then request the launch."
            )
        if not self.workspace.list_files(experiment_id):
            return f"Experiment #{experiment_id} has no code yet — write code first."

        cfg.update(
            {
                "image": image or settings.experiment_image,
                "command": shlex.split(command),
                "resources": {
                    "gpus": settings.compute_default_gpus if gpus is None else gpus,
                    "memory": memory,
                    "pids_limit": 0,
                    "time_limit_minutes": time_limit_minutes,
                },
            }
        )

        if settings.experiment_require_approval:
            await self.episodic.update_experiment(
                experiment_id, config=cfg, status="pending_approval"
            )
            return (
                f"Experiment #{experiment_id} is ready and awaiting approval.\n"
                f"Image: `{cfg['image']}`  |  Command: `{command}`  |  "
                f"GPUs: `{cfg['resources']['gpus'] or 'none'}`\n"
                f"Approve with `!approve {experiment_id}` to launch."
            )
        await self.episodic.update_experiment(experiment_id, config=cfg)
        return await self.approve_and_launch(experiment_id)

    async def approve_and_launch(self, experiment_id: int) -> str:
        cfg = await self._config(experiment_id)
        if cfg is None:
            return f"Experiment #{experiment_id} not found."
        if "command" not in cfg:
            return f"Experiment #{experiment_id} has no launch spec — request a launch first."
        if not self.attached:
            return "No GPU box attached. Use `!gpu <user@ip>` first, then `!approve`."

        # Pin this experiment to the box it runs on, so later status/log/cancel
        # calls hit the right (ephemeral) host even after a new box is attached.
        cfg["compute"] = {
            "host": self.backend.host, "user": self.backend.user,
            "port": self.backend.port, "key": self.backend.key,
        }

        res = cfg.get("resources", {})
        spec = JobSpec(
            experiment_id=experiment_id,
            image=cfg["image"],
            command=cfg["command"],
            env=_launch_env(experiment_id, cfg.get("env", {})),
            resources=Resources(
                gpus=res.get("gpus", ""),
                memory=res.get("memory", ""),
                pids_limit=res.get("pids_limit", 0),
                time_limit_minutes=res.get("time_limit_minutes", 0),
            ),
        )
        workspace_local = str(self.workspace.path_for(experiment_id))
        try:
            handle = await self.backend.submit(spec, workspace_local)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Launch failed for experiment %s", experiment_id)
            await self.episodic.update_experiment(
                experiment_id, status="failed", notes=f"launch error: {exc}"
            )
            return f"Launch failed for experiment #{experiment_id}: {exc}"

        cfg["backend"] = handle.backend
        cfg["handle"] = handle.to_dict()
        cfg["started_at"] = datetime.now(timezone.utc).isoformat()
        await self.episodic.update_experiment(
            experiment_id, config=cfg, status="running"
        )
        await self.episodic.log_action(
            "experiment_launched", f"exp {experiment_id} on {handle.backend}",
            metadata={"experiment_id": experiment_id, "handle": handle.to_dict()},
        )
        return f"Experiment #{experiment_id} launched on `{handle.backend}` (container {handle.container_id[:12]})."

    async def get_status(self, experiment_id: int) -> str:
        exp = await self.episodic.get_experiment(experiment_id)
        if exp is None:
            return f"Experiment #{experiment_id} not found."
        return f"Experiment #{experiment_id} ({exp['title']}): {exp['status']}, metrics={exp.get('metrics')}"

    async def mlflow_summary(self, experiment_id: int) -> str:
        """Fetch the experiment's MLflow run (params + latest metrics) on demand."""
        from . import mlflow

        get = getattr(self.backend, "mlflow_get", None)
        post = getattr(self.backend, "mlflow_post", None)
        if not (settings.mlflow_enabled and get and post):
            return "MLflow tracking is not configured for this runner."

        by_name = await get(
            "experiments/get-by-name",
            {"experiment_name": settings.mlflow_experiment_name},
        )
        exp_id = (by_name or {}).get("experiment", {}).get("experiment_id")
        if not exp_id:
            return "No MLflow experiment recorded yet."
        search = await post(
            "runs/search",
            {
                "experiment_ids": [exp_id],
                "filter": f"attributes.run_name = '{mlflow.run_name(experiment_id)}'",
                "max_results": 50,
            },
        )
        runs = (search or {}).get("runs") or []
        if not runs:
            return f"No MLflow run found yet for experiment #{experiment_id}."
        metrics = mlflow.parse_metrics(runs[0])
        params = mlflow.parse_params(runs[0])
        lines = [f"MLflow run for experiment #{experiment_id} ({mlflow.run_name(experiment_id)}):"]
        if params:
            lines.append("params: " + ", ".join(f"{k}={v}" for k, v in params.items()))
        if metrics:
            lines.append("metrics: " + ", ".join(f"{k}={v}" for k, v in metrics.items()))
        return "\n".join(lines) if len(lines) > 1 else lines[0] + " (no metrics logged yet)"

    async def get_logs(self, experiment_id: int, tail: int = 200) -> str:
        handle = await self._handle(experiment_id)
        if handle is None:
            return f"Experiment #{experiment_id} has no running job."
        return await self.backend.logs(handle, tail=tail)

    async def cancel(self, experiment_id: int) -> str:
        handle = await self._handle(experiment_id)
        if handle is None:
            return f"Experiment #{experiment_id} has no running job to cancel."
        await self.backend.cancel(handle)
        await self.episodic.update_experiment(experiment_id, status="cancelled")
        return f"Experiment #{experiment_id} cancelled."

    async def _handle(self, experiment_id: int) -> JobHandle | None:
        cfg = await self._config(experiment_id)
        if not cfg:
            return None
        self._apply_target(cfg)  # talk to the box this experiment ran on
        return JobHandle.from_dict(cfg.get("handle"))

    # -- polling --

    async def poll_active(self) -> list[StateChange]:
        if not self.enabled:
            return []
        changes: list[StateChange] = []
        for exp in await self.episodic.list_active_experiments():
            change = await self._poll_one(exp)
            if change is not None:
                changes.append(change)
        return changes

    async def _poll_one(self, exp: dict) -> StateChange | None:
        exp_id = exp["id"]
        cfg = exp.get("config") or {}
        handle = JobHandle.from_dict(cfg.get("handle"))
        if handle is None:
            return None
        if not self._apply_target(cfg):
            return None  # box for this experiment isn't reachable/recorded

        status = await self.backend.status(handle)
        if status.state not in TERMINAL_STATES:
            return None

        dest = str(Path(self.artifacts_dir) / f"exp_{exp_id}")
        try:
            await self.backend.fetch_artifacts(handle, dest)
        except Exception:  # noqa: BLE001
            logger.exception("Artifact fetch failed for experiment %s", exp_id)
        metrics = read_latest_metrics(dest)
        artifacts = _list_artifacts(dest)

        await self.episodic.update_experiment(
            exp_id, status=status.state.value, metrics=metrics, artifacts=artifacts,
        )
        await self._record_to_project(exp, dest, status.state.value, metrics, cfg)
        msg = (
            f"Experiment #{exp_id} ({exp['title']}) {status.state.value}."
            + (f" metrics: {metrics}" if metrics else "")
        )
        return StateChange(
            experiment_id=exp_id, channel_id=exp.get("channel_id"),
            title=exp["title"], state=status.state.value, message=msg,
        )

    async def _record_to_project(self, exp, dest, state, metrics, cfg) -> None:
        """Assemble the experiment folder under the project and register it.

        Copies the source code (workspace) + fetched results into
        outputs/projects/<slug>/experiments/exp_<id>/ with a metadata.json
        (status, metrics, image, command, compute host, MLflow run).
        """
        if self.projects is None:
            return
        try:
            project = await self.projects.ensure(exp.get("channel_id"))
            if project.get("id") is None:
                return
            exp_id = exp["id"]
            base = self.projects.kind_dir(project["slug"], "experiments") / f"exp_{exp_id}"
            (base / "code").mkdir(parents=True, exist_ok=True)
            (base / "results").mkdir(parents=True, exist_ok=True)
            _copy_tree(self.workspace.path_for(exp_id), base / "code")
            _copy_tree(Path(dest), base / "results")
            meta = {
                "experiment_id": exp_id, "title": exp.get("title"), "status": state,
                "metrics": metrics, "image": cfg.get("image"),
                "command": cfg.get("command"), "compute": cfg.get("compute", {}).get("host"),
                "mlflow_run": f"exp_{exp_id}",
            }
            (base / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
            rel = f"projects/{project['slug']}/experiments/exp_{exp_id}"
            await self.projects.add_artifact(
                project["id"], "experiments", exp.get("title") or f"exp_{exp_id}",
                rel, {"status": state, "metrics": metrics},
            )
        except Exception:  # noqa: BLE001 — recording must not break the poller
            logger.exception("Failed to record experiment %s to project", exp.get("id"))


def _list_artifacts(dest: str) -> list[str]:
    root = Path(dest)
    if not root.exists():
        return []
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def _copy_tree(src, dst) -> None:
    """Copy a directory's files into dst (best-effort; skips if src missing)."""
    import shutil

    src, dst = Path(src), Path(dst)
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        if p.is_file():
            target = dst / p.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
