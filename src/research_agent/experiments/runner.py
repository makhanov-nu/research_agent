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
    def __init__(self, episodic, backend, workspace, artifacts_dir: str):
        self.episodic = episodic
        self.backend = backend
        self.workspace = workspace
        self.artifacts_dir = artifacts_dir

    @property
    def enabled(self) -> bool:
        return self.backend is not None and getattr(self.episodic, "enabled", False)

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
        if not self.workspace.list_files(experiment_id):
            return f"Experiment #{experiment_id} has no code yet — write code first."

        cfg.update(
            {
                "image": image or settings.compute_base_image,
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

        res = cfg.get("resources", {})
        spec = JobSpec(
            experiment_id=experiment_id,
            image=cfg["image"],
            command=cfg["command"],
            env=cfg.get("env", {}),
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
        msg = (
            f"Experiment #{exp_id} ({exp['title']}) {status.state.value}."
            + (f" metrics: {metrics}" if metrics else "")
        )
        return StateChange(
            experiment_id=exp_id, channel_id=exp.get("channel_id"),
            title=exp["title"], state=status.state.value, message=msg,
        )


def _list_artifacts(dest: str) -> list[str]:
    root = Path(dest)
    if not root.exists():
        return []
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
