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
    JobStatus,
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
    def __init__(self, episodic, backend, workspace, artifacts_dir: str,
                 projects=None, memory=None, coder=None):
        self.episodic = episodic
        self.backend = backend
        self.workspace = workspace
        self.artifacts_dir = artifacts_dir
        self.projects = projects
        self.memory = memory  # MemoryManager: lessons (record/recall) loop
        # Optional ExperimentCoder used for auto-retry code patching.  When
        # None and auto_retry is enabled, build_default_coder() is tried lazily.
        self._coder = coder

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
        cfg = exp.get("config") or {}
        retry_count = int(cfg.get("auto_retry_count") or 0)
        retry_info = f", auto_retries={retry_count}" if retry_count > 0 else ""
        return (
            f"Experiment #{experiment_id} ({exp['title']}): "
            f"{exp['status']}, metrics={exp.get('metrics')}{retry_info}"
        )

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
        timed_out = False
        if status.state not in TERMINAL_STATES:
            overrun = self._time_limit_overrun(cfg)
            if overrun is None:
                return None  # still running, within its wall-clock budget
            # Wall-clock limit exceeded: stop the container and fail the run, so a
            # runaway job can't hold the GPU forever. This is the enforcement the
            # `time_limit_minutes` knob promises (poller-enforced).
            logger.info(
                "Experiment %s exceeded its %d-minute time limit; cancelling.",
                exp_id, overrun,
            )
            try:
                await self.backend.cancel(handle)
            except Exception as exc:  # noqa: BLE001 — keep it running so a later poll retries
                # If we can't stop it, the container may still be alive and holding
                # the GPU. Do NOT mark it terminal (it would drop out of the active
                # filter and never be retried) — leave it running with a note so the
                # next poll attempts the cancel again.
                logger.exception("Failed to stop timed-out experiment %s", exp_id)
                await self.episodic.update_experiment(
                    exp_id,
                    notes=(
                        f"time limit ({overrun}m) exceeded but cancellation failed "
                        f"({exc}); will retry on the next poll"
                    ),
                )
                return None
            status = JobStatus(JobState.FAILED, detail=f"time limit exceeded ({overrun}m)")
            timed_out = True

        dest = str(Path(self.artifacts_dir) / f"exp_{exp_id}")
        try:
            await self.backend.fetch_artifacts(handle, dest)
        except Exception:  # noqa: BLE001
            logger.exception("Artifact fetch failed for experiment %s", exp_id)
        metrics = read_latest_metrics(dest)
        artifacts = _list_artifacts(dest)
        state = status.state.value

        fields = {"status": state, "metrics": metrics, "artifacts": artifacts}
        if timed_out:
            fields["notes"] = (
                f"timed out: exceeded the {self._time_limit(cfg)}-minute wall-clock limit"
            )
        await self.episodic.update_experiment(exp_id, **fields)
        await self._record_to_project(exp, dest, state, metrics, cfg)

        # Auto-retry: if the run failed (not cancelled, not timed-out) and the
        # retry budget allows it, patch the code from logs and relaunch without
        # re-approval.  The retry gate is checked inside _maybe_auto_retry so
        # this call is always safe to make.
        if state == JobState.FAILED.value:
            retry_change = await self._maybe_auto_retry(exp, cfg, handle, state, timed_out)
            if retry_change is not None:
                # A retry was launched: report it and stop — do NOT run
                # _consolidate_outcome (the run is not definitively done yet).
                return retry_change

        reason = (
            f"exceeded its {self._time_limit(cfg)}-minute wall-clock limit"
            if timed_out else ""
        )
        await self._consolidate_outcome(exp, handle, state, metrics, reason=reason)

        # Include retry count in the success message when retries were needed.
        retry_count = int(cfg.get("auto_retry_count") or 0)
        retry_suffix = (
            f" (succeeded after {retry_count} auto-retr{'y' if retry_count == 1 else 'ies'})"
            if retry_count > 0 and state == JobState.SUCCEEDED.value
            else ""
        )
        msg = (
            f"Experiment #{exp_id} ({exp['title']}) {state}"
            + (" (timed out)" if timed_out else "")
            + (f" metrics: {metrics}" if metrics else "")
            + retry_suffix
        )
        return StateChange(
            experiment_id=exp_id, channel_id=exp.get("channel_id"),
            title=exp["title"], state=state, message=msg,
        )

    @staticmethod
    def _time_limit(cfg: dict) -> int:
        """Configured wall-clock limit in minutes (0 = unlimited)."""
        try:
            return int((cfg.get("resources") or {}).get("time_limit_minutes") or 0)
        except (TypeError, ValueError):
            return 0

    def _time_limit_overrun(self, cfg: dict) -> int | None:
        """The limit (minutes) if a running job has outrun it, else None.

        Reads the `started_at` stamped at launch; tolerant of missing/garbled
        values (returns None = "let it keep running") so a bad timestamp can
        never wrongly kill a job.
        """
        limit = self._time_limit(cfg)
        started = cfg.get("started_at")
        if limit <= 0 or not started:
            return None
        try:
            started_dt = datetime.fromisoformat(started)
        except (TypeError, ValueError):
            return None
        if started_dt.tzinfo is None:
            started_dt = started_dt.replace(tzinfo=timezone.utc)
        elapsed_min = (datetime.now(timezone.utc) - started_dt).total_seconds() / 60
        return limit if elapsed_min > limit else None

    def _get_coder(self):
        """Return the configured coder, or try to build one from settings."""
        if self._coder is not None:
            return self._coder
        # Lazy construction so the runner doesn't depend on LLM credentials
        # being present at construction time.
        try:
            from .coder import build_default_coder
            return build_default_coder()
        except Exception:  # noqa: BLE001
            logger.exception("Could not build default coder for auto-retry")
            return None

    async def _maybe_auto_retry(
        self, exp: dict, cfg: dict, handle, state: str, timed_out: bool
    ) -> "StateChange | None":
        """Attempt a bounded automatic fix-and-relaunch after a failed run.

        Returns a StateChange (meaning: a retry was launched; the poller
        should report THIS message and stop) or None (meaning: fall through
        to normal failure handling).

        Safety invariants enforced here:
        - Never retries a CANCELLED run (only failed, non-timeout).
        - Never modifies the JobSpec: the same image/command/resources are
          reused — only the workspace code files change.
        - Hard cap from settings.experiment_auto_retry; counter persisted in
          config JSONB so the cap survives restarts.
        - Every retry is logged via episodic.log_action for dashboard visibility.
        - Any exception inside this path is caught; the caller falls through to
          the normal failure path unchanged.
        """
        exp_id = exp["id"]

        # Gate 1: feature must be enabled globally.
        if settings.experiment_auto_retry <= 0:
            return None
        # Gate 2: only failed runs (not cancelled, not timed-out).
        if state != JobState.FAILED.value or timed_out:
            return None
        # Gate 3: retry counter must be below the cap.
        auto_retry_count = int(cfg.get("auto_retry_count") or 0)
        if auto_retry_count >= settings.experiment_auto_retry:
            return None
        # Gate 4: a coder must be available.
        coder = self._get_coder()
        if coder is None:
            return None

        try:
            attempt = auto_retry_count + 1
            cap = settings.experiment_auto_retry

            # 1. Fetch failure logs (best-effort; empty string is tolerated by revise()).
            logs = ""
            try:
                logs = await self.backend.logs(handle, tail=200)
            except Exception:  # noqa: BLE001
                logger.exception("Auto-retry: could not fetch logs for exp %s", exp_id)

            # 2. Recall failure lessons from memory (best-effort).
            lessons = ""
            if self.memory is not None:
                try:
                    lessons = await self.memory.recall_lessons(
                        exp.get("title") or f"experiment #{exp_id}", kind="experiment"
                    ) or ""
                except Exception:  # noqa: BLE001
                    logger.exception("Auto-retry: lesson recall failed for exp %s", exp_id)

            # 3. Read current workspace files.
            current_files: dict[str, str] = {}
            ws_root = self.workspace.path_for(exp_id)
            for rel in self.workspace.list_files(exp_id):
                try:
                    current_files[rel] = (ws_root / rel).read_text(errors="replace")
                except Exception:  # noqa: BLE001
                    pass

            # 4. Ask the coder to produce a patched file set.
            spec_str = (cfg.get("plan") or exp.get("title") or f"experiment #{exp_id}")
            fixed_files = await coder.revise(spec_str, current_files, logs, lessons)

            # 5. Write the patched files into the workspace.
            await self.write_code(exp_id, fixed_files)

            # 6. Increment counter and relaunch with the SAME JobSpec (immutability
            #    guaranteed: we rebuild the spec from cfg which has not changed).
            cfg["auto_retry_count"] = attempt
            await self.episodic.update_experiment(exp_id, config=cfg, status="running")

            res = cfg.get("resources", {})
            spec = JobSpec(
                experiment_id=exp_id,
                image=cfg["image"],
                command=cfg["command"],
                env=_launch_env(exp_id, cfg.get("env", {})),
                resources=Resources(
                    gpus=res.get("gpus", ""),
                    memory=res.get("memory", ""),
                    pids_limit=res.get("pids_limit", 0),
                    time_limit_minutes=res.get("time_limit_minutes", 0),
                ),
            )
            workspace_local = str(self.workspace.path_for(exp_id))
            new_handle = await self.backend.submit(spec, workspace_local)

            cfg["handle"] = new_handle.to_dict()
            cfg["started_at"] = datetime.now(timezone.utc).isoformat()
            await self.episodic.update_experiment(exp_id, config=cfg, status="running")
            await self.episodic.log_action(
                "experiment_auto_retry",
                f"exp {exp_id} retry {attempt}/{cap} — patched from logs, relaunched",
                metadata={
                    "experiment_id": exp_id,
                    "attempt": attempt,
                    "cap": cap,
                    "handle": new_handle.to_dict(),
                },
            )
            msg = (
                f"Experiment #{exp_id} ({exp['title']}) failed → "
                f"auto-retry {attempt}/{cap} launched (patched from logs)"
            )
            logger.info(msg)
            return StateChange(
                experiment_id=exp_id,
                channel_id=exp.get("channel_id"),
                title=exp["title"],
                state="auto_retry",
                message=msg,
            )

        except Exception:  # noqa: BLE001
            # Any error in the retry path must not swallow the real failure.
            logger.exception(
                "Auto-retry attempt failed for experiment %s; falling through to "
                "normal failure handling",
                exp_id,
            )
            return None

    async def _consolidate_outcome(self, exp, handle, state, metrics, reason: str = "") -> None:
        """Turn an experiment outcome into a durable, reusable lesson.

        On failure: fetch the container logs and have an LLM extract the root
        cause + how to avoid it. On success: record what worked. When `reason` is
        given (e.g. a timeout), it's used directly instead of fetching logs — the
        container is already gone, so there are none to read. Stored as a semantic
        lesson (recall_lessons surfaces it for future experiments).
        """
        if self.memory is None:
            return
        title = exp.get("title") or f"experiment #{exp['id']}"
        channel_id = exp.get("channel_id")
        try:
            await self.memory.log_experience(
                "experiment_outcome", f"{title}: {state} (metrics={metrics})",
                channel_id, {"experiment_id": exp["id"], "state": state},
            )
            if state == "failed" and reason:
                lesson = (
                    f"[experiment timeout] '{title}' {reason}. Next time shrink the "
                    "search space / n_trials / epochs, or raise the time limit."
                )
            elif state == "failed":
                logs = ""
                try:
                    logs = await self.backend.logs(handle, tail=120)
                except Exception:  # noqa: BLE001
                    pass
                lesson = await self._extract_failure_lesson(title, logs)
            elif state == "succeeded":
                lesson = (
                    f"Experiment '{title}' succeeded with metrics {metrics}. "
                    "This approach/configuration worked — reuse it as a baseline."
                )
            else:
                return
            if lesson:
                await self.memory.record_lesson(
                    lesson, kind="experiment", channel_id=channel_id, status=state,
                )
        except Exception:  # noqa: BLE001 — lessons must not break the poller
            logger.exception("Outcome consolidation failed for experiment %s", exp.get("id"))

    async def _extract_failure_lesson(self, title: str, logs: str) -> str:
        """LLM-extract a concise, reusable lesson from a failed run's logs."""
        if not logs.strip():
            return f"Experiment '{title}' failed but produced no logs — add error logging."
        from ..llm import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            resp = await get_llm().ainvoke([
                SystemMessage(content=(
                    "An ML experiment failed. From the logs, write ONE concise lesson "
                    "(1-3 sentences) capturing the root cause and how to AVOID it next "
                    "time. Be specific and actionable; start with the failure mode."
                )),
                HumanMessage(content=f"Experiment: {title}\n\nLogs (tail):\n{logs[-4000:]}"),
            ])
            text = resp.content
            if isinstance(text, list):
                text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)
            return f"[experiment failure] {title}: {text.strip()}"
        except Exception:  # noqa: BLE001
            logger.exception("Failure-lesson extraction failed")
            return ""

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
