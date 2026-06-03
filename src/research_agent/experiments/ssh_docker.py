"""SSH + Docker compute backend.

Runs each experiment as a detached Docker container on a remote GPU host reached
over SSH. The workspace is rsync'd up and mounted read-only; outputs land in a
mounted directory that's rsync'd back as artifacts. Secrets go through a remote
env-file (mounted via --env-file) so they never appear in the container's
process arguments.

The command-construction and status-parsing logic is pure and unit-tested; the
methods here wrap it with async subprocess execution of ssh/rsync.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
from pathlib import PurePosixPath
from urllib.parse import urlencode

from ..config import settings
from .image import build_experiment_dockerfile
from .mlflow import api_url, build_mlflow_server_command
from .types import Artifact, JobHandle, JobSpec, JobState, JobStatus

logger = logging.getLogger(__name__)

BACKEND_NAME = "ssh-docker"

# Provisions a bare Ubuntu GPU box: Docker + NVIDIA Container Toolkit, then a
# GPU-in-container smoke test. Idempotent; assumes the NVIDIA *driver* is present
# (cloud GPU images ship it) and the SSH user has sudo/root.
_PROVISION_SCRIPT = r"""
set -euo pipefail
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
export DEBIAN_FRONTEND=noninteractive

if ! command -v docker >/dev/null 2>&1; then
  echo "[provision] installing docker..."
  curl -fsSL https://get.docker.com | $SUDO sh
fi

if ! docker info 2>/dev/null | grep -qi 'Runtimes.*nvidia' && ! command -v nvidia-ctk >/dev/null 2>&1; then
  echo "[provision] installing NVIDIA Container Toolkit..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | $SUDO gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  $SUDO apt-get update -y
  $SUDO apt-get install -y nvidia-container-toolkit
  $SUDO nvidia-ctk runtime configure --runtime=docker
  $SUDO systemctl restart docker
fi

echo "[provision] versions:"
docker --version
echo "[provision] GPU-in-container smoke test:"
$SUDO docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi \
  --query-gpu=name,memory.total --format=csv,noheader || echo "[provision] WARNING: GPU smoke test failed"
echo "[provision] done"
"""


# --- pure helpers (unit-tested) ------------------------------------------------

def build_docker_run_command(
    *, name: str, image: str, command: list[str],
    workspace_remote: str, output_remote: str,
    gpus: str = "", memory: str = "", pids_limit: int = 0,
    env_file_remote: str | None = None,
    network: str = "", volumes: list[str] | None = None,
) -> list[str]:
    """Build the argv for a detached `docker run`. Pure: no I/O.

    `network` joins a shared docker network (so the job can reach the MLflow
    server by name); `volumes` are extra `-v` mounts (e.g. a persistent HF cache).
    """
    args = ["docker", "run", "-d", "--name", name]
    if network:
        args += ["--network", network]
    if gpus:
        args += ["--gpus", gpus]
    if memory:
        args += ["--memory", memory]
    if pids_limit:
        args += ["--pids-limit", str(pids_limit)]
    if env_file_remote:
        args += ["--env-file", env_file_remote]
    for vol in volumes or []:
        args += ["-v", vol]
    args += [
        "-v", f"{workspace_remote}:/workspace:ro",
        "-v", f"{output_remote}:/output",
        "-w", "/workspace",
        image,
        *command,
    ]
    return args


def parse_inspect_status(raw: str) -> JobStatus:
    """Parse `docker inspect -f '{{.State.Status}};{{.State.ExitCode}}'` output."""
    text = raw.strip()
    if not text:
        # No such container: treat as gone (cancelled/removed).
        return JobStatus(JobState.UNKNOWN, detail="container not found")
    status, _, code = text.partition(";")
    status = status.strip().lower()
    try:
        exit_code = int(code.strip()) if code.strip() != "" else None
    except ValueError:
        exit_code = None

    if status in {"created", "running", "restarting", "paused"}:
        return JobStatus(JobState.RUNNING, exit_code=exit_code, detail=status)
    if status in {"exited", "dead"}:
        if exit_code == 0:
            return JobStatus(JobState.SUCCEEDED, exit_code=0, detail=status)
        return JobStatus(JobState.FAILED, exit_code=exit_code, detail=status)
    return JobStatus(JobState.UNKNOWN, exit_code=exit_code, detail=status)


# --- backend -------------------------------------------------------------------

class SSHDockerBackend:
    name = BACKEND_NAME

    def __init__(self):
        self.host = settings.compute_ssh_host
        self.user = settings.compute_ssh_user
        self.port = settings.compute_ssh_port
        self.key = settings.compute_ssh_key
        self.workdir = settings.compute_workdir

    # -- target management (the GPU box is ephemeral: a fresh IP per experiment) --

    def set_target(
        self, host: str, user: str | None = None, port: int | None = None,
        key: str | None = None,
    ) -> None:
        """Point the backend at a (new) compute box. host may be 'user@ip'."""
        if "@" in host and user is None:
            user, host = host.split("@", 1)
        self.host = host.strip()
        if user:
            self.user = user.strip()
        if port:
            self.port = port
        if key is not None:
            self.key = key

    @property
    def configured(self) -> bool:
        return bool(self.host and self.user)

    # -- low-level exec --

    def _ssh_base(self) -> list[str]:
        opts = [
            "ssh", "-p", str(self.port),
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
        ]
        if self.key:
            opts += ["-i", self.key]
        opts.append(f"{self.user}@{self.host}")
        return opts

    async def _run(self, argv: list[str], stdin: str | None = None) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate(stdin.encode() if stdin is not None else None)
        return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")

    async def _ssh(self, remote_cmd: str, stdin: str | None = None) -> tuple[int, str, str]:
        return await self._run([*self._ssh_base(), remote_cmd], stdin=stdin)

    async def _ssh_checked(self, remote_cmd: str, stdin: str | None = None) -> str:
        code, out, err = await self._ssh(remote_cmd, stdin=stdin)
        if code != 0:
            raise RuntimeError(f"remote command failed ({code}): {err.strip() or out.strip()}")
        return out

    def _remote_paths(self, experiment_id: int) -> tuple[str, str, str]:
        base = str(PurePosixPath(self.workdir) / f"exp_{experiment_id}")
        return base, f"{base}/workspace", f"{base}/output"

    def _rsync_target(self, remote_path: str) -> str:
        ssh = f"ssh -p {self.port} -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
        if self.key:
            ssh += f" -i {shlex.quote(self.key)}"
        return ssh, f"{self.user}@{self.host}:{remote_path}"

    # -- provisioning a bare-Ubuntu GPU box --

    async def survey(self) -> str:
        """Return a short report of the box: OS, GPU, docker, nvidia runtime."""
        report = await self._ssh_checked(
            "set +e; "
            "echo OS: $(. /etc/os-release 2>/dev/null; echo $PRETTY_NAME); "
            "echo GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | paste -sd '; ' - || echo none); "
            "echo DOCKER: $(docker --version 2>/dev/null || echo none); "
            "echo NVIDIA_RUNTIME: $(docker info 2>/dev/null | grep -i 'Runtimes' | grep -qi nvidia && echo yes || echo no)"
        )
        return report.strip()

    async def provision(self) -> str:
        """Install Docker + the NVIDIA Container Toolkit on a fresh Ubuntu box.

        Idempotent: skips anything already present. Requires sudo (or root).
        Returns a report ending with a GPU-in-container smoke check, then builds
        the universal experiment image so the first run launches fast.
        """
        out = await self._ssh_checked("bash -s", stdin=_PROVISION_SCRIPT)
        image_note = await self.ensure_image()
        return out.strip() + ("\n" + image_note if image_note else "")

    async def ensure_image(self) -> str:
        """Ensure the universal experiment image exists on the box (build or pull).

        Idempotent: present -> no-op; a registry ref -> `docker pull`; otherwise
        build from the embedded Dockerfile over stdin (no build context needed).
        """
        img = settings.experiment_image
        _, present, _ = await self._ssh(
            f"docker image inspect {shlex.quote(img)} >/dev/null 2>&1 "
            f"&& echo yes || echo no"
        )
        if present.strip() == "yes":
            return ""
        # A registry-qualified ref (has a host with a dot or port): try to pull.
        head = img.split("/", 1)[0]
        if "/" in img and ("." in head or ":" in head):
            code, _, _ = await self._ssh(f"docker pull {shlex.quote(img)}")
            if code == 0:
                return f"Pulled experiment image {img}."
        dockerfile = build_experiment_dockerfile(settings.compute_base_image)
        await self._ssh_checked(f"docker build -t {shlex.quote(img)} -", stdin=dockerfile)
        return f"Built experiment image {img}."

    # -- MLflow infra (shared network + tracking server on the GPU box) --

    async def ensure_infra(self) -> None:
        """Create the shared docker network and start the MLflow server (idempotent)."""
        if not settings.mlflow_enabled:
            return
        net = settings.compute_network
        await self._ssh(f"docker network create {shlex.quote(net)} >/dev/null 2>&1; true")
        name = settings.mlflow_container
        _, running, _ = await self._ssh(
            f"docker inspect -f '{{{{.State.Running}}}}' {shlex.quote(name)} 2>/dev/null"
        )
        if running.strip() == "true":
            return
        await self._ssh(f"docker rm -f {shlex.quote(name)} >/dev/null 2>&1; true")
        argv = build_mlflow_server_command(
            container=name, image=settings.mlflow_image, network=net,
            port=settings.mlflow_port, volume=settings.mlflow_volume,
        )
        await self._ssh_checked(" ".join(shlex.quote(a) for a in argv))
        logger.info("Started MLflow server %s on the compute node.", name)

    async def ensure_ready(self) -> None:
        """Make sure both the MLflow infra and the experiment image are present."""
        await self.ensure_infra()
        await self.ensure_image()

    async def mlflow_get(self, path: str, params: dict | None = None) -> dict | None:
        """GET an MLflow REST endpoint on the remote (tunnelled over SSH)."""
        url = api_url(settings.mlflow_port, path)
        if params:
            url += "?" + urlencode(params)
        code, out, _ = await self._ssh(f"curl -sf --max-time 15 {shlex.quote(url)}")
        if code != 0 or not out.strip():
            return None
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return None

    async def mlflow_post(self, path: str, body: dict) -> dict | None:
        """POST a JSON body to an MLflow REST endpoint on the remote, over SSH."""
        url = api_url(settings.mlflow_port, path)
        cmd = (
            f"curl -sf --max-time 15 -H 'Content-Type: application/json' "
            f"-X POST {shlex.quote(url)} -d @-"
        )
        code, out, _ = await self._ssh(cmd, stdin=json.dumps(body))
        if code != 0 or not out.strip():
            return None
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return None

    # -- ComputeBackend --

    async def submit(self, spec: JobSpec, workspace_local: str) -> JobHandle:
        base, ws_remote, out_remote = self._remote_paths(spec.experiment_id)
        container = f"ra_exp_{spec.experiment_id}"

        # Ensure the MLflow server, shared network, and experiment image exist.
        await self.ensure_ready()

        # Remote dirs.
        await self._ssh_checked(
            f"mkdir -p {shlex.quote(ws_remote)} {shlex.quote(out_remote)}"
        )

        # Upload workspace.
        ssh_cmd, target = self._rsync_target(ws_remote + "/")
        local = workspace_local.rstrip("/") + "/"
        code, out, err = await self._run(
            ["rsync", "-az", "--delete", "-e", ssh_cmd, local, target]
        )
        if code != 0:
            raise RuntimeError(f"workspace rsync failed ({code}): {err.strip()}")

        # Secrets via a remote env-file (not in process args). 0600 perms.
        env_file_remote = None
        if spec.env:
            env_file_remote = f"{base}/.env.job"
            body = "".join(f"{k}={v}\n" for k, v in spec.env.items())
            await self._ssh_checked(
                f"umask 077; cat > {shlex.quote(env_file_remote)}", stdin=body
            )

        # Join the MLflow network and mount the persistent HF cache (datasets +
        # model weights downloaded once and reused across runs/trials).
        network = settings.compute_network if settings.mlflow_enabled else ""
        volumes = []
        if settings.compute_hf_cache_volume:
            volumes.append(
                f"{settings.compute_hf_cache_volume}:/root/.cache/huggingface"
            )

        # Remove any stale container of the same name, then launch.
        await self._ssh(f"docker rm -f {container} >/dev/null 2>&1; true")
        argv = build_docker_run_command(
            name=container,
            image=spec.image,
            command=spec.command,
            workspace_remote=ws_remote,
            output_remote=out_remote,
            gpus=spec.resources.gpus,
            memory=spec.resources.memory,
            pids_limit=spec.resources.pids_limit,
            env_file_remote=env_file_remote,
            network=network,
            volumes=volumes,
        )
        out = await self._ssh_checked(" ".join(shlex.quote(a) for a in argv))
        container_id = out.strip().splitlines()[-1] if out.strip() else container
        return JobHandle(
            backend=self.name,
            container_id=container_id,
            raw={"container_name": container, "output_remote": out_remote},
        )

    async def status(self, handle: JobHandle) -> JobStatus:
        ref = shlex.quote(handle.container_id or handle.raw.get("container_name", ""))
        code, out, _ = await self._ssh(
            f"docker inspect -f '{{{{.State.Status}}}};{{{{.State.ExitCode}}}}' {ref} 2>/dev/null"
        )
        if code != 0:
            return JobStatus(JobState.UNKNOWN, detail="container not found")
        return parse_inspect_status(out)

    async def logs(self, handle: JobHandle, tail: int = 200) -> str:
        ref = shlex.quote(handle.container_id or handle.raw.get("container_name", ""))
        _, out, err = await self._ssh(f"docker logs --tail {int(tail)} {ref} 2>&1")
        return out or err

    async def cancel(self, handle: JobHandle) -> None:
        ref = shlex.quote(handle.container_id or handle.raw.get("container_name", ""))
        await self._ssh(f"docker rm -f {ref}")

    async def fetch_artifacts(self, handle: JobHandle, dest_local: str) -> list[Artifact]:
        out_remote = handle.raw.get("output_remote")
        if not out_remote:
            return []
        from pathlib import Path

        Path(dest_local).mkdir(parents=True, exist_ok=True)
        ssh_cmd, source = self._rsync_target(out_remote + "/")
        code, _, err = await self._run(
            ["rsync", "-az", "-e", ssh_cmd, source, dest_local.rstrip("/") + "/"]
        )
        if code != 0:
            logger.warning("artifact rsync failed (%d): %s", code, err.strip())
            return []
        artifacts: list[Artifact] = []
        for p in sorted(Path(dest_local).rglob("*")):
            if p.is_file():
                artifacts.append(Artifact(path=str(p), size=p.stat().st_size))
        return artifacts
