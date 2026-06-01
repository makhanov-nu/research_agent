"""The ComputeBackend interface.

A backend knows how to submit a job, check its status, stream logs, cancel it,
and retrieve artifacts. The runner and registry are written against this
interface so SSHDockerBackend (v1) can later be joined by a worker-API or
HuggingFace-Jobs backend without changing callers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Artifact, JobHandle, JobSpec, JobStatus


@runtime_checkable
class ComputeBackend(Protocol):
    name: str

    async def submit(self, spec: JobSpec, workspace_local: str) -> JobHandle:
        """Ship the workspace and start the job; return a handle to track it."""
        ...

    async def status(self, handle: JobHandle) -> JobStatus:
        ...

    async def logs(self, handle: JobHandle, tail: int = 200) -> str:
        ...

    async def cancel(self, handle: JobHandle) -> None:
        ...

    async def fetch_artifacts(self, handle: JobHandle, dest_local: str) -> list[Artifact]:
        ...
