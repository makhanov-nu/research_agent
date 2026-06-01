"""Value types for the experiment runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class JobState(str, Enum):
    QUEUED = "queued"
    BUILDING = "building"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


TERMINAL_STATES = frozenset(
    {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}
)


@dataclass
class Resources:
    """Container resource limits for a run."""

    gpus: str = "all"            # docker --gpus value; "" disables GPU access
    memory: str = ""            # e.g. "8g"; "" = no limit
    pids_limit: int = 0          # 0 = no limit
    time_limit_minutes: int = 0  # 0 = no wall-clock limit (poller-enforced)


@dataclass
class JobSpec:
    """A fully-specified run, ready to dispatch to a backend."""

    experiment_id: int
    image: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    resources: Resources = field(default_factory=Resources)


@dataclass
class JobHandle:
    """Backend-specific reference to a submitted job, persisted in the registry."""

    backend: str
    container_id: str = ""
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"backend": self.backend, "container_id": self.container_id, "raw": self.raw}

    @classmethod
    def from_dict(cls, d: dict | None) -> "JobHandle | None":
        if not d:
            return None
        return cls(
            backend=d.get("backend", ""),
            container_id=d.get("container_id", ""),
            raw=d.get("raw", {}) or {},
        )


@dataclass
class JobStatus:
    state: JobState
    exit_code: int | None = None
    detail: str = ""


@dataclass
class Artifact:
    path: str
    size: int = 0
