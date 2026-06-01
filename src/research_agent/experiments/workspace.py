"""Local per-experiment workspaces where the agent authors code before dispatch."""

from __future__ import annotations

import os
from pathlib import Path


class Workspace:
    def __init__(self, base_dir: str):
        self.base = Path(base_dir)

    def path_for(self, experiment_id: int) -> Path:
        return self.base / f"exp_{experiment_id}"

    def write_files(self, experiment_id: int, files: dict[str, str]) -> list[str]:
        """Write relative path -> content under the experiment workspace.

        Rejects any path that escapes the workspace root (absolute paths or
        `..` traversal), so agent-authored filenames can't write outside it.
        """
        root = self.path_for(experiment_id)
        root.mkdir(parents=True, exist_ok=True)
        root_resolved = root.resolve()

        written: list[str] = []
        for rel, content in files.items():
            target = (root / rel).resolve()
            if target != root_resolved and not str(target).startswith(
                str(root_resolved) + os.sep
            ):
                raise ValueError(f"Unsafe workspace path: {rel!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            written.append(rel)
        return sorted(written)

    def list_files(self, experiment_id: int) -> list[str]:
        root = self.path_for(experiment_id)
        if not root.exists():
            return []
        return sorted(
            str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
        )
