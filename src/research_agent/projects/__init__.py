"""Project-scoped persistence: each chat (channel) is a project.

A project owns a folder tree (`outputs/projects/<slug>/{lit_review, council,
methodology, experiments, paper}/`) plus a DB record, so every artifact is both
saved on disk and registered for the web frontend to browse.
"""

from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig

from .store import KINDS, ProjectStore, slugify

__all__ = [
    "KINDS", "ProjectStore", "slugify", "channel_of", "resolve_project",
    "save_council_proposal",
]


def channel_of(config: RunnableConfig | None) -> str | None:
    if not config:
        return None
    return (config.get("configurable") or {}).get("thread_id")


async def resolve_project(projects: ProjectStore | None, config) -> dict | None:
    """Resolve (creating if needed) the project for the current channel."""
    if projects is None:
        return None
    return await projects.ensure(channel_of(config))


async def save_council_proposal(
    projects: ProjectStore | None, project: dict | None, topic: str, proposal: str
) -> str:
    """Save the consortium's validated proposal into the project council folder.

    Returns the path relative to the outputs dir (for `!getfile`), or "".
    """
    if projects is None or project is None:
        return ""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = f"{slugify(topic)}-{stamp}.md"
    directory = projects.kind_dir(project["slug"], "council")
    path = directory / name
    path.write_text(f"# Council proposal — {topic}\n\n{proposal}\n")
    rel = f"projects/{project['slug']}/council/{name}"
    if project.get("id") is not None:
        await projects.add_artifact(project["id"], "council", topic, rel, {})
    return rel
