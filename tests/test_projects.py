"""Tests for project-scoped persistence (folder-only mode, no DB)."""

from __future__ import annotations

import pytest

from research_agent.projects import save_council_proposal
from research_agent.projects.store import KINDS, ProjectStore, slugify


def test_slugify():
    assert slugify("My Cool Project!") == "my-cool-project"
    assert slugify("") == "project"


@pytest.mark.asyncio
async def test_ensure_creates_folder_tree(tmp_path):
    store = ProjectStore(pool=None, output_dir=str(tmp_path))
    proj = await store.ensure("chan-1", name="Speculative Decoding")
    assert proj["id"] is None  # folder-only without a DB
    assert proj["slug"] == "speculative-decoding"
    for kind in KINDS:
        assert (tmp_path / "projects" / "speculative-decoding" / kind).is_dir()


@pytest.mark.asyncio
async def test_kind_dir_is_under_project(tmp_path):
    store = ProjectStore(pool=None, output_dir=str(tmp_path))
    proj = await store.ensure("c", name="proj")
    d = store.kind_dir(proj["slug"], "lit_review")
    assert d == tmp_path / "projects" / "proj" / "lit_review"
    assert d.is_dir()


@pytest.mark.asyncio
async def test_save_council_proposal_writes_file(tmp_path):
    store = ProjectStore(pool=None, output_dir=str(tmp_path))
    proj = await store.ensure("c", name="proj")
    rel = await save_council_proposal(store, proj, "my topic", "## the idea")
    assert rel.startswith("projects/proj/council/")
    saved = tmp_path / rel
    assert saved.exists()
    text = saved.read_text()
    assert "# Council proposal — my topic" in text and "## the idea" in text


@pytest.mark.asyncio
async def test_add_artifact_noop_without_db(tmp_path):
    store = ProjectStore(pool=None, output_dir=str(tmp_path))
    # No DB -> registry calls are no-ops (return None), never raise.
    assert await store.add_artifact(None, "lit_review", "t", "p") is None
    assert await store.list_artifacts(1) == []
    assert await store.list_projects() == []
