"""Tests for the consortium's shared-transcript helpers and config."""

from __future__ import annotations

from research_agent.config import settings
from research_agent.consortium.consortium import build_document, render_transcript


def test_render_transcript_empty():
    assert render_transcript([]) == "(no discussion yet)"


def test_render_transcript_labels_speakers():
    out = render_transcript([("openai/gpt-5.5", "idea A"), ("deepseek/r1", "counter")])
    assert "[openai/gpt-5.5]:" in out
    assert "idea A" in out
    assert "[deepseek/r1]:" in out
    # speakers appear in order
    assert out.index("gpt-5.5") < out.index("deepseek/r1")


def test_build_document_contains_final_and_transcript():
    transcript = [("Background (literature)", "gaps..."), ("modelX", "proposal")]
    doc = build_document("my topic", "## 3 ideas", transcript)
    assert "# Research ideas — my topic" in doc
    assert "## 3 ideas" in doc
    assert "Full panel discussion" in doc
    assert "[modelX]:" in doc


def test_panel_models_parsing(monkeypatch):
    monkeypatch.setattr(settings, "consortium_models", "a/x , b/y,, c/z ")
    assert settings.panel_models == ["a/x", "b/y", "c/z"]


# --- session flow (fakes, no LLM) --------------------------------------------

import pytest

from research_agent.consortium.consortium import Consortium


class _FakeConsortium(Consortium):
    """Records calls and returns canned text instead of hitting any model."""

    def __init__(self, tmp_path):
        super().__init__([], ["m/a", "m/b"], "chair/x", str(tmp_path))
        self.said: list[tuple[str, str]] = []

    async def _agent_say(self, model, instruction, transcript):
        self.said.append((model, instruction))
        return f"{model} says something"


@pytest.mark.asyncio
async def test_session_round_collects_each_panelist(tmp_path):
    c = _FakeConsortium(tmp_path)
    session = c.new_session("topic")
    digest = await session.run_round()
    assert [m for m, _ in c.said] == ["m/a", "m/b"]  # each spoke once, in order
    assert session.round_no == 1
    assert "m/a says something" in digest and "m/b says something" in digest


@pytest.mark.asyncio
async def test_session_feedback_enters_transcript_and_next_round(tmp_path):
    c = _FakeConsortium(tmp_path)
    session = c.new_session("topic")
    await session.run_round()
    await session.run_round(feedback="focus on efficiency")
    speakers = [s for s, _ in session.transcript]
    assert "Researcher (you)" in speakers
    assert session.round_no == 2


@pytest.mark.asyncio
async def test_capture_council_records_experience_and_lesson():
    from research_agent.consortium import capture_council

    class _Mem:
        def __init__(self):
            self.exp = []
            self.lessons = []

        async def log_experience(self, kind, summary, channel_id=None, metadata=None):
            self.exp.append((kind, channel_id))

        async def record_lesson(self, text, *, kind, channel_id=None, **kw):
            self.lessons.append((kind, text))

    mem = _Mem()
    await capture_council(mem, "chan-1", "specdec", "the validated idea", "ideas/x.md", rounds=2)
    assert mem.exp and mem.exp[0][0] == "council_session"
    assert mem.lessons and mem.lessons[0][0] == "council"
    # no-op without a memory manager
    await capture_council(None, "c", "t", "i")


@pytest.mark.asyncio
async def test_session_finalize_saves_document(tmp_path):
    c = _FakeConsortium(tmp_path)
    session = c.new_session("my topic")
    await session.run_round()
    result = await session.finalize()
    assert session.finalized and result["rounds"] == 1
    saved = next((tmp_path / "ideas").glob("*.md"))
    assert "# Research ideas — my topic" in saved.read_text()
