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
