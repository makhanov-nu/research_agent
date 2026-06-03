"""Tests for the web layer's pure logic (path safety + session sealing)."""

from __future__ import annotations

import pytest

from research_agent.web.app import _safe_path


def test_safe_path_accepts_within_outputs(monkeypatch, tmp_path):
    from research_agent.web import app as app_mod

    monkeypatch.setattr(app_mod.settings, "output_dir", str(tmp_path))
    (tmp_path / "projects" / "p" / "lit_review").mkdir(parents=True)
    f = tmp_path / "projects" / "p" / "lit_review" / "x.tex"
    f.write_text("hi")
    assert _safe_path("projects/p/lit_review/x.tex") == f.resolve()


def test_safe_path_rejects_traversal(monkeypatch, tmp_path):
    from research_agent.web import app as app_mod

    monkeypatch.setattr(app_mod.settings, "output_dir", str(tmp_path))
    with pytest.raises(ValueError):
        _safe_path("../../etc/passwd")


def test_session_seal_roundtrip_and_tamper():
    itsdangerous = pytest.importorskip("itsdangerous")  # noqa: F841
    from research_agent.web import auth

    token = auth.seal_session({"email": "a@b.com", "name": "A"})
    assert auth.read_session(token) == {"email": "a@b.com", "name": "A"}
    assert auth.read_session(token + "tampered") is None
    assert auth.read_session(None) is None
