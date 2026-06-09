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


def test_request_annotations_resolve_to_injected_request():
    """Regression guard for the `request: Request` -> 422 bug.

    Endpoints defined inside create_app() annotate `request: Request`. Under
    `from __future__ import annotations` that annotation is the bare string
    "Request", which FastAPI can only resolve if the name is a module global
    (see the module-level `from starlette.requests import Request` in app.py).
    If that import is missing/removed, FastAPI silently treats `request` as a
    required *query* param and the route 422s on every authenticated call. This
    asserts no route exposes a `request` query param, and that the proxy/SSE
    routes correctly bind it as the injected Request.
    """
    pytest.importorskip("fastapi")
    from research_agent.web.app import create_app

    app = create_app()
    by_path = {getattr(r, "path", None): r for r in app.routes}

    for route in app.routes:
        names = {q.name for q in getattr(getattr(route, "dependant", None), "query_params", [])}
        assert "request" not in names, f"{getattr(route, 'path', route)} leaks a `request` query param"

    for path in ("/phoenix/{path:path}", "/api/tasks/stream"):
        assert by_path[path].dependant.request_param_name == "request"
