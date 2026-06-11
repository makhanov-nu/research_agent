"""Tests for settings helpers."""

from __future__ import annotations

from research_agent.config import settings


def test_pg_components_parses_dsn(monkeypatch):
    monkeypatch.setattr(
        settings, "database_url", "postgresql://alice:s3cr%40t@db.host:6543/mydb"
    )
    pg = settings.pg_components()
    assert pg["host"] == "db.host"
    assert pg["port"] == 6543
    assert pg["user"] == "alice"
    assert pg["password"] == "s3cr@t"  # URL-decoded
    assert pg["dbname"] == "mydb"


def test_memory_enabled_tracks_database_url(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "")
    assert settings.memory_enabled is False
    monkeypatch.setattr(settings, "database_url", "postgresql://x/y")
    assert settings.memory_enabled is True


def test_role_model_map_empty_string(monkeypatch):
    monkeypatch.setattr(settings, "role_models", "")
    assert settings.role_model_map == {}


def test_role_model_map_single_pair(monkeypatch):
    monkeypatch.setattr(settings, "role_models", "code_reader=qwen/qwen3.7-plus")
    assert settings.role_model_map == {"code_reader": "qwen/qwen3.7-plus"}


def test_role_model_map_multiple_pairs(monkeypatch):
    monkeypatch.setattr(
        settings, "role_models",
        "code_reader=qwen/qwen3.7-plus,methodology=deepseek/deepseek-v4-pro"
    )
    assert settings.role_model_map == {
        "code_reader": "qwen/qwen3.7-plus",
        "methodology": "deepseek/deepseek-v4-pro",
    }


def test_role_model_map_trims_whitespace(monkeypatch):
    monkeypatch.setattr(
        settings, "role_models",
        "  code_reader  =  qwen/qwen3.7-plus  ,  methodology = deepseek/deepseek-v4-pro  "
    )
    assert settings.role_model_map == {
        "code_reader": "qwen/qwen3.7-plus",
        "methodology": "deepseek/deepseek-v4-pro",
    }


def test_role_model_map_skips_malformed_pairs(monkeypatch):
    monkeypatch.setattr(
        settings, "role_models",
        "code_reader=qwen/qwen3.7-plus,invalid,=empty,methodology=deepseek/deepseek-v4-pro"
    )
    assert settings.role_model_map == {
        "code_reader": "qwen/qwen3.7-plus",
        "methodology": "deepseek/deepseek-v4-pro",
    }
