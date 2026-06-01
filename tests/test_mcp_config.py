"""Tests for MCP server config loading (no network)."""

from __future__ import annotations

import json

from research_agent import mcp_client
from research_agent.config import settings


def test_builtin_paperclip_added_when_key_set(monkeypatch):
    monkeypatch.setattr(settings, "paperclip_api_key", "gxl_test")
    monkeypatch.setattr(settings, "mcp_config_path", "does_not_exist.json")
    servers = mcp_client.load_server_config()
    assert "paperclip" in servers
    assert servers["paperclip"]["transport"] == "streamable_http"
    assert servers["paperclip"]["headers"]["X-API-Key"] == "gxl_test"


def test_no_servers_when_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "paperclip_api_key", "")
    monkeypatch.setattr(settings, "mcp_config_path", "does_not_exist.json")
    assert mcp_client.load_server_config() == {}


def test_json_file_merges_and_expands_env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "paperclip_api_key", "")
    monkeypatch.setenv("MY_KEY", "secret123")
    cfg = tmp_path / "mcp_servers.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "extra": {
                        "url": "https://x/mcp",
                        "transport": "streamable_http",
                        "headers": {"Authorization": "Bearer ${MY_KEY}"},
                    }
                }
            }
        )
    )
    monkeypatch.setattr(settings, "mcp_config_path", str(cfg))
    servers = mcp_client.load_server_config()
    assert servers["extra"]["headers"]["Authorization"] == "Bearer secret123"
