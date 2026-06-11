"""Load the agent's tools from MCP servers.

The agent's external capabilities (literature search/read, and later more) come
from MCP servers rather than hand-rolled API clients. paperclip is built in;
additional servers can be declared in a JSON config file (MCP_CONFIG_PATH) so
new capabilities can be plugged in without code changes.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from langchain_core.tools import BaseTool

from .config import settings

logger = logging.getLogger(__name__)

_ENV_RE = re.compile(r"\$\{([^}]+)\}")


def _expand_env(obj):
    """Recursively substitute ${VAR} occurrences from the environment."""
    if isinstance(obj, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def _builtin_servers() -> dict:
    """Servers configured directly from settings."""
    servers: dict = {}
    if settings.paperclip_api_key:
        servers["paperclip"] = {
            "url": settings.paperclip_url,
            "transport": "streamable_http",
            "headers": {"X-API-Key": settings.paperclip_api_key},
        }
    if settings.tavily_api_key:
        # Tavily's hosted MCP takes the key as a query param on the URL.
        sep = "&" if "?" in settings.tavily_mcp_url else "?"
        servers["tavily"] = {
            "url": f"{settings.tavily_mcp_url}{sep}tavilyApiKey={settings.tavily_api_key}",
            "transport": "streamable_http",
        }
    return servers


def load_server_config() -> dict:
    """Merge built-in servers with any declared in the JSON config file.

    The JSON file may either be a bare mapping of name -> connection, or wrapped
    in an `mcpServers` key (the convention used by Cursor / Claude Desktop).
    File entries override built-ins on name collision.
    """
    servers = _builtin_servers()

    path = Path(settings.mcp_config_path)
    if path.exists():
        raw = json.loads(path.read_text())
        raw = raw.get("mcpServers", raw) if isinstance(raw, dict) else {}
        servers.update(_expand_env(raw))

    return servers


async def load_mcp_tools() -> list[BaseTool]:
    """Connect to every configured MCP server and return their tools."""
    servers = load_server_config()
    if not servers:
        logger.warning(
            "No MCP servers configured. Set PAPERCLIP_API_KEY or provide %s. "
            "The agent will run without tools.",
            settings.mcp_config_path,
        )
        return []

    # Imported lazily so the package imports even before deps are installed.
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(servers)
    tools = await client.get_tools()
    # MCP servers surface errors as ToolException. LangGraph's default handler
    # only catches ToolInvocationError (validation failures), so plain
    # ToolException propagates out of ainvoke and crashes the task. Setting
    # handle_tool_error=True on each tool makes BaseTool catch it at the tool
    # level and return the error string to the LLM so it can retry.
    for t in tools:
        t.handle_tool_error = True
    logger.info(
        "Loaded %d tool(s) from MCP server(s): %s",
        len(tools),
        ", ".join(servers),
    )
    return tools
