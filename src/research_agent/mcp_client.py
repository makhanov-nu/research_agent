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

import httpx
from langchain_core.tools import BaseTool

from .config import settings

logger = logging.getLogger(__name__)

_ENV_RE = re.compile(r"\$\{([^}]+)\}")


def _make_resilient(tool: BaseTool) -> None:
    """Patch a tool so transport-level HTTP errors return a recoverable string.

    LangGraph's _default_handle_tool_errors only catches ToolException /
    ToolInvocationError. httpx.HTTPStatusError (e.g. 504 from paperclip) is
    neither, so it propagates out of the ToolNode and crashes the whole task.
    We intercept it in ainvoke before LangGraph ever sees it.

    BaseTool (StructuredTool) is a Pydantic model, so ordinary setattr is
    blocked for non-field names. object.__setattr__ bypasses the validator and
    puts the wrapper directly in the instance __dict__, which Python's attribute
    lookup finds before the class method (non-data descriptor).
    """
    tool.handle_tool_error = True
    orig_ainvoke = tool.ainvoke  # bound method on the instance

    async def _safe_ainvoke(inp, config=None, **kwargs):
        try:
            return await orig_ainvoke(inp, config, **kwargs)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            msg = f"MCP server returned HTTP {status}. The server may be temporarily unavailable; try the request again or use a different approach."
            logger.warning("MCP tool %s HTTP %s error", tool.name, status)
            return msg

    # Bypass Pydantic's __setattr__ which rejects non-field names.
    object.__setattr__(tool, "ainvoke", _safe_ainvoke)


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
    for t in tools:
        _make_resilient(t)
    logger.info(
        "Loaded %d tool(s) from MCP server(s): %s",
        len(tools),
        ", ".join(servers),
    )
    return tools
