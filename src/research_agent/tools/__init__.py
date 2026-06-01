"""Native (non-MCP) tools.

Currently empty — the agent's capabilities come from MCP servers (see
``research_agent.mcp_client``). Add hand-written LangChain tools here when a
capability isn't best served by an MCP server, and expose them so the graph can
include them alongside MCP tools.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

NATIVE_TOOLS: list[BaseTool] = []
