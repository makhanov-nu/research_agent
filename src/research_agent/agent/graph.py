"""LangGraph definition for the research agent.

A standard tool-using (ReAct) loop:

    START -> agent -> (tools? -> agent)* -> END

The agent node calls the LLM; if it requests tools, the ToolNode runs them and
control returns to the agent. Tools are loaded from MCP servers at build time,
so building the graph is async.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ..llm import get_llm
from ..mcp_client import load_mcp_tools
from ..prompts import SYSTEM_PROMPT
from .state import AgentState


async def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Build and compile the research agent graph.

    Args:
        checkpointer: Persistence backend for conversation state. Defaults to an
            in-memory saver (per-process; resets on restart).
    """
    tools = await load_mcp_tools()
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    async def agent_node(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_edge(START, "agent")

    if tools:
        builder.add_node("tools", ToolNode(tools))
        # tools_condition routes to "tools" when the LLM requested a tool call,
        # otherwise to END.
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")
    else:
        builder.add_edge("agent", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())
