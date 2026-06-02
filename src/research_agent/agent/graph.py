"""LangGraph definition for the research agent.

    START -> load_context -> agent -> (tools -> agent)* -> END

`load_context` manages memory each turn: it rolls older messages into a summary
when context grows past a threshold (saving tokens), recalls relevant facts +
learned procedures, tracks a monotonic token count, and raises a "checkpoint?"
nudge each time the conversation crosses a 20k-token band. The `agent` node then
builds its system prompt from the base persona plus that live memory context.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ..config import settings
from ..llm import get_llm
from ..mcp_client import load_mcp_tools
from ..memory.summarize import messages_to_drop, summarize_messages
from ..memory.tokens import (
    crossed_nudge_boundary,
    estimate_message_tokens,
    estimate_text_tokens,
    _content_to_text,
)
from ..prompts import compose_system_prompt
from .state import AgentState


def _latest_text(messages: list, message_type) -> str:
    for m in reversed(messages):
        if isinstance(m, message_type):
            return _content_to_text(m.content)
    return ""


async def build_graph(
    checkpointer: BaseCheckpointSaver | None = None, memory=None,
    experiment_runner=None, mcp_tools=None, consortium=None, task_store=None,
    dispatcher=None,
):
    """Build and compile the orchestrator agent graph.

    The orchestrator's tools delegate to specialized subagents (literature,
    review writing, ideation consortium, experiments). It does NOT get raw
    literature tools directly — those live inside the literature subagent — so
    intermediate search output never enters the orchestrator's context.

    Args:
        checkpointer: persistence backend for conversation state.
        memory: a MemoryManager, or None to run without long-term memory.
        experiment_runner: an ExperimentRunner, or None.
        mcp_tools: preloaded MCP tools to reuse; loaded here when not provided.
        consortium: a Consortium, or None.
    """
    if mcp_tools is None:
        mcp_tools = await load_mcp_tools()

    from ..agents import build_delegated_tools
    from ..writing.lit_review import LiteratureReviewer

    llm = get_llm()
    reviewer = LiteratureReviewer(get_llm(), mcp_tools, settings.output_dir)
    tools = build_delegated_tools(
        llm=get_llm(), mcp_tools=mcp_tools, reviewer=reviewer,
        experiment_runner=experiment_runner, consortium=consortium,
        task_store=task_store,
    )
    if dispatcher is not None:
        from ..agents.dispatcher import build_dispatch_tools

        tools += build_dispatch_tools(dispatcher)

    llm_with_tools = llm.bind_tools(tools) if tools else llm

    async def load_context(state: AgentState) -> dict:
        messages = state["messages"]
        summary = state.get("summary", "")
        updates: dict = {}

        # 1) Roll older messages into the summary when live context is large.
        live_tokens = estimate_message_tokens(messages) + estimate_text_tokens(summary)
        if (
            live_tokens > settings.summary_token_threshold
            and len(messages) > settings.summary_keep_last
        ):
            older = messages[: -settings.summary_keep_last]
            summary = await summarize_messages(llm, older, summary)
            updates["summary"] = summary
            updates["messages"] = messages_to_drop(messages, settings.summary_keep_last)

        # 2) Recall facts + learned procedures relevant to the latest user turn.
        query = _latest_text(messages, HumanMessage)
        if memory is not None:
            updates["context_block"] = await memory.build_context(query)
        else:
            updates["context_block"] = ""

        # 3) Monotonic conversation-size estimate -> 20k nudge bands.
        delta = estimate_text_tokens(query) + estimate_text_tokens(
            _latest_text(messages, AIMessage)
        )
        cumulative = state.get("cumulative_tokens", 0) + delta
        updates["cumulative_tokens"] = cumulative

        last_nudge = state.get("last_nudge_tokens", 0)
        if crossed_nudge_boundary(cumulative, last_nudge, settings.nudge_every_tokens):
            approx_k = round(cumulative / 1000)
            updates["nudge"] = (
                f"This conversation is now ~{approx_k}k tokens long. Briefly let the "
                "researcher know, and ask whether they'd like you to summarize and "
                "checkpoint it to long-term memory (they can reply or use "
                "`!checkpoint`). Then continue with their request."
            )
            updates["last_nudge_tokens"] = cumulative
        else:
            updates["nudge"] = ""

        return updates

    async def agent_node(state: AgentState) -> dict:
        system = compose_system_prompt(
            summary=state.get("summary", ""),
            context_block=state.get("context_block", ""),
            nudge=state.get("nudge", ""),
        )
        messages = [SystemMessage(content=system), *state["messages"]]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("load_context", load_context)
    builder.add_node("agent", agent_node)
    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "agent")

    if tools:
        builder.add_node("tools", ToolNode(tools))
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")
    else:
        builder.add_edge("agent", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())
