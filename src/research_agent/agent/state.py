"""Graph state for the research agent."""

from __future__ import annotations

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Shared state flowing through the graph.

    Extends MessagesState, which carries the running ``messages`` list with the
    add_messages reducer. Future milestones add fields here (e.g. the active
    research project, methodology drafts, experiment handles).
    """
