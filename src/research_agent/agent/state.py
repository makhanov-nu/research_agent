"""Graph state for the research agent."""

from __future__ import annotations

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Shared state flowing through the graph.

    Extends MessagesState (the running ``messages`` list with the add_messages
    reducer) with memory bookkeeping:

    - summary:           rolling summary of older, summarized-away turns
    - context_block:     facts/procedures recalled for the current turn (transient)
    - nudge:             a context note for the model to act on (transient)
    - cumulative_tokens: monotonic estimate of total conversation size
    - last_nudge_tokens: cumulative-token mark at which we last nudged the user
    """

    summary: str
    context_block: str
    nudge: str
    cumulative_tokens: int
    last_nudge_tokens: int
