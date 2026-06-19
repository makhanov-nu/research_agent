"""Graph state for the ideation consortium.

Two schemas:

- ``ConsortiumState`` — the top-level run: brief → propose (fan-out) → debate →
  extract → assemble pool → score (fan-out) → aggregate. Fields written by the
  parallel ``run_panelist`` / ``run_scorer`` branches carry additive reducers so
  concurrent superstep writes merge instead of raising ``InvalidUpdateError``.
- ``PanelistState`` — one panelist's bounded agent loop (agent ⇄ tools → finalize)
  used for both proposing and chair extraction. ``messages`` carries the standard
  ``add_messages`` reducer (via ``MessagesState``); the budget/attempt counters
  drive the guard edges that replace the raw ``recursion_limit`` wall.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import MessagesState


def merge_threads(a: dict | None, b: dict | None) -> dict:
    """Reducer: union of per-model propose threads from parallel branches."""
    return {**(a or {}), **(b or {})}


class ConsortiumState(TypedDict, total=False):
    """Shared state for one consortium round."""

    topic: str
    focus: str
    brief: str
    prior: str
    # Accumulated across the parallel panelist/scorer branches (additive reducers):
    ideas: Annotated[list[dict], operator.add]          # {text, source, by}
    threads: Annotated[dict[str, list], merge_threads]  # model -> propose msg history
    transcript: Annotated[list[tuple], operator.add]    # (speaker, text) debate turns
    ballots: Annotated[list[dict], operator.add]        # {model, scores:{id:(s,reason)}}
    flags: Annotated[list[str], operator.add]
    trace: Annotated[list[dict], operator.add]
    # Single-writer nodes:
    pool: list[dict]                                    # ideas with ids assigned
    ranked: list[dict]                                  # pool, scored + sorted


class PanelistState(MessagesState):
    """One panelist's bounded propose/extract loop.

    Extends ``MessagesState`` (``messages`` + ``add_messages`` reducer) with the
    bookkeeping the guard edges read.
    """

    model: str
    phase: str          # "propose" | "extract"
    shared: bool        # debate (shared) vs independent system framing
    budget: int         # allowed tool-call rounds before forced finalize
    max_ideas: int
    attempts: int       # finalize retries used
    ideas: list[dict]   # parsed result for this panelist
    payload: dict[str, Any]   # carries brief/instruction/transcript into the loop
