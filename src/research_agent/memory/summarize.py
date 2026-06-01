"""Rolling conversation summarization to save tokens.

When the live message history grows past a threshold, we fold all but the most
recent messages into a running summary and drop them from state (via
RemoveMessage). The summary is carried in graph state and also persisted to the
episodic store, so context stays small without losing the thread.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage

from .tokens import _content_to_text

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM = (
    "You maintain a running summary of an ongoing research conversation between "
    "a researcher and their research agent. Produce a dense, factual summary that "
    "preserves: the research goal, key decisions, methodology choices, important "
    "findings and the papers/sources behind them, open questions, and next steps. "
    "Prefer bullet points. Do not invent details."
)


async def summarize_messages(llm, messages: list, existing_summary: str = "") -> str:
    """Return an updated summary folding `messages` into `existing_summary`."""
    convo = "\n".join(
        f"{getattr(m, 'type', 'msg')}: {_content_to_text(getattr(m, 'content', ''))}"
        for m in messages
    )
    prior = f"Existing summary so far:\n{existing_summary}\n\n" if existing_summary else ""
    prompt = [
        SystemMessage(content=_SUMMARY_SYSTEM),
        HumanMessage(
            content=(
                f"{prior}New conversation turns to fold in:\n{convo}\n\n"
                "Return the updated summary."
            )
        ),
    ]
    resp = await llm.ainvoke(prompt)
    return _content_to_text(resp.content).strip()


def messages_to_drop(messages: list, keep_last: int) -> list[RemoveMessage]:
    """RemoveMessage ops for every message except the last `keep_last`."""
    if len(messages) <= keep_last:
        return []
    drop = messages[:-keep_last] if keep_last > 0 else messages
    return [RemoveMessage(id=m.id) for m in drop if getattr(m, "id", None)]
