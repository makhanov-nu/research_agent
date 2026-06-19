"""Diagnostic probe for the ideation consortium panel models.

Runs against the CONFIGURED provider (settings.llm_provider — OpenRouter or
DeepInfra) for every slug in `settings.panel_models` plus the chair. It answers
the two questions that gate the consortium StateGraph rewrite:

1. **Empty-content mechanism.** Some panelists (e.g. Kimi-K2.6) returned empty
   `.content` in real runs. For a trivial prompt we print `finish_reason`, the
   `additional_kwargs` keys, and whether a reasoning channel
   (`reasoning_content` / `reasoning`) carried the answer instead. This
   discriminates: reasoning-channel-not-captured vs max_tokens-truncation vs other.

2. **Structured-output support.** The rewrite wants `with_structured_output` to
   constrain idea generation *at the source* (no regex parsing). Reasoning models
   often don't support function-call / json-schema structured output cleanly. For
   each model we attempt `llm.with_structured_output(Idea).ainvoke(...)` and report
   whether a valid object came back. The result seeds each model's finalize
   strategy: "structured" (works) vs "free_text" (fall back to strip+parse).

Network-using; run where provider creds exist (local .env copied from the vps,
or on research-vps):  python scripts/diagnose_panel_models.py
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from research_agent.config import settings
from research_agent.llm import build_openrouter_chat


class Idea(BaseModel):
    """The minimal idea schema the rewrite would constrain generation to."""

    title: str = Field(description="Concise idea name")
    problem: str = Field(description="Problem and motivation, 1-2 sentences")
    method: str = Field(description="Key approach in one sentence")


def _reasoning_channel(msg) -> tuple[bool, int]:
    """(present, length) of any reasoning-channel content on the message."""
    ak = getattr(msg, "additional_kwargs", {}) or {}
    for key in ("reasoning_content", "reasoning"):
        val = ak.get(key)
        if val:
            return True, len(str(val))
    return False, 0


async def probe_content(model: str) -> str:
    """Trivial call; surface finish_reason / channels for the empty-content question."""
    llm = build_openrouter_chat(model, temperature=0.0, max_tokens=512)
    try:
        msg = await llm.ainvoke([HumanMessage(content="Reply with exactly: OK")])
    except Exception as exc:  # noqa: BLE001
        return f"  content : ERROR {type(exc).__name__}: {str(exc)[:160]}"
    content = (msg.content or "") if isinstance(msg.content, str) else str(msg.content)
    finish = (getattr(msg, "response_metadata", {}) or {}).get("finish_reason")
    ak_keys = sorted((getattr(msg, "additional_kwargs", {}) or {}).keys())
    has_reason, rlen = _reasoning_channel(msg)
    return (
        f"  content : len={len(content):<5} finish_reason={finish!r:<12} "
        f"additional_kwargs={ak_keys} reasoning_channel={has_reason}(len={rlen})"
    )


async def probe_structured(model: str) -> str:
    """Attempt structured output; report whether a valid Idea object returns."""
    llm = build_openrouter_chat(model, temperature=0.0, max_tokens=1024)
    prompt = (
        "Propose ONE tiny research idea about federated few-shot face recognition. "
        "Return it in the required structure."
    )
    try:
        structured = llm.with_structured_output(Idea)
        result = await structured.ainvoke([HumanMessage(content=prompt)])
    except Exception as exc:  # noqa: BLE001
        return f"  struct  : FAIL  {type(exc).__name__}: {str(exc)[:200]}"
    if isinstance(result, Idea) and result.title and result.method:
        return f"  struct  : OK    title={result.title[:60]!r}"
    return f"  struct  : WEAK  returned={type(result).__name__} value={str(result)[:120]!r}"


async def main() -> None:
    models = list(settings.panel_models) + [settings.consortium_chair_model]
    print(f"provider={settings.llm_provider}  models={len(models)}\n")
    for model in models:
        print(f"### {model}")
        # Sequential per model so one slow/failing model's output stays grouped.
        print(await probe_content(model))
        print(await probe_structured(model))
        print()


if __name__ == "__main__":
    asyncio.run(main())
