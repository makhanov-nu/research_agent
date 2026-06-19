"""Scoped real-network dry-run of the consortium fixes against DeepInfra.

Everything in the test suite is offline fakes; council_31 was 100% real-model
behaviour. This validates the three real-model failure modes the rewrite targets,
WITHOUT the full (expensive) ideate pipeline:

  - GLM-5.1  PROPOSE with real tools  -> budget guard + forced finalize + parse
                                         (the recursion-wall / over-search failure)
  - Kimi-K2.6 PROPOSE with real tools -> non-empty, parseable idea
                                         (the empty-output failure)
  - chair R1 EXTRACT over a canned debate -> <think>-leak parser hardening
                                             (the council_31 scaffold-as-idea bug)

Success (per the advisor's criteria): GLM AND Kimi each yield >=1 parseable idea,
the chair extract yields ideas with no <think>/scaffold leak, no provider 400s,
and the trace comes out as {type, content, speaker}.

Run:  python scripts/dryrun_consortium.py            # scoped (cheap)
      python scripts/dryrun_consortium.py --full      # full ideate() (pricier)
"""

from __future__ import annotations

import asyncio
import logging
import sys

from langchain_core.messages import HumanMessage

from research_agent.config import settings
from research_agent.consortium.consortium import (
    Consortium,
    _DEBATE_EXTRACT_CHAIR,
    _PROPOSE_INDEP,
)
from research_agent.consortium.graph import (
    EXTRACT_BUDGET,
    PANELIST_RECURSION,
    PROPOSE_BUDGET,
    build_panelist_subgraph,
)
from research_agent.mcp_client import load_mcp_tools


class _DryConsortium(Consortium):
    """Adds a per-request timeout so a hung DeepInfra call can't stall the run."""

    def _chat(self, model, *, max_tokens, with_tools=False):
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model, temperature=self.temperature, max_tokens=max_tokens,
            api_key=settings.deepinfra_api_key or None,
            base_url=settings.deepinfra_base_url, timeout=300, max_retries=1,
        )
        if with_tools and self.tools:
            llm = llm.bind_tools(self.tools)
        return llm


def _trace_shape_ok(messages) -> bool:
    from research_agent.consortium.consortium import _collect

    steps = _collect("probe", messages)
    return all({"type", "content", "speaker"} <= set(s) for s in steps) if steps else False


async def _run_panelist(sub, model, phase, instr, max_ideas, budget):
    out = await sub.ainvoke(
        {"model": model, "phase": phase, "shared": phase == "extract",
         "budget": budget, "max_ideas": max_ideas, "attempts": 0,
         "messages": [HumanMessage(content=instr)]},
        config={"recursion_limit": PANELIST_RECURSION},
    )
    msgs = out["messages"]
    tool_rounds = sum(1 for m in msgs if getattr(m, "tool_calls", None))
    return out.get("ideas", []), tool_rounds, msgs


async def quick(model: str = "zai-org/GLM-5.1") -> None:
    """Fastest signal: one model's propose at a small budget. Confirms the budget
    guard + dangling-tool_call fix on the real model with no 400."""
    print("loading MCP tools…", flush=True)
    tools = await load_mcp_tools()
    print(f"tools: {[getattr(t, 'name', '?') for t in tools]}", flush=True)
    c = _DryConsortium(tools, settings.panel_models, settings.consortium_chair_model,
                       settings.output_dir)
    sub = build_panelist_subgraph(c)
    instr = (f"Research brief:\nFederated few-shot face recognition; propose deeply "
             f"integrated FL+SSL+FSL ideas.\n\n{_PROPOSE_INDEP}")
    print(f"### QUICK PROPOSE — {model} (budget=3)", flush=True)
    try:
        ideas, rounds, msgs = await _run_panelist(
            sub, model, "propose", instr, 3, 3)
        print(f"  ideas={len(ideas)}  tool_rounds={rounds}  "
              f"final_chars={len(getattr(msgs[-1], 'content', '') or '')}  "
              f"trace_shape_ok={_trace_shape_ok(msgs)}", flush=True)
        if ideas:
            print("  sample:", " ".join(ideas[0]["text"][:160].split()), flush=True)
        else:
            print("  ⚠️ NO parseable idea", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR {type(exc).__name__}: {str(exc)[:240]}", flush=True)


async def scoped() -> None:
    print("loading MCP tools…", flush=True)
    tools = await load_mcp_tools()
    print(f"tools: {[getattr(t, 'name', '?') for t in tools]}", flush=True)
    c = _DryConsortium(tools, settings.panel_models, settings.consortium_chair_model,
                       settings.output_dir)
    sub = build_panelist_subgraph(c)

    brief = ("Federated few-shot face recognition. Propose Q1-level ideas that "
             "deeply integrate federated learning + self-supervised + few-shot, "
             "beyond sequential piping.")
    instr = f"Research brief:\n{brief}\n\n{_PROPOSE_INDEP}"

    for model in ["zai-org/GLM-5.1", "moonshotai/Kimi-K2.6"]:
        print(f"\n### PROPOSE — {model}")
        try:
            ideas, rounds, msgs = await _run_panelist(sub, model, "propose", instr, 3, PROPOSE_BUDGET)
            print(f"  ideas={len(ideas)}  tool_rounds={rounds}  "
                  f"final_chars={len(getattr(msgs[-1], 'content', '') or '')}  "
                  f"trace_shape_ok={_trace_shape_ok(msgs)}")
            if ideas:
                print("  sample:", " ".join(ideas[0]["text"][:140].split()))
            else:
                print("  ⚠️ NO parseable idea")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {type(exc).__name__}: {str(exc)[:220]}")

    print(f"\n### EXTRACT — chair {settings.consortium_chair_model}")
    transcript = (
        "[Brief]: federated few-shot face recognition, deep FL+SSL+FSL integration.\n\n"
        "[deepseek]: I propose FedCPA — joint contrastive+prototype alignment in one stage.\n\n"
        "[qwen]: I propose FedIB-Proto — an information-bottleneck prototypical objective."
    )
    instr2 = f"{transcript}\n\n---\n{_DEBATE_EXTRACT_CHAIR}"
    try:
        ideas, rounds, msgs = await _run_panelist(
            sub, settings.consortium_chair_model, "extract", instr2, 2, EXTRACT_BUDGET)
        print(f"  ideas={len(ideas)}  tool_rounds={rounds}  trace_shape_ok={_trace_shape_ok(msgs)}")
        for i in ideas:
            leaked = "<think>" in i["text"].lower() or "[concise name]" in i["text"].lower()
            print(f"  leak={leaked}:", " ".join(i["text"][:120].split()))
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR {type(exc).__name__}: {str(exc)[:220]}")


async def full() -> None:
    print("loading MCP tools…", flush=True)
    tools = await load_mcp_tools()
    c = _DryConsortium(tools, settings.panel_models, settings.consortium_chair_model,
                       settings.output_dir)
    result = await c.ideate(
        "Federated few-shot face recognition: deep FL+SSL+FSL integration beyond sequential piping."
    )
    print("\n=== flags ===")
    print(result["flags"])
    print("\n=== ideas (ranked) ===")
    print(result["ideas"][:3000])
    print(f"\nsaved: {result['rel_path']}  ({len(result['top'])} ideas, {len(result['trace'])} trace steps)")


if __name__ == "__main__":
    # INFO-level logging surfaces one line per HTTP request (httpx) so progress
    # is visible live instead of buffered until the end.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.INFO)
    if "--quick" in sys.argv:
        slug = next((a for a in sys.argv[1:] if "/" in a and not a.endswith(".py")),
                    "zai-org/GLM-5.1")
        asyncio.run(quick(slug))
    elif "--full" in sys.argv:
        asyncio.run(full())
    else:
        asyncio.run(scoped())
