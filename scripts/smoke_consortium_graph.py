"""Offline smoke-run for the consortium StateGraph (Phase 1).

No network: the `_chat` seam is stubbed with a fake model that ALWAYS asks to
search (the GLM-5.1 over-search failure) so the run proves the budget-guard edge
forces a finalize instead of hitting the recursion wall. Exercises the whole
topology end-to-end — brief → propose fan-out → debate → chair extract →
assemble → score fan-out (+chair rater) → aggregate — and checks the pool/ranking
come out the far side.

Run:  python scripts/smoke_consortium_graph.py
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
import uuid

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from research_agent.consortium.consortium import Consortium


@tool
def paperclip(command: str) -> str:
    """Fake literature search returning a fixed two-paper result."""
    return (
        "Found 2 papers  [s_fake]\n"
        "  1. A Fake Paper On Federated Few-Shot Faces\n"
        "     Doe, J.; Roe, R.\n"
        "     arx_2401.00001 · arXiv · 2024-01-01\n"
        "     https://arxiv.org/abs/2401.00001\n"
        "  2. Another Fake Paper\n"
        "     Smith, A.\n"
        "     arx_2401.00002 · arXiv · 2024-01-02\n"
        "     https://arxiv.org/abs/2401.00002\n"
    )


class _FakeChat:
    """Decides its reply from the last message; tool-bound => always search."""

    def __init__(self, model: str, with_tools: bool):
        self.model = model
        self.with_tools = with_tools

    def bind_tools(self, tools):  # never hit (stubbed _chat sets with_tools)
        return self

    async def ainvoke(self, messages):
        last = messages[-1]
        text = last.content if hasattr(last, "content") else str(last)
        if self.with_tools:  # simulate a model that won't stop searching
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "paperclip",
                    "args": {"command": 'search -s arxiv "federated few-shot"'},
                    "id": f"c{uuid.uuid4().hex[:8]}",
                }],
            )
        if "Score EACH idea" in text or "JSON object mapping" in text:
            ids = re.findall(r"#(\d+):", text)
            return AIMessage(content=json.dumps({i: 4 + (k % 6) for k, i in enumerate(ids)}))
        if "State your strongest research direction" in text:
            return AIMessage(content=f"{self.model}: strongest direction is X; open gap is Y.")
        n = 2 if "EXACTLY 2" in text else 3
        blocks = "\n".join(
            f"=== IDEA ===\n**Title:** {self.model} idea {k}\n"
            f"**Problem & motivation:** p{k}\n**Method:** m{k} `arx_2401.0000{k}`\n"
            f"**Expected contribution:** c{k}"
            for k in range(1, n + 1)
        )
        return AIMessage(content=blocks)


class _SmokeConsortium(Consortium):
    def _chat(self, model, *, max_tokens, with_tools=False):
        return _FakeChat(model, with_tools)

    async def make_brief(self, topic, focus=""):
        return (
            f"BRIEF for: {topic}",
            [{"type": "ai", "content": "<brief reasoning>", "speaker": "chair:brief"}],
        )


async def main() -> None:
    tmp = tempfile.mkdtemp()
    c = _SmokeConsortium(
        lit_tools=[paperclip],
        panel_models=["model-a", "model-b", "model-c"],
        chair_model="chair-r1",
        output_dir=tmp,
        recall=None,
    )
    session = c.new_session("federated few-shot face recognition")
    top = await session.run_round1()
    result = await session.finalize()

    n_indep = sum(1 for i in session.pool if i["source"] == "independent")
    n_debated = sum(1 for i in session.pool if i["source"] == "debated")
    rated = [i for i in top if i.get("n_scores", 0) > 0]

    print("=== consortium graph smoke-run ===")
    print(f"brief            : {session.brief!r}")
    print(f"pool size        : {len(session.pool)}  (independent={n_indep}, debated={n_debated})")
    print(f"ranked ideas     : {len(top)}")
    print(f"ideas with raters: {len(rated)} / {len(top)}")
    print(f"max raters/idea  : {max((i.get('n_scores', 0) for i in top), default=0)} "
          f"(panel=3 + chair)")
    print(f"flags            : {session.flags}")
    print(f"trace steps      : {len(session.trace)}")
    print(f"saved proposal   : {result['rel_path']}")
    print("top 3 by score   :")
    for i in top[:3]:
        print(f"  #{i['id']} score={i['score']:.1f} n_scores={i['n_scores']} "
              f"src={i['source']} by={i['by']}")

    # Topology assertions (the Phase-1 contract).
    assert session.brief.startswith("BRIEF"), "brief node didn't run"
    assert n_indep == 9, f"expected 3 panel × 3 ideas, got {n_indep}"
    assert n_debated == 2, f"expected 2 debated, got {n_debated}"
    assert len(top) == 11, f"expected 11 ranked, got {len(top)}"
    assert len(rated) == 11, "every idea should have at least the chair's score"
    assert max(i.get("n_scores", 0) for i in top) >= 2, "chair rater should add overlap"
    print("\nALL TOPOLOGY ASSERTIONS PASSED ✅")


if __name__ == "__main__":
    asyncio.run(main())
