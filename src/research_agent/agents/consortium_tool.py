"""Delegation tool wrapping the ideation consortium.

Returns only the synthesized ideas to the orchestrator; the full multi-model
transcript is saved to disk (retrievable via !getfile), so the orchestrator's
context stays small.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool


def build_consortium_tool(consortium) -> BaseTool:
    @tool("brainstorm_research_ideas")
    async def brainstorm_research_ideas(topic: str, focus: str = "") -> str:
        """Convene the multi-model consortium to debate and propose Q1-level
        research ideas on a topic (panel discusses in a shared session, then a
        chair synthesizes). Returns the synthesized ideas; the full transcript is
        saved to disk.
        """
        result = await consortium.ideate(topic, focus=focus)
        return (
            result["ideas"]
            + f"\n\n(Full shared-session transcript saved: {result['rel_path']})"
        )

    return brainstorm_research_ideas
