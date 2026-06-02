"""Literature research subagent (delegation tool)."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from .subagent import build_subagent_tool

_SYSTEM = """You are a literature research subagent. You receive one self-contained \
research question or task. Use the literature tools to search for and read the \
relevant papers, then return a concise, well-organized synthesis that directly \
answers the task: compare approaches, note what is well-established vs. contested, \
and flag open gaps. Always cite sources with identifiers/links (arXiv id, DOI, \
PMID, URL). Do not invent papers. Return ONLY the synthesis — the caller does not \
see your intermediate searches, so make the answer stand on its own."""

_DESCRIPTION = (
    "Delegate a literature-research task to the literature subagent. Pass a "
    "COMPLETE, self-contained instruction (topic, scope, what to find or "
    "compare). It searches and reads papers and returns a synthesized, cited "
    "answer. Use this for anything needing sources; do not recall papers yourself."
)


def build_literature_agent_tool(llm, lit_tools) -> BaseTool:
    return build_subagent_tool(
        name="research_literature",
        description=_DESCRIPTION,
        system_prompt=_SYSTEM,
        tools=lit_tools,
        llm=llm,
    )
