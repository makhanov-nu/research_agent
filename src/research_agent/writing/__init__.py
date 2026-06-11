"""Writing subagents: drafting LaTeX research artifacts.

`Writers` bundles the three LaTeX writers (literature review, methodology, paper)
so callers thread a single object instead of three. Build it once per process
with `build_writers(llm, tools, output_dir)`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .lit_review import LiteratureReviewer
from .methodology import MethodologyWriter
from .paper import PaperWriter


@dataclass
class Writers:
    reviewer: LiteratureReviewer
    methodologist: MethodologyWriter
    paper_writer: PaperWriter


def build_writers(llm, tools, output_dir: str, model_for_role=None) -> Writers:
    """Construct the LaTeX writers, all sharing the literature tools.

    Args:
        llm: The default LLM for roles without overrides.
        tools: The literature/MCP tools available to the writers.
        output_dir: Directory for saving artifacts.
        model_for_role: Optional callable(role: str) -> BaseChatModel to resolve
            per-role models. Defaults to a lambda returning llm for all roles.
    """
    if model_for_role is None:
        model_for_role = lambda role: llm  # noqa: E731

    return Writers(
        reviewer=LiteratureReviewer(model_for_role("literature_review"), tools, output_dir),
        methodologist=MethodologyWriter(model_for_role("methodology"), tools, output_dir),
        paper_writer=PaperWriter(model_for_role("paper_draft"), tools, output_dir),
    )


__all__ = [
    "LiteratureReviewer",
    "MethodologyWriter",
    "PaperWriter",
    "Writers",
    "build_writers",
]
