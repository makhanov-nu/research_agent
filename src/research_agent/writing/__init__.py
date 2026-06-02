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


def build_writers(llm, tools, output_dir: str) -> Writers:
    """Construct the LaTeX writers, all sharing the LLM and literature tools."""
    return Writers(
        reviewer=LiteratureReviewer(llm, tools, output_dir),
        methodologist=MethodologyWriter(llm, tools, output_dir),
        paper_writer=PaperWriter(llm, tools, output_dir),
    )


__all__ = [
    "LiteratureReviewer",
    "MethodologyWriter",
    "PaperWriter",
    "Writers",
    "build_writers",
]
