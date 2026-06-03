"""Literature-review drafting subagent.

Given a topic, runs a bounded tool-using loop (with the literature tools) to
gather and read related papers, then writes a structured LaTeX `Related Work`
section plus a matching BibTeX file, and saves both to disk. The parsing and
saving helpers live in `writing.latex` (pure and unit-tested); the
gathering/writing step uses the LLM.
"""

from __future__ import annotations

from .latex import (  # re-exported for backwards-compatible imports
    LatexWriter,
    count_bib_entries,
    extract_code_block,
    parse_latex_artifact,
    slugify,
)

# Back-compat alias: this module historically exposed `parse_review`.
parse_review = parse_latex_artifact

__all__ = [
    "LiteratureReviewer",
    "count_bib_entries",
    "extract_code_block",
    "parse_review",
    "slugify",
]

_SYSTEM = """You are writing the Related Work / Literature Review section of a \
research paper, in LaTeX. Work rigorously:

1. Use the available literature tools to FIND and READ real, relevant papers on \
the topic. Do not invent papers, authors, venues, years, or results — cite only \
sources you actually retrieved.
2. Organize the review thematically (not paper-by-paper): group approaches, \
compare them, trace how ideas developed, and explicitly identify open gaps the \
new work could address.
3. Cite with \\cite{key} keys that match BibTeX entries you provide.

Return EXACTLY two fenced code blocks and nothing else of substance:
- a ```latex block containing a self-contained \\section{Related Work} (using \
\\cite{...}); and
- a ```bibtex block containing every cited entry.
You may add one short plain-text sentence after the blocks summarizing coverage.
"""


class LiteratureReviewer(LatexWriter):
    system_prompt = _SYSTEM
    subdir = "lit_reviews"

    def save_review(self, name: str, latex: str, bibtex: str) -> tuple[str, str]:
        return self.save(name, latex, bibtex)

    async def draft(
        self, topic: str, focus: str = "", venue: str = "", save_name: str = "",
        dirpath=None,
    ) -> dict:
        task = f"Topic: {topic}"
        if focus:
            task += f"\nFocus on: {focus}"
        if venue:
            task += f"\nTarget venue/style: {venue}"
        task += "\nResearch the literature and write the Related Work section now."
        return await self._draft(
            task, slug_source=topic, save_name=save_name, dirpath=dirpath
        )
