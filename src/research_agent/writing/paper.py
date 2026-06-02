"""Paper-drafting subagent.

Drafts a research paper — or a specific section of one — in LaTeX, weaving
together the materials the orchestrator supplies (the idea/contribution, the
methodology, related work, and any experimental findings). It can use the
literature tools to fill citation gaps, but its main job is composition: turning
already-gathered material into clean, submission-style LaTeX prose.
"""

from __future__ import annotations

from .latex import LatexWriter

_SYSTEM = """You are an academic writer drafting a research paper in LaTeX for a \
top venue. You are given a brief and supporting material (contribution, \
methodology, related work, findings). Compose clear, precise, submission-style \
prose — not a list of notes.

Guidance:
- Write the sections the brief asks for. If it asks for a full paper, produce: \
title, abstract, \\section{Introduction}, \\section{Related Work}, \
\\section{Methodology}, \\section{Experiments}, \\section{Results}, \
\\section{Conclusion} — but reuse material already provided rather than \
re-deriving it.
- Make the contribution and its novelty explicit early; ground every empirical \
claim in the provided findings. Do NOT invent results, numbers, papers, or \
citations — if a citation or number is missing, use the literature tools to find \
it or insert a clearly-marked TODO for the researcher instead of fabricating.
- Use \\cite{key} keys that match BibTeX entries you provide.

Return EXACTLY two fenced code blocks and nothing else of substance:
- a ```latex block containing the requested LaTeX (a self-contained document or \
the requested section(s), using \\cite{...}); and
- a ```bibtex block containing every cited entry (may be empty if nothing is \
cited).
You may add one short plain-text sentence after the blocks listing any TODOs or \
gaps the researcher should fill.
"""


class PaperWriter(LatexWriter):
    system_prompt = _SYSTEM
    subdir = "papers"

    async def draft(
        self,
        brief: str,
        material: str = "",
        sections: str = "",
        venue: str = "",
        save_name: str = "",
    ) -> dict:
        task = f"Brief:\n{brief}"
        if sections:
            task += f"\n\nWrite these sections: {sections}"
        if venue:
            task += f"\nTarget venue/style: {venue}"
        if material:
            task += f"\n\nSupporting material to use (do not contradict it):\n{material}"
        task += "\n\nDraft the LaTeX now."
        return await self._draft(task, slug_source=brief, save_name=save_name)
