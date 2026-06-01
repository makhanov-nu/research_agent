"""Literature-review drafting subagent.

Given a topic, runs a bounded tool-using loop (with the literature tools) to
gather and read related papers, then writes a structured LaTeX `Related Work`
section plus a matching BibTeX file, and saves both to disk. The parsing and
saving helpers are pure and unit-tested; the gathering/writing step uses the LLM.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

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


def slugify(text: str, max_len: int = 50) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:max_len].rstrip("-")) or "review"


def extract_code_block(text: str, lang: str) -> str:
    """Return the contents of the first ```<lang> fenced block, or ""."""
    m = re.search(rf"```{lang}\b[ \t]*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def count_bib_entries(bibtex: str) -> int:
    return len(re.findall(r"^\s*@\w+\s*\{", bibtex, re.MULTILINE))


def parse_review(text: str) -> tuple[str, str, int]:
    """Split the model output into (latex, bibtex, n_refs).

    Falls back to treating the whole text as LaTeX if no fenced block is present.
    """
    latex = extract_code_block(text, "latex") or text.strip()
    bibtex = extract_code_block(text, "bibtex")
    return latex, bibtex, count_bib_entries(bibtex)


class LiteratureReviewer:
    def __init__(self, llm, tools, output_dir: str):
        self.llm = llm
        self.tools = tools
        self.output_dir = Path(output_dir) / "lit_reviews"

    def save_review(self, name: str, latex: str, bibtex: str) -> tuple[str, str]:
        """Write the .tex (+ .bib) files; return their paths."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tex_path = self.output_dir / f"{name}.tex"
        tex_path.write_text(latex)
        bib_path = self.output_dir / f"{name}.bib"
        if bibtex:
            bib_path.write_text(bibtex)
        return str(tex_path), str(bib_path) if bibtex else ""

    async def draft(
        self, topic: str, focus: str = "", venue: str = "", save_name: str = ""
    ) -> dict:
        from langgraph.prebuilt import create_react_agent

        agent = create_react_agent(self.llm, self.tools, prompt=_SYSTEM)
        task = f"Topic: {topic}"
        if focus:
            task += f"\nFocus on: {focus}"
        if venue:
            task += f"\nTarget venue/style: {venue}"
        task += "\nResearch the literature and write the Related Work section now."

        result = await agent.ainvoke(
            {"messages": [("user", task)]}, config={"recursion_limit": 40}
        )
        final = result["messages"][-1].content
        if isinstance(final, list):
            final = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in final
            )

        latex, bibtex, n_refs = parse_review(final)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = slugify(save_name or topic) + f"-{stamp}"
        tex_path, bib_path = self.save_review(name, latex, bibtex)
        logger.info("Wrote literature review %s (%d refs)", tex_path, n_refs)
        return {
            "tex_path": tex_path,
            "bib_path": bib_path,
            "n_refs": n_refs,
            "latex": latex,
        }
