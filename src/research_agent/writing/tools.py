"""Agent tools for writing research artifacts."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool, tool


def build_writing_tools(reviewer) -> list[BaseTool]:
    """Return writing tools bound to a LiteratureReviewer."""

    @tool
    async def draft_literature_review(
        topic: str, focus: str = "", venue: str = "", save_name: str = ""
    ) -> str:
        """Research the literature on a topic and write a LaTeX Related Work section.

        Gathers and reads related papers, then writes a thematically-organized
        LaTeX literature review with \\cite keys plus a matching BibTeX file, and
        saves both to disk.

        Args:
            topic: The subject of the review (e.g. "speculative decoding for LLMs").
            focus: Optional angle to emphasize (e.g. "training-free methods").
            venue: Optional target venue/style (e.g. "NeurIPS", "Nature Methods").
            save_name: Optional base filename; defaults to a slug of the topic.
        """
        result = await reviewer.draft(topic, focus=focus, venue=venue, save_name=save_name)
        tex = Path(result["tex_path"]).name
        return (
            f"Wrote a LaTeX literature review with {result['n_refs']} references.\n"
            f"Saved: `{tex}`"
            + (f" (+ `{Path(result['bib_path']).name}`)" if result["bib_path"] else "")
            + f"\nRetrieve it with `!getfile lit_reviews/{tex}`."
        )

    return [draft_literature_review]
