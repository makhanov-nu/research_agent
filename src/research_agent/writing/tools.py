"""Agent tools for writing research artifacts."""

from __future__ import annotations

from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool


def _channel(config: RunnableConfig | None) -> str | None:
    if not config:
        return None
    return (config.get("configurable") or {}).get("thread_id")


def build_writing_tools(reviewer, task_store=None) -> list[BaseTool]:
    """Return writing tools bound to a LiteratureReviewer."""

    @tool
    async def draft_literature_review(
        topic: str, focus: str = "", venue: str = "", save_name: str = "",
        config: RunnableConfig = None,
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
        task_id = None
        if task_store is not None:
            task_id = await task_store.create(
                "lit_review_writer", topic, _channel(config)
            )
            await task_store.mark_running(task_id)
        try:
            result = await reviewer.draft(
                topic, focus=focus, venue=venue, save_name=save_name
            )
        except Exception as exc:  # noqa: BLE001
            if task_store is not None:
                await task_store.fail(task_id, str(exc), [])
            return f"[literature review could not be written: {exc}]"

        tex = Path(result["tex_path"]).name
        summary = (
            f"Wrote a LaTeX literature review with {result['n_refs']} references.\n"
            f"Saved: `{tex}`"
            + (f" (+ `{Path(result['bib_path']).name}`)" if result["bib_path"] else "")
            + f"\nRetrieve it with `!getfile lit_reviews/{tex}`."
        )
        if task_store is not None:
            await task_store.finish(
                task_id, summary,
                [{"type": "artifact", "tex": result["tex_path"],
                  "bib": result["bib_path"], "n_refs": result["n_refs"]}],
            )
        return summary

    return [draft_literature_review]
