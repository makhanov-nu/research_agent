"""Agent tools for writing research artifacts (review, methodology, paper)."""

from __future__ import annotations

from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool


def _channel(config: RunnableConfig | None) -> str | None:
    if not config:
        return None
    return (config.get("configurable") or {}).get("thread_id")


async def _record(
    task_store, agent: str, input_text: str, config, coro, subdir: str
) -> str:
    """Run a writer coroutine, persist the task + artifact, return a summary."""
    task_id = None
    if task_store is not None:
        task_id = await task_store.create(agent, input_text, _channel(config))
        await task_store.mark_running(task_id)
    try:
        result = await coro
    except Exception as exc:  # noqa: BLE001 — report failure up, record it
        if task_store is not None:
            await task_store.fail(task_id, str(exc), [])
        return f"[{agent} could not complete: {exc}]"

    tex = Path(result["tex_path"]).name
    summary = (
        f"Wrote a LaTeX {agent.replace('_', ' ')} with {result['n_refs']} references.\n"
        f"Saved: `{tex}`"
        + (f" (+ `{Path(result['bib_path']).name}`)" if result["bib_path"] else "")
        + f"\nRetrieve it with `!getfile {subdir}/{tex}`."
    )
    if task_store is not None:
        await task_store.finish(
            task_id, summary,
            [{"type": "artifact", "tex": result["tex_path"],
              "bib": result["bib_path"], "n_refs": result["n_refs"]}],
        )
    return summary


def build_writing_tools(writers, task_store=None) -> list[BaseTool]:
    """Return writing tools bound to a `Writers` bundle."""

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
        return await _record(
            task_store, "literature_review", topic, config,
            writers.reviewer.draft(topic, focus=focus, venue=venue, save_name=save_name),
            "lit_reviews",
        )

    @tool
    async def design_methodology(
        idea: str, constraints: str = "", venue: str = "", save_name: str = "",
        config: RunnableConfig = None,
    ) -> str:
        """Design a rigorous methodology for a research idea and write it in LaTeX.

        Produces a LaTeX \\section{Methodology} covering research questions,
        approach, data/models, experimental design, baselines/ablations, metrics
        and protocol, reproducibility, and threats to validity — grounded in the
        literature (standard baselines/datasets/metrics) and cited. Saves the
        .tex (+ .bib) to disk.

        Args:
            idea: The research idea, question, or contribution to design around.
            constraints: Optional resources/limits (compute, data, time, models).
            venue: Optional target venue/style.
            save_name: Optional base filename; defaults to a slug of the idea.
        """
        return await _record(
            task_store, "methodology", idea, config,
            writers.methodologist.draft(
                idea, constraints=constraints, venue=venue, save_name=save_name
            ),
            "methodology",
        )

    @tool
    async def draft_paper(
        brief: str, material: str = "", sections: str = "", venue: str = "",
        save_name: str = "", config: RunnableConfig = None,
    ) -> str:
        """Draft a research paper (or specific sections) in LaTeX from given material.

        Composes submission-style LaTeX prose, weaving together the supplied
        contribution, methodology, related work, and findings. Does not fabricate
        results or citations; inserts clearly-marked TODOs for gaps. Saves the
        .tex (+ .bib) to disk.

        Args:
            brief: What to write and the framing (the contribution/story).
            material: Supporting content to use (methodology, findings, related
                work) — paste what you already have so it isn't re-derived.
            sections: Optional specific sections to write (e.g. "Intro, Method");
                omit for a full draft.
            venue: Optional target venue/style.
            save_name: Optional base filename; defaults to a slug of the brief.
        """
        return await _record(
            task_store, "paper_draft", brief, config,
            writers.paper_writer.draft(
                brief, material=material, sections=sections, venue=venue,
                save_name=save_name,
            ),
            "papers",
        )

    return [draft_literature_review, design_methodology, draft_paper]
