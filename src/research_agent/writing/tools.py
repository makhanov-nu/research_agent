"""Agent tools for writing research artifacts (review, methodology, paper).

Each artifact is saved into the current project's folder
(`outputs/projects/<slug>/<kind>/`) and registered in the project's artifact
table, so the web frontend can list and read it. The paper writer gathers the
project's existing lit review + methodology (+ experiment results) as material.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from ..config import settings
from ..projects import resolve_project


def _channel(config: RunnableConfig | None) -> str | None:
    if not config:
        return None
    return (config.get("configurable") or {}).get("thread_id")


def _rel_to_outputs(path: str) -> str:
    """Path relative to the outputs dir (what `!getfile` expects)."""
    try:
        return str(Path(path).resolve().relative_to(Path(settings.output_dir).resolve()))
    except ValueError:
        return Path(path).name


async def _gather_material(projects, project, limit: int = 6000) -> str:
    """Read the project's lit review + methodology as material for the paper."""
    if projects is None or project is None or project.get("id") is None:
        return ""
    chunks: list[str] = []
    for kind, label in (("lit_review", "Related Work"), ("methodology", "Methodology")):
        rows = await projects.list_artifacts(project["id"], kind)
        if not rows:
            continue
        path = Path(settings.output_dir) / rows[0]["rel_path"]
        if path.exists():
            chunks.append(f"=== {label} (from project) ===\n{path.read_text()[:limit]}")
    # Experiment results, if any.
    exp_rows = await projects.list_artifacts(project["id"], "experiments")
    if exp_rows:
        notes = "\n".join(
            f"- {r['title']}: {r.get('meta', {})}" for r in exp_rows[:10]
        )
        chunks.append(f"=== Experiment results (from project) ===\n{notes}")
    return "\n\n".join(chunks)


def build_writing_tools(writers, task_store=None, projects=None) -> list[BaseTool]:
    """Return writing tools bound to a `Writers` bundle and the project store."""

    async def _run(agent: str, kind: str, input_text: str, draft_coro_fn, config) -> str:
        project = await resolve_project(projects, config)
        dirpath = None
        if projects is not None and project is not None:
            dirpath = projects.kind_dir(project["slug"], kind)

        task_id = None
        if task_store is not None:
            task_id = await task_store.create(agent, input_text, _channel(config))
            await task_store.mark_running(task_id)
        try:
            result = await draft_coro_fn(dirpath)
        except Exception as exc:  # noqa: BLE001
            if task_store is not None:
                await task_store.fail(task_id, str(exc), [])
            return f"[{agent} could not complete: {exc}]"

        rel = _rel_to_outputs(result["tex_path"])
        if projects is not None and project is not None and project.get("id"):
            await projects.add_artifact(
                project["id"], kind, Path(result["tex_path"]).stem, rel,
                {"n_refs": result["n_refs"], "bib": _rel_to_outputs(result["bib_path"]) if result["bib_path"] else ""},
            )
        missing = result.get("missing_citations") or []
        warn = (
            f"\n⚠ {len(missing)} citation(s) with no BibTeX entry: "
            f"{', '.join(missing[:8])}{'…' if len(missing) > 8 else ''}"
            if missing else ""
        )
        summary = (
            f"Wrote a LaTeX {agent.replace('_', ' ')} with {result['n_refs']} "
            f"references"
            + (f" (project: {project['name']})" if project else "")
            + f".\nSaved: `{Path(result['tex_path']).name}` — retrieve with `!getfile {rel}`."
            + warn
        )
        if task_store is not None:
            # Persist the subagent's full reasoning/tool-call trace, with the saved
            # artifact (and any dangling citations) appended as a final step.
            trace = (result.get("trace") or []) + [
                {"type": "artifact", "tex": result["tex_path"],
                 "bib": result["bib_path"], "n_refs": result["n_refs"],
                 "missing_citations": missing}
            ]
            await task_store.finish(task_id, summary, trace)
        return summary

    @tool
    async def draft_literature_review(
        topic: str, focus: str = "", venue: str = "", save_name: str = "",
        config: RunnableConfig = None,
    ) -> str:
        """Research the literature on a topic and write a LaTeX Related Work section.

        Saves the thematically-organized review (with \\cite keys) + a BibTeX file
        into the project's lit_review folder.

        Args:
            topic: The subject of the review.
            focus: Optional angle to emphasize.
            venue: Optional target venue/style.
            save_name: Optional base filename; defaults to a slug of the topic.
        """
        return await _run(
            "literature_review", "lit_review", topic,
            lambda d: writers.reviewer.draft(
                topic, focus=focus, venue=venue, save_name=save_name, dirpath=d
            ),
            config,
        )

    @tool
    async def design_methodology(
        idea: str, constraints: str = "", venue: str = "", save_name: str = "",
        config: RunnableConfig = None,
    ) -> str:
        """Design a rigorous methodology for a research idea and write it in LaTeX.

        Saves a LaTeX \\section{Methodology} (+ BibTeX) into the project's
        methodology folder, grounded in the literature and cited.

        Args:
            idea: The research idea/contribution to design around.
            constraints: Optional resources/limits (compute, data, time, models).
            venue: Optional target venue/style.
            save_name: Optional base filename; defaults to a slug of the idea.
        """
        return await _run(
            "methodology", "methodology", idea,
            lambda d: writers.methodologist.draft(
                idea, constraints=constraints, venue=venue, save_name=save_name, dirpath=d
            ),
            config,
        )

    @tool
    async def draft_paper(
        brief: str, material: str = "", sections: str = "", venue: str = "",
        save_name: str = "", config: RunnableConfig = None,
    ) -> str:
        """Draft a research paper (or sections) in LaTeX from the project's material.

        If `material` is empty, the project's existing lit review, methodology, and
        experiment results are gathered automatically. Saves into the project's
        paper folder; inserts TODOs rather than fabricating results/citations.

        Args:
            brief: What to write and the framing (the contribution/story).
            material: Extra material to use; leave empty to auto-gather from the project.
            sections: Optional specific sections (e.g. "Intro, Method"); omit for full draft.
            venue: Optional target venue/style.
            save_name: Optional base filename; defaults to a slug of the brief.
        """
        project = await resolve_project(projects, config)
        gathered = material or await _gather_material(projects, project)
        return await _run(
            "paper_draft", "paper", brief,
            lambda d: writers.paper_writer.draft(
                brief, material=gathered, sections=sections, venue=venue,
                save_name=save_name, dirpath=d,
            ),
            config,
        )

    return [draft_literature_review, design_methodology, draft_paper]
