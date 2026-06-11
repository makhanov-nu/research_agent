"""Assemble the orchestrator's delegation tools from available resources.

This is the single place to register subagents. To add a new specialized agent,
write its builder and append its delegation tool here.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from pathlib import Path

from langchain_core.tools import tool

from ..writing.tools import build_writing_tools
from .code_reader import build_code_reader_tool
from .consortium_tool import build_consortium_tool
from .literature import build_literature_agent_tool


def _build_review_methodology_tool(llm, mcp_tools, output_dir: str, memory=None) -> BaseTool:
    """Validate an existing methodology file without regenerating it."""
    base = Path(output_dir).resolve()

    @tool("review_methodology",
          description=(
              "Run the methodology validator on an EXISTING methodology file. "
              "Use this when the researcher wants to check or critique a methodology "
              "that has already been written — do NOT call design_methodology for this. "
              "Provide the file path (relative to the output dir, e.g. "
              "'projects/my-project/methodology/design.tex') AND the original research "
              "task/idea the methodology was designed for. Returns VALID or a specific "
              "bullet list of issues."
          ))
    async def review_methodology(path: str, original_task: str) -> str:
        try:
            target = (base / path).resolve()
            target.relative_to(base)
        except (ValueError, Exception):
            return f"[review_methodology] Path rejected (must be inside {output_dir}): {path}"
        if not target.exists():
            return f"[review_methodology] File not found: {path}"
        try:
            methodology_text = target.read_text(errors="replace")
        except Exception as exc:
            return f"[review_methodology] Could not read {path}: {exc}"

        from .methodology_validator import validate_methodology
        try:
            is_valid, feedback = await validate_methodology(
                llm, mcp_tools, original_task, methodology_text, memory=memory
            )
        except Exception as exc:
            return f"[review_methodology] Validator error: {exc}"

        if is_valid:
            return "VALID — the methodology is sound and addresses the original task."
        return f"INVALID — the validator found issues:\n{feedback}"

    return review_methodology


def _build_read_artifact_tool(output_dir: str) -> BaseTool:
    base = Path(output_dir).resolve()

    @tool("read_project_artifact",
          description=(
              "Read a project artifact (LaTeX file, notes, etc.) that was previously "
              "saved to disk. Pass the relative path returned by a writing tool, e.g. "
              "'projects/my-project/methodology/design.tex'. Returns the file contents "
              "so you can use them as input for the next stage. Do NOT delegate this to "
              "research_literature — use this tool directly."
          ))
    def read_project_artifact(path: str) -> str:
        # Resolve against the output dir; reject traversals outside it.
        try:
            target = (base / path).resolve()
            target.relative_to(base)
        except (ValueError, Exception):
            return f"[read_project_artifact] Path rejected (must be inside {output_dir}): {path}"
        if not target.exists():
            return f"[read_project_artifact] File not found: {path}"
        try:
            text = target.read_text(errors="replace")
            if len(text) > 24_000:
                text = text[:24_000] + f"\n…[truncated; file is {len(text)} chars total]"
            return text
        except Exception as exc:
            return f"[read_project_artifact] Could not read {path}: {exc}"

    return read_project_artifact


def build_delegated_tools(
    *, llm, mcp_tools, writers, experiment_runner=None, consortium=None,
    task_store=None, projects=None, memory=None, output_dir: str = "outputs",
) -> list[BaseTool]:
    tools: list[BaseTool] = []

    # Direct file reader — so the orchestrator never delegates "read this file"
    # to research_literature or any other subagent.
    tools.append(_build_read_artifact_tool(output_dir))

    # Literature research subagent (owns the paperclip/MCP tools so the
    # orchestrator never sees raw search output). `memory` makes it learn.
    if mcp_tools:
        tools.append(
            build_literature_agent_tool(llm, mcp_tools, task_store, memory, projects)
        )
        # Code reader subagent: fetches and analyses GitHub repositories.
        tools.append(
            build_code_reader_tool(llm, mcp_tools, task_store, memory, projects)
        )
        # Standalone methodology reviewer — validates an existing file without
        # regenerating it. Distinct from design_methodology which writes from scratch.
        tools.append(
            _build_review_methodology_tool(llm, mcp_tools, output_dir, memory=memory)
        )

    # LaTeX writers: literature review, methodology, paper draft (each a subagent).
    tools += build_writing_tools(
        writers, task_store=task_store, projects=projects, memory=memory
    )

    # Multi-model ideation consortium.
    if consortium is not None:
        tools.append(
            build_consortium_tool(
                consortium, task_store=task_store, projects=projects, memory=memory
            )
        )

    # Experiment tools (lightweight, return concise status strings).
    if experiment_runner is not None and getattr(experiment_runner, "enabled", False):
        from ..config import settings
        from ..experiments.coder import ExperimentCoder
        from ..experiments.tools import build_experiment_tools

        coder = None
        if settings.openrouter_api_key:
            from ..llm import build_openrouter_chat

            coder = ExperimentCoder(
                build_openrouter_chat(
                    settings.experiment_coder_model, temperature=0.2, max_tokens=16384
                )
            )
        tools += build_experiment_tools(experiment_runner, coder=coder)

    return tools
