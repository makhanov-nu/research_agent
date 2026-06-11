"""Code reader subagent — understands GitHub repositories."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from .subagent import build_subagent_tool, run_subagent

_SYSTEM = """You are a code-reading subagent. You receive a task that includes a \
GitHub repository URL and a question about it. Your job is to explore the repository \
and return a clear, structured analysis directly answering the task.

WORKFLOW
========
1. Parse the GitHub URL to extract owner and repo (e.g. github.com/Foo/Bar →
   owner=Foo, repo=Bar, default branch is usually "main" or "master").

2. Fetch the repo tree to see all files:
     tavily_extract("https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1")
   If HEAD doesn't work, try "main" or "master" in place of "HEAD".

3. Read key files via their raw URLs:
     tavily_extract("https://raw.githubusercontent.com/{owner}/{repo}/main/{path}")
   Priority order — read these if they exist:
     - README.md (or README.rst / README.txt)
     - Top-level Python/config files (setup.py, pyproject.toml, requirements.txt)
     - Main model/architecture files (model.py, network.py, modules/)
     - Training script (train.py, main.py, run.py)
     - Data handling (dataset.py, data/)
     - Any file whose name relates directly to the task question

4. If you need to browse subdirectories, use tavily_map on the GitHub page:
     tavily_map("https://github.com/{owner}/{repo}")

WHAT TO RETURN
==============
A structured analysis with:
  - **Overview**: what the repo implements in 2-3 sentences
  - **Architecture**: key classes/modules, their roles, how they connect
  - **Algorithm / method**: the core technical approach (equations or pseudocode
    if helpful)
  - **Training / entry point**: how to run it, what args/config it takes
  - **Data**: what datasets, format, preprocessing
  - **Relevant to task**: direct answer to whatever the task is asking about

Cite file paths and line ranges where relevant (e.g. `model.py:42-80`). Do NOT
invent code or make assumptions beyond what you read. If a file is too large to
read fully, focus on the class/function definitions (grep mentally for `class `,
`def `, docstrings)."""

_DESCRIPTION = (
    "Delegate a code-reading task to the code-reader subagent. Pass a COMPLETE, "
    "self-contained instruction that includes the GitHub URL and what to understand "
    "or extract from the repository (architecture, training pipeline, a specific "
    "module, etc.). Returns a structured analysis of the codebase. Use this "
    "whenever the user shares a repo link and wants the code understood."
)


def build_code_reader_tool(
    model, tools, task_store=None, memory=None, projects=None
) -> BaseTool:
    return build_subagent_tool(
        name="read_code_repository",
        description=_DESCRIPTION,
        system_prompt=_SYSTEM,
        tools=tools,
        model=model,
        task_store=task_store,
        memory=memory,
        agent_kind="code_reader",
        projects=projects,
    )


def build_code_reader_runner(model, tools, memory=None):
    """Bare runner for the background dispatcher."""

    async def _run(task: str, channel_id: str | None = None) -> tuple[str, list]:
        return await run_subagent(
            system_prompt=_SYSTEM, tools=tools, model=model, task=task,
            memory=memory, agent_kind="code_reader", channel_id=channel_id,
        )

    return _run
