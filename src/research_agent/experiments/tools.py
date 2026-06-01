"""LangChain tools that let the agent drive experiments.

Built as closures over an ExperimentRunner. The current Discord channel is read
from the injected RunnableConfig (its thread_id), so proposed experiments are
tagged to the channel that will receive their reports — without exposing the
channel as a model-visible argument.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool


def _channel(config: RunnableConfig | None) -> str | None:
    if not config:
        return None
    return (config.get("configurable") or {}).get("thread_id")


def build_experiment_tools(runner) -> list[BaseTool]:
    """Return the experiment tools bound to `runner`."""

    @tool
    async def propose_experiment(
        title: str, hypothesis: str, plan: str, config: RunnableConfig = None
    ) -> str:
        """Register a new experiment (title, hypothesis, and a short plan).

        Returns the experiment id to use with the other experiment tools.
        """
        exp_id = await runner.propose(title, hypothesis, plan, _channel(config))
        return f"Created experiment #{exp_id}: {title}"

    @tool
    async def write_experiment_code(experiment_id: int, files: dict[str, str]) -> str:
        """Write code into an experiment's workspace.

        `files` maps relative path -> file content (e.g. {"train.py": "...",
        "requirements.txt": "..."}). The workspace is shipped to the compute
        node at launch and mounted at /workspace; outputs go to /output.
        """
        written = await runner.write_code(experiment_id, files)
        return f"Wrote {len(written)} file(s) to experiment #{experiment_id}: {', '.join(written)}"

    @tool
    async def launch_experiment(
        experiment_id: int, command: str, image: str = "",
        gpus: str = "", memory: str = "", time_limit_minutes: int = 0,
    ) -> str:
        """Request launch of an experiment on the compute node.

        `command` is the entrypoint run inside /workspace (e.g.
        "python train.py --epochs 3"). Leave `image`/`gpus` empty to use the
        configured defaults. Launches require human approval in Discord unless
        approval is disabled.
        """
        return await runner.request_launch(
            experiment_id, command=command, image=image,
            gpus=(gpus or None), memory=memory, time_limit_minutes=time_limit_minutes,
        )

    @tool
    async def experiment_status(experiment_id: int) -> str:
        """Get an experiment's current status and latest metrics."""
        return await runner.get_status(experiment_id)

    @tool
    async def experiment_logs(experiment_id: int, tail: int = 200) -> str:
        """Fetch the last `tail` lines of an experiment's container logs."""
        return await runner.get_logs(experiment_id, tail=tail)

    @tool
    async def cancel_experiment(experiment_id: int) -> str:
        """Cancel a running experiment."""
        return await runner.cancel(experiment_id)

    return [
        propose_experiment,
        write_experiment_code,
        launch_experiment,
        experiment_status,
        experiment_logs,
        cancel_experiment,
    ]
