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


def build_experiment_tools(runner, coder=None) -> list[BaseTool]:
    """Return the experiment tools bound to `runner` (and an optional `coder`)."""

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
    async def author_experiment_code(experiment_id: int, spec: str) -> str:
        """Have the Codex coder write the experiment's code from a spec.

        Pass a detailed spec (the methodology / technical specification: task,
        datasets, model, search space, metrics). The coder writes a runnable
        train.py (Optuna HPO + HuggingFace data + MLflow logging + a
        /output/metrics.jsonl summary) plus requirements.txt into the workspace.
        """
        if coder is None:
            return "No coder model is configured (set OPENROUTER_API_KEY)."
        # Inject lessons from past experiments (esp. failures on similar tasks).
        prompt = spec
        memory = getattr(runner, "memory", None)
        if memory is not None:
            lessons = await memory.recall_lessons(spec)
            if lessons:
                prompt = (
                    f"{spec}\n\n=== Lessons from past experiments (avoid repeating "
                    f"these mistakes) ===\n{lessons}"
                )
        try:
            files = await coder.author(prompt)
        except Exception as exc:  # noqa: BLE001
            return f"[coder failed to author the experiment: {exc}]"
        written = await runner.write_code(experiment_id, files)
        return (
            f"Codex authored {len(written)} file(s) for experiment "
            f"#{experiment_id}: {', '.join(written)}. Review, then launch."
        )

    @tool
    async def experiment_mlflow(experiment_id: int) -> str:
        """Get the experiment's MLflow run: params + latest/best metrics."""
        return await runner.mlflow_summary(experiment_id)

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
        author_experiment_code,
        write_experiment_code,
        launch_experiment,
        experiment_status,
        experiment_logs,
        experiment_mlflow,
        cancel_experiment,
    ]
