"""Turn logged task trajectories into training data for distilling small models.

Every completed task is a labeled example: ``(agent, input, result, trace,
quality, feedback)``. This package exports them as per-role JSONL datasets
(OpenAI chat format) you can fine-tune a small model on — the practical path to
the LoRA-per-role setup (one cheap specialist per subagent kind).
"""

from .export import export_dataset, system_prompt_for, task_to_example

__all__ = ["export_dataset", "system_prompt_for", "task_to_example"]
