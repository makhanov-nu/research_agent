"""The 'Codex' experiment-writer: turns a spec into runnable experiment code.

Given a methodology/technical spec, a coder model (OpenRouter, e.g.
`openai/gpt-5.5`) writes the files for a self-contained experiment — training
code with Optuna HPO, HuggingFace dataset loading, MLflow logging, and a
`/output/metrics.jsonl` summary the poller reads. The model emits files in a
simple delimited format that `parse_files` turns into a path->content mapping;
the runner writes them into the experiment workspace.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

FILE_MARKER = re.compile(r"^===\s*FILE:\s*(.+?)\s*===\s*$", re.MULTILINE)

_SYSTEM = """You are an expert ML research engineer (a "Codex"). You are given a \
technical specification for an experiment and must write COMPLETE, runnable code \
to execute it on a single-GPU Linux box inside a Docker container. The container \
mounts the code read-only at /workspace (your cwd) and a writable /output.

Hard requirements for the code you write:
1. Entry point `train.py`, runnable as `python train.py` with sane CLI defaults \
(argparse). No notebooks, no placeholders, no TODOs — it must actually run.
2. Data: load datasets via HuggingFace `datasets` (the HF cache is persistent and \
HF_TOKEN is in the environment). Don't hard-code absolute local paths.
3. Hyperparameter search: use Optuna (define-by-run `study.optimize`). Make the \
search space and `n_trials` configurable via argparse; keep defaults small.
4. Tracking: use MLflow. The env already sets MLFLOW_TRACKING_URI, \
MLFLOW_EXPERIMENT_NAME and MLFLOW_RUN_NAME — call \
`mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])`, start a PARENT run \
named by MLFLOW_RUN_NAME, and log each Optuna trial as a NESTED run (params + \
metrics). Log the best value/params to the parent run.
5. Summary: at the end, append ONE json object per line to /output/metrics.jsonl \
with the headline metrics (include `best_value` and key `best_params`), and save \
any model/plots under /output.
6. The runtime image ALREADY has: torch + CUDA, datasets, transformers, \
accelerate, evaluate, optuna, mlflow, scikit-learn, scipy, pandas, numpy, \
sentencepiece, tqdm, einops; for LLM fine-tuning: peft, trl, bitsandbytes; for \
computer vision: timm, albumentations, opencv-python-headless. Do NOT reinstall \
these. Only add a `requirements.txt` for something NOT listed (e.g. flash-attn); \
otherwise omit it.

Output format — emit each file as:
=== FILE: <relative/path> ===
<full file contents>

Emit `train.py` and `requirements.txt` at minimum. Output ONLY the files in that \
format, nothing else.
"""


def parse_files(text: str) -> dict[str, str]:
    """Parse '=== FILE: path ===' delimited output into {path: content}."""
    files: dict[str, str] = {}
    parts = FILE_MARKER.split(text)
    # parts = [preamble, path1, body1, path2, body2, ...]
    for i in range(1, len(parts) - 1, 2):
        path = parts[i].strip()
        body = parts[i + 1].strip("\n")
        if path:
            files[path] = body + "\n"
    return files


_REVISE_SYSTEM = """You are an expert ML research engineer (a "Codex"). A previous \
run of an experiment FAILED. Your job is to read the original spec, the current \
code files, the failure logs, and (optionally) lessons from past similar failures, \
then produce a COMPLETE fixed file set.

Rules:
- Return the FULL content of every file, not a diff — even files you did not change.
- Only change what is necessary to fix the failure.
- Do NOT change the experiment objective, dataset, model architecture, or resource \
requirements — only fix code bugs, import errors, API mismatches, missing arguments, \
or similar mechanical issues visible in the logs.
- Keep the same entry point (`train.py`) and the same output contract \
(/output/metrics.jsonl).

Output format — emit each file as:
=== FILE: <relative/path> ===
<full file contents>

Output ONLY the files in that format, nothing else.
"""


def _extract_content(resp) -> str:
    """Pull the text out of a LangChain ChatMessage (handles list-of-blocks)."""
    content = resp.content
    if isinstance(content, list):
        content = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return content


class ExperimentCoder:
    def __init__(self, llm):
        self.llm = llm

    async def author(self, spec: str) -> dict[str, str]:
        """Generate experiment files from a spec. Returns {path: content}."""
        from langchain_core.messages import HumanMessage, SystemMessage

        resp = await self.llm.ainvoke(
            [SystemMessage(content=_SYSTEM), HumanMessage(content=spec)]
        )
        content = _extract_content(resp)
        files = parse_files(content)
        if not files:
            # Fall back to treating the whole response as a single train.py.
            files = {"train.py": content.strip() + "\n"}
        logger.info("Coder authored %d file(s): %s", len(files), ", ".join(files))
        return files

    async def revise(
        self,
        spec: str,
        files: dict[str, str],
        logs: str,
        lessons: str = "",
    ) -> dict[str, str]:
        """Patch an existing file set to fix a failed run.

        Given the original spec, the current workspace files, the tail of the
        failure logs, and optional lessons from past failures, the coder returns
        a COMPLETE fixed file set (same contract as author()).

        Returns {path: content} — the caller should write these over the
        existing workspace files using runner.write_code().
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # Build a human turn that includes every piece of context the model needs.
        file_block = "\n\n".join(
            f"=== FILE: {path} ===\n{content}" for path, content in sorted(files.items())
        )
        lesson_section = (
            f"\n\n=== Lessons from past similar failures ===\n{lessons}" if lessons else ""
        )
        human_content = (
            f"Original spec:\n{spec}\n\n"
            f"=== Current files ===\n{file_block}\n\n"
            f"=== Failure logs (tail) ===\n{logs[-4000:]}"
            f"{lesson_section}\n\n"
            "Return the complete fixed file set."
        )

        resp = await self.llm.ainvoke(
            [SystemMessage(content=_REVISE_SYSTEM), HumanMessage(content=human_content)]
        )
        content = _extract_content(resp)
        fixed = parse_files(content)
        if not fixed:
            # If the model forgot the markers, treat whole response as train.py.
            fixed = {"train.py": content.strip() + "\n"}
        logger.info(
            "Coder revised %d file(s): %s", len(fixed), ", ".join(sorted(fixed))
        )
        return fixed


def build_default_coder() -> "ExperimentCoder | None":
    """Construct an ExperimentCoder from settings, or return None if unconfigured.

    Factored here so both agents/registry.py and ExperimentRunner can share the
    same construction logic without importing each other.
    """
    from ..config import settings

    if not settings.openrouter_api_key:
        return None
    from ..llm import build_openrouter_chat

    return ExperimentCoder(
        build_openrouter_chat(
            settings.experiment_coder_model, temperature=0.2, max_tokens=16384
        )
    )
