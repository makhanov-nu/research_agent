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
sentencepiece, tqdm. Do NOT reinstall these. Only add a `requirements.txt` if you \
import something NOT in that list (keep it minimal); otherwise omit it.

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


class ExperimentCoder:
    def __init__(self, llm):
        self.llm = llm

    async def author(self, spec: str) -> dict[str, str]:
        """Generate experiment files from a spec. Returns {path: content}."""
        from langchain_core.messages import HumanMessage, SystemMessage

        resp = await self.llm.ainvoke(
            [SystemMessage(content=_SYSTEM), HumanMessage(content=spec)]
        )
        content = resp.content
        if isinstance(content, list):
            content = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        files = parse_files(content)
        if not files:
            # Fall back to treating the whole response as a single train.py.
            files = {"train.py": content.strip() + "\n"}
        logger.info("Coder authored %d file(s): %s", len(files), ", ".join(files))
        return files
