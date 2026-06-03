"""The universal experiment image.

Rather than `pip install` the whole ML stack on every run, we bake one image
with the common research libraries (torch is already in the CUDA base) and reuse
it for every experiment. It's built once per GPU box at provision time (or pulled
if `experiment_image` points at a registry). Jobs add only their *extra* deps via
a small requirements.txt.

The Dockerfile has no build context (only FROM + RUN), so it builds straight from
stdin over SSH (`docker build -t <tag> -`) — no file upload needed.
"""

from __future__ import annotations

# Libraries every experiment can assume are present (kept broad but not huge).
_PINNED_PACKAGES = [
    "datasets>=2.19",
    "transformers>=4.41",
    "accelerate>=0.30",
    "evaluate>=0.4",
    "optuna>=3.6",
    "mlflow>=2.16",
    "scikit-learn>=1.4",
    "scipy>=1.11",
    "pandas>=2.0",
    "numpy>=1.26",
    "sentencepiece>=0.2",
    "tqdm>=4.66",
]


def build_experiment_dockerfile(base_image: str) -> str:
    """Return the universal experiment Dockerfile (FROM `base_image` + the stack)."""
    pkgs = " ".join(f'"{p}"' for p in _PINNED_PACKAGES)
    return (
        f"FROM {base_image}\n"
        "ENV PIP_NO_CACHE_DIR=1 \\\n"
        "    HF_HOME=/root/.cache/huggingface \\\n"
        "    PYTHONUNBUFFERED=1\n"
        f"RUN pip install --no-cache-dir {pkgs}\n"
    )
