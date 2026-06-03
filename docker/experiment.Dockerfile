# Universal experiment image.
#
# Baked once per GPU box (or pulled from a registry) and reused for every
# experiment, so jobs don't reinstall the ML stack on each run. The runner builds
# this from `experiments/image.build_experiment_dockerfile()` over SSH (stdin, no
# build context); this file is the human-readable mirror — keep them in sync.
#
# Build manually:  docker build -t research-agent/experiment:latest \
#                    -f docker/experiment.Dockerfile docker/
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    "datasets>=2.19" \
    "transformers>=4.41" \
    "accelerate>=0.30" \
    "evaluate>=0.4" \
    "optuna>=3.6" \
    "mlflow>=2.16" \
    "scikit-learn>=1.4" \
    "scipy>=1.11" \
    "pandas>=2.0" \
    "numpy>=1.26" \
    "sentencepiece>=0.2" \
    "tqdm>=4.66"
