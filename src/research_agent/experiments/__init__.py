"""Experiment runner: dispatch experiments to a compute node and track them.

Phase 1 dispatches to a single GPU node over SSH + Docker (registered via
config). The ComputeBackend abstraction keeps a worker API / HF Jobs pluggable
later. See docs/experiment-runner.md.
"""
