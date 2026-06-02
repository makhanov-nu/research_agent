"""Specialized subagents and the orchestrator's delegation tools.

The Discord-facing agent is an orchestrator: its tools delegate self-contained
jobs to specialized subagents (each its own ReAct loop with its own tools and
context). The orchestrator receives only each subagent's final output, keeping
its context lean. Add new agents by writing a builder and registering its
delegation tool in registry.py.
"""

from .registry import build_delegated_tools

__all__ = ["build_delegated_tools"]
