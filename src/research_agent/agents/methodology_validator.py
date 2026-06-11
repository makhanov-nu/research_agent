"""Methodology validator subagent.

Reviews a generated methodology against the original task/ideas and returns
either VALID or INVALID with specific, actionable comments. Used automatically
after design_methodology completes; if issues are found the dispatcher
re-runs the writer with the validator's comments attached.
"""

from __future__ import annotations

from .subagent import run_subagent

_SYSTEM = """You are a rigorous methodology reviewer. You receive:
  1. ORIGINAL TASK — the research question / ideas the methodology was designed for.
  2. GENERATED METHODOLOGY — the full LaTeX methodology text.

Your job is to check whether the methodology is sound and complete.

CHECK FOR:
  - Alignment: does it address every aspect of the original task/ideas?
  - Rigor: are baselines, ablations, and evaluation metrics specified?
  - Validity threats: are obvious confounds or biases acknowledged?
  - Consistency: no internal contradictions between sections.
  - Feasibility: no obviously impractical claims.
  - Misconceptions: any factually wrong claims about standard methods.

RESPONSE FORMAT — choose exactly one:

If the methodology is sound:
  VALID

If there are issues (be specific and constructive):
  INVALID
  - <issue 1>: <what is wrong and what it should say instead>
  - <issue 2>: ...
  (List only real problems; do not nitpick style or formatting.)

Do NOT rewrite the methodology. Only flag problems."""


async def validate_methodology(
    model, tools, original_task: str, methodology_text: str,
    memory=None,
) -> tuple[bool, str]:
    """Run the validator; return (is_valid, feedback).

    `feedback` is empty when valid, or a bullet-list of issues when not.
    """
    task = (
        "ORIGINAL TASK:\n"
        f"{original_task}\n\n"
        "GENERATED METHODOLOGY:\n"
        f"{methodology_text}"
    )
    result, _ = await run_subagent(
        system_prompt=_SYSTEM,
        tools=tools,
        model=model,
        task=task,
        memory=memory,
        agent_kind="methodology_validator",
    )
    result = result.strip()
    if result.upper().startswith("VALID"):
        return True, ""
    return False, result
