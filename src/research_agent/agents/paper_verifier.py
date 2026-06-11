"""Paper claims verifier subagent.

Given the original brief/material and the generated LaTeX, checks for
fabricated results or claims unsupported by the supplied material.  The paper
writer is instructed to insert TODOs instead of inventing numbers, so this
verifier looks for violations of that contract: specific numerical results,
tables, or bold empirical claims that cannot be traced back to the provided
material.

Used automatically after draft_paper completes.  Returns VALID or a bullet
list of issues; on issues, the dispatcher/review_loop triggers one revision
pass.  Recorded as its own task row ("paper_verifier") when task_store is
available, so it appears separately in the dashboard alongside the main draft
task — the same pattern as methodology_validator.
"""

from __future__ import annotations

from .subagent import run_subagent

_SYSTEM = """You are a research integrity reviewer for an AI paper-drafting \
assistant. You receive:
  1. ORIGINAL BRIEF — the research question / contribution the paper was asked to cover.
  2. SUPPORTING MATERIAL — the literature review, methodology, and experiment data \
that was supplied to the writer (may be empty, meaning the writer had no numbers).
  3. GENERATED LATEX — the paper draft produced by the writer.

Your job is to check whether the draft fabricates results.

The writer is ONLY allowed to use numerical results, tables, and quantitative \
comparisons that appear verbatim in SUPPORTING MATERIAL.  If SUPPORTING MATERIAL \
is empty or contains no numbers, the draft MUST contain TODO placeholders for \
all empirical claims instead of inventing values.

CHECK FOR:
  - Invented numbers: numerical results (accuracy, BLEU, loss, etc.) not present in \
SUPPORTING MATERIAL.
  - Invented citations: \\cite keys that do not correspond to papers mentioned in \
SUPPORTING MATERIAL (citation completeness is handled separately; flag only \
citations to results the writer couldn't have known).
  - Unjustified comparisons: statements like "outperforms X by Y%" with no basis \
in SUPPORTING MATERIAL.
  - Missing TODOs: places where a number is clearly needed but neither a TODO nor \
supporting data is present.

DO NOT flag:
  - Well-known background facts (e.g., "transformers use self-attention").
  - Standard dataset statistics that are matters of public record.
  - TODO markers — these are correct behavior.
  - Style, formatting, or structural issues.

RESPONSE FORMAT — choose exactly one:

If no fabrications are found:
  VALID

If fabrications exist (be specific: quote the offending phrase and explain why):
  INVALID
  - <offending phrase>: <reason it is unsupported / what should be a TODO>
  - ...
  (List only real fabrications; do not nitpick.)
"""


async def verify_paper(
    model, tools, brief: str, material: str, paper_text: str,
    memory=None,
) -> tuple[bool, str]:
    """Run the paper verifier; return (is_valid, feedback).

    ``feedback`` is empty when valid, or a bullet-list of issues when not.
    """
    task = (
        "ORIGINAL BRIEF:\n"
        f"{brief}\n\n"
        "SUPPORTING MATERIAL:\n"
        f"{material or '(none provided)'}\n\n"
        "GENERATED LATEX:\n"
        f"{paper_text}"
    )
    result, _ = await run_subagent(
        system_prompt=_SYSTEM,
        tools=tools,
        model=model,
        task=task,
        memory=memory,
        agent_kind="paper_verifier",
    )
    result = result.strip()
    if result.upper().startswith("VALID"):
        return True, ""
    return False, result
