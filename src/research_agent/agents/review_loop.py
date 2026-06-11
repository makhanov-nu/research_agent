"""Shared draft → critique → revise loop for research artifact writers.

Every artifact writer (literature review, methodology, paper draft) follows the
same quality-improvement pattern: draft, critique, optionally revise and
re-critique.  Factoring it here avoids repeating the bounded-loop logic and
ensures that rejected drafts, critiques, and accepted revisions are always
appended to the task trace in a consistent shape — the data we need for future
preference-pair fine-tuning.

Critiques are pluggable: a caller supplies an async ``critique_fn`` that takes
(original_task, draft_dict) and returns (is_valid: bool, feedback: str).  The
loop is bounded by ``settings.validation_rounds`` (total critique passes; 2
means at most one revision; <=1 means no revision at all).
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from ..config import settings

logger = logging.getLogger(__name__)

# Type aliases for clarity.
DraftFn = Callable[..., Awaitable[dict]]          # async () -> draft dict
CritiqueFn = Callable[[str, dict], Awaitable[tuple[bool, str]]]  # (task, draft) -> (valid, feedback)

_REVISION_PREFIX = (
    "REVISION REQUEST — a reviewer found the following issues in the previous "
    "draft. Fix them while preserving everything else:\n\n"
    "{feedback}"
    "\n\nORIGINAL TASK:\n{task}"
)


async def run_review_loop(
    *,
    original_task: str,
    draft_fn: DraftFn,
    critique_fn: CritiqueFn,
    trace: list,
    rounds: int | None = None,
) -> dict:
    """Run draft → critique → (revise → critique)* and return the accepted draft.

    Args:
        original_task: The task string the writer was originally given.
        draft_fn: Async callable that accepts a ``task`` keyword argument and
            returns a writer result dict (keys: latex, tex_path, bib_path,
            n_refs, missing_citations, trace).  Called with the original task
            on round 1 and a revision-prefixed task on subsequent rounds.
        critique_fn: Async callable that receives ``(original_task, draft_dict)``
            and returns ``(is_valid: bool, feedback: str)``.  Should be
            side-effect-free (no task_store interaction — the caller handles
            that for per-verifier tracking).
        trace: Mutable list; critique entries are *appended* in place so the
            caller's full_trace accumulates them.
        rounds: Override for settings.validation_rounds (used in tests).

    Returns:
        The accepted draft dict (possibly the original if already valid, or if
        all rounds are exhausted, or if the critique raised an exception).
    """
    max_rounds = rounds if rounds is not None else settings.validation_rounds
    # Round 1 produces the initial draft; after a failed critique, draft_fn is
    # re-invoked with the reviewer's feedback prepended to the task.
    current_draft = await draft_fn(task=original_task)

    for round_num in range(1, max_rounds + 1):
        try:
            is_valid, feedback = await critique_fn(original_task, current_draft)
        except Exception:  # noqa: BLE001 — never kill the job over a verifier crash
            logger.exception(
                "Critique round %d failed; accepting current draft", round_num
            )
            trace.append({
                "type": "critique",
                "round": round_num,
                "verifier": _verifier_name(critique_fn),
                "verdict": "error",
                "feedback": "Verifier raised an exception; draft accepted.",
                "superseded_draft": None,
            })
            break

        if is_valid:
            trace.append({
                "type": "critique",
                "round": round_num,
                "verifier": _verifier_name(critique_fn),
                "verdict": "valid",
                "feedback": feedback,
                "superseded_draft": None,
            })
            break

        # Issues found — record the rejected draft + critique, then revise
        # unless we are already at the last allowed round.
        rejected_latex = current_draft.get("latex", "")
        trace.append({
            "type": "critique",
            "round": round_num,
            "verifier": _verifier_name(critique_fn),
            "verdict": "invalid",
            "feedback": feedback,
            "superseded_draft": rejected_latex,
        })

        if round_num >= max_rounds:
            # Rounds exhausted — return the current (imperfect) draft.
            logger.info(
                "Critique round %d/%d: issues remain but round cap reached; "
                "accepting current draft.", round_num, max_rounds,
            )
            break

        # Re-draft with the feedback prepended.
        revision_task = _REVISION_PREFIX.format(
            feedback=feedback, task=original_task
        )
        logger.info("Critique round %d: revising draft.", round_num)
        try:
            current_draft = await draft_fn(task=revision_task)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Revision draft call failed on round %d; keeping previous draft",
                round_num,
            )
            break

    return current_draft


def _verifier_name(fn) -> str:
    """Best-effort: pull a human-readable name from the critique callable."""
    return getattr(fn, "__verifier_name__", None) or getattr(fn, "__name__", "unknown")


async def citation_critique(original_task: str, draft: dict) -> tuple[bool, str]:
    """Rule-based citation critique — no LLM needed.

    Returns (is_valid, feedback).  Issues only if missing_citations is
    non-empty.  Instructs the writer to define or remove the dangling keys.
    """
    missing = draft.get("missing_citations") or []
    if not missing:
        return True, ""
    keys = ", ".join(f"\\cite{{{k}}}" for k in missing[:20])
    feedback = (
        f"The draft contains {len(missing)} undefined BibTeX key(s): {keys}. "
        "For each key, either add a proper @article/@inproceedings/... entry to "
        "the ```bibtex block, or remove the \\cite{} command and replace it with "
        "an inline description. Do not invent BibTeX entries."
    )
    return False, feedback


# Tag citation_critique so trace entries identify it by name.
citation_critique.__verifier_name__ = "citation_check"
