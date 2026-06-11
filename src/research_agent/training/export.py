"""Export logged task trajectories into training-ready JSONL datasets.

Each completed task in the dashboard is ``(agent, input, result, trace, quality,
feedback)`` — a labeled example produced by a frontier model. This module turns
them into per-role datasets in OpenAI **chat** format:

    {"messages": [
        {"role": "system",    "content": <the agent's own system prompt>},
        {"role": "user",      "content": <task input>},
        {"role": "assistant", "content": <task result>}],
     "metadata": {...}}

Group by `agent` (= the subagent role), optionally keep only the trajectories you
marked good (`!feedback <id> good`), and you have an SFT corpus per role — the
input to a LoRA-per-role fine-tune. The pure transforms here are unit-tested; the
DB read + file write are thin wrappers.

Effective-label precedence (highest wins):
  1. User quality (``quality`` column) — explicit human verdict, most trusted.
  2. Auto quality (``auto_quality`` column) — rule-based signal derived from
     verifier verdicts and artifact missing-citations at finish time.
  3. Judge score (``judge_score`` column) — LLM-assigned 1–5; score ≥ 4 maps to
     "good", score ≤ 2 maps to "bad", 3 is discarded (no label).

The ``effective_label`` helper is pure and unit-tested.  Each exported example's
``metadata`` includes a ``label_source`` key ("user" | "auto" | "judge" | None).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_FALLBACK_SYSTEM = "You are a specialized research subagent. Complete the task precisely."
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def effective_label(row: dict) -> tuple[str | None, str | None]:
    """Derive the best available quality label for a task row.

    Precedence (highest wins):
      1. User quality   — ``row["quality"]`` is "good" or "bad".
      2. Auto quality   — ``row["auto_quality"]`` set by derive_auto_label().
      3. Judge score    — ``row["judge_score"]`` ≥ 4 → "good", ≤ 2 → "bad".

    Returns:
        (label, source) where label is "good" | "bad" | None and source is
        "user" | "auto" | "judge" | None.
    """
    uq = (row.get("quality") or "").strip().lower()
    if uq in ("good", "bad"):
        return uq, "user"

    aq = (row.get("auto_quality") or "").strip().lower()
    if aq in ("good", "bad"):
        return aq, "auto"

    js = row.get("judge_score")
    if js is not None:
        try:
            js = int(js)
        except (TypeError, ValueError):
            js = None
    if js is not None:
        if js >= 4:
            return "good", "judge"
        if js <= 2:
            return "bad", "judge"

    return None, None


def _safe_filename(agent: str) -> str:
    """A filesystem-safe stem for an agent name (it comes from stored data)."""
    return _UNSAFE_NAME.sub("_", (agent or "unknown").strip()).strip("._") or "unknown"


def system_prompt_for(agent: str) -> str:
    """The system prompt the given subagent role runs under.

    So each exported example carries the same framing the model saw at inference
    — what you want to bake into a fine-tuned/distilled model. Imported lazily to
    keep this module importable without the heavier agent deps.
    """
    try:
        from ..agents.literature import _SYSTEM as LITERATURE
        from ..writing.lit_review import LiteratureReviewer
        from ..writing.methodology import MethodologyWriter
        from ..writing.paper import PaperWriter
    except Exception:  # noqa: BLE001 — never let prompt lookup break an export
        return _FALLBACK_SYSTEM

    return {
        "research_literature": LITERATURE,
        "literature": LITERATURE,
        "literature_review": LiteratureReviewer.system_prompt,
        "methodology": MethodologyWriter.system_prompt,
        "paper_draft": PaperWriter.system_prompt,
    }.get(agent, _FALLBACK_SYSTEM)


def task_to_example(row: dict, *, include_trace: bool = False) -> dict | None:
    """Turn one task row into a chat-format SFT example, or None if unusable.

    Skips rows with no input or no result (nothing to learn from). When
    `include_trace` is set, the full reasoning/tool-call trace is attached too
    (for distilling the *process*, not just the final answer).

    The ``metadata`` dict includes ``label`` (the effective label) and
    ``label_source`` ("user" | "auto" | "judge" | None) so downstream consumers
    know how much to trust the label.
    """
    inp = (row.get("input") or "").strip()
    out = (row.get("result") or "").strip()
    if not inp or not out:
        return None
    label, label_source = effective_label(row)
    example: dict = {
        "messages": [
            {"role": "system", "content": system_prompt_for(row.get("agent", ""))},
            {"role": "user", "content": inp},
            {"role": "assistant", "content": out},
        ],
        "metadata": {
            "task_id": row.get("id"),
            "agent": row.get("agent"),
            "quality": row.get("quality"),
            "feedback": row.get("feedback"),
            "label": label,
            "label_source": label_source,
            "created_at": str(row.get("created_at") or ""),
        },
    }
    if include_trace:
        example["trace"] = row.get("trace") or []
    return example


async def export_dataset(
    task_store, out_dir, *, agents=None, good_only: bool = False,
    include_trace: bool = False, since=None,
    effective_label_filter: str | None = None,
) -> dict:
    """Export completed tasks as one JSONL file per agent role.

    Returns a manifest ``{agent: {"count", "path"}}`` (also written to
    `manifest.json`).

    Filtering options:
    - ``good_only``: keep only tasks where ``quality == "good"`` (user label only;
      legacy behaviour, unchanged from before).
    - ``effective_label_filter``: filter by effective label using the three-tier
      precedence (user > auto > judge).  Pass "good" or "bad".  Supersedes
      ``good_only`` when both are set.
    - ``agents``: restrict to specific roles.
    - ``since``: ISO lower bound on ``created_at``.
    """
    # For backward compatibility, good_only pushes down a DB-level filter on the
    # user quality column.  effective_label_filter post-filters in Python so it
    # can use all three label sources.
    quality = ("good",) if (good_only and not effective_label_filter) else None
    rows = await task_store.list_for_export(agents=agents, quality=quality, since=since)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_agent: dict[str, list[dict]] = {}
    for row in rows:
        if effective_label_filter:
            label, _ = effective_label(row)
            if label != effective_label_filter:
                continue
        example = task_to_example(row, include_trace=include_trace)
        if example is not None:
            by_agent.setdefault(row.get("agent") or "unknown", []).append(example)

    manifest: dict[str, dict] = {}
    for agent, examples in sorted(by_agent.items()):
        path = out / f"{_safe_filename(agent)}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, default=str, ensure_ascii=False) + "\n")
        manifest[agent] = {"count": len(examples), "path": str(path)}

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    logger.info(
        "Exported %d example(s) across %d role(s) to %s",
        sum(v["count"] for v in manifest.values()), len(manifest), out,
    )
    return manifest


def main() -> None:
    """CLI: dump the task trajectories as per-role JSONL training sets."""
    import argparse
    import asyncio

    from ..agents.task_store import TaskStore
    from ..config import settings
    from ..db import open_pool

    parser = argparse.ArgumentParser(
        description="Export logged task trajectories as per-role JSONL training data.",
    )
    parser.add_argument("--out", default=f"{settings.output_dir}/datasets",
                        help="output directory (default: outputs/datasets)")
    parser.add_argument("--agents", nargs="*", default=None,
                        help="restrict to these agent roles (default: all)")
    parser.add_argument("--good-only", action="store_true",
                        help="only tasks you marked `!feedback <id> good`")
    parser.add_argument("--effective-label", choices=["good", "bad"], default=None,
                        help="filter by effective label (user > auto > judge precedence)")
    parser.add_argument("--include-trace", action="store_true",
                        help="attach the full reasoning/tool-call trace to each example")
    parser.add_argument("--since", default=None,
                        help="ISO lower bound on created_at, e.g. 2026-01-01")
    args = parser.parse_args()

    if args.since:
        from datetime import datetime

        try:
            datetime.fromisoformat(args.since)
        except ValueError:
            parser.error(f"--since must be an ISO date/datetime, got {args.since!r}")

    async def _run() -> None:
        pool = await open_pool()
        if pool is None:
            print("No DATABASE_URL configured — there are no logged tasks to export.")
            return
        try:
            store = TaskStore(pool)
            await store.setup()  # ensure the tasks schema (+ label columns) exists
            manifest = await export_dataset(
                store, args.out, agents=args.agents, good_only=args.good_only,
                include_trace=args.include_trace, since=args.since,
                effective_label_filter=args.effective_label,
            )
        finally:
            await pool.close()

        total = sum(v["count"] for v in manifest.values())
        print(f"Exported {total} example(s) across {len(manifest)} role(s) → {args.out}")
        for agent, info in sorted(manifest.items()):
            print(f"  {agent}: {info['count']} → {info['path']}")

    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    main()
