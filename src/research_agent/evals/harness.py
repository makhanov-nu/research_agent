"""Eval harness for per-role subagent quality measurement.

This module is the acceptance test for future per-role fine-tuned (LoRA) models.
The workflow is:

  1. ``freeze``  — snapshot the N best tasks per role into a stable JSONL eval
                   set that must NOT change between runs (for result comparability).
  2. ``run``     — generate candidate answers for each eval input using a
                   specified model slug, then have the LLM judge compare each
                   candidate to the gold answer.  Prints mean score + win/tie/loss
                   rate and writes detailed results to a timestamped JSONL.

Limitations:
- Generation happens WITHOUT tools.  Subagents in production use MCP tool calls
  (literature search, methodology validation, etc.) that cannot be replicated in an
  offline eval loop.  The harness therefore measures *text quality given the same
  prompt*, not end-to-end task performance.  This is still a meaningful signal for
  measuring whether distillation improved prose quality, reasoning depth, and
  format adherence.

CLI: research-agent-eval freeze | run
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

_COMPARE_SYSTEM = (
    "You are an expert research-quality judge comparing two outputs for the same "
    "task.  Given the task input, a gold (reference) output, and a candidate "
    "output, evaluate the candidate.\n\n"
    "Respond with ONLY a JSON object:\n"
    '{"verdict": "better"|"tie"|"worse", "score": <integer 1-5>, '
    '"rationale": "<one concise sentence>"}\n'
    "  better — candidate is meaningfully better than gold.\n"
    "  tie    — roughly equivalent quality.\n"
    "  worse  — candidate is clearly worse than gold.\n"
    "  score  — absolute quality of the CANDIDATE (1–5).\n"
    "No markdown, no extra keys."
)

_COMPARE_USER_TMPL = (
    "Agent role: {agent}\n\n"
    "Task input:\n{input}\n\n"
    "Gold output:\n{gold}\n\n"
    "Candidate output:\n{candidate}"
)

_GEN_TMPL = (
    "You are {agent}.\n\n"
    "Task:\n{input}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_compare_response(text: str) -> dict | None:
    """Parse the judge comparison JSON.  Returns dict or None."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.startswith("```")).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        try:
            obj = json.loads(text[start:end])
        except json.JSONDecodeError:
            return None

    verdict = obj.get("verdict")
    score = obj.get("score")
    rationale = obj.get("rationale", "")
    if verdict not in ("better", "tie", "worse"):
        return None
    try:
        score = int(score)
    except (TypeError, ValueError):
        return None
    if not (1 <= score <= 5):
        return None
    return {"verdict": verdict, "score": score, "rationale": str(rationale).strip()}


def _label_priority(label_source: str | None) -> int:
    """Higher = more trusted.  Used to sort tasks for freeze selection."""
    return {"user": 3, "auto": 2, "judge": 1}.get(label_source or "", 0)


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------

async def cmd_freeze(
    *,
    roles: Sequence[str] | None = None,
    n_per_role: int = 50,
    out_dir: str = "outputs/evals",
    force: bool = False,
) -> None:
    """Snapshot the best tasks per role into stable JSONL eval sets.

    Selection: effective label == "good", ordered by label source priority
    (user > auto > judge) then by created_at (oldest first for stability).
    Up to ``n_per_role`` tasks per role.

    Refuses to overwrite an existing file unless ``--force`` is passed — frozen
    sets must stay frozen for comparability across model versions.
    """
    from ..agents.task_store import TaskStore
    from ..db import open_pool
    from ..training.export import effective_label

    pool = await open_pool()
    if pool is None:
        print("No DATABASE_URL configured — nothing to freeze.")
        return
    try:
        store = TaskStore(pool)
        await store.setup()

        clauses = ["status = 'done'", "result IS NOT NULL", "result <> ''"]
        params: list = []
        if roles:
            clauses.append("agent = ANY(%s)")
            params.append(list(roles))
        # Fetch a generous slice; we'll filter + rank in Python.
        params.append(n_per_role * 20)
        sql = (
            "SELECT id, agent, input, result, quality, auto_quality, "
            "judge_score, created_at "
            "FROM tasks WHERE " + " AND ".join(clauses) +
            " ORDER BY created_at, id LIMIT %s"
        )
        async with pool.connection() as conn:
            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()
    finally:
        await pool.close()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Group by agent, filter to good effective labels, rank by source priority.
    by_agent: dict[str, list[dict]] = {}
    for row in rows:
        row_d = dict(row)
        label, source = effective_label(row_d)
        if label != "good":
            continue
        row_d["_label_source"] = source
        by_agent.setdefault(row_d["agent"] or "unknown", []).append(row_d)

    if not by_agent:
        print("No good-labeled tasks found.")
        return

    for agent, candidates in sorted(by_agent.items()):
        # Sort: user-labeled first, then auto, then judge; within tier oldest first.
        candidates.sort(key=lambda r: (-_label_priority(r["_label_source"]), r["created_at"]))
        selected = candidates[:n_per_role]

        from ..training.export import _safe_filename
        path = out / f"{_safe_filename(agent)}.jsonl"

        if path.exists() and not force:
            print(
                f"SKIP {path} (already exists; use --force to overwrite). "
                "Frozen sets must stay frozen for comparability."
            )
            continue

        with path.open("w", encoding="utf-8") as f:
            for row in selected:
                f.write(json.dumps({
                    "task_id": row["id"],
                    "role": agent,
                    "input": (row.get("input") or "").strip(),
                    "gold": (row.get("result") or "").strip(),
                    "label_source": row["_label_source"],
                }, default=str, ensure_ascii=False) + "\n")

        print(f"Froze {len(selected)} example(s) for role '{agent}' → {path}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

async def cmd_run(
    *,
    role: str,
    eval_file: str,
    model_slug: str,
    out_dir: str = "outputs/evals/runs",
    temperature: float = 0.2,
) -> None:
    """Generate candidate outputs and compare them to gold answers.

    For each item in the eval JSONL:
    - Generate a candidate with the given model slug (no tools — see module
      docstring for why).
    - Ask the judge LLM to compare candidate vs gold (verdict + 1–5 score).

    Prints:
    - Mean absolute score for the candidate.
    - Win/tie/loss counts vs the gold outputs.

    Writes detailed results to outputs/evals/runs/<role>-<timestamp>.jsonl.
    """
    from ..llm import build_openrouter_chat, build_reflection_llm
    from ..training.export import system_prompt_for

    eval_path = Path(eval_file)
    if not eval_path.exists():
        print(f"Eval file not found: {eval_file}", file=sys.stderr)
        sys.exit(1)

    items = []
    with eval_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed line: %s", exc)

    if not items:
        print("Eval file is empty.")
        return

    gen_llm = build_openrouter_chat(model_slug, temperature=temperature)
    judge_llm = build_reflection_llm()

    system_prompt = system_prompt_for(role)

    from langchain_core.messages import HumanMessage, SystemMessage

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = out / f"{role}-{ts}.jsonl"

    results: list[dict] = []
    for idx, item in enumerate(items, 1):
        inp = (item.get("input") or "").strip()
        gold = (item.get("gold") or "").strip()
        task_id = item.get("task_id")
        label_source = item.get("label_source")

        if not inp or not gold:
            logger.warning("Item %d: missing input or gold, skipping", idx)
            continue

        # 1. Generate candidate (no tools).
        try:
            gen_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=inp),
            ]
            gen_resp = await gen_llm.ainvoke(gen_messages)
            candidate = gen_resp.content if hasattr(gen_resp, "content") else str(gen_resp)
            if isinstance(candidate, list):
                candidate = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in candidate
                )
            candidate = candidate.strip()
        except Exception:  # noqa: BLE001
            logger.exception("Generation failed for item %d (task_id=%s)", idx, task_id)
            continue

        # 2. Judge comparison.
        compare_messages = [
            SystemMessage(content=_COMPARE_SYSTEM),
            HumanMessage(content=_COMPARE_USER_TMPL.format(
                agent=role,
                input=inp[:3000],
                gold=gold[:4000],
                candidate=candidate[:4000],
            )),
        ]
        comparison: dict | None = None
        try:
            judge_resp = await judge_llm.ainvoke(compare_messages)
            judge_text = judge_resp.content if hasattr(judge_resp, "content") else str(judge_resp)
            if isinstance(judge_text, list):
                judge_text = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in judge_text
                )
            comparison = _parse_compare_response(judge_text)
            if comparison is None:
                logger.warning("Item %d: failed to parse judge comparison", idx)
        except Exception:  # noqa: BLE001
            logger.exception("Judge comparison failed for item %d", idx)

        record = {
            "task_id": task_id,
            "role": role,
            "input": inp,
            "gold": gold,
            "candidate": candidate,
            "label_source": label_source,
            "verdict": comparison["verdict"] if comparison else None,
            "score": comparison["score"] if comparison else None,
            "rationale": comparison["rationale"] if comparison else None,
        }
        results.append(record)

        verdict_str = comparison["verdict"] if comparison else "error"
        score_str = str(comparison["score"]) if comparison else "?"
        print(f"  [{idx}/{len(items)}] task={task_id}  verdict={verdict_str}  score={score_str}")

    # Write detailed results.
    with results_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")

    # Summary.
    scored = [r for r in results if r["score"] is not None]
    counts: dict[str, int] = {"better": 0, "tie": 0, "worse": 0}
    for r in results:
        if r["verdict"] in counts:
            counts[r["verdict"]] += 1

    print(f"\nEval complete: {len(results)} items, {len(scored)} with scores.")
    if scored:
        mean_score = sum(r["score"] for r in scored) / len(scored)
        print(f"Mean absolute score (candidate): {mean_score:.2f} / 5.0")
    print(
        f"Win/Tie/Loss vs gold: "
        f"{counts['better']} / {counts['tie']} / {counts['worse']}"
    )
    print(f"Detailed results → {results_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point: research-agent-eval."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Eval harness: freeze golden sets or run a model candidate against them.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- freeze ---
    freeze_p = sub.add_parser(
        "freeze",
        help="Snapshot best tasks per role into a stable JSONL eval set.",
    )
    freeze_p.add_argument(
        "--roles", nargs="*", default=None,
        help="agent roles to freeze (default: all with good labels)",
    )
    freeze_p.add_argument(
        "--n", type=int, default=50,
        help="max examples per role (default: 50)",
    )
    freeze_p.add_argument(
        "--out", default="outputs/evals",
        help="output directory (default: outputs/evals)",
    )
    freeze_p.add_argument(
        "--force", action="store_true",
        help="overwrite existing frozen eval files",
    )

    # --- run ---
    run_p = sub.add_parser(
        "run",
        help="Generate candidate outputs and compare to gold via the LLM judge.",
    )
    run_p.add_argument("role", help="agent role being evaluated")
    run_p.add_argument("eval_file", help="path to the frozen JSONL eval file")
    run_p.add_argument("model_slug", help="OpenRouter model slug to evaluate")
    run_p.add_argument(
        "--out", default="outputs/evals/runs",
        help="output directory for run results (default: outputs/evals/runs)",
    )
    run_p.add_argument(
        "--temperature", type=float, default=0.2,
        help="generation temperature (default: 0.2)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "freeze":
        asyncio.run(cmd_freeze(
            roles=args.roles,
            n_per_role=args.n,
            out_dir=args.out,
            force=args.force,
        ))
    elif args.cmd == "run":
        asyncio.run(cmd_run(
            role=args.role,
            eval_file=args.eval_file,
            model_slug=args.model_slug,
            out_dir=args.out,
            temperature=args.temperature,
        ))


if __name__ == "__main__":  # pragma: no cover
    main()
