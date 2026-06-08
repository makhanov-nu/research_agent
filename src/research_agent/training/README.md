# Training: distilling small per-role models from your own logs

The agent records every delegation as a labeled trajectory — `(agent, input,
result, trace, quality, feedback)` — produced by a frontier model. That is a
dataset. This package turns it into per-role training sets so you can distill a
**small, cheap specialist per subagent kind** (literature, methodology, paper…),
then serve them as **LoRA adapters over one shared base** and route by agent.

This is the controllable, low-cost alternative to a MoE: one base model in
memory + megabyte-sized adapters you fully control, instead of an opaque
token-routed mixture that's heavy to train and serve.

## 1. Collect data (just use the agent)

Every completed subagent task is logged automatically. Improve label quality by
rating results in Discord:

```
!feedback <task_id> good
!feedback <task_id> bad missed the 2024 survey; always include a limitations para
```

`good` marks a trajectory worth imitating; `bad <note>` records the correction.
Both write the `quality`/`feedback` columns used for filtering and DPO later.

## 2. Export per-role datasets

```bash
research-agent-export                      # all roles → outputs/datasets/<role>.jsonl
research-agent-export --good-only          # only trajectories you marked good
research-agent-export --agents methodology literature_review
research-agent-export --include-trace      # attach reasoning/tool-calls (distill the process)
```

Output is OpenAI **chat** JSONL, one example per line, carrying the role's own
system prompt:

```json
{"messages": [
  {"role": "system", "content": "You are a research methodologist..."},
  {"role": "user", "content": "design a methodology for ..."},
  {"role": "assistant", "content": "```latex\n\\section{Methodology}..."}],
 "metadata": {"task_id": 42, "agent": "methodology", "quality": "good", ...}}
```

A `manifest.json` summarizes counts per role.

## 3. Fine-tune a LoRA per role

When a role has enough good examples (hundreds–low thousands — until then,
the in-context lesson loop is doing the work), train an adapter:

```bash
pip install -e ".[train]"     # torch, transformers, trl, peft, accelerate (GPU box)

python -m research_agent.training.train_lora \
    --data outputs/datasets/methodology.jsonl \
    --base Qwen/Qwen2.5-7B-Instruct \
    --out adapters/methodology
```

`train_lora.py` is a reference SFT-LoRA recipe (edit freely). Repeat per role;
each produces a small adapter over the *same* base.

## 4. Serve + route

Serve the shared base with multiple adapters (e.g. vLLM multi-LoRA / S-LoRA) and
point each subagent at its adapter via `LLM_MODEL`/provider config — easy
trajectories to the small specialist, hard ones (or low-confidence) falling back
to the frontier model. A cascade, not a hard cutover.

## What to distill — and what not to

- **Good targets:** `research_literature`, `literature_review`, `methodology`,
  `paper_draft`, and the experiment coder — narrow, structured, repetitive.
- **Keep frontier:** the **consortium** panel. Its value is diversity (several
  independent frontier models) and novelty/judgment at the capability frontier;
  distilling it into one small model destroys both. (The chair/synthesis step is
  more structured and *could* be distilled if you want.)

## Beyond SFT

`bad` + correction labels give you preference pairs for **DPO** (chosen =
corrected, rejected = original) once you capture corrected outputs — a natural
phase-3 once the `feedback` column has enough signal.
