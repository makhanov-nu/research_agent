# Experiment Runner — Design (draft)

Status: **proposed** — pending sign-off on the decisions in the last section.

## Goal

Let the agent take a methodology + code and actually **run experiments** —
autonomously, on a **separate GPU machine**, while staying a collaborator that
reports progress and findings over Discord. Reuses the existing experiment
registry (`memory/episodic.py`) and the LangGraph tool loop.

## Constraints (from the deployment)

- **Orchestrator** = the bot host VPS (no GPU, 2 vCPU / 3.8 GB). Always-on brain.
  It decides, generates code, dispatches, tracks, and reports. It does **not**
  run heavy compute.
- **Compute** = a separate **GPU VPS** the user provides, plus **HuggingFace**
  (Hub for models/datasets; optionally HF Jobs for managed compute).
- Execution is **sandboxed in Docker** on the compute node.
- Operates autonomously but collaboratively; reports to Discord; can fan out via
  subagents.

## Topology

```
  ┌────────────────────────────┐         ssh / api         ┌─────────────────────────┐
  │  Orchestrator (bot host)    │ ────────────────────────► │  Compute node (GPU VPS) │
  │  - agent graph + tools      │   workspace sync (rsync)  │  - Docker engine + GPU  │
  │  - experiment registry (PG) │ ◄──────────────────────── │  - runs experiment      │
  │  - job poller + reporting   │   status / logs / metrics │    containers           │
  │  - Discord I/O              │                           │  - output dir + metrics │
  └────────────────────────────┘                           └─────────────────────────┘
                    │                                                    │
                    └───────────────── HuggingFace Hub ◄─────────────────┘
                              (datasets in, models/artifacts out)
```

## ComputeBackend abstraction

Dispatch is hidden behind one interface so we can start simple and swap later
(own GPU box → worker API → HF Jobs) without touching the agent or registry.

```python
class ComputeBackend(Protocol):
    async def submit(self, job: JobSpec) -> JobHandle
    async def status(self, handle: JobHandle) -> JobStatus      # queued/running/succeeded/failed/cancelled
    async def logs(self, handle: JobHandle, since: str | None) -> str
    async def cancel(self, handle: JobHandle) -> None
    async def fetch_artifacts(self, handle: JobHandle, dest: str) -> list[Artifact]
```

Implementations:
- **`SSHDockerBackend`** (v1): SSH to the GPU box, rsync the workspace, launch a
  **detached** `docker run` (so long training survives the SSH session), record
  the container id as the handle, then poll `docker inspect` / tail logs.
- `WorkerApiBackend` (later): a small FastAPI daemon on the GPU box; cleaner
  contract, supports multiple nodes and a queue.
- `HFJobsBackend` (later): submit to HuggingFace Jobs; poll job id.

## JobSpec & packaging

```python
@dataclass
class JobSpec:
    experiment_id: int
    workspace: str            # local dir with the experiment code (agent-authored)
    image: str                # pinned base ML image, or built from workspace/Dockerfile
    command: list[str]        # entrypoint, e.g. ["python", "train.py", "--epochs", "3"]
    env: dict[str, str]       # runtime secrets/config (HF_TOKEN, etc.) — never baked in
    resources: Resources      # gpus, memory, time_limit
    output_dir: str           # container path collected as artifacts (metrics.jsonl, ckpts)
```

- **Code packaging:** each experiment gets `workspace/<exp_id>/`. The agent
  writes code there; the backend rsyncs it to the compute node and mounts it
  read-only into the container. A pinned base image (CUDA + torch + common deps)
  is the default; a generated `Dockerfile`/`requirements.txt` is supported for
  custom deps. Reproducibility favored over convenience.
- **Secrets** (HF token, etc.) are injected as runtime env from orchestrator
  config — never written into the workspace or image layers.

## Job lifecycle

```
planned ─► building ─► running ─► succeeded
                          │
                          ├─► failed
                          └─► cancelled
```

Mapped onto the existing `experiments` table:
- `status` — the state above.
- `config` — JobSpec snapshot (image, command, resources) + the backend handle.
- `metrics` — latest parsed metrics (training script appends `metrics.jsonl`;
  backend tails it).
- `artifacts` — list of output refs (HF Hub URLs / file paths).

*(Adds two small columns or reuses JSONB: `backend` and `handle`. Minor migration.)*

## Async tracking & reporting

- The orchestrator never blocks on a run. A **job poller** (same pattern as the
  memory maintenance loop) periodically checks active experiments, updates the
  registry, and posts Discord updates **on state change** (started / failed /
  finished) plus optional periodic heartbeats with the latest metric.
- On completion the agent fetches metrics + artifacts, writes a **findings
  summary** to Discord and to semantic memory (feeding the reflection job).

## Agent tools (LangGraph)

Exposed to the model so it can run the loop:

- `propose_experiment(title, hypothesis, plan)` → registry row (`planned`).
- `write_experiment_code(exp_id, files)` → writes into `workspace/<exp_id>/`.
- `launch_experiment(exp_id, resources)` → dispatch via backend. **Gated** (see below).
- `experiment_status(exp_id)` / `experiment_logs(exp_id, tail)` / `cancel_experiment(exp_id)`.
- `list_experiments()` / `compare_experiments(ids)` → for ablations.
- `report_findings(exp_id)` → summarize + post.

## Safety & approval

- **Approval gate (default ON):** launching a GPU run costs real resources, so
  `launch_experiment` requires a human 👍 in Discord (with an estimate) before
  dispatch. Configurable to auto-approve within a time/cost budget.
- **Sandbox:** Docker with resource limits (`--gpus`, `--memory`, `--pids-limit`),
  no host mounts beyond workspace (ro) + output (rw), constrained network, a
  hard wall-clock timeout that auto-kills, non-root user.
- **Cleanup:** reap finished/stale containers, enforce a disk quota for outputs
  on the GPU box, idempotent re-submits.

## Subagents (parallel experiments)

A **sweep** primitive builds N JobSpecs from a config grid; a lead agent fans
them out (bounded by the backend's concurrency limit) and aggregates results.
v1 keeps concurrency small; the tool surface is designed so ablation sweeps and
worker subagents slot in without redesign.

## Phased build plan

1. **Phase 1 (MVP):** `ComputeBackend` + `SSHDockerBackend`, `JobSpec`, workspace
   rsync, detached `docker run`, status polling, registry wiring, the core tools
   (`propose` / `write_code` / `launch` / `status` / `logs`), Discord approval +
   completion report. Validate end-to-end with a trivial CPU job.
2. **Phase 2:** metrics streaming, findings reporting, artifacts → HF Hub.
3. **Phase 3:** sweeps + worker subagents for parallel ablations.
4. **Phase 4:** `HFJobsBackend`, budgets/auto-approval, richer sandboxing.

## Decisions needed (sign-off)

1. **Dispatch mechanism** — *recommend* `SSHDockerBackend` for v1 (matches the
   SSH setup, no extra services), with the backend interface keeping a worker
   API / HF Jobs pluggable later.
2. **Compute target** — *recommend* own GPU VPS as primary + HF Hub for
   datasets/artifacts. Add `HFJobsBackend` later if you want burst/managed GPUs.
3. **Approval** — *recommend* human approval in Discord before each launch
   (default), with an optional auto-approve budget. Or fully autonomous?
4. **Code packaging** — *recommend* rsync the agent-authored workspace + run in a
   pinned base image (Dockerfile/requirements only when custom deps are needed).
