# research_agent

A personal, autonomous research agent. The goal: a collaborator that explores
the literature, discusses and writes methodology, writes and runs the code for
that methodology, runs experiments (HuggingFace / cloud compute), reports
findings, innovates with you, and helps write the paper — reachable over Discord.

It's built on **LangGraph** + **LangChain v1** (provider-agnostic orchestration),
runs its models through **OpenRouter** by default (any Anthropic / OpenAI /
Google / DeepSeek model), and gets its external capabilities from **MCP servers**
(starting with [paperclip](https://paperclip.gxl.ai) for full-text papers,
clinical trials, and regulatory documents).

📚 **Documentation:** https://makhanov-nu.github.io/research_agent/

## Architecture: orchestrator + subagents

The Discord-facing agent is a thin **orchestrator**. Instead of doing everything
itself, its tools *delegate* to specialized **subagents** and receive only their
final results — keeping the orchestrator's context lean (divide and conquer).

```
                ┌──────────────── Discord ────────────────┐
                │                                          │
                ▼                                          │ results / events
        ResearchBot (orchestrator, LangGraph)              │
                │  delegates via tools                     │
   ┌────────────┼───────────────┬──────────────┬──────────┘
   ▼            ▼               ▼              ▼
 literature   LaTeX          consortium     experiment
 subagent     lit-review     (!ideate)      runner
 (paperclip   writer         multi-model    (SSH+Docker,
  MCP)                        debate         GPU node)
   │
   └─ every delegation is recorded in the Postgres **task dashboard**
      (result + full trace of reasoning & tool calls)
```

- **Subagents** run on LangChain v1 `create_agent` + middleware. A subagent
  returns only its final answer to the orchestrator; its full reasoning and tool
  calls are captured separately in the task `trace` column via
  `TaskRecorderMiddleware`.
- **Task dashboard** — a Postgres `tasks` table records every delegation
  (input, status, result, trace, timings). Inspect with `!tasks` / `!task <id>`
  / `!trace <id>`.
- **Background dispatcher** — `dispatch_task(agent, task)` runs subagents
  async/parallel (semaphore-capped at `MAX_PARALLEL_TASKS`). Completion is
  **push-based**: when a task finishes it wakes the orchestrator with an event
  (no polling).

## Capabilities

- **Literature** — a research subagent owns the paperclip MCP tools and returns
  grounded, cited synthesis. (The orchestrator has no raw MCP access.)
- **LaTeX literature review** — drafts a review from related work and saves
  `.tex` / `.bib` to the outputs directory; fetch with `!getfile <path>`.
- **Consortium** (`!ideate <topic>`) — convenes a multi-model panel (Claude Opus,
  GPT-5.5, Gemini 3 Pro, DeepSeek-R1) that grounds itself in the literature, then
  proposes / debates / argues over a **shared transcript** so the models hear each
  other, and a chair synthesizes **3 Q1-level conference/journal ideas**.
- **Experiment runner** — runs experiments on a **separate GPU node** registered
  via config, dispatched over SSH + Docker, with human approval before each
  launch. Dormant until `COMPUTE_SSH_*` is set. See
  [docs/experiment-runner.md](docs/experiment-runner.md).

## Memory

A memory subsystem on **Postgres + pgvector**, organized by the
semantic / episodic / procedural framework:

- **Semantic** (facts) — [mem0](https://github.com/mem0ai/mem0) over pgvector,
  with built-in entity linking and provenance on every fact. Embeddings via
  OpenRouter (`openai/text-embedding-3-small`).
- **Episodic** (experiences) — a Postgres "lab notebook": action log, per-channel
  activity, and an **experiment registry** (config, metrics, status, artifacts).
- **Procedural** (instructions) — learned preferences/procedures prepended to the
  system prompt (`!remember <text>`).
- **Working memory** — durable LangGraph **Postgres checkpointer** (survives restarts).
- **Token management** — rolling auto-summarization keeps live context small; a
  **20k-token nudge** asks whether to checkpoint to long-term memory; `!checkpoint`
  forces it.
- **Maintenance loop** — archives channels idle > 7 days into long-term memory and
  runs a **consolidation/reflection** pass (episodic → semantic insights).

Memory is optional: without `DATABASE_URL` the bot runs on in-process state.

## Commands

| Command | What it does |
|---|---|
| `!ideate <topic>` | Convene the multi-model consortium to propose 3 Q1 ideas |
| `!tasks` / `!task <id>` / `!trace <id>` | The task dashboard: recent tasks, status+result, full trace export |
| `!checkpoint` (`!summarize`) | Summarize this thread to long-term memory and reset live context |
| `!remember <text>` | Store a durable preference/instruction |
| `!getfile <path>` | Fetch a written output (e.g. a drafted LaTeX review) |
| `!runs` / `!approve <id>` / `!cancel <id>` | List / approve+launch / cancel experiments |
| `!help` | Show commands |

Otherwise, just talk to it — DM or **@-mention** it in a channel.

## Project layout

```
src/research_agent/
  config.py          # settings via env / .env (OpenRouter default)
  llm.py             # provider-agnostic model factory + per-model OpenRouter clients
  prompts.py         # orchestrator system prompt + composer
  mcp_client.py      # load tools from MCP servers
  db.py              # Postgres pool + durable checkpointer
  agent/             # orchestrator graph (load_context -> agent -> tools)
  agents/            # subagents + delegation
    registry.py      #   build the orchestrator's delegated tools
    subagent.py      #   create_agent runner + subagent-tool factory
    middleware.py    #   TaskRecorderMiddleware (capture trace, return result)
    task_store.py    #   Postgres tasks table (the dashboard)
    dispatcher.py    #   async parallel dispatch + push-based completion
    literature.py    #   literature research subagent
    consortium_tool.py
  consortium/        # multi-model shared-transcript debate (!ideate)
  memory/            # manager + semantic/episodic/procedural + summarize/tokens/maintenance
  writing/           # LaTeX literature-review writer
  experiments/       # ComputeBackend + SSHDockerBackend + runner + tools
  discord_bot/bot.py # Discord client -> graph bridge, commands
  cli.py             # local terminal REPL (no Discord)
  main.py            # entrypoint: runs the Discord bot
mcp_servers.example.json  # template for adding more MCP servers
docker-compose.yml        # Postgres + pgvector for memory
mkdocs.yml                # docs site (Material for MkDocs + mkdocstrings)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[memory]"   # core + memory; add ".[dev]" for tests, ".[docs]" for the docs site

cp .env.example .env         # then fill in the values
docker compose up -d         # Postgres + pgvector (for memory)
```

Required in `.env`:

- `DISCORD_TOKEN` — your Discord bot token
- `OPENROUTER_API_KEY` — model access via OpenRouter (default provider)
- `PAPERCLIP_API_KEY` — `gxl_...` key from https://paperclip.gxl.ai/keys
- `DATABASE_URL` — Postgres connection string (enables memory; schema auto-created)

The model is selected via `LLM_MODEL` (default `anthropic/claude-sonnet-4.6`).
To run against the Anthropic or OpenAI APIs directly instead of OpenRouter, set
`LLM_PROVIDER=anthropic` (+ `ANTHROPIC_API_KEY`) or `LLM_PROVIDER=openai`
(+ `OPENAI_API_KEY`).

### Discord bot setup

1. Create an app at https://discord.com/developers/applications → **Bot**.
2. Enable the **Message Content Intent**.
3. Copy the bot token into `DISCORD_TOKEN`.
4. Invite the bot to your server (OAuth2 → scope `bot`, with send-messages perms).

## Run

```bash
research-agent                 # or: python -m research_agent.main   (Discord)
python -m research_agent.cli   # local terminal REPL for testing
```

## Adding MCP servers

Copy `mcp_servers.example.json` to `mcp_servers.json` and add entries. `${VAR}`
is substituted from the environment. Supported transports: `streamable_http`,
`sse`, `stdio`.

## Documentation

Built with **Material for MkDocs** + **mkdocstrings**, auto-deployed to GitHub
Pages on every push to `main`.

```bash
pip install -e ".[docs]"
mkdocs serve     # live preview at http://127.0.0.1:8000
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```
