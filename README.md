# research_agent

A personal, autonomous research agent. The goal: a collaborator that explores
the literature, discusses and writes methodology, writes and runs the code for
that methodology, runs experiments (HuggingFace / cloud compute), reports
findings, innovates with you, and helps write the paper — reachable over Discord.

It's built on **LangGraph** (provider-agnostic orchestration), uses **Claude**
by default for the model, and gets its external capabilities from **MCP
servers** (starting with [paperclip](https://paperclip.gxl.ai) for full-text
papers, clinical trials, and regulatory documents).

## Status

**Milestone 1 — Discord + literature (done)**

- Discord bot ↔ agent ↔ literature (via MCP).
- Tool-using (ReAct) LangGraph loop.
- Pluggable MCP servers: paperclip built in, plus a JSON config for adding more.

**Milestone 2 — Memory (done)**

A memory subsystem on **Postgres + pgvector**, organized by the
semantic / episodic / procedural framework:

- **Semantic** (facts) — [mem0](https://github.com/mem0ai/mem0) over pgvector,
  with built-in entity linking and citation/provenance on every fact. OpenAI
  embeddings (Anthropic has no embeddings API).
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

**Milestone 3 — Experiment runner, Phase 1 (done)**

Run experiments on a **separate GPU node** registered via config, dispatched
over SSH + Docker. See [docs/experiment-runner.md](docs/experiment-runner.md).

- Pluggable `ComputeBackend` (v1 `SSHDockerBackend`); workspace rsync'd up and
  mounted, secrets via remote `--env-file`, outputs fetched as artifacts.
- Lifecycle tracked in the experiment registry; background poller reports
  completions to Discord.
- Agent tools: propose / write code / launch / status / logs / cancel.
- Human approval before each launch (`!approve <id>`), `!runs`, `!cancel <id>`.
- Disabled unless `COMPUTE_SSH_HOST`/`COMPUTE_SSH_USER` are set.

Roadmap (not yet wired up):

- Methodology design & writing; paper drafting
- Experiment Phase 2+: metrics streaming, findings reports, artifacts → HF Hub
- Subagents for parallel ablations

## Architecture

```
Discord  ──►  ResearchBot  ──►  LangGraph agent  ──►  MCP tools (paperclip, …)
                                   │  agent (LLM) ⇄ tools loop
                                   └  per-channel memory (checkpointer)
```

```
src/research_agent/
  config.py          # settings via env / .env
  llm.py             # provider-agnostic model factory (Anthropic default)
  prompts.py         # system prompt / persona + prompt composer
  mcp_client.py      # load tools from MCP servers
  db.py              # Postgres pool + durable checkpointer
  agent/
    state.py         # graph state (messages + memory bookkeeping)
    graph.py         # load_context -> agent -> tools loop
  memory/
    manager.py       # one interface over the three stores
    semantic.py      # facts (mem0 / pgvector)
    episodic.py      # experiences (lab notebook + experiment registry)
    procedural.py    # instructions (learned preferences/procedures)
    summarize.py     # rolling summarization
    tokens.py        # token estimate + nudge boundaries
    maintenance.py   # idle archival + consolidation/reflection
  discord_bot/
    bot.py           # Discord client -> graph bridge, commands
  cli.py             # local terminal REPL (no Discord)
  main.py            # entrypoint: runs the Discord bot
mcp_servers.example.json  # template for adding more MCP servers
docker-compose.yml        # Postgres + pgvector for memory
docker/                   # (planned) experiment sandbox
```

## Memory setup

```bash
docker compose up -d            # Postgres + pgvector
pip install -e ".[memory]"      # mem0, postgres checkpointer, psycopg
```

Then set `DATABASE_URL` and `OPENAI_API_KEY` in `.env`. Schema is created
automatically on first run. Commands: `!checkpoint`, `!remember <text>`, `!help`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add ".[openai]" for OpenAI, ".[dev]" for tests

cp .env.example .env        # then fill in the values
```

Required in `.env`:

- `DISCORD_TOKEN` — your Discord bot token
- `ANTHROPIC_API_KEY` — model access (or set `LLM_PROVIDER=openai` + `OPENAI_API_KEY`)
- `PAPERCLIP_API_KEY` — `gxl_...` key from https://paperclip.gxl.ai/keys

### Discord bot setup

1. Create an app at https://discord.com/developers/applications → **Bot**.
2. Enable the **Message Content Intent**.
3. Copy the bot token into `DISCORD_TOKEN`.
4. Invite the bot to your server (OAuth2 → scopes `bot`, with send-messages perms).

The bot replies in DMs and when you **@-mention** it in a channel.

## Run

```bash
research-agent                 # or: python -m research_agent.main   (Discord)
python -m research_agent.cli   # local terminal REPL for testing
```

## Adding MCP servers

Copy `mcp_servers.example.json` to `mcp_servers.json` and add entries. `${VAR}`
is substituted from the environment. Supported transports: `streamable_http`,
`sse`, `stdio`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
