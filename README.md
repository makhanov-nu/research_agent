# research_agent

A personal, autonomous research agent. The goal: a collaborator that explores
the literature, discusses and writes methodology, writes and runs the code for
that methodology, runs experiments (HuggingFace / cloud compute), reports
findings, innovates with you, and helps write the paper — reachable over Discord.

It's built on **LangGraph** (provider-agnostic orchestration), uses **Claude**
by default for the model, and gets its external capabilities from **MCP
servers** (starting with [paperclip](https://paperclip.gxl.ai) for full-text
papers, clinical trials, and regulatory documents).

## Status — Milestone 1

Working now:

- Discord bot ↔ agent ↔ literature (via MCP).
- Tool-using (ReAct) LangGraph loop with per-channel conversation memory.
- Pluggable MCP servers: paperclip built in, plus a JSON config for adding more.

Roadmap (not yet wired up):

- Methodology design & writing
- Code generation for methods
- Experiment execution in a **Docker sandbox** on the GCP server
- Findings reports & paper drafting
- Subagents for parallel deep-dives

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
  prompts.py         # system prompt / persona
  mcp_client.py      # load tools from MCP servers
  agent/
    state.py         # graph state
    graph.py         # the ReAct graph (async build)
  discord_bot/
    bot.py           # Discord client -> graph bridge
  cli.py             # local terminal REPL (no Discord)
  main.py            # entrypoint: runs the Discord bot
mcp_servers.example.json  # template for adding more MCP servers
docker/              # (planned) experiment sandbox
```

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
