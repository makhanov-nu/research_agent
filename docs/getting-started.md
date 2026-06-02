# Getting started

## Requirements

- Python **3.10+** (the deployed server runs 3.14)
- **Docker** (for Postgres + pgvector)
- A **Discord bot** token, an **OpenRouter** API key, and a **paperclip** key

## Install

```bash
git clone https://github.com/makhanov-nu/research_agent.git
cd research_agent

python -m venv .venv && source .venv/bin/activate
pip install -e ".[memory]"      # add ".[dev]" for tests, ".[docs]" for these docs
```

## Configure

```bash
cp .env.example .env
```

Fill in the essentials:

| Variable | What it is |
| --- | --- |
| `DISCORD_TOKEN` | Discord bot token |
| `LLM_PROVIDER` | `openrouter` (default), `anthropic`, or `openai` |
| `LLM_MODEL` | e.g. `anthropic/claude-sonnet-4.6` (OpenRouter slug) |
| `OPENROUTER_API_KEY` | OpenRouter key (also used for embeddings) |
| `PAPERCLIP_API_KEY` | `gxl_...` key from [paperclip](https://paperclip.gxl.ai/keys) |
| `DATABASE_URL` | Postgres DSN (matches `docker-compose.yml`) |

!!! note "Embeddings"
    Semantic memory needs an embedder. With `LLM_PROVIDER=openrouter`, embeddings
    are routed through OpenRouter's OpenAI-compatible endpoint
    (`openai/text-embedding-3-small`, 1536-dim) using `OPENROUTER_API_KEY` — no
    separate OpenAI key required. Without an embeddings key, semantic memory is
    disabled and the bot falls back to episodic + procedural memory.

See the full list in [Configuration](reference/config.md).

## Bring up the database

```bash
docker compose up -d        # Postgres + pgvector on 127.0.0.1:5432
```

Schemas (memory, experiments, tasks) are created automatically on first run.

## Run

=== "Discord bot"

    ```bash
    research-agent           # or: python -m research_agent.main
    ```

=== "Local REPL (no Discord)"

    ```bash
    python -m research_agent.cli
    ```

The bot replies in DMs and when **@-mentioned** in a channel it can see. Enable
the **Message Content Intent** in the Discord developer portal, and invite the
bot with the `bot` scope.

For a production install on a VPS (systemd, etc.), see [Deployment](deployment.md).

## Tests

```bash
pip install -e ".[dev]"
pytest
```
