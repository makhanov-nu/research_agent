"""Central configuration, loaded from environment / .env file."""

from __future__ import annotations

from urllib.parse import urlparse, unquote

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Discord ---
    discord_token: str = ""

    # --- LLM ---
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096

    # --- MCP servers (tools) ---
    paperclip_api_key: str = ""
    paperclip_url: str = "https://paperclip.gxl.ai/mcp"
    # Optional JSON file declaring additional MCP servers.
    mcp_config_path: str = "mcp_servers.json"

    # --- Memory ---
    # Postgres DSN, e.g. postgresql://user:pass@localhost:5432/research_agent
    # When empty, memory is disabled and the bot falls back to in-process state.
    database_url: str = ""
    # Embedder for mem0 (Anthropic has no embeddings API). Reads OPENAI_API_KEY.
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = 1536
    mem0_collection: str = "semantic_memory"
    # Single global memory pool -> one logical owner id.
    memory_user_id: str = "global"

    # Summarization / context management
    # Keep this many most-recent messages verbatim when summarizing.
    summary_keep_last: int = 6
    # Auto-summarize older turns once live context exceeds this many tokens.
    summary_token_threshold: int = 24000
    # Surface a "want to checkpoint?" nudge each time context crosses a multiple
    # of this many tokens.
    nudge_every_tokens: int = 20000

    # Archive channels with no activity for this many days.
    archive_idle_days: int = 7
    # How often (seconds) the background maintenance loop runs.
    maintenance_interval_seconds: int = 6 * 60 * 60

    # Name the agent answers to / signs off as.
    agent_name: str = "Beaker"

    @property
    def memory_enabled(self) -> bool:
        return bool(self.database_url)

    def pg_components(self) -> dict:
        """Parse database_url into the discrete fields mem0's pgvector wants."""
        parsed = urlparse(self.database_url)
        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 5432,
            "user": unquote(parsed.username) if parsed.username else "postgres",
            "password": unquote(parsed.password) if parsed.password else "",
            "dbname": parsed.path.lstrip("/") or "postgres",
        }


settings = Settings()
