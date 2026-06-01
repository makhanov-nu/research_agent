"""Central configuration, loaded from environment / .env file."""

from __future__ import annotations

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

    # Name the agent answers to / signs off as.
    agent_name: str = "Beaker"


settings = Settings()
