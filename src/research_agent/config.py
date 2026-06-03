"""Central configuration, loaded from environment / .env file."""

from __future__ import annotations

from urllib.parse import urlparse, unquote

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Populate os.environ from .env so SDKs that read the environment directly
# (Anthropic, OpenAI) pick up their keys. Does not override real env vars,
# so systemd/shell-provided values still win.
load_dotenv()


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
    # Provider: "openrouter" (default), "anthropic", or "openai".
    llm_provider: str = "openrouter"
    # Model id in the provider's namespace. OpenRouter uses slugs like
    # "anthropic/claude-sonnet-4.6"; Anthropic uses "claude-sonnet-4-6".
    llm_model: str = "anthropic/claude-sonnet-4.6"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096
    # OpenRouter (OpenAI-compatible) credentials.
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

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

    # --- Experiment compute node (registered via config) ---
    # SSH target for the GPU box. When host+user are empty, the runner is off.
    compute_ssh_host: str = ""
    compute_ssh_user: str = ""
    compute_ssh_port: int = 22
    compute_ssh_key: str = ""  # private key path; empty -> default keys / agent
    # Remote directory under which per-experiment workspaces/outputs live.
    compute_workdir: str = "~/research_agent_runs"
    # Base image the universal experiment image is built FROM.
    compute_base_image: str = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
    # The universal experiment image (built once per box, or a registry ref to
    # pull). Default runs use this; it bakes the common ML stack so jobs skip
    # re-installing it. A local tag (no registry host) is built on the box.
    experiment_image: str = "research-agent/experiment:latest"
    # Default docker --gpus value ("all", "0", or "" to disable GPU access).
    compute_default_gpus: str = "all"
    # Shared docker network (on the GPU box) joining experiment containers to the
    # MLflow server, so jobs can reach it by name.
    compute_network: str = "ra-net"
    # Persistent HuggingFace cache volume mounted into every job (so datasets and
    # model weights are downloaded once and reused across runs/trials).
    compute_hf_cache_volume: str = "ra-hf-cache"

    # --- Experiment authoring (Codex-style coder model) ---
    # OpenRouter slug for the model that writes/iterates experiment code.
    experiment_coder_model: str = "openai/gpt-5.5"
    # HuggingFace token passed into jobs (env-file) for dataset/model downloads.
    hf_token: str = ""

    # --- MLflow experiment tracking (server runs on the GPU box) ---
    mlflow_enabled: bool = True
    # MLflow tracking server container + image (started on the compute node).
    mlflow_container: str = "ra-mlflow"
    mlflow_image: str = "ghcr.io/mlflow/mlflow:v2.16.2"
    # Port the server binds (to 127.0.0.1) on the compute node.
    mlflow_port: int = 5000
    # Docker named volume holding the MLflow backend store (sqlite) + artifacts.
    mlflow_volume: str = "ra-mlflow-data"
    # MLflow experiment all runs are grouped under.
    mlflow_experiment_name: str = "research_agent"

    @property
    def mlflow_tracking_uri_internal(self) -> str:
        """URI experiment containers use to reach MLflow (over the shared network)."""
        return f"http://{self.mlflow_container}:{self.mlflow_port}"

    # Local directory where the agent authors experiment code before dispatch.
    experiment_workspace_dir: str = "workspace"
    # Local directory where fetched experiment artifacts are stored.
    experiment_artifacts_dir: str = "artifacts"
    # Directory for written outputs (LaTeX literature reviews, drafts, etc.).
    output_dir: str = "outputs"

    # --- Ideation consortium (multi-model debate via OpenRouter) ---
    # Comma-separated OpenRouter model slugs forming the panel.
    consortium_models: str = (
        "anthropic/claude-opus-4.7,openai/gpt-5.5,"
        "google/gemini-3-pro,deepseek/deepseek-r1"
    )
    # The model that synthesizes the final ideas.
    consortium_chair_model: str = "anthropic/claude-opus-4.7"
    consortium_temperature: float = 0.6
    # Number of debate rounds after the opening proposals.
    consortium_rounds: int = 1

    @property
    def panel_models(self) -> list[str]:
        return [m.strip() for m in self.consortium_models.split(",") if m.strip()]
    # Require a human approval in Discord before launching a run.
    experiment_require_approval: bool = True
    # How often (seconds) the job poller checks active runs.
    job_poll_interval_seconds: int = 60
    # Max subagent tasks the background dispatcher runs concurrently.
    max_parallel_tasks: int = 4

    # Name the agent answers to / signs off as.
    agent_name: str = "Beaker"

    @property
    def memory_enabled(self) -> bool:
        return bool(self.database_url)

    @property
    def compute_enabled(self) -> bool:
        return bool(self.compute_ssh_host and self.compute_ssh_user)

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
