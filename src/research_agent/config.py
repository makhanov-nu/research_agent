"""Central configuration, loaded from environment / .env file."""

from __future__ import annotations

from urllib.parse import urlparse, unquote

from dotenv import load_dotenv
from pydantic import Field
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
    # Provider: "openrouter" (default), "deepinfra", "anthropic", or "openai".
    llm_provider: str = "openrouter"
    # Model id in the provider's namespace. OpenRouter/DeepInfra use slugs like
    # "deepseek/deepseek-v4-pro"; Anthropic uses "claude-sonnet-4-6".
    llm_model: str = "deepseek/deepseek-v4-pro"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096
    # OpenRouter (OpenAI-compatible) credentials.
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # DeepInfra (OpenAI-compatible) credentials.
    deepinfra_api_key: str = ""
    deepinfra_base_url: str = "https://api.deepinfra.com/v1/openai"

    # --- MCP servers (tools) ---
    paperclip_api_key: str = ""
    paperclip_url: str = "https://paperclip.gxl.ai/mcp"
    # Tavily web search via its hosted MCP server (enables web search when set).
    tavily_api_key: str = ""
    tavily_mcp_url: str = "https://mcp.tavily.com/mcp/"
    # Optional JSON file declaring additional MCP servers.
    mcp_config_path: str = "mcp_servers.json"

    # --- Memory ---
    # Postgres DSN, e.g. postgresql://user:pass@localhost:5432/research_agent
    # When empty, memory is disabled and the bot falls back to in-process state.
    database_url: str = ""
    # Embedder for mem0 (Anthropic has no embeddings API). Reads OPENAI_API_KEY.
    # deepinfra provider: set EMBEDDING_MODEL=BAAI/bge-m3 + EMBEDDING_DIMS=1024
    # openrouter / openai: set EMBEDDING_MODEL=text-embedding-3-small + EMBEDDING_DIMS=1536
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

    # --- Draft → critique → revise loop ---
    # Total critique passes per artifact (1 = critique only, no revision;
    # 2 = one revision allowed; <=0 disables critiquing entirely).
    validation_rounds: int = Field(2, ge=0)

    # --- Self-improving lessons (the reflect-and-recall loop) ---
    # Master switch: recall relevant lessons before each subagent job, and reflect
    # the finished job into new lessons afterward.
    lessons_enabled: bool = True
    # How many past lessons to recall and inject into a job (top-K vector search).
    lesson_recall_limit: int = Field(5, ge=1)
    # Cheap model that distills a finished job into reusable lessons. Used only
    # when OPENROUTER_API_KEY is set (build_reflection_llm); otherwise the default
    # agent model is used. Runs once per job in the background, so keep it small.
    reflection_model: str = "qwen/qwen3.7-plus"
    # Max lessons to extract per job.
    reflection_max_lessons: int = Field(3, ge=1)
    # How many candidates to fetch before re-ranking recall (multiplier × limit).
    lesson_recall_oversample: int = Field(3, ge=1)
    # Maintenance: merge near-duplicate lessons when per-kind count exceeds this.
    lesson_consolidation_enabled: bool = True
    max_lessons_per_kind: int = Field(200, ge=10)

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

    # --- Discord message ingestion ---
    # Max size of an uploaded attachment we'll download + parse (bytes).
    attachment_max_bytes: int = 20 * 1024 * 1024
    # Max characters of extracted attachment text injected inline into a turn.
    # The FULL text is always saved as a project artifact; this only bounds what
    # we thread into the orchestrator's context so a large PDF can't blow it up.
    attachment_max_chars: int = 24_000

    # --- Web frontend (FastAPI + React SPA; auth via WorkOS AuthKit) ---
    web_host: str = "0.0.0.0"
    web_port: int = 8800
    # Public base URL the browser hits (used to build the auth redirect URI).
    web_base_url: str = "http://localhost:8800"
    # Secret used to sign the session cookie (set a long random value).
    web_session_secret: str = ""
    # WorkOS AuthKit credentials (https://workos.com/docs/authkit/overview).
    workos_api_key: str = ""
    workos_client_id: str = ""
    # Explicit callback URL (must match the WorkOS dashboard). When empty, it's
    # derived as <WEB_BASE_URL>/auth/callback.
    workos_redirect_uri: str = ""
    # Comma-separated allowlist of emails permitted to sign in ("" = allow any
    # successfully-authenticated WorkOS user).
    web_allowed_emails: str = ""

    # --- Observability: Arize Phoenix (free, self-hosted, local) ---
    # When enabled, agent/graph runs are auto-instrumented and streamed to a
    # local Phoenix collector. `phoenix_endpoint` is the OTLP traces endpoint
    # (e.g. http://localhost:6006/v1/traces); empty uses Phoenix's default.
    phoenix_enabled: bool = False
    phoenix_project: str = "research-agent"
    phoenix_endpoint: str = ""
    # The web app reverse-proxies the Phoenix UI at this path (behind auth), so
    # it's reachable through the frontend without exposing port 6006 publicly.
    phoenix_internal_url: str = "http://localhost:6006"
    phoenix_root_path: str = "/phoenix"

    @property
    def allowed_emails(self) -> list[str]:
        return [e.strip().lower() for e in self.web_allowed_emails.split(",") if e.strip()]

    @property
    def web_redirect_uri(self) -> str:
        # Prefer an explicit WORKOS_REDIRECT_URI; else derive from the base URL.
        return self.workos_redirect_uri or (self.web_base_url.rstrip("/") + "/auth/callback")

    @property
    def web_is_https(self) -> bool:
        return self.web_redirect_uri.startswith("https") or self.web_base_url.startswith("https")

    # --- Per-role model overrides (OpenRouter/DeepInfra only) ---
    # Comma-separated role=model_slug pairs to override the default model for
    # specific subagent roles. E.g. "code_reader=qwen/qwen3.7-plus,methodology=deepseek/deepseek-v4-pro".
    # Known roles: research_literature, code_reader, literature_review, methodology,
    # methodology_validator, paper_draft, paper_verifier, orchestrator.
    # Empty string = use default model for all roles. Overrides are ignored on
    # anthropic/openai providers (requires OpenRouter or DeepInfra).
    role_models: str = ""

    # --- Ideation consortium (two-track scored panel via OpenRouter) ---
    # Comma-separated OpenRouter slugs for the panel — latest reasoning models.
    # (Verify exact slugs on https://openrouter.ai/models.)
    consortium_models: str = (
        "deepseek/deepseek-v4-pro,z-ai/glm-5.1,"
        "qwen/qwen3.7-plus,moonshotai/kimi-k2.6"
    )
    # The chair: builds the brief, ranks the scores, leads the polish debate.
    consortium_chair_model: str = "deepseek/deepseek-r1"
    consortium_temperature: float = 0.6
    # Turns in the shared debate track (turn 1 opens, the rest react).
    consortium_debate_turns: int = 2
    # Back-compat (unused by the two-track flow).
    consortium_rounds: int = 1

    @property
    def panel_models(self) -> list[str]:
        return [m.strip() for m in self.consortium_models.split(",") if m.strip()]

    @property
    def role_model_map(self) -> dict[str, str]:
        """Parse role_models into a dict of role -> model_slug.

        Format: "role1=slug1,role2=slug2,...". Malformed pairs are silently skipped.
        Whitespace around roles and slugs is trimmed.
        """
        result = {}
        if not self.role_models:
            return result
        for pair in self.role_models.split(","):
            pair = pair.strip()
            if not pair or "=" not in pair:
                continue
            try:
                role, slug = pair.split("=", 1)
                role = role.strip()
                slug = slug.strip()
                if role and slug:
                    result[role] = slug
            except Exception:  # noqa: BLE001
                pass
        return result

    # Automatic fix-and-relaunch after a failed run.
    # When > 0 the runner will patch the experiment code from the failure logs
    # and relaunch up to this many times without re-approval.  Retries reuse
    # the exact same JobSpec / resources, so the original human approval covers
    # them.  0 disables the feature entirely.
    experiment_auto_retry: int = 2

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
