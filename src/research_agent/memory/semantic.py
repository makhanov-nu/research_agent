"""Semantic memory: facts, via mem0 over pgvector.

Stores durable facts about the user, projects, and findings — each tagged with
provenance (source + timestamp) so recalled claims stay verifiable. mem0's
built-in entity linking gives lightweight graph/entity structure with no extra
service. Single global pool: one logical owner id (settings.memory_user_id).

Degrades to a no-op when memory isn't configured (no DB / no embedder key).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..config import settings

logger = logging.getLogger(__name__)


def _build_config() -> dict:
    pg = settings.pg_components()
    return {
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "host": pg["host"],
                "port": pg["port"],
                "user": pg["user"],
                "password": pg["password"],
                "dbname": pg["dbname"],
                "collection_name": settings.mem0_collection,
                "embedding_model_dims": settings.embedding_dims,
                "diskann": False,
                "hnsw": True,
            },
        },
        "llm": {
            "provider": "anthropic",
            "config": {
                "model": settings.llm_model,
                "temperature": 0.1,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": settings.embedding_model,
                "embedding_dims": settings.embedding_dims,
            },
        },
    }


class SemanticMemory:
    def __init__(self):
        self._mem = None
        self._enabled = settings.memory_enabled and bool(
            settings.openai_api_key or _openai_key_in_env()
        )

    @property
    def enabled(self) -> bool:
        return self._enabled and self._mem is not None

    def setup(self) -> None:
        """Instantiate the mem0 client (synchronous; call off the event loop)."""
        if not self._enabled:
            logger.warning(
                "Semantic memory disabled (need DATABASE_URL + OPENAI_API_KEY)."
            )
            return
        try:
            from mem0 import Memory

            self._mem = Memory.from_config(_build_config())
            logger.info("Semantic memory (mem0/pgvector) ready.")
        except Exception:  # noqa: BLE001
            logger.exception("Failed to initialize mem0; semantic memory off.")
            self._enabled = False

    def remember(self, user_text: str, agent_text: str, source: str | None = None,
                 extra: dict | None = None) -> None:
        """Extract and store durable facts from one exchange, with provenance."""
        if not self.enabled:
            return
        metadata = {"as_of": datetime.now(timezone.utc).isoformat()}
        if source:
            metadata["source"] = source
        if extra:
            metadata.update(extra)
        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": agent_text},
        ]
        try:
            self._mem.add(messages, user_id=settings.memory_user_id, metadata=metadata)
        except Exception:  # noqa: BLE001
            logger.exception("Semantic memory write failed.")

    def recall(self, query: str, limit: int = 5) -> str:
        """Return relevant facts as a citation-annotated block for the prompt."""
        if not self.enabled:
            return ""
        try:
            res = self._mem.search(query, user_id=settings.memory_user_id, limit=limit)
        except Exception:  # noqa: BLE001
            logger.exception("Semantic memory search failed.")
            return ""

        results = res.get("results", res) if isinstance(res, dict) else res
        lines = []
        for item in results or []:
            text = item.get("memory") or item.get("text") or ""
            meta = item.get("metadata") or {}
            src = meta.get("source")
            cite = f" [source: {src}]" if src else ""
            if text:
                lines.append(f"- {text}{cite}")
        return "\n".join(lines)


def _openai_key_in_env() -> bool:
    import os

    return bool(os.environ.get("OPENAI_API_KEY"))
