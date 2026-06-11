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


def _embedder_block() -> dict:
    """mem0 embedder config.

    openrouter: OpenRouter's OpenAI-compatible embeddings endpoint
                (model slug needs openai/ prefix for OpenAI models).
    deepinfra:  DeepInfra's own embeddings endpoint — use EMBEDDING_MODEL /
                EMBEDDING_DIMS in .env to pick the model (e.g. BAAI/bge-m3 / 1024).
    other:      bare OpenAI API (OPENAI_API_KEY required).
    """
    provider = settings.llm_provider.lower()
    model = settings.embedding_model
    if provider == "openrouter":
        return {
            "provider": "openai",
            "config": {
                "model": model if "/" in model else f"openai/{model}",
                "embedding_dims": settings.embedding_dims,
                "api_key": settings.openrouter_api_key,
                "openai_base_url": settings.openrouter_base_url,
            },
        }
    if provider == "deepinfra":
        return {
            "provider": "openai",
            "config": {
                "model": model,
                "embedding_dims": settings.embedding_dims,
                "api_key": settings.deepinfra_api_key,
                "openai_base_url": settings.deepinfra_base_url,
            },
        }
    return {
        "provider": "openai",
        "config": {"model": model, "embedding_dims": settings.embedding_dims},
    }


def _llm_block() -> dict:
    """mem0 LLM config mirroring the agent's provider.

    OpenRouter and DeepInfra are OpenAI-compatible; mem0's `openai` provider
    with a custom base URL handles both.
    """
    provider = settings.llm_provider.lower()
    base = {"model": settings.llm_model, "temperature": 0.1, "max_tokens": 2000}

    if provider == "openrouter":
        return {
            "provider": "openai",
            "config": {
                **base,
                "api_key": settings.openrouter_api_key,
                "openai_base_url": settings.openrouter_base_url,
            },
        }
    if provider == "deepinfra":
        return {
            "provider": "openai",
            "config": {
                **base,
                "api_key": settings.deepinfra_api_key,
                "openai_base_url": settings.deepinfra_base_url,
            },
        }
    if provider == "openai":
        return {"provider": "openai", "config": base}
    return {"provider": "anthropic", "config": base}


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
        "llm": _llm_block(),
        "embedder": _embedder_block(),
    }


def _has_embedder_credentials() -> bool:
    """True when we have a key for the embeddings endpoint in use."""
    provider = settings.llm_provider.lower()
    if provider == "openrouter":
        return bool(settings.openrouter_api_key)
    if provider == "deepinfra":
        return bool(settings.deepinfra_api_key)
    return bool(settings.openai_api_key or _openai_key_in_env())


class SemanticMemory:
    def __init__(self):
        self._mem = None
        self._enabled = settings.memory_enabled and _has_embedder_credentials()

    @property
    def enabled(self) -> bool:
        return self._enabled and self._mem is not None

    def setup(self) -> None:
        """Instantiate the mem0 client (synchronous; call off the event loop)."""
        if not self._enabled:
            logger.warning(
                "Semantic memory disabled (need DATABASE_URL + an embeddings key)."
            )
            return
        try:
            import os
            from mem0 import Memory

            # mem0's OpenAILLM hard-checks OPENROUTER_API_KEY in os.environ at
            # instantiation and routes ALL LLM calls through OpenRouter when it's
            # present — ignoring our configured base_url/api_key. Hide it briefly
            # so mem0 uses the provider-specific endpoint we configured above.
            _saved_or_key = os.environ.pop("OPENROUTER_API_KEY", None)
            try:
                self._mem = Memory.from_config(_build_config())
            finally:
                if _saved_or_key is not None:
                    os.environ["OPENROUTER_API_KEY"] = _saved_or_key

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
            {"role": "user", "content": user_text.replace("\x00", "")},
            {"role": "assistant", "content": agent_text.replace("\x00", "")},
        ]
        try:
            self._mem.add(messages, user_id=settings.memory_user_id, metadata=metadata)
        except Exception:  # noqa: BLE001
            logger.exception("Semantic memory write failed.")

    def add_fact(self, text: str, source: str | None = None,
                 metadata: dict | None = None, infer: bool = False) -> str | None:
        """Store a single fact/lesson VERBATIM (infer=False) with provenance.

        Used for consolidated lessons/insights we don't want mem0 to re-extract
        or paraphrase — e.g. an experiment failure lesson or a council insight.

        Returns the mem0 memory id on success, or None on failure/disabled.
        """
        if not self.enabled:
            return None
        meta = {"as_of": datetime.now(timezone.utc).isoformat()}
        if source:
            meta["source"] = source
        if metadata:
            meta.update(metadata)
        try:
            result = self._mem.add(
                text, user_id=settings.memory_user_id, metadata=meta, infer=infer
            )
            # mem0 returns {"results": [{"id": ..., ...}, ...]} or a list
            return _extract_first_id(result)
        except Exception:  # noqa: BLE001
            logger.exception("Semantic fact write failed.")
            return None

    def delete_fact(self, memory_id: str) -> None:
        """Delete a single memory entry by id. Best-effort, no-op if disabled."""
        if not self.enabled or not memory_id:
            return
        try:
            self._mem.delete(memory_id)
        except Exception:  # noqa: BLE001
            logger.exception("Semantic fact delete failed for id=%s.", memory_id)

    def recall(self, query: str, limit: int = 5, only_type: str | None = None,
               only_kind: str | None = None) -> str:
        """Return relevant facts as a citation-annotated block for the prompt.

        `only_type`/`only_kind` keep only results whose metadata `type`/`kind`
        match (e.g. type="lesson", kind="literature"); filtered client-side so it
        works across mem0 versions. When filtering, we over-fetch then trim.
        """
        text_lines, _ = self._recall_inner(
            query, limit, only_type, only_kind, score_map=None
        )
        return "\n".join(text_lines)

    def recall_with_ids(
        self,
        query: str,
        limit: int = 5,
        only_type: str | None = None,
        only_kind: str | None = None,
        score_map: dict[str, float] | None = None,
    ) -> tuple[str, list[str]]:
        """Like recall() but also returns the mem0 ids of the returned items.

        `score_map` is an optional {memory_id: float} prior used to re-rank the
        candidates (e.g. Laplace-smoothed quality score from LessonStats). When
        provided, candidates are fetched at 3× the requested limit, sorted by
        blended score (0.5 * vector_relevance + 0.5 * quality_prior), and the
        top `limit` are returned.  Blending weights are equal and arbitrary — this
        is a soft prior only, not a calibrated ranker.

        Returns (formatted_block: str, ids: list[str]).
        """
        text_lines, ids = self._recall_inner(
            query, limit, only_type, only_kind, score_map=score_map
        )
        return "\n".join(text_lines), ids

    def _recall_inner(
        self,
        query: str,
        limit: int,
        only_type: str | None,
        only_kind: str | None,
        score_map: dict[str, float] | None,
    ) -> tuple[list[str], list[str]]:
        """Shared implementation for recall() and recall_with_ids()."""
        query = query.replace("\x00", "")
        if not self.enabled:
            return [], []
        scoped = only_type or only_kind
        # Over-fetch when we need to filter or re-rank.
        fetch_limit = limit * 3 if (scoped or score_map) else limit
        try:
            res = self._mem.search(
                query, filters={"user_id": settings.memory_user_id},
                limit=fetch_limit,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Semantic memory search failed.")
            return [], []

        raw = res.get("results", res) if isinstance(res, dict) else res
        # Filter first (cheap), then re-rank with score_map if provided.
        candidates: list[tuple[str, str, float]] = []  # (text_line, id, relevance)
        for idx, item in enumerate(raw or []):
            meta = item.get("metadata") or {}
            if only_type and meta.get("type") != only_type:
                continue
            if only_kind and meta.get("kind") != only_kind:
                continue
            text = item.get("memory") or item.get("text") or ""
            if not text:
                continue
            src = meta.get("source")
            cite = f" [source: {src}]" if src else ""
            line = f"- {text}{cite}"
            mem_id = item.get("id") or ""
            # mem0 may include a relevance/score field; fall back to rank-based.
            rel = item.get("score") or item.get("relevance") or max(0.0, 1.0 - idx * 0.1)
            candidates.append((line, mem_id, float(rel)))

        if score_map and candidates:
            # Blend vector relevance with quality prior using equal weights.
            candidates.sort(
                key=lambda c: 0.5 * c[2] + 0.5 * score_map.get(c[1], 0.5),
                reverse=True,
            )

        top = candidates[:limit]
        return [c[0] for c in top], [c[1] for c in top]


def _openai_key_in_env() -> bool:
    import os

    return bool(os.environ.get("OPENAI_API_KEY"))


def _extract_first_id(result) -> str | None:
    """Best-effort extraction of the first memory id from a mem0 add() response.

    mem0 returns varying shapes across versions:
      - {"results": [{"id": "...", ...}]}   (2.x)
      - [{"id": "..."}]                     (some builds)
      - {"id": "..."}                       (older 1.x)
    Returns None when no id can be found so callers degrade gracefully.
    """
    if not result:
        return None
    if isinstance(result, dict):
        if "id" in result:
            return str(result["id"])
        results = result.get("results") or []
        if results and isinstance(results[0], dict):
            return str(results[0].get("id", "")) or None
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return str(first.get("id", "")) or None
    return None
