"""Local, self-hosted tracing via Arize Phoenix (free; no SaaS).

Phoenix runs on your own box (a single container / `phoenix serve`) and stores
traces locally. We register an OpenTelemetry tracer that auto-instruments
LangChain/LangGraph (via OpenInference), so every agent + graph run streams to
the local Phoenix collector and is viewable in its UI — a LangSmith-like trace
view without paying for or self-hosting LangSmith.

`setup_tracing()` is a no-op unless `PHOENIX_ENABLED=true`, and never raises:
tracing must not take the app down.
"""

from __future__ import annotations

import logging

from .config import settings

logger = logging.getLogger(__name__)

_initialized = False


def setup_tracing() -> bool:
    """Enable Phoenix tracing if configured. Returns True when active."""
    global _initialized
    if _initialized or not settings.phoenix_enabled:
        return _initialized
    try:
        from phoenix.otel import register

        register(
            project_name=settings.phoenix_project,
            endpoint=settings.phoenix_endpoint or None,
            auto_instrument=True,  # picks up openinference-instrumentation-langchain
        )
        _initialized = True
        logger.info(
            "Phoenix tracing on (project=%s, collector=%s).",
            settings.phoenix_project, settings.phoenix_endpoint or "default",
        )
    except Exception:  # noqa: BLE001 — observability is best-effort
        logger.exception(
            "Phoenix tracing failed to start (install the 'obs' extra and run a "
            "Phoenix server); continuing without it."
        )
    return _initialized
