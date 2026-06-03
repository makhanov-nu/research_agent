"""Entrypoint for the web frontend: `research-agent-web`."""

from __future__ import annotations

import logging

from ..config import settings


def main() -> None:
    import uvicorn

    from ..observability import setup_tracing

    logging.basicConfig(level=logging.INFO)
    setup_tracing()
    uvicorn.run(
        "research_agent.web.app:create_app",
        factory=True,
        host=settings.web_host,
        port=settings.web_port,
    )


if __name__ == "__main__":
    main()
