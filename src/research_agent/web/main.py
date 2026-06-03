"""Entrypoint for the web frontend: `research-agent-web`."""

from __future__ import annotations

import logging

from ..config import settings


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "research_agent.web.app:create_app",
        factory=True,
        host=settings.web_host,
        port=settings.web_port,
    )


if __name__ == "__main__":
    main()
