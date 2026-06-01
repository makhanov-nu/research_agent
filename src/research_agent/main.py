"""Entrypoint: run the Discord-facing research agent."""

from __future__ import annotations

import logging

from .config import settings
from .discord_bot import ResearchBot


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not settings.discord_token:
        raise SystemExit(
            "DISCORD_TOKEN is not set. Copy .env.example to .env and fill it in."
        )

    bot = ResearchBot()
    bot.run(settings.discord_token, log_handler=None)


if __name__ == "__main__":
    main()
