"""Provider-agnostic LLM factory.

LangGraph is provider-agnostic, so the model is chosen here and bound to tools
in the graph. Swap providers via LLM_PROVIDER / LLM_MODEL in the environment.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from .config import settings


def get_llm() -> BaseChatModel:
    """Build the chat model described by the current settings."""
    provider = settings.llm_provider.lower()

    if provider == "openrouter":
        # OpenRouter is OpenAI-compatible; route via ChatOpenAI + base_url.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.openrouter_api_key or None,
            base_url=settings.openrouter_base_url,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER={settings.llm_provider!r}. "
        "Supported: 'openrouter', 'anthropic', 'openai'."
    )
