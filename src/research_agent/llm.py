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


def build_openrouter_chat(
    model: str, temperature: float = 0.6, max_tokens: int = 4096
) -> BaseChatModel:
    """Build a ChatOpenAI bound to a specific OpenRouter model slug.

    Used by the ideation consortium to talk to several models through the one
    OpenRouter key, independent of the agent's default provider.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=settings.openrouter_api_key or None,
        base_url=settings.openrouter_base_url,
    )


def build_reflection_llm() -> BaseChatModel:
    """Cheap model for the per-job reflection step (lesson distillation).

    Prefers a small OpenRouter model (independent of the agent's provider) so the
    once-per-job reflection stays cheap; falls back to the default agent model
    when OpenRouter isn't configured.
    """
    if settings.openrouter_api_key:
        return build_openrouter_chat(
            settings.reflection_model, temperature=0.2, max_tokens=512
        )
    return get_llm()
