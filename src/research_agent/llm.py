"""Provider-agnostic LLM factory.

LangGraph is provider-agnostic, so the model is chosen here and bound to tools
in the graph. Swap providers via LLM_PROVIDER / LLM_MODEL in the environment.
Supported providers: "openrouter", "deepinfra", "anthropic", "openai".
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from .config import settings


def _openai_compat(
    model: str, temperature: float, max_tokens: int, api_key: str, base_url: str
) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key or None,
        base_url=base_url,
    )


def get_llm() -> BaseChatModel:
    """Build the chat model described by the current settings."""
    provider = settings.llm_provider.lower()

    if provider == "openrouter":
        return _openai_compat(
            settings.llm_model, settings.llm_temperature, settings.llm_max_tokens,
            settings.openrouter_api_key, settings.openrouter_base_url,
        )

    if provider == "deepinfra":
        return _openai_compat(
            settings.llm_model, settings.llm_temperature, settings.llm_max_tokens,
            settings.deepinfra_api_key, settings.deepinfra_base_url,
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
        "Supported: 'openrouter', 'deepinfra', 'anthropic', 'openai'."
    )


def build_openrouter_chat(
    model: str, temperature: float = 0.6, max_tokens: int = 4096
) -> BaseChatModel:
    """Build a chat model for a specific model slug using the configured provider.

    Used by the consortium and reflection step. Routes to OpenRouter or DeepInfra
    depending on LLM_PROVIDER; both are OpenAI-compatible with the same model slugs.
    """
    provider = settings.llm_provider.lower()
    if provider == "deepinfra":
        return _openai_compat(model, temperature, max_tokens,
                              settings.deepinfra_api_key, settings.deepinfra_base_url)
    return _openai_compat(model, temperature, max_tokens,
                          settings.openrouter_api_key, settings.openrouter_base_url)


def build_reflection_llm() -> BaseChatModel:
    """Cheap model for the per-job reflection step (lesson distillation).

    Uses the configured provider when it's OpenRouter or DeepInfra (both have
    the reflection model); falls back to the default agent model otherwise.
    """
    provider = settings.llm_provider.lower()
    if provider in ("openrouter", "deepinfra"):
        return build_openrouter_chat(
            settings.reflection_model, temperature=0.2, max_tokens=512
        )
    return get_llm()
