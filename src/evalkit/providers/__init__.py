"""LLM providers for EvalKit."""

from evalkit.providers.base import Provider
from evalkit.providers.anthropic import AnthropicProvider
from evalkit.providers.openai import OpenAIProvider

PROVIDER_REGISTRY: dict[str, type[Provider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}

__all__ = ["Provider", "AnthropicProvider", "OpenAIProvider", "PROVIDER_REGISTRY"]
