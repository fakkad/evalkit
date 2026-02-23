"""Abstract base class for LLM providers."""

from __future__ import annotations

import abc
from typing import Any


class Provider(abc.ABC):
    """Base class for LLM providers. All providers are async with retry."""

    def __init__(self, model: str, params: dict[str, Any] | None = None):
        self.model = model
        self.params = params or {}

    @abc.abstractmethod
    async def generate(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the response text.

        Implementations should include retry with exponential backoff.
        """
        ...
