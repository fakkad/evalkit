"""Anthropic Messages API provider via httpx."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from evalkit.providers.base import Provider

_API_URL = "https://api.anthropic.com/v1/messages"
_MAX_RETRIES = 3


class AnthropicProvider(Provider):
    """Anthropic provider using the Messages API."""

    def __init__(self, model: str, params: dict[str, Any] | None = None):
        super().__init__(model, params)
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.temperature = self.params.get("temperature", 0)
        self.max_tokens = self.params.get("max_tokens", 1024)

    async def generate(self, prompt: str) -> str:
        """Call Anthropic Messages API with exponential backoff retry."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        async with httpx.AsyncClient() as client:
            for attempt in range(_MAX_RETRIES):
                try:
                    response = await client.post(
                        _API_URL,
                        headers=headers,
                        json=payload,
                        timeout=120.0,
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["content"][0]["text"]
                except (httpx.HTTPStatusError, httpx.RequestError) as e:
                    if attempt == _MAX_RETRIES - 1:
                        raise RuntimeError(
                            f"Anthropic API failed after {_MAX_RETRIES} retries: {e}"
                        ) from e
                    wait = 2 ** attempt
                    await asyncio.sleep(wait)

        raise RuntimeError("Unreachable")
