"""OpenAI Chat Completions API provider via httpx."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from evalkit.providers.base import Provider

_API_URL = "https://api.openai.com/v1/chat/completions"
_MAX_RETRIES = 3


class OpenAIProvider(Provider):
    """OpenAI provider using the Chat Completions API."""

    def __init__(self, model: str, params: dict[str, Any] | None = None):
        super().__init__(model, params)
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.temperature = self.params.get("temperature", 0)
        self.max_tokens = self.params.get("max_tokens", 1024)

    async def generate(self, prompt: str) -> str:
        """Call OpenAI Chat Completions API with exponential backoff retry."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
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
                    return data["choices"][0]["message"]["content"]
                except (httpx.HTTPStatusError, httpx.RequestError) as e:
                    if attempt == _MAX_RETRIES - 1:
                        raise RuntimeError(
                            f"OpenAI API failed after {_MAX_RETRIES} retries: {e}"
                        ) from e
                    wait = 2 ** attempt
                    await asyncio.sleep(wait)

        raise RuntimeError("Unreachable")
