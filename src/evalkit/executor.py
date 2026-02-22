"""Model executor — calls target model endpoint."""

from __future__ import annotations

import os
import time
from typing import Any

import anthropic


def call_model(
    input_text: str,
    model: str = "claude-sonnet-4-20250514",
    system_prompt: str | None = None,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Call the target model and return response with metadata."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    messages = [{"role": "user", "content": input_text}]
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    start = time.perf_counter()
    response = client.messages.create(**kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000

    text = response.content[0].text
    tokens = response.usage.input_tokens + response.usage.output_tokens

    return {
        "text": text,
        "latency_ms": elapsed_ms,
        "tokens_used": tokens,
    }
