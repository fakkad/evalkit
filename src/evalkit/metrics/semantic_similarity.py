"""Semantic similarity metric -- OpenAI embeddings + cosine similarity."""

from __future__ import annotations

import math
import os
from typing import Any

import httpx

from evalkit.metrics.base import Metric
from evalkit.models import MetricResult

# In-memory embedding cache keyed by (model, text)
_embedding_cache: dict[tuple[str, str], list[float]] = {}


async def _get_embedding(
    client: httpx.AsyncClient,
    text: str,
    model: str,
    api_key: str,
) -> list[float]:
    """Get embedding for text, using cache if available."""
    cache_key = (model, text)
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    response = await client.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"input": text, "model": model},
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json()
    embedding = data["data"][0]["embedding"]
    _embedding_cache[cache_key] = embedding
    return embedding


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticSimilarityMetric(Metric):
    """Embed both strings via OpenAI embeddings API, compute cosine similarity."""

    async def score(
        self,
        input: str,
        output: str,
        expected: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> MetricResult:
        params = params or {}
        model = params.get("model", "text-embedding-3-small")
        api_key = os.environ.get("OPENAI_API_KEY", "")

        if expected is None:
            return MetricResult(
                metric_name="semantic_similarity",
                score=0.0,
                details={"error": "no expected_output provided"},
            )

        if not api_key:
            return MetricResult(
                metric_name="semantic_similarity",
                score=0.0,
                details={"error": "OPENAI_API_KEY not set"},
            )

        async with httpx.AsyncClient() as client:
            emb_expected = await _get_embedding(client, expected, model, api_key)
            emb_output = await _get_embedding(client, output, model, api_key)

        similarity = _cosine_similarity(emb_expected, emb_output)
        # Clamp to [0, 1]
        similarity = max(0.0, min(1.0, similarity))

        return MetricResult(
            metric_name="semantic_similarity",
            score=similarity,
            details={
                "cosine_similarity": similarity,
                "embedding_model": model,
            },
        )
