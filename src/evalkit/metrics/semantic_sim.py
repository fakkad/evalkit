"""Semantic similarity metric — sentence-transformers cosine similarity."""

from __future__ import annotations

from typing import Any

from evalkit.models import MetricResult, MetricType

_model_cache: dict[str, Any] = {}


def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load sentence transformer model."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


class SemanticSimMetric:
    """Compare actual output to expected via cosine similarity of embeddings."""

    metric_type = MetricType.SEMANTIC_SIM

    def score(
        self,
        expected: str,
        actual: str,
        threshold: float = 0.8,
        params: dict[str, Any] | None = None,
    ) -> MetricResult:
        params = params or {}
        model_name = params.get("model", "all-MiniLM-L6-v2")

        model = _get_model(model_name)
        embeddings = model.encode([expected, actual], normalize_embeddings=True)
        cosine_sim = float(embeddings[0] @ embeddings[1])

        return MetricResult(
            metric_type=self.metric_type,
            score=cosine_sim,
            passed=cosine_sim >= threshold,
            threshold=threshold,
            details={
                "cosine_similarity": cosine_sim,
                "model": model_name,
            },
        )
