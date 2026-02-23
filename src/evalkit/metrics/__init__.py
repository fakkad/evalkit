"""Metric engines for EvalKit."""

from evalkit.metrics.base import Metric
from evalkit.metrics.exact_match import ExactMatchMetric
from evalkit.metrics.semantic_similarity import SemanticSimilarityMetric
from evalkit.metrics.llm_judge import LLMJudgeMetric
from evalkit.metrics.rubric import RubricMetric

METRIC_REGISTRY: dict[str, type[Metric]] = {
    "exact_match": ExactMatchMetric,
    "semantic_similarity": SemanticSimilarityMetric,
    "llm_judge": LLMJudgeMetric,
    "rubric": RubricMetric,
}

__all__ = [
    "Metric",
    "ExactMatchMetric",
    "SemanticSimilarityMetric",
    "LLMJudgeMetric",
    "RubricMetric",
    "METRIC_REGISTRY",
]
