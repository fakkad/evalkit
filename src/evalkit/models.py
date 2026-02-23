"""Core pydantic models for EvalKit."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """A single evaluation test case."""

    id: str
    input: str
    expected_output: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricConfig(BaseModel):
    """Configuration for a metric within a test suite."""

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class RubricCriterion(BaseModel):
    """A single criterion within a rubric metric."""

    criterion: str
    description: str
    weight: float = 1.0


class ModelConfig(BaseModel):
    """LLM provider and model configuration."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    params: dict[str, Any] = Field(default_factory=dict)


class TestSuite(BaseModel):
    """A collection of test cases loaded from YAML."""

    name: str
    description: str = ""
    model: ModelConfig = Field(default_factory=ModelConfig)
    test_cases: list[TestCase] = Field(default_factory=list)
    metrics: list[MetricConfig] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)


class MetricResult(BaseModel):
    """Result from a single metric evaluation."""

    metric_name: str
    score: float  # 0.0 to 1.0
    details: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Result for a single test case."""

    test_case_id: str
    model_response: str = ""
    metric_results: list[MetricResult] = Field(default_factory=list)
    passed: bool = True


class ThresholdViolation(BaseModel):
    """A threshold violation for a metric."""

    metric_name: str
    expected: float
    actual: float


class SuiteResult(BaseModel):
    """Aggregate result for an entire eval suite run."""

    suite_name: str
    description: str = ""
    model_config_used: ModelConfig = Field(default_factory=ModelConfig)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    results: list[EvalResult] = Field(default_factory=list)
    aggregate_scores: dict[str, float] = Field(default_factory=dict)
    threshold_violations: list[ThresholdViolation] = Field(default_factory=list)
    passed: bool = True
    total_cases: int = 0
    duration_ms: float = 0.0
