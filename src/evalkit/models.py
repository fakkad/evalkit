"""Core data models for EvalKit."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MetricType(str, Enum):
    EXACT_MATCH = "exact_match"
    SEMANTIC_SIM = "semantic_sim"
    LLM_JUDGE = "llm_judge"
    RUBRIC = "rubric"


class MetricConfig(BaseModel):
    """Configuration for a single metric within a test case."""

    type: MetricType
    weight: float = 1.0
    threshold: float = 0.5
    params: dict[str, Any] = Field(default_factory=dict)


class TestCase(BaseModel):
    """A single evaluation test case."""

    id: str
    input: str
    expected: str | None = None
    system_prompt: str | None = None
    metrics: list[MetricConfig]
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalSuite(BaseModel):
    """A collection of test cases loaded from YAML."""

    schema_version: int = 1
    name: str
    description: str = ""
    model: str = "claude-sonnet-4-20250514"
    default_system_prompt: str | None = None
    cases: list[TestCase]
    suite_pass_rate: float = 1.0  # fraction of cases that must pass


class MetricResult(BaseModel):
    """Result from a single metric evaluation."""

    metric_type: MetricType
    score: float  # 0.0 to 1.0
    passed: bool
    threshold: float
    details: dict[str, Any] = Field(default_factory=dict)


class TestResult(BaseModel):
    """Result for a single test case."""

    case_id: str
    input: str
    expected: str | None = None
    actual: str
    metric_results: list[MetricResult]
    passed: bool  # AND logic: all metrics must pass
    weighted_score: float
    latency_ms: float = 0.0
    tokens_used: int = 0


class EvalResult(BaseModel):
    """Aggregate result for an entire eval suite run."""

    schema_version: int = 1
    suite_name: str
    model: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    results: list[TestResult]
    pass_rate: float
    passed: bool  # suite_pass_rate met
    total_cases: int
    passed_cases: int
    failed_cases: int
    mean_score: float
    total_latency_ms: float = 0.0
    total_tokens: int = 0
