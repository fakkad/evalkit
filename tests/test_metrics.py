"""Tests for metric engines."""

import pytest

from evalkit.metrics.exact_match import ExactMatchMetric
from evalkit.metrics.semantic_similarity import SemanticSimilarityMetric, _cosine_similarity
from evalkit.metrics.llm_judge import LLMJudgeMetric
from evalkit.metrics.rubric import RubricMetric


class TestExactMatch:
    @pytest.fixture
    def metric(self):
        return ExactMatchMetric()

    @pytest.mark.asyncio
    async def test_identical_strings(self, metric):
        result = await metric.score(
            input="test", output="hello world", expected="hello world"
        )
        assert result.score == 1.0
        assert result.metric_name == "exact_match"

    @pytest.mark.asyncio
    async def test_case_insensitive(self, metric):
        result = await metric.score(
            input="test", output="Hello World", expected="hello world"
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_whitespace_normalization(self, metric):
        result = await metric.score(
            input="test", output="hello  world", expected="hello world"
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_mismatch(self, metric):
        result = await metric.score(
            input="test", output="goodbye", expected="hello"
        )
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_ignore_punctuation(self, metric):
        result = await metric.score(
            input="test",
            output="hello world",
            expected="hello, world!",
            params={"ignore_punctuation": True},
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_strict_mode(self, metric):
        result = await metric.score(
            input="test",
            output="hello",
            expected="Hello",
            params={"ignore_case": False, "normalize": False},
        )
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_no_expected(self, metric):
        result = await metric.score(input="test", output="hello")
        assert result.score == 0.0
        assert "error" in result.details

    @pytest.mark.asyncio
    async def test_fuzzy_matching(self, metric):
        result = await metric.score(
            input="test",
            output="hello world",
            expected="hello worl",
            params={"fuzzy_threshold": 0.8},
        )
        assert result.score > 0.8


class TestCosineUtil:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 1.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) + 1.0) < 1e-6
