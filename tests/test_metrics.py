"""Tests for metric engines (non-LLM metrics only)."""

from evalkit.metrics.exact_match import ExactMatchMetric
from evalkit.models import MetricType


class TestExactMatch:
    def setup_method(self):
        self.metric = ExactMatchMetric()

    def test_exact_match_identical(self):
        result = self.metric.score("hello world", "hello world")
        assert result.score == 1.0
        assert result.passed is True

    def test_exact_match_case_insensitive(self):
        result = self.metric.score("Hello World", "hello world")
        assert result.score == 1.0

    def test_exact_match_whitespace_normalization(self):
        result = self.metric.score("hello  world", "hello world")
        assert result.score == 1.0

    def test_exact_match_failure(self):
        result = self.metric.score("hello", "goodbye")
        assert result.score == 0.0
        assert result.passed is False

    def test_exact_match_ignore_punctuation(self):
        result = self.metric.score(
            "hello, world!", "hello world", params={"ignore_punctuation": True}
        )
        assert result.score == 1.0

    def test_exact_match_strict(self):
        result = self.metric.score(
            "Hello", "hello", params={"ignore_case": False, "normalize": False}
        )
        assert result.score == 0.0
