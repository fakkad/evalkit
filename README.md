# EvalKit

CLI eval harness for LLMs. Define test suites in YAML, score outputs with 4 metric types, enforce pass/fail thresholds. HTML report + JSON results.

## Install

```bash
pip install -e .
```

## Quick Start

```bash
# Generate an example suite
evalkit init --output suite.yaml

# Validate a suite
evalkit validate suite.yaml

# Run evals (requires ANTHROPIC_API_KEY / OPENAI_API_KEY)
evalkit run suite.yaml --output-dir ./results --format both

# Compare two runs
evalkit compare results1.json results2.json --output diff.html
```

## YAML Config Format

```yaml
name: "my-eval-suite"
description: "Tests for customer support bot"
model:
  provider: "anthropic"  # or "openai"
  model: "claude-sonnet-4-20250514"
  params:
    temperature: 0
    max_tokens: 1024
test_cases:
  - id: "greeting-1"
    input: "Hello, I need help with my order"
    expected_output: "I'd be happy to help you with your order"
    metadata:
      category: "greeting"
metrics:
  - type: "exact_match"
  - type: "semantic_similarity"
    params:
      model: "text-embedding-3-small"
  - type: "llm_judge"
    params:
      criteria: "helpfulness"
      model: "claude-haiku-4-5-20251001"
  - type: "rubric"
    params:
      rubric:
        - criterion: "tone"
          description: "Response is professional and empathetic"
          weight: 0.4
        - criterion: "accuracy"
          description: "Response addresses the user's actual request"
          weight: 0.6
thresholds:
  exact_match: 0.5
  semantic_similarity: 0.8
  llm_judge: 0.7
  rubric: 0.75
```

## Metrics

| Metric | Description |
|--------|-------------|
| `exact_match` | Normalized string comparison (lowercase, strip whitespace, optional fuzzy) |
| `semantic_similarity` | OpenAI embeddings API + cosine similarity, cached |
| `llm_judge` | G-Eval style: prompt + response + criteria to judge LLM, score 1-5, normalize to 0-1 |
| `rubric` | Multi-criteria weighted rubric: each criterion scored 1-5 by LLM, weighted average |

## CLI Commands

```
evalkit run <suite.yaml> [--output-dir ./results] [--format json|html|both]
evalkit compare <results1.json> <results2.json> [--output report.html]
evalkit validate <suite.yaml>
evalkit init [--output suite.yaml]
```

Exit code 0 = all thresholds met. Exit code 1 = threshold violations.

## Environment Variables

- `ANTHROPIC_API_KEY` -- required for Anthropic provider and Claude-based judge/rubric
- `OPENAI_API_KEY` -- required for OpenAI provider and semantic similarity embeddings

## License

MIT
