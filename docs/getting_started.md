# Getting Started

## Install

```bash
pip install -e ".[dev]"        # development
pip install -e ".[all]"         # all extras (Anthropic, OpenAI, LangGraph, CrewAI)
```

## Set API Key

```bash
export ANTHROPIC_API_KEY=your-api-key-here
export OPENAI_API_KEY=your-api-key-here
```

## CLI Usage

```bash
deep-reports generate \
  --source ./my-repo \
  --question "What is the core primitive?" \
  --output ./reports/ \
  --max-cost 0.50
```

## Python API

```python
from deep_reports import DeepReport

dr = DeepReport(
    sources=["./my-repo/"],
    question="What is the core primitive?",
    provider="anthropic",
    framework="native",
)
result = dr.generate()
print(result["report_md"])
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DR_PROVIDER` | `anthropic` | LLM provider |
| `DR_MODEL` | provider default | Model override |
| `DR_FRAMEWORK` | `native` | Orchestrator |
| `DR_MAX_COST_USD` | `1.0` | Budget cap |
| `DR_ALLOWED_SOURCE_ROOTS` | `.` | Allowed source paths |

### Per-Agent Models

```bash
export DR_AGENT_MODELS__GENERATOR=claude-sonnet-4-5
export DR_AGENT_MODELS__CRITIC_QUALITY=claude-haiku-4-5
```

### TOML Config (pyproject.toml)

```toml
[tool.deep-reports]
provider = "anthropic"
framework = "native"
max-cost = 0.50

[tool.deep-reports.agent-models]
generator = "claude-sonnet-4-5"
critic_quality = "claude-haiku-4-5"
```
