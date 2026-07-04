# Deep Reports SDK

Multi-provider, multi-orchestrator research report generator powered by LLM agents.  
Analyze source code, answer research questions, and produce structured technical reports — all configurable via environment variables, TOML, or CLI.

## Features

- **4 LLM providers**: Anthropic, OpenAI, LiteLLM, Ollama
- **3 orchestrators**: native (zero-dep), LangGraph, CrewAI
- **4 agents**: generator → evidence critic → quality critic → refiner (tri-agent escalation)
- **Cost control**: BudgetGuard with per-stage sub-budgets & env-var model pricing
- **Path security**: traverse guard with `allowed_source_roots`
- **Retry + timeout**: automatic retry via tenacity (optional) + configurable timeout
- **Fallback chain**: `FallbackProvider` tries providers in order on failure
- **CLI + Python API**: `deep-reports generate` or `DeepReport` class

## Install

```bash
pip install deep-reports-sdk
```

### Extras

```bash
pip install "deep-reports-sdk[anthropic]"   # Anthropic provider
pip install "deep-reports-sdk[openai]"       # OpenAI provider
pip install "deep-reports-sdk[litellm]"      # LiteLLM (50+ backends)
pip install "deep-reports-sdk[ollama]"       # Ollama local
pip install "deep-reports-sdk[langgraph]"    # LangGraph orchestrator
pip install "deep-reports-sdk[crewai]"       # CrewAI orchestrator
pip install "deep-reports-sdk[all]"          # everything
pip install "deep-reports-sdk[dev]"          # development (pytest, ruff, mypy)
```

### From source

```bash
git clone https://github.com/tnt305/deep-reports-sdk.git
cd deep-reports-sdk
pip install -e ".[dev]"
```

## Quick Start

### CLI

```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Generate a report
deep-reports generate \
  --source ./my-project/src \
  --question "What is the core architecture?" \
  --provider anthropic \
  --output ./reports/
```

### Python

```python
from deep_reports import DeepReport

dr = DeepReport(
    sources=["./my-project/src"],
    question="What is the core primitive?",
    provider="anthropic",
    framework="native",
    max_cost_usd=1.0,
)
result = dr.generate()
print(result["report_md"])
print(f"Cost: ${result['cost_usd']:.4f}")
```

## Configuration

Priority (highest first):
1. CLI flags / constructor kwargs
2. Environment variables (`DR_*` prefix)
3. `pyproject.toml` `[tool.deep-reports]` section
4. `.env` file
5. Defaults

### Key env vars

| Variable | Default | Description |
|---|---|---|
| `DR_PROVIDER` | `anthropic` | `anthropic`, `openai`, `litellm`, `ollama` |
| `DR_MODEL` | *(provider default)* | Model name override |
| `DR_FRAMEWORK` | `native` | `native`, `langgraph`, `crewai` |
| `DR_MAX_COST_USD` | `1.0` | Budget cap |
| `DR_REQUEST_TIMEOUT` | `120.0` | LLM API timeout (seconds) |
| `DR_LOG_LEVEL` | `INFO` | Logging level |
| `DR_ALLOWED_SOURCE_ROOTS` | `.` | Comma-separated allowed paths |
| `DR_PROVIDER_FALLBACK` | *(empty)* | Comma-separated fallback chain |
| `DR_AGENT_MODEL__GENERATOR` | *(global model)* | Per-agent model override |
| `DR_MODEL_PRICE_IN` | *(pricing table)* | Custom input price per 1K tokens |
| `DR_MODEL_PRICE_OUT` | *(pricing table)* | Custom output price per 1K tokens |

### pyproject.toml

```toml
[tool.deep-reports]
provider = "anthropic"
framework = "native"
agents = "all"
max-cost = 1.0

[tool.deep-reports.agent-models]
generator = "claude-sonnet-4-5"
critic-evidence = "claude-haiku-4-5"
critic-quality = "claude-haiku-4-5"
```

## Provider Setup

### Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
```

### LiteLLM

```bash
export LITELLM_API_KEY=...
# Uses litellm.completion() — supports 50+ backends
```

### Ollama (local)

```bash
# Start Ollama, then:
ollama pull llama3
# The SDK uses raw HTTP — no SDK dependency
```

### Fallback Chain

```bash
export DR_PROVIDER_FALLBACK=anthropic,openai
# tries Anthropic first, falls back to OpenAI on failure
```

## Orchestrators

| Framework | Deps | Use case |
|---|---|---|
| `native` | none | Simple pipelines, no extra deps |
| `langgraph` | `langgraph` | Complex branching, state graphs |
| `crewai` | `crewai` | Multi-agent teams with roles |

## Agent Pipeline

```
generator ──→ evidence_critic ──→ quality_critic ──→ END
                  │                    │
                  ▼ (≥3 blockers)       ▼ (score < 5)
               refiner ◄────────────── refiner
                  │
                  └──→ evidence_critic (max 3 iterations)
```

| Agent | Role |
|---|---|
| `generator` | Writes report from sources + question |
| `evidence_critic` | Validates every citation against source files |
| `quality_critic` | Scores report on 8 dimensions (1-10) |
| `refiner` | Revises report based on critic feedback |

Agent sets: `all` (4 agents), `core` (generator + evidence), `minimal` (generator only).

## Cost Control

```python
from deep_reports.cost import BudgetGuard

guard = BudgetGuard(
    max_usd=1.0,
    sub_budgets={"generator": 0.30, "critic": 0.10},
)
cost = guard.add("claude-sonnet-4-5", input_tokens=500, output_tokens=200)
```

Per-model pricing via env vars:
```bash
DR_MODEL_PRICE_CLAUDE_SONNET_4_5_IN=0.003
DR_MODEL_PRICE_CLAUDE_SONNET_4_5_OUT=0.015
```

## Security

Path traversal is blocked by default. Only paths under `allowed_source_roots` are readable.

```bash
# Allow reading from specific directories
export DR_ALLOWED_SOURCE_ROOTS=/home/user/projects,/tmp/work
```

Symlinks pointing outside allowed roots are rejected.

## API Reference

### `DeepReport`

```python
DeepReport(
    sources: list[str],          # source files/dirs to analyze
    question: str,               # research question
    provider: str | None = None, # "anthropic" | "openai" | "litellm" | "ollama"
    framework: str = "native",   # "native" | "langgraph" | "crewai"
    model: str | None = None,    # model override
    max_cost_usd: float = 1.0,   # budget cap
    agents: str = "all",         # "all" | "core" | "minimal"
    config: DeepReportsConfig | None = None,  # full config object
)
```

Returns `PipelineState` dict:
- `report_md` — generated report (Markdown)
- `cost_usd` — total cost
- `evidence_issues` — list of citation issues
- `quality_scores` — per-dimension scores
- `quality_summary` — overall assessment

### CLI

```bash
deep-reports generate --source PATH [--source PATH ...] --question Q [options]
deep-reports version
```

## Development

```bash
pip install -e ".[dev]"
ruff check src/
mypy src/
pytest tests/unit/
```

## License

MIT
