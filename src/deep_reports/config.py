"""Configuration — pydantic Settings with env var + TOML support."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


AgentList = Literal["all", "core", "minimal"]
Framework = Literal["native", "langgraph", "crewai"]
Provider = Literal["anthropic", "openai", "litellm", "ollama"]


class DeepReportsConfig(BaseSettings):
    """
    Configuration for Deep Reports SDK.

    Load priority (highest to lowest):
      1. CLI overrides (applied via with_overrides())
      2. Environment variables (DR_* prefix)
      3. pyproject.toml [tool.deep-reports] section
      4. .env file
      5. Defaults below

    Per-agent model overrides can be set via:
      DR_AGENT_MODEL__GENERATOR=claude-sonnet-4
      DR_AGENT_MODEL__CRITIC_QUALITY=claude-haiku-4-5
    """

    model_config = SettingsConfigDict(
        env_prefix="DR_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    provider: Provider = "anthropic"
    model: str = ""  # empty → provider's default
    framework: Framework = "native"
    agents: AgentList = "all"
    max_cost_usd: float = 1.0
    log_level: str = "INFO"

    # Per-agent model overrides (e.g. DR_AGENT_MODEL__GENERATOR=claude-sonnet-4)
    agent_models: dict[str, str] = {}

    # Sub-budgets for cost control (e.g. {"generator": 0.30, "critic": 0.10})
    sub_budgets: dict[str, float] = {}

    # Allowed source roots for path security
    allowed_source_roots: list[str] = ["."]

    # Output directory
    output_dir: str = "./reports/"

    # Request timeout in seconds for LLM API calls
    request_timeout: float = 120.0

    # Cache directory (Phase 5)
    cache_dir: str = "./.deep-reports-cache/"

    # Evaluation mode (Phase 5)
    eval_mode: Literal["critics", "deepeval", "all"] = "critics"

    def with_overrides(self, **kw: object) -> "DeepReportsConfig":
        """Create a new config with CLI overrides applied."""
        merged = dict(self.model_dump())
        merged.update(kw)
        return DeepReportsConfig(**merged)

    def get_agent_model(self, agent_name: str) -> str:
        """Resolve model for an agent: per-agent override → global model → provider default."""
        if agent_name in self.agent_models and self.agent_models[agent_name]:
            return self.agent_models[agent_name]
        if self.model:
            return self.model
        return PROVIDER_DEFAULTS.get(self.provider, "")

    @classmethod
    def from_toml(cls, path: str | Path) -> "DeepReportsConfig":
        """Load from a TOML file's [tool.deep-reports] section."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore

        with open(path, "rb") as f:
            data = tomllib.load(f)

        section = data.get("tool", {}).get("deep-reports", {})
        agent_models_raw = section.get("agent-models", {})
        sub_budgets_raw = section.get("sub-budgets", {})

        return cls(
            provider=section.get("provider", "anthropic"),
            framework=section.get("framework", "native"),
            agents=section.get("agents", "all"),
            max_cost_usd=float(section.get("max-cost", 1.0)),
            output_dir=section.get("output-dir", "./reports/"),
            agent_models=agent_models_raw,
            sub_budgets={k: float(v) for k, v in sub_budgets_raw.items()},
        )


PROVIDER_DEFAULTS: dict[Provider, str] = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o-mini",
    "litellm": "claude-sonnet-4-5",
    "ollama": "llama3",
}
