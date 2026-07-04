"""Unit tests for config.py — no API keys needed."""

from __future__ import annotations

import pytest


class TestDeepReportsConfig:
    def test_defaults(self):
        from deep_reports.config import DeepReportsConfig

        cfg = DeepReportsConfig()
        assert cfg.provider == "anthropic"
        assert cfg.framework == "native"
        assert cfg.max_cost_usd == 1.0
        assert cfg.log_level == "INFO"
        assert cfg.allowed_source_roots == ["."]

    def test_env_override(self):
        from deep_reports.config import DeepReportsConfig

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("DR_PROVIDER", "openai")
            mp.setenv("DR_MAX_COST_USD", "2.5")
            cfg = DeepReportsConfig()
            assert cfg.provider == "openai"
            assert cfg.max_cost_usd == 2.5

    def test_with_overrides(self):
        from deep_reports.config import DeepReportsConfig

        cfg = DeepReportsConfig()
        cfg2 = cfg.with_overrides(provider="openai", max_cost_usd=0.5)
        assert cfg.provider == "anthropic"  # original unchanged
        assert cfg2.provider == "openai"
        assert cfg2.max_cost_usd == 0.5

    def test_get_agent_model_per_agent_override(self):
        from deep_reports.config import DeepReportsConfig

        cfg = DeepReportsConfig(
            model="claude-sonnet-4-5",
            agent_models={"generator": "claude-opus-4"},
        )
        assert cfg.get_agent_model("generator") == "claude-opus-4"  # per-agent wins
        assert cfg.get_agent_model("evidence_critic") == "claude-sonnet-4-5"  # global fallback

    def test_get_agent_model_empty_global(self):
        from deep_reports.config import DeepReportsConfig

        cfg = DeepReportsConfig(provider="openai")
        # no global model set
        assert cfg.get_agent_model("generator") == "gpt-4o-mini"  # provider default

    def test_provider_defaults(self):
        from deep_reports.config import PROVIDER_DEFAULTS

        assert PROVIDER_DEFAULTS["anthropic"] == "claude-sonnet-4-5"
        assert PROVIDER_DEFAULTS["openai"] == "gpt-4o-mini"
        assert PROVIDER_DEFAULTS["ollama"] == "llama3"

    def test_sub_budgets(self):
        from deep_reports.config import DeepReportsConfig

        cfg = DeepReportsConfig(
            sub_budgets={"generator": 0.3, "critic": 0.1},
        )
        assert cfg.sub_budgets == {"generator": 0.3, "critic": 0.1}

    def test_agent_model_via_nested_env(self):
        from deep_reports.config import DeepReportsConfig

        # DR_AGENT_MODELS__GENERATOR → agent_models["generator"] (lowercase key)
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("DR_AGENT_MODELS__GENERATOR", "claude-opus-4")
            cfg = DeepReportsConfig()
            assert cfg.agent_models.get("generator") == "claude-opus-4"

    def test_agent_models_direct_assignment(self):
        from deep_reports.config import DeepReportsConfig

        cfg = DeepReportsConfig(
            agent_models={"generator": "claude-opus-4", "critic_quality": "claude-haiku-4-5"},
        )
        assert cfg.get_agent_model("generator") == "claude-opus-4"
        assert cfg.get_agent_model("critic_quality") == "claude-haiku-4-5"
        # evidence_critic not set → falls back to provider default
        assert cfg.get_agent_model("evidence_critic") == "claude-sonnet-4-5"

    def test_from_toml_missing_file(self, tmp_path):
        from deep_reports.config import DeepReportsConfig

        with pytest.raises(FileNotFoundError):
            DeepReportsConfig.from_toml(tmp_path / "missing.toml")
