"""Unit tests for providers/__init__.py — FallbackProvider + factory."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from deep_reports.providers import (
    FallbackProvider,
    AllProvidersFailed,
    get_provider,
)
from deep_reports.providers.base import ProviderResponse


def _make_resp(content: str, provider: str = "test") -> ProviderResponse:
    return ProviderResponse(
        content=content,
        model="test-model",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        provider=provider,
    )


class TestFallbackProvider:
    def test_returns_first_success(self):
        p1 = MagicMock()
        p1.name = "p1"
        p1.complete.return_value = _make_resp("from p1")
        p2 = MagicMock()
        p2.name = "p2"
        p2.complete.return_value = _make_resp("from p2")

        fb = FallbackProvider([p1, p2])
        resp = fb.complete(messages=[{"role": "user", "content": "hi"}])

        assert resp.content == "from p1"
        assert p2.complete.call_count == 0  # p1 succeeded, p2 not called

    def test_falls_back_on_first_failure(self):
        p1 = MagicMock()
        p1.name = "p1"
        p1.complete.side_effect = RuntimeError("p1 down")
        p2 = MagicMock()
        p2.name = "p2"
        p2.complete.return_value = _make_resp("from p2")

        fb = FallbackProvider([p1, p2])
        resp = fb.complete(messages=[{"role": "user", "content": "hi"}])

        assert resp.content == "from p2"
        assert p1.complete.call_count == 1
        assert p2.complete.call_count == 1

    def test_raises_all_failed(self):
        p1 = MagicMock()
        p1.name = "p1"
        p1.complete.side_effect = RuntimeError("p1 down")
        p2 = MagicMock()
        p2.name = "p2"
        p2.complete.side_effect = RuntimeError("p2 down")

        fb = FallbackProvider([p1, p2])
        with pytest.raises(AllProvidersFailed) as exc:
            fb.complete(messages=[{"role": "user", "content": "hi"}])

        errors = exc.value.args[0]
        assert "p1" in str(errors)
        assert "p2" in str(errors)

    def test_passes_kwargs_to_providers(self):
        p1 = MagicMock()
        p1.name = "p1"
        p1.complete.side_effect = RuntimeError("fail")

        p2 = MagicMock()
        p2.name = "p2"
        p2.complete.return_value = _make_resp("ok")

        fb = FallbackProvider([p1, p2])
        fb.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o",
            temperature=0.7,
        )

        p2.complete.assert_called_once()
        call_kwargs = p2.complete.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["temperature"] == 0.7

    def test_name_and_default_model(self):
        fb = FallbackProvider([MagicMock()])
        assert fb.name == "fallback"
        assert fb.default_model == "claude-sonnet-4-5"


class TestGetProvider:
    def test_get_anthropic_provider(self):
        p = get_provider("anthropic")
        assert p.name == "anthropic"

    def test_get_openai_provider(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-for-initialization")
        p = get_provider("openai")
        assert p.name == "openai"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown-provider")

    def test_get_provider_with_model_override(self):
        p = get_provider("anthropic", model="claude-opus-4")
        assert p._default_model == "claude-opus-4"
