"""Anthropic SDK adapter."""

from __future__ import annotations

import os
import time
from typing import Any

from deep_reports.cost import BudgetGuard
from deep_reports.providers.base import ProviderResponse
from deep_reports.providers import call_with_retry


class AnthropicProvider:
    """Anthropic Messages API adapter."""

    name = "anthropic"
    default_model = "claude-sonnet-4-5"

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 120.0):
        import anthropic

        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._client = anthropic.Anthropic(api_key=self._api_key, timeout=timeout)
        self._async_client = anthropic.AsyncAnthropic(api_key=self._api_key, timeout=timeout)
        self._default_model = model or self.default_model

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        budget: BudgetGuard | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        model = model or self._default_model
        max_tokens = max_tokens or 4096

        start = time.monotonic()
        response = call_with_retry(
            self._client.messages.create,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
            **kwargs,
        )
        latency_ms = (time.monotonic() - start) * 1000

        # Normalize: list[TextBlock] → plain str
        text_parts = [
            block.text for block in response.content
            if hasattr(block, "text")
        ]
        content = "\n".join(text_parts)

        cost = 0.0
        if budget is not None:
            cost = budget.add(
                model,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )

        return ProviderResponse(
            content=content,
            model=response.model or model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=cost,
            provider=self.name,
            latency_ms=latency_ms,
        )

    async def acomplete(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        budget: BudgetGuard | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        model = model or self._default_model
        max_tokens = max_tokens or 4096

        start = time.monotonic()
        response = await call_with_retry(
            self._async_client.messages.create,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
            **kwargs,
        )
        latency_ms = (time.monotonic() - start) * 1000

        text_parts = [
            block.text for block in response.content
            if hasattr(block, "text")
        ]
        content = "\n".join(text_parts)

        cost = 0.0
        if budget is not None:
            cost = budget.add(
                model,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )

        return ProviderResponse(
            content=content,
            model=response.model or model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=cost,
            provider=self.name,
            latency_ms=latency_ms,
        )
