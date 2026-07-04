"""LiteLLM router adapter."""

from __future__ import annotations

import os
import time
from typing import Any

from deep_reports.cost import BudgetGuard
from deep_reports.providers.base import ProviderResponse
from deep_reports.providers import call_with_retry


class LiteLLMProvider:
    """LiteLLM router adapter — routes to 50+ backends via unified API."""

    name = "litellm"
    default_model = "claude-sonnet-4-5"

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 120.0):
        import litellm

        self._api_key = api_key or os.getenv("LITELLM_API_KEY", "")
        self._default_model = model or self.default_model
        self._timeout = timeout
        # Configure litellm with API key
        litellm.drop_params = True

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
        import litellm

        model = model or self._default_model
        start = time.monotonic()

        all_messages: list[dict[str, str]] = (
            [{"role": "system", "content": system}] + list(messages) if system else list(messages)
        )

        response = call_with_retry(
            litellm.completion,
            model=model,
            messages=all_messages,
            temperature=temperature,
            max_tokens=max_tokens or 4096,
            api_key=self._api_key or None,
            timeout=self._timeout,
            **kwargs,
        )
        latency_ms = (time.monotonic() - start) * 1000

        # LiteLLM normalizes content to str already
        raw = response.choices[0].message.content
        content = raw if isinstance(raw, str) else str(raw)

        usage = response.usage
        in_tok = getattr(usage, "prompt_tokens", 0)
        out_tok = getattr(usage, "completion_tokens", 0)

        cost = 0.0
        if budget is not None and (in_tok or out_tok):
            cost = budget.add(model, in_tok, out_tok)

        return ProviderResponse(
            content=content,
            model=getattr(response, "model", model),
            input_tokens=in_tok,
            output_tokens=out_tok,
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
        import litellm

        model = model or self._default_model
        start = time.monotonic()

        all_messages: list[dict[str, str]] = (
            [{"role": "system", "content": system}] + list(messages) if system else list(messages)
        )

        response = await call_with_retry(
            litellm.acompletion,
            model=model,
            messages=all_messages,
            temperature=temperature,
            max_tokens=max_tokens or 4096,
            api_key=self._api_key or None,
            timeout=self._timeout,
            **kwargs,
        )
        latency_ms = (time.monotonic() - start) * 1000

        raw = response.choices[0].message.content
        content = raw if isinstance(raw, str) else str(raw)

        usage = response.usage
        in_tok = getattr(usage, "prompt_tokens", 0)
        out_tok = getattr(usage, "completion_tokens", 0)

        cost = 0.0
        if budget is not None and (in_tok or out_tok):
            cost = budget.add(model, in_tok, out_tok)

        return ProviderResponse(
            content=content,
            model=getattr(response, "model", model),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            provider=self.name,
            latency_ms=latency_ms,
        )
