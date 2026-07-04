"""OpenAI SDK adapter."""

from __future__ import annotations

import os
import time
from typing import Any

from openai import OpenAI, AsyncOpenAI

from deep_reports.cost import BudgetGuard
from deep_reports.providers.base import ProviderResponse
from deep_reports.providers import call_with_retry


class OpenAIProvider:
    """OpenAI Chat Completions API adapter."""

    name = "openai"
    default_model = "gpt-4o-mini"

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 120.0):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._client = OpenAI(api_key=self._api_key, timeout=timeout)
        self._async_client = AsyncOpenAI(api_key=self._api_key, timeout=timeout)
        self._default_model = model or self.default_model

    def _build_messages(
        self,
        messages: list[dict[str, str]],
        system: str | None,
    ) -> list[dict[str, str]]:
        """Prepend system prompt if provided."""
        if system:
            return [{"role": "system", "content": system}] + list(messages)
        return list(messages)

    def _normalize_content(self, content: str | list[Any]) -> str:
        """Normalize OpenAI content to plain str."""
        if isinstance(content, str):
            return content
        # list[ContentPartText] — extract text
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            elif hasattr(item, "text"):
                parts.append(item.text)
        return "\n".join(parts)

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
        built = self._build_messages(messages, system)

        start = time.monotonic()
        response = call_with_retry(
            self._client.chat.completions.create,
            model=model,
            messages=built,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        latency_ms = (time.monotonic() - start) * 1000

        raw = response.choices[0].message.content
        content = self._normalize_content(raw)

        in_tok = response.usage.prompt_tokens if response.usage else 0
        out_tok = response.usage.completion_tokens if response.usage else 0

        cost = 0.0
        if budget is not None and (in_tok or out_tok):
            cost = budget.add(model, in_tok, out_tok)

        return ProviderResponse(
            content=content,
            model=response.model or model,
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
        model = model or self._default_model
        built = self._build_messages(messages, system)

        start = time.monotonic()
        response = await call_with_retry(
            self._async_client.chat.completions.create,
            model=model,
            messages=built,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        latency_ms = (time.monotonic() - start) * 1000

        raw = response.choices[0].message.content
        content = self._normalize_content(raw)

        in_tok = response.usage.prompt_tokens if response.usage else 0
        out_tok = response.usage.completion_tokens if response.usage else 0

        cost = 0.0
        if budget is not None and (in_tok or out_tok):
            cost = budget.add(model, in_tok, out_tok)

        return ProviderResponse(
            content=content,
            model=response.model or model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            provider=self.name,
            latency_ms=latency_ms,
        )
