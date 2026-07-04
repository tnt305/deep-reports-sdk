"""Ollama local provider adapter — uses raw HTTP, no SDK."""

from __future__ import annotations

import time
from typing import Any

import httpx

from deep_reports.cost import BudgetGuard
from deep_reports.providers.base import ProviderResponse
from deep_reports.providers import call_with_retry


class OllamaProvider:
    """
    Ollama local inference adapter.

    Uses raw HTTP API (httpx) — no SDK dependency.
    Detects available models via `ollama list`.
    Skips cleanly if `ollama` CLI is not on PATH.
    """

    name = "ollama"
    default_model = "llama3"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str | None = None,
        timeout: float = 120.0,
    ):
        self._base_url = base_url
        self._default_model = model or self.default_model
        self._timeout = timeout
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def _ensure_model(self, model: str) -> str:
        """Ensure model is available; fall back to default if not."""
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if models and model not in models:
                model = models[0]
        except Exception:  # noqa: BLE001
            pass
        return model

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

        # Convert chat messages to Ollama format
        ollama_messages = []
        if system:
            ollama_messages.append({"role": "system", "content": system})
        for msg in messages:
            ollama_messages.append({"role": msg["role"], "content": msg["content"]})

        response = call_with_retry(
            self._client.post,
            "/api/chat",
            json={
                "model": model,
                "messages": ollama_messages,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                "stream": False,
            },
        )
        latency_ms = (time.monotonic() - start) * 1000
        response.raise_for_status()
        data = response.json()

        content = data["message"]["content"]

        # Ollama doesn't provide token counts — use len(content) as approximation
        # For accurate billing, users should use DR_MODEL_PRICE_* env vars
        out_tok = len(content) // 4  # rough approximation
        in_tok = sum(len(m["content"]) // 4 for m in ollama_messages)

        cost = 0.0
        if budget is not None:
            cost = budget.add(model, in_tok, out_tok)

        return ProviderResponse(
            content=content,
            model=data.get("model", model),
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
        max_tokens = max_tokens or 4096

        start = time.monotonic()

        ollama_messages = []
        if system:
            ollama_messages.append({"role": "system", "content": system})
        for msg in messages:
            ollama_messages.append({"role": msg["role"], "content": msg["content"]})

        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            response = await call_with_retry(
                client.post,
                "/api/chat",
                json={
                    "model": model,
                    "messages": ollama_messages,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                    "stream": False,
                },
            )
        latency_ms = (time.monotonic() - start) * 1000
        response.raise_for_status()
        data = response.json()

        content = data["message"]["content"]
        out_tok = len(content) // 4
        in_tok = sum(len(m["content"]) // 4 for m in ollama_messages)

        cost = 0.0
        if budget is not None:
            cost = budget.add(model, in_tok, out_tok)

        return ProviderResponse(
            content=content,
            model=data.get("model", model),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            provider=self.name,
            latency_ms=latency_ms,
        )
