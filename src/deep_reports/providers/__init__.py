"""Provider factory with fallback chain support (A1)."""

from __future__ import annotations

import logging
from typing import Any

from deep_reports.cost import BudgetGuard
from deep_reports.providers.base import Provider, ProviderResponse


logger = logging.getLogger("deep_reports.providers")


def _import_tenacity():
    try:
        from tenacity import retry, stop, stop_after_attempt, wait_exponential
        return retry, stop, stop_after_attempt, wait_exponential
    except ImportError:
        return None, None, None, None


_retry_kwargs = dict(stop=None, wait=None)


def _get_retry_kwargs():
    """Return tenacity kwargs if tenacity is installed, else None."""
    global _retry_kwargs
    if _retry_kwargs["stop"] is None:
        retry_mod, stop_mod, stop_after_attempt_mod, wait_exponential_mod = _import_tenacity()
        if retry_mod is not None:
            _retry_kwargs = {
                "stop": stop_after_attempt_mod(max_attempt_number=3),
                "wait": wait_exponential_mod(multiplier=1, min=2, max=10),
            }
    return _retry_kwargs


def call_with_retry(fn, *args, **kwargs):
    """Call fn with retry if tenacity is available, otherwise direct call."""
    kw = _get_retry_kwargs()
    if kw["stop"] is not None:
        from tenacity import retry as _tenacity_retry
        return _tenacity_retry(**kw)(fn)(*args, **kwargs)
    return fn(*args, **kwargs)


# --- FallbackProvider (A1) ---
class AllProvidersFailed(Exception):
    """Raised when all providers in the fallback chain have failed."""

    pass


class FallbackProvider:
    """
    Try providers in order; return first success. Wraps any number of providers.

    Usage:
        provider = FallbackProvider([
            get_provider("anthropic"),
            get_provider("openai"),
        ])
        resp = provider.complete(messages)
        # → tries anthropic; on failure, tries openai; on both failure, raises AllProvidersFailed
    """

    name = "fallback"
    default_model = "claude-sonnet-4-5"

    def __init__(self, providers: list[Provider]):
        self._providers = providers

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
        errors: list[str] = []
        for p in self._providers:
            try:
                return p.complete(
                    messages,
                    system=system,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    budget=budget,
                    **kwargs,
                )
            except Exception as e:  # noqa: BLE001
                errors.append(f"{p.name}: {e}")
                logger.warning(f"Provider {p.name} failed: {e}")
                continue
        raise AllProvidersFailed(errors)

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
        errors: list[str] = []
        for p in self._providers:
            try:
                return await p.acomplete(
                    messages,
                    system=system,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    budget=budget,
                    **kwargs,
                )
            except Exception as e:  # noqa: BLE001
                errors.append(f"{p.name}: {e}")
                logger.warning(f"Provider {p.name} failed: {e}")
                continue
        raise AllProvidersFailed(errors)


# --- Module-level budget guard (legacy single-threaded default) ---
_default_budget_guard: BudgetGuard | None = None


def set_default_budget_guard(guard: BudgetGuard | None) -> BudgetGuard | None:
    """Set the shared BudgetGuard for providers that don't receive one explicitly."""
    global _default_budget_guard
    prev = _default_budget_guard
    _default_budget_guard = guard
    return prev


def get_default_budget_guard() -> BudgetGuard | None:
    """Get the shared BudgetGuard."""
    return _default_budget_guard


# --- Factory ---
def get_provider(
    name: str,
    **kw: Any,
) -> Provider:
    """
    Return a provider instance by name.

    Supported names: "anthropic", "openai", "litellm", "ollama"
    Pass DR_PROVIDER_FALLBACK=anthropic,openai to get a FallbackProvider.
    """
    import os

    fallback_names = os.getenv("DR_PROVIDER_FALLBACK", "")
    if fallback_names:
        names = [n.strip() for n in fallback_names.split(",") if n.strip()]
        providers_list = [get_provider(n, **kw) for n in names]
        return FallbackProvider(providers_list)

    if name == "anthropic":
        from deep_reports.providers.anthropic import AnthropicProvider
        return AnthropicProvider(**kw)
    if name == "openai":
        from deep_reports.providers.openai import OpenAIProvider
        return OpenAIProvider(**kw)
    if name == "litellm":
        from deep_reports.providers.litellm import LiteLLMProvider
        return LiteLLMProvider(**kw)
    if name == "ollama":
        from deep_reports.providers.ollama import OllamaProvider
        return OllamaProvider(**kw)
    raise ValueError(f"Unknown provider {name!r}. Supported: anthropic, openai, litellm, ollama")
