"""Provider Protocol and response type."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from deep_reports.cost import BudgetGuard


class ProviderResponse(BaseModel):
    """
    Uniform return from all providers. `content` is ALWAYS a plain `str`.

    Each adapter is responsible for extracting `.text` from native block
    structures (Anthropic's list of TextBlock, OpenAI's content blocks, etc.)
    before returning. This is a HARD CONTRACT — downstream code (agents, facade,
    CLI) calls `.content.strip()` and `json.loads(content)`.
    """

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float = 0.0
    provider: str
    latency_ms: float = 0.0


@runtime_checkable
class Provider(Protocol):
    """
    Protocol for LLM provider adapters.

    All implementations MUST:
      - Return `ProviderResponse.content` as a plain `str` (never list/raw blocks)
      - Wire BudgetGuard into cost tracking (raise BudgetExceeded if over cap)
      - Support both sync `complete()` and async `acomplete()` methods
    """

    name: str
    default_model: str

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        budget: "BudgetGuard | None" = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """
        Synchronous LLM completion.

        Args:
            messages:      List of {"role": "user"|"assistant", "content": str}
            system:        System prompt (prepended to messages)
            model:         Override the default model
            temperature:   Sampling temperature (default 0.0)
            max_tokens:    Max output tokens (default None = provider default)
            budget:        BudgetGuard for cost tracking (REQUIRED in production)

        Returns:
            ProviderResponse with `content` as plain `str`.
        """
        ...

    async def acomplete(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        budget: "BudgetGuard | None" = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Async LLM completion — mirrors `complete()`."""
        ...
