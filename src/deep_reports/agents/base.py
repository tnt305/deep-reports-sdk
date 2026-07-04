"""Agent base — Protocol + AgentContext + AgentResult."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from deep_reports.cost import BudgetGuard
    from deep_reports.providers.base import Provider


# Re-export budget guard helpers from providers module
from deep_reports.providers import (  # noqa: F401
    set_default_budget_guard,
    get_default_budget_guard,
)


class AgentResult(BaseModel):
    """
    Result returned by an agent after invocation.
    """
    content: str
    structured: dict | None = None  # parsed JSON if agent returns structured output
    cost_usd: float
    tokens: dict[str, int]
    agent_name: str


class AgentContext(BaseModel):
    """
    Shared context flowing into every agent invocation.
    """
    state: dict[str, Any]  # PipelineState — question, sources, partial report_md, etc.
    history: list[dict[str, str]] = []  # message history
    budget: "BudgetGuard | None" = None
    provider: "Provider"

    def get_budget(self) -> "BudgetGuard | None":
        """
        Resolve budget: explicit AgentContext.budget wins;
        falls back to the module-level default set by set_default_budget_guard().
        """
        if self.budget is not None:
            return self.budget
        return get_default_budget_guard()


@runtime_checkable
class Agent(Protocol):
    """
    Protocol for report pipeline agents.

    All agents must implement:
      - name: unique identifier
      - role: pipeline role ("generator", "evidence_critic", "quality_critic", "refiner")
      - persona: system-prompt description of the agent's role
      - invoke(ctx: AgentContext) -> AgentResult
    """

    name: str
    role: str
    persona: str

    def invoke(self, ctx: AgentContext) -> AgentResult:
        """Run the agent on the given context."""
        ...
