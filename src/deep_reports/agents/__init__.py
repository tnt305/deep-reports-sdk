"""Agent registry — factory for named agents."""

from __future__ import annotations

from deep_reports.agents.base import Agent, AgentContext, AgentResult


# Registry of available agents
_AGENTS: dict[str, type[Agent]] = {}


def _register_agents():
    """Register all built-in agents."""
    from deep_reports.agents.generator import ReportGenerator
    from deep_reports.agents.critic_evidence import EvidenceCritic
    from deep_reports.agents.critic_quality import QualityCritic
    from deep_reports.agents.refiner import Refiner

    for cls in [ReportGenerator, EvidenceCritic, QualityCritic, Refiner]:
        _AGENTS[cls.name] = cls


def get_agent(name: str) -> Agent:
    """Return an agent instance by name."""
    if not _AGENTS:
        _register_agents()
    if name not in _AGENTS:
        raise ValueError(
            f"Unknown agent {name!r}. Available: {list(_AGENTS.keys())}"
        )
    return _AGENTS[name]()


def list_agents() -> list[str]:
    """Return list of registered agent names."""
    if not _AGENTS:
        _register_agents()
    return list(_AGENTS.keys())


__all__ = [
    "Agent",
    "AgentContext",
    "AgentResult",
    "get_agent",
    "list_agents",
]
