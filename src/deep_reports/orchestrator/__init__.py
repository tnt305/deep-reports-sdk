"""Orchestrator factory."""

from __future__ import annotations

from deep_reports.orchestrator.base import Orchestrator, PipelineState


def get_orchestrator(framework: str) -> Orchestrator:
    """
    Return an orchestrator instance by framework name.

    Supported: "native" (default, no extras), "langgraph", "crewai".
    """
    if framework == "native":
        from deep_reports.orchestrator.native import NativeOrchestrator
        return NativeOrchestrator()
    if framework == "langgraph":
        from deep_reports.orchestrator.langgraph_adapter import LangGraphOrchestrator
        return LangGraphOrchestrator()
    if framework == "crewai":
        from deep_reports.orchestrator.crewai_adapter import CrewAIOrchestrator
        return CrewAIOrchestrator()
    raise ValueError(
        f"Unknown framework {framework!r}. Supported: native, langgraph, crewai"
    )


__all__ = ["Orchestrator", "PipelineState", "get_orchestrator"]
