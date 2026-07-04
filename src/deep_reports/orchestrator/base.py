"""Orchestrator base — Protocol + PipelineState."""

from __future__ import annotations

from typing import Any, Callable, TypedDict



class PipelineState(TypedDict, total=False):
    """
    State flowing through the pipeline between agents.
    All fields are optional (TypedDict total=False) so intermediate stages
    can progressively populate them.
    """
    question: str
    sources: list[str]
    report_md: str | None
    critic_verdict: str | None
    evidence_issues: list[dict[str, Any]]
    quality_scores: dict[str, float] | None
    quality_summary: str | None
    quality_overall: float | None
    cost_usd: float
    iteration: int
    checkpoint_dir: str | None  # set by facade for A6 checkpointing


class Orchestrator:
    """
    Base class for orchestrator implementations.
    Subclass to implement Native, LangGraph, or CrewAI backends.

    All orchestrators MUST implement:
      - add_node(name, fn)
      - add_edge(src, dst)
      - add_conditional_edge(src, router, path_map)
      - run(initial, max_iterations) -> PipelineState
    """

    framework: str = "base"

    def add_node(
        self,
        name: str,
        fn: Callable[..., PipelineState],
    ) -> None:
        """Register a node (agent function) in the graph."""
        raise NotImplementedError

    def add_edge(self, src: str, dst: str) -> None:
        """Add a directed edge from src to dst."""
        raise NotImplementedError

    def add_conditional_edge(
        self,
        src: str,
        router: Callable[[dict[str, Any]], str],
        path_map: dict[str, str],
    ) -> None:
        """
        Add a conditional edge. Router returns a key from path_map,
        which maps to the next node name.
        """
        raise NotImplementedError

    def run(
        self,
        initial: PipelineState,
        *,
        max_iterations: int = 5,
    ) -> PipelineState:
        """Execute the pipeline starting from initial state."""
        raise NotImplementedError
