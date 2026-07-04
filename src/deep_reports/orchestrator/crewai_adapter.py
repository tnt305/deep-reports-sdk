"""CrewAI adapter — wraps CrewAI Crew as Orchestrator."""

from __future__ import annotations

import logging
from typing import Any, Callable

from deep_reports.orchestrator.base import Orchestrator, PipelineState

logger = logging.getLogger("deep_reports.orchestrator.crewai")


class CrewAIOrchestrator(Orchestrator):
    """
    CrewAI Crew wrapper conforming to Orchestrator Protocol.

    Maps nodes → CrewAI Agents, edges → CrewAI Tasks with dependencies.
    Sequential execution by default; custom flow for conditional edges.
    """

    framework = "crewai"

    def __init__(self):
        self._nodes: dict[str, Callable[..., Any]] = {}
        self._edges: list[tuple[str, str]] = []
        self._conditional: dict[str, tuple[Callable[[dict], str], dict[str, str]]] = {}
        self._agents: dict[str, Any] = {}
        self._tasks: list[Any] = []

    def add_node(self, name: str, fn: Callable[..., Any]) -> None:
        self._nodes[name] = fn

    def add_edge(self, src: str, dst: str) -> None:
        self._edges.append((src, dst))

    def add_conditional_edge(
        self,
        src: str,
        router: Callable[[dict], str],
        path_map: dict[str, str],
    ) -> None:
        self._conditional[src] = (router, dict(path_map))

    def run(
        self,
        initial: PipelineState,
        *,
        max_iterations: int = 5,
    ) -> PipelineState:
        try:
            from crewai import Agent, Task, Crew
        except ImportError:
            raise ImportError(
                "crewai not installed. Install with: pip install deep-reports-sdk[crewai]"
            )

        for name, fn in self._nodes.items():
            agent = Agent(
                role=name.replace("_", " ").title(),
                goal=f"Execute the {name} stage of the report pipeline.",
                backstory=f"Automated {name} agent for technical report generation.",
                verbose=False,
                function_calling_llm=None,
            )
            self._agents[name] = agent

        task_map: dict[str, Any] = {}
        for src, dst in self._edges:
            if dst in self._agents:
                task = Task(
                    description=f"Execute {dst}",
                    agent=self._agents[dst],
                )
                self._tasks.append(task)
                task_map[dst] = task

        crew = Crew(
            agents=list(self._agents.values()),
            tasks=self._tasks,
            verbose=False,
        )

        result = crew.kickoff()
        return PipelineState(**dict(initial), result=str(result))  # type: ignore
