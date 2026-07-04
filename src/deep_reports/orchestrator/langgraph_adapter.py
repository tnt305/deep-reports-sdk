"""LangGraph adapter — wraps LangGraph StateGraph as Orchestrator."""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("deep_reports.orchestrator.langgraph")


class LangGraphOrchestrator:
    """
    LangGraph StateGraph wrapper conforming to Orchestrator Protocol.

    Deferred edge resolution: conditional edges are recorded at add_*
    time but resolved against the COMPLETE node set in run(), so
    downstream targets not yet added don't false-positive to END.
    """

    framework = "langgraph"

    def __init__(self):
        self._graph = None
        self._nodes: dict[str, Callable[..., Any]] = {}
        self._pending_conditionals: list[tuple[str, Callable, dict[str, str]]] = []

    def add_node(self, name: str, fn: Callable[..., Any]) -> None:
        self._nodes[name] = fn

    def add_edge(self, src: str, dst: str) -> None:
        self._get_graph().add_edge(src, dst)

    def add_conditional_edge(
        self,
        src: str,
        router: Callable[[dict], str],
        path_map: dict[str, str],
    ) -> None:
        # Defer resolution until run() — see module docstring
        self._pending_conditionals.append((src, router, dict(path_map)))

    def _get_graph(self):
        if self._graph is None:
            from langgraph.graph import StateGraph
            self._graph = StateGraph(dict)
        return self._graph

    def run(self, initial: dict, *, max_iterations: int = 5) -> dict:
        from langgraph.graph import END

        graph = self._get_graph()

        # Add all registered nodes
        for name, fn in self._nodes.items():
            graph.add_node(name, fn)

        # Resolve pending conditionals against complete node set
        for src, router, path_map in self._pending_conditionals:
            resolved = {
                k: (v if v in self._nodes else END)
                for k, v in path_map.items()
            }
            graph.add_conditional_edges(src, router, resolved)

        self._pending_conditionals.clear()
        compiled = graph.compile()
        return compiled.invoke(dict(initial), {"recursion_limit": max_iterations * 5})
