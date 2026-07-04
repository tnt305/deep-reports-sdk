"""NativeOrchestrator — in-memory state machine, no external deps."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Callable

from deep_reports.orchestrator.base import Orchestrator, PipelineState


logger = logging.getLogger("deep_reports.orchestrator.native")


class NativeOrchestrator(Orchestrator):
    """
    In-memory state machine orchestrator.

    Features:
      - dict-of-lists edges: multiple outgoing edges from one source all fire
      - Conditional edges with explicit path_map validation
      - Snapshot/restore: re-runnable across multiple pipeline runs
      - Cycle detection: distinguishes cycle vs legitimate long pipeline
      - Reusable: original topology restored after each run()
    """

    framework = "native"

    def __init__(self):
        self._nodes: dict[str, Callable[..., PipelineState]] = {}
        # CRITICAL: dict-of-lists so multiple edges from same source all fire.
        self._edges: dict[str, list[str]] = {}
        # Conditional: source → (router_fn, path_map)
        self._conditional: dict[str, tuple[Callable[[PipelineState], str], dict[str, str]]] = {}
        # Track entry point: first registered node, or first node with an edge from a non-existent source
        self._entry: str | None = None

    def add_node(self, name: str, fn: Callable[..., PipelineState]) -> None:
        self._nodes[name] = fn
        if self._entry is None:
            self._entry = name

    def add_edge(self, src: str, dst: str | list[str]) -> None:
        """Add one or more outgoing edges from src."""
        targets = [dst] if isinstance(dst, str) else list(dst)
        self._edges.setdefault(src, []).extend(targets)
        # If src is not a registered node, it becomes the entry point.
        # This handles the case where add_edge() is called before add_node(src).
        if src not in self._nodes and self._entry is None:
            self._entry = src

    def add_conditional_edge(
        self,
        src: str,
        router: Callable[[PipelineState], str],
        path_map: dict[str, str],
    ) -> None:
        self._conditional[src] = (router, dict(path_map))

    def run(
        self,
        initial: PipelineState,
        *,
        max_iterations: int = 10,
    ) -> PipelineState:
        """
        Execute the pipeline.

        State snapshots are taken at run() entry and restored on exit,
        making the orchestrator reusable across multiple run() calls.
        """
        # Snapshot topology (see ADR: reusable orchestrator)
        cond_snapshot = {
            k: (r, dict(pm))
            for k, (r, pm) in self._conditional.items()
        }
        edges_snapshot = {k: list(v) for k, v in self._edges.items()}

        try:
            state = dict(initial)
            current = self._entry or next(iter(self._nodes), "END")
            executed = 0
            visit_log: list[str] = []  # for cycle detection

            for it in range(max_iterations):
                if current == "END" or current not in self._nodes:
                    break

                # Execute current node first (see fix: execute before edge check)
                logger.debug(f"[native] executing node: {current}")
                result = self._nodes[current](state)
                if isinstance(result, dict):
                    state.update(result)
                executed += 1
                visit_log.append(current)

                # Resolve outgoing edges to find next node(s)
                if current in self._conditional:
                    router, path_map = self._conditional[current]
                    chosen = router(state)
                    if chosen not in path_map:
                        valid = sorted(path_map.keys())
                        raise ValueError(
                            f"Router for {current!r} returned {chosen!r}, "
                            f"but valid keys are {valid}. "
                            f"Add {chosen!r} to path_map or fix the router."
                        )
                    next_names = [path_map[chosen]]
                elif current in self._edges:
                    next_names = list(self._edges[current])
                else:
                    # Dead-end: no outgoing edges → return what we have.
                    break

                # Execute ALL targets from co-fan-out BEFORE advancing.
                # This ensures branch_a AND branch_b both run before we move on.
                for nxt in next_names:
                    if nxt == "END" or nxt not in self._nodes:
                        continue
                    logger.debug(f"[native] executing co-fan-out node: {nxt}")
                    r = self._nodes[nxt](state)
                    if isinstance(r, dict):
                        state.update(r)
                    executed += 1

                # Advance to the last target so the next loop iteration follows
                # downstream from the last co-fan-out node.
                current = next_names[-1] if next_names else "END"

            # Check iteration limit
            if it == max_iterations - 1 and current not in ("END",) and current in self._nodes:
                node_counts = Counter(n for n in visit_log)
                cycle_candidates = [n for n, c in node_counts.items() if c >= 2]
                if cycle_candidates:
                    cycle_path = " → ".join(
                        f"{n} (×{c})" for n, c in node_counts.most_common(3)
                    )
                    raise RuntimeError(
                        f"Orchestrator detected a cycle after {executed} executions "
                        f"of {max_iterations} allowed. Likely-loop nodes: {cycle_path}."
                    )
                raise RuntimeError(
                    f"Orchestrator reached max_iterations={max_iterations} "
                    f"after {executed} node executions. "
                    f"Raise max_iterations or split the graph."
                )

            return state  # type: ignore[return-value]

        finally:
            # Restore topology for next run()
            self._conditional = {k: (r, dict(pm)) for k, (r, pm) in cond_snapshot.items()}
            self._edges = {k: list(v) for k, v in edges_snapshot.items()}
