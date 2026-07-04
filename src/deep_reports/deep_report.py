"""DeepReport facade — public API for Python SDK."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from deep_reports.agents import get_agent
from deep_reports.agents.base import AgentContext
from deep_reports.config import DeepReportsConfig
from deep_reports.cost import BudgetGuard
from deep_reports.orchestrator import get_orchestrator
from deep_reports.providers import get_provider
from deep_reports.security import validate_paths, get_allowed_roots


logger = logging.getLogger("deep_reports")


AgentList = Literal["all", "core", "minimal"]


AGENT_SETS: dict[AgentList, list[str]] = {
    "all": ["generator", "evidence_critic", "quality_critic", "refiner"],
    "core": ["generator", "evidence_critic"],
    "minimal": ["generator"],
}


# Escalation thresholds
BLOCKER_THRESHOLD = 3
QUALITY_THRESHOLD = 5.0
MAX_REFINER_ITERATIONS = 3


class DeepReport:
    """
    Main SDK facade — orchestrates the full report generation pipeline.

    Usage:
        dr = DeepReport(
            sources=["./my-repo"],
            question="What is the core primitive?",
            provider="anthropic",
            framework="native",
            max_cost_usd=1.0,
        )
        result = dr.generate()
        print(result["report_md"])
    """

    def __init__(
        self,
        sources: list[str],
        question: str,
        *,
        provider: str | None = None,
        framework: str = "native",
        model: str | None = None,
        output_dir: str = "./reports/",
        max_cost_usd: float = 1.0,
        agents: AgentList = "all",
        config: DeepReportsConfig | None = None,
    ):
        self._config = config or DeepReportsConfig()
        if provider:
            self._config = self._config.with_overrides(provider=provider)  # type: ignore[assignment]
        if model:
            self._config = self._config.with_overrides(model=model)  # type: ignore[assignment]
        if framework:
            self._config.framework = framework  # type: ignore[assignment]
        if output_dir:
            self._config.output_dir = output_dir
        if max_cost_usd:
            self._config.max_cost_usd = max_cost_usd

        self._config.agents = agents  # type: ignore[assignment]

        # Security: validate source paths BEFORE any I/O
        roots = self._config.allowed_source_roots
        self._validated_sources = [
            str(p) for p in validate_paths(sources, allowed_roots=roots)
        ]

        self._question = question
        self._budget = BudgetGuard(
            max_usd=self._config.max_cost_usd,
            sub_budgets=self._config.sub_budgets or None,
        )

        # Initialize provider
        actual_provider = self._config.provider
        self._provider = get_provider(
            actual_provider,
            model=model or self._config.get_agent_model("generator"),
            timeout=self._config.request_timeout,
        )

        # Initialize orchestrator
        self._orchestrator = get_orchestrator(self._config.framework)

    def _make_agent_fn(self, agent_name: str):
        """Build a pipeline node function for an agent.

        F1 fix: after each critic agent runs, extract structured fields
        into shared state so downstream agents (e.g. Refiner) receive feedback.
        F3 fix: pass allowed_roots to agents so they re-validate at I/O layer.
        """
        roots = self._config.allowed_source_roots

        def fn(state: dict) -> dict:
            agent = get_agent(agent_name)
            # Wire allowed_roots into agents that read files (F3 security boundary)
            if hasattr(agent, "_allowed_roots"):
                agent._allowed_roots = roots  # type: ignore[attr-defined]
            ctx = AgentContext(
                state={**state},
                provider=self._provider,
                budget=self._budget,
            )
            result = agent.invoke(ctx)

            # F1: extract structured critic output into state
            new_state = {
                **state,
                "report_md": result.content,
                "cost_usd": self._budget.spent,
                f"{agent_name}_result": result.model_dump(),
            }
            if result.structured is not None:
                if agent_name == "evidence_critic":
                    new_state["evidence_issues"] = result.structured.get("issues", [])
                    new_state["evidence_summary"] = result.structured.get("summary", "")
                elif agent_name == "quality_critic":
                    new_state["quality_scores"] = result.structured.get("scores", {})
                    new_state["quality_summary"] = result.structured.get("summary", "")
                    new_state["quality_overall"] = result.structured.get("overall")

            # F1-HIGH: increment iteration counter on refiner to cap the escalation loop
            if agent_name == "refiner":
                new_state["iteration"] = state.get("iteration", 1) + 1

            return new_state
        return fn

    def _build_graph(self, agent_names: list[str]) -> None:
        """Wire graph nodes and edges into the orchestrator.

        F2 fix: conditional routing with tri-agent escalation.
        - generator → evidence_critic (always, after generation)
        - evidence_critic → refiner (if blockers >= threshold)
                     → quality_critic (if no blockers)
        - quality_critic → refiner (if score < threshold)
                       → END (if score >= threshold)
        - refiner → evidence_critic (loop, max MAX_REFINER_ITERATIONS)
                   → END (if no improvement after MAX_REFINER_ITERATIONS)

        MEDIUM fix: guard all edges with existence checks so "core" and "minimal"
        agent sets work without refiner nodes.
        """
        for name in agent_names:
            self._orchestrator.add_node(name, self._make_agent_fn(name))

        # generator → evidence_critic (only if both nodes exist)
        if "generator" in agent_names and "evidence_critic" in agent_names:
            self._orchestrator.add_edge("generator", "evidence_critic")
        elif "generator" in agent_names:
            # generator only (minimal mode) — nothing to chain to
            pass

        # evidence_critic: conditional route
        def evidence_router(state: dict) -> str:
            issues = state.get("evidence_issues", [])
            blockers = sum(1 for i in issues if i.get("severity") == "blocker")
            if blockers >= BLOCKER_THRESHOLD:
                if "refiner" not in agent_names:
                    logger.warning(
                        "evidence_critic found %d blockers (>= threshold %d) but "
                        "'refiner' not in agent set — pipeline exits early with unresolved blockers",
                        blockers, BLOCKER_THRESHOLD,
                    )
                return "escalate"
            return "ok"

        # Route escalate to refiner if present, else END; ok to quality_critic if present, else END
        if "evidence_critic" in agent_names:
            ev_map = {}
            ev_map["escalate"] = "refiner" if "refiner" in agent_names else "END"
            ev_map["ok"] = "quality_critic" if "quality_critic" in agent_names else "END"
            self._orchestrator.add_conditional_edge("evidence_critic", evidence_router, ev_map)

        # quality_critic: conditional route
        if "quality_critic" in agent_names:
            def quality_router(state: dict) -> str:
                score = state.get("quality_overall")
                if score is None:
                    return "ok"
                try:
                    score_val = float(score) if isinstance(score, (int, float, str)) else None
                except (ValueError, TypeError):
                    score_val = None
                if score_val is None or score_val < QUALITY_THRESHOLD:
                    return "refine"
                return "done"

            qc_map = {}
            if "refiner" in agent_names:
                qc_map["refine"] = "refiner"
            else:
                qc_map["refine"] = "END"
            qc_map["done"] = "END"
            self._orchestrator.add_conditional_edge("quality_critic", quality_router, qc_map)

        # refiner: conditional loop with iteration cap (only if refiner exists)
        if "refiner" in agent_names:
            def refiner_router(state: dict) -> str:
                iteration = state.get("iteration", 1)
                if iteration > MAX_REFINER_ITERATIONS:
                    return "done"
                return "loop"

            rm_map = {}
            if "evidence_critic" in agent_names:
                rm_map["loop"] = "evidence_critic"
            else:
                rm_map["loop"] = "END"
            rm_map["done"] = "END"
            self._orchestrator.add_conditional_edge("refiner", refiner_router, rm_map)

    def generate(self) -> dict:
        """
        Run the full pipeline. Returns PipelineState with report_md, cost_usd, etc.

        Escalation logic:
          - evidence_critic finds >= 3 blockers → Refiner
          - quality_critic scores < 5 → Refiner
          - Refiner loops back to evidence_critic, max 3 iterations
        """
        agent_names = AGENT_SETS.get(self._config.agents, AGENT_SETS["all"])
        self._build_graph(agent_names)

        initial = {
            "question": self._question,
            "sources": self._validated_sources,
            "report_md": None,
            "iteration": 1,
            "cost_usd": 0.0,
            "evidence_issues": [],
            "quality_scores": {},
            "quality_summary": "",
            "quality_overall": None,
        }

        logger.info(f"[deep-reports] Starting pipeline: {self._config.framework}/{self._config.provider}")
        final = self._orchestrator.run(initial, max_iterations=20)
        final["cost_usd"] = self._budget.spent

        # F4: validate output_dir before writing
        self._write_output(final)

        return final

    def _write_output(self, state: dict) -> None:
        """Write report.md and cost log to output_dir.

        F4 fix: restrict output_dir to safe paths (under cwd or ~/.deep-reports/).
        Rejects absolute paths outside allowed roots, and any path traversal.
        """
        raw_output = Path(self._config.output_dir)
        allowed_roots = get_allowed_roots()

        # Resolve output path and check it's within allowed roots
        try:
            resolved = validate_paths([str(raw_output)], allowed_roots=allowed_roots)[0]
        except Exception as exc:
            # Narrow: PathNotAllowed is expected; re-raise unexpected validator bugs.
            from deep_reports.security import PathNotAllowed
            if not isinstance(exc, PathNotAllowed):
                raise
            resolved = Path.cwd() / "reports"
            logger.warning(
                f"Output dir {raw_output} outside allowed roots {allowed_roots}; "
                f"writing to {resolved} instead"
            )

        output_dir = Path(resolved)
        output_dir.mkdir(parents=True, exist_ok=True)

        if state.get("report_md"):
            report_path = output_dir / "report.md"
            report_path.write_text(state["report_md"], encoding="utf-8")
            logger.info(f"[deep-reports] Report → {report_path}")

        log_path = output_dir / "cost.log"
        log_path.write_text(
            f"total_cost_usd={self._budget.spent:.6f}\n"
            f"provider={self._config.provider}\n"
            f"framework={self._config.framework}\n",
            encoding="utf-8",
        )
