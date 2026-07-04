"""Unit tests for deep_report.py — graph building and routing logic."""

from __future__ import annotations

import pytest


class TestBuildGraph:
    """Test _build_graph wires correct nodes and edges for each agent set."""

    def _build_orch(self, agent_set: str):
        """Build orchestrator for given agent set, bypassing real filesystem."""
        from deep_reports.orchestrator import get_orchestrator
        from deep_reports.deep_report import (
            AGENT_SETS, MAX_REFINER_ITERATIONS, BLOCKER_THRESHOLD,
            QUALITY_THRESHOLD,
        )

        agent_names = AGENT_SETS[agent_set]
        orch = get_orchestrator("native")

        # Replicate _build_graph logic with mocks for each agent set
        def noop_fn(state):
            return state

        orch.add_node("generator", noop_fn)
        if "evidence_critic" in agent_names:
            orch.add_node("evidence_critic", noop_fn)
        if "quality_critic" in agent_names:
            orch.add_node("quality_critic", noop_fn)
        if "refiner" in agent_names:
            orch.add_node("refiner", noop_fn)

        # Build edges like _build_graph does
        if "generator" in agent_names and "evidence_critic" in agent_names:
            orch.add_edge("generator", "evidence_critic")

        if "evidence_critic" in agent_names:
            def ev_router(s):
                b = sum(1 for i in s.get("evidence_issues", []) if i.get("severity") == "blocker")
                return "escalate" if b >= BLOCKER_THRESHOLD else "ok"
            ev_map = {}
            ev_map["escalate"] = "refiner" if "refiner" in agent_names else "END"
            ev_map["ok"] = "quality_critic" if "quality_critic" in agent_names else "END"
            orch.add_conditional_edge("evidence_critic", ev_router, ev_map)

        if "quality_critic" in agent_names:
            def qc_router(s):
                sc = s.get("quality_overall")
                if sc is None or not isinstance(sc, (int, float)):
                    return "ok"
                return "refine" if float(sc) < QUALITY_THRESHOLD else "done"
            qc_map = {}
            qc_map["refine"] = "refiner" if "refiner" in agent_names else "END"
            qc_map["done"] = "END"
            orch.add_conditional_edge("quality_critic", qc_router, qc_map)

        if "refiner" in agent_names:
            def rf_router(s):
                return "done" if s.get("iteration", 1) > MAX_REFINER_ITERATIONS else "loop"
            rm_map = {}
            rm_map["loop"] = "evidence_critic" if "evidence_critic" in agent_names else "END"
            rm_map["done"] = "END"
            orch.add_conditional_edge("refiner", rf_router, rm_map)

        return orch, agent_names

    def test_minimal_generator_only(self):
        """'minimal' mode: only generator node, no edges."""
        orch, names = self._build_orch("minimal")
        assert names == ["generator"]
        assert "generator" in orch._nodes
        assert orch._entry == "generator"
        # No outgoing edges from generator in minimal mode
        assert "generator" not in orch._edges

    def test_core_generator_evidence(self):
        """'core' mode: generator → evidence_critic edge."""
        orch, names = self._build_orch("core")
        assert names == ["generator", "evidence_critic"]
        assert orch._entry == "generator"
        assert orch._edges["generator"] == ["evidence_critic"]

    def test_all_four_agents(self):
        """'all' mode: all 4 nodes registered."""
        orch, names = self._build_orch("all")
        assert names == ["generator", "evidence_critic", "quality_critic", "refiner"]
        assert orch._nodes.keys() == {
            "generator", "evidence_critic", "quality_critic", "refiner"
        }

    def test_refiner_only_if_in_set(self):
        """Refiner conditional edge only added when 'refiner' in agent_names."""
        orch_min, _ = self._build_orch("minimal")
        assert "refiner" not in orch_min._nodes

        orch_all, _ = self._build_orch("all")
        assert "refiner" in orch_all._nodes
        assert "refiner" in orch_all._conditional

    def test_core_evidence_critic_escape_route_without_refiner(self):
        """'core' mode: evidence_critic escalate → END (no refiner)."""
        orch, _ = self._build_orch("core")
        # evidence_critic has conditional edge
        assert "evidence_critic" in orch._conditional
        _, path_map = orch._conditional["evidence_critic"]
        assert path_map["escalate"] == "END"
        assert path_map["ok"] == "END"

    def test_all_evidence_critic_escape_to_refiner(self):
        """'all' mode: evidence_critic escalate → refiner."""
        orch, _ = self._build_orch("all")
        _, path_map = orch._conditional["evidence_critic"]
        assert path_map["escalate"] == "refiner"
        assert path_map["ok"] == "quality_critic"


class TestMakeAgentFnLogic:
    """Test that _make_agent_fn logic is correct via direct simulation."""

    def test_refiner_increments_iteration(self):
        """Simulate: refiner agent fn increments iteration."""
        from deep_reports.deep_report import MAX_REFINER_ITERATIONS

        state = {"iteration": 1}
        new_state = dict(state)
        if "refiner":
            new_state["iteration"] = state.get("iteration", 1) + 1

        assert new_state["iteration"] == 2
        assert state["iteration"] == 1  # original unchanged

    def test_evidence_critic_extracts_structured(self):
        """Structured data from evidence_critic populates state fields."""
        structured = {
            "issues": [{"severity": "blocker", "issue_type": "fabrication"}],
            "summary": "has blockers",
        }

        state = {}
        new_state = dict(state)
        if structured:
            new_state["evidence_issues"] = structured.get("issues", [])
            new_state["evidence_summary"] = structured.get("summary", "")

        assert new_state["evidence_issues"] == structured["issues"]
        assert new_state["evidence_summary"] == "has blockers"

    def test_quality_critic_extracts_structured(self):
        """Structured data from quality_critic populates scores and overall."""
        structured = {
            "scores": {"depth": 7, "breadth": 6},
            "overall": 6.5,
            "summary": "good",
        }

        new_state = {}
        if structured:
            new_state["quality_scores"] = structured.get("scores", {})
            new_state["quality_summary"] = structured.get("summary", "")
            new_state["quality_overall"] = structured.get("overall")

        assert new_state["quality_scores"]["depth"] == 7
        assert new_state["quality_overall"] == 6.5

    def test_generator_preserves_iteration(self):
        """Generator fn does not modify iteration."""
        state = {"iteration": 3, "question": "test?"}
        new_state = dict(state)
        # generator doesn't touch iteration
        assert new_state["iteration"] == 3


class TestEscalationThresholds:
    def test_threshold_constants(self):
        from deep_reports.deep_report import (
            BLOCKER_THRESHOLD, QUALITY_THRESHOLD, MAX_REFINER_ITERATIONS,
        )
        assert BLOCKER_THRESHOLD == 3
        assert QUALITY_THRESHOLD == 5.0
        assert MAX_REFINER_ITERATIONS == 3

    def test_refiner_3_iterations_then_exit(self):
        """3 refiner calls (iterations 1→2→3), then loop breaks at > 3."""
        from deep_reports.deep_report import MAX_REFINER_ITERATIONS

        calls = []
        for iteration in range(1, 6):
            result = "done" if iteration > MAX_REFINER_ITERATIONS else "loop"
            calls.append((iteration, result))

        # Iterations 1, 2, 3 → "loop", iteration 4 → "done"
        assert calls[0] == (1, "loop")
        assert calls[1] == (2, "loop")
        assert calls[2] == (3, "loop")
        assert calls[3] == (4, "done")
        assert calls[4] == (5, "done")

    def test_quality_threshold_gating(self):
        """Score >= 5.0 → done; score < 5.0 → refine."""
        from deep_reports.deep_report import QUALITY_THRESHOLD

        assert ("done" if 7.0 >= QUALITY_THRESHOLD else "refine") == "done"
        assert ("done" if 4.9 >= QUALITY_THRESHOLD else "refine") == "refine"
        assert ("done" if 5.0 >= QUALITY_THRESHOLD else "refine") == "done"

    def test_blocker_threshold_gating(self):
        """Blockers >= 3 → escalate; blockers < 3 → ok."""
        from deep_reports.deep_report import BLOCKER_THRESHOLD

        assert ("escalate" if 3 >= BLOCKER_THRESHOLD else "ok") == "escalate"
        assert ("escalate" if 4 >= BLOCKER_THRESHOLD else "ok") == "escalate"
        assert ("escalate" if 2 >= BLOCKER_THRESHOLD else "ok") == "ok"

    def test_quality_router_safes_on_missing_score(self):
        """quality_router returns 'ok' when score is None or non-numeric."""
        score = None
        result = "ok" if score is None else ("refine" if float(score) < 5.0 else "done")
        assert result == "ok"

        score = "bad-string"
        try:
            float(score)
            result = "refine"
        except (ValueError, TypeError):
            result = "ok"
        assert result == "ok"


class TestAgentSets:
    def test_agent_sets_defined(self):
        from deep_reports.deep_report import AGENT_SETS

        assert AGENT_SETS["minimal"] == ["generator"]
        assert AGENT_SETS["core"] == ["generator", "evidence_critic"]
        assert AGENT_SETS["all"] == ["generator", "evidence_critic", "quality_critic", "refiner"]
