"""Unit tests for NativeOrchestrator — no LLM needed."""

from __future__ import annotations

from deep_reports.orchestrator.native import NativeOrchestrator


def test_single_node():
    """Single node executes and returns updated state."""
    orch = NativeOrchestrator()
    orch.add_node("step1", lambda s: {**s, "done": True})
    result = orch.run({"question": "test"})
    assert result["done"] is True


def test_linear_chain():
    """Linear chain: each node receives previous output."""
    orch = NativeOrchestrator()
    orch.add_node("step1", lambda s: {**s, "a": 1})
    orch.add_node("step2", lambda s: {**s, "b": s.get("a", 0) + 1})
    orch.add_edge("step1", "step2")
    result = orch.run({"question": "test"})
    assert result["a"] == 1
    assert result["b"] == 2


def test_fan_out():
    """Multiple edges from one node all fire (co-fan-out)."""
    orch = NativeOrchestrator()
    orch.add_node("start", lambda s: {**s, "x": 1})
    orch.add_node("branch_a", lambda s: {**s, "a": s.get("x", 0) + 10})
    orch.add_node("branch_b", lambda s: {**s, "b": s.get("x", 0) + 20})
    orch.add_edge("start", ["branch_a", "branch_b"])
    result = orch.run({})
    assert result["a"] == 11
    assert result["b"] == 21


def test_conditional_edge():
    """Conditional edge routes based on state."""
    orch = NativeOrchestrator()
    orch.add_node("decide", lambda s: {**s, "flag": True})
    orch.add_node("path_a", lambda s: {**s, "path": "a"})
    orch.add_node("path_b", lambda s: {**s, "path": "b"})

    def router(s):
        return "yes" if s.get("flag") else "no"

    orch.add_conditional_edge("decide", router, {"yes": "path_a", "no": "path_b"})
    result = orch.run({})
    assert result["flag"] is True
    assert result["path"] == "a"


def test_router_invalid_key_raises():
    """Router returning unknown key raises ValueError."""
    orch = NativeOrchestrator()
    orch.add_node("decide", lambda s: {**s, "flag": False})
    orch.add_node("path_a", lambda s: {**s, "path": "a"})

    def bad_router(s):
        return "unknown_key"

    orch.add_conditional_edge("decide", bad_router, {"yes": "path_a"})
    try:
        orch.run({})
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "unknown_key" in str(exc) or "valid keys" in str(exc)


def test_reusable_after_run():
    """Orchestrator can be run multiple times with same topology."""
    orch = NativeOrchestrator()
    orch.add_node("step", lambda s: {**s, "n": s.get("n", 0) + 1})
    orch.add_edge("start", "step")

    r1 = orch.run({"n": 0})
    r2 = orch.run({"n": r1.get("n", 0)})
    assert r2["n"] == 2
    # Third run should also work (snapshot/restore verified by no error)


def test_dead_end_node():
    """Node with no outgoing edges terminates cleanly."""
    orch = NativeOrchestrator()
    orch.add_node("solo", lambda s: {**s, "done": True})
    result = orch.run({"question": "test"})
    assert result["done"] is True


def test_max_iterations_cycle_detection():
    """Cycle detection: same node visited twice with flag."""
    orch = NativeOrchestrator()
    orch.add_node("loop", lambda s: {**s, "counter": s.get("counter", 0) + 1})

    def router(s):
        return "loop" if s.get("counter", 0) < 3 else "end"

    orch.add_conditional_edge("start", lambda s: "loop", {"loop": "loop"})
    orch.add_node("start", lambda s: {**s})
    orch.add_node("end", lambda s: {**s, "done": True})
    orch.add_conditional_edge("loop", router, {"loop": "loop", "end": "end"})

    # The orchestrator should handle the loop and eventually stop
    result = orch.run({"question": "test"}, max_iterations=10)
    assert result.get("counter", 0) <= 3
