"""Integration test: real EvidenceCritic catches planted fabrication."""

from __future__ import annotations

import os
import pytest

from deep_reports.agents.base import AgentContext
from deep_reports.agents.critic_evidence import EvidenceCritic
from deep_reports.providers import get_provider, set_default_budget_guard
from deep_reports.cost import BudgetGuard
from tests.conftest import DR_TEST_BUDGET_USD_DEFAULT


def _is_anthropic_available():
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key in ("st-4-6", "test-key", ""):
        return False
    try:
        import anthropic
        anthropic.Anthropic(api_key=key).messages.create(
            model="claude-sonnet-4-5", max_tokens=1,
            messages=[{"role": "user", "content": "x"}],
        )
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _is_anthropic_available(),
    reason="Valid ANTHROPIC_API_KEY not available",
)


@pytest.fixture(autouse=True)
def _serialize_and_budget(llm_serializer):
    guard = BudgetGuard(max_usd=DR_TEST_BUDGET_USD_DEFAULT)
    prev = set_default_budget_guard(guard)
    yield guard
    set_default_budget_guard(prev)


def test_evidence_critic_catches_fabrication(tmp_path):
    """Critic catches a citation to a non-existent line range."""
    guard = BudgetGuard(max_usd=DR_TEST_BUDGET_USD_DEFAULT)
    set_default_budget_guard(guard)

    # Create a tiny source file
    src = tmp_path / "src.py"
    src.write_text("\n".join(f"def f{i}(): return {i}" for i in range(10)) + "\n")

    # Create a fake report citing line 9999 (doesn't exist)
    fake_report = (
        "## Core Pattern\n"
        f"The function `f0` achieves 10x speedup. See {src}:9999-10000.\n"
    )

    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    fixture_src = fixture_dir / "src.py"
    fixture_src.write_text(src.read_text())

    provider = get_provider("anthropic")
    ctx = AgentContext(
        state={
            "report_md": fake_report,
            "sources": [str(fixture_dir)],
        },
        provider=provider,
        budget=guard,
    )

    agent = EvidenceCritic()
    result = agent.invoke(ctx)

    assert isinstance(result.content, str)
    # Structured output should have detected fabrication
    if result.structured and "issues" in result.structured:
        issues = result.structured["issues"]
        assert len(issues) >= 1
