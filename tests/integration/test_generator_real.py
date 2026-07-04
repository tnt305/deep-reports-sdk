"""Integration test: real Generator agent on fixture repo."""

from __future__ import annotations

import os
import pytest
from pathlib import Path

from deep_reports.agents.base import AgentContext
from deep_reports.agents.generator import ReportGenerator
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


def test_generator_produces_structured_output():
    """Generator produces a Markdown report from a real fixture repo."""
    guard = BudgetGuard(max_usd=DR_TEST_BUDGET_USD_DEFAULT)
    set_default_budget_guard(guard)

    fixture_path = str(Path(__file__).parent.parent / "fixtures" / "sample_repo")

    provider = get_provider("anthropic")
    ctx = AgentContext(
        state={
            "question": "What is this library's core primitive?",
            "sources": [fixture_path],
            "report_md": None,
        },
        provider=provider,
        budget=guard,
    )

    agent = ReportGenerator()
    result = agent.invoke(ctx)

    assert isinstance(result.content, str)
    assert len(result.content) > 100
    assert result.cost_usd > 0
    assert "Core" in result.content or "primitive" in result.content.lower()
    assert guard.spent > 0
