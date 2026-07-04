"""Integration test: real QualityCritic scores a report."""

from __future__ import annotations

import os
import pytest

from deep_reports.agents.base import AgentContext
from deep_reports.agents.critic_quality import QualityCritic
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


def test_quality_critic_scores_report():
    """QualityCritic produces 8-dimension scores."""
    guard = BudgetGuard(max_usd=DR_TEST_BUDGET_USD_DEFAULT)
    set_default_budget_guard(guard)

    sample_report = """# Technical Report: Sample Library

## Executive Summary
This library provides basic arithmetic operations.

## Core Primitive
The core primitive is the `add` function.

## Evidence
The `add` function at line 1 adds two integers.
"""

    provider = get_provider("anthropic")
    ctx = AgentContext(
        state={"report_md": sample_report},
        provider=provider,
        budget=guard,
    )

    agent = QualityCritic()
    result = agent.invoke(ctx)

    assert isinstance(result.content, str)
    assert result.cost_usd > 0
    # Should have parsed structured scores
    if result.structured and "scores" in result.structured:
        scores = result.structured["scores"]
        assert len(scores) == 8
        for key, val in scores.items():
            assert isinstance(val, (int, float))
            assert 0 <= val <= 10
