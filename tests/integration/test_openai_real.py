"""Integration test: real OpenAI API call (real LLM, no mocks)."""

from __future__ import annotations

import os
import pytest

from deep_reports.providers import get_provider, set_default_budget_guard
from deep_reports.cost import BudgetGuard
from tests.conftest import DR_TEST_BUDGET_USD_DEFAULT


pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@pytest.fixture(autouse=True)
def _serialize_and_budget(llm_serializer):
    guard = BudgetGuard(max_usd=DR_TEST_BUDGET_USD_DEFAULT)
    prev = set_default_budget_guard(guard)
    yield guard
    set_default_budget_guard(prev)


def test_openai_completes_real_call():
    """Send a real request to OpenAI API; verify structured response."""
    guard = BudgetGuard(max_usd=DR_TEST_BUDGET_USD_DEFAULT)
    set_default_budget_guard(guard)

    provider = get_provider("openai")
    resp = provider.complete(
        messages=[{"role": "user", "content": "Say 'hi' and nothing else."}],
        max_tokens=10,
        budget=guard,
    )

    assert isinstance(resp.content, str)
    assert resp.content.strip().lower().startswith("hi")
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0
    assert resp.cost_usd > 0
    assert resp.provider == "openai"
    assert guard.spent > 0


def test_openai_system_prompt():
    """System prompt influences response style."""
    guard = BudgetGuard(max_usd=DR_TEST_BUDGET_USD_DEFAULT)
    set_default_budget_guard(guard)

    provider = get_provider("openai")
    resp = provider.complete(
        messages=[{"role": "user", "content": "What is 1+1?"}],
        system="You only respond with single words.",
        max_tokens=5,
        budget=guard,
    )

    assert isinstance(resp.content, str)
    words = resp.content.strip().split()
    assert len(words) <= 3  # single-word style
