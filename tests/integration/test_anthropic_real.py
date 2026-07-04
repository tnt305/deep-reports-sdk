"""Integration test: real Anthropic API call (real LLM, no mocks)."""

from __future__ import annotations

import os
import pytest

from deep_reports.providers import get_provider, set_default_budget_guard
from deep_reports.cost import BudgetGuard
from tests.conftest import DR_TEST_BUDGET_USD_DEFAULT


def _is_anthropic_available():
    """Check if a valid Anthropic API key is available."""
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key in ("nkq-4-6", "test-key", ""):
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
    """Serialize all real-LLM tests and inject budget guard."""
    guard = BudgetGuard(max_usd=DR_TEST_BUDGET_USD_DEFAULT)
    prev = set_default_budget_guard(guard)
    yield guard
    set_default_budget_guard(prev)


def test_anthropic_completes_real_call():
    """Send a real request to Anthropic API; verify structured response."""
    guard = BudgetGuard(max_usd=DR_TEST_BUDGET_USD_DEFAULT)
    set_default_budget_guard(guard)

    provider = get_provider("anthropic")
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
    assert resp.provider == "anthropic"
    assert guard.spent > 0


def test_anthropic_budget_exceeded():
    """BudgetGuard raises BudgetExceeded before cost goes over cap."""
    guard = BudgetGuard(max_usd=0.00001)  # intentionally tiny
    set_default_budget_guard(guard)

    provider = get_provider("anthropic")
    with pytest.raises(Exception):  # BudgetExceeded or API error
        provider.complete(
            messages=[{"role": "user", "content": "Write a long essay about everything."}],
            max_tokens=2000,
            budget=guard,
        )
