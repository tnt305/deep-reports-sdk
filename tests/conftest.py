"""Root conftest — pytest fixtures and integration-test skip logic."""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

import pytest

# Shared constant: per-test budget for integration tests
DR_TEST_BUDGET_USD_DEFAULT = 0.15

INTEGRATION_DIR = str(Path(__file__).parent.resolve() / "integration")


def _has_any_provider_key() -> bool:
    """Return True if any cloud API key is set AND valid, OR Ollama CLI is available."""
    # Check for API keys AND validate them
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")).messages.create(
                model="claude-sonnet-4-5", max_tokens=1,
                messages=[{"role": "user", "content": "x"}],
            )
            return True
        except Exception:
            pass
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create(
                model="gpt-4o-mini", max_tokens=1,
                messages=[{"role": "user", "content": "x"}],
            )
            return True
        except Exception:
            pass
    local = shutil.which("ollama") is not None
    return local


# --- Session-scoped lock serializes real-LLM test execution ---
# This prevents rate-limit conflicts when pytest-xdist runs tests in parallel.
_llm_lock = threading.Lock()


@pytest.fixture(scope="session")
def llm_serializer():
    """Ensure real-LLM tests never run in parallel."""
    return _llm_lock


def pytest_collection_modifyitems(config, items):
    """Skip every integration test when no provider API key is available."""
    if _has_any_provider_key():
        return
    skip = pytest.mark.skip(reason="No provider API key in env; skip real-test")
    for item in items:
        if INTEGRATION_DIR in str(item.fspath):
            item.add_marker(skip)
