"""Unit tests for cost.py — BudgetGuard and price lookup."""

from __future__ import annotations

import pytest

from deep_reports.cost import (
    BudgetExceeded,
    BudgetGuard,
    UnknownModelPrice,
    _get_price,
    compute_cost,
)


class TestGetPrice:
    def test_exact_key(self):
        assert _get_price("claude-sonnet-4-5") == (0.003, 0.015)
        assert _get_price("gpt-4o-mini") == (0.00015, 0.0006)
        assert _get_price("llama3") == (0.0, 0.0)

    def test_prefix_match(self):
        # Longer model variant still matches family prefix
        assert _get_price("claude-opus-4-8") == (0.015, 0.075)
        assert _get_price("claude-haiku-4-5") == (0.0008, 0.004)

    def test_unknown_raises(self):
        with pytest.raises(UnknownModelPrice) as exc:
            _get_price("unknown-model-v99")
        assert "unknown-model-v99" in str(exc.value)
        assert "DR_MODEL_PRICE_" in str(exc.value)


class TestComputeCost:
    def test_claude_sonnet(self):
        cost = compute_cost("claude-sonnet-4-5", in_tok=1000, out_tok=500)
        expected = (1000 / 1000) * 0.003 + (500 / 1000) * 0.015
        assert abs(cost - expected) < 1e-9

    def test_free_model(self):
        cost = compute_cost("llama3", in_tok=1000, out_tok=500)
        assert cost == 0.0


class TestBudgetGuard:
    def test_add_tracks_total(self):
        guard = BudgetGuard(max_usd=1.0)
        cost = guard.add("gpt-4o-mini", in_tok=1000, out_tok=500)
        assert cost > 0
        assert guard.spent == cost

    def test_raises_on_exceed(self):
        guard = BudgetGuard(max_usd=0.0001)
        with pytest.raises(BudgetExceeded) as exc:
            guard.add("gpt-4o", in_tok=100000, out_tok=50000)
        assert "total" in str(exc.value)

    def test_sub_budgets(self):
        guard = BudgetGuard(
            max_usd=1.0,
            sub_budgets={"generator": 0.01, "critic": 0.005},
        )
        # Generator spend
        guard.add("gpt-4o-mini", in_tok=1000, out_tok=100, stage="generator")
        assert guard.spent > 0
        assert guard.sub_remaining("generator") is not None
        assert guard.sub_remaining("unknown") is None

    def test_sub_budget_exceeded(self):
        guard = BudgetGuard(
            max_usd=10.0,
            sub_budgets={"critic": 0.00001},  # tiny sub-budget
        )
        with pytest.raises(BudgetExceeded) as exc:
            guard.add("gpt-4o", in_tok=100000, out_tok=50000, stage="critic")
        assert "critic" in str(exc.value)

    def test_zero_cost_model(self):
        guard = BudgetGuard(max_usd=0.0)
        cost = guard.add("llama3", in_tok=1000, out_tok=500)
        assert cost == 0.0
        assert guard.spent == 0.0
