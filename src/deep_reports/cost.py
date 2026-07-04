"""Token pricing + BudgetGuard with sub-budget support."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Final


class UnknownModelPrice(Exception):
    """Raised when no price is found for a model and no env-var override exists."""

    pass


class BudgetExceeded(Exception):
    """Raised when total cost exceeds the configured budget."""

    pass


# (input_usd_per_1k, output_usd_per_1k)
# Prices keyed by family prefix for longest-prefix matching.
PRICES: "Final[dict[str, tuple[float, float]]]" = {
    # Anthropic family
    "claude-haiku-4":   (0.0008,  0.004),
    "claude-sonnet-4":  (0.003,   0.015),
    "claude-opus-4":    (0.015,   0.075),
    # OpenAI family
    "gpt-4o-mini":      (0.00015, 0.0006),
    "gpt-4o":           (0.0025,  0.01),
    "o1":               (0.015,   0.06),
    # Ollama / local models — FREE
    "llama3":           (0.0, 0.0),
    "llama2":           (0.0, 0.0),
    "mistral":          (0.0, 0.0),
    "mixtral":          (0.0, 0.0),
    "qwen":             (0.0, 0.0),
    "gemma":            (0.0, 0.0),
    "phi":              (0.0, 0.0),
}


def _get_price(model: str) -> tuple[float, float]:
    """Look up price for model. Strategy (priority order):
       1. Per-model env-var override DR_MODEL_PRICE_<MODEL>_IN/_OUT
       2. Generic DR_MODEL_PRICE_IN/_OUT (applies to any model)
       3. Exact key in PRICES dict
       4. Longest-prefix match
       5. Raise UnknownModelPrice
    """
    safe = re.sub(r"[^A-Z0-9]+", "_", model.upper()).strip("_")

    # 1. Per-model exact env-var override
    env_in  = os.getenv(f"DR_MODEL_PRICE_{safe}_IN")
    env_out = os.getenv(f"DR_MODEL_PRICE_{safe}_OUT")
    if env_in is not None or env_out is not None:
        if env_in is None or env_out is None:
            missing = f"DR_MODEL_PRICE_{safe}_IN" if env_in is None else f"DR_MODEL_PRICE_{safe}_OUT"
            raise UnknownModelPrice(
                f"Partial price override for {model!r}: {missing!r} is not set. "
                f"Both DR_MODEL_PRICE_{safe}_IN AND _OUT must be set together, or neither."
            )
        try:
            return float(env_in), float(env_out)
        except ValueError as e:
            raise UnknownModelPrice(f"Invalid DR_MODEL_PRICE_{safe}_* values: {e!r}") from None

    # 2. Generic per-model env vars
    gen_in  = os.getenv("DR_MODEL_PRICE_IN")
    gen_out = os.getenv("DR_MODEL_PRICE_OUT")
    if gen_in is not None or gen_out is not None:
        if gen_in is None or gen_out is None:
            missing = "DR_MODEL_PRICE_IN" if gen_in is None else "DR_MODEL_PRICE_OUT"
            raise UnknownModelPrice(
                f"Partial generic price override: {missing!r} is not set. "
                f"Both DR_MODEL_PRICE_IN AND DR_MODEL_PRICE_OUT must be set together, or neither."
            )
        try:
            return float(gen_in), float(gen_out)
        except ValueError as e:
            raise UnknownModelPrice(f"Invalid DR_MODEL_PRICE_IN/OUT values: {e!r}") from None

    # 3 & 4. Static dict: exact then longest-prefix
    if model in PRICES:
        return PRICES[model]
    for key in sorted(PRICES, key=len, reverse=True):
        if model.startswith(key):
            return PRICES[key]

    # 5. Hard fail
    supported = ", ".join(sorted(PRICES))
    raise UnknownModelPrice(
        f"No price for model {model!r}. "
        f"Fix without editing cost.py:\n"
        f"  1. Set DR_MODEL_PRICE_{safe}_IN and _OUT (per-model)\n"
        f"  2. Set DR_MODEL_PRICE_IN and DR_MODEL_PRICE_OUT (generic)\n"
        f"Hardcoded families: {supported}"
    )


def compute_cost(model: str, in_tok: int, out_tok: int) -> float:
    """Compute cost in USD from token counts and model price."""
    in_rate, out_rate = _get_price(model)
    return (in_tok / 1000) * in_rate + (out_tok / 1000) * out_rate


class BudgetGuard:
    """
    Track and cap LLM spend. Supports total budget + optional per-stage sub-budgets.

    Usage:
        guard = BudgetGuard(max_usd=1.0, sub_budgets={"generator": 0.30, "critic": 0.10})
        cost = guard.add("claude-sonnet-4-5", 500, 200)
        # raises BudgetExceeded if total or sub-budget exceeded
    """

    def __init__(
        self,
        max_usd: float = 1.0,
        sub_budgets: dict[str, float] | None = None,
    ):
        self.max = max_usd
        self.spent = 0.0
        self.sub_budgets = dict(sub_budgets or {})
        self._sub_spent: dict[str, float] = {}

    def add(
        self,
        model: str,
        in_tok: int,
        out_tok: int,
        *,
        stage: str = "",
    ) -> float:
        """
        Compute cost, accumulate, check limits, raise if exceeded.
        Returns the cost in USD for this call.
        """
        cost = compute_cost(model, in_tok, out_tok)
        self.spent += cost

        if self.spent > self.max:
            raise BudgetExceeded(
                f"Spent ${self.spent:.4f} > max ${self.max:.2f} (total)"
            )

        if stage and stage in self.sub_budgets:
            self._sub_spent[stage] = self._sub_spent.get(stage, 0) + cost
            if self._sub_spent[stage] > self.sub_budgets[stage]:
                raise BudgetExceeded(
                    f"Stage {stage!r} spent ${self._sub_spent[stage]:.4f} "
                    f"> ${self.sub_budgets[stage]:.2f}"
                )

        return cost

    def sub_remaining(self, stage: str) -> float | None:
        """Return remaining budget for a stage, or None if stage has no sub-budget."""
        if stage not in self.sub_budgets:
            return None
        spent = self._sub_spent.get(stage, 0)
        return max(0.0, self.sub_budgets[stage] - spent)
