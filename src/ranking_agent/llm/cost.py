"""Token and USD accounting with a hard per-run budget."""

from __future__ import annotations

from dataclasses import dataclass, field

from ranking_agent.config import MODEL_PRICING


class BudgetExceeded(RuntimeError):
    """Raised when a run would exceed its configured USD budget."""


@dataclass
class LLMUsage:
    model: str = "mock"
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    @classmethod
    def priced(cls, model: str, tokens_in: int, tokens_out: int) -> LLMUsage:
        in_rate, out_rate = MODEL_PRICING.get(model, (0.0, 0.0))
        cost = (tokens_in * in_rate + tokens_out * out_rate) / 1_000_000
        return cls(model=model, tokens_in=tokens_in, tokens_out=tokens_out,
                   cost_usd=round(cost, 8))


@dataclass
class CostTracker:
    """Accumulates usage for one run; checks budget before each call."""

    budget_usd: float = 1.0
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    by_model: dict[str, float] = field(default_factory=dict)

    def add(self, usage: LLMUsage) -> LLMUsage:
        self.calls += 1
        self.tokens_in += usage.tokens_in
        self.tokens_out += usage.tokens_out
        self.cost_usd = round(self.cost_usd + usage.cost_usd, 8)
        self.by_model[usage.model] = round(
            self.by_model.get(usage.model, 0.0) + usage.cost_usd, 8
        )
        return usage

    def check_budget(self) -> None:
        if self.budget_usd > 0 and self.cost_usd >= self.budget_usd:
            raise BudgetExceeded(
                f"run budget of ${self.budget_usd:.4f} reached "
                f"(spent ${self.cost_usd:.4f} over {self.calls} calls)"
            )

    def summary(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": round(self.cost_usd, 6),
            "by_model": self.by_model,
        }
