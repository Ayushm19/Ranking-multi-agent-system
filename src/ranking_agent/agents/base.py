"""Agent base class with tracing, validation, budget, and typed fallback."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from ranking_agent.config import Settings, get_settings
from ranking_agent.llm.client import LLMClient
from ranking_agent.llm.cost import BudgetExceeded
from ranking_agent.models.trace import RunTrace, span_recorder
from ranking_agent.progress import say

logger = logging.getLogger(__name__)


class AgentContext:
    """Shared per-run state handed to every agent."""

    def __init__(
        self,
        client: LLMClient,
        trace: RunTrace | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.client = client
        self.trace = trace or RunTrace()
        self.settings = settings or get_settings()

    @property
    def cost_usd(self) -> float:
        return self.client.cost.cost_usd


class BaseAgent[T](ABC):
    """One responsibility, one prompt, one typed output."""

    name: str = "agent"
    task: str = "agent"

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    # ── to implement ──
    @abstractmethod
    def system_prompt(self) -> str:
        """Instructions that define this agent's role and output contract."""

    @abstractmethod
    def user_prompt(self, **kwargs: Any) -> str:
        """The request payload for one invocation."""

    @abstractmethod
    def parse(self, payload: dict[str, Any], **kwargs: Any) -> T:
        """Turn validated JSON into this agent's typed output."""

    @abstractmethod
    def fallback(self, **kwargs: Any) -> T:
        """Typed output used when the model cannot be reached or parsed."""

    # ── execution ──
    async def run(self, **kwargs: Any) -> T:
        """Execute the agent, recording a span whatever the outcome."""
        with span_recorder(self.ctx.trace, f"agent.{self.name}", kind="agent") as span:
            model = self.model_name()
            try:
                payload, usage = await self.ctx.client.complete_json(
                    system=self.system_prompt(),
                    user=self.user_prompt(**kwargs),
                    task=self.task,
                    model=model,
                )
                span.tokens_in = usage.tokens_in
                span.tokens_out = usage.tokens_out
                span.cost_usd = usage.cost_usd
                result = self.parse(payload, **kwargs)
                span.attributes["model"] = usage.model
                detail = self._success_detail(result)
                if detail:
                    say(f"ok {self.name} — {detail}")
                else:
                    say(f"ok {self.name}")
                return result
            except BudgetExceeded as exc:
                # Budget exceeded → skip, not fail.
                span.status = "skipped"
                span.error = str(exc)
                say(f"skip {self.name} — {exc}")
                return self.fallback(**kwargs)
            except Exception as exc:  # noqa: BLE001 - degrade, never crash the run
                span.status = "degraded"
                span.error = f"{type(exc).__name__}: {exc}"
                say(f"fail {self.name} — degraded to fallback: {exc}")
                return self.fallback(**kwargs)

    def _success_detail(self, result: T) -> str | None:
        """Optional short clause for the success progress line."""
        score = getattr(result, "raw_score", None)
        if isinstance(score, (int, float)):
            return f"score={float(score):.0f}"
        verdict = getattr(result, "verdict", None)
        if verdict:
            return f"verdict={verdict}"
        title = getattr(result, "role_title", None)
        if title:
            return f"role={title}"
        contact = getattr(result, "contact", None)
        name = getattr(contact, "name", None) if contact is not None else None
        if name:
            years = getattr(result, "total_years_experience", None)
            if years is not None:
                return f"name={name} years={years}"
            return f"name={name}"
        return None

    def model_name(self) -> str:
        return self.ctx.settings.model

    # ── helpers ──
    @staticmethod
    def clip(text: str | None, limit: int) -> str:
        """Bound prompt size while keeping both ends of the document.

        Resumes put identity and recency at the top and education/older roles at
        the bottom; a head-only truncation loses the tail that dates often live in.
        """
        body = (text or "").strip()
        if len(body) <= limit:
            return body
        head = int(limit * 0.75)
        tail = limit - head
        return f"{body[:head]}\n\n[...truncated...]\n\n{body[-tail:]}"

    @staticmethod
    def as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def as_str_list(value: Any, limit: int = 50) -> list[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if not isinstance(value, list):
            return []
        out = [str(v).strip() for v in value if str(v).strip()]
        return out[:limit]
