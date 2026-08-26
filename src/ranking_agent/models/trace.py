"""Execution trace: one span per agent / guardrail / deterministic step."""

from __future__ import annotations

import time
import uuid
from types import TracebackType

from pydantic import BaseModel, Field


class Span(BaseModel):
    """A single unit of work inside a run."""

    name: str
    kind: str = "step"  # agent | guardrail | tool | step
    status: str = "ok"  # ok | error | skipped | degraded
    duration_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    retries: int = 0
    error: str | None = None
    attributes: dict[str, object] = Field(default_factory=dict)


class RunTrace(BaseModel):
    """Ordered spans plus run-level totals."""

    run_id: str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex[:12]}")
    spans: list[Span] = Field(default_factory=list)
    total_duration_ms: int = 0

    @property
    def total_cost_usd(self) -> float:
        return round(sum(s.cost_usd for s in self.spans), 6)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens_in + s.tokens_out for s in self.spans)

    @property
    def failed_spans(self) -> list[Span]:
        return [s for s in self.spans if s.status == "error"]

    def add(self, span: Span) -> Span:
        self.spans.append(span)
        return span

    def get(self, name: str) -> Span | None:
        return next((s for s in self.spans if s.name == name), None)

    def summary(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "spans": len(self.spans),
            "failed": len(self.failed_spans),
            "tokens": self.total_tokens,
            "cost_usd": self.total_cost_usd,
            "duration_ms": self.total_duration_ms,
        }


class span_recorder:
    """Context manager that times a block and appends its span to a trace.

    Usage::

        with span_recorder(trace, "agent.jd_analyst", kind="agent") as sp:
            sp.attributes["role"] = profile.role_title
    """

    def __init__(self, trace: RunTrace, name: str, kind: str = "step") -> None:
        self.trace = trace
        self.span = Span(name=name, kind=kind)
        self._t0 = 0.0

    def __enter__(self) -> Span:
        self._t0 = time.perf_counter()
        return self.span

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self.span.duration_ms = round((time.perf_counter() - self._t0) * 1000)
        if exc is not None:
            self.span.status = "error"
            self.span.error = f"{type(exc).__name__}: {exc}"
        self.trace.add(self.span)
        return False
