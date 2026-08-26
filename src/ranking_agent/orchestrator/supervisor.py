"""Supervisor — one candidate through the agent graph."""

from __future__ import annotations

import asyncio
import logging
import time

from ranking_agent.agents.base import AgentContext
from ranking_agent.agents.critic import CriticAgent
from ranking_agent.agents.evaluators import EVALUATOR_CLASSES, DimensionEvaluator
from ranking_agent.agents.jd_analyst import JDAnalystAgent
from ranking_agent.agents.resume_parser import ResumeParserAgent
from ranking_agent.agents.scorer import score as score_result
from ranking_agent.agents.verifier import EvidenceVerifier
from ranking_agent.config import Settings, get_settings
from ranking_agent.guardrails.input_guard import check_jd, check_resume_text
from ranking_agent.guardrails.output_guard import apply_output_guards
from ranking_agent.guardrails.pii import redact
from ranking_agent.llm.client import LLMClient
from ranking_agent.models.jd import JobProfile
from ranking_agent.models.resume import ResumeProfile
from ranking_agent.models.scoring import (
    DimensionOpinion,
    GuardrailReport,
    RankResult,
)
from ranking_agent.models.trace import RunTrace, span_recorder
from ranking_agent.progress import say

logger = logging.getLogger(__name__)


class GuardrailRejection(ValueError):
    """Input failed a hard guardrail; the candidate cannot be scored."""


class Supervisor:
    """Runs the agent graph for a single (JD, resume) pair."""

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or LLMClient(settings=self.settings)

    # ── public entry points ──
    async def analyze_jd(self, jd_text: str, trace: RunTrace | None = None) -> JobProfile:
        """JD analysis on its own — reused across a batch so it runs once."""
        tr = trace or RunTrace()
        guard = self._guard_jd(jd_text, tr)
        ctx = AgentContext(self.client, tr, self.settings)
        profile = await JDAnalystAgent(ctx).run(jd_text=guard.text)
        profile.sanitized = guard.sanitized
        profile.warnings.extend(guard.warnings)
        return profile

    async def rank_one(
        self,
        *,
        resume_text: str,
        jd_text: str | None = None,
        job_profile: JobProfile | None = None,
        source_file: str | None = None,
        include_trace: bool = True,
    ) -> RankResult:
        """Full graph for one candidate.

        Raises ``GuardrailRejection`` when the input cannot be scored at all; every
        other failure degrades a dimension and is reported inside the result.
        """
        t0 = time.perf_counter()
        trace = RunTrace()
        report = GuardrailReport()

        # ── 1. Input guardrails ──
        if job_profile is None:
            if not jd_text:
                raise GuardrailRejection("either jd_text or job_profile is required")
            job_profile = await self.analyze_jd(jd_text, trace)

        with span_recorder(trace, "guardrail.input", kind="guardrail") as span:
            guard = check_resume_text(resume_text, self.settings)
            span.attributes.update(
                {"ok": guard.ok, "chars": len(guard.text), "warnings": guard.warnings}
            )
            report.input_ok = guard.ok
            report.input_failures = list(guard.failures)
            if guard.injection:
                report.injection_detected = guard.injection.detected
                report.injection_severity = guard.injection.severity
                report.injection_patterns = list(guard.injection.patterns)
            if not guard.ok:
                span.status = "error"
                raise GuardrailRejection("; ".join(guard.failures))

        # ── 2. PII redaction ──
        with span_recorder(trace, "guardrail.pii", kind="guardrail") as span:
            redaction = redact(
                guard.text, enabled=self.settings.enable_pii_redaction
            )
            report.pii_redactions = redaction.redactions
            span.attributes["redactions"] = redaction.redactions

        ctx = AgentContext(self.client, trace, self.settings)

        # ── 3. Parse the resume ──
        resume = await ResumeParserAgent(ctx).run(
            resume_text=redaction.text, contact=redaction.contact
        )
        resume.sanitized = guard.sanitized
        resume.warnings.extend(guard.warnings)

        # ── 4. Evaluator fan-out ──
        opinions = await self._run_evaluators(ctx, job_profile, resume)

        # ── 5. Evidence verification ──
        verification = EvidenceVerifier(trace, self.settings).run(
            opinions, resume.source_text
        )

        # ── 6-8. Score, guard, critique, repair ──
        result = await self._score_guard_critique(
            ctx=ctx,
            jd=job_profile,
            resume=resume,
            opinions=opinions,
            verification=verification,
            trace=trace,
            report=report,
        )

        result.source_file = source_file
        result.cost_usd = round(self.client.cost.cost_usd, 6)
        trace.total_duration_ms = round((time.perf_counter() - t0) * 1000)
        result.trace = trace if include_trace else None

        grounded = (
            result.verification.groundedness if result.verification is not None else None
        )
        say(
            f"done {result.candidate_name or source_file or 'candidate'} — "
            f"score={result.final_score:.2f} "
            f"years={result.candidate_years if result.candidate_years is not None else '?'} "
            f"groundedness={f'{grounded:.0%}' if grounded is not None else '?'} "
            f"review={'yes' if result.needs_human_review else 'no'} "
            f"cost=${result.cost_usd:.4f} ({trace.total_duration_ms}ms)"
        )
        return result

    # ── internals ──
    def _guard_jd(self, jd_text: str, trace: RunTrace):
        with span_recorder(trace, "guardrail.jd", kind="guardrail") as span:
            guard = check_jd(jd_text, self.settings)
            span.attributes.update({"ok": guard.ok, "warnings": guard.warnings})
            if not guard.ok:
                span.status = "error"
                raise GuardrailRejection("; ".join(guard.failures))
            return guard

    async def _run_evaluators(
        self, ctx: AgentContext, jd: JobProfile, resume: ResumeProfile
    ) -> list[DimensionOpinion]:
        """Run all dimension evaluators concurrently under a semaphore."""
        sem = asyncio.Semaphore(self.settings.agent_concurrency)

        async def _one(cls: type[DimensionEvaluator]) -> DimensionOpinion:
            async with sem:
                return await cls(ctx).run(jd=jd, resume=resume)

        with span_recorder(ctx.trace, "fanout.evaluators", kind="step") as span:
            opinions = await asyncio.gather(
                *(_one(cls) for cls in EVALUATOR_CLASSES)
            )
            span.attributes.update(
                {
                    "evaluators": len(opinions),
                    "degraded": sum(1 for o in opinions if o.degraded),
                }
            )
        return list(opinions)

    async def _score_guard_critique(
        self,
        *,
        ctx: AgentContext,
        jd: JobProfile,
        resume: ResumeProfile,
        opinions: list[DimensionOpinion],
        verification,
        trace: RunTrace,
        report: GuardrailReport,
    ) -> RankResult:
        """Score, guard, critique, and repair at most `max_repair_loops` times."""
        loops = 0
        result = score_result(
            jd=jd,
            resume=resume,
            opinions=opinions,
            verification=verification,
            trace=trace,
            settings=self.settings,
        )
        say(f"ok scorer — score={result.final_score:.2f}")
        with span_recorder(trace, "guardrail.output", kind="guardrail") as span:
            result, report = apply_output_guards(
                result, report=report, settings=self.settings
            )
            span.attributes.update(
                {
                    "violations": report.output_violations,
                    "bias_flags": report.bias_flags,
                    "needs_review": result.needs_human_review,
                }
            )

        if not self.settings.enable_critic:
            return result

        while loops <= self.settings.max_repair_loops:
            verdict = await CriticAgent(ctx).run(result=result, jd=jd)
            result.critic = verdict

            if verdict.verdict == "accept" or loops >= self.settings.max_repair_loops:
                if verdict.verdict == "reject":
                    result.needs_human_review = True
                    result.review_reasons.append(
                        "critic rejected the evaluation: "
                        + ("; ".join(verdict.issues) or verdict.reasoning)
                    )
                break

            # ── Repair: apply the critic's dimension revisions and re-score ──
            applied = self._apply_revisions(opinions, verdict.suggested_dimension_revisions)
            loops += 1
            result.repair_loops = loops
            if not applied:
                if verdict.issues:
                    result.needs_human_review = True
                    result.review_reasons.append(
                        "critic flagged unresolved issues: " + "; ".join(verdict.issues)
                    )
                break

            with span_recorder(trace, f"repair.{loops}", kind="step") as span:
                span.attributes["revised_dimensions"] = applied
                verification = EvidenceVerifier(trace, self.settings).run(
                    opinions, resume.source_text
                )
                result = score_result(
                    jd=jd,
                    resume=resume,
                    opinions=opinions,
                    verification=verification,
                    trace=trace,
                    settings=self.settings,
                )
                say(f"ok scorer — score={result.final_score:.2f} (repair {loops})")
                result.repair_loops = loops
                result, report = apply_output_guards(
                    result, report=report, settings=self.settings
                )

        return result

    @staticmethod
    def _apply_revisions(
        opinions: list[DimensionOpinion], revisions: dict[str, float]
    ) -> list[str]:
        """Apply critic-suggested scores, bounded to avoid a runaway correction.

        A revision is capped at 20 points from the evaluator's own score: the critic
        is a reviewer, not a replacement evaluator, and an unbounded override would
        make the specialist agents decorative.
        """
        if not revisions:
            return []
        applied: list[str] = []
        by_dim = {o.dimension: o for o in opinions}
        for dim, suggested in revisions.items():
            opinion = by_dim.get(dim)
            if opinion is None:
                continue
            delta = max(-20.0, min(20.0, suggested - opinion.raw_score))
            if abs(delta) < 1.0:
                continue
            opinion.raw_score = round(
                max(0.0, min(100.0, opinion.raw_score + delta)), 1
            )
            opinion.reasoning = (
                f"{opinion.reasoning} [revised by critic: {delta:+.1f}]".strip()
            )
            applied.append(dim)
        return applied
