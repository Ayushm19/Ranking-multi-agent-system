"""Evidence Verifier — deterministic quote checks against the source resume."""

from __future__ import annotations

from ranking_agent.config import Settings, get_settings
from ranking_agent.guardrails.groundedness import build_report
from ranking_agent.models.scoring import DimensionOpinion, VerificationReport
from ranking_agent.models.trace import RunTrace, span_recorder
from ranking_agent.progress import say


class EvidenceVerifier:
    """Annotates opinions with per-quote verification and returns a report."""

    name = "evidence_verifier"

    def __init__(
        self, trace: RunTrace | None = None, settings: Settings | None = None
    ) -> None:
        self.trace = trace or RunTrace()
        self.settings = settings or get_settings()

    def run(
        self, opinions: list[DimensionOpinion], source_text: str
    ) -> VerificationReport:
        with span_recorder(
            self.trace, f"agent.{self.name}", kind="guardrail"
        ) as span:
            report = build_report(
                opinions,
                source_text,
                threshold=self.settings.groundedness_fuzzy_threshold,
            )
            span.attributes.update(
                {
                    "groundedness": report.groundedness,
                    "claims": report.total_claims,
                    "verified": report.verified_claims,
                    "unverified": len(report.unverified),
                }
            )
            if report.groundedness < self.settings.groundedness_warn_below:
                span.status = "degraded"
            say(
                f"ok {self.name} — groundedness={report.groundedness:.0%} "
                f"({report.verified_claims}/{report.total_claims})"
            )
            return report
