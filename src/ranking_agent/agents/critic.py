"""Critic agent — reviews the assembled result; may trigger one repair loop."""

from __future__ import annotations

from typing import Any

from ranking_agent.agents.base import BaseAgent
from ranking_agent.models.jd import JobProfile
from ranking_agent.models.scoring import CriticVerdict, RankResult

_SYSTEM = """\
You are the Critic in a resume-ranking system. You review a completed evaluation
for defensibility — you do NOT re-score the candidate.

Return ONLY:
{
  "verdict": "accept"|"revise"|"reject",
  "confidence": number,
  "issues": [string],
  "suggested_dimension_revisions": {"<dimension>": number},
  "reasoning": string
}

Use:
- "accept" when the score is defensible from the evidence shown.
- "revise" when a specific dimension is inconsistent with its own evidence, or
  dimensions contradict each other without explanation. List the dimension in
  suggested_dimension_revisions with the score you would defend.
- "reject" only when the evaluation is unusable: no verified evidence at all, or
  reasoning that contradicts the resume.

Check for:
1. Dimensions scored high while their evidence was unverified.
2. Dimension scores that contradict each other with no stated reason.
3. Reasoning that asserts experience the evidence does not show.
4. Any reasoning touching age, gender, nationality, religion, family status,
   disability, or college prestige — always an issue, regardless of score.
5. Scores clustered at exactly 50 (a sign of fallback placeholders, not judgement).

Be concise. Do not restate the resume.
"""


class CriticAgent(BaseAgent[CriticVerdict]):
    """Self-review pass over the assembled result."""

    name = "critic"
    task = "critic"

    def system_prompt(self) -> str:
        return _SYSTEM

    def user_prompt(self, *, result: RankResult, jd: JobProfile, **_: Any) -> str:
        lines = [
            "[JOB]",
            f"Title: {jd.role_title or 'unspecified'} | Seniority: {jd.seniority or 'unspecified'}",
            f"Must-haves: {', '.join(c.name for c in jd.must_haves()) or 'unspecified'}",
            "",
            "[EVALUATION]",
            f"Final score: {result.final_score} "
            f"(level {result.experience_level}, "
            f"{result.candidate_years if result.candidate_years is not None else '?'} yrs)",
            f"Groundedness: {result.verification.groundedness:.0%} "
            f"({result.verification.verified_claims}/"
            f"{result.verification.total_claims} quotes verified)",
            "",
            "Dimensions:",
        ]
        for dim in result.dimension_scores:
            flag = " [DEGRADED/fallback]" if dim.degraded else ""
            lines.append(
                f"- {dim.label}: raw={dim.raw_score} weight={dim.weight} "
                f"grounded={dim.groundedness:.0%}{flag}"
            )
            if dim.reasoning:
                lines.append(f"    reasoning: {dim.reasoning[:320]}")
            for quote in dim.evidence[:2]:
                lines.append(f"    evidence: {quote[:180]}")

        if result.penalties:
            lines.append("\nPenalties:")
            lines.extend(
                f"- {p.name}: {p.reason} ({p.score_before} -> {p.score_after})"
                for p in result.penalties
            )
        gaps = [m.skill for m in result.gaps()]
        if gaps:
            lines.append("\nUnmatched required skills: " + ", ".join(gaps[:15]))
        if result.verification.notes:
            lines.append("\nVerification notes: " + "; ".join(result.verification.notes[:6]))
        return "\n".join(lines)

    def parse(self, payload: dict[str, Any], **_: Any) -> CriticVerdict:
        verdict = str(payload.get("verdict") or "accept").lower()
        if verdict not in ("accept", "revise", "reject"):
            verdict = "accept"

        revisions: dict[str, float] = {}
        raw_revisions = payload.get("suggested_dimension_revisions")
        if isinstance(raw_revisions, dict):
            for key, value in raw_revisions.items():
                score = self.as_float(value, -1.0)
                if 0.0 <= score <= 100.0:
                    revisions[str(key)] = score

        return CriticVerdict(
            verdict=verdict,  # type: ignore[arg-type]
            confidence=self.as_float(payload.get("confidence"), 0.5),
            issues=self.as_str_list(payload.get("issues"), 12),
            suggested_dimension_revisions=revisions,
            reasoning=str(payload.get("reasoning") or "").strip(),
        )

    def fallback(self, **_: Any) -> CriticVerdict:
        """No critic means no self-review signal — accept, but say it was skipped."""
        return CriticVerdict(
            verdict="accept",
            confidence=0.0,
            issues=["critic unavailable — result returned without self-review"],
            reasoning="Critic agent could not be reached.",
        )
