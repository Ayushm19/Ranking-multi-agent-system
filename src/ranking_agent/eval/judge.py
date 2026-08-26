"""LLM-as-judge — independent correctness check; does not mutate scores."""

from __future__ import annotations

from typing import Any

from ranking_agent.agents.base import BaseAgent
from ranking_agent.models.jd import JobProfile
from ranking_agent.models.resume import ResumeProfile
from ranking_agent.models.scoring import JudgeReport, RankResult

_SYSTEM = """\
You are an independent Evaluation Judge auditing a resume-ranking system.

You receive a job description, a candidate resume, and the score the system
produced. Decide whether that score is defensible. You are grading the SYSTEM,
not the candidate.

Return ONLY:
{
  "verdict": "correct"|"partially_correct"|"incorrect",
  "score_error_estimate": number,
  "dimension_disagreements": [string],
  "reasoning": string
}

Definitions:
- "correct": the score is within roughly 8 points of what the evidence supports.
- "partially_correct": directionally right but miscalibrated, or one dimension is
  clearly wrong while the overall ordering would survive.
- "incorrect": the score would put this candidate in the wrong band entirely
  (a strong match scored as weak, or a weak match scored as strong).

score_error_estimate is signed: positive means the system scored too HIGH,
negative means too LOW. Use 0 when the score looks right.

List a dimension in dimension_disagreements as "<dimension>: <why>".
Judge only demonstrated capability — never age, gender, nationality, religion,
family status, disability, or college prestige.
"""


class JudgeAgent(BaseAgent[JudgeReport]):
    """Independent correctness evaluation of one produced result."""

    name = "judge"
    task = "judge"

    def model_name(self) -> str:
        return self.ctx.settings.judge_model

    def system_prompt(self) -> str:
        return _SYSTEM

    def user_prompt(
        self,
        *,
        result: RankResult,
        jd: JobProfile,
        resume: ResumeProfile,
        **_: Any,
    ) -> str:
        dims = "\n".join(
            f"- {d.label}: {d.raw_score} (weight {d.weight}"
            + (", fallback" if d.degraded else "")
            + f") — {d.reasoning[:220]}"
            for d in result.dimension_scores
        )
        penalties = (
            "\n".join(f"- {p.name}: {p.reason}" for p in result.penalties)
            or "(none)"
        )
        gaps = ", ".join(m.skill for m in result.gaps()[:15]) or "(none)"
        min_years = (
            jd.required_years_min if jd.required_years_min is not None else "unspecified"
        )
        return (
            f"[JOB]\n"
            f"Title: {jd.role_title or 'unspecified'}\n"
            f"Seniority: {jd.seniority or 'unspecified'} | "
            f"Min years: {min_years}\n"
            f"Must-haves: {', '.join(c.name for c in jd.must_haves()) or 'unspecified'}\n\n"
            f"[RESUME]\n{self.clip(resume.source_text, 12_000)}\n\n"
            f"[SYSTEM OUTPUT]\n"
            f"Final score: {result.final_score} / 100\n"
            f"Experience level: {result.experience_level} "
            f"({result.candidate_years if result.candidate_years is not None else '?'} yrs)\n"
            f"Groundedness: {result.verification.groundedness:.0%}\n"
            f"Unmatched required skills: {gaps}\n"
            f"Dimensions:\n{dims}\n"
            f"Penalties:\n{penalties}"
        )

    def parse(self, payload: dict[str, Any], **_: Any) -> JudgeReport:
        verdict = str(payload.get("verdict") or "partially_correct").lower()
        if verdict not in ("correct", "partially_correct", "incorrect"):
            verdict = "partially_correct"
        return JudgeReport(
            verdict=verdict,  # type: ignore[arg-type]
            score_error_estimate=self.as_float(payload.get("score_error_estimate")),
            dimension_disagreements=self.as_str_list(
                payload.get("dimension_disagreements"), 12
            ),
            reasoning=str(payload.get("reasoning") or "").strip(),
            judge_model=self.model_name(),
        )

    def fallback(self, **_: Any) -> JudgeReport:
        return JudgeReport(
            verdict="partially_correct",
            reasoning="Judge unavailable — no independent evaluation was performed.",
            judge_model=self.model_name(),
        )
