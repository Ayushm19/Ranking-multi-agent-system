"""Scoring contracts: agent opinions, verification, penalties, final result."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ranking_agent.models.jd import JobProfile
from ranking_agent.models.trace import RunTrace

MatchType = Literal["exact", "transferable", "partial", "gap"]
Verdict = Literal["accept", "revise", "reject"]
JudgeVerdict = Literal["correct", "partially_correct", "incorrect"]


class Evidence(BaseModel):
    """A quote an agent used to justify a judgement.

    ``verified`` is filled by the Evidence Verifier — a quote the agent could not
    have read in the resume is the clearest hallucination signal available.
    """

    quote: str
    claim: str | None = None
    verified: bool | None = None
    match_score: int | None = None  # 0-100 fuzzy similarity against source


class SkillMatch(BaseModel):
    skill: str
    match_type: MatchType
    score: float = 0.0
    resume_evidence: str | None = None

    @field_validator("score")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(100.0, v))


class DimensionOpinion(BaseModel):
    """One evaluator agent's bounded opinion on one dimension.

    Agents return a 0-100 raw score with evidence; they never see or set weights,
    so no single agent can move the final number on its own.
    """

    dimension: str
    raw_score: float = 50.0
    confidence: float = 0.5
    reasoning: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    agent: str = ""
    degraded: bool = False
    skill_matches: list[SkillMatch] = Field(default_factory=list)

    @field_validator("raw_score")
    @classmethod
    def _clamp_score(cls, v: float) -> float:
        return max(0.0, min(100.0, v))

    @field_validator("confidence")
    @classmethod
    def _clamp_conf(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @property
    def groundedness(self) -> float:
        checked = [e for e in self.evidence if e.verified is not None]
        if not checked:
            return 1.0
        return sum(1 for e in checked if e.verified) / len(checked)


class DimensionScore(BaseModel):
    """A weighted dimension score."""

    dimension: str
    label: str
    raw_score: float
    weight: float
    weighted_score: float
    reasoning: str = ""
    evidence: list[str] = Field(default_factory=list)
    groundedness: float = 1.0
    degraded: bool = False


class Penalty(BaseModel):
    name: str
    reason: str
    score_before: float
    score_after: float

    @property
    def delta(self) -> float:
        return round(self.score_after - self.score_before, 2)


class VerificationReport(BaseModel):
    """Output of the Evidence Verifier across all dimensions."""

    total_claims: int = 0
    verified_claims: int = 0
    unverified: list[Evidence] = Field(default_factory=list)
    groundedness: float = 1.0
    notes: list[str] = Field(default_factory=list)


class GuardrailReport(BaseModel):
    """What the guardrail layers did. Always present, even when everything passed."""

    input_ok: bool = True
    input_failures: list[str] = Field(default_factory=list)
    injection_detected: bool = False
    injection_severity: int = 0
    injection_patterns: list[str] = Field(default_factory=list)
    pii_redactions: int = 0
    bias_flags: list[str] = Field(default_factory=list)
    output_violations: list[str] = Field(default_factory=list)
    clamped_scores: list[str] = Field(default_factory=list)

    @property
    def clean(self) -> bool:
        return (
            self.input_ok
            and not self.injection_detected
            and not self.bias_flags
            and not self.output_violations
        )


class CriticVerdict(BaseModel):
    """Self-review of the assembled result before it is returned."""

    verdict: Verdict = "accept"
    confidence: float = 0.5
    issues: list[str] = Field(default_factory=list)
    suggested_dimension_revisions: dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""


class JudgeReport(BaseModel):
    """Independent evaluation of whether the produced answer is right."""

    verdict: JudgeVerdict = "partially_correct"
    score_error_estimate: float = 0.0
    dimension_disagreements: list[str] = Field(default_factory=list)
    reasoning: str = ""
    judge_model: str = ""


class ConsistencyReport(BaseModel):
    """Stability of the verdict across repeated runs."""

    runs: int = 0
    scores: list[float] = Field(default_factory=list)
    mean: float = 0.0
    stdev: float = 0.0
    spread: float = 0.0
    unstable: bool = False


class RankResult(BaseModel):
    """Final per-candidate answer."""

    rank: int = 0
    candidate_name: str | None = None
    candidate_email: str | None = None
    source_file: str | None = None

    final_score: float = 0.0
    experience_level: str = "MID"
    candidate_years: float | None = None

    dimension_scores: list[DimensionScore] = Field(default_factory=list)
    skill_matches: list[SkillMatch] = Field(default_factory=list)
    penalties: list[Penalty] = Field(default_factory=list)

    verification: VerificationReport = Field(default_factory=VerificationReport)
    guardrails: GuardrailReport = Field(default_factory=GuardrailReport)
    critic: CriticVerdict | None = None
    judge: JudgeReport | None = None
    consistency: ConsistencyReport | None = None

    needs_human_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    repair_loops: int = 0

    trace: RunTrace | None = None
    cost_usd: float = 0.0

    @property
    def score_before_penalties(self) -> float:
        return round(sum(d.weighted_score for d in self.dimension_scores), 2)

    def gaps(self) -> list[SkillMatch]:
        return [m for m in self.skill_matches if m.match_type == "gap"]


class RankingSession(BaseModel):
    """Batch response — same external shape as the API this replaces."""

    job_profile: JobProfile
    results: list[RankResult] = Field(default_factory=list)
    total_candidates: int = 0
    processing_errors: list[str] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    duration_ms: int = 0
