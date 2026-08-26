"""Typed contracts shared by every agent, guardrail, and evaluator."""

from ranking_agent.models.jd import Capability, JobProfile
from ranking_agent.models.resume import (
    ContactInfo,
    EducationEntry,
    ExperienceEntry,
    ProjectEntry,
    ResumeProfile,
)
from ranking_agent.models.scoring import (
    ConsistencyReport,
    CriticVerdict,
    DimensionOpinion,
    DimensionScore,
    Evidence,
    GuardrailReport,
    JudgeReport,
    Penalty,
    RankingSession,
    RankResult,
    SkillMatch,
    VerificationReport,
)
from ranking_agent.models.trace import RunTrace, Span, span_recorder

__all__ = [
    "Capability",
    "ConsistencyReport",
    "ContactInfo",
    "CriticVerdict",
    "DimensionOpinion",
    "DimensionScore",
    "EducationEntry",
    "Evidence",
    "ExperienceEntry",
    "GuardrailReport",
    "JobProfile",
    "JudgeReport",
    "Penalty",
    "ProjectEntry",
    "RankResult",
    "RankingSession",
    "ResumeProfile",
    "RunTrace",
    "SkillMatch",
    "Span",
    "VerificationReport",
    "span_recorder",
]
