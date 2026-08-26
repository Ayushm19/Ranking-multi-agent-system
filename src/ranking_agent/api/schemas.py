"""Request/response schemas for the HTTP layer."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ranking_agent.models.jd import JobProfile
from ranking_agent.models.scoring import (
    ConsistencyReport,
    JudgeReport,
    RankingSession,
    RankResult,
)


class CandidateTextInput(BaseModel):
    id: str = Field(..., description="Identifier echoed back as source_file")
    text: str = Field(..., description="Plain-text resume")


class RankTextRequest(BaseModel):
    """Text-based ranking — used by tests, the eval harness, and non-PDF callers."""

    jd_text: str
    candidates: list[CandidateTextInput]
    include_trace: bool = True


class AnalyzeJDRequest(BaseModel):
    jd_text: str


class EvaluateRequest(BaseModel):
    """Run the independent quality checks on one candidate."""

    jd_text: str
    resume_text: str
    run_judge: bool = True
    run_consistency: bool = False
    consistency_runs: int | None = None


class EvaluateResponse(BaseModel):
    result: RankResult
    judge: JudgeReport | None = None
    consistency: ConsistencyReport | None = None


class GoldenEvalRequest(BaseModel):
    """Inline golden dataset, or a path to one on disk."""

    dataset: dict | None = None
    dataset_path: str | None = None
    band: float = 10.0


class HealthResponse(BaseModel):
    status: str
    version: str
    provider: str
    model: str
    critic_enabled: bool
    pii_redaction: bool
    bias_screen: bool
    max_candidates: int


__all__ = [
    "AnalyzeJDRequest",
    "CandidateTextInput",
    "EvaluateRequest",
    "EvaluateResponse",
    "GoldenEvalRequest",
    "HealthResponse",
    "JobProfile",
    "RankTextRequest",
    "RankingSession",
]
