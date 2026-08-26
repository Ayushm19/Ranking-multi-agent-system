"""HTTP layer."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ranking_agent import __version__
from ranking_agent.agents.base import AgentContext
from ranking_agent.api.schemas import (
    AnalyzeJDRequest,
    EvaluateRequest,
    EvaluateResponse,
    GoldenEvalRequest,
    HealthResponse,
    RankTextRequest,
)
from ranking_agent.config import get_settings
from ranking_agent.eval.consistency import measure_consistency
from ranking_agent.eval.harness import load_dataset, run_harness
from ranking_agent.eval.judge import JudgeAgent
from ranking_agent.guardrails.pii import redact
from ranking_agent.llm.client import LLMClient
from ranking_agent.models.jd import JobProfile
from ranking_agent.models.scoring import RankingSession
from ranking_agent.observability import configure_logging, log_trace
from ranking_agent.orchestrator.batch import BatchRanker, CandidateInput
from ranking_agent.orchestrator.supervisor import GuardrailRejection, Supervisor
from ranking_agent.progress import say
from ranking_agent.tools.pdf import clean_text

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    configure_logging()
    say(
        f"ranking-agent {__version__} starting "
        f"(provider={settings.provider} model={settings.model})"
    )
    yield
    say("ranking-agent shutting down")


app = FastAPI(
    title="Multi-Agent Resume Ranking",
    version=__version__,
    description=(
        "Multi-agent resume ranking with guardrails, evidence verification, "
        "self-critique, and built-in evaluation."
    ),
    lifespan=lifespan,
)

# ── Web UI (explicit routes; avoid mounting at "/") ──
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

if (WEB_DIR / "static").is_dir():
    app.mount(
        "/static", StaticFiles(directory=WEB_DIR / "static"), name="static"
    )


@app.get("/", include_in_schema=False)
async def landing_page() -> FileResponse:
    """Landing page with the animated pipeline demo."""
    return FileResponse(WEB_DIR / "index.html")


@app.get("/app", include_in_schema=False)
async def workbench_page() -> FileResponse:
    """Paste a JD and resumes, see the full scored output."""
    return FileResponse(WEB_DIR / "app.html")


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    s = get_settings()
    return HealthResponse(
        status="ok",
        version=__version__,
        provider=s.provider,
        model=s.model,
        critic_enabled=s.enable_critic,
        pii_redaction=s.enable_pii_redaction,
        bias_screen=s.enable_bias_screen,
        max_candidates=s.max_candidates_per_batch,
    )


@app.post("/analyze-jd", response_model=JobProfile)
async def analyze_jd(req: AnalyzeJDRequest) -> JobProfile:
    """JD Analyst agent on its own."""
    try:
        return await Supervisor().analyze_jd(req.jd_text)
    except GuardrailRejection as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/rank", response_model=RankingSession)
async def rank(
    jd_text: Annotated[str, Form()],
    files: Annotated[list[UploadFile], File()],
) -> RankingSession:
    """Rank uploaded resumes against one JD (multipart ``jd_text`` + ``files``)."""
    settings = get_settings()
    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="jd_text cannot be empty")
    if not files:
        raise HTTPException(status_code=400, detail="at least one file is required")
    if len(files) > settings.max_candidates_per_batch:
        raise HTTPException(
            status_code=400,
            detail=f"maximum {settings.max_candidates_per_batch} resumes per batch",
        )

    candidates: list[CandidateInput] = []
    for upload in files:
        raw = await upload.read()
        candidates.append(
            CandidateInput(filename=upload.filename or "resume.pdf", raw_bytes=raw)
        )

    return await _run_batch(jd_text, candidates)


@app.post("/rank/text", response_model=RankingSession)
async def rank_text(req: RankTextRequest) -> RankingSession:
    """Rank plain-text resumes — convenient for tests and eval runs."""
    if not req.candidates:
        raise HTTPException(status_code=400, detail="at least one candidate is required")
    candidates = [
        CandidateInput(filename=f"{c.id}.txt", text=clean_text(c.text))
        for c in req.candidates
    ]
    return await _run_batch(req.jd_text, candidates)


async def _run_batch(
    jd_text: str, candidates: list[CandidateInput]
) -> RankingSession:
    settings = get_settings()
    say(
        f"rank start — {len(candidates)} candidate(s) via {settings.provider}/{settings.model}"
    )
    ranker = BatchRanker(client=LLMClient(settings=settings), settings=settings)
    try:
        session = await ranker.rank(
            jd_text=jd_text, candidates=candidates, include_trace=True
        )
    except GuardrailRejection as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    for result in session.results:
        if result.trace:
            log_trace(result.trace)
    say(
        f"rank finish — scored {len(session.results)}/{session.total_candidates} "
        f"cost=${session.total_cost_usd:.4f}"
    )
    return session


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """Score one candidate, then optional judge / consistency checks."""
    settings = get_settings()
    client = LLMClient(settings=settings)
    supervisor = Supervisor(client, settings)

    try:
        job_profile = await supervisor.analyze_jd(req.jd_text)
        result = await supervisor.rank_one(
            resume_text=clean_text(req.resume_text),
            job_profile=job_profile,
            include_trace=True,
        )
    except GuardrailRejection as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    judge_report = None
    if req.run_judge:
        redaction = redact(
            clean_text(req.resume_text), enabled=settings.enable_pii_redaction
        )
        from ranking_agent.models.resume import ResumeProfile

        judged_resume = ResumeProfile(
            contact=redaction.contact, source_text=redaction.text
        )
        ctx = AgentContext(client, result.trace or None, settings)
        judge_report = await JudgeAgent(ctx).run(
            result=result, jd=job_profile, resume=judged_resume
        )
        result.judge = judge_report

    consistency = None
    if req.run_consistency:
        consistency = await measure_consistency(
            resume_text=clean_text(req.resume_text),
            job_profile=job_profile,
            supervisor=supervisor,
            runs=req.consistency_runs,
            settings=settings,
        )
        result.consistency = consistency

    return EvaluateResponse(
        result=result, judge=judge_report, consistency=consistency
    )


@app.post("/eval/golden")
async def eval_golden(req: GoldenEvalRequest) -> dict:
    """Run the golden-set harness and return ranking metrics."""
    if not req.dataset and not req.dataset_path:
        raise HTTPException(
            status_code=400, detail="provide either dataset or dataset_path"
        )
    try:
        dataset = req.dataset or load_dataset(req.dataset_path or "")
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"bad dataset: {exc}") from exc

    report = await run_harness(dataset, settings=get_settings(), band=req.band)
    return report.as_dict()
