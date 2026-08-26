"""Batch ranking — many candidates against one JD."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from ranking_agent.config import Settings, get_settings
from ranking_agent.guardrails.input_guard import check_document
from ranking_agent.llm.client import LLMClient
from ranking_agent.models.scoring import RankingSession, RankResult
from ranking_agent.orchestrator.supervisor import GuardrailRejection, Supervisor
from ranking_agent.progress import say
from ranking_agent.tools.pdf import extract_text

logger = logging.getLogger(__name__)


@dataclass
class CandidateInput:
    """One candidate, as either a file or pre-extracted text."""

    filename: str
    raw_bytes: bytes | None = None
    text: str | None = None

    def resolve_text(self) -> str:
        if self.text is not None:
            return self.text
        if self.raw_bytes is None:
            raise ValueError(f"{self.filename}: no bytes and no text provided")
        return extract_text(self.filename, self.raw_bytes)


class BatchRanker:
    """Ranks a set of candidates against one job description."""

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or LLMClient(settings=self.settings)
        self.supervisor = Supervisor(self.client, self.settings)

    async def rank(
        self,
        *,
        jd_text: str,
        candidates: list[CandidateInput],
        include_trace: bool = True,
    ) -> RankingSession:
        t0 = time.perf_counter()

        if not candidates:
            raise ValueError("at least one candidate is required")
        if len(candidates) > self.settings.max_candidates_per_batch:
            raise ValueError(
                f"maximum {self.settings.max_candidates_per_batch} candidates per batch"
            )

        job_profile = await self.supervisor.analyze_jd(jd_text)

        sem = asyncio.Semaphore(self.settings.batch_concurrency)
        errors: list[str] = []

        async def _one(candidate: CandidateInput) -> RankResult | None:
            async with sem:
                try:
                    doc_guard = check_document(
                        candidate.filename,
                        candidate.raw_bytes if candidate.text is None else None,
                        self.settings,
                    )
                    if not doc_guard.ok:
                        raise GuardrailRejection("; ".join(doc_guard.failures))

                    text = await asyncio.to_thread(candidate.resolve_text)
                    return await self.supervisor.rank_one(
                        resume_text=text,
                        job_profile=job_profile,
                        source_file=candidate.filename,
                        include_trace=include_trace,
                    )
                except GuardrailRejection as exc:
                    logger.warning("rejected %s: %s", candidate.filename, exc)
                    errors.append(f"{candidate.filename}: rejected — {exc}")
                    return None
                except Exception as exc:
                    logger.exception("failed %s", candidate.filename)
                    errors.append(f"{candidate.filename}: {type(exc).__name__}: {exc}")
                    return None

        outcomes = await asyncio.gather(*(_one(c) for c in candidates))
        results = [r for r in outcomes if r is not None]

        results.sort(key=lambda r: r.final_score, reverse=True)
        for i, result in enumerate(results, start=1):
            result.rank = i

        duration_ms = round((time.perf_counter() - t0) * 1000)
        say(
            f"ranked {len(results)}/{len(candidates)} candidates in "
            f"{duration_ms / 1000:.2f}s (cost ${self.client.cost.cost_usd:.4f})"
        )

        return RankingSession(
            job_profile=job_profile,
            results=results,
            total_candidates=len(candidates),
            processing_errors=errors,
            total_cost_usd=round(self.client.cost.cost_usd, 6),
            duration_ms=duration_ms,
        )
