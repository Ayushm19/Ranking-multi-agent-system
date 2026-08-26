"""Self-consistency: K identical runs; report score spread."""

from __future__ import annotations

import asyncio
import statistics

from ranking_agent.config import Settings, get_settings
from ranking_agent.models.jd import JobProfile
from ranking_agent.models.scoring import ConsistencyReport
from ranking_agent.orchestrator.supervisor import Supervisor


async def measure_consistency(
    *,
    resume_text: str,
    job_profile: JobProfile,
    supervisor: Supervisor | None = None,
    runs: int | None = None,
    settings: Settings | None = None,
) -> ConsistencyReport:
    """Score one candidate repeatedly and report the spread."""
    s = settings or get_settings()
    n = max(2, runs or s.consistency_runs)
    sup = supervisor or Supervisor(settings=s)

    async def _one() -> float:
        result = await sup.rank_one(
            resume_text=resume_text,
            job_profile=job_profile,
            include_trace=False,
        )
        return result.final_score

    scores = list(await asyncio.gather(*(_one() for _ in range(n))))
    mean = statistics.fmean(scores)
    stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    spread = max(scores) - min(scores)

    return ConsistencyReport(
        runs=n,
        scores=[round(x, 2) for x in scores],
        mean=round(mean, 2),
        stdev=round(stdev, 3),
        spread=round(spread, 2),
        unstable=stdev > s.consistency_unstable_stdev,
    )
