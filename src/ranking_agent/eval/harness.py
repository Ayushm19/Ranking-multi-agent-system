"""Golden-set regression harness.

Dataset format (JSON)::

    {
      "name": "backend-2026-q1",
      "cases": [
        {
          "jd": "...job description text...",
          "candidates": [
            {"id": "alice", "text": "...resume...", "label": 90},
            {"id": "bob",   "text": "...resume...", "label": 45}
          ]
        }
      ]
    }

``label`` is the human score (0-100).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ranking_agent.config import Settings, get_settings
from ranking_agent.eval.metrics import RankingMetrics, evaluate_ranking
from ranking_agent.llm.client import LLMClient
from ranking_agent.orchestrator.batch import BatchRanker, CandidateInput

logger = logging.getLogger(__name__)


@dataclass
class CaseReport:
    jd_preview: str
    metrics: RankingMetrics
    predicted: dict[str, float]
    labels: dict[str, float]
    errors: list[str] = field(default_factory=list)
    mean_groundedness: float = 1.0
    review_flagged: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "jd_preview": self.jd_preview,
            "metrics": self.metrics.as_dict(),
            "predicted": self.predicted,
            "labels": self.labels,
            "errors": self.errors,
            "mean_groundedness": round(self.mean_groundedness, 4),
            "review_flagged": self.review_flagged,
        }


@dataclass
class HarnessReport:
    dataset: str
    cases: list[CaseReport] = field(default_factory=list)
    total_cost_usd: float = 0.0

    @property
    def aggregate(self) -> dict[str, float]:
        """Macro-averaged metrics across cases."""
        scored = [c for c in self.cases if c.metrics.n >= 2]
        if not scored:
            return {}
        count = len(scored)
        return {
            "cases": float(count),
            "ndcg": round(sum(c.metrics.ndcg_at_k for c in scored) / count, 4),
            "spearman": round(sum(c.metrics.spearman for c in scored) / count, 4),
            "pairwise_accuracy": round(
                sum(c.metrics.pairwise_accuracy for c in scored) / count, 4
            ),
            "mae": round(sum(c.metrics.mae for c in scored) / count, 3),
            "within_band": round(sum(c.metrics.within_band for c in scored) / count, 4),
            "mean_groundedness": round(
                sum(c.mean_groundedness for c in scored) / count, 4
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "aggregate": self.aggregate,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "cases": [c.as_dict() for c in self.cases],
        }


def load_dataset(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "cases" not in data:
        raise ValueError("dataset must be an object with a 'cases' array")
    return data


async def run_harness(
    dataset: dict[str, Any],
    *,
    settings: Settings | None = None,
    band: float = 10.0,
) -> HarnessReport:
    """Run every case and collect metrics."""
    s = settings or get_settings()
    report = HarnessReport(dataset=str(dataset.get("name") or "unnamed"))

    for case in dataset.get("cases") or []:
        jd_text = str(case.get("jd") or "")
        raw_candidates = case.get("candidates") or []
        if not jd_text or not raw_candidates:
            logger.warning("skipping case with missing jd or candidates")
            continue

        labels = {
            str(c.get("id") or f"cand{i}"): float(c.get("label") or 0.0)
            for i, c in enumerate(raw_candidates)
        }
        inputs = [
            CandidateInput(
                filename=str(c.get("id") or f"cand{i}") + ".txt",
                text=str(c.get("text") or ""),
            )
            for i, c in enumerate(raw_candidates)
        ]

        ranker = BatchRanker(client=LLMClient(settings=s), settings=s)
        session = await ranker.rank(
            jd_text=jd_text, candidates=inputs, include_trace=False
        )

        predicted = {
            (r.source_file or "").removesuffix(".txt"): r.final_score
            for r in session.results
        }
        # Skip rejected docs (no score) so guardrails don't look like zero ranks.
        common = [cid for cid in labels if cid in predicted]
        metrics = evaluate_ranking(
            [predicted[c] for c in common],
            [labels[c] for c in common],
            band=band,
        )

        grounded = [r.verification.groundedness for r in session.results]
        report.cases.append(
            CaseReport(
                jd_preview=jd_text[:120],
                metrics=metrics,
                predicted={c: predicted[c] for c in common},
                labels={c: labels[c] for c in common},
                errors=list(session.processing_errors),
                mean_groundedness=(sum(grounded) / len(grounded)) if grounded else 1.0,
                review_flagged=sum(1 for r in session.results if r.needs_human_review),
            )
        )
        report.total_cost_usd += session.total_cost_usd

    return report
