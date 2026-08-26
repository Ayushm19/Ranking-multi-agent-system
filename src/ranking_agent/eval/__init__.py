"""Evaluation: judge, ranking metrics, self-consistency, golden-set harness."""

from ranking_agent.eval.consistency import measure_consistency
from ranking_agent.eval.harness import (
    CaseReport,
    HarnessReport,
    load_dataset,
    run_harness,
)
from ranking_agent.eval.judge import JudgeAgent
from ranking_agent.eval.metrics import (
    RankingMetrics,
    evaluate_ranking,
    mean_absolute_error,
    ndcg,
    pairwise_accuracy,
    spearman,
    within_band,
)

__all__ = [
    "CaseReport",
    "HarnessReport",
    "JudgeAgent",
    "RankingMetrics",
    "evaluate_ranking",
    "load_dataset",
    "mean_absolute_error",
    "measure_consistency",
    "ndcg",
    "pairwise_accuracy",
    "run_harness",
    "spearman",
    "within_band",
]
