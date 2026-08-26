"""Ranking-quality metrics against human labels (no LLM)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class RankingMetrics:
    n: int = 0
    ndcg_at_k: float = 0.0
    k: int = 0
    spearman: float = 0.0
    pairwise_accuracy: float = 0.0
    mae: float = 0.0
    within_band: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            f"ndcg@{self.k}": round(self.ndcg_at_k, 4),
            "spearman": round(self.spearman, 4),
            "pairwise_accuracy": round(self.pairwise_accuracy, 4),
            "mae": round(self.mae, 3),
            "within_band": round(self.within_band, 4),
            "notes": self.notes,
        }


def dcg(relevances: list[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg(predicted: list[float], relevance: list[float], k: int | None = None) -> float:
    """Normalised discounted cumulative gain.

    Chosen as the primary ordering metric because ranking errors at the top of a
    shortlist matter far more than errors at the bottom, and NDCG's positional
    discount reflects that.
    """
    if not predicted or len(predicted) != len(relevance):
        return 0.0
    cut = k or len(predicted)

    order = sorted(range(len(predicted)), key=lambda i: predicted[i], reverse=True)
    ranked_rel = [relevance[i] for i in order][:cut]
    ideal_rel = sorted(relevance, reverse=True)[:cut]

    ideal = dcg(ideal_rel)
    return dcg(ranked_rel) / ideal if ideal > 0 else 0.0


def _ranks(values: list[float]) -> list[float]:
    """Average ranks, so ties do not distort the correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for idx in order[i : j + 1]:
            ranks[idx] = avg
        i = j + 1
    return ranks


def spearman(predicted: list[float], actual: list[float]) -> float:
    """Spearman rank correlation in [-1, 1]."""
    n = len(predicted)
    if n < 2 or n != len(actual):
        return 0.0
    rp, ra = _ranks(predicted), _ranks(actual)
    mp, ma = sum(rp) / n, sum(ra) / n
    num = sum((rp[i] - mp) * (ra[i] - ma) for i in range(n))
    den_p = math.sqrt(sum((rp[i] - mp) ** 2 for i in range(n)))
    den_a = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n)))
    if den_p == 0 or den_a == 0:
        return 0.0
    return max(-1.0, min(1.0, num / (den_p * den_a)))  # clamp fp drift


def pairwise_accuracy(
    predicted: list[float], actual: list[float], *, tolerance: float = 0.0
) -> float:
    """Fraction of candidate pairs ordered correctly."""
    n = len(predicted)
    if n < 2 or n != len(actual):
        return 0.0
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(actual[i] - actual[j]) <= tolerance:
                continue
            total += 1
            actual_order = actual[i] > actual[j]
            pred_order = predicted[i] > predicted[j]
            if actual_order == pred_order:
                correct += 1
    return correct / total if total else 0.0


def mean_absolute_error(predicted: list[float], actual: list[float]) -> float:
    if not predicted or len(predicted) != len(actual):
        return 0.0
    return sum(abs(p - a) for p, a in zip(predicted, actual, strict=True)) / len(predicted)


def within_band(
    predicted: list[float], actual: list[float], *, band: float = 10.0
) -> float:
    """Fraction of predictions within ``band`` points of the label.

    Calibration matters separately from ordering: a shortlist threshold ("review
    everyone above 70") is meaningless if the absolute scale drifts.
    """
    if not predicted or len(predicted) != len(actual):
        return 0.0
    hits = sum(
        1 for p, a in zip(predicted, actual, strict=True) if abs(p - a) <= band
    )
    return hits / len(predicted)


def evaluate_ranking(
    predicted: list[float],
    actual: list[float],
    *,
    k: int | None = None,
    band: float = 10.0,
) -> RankingMetrics:
    """Compute the full metric set for one JD's candidate set."""
    n = len(predicted)
    notes: list[str] = []
    if n != len(actual):
        return RankingMetrics(notes=["predicted/actual length mismatch"])
    if n < 2:
        notes.append("fewer than 2 candidates — ordering metrics are not meaningful")

    cut = k or n
    return RankingMetrics(
        n=n,
        ndcg_at_k=ndcg(predicted, actual, cut),
        k=cut,
        spearman=spearman(predicted, actual),
        pairwise_accuracy=pairwise_accuracy(predicted, actual),
        mae=mean_absolute_error(predicted, actual),
        within_band=within_band(predicted, actual, band=band),
        notes=notes,
    )
