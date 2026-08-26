"""Output guardrails on the assembled RankResult."""

from __future__ import annotations

from ranking_agent.config import Settings, get_settings
from ranking_agent.guardrails.bias import scan_for_bias
from ranking_agent.models.scoring import GuardrailReport, RankResult


def apply_output_guards(
    result: RankResult,
    *,
    report: GuardrailReport | None = None,
    settings: Settings | None = None,
) -> tuple[RankResult, GuardrailReport]:
    """Validate / clamp the final result; flag issues needing human review."""
    s = settings or get_settings()
    rep = report or GuardrailReport()
    violations: list[str] = []
    clamped: list[str] = []
    review_reasons: list[str] = []

    # ── Score bounds ──
    if not 0.0 <= result.final_score <= 100.0:
        clamped.append(f"final_score {result.final_score}")
        result.final_score = max(0.0, min(100.0, result.final_score))

    for dim in result.dimension_scores:
        if not 0.0 <= dim.raw_score <= 100.0:
            clamped.append(f"{dim.dimension}.raw_score {dim.raw_score}")
            dim.raw_score = max(0.0, min(100.0, dim.raw_score))
            dim.weighted_score = round(dim.raw_score * dim.weight, 2)

    # ── Weight integrity ──
    weight_sum = sum(d.weight for d in result.dimension_scores)
    if result.dimension_scores and abs(weight_sum - 1.0) > 0.01:
        violations.append(
            f"dimension weights sum to {weight_sum:.3f}, expected 1.0"
        )

    # ── Evidence coverage ──
    unevidenced = [
        d.dimension
        for d in result.dimension_scores
        if not d.evidence and not d.degraded and d.raw_score >= 70
    ]
    if unevidenced:
        violations.append(
            "high scores without evidence: " + ", ".join(unevidenced)
        )

    # ── Groundedness ──
    grounded = result.verification.groundedness
    if grounded < s.groundedness_review_below:
        review_reasons.append(
            f"groundedness {grounded:.0%} below review floor "
            f"{s.groundedness_review_below:.0%}"
        )
    elif grounded < s.groundedness_warn_below:
        violations.append(f"groundedness {grounded:.0%} below target")

    # ── Bias screen ──
    if s.enable_bias_screen:
        for dim in result.dimension_scores:
            scan = scan_for_bias(dim.reasoning)
            if scan.flagged:
                rep.bias_flags.extend(scan.categories)
                dim.reasoning = scan.sanitized_text
                review_reasons.append(
                    f"{dim.dimension} reasoning cited protected attributes "
                    f"({', '.join(scan.categories)})"
                )
        rep.bias_flags = sorted(set(rep.bias_flags))

    # ── Degraded agents ──
    degraded = [d.dimension for d in result.dimension_scores if d.degraded]
    if len(degraded) >= 3:
        review_reasons.append(
            f"{len(degraded)} of {len(result.dimension_scores)} dimensions came "
            "from fallbacks, not model judgements"
        )

    rep.output_violations.extend(violations)
    rep.clamped_scores.extend(clamped)

    if review_reasons:
        result.needs_human_review = True
        result.review_reasons.extend(review_reasons)

    result.guardrails = rep
    return result, rep
