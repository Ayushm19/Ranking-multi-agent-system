"""Deterministic scorer — applies weights/penalties; no LLM."""

from __future__ import annotations

from ranking_agent.config import (
    DIMENSION_LABELS,
    Settings,
    get_settings,
    validate_weights,
)
from ranking_agent.models.jd import JobProfile
from ranking_agent.models.resume import ResumeProfile
from ranking_agent.models.scoring import (
    DimensionOpinion,
    DimensionScore,
    Penalty,
    RankResult,
    VerificationReport,
)
from ranking_agent.models.trace import RunTrace, span_recorder
from ranking_agent.tools.dates import infer_experience_level
from ranking_agent.tools.skills import match_skills

validate_weights()


def _experience_adequacy_penalty(
    score: float, jd: JobProfile, years: float | None
) -> Penalty | None:
    """Penalise shortfall vs JD min years (capped)."""
    required = jd.required_years_min
    if required is None or years is None or years >= required:
        return None
    shortfall = required - years
    if shortfall < 0.75:
        return None
    ratio = min(shortfall / max(required, 1.0), 1.0)
    penalty = round(min(15.0, 15.0 * ratio), 2)
    after = round(max(0.0, score - penalty), 2)
    return Penalty(
        name="Experience Adequacy",
        reason=(
            f"{years} years of experience against a stated minimum of {required} "
            f"({shortfall:.1f} years short)"
        ),
        score_before=round(score, 2),
        score_after=after,
    )


def _groundedness_penalty(
    score: float, report: VerificationReport, settings: Settings
) -> Penalty | None:
    """Penalise unverifiable evidence proportional to claim volume."""
    if report.total_claims == 0:
        return None
    if report.groundedness >= settings.groundedness_warn_below:
        return None
    shortfall = settings.groundedness_warn_below - report.groundedness
    penalty = round(
        min(
            settings.max_groundedness_penalty,
            settings.max_groundedness_penalty
            * (shortfall / max(settings.groundedness_warn_below, 0.01)),
        ),
        2,
    )
    after = round(max(0.0, score - penalty), 2)
    return Penalty(
        name="Unverified Evidence",
        reason=(
            f"only {report.verified_claims}/{report.total_claims} evidence quotes "
            f"({report.groundedness:.0%}) were found in the resume"
        ),
        score_before=round(score, 2),
        score_after=after,
    )


def _missing_must_have_penalty(score: float, result_gaps: list[str]) -> Penalty | None:
    """Penalise missing must-have skills (1.2 pts each, cap 10)."""
    if not result_gaps:
        return None
    penalty = round(min(10.0, 1.2 * len(result_gaps)), 2)
    after = round(max(0.0, score - penalty), 2)
    return Penalty(
        name="Missing Must-Have Skills",
        reason=f"{len(result_gaps)} required skills absent (−{penalty}): {', '.join(result_gaps[:8])}",
        score_before=round(score, 2),
        score_after=after,
    )


def score(
    *,
    jd: JobProfile,
    resume: ResumeProfile,
    opinions: list[DimensionOpinion],
    verification: VerificationReport,
    trace: RunTrace | None = None,
    settings: Settings | None = None,
) -> RankResult:
    """Combine agent opinions into a final score."""
    s = settings or get_settings()
    tr = trace or RunTrace()

    with span_recorder(tr, "scorer", kind="step") as span:
        years = resume.total_years_experience
        level = infer_experience_level(years)
        weights = s.weights_for(level)

        by_dim = {o.dimension: o for o in opinions}
        dimension_scores: list[DimensionScore] = []
        for key, weight in weights.items():
            opinion = by_dim.get(key)
            if opinion is None:
                # Missing opinion: still emit the dimension so weights stay complete.
                dimension_scores.append(
                    DimensionScore(
                        dimension=key,
                        label=DIMENSION_LABELS.get(key, key),
                        raw_score=50.0,
                        weight=weight,
                        weighted_score=round(50.0 * weight, 2),
                        reasoning="No evaluator opinion available.",
                        degraded=True,
                    )
                )
                continue
            dimension_scores.append(
                DimensionScore(
                    dimension=key,
                    label=DIMENSION_LABELS.get(key, key),
                    raw_score=round(opinion.raw_score, 1),
                    weight=weight,
                    weighted_score=round(opinion.raw_score * weight, 2),
                    reasoning=opinion.reasoning,
                    evidence=[e.quote for e in opinion.evidence if e.verified],
                    groundedness=round(opinion.groundedness, 4),
                    degraded=opinion.degraded,
                )
            )

        base = round(sum(d.weighted_score for d in dimension_scores), 2)

        skill_matches = next(
            (o.skill_matches for o in opinions if o.skill_matches), []
        )
        # Absence penalty applies to must-haves only.
        must_labels = {
            m.skill
            for m in match_skills([c.name for c in jd.must_haves()], [])
        }
        gaps = [
            m.skill
            for m in skill_matches
            if m.match_type == "gap" and m.skill in must_labels
        ]

        current = base
        penalties: list[Penalty] = []
        for candidate_penalty in (
            _groundedness_penalty(current, verification, s),
            _experience_adequacy_penalty(current, jd, years),
            _missing_must_have_penalty(current, gaps),
        ):
            if candidate_penalty is not None:
                penalties.append(candidate_penalty)
                current = candidate_penalty.score_after

        final = round(max(0.0, min(100.0, current)), 2)
        span.attributes.update(
            {
                "level": level,
                "base": base,
                "final": final,
                "penalties": len(penalties),
            }
        )

        return RankResult(
            candidate_name=resume.contact.name,
            candidate_email=resume.contact.email,
            final_score=final,
            experience_level=level,
            candidate_years=years,
            dimension_scores=dimension_scores,
            skill_matches=skill_matches,
            penalties=penalties,
            verification=verification,
            trace=tr,
        )
