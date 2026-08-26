"""Evaluator agents — one per scoring dimension; each returns a DimensionOpinion."""

from __future__ import annotations

from typing import Any

from ranking_agent.agents.base import BaseAgent
from ranking_agent.models.jd import JobProfile
from ranking_agent.models.resume import ResumeProfile
from ranking_agent.models.scoring import DimensionOpinion, Evidence, SkillMatch
from ranking_agent.tools.skills import coverage_score, match_skills

_SHARED_RULES = """\
Rules that apply to every judgement:
1. Score 0-100. Anchors: 90+ = direct, demonstrated match; 70-89 = strong with
   minor gaps; 50-69 = partial/adjacent; 30-49 = weak; below 30 = essentially absent.
2. Every claim must be supported by a VERBATIM quote from the resume. Copy the
   quote exactly — it is automatically checked against the source document, and an
   unverifiable quote reduces the candidate's score.
3. If the resume does not support a judgement, score low and say what is missing.
   Never invent experience the resume does not state.
4. NEVER reason about age, gender, nationality, ethnicity, religion, marital or
   family status, disability, or university prestige. Judge only demonstrated
   capability.
5. Return ONLY this JSON object:
{
  "raw_score": number,
  "confidence": number,
  "reasoning": string,
  "evidence": [{"quote": string, "claim": string}]
}"""


def _dimension_system(role: str, focus: str) -> str:
    return f"You are the {role} in a resume-ranking system.\n\n{focus}\n\n{_SHARED_RULES}"


def _jd_block(jd: JobProfile) -> str:
    must = ", ".join(c.name for c in jd.must_haves()) or "(not specified)"
    nice = ", ".join(c.name for c in jd.nice_to_haves()) or "(none)"
    years = jd.required_years_min if jd.required_years_min is not None else "unspecified"
    lines = [
        "[JOB]",
        f"Title: {jd.role_title or 'unspecified'}",
        f"Seniority: {jd.seniority or 'unspecified'}",
        f"Required years (min): {years}",
        f"Domain: {jd.domain or 'unspecified'} | Industry: {jd.industry or 'unspecified'}",
        f"Must-have capabilities: {must}",
        f"Nice-to-have capabilities: {nice}",
    ]
    if jd.responsibilities:
        lines.append("Responsibilities: " + "; ".join(jd.responsibilities[:12]))
    if jd.scale_signals:
        lines.append("Scale signals: " + "; ".join(jd.scale_signals[:8]))
    return "\n".join(lines)


def _resume_block(resume: ResumeProfile, limit: int = 14_000) -> str:
    parts = ["[RESUME]"]
    if resume.contact.name:
        parts.append(f"Candidate: {resume.contact.name}")
    if resume.total_years_experience is not None:
        parts.append(f"Total experience (computed): {resume.total_years_experience} years")
    if resume.summary:
        parts.append(f"Summary: {resume.summary}")
    if resume.skills:
        parts.append("Skills: " + ", ".join(resume.skills[:80]))
    for exp in resume.experiences[:12]:
        head = " | ".join(
            p
            for p in (
                exp.title,
                exp.company,
                f"{exp.start_date or '?'} - "
                f"{'Present' if exp.is_current else (exp.end_date or '?')}",
            )
            if p
        )
        parts.append(f"\nRole: {head}")
        if exp.skills:
            parts.append("  Technologies: " + ", ".join(exp.skills[:30]))
        if exp.description:
            parts.append(f"  {exp.description}")
    for proj in resume.projects[:8]:
        parts.append(f"\nProject: {proj.name or 'unnamed'}")
        if proj.tech_stack:
            parts.append("  Tech: " + ", ".join(proj.tech_stack[:25]))
        if proj.description:
            parts.append(f"  {proj.description}")
    for edu in resume.educations[:5]:
        parts.append(
            "Education: "
            + " | ".join(
                p for p in (edu.degree, edu.institution, edu.end_date) if p
            )
        )

    body = "\n".join(parts)
    if len(body) < 400 and resume.source_text:
        # Thin parse → fall back to raw resume text.
        body = f"[RESUME]\n{resume.source_text}"
    return body[:limit]


class DimensionEvaluator(BaseAgent[DimensionOpinion]):
    """Shared machinery for the four LLM evaluator agents."""

    dimension: str = "role_fit"
    role_label: str = "Evaluator"
    focus: str = ""

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"eval_{self.dimension}"

    @property
    def task(self) -> str:  # type: ignore[override]
        return self.dimension

    def system_prompt(self) -> str:
        return _dimension_system(self.role_label, self.focus)

    def user_prompt(
        self, *, jd: JobProfile, resume: ResumeProfile, **_: Any
    ) -> str:
        return f"{_jd_block(jd)}\n\n{_resume_block(resume)}"

    def parse(self, payload: dict[str, Any], **_: Any) -> DimensionOpinion:
        evidence = [
            Evidence(
                quote=str(item.get("quote") or "").strip(),
                claim=(str(item["claim"]).strip() if item.get("claim") else None),
            )
            for item in (payload.get("evidence") or [])
            if isinstance(item, dict) and str(item.get("quote") or "").strip()
        ]
        return DimensionOpinion(
            dimension=self.dimension,
            raw_score=self.as_float(payload.get("raw_score"), 50.0),
            confidence=self.as_float(payload.get("confidence"), 0.5),
            reasoning=str(payload.get("reasoning") or "").strip(),
            evidence=evidence[:8],
            agent=self.name,
        )

    def fallback(self, **_: Any) -> DimensionOpinion:
        """Neutral degraded opinion (score 50)."""
        return DimensionOpinion(
            dimension=self.dimension,
            raw_score=50.0,
            confidence=0.0,
            reasoning=f"{self.name} unavailable — neutral placeholder score applied.",
            agent=self.name,
            degraded=True,
        )


class RoleFitEvaluator(DimensionEvaluator):
    dimension = "role_fit"
    role_label = "Role Fit Evaluator"
    focus = (
        "Judge whether the candidate's career is the same KIND of work as the job, "
        "at a comparable level of responsibility. A backend engineer applying to a "
        "backend role is a strong fit; a QA engineer applying to a backend role is "
        "not, however skilled they are. Seniority mismatch in either direction "
        "reduces the score."
    )


class CapabilityEvaluator(DimensionEvaluator):
    dimension = "capability_match"
    role_label = "Capability Match Evaluator"
    focus = (
        "Judge coverage of the JD's must-have CAPABILITIES — what the person must "
        "be able to do (design systems, own delivery, lead migrations), not which "
        "tools they list. Weigh demonstrated ownership above exposure."
    )


class ExperienceQualityEvaluator(DimensionEvaluator):
    dimension = "experience_quality"
    role_label = "Experience Quality Evaluator"
    focus = (
        "Judge depth and trajectory: tenure length, progression, scope growth, and "
        "whether the described work is substantive rather than a list of tools. "
        "Frequent very short stints with no progression lower the score; sustained "
        "ownership and promotion raise it."
    )


class ImpactEvaluator(DimensionEvaluator):
    dimension = "scale_impact"
    role_label = "Scale & Impact Evaluator"
    focus = (
        "Judge the scale the candidate has operated at and the impact they can "
        "evidence: users, traffic, data volume, team size, cost or latency "
        "reduction. Quantified outcomes score high; unquantified responsibility "
        "claims score mid; no evidence of scale scores low."
    )


class DomainEvaluator(DimensionEvaluator):
    dimension = "domain_context"
    role_label = "Domain & Context Evaluator"
    focus = (
        "Judge alignment of industry, domain, and working context (regulated vs "
        "consumer, product vs services, startup vs enterprise). Adjacent domains "
        "with transferable constraints score mid, not low."
    )


class SkillMatchEvaluator(DimensionEvaluator):
    """Skill matching: deterministic coverage first, LLM only for transferability.

    Exact skill overlap is a lookup, so it is computed in code and cannot drift.
    The model is asked only to judge whether unmatched requirements are covered by
    transferable experience — the part that actually needs reasoning.
    """

    dimension = "skill_match"
    role_label = "Skill Match Evaluator"
    focus = (
        "You are given a deterministic skill-coverage report. Your job is to judge "
        "whether the GAPS are genuinely disqualifying or are covered by "
        "transferable experience evidenced in the resume, and to return an adjusted "
        "overall skill score. Also return a 'skill_matches' array with any gap you "
        "believe should be reclassified: "
        '[{"skill": string, "match_type": "exact"|"transferable"|"partial"|"gap", '
        '"score": number, "resume_evidence": string}]'
    )

    def user_prompt(
        self, *, jd: JobProfile, resume: ResumeProfile, **_: Any
    ) -> str:
        matches = self.deterministic_matches(jd, resume)
        report = "\n".join(
            f"  - {m.skill}: {m.match_type}"
            + (f" (via {m.resume_evidence})" if m.resume_evidence else "")
            for m in matches
        ) or "  (JD listed no explicit skills)"
        return (
            f"{_jd_block(jd)}\n\n"
            f"[DETERMINISTIC SKILL COVERAGE] baseline score "
            f"{coverage_score(matches)}\n{report}\n\n"
            f"{_resume_block(resume)}"
        )

    def parse(self, payload: dict[str, Any], **kwargs: Any) -> DimensionOpinion:
        opinion = super().parse(payload, **kwargs)
        jd: JobProfile = kwargs["jd"]
        resume: ResumeProfile = kwargs["resume"]
        baseline = self.deterministic_matches(jd, resume)

        overrides = {
            str(raw.get("skill", "")).strip().lower(): raw
            for raw in (payload.get("skill_matches") or [])
            if isinstance(raw, dict) and raw.get("skill")
        }
        merged: list[SkillMatch] = []
        for match in baseline:
            raw = overrides.get(match.skill.lower())
            # Model may only upgrade deterministic gaps, not downgrade matches.
            if raw and match.match_type == "gap":
                mtype = str(raw.get("match_type") or "gap").lower()
                if mtype in ("exact", "transferable", "partial"):
                    default_score = {
                        "exact": 100.0,
                        "transferable": 70.0,
                        "partial": 45.0,
                    }[mtype]
                    merged.append(
                        SkillMatch(
                            skill=match.skill,
                            match_type=mtype,  # type: ignore[arg-type]
                            score=self.as_float(raw.get("score"), default_score),
                            resume_evidence=(
                                str(raw["resume_evidence"])
                                if raw.get("resume_evidence")
                                else None
                            ),
                        )
                    )
                    continue
            merged.append(match)

        opinion.skill_matches = merged
        # 60% deterministic coverage, 40% model score.
        det = coverage_score(merged)
        opinion.raw_score = round(0.6 * det + 0.4 * opinion.raw_score, 1)
        return opinion

    def fallback(self, **kwargs: Any) -> DimensionOpinion:
        """Deterministic coverage score without the model."""
        jd: JobProfile = kwargs["jd"]
        resume: ResumeProfile = kwargs["resume"]
        matches = self.deterministic_matches(jd, resume)
        opinion = DimensionOpinion(
            dimension=self.dimension,
            raw_score=coverage_score(matches),
            confidence=0.4,
            reasoning=(
                "Deterministic skill coverage only (LLM transferability pass "
                "unavailable)."
            ),
            agent=self.name,
            degraded=True,
        )
        opinion.skill_matches = matches
        return opinion

    @staticmethod
    def deterministic_matches(
        jd: JobProfile, resume: ResumeProfile
    ) -> list[SkillMatch]:
        required = [c.name for c in jd.capabilities]
        candidate = list(resume.all_skill_keys())
        if not candidate and resume.source_text:
            # Empty skill list after parse failure → scan raw text.
            lowered = resume.source_text.lower()
            candidate = [r for r in required if r.lower() in lowered]
        return match_skills(required, candidate)


EVALUATOR_CLASSES: tuple[type[DimensionEvaluator], ...] = (
    RoleFitEvaluator,
    CapabilityEvaluator,
    SkillMatchEvaluator,
    ExperienceQualityEvaluator,
    ImpactEvaluator,
    DomainEvaluator,
)
