"""Resume Parser agent — redacted text to a structured profile."""

from __future__ import annotations

from typing import Any

from ranking_agent.agents.base import BaseAgent
from ranking_agent.models.resume import (
    ContactInfo,
    EducationEntry,
    ExperienceEntry,
    ProjectEntry,
    ResumeProfile,
)
from ranking_agent.tools.dates import total_years
from ranking_agent.tools.skills import normalize_all

_SYSTEM = """\
You are a Resume Parser in a resume-ranking system.

Contact details (email, phone, social links) have already been removed and
replaced with placeholders like [EMAIL_1]. Do NOT try to recover or output them —
another component handles contact data.

Return ONLY a JSON object:

{
  "name": string|null,
  "summary": string|null,
  "skills": [string],
  "experiences": [
    {"title": string|null, "company": string|null, "location": string|null,
     "start_date": string|null, "end_date": string|null, "is_current": boolean,
     "description": string|null, "skills": [string], "quantifiers": [string]}
  ],
  "educations": [{"institution": string|null, "degree": string|null,
                  "field_of_study": string|null, "end_date": string|null}],
  "projects": [{"name": string|null, "description": string|null, "tech_stack": [string]}],
  "certifications": [string],
  "others": [string]
}

Rules:
1. Use ONLY text present in the resume. NEVER invent an employer, date, or skill.
2. Dates as "YYYY-MM", or "YYYY" when only a year is given. Set is_current=true
   for ongoing roles and leave end_date null.
3. "skills" at the top level is the candidate's global skills section only.
   A role's "skills" list is filled ONLY when that role explicitly lists
   technologies (a "Skills:"/"Technologies:" line). Never infer skills from prose
   and never copy the global list into a role.
4. "quantifiers" are numeric impact statements copied verbatim
   ("reduced latency by 40%", "team of 12").
5. Lose nothing: content that fits no field goes into "others" verbatim.
6. Order experiences most-recent first.
"""


class ResumeParserAgent(BaseAgent[ResumeProfile]):
    """Produces the `ResumeProfile` that all evaluators read."""

    name = "resume_parser"
    task = "resume_parser"

    def system_prompt(self) -> str:
        return _SYSTEM

    def user_prompt(self, *, resume_text: str, **_: Any) -> str:
        return f"[RESUME]\n{self.clip(resume_text, 28_000)}"

    def parse(
        self,
        payload: dict[str, Any],
        *,
        resume_text: str,
        contact: ContactInfo | None = None,
        **_: Any,
    ) -> ResumeProfile:
        experiences = [
            ExperienceEntry(
                title=self._opt(raw.get("title")),
                company=self._opt(raw.get("company")),
                location=self._opt(raw.get("location")),
                start_date=self._opt(raw.get("start_date")),
                end_date=self._opt(raw.get("end_date")),
                is_current=bool(raw.get("is_current")),
                description=self._opt(raw.get("description")),
                skills=normalize_all(self.as_str_list(raw.get("skills"), 40)),
                quantifiers=self.as_str_list(raw.get("quantifiers"), 20),
            )
            for raw in (payload.get("experiences") or [])
            if isinstance(raw, dict)
        ]

        educations = [
            EducationEntry(
                institution=self._opt(raw.get("institution")),
                degree=self._opt(raw.get("degree")),
                field_of_study=self._opt(raw.get("field_of_study")),
                end_date=self._opt(raw.get("end_date")),
            )
            for raw in (payload.get("educations") or [])
            if isinstance(raw, dict)
        ]

        projects = [
            ProjectEntry(
                name=self._opt(raw.get("name")),
                description=self._opt(raw.get("description")),
                tech_stack=normalize_all(self.as_str_list(raw.get("tech_stack"), 40)),
            )
            for raw in (payload.get("projects") or [])
            if isinstance(raw, dict)
        ]

        merged_contact = contact or ContactInfo()
        if not merged_contact.name and payload.get("name"):
            merged_contact = merged_contact.model_copy(
                update={"name": str(payload["name"]).strip() or None}
            )

        return ResumeProfile(
            contact=merged_contact,
            summary=self._opt(payload.get("summary")),
            skills=normalize_all(self.as_str_list(payload.get("skills"), 120)),
            experiences=experiences[:25],
            educations=educations[:10],
            projects=projects[:20],
            certifications=self.as_str_list(payload.get("certifications"), 30),
            others=self.as_str_list(payload.get("others"), 30),
            source_text=resume_text,
            total_years_experience=total_years(experiences),
        )

    def fallback(
        self,
        *,
        resume_text: str,
        contact: ContactInfo | None = None,
        **_: Any,
    ) -> ResumeProfile:
        """Raw-text-only profile when the parser is unavailable."""
        return ResumeProfile(
            contact=contact or ContactInfo(),
            source_text=resume_text,
            warnings=["resume_parser unavailable — evaluating against raw text only"],
        )

    @staticmethod
    def _opt(value: Any) -> str | None:
        text = str(value).strip() if value is not None else ""
        return text or None
