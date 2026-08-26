"""Resume contracts produced by the Resume Parser agent."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
    """Contact fields recovered by regex (not the LLM)."""

    name: str | None = None
    email: str | None = None
    phone: str | None = None
    linkedin: str | None = None
    github: str | None = None
    location: str | None = None


class ExperienceEntry(BaseModel):
    title: str | None = None
    company: str | None = None
    location: str | None = None
    start_date: str | None = None  # YYYY-MM or YYYY
    end_date: str | None = None
    is_current: bool = False
    description: str | None = None
    skills: list[str] = Field(default_factory=list)
    quantifiers: list[str] = Field(default_factory=list)


class EducationEntry(BaseModel):
    institution: str | None = None
    degree: str | None = None
    field_of_study: str | None = None
    end_date: str | None = None


class ProjectEntry(BaseModel):
    name: str | None = None
    description: str | None = None
    tech_stack: list[str] = Field(default_factory=list)


class ResumeProfile(BaseModel):
    """Structured resume plus source text for evidence checks."""

    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: str | None = None
    skills: list[str] = Field(default_factory=list)
    experiences: list[ExperienceEntry] = Field(default_factory=list)
    educations: list[EducationEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    others: list[str] = Field(default_factory=list)

    source_text: str = ""  # redacted text used for groundedness
    total_years_experience: float | None = None
    sanitized: bool = False
    warnings: list[str] = Field(default_factory=list)

    def all_skill_keys(self) -> set[str]:
        keys = {s.strip().lower() for s in self.skills if s.strip()}
        for exp in self.experiences:
            keys |= {s.strip().lower() for s in exp.skills if s.strip()}
        for proj in self.projects:
            keys |= {s.strip().lower() for s in proj.tech_stack if s.strip()}
        return {k for k in keys if k}

    def experience_text(self) -> str:
        parts: list[str] = []
        for exp in self.experiences:
            head = " | ".join(
                p for p in (exp.title, exp.company, exp.start_date, exp.end_date) if p
            )
            parts.append(head)
            if exp.description:
                parts.append(exp.description)
        return "\n".join(parts).strip()
