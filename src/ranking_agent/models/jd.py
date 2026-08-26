"""Job-description contracts produced by the JD Analyst agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Importance = Literal["must_have", "nice_to_have", "adjacent"]


class Capability(BaseModel):
    """One thing the role requires, normalised for matching."""

    name: str
    normalized: str = ""
    importance: Importance = "must_have"
    category: str | None = None
    evidence: str | None = None

    def key(self) -> str:
        return (self.normalized or self.name).strip().lower()


class JobProfile(BaseModel):
    """Structured JD (all fields optional)."""

    raw_jd: str = ""
    role_title: str | None = None
    role_family: str | None = None
    seniority: Literal["senior", "mid", "junior", "fresher"] | None = None
    required_years_min: float | None = None
    required_years_max: float | None = None
    domain: str | None = None
    industry: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    capabilities: list[Capability] = Field(default_factory=list)
    scale_signals: list[str] = Field(default_factory=list)
    sanitized: bool = False
    warnings: list[str] = Field(default_factory=list)

    def must_haves(self) -> list[Capability]:
        return [c for c in self.capabilities if c.importance == "must_have"]

    def nice_to_haves(self) -> list[Capability]:
        return [c for c in self.capabilities if c.importance == "nice_to_have"]

    def all_skill_keys(self) -> set[str]:
        return {c.key() for c in self.capabilities if c.key()}
