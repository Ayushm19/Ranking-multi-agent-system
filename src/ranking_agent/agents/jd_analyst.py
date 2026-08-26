"""JD Analyst agent — turns a raw job description into a structured profile."""

from __future__ import annotations

import re
from typing import Any

from ranking_agent.agents.base import BaseAgent
from ranking_agent.models.jd import Capability, JobProfile
from ranking_agent.tools.skills import normalize_skill

_SYSTEM = """\
You are a Job Description Analyst in a resume-ranking system.

Deconstruct the job description into structured requirements. Return ONLY a JSON
object with these keys:

{
  "role_title": string|null,
  "role_family": string|null,
  "seniority": "senior"|"mid"|"junior"|"fresher"|null,
  "required_years_min": number|null,
  "required_years_max": number|null,
  "domain": string|null,
  "industry": string|null,
  "responsibilities": [string],
  "capabilities": [
    {"name": string, "normalized": string,
     "importance": "must_have"|"nice_to_have"|"adjacent", "category": string|null}
  ],
  "scale_signals": [string]
}

Rules:
1. Use ONLY what the job description states. Never invent a requirement.
2. A capability is "must_have" only when the JD frames it as required
   ("must", "required", "strong experience in"). Preferred/bonus items are
   "nice_to_have". Peripheral or context-only mentions are "adjacent".
3. Split compound requirements ("Python and Django") into separate capabilities.
4. required_years_min is a number, not text: "5+ years" -> 5.
5. Do NOT extract requirements about age, gender, nationality, religion, marital
   status, or disability. If the JD contains them, omit them entirely.
6. Return null rather than guessing when the JD is silent on a field.
"""

_YEARS_RE = re.compile(
    r"(\d{1,2})\s*(?:\+|plus)?\s*(?:-|to|–)?\s*(\d{1,2})?\s*\+?\s*(?:years|yrs)",
    re.IGNORECASE,
)


class JDAnalystAgent(BaseAgent[JobProfile]):
    """Produces a `JobProfile` that every downstream evaluator scores against."""

    name = "jd_analyst"
    task = "jd_analyst"

    def system_prompt(self) -> str:
        return _SYSTEM

    def user_prompt(self, *, jd_text: str, **_: Any) -> str:
        return f"[JOB]\n{self.clip(jd_text, 24_000)}"

    def parse(self, payload: dict[str, Any], *, jd_text: str, **_: Any) -> JobProfile:
        caps: list[Capability] = []
        for raw in payload.get("capabilities") or []:
            if isinstance(raw, str):
                name = raw.strip()
                if name:
                    caps.append(
                        Capability(name=name, normalized=normalize_skill(name))
                    )
                continue
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            if not name:
                continue
            importance = str(raw.get("importance") or "must_have").lower()
            if importance not in ("must_have", "nice_to_have", "adjacent"):
                importance = "must_have"
            caps.append(
                Capability(
                    name=name,
                    normalized=normalize_skill(
                        str(raw.get("normalized") or name)
                    ),
                    importance=importance,  # type: ignore[arg-type]
                    category=(str(raw["category"]) if raw.get("category") else None),
                )
            )

        seniority = payload.get("seniority")
        if seniority not in ("senior", "mid", "junior", "fresher", None):
            seniority = None

        years_min = payload.get("required_years_min")
        if years_min is None:
            years_min = self._regex_years(jd_text)

        return JobProfile(
            raw_jd=jd_text,
            role_title=self._opt_str(payload.get("role_title")),
            role_family=self._opt_str(payload.get("role_family")),
            seniority=seniority,  # type: ignore[arg-type]
            required_years_min=(
                self.as_float(years_min) if years_min is not None else None
            ),
            required_years_max=(
                self.as_float(payload.get("required_years_max"))
                if payload.get("required_years_max") is not None
                else None
            ),
            domain=self._opt_str(payload.get("domain")),
            industry=self._opt_str(payload.get("industry")),
            responsibilities=self.as_str_list(payload.get("responsibilities"), 30),
            capabilities=caps[:40],
            scale_signals=self.as_str_list(payload.get("scale_signals"), 15),
        )

    def fallback(self, *, jd_text: str, **_: Any) -> JobProfile:
        """Deterministic JD profile so ranking can proceed without the model."""
        return JobProfile(
            raw_jd=jd_text,
            required_years_min=self._regex_years(jd_text),
            capabilities=[],
            warnings=["jd_analyst unavailable — using regex-only JD profile"],
        )

    @staticmethod
    def _opt_str(value: Any) -> str | None:
        text = str(value).strip() if value is not None else ""
        return text or None

    @staticmethod
    def _regex_years(jd_text: str) -> float | None:
        match = _YEARS_RE.search(jd_text or "")
        return float(match.group(1)) if match else None
